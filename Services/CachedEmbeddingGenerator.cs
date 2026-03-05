using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace NetworkMonitor.Search.Services;

/// <summary>
/// Persistent OpenSearch-backed cache for embedding vectors.
/// Cache key is deterministic from model identity + text + pad settings.
/// </summary>
public sealed class CachedEmbeddingGenerator : IEmbeddingGenerator
{
    private readonly IEmbeddingGenerator _inner;
    private readonly ILogger<CachedEmbeddingGenerator> _logger;
    private readonly string _modelIdentity;
    private readonly HttpClient _httpClient;
    private readonly string _indexName;
    private int _indexInitialized;
    private readonly ConcurrentDictionary<string, List<float>> _memoryCache = new(StringComparer.Ordinal);
    private readonly SemaphoreSlim _ioLock = new(1, 1);
    private long _memoryHitCount;
    private long _storeHitCount;
    private long _missCount;
    private long _storeWriteCount;
    private long _storeConflictCount;
    private long _storeReadErrorCount;

    public CachedEmbeddingGenerator(
        IEmbeddingGenerator inner,
        string openSearchUrl,
        string openSearchUser,
        string openSearchKey,
        string modelIdentity,
        ILogger<CachedEmbeddingGenerator> logger)
        : this(
            inner,
            BuildHttpClient(openSearchUrl, openSearchUser, openSearchKey),
            modelIdentity,
            logger)
    {
    }

    public CachedEmbeddingGenerator(
        IEmbeddingGenerator inner,
        HttpClient httpClient,
        string modelIdentity,
        ILogger<CachedEmbeddingGenerator> logger)
    {
        _inner = inner;
        _logger = logger;
        _modelIdentity = string.IsNullOrWhiteSpace(modelIdentity) ? "unknown_model" : modelIdentity;
        _indexName = "llm_embedding_cache";
        _httpClient = httpClient;
        _httpClient.Timeout = Timeout.InfiniteTimeSpan;
    }

    public async Task<List<float>> GenerateEmbeddingAsync(string text, int padToTokens, bool pad = false)
    {
        text ??= string.Empty;
        var key = BuildCacheKey(text, padToTokens, pad);

        if (_memoryCache.TryGetValue(key, out var inMemory) && inMemory.Count > 0)
        {
            var count = Interlocked.Increment(ref _memoryHitCount);
            _logger.LogInformation("EmbeddingCache hit=memory key={CacheKey} memory_hits={MemoryHits} store_hits={StoreHits} misses={Misses}", key, count, Volatile.Read(ref _storeHitCount), Volatile.Read(ref _missCount));
            return new List<float>(inMemory);
        }

        await EnsureCacheIndexExistsAsync();

        var fromStore = await TryGetFromStoreAsync(key);
        if (fromStore.Count > 0)
        {
            _memoryCache[key] = fromStore;
            var count = Interlocked.Increment(ref _storeHitCount);
            _logger.LogInformation("EmbeddingCache hit=store key={CacheKey} memory_hits={MemoryHits} store_hits={StoreHits} misses={Misses}", key, Volatile.Read(ref _memoryHitCount), count, Volatile.Read(ref _missCount));
            return new List<float>(fromStore);
        }

        var generated = await _inner.GenerateEmbeddingAsync(text, padToTokens, pad);
        if (generated == null || generated.Count == 0)
        {
            return generated ?? new List<float>();
        }
        var misses = Interlocked.Increment(ref _missCount);
        _logger.LogInformation("EmbeddingCache miss key={CacheKey} memory_hits={MemoryHits} store_hits={StoreHits} misses={Misses}", key, Volatile.Read(ref _memoryHitCount), Volatile.Read(ref _storeHitCount), misses);

        _memoryCache[key] = generated;

        try
        {
            await UpsertToStoreAsync(key, generated);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Embedding cache write failed for key {CacheKey}", key);
        }

        return new List<float>(generated);
    }

    private string BuildCacheKey(string text, int padToTokens, bool pad)
    {
        var normalized = NormalizeText(text);
        var padPart = pad ? $"pad:1:tokens:{padToTokens}" : "pad:0";
        var raw = $"{_modelIdentity}|{padPart}|{normalized}";
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(raw))).ToLowerInvariant();
    }

    private static string NormalizeText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        var trimmed = text.Trim();
        var sb = new StringBuilder(trimmed.Length);
        bool inWhitespace = false;

        foreach (var ch in trimmed)
        {
            if (char.IsWhiteSpace(ch))
            {
                if (!inWhitespace)
                {
                    sb.Append(' ');
                    inWhitespace = true;
                }
            }
            else
            {
                sb.Append(ch);
                inWhitespace = false;
            }
        }

        return sb.ToString();
    }

    private async Task EnsureCacheIndexExistsAsync()
    {
        if (Interlocked.CompareExchange(ref _indexInitialized, 1, 1) == 1)
        {
            return;
        }

        await _ioLock.WaitAsync();
        try
        {
            if (_indexInitialized == 1)
            {
                return;
            }

            var exists = await _httpClient.GetAsync($"/{_indexName}");
            if (exists.IsSuccessStatusCode)
            {
                _indexInitialized = 1;
                return;
            }

            var mapping = @"
{
  ""settings"": {
    ""index"": {
      ""number_of_shards"": 1,
      ""number_of_replicas"": 1
    }
  },
  ""mappings"": {
    ""properties"": {
      ""model_identity"": { ""type"": ""keyword"" },
      ""embedding"": { ""type"": ""float"" },
      ""updated_at"": { ""type"": ""date"" }
    }
  }
}";
            using var content = new StringContent(mapping, Encoding.UTF8, "application/json");
            var create = await _httpClient.PutAsync($"/{_indexName}", content);
            if (!create.IsSuccessStatusCode)
            {
                var body = await create.Content.ReadAsStringAsync();
                throw new InvalidOperationException($"Failed creating embedding cache index '{_indexName}': {(int)create.StatusCode} {body}");
            }

            _indexInitialized = 1;
        }
        finally
        {
            _ioLock.Release();
        }
    }

    private async Task<List<float>> TryGetFromStoreAsync(string key)
    {
        var resp = await _httpClient.GetAsync($"/{_indexName}/_doc/{key}");
        if (!resp.IsSuccessStatusCode)
        {
            if ((int)resp.StatusCode != 404)
            {
                Interlocked.Increment(ref _storeReadErrorCount);
                _logger.LogWarning("EmbeddingCache store read failed key={CacheKey} status={StatusCode}", key, (int)resp.StatusCode);
            }
            return new List<float>();
        }

        var payload = await resp.Content.ReadAsStringAsync();
        if (string.IsNullOrWhiteSpace(payload))
        {
            return new List<float>();
        }

        var root = JObject.Parse(payload);
        var source = root["_source"] as JObject;
        if (source == null)
        {
            return new List<float>();
        }

        var model = source["model_identity"]?.Value<string>() ?? string.Empty;
        if (!string.Equals(model, _modelIdentity, StringComparison.Ordinal))
        {
            return new List<float>();
        }

        var arr = source["embedding"] as JArray;
        if (arr == null || arr.Count == 0)
        {
            return new List<float>();
        }

        return arr.Values<float>().ToList();
    }

    private async Task UpsertToStoreAsync(string key, List<float> embedding)
    {
        var body = new
        {
            model_identity = _modelIdentity,
            embedding,
            updated_at = DateTime.UtcNow
        };
        var json = JsonConvert.SerializeObject(body);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        // Create-only so we never rewrite existing cache entries.
        var resp = await _httpClient.PutAsync($"/{_indexName}/_create/{key}", content);
        if ((int)resp.StatusCode == 409)
        {
            var conflicts = Interlocked.Increment(ref _storeConflictCount);
            _logger.LogInformation("EmbeddingCache store_conflict key={CacheKey} conflicts={Conflicts}", key, conflicts);
            return;
        }
        if (!resp.IsSuccessStatusCode)
        {
            var error = await resp.Content.ReadAsStringAsync();
            throw new InvalidOperationException($"Embedding cache upsert failed: {(int)resp.StatusCode} {error}");
        }
        var writes = Interlocked.Increment(ref _storeWriteCount);
        _logger.LogInformation("EmbeddingCache store_write key={CacheKey} writes={Writes}", key, writes);
    }

    private static HttpClient BuildHttpClient(string openSearchUrl, string openSearchUser, string openSearchKey)
    {
        var handler = new HttpClientHandler
        {
            ServerCertificateCustomValidationCallback = (_, _, _, _) => true
        };
        var client = new HttpClient(handler, disposeHandler: true)
        {
            BaseAddress = new Uri(openSearchUrl),
            Timeout = Timeout.InfiniteTimeSpan
        };
        var authBytes = Encoding.ASCII.GetBytes($"{openSearchUser}:{openSearchKey}");
        client.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Basic", Convert.ToBase64String(authBytes));
        return client;
    }
}
