
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using NetworkMonitor.Objects;

namespace NetworkMonitor.Search.Services;
// ========== NovitaEmbeddingGenerator.cs ==========
public class NovitaEmbeddingGenerator : IEmbeddingGenerator
{
    private readonly MLParams _mlParams;
    private readonly AutoTokenizer _tokenizer;
    private readonly ApiRateLimiter _rateLimiter;
    private readonly NovitaApiClient _client;
    private readonly ILogger _logger;

    public NovitaEmbeddingGenerator(
        MLParams mlParams,
        ILogger<NovitaEmbeddingGenerator> logger,
        NovitaApiClient client)
    {
        _mlParams = mlParams;
        _tokenizer = new AutoTokenizer(mlParams.EmbeddingModelDir, mlParams.MaxTokenLengthCap);
        _logger = logger;
        _rateLimiter = new ApiRateLimiter();
        _client = client;
    }

    public async Task<List<float>> GenerateEmbeddingAsync(string text, int padToTokens, bool pad = false)
    {
        int maxCap = padToTokens;
        string Truncate(string input, int cap)
        {
            var ids = _tokenizer.TokenizeNoPad(input).InputIds;
            return _tokenizer.Decode(ids.Take(cap).ToList());
        }

        for (int attempt = 0; attempt < 10; attempt++)
        {
            await _rateLimiter.WaitAsync();

            var result = await _client.GetEmbeddingAsync(
                _mlParams.LlmHFKey,
                _mlParams.EmbeddingApiModel,
                _mlParams.EmbeddingApiUrl,
                Truncate(text, maxCap)
            );
            if (!string.IsNullOrEmpty(result.error))
            {
                if (result.error.Contains("maximum context length", StringComparison.OrdinalIgnoreCase))
                {
                    maxCap = Math.Max(500, maxCap - 500);
                    _logger.LogDebug("Truncated input to {TokenCount} tokens", maxCap);
                    continue;
                }
                _rateLimiter.NotifyFailure(result.rateLimited);
                return new List<float>();
            }

            _rateLimiter.NotifySuccess();
            return result.embedding;
        }

        _logger.LogError("All retry attempts failed for embedding generation");
        return new List<float>();
    }
}

// ========== IEmbeddingProvider.cs ==========
public interface IEmbeddingProvider
{
    Task<List<float>> GetEmbeddingAsync(string inputText, int tokenLimit);
}

// ========== ApiRateLimiter.cs ==========
public class ApiRateLimiter
{
    private readonly SemaphoreSlim _semaphore = new SemaphoreSlim(1, 1);
    private TimeSpan _delay = TimeSpan.FromMilliseconds(1000);
    private readonly TimeSpan _minDelay = TimeSpan.FromMilliseconds(1000);
    private readonly TimeSpan _maxDelay = TimeSpan.FromSeconds(120);
    private int _successStreak = 0;
    private DateTime _lastCall = DateTime.MinValue;
    private const int StreakToDecrease = 3;

    public async Task WaitAsync()
    {
        await _semaphore.WaitAsync();
        try
        {
            var sinceLast = DateTime.UtcNow - _lastCall;
            if (sinceLast < _delay)
                await Task.Delay(_delay - sinceLast);
            _lastCall = DateTime.UtcNow;
        }
        finally { _semaphore.Release(); }
    }

    public void NotifySuccess()
    {
        _successStreak++;
        if (_successStreak >= StreakToDecrease)
        {
            _delay = TimeSpan.FromMilliseconds(Math.Max(_minDelay.TotalMilliseconds, _delay.TotalMilliseconds * 0.8));
            _successStreak = 0;
        }
    }

    public void NotifyFailure(bool rateLimited)
    {
        _successStreak = 0;
        if (rateLimited)
        {
            _delay = TimeSpan.FromMilliseconds(Math.Min(_maxDelay.TotalMilliseconds, _delay.TotalMilliseconds * 2.0 + 1000));
        }
    }
}

// ========== NovitaApiClient.cs ==========
public class NovitaApiClient
{
    private readonly HttpClient _client;
    private readonly ILogger<NovitaApiClient> _logger;

    public NovitaApiClient(HttpClient client, ILogger<NovitaApiClient> logger)
    {
        _client = client;
        _logger = logger;
    }

    public async Task<(List<float> embedding, bool rateLimited, string error)> GetEmbeddingAsync(string apiKey, string model, string apiUrl, string text)
    {
        _client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

        var req = new
        {
            model,
            input = text,
            encoding_format = "float"
        };

        var content = new StringContent(JsonConvert.SerializeObject(req), Encoding.UTF8, "application/json");

        var resp = await _client.PostAsync(apiUrl, content);
        var body = await resp.Content.ReadAsStringAsync();

        if (!resp.IsSuccessStatusCode)
        {
            _logger.LogWarning("Novita error {StatusCode}: {Body}", resp.StatusCode, body);
            bool is429 = (int)resp.StatusCode == 429;
            return (new List<float>(), is429, body);
        }

        var parsed = JsonConvert.DeserializeObject<OpenAIEmbeddingResponse>(body);
        return (parsed?.data?.FirstOrDefault()?.embedding ?? new List<float>(), false, null);
    }

    private class OpenAIEmbeddingResponse
    {
        public List<OpenAIEmbeddingData> data { get; set; }
    }

    private class OpenAIEmbeddingData
    {
        public List<float> embedding { get; set; }
    }
}
