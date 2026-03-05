using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using Newtonsoft.Json.Linq;
using Xunit;

namespace NetworkMonitor.Search.Services.Tests;

public class CachedEmbeddingGeneratorTests
{
    [Fact]
    public async Task GenerateEmbeddingAsync_MissThenMemoryHit()
    {
        var handler = new FakeOpenSearchHandler();
        var httpClient = new HttpClient(handler) { BaseAddress = new Uri("https://opensearch:9200") };
        var inner = new CountingEmbeddingGenerator(new List<float> { 1.1f, 2.2f, 3.3f });
        var cache = new CachedEmbeddingGenerator(inner, httpClient, "modelA", NullLogger<CachedEmbeddingGenerator>.Instance);

        var first = await cache.GenerateEmbeddingAsync("hello world", 128);
        var second = await cache.GenerateEmbeddingAsync("hello world", 128);

        Assert.Equal(1, inner.CallCount);
        Assert.Equal(first, second);
        Assert.Equal(1, handler.CreateCallCount);
    }

    [Fact]
    public async Task GenerateEmbeddingAsync_StoreHitAcrossInstance_DoesNotCallInner()
    {
        var handler = new FakeOpenSearchHandler();
        var client1 = new HttpClient(handler) { BaseAddress = new Uri("https://opensearch:9200") };
        var client2 = new HttpClient(handler) { BaseAddress = new Uri("https://opensearch:9200") };

        var seedInner = new CountingEmbeddingGenerator(new List<float> { 9f, 8f, 7f });
        var seedCache = new CachedEmbeddingGenerator(seedInner, client1, "modelA", NullLogger<CachedEmbeddingGenerator>.Instance);
        var seeded = await seedCache.GenerateEmbeddingAsync("repeat phrase", 128);

        var secondInner = new CountingEmbeddingGenerator(new List<float> { 0.5f });
        var secondCache = new CachedEmbeddingGenerator(secondInner, client2, "modelA", NullLogger<CachedEmbeddingGenerator>.Instance);
        var loaded = await secondCache.GenerateEmbeddingAsync("repeat phrase", 128);

        Assert.Equal(1, seedInner.CallCount);
        Assert.Equal(0, secondInner.CallCount);
        Assert.Equal(seeded, loaded);
        Assert.True(handler.DocGetHitCount >= 1);
    }

    [Fact]
    public async Task GenerateEmbeddingAsync_CreateConflict_DoesNotThrowAndReturnsEmbedding()
    {
        var handler = new FakeOpenSearchHandler { ForceCreateConflict = true };
        var httpClient = new HttpClient(handler) { BaseAddress = new Uri("https://opensearch:9200") };
        var inner = new CountingEmbeddingGenerator(new List<float> { 4f, 5f });
        var cache = new CachedEmbeddingGenerator(inner, httpClient, "modelA", NullLogger<CachedEmbeddingGenerator>.Instance);

        var value = await cache.GenerateEmbeddingAsync("conflict path", 128);

        Assert.Equal(new List<float> { 4f, 5f }, value);
        Assert.Equal(1, inner.CallCount);
        Assert.Equal(1, handler.CreateCallCount);
    }

    [Fact]
    public async Task GenerateEmbeddingAsync_ModelIdentityMismatch_TreatedAsMiss()
    {
        var handler = new FakeOpenSearchHandler();
        var client1 = new HttpClient(handler) { BaseAddress = new Uri("https://opensearch:9200") };
        var client2 = new HttpClient(handler) { BaseAddress = new Uri("https://opensearch:9200") };

        var aInner = new CountingEmbeddingGenerator(new List<float> { 1f, 2f });
        var cacheA = new CachedEmbeddingGenerator(aInner, client1, "modelA", NullLogger<CachedEmbeddingGenerator>.Instance);
        await cacheA.GenerateEmbeddingAsync("same text", 64);

        var bInner = new CountingEmbeddingGenerator(new List<float> { 3f, 4f });
        var cacheB = new CachedEmbeddingGenerator(bInner, client2, "modelB", NullLogger<CachedEmbeddingGenerator>.Instance);
        var bValue = await cacheB.GenerateEmbeddingAsync("same text", 64);

        Assert.Equal(1, bInner.CallCount);
        Assert.Equal(new List<float> { 3f, 4f }, bValue);
    }

    private sealed class CountingEmbeddingGenerator : IEmbeddingGenerator
    {
        private readonly List<float> _vector;

        public int CallCount { get; private set; }

        public CountingEmbeddingGenerator(List<float> vector)
        {
            _vector = vector;
        }

        public Task<List<float>> GenerateEmbeddingAsync(string text, int padToTokens, bool pad = false)
        {
            CallCount++;
            return Task.FromResult(_vector.ToList());
        }
    }

    private sealed class FakeOpenSearchHandler : HttpMessageHandler
    {
        private readonly ConcurrentDictionary<string, JObject> _docs = new(StringComparer.Ordinal);
        private bool _indexExists;

        public bool ForceCreateConflict { get; set; }
        public int CreateCallCount { get; private set; }
        public int DocGetHitCount { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = request.RequestUri?.AbsolutePath ?? string.Empty;
            var method = request.Method.Method.ToUpperInvariant();

            if (path.Equals("/llm_embedding_cache", StringComparison.Ordinal))
            {
                if (method == HttpMethod.Get.Method)
                {
                    return _indexExists
                        ? Json(HttpStatusCode.OK, @"{""llm_embedding_cache"":{}}")
                        : Json(HttpStatusCode.NotFound, @"{""error"":""index_not_found_exception""}");
                }

                if (method == HttpMethod.Put.Method)
                {
                    _indexExists = true;
                    return Json(HttpStatusCode.OK, @"{""acknowledged"":true}");
                }
            }

            if (path.StartsWith("/llm_embedding_cache/_doc/", StringComparison.Ordinal))
            {
                var id = path["/llm_embedding_cache/_doc/".Length..];
                if (_docs.TryGetValue(id, out var source))
                {
                    DocGetHitCount++;
                    var payload = new JObject
                    {
                        ["_source"] = source
                    };
                    return Json(HttpStatusCode.OK, payload.ToString());
                }

                return Json(HttpStatusCode.NotFound, @"{""found"":false}");
            }

            if (path.StartsWith("/llm_embedding_cache/_create/", StringComparison.Ordinal) &&
                method == HttpMethod.Put.Method)
            {
                CreateCallCount++;
                var id = path["/llm_embedding_cache/_create/".Length..];
                if (ForceCreateConflict || _docs.ContainsKey(id))
                {
                    return Json(HttpStatusCode.Conflict, @"{""result"":""conflict""}");
                }

                var body = await request.Content!.ReadAsStringAsync(cancellationToken);
                _docs[id] = JObject.Parse(body);
                return Json(HttpStatusCode.Created, @"{""result"":""created""}");
            }

            return Json(HttpStatusCode.NotFound, @"{""error"":""unhandled_route""}");
        }

        private static HttpResponseMessage Json(HttpStatusCode code, string json)
        {
            return new HttpResponseMessage(code)
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json")
            };
        }
    }
}
