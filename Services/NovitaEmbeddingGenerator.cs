using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace NetworkMonitor.Search.Services
{
    public class NovitaEmbeddingGenerator : IEmbeddingGenerator
    {
        private readonly string _apiKey;
        private readonly string _model;
        private readonly string _apiUrl;

        // Static semaphore and last call time to coordinate delay across all instances
        private static readonly SemaphoreSlim _callSemaphore = new SemaphoreSlim(1, 1);
        private static DateTime _lastCallTime = DateTime.MinValue;
        private static readonly TimeSpan _minDelay = TimeSpan.FromSeconds(10);

        public NovitaEmbeddingGenerator(string apiKey, string model = "baai/bge-m3", string apiUrl = "https://api.novita.ai/v3/openai/embeddings")
        {
            _apiKey = apiKey;
            _model = model;
            _apiUrl = apiUrl;
        }

        public async Task<List<float>> GenerateEmbeddingAsync(string text, int padToTokens, bool pad = false)
        {
            await _callSemaphore.WaitAsync();
            try
            {
                var now = DateTime.UtcNow;
                var timeSinceLast = now - _lastCallTime;
                if (timeSinceLast < _minDelay)
                {
                    var delay = _minDelay - timeSinceLast;
                    await Task.Delay(delay);
                }
                _lastCallTime = DateTime.UtcNow;
            }
            finally
            {
                _callSemaphore.Release();
            }

            using var client = new HttpClient();
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
            client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

            var requestBody = new
            {
                model = _model,
                input = text, // could be string or string[] for batching
                encoding_format = "float"
            };

            var json = JsonConvert.SerializeObject(requestBody);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await client.PostAsync(_apiUrl, content);
            if (!response.IsSuccessStatusCode)
            {
                throw new Exception($"Novita API request failed: {response.StatusCode} {await response.Content.ReadAsStringAsync()}");
            }

            var responseBody = await response.Content.ReadAsStringAsync();
            var result = JsonConvert.DeserializeObject<OpenAIEmbeddingResponse>(responseBody);

            if (result?.data == null || result.data.Count == 0)
                throw new Exception("No embedding returned from Novita API.");

            // Return the first embedding (for single input)
            return result.data[0].embedding;
        }

        // OpenAI-compatible response
        private class OpenAIEmbeddingResponse
        {
            public string @object { get; set; }
            public List<OpenAIEmbeddingData> data { get; set; }
            public string model { get; set; }
            public OpenAIUsage usage { get; set; }
        }

        private class OpenAIEmbeddingData
        {
            public string @object { get; set; }
            public int index { get; set; }
            public List<float> embedding { get; set; }
        }

        private class OpenAIUsage
        {
            public int prompt_tokens { get; set; }
            public int total_tokens { get; set; }
        }
    }
}
