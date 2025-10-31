using OpenSearch.Client;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Net.Http;
using System.Net.Http.Headers;
using OpenSearch.Net;
using NetworkMonitor.Objects;
using System.Threading;

namespace NetworkMonitor.Search.Services;


public class OpenSearchHelper
{
    private readonly OpenSearchClient _client;
    private IEmbeddingGenerator _embeddingGenerator;
    private OSModelParams _modelParams;
    private readonly IReadOnlyList<IIndexingStrategy> _strategies;
    private readonly HttpClient _httpClient;

    public Uri SearchUri => _modelParams.SearchUri;

    public OpenSearchHelper(OSModelParams modelParams,
                              IEmbeddingGenerator embeddingGenerator,
                              params IIndexingStrategy[] strategies)
    {

        _strategies = strategies;

        _modelParams = modelParams;
        _embeddingGenerator = embeddingGenerator;
        // Initialize OpenSearch client
        var settings = new ConnectionSettings(_modelParams.SearchUri)
            .DefaultIndex(_modelParams.DefaultIndex)
            .BasicAuthentication(_modelParams.User, _modelParams.Key)
            .ServerCertificateValidationCallback((o, certificate, chain, errors) => true);

        _client = new OpenSearchClient(settings);

        var handler = new HttpClientHandler
        {
            ServerCertificateCustomValidationCallback = (message, cert, chain, sslPolicyErrors) => true
        };

        _httpClient = new HttpClient(handler, disposeHandler: true)
        {
            BaseAddress = _modelParams.SearchUri,
            Timeout = Timeout.InfiniteTimeSpan
        };

        var authBytes = Encoding.ASCII.GetBytes($"{_modelParams.User}:{_modelParams.Key}");
        _httpClient.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Basic", Convert.ToBase64String(authBytes));
    }

    // Method to generate embeddings for a document (async)
    private async Task<List<float>> GenerateEmbeddingAsync(string text, int padToTokens)
    {
        return await _embeddingGenerator.GenerateEmbeddingAsync(text, padToTokens);
    }

    // Method to load documents or securitybooks from JSON and index in OpenSearch
    public async Task<ResultObj> IndexDocumentsAsync(IEnumerable<object> items,
                                                     int padToTokens,
                                                     bool incrementalUpdate = false)
    {
        var result = new ResultObj { Message = "IndexDocumentsAsync: " };
        bool failed = false;
        int created = 0;
        int updated = 0;
        int skipped = 0;
        string? lastIndexName = null;

        foreach (var item in items)
        {
            // Pick the first strategy that says it can handle this artefact
            var strategy = _strategies.FirstOrDefault(s => s.CanHandle(item));
            if (strategy is null)
            {
                result.Message += $"No strategy found for type {item.GetType().Name}. Skipping. ";
                failed = true;
                continue;
            }

            try
            {
                var id = strategy.ComputeId(item);
                var index = strategy.IndexName;
                var contentHash = strategy.ComputeContentHash(item);
                bool exists = false;
                string? existingHash = null;
                lastIndexName = index;

                if (incrementalUpdate)
                {
                    var getResponse = await _client.LowLevel.GetAsync<StringResponse>(index, id);

                    if (getResponse.HttpStatusCode == 200 && !string.IsNullOrEmpty(getResponse.Body))
                    {
                        exists = true;
                        try
                        {
                            var payload = JObject.Parse(getResponse.Body);
                            existingHash = payload["_source"]?["content_hash"]?.Value<string>();
                        }
                        catch (JsonException)
                        {
                            // Ignore parse failures; treat as hash mismatch to force update.
                        }
                    }
                    else if (getResponse.HttpStatusCode == 404)
                    {
                        exists = false;
                    }
                    else if (!getResponse.Success)
                    {
                        failed = true;
                        result.Message += $"Failed to check {index}/{id}: {getResponse.DebugInformation} ";
                        continue;
                    }
                }
                else
                {
                    var existsResponse = await _client.DocumentExistsAsync<object>(id, idx => idx.Index(index));
                    exists = existsResponse.Exists;
                    if (exists)
                    {
                        skipped++;
                        result.Message += $"{index}/{id} already exists. Skipping. ";
                        continue;
                    }
                }

                if (incrementalUpdate && exists &&
                    string.Equals(existingHash, contentHash, StringComparison.OrdinalIgnoreCase))
                {
                    skipped++;
                    result.Message += $"{index}/{id} up-to-date. ";
                    continue;
                }

                await strategy.EnsureEmbeddingsAsync(item, _embeddingGenerator, padToTokens);

                var body = strategy.BuildIndexDocument(item);
                var docJson = JObject.FromObject(body ?? new { });
                docJson["content_hash"] = contentHash;
                docJson["updated_at"] = DateTime.UtcNow;

                var resp = await _client.LowLevel.IndexAsync<StringResponse>(
                    index,
                    id,
                    PostData.String(docJson.ToString(Formatting.None)));

                if (!resp.Success)
                {
                    failed = true;
                    result.Message += $"Failed to index {index}/{id}: {resp.DebugInformation} ";
                }
                else
                {
                    if (exists)
                    {
                        updated++;
                        result.Message += $"Updated {index}/{id}. ";
                    }
                    else
                    {
                        created++;
                        result.Message += $"Indexed {index}/{id}. ";
                    }
                }
            }
            catch (Exception ex)
            {
                failed = true;
                result.Message += $"Error for {item.GetType().Name}: {ex.Message} ";
            }
        }

        if (incrementalUpdate)
        {
            result.Message += $"Summary => Created:{created}, Updated:{updated}, Skipped:{skipped}. ";
            Console.WriteLine($"Incremental summary for '{lastIndexName ?? "index"}': Created={created}, Updated={updated}, Skipped={skipped}.");
        }
        else
        {
            Console.WriteLine($"Indexing complete for '{lastIndexName ?? "index"}': Created={created}, Updated={updated}, Skipped={skipped}.");
        }

        result.Success = !failed;
        return result;
    }

    private async Task<bool> TryPopulateExistingHashesAsync(List<IndexDocInfo> docInfos)
    {
        const int batchSize = 256;

        for (int i = 0; i < docInfos.Count; i += batchSize)
        {
            var batch = docInfos.Skip(i).Take(batchSize).ToList();
            var payload = new
            {
                docs = batch.Select(info => new
                {
                    _index = info.Index,
                    _id = info.Id,
                    _source = new[] { "content_hash" }
                })
            };

            var response = await _client.LowLevel.DoRequestAsync<StringResponse>(
                OpenSearch.Net.HttpMethod.POST,
                "_mget",
                cancellationToken: CancellationToken.None,
                data: PostData.String(JsonConvert.SerializeObject(payload)));

            if (!response.Success || string.IsNullOrWhiteSpace(response.Body))
                return false;

            try
            {
                var root = JObject.Parse(response.Body);
                if (root["docs"] is not JArray docsArray)
                    return false;

                for (int j = 0; j < batch.Count && j < docsArray.Count; j++)
                {
                    if (docsArray[j] is not JObject doc)
                        continue;

                    var info = batch[j];
                    bool found = doc.Value<bool?>("found") ?? false;
                    info.Exists = found;
                    if (found)
                    {
                        info.ExistingHash = doc["_source"]?["content_hash"]?.Value<string>();
                    }
                }
            }
            catch (JsonException)
            {
                return false;
            }
        }

        return true;
    }

    private async Task<bool> TryPopulateSingleDocumentStateAsync(IndexDocInfo info)
    {
        var response = await _client.LowLevel.GetAsync<StringResponse>(info.Index, info.Id);

        if (response.HttpStatusCode == 404)
        {
            info.Exists = false;
            info.ExistingHash = null;
            return true;
        }

        if (!response.Success || string.IsNullOrWhiteSpace(response.Body))
            return false;

        try
        {
            var payload = JObject.Parse(response.Body);
            info.Exists = true;
            info.ExistingHash = payload["_source"]?["content_hash"]?.Value<string>();
            return true;
        }
        catch (JsonException)
        {
            info.Exists = true;
            info.ExistingHash = null;
            return true;
        }
    }

    private sealed class IndexDocInfo
    {
        public IndexDocInfo(object item, IIndexingStrategy strategy, string index, string id, string contentHash)
        {
            Item = item;
            Strategy = strategy;
            Index = index;
            Id = id;
            ContentHash = contentHash;
        }

        public object Item { get; }
        public IIndexingStrategy Strategy { get; }
        public string Index { get; }
        public string Id { get; }
        public string ContentHash { get; }
        public bool Exists { get; set; }
        public string? ExistingHash { get; set; }
    }
    // in OpenSearchHelper
    private IIndexingStrategy StrategyForIndex(string index) =>
         _strategies.FirstOrDefault(s => s.CanHandle(index))
        ?? throw new InvalidOperationException($"No strategy for index '{index}'");

    // Method to search for similar documents using precomputed embeddings
    // Accepts an optional vectorFieldName parameter to support different field names per index/object.
    public async Task<SearchResponseObj> SearchDocumentsAsync(
        string queryText,
        string indexName,
        int padToTokens,
        VectorSearchMode mode = VectorSearchMode.content,
        TimeSpan? requestTimeout = null,
        CancellationToken cancellationToken = default)
    {

        var queryEmbedding = await GenerateEmbeddingAsync(queryText, padToTokens);
        var searchResponse = new SearchResponseObj();

        if (queryEmbedding.Count == 0)
        {
            throw new Exception("Failed to generate query embedding.");
        }

        var strategy = StrategyForIndex(indexName);
        string vectorFieldName = strategy.GetVectorField(mode);

        // Construct the k-NN search request body with dynamic field name
        var requestBody = new
        {
            size = 3,
            query = new
            {
                knn = new Dictionary<string, object>
                {
                    [vectorFieldName] = new
                    {
                        vector = queryEmbedding,
                        k = 3
                    }
                }
            }
        };

        // Serialize the request body to JSON using Newtonsoft.Json
        var jsonContent = JsonConvert.SerializeObject(requestBody);
        using var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

        // Send the POST request to the specified index
        var response = await PostWithTimeoutAsync($"/{indexName}/_search", content, requestTimeout, cancellationToken);

        // Process the response
        if (response.IsSuccessStatusCode)
        {
            var responseBody = await response.Content.ReadAsStringAsync();
            // Console.WriteLine("Search Results:");
            // Console.WriteLine(responseBody);

            // Deserialize the JSON response into the SearchResponse object
            searchResponse = JsonConvert.DeserializeObject<SearchResponseObj>(responseBody);

        }
        else
        {
            throw new Exception($"Search failed: {response.ReasonPhrase}");
        }
        if (searchResponse == null) searchResponse = new SearchResponseObj();
        return searchResponse;
    }

    public async Task<SearchResponseObj> MultiFieldKnnSearchAsync(
        string queryText,
        int kPerField,
        Dictionary<string, float>? fieldWeights,
        string indexName,
        int padToTokens,
        TimeSpan? requestTimeout = null,
        CancellationToken cancellationToken = default)
    {

        var queryEmbedding = await GenerateEmbeddingAsync(queryText, padToTokens);

        if (queryEmbedding.Count == 0)
            throw new Exception("Failed to generate query embedding.");

        var strategy = StrategyForIndex(indexName);

        // caller‑supplied weights override defaults
        fieldWeights = fieldWeights?.Count > 0
            ? strategy.GetDefaultFieldWeights()
                      .Concat(fieldWeights)
                      .GroupBy(kv => kv.Key)
                      .ToDictionary(g => g.Key, g => g.Last().Value)
            : new Dictionary<string, float>(strategy.GetDefaultFieldWeights());


        var shouldClauses = new List<object>();
        foreach (var (field, weight) in fieldWeights)
        {
            shouldClauses.Add(new
            {
                function_score = new
                {
                    knn = new Dictionary<string, object>
                    {
                        [field] = new { vector = queryEmbedding, k = kPerField }
                    },
                    weight
                }
            });
        }

        var requestBody = new
        {
            size = kPerField,
            query = new
            {
                @bool = new { should = shouldClauses }
            }
        };

        var json = JsonConvert.SerializeObject(requestBody);
        using var requestContent = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await PostWithTimeoutAsync($"/{indexName}/_search", requestContent, requestTimeout, cancellationToken);

        if (!response.IsSuccessStatusCode)
            throw new Exception($"Search failed: {response.ReasonPhrase}");

        var responseBody = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<SearchResponseObj>(responseBody) ??
               new SearchResponseObj();
    }

    private async Task<HttpResponseMessage> PostWithTimeoutAsync(
        string requestUri,
        HttpContent content,
        TimeSpan? requestTimeout,
        CancellationToken cancellationToken)
    {
        if (requestTimeout.HasValue)
        {
            var timeout = requestTimeout.Value;
            if (timeout != Timeout.InfiniteTimeSpan && timeout > TimeSpan.Zero)
            {
                using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                cts.CancelAfter(timeout);
                return await _httpClient.PostAsync(requestUri, content, cts.Token);
            }
        }

        if (cancellationToken.CanBeCanceled)
            return await _httpClient.PostAsync(requestUri, content, cancellationToken);

        return await _httpClient.PostAsync(requestUri, content);
    }

    public async Task<ResultObj> EnsureIndexExistsAsync(
     string indexName = "", bool recreateIndex = false)
    {
        var result = new ResultObj { Message = "EnsureIndexExistsAsync: " };
        try
        {
            if (string.IsNullOrWhiteSpace(indexName))
                indexName = _modelParams.DefaultIndex;

            // 1. Delete if requested
            if (recreateIndex)
            {
                var del = await DeleteIndexAsync(indexName);
                result.Message += del.Message;
                if (!del.Success) return del;
            }

            // 2. Only create if missing
            var exists = await _client.Indices.ExistsAsync(indexName);
            if (exists.Exists)
            {
                result.Success = true;
                result.Message += recreateIndex
                    ? "index already exists after recreation check"
                    : "index already exists (retained)";
                return result;
            }

            // 3. Ask strategy for mapping
            var strategy = StrategyForIndex(indexName);
            var mapping = strategy.GetIndexMapping(_modelParams.EmbeddingModelVecDim);

            var create = await _client.LowLevel.Indices.CreateAsync<StringResponse>(
                             indexName, PostData.String(mapping));

            result.Success = create.Success;
            result.Message += create.Success
                ? "index created"
                : $"Failed: {create.DebugInformation}";
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.Message += ex.Message;
        }
        return result;
    }

    public async Task<ResultObj> DeleteIndexAsync(string indexName = "")
    {
        var result = new ResultObj() { Message = " DeleteIndexAsync : " };
        try
        {

            if (indexName == "") indexName = _modelParams.DefaultIndex;
            var existsResponse = await _client.Indices.ExistsAsync(indexName);
            if (existsResponse.Exists)
            {
                var deleteResponse = await _client.Indices.DeleteAsync(indexName);
                if (deleteResponse.IsValid)
                {
                    result.Message += $"Index '{indexName}' deleted successfully.";
                    result.Success = true;
                }
                else
                {
                    result.Message += $"Failed to delete index '{indexName}': {deleteResponse.DebugInformation}";
                    result.Success = false;
                    return result;
                }
            }
            else
            {
                result.Message += $"Index '{indexName}' does not exist. No action taken.";
                result.Success = true;
            }
        }
        catch (Exception e)
        {
            result.Success = false;
            result.Message += e.Message;
        }
        return result;
    }

    // Method to compute a SHA256 hash for unique document IDs
    private static string ComputeSha256Hash(string rawData)
    {
        using (SHA256 sha256Hash = SHA256.Create())
        {
            byte[] bytes = sha256Hash.ComputeHash(Encoding.UTF8.GetBytes(rawData));
            StringBuilder builder = new StringBuilder();
            foreach (byte b in bytes)
            {
                builder.Append(b.ToString("x2"));
            }
            return builder.ToString();
        }
    }
}
