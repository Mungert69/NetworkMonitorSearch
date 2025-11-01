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
using Microsoft.Extensions.Logging;

namespace NetworkMonitor.Search.Services;


public class OpenSearchHelper
{
    private readonly OpenSearchClient _client;
    private IEmbeddingGenerator _embeddingGenerator;
    private OSModelParams _modelParams;
    private readonly IReadOnlyList<IIndexingStrategy> _strategies;
    private readonly ILogger<OpenSearchHelper> _logger;
    private readonly HttpClient _httpClient;

    public Uri SearchUri => _modelParams.SearchUri;

    public OpenSearchHelper(OSModelParams modelParams,
                              IEmbeddingGenerator embeddingGenerator,
                              ILogger<OpenSearchHelper> logger,
                              params IIndexingStrategy[] strategies)
    {

        _strategies = strategies;

        _modelParams = modelParams;
        _embeddingGenerator = embeddingGenerator;
        _logger = logger;
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

        var itemList = items?.ToList() ?? new List<object>();
        if (itemList.Count == 0)
        {
            result.Success = true;
            result.Message += "No items to index. ";
            _logger.LogTrace("No items supplied for indexing; exiting early.");
            return result;
        }

        var docInfos = new List<IndexDocInfo>(itemList.Count);
        int totalDocuments = 0;

        foreach (var item in itemList)
        {
            var strategy = _strategies.FirstOrDefault(s => s.CanHandle(item));
            if (strategy is null)
            {
                result.Message += $"No strategy found for type {item.GetType().Name}. Skipping. ";
                failed = true;
                _logger.LogWarning("Skipping artefact of type {Type} because no strategy can handle it.", item.GetType().FullName);
                continue;
            }

            try
            {
                var info = new IndexDocInfo(
                    item,
                    strategy,
                    strategy.IndexName,
                    strategy.ComputeId(item),
                    strategy.ComputeContentHash(item));
                docInfos.Add(info);
                _logger.LogTrace("Prepared {Index}/{Id} with content hash {Hash}.", info.Index, info.Id, info.ContentHash);
                totalDocuments++;
            }
            catch (Exception ex)
            {
                failed = true;
                result.Message += $"Failed preparing {item.GetType().Name}: {ex.Message} ";
                _logger.LogError(ex, "Failed preparing artefact {Type} for indexing.", item.GetType().FullName);
            }
        }

        if (docInfos.Count == 0)
        {
            result.Success = !failed;
            _logger.LogTrace("No valid artefacts remained after preparation; returning.");
            return result;
        }

        bool hashesPrefetched = false;
        if (incrementalUpdate)
        {
            _logger.LogTrace("Attempting batched hash prefetch for {Count} documents.", docInfos.Count);
            hashesPrefetched = await TryPopulateExistingHashesAsync(docInfos);
            if (!hashesPrefetched)
            {
                result.Message += "Warning: failed to prefetch existing hashes; falling back to per-document checks. ";
                _logger.LogWarning("Hash prefetch via _mget failed; will fall back to per-document GET checks.");
            }
        }

        foreach (var info in docInfos)
        {
            lastIndexName = info.Index;
            info.EmbeddingsReused = false;
            _logger.LogTrace("Processing {Index}/{Id}: incremental={Incremental} initialExists={Exists} hash={Hash}",
                info.Index, info.Id, incrementalUpdate, info.Exists, info.ContentHash);

            try
            {
                if (incrementalUpdate)
                {
                    if (!hashesPrefetched)
                    {
                        var populated = await TryPopulateSingleDocumentStateAsync(info);
                        if (!populated)
                        {
                            failed = true;
                            result.Message += $"Failed to check {info.Index}/{info.Id}. ";
                            _logger.LogWarning("Failed to retrieve state for {Index}/{Id}; skipping.", info.Index, info.Id);
                            continue;
                        }
                    }

                    if (info.Exists && info.ExistingHash != null &&
                        string.Equals(info.ExistingHash, info.ContentHash, StringComparison.OrdinalIgnoreCase))
                    {
                        skipped++;
                        result.Message += $"{info.Index}/{info.Id} up-to-date. ";
                        _logger.LogTrace("Skipping {Index}/{Id}; stored hash {ExistingHash} matches current hash {Hash}.",
                            info.Index, info.Id, info.ExistingHash, info.ContentHash);
                        continue;
                    }

                    if (info.Exists && info.ExistingHash != null)
                    {
                        _logger.LogTrace("Hash mismatch for {Index}/{Id}; stored {ExistingHash}, incoming {Hash}.",
                            info.Index, info.Id, info.ExistingHash, info.ContentHash);
                    }
                    else if (!info.Exists)
                    {
                        _logger.LogTrace("Document {Index}/{Id} not currently indexed; will create.", info.Index, info.Id);
                    }
                    else if (info.Exists)
                    {
                        _logger.LogTrace("Document {Index}/{Id} found without content hash; will backfill.", info.Index, info.Id);
                    }

                    if (info.Exists && string.IsNullOrEmpty(info.ExistingHash))
                    {
                        var source = info.ExistingSource ?? await FetchSourceAsync(info);
                        if (source != null && info.Strategy.TryHydrateFromDocument(info.Item, source))
                        {
                            info.EmbeddingsReused = true;
                            _logger.LogTrace("Reused stored embeddings for {Index}/{Id} based on persisted document.", info.Index, info.Id);
                        }
                        else
                        {
                            _logger.LogTrace("Stored document for {Index}/{Id} missing or content mismatch; embeddings will be regenerated.", info.Index, info.Id);
                        }
                    }
                }
                else
                {
                    if (!info.Exists)
                    {
                        var existsResponse = await _client.DocumentExistsAsync<object>(info.Id, idx => idx.Index(info.Index));
                        info.Exists = existsResponse.Exists;
                        _logger.LogTrace("DocumentExists check for {Index}/{Id}: exists={Exists}", info.Index, info.Id, info.Exists);
                    }

                    if (info.Exists)
                    {
                        skipped++;
                        result.Message += $"{info.Index}/{info.Id} already exists. Skipping. ";
                        _logger.LogTrace("Skipping {Index}/{Id}; document already exists and recreateIndex not requested.", info.Index, info.Id);
                        continue;
                    }
                }

                await info.Strategy.EnsureEmbeddingsAsync(info.Item, _embeddingGenerator, padToTokens);
                if (info.EmbeddingsReused)
                {
                    _logger.LogTrace("Confirmed embeddings reused for {Index}/{Id}; EnsureEmbeddingsAsync completed without regeneration.", info.Index, info.Id);
                }

                var body = info.Strategy.BuildIndexDocument(info.Item);
                var docJson = JObject.FromObject(body ?? new { });
                docJson["content_hash"] = info.ContentHash;
                docJson["updated_at"] = DateTime.UtcNow;

                var resp = await _client.LowLevel.IndexAsync<StringResponse>(
                    info.Index,
                    info.Id,
                    PostData.String(docJson.ToString(Formatting.None)));

                if (!resp.Success)
                {
                    failed = true;
                    result.Message += $"Failed to index {info.Index}/{info.Id}: {resp.DebugInformation} ";
                    _logger.LogError("Failed to index {Index}/{Id}: {Info}", info.Index, info.Id, resp.DebugInformation);
                }
                else
                {
                    if (info.Exists)
                    {
                        updated++;
                        result.Message += $"Updated {info.Index}/{info.Id}. ";
                        _logger.LogTrace("Updated {Index}/{Id}.", info.Index, info.Id);
                    }
                    else
                    {
                        created++;
                        result.Message += $"Indexed {info.Index}/{info.Id}. ";
                        _logger.LogTrace("Created {Index}/{Id}.", info.Index, info.Id);
                    }

                    info.Exists = true;
                    info.ExistingHash = info.ContentHash;
                }
            }
            catch (Exception ex)
            {
                failed = true;
                result.Message += $"Error for {info.Index}/{info.Id}: {ex.Message} ";
                _logger.LogError(ex, "Unexpected error while processing {Index}/{Id}.", info.Index, info.Id);
            }
        }

        result.Message += $"Summary => Total:{totalDocuments}, Created:{created}, Updated:{updated}, Skipped:{skipped}. ";

        if (incrementalUpdate)
        {
            _logger.LogInformation("Incremental summary for {Index}: Total={Total}, Created={Created}, Updated={Updated}, Skipped={Skipped}.",
                lastIndexName ?? "index", totalDocuments, created, updated, skipped);
        }
        else
        {
            _logger.LogInformation("Indexing summary for {Index}: Total={Total}, Created={Created}, Updated={Updated}, Skipped={Skipped}.",
                lastIndexName ?? "index", totalDocuments, created, updated, skipped);
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
            foreach (var info in batch)
            {
                info.ExistingSource = null;
            }

            var payload = new
            {
                docs = batch.Select(info => new
                {
                    _index = info.Index,
                    _id = info.Id,
                    _source = new[] { "content_hash" }
                })
            };

            _logger.LogTrace("Requesting batched hashes for {Count} documents.", batch.Count);

            var response = await _client.LowLevel.DoRequestAsync<StringResponse>(
                OpenSearch.Net.HttpMethod.POST,
                "_mget",
                cancellationToken: CancellationToken.None,
                data: PostData.String(JsonConvert.SerializeObject(payload)));

            if (!response.Success || string.IsNullOrWhiteSpace(response.Body))
            {
                _logger.LogWarning("_mget hash prefetch failed with status {StatusCode}.", response.HttpStatusCode);
                return false;
            }

            try
            {
                var root = JObject.Parse(response.Body);
                if (root["docs"] is not JArray docsArray)
                {
                    _logger.LogWarning("Unexpected _mget response structure; missing docs array.");
                    return false;
                }

                for (int j = 0; j < batch.Count && j < docsArray.Count; j++)
                {
                    if (docsArray[j] is not JObject doc)
                        continue;

                    var info = batch[j];
                    bool found = doc.Value<bool?>("found") ?? false;
                    info.Exists = found;
                    info.ExistingHash = found
                        ? doc["_source"]?["content_hash"]?.Value<string>()
                        : null;

                    _logger.LogTrace("Prefetch result for {Index}/{Id}: found={Found} hash={Hash}",
                        info.Index, info.Id, found, info.ExistingHash);
                }
            }
            catch (JsonException ex)
            {
                _logger.LogWarning(ex, "Failed to parse _mget response.");
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
            info.ExistingSource = null;
            _logger.LogTrace("Document {Index}/{Id} not found during individual state check.", info.Index, info.Id);
            return true;
        }

        if (!response.Success || string.IsNullOrWhiteSpace(response.Body))
        {
            _logger.LogWarning("Failed individual state check for {Index}/{Id}; status {Status}.", info.Index, info.Id, response.HttpStatusCode);
            return false;
        }

        try
        {
            var payload = JObject.Parse(response.Body);
            info.Exists = true;
            info.ExistingHash = payload["_source"]?["content_hash"]?.Value<string>();
            info.ExistingSource = payload["_source"] as JObject;
            _logger.LogTrace("Individual state for {Index}/{Id}: hash={Hash}.", info.Index, info.Id, info.ExistingHash);
            return true;
        }
        catch (JsonException ex)
        {
            info.Exists = true;
            info.ExistingHash = null;
            info.ExistingSource = null;
            _logger.LogWarning(ex, "Failed to parse individual GET response for {Index}/{Id}.", info.Index, info.Id);
            return true;
        }
    }

    private async Task<JObject?> FetchSourceAsync(IndexDocInfo info)
    {
        var response = await _client.LowLevel.GetAsync<StringResponse>(info.Index, info.Id);

        if (response.HttpStatusCode == 404)
        {
            info.Exists = false;
            info.ExistingSource = null;
            _logger.LogDebug("FetchSourceAsync: {Index}/{Id} not found.", info.Index, info.Id);
            return null;
        }

        if (!response.Success || string.IsNullOrWhiteSpace(response.Body))
        {
            _logger.LogWarning("FetchSourceAsync failed for {Index}/{Id}; status {Status}.", info.Index, info.Id, response.HttpStatusCode);
            return null;
        }

        try
        {
            var payload = JObject.Parse(response.Body);
            info.Exists = true;
            var source = payload["_source"] as JObject;
            info.ExistingSource = source;
            _logger.LogTrace("FetchSourceAsync: obtained source for {Index}/{Id}.", info.Index, info.Id);
            return source;
        }
        catch (JsonException ex)
        {
            info.Exists = true;
            info.ExistingSource = null;
            _logger.LogWarning(ex, "FetchSourceAsync: failed to parse source for {Index}/{Id}.", info.Index, info.Id);
            return null;
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
        public JObject? ExistingSource { get; set; }
        public bool EmbeddingsReused { get; set; }
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
