using OpenSearch.Client;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
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
    private const string HistoryIndexName = "llm_history";
    private const string HistoryTurnsIndexName = "llm_history_turns";
    private readonly ConcurrentDictionary<string, HashSet<string>> _currentVisibleTurnIdsBySession =
        new(StringComparer.Ordinal);

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
        CancellationToken cancellationToken = default,
        string? userId = null,
        string? sessionId = null,
        int topK = 0,
        bool includeToolTurns = false,
        bool includeMetadata = false,
        string? anchorDocId = null,
        string? anchorChunkId = null,
        int neighborWindow = 0,
        string? filterDocId = null,
        string? filterChunkId = null,
        string? filterSourceFile = null,
        string? filterSectionPath = null,
        int filterPageStart = 0,
        int filterPageEnd = 0,
        int filterChunkIndexMin = 0,
        int filterChunkIndexMax = 0)
    {
        if (indexName.Equals(HistoryTurnsIndexName, StringComparison.OrdinalIgnoreCase))
        {
            return await SearchHistoryTurnsAsync(
                queryText,
                indexName,
                requestTimeout,
                cancellationToken,
                userId,
                sessionId,
                topK,
                includeToolTurns);
        }

        if (string.IsNullOrWhiteSpace(queryText))
        {
            return await SearchAnchorOrFilterOnlyAsync(
                indexName,
                topK,
                requestTimeout,
                cancellationToken,
                includeMetadata,
                anchorDocId,
                anchorChunkId,
                neighborWindow,
                filterDocId,
                filterChunkId,
                filterSourceFile,
                filterSectionPath,
                filterPageStart,
                filterPageEnd,
                filterChunkIndexMin,
                filterChunkIndexMax);
        }

        int size = topK > 0 ? Math.Clamp(topK, 1, 20) : (includeMetadata ? 6 : 3);
        var queryEmbedding = await GenerateEmbeddingAsync(queryText, padToTokens);
        var searchResponse = new SearchResponseObj();

        if (queryEmbedding.Count == 0)
        {
            throw new Exception("Failed to generate query embedding.");
        }

        var strategy = StrategyForIndex(indexName);
        string vectorFieldName = strategy.GetVectorField(mode);

        // Construct the k-NN search request body with dynamic field name
        var metadataFilters = BuildMetadataFilters(
            filterDocId,
            filterChunkId,
            filterSourceFile,
            filterSectionPath,
            filterPageStart,
            filterPageEnd,
            filterChunkIndexMin,
            filterChunkIndexMax);

        object queryObj;
        if (metadataFilters.Count == 0)
        {
            queryObj = new
            {
                knn = new Dictionary<string, object>
                {
                    [vectorFieldName] = new
                    {
                        vector = queryEmbedding,
                        k = size
                    }
                }
            };
        }
        else
        {
            queryObj = new
            {
                @bool = new
                {
                    must = new object[]
                    {
                        new
                        {
                            knn = new Dictionary<string, object>
                            {
                                [vectorFieldName] = new
                                {
                                    vector = queryEmbedding,
                                    k = size
                                }
                            }
                        }
                    },
                    filter = metadataFilters
                }
            };
        }

        var requestBody = new
        {
            size,
            query = queryObj
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

        if (includeMetadata && neighborWindow > 0)
        {
            var baseHits = searchResponse.Hits?.HitsList ?? new List<Hit>();
            if (baseHits.Count > 0)
            {
                Hit? anchorHit = await ResolveAnchorHitAsync(
                    indexName,
                    anchorDocId,
                    anchorChunkId,
                    requestTimeout,
                    cancellationToken);

                anchorHit ??= baseHits.FirstOrDefault();

                if (TryGetLocator(anchorHit, out var resolvedDocId, out var resolvedChunkIndex))
                {
                    var neighbors = await FetchNeighborHitsAsync(
                        indexName,
                        resolvedDocId,
                        resolvedChunkIndex,
                        Math.Clamp(neighborWindow, 1, 8),
                        requestTimeout,
                        cancellationToken);

                    if (neighbors.Count > 0)
                    {
                        var merged = new List<Hit>(baseHits.Count + neighbors.Count);
                        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

                        foreach (var hit in baseHits)
                        {
                            if (string.IsNullOrWhiteSpace(hit.Id) || seen.Add(hit.Id))
                                merged.Add(hit);
                        }

                        foreach (var hit in neighbors)
                        {
                            if (string.IsNullOrWhiteSpace(hit.Id) || seen.Add(hit.Id))
                                merged.Add(hit);
                        }

                        int limit = Math.Clamp(size + (Math.Clamp(neighborWindow, 1, 8) * 2), 1, 50);
                        searchResponse.Hits ??= new Hits();
                        searchResponse.Hits.HitsList = merged.Take(limit).ToList();
                    }
                }
            }
        }

        return searchResponse;
    }

    private async Task<SearchResponseObj> SearchAnchorOrFilterOnlyAsync(
        string indexName,
        int topK,
        TimeSpan? requestTimeout,
        CancellationToken cancellationToken,
        bool includeMetadata,
        string? anchorDocId,
        string? anchorChunkId,
        int neighborWindow,
        string? filterDocId,
        string? filterChunkId,
        string? filterSourceFile,
        string? filterSectionPath,
        int filterPageStart,
        int filterPageEnd,
        int filterChunkIndexMin,
        int filterChunkIndexMax)
    {
        int size = topK > 0 ? Math.Clamp(topK, 1, 20) : (includeMetadata ? 6 : 3);
        var searchResponse = new SearchResponseObj();

        if (includeMetadata && neighborWindow > 0 && !string.IsNullOrWhiteSpace(anchorChunkId))
        {
            var anchorHit = await ResolveAnchorHitAsync(
                indexName,
                anchorDocId,
                anchorChunkId,
                requestTimeout,
                cancellationToken);

            if (TryGetLocator(anchorHit, out var resolvedDocId, out var resolvedChunkIndex))
            {
                var neighbors = await FetchNeighborHitsAsync(
                    indexName,
                    resolvedDocId,
                    resolvedChunkIndex,
                    Math.Clamp(neighborWindow, 1, 8),
                    requestTimeout,
                    cancellationToken);

                var merged = new List<Hit>();
                var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

                if (anchorHit != null && (string.IsNullOrWhiteSpace(anchorHit.Id) || seen.Add(anchorHit.Id)))
                    merged.Add(anchorHit);

                foreach (var hit in neighbors)
                {
                    if (string.IsNullOrWhiteSpace(hit.Id) || seen.Add(hit.Id))
                        merged.Add(hit);
                }

                searchResponse.Hits = new Hits
                {
                    HitsList = merged.Take(Math.Clamp(size + (Math.Clamp(neighborWindow, 1, 8) * 2), 1, 50)).ToList(),
                    MaxScore = 0
                };
                searchResponse.Took = 0;
                searchResponse.TimedOut = false;
                return searchResponse;
            }
        }

        var metadataFilters = BuildMetadataFilters(
            filterDocId,
            filterChunkId,
            filterSourceFile,
            filterSectionPath,
            filterPageStart,
            filterPageEnd,
            filterChunkIndexMin,
            filterChunkIndexMax);

        if (metadataFilters.Count == 0)
            return searchResponse;

        var requestBody = new
        {
            size,
            sort = new object[]
            {
                new Dictionary<string, object> { ["chunk_index"] = new { order = "asc" } }
            },
            query = new
            {
                @bool = new
                {
                    filter = metadataFilters
                }
            }
        };

        var jsonContent = JsonConvert.SerializeObject(requestBody);
        using var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
        var response = await PostWithTimeoutAsync($"/{indexName}/_search", content, requestTimeout, cancellationToken);
        if (!response.IsSuccessStatusCode)
            return searchResponse;

        var responseBody = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<SearchResponseObj>(responseBody) ?? new SearchResponseObj();
    }

    private static List<object> BuildMetadataFilters(
        string? filterDocId,
        string? filterChunkId,
        string? filterSourceFile,
        string? filterSectionPath,
        int filterPageStart,
        int filterPageEnd,
        int filterChunkIndexMin,
        int filterChunkIndexMax)
    {
        var filters = new List<object>();

        if (!string.IsNullOrWhiteSpace(filterDocId))
            filters.Add(BuildExactTermFilter("doc_id", filterDocId));
        if (!string.IsNullOrWhiteSpace(filterChunkId))
            filters.Add(BuildExactTermFilter("chunk_id", filterChunkId));
        if (!string.IsNullOrWhiteSpace(filterSourceFile))
            filters.Add(BuildExactTermFilter("source_file", filterSourceFile));
        if (!string.IsNullOrWhiteSpace(filterSectionPath))
            filters.Add(BuildExactTermFilter("section_path", filterSectionPath));

        if (filterPageStart > 0 && filterPageEnd > 0)
        {
            // Overlap semantics: chunk window intersects requested page window.
            filters.Add(new { range = new Dictionary<string, object> { ["page_start"] = new { lte = filterPageEnd } } });
            filters.Add(new { range = new Dictionary<string, object> { ["page_end"] = new { gte = filterPageStart } } });
        }
        else if (filterPageStart > 0)
        {
            filters.Add(new { range = new Dictionary<string, object> { ["page_end"] = new { gte = filterPageStart } } });
        }
        else if (filterPageEnd > 0)
        {
            filters.Add(new { range = new Dictionary<string, object> { ["page_start"] = new { lte = filterPageEnd } } });
        }

        if (filterChunkIndexMin > 0 || filterChunkIndexMax > 0)
        {
            var chunkRange = new Dictionary<string, object>();
            if (filterChunkIndexMin > 0)
                chunkRange["gte"] = filterChunkIndexMin;
            if (filterChunkIndexMax > 0)
                chunkRange["lte"] = filterChunkIndexMax;
            filters.Add(new { range = new Dictionary<string, object> { ["chunk_index"] = chunkRange } });
        }

        return filters;
    }

    private static object BuildExactTermFilter(string field, string value)
    {
        return new
        {
            @bool = new
            {
                should = new object[]
                {
                    new { term = new Dictionary<string, object> { [field] = value } },
                    new { term = new Dictionary<string, object> { [$"{field}.keyword"] = value } }
                },
                minimum_should_match = 1
            }
        };
    }

    private async Task<Hit?> ResolveAnchorHitAsync(
        string indexName,
        string? anchorDocId,
        string? anchorChunkId,
        TimeSpan? requestTimeout,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(anchorChunkId))
            return null;

        var filters = new List<object>
        {
            BuildExactTermFilter("chunk_id", anchorChunkId)
        };
        if (!string.IsNullOrWhiteSpace(anchorDocId))
        {
            filters.Add(BuildExactTermFilter("doc_id", anchorDocId));
        }

        var requestBody = new
        {
            size = 1,
            query = new
            {
                @bool = new
                {
                    filter = filters
                }
            }
        };

        var json = JsonConvert.SerializeObject(requestBody);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await PostWithTimeoutAsync($"/{indexName}/_search", content, requestTimeout, cancellationToken);
        if (!response.IsSuccessStatusCode)
            return null;

        var body = await response.Content.ReadAsStringAsync();
        var parsed = JsonConvert.DeserializeObject<SearchResponseObj>(body);
        return parsed?.Hits?.HitsList?.FirstOrDefault();
    }

    private async Task<List<Hit>> FetchNeighborHitsAsync(
        string indexName,
        string docId,
        int chunkIndex,
        int window,
        TimeSpan? requestTimeout,
        CancellationToken cancellationToken)
    {
        int minChunk = Math.Max(0, chunkIndex - window);
        int maxChunk = chunkIndex + window;
        int size = Math.Clamp((window * 2) + 1, 1, 25);

        var requestBody = new
        {
            size,
            sort = new object[]
            {
                new Dictionary<string, object> { ["chunk_index"] = new { order = "asc" } }
            },
            query = new
            {
                @bool = new
                {
                    filter = new object[]
                    {
                        new { term = new Dictionary<string, object> { ["doc_id"] = docId } },
                        new
                        {
                            range = new Dictionary<string, object>
                            {
                                ["chunk_index"] = new { gte = minChunk, lte = maxChunk }
                            }
                        }
                    }
                }
            }
        };

        var json = JsonConvert.SerializeObject(requestBody);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await PostWithTimeoutAsync($"/{indexName}/_search", content, requestTimeout, cancellationToken);
        if (!response.IsSuccessStatusCode)
            return new List<Hit>();

        var body = await response.Content.ReadAsStringAsync();
        var parsed = JsonConvert.DeserializeObject<SearchResponseObj>(body);
        return parsed?.Hits?.HitsList ?? new List<Hit>();
    }

    private static bool TryGetLocator(Hit? hit, out string docId, out int chunkIndex)
    {
        docId = string.Empty;
        chunkIndex = -1;
        if (hit?.Source?.ExtensionData == null)
            return false;

        if (!hit.Source.ExtensionData.TryGetValue("doc_id", out var docIdToken) || docIdToken == null)
            return false;

        if (!hit.Source.ExtensionData.TryGetValue("chunk_index", out var chunkIndexToken) || chunkIndexToken == null)
            return false;

        docId = docIdToken.ToString() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(docId))
            return false;

        if (!int.TryParse(chunkIndexToken.ToString(), out chunkIndex))
            return false;

        return chunkIndex >= 0;
    }

    private async Task<SearchResponseObj> SearchHistoryTurnsAsync(
        string queryText,
        string indexName,
        TimeSpan? requestTimeout,
        CancellationToken cancellationToken,
        string? userId,
        string? sessionId,
        int topK,
        bool includeToolTurns)
    {
        await EnsureHistoryTurnsIndexExistsAsync();
        var size = topK > 0 ? Math.Min(topK, 20) : 5;

        var filters = new List<object>();
        if (!string.IsNullOrWhiteSpace(userId))
            filters.Add(new { term = new Dictionary<string, object> { ["user_id"] = userId } });
        if (!string.IsNullOrWhiteSpace(sessionId))
            filters.Add(new { term = new Dictionary<string, object> { ["session_id"] = sessionId } });

        var mustNot = new List<object>();
        if (!includeToolTurns)
        {
            mustNot.Add(new { terms = new Dictionary<string, object> { ["turn_type"] = new[] { "tool_call", "tool_response" } } });
        }

        object queryObj;
        if (string.IsNullOrWhiteSpace(queryText))
        {
            queryObj = new
            {
                @bool = new
                {
                    filter = filters,
                    must_not = mustNot
                }
            };
        }
        else
        {
            queryObj = new
            {
                @bool = new
                {
                    filter = filters,
                    must_not = mustNot,
                    must = new object[]
                    {
                        new
                        {
                            multi_match = new
                            {
                                query = queryText,
                                fields = new[] { "output^3", "input^1", "tool_name^2", "tool_status^1" }
                            }
                        }
                    }
                }
            };
        }

        var requestBody = new
        {
            size,
            sort = new object[]
            {
                "_score",
                new Dictionary<string, object> { ["start_unix_time"] = new { order = "desc" } },
                new Dictionary<string, object> { ["turn_index"] = new { order = "desc" } }
            },
            query = queryObj
        };

        var json = JsonConvert.SerializeObject(requestBody);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await PostWithTimeoutAsync($"/{indexName}/_search", content, requestTimeout, cancellationToken);
        if (!response.IsSuccessStatusCode)
            throw new Exception($"Search failed: {response.ReasonPhrase}");

        var responseBody = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<SearchResponseObj>(responseBody) ?? new SearchResponseObj();
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

    private static string BuildHistoryDocId(string serviceId, string sessionId)
    {
        var cleanService = string.IsNullOrWhiteSpace(serviceId) ? "default" : serviceId.Trim().ToLowerInvariant();
        var cleanSession = string.IsNullOrWhiteSpace(sessionId) ? "missing" : sessionId.Trim().ToLowerInvariant();
        var raw = $"{cleanService}:{cleanSession}";
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(raw))).ToLowerInvariant();
    }

    public async Task EnsureHistoryIndexExistsAsync()
    {
        var exists = await _client.Indices.ExistsAsync(HistoryIndexName);
        if (exists.Exists)
        {
            _logger.LogDebug("History index '{Index}' already exists.", HistoryIndexName);
            return;
        }

        _logger.LogInformation("History index '{Index}' not found. Creating it now.", HistoryIndexName);

        var mapping = @"
{
  ""settings"": {
    ""index"": { ""number_of_shards"": 1, ""number_of_replicas"": 1 }
  },
  ""mappings"": {
    ""properties"": {
      ""service_id"": { ""type"": ""keyword"" },
      ""session_id"": { ""type"": ""keyword"" },
      ""user_id"": { ""type"": ""keyword"" },
      ""start_unix_time"": { ""type"": ""long"" },
      ""name"": { ""type"": ""text"" },
      ""llm_type"": { ""type"": ""keyword"" },
      ""history_json"": { ""type"": ""text"" },
      ""updated_at"": { ""type"": ""date"" }
    }
  }
}";
        var create = await _client.LowLevel.Indices.CreateAsync<StringResponse>(HistoryIndexName, PostData.String(mapping));
        if (!create.Success)
        {
            _logger.LogError("Failed creating history index '{Index}'. Debug={Debug}", HistoryIndexName, create.DebugInformation);
            throw new InvalidOperationException($"Failed creating history index '{HistoryIndexName}': {create.DebugInformation}");
        }

        _logger.LogInformation("History index '{Index}' created successfully.", HistoryIndexName);
    }

    public async Task<HistoryStoreResponse> UpsertHistoryAsync(HistoryStoreRequest request)
    {
        await EnsureHistoryIndexExistsAsync();

        var docId = BuildHistoryDocId(request.ServiceId, request.SessionId);
        _logger.LogInformation(
            "History upsert start index={Index} docId={DocId} service={ServiceId} session={SessionId} user={UserId} historyBytes={HistoryBytes}",
            HistoryIndexName,
            docId,
            request.ServiceId,
            request.SessionId,
            request.UserId,
            request.HistoryJson?.Length ?? 0);
        var body = new
        {
            service_id = request.ServiceId,
            session_id = request.SessionId,
            user_id = request.UserId,
            start_unix_time = request.StartUnixTime,
            name = request.Name,
            llm_type = request.LlmType,
            history_json = request.HistoryJson,
            updated_at = DateTime.UtcNow
        };
        var json = JsonConvert.SerializeObject(body);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await _httpClient.PutAsync($"/{HistoryIndexName}/_doc/{docId}?refresh=true", content);
        var responseBody = await response.Content.ReadAsStringAsync();
        _logger.LogInformation(
            "History upsert end index={Index} docId={DocId} status={StatusCode} success={Success}",
            HistoryIndexName,
            docId,
            (int)response.StatusCode,
            response.IsSuccessStatusCode);
        if (response.IsSuccessStatusCode)
        {
            await IndexHistoryTurnsAsync(request);
        }

        return new HistoryStoreResponse
        {
            Success = response.IsSuccessStatusCode,
            Message = response.IsSuccessStatusCode
                ? "History upserted."
                : $"Failed to upsert history: {response.StatusCode} {responseBody}"
        };
    }

    private sealed class TurnDoc
    {
        public int TurnIndex { get; set; }
        public string Role { get; set; } = "";
        public string TurnType { get; set; } = "text";
        public string Input { get; set; } = "";
        public string Output { get; set; } = "";
        public string ToolName { get; set; } = "";
        public string ToolStatus { get; set; } = "";
        public string ToolCallId { get; set; } = "";
    }

    private async Task IndexHistoryTurnsAsync(HistoryStoreRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.HistoryJson)) return;
        await EnsureHistoryTurnsIndexExistsAsync();

        var root = JObject.Parse(request.HistoryJson);
        var historyArray = (root["history"] as JArray) ?? (root["History"] as JArray);
        if (historyArray == null || historyArray.Count == 0) return;

        var turns = BuildTurnDocs(historyArray);
        foreach (var turn in turns)
        {
            var turnDocId = BuildHistoryDocId(request.ServiceId, $"{request.SessionId}:{turn.TurnIndex}");
            var body = new
            {
                service_id = request.ServiceId,
                session_id = request.SessionId,
                user_id = request.UserId,
                start_unix_time = request.StartUnixTime,
                llm_type = request.LlmType,
                turn_index = turn.TurnIndex,
                role = turn.Role,
                turn_type = turn.TurnType,
                input = turn.Input,
                output = turn.Output,
                tool_name = turn.ToolName,
                tool_status = turn.ToolStatus,
                tool_call_id = turn.ToolCallId,
                updated_at = DateTime.UtcNow
            };
            var json = JsonConvert.SerializeObject(body);
            using var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await _httpClient.PutAsync($"/{HistoryTurnsIndexName}/_doc/{turnDocId}?refresh=false", content);
            if (!response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                _logger.LogWarning(
                    "History turn upsert failed index={Index} docId={DocId} status={Status} body={Body}",
                    HistoryTurnsIndexName, turnDocId, (int)response.StatusCode, responseBody);
            }
        }
    }

    private static List<TurnDoc> BuildTurnDocs(JArray historyArray)
    {
        var turns = new List<TurnDoc>(historyArray.Count);
        for (var index = 0; index < historyArray.Count; index++)
        {
            if (historyArray[index] is not JObject msg) continue;
            var role = GetString(msg, "role", "Role").ToLowerInvariant();
            var content = GetString(msg, "content", "Content");
            var hasToolCalls = HasToolCalls(msg, out var toolNames);
            var toolCallId = GetString(msg, "toolCallId", "tool_call_id", "ToolCallId");
            var toolName = GetString(msg, "name", "Name");

            if (role == "user" || role == "assistant")
            {
                if (!string.IsNullOrWhiteSpace(content))
                {
                    turns.Add(new TurnDoc
                    {
                        TurnIndex = index,
                        Role = role,
                        TurnType = "text",
                        Input = $"{role} turn",
                        Output = content
                    });
                    continue;
                }

                if (hasToolCalls)
                {
                    turns.Add(new TurnDoc
                    {
                        TurnIndex = index,
                        Role = role,
                        TurnType = "tool_call",
                        Input = "assistant tool call",
                        Output = $"Assistant invoked tool(s): {toolNames}",
                        ToolName = toolNames,
                        ToolStatus = "requested",
                        ToolCallId = toolCallId
                    });
                }

                continue;
            }

            if (role == "tool")
            {
                var status = InferToolStatus(content);
                turns.Add(new TurnDoc
                {
                    TurnIndex = index,
                    Role = role,
                    TurnType = "tool_response",
                    Input = "tool response",
                    Output = $"Tool '{toolName}' returned status: {status}",
                    ToolName = toolName,
                    ToolStatus = status,
                    ToolCallId = toolCallId
                });
            }
        }

        return turns;
    }

    private static bool HasToolCalls(JObject message, out string toolNames)
    {
        toolNames = string.Empty;
        var token = message["toolCalls"] ?? message["tool_calls"] ?? message["ToolCalls"];
        if (token is not JArray arr || arr.Count == 0) return false;

        var names = new List<string>();
        foreach (var item in arr.OfType<JObject>())
        {
            var functionObj = item["function"] as JObject;
            var name = functionObj?["name"]?.Value<string>()
                       ?? item["name"]?.Value<string>()
                       ?? string.Empty;
            if (!string.IsNullOrWhiteSpace(name)) names.Add(name);
        }
        toolNames = names.Count == 0 ? "unknown" : string.Join(",", names.Distinct(StringComparer.OrdinalIgnoreCase));
        return true;
    }

    private static string InferToolStatus(string content)
    {
        if (string.IsNullOrWhiteSpace(content)) return "unknown";
        var lower = content.ToLowerInvariant();
        if (lower.Contains("timeout")) return "timeout";
        if (lower.Contains("cancel")) return "canceled";
        if (lower.Contains("error") || lower.Contains("exception") || lower.Contains("failed")) return "error";
        return "success";
    }

    private static string GetString(JObject obj, params string[] names)
    {
        foreach (var name in names)
        {
            var token = obj[name];
            if (token == null || token.Type == JTokenType.Null) continue;
            if (token.Type == JTokenType.String) return token.Value<string>() ?? string.Empty;
            if (token.Type == JTokenType.Array)
            {
                var array = token as JArray;
                if (array == null) continue;
                var parts = new List<string>();
                foreach (var entry in array.OfType<JObject>())
                {
                    var type = entry["type"]?.Value<string>() ?? "";
                    if (type.Equals("text", StringComparison.OrdinalIgnoreCase))
                    {
                        var text = entry["text"]?.Value<string>();
                        if (!string.IsNullOrWhiteSpace(text)) parts.Add(text);
                    }
                }
                if (parts.Count > 0) return string.Join("\n", parts);
            }
            return token.ToString(Formatting.None);
        }
        return string.Empty;
    }

    public async Task EnsureHistoryTurnsIndexExistsAsync()
    {
        var ensureResult = await EnsureHistoryTurnsIndexExistsAsync(HistoryTurnsIndexName);
        if (!ensureResult.Success)
            throw new InvalidOperationException($"Failed creating history turns index '{HistoryTurnsIndexName}': {ensureResult.Message}");
    }

    public async Task<HistoryStoreResponse> GetHistoryAsync(HistoryStoreRequest request)
    {
        await EnsureHistoryIndexExistsAsync();
        var docId = BuildHistoryDocId(request.ServiceId, request.SessionId);
        _logger.LogInformation(
            "History get start index={Index} docId={DocId} service={ServiceId} session={SessionId}",
            HistoryIndexName,
            docId,
            request.ServiceId,
            request.SessionId);
        var response = await _httpClient.GetAsync($"/{HistoryIndexName}/_doc/{docId}");
        var responseBody = await response.Content.ReadAsStringAsync();
        if (!response.IsSuccessStatusCode)
        {
            _logger.LogWarning(
                "History get failed index={Index} docId={DocId} status={StatusCode} body={Body}",
                HistoryIndexName,
                docId,
                (int)response.StatusCode,
                responseBody);
            return new HistoryStoreResponse
            {
                Success = false,
                Message = $"Failed to load history: {response.StatusCode} {responseBody}"
            };
        }

        var json = JObject.Parse(responseBody);
        var source = json["_source"] as JObject;
        if (source == null)
        {
            return new HistoryStoreResponse { Success = false, Message = "History source is missing." };
        }

        _logger.LogInformation("History get success index={Index} docId={DocId}", HistoryIndexName, docId);
        return new HistoryStoreResponse
        {
            Success = true,
            Message = "History loaded.",
            Item = new HistoryStoreResultItem
            {
                ServiceId = source.Value<string>("service_id") ?? "",
                SessionId = source.Value<string>("session_id") ?? "",
                UserId = source.Value<string>("user_id") ?? "",
                StartUnixTime = source.Value<long?>("start_unix_time") ?? 0,
                Name = source.Value<string>("name") ?? "",
                LlmType = source.Value<string>("llm_type") ?? "",
                HistoryJson = source.Value<string>("history_json") ?? ""
            }
        };
    }

    public async Task<HistoryStoreResponse> DeleteHistoryAsync(HistoryStoreRequest request)
    {
        await EnsureHistoryIndexExistsAsync();
        var docId = BuildHistoryDocId(request.ServiceId, request.SessionId);
        _logger.LogInformation(
            "History delete start index={Index} docId={DocId} service={ServiceId} session={SessionId}",
            HistoryIndexName,
            docId,
            request.ServiceId,
            request.SessionId);
        var response = await _httpClient.DeleteAsync($"/{HistoryIndexName}/_doc/{docId}?refresh=true");
        var responseBody = await response.Content.ReadAsStringAsync();
        _logger.LogInformation(
            "History delete end index={Index} docId={DocId} status={StatusCode} success={Success}",
            HistoryIndexName,
            docId,
            (int)response.StatusCode,
            response.IsSuccessStatusCode);

        return new HistoryStoreResponse
        {
            Success = response.IsSuccessStatusCode,
            Message = response.IsSuccessStatusCode
                ? "History deleted."
                : $"Failed to delete history: {response.StatusCode} {responseBody}"
        };
    }

    public async Task<HistoryStoreResponse> ListHistoryAsync(HistoryStoreRequest request)
    {
        await EnsureHistoryIndexExistsAsync();
        var size = request.Limit <= 0 ? 100 : Math.Min(request.Limit, 500);
        _logger.LogInformation(
            "History list start index={Index} service={ServiceId} user={UserId} limit={Limit}",
            HistoryIndexName,
            request.ServiceId,
            request.UserId,
            size);

        var filters = new List<object>();
        if (!string.IsNullOrWhiteSpace(request.ServiceId))
        {
            filters.Add(new { term = new Dictionary<string, object> { ["service_id"] = request.ServiceId } });
        }
        if (!string.IsNullOrWhiteSpace(request.UserId))
        {
            filters.Add(new { term = new Dictionary<string, object> { ["user_id"] = request.UserId } });
        }

        object queryObj = filters.Count == 0
            ? new { match_all = new { } }
            : new { @bool = new { filter = filters } };

        var searchRequest = new
        {
            size,
            sort = new object[] { new Dictionary<string, object> { ["start_unix_time"] = new { order = "desc" } } },
            query = queryObj
        };

        var payload = JsonConvert.SerializeObject(searchRequest);
        using var content = new StringContent(payload, Encoding.UTF8, "application/json");
        var response = await _httpClient.PostAsync($"/{HistoryIndexName}/_search", content);
        var responseBody = await response.Content.ReadAsStringAsync();
        if (!response.IsSuccessStatusCode)
        {
            _logger.LogWarning(
                "History list failed index={Index} status={StatusCode} body={Body}",
                HistoryIndexName,
                (int)response.StatusCode,
                responseBody);
            return new HistoryStoreResponse
            {
                Success = false,
                Message = $"Failed to list history: {response.StatusCode} {responseBody}"
            };
        }

        var root = JObject.Parse(responseBody);
        var hits = root["hits"]?["hits"] as JArray ?? new JArray();
        var items = new List<HistoryStoreResultItem>(hits.Count);
        foreach (var hit in hits.OfType<JObject>())
        {
            var source = hit["_source"] as JObject;
            if (source == null) continue;
            items.Add(new HistoryStoreResultItem
            {
                ServiceId = source.Value<string>("service_id") ?? "",
                SessionId = source.Value<string>("session_id") ?? "",
                UserId = source.Value<string>("user_id") ?? "",
                StartUnixTime = source.Value<long?>("start_unix_time") ?? 0,
                Name = source.Value<string>("name") ?? "",
                LlmType = source.Value<string>("llm_type") ?? "",
                HistoryJson = source.Value<string>("history_json") ?? ""
            });
        }

        _logger.LogInformation(
            "History list success index={Index} service={ServiceId} user={UserId} returned={Count}",
            HistoryIndexName,
            request.ServiceId,
            request.UserId,
            items.Count);
        return new HistoryStoreResponse
        {
            Success = true,
            Message = $"Loaded {items.Count} histories.",
            Items = items
        };
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

    public async Task<ResultObj> EnsureHistoryTurnsIndexExistsAsync(string indexName = "llm_history_turns")
    {
        var result = new ResultObj { Message = "EnsureHistoryTurnsIndexExistsAsync: " };
        try
        {
            var exists = await _client.Indices.ExistsAsync(indexName);
            if (exists.Exists)
            {
                result.Success = true;
                result.Message += $"History index '{indexName}' already exists.";
                return result;
            }

            var mapping = $@"
{{
  ""settings"": {{ ""index"": {{ ""knn"": true }} }},
  ""mappings"": {{
    ""properties"": {{
      ""service_id"": {{ ""type"": ""keyword"" }},
      ""session_id"": {{ ""type"": ""keyword"" }},
      ""user_id"": {{ ""type"": ""keyword"" }},
      ""llm_type"": {{ ""type"": ""keyword"" }},
      ""role"": {{ ""type"": ""keyword"" }},
      ""turn_index"": {{ ""type"": ""integer"" }},
      ""turn_unix_time"": {{ ""type"": ""long"" }},
      ""text"": {{ ""type"": ""text"" }},
      ""content_hash"": {{ ""type"": ""keyword"" }},
      ""turn_embedding"": {{
        ""type"": ""knn_vector"",
        ""dimension"": {_modelParams.EmbeddingModelVecDim},
        ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }}
      }}
    }}
  }}
}}";

            var create = await _client.LowLevel.Indices.CreateAsync<StringResponse>(indexName, PostData.String(mapping));
            result.Success = create.Success;
            result.Message += create.Success
                ? $"History index '{indexName}' created."
                : $"Failed creating history index '{indexName}': {create.DebugInformation}";
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.Message += ex.Message;
        }

        return result;
    }

    public async Task<ResultObj> UpsertHistoryTurnsAsync(HistoryStoreRequest request, int padToTokens, string indexName = "llm_history_turns")
    {
        var result = new ResultObj { Success = false, Message = "UpsertHistoryTurnsAsync: " };
        if (request == null || string.IsNullOrWhiteSpace(request.HistoryJson))
        {
            result.Message += "No history payload.";
            return result;
        }

        try
        {
            var turns = ParseHistoryTurns(request);
            if (turns.Count == 0)
            {
                result.Success = true;
                result.Message += "No indexable turns in payload.";
                return result;
            }

            var docs = turns.Select(t =>
            {
                var hash = ComputeSha256Hash(t.Text);
                var docId = ComputeSha256Hash($"{request.ServiceId}|{request.SessionId}|{t.TurnIndex}|{t.Role}|{hash}");
                return new HistoryTurnDoc
                {
                    Id = docId,
                    ServiceId = request.ServiceId ?? string.Empty,
                    SessionId = request.SessionId ?? string.Empty,
                    UserId = request.UserId ?? string.Empty,
                    LlmType = request.LlmType ?? string.Empty,
                    Role = t.Role,
                    TurnIndex = t.TurnIndex,
                    TurnUnixTime = t.TurnUnixTime,
                    Text = t.Text,
                    ContentHash = hash
                };
            }).ToList();

            if (!string.IsNullOrWhiteSpace(request.SessionId))
            {
                var visibleIds = new HashSet<string>(docs.Select(d => d.Id), StringComparer.Ordinal);
                _currentVisibleTurnIdsBySession.AddOrUpdate(
                    request.SessionId,
                    _ => visibleIds,
                    (_, _) => visibleIds);
                _logger.LogDebug(
                    "Updated current visible turns for session {SessionId}: count={Count}",
                    request.SessionId,
                    visibleIds.Count);
            }

            var existing = await GetExistingHistoryTurnIdsAsync(indexName, docs.Select(d => d.Id).ToList());
            var created = 0;
            var skipped = 0;

            foreach (var doc in docs)
            {
                if (existing.Contains(doc.Id))
                {
                    skipped++;
                    continue;
                }

                doc.TurnEmbedding = await GenerateEmbeddingAsync(doc.Text, padToTokens);
                if (doc.TurnEmbedding.Count == 0)
                {
                    _logger.LogWarning("History turn embedding empty for session {SessionId} turn {TurnIndex}", doc.SessionId, doc.TurnIndex);
                    continue;
                }

                var body = new
                {
                    service_id = doc.ServiceId,
                    session_id = doc.SessionId,
                    user_id = doc.UserId,
                    llm_type = doc.LlmType,
                    role = doc.Role,
                    turn_index = doc.TurnIndex,
                    turn_unix_time = doc.TurnUnixTime,
                    text = doc.Text,
                    content_hash = doc.ContentHash,
                    turn_embedding = doc.TurnEmbedding
                };

                var resp = await _client.LowLevel.IndexAsync<StringResponse>(indexName, doc.Id, PostData.Serializable(body));
                if (resp.Success)
                {
                    created++;
                }
                else
                {
                    _logger.LogWarning("Failed indexing history turn {DocId}: {Debug}", doc.Id, resp.DebugInformation);
                }
            }

            result.Success = true;
            result.Message += $"Indexed turns created={created}, skipped_existing={skipped}, total={docs.Count}.";
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.Message += ex.Message;
        }

        return result;
    }

    public async Task<ResultObj> DeleteHistoryTurnsBySessionAsync(string sessionId, string serviceId = "", string indexName = "llm_history_turns")
    {
        var result = new ResultObj { Success = false, Message = "DeleteHistoryTurnsBySessionAsync: " };
        if (string.IsNullOrWhiteSpace(sessionId))
        {
            result.Message += "sessionId is required.";
            return result;
        }

        try
        {
            var must = new List<object>
            {
                new { term = new Dictionary<string, object> { ["session_id"] = sessionId } }
            };

            if (!string.IsNullOrWhiteSpace(serviceId))
            {
                must.Add(new { term = new Dictionary<string, object> { ["service_id"] = serviceId } });
            }

            var query = new
            {
                query = new
                {
                    @bool = new
                    {
                        must
                    }
                }
            };

            var response = await _client.LowLevel.DeleteByQueryAsync<StringResponse>(indexName, PostData.Serializable(query));
            result.Success = response.Success;
            result.Message += response.Success
                ? $"Deleted history turns for session {sessionId}."
                : $"Delete by query failed: {response.DebugInformation}";
            if (response.Success)
            {
                _currentVisibleTurnIdsBySession.TryRemove(sessionId, out _);
            }
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.Message += ex.Message;
        }

        return result;
    }

    public async Task<List<MemoryQueryResult>> SearchHistoryTurnsAsync(
        string queryText,
        int padToTokens,
        string indexName = "llm_history_turns",
        string userId = "",
        string sessionId = "",
        string currentSessionId = "",
        int topK = 8,
        bool includeToolTurns = false,
        TimeSpan? requestTimeout = null,
        CancellationToken cancellationToken = default)
    {
        var queryEmbedding = await GenerateEmbeddingAsync(queryText, padToTokens);
        if (queryEmbedding.Count == 0)
        {
            throw new Exception("Failed to generate memory query embedding.");
        }

        topK = Math.Clamp(topK <= 0 ? 8 : topK, 1, 50);

        var filters = new List<object>();
        if (!string.IsNullOrWhiteSpace(userId))
        {
            filters.Add(new { term = new Dictionary<string, object> { ["user_id"] = userId } });
        }
        if (!string.IsNullOrWhiteSpace(sessionId))
        {
            filters.Add(new { term = new Dictionary<string, object> { ["session_id"] = sessionId } });
        }
        HashSet<string>? currentVisibleTurnIds = null;
        if (!string.IsNullOrWhiteSpace(currentSessionId))
        {
            _currentVisibleTurnIdsBySession.TryGetValue(currentSessionId, out currentVisibleTurnIds);
        }
        var currentVisibleCount = currentVisibleTurnIds?.Count ?? 0;
        var candidateK = topK + currentVisibleCount;

        var mustNot = new List<object>();
        if (!includeToolTurns)
        {
            mustNot.Add(new { term = new Dictionary<string, object> { ["role"] = "tool" } });
        }

        var knnFilter = new
        {
            @bool = new
            {
                filter = filters,
                must_not = mustNot
            }
        };

        var requestBody = new
        {
            size = candidateK,
            query = new
            {
                knn = new Dictionary<string, object>
                {
                    ["turn_embedding"] = new
                    {
                        vector = queryEmbedding,
                        k = candidateK,
                        filter = knnFilter
                    }
                }
            }
        };

        var json = JsonConvert.SerializeObject(requestBody);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await PostWithTimeoutAsync($"/{indexName}/_search", content, requestTimeout, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync();
            throw new Exception($"Memory search failed: {(int)response.StatusCode} {response.ReasonPhrase}. Body: {errorBody}");
        }

        var body = await response.Content.ReadAsStringAsync();
        var root = JObject.Parse(body);
        var hits = root["hits"]?["hits"] as JArray ?? new JArray();
        var results = new List<MemoryQueryResult>(topK);
        var raw = new List<MemoryQueryResult>(topK);

        foreach (var hit in hits.OfType<JObject>())
        {
            var source = hit["_source"] as JObject;
            if (source == null) continue;

            var hitSessionId = source["session_id"]?.Value<string>() ?? string.Empty;
            var hitId = hit["_id"]?.Value<string>() ?? string.Empty;
            if (!string.IsNullOrWhiteSpace(currentSessionId)
                && string.Equals(hitSessionId, currentSessionId, StringComparison.Ordinal)
                && !string.IsNullOrWhiteSpace(hitId)
                && currentVisibleTurnIds != null
                && currentVisibleTurnIds.Contains(hitId))
            {
                continue;
            }

            var metadata = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["index"] = hit["_index"]?.Value<string>() ?? string.Empty,
                ["id"] = hitId
            };

            raw.Add(new MemoryQueryResult
            {
                Role = source["role"]?.Value<string>() ?? string.Empty,
                Text = source["text"]?.Value<string>() ?? string.Empty,
                ServiceId = source["service_id"]?.Value<string>() ?? string.Empty,
                SessionId = source["session_id"]?.Value<string>() ?? string.Empty,
                UserId = source["user_id"]?.Value<string>() ?? string.Empty,
                TurnIndex = source["turn_index"]?.Value<int>() ?? 0,
                TurnUnixTime = source["turn_unix_time"]?.Value<long>() ?? 0,
                Score = hit["_score"]?.Value<float>() ?? 0f,
                Metadata = metadata
            });

            if (raw.Count >= topK)
            {
                break;
            }
        }

        await EnrichMemoryResultsWithContextAsync(raw, queryText, indexName, requestTimeout, cancellationToken);
        results.AddRange(raw);
        return results;
    }

    public async Task<List<MemoryContextTurn>> GetHistoryTurnWindowAsync(
        string sessionId,
        int turnIndex,
        int widthBefore,
        int widthAfter,
        string indexName = "llm_history_turns",
        TimeSpan? requestTimeout = null,
        CancellationToken cancellationToken = default)
    {
        widthBefore = Math.Clamp(widthBefore, 0, 20);
        widthAfter = Math.Clamp(widthAfter, 0, 20);
        var from = Math.Max(0, turnIndex - widthBefore);
        var to = Math.Max(from, turnIndex + widthAfter);

        var body = new
        {
            size = (to - from) + 1,
            sort = new object[]
            {
                new Dictionary<string, object> { ["turn_index"] = "asc" }
            },
            query = new
            {
                @bool = new
                {
                    filter = new object[]
                    {
                        new { term = new Dictionary<string, object> { ["session_id"] = sessionId } },
                        new
                        {
                            range = new Dictionary<string, object>
                            {
                                ["turn_index"] = new
                                {
                                    gte = from,
                                    lte = to
                                }
                            }
                        }
                    }
                }
            }
        };

        var json = JsonConvert.SerializeObject(body);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await PostWithTimeoutAsync($"/{indexName}/_search", content, requestTimeout, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            throw new Exception($"History turn window search failed: {response.ReasonPhrase}");
        }

        var payload = await response.Content.ReadAsStringAsync();
        var root = JObject.Parse(payload);
        var hits = root["hits"]?["hits"] as JArray ?? new JArray();
        var turns = new List<MemoryContextTurn>(hits.Count);

        foreach (var hit in hits.OfType<JObject>())
        {
            var source = hit["_source"] as JObject;
            if (source == null) continue;
            turns.Add(new MemoryContextTurn
            {
                Role = source["role"]?.Value<string>() ?? string.Empty,
                Text = source["text"]?.Value<string>() ?? string.Empty,
                TurnIndex = source["turn_index"]?.Value<int>() ?? 0,
                TurnUnixTime = source["turn_unix_time"]?.Value<long>() ?? 0
            });
        }

        return turns;
    }

    private async Task EnrichMemoryResultsWithContextAsync(
        List<MemoryQueryResult> results,
        string queryText,
        string indexName,
        TimeSpan? requestTimeout,
        CancellationToken cancellationToken)
    {
        if (results.Count == 0)
        {
            return;
        }

        // Confidence is relative to the best returned hit for this query.
        var maxScore = results.Max(r => r.Score);
        var sourceScope = results.Any(r => !string.IsNullOrWhiteSpace(r.SessionId))
            ? "session_or_user_scoped"
            : "user_global";

        // Build session->requested-index set so we fetch context turns in one request per session.
        var sessionIndexMap = new Dictionary<string, HashSet<int>>(StringComparer.Ordinal);
        foreach (var item in results)
        {
            if (string.IsNullOrWhiteSpace(item.SessionId))
            {
                continue;
            }

            if (!sessionIndexMap.TryGetValue(item.SessionId, out var set))
            {
                set = new HashSet<int>();
                sessionIndexMap[item.SessionId] = set;
            }

            set.Add(item.TurnIndex - 1);
            set.Add(item.TurnIndex);
            set.Add(item.TurnIndex + 1);
        }

        var sessionTurns = new Dictionary<string, Dictionary<int, MemoryContextTurn>>(StringComparer.Ordinal);
        foreach (var pair in sessionIndexMap)
        {
            var indices = pair.Value.Where(i => i >= 0).Distinct().ToList();
            if (indices.Count == 0)
            {
                continue;
            }

            var turns = await FetchSessionTurnsAsync(indexName, pair.Key, indices, requestTimeout, cancellationToken);
            sessionTurns[pair.Key] = turns;
        }

        foreach (var item in results)
        {
            var ratio = maxScore > 0f ? (item.Score / maxScore) : 0f;
            var confidence = ratio >= 0.9f ? "high" : ratio >= 0.75f ? "medium" : "low";

            item.ConfidenceBand = confidence;
            item.WhyMatched = $"Semantic match score={item.Score:0.###} for intent='{queryText}'.";
            item.QueryIntent = queryText;
            item.SourceScope = sourceScope;

            if (sessionTurns.TryGetValue(item.SessionId, out var turnMap))
            {
                if (turnMap.TryGetValue(item.TurnIndex - 1, out var before))
                {
                    item.ContextBefore.Add(before);
                }
                if (turnMap.TryGetValue(item.TurnIndex + 1, out var after))
                {
                    item.ContextAfter.Add(after);
                }
            }
        }
    }

    private async Task<Dictionary<int, MemoryContextTurn>> FetchSessionTurnsAsync(
        string indexName,
        string sessionId,
        List<int> turnIndices,
        TimeSpan? requestTimeout,
        CancellationToken cancellationToken)
    {
        var result = new Dictionary<int, MemoryContextTurn>();
        if (string.IsNullOrWhiteSpace(sessionId) || turnIndices.Count == 0)
        {
            return result;
        }

        var body = new
        {
            size = turnIndices.Count,
            query = new
            {
                @bool = new
                {
                    filter = new object[]
                    {
                        new { term = new Dictionary<string, object> { ["session_id"] = sessionId } },
                        new { terms = new Dictionary<string, object> { ["turn_index"] = turnIndices } }
                    }
                }
            }
        };

        var json = JsonConvert.SerializeObject(body);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await PostWithTimeoutAsync($"/{indexName}/_search", content, requestTimeout, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            return result;
        }

        var payload = await response.Content.ReadAsStringAsync();
        var root = JObject.Parse(payload);
        var hits = root["hits"]?["hits"] as JArray ?? new JArray();
        foreach (var hit in hits.OfType<JObject>())
        {
            var source = hit["_source"] as JObject;
            if (source == null) continue;
            var idx = source["turn_index"]?.Value<int>() ?? -1;
            if (idx < 0) continue;

            result[idx] = new MemoryContextTurn
            {
                Role = source["role"]?.Value<string>() ?? string.Empty,
                Text = source["text"]?.Value<string>() ?? string.Empty,
                TurnIndex = idx,
                TurnUnixTime = source["turn_unix_time"]?.Value<long>() ?? 0
            };
        }

        return result;
    }

    private async Task<HashSet<string>> GetExistingHistoryTurnIdsAsync(string indexName, List<string> ids)
    {
        var existing = new HashSet<string>(StringComparer.Ordinal);
        if (ids.Count == 0) return existing;

        var payload = new
        {
            docs = ids.Select(id => new
            {
                _index = indexName,
                _id = id,
                _source = false
            })
        };

        var response = await _client.LowLevel.DoRequestAsync<StringResponse>(
            OpenSearch.Net.HttpMethod.POST,
            "_mget",
            cancellationToken: CancellationToken.None,
            data: PostData.Serializable(payload));

        if (!response.Success || string.IsNullOrWhiteSpace(response.Body))
        {
            return existing;
        }

        var root = JObject.Parse(response.Body);
        var docs = root["docs"] as JArray;
        if (docs == null) return existing;

        foreach (var doc in docs.OfType<JObject>())
        {
            if ((doc["found"]?.Value<bool>() ?? false))
            {
                var id = doc["_id"]?.Value<string>();
                if (!string.IsNullOrWhiteSpace(id))
                {
                    existing.Add(id);
                }
            }
        }

        return existing;
    }

    private static List<ParsedHistoryTurn> ParseHistoryTurns(HistoryStoreRequest request)
    {
        var turns = new List<ParsedHistoryTurn>();
        var root = JObject.Parse(request.HistoryJson);
        var history = root["history"] as JArray;
        if (history == null) return turns;

        long baseUnix = request.StartUnixTime > 0
            ? request.StartUnixTime
            : DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        for (var i = 0; i < history.Count; i++)
        {
            if (history[i] is not JObject msg) continue;
            var role = msg["role"]?.Value<string>() ?? string.Empty;
            var content = msg["content"]?.Value<string>() ?? string.Empty;

            string text = string.Empty;
            if (!string.IsNullOrWhiteSpace(content))
            {
                text = content.Trim();
            }
            else if (string.Equals(role, "assistant", StringComparison.OrdinalIgnoreCase) && msg["tool_calls"] is JArray toolCalls && toolCalls.Count > 0)
            {
                text = "Assistant issued a tool call.";
            }
            else if (string.Equals(role, "tool", StringComparison.OrdinalIgnoreCase))
            {
                text = ExtractToolStatusText(content);
            }

            if (string.IsNullOrWhiteSpace(text)) continue;

            turns.Add(new ParsedHistoryTurn
            {
                Role = string.IsNullOrWhiteSpace(role) ? "unknown" : role.Trim().ToLowerInvariant(),
                Text = text,
                TurnIndex = i,
                TurnUnixTime = baseUnix + i
            });
        }

        return turns;
    }

    private static string ExtractToolStatusText(string content)
    {
        if (string.IsNullOrWhiteSpace(content))
        {
            return "Tool response received.";
        }

        try
        {
            var token = JToken.Parse(content);
            if (token is JObject obj)
            {
                var status = obj["status"]?.ToString();
                var success = obj["success"]?.ToString();
                var message = obj["message"]?.ToString();
                var error = obj["error"]?.ToString();
                var compact = $"{status} {success} {message} {error}".Trim();
                if (!string.IsNullOrWhiteSpace(compact))
                {
                    return compact.Length > 280 ? compact.Substring(0, 280) : compact;
                }
            }
        }
        catch
        {
            // treat as plain text below
        }

        var plain = content.Trim();
        return plain.Length > 280 ? plain.Substring(0, 280) : plain;
    }

    private sealed class ParsedHistoryTurn
    {
        public string Role { get; set; } = string.Empty;
        public string Text { get; set; } = string.Empty;
        public int TurnIndex { get; set; }
        public long TurnUnixTime { get; set; }
    }

    private sealed class HistoryTurnDoc
    {
        public string Id { get; set; } = string.Empty;
        public string ServiceId { get; set; } = string.Empty;
        public string SessionId { get; set; } = string.Empty;
        public string UserId { get; set; } = string.Empty;
        public string LlmType { get; set; } = string.Empty;
        public string Role { get; set; } = string.Empty;
        public int TurnIndex { get; set; }
        public long TurnUnixTime { get; set; }
        public string Text { get; set; } = string.Empty;
        public string ContentHash { get; set; } = string.Empty;
        public List<float> TurnEmbedding { get; set; } = new List<float>();
    }
}
