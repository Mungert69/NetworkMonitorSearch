using System;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Threading;
using System.Net.Http;
using Microsoft.Extensions.Logging;
using NetworkMonitor.Objects;
using NetworkMonitor.Objects.Repository;
using NetworkMonitor.Utils.Helpers;
using Microsoft.Extensions.Caching.Memory;
using System.Linq;

namespace NetworkMonitor.Search.Services
{
    public interface IOpenSearchService
    {
        Task Init();
        Task<ResultObj> QueryIndexAsync(QueryIndexRequest queryIndexRequest);

        // New methods for snapshot and bulk index creation
        Task<ResultObj> CreateSnapshotAsync(string snapshotRepo, string snapshotName, string indices = "documents,mitre,securitybooks");
        Task<ResultObj> RestoreSnapshotAsync(string snapshotRepo, string snapshotName, string indices = "documents,mitre,securitybooks");
        Task<ResultObj> CreateIndicesFromDataDirAsync(CreateIndexRequest createIndexRequest);

        // Add both overloads for CreateIndexAsync
        Task<ResultObj> CreateIndexAsync(CreateIndexRequest createIndexRequest, int padToTokens);
        Task<ResultObj> CreateIndexAsync(CreateIndexRequest createIndexRequest);
    }

    public class OpenSearchService : IOpenSearchService
    {
        private readonly OpenSearchHelper _openSearchHelper;
        private readonly string _llmEncryptKey;
        private OSModelParams _modelParams = new OSModelParams();
        private readonly ILogger _logger;
        private readonly IRabbitRepo _rabbitRepo;
        private readonly string _dataDir;
        private readonly MemoryCache _cache = new MemoryCache(new MemoryCacheOptions());
        private int _maxTokenLengthCap;
        private int _minTokenLengthCap;
        private int _llmThreads;
        private readonly IIndexingStrategy[] _strategies;

        public OpenSearchService(
            ILogger<OpenSearchService> logger,
            MLParams mlParams,
            SystemParams systemParams,
            IRabbitRepo rabbitRepo,
            IEmbeddingGenerator embeddingGenerator
        )
        {
            _logger = logger;
            _rabbitRepo = rabbitRepo;

            // Map MLParams to OSModelParams
            _modelParams = new OSModelParams
            {
                EmbeddingModelDir = mlParams.EmbeddingModelDir,
                EmbeddingModelVecDim = mlParams.EmbeddingModelVecDim,
                User = mlParams.OpenSearchUser,
                Key = mlParams.OpenSearchKey,
                Url = mlParams.OpenSearchUrl,
                DefaultIndex = mlParams.OpenSearchDefaultIndex,
                HttpTimeout = mlParams.OpenSearchHttpTimeoutSeconds > 0
                    ? TimeSpan.FromSeconds(mlParams.OpenSearchHttpTimeoutSeconds)
                    : Timeout.InfiniteTimeSpan
                // Add any other properties that need to be mapped
            };
            _llmThreads = mlParams.LlmThreads;
            _maxTokenLengthCap = mlParams.MaxTokenLengthCap;
            _minTokenLengthCap = mlParams.MinTokenLengthCap;
            
            _dataDir = systemParams.DataDir;
            _llmEncryptKey = systemParams.LLMEncryptKey;

            if (mlParams.OpenSearchIndexTimeoutSeconds != null)
            {
                foreach (var kvp in mlParams.OpenSearchIndexTimeoutSeconds)
                {
                    if (string.IsNullOrWhiteSpace(kvp.Key))
                        continue;
                    _indexQueryTimeouts[kvp.Key] = kvp.Value > 0
                        ? TimeSpan.FromSeconds(kvp.Value)
                        : Timeout.InfiniteTimeSpan;
                }
            }


            _strategies = new IIndexingStrategy[]
            {
                new MitreIndexingStrategy(),
                new DocumentIndexingStrategy(),
                new SecurityBookIndexingStrategy(),
                new QuantumBookIndexingStrategy(),
                new BlogIndexingStrategy()
            };

            _openSearchHelper = new OpenSearchHelper(_modelParams, embeddingGenerator, _strategies);

            // Log all parameters read in the constructor
            _logger.LogInformation(
                $"OpenSearchService initialized with: EmbeddingModelDir={_modelParams.EmbeddingModelDir}\nEmbeddingModelVecDim={_modelParams.EmbeddingModelVecDim}\nOpenSearchUser={_modelParams.User}\n" +
                $"OpenSearchUrl={_modelParams.Url}\nMaxTokenLengthCap={_maxTokenLengthCap}\nMinTokenLengthCap={_minTokenLengthCap}\n" +
                $"OpenSearchDefaultIndex={_modelParams.DefaultIndex}\nDataDir={_dataDir}\nLlmThreads={_llmThreads}\n"
            );
        }

        // Create a snapshot for the given indices
        public async Task<ResultObj> CreateSnapshotAsync(string snapshotRepo, string snapshotName, string indices = "documents,mitre,securitybooks,blogs")
        {
            var result = new ResultObj();
            try
            {
                using var handler = new System.Net.Http.HttpClientHandler
                {
                    ServerCertificateCustomValidationCallback = (message, cert, chain, sslPolicyErrors) => true
                };
                using var httpClient = new System.Net.Http.HttpClient(handler);
                var url = $"{_modelParams.Url}/_snapshot/{snapshotRepo}/{snapshotName}";
                var requestBody = new
                {
                    indices = indices,
                    ignore_unavailable = true,
                    include_global_state = false
                };
                var jsonContent = JsonConvert.SerializeObject(requestBody);
                var content = new System.Net.Http.StringContent(jsonContent, System.Text.Encoding.UTF8, "application/json");
                var byteArray = System.Text.Encoding.ASCII.GetBytes($"{_modelParams.User}:{_modelParams.Key}");
                httpClient.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Basic", Convert.ToBase64String(byteArray));
                var response = await httpClient.PutAsync(url, content);
                if (response.IsSuccessStatusCode)
                {
                    result.Success = true;
                    result.Message = $"Snapshot '{snapshotName}' created successfully in repo '{snapshotRepo}'.";
                }
                else
                {
                    result.Success = false;
                    result.Message = $"Failed to create snapshot: {response.StatusCode} {await response.Content.ReadAsStringAsync()}";
                }
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Message = $"Exception during snapshot creation: {ex.Message}";
            }
            return result;
        }

        // Restore a snapshot for the given indices
        public async Task<ResultObj> RestoreSnapshotAsync(string snapshotRepo, string snapshotName, string indices )
        {
            var result = new ResultObj();
            try
            {
                using var handler = new System.Net.Http.HttpClientHandler
                {
                    ServerCertificateCustomValidationCallback = (message, cert, chain, sslPolicyErrors) => true
                };
                using var httpClient = new System.Net.Http.HttpClient(handler);
                var url = $"{_modelParams.Url}/_snapshot/{snapshotRepo}/{snapshotName}/_restore";
                var requestBody = new
                {
                    indices = indices,
                    include_global_state = false
                };
                var jsonContent = JsonConvert.SerializeObject(requestBody);
                var content = new System.Net.Http.StringContent(jsonContent, System.Text.Encoding.UTF8, "application/json");
                var byteArray = System.Text.Encoding.ASCII.GetBytes($"{_modelParams.User}:{_modelParams.Key}");
                httpClient.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Basic", Convert.ToBase64String(byteArray));
                var response = await httpClient.PostAsync(url, content);
                if (response.IsSuccessStatusCode)
                {
                    result.Success = true;
                    result.Message = $"Snapshot '{snapshotName}' restored successfully from repo '{snapshotRepo}'.";
                }
                else
                {
                    result.Success = false;
                    result.Message = $"Failed to restore snapshot: {response.StatusCode} {await response.Content.ReadAsStringAsync()}";
                }
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Message = $"Exception during snapshot restore: {ex.Message}";
            }
            return result;
        }


        public Task Init()
        {
            return Task.CompletedTask;
        }

        // Store the dataSamples for use in embedding generator initialization
        private IEnumerable<string>? _pendingDataSamples = null;

        // Store padToTokens per index
        private readonly Dictionary<string, int> _indexMaxTokens = new();
        private readonly Dictionary<string, TimeSpan> _indexQueryTimeouts = new(StringComparer.OrdinalIgnoreCase);



        private void SaveIndexMaxTokens(string indexName, int padToTokens, int actualMaxTokens = -1)
        {
            // Save in memory
            _indexMaxTokens[indexName] = padToTokens;
            // Persist to disk (simple JSON file per index)
            var configDir = Path.Combine(_dataDir, "index_config");
            Directory.CreateDirectory(configDir);
            var file = Path.Combine(configDir, $"{indexName}_padtokens.json");
            File.WriteAllText(file, JsonConvert.SerializeObject(new { padToTokens, actualMaxTokens }));
        }

        private (int? padToTokens, int? actualMaxTokens) LoadIndexMaxTokens(string indexName)
        {
            // Try in-memory first
            if (_indexMaxTokens.TryGetValue(indexName, out var val))
                return (val, null);
            // Try disk
            var configDir = Path.Combine(_dataDir, "index_config");
            var file = Path.Combine(configDir, $"{indexName}_padtokens.json");
            if (File.Exists(file))
            {
                var obj = JsonConvert.DeserializeObject<dynamic>(File.ReadAllText(file));
                int loaded = (int)obj.padToTokens;
                int? actual = null;
                try
                {
                    actual = obj.actualMaxTokens != null ? (int)obj.actualMaxTokens : (int?)null;
                }
                catch { }
                _indexMaxTokens[indexName] = loaded;
                return (loaded, actual);
            }
            return (null, null);
        }

        private TimeSpan? ResolveQueryTimeout(string indexName)
        {
            if (!string.IsNullOrWhiteSpace(indexName) &&
                _indexQueryTimeouts.TryGetValue(indexName, out var overrideTimeout))
            {
                return overrideTimeout == Timeout.InfiniteTimeSpan ? (TimeSpan?)null : overrideTimeout;
            }

            return _modelParams.HttpTimeout == Timeout.InfiniteTimeSpan ? (TimeSpan?)null : _modelParams.HttpTimeout;
        }


        // Overload: CreateIndexAsync that looks up padToTokens from index name
        public async Task<ResultObj> CreateIndexAsync(CreateIndexRequest createIndexRequest)
        {
            if (createIndexRequest == null || string.IsNullOrWhiteSpace(createIndexRequest.IndexName))
            {
                return new ResultObj { Success = false, Message = "Error: createIndexRequest or IndexName is null." };
            }

            // Try to load pad tokens for this index, fail if not found
            var (padToTokens, _) = LoadIndexMaxTokens(createIndexRequest.IndexName);
            if (!padToTokens.HasValue)
            {
                return new ResultObj
                {
                    Success = false,
                    Message = $"Error: Could not find padToTokens for index '{createIndexRequest.IndexName}'."
                };
            }

            return await CreateIndexAsync(createIndexRequest, padToTokens.Value);
        }
        public async Task<ResultObj> CreateIndexAsync(CreateIndexRequest createIndexRequest, int padToTokens)
        {
            var result = new ResultObj { Success = false, Message = "MessageAPI: CreateIndexAsync: " };

            if (createIndexRequest == null)
            {
                result.Message += "Error: createIndexRequest is null.";
                return result;
            }

            if (EncryptHelper.IsBadKey(_llmEncryptKey, createIndexRequest.AuthKey, createIndexRequest.AppID))
            {
                result.Message += $" Error : Failed QueryIndexAsync bad AuthKey for AppID {createIndexRequest.AppID}";
                _logger.LogError(result.Message);
                return result;
            }

            if (string.IsNullOrWhiteSpace(createIndexRequest.IndexName))
            {
                result.Message += "Error: indexName is null or empty.";
                return result;
            }

            if (string.IsNullOrWhiteSpace(createIndexRequest.JsonMapping) && string.IsNullOrWhiteSpace(createIndexRequest.JsonFile))
            {
                result.Message += "Error: JsonMapping and JsonFile are null or empty.";
                return result;
            }

            try
            {
                string jsonContent = !string.IsNullOrEmpty(createIndexRequest.JsonFile)
                    ? await File.ReadAllTextAsync(createIndexRequest.JsonFile)
                    : createIndexRequest.JsonMapping;

                if (string.IsNullOrWhiteSpace(jsonContent))
                {
                    result.Message += "Error: Json is null or empty.";
                    return result;
                }

                var deserializer = _strategies.FirstOrDefault(d => d.CanHandle(createIndexRequest.IndexName));
                if (deserializer == null)
                {
                    result.Message += $"No deserialization strategy for index '{createIndexRequest.IndexName}'.";
                    return result;
                }

                var items = deserializer.Deserialize(jsonContent);
                if (items == null || items.Count == 0)
                {
                    result.Message += $"No items parsed from index '{createIndexRequest.IndexName}'.";
                    return result;
                }

                Console.WriteLine($"Deserialization for index '{createIndexRequest.IndexName}' succeeded. Indexing with {padToTokens} tokens.");

                var resultEn = await _openSearchHelper.EnsureIndexExistsAsync(indexName: createIndexRequest.IndexName, recreateIndex: createIndexRequest.RecreateIndex);
                if (!resultEn.Success) return resultEn;

                var resultIn = await _openSearchHelper.IndexDocumentsAsync(items, padToTokens);
                createIndexRequest.Success = resultEn.Success && resultIn.Success;
                createIndexRequest.Message += resultEn.Message + resultIn.Message;

                await _rabbitRepo.PublishAsync("createIndexResult" + createIndexRequest.AppID, createIndexRequest);

                result.Success = createIndexRequest.Success;
                result.Message += createIndexRequest.Message;
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Message += $"Error: Failed to create index '{createIndexRequest.IndexName}'. Exception: {ex.Message}";
            }

            return result;
        }

        public async Task<ResultObj> CreateIndicesFromDataDirAsync(CreateIndexRequest createIndexRequest)
        {
            var result = new ResultObj();
            result.Success = true;
            result.Message = $"Starting CreateIndicesFromDataDirAsync for {createIndexRequest.JsonFile}\n";
            string dataDir = _dataDir;
            if (string.IsNullOrWhiteSpace(dataDir) || !Directory.Exists(dataDir))
            {
                result.Success = false;
                result.Message += $"Error: Data Directory '{dataDir}' does not exist.";
                return result;
            }

            var indexDirs = Directory.GetDirectories(dataDir);
            if (indexDirs.Length == 0)
            {
                result.Message += "No subdirectories (indices) found in data dir.";
                return result;
            }

            foreach (var indexDir in indexDirs)
            {
                var indexName = Path.GetFileName(indexDir);
                if (string.Equals(indexName, "index_config", StringComparison.OrdinalIgnoreCase))
                {
                    result.Message += $"Skipping special directory '{indexName}'.\n";
                    continue;
                }

                var jsonFiles = Directory.GetFiles(indexDir, "*.json");
                if (jsonFiles.Length == 0)
                {
                    result.Message += $"Index '{indexName}': No JSON files found, skipping.\n";
                    continue;
                }

                int padToTokens = _minTokenLengthCap;
                int actualMaxTokens = _minTokenLengthCap;

                var strategy = _strategies.FirstOrDefault(s => s.CanHandle(indexName));
                if (strategy == null)
                {
                    result.Message += $"No indexing strategy found for index '{indexName}', skipping.\n";
                    continue;
                }

                (padToTokens, actualMaxTokens) = strategy.EstimatePadding(jsonFiles, _modelParams.EmbeddingModelDir, _maxTokenLengthCap, _minTokenLengthCap);

                padToTokens = Math.Min(padToTokens, _maxTokenLengthCap);
                var (loadedMax, loadedActual) = LoadIndexMaxTokens(indexName);
                if (!loadedMax.HasValue)
                {
                    SaveIndexMaxTokens(indexName, padToTokens, actualMaxTokens);
                }
                else
                {
                    padToTokens = loadedMax.Value;
                    if (loadedActual.HasValue)
                        actualMaxTokens = loadedActual.Value;
                }

                for (int i = 0; i < jsonFiles.Length; i++)
                {
                    var jsonFile = jsonFiles[i];
                    var req = new CreateIndexRequest
                    {
                        IndexName = indexName,
                        JsonFile = jsonFile,
                        AppID = createIndexRequest.AppID,
                        AuthKey = createIndexRequest.AuthKey,
                        RecreateIndex = (i == 0), // Only recreate for the first file
                        JsonMapping = "",
                        MessageID = createIndexRequest.MessageID
                    };

                    var createResult = await CreateIndexAsync(req, padToTokens);
                    result.Message += $"Index '{indexName}', File '{Path.GetFileName(jsonFile)}': MaxTokensUsed {padToTokens}, ActualMaxTokens {actualMaxTokens} : {createResult.Message}\n";
                    if (!createResult.Success)
                        result.Success = false;
                }
            }

            return result;
        }

        public async Task<ResultObj> QueryIndexAsync(QueryIndexRequest queryIndexRequest)
        {
            var result = new ResultObj();
            result.Success = true;
            result.Message = "MessageAPI: QueryIndexAsync: ";

            // Sanity checks
            if (queryIndexRequest == null)
            {
                result.Message += "Error: queryIndexRequest is null.";
                result.Success = false;
                queryIndexRequest = new QueryIndexRequest();
            }
            queryIndexRequest.Success = false;

            if (EncryptHelper.IsBadKey(_llmEncryptKey, queryIndexRequest.AuthKey, queryIndexRequest.AppID))
            {
                //result.Success = false;
                result.Message += $" Error : Failed QueryIndexAsync bad AuthKey for AppID {queryIndexRequest.AppID}";       
                _logger.LogError(result.Message);
                return result;
            }

            if (string.IsNullOrWhiteSpace(queryIndexRequest.IndexName))
            {
                result.Message += "Error: indexName is null or empty.";
                result.Success = false;

            }

            if (string.IsNullOrWhiteSpace(queryIndexRequest.QueryText))
            {
                result.Message += "Error: queryText is null or empty.";
                result.Success = false;
            }
            string appID = queryIndexRequest?.AppID ?? "";
            /*if (appID != "nmap" && appID != "meta")
            {
                result.Message += $" Warning : not applying Rag for LLM type {appID} .";
                result.Success = false;
            }*/

            try
            {
                var queryResults = new List<QueryResultObj>();
                string cacheKey = $"query:{queryIndexRequest.IndexName}:{queryIndexRequest.QueryText}";

                if (_cache.TryGetValue(cacheKey, out List<QueryResultObj> cachedResults))
                {
                    queryIndexRequest.QueryResults = cachedResults;
                    queryIndexRequest.Success = true;
                    result.Message += $"Cache hit for query on index '{queryIndexRequest.IndexName}'.";
                }
                else
                {
                    if (result.Success)
                    {
                        // Load the pad to tokens for this index
                        var (padToTokens, _) = LoadIndexMaxTokens(queryIndexRequest.IndexName);
                        int useMaxTokens = padToTokens ?? _minTokenLengthCap;

                        var targetUri = _openSearchHelper.SearchUri;
                        _logger.LogInformation(
                            "MessageAPI: QueryIndexAsync: connecting to OpenSearch at {Uri} for index {IndexName}",
                            targetUri,
                            queryIndexRequest.IndexName);
                        result.Message += $"Attempting OpenSearch query on '{targetUri}' for index '{queryIndexRequest.IndexName}'. ";

                        var searchResponse = await _openSearchHelper.SearchDocumentsAsync(
                            queryIndexRequest.QueryText,
                            queryIndexRequest.IndexName,
                            useMaxTokens,
                            queryIndexRequest.VectorSearchMode,
                            ResolveQueryTimeout(queryIndexRequest.IndexName));

                        if (searchResponse != null)
                        {
                            int hitCount = searchResponse.Hits?.HitsList?.Count ?? 0;
                            float maxScore = searchResponse.Hits?.MaxScore ?? 0;
                            int took = searchResponse.Took;
                            bool timedOut = searchResponse.TimedOut;

                            foreach (var hit in searchResponse.Hits.HitsList)
                            {
                                var metadata = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

                                if (!string.IsNullOrWhiteSpace(hit.Index))
                                {
                                    metadata["index"] = hit.Index;
                                }

                                if (hit.Source?.ExtensionData != null)
                                {
                                    foreach (var kvp in hit.Source.ExtensionData)
                                    {
                                        if (kvp.Key.EndsWith("_embedding", StringComparison.OrdinalIgnoreCase))
                                        {
                                            continue;
                                        }

                                        if (kvp.Key.Equals("content", StringComparison.OrdinalIgnoreCase))
                                        {
                                            continue;
                                        }

                                        var value = kvp.Value;
                                        switch (value?.Type)
                                        {
                                            case JTokenType.Array:
                                                metadata[kvp.Key] = value.ToString(Formatting.None);
                                                break;
                                            case JTokenType.Object:
                                                metadata[kvp.Key] = value.ToString(Formatting.None);
                                                break;
                                            case JTokenType.Null:
                                                metadata[kvp.Key] = string.Empty;
                                                break;
                                            default:
                                                metadata[kvp.Key] = value?.ToString() ?? string.Empty;
                                                break;
                                        }
                                    }
                                }

                                // Preserve the canonical title/summary when present.
                                if (!string.IsNullOrWhiteSpace(hit.Source?.Input))
                                {
                                    metadata.TryAdd("title", hit.Source.Input);
                                }

                                if (!metadata.ContainsKey("summary") && hit.Source?.Output != null)
                                {
                                    metadata["summary"] = hit.Source.Output;
                                }

                                queryResults.Add(new QueryResultObj
                                {
                                    Input = hit.Source?.Input ?? string.Empty,
                                    Output = hit.Source?.Output ?? string.Empty,
                                    Score = hit.Score,
                                    Metadata = metadata
                                });
                            }
                            queryIndexRequest.Success = true;
                            result.Message += $"Query executed successfully on index '{queryIndexRequest.IndexName}'. ";
                            result.Message += $"Hits: {hitCount}, MaxScore: {maxScore}, Took: {took}ms, TimedOut: {timedOut}.";
                        }
                    }
                    queryIndexRequest.QueryResults = queryResults;
                    // Cache the results forever (until service restart)
                    _cache.Set(cacheKey, queryResults);
                }
                queryIndexRequest.Message = result.Message;
                if (string.IsNullOrEmpty(queryIndexRequest.RoutingKey))  await _rabbitRepo.PublishAsync<QueryIndexRequest>($"{queryIndexRequest.AppID}QueryIndexResult" , queryIndexRequest);
                else await _rabbitRepo.PublishAsync<QueryIndexRequest>("queryIndexResult" + queryIndexRequest.AppID, queryIndexRequest, queryIndexRequest.RoutingKey);
                result.Success = queryIndexRequest.Success;
                result.Message += queryIndexRequest.Message;
            }
            catch (Exception ex)
            {
                result.Success = false;
                var targetUri = _openSearchHelper.SearchUri;
                var failureStage = DescribeFailureStage(ex);
                result.Message += $"Error during {failureStage} for index '{queryIndexRequest.IndexName}' targeting '{targetUri}': {ex.GetType().Name}: {ex.Message}";
                _logger.LogError(ex,
                    "MessageAPI: QueryIndexAsync: error during {Stage} for index {IndexName} at {Uri}",
                    failureStage,
                    queryIndexRequest.IndexName,
                    targetUri);
            }

            return result;
        }

        private static string DescribeFailureStage(Exception ex)
        {
            if (ex is InvalidOperationException inv &&
                inv.Message.Contains("Embedding provider", StringComparison.OrdinalIgnoreCase))
                return "embedding provider request";

            if (ex is TimeoutException timeout &&
                timeout.Message.Contains("Embedding", StringComparison.OrdinalIgnoreCase))
                return "embedding provider timeout";

            if (ex is TaskCanceledException)
                return "HTTP request timeout";

            if (ex is HttpRequestException)
                return "HTTP request to external service";

            if (ex.InnerException != null)
                return DescribeFailureStage(ex.InnerException);

            return ex.GetType().Name;
        }


    }
}
