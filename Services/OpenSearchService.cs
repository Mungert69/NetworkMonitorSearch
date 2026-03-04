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
        Task<ResultObj> QueryIndexAsync(QueryIndexRequest? queryIndexRequest);
        Task<HistoryStoreResponse> HandleHistoryStoreAsync(HistoryStoreRequest request);

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
            IEmbeddingGenerator embeddingGenerator,
            ILoggerFactory loggerFactory
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

            _openSearchHelper = new OpenSearchHelper(
                _modelParams,
                embeddingGenerator,
                loggerFactory.CreateLogger<OpenSearchHelper>(),
                _strategies);

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

        public async Task<HistoryStoreResponse> HandleHistoryStoreAsync(HistoryStoreRequest request)
        {
            if (request == null)
            {
                return new HistoryStoreResponse { Success = false, Message = "Request is null." };
            }

            if (EncryptHelper.IsBadKey(_llmEncryptKey, request.AuthKey, request.AppID))
            {
                _logger.LogWarning(
                    "HistoryStore auth failed for app={AppID} operation={Operation} service={ServiceId} session={SessionId}",
                    request.AppID,
                    request.Operation,
                    request.ServiceId,
                    request.SessionId);
                return new HistoryStoreResponse
                {
                    Success = false,
                    Message = $"Failed history store request, bad AuthKey for AppID {request.AppID}"
                };
            }

            try
            {
                _logger.LogInformation(
                    "HistoryStore dispatch operation={Operation} service={ServiceId} session={SessionId} user={UserId} app={AppID} messageId={MessageID}",
                    request.Operation,
                    request.ServiceId,
                    request.SessionId,
                    request.UserId,
                    request.AppID,
                    request.MessageID);

                var response = request.Operation switch
                {
                    HistoryStoreOperation.upsert => await _openSearchHelper.UpsertHistoryAsync(request),
                    HistoryStoreOperation.get => await _openSearchHelper.GetHistoryAsync(request),
                    HistoryStoreOperation.delete => await _openSearchHelper.DeleteHistoryAsync(request),
                    HistoryStoreOperation.list => await _openSearchHelper.ListHistoryAsync(request),
                    _ => new HistoryStoreResponse
                    {
                        Success = false,
                        Message = $"Unsupported operation: {request.Operation}"
                    }
                };

                response.MessageID = request.MessageID;
                var responseExchange = string.IsNullOrWhiteSpace(request.ResponseExchange)
                    ? $"{request.AppID}HistoryStoreResult"
                    : request.ResponseExchange;
                if (string.IsNullOrWhiteSpace(request.RoutingKey))
                {
                    await _rabbitRepo.PublishAsync(responseExchange, response);
                }
                else
                {
                    await _rabbitRepo.PublishAsync(responseExchange, response, request.RoutingKey);
                }

                _logger.LogInformation(
                    "HistoryStore completed operation={Operation} service={ServiceId} session={SessionId} success={Success} responseExchange={ResponseExchange}",
                    request.Operation,
                    request.ServiceId,
                    request.SessionId,
                    response.Success,
                    responseExchange);

                return response;
            }
            catch (Exception ex)
            {
                _logger.LogError(
                    ex,
                    "HistoryStore failed operation={Operation} service={ServiceId} session={SessionId} app={AppID}",
                    request.Operation,
                    request.ServiceId,
                    request.SessionId,
                    request.AppID);
                return new HistoryStoreResponse
                {
                    Success = false,
                    Message = $"History store operation failed: {ex.GetType().Name}: {ex.Message}"
                };
            }
        }

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
                var obj = JsonConvert.DeserializeObject<JObject>(File.ReadAllText(file));
                if (obj == null)
                    return (null, null);

                int loaded = obj.Value<int?>("padToTokens") ?? _minTokenLengthCap;
                int? actual = null;
                try
                {
                    actual = obj["actualMaxTokens"]?.Type == JTokenType.Null
                        ? null
                        : obj.Value<int?>("actualMaxTokens");
                }
                catch { }
                _indexMaxTokens[indexName] = loaded;
                return (loaded, actual);
            }
            return (null, null);
        }

        private (int padToTokens, int actualMaxTokens) CalculatePadToTokensFromItems(string indexName, IIndexingStrategy strategy, IEnumerable<object> items)
        {
            int pad = _minTokenLengthCap;
            int maxSeen = _minTokenLengthCap;

            try
            {
                var tokenizer = new AutoTokenizer(_modelParams.EmbeddingModelDir, _maxTokenLengthCap);

                foreach (var item in items)
                {
                    foreach (var text in strategy.GetFields(item) ?? Array.Empty<string>())
                    {
                        if (string.IsNullOrWhiteSpace(text))
                            continue;

                        int tokens = tokenizer.CountTokens(text);
                        maxSeen = Math.Max(maxSeen, tokens);
                        pad = Math.Max(pad, tokens);
                        if (pad >= _maxTokenLengthCap)
                            break;
                    }
                }

                pad = Math.Clamp(pad, _minTokenLengthCap, _maxTokenLengthCap);
                maxSeen = Math.Max(maxSeen, _minTokenLengthCap);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to calculate padToTokens for index {Index}; defaulting to {Pad}.", indexName, _minTokenLengthCap);
                pad = _minTokenLengthCap;
                maxSeen = _minTokenLengthCap;
            }

            return (pad, maxSeen);
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
            int padValue = padToTokens.HasValue ? padToTokens.Value : -1;

            return await CreateIndexAsync(createIndexRequest, padValue);
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

                var sourceDescriptor = !string.IsNullOrWhiteSpace(createIndexRequest.JsonFile)
                    ? createIndexRequest.JsonFile
                    : "inline payload";
                Console.WriteLine($"Loaded {items.Count} documents for index '{createIndexRequest.IndexName}' from '{sourceDescriptor}'.");
                result.Message += $"Documents={items.Count}. ";

                int effectivePadToTokens = padToTokens;
                if (effectivePadToTokens <= 0)
                {
                    var (calculatedPad, actualMaxTokens) = CalculatePadToTokensFromItems(createIndexRequest.IndexName, deserializer, items);
                    effectivePadToTokens = calculatedPad;
                    SaveIndexMaxTokens(createIndexRequest.IndexName, effectivePadToTokens, actualMaxTokens);
                    Console.WriteLine($"Calculated padToTokens for index '{createIndexRequest.IndexName}' = {effectivePadToTokens} (actual max {actualMaxTokens}).");
                    result.Message += $"Computed padToTokens={effectivePadToTokens}, actualMaxTokens={actualMaxTokens}. ";
                }
                else
                {
                    SaveIndexMaxTokens(createIndexRequest.IndexName, effectivePadToTokens);
                }

                Console.WriteLine($"Deserialization for index '{createIndexRequest.IndexName}' succeeded. Indexing with {effectivePadToTokens} tokens.");

                if (createIndexRequest.IncrementalUpdate)
                {
                    Console.WriteLine($"Incremental mode: retaining index '{createIndexRequest.IndexName}' and updating changed documents only.");
                    result.Message += $"Mode=incremental update for '{createIndexRequest.IndexName}'. ";
                }
                else if (createIndexRequest.RecreateIndex)
                {
                    Console.WriteLine($"Full rebuild mode: index '{createIndexRequest.IndexName}' will be deleted and recreated.");
                    result.Message += $"Mode=full rebuild for '{createIndexRequest.IndexName}'. ";
                }
                else
                {
                    Console.WriteLine($"Append mode: indexing new documents into '{createIndexRequest.IndexName}'.");
                    result.Message += $"Mode=append for '{createIndexRequest.IndexName}'. ";
                }

                var resultEn = await _openSearchHelper.EnsureIndexExistsAsync(indexName: createIndexRequest.IndexName, recreateIndex: createIndexRequest.RecreateIndex);
                if (!resultEn.Success) return resultEn;

                var resultIn = await _openSearchHelper.IndexDocumentsAsync(items, effectivePadToTokens, createIndexRequest.IncrementalUpdate);
                createIndexRequest.Success = resultEn.Success && resultIn.Success;
                createIndexRequest.Message += resultEn.Message + resultIn.Message;

                var responseExchange = string.IsNullOrWhiteSpace(createIndexRequest.ResponseExchange)
                    ? "createIndexResult" + createIndexRequest.AppID
                    : createIndexRequest.ResponseExchange;
                await _rabbitRepo.PublishAsync(responseExchange, createIndexRequest);

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
                        MessageID = createIndexRequest.MessageID,
                        ResponseExchange = createIndexRequest.ResponseExchange,
                        IncrementalUpdate = createIndexRequest.IncrementalUpdate
                    };

                    var createResult = await CreateIndexAsync(req, padToTokens);
                    result.Message += $"Index '{indexName}', File '{Path.GetFileName(jsonFile)}': MaxTokensUsed {padToTokens}, ActualMaxTokens {actualMaxTokens} : {createResult.Message}\n";
                    if (!createResult.Success)
                        result.Success = false;
                }
            }

            return result;
        }

        public async Task<ResultObj> QueryIndexAsync(QueryIndexRequest? queryIndexRequest)
        {
            var result = new ResultObj();
            result.Success = true;
            result.Message = "MessageAPI: QueryIndexAsync: ";

            // Sanity checks
            if (queryIndexRequest == null)
            {
                result.Message += "Error: queryIndexRequest is null.";
                result.Success = false;
            }
            var request = queryIndexRequest ?? new QueryIndexRequest();
            request.Success = false;

            if (EncryptHelper.IsBadKey(_llmEncryptKey, request.AuthKey, request.AppID))
            {
                //result.Success = false;
                result.Message += $" Error : Failed QueryIndexAsync bad AuthKey for AppID {request.AppID}";
                _logger.LogError(result.Message);
                return result;
            }

            if (string.IsNullOrWhiteSpace(request.IndexName))
            {
                result.Message += "Error: indexName is null or empty.";
                result.Success = false;

            }

            if (string.IsNullOrWhiteSpace(request.QueryText))
            {
                result.Message += "Error: queryText is null or empty.";
                result.Success = false;
            }
            string appID = request.AppID ?? "";
            /*if (appID != "nmap" && appID != "meta")
            {
                result.Message += $" Warning : not applying Rag for LLM type {appID} .";
                result.Success = false;
            }*/

            try
            {
                var queryResults = new List<QueryResultObj>();
                string cacheKey = $"query:{request.IndexName}:{request.QueryText}";

                if (_cache.TryGetValue(cacheKey, out List<QueryResultObj>? cachedResults))
                {
                    request.QueryResults = cachedResults ?? new List<QueryResultObj>();
                    request.Success = true;
                    result.Message += $"Cache hit for query on index '{request.IndexName}'.";
                }
                else
                {
                    if (result.Success)
                    {
                        // Load the pad to tokens for this index
                        var (padToTokens, _) = LoadIndexMaxTokens(request.IndexName);
                        int useMaxTokens = padToTokens ?? _minTokenLengthCap;

                        var targetUri = _openSearchHelper.SearchUri;
                        _logger.LogInformation(
                            "MessageAPI: QueryIndexAsync: connecting to OpenSearch at {Uri} for index {IndexName}",
                            targetUri,
                            request.IndexName);
                        result.Message += $"Attempting OpenSearch query on '{targetUri}' for index '{request.IndexName}'. ";

                        var searchResponse = await _openSearchHelper.SearchDocumentsAsync(
                            request.QueryText,
                            request.IndexName,
                            useMaxTokens,
                            request.VectorSearchMode,
                            ResolveQueryTimeout(request.IndexName));

                        if (searchResponse != null)
                        {
                            var hitsList = searchResponse.Hits?.HitsList ?? new List<Hit>();
                            int hitCount = hitsList.Count;
                            float maxScore = searchResponse.Hits?.MaxScore ?? 0;
                            int took = searchResponse.Took;
                            bool timedOut = searchResponse.TimedOut;

                            var strategy = _strategies.FirstOrDefault(s => s.CanHandle(request.IndexName));
                            foreach (var hit in hitsList)
                            {
                                if (strategy != null)
                                    queryResults.Add(strategy.MapSearchHitToResult(hit));
                                else
                                    queryResults.Add(new QueryResultObj
                                    {
                                        Input = hit.Source?.Input ?? string.Empty,
                                        Output = hit.Source?.Output ?? string.Empty,
                                        Score = hit.Score,
                                        Metadata = new Dictionary<string, string>()
                                    });
                            }
                            request.Success = true;
                            result.Message += $"Query executed successfully on index '{request.IndexName}'. ";
                            result.Message += $"Hits: {hitCount}, MaxScore: {maxScore}, Took: {took}ms, TimedOut: {timedOut}.";
                        }
                    }
                    request.QueryResults = queryResults;
                    // Cache the results forever (until service restart)
                    _cache.Set(cacheKey, queryResults);
                }
                request.Message = result.Message;
                var responseExchange = string.IsNullOrWhiteSpace(request.ResponseExchange)
                    ? $"{request.AppID}QueryIndexResult"
                    : request.ResponseExchange;

                if (string.IsNullOrEmpty(request.RoutingKey))
                {
                    await _rabbitRepo.PublishAsync<QueryIndexRequest>(responseExchange, request);
                }
                else
                {
                    await _rabbitRepo.PublishAsync<QueryIndexRequest>(responseExchange, request, request.RoutingKey);
                }
                result.Success = request.Success;
                result.Message += request.Message;
            }
            catch (Exception ex)
            {
                result.Success = false;
                var targetUri = _openSearchHelper.SearchUri;
                var failureStage = DescribeFailureStage(ex);
                result.Message += $"Error during {failureStage} for index '{request.IndexName}' targeting '{targetUri}': {ex.GetType().Name}: {ex.Message}";
                _logger.LogError(ex,
                    "MessageAPI: QueryIndexAsync: error during {Stage} for index {IndexName} at {Uri}",
                    failureStage,
                    request.IndexName,
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
