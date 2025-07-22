using System;
using System.IO;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.Threading.Tasks;
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
        Task<ResultObj> CreateSnapshotAsync(string snapshotRepo, string snapshotName, string indices = "documents,securitybooks");
        Task<ResultObj> RestoreSnapshotAsync(string snapshotRepo, string snapshotName, string indices = "documents,securitybooks");
        Task<ResultObj> CreateIndicesFromDataDirAsync(CreateIndexRequest createIndexRequest);

        // Add both overloads for CreateIndexAsync
        Task<ResultObj> CreateIndexAsync(CreateIndexRequest createIndexRequest, int padToTokens);
        Task<ResultObj> CreateIndexAsync(CreateIndexRequest createIndexRequest);
    }

    public class OpenSearchService : IOpenSearchService
    {
        private readonly OpenSearchHelper _openSearchHelper;
        private readonly string _encryptKey;
        private OSModelParams _modelParams = new OSModelParams();
        private readonly ILogger _logger;
        private readonly IRabbitRepo _rabbitRepo;
        private readonly string _dataDir;
        private readonly MemoryCache _cache = new MemoryCache(new MemoryCacheOptions());
        private int _maxTokenLengthCap;
        private int _minTokenLengthCap;
        private int _llmThreads;
        private readonly List<ITokenEstimationStrategy> _tokenEstimators;
        private readonly IIndexingStrategy[] _strategies;

        public OpenSearchService(
            ILogger<OpenSearchService> logger,
            ISystemParamsHelper systemParamsHelper,
            IRabbitRepo rabbitRepo,
            IEmbeddingGenerator embeddingGenerator
        )
        {
            _logger = logger;
            _encryptKey = systemParamsHelper.GetSystemParams().LLMEncryptKey;
            _modelParams.EmbeddingModelDir = systemParamsHelper.GetMLParams().EmbeddingModelDir;
            _modelParams.EmbeddingModelVecDim = systemParamsHelper.GetMLParams().EmbeddingModelVecDim;
            _modelParams.Key = systemParamsHelper.GetMLParams().OpenSearchKey;
            _modelParams.User = systemParamsHelper.GetMLParams().OpenSearchUser;
            _modelParams.Url = systemParamsHelper.GetMLParams().OpenSearchUrl;
            _maxTokenLengthCap = systemParamsHelper.GetMLParams().MaxTokenLengthCap;
            _minTokenLengthCap = systemParamsHelper.GetMLParams().MinTokenLengthCap;
            _modelParams.DefaultIndex = systemParamsHelper.GetMLParams().OpenSearchDefaultIndex;
            _rabbitRepo = rabbitRepo;
            _dataDir = systemParamsHelper.GetSystemParams().DataDir;

            _llmThreads = systemParamsHelper.GetMLParams().LlmThreads;
            _strategies = new IIndexingStrategy[]
            {
                new DocumentIndexingStrategy(),
                new SecurityBookIndexingStrategy()
            };


            _tokenEstimators = new List<ITokenEstimationStrategy>
            {
                new DocumentTokenEstimationStrategy(),
                new SecurityBookTokenEstimationStrategy()
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
        public async Task<ResultObj> CreateSnapshotAsync(string snapshotRepo, string snapshotName, string indices = "documents,securitybooks")
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
        public async Task<ResultObj> RestoreSnapshotAsync(string snapshotRepo, string snapshotName, string indices = "documents,securitybooks")
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

            if (EncryptHelper.IsBadKey(_encryptKey, createIndexRequest.AuthKey, createIndexRequest.AppID))
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

                var tempTokenizer = new AutoTokenizer(_modelParams.EmbeddingModelDir, _maxTokenLengthCap);
                int padToTokens = _minTokenLengthCap;
                int actualMaxTokens = _minTokenLengthCap;
                bool bailedEarly = false;

                var estimator = _tokenEstimators.FirstOrDefault(t => t.CanHandle(indexName));
                if (estimator == null)
                {
                    result.Message += $"No token estimation strategy found for index '{indexName}', skipping.\n";
                    continue;
                }

                foreach (var jsonFile in jsonFiles)
                {
                    if (bailedEarly) break;
                    var jsonContent = File.ReadAllText(jsonFile);
                    var sampleItems = _strategies.First(d => d.CanHandle(indexName)).Deserialize(jsonContent);
                    foreach (var item in sampleItems)
                    {
                        var fields = estimator.GetFields(item);
                        foreach (var text in fields)
                        {
                            int tokenCount = tempTokenizer.CountTokens(text);
                            if (tokenCount > actualMaxTokens)
                                actualMaxTokens = tokenCount;
                            if (tokenCount > padToTokens)
                                padToTokens = tokenCount;
                            if (padToTokens >= _maxTokenLengthCap)
                            {
                                bailedEarly = true;
                                break;
                            }
                        }
                        if (bailedEarly) break;
                    }
                }

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

                bool shouldRecreateIndex = jsonFiles.Length > 0;

                foreach (var jsonFile in jsonFiles)
                {
                    var req = new CreateIndexRequest
                    {
                        IndexName = indexName,
                        JsonFile = jsonFile,
                        AppID = createIndexRequest.AppID,
                        AuthKey = createIndexRequest.AuthKey,
                        RecreateIndex = shouldRecreateIndex,
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

            if (EncryptHelper.IsBadKey(_encryptKey, queryIndexRequest.AuthKey, queryIndexRequest.AppID))
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


                        var searchResponse = await _openSearchHelper.SearchDocumentsAsync(queryIndexRequest.QueryText, queryIndexRequest.IndexName, useMaxTokens, queryIndexRequest.VectorSearchMode);

                        if (searchResponse != null)
                        {
                            foreach (var hit in searchResponse.Hits.HitsList)
                            {
                                queryResults.Add(new QueryResultObj
                                {
                                    Input = hit.Source.Input,
                                    Output = hit.Source.Output
                                });
                            }
                            queryIndexRequest.Success = true;
                            result.Message += $"Query executed successfully on index '{queryIndexRequest.IndexName}'.";
                        }
                    }
                    queryIndexRequest.QueryResults = queryResults;
                    // Cache the results forever (until service restart)
                    _cache.Set(cacheKey, queryResults);
                }
                queryIndexRequest.Message = result.Message;
                await _rabbitRepo.PublishAsync<QueryIndexRequest>("queryIndexResult" + queryIndexRequest.AppID, queryIndexRequest, queryIndexRequest.RoutingKey);
                result.Success = queryIndexRequest.Success;
                result.Message += queryIndexRequest.Message;
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Message += $"Error: Failed to query index '{queryIndexRequest.IndexName}'. Exception: {ex.Message}";
            }

            return result;
        }


    }
}