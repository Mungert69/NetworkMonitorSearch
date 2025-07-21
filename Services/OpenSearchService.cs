using System;
using System.IO;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using NetworkMonitor.Objects;
using NetworkMonitor.Objects.Repository;
using NetworkMonitor.Utils.Helpers;
using Newtonsoft.Json;
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

            _openSearchHelper = new OpenSearchHelper(_modelParams, embeddingGenerator);

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



        private void SaveIndexMaxTokens(string indexName, int padToTokens)
        {
            // Save in memory
            _indexMaxTokens[indexName] = padToTokens;
            // Persist to disk (simple JSON file per index)
            var configDir = Path.Combine(_dataDir, "index_config");
            Directory.CreateDirectory(configDir);
            var file = Path.Combine(configDir, $"{indexName}_padtokens.json");
            File.WriteAllText(file, JsonConvert.SerializeObject(new { padToTokens }));
        }

        private int? LoadIndexMaxTokens(string indexName)
        {
            // Try in-memory first
            if (_indexMaxTokens.TryGetValue(indexName, out var val))
                return val;
            // Try disk
            var configDir = Path.Combine(_dataDir, "index_config");
            var file = Path.Combine(configDir, $"{indexName}_padtokens.json");
            if (File.Exists(file))
            {
                var obj = JsonConvert.DeserializeObject<dynamic>(File.ReadAllText(file));
                int loaded = (int)obj.padToTokens;
                _indexMaxTokens[indexName] = loaded;
                return loaded;
            }
            return null;
        }

        public async Task<ResultObj> CreateIndexAsync(CreateIndexRequest createIndexRequest, int padToTokens)
        {
            var result = new ResultObj();
            result.Success = false;
            result.Message = "MessageAPI: CreateIndexAsync: ";

            // Sanity checks
            if (createIndexRequest == null)
            {
                result.Message += "Error: createIndexRequest is null.";
                return result;
            }

            if (EncryptHelper.IsBadKey(_encryptKey, createIndexRequest.AuthKey, createIndexRequest.AppID))
            {
                result.Success = false;
                result.Message = $" Error : Failed QueryIndexAsync bad AuthKey for AppID {createIndexRequest.AppID}";
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
                result.Message += "Error: JsonMapping  and JsonFile are null or empty.";
                return result;
            }

            try
            {
                string jsonContent = "";
                if (!string.IsNullOrEmpty(createIndexRequest.JsonFile))
                {
                    jsonContent = await File.ReadAllTextAsync(createIndexRequest.JsonFile);

                }
                else jsonContent = createIndexRequest.JsonMapping;
                if (string.IsNullOrEmpty(jsonContent))
                {
                    result.Message += "Error: Json is null or empty.";
                    return result;
                }
                // Dynamically deserialize based on index name
                List<object> items = null;
                if (createIndexRequest.IndexName == "securitybooks")
                {
                    var securityBooks = JsonConvert.DeserializeObject<List<SecurityBook>>(jsonContent);
                    if (securityBooks == null)
                    {
                        result.Message += "Error: Failed to deserialize JSON mapping for securitybooks.";
                        return result;
                    }
                    items = securityBooks.Cast<object>().ToList();
                }
                else
                {
                    var documents = JsonConvert.DeserializeObject<List<Document>>(jsonContent);
                    if (documents == null)
                    {
                        result.Message += "Error: Failed to deserialize JSON mapping for documents.";
                        return result;
                    }
                    items = documents.Cast<object>().ToList();
                }

                Console.WriteLine($"JSON deserialization succeeded. Proceeding with indexing using {padToTokens} tokens of the input.");

                var resultEn = await _openSearchHelper.EnsureIndexExistsAsync(indexName: createIndexRequest.IndexName, recreateIndex: createIndexRequest.RecreateIndex);
                if (!resultEn.Success) return resultEn;
                var resultIn = await _openSearchHelper.IndexDocumentsAsync(items, createIndexRequest.IndexName, padToTokens);
                createIndexRequest.Success = resultEn.Success && resultIn.Success;
                createIndexRequest.Message += resultEn.Message + resultIn.Message;

                await _rabbitRepo.PublishAsync<CreateIndexRequest>("createIndexResult" + createIndexRequest.AppID, createIndexRequest);
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

        // Overload: CreateIndexAsync that looks up padToTokens from index name
        public async Task<ResultObj> CreateIndexAsync(CreateIndexRequest createIndexRequest)
        {
            if (createIndexRequest == null || string.IsNullOrWhiteSpace(createIndexRequest.IndexName))
            {
                return new ResultObj { Success = false, Message = "Error: createIndexRequest or IndexName is null." };
            }

            // Try to load pad tokens for this index, fail if not found
            int? padToTokens = LoadIndexMaxTokens(createIndexRequest.IndexName);
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
        /// <summary>
        /// Reads a data directory, treats each subdirectory as an index name, and for each JSON file in each subdirectory,
        /// calculates the pad to token length for all files in the index, stores it, and then runs CreateIndexAsync for each file.
        /// </summary>
        /// <param name="createIndexRequest">A CreateIndexRequest object with parameters to use (AppID, AuthKey, RecreateIndex, etc). Set JsonFile and IndexName to empty.</param>
        /// <returns>A ResultObj summarizing the operation.</returns>
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

            // Skip the index_config directory if present
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

                // Instead of loading all data into memory, just keep track of the pad to token count as we go
                var tempTokenizer = new AutoTokenizer(_modelParams.EmbeddingModelDir,_maxTokenLengthCap);
                int padToTokens = _minTokenLengthCap;
                bool bailedEarly = false;
                foreach (var jsonFile in jsonFiles)
                {
                    if (bailedEarly) break;
                    var jsonContent = File.ReadAllText(jsonFile);
                    if (indexName == "securitybooks")
                    {
                        var securityBooks = JsonConvert.DeserializeObject<List<SecurityBook>>(jsonContent);
                        if (securityBooks != null)
                        {
                            foreach (var sb in securityBooks)
                            {
                                foreach (var text in new[] { sb.Input, sb.Output, sb.Summary })
                                {
                                    if (!string.IsNullOrWhiteSpace(text))
                                    {
                                        int tokenCount = tempTokenizer.CountTokens(text);
                                        if (tokenCount > padToTokens)
                                            padToTokens = tokenCount;
                                        if (padToTokens >= _maxTokenLengthCap)
                                        {
                                            bailedEarly = true;
                                            break;
                                        }
                                    }
                                }
                                if (bailedEarly) break;
                            }
                        }
                    }
                    else
                    {
                        var documents = JsonConvert.DeserializeObject<List<Document>>(jsonContent);
                        if (documents != null)
                        {
                            foreach (var d in documents)
                            {
                                foreach (var text in new[] { d.Input, d.Output })
                                {
                                    if (!string.IsNullOrWhiteSpace(text))
                                    {
                                        int tokenCount = tempTokenizer.CountTokens(text);
                                        if (tokenCount > padToTokens)
                                            padToTokens = tokenCount;
                                        if (padToTokens >= _maxTokenLengthCap)
                                        {
                                            bailedEarly = true;
                                            break;
                                        }
                                    }
                                }
                                if (bailedEarly) break;
                            }
                        }
                    }
                }

                // Cap at the value defined in EmbeddingGenerator
                padToTokens = Math.Min(padToTokens, _maxTokenLengthCap);

                // Store pad to token length for this index (if not already set)
                var loadedMax = LoadIndexMaxTokens(indexName);
                if (!loadedMax.HasValue)
                {
                    SaveIndexMaxTokens(indexName, padToTokens);
                }
                else
                {
                    padToTokens = loadedMax.Value;
                }

                // If there are any .json files, set RecreateIndex to true
                bool shouldRecreateIndex = jsonFiles.Length > 0;

                foreach (var jsonFile in jsonFiles)
                {
                    // Clone the request and set the index and file for this run
                    var req = new CreateIndexRequest
                    {
                        IndexName = indexName,
                        JsonFile = jsonFile,
                        AppID = createIndexRequest.AppID,
                        AuthKey = createIndexRequest.AuthKey,
                        RecreateIndex = shouldRecreateIndex,
                        JsonMapping = "", // always use file for this
                        MessageID = createIndexRequest.MessageID
                    };

                    // Pass the padToTokens for this index
                    var createResult = await CreateIndexAsync(req, padToTokens);
                    result.Message += $"Index '{indexName}', File '{Path.GetFileName(jsonFile)}': MaxTokens {padToTokens} : {createResult.Message}\n";
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
                        int? padToTokens = LoadIndexMaxTokens(queryIndexRequest.IndexName);
                        int useMaxTokens = padToTokens ?? _minTokenLengthCap;
                        var searchResponse = await _openSearchHelper.SearchDocumentsAsync(queryIndexRequest.QueryText, queryIndexRequest.IndexName, useMaxTokens);

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