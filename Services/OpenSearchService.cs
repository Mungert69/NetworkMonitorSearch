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
        Task<ResultObj> CreateIndexAsync(CreateIndexRequest createIndexRequest);
        Task<ResultObj> QueryIndexAsync(QueryIndexRequest queryIndexRequest);

        // New methods for snapshot and bulk index creation
        Task<ResultObj> CreateSnapshotAsync(string snapshotRepo, string snapshotName, string indices = "documents,securitybooks");
        Task<ResultObj> RestoreSnapshotAsync(string snapshotRepo, string snapshotName, string indices = "documents,securitybooks");
        Task<ResultObj> CreateIndicesFromDataDirAsync(CreateIndexRequest createIndexRequest);
    }

    public class OpenSearchService : IOpenSearchService
    {
        private readonly OpenSearchHelper _openSearchHelper;
        private readonly string _encryptKey;
        private OSModelParams _modelParams = new OSModelParams();
        private readonly ILogger _logger;
        private readonly IRabbitRepo _rabbitRepo;
        private readonly MemoryCache _cache = new MemoryCache(new MemoryCacheOptions());

        public OpenSearchService(ILogger<OpenSearchService> logger, ISystemParamsHelper systemParamsHelper, IRabbitRepo rabbitRepo)
        {
            _logger = logger;
            _encryptKey = systemParamsHelper.GetSystemParams().LLMEncryptKey;
            _modelParams.BertModelDir = systemParamsHelper.GetMLParams().BertModelDir;
            _modelParams.BertModelVecDim = systemParamsHelper.GetMLParams().BertModelVecDim;
            _modelParams.Key = systemParamsHelper.GetMLParams().OpenSearchKey;
            _modelParams.User = systemParamsHelper.GetMLParams().OpenSearchUser;
            _modelParams.Url = systemParamsHelper.GetMLParams().OpenSearchUrl;
            _modelParams.DefaultIndex = systemParamsHelper.GetMLParams().OpenSearchDefaultIndex;
            _rabbitRepo = rabbitRepo;
            _openSearchHelper = new OpenSearchHelper(_modelParams);
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

        public async Task<ResultObj> CreateIndexAsync(CreateIndexRequest createIndexRequest)
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

                Console.WriteLine("JSON deserialization succeeded. Proceeding with indexing.");

                var resultEn = await _openSearchHelper.EnsureIndexExistsAsync(indexName: createIndexRequest.IndexName, recreateIndex: createIndexRequest.RecreateIndex);
                if (!resultEn.Success) return resultEn;
                var resultIn = await _openSearchHelper.IndexDocumentsAsync(items, createIndexRequest.IndexName);
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
            if (appID != "nmap" && appID != "meta")
            {
                result.Message += $" Warning : not applying Rag for LLM type {appID} .";
                result.Success = false;
            }

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
                        var searchResponse = await _openSearchHelper.SearchDocumentsAsync(queryIndexRequest.QueryText, queryIndexRequest.IndexName);

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


        /// <summary>
        /// Reads a data directory, treats each subdirectory as an index name, and for each JSON file in each subdirectory,
        /// runs CreateIndexAsync for that file and index, using parameters from the provided CreateIndexRequest.
        /// </summary>
        /// <param name="createIndexRequest">A CreateIndexRequest object with parameters to use (AppID, AuthKey, RecreateIndex, etc). Set JsonFile and IndexName to empty.</param>
        /// <returns>A ResultObj summarizing the operation.</returns>
        public async Task<ResultObj> CreateIndicesFromDataDirAsync(CreateIndexRequest createIndexRequest)
        {
            var result = new ResultObj();
            result.Success = true;
            result.Message = $"Starting CreateIndicesFromDataDirAsync for {createIndexRequest.JsonFile}\n";

            string dataDir = createIndexRequest.JsonFile;
            if (string.IsNullOrWhiteSpace(dataDir) || !Directory.Exists(dataDir))
            {
                result.Success = false;
                result.Message += $"Error: Directory '{dataDir}' does not exist.";
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
                var jsonFiles = Directory.GetFiles(indexDir, "*.json");
                if (jsonFiles.Length == 0)
                {
                    result.Message += $"Index '{indexName}': No JSON files found, skipping.\n";
                    continue;
                }

                foreach (var jsonFile in jsonFiles)
                {
                    // Clone the request and set the index and file for this run
                    var req = new CreateIndexRequest
                    {
                        IndexName = indexName,
                        JsonFile = jsonFile,
                        AppID = createIndexRequest.AppID,
                        AuthKey = createIndexRequest.AuthKey,
                        RecreateIndex = createIndexRequest.RecreateIndex,
                        JsonMapping = "", // always use file for this
                        MessageID = createIndexRequest.MessageID
                    };

                    var createResult = await CreateIndexAsync(req);
                    result.Message += $"Index '{indexName}', File '{Path.GetFileName(jsonFile)}': {createResult.Message}\n";
                    if (!createResult.Success)
                        result.Success = false;
                }
            }

            return result;
        }
    }
}