using System;
using System.IO;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using NetworkMonitor.Objects;
using NetworkMonitor.Objects.Repository;
using NetworkMonitor.Utils.Helpers;
using Newtonsoft.Json;

namespace NetworkMonitor.Search.Services
{
    public interface IOpenSearchService
    {
        Task Init();
        Task<ResultObj> CreateIndexAsync(CreateIndexRequest createIndexRequest);
        Task<ResultObj> QueryIndexAsync(QueryIndexRequest queryIndexRequest);
    }

    public class OpenSearchService : IOpenSearchService
    {
        private readonly OpenSearchHelper _openSearchHelper;
        private readonly string _encryptKey;
        private OSModelParams _modelParams = new OSModelParams();
        private readonly ILogger _logger;
        private readonly IRabbitRepo _rabbitRepo;

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
                var documents = JsonConvert.DeserializeObject<List<Document>>(jsonContent);

                // Check for deserialization issues
                if (documents == null)
                {
                    result.Message += "Error: Failed to deserialize JSON mapping. Please check the file format and field names.";
                    return result;
                }

                Console.WriteLine("JSON deserialization succeeded. Proceeding with indexing.");

                await _openSearchHelper.EnsureIndexExistsAsync(indexName: createIndexRequest.IndexName, recreateIndex: createIndexRequest.RecreateIndex);

                await _openSearchHelper.IndexDocumentsAsync(documents);
                createIndexRequest.Success = true;
                createIndexRequest.Message += $"Index '{createIndexRequest.IndexName}' created successfully.";
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