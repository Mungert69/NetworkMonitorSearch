using System;
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
        private int _bertModelVecDim = 128;
        private readonly ILogger _logger;
        private readonly IRabbitRepo _rabbitRepo;

        public OpenSearchService(ILogger<OpenSearchService> logger, ISystemParamsHelper systemParamsHelper, IRabbitRepo rabbitRepo)
        {
            _logger = logger;
            _encryptKey = systemParamsHelper.GetSystemParams().EmailEncryptKey;
            string modelDir = systemParamsHelper.GetMLParams().BertModelDir;
            _bertModelVecDim = systemParamsHelper.GetMLParams().BertModelVecDim;
            _rabbitRepo = rabbitRepo;
            _openSearchHelper = new OpenSearchHelper(modelDir);
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

            if (string.IsNullOrWhiteSpace(createIndexRequest.JsonMapping))
            {
                result.Message += "Error: jsonMapping is null or empty.";
                return result;
            }

            try
            {
                // Deserialize the JSON mapping
                var documents = JsonConvert.DeserializeObject<List<Document>>(createIndexRequest.JsonMapping);

                // Check for deserialization issues
                if (documents == null)
                {
                    result.Message += "Error: Failed to deserialize JSON mapping. Please check the file format and field names.";
                    return result;
                }

                Console.WriteLine("JSON deserialization succeeded. Proceeding with indexing.");

                await _openSearchHelper.EnsureIndexExistsAsync(indexName: createIndexRequest.IndexName, recreateIndex: createIndexRequest.RecreateIndex, bertModelVecDim: _bertModelVecDim);

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
            result.Success=false;
            result.Message="MessageAPI: QueryIndexAsync: ";

            // Sanity checks
            if (queryIndexRequest == null)
            {
                result.Message += "Error: queryIndexRequest is null.";
                result.Success = false;
                return result;
            }

            if (EncryptHelper.IsBadKey(_encryptKey, queryIndexRequest.AuthKey, queryIndexRequest.AppID))
            {
                result.Success = false;
                result.Message = $" Error : Failed QueryIndexAsync bad AuthKey for AppID {queryIndexRequest.AppID}";
                _logger.LogError(result.Message);
                return result;
            }

            if (string.IsNullOrWhiteSpace(queryIndexRequest.IndexName))
            {
                result.Message += "Error: indexName is null or empty.";
                result.Success = false;
                return result;
            }

            if (string.IsNullOrWhiteSpace(queryIndexRequest.QueryText))
            {
                result.Message += "Error: queryText is null or empty.";
                result.Success = false;
                return result;
            }

            try
            {
                // Perform the search query
                var searchResponse = await _openSearchHelper.SearchDocumentsAsync(queryIndexRequest.QueryText, queryIndexRequest.IndexName);

                if (searchResponse == null)
                {
                    result.Message += "Error: Search response is null.";
                    result.Success = false;
                    return result;
                }

                // Extract input and output data from the search results
                var queryResults = new List<QueryResultObj>();
                foreach (var hit in searchResponse.Hits.HitsList)
                {
                    queryResults.Add(new QueryResultObj
                    {
                        Input = hit.Source.Input,
                        Output = hit.Source.Output
                    });
                }

                // Add the query results to the TResultObj
                queryIndexRequest.QueryResults = queryResults;
                queryIndexRequest.Success = true;
                queryIndexRequest.Message += $"Query executed successfully on index '{queryIndexRequest.IndexName}'.";
                await _rabbitRepo.PublishAsync<QueryIndexRequest>("queryIndexResult" + queryIndexRequest.AppID, queryIndexRequest);
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