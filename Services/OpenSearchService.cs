using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NetworkMonitor.Objects;
using Newtonsoft.Json;

namespace NetworkMonitor.Search.Services
{
    public interface IOpenSearchService
    {
        Task<ResultObj> CreateIndexAsync(CreateIndexRequest createIndexRequest);
        Task<ResultObj> QueryIndexAsync(QueryIndexRequest queryIndexRequest);
    }

    public class OpenSearchService : IOpenSearchService
    {
        private readonly OpenSearchHelper _openSearchHelper;

        public OpenSearchService(string modelDir)
        {
            _openSearchHelper = new OpenSearchHelper(modelDir);
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

                // Ensure the index exists
                await _openSearchHelper.EnsureIndexExistsAsync(createIndexRequest.IndexName, createIndexRequest.RecreateIndex);

                // Index documents in OpenSearch
                await _openSearchHelper.IndexDocumentsAsync(documents);

                result.Success = true;
                result.Message += $"Index '{createIndexRequest.IndexName}' created successfully.";
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
            result.Success = false;
            result.Message = "MessageAPI: QueryIndexAsync: ";

            // Sanity checks
            if (queryIndexRequest == null)
            {
                result.Message += "Error: queryIndexRequest is null.";
                return result;
            }

            if (string.IsNullOrWhiteSpace(queryIndexRequest.IndexName))
            {
                result.Message += "Error: indexName is null or empty.";
                return result;
            }

            if (string.IsNullOrWhiteSpace(queryIndexRequest.QueryText))
            {
                result.Message += "Error: queryText is null or empty.";
                return result;
            }

            try
            {
                // Perform the search query
                await _openSearchHelper.SearchDocumentsAsync(queryIndexRequest.QueryText, queryIndexRequest.IndexName);

                result.Success = true;
                result.Message += $"Query executed successfully on index '{queryIndexRequest.IndexName}'.";
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