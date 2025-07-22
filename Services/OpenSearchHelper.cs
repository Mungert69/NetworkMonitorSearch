using OpenSearch.Client;
using Newtonsoft.Json;
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

namespace NetworkMonitor.Search.Services;


public class OpenSearchHelper
{
    private readonly OpenSearchClient _client;
    private IEmbeddingGenerator _embeddingGenerator;
    private OSModelParams _modelParams;
    private readonly IReadOnlyList<IIndexingStrategy> _strategies;

    public OpenSearchHelper(OSModelParams modelParams,
                              IEmbeddingGenerator embeddingGenerator,
                              params IIndexingStrategy[] strategies)
    {

        _strategies = strategies;

        _modelParams = modelParams;
        _embeddingGenerator = embeddingGenerator;
        // Initialize OpenSearch client
        var settings = new ConnectionSettings(_modelParams.SearchUri)
            .DefaultIndex(_modelParams.DefaultIndex)
            .BasicAuthentication(_modelParams.User, _modelParams.Key)
            .ServerCertificateValidationCallback((o, certificate, chain, errors) => true);

        _client = new OpenSearchClient(settings);
    }

    // Method to generate embeddings for a document (async)
    private async Task<List<float>> GenerateEmbeddingAsync(string text, int padToTokens)
    {
        return await _embeddingGenerator.GenerateEmbeddingAsync(text, padToTokens);
    }

    // Method to load documents or securitybooks from JSON and index in OpenSearch
    public async Task<ResultObj> IndexDocumentsAsync(IEnumerable<object> items,
                                                     int padToTokens)
    {
        var result = new ResultObj { Message = "IndexDocumentsAsync: " };
        bool failed = false;

        foreach (var item in items)
        {
            // Pick the first strategy that says it can handle this artefact
            var strategy = _strategies.FirstOrDefault(s => s.CanHandle(item));
            if (strategy is null)
            {
                result.Message += $"No strategy found for type {item.GetType().Name}. Skipping. ";
                failed = true;
                continue;
            }

            try
            {
                await strategy.EnsureEmbeddingsAsync(item, _embeddingGenerator, padToTokens);

                var id = strategy.ComputeId(item);
                var index = strategy.IndexName;
                var exists = await _client.DocumentExistsAsync<object>(id, idx => idx.Index(index));

                if (exists.Exists)
                {
                    result.Message += $"{index}/{id} already exists. Skipping. ";
                    continue;
                }

                var body = strategy.BuildIndexDocument(item);
                var resp = await _client.IndexAsync(body, i => i.Index(index).Id(id));

                if (!resp.IsValid)
                {
                    failed = true;
                    result.Message += $"Failed to index {index}/{id}: {resp.ServerError} ";
                }
                else
                {
                    result.Message += $"Indexed {index}/{id}. ";
                }
            }
            catch (Exception ex)
            {
                failed = true;
                result.Message += $"Error for {item.GetType().Name}: {ex.Message} ";
            }
        }

        result.Success = !failed;
        return result;
    }

    // Method to search for similar documents using precomputed embeddings
    public async Task<SearchResponseObj> SearchDocumentsAsync(string queryText, string indexName, int padToTokens)
    {
        var queryEmbedding = await GenerateEmbeddingAsync(queryText, padToTokens);
        var searchResponse = new SearchResponseObj();

        if (queryEmbedding.Count == 0)
        {
            throw new Exception("Failed to generate query embedding.");
        }

        // Create an HttpClient instance for sending the request
        var handler = new HttpClientHandler
        {
            ServerCertificateCustomValidationCallback = (message, cert, chain, sslPolicyErrors) => true
        };

        using var httpClient = new HttpClient(handler)
        {
            BaseAddress = _modelParams.SearchUri
        };
        httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Basic", Convert.ToBase64String(Encoding.ASCII.GetBytes($"{_modelParams.User}:{_modelParams.Key}")));

        // Construct the k-NN search request body
        var requestBody = new
        {
            size = 3,
            query = new
            {
                knn = new
                {
                    embedding = new
                    {
                        vector = queryEmbedding,
                        k = 3
                    }
                }
            }
        };

        // Serialize the request body to JSON using Newtonsoft.Json
        var jsonContent = JsonConvert.SerializeObject(requestBody);
        var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

        // Send the POST request to the specified index
        var response = await httpClient.PostAsync($"/{indexName}/_search", content);

        // Process the response
        if (response.IsSuccessStatusCode)
        {
            var responseBody = await response.Content.ReadAsStringAsync();
            Console.WriteLine("Search Results:");
            Console.WriteLine(responseBody);

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
        int padToTokens)
    {

        var queryEmbedding = await GenerateEmbeddingAsync(queryText, padToTokens);

        if (queryEmbedding.Count == 0)
            throw new Exception("Failed to generate query embedding.");

        // Default equal weights if none provided
        fieldWeights ??= new Dictionary<string, float>
        {
            ["input_embedding"] = 1f,
            ["output_embedding"] = 1f,
            ["summary_embedding"] = 1f
        };

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
        using var client = new HttpClient(new HttpClientHandler
        {
            ServerCertificateCustomValidationCallback = (m, c, ch, e) => true
        })
        {
            BaseAddress = _modelParams.SearchUri
        };
        client.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Basic", Convert.ToBase64String(
                Encoding.ASCII.GetBytes($"{_modelParams.User}:{_modelParams.Key}")));

        var response = await client.PostAsync($"/{indexName}/_search",
            new StringContent(json, Encoding.UTF8, "application/json"));

        if (!response.IsSuccessStatusCode)
            throw new Exception($"Search failed: {response.ReasonPhrase}");

        var content = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<SearchResponseObj>(content) ??
               new SearchResponseObj();
    }

    public async Task<ResultObj> EnsureIndexExistsAsync(string indexName = "", bool recreateIndex = false)
    {
        var result = new ResultObj() { Message = " EnsureIndexExistsAsync : " };
        try
        {

            if (indexName == "") indexName = _modelParams.DefaultIndex;
            if (recreateIndex)
            {
                var resultDel = await DeleteIndexAsync(indexName);
                result.Message += resultDel.Message;
                if (!resultDel.Success) return resultDel;
            }

            var existsResponse = await _client.Indices.ExistsAsync(indexName);
            if (!existsResponse.Exists)
            {
                // Use low-level client to create the index with the knn_vector mapping and knn enabled
                string mapping;
                if (indexName == "securitybooks")
                {
                    mapping = @"
                    {
                        ""settings"": {
                            ""index"": {
                                ""knn"": true
                            }
                        },
                        ""mappings"": {
                            ""properties"": {
                                ""input"": { ""type"": ""text"" },
                                ""output"": { ""type"": ""text"" },
                                ""summary"": { ""type"": ""text"" },
                                ""input_embedding"": { 
                                    ""type"": ""knn_vector"", 
                                    ""dimension"": " + _modelParams.EmbeddingModelVecDim + @",
                                    ""method"": {
                                        ""name"": ""hnsw"",
                                        ""space_type"": ""l2"",
                                        ""engine"": ""faiss""
                                    }
                                },
                                ""output_embedding"": { 
                                    ""type"": ""knn_vector"", 
                                    ""dimension"": " + _modelParams.EmbeddingModelVecDim + @",
                                    ""method"": {
                                        ""name"": ""hnsw"",
                                        ""space_type"": ""l2"",
                                        ""engine"": ""faiss""
                                    }
                                },
                                ""summary_embedding"": { 
                                    ""type"": ""knn_vector"", 
                                    ""dimension"": " + _modelParams.EmbeddingModelVecDim + @",
                                    ""method"": {
                                        ""name"": ""hnsw"",
                                        ""space_type"": ""l2"",
                                        ""engine"": ""faiss""
                                    }
                                }
                            }
                        }
                    }";
                }
                else
                {
                    mapping = @"
                    {
                        ""settings"": {
                            ""index"": {
                                ""knn"": true
                            }
                        },
                        ""mappings"": {
                            ""properties"": {
                                ""input"": { ""type"": ""text"" },
                                ""output"": { ""type"": ""text"" },
                                ""embedding"": { 
                                    ""type"": ""knn_vector"", 
                                    ""dimension"": " + _modelParams.EmbeddingModelVecDim + @",
                                    ""method"": {
                                        ""name"": ""hnsw"",
                                        ""space_type"": ""l2"",
                                        ""engine"": ""faiss""
                                    }
                                }
                            }
                        }
                    }";
                }

                var createIndexResponse = await _client.LowLevel.Indices.CreateAsync<StringResponse>(indexName, PostData.String(mapping));

                if (!createIndexResponse.Success)
                {
                    result.Message = $"Failed to create index: {createIndexResponse.DebugInformation}";
                }
                else result.Message += " Success : create index ";
                result.Success = createIndexResponse.Success;
            }
            else
            {
                result.Message += " Success : index already exists ";
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