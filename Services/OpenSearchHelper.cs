using OpenSearch.Client;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Net.Http;
using System.Net.Http.Headers;
using OpenSearch.Net;
using NetworkMonitor.Objects;
namespace NetworkMonitor.Search.Services;

public class Document
{
    public string Instruction { get; set; } = "";
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public List<float> Embedding { get; set; } = new();
}
public class OpenSearchHelper
{
    private readonly OpenSearchClient _client;
    private readonly EmbeddingGenerator _embeddingGenerator;
    private OSModelParams _modelParams;

    public OpenSearchHelper(OSModelParams modelParams)
    {
        _modelParams = modelParams;
        // Initialize OpenSearch client
        var settings = new ConnectionSettings(_modelParams.SearchUri)
            .DefaultIndex(_modelParams.DefaultIndex)
            .BasicAuthentication(_modelParams.User, _modelParams.Key)
            .ServerCertificateValidationCallback((o, certificate, chain, errors) => true);

        _client = new OpenSearchClient(settings);

        // Initialize the embedding generator
        _embeddingGenerator = new EmbeddingGenerator(_modelParams.BertModelDir);
    }

    // Method to generate embeddings for a document
    private List<float> GenerateEmbedding(string text)
    {
        return _embeddingGenerator.GenerateEmbedding(text);
    }

    // Method to load documents from JSON and index in OpenSearch
    public async Task<ResultObj> IndexDocumentsAsync(IEnumerable<Document> documents, string indexName = "")
    {
        var result = new ResultObj() { Message = " EnsureIndexExistsAsync : " };
        bool oneFail = false;
        try
        {
            if (indexName == "") indexName = _modelParams.DefaultIndex;
            foreach (var document in documents)
            {
                // Generate embedding only when needed
                if (document.Embedding == null || document.Embedding.Count == 0)
                {
                    document.Embedding = GenerateEmbedding(document.Input);
                }

                string documentId = ComputeSha256Hash(document.Input);

                // Check if the document already exists
                var existsResponse = await _client.DocumentExistsAsync(DocumentPath<Document>.Id(documentId), d => d.Index(indexName));

                if (existsResponse.Exists)
                {
                    result.Message += $"Document with ID {documentId} already exists. Skipping indexing.";
                    continue;  // Skip this document if it already exists
                }

                // Index the new document
                var indexResponse = await _client.IndexAsync(new
                {
                    input = document.Input,
                    output = document.Output,
                    embedding = document.Embedding
                }, i => i.Index(indexName).Id(documentId));

                if (!indexResponse.IsValid)
                {
                    oneFail = true;
                    result.Message += $"Failed to index document with ID {documentId}: {indexResponse.ServerError}";
                }
                else
                {
                    result.Message += $"Indexing document ID {documentId} with embedding: {string.Join(",", document.Embedding)}";
                }

            }
            result.Success = !oneFail;
        }
        catch (Exception e)
        {
            result.Success = false;
            result.Message += e.Message;
        }
        return result;
    }

    // Method to search for similar documents using precomputed embeddings
    public async Task<SearchResponseObj> SearchDocumentsAsync(string queryText, string indexName = "")
    {
        if (indexName == "") indexName = _modelParams.DefaultIndex;
        // Generate embedding for the query text
        var queryEmbedding = GenerateEmbedding(queryText);
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
                var createIndexResponse = await _client.LowLevel.Indices.CreateAsync<StringResponse>(indexName, PostData.String(@"
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
                            ""dimension"": " + _modelParams.BertModelVecDim + @",
                            ""method"": {
                                ""name"": ""hnsw"",
                                ""space_type"": ""l2"",
                                ""engine"": ""faiss""
                            }
                        }
                    }
                }
            }"));

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
                result.Success = false;
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