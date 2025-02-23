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

public class Document
{
    public string Instruction { get; set; } // This field is optional; add if needed
    public string Input { get; set; }       // The query text
    public string Output { get; set; }      // The response text
    public List<float> Embedding { get; set; }  // Precomputed embedding
}
public class OpenSearchHelper
{
    private readonly OpenSearchClient _client;
    private readonly EmbeddingGenerator _embeddingGenerator;

    public OpenSearchHelper(string modelDir)
    {
        // Initialize OpenSearch client
        var settings = new ConnectionSettings(new Uri("https://localhost:9200"))
            .DefaultIndex("documents")
            .BasicAuthentication("admin", "Ac.0462110")
            .ServerCertificateValidationCallback((o, certificate, chain, errors) => true);

        _client = new OpenSearchClient(settings);

        // Initialize the embedding generator
        _embeddingGenerator = new EmbeddingGenerator(modelDir);
    }

    // Method to generate embeddings for a document
    private List<float> GenerateEmbedding(string text)
    {
        return _embeddingGenerator.GenerateEmbedding(text);
    }

    // Method to load documents from JSON and index in OpenSearch
    public async Task IndexDocumentsAsync(IEnumerable<Document> documents, string indexName = "documents")
    {
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
                Console.WriteLine($"Document with ID {documentId} already exists. Skipping indexing.");
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
                Console.WriteLine($"Failed to index document with ID {documentId}: {indexResponse.ServerError}");
            }
            else
            {
                Console.WriteLine($"Indexing document ID {documentId} with embedding: {string.Join(",", document.Embedding)}");
            }
        }
    }

    // Method to search for similar documents using precomputed embeddings
    public async Task SearchDocumentsAsync(string queryText, string indexName = "documents")
    {
        // Generate embedding for the query text
        var queryEmbedding = GenerateEmbedding(queryText);

        if (queryEmbedding.Count == 0)
        {
            Console.WriteLine("Failed to generate query embedding.");
            return;
        }

        // Create an HttpClient instance for sending the request
        var handler = new HttpClientHandler
        {
            ServerCertificateCustomValidationCallback = (message, cert, chain, sslPolicyErrors) => true
        };

        using var httpClient = new HttpClient(handler)
        {
            BaseAddress = new Uri("https://localhost:9200")
        };
        httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Basic", Convert.ToBase64String(Encoding.ASCII.GetBytes("admin:Ac.0462110")));

        // Construct the k-NN search request body
        var requestBody = new
        {
            size = 3,
            query = new
            {
                knn = new
                {
                    embedding = new  // Ensure this matches the field name in the index mapping
                    {
                        vector = queryEmbedding,  // Ensure this is a valid 128-dimensional vector
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
            Console.WriteLine(responseBody);  // Deserialize and process this JSON as needed
        }
        else
        {
            Console.WriteLine($"Search failed: {response.ReasonPhrase}");
        }
    }

    public async Task EnsureIndexExistsAsync(string indexName = "documents", bool recreateIndex = false)
    {
        if (recreateIndex)
        {
            await DeleteIndexAsync(indexName);
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
                            ""dimension"": 128,
                            ""method"": {
                                ""name"": ""hnsw"",
                                ""space_type"": ""l2"",
                                ""engine"": ""nmslib""
                            }
                        }
                    }
                }
            }"));

            if (!createIndexResponse.Success)
            {
                Console.WriteLine($"Failed to create index: {createIndexResponse.Body}");
            }
        }
    }

    public async Task DeleteIndexAsync(string indexName = "documents")
    {
        var existsResponse = await _client.Indices.ExistsAsync(indexName);
        if (existsResponse.Exists)
        {
            var deleteResponse = await _client.Indices.DeleteAsync(indexName);
            if (deleteResponse.IsValid)
            {
                Console.WriteLine($"Index '{indexName}' deleted successfully.");
            }
            else
            {
                Console.WriteLine($"Failed to delete index '{indexName}': {deleteResponse.DebugInformation}");
            }
        }
        else
        {
            Console.WriteLine($"Index '{indexName}' does not exist. No action taken.");
        }
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