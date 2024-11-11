using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;            // For JSON deserialization
using OpenSearch.Client;          // For OpenSearch integration

public class Program
{
    public static async Task Main(string[] args)
    {
        var openSearchHelper = new OpenSearchHelper();

        // Load data from JSON file with precomputed embeddings
        string filePath = Path.Combine(Directory.GetCurrentDirectory(), "output_with_embeddings.json");

        // Load data from JSON file with precomputed embeddings
        var documents = JsonConvert.DeserializeObject<List<Document>>(File.ReadAllText(filePath));

        // Check for deserialization issues
        if (documents == null)
        {
            Console.WriteLine("Failed to deserialize JSON file. Please check the file format and field names.");
        }
        else
        {
            Console.WriteLine("JSON deserialization succeeded. Proceeding with indexing.");
        }
        await openSearchHelper.EnsureIndexExistsAsync();
        // Index documents in OpenSearch
        await openSearchHelper.IndexDocumentsAsync(documents);

        string queryEmbeddingPath = Path.Combine(Directory.GetCurrentDirectory(), "query_embedding.json");
        await openSearchHelper.SearchDocumentsAsync(queryEmbeddingPath);
    }


}

