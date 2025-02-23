using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;
using NetworkMonitor.Search.Services;
namespace NetworkMonitor.Search;
public class Program
{
    public static async Task Main(string[] args)
    {
        var openSearchHelper = new OpenSearchHelper("stsb-bert-tiny-onnx");

        // Load data from JSON file
        string filePath = Path.Combine(Directory.GetCurrentDirectory(), "input_data.json");
        var documents = JsonConvert.DeserializeObject<List<Document>>(File.ReadAllText(filePath));

        // Check for deserialization issues
        if (documents == null)
        {
            Console.WriteLine("Failed to deserialize JSON file. Please check the file format and field names.");
            return;
        }

        Console.WriteLine("JSON deserialization succeeded. Proceeding with indexing.");

        // Ensure the OpenSearch index exists
        await openSearchHelper.EnsureIndexExistsAsync(recreateIndex : true);

        // Index documents in OpenSearch
        await openSearchHelper.IndexDocumentsAsync(documents);

        // Perform a search
        string queryText = "exchange database"; // Replace with actual query text
        await openSearchHelper.SearchDocumentsAsync(queryText);
    }
}