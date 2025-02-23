using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;
using NetworkMonitor.Search.Services;
using NetworkMonitor.Objects;

namespace NetworkMonitor.Search
{
    public class Program
    {
        private readonly IOpenSearchService _openSearchService;

        // Constructor for dependency injection
        public Program(IOpenSearchService openSearchService)
        {
            _openSearchService = openSearchService;
        }

        // Main entry point
        public static async Task Main(string[] args)
        {
            // Create the OpenSearchService instance
            var openSearchService = new OpenSearchService("stsb-bert-tiny-onnx");

            // Create the Program instance with dependency injection
            var program = new Program(openSearchService);

            // Run the program
            await program.RunAsync();
        }

        // Main logic of the program
        public async Task RunAsync()
        {
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

            // Create the index and index documents
            var createIndexRequest = new CreateIndexRequest
            {
                IndexName = "documents",
                JsonMapping = File.ReadAllText(filePath) // Use the JSON file content as the mapping
            };

            var createIndexResult = await _openSearchService.CreateIndexAsync(createIndexRequest);
            if (!createIndexResult.Success)
            {
                Console.WriteLine($"Failed to create index: {createIndexResult.Message}");
                return;
            }

            Console.WriteLine("Index created successfully. Proceeding with querying.");

            // Perform a search
            var queryIndexRequest = new QueryIndexRequest
            {
                IndexName = "documents",
                QueryText = "exchange database" // Replace with actual query text
            };

            var queryIndexResult = await _openSearchService.QueryIndexAsync(queryIndexRequest);
            if (!queryIndexResult.Success)
            {
                Console.WriteLine($"Failed to query index: {queryIndexResult.Message}");
                return;
            }

            Console.WriteLine("Query executed successfully.");
        }
    }
}