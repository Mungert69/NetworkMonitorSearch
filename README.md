# NetworkMonitorSearch

Search and embedding service for NetworkMonitor. It generates embeddings,
interfaces with search backends, and exposes Rabbit-driven search workflows
for LLM and UI components.

## Entry points
- `Program.cs` starts the host.
- `Startup.cs` configures RabbitMQ, embedding generators, and services.

## Key folders
- `Services/` OpenSearch integration, embedding generation, and listeners.

## Run locally
```bash
dotnet restore
dotnet run --project NetworkMonitorSearch.csproj
```

## Related projects
- `NetworkMonitorLLM` consumes search results for RAG.
- `NetworkMonitorService` orchestrates user-facing search requests.
