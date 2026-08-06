# Development

## Code map

| Location | Change when… |
|---|---|
| `Startup.cs` | Adding a service, changing lifecycle setup, or replacing the embedding decorator. |
| `Services/RabbitListener.cs` | Adding a Rabbit operation or routing it to a workflow. |
| `Services/OpenSearchService.cs` | Changing authorization, request orchestration, response publishing, or query caching. |
| `Services/OpenSearchHelper.cs` | Changing mappings, index persistence, KNN/hybrid queries, history storage, or low-level OpenSearch behavior. |
| `Services/Strategies/IndexStrategy.cs` | Supporting a new source/index shape or embedding fields. |
| `Services/*Embedding*.cs`, `AutoTokenizer.cs` | Provider/model, tokenization, or cache behavior. |
| `Services/Tests/` | Adding regression coverage for service behavior. |

`Services/Onnx.cs` is generated protocol code; do not hand-edit it unless regenerating from its source schema/process.

## Adding a new searchable index

1. Define or reuse the DTO in the shared model project.
2. Add an `IIndexingStrategy` implementation with index name, deserialization, mapping, stable ID/content hash, embedding generation, and hit mapping.
3. Register it in the `_strategies` array in `OpenSearchService`.
4. Add tests for hash behavior, source mapping, and expected vector fields.
5. Ensure configuration and callers use exactly the same index name.
6. Update [OpenSearch and indexing](OpenSearch-and-Indexing) and this wiki's [index](index).

## Tests

The project includes xUnit tests for embedding caching, OpenSearch metadata filtering, result formatting, history parsing, and strategy hash behavior. Run all tests with `dotnet test --nologo`. Tests should not require a live RabbitMQ or OpenSearch instance unless explicitly expanded to integration coverage.

## Change discipline

Preserve stable content hashes and IDs when changing indexing behavior unless a reindex/migration is intentionally planned. A hash defines incremental-update behavior; a changed ID produces a new OpenSearch document. Confirm vector dimension compatibility before deploying a model change, and update consumers before changing Rabbit payload or default response-exchange semantics.

For documentation upkeep, follow [AGENTS.md](AGENTS.md).
