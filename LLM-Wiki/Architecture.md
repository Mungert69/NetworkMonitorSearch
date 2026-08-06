# Architecture

## Host lifecycle

`Program.CreateHostBuilder` loads required `appsettings.json` plus environment variables, then calls `Startup.ConfigureServices`. Startup registers RabbitMQ, configuration helpers, OpenSearch, the selected embedding generator, and a file repository rooted at `./state/networkmonitorsearch`.

`HostInitActions` initializes services in this order:

1. Connect and set up `IRabbitRepo`.
2. Initialize `IOpenSearchService`.
3. Set up `IRabbitListener` consumers.

On application stop, a cancellation token is signalled and both RabbitMQ components are shut down. The configured shutdown timeout is 30 seconds.

## Components

| Component | Responsibility |
|---|---|
| `RabbitListener` | Declares six consumers, validates requests through downstream services, and acknowledges through its shared listener base. |
| `OpenSearchService` | Implements request workflows, authorization, response publishing, result caching, index token metadata, and snapshots. |
| `OpenSearchHelper` | Creates mappings, indexes documents, runs vector/hybrid queries, and owns history/index persistence details. |
| `IIndexingStrategy` implementations | Deserialize a supported source type, choose embedding fields/mappings, calculate IDs/hashes, and map hits to results. |
| `IEmbeddingGenerator` | Generates vectors from local ONNX or the remote Novita API; the DI registration wraps it in `CachedEmbeddingGenerator`. |

## Data flow

For a document query, the listener forwards `QueryIndexRequest` to `OpenSearchService`. The service checks its auth key, loads the recorded token padding for the index, invokes `OpenSearchHelper`, maps hits using the relevant indexing strategy, formats a message, caches the result for the process lifetime, and publishes the amended request as the response.

Indexing follows the reverse path: JSON is deserialized by the strategy, an index mapping is ensured, stable content hashes are compared in incremental mode, missing vectors are generated, and documents are written to OpenSearch.

## Important design constraints

- Service configuration is resolved by shared `SystemParamsHelper`; no sample `appsettings.json` is tracked here.
- Search results are cached in memory without expiry, so restart the service to clear query-cache entries.
- The helper accepts the configured OpenSearch TLS certificate without validation. Treat the network path as trusted and avoid exposing it publicly.
- Changes to a request DTO or Rabbit naming behavior need coordinating changes in `NetworkMonitorLib` and callers.

Related: [RabbitMQ contract](RabbitMQ-Contract), [OpenSearch and indexing](OpenSearch-and-Indexing), [Operations](Operations).
