using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using NetworkMonitor.Objects;

namespace NetworkMonitor.Search.Services;

public static class QueryResultFormatter
{
    private static readonly HashSet<string> LocatorKeys = new(StringComparer.OrdinalIgnoreCase)
    {
        "doc_id",
        "chunk_id",
        "chunk_index",
        "chunk_count",
        "page_start",
        "page_end",
        "section_path",
        "prev_chunk_id",
        "next_chunk_id"
    };

    public static string Format(QueryIndexRequest request, IReadOnlyList<QueryResultObj>? queryResults, string? fallbackMessage = null)
    {
        var results = queryResults ?? Array.Empty<QueryResultObj>();
        if (results.Count == 0)
        {
            return string.IsNullOrWhiteSpace(fallbackMessage)
                ? "No results returned from index query."
                : fallbackMessage;
        }

        if (request.IncludeMetadata)
        {
            var shapedResults = results.Select(item =>
            {
                var metadata = item.Metadata ?? new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                bool hasLocator = metadata.Keys.Any(LocatorKeys.Contains);
                return new
                {
                    input = item.Input ?? string.Empty,
                    output = item.Output ?? string.Empty,
                    score = item.Score,
                    source_type = hasLocator ? "rag_chunk" : "rag_text_only",
                    metadata = metadata.Count > 0 ? metadata : null
                };
            }).ToList();

            bool anyLocator = shapedResults.Any(r => string.Equals(r.source_type, "rag_chunk", StringComparison.Ordinal));
            var payload = new
            {
                format = "query_result_v2",
                index_name = request.IndexName,
                vector_search_mode = request.VectorSearchMode.ToString(),
                top_k = request.TopK,
                result_count = shapedResults.Count,
                locator_support = anyLocator ? "partial_or_full" : "none",
                locator_note = anyLocator
                    ? "Locator metadata is present for some or all results. Use anchor_doc_id/anchor_chunk_id for follow-up expansion."
                    : "This index returned semantic text results without locator metadata. This is common for FAQ and MITRE content.",
                anchor = new
                {
                    anchor_doc_id = request.AnchorDocId,
                    anchor_chunk_id = request.AnchorChunkId,
                    neighbor_window = request.NeighborWindow
                },
                results = shapedResults
            };

            return JsonSerializer.Serialize(payload, new JsonSerializerOptions
            {
                WriteIndented = true
            });
        }

        var sb = new StringBuilder();
        sb.AppendLine("Index Query Results:");
        foreach (var item in results)
        {
            sb.AppendLine($"Input: {item.Input}");
            sb.AppendLine($"Output: {item.Output}");
            sb.AppendLine("---");
        }
        return sb.ToString();
    }
}
