using System;
using System.Collections;
using System.Reflection;
using Newtonsoft.Json;
using Xunit;

namespace NetworkMonitor.Search.Services;

public class OpenSearchHelperMetadataFiltersTests
{
    private static object InvokeBuildMetadataFilters(
        string? filterDocId = null,
        string? filterChunkId = null,
        string? filterSourceFile = null,
        string? filterSectionPath = null,
        int filterPageStart = 0,
        int filterPageEnd = 0,
        int filterChunkIndexMin = 0,
        int filterChunkIndexMax = 0)
    {
        var method = typeof(OpenSearchHelper).GetMethod(
            "BuildMetadataFilters",
            BindingFlags.NonPublic | BindingFlags.Static);

        Assert.NotNull(method);

        var result = method!.Invoke(null, new object?[]
        {
            filterDocId,
            filterChunkId,
            filterSourceFile,
            filterSectionPath,
            filterPageStart,
            filterPageEnd,
            filterChunkIndexMin,
            filterChunkIndexMax
        });

        Assert.NotNull(result);
        return result!;
    }

    [Fact]
    public void BuildMetadataFilters_WithExactFieldsAndRanges_CreatesExpectedClauses()
    {
        var filters = InvokeBuildMetadataFilters(
            filterDocId: "doc-abc",
            filterChunkId: "chunk-007",
            filterSourceFile: "book.json",
            filterSectionPath: "Part I",
            filterPageStart: 120,
            filterPageEnd: 140,
            filterChunkIndexMin: 30,
            filterChunkIndexMax: 40);

        var list = Assert.IsAssignableFrom<IEnumerable>(filters);
        var json = JsonConvert.SerializeObject(list);

        Assert.Contains("\"doc_id\"", json, StringComparison.Ordinal);
        Assert.Contains("\"doc_id.keyword\"", json, StringComparison.Ordinal);
        Assert.Contains("\"chunk_id\"", json, StringComparison.Ordinal);
        Assert.Contains("\"chunk_id.keyword\"", json, StringComparison.Ordinal);
        Assert.Contains("\"source_file\"", json, StringComparison.Ordinal);
        Assert.Contains("\"source_file.keyword\"", json, StringComparison.Ordinal);
        Assert.Contains("\"section_path\"", json, StringComparison.Ordinal);
        Assert.Contains("\"section_path.keyword\"", json, StringComparison.Ordinal);

        // Overlap semantics for page windows:
        // page_start <= filterPageEnd and page_end >= filterPageStart
        Assert.Contains("\"page_start\"", json, StringComparison.Ordinal);
        Assert.Contains("\"lte\":140", json, StringComparison.Ordinal);
        Assert.Contains("\"page_end\"", json, StringComparison.Ordinal);
        Assert.Contains("\"gte\":120", json, StringComparison.Ordinal);

        Assert.Contains("\"chunk_index\"", json, StringComparison.Ordinal);
        Assert.Contains("\"gte\":30", json, StringComparison.Ordinal);
        Assert.Contains("\"lte\":40", json, StringComparison.Ordinal);
    }

    [Fact]
    public void BuildMetadataFilters_WithNoInputs_ReturnsEmptyFilterSet()
    {
        var filters = InvokeBuildMetadataFilters();
        var list = Assert.IsAssignableFrom<IEnumerable>(filters);
        var json = JsonConvert.SerializeObject(list);

        Assert.Equal("[]", json);
    }
}
