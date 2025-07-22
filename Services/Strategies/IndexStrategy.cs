//  NetworkMonitor.Search.Strategies ------------------------------------------------
using System;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using NetworkMonitor.Objects;

namespace NetworkMonitor.Search.Services;

/// <summary>
/// Contract every indexable artefact must satisfy.
/// All methods are **type‑agnostic** for OpenSearchHelper,
/// but individual strategy classes know their concrete type.
/// </summary>
public interface IIndexingStrategy
{
    /// Logical name of the OpenSearch index this artefact lives in
    string IndexName { get; }

    /// Ensure that the artefact has every embedding it needs; generate
    /// anything that is missing.
    Task EnsureEmbeddingsAsync(object item,
                               IEmbeddingGenerator generator,
                               int padToTokens);

    /// A deterministic, collision‑resistant ID (e.g. SHA‑256 of some field)
    string ComputeId(object item);

    /// Shape the payload to be written into OpenSearch.
    object BuildIndexDocument(object item);

    /// Returns true if this strategy can handle <paramref name="item"/>.
    bool CanHandle(object item);
}

//  ------------------------------------------------------------------------------
//  Helper for computing deterministic IDs from strings
internal static class IdHelper
{
    internal static string Sha256(string text)
    {
        using var sha = SHA256.Create();
        var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(text));
        var sb = new StringBuilder(bytes.Length * 2);
        foreach (var b in bytes) sb.Append(b.ToString("x2"));
        return sb.ToString();
    }
}

//  ------------------------------------------------------------------------------
//  Strategy for plain 'Document'
public sealed class DocumentIndexingStrategy : IIndexingStrategy
{
    public string IndexName => "documents";

    public bool CanHandle(object item) => item is Document;

    public async Task EnsureEmbeddingsAsync(object item,
                                            IEmbeddingGenerator generator,
                                            int padToTokens)
    {
        var doc = (Document)item;
        if (doc.Embedding is { Count: > 0 }) return;

        doc.Embedding = await generator.GenerateEmbeddingAsync(doc.Output, padToTokens);
        if (doc.Embedding.Count == 0)
            throw new InvalidOperationException("Failed to generate embedding for Document.");
    }

    public string ComputeId(object item) =>
        IdHelper.Sha256(((Document)item).Output);

    public object BuildIndexDocument(object item)
    {
        var d = (Document)item;
        return new
        {
            input = d.Input,
            output = d.Output,
            embedding = d.Embedding
        };
    }
}

//  ------------------------------------------------------------------------------
//  Strategy for 'SecurityBook'
public sealed class SecurityBookIndexingStrategy : IIndexingStrategy
{
    public string IndexName => "securitybooks";

    public bool CanHandle(object item) => item is SecurityBook;

    public async Task EnsureEmbeddingsAsync(object item,
                                            IEmbeddingGenerator generator,
                                            int padToTokens)
    {
        var sb = (SecurityBook)item;

        async Task Ensure(Func<List<float>> get, Action<List<float>> set, string sourceText)
        {
            if (get() is { Count: > 0 }) return;

            var emb = await generator.GenerateEmbeddingAsync(sourceText, padToTokens);
            if (emb.Count == 0)
                throw new InvalidOperationException($"Failed to generate embedding for '{sourceText}'.");
            set(emb);
        }

        await Ensure(() => sb.InputEmbedding, e => sb.InputEmbedding = e, sb.Input);
        await Ensure(() => sb.OutputEmbedding, e => sb.OutputEmbedding = e, sb.Output);
        await Ensure(() => sb.SummaryEmbedding, e => sb.SummaryEmbedding = e, sb.Summary);
    }

    public string ComputeId(object item) =>
        IdHelper.Sha256(((SecurityBook)item).Output);

    public object BuildIndexDocument(object item)
    {
        var sb = (SecurityBook)item;
        return new
        {
            input = sb.Input,
            output = sb.Output,
            summary = sb.Summary,
            input_embedding = sb.InputEmbedding,
            output_embedding = sb.OutputEmbedding,
            summary_embedding = sb.SummaryEmbedding
        };
    }

}

public class Document
{
    public string Instruction { get; set; } = "";
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public List<float> Embedding { get; set; } = new();
}

public class SecurityBook
{
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public string Summary { get; set; } = "";
    public List<float> InputEmbedding { get; set; } = new();
    public List<float> OutputEmbedding { get; set; } = new();
    public List<float> SummaryEmbedding { get; set; } = new();
}