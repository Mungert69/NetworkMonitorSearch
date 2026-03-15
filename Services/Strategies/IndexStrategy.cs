//  NetworkMonitor.Search.Strategies ------------------------------------------------
using System;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using NetworkMonitor.Objects;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace NetworkMonitor.Search.Services;

/// <summary>
/// Contract every indexable artefact must satisfy.
/// All methods are **type‑agnostic** for OpenSearchHelper,
/// but individual strategy classes know their concrete type.
/// </summary>
public interface IMultiVectorBook
{
    string Input { get; }
    string Output { get; }
    string Summary { get; }
    List<float> InputEmbedding { get; set; }
    List<float> OutputEmbedding { get; set; }
    List<float> SummaryEmbedding { get; set; }
}

public interface IHasExtensionData
{
    IDictionary<string, JToken> ExtensionData { get; }
}


public interface IIndexingStrategy
{
    string IndexName { get; }
    string GetVectorField(VectorSearchMode mode);
    IReadOnlyDictionary<string, float> GetDefaultFieldWeights();
    string GetIndexMapping(int vectorDimension);
    List<object> Deserialize(string json);
    Task EnsureEmbeddingsAsync(object item, IEmbeddingGenerator generator, int padToTokens);
    string ComputeId(object item);
    object BuildIndexDocument(object item);
    string ComputeContentHash(object item);
    bool TryHydrateFromDocument(object item, JObject source);
    bool CanHandle(object item);
    bool CanHandle(string indexName);

    // Token estimation
    IEnumerable<string> GetFields(object item);
    (int padToTokens, int actualMax) EstimatePadding(IEnumerable<string> jsonFiles, string embeddingModelDir, int maxCap, int minCap);

    // New: Map a search hit to a QueryResultObj
    QueryResultObj MapSearchHitToResult(NetworkMonitor.Objects.Hit hit);
}

/// <summary>
/// Generic base class for index strategies.
/// </summary>
public abstract class IndexingStrategyBase<T> : IIndexingStrategy where T : class, new()
{
    public abstract string IndexName { get; }
    public abstract string GetVectorField(VectorSearchMode mode);
    public abstract IReadOnlyDictionary<string, float> GetDefaultFieldWeights();
    public abstract string GetIndexMapping(int vectorDimension);

    public virtual List<object> Deserialize(string json)
    {
        var list = JsonConvert.DeserializeObject<List<T>>(json);
        return list?.Cast<object>().ToList() ?? new List<object>();
    }

    public abstract Task EnsureEmbeddingsAsync(object item, IEmbeddingGenerator generator, int padToTokens);
    public abstract string ComputeId(object item);
    public abstract object BuildIndexDocument(object item);
    public virtual string ComputeContentHash(object item)
    {
        var fields = GetFields(item) ?? Enumerable.Empty<string>();
        var normalized = fields.Select(f => f ?? string.Empty);
        var payload = string.Join("|", normalized);
        return IdHelper.Sha256(payload);
    }
    public virtual bool TryHydrateFromDocument(object item, JObject source) => false;

    public virtual bool CanHandle(object item) => item is T;
    public virtual bool CanHandle(string indexName) => indexName.Equals(IndexName, StringComparison.OrdinalIgnoreCase);

    // Token estimation logic
    public abstract IEnumerable<string> GetFields(object item);

    public virtual (int padToTokens, int actualMax) EstimatePadding(IEnumerable<string> jsonFiles, string modelDir, int maxCap, int minCap)
    {
        var tokenizer = new AutoTokenizer(modelDir, maxCap);
        int pad = minCap;
        int maxSeen = minCap;

        foreach (var file in jsonFiles)
        {
            var items = JsonConvert.DeserializeObject<List<T>>(System.IO.File.ReadAllText(file)) ?? new();
            foreach (var item in items)
            {
                foreach (var text in GetFields(item))
                {
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        int tokens = tokenizer.CountTokens(text);
                        maxSeen = Math.Max(maxSeen, tokens);
                        pad = Math.Max(pad, tokens);
                        if (pad >= maxCap) return (pad, maxSeen);
                    }
                }
            }
        }
        return (pad, maxSeen);
    }

    // Default implementation: for indices with input/output fields
    public virtual QueryResultObj MapSearchHitToResult(NetworkMonitor.Objects.Hit hit)
    {
        var metadata = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        if (!string.IsNullOrWhiteSpace(hit.Index))
            metadata["index"] = hit.Index;

        if (hit.Source?.ExtensionData != null)
        {
            foreach (var kvp in hit.Source.ExtensionData)
            {
                if (kvp.Key.EndsWith("_embedding", StringComparison.OrdinalIgnoreCase)) continue;
                if (kvp.Key.Equals("content", StringComparison.OrdinalIgnoreCase)) continue;
                var value = kvp.Value;
                metadata[kvp.Key] = value?.ToString() ?? string.Empty;
            }
        }

        if (!string.IsNullOrWhiteSpace(hit.Source?.Input))
            metadata.TryAdd("title", hit.Source.Input);

        if (!metadata.ContainsKey("summary") && hit.Source?.Output != null)
            metadata["summary"] = hit.Source.Output;

        return new QueryResultObj
        {
            Input = hit.Source?.Input ?? string.Empty,
            Output = hit.Source?.Output ?? string.Empty,
            Score = hit.Score,
            Metadata = metadata
        };
    }
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
//  Strategy for plain 'Documents'
public sealed class DocumentIndexingStrategy : IndexingStrategyBase<Document>
{
    public override string IndexName => "documents";
    public string ContentVectorFieldName => "output_embedding";
    public string QuestionVectorFieldName => "input_embedding";

    public override string GetVectorField(VectorSearchMode mode) => mode switch
    {
        VectorSearchMode.question => QuestionVectorFieldName,
        _ => ContentVectorFieldName
    };
    public override IReadOnlyDictionary<string, float> GetDefaultFieldWeights() =>
        new Dictionary<string, float>
        {
            [QuestionVectorFieldName] = 1f,
            [ContentVectorFieldName] = 1f
        };

    public override IEnumerable<string> GetFields(object item)
    {
        if (item is Document doc)
            return new[] { doc.Input, doc.Output };
        return Enumerable.Empty<string>();
    }

    public override async Task EnsureEmbeddingsAsync(object item, IEmbeddingGenerator generator, int padToTokens)
    {
        var sb = (Document)item;

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
    }

    public override string ComputeId(object item) =>
        IdHelper.Sha256(((Document)item).Output);

    public override object BuildIndexDocument(object item)
    {
        var sb = (Document)item;
        return new
        {
            input = sb.Input,
            output = sb.Output,
            input_embedding = sb.InputEmbedding,
            output_embedding = sb.OutputEmbedding
        };
    }

    public override bool TryHydrateFromDocument(object item, JObject source)
    {
        if (item is not Document doc || source is null) return false;

        var storedInput = source["input"]?.Value<string>() ?? string.Empty;
        var storedOutput = source["output"]?.Value<string>() ?? string.Empty;

        if (!string.Equals(doc.Input ?? string.Empty, storedInput, StringComparison.Ordinal) ||
            !string.Equals(doc.Output ?? string.Empty, storedOutput, StringComparison.Ordinal))
        {
            return false;
        }

        doc.InputEmbedding = source["input_embedding"]?.ToObject<List<float>>() ?? new List<float>();
        doc.OutputEmbedding = source["output_embedding"]?.ToObject<List<float>>() ?? new List<float>();

        return doc.InputEmbedding.Count > 0 && doc.OutputEmbedding.Count > 0;
    }

    public override string GetIndexMapping(int dim) => $@"
{{
  ""settings"": {{ ""index"": {{ ""knn"": true }} }},
  ""mappings"": {{
    ""properties"": {{
      ""input""  : {{ ""type"": ""text"" }},
      ""output"" : {{ ""type"": ""text"" }},
      ""input_embedding"" :  {{ ""type"": ""knn_vector"", ""dimension"": {dim},
                               ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }},
      ""output_embedding"" : {{ ""type"": ""knn_vector"", ""dimension"": {dim},
                               ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }}
    }}
  }}
}}";
}

public sealed class MitreIndexingStrategy : IndexingStrategyBase<Mitre>
{
    public override string IndexName => "mitre";
    public string ContentVectorFieldName => "embedding";
    public override string GetVectorField(VectorSearchMode mode) => ContentVectorFieldName;
    public override IReadOnlyDictionary<string, float> GetDefaultFieldWeights() =>
        new Dictionary<string, float> { [ContentVectorFieldName] = 1f };

    public override IEnumerable<string> GetFields(object item)
    {
        if (item is Mitre doc)
            return new[] { doc.Input, doc.Output };
        return Enumerable.Empty<string>();
    }

    public override async Task EnsureEmbeddingsAsync(object item, IEmbeddingGenerator generator, int padToTokens)
    {
        var doc = (Mitre)item;
        if (doc.Embedding is { Count: > 0 }) return;

        doc.Embedding = await generator.GenerateEmbeddingAsync(doc.Output, padToTokens);
        if (doc.Embedding.Count == 0)
            throw new InvalidOperationException("Failed to generate embedding for Document.");
    }

    public override string ComputeId(object item) =>
        IdHelper.Sha256(((Mitre)item).Output);

    public override object BuildIndexDocument(object item)
    {
        var d = (Mitre)item;
        return new
        {
            input = d.Input,
            output = d.Output,
            embedding = d.Embedding
        };
    }

    public override string ComputeContentHash(object item)
    {
        var mitre = (Mitre)item;
        return IdHelper.Sha256(mitre.Output ?? string.Empty);
    }

    public override bool TryHydrateFromDocument(object item, JObject source)
    {
        if (item is not Mitre doc || source is null) return false;

        var storedOutput = source["output"]?.Value<string>() ?? string.Empty;
        if (!string.Equals(doc.Output ?? string.Empty, storedOutput, StringComparison.Ordinal))
            return false;

        doc.Embedding = source["embedding"]?.ToObject<List<float>>() ?? new List<float>();
        return doc.Embedding.Count > 0;
    }

    public override string GetIndexMapping(int dim) => $@"
{{
  ""settings"": {{ ""index"": {{ ""knn"": true }} }},
  ""mappings"": {{
    ""properties"": {{
      ""input""     : {{ ""type"": ""text"" }},
      ""output""    : {{ ""type"": ""text"" }},
      ""embedding"" : {{
        ""type""  : ""knn_vector"",
        ""dimension"" : {dim},
        ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }}
      }}
    }}
  }}
}}";
}
public abstract class MultiVectorBookIndexingStrategyBase<T> : IndexingStrategyBase<T>
    where T : class, IMultiVectorBook, new()
{
    private const string AltQuestion1Field = "alt_question_1";
    private const string AltQuestion2Field = "alt_question_2";
    private const string AltQuestion3Field = "alt_question_3";
    private const string AltQuestion1EmbeddingField = "alt_question_1_embedding";
    private const string AltQuestion2EmbeddingField = "alt_question_2_embedding";
    private const string AltQuestion3EmbeddingField = "alt_question_3_embedding";

    private readonly bool _enableAltQuestionFields;

    protected MultiVectorBookIndexingStrategyBase(bool enableAltQuestionFields = false)
    {
        _enableAltQuestionFields = enableAltQuestionFields;
    }

    public string ContentVectorFieldName => "output_embedding";
    public string QuestionVectorFieldName => "input_embedding";
    public string SummaryVectorFieldName => "summary_embedding";

    public override string GetVectorField(VectorSearchMode mode) => mode switch
    {
        VectorSearchMode.question => QuestionVectorFieldName,
        VectorSearchMode.summary => SummaryVectorFieldName,
        _ => ContentVectorFieldName
    };

    public override IReadOnlyDictionary<string, float> GetDefaultFieldWeights() =>
        _enableAltQuestionFields
            ? new Dictionary<string, float>
            {
                [QuestionVectorFieldName] = 1f,
                [AltQuestion1EmbeddingField] = 0.9f,
                [AltQuestion2EmbeddingField] = 0.8f,
                [AltQuestion3EmbeddingField] = 0.7f,
                [ContentVectorFieldName] = 1f,
                [SummaryVectorFieldName] = 1f
            }
            : new Dictionary<string, float>
        {
            [QuestionVectorFieldName] = 1f,
            [ContentVectorFieldName] = 1f,
            [SummaryVectorFieldName] = 1f
        };

    private static string? GetExtensionString(T book, string key)
    {
        if (book is not IHasExtensionData ext || ext.ExtensionData == null)
            return null;

        if (!ext.ExtensionData.TryGetValue(key, out var token) || token == null)
            return null;

        var value = token.Type == JTokenType.String ? token.Value<string>() : token.ToString();
        if (string.IsNullOrWhiteSpace(value))
            return null;
        return value.Trim();
    }

    public override IEnumerable<string> GetFields(object item)
    {
        var book = (T)item; // Strongly typed
        var fields = new List<string> { book.Input, book.Output, book.Summary };

        if (book is IHasExtensionData ext && ext.ExtensionData != null)
        {
            foreach (var kvp in ext.ExtensionData)
            {
                if (kvp.Key.EndsWith("_embedding", StringComparison.OrdinalIgnoreCase)) continue;
                if (kvp.Key.Equals("input", StringComparison.OrdinalIgnoreCase)) continue;
                if (kvp.Key.Equals("output", StringComparison.OrdinalIgnoreCase)) continue;
                if (kvp.Key.Equals("summary", StringComparison.OrdinalIgnoreCase)) continue;
                fields.Add($"{kvp.Key}={kvp.Value?.ToString() ?? string.Empty}");
            }
        }

        return fields;
    }

    public override async Task EnsureEmbeddingsAsync(object item, IEmbeddingGenerator generator, int padToTokens)
    {
        var book = (T)item;

        async Task Ensure(Func<List<float>> get, Action<List<float>> set, string sourceText)
        {
            if (get() is { Count: > 0 }) return;
            var emb = await generator.GenerateEmbeddingAsync(sourceText, padToTokens);
            if (emb.Count == 0) throw new InvalidOperationException($"Failed to generate embedding for '{sourceText}'.");
            set(emb);
        }

        await Ensure(() => book.InputEmbedding, e => book.InputEmbedding = e, book.Input);
        await Ensure(() => book.OutputEmbedding, e => book.OutputEmbedding = e, book.Output);
        await Ensure(() => book.SummaryEmbedding, e => book.SummaryEmbedding = e, book.Summary);

        if (_enableAltQuestionFields && book is IHasExtensionData ext && ext.ExtensionData != null)
        {
            async Task EnsureAltEmbedding(string fieldName, string embeddingFieldName)
            {
                var altText = GetExtensionString(book, fieldName);
                if (string.IsNullOrWhiteSpace(altText))
                {
                    ext.ExtensionData.Remove(embeddingFieldName);
                    return;
                }

                if (ext.ExtensionData.TryGetValue(embeddingFieldName, out var existingToken) &&
                    existingToken is JArray existingArray &&
                    existingArray.Count > 0)
                {
                    return;
                }

                var emb = await generator.GenerateEmbeddingAsync(altText, padToTokens);
                if (emb.Count == 0)
                    return;
                ext.ExtensionData[embeddingFieldName] = JArray.FromObject(emb);
            }

            await EnsureAltEmbedding(AltQuestion1Field, AltQuestion1EmbeddingField);
            await EnsureAltEmbedding(AltQuestion2Field, AltQuestion2EmbeddingField);
            await EnsureAltEmbedding(AltQuestion3Field, AltQuestion3EmbeddingField);
        }
    }

    public override string ComputeId(object item) => IdHelper.Sha256(((T)item).Output);

    public override object BuildIndexDocument(object item)
    {
        var book = (T)item;
        var doc = JObject.FromObject(new
        {
            input = book.Input,
            output = book.Output,
            summary = book.Summary,
            input_embedding = book.InputEmbedding,
            output_embedding = book.OutputEmbedding,
            summary_embedding = book.SummaryEmbedding
        });

        if (book is IHasExtensionData ext && ext.ExtensionData != null)
        {
            foreach (var kvp in ext.ExtensionData)
            {
                if (kvp.Key.EndsWith("_embedding", StringComparison.OrdinalIgnoreCase))
                {
                    if (!_enableAltQuestionFields ||
                        !(kvp.Key.Equals(AltQuestion1EmbeddingField, StringComparison.OrdinalIgnoreCase) ||
                          kvp.Key.Equals(AltQuestion2EmbeddingField, StringComparison.OrdinalIgnoreCase) ||
                          kvp.Key.Equals(AltQuestion3EmbeddingField, StringComparison.OrdinalIgnoreCase)))
                    {
                        continue;
                    }
                }
                if (kvp.Key.Equals("input", StringComparison.OrdinalIgnoreCase)) continue;
                if (kvp.Key.Equals("output", StringComparison.OrdinalIgnoreCase)) continue;
                if (kvp.Key.Equals("summary", StringComparison.OrdinalIgnoreCase)) continue;
                if (kvp.Value is null) continue;
                doc[kvp.Key] = kvp.Value.DeepClone();
            }
        }

        return doc;
    }

    public override bool TryHydrateFromDocument(object item, JObject source)
    {
        if (item is not T book || source is null) return false;

        var storedInput = source["input"]?.Value<string>() ?? string.Empty;
        var storedOutput = source["output"]?.Value<string>() ?? string.Empty;
        var storedSummary = source["summary"]?.Value<string>() ?? string.Empty;

        if (!string.Equals(book.Input ?? string.Empty, storedInput, StringComparison.Ordinal) ||
            !string.Equals(book.Output ?? string.Empty, storedOutput, StringComparison.Ordinal) ||
            !string.Equals(book.Summary ?? string.Empty, storedSummary, StringComparison.Ordinal))
        {
            return false;
        }

        book.InputEmbedding = source["input_embedding"]?.ToObject<List<float>>() ?? new List<float>();
        book.OutputEmbedding = source["output_embedding"]?.ToObject<List<float>>() ?? new List<float>();
        book.SummaryEmbedding = source["summary_embedding"]?.ToObject<List<float>>() ?? new List<float>();

        return book.InputEmbedding.Count > 0 &&
               book.OutputEmbedding.Count > 0 &&
               book.SummaryEmbedding.Count > 0;
    }

    public override string GetIndexMapping(int dim)
    {
        var altFields = _enableAltQuestionFields
            ? $@",
      ""alt_question_1"": {{ ""type"": ""text"" }},
      ""alt_question_2"": {{ ""type"": ""text"" }},
      ""alt_question_3"": {{ ""type"": ""text"" }},
      ""alt_question_1_embedding"" : {{ ""type"": ""knn_vector"", ""dimension"": {dim},
                                       ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }},
      ""alt_question_2_embedding"" : {{ ""type"": ""knn_vector"", ""dimension"": {dim},
                                       ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }},
      ""alt_question_3_embedding"" : {{ ""type"": ""knn_vector"", ""dimension"": {dim},
                                       ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }}"
            : "";

        return $@"
{{
  ""settings"": {{ ""index"": {{ ""knn"": true }} }},
  ""mappings"": {{
    ""properties"": {{
      ""input""  : {{ ""type"": ""text"" }},
      ""output"" : {{ ""type"": ""text"" }},
      ""summary"": {{ ""type"": ""text"" }},
      ""input_embedding"" :  {{ ""type"": ""knn_vector"", ""dimension"": {dim},
                               ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }},
      ""output_embedding"" : {{ ""type"": ""knn_vector"", ""dimension"": {dim},
                               ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }},
      ""summary_embedding"" : {{ ""type"": ""knn_vector"", ""dimension"": {dim},
                                ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }}{altFields}
    }}
  }}
}}";
    }
}

public sealed class SecurityBookIndexingStrategy : MultiVectorBookIndexingStrategyBase<SecurityBook>
{
    public SecurityBookIndexingStrategy(bool enableAltQuestionFields = false)
        : base(enableAltQuestionFields)
    {
    }

    public override string IndexName => "securitybooks";
}

public sealed class QuantumBookIndexingStrategy : MultiVectorBookIndexingStrategyBase<QuantumBook>
{
    public override string IndexName => "quantumbooks";
}

public class Document
{
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public List<float> InputEmbedding { get; set; } = new();
    public List<float> OutputEmbedding { get; set; } = new();
}

public class Mitre
{
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public List<float> Embedding { get; set; } = new();
}
public class SecurityBook : IMultiVectorBook, IHasExtensionData
{
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public string Summary { get; set; } = "";
    public List<float> InputEmbedding { get; set; } = new();
    public List<float> OutputEmbedding { get; set; } = new();
    public List<float> SummaryEmbedding { get; set; } = new();
    [JsonExtensionData]
    public IDictionary<string, JToken> ExtensionData { get; set; } = new Dictionary<string, JToken>(StringComparer.OrdinalIgnoreCase);
}

public class QuantumBook : IMultiVectorBook, IHasExtensionData
{
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public string Summary { get; set; } = "";
    public List<float> InputEmbedding { get; set; } = new();
    public List<float> OutputEmbedding { get; set; } = new();
    public List<float> SummaryEmbedding { get; set; } = new();
    [JsonExtensionData]
    public IDictionary<string, JToken> ExtensionData { get; set; } = new Dictionary<string, JToken>(StringComparer.OrdinalIgnoreCase);
}

public sealed class BlogIndexingStrategy : IndexingStrategyBase<BlogIndexDocument>
{
    private const string TitleVectorFieldName = "title_embedding";
    private const string ContentVectorFieldName = "content_embedding";

    public override string IndexName => "blogs";

    public override string GetVectorField(VectorSearchMode mode) =>
        mode switch
        {
            VectorSearchMode.question => TitleVectorFieldName,
            _ => ContentVectorFieldName
        };

    public override IReadOnlyDictionary<string, float> GetDefaultFieldWeights() =>
        new Dictionary<string, float>
        {
            [TitleVectorFieldName] = 0.8f,
            [ContentVectorFieldName] = 1f
        };

    public override IEnumerable<string> GetFields(object item)
    {
        if (item is BlogIndexDocument blog)
        {
            return new[]
            {
                blog.Title,
                blog.Summary,
                blog.Content
            };
        }

        return Array.Empty<string>();
    }

    public override async Task EnsureEmbeddingsAsync(object item, IEmbeddingGenerator generator, int padToTokens)
    {
        if (item is not BlogIndexDocument blog)
        {
            throw new InvalidOperationException("BlogIndexingStrategy encountered unexpected item type.");
        }

        async Task EnsureAsync(Func<List<float>> get, Action<List<float>> set, string sourceText)
        {
            if (get() is { Count: > 0 }) return;
            if (string.IsNullOrWhiteSpace(sourceText))
            {
                set(new List<float>());
                return;
            }

            var embedding = await generator.GenerateEmbeddingAsync(sourceText, padToTokens);
            if (embedding.Count == 0)
            {
                throw new InvalidOperationException("Failed to generate embedding for blog content.");
            }
            set(embedding);
        }

        await EnsureAsync(() => blog.TitleEmbedding, e => blog.TitleEmbedding = e, blog.Title);
        var contentSource = !string.IsNullOrWhiteSpace(blog.Content) ? blog.Content : blog.Summary;
        await EnsureAsync(() => blog.ContentEmbedding, e => blog.ContentEmbedding = e, contentSource ?? blog.Title);
    }

    public override string ComputeId(object item)
    {
        var blog = (BlogIndexDocument)item;
        if (!string.IsNullOrWhiteSpace(blog.Slug))
        {
            return blog.Slug.ToLowerInvariant();
        }

        return IdHelper.Sha256($"{blog.Title}:{blog.Url}");
    }

    public override string ComputeContentHash(object item)
    {
        if (item is not BlogIndexDocument blog)
        {
            return base.ComputeContentHash(item);
        }

        var contentSource = !string.IsNullOrWhiteSpace(blog.Content)
            ? blog.Content
            : blog.Summary;

        var builder = new StringBuilder();
        builder.Append(blog.Title ?? string.Empty)
               .Append('|')
               .Append(contentSource ?? string.Empty);

        return IdHelper.Sha256(builder.ToString());
    }

    public override object BuildIndexDocument(object item)
    {
        var blog = (BlogIndexDocument)item;
        return new
        {
            title = blog.Title,
            slug = blog.Slug,
            summary = blog.Summary,
            content = blog.Content,
            categories = blog.Categories ?? new List<string>(),
            url = blog.Url,
            image = blog.Image,
            author = blog.Author,
            published_at = blog.PublishedAt,
            title_embedding = blog.TitleEmbedding ?? new List<float>(),
            content_embedding = blog.ContentEmbedding ?? new List<float>()
        };
    }

    public override string GetIndexMapping(int dim) => $@"
{{
  ""settings"": {{
    ""index"": {{
      ""knn"": true
    }}
  }},
  ""mappings"": {{
    ""properties"": {{
      ""title"": {{ ""type"": ""text"" }},
      ""slug"": {{ ""type"": ""keyword"" }},
      ""summary"": {{ ""type"": ""text"" }},
      ""content"": {{ ""type"": ""text"" }},
      ""categories"": {{ ""type"": ""keyword"" }},
      ""url"": {{ ""type"": ""keyword"" }},
      ""image"": {{ ""type"": ""keyword"" }},
      ""author"": {{ ""type"": ""keyword"" }},
      ""published_at"": {{ ""type"": ""date"", ""ignore_malformed"": true }},
      ""title_embedding"": {{
        ""type"": ""knn_vector"",
        ""dimension"": {dim},
        ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }}
      }},
      ""content_embedding"": {{
        ""type"": ""knn_vector"",
        ""dimension"": {dim},
        ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }}
      }}
    }}
  }}
}}";

    // Blog-specific mapping for search results
    public override QueryResultObj MapSearchHitToResult(NetworkMonitor.Objects.Hit hit)
    {
        var metadata = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        if (!string.IsNullOrWhiteSpace(hit.Index))
            metadata["index"] = hit.Index;

        var ext = hit.Source?.ExtensionData;
        var input = ext != null && ext.TryGetValue("title", out var titleToken)
            ? titleToken?.ToString() ?? string.Empty
            : string.Empty;
        var output = ext != null && ext.TryGetValue("summary", out var summaryToken)
            ? summaryToken?.ToString() ?? string.Empty
            : string.Empty;

        // Optionally, fallback to content if summary is empty
        if (string.IsNullOrWhiteSpace(output) && ext != null && ext.TryGetValue("content", out var contentToken))
            output = contentToken?.ToString() ?? string.Empty;

        // Add all other fields to metadata
        if (ext != null)
        {
            foreach (var kvp in ext)
            {
                if (kvp.Key.EndsWith("_embedding", StringComparison.OrdinalIgnoreCase)) continue;
                metadata[kvp.Key] = kvp.Value?.ToString() ?? string.Empty;
            }
        }

        return new QueryResultObj
        {
            Input = input,
            Output = output,
            Score = hit.Score,
            Metadata = metadata
        };
    }

    public override bool TryHydrateFromDocument(object item, JObject source)
    {
        if (item is not BlogIndexDocument blog || source is null) return false;

        var storedTitle = source["title"]?.Value<string>() ?? string.Empty;
        var storedContent = source["content"]?.Value<string>() ?? string.Empty;
        var storedSummary = source["summary"]?.Value<string>() ?? string.Empty;

        if (!string.Equals(blog.Title ?? string.Empty, storedTitle, StringComparison.Ordinal))
            return false;

        var desiredContent = !string.IsNullOrWhiteSpace(blog.Content)
            ? blog.Content
            : blog.Summary ?? string.Empty;

        var storedContentSource = !string.IsNullOrWhiteSpace(storedContent)
            ? storedContent
            : storedSummary;

        if (!string.Equals(desiredContent ?? string.Empty, storedContentSource ?? string.Empty, StringComparison.Ordinal))
            return false;

        blog.TitleEmbedding = source["title_embedding"]?.ToObject<List<float>>() ?? new List<float>();
        blog.ContentEmbedding = source["content_embedding"]?.ToObject<List<float>>() ?? new List<float>();

        return blog.TitleEmbedding.Count > 0 && blog.ContentEmbedding.Count > 0;
    }
}
