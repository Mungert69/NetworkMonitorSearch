//  NetworkMonitor.Search.Strategies ------------------------------------------------
using System;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using NetworkMonitor.Objects;
using Newtonsoft.Json;

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
    bool CanHandle(object item);
    bool CanHandle(string indexName);

    // Token estimation
    IEnumerable<string> GetFields(object item);
    (int padToTokens, int actualMax) EstimatePadding(IEnumerable<string> jsonFiles, string embeddingModelDir, int maxCap, int minCap);
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
        new Dictionary<string, float>
        {
            [QuestionVectorFieldName] = 1f,
            [ContentVectorFieldName] = 1f,
            [SummaryVectorFieldName] = 1f
        };

    public override IEnumerable<string> GetFields(object item)
    {
        var book = (T)item; // Strongly typed
        return new[] { book.Input, book.Output, book.Summary };
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
    }

    public override string ComputeId(object item) => IdHelper.Sha256(((T)item).Output);

    public override object BuildIndexDocument(object item)
    {
        var book = (T)item;
        return new
        {
            input = book.Input,
            output = book.Output,
            summary = book.Summary,
            input_embedding = book.InputEmbedding,
            output_embedding = book.OutputEmbedding,
            summary_embedding = book.SummaryEmbedding
        };
    }

    public override string GetIndexMapping(int dim) => $@"
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
                                ""method"": {{ ""name"": ""hnsw"", ""space_type"": ""l2"", ""engine"": ""faiss"" }} }}
    }}
  }}
}}";
}

public sealed class SecurityBookIndexingStrategy : MultiVectorBookIndexingStrategyBase<SecurityBook>
{
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
public class SecurityBook : IMultiVectorBook
{
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public string Summary { get; set; } = "";
    public List<float> InputEmbedding { get; set; } = new();
    public List<float> OutputEmbedding { get; set; } = new();
    public List<float> SummaryEmbedding { get; set; } = new();
}

public class QuantumBook : IMultiVectorBook
{
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public string Summary { get; set; } = "";
    public List<float> InputEmbedding { get; set; } = new();
    public List<float> OutputEmbedding { get; set; } = new();
    public List<float> SummaryEmbedding { get; set; } = new();
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
}
