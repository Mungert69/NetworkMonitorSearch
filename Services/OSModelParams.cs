using System.Net;
using System;
using System.Collections.Generic;
namespace NetworkMonitor.Search.Services;

public class OSModelParams
{


    private string _url;
    private string _user;
    private string _key;
    private string _embeddingModelDir;
    private int _embeddingModelVecDim = 128;
    private string _defaultIndex;
    private TimeSpan _httpTimeout = TimeSpan.FromSeconds(100);
    private HashSet<string> _hybridIndices = new(StringComparer.OrdinalIgnoreCase);
    private bool _hybridRerankEnabled = true;
    private int _hybridCandidateMultiplier = 4;
    private int _hybridMinCandidates = 12;
    private int _hybridRrfK = 60;
    private float _hybridVectorWeight = 1.0f;
    private float _hybridLexicalWeight = 1.0f;
    private bool _enableAltQuestionFields = false;

    public Uri SearchUri => new Uri(_url);

    public string Url { get => _url; set => _url = value; }
    public string Key { get => _key; set => _key = value; }
    public string EmbeddingModelDir { get => _embeddingModelDir; set => _embeddingModelDir = value; }
    public string User { get => _user; set => _user = value; }
    public string DefaultIndex { get => _defaultIndex; set => _defaultIndex = value; }
    public int EmbeddingModelVecDim { get => _embeddingModelVecDim; set => _embeddingModelVecDim = value; }
    public TimeSpan HttpTimeout { get => _httpTimeout; set => _httpTimeout = value; }
    public HashSet<string> HybridIndices
    {
        get => _hybridIndices;
        set => _hybridIndices = value ?? new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    }
    public bool HybridRerankEnabled { get => _hybridRerankEnabled; set => _hybridRerankEnabled = value; }
    public int HybridCandidateMultiplier { get => _hybridCandidateMultiplier; set => _hybridCandidateMultiplier = value; }
    public int HybridMinCandidates { get => _hybridMinCandidates; set => _hybridMinCandidates = value; }
    public int HybridRrfK { get => _hybridRrfK; set => _hybridRrfK = value; }
    public float HybridVectorWeight { get => _hybridVectorWeight; set => _hybridVectorWeight = value; }
    public float HybridLexicalWeight { get => _hybridLexicalWeight; set => _hybridLexicalWeight = value; }
    public bool EnableAltQuestionFields { get => _enableAltQuestionFields; set => _enableAltQuestionFields = value; }
}
