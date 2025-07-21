using System.Net;
using System;
namespace NetworkMonitor.Search.Services;

public class OSModelParams{


    private string _url;
    private string _user;
    private string _key;
    private string _embeddingModelDir;
    private int _embeddingModelVecDim=128;
    private string _defaultIndex;

    public Uri SearchUri  => new Uri(_url);

    public string Url { get => _url; set => _url = value; }
    public string Key { get => _key; set => _key = value; }
    public string EmbeddingModelDir { get => _embeddingModelDir; set => _embeddingModelDir = value; }
    public string User { get => _user; set => _user = value; }
    public string DefaultIndex { get => _defaultIndex; set => _defaultIndex = value; }
    public int EmbeddingModelVecDim { get => _embeddingModelVecDim; set => _embeddingModelVecDim = value; }
}
