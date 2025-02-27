using System.Net;
using System;
namespace NetworkMonitor.Search.Services;

public class OSModelParams{


    private string _url;
    private string _user;
    private string _key;
    private string _bertModelDir;
    private int _bertModelVecDim=128;
    private string _defaultIndex;

    public Uri SearchUri  => new Uri(_url);

    public string Url { get => _url; set => _url = value; }
    public string Key { get => _key; set => _key = value; }
    public string BertModelDir { get => _bertModelDir; set => _bertModelDir = value; }
    public string User { get => _user; set => _user = value; }
    public string DefaultIndex { get => _defaultIndex; set => _defaultIndex = value; }
    public int BertModelVecDim { get => _bertModelVecDim; set => _bertModelVecDim = value; }
}
