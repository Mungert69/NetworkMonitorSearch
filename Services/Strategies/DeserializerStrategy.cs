using System;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using Newtonsoft.Json;
using NetworkMonitor.Objects;

namespace NetworkMonitor.Search.Services;

public interface IIndexDeserializerStrategy
{
    bool CanHandle(string indexName);
    List<object> Deserialize(string json);
}
public class DocumentDeserializerStrategy : IIndexDeserializerStrategy
{
    public bool CanHandle(string indexName) => indexName.Equals("documents", StringComparison.OrdinalIgnoreCase);

    public List<object> Deserialize(string json)
    {
        var list = JsonConvert.DeserializeObject<List<Document>>(json);
        return list?.Cast<object>().ToList() ?? new List<object>();
    }
}
public class SecurityBookDeserializerStrategy : IIndexDeserializerStrategy
{
    public bool CanHandle(string indexName) => indexName.Equals("securitybooks", StringComparison.OrdinalIgnoreCase);

    public List<object> Deserialize(string json)
    {
        var list = JsonConvert.DeserializeObject<List<SecurityBook>>(json);
        return list?.Cast<object>().ToList() ?? new List<object>();
    }
}
