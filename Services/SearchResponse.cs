using Newtonsoft.Json;
using System.Collections.Generic;

namespace NetworkMonitor.Objects
{
  
    public class SearchResponseObj
    {
        [JsonProperty("took")]
        public int Took { get; set; }

        [JsonProperty("timed_out")]
        public bool TimedOut { get; set; }

        [JsonProperty("_shards")]
        public Shards Shards { get; set; }

        [JsonProperty("hits")]
        public Hits Hits { get; set; }
    }

    public class Shards
    {
        [JsonProperty("total")]
        public int Total { get; set; }

        [JsonProperty("successful")]
        public int Successful { get; set; }

        [JsonProperty("skipped")]
        public int Skipped { get; set; }

        [JsonProperty("failed")]
        public int Failed { get; set; }
    }

    public class Hits
    {
        [JsonProperty("total")]
        public Total Total { get; set; }

        [JsonProperty("max_score")]
        public float MaxScore { get; set; }

        [JsonProperty("hits")]
        public List<Hit> HitsList { get; set; }
    }

    public class Total
    {
        [JsonProperty("value")]
        public int Value { get; set; }

        [JsonProperty("relation")]
        public string Relation { get; set; } = "";
    }

    public class Hit
    {
        [JsonProperty("_index")]
        public string Index { get; set; } = "";

        [JsonProperty("_id")]
        public string Id { get; set; } = "";

        [JsonProperty("_score")]
        public float Score { get; set; }

        [JsonProperty("_source")]
        public Source Source { get; set; }
    }

    public class Source
    {
        [JsonProperty("input")]
        public string Input { get; set; } = "";

        [JsonProperty("output")]
        public string Output { get; set; } = "";

        [JsonProperty("embedding")]
        public List<float> Embedding { get; set; }
    }
}