using Microsoft.Extensions.Logging;
using System;
using NetworkMonitor.Objects;

namespace NetworkMonitor.Search.Services
{
    public interface IEmbeddingGeneratorFactory
    {
        IEmbeddingGenerator Create();
    }

    public class EmbeddingGeneratorFactory : IEmbeddingGeneratorFactory
    {
        private readonly MLParams _mlParams;
        private readonly ILoggerFactory _loggerFactory;

        public EmbeddingGeneratorFactory(MLParams mlParams, ILoggerFactory loggerFactory)
        {
            _mlParams = mlParams;
            _loggerFactory = loggerFactory;
        }

        public IEmbeddingGenerator Create()
        {
            if (_mlParams.EmbeddingProvider.ToLower() == "api")
            {
                if (string.IsNullOrWhiteSpace(_mlParams.LlmEmbedKey))
                    throw new Exception("LlmEmbedKey must be set in config for API embedding provider.");
                return new NovitaEmbeddingGenerator(
                    _mlParams,
                    _loggerFactory.CreateLogger<NovitaEmbeddingGenerator>(),
                    new NovitaApiClient(_mlParams, _loggerFactory.CreateLogger<NovitaApiClient>())
                );
            }
            else
            {
                return new EmbeddingGenerator(
                    _mlParams,
                    _loggerFactory.CreateLogger<EmbeddingGenerator>()
                );
            }
        }
    }
}
