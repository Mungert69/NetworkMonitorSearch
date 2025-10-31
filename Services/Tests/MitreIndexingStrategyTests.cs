using NetworkMonitor.Objects;
using NetworkMonitor.Search.Services;
using Xunit;

namespace NetworkMonitorSearch.Tests.Services.Strategies
{
    public class MitreIndexingStrategyTests
    {
        private readonly MitreIndexingStrategy _strategy = new();

        [Fact]
        public void ComputeContentHash_OnlyDependsOnOutput()
        {
            var doc = new Mitre
            {
                Input = "Initial Mitre tactic description.",
                Output = "Mitre technique detail."
            };

            var hash1 = _strategy.ComputeContentHash(doc);

            doc.Input = "Changed input";
            var hash2 = _strategy.ComputeContentHash(doc);

            Assert.Equal(hash1, hash2);

            doc.Output = "Updated output content.";
            var hash3 = _strategy.ComputeContentHash(doc);

            Assert.NotEqual(hash1, hash3);
        }
    }
}
