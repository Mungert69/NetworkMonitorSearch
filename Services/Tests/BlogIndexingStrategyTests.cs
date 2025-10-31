using System;
using System.Collections.Generic;
using NetworkMonitor.Objects;
using NetworkMonitor.Search.Services;
using Xunit;

namespace NetworkMonitorSearch.Tests.Services.Strategies
{
    public class BlogIndexingStrategyTests
    {
        private readonly BlogIndexingStrategy _strategy = new();

        [Fact]
        public void ComputeContentHash_IgnoresNonEmbeddedFields()
        {
            var blog = new BlogIndexDocument
            {
                Title = "Quantum Basics",
                Content = "The full content body for the blog post.",
                Summary = "Short summary of the post.",
                Author = "Alice",
                Slug = "quantum-basics",
                Url = "/blog/quantum-basics",
                Categories = new List<string> { "security" },
                PublishedAt = new DateTime(2024, 6, 1)
            };

            var hash1 = _strategy.ComputeContentHash(blog);

            blog.Author = "Bob";
            blog.PublishedAt = blog.PublishedAt?.AddDays(2);
            blog.Categories.Add("quantum");
            blog.Url = "/blog/q-basics";

            var hash2 = _strategy.ComputeContentHash(blog);

            Assert.Equal(hash1, hash2);

            blog.Content = "An updated content body that should change the hash.";
            var hash3 = _strategy.ComputeContentHash(blog);

            Assert.NotEqual(hash1, hash3);
        }

        [Fact]
        public void ComputeContentHash_FallsBackToSummaryWhenContentMissing()
        {
            var blog = new BlogIndexDocument
            {
                Title = "Weekly Security Roundup",
                Content = string.Empty,
                Summary = "Initial summary"
            };

            var hash1 = _strategy.ComputeContentHash(blog);

            blog.Summary = "Summary updated with new findings";
            var hash2 = _strategy.ComputeContentHash(blog);

            Assert.NotEqual(hash1, hash2);
        }
    }
}
