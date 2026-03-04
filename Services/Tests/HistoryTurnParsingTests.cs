using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NetworkMonitor.Search.Services;
using Xunit;

namespace NetworkMonitorSearch.Tests.Services;

public class HistoryTurnParsingTests
{
    [Fact]
    public void BuildTurnDocs_ParsesTextToolCallAndToolResponse()
    {
        var history = new JArray
        {
            new JObject
            {
                ["role"] = "user",
                ["content"] = "Check previous backup settings"
            },
            new JObject
            {
                ["role"] = "assistant",
                ["content"] = "",
                ["toolCallId"] = "call_1",
                ["toolCalls"] = new JArray
                {
                    new JObject
                    {
                        ["function"] = new JObject
                        {
                            ["name"] = "get_backup_config"
                        }
                    }
                }
            },
            new JObject
            {
                ["role"] = "tool",
                ["name"] = "get_backup_config",
                ["toolCallId"] = "call_1",
                ["content"] = "failed due to timeout while contacting node"
            }
        };

        var buildMethod = typeof(OpenSearchHelper).GetMethod("BuildTurnDocs", BindingFlags.Static | BindingFlags.NonPublic);
        Assert.NotNull(buildMethod);

        var turnsObj = buildMethod!.Invoke(null, new object[] { history });
        var turns = ((IEnumerable)turnsObj!).Cast<object>().ToList();

        Assert.Equal(3, turns.Count);

        Assert.Equal("text", GetProperty(turns[0], "TurnType"));
        Assert.Equal("user", GetProperty(turns[0], "Role"));
        Assert.Contains("Check previous backup settings", GetProperty(turns[0], "Output"), StringComparison.Ordinal);

        Assert.Equal("tool_call", GetProperty(turns[1], "TurnType"));
        Assert.Equal("get_backup_config", GetProperty(turns[1], "ToolName"));
        Assert.Equal("requested", GetProperty(turns[1], "ToolStatus"));

        Assert.Equal("tool_response", GetProperty(turns[2], "TurnType"));
        Assert.Equal("get_backup_config", GetProperty(turns[2], "ToolName"));
        Assert.Equal("timeout", GetProperty(turns[2], "ToolStatus"));
        Assert.Equal("call_1", GetProperty(turns[2], "ToolCallId"));
    }

    [Theory]
    [InlineData("request timeout while reading response", "timeout")]
    [InlineData("user canceled command", "canceled")]
    [InlineData("operation failed with exception", "error")]
    [InlineData("completed successfully", "success")]
    [InlineData("", "unknown")]
    public void InferToolStatus_ClassifiesExpectedStates(string content, string expected)
    {
        var statusMethod = typeof(OpenSearchHelper).GetMethod("InferToolStatus", BindingFlags.Static | BindingFlags.NonPublic);
        Assert.NotNull(statusMethod);

        var actual = (string?)statusMethod!.Invoke(null, new object[] { content });

        Assert.Equal(expected, actual);
    }

    private static string GetProperty(object instance, string propertyName)
    {
        var prop = instance.GetType().GetProperty(propertyName, BindingFlags.Instance | BindingFlags.Public);
        Assert.NotNull(prop);
        return prop!.GetValue(instance)?.ToString() ?? string.Empty;
    }
}
