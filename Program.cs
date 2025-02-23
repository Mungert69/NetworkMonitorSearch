using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using NetworkMonitor.Objects.Factory;
using System;

namespace NetworkMonitor.Search
{
    public class Program
    {
        public static void Main(string[] args)
        {
            string appFile = "appsettings.json";
            IConfigurationRoot config = new ConfigurationBuilder()
                .AddJsonFile(appFile, optional: false)
                .Build();

            IHost host = CreateHostBuilder(config).Build();
            host.Run();
        }

        public static IHostBuilder CreateHostBuilder(IConfigurationRoot config) =>
            Host.CreateDefaultBuilder()
                .ConfigureAppConfiguration(builder =>
                {
                    builder.AddConfiguration(config);
                })
                .ConfigureServices((hostContext, services) =>
                {
                    // Register your Startup class's ConfigureServices method
                    var startup = new Startup(hostContext.Configuration);
                    startup.ConfigureServices(services);
                });
    }
}
