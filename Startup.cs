using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using NetworkMonitor.Search.Services;
using Microsoft.AspNetCore.Http;
using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using NetworkMonitor.Utils;
using NetworkMonitor.Objects.Factory;
using NetworkMonitor.Objects.Repository;
using HostInitActions;
using Microsoft.Extensions.Logging;
using NetworkMonitor.Utils.Helpers;
using System.Net.Http;
using NetworkMonitor.Objects;


namespace NetworkMonitor.Search
{
    public class Startup
    {
        private readonly CancellationTokenSource _cancellationTokenSource;
        public Startup(IConfiguration configuration)
        {
            _cancellationTokenSource = new CancellationTokenSource();
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        private IServiceCollection _services;

        // This method gets called by the runtime. Use this method to add services to the container.
        public void ConfigureServices(IServiceCollection services)
        {
            _services = services;
            services.AddLogging(builder =>
               {
                   builder.AddSimpleConsole(options =>
                        {
                            options.TimestampFormat = "yyyy-MM-dd HH:mm:ss ";
                            options.IncludeScopes = true;
                        });
               });

            services.Configure<HostOptions>(s => s.ShutdownTimeout = TimeSpan.FromMinutes(1));
            services.AddSingleton(_cancellationTokenSource);
            services.AddSingleton<IRabbitRepo, RabbitRepo>();
            services.AddSingleton<IRabbitListener, RabbitListener>();
            services.AddSingleton<ISystemParamsHelper, SystemParamsHelper>();
            services.AddSingleton<IOpenSearchService, OpenSearchService>();
            // Register MLParams as a singleton, constructed once from ISystemParamsHelper
            services.AddSingleton<MLParams>(sp =>
            {
                var systemParamsHelper = sp.GetRequiredService<ISystemParamsHelper>();
                return systemParamsHelper.GetMLParams();
            });
             services.AddSingleton<SystemParams>(sp =>
            {
                var systemParamsHelper = sp.GetRequiredService<ISystemParamsHelper>();
                return systemParamsHelper.GetSystemParams();
            });

            services.AddSingleton<NovitaApiClient>();

            services.AddSingleton<IEmbeddingGeneratorFactory, EmbeddingGeneratorFactory>();
            services.AddSingleton<IEmbeddingGenerator>(sp =>
            {
                var factory = sp.GetRequiredService<IEmbeddingGeneratorFactory>();
                return factory.Create();
            });

          

            services.AddSingleton<IFileRepo, FileRepo>();
            services.AddAsyncServiceInitialization()
                .AddInitAction<IRabbitRepo>(async (rabbitRepo) =>
                    {
                        await rabbitRepo.ConnectAndSetUp();
                    })
                .AddInitAction<IOpenSearchService>(async (openSearchService) =>
                    {
                        await openSearchService.Init();
                    })
                .AddInitAction<IRabbitListener>(async (rabbitListener) =>
                    {
                        await rabbitListener.Setup();
                    });
        }


    }
}
