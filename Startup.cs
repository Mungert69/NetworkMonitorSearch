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
                   builder.AddConfiguration(Configuration.GetSection("Logging"));
                   builder.AddSimpleConsole(options =>
                        {
                            options.TimestampFormat = "yyyy-MM-dd HH:mm:ss ";
                            options.IncludeScopes = true;
                        });
               });

            services.Configure<HostOptions>(s => s.ShutdownTimeout = TimeSpan.FromSeconds(30));
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
                var baseGenerator = factory.Create();
                var mlParams = sp.GetRequiredService<MLParams>();
                var logger = sp.GetRequiredService<ILogger<CachedEmbeddingGenerator>>();
                var modelIdentity = $"{mlParams.EmbeddingProvider}|{mlParams.EmbeddingModelDir}|{mlParams.EmbeddingModelVecDim}";
                return new CachedEmbeddingGenerator(
                    baseGenerator,
                    mlParams.OpenSearchUrl,
                    mlParams.OpenSearchUser,
                    mlParams.OpenSearchKey,
                    modelIdentity,
                    logger);
            });

          

            services.AddSingleton<IFileRepo, FileRepo>(
                 provider =>
                 {
                     return new FileRepo(false, "./state");
                 }
             );
            services.AddAsyncServiceInitialization()
                .AddInitAction<IRabbitRepo>(async (rabbitRepo) =>
                    {
                        await rabbitRepo.ConnectAndSetUp(_cancellationTokenSource.Token);
                    })
                .AddInitAction<IOpenSearchService>(async (openSearchService) =>
                    {
                        await openSearchService.Init();
                    })
                .AddInitAction<IRabbitListener>(async (rabbitListener) =>
                    {
                        await rabbitListener.Setup(_cancellationTokenSource.Token);
                    });
        }

        public void Configure(IApplicationBuilder app, IWebHostEnvironment env, IHostApplicationLifetime appLifetime)
        {
            appLifetime.ApplicationStopping.Register(() =>
            {
                _cancellationTokenSource.Cancel();

                var rabbitRepo = app.ApplicationServices.GetService<IRabbitRepo>();
                if (rabbitRepo != null)
                {
                    rabbitRepo.Shutdown().GetAwaiter().GetResult();
                }

                var rabbitListener = app.ApplicationServices.GetService<IRabbitListener>();
                if (rabbitListener != null)
                {
                    rabbitListener.Shutdown().GetAwaiter().GetResult();
                }
            });
        }

    }
}
