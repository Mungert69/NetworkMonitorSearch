using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using Microsoft.Extensions.Logging;
using System;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.Text;
using NetworkMonitor.Objects;
using NetworkMonitor.Objects.Repository;
using NetworkMonitor.Utils.Helpers;

namespace NetworkMonitor.Search.Services
{
    public interface IRabbitListener
    {
        Task<ResultObj> CreateIndex(CreateIndexRequest createIndexRequest);
        Task<ResultObj> QueryIndex(QueryIndexRequest queryIndexRequest);
        Task Shutdown();
        Task<ResultObj> Setup();
    }

    public class RabbitListener : RabbitListenerBase, IRabbitListener
    {
        private readonly IOpenSearchService _openSearchService;

        public RabbitListener(IOpenSearchService openSearchService, ILogger<RabbitListenerBase> logger, ISystemParamsHelper systemParamsHelper)
            : base(logger, DeriveSystemUrl(systemParamsHelper))
        {
            _openSearchService = openSearchService;
        }

        private static SystemUrl DeriveSystemUrl(ISystemParamsHelper systemParamsHelper)
        {
            return systemParamsHelper.GetSystemParams().ThisSystemUrl;
        }

        protected override void InitRabbitMQObjs()
        {
            _rabbitMQObjs.Add(new RabbitMQObj()
            {
                ExchangeName = "createIndex",
                FuncName = "createIndex",
                MessageTimeout = 60000
            });

            _rabbitMQObjs.Add(new RabbitMQObj()
            {
                ExchangeName = "queryIndex",
                FuncName = "queryIndex",
                MessageTimeout = 60000
            });
        }

        protected override async Task<ResultObj> DeclareConsumers()
        {
            var result = new ResultObj();
            try
            {
                foreach (var rabbitMQObj in _rabbitMQObjs)
                {
                    if (rabbitMQObj.ConnectChannel != null)
                    {
                        rabbitMQObj.Consumer = new AsyncEventingBasicConsumer(rabbitMQObj.ConnectChannel);

                        switch (rabbitMQObj.FuncName)
                        {
                            case "createIndex":
                                await rabbitMQObj.ConnectChannel.BasicQosAsync(prefetchSize: 0, prefetchCount: 1, global: false);
                                rabbitMQObj.Consumer.ReceivedAsync += async (model, ea) =>
                            {
                                try
                                {

                                    result = await CreateIndex(ConvertToObject<CreateIndexRequest>(model, ea));
                                    await rabbitMQObj.ConnectChannel.BasicAckAsync(ea.DeliveryTag, false);
                                }
                                catch (Exception ex)
                                {
                                    _logger.LogError(" Error : RabbitListener.DeclareConsumers.createIndex " + ex.Message);
                                }
                            };
                                break;
                            case "queryIndex":
                                await rabbitMQObj.ConnectChannel.BasicQosAsync(prefetchSize: 0, prefetchCount: 1, global: false);
                                rabbitMQObj.Consumer.ReceivedAsync += async (model, ea) =>
                            {
                                try
                                {

                                    result = await QueryIndex(ConvertToObject<QueryIndexRequest>(model, ea));
                                    await rabbitMQObj.ConnectChannel.BasicAckAsync(ea.DeliveryTag, false);
                                }
                                catch (Exception ex)
                                {
                                    _logger.LogError(" Error : RabbitListener.DeclareConsumers.queryIndex " + ex.Message);
                                }
                            };
                                break;
                        }

                    }
                }

                result.Success = true;
                result.Message = "Success: Declared all consumers.";
            }
            catch (Exception e)
            {
                result.Success = false;
                result.Message = $"Error: Failed to declare consumers. Error was: {e.Message}";
            }
            return result;
        }

        public async Task<ResultObj> CreateIndex(CreateIndexRequest createIndexRequest)
        {
            var result = new ResultObj();
            result.Success = false;
            result.Message = "MessageAPI: CreateIndex: ";
            if (createIndexRequest == null)
            {
                result.Success = false;
                result.Message += "Error: createIndexRequest is null.";
                return result;
            }

            try
            {
                // Call the OpenSearch service to create the index
                var createIndexResult = await _openSearchService.CreateIndexAsync(createIndexRequest);
                result.Success = createIndexResult.Success;
                result.Message += createIndexResult.Message;

                _logger.LogInformation(result.Message);
            }
            catch (Exception e)
            {
                result.Success = false;
                result.Message += $"Error: Failed to create index. Error was: {e.Message}";
                _logger.LogError(result.Message);
            }
            return result;
        }

        public async Task<ResultObj> QueryIndex(QueryIndexRequest queryIndexRequest)
        {
            var result = new ResultObj();
            result.Success = false;
            result.Message = "MessageAPI: QueryIndex: ";
            if (queryIndexRequest == null)
            {
                result.Success = false;
                result.Message += "Error: queryIndexRequest is null.";
                return result;
            }

            try
            {
                // Call the OpenSearch service to query the index
                var queryIndexResult = await _openSearchService.QueryIndexAsync(queryIndexRequest);
                result.Success = queryIndexResult.Success;
                result.Message += queryIndexResult.Message;

                _logger.LogInformation(result.Message);
            }
            catch (Exception e)
            {
                result.Success = false;
                result.Message += $"Error: Failed to query index. Error was: {e.Message}";
                _logger.LogError(result.Message);
            }
            return result;
        }


    }


}