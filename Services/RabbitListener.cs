using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using Microsoft.Extensions.Logging;
using System;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.Text;
using NetworkMonitor.Objects;

namespace NetworkMonitor.Scheduler.Services
{
    public interface IRabbitListener
    {
        ResultObj CreateIndex(CreateIndexRequest createIndexRequest);
        ResultObj QueryIndex(QueryIndexRequest queryIndexRequest);
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
                    rabbitMQObj.Consumer = new AsyncEventingBasicConsumer(rabbitMQObj.ConnectChannel);
                    if (rabbitMQObj.ConnectChannel != null)
                    {
                        await rabbitMQObj.ConnectChannel.BasicQosAsync(prefetchSize: 0, prefetchCount: 1, global: false);

                        switch (rabbitMQObj.FuncName)
                        {
                            case "createIndex":
                                rabbitMQObj.Consumer.Received += async (model, ea) =>
                                {
                                    var createIndexRequest = ConvertToObject<CreateIndexRequest>(model, ea);
                                    result = CreateIndex(createIndexRequest);
                                    await rabbitMQObj.ConnectChannel.BasicAckAsync(ea.DeliveryTag, false);
                                };
                                break;

                            case "queryIndex":
                                rabbitMQObj.Consumer.Received += async (model, ea) =>
                                {
                                    var queryIndexRequest = ConvertToObject<QueryIndexRequest>(model, ea);
                                    result = QueryIndex(queryIndexRequest);
                                    await rabbitMQObj.ConnectChannel.BasicAckAsync(ea.DeliveryTag, false);
                                };
                                break;
                        }

                        await rabbitMQObj.ConnectChannel.BasicConsumeAsync(queue: rabbitMQObj.ExchangeName, autoAck: false, consumer: rabbitMQObj.Consumer);
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

        public ResultObj CreateIndex(CreateIndexRequest createIndexRequest)
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
                var createIndexResult = _openSearchService.CreateIndexAsync(createIndexRequest).Result;
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

        public ResultObj QueryIndex(QueryIndexRequest queryIndexRequest)
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
                var queryIndexResult = _openSearchService.QueryIndexAsync(queryIndexRequest).Result;
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