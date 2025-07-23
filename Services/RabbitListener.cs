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
        Task<ResultObj> CreateSnapshot(CreateSnapshotRequest createSnapshotRequest);
        Task Shutdown();
        Task<ResultObj> Setup();
       
    }

    public class RabbitListener : RabbitListenerBase, IRabbitListener
    {
        private readonly IOpenSearchService _openSearchService;

        public RabbitListener(IOpenSearchService openSearchService, ILogger<RabbitListenerBase> logger, SystemParams systemParams)
            : base(logger, DeriveSystemUrl(systemParams))
        {
            _openSearchService = openSearchService;
        }

        private static SystemUrl DeriveSystemUrl(SystemParams systemParams)
        {
            return systemParams.ThisSystemUrl;
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

            _rabbitMQObjs.Add(new RabbitMQObj()
            {
                ExchangeName = "createSnapshot",
                FuncName = "createSnapshot",
                MessageTimeout = 60000
            });
        }

        protected override async Task<ResultObj> DeclareConsumers()
        {
            var result = new ResultObj();
            try
            {
                 await Parallel.ForEachAsync(_rabbitMQObjs, async (rabbitMQObj, cancellationToken) =>
                {

                    if (rabbitMQObj.ConnectChannel != null)
                    {

                        rabbitMQObj.Consumer = new AsyncEventingBasicConsumer(rabbitMQObj.ConnectChannel);
                        await rabbitMQObj.ConnectChannel.BasicConsumeAsync(
                                queue: rabbitMQObj.QueueName,
                                autoAck: false,
                                consumer: rabbitMQObj.Consumer
                            );


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
                            case "createSnapshot":
                                await rabbitMQObj.ConnectChannel.BasicQosAsync(prefetchSize: 0, prefetchCount: 1, global: false);
                                rabbitMQObj.Consumer.ReceivedAsync += async (model, ea) =>
                                {
                                    try
                                    {
                                        result = await CreateSnapshot(ConvertToObject<CreateSnapshotRequest>(model, ea));
                                        await rabbitMQObj.ConnectChannel.BasicAckAsync(ea.DeliveryTag, false);
                                    }
                                    catch (Exception ex)
                                    {
                                        _logger.LogError(" Error : RabbitListener.DeclareConsumers.createSnapshot " + ex.Message);
                                    }
                                };
                                break;
                        }

                    }
                });

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

        public async Task<ResultObj> CreateIndex(CreateIndexRequest? createIndexRequest)
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
                ResultObj createIndexResult;
                // Only allow bulk/directory mode, single-file indexing is not supported here anymore
                if (createIndexRequest.CreateFromJsonDataDir)
                {
                    createIndexResult = await _openSearchService.CreateIndicesFromDataDirAsync(createIndexRequest);
                }
                else
                {
                    createIndexResult = await _openSearchService.CreateIndexAsync(createIndexRequest);
                }
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

        public async Task<ResultObj> QueryIndex(QueryIndexRequest? queryIndexRequest)
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

 public async Task<ResultObj> CreateSnapshot(CreateSnapshotRequest? createSnapshotRequest)
        {
            var result = new ResultObj();
            result.Success = false;
            result.Message = "MessageAPI: CreateSnapshot: ";
            if (createSnapshotRequest == null)
            {
                result.Success = false;
                result.Message += "Error: createSnapshotRequest is null.";
                return result;
            }

            try
            {
                // Call the OpenSearch service to create the snapshot
                var createSnapshotResult = await _openSearchService.CreateSnapshotAsync(
                    createSnapshotRequest.SnapshotRepo,
                    createSnapshotRequest.SnapshotName,
                    createSnapshotRequest.Indices
                );
                result.Success = createSnapshotResult.Success;
                result.Message += createSnapshotResult.Message;

                _logger.LogInformation(result.Message);
            }
            catch (Exception e)
            {
                result.Success = false;
                result.Message += $"Error: Failed to create snapshot. Error was: {e.Message}";
                _logger.LogError(result.Message);
            }
            return result;
        }

    }


}