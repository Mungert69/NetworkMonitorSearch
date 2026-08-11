using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using Microsoft.Extensions.Logging;
using System;
using System.Threading;
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
        Task<ResultObj> QueryMemory(MemoryQueryRequest memoryQueryRequest);
        Task<ResultObj> QueryMemoryTurnWindow(MemoryTurnWindowRequest request);
        Task<ResultObj> QueryMemoryTurnRange(MemoryTurnRangeRequest request);
        Task<ResultObj> HistoryStore(HistoryStoreRequest historyStoreRequest);
        Task<ResultObj> CreateSnapshot(CreateSnapshotRequest createSnapshotRequest);
        Task Shutdown();
        Task<ResultObj> Setup();
        Task<ResultObj> Setup(CancellationToken cancellationToken);
       
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
                ExchangeName = "queryMemory",
                FuncName = "queryMemory",
                MessageTimeout = 60000
            });
            _rabbitMQObjs.Add(new RabbitMQObj()
            {
                ExchangeName = "queryMemoryTurnWindow",
                FuncName = "queryMemoryTurnWindow",
                MessageTimeout = 60000
            });
            _rabbitMQObjs.Add(new RabbitMQObj()
            {
                ExchangeName = "queryMemoryTurnRange",
                FuncName = "queryMemoryTurnRange",
                MessageTimeout = 60000
            });
            _rabbitMQObjs.Add(new RabbitMQObj()
            {
                ExchangeName = "historyStore",
                FuncName = "historyStore",
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
                                await RegisterConsumerHandlerAsync(rabbitMQObj, 1, "createIndex", async (model, ea) =>
                                {
                                    result = await CreateIndex(ConvertToObject<CreateIndexRequest>(model, ea));
                                });
                                break;
                            case "queryIndex":
                                await RegisterConsumerHandlerAsync(rabbitMQObj, 1, "queryIndex", async (model, ea) =>
                                {
                                    result = await QueryIndex(ConvertToObject<QueryIndexRequest>(model, ea));
                                });
                                break;
                            case "createSnapshot":
                                await RegisterConsumerHandlerAsync(rabbitMQObj, 1, "createSnapshot", async (model, ea) =>
                                {
                                    result = await CreateSnapshot(ConvertToObject<CreateSnapshotRequest>(model, ea));
                                });
                                break;
                            case "queryMemory":
                                await RegisterConsumerHandlerAsync(rabbitMQObj, 1, "queryMemory", async (model, ea) =>
                                {
                                    result = await QueryMemory(ConvertToObject<MemoryQueryRequest>(model, ea));
                                });
                                break;
                            case "historyStore":
                                await RegisterConsumerHandlerAsync(rabbitMQObj, 1, "historyStore", async (model, ea) =>
                                {
                                    result = await HistoryStore(ConvertToObject<HistoryStoreRequest>(model, ea));
                                });
                                break;
                            case "queryMemoryTurnWindow":
                                await RegisterConsumerHandlerAsync(rabbitMQObj, 1, "queryMemoryTurnWindow", async (model, ea) =>
                                {
                                    result = await QueryMemoryTurnWindow(ConvertToObject<MemoryTurnWindowRequest>(model, ea));
                                });
                                break;
                            case "queryMemoryTurnRange":
                                await RegisterConsumerHandlerAsync(rabbitMQObj, 1, "queryMemoryTurnRange", async (model, ea) =>
                                {
                                    result = await QueryMemoryTurnRange(ConvertToObject<MemoryTurnRangeRequest>(model, ea));
                                });
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

        public async Task<ResultObj> QueryMemory(MemoryQueryRequest? memoryQueryRequest)
        {
            var result = new ResultObj();
            result.Success = false;
            result.Message = "MessageAPI: QueryMemory: ";
            if (memoryQueryRequest == null)
            {
                result.Message += "Error: memoryQueryRequest is null.";
                return result;
            }

            try
            {
                var queryMemoryResult = await _openSearchService.QueryMemoryAsync(memoryQueryRequest);
                result.Success = queryMemoryResult.Success;
                result.Message += queryMemoryResult.Message;
                _logger.LogInformation(result.Message);
            }
            catch (Exception e)
            {
                result.Success = false;
                result.Message += $"Error: Failed to query memory. Error was: {e.Message}";
                _logger.LogError(result.Message);
            }
            return result;
        }

        public async Task<ResultObj> HistoryStore(HistoryStoreRequest? historyStoreRequest)
        {
            var result = new ResultObj();
            result.Success = false;
            result.Message = "MessageAPI: HistoryStore: ";
            if (historyStoreRequest == null)
            {
                result.Message += "Error: historyStoreRequest is null.";
                return result;
            }

            try
            {
                var historyResult = await _openSearchService.HistoryStoreAsync(historyStoreRequest);
                result.Success = historyResult.Success;
                result.Message += historyResult.Message;
                _logger.LogInformation(result.Message);
            }
            catch (Exception e)
            {
                result.Success = false;
                result.Message += $"Error: Failed to process history store message. Error was: {e.Message}";
                _logger.LogError(result.Message);
            }

            return result;
        }

        public async Task<ResultObj> QueryMemoryTurnWindow(MemoryTurnWindowRequest? request)
        {
            var result = new ResultObj();
            result.Success = false;
            result.Message = "MessageAPI: QueryMemoryTurnWindow: ";
            if (request == null)
            {
                result.Message += "Error: request is null.";
                return result;
            }

            try
            {
                var queryResult = await _openSearchService.QueryMemoryTurnWindowAsync(request);
                result.Success = queryResult.Success;
                result.Message += queryResult.Message;
                _logger.LogInformation(result.Message);
            }
            catch (Exception e)
            {
                result.Success = false;
                result.Message += $"Error: Failed to query memory turn window. Error was: {e.Message}";
                _logger.LogError(result.Message);
            }
            return result;
        }

        public async Task<ResultObj> QueryMemoryTurnRange(MemoryTurnRangeRequest? request)
        {
            var result = new ResultObj();
            result.Success = false;
            result.Message = "MessageAPI: QueryMemoryTurnRange: ";
            if (request == null)
            {
                result.Message += "Error: request is null.";
                return result;
            }

            try
            {
                var queryResult = await _openSearchService.QueryMemoryTurnRangeAsync(request);
                result.Success = queryResult.Success;
                result.Message += queryResult.Message;
                _logger.LogInformation(result.Message);
            }
            catch (Exception e)
            {
                result.Success = false;
                result.Message += $"Error: Failed to query memory turn range. Error was: {e.Message}";
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
