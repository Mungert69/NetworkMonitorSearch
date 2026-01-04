# Services

Service-layer components for NetworkMonitorSearch. This folder hosts orchestrators, background workflows, and integration adapters used by the app entry points.\n\nCommon responsibilities:\n- Long-running workflows and background tasks\n- RabbitMQ listeners, schedulers, or API clients\n- Coordination between data, messaging, and external services\n\nWhen adding services, register them in the DI container (Program.cs, Startup.cs, or MauiProgram.cs).
