import asyncio
from core.logger import get_logger
from core.grpc_server import serve_grpc


logger = get_logger(__name__)


async def run_server():
    """Запуск сервера"""

    server = None
    try:
        logger.info('Starting LLM gRPC Server...')
        server = await serve_grpc()
        logger.info('LLM gRPC Server is running on localhost:50051')
        logger.info('Press Ctrl+C to stop the server...')
        await asyncio.Future()

    except KeyboardInterrupt:
        logger.info('Received Ctrl+C, shutting down...')
    except Exception as e:
        logger.error(f'Server error: {str(e)}')
        raise
    finally:
        if server:
            logger.info('Stopping server...')
            await server.stop(5)
            logger.info('Server stopped successfully')


def main():
    """Основная функция"""
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print('\nServer stopped by user')
    except Exception as e:
        logger.error(f'Fatal error: {str(e)}')


if __name__ == "__main__":
    main()