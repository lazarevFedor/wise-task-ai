import asyncio
from logger import get_logger
from grpc_server import serve_grpc


logger = get_logger(__name__)


async def run_server():
    """
    Start the server.

    Initializes and starts the gRPC server using serve_grpc(),
    logs status messages, waits indefinitely for incoming requests,
    handles shutdown signals, and gracefully stops the server.
    """
    server = None
    try:
        logger.info('Starting LLM gRPC Server...')
        server = await serve_grpc()
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
    """
    Main function.

    Runs the asynchronous server startup routine using asyncio.run(),
    and handles top-level exceptions or user interruptions.
    """
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.debug('Server stopped gracefully')
    except Exception as e:
        logger.error(f'Fatal error: {str(e)}')


if __name__ == '__main__':
    main()
