import os
from pathlib import Path
from dotenv import load_dotenv


env_path = Path(__name__).parent.parent.parent / '.env'
load_dotenv(env_path)

class Config:
    LLM_GRPC_HOST = os.getenv('LLM_GRPC_HOST', 'localhost')
    LLM_GRPC_PORT = int(os.getenv('LLM_GRPC_PORT', '8081'))

    LLM_OLLAMA_URLS = os.getenv('LLM_OLLAMA_URLS', 'http://localhost:11434').split(',')
    LLM_DEFAULT_MODEL = os.getenv('LLM_DEFAULT_MODEL', 'llama3.2:3b-instruct-q4_K_M')

    LLM_MAX_CONCURRENT_REQUESTS = int(os.getenv('LLM_MAX_CONCURRENT_REQUESTS', '3'))
    LLM_REQUEST_TIMEOUT = float(os.getenv('LLM_REQUEST_TIMEOUT', '120.0'))

    @classmethod
    def print_config(cls):
        from logger import get_logger
        logger = get_logger(__name__)

        logger.info('App config:')
        logger.info(f'gRPC server: {cls.LLM_GRPC_HOST}:{cls.LLM_GRPC_PORT}')
        logger.info(f'Ollama URLs: {cls.LLM_OLLAMA_URLS}')
        logger.info(f'Max concurrent requests: {cls.LLM_MAX_CONCURRENT_REQUESTS}')
        logger.info(f'Request timeout: {cls.LLM_REQUEST_TIMEOUT}')
        logger.info(f'Default model: {cls.LLM_DEFAULT_MODEL}')

config = Config()