from os import getenv
from logger import get_logger


class Config:
    LLM_GRPC_HOST = getenv('LLM_GRPC_HOST', 'localhost')
    LLM_GRPC_PORT = int(getenv('LLM_GRPC_PORT', '8084'))

    LLM_OLLAMA_URLS = getenv('LLM_OLLAMA_URLS', 'http://ollama:11434').split(',')
    LLM_DEFAULT_MODEL = getenv('LLM_DEFAULT_MODEL', 'llama3.2:3b-instruct-q4_K_M')

    LLM_MAX_CONCURRENT_REQUESTS = int(getenv('LLM_MAX_CONCURRENT_REQUESTS', '3'))
    LLM_REQUEST_TIMEOUT = float(getenv('LLM_REQUEST_TIMEOUT', '120.0'))

    @classmethod
    def print_config(cls):
        logger = get_logger(__name__)

        logger.info(f'App config:\n'
                    f'gRPC server: {cls.LLM_GRPC_HOST}:{cls.LLM_GRPC_PORT}\n'
                    f'Ollama URLs: {cls.LLM_OLLAMA_URLS}\n'
                    f'Max concurrent requests: {cls.LLM_MAX_CONCURRENT_REQUESTS}\n'
                    f'Request timeout: {cls.LLM_REQUEST_TIMEOUT}\n'
                    f'Default model: {cls.LLM_DEFAULT_MODEL}'
                    )


config = Config()
config.print_config()
