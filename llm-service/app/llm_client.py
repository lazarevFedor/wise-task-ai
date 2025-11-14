import aiohttp
import asyncio
from typing import Optional
from exceptions import LLMClientError, LLMTimeoutError, LLMUnavailableError
from logger import get_logger
from config import config


class LLMClient:
    """
    Class LLMClient implements interaction with Ollama LLM.
    This client receives input prompt and context from the DB from Main-server,
    generates an answer using Ollama LLM and returns the response.
    """

    def __init__(self):
        """
        Initializes LLMClient.

        ollama_urls:
        List of Ollama server URLs. Defaults to ['http://localhost:11434'].
        max_concurrent_requests:
        Maximum number of concurrent requests. Defaults to 3.
        request_timeout:
        Request timeout in seconds. Defaults to 120.0.
        """
        self.logger = get_logger(__name__)

        self.logger.info(
            'Initializing LLMClient: max_concurrent_requests = %d',
            config.LLM_MAX_CONCURRENT_REQUESTS
        )

        if config.LLM_OLLAMA_URLS is None:
            self.ollama_urls = ['http://localhost:11434']
        else:
            if not config.LLM_OLLAMA_URLS:
                raise ValueError('No ollama urls provided')
            self.ollama_urls = config.LLM_OLLAMA_URLS

        self.max_concurrent_requests = config.LLM_MAX_CONCURRENT_REQUESTS
        self.semaphore = asyncio.Semaphore(config.LLM_MAX_CONCURRENT_REQUESTS)
        self.request_timeout = config.LLM_REQUEST_TIMEOUT
        self.session: Optional[aiohttp.ClientSession] = None
        self.current_url_index = 0
        self._lock = asyncio.Lock()
        self._request_counter = 0
        self._error_counter = 0
        self.logger.info('Initializing LLMClient: client created. '
                         'List of Ollama URLs: %s', str(self.ollama_urls))

    async def initialize(self):
        """
        Initializes the client session
        with aiohttp ClientSession and timeout configuration.
        """
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def _get_next_url(self) -> Optional[str]:
        """
        Gets the next URL from the ollama_urls list in a round-robin fashion.

        Uses a lock to ensure thread-safety for the index update.

        Returns:
            Optional[str]: The next Ollama URL, or None if no URLs are available.
        """
        async with self._lock:
            url = self.ollama_urls[self.current_url_index]
            self.current_url_index = (self.current_url_index + 1) % len(self.ollama_urls)
            return url

    async def generate(self, prompt: str, model: str = None) -> str:
        """
        Generates an answer from the given prompt using the Ollama API.

        Selects the next available Ollama URL, sends a POST request to /api/generate,
        and handles various error conditions
        like timeouts, HTTP errors, and connection issues.

        Args:
            prompt (str): The input prompt for the LLM.
            model (str, optional): The model to use.
            Defaults to 'llama3.2:3b-instruct-q4_K_M'.

        Returns:
            str: The generated response from the LLM.

        Raises:
            RuntimeError:
            If the session is not initialized (use as async context manager).
            LLMTimeoutError:
            If the request times out.
            LLMUnavailableError:
            If the server is unavailable or returns an error.
            LLMClientError:
            For other unknown errors.
        """
        if model is None:
            model = config.LLM_DEFAULT_MODEL

        base_url = await self._get_next_url()
        url = f'{base_url}/api/generate'

        if self.session is None:
            raise RuntimeError(
                'LLMClient must be used as async context manager. '
                'Use: async with LLMClient() as client:'
            )

        data = {
            'model': model,
            'prompt': prompt,
            'stream': False,
        }

        self._request_counter += 1

        try:
            async with self.semaphore:
                async with self.session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '')
                    else:
                        error_description = await response.text()
                        self._error_counter += 1
                        self.logger.error(f'LLMClient: Got error: {error_description}, '
                                          f'error code: {response.status}')
                        raise LLMUnavailableError(
                            url=url,
                            error_description=f'HTTP {response.status}: '
                                              f'{error_description}',
                        )
        except asyncio.TimeoutError:
            self._error_counter += 1
            self.logger.warning(f'LLMClient: Got error: {asyncio.TimeoutError}, '
                                f'timeout: {self.request_timeout}')
            raise LLMTimeoutError('generate', self.request_timeout)
        except aiohttp.ClientConnectorError as e:
            self._error_counter += 1
            self.logger.error(f'LLMClient: Got error: {e}, error: {e}')
            raise LLMUnavailableError(url, f'Connection error: {e}')
        except Exception as e:
            self._error_counter += 1
            self.logger.error(f'LLMClient: Got error: {e}, error: {e}')
            raise LLMClientError(f'Unknown error: {str(e)}')

    async def close(self):
        """
        Closes the client session if it is open.
        """
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def __aenter__(self):
        """
        Async context manager entry point.

        Initializes the client and returns self.

        Returns:
            LLMClient: The initialized client instance.

        Raises:
            LLMClientError: If initialization fails.
        """
        try:
            await self.initialize()
            return self
        except Exception as e:
            await self.close()
            raise LLMClientError(f'Failed to initialize LLMClient: {str(e)}')

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit point.

        Closes the client and logs any exceptions.

        Args:
            exc_type: The exception type.
            exc_val: The exception value.
            exc_tb: The exception traceback.

        Returns:
            bool: False if an exception occurred, None otherwise.
        """
        await self.close()
        if exc_type is not None:
            self.logger.critical(f'LLMClient: Got error: {exc_type}, error: {exc_val}')
            return False
        else:
            return None
