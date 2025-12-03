import aiohttp
import asyncio
from typing import Optional
from exceptions import LLMClientError, LLMTimeoutError, LLMUnavailableError
from logger import get_logger
from config import config
from re import search


def cut_incomplete_sentence_smart(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    if search(r'[.!?…)"\u201d\u201c\u2019]\s*$', text):
        return text

    match = search(r'(.*[.!?…][)"\u201d\u201c\u2019]?)\s+', text)
    if match:
        return match.group(1)

    match2 = search(r'(.*[.!?…][)"\u201d\u201c\u2019}]?)', text)
    if match2:
        return match2.group(1)

    return text


class LLMClient:
    """
    Class LLMClient implements interaction with llama.cpp LLM.
    This client receives input prompt and context from the DB from Main-server,
    generates an answer using llama.cpp LLM and returns the response.
    """

    def __init__(self):
        """
        Initializes LLMClient.

        llama_urls:
        List of llama.cpp server URLs. Defaults to ['http://localhost:11343'].
        max_concurrent_requests:
        Maximum number of concurrent requests. Defaults to 3.
        request_timeout:
        Request timeout in seconds. Defaults to 120.0.
        """
        self.logger = get_logger(__name__)

        self.logger.debug(
            'Initializing LLMClient: max_concurrent_requests = %d',
            config.LLM_MAX_CONCURRENT_REQUESTS
        )

        if config.LLM_LLAMA_URLS is None:
            self.llama_urls = ['http://localhost:11343']
        else:
            if not config.LLM_LLAMA_URLS:
                raise ValueError('No llama.cpp urls provided')
            self.llama_urls = config.LLM_LLAMA_URLS

        self.max_concurrent_requests = config.LLM_MAX_CONCURRENT_REQUESTS
        self.semaphore = asyncio.Semaphore(config.LLM_MAX_CONCURRENT_REQUESTS)
        self.request_timeout = config.LLM_REQUEST_TIMEOUT
        self.session: Optional[aiohttp.ClientSession] = None
        self.current_url_index = 0
        self._lock = asyncio.Lock()
        self._request_counter = 0
        self._error_counter = 0
        self.logger.debug('Initializing LLMClient: client created. '
                          'List of llama.cpp URLs: %s',
                          str(self.llama_urls))

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
        Gets the next URL from the llama.cpp_urls list in a round-robin fashion.

        Uses a lock to ensure thread-safety for the index update.

        Returns:
            Optional[str]: The next llama URL, or None if no URLs are available.
        """
        async with self._lock:
            url = self.llama_urls[self.current_url_index]
            self.current_url_index = (self.current_url_index + 1) % len(self.llama_urls)
            return url

    async def generate(self, prompt: str,
                       model: str = None,
                       query_type: str = None) -> str:
        """
        Generates an answer from the given prompt using the llama.cpp.

        Selects the next available llama.cpp URL, sends a POST request to /api/generate,
        and handles various error conditions
        like timeouts, HTTP errors, and connection issues.

        Args:
            prompt (str): The input prompt for the LLM.
            model (str, optional): The model to use.
            query_type (str, optional): The query type to use.
            Defaults to 'Qwen2.5:3B-Instruct'.

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
        url = f'{base_url}/v1/completions'

        if self.session is None:
            raise RuntimeError(
                'LLMClient must be used as async context manager. '
                'Use: async with LLMClient() as client:'
            )

        temp = 0.3
        top_p = 0.5
        top_k = 20
        repeat_penalty = 1.05
        num_predict = 64

        if query_type == 'explanation':
            temp = 0.5
            top_p = 0.7
            top_k = 40
            repeat_penalty = 1.05
            num_predict = 256

        data = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'temperature': temp,
            'top_p': top_p,
            'top_k': top_k,
            'repeat_penalty': repeat_penalty,
            'num_predict': num_predict,
            'max_tokens': num_predict,
            'seed': 42,
            'stop': ['Ответ:']
        }

        self._request_counter += 1

        try:
            async with self.semaphore:
                async with self.session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return cut_incomplete_sentence_smart(
                            result.get('choices', [{}])[0].get('text', '').strip()
                        )
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

    async def health_check(self):
        """
        Performs a health check of the Llama CPP server.

        Sends a GET request to http://llama_cpp:11343/health and checks if the
        response status is 200. If not, raises an exception.
        Creates a separate session specifically for the health check.

        Raises:
            LLMTimeoutError: If the request times out.
            LLMUnavailableError: If the server is unavailable or returns non-200 status.
            LLMClientError: For other unknown errors.
        """
        health_url = "http://llama_cpp:11343/health"

        timeout = aiohttp.ClientTimeout(total=10.0)
        session = None

        try:
            session = aiohttp.ClientSession(timeout=timeout)

            response = await session.get(health_url)

            if response.status == 200:
                self.logger.debug('Health check passed: server is healthy')
                response.close()
                return
            else:
                error_description = await response.text()
                response.close()
                self.logger.error(f'Health check failed: {error_description}, '
                                  f'error code: {response.status}')
                raise LLMUnavailableError(
                    url=health_url,
                    error_description=f'HTTP {response.status}: {error_description}',
                )

        except asyncio.TimeoutError:
            self.logger.warning(f'Health check timeout: {timeout.total}')
            raise LLMTimeoutError('health_check', timeout.total)
        except aiohttp.ClientConnectorError as e:
            self.logger.error(f'Health check connection error: {e}')
            raise LLMUnavailableError(health_url, f'Connection error: {e}')
        except Exception as e:
            self.logger.error(f'Health check unknown error: {e}')
            raise LLMClientError(f'Unknown error during health check: {str(e)}')
        finally:
            if session and not session.closed:
                await session.close()
