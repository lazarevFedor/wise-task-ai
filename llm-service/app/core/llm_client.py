import aiohttp
import asyncio
from typing import List, Optional
from exceptions import LLMClientError, LLMTimeoutError, LLMUnavailableError


class LLMClient:
    def __init__(self, ollama_urls: List[str] = None, max_concurrent_requests: int = 3, request_timeout: float = 60.0):
        self.ollama_urls = ollama_urls or ["http://localhost:11434"]
        if not self.ollama_urls:
            raise ValueError('No ollama urls provided')
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.request_timeout = request_timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.current_url_index = 0
        self._lock = asyncio.Lock()
        self._request_counter = 0
        self._error_counter = 0


    async def initialize(self):
        """Initialise the session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def _get_next_url(self) -> Optional[str]:
        """Gets the next url from the ollama_urls list."""
        async with self._lock:
            url = self.ollama_urls[self.current_url_index]
            self.current_url_index = (self.current_url_index + 1) % len(self.ollama_urls)
            return url

    async def generate(self, prompt: str, model: str = None) -> str:
        """Generates answer from the given prompt."""
        base_url = await self._get_next_url()
        url = f'{base_url}/api/generate'

        if self.session is None:
            raise RuntimeError(
                'LLMClient must be used as async context manager. '
                'Use: async with LLMClient() as client:'
            )

        data = {
            'model': model or "llama3.2:3b-instruct-q4_K_M",
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
                        raise LLMUnavailableError(
                            url=url,
                            error_description=f'HTTP {response.status}: {error_description}',
                        )
        except asyncio.TimeoutError:
            self._error_counter += 1
            raise LLMTimeoutError('generate', self.request_timeout)
        except aiohttp.ClientConnectorError as e:
            self._error_counter += 1
            raise LLMUnavailableError(url, f'Connection error: {e}')
        except Exception as e:
            self._error_counter += 1
            raise LLMClientError(f'Unknown error: {str(e)}')

    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def __aenter__(self):
        """Async context manager."""
        try:
            await self.initialize()
            return self
        except Exception as e:
            await self.close()
            raise LLMClientError(f'Failed to initialize LLMClient: {str(e)}')

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager."""
        await self.close()
        if exc_type is not None:
            # Место для лога. Убрать return None
            return None
        else:
            return False
