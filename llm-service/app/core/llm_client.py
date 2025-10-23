import aiohttp
import asyncio
from typing import List, Optional
from exceptions import LLMClientError, LLMTimeoutError, LLMUnavailableError


class LLMClient:
    def __init__(self, ollama_urls: List[str] = None):
        self.ollama_urls = ollama_urls or ["http://localhost:11434"]

        self.session: Optional[aiohttp.ClientSession] = None

        self.current_url_index = 0

    async def initialize(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def generate(self, prompt: str) -> str:
        if self.session is None:
            await self.initialize()

        url = self.ollama_urls[0] + "/api/generate"

        data = {
            "model": "llama3.2:3b-instruct-q4_K_M",
            "prompt": prompt,
            "stream": False
        }

        try:
            async with self.session.post(url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "")
                else:
                    error = await response.text()
                    raise LLMUnavailableError(url, f'HTTP status {response.status}: {error}')

        except aiohttp.ClientError:
            raise LLMUnavailableError(url, "Failed to connect to server")
        except asyncio.Timeout:
            raise LLMTimeoutError("generate", 60.0)
        except Exception as e:
            raise LLMClientError(f'Unknown error: {str(e)}')

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None