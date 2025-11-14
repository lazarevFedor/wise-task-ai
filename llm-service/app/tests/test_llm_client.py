import pytest
from unittest.mock import AsyncMock, patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_client


class TestLLMClient:
    @pytest.fixture
    def llm_client(self):
        """Create an instance of LLMClient"""
        return llm_client.LLMClient()

    @pytest.fixture
    def mock_session(self):
        """Mock client session"""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            mock_session.closed = False

            mock_post_context_manager = AsyncMock()
            mock_response = AsyncMock()

            mock_post_context_manager.__aenter__.return_value = mock_response
            mock_post_context_manager.__aexit__.return_value = None

            yield mock_session

    def test_init_with_default_values(self):
        """Test init with default values."""
        client = llm_client.LLMClient()
        assert client.ollama_urls == ['http://ollama:11434']
        assert client.max_concurrent_requests == 3
        assert client.request_timeout == 120.0

    @pytest.mark.asyncio
    async def test_initialize_creates_session(self, llm_client, mock_session):
        """Test creates session"""
        await llm_client.initialize()
        assert llm_client.session is not None

    @pytest.mark.asyncio
    async def test_generate_without_session_raises_error(self, llm_client):
        """Test for generation without session raises error"""
        with pytest.raises(RuntimeError) as exc_info:
            await llm_client.generate('Question')

        assert 'async context manager' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_close_closes_session(self, llm_client, mock_session):
        """Test for close session closes correctly"""
        await llm_client.initialize()
        await llm_client.close()

        assert mock_session.close.called
        assert llm_client.session is None
