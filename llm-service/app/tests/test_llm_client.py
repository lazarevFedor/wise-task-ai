import pytest
from unittest.mock import AsyncMock, patch
from ..llm_client import LLMClient


class TestLLMClient:
    @pytest.fixture
    def llm_client(self):
        """Create an instance of LLMClient"""
        return LLMClient(
            ollama_urls=['http://localhost:11434', 'http://localhost:11435'],
            max_concurrent_requests=2,
            request_timeout=30.0
        )

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
        client = LLMClient()
        assert client.ollama_urls == ['http://localhost:11434']
        assert client.max_concurrent_requests == 3
        assert client.request_timeout == 60.0

    def test_init_with_custom_values(self):
        """Test init with custom values"""
        client = LLMClient(
            ollama_urls=['http://custom:11435'],
            max_concurrent_requests=5,
            request_timeout=10.0
        )
        assert client.ollama_urls == ['http://custom:11435']
        assert client.max_concurrent_requests == 5
        assert client.request_timeout == 10.0

    def test_init_with_empty_urls_raises_error(self):
        """Test for empty urls raises error"""
        with pytest.raises(ValueError, match='No ollama urls provided'):
            LLMClient(ollama_urls=[])

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
