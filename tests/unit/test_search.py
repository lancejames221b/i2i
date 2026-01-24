"""Tests for i2i search backends."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from i2i.search import (
    SearchResult,
    SearchBackend,
    BraveSearchBackend,
    SerpAPIBackend,
    TavilySearchBackend,
    SearchRegistry,
)


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self):
        """Can create a SearchResult with required fields."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet"
        )
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"
        assert result.score is None
        assert result.date is None

    def test_create_search_result_with_optional_fields(self):
        """Can create a SearchResult with all fields."""
        result = SearchResult(
            title="Test",
            url="https://example.com",
            snippet="Snippet",
            score=0.95,
            date="2024-01-15"
        )
        assert result.score == 0.95
        assert result.date == "2024-01-15"


class TestBraveSearchBackend:
    """Tests for BraveSearchBackend."""

    @pytest.fixture
    def backend(self):
        """Create a BraveSearchBackend instance."""
        return BraveSearchBackend()

    def test_backend_name(self, backend):
        """Backend name should be 'brave'."""
        assert backend.backend_name == "brave"

    def test_is_configured_false_without_api_key(self, backend):
        """is_configured should return False without API key."""
        backend.api_key = None
        assert backend.is_configured() is False

    def test_is_configured_true_with_api_key(self):
        """is_configured should return True with API key."""
        with patch.dict("os.environ", {"BRAVE_API_KEY": "test-key"}):
            backend = BraveSearchBackend()
            assert backend.is_configured() is True

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_not_configured(self, backend):
        """search should return empty list when not configured."""
        backend.api_key = None
        results = await backend.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """search should return SearchResult objects."""
        backend = BraveSearchBackend()
        backend.api_key = "test-key"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Result 1", "url": "https://example.com/1", "description": "Desc 1"},
                    {"title": "Result 2", "url": "https://example.com/2", "description": "Desc 2"},
                ]
            }
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            results = await backend.search("test query", num_results=2)

            assert len(results) == 2
            assert results[0].title == "Result 1"
            assert results[0].url == "https://example.com/1"
            assert results[0].snippet == "Desc 1"


class TestSerpAPIBackend:
    """Tests for SerpAPIBackend."""

    @pytest.fixture
    def backend(self):
        """Create a SerpAPIBackend instance."""
        return SerpAPIBackend()

    def test_backend_name(self, backend):
        """Backend name should be 'serpapi'."""
        assert backend.backend_name == "serpapi"

    def test_is_configured_false_without_api_key(self, backend):
        """is_configured should return False without API key."""
        backend.api_key = None
        assert backend.is_configured() is False

    def test_is_configured_true_with_api_key(self):
        """is_configured should return True with API key."""
        with patch.dict("os.environ", {"SERPAPI_API_KEY": "test-key"}):
            backend = SerpAPIBackend()
            assert backend.is_configured() is True

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_not_configured(self, backend):
        """search should return empty list when not configured."""
        backend.api_key = None
        results = await backend.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """search should return SearchResult objects."""
        backend = SerpAPIBackend()
        backend.api_key = "test-key"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "organic_results": [
                {"title": "Result 1", "link": "https://example.com/1", "snippet": "Snippet 1"},
                {"title": "Result 2", "link": "https://example.com/2", "snippet": "Snippet 2"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            results = await backend.search("test query", num_results=2)

            assert len(results) == 2
            assert results[0].title == "Result 1"
            assert results[0].url == "https://example.com/1"
            assert results[0].snippet == "Snippet 1"


class TestTavilySearchBackend:
    """Tests for TavilySearchBackend."""

    @pytest.fixture
    def backend(self):
        """Create a TavilySearchBackend instance."""
        return TavilySearchBackend()

    def test_backend_name(self, backend):
        """Backend name should be 'tavily'."""
        assert backend.backend_name == "tavily"

    def test_is_configured_false_without_api_key(self, backend):
        """is_configured should return False without API key."""
        backend.api_key = None
        assert backend.is_configured() is False

    def test_is_configured_true_with_api_key(self):
        """is_configured should return True with API key."""
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
            backend = TavilySearchBackend()
            assert backend.is_configured() is True

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_not_configured(self, backend):
        """search should return empty list when not configured."""
        backend.api_key = None
        results = await backend.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """search should return SearchResult objects with scores."""
        backend = TavilySearchBackend()
        backend.api_key = "test-key"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Result 1", "url": "https://example.com/1", "content": "Content 1", "score": 0.95},
                {"title": "Result 2", "url": "https://example.com/2", "content": "Content 2", "score": 0.85},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            results = await backend.search("test query", num_results=2)

            assert len(results) == 2
            assert results[0].title == "Result 1"
            assert results[0].url == "https://example.com/1"
            assert results[0].snippet == "Content 1"
            assert results[0].score == 0.95


class TestSearchRegistry:
    """Tests for SearchRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a SearchRegistry instance."""
        return SearchRegistry()

    def test_registry_has_all_backends(self, registry):
        """Registry should have all backends registered."""
        backends = registry.list_backends()
        assert "brave" in backends
        assert "serpapi" in backends
        assert "tavily" in backends

    def test_get_backend_returns_correct_backend(self, registry):
        """get_backend should return the correct backend."""
        brave = registry.get_backend("brave")
        assert isinstance(brave, BraveSearchBackend)

        serpapi = registry.get_backend("serpapi")
        assert isinstance(serpapi, SerpAPIBackend)

        tavily = registry.get_backend("tavily")
        assert isinstance(tavily, TavilySearchBackend)

    def test_get_backend_returns_none_for_unknown(self, registry):
        """get_backend should return None for unknown backend."""
        assert registry.get_backend("unknown") is None

    def test_list_configured_returns_empty_without_keys(self, registry):
        """list_configured should return empty list without API keys."""
        configured = registry.list_configured()
        # No keys configured in test environment
        assert isinstance(configured, list)

    def test_list_configured_returns_configured_backends(self):
        """list_configured should return backends with API keys."""
        with patch.dict("os.environ", {"BRAVE_API_KEY": "test-key"}):
            registry = SearchRegistry()
            configured = registry.list_configured()
            assert "brave" in configured

    @pytest.mark.asyncio
    async def test_search_returns_empty_without_configured_backends(self, registry):
        """search should return empty list without configured backends."""
        results = await registry.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_uses_specified_backend(self):
        """search should use specified backend when provided."""
        registry = SearchRegistry()

        # Mock the brave backend as configured
        mock_brave = MagicMock()
        mock_brave.is_configured.return_value = True
        mock_brave.search = AsyncMock(return_value=[
            SearchResult(title="Test", url="https://test.com", snippet="Test")
        ])
        registry._backends["brave"] = mock_brave

        results = await registry.search("test query", backend="brave")

        assert len(results) == 1
        mock_brave.search.assert_called_once_with("test query", 5)

    @pytest.mark.asyncio
    async def test_search_fallback_to_first_configured(self):
        """search should fallback to first configured backend."""
        registry = SearchRegistry()

        # Mock serpapi as not configured, brave as configured
        mock_brave = MagicMock()
        mock_brave.is_configured.return_value = True
        mock_brave.search = AsyncMock(return_value=[
            SearchResult(title="Test", url="https://test.com", snippet="Test")
        ])

        mock_serpapi = MagicMock()
        mock_serpapi.is_configured.return_value = False

        registry._backends = {"serpapi": mock_serpapi, "brave": mock_brave}

        results = await registry.search("test query")

        assert len(results) == 1
        mock_brave.search.assert_called_once()
        mock_serpapi.search.assert_not_called()
