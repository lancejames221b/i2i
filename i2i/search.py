"""
Search backends for RAG-grounded verification.

This module provides a pluggable search interface for grounding
verification with external sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    score: Optional[float] = None
    date: Optional[str] = None


class SearchBackend(ABC):
    """Abstract base class for search backends."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this search backend."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if this backend is properly configured."""
        pass

    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Execute a search query and return results."""
        pass


class BraveSearchBackend(SearchBackend):
    """
    Brave Search API backend.

    Requires BRAVE_API_KEY environment variable.
    Get an API key at: https://brave.com/search/api/
    """

    def __init__(self):
        self.api_key = os.getenv("BRAVE_API_KEY")
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    @property
    def backend_name(self) -> str:
        return "brave"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        import httpx

        if not self.is_configured():
            return []

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    self.base_url,
                    params={"q": query, "count": num_results},
                    headers={
                        "X-Subscription-Token": self.api_key,
                        "Accept": "application/json",
                    }
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for r in data.get("web", {}).get("results", []):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("description", ""),
                    ))
                return results
        except Exception:
            return []


class SerpAPIBackend(SearchBackend):
    """
    SerpAPI backend (Google Search results).

    Requires SERPAPI_API_KEY environment variable.
    Get an API key at: https://serpapi.com/
    """

    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")
        self.base_url = "https://serpapi.com/search"

    @property
    def backend_name(self) -> str:
        return "serpapi"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        import httpx

        if not self.is_configured():
            return []

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    self.base_url,
                    params={
                        "q": query,
                        "api_key": self.api_key,
                        "num": num_results,
                        "engine": "google",
                    }
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for r in data.get("organic_results", []):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("link", ""),
                        snippet=r.get("snippet", ""),
                    ))
                return results
        except Exception:
            return []


class TavilySearchBackend(SearchBackend):
    """
    Tavily Search API backend (AI-optimized search).

    Requires TAVILY_API_KEY environment variable.
    Get an API key at: https://tavily.com/
    """

    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com/search"

    @property
    def backend_name(self) -> str:
        return "tavily"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        import httpx

        if not self.is_configured():
            return []

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.base_url,
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": num_results,
                        "search_depth": "basic",
                    }
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for r in data.get("results", []):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("content", ""),
                        score=r.get("score"),
                    ))
                return results
        except Exception:
            return []


class SearchRegistry:
    """Registry of available search backends."""

    def __init__(self):
        self._backends: Dict[str, SearchBackend] = {}
        self._register_backend(BraveSearchBackend())
        self._register_backend(SerpAPIBackend())
        self._register_backend(TavilySearchBackend())

    def _register_backend(self, backend: SearchBackend):
        """Register a search backend."""
        self._backends[backend.backend_name] = backend

    def get_backend(self, name: str) -> Optional[SearchBackend]:
        """Get a backend by name."""
        return self._backends.get(name)

    def list_backends(self) -> List[str]:
        """List all registered backend names."""
        return list(self._backends.keys())

    def list_configured(self) -> List[str]:
        """List backends that are properly configured."""
        return [name for name, b in self._backends.items() if b.is_configured()]

    async def search(
        self,
        query: str,
        backend: Optional[str] = None,
        num_results: int = 5
    ) -> List[SearchResult]:
        """
        Search using specified or first available backend.

        Args:
            query: Search query string
            backend: Specific backend to use (optional)
            num_results: Number of results to return

        Returns:
            List of SearchResult objects
        """
        # Use specified backend if provided
        if backend:
            b = self._backends.get(backend)
            if b and b.is_configured():
                return await b.search(query, num_results)

        # Fallback to first configured backend
        for b in self._backends.values():
            if b.is_configured():
                return await b.search(query, num_results)

        return []
