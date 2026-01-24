"""Tests for i2i provider adapters."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
import httpx

from i2i.providers import (
    OllamaAdapter,
    LiteLLMAdapter,
    ProviderRegistry,
)
from i2i.schema import Message, MessageType, Response, ConfidenceLevel


class TestOllamaAdapter:
    """Tests for OllamaAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create an OllamaAdapter instance."""
        return OllamaAdapter()

    def test_provider_name(self, adapter):
        """Provider name should be 'ollama'."""
        assert adapter.provider_name == "ollama"

    def test_available_models_includes_common_models(self, adapter):
        """Available models should include common Ollama models."""
        models = adapter.available_models
        assert "llama3.2" in models
        assert "mistral" in models
        assert "codellama" in models
        assert "phi3" in models

    def test_default_base_url(self, adapter):
        """Default base URL should be localhost:11434."""
        assert adapter.base_url == "http://localhost:11434"

    def test_custom_base_url_from_env(self):
        """Base URL should be configurable via environment variable."""
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://custom:8080"}):
            adapter = OllamaAdapter()
            assert adapter.base_url == "http://custom:8080"

    def test_is_configured_returns_false_when_not_running(self, adapter):
        """is_configured should return False when Ollama is not running."""
        adapter.base_url = "http://localhost:99999"  # Invalid port
        assert adapter.is_configured() is False

    def test_is_configured_returns_true_when_running(self, adapter):
        """is_configured should return True when Ollama responds."""
        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            assert adapter.is_configured() is True

    def test_is_configured_handles_connection_error(self, adapter):
        """is_configured should return False on connection errors."""
        with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            assert adapter.is_configured() is False

    def test_get_running_models_returns_list(self, adapter):
        """get_running_models should return list of model names."""
        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3.2:latest"},
                    {"name": "mistral:7b"},
                ]
            }
            mock_get.return_value = mock_response

            models = adapter.get_running_models()
            assert "llama3.2" in models
            assert "mistral" in models

    def test_get_running_models_returns_empty_on_error(self, adapter):
        """get_running_models should return empty list on error."""
        with patch("httpx.get", side_effect=Exception("Connection error")):
            models = adapter.get_running_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_query_returns_response(self, adapter):
        """query should return a Response object."""
        message = Message(
            type=MessageType.QUERY,
            content="What is 2+2?",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "The answer is 4."},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            response = await adapter.query(message, "llama3.2")

            assert isinstance(response, Response)
            assert response.content == "The answer is 4."
            assert response.model == "ollama/llama3.2"
            assert response.input_tokens == 10
            assert response.output_tokens == 5

    @pytest.mark.asyncio
    async def test_query_builds_correct_request(self, adapter):
        """query should build the correct API request."""
        message = Message(
            type=MessageType.QUERY,
            content="Hello",
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Hi there!"},
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            await adapter.query(message, "llama3.2")

            # Verify the API call
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "http://localhost:11434/api/chat" in str(call_args)

            # Check request body
            request_body = call_args.kwargs.get("json", call_args[1].get("json"))
            assert request_body["model"] == "llama3.2"
            assert request_body["stream"] is False
            assert len(request_body["messages"]) == 1
            assert request_body["messages"][0]["role"] == "user"
            assert request_body["messages"][0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_query_includes_context_messages(self, adapter):
        """query should include context messages in the request."""
        context_msg = Message(
            type=MessageType.QUERY,
            content="Previous message",
            sender="assistant",
        )
        message = Message(
            type=MessageType.QUERY,
            content="Follow-up question",
            context=[context_msg],
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Response"},
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            await adapter.query(message, "llama3.2")

            call_args = mock_client.post.call_args
            request_body = call_args.kwargs.get("json", call_args[1].get("json"))
            assert len(request_body["messages"]) == 2
            assert request_body["messages"][0]["role"] == "assistant"
            assert request_body["messages"][1]["role"] == "user"

    def test_extract_confidence_very_high(self, adapter):
        """_extract_confidence should detect VERY_HIGH confidence."""
        assert adapter._extract_confidence("I'm certain this is correct.") == ConfidenceLevel.VERY_HIGH
        assert adapter._extract_confidence("This is definitely true.") == ConfidenceLevel.VERY_HIGH

    def test_extract_confidence_high(self, adapter):
        """_extract_confidence should detect HIGH confidence."""
        assert adapter._extract_confidence("I believe this is accurate.") == ConfidenceLevel.HIGH
        assert adapter._extract_confidence("This is likely correct.") == ConfidenceLevel.HIGH

    def test_extract_confidence_medium(self, adapter):
        """_extract_confidence should detect MEDIUM confidence."""
        assert adapter._extract_confidence("I think this might work.") == ConfidenceLevel.MEDIUM

    def test_extract_confidence_low(self, adapter):
        """_extract_confidence should detect LOW confidence."""
        assert adapter._extract_confidence("I'm not sure about this.") == ConfidenceLevel.LOW

    def test_extract_confidence_very_low(self, adapter):
        """_extract_confidence should detect VERY_LOW confidence."""
        assert adapter._extract_confidence("I don't know the answer.") == ConfidenceLevel.VERY_LOW

    def test_extract_confidence_default(self, adapter):
        """_extract_confidence should default to MEDIUM."""
        assert adapter._extract_confidence("The answer is 42.") == ConfidenceLevel.MEDIUM


class TestProviderRegistryWithOllama:
    """Tests for ProviderRegistry including OllamaAdapter."""

    def test_registry_includes_ollama(self):
        """ProviderRegistry should include OllamaAdapter."""
        registry = ProviderRegistry()
        assert "ollama" in registry._adapters
        assert isinstance(registry._adapters["ollama"], OllamaAdapter)

    def test_registry_maps_ollama_models(self):
        """ProviderRegistry should map Ollama models to provider."""
        registry = ProviderRegistry()
        assert registry._model_to_provider.get("llama3.2") == "ollama"
        assert registry._model_to_provider.get("ollama/llama3.2") == "ollama"
        assert registry._model_to_provider.get("mistral") == "ollama"
        assert registry._model_to_provider.get("codellama") == "ollama"

    def test_get_adapter_for_ollama_model(self):
        """get_adapter should return OllamaAdapter for Ollama models."""
        registry = ProviderRegistry()
        adapter = registry.get_adapter("llama3.2")
        assert isinstance(adapter, OllamaAdapter)

    def test_get_adapter_with_provider_prefix(self):
        """get_adapter should work with provider/model format."""
        registry = ProviderRegistry()
        adapter = registry.get_adapter("ollama/llama3.2")
        assert isinstance(adapter, OllamaAdapter)

    def test_list_available_models_includes_ollama_when_configured(self):
        """list_available_models should include Ollama when running."""
        registry = ProviderRegistry()

        # Mock Ollama as configured
        with patch.object(registry._adapters["ollama"], "is_configured", return_value=True):
            models = registry.list_available_models()
            assert "ollama" in models
            assert "llama3.2" in models["ollama"]

    def test_list_configured_providers_includes_ollama_when_running(self):
        """list_configured_providers should include Ollama when running."""
        registry = ProviderRegistry()

        with patch.object(registry._adapters["ollama"], "is_configured", return_value=True):
            providers = registry.list_configured_providers()
            assert "ollama" in providers


class TestLiteLLMAdapter:
    """Tests for LiteLLMAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create a LiteLLMAdapter instance."""
        return LiteLLMAdapter()

    def test_provider_name(self, adapter):
        """Provider name should be 'litellm'."""
        assert adapter.provider_name == "litellm"

    def test_default_api_base(self, adapter):
        """Default API base should be localhost:4000."""
        assert adapter.api_base == "http://localhost:4000"

    def test_default_api_key(self, adapter):
        """Default API key should be set."""
        assert adapter.api_key == "sk-1234"

    def test_custom_api_base_from_env(self):
        """API base should be configurable via environment variable."""
        with patch.dict("os.environ", {"LITELLM_API_BASE": "http://proxy:8000"}):
            adapter = LiteLLMAdapter()
            assert adapter.api_base == "http://proxy:8000"

    def test_custom_api_key_from_env(self):
        """API key should be configurable via environment variable."""
        with patch.dict("os.environ", {"LITELLM_API_KEY": "my-secret-key"}):
            adapter = LiteLLMAdapter()
            assert adapter.api_key == "my-secret-key"

    def test_models_from_env(self):
        """Models should be configurable via environment variable."""
        with patch.dict("os.environ", {"LITELLM_MODELS": "gpt-4,claude-3,mistral"}):
            adapter = LiteLLMAdapter()
            models = adapter.available_models
            assert "gpt-4" in models
            assert "claude-3" in models
            assert "mistral" in models

    def test_models_from_env_handles_whitespace(self):
        """Models from env should handle whitespace."""
        with patch.dict("os.environ", {"LITELLM_MODELS": " gpt-4 , claude-3 "}):
            adapter = LiteLLMAdapter()
            models = adapter.available_models
            assert "gpt-4" in models
            assert "claude-3" in models

    def test_is_configured_returns_false_when_not_running(self, adapter):
        """is_configured should return False when proxy is not running."""
        adapter.api_base = "http://localhost:99999"  # Invalid port
        assert adapter.is_configured() is False

    def test_is_configured_returns_true_when_running(self, adapter):
        """is_configured should return True when proxy responds."""
        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            assert adapter.is_configured() is True

    def test_is_configured_handles_connection_error(self, adapter):
        """is_configured should return False on connection errors."""
        with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            assert adapter.is_configured() is False

    def test_fetch_models_returns_list(self, adapter):
        """_fetch_models should return list of model IDs."""
        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "gpt-4"},
                    {"id": "claude-3-opus"},
                ]
            }
            mock_get.return_value = mock_response

            models = adapter._fetch_models()
            assert "gpt-4" in models
            assert "claude-3-opus" in models

    def test_fetch_models_returns_empty_on_error(self, adapter):
        """_fetch_models should return empty list on error."""
        with patch("httpx.get", side_effect=Exception("Connection error")):
            models = adapter._fetch_models()
            assert models == []

    def test_fetch_models_caches_result(self, adapter):
        """_fetch_models should cache the result."""
        adapter._models = ["cached-model"]
        assert adapter._fetch_models() == ["cached-model"]

    @pytest.mark.asyncio
    async def test_query_returns_response(self, adapter):
        """query should return a Response object."""
        message = Message(
            type=MessageType.QUERY,
            content="What is 2+2?",
        )

        # Mock the OpenAI client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="The answer is 4."))]
        mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_completion

        with patch.object(adapter, '_client', mock_client):
            response = await adapter.query(message, "gpt-4")

            assert isinstance(response, Response)
            assert response.content == "The answer is 4."
            assert response.model == "litellm/gpt-4"
            assert response.input_tokens == 10
            assert response.output_tokens == 5

    @pytest.mark.asyncio
    async def test_query_builds_correct_messages(self, adapter):
        """query should build correct messages array."""
        context_msg = Message(
            type=MessageType.QUERY,
            content="Previous message",
            sender="assistant",
        )
        message = Message(
            type=MessageType.QUERY,
            content="Follow-up question",
            context=[context_msg],
        )

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_completion.usage = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_completion

        with patch.object(adapter, '_client', mock_client):
            await adapter.query(message, "gpt-4")

            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
            assert len(messages) == 2
            assert messages[0]["role"] == "assistant"
            assert messages[1]["role"] == "user"

    def test_extract_confidence_very_high(self, adapter):
        """_extract_confidence should detect VERY_HIGH confidence."""
        assert adapter._extract_confidence("I'm certain this is correct.") == ConfidenceLevel.VERY_HIGH

    def test_extract_confidence_high(self, adapter):
        """_extract_confidence should detect HIGH confidence."""
        assert adapter._extract_confidence("I believe this is accurate.") == ConfidenceLevel.HIGH

    def test_extract_confidence_medium(self, adapter):
        """_extract_confidence should detect MEDIUM confidence."""
        assert adapter._extract_confidence("I think this might work.") == ConfidenceLevel.MEDIUM

    def test_extract_confidence_low(self, adapter):
        """_extract_confidence should detect LOW confidence."""
        assert adapter._extract_confidence("I'm not sure about this.") == ConfidenceLevel.LOW

    def test_extract_confidence_very_low(self, adapter):
        """_extract_confidence should detect VERY_LOW confidence."""
        assert adapter._extract_confidence("I don't know the answer.") == ConfidenceLevel.VERY_LOW

    def test_extract_confidence_default(self, adapter):
        """_extract_confidence should default to MEDIUM."""
        assert adapter._extract_confidence("The answer is 42.") == ConfidenceLevel.MEDIUM


class TestProviderRegistryWithLiteLLM:
    """Tests for ProviderRegistry including LiteLLMAdapter."""

    def test_registry_includes_litellm(self):
        """ProviderRegistry should include LiteLLMAdapter."""
        registry = ProviderRegistry()
        assert "litellm" in registry._adapters
        assert isinstance(registry._adapters["litellm"], LiteLLMAdapter)

    def test_get_adapter_with_litellm_prefix(self):
        """get_adapter should work with litellm/model format."""
        registry = ProviderRegistry()
        adapter = registry.get_adapter("litellm/gpt-4")
        assert isinstance(adapter, LiteLLMAdapter)

    def test_list_available_models_includes_litellm_when_configured(self):
        """list_available_models should include LiteLLM when running."""
        registry = ProviderRegistry()

        with patch.object(registry._adapters["litellm"], "is_configured", return_value=True):
            with patch.object(type(registry._adapters["litellm"]), "available_models", new_callable=PropertyMock, return_value=["gpt-4", "claude-3"]):
                models = registry.list_available_models()
                assert "litellm" in models

    def test_list_configured_providers_includes_litellm_when_running(self):
        """list_configured_providers should include LiteLLM when running."""
        registry = ProviderRegistry()

        with patch.object(registry._adapters["litellm"], "is_configured", return_value=True):
            providers = registry.list_configured_providers()
            assert "litellm" in providers


# ==================== Perplexity Adapter Tests ====================

from i2i.providers import PerplexityAdapter


class TestPerplexityAdapter:
    """Tests for PerplexityAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create a PerplexityAdapter instance."""
        return PerplexityAdapter()

    def test_provider_name(self, adapter):
        """Provider name should be 'perplexity'."""
        assert adapter.provider_name == "perplexity"

    def test_available_models(self, adapter):
        """Should list all Perplexity models."""
        models = adapter.available_models
        assert "sonar" in models
        assert "sonar-pro" in models
        assert "sonar-deep-research" in models
        assert "sonar-reasoning-pro" in models

    def test_is_configured_returns_false_without_api_key(self, adapter):
        """is_configured should return False when no API key is set."""
        adapter.api_key = None
        assert adapter.is_configured() is False

    def test_is_configured_returns_true_with_api_key(self):
        """is_configured should return True when API key is set."""
        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "pplx-test-key"}):
            adapter = PerplexityAdapter()
            assert adapter.is_configured() is True

    def test_client_uses_perplexity_base_url(self):
        """Client should be configured with Perplexity base URL."""
        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "pplx-test-key"}):
            adapter = PerplexityAdapter()
            with patch("openai.AsyncOpenAI") as mock_openai:
                _ = adapter.client
                mock_openai.assert_called_once_with(
                    api_key="pplx-test-key",
                    base_url="https://api.perplexity.ai"
                )

    @pytest.mark.asyncio
    async def test_query_returns_response(self):
        """query should return a Response object."""
        adapter = PerplexityAdapter()
        adapter.api_key = "pplx-test-key"

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_response.search_results = [{"url": "https://example.com"}]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        adapter._client = mock_client

        message = Message(type=MessageType.QUERY, content="What is Python?")
        response = await adapter.query(message, "sonar")

        assert response.content == "Test response"
        assert response.model == "perplexity/sonar"
        assert response.input_tokens == 10
        assert response.output_tokens == 20

    @pytest.mark.asyncio
    async def test_query_extracts_citations(self):
        """query should extract citations from search_results."""
        adapter = PerplexityAdapter()
        adapter.api_key = "pplx-test-key"

        # Mock response with search_results
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Answer with citations"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_response.search_results = [
            {"url": "https://example.com/1"},
            {"url": "https://example.com/2"},
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        adapter._client = mock_client

        message = Message(type=MessageType.QUERY, content="Question")
        response = await adapter.query(message, "sonar-pro")

        assert response.citations == ["https://example.com/1", "https://example.com/2"]

    def test_extract_confidence_very_high(self, adapter):
        """Should extract VERY_HIGH confidence."""
        assert adapter._extract_confidence("I'm certain this is correct") == ConfidenceLevel.VERY_HIGH

    def test_extract_confidence_high(self, adapter):
        """Should extract HIGH confidence."""
        assert adapter._extract_confidence("I believe this is true") == ConfidenceLevel.HIGH

    def test_extract_confidence_medium(self, adapter):
        """Should extract MEDIUM confidence."""
        assert adapter._extract_confidence("I think it might be") == ConfidenceLevel.MEDIUM

    def test_extract_confidence_low(self, adapter):
        """Should extract LOW confidence."""
        assert adapter._extract_confidence("I'm not sure about this") == ConfidenceLevel.LOW

    def test_extract_confidence_very_low(self, adapter):
        """Should extract VERY_LOW confidence."""
        assert adapter._extract_confidence("I don't know the answer") == ConfidenceLevel.VERY_LOW


class TestProviderRegistryWithPerplexity:
    """Tests for ProviderRegistry with Perplexity integration."""

    def test_registry_includes_perplexity(self):
        """ProviderRegistry should include Perplexity adapter."""
        registry = ProviderRegistry()
        assert "perplexity" in registry._adapters

    def test_get_adapter_with_perplexity_prefix(self):
        """get_adapter should work with 'perplexity/' prefix."""
        registry = ProviderRegistry()
        adapter = registry.get_adapter("perplexity/sonar")
        assert isinstance(adapter, PerplexityAdapter)

    def test_list_configured_providers_includes_perplexity_when_configured(self):
        """list_configured_providers should include Perplexity when API key is set."""
        registry = ProviderRegistry()

        with patch.object(registry._adapters["perplexity"], "is_configured", return_value=True):
            providers = registry.list_configured_providers()
            assert "perplexity" in providers
