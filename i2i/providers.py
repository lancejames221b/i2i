"""
Provider adapters for different AI APIs.

This module provides a unified interface for querying different AI models,
abstracting away the differences in their APIs.
"""

import os
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from .schema import Message, Response, ConfidenceLevel

load_dotenv()


class ProviderAdapter(ABC):
    """Abstract base class for AI provider adapters."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """Return list of available model identifiers."""
        pass

    @abstractmethod
    async def query(self, message: Message, model: str) -> Response:
        """Send a message and get a response."""
        pass

    def is_configured(self) -> bool:
        """Check if the provider is properly configured."""
        return True


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI models (GPT-4, GPT-3.5, etc.)."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def available_models(self) -> List[str]:
        return [
            # GPT-5.x Series (Flagship - Dec 2025)
            "gpt-5.2",                    # Latest flagship model
            "gpt-5.2-chat-latest",        # ChatGPT optimized
            "gpt-5",                      # Previous flagship (aliases to gpt-5-2025-08-07)
            "gpt-5-mini",                 # Fast/affordable GPT-5
            "gpt-5-pro",                  # Enhanced reasoning
            # O-Series Reasoning Models (Apr 2025)
            "o3",                         # Flagship reasoning model
            "o3-pro",                     # Most intelligent - deep thinking
            "o4-mini",                    # Fast cost-efficient reasoning
            # GPT-4.1 Series (Apr 2025)
            "gpt-4.1",                    # Optimized for coding, 1M context
            "gpt-4.1-mini",               # Balanced performance
            "gpt-4.1-nano",               # Fastest/cheapest
            # Coding Models
            "codex-mini-latest",          # Codex CLI optimized
            # Legacy (still available)
            "gpt-4o",                     # Previous gen
            "gpt-4o-mini",                # Previous gen mini
        ]

    def is_configured(self) -> bool:
        return self.api_key is not None

    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def query(self, message: Message, model: str) -> Response:
        start_time = time.time()

        # Build messages array
        messages = []
        if message.context:
            for ctx_msg in message.context:
                role = "assistant" if ctx_msg.sender else "user"
                messages.append({"role": role, "content": ctx_msg.content})
        messages.append({"role": "user", "content": message.content})

        # Add system prompt for protocol awareness
        system_prompt = """You are participating in an AI-to-AI communication protocol.
When responding:
1. Be precise and substantive
2. Explicitly state your confidence level
3. Note any caveats or limitations
4. If challenging or verifying, be specific about what you agree/disagree with"""

        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.7,
        )

        latency = (time.time() - start_time) * 1000

        return Response(
            message_id=message.id,
            model=f"openai/{model}",
            content=response.choices[0].message.content,
            confidence=self._extract_confidence(response.choices[0].message.content),
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency,
        )

    def _extract_confidence(self, content: str) -> ConfidenceLevel:
        """Heuristically extract confidence from response content."""
        content_lower = content.lower()
        if any(phrase in content_lower for phrase in ["i'm certain", "definitely", "absolutely", "i'm confident"]):
            return ConfidenceLevel.VERY_HIGH
        elif any(phrase in content_lower for phrase in ["i believe", "likely", "probably"]):
            return ConfidenceLevel.HIGH
        elif any(phrase in content_lower for phrase in ["i think", "possibly", "might"]):
            return ConfidenceLevel.MEDIUM
        elif any(phrase in content_lower for phrase in ["i'm not sure", "uncertain", "hard to say"]):
            return ConfidenceLevel.LOW
        elif any(phrase in content_lower for phrase in ["i don't know", "impossible to determine"]):
            return ConfidenceLevel.VERY_LOW
        return ConfidenceLevel.MEDIUM


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic models (Claude)."""

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def available_models(self) -> List[str]:
        return [
            "claude-opus-4-5-20251101",    # Most intelligent (Nov 2025)
            "claude-sonnet-4-5-20250929",  # Best for agents & coding (Sep 2025)
            "claude-haiku-4-5-20251001",   # Fastest (Oct 2025)
            "claude-opus-4-20250514",      # Legacy
            "claude-sonnet-4-20250514",    # Legacy
        ]

    def is_configured(self) -> bool:
        return self.api_key is not None

    @property
    def client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def query(self, message: Message, model: str) -> Response:
        start_time = time.time()

        # Build messages array
        messages = []
        if message.context:
            for ctx_msg in message.context:
                role = "assistant" if ctx_msg.sender else "user"
                messages.append({"role": role, "content": ctx_msg.content})
        messages.append({"role": "user", "content": message.content})

        system_prompt = """You are participating in an AI-to-AI communication protocol.
When responding:
1. Be precise and substantive
2. Explicitly state your confidence level
3. Note any caveats or limitations
4. If challenging or verifying, be specific about what you agree/disagree with"""

        response = await self.client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )

        latency = (time.time() - start_time) * 1000

        content = response.content[0].text if response.content else ""

        return Response(
            message_id=message.id,
            model=f"anthropic/{model}",
            content=content,
            confidence=self._extract_confidence(content),
            input_tokens=response.usage.input_tokens if response.usage else None,
            output_tokens=response.usage.output_tokens if response.usage else None,
            latency_ms=latency,
        )

    def _extract_confidence(self, content: str) -> ConfidenceLevel:
        """Heuristically extract confidence from response content."""
        content_lower = content.lower()
        # Claude tends to express more epistemic humility
        if any(phrase in content_lower for phrase in ["i'm certain", "definitely", "absolutely"]):
            return ConfidenceLevel.HIGH  # Claude rarely says "very high"
        elif any(phrase in content_lower for phrase in ["i believe", "likely"]):
            return ConfidenceLevel.MEDIUM
        elif any(phrase in content_lower for phrase in ["i think", "possibly", "might", "uncertain"]):
            return ConfidenceLevel.LOW
        elif any(phrase in content_lower for phrase in ["i don't know", "genuinely uncertain", "hard to say"]):
            return ConfidenceLevel.VERY_LOW
        return ConfidenceLevel.MEDIUM


class GoogleAdapter(ProviderAdapter):
    """Adapter for Google models (Gemini)."""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self._configured = False

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def available_models(self) -> List[str]:
        return [
            "gemini-3-pro-preview",      # Latest Gemini 3 (Jan 2026)
            "gemini-3-flash-preview",    # Fast Gemini 3
            "gemini-3-deep-think-preview",  # Deep reasoning
            "gemini-1.5-pro", "gemini-1.5-flash",  # Legacy
        ]

    def is_configured(self) -> bool:
        return self.api_key is not None

    async def query(self, message: Message, model: str) -> Response:
        import google.generativeai as genai

        start_time = time.time()

        genai.configure(api_key=self.api_key)
        gen_model = genai.GenerativeModel(model)

        # Build conversation
        prompt = message.content
        if message.context:
            context_str = "\n".join([f"Previous: {m.content}" for m in message.context])
            prompt = f"{context_str}\n\nCurrent query: {message.content}"

        response = await asyncio.to_thread(
            gen_model.generate_content,
            prompt
        )

        latency = (time.time() - start_time) * 1000

        content = response.text if response.text else ""

        return Response(
            message_id=message.id,
            model=f"google/{model}",
            content=content,
            confidence=ConfidenceLevel.MEDIUM,
            latency_ms=latency,
        )


class MistralAdapter(ProviderAdapter):
    """Adapter for Mistral models."""

    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self._client = None

    @property
    def provider_name(self) -> str:
        return "mistral"

    @property
    def available_models(self) -> List[str]:
        return [
            "mistral-large-3",           # Latest flagship (Jan 2026)
            "devstral-2",                # Coding specialist
            "ministral-3-14b",           # Fast compact model
            "mistral-large-latest",      # Legacy alias
            "codestral-latest",          # Legacy coding
        ]

    def is_configured(self) -> bool:
        return self.api_key is not None

    @property
    def client(self):
        if self._client is None:
            from mistralai import Mistral
            self._client = Mistral(api_key=self.api_key)
        return self._client

    async def query(self, message: Message, model: str) -> Response:
        start_time = time.time()

        messages = []
        if message.context:
            for ctx_msg in message.context:
                role = "assistant" if ctx_msg.sender else "user"
                messages.append({"role": role, "content": ctx_msg.content})
        messages.append({"role": "user", "content": message.content})

        response = await asyncio.to_thread(
            self.client.chat.complete,
            model=model,
            messages=messages,
        )

        latency = (time.time() - start_time) * 1000

        content = response.choices[0].message.content if response.choices else ""

        return Response(
            message_id=message.id,
            model=f"mistral/{model}",
            content=content,
            confidence=ConfidenceLevel.MEDIUM,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency,
        )


class GroqAdapter(ProviderAdapter):
    """Adapter for Groq (fast Llama inference)."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self._client = None

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def available_models(self) -> List[str]:
        return [
            "meta-llama/llama-4-maverick-17b-128e-instruct",  # Latest Llama 4 (Jan 2026)
            "llama-3.3-70b-versatile",   # Stable large model
            "llama-3.1-8b-instant",      # Fast small model
            "mixtral-8x7b-32768",        # MoE model
        ]

    def is_configured(self) -> bool:
        return self.api_key is not None

    @property
    def client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    async def query(self, message: Message, model: str) -> Response:
        start_time = time.time()

        messages = []
        if message.context:
            for ctx_msg in message.context:
                role = "assistant" if ctx_msg.sender else "user"
                messages.append({"role": role, "content": ctx_msg.content})
        messages.append({"role": "user", "content": message.content})

        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=model,
            messages=messages,
        )

        latency = (time.time() - start_time) * 1000

        content = response.choices[0].message.content if response.choices else ""

        return Response(
            message_id=message.id,
            model=f"groq/{model}",
            content=content,
            confidence=ConfidenceLevel.MEDIUM,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency,
        )


class CohereAdapter(ProviderAdapter):
    """Adapter for Cohere models."""

    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY")
        self._client = None

    @property
    def provider_name(self) -> str:
        return "cohere"

    @property
    def available_models(self) -> List[str]:
        return [
            "command-a-03-2025",         # Latest Command A (Mar 2025)
            "command-a-reasoning",       # Enhanced reasoning
            "command-a-translate-08-2025",  # Multilingual specialist
            "command-r-plus", "command-r",  # Legacy
        ]

    def is_configured(self) -> bool:
        return self.api_key is not None

    @property
    def client(self):
        if self._client is None:
            import cohere
            self._client = cohere.Client(api_key=self.api_key)
        return self._client

    async def query(self, message: Message, model: str) -> Response:
        start_time = time.time()

        response = await asyncio.to_thread(
            self.client.chat,
            model=model,
            message=message.content,
        )

        latency = (time.time() - start_time) * 1000

        content = response.text if response.text else ""

        return Response(
            message_id=message.id,
            model=f"cohere/{model}",
            content=content,
            confidence=ConfidenceLevel.MEDIUM,
            latency_ms=latency,
        )


class OllamaAdapter(ProviderAdapter):
    """Adapter for Ollama local models."""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def available_models(self) -> List[str]:
        return [
            "llama3.2", "llama3.1", "llama2",
            "mistral", "mixtral",
            "codellama", "deepseek-coder",
            "phi3", "gemma2", "qwen2.5",
        ]

    def is_configured(self) -> bool:
        """Check if Ollama is running."""
        try:
            import httpx
            r = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def get_running_models(self) -> List[str]:
        """Fetch actually available models from Ollama."""
        try:
            import httpx
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                return [m["name"].split(":")[0] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    async def query(self, message: Message, model: str) -> Response:
        """Query Ollama /api/chat endpoint."""
        import httpx

        start_time = time.time()

        # Build messages array
        messages = []
        if message.context:
            for ctx_msg in message.context:
                role = "assistant" if ctx_msg.sender else "user"
                messages.append({"role": role, "content": ctx_msg.content})
        messages.append({"role": "user", "content": message.content})

        # Call Ollama API
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                }
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.time() - start_time) * 1000
        content = data.get("message", {}).get("content", "")

        return Response(
            message_id=message.id,
            model=f"ollama/{model}",
            content=content,
            confidence=self._extract_confidence(content),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            latency_ms=latency_ms,
        )

    def _extract_confidence(self, content: str) -> ConfidenceLevel:
        """Heuristically extract confidence from response content."""
        content_lower = content.lower()
        if any(phrase in content_lower for phrase in ["i'm certain", "definitely", "absolutely", "i'm confident"]):
            return ConfidenceLevel.VERY_HIGH
        elif any(phrase in content_lower for phrase in ["i believe", "likely", "probably"]):
            return ConfidenceLevel.HIGH
        elif any(phrase in content_lower for phrase in ["i think", "possibly", "might"]):
            return ConfidenceLevel.MEDIUM
        elif any(phrase in content_lower for phrase in ["i'm not sure", "uncertain", "hard to say"]):
            return ConfidenceLevel.LOW
        elif any(phrase in content_lower for phrase in ["i don't know", "impossible to determine"]):
            return ConfidenceLevel.VERY_LOW
        return ConfidenceLevel.MEDIUM


class LiteLLMAdapter(ProviderAdapter):
    """Adapter for LiteLLM proxy (unified access to 100+ LLMs)."""

    def __init__(self):
        self.api_base = os.getenv("LITELLM_API_BASE", "http://localhost:4000")
        self.api_key = os.getenv("LITELLM_API_KEY", "sk-1234")
        self._client = None
        self._models = None

    @property
    def provider_name(self) -> str:
        return "litellm"

    @property
    def available_models(self) -> List[str]:
        """Return available models from env var or fetch from proxy."""
        env_models = os.getenv("LITELLM_MODELS", "")
        if env_models:
            return [m.strip() for m in env_models.split(",") if m.strip()]
        return self._fetch_models()

    def _fetch_models(self) -> List[str]:
        """Fetch available models from LiteLLM /models endpoint."""
        if self._models is not None:
            return self._models
        try:
            import httpx
            r = httpx.get(
                f"{self.api_base}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5.0
            )
            if r.status_code == 200:
                data = r.json()
                self._models = [m["id"] for m in data.get("data", [])]
                return self._models
        except Exception:
            pass
        return []

    def is_configured(self) -> bool:
        """Check if LiteLLM proxy is reachable."""
        try:
            import httpx
            r = httpx.get(f"{self.api_base}/health", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=f"{self.api_base}/v1"
            )
        return self._client

    async def query(self, message: Message, model: str) -> Response:
        start_time = time.time()

        # Build messages array
        messages = []
        if message.context:
            for ctx_msg in message.context:
                role = "assistant" if ctx_msg.sender else "user"
                messages.append({"role": role, "content": ctx_msg.content})
        messages.append({"role": "user", "content": message.content})

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )

        latency = (time.time() - start_time) * 1000
        content = response.choices[0].message.content

        return Response(
            message_id=message.id,
            model=f"litellm/{model}",
            content=content,
            confidence=self._extract_confidence(content),
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency,
        )

    def _extract_confidence(self, content: str) -> ConfidenceLevel:
        """Heuristically extract confidence from response content."""
        content_lower = content.lower()
        if any(phrase in content_lower for phrase in ["i'm certain", "definitely", "absolutely", "i'm confident"]):
            return ConfidenceLevel.VERY_HIGH
        elif any(phrase in content_lower for phrase in ["i believe", "likely", "probably"]):
            return ConfidenceLevel.HIGH
        elif any(phrase in content_lower for phrase in ["i think", "possibly", "might"]):
            return ConfidenceLevel.MEDIUM
        elif any(phrase in content_lower for phrase in ["i'm not sure", "uncertain", "hard to say"]):
            return ConfidenceLevel.LOW
        elif any(phrase in content_lower for phrase in ["i don't know", "impossible to determine"]):
            return ConfidenceLevel.VERY_LOW
        return ConfidenceLevel.MEDIUM


class PerplexityAdapter(ProviderAdapter):
    """
    Adapter for Perplexity AI.

    Perplexity provides RAG-native models with built-in web search and citations.
    Uses OpenAI-compatible API at https://api.perplexity.ai
    """

    MODELS = ["sonar", "sonar-pro", "sonar-deep-research", "sonar-reasoning-pro"]

    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self._client = None

    @property
    def provider_name(self) -> str:
        return "perplexity"

    @property
    def available_models(self) -> List[str]:
        return self.MODELS

    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.perplexity.ai"
            )
        return self._client

    async def query(self, message: Message, model: str) -> Response:
        start_time = time.time()

        # Build messages
        messages = []
        if message.context:
            for ctx_msg in message.context:
                role = "assistant" if ctx_msg.sender else "user"
                messages.append({"role": role, "content": ctx_msg.content})
        messages.append({"role": "user", "content": message.content})

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,  # Lower for factual queries
        )

        latency = (time.time() - start_time) * 1000
        content = response.choices[0].message.content

        # Extract citations from response (Perplexity includes search_results)
        citations = None
        if hasattr(response, 'search_results') and response.search_results:
            citations = [r.get('url') for r in response.search_results if r.get('url')]

        return Response(
            message_id=message.id,
            model=f"perplexity/{model}",
            content=content,
            confidence=self._extract_confidence(content),
            citations=citations,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency,
        )

    def _extract_confidence(self, content: str) -> ConfidenceLevel:
        """Heuristically extract confidence from response content."""
        content_lower = content.lower()
        if any(phrase in content_lower for phrase in ["i'm certain", "definitely", "absolutely", "i'm confident"]):
            return ConfidenceLevel.VERY_HIGH
        elif any(phrase in content_lower for phrase in ["i believe", "likely", "probably"]):
            return ConfidenceLevel.HIGH
        elif any(phrase in content_lower for phrase in ["i think", "possibly", "might"]):
            return ConfidenceLevel.MEDIUM
        elif any(phrase in content_lower for phrase in ["i'm not sure", "uncertain", "hard to say"]):
            return ConfidenceLevel.LOW
        elif any(phrase in content_lower for phrase in ["i don't know", "impossible to determine"]):
            return ConfidenceLevel.VERY_LOW
        return ConfidenceLevel.MEDIUM


class ProviderRegistry:
    """
    Registry of all available AI providers.

    This class manages provider adapters and provides a unified
    interface for querying any supported model.
    """

    def __init__(self):
        self._adapters: Dict[str, ProviderAdapter] = {}
        self._model_to_provider: Dict[str, str] = {}

        # Register all adapters
        self._register_adapter(OpenAIAdapter())
        self._register_adapter(AnthropicAdapter())
        self._register_adapter(GoogleAdapter())
        self._register_adapter(MistralAdapter())
        self._register_adapter(GroqAdapter())
        self._register_adapter(CohereAdapter())
        self._register_adapter(OllamaAdapter())
        self._register_adapter(LiteLLMAdapter())
        self._register_adapter(PerplexityAdapter())

    def _register_adapter(self, adapter: ProviderAdapter):
        """Register a provider adapter."""
        self._adapters[adapter.provider_name] = adapter
        for model in adapter.available_models:
            self._model_to_provider[model] = adapter.provider_name
            # Also register with provider prefix
            self._model_to_provider[f"{adapter.provider_name}/{model}"] = adapter.provider_name

    def get_adapter(self, model: str) -> Optional[ProviderAdapter]:
        """Get the adapter for a given model."""
        # Check if model has provider prefix
        if "/" in model:
            provider_name = model.split("/")[0]
        else:
            provider_name = self._model_to_provider.get(model)

        if provider_name:
            return self._adapters.get(provider_name)
        return None

    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models by provider."""
        result = {}
        for provider_name, adapter in self._adapters.items():
            if adapter.is_configured():
                result[provider_name] = adapter.available_models
        return result

    def list_configured_providers(self) -> List[str]:
        """List providers that are properly configured."""
        return [name for name, adapter in self._adapters.items() if adapter.is_configured()]

    async def query(self, message: Message, model: str) -> Response:
        """Query a specific model."""
        adapter = self.get_adapter(model)
        if not adapter:
            raise ValueError(f"Unknown model: {model}")
        if not adapter.is_configured():
            raise ValueError(f"Provider {adapter.provider_name} is not configured (missing API key)")

        # Extract just the model name if it has provider prefix
        model_name = model.split("/")[-1] if "/" in model else model

        return await adapter.query(message, model_name)

    async def query_multiple(self, message: Message, models: List[str]) -> List[Response]:
        """Query multiple models in parallel."""
        tasks = [self.query(message, model) for model in models]
        return await asyncio.gather(*tasks, return_exceptions=True)
