"""Mock provider adapters for testing."""

from typing import List, Optional, Dict, Any
from unittest.mock import AsyncMock, MagicMock
import uuid

from i2i.schema import Message, Response, ConfidenceLevel


class MockProviderAdapter:
    """Mock implementation of ProviderAdapter for testing."""

    def __init__(
        self,
        name: str = "mock",
        models: Optional[List[str]] = None,
        configured: bool = True,
        responses: Optional[Dict[str, str]] = None,
    ):
        self._name = name
        self._models = models or ["mock-model-1", "mock-model-2"]
        self._configured = configured
        self._responses = responses or {}
        self._call_count = 0
        self._calls: List[tuple] = []

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def available_models(self) -> List[str]:
        return self._models

    def is_configured(self) -> bool:
        return self._configured

    async def query(self, message: Message, model: str) -> Response:
        self._call_count += 1
        self._calls.append((message, model))

        # Return canned response if available
        content = self._responses.get(model, f"Mock response from {model}")

        return Response(
            id=str(uuid.uuid4()),
            message_id=message.id,
            model=f"{self._name}/{model}",
            content=content,
            confidence=ConfidenceLevel.MEDIUM,
            input_tokens=10,
            output_tokens=50,
            latency_ms=100.0,
        )


def create_mock_response(
    message_id: str = "test-msg-id",
    model: str = "mock/model",
    content: str = "Test response content",
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    input_tokens: int = 10,
    output_tokens: int = 50,
    latency_ms: float = 100.0,
) -> Response:
    """Helper to create mock Response objects."""
    return Response(
        id=str(uuid.uuid4()),
        message_id=message_id,
        model=model,
        content=content,
        confidence=confidence,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )


def create_mock_message(
    content: str = "Test query",
    sender: Optional[str] = None,
) -> Message:
    """Helper to create mock Message objects."""
    from i2i.schema import MessageType
    return Message(
        type=MessageType.QUERY,
        content=content,
        sender=sender,
    )


class MockProviderRegistry:
    """Mock registry for testing protocol flows."""

    def __init__(self, adapters: Optional[Dict[str, MockProviderAdapter]] = None):
        self._adapters = adapters or {
            "mock1": MockProviderAdapter("mock1", ["model-a", "model-b"]),
            "mock2": MockProviderAdapter("mock2", ["model-c", "model-d"]),
        }
        self._model_to_provider = {}
        for name, adapter in self._adapters.items():
            for model in adapter.available_models:
                self._model_to_provider[model] = name
                self._model_to_provider[f"{name}/{model}"] = name

    def get_adapter(self, model: str) -> Optional[MockProviderAdapter]:
        if "/" in model:
            provider_name = model.split("/")[0]
        else:
            provider_name = self._model_to_provider.get(model)
        return self._adapters.get(provider_name) if provider_name else None

    async def query(self, message: Message, model: str) -> Response:
        adapter = self.get_adapter(model)
        if not adapter:
            raise ValueError(f"Unknown model: {model}")
        model_name = model.split("/")[-1] if "/" in model else model
        return await adapter.query(message, model_name)

    async def query_multiple(
        self, message: Message, models: List[str]
    ) -> List[Response]:
        results = []
        for model in models:
            try:
                response = await self.query(message, model)
                results.append(response)
            except Exception as e:
                results.append(e)
        return results

    def list_available_models(self) -> Dict[str, List[str]]:
        return {name: adapter.available_models for name, adapter in self._adapters.items()}

    def list_configured_providers(self) -> List[str]:
        return [name for name, adapter in self._adapters.items() if adapter.is_configured()]


# Preset response sets for different test scenarios
AGREEING_RESPONSES = {
    "model-a": "Paris is the capital of France.",
    "model-b": "Paris is the capital of France.",
    "model-c": "Paris is the capital of France, known as the City of Light.",
}

DISAGREEING_RESPONSES = {
    "model-a": "The sky is blue due to Rayleigh scattering.",
    "model-b": "The sky appears blue because of atmospheric refraction.",
    "model-c": "The sky is actually not blue, it's an optical illusion.",
}

VERIFICATION_RESPONSES = {
    "model-a": """VERDICT: TRUE
The claim is accurate. The Eiffel Tower is indeed approximately 324 meters tall.""",
    "model-b": """VERDICT: TRUE
Verified. The Eiffel Tower stands at 324 meters including its antenna.""",
}

VERIFICATION_FALSE_RESPONSES = {
    "model-a": """VERDICT: FALSE
This is a common misconception. Napoleon was actually average height for his time (5'7").
ISSUES: Height measurement confusion between French and English inches.""",
    "model-b": """VERDICT: FALSE
Incorrect. Napoleon was not short by contemporary standards.
ISSUES: Propaganda from British cartoonists exaggerated this myth.""",
}
