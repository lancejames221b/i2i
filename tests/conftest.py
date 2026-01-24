"""Shared pytest fixtures for i2i test suite."""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from i2i.schema import (
    Message,
    MessageType,
    Response,
    ConfidenceLevel,
    ConsensusLevel,
    EpistemicType,
)
from tests.fixtures.mock_providers import (
    MockProviderAdapter,
    MockProviderRegistry,
    create_mock_response,
    create_mock_message,
    AGREEING_RESPONSES,
    DISAGREEING_RESPONSES,
)


# ==================== Schema Fixtures ====================


@pytest.fixture
def sample_message():
    """Create a sample Message for testing."""
    return Message(
        type=MessageType.QUERY,
        content="What is the capital of France?",
        sender=None,
    )


@pytest.fixture
def sample_response():
    """Create a sample Response for testing."""
    return Response(
        message_id="test-msg-123",
        model="mock/test-model",
        content="Paris is the capital of France.",
        confidence=ConfidenceLevel.HIGH,
        input_tokens=10,
        output_tokens=25,
        latency_ms=150.5,
    )


@pytest.fixture
def multiple_responses():
    """Create multiple responses for consensus testing."""
    return [
        create_mock_response(
            model="provider1/model-a",
            content="Paris is the capital of France.",
            confidence=ConfidenceLevel.HIGH,
        ),
        create_mock_response(
            model="provider2/model-b",
            content="Paris is the capital of France, known as the City of Light.",
            confidence=ConfidenceLevel.HIGH,
        ),
        create_mock_response(
            model="provider3/model-c",
            content="The capital of France is Paris.",
            confidence=ConfidenceLevel.MEDIUM,
        ),
    ]


@pytest.fixture
def disagreeing_responses():
    """Create disagreeing responses for divergence testing."""
    return [
        create_mock_response(
            model="provider1/model-a",
            content="Python is the best programming language for beginners.",
            confidence=ConfidenceLevel.HIGH,
        ),
        create_mock_response(
            model="provider2/model-b",
            content="JavaScript is the best language for beginners due to web access.",
            confidence=ConfidenceLevel.HIGH,
        ),
        create_mock_response(
            model="provider3/model-c",
            content="Scratch is best for absolute beginners, Python for adults.",
            confidence=ConfidenceLevel.MEDIUM,
        ),
    ]


# ==================== Provider Fixtures ====================


@pytest.fixture
def mock_provider():
    """Create a single mock provider adapter."""
    return MockProviderAdapter(
        name="test-provider",
        models=["model-1", "model-2", "model-3"],
        configured=True,
    )


@pytest.fixture
def mock_registry():
    """Create a mock provider registry with multiple providers."""
    return MockProviderRegistry()


@pytest.fixture
def agreeing_registry():
    """Registry with providers that return agreeing responses."""
    adapter1 = MockProviderAdapter("p1", ["model-a"], responses={"model-a": AGREEING_RESPONSES["model-a"]})
    adapter2 = MockProviderAdapter("p2", ["model-b"], responses={"model-b": AGREEING_RESPONSES["model-b"]})
    adapter3 = MockProviderAdapter("p3", ["model-c"], responses={"model-c": AGREEING_RESPONSES["model-c"]})
    return MockProviderRegistry({"p1": adapter1, "p2": adapter2, "p3": adapter3})


@pytest.fixture
def disagreeing_registry():
    """Registry with providers that return disagreeing responses."""
    adapter1 = MockProviderAdapter("p1", ["model-a"], responses={"model-a": DISAGREEING_RESPONSES["model-a"]})
    adapter2 = MockProviderAdapter("p2", ["model-b"], responses={"model-b": DISAGREEING_RESPONSES["model-b"]})
    adapter3 = MockProviderAdapter("p3", ["model-c"], responses={"model-c": DISAGREEING_RESPONSES["model-c"]})
    return MockProviderRegistry({"p1": adapter1, "p2": adapter2, "p3": adapter3})


# ==================== Config Fixtures ====================


@pytest.fixture
def clean_env():
    """Ensure clean environment without I2I_ variables."""
    original_env = os.environ.copy()
    # Remove any I2I_ prefixed variables
    for key in list(os.environ.keys()):
        if key.startswith("I2I_"):
            del os.environ[key]
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_home_dir(temp_config_dir):
    """Mock Path.home() to return temp directory."""
    with patch("pathlib.Path.home", return_value=temp_config_dir):
        yield temp_config_dir


# ==================== API Key Fixtures ====================


@pytest.fixture
def mock_api_keys():
    """Set mock API keys in environment."""
    keys = {
        "OPENAI_API_KEY": "sk-test-openai-key",
        "ANTHROPIC_API_KEY": "sk-ant-test-anthropic-key",
        "GOOGLE_API_KEY": "test-google-key",
        "MISTRAL_API_KEY": "test-mistral-key",
        "GROQ_API_KEY": "gsk_test-groq-key",
        "COHERE_API_KEY": "test-cohere-key",
    }
    with patch.dict(os.environ, keys):
        yield keys


@pytest.fixture
def no_api_keys():
    """Ensure no API keys are set."""
    keys_to_remove = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
        "COHERE_API_KEY",
    ]
    original = {k: os.environ.get(k) for k in keys_to_remove}
    for key in keys_to_remove:
        os.environ.pop(key, None)
    yield
    for key, value in original.items():
        if value is not None:
            os.environ[key] = value


# ==================== Consensus Fixtures ====================


@pytest.fixture
def high_consensus_responses():
    """Responses that should produce HIGH consensus."""
    base_content = "The Earth orbits the Sun in approximately 365.25 days."
    return [
        create_mock_response(model="m1", content=base_content),
        create_mock_response(model="m2", content=base_content),
        create_mock_response(model="m3", content="The Earth takes about 365.25 days to orbit the Sun."),
    ]


@pytest.fixture
def low_consensus_responses():
    """Responses that should produce LOW consensus."""
    return [
        create_mock_response(model="m1", content="AI will surpass human intelligence by 2030."),
        create_mock_response(model="m2", content="AI may never truly achieve human-level intelligence."),
        create_mock_response(model="m3", content="The timeline for AGI is highly uncertain."),
    ]


# ==================== Epistemic Fixtures ====================


@pytest.fixture
def answerable_questions():
    """Questions that should classify as ANSWERABLE."""
    return [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the chemical formula for water?",
        "When did World War II end?",
    ]


@pytest.fixture
def uncertain_questions():
    """Questions that should classify as UNCERTAIN."""
    return [
        "Will it rain tomorrow in New York?",
        "What will the stock market do next week?",
        "Will this startup succeed?",
    ]


@pytest.fixture
def idle_questions():
    """Questions that should classify as IDLE."""
    return [
        "Is consciousness substrate-independent?",
        "Do we have free will?",
        "What is the meaning of life?",
        "Does God exist?",
    ]


@pytest.fixture
def malformed_questions():
    """Questions that should classify as MALFORMED."""
    return [
        "What color is the square circle?",
        "When will yesterday happen tomorrow?",
        "How many corners does a round square have?",
    ]


# ==================== Router Fixtures ====================


@pytest.fixture
def code_queries():
    """Queries that should classify as CODE_GENERATION."""
    return [
        "Write a Python function to sort a list",
        "Implement a binary search tree in JavaScript",
        "Create a REST API endpoint in FastAPI",
    ]


@pytest.fixture
def math_queries():
    """Queries that should classify as MATHEMATICAL."""
    return [
        "Calculate the derivative of x^3 + 2x",
        "Solve for x: 2x + 5 = 13",
        "What is the integral of sin(x)?",
    ]


@pytest.fixture
def creative_queries():
    """Queries that should classify as CREATIVE_WRITING."""
    return [
        "Write a poem about autumn",
        "Create a short story about a time traveler",
        "Compose a haiku about programming",
    ]
