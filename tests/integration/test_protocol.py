"""Integration tests for the AICP protocol."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from i2i.protocol import AICP
from i2i.schema import (
    ConsensusResult,
    ConsensusLevel,
    VerificationResult,
    EpistemicClassification,
    EpistemicType,
    ConfidenceLevel,
)
from i2i.router import RoutingResult, RoutingStrategy, TaskType
from tests.fixtures.mock_providers import (
    MockProviderRegistry,
    MockProviderAdapter,
    create_mock_response,
    AGREEING_RESPONSES,
    DISAGREEING_RESPONSES,
)


class TestAICPInitialization:
    """Tests for AICP initialization."""

    def test_creates_with_default_config(self):
        """AICP should initialize with default configuration."""
        with patch("i2i.protocol.ProviderRegistry") as MockRegistry:
            protocol = AICP()
            assert protocol is not None
            assert protocol.registry is not None

    def test_creates_with_custom_config(self):
        """AICP should accept custom configuration."""
        with patch("i2i.protocol.ProviderRegistry"):
            config = {"custom_key": "custom_value"}
            protocol = AICP(config=config)
            assert protocol.config["custom_key"] == "custom_value"


class TestConsensusQuery:
    """Integration tests for consensus queries."""

    @pytest.fixture
    def protocol_with_mock_registry(self):
        """Create AICP with mock registry."""
        adapter1 = MockProviderAdapter("p1", ["model-a"], responses={"model-a": AGREEING_RESPONSES["model-a"]})
        adapter2 = MockProviderAdapter("p2", ["model-b"], responses={"model-b": AGREEING_RESPONSES["model-b"]})
        registry = MockProviderRegistry({"p1": adapter1, "p2": adapter2})

        with patch("i2i.protocol.ProviderRegistry", return_value=registry):
            protocol = AICP()
            protocol.registry = registry
            protocol.consensus_engine.registry = registry
            return protocol

    async def test_consensus_query_returns_result(self, protocol_with_mock_registry):
        """consensus_query should return a ConsensusResult."""
        result = await protocol_with_mock_registry.consensus_query(
            query="What is the capital of France?",
            models=["model-a", "model-b"],
        )
        assert isinstance(result, ConsensusResult)
        assert result.query == "What is the capital of France?"

    async def test_consensus_query_includes_responses(self, protocol_with_mock_registry):
        """ConsensusResult should include individual responses."""
        result = await protocol_with_mock_registry.consensus_query(
            query="Test query",
            models=["model-a", "model-b"],
        )
        assert len(result.responses) == 2
        assert all(r.content for r in result.responses)

    async def test_consensus_query_determines_level(self, protocol_with_mock_registry):
        """ConsensusResult should have a consensus level."""
        result = await protocol_with_mock_registry.consensus_query(
            query="Test",
            models=["model-a", "model-b"],
        )
        assert result.consensus_level in list(ConsensusLevel)


class TestVerifyClaim:
    """Integration tests for claim verification."""

    @pytest.fixture
    def protocol_with_verifiers(self):
        """Create AICP with verification-capable registry."""
        adapter = MockProviderAdapter(
            "verifier", ["v-model"],
            responses={"v-model": "VERDICT: TRUE\nCONFIDENCE: HIGH\nThe claim is accurate."}
        )
        registry = MockProviderRegistry({"verifier": adapter})

        with patch("i2i.protocol.ProviderRegistry", return_value=registry):
            protocol = AICP()
            protocol.registry = registry
            protocol.verification_engine.registry = registry
            return protocol

    async def test_verify_claim_returns_result(self, protocol_with_verifiers):
        """verify_claim should return a VerificationResult."""
        result = await protocol_with_verifiers.verify_claim(
            claim="The Eiffel Tower is in Paris.",
            verifiers=["v-model"],
        )
        assert isinstance(result, VerificationResult)
        assert result.original_claim == "The Eiffel Tower is in Paris."

    async def test_verified_claim_has_verified_true(self, protocol_with_verifiers):
        """Verified claim should have verified=True."""
        result = await protocol_with_verifiers.verify_claim(
            claim="Test claim",
            verifiers=["v-model"],
        )
        assert result.verified is True


class TestClassifyQuestion:
    """Integration tests for epistemic classification."""

    @pytest.fixture
    def protocol_with_classifier(self):
        """Create AICP with classification-capable registry."""
        adapter = MockProviderAdapter(
            "classifier", ["c-model"],
            responses={"c-model": "CLASSIFICATION: answerable\nCONFIDENCE: 0.9\nREASONING: Factual question.\nIS_ACTIONABLE: yes"}
        )
        registry = MockProviderRegistry({"classifier": adapter})

        with patch("i2i.protocol.ProviderRegistry", return_value=registry):
            protocol = AICP()
            protocol.registry = registry
            protocol.epistemic_classifier.registry = registry
            return protocol

    async def test_classify_question_returns_result(self, protocol_with_classifier):
        """classify_question should return EpistemicClassification."""
        with patch("i2i.epistemic.get_epistemic_models", return_value=["c-model"]):
            result = await protocol_with_classifier.classify_question(
                question="What is 2 + 2?",
                classifiers=["c-model"],
            )
        assert isinstance(result, EpistemicClassification)
        assert result.question == "What is 2 + 2?"

    def test_quick_classify_no_api_calls(self, protocol_with_classifier):
        """quick_classify should work without API calls."""
        result = protocol_with_classifier.quick_classify("What is the capital of France?")
        assert result == EpistemicType.ANSWERABLE

    def test_quick_classify_idle_question(self, protocol_with_classifier):
        """quick_classify should identify idle questions."""
        # Use a question with exact pattern match for philosophical markers
        # Avoid "what is" prefix which triggers ANSWERABLE
        result = protocol_with_classifier.quick_classify("Does consciousness exist independently?")
        assert result == EpistemicType.IDLE


class TestSmartQuery:
    """Integration tests for smart_query workflow."""

    @pytest.fixture
    def protocol_with_full_mock(self):
        """Create AICP with fully mocked components."""
        adapter = MockProviderAdapter(
            "mock", ["model"],
            responses={"model": "Paris is the capital of France."}
        )
        registry = MockProviderRegistry({"mock": adapter})

        with patch("i2i.protocol.ProviderRegistry", return_value=registry):
            protocol = AICP()
            protocol.registry = registry
            return protocol

    async def test_smart_query_includes_classification(self, protocol_with_full_mock):
        """smart_query should include epistemic classification."""
        # Mock the classify_question method
        mock_classification = EpistemicClassification(
            question="Test",
            classification=EpistemicType.ANSWERABLE,
            confidence=0.9,
            reasoning="Factual question",
            is_actionable=True,
        )
        protocol_with_full_mock.classify_question = AsyncMock(return_value=mock_classification)

        # Mock consensus_query
        mock_consensus = ConsensusResult(
            query="Test",
            models_queried=["model"],
            responses=[create_mock_response(content="Answer")],
            consensus_level=ConsensusLevel.HIGH,
            consensus_answer="Answer",
        )
        protocol_with_full_mock.consensus_query = AsyncMock(return_value=mock_consensus)

        result = await protocol_with_full_mock.smart_query(
            query="What is the capital of France?",
            require_consensus=True,
        )

        assert "classification" in result
        assert result["classification"].classification == EpistemicType.ANSWERABLE

    async def test_smart_query_warns_on_malformed(self, protocol_with_full_mock):
        """smart_query should warn for malformed questions."""
        mock_classification = EpistemicClassification(
            question="Test",
            classification=EpistemicType.MALFORMED,
            confidence=0.9,
            reasoning="Self-contradictory",
            is_actionable=False,
        )
        protocol_with_full_mock.classify_question = AsyncMock(return_value=mock_classification)

        result = await protocol_with_full_mock.smart_query(
            query="What color is the square circle?",
            require_consensus=False,
        )

        assert any("malformed" in w.lower() for w in result["warnings"])


class TestListMethods:
    """Tests for listing methods."""

    def test_list_available_models(self):
        """list_available_models should return model dictionary."""
        adapter = MockProviderAdapter("test", ["model-1", "model-2"])
        registry = MockProviderRegistry({"test": adapter})

        with patch("i2i.protocol.ProviderRegistry", return_value=registry):
            protocol = AICP()
            protocol.registry = registry

        models = protocol.list_available_models()
        assert isinstance(models, dict)

    def test_list_configured_providers(self):
        """list_configured_providers should return provider list."""
        adapter = MockProviderAdapter("test", ["model"], configured=True)
        registry = MockProviderRegistry({"test": adapter})

        with patch("i2i.protocol.ProviderRegistry", return_value=registry):
            protocol = AICP()
            protocol.registry = registry

        providers = protocol.list_configured_providers()
        assert isinstance(providers, list)


class TestQueryMethods:
    """Tests for direct query methods."""

    @pytest.fixture
    def protocol(self):
        """Create AICP with mock registry."""
        adapter = MockProviderAdapter("test", ["model"], responses={"model": "Test response"})
        registry = MockProviderRegistry({"test": adapter})

        with patch("i2i.protocol.ProviderRegistry", return_value=registry):
            protocol = AICP()
            protocol.registry = registry
            return protocol

    async def test_query_single_model(self, protocol):
        """query should work with a single model."""
        response = await protocol.query(
            prompt="Hello",
            model="model",
        )
        assert response.content == "Test response"

    async def test_query_multiple_models(self, protocol):
        """query_multiple should return list of responses."""
        responses = await protocol.query_multiple(
            prompt="Hello",
            models=["model"],
        )
        assert isinstance(responses, list)
        assert len(responses) == 1


class TestRoutedQuery:
    """Tests for routed query functionality."""

    @pytest.fixture
    def protocol_with_router(self):
        """Create AICP with router-capable mock."""
        adapter = MockProviderAdapter("test", ["model"], responses={"model": "Routed response"})
        registry = MockProviderRegistry({"test": adapter})

        with patch("i2i.protocol.ProviderRegistry", return_value=registry):
            protocol = AICP()
            protocol.registry = registry
            return protocol

    def test_classify_task(self, protocol_with_router):
        """classify_task should return task type and confidence."""
        task_type, confidence = protocol_with_router.classify_task(
            "Write a Python function to sort a list"
        )
        assert task_type == TaskType.CODE_GENERATION
        assert confidence > 0.5

    def test_get_model_recommendation(self, protocol_with_router):
        """get_model_recommendation should return recommendations."""
        recommendations = protocol_with_router.get_model_recommendation(
            TaskType.CODE_GENERATION
        )
        assert isinstance(recommendations, dict)
