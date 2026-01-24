"""Tests for i2i model router and task classifier."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict

from i2i.router import (
    TaskType,
    RoutingStrategy,
    ModelCapability,
    RoutingDecision,
    TaskClassifier,
    ModelRouter,
)
from tests.fixtures.mock_providers import MockProviderRegistry, MockProviderAdapter


class TestTaskType:
    """Tests for TaskType enum."""

    def test_all_task_types_exist(self):
        """Verify key task types are defined."""
        # Reasoning
        assert TaskType.LOGICAL_REASONING == "logical_reasoning"
        assert TaskType.MATHEMATICAL == "mathematical"

        # Creative
        assert TaskType.CREATIVE_WRITING == "creative_writing"
        assert TaskType.COPYWRITING == "copywriting"

        # Technical
        assert TaskType.CODE_GENERATION == "code_generation"
        assert TaskType.CODE_REVIEW == "code_review"
        assert TaskType.CODE_DEBUGGING == "code_debugging"

        # Knowledge
        assert TaskType.FACTUAL_QA == "factual_qa"
        assert TaskType.RESEARCH == "research"
        assert TaskType.SUMMARIZATION == "summarization"

    def test_unknown_task_type_exists(self):
        """UNKNOWN task type exists for unclassified queries."""
        assert TaskType.UNKNOWN == "unknown"


class TestRoutingStrategy:
    """Tests for RoutingStrategy enum."""

    def test_all_strategies_exist(self):
        """Verify all routing strategies are defined."""
        assert RoutingStrategy.BEST_QUALITY == "best_quality"
        assert RoutingStrategy.BEST_SPEED == "best_speed"
        assert RoutingStrategy.BEST_VALUE == "best_value"
        assert RoutingStrategy.BALANCED == "balanced"
        assert RoutingStrategy.ENSEMBLE == "ensemble"
        assert RoutingStrategy.FALLBACK_CHAIN == "fallback_chain"


class TestModelCapability:
    """Tests for ModelCapability model."""

    def test_create_capability(self):
        """Create a ModelCapability with required fields."""
        cap = ModelCapability(
            model_id="test-model",
            provider="test-provider",
        )
        assert cap.model_id == "test-model"
        assert cap.provider == "test-provider"

    def test_default_values(self):
        """Check default values for ModelCapability."""
        cap = ModelCapability(model_id="test", provider="test")
        assert cap.avg_latency_ms == 1000.0
        assert cap.cost_per_1k_tokens == 0.01
        assert cap.context_window == 8192
        assert cap.supports_vision is False
        assert cap.supports_function_calling is False

    def test_task_scores(self):
        """Task scores can be set per task type."""
        cap = ModelCapability(
            model_id="test",
            provider="test",
            task_scores={
                TaskType.CODE_GENERATION: 95,
                TaskType.CREATIVE_WRITING: 80,
            },
        )
        assert cap.task_scores[TaskType.CODE_GENERATION] == 95
        assert cap.task_scores[TaskType.CREATIVE_WRITING] == 80


class TestTaskClassifier:
    """Tests for TaskClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create a TaskClassifier without registry."""
        return TaskClassifier()

    def test_classify_code_generation(self, classifier):
        """Code-related queries should classify as CODE_GENERATION."""
        # Use queries that match the exact patterns in TASK_PATTERNS
        code_queries = [
            "write code for a calculator",
            "implement a sorting algorithm",
            "create a function that filters data",
        ]
        for query in code_queries:
            task_type, confidence = classifier.classify(query)
            assert task_type == TaskType.CODE_GENERATION, f"Failed for: {query}"

    def test_classify_math(self, classifier, math_queries):
        """Math queries should classify as MATHEMATICAL."""
        for query in math_queries:
            task_type, confidence = classifier.classify(query)
            assert task_type == TaskType.MATHEMATICAL
            assert confidence > 0.5

    def test_classify_creative_writing(self, classifier):
        """Creative queries should classify as CREATIVE_WRITING."""
        # Use queries that match the exact patterns in TASK_PATTERNS
        creative_queries = [
            "write a story about a dragon",
            "write a poem about nature",
            "compose a haiku about coding",
        ]
        for query in creative_queries:
            task_type, confidence = classifier.classify(query)
            assert task_type == TaskType.CREATIVE_WRITING, f"Failed for: {query}"

    def test_classify_factual_qa(self, classifier):
        """Factual questions should classify as FACTUAL_QA."""
        # Use queries that exactly match TASK_PATTERNS
        queries = [
            "what is the capital of Japan?",
            "who is the president?",
            "tell me about the Roman Empire",
        ]
        for query in queries:
            task_type, _ = classifier.classify(query)
            assert task_type == TaskType.FACTUAL_QA, f"Failed for: {query}"

    def test_classify_code_debugging(self, classifier):
        """Debug queries should classify as CODE_DEBUGGING."""
        # Use queries that match CODE_DEBUGGING patterns without CODE_GENERATION conflicts
        queries = [
            "debug this application",
            "why doesn't this work",
            "fix this error: TypeError",
        ]
        for query in queries:
            task_type, _ = classifier.classify(query)
            assert task_type == TaskType.CODE_DEBUGGING, f"Failed for: {query}"

    def test_classify_summarization(self, classifier):
        """Summary queries should classify as SUMMARIZATION."""
        queries = [
            "Summarize this article",
            "Give me a TLDR",
            "What are the key points?",
        ]
        for query in queries:
            task_type, _ = classifier.classify(query)
            assert task_type == TaskType.SUMMARIZATION

    def test_classify_translation(self, classifier):
        """Translation queries should classify as TRANSLATION."""
        queries = [
            "Translate this to French",
            "How do you say hello in Spanish?",
            "Translation to German please",
        ]
        for query in queries:
            task_type, _ = classifier.classify(query)
            assert task_type == TaskType.TRANSLATION

    def test_classify_chat(self, classifier):
        """Chat messages should classify as CHAT."""
        queries = ["Hello!", "Hi there", "Thanks!", "Okay"]
        for query in queries:
            task_type, _ = classifier.classify(query)
            assert task_type == TaskType.CHAT

    def test_unknown_query_returns_unknown(self, classifier):
        """Ambiguous queries return UNKNOWN with low confidence."""
        query = "xyz abc 123"
        task_type, confidence = classifier.classify(query)
        assert task_type == TaskType.UNKNOWN
        assert confidence < 0.5

    def test_confidence_increases_with_more_patterns(self, classifier):
        """More pattern matches should increase confidence."""
        simple_query = "write code"
        detailed_query = "write code to implement a function that builds a script"

        _, simple_conf = classifier.classify(simple_query)
        _, detailed_conf = classifier.classify(detailed_query)

        # More matches should yield higher confidence
        assert detailed_conf >= simple_conf

    def test_case_insensitive(self, classifier):
        """Classification should be case-insensitive."""
        query1 = "write a poem"
        query2 = "WRITE A POEM"
        query3 = "Write A Poem"

        type1, _ = classifier.classify(query1)
        type2, _ = classifier.classify(query2)
        type3, _ = classifier.classify(query3)

        assert type1 == type2 == type3 == TaskType.CREATIVE_WRITING


class TestModelRouter:
    """Tests for ModelRouter class."""

    @pytest.fixture
    def router(self, mock_registry):
        """Create a ModelRouter with mock registry."""
        return ModelRouter(mock_registry)

    @pytest.fixture
    def router_with_capabilities(self, mock_registry):
        """Create a ModelRouter with test capabilities."""
        capabilities = {
            "model-a": ModelCapability(
                model_id="model-a",
                provider="mock1",
                task_scores={
                    TaskType.CODE_GENERATION: 95,
                    TaskType.CREATIVE_WRITING: 60,
                },
                avg_latency_ms=500,
                cost_per_1k_tokens=0.01,
            ),
            "model-b": ModelCapability(
                model_id="model-b",
                provider="mock2",
                task_scores={
                    TaskType.CODE_GENERATION: 70,
                    TaskType.CREATIVE_WRITING: 90,
                },
                avg_latency_ms=300,
                cost_per_1k_tokens=0.005,
            ),
        }
        return ModelRouter(mock_registry, capabilities=capabilities)

    def test_get_available_models(self, router_with_capabilities):
        """get_available_models returns configured models with capabilities."""
        available = router_with_capabilities.get_available_models()
        assert isinstance(available, list)

    def test_classifier_attribute(self, router):
        """Router should have a TaskClassifier."""
        assert hasattr(router, 'classifier')
        assert isinstance(router.classifier, TaskClassifier)

    async def test_route_returns_routing_decision(self, router_with_capabilities):
        """route() should return a RoutingDecision."""
        decision = await router_with_capabilities.route(
            "Write a Python function",
            strategy=RoutingStrategy.BEST_QUALITY,
        )
        assert isinstance(decision, RoutingDecision)
        assert decision.detected_task is not None
        assert len(decision.selected_models) > 0

    async def test_route_detects_task_type(self, router_with_capabilities):
        """route() should detect the task type from query."""
        decision = await router_with_capabilities.route(
            "Write a Python function to sort a list",
            strategy=RoutingStrategy.BALANCED,
        )
        assert decision.detected_task == TaskType.CODE_GENERATION

    def test_get_model_recommendation(self, router_with_capabilities):
        """get_model_recommendation returns recommendations for task type."""
        recommendations = router_with_capabilities.get_model_recommendation(
            TaskType.CODE_GENERATION
        )
        # Should return recommendations for different strategies
        assert isinstance(recommendations, dict)


class TestRoutingDecision:
    """Tests for RoutingDecision model."""

    def test_create_routing_decision(self):
        """Create a RoutingDecision with required fields."""
        decision = RoutingDecision(
            query="Test query",
            detected_task=TaskType.CODE_GENERATION,
            task_confidence=0.9,
            selected_models=["model-a", "model-b"],
            strategy_used=RoutingStrategy.BEST_QUALITY,
            reasoning="Selected for code generation",
        )
        assert decision.query == "Test query"
        assert decision.detected_task == TaskType.CODE_GENERATION
        assert len(decision.selected_models) == 2

    def test_routing_decision_with_estimates(self):
        """RoutingDecision can include cost and latency estimates."""
        decision = RoutingDecision(
            query="Test",
            detected_task=TaskType.FACTUAL_QA,
            task_confidence=0.8,
            selected_models=["model-a"],
            strategy_used=RoutingStrategy.BEST_VALUE,
            reasoning="Cost-optimized",
            estimated_cost=0.001,
            estimated_latency_ms=500.0,
        )
        assert decision.estimated_cost == 0.001
        assert decision.estimated_latency_ms == 500.0

    def test_routing_decision_with_alternatives(self):
        """RoutingDecision can include alternative model suggestions."""
        decision = RoutingDecision(
            query="Test",
            detected_task=TaskType.CHAT,
            task_confidence=0.7,
            selected_models=["model-a"],
            strategy_used=RoutingStrategy.BALANCED,
            reasoning="Selected best overall",
            alternatives=[
                {"model": "model-b", "score": 85},
                {"model": "model-c", "score": 80},
            ],
        )
        assert len(decision.alternatives) == 2
        assert decision.alternatives[0]["model"] == "model-b"


class TestTaskPatterns:
    """Tests for TaskClassifier pattern matching."""

    @pytest.fixture
    def classifier(self):
        return TaskClassifier()

    def test_code_patterns_exist(self, classifier):
        """CODE_GENERATION patterns include expected keywords."""
        patterns = classifier.TASK_PATTERNS[TaskType.CODE_GENERATION]
        assert "write code" in patterns
        assert "implement" in patterns
        assert "function" in patterns

    def test_math_patterns_exist(self, classifier):
        """MATHEMATICAL patterns include expected keywords."""
        patterns = classifier.TASK_PATTERNS[TaskType.MATHEMATICAL]
        assert "calculate" in patterns
        assert "solve" in patterns
        assert "equation" in patterns

    def test_creative_patterns_exist(self, classifier):
        """CREATIVE_WRITING patterns include expected keywords."""
        patterns = classifier.TASK_PATTERNS[TaskType.CREATIVE_WRITING]
        assert "poem" in patterns
        assert "write a story" in patterns  # Full pattern, not just "story"
        assert "creative" in patterns

    def test_factual_patterns_exist(self, classifier):
        """FACTUAL_QA patterns include expected keywords."""
        patterns = classifier.TASK_PATTERNS[TaskType.FACTUAL_QA]
        assert "what is" in patterns
        assert "who is" in patterns
        assert "when did" in patterns
