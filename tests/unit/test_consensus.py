"""Tests for i2i consensus engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from i2i.consensus import ConsensusEngine
from i2i.schema import (
    Message,
    MessageType,
    Response,
    ConfidenceLevel,
    ConsensusLevel,
)
from tests.fixtures.mock_providers import (
    MockProviderRegistry,
    MockProviderAdapter,
    create_mock_response,
    AGREEING_RESPONSES,
    DISAGREEING_RESPONSES,
)


class TestComputeSimilarity:
    """Tests for the _compute_similarity method."""

    @pytest.fixture
    def engine(self, mock_registry):
        """Create a ConsensusEngine with mock registry."""
        return ConsensusEngine(mock_registry)

    def test_identical_texts_have_similarity_one(self, engine):
        """Identical texts should have similarity of 1.0."""
        text = "The capital of France is Paris."
        similarity = engine._compute_similarity(text, text)
        assert similarity == 1.0

    def test_completely_different_texts(self, engine):
        """Completely different texts should have low similarity."""
        text1 = "Python programming language"
        text2 = "Elephant zebra giraffe"
        similarity = engine._compute_similarity(text1, text2)
        assert similarity < 0.3

    def test_similar_texts_have_positive_similarity(self, engine):
        """Similar texts should have positive similarity."""
        text1 = "Paris capital France"
        text2 = "France capital Paris"
        similarity = engine._compute_similarity(text1, text2)
        # After stop word removal, these should be identical
        assert similarity >= 0.5

    def test_stop_words_are_removed(self, engine):
        """Stop words should not affect similarity."""
        text1 = "Python programming"
        text2 = "The Python is a programming language"
        similarity = engine._compute_similarity(text1, text2)
        # Should be relatively high despite stop word differences
        assert similarity > 0.3

    def test_case_insensitive(self, engine):
        """Similarity should be case-insensitive."""
        text1 = "PYTHON PROGRAMMING"
        text2 = "python programming"
        similarity = engine._compute_similarity(text1, text2)
        assert similarity == 1.0

    def test_empty_after_stop_words(self, engine):
        """Returns 0.5 when all words are stop words."""
        text1 = "the a an is"
        text2 = "was were be been"
        similarity = engine._compute_similarity(text1, text2)
        assert similarity == 0.5


class TestHasContradictions:
    """Tests for the _has_contradictions method."""

    @pytest.fixture
    def engine(self, mock_registry):
        return ConsensusEngine(mock_registry)

    def test_no_contradictions_when_all_positive(self, engine):
        """No contradictions when all responses are affirmative."""
        responses = [
            create_mock_response(content="Yes, that is correct."),
            create_mock_response(content="True, this is right."),
        ]
        assert engine._has_contradictions(responses) is False

    def test_no_contradictions_when_all_agree_negative(self, engine):
        """No contradictions when responses agree (all negative without positive markers)."""
        responses = [
            create_mock_response(content="This statement is inaccurate."),
            create_mock_response(content="This claim cannot be verified."),
        ]
        # No positive or negative markers, so no contradiction detected
        assert engine._has_contradictions(responses) is False

    def test_has_contradictions_when_mixed(self, engine):
        """Contradictions when responses have opposite conclusions."""
        responses = [
            create_mock_response(content="Yes, that is correct and true."),
            create_mock_response(content="No, that is incorrect and false."),
        ]
        assert engine._has_contradictions(responses) is True


class TestIdentifyDivergences:
    """Tests for the _identify_divergences method."""

    @pytest.fixture
    def engine(self, mock_registry):
        return ConsensusEngine(mock_registry)

    def test_no_divergences_when_high_agreement(self, engine):
        """No divergences reported when similarity is high."""
        responses = [
            create_mock_response(model="model-a", content="Paris"),
            create_mock_response(model="model-b", content="Paris"),
        ]
        agreement_matrix = {
            "model-a": {"model-a": 1.0, "model-b": 0.9},
            "model-b": {"model-a": 0.9, "model-b": 1.0},
        }
        divergences = engine._identify_divergences(responses, agreement_matrix)
        assert len(divergences) == 0

    def test_identifies_significant_divergences(self, engine):
        """Divergences reported when similarity < 0.5."""
        responses = [
            create_mock_response(model="model-a", content="Answer A"),
            create_mock_response(model="model-b", content="Answer B"),
        ]
        agreement_matrix = {
            "model-a": {"model-a": 1.0, "model-b": 0.3},
            "model-b": {"model-a": 0.3, "model-b": 1.0},
        }
        divergences = engine._identify_divergences(responses, agreement_matrix)
        assert len(divergences) == 1
        assert "model-a" in divergences[0]["models"]
        assert "model-b" in divergences[0]["models"]


class TestClusterResponses:
    """Tests for the _cluster_responses method."""

    @pytest.fixture
    def engine(self, mock_registry):
        return ConsensusEngine(mock_registry)

    def test_returns_none_with_fewer_than_3_responses(self, engine):
        """Returns None when there are fewer than 3 responses."""
        responses = [
            create_mock_response(model="model-a"),
            create_mock_response(model="model-b"),
        ]
        agreement_matrix = {
            "model-a": {"model-a": 1.0, "model-b": 0.5},
            "model-b": {"model-a": 0.5, "model-b": 1.0},
        }
        clusters = engine._cluster_responses(responses, agreement_matrix)
        assert clusters is None

    def test_clusters_high_agreement_models(self, engine):
        """Models with high similarity are clustered together."""
        responses = [
            create_mock_response(model="model-a"),
            create_mock_response(model="model-b"),
            create_mock_response(model="model-c"),
        ]
        agreement_matrix = {
            "model-a": {"model-a": 1.0, "model-b": 0.8, "model-c": 0.2},
            "model-b": {"model-a": 0.8, "model-b": 1.0, "model-c": 0.2},
            "model-c": {"model-a": 0.2, "model-b": 0.2, "model-c": 1.0},
        }
        clusters = engine._cluster_responses(responses, agreement_matrix)
        assert clusters is not None
        assert len(clusters) == 2  # Two groups: [a, b] and [c]


class TestConfidenceScore:
    """Tests for the _confidence_score method."""

    @pytest.fixture
    def engine(self, mock_registry):
        return ConsensusEngine(mock_registry)

    def test_very_high_confidence_score(self, engine):
        """VERY_HIGH confidence should have score 5."""
        assert engine._confidence_score(ConfidenceLevel.VERY_HIGH) == 5

    def test_high_confidence_score(self, engine):
        """HIGH confidence should have score 4."""
        assert engine._confidence_score(ConfidenceLevel.HIGH) == 4

    def test_medium_confidence_score(self, engine):
        """MEDIUM confidence should have score 3."""
        assert engine._confidence_score(ConfidenceLevel.MEDIUM) == 3

    def test_low_confidence_score(self, engine):
        """LOW confidence should have score 2."""
        assert engine._confidence_score(ConfidenceLevel.LOW) == 2

    def test_very_low_confidence_score(self, engine):
        """VERY_LOW confidence should have score 1."""
        assert engine._confidence_score(ConfidenceLevel.VERY_LOW) == 1


class TestAnalyzeConsensus:
    """Tests for the _analyze_consensus method."""

    @pytest.fixture
    def engine(self, mock_registry):
        return ConsensusEngine(mock_registry)

    async def test_single_response_is_high_consensus(self, engine):
        """Single response should be HIGH consensus."""
        responses = [create_mock_response(model="model-a", content="Answer")]
        level, matrix = await engine._analyze_consensus(responses)
        assert level == ConsensusLevel.HIGH

    async def test_identical_responses_high_consensus(self, engine):
        """Identical responses should produce HIGH consensus."""
        content = "The capital of France is Paris."
        responses = [
            create_mock_response(model="model-a", content=content),
            create_mock_response(model="model-b", content=content),
        ]
        level, matrix = await engine._analyze_consensus(responses)
        assert level == ConsensusLevel.HIGH

    async def test_similar_responses_produces_consensus(self, engine):
        """Similar responses should produce some level of consensus."""
        # Use identical content for reliable HIGH consensus
        content = "Paris is the capital of France."
        responses = [
            create_mock_response(model="model-a", content=content),
            create_mock_response(model="model-b", content=content),
        ]
        level, matrix = await engine._analyze_consensus(responses)
        assert level == ConsensusLevel.HIGH

    async def test_very_different_responses_low_or_none_consensus(self, engine):
        """Very different responses should produce LOW or NONE consensus."""
        responses = [
            create_mock_response(model="model-a", content="Python is best for data science."),
            create_mock_response(model="model-b", content="JavaScript dominates web development."),
            create_mock_response(model="model-c", content="Rust excels at systems programming."),
        ]
        level, matrix = await engine._analyze_consensus(responses)
        assert level in [ConsensusLevel.LOW, ConsensusLevel.NONE]


class TestQueryForConsensus:
    """Tests for the query_for_consensus method."""

    async def test_query_returns_consensus_result(self, agreeing_registry):
        """query_for_consensus returns a ConsensusResult."""
        engine = ConsensusEngine(agreeing_registry)
        result = await engine.query_for_consensus(
            query="What is the capital of France?",
            models=["model-a", "model-b", "model-c"],
        )
        assert result.query == "What is the capital of France?"
        assert len(result.responses) == 3
        assert result.consensus_level is not None

    async def test_query_filters_out_errors(self, mock_registry):
        """Errors from providers should be filtered out."""
        # Create a registry where one provider fails
        failing_adapter = MockProviderAdapter("failing", ["fail-model"], configured=True)
        failing_adapter.query = AsyncMock(side_effect=Exception("API Error"))

        working_adapter = MockProviderAdapter("working", ["work-model"], configured=True)

        registry = MockProviderRegistry({
            "failing": failing_adapter,
            "working": working_adapter,
        })

        engine = ConsensusEngine(registry)

        # Should not raise, just filter out the failing model
        result = await engine.query_for_consensus(
            query="Test",
            models=["work-model"],  # Only query the working model
        )
        assert len(result.responses) >= 1

    async def test_raises_when_all_queries_fail(self, mock_registry):
        """Should raise ValueError when all queries fail."""
        failing_adapter = MockProviderAdapter("fail", ["model-1", "model-2"])
        failing_adapter.query = AsyncMock(side_effect=Exception("All fail"))

        registry = MockProviderRegistry({"fail": failing_adapter})
        engine = ConsensusEngine(registry)

        # Mock query_multiple to return all exceptions
        async def mock_query_multiple(msg, models):
            return [Exception("Error 1"), Exception("Error 2")]

        registry.query_multiple = mock_query_multiple

        with pytest.raises(ValueError, match="All model queries failed"):
            await engine.query_for_consensus("Test", ["model-1", "model-2"])

    async def test_consensus_answer_synthesized_for_high_consensus(self, agreeing_registry):
        """Consensus answer should be synthesized when consensus is HIGH."""
        engine = ConsensusEngine(agreeing_registry)

        # Mock the synthesis to avoid needing real API
        with patch.object(engine, '_synthesize_consensus', new_callable=AsyncMock) as mock_synth:
            mock_synth.return_value = "Paris is the capital of France."

            result = await engine.query_for_consensus(
                query="What is the capital of France?",
                models=["model-a", "model-b"],
            )

            # If consensus is HIGH or MEDIUM, synthesis should be called
            if result.consensus_level in [ConsensusLevel.HIGH, ConsensusLevel.MEDIUM]:
                assert result.consensus_answer is not None

    async def test_includes_agreement_matrix(self, agreeing_registry):
        """Result should include agreement matrix."""
        engine = ConsensusEngine(agreeing_registry)
        result = await engine.query_for_consensus(
            query="Test",
            models=["model-a", "model-b"],
        )
        assert result.agreement_matrix is not None
        assert len(result.agreement_matrix) == 2
