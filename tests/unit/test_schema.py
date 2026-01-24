"""Tests for i2i schema models and enums."""

import pytest
import uuid
from datetime import datetime, timezone

from i2i.schema import (
    Message,
    MessageType,
    Response,
    ConfidenceLevel,
    ConsensusLevel,
    EpistemicType,
    ConsensusResult,
    VerificationResult,
    EpistemicClassification,
)


class TestMessageType:
    """Tests for MessageType enum."""

    def test_all_message_types_exist(self):
        """Verify all expected message types are defined."""
        assert MessageType.QUERY == "query"
        assert MessageType.CHALLENGE == "challenge"
        assert MessageType.VERIFY == "verify"
        assert MessageType.SYNTHESIZE == "synthesize"
        assert MessageType.CLASSIFY == "classify"
        assert MessageType.META == "meta"

    def test_message_type_values_are_strings(self):
        """MessageType values should be strings for JSON serialization."""
        for mt in MessageType:
            assert isinstance(mt.value, str)


class TestEpistemicType:
    """Tests for EpistemicType enum."""

    def test_all_epistemic_types_exist(self):
        """Verify all expected epistemic types are defined."""
        assert EpistemicType.ANSWERABLE == "answerable"
        assert EpistemicType.UNCERTAIN == "uncertain"
        assert EpistemicType.UNDERDETERMINED == "underdetermined"
        assert EpistemicType.IDLE == "idle"
        assert EpistemicType.MALFORMED == "malformed"

    def test_epistemic_type_count(self):
        """There should be exactly 5 epistemic types."""
        assert len(EpistemicType) == 5


class TestConsensusLevel:
    """Tests for ConsensusLevel enum."""

    def test_all_consensus_levels_exist(self):
        """Verify all expected consensus levels are defined."""
        assert ConsensusLevel.HIGH == "high"
        assert ConsensusLevel.MEDIUM == "medium"
        assert ConsensusLevel.LOW == "low"
        assert ConsensusLevel.NONE == "none"
        assert ConsensusLevel.CONTRADICTORY == "contradictory"

    def test_consensus_level_count(self):
        """There should be exactly 5 consensus levels."""
        assert len(ConsensusLevel) == 5


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_all_confidence_levels_exist(self):
        """Verify all expected confidence levels are defined."""
        assert ConfidenceLevel.VERY_HIGH == "very_high"
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"
        assert ConfidenceLevel.VERY_LOW == "very_low"


class TestMessage:
    """Tests for Message model."""

    def test_message_creation_minimal(self):
        """Create message with minimal required fields."""
        msg = Message(type=MessageType.QUERY, content="Hello")
        assert msg.type == MessageType.QUERY
        assert msg.content == "Hello"
        assert msg.sender is None
        assert msg.recipient is None

    def test_message_auto_generates_id(self):
        """Message should auto-generate a UUID id."""
        msg = Message(type=MessageType.QUERY, content="Test")
        assert msg.id is not None
        # Should be valid UUID format
        uuid.UUID(msg.id)

    def test_message_auto_generates_timestamp(self):
        """Message should auto-generate a timestamp."""
        before = datetime.now(timezone.utc)
        msg = Message(type=MessageType.QUERY, content="Test")
        after = datetime.now(timezone.utc)
        assert before <= msg.timestamp <= after

    def test_message_with_all_fields(self):
        """Create message with all optional fields."""
        msg = Message(
            type=MessageType.CHALLENGE,
            content="I challenge this claim",
            sender="challenger-model",
            recipient="original-model",
            target_message_id="original-msg-123",
            metadata={"priority": "high"},
        )
        assert msg.type == MessageType.CHALLENGE
        assert msg.sender == "challenger-model"
        assert msg.recipient == "original-model"
        assert msg.target_message_id == "original-msg-123"
        assert msg.metadata["priority"] == "high"

    def test_message_context_nesting(self):
        """Message context can contain other messages."""
        ctx_msg = Message(type=MessageType.QUERY, content="Previous message")
        msg = Message(
            type=MessageType.QUERY,
            content="Follow-up question",
            context=[ctx_msg],
        )
        assert len(msg.context) == 1
        assert msg.context[0].content == "Previous message"

    def test_message_json_serialization(self):
        """Message should serialize to JSON."""
        msg = Message(type=MessageType.QUERY, content="Test")
        json_str = msg.model_dump_json()
        assert "query" in json_str
        assert "Test" in json_str

    def test_message_metadata_defaults_to_empty_dict(self):
        """Metadata should default to empty dict, not None."""
        msg = Message(type=MessageType.QUERY, content="Test")
        assert msg.metadata == {}
        assert isinstance(msg.metadata, dict)


class TestResponse:
    """Tests for Response model."""

    def test_response_creation(self):
        """Create response with required fields."""
        resp = Response(
            message_id="msg-123",
            model="test/model",
            content="Response text",
            confidence=ConfidenceLevel.HIGH,
        )
        assert resp.message_id == "msg-123"
        assert resp.model == "test/model"
        assert resp.content == "Response text"
        assert resp.confidence == ConfidenceLevel.HIGH

    def test_response_auto_generates_id(self):
        """Response should auto-generate a UUID id."""
        resp = Response(
            message_id="msg-123",
            model="test/model",
            content="Test",
            confidence=ConfidenceLevel.MEDIUM,
        )
        assert resp.id is not None
        uuid.UUID(resp.id)

    def test_response_with_token_counts(self):
        """Response can track token counts."""
        resp = Response(
            message_id="msg-123",
            model="test/model",
            content="Test",
            confidence=ConfidenceLevel.MEDIUM,
            input_tokens=100,
            output_tokens=250,
            latency_ms=523.5,
        )
        assert resp.input_tokens == 100
        assert resp.output_tokens == 250
        assert resp.latency_ms == 523.5

    def test_response_caveats_default_empty(self):
        """Caveats should default to empty list."""
        resp = Response(
            message_id="msg-123",
            model="test/model",
            content="Test",
            confidence=ConfidenceLevel.MEDIUM,
        )
        assert resp.caveats == []

    def test_response_with_reasoning(self):
        """Response can include chain-of-thought reasoning."""
        resp = Response(
            message_id="msg-123",
            model="test/model",
            content="42",
            confidence=ConfidenceLevel.HIGH,
            reasoning="First I considered X, then Y, therefore 42.",
        )
        assert resp.reasoning is not None
        assert "considered" in resp.reasoning


class TestConsensusResult:
    """Tests for ConsensusResult model."""

    def test_consensus_result_creation(self, multiple_responses):
        """Create ConsensusResult with required fields."""
        result = ConsensusResult(
            query="What is the capital of France?",
            models_queried=["model-a", "model-b", "model-c"],
            responses=multiple_responses,
            consensus_level=ConsensusLevel.HIGH,
            consensus_answer="Paris is the capital of France.",
        )
        assert result.query == "What is the capital of France?"
        assert len(result.models_queried) == 3
        assert len(result.responses) == 3
        assert result.consensus_level == ConsensusLevel.HIGH
        assert "Paris" in result.consensus_answer

    def test_consensus_result_with_divergences(self, multiple_responses):
        """ConsensusResult can include divergence information."""
        result = ConsensusResult(
            query="Best programming language?",
            models_queried=["model-a", "model-b"],
            responses=multiple_responses[:2],
            consensus_level=ConsensusLevel.LOW,
            divergences=[
                {"models": ["model-a", "model-b"], "summary": "Different recommendations"}
            ],
        )
        assert len(result.divergences) == 1
        assert result.divergences[0]["summary"] == "Different recommendations"

    def test_consensus_result_with_clusters(self, multiple_responses):
        """ConsensusResult can include response clusters."""
        result = ConsensusResult(
            query="Test",
            models_queried=["a", "b", "c"],
            responses=multiple_responses,
            consensus_level=ConsensusLevel.MEDIUM,
            clusters=[["a", "b"], ["c"]],
        )
        assert result.clusters == [["a", "b"], ["c"]]


class TestVerificationResult:
    """Tests for VerificationResult model."""

    def test_verification_result_verified(self, sample_response):
        """Create a verified VerificationResult."""
        result = VerificationResult(
            original_claim="The Eiffel Tower is 324 meters tall",
            verifiers=["model-a", "model-b"],
            verification_responses=[sample_response],
            verified=True,
            confidence=0.95,
        )
        assert result.verified is True
        assert result.confidence == 0.95
        assert len(result.issues_found) == 0

    def test_verification_result_not_verified(self, sample_response):
        """Create a non-verified VerificationResult with issues."""
        result = VerificationResult(
            original_claim="Napoleon was very short",
            original_source="gpt-4",
            verifiers=["claude-3"],
            verification_responses=[sample_response],
            verified=False,
            confidence=0.85,
            issues_found=["Napoleon was average height for his time"],
            corrections="Napoleon was approximately 5'7\" (170cm)",
        )
        assert result.verified is False
        assert len(result.issues_found) == 1
        assert result.corrections is not None


class TestEpistemicClassification:
    """Tests for EpistemicClassification model."""

    def test_answerable_classification(self):
        """Create an ANSWERABLE classification."""
        result = EpistemicClassification(
            question="What is 2 + 2?",
            classification=EpistemicType.ANSWERABLE,
            confidence=0.99,
            reasoning="This is a basic arithmetic question with a definite answer.",
            is_actionable=True,
        )
        assert result.classification == EpistemicType.ANSWERABLE
        assert result.is_actionable is True

    def test_idle_classification(self):
        """Create an IDLE classification with why_idle."""
        result = EpistemicClassification(
            question="Is consciousness substrate-independent?",
            classification=EpistemicType.IDLE,
            confidence=0.8,
            reasoning="This is a philosophical question without empirical resolution.",
            is_actionable=False,
            why_idle="No answer would change any practical decision or action.",
        )
        assert result.classification == EpistemicType.IDLE
        assert result.is_actionable is False
        assert result.why_idle is not None

    def test_underdetermined_classification(self):
        """Create an UNDERDETERMINED classification with hypotheses."""
        result = EpistemicClassification(
            question="Did Shakespeare write all his plays?",
            classification=EpistemicType.UNDERDETERMINED,
            confidence=0.7,
            reasoning="Multiple hypotheses fit the historical evidence.",
            is_actionable=False,
            competing_hypotheses=[
                "Shakespeare wrote all plays himself",
                "Shakespeare collaborated with others",
                "Some plays attributed to others",
            ],
        )
        assert result.classification == EpistemicType.UNDERDETERMINED
        assert len(result.competing_hypotheses) == 3

    def test_classification_with_reformulation(self):
        """Classification can suggest a better question formulation."""
        result = EpistemicClassification(
            question="What is the meaning of life?",
            classification=EpistemicType.IDLE,
            confidence=0.85,
            reasoning="Abstract philosophical question.",
            is_actionable=False,
            suggested_reformulation="What activities give you a sense of purpose?",
        )
        assert result.suggested_reformulation is not None
        assert "purpose" in result.suggested_reformulation
