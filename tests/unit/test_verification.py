"""Tests for i2i verification engine."""

import pytest
from unittest.mock import AsyncMock, patch

from i2i.verification import VerificationEngine
from i2i.schema import Response, ConfidenceLevel, VerificationResult
from tests.fixtures.mock_providers import (
    MockProviderRegistry,
    MockProviderAdapter,
    create_mock_response,
    VERIFICATION_RESPONSES,
    VERIFICATION_FALSE_RESPONSES,
)


class TestBuildVerificationPrompt:
    """Tests for verification prompt construction."""

    @pytest.fixture
    def engine(self, mock_registry):
        return VerificationEngine(mock_registry)

    def test_prompt_includes_claim(self, engine):
        """Prompt should include the claim to verify."""
        prompt = engine._build_verification_prompt(
            claim="The Eiffel Tower is 324 meters tall",
            original_source=None,
            context=None,
        )
        assert "The Eiffel Tower is 324 meters tall" in prompt

    def test_prompt_includes_source_when_provided(self, engine):
        """Prompt should include source when provided."""
        prompt = engine._build_verification_prompt(
            claim="Test claim",
            original_source="gpt-4",
            context=None,
        )
        assert "gpt-4" in prompt
        assert "Source:" in prompt

    def test_prompt_includes_context_when_provided(self, engine):
        """Prompt should include context when provided."""
        prompt = engine._build_verification_prompt(
            claim="Test claim",
            original_source=None,
            context="This was stated during a discussion about physics.",
        )
        assert "Context:" in prompt
        assert "physics" in prompt

    def test_prompt_includes_verification_instructions(self, engine):
        """Prompt should include instructions for verification response."""
        prompt = engine._build_verification_prompt("Test", None, None)
        assert "VERDICT" in prompt
        assert "TRUE" in prompt
        assert "FALSE" in prompt
        assert "CONFIDENCE" in prompt
        assert "ISSUES" in prompt
        assert "CORRECTIONS" in prompt


class TestAnalyzeVerification:
    """Tests for verification response analysis."""

    @pytest.fixture
    def engine(self, mock_registry):
        return VerificationEngine(mock_registry)

    def test_true_verdict_returns_verified(self, engine):
        """VERDICT: TRUE should return verified=True."""
        responses = [
            create_mock_response(content="VERDICT: TRUE\nThe claim is accurate."),
            create_mock_response(content="VERDICT: TRUE\nThis is correct."),
        ]
        verified, confidence, issues, corrections = engine._analyze_verification(
            "Test claim", responses
        )
        assert verified is True

    def test_false_verdict_returns_not_verified(self, engine):
        """VERDICT: FALSE should return verified=False."""
        responses = [
            create_mock_response(content="VERDICT: FALSE\nThe claim is incorrect."),
            create_mock_response(content="VERDICT: FALSE\nThis is wrong."),
        ]
        verified, confidence, issues, corrections = engine._analyze_verification(
            "Test claim", responses
        )
        assert verified is False

    def test_mixed_verdicts_calculate_average(self, engine):
        """Mixed verdicts should average to determine verification."""
        responses = [
            create_mock_response(content="VERDICT: TRUE\nCorrect."),
            create_mock_response(content="VERDICT: FALSE\nIncorrect."),
        ]
        verified, confidence, issues, corrections = engine._analyze_verification(
            "Test claim", responses
        )
        # 50% agreement, threshold is 60%, so not verified
        assert verified is False

    def test_partially_true_counts_as_half(self, engine):
        """PARTIALLY TRUE should count as 0.5."""
        responses = [
            create_mock_response(content="VERDICT: PARTIALLY TRUE\nSome accuracy issues."),
            create_mock_response(content="VERDICT: TRUE\nCorrect."),
        ]
        verified, confidence, issues, corrections = engine._analyze_verification(
            "Test claim", responses
        )
        # (0.5 + 1.0) / 2 = 0.75 >= 0.6, so verified
        assert verified is True

    def test_extracts_issues_from_section(self, engine):
        """Should extract issues from ISSUES: section."""
        responses = [
            create_mock_response(content="""VERDICT: FALSE
ISSUES:
1. First issue found
2. Second issue found
CORRECTIONS: The correct answer is X"""),
        ]
        verified, confidence, issues, corrections = engine._analyze_verification(
            "Test", responses
        )
        assert len(issues) >= 1

    def test_extracts_corrections_from_section(self, engine):
        """Should extract corrections from CORRECTIONS: section."""
        responses = [
            create_mock_response(content="""VERDICT: FALSE
ISSUES: Some issues here
CORRECTIONS: The correct information is ABC123
REASONING: Because of reasons"""),
        ]
        verified, confidence, issues, corrections = engine._analyze_verification(
            "Test", responses
        )
        assert corrections is not None
        # Case-insensitive check since parser may lowercase
        assert "abc123" in corrections.lower()

    def test_confidence_decreases_with_disagreement(self, engine):
        """Confidence should be lower when verifiers disagree."""
        agreeing_responses = [
            create_mock_response(content="VERDICT: TRUE"),
            create_mock_response(content="VERDICT: TRUE"),
        ]
        disagreeing_responses = [
            create_mock_response(content="VERDICT: TRUE"),
            create_mock_response(content="VERDICT: FALSE"),
        ]

        _, conf_agree, _, _ = engine._analyze_verification("Test", agreeing_responses)
        _, conf_disagree, _, _ = engine._analyze_verification("Test", disagreeing_responses)

        assert conf_agree > conf_disagree

    def test_heuristic_detection_for_correct(self, engine):
        """Should detect positive verdict from 'correct' keyword."""
        responses = [
            create_mock_response(content="The claim is correct and accurate."),
        ]
        verified, _, _, _ = engine._analyze_verification("Test", responses)
        assert verified is True

    def test_heuristic_detection_for_incorrect(self, engine):
        """Should detect negative verdict from 'incorrect' keyword."""
        responses = [
            create_mock_response(content="The claim is incorrect and misleading."),
        ]
        verified, _, _, _ = engine._analyze_verification("Test", responses)
        assert verified is False


class TestVerifyClaim:
    """Integration tests for verify_claim method."""

    async def test_verify_returns_verification_result(self, mock_registry):
        """verify_claim should return a VerificationResult."""
        adapter = MockProviderAdapter(
            "test", ["verifier-model"],
            responses={"verifier-model": "VERDICT: TRUE\nThe claim is correct."}
        )
        registry = MockProviderRegistry({"test": adapter})
        engine = VerificationEngine(registry)

        result = await engine.verify_claim(
            claim="Paris is the capital of France",
            verifier_models=["verifier-model"],
        )

        assert isinstance(result, VerificationResult)
        assert result.original_claim == "Paris is the capital of France"
        assert result.verified is True

    async def test_includes_original_source(self, mock_registry):
        """Result should include original source if provided."""
        adapter = MockProviderAdapter(
            "test", ["model"],
            responses={"model": "VERDICT: TRUE"}
        )
        registry = MockProviderRegistry({"test": adapter})
        engine = VerificationEngine(registry)

        result = await engine.verify_claim(
            claim="Test claim",
            verifier_models=["model"],
            original_source="gpt-4",
        )

        assert result.original_source == "gpt-4"

    async def test_raises_when_all_queries_fail(self, mock_registry):
        """Should raise ValueError when all verifiers fail."""
        failing_adapter = MockProviderAdapter("fail", ["model"])
        failing_adapter.query = AsyncMock(side_effect=Exception("API Error"))

        registry = MockProviderRegistry({"fail": failing_adapter})
        engine = VerificationEngine(registry)

        async def mock_query_multiple(msg, models):
            return [Exception("Error")]

        registry.query_multiple = mock_query_multiple

        with pytest.raises(ValueError, match="All verification queries failed"):
            await engine.verify_claim("Test", ["model"])


class TestBuildChallengePrompt:
    """Tests for challenge prompt construction."""

    @pytest.fixture
    def engine(self, mock_registry):
        return VerificationEngine(mock_registry)

    def test_prompt_includes_response_content(self, engine):
        """Challenge prompt should include the original response."""
        prompt = engine._build_challenge_prompt(
            response_content="The answer is 42.",
            source_model="test-model",
            challenge_type="general",
        )
        assert "The answer is 42." in prompt
        assert "test-model" in prompt

    def test_different_challenge_types(self, engine):
        """Different challenge types should have different instructions."""
        general = engine._build_challenge_prompt("Test", "model", "general")
        factual = engine._build_challenge_prompt("Test", "model", "factual")
        logical = engine._build_challenge_prompt("Test", "model", "logical")
        ethical = engine._build_challenge_prompt("Test", "model", "ethical")

        assert "weaknesses" in general.lower()
        assert "factual accuracy" in factual.lower()
        assert "logical" in logical.lower()
        assert "ethical" in ethical.lower()


class TestParseChallenge:
    """Tests for parsing challenge responses."""

    @pytest.fixture
    def engine(self, mock_registry):
        return VerificationEngine(mock_registry)

    def test_parses_validity_yes(self, engine):
        """Should parse VALIDITY: YES."""
        response = create_mock_response(content="VALIDITY: YES\nThe response is sound.")
        challenge = engine._parse_challenge(response)
        assert challenge["validity"] == "yes"

    def test_parses_validity_no(self, engine):
        """Should parse VALIDITY: NO."""
        response = create_mock_response(content="VALIDITY: NO\nThe response is flawed.")
        challenge = engine._parse_challenge(response)
        assert challenge["validity"] == "no"

    def test_parses_validity_partially(self, engine):
        """Should parse VALIDITY: PARTIALLY."""
        response = create_mock_response(content="VALIDITY: PARTIALLY\nMixed quality.")
        challenge = engine._parse_challenge(response)
        assert challenge["validity"] == "partially"

    def test_extracts_overall_assessment(self, engine):
        """Should extract overall assessment section."""
        response = create_mock_response(content="""VALIDITY: YES
WEAKNESSES: None significant.
OVERALL ASSESSMENT: This is a solid response with minor issues.""")
        challenge = engine._parse_challenge(response)
        assert "solid response" in challenge["assessment"]


class TestEvaluateChallenges:
    """Tests for challenge evaluation."""

    @pytest.fixture
    def engine(self, mock_registry):
        return VerificationEngine(mock_registry)

    def test_no_challenges_withstands(self, engine):
        """Empty challenges should return withstands=True."""
        assert engine._evaluate_challenges([]) is True

    def test_all_valid_challenges_withstands(self, engine):
        """All VALIDITY: YES challenges should return withstands=True."""
        challenges = [
            {"challenge": {"validity": "yes"}},
            {"challenge": {"validity": "yes"}},
        ]
        assert engine._evaluate_challenges(challenges) is True

    def test_all_invalid_challenges_fails(self, engine):
        """All VALIDITY: NO challenges should return withstands=False."""
        challenges = [
            {"challenge": {"validity": "no"}},
            {"challenge": {"validity": "no"}},
        ]
        assert engine._evaluate_challenges(challenges) is False

    def test_mixed_challenges_evaluates_average(self, engine):
        """Mixed challenges should evaluate based on average."""
        challenges = [
            {"challenge": {"validity": "yes"}},  # 1.0
            {"challenge": {"validity": "no"}},   # 0.0
        ]
        # Average is 0.5, threshold is 0.5, so should pass
        assert engine._evaluate_challenges(challenges) is True


class TestSummarizeChallenges:
    """Tests for challenge summarization."""

    @pytest.fixture
    def engine(self, mock_registry):
        return VerificationEngine(mock_registry)

    def test_no_challenges_summary(self, engine):
        """No challenges should return appropriate message."""
        summary = engine._summarize_challenges([])
        assert "No challenges" in summary

    def test_combines_assessments(self, engine):
        """Should combine multiple challenge assessments."""
        challenges = [
            {"challenge": {"assessment": "First assessment."}},
            {"challenge": {"assessment": "Second assessment."}},
        ]
        summary = engine._summarize_challenges(challenges)
        assert "First assessment" in summary
        assert "Second assessment" in summary
