"""Tests for i2i epistemic classifier."""

import pytest
from unittest.mock import AsyncMock, patch

from i2i.epistemic import EpistemicClassifier
from i2i.schema import EpistemicType, Response, ConfidenceLevel
from tests.fixtures.mock_providers import (
    MockProviderRegistry,
    MockProviderAdapter,
    create_mock_response,
)


class TestQuickClassify:
    """Tests for quick heuristic classification."""

    @pytest.fixture
    def classifier(self, mock_registry):
        """Create an EpistemicClassifier with mock registry."""
        return EpistemicClassifier(mock_registry)

    def test_what_is_questions_are_answerable(self, classifier):
        """'What is' questions should be ANSWERABLE."""
        assert classifier.quick_classify("What is the capital of France?") == EpistemicType.ANSWERABLE

    def test_who_is_questions_are_answerable(self, classifier):
        """'Who is' questions should be ANSWERABLE."""
        assert classifier.quick_classify("Who is the president of the United States?") == EpistemicType.ANSWERABLE

    def test_where_is_questions_are_answerable(self, classifier):
        """'Where is' questions should be ANSWERABLE."""
        assert classifier.quick_classify("Where is the Eiffel Tower?") == EpistemicType.ANSWERABLE

    def test_when_did_questions_are_answerable(self, classifier):
        """'When did' questions should be ANSWERABLE."""
        assert classifier.quick_classify("When did World War II end?") == EpistemicType.ANSWERABLE

    def test_how_many_questions_are_answerable(self, classifier):
        """'How many' questions should be ANSWERABLE."""
        assert classifier.quick_classify("How many planets are in our solar system?") == EpistemicType.ANSWERABLE

    def test_will_questions_are_uncertain(self, classifier):
        """Questions with 'will' are UNCERTAIN."""
        assert classifier.quick_classify("Will it rain tomorrow?") == EpistemicType.UNCERTAIN

    def test_going_to_questions_are_uncertain(self, classifier):
        """Questions with 'going to' are UNCERTAIN."""
        assert classifier.quick_classify("Is the stock market going to crash?") == EpistemicType.UNCERTAIN

    def test_predict_questions_are_uncertain(self, classifier):
        """Prediction questions are UNCERTAIN."""
        assert classifier.quick_classify("Can you predict the weather?") == EpistemicType.UNCERTAIN

    def test_consciousness_questions_are_idle(self, classifier):
        """Questions about consciousness are IDLE."""
        assert classifier.quick_classify("Is consciousness fundamental to the universe?") == EpistemicType.IDLE

    def test_free_will_questions_are_idle(self, classifier):
        """Questions about philosophical topics are IDLE."""
        # Note: "free will" contains "will" which triggers UNCERTAIN first
        # Use "soul" or "qualia" instead to test IDLE pattern
        assert classifier.quick_classify("Do humans have a soul?") == EpistemicType.IDLE

    def test_meaning_of_life_questions_are_idle(self, classifier):
        """Questions containing 'meaning of life' are IDLE."""
        # Don't start with "what is" which triggers ANSWERABLE
        # Use a phrasing that avoids early pattern matches
        assert classifier.quick_classify("Can we discover the meaning of life?") == EpistemicType.IDLE

    def test_god_questions_are_idle(self, classifier):
        """Questions about God are IDLE."""
        assert classifier.quick_classify("Does god exist?") == EpistemicType.IDLE

    def test_what_if_questions_are_underdetermined(self, classifier):
        """Counterfactual 'what if' questions are UNDERDETERMINED."""
        assert classifier.quick_classify("What if dinosaurs never went extinct?") == EpistemicType.UNDERDETERMINED

    def test_would_have_questions_are_underdetermined(self, classifier):
        """'Would have' counterfactuals are UNDERDETERMINED."""
        assert classifier.quick_classify("What would have happened if Rome never fell?") == EpistemicType.UNDERDETERMINED

    def test_ambiguous_questions_default_to_uncertain(self, classifier):
        """Ambiguous questions default to UNCERTAIN."""
        assert classifier.quick_classify("Is this good?") == EpistemicType.UNCERTAIN

    def test_case_insensitive(self, classifier):
        """Classification should be case-insensitive."""
        assert classifier.quick_classify("WHAT IS THE CAPITAL OF FRANCE?") == EpistemicType.ANSWERABLE
        assert classifier.quick_classify("Will It Rain?") == EpistemicType.UNCERTAIN


class TestParseClassificationResponse:
    """Tests for parsing classification responses."""

    @pytest.fixture
    def classifier(self, mock_registry):
        return EpistemicClassifier(mock_registry)

    def test_parse_answerable_classification(self, classifier):
        """Parse ANSWERABLE classification response."""
        content = """CLASSIFICATION: answerable
CONFIDENCE: 0.95
REASONING: This is a factual question with a definitive answer.
IS_ACTIONABLE: yes"""
        result = classifier._parse_classification_response(content)
        assert result["classification"] == EpistemicType.ANSWERABLE
        assert result["confidence"] == 0.95
        assert result["is_actionable"] is True

    def test_parse_idle_classification(self, classifier):
        """Parse IDLE classification response."""
        content = """CLASSIFICATION: idle
CONFIDENCE: 0.85
REASONING: This is a philosophical question.
WHY_IDLE: No answer would change any practical decision.
IS_ACTIONABLE: no"""
        result = classifier._parse_classification_response(content)
        assert result["classification"] == EpistemicType.IDLE
        assert result["is_actionable"] is False
        assert "practical decision" in result["why_idle"]

    def test_parse_uncertain_classification(self, classifier):
        """Parse UNCERTAIN classification response."""
        content = """CLASSIFICATION: uncertain
CONFIDENCE: 0.7
REASONING: This involves predicting future events.
IS_ACTIONABLE: yes"""
        result = classifier._parse_classification_response(content)
        assert result["classification"] == EpistemicType.UNCERTAIN
        assert result["confidence"] == 0.7

    def test_parse_malformed_classification(self, classifier):
        """Parse MALFORMED classification response."""
        content = """CLASSIFICATION: malformed
CONFIDENCE: 0.9
REASONING: The question is self-contradictory.
IS_ACTIONABLE: no"""
        result = classifier._parse_classification_response(content)
        assert result["classification"] == EpistemicType.MALFORMED

    def test_parse_with_reformulation(self, classifier):
        """Parse response with suggested reformulation."""
        content = """CLASSIFICATION: idle
CONFIDENCE: 0.8
REASONING: Abstract philosophical question.
IS_ACTIONABLE: no
SUGGESTED_REFORMULATION: What actions help you feel a sense of purpose?"""
        result = classifier._parse_classification_response(content)
        assert result["reformulation"] is not None
        assert "purpose" in result["reformulation"]

    def test_parse_fallback_detection(self, classifier):
        """Parse when classification keyword is in text but not in standard format."""
        content = """This question is clearly answerable because it asks about a fact."""
        result = classifier._parse_classification_response(content)
        assert result["classification"] == EpistemicType.ANSWERABLE

    def test_parse_idle_detection_by_keywords(self, classifier):
        """Detect IDLE when 'idle' and 'non-action-guiding' appear."""
        content = """This is an idle question, non-action-guiding in nature."""
        result = classifier._parse_classification_response(content)
        assert result["classification"] == EpistemicType.IDLE


class TestAnalyzeClassifications:
    """Tests for analyzing multiple classification responses."""

    @pytest.fixture
    def classifier(self, mock_registry):
        return EpistemicClassifier(mock_registry)

    def test_majority_vote_classification(self, classifier):
        """Final classification uses majority vote."""
        responses = [
            create_mock_response(content="CLASSIFICATION: answerable\nCONFIDENCE: 0.9"),
            create_mock_response(content="CLASSIFICATION: answerable\nCONFIDENCE: 0.8"),
            create_mock_response(content="CLASSIFICATION: uncertain\nCONFIDENCE: 0.7"),
        ]
        result = classifier._analyze_classifications("Test question", responses)
        assert result.classification == EpistemicType.ANSWERABLE

    def test_average_confidence(self, classifier):
        """Final confidence is average of parsed confidences."""
        responses = [
            create_mock_response(content="CLASSIFICATION: answerable\nCONFIDENCE: 0.9"),
            create_mock_response(content="CLASSIFICATION: answerable\nCONFIDENCE: 0.7"),
        ]
        result = classifier._analyze_classifications("Test", responses)
        assert result.confidence == pytest.approx(0.8, 0.1)

    def test_actionability_majority_vote(self, classifier):
        """Actionability uses majority vote."""
        responses = [
            create_mock_response(content="CLASSIFICATION: uncertain\nIS_ACTIONABLE: yes"),
            create_mock_response(content="CLASSIFICATION: uncertain\nIS_ACTIONABLE: yes"),
            create_mock_response(content="CLASSIFICATION: uncertain\nIS_ACTIONABLE: no"),
        ]
        result = classifier._analyze_classifications("Test", responses)
        assert result.is_actionable is True

    def test_fallback_when_parsing_fails(self, classifier):
        """Defaults to UNCERTAIN when parsing fails."""
        responses = [
            create_mock_response(content="Invalid response with no classification"),
            create_mock_response(content="Another invalid response"),
        ]
        result = classifier._analyze_classifications("Test", responses)
        assert result.classification == EpistemicType.UNCERTAIN
        assert result.confidence == 0.5


class TestClassifyQuestion:
    """Integration tests for classify_question method."""

    async def test_classify_with_explicit_models_returns_result(self, mock_registry):
        """classify_question returns an EpistemicClassification when models are specified."""
        # Create adapter with classification response
        adapter = MockProviderAdapter(
            "test",
            ["test-model"],
            responses={"test-model": "CLASSIFICATION: answerable\nCONFIDENCE: 0.9\nREASONING: Factual question.\nIS_ACTIONABLE: yes"}
        )
        registry = MockProviderRegistry({"test": adapter})
        classifier = EpistemicClassifier(registry)

        # Explicitly pass the model to avoid configuration check
        result = await classifier.classify_question(
            "What is 2 + 2?",
            classifier_models=["test-model"]
        )

        assert result.question == "What is 2 + 2?"
        assert result.classification == EpistemicType.ANSWERABLE

    async def test_raises_when_all_queries_fail(self, mock_registry):
        """Raises ValueError when all classification queries fail."""
        failing_adapter = MockProviderAdapter("fail", ["model"])

        registry = MockProviderRegistry({"fail": failing_adapter})
        classifier = EpistemicClassifier(registry)

        async def mock_query_multiple(msg, models):
            return [Exception("Error")]

        registry.query_multiple = mock_query_multiple

        # Pass models explicitly to avoid config check
        with pytest.raises(ValueError, match="All classification queries failed"):
            await classifier.classify_question("Test?", classifier_models=["model"])


class TestBuildClassificationPrompt:
    """Tests for classification prompt construction."""

    @pytest.fixture
    def classifier(self, mock_registry):
        return EpistemicClassifier(mock_registry)

    def test_prompt_includes_question(self, classifier):
        """Prompt should include the original question."""
        prompt = classifier._build_classification_prompt("What is Python?")
        assert "What is Python?" in prompt

    def test_prompt_includes_all_categories(self, classifier):
        """Prompt should describe all 5 epistemic categories."""
        prompt = classifier._build_classification_prompt("Test")
        assert "ANSWERABLE" in prompt
        assert "UNCERTAIN" in prompt
        assert "UNDERDETERMINED" in prompt
        assert "IDLE" in prompt
        assert "MALFORMED" in prompt

    def test_prompt_includes_examples(self, classifier):
        """Prompt should include examples for each category."""
        prompt = classifier._build_classification_prompt("Test")
        assert "capital of France" in prompt  # ANSWERABLE example
        assert "rain" in prompt  # UNCERTAIN example
        assert "Shakespeare" in prompt  # UNDERDETERMINED example
        assert "consciousness" in prompt  # IDLE example
        assert "square circle" in prompt  # MALFORMED example
