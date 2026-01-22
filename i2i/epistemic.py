"""
Epistemic classification system.

This module classifies questions/claims into epistemic categories:
- Answerable: Can be resolved with available information
- Uncertain: Answerable but with significant uncertainty
- Underdetermined: Multiple hypotheses fit the data equally
- Idle: Well-formed but non-action-guiding

This was directly inspired by the Claude-ChatGPT conversation about
AI consciousness, where ChatGPT noted that some questions are
"well-formed but idle" - coherent but non-action-guiding.
"""

import asyncio
from typing import List, Optional, Dict, Any

from .schema import (
    Message,
    MessageType,
    Response,
    EpistemicClassification,
    EpistemicType,
)
from .providers import ProviderRegistry


class EpistemicClassifier:
    """
    Classifier for determining the epistemic status of questions.
    """

    def __init__(self, registry: ProviderRegistry):
        self.registry = registry

    async def classify_question(
        self,
        question: str,
        classifier_models: Optional[List[str]] = None,
    ) -> EpistemicClassification:
        """
        Classify a question's epistemic status.

        Args:
            question: The question to classify
            classifier_models: Models to use for classification (uses defaults if None)

        Returns:
            EpistemicClassification with analysis
        """
        if classifier_models is None:
            # Use capable models for classification - prefer what's available
            available = self.registry.list_configured_providers()
            classifier_models = []
            if "anthropic" in available:
                classifier_models.append("claude-3-5-sonnet-20241022")
            if "openai" in available:
                classifier_models.append("gpt-4o")
            if "google" in available:
                classifier_models.append("gemini-1.5-pro")
            if not classifier_models:
                raise ValueError("No classifiers available. Configure at least one provider.")

        classification_prompt = self._build_classification_prompt(question)

        message = Message(
            type=MessageType.CLASSIFY,
            content=classification_prompt,
        )

        # Query classifiers
        responses = await self.registry.query_multiple(message, classifier_models)
        valid_responses = [r for r in responses if isinstance(r, Response)]

        if not valid_responses:
            raise ValueError("All classification queries failed")

        # Analyze classifications
        classification = self._analyze_classifications(question, valid_responses)

        return classification

    def _build_classification_prompt(self, question: str) -> str:
        """Build the classification prompt."""
        return f"""Analyze the following question and classify its epistemic status:

QUESTION: "{question}"

Please classify this question into ONE of these categories:

1. ANSWERABLE - The question can be definitively answered with available information
   Example: "What is the capital of France?"

2. UNCERTAIN - The question can be answered but with significant uncertainty
   Example: "Will it rain in New York next Tuesday?"

3. UNDERDETERMINED - Multiple hypotheses fit the available data equally well
   Example: "Did Shakespeare write all the plays attributed to him?"

4. IDLE - The question is well-formed but non-action-guiding; answering it wouldn't
   change any decisions or behaviors
   Example: "Is consciousness substrate-independent?"

5. MALFORMED - The question is incoherent, self-contradictory, or based on false premises
   Example: "What color is the square circle?"

Respond with:
CLASSIFICATION: [category]
CONFIDENCE: [0.0-1.0]
REASONING: [detailed explanation]

If UNDERDETERMINED, also list:
COMPETING_HYPOTHESES: [list the equally-supported hypotheses]

If UNCERTAIN, also list:
UNCERTAINTY_SOURCES: [what makes it uncertain]

If IDLE, also explain:
WHY_IDLE: [why answering wouldn't guide action]

IS_ACTIONABLE: [yes/no - would an answer change any real-world decision?]

If the question could be reformulated to be more tractable, provide:
SUGGESTED_REFORMULATION: [a more answerable version of the question]"""

    def _analyze_classifications(
        self,
        question: str,
        responses: List[Response],
    ) -> EpistemicClassification:
        """Analyze classification responses and produce final classification."""
        classifications = []
        confidences = []
        reasonings = []
        competing_hypotheses = []
        uncertainty_sources = []
        why_idle_reasons = []
        is_actionable_votes = []
        reformulations = []

        for resp in responses:
            parsed = self._parse_classification_response(resp.content)
            if parsed["classification"]:
                classifications.append(parsed["classification"])
                confidences.append(parsed["confidence"])
                reasonings.append(parsed["reasoning"])

                if parsed["competing_hypotheses"]:
                    competing_hypotheses.extend(parsed["competing_hypotheses"])
                if parsed["uncertainty_sources"]:
                    uncertainty_sources.extend(parsed["uncertainty_sources"])
                if parsed["why_idle"]:
                    why_idle_reasons.append(parsed["why_idle"])
                if parsed["is_actionable"] is not None:
                    is_actionable_votes.append(parsed["is_actionable"])
                if parsed["reformulation"]:
                    reformulations.append(parsed["reformulation"])

        # Determine final classification (majority vote)
        if not classifications:
            # Default to uncertain if parsing failed
            final_classification = EpistemicType.UNCERTAIN
            final_confidence = 0.5
            final_reasoning = "Unable to parse classifier responses"
        else:
            # Count votes
            from collections import Counter
            vote_counts = Counter(classifications)
            final_classification = vote_counts.most_common(1)[0][0]
            final_confidence = sum(confidences) / len(confidences)
            final_reasoning = reasonings[0] if reasonings else ""

        # Determine actionability (majority vote)
        is_actionable = True
        if is_actionable_votes:
            is_actionable = sum(is_actionable_votes) / len(is_actionable_votes) >= 0.5

        # Deduplicate lists
        competing_hypotheses = list(set(competing_hypotheses)) if competing_hypotheses else None
        uncertainty_sources = list(set(uncertainty_sources)) if uncertainty_sources else None
        why_idle = why_idle_reasons[0] if why_idle_reasons else None
        reformulation = reformulations[0] if reformulations else None

        return EpistemicClassification(
            question=question,
            classification=final_classification,
            confidence=final_confidence,
            reasoning=final_reasoning,
            competing_hypotheses=competing_hypotheses,
            uncertainty_sources=uncertainty_sources,
            why_idle=why_idle,
            is_actionable=is_actionable,
            suggested_reformulation=reformulation,
        )

    def _parse_classification_response(self, content: str) -> Dict[str, Any]:
        """Parse a classification response into structured format."""
        result = {
            "classification": None,
            "confidence": 0.5,
            "reasoning": "",
            "competing_hypotheses": [],
            "uncertainty_sources": [],
            "why_idle": None,
            "is_actionable": None,
            "reformulation": None,
        }

        content_lower = content.lower()

        # Extract classification
        for etype in EpistemicType:
            if f"classification: {etype.value}" in content_lower or f"classification:{etype.value}" in content_lower:
                result["classification"] = etype
                break

        # If not found explicitly, look for the word
        if result["classification"] is None:
            if "answerable" in content_lower and "not answerable" not in content_lower:
                result["classification"] = EpistemicType.ANSWERABLE
            elif "underdetermined" in content_lower:
                result["classification"] = EpistemicType.UNDERDETERMINED
            elif "idle" in content_lower and "non-action-guiding" in content_lower:
                result["classification"] = EpistemicType.IDLE
            elif "malformed" in content_lower or "incoherent" in content_lower:
                result["classification"] = EpistemicType.MALFORMED
            elif "uncertain" in content_lower:
                result["classification"] = EpistemicType.UNCERTAIN

        # Extract confidence
        import re
        confidence_match = re.search(r'confidence:\s*([\d.]+)', content_lower)
        if confidence_match:
            try:
                result["confidence"] = float(confidence_match.group(1))
            except ValueError:
                pass

        # Extract reasoning
        if "reasoning:" in content_lower:
            reasoning_start = content_lower.index("reasoning:")
            reasoning_end = len(content)
            # Find next section marker
            for marker in ["competing_hypotheses:", "uncertainty_sources:", "why_idle:", "is_actionable:", "suggested_reformulation:"]:
                if marker in content_lower[reasoning_start:]:
                    marker_pos = content_lower.index(marker, reasoning_start)
                    reasoning_end = min(reasoning_end, marker_pos)
            result["reasoning"] = content[reasoning_start + 10:reasoning_end].strip()

        # Extract is_actionable
        if "is_actionable: yes" in content_lower or "is_actionable:yes" in content_lower:
            result["is_actionable"] = True
        elif "is_actionable: no" in content_lower or "is_actionable:no" in content_lower:
            result["is_actionable"] = False

        # Extract reformulation
        if "suggested_reformulation:" in content_lower:
            reform_start = content_lower.index("suggested_reformulation:")
            result["reformulation"] = content[reform_start + 24:].strip().split("\n")[0]

        # Extract why_idle
        if "why_idle:" in content_lower:
            idle_start = content_lower.index("why_idle:")
            idle_end = len(content)
            for marker in ["is_actionable:", "suggested_reformulation:"]:
                if marker in content_lower[idle_start:]:
                    idle_end = min(idle_end, content_lower.index(marker, idle_start))
            result["why_idle"] = content[idle_start + 9:idle_end].strip()

        return result

    async def batch_classify(
        self,
        questions: List[str],
        classifier_models: Optional[List[str]] = None,
    ) -> List[EpistemicClassification]:
        """Classify multiple questions in parallel."""
        tasks = [
            self.classify_question(q, classifier_models)
            for q in questions
        ]
        return await asyncio.gather(*tasks)

    def quick_classify(self, question: str) -> EpistemicType:
        """
        Quick heuristic classification without API calls.

        This is useful for pre-filtering before expensive API classification.
        """
        question_lower = question.lower()

        # Factual questions are usually answerable
        factual_markers = [
            "what is", "who is", "where is", "when did", "how many",
            "what are", "who are", "where are", "when was", "how much"
        ]
        if any(question_lower.startswith(marker) for marker in factual_markers):
            return EpistemicType.ANSWERABLE

        # Future predictions are uncertain
        future_markers = ["will", "going to", "predict", "forecast", "expect"]
        if any(marker in question_lower for marker in future_markers):
            return EpistemicType.UNCERTAIN

        # Philosophical questions are often idle
        philosophical_markers = [
            "consciousness", "free will", "meaning of life", "existence",
            "soul", "afterlife", "god", "reality", "subjective experience",
            "qualia", "what is it like to be"
        ]
        if any(marker in question_lower for marker in philosophical_markers):
            return EpistemicType.IDLE

        # Hypotheticals about untestable alternatives
        counterfactual_markers = [
            "what if", "would have", "could have", "might have been"
        ]
        if any(marker in question_lower for marker in counterfactual_markers):
            return EpistemicType.UNDERDETERMINED

        # Default to uncertain
        return EpistemicType.UNCERTAIN
