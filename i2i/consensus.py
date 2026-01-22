"""
Consensus and divergence detection for multi-model queries.

This module analyzes responses from multiple AI models to determine
levels of agreement, identify divergences, and synthesize consensus answers.
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .schema import (
    Message,
    MessageType,
    Response,
    ConsensusResult,
    ConsensusLevel,
    ConfidenceLevel,
)
from .providers import ProviderRegistry


class ConsensusEngine:
    """
    Engine for detecting consensus and divergence across AI models.
    """

    def __init__(self, registry: ProviderRegistry):
        self.registry = registry

    async def query_for_consensus(
        self,
        query: str,
        models: List[str],
        context: Optional[List[Message]] = None,
    ) -> ConsensusResult:
        """
        Query multiple models and analyze their consensus.

        Args:
            query: The question/prompt to send
            models: List of model identifiers to query
            context: Optional conversation context

        Returns:
            ConsensusResult with analysis of agreement/disagreement
        """
        # Create the message
        message = Message(
            type=MessageType.QUERY,
            content=query,
            context=context,
        )

        # Query all models in parallel
        responses = await self.registry.query_multiple(message, models)

        # Filter out errors
        valid_responses = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Response):
                valid_responses.append(resp)
            else:
                # Log error but continue
                print(f"Error from {models[i]}: {resp}")

        if not valid_responses:
            raise ValueError("All model queries failed")

        # Analyze consensus
        consensus_level, agreement_matrix = await self._analyze_consensus(valid_responses)

        # Identify divergences
        divergences = self._identify_divergences(valid_responses, agreement_matrix)

        # Cluster responses if there are camps
        clusters = self._cluster_responses(valid_responses, agreement_matrix)

        # Synthesize consensus answer if consensus exists
        consensus_answer = None
        if consensus_level in [ConsensusLevel.HIGH, ConsensusLevel.MEDIUM]:
            consensus_answer = await self._synthesize_consensus(query, valid_responses)

        return ConsensusResult(
            query=query,
            models_queried=[r.model for r in valid_responses],
            responses=valid_responses,
            consensus_level=consensus_level,
            consensus_answer=consensus_answer,
            divergences=divergences,
            agreement_matrix=agreement_matrix,
            clusters=clusters,
        )

    async def _analyze_consensus(
        self, responses: List[Response]
    ) -> Tuple[ConsensusLevel, Dict[str, Dict[str, float]]]:
        """
        Analyze the level of consensus among responses.

        Uses semantic similarity between responses to determine agreement.
        """
        if len(responses) < 2:
            return ConsensusLevel.HIGH, {}

        # Build agreement matrix using simple text similarity
        # In production, you'd use embeddings for semantic similarity
        agreement_matrix = {}

        for i, r1 in enumerate(responses):
            agreement_matrix[r1.model] = {}
            for j, r2 in enumerate(responses):
                if i == j:
                    agreement_matrix[r1.model][r2.model] = 1.0
                else:
                    similarity = self._compute_similarity(r1.content, r2.content)
                    agreement_matrix[r1.model][r2.model] = similarity

        # Compute average pairwise agreement
        similarities = []
        for i, r1 in enumerate(responses):
            for j, r2 in enumerate(responses):
                if i < j:
                    similarities.append(agreement_matrix[r1.model][r2.model])

        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0

        # Determine consensus level
        if avg_similarity >= 0.85:
            level = ConsensusLevel.HIGH
        elif avg_similarity >= 0.6:
            level = ConsensusLevel.MEDIUM
        elif avg_similarity >= 0.3:
            level = ConsensusLevel.LOW
        else:
            # Check for active contradiction
            if self._has_contradictions(responses):
                level = ConsensusLevel.CONTRADICTORY
            else:
                level = ConsensusLevel.NONE

        return level, agreement_matrix

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        This is a simple implementation using word overlap.
        In production, use embedding-based similarity.
        """
        # Normalize and tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                      'from', 'as', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'between', 'under', 'again', 'further',
                      'then', 'once', 'here', 'there', 'when', 'where', 'why',
                      'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                      'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                      'because', 'until', 'while', 'this', 'that', 'these', 'those',
                      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
                      'who', 'whom', 'whose', 'my', 'your', 'his', 'her', 'its',
                      'our', 'their'}

        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if not words1 or not words2:
            return 0.5  # Neutral if no meaningful words

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _has_contradictions(self, responses: List[Response]) -> bool:
        """
        Check if responses contain explicit contradictions.
        """
        # Look for explicit disagreement markers
        disagreement_markers = [
            "no", "not", "false", "incorrect", "wrong",
            "disagree", "contrary", "opposite", "however",
            "actually", "in fact"
        ]

        # Simple heuristic: if responses have opposite boolean conclusions
        positive_count = 0
        negative_count = 0

        for resp in responses:
            content_lower = resp.content.lower()

            # Check for affirmative vs negative conclusions
            if any(marker in content_lower[:200] for marker in ["yes", "correct", "true", "right"]):
                positive_count += 1
            if any(marker in content_lower[:200] for marker in ["no", "incorrect", "false", "wrong"]):
                negative_count += 1

        # Contradiction if there's a split
        return positive_count > 0 and negative_count > 0

    def _identify_divergences(
        self,
        responses: List[Response],
        agreement_matrix: Dict[str, Dict[str, float]],
    ) -> List[Dict]:
        """
        Identify specific points of divergence between responses.
        """
        divergences = []

        for i, r1 in enumerate(responses):
            for j, r2 in enumerate(responses):
                if i < j:
                    similarity = agreement_matrix[r1.model][r2.model]
                    if similarity < 0.5:  # Significant divergence
                        divergences.append({
                            "models": [r1.model, r2.model],
                            "similarity": similarity,
                            "summary": f"{r1.model} and {r2.model} diverge significantly",
                            "model_1_stance": r1.content[:200] + "...",
                            "model_2_stance": r2.content[:200] + "...",
                        })

        return divergences

    def _cluster_responses(
        self,
        responses: List[Response],
        agreement_matrix: Dict[str, Dict[str, float]],
    ) -> Optional[List[List[str]]]:
        """
        Cluster responses into groups of agreeing models.
        """
        if len(responses) < 3:
            return None

        # Simple clustering: group models with high pairwise similarity
        clusters = []
        assigned = set()

        for r1 in responses:
            if r1.model in assigned:
                continue

            cluster = [r1.model]
            assigned.add(r1.model)

            for r2 in responses:
                if r2.model in assigned:
                    continue
                if agreement_matrix[r1.model][r2.model] >= 0.7:
                    cluster.append(r2.model)
                    assigned.add(r2.model)

            if cluster:
                clusters.append(cluster)

        # Add any remaining as singletons
        for r in responses:
            if r.model not in assigned:
                clusters.append([r.model])

        return clusters if len(clusters) > 1 else None

    async def _synthesize_consensus(
        self,
        query: str,
        responses: List[Response],
    ) -> str:
        """
        Synthesize a consensus answer from multiple responses.

        Uses one of the models to create a synthesis.
        """
        # Build a synthesis prompt
        responses_text = "\n\n".join([
            f"Model {r.model}:\n{r.content}"
            for r in responses
        ])

        synthesis_prompt = f"""Multiple AI models were asked: "{query}"

Their responses were:

{responses_text}

Please synthesize these responses into a single, coherent answer that:
1. Captures the points of agreement
2. Notes any significant differences in perspective
3. Provides the most accurate and complete answer based on the consensus

Synthesized answer:"""

        # Use the first available model to synthesize
        message = Message(
            type=MessageType.SYNTHESIZE,
            content=synthesis_prompt,
        )

        # Try to use a capable model for synthesis
        synthesis_models = ["gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro"]
        for model in synthesis_models:
            try:
                response = await self.registry.query(message, model)
                return response.content
            except Exception:
                continue

        # Fallback: return the response with highest confidence
        best_response = max(responses, key=lambda r: self._confidence_score(r.confidence))
        return best_response.content

    def _confidence_score(self, confidence: ConfidenceLevel) -> int:
        """Convert confidence level to numeric score."""
        scores = {
            ConfidenceLevel.VERY_HIGH: 5,
            ConfidenceLevel.HIGH: 4,
            ConfidenceLevel.MEDIUM: 3,
            ConfidenceLevel.LOW: 2,
            ConfidenceLevel.VERY_LOW: 1,
        }
        return scores.get(confidence, 3)
