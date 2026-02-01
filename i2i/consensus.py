"""
Consensus and divergence detection for multi-model queries.

This module analyzes responses from multiple AI models to determine
levels of agreement, identify divergences, and synthesize consensus answers.

Supports:
1. Standard mode: Query each model once, compute pairwise similarity
2. Statistical mode: Query each model n times, compute variance for confidence
3. Homogeneous consortium detection and optimization when models
   are from the same family (e.g., all Claude, all GPT).
"""

import asyncio
import logging
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict
import numpy as np

from .schema import (
    Message,
    MessageType,
    Response,
    ConsensusResult,
    ConsensusLevel,
    ConfidenceLevel,
    ModelStatistics,
    StatisticalConsensusResult,
)
from .providers import ProviderRegistry
from .config import (
    get_synthesis_models,
    get_statistical_mode_config,
    is_statistical_mode_enabled,
    get_statistical_n_runs,
    get_statistical_temperature,
    get_statistical_outlier_threshold,
    feature_enabled,
    get_high_threshold,
    get_medium_threshold,
    get_low_threshold,
    get_clustering_threshold,
    get_divergence_threshold,
)
from .embeddings import (
    EmbeddingProvider,
    compute_centroid,
    compute_std_dev,
    find_outliers,
    find_representative,
    compute_weighted_centroid,
    consistency_score,
    cosine_similarity,
)

logger = logging.getLogger(__name__)


class ConsortiumType(str, Enum):
    """Classification of model consortium diversity."""
    HETEROGENEOUS = "heterogeneous"  # Different model families (ideal for diversity)
    HOMOGENEOUS = "homogeneous"      # Same family (all Claude, all GPT, etc.)
    MIXED = "mixed"                  # Mostly same with some different


class ModelFamily(str, Enum):
    """Known model families/providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    META = "meta"          # Llama models
    COHERE = "cohere"
    UNKNOWN = "unknown"


def detect_model_family(model: str) -> ModelFamily:
    """
    Detect the model family from a model identifier.

    Args:
        model: Model identifier (e.g., "claude-3-sonnet", "gpt-4o", "gemini-1.5-pro")

    Returns:
        The detected ModelFamily.
    """
    model_lower = model.lower()

    # Strip provider prefix if present
    if "/" in model_lower:
        model_lower = model_lower.split("/")[-1]

    # Anthropic (Claude)
    if any(x in model_lower for x in ["claude", "anthropic"]):
        return ModelFamily.ANTHROPIC

    # OpenAI (GPT, O-series)
    if any(x in model_lower for x in ["gpt", "o3", "o4", "openai", "codex"]):
        return ModelFamily.OPENAI

    # Google (Gemini)
    if any(x in model_lower for x in ["gemini", "google", "palm"]):
        return ModelFamily.GOOGLE

    # Mistral
    if any(x in model_lower for x in ["mistral", "mixtral", "codestral", "devstral", "ministral"]):
        return ModelFamily.MISTRAL

    # Meta (Llama)
    if any(x in model_lower for x in ["llama", "meta"]):
        return ModelFamily.META

    # Cohere
    if any(x in model_lower for x in ["command", "cohere"]):
        return ModelFamily.COHERE

    return ModelFamily.UNKNOWN


def detect_consortium_type(models: List[str]) -> Tuple[ConsortiumType, Dict[ModelFamily, List[str]]]:
    """
    Analyze a list of models to determine consortium type.

    Args:
        models: List of model identifiers.

    Returns:
        Tuple of (ConsortiumType, family_mapping) where family_mapping
        maps each family to its models.
    """
    family_mapping: Dict[ModelFamily, List[str]] = defaultdict(list)

    for model in models:
        family = detect_model_family(model)
        family_mapping[family].append(model)

    # Count unique families (excluding UNKNOWN)
    known_families = {f for f in family_mapping.keys() if f != ModelFamily.UNKNOWN}
    num_families = len(known_families)

    if num_families == 0:
        # All unknown - treat as heterogeneous (we don't know)
        return ConsortiumType.HETEROGENEOUS, dict(family_mapping)
    elif num_families == 1:
        return ConsortiumType.HOMOGENEOUS, dict(family_mapping)
    elif num_families >= len(models) * 0.8:
        # 80%+ different families = heterogeneous
        return ConsortiumType.HETEROGENEOUS, dict(family_mapping)
    else:
        return ConsortiumType.MIXED, dict(family_mapping)


class ConsensusEngine:
    """
    Engine for detecting consensus and divergence across AI models.

    Supports homogeneous consortium optimization when all models are
    from the same family and the feature flag is enabled.
    """

    def __init__(self, registry: ProviderRegistry):
        self.registry = registry

    def analyze_consortium(self, models: List[str]) -> Tuple[ConsortiumType, Dict[ModelFamily, List[str]]]:
        """
        Analyze the consortium of models for diversity.

        Returns consortium type and family breakdown.
        """
        return detect_consortium_type(models)

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
        # Analyze consortium type
        consortium_type, family_mapping = self.analyze_consortium(models)

        # Log consortium info and apply optimizations if enabled
        if feature_enabled("homogeneous_optimization"):
            if consortium_type == ConsortiumType.HOMOGENEOUS:
                family = list(family_mapping.keys())[0]
                logger.info(
                    f"Homogeneous consortium detected: all models from {family.value} family. "
                    "Diversity metrics may be less meaningful (correlated errors possible)."
                )
            elif consortium_type == ConsortiumType.MIXED:
                logger.info(
                    f"Mixed consortium: {dict(family_mapping)}. "
                    "Consider using fully heterogeneous models for better diversity."
                )

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
                logger.warning(f"Error from {models[i]}: {resp}")

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

        # Build metadata with consortium info
        metadata = {
            "consortium_type": consortium_type.value,
            "family_breakdown": {k.value: v for k, v in family_mapping.items()},
        }

        # Add warning if homogeneous
        if consortium_type == ConsortiumType.HOMOGENEOUS:
            metadata["warning"] = (
                "Homogeneous consortium: all models from same family. "
                "High agreement may reflect shared training biases rather than true consensus."
            )

        return ConsensusResult(
            query=query,
            models_queried=[r.model for r in valid_responses],
            responses=valid_responses,
            consensus_level=consensus_level,
            consensus_answer=consensus_answer,
            divergences=divergences,
            agreement_matrix=agreement_matrix,
            clusters=clusters,
            metadata=metadata,
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

        # Get configurable thresholds
        high_thresh = get_high_threshold()
        medium_thresh = get_medium_threshold()
        low_thresh = get_low_threshold()

        # Determine consensus level using configurable thresholds
        if avg_similarity >= high_thresh:
            level = ConsensusLevel.HIGH
        elif avg_similarity >= medium_thresh:
            level = ConsensusLevel.MEDIUM
        elif avg_similarity >= low_thresh:
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
        divergence_thresh = get_divergence_threshold()

        for i, r1 in enumerate(responses):
            for j, r2 in enumerate(responses):
                if i < j:
                    similarity = agreement_matrix[r1.model][r2.model]
                    if similarity < divergence_thresh:  # Significant divergence
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
        clustering_thresh = get_clustering_threshold()

        for r1 in responses:
            if r1.model in assigned:
                continue

            cluster = [r1.model]
            assigned.add(r1.model)

            for r2 in responses:
                if r2.model in assigned:
                    continue
                if agreement_matrix[r1.model][r2.model] >= clustering_thresh:
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

        # Try to use a capable model for synthesis (configurable)
        synthesis_models = get_synthesis_models()
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

    # ==================== Statistical Mode ====================

    async def query_for_consensus_statistical(
        self,
        query: str,
        models: List[str],
        n_runs: Optional[int] = None,
        temperature: Optional[float] = None,
        outlier_threshold: Optional[float] = None,
        context: Optional[List[Message]] = None,
    ) -> StatisticalConsensusResult:
        """
        Query multiple models with n runs each for statistical consensus.

        This method runs each model n times and computes variance to
        estimate model confidence. Models with lower variance are weighted
        higher in the consensus.

        Args:
            query: The question/prompt to send
            models: List of model identifiers to query
            n_runs: Number of runs per model (default from config)
            temperature: Temperature for queries (default from config)
            outlier_threshold: Std devs for outlier detection (default from config)
            context: Optional conversation context

        Returns:
            StatisticalConsensusResult with variance analysis
        """
        # Get config defaults
        config = get_statistical_mode_config()
        n_runs = n_runs or config.get("n_runs", 5)
        temperature = temperature if temperature is not None else config.get("temperature", 0.7)
        outlier_threshold = outlier_threshold or config.get("outlier_threshold", 2.0)

        # Initialize embedding provider
        embedding_provider = EmbeddingProvider()

        # Create the message
        message = Message(
            type=MessageType.QUERY,
            content=query,
            context=context,
        )

        # Query each model n times in parallel
        all_model_responses: Dict[str, List[Response]] = {}
        all_tasks = []

        for model in models:
            model_tasks = [
                self._query_with_temperature(message, model, temperature)
                for _ in range(n_runs)
            ]
            all_tasks.extend([(model, task) for task in model_tasks])

        # Flatten and execute all queries in parallel
        flat_tasks = [task for _, task in all_tasks]
        results = await asyncio.gather(*flat_tasks, return_exceptions=True)

        # Organize results by model
        idx = 0
        for model in models:
            all_model_responses[model] = []
            for _ in range(n_runs):
                result = results[idx]
                if isinstance(result, Response):
                    all_model_responses[model].append(result)
                else:
                    print(f"Error from {model} run {idx}: {result}")
                idx += 1

        # Filter models with at least 1 successful response
        valid_models = [m for m in models if all_model_responses[m]]
        if not valid_models:
            raise ValueError("All model queries failed")

        # Compute statistics for each model
        model_statistics: Dict[str, ModelStatistics] = {}
        model_centroids: List[np.ndarray] = []
        model_variances: List[float] = []

        for model in valid_models:
            responses = all_model_responses[model]
            stats = await self._compute_model_statistics(
                model, responses, embedding_provider, outlier_threshold
            )
            model_statistics[model] = stats

            if stats.centroid_embedding:
                model_centroids.append(np.array(stats.centroid_embedding))
                model_variances.append(stats.intra_model_std_dev ** 2)

        # Compute inter-model consensus using representative responses
        representative_responses = [
            stats.representative_response
            for stats in model_statistics.values()
            if stats.representative_response
        ]

        # Analyze consensus on representative responses
        consensus_level, agreement_matrix = await self._analyze_consensus(representative_responses)

        # Identify divergences
        divergences = self._identify_divergences(representative_responses, agreement_matrix)

        # Cluster responses
        clusters = self._cluster_responses(representative_responses, agreement_matrix)

        # Compute weighted consensus centroid
        weighted_centroid = None
        if model_centroids and model_variances:
            weighted_centroid = compute_weighted_centroid(model_centroids, model_variances)

        # Compute overall confidence
        avg_consistency = sum(s.consistency_score for s in model_statistics.values()) / len(model_statistics)

        # Synthesize consensus answer
        consensus_answer = None
        if consensus_level in [ConsensusLevel.HIGH, ConsensusLevel.MEDIUM]:
            consensus_answer = await self._synthesize_consensus(query, representative_responses)

        return StatisticalConsensusResult(
            query=query,
            models_queried=valid_models,
            n_runs_per_model=n_runs,
            temperature=temperature,
            model_statistics=model_statistics,
            consensus_level=consensus_level,
            consensus_answer=consensus_answer,
            divergences=divergences,
            agreement_matrix=agreement_matrix,
            clusters=clusters,
            weighted_consensus_embedding=weighted_centroid.tolist() if weighted_centroid is not None else None,
            overall_confidence=avg_consistency,
            total_queries=len(valid_models) * n_runs,
            total_cost_multiplier=float(n_runs),
        )

    async def _query_with_temperature(
        self,
        message: Message,
        model: str,
        temperature: float,
    ) -> Response:
        """Query a model with specific temperature setting."""
        # Note: temperature parameter would need to be passed through the provider
        # For now, we use the default provider query
        # TODO: Add temperature parameter to provider adapters
        return await self.registry.query(message, model)

    async def _compute_model_statistics(
        self,
        model: str,
        responses: List[Response],
        embedding_provider: EmbeddingProvider,
        outlier_threshold: float,
    ) -> ModelStatistics:
        """
        Compute statistics for a single model's n responses.

        Args:
            model: Model identifier
            responses: List of n responses from the model
            embedding_provider: Provider for text embeddings
            outlier_threshold: Std devs for outlier detection

        Returns:
            ModelStatistics with variance analysis
        """
        if not responses:
            return ModelStatistics(
                model=model,
                n_runs=0,
                consistency_score=0.0,
            )

        # Get embeddings for all responses
        texts = [r.content for r in responses]
        embeddings = await embedding_provider.embed_batch(texts)

        # Compute centroid
        centroid = compute_centroid(embeddings)

        # Compute standard deviation
        std_dev = compute_std_dev(embeddings, centroid)

        # Find outliers
        outlier_indices = find_outliers(embeddings, centroid, outlier_threshold)

        # Find representative response (closest to centroid)
        representative_idx = find_representative(embeddings, centroid)
        representative = responses[representative_idx]

        return ModelStatistics(
            model=model,
            n_runs=len(responses),
            centroid_embedding=centroid.tolist(),
            intra_model_std_dev=std_dev,
            consistency_score=consistency_score(std_dev),
            representative_response=representative,
            outlier_indices=outlier_indices,
            all_responses=responses,
        )

    async def _analyze_consensus_with_embeddings(
        self,
        responses: List[Response],
        embedding_provider: EmbeddingProvider,
    ) -> Tuple[ConsensusLevel, Dict[str, Dict[str, float]]]:
        """
        Analyze consensus using embedding-based similarity.

        This is more accurate than Jaccard similarity for semantic comparison.
        """
        if len(responses) < 2:
            return ConsensusLevel.HIGH, {}

        # Get embeddings
        texts = [r.content for r in responses]
        embeddings = await embedding_provider.embed_batch(texts)

        # Build agreement matrix
        agreement_matrix = {}
        for i, r1 in enumerate(responses):
            agreement_matrix[r1.model] = {}
            for j, r2 in enumerate(responses):
                if i == j:
                    agreement_matrix[r1.model][r2.model] = 1.0
                else:
                    similarity = cosine_similarity(embeddings[i], embeddings[j])
                    agreement_matrix[r1.model][r2.model] = float(similarity)

        # Compute average pairwise similarity
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarities.append(
                    agreement_matrix[responses[i].model][responses[j].model]
                )

        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0

        # Get configurable thresholds
        high_thresh = get_high_threshold()
        medium_thresh = get_medium_threshold()
        low_thresh = get_low_threshold()

        # Determine consensus level using configurable thresholds
        if avg_similarity >= high_thresh:
            level = ConsensusLevel.HIGH
        elif avg_similarity >= medium_thresh:
            level = ConsensusLevel.MEDIUM
        elif avg_similarity >= low_thresh:
            level = ConsensusLevel.LOW
        else:
            if self._has_contradictions(responses):
                level = ConsensusLevel.CONTRADICTORY
            else:
                level = ConsensusLevel.NONE

        return level, agreement_matrix
