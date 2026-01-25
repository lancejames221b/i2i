"""
Core message schema and types for the AI-to-AI Communication Protocol.

This defines the standardized format for inter-AI communication,
including message types, response structures, and classification enums.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid


class MessageType(str, Enum):
    """Types of messages in the AI-to-AI protocol."""
    QUERY = "query"                    # Standard question/prompt
    CHALLENGE = "challenge"            # Challenge another AI's response
    VERIFY = "verify"                  # Request verification of a claim
    SYNTHESIZE = "synthesize"          # Request synthesis of multiple responses
    CLASSIFY = "classify"              # Request epistemic classification
    META = "meta"                      # Meta-communication about the protocol itself


class EpistemicType(str, Enum):
    """Classification of question/claim epistemics."""
    ANSWERABLE = "answerable"          # Can be resolved with available information
    UNCERTAIN = "uncertain"            # Answerable but with significant uncertainty
    UNDERDETERMINED = "underdetermined" # Multiple hypotheses fit data equally
    IDLE = "idle"                      # Well-formed but non-action-guiding
    MALFORMED = "malformed"            # Question is incoherent or self-contradictory


class ConsensusLevel(str, Enum):
    """Level of agreement between AI models."""
    HIGH = "high"           # Strong agreement (>90% semantic similarity)
    MEDIUM = "medium"       # Moderate agreement (60-90%)
    LOW = "low"             # Weak agreement (30-60%)
    NONE = "none"           # No meaningful agreement (<30%)
    CONTRADICTORY = "contradictory"  # Active disagreement/contradiction


class ConfidenceLevel(str, Enum):
    """Self-reported confidence in a response."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class Message(BaseModel):
    """
    Standardized message format for AI-to-AI communication.

    This is the fundamental unit of the protocol - all AI interactions
    are encoded as Messages.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    content: str
    sender: Optional[str] = None       # Model identifier (e.g., "gpt-4", "claude-3")
    recipient: Optional[str] = None    # Target model (None = broadcast)
    context: Optional[List["Message"]] = None  # Previous messages in thread
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # For challenge/verify messages
    target_message_id: Optional[str] = None  # Message being challenged/verified

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Response(BaseModel):
    """
    Standardized response from an AI model.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_id: str                    # ID of message being responded to
    model: str                         # Model that generated this response
    content: str                       # The actual response text
    confidence: ConfidenceLevel        # Self-assessed confidence
    reasoning: Optional[str] = None    # Chain-of-thought or explanation
    caveats: List[str] = Field(default_factory=list)  # Stated limitations
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Token/cost tracking
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None

    # RAG/Search citations
    citations: Optional[List[str]] = None  # Source URLs from RAG providers


class ConsensusResult(BaseModel):
    """
    Result of a consensus query across multiple models.
    """
    query: str
    models_queried: List[str]
    responses: List[Response]

    # Consensus analysis
    consensus_level: ConsensusLevel
    consensus_answer: Optional[str] = None  # Synthesized answer if consensus exists

    # Divergence analysis
    divergences: List[Dict[str, Any]] = Field(default_factory=list)
    agreement_matrix: Optional[Dict[str, Dict[str, float]]] = None  # Pairwise similarity

    # Clustering (for when there are multiple "camps")
    clusters: Optional[List[List[str]]] = None  # Groups of agreeing models

    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """
    Result of cross-verification of a claim.
    """
    original_claim: str
    original_source: Optional[str] = None  # Model that made the claim

    verifiers: List[str]               # Models that verified
    verification_responses: List[Response]

    # Verification outcome
    verified: bool
    confidence: float                  # 0-1 confidence in verification
    issues_found: List[str] = Field(default_factory=list)
    corrections: Optional[str] = None  # Suggested corrections if not verified

    # RAG/Search grounding
    source_citations: List[str] = Field(default_factory=list)  # Source URLs used
    retrieved_sources: List[Dict[str, Any]] = Field(default_factory=list)  # Full source info

    metadata: Dict[str, Any] = Field(default_factory=dict)


class EpistemicClassification(BaseModel):
    """
    Classification of a question's epistemic status.
    """
    question: str
    classification: EpistemicType
    confidence: float                  # 0-1 confidence in classification
    reasoning: str                     # Why this classification

    # For underdetermined questions
    competing_hypotheses: Optional[List[str]] = None

    # For uncertain questions
    uncertainty_sources: Optional[List[str]] = None

    # For idle questions
    why_idle: Optional[str] = None     # Explanation of why non-action-guiding

    # Actionability
    is_actionable: bool
    suggested_reformulation: Optional[str] = None  # More tractable version

    metadata: Dict[str, Any] = Field(default_factory=dict)


class DivergenceReport(BaseModel):
    """
    Report on systematic divergences between models.
    """
    models_compared: List[str]
    num_queries: int

    # Divergence patterns
    systematic_divergences: List[Dict[str, Any]]
    agreement_rate: float              # Overall agreement rate

    # Per-model analysis
    model_profiles: Dict[str, Dict[str, Any]]  # Epistemic "fingerprints"

    # Recommendations
    best_model_for: Dict[str, str]     # Task type -> recommended model

    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelStatistics(BaseModel):
    """
    Per-model statistics from n runs (statistical consensus mode).

    This captures the variance/consistency of a single model's responses
    across multiple runs, enabling confidence estimation.
    """
    model: str
    n_runs: int
    centroid_embedding: Optional[List[float]] = None  # Mean position in embedding space
    intra_model_std_dev: float = 0.0  # Spread of responses (lower = more consistent)
    consistency_score: float = 1.0  # 1 / (1 + std_dev), higher = more confident
    representative_response: Optional[Response] = None  # Response closest to centroid
    outlier_indices: List[int] = Field(default_factory=list)  # Indices of outlier responses
    all_responses: List[Response] = Field(default_factory=list)  # All n responses


class StatisticalConsensusResult(BaseModel):
    """
    Enhanced consensus result with statistical measures from n-run averaging.

    When statistical_mode is enabled, each model is queried n times and
    variance is computed to estimate model confidence and detect outliers.
    """
    # Core query info
    query: str
    models_queried: List[str]

    # Statistical configuration
    n_runs_per_model: int
    temperature: float

    # Per-model statistics
    model_statistics: Dict[str, ModelStatistics] = Field(default_factory=dict)

    # Aggregate consensus (same as ConsensusResult)
    consensus_level: ConsensusLevel
    consensus_answer: Optional[str] = None
    divergences: List[Dict[str, Any]] = Field(default_factory=list)
    agreement_matrix: Optional[Dict[str, Dict[str, float]]] = None
    clusters: Optional[List[List[str]]] = None

    # Statistical enhancements
    weighted_consensus_embedding: Optional[List[float]] = None  # Inverse-variance weighted centroid
    overall_confidence: float = 0.0  # Aggregate confidence based on inter/intra variance
    total_queries: int = 0  # n_runs * num_models
    total_cost_multiplier: float = 1.0  # Cost relative to single-query mode

    metadata: Dict[str, Any] = Field(default_factory=dict)


# Update forward references
Message.model_rebuild()
ModelStatistics.model_rebuild()
