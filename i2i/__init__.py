"""
i2i - AI-to-AI Communication Protocol

When AIs see eye to eye: Multi-model consensus, cross-verification,
epistemic classification, and intelligent routing for trustworthy AI outputs.
"""

from .protocol import AICP
from .schema import (
    Message,
    Response,
    ConsensusResult,
    VerificationResult,
    EpistemicClassification,
    EpistemicType,
    ConsensusLevel,
    ModelStatistics,
    StatisticalConsensusResult,
    # Multimodal support
    ContentType,
    Attachment,
)
from .providers import ProviderRegistry, model_supports_vision, VISION_CAPABLE_MODELS
from .router import (
    ModelRouter,
    TaskClassifier,
    TaskType,
    RoutingStrategy,
    RoutingDecision,
    RoutingResult,
    ModelCapability,
)
from .config import (
    Config,
    get_config,
    set_config,
    reset_config,
    get_consensus_models,
    get_classifier_model,
    get_synthesis_models,
    get_verification_models,
    get_epistemic_models,
    get_statistical_mode_config,
    is_statistical_mode_enabled,
    get_statistical_n_runs,
    get_statistical_temperature,
    feature_enabled,
    DEFAULTS,
)
from .embeddings import (
    EmbeddingProvider,
    cosine_similarity,
    compute_centroid,
    compute_std_dev,
    consistency_score,
)
from .consensus import (
    ConsensusEngine,
    ConsortiumType,
    ModelFamily,
    detect_model_family,
    detect_consortium_type,
)
from .search import (
    SearchBackend,
    SearchResult,
    SearchRegistry,
    BraveSearchBackend,
    SerpAPIBackend,
    TavilySearchBackend,
)
from .task_classifier import (
    ConsensusTaskCategory,
    ConsensusRecommendation,
    recommend_consensus,
    is_consensus_appropriate,
    get_task_category,
    should_warn_about_consensus,
    get_confidence_calibration,
    router_task_to_consensus_category,
)

# LangChain integration (optional)
try:
    from .integrations.langchain import (
        I2IVerifier,
        I2IVerifiedOutput,
    )

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

__version__ = "0.2.1"  # LangChain integration
__all__ = [
    # Core protocol
    "AICP",
    # Schema types
    "Message",
    "Response",
    "ConsensusResult",
    "VerificationResult",
    "EpistemicClassification",
    "EpistemicType",
    "ConsensusLevel",
    "ModelStatistics",
    "StatisticalConsensusResult",
    # Multimodal support
    "ContentType",
    "Attachment",
    # Provider management
    "ProviderRegistry",
    "model_supports_vision",
    "VISION_CAPABLE_MODELS",
    # Consensus & Consortium
    "ConsensusEngine",
    "ConsortiumType",
    "ModelFamily",
    "detect_model_family",
    "detect_consortium_type",
    # Routing
    "ModelRouter",
    "TaskClassifier",
    "TaskType",
    "RoutingStrategy",
    "RoutingDecision",
    "RoutingResult",
    "ModelCapability",
    # Configuration
    "Config",
    "DEFAULTS",
    "get_config",
    "set_config",
    "reset_config",
    "get_consensus_models",
    "get_classifier_model",
    "get_synthesis_models",
    "get_verification_models",
    "get_epistemic_models",
    "get_statistical_mode_config",
    "is_statistical_mode_enabled",
    "get_statistical_n_runs",
    "get_statistical_temperature",
    "feature_enabled",
    # Embeddings
    "EmbeddingProvider",
    "cosine_similarity",
    "compute_centroid",
    "compute_std_dev",
    "consistency_score",
    # Search/RAG
    "SearchBackend",
    "SearchResult",
    "SearchRegistry",
    "BraveSearchBackend",
    "SerpAPIBackend",
    "TavilySearchBackend",
    # Task-aware consensus
    "ConsensusTaskCategory",
    "ConsensusRecommendation",
    "recommend_consensus",
    "is_consensus_appropriate",
    "get_task_category",
    "should_warn_about_consensus",
    "get_confidence_calibration",
    "router_task_to_consensus_category",
]

# Add LangChain exports if available
if _LANGCHAIN_AVAILABLE:
    __all__.extend([
        "I2IVerifier",
        "I2IVerifiedOutput",
    ])
