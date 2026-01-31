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
    feature_enabled,
    DEFAULTS,
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

__version__ = "0.1.0"
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
    "feature_enabled",
    # Search/RAG
    "SearchBackend",
    "SearchResult",
    "SearchRegistry",
    "BraveSearchBackend",
    "SerpAPIBackend",
    "TavilySearchBackend",
]
