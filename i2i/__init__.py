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
)
from .providers import ProviderRegistry
from .router import (
    ModelRouter,
    TaskClassifier,
    TaskType,
    RoutingStrategy,
    RoutingDecision,
    RoutingResult,
    ModelCapability,
)

__version__ = "0.2.0"
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
    # Provider management
    "ProviderRegistry",
    # Routing
    "ModelRouter",
    "TaskClassifier",
    "TaskType",
    "RoutingStrategy",
    "RoutingDecision",
    "RoutingResult",
    "ModelCapability",
]
