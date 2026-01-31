"""
Core message schema and types for the AI-to-AI Communication Protocol.

This defines the standardized format for inter-AI communication,
including message types, response structures, and classification enums.

Supports multimodal content (images, audio, video) when the multimodal
feature flag is enabled.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
import uuid
import base64


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


# ==================== Multimodal Support ====================


class ContentType(str, Enum):
    """Types of content that can be attached to messages."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class Attachment(BaseModel):
    """
    Multimodal attachment for messages.

    Supports inline base64 data or URL references. When a model doesn't
    support the content type, falls back to the description field.
    """
    content_type: ContentType
    data: Optional[str] = None          # base64-encoded content for inline
    url: Optional[str] = None           # URL reference (https://, data:, etc.)
    mime_type: Optional[str] = None     # e.g., "image/png", "audio/mp3"
    description: Optional[str] = None   # Alt text / fallback for non-supporting models
    filename: Optional[str] = None      # Original filename if applicable

    @model_validator(mode='after')
    def validate_data_or_url(self) -> 'Attachment':
        """Ensure either data or url is provided."""
        if self.data is None and self.url is None:
            raise ValueError("Either 'data' (base64) or 'url' must be provided")
        return self

    def get_base64_data(self) -> Optional[str]:
        """Get base64 data, extracting from data URI if necessary."""
        if self.data:
            return self.data
        if self.url and self.url.startswith("data:"):
            # Extract base64 from data URI: data:mime;base64,<data>
            try:
                _, encoded = self.url.split(",", 1)
                return encoded
            except ValueError:
                return None
        return None

    def infer_mime_type(self) -> Optional[str]:
        """Infer MIME type from data URI or filename."""
        if self.mime_type:
            return self.mime_type
        if self.url and self.url.startswith("data:"):
            try:
                meta = self.url.split(",")[0]  # data:image/png;base64
                return meta.split(":")[1].split(";")[0]
            except (IndexError, ValueError):
                pass
        if self.filename:
            ext = self.filename.lower().split(".")[-1]
            mime_map = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp",
                "mp3": "audio/mp3",
                "wav": "audio/wav",
                "mp4": "video/mp4",
                "pdf": "application/pdf",
            }
            return mime_map.get(ext)
        return None

    @classmethod
    def from_file(cls, path: str, description: Optional[str] = None) -> "Attachment":
        """Create an attachment from a local file path."""
        import mimetypes
        from pathlib import Path

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        mime_type, _ = mimetypes.guess_type(str(file_path))
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

        # Determine content type from MIME
        content_type = ContentType.DOCUMENT
        if mime_type:
            if mime_type.startswith("image/"):
                content_type = ContentType.IMAGE
            elif mime_type.startswith("audio/"):
                content_type = ContentType.AUDIO
            elif mime_type.startswith("video/"):
                content_type = ContentType.VIDEO

        return cls(
            content_type=content_type,
            data=data,
            mime_type=mime_type,
            filename=file_path.name,
            description=description,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        content_type: ContentType = ContentType.IMAGE,
        description: Optional[str] = None,
    ) -> "Attachment":
        """Create an attachment from a URL reference."""
        return cls(
            content_type=content_type,
            url=url,
            description=description,
        )


class Message(BaseModel):
    """
    Standardized message format for AI-to-AI communication.

    This is the fundamental unit of the protocol - all AI interactions
    are encoded as Messages.

    Supports multimodal content via the attachments field when the
    multimodal feature flag is enabled.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    content: str
    sender: Optional[str] = None       # Model identifier (e.g., "gpt-4", "claude-3")
    recipient: Optional[str] = None    # Target model (None = broadcast)
    context: Optional[List["Message"]] = None  # Previous messages in thread
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # For challenge/verify messages
    target_message_id: Optional[str] = None  # Message being challenged/verified

    # Multimodal attachments (images, audio, video, documents)
    attachments: List[Attachment] = Field(default_factory=list)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def has_attachments(self) -> bool:
        """Check if this message has any attachments."""
        return len(self.attachments) > 0

    def get_attachments_by_type(self, content_type: ContentType) -> List[Attachment]:
        """Get all attachments of a specific type."""
        return [a for a in self.attachments if a.content_type == content_type]

    def get_image_attachments(self) -> List[Attachment]:
        """Get all image attachments."""
        return self.get_attachments_by_type(ContentType.IMAGE)

    def add_attachment(self, attachment: Attachment) -> None:
        """Add an attachment to this message."""
        self.attachments.append(attachment)

    def get_text_with_descriptions(self) -> str:
        """
        Get content with attachment descriptions appended.

        Useful for fallback when a model doesn't support multimodal.
        """
        if not self.attachments:
            return self.content

        descriptions = []
        for i, att in enumerate(self.attachments):
            if att.description:
                descriptions.append(f"[Attachment {i+1}: {att.description}]")
            else:
                descriptions.append(f"[Attachment {i+1}: {att.content_type.value}]")

        return f"{self.content}\n\n" + "\n".join(descriptions)


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


# Update forward references
Message.model_rebuild()
