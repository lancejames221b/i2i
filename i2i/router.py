"""
Intelligent Model Router for i2i Protocol.

Routes queries to the optimal model(s) based on:
- Task type classification
- Model capability profiles
- Cost/speed/quality trade-offs
- Historical performance data
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import json
import os

from .schema import Response, ConfidenceLevel
from .providers import ProviderRegistry, ProviderAdapter
from .config import get_classifier_model


class TaskType(str, Enum):
    """Classification of query task types."""
    # Reasoning & Analysis
    LOGICAL_REASONING = "logical_reasoning"
    MATHEMATICAL = "mathematical"
    SCIENTIFIC = "scientific"
    ANALYTICAL = "analytical"

    # Creative
    CREATIVE_WRITING = "creative_writing"
    COPYWRITING = "copywriting"
    BRAINSTORMING = "brainstorming"

    # Technical
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_DEBUGGING = "code_debugging"
    TECHNICAL_DOCS = "technical_docs"

    # Knowledge & Research
    FACTUAL_QA = "factual_qa"
    RESEARCH = "research"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"

    # Conversation
    CHAT = "chat"
    ROLEPLAY = "roleplay"
    INSTRUCTION_FOLLOWING = "instruction_following"

    # Specialized
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"

    # Meta
    UNKNOWN = "unknown"


class RoutingStrategy(str, Enum):
    """Strategy for model selection."""
    BEST_QUALITY = "best_quality"      # Optimize for output quality
    BEST_SPEED = "best_speed"          # Optimize for latency
    BEST_VALUE = "best_value"          # Optimize for cost-effectiveness
    BALANCED = "balanced"              # Balance all factors
    ENSEMBLE = "ensemble"              # Use multiple models and synthesize
    FALLBACK_CHAIN = "fallback_chain"  # Try models in order until success


class ModelCapability(BaseModel):
    """Capability profile for a single model."""
    model_id: str
    provider: str

    # Task scores (0-100, higher is better)
    task_scores: Dict[str, float] = Field(default_factory=dict)

    # Performance metrics
    avg_latency_ms: float = 1000.0
    cost_per_1k_tokens: float = 0.01
    context_window: int = 8192
    max_output_tokens: int = 4096

    # Reliability
    uptime_percent: float = 99.0
    error_rate: float = 0.01

    # Special capabilities
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = True

    # Quality indicators
    reasoning_depth: float = 70.0      # 0-100
    creativity_score: float = 70.0     # 0-100
    instruction_following: float = 80.0 # 0-100
    factual_accuracy: float = 75.0     # 0-100


class RoutingDecision(BaseModel):
    """Result of a routing decision."""
    query: str
    detected_task: TaskType
    task_confidence: float

    selected_models: List[str]
    strategy_used: RoutingStrategy

    reasoning: str
    estimated_cost: Optional[float] = None
    estimated_latency_ms: Optional[float] = None

    alternatives: List[Dict[str, Any]] = Field(default_factory=list)


class RoutingResult(BaseModel):
    """Full result including routing decision and response."""
    decision: RoutingDecision
    responses: List[Response]

    # If ensemble/synthesis was used
    synthesized_response: Optional[str] = None

    # Performance tracking
    actual_latency_ms: float
    actual_cost: Optional[float] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)


# Default capability profiles based on empirical observations
# Updated January 2026 with latest model versions
DEFAULT_MODEL_CAPABILITIES: Dict[str, ModelCapability] = {
    # ==================== Anthropic (Claude 4.5 Series) ====================
    # Claude Opus 4.5 - Most intelligent model (Nov 2025)
    "claude-opus-4-5-20251101": ModelCapability(
        model_id="claude-opus-4-5-20251101",
        provider="anthropic",
        task_scores={
            TaskType.CODE_GENERATION: 98,
            TaskType.CODE_REVIEW: 98,
            TaskType.CODE_DEBUGGING: 97,
            TaskType.LOGICAL_REASONING: 99,
            TaskType.ANALYTICAL: 99,
            TaskType.CREATIVE_WRITING: 97,
            TaskType.RESEARCH: 98,
            TaskType.SCIENTIFIC: 97,
            TaskType.LEGAL: 95,
            TaskType.MEDICAL: 93,
            TaskType.INSTRUCTION_FOLLOWING: 98,
        },
        avg_latency_ms=1500,
        cost_per_1k_tokens=0.005,  # $5 input / $25 output per MTok
        context_window=200000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=99,
        creativity_score=97,
        instruction_following=98,
        factual_accuracy=96,
    ),
    # Claude Sonnet 4.5 - Best for agents & coding (Sep 2025)
    "claude-sonnet-4-5-20250929": ModelCapability(
        model_id="claude-sonnet-4-5-20250929",
        provider="anthropic",
        task_scores={
            TaskType.CODE_GENERATION: 97,
            TaskType.CODE_REVIEW: 97,
            TaskType.CODE_DEBUGGING: 95,
            TaskType.CREATIVE_WRITING: 94,
            TaskType.ANALYTICAL: 93,
            TaskType.LOGICAL_REASONING: 92,
            TaskType.INSTRUCTION_FOLLOWING: 97,
            TaskType.TECHNICAL_DOCS: 94,
            TaskType.RESEARCH: 92,
            TaskType.SUMMARIZATION: 93,
            TaskType.CHAT: 92,
        },
        avg_latency_ms=700,
        cost_per_1k_tokens=0.003,  # $3 input / $15 output per MTok
        context_window=1000000,  # 1M context in preview
        max_output_tokens=8192,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=93,
        creativity_score=92,
        instruction_following=97,
        factual_accuracy=91,
    ),
    # Claude Haiku 4.5 - Fastest Claude model
    "claude-haiku-4-5-20251001": ModelCapability(
        model_id="claude-haiku-4-5-20251001",
        provider="anthropic",
        task_scores={
            TaskType.CHAT: 88,
            TaskType.SUMMARIZATION: 88,
            TaskType.FACTUAL_QA: 85,
            TaskType.TRANSLATION: 88,
            TaskType.INSTRUCTION_FOLLOWING: 90,
            TaskType.CODE_GENERATION: 82,
        },
        avg_latency_ms=200,
        cost_per_1k_tokens=0.001,  # $1 input / $5 output per MTok
        context_window=200000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=78,
        creativity_score=76,
        instruction_following=90,
        factual_accuracy=82,
    ),

    # ==================== OpenAI (Full 2025/2026 Model Lineup) ====================

    # --- GPT-5 Series (Flagship - Aug-Dec 2025) ---
    # GPT-5.2 - Latest flagship (Dec 2025)
    "gpt-5.2": ModelCapability(
        model_id="gpt-5.2",
        provider="openai",
        task_scores={
            TaskType.CODE_GENERATION: 98,
            TaskType.LOGICAL_REASONING: 99,
            TaskType.MATHEMATICAL: 98,
            TaskType.ANALYTICAL: 98,
            TaskType.RESEARCH: 97,
            TaskType.CREATIVE_WRITING: 95,
            TaskType.SCIENTIFIC: 97,
            TaskType.INSTRUCTION_FOLLOWING: 98,
            TaskType.FACTUAL_QA: 96,
        },
        avg_latency_ms=1200,
        cost_per_1k_tokens=0.008,
        context_window=196000,
        max_output_tokens=32768,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=99,
        creativity_score=95,
        instruction_following=98,
        factual_accuracy=96,
    ),
    # GPT-5 - Previous flagship (Aug 2025)
    "gpt-5": ModelCapability(
        model_id="gpt-5",
        provider="openai",
        task_scores={
            TaskType.CODE_GENERATION: 97,
            TaskType.LOGICAL_REASONING: 97,
            TaskType.MATHEMATICAL: 96,
            TaskType.ANALYTICAL: 96,
            TaskType.RESEARCH: 95,
            TaskType.CREATIVE_WRITING: 93,
            TaskType.SCIENTIFIC: 95,
            TaskType.INSTRUCTION_FOLLOWING: 96,
        },
        avg_latency_ms=1000,
        cost_per_1k_tokens=0.006,
        context_window=196000,
        max_output_tokens=32768,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=97,
        creativity_score=93,
        instruction_following=96,
        factual_accuracy=94,
    ),
    # GPT-5 Mini - Fast/affordable GPT-5
    "gpt-5-mini": ModelCapability(
        model_id="gpt-5-mini",
        provider="openai",
        task_scores={
            TaskType.CHAT: 92,
            TaskType.FACTUAL_QA: 90,
            TaskType.SUMMARIZATION: 91,
            TaskType.CODE_GENERATION: 88,
            TaskType.INSTRUCTION_FOLLOWING: 92,
        },
        avg_latency_ms=400,
        cost_per_1k_tokens=0.002,
        context_window=196000,
        max_output_tokens=16384,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=88,
        creativity_score=85,
        instruction_following=92,
        factual_accuracy=88,
    ),
    # GPT-5 Pro - Enhanced reasoning
    "gpt-5-pro": ModelCapability(
        model_id="gpt-5-pro",
        provider="openai",
        task_scores={
            TaskType.LOGICAL_REASONING: 99,
            TaskType.MATHEMATICAL: 99,
            TaskType.SCIENTIFIC: 98,
            TaskType.ANALYTICAL: 99,
            TaskType.RESEARCH: 98,
            TaskType.CODE_GENERATION: 97,
        },
        avg_latency_ms=3000,
        cost_per_1k_tokens=0.015,
        context_window=196000,
        max_output_tokens=32768,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=99,
        creativity_score=90,
        instruction_following=97,
        factual_accuracy=98,
    ),

    # --- O-Series Reasoning Models (Apr 2025) ---
    # o3 - Flagship reasoning model
    "o3": ModelCapability(
        model_id="o3",
        provider="openai",
        task_scores={
            TaskType.CODE_GENERATION: 99,
            TaskType.MATHEMATICAL: 99,
            TaskType.LOGICAL_REASONING: 99,
            TaskType.SCIENTIFIC: 98,
            TaskType.ANALYTICAL: 98,
            TaskType.CODE_DEBUGGING: 98,
        },
        avg_latency_ms=5000,  # Deep thinking takes time
        cost_per_1k_tokens=0.012,
        context_window=200000,
        max_output_tokens=100000,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=99,
        creativity_score=85,
        instruction_following=95,
        factual_accuracy=97,
    ),
    # o3-pro - Most intelligent model (deep thinking)
    "o3-pro": ModelCapability(
        model_id="o3-pro",
        provider="openai",
        task_scores={
            TaskType.MATHEMATICAL: 100,
            TaskType.LOGICAL_REASONING: 100,
            TaskType.SCIENTIFIC: 99,
            TaskType.CODE_GENERATION: 99,
            TaskType.ANALYTICAL: 99,
            TaskType.RESEARCH: 98,
        },
        avg_latency_ms=15000,  # Extended thinking
        cost_per_1k_tokens=0.05,
        context_window=200000,
        max_output_tokens=100000,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=100,
        creativity_score=82,
        instruction_following=94,
        factual_accuracy=99,
    ),
    # o4-mini - Fast cost-efficient reasoning
    "o4-mini": ModelCapability(
        model_id="o4-mini",
        provider="openai",
        task_scores={
            TaskType.MATHEMATICAL: 96,
            TaskType.CODE_GENERATION: 95,
            TaskType.LOGICAL_REASONING: 94,
            TaskType.ANALYTICAL: 92,
            TaskType.INSTRUCTION_FOLLOWING: 90,
        },
        avg_latency_ms=1500,
        cost_per_1k_tokens=0.003,
        context_window=200000,
        max_output_tokens=65536,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=94,
        creativity_score=78,
        instruction_following=90,
        factual_accuracy=92,
    ),

    # --- GPT-4.1 Series (Apr 2025) ---
    # GPT-4.1 - Optimized for coding, 1M context
    "gpt-4.1": ModelCapability(
        model_id="gpt-4.1",
        provider="openai",
        task_scores={
            TaskType.CODE_GENERATION: 95,
            TaskType.MATHEMATICAL: 94,
            TaskType.LOGICAL_REASONING: 93,
            TaskType.FACTUAL_QA: 92,
            TaskType.INSTRUCTION_FOLLOWING: 95,
            TaskType.CREATIVE_WRITING: 88,
            TaskType.ANALYTICAL: 92,
            TaskType.RESEARCH: 90,
        },
        avg_latency_ms=500,
        cost_per_1k_tokens=0.002,
        context_window=1000000,  # 1M context
        max_output_tokens=32768,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=92,
        creativity_score=86,
        instruction_following=95,
        factual_accuracy=90,
    ),
    # GPT-4.1 Mini - Balanced performance
    "gpt-4.1-mini": ModelCapability(
        model_id="gpt-4.1-mini",
        provider="openai",
        task_scores={
            TaskType.CHAT: 88,
            TaskType.FACTUAL_QA: 85,
            TaskType.SUMMARIZATION: 86,
            TaskType.CODE_GENERATION: 84,
            TaskType.INSTRUCTION_FOLLOWING: 88,
        },
        avg_latency_ms=300,
        cost_per_1k_tokens=0.0004,
        context_window=1000000,
        max_output_tokens=16384,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=78,
        creativity_score=74,
        instruction_following=88,
        factual_accuracy=83,
    ),
    # GPT-4.1 Nano - Fastest/cheapest
    "gpt-4.1-nano": ModelCapability(
        model_id="gpt-4.1-nano",
        provider="openai",
        task_scores={
            TaskType.CHAT: 82,
            TaskType.FACTUAL_QA: 78,
            TaskType.SUMMARIZATION: 80,
            TaskType.INSTRUCTION_FOLLOWING: 82,
        },
        avg_latency_ms=150,
        cost_per_1k_tokens=0.0001,
        context_window=1000000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=68,
        creativity_score=65,
        instruction_following=82,
        factual_accuracy=75,
    ),

    # --- Coding Models ---
    # Codex Mini - Optimized for Codex CLI
    "codex-mini-latest": ModelCapability(
        model_id="codex-mini-latest",
        provider="openai",
        task_scores={
            TaskType.CODE_GENERATION: 94,
            TaskType.CODE_REVIEW: 92,
            TaskType.CODE_DEBUGGING: 93,
            TaskType.TECHNICAL_DOCS: 88,
        },
        avg_latency_ms=400,
        cost_per_1k_tokens=0.003,
        context_window=128000,
        max_output_tokens=16384,
        supports_vision=False,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=88,
        creativity_score=70,
        instruction_following=90,
        factual_accuracy=88,
    ),

    # --- Legacy (still available) ---
    # GPT-4o - Previous generation
    "gpt-4o": ModelCapability(
        model_id="gpt-4o",
        provider="openai",
        task_scores={
            TaskType.CODE_GENERATION: 90,
            TaskType.CREATIVE_WRITING: 88,
            TaskType.ANALYTICAL: 88,
            TaskType.CHAT: 90,
            TaskType.INSTRUCTION_FOLLOWING: 90,
        },
        avg_latency_ms=600,
        cost_per_1k_tokens=0.0025,
        context_window=128000,
        max_output_tokens=16384,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=88,
        creativity_score=86,
        instruction_following=90,
        factual_accuracy=87,
    ),
    # GPT-4o Mini - Previous gen mini
    "gpt-4o-mini": ModelCapability(
        model_id="gpt-4o-mini",
        provider="openai",
        task_scores={
            TaskType.CHAT: 85,
            TaskType.FACTUAL_QA: 82,
            TaskType.SUMMARIZATION: 83,
            TaskType.INSTRUCTION_FOLLOWING: 85,
        },
        avg_latency_ms=250,
        cost_per_1k_tokens=0.00015,
        context_window=128000,
        max_output_tokens=16384,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=75,
        creativity_score=72,
        instruction_following=85,
        factual_accuracy=80,
    ),

    # ==================== Google (Gemini 3 Series) ====================
    # Gemini 3 Pro - Most powerful Gemini (Nov 2025)
    "gemini-3-pro-preview": ModelCapability(
        model_id="gemini-3-pro-preview",
        provider="google",
        task_scores={
            TaskType.FACTUAL_QA: 94,
            TaskType.RESEARCH: 95,
            TaskType.SUMMARIZATION: 93,
            TaskType.CODE_GENERATION: 92,
            TaskType.ANALYTICAL: 93,
            TaskType.TRANSLATION: 94,
            TaskType.LOGICAL_REASONING: 92,
        },
        avg_latency_ms=600,
        cost_per_1k_tokens=0.00175,
        context_window=1000000,
        max_output_tokens=64000,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=92,
        creativity_score=86,
        instruction_following=91,
        factual_accuracy=93,
    ),
    # Gemini 3 Flash - Pro-level at Flash pricing (Dec 2025)
    "gemini-3-flash-preview": ModelCapability(
        model_id="gemini-3-flash-preview",
        provider="google",
        task_scores={
            TaskType.CHAT: 90,
            TaskType.SUMMARIZATION: 91,
            TaskType.FACTUAL_QA: 88,
            TaskType.TRANSLATION: 90,
            TaskType.CODE_GENERATION: 86,
            TaskType.RESEARCH: 88,
        },
        avg_latency_ms=250,
        cost_per_1k_tokens=0.0001,
        context_window=1000000,
        max_output_tokens=64000,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=85,
        creativity_score=82,
        instruction_following=88,
        factual_accuracy=86,
    ),
    # Gemini 3 Deep Think - For complex reasoning
    "gemini-3-deep-think-preview": ModelCapability(
        model_id="gemini-3-deep-think-preview",
        provider="google",
        task_scores={
            TaskType.MATHEMATICAL: 97,
            TaskType.LOGICAL_REASONING: 97,
            TaskType.SCIENTIFIC: 96,
            TaskType.ANALYTICAL: 96,
            TaskType.RESEARCH: 95,
        },
        avg_latency_ms=5000,
        cost_per_1k_tokens=0.005,
        context_window=1000000,
        max_output_tokens=64000,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=98,
        creativity_score=78,
        instruction_following=88,
        factual_accuracy=95,
    ),

    # ==================== Mistral (Mistral 3 Series) ====================
    # Mistral Large 3 - Open weight frontier model (Dec 2025)
    "mistral-large-3": ModelCapability(
        model_id="mistral-large-3",
        provider="mistral",
        task_scores={
            TaskType.CODE_GENERATION: 94,
            TaskType.LOGICAL_REASONING: 92,
            TaskType.INSTRUCTION_FOLLOWING: 93,
            TaskType.TRANSLATION: 96,  # Strong multilingual
            TaskType.CHAT: 91,
            TaskType.ANALYTICAL: 90,
        },
        avg_latency_ms=500,
        cost_per_1k_tokens=0.002,
        context_window=256000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=90,
        creativity_score=85,
        instruction_following=93,
        factual_accuracy=88,
    ),
    # Devstral 2 - Coding specialist (Dec 2025)
    "devstral-2": ModelCapability(
        model_id="devstral-2",
        provider="mistral",
        task_scores={
            TaskType.CODE_GENERATION: 97,
            TaskType.CODE_REVIEW: 96,
            TaskType.CODE_DEBUGGING: 97,
            TaskType.TECHNICAL_DOCS: 92,
        },
        avg_latency_ms=600,
        cost_per_1k_tokens=0.003,
        context_window=128000,
        max_output_tokens=8192,
        supports_vision=False,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=88,
        creativity_score=72,
        instruction_following=90,
        factual_accuracy=85,
    ),
    # Ministral 3 14B - Edge/local model
    "ministral-3-14b": ModelCapability(
        model_id="ministral-3-14b",
        provider="mistral",
        task_scores={
            TaskType.CHAT: 82,
            TaskType.CODE_GENERATION: 80,
            TaskType.FACTUAL_QA: 78,
            TaskType.INSTRUCTION_FOLLOWING: 84,
        },
        avg_latency_ms=200,
        cost_per_1k_tokens=0.0003,
        context_window=128000,
        max_output_tokens=4096,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=72,
        creativity_score=70,
        instruction_following=84,
        factual_accuracy=76,
    ),

    # ==================== Groq (Llama 4 & Fast Inference) ====================
    # Llama 4 Maverick via Groq (Apr 2025)
    "meta-llama/llama-4-maverick-17b-128e-instruct": ModelCapability(
        model_id="meta-llama/llama-4-maverick-17b-128e-instruct",
        provider="groq",
        task_scores={
            TaskType.CHAT: 90,
            TaskType.CODE_GENERATION: 88,
            TaskType.FACTUAL_QA: 86,
            TaskType.INSTRUCTION_FOLLOWING: 90,
            TaskType.CREATIVE_WRITING: 84,
        },
        avg_latency_ms=100,  # Groq is FAST
        cost_per_1k_tokens=0.0005,
        context_window=128000,
        max_output_tokens=8192,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=84,
        creativity_score=82,
        instruction_following=90,
        factual_accuracy=84,
    ),
    # Llama 3.3 70B Versatile on Groq
    "llama-3.3-70b-versatile": ModelCapability(
        model_id="llama-3.3-70b-versatile",
        provider="groq",
        task_scores={
            TaskType.CHAT: 86,
            TaskType.CODE_GENERATION: 84,
            TaskType.FACTUAL_QA: 82,
            TaskType.INSTRUCTION_FOLLOWING: 86,
        },
        avg_latency_ms=120,
        cost_per_1k_tokens=0.0007,
        context_window=128000,
        max_output_tokens=8000,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=80,
        creativity_score=78,
        instruction_following=86,
        factual_accuracy=80,
    ),

    # ==================== Cohere (Command A Series) ====================
    # Command A - Most performant Command model (Mar 2025)
    "command-a-03-2025": ModelCapability(
        model_id="command-a-03-2025",
        provider="cohere",
        task_scores={
            TaskType.RESEARCH: 92,
            TaskType.SUMMARIZATION: 93,
            TaskType.FACTUAL_QA: 90,
            TaskType.CHAT: 88,
            TaskType.TRANSLATION: 90,
            TaskType.ANALYTICAL: 88,
        },
        avg_latency_ms=500,
        cost_per_1k_tokens=0.0025,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=85,
        creativity_score=82,
        instruction_following=88,
        factual_accuracy=88,
    ),
    # Command A Reasoning - For complex agentic tasks
    "command-a-reasoning": ModelCapability(
        model_id="command-a-reasoning",
        provider="cohere",
        task_scores={
            TaskType.LOGICAL_REASONING: 94,
            TaskType.ANALYTICAL: 93,
            TaskType.RESEARCH: 92,
            TaskType.INSTRUCTION_FOLLOWING: 92,
        },
        avg_latency_ms=1500,
        cost_per_1k_tokens=0.004,
        context_window=256000,
        max_output_tokens=8192,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=94,
        creativity_score=78,
        instruction_following=92,
        factual_accuracy=90,
    ),
    # Command A Translate - Machine translation specialist
    "command-a-translate-08-2025": ModelCapability(
        model_id="command-a-translate-08-2025",
        provider="cohere",
        task_scores={
            TaskType.TRANSLATION: 98,
        },
        avg_latency_ms=300,
        cost_per_1k_tokens=0.001,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=False,
        supports_json_mode=False,
        reasoning_depth=60,
        creativity_score=50,
        instruction_following=85,
        factual_accuracy=95,
    ),

    # ==================== Ollama (Local Models - Free!) ====================
    # Llama 3.2 - Latest Meta model via Ollama
    "llama3.2": ModelCapability(
        model_id="llama3.2",
        provider="ollama",
        task_scores={
            TaskType.CODE_GENERATION: 78,
            TaskType.CODE_REVIEW: 75,
            TaskType.CODE_DEBUGGING: 73,
            TaskType.FACTUAL_QA: 75,
            TaskType.CREATIVE_WRITING: 72,
            TaskType.LOGICAL_REASONING: 74,
            TaskType.CHAT: 80,
            TaskType.SUMMARIZATION: 76,
            TaskType.INSTRUCTION_FOLLOWING: 78,
        },
        avg_latency_ms=2000,
        cost_per_1k_tokens=0.0,  # Free!
        context_window=128000,
        max_output_tokens=4096,
        supports_vision=False,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=72,
        creativity_score=70,
        instruction_following=78,
        factual_accuracy=73,
    ),
    # CodeLlama - Coding specialist
    "codellama": ModelCapability(
        model_id="codellama",
        provider="ollama",
        task_scores={
            TaskType.CODE_GENERATION: 82,
            TaskType.CODE_REVIEW: 78,
            TaskType.CODE_DEBUGGING: 80,
            TaskType.TECHNICAL_DOCS: 70,
        },
        avg_latency_ms=2000,
        cost_per_1k_tokens=0.0,
        context_window=16384,
        max_output_tokens=4096,
        supports_function_calling=False,
        supports_json_mode=False,
        reasoning_depth=70,
        creativity_score=55,
        instruction_following=75,
        factual_accuracy=72,
    ),
    # Mistral via Ollama - Fast general purpose
    "mistral": ModelCapability(
        model_id="mistral",
        provider="ollama",
        task_scores={
            TaskType.FACTUAL_QA: 74,
            TaskType.LOGICAL_REASONING: 72,
            TaskType.CREATIVE_WRITING: 70,
            TaskType.CHAT: 78,
            TaskType.SUMMARIZATION: 75,
            TaskType.INSTRUCTION_FOLLOWING: 76,
        },
        avg_latency_ms=1500,
        cost_per_1k_tokens=0.0,
        context_window=32768,
        max_output_tokens=4096,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=70,
        creativity_score=68,
        instruction_following=76,
        factual_accuracy=71,
    ),
    # Phi-3 - Microsoft's small but capable model
    "phi3": ModelCapability(
        model_id="phi3",
        provider="ollama",
        task_scores={
            TaskType.CODE_GENERATION: 75,
            TaskType.LOGICAL_REASONING: 73,
            TaskType.MATHEMATICAL: 76,
            TaskType.FACTUAL_QA: 70,
            TaskType.INSTRUCTION_FOLLOWING: 74,
        },
        avg_latency_ms=1000,
        cost_per_1k_tokens=0.0,
        context_window=4096,
        max_output_tokens=2048,
        supports_function_calling=False,
        supports_json_mode=True,
        reasoning_depth=68,
        creativity_score=60,
        instruction_following=74,
        factual_accuracy=69,
    ),
    # Gemma 2 - Google's open model
    "gemma2": ModelCapability(
        model_id="gemma2",
        provider="ollama",
        task_scores={
            TaskType.CHAT: 76,
            TaskType.FACTUAL_QA: 73,
            TaskType.CREATIVE_WRITING: 71,
            TaskType.SUMMARIZATION: 74,
            TaskType.INSTRUCTION_FOLLOWING: 75,
        },
        avg_latency_ms=1200,
        cost_per_1k_tokens=0.0,
        context_window=8192,
        max_output_tokens=4096,
        supports_function_calling=False,
        supports_json_mode=True,
        reasoning_depth=68,
        creativity_score=67,
        instruction_following=75,
        factual_accuracy=70,
    ),
    # DeepSeek Coder - Coding specialist
    "deepseek-coder": ModelCapability(
        model_id="deepseek-coder",
        provider="ollama",
        task_scores={
            TaskType.CODE_GENERATION: 85,
            TaskType.CODE_REVIEW: 80,
            TaskType.CODE_DEBUGGING: 82,
            TaskType.TECHNICAL_DOCS: 72,
        },
        avg_latency_ms=2500,
        cost_per_1k_tokens=0.0,
        context_window=16384,
        max_output_tokens=4096,
        supports_function_calling=False,
        supports_json_mode=True,
        reasoning_depth=75,
        creativity_score=55,
        instruction_following=78,
        factual_accuracy=74,
    ),
    # Qwen 2.5 - Alibaba's multilingual model
    "qwen2.5": ModelCapability(
        model_id="qwen2.5",
        provider="ollama",
        task_scores={
            TaskType.CODE_GENERATION: 80,
            TaskType.TRANSLATION: 82,
            TaskType.FACTUAL_QA: 76,
            TaskType.CHAT: 78,
            TaskType.INSTRUCTION_FOLLOWING: 79,
        },
        avg_latency_ms=1800,
        cost_per_1k_tokens=0.0,
        context_window=32768,
        max_output_tokens=4096,
        supports_function_calling=True,
        supports_json_mode=True,
        reasoning_depth=74,
        creativity_score=70,
        instruction_following=79,
        factual_accuracy=73,
    ),

    # ==================== Perplexity (RAG-native) ====================
    # Sonar - Lightweight search model
    "perplexity/sonar": ModelCapability(
        model_id="perplexity/sonar",
        provider="perplexity",
        task_scores={
            TaskType.FACTUAL_QA: 90,
            TaskType.RESEARCH: 92,
            TaskType.SUMMARIZATION: 85,
            TaskType.CHAT: 80,
        },
        avg_latency_ms=800,
        cost_per_1k_tokens=0.001,
        context_window=127000,
        max_output_tokens=4096,
        supports_function_calling=False,
        supports_json_mode=False,
        reasoning_depth=75,
        creativity_score=60,
        instruction_following=80,
        factual_accuracy=92,
    ),
    # Sonar Pro - Advanced search with citations
    "perplexity/sonar-pro": ModelCapability(
        model_id="perplexity/sonar-pro",
        provider="perplexity",
        task_scores={
            TaskType.FACTUAL_QA: 95,
            TaskType.RESEARCH: 97,
            TaskType.SUMMARIZATION: 90,
            TaskType.ANALYTICAL: 88,
            TaskType.CHAT: 85,
        },
        avg_latency_ms=1500,
        cost_per_1k_tokens=0.005,
        context_window=200000,
        max_output_tokens=8192,
        supports_function_calling=False,
        supports_json_mode=False,
        reasoning_depth=85,
        creativity_score=65,
        instruction_following=85,
        factual_accuracy=96,
    ),
    # Sonar Deep Research - Exhaustive multi-step research
    "perplexity/sonar-deep-research": ModelCapability(
        model_id="perplexity/sonar-deep-research",
        provider="perplexity",
        task_scores={
            TaskType.RESEARCH: 99,
            TaskType.FACTUAL_QA: 97,
            TaskType.ANALYTICAL: 95,
            TaskType.SUMMARIZATION: 92,
        },
        avg_latency_ms=5000,
        cost_per_1k_tokens=0.02,
        context_window=200000,
        max_output_tokens=16384,
        supports_function_calling=False,
        supports_json_mode=False,
        reasoning_depth=95,
        creativity_score=60,
        instruction_following=88,
        factual_accuracy=98,
    ),
    # Sonar Reasoning Pro - Premier reasoning with search
    "perplexity/sonar-reasoning-pro": ModelCapability(
        model_id="perplexity/sonar-reasoning-pro",
        provider="perplexity",
        task_scores={
            TaskType.LOGICAL_REASONING: 94,
            TaskType.RESEARCH: 95,
            TaskType.FACTUAL_QA: 93,
            TaskType.ANALYTICAL: 94,
            TaskType.SCIENTIFIC: 90,
        },
        avg_latency_ms=3000,
        cost_per_1k_tokens=0.01,
        context_window=200000,
        max_output_tokens=16384,
        supports_function_calling=False,
        supports_json_mode=False,
        reasoning_depth=94,
        creativity_score=65,
        instruction_following=88,
        factual_accuracy=95,
    ),
}


class TaskClassifier:
    """Classifies queries into task types."""

    # Keywords and patterns for each task type
    TASK_PATTERNS: Dict[TaskType, List[str]] = {
        TaskType.CODE_GENERATION: [
            "write code", "implement", "create a function", "build a",
            "code for", "script that", "program to", "def ", "function",
            "class ", "```", "coding", "developer", "software"
        ],
        TaskType.CODE_REVIEW: [
            "review this code", "code review", "what's wrong with",
            "improve this code", "refactor", "optimize this"
        ],
        TaskType.CODE_DEBUGGING: [
            "debug", "fix this error", "why doesn't", "bug", "issue with",
            "not working", "exception", "error:", "traceback"
        ],
        TaskType.MATHEMATICAL: [
            "calculate", "solve", "equation", "integral", "derivative",
            "math", "algebra", "geometry", "proof", "theorem", "∫", "∑",
            "probability", "statistics", "compute"
        ],
        TaskType.LOGICAL_REASONING: [
            "logic", "deduce", "infer", "therefore", "conclude",
            "reasoning", "argument", "premise", "if then", "implies"
        ],
        TaskType.CREATIVE_WRITING: [
            "write a story", "poem", "creative", "fiction", "narrative",
            "novel", "character", "plot", "write me a", "compose", "haiku",
            "sonnet", "limerick", "lyrics", "song", "prose", "essay"
        ],
        TaskType.COPYWRITING: [
            "marketing", "ad copy", "slogan", "tagline", "headline",
            "sales", "persuade", "landing page", "email campaign"
        ],
        TaskType.BRAINSTORMING: [
            "brainstorm", "ideas for", "suggestions", "what are some",
            "help me think", "creative ideas", "possibilities"
        ],
        TaskType.FACTUAL_QA: [
            "what is", "who is", "when did", "where is", "how many",
            "define", "explain", "tell me about", "fact"
        ],
        TaskType.RESEARCH: [
            "research", "analyze", "compare", "investigate", "study",
            "deep dive", "comprehensive", "detailed analysis"
        ],
        TaskType.SUMMARIZATION: [
            "summarize", "summary", "tldr", "brief", "key points",
            "main ideas", "condense", "shorten"
        ],
        TaskType.TRANSLATION: [
            "translate", "translation", "in french", "in spanish",
            "in german", "in chinese", "in japanese", "to english"
        ],
        TaskType.CHAT: [
            "hello", "hi", "hey", "how are you", "thanks", "okay",
            "got it", "sure", "yes", "no", "maybe"
        ],
        TaskType.ROLEPLAY: [
            "pretend", "roleplay", "act as", "you are a", "imagine you",
            "play the role", "character"
        ],
        TaskType.TECHNICAL_DOCS: [
            "documentation", "readme", "api docs", "technical writing",
            "specification", "architecture"
        ],
        TaskType.SCIENTIFIC: [
            "scientific", "hypothesis", "experiment", "physics", "chemistry",
            "biology", "research paper", "methodology"
        ],
        TaskType.ANALYTICAL: [
            "analyze", "analysis", "evaluate", "assess", "examine",
            "break down", "interpret", "insights"
        ],
        TaskType.LEGAL: [
            "legal", "law", "contract", "liability", "court", "attorney",
            "compliance", "regulation", "statute"
        ],
        TaskType.MEDICAL: [
            "medical", "health", "symptom", "diagnosis", "treatment",
            "medicine", "clinical", "patient"
        ],
        TaskType.FINANCIAL: [
            "financial", "investment", "stock", "portfolio", "accounting",
            "budget", "revenue", "profit", "trading"
        ],
    }

    def __init__(self, registry: Optional[ProviderRegistry] = None):
        self.registry = registry

    def classify(self, query: str) -> Tuple[TaskType, float]:
        """
        Classify a query into a task type.

        Returns (task_type, confidence) tuple.
        """
        query_lower = query.lower()

        # Score each task type
        scores: Dict[TaskType, int] = {}
        for task_type, patterns in self.TASK_PATTERNS.items():
            score = sum(1 for p in patterns if p in query_lower)
            if score > 0:
                scores[task_type] = score

        if not scores:
            return TaskType.UNKNOWN, 0.3

        # Get the highest scoring task
        best_task = max(scores, key=scores.get)
        best_score = scores[best_task]

        # Calculate confidence based on score differential
        total_score = sum(scores.values())
        confidence = min(0.95, 0.5 + (best_score / total_score) * 0.5)

        return best_task, confidence

    async def classify_with_ai(
        self,
        query: str,
        classifier_model: Optional[str] = None
    ) -> Tuple[TaskType, float]:
        # Use config default if not specified
        if classifier_model is None:
            classifier_model = get_classifier_model()
        """Use an AI model to classify the task (more accurate but slower)."""
        if not self.registry:
            return self.classify(query)

        provider = self.registry.get_provider_for_model(classifier_model)
        if not provider:
            return self.classify(query)

        task_list = ", ".join([t.value for t in TaskType])

        classification_prompt = f"""Classify this query into exactly one task type.

Task types: {task_list}

Query: {query}

Respond with ONLY a JSON object:
{{"task_type": "<type>", "confidence": <0.0-1.0>}}"""

        try:
            response = await provider.query(classification_prompt, model=classifier_model)
            # Parse the JSON response
            import re
            json_match = re.search(r'\{[^}]+\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
                task_type = TaskType(result["task_type"])
                confidence = float(result["confidence"])
                return task_type, confidence
        except Exception:
            pass

        return self.classify(query)


class ModelRouter:
    """
    Intelligent router that selects optimal model(s) for each query.
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        capabilities: Optional[Dict[str, ModelCapability]] = None,
        performance_log_path: Optional[str] = None
    ):
        self.registry = registry
        self.capabilities = capabilities or DEFAULT_MODEL_CAPABILITIES
        self.classifier = TaskClassifier(registry)
        self.performance_log_path = performance_log_path
        self.performance_history: List[Dict[str, Any]] = []

        # Load performance history if available
        if performance_log_path and os.path.exists(performance_log_path):
            try:
                with open(performance_log_path, 'r') as f:
                    self.performance_history = json.load(f)
            except Exception:
                pass

    def get_available_models(self) -> List[str]:
        """Get list of models that are both configured and have capability profiles."""
        configured_providers = self.registry.list_configured_providers()
        available = []

        for model_id, cap in self.capabilities.items():
            if cap.provider in configured_providers:
                available.append(model_id)

        return available

    def score_model(
        self,
        model_id: str,
        task_type: TaskType,
        strategy: RoutingStrategy
    ) -> float:
        """Score a model for a given task and strategy."""
        if model_id not in self.capabilities:
            return 0.0

        cap = self.capabilities[model_id]

        # Get task-specific score (default to 50 if not specified)
        task_score = cap.task_scores.get(task_type.value, 50.0)

        # Weight factors based on strategy
        if strategy == RoutingStrategy.BEST_QUALITY:
            # Heavily weight task score and reasoning depth
            return (
                task_score * 0.6 +
                cap.reasoning_depth * 0.2 +
                cap.factual_accuracy * 0.2
            )

        elif strategy == RoutingStrategy.BEST_SPEED:
            # Weight latency heavily (inverse - lower is better)
            latency_score = max(0, 100 - (cap.avg_latency_ms / 100))
            return (
                latency_score * 0.5 +
                task_score * 0.3 +
                (100 - cap.error_rate * 1000) * 0.2
            )

        elif strategy == RoutingStrategy.BEST_VALUE:
            # Balance quality vs cost
            cost_score = max(0, 100 - (cap.cost_per_1k_tokens * 5000))
            return (
                task_score * 0.4 +
                cost_score * 0.4 +
                (100 - cap.avg_latency_ms / 50) * 0.2
            )

        else:  # BALANCED
            latency_score = max(0, 100 - (cap.avg_latency_ms / 100))
            cost_score = max(0, 100 - (cap.cost_per_1k_tokens * 5000))
            return (
                task_score * 0.4 +
                latency_score * 0.2 +
                cost_score * 0.2 +
                cap.reasoning_depth * 0.1 +
                cap.instruction_following * 0.1
            )

    def select_models(
        self,
        task_type: TaskType,
        strategy: RoutingStrategy,
        num_models: int = 1,
        exclude_models: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Select the best model(s) for a task.

        Returns list of (model_id, score) tuples.
        """
        available = self.get_available_models()
        exclude = set(exclude_models or [])

        # Score all available models
        scored = []
        for model_id in available:
            if model_id not in exclude:
                score = self.score_model(model_id, task_type, strategy)
                scored.append((model_id, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:num_models]

    async def route(
        self,
        query: str,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        force_task: Optional[TaskType] = None,
        force_models: Optional[List[str]] = None,
        num_models: int = 1,
        use_ai_classifier: bool = False
    ) -> RoutingDecision:
        """
        Determine the optimal routing for a query.
        """
        # Classify the task
        if force_task:
            task_type, task_conf = force_task, 1.0
        elif use_ai_classifier:
            task_type, task_conf = await self.classifier.classify_with_ai(query)
        else:
            task_type, task_conf = self.classifier.classify(query)

        # Select models
        if force_models:
            selected = [(m, 100.0) for m in force_models]
        elif strategy == RoutingStrategy.ENSEMBLE:
            # For ensemble, select top 3 diverse models
            selected = self.select_models(task_type, RoutingStrategy.BEST_QUALITY, 3)
        elif strategy == RoutingStrategy.FALLBACK_CHAIN:
            # Select models in order of preference
            selected = self.select_models(task_type, RoutingStrategy.BALANCED, 3)
        else:
            selected = self.select_models(task_type, strategy, num_models)

        if not selected:
            raise ValueError("No models available for routing")

        selected_models = [m for m, _ in selected]

        # Estimate cost and latency
        est_cost = None
        est_latency = None
        if selected_models[0] in self.capabilities:
            cap = self.capabilities[selected_models[0]]
            est_latency = cap.avg_latency_ms
            # Rough estimate: assume 500 tokens
            est_cost = cap.cost_per_1k_tokens * 0.5

        # Get alternatives
        all_scored = self.select_models(task_type, strategy, 5)
        alternatives = [
            {"model": m, "score": s}
            for m, s in all_scored
            if m not in selected_models
        ]

        return RoutingDecision(
            query=query,
            detected_task=task_type,
            task_confidence=task_conf,
            selected_models=selected_models,
            strategy_used=strategy,
            reasoning=f"Task classified as {task_type.value} (confidence: {task_conf:.2f}). "
                     f"Selected {selected_models[0]} based on {strategy.value} strategy.",
            estimated_cost=est_cost,
            estimated_latency_ms=est_latency,
            alternatives=alternatives
        )

    async def route_and_execute(
        self,
        query: str,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        **kwargs
    ) -> RoutingResult:
        """
        Route a query and execute it on the selected model(s).
        """
        import time
        start_time = time.time()

        # Get routing decision
        decision = await self.route(query, strategy, **kwargs)

        responses = []

        if strategy == RoutingStrategy.FALLBACK_CHAIN:
            # Try models in sequence until one succeeds
            for model_id in decision.selected_models:
                try:
                    provider = self.registry.get_provider_for_model(model_id)
                    if provider:
                        response = await provider.query(query, model=model_id)
                        responses.append(response)
                        break  # Success, stop trying
                except Exception as e:
                    continue  # Try next model

        elif strategy == RoutingStrategy.ENSEMBLE:
            # Query all selected models in parallel
            tasks = []
            for model_id in decision.selected_models:
                provider = self.registry.get_provider_for_model(model_id)
                if provider:
                    tasks.append(provider.query(query, model=model_id))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                responses = [r for r in results if isinstance(r, Response)]

        else:
            # Single model query
            model_id = decision.selected_models[0]
            provider = self.registry.get_provider_for_model(model_id)
            if provider:
                response = await provider.query(query, model=model_id)
                responses.append(response)

        elapsed_ms = (time.time() - start_time) * 1000

        # Synthesize if ensemble
        synthesized = None
        if strategy == RoutingStrategy.ENSEMBLE and len(responses) > 1:
            synthesized = self._synthesize_responses(responses)

        # Calculate actual cost
        actual_cost = sum(
            (r.input_tokens or 0) + (r.output_tokens or 0)
            for r in responses
        ) * 0.00001  # Rough estimate

        result = RoutingResult(
            decision=decision,
            responses=responses,
            synthesized_response=synthesized,
            actual_latency_ms=elapsed_ms,
            actual_cost=actual_cost
        )

        # Log performance
        self._log_performance(result)

        return result

    def _synthesize_responses(self, responses: List[Response]) -> str:
        """Synthesize multiple responses into one."""
        if not responses:
            return ""

        if len(responses) == 1:
            return responses[0].content

        # Simple synthesis: find common themes
        contents = [r.content for r in responses]
        models = [r.model for r in responses]

        synthesis = f"**Synthesized from {len(responses)} models** ({', '.join(models)}):\n\n"
        synthesis += contents[0]  # Use first response as base

        # Note any divergences
        if len(set(len(c) for c in contents)) > 1:
            synthesis += "\n\n*Note: Response lengths varied significantly between models.*"

        return synthesis

    def _log_performance(self, result: RoutingResult):
        """Log performance for future optimization."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "task_type": result.decision.detected_task.value,
            "strategy": result.decision.strategy_used.value,
            "models_used": result.decision.selected_models,
            "latency_ms": result.actual_latency_ms,
            "cost": result.actual_cost,
            "num_responses": len(result.responses),
        }

        self.performance_history.append(log_entry)

        # Save to file if configured
        if self.performance_log_path:
            try:
                with open(self.performance_log_path, 'w') as f:
                    json.dump(self.performance_history[-1000:], f)  # Keep last 1000
            except Exception:
                pass

    def get_model_recommendation(self, task_type: TaskType) -> Dict[str, Any]:
        """Get a recommendation for which model to use for a task type."""
        available = self.get_available_models()

        if not available:
            return {"error": "No models available"}

        recommendations = {
            "task_type": task_type.value,
            "best_quality": None,
            "best_speed": None,
            "best_value": None,
            "balanced": None,
        }

        for strategy in [RoutingStrategy.BEST_QUALITY, RoutingStrategy.BEST_SPEED,
                        RoutingStrategy.BEST_VALUE, RoutingStrategy.BALANCED]:
            selected = self.select_models(task_type, strategy, 1)
            if selected:
                model_id, score = selected[0]
                cap = self.capabilities.get(model_id)
                recommendations[strategy.value] = {
                    "model": model_id,
                    "score": round(score, 1),
                    "latency_ms": cap.avg_latency_ms if cap else None,
                    "cost_per_1k": cap.cost_per_1k_tokens if cap else None,
                }

        return recommendations

    def update_capability(
        self,
        model_id: str,
        task_type: TaskType,
        new_score: float,
        blend_factor: float = 0.3
    ):
        """
        Update a model's capability score based on observed performance.

        Uses exponential moving average to blend new observations.
        """
        if model_id not in self.capabilities:
            return

        cap = self.capabilities[model_id]
        old_score = cap.task_scores.get(task_type.value, 50.0)

        # Blend old and new scores
        cap.task_scores[task_type.value] = (
            old_score * (1 - blend_factor) + new_score * blend_factor
        )
