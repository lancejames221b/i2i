"""
LangChain integration for i2i consensus verification.

Provides I2IConsensusLLM, a wrapper that adds multi-model consensus
verification to any LangChain-compatible LLM. Leverages task-aware
consensus recommendations to skip consensus when inappropriate
(e.g., math problems, creative writing).

Usage:
    from langchain_openai import ChatOpenAI
    from i2i.integrations.langchain import I2IConsensusLLM

    # Wrap any LangChain LLM
    base_llm = ChatOpenAI(model="gpt-4")
    consensus_llm = I2IConsensusLLM(
        base_llm=base_llm,
        consensus_models=["claude-3-sonnet", "gemini-pro"],
        min_confidence=0.75,
    )

    # Use like any LLM - consensus happens transparently
    response = await consensus_llm.ainvoke("What causes inflation?")

    # Confidence metadata available on the response
    print(response.response_metadata.get("confidence_calibration"))
"""

import asyncio
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import Field

# LangChain imports - these are optional dependencies
try:
    from langchain_core.language_models import BaseLLM
    from langchain_core.language_models.base import LanguageModelInput
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.outputs import Generation, LLMResult
    from langchain_core.messages import BaseMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create stub classes for type hints when LangChain not installed
    class BaseLLM:
        pass

from ..protocol import AICP
from ..schema import ConsensusLevel, ConsensusResult
from ..task_classifier import (
    recommend_consensus,
    ConsensusRecommendation,
    ConsensusTaskCategory,
    get_confidence_calibration,
)


class LowConfidenceError(Exception):
    """
    Raised when consensus confidence falls below the minimum threshold.

    Contains the full ConsensusResult for inspection and recovery.

    Attributes:
        result: The ConsensusResult that triggered the error
        confidence: The confidence_calibration value that was too low
        threshold: The min_confidence threshold that was not met
        message: Human-readable error message
    """

    def __init__(
        self,
        result: ConsensusResult,
        threshold: float,
        message: Optional[str] = None,
    ):
        self.result = result
        self.confidence = result.confidence_calibration or 0.0
        self.threshold = threshold

        if message is None:
            message = (
                f"Consensus confidence ({self.confidence:.2f}) below minimum "
                f"threshold ({threshold:.2f}). Consensus level: {result.consensus_level.value}. "
                f"Models queried: {', '.join(result.models_queried)}"
            )

        super().__init__(message)


class ConsensusMode(str, Enum):
    """How to handle consensus checking."""
    AUTO = "auto"          # Use task-aware recommendations (default)
    ALWAYS = "always"      # Always run consensus regardless of task type
    NEVER = "never"        # Never run consensus (passthrough to base LLM)
    FORCE = "force"        # Force consensus even for inappropriate tasks


def _check_langchain_available():
    """Raise ImportError if LangChain is not installed."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for I2IConsensusLLM. "
            "Install with: pip install langchain-core"
        )


class I2IConsensusLLM(BaseLLM if LANGCHAIN_AVAILABLE else object):
    """
    LangChain LLM wrapper that adds i2i multi-model consensus verification.

    Wraps any BaseLLM and transparently adds consensus verification when
    appropriate. Uses task-aware recommendations to skip consensus for
    tasks where it hurts (math, creative writing) and applies it for
    tasks where it helps (factual, verification, commonsense).

    Attributes:
        base_llm: The underlying LangChain LLM to wrap
        consensus_models: List of model identifiers for consensus queries
        min_confidence: Block responses below this calibrated confidence (0-1)
        warn_confidence: Log warnings below this confidence (0-1)
        consensus_mode: How to handle consensus (AUTO, ALWAYS, NEVER, FORCE)
        protocol: The AICP protocol instance (created automatically if None)

    Raises:
        LowConfidenceError: When consensus confidence < min_confidence
        ImportError: If LangChain is not installed

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from i2i.integrations.langchain import I2IConsensusLLM
        >>>
        >>> base_llm = ChatOpenAI(model="gpt-4")
        >>> llm = I2IConsensusLLM(
        ...     base_llm=base_llm,
        ...     consensus_models=["claude-3-sonnet", "gemini-pro"],
        ...     min_confidence=0.75,
        ...     warn_confidence=0.90,
        ... )
        >>>
        >>> # Factual question - consensus applied
        >>> result = await llm.ainvoke("What is the capital of France?")
        >>> print(result.response_metadata["consensus_level"])  # "high"
        >>>
        >>> # Math question - consensus skipped (task-aware)
        >>> result = await llm.ainvoke("Calculate 15 * 23")
        >>> print(result.response_metadata.get("consensus_skipped"))  # True
    """

    # Pydantic model config
    base_llm: Any = Field(description="The underlying LangChain LLM to wrap")
    consensus_models: List[str] = Field(
        default_factory=list,
        description="Model identifiers for consensus queries"
    )
    min_confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum calibrated confidence to accept (blocks below)"
    )
    warn_confidence: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for warnings"
    )
    consensus_mode: ConsensusMode = Field(
        default=ConsensusMode.AUTO,
        description="How to handle consensus checking"
    )
    protocol: Any = Field(
        default=None,
        description="AICP protocol instance (created if None)"
    )

    # Internal state
    _initialized: bool = False

    def __init__(self, **kwargs):
        _check_langchain_available()
        super().__init__(**kwargs)

        # Initialize protocol if not provided
        if self.protocol is None:
            self.protocol = AICP()

        self._initialized = True

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        base_type = getattr(self.base_llm, "_llm_type", "unknown")
        return f"i2i-consensus-{base_type}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters for this LLM."""
        base_params = getattr(self.base_llm, "_identifying_params", {})
        return {
            **base_params,
            "i2i_consensus_models": self.consensus_models,
            "i2i_min_confidence": self.min_confidence,
            "i2i_warn_confidence": self.warn_confidence,
            "i2i_consensus_mode": self.consensus_mode.value,
        }

    def _should_use_consensus(self, prompt: str) -> tuple[bool, Optional[ConsensusRecommendation]]:
        """
        Determine if consensus should be used for this prompt.

        Returns:
            Tuple of (should_use_consensus, recommendation)
        """
        if self.consensus_mode == ConsensusMode.NEVER:
            return False, None

        if self.consensus_mode == ConsensusMode.ALWAYS:
            return True, None

        if self.consensus_mode == ConsensusMode.FORCE:
            rec = recommend_consensus(prompt, force=True)
            return True, rec

        # AUTO mode - use task-aware recommendations
        rec = recommend_consensus(prompt)
        return rec.should_use_consensus, rec

    def _build_response_metadata(
        self,
        consensus_result: Optional[ConsensusResult],
        recommendation: Optional[ConsensusRecommendation],
        consensus_skipped: bool = False,
    ) -> Dict[str, Any]:
        """Build metadata dict for the response."""
        metadata = {}

        if consensus_skipped:
            metadata["consensus_skipped"] = True
            if recommendation:
                metadata["skip_reason"] = recommendation.reason
                metadata["task_category"] = recommendation.task_category.value
                metadata["suggested_approach"] = recommendation.suggested_approach
            return metadata

        if consensus_result:
            metadata["consensus_level"] = consensus_result.consensus_level.value
            metadata["confidence_calibration"] = consensus_result.confidence_calibration
            metadata["task_category"] = consensus_result.task_category
            metadata["consensus_appropriate"] = consensus_result.consensus_appropriate
            metadata["models_queried"] = consensus_result.models_queried

            if consensus_result.divergences:
                metadata["divergence_count"] = len(consensus_result.divergences)

            if recommendation:
                metadata["task_recommendation"] = {
                    "should_use_consensus": recommendation.should_use_consensus,
                    "confidence": recommendation.confidence,
                    "reason": recommendation.reason,
                }

        return metadata

    async def _run_consensus(self, prompt: str) -> ConsensusResult:
        """Run consensus query through the protocol."""
        models = self.consensus_models if self.consensus_models else None
        return await self.protocol.consensus_query(
            query=prompt,
            models=models,
            task_aware=True,
        )

    def _check_confidence(self, result: ConsensusResult) -> None:
        """Check confidence against thresholds, raise if too low."""
        confidence = result.confidence_calibration or 0.0

        if confidence < self.min_confidence:
            raise LowConfidenceError(result, self.min_confidence)

        if confidence < self.warn_confidence:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Consensus confidence ({confidence:.2f}) below warning "
                f"threshold ({self.warn_confidence:.2f}). "
                f"Level: {result.consensus_level.value}"
            )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous call - runs async consensus in event loop.

        For better performance, use ainvoke() directly.
        """
        _check_langchain_available()

        # Check if consensus should be applied
        should_use, recommendation = self._should_use_consensus(prompt)

        if not should_use:
            # Skip consensus - passthrough to base LLM
            return self.base_llm._call(prompt, stop=stop, run_manager=run_manager, **kwargs)

        # Run async consensus in event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, use create_task
            import nest_asyncio
            nest_asyncio.apply()
            result = asyncio.run(self._run_consensus(prompt))
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            result = asyncio.run(self._run_consensus(prompt))

        # Check confidence thresholds
        self._check_confidence(result)

        return result.consensus_answer or ""

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async call with consensus verification.

        This is the preferred method for I2IConsensusLLM as consensus
        queries benefit greatly from async execution.
        """
        _check_langchain_available()

        # Check if consensus should be applied
        should_use, recommendation = self._should_use_consensus(prompt)

        if not should_use:
            # Skip consensus - passthrough to base LLM
            if hasattr(self.base_llm, "_acall"):
                return await self.base_llm._acall(prompt, stop=stop, run_manager=run_manager, **kwargs)
            else:
                # Fallback to sync call if no async available
                return self.base_llm._call(prompt, stop=stop, run_manager=run_manager, **kwargs)

        # Run consensus query
        result = await self._run_consensus(prompt)

        # Check confidence thresholds
        self._check_confidence(result)

        return result.consensus_answer or ""

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for multiple prompts."""
        _check_langchain_available()

        generations = []
        for prompt in prompts:
            should_use, recommendation = self._should_use_consensus(prompt)

            if not should_use:
                # Skip consensus - passthrough
                text = self.base_llm._call(prompt, stop=stop, **kwargs)
                metadata = self._build_response_metadata(None, recommendation, consensus_skipped=True)
                generations.append([Generation(text=text, generation_info=metadata)])
            else:
                # Run consensus
                try:
                    loop = asyncio.get_running_loop()
                    import nest_asyncio
                    nest_asyncio.apply()
                    result = asyncio.run(self._run_consensus(prompt))
                except RuntimeError:
                    result = asyncio.run(self._run_consensus(prompt))

                self._check_confidence(result)

                metadata = self._build_response_metadata(result, recommendation)
                generations.append([Generation(
                    text=result.consensus_answer or "",
                    generation_info=metadata,
                )])

        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async generate responses for multiple prompts."""
        _check_langchain_available()

        generations = []

        # Process prompts - could be parallelized but keeping simple for now
        for prompt in prompts:
            should_use, recommendation = self._should_use_consensus(prompt)

            if not should_use:
                # Skip consensus - passthrough
                if hasattr(self.base_llm, "_acall"):
                    text = await self.base_llm._acall(prompt, stop=stop, **kwargs)
                else:
                    text = self.base_llm._call(prompt, stop=stop, **kwargs)
                metadata = self._build_response_metadata(None, recommendation, consensus_skipped=True)
                generations.append([Generation(text=text, generation_info=metadata)])
            else:
                # Run consensus
                result = await self._run_consensus(prompt)
                self._check_confidence(result)

                metadata = self._build_response_metadata(result, recommendation)
                generations.append([Generation(
                    text=result.consensus_answer or "",
                    generation_info=metadata,
                )])

        return LLMResult(generations=generations)

    def with_config(
        self,
        min_confidence: Optional[float] = None,
        warn_confidence: Optional[float] = None,
        consensus_mode: Optional[ConsensusMode] = None,
        consensus_models: Optional[List[str]] = None,
    ) -> "I2IConsensusLLM":
        """
        Create a copy with updated configuration.

        Useful for temporarily adjusting thresholds or behavior.

        Args:
            min_confidence: New minimum confidence threshold
            warn_confidence: New warning confidence threshold
            consensus_mode: New consensus mode
            consensus_models: New model list

        Returns:
            New I2IConsensusLLM instance with updated config
        """
        return I2IConsensusLLM(
            base_llm=self.base_llm,
            consensus_models=consensus_models or self.consensus_models,
            min_confidence=min_confidence if min_confidence is not None else self.min_confidence,
            warn_confidence=warn_confidence if warn_confidence is not None else self.warn_confidence,
            consensus_mode=consensus_mode or self.consensus_mode,
            protocol=self.protocol,
        )

    def always_consensus(self) -> "I2IConsensusLLM":
        """Return copy that always runs consensus regardless of task type."""
        return self.with_config(consensus_mode=ConsensusMode.ALWAYS)

    def never_consensus(self) -> "I2IConsensusLLM":
        """Return copy that never runs consensus (passthrough mode)."""
        return self.with_config(consensus_mode=ConsensusMode.NEVER)
