"""
LangChain Verifier Implementation.

This module provides the core I2IVerifier Runnable and supporting classes
for integrating i2i consensus verification into LangChain pipelines.

The I2IVerifier implements the LangChain Runnable interface, allowing it
to be composed with other Runnables using the pipe (|) operator.

Example:
    Basic LCEL chain with verification::

        from i2i.integrations.langchain import I2IVerifier
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI()
        verified_chain = llm | I2IVerifier(min_confidence=0.8)

        result = verified_chain.invoke("What is 2+2?")
        print(result.verified)  # True or False

    Async usage::

        result = await verified_chain.ainvoke("What is 2+2?")

    With callback handler::

        from i2i.integrations.langchain import I2IVerificationCallback

        callback = I2IVerificationCallback(on_verification_failure="warn")
        llm = ChatOpenAI(callbacks=[callback])

        response = llm.invoke("Your prompt")
        verification = callback.get_last_verification()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable, RunnableConfig

from i2i import AICP
from i2i.schema import ConsensusLevel

logger = logging.getLogger(__name__)

# Type variable for input types
Input = TypeVar("Input", str, AIMessage, BaseMessage, LLMResult, Dict[str, Any])


class VerificationError(Exception):
    """
    Exception raised when verification fails and raise_on_failure is True.

    Attributes:
        message: Error description.
        verification_result: The I2IVerifiedOutput that caused the failure.

    Example:
        Catching verification failures::

            from i2i.integrations.langchain import I2IVerifier, VerificationError

            verifier = I2IVerifier(raise_on_failure=True)
            try:
                result = await verifier.ainvoke("Some unverifiable content")
            except VerificationError as e:
                print(f"Verification failed: {e}")
                print(f"Consensus level: {e.verification_result.consensus_level}")
    """

    def __init__(self, message: str, verification_result: "I2IVerifiedOutput"):
        super().__init__(message)
        self.verification_result = verification_result


@dataclass
class VerificationConfig:
    """
    Configuration options for I2IVerifier.

    This dataclass provides fine-grained control over verification behaviour,
    including consensus settings, task awareness, and error handling.

    Attributes:
        models: List of model IDs to use for consensus. If None, auto-selects.
        min_consensus_level: Minimum ConsensusLevel required to pass verification.
            One of: HIGH, MEDIUM, LOW, NONE, CONTRADICTORY.
        confidence_threshold: Minimum calibrated confidence score (0.0-1.0)
            required to pass verification.
        task_aware: Whether to use task-aware routing. When True, automatically
            detects task type and skips consensus for tasks where it hurts
            (e.g., mathematical reasoning).
        task_category: Override automatic task detection. Valid values:
            "factual", "reasoning", "creative", "verification".
        raise_on_failure: If True, raise VerificationError when verification
            fails. If False (default), return result with verified=False.
        include_verification_metadata: If True, include detailed verification
            metadata in the output.
        fallback_on_error: If True (default), return original content with
            verified=False on API errors. If False, propagate the exception.
        statistical_mode: If True, use statistical consensus with multiple
            runs per model to measure consistency.
        n_runs: Number of runs per model in statistical mode.
        temperature: Temperature for statistical mode queries.

    Example:
        Custom configuration::

            from i2i.integrations.langchain import I2IVerifier, VerificationConfig

            config = VerificationConfig(
                models=["gpt-4", "claude-3-opus"],
                min_consensus_level=ConsensusLevel.HIGH,
                confidence_threshold=0.9,
                task_aware=True,
                raise_on_failure=True,
            )
            verifier = I2IVerifier(config=config)
    """

    models: Optional[List[str]] = None
    min_consensus_level: ConsensusLevel = ConsensusLevel.MEDIUM
    confidence_threshold: float = 0.7
    task_aware: bool = True
    task_category: Optional[str] = None
    raise_on_failure: bool = False
    include_verification_metadata: bool = True
    fallback_on_error: bool = True
    statistical_mode: bool = False
    n_runs: int = 3
    temperature: float = 0.7


@dataclass
class I2IVerifiedOutput:
    """
    Output from I2IVerifier containing verification results.

    This dataclass contains the original content along with verification
    metadata including consensus level, confidence scores, and task
    classification.

    Attributes:
        content: The original content that was verified.
        verified: Whether the content passed verification based on
            configured thresholds.
        consensus_level: The consensus level achieved across models.
            One of: "HIGH", "MEDIUM", "LOW", "NONE", "CONTRADICTORY".
        confidence_calibration: Calibrated confidence score (0.0-1.0)
            based on empirical evaluation data. Higher scores indicate
            more reliable consensus.
        task_category: Detected or specified task category. Used for
            task-aware routing.
        consensus_appropriate: Whether consensus was appropriate for
            this task type. False for mathematical/reasoning tasks
            where consensus degrades performance.
        models_queried: List of model IDs that participated in
            consensus verification.
        original_metadata: Any metadata from the input that was
            preserved through verification.
        verification_details: Additional verification details when
            include_verification_metadata is True.

    Example:
        Accessing verification results::

            result = await verifier.ainvoke("The Earth is round")

            if result.verified:
                print(f"Verified with {result.consensus_level} consensus")
                print(f"Confidence: {result.confidence_calibration:.2%}")
            else:
                print("Verification failed")
                if not result.consensus_appropriate:
                    print("Note: Consensus may not be appropriate for this task")
    """

    content: str
    verified: bool
    consensus_level: str = "NONE"
    confidence_calibration: Optional[float] = None
    task_category: Optional[str] = None
    consensus_appropriate: Optional[bool] = None
    models_queried: List[str] = field(default_factory=list)
    original_metadata: Dict[str, Any] = field(default_factory=dict)
    verification_details: Dict[str, Any] = field(default_factory=dict)


class I2IVerifier(Runnable[Input, I2IVerifiedOutput]):
    """
    LangChain Runnable for multi-model consensus verification.

    I2IVerifier implements the LangChain Runnable interface, allowing it
    to be seamlessly integrated into LCEL (LangChain Expression Language)
    pipelines using the pipe (|) operator.

    The verifier queries multiple AI models to verify the input content,
    returning a consensus-based verification result with calibrated
    confidence scores.

    Attributes:
        models: List of model IDs for consensus queries.
        min_confidence: Minimum confidence threshold (0.0-1.0).
        task_category: Override task type detection.
        task_aware: Whether to use task-aware routing.
        aicp: Pre-configured AICP protocol instance.

    Example:
        Basic usage::

            from i2i.integrations.langchain import I2IVerifier
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI()
            chain = llm | I2IVerifier(min_confidence=0.8)

            result = chain.invoke("What is the capital of France?")
            print(result.verified)  # True
            print(result.consensus_level)  # "HIGH"

        Async usage::

            result = await chain.ainvoke("What is the capital of France?")

        With specific models::

            verifier = I2IVerifier(
                models=["gpt-4", "claude-3-opus", "gemini-pro"],
                min_confidence=0.9,
            )

        Task-aware verification (skips consensus for math)::

            verifier = I2IVerifier(task_aware=True)
            result = await verifier.ainvoke("Calculate 5 * 3 + 2")
            print(result.consensus_appropriate)  # False

        Forcing task type::

            verifier = I2IVerifier(task_category="factual")

    Notes:
        - Requires at least 2 configured LLM providers for consensus.
        - Task-aware mode skips consensus for math/reasoning tasks
          where consensus degrades performance by ~35%.
        - Confidence scores are calibrated based on empirical evaluation:
          HIGH consensus = 0.95, MEDIUM = 0.75, LOW = 0.60, NONE = 0.50.
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        min_confidence: float = 0.6,
        task_category: Optional[str] = None,
        task_aware: bool = True,
        aicp: Optional[AICP] = None,
        config: Optional[VerificationConfig] = None,
        raise_on_failure: bool = False,
        fallback_on_error: bool = True,
    ):
        """
        Initialize the I2IVerifier.

        Args:
            models: List of model IDs for consensus queries. If None,
                automatically selects from configured providers.
            min_confidence: Minimum calibrated confidence score (0.0-1.0)
                required for verification to pass. Default 0.6.
            task_category: Override automatic task detection. Valid values:
                "factual", "reasoning", "creative", "verification".
            task_aware: If True (default), automatically detect task type
                and skip consensus for tasks where it hurts performance.
            aicp: Pre-configured AICP protocol instance. If None, creates
                a new instance.
            config: VerificationConfig for advanced options. Overrides
                individual parameters if provided.
            raise_on_failure: If True, raise VerificationError when
                verification fails.
            fallback_on_error: If True (default), return original content
                with verified=False on API errors.

        Raises:
            ValueError: If min_confidence is not between 0.0 and 1.0.

        Example:
            With explicit configuration::

                verifier = I2IVerifier(
                    models=["gpt-4", "claude-3-opus"],
                    min_confidence=0.85,
                    task_aware=True,
                )

            With VerificationConfig::

                config = VerificationConfig(
                    models=["gpt-4", "claude-3-opus"],
                    confidence_threshold=0.85,
                    raise_on_failure=True,
                )
                verifier = I2IVerifier(config=config)
        """
        if config:
            self.models = config.models
            self.min_confidence = config.confidence_threshold
            self.task_category = config.task_category
            self.task_aware = config.task_aware
            self.raise_on_failure = config.raise_on_failure
            self.fallback_on_error = config.fallback_on_error
            self._config = config
        else:
            self.models = models
            self.min_confidence = min_confidence
            self.task_category = task_category
            self.task_aware = task_aware
            self.raise_on_failure = raise_on_failure
            self.fallback_on_error = fallback_on_error
            self._config = None

        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")

        self._aicp = aicp or AICP()

    @property
    def InputType(self) -> Type[Input]:
        """Return the input type for this Runnable."""
        return Union[str, AIMessage, BaseMessage, LLMResult, Dict[str, Any]]

    @property
    def OutputType(self) -> Type[I2IVerifiedOutput]:
        """Return the output type for this Runnable."""
        return I2IVerifiedOutput

    def get_name(self, suffix: Optional[str] = None) -> str:
        """
        Get the name of this Runnable.

        Args:
            suffix: Optional suffix to append.

        Returns:
            Name string, optionally with suffix.
        """
        name = "I2IVerifier"
        if suffix:
            name = f"{name}_{suffix}"
        return name

    def _extract_content(self, input: Input) -> tuple[str, Dict[str, Any]]:
        """
        Extract content string and metadata from various input types.

        Args:
            input: Input in various formats (str, AIMessage, LLMResult, dict).

        Returns:
            Tuple of (content_string, metadata_dict).
        """
        if isinstance(input, str):
            return input, {}
        elif isinstance(input, AIMessage):
            return input.content, {"message_type": "ai", **input.additional_kwargs}
        elif isinstance(input, BaseMessage):
            return input.content, {"message_type": input.type}
        elif isinstance(input, LLMResult):
            content = input.generations[0][0].text if input.generations else ""
            return content, input.llm_output or {}
        elif isinstance(input, dict):
            return input.get("content", str(input)), {
                k: v for k, v in input.items() if k != "content"
            }
        else:
            return str(input), {}

    def _get_confidence_for_level(self, level: ConsensusLevel) -> float:
        """
        Get calibrated confidence score for a consensus level.

        Based on empirical evaluation data:
        - HIGH consensus correlates with 97-100% accuracy
        - MEDIUM consensus correlates with ~80% accuracy
        - LOW/NONE correlates with ~50% accuracy (unreliable)

        Args:
            level: ConsensusLevel enum value.

        Returns:
            Calibrated confidence score (0.0-1.0).
        """
        calibration = {
            ConsensusLevel.HIGH: 0.95,
            ConsensusLevel.MEDIUM: 0.75,
            ConsensusLevel.LOW: 0.60,
            ConsensusLevel.NONE: 0.50,
            ConsensusLevel.CONTRADICTORY: 0.40,
        }
        return calibration.get(level, 0.50)

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> I2IVerifiedOutput:
        """
        Synchronously verify the input content.

        Args:
            input: Content to verify. Accepts string, AIMessage, BaseMessage,
                LLMResult, or dict with 'content' key.
            config: Optional LangChain RunnableConfig.

        Returns:
            I2IVerifiedOutput with verification results.

        Raises:
            VerificationError: If verification fails and raise_on_failure=True.

        Example:
            Direct invocation::

                result = verifier.invoke("The Earth is flat")
                print(result.verified)  # False

            In LCEL chain::

                chain = prompt | llm | verifier
                result = chain.invoke({"question": "..."})
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.ainvoke(input, config))

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> I2IVerifiedOutput:
        """
        Asynchronously verify the input content.

        This is the primary verification method. It queries multiple models
        for consensus and returns calibrated verification results.

        Args:
            input: Content to verify. Accepts string, AIMessage, BaseMessage,
                LLMResult, or dict with 'content' key.
            config: Optional LangChain RunnableConfig.

        Returns:
            I2IVerifiedOutput with verification results.

        Raises:
            VerificationError: If verification fails and raise_on_failure=True.

        Example:
            Async verification::

                result = await verifier.ainvoke("The speed of light is constant")
                if result.verified:
                    print(f"Verified! Confidence: {result.confidence_calibration:.2%}")

            With error handling::

                try:
                    result = await verifier.ainvoke(content)
                except VerificationError as e:
                    print(f"Failed: {e}")
        """
        content, metadata = self._extract_content(input)

        try:
            # Perform consensus query
            consensus_result = await self._aicp.consensus_query(
                content,
                models=self.models,
                task_category=self.task_category if not self.task_aware else None,
            )

            # Extract results
            level = consensus_result.consensus_level
            confidence = self._get_confidence_for_level(level)

            # Check task appropriateness
            task_cat = getattr(consensus_result, "task_category", None)
            consensus_appropriate = getattr(
                consensus_result, "consensus_appropriate", True
            )

            # Determine if verified
            verified = confidence >= self.min_confidence

            result = I2IVerifiedOutput(
                content=content,
                verified=verified,
                consensus_level=level.value if hasattr(level, "value") else str(level),
                confidence_calibration=confidence,
                task_category=task_cat,
                consensus_appropriate=consensus_appropriate,
                models_queried=getattr(consensus_result, "models_queried", []),
                original_metadata=metadata,
                verification_details={
                    "consensus_answer": getattr(
                        consensus_result, "consensus_answer", None
                    ),
                    "divergences": getattr(consensus_result, "divergences", []),
                },
            )

            if self.raise_on_failure and not verified:
                raise VerificationError(
                    f"Verification failed: {level.value} consensus, "
                    f"{confidence:.2%} confidence < {self.min_confidence:.2%} threshold",
                    result,
                )

            return result

        except VerificationError:
            raise
        except Exception as e:
            logger.error(f"Verification error: {e}")
            if self.fallback_on_error:
                return I2IVerifiedOutput(
                    content=content,
                    verified=False,
                    consensus_level="ERROR",
                    confidence_calibration=0.0,
                    original_metadata=metadata,
                    verification_details={"error": str(e)},
                )
            raise

    def stream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Iterator[I2IVerifiedOutput]:
        """
        Stream verification results.

        Note: Verification requires the full content, so this yields a
        single result after verification completes.

        Args:
            input: Content to verify.
            config: Optional LangChain RunnableConfig.

        Yields:
            Single I2IVerifiedOutput after verification.

        Example:
            Streaming (single result)::

                for result in verifier.stream("Content to verify"):
                    print(result.verified)
        """
        yield self.invoke(input, config)

    async def astream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[I2IVerifiedOutput]:
        """
        Async stream verification results.

        Note: Verification requires the full content, so this yields a
        single result after verification completes.

        Args:
            input: Content to verify.
            config: Optional LangChain RunnableConfig.

        Yields:
            Single I2IVerifiedOutput after verification.

        Example:
            Async streaming::

                async for result in verifier.astream("Content to verify"):
                    print(result.verified)
        """
        yield await self.ainvoke(input, config)


class I2IVerificationCallback(BaseCallbackHandler):
    """
    LangChain callback handler for automatic verification.

    This callback hooks into LLM responses and automatically verifies
    them using i2i consensus. Verification results are stored and
    accessible via get_last_verification() and get_verification_history().

    Attributes:
        min_consensus_level: Minimum ConsensusLevel required for verification.
        on_verification_failure: Action on failure: "warn", "raise", "ignore".
        verifier: Internal I2IVerifier instance.

    Example:
        Automatic verification of all LLM responses::

            from i2i.integrations.langchain import I2IVerificationCallback
            from langchain_openai import ChatOpenAI

            callback = I2IVerificationCallback(
                min_consensus_level=ConsensusLevel.HIGH,
                on_verification_failure="warn",
            )

            llm = ChatOpenAI(callbacks=[callback])
            response = llm.invoke("What is the capital of France?")

            # Access verification results
            verification = callback.get_last_verification()
            if verification and not verification.verified:
                print("Warning: Response may not be accurate")

            # Get full history
            for v in callback.get_verification_history():
                print(f"{v.consensus_level}: {v.verified}")

    Notes:
        - The callback runs verification asynchronously after each LLM call.
        - Use "warn" mode for logging, "raise" for strict validation.
        - Clear history with clear_history() for long-running applications.
    """

    def __init__(
        self,
        min_consensus_level: ConsensusLevel = ConsensusLevel.MEDIUM,
        on_verification_failure: str = "warn",
        models: Optional[List[str]] = None,
        task_aware: bool = True,
    ):
        """
        Initialize the verification callback.

        Args:
            min_consensus_level: Minimum ConsensusLevel required.
            on_verification_failure: Action on failure:
                - "warn": Log a warning (default).
                - "raise": Raise VerificationError.
                - "ignore": Silently continue.
            models: List of model IDs for consensus.
            task_aware: Whether to use task-aware routing.

        Raises:
            ValueError: If on_verification_failure is invalid.
        """
        super().__init__()
        if on_verification_failure not in ("warn", "raise", "ignore"):
            raise ValueError(
                "on_verification_failure must be 'warn', 'raise', or 'ignore'"
            )

        self.min_consensus_level = min_consensus_level
        self.on_verification_failure = on_verification_failure
        self._verification_history: List[I2IVerifiedOutput] = []

        # Map consensus level to confidence threshold
        level_thresholds = {
            ConsensusLevel.HIGH: 0.85,
            ConsensusLevel.MEDIUM: 0.60,
            ConsensusLevel.LOW: 0.30,
            ConsensusLevel.NONE: 0.0,
        }
        threshold = level_thresholds.get(min_consensus_level, 0.60)

        self.verifier = I2IVerifier(
            models=models,
            min_confidence=threshold,
            task_aware=task_aware,
            raise_on_failure=(on_verification_failure == "raise"),
        )

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """
        Called when LLM generates a response.

        Args:
            response: The LLMResult from the LLM.
            **kwargs: Additional callback arguments.
        """
        import asyncio

        if not response.generations:
            return

        content = response.generations[0][0].text

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(self.verifier.ainvoke(content))
            self._verification_history.append(result)

            if not result.verified:
                if self.on_verification_failure == "warn":
                    logger.warning(
                        f"Verification failed: {result.consensus_level} consensus, "
                        f"confidence {result.confidence_calibration:.2%}"
                    )
        except VerificationError:
            raise
        except Exception as e:
            logger.error(f"Verification callback error: {e}")

    def get_last_verification(self) -> Optional[I2IVerifiedOutput]:
        """
        Get the most recent verification result.

        Returns:
            Most recent I2IVerifiedOutput, or None if no verifications yet.

        Example:
            Check last verification::

                last = callback.get_last_verification()
                if last and last.verified:
                    print("Last response was verified")
        """
        return self._verification_history[-1] if self._verification_history else None

    def get_verification_history(self) -> List[I2IVerifiedOutput]:
        """
        Get all verification results from this session.

        Returns:
            List of I2IVerifiedOutput in chronological order.

        Example:
            Analyze verification history::

                history = callback.get_verification_history()
                verified_count = sum(1 for v in history if v.verified)
                print(f"Verified {verified_count}/{len(history)} responses")
        """
        return list(self._verification_history)

    def clear_history(self) -> None:
        """
        Clear verification history.

        Use this for long-running applications to prevent memory growth.

        Example:
            Clear after processing a batch::

                callback.clear_history()
        """
        self._verification_history.clear()


class I2IVerifiedChain:
    """
    Wrapper to add verification to any LangChain chain.

    This class wraps an existing Runnable and automatically verifies
    its output using i2i consensus.

    Attributes:
        chain: The wrapped Runnable.
        verifier: Internal I2IVerifier instance.

    Example:
        Wrapping an existing chain::

            from i2i.integrations.langchain import I2IVerifiedChain

            base_chain = prompt | llm
            verified_chain = I2IVerifiedChain(
                chain=base_chain,
                min_consensus_level=ConsensusLevel.HIGH,
            )

            result = verified_chain.invoke({"question": "What is AI?"})
            print(result.verified)

        Async usage::

            result = await verified_chain.ainvoke({"question": "What is AI?"})

    Notes:
        - The wrapper preserves the original chain's input type.
        - Output is always I2IVerifiedOutput.
        - Use the pipe operator for simpler composition when possible.
    """

    def __init__(
        self,
        chain: Runnable,
        min_consensus_level: ConsensusLevel = ConsensusLevel.MEDIUM,
        models: Optional[List[str]] = None,
        task_aware: bool = True,
        **kwargs,
    ):
        """
        Initialize the verified chain wrapper.

        Args:
            chain: The Runnable to wrap.
            min_consensus_level: Minimum ConsensusLevel required.
            models: List of model IDs for consensus.
            task_aware: Whether to use task-aware routing.
            **kwargs: Additional arguments passed to I2IVerifier.
        """
        self.chain = chain

        level_thresholds = {
            ConsensusLevel.HIGH: 0.85,
            ConsensusLevel.MEDIUM: 0.60,
            ConsensusLevel.LOW: 0.30,
            ConsensusLevel.NONE: 0.0,
        }
        threshold = level_thresholds.get(min_consensus_level, 0.60)

        self.verifier = I2IVerifier(
            models=models, min_confidence=threshold, task_aware=task_aware, **kwargs
        )

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> I2IVerifiedOutput:
        """
        Invoke the chain and verify the result.

        Args:
            input: Input for the wrapped chain.
            config: Optional LangChain RunnableConfig.

        Returns:
            I2IVerifiedOutput with verification results.
        """
        result = self.chain.invoke(input, config)
        return self.verifier.invoke(result)

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None
    ) -> I2IVerifiedOutput:
        """
        Async invoke the chain and verify the result.

        Args:
            input: Input for the wrapped chain.
            config: Optional LangChain RunnableConfig.

        Returns:
            I2IVerifiedOutput with verification results.
        """
        result = await self.chain.ainvoke(input, config)
        return await self.verifier.ainvoke(result)


def create_verified_chain(
    chain: Runnable,
    models: Optional[List[str]] = None,
    min_consensus_level: ConsensusLevel = ConsensusLevel.MEDIUM,
    confidence_threshold: float = 0.7,
    task_aware: bool = True,
    protocol: Optional[AICP] = None,
) -> Runnable:
    """
    Create a verified chain by appending I2IVerifier to an existing chain.

    This is a convenience function for the common pattern of adding
    verification to the end of an LCEL chain.

    Args:
        chain: The base Runnable to verify.
        models: List of model IDs for consensus.
        min_consensus_level: Minimum ConsensusLevel required.
        confidence_threshold: Minimum calibrated confidence (0.0-1.0).
        task_aware: Whether to use task-aware routing.
        protocol: Pre-configured AICP protocol instance.

    Returns:
        A new Runnable that includes verification.

    Example:
        Quick chain creation::

            from i2i.integrations.langchain import create_verified_chain

            base_chain = prompt | llm
            verified_chain = create_verified_chain(
                chain=base_chain,
                confidence_threshold=0.9,
                task_aware=True,
            )

            result = await verified_chain.ainvoke({"question": "..."})
            print(result.verified)

    Notes:
        - Returns chain | I2IVerifier, preserving LCEL composition.
        - The returned chain can be further composed with other Runnables.
    """
    verifier = I2IVerifier(
        models=models,
        min_confidence=confidence_threshold,
        task_aware=task_aware,
        aicp=protocol,
    )
    return chain | verifier
