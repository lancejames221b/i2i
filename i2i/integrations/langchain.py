"""
LangChain integration for i2i multi-model consensus and verification.

This module provides components for integrating i2i's consensus and
verification capabilities into LangChain pipelines:

- I2IVerifier: A Runnable that verifies LLM outputs using multi-model consensus
- I2IVerificationCallback: Callback handler for automatic verification
- I2IVerifiedChain: Pre-built chain wrapper with verification

Example usage:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from i2i.integrations.langchain import I2IVerifier

    # Create a chain with verification
    llm = ChatOpenAI(model="gpt-4")
    prompt = ChatPromptTemplate.from_template("What is {question}?")
    verifier = I2IVerifier()

    chain = prompt | llm | verifier
    result = await chain.ainvoke({"question": "the capital of France"})
"""

from __future__ import annotations

import asyncio
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage, AIMessage
    from langchain_core.outputs import LLMResult
    from langchain_core.runnables import Runnable, RunnableConfig, RunnableSerializable
    from langchain_core.runnables.utils import Input, Output
    from pydantic import BaseModel, Field

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create stub types for when LangChain isn't installed
    BaseCallbackHandler = object
    Runnable = object
    RunnableSerializable = object

from ..protocol import AICP
from ..schema import ConsensusLevel, VerificationResult
from ..task_classifier import ConsensusTaskCategory, get_task_category


class VerificationConfig(BaseModel):
    """Configuration for I2IVerifier."""

    # Consensus settings
    models: Optional[List[str]] = None  # Models to use; None = auto-select
    min_consensus_level: ConsensusLevel = ConsensusLevel.MEDIUM

    # Task-aware settings
    task_aware: bool = True  # Use task classification to adjust thresholds
    task_category: Optional[str] = None  # Override task detection

    # Confidence thresholds
    confidence_threshold: float = 0.7  # Minimum calibrated confidence to pass

    # Behavior
    raise_on_failure: bool = False  # Raise exception if verification fails
    include_verification_metadata: bool = True  # Add verification details to output
    fallback_on_error: bool = True  # Return original output if verification fails

    # Statistical mode
    statistical_mode: bool = False
    n_runs: int = 3
    temperature: float = 0.7


class VerificationOutput(BaseModel):
    """Output from I2IVerifier including verification metadata."""

    content: str
    verified: bool
    consensus_level: Optional[str] = None
    confidence: Optional[float] = None
    task_category: Optional[str] = None
    consensus_appropriate: Optional[bool] = None
    verification_details: Optional[Dict[str, Any]] = None
    original_content: Optional[str] = None  # If content was modified


if LANGCHAIN_AVAILABLE:
    class I2IVerifier(RunnableSerializable[Union[str, BaseMessage], VerificationOutput]):
        """
        A LangChain Runnable that verifies LLM outputs using i2i multi-model consensus.

        The verifier can be used in LCEL chains to automatically verify outputs:

            chain = prompt | llm | I2IVerifier()

        Or with custom configuration:

            verifier = I2IVerifier(
                min_consensus_level=ConsensusLevel.HIGH,
                confidence_threshold=0.9,
                task_aware=True
            )
            chain = prompt | llm | verifier

        The verifier performs:
        1. Task classification (factual, reasoning, creative, etc.)
        2. Consensus query across multiple models
        3. Confidence calibration based on task type
        4. Optional rejection of low-confidence outputs
        """

        config: VerificationConfig = Field(default_factory=VerificationConfig)
        protocol: Optional[AICP] = Field(default=None, exclude=True)

        class Config:
            arbitrary_types_allowed = True

        def __init__(
            self,
            models: Optional[List[str]] = None,
            min_consensus_level: ConsensusLevel = ConsensusLevel.MEDIUM,
            confidence_threshold: float = 0.7,
            task_aware: bool = True,
            task_category: Optional[str] = None,
            raise_on_failure: bool = False,
            include_verification_metadata: bool = True,
            fallback_on_error: bool = True,
            statistical_mode: bool = False,
            n_runs: int = 3,
            temperature: float = 0.7,
            protocol: Optional[AICP] = None,
            **kwargs,
        ):
            """
            Initialize the I2IVerifier.

            Args:
                models: List of model identifiers to use for verification.
                    If None, uses configured defaults.
                min_consensus_level: Minimum consensus level to pass verification.
                confidence_threshold: Minimum calibrated confidence (0-1) to pass.
                task_aware: Use task classification to adjust verification strategy.
                task_category: Override task detection with explicit category.
                raise_on_failure: Raise VerificationError if verification fails.
                include_verification_metadata: Include details in output.
                fallback_on_error: Return original on errors instead of raising.
                statistical_mode: Use statistical consensus (n-runs per model).
                n_runs: Number of runs per model for statistical mode.
                temperature: Temperature for statistical mode queries.
                protocol: Pre-configured AICP instance (optional).
            """
            config = VerificationConfig(
                models=models,
                min_consensus_level=min_consensus_level,
                confidence_threshold=confidence_threshold,
                task_aware=task_aware,
                task_category=task_category,
                raise_on_failure=raise_on_failure,
                include_verification_metadata=include_verification_metadata,
                fallback_on_error=fallback_on_error,
                statistical_mode=statistical_mode,
                n_runs=n_runs,
                temperature=temperature,
            )
            super().__init__(config=config, **kwargs)
            self.protocol = protocol

        def _get_protocol(self) -> AICP:
            """Get or create AICP instance."""
            if self.protocol is None:
                self.protocol = AICP()
            return self.protocol

        def _extract_content(self, input: Union[str, BaseMessage]) -> str:
            """Extract string content from input."""
            if isinstance(input, str):
                return input
            if isinstance(input, BaseMessage):
                return str(input.content)
            # Try to get content attribute
            if hasattr(input, "content"):
                return str(input.content)
            return str(input)

        async def _verify(self, content: str) -> VerificationOutput:
            """Perform verification using i2i consensus."""
            protocol = self._get_protocol()

            try:
                # Determine task category
                task_cat = None
                if self.config.task_category:
                    task_cat = self.config.task_category
                elif self.config.task_aware:
                    detected = get_task_category(content)
                    task_cat = detected.value if detected else None

                # Perform consensus query
                result = await protocol.consensus_query(
                    query=f"Verify this statement is accurate: {content}",
                    models=self.config.models,
                    statistical_mode=self.config.statistical_mode,
                    n_runs=self.config.n_runs if self.config.statistical_mode else None,
                    temperature=self.config.temperature if self.config.statistical_mode else None,
                    task_aware=self.config.task_aware,
                    task_category=task_cat,
                )

                # Extract consensus level and confidence
                consensus_level = result.consensus_level
                calibrated_confidence = result.confidence_calibration or 0.5
                consensus_appropriate = result.consensus_appropriate

                # Determine if verification passed
                level_order = [
                    ConsensusLevel.NONE,
                    ConsensusLevel.CONTRADICTORY,
                    ConsensusLevel.LOW,
                    ConsensusLevel.MEDIUM,
                    ConsensusLevel.HIGH,
                ]
                meets_level = level_order.index(consensus_level) >= level_order.index(
                    self.config.min_consensus_level
                )
                meets_confidence = calibrated_confidence >= self.config.confidence_threshold
                verified = meets_level and meets_confidence

                # Build verification details
                details = None
                if self.config.include_verification_metadata:
                    details = {
                        "models_queried": result.models_queried,
                        "consensus_answer": result.consensus_answer,
                        "divergences": len(result.divergences),
                        "metadata": result.metadata,
                    }

                return VerificationOutput(
                    content=content,
                    verified=verified,
                    consensus_level=consensus_level.value if consensus_level else None,
                    confidence=calibrated_confidence,
                    task_category=task_cat,
                    consensus_appropriate=consensus_appropriate,
                    verification_details=details,
                )

            except Exception as e:
                if not self.config.fallback_on_error:
                    raise
                # Fallback: return unverified
                return VerificationOutput(
                    content=content,
                    verified=False,
                    verification_details={"error": str(e)} if self.config.include_verification_metadata else None,
                )

        async def ainvoke(
            self,
            input: Union[str, BaseMessage],
            config: Optional[RunnableConfig] = None,
            **kwargs,
        ) -> VerificationOutput:
            """
            Asynchronously verify the input.

            Args:
                input: String or LangChain message to verify.
                config: Optional runnable config.

            Returns:
                VerificationOutput with verification results.

            Raises:
                VerificationError: If raise_on_failure=True and verification fails.
            """
            content = self._extract_content(input)
            result = await self._verify(content)

            if not result.verified and self.config.raise_on_failure:
                raise VerificationError(
                    f"Verification failed: consensus_level={result.consensus_level}, "
                    f"confidence={result.confidence}"
                )

            return result

        def invoke(
            self,
            input: Union[str, BaseMessage],
            config: Optional[RunnableConfig] = None,
            **kwargs,
        ) -> VerificationOutput:
            """
            Synchronously verify the input.

            Uses asyncio.run() internally. For better performance in async
            contexts, use ainvoke() directly.
            """
            return asyncio.get_event_loop().run_until_complete(
                self.ainvoke(input, config, **kwargs)
            )

        @property
        def InputType(self) -> Type:
            """Return input type."""
            return Union[str, BaseMessage]

        @property
        def OutputType(self) -> Type:
            """Return output type."""
            return VerificationOutput


    class I2IVerificationCallback(BaseCallbackHandler):
        """
        LangChain callback handler that automatically verifies LLM outputs.

        This callback intercepts LLM responses and verifies them using i2i.
        It can be used with any LangChain LLM by adding it to callbacks:

            callback = I2IVerificationCallback(on_verification_failure="warn")
            llm = ChatOpenAI(callbacks=[callback])

        Verification results are available via the get_last_verification() method.
        """

        def __init__(
            self,
            models: Optional[List[str]] = None,
            min_consensus_level: ConsensusLevel = ConsensusLevel.MEDIUM,
            confidence_threshold: float = 0.7,
            on_verification_failure: str = "warn",  # "warn", "raise", "ignore"
            task_aware: bool = True,
            protocol: Optional[AICP] = None,
        ):
            """
            Initialize the verification callback.

            Args:
                models: Models to use for verification.
                min_consensus_level: Minimum consensus level to pass.
                confidence_threshold: Minimum confidence to pass.
                on_verification_failure: Action on failure ("warn", "raise", "ignore").
                task_aware: Use task classification.
                protocol: Pre-configured AICP instance.
            """
            super().__init__()
            self.verifier = I2IVerifier(
                models=models,
                min_consensus_level=min_consensus_level,
                confidence_threshold=confidence_threshold,
                task_aware=task_aware,
                raise_on_failure=on_verification_failure == "raise",
                protocol=protocol,
            )
            self.on_failure = on_verification_failure
            self._last_verification: Optional[VerificationOutput] = None
            self._verification_history: List[VerificationOutput] = []

        def on_llm_end(self, response: LLMResult, **kwargs) -> None:
            """Called when LLM finishes. Verifies the output."""
            for generations in response.generations:
                for gen in generations:
                    content = gen.text if hasattr(gen, "text") else str(gen)
                    # Run verification synchronously
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Schedule for later execution
                            asyncio.ensure_future(self._verify_async(content))
                        else:
                            result = loop.run_until_complete(
                                self.verifier.ainvoke(content)
                            )
                            self._handle_result(result)
                    except RuntimeError:
                        # No event loop, create one
                        result = asyncio.run(self.verifier.ainvoke(content))
                        self._handle_result(result)

        async def _verify_async(self, content: str) -> None:
            """Async verification for running event loops."""
            result = await self.verifier.ainvoke(content)
            self._handle_result(result)

        def _handle_result(self, result: VerificationOutput) -> None:
            """Handle verification result."""
            self._last_verification = result
            self._verification_history.append(result)

            if not result.verified and self.on_failure == "warn":
                import warnings
                warnings.warn(
                    f"Verification failed: {result.consensus_level} consensus, "
                    f"{result.confidence:.2f} confidence"
                )

        def get_last_verification(self) -> Optional[VerificationOutput]:
            """Get the most recent verification result."""
            return self._last_verification

        def get_verification_history(self) -> List[VerificationOutput]:
            """Get all verification results from this session."""
            return self._verification_history.copy()

        def clear_history(self) -> None:
            """Clear verification history."""
            self._verification_history.clear()
            self._last_verification = None


    class I2IVerifiedChain:
        """
        A wrapper that adds i2i verification to any LangChain chain.

        This provides a convenient way to verify outputs from complex chains:

            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate

            llm = ChatOpenAI(model="gpt-4")
            prompt = ChatPromptTemplate.from_template("Explain {topic}")
            base_chain = prompt | llm

            verified_chain = I2IVerifiedChain(
                chain=base_chain,
                min_consensus_level=ConsensusLevel.HIGH
            )

            result = await verified_chain.ainvoke({"topic": "quantum computing"})
        """

        def __init__(
            self,
            chain: Runnable,
            models: Optional[List[str]] = None,
            min_consensus_level: ConsensusLevel = ConsensusLevel.MEDIUM,
            confidence_threshold: float = 0.7,
            task_aware: bool = True,
            task_category: Optional[str] = None,
            raise_on_failure: bool = False,
            protocol: Optional[AICP] = None,
        ):
            """
            Wrap a chain with verification.

            Args:
                chain: The LangChain chain to wrap.
                models: Models for verification.
                min_consensus_level: Minimum consensus to pass.
                confidence_threshold: Minimum confidence to pass.
                task_aware: Use task classification.
                task_category: Override task detection.
                raise_on_failure: Raise on verification failure.
                protocol: Pre-configured AICP instance.
            """
            self.chain = chain
            self.verifier = I2IVerifier(
                models=models,
                min_consensus_level=min_consensus_level,
                confidence_threshold=confidence_threshold,
                task_aware=task_aware,
                task_category=task_category,
                raise_on_failure=raise_on_failure,
                protocol=protocol,
            )

        async def ainvoke(
            self,
            input: Any,
            config: Optional[RunnableConfig] = None,
            **kwargs,
        ) -> VerificationOutput:
            """
            Execute chain and verify output asynchronously.

            Args:
                input: Input to the chain.
                config: Optional runnable config.

            Returns:
                VerificationOutput with verified result.
            """
            # Execute the chain
            chain_result = await self.chain.ainvoke(input, config, **kwargs)

            # Verify the result
            return await self.verifier.ainvoke(chain_result, config)

        def invoke(
            self,
            input: Any,
            config: Optional[RunnableConfig] = None,
            **kwargs,
        ) -> VerificationOutput:
            """
            Execute chain and verify output synchronously.
            """
            return asyncio.get_event_loop().run_until_complete(
                self.ainvoke(input, config, **kwargs)
            )

        def __or__(self, other: Runnable) -> Runnable:
            """Support chaining with | operator."""
            from langchain_core.runnables import RunnableSequence
            return RunnableSequence(first=self.chain | self.verifier, last=other)


    class VerificationError(Exception):
        """Raised when verification fails and raise_on_failure is True."""
        pass


    def create_verified_chain(
        chain: Runnable,
        models: Optional[List[str]] = None,
        min_consensus_level: ConsensusLevel = ConsensusLevel.MEDIUM,
        confidence_threshold: float = 0.7,
        task_aware: bool = True,
        protocol: Optional[AICP] = None,
    ) -> Runnable:
        """
        Create a verified chain using LCEL composition.

        This is a convenience function that adds an I2IVerifier to any chain.

        Args:
            chain: The chain to add verification to.
            models: Models for verification.
            min_consensus_level: Minimum consensus to pass.
            confidence_threshold: Minimum confidence to pass.
            task_aware: Use task classification.
            protocol: Pre-configured AICP instance.

        Returns:
            A new Runnable with verification added.

        Example:
            verified = create_verified_chain(prompt | llm)
            result = await verified.ainvoke({"question": "What is 2+2?"})
        """
        verifier = I2IVerifier(
            models=models,
            min_consensus_level=min_consensus_level,
            confidence_threshold=confidence_threshold,
            task_aware=task_aware,
            protocol=protocol,
        )
        return chain | verifier


else:
    # Stubs when LangChain is not installed
    class I2IVerifier:
        """LangChain is not installed. Install with: pip install langchain-core"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LangChain integration requires langchain-core. "
                "Install with: pip install langchain-core"
            )

    class I2IVerificationCallback:
        """LangChain is not installed. Install with: pip install langchain-core"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LangChain integration requires langchain-core. "
                "Install with: pip install langchain-core"
            )

    class I2IVerifiedChain:
        """LangChain is not installed. Install with: pip install langchain-core"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LangChain integration requires langchain-core. "
                "Install with: pip install langchain-core"
            )

    class VerificationError(Exception):
        """Raised when verification fails and raise_on_failure is True."""
        pass

    def create_verified_chain(*args, **kwargs):
        """LangChain is not installed. Install with: pip install langchain-core"""
        raise ImportError(
            "LangChain integration requires langchain-core. "
            "Install with: pip install langchain-core"
        )
