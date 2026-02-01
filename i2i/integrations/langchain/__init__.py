"""
LangChain Integration for i2i.

This module provides LangChain-compatible components for integrating
i2i's multi-model consensus verification into LCEL (LangChain Expression
Language) pipelines.

Example:
    Basic usage with LCEL chain::

        from i2i.integrations.langchain import I2IVerifier
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(model="gpt-4")
        prompt = ChatPromptTemplate.from_template("Answer: {question}")

        # Add verification to your chain
        chain = prompt | llm | I2IVerifier(min_confidence=0.8)

        result = chain.invoke({"question": "What is the capital of France?"})
        print(result.verified)  # True
        print(result.consensus_level)  # "HIGH"

Classes:
    I2IVerifier: Main Runnable component for LCEL chains.
    I2IVerifiedOutput: Output model containing verification results.
    I2IVerificationCallback: Callback handler for automatic verification.
    I2IVerifiedChain: Wrapper to add verification to any chain.
    VerificationConfig: Configuration options for verification.
    VerificationError: Exception raised when verification fails.

Functions:
    create_verified_chain: Helper to create a verified chain.
"""

try:
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.callbacks import BaseCallbackHandler

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


if _LANGCHAIN_AVAILABLE:
    from i2i.integrations.langchain.verifier import (
        I2IVerifiedOutput,
        I2IVerifier,
        I2IVerificationCallback,
        I2IVerifiedChain,
        VerificationConfig,
        VerificationError,
        create_verified_chain,
    )

    __all__ = [
        "I2IVerifiedOutput",
        "I2IVerifier",
        "I2IVerificationCallback",
        "I2IVerifiedChain",
        "VerificationConfig",
        "VerificationError",
        "create_verified_chain",
    ]
else:

    class _LangChainNotAvailable:
        """Placeholder when LangChain is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LangChain integration requires langchain-core. "
                "Install with: pip install i2i-mcip[langchain] "
                "or: pip install langchain-core>=0.1.0"
            )

    I2IVerifiedOutput = _LangChainNotAvailable
    I2IVerifier = _LangChainNotAvailable
    I2IVerificationCallback = _LangChainNotAvailable
    I2IVerifiedChain = _LangChainNotAvailable
    VerificationConfig = _LangChainNotAvailable
    VerificationError = _LangChainNotAvailable

    def create_verified_chain(*args, **kwargs):
        """Create a verified chain (requires langchain-core)."""
        raise ImportError(
            "LangChain integration requires langchain-core. "
            "Install with: pip install i2i-mcip[langchain] "
            "or: pip install langchain-core>=0.1.0"
        )

    __all__ = [
        "I2IVerifiedOutput",
        "I2IVerifier",
        "I2IVerificationCallback",
        "I2IVerifiedChain",
        "VerificationConfig",
        "VerificationError",
        "create_verified_chain",
    ]
