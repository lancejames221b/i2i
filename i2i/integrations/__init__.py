"""
i2i Integrations Package.

This package provides integrations with popular AI/ML frameworks.

Available integrations:
    - langchain: LangChain LCEL integration for consensus verification
"""

from i2i.integrations.langchain import (
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
