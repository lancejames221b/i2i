"""
i2i integrations with popular AI/ML frameworks.

This module provides seamless integration with:
- LangChain: Multi-model verification for RAG pipelines
"""

from .langchain import (
    I2IVerifier,
    I2IVerificationCallback,
    I2IVerifiedChain,
    create_verified_chain,
)

__all__ = [
    "I2IVerifier",
    "I2IVerificationCallback",
    "I2IVerifiedChain",
    "create_verified_chain",
]
