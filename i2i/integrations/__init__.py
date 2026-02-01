"""
i2i integrations with external frameworks.

Provides wrappers and adapters for integrating i2i consensus verification
with popular AI/ML frameworks like LangChain.
"""

from .langchain import (
    I2IConsensusLLM,
    LowConfidenceError,
    ConsensusMode,
)

__all__ = [
    "I2IConsensusLLM",
    "LowConfidenceError",
    "ConsensusMode",
]
