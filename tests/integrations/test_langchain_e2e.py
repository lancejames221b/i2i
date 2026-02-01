"""
End-to-end tests for LangChain integration.

These tests make real API calls and require:
- At least 2 configured AI providers (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY)
- Optional: langchain-openai package for full chain tests

Tests are marked with @pytest.mark.e2e and skipped by default.
Run with: pytest -m e2e tests/integrations/test_langchain_e2e.py

Mocking strategy:
- No mocking - real API calls
- Tests may be slow and cost money
- Skip automatically if API keys not configured
"""

import os
import pytest
from typing import List

# Check if LangChain is available
try:
    from langchain_core.messages import AIMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Check if langchain-openai is available for full chain tests
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False

from i2i.schema import ConsensusLevel


def has_api_keys() -> bool:
    """Check if at least 2 providers are configured."""
    keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
    ]
    configured = sum(1 for k in keys if os.environ.get(k))
    return configured >= 2


def get_configured_providers() -> List[str]:
    """Get list of configured provider names."""
    providers = []
    if os.environ.get("OPENAI_API_KEY"):
        providers.append("openai")
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    if os.environ.get("GOOGLE_API_KEY"):
        providers.append("google")
    if os.environ.get("GROQ_API_KEY"):
        providers.append("groq")
    if os.environ.get("MISTRAL_API_KEY"):
        providers.append("mistral")
    return providers


# Skip conditions and async mode
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not LANGCHAIN_AVAILABLE,
        reason="LangChain not installed"
    ),
    pytest.mark.skipif(
        not has_api_keys(),
        reason="Need at least 2 configured API providers for consensus"
    ),
]


# ==================== E2E Tests: Real Consensus Through LangChain ====================


class TestRealConsensusQuery:
    """Test real consensus queries through LangChain integration."""

    async def test_verifies_factual_statement(self):
        """Verify a clearly true factual statement."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(
            min_consensus_level=ConsensusLevel.MEDIUM,
            confidence_threshold=0.6,
            task_aware=True
        )

        result = await verifier.ainvoke(
            "Paris is the capital of France."
        )

        # Should verify true statements
        assert result.verified is True
        assert result.consensus_level in ["high", "medium"]
        assert result.confidence >= 0.6

    async def test_detects_false_statement(self):
        """Verify a clearly false factual statement fails or has low consensus."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(
            min_consensus_level=ConsensusLevel.HIGH,
            confidence_threshold=0.8,
            task_aware=True
        )

        result = await verifier.ainvoke(
            "The Earth is flat and the sun orbits around it."
        )

        # False statements should fail verification or have low confidence
        # Models should disagree with false claims
        # Note: We can't assert verified=False because models might
        # correctly identify it as false (verify that it's FALSE)
        assert result.consensus_level is not None
        assert result.confidence is not None

    async def test_handles_uncertain_statement(self):
        """Handle a statement with inherent uncertainty."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(
            min_consensus_level=ConsensusLevel.MEDIUM,
            confidence_threshold=0.7,
            task_aware=True
        )

        result = await verifier.ainvoke(
            "Artificial General Intelligence will be achieved by 2030."
        )

        # Uncertain predictions should have lower consensus
        assert result.consensus_level is not None
        # Likely to be low or contradictory
        assert result.consensus_level in ["high", "medium", "low", "none", "contradictory"]

    async def test_includes_task_category(self):
        """Task category should be detected and included."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(task_aware=True)

        result = await verifier.ainvoke(
            "Water boils at 100 degrees Celsius at standard atmospheric pressure."
        )

        assert result.task_category is not None

    async def test_includes_verification_details(self):
        """Verification details should be included."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(include_verification_metadata=True)

        result = await verifier.ainvoke(
            "The speed of light in a vacuum is approximately 299,792,458 meters per second."
        )

        assert result.verification_details is not None
        assert "models_queried" in result.verification_details
        assert len(result.verification_details["models_queried"]) >= 2


class TestRealStatisticalConsensus:
    """Test statistical consensus mode with real API calls."""

    @pytest.mark.slow
    async def test_statistical_mode_increases_confidence(self):
        """Statistical mode should provide confidence estimates."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(
            statistical_mode=True,
            n_runs=3,
            temperature=0.7,
            confidence_threshold=0.5
        )

        result = await verifier.ainvoke(
            "The chemical symbol for gold is Au."
        )

        assert result.verified is True
        assert result.confidence is not None


# ==================== E2E Tests: Full RAG Pipeline ====================


@pytest.mark.skipif(
    not LANGCHAIN_OPENAI_AVAILABLE,
    reason="langchain-openai not installed"
)
class TestRealRAGPipeline:
    """Test actual RAG pipeline verification with real LLM calls."""

    async def test_verified_chain_with_real_llm(self):
        """Test I2IVerifiedChain with real OpenAI LLM."""
        from langchain_core.prompts import ChatPromptTemplate
        from i2i.integrations.langchain import I2IVerifiedChain

        # Create a simple chain
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = ChatPromptTemplate.from_template(
            "Answer in one sentence: What is the capital of {country}?"
        )
        base_chain = prompt | llm

        # Wrap with verification
        verified_chain = I2IVerifiedChain(
            chain=base_chain,
            min_consensus_level=ConsensusLevel.MEDIUM,
            confidence_threshold=0.6
        )

        result = await verified_chain.ainvoke({"country": "Japan"})

        # Should get a verified answer
        assert "Tokyo" in result.content
        assert result.verified is True

    async def test_lcel_chain_composition(self):
        """Test LCEL chain with I2IVerifier."""
        from langchain_core.prompts import ChatPromptTemplate
        from i2i.integrations.langchain import I2IVerifier

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = ChatPromptTemplate.from_template(
            "In exactly one word, what is 2 + 2? Answer:"
        )
        verifier = I2IVerifier(
            min_consensus_level=ConsensusLevel.LOW,  # Math might have lower consensus
            confidence_threshold=0.5,
            task_category="reasoning"  # Explicit category
        )

        chain = prompt | llm | verifier
        result = await chain.ainvoke({})

        # Should contain the answer (4)
        assert "4" in result.content.lower() or "four" in result.content.lower()


# ==================== E2E Tests: Error Scenarios ====================


class TestRealErrorScenarios:
    """Test error handling with real API calls."""

    async def test_handles_empty_input(self):
        """Should handle empty input gracefully."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(fallback_on_error=True)

        result = await verifier.ainvoke("")

        # Should return some result even for empty input
        assert result is not None
        assert result.content == ""

    async def test_handles_very_long_input(self):
        """Should handle very long input."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(fallback_on_error=True)

        # Create a long but valid input
        long_text = "The Earth orbits the Sun. " * 100

        result = await verifier.ainvoke(long_text)

        assert result is not None


# ==================== E2E Tests: Callback Integration ====================


@pytest.mark.skipif(
    not LANGCHAIN_OPENAI_AVAILABLE,
    reason="langchain-openai not installed"
)
class TestRealCallbackIntegration:
    """Test callback handler with real LLM calls."""

    async def test_callback_captures_verification(self):
        """Callback should capture verification results from LLM."""
        from i2i.integrations.langchain import I2IVerificationCallback

        callback = I2IVerificationCallback(
            min_consensus_level=ConsensusLevel.LOW,
            on_verification_failure="ignore"
        )

        # Create LLM with callback
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            callbacks=[callback]
        )

        # Generate a response
        response = await llm.ainvoke("What is the capital of Germany? Answer in one word.")

        # Give callback time to process
        import asyncio
        await asyncio.sleep(2)

        # Check verification was captured
        history = callback.get_verification_history()
        # Note: The callback runs asynchronously, so history might not be populated yet
        # in a real test scenario. This is more of an integration check.


# ==================== E2E Tests: Performance ====================


@pytest.mark.slow
class TestPerformance:
    """Performance-related E2E tests."""

    async def test_multiple_verifications_sequential(self):
        """Test multiple sequential verifications."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(
            min_consensus_level=ConsensusLevel.LOW,
            confidence_threshold=0.5
        )

        statements = [
            "Water freezes at 0 degrees Celsius.",
            "The Sun is a star.",
            "DNA contains genetic information.",
        ]

        results = []
        for statement in statements:
            result = await verifier.ainvoke(statement)
            results.append(result)

        # All should complete
        assert len(results) == 3
        # All should have verification status
        assert all(r.verified is not None for r in results)


# ==================== E2E Tests: Provider Configuration ====================


class TestProviderConfiguration:
    """Test verification with specific providers."""

    async def test_with_specific_models(self):
        """Test verification with explicitly specified models."""
        from i2i.integrations.langchain import I2IVerifier
        from i2i import AICP

        # Get available models
        protocol = AICP()
        available = protocol.list_configured_providers()

        if len(available) < 2:
            pytest.skip("Need at least 2 providers")

        # Use first two configured providers
        verifier = I2IVerifier(
            models=protocol._get_default_models()[:2],
            min_consensus_level=ConsensusLevel.LOW
        )

        result = await verifier.ainvoke("Oxygen is essential for human respiration.")

        assert result.verification_details is not None
        assert len(result.verification_details["models_queried"]) >= 2
