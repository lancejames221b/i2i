"""Unit tests for LangChain integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from i2i.schema import ConsensusLevel, ConsensusResult, Response, ConfidenceLevel
from i2i.task_classifier import ConsensusTaskCategory, ConsensusRecommendation


# Skip all tests if LangChain not installed
pytest.importorskip("langchain_core")

from i2i.integrations.langchain import (
    I2IConsensusLLM,
    LowConfidenceError,
    ConsensusMode,
    LANGCHAIN_AVAILABLE,
)
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, LLMResult


class MockBaseLLM(BaseLLM):
    """Mock LangChain LLM for testing."""

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        return f"Mock response for: {prompt[:50]}"

    async def _acall(self, prompt, stop=None, run_manager=None, **kwargs):
        return f"Mock async response for: {prompt[:50]}"

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        generations = [
            [Generation(text=f"Mock response for: {p[:50]}")]
            for p in prompts
        ]
        return LLMResult(generations=generations)

    @property
    def _llm_type(self):
        return "mock"


@pytest.fixture
def mock_base_llm():
    """Create a mock base LLM."""
    return MockBaseLLM()


@pytest.fixture
def mock_protocol():
    """Create a mock AICP protocol."""
    protocol = MagicMock()
    protocol.consensus_query = AsyncMock()
    return protocol


@pytest.fixture
def high_confidence_result():
    """Create a consensus result with high confidence."""
    return ConsensusResult(
        query="What is the capital of France?",
        models_queried=["model-a", "model-b", "model-c"],
        responses=[],
        consensus_level=ConsensusLevel.HIGH,
        consensus_answer="Paris is the capital of France.",
        confidence_calibration=0.95,
        consensus_appropriate=True,
        task_category="factual",
    )


@pytest.fixture
def low_confidence_result():
    """Create a consensus result with low confidence."""
    return ConsensusResult(
        query="Will it rain tomorrow?",
        models_queried=["model-a", "model-b"],
        responses=[],
        consensus_level=ConsensusLevel.LOW,
        consensus_answer="It might rain.",
        confidence_calibration=0.55,
        consensus_appropriate=True,
        task_category="uncertain",
    )


@pytest.fixture
def medium_confidence_result():
    """Create a consensus result with medium confidence."""
    return ConsensusResult(
        query="What is the best programming language?",
        models_queried=["model-a", "model-b"],
        responses=[],
        consensus_level=ConsensusLevel.MEDIUM,
        consensus_answer="It depends on the use case.",
        confidence_calibration=0.78,
        consensus_appropriate=True,
        task_category="commonsense",
    )


class TestI2IConsensusLLM:
    """Tests for the I2IConsensusLLM wrapper."""

    def test_init_with_defaults(self, mock_base_llm):
        """Test initialization with default parameters."""
        llm = I2IConsensusLLM(base_llm=mock_base_llm)

        assert llm.base_llm == mock_base_llm
        assert llm.min_confidence == 0.75
        assert llm.warn_confidence == 0.90
        assert llm.consensus_mode == ConsensusMode.AUTO
        assert llm.protocol is not None

    def test_init_with_custom_params(self, mock_base_llm, mock_protocol):
        """Test initialization with custom parameters."""
        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            consensus_models=["claude-3-sonnet", "gpt-4"],
            min_confidence=0.80,
            warn_confidence=0.95,
            consensus_mode=ConsensusMode.ALWAYS,
            protocol=mock_protocol,
        )

        assert llm.consensus_models == ["claude-3-sonnet", "gpt-4"]
        assert llm.min_confidence == 0.80
        assert llm.warn_confidence == 0.95
        assert llm.consensus_mode == ConsensusMode.ALWAYS
        assert llm.protocol == mock_protocol

    def test_llm_type(self, mock_base_llm):
        """Test _llm_type property."""
        llm = I2IConsensusLLM(base_llm=mock_base_llm)
        assert llm._llm_type == "i2i-consensus-mock"

    def test_identifying_params(self, mock_base_llm):
        """Test _identifying_params includes i2i config."""
        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            consensus_models=["model-a"],
            min_confidence=0.80,
        )

        params = llm._identifying_params
        assert params["i2i_consensus_models"] == ["model-a"]
        assert params["i2i_min_confidence"] == 0.80
        assert params["i2i_consensus_mode"] == "auto"


class TestConsensusMode:
    """Tests for different consensus modes."""

    def test_should_use_consensus_never_mode(self, mock_base_llm):
        """Test NEVER mode always returns False."""
        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            consensus_mode=ConsensusMode.NEVER,
        )

        should_use, rec = llm._should_use_consensus("What is 2+2?")
        assert should_use is False
        assert rec is None

    def test_should_use_consensus_always_mode(self, mock_base_llm):
        """Test ALWAYS mode always returns True."""
        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            consensus_mode=ConsensusMode.ALWAYS,
        )

        # Even for math questions (normally skipped)
        should_use, rec = llm._should_use_consensus("Calculate 15 * 23")
        assert should_use is True

    def test_should_use_consensus_auto_factual(self, mock_base_llm):
        """Test AUTO mode enables consensus for factual questions."""
        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            consensus_mode=ConsensusMode.AUTO,
        )

        should_use, rec = llm._should_use_consensus("What is the capital of France?")
        assert should_use is True
        assert rec is not None
        assert rec.task_category == ConsensusTaskCategory.FACTUAL

    def test_should_use_consensus_auto_math(self, mock_base_llm):
        """Test AUTO mode skips consensus for math questions."""
        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            consensus_mode=ConsensusMode.AUTO,
        )

        should_use, rec = llm._should_use_consensus("Calculate the derivative of x^3")
        assert should_use is False
        assert rec is not None
        assert rec.task_category == ConsensusTaskCategory.REASONING

    def test_should_use_consensus_auto_creative(self, mock_base_llm):
        """Test AUTO mode skips consensus for creative tasks."""
        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            consensus_mode=ConsensusMode.AUTO,
        )

        should_use, rec = llm._should_use_consensus("Write a poem about autumn")
        assert should_use is False
        assert rec is not None
        assert rec.task_category == ConsensusTaskCategory.CREATIVE


class TestLowConfidenceError:
    """Tests for LowConfidenceError exception."""

    def test_error_attributes(self, low_confidence_result):
        """Test LowConfidenceError contains expected attributes."""
        error = LowConfidenceError(low_confidence_result, threshold=0.75)

        assert error.result == low_confidence_result
        assert error.confidence == 0.55
        assert error.threshold == 0.75
        assert "0.55" in str(error)
        assert "0.75" in str(error)

    def test_error_message(self, low_confidence_result):
        """Test LowConfidenceError generates informative message."""
        error = LowConfidenceError(low_confidence_result, threshold=0.75)

        assert "Consensus confidence" in str(error)
        assert "below minimum threshold" in str(error)
        assert low_confidence_result.consensus_level.value in str(error)


class TestConfidenceThresholds:
    """Tests for confidence threshold checking."""

    @pytest.mark.asyncio
    async def test_high_confidence_passes(self, mock_base_llm, mock_protocol, high_confidence_result):
        """Test high confidence result passes threshold check."""
        mock_protocol.consensus_query.return_value = high_confidence_result

        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            min_confidence=0.75,
            consensus_mode=ConsensusMode.ALWAYS,
            protocol=mock_protocol,
        )

        # Should not raise
        result = await llm._acall("What is the capital of France?")
        assert result == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_low_confidence_raises(self, mock_base_llm, mock_protocol, low_confidence_result):
        """Test low confidence result raises LowConfidenceError."""
        mock_protocol.consensus_query.return_value = low_confidence_result

        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            min_confidence=0.75,
            consensus_mode=ConsensusMode.ALWAYS,
            protocol=mock_protocol,
        )

        with pytest.raises(LowConfidenceError) as exc_info:
            await llm._acall("Will it rain tomorrow?")

        assert exc_info.value.confidence == 0.55
        assert exc_info.value.threshold == 0.75

    @pytest.mark.asyncio
    async def test_warn_confidence_logs(self, mock_base_llm, mock_protocol, medium_confidence_result):
        """Test medium confidence logs warning but doesn't raise."""
        mock_protocol.consensus_query.return_value = medium_confidence_result

        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            min_confidence=0.70,  # Below 0.78, so passes
            warn_confidence=0.85,  # Above 0.78, so warns
            consensus_mode=ConsensusMode.ALWAYS,
            protocol=mock_protocol,
        )

        with patch("logging.Logger.warning") as mock_warn:
            result = await llm._acall("What is the best programming language?")

        assert result == "It depends on the use case."


class TestPassthrough:
    """Tests for passthrough to base LLM when consensus skipped."""

    @pytest.mark.asyncio
    async def test_passthrough_never_mode(self, mock_base_llm, mock_protocol):
        """Test NEVER mode passes through to base LLM."""
        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            consensus_mode=ConsensusMode.NEVER,
            protocol=mock_protocol,
        )

        result = await llm._acall("Any question here")

        # Should not call consensus
        mock_protocol.consensus_query.assert_not_called()
        assert "Mock async response" in result

    @pytest.mark.asyncio
    async def test_passthrough_task_aware(self, mock_base_llm, mock_protocol):
        """Test AUTO mode passes through for inappropriate tasks."""
        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            consensus_mode=ConsensusMode.AUTO,
            protocol=mock_protocol,
        )

        # Math question - should skip consensus
        result = await llm._acall("Calculate 15 * 23 step by step")

        # Should not call consensus
        mock_protocol.consensus_query.assert_not_called()
        assert "Mock async response" in result


class TestResponseMetadata:
    """Tests for response metadata building."""

    def test_metadata_with_consensus(self, mock_base_llm, high_confidence_result):
        """Test metadata includes consensus information."""
        llm = I2IConsensusLLM(base_llm=mock_base_llm)

        metadata = llm._build_response_metadata(high_confidence_result, None)

        assert metadata["consensus_level"] == "high"
        assert metadata["confidence_calibration"] == 0.95
        assert metadata["task_category"] == "factual"
        assert metadata["consensus_appropriate"] is True
        assert "model-a" in metadata["models_queried"]

    def test_metadata_when_skipped(self, mock_base_llm):
        """Test metadata when consensus is skipped."""
        llm = I2IConsensusLLM(base_llm=mock_base_llm)

        rec = ConsensusRecommendation(
            should_use_consensus=False,
            task_category=ConsensusTaskCategory.REASONING,
            confidence=0.9,
            reason="Math question - consensus degrades accuracy",
            suggested_approach="Use single model with chain-of-thought",
        )

        metadata = llm._build_response_metadata(None, rec, consensus_skipped=True)

        assert metadata["consensus_skipped"] is True
        assert metadata["task_category"] == "reasoning"
        assert "Math" in metadata["skip_reason"]


class TestWithConfig:
    """Tests for with_config method."""

    def test_with_config_creates_copy(self, mock_base_llm, mock_protocol):
        """Test with_config creates new instance with updated values."""
        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            min_confidence=0.75,
            protocol=mock_protocol,
        )

        new_llm = llm.with_config(min_confidence=0.90)

        assert new_llm is not llm
        assert new_llm.min_confidence == 0.90
        assert llm.min_confidence == 0.75  # Original unchanged
        assert new_llm.protocol == mock_protocol  # Shared

    def test_always_consensus_helper(self, mock_base_llm):
        """Test always_consensus() convenience method."""
        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            consensus_mode=ConsensusMode.AUTO,
        )

        always_llm = llm.always_consensus()
        assert always_llm.consensus_mode == ConsensusMode.ALWAYS

    def test_never_consensus_helper(self, mock_base_llm):
        """Test never_consensus() convenience method."""
        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            consensus_mode=ConsensusMode.AUTO,
        )

        never_llm = llm.never_consensus()
        assert never_llm.consensus_mode == ConsensusMode.NEVER


class TestGenerate:
    """Tests for _generate and _agenerate methods."""

    @pytest.mark.asyncio
    async def test_agenerate_multiple_prompts(self, mock_base_llm, mock_protocol, high_confidence_result):
        """Test _agenerate handles multiple prompts."""
        mock_protocol.consensus_query.return_value = high_confidence_result

        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            consensus_mode=ConsensusMode.ALWAYS,
            protocol=mock_protocol,
        )

        result = await llm._agenerate(["Question 1?", "Question 2?"])

        assert len(result.generations) == 2
        assert mock_protocol.consensus_query.call_count == 2

    @pytest.mark.asyncio
    async def test_agenerate_mixed_tasks(self, mock_base_llm, mock_protocol, high_confidence_result):
        """Test _agenerate with mix of consensus/passthrough tasks."""
        mock_protocol.consensus_query.return_value = high_confidence_result

        llm = I2IConsensusLLM(
            base_llm=mock_base_llm,
            consensus_mode=ConsensusMode.AUTO,
            protocol=mock_protocol,
        )

        result = await llm._agenerate([
            "What is the capital of France?",  # Factual - consensus
            "Calculate 2 + 2",  # Math - passthrough
        ])

        assert len(result.generations) == 2
        # Only the factual question should trigger consensus
        assert mock_protocol.consensus_query.call_count == 1

        # Check metadata
        factual_meta = result.generations[0][0].generation_info
        assert factual_meta.get("consensus_level") == "high"

        math_meta = result.generations[1][0].generation_info
        assert math_meta.get("consensus_skipped") is True
