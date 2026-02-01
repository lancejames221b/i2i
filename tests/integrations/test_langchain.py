"""
Unit and integration tests for LangChain integration.

Tests cover:
- I2IVerifier initialization and configuration
- Task detection (factual vs math vs creative)
- Confidence thresholds
- Error handling for API failures
- LCEL compatibility
- Async invoke
- Metadata passthrough

Mocking strategy:
- i2i.AICP is mocked for unit tests
- LangChain's FakeLLM is used for integration tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

# Check if LangChain is available
try:
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import LLMResult, Generation
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from i2i.schema import ConsensusLevel, ConsensusResult, Response, ConfidenceLevel
from i2i.task_classifier import ConsensusTaskCategory


# Apply markers to all tests
pytestmark = [
    pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed"),
    pytest.mark.asyncio,
]


# ==================== Fixtures ====================


@pytest.fixture
def mock_consensus_result_high():
    """Create a high consensus result for testing."""
    return ConsensusResult(
        query="Test query",
        models_queried=["gpt-4", "claude-3", "gemini-pro"],
        responses=[
            Response(
                message_id="msg-1",
                model="gpt-4",
                content="Paris is the capital of France.",
                confidence=ConfidenceLevel.HIGH,
            ),
            Response(
                message_id="msg-2",
                model="claude-3",
                content="Paris is the capital of France.",
                confidence=ConfidenceLevel.HIGH,
            ),
        ],
        consensus_level=ConsensusLevel.HIGH,
        consensus_answer="Paris is the capital of France.",
        consensus_appropriate=True,
        confidence_calibration=0.95,
        task_category="factual",
    )


@pytest.fixture
def mock_consensus_result_low():
    """Create a low consensus result for testing."""
    return ConsensusResult(
        query="Test query",
        models_queried=["gpt-4", "claude-3"],
        responses=[
            Response(
                message_id="msg-1",
                model="gpt-4",
                content="AI will achieve AGI by 2030.",
                confidence=ConfidenceLevel.MEDIUM,
            ),
            Response(
                message_id="msg-2",
                model="claude-3",
                content="AGI is decades away.",
                confidence=ConfidenceLevel.MEDIUM,
            ),
        ],
        consensus_level=ConsensusLevel.LOW,
        consensus_answer=None,
        divergences=[{"type": "opinion"}],
        consensus_appropriate=False,
        confidence_calibration=0.3,
        task_category="uncertain",
    )


@pytest.fixture
def mock_protocol(mock_consensus_result_high):
    """Create a mock AICP protocol."""
    protocol = MagicMock()
    protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)
    protocol.verify_claim = AsyncMock()
    protocol.classify_question = AsyncMock()
    return protocol


# ==================== Unit Tests: I2IVerifier Initialization ====================


class TestI2IVerifierInitialization:
    """Test I2IVerifier initialization with various configs."""

    def test_default_initialization(self):
        """I2IVerifier should initialize with defaults."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier()

        assert verifier.config.min_consensus_level == ConsensusLevel.MEDIUM
        assert verifier.config.confidence_threshold == 0.7
        assert verifier.config.task_aware is True
        assert verifier.config.raise_on_failure is False

    def test_custom_models(self):
        """I2IVerifier should accept custom model list."""
        from i2i.integrations.langchain import I2IVerifier

        models = ["gpt-4", "claude-3-opus"]
        verifier = I2IVerifier(models=models)

        assert verifier.config.models == models

    def test_high_consensus_threshold(self):
        """I2IVerifier should accept HIGH consensus requirement."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(min_consensus_level=ConsensusLevel.HIGH)

        assert verifier.config.min_consensus_level == ConsensusLevel.HIGH

    def test_custom_confidence_threshold(self):
        """I2IVerifier should accept custom confidence threshold."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(confidence_threshold=0.9)

        assert verifier.config.confidence_threshold == 0.9

    def test_task_aware_disabled(self):
        """I2IVerifier should allow disabling task awareness."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(task_aware=False)

        assert verifier.config.task_aware is False

    def test_explicit_task_category(self):
        """I2IVerifier should accept explicit task category."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(task_category="factual")

        assert verifier.config.task_category == "factual"

    def test_raise_on_failure_enabled(self):
        """I2IVerifier should support raise_on_failure mode."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(raise_on_failure=True)

        assert verifier.config.raise_on_failure is True

    def test_statistical_mode_config(self):
        """I2IVerifier should support statistical mode configuration."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(
            statistical_mode=True,
            n_runs=5,
            temperature=0.8
        )

        assert verifier.config.statistical_mode is True
        assert verifier.config.n_runs == 5
        assert verifier.config.temperature == 0.8

    def test_accepts_protocol_instance(self, mock_protocol):
        """I2IVerifier should accept pre-configured protocol."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier(protocol=mock_protocol)

        assert verifier.protocol == mock_protocol


# ==================== Unit Tests: Task Detection ====================


class TestTaskDetection:
    """Test task classification integration."""

    async def test_detects_factual_task(self, mock_protocol, mock_consensus_result_high):
        """Verifier should detect factual tasks."""
        from i2i.integrations.langchain import I2IVerifier

        mock_consensus_result_high.task_category = "factual"
        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(task_aware=True, protocol=mock_protocol)
        result = await verifier.ainvoke("Paris is the capital of France.")

        assert result.task_category == "factual"

    async def test_detects_reasoning_task(self, mock_protocol):
        """Verifier should detect reasoning/math tasks."""
        from i2i.integrations.langchain import I2IVerifier

        math_result = ConsensusResult(
            query="Calculate 2+2",
            models_queried=["gpt-4"],
            responses=[Response(
                message_id="m1",
                model="gpt-4",
                content="4",
                confidence=ConfidenceLevel.HIGH
            )],
            consensus_level=ConsensusLevel.HIGH,
            consensus_answer="4",
            task_category="reasoning",
            consensus_appropriate=False,  # Consensus not appropriate for math
            confidence_calibration=0.75,
        )
        mock_protocol.consensus_query = AsyncMock(return_value=math_result)

        verifier = I2IVerifier(task_aware=True, protocol=mock_protocol)
        result = await verifier.ainvoke("The answer to 2+2 is 4.")

        assert result.task_category == "reasoning"
        assert result.consensus_appropriate is False

    async def test_explicit_category_overrides_detection(self, mock_protocol, mock_consensus_result_high):
        """Explicit task_category should override auto-detection."""
        from i2i.integrations.langchain import I2IVerifier

        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(
            task_category="creative",
            task_aware=True,
            protocol=mock_protocol
        )
        result = await verifier.ainvoke("Test content")

        # Check that the explicit category was passed to consensus_query
        call_kwargs = mock_protocol.consensus_query.call_args.kwargs
        assert call_kwargs.get("task_category") == "creative"

    async def test_task_aware_disabled_skips_detection(self, mock_protocol):
        """With task_aware=False, should skip task classification."""
        from i2i.integrations.langchain import I2IVerifier

        result = ConsensusResult(
            query="Test",
            models_queried=["gpt-4"],
            responses=[],
            consensus_level=ConsensusLevel.MEDIUM,
            consensus_answer="Test answer",
            # No task-aware fields
        )
        mock_protocol.consensus_query = AsyncMock(return_value=result)

        verifier = I2IVerifier(task_aware=False, protocol=mock_protocol)
        output = await verifier.ainvoke("Test input")

        call_kwargs = mock_protocol.consensus_query.call_args.kwargs
        assert call_kwargs.get("task_aware") is False


# ==================== Unit Tests: Confidence Thresholds ====================


class TestConfidenceThresholds:
    """Test confidence threshold enforcement."""

    async def test_high_confidence_passes(self, mock_protocol, mock_consensus_result_high):
        """High confidence should pass verification."""
        from i2i.integrations.langchain import I2IVerifier

        mock_consensus_result_high.confidence_calibration = 0.95
        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(confidence_threshold=0.7, protocol=mock_protocol)
        result = await verifier.ainvoke("Test content")

        assert result.verified is True
        assert result.confidence == 0.95

    async def test_low_confidence_fails(self, mock_protocol, mock_consensus_result_low):
        """Low confidence should fail verification."""
        from i2i.integrations.langchain import I2IVerifier

        mock_consensus_result_low.confidence_calibration = 0.3
        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_low)

        verifier = I2IVerifier(confidence_threshold=0.7, protocol=mock_protocol)
        result = await verifier.ainvoke("Test content")

        assert result.verified is False
        assert result.confidence == 0.3

    async def test_exactly_at_threshold_passes(self, mock_protocol, mock_consensus_result_high):
        """Confidence exactly at threshold should pass."""
        from i2i.integrations.langchain import I2IVerifier

        mock_consensus_result_high.confidence_calibration = 0.7
        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(confidence_threshold=0.7, protocol=mock_protocol)
        result = await verifier.ainvoke("Test content")

        assert result.verified is True

    async def test_just_below_threshold_fails(self, mock_protocol, mock_consensus_result_high):
        """Confidence just below threshold should fail."""
        from i2i.integrations.langchain import I2IVerifier

        mock_consensus_result_high.confidence_calibration = 0.699
        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(confidence_threshold=0.7, protocol=mock_protocol)
        result = await verifier.ainvoke("Test content")

        assert result.verified is False

    async def test_strict_threshold(self, mock_protocol, mock_consensus_result_high):
        """Strict 0.95 threshold should require very high confidence."""
        from i2i.integrations.langchain import I2IVerifier

        mock_consensus_result_high.confidence_calibration = 0.90
        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(confidence_threshold=0.95, protocol=mock_protocol)
        result = await verifier.ainvoke("Test content")

        assert result.verified is False


# ==================== Unit Tests: Consensus Level Thresholds ====================


class TestConsensusLevelThresholds:
    """Test consensus level enforcement."""

    @pytest.mark.parametrize("level,expected", [
        (ConsensusLevel.HIGH, True),
        (ConsensusLevel.MEDIUM, True),
        (ConsensusLevel.LOW, False),
        (ConsensusLevel.NONE, False),
        (ConsensusLevel.CONTRADICTORY, False),
    ])
    async def test_consensus_level_thresholds(self, mock_protocol, mock_consensus_result_high, level, expected):
        """Verify consensus level thresholds work correctly."""
        from i2i.integrations.langchain import I2IVerifier

        mock_consensus_result_high.consensus_level = level
        mock_consensus_result_high.confidence_calibration = 0.95  # High confidence
        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(
            min_consensus_level=ConsensusLevel.MEDIUM,
            confidence_threshold=0.5,  # Low threshold to isolate level check
            protocol=mock_protocol
        )
        result = await verifier.ainvoke("Test content")

        assert result.verified == expected
        assert result.consensus_level == level.value


# ==================== Unit Tests: Error Handling ====================


class TestErrorHandling:
    """Test error handling for API failures."""

    async def test_fallback_on_api_error(self, mock_protocol):
        """Should return unverified on API error with fallback enabled."""
        from i2i.integrations.langchain import I2IVerifier

        mock_protocol.consensus_query = AsyncMock(
            side_effect=Exception("API connection failed")
        )

        verifier = I2IVerifier(
            fallback_on_error=True,
            protocol=mock_protocol
        )
        result = await verifier.ainvoke("Test content")

        assert result.verified is False
        assert result.content == "Test content"
        assert "error" in result.verification_details

    async def test_raises_on_api_error_when_disabled(self, mock_protocol):
        """Should raise exception when fallback is disabled."""
        from i2i.integrations.langchain import I2IVerifier

        mock_protocol.consensus_query = AsyncMock(
            side_effect=Exception("API connection failed")
        )

        verifier = I2IVerifier(
            fallback_on_error=False,
            protocol=mock_protocol
        )

        with pytest.raises(Exception, match="API connection failed"):
            await verifier.ainvoke("Test content")

    async def test_raises_on_failure_when_enabled(self, mock_protocol, mock_consensus_result_low):
        """Should raise VerificationError when raise_on_failure is True."""
        from i2i.integrations.langchain import I2IVerifier, VerificationError

        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_low)

        verifier = I2IVerifier(
            raise_on_failure=True,
            confidence_threshold=0.9,  # Will fail
            protocol=mock_protocol
        )

        with pytest.raises(VerificationError):
            await verifier.ainvoke("Test content")

    async def test_does_not_raise_when_disabled(self, mock_protocol, mock_consensus_result_low):
        """Should not raise when raise_on_failure is False."""
        from i2i.integrations.langchain import I2IVerifier

        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_low)

        verifier = I2IVerifier(
            raise_on_failure=False,
            confidence_threshold=0.9,  # Will fail
            protocol=mock_protocol
        )

        result = await verifier.ainvoke("Test content")
        assert result.verified is False  # Failed but no exception


# ==================== Integration Tests: LCEL Compatibility ====================


class TestLCELCompatibility:
    """Test LangChain Expression Language compatibility."""

    async def test_accepts_string_input(self, mock_protocol, mock_consensus_result_high):
        """Verifier should accept string input."""
        from i2i.integrations.langchain import I2IVerifier

        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(protocol=mock_protocol)
        result = await verifier.ainvoke("Plain string input")

        assert result.content == "Plain string input"

    async def test_accepts_ai_message(self, mock_protocol, mock_consensus_result_high):
        """Verifier should accept AIMessage input."""
        from i2i.integrations.langchain import I2IVerifier

        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(protocol=mock_protocol)
        message = AIMessage(content="AI message content")
        result = await verifier.ainvoke(message)

        assert result.content == "AI message content"

    async def test_accepts_human_message(self, mock_protocol, mock_consensus_result_high):
        """Verifier should accept HumanMessage input."""
        from i2i.integrations.langchain import I2IVerifier

        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(protocol=mock_protocol)
        message = HumanMessage(content="Human message content")
        result = await verifier.ainvoke(message)

        assert result.content == "Human message content"

    def test_input_type_is_correct(self):
        """InputType should be Union[str, BaseMessage]."""
        from i2i.integrations.langchain import I2IVerifier

        verifier = I2IVerifier()
        # Just check it doesn't raise
        _ = verifier.InputType

    def test_output_type_is_verification_output(self):
        """OutputType should be VerificationOutput."""
        from i2i.integrations.langchain import I2IVerifier, VerificationOutput

        verifier = I2IVerifier()
        assert verifier.OutputType == VerificationOutput


# ==================== Integration Tests: Async/Sync ====================


class TestAsyncSync:
    """Test async and sync invoke methods."""

    async def test_ainvoke_works(self, mock_protocol, mock_consensus_result_high):
        """ainvoke should work correctly."""
        from i2i.integrations.langchain import I2IVerifier

        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(protocol=mock_protocol)
        result = await verifier.ainvoke("Test input")

        assert result is not None
        assert result.content == "Test input"

    def test_invoke_works_sync(self, mock_protocol, mock_consensus_result_high):
        """invoke should work in sync context."""
        from i2i.integrations.langchain import I2IVerifier

        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(protocol=mock_protocol)

        # Run in a new event loop
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(verifier.ainvoke("Test input"))
            assert result.content == "Test input"
        finally:
            loop.close()


# ==================== Integration Tests: Metadata Passthrough ====================


class TestMetadataPassthrough:
    """Test metadata is properly passed through."""

    async def test_includes_verification_metadata(self, mock_protocol, mock_consensus_result_high):
        """Should include verification details in output."""
        from i2i.integrations.langchain import I2IVerifier

        mock_consensus_result_high.metadata = {"custom": "data"}
        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(
            include_verification_metadata=True,
            protocol=mock_protocol
        )
        result = await verifier.ainvoke("Test")

        assert result.verification_details is not None
        assert "models_queried" in result.verification_details
        assert "consensus_answer" in result.verification_details

    async def test_excludes_metadata_when_disabled(self, mock_protocol, mock_consensus_result_high):
        """Should exclude verification details when disabled."""
        from i2i.integrations.langchain import I2IVerifier

        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(
            include_verification_metadata=False,
            protocol=mock_protocol
        )
        result = await verifier.ainvoke("Test")

        # Details should be None when disabled
        assert result.verification_details is None

    async def test_preserves_original_content(self, mock_protocol, mock_consensus_result_high):
        """Should preserve original content in output."""
        from i2i.integrations.langchain import I2IVerifier

        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        verifier = I2IVerifier(protocol=mock_protocol)
        result = await verifier.ainvoke("Original content here")

        assert result.content == "Original content here"


# ==================== Integration Tests: Callback Handler ====================


class TestVerificationCallback:
    """Test I2IVerificationCallback."""

    def test_callback_initialization(self, mock_protocol):
        """Callback should initialize correctly."""
        from i2i.integrations.langchain import I2IVerificationCallback

        callback = I2IVerificationCallback(
            models=["gpt-4"],
            min_consensus_level=ConsensusLevel.HIGH,
            on_verification_failure="warn",
            protocol=mock_protocol
        )

        assert callback.verifier is not None
        assert callback.on_failure == "warn"

    def test_get_last_verification_initially_none(self, mock_protocol):
        """get_last_verification should return None initially."""
        from i2i.integrations.langchain import I2IVerificationCallback

        callback = I2IVerificationCallback(protocol=mock_protocol)

        assert callback.get_last_verification() is None

    def test_get_verification_history_initially_empty(self, mock_protocol):
        """get_verification_history should return empty list initially."""
        from i2i.integrations.langchain import I2IVerificationCallback

        callback = I2IVerificationCallback(protocol=mock_protocol)

        assert callback.get_verification_history() == []

    def test_clear_history(self, mock_protocol):
        """clear_history should reset state."""
        from i2i.integrations.langchain import I2IVerificationCallback

        callback = I2IVerificationCallback(protocol=mock_protocol)
        # Manually set some history
        callback._verification_history = ["item1", "item2"]
        callback._last_verification = "last"

        callback.clear_history()

        assert callback.get_verification_history() == []
        assert callback.get_last_verification() is None


# ==================== Integration Tests: Verified Chain ====================


class TestVerifiedChain:
    """Test I2IVerifiedChain wrapper."""

    def test_chain_initialization(self, mock_protocol):
        """VerifiedChain should initialize with a chain."""
        from i2i.integrations.langchain import I2IVerifiedChain

        # Mock chain
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value="Chain output")

        wrapped = I2IVerifiedChain(
            chain=mock_chain,
            min_consensus_level=ConsensusLevel.HIGH,
            protocol=mock_protocol
        )

        assert wrapped.chain == mock_chain
        assert wrapped.verifier.config.min_consensus_level == ConsensusLevel.HIGH

    async def test_chain_executes_and_verifies(self, mock_protocol, mock_consensus_result_high):
        """VerifiedChain should execute chain then verify."""
        from i2i.integrations.langchain import I2IVerifiedChain

        mock_protocol.consensus_query = AsyncMock(return_value=mock_consensus_result_high)

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=AIMessage(content="Chain result"))

        wrapped = I2IVerifiedChain(chain=mock_chain, protocol=mock_protocol)
        result = await wrapped.ainvoke({"input": "test"})

        # Chain should be called
        mock_chain.ainvoke.assert_called_once()
        # Result should be verified
        assert result.content == "Chain result"
        assert result.verified is True


# ==================== Integration Tests: create_verified_chain Helper ====================


class TestCreateVerifiedChain:
    """Test create_verified_chain helper function."""

    def test_creates_chain_with_verifier(self, mock_protocol):
        """create_verified_chain should return a Runnable."""
        from i2i.integrations.langchain import create_verified_chain

        mock_chain = MagicMock()
        mock_chain.__or__ = MagicMock(return_value=MagicMock())

        result = create_verified_chain(
            chain=mock_chain,
            min_consensus_level=ConsensusLevel.HIGH,
            protocol=mock_protocol
        )

        # Should use | operator to chain
        mock_chain.__or__.assert_called_once()


# ==================== Unit Tests: VerificationOutput ====================


class TestVerificationOutput:
    """Test VerificationOutput model."""

    def test_basic_output(self):
        """VerificationOutput should hold basic fields."""
        from i2i.integrations.langchain import VerificationOutput

        output = VerificationOutput(
            content="Test content",
            verified=True,
            consensus_level="high",
            confidence=0.95
        )

        assert output.content == "Test content"
        assert output.verified is True
        assert output.consensus_level == "high"
        assert output.confidence == 0.95

    def test_output_with_all_fields(self):
        """VerificationOutput should support all fields."""
        from i2i.integrations.langchain import VerificationOutput

        output = VerificationOutput(
            content="Test",
            verified=True,
            consensus_level="medium",
            confidence=0.75,
            task_category="factual",
            consensus_appropriate=True,
            verification_details={"key": "value"},
            original_content="Original"
        )

        assert output.task_category == "factual"
        assert output.consensus_appropriate is True
        assert output.verification_details == {"key": "value"}
        assert output.original_content == "Original"


# ==================== Unit Tests: VerificationConfig ====================


class TestVerificationConfig:
    """Test VerificationConfig model."""

    def test_default_config(self):
        """VerificationConfig should have sensible defaults."""
        from i2i.integrations.langchain import VerificationConfig

        config = VerificationConfig()

        assert config.models is None
        assert config.min_consensus_level == ConsensusLevel.MEDIUM
        assert config.confidence_threshold == 0.7
        assert config.task_aware is True
        assert config.raise_on_failure is False
        assert config.fallback_on_error is True

    def test_custom_config(self):
        """VerificationConfig should accept custom values."""
        from i2i.integrations.langchain import VerificationConfig

        config = VerificationConfig(
            models=["gpt-4"],
            min_consensus_level=ConsensusLevel.HIGH,
            confidence_threshold=0.9,
            task_aware=False,
            raise_on_failure=True
        )

        assert config.models == ["gpt-4"]
        assert config.min_consensus_level == ConsensusLevel.HIGH
        assert config.confidence_threshold == 0.9
        assert config.task_aware is False
        assert config.raise_on_failure is True
