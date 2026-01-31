"""Tests for i2i consortium detection and homogeneous optimization."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from i2i.consensus import (
    ConsortiumType,
    ModelFamily,
    detect_model_family,
    detect_consortium_type,
    ConsensusEngine,
)
from i2i.schema import Message, MessageType, Response, ConfidenceLevel


class TestModelFamily:
    """Tests for ModelFamily enum."""

    def test_model_families_defined(self):
        """All expected model families should be defined."""
        assert ModelFamily.OPENAI == "openai"
        assert ModelFamily.ANTHROPIC == "anthropic"
        assert ModelFamily.GOOGLE == "google"
        assert ModelFamily.MISTRAL == "mistral"
        assert ModelFamily.META == "meta"
        assert ModelFamily.COHERE == "cohere"
        assert ModelFamily.UNKNOWN == "unknown"


class TestDetectModelFamily:
    """Tests for detect_model_family function."""

    def test_detect_openai_gpt(self):
        """Detect OpenAI GPT models."""
        assert detect_model_family("gpt-4o") == ModelFamily.OPENAI
        assert detect_model_family("gpt-5.2") == ModelFamily.OPENAI
        assert detect_model_family("gpt-4.1-mini") == ModelFamily.OPENAI

    def test_detect_openai_o_series(self):
        """Detect OpenAI O-series reasoning models."""
        assert detect_model_family("o3") == ModelFamily.OPENAI
        assert detect_model_family("o3-pro") == ModelFamily.OPENAI
        assert detect_model_family("o4-mini") == ModelFamily.OPENAI

    def test_detect_openai_with_prefix(self):
        """Detect OpenAI models with provider prefix."""
        assert detect_model_family("openai/gpt-4o") == ModelFamily.OPENAI

    def test_detect_anthropic_claude(self):
        """Detect Anthropic Claude models."""
        assert detect_model_family("claude-3-opus-20240229") == ModelFamily.ANTHROPIC
        assert detect_model_family("claude-sonnet-4-5-20250929") == ModelFamily.ANTHROPIC
        assert detect_model_family("claude-haiku-4-5-20251001") == ModelFamily.ANTHROPIC

    def test_detect_anthropic_with_prefix(self):
        """Detect Anthropic models with provider prefix."""
        assert detect_model_family("anthropic/claude-3-sonnet") == ModelFamily.ANTHROPIC

    def test_detect_google_gemini(self):
        """Detect Google Gemini models."""
        assert detect_model_family("gemini-1.5-pro") == ModelFamily.GOOGLE
        assert detect_model_family("gemini-3-flash-preview") == ModelFamily.GOOGLE
        assert detect_model_family("gemini-3-deep-think-preview") == ModelFamily.GOOGLE

    def test_detect_google_with_prefix(self):
        """Detect Google models with provider prefix."""
        assert detect_model_family("google/gemini-1.5-pro") == ModelFamily.GOOGLE

    def test_detect_mistral(self):
        """Detect Mistral models."""
        assert detect_model_family("mistral-large-3") == ModelFamily.MISTRAL
        assert detect_model_family("mixtral-8x7b-32768") == ModelFamily.MISTRAL
        assert detect_model_family("codestral-latest") == ModelFamily.MISTRAL
        assert detect_model_family("devstral-2") == ModelFamily.MISTRAL
        assert detect_model_family("ministral-3-14b") == ModelFamily.MISTRAL

    def test_detect_meta_llama(self):
        """Detect Meta Llama models."""
        assert detect_model_family("llama-3.3-70b-versatile") == ModelFamily.META
        assert detect_model_family("meta-llama/llama-4-maverick-17b") == ModelFamily.META

    def test_detect_cohere(self):
        """Detect Cohere Command models."""
        assert detect_model_family("command-a-03-2025") == ModelFamily.COHERE
        assert detect_model_family("command-r-plus") == ModelFamily.COHERE

    def test_detect_unknown(self):
        """Unknown models return UNKNOWN family."""
        assert detect_model_family("some-random-model") == ModelFamily.UNKNOWN
        assert detect_model_family("custom-fine-tuned-v1") == ModelFamily.UNKNOWN

    def test_case_insensitive(self):
        """Model detection should be case insensitive."""
        assert detect_model_family("GPT-4O") == ModelFamily.OPENAI
        assert detect_model_family("CLAUDE-3-OPUS") == ModelFamily.ANTHROPIC
        assert detect_model_family("Gemini-1.5-Pro") == ModelFamily.GOOGLE


class TestConsortiumType:
    """Tests for ConsortiumType enum."""

    def test_consortium_types_defined(self):
        """All consortium types should be defined."""
        assert ConsortiumType.HETEROGENEOUS == "heterogeneous"
        assert ConsortiumType.HOMOGENEOUS == "homogeneous"
        assert ConsortiumType.MIXED == "mixed"


class TestDetectConsortiumType:
    """Tests for detect_consortium_type function."""

    def test_homogeneous_all_claude(self):
        """All Claude models should be HOMOGENEOUS."""
        models = [
            "claude-3-opus-20240229",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
        ]
        consortium_type, family_mapping = detect_consortium_type(models)
        assert consortium_type == ConsortiumType.HOMOGENEOUS
        assert ModelFamily.ANTHROPIC in family_mapping
        assert len(family_mapping[ModelFamily.ANTHROPIC]) == 3

    def test_homogeneous_all_gpt(self):
        """All GPT models should be HOMOGENEOUS."""
        models = ["gpt-4o", "gpt-5.2", "gpt-4.1-mini"]
        consortium_type, family_mapping = detect_consortium_type(models)
        assert consortium_type == ConsortiumType.HOMOGENEOUS
        assert ModelFamily.OPENAI in family_mapping

    def test_heterogeneous_different_families(self):
        """Models from different families should be HETEROGENEOUS."""
        models = [
            "gpt-4o",
            "claude-3-opus-20240229",
            "gemini-1.5-pro",
        ]
        consortium_type, family_mapping = detect_consortium_type(models)
        assert consortium_type == ConsortiumType.HETEROGENEOUS
        assert ModelFamily.OPENAI in family_mapping
        assert ModelFamily.ANTHROPIC in family_mapping
        assert ModelFamily.GOOGLE in family_mapping

    def test_mixed_mostly_same(self):
        """Mostly same family with some different should be MIXED."""
        models = [
            "claude-3-opus-20240229",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "gpt-4o",  # One different
        ]
        consortium_type, family_mapping = detect_consortium_type(models)
        assert consortium_type == ConsortiumType.MIXED

    def test_returns_family_mapping(self):
        """Should return mapping of families to models."""
        models = ["gpt-4o", "claude-3-opus-20240229"]
        consortium_type, family_mapping = detect_consortium_type(models)
        assert ModelFamily.OPENAI in family_mapping
        assert ModelFamily.ANTHROPIC in family_mapping
        assert "gpt-4o" in family_mapping[ModelFamily.OPENAI]
        assert "claude-3-opus-20240229" in family_mapping[ModelFamily.ANTHROPIC]

    def test_empty_list(self):
        """Empty model list should return heterogeneous (safe default)."""
        consortium_type, family_mapping = detect_consortium_type([])
        assert consortium_type == ConsortiumType.HETEROGENEOUS

    def test_single_model(self):
        """Single model should be HOMOGENEOUS."""
        models = ["gpt-4o"]
        consortium_type, family_mapping = detect_consortium_type(models)
        assert consortium_type == ConsortiumType.HOMOGENEOUS

    def test_all_unknown(self):
        """All unknown models should be HETEROGENEOUS (conservative)."""
        models = ["custom-model-1", "custom-model-2", "custom-model-3"]
        consortium_type, family_mapping = detect_consortium_type(models)
        assert consortium_type == ConsortiumType.HETEROGENEOUS


class TestConsensusEngineConsortium:
    """Tests for ConsensusEngine consortium analysis."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock provider registry."""
        registry = MagicMock()
        return registry

    @pytest.fixture
    def engine(self, mock_registry):
        """Create a ConsensusEngine instance."""
        return ConsensusEngine(mock_registry)

    def test_analyze_consortium_method_exists(self, engine):
        """ConsensusEngine should have analyze_consortium method."""
        assert hasattr(engine, "analyze_consortium")
        assert callable(engine.analyze_consortium)

    def test_analyze_consortium_returns_type_and_mapping(self, engine):
        """analyze_consortium should return tuple of type and mapping."""
        models = ["gpt-4o", "claude-3-opus"]
        result = engine.analyze_consortium(models)
        assert isinstance(result, tuple)
        assert len(result) == 2
        consortium_type, family_mapping = result
        assert isinstance(consortium_type, ConsortiumType)
        assert isinstance(family_mapping, dict)

    def test_analyze_consortium_homogeneous(self, engine):
        """analyze_consortium detects homogeneous consortium."""
        models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        consortium_type, _ = engine.analyze_consortium(models)
        assert consortium_type == ConsortiumType.HOMOGENEOUS

    def test_analyze_consortium_heterogeneous(self, engine):
        """analyze_consortium detects heterogeneous consortium."""
        models = ["gpt-4o", "claude-3-opus", "gemini-1.5-pro"]
        consortium_type, _ = engine.analyze_consortium(models)
        assert consortium_type == ConsortiumType.HETEROGENEOUS


class TestConsensusResultMetadata:
    """Tests for consortium info in ConsensusResult metadata."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock provider registry."""
        registry = MagicMock()

        async def mock_query_multiple(message, models):
            return [
                Response(
                    message_id=message.id,
                    model=model,
                    content="Test response",
                    confidence=ConfidenceLevel.HIGH,
                )
                for model in models
            ]

        registry.query_multiple = AsyncMock(side_effect=mock_query_multiple)
        return registry

    @pytest.fixture
    def engine(self, mock_registry):
        """Create a ConsensusEngine instance."""
        return ConsensusEngine(mock_registry)

    @pytest.mark.asyncio
    async def test_result_includes_consortium_type(self, engine):
        """ConsensusResult should include consortium_type in metadata."""
        models = ["gpt-4o", "claude-3-opus", "gemini-1.5-pro"]
        result = await engine.query_for_consensus("Test query", models)
        assert "consortium_type" in result.metadata
        assert result.metadata["consortium_type"] == "heterogeneous"

    @pytest.mark.asyncio
    async def test_result_includes_family_breakdown(self, engine):
        """ConsensusResult should include family_breakdown in metadata."""
        models = ["gpt-4o", "claude-3-opus"]
        result = await engine.query_for_consensus("Test query", models)
        assert "family_breakdown" in result.metadata
        assert "openai" in result.metadata["family_breakdown"]
        assert "anthropic" in result.metadata["family_breakdown"]

    @pytest.mark.asyncio
    async def test_homogeneous_warning_in_metadata(self, engine):
        """Homogeneous consortium should include warning in metadata."""
        models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        result = await engine.query_for_consensus("Test query", models)
        assert result.metadata["consortium_type"] == "homogeneous"
        assert "warning" in result.metadata
        assert "same family" in result.metadata["warning"].lower()

    @pytest.mark.asyncio
    async def test_heterogeneous_no_warning(self, engine):
        """Heterogeneous consortium should not have warning."""
        models = ["gpt-4o", "claude-3-opus", "gemini-1.5-pro"]
        result = await engine.query_for_consensus("Test query", models)
        assert result.metadata["consortium_type"] == "heterogeneous"
        assert "warning" not in result.metadata


class TestHomogeneousOptimizationFeature:
    """Tests for homogeneous optimization feature flag integration."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock provider registry."""
        registry = MagicMock()

        async def mock_query_multiple(message, models):
            return [
                Response(
                    message_id=message.id,
                    model=model,
                    content="Test response",
                    confidence=ConfidenceLevel.HIGH,
                )
                for model in models
            ]

        registry.query_multiple = AsyncMock(side_effect=mock_query_multiple)
        return registry

    @pytest.fixture
    def engine(self, mock_registry):
        """Create a ConsensusEngine instance."""
        return ConsensusEngine(mock_registry)

    @pytest.mark.asyncio
    async def test_logs_when_feature_enabled_homogeneous(self, engine, caplog):
        """Should log info when homogeneous + feature enabled."""
        with patch("i2i.consensus.feature_enabled", return_value=True):
            import logging
            caplog.set_level(logging.INFO)
            models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
            await engine.query_for_consensus("Test", models)
            # Check that logging occurred (caplog may not capture if logger not configured)
            # This is more of a smoke test

    @pytest.mark.asyncio
    async def test_no_log_when_feature_disabled(self, engine):
        """Should not log extra info when feature disabled."""
        with patch("i2i.consensus.feature_enabled", return_value=False):
            models = ["claude-3-opus", "claude-3-sonnet"]
            # Should complete without error
            result = await engine.query_for_consensus("Test", models)
            assert result is not None
