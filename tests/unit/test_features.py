"""Tests for i2i feature flags system."""

import os
import pytest
from unittest.mock import patch

from i2i.config import (
    Config,
    DEFAULTS,
    apply_env_overrides,
    feature_enabled,
    reset_config,
    FEATURE_ENV_MAPPINGS,
)


class TestFeatureFlagsDefaults:
    """Tests for default feature flag values."""

    def test_features_section_exists(self):
        """DEFAULTS should have a features section."""
        assert "features" in DEFAULTS
        assert isinstance(DEFAULTS["features"], dict)

    def test_multimodal_default_false(self):
        """Multimodal should be disabled by default."""
        assert DEFAULTS["features"]["multimodal"] is False

    def test_homogeneous_optimization_default_false(self):
        """Homogeneous optimization should be disabled by default."""
        assert DEFAULTS["features"]["homogeneous_optimization"] is False

    def test_self_consistency_sampling_default_false(self):
        """Self-consistency sampling should be disabled by default."""
        assert DEFAULTS["features"]["self_consistency_sampling"] is False

    def test_multi_round_debate_default_true(self):
        """Multi-round debate should be enabled by default (already implemented)."""
        assert DEFAULTS["features"]["multi_round_debate"] is True

    def test_latent_collaboration_default_false(self):
        """Latent collaboration should be disabled by default."""
        assert DEFAULTS["features"]["latent_collaboration"] is False


class TestConfigFeatureMethods:
    """Tests for Config class feature methods."""

    @pytest.fixture
    def config(self):
        """Create a fresh Config instance."""
        reset_config()
        return Config.load_defaults()

    def test_feature_enabled_returns_bool(self, config):
        """feature_enabled should return a boolean."""
        result = config.feature_enabled("multimodal")
        assert isinstance(result, bool)

    def test_feature_enabled_default_false(self, config):
        """feature_enabled returns False for disabled features."""
        assert config.feature_enabled("multimodal") is False
        assert config.feature_enabled("homogeneous_optimization") is False

    def test_feature_enabled_default_true(self, config):
        """feature_enabled returns True for enabled features."""
        assert config.feature_enabled("multi_round_debate") is True

    def test_feature_enabled_unknown_feature(self, config):
        """feature_enabled returns False for unknown features."""
        assert config.feature_enabled("nonexistent_feature") is False

    def test_enable_feature(self, config):
        """enable_feature should enable a feature."""
        assert config.feature_enabled("multimodal") is False
        config.enable_feature("multimodal")
        assert config.feature_enabled("multimodal") is True

    def test_disable_feature(self, config):
        """disable_feature should disable a feature."""
        assert config.feature_enabled("multi_round_debate") is True
        config.disable_feature("multi_round_debate")
        assert config.feature_enabled("multi_round_debate") is False

    def test_list_features(self, config):
        """list_features should return all features."""
        features = config.list_features()
        assert isinstance(features, dict)
        assert "multimodal" in features
        assert "homogeneous_optimization" in features
        assert "self_consistency_sampling" in features
        assert "multi_round_debate" in features
        assert "latent_collaboration" in features


class TestFeatureEnvOverrides:
    """Tests for feature flag environment variable overrides."""

    def test_env_mappings_exist(self):
        """Feature env mappings should be defined."""
        assert len(FEATURE_ENV_MAPPINGS) == 5
        assert "I2I_FEATURE_MULTIMODAL" in FEATURE_ENV_MAPPINGS
        assert "I2I_FEATURE_HOMOGENEOUS_OPTIMIZATION" in FEATURE_ENV_MAPPINGS

    def test_multimodal_env_override_true(self, clean_env):
        """I2I_FEATURE_MULTIMODAL=true enables multimodal."""
        config = {"features": {"multimodal": False}}
        with patch.dict(os.environ, {"I2I_FEATURE_MULTIMODAL": "true"}):
            result = apply_env_overrides(config)
        assert result["features"]["multimodal"] is True

    def test_multimodal_env_override_false(self, clean_env):
        """I2I_FEATURE_MULTIMODAL=false disables multimodal."""
        config = {"features": {"multimodal": True}}
        with patch.dict(os.environ, {"I2I_FEATURE_MULTIMODAL": "false"}):
            result = apply_env_overrides(config)
        assert result["features"]["multimodal"] is False

    def test_env_override_accepts_1(self, clean_env):
        """Environment variable '1' should be treated as true."""
        config = {"features": {"multimodal": False}}
        with patch.dict(os.environ, {"I2I_FEATURE_MULTIMODAL": "1"}):
            result = apply_env_overrides(config)
        assert result["features"]["multimodal"] is True

    def test_env_override_accepts_yes(self, clean_env):
        """Environment variable 'yes' should be treated as true."""
        config = {"features": {"multimodal": False}}
        with patch.dict(os.environ, {"I2I_FEATURE_MULTIMODAL": "yes"}):
            result = apply_env_overrides(config)
        assert result["features"]["multimodal"] is True

    def test_env_override_accepts_on(self, clean_env):
        """Environment variable 'on' should be treated as true."""
        config = {"features": {"multimodal": False}}
        with patch.dict(os.environ, {"I2I_FEATURE_MULTIMODAL": "on"}):
            result = apply_env_overrides(config)
        assert result["features"]["multimodal"] is True

    def test_env_override_case_insensitive(self, clean_env):
        """Environment variable values should be case insensitive."""
        config = {"features": {"multimodal": False}}
        with patch.dict(os.environ, {"I2I_FEATURE_MULTIMODAL": "TRUE"}):
            result = apply_env_overrides(config)
        assert result["features"]["multimodal"] is True

    def test_homogeneous_optimization_env_override(self, clean_env):
        """I2I_FEATURE_HOMOGENEOUS_OPTIMIZATION should work."""
        config = {"features": {"homogeneous_optimization": False}}
        with patch.dict(os.environ, {"I2I_FEATURE_HOMOGENEOUS_OPTIMIZATION": "true"}):
            result = apply_env_overrides(config)
        assert result["features"]["homogeneous_optimization"] is True

    def test_multiple_env_overrides(self, clean_env):
        """Multiple feature flags can be overridden at once."""
        config = {"features": {"multimodal": False, "homogeneous_optimization": False}}
        with patch.dict(os.environ, {
            "I2I_FEATURE_MULTIMODAL": "true",
            "I2I_FEATURE_HOMOGENEOUS_OPTIMIZATION": "true",
        }):
            result = apply_env_overrides(config)
        assert result["features"]["multimodal"] is True
        assert result["features"]["homogeneous_optimization"] is True


class TestFeatureEnabledConvenience:
    """Tests for the convenience feature_enabled function."""

    def test_convenience_function_exists(self):
        """feature_enabled convenience function should exist."""
        from i2i.config import feature_enabled
        assert callable(feature_enabled)

    def test_convenience_function_returns_bool(self, clean_env):
        """Convenience function should return boolean."""
        reset_config()
        result = feature_enabled("multimodal")
        assert isinstance(result, bool)

    def test_convenience_function_reflects_env(self, clean_env):
        """Convenience function should reflect env overrides."""
        reset_config()
        # Fresh config without env override
        assert feature_enabled("multimodal") is False

        # With env override (need to reset config to pick up change)
        with patch.dict(os.environ, {"I2I_FEATURE_MULTIMODAL": "true"}):
            reset_config()  # Force reload
            from i2i.config import get_config
            config = get_config()
            assert config.feature_enabled("multimodal") is True
