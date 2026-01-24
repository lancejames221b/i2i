"""Tests for i2i configuration system."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from i2i.config import (
    Config,
    DEFAULTS,
    deep_merge,
    apply_env_overrides,
    get_config,
    set_config,
    reset_config,
    get_consensus_models,
    get_classifier_model,
    load_json_config,
)


class TestDeepMerge:
    """Tests for deep_merge function."""

    def test_merge_flat_dicts(self):
        """Merge two flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self):
        """Merge nested dictionaries recursively."""
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3, "c": 4}}
        result = deep_merge(base, override)
        assert result == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_merge_does_not_modify_originals(self):
        """deep_merge should not modify input dictionaries."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        result = deep_merge(base, override)
        assert base == {"a": {"b": 1}}
        assert override == {"a": {"c": 2}}

    def test_override_replaces_non_dict_values(self):
        """Override replaces entire value if not both dicts."""
        base = {"a": {"b": 1}}
        override = {"a": "replaced"}
        result = deep_merge(base, override)
        assert result == {"a": "replaced"}

    def test_override_with_list(self):
        """Override replaces list entirely."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = deep_merge(base, override)
        assert result == {"items": [4, 5]}


class TestApplyEnvOverrides:
    """Tests for environment variable overrides."""

    def test_consensus_model_override(self, clean_env):
        """Override consensus models via env vars."""
        config = {"models": {"consensus": ["model-a", "model-b", "model-c"]}}
        with patch.dict(os.environ, {"I2I_CONSENSUS_MODEL_1": "new-model"}):
            result = apply_env_overrides(config)
        assert result["models"]["consensus"][0] == "new-model"

    def test_classifier_model_override(self, clean_env):
        """Override classifier model via env var."""
        config = {"models": {"classifier": "old-model"}}
        with patch.dict(os.environ, {"I2I_CLASSIFIER_MODEL": "new-classifier"}):
            result = apply_env_overrides(config)
        assert result["models"]["classifier"] == "new-classifier"

    def test_routing_strategy_override(self, clean_env):
        """Override routing strategy via env var."""
        config = {"routing": {"default_strategy": "balanced"}}
        with patch.dict(os.environ, {"I2I_ROUTING_STRATEGY": "best_quality"}):
            result = apply_env_overrides(config)
        assert result["routing"]["default_strategy"] == "best_quality"

    def test_ai_classifier_boolean_override(self, clean_env):
        """Override use_ai_classifier boolean via env var."""
        config = {"routing": {"use_ai_classifier": False}}
        with patch.dict(os.environ, {"I2I_USE_AI_CLASSIFIER": "true"}):
            result = apply_env_overrides(config)
        assert result["routing"]["use_ai_classifier"] is True

    def test_no_env_vars_leaves_config_unchanged(self, clean_env):
        """Config unchanged when no env vars are set."""
        config = {"models": {"consensus": ["a", "b", "c"], "classifier": "x"}}
        result = apply_env_overrides(config)
        assert result == config


class TestConfig:
    """Tests for Config class."""

    def test_load_defaults(self):
        """Config.load_defaults returns built-in defaults."""
        config = Config.load_defaults()
        assert config.get("version") == "1.0"
        assert "gpt-5.2" in config.get("models.consensus")

    def test_get_with_dot_notation(self):
        """Get nested values using dot notation."""
        config = Config({"outer": {"inner": {"value": 42}}})
        assert config.get("outer.inner.value") == 42

    def test_get_with_default(self):
        """Get returns default for missing keys."""
        config = Config({})
        assert config.get("nonexistent", "default") == "default"

    def test_get_list_index(self):
        """Get can access list elements by index."""
        config = Config({"items": ["a", "b", "c"]})
        assert config.get("items.1") == "b"

    def test_set_with_dot_notation(self):
        """Set nested values using dot notation."""
        config = Config({"outer": {}})
        config.set("outer.inner", "value")
        assert config.get("outer.inner") == "value"

    def test_set_creates_nested_dicts(self):
        """Set creates intermediate dicts as needed."""
        config = Config({})
        config.set("a.b.c", "value")
        assert config.get("a.b.c") == "value"

    def test_add_to_list(self):
        """Add appends value to list."""
        config = Config({"items": ["a", "b"]})
        result = config.add("items", "c")
        assert result is True
        assert "c" in config.get("items")

    def test_add_duplicate_returns_false(self):
        """Add returns False if value already exists."""
        config = Config({"items": ["a", "b"]})
        result = config.add("items", "a")
        assert result is False
        assert config.get("items").count("a") == 1

    def test_add_to_non_list_raises(self):
        """Add raises ValueError for non-list keys."""
        config = Config({"value": "string"})
        with pytest.raises(ValueError, match="not a list"):
            config.add("value", "x")

    def test_remove_from_list(self):
        """Remove removes value from list."""
        config = Config({"items": ["a", "b", "c"]})
        result = config.remove("items", "b")
        assert result is True
        assert "b" not in config.get("items")

    def test_remove_nonexistent_returns_false(self):
        """Remove returns False if value not found."""
        config = Config({"items": ["a", "b"]})
        result = config.remove("items", "c")
        assert result is False

    def test_reset_to_defaults(self):
        """Reset restores built-in defaults."""
        config = Config({"custom": "value"})
        config.reset()
        assert config.get("custom") is None
        assert "gpt-5.2" in config.get("models.consensus")

    def test_to_dict(self):
        """to_dict returns config as dictionary."""
        config = Config({"a": 1, "b": 2})
        result = config.to_dict()
        assert result == {"a": 1, "b": 2}
        # Should be a copy, not the original
        result["a"] = 999
        assert config.get("a") == 1

    def test_save_and_load(self, temp_config_dir):
        """Config can be saved and loaded from file."""
        config = Config({"test": "value"})
        path = temp_config_dir / "test_config.json"
        config.save(path)

        loaded = Config.load(path)
        assert loaded.get("test") == "value"

    def test_convenience_properties(self):
        """Test convenience property accessors."""
        config = Config.load_defaults()
        assert isinstance(config.consensus_models, list)
        assert isinstance(config.classifier_model, str)
        assert isinstance(config.synthesis_models, list)
        assert isinstance(config.verification_models, list)
        assert isinstance(config.epistemic_models, list)


class TestGlobalConfig:
    """Tests for global config functions."""

    def test_get_config_returns_config(self, clean_env):
        """get_config returns a Config instance."""
        reset_config()  # Ensure clean state
        config = get_config()
        assert isinstance(config, Config)

    def test_get_config_is_singleton(self, clean_env):
        """get_config returns same instance on repeated calls."""
        reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config_overrides_global(self, clean_env):
        """set_config overrides the global config."""
        reset_config()
        custom = Config({"custom": "value"})
        set_config(custom)
        assert get_config().get("custom") == "value"
        reset_config()

    def test_reset_config_clears_singleton(self, clean_env):
        """reset_config clears the cached singleton."""
        reset_config()
        config1 = get_config()
        reset_config()
        config2 = get_config()
        # Should be different instances after reset
        assert config1 is not config2


class TestConvenienceAccessors:
    """Tests for module-level convenience functions."""

    def test_get_consensus_models(self, clean_env):
        """get_consensus_models returns list of models."""
        reset_config()
        models = get_consensus_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_get_classifier_model(self, clean_env):
        """get_classifier_model returns a model string."""
        reset_config()
        model = get_classifier_model()
        assert isinstance(model, str)
        assert len(model) > 0


class TestLoadJsonConfig:
    """Tests for JSON config file loading."""

    def test_load_valid_json(self, temp_config_dir):
        """Load a valid JSON config file."""
        path = temp_config_dir / "config.json"
        path.write_text('{"key": "value"}')
        result = load_json_config(path)
        assert result == {"key": "value"}

    def test_load_nonexistent_returns_empty(self, temp_config_dir):
        """Loading nonexistent file returns empty dict."""
        path = temp_config_dir / "nonexistent.json"
        result = load_json_config(path)
        assert result == {}

    def test_load_invalid_json_returns_empty(self, temp_config_dir):
        """Loading invalid JSON returns empty dict."""
        path = temp_config_dir / "invalid.json"
        path.write_text("not valid json {{{")
        result = load_json_config(path)
        assert result == {}


class TestDefaults:
    """Tests for built-in default configuration."""

    def test_defaults_has_required_keys(self):
        """DEFAULTS contains all required top-level keys."""
        assert "version" in DEFAULTS
        assert "models" in DEFAULTS
        assert "routing" in DEFAULTS
        assert "consensus" in DEFAULTS
        assert "providers" in DEFAULTS

    def test_defaults_models_complete(self):
        """DEFAULTS.models has all required model categories."""
        models = DEFAULTS["models"]
        assert "consensus" in models
        assert "classifier" in models
        assert "synthesis" in models
        assert "verification" in models
        assert "epistemic" in models

    def test_defaults_providers_complete(self):
        """DEFAULTS.providers has all supported providers."""
        providers = DEFAULTS["providers"]
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert "mistral" in providers
        assert "groq" in providers
        assert "cohere" in providers
