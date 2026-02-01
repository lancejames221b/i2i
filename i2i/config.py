"""
Configuration module for i2i.

Supports multiple configuration sources with the following priority:
1. Environment variables (highest priority)
2. User config file (~/.i2i/config.json)
3. Project config file (./config.json)
4. Built-in defaults (lowest priority)

Use the CLI to manage configuration:
    i2i config show           # Show current config
    i2i config set models.classifier gpt-5.2
    i2i config add models.consensus o3
    i2i config remove models.consensus gemini-3-flash-preview
    i2i config reset          # Reset to defaults
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from copy import deepcopy


# Default configuration - built-in fallback
DEFAULTS = {
    "version": "1.0",
    "models": {
        "consensus": [
            "gpt-5.2",
            "claude-sonnet-4-5-20250929",
            "gemini-3-flash-preview",
        ],
        "classifier": "claude-haiku-4-5-20251001",
        "synthesis": [
            "gpt-5.2",
            "claude-sonnet-4-5-20250929",
            "gemini-3-flash-preview",
        ],
        "verification": [
            "gpt-5.2",
            "claude-sonnet-4-5-20250929",
        ],
        "epistemic": [
            "claude-sonnet-4-5-20250929",
            "gpt-5.2",
            "gemini-3-flash-preview",
        ],
    },
    "routing": {
        "default_strategy": "balanced",
        "use_ai_classifier": False,
        "fallback_enabled": True,
    },
    "consensus": {
        "min_agreement_threshold": 0.7,
        "max_rounds": 3,
        "require_unanimous": False,
        # Consensus level thresholds (similarity score boundaries)
        # These determine how average pairwise similarity maps to consensus levels
        "thresholds": {
            "high": 0.85,      # >= 0.85 = HIGH consensus
            "medium": 0.60,    # >= 0.60 = MEDIUM consensus
            "low": 0.30,       # >= 0.30 = LOW consensus
            # Below 0.30 = NONE or CONTRADICTORY
        },
        # Clustering threshold for grouping agreeing models
        "clustering_threshold": 0.70,
        # Divergence threshold for identifying significant disagreement
        "divergence_threshold": 0.50,
    },
    "statistical_mode": {
        "enabled": False,  # Feature flag - set True to enable by default
        "n_runs": 5,  # Number of runs per model
        "temperature": 0.7,  # Temperature for variance (0 = deterministic)
        "outlier_threshold": 2.0,  # Standard deviations for outlier detection
        "use_embeddings": True,  # Use embedding-based similarity (vs Jaccard)
        "embedding_model": "text-embedding-3-small",  # OpenAI embedding model
        "parallel_queries": True,  # Run n queries in parallel
    },
    "providers": {
        "openai": {"enabled": True, "timeout_ms": 30000},
        "anthropic": {"enabled": True, "timeout_ms": 30000},
        "google": {"enabled": True, "timeout_ms": 30000},
        "mistral": {"enabled": True, "timeout_ms": 30000},
        "groq": {"enabled": True, "timeout_ms": 30000},
        "cohere": {"enabled": True, "timeout_ms": 30000},
    },
    # Feature flags for experimental/optional capabilities
    # Reference: Wang et al. 2022 (self-consistency), Du et al. 2023 (debate), Zou et al. 2025 (latent)
    "features": {
        "multimodal": False,                   # Enable image/non-text content handling
        "homogeneous_optimization": False,     # Optimize for same-family model consortiums
        "self_consistency_sampling": False,    # Wang et al. 2022 - sample multiple from same model
        "multi_round_debate": True,            # Du et al. 2023 - already partially implemented
        "latent_collaboration": False,         # Zou et al. 2025 - token reduction via latent space
    },
}

# Feature flag environment variable mappings
FEATURE_ENV_MAPPINGS = {
    "I2I_FEATURE_MULTIMODAL": "features.multimodal",
    "I2I_FEATURE_HOMOGENEOUS_OPTIMIZATION": "features.homogeneous_optimization",
    "I2I_FEATURE_SELF_CONSISTENCY": "features.self_consistency_sampling",
    "I2I_FEATURE_MULTI_ROUND_DEBATE": "features.multi_round_debate",
    "I2I_FEATURE_LATENT_COLLABORATION": "features.latent_collaboration",
}


def get_config_paths() -> List[Path]:
    """Get list of config file paths in priority order."""
    paths = []

    # User config directory
    user_config = Path.home() / ".i2i" / "config.json"
    if user_config.exists():
        paths.append(user_config)

    # Project config (current directory)
    project_config = Path.cwd() / "config.json"
    if project_config.exists():
        paths.append(project_config)

    # Package default config
    package_config = Path(__file__).parent.parent / "config.json"
    if package_config.exists():
        paths.append(package_config)

    return paths


def get_user_config_path() -> Path:
    """Get the user config path (creates directory if needed)."""
    user_dir = Path.home() / ".i2i"
    user_dir.mkdir(exist_ok=True)
    return user_dir / "config.json"


def load_json_config(path: Path) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return {}


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config."""
    config = deepcopy(config)

    # Model overrides
    env_mappings = {
        "I2I_CONSENSUS_MODEL_1": ("models", "consensus", 0),
        "I2I_CONSENSUS_MODEL_2": ("models", "consensus", 1),
        "I2I_CONSENSUS_MODEL_3": ("models", "consensus", 2),
        "I2I_CLASSIFIER_MODEL": ("models", "classifier"),
        "I2I_SYNTHESIS_MODEL_1": ("models", "synthesis", 0),
        "I2I_SYNTHESIS_MODEL_2": ("models", "synthesis", 1),
        "I2I_SYNTHESIS_MODEL_3": ("models", "synthesis", 2),
        "I2I_VERIFICATION_MODEL_1": ("models", "verification", 0),
        "I2I_VERIFICATION_MODEL_2": ("models", "verification", 1),
        "I2I_EPISTEMIC_MODEL_1": ("models", "epistemic", 0),
        "I2I_EPISTEMIC_MODEL_2": ("models", "epistemic", 1),
        "I2I_EPISTEMIC_MODEL_3": ("models", "epistemic", 2),
    }

    for env_var, path in env_mappings.items():
        value = os.getenv(env_var)
        if value:
            if len(path) == 2:
                # Direct assignment (e.g., classifier)
                config[path[0]][path[1]] = value
            elif len(path) == 3:
                # List index assignment
                section, key, idx = path
                if idx < len(config[section][key]):
                    config[section][key][idx] = value

    # Routing overrides
    if os.getenv("I2I_ROUTING_STRATEGY"):
        config["routing"]["default_strategy"] = os.getenv("I2I_ROUTING_STRATEGY")
    if os.getenv("I2I_USE_AI_CLASSIFIER"):
        config["routing"]["use_ai_classifier"] = os.getenv("I2I_USE_AI_CLASSIFIER").lower() == "true"

    # Statistical mode overrides
    if os.getenv("I2I_STATISTICAL_MODE"):
        config["statistical_mode"]["enabled"] = os.getenv("I2I_STATISTICAL_MODE").lower() == "true"
    if os.getenv("I2I_STATISTICAL_N_RUNS"):
        config["statistical_mode"]["n_runs"] = int(os.getenv("I2I_STATISTICAL_N_RUNS"))
    if os.getenv("I2I_STATISTICAL_TEMPERATURE"):
        config["statistical_mode"]["temperature"] = float(os.getenv("I2I_STATISTICAL_TEMPERATURE"))
    if os.getenv("I2I_STATISTICAL_OUTLIER_THRESHOLD"):
        config["statistical_mode"]["outlier_threshold"] = float(os.getenv("I2I_STATISTICAL_OUTLIER_THRESHOLD"))
    if os.getenv("I2I_STATISTICAL_USE_EMBEDDINGS"):
        config["statistical_mode"]["use_embeddings"] = os.getenv("I2I_STATISTICAL_USE_EMBEDDINGS").lower() == "true"
    if os.getenv("I2I_STATISTICAL_EMBEDDING_MODEL"):
        config["statistical_mode"]["embedding_model"] = os.getenv("I2I_STATISTICAL_EMBEDDING_MODEL")

    # Feature flag overrides (I2I_FEATURE_* env vars)
    for env_var, config_path in FEATURE_ENV_MAPPINGS.items():
        value = os.getenv(env_var)
        if value is not None:
            # Parse boolean from string
            bool_value = value.lower() in ("true", "1", "yes", "on")
            parts = config_path.split(".")
            target = config
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = bool_value

    return config


class Config:
    """
    Configuration manager for i2i.

    Usage:
        config = Config.load()
        models = config.get("models.consensus")
        config.set("models.classifier", "gpt-5.2")
        config.add("models.consensus", "o3")
        config.remove("models.consensus", "gemini-3-flash-preview")
        config.save()
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or deepcopy(DEFAULTS)
        self._source_path: Optional[Path] = None

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Load configuration from files and environment."""
        # Start with defaults
        data = deepcopy(DEFAULTS)

        # Load and merge config files (lowest to highest priority)
        if path:
            paths = [path]
        else:
            paths = list(reversed(get_config_paths()))  # Reverse so highest priority is last

        source_path = None
        for config_path in paths:
            file_config = load_json_config(config_path)
            if file_config:
                data = deep_merge(data, file_config)
                source_path = config_path

        # Apply environment variable overrides (highest priority)
        data = apply_env_overrides(data)

        config = cls(data)
        config._source_path = source_path
        return config

    @classmethod
    def load_defaults(cls) -> "Config":
        """Load only the built-in defaults."""
        return cls(deepcopy(DEFAULTS))

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Examples:
            config.get("models.consensus")
            config.get("routing.default_strategy")
            config.get("providers.openai.timeout_ms")
        """
        parts = key.split(".")
        value = self._data
        try:
            for part in parts:
                if isinstance(value, list):
                    value = value[int(part)]
                else:
                    value = value[part]
            return value
        except (KeyError, IndexError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Examples:
            config.set("models.classifier", "gpt-5.2")
            config.set("routing.use_ai_classifier", True)
        """
        parts = key.split(".")
        target = self._data
        for part in parts[:-1]:
            if isinstance(target, list):
                target = target[int(part)]
            else:
                if part not in target:
                    target[part] = {}
                target = target[part]

        final_key = parts[-1]
        if isinstance(target, list):
            target[int(final_key)] = value
        else:
            target[final_key] = value

    def add(self, key: str, value: Any) -> bool:
        """
        Add a value to a list configuration.

        Returns True if added, False if already exists.
        """
        current = self.get(key)
        if not isinstance(current, list):
            raise ValueError(f"{key} is not a list")
        if value not in current:
            current.append(value)
            return True
        return False

    def remove(self, key: str, value: Any) -> bool:
        """
        Remove a value from a list configuration.

        Returns True if removed, False if not found.
        """
        current = self.get(key)
        if not isinstance(current, list):
            raise ValueError(f"{key} is not a list")
        if value in current:
            current.remove(value)
            return True
        return False

    def save(self, path: Optional[Path] = None) -> Path:
        """Save configuration to a file."""
        if path is None:
            path = get_user_config_path()

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._data, f, indent=2)

        self._source_path = path
        return path

    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._data = deepcopy(DEFAULTS)

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return deepcopy(self._data)

    # ==================== Feature Flags ====================

    def feature_enabled(self, feature: str) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            feature: Feature name (e.g., "multimodal", "homogeneous_optimization")

        Returns:
            True if the feature is enabled, False otherwise.

        Examples:
            config.feature_enabled("multimodal")
            config.feature_enabled("self_consistency_sampling")
        """
        return self.get(f"features.{feature}", False)

    def enable_feature(self, feature: str) -> None:
        """Enable a feature flag."""
        self.set(f"features.{feature}", True)

    def disable_feature(self, feature: str) -> None:
        """Disable a feature flag."""
        self.set(f"features.{feature}", False)

    def list_features(self) -> Dict[str, bool]:
        """List all feature flags and their current state."""
        return self.get("features", {})

    @property
    def source_path(self) -> Optional[Path]:
        """Path of the loaded config file, if any."""
        return self._source_path

    # Convenience properties for common access patterns
    @property
    def consensus_models(self) -> List[str]:
        return self.get("models.consensus", [])

    @property
    def classifier_model(self) -> str:
        return self.get("models.classifier", "claude-haiku-4-5-20251001")

    @property
    def synthesis_models(self) -> List[str]:
        return self.get("models.synthesis", [])

    @property
    def verification_models(self) -> List[str]:
        return self.get("models.verification", [])

    @property
    def epistemic_models(self) -> List[str]:
        return self.get("models.epistemic", [])


# Global config instance - lazy loaded
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def set_config(config: Config) -> None:
    """Override the global configuration."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset to default configuration."""
    global _config
    _config = None


# Convenience accessors (backwards compatible)
def get_consensus_models() -> List[str]:
    """Get default models for consensus queries."""
    return get_config().consensus_models


def get_classifier_model() -> str:
    """Get default model for task classification."""
    return get_config().classifier_model


def get_synthesis_models() -> List[str]:
    """Get default models for synthesis."""
    return get_config().synthesis_models


def get_verification_models() -> List[str]:
    """Get default models for verification."""
    return get_config().verification_models


def get_epistemic_models() -> List[str]:
    """Get default models for epistemic classification."""
    return get_config().epistemic_models


def get_statistical_mode_config() -> Dict[str, Any]:
    """Get statistical mode configuration."""
    return get_config().get("statistical_mode", DEFAULTS["statistical_mode"])


def is_statistical_mode_enabled() -> bool:
    """Check if statistical mode is enabled by default."""
    return get_config().get("statistical_mode.enabled", False)


def get_statistical_n_runs() -> int:
    """Get the number of runs per model for statistical mode."""
    return get_config().get("statistical_mode.n_runs", 5)


def get_statistical_temperature() -> float:
    """Get the temperature for statistical mode queries."""
    return get_config().get("statistical_mode.temperature", 0.7)


def get_statistical_outlier_threshold() -> float:
    """Get the outlier detection threshold (in standard deviations)."""
    return get_config().get("statistical_mode.outlier_threshold", 2.0)


def get_embedding_model() -> str:
    """Get the embedding model for statistical similarity."""
    return get_config().get("statistical_mode.embedding_model", "text-embedding-3-small")


def feature_enabled(feature: str) -> bool:
    """Check if a feature flag is enabled (convenience function)."""
    return get_config().feature_enabled(feature)


# ==================== Consensus Threshold Accessors ====================

def get_consensus_thresholds() -> Dict[str, float]:
    """
    Get the consensus level thresholds.

    Returns:
        Dict with 'high', 'medium', 'low' threshold values.
    """
    defaults = {"high": 0.85, "medium": 0.60, "low": 0.30}
    return get_config().get("consensus.thresholds", defaults)


def get_high_threshold() -> float:
    """Get the HIGH consensus threshold (default: 0.85)."""
    return get_consensus_thresholds().get("high", 0.85)


def get_medium_threshold() -> float:
    """Get the MEDIUM consensus threshold (default: 0.60)."""
    return get_consensus_thresholds().get("medium", 0.60)


def get_low_threshold() -> float:
    """Get the LOW consensus threshold (default: 0.30)."""
    return get_consensus_thresholds().get("low", 0.30)


def get_clustering_threshold() -> float:
    """Get the clustering threshold for grouping models (default: 0.70)."""
    return get_config().get("consensus.clustering_threshold", 0.70)


def get_divergence_threshold() -> float:
    """Get the divergence threshold for identifying disagreement (default: 0.50)."""
    return get_config().get("consensus.divergence_threshold", 0.50)
