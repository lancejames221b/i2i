from pathlib import Path

from i2i.config import load_json_config


def test_project_config_uses_requested_mcip_panel():
    """MCIP panel should match the requested runtime model set."""
    repo_root = Path(__file__).resolve().parents[2]
    config = load_json_config(repo_root / "config.json")
    models = config["models"]

    expected_panel = [
        "gpt-5.3-codex",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    ]

    assert models["consensus"] == expected_panel
    assert models["synthesis"] == expected_panel
    assert models["verification"] == expected_panel
    assert models["epistemic"] == [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "gpt-5.3-codex",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    ]
