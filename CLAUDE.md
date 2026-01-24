# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**i2i** (eye-to-eye) is a Python implementation of the **MCIP** (Multi-model Consensus and Inference Protocol) - a standardized protocol for AI-to-AI communication. It enables consensus queries across multiple LLM providers, cross-verification of claims, epistemic classification of questions, and intelligent model routing.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install with dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check i2i/
black --check i2i/

# Format code
black i2i/

# Run tests
pytest
pytest -xvs tests/test_specific.py::test_function  # single test
```

## Demo CLI Usage

```bash
# Check provider status
python demo.py status

# Consensus query across models
python demo.py consensus "What causes inflation?"

# Verify a claim
python demo.py verify "Einstein failed math in school"

# Classify question epistemically
python demo.py classify "Do we have free will?" --quick

# Intelligent routing
python demo.py route "Write a Python sort function" --strategy best_quality --execute

# Multi-model debate
python demo.py debate "Should AI have rights?" --rounds 3

# Get model recommendations
python demo.py recommend code_generation
```

## Architecture

```
AICP (protocol.py)          # Main entry point - orchestrates all components
    ├── ProviderRegistry    # Manages provider adapters (providers.py)
    ├── ConsensusEngine     # Multi-model agreement detection (consensus.py)
    ├── VerificationEngine  # Cross-verification of claims (verification.py)
    ├── EpistemicClassifier # Question answerability classification (epistemic.py)
    └── ModelRouter         # Task detection and model selection (router.py)
```

### Core Components

- **`AICP`** (`protocol.py`): Primary interface. Key methods: `consensus_query()`, `verify_claim()`, `classify_question()`, `routed_query()`, `smart_query()`, `debate()`

- **`ProviderRegistry`** (`providers.py`): Abstract adapter pattern for AI providers (OpenAI, Anthropic, Google, Mistral, Groq, Cohere). Each provider implements `ProviderAdapter` interface.

- **`ConsensusEngine`** (`consensus.py`): Queries multiple models, calculates similarity using Jaccard on normalized tokens, determines consensus level (HIGH/MEDIUM/LOW/NONE/CONTRADICTORY)

- **`ModelRouter`** (`router.py`): Classifies task type from query text, maintains capability scores per model, selects optimal model(s) based on strategy (BEST_QUALITY/BEST_SPEED/BEST_VALUE/BALANCED/ENSEMBLE)

- **`EpistemicClassifier`** (`epistemic.py`): Classifies questions as ANSWERABLE/UNCERTAIN/UNDERDETERMINED/IDLE/MALFORMED. Includes `quick_classify()` heuristic that avoids API calls.

### Data Flow

```
User Query
    │
    ▼
AICP.smart_query()
    │
    ├──► EpistemicClassifier (is this answerable?)
    │
    ├──► ConsensusEngine (query multiple models)
    │         │
    │         └──► ProviderRegistry.query_multiple()
    │                   │
    │                   └──► [OpenAI, Anthropic, Google, ...]
    │
    └──► VerificationEngine (optional cross-check)
```

## Configuration

Models are configured via (in priority order):
1. Environment variables: `I2I_CONSENSUS_MODEL_1`, `I2I_CLASSIFIER_MODEL`, etc.
2. `config.json` in project root
3. `~/.i2i/config.json` for user defaults

Key environment variables:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...
```

Requires **at least 2 configured providers** for consensus features.

## Key Types (schema.py)

- `ConsensusLevel`: HIGH (≥85%), MEDIUM (60-84%), LOW (30-59%), NONE (<30%), CONTRADICTORY
- `EpistemicType`: ANSWERABLE, UNCERTAIN, UNDERDETERMINED, IDLE, MALFORMED
- `TaskType`: CODE_GENERATION, MATHEMATICAL, CREATIVE_WRITING, FACTUAL_QA, etc.
- `RoutingStrategy`: BEST_QUALITY, BEST_SPEED, BEST_VALUE, BALANCED, ENSEMBLE, FALLBACK_CHAIN

## Adding a New Provider

1. Create adapter class in `providers.py` extending `ProviderAdapter`
2. Implement: `provider_name`, `available_models`, `is_configured()`, `query()`
3. Register in `ProviderRegistry.__init__`
4. Add model capabilities in `router.py` `MODEL_CAPABILITIES` dict
