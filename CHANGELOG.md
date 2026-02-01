# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **LangChain Integration**: Full integration with LangChain LCEL pipelines
  - `I2IVerifier` Runnable for adding consensus verification to chains
  - `I2IVerificationCallback` for automatic verification of LLM responses
  - `I2IVerifiedChain` wrapper for existing chains
  - `create_verified_chain()` helper function
  - Task-aware verification that skips consensus for math/reasoning tasks
  - Calibrated confidence scores based on empirical evaluation data
  - Comprehensive documentation at `docs/integrations/langchain.md`
  - Install with: `pip install i2i-mcip[langchain]`

## [0.2.1] - 2025-01-31

### Added

- Task-aware consensus with calibrated confidence scores
- `task_category` parameter for explicit task type specification
- `consensus_appropriate` field in consensus results
- `recommend_consensus()` function to check if consensus is appropriate
- CLI benchmark commands for evaluation

### Changed

- Consensus results now include `confidence_calibration` field
- Default behaviour warns when consensus is used for math/reasoning tasks

## [0.2.0] - 2025-01-30

### Added

- Task-aware consensus engine based on empirical evaluation
- Statistical consensus mode with n-run averaging
- Task classifier for automatic task type detection
- Calibrated confidence scores (HIGH=0.95, MEDIUM=0.75, LOW=0.60, NONE=0.50)

### Changed

- ConsensusResult now includes task-aware fields
- Improved documentation with evaluation results

## [0.1.0] - 2025-01-15

### Added

- Initial release of i2i (MCIP implementation)
- Multi-model consensus queries across providers
- Claim verification with cross-model fact-checking
- Epistemic classification (ANSWERABLE, UNCERTAIN, UNDERDETERMINED, IDLE, MALFORMED)
- Intelligent model routing with capability matrix
- Multi-model debate functionality
- Support for OpenAI, Anthropic, Google, Mistral, Groq, Cohere providers
- Ollama support for local models
- LiteLLM proxy support for 100+ models
- Perplexity integration for RAG-native queries
- Search-grounded verification (Brave, SerpAPI, Tavily)
- CLI tool with consensus, verify, classify, route, debate commands
- Configuration via environment variables, config.json, or CLI
