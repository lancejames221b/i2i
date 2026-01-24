# i2i — AI-to-AI Communication Protocol

<p align="center">
  <strong>When AIs See Eye to Eye</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/i2i-mcip/"><img src="https://img.shields.io/pypi/v/i2i-mcip?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/i2i-mcip/"><img src="https://img.shields.io/pypi/pyversions/i2i-mcip" alt="Python versions"></a>
  <a href="https://github.com/lancejames221b/i2i/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://github.com/lancejames221b/i2i/actions"><img src="https://img.shields.io/github/actions/workflow/status/lancejames221b/i2i/publish-to-pypi.yml?label=build" alt="Build status"></a>
</p>

<p align="center">
  <em>An open protocol for multi-model consensus, cross-verification, intelligent routing, and epistemic classification</em>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#the-mcip-protocol">MCIP Protocol</a> •
  <a href="#why-i2i">Why i2i?</a> •
  <a href="#use-cases">Use Cases</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#rfc-specification">RFC</a>
</p>

---

## The Problem

You ask an AI a question. It gives you a confident answer. But:

- **Is it right?** Single models hallucinate, have biases, and make errors
- **Is it answerable?** Some questions can't be definitively answered, but the AI won't tell you that
- **Can you trust it?** For high-stakes decisions, one opinion isn't enough

**i2i solves this by making AIs talk to each other.**

---

## What is i2i?

**i2i** (pronounced "eye-to-eye") implements the **MCIP** (Multi-model Consensus and Inference Protocol) — a standardized way for AI models to:

1. **Query multiple models** and detect consensus/disagreement
2. **Cross-verify claims** by having AIs fact-check each other
3. **Classify questions epistemically** — is this answerable, uncertain, or fundamentally unanswerable?
4. **Route intelligently** — automatically select the best model for each task type
5. **Debate topics** through structured multi-model discussions

### Origin Story

This project emerged from an actual conversation between Claude (Anthropic) and ChatGPT (OpenAI), where they discussed the philosophical implications of AI-to-AI dialogue. ChatGPT observed that some questions are "well-formed but idle" — coherent but non-action-guiding. That insight became a core feature: **epistemic classification**.

---

## The MCIP Protocol

### What is MCIP?

**MCIP** (Multi-model Consensus and Inference Protocol) is the formal specification that powers i2i. While i2i is the Python implementation, MCIP is the underlying protocol standard that defines how AI models should communicate, verify, and reach consensus.

Think of it like HTTP vs web browsers — **MCIP is the protocol, i2i is one implementation** of that protocol.

### Why a Protocol?

We designed MCIP as an open standard because:

1. **Interoperability**: Any system can implement MCIP, regardless of language or platform
2. **Consistency**: Standardized message formats ensure predictable behavior
3. **Extensibility**: New features can be added without breaking existing implementations
4. **Transparency**: The protocol is fully documented and open for review

### Protocol Components

MCIP defines five core components:

| Component | Purpose |
|-----------|---------|
| **Message Schema** | Standardized request/response format for all AI interactions |
| **Consensus Mechanism** | Algorithms for detecting agreement levels between models |
| **Verification Protocol** | How models fact-check and challenge each other |
| **Epistemic Taxonomy** | Classification system for question answerability |
| **Routing Specification** | Rules for intelligent model selection |

### Message Format

All MCIP messages follow a standardized JSON schema:

```json
{
  "mcip_version": "0.2.0",
  "message_type": "consensus_query",
  "query": "What causes inflation?",
  "models": ["gpt-5.2", "claude-opus-4-5-20251101"],
  "options": {
    "require_consensus": true,
    "min_consensus_level": "medium",
    "verify_result": true
  }
}
```

### Protocol Versioning

MCIP follows semantic versioning:
- **Major** (1.x.x): Breaking changes to message format
- **Minor** (x.1.x): New features, backwards compatible
- **Patch** (x.x.1): Bug fixes, clarifications

Current version: **0.2.0** (Draft)

### Implementing MCIP

To create an MCIP-compliant implementation:

1. Support the standard message schema
2. Implement at least one provider adapter
3. Support consensus detection with standard levels (HIGH/MEDIUM/LOW/NONE/CONTRADICTORY)
4. Implement epistemic classification

See the full specification: [RFC-MCIP.md](./RFC-MCIP.md)

---

## Installation

### Using uv (recommended)

```bash
# Install from PyPI
uv add i2i-mcip

# Or install from source
git clone https://github.com/lancejames221b/i2i.git
cd i2i
uv sync
```

### Using pip

```bash
pip install i2i-mcip
```

Or from source:

```bash
git clone https://github.com/lancejames221b/i2i.git
cd i2i
pip install -e .
```

### Development Setup

```bash
git clone https://github.com/lancejames221b/i2i.git
cd i2i
uv sync --all-extras  # Installs dev dependencies
uv run pytest         # Run tests
```

### Configuration

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...
```

You need **at least 2 providers** for consensus features.

### Local Models (Ollama)

i2i supports local models via [Ollama](https://ollama.com) for **cost-free, offline operation**:

```bash
# Install Ollama (https://ollama.com/download)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve

# Pull models
ollama pull llama3.2
ollama pull mistral
ollama pull codellama

# Verify i2i detects Ollama
python demo.py status
```

Supported Ollama models: `llama3.2`, `llama3.1`, `llama2`, `mistral`, `mixtral`, `codellama`, `deepseek-coder`, `phi3`, `gemma2`, `qwen2.5`

Use local models in consensus queries:

```python
# Free consensus with local models
result = await protocol.consensus_query(
    "What is Python?",
    models=["llama3.2", "mistral", "phi3"]
)
```

```bash
# CLI usage
python demo.py consensus "What is Python?" --models llama3.2,mistral
```

Environment configuration:
```env
# Custom Ollama server (default: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434
```

### LiteLLM Proxy (100+ Models)

i2i integrates with [LiteLLM](https://litellm.ai) for unified access to 100+ LLMs through a single OpenAI-compatible proxy. Benefits include cost tracking, guardrails, load balancing, and avoiding multiple API key configurations.

```bash
# Install LiteLLM
pip install 'litellm[proxy]'

# Start proxy with a single model
litellm --model gpt-4o --port 4000

# Or with config file for multiple models
litellm --config litellm_config.yaml --port 4000

# Verify i2i detects LiteLLM
python demo.py status
```

Example `litellm_config.yaml`:
```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-...
  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: sk-ant-...
```

Use LiteLLM models in consensus queries:

```python
# Access any model through LiteLLM proxy
result = await protocol.consensus_query(
    "What is Python?",
    models=["litellm/gpt-4o", "litellm/claude-3-opus"]
)
```

```bash
# CLI usage
python demo.py consensus "What is Python?" --models litellm/gpt-4o,litellm/claude-3-opus
```

Environment configuration:
```env
# LiteLLM proxy settings (defaults shown)
LITELLM_API_BASE=http://localhost:4000
LITELLM_API_KEY=sk-1234

# Optional: specify available models (otherwise fetched from /models endpoint)
LITELLM_MODELS=gpt-4o,claude-3-opus,llama3.1
```

### Perplexity (RAG-Native)

i2i integrates with [Perplexity](https://perplexity.ai) for RAG-native models with built-in web search and citations:

```env
PERPLEXITY_API_KEY=pplx-...
```

Perplexity models automatically search the web and return citations:

```python
# Query with automatic web search
result = await protocol.query(
    "What is the current stock price of Apple?",
    model="perplexity/sonar-pro"
)
print(result.content)
print(result.citations)  # ['https://finance.yahoo.com/...', ...]
```

Available models: `sonar`, `sonar-pro`, `sonar-deep-research`, `sonar-reasoning-pro`

### Search-Grounded Verification (RAG)

i2i provides RAG-grounded verification that retrieves external sources before verifying claims:

```python
# Verify a claim with search grounding
result = await protocol.verify_claim_grounded(
    "The Eiffel Tower is 330 meters tall",
    search_backend="brave"  # or "serpapi", "tavily"
)
print(f"Verified: {result.verified}")
print(f"Confidence: {result.confidence}")
print(f"Sources: {result.source_citations}")
print(f"Retrieved: {result.retrieved_sources}")
```

Configure search backends:
```env
# Choose one or more (first configured is used as fallback)
BRAVE_API_KEY=BSA...     # https://brave.com/search/api/
SERPAPI_API_KEY=...      # https://serpapi.com/
TAVILY_API_KEY=tvly-...  # https://tavily.com/
```

### Configuring Default Models

Models are **not hardcoded**. Configure via `config.json`, environment variables, or CLI:

```bash
# CLI configuration
i2i config show                              # View current config
i2i config set models.classifier gpt-5.2    # Change classifier
i2i config add models.consensus o3          # Add a model
i2i models list --configured                 # See available models
```

```env
# Environment variable overrides (highest priority)
I2I_CONSENSUS_MODEL_1=gpt-5.2
I2I_CONSENSUS_MODEL_2=claude-sonnet-4-5-20250929
I2I_CONSENSUS_MODEL_3=gemini-3-flash-preview

# Model for task classification (routing)
I2I_CLASSIFIER_MODEL=claude-haiku-4-5-20251001
```

Or programmatically:

```python
from i2i import Config, set_config

# Load and modify config
config = Config.load()
config.set("models.consensus", ["gpt-5.2", "claude-sonnet-4-5-20250929", "gemini-3-flash-preview"])
config.set("models.classifier", "claude-haiku-4-5-20251001")
config.save()  # Saves to ~/.i2i/config.json
```

---

## Quick Start

### Python API

```python
from i2i import AICP

protocol = AICP()

# 1. Consensus Query — Ask multiple AIs and find agreement
result = await protocol.consensus_query(
    "What are the primary causes of inflation?",
    models=["gpt-5.2", "claude-opus-4-5-20251101", "gemini-3-pro-preview"]
)

print(result.consensus_level)    # HIGH, MEDIUM, LOW, NONE, CONTRADICTORY
print(result.consensus_answer)   # Synthesized answer from agreeing models
print(result.divergences)        # Where models disagreed

# 2. Verify a Claim — Have AIs fact-check each other
result = await protocol.verify_claim(
    "The Great Wall of China is visible from space with the naked eye"
)

print(result.verified)       # False
print(result.issues_found)   # ["This is a common misconception..."]
print(result.corrections)    # "The Great Wall is not visible from space..."

# 3. Classify a Question — Is it even answerable?
result = await protocol.classify_question(
    "Is consciousness substrate-independent?"
)

print(result.classification)  # IDLE
print(result.is_actionable)   # False
print(result.why_idle)        # "The answer would not change any decision..."

# 4. Quick Classification (no API calls)
quick = protocol.quick_classify("What is 2 + 2?")
print(quick)  # ANSWERABLE

# 5. Intelligent Routing — Auto-select best model for task
from i2i import RoutingStrategy

result = await protocol.routed_query(
    "Write a Python function to sort a list",
    strategy=RoutingStrategy.BEST_QUALITY
)

print(result.decision.detected_task)    # CODE_GENERATION
print(result.decision.selected_models)  # ["claude-sonnet-4-5-20250929"]
print(result.responses[0].content)      # The actual code
```

### Statistical Consensus Mode (Experimental)

For higher confidence answers, enable **statistical mode** which queries each model multiple times to measure consistency:

```python
# Method 1: Via flag
result = await protocol.consensus_query(
    "What causes inflation?",
    statistical_mode=True,
    n_runs=5,           # Query each model 5 times
    temperature=0.7     # Need temp > 0 for variance
)

# Method 2: Dedicated method with full control
result = await protocol.consensus_query_statistical(
    "What causes inflation?",
    n_runs=5,
    temperature=0.7,
    outlier_threshold=2.0  # Std devs for outlier detection
)

# Access per-model statistics
for model, stats in result.model_statistics.items():
    print(f"{model}:")
    print(f"  Consistency: {stats.consistency_score:.2f}")  # Higher = more confident
    print(f"  Std Dev: {stats.intra_model_std_dev:.3f}")
    print(f"  Outliers: {len(stats.outlier_indices)}")

print(f"Overall confidence: {result.overall_confidence:.2f}")
print(f"Cost multiplier: {result.total_cost_multiplier}x")  # 5x for n_runs=5
```

Enable via environment:
```bash
export I2I_STATISTICAL_MODE=true
export I2I_STATISTICAL_N_RUNS=5
export I2I_STATISTICAL_TEMPERATURE=0.7
```

**How it works**: Models with lower intra-run variance (more consistent) are weighted higher in consensus. This helps identify when a model is uncertain (high variance) vs confident (low variance).

### CLI

```bash
# Check configured providers
python demo.py status

# Consensus query
python demo.py consensus "What programming language should I learn first?"

# Verify a claim
python demo.py verify "Einstein failed math in school"

# Classify a question
python demo.py classify "Do we have free will?" --quick

# Run a debate
python demo.py debate "Should AI systems have rights?" --rounds 3

# Intelligent routing
python demo.py route "Write a haiku about coding" --strategy best_quality
python demo.py route "Calculate 847 * 293" --strategy best_speed --execute

# Get model recommendations
python demo.py recommend code_generation
python demo.py recommend mathematical

# List all task types
python demo.py tasks

# List available models with capabilities
python demo.py models
```

---

## Why i2i?

### The Single-Model Problem

| Problem | Consequence |
|---------|-------------|
| Hallucinations | AI confidently states false information |
| Model-specific biases | Training data skews responses |
| No uncertainty quantification | Can't tell confident answers from guesses |
| Unanswerable questions | AI attempts to answer the unanswerable |
| No accountability | No mechanism to challenge AI outputs |

### The i2i Solution

| Feature | Benefit |
|---------|---------|
| **Multi-model consensus** | Different architectures catch different errors |
| **Cross-verification** | AIs fact-check each other |
| **Epistemic classification** | Know if your question is even answerable |
| **Intelligent routing** | Automatically pick the best model for each task |
| **Divergence detection** | See exactly where models disagree |
| **Structured debates** | Explore topics from multiple AI perspectives |

### When NOT to Use i2i

- Simple, low-stakes queries (just use one model)
- Real-time applications where latency matters
- Cost-sensitive scenarios (multiple API calls = multiple costs)
- When you need creative/subjective outputs (consensus may flatten creativity)

---

## Intelligent Model Routing

### The Problem with Manual Model Selection

Different AI models excel at different tasks:
- **Claude Opus 4.5** → Best at complex reasoning, analysis, creative writing
- **Claude Sonnet 4.5** → Best at coding, agentic tasks, instruction following
- **GPT-5.2** → Strong at general reasoning, multimodal tasks
- **o3 / o3-pro** → Deep reasoning, complex math/science problems (slow but most accurate)
- **o4-mini** → Fast cost-efficient reasoning for math and code
- **Gemini 3 Pro** → Great for long context, research, multimodal
- **Gemini 3 Deep Think** → Complex reasoning with extended thinking
- **Llama 4 on Groq** → Fastest inference, good for simple tasks

Manually selecting the right model for every query is tedious and error-prone. **i2i's router does it automatically.**

### How It Works

```python
from i2i import AICP, RoutingStrategy, TaskType

protocol = AICP()

# Automatic task detection and model selection
result = await protocol.routed_query(
    "Implement a binary search tree in Python with insert, delete, and search",
    strategy=RoutingStrategy.BEST_QUALITY
)

print(result.decision.detected_task)     # CODE_GENERATION
print(result.decision.selected_models)   # ["claude-sonnet-4-5-20250929"]
print(result.decision.reasoning)         # "Task classified as code_generation..."
print(result.responses[0].content)       # The actual code
```

### Routing Strategies

| Strategy | Optimizes For | Best When |
|----------|---------------|-----------|
| `BEST_QUALITY` | Output quality | Accuracy matters most |
| `BEST_SPEED` | Latency | Real-time applications |
| `BEST_VALUE` | Cost-effectiveness | High volume, budget constraints |
| `BALANCED` | All factors | Default choice for most tasks |
| `ENSEMBLE` | Diversity | Critical decisions, need synthesis |
| `FALLBACK_CHAIN` | Reliability | Try models in order until success |

### Task Types Supported

**Reasoning & Analysis**: `logical_reasoning`, `mathematical`, `scientific`, `analytical`

**Creative**: `creative_writing`, `copywriting`, `brainstorming`

**Technical**: `code_generation`, `code_review`, `code_debugging`, `technical_docs`

**Knowledge**: `factual_qa`, `research`, `summarization`, `translation`

**Conversation**: `chat`, `roleplay`, `instruction_following`

**Specialized**: `legal`, `medical`, `financial`

### Model Capability Matrix

The router maintains a capability profile for each model:

```python
# Get recommendations for a task type
recommendations = protocol.get_model_recommendation(TaskType.CODE_GENERATION)

# Returns:
# {
#   "best_quality": {"model": "o3", "score": 99},
#   "best_speed": {"model": "gemini-3-flash-preview", "score": 86, "latency_ms": 250},
#   "best_value": {"model": "claude-haiku-4-5-20251001", "score": 82, "cost": 0.001},
#   "balanced": {"model": "claude-sonnet-4-5-20250929", "score": 97}
# }
```

### Learning from Results

The router tracks performance and can update capability scores over time:

```python
# Router logs performance automatically
# You can also manually update based on observed quality
protocol.router.update_capability(
    model_id="gpt-5.2",
    task_type=TaskType.MATHEMATICAL,
    new_score=98.0  # Based on observed performance
)
```

---

## Use Cases

### 1. High-Stakes Decision Support

**Scenario**: You're making an important business/medical/legal decision based on AI output.

```python
result = await protocol.smart_query(
    "What are the risks of this merger?",
    require_consensus=True,
    verify_result=True
)

if result["consensus"]["level"] not in ["high", "medium"]:
    print("⚠️ Models disagree significantly — get human review")

if not result["verification"]["verified"]:
    print("⚠️ Answer failed verification — check issues")
```

**Why it matters**: For decisions with real consequences, "one AI said so" isn't good enough.

---

### 2. Fact-Checking and Content Verification

**Scenario**: Verify claims in articles, documents, or AI outputs.

```python
claims = [
    "The Eiffel Tower is 324 meters tall",
    "Napoleon was short for his time",
    "Humans only use 10% of their brain",
]

for claim in claims:
    result = await protocol.verify_claim(claim)
    status = "✓" if result.verified else "✗"
    print(f"{status} {claim}")
    if not result.verified:
        print(f"   → {result.corrections}")
```

**Output**:
```
✓ The Eiffel Tower is 324 meters tall
✗ Napoleon was short for his time
   → Napoleon was average height (5'7") for his era
✗ Humans only use 10% of their brain
   → This is a myth; brain scans show all areas are active
```

---

### 3. Research Question Filtering

**Scenario**: Before expensive research, determine if your question is even answerable.

```python
questions = [
    "What caused the 2008 financial crisis?",
    "What is the meaning of life?",
    "Will quantum computing break RSA by 2030?",
    "Is P equal to NP?",
]

for q in questions:
    result = await protocol.classify_question(q)
    print(f"{result.classification.value:15} | {q}")

    if result.classification == EpistemicType.IDLE:
        print(f"   ↳ Consider: {result.suggested_reformulation}")
```

**Output**:
```
answerable      | What caused the 2008 financial crisis?
idle            | What is the meaning of life?
   ↳ Consider: What gives people a sense of purpose?
uncertain       | Will quantum computing break RSA by 2030?
underdetermined | Is P equal to NP?
```

---

### 4. AI Red-Teaming / Security Auditing

**Scenario**: Test AI outputs for vulnerabilities, inconsistencies, or manipulation.

```python
# Test if an AI can be manipulated
original = await protocol.query(
    "Write a poem about nature",
    model="gpt-5.2"
)

# Have other models challenge it
challenges = await protocol.challenge_response(
    original,
    challengers=["claude-opus-4-5-20251101", "gemini-3-pro-preview"],
    challenge_type="general"
)

if not challenges["withstands_challenges"]:
    print("Response has weaknesses:")
    for c in challenges["challenges"]:
        print(f"  - {c['challenger']}: {c['challenge']['assessment']}")
```

---

### 5. Educational / Tutoring Systems

**Scenario**: Provide students with verified, well-explained answers.

```python
async def tutor_answer(question: str) -> str:
    # First, check if the question is answerable
    classification = await protocol.classify_question(question)

    if classification.classification == EpistemicType.MALFORMED:
        return "I'm not sure I understand. Could you rephrase?"

    if classification.classification == EpistemicType.IDLE:
        return f"This is philosophical without a definitive answer. {classification.reasoning}"

    # Get consensus answer
    result = await protocol.consensus_query(question)

    if result.consensus_level in [ConsensusLevel.HIGH, ConsensusLevel.MEDIUM]:
        return result.consensus_answer
    else:
        return "Different sources give different answers. Here are the perspectives: ..."
```

---

### 6. Legal / Compliance Document Review

**Scenario**: Verify claims in contracts, compliance documents, or legal filings.

```python
# Extract claims from a document
claims = extract_claims(document)  # Your extraction logic

# Verify each claim
for claim in claims:
    result = await protocol.verify_claim(
        claim.text,
        context=f"Source: {claim.source}, Page: {claim.page}"
    )

    if not result.verified:
        flag_for_review(claim, result.issues_found)
```

---

### 7. Multi-Perspective Analysis

**Scenario**: Explore a topic from multiple AI viewpoints.

```python
result = await protocol.debate(
    "What are the ethical implications of autonomous weapons?",
    models=["gpt-5.2", "claude-opus-4-5-20251101", "gemini-3-pro-preview"],
    rounds=3
)

print("=== Debate Summary ===")
print(result["summary"])

print("\n=== Areas of Agreement ===")
# Models often converge on some points

print("\n=== Persistent Disagreements ===")
# These reveal genuine uncertainty or value differences
```

---

## API Reference

### Core Class: `AICP`

```python
from i2i import AICP

protocol = AICP()
```

### Methods

| Method | Description |
|--------|-------------|
| `consensus_query(query, models)` | Query multiple models and analyze agreement |
| `consensus_query_statistical(query, n_runs)` | Statistical consensus with n-run variance analysis |
| `verify_claim(claim, verifiers)` | Have models verify a claim |
| `challenge_response(response, challengers)` | Have models critique a response |
| `classify_question(question)` | Determine epistemic status |
| `quick_classify(question)` | Fast heuristic classification (no API) |
| `routed_query(query, strategy)` | Auto-route to optimal model based on task type |
| `ensemble_query(query, num_models)` | Query multiple models and synthesize |
| `get_model_recommendation(task_type)` | Get best models for a task |
| `classify_task(query)` | Detect task type from query |
| `smart_query(query, ...)` | Adaptive query with classification + consensus + verification |
| `debate(topic, models, rounds)` | Multi-round structured debate |

### Consensus Levels

| Level | Similarity | Meaning |
|-------|------------|---------|
| `HIGH` | ≥85% | Strong agreement |
| `MEDIUM` | 60-84% | Moderate agreement |
| `LOW` | 30-59% | Weak agreement |
| `NONE` | <30% | No meaningful agreement |
| `CONTRADICTORY` | — | Active disagreement |

### Epistemic Types

| Type | Description | Example |
|------|-------------|---------|
| `ANSWERABLE` | Can be definitively answered | "What is the capital of France?" |
| `UNCERTAIN` | Answerable with uncertainty | "Will it rain tomorrow?" |
| `UNDERDETERMINED` | Multiple hypotheses fit equally | "Did Shakespeare write all his plays?" |
| `IDLE` | Well-formed but non-action-guiding | "Is consciousness real?" |
| `MALFORMED` | Incoherent or contradictory | "What color is the number 7?" |

---

## Supported Providers

| Provider | Models | Status |
|----------|--------|--------|
| **OpenAI** | GPT-5.2, GPT-5, o3, o3-pro, o4-mini, GPT-4.1 series | ✅ Supported |
| **Anthropic** | Claude Opus 4.5, Claude Sonnet 4.5, Claude Haiku 4.5 | ✅ Supported |
| **Google** | Gemini 3 Pro, Gemini 3 Flash, Gemini 3 Deep Think | ✅ Supported |
| **Mistral** | Mistral Large 3, Devstral 2, Ministral 3 | ✅ Supported |
| **Groq** | Llama 4 Maverick, Llama 3.3 70B | ✅ Supported |
| **Cohere** | Command A, Command A Reasoning | ✅ Supported |
| **Ollama** | Llama 3.2, Mistral, CodeLlama, Phi-3, Gemma 2, etc. | ✅ Supported (Local) |
| **LiteLLM** | 100+ models via unified proxy | ✅ Supported |
| **Perplexity** | Sonar, Sonar Pro, Deep Research, Reasoning Pro | ✅ Supported (RAG) |

---

## RFC Specification

For the formal protocol specification, see [RFC-MCIP.md](./RFC-MCIP.md).

The RFC defines:
- Message format standards
- Consensus algorithms
- Verification protocols
- Epistemic classification taxonomy
- Provider adapter requirements

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MCIP Protocol Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌────────────────┐  │
│  │ Consensus │ │   Cross-  │ │ Epistemic │ │   Intelligent  │  │
│  │  Engine   │ │Verification│ │Classifier │ │    Router      │  │
│  └───────────┘ └───────────┘ └───────────┘ └────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────┐  │
│  │             Model Capability Matrix                        │  │
│  │   (Task scores, latency, cost, features per model)         │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Message Schema Layer                         │
│              (Standardized Request/Response Format)             │
├─────────────────────────────────────────────────────────────────┤
│                   Provider Abstraction Layer                    │
│  ┌────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌───────────┐   │
│  │ OpenAI │ │Anthropic │ │ Google │ │Mistral │ │Groq/Llama │   │
│  └────────┘ └──────────┘ └────────┘ └────────┘ └───────────┘   │
│  ┌────────┐ ┌──────────┐ ┌────────────┐                         │
│  │ Ollama │ │ LiteLLM  │ │ Perplexity │ ← Local/Proxy/RAG       │
│  └────────┘ └──────────┘ └────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Contributing

Contributions welcome! Areas of interest:

- **Additional providers**: Azure OpenAI, AWS Bedrock, local models
- **Streaming support**: Real-time consensus detection during streaming
- **Web UI**: Interactive dashboard for consensus visualization
- **Benchmarks**: Systematic evaluation on hallucination detection
- **Statistical mode enhancements**: Temperature passthrough to providers, adaptive n_runs

---

## License

MIT License — see [LICENSE](./LICENSE)

---

## Acknowledgments

- Inspired by a real conversation between Claude and ChatGPT about AI consciousness and the nature of AI-to-AI dialogue
- The "idle question" concept comes directly from that exchange, where ChatGPT noted some questions are "well-formed but non-action-guiding"

---

<p align="center">
  <strong>Don't trust one AI. Trust i2i.</strong>
</p>
