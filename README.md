# i2i — AI-to-AI Communication Protocol

<p align="center">
  <strong>When AIs See Eye to Eye</strong>
</p>

<p align="center">
  <em>An open protocol for multi-model consensus, cross-verification, intelligent routing, and epistemic classification</em>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
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

## Installation

```bash
pip install i2i-mcip
```

Or install from source:

```bash
git clone https://github.com/lancejames221b/i2i.git
cd i2i
pip install -r requirements.txt
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

---

## Quick Start

### Python API

```python
from i2i import AICP

protocol = AICP()

# 1. Consensus Query — Ask multiple AIs and find agreement
result = await protocol.consensus_query(
    "What are the primary causes of inflation?",
    models=["gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro"]
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
print(result.decision.selected_models)  # ["claude-3-5-sonnet-20241022"]
print(result.responses[0].content)      # The actual code
```

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
- **Claude 3.5 Sonnet** → Best at coding, instruction following
- **GPT-4o** → Strong at math, factual QA
- **o1-preview** → Deep reasoning, complex problems (but slow)
- **Gemini 1.5 Pro** → Great for long context, research
- **Llama on Groq** → Fastest inference, good for simple tasks

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
print(result.decision.selected_models)   # ["claude-3-5-sonnet-20241022"]
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
#   "best_quality": {"model": "claude-3-5-sonnet-20241022", "score": 95},
#   "best_speed": {"model": "mixtral-8x7b-32768", "score": 75, "latency_ms": 100},
#   "best_value": {"model": "claude-3-haiku-20240307", "score": 85, "cost": 0.00025},
#   "balanced": {"model": "gpt-4o", "score": 92}
# }
```

### Learning from Results

The router tracks performance and can update capability scores over time:

```python
# Router logs performance automatically
# You can also manually update based on observed quality
protocol.router.update_capability(
    model_id="gpt-4o",
    task_type=TaskType.MATHEMATICAL,
    new_score=95.0  # Based on observed performance
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
    model="gpt-4o"
)

# Have other models challenge it
challenges = await protocol.challenge_response(
    original,
    challengers=["claude-3-5-sonnet", "gemini-1.5-pro"],
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
    models=["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"],
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
| **OpenAI** | GPT-4o, GPT-4-turbo, o1-preview | ✅ Supported |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus | ✅ Supported |
| **Google** | Gemini 1.5 Pro, Gemini 1.5 Flash | ✅ Supported |
| **Mistral** | Mistral Large, Codestral | ✅ Supported |
| **Groq** | Llama 3.3 70B, Mixtral 8x7B | ✅ Supported |
| **Cohere** | Command R+, Command R | ✅ Supported |

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

### Routing Strategies

| Strategy | Description |
|----------|-------------|
| `BEST_QUALITY` | Select model with highest task score |
| `BEST_SPEED` | Prioritize low-latency models |
| `BEST_VALUE` | Optimize for cost-effectiveness |
| `BALANCED` | Balance quality, speed, and cost |
| `ENSEMBLE` | Use multiple models, synthesize responses |
| `FALLBACK_CHAIN` | Try models in order until success |

### Task Types

| Type | Examples |
|------|----------|
| `CODE_GENERATION` | "Write a function", "implement", "create class" |
| `MATHEMATICAL` | "Calculate", "solve", "prove theorem" |
| `CREATIVE_WRITING` | "Write a story", "compose poem", "creative" |
| `FACTUAL_QA` | "What is", "who was", "define" |
| `LOGICAL_REASONING` | "Deduce", "if-then", "implies" |
| `RESEARCH` | "Analyze", "compare", "investigate" |
| ... | (20+ task types supported) |

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
└─────────────────────────────────────────────────────────────────┘
```

---

## Contributing

Contributions welcome! Areas of interest:

- **Additional providers**: Azure OpenAI, AWS Bedrock, local models
- **Embedding-based similarity**: Replace word overlap with semantic embeddings
- **Streaming support**: Real-time consensus detection during streaming
- **Web UI**: Interactive dashboard for consensus visualization
- **Benchmarks**: Systematic evaluation on hallucination detection

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
