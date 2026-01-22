# RFC: MCIP - Multi-model Consensus and Inference Protocol

**RFC Number:** MCIP-001
**Title:** Multi-model Consensus and Inference Protocol
**Status:** Draft
**Category:** Standards Track
**Authors:** Lance James, Claude (Anthropic)
**Created:** 2026-01-21
**Version:** 0.2.0

---

## Abstract

This document specifies the Multi-model Consensus and Inference Protocol (MCIP), a standardized protocol for orchestrating communication between multiple Large Language Model (LLM) systems. MCIP enables consensus-based querying, cross-verification of AI outputs, epistemic classification of queries, and structured multi-model debates. The protocol addresses the growing need for reliable, verifiable AI outputs by leveraging architectural diversity across AI systems.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Terminology](#2-terminology)
3. [Protocol Overview](#3-protocol-overview)
4. [Message Format](#4-message-format)
5. [Consensus Mechanism](#5-consensus-mechanism)
6. [Verification Protocol](#6-verification-protocol)
7. [Epistemic Classification](#7-epistemic-classification)
8. [Intelligent Model Routing](#8-intelligent-model-routing)
9. [Provider Abstraction Layer](#9-provider-abstraction-layer)
10. [Security Considerations](#10-security-considerations)
11. [Implementation Requirements](#11-implementation-requirements)
12. [Future Work](#12-future-work)
13. [References](#13-references)

---

## 1. Introduction

### 1.1 Motivation

Current AI systems operate in isolation, with users relying on single-model outputs without independent verification. This creates several problems:

1. **Single points of failure**: Model-specific biases, hallucinations, or errors propagate unchecked
2. **No confidence calibration**: Users cannot assess output reliability
3. **Epistemic opacity**: Users cannot distinguish answerable questions from underdetermined or "idle" ones
4. **Limited accountability**: No mechanism for AI systems to challenge or verify each other

### 1.2 Goals

MCIP aims to:

- Define a standard message format for inter-AI communication
- Establish consensus mechanisms for multi-model agreement detection
- Enable cross-verification of claims between AI systems
- Provide epistemic classification of queries to prevent wasted computation
- Create a provider-agnostic abstraction layer for model orchestration

### 1.3 Scope

This specification covers:
- Message schemas and response formats
- Consensus detection algorithms
- Verification protocols
- Epistemic classification taxonomy
- Provider adapter requirements

Out of scope:
- Specific AI model implementations
- Network transport protocols (assumes HTTP/HTTPS)
- Authentication mechanisms (implementation-specific)

---

## 2. Terminology

**Consensus Query**: A query sent to multiple AI models with the expectation of analyzing agreement levels.

**Cross-Verification**: The process of having one or more AI models verify claims made by another.

**Epistemic Status**: Classification of a question's answerability (ANSWERABLE, UNCERTAIN, UNDERDETERMINED, IDLE, MALFORMED).

**Idle Question**: A well-formed question whose answer would not guide any action or decision.

**Model Divergence**: A significant disagreement between AI model outputs on the same query.

**Provider**: An AI service offering one or more models (e.g., OpenAI, Anthropic, Google).

**Synthesis**: The process of combining multiple model outputs into a unified response.

---

## 3. Protocol Overview

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MCIP Protocol Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Consensus   │  │    Cross-    │  │      Epistemic         │ │
│  │   Engine     │  │ Verification │  │    Classification      │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Message Schema Layer                          │
│              (Standardized Request/Response Format)              │
├─────────────────────────────────────────────────────────────────┤
│                   Provider Abstraction Layer                     │
│  ┌────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌─────────────┐ │
│  │ OpenAI │ │Anthropic │ │ Google │ │Mistral │ │   Others    │ │
│  └────────┘ └──────────┘ └────────┘ └────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Core Operations

| Operation | Description |
|-----------|-------------|
| `QUERY` | Standard prompt/question to one or more models |
| `CONSENSUS_QUERY` | Query with multi-model agreement analysis |
| `VERIFY` | Request verification of a claim |
| `CHALLENGE` | Request critical analysis of a response |
| `CLASSIFY` | Determine epistemic status of a question |
| `SYNTHESIZE` | Combine multiple responses into one |
| `DEBATE` | Structured multi-round discussion |

---

## 4. Message Format

### 4.1 Request Message

All MCIP requests MUST conform to the following schema:

```json
{
  "id": "uuid-v4",
  "type": "QUERY | VERIFY | CHALLENGE | CLASSIFY | SYNTHESIZE",
  "content": "string",
  "sender": "model-identifier | null",
  "recipient": "model-identifier | null",
  "context": ["array of previous messages"],
  "metadata": {
    "timestamp": "ISO-8601",
    "priority": "LOW | NORMAL | HIGH",
    "timeout_ms": "integer",
    "custom": {}
  },
  "target_message_id": "uuid | null"
}
```

#### 4.1.1 Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | UUID v4 | Yes | Unique message identifier |
| `type` | Enum | Yes | Message type (see Section 3.2) |
| `content` | String | Yes | The prompt, claim, or question |
| `sender` | String | No | Originating model identifier |
| `recipient` | String | No | Target model (null = broadcast) |
| `context` | Array | No | Conversation history |
| `metadata` | Object | No | Additional metadata |
| `target_message_id` | UUID | No | For VERIFY/CHALLENGE: message being examined |

### 4.2 Response Message

```json
{
  "id": "uuid-v4",
  "message_id": "uuid-v4",
  "model": "provider/model-name",
  "content": "string",
  "confidence": "VERY_HIGH | HIGH | MEDIUM | LOW | VERY_LOW",
  "reasoning": "string | null",
  "caveats": ["array of strings"],
  "metadata": {
    "timestamp": "ISO-8601",
    "input_tokens": "integer",
    "output_tokens": "integer",
    "latency_ms": "float"
  }
}
```

#### 4.2.1 Confidence Levels

| Level | Description | Typical Indicators |
|-------|-------------|-------------------|
| `VERY_HIGH` | Near certainty | "definitely", "certainly", "I'm confident" |
| `HIGH` | Strong confidence | "I believe", "likely", "probably" |
| `MEDIUM` | Moderate confidence | "I think", "possibly", "might" |
| `LOW` | Weak confidence | "I'm not sure", "uncertain" |
| `VERY_LOW` | Minimal confidence | "I don't know", "impossible to determine" |

---

## 5. Consensus Mechanism

### 5.1 Consensus Levels

| Level | Threshold | Description |
|-------|-----------|-------------|
| `HIGH` | ≥85% similarity | Strong agreement across models |
| `MEDIUM` | 60-84% similarity | Moderate agreement |
| `LOW` | 30-59% similarity | Weak agreement |
| `NONE` | <30% similarity | No meaningful agreement |
| `CONTRADICTORY` | Active contradiction | Models explicitly disagree |

### 5.2 Similarity Calculation

Implementations MUST support semantic similarity measurement. The reference implementation uses:

1. **Text Normalization**: Lowercase, tokenize, remove stop words
2. **Similarity Metric**: Jaccard similarity on meaningful tokens
3. **Semantic Enhancement** (RECOMMENDED): Embedding-based cosine similarity

```
similarity(R1, R2) = |tokens(R1) ∩ tokens(R2)| / |tokens(R1) ∪ tokens(R2)|
```

### 5.3 Consensus Result

```json
{
  "query": "string",
  "models_queried": ["array of model identifiers"],
  "responses": ["array of Response objects"],
  "consensus_level": "HIGH | MEDIUM | LOW | NONE | CONTRADICTORY",
  "consensus_answer": "string | null",
  "divergences": [
    {
      "models": ["model-a", "model-b"],
      "similarity": 0.35,
      "summary": "description of divergence"
    }
  ],
  "agreement_matrix": {
    "model-a": {"model-b": 0.85, "model-c": 0.72},
    "model-b": {"model-a": 0.85, "model-c": 0.68}
  },
  "clusters": [["model-a", "model-b"], ["model-c"]]
}
```

### 5.4 Synthesis Rules

When consensus_level is HIGH or MEDIUM, implementations SHOULD synthesize a unified answer:

1. Identify common claims across responses
2. Note any meaningful differences in framing
3. Present the most complete accurate answer
4. Flag any residual uncertainty

---

## 6. Verification Protocol

### 6.1 Verification Request

To verify a claim, send a VERIFY message with:
- `content`: The claim to verify
- `target_message_id`: (Optional) Reference to original message

### 6.2 Verification Response

```json
{
  "original_claim": "string",
  "original_source": "model-identifier | null",
  "verifiers": ["array of model identifiers"],
  "verified": true | false,
  "confidence": 0.0-1.0,
  "issues_found": ["array of issue descriptions"],
  "corrections": "string | null"
}
```

### 6.3 Verification Verdicts

Verifiers MUST respond with one of:
- `TRUE`: Claim is accurate
- `FALSE`: Claim is inaccurate
- `PARTIALLY_TRUE`: Claim has accurate and inaccurate elements
- `UNVERIFIABLE`: Cannot be determined with available information

### 6.4 Challenge Protocol

The CHALLENGE operation enables adversarial analysis:

```json
{
  "type": "CHALLENGE",
  "content": "response content to challenge",
  "metadata": {
    "challenge_type": "general | factual | logical | ethical"
  }
}
```

Challengers MUST provide:
1. `VALIDITY`: Is the response fundamentally sound?
2. `WEAKNESSES`: Specific errors or issues
3. `COUNTERARGUMENTS`: Alternative perspectives
4. `IMPROVEMENT`: Suggested enhancements

---

## 7. Epistemic Classification

### 7.1 Classification Taxonomy

| Type | Definition | Example |
|------|------------|---------|
| `ANSWERABLE` | Can be definitively resolved | "What is the capital of France?" |
| `UNCERTAIN` | Answerable with uncertainty | "Will it rain tomorrow?" |
| `UNDERDETERMINED` | Multiple hypotheses fit equally | "Did Shakespeare write all his plays?" |
| `IDLE` | Well-formed but non-action-guiding | "Is consciousness substrate-independent?" |
| `MALFORMED` | Incoherent or self-contradictory | "What color is the square circle?" |

### 7.2 Classification Response

```json
{
  "question": "string",
  "classification": "ANSWERABLE | UNCERTAIN | UNDERDETERMINED | IDLE | MALFORMED",
  "confidence": 0.0-1.0,
  "reasoning": "string",
  "is_actionable": true | false,
  "competing_hypotheses": ["for UNDERDETERMINED"],
  "uncertainty_sources": ["for UNCERTAIN"],
  "why_idle": "for IDLE questions",
  "suggested_reformulation": "more tractable version | null"
}
```

### 7.3 Actionability

A question is ACTIONABLE if its answer would:
- Change a decision
- Guide behavior
- Resolve a practical uncertainty
- Enable or prevent an action

IDLE questions are coherent but their answers do not change any decision.

### 7.4 Pre-filtering Recommendation

Implementations SHOULD offer quick heuristic classification to filter queries before expensive API calls:

```
IF question starts with factual markers ("what is", "who is", etc.)
  THEN likely ANSWERABLE
IF question contains future markers ("will", "going to", etc.)
  THEN likely UNCERTAIN
IF question contains philosophical markers ("consciousness", "free will", etc.)
  THEN likely IDLE
```

---

## 8. Intelligent Model Routing

### 8.1 Motivation

Different AI models exhibit varying performance characteristics across task types:
- **Code generation**: Some models produce more correct, idiomatic code
- **Mathematical reasoning**: Certain models excel at multi-step calculations
- **Creative writing**: Some models generate more engaging, varied prose
- **Factual QA**: Knowledge cutoff dates and training data affect accuracy

Manual model selection is tedious and error-prone. Intelligent routing automates this process.

### 8.2 Task Classification Taxonomy

| Type | Description | Keywords |
|------|-------------|----------|
| `CODE_GENERATION` | Writing new code | "implement", "write code", "create function" |
| `CODE_REVIEW` | Analyzing existing code | "review this code", "improve", "refactor" |
| `CODE_DEBUGGING` | Finding/fixing bugs | "debug", "fix error", "why doesn't" |
| `MATHEMATICAL` | Calculations and proofs | "calculate", "solve", "prove" |
| `LOGICAL_REASONING` | Deduction and inference | "deduce", "infer", "therefore" |
| `SCIENTIFIC` | Research and methodology | "hypothesis", "experiment", "methodology" |
| `CREATIVE_WRITING` | Fiction, poetry, narrative | "write a story", "poem", "creative" |
| `COPYWRITING` | Marketing and sales content | "ad copy", "slogan", "marketing" |
| `FACTUAL_QA` | Knowledge retrieval | "what is", "who is", "define" |
| `RESEARCH` | Analysis and investigation | "analyze", "compare", "investigate" |
| `SUMMARIZATION` | Content condensation | "summarize", "tldr", "key points" |
| `TRANSLATION` | Language conversion | "translate", "in french" |
| `CHAT` | Conversational interaction | "hello", "thanks", "okay" |

### 8.3 Model Capability Profile

Each model MUST have a capability profile:

```json
{
  "model_id": "claude-3-5-sonnet-20241022",
  "provider": "anthropic",
  "task_scores": {
    "code_generation": 95,
    "code_review": 95,
    "creative_writing": 92,
    "logical_reasoning": 88
  },
  "avg_latency_ms": 800,
  "cost_per_1k_tokens": 0.003,
  "context_window": 200000,
  "max_output_tokens": 8192,
  "supports_vision": true,
  "supports_function_calling": true,
  "supports_json_mode": true,
  "reasoning_depth": 90,
  "creativity_score": 88,
  "instruction_following": 95,
  "factual_accuracy": 88
}
```

### 8.4 Routing Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `BEST_QUALITY` | Select highest task score | Accuracy-critical tasks |
| `BEST_SPEED` | Prioritize low latency | Real-time applications |
| `BEST_VALUE` | Optimize cost-effectiveness | High volume, budget limits |
| `BALANCED` | Weight all factors equally | General use |
| `ENSEMBLE` | Query multiple, synthesize | Critical decisions |
| `FALLBACK_CHAIN` | Try models in order | Reliability requirements |

### 8.5 Routing Decision Format

```json
{
  "query": "string",
  "detected_task": "TaskType",
  "task_confidence": 0.0-1.0,
  "selected_models": ["model_id"],
  "strategy_used": "RoutingStrategy",
  "reasoning": "string",
  "estimated_cost": 0.0,
  "estimated_latency_ms": 0.0,
  "alternatives": [
    {"model": "model_id", "score": 0.0}
  ]
}
```

### 8.6 Model Scoring Algorithm

For a given task type and strategy, the model score is calculated as:

**BEST_QUALITY:**
```
score = (task_score * 0.6) + (reasoning_depth * 0.2) + (factual_accuracy * 0.2)
```

**BEST_SPEED:**
```
latency_score = max(0, 100 - (avg_latency_ms / 100))
score = (latency_score * 0.5) + (task_score * 0.3) + (reliability * 0.2)
```

**BEST_VALUE:**
```
cost_score = max(0, 100 - (cost_per_1k_tokens * 5000))
score = (task_score * 0.4) + (cost_score * 0.4) + (latency_score * 0.2)
```

**BALANCED:**
```
score = (task_score * 0.4) + (latency_score * 0.2) + (cost_score * 0.2) +
        (reasoning_depth * 0.1) + (instruction_following * 0.1)
```

### 8.7 Learning from Results

Implementations SHOULD update capability scores based on observed performance:

```python
new_score = old_score * (1 - blend_factor) + observed_score * blend_factor
```

Where `blend_factor` is typically 0.1-0.3 for gradual adaptation.

### 8.8 Routing Response Format

```json
{
  "decision": { /* RoutingDecision */ },
  "responses": [ /* Response[] */ ],
  "synthesized_response": "string | null",
  "actual_latency_ms": 0.0,
  "actual_cost": 0.0
}
```

---

## 9. Provider Abstraction Layer

### 9.1 Provider Adapter Interface

All provider adapters MUST implement:

```python
class ProviderAdapter(ABC):
    @property
    def provider_name(self) -> str: ...

    @property
    def available_models(self) -> List[str]: ...

    def is_configured(self) -> bool: ...

    async def query(self, message: Message, model: str) -> Response: ...
```

### 9.2 Model Identifiers

Models MUST be identified using the format:
```
{provider}/{model-name}
```

Examples:
- `openai/gpt-4o`
- `anthropic/claude-3-5-sonnet-20241022`
- `google/gemini-1.5-pro`

### 9.3 Required Providers

Conforming implementations MUST support at least:
- OpenAI (GPT-4 family)
- Anthropic (Claude family)

RECOMMENDED additional providers:
- Google (Gemini)
- Mistral
- Groq (Llama)
- Cohere

---

## 10. Security Considerations

### 10.1 Prompt Injection

Cross-model verification provides natural defense against prompt injection:
- Injected instructions unlikely to affect all models identically
- Consensus mechanisms detect anomalous responses
- Challenge protocols can identify manipulated outputs

### 10.2 Data Privacy

Implementations MUST:
- Not log sensitive query content without consent
- Support opt-out of response storage
- Clearly document which providers receive query data

### 10.3 Cost Controls

Implementations SHOULD:
- Provide cost estimation before multi-model queries
- Support budget limits per query/session
- Allow model tier selection (expensive vs. fast)

### 10.4 Rate Limiting

Implementations MUST:
- Respect provider rate limits
- Implement exponential backoff
- Support graceful degradation when providers unavailable

---

## 11. Implementation Requirements

### 11.1 Conformance Levels

**MCIP-Basic**: Single-model queries with standard message format

**MCIP-Consensus**: Basic + multi-model consensus queries

**MCIP-Full**: Consensus + verification + epistemic classification + debate

### 11.2 Required Features by Level

| Feature | Basic | Consensus | Full |
|---------|-------|-----------|------|
| Standard message format | ✓ | ✓ | ✓ |
| Single-model query | ✓ | ✓ | ✓ |
| Multi-model parallel query | | ✓ | ✓ |
| Consensus detection | | ✓ | ✓ |
| Response synthesis | | ✓ | ✓ |
| Cross-verification | | | ✓ |
| Challenge protocol | | | ✓ |
| Epistemic classification | | | ✓ |
| Structured debate | | | ✓ |

### 11.3 Error Handling

Implementations MUST:
- Return partial results when some models fail
- Clearly indicate which models succeeded/failed
- Provide actionable error messages
- Support configurable retry policies

---

## 12. Future Work

### 12.1 Planned Extensions

1. **Streaming Consensus**: Real-time consensus detection during streaming responses
2. **Persistent Divergence Tracking**: Historical analysis of model disagreements
3. **Weighted Consensus**: Model-specific weights based on domain expertise
4. **Federated MCIP**: Cross-organization consensus without sharing prompts
5. **Embedding-based Similarity**: Standard embedding requirements for semantic comparison

### 12.2 Research Directions

1. **Optimal Model Selection**: Predicting which models will agree/disagree
2. **Consensus Prediction**: Estimating consensus level before querying
3. **Epistemic Uncertainty Quantification**: Formal uncertainty bounds
4. **Adversarial Robustness**: Consensus behavior under adversarial prompts

---

## 13. References

1. OpenAI API Documentation. https://platform.openai.com/docs
2. Anthropic API Documentation. https://docs.anthropic.com
3. Google Gemini API. https://ai.google.dev/docs
4. "Constitutional AI" - Bai et al., 2022
5. "Self-Consistency Improves Chain of Thought Reasoning" - Wang et al., 2022

---

## Appendix A: Reference Implementation

A reference implementation in Python is available at:
https://github.com/[TBD]/mcip-python

The implementation provides:
- Full MCIP-Full conformance
- CLI demonstration tool
- Provider adapters for 6 major AI services
- Comprehensive test suite

---

## Appendix B: Example Workflows

### B.1 High-Stakes Query

```python
# For important decisions, use full pipeline
result = await mcip.smart_query(
    "Should we proceed with the merger?",
    require_consensus=True,
    verify_result=True
)

if result.classification.classification == EpistemicType.IDLE:
    print("Warning: This question may not have an actionable answer")

if result.consensus.level not in [ConsensusLevel.HIGH, ConsensusLevel.MEDIUM]:
    print("Warning: Models disagree significantly")
```

### B.2 Fact Verification

```python
# Verify a claim from a document
result = await mcip.verify_claim(
    "The company was founded in 1985",
    verifiers=["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"]
)

if not result.verified:
    print(f"Issues found: {result.issues_found}")
    print(f"Suggested correction: {result.corrections}")
```

### B.3 Pre-filtering Questions

```python
# Save costs by filtering unanswerable questions
quick_class = mcip.quick_classify(user_question)

if quick_class == EpistemicType.IDLE:
    return "This question cannot be definitively answered."
elif quick_class == EpistemicType.MALFORMED:
    return "Please rephrase your question."
else:
    # Proceed with expensive multi-model query
    result = await mcip.consensus_query(user_question)
```

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-01-21 | Initial draft |
| 0.2.0 | 2026-01-22 | Added Intelligent Model Routing (Section 8) |

---

*This specification is released under CC BY 4.0.*
