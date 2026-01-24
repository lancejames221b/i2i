# Mathematical Viability Analysis: Statistical Averaging Across N Runs

**Analysis requested by:** LinkedIn comment from Michael Hotchkiss
**Date:** 2026-01-24

## The Suggestion

> "Been thinking about how results from individual AI models should be considered across 'n' runs as a statistical average. Then you can get an answer set with sort of a standard deviation."

## Executive Summary

**Yes, this is mathematically viable**, but requires careful implementation. The key insight is that text responses must be projected into a metric space (typically via embeddings) before statistical operations become meaningful. This approach offers significant benefits for measuring model consistency and confidence, but comes with cost/latency tradeoffs.

---

## Current i2i Implementation

The current consensus mechanism in `consensus.py`:

1. Queries each model **once**
2. Computes pairwise similarity using Jaccard index on normalized tokens
3. Averages similarities to determine consensus level (HIGH/MEDIUM/LOW/NONE)

```python
# Current: Single query per model
responses = await self.registry.query_multiple(message, models)
similarity = self._compute_similarity(r1.content, r2.content)  # Jaccard
```

**Limitation:** No measure of intra-model variance (how consistent is a single model with itself?)

---

## Mathematical Framework for N-Run Averaging

### The Core Challenge

Text responses are not scalar values—you cannot simply "average" strings like numbers:

```
Response 1: "The capital of France is Paris"
Response 2: "Paris is the capital city of France"
Average: ???
```

### Solution: Embedding Space Projection

Project text into a vector space where statistical operations are well-defined:

```
text → embedding_model → R^d (d-dimensional vector)
```

**In embedding space:**
- Mean (centroid) is computable
- Variance/standard deviation is computable
- Distance metrics (cosine, L2) provide similarity

### Proposed Statistical Model

For model `m` with `n` runs:

```
Let E_m = {e_m^1, e_m^2, ..., e_m^n}  where e_m^i ∈ R^d
```

**Centroid (mean response):**
```
μ_m = (1/n) Σ e_m^i
```

**Intra-model variance:**
```
σ²_m = (1/n) Σ ||e_m^i - μ_m||²
```

**Standard deviation:**
```
σ_m = √(σ²_m)
```

### Interpretation

| Metric | Meaning |
|--------|---------|
| Low σ_m | Model is consistent/confident |
| High σ_m | Model is uncertain/inconsistent |
| μ_m | "Average" semantic position of responses |

---

## Mathematical Benefits

### 1. Confidence Estimation

A model with high intra-run variance is less reliable:

```python
confidence_m = 1 / (1 + σ_m)  # Higher variance → lower confidence
```

### 2. Weighted Consensus

Weight models by inverse variance (more consistent = more weight):

```python
weight_m = 1 / (σ²_m + ε)  # ε prevents division by zero
consensus = Σ(weight_m * μ_m) / Σ(weight_m)
```

This is analogous to **inverse-variance weighting** in meta-analysis.

### 3. Outlier Detection

Responses far from the centroid may be hallucinations:

```python
is_outlier = ||e_m^i - μ_m|| > k * σ_m  # k typically 2-3
```

### 4. Agreement Intervals

Like confidence intervals for numeric data:

```
Agreement zone = {x : ||x - μ_consensus|| < 2σ_consensus}
```

---

## Implementation Approaches

### Approach A: Embedding-Based (Recommended)

```python
async def query_with_variance(
    self,
    query: str,
    model: str,
    n_runs: int = 5,
    temperature: float = 0.7,
) -> StatisticalResponse:

    # Run n queries
    responses = await asyncio.gather(*[
        self.registry.query(message, model, temperature=temperature)
        for _ in range(n_runs)
    ])

    # Embed all responses
    embeddings = [embed(r.content) for r in responses]

    # Compute statistics
    centroid = np.mean(embeddings, axis=0)
    variance = np.var(embeddings, axis=0).mean()  # average variance
    std_dev = np.sqrt(variance)

    # Find response closest to centroid (representative)
    distances = [cosine_distance(e, centroid) for e in embeddings]
    representative_idx = np.argmin(distances)

    return StatisticalResponse(
        representative=responses[representative_idx],
        centroid=centroid,
        std_dev=std_dev,
        all_responses=responses,
        consistency_score=1 / (1 + std_dev),
    )
```

### Approach B: Structured Output Voting

For questions with discrete answers, use majority voting:

```python
def aggregate_structured(responses: List[Dict]) -> Dict:
    """For structured outputs like {"answer": "Paris", "confidence": 0.9}"""
    from collections import Counter

    answers = [r["answer"] for r in responses]
    counter = Counter(answers)

    majority_answer, count = counter.most_common(1)[0]
    agreement_rate = count / len(responses)

    return {
        "answer": majority_answer,
        "agreement_rate": agreement_rate,  # analogous to 1 - std_dev
        "distribution": dict(counter),
    }
```

### Approach C: Token-Level Analysis (Advanced)

If log-probabilities are available:

```python
def token_consistency(log_probs_runs: List[List[float]]) -> float:
    """Measure consistency at the token level across runs."""
    # High variance in token probs = model uncertainty
    variances = np.var(log_probs_runs, axis=0)
    return np.mean(variances)
```

---

## Practical Considerations

### Temperature Requirements

**Critical:** At `temperature=0`, responses are deterministic—no variance to measure.

| Temperature | Effect |
|-------------|--------|
| 0.0 | Identical responses (useless for variance) |
| 0.3-0.5 | Slight variation, good for consistency check |
| 0.7-1.0 | Natural variation, good for full analysis |
| >1.0 | High variance, may obscure signal |

**Recommendation:** Use `temperature=0.7` for statistical analysis.

### Cost Multiplier

Running `n` queries multiplies API costs:

| N runs | Models | Total queries | Cost multiplier |
|--------|--------|---------------|-----------------|
| 1 | 3 | 3 | 1x |
| 3 | 3 | 9 | 3x |
| 5 | 3 | 15 | 5x |
| 10 | 3 | 30 | 10x |

**Mitigation strategies:**
- Use smaller/cheaper models for variance estimation
- Cache embeddings
- Adaptive n: start with n=3, increase if variance is high

### Latency Impact

Parallelization is key:

```python
# Bad: Sequential (n * latency)
for i in range(n):
    responses.append(await query(model))

# Good: Parallel (max_latency, not sum)
responses = await asyncio.gather(*[query(model) for _ in range(n)])
```

---

## Proposed Schema Extension

```python
class StatisticalConsensusResult(BaseModel):
    """Enhanced consensus result with statistical measures."""

    # Existing fields
    query: str
    consensus_level: ConsensusLevel
    consensus_answer: Optional[str]

    # New statistical fields
    n_runs_per_model: int
    model_statistics: Dict[str, ModelStatistics]
    weighted_consensus: Optional[np.ndarray]  # Embedding centroid
    overall_confidence: float  # Based on inter/intra variance

class ModelStatistics(BaseModel):
    """Per-model statistics from n runs."""
    model: str
    n_runs: int
    centroid_embedding: List[float]
    intra_model_std_dev: float
    consistency_score: float  # 1 / (1 + std_dev)
    representative_response: Response
    outlier_count: int
```

---

## Theoretical Grounding

This approach has precedent in:

1. **Ensemble Methods (ML):** Multiple models/runs averaged for robustness
2. **Bayesian Deep Learning:** Epistemic uncertainty from prediction variance
3. **Monte Carlo Dropout:** Variance from multiple forward passes
4. **Meta-Analysis:** Inverse-variance weighting across studies

The key insight from statistics: **variance estimates require multiple samples**. A single query gives no information about model confidence in its own answer.

---

## Recommendation

**Implement as an optional mode** with clear cost/benefit documentation:

```python
# Standard (current behavior)
result = await aicp.consensus_query(query, models)

# Statistical mode (new)
result = await aicp.consensus_query(
    query,
    models,
    statistical_mode=True,
    n_runs=5,
    temperature=0.7,
)
```

This preserves backward compatibility while offering enhanced analysis for users who need confidence estimation.

---

## Conclusion

Michael Hotchkiss's suggestion is **mathematically sound** and would add significant value to i2i:

| Aspect | Viability | Notes |
|--------|-----------|-------|
| Mathematical foundation | ✅ Strong | Well-established in statistics/ML |
| Technical implementation | ✅ Feasible | Requires embedding model, parallelization |
| Cost/benefit | ⚠️ Tradeoff | 3-5x cost for confidence estimates |
| Practical value | ✅ High | Distinguishes confident vs. uncertain consensus |

The key transformation: **text → embeddings → statistical operations become valid**.
