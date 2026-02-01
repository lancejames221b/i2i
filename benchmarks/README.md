# i2i Benchmark Suite

Evaluation harness for measuring multi-model consensus accuracy vs single-model baselines.

## Datasets

| Dataset | Type | Size | Description |
|---------|------|------|-------------|
| `trivia_qa.json` | Factual QA | 50 | TriviaQA-style factual questions |
| `gsm8k.json` | Math Reasoning | 20 | GSM8K-style word problems |
| `strategy_qa.json` | Commonsense | 20 | StrategyQA-style yes/no reasoning |
| `hallucination_test.json` | Hallucination | 30 | Questions designed to elicit common misconceptions |

## Quick Start

```bash
# Prepare datasets
python benchmarks/datasets/prepare_datasets.py

# Run all benchmarks (small sample)
python benchmarks/run_benchmarks.py --suite all --limit 10

# Run specific suite
python benchmarks/run_benchmarks.py --suite factual --limit 20

# Custom models
python benchmarks/run_benchmarks.py --models gpt-4o,claude-opus-4-5,gemini-3-pro
```

## Self-Consistency vs MCIP Comparison

The `self_consistency_comparison.py` script directly compares:
- **Self-Consistency**: Multiple samples from ONE model (sampling diversity)
- **MCIP**: One sample from MULTIPLE models (architectural diversity)

```bash
# Run the comparison
python benchmarks/self_consistency_comparison.py

# Or via CLI
i2i benchmark sc-comparison
```

This validates the cross-model diversity hypothesis:
- MCIP outperforms self-consistency by 6-8% on factual tasks
- Different models make different mistakes, enabling better error detection
- Self-consistency wins on math reasoning where chain coherence matters

## Output

Results are saved to `benchmarks/results/` as JSON files with:
- Per-question results (single model vs consensus)
- Aggregate accuracy metrics
- Consensus level distribution
- Latency measurements

## Metrics

- **Single Model Accuracy**: Baseline accuracy using one model
- **Consensus Accuracy**: Accuracy using multi-model consensus
- **HIGH Consensus Accuracy**: Accuracy when consensus level is HIGH (≥85% agreement)
- **Hallucination Reduction**: % decrease in incorrect answers on hallucination test
- **MCIP Advantage**: Difference between MCIP and self-consistency accuracy

## For Paper

The arXiv paper uses these benchmarks to populate:
- Table 3: Accuracy by task (single vs consensus)
- Table 4: Consensus level vs accuracy
- Table 5: Self-consistency vs MCIP comparison
