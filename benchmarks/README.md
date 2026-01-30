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

## For Paper

The arXiv paper uses these benchmarks to populate Table 3 (accuracy by task) and Table 4 (consensus level vs accuracy).
