#!/usr/bin/env python3
"""
Run i2i benchmarks for paper evaluation.

Usage:
    python run_benchmarks.py --suite all
    python run_benchmarks.py --suite factual --limit 10
    python run_benchmarks.py --suite hallucination --models gpt-4o,claude-sonnet-4-5-20250929
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.harness import BenchmarkHarness


DATASET_DIR = Path(__file__).parent / "datasets"


def load_dataset(name: str):
    """Load a dataset by name."""
    path = DATASET_DIR / f"{name}.json"
    if not path.exists():
        print(f"Dataset not found: {path}")
        print("Run: python benchmarks/datasets/prepare_datasets.py")
        sys.exit(1)
    
    with open(path) as f:
        return json.load(f)


async def run_suite(suite: str, models: list, limit: int = None):
    """Run a benchmark suite."""
    harness = BenchmarkHarness(
        models=models,
        single_model=models[0] if models else "gpt-4o"
    )
    
    suites_to_run = []
    
    if suite in ["all", "factual"]:
        suites_to_run.append(("trivia_qa", "factual_qa", "Factual QA (TriviaQA)"))
    
    if suite in ["all", "reasoning"]:
        suites_to_run.append(("gsm8k", "mathematical", "Mathematical Reasoning (GSM8K)"))
        suites_to_run.append(("strategy_qa", "commonsense", "Commonsense Reasoning (StrategyQA)"))
    
    if suite in ["all", "hallucination"]:
        suites_to_run.append(("hallucination_test", "hallucination", "Hallucination Detection"))
    
    results = []
    for dataset_name, task_type, display_name in suites_to_run:
        data = load_dataset(dataset_name)
        result = await harness.run_benchmark(
            data,
            task_type=task_type,
            name=dataset_name,
            limit=limit
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for r in results:
        print(f"\n{r.name}:")
        print(f"  Single Model: {r.single_model_accuracy:.1f}%")
        print(f"  Consensus:    {r.consensus_accuracy:.1f}%")
        print(f"  HIGH level:   {r.high_consensus_accuracy:.1f}%")
        improvement = r.consensus_accuracy - r.single_model_accuracy
        print(f"  Improvement:  {improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Run i2i benchmarks")
    parser.add_argument(
        "--suite",
        choices=["all", "factual", "reasoning", "hallucination"],
        default="all",
        help="Which benchmark suite to run"
    )
    parser.add_argument(
        "--models",
        default="gpt-4o,claude-sonnet-4-5-20250929,gemini-2.0-flash",
        help="Comma-separated list of models"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions per dataset"
    )
    
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    
    print(f"Running i2i benchmarks")
    print(f"Suite: {args.suite}")
    print(f"Models: {models}")
    print(f"Limit: {args.limit or 'none'}")
    
    asyncio.run(run_suite(args.suite, models, args.limit))


if __name__ == "__main__":
    main()
