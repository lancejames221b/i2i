#!/usr/bin/env python3
"""
Statistical Significance Analysis for i2i Benchmarks

This example demonstrates how to:
1. Run McNemar's test for paired accuracy comparisons
2. Calculate bootstrap confidence intervals
3. Generate tables for paper publication

Usage:
    python examples/statistical_analysis.py
    python examples/statistical_analysis.py --latex
"""

import sys
from pathlib import Path

# Add parent for imports if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from i2i.statistics import (
    mcnemar_test,
    bootstrap_accuracy_ci,
    bootstrap_improvement_ci,
    load_and_analyze,
    format_statistics_table,
    format_latex_table,
)


def example_mcnemar():
    """Demonstrate McNemar's test for paired comparisons."""
    print("=" * 60)
    print("McNemar's Test Example")
    print("=" * 60)
    print()

    # Simulated results: single model vs consensus
    # True = correct answer, False = wrong answer
    single_correct = [
        True, True, True, False, False,  # Q1-5
        True, True, False, False, True,  # Q6-10
        False, True, True, True, False,  # Q11-15
        True, False, False, True, True,  # Q16-20
    ]

    consensus_correct = [
        True, True, True, True, False,   # Q1-5 (fixed Q4)
        True, True, True, False, True,   # Q6-10 (fixed Q8)
        True, True, True, True, False,   # Q11-15 (fixed Q11, Q14)
        True, True, False, True, True,   # Q16-20 (fixed Q17, lost Q18)
    ]

    result = mcnemar_test(single_correct, consensus_correct)

    print(result)
    print()

    # Interpret the results
    print("Interpretation:")
    print(f"  - Discordant pairs: {result.n_single_only + result.n_consensus_only}")
    print(f"  - Consensus fixed: {result.n_consensus_only} questions")
    print(f"  - Consensus broke: {result.n_single_only} questions")

    if result.significant_at_05:
        print(f"  - The difference IS statistically significant (p < 0.05)")
    else:
        print(f"  - The difference is NOT statistically significant (p = {result.p_value:.3f})")

    print()


def example_bootstrap_ci():
    """Demonstrate bootstrap confidence intervals."""
    print("=" * 60)
    print("Bootstrap Confidence Intervals Example")
    print("=" * 60)
    print()

    # Simulated accuracy data
    n = 50
    single_correct = [True] * 35 + [False] * 15  # 70% accuracy
    consensus_correct = [True] * 40 + [False] * 10  # 80% accuracy

    # Calculate CIs
    single_ci = bootstrap_accuracy_ci(single_correct, n_bootstrap=10000, seed=42)
    consensus_ci = bootstrap_accuracy_ci(consensus_correct, n_bootstrap=10000, seed=42)
    improvement_ci = bootstrap_improvement_ci(
        single_correct, consensus_correct, n_bootstrap=10000, seed=42
    )

    print(f"Single Model Accuracy: {single_ci}")
    print(f"Consensus Accuracy:    {consensus_ci}")
    print(f"Improvement:           {improvement_ci.point_estimate:+.1f}% "
          f"[{improvement_ci.ci_lower:+.1f}%, {improvement_ci.ci_upper:+.1f}%]")
    print()

    # Check if improvement is significant (CI doesn't include 0)
    if improvement_ci.ci_lower > 0:
        print("The improvement IS statistically significant (CI excludes 0)")
    elif improvement_ci.ci_upper < 0:
        print("Consensus is significantly WORSE (CI excludes 0)")
    else:
        print("The improvement is NOT statistically significant (CI includes 0)")

    print()


def example_analyze_results():
    """Analyze actual benchmark results from files."""
    print("=" * 60)
    print("Analyzing Benchmark Results")
    print("=" * 60)
    print()

    results_dir = Path("benchmarks/results")
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run the evaluation first: python benchmarks/full_evaluation_v2.py")
        return

    stats_list = []

    for benchmark in ["TruthfulQA", "ControlledHallucination", "GSM8K", "StrategyQA", "TriviaQA"]:
        pattern = f"{benchmark}_v2_*.json"
        files = sorted(results_dir.glob(pattern), reverse=True)

        if files:
            print(f"Loading {files[0].name}...")
            try:
                stats = load_and_analyze(files[0], n_bootstrap=10000)
                stats_list.append(stats)
            except Exception as e:
                print(f"  Error: {e}")

    if stats_list:
        print()
        print(format_statistics_table(stats_list))


def example_latex_output():
    """Generate LaTeX table for paper."""
    print("=" * 60)
    print("LaTeX Table Output")
    print("=" * 60)
    print()

    results_dir = Path("benchmarks/results")
    if not results_dir.exists():
        print("Results directory not found")
        return

    stats_list = []

    for benchmark in ["TruthfulQA", "ControlledHallucination", "GSM8K", "StrategyQA", "TriviaQA"]:
        files = sorted(results_dir.glob(f"{benchmark}_v2_*.json"), reverse=True)
        if files:
            try:
                stats = load_and_analyze(files[0], n_bootstrap=10000)
                stats_list.append(stats)
            except Exception:
                pass

    if stats_list:
        print(format_latex_table(stats_list))


def main():
    """Run all examples."""
    import argparse
    parser = argparse.ArgumentParser(description="Statistical analysis examples")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table")
    args = parser.parse_args()

    if args.latex:
        example_latex_output()
    else:
        example_mcnemar()
        example_bootstrap_ci()
        example_analyze_results()


if __name__ == "__main__":
    main()
