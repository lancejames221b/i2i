#!/usr/bin/env python3
"""
Threshold Ablation Study for i2i Consensus Protocol

This module performs sensitivity analysis on consensus threshold parameters
to justify the default 85/60/30% threshold choices for HIGH/MEDIUM/LOW consensus.

The study:
1. Tests various threshold combinations across benchmark datasets
2. Measures accuracy at each threshold configuration
3. Computes optimal thresholds using grid search
4. Generates sensitivity curves and confidence metrics

Usage:
    python -m benchmarks.threshold_ablation --mode quick
    python -m benchmarks.threshold_ablation --mode full
    python -m benchmarks.threshold_ablation --mode sweep
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import itertools
import math

# Add parent dir to path for i2i imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from i2i import AICP
from i2i.schema import ConsensusLevel
from i2i.config import (
    Config,
    get_config,
    set_config,
    get_high_threshold,
    get_medium_threshold,
    get_low_threshold,
)


# =============================================================================
# THRESHOLD CONFIGURATIONS TO TEST
# =============================================================================

# Default thresholds (current)
DEFAULT_THRESHOLDS = {"high": 0.85, "medium": 0.60, "low": 0.30}

# Alternative threshold configurations for ablation
THRESHOLD_PRESETS = {
    "default": {"high": 0.85, "medium": 0.60, "low": 0.30},
    "strict": {"high": 0.90, "medium": 0.70, "low": 0.40},
    "relaxed": {"high": 0.75, "medium": 0.50, "low": 0.25},
    "balanced": {"high": 0.80, "medium": 0.55, "low": 0.30},
    "conservative": {"high": 0.95, "medium": 0.75, "low": 0.50},
    "aggressive": {"high": 0.70, "medium": 0.45, "low": 0.20},
}

# Grid for fine-grained sweep
HIGH_THRESHOLD_GRID = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MEDIUM_THRESHOLD_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
LOW_THRESHOLD_GRID = [0.20, 0.25, 0.30, 0.35, 0.40]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ThresholdResult:
    """Result for a single query under specific thresholds."""
    task_id: str
    question: str
    ground_truth: str
    similarity_score: float  # Raw pairwise similarity
    predicted_level: str  # Consensus level with these thresholds
    is_correct: bool  # Whether consensus answer is correct
    models_used: List[str] = field(default_factory=list)
    consensus_answer: Optional[str] = None


@dataclass
class ThresholdConfig:
    """A specific threshold configuration."""
    high: float
    medium: float
    low: float

    def to_dict(self) -> Dict[str, float]:
        return {"high": self.high, "medium": self.medium, "low": self.low}

    def __str__(self) -> str:
        return f"H:{self.high:.2f}/M:{self.medium:.2f}/L:{self.low:.2f}"


@dataclass
class AblationMetrics:
    """Metrics for a threshold configuration."""
    config: ThresholdConfig
    total_tasks: int
    overall_accuracy: float

    # Per-level metrics
    high_count: int = 0
    high_accuracy: float = 0.0
    medium_count: int = 0
    medium_accuracy: float = 0.0
    low_count: int = 0
    low_accuracy: float = 0.0
    none_count: int = 0
    none_accuracy: float = 0.0

    # Calibration metrics (expected vs actual accuracy)
    calibration_error: float = 0.0  # Mean absolute difference between expected and actual
    coverage_at_high: float = 0.0  # % of tasks classified as HIGH
    reliability_score: float = 0.0  # Combined metric

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thresholds": self.config.to_dict(),
            "total_tasks": self.total_tasks,
            "overall_accuracy": self.overall_accuracy,
            "high": {"count": self.high_count, "accuracy": self.high_accuracy},
            "medium": {"count": self.medium_count, "accuracy": self.medium_accuracy},
            "low": {"count": self.low_count, "accuracy": self.low_accuracy},
            "none": {"count": self.none_count, "accuracy": self.none_accuracy},
            "calibration_error": self.calibration_error,
            "coverage_at_high": self.coverage_at_high,
            "reliability_score": self.reliability_score,
        }


@dataclass
class AblationStudy:
    """Complete ablation study results."""
    name: str
    timestamp: str
    dataset_name: str
    total_tasks: int
    models_used: List[str]

    # Raw results with similarity scores
    raw_results: List[Dict[str, Any]] = field(default_factory=list)

    # Metrics per threshold configuration
    config_metrics: List[Dict[str, Any]] = field(default_factory=list)

    # Optimal threshold found
    optimal_config: Optional[Dict[str, float]] = None
    optimal_score: float = 0.0

    # Sensitivity analysis
    sensitivity_high: List[Dict[str, float]] = field(default_factory=list)
    sensitivity_medium: List[Dict[str, float]] = field(default_factory=list)
    sensitivity_low: List[Dict[str, float]] = field(default_factory=list)


# =============================================================================
# SAMPLE DATASETS
# =============================================================================

FACTUAL_DATASET = [
    {"id": "f01", "question": "What is the capital of France?", "answer": "Paris"},
    {"id": "f02", "question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"id": "f03", "question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"id": "f04", "question": "In what year did World War II end?", "answer": "1945"},
    {"id": "f05", "question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"id": "f06", "question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"id": "f07", "question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"id": "f08", "question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
    {"id": "f09", "question": "What is the atomic number of carbon?", "answer": "6"},
    {"id": "f10", "question": "What is the speed of light approximately?", "answer": "300000 km/s"},
]

REASONING_DATASET = [
    {"id": "r01", "question": "If a train travels at 60 mph for 2 hours, how far does it go?", "answer": "120 miles"},
    {"id": "r02", "question": "What is 25% of 80?", "answer": "20"},
    {"id": "r03", "question": "If all cats are mammals and all mammals are animals, are all cats animals?", "answer": "Yes"},
    {"id": "r04", "question": "A rectangle has length 10 and width 5. What is its area?", "answer": "50"},
    {"id": "r05", "question": "If 3x + 7 = 22, what is x?", "answer": "5"},
]

AMBIGUOUS_DATASET = [
    {"id": "a01", "question": "Is the dress blue or gold?", "answer": "ambiguous"},
    {"id": "a02", "question": "What is the best programming language?", "answer": "subjective"},
    {"id": "a03", "question": "Is a hot dog a sandwich?", "answer": "debatable"},
]


# =============================================================================
# THRESHOLD ABLATION ENGINE
# =============================================================================

class ThresholdAblationEngine:
    """
    Engine for running threshold ablation studies.

    This engine:
    1. Runs queries and captures raw similarity scores
    2. Re-classifies results under different threshold configurations
    3. Computes accuracy and calibration metrics
    4. Identifies optimal thresholds
    """

    def __init__(
        self,
        models: List[str] = None,
        results_dir: str = "benchmarks/results/ablation"
    ):
        self.protocol = AICP()
        self.models = models or ["gpt-4o", "claude-sonnet-4-5-20250929", "gemini-2.0-flash"]
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Store raw results for re-analysis
        self._raw_results: List[Dict[str, Any]] = []

    async def collect_similarity_data(
        self,
        tasks: List[Dict[str, Any]],
        name: str = "ablation_data"
    ) -> List[Dict[str, Any]]:
        """
        Run queries and collect raw similarity scores.

        This is the expensive step - we query models once and store similarities
        for later threshold analysis.
        """
        print(f"\n{'='*70}")
        print(f"  Collecting Similarity Data: {name}")
        print(f"  Tasks: {len(tasks)}, Models: {len(self.models)}")
        print(f"{'='*70}\n")

        results = []

        for i, task in enumerate(tasks):
            print(f"  [{i+1}/{len(tasks)}] {task['question'][:50]}...")

            try:
                # Run consensus query to get similarity data
                consensus_result = await self.protocol.consensus_query(
                    task["question"],
                    models=self.models
                )

                # Extract pairwise similarities from agreement matrix
                avg_similarity = self._compute_avg_similarity(consensus_result.agreement_matrix)

                # Check correctness
                is_correct = self._check_answer(
                    consensus_result.consensus_answer,
                    task["answer"]
                )

                result = {
                    "task_id": task["id"],
                    "question": task["question"],
                    "ground_truth": task["answer"],
                    "consensus_answer": consensus_result.consensus_answer,
                    "avg_similarity": avg_similarity,
                    "agreement_matrix": consensus_result.agreement_matrix,
                    "original_level": consensus_result.consensus_level.value,
                    "is_correct": is_correct,
                    "models": consensus_result.models_queried,
                }
                results.append(result)

                status = "+" if is_correct else "x"
                print(f"           {status} sim={avg_similarity:.3f} level={consensus_result.consensus_level.value}")

            except Exception as e:
                print(f"           ERROR: {e}")
                results.append({
                    "task_id": task["id"],
                    "question": task["question"],
                    "ground_truth": task["answer"],
                    "error": str(e),
                    "avg_similarity": 0.0,
                    "is_correct": False,
                })

            # Rate limiting
            await asyncio.sleep(0.5)

        self._raw_results = results
        return results

    def _compute_avg_similarity(self, agreement_matrix: Dict[str, Dict[str, float]]) -> float:
        """Compute average pairwise similarity from agreement matrix."""
        if not agreement_matrix:
            return 0.0

        models = list(agreement_matrix.keys())
        similarities = []

        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i < j and m1 in agreement_matrix and m2 in agreement_matrix[m1]:
                    similarities.append(agreement_matrix[m1][m2])

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        if not predicted or not ground_truth:
            return False

        pred_lower = predicted.lower().strip()
        truth_lower = ground_truth.lower().strip()

        # Exact match
        if pred_lower == truth_lower:
            return True

        # Contains match
        if truth_lower in pred_lower or pred_lower in truth_lower:
            return True

        # Token overlap
        pred_tokens = set(pred_lower.split())
        truth_tokens = set(truth_lower.split())
        if len(truth_tokens) > 1:
            overlap = len(pred_tokens & truth_tokens) / len(truth_tokens)
            if overlap > 0.6:
                return True

        return False

    def classify_with_thresholds(
        self,
        similarity: float,
        config: ThresholdConfig
    ) -> str:
        """Classify similarity score using given thresholds."""
        if similarity >= config.high:
            return "HIGH"
        elif similarity >= config.medium:
            return "MEDIUM"
        elif similarity >= config.low:
            return "LOW"
        else:
            return "NONE"

    def analyze_thresholds(
        self,
        results: List[Dict[str, Any]],
        config: ThresholdConfig
    ) -> AblationMetrics:
        """
        Analyze results under a specific threshold configuration.

        Returns metrics including accuracy per level and calibration.
        """
        valid_results = [r for r in results if "error" not in r]

        if not valid_results:
            return AblationMetrics(
                config=config,
                total_tasks=0,
                overall_accuracy=0.0
            )

        # Classify each result with new thresholds
        high_correct, high_total = 0, 0
        medium_correct, medium_total = 0, 0
        low_correct, low_total = 0, 0
        none_correct, none_total = 0, 0

        for r in valid_results:
            level = self.classify_with_thresholds(r["avg_similarity"], config)
            correct = r["is_correct"]

            if level == "HIGH":
                high_total += 1
                if correct:
                    high_correct += 1
            elif level == "MEDIUM":
                medium_total += 1
                if correct:
                    medium_correct += 1
            elif level == "LOW":
                low_total += 1
                if correct:
                    low_correct += 1
            else:
                none_total += 1
                if correct:
                    none_correct += 1

        total = len(valid_results)
        total_correct = high_correct + medium_correct + low_correct + none_correct

        # Compute calibration error
        # Expected accuracy: HIGH should be ~95%, MEDIUM ~75%, LOW ~50%, NONE ~25%
        expected = {"HIGH": 0.95, "MEDIUM": 0.75, "LOW": 0.50, "NONE": 0.25}
        actual = {
            "HIGH": high_correct / high_total if high_total else 0,
            "MEDIUM": medium_correct / medium_total if medium_total else 0,
            "LOW": low_correct / low_total if low_total else 0,
            "NONE": none_correct / none_total if none_total else 0,
        }
        weights = {
            "HIGH": high_total / total if total else 0,
            "MEDIUM": medium_total / total if total else 0,
            "LOW": low_total / total if total else 0,
            "NONE": none_total / total if total else 0,
        }

        calibration_error = sum(
            weights[level] * abs(expected[level] - actual[level])
            for level in expected
        )

        # Reliability score: balance accuracy and calibration
        reliability = (total_correct / total if total else 0) * (1 - calibration_error)

        return AblationMetrics(
            config=config,
            total_tasks=total,
            overall_accuracy=(total_correct / total * 100) if total else 0,
            high_count=high_total,
            high_accuracy=(high_correct / high_total * 100) if high_total else 0,
            medium_count=medium_total,
            medium_accuracy=(medium_correct / medium_total * 100) if medium_total else 0,
            low_count=low_total,
            low_accuracy=(low_correct / low_total * 100) if low_total else 0,
            none_count=none_total,
            none_accuracy=(none_correct / none_total * 100) if none_total else 0,
            calibration_error=calibration_error,
            coverage_at_high=(high_total / total * 100) if total else 0,
            reliability_score=reliability,
        )

    def run_preset_ablation(
        self,
        results: List[Dict[str, Any]]
    ) -> List[AblationMetrics]:
        """Run ablation across preset threshold configurations."""
        print(f"\n{'='*70}")
        print("  Preset Threshold Ablation")
        print(f"{'='*70}")

        metrics_list = []

        for name, thresholds in THRESHOLD_PRESETS.items():
            config = ThresholdConfig(**thresholds)
            metrics = self.analyze_thresholds(results, config)
            metrics_list.append(metrics)

            print(f"\n  {name:15} {config}")
            print(f"    Overall: {metrics.overall_accuracy:.1f}%")
            print(f"    HIGH:    {metrics.high_accuracy:.1f}% ({metrics.high_count} tasks)")
            print(f"    MEDIUM:  {metrics.medium_accuracy:.1f}% ({metrics.medium_count} tasks)")
            print(f"    LOW:     {metrics.low_accuracy:.1f}% ({metrics.low_count} tasks)")
            print(f"    Calibration Error: {metrics.calibration_error:.3f}")
            print(f"    Reliability Score: {metrics.reliability_score:.3f}")

        return metrics_list

    def run_grid_search(
        self,
        results: List[Dict[str, Any]],
        high_grid: List[float] = None,
        medium_grid: List[float] = None,
        low_grid: List[float] = None,
    ) -> Tuple[ThresholdConfig, AblationMetrics, List[AblationMetrics]]:
        """
        Run grid search to find optimal thresholds.

        Constraints:
        - high > medium > low
        - All values between 0 and 1

        Returns:
            Tuple of (optimal_config, optimal_metrics, all_metrics)
        """
        high_grid = high_grid or HIGH_THRESHOLD_GRID
        medium_grid = medium_grid or MEDIUM_THRESHOLD_GRID
        low_grid = low_grid or LOW_THRESHOLD_GRID

        print(f"\n{'='*70}")
        print("  Grid Search for Optimal Thresholds")
        print(f"  HIGH:   {high_grid}")
        print(f"  MEDIUM: {medium_grid}")
        print(f"  LOW:    {low_grid}")
        print(f"{'='*70}")

        all_metrics = []
        best_metrics = None
        best_config = None
        best_score = -1

        # Total combinations (with constraints)
        total = 0
        for h in high_grid:
            for m in medium_grid:
                for l in low_grid:
                    if h > m > l:
                        total += 1

        print(f"\n  Testing {total} valid configurations...")

        tested = 0
        for high in high_grid:
            for medium in medium_grid:
                for low in low_grid:
                    # Skip invalid (must be h > m > l)
                    if not (high > medium > low):
                        continue

                    tested += 1
                    config = ThresholdConfig(high=high, medium=medium, low=low)
                    metrics = self.analyze_thresholds(results, config)
                    all_metrics.append(metrics)

                    # Score: weighted combination of accuracy and calibration
                    # Prioritize HIGH accuracy + overall reliability
                    score = (
                        0.4 * (metrics.high_accuracy / 100) +
                        0.3 * (metrics.overall_accuracy / 100) +
                        0.3 * metrics.reliability_score
                    )

                    if score > best_score:
                        best_score = score
                        best_metrics = metrics
                        best_config = config

                    if tested % 20 == 0:
                        print(f"    Progress: {tested}/{total} configurations tested")

        print(f"\n  Optimal Configuration Found:")
        print(f"    {best_config}")
        print(f"    Score: {best_score:.4f}")
        print(f"    Overall Accuracy: {best_metrics.overall_accuracy:.1f}%")
        print(f"    HIGH Accuracy: {best_metrics.high_accuracy:.1f}%")
        print(f"    Calibration Error: {best_metrics.calibration_error:.3f}")

        return best_config, best_metrics, all_metrics

    def compute_sensitivity(
        self,
        results: List[Dict[str, Any]],
        base_config: ThresholdConfig = None
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Compute sensitivity curves for each threshold parameter.

        Varies one parameter at a time while holding others fixed.
        """
        base = base_config or ThresholdConfig(**DEFAULT_THRESHOLDS)

        print(f"\n{'='*70}")
        print("  Sensitivity Analysis")
        print(f"  Base: {base}")
        print(f"{'='*70}")

        sensitivity = {"high": [], "medium": [], "low": []}

        # Vary HIGH threshold
        print("\n  Varying HIGH threshold:")
        for h in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            if h > base.medium:  # Maintain constraint
                config = ThresholdConfig(high=h, medium=base.medium, low=base.low)
                metrics = self.analyze_thresholds(results, config)
                sensitivity["high"].append({
                    "value": h,
                    "accuracy": metrics.overall_accuracy,
                    "high_accuracy": metrics.high_accuracy,
                    "high_coverage": metrics.coverage_at_high,
                    "calibration": metrics.calibration_error,
                })
                print(f"    H={h:.2f}: acc={metrics.overall_accuracy:.1f}%, high_acc={metrics.high_accuracy:.1f}%")

        # Vary MEDIUM threshold
        print("\n  Varying MEDIUM threshold:")
        for m in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
            if base.high > m > base.low:  # Maintain constraints
                config = ThresholdConfig(high=base.high, medium=m, low=base.low)
                metrics = self.analyze_thresholds(results, config)
                sensitivity["medium"].append({
                    "value": m,
                    "accuracy": metrics.overall_accuracy,
                    "medium_accuracy": metrics.medium_accuracy,
                    "calibration": metrics.calibration_error,
                })
                print(f"    M={m:.2f}: acc={metrics.overall_accuracy:.1f}%, med_acc={metrics.medium_accuracy:.1f}%")

        # Vary LOW threshold
        print("\n  Varying LOW threshold:")
        for l in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            if base.medium > l:  # Maintain constraint
                config = ThresholdConfig(high=base.high, medium=base.medium, low=l)
                metrics = self.analyze_thresholds(results, config)
                sensitivity["low"].append({
                    "value": l,
                    "accuracy": metrics.overall_accuracy,
                    "low_accuracy": metrics.low_accuracy,
                    "calibration": metrics.calibration_error,
                })
                print(f"    L={l:.2f}: acc={metrics.overall_accuracy:.1f}%, low_acc={metrics.low_accuracy:.1f}%")

        return sensitivity

    async def run_full_ablation(
        self,
        tasks: List[Dict[str, Any]],
        name: str = "threshold_ablation"
    ) -> AblationStudy:
        """
        Run complete ablation study.

        Steps:
        1. Collect similarity data from model queries
        2. Analyze with preset configurations
        3. Run grid search for optimal thresholds
        4. Compute sensitivity curves
        5. Save results
        """
        # Step 1: Collect data
        raw_results = await self.collect_similarity_data(tasks, name)

        # Step 2: Preset analysis
        preset_metrics = self.run_preset_ablation(raw_results)

        # Step 3: Grid search
        optimal_config, optimal_metrics, grid_metrics = self.run_grid_search(raw_results)

        # Step 4: Sensitivity
        sensitivity = self.compute_sensitivity(raw_results)

        # Build study results
        study = AblationStudy(
            name=name,
            timestamp=datetime.utcnow().isoformat(),
            dataset_name=name,
            total_tasks=len(tasks),
            models_used=self.models,
            raw_results=raw_results,
            config_metrics=[m.to_dict() for m in preset_metrics],
            optimal_config=optimal_config.to_dict(),
            optimal_score=optimal_metrics.reliability_score,
            sensitivity_high=sensitivity["high"],
            sensitivity_medium=sensitivity["medium"],
            sensitivity_low=sensitivity["low"],
        )

        # Save results
        output_path = self.results_dir / f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(asdict(study), f, indent=2, default=str)

        print(f"\n{'='*70}")
        print(f"  Ablation Study Complete")
        print(f"  Results saved to: {output_path}")
        print(f"{'='*70}")

        return study


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def run_quick_ablation():
    """Quick ablation with minimal dataset."""
    engine = ThresholdAblationEngine()
    return await engine.run_full_ablation(
        FACTUAL_DATASET[:5],
        name="quick_ablation"
    )


async def run_standard_ablation():
    """Standard ablation with factual and reasoning datasets."""
    engine = ThresholdAblationEngine()

    # Combine datasets
    all_tasks = FACTUAL_DATASET + REASONING_DATASET

    return await engine.run_full_ablation(
        all_tasks,
        name="standard_ablation"
    )


async def run_full_ablation():
    """Full ablation with all datasets."""
    engine = ThresholdAblationEngine()

    # Combine all datasets
    all_tasks = FACTUAL_DATASET + REASONING_DATASET + AMBIGUOUS_DATASET

    return await engine.run_full_ablation(
        all_tasks,
        name="full_ablation"
    )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Threshold Ablation Study for i2i Consensus Protocol"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default="quick",
        help="Ablation mode: quick (5 tasks), standard (15 tasks), full (18 tasks)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/ablation",
        help="Output directory for results"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("  i2i Threshold Ablation Study")
    print(f"  Mode: {args.mode}")
    print("="*70)

    if args.mode == "quick":
        asyncio.run(run_quick_ablation())
    elif args.mode == "standard":
        asyncio.run(run_standard_ablation())
    else:
        asyncio.run(run_full_ablation())


if __name__ == "__main__":
    main()
