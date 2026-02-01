"""
Statistical significance tests for i2i evaluation.

Provides:
- McNemar's test for paired accuracy comparisons
- Bootstrap confidence intervals for accuracy metrics
- Effect size calculations (Cohen's g for McNemar)

Reference:
- McNemar, Q. (1947). Note on the sampling error of the difference between
  correlated proportions or percentages. Psychometrika, 12(2), 153-157.
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import random


@dataclass
class McNemarResult:
    """Result of McNemar's test for paired comparisons."""

    # Contingency table cells
    n_both_correct: int      # a: single correct, consensus correct
    n_single_only: int       # b: single correct, consensus wrong
    n_consensus_only: int    # c: single wrong, consensus correct
    n_both_wrong: int        # d: single wrong, consensus wrong

    # Test statistics
    chi_square: float        # McNemar's chi-square statistic
    chi_square_corrected: float  # With Yates continuity correction
    p_value: float           # Two-tailed p-value (corrected)
    p_value_exact: Optional[float] = None  # Exact binomial p-value when n < 25

    # Effect size
    odds_ratio: float = 0.0  # b/c ratio
    cohens_g: float = 0.0    # Effect size for McNemar

    # Interpretation
    significant_at_05: bool = False
    significant_at_01: bool = False
    direction: str = ""      # "consensus_better", "single_better", or "no_difference"

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "McNemar's Test Results",
            "=" * 40,
            "",
            "Contingency Table:",
            f"                 Consensus Correct  Consensus Wrong",
            f"  Single Correct       {self.n_both_correct:4d}              {self.n_single_only:4d}",
            f"  Single Wrong         {self.n_consensus_only:4d}              {self.n_both_wrong:4d}",
            "",
            f"Chi-square (corrected): {self.chi_square_corrected:.3f}",
            f"p-value: {self.p_value:.4f}" + (f" (exact: {self.p_value_exact:.4f})" if self.p_value_exact else ""),
            f"Effect size (Cohen's g): {self.cohens_g:.3f}",
            "",
            f"Significant at α=0.05: {'Yes' if self.significant_at_05 else 'No'}",
            f"Significant at α=0.01: {'Yes' if self.significant_at_01 else 'No'}",
            f"Direction: {self.direction}",
        ]
        return "\n".join(lines)


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a metric."""

    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    n_bootstrap: int = 10000
    n_samples: int = 0

    # Standard error from bootstrap
    bootstrap_se: float = 0.0

    def margin_of_error(self) -> float:
        """Half-width of confidence interval."""
        return (self.ci_upper - self.ci_lower) / 2

    def __str__(self) -> str:
        """Format as: estimate [lower, upper]."""
        return f"{self.point_estimate:.1f}% [{self.ci_lower:.1f}%, {self.ci_upper:.1f}%]"

    def to_latex(self) -> str:
        """Format for LaTeX tables."""
        return f"{self.point_estimate:.1f} $\\pm$ {self.margin_of_error():.1f}"


@dataclass
class BenchmarkStatistics:
    """Complete statistical analysis for a benchmark."""

    benchmark_name: str
    n_samples: int

    # Accuracy metrics with CIs
    single_model_accuracy: BootstrapCI
    consensus_accuracy: BootstrapCI
    improvement: BootstrapCI

    # By consensus level (if available)
    high_consensus_accuracy: Optional[BootstrapCI] = None
    medium_consensus_accuracy: Optional[BootstrapCI] = None
    low_consensus_accuracy: Optional[BootstrapCI] = None

    # McNemar's test
    mcnemar: Optional[McNemarResult] = None


def _chi_square_to_p_value(chi_sq: float, df: int = 1) -> float:
    """
    Convert chi-square statistic to p-value using gamma function approximation.
    Uses the chi-square distribution CDF.
    """
    if chi_sq <= 0:
        return 1.0

    # For df=1, use normal approximation: chi_sq ~ Z^2
    # P(chi_sq > x) = 2 * P(Z > sqrt(x))
    z = math.sqrt(chi_sq)

    # Standard normal CDF approximation (Abramowitz & Stegun)
    def norm_cdf(x: float) -> float:
        """Standard normal CDF."""
        if x < 0:
            return 1 - norm_cdf(-x)
        # Approximation coefficients
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
        return y

    # Two-tailed p-value
    p_value = 2 * (1 - norm_cdf(z))
    return max(0.0, min(1.0, p_value))


def _binomial_exact_p(n: int, k: int, p: float = 0.5) -> float:
    """
    Exact two-tailed binomial p-value for k successes in n trials.
    Used for McNemar's exact test when discordant pairs < 25.
    """
    if n == 0:
        return 1.0

    # Binomial coefficient using log-gamma for numerical stability
    def log_binom(n: int, k: int) -> float:
        if k < 0 or k > n:
            return float('-inf')
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    # Calculate P(X = k) for binomial
    def binom_pmf(n: int, k: int, p: float) -> float:
        if k < 0 or k > n:
            return 0.0
        return math.exp(log_binom(n, k) + k * math.log(p) + (n - k) * math.log(1 - p))

    # Two-tailed: sum probabilities for values as extreme or more extreme
    observed_prob = binom_pmf(n, k, p)
    p_value = 0.0

    for i in range(n + 1):
        prob_i = binom_pmf(n, i, p)
        if prob_i <= observed_prob + 1e-10:  # Add small epsilon for numerical stability
            p_value += prob_i

    return min(1.0, p_value)


def mcnemar_test(
    single_correct: List[bool],
    consensus_correct: List[bool]
) -> McNemarResult:
    """
    Perform McNemar's test for paired binary outcomes.

    Compares single-model vs consensus accuracy on the same questions.
    Tests the null hypothesis that the marginal proportions are equal.

    Args:
        single_correct: List of True/False for single model correctness
        consensus_correct: List of True/False for consensus correctness

    Returns:
        McNemarResult with test statistics and interpretation
    """
    if len(single_correct) != len(consensus_correct):
        raise ValueError("Lists must have same length")

    # Build contingency table
    n_both_correct = 0   # a
    n_single_only = 0    # b: single right, consensus wrong
    n_consensus_only = 0 # c: single wrong, consensus right
    n_both_wrong = 0     # d

    for s, c in zip(single_correct, consensus_correct):
        if s and c:
            n_both_correct += 1
        elif s and not c:
            n_single_only += 1
        elif not s and c:
            n_consensus_only += 1
        else:
            n_both_wrong += 1

    b = n_single_only
    c = n_consensus_only
    n_discordant = b + c

    # McNemar's chi-square statistic (with Yates continuity correction)
    if n_discordant == 0:
        chi_sq = 0.0
        chi_sq_corrected = 0.0
    else:
        chi_sq = (b - c) ** 2 / n_discordant
        # Yates continuity correction
        chi_sq_corrected = (abs(b - c) - 1) ** 2 / n_discordant if n_discordant > 0 else 0.0

    # P-value
    p_value = _chi_square_to_p_value(chi_sq_corrected)

    # Exact test for small samples (n_discordant < 25)
    p_value_exact = None
    if n_discordant < 25 and n_discordant > 0:
        # Use smaller of b, c as the test statistic
        k = min(b, c)
        p_value_exact = _binomial_exact_p(n_discordant, k, 0.5)

    # Effect size: Cohen's g = (p_c - 0.5) where p_c = c/(b+c)
    # Range: -0.5 to 0.5, positive means consensus better
    if n_discordant > 0:
        p_c = c / n_discordant
        cohens_g = p_c - 0.5
    else:
        cohens_g = 0.0

    # Odds ratio (b/c)
    odds_ratio = b / c if c > 0 else float('inf') if b > 0 else 1.0

    # Determine direction
    if c > b:
        direction = "consensus_better"
    elif b > c:
        direction = "single_better"
    else:
        direction = "no_difference"

    # Use exact p-value if available, otherwise corrected
    final_p = p_value_exact if p_value_exact is not None else p_value

    return McNemarResult(
        n_both_correct=n_both_correct,
        n_single_only=n_single_only,
        n_consensus_only=n_consensus_only,
        n_both_wrong=n_both_wrong,
        chi_square=chi_sq,
        chi_square_corrected=chi_sq_corrected,
        p_value=final_p,
        p_value_exact=p_value_exact,
        odds_ratio=odds_ratio,
        cohens_g=cohens_g,
        significant_at_05=final_p < 0.05,
        significant_at_01=final_p < 0.01,
        direction=direction,
    )


def bootstrap_accuracy_ci(
    correct: List[bool],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: Optional[int] = None
) -> BootstrapCI:
    """
    Calculate bootstrap confidence interval for accuracy.

    Uses the percentile method for CI calculation.

    Args:
        correct: List of True/False for correctness
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        BootstrapCI with point estimate and confidence interval
    """
    if not correct:
        return BootstrapCI(
            metric_name="accuracy",
            point_estimate=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            ci_level=ci_level,
            n_bootstrap=n_bootstrap,
            n_samples=0,
        )

    if seed is not None:
        random.seed(seed)

    n = len(correct)
    point_estimate = sum(correct) / n * 100  # As percentage

    # Bootstrap resampling
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = random.choices(correct, k=n)
        acc = sum(sample) / n * 100
        bootstrap_estimates.append(acc)

    # Percentile method for CI
    alpha = 1 - ci_level
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    sorted_estimates = sorted(bootstrap_estimates)
    ci_lower = sorted_estimates[lower_idx]
    ci_upper = sorted_estimates[upper_idx]

    # Bootstrap standard error
    mean_bootstrap = sum(bootstrap_estimates) / n_bootstrap
    variance = sum((x - mean_bootstrap) ** 2 for x in bootstrap_estimates) / (n_bootstrap - 1)
    bootstrap_se = math.sqrt(variance)

    return BootstrapCI(
        metric_name="accuracy",
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        n_samples=n,
        bootstrap_se=bootstrap_se,
    )


def bootstrap_improvement_ci(
    single_correct: List[bool],
    consensus_correct: List[bool],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: Optional[int] = None
) -> BootstrapCI:
    """
    Calculate bootstrap CI for improvement (consensus - single accuracy).

    Uses paired bootstrap to preserve correlation structure.

    Args:
        single_correct: List of True/False for single model
        consensus_correct: List of True/False for consensus
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        seed: Random seed

    Returns:
        BootstrapCI for the improvement metric
    """
    if len(single_correct) != len(consensus_correct):
        raise ValueError("Lists must have same length")

    if not single_correct:
        return BootstrapCI(
            metric_name="improvement",
            point_estimate=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            ci_level=ci_level,
            n_bootstrap=n_bootstrap,
            n_samples=0,
        )

    if seed is not None:
        random.seed(seed)

    n = len(single_correct)

    # Point estimate
    single_acc = sum(single_correct) / n * 100
    consensus_acc = sum(consensus_correct) / n * 100
    point_estimate = consensus_acc - single_acc

    # Paired bootstrap
    pairs = list(zip(single_correct, consensus_correct))
    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        # Sample pairs with replacement
        sample = random.choices(pairs, k=n)
        single_sample = [p[0] for p in sample]
        consensus_sample = [p[1] for p in sample]

        s_acc = sum(single_sample) / n * 100
        c_acc = sum(consensus_sample) / n * 100
        improvement = c_acc - s_acc
        bootstrap_estimates.append(improvement)

    # Percentile CI
    alpha = 1 - ci_level
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    sorted_estimates = sorted(bootstrap_estimates)
    ci_lower = sorted_estimates[lower_idx]
    ci_upper = sorted_estimates[upper_idx]

    # Bootstrap SE
    mean_bootstrap = sum(bootstrap_estimates) / n_bootstrap
    variance = sum((x - mean_bootstrap) ** 2 for x in bootstrap_estimates) / (n_bootstrap - 1)
    bootstrap_se = math.sqrt(variance)

    return BootstrapCI(
        metric_name="improvement",
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        n_samples=n,
        bootstrap_se=bootstrap_se,
    )


def analyze_benchmark_results(
    results: List[Dict[str, Any]],
    benchmark_name: str,
    n_bootstrap: int = 10000,
    seed: Optional[int] = 42
) -> BenchmarkStatistics:
    """
    Perform complete statistical analysis on benchmark results.

    Args:
        results: List of result dicts with single_model_correct and consensus_correct
        benchmark_name: Name of the benchmark
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        BenchmarkStatistics with all tests and CIs
    """
    # Extract correctness arrays
    single_correct = []
    consensus_correct = []

    # By consensus level
    high_correct = []
    medium_correct = []
    low_correct = []

    for r in results:
        if r.get("error") is not None:
            continue

        s_correct = r.get("single_model_correct", False)
        c_correct = r.get("consensus_correct", False)
        level = r.get("consensus_level", "NONE")

        single_correct.append(bool(s_correct))
        consensus_correct.append(bool(c_correct))

        if level == "HIGH":
            high_correct.append(bool(c_correct))
        elif level == "MEDIUM":
            medium_correct.append(bool(c_correct))
        elif level == "LOW":
            low_correct.append(bool(c_correct))

    n_samples = len(single_correct)

    # McNemar's test
    mcnemar = mcnemar_test(single_correct, consensus_correct)

    # Bootstrap CIs
    single_ci = bootstrap_accuracy_ci(single_correct, n_bootstrap, seed=seed)
    single_ci.metric_name = "single_model_accuracy"

    consensus_ci = bootstrap_accuracy_ci(consensus_correct, n_bootstrap, seed=seed)
    consensus_ci.metric_name = "consensus_accuracy"

    improvement_ci = bootstrap_improvement_ci(
        single_correct, consensus_correct, n_bootstrap, seed=seed
    )

    # By consensus level
    high_ci = None
    if high_correct:
        high_ci = bootstrap_accuracy_ci(high_correct, n_bootstrap, seed=seed)
        high_ci.metric_name = "high_consensus_accuracy"

    medium_ci = None
    if medium_correct:
        medium_ci = bootstrap_accuracy_ci(medium_correct, n_bootstrap, seed=seed)
        medium_ci.metric_name = "medium_consensus_accuracy"

    low_ci = None
    if low_correct:
        low_ci = bootstrap_accuracy_ci(low_correct, n_bootstrap, seed=seed)
        low_ci.metric_name = "low_consensus_accuracy"

    return BenchmarkStatistics(
        benchmark_name=benchmark_name,
        n_samples=n_samples,
        single_model_accuracy=single_ci,
        consensus_accuracy=consensus_ci,
        improvement=improvement_ci,
        high_consensus_accuracy=high_ci,
        medium_consensus_accuracy=medium_ci,
        low_consensus_accuracy=low_ci,
        mcnemar=mcnemar,
    )


def load_and_analyze(
    results_path: Path,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> BenchmarkStatistics:
    """
    Load benchmark results from JSON and perform statistical analysis.

    Args:
        results_path: Path to benchmark results JSON
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        BenchmarkStatistics
    """
    with open(results_path) as f:
        data = json.load(f)

    benchmark_name = data.get("name", results_path.stem)
    results = data.get("results", [])

    return analyze_benchmark_results(results, benchmark_name, n_bootstrap, seed)


def format_statistics_table(stats_list: List[BenchmarkStatistics]) -> str:
    """
    Format multiple benchmark statistics as a table for paper.

    Args:
        stats_list: List of BenchmarkStatistics

    Returns:
        Formatted table string
    """
    lines = []

    # Header
    lines.append("=" * 100)
    lines.append(f"{'Benchmark':<25} {'n':>5} {'Single Model':>18} {'Consensus':>18} {'Improvement':>18} {'p-value':>10}")
    lines.append("=" * 100)

    for stats in stats_list:
        single = stats.single_model_accuracy
        cons = stats.consensus_accuracy
        imp = stats.improvement
        mcn = stats.mcnemar

        p_str = f"{mcn.p_value:.4f}" if mcn else "N/A"
        sig = "*" if mcn and mcn.significant_at_05 else ""
        if mcn and mcn.significant_at_01:
            sig = "**"

        lines.append(
            f"{stats.benchmark_name:<25} "
            f"{stats.n_samples:>5} "
            f"{single.point_estimate:>6.1f} ±{single.margin_of_error():>4.1f}% "
            f"{cons.point_estimate:>6.1f} ±{cons.margin_of_error():>4.1f}% "
            f"{imp.point_estimate:>+6.1f} ±{imp.margin_of_error():>4.1f}% "
            f"{p_str:>8}{sig}"
        )

    lines.append("=" * 100)
    lines.append("* p < 0.05, ** p < 0.01")
    lines.append("95% bootstrap confidence intervals (10,000 samples)")

    return "\n".join(lines)


def format_latex_table(stats_list: List[BenchmarkStatistics]) -> str:
    """
    Format statistics as LaTeX table for paper.

    Args:
        stats_list: List of BenchmarkStatistics

    Returns:
        LaTeX table string
    """
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Benchmark Results with 95\\% Bootstrap Confidence Intervals}",
        "\\label{tab:results}",
        "\\begin{tabular}{lrcccc}",
        "\\toprule",
        "Benchmark & n & Single Model & Consensus & Improvement & McNemar p \\\\",
        "\\midrule",
    ]

    for stats in stats_list:
        single = stats.single_model_accuracy
        cons = stats.consensus_accuracy
        imp = stats.improvement
        mcn = stats.mcnemar

        sig = ""
        if mcn and mcn.significant_at_01:
            sig = "$^{**}$"
        elif mcn and mcn.significant_at_05:
            sig = "$^{*}$"

        p_str = f"{mcn.p_value:.3f}" if mcn else "---"

        lines.append(
            f"{stats.benchmark_name} & "
            f"{stats.n_samples} & "
            f"{single.to_latex()}\\% & "
            f"{cons.to_latex()}\\% & "
            f"{imp.point_estimate:+.1f} $\\pm$ {imp.margin_of_error():.1f}\\% & "
            f"{p_str}{sig} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item $^{*}p < 0.05$, $^{**}p < 0.01$ (McNemar's test with Yates correction)",
        "\\end{tablenotes}",
        "\\end{table}",
    ])

    return "\n".join(lines)
