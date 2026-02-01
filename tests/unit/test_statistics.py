"""
Tests for statistical significance functions.
"""

import pytest
from i2i.statistics import (
    mcnemar_test,
    bootstrap_accuracy_ci,
    bootstrap_improvement_ci,
    analyze_benchmark_results,
    McNemarResult,
    BootstrapCI,
)


class TestMcNemarTest:
    """Tests for McNemar's test implementation."""

    def test_perfect_agreement(self):
        """Both methods give identical results."""
        single = [True, True, False, False, True]
        consensus = [True, True, False, False, True]

        result = mcnemar_test(single, consensus)

        assert result.n_both_correct == 3
        assert result.n_both_wrong == 2
        assert result.n_single_only == 0
        assert result.n_consensus_only == 0
        assert result.chi_square == 0.0
        assert result.p_value == 1.0
        assert result.direction == "no_difference"
        assert not result.significant_at_05

    def test_consensus_better(self):
        """Consensus corrects single-model errors."""
        # 10 cases where consensus fixes single's errors
        # 2 cases where single was right but consensus wrong
        single = [False] * 10 + [True] * 2 + [True] * 5 + [False] * 3
        consensus = [True] * 10 + [False] * 2 + [True] * 5 + [False] * 3

        result = mcnemar_test(single, consensus)

        assert result.n_consensus_only == 10  # c
        assert result.n_single_only == 2  # b
        assert result.direction == "consensus_better"
        assert result.cohens_g > 0  # Positive = consensus better

    def test_single_better(self):
        """Single model outperforms consensus."""
        # More cases where single is right but consensus wrong
        single = [True] * 8 + [False] * 2 + [True] * 5 + [False] * 5
        consensus = [False] * 8 + [True] * 2 + [True] * 5 + [False] * 5

        result = mcnemar_test(single, consensus)

        assert result.n_single_only == 8  # b
        assert result.n_consensus_only == 2  # c
        assert result.direction == "single_better"
        assert result.cohens_g < 0  # Negative = single better

    def test_yates_correction(self):
        """Yates continuity correction is applied."""
        single = [False] * 5 + [True] * 3 + [True, True]
        consensus = [True] * 5 + [False] * 3 + [True, True]

        result = mcnemar_test(single, consensus)

        # Yates correction reduces chi-square
        assert result.chi_square_corrected <= result.chi_square

    def test_exact_test_small_sample(self):
        """Exact binomial test used when discordant pairs < 25."""
        single = [False, True, True, False, True]
        consensus = [True, False, True, False, True]

        result = mcnemar_test(single, consensus)

        # Should have exact p-value for small samples
        assert result.p_value_exact is not None

    def test_significance_levels(self):
        """Significance flags work correctly."""
        # Create highly significant difference
        single = [False] * 30 + [True] * 5 + [True, False] * 5
        consensus = [True] * 30 + [False] * 5 + [True, False] * 5

        result = mcnemar_test(single, consensus)

        # Should be significant at both levels
        assert result.significant_at_05
        assert result.significant_at_01

    def test_mismatched_lengths(self):
        """Raises error for different length inputs."""
        single = [True, False, True]
        consensus = [True, False]

        with pytest.raises(ValueError):
            mcnemar_test(single, consensus)

    def test_empty_lists(self):
        """Handles empty input gracefully."""
        result = mcnemar_test([], [])

        assert result.n_both_correct == 0
        assert result.chi_square == 0.0
        assert result.p_value == 1.0


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_perfect_accuracy(self):
        """CI for 100% accuracy."""
        correct = [True] * 50

        ci = bootstrap_accuracy_ci(correct, n_bootstrap=1000, seed=42)

        assert ci.point_estimate == 100.0
        assert ci.ci_lower >= 90.0  # Should be near 100
        assert ci.ci_upper == 100.0
        assert ci.n_samples == 50

    def test_zero_accuracy(self):
        """CI for 0% accuracy."""
        correct = [False] * 50

        ci = bootstrap_accuracy_ci(correct, n_bootstrap=1000, seed=42)

        assert ci.point_estimate == 0.0
        assert ci.ci_lower == 0.0
        assert ci.ci_upper <= 10.0

    def test_50_percent_accuracy(self):
        """CI for 50% accuracy."""
        correct = [True, False] * 25

        ci = bootstrap_accuracy_ci(correct, n_bootstrap=1000, seed=42)

        assert ci.point_estimate == 50.0
        # 95% CI for n=50, p=0.5 should be roughly [36%, 64%]
        assert 30.0 <= ci.ci_lower <= 45.0
        assert 55.0 <= ci.ci_upper <= 70.0

    def test_small_sample_wide_ci(self):
        """Small samples have wider CIs."""
        correct_small = [True] * 5 + [False] * 5
        correct_large = [True] * 50 + [False] * 50

        ci_small = bootstrap_accuracy_ci(correct_small, n_bootstrap=1000, seed=42)
        ci_large = bootstrap_accuracy_ci(correct_large, n_bootstrap=1000, seed=42)

        # Both have 50% accuracy
        assert ci_small.point_estimate == ci_large.point_estimate == 50.0

        # Small sample has wider CI
        assert ci_small.margin_of_error() > ci_large.margin_of_error()

    def test_reproducibility(self):
        """Same seed gives same results."""
        correct = [True, False] * 25

        ci1 = bootstrap_accuracy_ci(correct, n_bootstrap=1000, seed=42)
        ci2 = bootstrap_accuracy_ci(correct, n_bootstrap=1000, seed=42)

        assert ci1.ci_lower == ci2.ci_lower
        assert ci1.ci_upper == ci2.ci_upper

    def test_empty_list(self):
        """Handles empty input."""
        ci = bootstrap_accuracy_ci([])

        assert ci.point_estimate == 0.0
        assert ci.n_samples == 0

    def test_string_output(self):
        """String formatting works."""
        correct = [True] * 30 + [False] * 20

        ci = bootstrap_accuracy_ci(correct, n_bootstrap=1000, seed=42)

        # Should format as "60.0% [X.X%, Y.Y%]"
        s = str(ci)
        assert "60.0%" in s
        assert "[" in s and "]" in s


class TestBootstrapImprovement:
    """Tests for paired improvement CI."""

    def test_no_improvement(self):
        """Same accuracy for both methods."""
        single = [True, False] * 25
        consensus = [True, False] * 25

        ci = bootstrap_improvement_ci(single, consensus, n_bootstrap=1000, seed=42)

        assert ci.point_estimate == 0.0
        # CI should include 0
        assert ci.ci_lower <= 0.0 <= ci.ci_upper

    def test_positive_improvement(self):
        """Consensus better than single."""
        single = [True] * 30 + [False] * 20
        consensus = [True] * 40 + [False] * 10

        ci = bootstrap_improvement_ci(single, consensus, n_bootstrap=1000, seed=42)

        assert ci.point_estimate == 20.0  # 80% - 60% = 20%
        assert ci.ci_lower > 0.0  # Should be significantly positive

    def test_negative_improvement(self):
        """Single better than consensus."""
        single = [True] * 40 + [False] * 10
        consensus = [True] * 30 + [False] * 20

        ci = bootstrap_improvement_ci(single, consensus, n_bootstrap=1000, seed=42)

        assert ci.point_estimate == -20.0
        assert ci.ci_upper < 0.0  # Should be significantly negative

    def test_paired_bootstrap(self):
        """Paired bootstrap preserves correlation."""
        # Create correlated data
        single = [True, True, False, False] * 10
        consensus = [True, True, True, False] * 10  # Fixes one error per 4

        ci = bootstrap_improvement_ci(single, consensus, n_bootstrap=1000, seed=42)

        # Should show +25% improvement (50% -> 75%)
        assert 20.0 <= ci.point_estimate <= 30.0

    def test_mismatched_lengths(self):
        """Raises error for different lengths."""
        with pytest.raises(ValueError):
            bootstrap_improvement_ci([True, False], [True])


class TestAnalyzeBenchmarkResults:
    """Tests for full benchmark analysis."""

    def test_basic_analysis(self):
        """Analyzes standard benchmark results."""
        results = [
            {
                "single_model_correct": True,
                "consensus_correct": True,
                "consensus_level": "HIGH"
            },
            {
                "single_model_correct": False,
                "consensus_correct": True,
                "consensus_level": "HIGH"
            },
            {
                "single_model_correct": True,
                "consensus_correct": False,
                "consensus_level": "MEDIUM"
            },
            {
                "single_model_correct": False,
                "consensus_correct": False,
                "consensus_level": "LOW"
            },
        ]

        stats = analyze_benchmark_results(results, "test_benchmark", n_bootstrap=100)

        assert stats.benchmark_name == "test_benchmark"
        assert stats.n_samples == 4
        assert stats.single_model_accuracy.point_estimate == 50.0
        assert stats.consensus_accuracy.point_estimate == 50.0
        assert stats.mcnemar is not None

    def test_skips_errors(self):
        """Skips results with errors."""
        results = [
            {"single_model_correct": True, "consensus_correct": True, "error": None},
            {"single_model_correct": False, "consensus_correct": True, "error": "timeout"},
            {"single_model_correct": True, "consensus_correct": True, "error": None},
        ]

        stats = analyze_benchmark_results(results, "test", n_bootstrap=100)

        # Should only count 2 results (skip the one with error)
        assert stats.n_samples == 2

    def test_by_consensus_level(self):
        """Computes accuracy by consensus level."""
        results = [
            {"single_model_correct": True, "consensus_correct": True, "consensus_level": "HIGH"},
            {"single_model_correct": False, "consensus_correct": True, "consensus_level": "HIGH"},
            {"single_model_correct": True, "consensus_correct": False, "consensus_level": "MEDIUM"},
            {"single_model_correct": False, "consensus_correct": False, "consensus_level": "LOW"},
        ]

        stats = analyze_benchmark_results(results, "test", n_bootstrap=100)

        # HIGH: 2/2 = 100%
        assert stats.high_consensus_accuracy is not None
        assert stats.high_consensus_accuracy.point_estimate == 100.0

        # MEDIUM: 0/1 = 0%
        assert stats.medium_consensus_accuracy is not None
        assert stats.medium_consensus_accuracy.point_estimate == 0.0


class TestMcNemarResultString:
    """Tests for McNemar result string representation."""

    def test_str_output(self):
        """String output contains key information."""
        result = McNemarResult(
            n_both_correct=10,
            n_single_only=3,
            n_consensus_only=7,
            n_both_wrong=5,
            chi_square=1.6,
            chi_square_corrected=0.9,
            p_value=0.34,
            odds_ratio=0.43,
            cohens_g=0.2,
            significant_at_05=False,
            significant_at_01=False,
            direction="consensus_better",
        )

        s = str(result)

        assert "McNemar" in s
        assert "10" in s  # n_both_correct
        assert "p-value" in s
        assert "consensus_better" in s


class TestBootstrapCILatex:
    """Tests for LaTeX formatting."""

    def test_latex_output(self):
        """LaTeX output is properly formatted."""
        ci = BootstrapCI(
            metric_name="accuracy",
            point_estimate=75.5,
            ci_lower=68.2,
            ci_upper=82.8,
            ci_level=0.95,
            n_bootstrap=10000,
            n_samples=50,
            bootstrap_se=3.7,
        )

        latex = ci.to_latex()

        assert "75.5" in latex
        assert "$\\pm$" in latex
