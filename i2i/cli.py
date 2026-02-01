"""
CLI for i2i - AI-to-AI Communication Protocol.

Commands:
    i2i config show              Show current configuration
    i2i config get KEY           Get a specific config value
    i2i config set KEY VALUE     Set a config value
    i2i config add KEY VALUE     Add a value to a list
    i2i config remove KEY VALUE  Remove a value from a list
    i2i config reset             Reset to defaults
    i2i config init              Initialize config file
    i2i models list              List available models
    i2i models info MODEL        Show model capabilities
"""

import json
import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint

from .config import Config, get_config, get_user_config_path, DEFAULTS
from .providers import ProviderRegistry
from .router import DEFAULT_MODEL_CAPABILITIES, TaskType

app = typer.Typer(
    name="i2i",
    help="i2i - AI-to-AI Communication Protocol CLI",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Manage i2i configuration")
models_app = typer.Typer(help="Manage and inspect models")

app.add_typer(config_app, name="config")
app.add_typer(models_app, name="models")

console = Console()


# ============ Config Commands ============

@config_app.command("show")
def config_show(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Show current configuration."""
    config = Config.load()

    if json_output:
        console.print(Syntax(json.dumps(config.to_dict(), indent=2), "json"))
        return

    # Pretty print with rich
    console.print(Panel.fit("[bold]i2i Configuration[/bold]"))

    if config.source_path:
        console.print(f"[dim]Loaded from: {config.source_path}[/dim]\n")
    else:
        console.print("[dim]Using built-in defaults[/dim]\n")

    # Models section
    console.print("[bold cyan]Models[/bold cyan]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Type")
    table.add_column("Model(s)")

    table.add_row("Consensus", ", ".join(config.consensus_models))
    table.add_row("Classifier", config.classifier_model)
    table.add_row("Synthesis", ", ".join(config.synthesis_models))
    table.add_row("Verification", ", ".join(config.verification_models))
    table.add_row("Epistemic", ", ".join(config.epistemic_models))

    console.print(table)

    # Routing section
    console.print("\n[bold cyan]Routing[/bold cyan]")
    routing = config.get("routing", {})
    console.print(f"  Strategy: {routing.get('default_strategy', 'balanced')}")
    console.print(f"  AI Classifier: {routing.get('use_ai_classifier', False)}")
    console.print(f"  Fallback: {routing.get('fallback_enabled', True)}")

    # Consensus section
    console.print("\n[bold cyan]Consensus[/bold cyan]")
    consensus = config.get("consensus", {})
    console.print(f"  Agreement Threshold: {consensus.get('min_agreement_threshold', 0.7)}")
    console.print(f"  Max Rounds: {consensus.get('max_rounds', 3)}")


@config_app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Config key (dot notation, e.g., models.consensus)"),
):
    """Get a specific configuration value."""
    config = Config.load()
    value = config.get(key)

    if value is None:
        console.print(f"[red]Key not found: {key}[/red]")
        raise typer.Exit(1)

    if isinstance(value, (dict, list)):
        console.print(Syntax(json.dumps(value, indent=2), "json"))
    else:
        console.print(value)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key (dot notation)"),
    value: str = typer.Argument(..., help="Value to set"),
    save_to: Optional[Path] = typer.Option(None, "--save-to", "-o", help="Save to specific path"),
):
    """Set a configuration value."""
    config = Config.load()

    # Try to parse value as JSON for complex types
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        # Treat as string, but convert booleans
        if value.lower() == "true":
            parsed_value = True
        elif value.lower() == "false":
            parsed_value = False
        else:
            try:
                parsed_value = float(value) if "." in value else int(value)
            except ValueError:
                parsed_value = value

    config.set(key, parsed_value)
    save_path = config.save(save_to)

    console.print(f"[green]Set {key} = {parsed_value}[/green]")
    console.print(f"[dim]Saved to: {save_path}[/dim]")


@config_app.command("add")
def config_add(
    key: str = typer.Argument(..., help="Config key for a list (e.g., models.consensus)"),
    value: str = typer.Argument(..., help="Value to add"),
    save_to: Optional[Path] = typer.Option(None, "--save-to", "-o", help="Save to specific path"),
):
    """Add a value to a list configuration."""
    config = Config.load()

    try:
        if config.add(key, value):
            save_path = config.save(save_to)
            console.print(f"[green]Added '{value}' to {key}[/green]")
            console.print(f"[dim]Saved to: {save_path}[/dim]")
        else:
            console.print(f"[yellow]'{value}' already exists in {key}[/yellow]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("remove")
def config_remove(
    key: str = typer.Argument(..., help="Config key for a list"),
    value: str = typer.Argument(..., help="Value to remove"),
    save_to: Optional[Path] = typer.Option(None, "--save-to", "-o", help="Save to specific path"),
):
    """Remove a value from a list configuration."""
    config = Config.load()

    try:
        if config.remove(key, value):
            save_path = config.save(save_to)
            console.print(f"[green]Removed '{value}' from {key}[/green]")
            console.print(f"[dim]Saved to: {save_path}[/dim]")
        else:
            console.print(f"[yellow]'{value}' not found in {key}[/yellow]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@config_app.command("reset")
def config_reset(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    save_to: Optional[Path] = typer.Option(None, "--save-to", "-o", help="Save to specific path"),
):
    """Reset configuration to defaults."""
    if not confirm:
        confirm = typer.confirm("Reset configuration to defaults?")

    if confirm:
        config = Config.load_defaults()
        save_path = config.save(save_to)
        console.print("[green]Configuration reset to defaults[/green]")
        console.print(f"[dim]Saved to: {save_path}[/dim]")
    else:
        console.print("[yellow]Cancelled[/yellow]")


@config_app.command("init")
def config_init(
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Path for config file"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
):
    """Initialize a new configuration file."""
    if path is None:
        path = get_user_config_path()

    if path.exists() and not force:
        console.print(f"[yellow]Config file already exists: {path}[/yellow]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)

    config = Config.load_defaults()
    config.save(path)

    console.print(f"[green]Created config file: {path}[/green]")
    console.print("\nEdit this file or use 'i2i config set' to customize.")


@config_app.command("path")
def config_path():
    """Show configuration file paths."""
    from .config import get_config_paths

    console.print("[bold]Config file search paths (in priority order):[/bold]\n")

    user_path = get_user_config_path()
    console.print(f"  1. [cyan]{user_path}[/cyan]", end="")
    console.print(" [green](exists)[/green]" if user_path.exists() else " [dim](not found)[/dim]")

    project_path = Path.cwd() / "config.json"
    console.print(f"  2. [cyan]{project_path}[/cyan]", end="")
    console.print(" [green](exists)[/green]" if project_path.exists() else " [dim](not found)[/dim]")

    package_path = Path(__file__).parent.parent / "config.json"
    console.print(f"  3. [cyan]{package_path}[/cyan]", end="")
    console.print(" [green](exists)[/green]" if package_path.exists() else " [dim](not found)[/dim]")


# ============ Models Commands ============

@models_app.command("list")
def models_list(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Filter by provider"),
    configured_only: bool = typer.Option(False, "--configured", "-c", help="Only show configured providers"),
):
    """List available models."""
    registry = ProviderRegistry()

    if configured_only:
        available = registry.list_available_models()
    else:
        # Show all models from capabilities
        available = {}
        for model_id, cap in DEFAULT_MODEL_CAPABILITIES.items():
            if cap.provider not in available:
                available[cap.provider] = []
            available[cap.provider].append(model_id)

    if provider:
        available = {k: v for k, v in available.items() if k == provider}

    if not available:
        console.print("[yellow]No models found[/yellow]")
        return

    for prov, models in sorted(available.items()):
        is_configured = prov in registry.list_configured_providers()
        status = "[green]configured[/green]" if is_configured else "[red]not configured[/red]"

        console.print(f"\n[bold cyan]{prov}[/bold cyan] ({status})")
        for model in sorted(models):
            cap = DEFAULT_MODEL_CAPABILITIES.get(model)
            if cap:
                cost = f"${cap.cost_per_1k_tokens:.4f}/1k"
                latency = f"{cap.avg_latency_ms}ms"
                console.print(f"  - {model} [dim]({cost}, ~{latency})[/dim]")
            else:
                console.print(f"  - {model}")


@models_app.command("info")
def models_info(
    model: str = typer.Argument(..., help="Model ID to inspect"),
):
    """Show detailed information about a model."""
    cap = DEFAULT_MODEL_CAPABILITIES.get(model)

    if not cap:
        console.print(f"[red]Model not found: {model}[/red]")
        console.print("\nAvailable models:")
        for m in sorted(DEFAULT_MODEL_CAPABILITIES.keys()):
            console.print(f"  - {m}")
        raise typer.Exit(1)

    console.print(Panel.fit(f"[bold]{model}[/bold]"))

    console.print(f"\n[bold]Provider:[/bold] {cap.provider}")
    console.print(f"[bold]Context Window:[/bold] {cap.context_window:,} tokens")
    console.print(f"[bold]Max Output:[/bold] {cap.max_output_tokens:,} tokens")
    console.print(f"[bold]Avg Latency:[/bold] {cap.avg_latency_ms}ms")
    console.print(f"[bold]Cost:[/bold] ${cap.cost_per_1k_tokens:.4f}/1k tokens")

    # Capabilities
    console.print("\n[bold cyan]Capabilities:[/bold cyan]")
    caps = []
    if cap.supports_vision:
        caps.append("vision")
    if cap.supports_function_calling:
        caps.append("function calling")
    if cap.supports_json_mode:
        caps.append("JSON mode")
    if cap.supports_streaming:
        caps.append("streaming")
    console.print(f"  {', '.join(caps) if caps else 'None'}")

    # Quality scores
    console.print("\n[bold cyan]Quality Scores:[/bold cyan]")
    console.print(f"  Reasoning: {cap.reasoning_depth}/100")
    console.print(f"  Creativity: {cap.creativity_score}/100")
    console.print(f"  Instruction Following: {cap.instruction_following}/100")
    console.print(f"  Factual Accuracy: {cap.factual_accuracy}/100")

    # Task scores
    if cap.task_scores:
        console.print("\n[bold cyan]Task Scores:[/bold cyan]")
        table = Table(show_header=False)
        table.add_column("Task", style="dim")
        table.add_column("Score")

        for task, score in sorted(cap.task_scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
            table.add_row(task, f"{bar} {score}")

        console.print(table)


@models_app.command("compare")
def models_compare(
    models: List[str] = typer.Argument(..., help="Models to compare (2 or more)"),
):
    """Compare multiple models side by side."""
    if len(models) < 2:
        console.print("[red]Please specify at least 2 models to compare[/red]")
        raise typer.Exit(1)

    caps = []
    for model in models:
        cap = DEFAULT_MODEL_CAPABILITIES.get(model)
        if not cap:
            console.print(f"[red]Model not found: {model}[/red]")
            raise typer.Exit(1)
        caps.append(cap)

    table = Table(title="Model Comparison")
    table.add_column("Attribute", style="bold")
    for model in models:
        table.add_column(model)

    # Basic info
    table.add_row("Provider", *[c.provider for c in caps])
    table.add_row("Context", *[f"{c.context_window:,}" for c in caps])
    table.add_row("Max Output", *[f"{c.max_output_tokens:,}" for c in caps])
    table.add_row("Latency", *[f"{c.avg_latency_ms}ms" for c in caps])
    table.add_row("Cost/1k", *[f"${c.cost_per_1k_tokens:.4f}" for c in caps])
    table.add_row("", *["" for _ in caps])  # Spacer
    table.add_row("[bold]Quality[/bold]", *["" for _ in caps])
    table.add_row("Reasoning", *[f"{c.reasoning_depth}" for c in caps])
    table.add_row("Creativity", *[f"{c.creativity_score}" for c in caps])
    table.add_row("Instructions", *[f"{c.instruction_following}" for c in caps])
    table.add_row("Accuracy", *[f"{c.factual_accuracy}" for c in caps])

    console.print(table)


# ============ Main Entry Point ============

@app.command("version")
def version():
    """Show i2i version."""
    console.print("i2i v0.2.0")


# ============ Benchmark Commands ============

benchmark_app = typer.Typer(help="Run evaluation benchmarks")
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("check")
def benchmark_check(
    question: str = typer.Argument(..., help="Question to classify"),
):
    """Check if consensus is appropriate for a question."""
    from .task_classifier import recommend_consensus
    
    rec = recommend_consensus(question)
    
    status = "[green]✓ USE CONSENSUS[/green]" if rec.should_use_consensus else "[red]✗ SKIP CONSENSUS[/red]"
    console.print(f"\n{status}\n")
    console.print(f"[bold]Question:[/bold] {question}")
    console.print(f"[bold]Task Category:[/bold] {rec.task_category.value}")
    console.print(f"[bold]Classification Confidence:[/bold] {rec.confidence:.0%}")
    console.print(f"\n[bold]Reason:[/bold] {rec.reason}")
    console.print(f"\n[bold]Suggested Approach:[/bold] {rec.suggested_approach}")


@benchmark_app.command("calibration")
def benchmark_calibration():
    """Show confidence calibration table."""
    from .task_classifier import get_confidence_calibration
    
    console.print("\n[bold]Confidence Calibration (based on 400-question evaluation)[/bold]\n")
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Consensus Level")
    table.add_column("Confidence")
    table.add_column("Interpretation")
    
    levels = [
        ("HIGH (≥85% agreement)", "high", "Trust the answer"),
        ("MEDIUM (60-84%)", "medium", "Probably correct"),
        ("LOW (30-59%)", "low", "Use with caution"),
        ("NONE (<30%)", "none", "Likely hallucination"),
    ]
    
    for name, level, interp in levels:
        conf = get_confidence_calibration(level)
        table.add_row(name, f"{conf:.2f}", interp)
    
    console.print(table)
    
    console.print("\n[dim]Based on evaluation: HIGH consensus = 95-100% accuracy on factual tasks[/dim]")


@benchmark_app.command("summary")
def benchmark_summary():
    """Show summary of evaluation findings."""
    console.print("\n[bold]i2i Evaluation Summary[/bold]")
    console.print("[dim]400 questions, 5 benchmarks, 4 models (GPT-5.2, Claude, Gemini, Grok)[/dim]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Benchmark")
    table.add_column("Single Model")
    table.add_column("Consensus")
    table.add_column("Change")
    table.add_column("HIGH Acc")

    results = [
        ("TriviaQA (Factual)", "93.3%", "94.0%", "[green]+0.7%[/green]", "97.8%"),
        ("TruthfulQA", "78%", "78%", "0%", "100%"),
        ("StrategyQA", "80%", "80%", "0%", "94.7%"),
        ("Hallucination", "38%", "44%", "[green]+6%[/green]", "100%"),
        ("GSM8K (Math)", "95%", "60%", "[red]-35%[/red]", "69.9%"),
    ]

    for row in results:
        table.add_row(*row)

    console.print(table)

    console.print("\n[bold green]✓ Use consensus for:[/bold green]")
    console.print("  • Factual questions (HIGH consensus = 97-100% accuracy)")
    console.print("  • Hallucination detection (+6% improvement)")
    console.print("  • Commonsense reasoning (HIGH consensus = 95% accuracy)")

    console.print("\n[bold red]✗ Don't use consensus for:[/bold red]")
    console.print("  • Math/reasoning (consensus DEGRADES by 35%)")
    console.print("  • Creative writing (flattens diversity)")
    console.print("  • Code generation (specific correct answers)")


@benchmark_app.command("stats")
def benchmark_stats(
    results_dir: Path = typer.Option(
        Path("benchmarks/results"),
        "--dir", "-d",
        help="Directory containing benchmark results"
    ),
    n_bootstrap: int = typer.Option(10000, "--bootstrap", "-b", help="Number of bootstrap samples"),
    latex: bool = typer.Option(False, "--latex", "-l", help="Output as LaTeX table"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Run statistical significance tests on benchmark results."""
    from .statistics import load_and_analyze, format_statistics_table, format_latex_table
    from dataclasses import asdict

    # Find latest result files for each benchmark
    benchmarks = ["TruthfulQA", "ControlledHallucination", "GSM8K", "StrategyQA", "TriviaQA"]
    stats_list = []

    console.print("\n[bold]Statistical Analysis of Benchmark Results[/bold]")
    console.print(f"[dim]Bootstrap samples: {n_bootstrap:,}[/dim]\n")

    for benchmark in benchmarks:
        # Find the latest file for this benchmark
        pattern = f"{benchmark}_v2_*.json"
        files = sorted(results_dir.glob(pattern), reverse=True)

        if not files:
            console.print(f"[yellow]No results found for {benchmark}[/yellow]")
            continue

        latest_file = files[0]
        console.print(f"[dim]Loading {latest_file.name}...[/dim]")

        try:
            stats = load_and_analyze(latest_file, n_bootstrap=n_bootstrap)
            stats_list.append(stats)
        except Exception as e:
            console.print(f"[red]Error loading {benchmark}: {e}[/red]")

    if not stats_list:
        console.print("[red]No benchmark results found[/red]")
        raise typer.Exit(1)

    console.print()

    if json_output:
        # Output as JSON
        output = []
        for stats in stats_list:
            output.append({
                "benchmark": stats.benchmark_name,
                "n_samples": stats.n_samples,
                "single_model_accuracy": {
                    "point_estimate": stats.single_model_accuracy.point_estimate,
                    "ci_lower": stats.single_model_accuracy.ci_lower,
                    "ci_upper": stats.single_model_accuracy.ci_upper,
                },
                "consensus_accuracy": {
                    "point_estimate": stats.consensus_accuracy.point_estimate,
                    "ci_lower": stats.consensus_accuracy.ci_lower,
                    "ci_upper": stats.consensus_accuracy.ci_upper,
                },
                "improvement": {
                    "point_estimate": stats.improvement.point_estimate,
                    "ci_lower": stats.improvement.ci_lower,
                    "ci_upper": stats.improvement.ci_upper,
                },
                "mcnemar": {
                    "chi_square": stats.mcnemar.chi_square_corrected,
                    "p_value": stats.mcnemar.p_value,
                    "significant_05": stats.mcnemar.significant_at_05,
                    "significant_01": stats.mcnemar.significant_at_01,
                    "cohens_g": stats.mcnemar.cohens_g,
                    "direction": stats.mcnemar.direction,
                } if stats.mcnemar else None,
            })
        console.print(Syntax(json.dumps(output, indent=2), "json"))
    elif latex:
        # Output as LaTeX
        console.print(format_latex_table(stats_list))
    else:
        # Pretty table output
        console.print(format_statistics_table(stats_list))

        # Additional details
        console.print("\n[bold]McNemar's Test Details[/bold]\n")
        for stats in stats_list:
            if stats.mcnemar:
                mcn = stats.mcnemar
                console.print(f"[cyan]{stats.benchmark_name}[/cyan]")
                console.print(f"  Contingency: both_correct={mcn.n_both_correct}, single_only={mcn.n_single_only}, consensus_only={mcn.n_consensus_only}, both_wrong={mcn.n_both_wrong}")
                console.print(f"  Chi-square (corrected): {mcn.chi_square_corrected:.3f}")
                console.print(f"  Effect size (Cohen's g): {mcn.cohens_g:.3f}")
                if mcn.p_value_exact is not None:
                    console.print(f"  p-value (exact binomial): {mcn.p_value_exact:.4f}")
                else:
                    console.print(f"  p-value: {mcn.p_value:.4f}")
                console.print()


@benchmark_app.command("mcnemar")
def benchmark_mcnemar(
    results_file: Path = typer.Argument(..., help="Path to benchmark results JSON"),
):
    """Run McNemar's test on a single benchmark file."""
    from .statistics import load_and_analyze

    if not results_file.exists():
        console.print(f"[red]File not found: {results_file}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]McNemar's Test Analysis[/bold]")
    console.print(f"[dim]File: {results_file}[/dim]\n")

    stats = load_and_analyze(results_file)

    if stats.mcnemar:
        console.print(str(stats.mcnemar))

        console.print(f"\n[bold]Accuracy with 95% Bootstrap CIs[/bold]")
        console.print(f"  Single Model: {stats.single_model_accuracy}")
        console.print(f"  Consensus:    {stats.consensus_accuracy}")
        console.print(f"  Improvement:  {stats.improvement}")

        if stats.high_consensus_accuracy:
            console.print(f"\n[bold]By Consensus Level[/bold]")
            console.print(f"  HIGH:   {stats.high_consensus_accuracy}")
        if stats.medium_consensus_accuracy:
            console.print(f"  MEDIUM: {stats.medium_consensus_accuracy}")
        if stats.low_consensus_accuracy:
            console.print(f"  LOW:    {stats.low_consensus_accuracy}")
    else:
        console.print("[yellow]Could not compute McNemar's test[/yellow]")


@app.command("status")
def status():
    """Show i2i status and configured providers."""
    registry = ProviderRegistry()
    config = Config.load()

    console.print(Panel.fit("[bold]i2i Status[/bold]"))

    # Providers
    console.print("\n[bold cyan]Providers:[/bold cyan]")
    configured = registry.list_configured_providers()
    all_providers = ["openai", "anthropic", "google", "mistral", "groq", "cohere"]

    for prov in all_providers:
        if prov in configured:
            console.print(f"  [green]✓[/green] {prov}")
        else:
            console.print(f"  [red]✗[/red] {prov} [dim](API key not set)[/dim]")

    # Current models
    console.print("\n[bold cyan]Active Models:[/bold cyan]")
    console.print(f"  Consensus: {', '.join(config.consensus_models)}")
    console.print(f"  Classifier: {config.classifier_model}")

    # Config source
    if config.source_path:
        console.print(f"\n[dim]Config: {config.source_path}[/dim]")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
