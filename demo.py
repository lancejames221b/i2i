#!/usr/bin/env python3
"""
i2i Demo CLI

Demonstrates the AI-to-AI Communication Protocol capabilities:
- Consensus queries across multiple models
- Cross-verification of claims
- Epistemic classification
- AI debates
- Intelligent model routing
"""

import asyncio
import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer

# Add parent directory to path for imports
sys.path.insert(0, '.')

from i2i import (
    AICP,
    ConsensusLevel,
    EpistemicType,
    ModelRouter,
    TaskType,
    RoutingStrategy,
    ProviderRegistry,
)

app = typer.Typer(help="i2i - AI-to-AI Communication Protocol Demo")
console = Console()


def print_header():
    """Print the demo header."""
    console.print(Panel.fit(
        "[bold blue]i2i - AI-to-AI Communication Protocol[/bold blue]\n"
        "[dim]Multi-model consensus, verification, routing, and epistemic classification[/dim]",
        border_style="blue"
    ))


@app.command()
def status():
    """Check which AI providers are configured."""
    print_header()
    console.print("\n[bold]Provider Status:[/bold]\n")

    protocol = AICP()
    available = protocol.list_available_models()

    table = Table(title="Configured Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Models", style="green")
    table.add_column("Status", style="yellow")

    for provider, models in available.items():
        table.add_row(
            provider,
            ", ".join(models[:3]) + ("..." if len(models) > 3 else ""),
            "✓ Ready"
        )

    console.print(table)

    configured = protocol.list_configured_providers()
    console.print(f"\n[bold green]✓ {len(configured)} providers configured[/bold green]")

    if len(configured) < 2:
        console.print("[bold red]⚠ Need at least 2 providers for consensus features[/bold red]")


@app.command()
def consensus(
    query: str = typer.Argument(..., help="The question to ask"),
    models: Optional[str] = typer.Option(None, "--models", "-m", help="Comma-separated model list"),
):
    """Query multiple AI models and find consensus."""
    print_header()
    asyncio.run(_consensus_async(query, models))


async def _consensus_async(query: str, models_str: Optional[str]):
    """Async implementation of consensus command."""
    protocol = AICP()

    models = None
    if models_str:
        models = [m.strip() for m in models_str.split(",")]

    console.print(f"\n[bold]Query:[/bold] {query}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Querying models for consensus...", total=None)

        try:
            result = await protocol.consensus_query(query, models=models)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

        progress.remove_task(task)

    # Display results
    console.print(Panel.fit(
        f"[bold]Consensus Level:[/bold] {_format_consensus_level(result.consensus_level)}\n"
        f"[bold]Models Queried:[/bold] {len(result.models_queried)}",
        title="Results",
        border_style="green" if result.consensus_level in [ConsensusLevel.HIGH, ConsensusLevel.MEDIUM] else "yellow"
    ))

    # Show consensus answer if available
    if result.consensus_answer:
        console.print("\n[bold]Consensus Answer:[/bold]")
        console.print(Panel(Markdown(result.consensus_answer), border_style="blue"))

    # Show divergences if any
    if result.divergences:
        console.print(f"\n[bold yellow]Divergences Found: {len(result.divergences)}[/bold yellow]")
        for div in result.divergences[:3]:
            console.print(f"  • {div['summary']}")

    # Show individual responses
    console.print("\n[bold]Individual Responses:[/bold]")
    for resp in result.responses:
        console.print(Panel(
            resp.content[:500] + ("..." if len(resp.content) > 500 else ""),
            title=f"[cyan]{resp.model}[/cyan] (confidence: {resp.confidence.value})",
            border_style="dim"
        ))


@app.command()
def verify(
    claim: str = typer.Argument(..., help="The claim to verify"),
    verifiers: Optional[str] = typer.Option(None, "--verifiers", "-v", help="Comma-separated verifier models"),
):
    """Have multiple AI models verify a claim."""
    print_header()
    asyncio.run(_verify_async(claim, verifiers))


async def _verify_async(claim: str, verifiers_str: Optional[str]):
    """Async implementation of verify command."""
    protocol = AICP()

    verifiers = None
    if verifiers_str:
        verifiers = [v.strip() for v in verifiers_str.split(",")]

    console.print(f"\n[bold]Claim:[/bold] {claim}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Verifying claim...", total=None)

        try:
            result = await protocol.verify_claim(claim, verifiers=verifiers)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

        progress.remove_task(task)

    # Display results
    status_color = "green" if result.verified else "red"
    status_text = "✓ VERIFIED" if result.verified else "✗ NOT VERIFIED"

    console.print(Panel.fit(
        f"[bold {status_color}]{status_text}[/bold {status_color}]\n"
        f"[bold]Confidence:[/bold] {result.confidence:.1%}\n"
        f"[bold]Verifiers:[/bold] {', '.join(result.verifiers)}",
        title="Verification Result",
        border_style=status_color
    ))

    if result.issues_found:
        console.print("\n[bold yellow]Issues Found:[/bold yellow]")
        for issue in result.issues_found:
            console.print(f"  • {issue}")

    if result.corrections:
        console.print("\n[bold]Suggested Corrections:[/bold]")
        console.print(Panel(result.corrections, border_style="yellow"))


@app.command()
def classify(
    question: str = typer.Argument(..., help="The question to classify"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Use quick heuristic (no API calls)"),
):
    """Classify a question's epistemic status."""
    print_header()

    if quick:
        protocol = AICP()
        result = protocol.quick_classify(question)
        console.print(f"\n[bold]Question:[/bold] {question}")
        console.print(f"\n[bold]Quick Classification:[/bold] {_format_epistemic_type(result)}")
        console.print("[dim](This is a heuristic classification - use without --quick for full analysis)[/dim]")
    else:
        asyncio.run(_classify_async(question))


async def _classify_async(question: str):
    """Async implementation of classify command."""
    protocol = AICP()

    console.print(f"\n[bold]Question:[/bold] {question}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Classifying question...", total=None)

        try:
            result = await protocol.classify_question(question)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

        progress.remove_task(task)

    # Display results
    console.print(Panel.fit(
        f"[bold]Classification:[/bold] {_format_epistemic_type(result.classification)}\n"
        f"[bold]Confidence:[/bold] {result.confidence:.1%}\n"
        f"[bold]Actionable:[/bold] {'Yes' if result.is_actionable else 'No'}",
        title="Epistemic Classification",
        border_style=_get_epistemic_color(result.classification)
    ))

    console.print("\n[bold]Reasoning:[/bold]")
    console.print(Panel(result.reasoning, border_style="dim"))

    if result.classification == EpistemicType.UNDERDETERMINED and result.competing_hypotheses:
        console.print("\n[bold]Competing Hypotheses:[/bold]")
        for hyp in result.competing_hypotheses:
            console.print(f"  • {hyp}")

    if result.classification == EpistemicType.IDLE and result.why_idle:
        console.print("\n[bold]Why Idle:[/bold]")
        console.print(f"  {result.why_idle}")

    if result.suggested_reformulation:
        console.print("\n[bold]Suggested Reformulation:[/bold]")
        console.print(Panel(result.suggested_reformulation, border_style="green"))


@app.command()
def smart(
    query: str = typer.Argument(..., help="Your question"),
    verify_answer: bool = typer.Option(False, "--verify", help="Verify the final answer"),
):
    """Smart query that adapts based on question type."""
    print_header()
    asyncio.run(_smart_async(query, verify_answer))


async def _smart_async(query: str, verify_answer: bool):
    """Async implementation of smart command."""
    protocol = AICP()

    console.print(f"\n[bold]Query:[/bold] {query}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing smart query...", total=None)

        try:
            result = await protocol.smart_query(
                query,
                require_consensus=True,
                verify_result=verify_answer,
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

        progress.remove_task(task)

    # Show classification
    if result["classification"]:
        console.print(Panel.fit(
            f"[bold]Classification:[/bold] {_format_epistemic_type(result['classification'].classification)}",
            title="Question Analysis",
            border_style="cyan"
        ))

    # Show warnings
    if result["warnings"]:
        for warning in result["warnings"]:
            console.print(f"[yellow]⚠ {warning}[/yellow]")

    # Show consensus info
    if result["consensus"]:
        consensus = result["consensus"]
        console.print(f"\n[bold]Consensus:[/bold] {consensus['level']} ({consensus['models_queried']} models)")

    # Show answer
    if result["answer"]:
        console.print("\n[bold]Answer:[/bold]")
        console.print(Panel(Markdown(result["answer"]), border_style="green"))

    # Show verification
    if result["verification"]:
        ver = result["verification"]
        status = "✓ Verified" if ver["verified"] else "✗ Not Verified"
        console.print(f"\n[bold]Verification:[/bold] {status} ({ver['confidence']:.1%} confidence)")


@app.command()
def debate(
    topic: str = typer.Argument(..., help="The debate topic"),
    rounds: int = typer.Option(2, "--rounds", "-r", help="Number of debate rounds"),
    models: Optional[str] = typer.Option(None, "--models", "-m", help="Comma-separated model list"),
):
    """Have multiple AI models debate a topic."""
    print_header()
    asyncio.run(_debate_async(topic, rounds, models))


async def _debate_async(topic: str, rounds: int, models_str: Optional[str]):
    """Async implementation of debate command."""
    protocol = AICP()

    model_list = None
    if models_str:
        model_list = [m.strip() for m in models_str.split(",")]

    console.print(f"\n[bold]Debate Topic:[/bold] {topic}")
    console.print(f"[bold]Rounds:[/bold] {rounds}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running debate...", total=None)

        try:
            result = await protocol.debate(topic, models=model_list, rounds=rounds)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

        progress.remove_task(task)

    # Display debate
    console.print(f"\n[bold]Participants:[/bold] {', '.join(result['participants'])}\n")

    for round_data in result["rounds"]:
        console.print(f"\n[bold cyan]═══ Round {round_data['round']} ({round_data['type']}) ═══[/bold cyan]\n")
        for resp in round_data["responses"]:
            console.print(Panel(
                resp["content"][:800] + ("..." if len(resp["content"]) > 800 else ""),
                title=f"[bold]{resp['model']}[/bold]",
                border_style="blue"
            ))

    # Show summary
    if result["summary"]:
        console.print("\n[bold green]═══ Debate Summary ═══[/bold green]\n")
        console.print(Panel(Markdown(result["summary"]), border_style="green"))


# ==================== Routing Commands ====================

@app.command()
def route(
    query: str = typer.Argument(..., help="The query to route"),
    strategy: str = typer.Option("balanced", "--strategy", "-s",
                                  help="Strategy: best_quality, best_speed, best_value, balanced, ensemble"),
    execute: bool = typer.Option(False, "--execute", "-e", help="Execute the query after routing"),
):
    """Route a query to the optimal model(s)."""
    print_header()
    asyncio.run(_route_async(query, strategy, execute))


async def _route_async(query: str, strategy_str: str, execute: bool):
    """Async implementation of route command."""
    registry = ProviderRegistry()
    router = ModelRouter(registry)

    # Parse strategy
    try:
        strategy = RoutingStrategy(strategy_str)
    except ValueError:
        console.print(f"[red]Invalid strategy: {strategy_str}[/red]")
        console.print("Valid strategies: best_quality, best_speed, best_value, balanced, ensemble")
        return

    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print(f"[bold]Strategy:[/bold] {strategy.value}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing query and selecting model...", total=None)

        try:
            if execute:
                result = await router.route_and_execute(query, strategy=strategy)
                decision = result.decision
            else:
                decision = await router.route(query, strategy=strategy)
                result = None
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

        progress.remove_task(task)

    # Display routing decision
    console.print(Panel.fit(
        f"[bold]Detected Task:[/bold] {_format_task_type(decision.detected_task)} "
        f"(confidence: {decision.task_confidence:.0%})\n"
        f"[bold]Selected Model(s):[/bold] {', '.join(decision.selected_models)}\n"
        f"[bold]Strategy:[/bold] {decision.strategy_used.value}\n"
        f"[bold]Est. Latency:[/bold] {decision.estimated_latency_ms:.0f}ms\n"
        f"[bold]Est. Cost:[/bold] ${decision.estimated_cost:.4f}" if decision.estimated_cost else "",
        title="Routing Decision",
        border_style="cyan"
    ))

    console.print(f"\n[dim]{decision.reasoning}[/dim]")

    # Show alternatives
    if decision.alternatives:
        console.print("\n[bold]Alternatives:[/bold]")
        for alt in decision.alternatives[:3]:
            console.print(f"  • {alt['model']} (score: {alt['score']:.1f})")

    # Show execution result if requested
    if execute and result:
        console.print(f"\n[bold green]═══ Execution Result ═══[/bold green]")
        console.print(f"[dim]Actual latency: {result.actual_latency_ms:.0f}ms[/dim]\n")

        for resp in result.responses:
            console.print(Panel(
                resp.content[:800] + ("..." if len(resp.content) > 800 else ""),
                title=f"[cyan]{resp.model}[/cyan]",
                border_style="green"
            ))

        if result.synthesized_response:
            console.print("\n[bold]Synthesized Response:[/bold]")
            console.print(Panel(Markdown(result.synthesized_response), border_style="blue"))


@app.command()
def recommend(
    task: str = typer.Argument(..., help="Task type (e.g., code_generation, creative_writing, mathematical)"),
):
    """Get model recommendations for a task type."""
    print_header()

    # Parse task type
    try:
        task_type = TaskType(task)
    except ValueError:
        console.print(f"[red]Invalid task type: {task}[/red]")
        console.print("\n[bold]Valid task types:[/bold]")
        for t in TaskType:
            console.print(f"  • {t.value}")
        return

    registry = ProviderRegistry()
    router = ModelRouter(registry)

    recommendations = router.get_model_recommendation(task_type)

    if "error" in recommendations:
        console.print(f"[red]{recommendations['error']}[/red]")
        return

    console.print(f"\n[bold]Recommendations for:[/bold] {_format_task_type(task_type)}\n")

    table = Table(title="Model Recommendations by Strategy")
    table.add_column("Strategy", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Score", style="yellow")
    table.add_column("Latency", style="magenta")
    table.add_column("Cost/1K", style="red")

    for strategy in ["best_quality", "best_speed", "best_value", "balanced"]:
        rec = recommendations.get(strategy)
        if rec:
            table.add_row(
                strategy.replace("_", " ").title(),
                rec["model"],
                f"{rec['score']:.1f}",
                f"{rec['latency_ms']:.0f}ms" if rec.get("latency_ms") else "-",
                f"${rec['cost_per_1k']:.4f}" if rec.get("cost_per_1k") else "-",
            )

    console.print(table)


@app.command()
def tasks():
    """List all supported task types."""
    print_header()

    console.print("\n[bold]Supported Task Types:[/bold]\n")

    categories = {
        "Reasoning & Analysis": [TaskType.LOGICAL_REASONING, TaskType.MATHEMATICAL,
                                  TaskType.SCIENTIFIC, TaskType.ANALYTICAL],
        "Creative": [TaskType.CREATIVE_WRITING, TaskType.COPYWRITING, TaskType.BRAINSTORMING],
        "Technical": [TaskType.CODE_GENERATION, TaskType.CODE_REVIEW,
                      TaskType.CODE_DEBUGGING, TaskType.TECHNICAL_DOCS],
        "Knowledge & Research": [TaskType.FACTUAL_QA, TaskType.RESEARCH,
                                  TaskType.SUMMARIZATION, TaskType.TRANSLATION],
        "Conversation": [TaskType.CHAT, TaskType.ROLEPLAY, TaskType.INSTRUCTION_FOLLOWING],
        "Specialized": [TaskType.LEGAL, TaskType.MEDICAL, TaskType.FINANCIAL],
    }

    for category, task_types in categories.items():
        console.print(f"\n[bold cyan]{category}[/bold cyan]")
        for tt in task_types:
            console.print(f"  • {tt.value}")


@app.command()
def models():
    """List all models with their capability scores."""
    print_header()

    registry = ProviderRegistry()
    router = ModelRouter(registry)

    available = router.get_available_models()

    if not available:
        console.print("[red]No models available. Configure API keys in .env[/red]")
        return

    console.print(f"\n[bold]Available Models:[/bold] {len(available)}\n")

    for model_id in available:
        cap = router.capabilities.get(model_id)
        if cap:
            # Get top 3 tasks for this model
            top_tasks = sorted(cap.task_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_tasks_str = ", ".join([f"{t}:{s:.0f}" for t, s in top_tasks])

            console.print(Panel(
                f"[bold]Provider:[/bold] {cap.provider}\n"
                f"[bold]Latency:[/bold] {cap.avg_latency_ms:.0f}ms | "
                f"[bold]Cost:[/bold] ${cap.cost_per_1k_tokens:.4f}/1K\n"
                f"[bold]Context:[/bold] {cap.context_window:,} tokens\n"
                f"[bold]Best at:[/bold] {top_tasks_str}\n"
                f"[bold]Features:[/bold] "
                f"{'👁 Vision ' if cap.supports_vision else ''}"
                f"{'🔧 Functions ' if cap.supports_function_calling else ''}"
                f"{'📋 JSON ' if cap.supports_json_mode else ''}",
                title=f"[cyan]{model_id}[/cyan]",
                border_style="dim"
            ))


# ==================== Ablation Commands ====================

@app.command()
def ablation(
    mode: str = typer.Option("quick", "--mode", "-m",
                              help="Mode: quick (5 tasks), standard (15 tasks), full (18 tasks)"),
    output: str = typer.Option("benchmarks/results/ablation", "--output", "-o",
                                help="Output directory for results"),
):
    """Run threshold ablation study to analyze consensus threshold sensitivity."""
    print_header()
    console.print("\n[bold]Threshold Ablation Study[/bold]\n")
    console.print(f"Mode: {mode}")
    console.print(f"Output: {output}\n")

    asyncio.run(_ablation_async(mode, output))


async def _ablation_async(mode: str, output: str):
    """Async implementation of ablation command."""
    from benchmarks.threshold_ablation import (
        ThresholdAblationEngine,
        FACTUAL_DATASET,
        REASONING_DATASET,
        AMBIGUOUS_DATASET,
    )

    engine = ThresholdAblationEngine(results_dir=output)

    # Select tasks based on mode
    if mode == "quick":
        tasks = FACTUAL_DATASET[:5]
        name = "quick_ablation"
    elif mode == "standard":
        tasks = FACTUAL_DATASET + REASONING_DATASET
        name = "standard_ablation"
    else:
        tasks = FACTUAL_DATASET + REASONING_DATASET + AMBIGUOUS_DATASET
        name = "full_ablation"

    console.print(f"[dim]Running {len(tasks)} tasks...[/dim]\n")

    try:
        study = await engine.run_full_ablation(tasks, name=name)

        # Display summary
        console.print("\n[bold green]Ablation Study Complete[/bold green]\n")

        table = Table(title="Threshold Configuration Results")
        table.add_column("Config", style="cyan")
        table.add_column("Overall Acc", style="green")
        table.add_column("HIGH Acc", style="yellow")
        table.add_column("Calibration", style="magenta")
        table.add_column("Reliability", style="blue")

        for m in study.config_metrics:
            thresh = m["thresholds"]
            config_str = f"H:{thresh['high']:.2f}/M:{thresh['medium']:.2f}/L:{thresh['low']:.2f}"
            table.add_row(
                config_str,
                f"{m['overall_accuracy']:.1f}%",
                f"{m['high']['accuracy']:.1f}%",
                f"{m['calibration_error']:.3f}",
                f"{m['reliability_score']:.3f}",
            )

        console.print(table)

        if study.optimal_config:
            console.print(f"\n[bold]Optimal Configuration:[/bold]")
            console.print(f"  HIGH:   {study.optimal_config['high']:.2f}")
            console.print(f"  MEDIUM: {study.optimal_config['medium']:.2f}")
            console.print(f"  LOW:    {study.optimal_config['low']:.2f}")
            console.print(f"  Score:  {study.optimal_score:.4f}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def thresholds():
    """Show current consensus threshold configuration."""
    print_header()

    from i2i.config import (
        get_consensus_thresholds,
        get_clustering_threshold,
        get_divergence_threshold,
    )

    console.print("\n[bold]Consensus Thresholds[/bold]\n")

    thresholds = get_consensus_thresholds()

    table = Table(title="Current Configuration")
    table.add_column("Level", style="cyan")
    table.add_column("Threshold", style="green")
    table.add_column("Description", style="dim")

    table.add_row("HIGH", f">= {thresholds['high']:.0%}", "Strong agreement between models")
    table.add_row("MEDIUM", f">= {thresholds['medium']:.0%}", "Moderate agreement")
    table.add_row("LOW", f">= {thresholds['low']:.0%}", "Weak agreement")
    table.add_row("NONE", f"< {thresholds['low']:.0%}", "No meaningful agreement")

    console.print(table)

    console.print(f"\n[dim]Clustering threshold: {get_clustering_threshold():.2f}[/dim]")
    console.print(f"[dim]Divergence threshold: {get_divergence_threshold():.2f}[/dim]")

    console.print("\n[bold]Configure via:[/bold]")
    console.print("  Environment: I2I_CONSENSUS_HIGH_THRESHOLD, etc.")
    console.print("  Config file: ~/.i2i/config.json")


@app.command()
def sensitivity(
    parameter: str = typer.Argument(..., help="Parameter to analyze: high, medium, or low"),
):
    """Show sensitivity curve for a specific threshold parameter."""
    print_header()

    if parameter not in ["high", "medium", "low"]:
        console.print(f"[red]Invalid parameter: {parameter}[/red]")
        console.print("Valid parameters: high, medium, low")
        return

    # Load recent ablation results if available
    from pathlib import Path
    import json

    results_dir = Path("benchmarks/results/ablation")
    if not results_dir.exists():
        console.print("[yellow]No ablation results found. Run 'demo.py ablation' first.[/yellow]")
        return

    # Find most recent results
    results_files = sorted(results_dir.glob("*.json"), reverse=True)
    if not results_files:
        console.print("[yellow]No ablation results found. Run 'demo.py ablation' first.[/yellow]")
        return

    with open(results_files[0]) as f:
        study = json.load(f)

    sensitivity_key = f"sensitivity_{parameter}"
    if sensitivity_key not in study or not study[sensitivity_key]:
        console.print(f"[yellow]No sensitivity data for {parameter} in results.[/yellow]")
        return

    console.print(f"\n[bold]Sensitivity Analysis: {parameter.upper()} Threshold[/bold]\n")

    table = Table(title=f"{parameter.upper()} Threshold Sensitivity")
    table.add_column("Threshold", style="cyan")
    table.add_column("Overall Acc", style="green")
    table.add_column(f"{parameter.upper()} Acc", style="yellow")
    table.add_column("Calibration", style="magenta")

    for entry in study[sensitivity_key]:
        acc_key = f"{parameter}_accuracy" if parameter != "high" else "high_accuracy"
        acc_value = entry.get(acc_key, entry.get("accuracy", 0))
        table.add_row(
            f"{entry['value']:.2f}",
            f"{entry['accuracy']:.1f}%",
            f"{acc_value:.1f}%",
            f"{entry['calibration']:.3f}",
        )

    console.print(table)


# ==================== Helper Functions ====================

def _format_task_type(task_type: TaskType) -> str:
    """Format task type with color."""
    colors = {
        "code": "green",
        "creative": "magenta",
        "mathematical": "yellow",
        "logical": "cyan",
        "factual": "blue",
        "chat": "white",
    }
    # Find matching color based on task name
    color = "white"
    for key, c in colors.items():
        if key in task_type.value:
            color = c
            break
    return f"[{color}]{task_type.value.upper().replace('_', ' ')}[/{color}]"


def _format_consensus_level(level: ConsensusLevel) -> str:
    """Format consensus level with color."""
    colors = {
        ConsensusLevel.HIGH: "bold green",
        ConsensusLevel.MEDIUM: "yellow",
        ConsensusLevel.LOW: "orange3",
        ConsensusLevel.NONE: "red",
        ConsensusLevel.CONTRADICTORY: "bold red",
    }
    return f"[{colors.get(level, 'white')}]{level.value.upper()}[/{colors.get(level, 'white')}]"


def _format_epistemic_type(etype: EpistemicType) -> str:
    """Format epistemic type with color."""
    colors = {
        EpistemicType.ANSWERABLE: "bold green",
        EpistemicType.UNCERTAIN: "yellow",
        EpistemicType.UNDERDETERMINED: "orange3",
        EpistemicType.IDLE: "cyan",
        EpistemicType.MALFORMED: "red",
    }
    return f"[{colors.get(etype, 'white')}]{etype.value.upper()}[/{colors.get(etype, 'white')}]"


def _get_epistemic_color(etype: EpistemicType) -> str:
    """Get border color for epistemic type."""
    colors = {
        EpistemicType.ANSWERABLE: "green",
        EpistemicType.UNCERTAIN: "yellow",
        EpistemicType.UNDERDETERMINED: "orange3",
        EpistemicType.IDLE: "cyan",
        EpistemicType.MALFORMED: "red",
    }
    return colors.get(etype, "white")


if __name__ == "__main__":
    app()
