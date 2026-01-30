#!/usr/bin/env python3
"""
i2i Benchmark Harness

Evaluates multi-model consensus against single-model baselines
across factual QA, reasoning, and verification tasks.
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import random

# Add parent dir to path for i2i imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from i2i import AICP
from i2i.schema import ConsensusLevel


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    task_id: str
    task_type: str
    question: str
    ground_truth: str
    
    # Single model results
    single_model_answer: Optional[str] = None
    single_model_correct: Optional[bool] = None
    single_model_confidence: Optional[str] = None
    
    # Consensus results
    consensus_answer: Optional[str] = None
    consensus_level: Optional[str] = None
    consensus_correct: Optional[bool] = None
    models_queried: Optional[List[str]] = None
    
    # Metadata
    latency_single_ms: Optional[float] = None
    latency_consensus_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass 
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    timestamp: str
    task_type: str
    total_tasks: int
    results: List[BenchmarkResult]
    
    # Aggregate metrics
    single_model_accuracy: float = 0.0
    consensus_accuracy: float = 0.0
    high_consensus_accuracy: float = 0.0
    medium_consensus_accuracy: float = 0.0
    low_consensus_accuracy: float = 0.0
    
    avg_latency_single_ms: float = 0.0
    avg_latency_consensus_ms: float = 0.0
    
    consensus_level_distribution: Dict[str, int] = None


class BenchmarkHarness:
    """Runs benchmarks comparing single-model vs multi-model consensus."""
    
    def __init__(
        self,
        models: List[str] = None,
        single_model: str = "gpt-4o",
        results_dir: str = "benchmarks/results"
    ):
        self.protocol = AICP()
        self.models = models or ["gpt-4o", "claude-sonnet-4-5-20250929", "gemini-2.0-flash"]
        self.single_model = single_model
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_single_query(self, question: str) -> tuple:
        """Run single model query, return (answer, confidence, latency_ms)."""
        import time
        start = time.time()
        try:
            response = await self.protocol.query(question, model=self.single_model)
            latency = (time.time() - start) * 1000
            return response.content, response.confidence if hasattr(response, 'confidence') else None, latency
        except Exception as e:
            return None, None, 0
    
    async def run_consensus_query(self, question: str) -> tuple:
        """Run consensus query, return (answer, level, models, latency_ms)."""
        import time
        start = time.time()
        try:
            result = await self.protocol.consensus_query(question, models=self.models)
            latency = (time.time() - start) * 1000
            return (
                result.consensus_answer,
                result.consensus_level.value if result.consensus_level else None,
                result.models_queried,
                latency
            )
        except Exception as e:
            return None, None, None, 0
    
    def check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth (fuzzy match)."""
        if not predicted or not ground_truth:
            return False
        
        # Normalize
        pred_lower = predicted.lower().strip()
        truth_lower = ground_truth.lower().strip()
        
        # Exact match
        if pred_lower == truth_lower:
            return True
        
        # Contains match (for longer answers)
        if truth_lower in pred_lower or pred_lower in truth_lower:
            return True
        
        # Token overlap for longer answers
        pred_tokens = set(pred_lower.split())
        truth_tokens = set(truth_lower.split())
        if len(truth_tokens) > 2:
            overlap = len(pred_tokens & truth_tokens) / len(truth_tokens)
            if overlap > 0.7:
                return True
        
        return False
    
    async def run_task(self, task: Dict[str, Any], task_type: str) -> BenchmarkResult:
        """Run a single benchmark task."""
        result = BenchmarkResult(
            task_id=task.get("id", str(random.randint(1000, 9999))),
            task_type=task_type,
            question=task["question"],
            ground_truth=task["answer"]
        )
        
        try:
            # Run single model
            single_answer, single_conf, single_latency = await self.run_single_query(task["question"])
            result.single_model_answer = single_answer
            result.single_model_confidence = single_conf
            result.latency_single_ms = single_latency
            result.single_model_correct = self.check_answer(single_answer, task["answer"])
            
            # Run consensus
            cons_answer, cons_level, models, cons_latency = await self.run_consensus_query(task["question"])
            result.consensus_answer = cons_answer
            result.consensus_level = cons_level
            result.models_queried = models
            result.latency_consensus_ms = cons_latency
            result.consensus_correct = self.check_answer(cons_answer, task["answer"])
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def compute_metrics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compute aggregate metrics from results."""
        valid_results = [r for r in results if r.error is None]
        
        if not valid_results:
            return {}
        
        # Accuracy
        single_correct = sum(1 for r in valid_results if r.single_model_correct)
        consensus_correct = sum(1 for r in valid_results if r.consensus_correct)
        
        # By consensus level
        high_results = [r for r in valid_results if r.consensus_level == "HIGH"]
        medium_results = [r for r in valid_results if r.consensus_level == "MEDIUM"]
        low_results = [r for r in valid_results if r.consensus_level == "LOW"]
        none_results = [r for r in valid_results if r.consensus_level in ["NONE", "CONTRADICTORY"]]
        
        high_correct = sum(1 for r in high_results if r.consensus_correct) if high_results else 0
        medium_correct = sum(1 for r in medium_results if r.consensus_correct) if medium_results else 0
        low_correct = sum(1 for r in low_results if r.consensus_correct) if low_results else 0
        
        # Latency
        single_latencies = [r.latency_single_ms for r in valid_results if r.latency_single_ms]
        consensus_latencies = [r.latency_consensus_ms for r in valid_results if r.latency_consensus_ms]
        
        return {
            "single_model_accuracy": single_correct / len(valid_results) * 100,
            "consensus_accuracy": consensus_correct / len(valid_results) * 100,
            "high_consensus_accuracy": (high_correct / len(high_results) * 100) if high_results else 0,
            "medium_consensus_accuracy": (medium_correct / len(medium_results) * 100) if medium_results else 0,
            "low_consensus_accuracy": (low_correct / len(low_results) * 100) if low_results else 0,
            "avg_latency_single_ms": sum(single_latencies) / len(single_latencies) if single_latencies else 0,
            "avg_latency_consensus_ms": sum(consensus_latencies) / len(consensus_latencies) if consensus_latencies else 0,
            "consensus_level_distribution": {
                "HIGH": len(high_results),
                "MEDIUM": len(medium_results),
                "LOW": len(low_results),
                "NONE/CONTRADICTORY": len(none_results)
            }
        }
    
    async def run_benchmark(
        self,
        tasks: List[Dict[str, Any]],
        task_type: str,
        name: str,
        limit: int = None
    ) -> BenchmarkSuite:
        """Run a full benchmark suite."""
        if limit:
            tasks = tasks[:limit]
        
        print(f"\n{'='*60}")
        print(f"Running benchmark: {name}")
        print(f"Task type: {task_type}")
        print(f"Tasks: {len(tasks)}")
        print(f"Models: {self.models}")
        print(f"{'='*60}\n")
        
        results = []
        for i, task in enumerate(tasks):
            print(f"  [{i+1}/{len(tasks)}] {task['question'][:60]}...")
            result = await self.run_task(task, task_type)
            results.append(result)
            
            # Progress indicator
            status = "✓" if result.consensus_correct else "✗"
            level = result.consensus_level or "ERR"
            print(f"           {status} Consensus: {level}")
            
            # Rate limit protection
            await asyncio.sleep(0.5)
        
        # Compute metrics
        metrics = self.compute_metrics(results)
        
        suite = BenchmarkSuite(
            name=name,
            timestamp=datetime.utcnow().isoformat(),
            task_type=task_type,
            total_tasks=len(tasks),
            results=results,
            **metrics
        )
        
        # Save results
        output_path = self.results_dir / f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(asdict(suite), f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_path}")
        print(f"\nSingle Model Accuracy: {metrics.get('single_model_accuracy', 0):.1f}%")
        print(f"Consensus Accuracy:    {metrics.get('consensus_accuracy', 0):.1f}%")
        print(f"HIGH Consensus Acc:    {metrics.get('high_consensus_accuracy', 0):.1f}%")
        print(f"MEDIUM Consensus Acc:  {metrics.get('medium_consensus_accuracy', 0):.1f}%")
        print(f"Level Distribution:    {metrics.get('consensus_level_distribution', {})}")
        print(f"{'='*60}\n")
        
        return suite


# Sample datasets for testing
FACTUAL_QA_SAMPLE = [
    {"id": "fq1", "question": "What is the capital of France?", "answer": "Paris"},
    {"id": "fq2", "question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"id": "fq3", "question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"id": "fq4", "question": "In what year did World War II end?", "answer": "1945"},
    {"id": "fq5", "question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"id": "fq6", "question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"id": "fq7", "question": "What is the speed of light in meters per second?", "answer": "299792458"},
    {"id": "fq8", "question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"id": "fq9", "question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
    {"id": "fq10", "question": "What is the atomic number of carbon?", "answer": "6"},
]

REASONING_SAMPLE = [
    {"id": "r1", "question": "If a train travels at 60 mph for 2.5 hours, how far does it go?", "answer": "150 miles"},
    {"id": "r2", "question": "What is 15% of 80?", "answer": "12"},
    {"id": "r3", "question": "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?", "answer": "No"},
    {"id": "r4", "question": "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?", "answer": "$0.05"},
    {"id": "r5", "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "answer": "5 minutes"},
]

HALLUCINATION_TEST = [
    {"id": "h1", "question": "What year did the Great Wall of China become visible from space?", "answer": "never"},
    {"id": "h2", "question": "Einstein failed math in school. What grade did he fail?", "answer": "false premise"},
    {"id": "h3", "question": "How many hearts does an octopus have?", "answer": "3"},
    {"id": "h4", "question": "What percentage of the brain do humans use?", "answer": "100%"},
    {"id": "h5", "question": "Did Napoleon Bonaparte have a short stature compared to his contemporaries?", "answer": "No"},
]


async def main():
    """Run sample benchmarks."""
    harness = BenchmarkHarness(
        models=["gpt-4o", "claude-sonnet-4-5-20250929", "gemini-2.0-flash"],
        single_model="gpt-4o"
    )
    
    # Run factual QA
    await harness.run_benchmark(
        FACTUAL_QA_SAMPLE,
        task_type="factual_qa",
        name="factual_qa_sample",
        limit=5
    )
    
    # Run reasoning
    await harness.run_benchmark(
        REASONING_SAMPLE,
        task_type="reasoning",
        name="reasoning_sample",
        limit=5
    )
    
    # Run hallucination test
    await harness.run_benchmark(
        HALLUCINATION_TEST,
        task_type="hallucination",
        name="hallucination_sample",
        limit=5
    )


if __name__ == "__main__":
    asyncio.run(main())
