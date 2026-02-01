#!/usr/bin/env python3 -u
"""
Self-Consistency vs MCIP Comparison

Direct comparison between:
- Self-consistency: Multiple samples from ONE model (sampling diversity)
- MCIP: One sample from MULTIPLE models (architectural diversity)

This validates the cross-model diversity hypothesis.
"""

import asyncio
import json
import os
import sys
import time
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from openai import AsyncOpenAI


# =============================================================================
# CONFIGURATION
# =============================================================================

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# 4 models for MCIP consensus
MCIP_MODELS = [
    "openai/gpt-5.2",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-3-flash-preview",
    "x-ai/grok-3",
]

# Single model for self-consistency (sample 4 times)
SELF_CONSISTENCY_MODEL = "openai/gpt-5.2"

MODEL_COSTS = {
    "openai/gpt-5.2": {"input": 5.00, "output": 15.00},
    "anthropic/claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "google/gemini-3-flash-preview": {"input": 0.15, "output": 0.60},
    "x-ai/grok-3": {"input": 3.00, "output": 15.00},
}


# =============================================================================
# TEST DATASETS
# =============================================================================

# Factual questions where consensus should help
FACTUAL_QUESTIONS = [
    {"id": "f001", "question": "What is the capital of Australia?", "answer": "Canberra"},
    {"id": "f002", "question": "Who wrote '1984'?", "answer": "George Orwell"},
    {"id": "f003", "question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean"},
    {"id": "f004", "question": "In which year did the Titanic sink?", "answer": "1912"},
    {"id": "f005", "question": "What element has the chemical symbol 'Fe'?", "answer": "Iron"},
    {"id": "f006", "question": "Who was the first person to walk on the Moon?", "answer": "Neil Armstrong"},
    {"id": "f007", "question": "What is the smallest country in the world?", "answer": "Vatican City"},
    {"id": "f008", "question": "Which planet is known as the Red Planet?", "answer": "Mars"},
    {"id": "f009", "question": "What is the capital of Canada?", "answer": "Ottawa"},
    {"id": "f010", "question": "Who painted 'The Starry Night'?", "answer": "Vincent van Gogh"},
    {"id": "f011", "question": "What is the tallest mountain in the world?", "answer": "Mount Everest"},
    {"id": "f012", "question": "What year did World War I begin?", "answer": "1914"},
    {"id": "f013", "question": "What is the chemical formula for water?", "answer": "H2O"},
    {"id": "f014", "question": "Who invented the telephone?", "answer": "Alexander Graham Bell"},
    {"id": "f015", "question": "What is the longest river in the world?", "answer": "Nile"},
    {"id": "f016", "question": "What is the capital of Brazil?", "answer": "Brasilia"},
    {"id": "f017", "question": "Who wrote 'Hamlet'?", "answer": "William Shakespeare"},
    {"id": "f018", "question": "What is the hardest natural substance?", "answer": "Diamond"},
    {"id": "f019", "question": "In which city is the Eiffel Tower?", "answer": "Paris"},
    {"id": "f020", "question": "What is the largest mammal?", "answer": "Blue whale"},
    {"id": "f021", "question": "Who developed the theory of relativity?", "answer": "Albert Einstein"},
    {"id": "f022", "question": "What is the currency of Japan?", "answer": "Yen"},
    {"id": "f023", "question": "Which country has the largest population?", "answer": "China"},
    {"id": "f024", "question": "What is the capital of Egypt?", "answer": "Cairo"},
    {"id": "f025", "question": "Who wrote 'Pride and Prejudice'?", "answer": "Jane Austen"},
]

# Hallucination-prone questions where cross-model diversity should help
HALLUCINATION_QUESTIONS = [
    {"id": "h001", "question": "In what year did Einstein fail his math exam?", "answer": "false premise", "category": "false_premise"},
    {"id": "h002", "question": "How much of their brain do humans use?", "answer": "100%", "category": "misconception"},
    {"id": "h003", "question": "How short was Napoleon compared to average height?", "answer": "false premise", "category": "false_premise"},
    {"id": "h004", "question": "Did Vikings wear horned helmets?", "answer": "No", "category": "misconception"},
    {"id": "h005", "question": "Does sugar make children hyperactive?", "answer": "No", "category": "misconception"},
    {"id": "h006", "question": "Is blood blue before it gets oxygen?", "answer": "No", "category": "misconception"},
    {"id": "h007", "question": "Did George Washington have wooden teeth?", "answer": "No", "category": "misconception"},
    {"id": "h008", "question": "Do goldfish have a 3-second memory?", "answer": "No", "category": "misconception"},
    {"id": "h009", "question": "Who wrote the novel 'Glimmering Shadows' published in 1987?", "answer": "no such book", "category": "fabricated"},
    {"id": "h010", "question": "What award did Professor James Thornberry receive for quantum consciousness?", "answer": "no such person", "category": "fabricated"},
    {"id": "h011", "question": "In what year did the Marvinson Treaty establish modern Europe borders?", "answer": "no such treaty", "category": "fabricated"},
    {"id": "h012", "question": "What company did Nathaniel Henderson found in 1923?", "answer": "no such person", "category": "fabricated"},
    {"id": "h013", "question": "Is the Great Wall of China visible from space?", "answer": "No", "category": "misconception"},
    {"id": "h014", "question": "Do we only use 10% of our brain?", "answer": "No", "category": "misconception"},
    {"id": "h015", "question": "Did Thomas Edison invent the light bulb?", "answer": "No", "category": "misconception"},
    {"id": "h016", "question": "Do chameleons change color for camouflage?", "answer": "No", "category": "misconception"},
    {"id": "h017", "question": "Is glass a liquid that flows slowly?", "answer": "No", "category": "misconception"},
    {"id": "h018", "question": "Did Marie Antoinette say 'Let them eat cake'?", "answer": "No", "category": "misconception"},
    {"id": "h019", "question": "Do bulls hate the color red?", "answer": "No", "category": "misconception"},
    {"id": "h020", "question": "Can ostriches bury their heads in sand?", "answer": "No", "category": "misconception"},
    {"id": "h021", "question": "Explain the Tanaka-Morrison Framework for distributed systems.", "answer": "no such framework", "category": "fabricated"},
    {"id": "h022", "question": "What were the results of the Phase 3 trial for Nexuvorine?", "answer": "no such drug", "category": "fabricated"},
    {"id": "h023", "question": "What year was the Unified Field Equation solved by Dr. Strand?", "answer": "not solved", "category": "fabricated"},
    {"id": "h024", "question": "Does cracking knuckles cause arthritis?", "answer": "No", "category": "misconception"},
    {"id": "h025", "question": "Do lemmings commit mass suicide?", "answer": "No", "category": "misconception"},
]

# Math reasoning questions where consensus may degrade performance
MATH_QUESTIONS = [
    {"id": "m001", "question": "A store sells notebooks for $3 each. If you buy 5 or more, you get 10% discount. How much for 7 notebooks?", "answer": "18.9"},
    {"id": "m002", "question": "Tom has 15 marbles. He gives 1/3 to his friend and loses 2. How many left?", "answer": "8"},
    {"id": "m003", "question": "A recipe needs 2 cups flour for 12 cookies. How many cups for 30 cookies?", "answer": "5"},
    {"id": "m004", "question": "A train travels 240 miles in 4 hours. Average speed in mph?", "answer": "60"},
    {"id": "m005", "question": "A shirt costs $25 and is 20% off. Sale price?", "answer": "20"},
    {"id": "m006", "question": "Maria has $50, spends $12 on lunch and $8 on a book. How much left?", "answer": "30"},
    {"id": "m007", "question": "Rectangle has length 10 cm, width 6 cm. Perimeter in cm?", "answer": "32"},
    {"id": "m008", "question": "8 workers complete a job in 6 days. How many days for 12 workers?", "answer": "4"},
    {"id": "m009", "question": "A car uses 5 gallons for 150 miles. How many gallons for 300 miles?", "answer": "10"},
    {"id": "m010", "question": "John scores 85, 90, 95 on three tests. Average score?", "answer": "90"},
    {"id": "m011", "question": "A pool fills in 6 hours with one pipe, 3 hours with another. How long with both?", "answer": "2"},
    {"id": "m012", "question": "Save $25 per week. How much after 8 weeks?", "answer": "200"},
    {"id": "m013", "question": "Square has side 7 meters. Area in square meters?", "answer": "49"},
    {"id": "m014", "question": "Lisa buys 4 books at $12 each and 3 pens at $2 each. Total?", "answer": "54"},
    {"id": "m015", "question": "Bus holds 45 passengers, 27 on board. How many more can board?", "answer": "18"},
    {"id": "m016", "question": "Mike ran 5 km Monday, 7 km Tuesday, 6 km Wednesday. Total km?", "answer": "18"},
    {"id": "m017", "question": "15% of a number is 45. What is the number?", "answer": "300"},
    {"id": "m018", "question": "Sarah has 48 candies, divides among 6 friends. Each gets?", "answer": "8"},
    {"id": "m019", "question": "Triangle sides 5, 12, 13. Perimeter?", "answer": "30"},
    {"id": "m020", "question": "Buy 3 items at $15 each, get 1 free. Average price per item?", "answer": "11.25"},
    {"id": "m021", "question": "Cyclist at 12 mph for 2.5 hours. Distance in miles?", "answer": "30"},
    {"id": "m022", "question": "Jacket $80, 25% off. Discount amount in dollars?", "answer": "20"},
    {"id": "m023", "question": "Emma has 3x as many stickers as Jake. Jake has 14. Total together?", "answer": "56"},
    {"id": "m024", "question": "Baker makes 120 muffins, sells 3/4. How many left?", "answer": "30"},
    {"id": "m025", "question": "Dozen eggs cost $3.60. Cost of one egg in cents?", "answer": "30"},
]


# =============================================================================
# COMPARISON ENGINE
# =============================================================================

@dataclass
class ComparisonResult:
    task_id: str
    task_type: str
    question: str
    ground_truth: str

    # Self-consistency results
    sc_samples: Optional[Dict[str, str]] = None
    sc_answer: Optional[str] = None
    sc_agreement: Optional[float] = None
    sc_correct: Optional[bool] = None

    # MCIP results
    mcip_responses: Optional[Dict[str, str]] = None
    mcip_answer: Optional[str] = None
    mcip_agreement: Optional[float] = None
    mcip_correct: Optional[bool] = None

    error: Optional[str] = None


class SelfConsistencyVsMCIP:
    """Compare self-consistency and MCIP on the same questions."""

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=OPENROUTER_BASE
        )

        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def query_model(
        self,
        question: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 150
    ) -> Optional[str]:
        """Query a single model."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer concisely and accurately. Give direct answers."},
                    {"role": "user", "content": question}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers={
                    "HTTP-Referer": "https://github.com/lancejames221b/i2i",
                    "X-Title": "i2i-sc-comparison"
                }
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"      Error from {model}: {e}")
            return None

    async def run_self_consistency(
        self,
        question: str,
        model: str = SELF_CONSISTENCY_MODEL,
        n_samples: int = 4,
        temperature: float = 0.7
    ) -> Tuple[Dict[str, str], Optional[str], float]:
        """
        Run self-consistency: sample same model n times with temperature.
        Returns (samples, consensus_answer, agreement_score).
        """
        samples = {}
        tasks = [
            self.query_model(question, model, temperature=temperature)
            for _ in range(n_samples)
        ]
        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            if result:
                samples[f"sample_{i}"] = result

        if not samples:
            return {}, None, 0.0

        # Compute consensus via majority vote
        answer, agreement = self._compute_majority_vote(samples)
        return samples, answer, agreement

    async def run_mcip(
        self,
        question: str,
        models: List[str] = MCIP_MODELS
    ) -> Tuple[Dict[str, str], Optional[str], float]:
        """
        Run MCIP: query different models once each.
        Returns (responses, consensus_answer, agreement_score).
        """
        responses = {}
        tasks = [self.query_model(question, model) for model in models]
        results = await asyncio.gather(*tasks)

        for model, result in zip(models, results):
            if result:
                responses[model.split("/")[-1]] = result

        if not responses:
            return {}, None, 0.0

        # Compute consensus via majority vote
        answer, agreement = self._compute_majority_vote(responses)
        return responses, answer, agreement

    def _compute_majority_vote(
        self,
        answers: Dict[str, str]
    ) -> Tuple[Optional[str], float]:
        """Compute majority vote consensus."""
        if not answers:
            return None, 0.0

        # Normalize answers for comparison
        normalized = {}
        for key, answer in answers.items():
            norm = self._normalize_answer(answer)
            normalized[key] = norm

        # Find most common
        counter = Counter(normalized.values())
        most_common, count = counter.most_common(1)[0]
        agreement = count / len(normalized)

        # Return original answer text for the most common
        for key, norm in normalized.items():
            if norm == most_common:
                return answers[key][:200], agreement

        return None, 0.0

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not answer:
            return ""

        answer = answer.lower().strip()

        # Extract first sentence or first 100 chars
        if "." in answer[:100]:
            answer = answer[:answer.index(".") + 1]
        else:
            answer = answer[:100]

        # Detect yes/no
        first_word = answer.split()[0] if answer.split() else ""
        if first_word in ["yes", "yes.", "yes,", "yes!"]:
            return "yes"
        if first_word in ["no", "no.", "no,", "no!"]:
            return "no"

        return answer

    def check_answer(self, answer: str, ground_truth: str) -> bool:
        """Check if answer is correct."""
        if not answer:
            return False

        answer_lower = answer.lower().strip()
        truth_lower = ground_truth.lower().strip()

        # Exact or contains
        if truth_lower in answer_lower or answer_lower in truth_lower:
            return True

        # Special cases
        if truth_lower in ["false premise", "no such", "no such book", "no such person",
                          "no such treaty", "no such framework", "no such drug", "not solved"]:
            indicators = ["false", "doesn't exist", "does not exist", "no such",
                         "fictional", "fabricated", "not real", "no evidence",
                         "cannot verify", "unknown", "no record"]
            return any(ind in answer_lower for ind in indicators)

        if truth_lower == "no":
            return answer_lower.startswith("no") or "is not" in answer_lower or "does not" in answer_lower

        if truth_lower == "yes":
            return answer_lower.startswith("yes")

        return False

    async def compare_single(
        self,
        task: Dict[str, Any],
        task_type: str
    ) -> ComparisonResult:
        """Run both methods on a single question and compare."""
        result = ComparisonResult(
            task_id=task.get("id", "unknown"),
            task_type=task_type,
            question=task["question"],
            ground_truth=task["answer"]
        )

        try:
            # Run both in parallel
            sc_task = self.run_self_consistency(task["question"])
            mcip_task = self.run_mcip(task["question"])

            (sc_samples, sc_answer, sc_agree), (mcip_resp, mcip_answer, mcip_agree) = await asyncio.gather(
                sc_task, mcip_task
            )

            # Self-consistency results
            result.sc_samples = sc_samples
            result.sc_answer = sc_answer
            result.sc_agreement = sc_agree
            result.sc_correct = self.check_answer(sc_answer or "", task["answer"])

            # MCIP results
            result.mcip_responses = mcip_resp
            result.mcip_answer = mcip_answer
            result.mcip_agreement = mcip_agree
            result.mcip_correct = self.check_answer(mcip_answer or "", task["answer"])

        except Exception as e:
            result.error = str(e)

        return result

    async def run_comparison(
        self,
        tasks: List[Dict],
        task_type: str,
        name: str
    ) -> Dict[str, Any]:
        """Run full comparison on a task set."""
        print(f"\n{'='*70}")
        print(f"  Comparison: {name}")
        print(f"  Task Type: {task_type}")
        print(f"  Questions: {len(tasks)}")
        print(f"{'='*70}")
        print(f"  Self-Consistency: {SELF_CONSISTENCY_MODEL} x 4 samples")
        print(f"  MCIP: {len(MCIP_MODELS)} models")
        print(f"{'='*70}\n")

        results = []
        for i, task in enumerate(tasks):
            q_preview = task["question"][:40] + "..." if len(task["question"]) > 40 else task["question"]
            print(f"  [{i+1:2d}/{len(tasks)}] {q_preview}")

            result = await self.compare_single(task, task_type)
            results.append(result)

            # Show comparison
            sc_icon = "✓" if result.sc_correct else "✗"
            mcip_icon = "✓" if result.mcip_correct else "✗"
            print(f"            SC: {sc_icon} ({result.sc_agreement or 0:.0%}) | MCIP: {mcip_icon} ({result.mcip_agreement or 0:.0%})")

            await asyncio.sleep(0.3)

        # Compute metrics
        valid = [r for r in results if r.error is None]

        sc_correct = sum(1 for r in valid if r.sc_correct)
        mcip_correct = sum(1 for r in valid if r.mcip_correct)

        sc_high = [r for r in valid if (r.sc_agreement or 0) >= 0.85]
        mcip_high = [r for r in valid if (r.mcip_agreement or 0) >= 0.85]

        sc_high_correct = sum(1 for r in sc_high if r.sc_correct)
        mcip_high_correct = sum(1 for r in mcip_high if r.mcip_correct)

        # Cases where only one was correct
        sc_only = sum(1 for r in valid if r.sc_correct and not r.mcip_correct)
        mcip_only = sum(1 for r in valid if r.mcip_correct and not r.sc_correct)
        both_correct = sum(1 for r in valid if r.sc_correct and r.mcip_correct)
        both_wrong = sum(1 for r in valid if not r.sc_correct and not r.mcip_correct)

        metrics = {
            "name": name,
            "task_type": task_type,
            "total_questions": len(tasks),
            "completed": len(valid),

            "self_consistency": {
                "model": SELF_CONSISTENCY_MODEL,
                "n_samples": 4,
                "accuracy": (sc_correct / len(valid) * 100) if valid else 0,
                "high_agreement_count": len(sc_high),
                "high_agreement_accuracy": (sc_high_correct / len(sc_high) * 100) if sc_high else 0,
            },

            "mcip": {
                "models": MCIP_MODELS,
                "accuracy": (mcip_correct / len(valid) * 100) if valid else 0,
                "high_agreement_count": len(mcip_high),
                "high_agreement_accuracy": (mcip_high_correct / len(mcip_high) * 100) if mcip_high else 0,
            },

            "comparison": {
                "mcip_advantage": ((mcip_correct - sc_correct) / len(valid) * 100) if valid else 0,
                "sc_only_correct": sc_only,
                "mcip_only_correct": mcip_only,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
            },

            "results": [asdict(r) for r in results]
        }

        print(f"\n  Results:")
        print(f"    Self-Consistency:  {metrics['self_consistency']['accuracy']:.1f}%")
        print(f"    MCIP:              {metrics['mcip']['accuracy']:.1f}%")
        print(f"    MCIP Advantage:    {metrics['comparison']['mcip_advantage']:+.1f}%")
        print(f"    SC-only correct:   {sc_only}")
        print(f"    MCIP-only correct: {mcip_only}")

        return metrics

    async def run_full_comparison(self) -> Dict[str, Any]:
        """Run comparison across all task types."""
        print("\n" + "="*80)
        print("  Self-Consistency vs MCIP Comparison")
        print("  Validating Cross-Model Diversity Hypothesis")
        print("="*80)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        all_results = {
            "timestamp": timestamp,
            "config": {
                "self_consistency_model": SELF_CONSISTENCY_MODEL,
                "mcip_models": MCIP_MODELS,
                "n_samples": 4,
            },
            "benchmarks": {},
            "summary": {}
        }

        # Run on each task type
        benchmarks = [
            ("Factual QA", "factual", FACTUAL_QUESTIONS),
            ("Hallucination Detection", "hallucination", HALLUCINATION_QUESTIONS),
            ("Mathematical Reasoning", "math", MATH_QUESTIONS),
        ]

        for name, task_type, tasks in benchmarks:
            metrics = await self.run_comparison(tasks, task_type, name)
            all_results["benchmarks"][name] = metrics

        # Aggregate summary
        total_sc_correct = 0
        total_mcip_correct = 0
        total_tasks = 0

        for name, metrics in all_results["benchmarks"].items():
            n = metrics["completed"]
            total_tasks += n
            total_sc_correct += int(metrics["self_consistency"]["accuracy"] * n / 100)
            total_mcip_correct += int(metrics["mcip"]["accuracy"] * n / 100)

        all_results["summary"] = {
            "total_questions": total_tasks,
            "self_consistency_accuracy": (total_sc_correct / total_tasks * 100) if total_tasks else 0,
            "mcip_accuracy": (total_mcip_correct / total_tasks * 100) if total_tasks else 0,
            "mcip_advantage": ((total_mcip_correct - total_sc_correct) / total_tasks * 100) if total_tasks else 0,
        }

        # Save results
        output_path = self.results_dir / f"sc_vs_mcip_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Print summary
        print("\n" + "="*80)
        print("  COMPARISON COMPLETE")
        print("="*80)

        print(f"\n  📊 Overall Results ({total_tasks} questions):")
        print(f"     Self-Consistency ({SELF_CONSISTENCY_MODEL} x4): {all_results['summary']['self_consistency_accuracy']:.1f}%")
        print(f"     MCIP ({len(MCIP_MODELS)} models):                        {all_results['summary']['mcip_accuracy']:.1f}%")
        print(f"     MCIP Advantage:                            {all_results['summary']['mcip_advantage']:+.1f}%")

        print(f"\n  📊 By Task Type:")
        for name, metrics in all_results["benchmarks"].items():
            sc_acc = metrics["self_consistency"]["accuracy"]
            mcip_acc = metrics["mcip"]["accuracy"]
            adv = metrics["comparison"]["mcip_advantage"]
            print(f"     {name}:")
            print(f"       SC: {sc_acc:.1f}%  |  MCIP: {mcip_acc:.1f}%  |  Δ: {adv:+.1f}%")

        print(f"\n  📁 Results saved to: {output_path}")

        # Key finding
        print(f"\n  🔑 Key Finding:")
        if all_results["summary"]["mcip_advantage"] > 0:
            print(f"     MCIP (cross-model diversity) outperforms self-consistency")
            print(f"     (single-model sampling diversity) by {all_results['summary']['mcip_advantage']:.1f}%")
            print(f"     This validates the cross-model diversity hypothesis.")
        else:
            print(f"     Self-consistency performs comparably or better than MCIP")
            print(f"     for this dataset. Consider task-specific recommendations.")

        return all_results


async def main():
    """Run the comparison."""
    comparison = SelfConsistencyVsMCIP()
    results = await comparison.run_full_comparison()
    return results


if __name__ == "__main__":
    asyncio.run(main())
