#!/usr/bin/env python3
"""
Epistemic Filter Example

This demonstrates using AICP to filter questions before expensive processing.
The idea: detect "idle" or "underdetermined" questions early to save time/money.
"""

import asyncio
import sys
sys.path.insert(0, '..')

from i2i import AICP, EpistemicType


# Sample questions to filter
QUESTIONS = [
    # Clearly answerable
    "What is the chemical formula for water?",
    "Who wrote Romeo and Juliet?",
    "What is 2 + 2?",

    # Uncertain but answerable
    "What will the weather be like tomorrow in San Francisco?",
    "How many people will attend the next World Cup final?",

    # Underdetermined
    "Did Lee Harvey Oswald act alone?",
    "What caused the extinction of the dinosaurs?",
    "Is string theory correct?",

    # Idle (non-action-guiding)
    "Is consciousness substrate-independent?",
    "Do we have free will?",
    "What is the meaning of life?",
    "Is there an objective morality?",

    # Potentially malformed
    "What is the sound of one hand clapping?",
]


async def main():
    protocol = AICP()

    print("=" * 70)
    print("AICP - Epistemic Pre-Filter Demo")
    print("=" * 70)
    print("\nThis demonstrates filtering questions by epistemic status")
    print("before sending them to expensive AI queries.\n")

    # Quick classification (no API calls - just heuristics)
    print("QUICK CLASSIFICATION (Heuristic - No API Calls)")
    print("-" * 70)

    categories = {t: [] for t in EpistemicType}

    for question in QUESTIONS:
        result = protocol.quick_classify(question)
        categories[result].append(question)
        print(f"  [{result.value:15}] {question[:50]}...")

    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    for etype, questions in categories.items():
        if questions:
            print(f"\n{etype.value.upper()} ({len(questions)} questions):")
            for q in questions:
                print(f"  • {q[:60]}...")

    # Full classification of interesting cases
    print("\n" + "=" * 70)
    print("FULL CLASSIFICATION (API Calls)")
    print("=" * 70)

    interesting_questions = [
        "Is consciousness substrate-independent?",
        "What caused the 2008 financial crisis?",
        "Will AI surpass human intelligence by 2030?",
    ]

    for question in interesting_questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)

        try:
            result = await protocol.classify_question(question)

            print(f"Classification: {result.classification.value}")
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Actionable: {result.is_actionable}")

            if result.classification == EpistemicType.IDLE:
                print(f"Why Idle: {result.why_idle}")

            if result.classification == EpistemicType.UNDERDETERMINED:
                if result.competing_hypotheses:
                    print(f"Competing Hypotheses: {result.competing_hypotheses}")

            if result.suggested_reformulation:
                print(f"Better Question: {result.suggested_reformulation}")

        except Exception as e:
            print(f"Error: {e}")

    # Demonstrate filtering workflow
    print("\n" + "=" * 70)
    print("PRACTICAL WORKFLOW: Filter Before Expensive Queries")
    print("=" * 70)

    user_question = "Is AI consciousness possible?"
    print(f"\nUser asks: {user_question}")

    # Step 1: Quick filter
    quick_result = protocol.quick_classify(user_question)
    print(f"Quick classification: {quick_result.value}")

    if quick_result == EpistemicType.IDLE:
        print("\n⚠️  This question is likely 'idle' - non-action-guiding.")
        print("    Recommend: Clarify what decision this answer would inform.")
        print("    Skipping expensive consensus query...")
    elif quick_result == EpistemicType.UNDERDETERMINED:
        print("\n⚠️  This question may be underdetermined.")
        print("    Recommend: Consider if multiple valid answers exist.")
        print("    Proceeding with consensus query to explore perspectives...")
    else:
        print("\n✓  Question appears answerable.")
        print("    Proceeding with consensus query...")

    print("\nThis filtering saves API costs on unanswerable questions!")


if __name__ == "__main__":
    asyncio.run(main())
