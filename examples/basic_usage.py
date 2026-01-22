#!/usr/bin/env python3
"""
Basic AICP Usage Examples

This script demonstrates the core features of the AI-to-AI Communication Protocol.
"""

import asyncio
import sys
sys.path.insert(0, '..')

from i2i import AICP, EpistemicType, ConsensusLevel


async def main():
    # Initialize the protocol
    protocol = AICP()

    print("=" * 60)
    print("AICP - AI-to-AI Communication Protocol")
    print("=" * 60)

    # Check available providers
    print("\n1. Available Providers:")
    print("-" * 40)
    available = protocol.list_available_models()
    for provider, models in available.items():
        print(f"  {provider}: {', '.join(models[:3])}...")

    # Example 1: Consensus Query
    print("\n\n2. Consensus Query Example")
    print("-" * 40)
    print("Query: What causes inflation?")

    try:
        result = await protocol.consensus_query(
            "What causes inflation? Give a brief answer.",
            models=["gpt-4o", "claude-3-5-sonnet-20241022"]  # Use 2 models for demo
        )

        print(f"\nConsensus Level: {result.consensus_level.value}")
        print(f"Models Queried: {result.models_queried}")

        if result.consensus_answer:
            print(f"\nConsensus Answer:\n{result.consensus_answer[:500]}...")

        if result.divergences:
            print(f"\nDivergences Found: {len(result.divergences)}")

    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Claim Verification
    print("\n\n3. Claim Verification Example")
    print("-" * 40)
    claim = "The speed of light is approximately 300,000 kilometers per second"
    print(f"Claim: {claim}")

    try:
        result = await protocol.verify_claim(claim)

        status = "✓ VERIFIED" if result.verified else "✗ NOT VERIFIED"
        print(f"\nResult: {status}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Verifiers: {result.verifiers}")

        if result.issues_found:
            print(f"Issues: {result.issues_found}")

    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Epistemic Classification
    print("\n\n4. Epistemic Classification Example")
    print("-" * 40)

    questions = [
        "What is the capital of France?",
        "Will it rain in New York next Tuesday?",
        "Is consciousness substrate-independent?",
        "Did Shakespeare write all his plays himself?",
    ]

    for question in questions:
        # Quick classification (no API calls)
        quick = protocol.quick_classify(question)
        print(f"\n  Q: {question}")
        print(f"  Quick Classification: {quick.value}")

    # Full classification of one question
    print("\n\nFull Classification:")
    try:
        result = await protocol.classify_question(
            "Is consciousness substrate-independent?"
        )

        print(f"  Classification: {result.classification.value}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Actionable: {result.is_actionable}")
        print(f"  Reasoning: {result.reasoning[:200]}...")

        if result.why_idle:
            print(f"  Why Idle: {result.why_idle[:200]}...")

    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Smart Query
    print("\n\n5. Smart Query Example")
    print("-" * 40)
    print("Query: What is the current population of Tokyo?")

    try:
        result = await protocol.smart_query(
            "What is the current population of Tokyo?",
            require_consensus=True,
            verify_result=True,
        )

        print(f"\nClassification: {result['classification'].classification.value}")
        print(f"Consensus Level: {result['consensus']['level']}")

        if result['warnings']:
            print(f"Warnings: {result['warnings']}")

        print(f"\nAnswer: {result['answer'][:300]}...")

        if result['verification']:
            ver = result['verification']
            print(f"Verification: {'✓' if ver['verified'] else '✗'} ({ver['confidence']:.1%})")

    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
