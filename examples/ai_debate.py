#!/usr/bin/env python3
"""
AI Debate Example

This script demonstrates having multiple AI models debate a topic,
similar to the Claude-ChatGPT conversation that inspired this project.
"""

import asyncio
import sys
sys.path.insert(0, '..')

from i2i import AICP


async def main():
    protocol = AICP()

    print("=" * 70)
    print("AICP - AI Debate Demo")
    print("=" * 70)

    # The topic that inspired this project
    topic = """
    What are the philosophical implications of two different AI systems
    having a conversation? Is there something meaningful in AI-to-AI dialogue,
    or is it fundamentally different from human conversation in ways that
    make it less significant?
    """

    print(f"\nDebate Topic:\n{topic.strip()}")
    print("\n" + "-" * 70)

    # Get available models
    available = protocol.list_configured_providers()
    print(f"\nConfigured providers: {available}")

    if len(available) < 2:
        print("\nNeed at least 2 providers for a debate.")
        print("Please configure API keys in .env file.")
        return

    # Run the debate
    print("\nStarting debate...\n")

    try:
        result = await protocol.debate(
            topic=topic.strip(),
            rounds=2,  # 2 rounds of responses
        )

        # Print the debate
        print(f"\nParticipants: {', '.join(result['participants'])}")

        for round_data in result["rounds"]:
            print(f"\n{'=' * 70}")
            print(f"ROUND {round_data['round']} ({round_data['type'].upper()})")
            print("=" * 70)

            for resp in round_data["responses"]:
                print(f"\n[{resp['model']}]:")
                print("-" * 40)
                print(resp["content"])
                print()

        # Print summary
        if result["summary"]:
            print("\n" + "=" * 70)
            print("DEBATE SUMMARY")
            print("=" * 70)
            print(result["summary"])

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
