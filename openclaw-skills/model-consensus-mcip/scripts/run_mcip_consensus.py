#!/usr/bin/env python3
"""Run consensus using the real local i2i MCIP implementation.

Usage:
  uv run python scripts/run_mcip_consensus.py --query "What causes inflation?"
  uv run python scripts/run_mcip_consensus.py --query "..." --models gpt-5.2,claude-sonnet-4-5-20250929,gemini-3-flash-preview
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

I2I_ROOT = Path("/home/generic/dev/i2i")
if str(I2I_ROOT) not in sys.path:
    sys.path.insert(0, str(I2I_ROOT))


async def run(query: str, models: list[str] | None) -> int:
    try:
        from i2i.protocol import AICP
    except Exception as exc:
        print(json.dumps({"error": f"Failed to import i2i Protocol: {exc}"}, indent=2))
        return 1

    protocol = AICP()

    try:
        result = await protocol.consensus_query(query=query, models=models)
    except Exception as exc:
        print(json.dumps({"error": f"consensus_query failed: {exc}"}, indent=2))
        return 1

    payload = {
        "query": query,
        "models": models,
        "consensus_level": getattr(result, "consensus_level", None).value
        if getattr(result, "consensus_level", None)
        else None,
        "consensus_answer": getattr(result, "consensus_answer", None),
        "consensus_appropriate": getattr(result, "consensus_appropriate", None),
        "task_category": getattr(result, "task_category", None),
        "confidence_calibration": getattr(result, "confidence_calibration", None),
        "metadata": getattr(result, "metadata", None),
    }

    # Best-effort include raw response summaries
    responses = []
    for r in getattr(result, "responses", []) or []:
        responses.append(
            {
                "model": getattr(r, "model", None),
                "provider": getattr(r, "provider", None),
                "confidence": getattr(r, "confidence", None),
                "content": (getattr(r, "content", "") or "")[:500],
                "error": getattr(r, "error", None),
            }
        )
    payload["responses"] = responses

    print(json.dumps(payload, indent=2, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real i2i MCIP consensus query")
    parser.add_argument("--query", required=True, help="Consensus query text")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model IDs supported by i2i (optional)",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    return asyncio.run(run(args.query, models))


if __name__ == "__main__":
    raise SystemExit(main())
