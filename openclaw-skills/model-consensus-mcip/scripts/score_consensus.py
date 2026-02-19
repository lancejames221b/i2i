#!/usr/bin/env python3
"""Score model consensus outputs.

Input JSON format:
[
  {
    "model": "gemini-pro",
    "recommendation": "Use optimistic finality bridge with fraud proofs",
    "confidence": 78,
    "risks": ["..."],
    "assumptions": ["..."]
  }
]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from statistics import mean


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: score_consensus.py <responses.json>")
        return 1

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not rows:
        print(json.dumps({"error": "No rows provided"}, indent=2))
        return 1

    recs = [normalize(r.get("recommendation", "")) for r in rows if r.get("recommendation")]
    if not recs:
        print(json.dumps({"error": "No recommendations provided"}, indent=2))
        return 1

    counter = Counter(recs)
    winner, support = counter.most_common(1)[0]
    n = len(rows)
    support_ratio = support / n

    confidences = [int(r.get("confidence", 0)) for r in rows]
    avg_conf = round(mean(confidences), 1)

    if support_ratio >= 0.75 and avg_conf >= 80:
        band = "High"
    elif support_ratio >= 0.5 and avg_conf >= 60:
        band = "Medium"
    else:
        band = "Low"

    result = {
        "winner_recommendation": winner,
        "support": support,
        "panel_size": n,
        "support_ratio": round(support_ratio, 2),
        "average_confidence": avg_conf,
        "confidence_band": band,
        "distribution": dict(counter),
    }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
