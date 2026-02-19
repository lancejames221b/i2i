---
name: model-consensus-mcip
description: Orchestrate multi-model consensus (Gemini 3 Pro/Flash, Sonnet 4.6, Codex 5.3, Opus 4.6) for i2i/MCIP design, protocol analysis, and implementation decisions in OpenClaw. Use when you need higher-confidence outputs by collecting independent model opinions, reconciling disagreements, and producing a single consensus recommendation with rationale and confidence.
---

# Model Consensus MCIP

Run consensus through your real i2i MCIP stack, not a generic orchestration.

## When to Use

- You need a high-confidence decision for MCIP/i2i architecture, protocol behaviour, or rollout choices.
- You want independent model perspectives reconciled into one execution-ready recommendation.
- You need protocol-native consensus output (`HIGH|MEDIUM|LOW|NONE|CONTRADICTORY`) instead of ad-hoc voting.
- You need a safe "no consensus" outcome when evidence is weak or contradictory.

## Canonical Protocol Sources (read first)

- `/home/generic/dev/i2i/RFC-MCIP.md` (protocol spec)
- `/home/generic/dev/i2i/i2i/protocol.py` (runtime API)
- `/home/generic/dev/i2i/i2i/schema.py` (message/result schema)
- `/home/generic/dev/i2i/config.json` (default model set)

Treat these as source of truth for levels, fields, and behaviour.

## Workflow

1. Normalize the question into MCIP terms
- Frame as a `consensus_query` task.
- Preserve protocol constraints: safety, liveness, interoperability, implementation cost, rollout risk.

2. Run consensus via real i2i protocol
- Prefer using `scripts/run_mcip_consensus.py`.
- Execute from repo root:
  - `cd /home/generic/dev/i2i`
  - `uv run python /home/generic/dev/skills/model-consensus-mcip/scripts/run_mcip_consensus.py --query "..."`
- This calls `Protocol.consensus_query(...)` from the i2i library.

3. Model panel selection
- Default to i2i config model set when available.
- If user asks for OpenClaw aliases (`gemini-pro`, `sonnet`, `codex`, `opus`), map to i2i-compatible IDs:
  - `gemini-pro` -> `gemini-3-flash-preview` (or configured Gemini model)
  - `sonnet` -> `claude-sonnet-4-5-20250929` (or configured Sonnet model)
  - `codex` -> `gpt-5.2` (proxy for OpenAI coding-capable panel member)
  - `opus` -> use only as tie-breaker in synthesis stage if configured in i2i provider/model list
- Never invent model IDs; verify against configured providers.

4. Reconcile with protocol-native outputs
- Use MCIP consensus levels exactly: `HIGH | MEDIUM | LOW | NONE | CONTRADICTORY`.
- Treat `LOW/NONE/CONTRADICTORY` as no reliable consensus.
- Include task-aware metadata when present (`consensus_appropriate`, `task_category`, `confidence_calibration`).

5. Produce execution-ready recommendation
- Include protocol impact, implementation steps, validation, and rollback trigger.
- If confidence is weak, return required follow-up experiments instead of forcing a decision.

## Output Contract

Use this structure:

```markdown
## Consensus Decision
- Recommendation:
- Confidence:
- Why this wins:

## MCIP / i2i Impact
- Consensus safety:
- Cross-chain interoperability impact:
- Failure domain changes:

## Implementation Plan
1.
2.
3.

## Validation
- Unit tests:
- Simulation/fuzz:
- Cross-chain integration tests:
- Adversarial tests:

## Open Risks
- Risk:
  - Mitigation:
  - Owner:
```

## Model Routing Rules

- Default panel: `gemini-pro + sonnet high + codex high`
- Add `opus high` only when:
  - split decision remains after reconciliation
  - safety-critical disagreement exists
  - economic/security assumptions conflict
- For lightweight checks, use `gemini-flash + sonnet medium` only.

## Implementation Notes for OpenClaw

- Primary path: run real i2i MCIP (`Protocol.consensus_query`) instead of manual orchestration.
- Secondary path: use independent model panel prompts only when i2i runtime is unavailable.
- Do not leak one model’s answer into another model’s prompt before first-pass outputs are collected.
- If consensus level is `LOW/NONE/CONTRADICTORY`, return “no consensus” with required follow-up experiments.

## References

- Use `references/consensus-prompt-template.md` for independent panel fallback template.
- Use `references/mcip-binding.md` for concrete protocol binding rules and field mapping.
- Use `scripts/run_mcip_consensus.py` to execute consensus on the real i2i implementation.
- Use `scripts/score_consensus.py` to aggregate model panel JSON when running fallback orchestration.
