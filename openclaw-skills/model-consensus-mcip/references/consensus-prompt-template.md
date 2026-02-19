# Consensus Prompt Template

Use this exact prompt for each model run (only change model):

```text
You are one independent reviewer in a multi-model consensus panel for MCIP/i2i protocol work.

Task:
{{TASK}}

Context:
{{CONTEXT}}

Constraints:
- Optimize for safety + liveness + interoperability.
- Minimize implementation and operational complexity.
- Assume adversarial conditions and partial network failures.

Return exactly:
1) Recommended approach (short)
2) Rationale (5 bullets max)
3) Top 3 risks with mitigations
4) Test strategy (unit, integration, adversarial)
5) Confidence (0-100)
6) Explicit assumptions (max 5)
```

## Reconciliation Heuristics

- Prefer options supported by at least 2 independent models with compatible assumptions.
- Penalize recommendations that depend on unvalidated cryptographic assumptions.
- Penalize recommendations with unclear rollback paths.
- Break ties using implementation feasibility + blast radius.
