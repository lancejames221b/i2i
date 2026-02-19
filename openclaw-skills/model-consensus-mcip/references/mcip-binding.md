# MCIP Binding (Actual Protocol)

This skill is bound to the local i2i implementation at:
- `/home/generic/dev/i2i`

## Protocol Truth

- Version and message semantics: `/home/generic/dev/i2i/RFC-MCIP.md`
- Runtime entrypoint: `i2i.protocol.Protocol.consensus_query`
- Result schema and levels: `/home/generic/dev/i2i/i2i/schema.py`

Consensus levels are fixed:
- `HIGH`
- `MEDIUM`
- `LOW`
- `NONE`
- `CONTRADICTORY`

## Required Behaviour

1. Prefer real protocol execution over synthetic prompt voting.
2. Preserve `ConsensusResult` fields from i2i output.
3. Do not remap consensus levels to custom enums.
4. If result is `LOW/NONE/CONTRADICTORY`, require follow-up validation steps.

## OpenClaw Alias Guidance

OpenClaw runtime aliases (`gemini-pro`, `sonnet`, `codex`, `opus`) are control-plane names.
When calling i2i, use i2i-supported provider model IDs from config/runtime.

Recommended practical mapping (verify against configured providers):
- `gemini-pro` -> configured Google model
- `sonnet` -> configured Anthropic Sonnet model
- `codex` -> configured OpenAI model for engineering-centric opinions
- `opus` -> optional tie-breaker model only if configured

Never assume all aliases exist in i2i config; verify first.
