# Competition Notes

> Internal-only competitive positioning and late-stage prioritization note.
> Do not cite competitor repos in public-facing docs.

## Summary

Our strongest comparative advantages are:

- a clear 3-task easy-to-hard ladder
- deterministic, dense partial-credit reward
- compact judge-friendly architecture
- a strong heuristic baseline

The strongest external competitor pattern is higher simulator depth or broader architecture ambition, especially in long-horizon environments. Our best response is reliability and clarity, not late complexity.

## What Matters Most

Judges are most likely to reward:

1. correctness and rerunnability
2. real-world domain quality
3. task and grader quality
4. reward usefulness for RL
5. clean packaging and deployment
6. baseline reproducibility

## Key Competitive Read

### Where we are strong

- helpdesk routing is a real enterprise workflow
- the task ladder is explicit and curriculum-friendly
- dense deterministic scoring is more RL-friendly than binary-only grading
- the repo is easier for judges to understand quickly than heavier simulator-style projects

### Where strong competitors can beat us

- simulator depth and richer state
- long-horizon control realism
- larger datasets or generated scenario breadth
- broader tooling such as MCP integrations

## Priority Responses

The highest-value late-stage moves are:

1. strengthen validation proof
2. keep scorer crispness explicit and tested
3. document grounded scoring clearly
4. prove Docker and validator readiness
5. avoid architecture churn

## Late-Stage Rules

- do not add MCP
- do not do a reward-architecture refactor
- do not expand the runtime dataset late
- do not make broad inference changes
- only add tiny RL-signal improvements if fully tested and benchmark-stable

## Practical Action List

### Must keep

- unit, smoke, and integration tests
- scorer crispness checks
- grounding audit evidence
- Docker smoke proof
- `openenv validate` readiness
- clean judge-facing docs

### Nice to have only if fully green

- richer history fields
- `queue_size` reset kwarg
- short TRL / GRPO README example

## Competitor Snapshot

The field includes:

- simple reference environments that we clearly outperform on realism
- strong but binary-reward environments where we win on RL signal quality
- ambitious simulator-style environments that win on technical scope but are harder to judge quickly

Our best positioning is not "most complex"; it is "most defensible, trainable, and rerunnable."
