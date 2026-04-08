# Project Status

This is the canonical repo status file.

It should answer two questions quickly:

1. what the project can do right now
2. what actually changed during the recent benchmark-upgrade thread

## Current Snapshot

As of April 8, 2026:

- the active branch is `main`
- the last runtime-changing benchmark checkpoint before this cleanup pass was `1d9d3ee`
- the latest runtime-changing checkpoint passed `openenv validate`
- the latest full test checkpoint passed `175` tests
- the environment now behaves like a real queue-management benchmark, not a single-ticket classifier
- stale review branches and nonessential planning docs have been removed so the repo stays submission-clean

## What The Project Does Today

The current repo supports:

- full routing on all three tasks: `issue_type`, `priority`, `assignment_group`, and `resolution_action`
- partial observability that gets harder as the task difficulty rises
- five action types: `submit`, `investigate`, `request_info`, `defer`, and `open_incident`
- queue-level carry-over state such as capacity pressure, incident slots, SLA risk, and deferred tickets
- cluster-aware episodes where one ticket can make later related tickets easier or harder
- deterministic follow-up tickets when earlier handling was weak or incomplete
- a terminal score that blends routing quality with queue-management quality
- a local policy-learning loop that compares and searches over deterministic policies
- a modern landing page at `/web` instead of the original plain HTML table

## Validation State

The latest validated runtime state before this cleanup pass included:

- passing `openenv validate`
- passing full `python -m unittest discover -s tests -p "test_*.py" -v`
- a passing Hugging Face Space and Docker-ready packaging setup
- synchronized pushes to both `origin/main` and `space/main`

This cleanup pass is documentation and repo hygiene only. It does not change the environment contract.

## Full Commit Timeline From Git History

The entries below are taken directly from the local `main` history, which matches `origin/main`.

### March 31, 2026

- `10:47 IST` `3752981` `Initial commit`
- `11:20 IST` `eae2b1d` `March 30 - April 1st : sever/`
- `11:27 IST` `9e71ac4` `Merge pull request #2 from suyashkumar102/main`
- `13:29 IST` `61398c0` `April 2nd tasks`
- `20:28 IST` `7564d6c` `Fix dataset loader for UTF-8 BOM on Windows`

### April 1, 2026

- `18:28 IST` `4f3bed5` `fix openenv.yaml: use git URL for openenv-core dep, matches requirements.txt`
- `20:11 IST` `969eaef` `Merge pull request #3 from suyashkumar102/main`
- `20:50 IST` `3b8bf40` `Improve dataset realism and consolidate project status log`
- `20:59 IST` `1b9e464` `Update docs after first runtime validation pass`

### April 2, 2026

- `22:16 IST` `5b9f288` `fix: expand inference docstring and add git to Dockerfile`
- `22:18 IST` `5de9815` `add analysis folder`
- `22:39 IST` `9e384ef` `Merge pull request #4 from suyashkumar102/main`
- `23:37 IST` `6753cde` `Finish Roopal April 5-6 docs and repo audit`
- `23:40 IST` `c35bcc6` `Merge remote-tracking branch 'origin/main' into codex/apr5-apr6-roopal`

### April 3, 2026

- `00:50 IST` `c16104f` `Add GitHub Actions Docker smoke test`
- `00:55 IST` `54d32f8` `Merge pull request #5 from Roopalgn/codex/apr5-apr6-roopal`
- `01:19 IST` `7a88607` `Update final submission roadmap`
- `01:27 IST` `706f85f` `Merge branch 'codex/apr5-apr6-roopal'`
- `02:20 IST` `6f27f26` `Update final submission roadmap`
- `02:30 IST` `375aa81` `Update final submission roadmap`
- `11:47 IST` `ae36543` `Add grader and dataset unit tests with scoring contract`
- `12:59 IST` `72d2634` `Consolidate requirements docs and align roadmap with official submission rules`
- `18:19 IST` `6920aae` `Complete Roopal roadmap work for April 4-7`
- `20:36 IST` `795d5f1` `Update final submission roadmap`
- `21:44 IST` `82aca6e` `Make inference.py compliant with submission checklist`

### April 4, 2026

- `10:32 IST` `0fd10c5` `add smoke/integration tests, fix logging, openenvignore, status updates`
- `10:34 IST` `f57e6a7` `fix port 8000->7860 in app.py/openenv.yaml, add pyproject script entry, fix stubs`
- `10:35 IST` `fd636ad` `gitignore build/ and uv.lock`
- `10:41 IST` `ca7bdbd` `remove uv.lock from gitignore`
- `11:45 IST` `32f4c09` `fix inference stdout and README docker port`
- `11:50 IST` `3707fc3` `Merge pull request #6 from suyashkumar102/main`
- `12:12 IST` `5dd60ae` `uv.lock`
- `14:33 IST` `89ca22f` `Clean up internal docs and finalize validation state`

### April 5, 2026

- `20:53 IST` `42dd095` `feat: competitive upgrade for hackathon submission`
- `20:56 IST` `2a0f057` `docs: add deep competitive gap report and gap analysis`
- `22:22 IST` `6c5051f` `fix: resolve full test suite failures from PR review`

### April 6, 2026

- `12:42 IST` `c64d203` `Finalize gap fixes and lightweight competitive upgrades`
- `12:54 IST` `52ab5fa` `Merge branch 'main' into final-submit-gap-fixes`
- `13:34 IST` `186fd65` `Merge pull request #10 from suyashkumar102/final-submit-gap-fixes`
- `14:14 IST` `2216a4d` `Add root Dockerfile for Hugging Face Space`
- `17:09 IST` `8ccf96d` `Ignore action metadata in extra field validation`
- `21:15 IST` `67ce1eb` `Add policy learning loop and strengthen RL-style environment`

### April 7, 2026

- `11:37 IST` `8ada670` `Use evaluator API_KEY for LLM proxy and strengthen env`
- `12:15 IST` `2d5c8e6` `Pin python base image digest for stable Docker builds`
- `13:16 IST` `bfc789d` `Enable proxy LLM mode with API_KEY and real default model`
- `13:29 IST` `e3cd5c5` `Use AWS public ECR mirror for python base image`
- `13:57 IST` `ff634dc` `Run all tasks by default and keep task scores inside open interval`
- `14:09 IST` `e3dfee6` `Clamp grader task scores to open interval`
- `14:51 IST` `c0d489c` `Keep invalid-action task scores inside open interval`
- `15:07 IST` `a5859dc` `Normalize remaining score fields into open interval`
- `15:43 IST` `d6d9493` `Clamp reported task scores to open interval and match sample logs`
- `21:43 IST` `d378e5d` `Strengthen hard-task investigation and grading`

### April 8, 2026

- `03:59 IST` `8241eb5` `Add queue-planning helpdesk routing mechanics`
- `07:03 IST` `043d9e1` `Upgrade helpdesk env with queue dynamics and operational actions`
- `10:06 IST` `454cef3` `Add cluster-aware queue dynamics to helpdesk env`
- `11:45 IST` `1d9d3ee` `Strengthen queue benchmark and refresh landing page`

## Net Result Of The Thread

Compared with the starting point, the repo is now materially stronger in five ways:

- Phase 2 compliance issues were fixed without breaking the evaluator contract
- the benchmark became more agentic through queue mutation, operational actions, and downstream consequences
- the hard task stopped being a near-trivial keyword-routing problem
- the grader and final reward became more aligned with real queue-management quality
- the public presentation improved through cleaner docs and a better landing page

This cleanup and publishing pass also:

- expands `PROJECT_STATUS.md` to cover the full repo history instead of only the late-stage sprint
- rewrites `KNOWLEDGE.md` as a mentor-style guide for a beginner builder
- removes stale planning and internal analysis docs that no longer reflect the shipped benchmark
- leaves `required.md` as the retained requirements checklist

## Remaining Optional Gaps

The project is strong, but a few optional upgrades still exist if more time is ever available:

- replace more authored queue rules with even more emergent simulator dynamics
- grow the dataset further with less taxonomy-friendly wording
- move from policy search toward a more clearly trainable learning setup
- gather stronger benchmark comparisons against external LLM baselines

## Repo Hygiene Notes

This cleanup pass also keeps the repo focused by:

- retaining `required.md` as the requirement checklist
- keeping `README.md`, `KNOWLEDGE.md`, and `PROJECT_STATUS.md` as the main public guidance
- removing stale planning and gap-analysis files that no longer reflect the current state
