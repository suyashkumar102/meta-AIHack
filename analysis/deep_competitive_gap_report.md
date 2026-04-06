# Deep Codebase Comparison: OpenEnv Reference Environments vs This Helpdesk Project

## Scope and Method

This report was written from a direct code read, not from README-driven interpretation. I treated the `OpenEnv/envs` directory as the reference baseline you pointed to, and I compared it against the implementation that lives in this repository root plus `server/`.

I focused on code that actually defines runtime behavior:

- `models.py`
- `inference.py`
- `client.py`
- `vocabulary.py`
- `server/environment.py`
- `server/tasks.py`
- `server/grader.py`
- `server/reward.py`
- `server/app.py`
- `tests/*.py`
- `data/dataset.json`


That reading set is enough to answer the question that matters: what design moves make the strongest reference environments hard to beat, where your project is currently thinner than it looks, and what concrete changes would make your environment competitive instead of merely correct.

## Executive Verdict

Your project is a clean, readable, deterministic mini-benchmark. It is not yet a high-ceiling agent benchmark.

That sounds harsh, but it is also the clearest way to unlock the right next move. Right now your environment behaves much more like a structured multi-label classification task wrapped in OpenEnv than like the richer reference environments that expose hidden state, tool use, long-horizon consequences, multi-step reasoning, or grounded interaction with external systems. The code is good enough as a starter environment. It is not yet strong enough to beat the best reference projects on depth, realism, or benchmark credibility.

The good news is that the codebase is small, coherent, and fixable. The bad news is that the gap is not a one-line polish gap. It is a benchmark design gap.

The strongest OpenEnv reference environments win for one or more of these reasons:

- they expose a real action surface, not just label prediction
- they make the agent inspect state rather than infer everything from one text blob
- they reward process, not only end labels
- they support long-horizon or multi-step behavior
- they are harder to brute-force with dataset-specific heuristics
- they are backed by real engines, shells, browsers, tools, or stateful simulators
- they treat evaluation as a first-class system, not as a tiny helper function

Your project currently loses on most of those axes.

At the same time, your project has an underrated advantage: the domain is practical, legible, and product-shaped. IT helpdesk routing is a great benchmark domain if you push it harder. It naturally supports ambiguity, policy lookup, account context, queue optimization, escalation rules, duplicates, follow-up chains, customer sentiment, service health, SLA clocks, and partial observability. In other words, the domain is better than the current implementation. The environment has room to grow into something much stronger without abandoning the idea.

So the answer is not “throw this away and copy BrowserGym.” The answer is “turn this from a label benchmark into a realistic triage operations environment.”

## What the Reference Environments Actually Do Better

### 1. They expose richer action spaces

The single biggest difference between your code and the strongest reference projects is that the agent in your environment does very little. In your environment, the step is basically “predict some labels for this ticket.” In the stronger reference environments, the agent interacts.

`BrowserGymEnvironment` accepts an `action_str` and pushes it into a live browser benchmark. That means the benchmark difficulty comes from action selection in stateful UI space, not just from text classification. `OpenAppEnvironment` similarly supports `click`, `fill`, `select_option`, `goto`, `scroll`, and `send_keys`, and even mixes BrowserGym-style element IDs with raw Playwright CSS selectors for pragmatic reliability. `GitTaskEnvironment` supports clone, list, and git command execution against a Gitea-backed workspace. `Tbench2Environment` supports `exec`, `write`, `view`, `wait`, `kill`, `write_file`, and `evaluate`, which is much closer to real agent work. `FinQAEnvironment` turns the task into tool use over tables, SQL, and answer submission. `REPLEnvironment` exposes code execution with optional recursive LLM calls. `TextArenaEnvironment` takes natural-language moves and advances a game engine.

Your environment exposes none of that. The agent does not gather missing evidence. It does not inspect a related ticket. It does not search a KB. It does not look up account tier. It does not check service health. It does not add an internal note. It does not choose between acknowledging first and escalating later. It does not defer. It does not ask for more information. It does not resolve duplicates. It does not manage a queue. It only emits one shot structured output.

That makes the benchmark much easier to game, much easier to overfit, and much less diagnostic of real agent competence.

### 2. They separate visible observation from hidden truth

The strongest reference environments keep some truth state behind the curtain. The agent sees an observation. The environment owns more. That separation is what makes an environment feel like an environment instead of a dataframe with reward labels.

In `ChessEnvironment`, the agent observes legal moves, FEN, checks, and result state, but the environment owns board progression, opponent strategy, and trajectory reward accumulation. In `MazeEnvironment`, the environment tracks maze status and legal movement dynamics. In `TextArenaEnvironment`, the wrapped engine owns turn state, raw logs, rewards, role mapping, and step info. In `FinQAEnvironment`, the agent sees the question and tools, but the hidden ground truth answer, question identity, and full structured table data live behind the environment. In `Tbench2Environment`, the hidden truth is in the task files and tests. In `BrowserGymEnvironment`, the browser session and benchmark internals are hidden behind the observation.

Your environment has much less hidden truth than it should. The ticket label is hidden, yes, but the benchmark structure is shallow. More importantly, the code already hints at richer hidden structure and then fails to expose or exploit it. `HelpdeskTicketRecord` includes `ambiguity_note` and `related_ticket_id`, but `_build_observation()` throws both away and only exposes `ticket_id`, `title`, `requester`, and `description`. So even though the dataset contains follow-up relationships and ambiguity annotations, the environment does not actually let the agent work with them as structured state. That is a missed opportunity and a design leak at the same time.

The dataset is telling you the domain wants threads, ambiguity, and context. The environment currently flattens it back into plain text.

### 3. They reward more than a final label match

The reference environments do not all have brilliant reward design, but the best ones take reward seriously.

`REPLEnvironment` combines an outcome rubric with optional process reward. It can reward successful execution, penalize failures, and separately judge the final answer. `ChessEnvironment` uses a trajectory rubric with exponential discounting to assign credit across a game. `FinQAEnvironment` does robust answer normalization, including boxed answers, percentages, fractions, and multi-value comparisons. `TextArenaEnvironment` overlays auxiliary reward signals such as Wordle greens, yellows, repetitions, and correctness. `Tbench2Environment` evaluates by actually running tests, which is a grounded form of outcome reward.

Your reward design is better than “exact match only,” but it is still thin. `grade_action()` uses one handcrafted issue similarity table, one handcrafted priority proximity table, and exact match for assignment group and resolution action. `compute_step_reward()` is just clamping. `compute_trajectory_reward()` averages scores and subtracts an overshoot penalty.

That sounds reasonable until you inspect the runtime path. In practice, the overshoot penalty is effectively dead logic. `step()` increments the ticket index once per ticket and sets done when the index reaches queue length. A later `step()` call raises an error. That means `steps_taken` cannot exceed `queue_size` during normal episode execution, so the overshoot branch in `compute_trajectory_reward()` has no meaningful role in the current environment. The code suggests the benchmark penalizes wasteful action loops, but the environment does not actually allow them.

The deeper issue is that the reward judges only final fields, not triage quality as a process. There is no penalty for unnecessary escalation unless the final field is wrong. There is no reward for correctly identifying a duplicate and linking it. There is no cost model for routing everything to security “just in case.” There is no SLA-aware penalty for under-prioritizing a time-sensitive issue that still happens to hit some partial-credit similarity. There is no queue-level reward. There is no explanation consistency. There is no tool efficiency score because there are no tools. There is no notion of customer harm, resolver cost, escalation burden, or backlog impact.

The strongest environments earn their credibility by making reward a modeling decision. Your reward is still a convenience function.

### 4. They support multi-step or long-horizon behavior

Even the simpler reference environments tend to have longer horizon than your three-task ladder suggests.

`ChessEnvironment` is naturally long horizon. `BrowserGymEnvironment` and `OpenAppEnvironment` are stepwise interactions. `TextArenaEnvironment` proceeds over turns. `Tbench2Environment` supports iterative shell work and explicit evaluation. `REPLEnvironment` supports repeated code execution over an evolving namespace. `FinQAEnvironment` allows repeated tool calls up to `max_steps` before submission. Even `ReasoningGymEnvironment`, which is single-step, supports parameterized dataset generation and configurable tasks.

Your environment has multiple steps inside an episode, but they are just a queue of independent tickets. Each step is still one-shot labeling. Tickets do not affect each other. The queue order does not matter. There is no resource constraint. There is no carry-over state except a score list and counters. No later ticket depends on an earlier action. No policy evolves over the episode. No investigation outcome from step one informs step two.

So while the environment is technically episodic, it is not operationally long horizon. It is batching.

That difference matters. The best agents and best benchmarks separate “can classify one item” from “can operate over a process.” Right now your environment mainly measures the first.

### 5. They parameterize tasks rather than freezing one tiny benchmark

`ReasoningGymEnvironment` rebuilds datasets from `dataset_name`, `dataset_config`, `dataset_specs`, `seed`, and `size`. `BrowserGymEnvironment` can choose a benchmark and task. `Tbench2Environment` can resolve tasks by task ID or path, even downloading a repo cache if needed. `GitTaskEnvironment` supports task-specific base repo states. `REPLEnvironment` can accept context, task prompt, expected answer, recursion depth, and model parameters at reset. `FinQAEnvironment` iterates over a question bank with real data-backed tools.

Your environment has three tasks, but they are not truly different environments. They are the same tickets with a different subset of fields exposed through `allowed_fields`. That is a very weak notion of task diversity. Task difficulty is not created by different data generating processes, different hidden state, different workflows, or different action surfaces. It is created by output dimensionality alone.

That means the easy, medium, and hard tasks are less like three tasks and more like one task with three scoring schemas.

### 6. They take concurrency and runtime isolation seriously

Several reference environments explicitly set `SUPPORTS_CONCURRENT_SESSIONS = True`, including `REPLEnvironment`, `Tbench2Environment`, `ReasoningGymEnvironment`, `MazeEnvironment`, and some others. The framework core in `http_server.py` is built around WebSocket sessions, session capacity, session info, session factories, and asynchronous handling. `MCPEnvironment` has explicit async and sync step paths because the framework authors ran into real event-loop and deadlock issues. `Tbench2DockerEnvironment` handles Docker-in-Docker by copying task directories into containers rather than assuming host bind mounts. `Calendar` builds database sessions per tenant. `GitTaskEnvironment` assumes isolated workspaces. `BrowserGymEnvironment` does cleanup of resources.

Your environment inherits some capability from OpenEnv, but your own code does not actually engage with that depth. The server is mostly a minimal `create_app()` call plus a `/tasks` endpoint. There is no custom metadata. No custom concurrency choices. No session isolation logic beyond what the base server gives you. No runtime cleanup concerns because the environment owns almost no external resources. That simplicity is pleasant, but it also means the project is not stress-tested as a real environment service.

### 7. They integrate grounded external systems or simulators

This is where the biggest credibility gap appears.

`FinQAEnvironment` grounds answers in company tables and SQL. `GitTaskEnvironment` grounds tasks in actual repositories. `Tbench2Environment` grounds them in actual shell execution and tests. `BrowserGymEnvironment` grounds tasks in web environments. `TextArenaEnvironment` grounds them in game engines. `ChessEnvironment` grounds them in a real board state. `Calendar` grounds them in a stateful API-backed application.

Your environment is grounded in a JSON dataset. That is fine for a prototype, but it is dramatically easier to shortcut. If the environment does not provide tools, latent objects, or stateful consequences, the fastest route to a good score is to learn the labeling policy over the text. That is exactly what your current `inference.py` is doing.

If you want to beat more ambitious projects, you need to force the agent to do more than map n-grams to labels.

## Deep Audit of Your Current Project

### Overall strengths before the critique

Before I get more surgical, it is worth naming what is already good:

- The codebase is small enough to understand quickly.
- The naming is clear and the domain is coherent.
- Pydantic validation is used correctly in the core models.
- The taxonomy in `vocabulary.py` is readable and operational.
- The environment is deterministic given a seed.
- The three-task ladder is a decent pedagogical introduction.
- The tests, while limited, are not absent.
- The dataset has at least some intentional ambiguity and follow-up cases.

So this is not a bad project. It is a project that has not yet converted a good domain into a hard benchmark.

### Domain model and task structure

`vocabulary.py` defines a clean label space:

- 9 issue types
- 4 priorities
- 6 assignment groups
- 5 resolution actions
- 3 task IDs

The mapping dictionaries immediately reveal one important structural weakness: assignment group is fully determined by issue type. Every issue type maps to exactly one assignment group. That means the “assignment_group” prediction in task 3 is not an independent reasoning problem. Once the model gets issue type right, assignment group is a lookup. That collapses the apparent complexity of the hardest task.

The same problem exists, though less absolutely, for resolution action. `ISSUE_TYPE_TO_RESOLUTION_ACTION` already maps every issue type to a default resolution action. The dataset confirms that several issue types only ever use one resolution action:

- `feature_request -> acknowledge`
- `general_inquiry -> acknowledge`
- `onboarding -> fulfill`
- `service_request -> assign`
- `spam_phishing -> ignore`

Only a subset of issue types vary their resolution action in practice. So task 3 looks like a four-field prediction problem, but much of it is structurally reducible to issue type plus a few keyword exceptions. That is not how hard triage environments should work if the goal is to test agentic reasoning.

`server/tasks.py` compounds this by defining difficulty purely as output field count:

- Task 1: issue type only
- Task 2: issue type plus priority
- Task 3: full routing

The ticket pool is the same across tasks. There is no task-specific curation, no task-family-specific observation, no different process constraints, and no different control surface. The only thing that changes is what the grader will read from the submitted action.

That means your easy-medium-hard ladder is mostly a scoring ladder, not an environment ladder.

### Observation and state design

`HelpdeskTicketObservation` contains:

- task metadata
- `allowed_fields`
- `current_ticket`
- queue counts
- history

`current_ticket` exposes only:

- `ticket_id`
- `title`
- `requester`
- `description`

This is too little for a benchmark that wants to simulate real helpdesk operations, and it is oddly little given what your data already stores. `HelpdeskTicketRecord` also includes:

- `ambiguity_note`
- `related_ticket_id`

Those two fields are exactly the sort of structured hints that could turn this from flat classification into contextual triage. Yet `_build_observation()` discards them. That means the dataset contains richer structure than the observation contract.

The state is also minimal:

- `current_task_id`
- `seed`
- `queue_ticket_ids`
- `current_ticket_index`
- `per_ticket_scores`
- `total_reward`

This is enough for bookkeeping, but not enough for operational simulation. There is no notion of:

- queue ordering rationale
- account status
- customer tier
- outage context
- prior communication attempts
- internal notes
- pending escalations
- workload or resolver capacity
- elapsed time or SLA timers
- deduplication chains
- partial investigation state

The result is that the environment never becomes more informative or more demanding as the episode progresses. The state is a score ledger, not a world model.

Compare that with the stronger references:

- `BrowserGymState` tracks benchmark, task, URL, goal, max steps, cumulative reward.
- `REPLState` tracks context, prompt, iteration, namespace keys, final answer, total execution time.
- `Tbench2State` tracks task, session, command history, terminal readiness, last output.
- `TextArenaState` tracks turn, raw state, last reward, last info, environment identity.
- `FinQAState` tracks current question, company, ground truth, question ID.

Those states are not just counters. They represent the environment’s evolving operational memory. Yours mostly does not.

### Environment lifecycle

`HelpdeskTicketRoutingEnvironment.reset()` is straightforward:

- coerce `seed`
- get task definition
- seed RNG
- sample a queue size from 3 to 5
- sample that many tickets from the fixed dataset
- initialize state
- return the first observation

`step()`:

- validates reset happened
- grades action against current ticket
- computes reward
- advances to next ticket
- if done, computes trajectory reward
- otherwise returns immediate step reward

This is tidy. It is also shallow.

There is no environment mutation other than index movement. No internal state changes based on the chosen action. No branching. No action-dependent future ticket behavior. No queue reprioritization. No retries. No note writing. No escalation backlog. No “wrong earlier action causes downstream penalty.” The only environment response is score feedback.

A benchmark like this can still be useful, but it sits much closer to supervised evaluation than to agentic interaction. That becomes a competitive problem when the reference set includes environments where actions actually transform the world.

One subtle but important weakness is that `step()` does not enforce the task contract tightly. `HelpdeskTicketAction` allows all four fields to be present on any task, and `grade_action()` simply reads the fields relevant to the chosen `task_id`. Extra fields are ignored. That means the environment tells the agent “allowed_fields are X,” but it does not enforce “only X may be submitted.” It is not catastrophic, but it reflects a looser benchmark contract than the environment surface suggests.

### Grader and reward design

`server/grader.py` is the most benchmark-defining file in the project, and it currently underdelivers relative to its importance.

What is good:

- it has partial credit for issue-type confusions
- it has proximity-based scoring for priority
- task weights sum to 1
- it is deterministic
- it is easy to reason about

What is weak:

- the similarity tables are static, narrow, and handcrafted
- assignment group and resolution action are exact-match only even though the environment does not expose enough context to make some distinctions fully grounded
- there is no calibration check on over-escalation
- there is no queue-level objective
- there is no policy compliance signal
- there is no explanation consistency
- there is no distinction between “reasonable but conservative” and “reckless but lucky”

The biggest conceptual weakness is that the reward is local and label-centric. A strong helpdesk environment should care about operational behavior, not just answer key overlap.

For example, suppose two actions both get the final resolution action wrong:

- one escalates a low-risk general inquiry to security
- one acknowledges a critical account lockout without escalation

Today those mistakes mostly show up as missed fields in a flat weighted sum. But in real operations they are qualitatively different failures. One wastes specialist capacity. The other is a dangerous underreaction. A competitive benchmark should encode that asymmetry.

There is also a concrete implementation weakness in `compute_trajectory_reward()`. It computes:

- average per-ticket score
- minus `0.03 * overshoot`

But `overshoot = max(0, steps_taken - queue_size)`, and the environment ends the episode when the current ticket index reaches queue length. After that point, further stepping raises an error. So in the normal execution path, overshoot is effectively always zero. The code suggests the environment cares about extra wasted steps, but the environment does not actually permit them. That means part of the trajectory logic is decorative rather than active.

In strong benchmarks, reward code usually reveals the benchmark’s philosophy. In your project, the reward code mostly reveals the current label schema.

### Dataset design

`data/dataset.json` currently holds 45 tickets. The class distribution is not terrible for a prototype, but it is still small:

- `application_support`: 9
- `billing_license`: 7
- `service_request`: 6
- `security_compliance`: 5
- `spam_phishing`: 5
- `identity_access`: 4
- `onboarding`: 4
- `general_inquiry`: 3
- `feature_request`: 2

That is a tiny dataset for any benchmark that hopes to resist memorization or heuristic overfitting. The especially small classes are a concern. A benchmark with 2 feature requests and 3 general inquiries is not meaningfully testing generalization in those categories.

The priority distribution is also limited:

- critical: 9
- high: 15
- medium: 12
- low: 9

That is balanced enough to be usable, but not rich enough to encode the true structure of priority assignment. There is no obvious representation of customer segment, contractual urgency, outage blast radius, legal exposure, dependency graphs, or business calendar sensitivity. Priority is largely being inferred from words in the title and description, which is exactly what a heuristic baseline will exploit.

The dataset does have four ambiguous records and three follow-up linked records. That is good. But because the environment does not structurally expose `ambiguity_note` or `related_ticket_id`, those richer cases do not actually become richer environment mechanics. They mostly remain hints for the benchmark designer, not tools for the agent.

The follow-up handling is especially underused. Tickets like `ticket-038` and `ticket-045` clearly encode longitudinal customer frustration and repeated failure, which should change triage behavior. But the environment treats them like standalone text blobs. There is no action to inspect previous tickets. No thread retrieval. No stateful consequence from unresolved history. The environment has the seed of longitudinal realism and then does not build on it.

There is also no train/eval split, no hidden split, no procedural generation, no adversarial generation, and no OOD slice. The same fixed dataset defines the universe. That is fine for unit tests. It is weak for a benchmark intended to compete.

### Inference baseline and benchmark leakage

`inference.py` is more important than it may look, because it tells you how easy the benchmark is to shortcut.

The heuristic path:

- scans ticket text for fixed issue-type keywords in fixed order
- assigns priority from small keyword buckets
- assigns resolution action from issue type plus a few escalation and fulfillment keywords
- assigns assignment group from issue type mapping

That baseline is not merely a harmless example. It is a diagnostic of benchmark leakage. The easier it is to hand-author a ruleset that tracks your label policy, the less benchmark headroom you have.

And in this codebase, the baseline is not just simple. It is tightly coupled to the environment’s ontology:

- it uses the exact taxonomy constants
- it exploits the one-to-one issue-to-assignment mapping
- it exploits mostly deterministic issue-to-resolution defaults
- it assumes priority is keyword-addressable from the visible text alone

That means the benchmark currently invites ontology-driven shortcutting.

There is an even more concerning signal. The tests describe a heuristic baseline around `0.9400`, but a local code-faithful replay of the rule ordering in PowerShell over the full `data/dataset.json` gives a much weaker picture:

- issue type exact accuracy: about `0.7333`
- priority exact accuracy: about `0.3778`
- assignment exact accuracy: about `0.7333`
- resolution exact accuracy: about `0.6889`
- full task-3 exact match: about `0.2444`
- approximate weighted average score across tasks 1, 2, and 3: about `0.7344`

The exact number is less important than what it implies: the benchmark narrative about heuristic strength and the actual rule behavior appear out of sync. That can happen for several reasons:

- the tests are stale relative to current data
- the claimed baseline was measured on sampled queues rather than the whole dataset
- the heuristic ordering now creates more collisions than expected
- the benchmark evolved without a full-baseline recomputation

Whatever the cause, it is a warning sign. When benchmark claims and benchmark code diverge, trust in the environment falls.

### Test strategy

Your project has six test files. That is good relative to many small hackathon projects. But the content of the tests matters more than the count.

The most important limitation is that multiple tests stub the OpenEnv types, interfaces, or `create_app()` implementation rather than exercising the real installed framework. `tests/openenv_test_stubs.py` injects fake `openenv.core.env_server.types`. `tests/test_environment_smoke.py` and `tests/test_api_integration.py` patch in a fake `Environment` base class. `tests/test_api_integration.py` also installs a stub `create_app` that returns a small FastAPI app with simplified routes.

That means much of the test suite verifies your code against a locally simulated OpenEnv contract, not against the actual `openenv-core` dependency declared in `pyproject.toml`.

This is a big competitive weakness because the reference repository’s core is full of behavior that your test harness never touches:

- WebSocket `/ws` interactions
- session handling
- concurrency settings
- serialization edge cases
- metadata and schema endpoints
- MCP endpoints
- async step paths
- actual `EnvClient` protocol semantics

Your tests mostly prove that the environment behaves under your own simplified assumptions. That is useful, but it is not the same as proving robust OpenEnv integration.

The other limitation is that the tests are mostly shallow-contract tests:

- reset returns something valid
- step increments counts
- reward is in `[0, 1]`
- task IDs are present
- heuristic episodes do not error

Those are necessary. They are not sufficient for a competitive benchmark.

What is missing includes:

- real WebSocket end-to-end tests
- invalid action contract tests with actual framework validation
- tests for extra fields on restricted tasks
- concurrency tests
- seed reproducibility tests across actual server sessions
- golden regression tests on full-dataset benchmark score
- hidden/eval split integrity tests
- tests for ambiguity and follow-up handling
- tests that verify the environment is hard in the intended way, not just runnable

In short, the current test suite validates operability, not benchmark integrity.

## Critical Gaps That Matter Most

This section is the most actionable part of the report. If the goal is to beat stronger reference projects, these are the gaps that matter.

### Gap 1: The project is benchmarked as an environment, but designed as a classifier

The core problem is conceptual. Your code uses the OpenEnv interface, but the actual task shape is still mostly multi-label classification over short ticket text.

The better reference environments are hard because the agent has to interact:

- `BrowserGymEnvironment` asks the agent to act in a browser.
- `FinQAEnvironment` asks the agent to inspect tools and query structured data.
- `REPLEnvironment` asks the agent to iteratively execute code and decide when to finalize.
- `Tbench2Environment` asks the agent to manipulate a terminal workspace and then survive evaluation.
- `TextArenaEnvironment` asks the agent to play through game turns.

Your environment asks the agent to emit labels. Even when multiple tickets appear in a queue, the agent is still doing the same one-shot operation repeatedly. It is not exploring, not investigating, not mutating meaningful state, not managing resources, and not making action-sequence tradeoffs.

That difference is bigger than it looks. Once the benchmark is classifier-shaped, the fastest route to good performance is classifier-shaped too. The environment does not force the agent to behave like an operator. It only asks it to sound like one.

That is why the next leap must be architectural, not cosmetic.

### Gap 2: The hardest task is structurally easier than it claims

Task 3 appears to be a four-field routing task, but the ontology collapses much of the difficulty.

`ISSUE_TYPE_TO_ASSIGNMENT_GROUP` is one-to-one. If the agent gets issue type right, assignment group is already implied. That means one quarter of the task-3 score is mostly a lookup rather than a separate judgment call.

Resolution action is not fully deterministic, but it is still heavily compressed by issue type defaults. Several issue types have only one action in practice across the dataset. Others vary under small numbers of recognizable phrases such as legal threat, follow-up pressure, or explicit request wording.

So the “hard” task is closer to:

- infer issue type
- infer urgency from a few cues
- apply one deterministic mapping
- apply one mostly deterministic mapping with a few exceptions

That is not trivial, but it is much less rich than real service-desk routing. Real hard cases exist when the same visible ticket text can map to different actions depending on hidden context such as account tier, live incident status, prior history, or internal policy. Your environment does not currently model those cases.

### Gap 3: The environment underuses the best parts of its own data

Your dataset is more interesting than your observation contract.

`HelpdeskTicketRecord` contains `ambiguity_note` and `related_ticket_id`. Those are exactly the kinds of fields that could turn this into a stronger environment:

- ambiguity makes decisions less keyword-deterministic
- related ticket IDs create thread continuity
- follow-ups create escalation pressure and temporal realism

But `_build_observation()` discards them and only exposes the basic ticket text fields.

That has two consequences:

First, the richer authored structure is lost to the agent. Second, the benchmark stops short of the very complexity the dataset author was already beginning to encode.

This is one of the clearest signs that the current project is a first version. The seeds of a deeper environment are already present in the data model. The runtime contract just does not use them.

### Gap 4: There is no investigation loop

In real helpdesk operations, the visible complaint is rarely the whole decision problem.

An operator often needs to know:

- whether the requester is on an enterprise contract
- whether the problem aligns with an active outage
- whether the user is an admin
- whether prior tickets already established a root cause
- whether a security signal exists on the account
- whether a compliance deadline is legally binding
- whether the request is actually a duplicate

Your environment has no tool loop for this. The agent sees a title, requester, and description, then is expected to decide everything directly.

That makes the environment much easier to brute-force and much less realistic than the domains represented by the best reference projects. `FinQAEnvironment` does not ask the model to guess answers from wording alone; it gives tools. `GitTaskEnvironment` gives a repo. `Tbench2Environment` gives a terminal. `BrowserGymEnvironment` gives a browser. Your helpdesk environment gives a paragraph.

The fastest path to a stronger benchmark is to add internal tools and make the hardest scenarios impossible to solve reliably without using them.

### Gap 5: There is almost no internal economics

A good environment usually has some notion of tradeoff or cost even if it is not expressed as money.

In your environment:

- there is no time budget
- there is no backlog pressure
- there is no penalty for over-escalating except field mismatch
- there is no cost for routing everything to the safest specialist
- there is no consequence for queue ordering
- there is no tension between fast response and careful investigation

The queue exists, but it is not an economy. It is just a list.

That means the environment cannot really test operational judgment. It can only test whether the final labels match the benchmark designer’s answer key. Stronger environments force decisions under constraints. Your current implementation mostly scores unconstrained annotation.

### Gap 6: The reward story is thinner than the benchmark story

`grade_action()` is neat and deterministic, but it still mainly scores label overlap. It does not score operator quality.

There is no difference between:

- a cautious but slightly conservative routing choice
- a reckless underreaction that happens to get some partial credit
- an unnecessary escalation that wastes the security team
- a smart intermediate step that gathers evidence before final routing

Those distinctions do not exist because the action surface does not allow them and the reward design does not look for them.

There is also a direct implementation issue: `compute_trajectory_reward()` includes an overshoot penalty, but because the environment ends when the queue is exhausted and refuses later steps, overshoot does not really happen in the normal path. So part of the trajectory logic looks more meaningful than it actually is.

When reward code contains dead or decorative logic, trust in the benchmark drops.

### Gap 7: The current benchmark is highly vulnerable to ontology memorization

The more the task can be solved by memorizing your ontology and keyword policy, the lower the ceiling of the benchmark.

Right now the environment is vulnerable because:

- the dataset is small
- the label space is public and fixed
- some output fields are deterministic functions of others
- the observation is a short text blob
- the heuristic baseline directly encodes the ontology
- there is no hidden split or generator-based variation

The current inference script is a warning sign here. It is not just a demo baseline. It is evidence that a carefully chosen keyword system can cover a large fraction of the problem structure because the problem structure is currently that compressible.

If you want to build something harder to game, the benchmark must stop being reducible to a keyword policy plus a few ontology tables.

### Gap 8: The tests are too synthetic for the actual risk profile

The test suite checks that the environment is runnable. It does not yet prove that the benchmark is trustworthy.

The biggest limitation is the heavy use of stubs around the OpenEnv dependency boundary. Several tests replace the real OpenEnv types, interfaces, or `create_app()` implementation. That helps local testability, but it means the suite is not validating actual WebSocket session behavior, actual framework serialization, actual schema generation, or actual concurrency handling.

That is a serious gap if the environment is meant to compete with stronger projects. Reference environments are embedded in a framework that supports:

- WebSocket sessions
- session capacity and session info
- schema endpoints
- metadata endpoints
- MCP endpoints
- sync and async execution paths

Your current tests mostly validate business logic under a simplified local harness. That is still useful. It is just not enough to prove benchmark robustness.

There is also no strong integrity suite around the benchmark itself. Missing pieces include:

- full-dataset regression scoring
- hidden split integrity
- adversarial edge-case suites
- benchmark versioning checks
- ambiguity and follow-up behavior tests
- contract tests that verify the hard task is genuinely hard in the intended way

If you want the project to be taken seriously, the environment and the benchmark need separate test surfaces.

### Gap 9: The benchmark narrative and executable reality are drifting apart

A benchmark becomes fragile when people cannot tell which number to trust.

Your tests imply a strong heuristic baseline. The environment code and local replay of the actual heuristic rules over the dataset suggest a weaker story. That discrepancy may be caused by stale thresholds, changed data, queue sampling effects, or unrefreshed benchmark assumptions. Whatever the reason, it is not a small issue.

Strong benchmarks need executable answers to simple questions:

- what is the official baseline?
- how is it measured?
- on which split?
- with what seeds?
- on which version of the data?
- under which scenario families?

Right now those answers are not fully stabilized in code. The result is that the benchmark is harder to trust than it should be.

That may sound administrative, but it is actually competitive. A benchmark that feels ad hoc will lose to a benchmark that feels governed, even if both are interesting.

### Gap 10: The project does not yet have a competitive moat

The strongest environments in the reference set each have a clear identity:

- BrowserGym: browser-native multimodal interaction
- FinQA: tool-mediated reasoning over structured finance data
- REPL: iterative code execution and rubric-based finalization
- TBench2: terminal tasks grounded by executable evaluation
- Calendar: stateful tool ecosystem over application APIs
- Chess: adversarial long-horizon board play

Your current identity is “helpdesk routing from short ticket text.” That is useful, but not yet distinctive enough to dominate.

The domain itself can support a much stronger identity:

- service desk triage under partial observability
- enterprise support operations with tool use and policy constraints
- multi-ticket queue management under SLA and escalation economics

That is the moat you should build. The domain is good enough. The current benchmark shape is not yet deep enough to own it.

## What Specific Reference Environments Teach You

### BrowserGym: rich observations create real decision space

`BrowserGymObservation` includes text, URL, optional screenshot, goal, accessibility tree text, pruned HTML, error strings, and action-error flags. `BrowserGymEnvironment` carefully converts raw benchmark objects into those modalities and preserves additional metadata while filtering large raw fields.

The lesson is not “copy browser features.” The lesson is that an observation should support several reasoning strategies at once. Strong environments do not force everything through one narrow channel if the domain can naturally expose more.

Your helpdesk environment should likely move from a plain ticket view to a mixed observation view that includes structured context, queue state, optional note previews, and pointers to retrievable evidence. A stronger observation contract makes the environment harder to solve with surface heuristics and easier to use for real agent development.

### FinQA: tool use transforms a QA task into an environment

`FinQAEnvironment` is one of the most relevant reference environments for your redesign. It takes a question-answering domain that could have been implemented as “read prompt, output answer” and instead builds a tool-mediated workflow:

- list tools
- inspect table descriptions
- inspect table metadata
- run SQL queries
- submit final answer

The ground truth is hidden. The agent has to do work. The reward system then normalizes answer formats so the benchmark is measuring reasoning rather than answer string quirks.

Your helpdesk project should follow that pattern. The hard task should not be “read ticket and guess routing.” It should be “use service desk tools to investigate and then submit routing.” That would immediately raise the benchmark ceiling.

### REPL: process reward and outcome reward should be separate

`REPLEnvironment` is instructive because it distinguishes execution quality from final answer quality. The environment tracks iterations, namespace state, execution results, and finalization patterns. The rubric layer then separates outcome reward from process reward.

That is directly applicable to helpdesk operations. A strong service desk environment should separately measure:

- whether the final routing/action was correct
- whether the agent investigated responsibly
- whether the agent made avoidable operational mistakes
- whether the agent wasted steps or overused escalation

Without that split, you cannot tell the difference between good operations and lucky guessing.

### TBench2: grounded evaluation is a moat

`Tbench2Environment` is powerful because success is not a declared label. It is an executable check. The agent can manipulate a workspace and then call `evaluate`, which runs tests. That style of evaluation is very hard to fake and very easy to defend.

Helpdesk will not use pytest in the same way, but the principle transfers cleanly. A stronger helpdesk benchmark should evaluate against hidden operational truth and downstream effects, not just a visible label table. If the environment can compute whether the chosen action violated SLA policy, ignored an active incident, or misrouted a duplicate chain, then benchmark credibility goes up immediately.

### Calendar MCP: tool ecosystems can scale if the boundary is clean

The Calendar stack shows how a domain can become more realistic without exploding the action schema. The environment exposes tools, request context, user context, and database-backed state. Tool handlers are generic where possible and dynamic routing does a lot of the heavy lifting.

For your domain, that is a strong hint that helpdesk should probably become tool-centric. Instead of stuffing everything into one giant action object, expose a small set of operational tools. This will scale better, feel more realistic, and let you design harder scenarios without turning the action model into a kitchen sink.

### GitTask: reproducible scenario resets matter

`GitTaskEnvironment` is not the most feature-rich environment in the set, but it gets one important thing right: reproducible task state. Reset means something concrete. The environment can put you back into a known repo state efficiently.

You need the same discipline in scenario design. Instead of sampling any 3 to 5 tickets from one public pool, define reproducible episode families:

- urgent outage follow-up
- mixed billing queue
- false-positive security scare
- onboarding plus access control bundle
- executive escalation chain

Once episodes become scenario-driven rather than ticket-sampled, the benchmark will feel much more intentional.

### Chess and TextArena: delayed reward and auxiliary signals are valuable

`ChessEnvironment` plus `ChessWinLossRubric` shows how delayed reward can be modeled cleanly across a trajectory. `TextArenaEnvironment` plus its reward providers shows how auxiliary signals can coexist with the main reward without replacing it. Those patterns matter because helpdesk operations are not fully one-shot even when the final routing choice is what gets judged.

In a stronger version of your environment, you could preserve a main final reward while also emitting auxiliary channels such as:

- evidence quality
- duplicate-handling quality
- escalation efficiency
- SLA awareness
- customer experience quality
- policy compliance

Even if you keep one main scalar reward for training or evaluation, those auxiliary signals would make the benchmark much more diagnosable.

### ReasoningGym and Maze: simplicity is fine if it is honest

`ReasoningGymEnvironment` is a simple parameterized single-step environment. `MazeEnvironment` is a simple gridworld. Neither one pretends to be deeper than it is. That honesty is useful as a design lesson.

If you want to keep a light version of your current project, that is perfectly reasonable. But then it should be presented as a starter triage benchmark, not as a fully realized agentic operations environment. If you want to claim higher competitive value, the environment itself needs to support that claim with deeper mechanics.

## A Concrete Design for Beating the Stronger Projects

The right goal is not to imitate the broadest reference project. The right goal is to go much deeper in one domain you already own.

You do not need to out-BrowserGym BrowserGym. You do not need to out-TBench2 TBench2. You need to become clearly better at service desk operations simulation than the reference set is today.

### North star: build a service operations simulator

The strongest future version of this project looks more like an IT service desk simulator than a label prediction benchmark.

Core properties of that simulator should be:

- partially observed ticket and account state
- internal tools for investigation
- scenario families rather than one static pool
- multi-step resolution workflows
- queue-level tradeoffs
- policy-aware reward
- hidden evaluation truth

If you hit those properties, you will not just be polishing the current environment. You will be changing the category of the benchmark.

### Proposed visible entities

The agent should see richer but still realistic objects, for example:

- ticket thread summary
- current requester details
- account/org summary
- queue overview
- recent internal note previews
- live incident banner or incident tool access
- available tools
- allowed actions
- task budget and SLA hints

That does not mean every observation must be huge. It means the visible world should make the agent reason like an operator instead of like a labeler.

### Proposed hidden entities

The environment should own hidden state that determines the correct policy:

- canonical root-cause category
- customer tier
- resolver ownership
- actual business impact
- active incident linkage
- prior unresolved duplicates
- whether manual escalation is necessary or wasteful
- whether policy requires a specific handling path
- whether the ticket is self-servable by documented guidance

These hidden variables are what create genuinely hard cases. Two tickets that look similar on the surface should sometimes route differently because the hidden state differs.

### Proposed action surface

I would split the action space into investigation actions and commitment actions.

Investigation actions:

- `lookup_requester`
- `get_account_plan`
- `get_related_tickets`
- `check_service_health`
- `search_kb`
- `inspect_internal_notes`
- `get_security_signals`
- `get_asset_or_license_state`

Operational actions:

- `add_internal_note`
- `request_more_info`
- `merge_duplicate`
- `set_priority`
- `assign_group`
- `escalate`
- `acknowledge`
- `submit_final_decision`

This preserves your current routing taxonomy while forcing the agent to earn the final answer through interaction.

### Proposed task families

Replace the current output-field ladder with scenario families.

1. **Baseline classification**
   Keep a simple version of the current task for calibration.

2. **Priority under operational context**
   Add visible account metadata and SLA hints.

3. **Tool-assisted routing**
   Hard cases require evidence retrieval.

4. **Follow-up chain handling**
   Correct routing depends on thread history and prior failures.

5. **Duplicate resolution**
   The agent must detect and merge with existing tickets or note the linkage.

6. **Queue management**
   Multiple tickets compete for limited steps or limited escalation budget.

7. **Incident-aware triage**
   Correct behavior depends on checking active incident state.

8. **Policy-constrained operations**
   Compliance, security, or executive-account policies change what the correct action is.

Now difficulty comes from task structure, not just output dimensionality.

### Proposed reward design

A strong reward design for this domain should likely have four layers.

Layer 1: **final outcome correctness**

- correct issue family
- correct priority
- correct resolver team
- correct action

Layer 2: **operational policy correctness**

- no violation of mandatory escalation rules
- no unjustified critical priority
- no missed compliance deadlines
- no unsupported closure

Layer 3: **process quality**

- useful tool use
- correct duplicate inspection
- efficient evidence gathering
- no unnecessary specialist escalation

Layer 4: **episode economics**

- queue-wide quality
- backlog harm
- escalation cost
- SLA miss cost

That may sound like a lot, but you do not need to expose all of it as one scalar at once. Some of it can be stored as metadata or auxiliary reward channels first.

### Proposed data strategy

Do not try to hand-author ten thousand fully custom tickets from scratch. Instead, build a layered data strategy.

Layer A: curated seed cases

- your best handcrafted exemplars
- ambiguous pairs
- follow-up chains
- adversarial near-neighbors

Layer B: templated scenario generation

- same underlying issue with different requester tiers
- same wording with different hidden incident context
- duplicate vs non-duplicate versions
- billing dispute with and without outage linkage

Layer C: hidden benchmark splits

- development split
- public validation split
- private evaluation split

Layer D: scenario tagging

- issue family
- ambiguity level
- investigation depth required
- tool requirement
- risk class
- queue pressure

This approach gives you scale without giving up control.

## File-by-File Improvement Plan for This Repository

This section ties the redesign back to the actual code you already have. The point is to show how the current repo can evolve into the stronger benchmark rather than be abandoned.

### `models.py`

Right now the models encode the benchmark as a label submission problem. That is fine for version one and too restrictive for version two.

I would keep the existing validation patterns, but I would expand the schema into typed action families and typed observation payloads.

Recommended direction:

- keep `HelpdeskTicketRecord`, but add typed visible vs hidden fields
- replace the loose `current_ticket: Optional[dict[str, str]]` with a ticket-view model
- split actions into investigation actions and final submission actions
- add typed structures for tool results, notes, queue items, and thread previews
- enrich state with scenario metadata, action audit trail, and resource counters

Why this matters:

As long as the schema itself says “the agent submits optional routing fields,” every other part of the environment will naturally stay classifier-shaped. Schema is architecture. If you want the environment to feel agentic, the models have to make agentic behavior first-class.

### `server/environment.py`

This file is currently the main reason the benchmark feels thin. It is clean, but it is clean because it has very little world logic.

I would evolve it in stages.

Stage 1:

- expose structured thread/follow-up information
- enforce task contracts more tightly
- store full action history, not just scores
- make scenario metadata visible

Stage 2:

- add tool dispatch for investigation actions
- maintain scenario-local hidden state
- let actions mutate environment state
- support final decision submission separately from intermediate investigation

Stage 3:

- add queue-level episodes with budget constraints
- let earlier choices affect later ticket handling
- introduce scenario-specific logic for duplicates, incidents, and policy constraints

Why this matters:

This file should become the simulator, not just the grader entrypoint.

### `server/tasks.py`

This file needs the most conceptual change after the environment itself.

The current task list is:

- task 1: issue type only
- task 2: issue type plus priority
- task 3: full routing

That is too narrow. I would turn `tasks.py` into a scenario-family registry instead.

For example:

- `single_ticket_classification`
- `priority_under_sla`
- `tool_assisted_routing`
- `duplicate_chain_resolution`
- `incident_aware_triage`
- `queue_optimization`
- `policy_constrained_security_case`

Each task family should define:

- visible observation contract
- allowed actions
- hidden truth generator
- episode budget
- reward composition
- benchmark split membership

Why this matters:

Right now tasks differ by scoring columns. A strong benchmark needs tasks that differ by problem structure.

### `server/grader.py`

This file should stop being only a lookup-based scorer and become the place where service-desk policy is encoded.

I would keep the basic idea of partial credit, but move from a pure field-overlap worldview to a policy-and-outcome worldview.

Examples of richer scoring logic:

- small penalty for unnecessary escalation
- strong penalty for under-prioritizing active access outages
- reward for correctly linking duplicates
- reward for choosing acknowledgment before final resolution when that is the right workflow
- penalty for routing compliance work to general support
- scenario-aware scoring where the same visible ticket can score differently depending on retrieved evidence

Why this matters:

The grader is the actual benchmark. It should reflect operational quality, not only taxonomy overlap.

### `server/reward.py`

This file is a good place to simplify and then rebuild.

First, remove or redesign logic that is not meaningfully active, such as the current overshoot penalty that normal episode flow does not really trigger.

Then add reward layers deliberately:

- final decision score
- process score
- economics score
- optional auxiliary diagnostics

Why this matters:

A benchmark becomes much easier to improve if the reward code honestly reflects what is being optimized.

### `server/app.py`

This file is currently fine for a minimal environment, but it should grow once the environment grows.

Recommended additions:

- environment metadata endpoint support if you want richer UI or benchmark introspection
- possibly custom routes for benchmark info, scenario families, or baseline metadata
- cleaner packaging around path setup once the project stabilizes

Why this matters:

This is not the highest-priority file, but stronger benchmark ergonomics do help credibility and usability.

### `data/dataset.json`

This file should evolve from “the benchmark” into “part of the benchmark.”

Keep a curated hand-authored slice, but do not let one public JSON file define the whole environment forever.

Recommended evolution:

- expand the dataset substantially
- add many more feature request and general inquiry cases
- add multiple duplicate chains
- add hidden context fields
- add templated variants of existing scenarios
- create a private evaluation bank

Why this matters:

A tiny fixed public dataset makes memorization too easy and benchmark claims too brittle.

### `inference.py`

This file is useful, but it currently plays several roles at once:

- demo script
- heuristic baseline
- optional LLM runner
- environment smoke path

I would separate those responsibilities.

Recommended structure:

- one official deterministic baseline runner
- one optional tool-using baseline runner once tools exist
- one separate example script for simple local usage
- one benchmark harness that records split, seed, scenario family, and version

Why this matters:

Benchmarks need reproducible baselines more than they need convenient demos.

### `tests/`

The most important change after environment design is testing philosophy.

I would split tests into at least four groups:

1. **unit tests**
   Validation, scoring primitives, dataset loaders, tool helpers.

2. **real integration tests**
   Actual OpenEnv app, actual serialization, actual WebSocket interactions.

3. **benchmark regression tests**
   Fixed scenario suites, stable baseline scores, hidden split checks.

4. **integrity tests**
   No task leakage, no duplicate split contamination, no benchmark version drift.

Why this matters:

A serious benchmark is a data product, an environment product, and an evaluation product. The tests should reflect all three.

## Practical Roadmap

### Phase 1: Make the current environment honest and sturdier

This is the fastest and cheapest improvement phase. Do this even if you are not ready for a full redesign.

Goals:

- expose thread/follow-up structure
- tighten task contracts
- recompute and stabilize baseline measurements
- add a hidden evaluation split
- remove decorative reward logic
- improve test realism

Deliverables:

- stronger observation model
- benchmark regression script
- real integration tests
- scenario-family-aware tasks, even if still text-only

This phase will not yet make the environment winner-beating, but it will make it much more defensible.

### Phase 2: Add tool-assisted investigation

This is the highest-return phase because it changes the category of the benchmark.

Minimum viable tool set:

- requester/account lookup
- related-ticket retrieval
- service health lookup
- KB search
- final decision submission

Once those exist, create scenario families where the visible ticket text is insufficient without tool use. That immediately raises the benchmark ceiling and reduces shortcutability.

### Phase 3: Add operational economics and queue-level behavior

After tool use works, add:

- queue-wide episodes
- time or action budgets
- escalation cost
- SLA miss cost
- duplicate-handling benefit
- specialist-capacity awareness

This turns the environment from a case-by-case annotation task into an operational management task.

### Phase 4: Add benchmark governance

At this point you should formalize:

- public vs private splits
- scenario-family tags
- official baselines
- benchmark versioning
- scorecards by scenario family
- release notes for benchmark changes

This is what makes the project not just interesting, but trustworthy.

## Prioritized Recommendation List

If I had to choose only ten improvements, in order, I would choose these:

1. Stop defining difficulty only by `allowed_fields`.
2. Add investigation tools and final submission as separate actions.
3. Break the deterministic issue-type-to-assignment shortcut.
4. Make resolution depend on hidden operational context more often.
5. Surface follow-up and related-ticket structure.
6. Expand data and add hidden eval splits.
7. Add process-aware reward and remove dead trajectory logic.
8. Add queue-level economics and limited budgets.
9. Replace stub-heavy integration tests with real framework tests.
10. Publish a stable benchmark harness and official baseline measurement.

## Final Assessment

After a deep code read, my conclusion is simple:

Your project is promising, readable, and based on a very strong domain. But in its current form it is still a compact routing benchmark, not yet a high-ceiling service-operations environment.

The better reference environments in `OpenEnv/envs` are better not because they are bigger for the sake of being bigger, but because they force the agent to operate inside state, tools, or consequences that cannot be collapsed into label mapping so easily.

The encouraging part is that your domain can support exactly that kind of benchmark. IT helpdesk operations naturally contain ambiguity, hidden context, tool use, policy constraints, long threads, queue pressure, and downstream costs. Very few toy domains offer that combination so cleanly.

So the right move is not to abandon the project. The right move is to evolve it.

If you keep the current shape and only add more tickets, you will get a better classifier benchmark. That may be useful, but it probably will not beat the strongest reference projects.

If you turn this into a tool-assisted, partially observed, multi-step service-operations simulator with stronger reward design and stronger benchmark governance, then you can absolutely build something more compelling than many of the reference environments, because your domain has the right raw material for a benchmark that is both realistic and highly evaluable.

The domain is already winner material.

The current implementation is starter material.

The opportunity is to close that gap deliberately.

## Appendix A: Comparative Scorecard

The table below is not a scientific benchmark. It is a code-read scorecard based on the implementations reviewed in this report. The goal is to make the gap tangible.

| Dimension | Your project now | Strong reference environments |
| --- | --- | --- |
| Action richness | Low | Medium to very high |
| Hidden state depth | Low | Medium to high |
| Tool use | None | Present in FinQA, Calendar, TBench2, Git, REPL |
| Multistep interaction | Low-medium | Medium to high |
| Queue/process economics | Very low | Medium in some envs, high in operational ones |
| Reward sophistication | Low-medium | Medium to high |
| Benchmark anti-overfitting | Low | Medium |
| Runtime realism | Low | Medium to high |
| Testing depth | Low-medium | Medium to high at repo scale |
| Domain relevance | High | Varies by env |
| Potential ceiling | High | Already demonstrated in several envs |

The most important row here is the last one. Your current implementation is not yet at the same level as the strongest references, but the domain ceiling is absolutely high enough to catch up and possibly surpass them if you execute the redesign well.

## Appendix B: What You Should Preserve

When teams hear “major redesign,” they often accidentally throw away the parts that were already working. I do not recommend that here.

The current project has several strengths that should be preserved as you expand it:

### 1. Preserve the compactness of the taxonomy

The label space in `vocabulary.py` is clear and product-shaped. It is not bloated. Even when the environment becomes tool-based and stateful, keep the routing ontology understandable. The problem with the current benchmark is not that the taxonomy is wrong. The problem is that the environment around the taxonomy is too thin.

### 2. Preserve deterministic core scoring where possible

Even after you add process reward and hidden context, keep as much deterministic scoring as possible. One reason your current project is easy to debug is that the grader is inspectable. Do not replace everything with opaque LLM judging if you can avoid it. Use explicit hidden truth and rule-based evaluation for most of the benchmark, and reserve softer judging only for areas that truly need it.

### 3. Preserve readability

The current codebase is easy to onboard into. That is an asset. Several bigger reference environments are strong, but also much harder to reason about quickly because they wrap external systems or broad framework machinery. As you deepen this project, keep modules well-separated:

- models
- scenario generation
- environment runtime
- tools
- scoring
- reward composition
- benchmark harness

That separation will make future iteration much faster.

### 4. Preserve seeded reproducibility

Your existing environment is deterministic under a seed, and that is worth keeping. Stronger benchmarks become much easier to trust when a given scenario family plus seed reproduces the same world state. As you add hidden context and generators, make seed behavior even more explicit instead of less.

### 5. Preserve explicit validation

The Pydantic validation in the current models is a quiet strength. Keep that discipline. As the action surface grows, validation becomes more important, not less. Tools and action types should reject malformed inputs cleanly so that environment failures are informative rather than muddy.

## Appendix C: Example Scenario Families for Version 2

To make the redesign more concrete, here are example scenario families that would feel much closer to a winner-level helpdesk benchmark.

### Scenario Family 1: Access outage with incident ambiguity

Visible state:

- multiple users report being locked out
- one requester sounds urgent
- another sounds like a normal password reset

Hidden state:

- there is an active identity provider outage
- some tickets are duplicate symptoms of the same incident

Tools needed:

- `check_service_health`
- `get_related_tickets`
- `lookup_requester_role`

What this tests:

- whether the agent distinguishes isolated access issues from systemic incidents
- whether it avoids handling every case as an independent ticket
- whether it correctly prioritizes executive or admin users without overreacting on every case

### Scenario Family 2: Billing dispute tied to product defect

Visible state:

- customer says they were charged incorrectly
- another case mentions checkout failures

Hidden state:

- the billing dispute is caused by a known application defect that duplicated transactions

Tools needed:

- `search_related_tickets`
- `check_service_health`
- `read_internal_incident_note`

What this tests:

- whether the agent routes based on real causal structure rather than superficial department ownership
- whether it recognizes that pure billing handling is insufficient because engineering is involved

### Scenario Family 3: Compliance deadline with account-context twist

Visible state:

- requester references GDPR or legal obligation

Hidden state:

- some requests are legitimate deletion requests
- some are actually admin-level data export requests misphrased as deletion
- some belong to customers on contracts with defined response obligations

Tools needed:

- `lookup_contract_tier`
- `retrieve_policy_snippet`
- `get_account_data_scope`

What this tests:

- whether the agent can combine legal wording with account and policy context
- whether it overroutes all legal-sounding tickets to the same team

### Scenario Family 4: Duplicate-heavy queue optimization

Visible state:

- ten tickets in a queue
- several appear to be related

Hidden state:

- six are duplicates of two underlying issues
- one low-volume ticket is actually the most SLA-critical

Tools needed:

- `search_related_tickets`
- `merge_duplicate`
- `set_priority`
- `submit_queue_plan`

What this tests:

- whether the agent can manage a queue as a system
- whether it reduces work through linkage
- whether it balances urgency against volume

### Scenario Family 5: Feature request versus broken workflow

Visible state:

- customer asks for export filters or better reporting

Hidden state:

- in some scenarios the feature genuinely does not exist
- in others the feature exists but the customer lacks permissions or is using the wrong path

Tools needed:

- `search_kb`
- `lookup_plan_features`
- `inspect_recent_product_change`

What this tests:

- whether the agent treats every request for missing functionality as a feature request
- whether it can separate education/support from roadmap input

## Appendix D: Red Flags to Avoid During the Redesign

There are a few ways a redesign like this can go wrong. Avoid these.

### 1. Do not add tools that are merely decorative

If a hard task can still be solved reliably without using the tools, then the tool surface is just benchmark theater. The hard scenario families should be designed so that retrieved evidence actually changes the correct answer.

### 2. Do not make every scenario gigantic

Richer does not mean bloated. Some scenarios should stay compact. The goal is meaningful hidden context, not maximum token count.

### 3. Do not replace all scoring with LLM judging

Use explicit hidden truth and deterministic scoring wherever possible. Opaque judging should be a last resort, not a default.

### 4. Do not let the ontology become a maze

Your current taxonomy is pleasantly clean. Keep it that way. More realism should come from state and evidence, not from exploding the label space into dozens of nearly indistinguishable categories.

### 5. Do not forget benchmark governance

If you add scenario generation but do not formalize splits, baselines, and versioning, you will create a cooler environment without creating a more trustworthy benchmark.
