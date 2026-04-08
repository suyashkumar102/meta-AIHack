# IT Helpdesk Ticket Routing OpenEnv - Mentor Guide

This document is written as if I am mentoring someone who only knows basic Python and wants to understand how to build this project well.

The goal is not to teach every code detail. The goal is to explain the real-world thinking behind the project so you understand what you are building, why each piece exists, and how all the parts fit together.

## Start With The Big Picture

This project is a small simulation of an IT helpdesk team.

A company receives support tickets like:

- "I was charged twice after the integration outage"
- "My admin account is locked and I cannot access payroll"
- "Can we extend this contractor account for two more weeks?"
- "We think this email is a phishing attempt"

A human helpdesk lead does not just read those tickets and say "this is category X."
They also decide:

- how urgent it is
- which team should own it
- what the next action should be
- whether to gather more information first
- whether this is big enough to open an incident
- whether to delay one ticket because a more important cluster is coming

That is why this project is stronger than a simple text classifier. It tries to model a small operational workflow, not just a label lookup.

## What OpenEnv Means In Plain English

OpenEnv is a way of turning a real task into an environment that an agent can interact with step by step.

Instead of asking a model one question and scoring one answer, we create a loop:

1. the environment shows the agent the current situation
2. the agent chooses an action
3. the environment changes state
4. the agent sees the new situation
5. this continues until the episode ends

That matters because many real jobs are not one-shot question answering. They involve:

- incomplete information
- intermediate choices
- trade-offs
- consequences that show up later

Helpdesk work fits this pattern well.

## The Real-World Problem We Chose

The business problem is IT helpdesk ticket routing.

In a real company, support work usually has four important decisions:

1. `issue_type`
   - What kind of problem is this really?
   - Example: billing issue, access issue, phishing report, onboarding request.
2. `priority`
   - How urgent is it?
   - Example: low, medium, high, critical.
3. `assignment_group`
   - Which team should own it?
   - Example: service desk, security team, procurement, onboarding ops.
4. `resolution_action`
   - What should happen next?
   - Example: fulfill it directly, assign it, escalate it, acknowledge it, or ignore it.

These four decisions are the heart of the benchmark.

## Why This Problem Is Good For A Hackathon

This use case is strong because it has the right mix of realism and clarity.

It is realistic:

- companies really do route tickets like this every day
- mistakes are costly
- urgency and ownership matter

It is structured:

- the inputs are messy natural language
- the outputs are typed and easy to score

It is judge-friendly:

- someone can understand the workflow quickly
- the labels are concrete

It is agentic:

- the agent can investigate
- the agent can ask for more info
- the agent can defer
- the agent can open an incident
- earlier decisions can affect later tickets

## The Mental Model: Think Like A Shift Lead

The best way to understand the environment is to imagine you are the helpdesk shift lead for the next 20 minutes.

Tickets are arriving in a short queue.

You cannot treat each ticket as if it lives alone.

Sometimes:

- two tickets are part of the same outage
- one customer keeps opening related follow-ups
- your security team has limited bandwidth
- if you ignore a risky ticket now, it will create another ticket later
- if you open an incident early, later related tickets become easier to manage

That is the real heart of the benchmark.

## What The Agent Actually Does

The agent interacts with the environment one step at a time.

For each ticket, it can choose one of several actions.

### 1. `submit`

This means:

"I know enough. Here is my routing decision."

The agent provides:

- issue type
- priority
- assignment group
- resolution action

Real-world example:

A ticket says, "A new contractor starts Monday and needs access to the standard onboarding apps."

The agent may decide:

- issue type: `onboarding`
- priority: `medium`
- assignment group: `onboarding_ops`
- resolution action: `fulfill`

### 2. `investigate`

This means:

"I do not want to commit yet. Let me look up one more internal signal."

This is similar to a real support lead opening internal notes, checking a related case, or reviewing requester history before making a decision.

### 3. `request_info`

This means:

"The current ticket is missing something important. I want clarification before routing it strongly."

Real-world example:

A customer writes:

"We need help before the board meeting."

That is too vague. You may need to know:

- what system is affected
- whether it is a live outage
- whether security is involved

### 4. `defer`

This means:

"I am intentionally pushing this later in the queue because another item is more urgent or I expect better context soon."

This is not the same as ignoring the ticket.
It is a strategic queue decision.

Real-world example:

You have one ticket about a pricing clarification and another about a company-wide identity lockout.
You may defer the pricing question so you can stabilize the outage cluster first.

### 5. `open_incident`

This means:

"This is bigger than a normal ticket. I need to reserve incident-handling capacity."

Real-world example:

If multiple customers are reporting the same outage or privileged-access failure, opening an incident early can prevent chaos later in the queue.

## Why The Tools Exist

The investigation tools are there because real support work is rarely solved from the first sentence alone.

The environment includes tools such as:

- related ticket lookup
- requester history lookup
- internal routing note lookup
- queue capacity forecast
- queue cluster summary

Think of these as controlled windows into the rest of the system.

They matter because some tickets are intentionally incomplete.

For example:

- the visible ticket may look like a normal billing issue
- the internal routing note may reveal it is actually connected to an application outage
- the queue cluster summary may reveal there are two more related tickets behind it
- the capacity forecast may reveal the preferred team is overloaded, so a fallback route becomes reasonable

This is how the project creates decision-making instead of simple label prediction.

## Why Earlier Decisions Affect Later Tickets

This is one of the most important ideas in the whole project.

If your benchmark has no carry-over state, it is often just classification repeated several times.

This project tries to avoid that by making the queue matter.

Examples:

- if you handle an outage ticket well, later tickets from the same cluster become easier to route
- if you handle it poorly, later tickets can become more urgent or more confused
- if you open an incident, related tickets may already have incident coverage
- if you defer too many things, SLA pressure grows
- if you burn the wrong team's capacity early, later tickets may need fallback routing

In simple terms:

the world changes because of what the agent did earlier.

That is what makes the benchmark feel more like operations and less like a quiz.

## The Three Tasks And Why They Exist

All three tasks now use full routing. That is an important design choice.

We are not making one task "just classify the issue type" anymore. We keep the core job the same and change how hard the world is.

### Task 1: Guided Full Routing

This is the easiest version.

The ticket is mostly visible.
The agent still performs full routing, but the world is simpler and more single-ticket.

This task teaches:

"Can you route a normal helpdesk ticket correctly?"

### Task 2: Contextual Full Routing

This is the medium version.

Now some useful context is hidden unless the agent investigates or asks for more information.
There is also moderate queue carry-over.

This task teaches:

"Can you route well when the ticket alone is not enough?"

### Task 3: Adaptive Queue Routing

This is the hard version.

Now the agent must handle:

- hidden decisive context
- queue capacity pressure
- incidents
- clustered requests
- deferrals
- follow-up tickets created by weak earlier handling

This task teaches:

"Can you manage the queue like an operator, not just label a ticket?"

## What The Dataset Must Do

The dataset is not just a list of random support messages.

It must teach the benchmark what "good routing" looks like.

A useful dataset for this project needs:

- clear easy examples
- medium examples where urgency matters
- ambiguous examples where the wording can mislead a naive policy
- related tickets that belong to the same cluster
- tickets where fallback routing can still be acceptable
- tickets where weak handling should logically create follow-up work

Real-world example:

If a ticket says:

"The seat increase is blocked and finance is also confused about prorating"

that is not a perfectly clean one-label case.
It could pull toward procurement, license operations, or service desk depending on queue pressure and business context.

Those are the kinds of examples that make the environment interesting.

## How Scoring Works Conceptually

The grader should feel like a tough but fair manager.

It should not be vague.

It should not say:

"Anything somewhat close gets points."

Instead, it should say:

- exact answers get the most credit
- a few near misses can receive partial credit
- fallback routes only count when they were explicitly designed to count
- clearly wrong answers get low or zero credit

That is why the grader is deterministic and narrow.

This matters for two reasons:

1. judges can trust the benchmark
2. an agent actually gets a meaningful learning signal

## Why Reward Is Not Exactly The Same As Grading

This is a subtle but important idea.

The final rubric score tells us how good the overall episode was.

The step reward helps the agent learn during the episode.

You can think of it like coaching during a football match:

- the final match result is the real outcome
- the coach's feedback during the game helps the team adjust sooner

In this project:

- terminal reward reflects overall routing plus queue-management quality
- step rewards make the environment less sparse
- unnecessary investigation or poor operational choices can carry penalties

So the final score is the verdict, while the step reward is the training signal.

## The Difference Between "Correct Ticket Routing" And "Good Queue Management"

This difference separates average benchmarks from stronger ones.

A ticket can be locally correct but globally poor.

Example:

- yes, security might be the best owner for a certain ticket
- but if the security queue is already overloaded and the task explicitly allows a fallback operational route, a smart agent may choose the alternate route

That is why this project now includes:

- alternate acceptable routes on selected tickets
- capacity-aware routing
- queue-management score
- cluster stabilization and destabilization

A good benchmark should reward not just being correct in isolation, but being operationally sensible.

## How To Explain The Main Files To A Beginner

If you are teaching this project to someone new, use these analogies.

### `server/tasks.py`

This is the curriculum.

It says:

- what the tasks are
- how hard they are
- what kinds of tickets exist

### `data/dataset.json`

This is the casebook.

It is the collection of real-looking helpdesk scenarios that power the environment.

### `server/environment.py`

This is the game master.

It keeps track of:

- which ticket is current
- what the queue looks like
- what happened earlier
- what the next observation should be

### `server/grader.py`

This is the scorekeeper.

It decides how good a routing answer was.

### `server/reward.py`

This is the coach.

It turns raw outcomes into feedback signals the agent can learn from.

### `inference.py`

This is the example player.

It shows how an agent can interact with the environment.

### `server/app.py`

This is the front desk.

It exposes the environment through web endpoints so tools and evaluators can use it.

## How I Would Teach A Beginner To Build This Project From Scratch

If you were starting from zero, I would teach the build order like this.

### Step 1: Choose A Real Workflow

Do not start with code.
Start with the business process.

Ask:

- who is the user?
- what decision are they making?
- what makes that decision hard?
- what happens if they get it wrong?

For us, the answers were:

- the user is a helpdesk routing agent
- the decisions are issue type, priority, owner, and next action
- the hard parts are ambiguity, queue pressure, and incomplete information
- mistakes cause delays, wrong ownership, and follow-up work

### Step 2: Freeze The Vocabulary

Before coding, decide the labels clearly.

If the team keeps changing label names midway, everything becomes unstable:

- dataset
- grader
- prompts
- docs
- tests

This is why a frozen vocabulary is so important.

### Step 3: Build Realistic Example Cases

Write tickets the way real people write them:

- incomplete
- emotional
- slightly messy
- not perfectly labeled in the text

If every ticket literally contains the answer, the benchmark becomes a keyword game.

### Step 4: Decide What The Agent Sees Immediately

Not everything should be visible at once.

Ask:

- what would a real support analyst know right away?
- what would require investigation?
- what would require asking someone?

That decision creates the need for tools and intermediate actions.

### Step 5: Add Actions Beyond Final Submission

If the only action is "submit the answer," you are probably building classification.

To make it feel operational, add actions that shape the path:

- investigate
- ask for clarification
- defer
- escalate or open incident

These are realistic and easy to explain.

### Step 6: Make State Carry Over

This is where many projects stay shallow.

You need earlier choices to matter later.

For example:

- capacity should be reduced after use
- related tickets should react to earlier handling
- follow-up tickets should appear when earlier work was weak

Without this, you do not really have a sequential benchmark.

### Step 7: Design Deterministic Grading

The grader should be explainable to a judge in under a minute.

That usually means:

- exact match for most things
- a small number of explicit partial-credit rules
- no secret fuzzy logic

### Step 8: Add Reward Shaping Carefully

Reward shaping should help learning, not distort the benchmark.

Good shaping:

- rewards useful investigation
- discourages wasteful probing
- gently rewards good operational flow

Bad shaping:

- makes a silly exploit better than actually solving the task

### Step 9: Build A Baseline Agent

Always include a runner that can play the environment.

It does not need to be perfect.
It just needs to prove the environment works and give judges something concrete to run.

### Step 10: Make It Easy To Validate And Deploy

A good benchmark is not just interesting. It is runnable.

That means:

- clean metadata
- clear docs
- Docker support
- validation passing
- a landing page that makes sense to a judge

## Common Beginner Mistakes To Avoid

### Mistake 1: Building A Fancy Classifier And Calling It An Environment

If nothing carries over between steps, you probably do not have a true environment yet.

### Mistake 2: Making The Grader Too Fuzzy

If almost every answer gets partial credit, your score stops being trustworthy.

### Mistake 3: Making The Hard Task Easy For Heuristics

If a simple keyword rule gets near-perfect scores, the benchmark will not feel meaningful.

### Mistake 4: Adding Random Complexity Instead Of Business Logic

Harder is not always better.
Complexity should come from realistic workflow pressure, not arbitrary tricks.

### Mistake 5: Writing Docs Only For Teammates

Hackathon judges are outsiders.
Your docs must help a smart new reader understand the project quickly.

## How To Talk About This Project In A Demo

If you need to explain the project fast, say this:

"We built an OpenEnv benchmark for IT helpdesk routing. The agent does not just classify tickets. It manages a short operational queue, can investigate hidden context, request clarification, defer work, open incidents, and make routing choices whose consequences affect later tickets. The scoring is deterministic, but the environment still has real trade-offs because queue pressure and related-ticket clusters change what good handling looks like."

That is the shortest honest pitch.

## What Makes This Project Strong Today

The current version is strongest in these areas:

- clear real-world workflow
- structured, judge-friendly outputs
- deterministic grading
- multi-step operational actions
- queue-level consequences
- cluster-aware carry-over state
- clean packaging and validation story

## What Would Make It Even Stronger Later

If this project kept growing after the hackathon, the next upgrades would be:

- make more of the consequences emerge from a general simulator instead of authored rules
- increase the data diversity further
- train stronger learned policies instead of relying mainly on deterministic policy search
- add more business objectives like cost, customer satisfaction, and resolver fatigue

## One-Minute Recap

If you forget everything else, remember this:

- this project simulates helpdesk queue management, not just ticket classification
- the agent must choose both what the ticket means and what to do next
- some useful context is hidden and must be uncovered through actions
- earlier choices affect later tickets
- the grader is deterministic so the benchmark stays trustworthy
- the project is built to be understandable, runnable, and useful as an OpenEnv environment
