import sys
from html import escape
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path so `models` and `server` are importable
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from fastapi import HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, RedirectResponse
from openenv.core.env_server import create_app

from models import HelpdeskTicketAction, HelpdeskTicketObservation
from server.environment import HelpdeskTicketRoutingEnvironment
from server.grader import grade_action
from server.tasks import TASKS, load_dataset
from vocabulary import APP_ENV_NAME, PROJECT_TITLE, TEAM_NAME

app = create_app(
    HelpdeskTicketRoutingEnvironment,
    HelpdeskTicketAction,
    HelpdeskTicketObservation,
    env_name=APP_ENV_NAME,
)


class GraderRequest(BaseModel):
    task_id: int
    ticket_id: str
    action: dict[str, Any]


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/web", status_code=307)


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": t["id"],
                "name": t["name"],
                "difficulty": t["difficulty"],
                "instructions": t["instructions"],
                "allowed_fields": t["allowed_fields"],
            }
            for t in TASKS.values()
        ]
    }


@app.get("/web", response_class=HTMLResponse)
def web_ui():
    dataset = load_dataset()
    dataset_size = len(dataset)
    alternate_route_count = sum(
        1 for ticket in dataset if ticket.alternate_route_score_multiplier > 0.0
    )
    clustered_case_count = sum(1 for ticket in dataset if ticket.service_cluster_id)
    hidden_context_case_count = sum(
        1
        for ticket in dataset
        if ticket.ambiguity_note
        or ticket.related_ticket_id
        or ticket.planning_note
        or ticket.customer_update_note
    )
    incident_sensitive_count = sum(1 for ticket in dataset if ticket.incident_recommended)

    difficulty_labels = {
        "easy": "Guided",
        "medium": "Contextual",
        "hard": "Adaptive",
    }
    task_cards = "".join(
        f"""
        <article class="task-card difficulty-{escape(t['difficulty'])}">
          <div class="task-head">
            <span class="task-id">Task {t['id']}</span>
            <span class="difficulty-pill">{escape(difficulty_labels.get(t['difficulty'], t['difficulty']).upper())}</span>
          </div>
          <h3>{escape(t['name'])}</h3>
          <p>{escape(t['instructions'])}</p>
          <div class="field-row">
            {''.join(f'<span class="field-chip">{escape(field)}</span>' for field in t['allowed_fields'])}
          </div>
        </article>
        """
        for t in TASKS.values()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{escape(APP_ENV_NAME)}</title>
    <style>
      :root {{
        --bg: #07131b;
        --bg-soft: #0b1c27;
        --panel: rgba(15, 32, 44, 0.84);
        --panel-strong: rgba(12, 26, 37, 0.94);
        --line: rgba(173, 215, 230, 0.16);
        --line-strong: rgba(173, 215, 230, 0.28);
        --text: #ecf5f7;
        --muted: #97aeb7;
        --accent: #4fd1c5;
        --accent-strong: #1cb0a4;
        --accent-warm: #ffb454;
        --success: #7fdf9f;
        --shadow: 0 28px 80px rgba(0, 0, 0, 0.32);
        --radius-xl: 28px;
        --radius-lg: 20px;
        --radius-md: 14px;
      }}

      * {{
        box-sizing: border-box;
      }}

      html {{
        scroll-behavior: smooth;
      }}

      body {{
        margin: 0;
        min-height: 100vh;
        color: var(--text);
        background:
          radial-gradient(circle at 12% 18%, rgba(79, 209, 197, 0.18), transparent 26%),
          radial-gradient(circle at 82% 20%, rgba(255, 180, 84, 0.16), transparent 22%),
          radial-gradient(circle at 50% 100%, rgba(79, 209, 197, 0.12), transparent 35%),
          linear-gradient(180deg, #07131b 0%, #0b1821 52%, #07131b 100%);
        font-family: "Aptos", "Segoe UI Variable Text", "Trebuchet MS", sans-serif;
      }}

      body::before {{
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
          linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 36px 36px;
        mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.9), transparent 92%);
      }}

      .shell {{
        width: min(1180px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 28px 0 56px;
      }}

      .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        margin-bottom: 22px;
        padding: 16px 20px;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(10, 23, 32, 0.68);
        backdrop-filter: blur(14px);
      }}

      .brand {{
        display: flex;
        align-items: center;
        gap: 14px;
      }}

      .brand-mark {{
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background:
          linear-gradient(145deg, rgba(79, 209, 197, 0.96), rgba(28, 176, 164, 0.75));
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.32);
        position: relative;
      }}

      .brand-mark::after {{
        content: "";
        position: absolute;
        inset: 10px;
        border-radius: 10px;
        border: 2px solid rgba(7, 19, 27, 0.75);
      }}

      .eyebrow {{
        margin: 0 0 4px;
        color: var(--accent);
        font-size: 0.78rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
      }}

      .brand h1 {{
        margin: 0;
        font-family: "Bahnschrift", "Aptos Display", "Trebuchet MS", sans-serif;
        font-size: 1.05rem;
        letter-spacing: 0.03em;
      }}

      .nav-links {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }}

      .nav-links a,
      .button {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        text-decoration: none;
        color: var(--text);
        border-radius: 999px;
        border: 1px solid var(--line);
        padding: 11px 16px;
        font-size: 0.94rem;
        transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
      }}

      .nav-links a:hover,
      .button:hover {{
        transform: translateY(-1px);
        border-color: var(--line-strong);
      }}

      .button.primary {{
        background: linear-gradient(135deg, rgba(79, 209, 197, 0.22), rgba(28, 176, 164, 0.18));
        border-color: rgba(79, 209, 197, 0.35);
      }}

      .button.secondary {{
        background: linear-gradient(135deg, rgba(255, 180, 84, 0.14), rgba(255, 180, 84, 0.08));
        border-color: rgba(255, 180, 84, 0.25);
      }}

      .hero {{
        position: relative;
        overflow: hidden;
        display: grid;
        grid-template-columns: minmax(0, 1.3fr) minmax(300px, 0.9fr);
        gap: 24px;
        padding: 36px;
        border: 1px solid var(--line);
        border-radius: var(--radius-xl);
        background:
          linear-gradient(160deg, rgba(15, 33, 44, 0.92), rgba(8, 21, 29, 0.9)),
          radial-gradient(circle at top right, rgba(255, 180, 84, 0.16), transparent 28%);
        box-shadow: var(--shadow);
      }}

      .hero::after {{
        content: "";
        position: absolute;
        inset: auto -8% -32% 44%;
        height: 340px;
        background: radial-gradient(circle, rgba(79, 209, 197, 0.2), transparent 62%);
        pointer-events: none;
      }}

      .hero-copy,
      .hero-panel {{
        position: relative;
        z-index: 1;
      }}

      .hero-copy h2 {{
        margin: 0 0 14px;
        max-width: 10.5ch;
        font-family: "Bahnschrift", "Aptos Display", "Trebuchet MS", sans-serif;
        font-size: clamp(2.7rem, 6vw, 4.8rem);
        line-height: 0.95;
        letter-spacing: -0.05em;
      }}

      .hero-copy p {{
        margin: 0;
        max-width: 62ch;
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.7;
      }}

      .hero-kickers {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 18px 0 22px;
      }}

      .kicker {{
        padding: 9px 14px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.04);
        color: #d5e4e9;
        font-size: 0.9rem;
      }}

      .hero-actions {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 26px;
      }}

      .hero-panel {{
        align-self: stretch;
        display: grid;
        gap: 14px;
        padding: 18px;
        border-radius: 22px;
        border: 1px solid rgba(79, 209, 197, 0.16);
        background: rgba(7, 19, 27, 0.46);
        backdrop-filter: blur(14px);
      }}

      .panel-title {{
        margin: 0;
        font-size: 0.88rem;
        color: var(--muted);
        letter-spacing: 0.14em;
        text-transform: uppercase;
      }}

      .signal-card {{
        padding: 16px;
        border-radius: 18px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.035);
      }}

      .signal-card strong {{
        display: block;
        margin-bottom: 6px;
        font-size: 1rem;
      }}

      .signal-card span {{
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.55;
      }}

      .stats-grid,
      .feature-grid,
      .task-grid,
      .shortcut-grid {{
        display: grid;
        gap: 16px;
        margin-top: 20px;
      }}

      .stats-grid {{
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }}

      .feature-grid {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }}

      .task-grid {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }}

      .shortcut-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}

      .stat-card,
      .feature-card,
      .shortcut-card,
      .task-card {{
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        background: var(--panel);
        backdrop-filter: blur(16px);
        box-shadow: var(--shadow);
      }}

      .stat-card {{
        padding: 20px;
      }}

      .stat-card .value {{
        display: block;
        margin-bottom: 8px;
        font-family: "Bahnschrift", "Aptos Display", "Trebuchet MS", sans-serif;
        font-size: 2rem;
        letter-spacing: -0.04em;
      }}

      .stat-card .label,
      .stat-card .hint,
      .feature-card p,
      .shortcut-card p,
      .task-card p {{
        color: var(--muted);
      }}

      .stat-card .label {{
        display: block;
        margin-bottom: 6px;
        font-size: 0.92rem;
      }}

      .stat-card .hint {{
        font-size: 0.86rem;
        line-height: 1.5;
      }}

      .section {{
        margin-top: 24px;
        padding: 28px;
        border: 1px solid var(--line);
        border-radius: var(--radius-xl);
        background: linear-gradient(180deg, rgba(11, 26, 37, 0.84), rgba(9, 21, 30, 0.88));
      }}

      .section-head {{
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 18px;
        margin-bottom: 18px;
      }}

      .section-head h3 {{
        margin: 0 0 8px;
        font-family: "Bahnschrift", "Aptos Display", "Trebuchet MS", sans-serif;
        font-size: 1.75rem;
        letter-spacing: -0.03em;
      }}

      .section-head p {{
        margin: 0;
        max-width: 64ch;
        color: var(--muted);
        line-height: 1.65;
      }}

      .feature-card,
      .shortcut-card {{
        padding: 20px;
      }}

      .feature-card h4,
      .shortcut-card h4,
      .task-card h3 {{
        margin: 0 0 10px;
        font-size: 1.04rem;
      }}

      .task-card {{
        padding: 20px;
        position: relative;
        overflow: hidden;
      }}

      .task-card::before {{
        content: "";
        position: absolute;
        inset: 0 auto auto 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, rgba(79, 209, 197, 0.95), rgba(255, 180, 84, 0.72));
      }}

      .task-card.difficulty-easy::before {{
        background: linear-gradient(90deg, rgba(127, 223, 159, 0.95), rgba(79, 209, 197, 0.7));
      }}

      .task-card.difficulty-medium::before {{
        background: linear-gradient(90deg, rgba(79, 209, 197, 0.95), rgba(120, 196, 230, 0.72));
      }}

      .task-card.difficulty-hard::before {{
        background: linear-gradient(90deg, rgba(255, 180, 84, 0.95), rgba(255, 122, 72, 0.78));
      }}

      .task-head {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 16px;
      }}

      .task-id {{
        color: var(--muted);
        font-size: 0.84rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }}

      .difficulty-pill {{
        padding: 7px 10px;
        border-radius: 999px;
        border: 1px solid var(--line);
        font-size: 0.74rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #f6fafb;
        background: rgba(255, 255, 255, 0.05);
      }}

      .field-row,
      .chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
      }}

      .field-chip,
      .mini-chip {{
        padding: 8px 11px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.04);
        color: #d9e7eb;
        font-size: 0.82rem;
      }}

      .feature-card ul {{
        margin: 12px 0 0;
        padding-left: 18px;
        color: var(--muted);
        line-height: 1.65;
      }}

      .shortcut-card code {{
        display: block;
        margin: 12px 0 14px;
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #d9fcf7;
        font-family: "Cascadia Code", "Consolas", monospace;
        font-size: 0.88rem;
        white-space: nowrap;
        overflow-x: auto;
      }}

      .footer {{
        margin-top: 20px;
        padding: 18px 6px 8px;
        color: var(--muted);
        font-size: 0.92rem;
      }}

      @keyframes rise {{
        from {{
          opacity: 0;
          transform: translateY(12px);
        }}
        to {{
          opacity: 1;
          transform: translateY(0);
        }}
      }}

      .hero,
      .section,
      .stat-card,
      .task-card,
      .feature-card,
      .shortcut-card {{
        animation: rise 420ms ease both;
      }}

      @media (max-width: 980px) {{
        .hero,
        .stats-grid,
        .feature-grid,
        .task-grid,
        .shortcut-grid {{
          grid-template-columns: 1fr;
        }}

        .topbar,
        .section-head {{
          border-radius: 24px;
          flex-direction: column;
          align-items: flex-start;
        }}
      }}

      @media (max-width: 640px) {{
        .shell {{
          width: min(100vw - 18px, 1180px);
          padding-top: 14px;
        }}

        .hero,
        .section {{
          padding: 22px;
        }}

        .hero-copy h2 {{
          max-width: none;
          font-size: clamp(2.4rem, 14vw, 3.5rem);
        }}

        .nav-links,
        .hero-actions {{
          width: 100%;
        }}

        .nav-links a,
        .button {{
          flex: 1 1 180px;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="shell">
      <header class="topbar">
        <div class="brand">
          <div class="brand-mark" aria-hidden="true"></div>
          <div>
            <p class="eyebrow">OpenEnv Environment</p>
            <h1>{escape(PROJECT_TITLE)}</h1>
          </div>
        </div>
        <nav class="nav-links">
          <a href="/health">Health</a>
          <a href="/tasks">Tasks JSON</a>
          <a href="/docs">API Docs</a>
        </nav>
      </header>

      <section class="hero">
        <div class="hero-copy">
          <p class="eyebrow">{escape(APP_ENV_NAME)}</p>
          <h2>Queue decisions that actually carry forward.</h2>
          <p>
            A sleek benchmark surface for sequential helpdesk routing: hidden context,
            cluster-aware follow-ons, incident handling, deferrals, and a terminal rubric
            that rewards queue strategy instead of isolated classification alone.
          </p>
          <div class="hero-kickers">
            <span class="kicker">Task family: easy to hard</span>
            <span class="kicker">Closed-form grader</span>
            <span class="kicker">Queue-level terminal objective</span>
          </div>
          <div class="hero-actions">
            <a class="button primary" href="/docs">Explore the API</a>
            <a class="button secondary" href="/baseline?task_id=3&amp;seed=42">Run Hard Baseline</a>
            <a class="button" href="/tasks">Inspect Task Definitions</a>
          </div>
        </div>

        <aside class="hero-panel">
          <p class="panel-title">Why This Stands Out</p>
          <div class="signal-card">
            <strong>Not just ticket labels</strong>
            <span>Medium and hard episodes now carry cluster state, follow-up debt, queue pressure, and operational actions across the whole episode.</span>
          </div>
          <div class="signal-card">
            <strong>Judge-friendly surface</strong>
            <span>Clear API entry points, deterministic grading, and a landing page that explains the benchmark without making anyone read code first.</span>
          </div>
          <div class="signal-card">
            <strong>Built by {escape(TEAM_NAME)}</strong>
            <span>Designed for OpenEnv evaluation, local policy comparison, and fast demoability during judging.</span>
          </div>
        </aside>
      </section>

      <section class="stats-grid" aria-label="Benchmark stats">
        <article class="stat-card">
          <span class="value">{dataset_size}</span>
          <span class="label">Tickets in the grounded dataset</span>
          <span class="hint">Curated records plus queue mutation mechanics create repeatable but non-trivial episodes.</span>
        </article>
        <article class="stat-card">
          <span class="value">{alternate_route_count}</span>
          <span class="label">Capacity-aware alternate routes</span>
          <span class="hint">The grader can reward declared fallback routes instead of collapsing to all-or-nothing exact match.</span>
        </article>
        <article class="stat-card">
          <span class="value">{clustered_case_count}</span>
          <span class="label">Cluster-linked or coordinated cases</span>
          <span class="hint">Handling one ticket can stabilize or destabilize the downstream tickets in the same workstream.</span>
        </article>
        <article class="stat-card">
          <span class="value">{hidden_context_case_count}</span>
          <span class="label">Hidden-context routing cases</span>
          <span class="hint">Investigation tools matter because key evidence does not appear in the initial observation by default.</span>
        </article>
      </section>

      <section class="section">
        <div class="section-head">
          <div>
            <p class="eyebrow">Task Ladder</p>
            <h3>One benchmark family, not three disconnected demos</h3>
          </div>
          <p>
            The difficulty ladder keeps the same full-routing output while progressively changing
            observability, queue dependencies, and operational pressure.
          </p>
        </div>
        <div class="task-grid">
          {task_cards}
        </div>
      </section>

      <section class="section">
        <div class="section-head">
          <div>
            <p class="eyebrow">Environment Signals</p>
            <h3>What the agent is balancing</h3>
          </div>
          <p>
            The benchmark is designed so strong policy choices change later tickets, incident
            coverage, and terminal queue quality instead of just nudging shaped reward.
          </p>
        </div>
        <div class="feature-grid">
          <article class="feature-card">
            <h4>Hidden context retrieval</h4>
            <p>Related-ticket previews, requester history, internal routing notes, queue cluster summaries, and capacity forecasts are revealed through explicit tool use.</p>
            <div class="chip-row">
              <span class="mini-chip">investigate</span>
              <span class="mini-chip">request_info</span>
              <span class="mini-chip">cluster summary</span>
            </div>
          </article>
          <article class="feature-card">
            <h4>Operational actions with consequences</h4>
            <p>Deferrals can raise later urgency, incident handling can reduce downstream debt, and weak handling can spawn or worsen follow-up work.</p>
            <div class="chip-row">
              <span class="mini-chip">defer</span>
              <span class="mini-chip">open_incident</span>
              <span class="mini-chip">follow-up spawning</span>
            </div>
          </article>
          <article class="feature-card">
            <h4>Queue-level terminal rubric</h4>
            <p>Final scoring blends routing trajectory quality with queue management quality so agents are rewarded for coherent episode strategy, not just isolated ticket matches.</p>
            <div class="chip-row">
              <span class="mini-chip">terminal rubric</span>
              <span class="mini-chip">queue quality</span>
              <span class="mini-chip">planning-aware</span>
            </div>
          </article>
        </div>
      </section>

      <section class="section">
        <div class="section-head">
          <div>
            <p class="eyebrow">Quick Routes</p>
            <h3>Fast ways to demo the environment</h3>
          </div>
          <p>
            Useful entry points for judges, reviewers, or anyone trying to get signal from the project quickly.
          </p>
        </div>
        <div class="shortcut-grid">
          <article class="shortcut-card">
            <h4>Interactive API docs</h4>
            <p>Browse the full OpenEnv-compatible surface, request models, and built-in helper endpoints.</p>
            <code>GET /docs</code>
            <a class="button primary" href="/docs">Open Docs</a>
          </article>
          <article class="shortcut-card">
            <h4>Task manifest</h4>
            <p>Inspect the easy, medium, and hard task definitions exactly as exposed by the server.</p>
            <code>GET /tasks</code>
            <a class="button" href="/tasks">View Tasks</a>
          </article>
          <article class="shortcut-card">
            <h4>Hard-task baseline rollout</h4>
            <p>See a deterministic baseline episode over the hardest queue with the current environment logic.</p>
            <code>GET /baseline?task_id=3&amp;seed=42</code>
            <a class="button secondary" href="/baseline?task_id=3&amp;seed=42">Run Baseline</a>
          </article>
          <article class="shortcut-card">
            <h4>Health and deployment status</h4>
            <p>Quick check that the service is alive and ready for OpenEnv-style evaluation requests.</p>
            <code>GET /health</code>
            <a class="button" href="/health">Check Health</a>
          </article>
        </div>
      </section>

      <footer class="footer">
        <span>{escape(PROJECT_TITLE)} • {escape(APP_ENV_NAME)} • {incident_sensitive_count} incident-sensitive records surfaced in the current dataset snapshot.</span>
      </footer>
    </main>
  </body>
</html>"""
    return HTMLResponse(content=html)


def _build_baseline_submit_action(
    ticket: dict[str, Any], allowed_fields: list[str]
) -> HelpdeskTicketAction:
    import inference

    candidate = inference.heuristic_action(ticket, allowed_fields)
    candidate, _ = inference.apply_domain_overrides(ticket, candidate, allowed_fields)
    return HelpdeskTicketAction(**candidate)


@app.get("/baseline")
def baseline_rollout(task_id: int = 1, seed: int = 42):
    import inference

    env = HelpdeskTicketRoutingEnvironment()
    observation = env.reset(seed=seed, task_id=task_id)
    steps: list[dict[str, Any]] = []

    while not observation.done:
        ticket = observation.current_ticket
        if ticket is None:
            break

        investigate, tool_name = inference.should_investigate(ticket, observation.history)
        if (
            investigate
            and tool_name is not None
            and observation.investigation_budget_remaining > 0
        ):
            investigate_action = HelpdeskTicketAction(
                action_type="investigate",
                tool_name=tool_name,
                tool_target_ticket_id=ticket.get("related_ticket_id"),
            )
            observation = env.step(investigate_action)
            steps.append(
                {
                    "action": investigate_action.model_dump(exclude_none=True),
                    "reward": observation.reward,
                    "done": observation.done,
                    "action_source": "baseline_investigate",
                }
            )
            if observation.done:
                break
            ticket = observation.current_ticket
            if ticket is None:
                break

        action = _build_baseline_submit_action(
            inference.merge_ticket_context(ticket, observation),
            list(observation.allowed_fields),
        )
        observation = env.step(action)
        steps.append(
            {
                "action": action.model_dump(exclude_none=True),
                "reward": observation.reward,
                "done": observation.done,
                "action_source": "baseline_submit",
            }
        )

    return {
        "task_id": task_id,
        "seed": seed,
        "step_count": len(steps),
        "final_reward": observation.reward,
        "rubric_reward": observation.rubric_reward,
        "steps": steps,
    }


@app.post("/grader")
def grader_preview(request: GraderRequest):
    ticket = next(
        (record for record in load_dataset() if record.ticket_id == request.ticket_id),
        None,
    )
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticket_id: {request.ticket_id}")

    try:
        action = HelpdeskTicketAction.model_validate(request.action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    score, breakdown = grade_action(action, ticket, request.task_id)
    return {
        "task_id": request.task_id,
        "ticket_id": request.ticket_id,
        "score": score,
        "breakdown": breakdown,
        "expected": {
            "issue_type": ticket.issue_type,
            "priority": ticket.priority,
            "assignment_group": ticket.assignment_group,
            "resolution_action": ticket.resolution_action,
        },
        "submitted": action.model_dump(exclude_none=True),
    }


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
