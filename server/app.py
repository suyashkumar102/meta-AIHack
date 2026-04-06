import sys
from pathlib import Path

# Ensure repo root is on sys.path so `models` and `server` are importable
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_app

from models import HelpdeskTicketAction, HelpdeskTicketObservation
from server.environment import HelpdeskTicketRoutingEnvironment
from server.tasks import TASKS
from vocabulary import APP_ENV_NAME

app = create_app(
    HelpdeskTicketRoutingEnvironment,
    HelpdeskTicketAction,
    HelpdeskTicketObservation,
    env_name=APP_ENV_NAME,
)


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
    task_rows = "".join(
        f"<tr><td>{t['id']}</td><td>{t['name']}</td><td>{t['difficulty']}</td></tr>"
        for t in TASKS.values()
    )
    html = f"""<!DOCTYPE html>
<html><head><title>{APP_ENV_NAME}</title></head>
<body>
<h1>{APP_ENV_NAME}</h1>
<p>Version: 0.1.0 | <a href="/health">Health</a> | <a href="/docs">API Docs</a></p>
<h2>Tasks</h2>
<table border="1"><tr><th>ID</th><th>Name</th><th>Difficulty</th></tr>
{task_rows}
</table>
</body></html>"""
    return HTMLResponse(content=html)


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
