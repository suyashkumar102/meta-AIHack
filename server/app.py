import sys
from pathlib import Path

# Ensure repo root is on sys.path so `models` and `server` are importable
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
