from __future__ import annotations

import json
from pathlib import Path

from models import HelpdeskTicketRecord
from vocabulary import TASK_IDS


TASKS = {
    1: {
        "id": 1,
        "name": "Issue Type Classification",
        "difficulty": "easy",
        "instructions": (
            "Read the ticket and select the single best IT issue type. "
            "You may investigate first, then submit a final routing answer."
        ),
        "allowed_fields": ["issue_type"],
    },
    2: {
        "id": 2,
        "name": "Issue Type And Priority",
        "difficulty": "medium",
        "instructions": (
            "Read the ticket, select the best IT issue type, and estimate the "
            "correct operational priority. If the observation includes ambiguity "
            "or follow-up context, use it. You may investigate before you submit."
        ),
        "allowed_fields": ["issue_type", "priority"],
    },
    3: {
        "id": 3,
        "name": "Full Ticket Routing",
        "difficulty": "hard",
        "instructions": (
            "Perform full helpdesk routing by selecting the best issue type, "
            "priority, assignment group, and resolution action for the ticket. "
            "Use any ambiguity notes or related-ticket previews when present. "
            "You may investigate with tools before you submit the final action."
        ),
        "allowed_fields": [
            "issue_type",
            "priority",
            "assignment_group",
            "resolution_action",
        ],
    },
}

assert tuple(TASKS.keys()) == TASK_IDS


def load_dataset() -> list[HelpdeskTicketRecord]:
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "dataset.json"
    # Accept UTF-8 files saved with a BOM, which is common on Windows editors.
    with dataset_path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    return [HelpdeskTicketRecord.model_validate(r) for r in raw]


def get_task_definition(task_id: int) -> dict:
    if task_id not in TASKS:
        raise ValueError(f"Unsupported task_id: {task_id}")
    return TASKS[task_id]
