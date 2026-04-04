from __future__ import annotations

import sys
import types
from typing import Any, Optional

from pydantic import BaseModel


def install_openenv_type_stubs() -> None:
    openenv_module = types.ModuleType("openenv")
    core_module = types.ModuleType("openenv.core")
    env_server_module = types.ModuleType("openenv.core.env_server")
    types_module = types.ModuleType("openenv.core.env_server.types")

    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        done: bool = False
        reward: Optional[float] = None
        metadata: dict[str, Any] = {}

    class State(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0

    types_module.Action = Action
    types_module.Observation = Observation
    types_module.State = State

    env_server_module.types = types_module
    core_module.env_server = env_server_module
    openenv_module.core = core_module

    sys.modules["openenv"] = openenv_module
    sys.modules["openenv.core"] = core_module
    sys.modules["openenv.core.env_server"] = env_server_module
    sys.modules["openenv.core.env_server.types"] = types_module


install_openenv_type_stubs()
