import sys

import pytest

from delta_mem_sidecar.official_runtime import OfficialDeltaRuntime
from delta_mem_sidecar.runtime import ChatMessage


def test_official_runtime_missing_upstream_dependency_is_actionable(monkeypatch) -> None:
    for module_name in list(sys.modules):
        if module_name == "deltamem" or module_name.startswith("deltamem."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        sys,
        "path",
        [path for path in sys.path if "delta-Mem-upstream" not in path],
    )

    runtime = OfficialDeltaRuntime(
        model_path="/models/qwen",
        adapter_dir="/models/delta-adapter",
    )
    state = runtime.fresh_state()

    with pytest.raises(RuntimeError, match="Official .+ runtime is not installed"):
        runtime.generate(
            messages=[ChatMessage(role="user", content="hello")],
            state=state,
        )
