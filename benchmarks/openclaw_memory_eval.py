from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    name: str
    session_key: str
    turns: list[str]
    expected_substrings: list[str]


CASES = [
    EvalCase(
        name="preference_recall_after_distractor",
        session_key="eval:preference",
        turns=[
            "Remember that my editor preference is Helix.",
            "Ignore the editor topic and talk about build logs.",
            "What editor preference did I give you?",
        ],
        expected_substrings=["Helix"],
    ),
    EvalCase(
        name="correction_adoption",
        session_key="eval:correction",
        turns=[
            "Remember that the project codename is Pike.",
            "Correction: the project codename is Delta Pike.",
            "What is the current project codename?",
        ],
        expected_substrings=["Delta Pike"],
    ),
    EvalCase(
        name="session_isolation",
        session_key="eval:isolation:a",
        turns=[
            "Remember that this session's color is blue.",
            "What is this session's color?",
        ],
        expected_substrings=["blue"],
    ),
    EvalCase(
        name="task_state_continuity",
        session_key="eval:task-state",
        turns=[
            "We have completed todo items 1 and 2. Todo 3 is next.",
            "After a long unrelated discussion about deployment, what todo is next?",
        ],
        expected_substrings=["3"],
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenClaw-style memory eval harness.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--model", default="delta-mem-fake")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true", help="Emit JSON lines.")
    args = parser.parse_args()

    results = [
        run_case(case, base_url=args.base_url, model=args.model, timeout=args.timeout)
        for case in CASES
    ]

    if args.json:
        for result in results:
            print(json.dumps(result, sort_keys=True))
    else:
        for result in results:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status} {result['name']} ({result['latency_ms']} ms)")
            if not result["passed"]:
                print(f"  expected: {result['expected_substrings']}")
                print(f"  got: {result['final_response']!r}")

    failed = sum(1 for result in results if not result["passed"])
    return 1 if failed else 0


def run_case(case: EvalCase, *, base_url: str, model: str, timeout: float) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    final_response = ""
    started = time.monotonic()

    for turn in case.turns:
        messages.append({"role": "user", "content": turn})
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": 0,
        }
        response = post_chat(
            base_url=base_url,
            payload=payload,
            session_key=case.session_key,
            timeout=timeout,
        )
        final_response = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": final_response})

    latency_ms = round((time.monotonic() - started) * 1000)
    passed = all(expected.lower() in final_response.lower() for expected in case.expected_substrings)
    return {
        "name": case.name,
        "session_key": case.session_key,
        "passed": passed,
        "latency_ms": latency_ms,
        "expected_substrings": case.expected_substrings,
        "final_response": final_response,
    }


def post_chat(
    *,
    base_url: str,
    payload: dict[str, Any],
    session_key: str,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-OpenClaw-Session-Key": session_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


if __name__ == "__main__":
    sys.exit(main())
