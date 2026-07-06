#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stop hook — intercept wrap-up while a Trellis task is still un-finished.

Trellis marks "the line is actively on a task" by writing that task's path into
`.trellis/.current-task` (via `task.py start`); `task.py finish` clears it. If a
session tries to stop while `.current-task` is still set, the task was never
finished — this hook surfaces that once at stop time with an interception list,
so `task.py finish` / archive doesn't get silently skipped.

Contract (Claude Code Stop hook):
  stdin  : JSON payload; we read `stop_hook_active` and `cwd`.
  stdout : `{"decision":"block","reason":"..."}` to hold the stop once, or
           nothing (exit 0) to allow the stop.

Loop-safe: if `stop_hook_active` is already true (we blocked on the previous
stop), we allow the stop — the reminder is one-shot, never an infinite hold.
Fail-open: any error → allow the stop (this hook must never wedge a session).

Self-test:  python3 finish-task-stop.py --self-test
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def find_trellis_root(start: Path) -> Optional[Path]:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".trellis").is_dir():
            return cand
    return None


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _developer(root: Path) -> Optional[str]:
    dev = root / ".trellis" / ".developer"
    try:
        for line in dev.read_text(encoding="utf-8").splitlines():
            if line.startswith("name="):
                return line.split("=", 1)[1].strip()
    except OSError:
        return None
    return None


def gather_dangling(root: Path) -> Tuple[Optional[dict], List[Tuple[str, str]]]:
    """Return (current-task-info-or-None, [(dir, title) sibling in_progress ...]).

    current-task-info: {"dir","title","status"} for the un-finished current task,
    or None when `.current-task` is empty/absent (nothing to intercept).
    """
    cur_file = root / ".trellis" / ".current-task"
    current: Optional[dict] = None
    try:
        cur_rel = cur_file.read_text(encoding="utf-8").strip()
    except OSError:
        cur_rel = ""

    if cur_rel:
        tj = root / cur_rel / "task.json"
        data = _read_json(tj) or {}
        current = {
            "dir": cur_rel,
            "title": data.get("title") or data.get("name") or Path(cur_rel).name,
            "status": data.get("status", "unknown"),
        }

    # informational: other in_progress tasks owned by this developer
    dev = _developer(root)
    siblings: List[Tuple[str, str]] = []
    tasks_dir = root / ".trellis" / "tasks"
    if tasks_dir.is_dir():
        for tj in sorted(tasks_dir.glob("*/task.json")):
            data = _read_json(tj)
            if not data:
                continue
            if data.get("status") != "in_progress":
                continue
            if dev and (data.get("assignee") or "") != dev:
                continue
            rel = f".trellis/tasks/{tj.parent.name}"
            if current and rel == current["dir"]:
                continue
            siblings.append((tj.parent.name, data.get("title") or data.get("name") or tj.parent.name))
    return current, siblings


def build_reason(current: dict, siblings: List[Tuple[str, str]]) -> str:
    lines = [
        "Stop intercepted: a Trellis task is still current (never finished).",
        "",
        f"  ● CURRENT (un-finished): {current['dir']}  [status: {current['status']}]",
        f"      {current['title']}",
        "",
        "Before stopping, do ONE of:",
        "  - finish it:   python3 ./.trellis/scripts/task.py finish",
        "  - or archive:  python3 ./.trellis/scripts/task.py archive "
        f"{Path(current['dir']).name} --no-commit",
        "  - or, if intentionally pausing mid-task, say so explicitly and stop again.",
    ]
    if siblings:
        lines.append("")
        lines.append(f"  (also {len(siblings)} other in_progress task(s) for this dev — review for staleness:)")
        for name, title in siblings[:8]:
            lines.append(f"      - {name}  —  {title[:70]}")
        if len(siblings) > 8:
            lines.append(f"      … +{len(siblings) - 8} more")
    return "\n".join(lines)


def decide(payload: dict, root: Optional[Path]) -> Optional[dict]:
    """Return the block-decision dict, or None to allow the stop."""
    if payload.get("stop_hook_active"):
        return None  # loop-safe: we already blocked once
    if root is None:
        return None  # not a Trellis project
    current, siblings = gather_dangling(root)
    if current is None:
        return None  # no un-finished current task → allow stop
    return {"decision": "block", "reason": build_reason(current, siblings)}


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return run_selftest()
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {}
    cwd = Path(payload.get("cwd") or os.getcwd())
    root = find_trellis_root(cwd)
    try:
        decision = decide(payload, root)
    except Exception:
        decision = None  # fail-open
    if decision is not None:
        print(json.dumps(decision))
    return 0


# ─── self-test ────────────────────────────────────────────────────────────────

def run_selftest() -> int:
    import tempfile

    fails: List[str] = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / ".trellis" / "tasks" / "07-09-demo").mkdir(parents=True)
        (root / ".trellis" / ".developer").write_text("name=claude\n", encoding="utf-8")
        (root / ".trellis" / "tasks" / "07-09-demo" / "task.json").write_text(
            json.dumps({"title": "demo task", "status": "in_progress", "assignee": "claude"}),
            encoding="utf-8",
        )
        curfile = root / ".trellis" / ".current-task"

        # 1. current-task set → block
        curfile.write_text(".trellis/tasks/07-09-demo\n", encoding="utf-8")
        d = decide({"stop_hook_active": False}, root)
        if not (d and d.get("decision") == "block" and "task.py finish" in d.get("reason", "")):
            fails.append(f"set current-task should BLOCK with finish hint, got {d}")

        # 2. stop_hook_active guard → allow (loop-safe)
        d = decide({"stop_hook_active": True}, root)
        if d is not None:
            fails.append(f"stop_hook_active should ALLOW (loop-safe), got {d}")

        # 3. empty current-task → allow
        curfile.write_text("", encoding="utf-8")
        d = decide({"stop_hook_active": False}, root)
        if d is not None:
            fails.append(f"empty current-task should ALLOW, got {d}")

        # 4. non-trellis root → allow
        d = decide({"stop_hook_active": False}, None)
        if d is not None:
            fails.append(f"non-trellis should ALLOW, got {d}")

    if fails:
        print("[finish-task-stop] SELF-TEST FAILED:")
        for f in fails:
            print(f"    ✗ {f}")
        return 1
    print("[finish-task-stop] SELF-TEST PASSED "
          "(block-on-current / loop-safe / allow-empty / allow-non-trellis).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
