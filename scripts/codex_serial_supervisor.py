#!/usr/bin/env python3
"""Run serial Codex worker turns and package evidence for Hermes review.

In loop mode the runner supervises the process while a persistent Hermes chat
session reviews each finished Codex run and decides whether the next worker
should receive the canonical base prompt plus a transient module-sized task
brief derived from repository evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any
from urllib import parse, request


REPO_DEFAULT_ARTIFACT_ROOT = "artifacts/tmp/hermes_codex_supervisor"
DEFAULT_BASE_PROMPT = Path(__file__).with_name("codex_serial_supervisor_prompt.md")
DEFAULT_STOP_FILE_NAME = "STOP"
DEFAULT_MANUAL_STEERING_FILE_NAME = "manual_steering.md"
DEFAULT_MANUAL_STEERING_ONCE_FILE_NAME = "manual_steering_once.md"
ACTIVE_LOOP_FILE_NAME = "active_loop.json"
HERMES_SESSION_LOCK_FILE_NAME = "hermes_session.lock"
DEFAULT_HERMES_SESSION_NAME = "TianchenRV Hermes Supervisor"
DEFAULT_SUPERVISOR_MODEL = "gpt-5.5"
DEFAULT_CODEX_REASONING_EFFORT = "xhigh"
HERMES_SESSION_ID_PATTERN = re.compile(r"\b\d{8}_\d{6}_[0-9a-f]+\b")
PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
NO_PROXY_LOCAL = ("127.0.0.1", "localhost", "::1")
HERMES_HOME = Path.home() / ".hermes"
TELEGRAM_API = "https://api.telegram.org"
CODEX_TRANSIENT_ERROR_PATTERNS = (
    "openai api error",
    "stream error",
    "connection reset",
    "timeout",
    "timed out",
    "rate limit",
    "server error",
    "temporarily unavailable",
    "model unavailable",
    "selected model is at capacity",
    "empty/aborted response",
    "transport error",
    "turn.failed",
    "error sending request",
    "connection error",
    "reconnecting",
)


def utc_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_git(repo: Path, args: list[str], timeout: int = 30) -> dict[str, Any]:
    cmd = ["git", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash snapshot
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def read_text(path: Path, max_chars: int = 20000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    except OSError as exc:
        return f"[read error: {exc}]"
    if len(data) > max_chars:
        return data[: max_chars // 2] + "\n...[truncated]...\n" + data[-max_chars // 2 :]
    return data


def tail_text(path: Path, max_chars: int = 6000) -> str:
    text = read_text(path, max_chars=max_chars)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def compact_text(text: str, max_chars: int = 2000, max_lines: int = 80) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    lines = stripped.splitlines()
    if len(lines) > max_lines:
        head_count = max(1, max_lines // 2)
        tail_count = max(1, max_lines - head_count)
        lines = [
            *lines[:head_count],
            f"...[truncated {len(stripped.splitlines()) - max_lines} lines]...",
            *lines[-tail_count:],
        ]
    compacted = "\n".join(lines)
    if len(compacted) > max_chars:
        return (
            compacted[: max_chars // 2]
            + "\n...[truncated]...\n"
            + compacted[-max_chars // 2 :]
        )
    return compacted


def summarize_previous_prompt(text: str, max_chars: int = 1200) -> str:
    stripped = text.strip()
    if not stripped:
        return "(none)"
    if len(stripped) <= max_chars:
        return stripped
    wanted_prefixes = (
        "Task title",
        "Direction title",
        "Module owner",
        "Why now",
        "Module deliverable",
        "What capability",
        "If unfinished",
        "Final report",
    )
    lines = stripped.splitlines()
    selected: list[str] = [f"[previous prompt summarized from {len(stripped)} chars]"]
    for index, line in enumerate(lines):
        if line.strip().startswith(wanted_prefixes):
            selected.append(line.strip())
            for follow in lines[index + 1 : index + 4]:
                if follow.strip():
                    selected.append(f"  {follow.strip()}")
                    break
    if len(selected) == 1:
        selected.extend(lines[:20])
    return compact_text("\n".join(selected), max_chars=max_chars, max_lines=40)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _snapshot_git_field(snapshot: Any, key: str) -> str:
    if not isinstance(snapshot, dict):
        return ""
    git = snapshot.get("git")
    if not isinstance(git, dict):
        return ""
    item = git.get(key)
    if not isinstance(item, dict):
        return ""
    return str(item.get("stdout") or "").strip()


def codex_transient_failure_reason(run_dir: Path) -> str:
    """Return a short reason when a failed Codex run looks transient."""
    manifest = load_json(run_dir / "manifest.json")
    if not isinstance(manifest, dict):
        return ""
    code = manifest.get("codex_exit_code")
    if code in (None, 0, "0"):
        return ""
    evidence = "\n".join(
        [
            str(manifest.get("runner_error") or ""),
            read_text(run_dir / "codex.stderr.log", max_chars=40000),
            read_text(run_dir / "codex.stdout.jsonl", max_chars=80000),
            read_text(run_dir / "last_message.md", max_chars=12000),
        ]
    ).lower()
    for pattern in CODEX_TRANSIENT_ERROR_PATTERNS:
        if pattern in evidence:
            return f"codex_transient:{pattern}"
    return ""


def codex_retry_blocker(run_dir: Path) -> str:
    """Return why a failed transient worker must not be retried.

    Repository mutations are not blockers for transient retry. They mean the
    retry must continue from the live repo state and previous run artifacts
    rather than replaying the original task from scratch.
    """
    return ""


def codex_retry_needs_continuation(run_dir: Path) -> bool:
    """Return True when retry should continue dirty work left by a failed worker."""
    before = load_json(run_dir / "snapshot_before.json")
    after = load_json(run_dir / "snapshot_after.json")
    before_head = _snapshot_git_field(before, "head")
    after_head = _snapshot_git_field(after, "head")
    before_status = _snapshot_git_field(before, "status_short")
    after_status = _snapshot_git_field(after, "status_short")
    return before_head != after_head or before_status != after_status


def build_codex_continuation_retry_prompt(
    run_dir: Path,
    retry_reason: str,
    delta: str,
    prompt_override: str,
) -> str:
    """Build a worker brief for a retry over an already-mutated worktree."""
    before = load_json(run_dir / "snapshot_before.json")
    after = load_json(run_dir / "snapshot_after.json")
    before_head = _snapshot_git_field(before, "head") or "(unknown)"
    after_head = _snapshot_git_field(after, "head") or "(unknown)"
    before_status = _snapshot_git_field(before, "status_short") or "(clean)"
    after_status = _snapshot_git_field(after, "status_short") or "(clean)"
    original_brief = prompt_override.strip()
    if not original_brief and delta.strip():
        original_brief = "Original legacy Hermes delta:\n\n" + delta.strip()

    continuation = f"""## Codex Transient Continuation Retry

The previous Codex worker for this same supervisor round failed because of a
transient model/API/stream problem: `{retry_reason}`.

This is a continuation retry, not a new task selection and not a reset. The
previous attempt may already have changed the worktree. Do not discard,
overwrite, or revert that work unless repository evidence proves it is unsafe.

Previous run artifacts:

```text
run_dir: {run_dir}
manifest: {run_dir / "manifest.json"}
stdout: {run_dir / "codex.stdout.jsonl"}
stderr: {run_dir / "codex.stderr.log"}
last_message: {run_dir / "last_message.md"}
review_input: {run_dir / "review_input.md"}
snapshot_before: {run_dir / "snapshot_before.json"}
snapshot_after: {run_dir / "snapshot_after.json"}
```

Git state changed during the failed attempt:

```text
before_head: {before_head}
after_head:  {after_head}

before:
{before_status}

after:
{after_status}
```

Required continuation behavior:

1. Inspect the live worktree, the active Trellis task if present, and the
   previous run artifacts above.
2. Continue the same task/round from the existing dirty state.
3. If the dirty changes are coherent, validate, self-repair as needed,
   finish/archive the task, and create one coherent commit.
4. If the previous attempt already committed and archived the task cleanly,
   verify that state from the live repo and previous artifacts, then report the
   completed state without creating a new unrelated task or duplicate commit.
5. If the dirty changes are unsafe or incomplete beyond a bounded repair, keep
   the task open with the exact continuation point and do not pretend it is
   finished.
6. Do not create an unrelated Trellis task and do not fall back to a broad
   base-prompt-only direction.
"""

    if original_brief:
        return original_brief + "\n\n---\n\n" + continuation
    return continuation


def infer_round_index_from_run_dir(run_dir: Path) -> int:
    match = re.search(r"-r(\d{4})-", run_dir.name)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0


def write_stop_request(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "requested_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pid": os.getpid(),
        "reason": reason,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def resolve_task_dir(repo: Path, ref: str) -> Path:
    p = Path(ref.strip())
    return p if p.is_absolute() else repo / p


def current_task(repo: Path) -> dict[str, Any]:
    marker = repo / ".trellis" / ".current-task"
    ref = read_text(marker, max_chars=2000).strip()
    if not ref:
        return {"ref": "", "exists": False}
    task_dir = resolve_task_dir(repo, ref)
    task_json = load_json(task_dir / "task.json")
    prd_path = task_dir / "prd.md"
    if not prd_path.exists():
        prd_path = task_dir / "PRD.md"
    return {
        "ref": ref,
        "exists": task_dir.exists(),
        "path": str(task_dir),
        "task_json": task_json,
        "prd_head": read_text(prd_path, max_chars=12000),
        "implement_jsonl": read_text(task_dir / "implement.jsonl", max_chars=8000),
        "check_jsonl": read_text(task_dir / "check.jsonl", max_chars=8000),
    }


def active_tasks(repo: Path) -> list[dict[str, Any]]:
    tasks_root = repo / ".trellis" / "tasks"
    rows: list[dict[str, Any]] = []
    if not tasks_root.exists():
        return rows
    for task_json_path in sorted(tasks_root.glob("*/task.json")):
        data = load_json(task_json_path)
        if isinstance(data, dict) and data.get("status") not in (None, "completed"):
            rows.append(
                {
                    "path": str(task_json_path.parent.relative_to(repo)),
                    "id": data.get("id"),
                    "title": data.get("title"),
                    "status": data.get("status"),
                    "scope": data.get("scope"),
                    "commit": data.get("commit"),
                }
            )
    return rows


def collect_snapshot(repo: Path) -> dict[str, Any]:
    workspace_index = repo / ".trellis" / "workspace" / "index.md"
    head = run_git(repo, ["rev-parse", "HEAD"])
    return {
        "repo": str(repo),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": {
            "root": run_git(repo, ["rev-parse", "--show-toplevel"]),
            "branch": run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"]),
            "head": head,
            "status_short": run_git(repo, ["status", "--short"]),
            "recent_log": run_git(repo, ["log", "--oneline", "-12"]),
        },
        "trellis": {
            "current_task": current_task(repo),
            "active_tasks": active_tasks(repo),
            "workspace_index": read_text(workspace_index, max_chars=8000),
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(data, ensure_ascii=False) + "\n")


def artifact_root_path(repo: Path, value: str) -> Path:
    root = Path(value)
    if not root.is_absolute():
        root = repo / root
    return root


def stop_file_path(repo: Path, args: argparse.Namespace) -> Path:
    if getattr(args, "stop_file", ""):
        path = Path(args.stop_file)
        return path if path.is_absolute() else repo / path
    return artifact_root_path(repo, args.artifact_root) / DEFAULT_STOP_FILE_NAME


def manual_steering_path(repo: Path, args: argparse.Namespace) -> Path:
    value = getattr(args, "manual_steering_file", "")
    if value:
        path = Path(value)
        return path if path.is_absolute() else repo / path
    return artifact_root_path(repo, args.artifact_root) / DEFAULT_MANUAL_STEERING_FILE_NAME


def manual_steering_once_path(repo: Path, args: argparse.Namespace) -> Path:
    value = getattr(args, "manual_steering_once_file", "")
    if value:
        path = Path(value)
        return path if path.is_absolute() else repo / path
    return artifact_root_path(repo, args.artifact_root) / DEFAULT_MANUAL_STEERING_ONCE_FILE_NAME


def read_manual_steering(repo: Path, args: argparse.Namespace, max_chars: int = 12000) -> str:
    return read_text(manual_steering_path(repo, args), max_chars=max_chars).strip()


def read_review_steering(repo: Path, args: argparse.Namespace, max_chars: int = 12000) -> str:
    durable_path = manual_steering_path(repo, args)
    once_path = manual_steering_once_path(repo, args)
    per_file_chars = max(2000, max_chars // 2)
    durable = read_text(durable_path, max_chars=per_file_chars).strip()
    once = ""
    if once_path.resolve() != durable_path.resolve():
        once = read_text(once_path, max_chars=per_file_chars).strip()

    sections: list[str] = []
    if durable:
        sections.append(
            "## Durable Manual Steering\n"
            f"path: {durable_path}\n\n"
            f"{durable}"
        )
    if once:
        sections.append(
            "## One-Shot Manual Steering\n"
            f"path: {once_path}\n"
            "This one-shot steering is consumed only after an official Hermes review returns parseable strict JSON.\n\n"
            f"{once}"
        )
    return "\n\n".join(sections)


def consume_manual_steering_once(
    repo: Path,
    args: argparse.Namespace,
    loop_dir: Path,
    round_index: int,
    review_prompt_path: Path,
) -> dict[str, Any]:
    once_path = manual_steering_once_path(repo, args)
    durable_path = manual_steering_path(repo, args)
    result: dict[str, Any] = {
        "path": str(once_path),
        "exists": once_path.exists(),
        "consumed": False,
        "archived_to": "",
        "review_prompt": str(review_prompt_path),
    }
    if once_path.resolve() == durable_path.resolve():
        result["reason"] = "one-shot path matches durable manual steering path; not consuming durable steering"
        return result

    text = read_text(once_path, max_chars=2_000_000).strip()
    if not text:
        result["reason"] = "empty_or_missing"
        return result

    archive_dir = loop_dir / "consumed_manual_steering"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"round_{round_index:04d}_{once_path.name}"
    archive_path.write_text(text + "\n", encoding="utf-8")
    try:
        once_path.unlink()
        result["cleared_by"] = "unlink"
    except FileNotFoundError:
        result["cleared_by"] = "already_missing"
    except OSError as exc:
        try:
            once_path.write_text("", encoding="utf-8")
            result["cleared_by"] = "truncate"
            result["clear_warning"] = str(exc)
        except OSError as second_exc:
            result["cleared_by"] = "failed"
            result["clear_error"] = f"{exc}; truncate failed: {second_exc}"
    result["consumed"] = True
    result["archived_to"] = str(archive_path)
    result["chars"] = len(text)
    return result


def latest_saved_hermes_session(root: Path) -> dict[str, Any]:
    active_loop = load_json(root / ACTIVE_LOOP_FILE_NAME)
    if isinstance(active_loop, dict) and active_loop.get("hermes_session_id"):
        return {
            "session_id": str(active_loop.get("hermes_session_id") or ""),
            "source": str(root / ACTIVE_LOOP_FILE_NAME),
        }

    loops_root = root / "loops"
    if not loops_root.exists():
        return {"session_id": "", "source": ""}
    loop_dirs = [p for p in loops_root.iterdir() if p.is_dir()]
    loop_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for loop_dir in loop_dirs:
        for name in ("hermes_session.json", "loop_manifest.json"):
            data = load_json(loop_dir / name)
            if isinstance(data, dict) and data.get("hermes_session_id"):
                return {
                    "session_id": str(data.get("hermes_session_id") or ""),
                    "source": str(loop_dir / name),
                }
    return {"session_id": "", "source": ""}


def resolve_hermes_session_id(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    explicit = str(getattr(args, "hermes_session_id", "") or "")
    if explicit:
        return {"session_id": explicit, "source": "explicit"}
    if not getattr(args, "resume_latest_hermes_session", False):
        return {"session_id": "", "source": "disabled"}
    found = latest_saved_hermes_session(artifact_root_path(repo, args.artifact_root))
    session_id = str(found.get("session_id") or "")
    if session_id:
        args.hermes_session_id = session_id
    return found


def read_delta(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.delta:
        parts.append(args.delta.strip())
    if args.delta_file:
        parts.append(read_text(Path(args.delta_file), max_chars=30000).strip())
    return "\n\n".join(p for p in parts if p)


def read_prompt_override(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if getattr(args, "prompt_override", ""):
        parts.append(args.prompt_override.strip())
    if getattr(args, "prompt_override_file", ""):
        parts.append(read_text(Path(args.prompt_override_file), max_chars=200000).strip())
    return "\n\n".join(p for p in parts if p)


def read_initial_delta(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.initial_delta:
        parts.append(args.initial_delta.strip())
    if args.initial_delta_file:
        parts.append(read_text(Path(args.initial_delta_file), max_chars=30000).strip())
    return "\n\n".join(p for p in parts if p)


def build_prompt(base_prompt: Path, delta: str, prompt_override: str = "") -> str:
    base = read_text(base_prompt, max_chars=200000).strip()
    if not base:
        raise SystemExit(f"base prompt not found or empty: {base_prompt}")
    if prompt_override.strip():
        return (
            base
            + "\n\n---\n\n"
            + "## Current Task Brief\n\n"
            + prompt_override.strip()
            + "\n"
        )
    if not delta.strip():
        return base + "\n"
    return (
        base
        + "\n\n---\n\n"
        + "## Legacy Hermes Delta\n\n"
        + delta.strip()
        + "\n"
    )


def apply_prompt_edits(base_prompt_text: str, edits: Any) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(edits, list):
        return base_prompt_text, []
    current = base_prompt_text
    results: list[dict[str, Any]] = []
    for index, edit in enumerate(edits, start=1):
        if not isinstance(edit, dict):
            results.append({"index": index, "applied": False, "reason": "edit is not an object"})
            continue
        text = str(edit.get("text") or edit.get("replace") or "").strip()
        if "find" in edit:
            find = str(edit.get("find") or "")
            replace = str(edit.get("replace") or "")
            count = current.count(find) if find else 0
            if count != 1:
                results.append({"index": index, "applied": False, "reason": f"find matched {count} times"})
                continue
            current = current.replace(find, replace, 1)
            results.append({"index": index, "applied": True, "operation": "replace"})
            continue
        if "insert_after" in edit:
            marker = str(edit.get("insert_after") or "")
            count = current.count(marker) if marker else 0
            if count != 1 or not text:
                results.append({"index": index, "applied": False, "reason": f"insert_after matched {count} times"})
                continue
            current = current.replace(marker, marker.rstrip() + "\n\n" + text, 1)
            results.append({"index": index, "applied": True, "operation": "insert_after"})
            continue
        if "insert_before" in edit:
            marker = str(edit.get("insert_before") or "")
            count = current.count(marker) if marker else 0
            if count != 1 or not text:
                results.append({"index": index, "applied": False, "reason": f"insert_before matched {count} times"})
                continue
            current = current.replace(marker, text + "\n\n" + marker, 1)
            results.append({"index": index, "applied": True, "operation": "insert_before"})
            continue
        results.append({"index": index, "applied": False, "reason": "unsupported edit operation"})
    return current, results


def sanitized_env(keep_proxy: bool) -> dict[str, str]:
    env = dict(os.environ)
    if not keep_proxy:
        for key in PROXY_KEYS:
            env.pop(key, None)
    existing = env.get("NO_PROXY") or env.get("no_proxy") or ""
    values = [v.strip() for v in existing.split(",") if v.strip()]
    for local in NO_PROXY_LOCAL:
        if local not in values:
            values.append(local)
    env["NO_PROXY"] = ",".join(values)
    env["no_proxy"] = env["NO_PROXY"]
    return env


def telegram_settings(args: argparse.Namespace) -> dict[str, str]:
    dotenv = load_dotenv(HERMES_HOME / ".env")
    cfg = load_yaml_config(HERMES_HOME / "config.yaml")
    telegram_cfg = cfg.get("telegram") if isinstance(cfg.get("telegram"), dict) else {}

    token = (
        args.telegram_bot_token
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or dotenv.get("TELEGRAM_BOT_TOKEN")
        or telegram_cfg.get("bot_token", "")
        or telegram_cfg.get("token", "")
    )
    chat_id = (
        args.telegram_chat_id
        or os.getenv("TELEGRAM_CHAT_ID")
        or os.getenv("TELEGRAM_HOME_CHANNEL")
        or dotenv.get("TELEGRAM_CHAT_ID")
        or dotenv.get("TELEGRAM_HOME_CHANNEL")
        or telegram_cfg.get("home_channel", "")
        or telegram_cfg.get("home", "")
        or telegram_cfg.get("chat_id", "")
    )
    proxy = (
        args.telegram_proxy
        or os.getenv("TELEGRAM_PROXY")
        or dotenv.get("TELEGRAM_PROXY")
        or telegram_cfg.get("proxy_url", "")
    )
    return {
        "token": str(token or ""),
        "chat_id": str(chat_id or ""),
        "proxy": str(proxy or ""),
    }


def short_sha(value: str) -> str:
    value = (value or "").strip()
    return value[:8] if len(value) >= 8 else value


def git_stdout(snapshot: dict[str, Any], key: str) -> str:
    return str(((snapshot.get("git") or {}).get(key) or {}).get("stdout") or "")


def task_ref_and_status(snapshot: dict[str, Any]) -> str:
    task = (snapshot.get("trellis") or {}).get("current_task") or {}
    ref = str(task.get("ref") or "")
    status = task_status(snapshot)
    if ref and status:
        return f"{ref} ({status})"
    return ref or "(none)"


def telegram_message(
    phase: str,
    run_dir: Path,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    code: int | None = None,
) -> str:
    repo = Path((after or before or {}).get("repo", "")).name or "repo"
    lines = [
        f"Codex supervisor {phase}",
        f"repo: {repo}",
        f"run: {run_dir.name}",
    ]
    if before:
        lines.append(f"before: {short_sha(git_stdout(before, 'head'))}")
        lines.append(f"task: {task_ref_and_status(before)}")
    if after:
        before_head = short_sha(git_stdout(before or {}, "head"))
        after_head = short_sha(git_stdout(after, "head"))
        if after_head and after_head != before_head:
            lines.append(f"after: {after_head}")
        status = git_stdout(after, "status_short")
        lines.append(f"worktree: {'dirty' if status else 'clean'}")
    if code is not None:
        lines.append(f"codex_exit_code: {code}")
    elif after is not None:
        lines.append("codex_exit_code: not_run")
    lines.append(f"review: {run_dir / 'review_input.md'}")
    return "\n".join(lines)[:3800]


def write_telegram_result(run_dir: Path, phase: str, result: dict[str, Any]) -> None:
    safe = {k: v for k, v in result.items() if k != "token"}
    write_json(run_dir / f"telegram_{phase}.json", safe)


def send_telegram(
    args: argparse.Namespace,
    run_dir: Path,
    phase: str,
    text: str,
) -> dict[str, Any]:
    if args.telegram_event == "off" or args.no_telegram:
        result = {"status": "skipped", "reason": "disabled"}
        write_telegram_result(run_dir, phase, result)
        return result
    settings = telegram_settings(args)
    token = settings["token"]
    chat_id = settings["chat_id"]
    if not token or not chat_id:
        result = {"status": "skipped", "reason": "missing TELEGRAM_BOT_TOKEN or TELEGRAM_HOME_CHANNEL/TELEGRAM_CHAT_ID"}
        write_telegram_result(run_dir, phase, result)
        return result

    payload = parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    req = request.Request(
        f"{TELEGRAM_API}/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    opener = request.build_opener()
    if settings["proxy"]:
        opener = request.build_opener(
            request.ProxyHandler({"http": settings["proxy"], "https": settings["proxy"]})
        )
    try:
        with opener.open(req, timeout=args.telegram_timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        result = {
            "status": "sent" if parsed.get("ok") else "failed",
            "ok": bool(parsed.get("ok")),
            "chat_id": chat_id,
            "proxy": settings["proxy"],
            "response": parsed,
        }
    except Exception as exc:  # noqa: BLE001 - notification failure must not fail runner
        result = {
            "status": "failed",
            "ok": False,
            "chat_id": chat_id,
            "proxy": settings["proxy"],
            "error": str(exc),
        }
    write_telegram_result(run_dir, phase, result)
    return result


class FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"another codex supervisor run is active: {self.path}") from exc
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


class BlockingFileLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self) -> "BlockingFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def stream_pipe(pipe: Any, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", errors="replace") as out:
        while True:
            chunk = pipe.readline()
            if not chunk:
                break
            out.write(chunk)
            out.flush()


def run_codex(args: argparse.Namespace, repo: Path, run_dir: Path, prompt: str) -> int:
    stdout_path = run_dir / "codex.stdout.jsonl"
    stderr_path = run_dir / "codex.stderr.log"
    last_message_path = run_dir / "last_message.md"
    cmd = [
        args.codex_bin,
        "exec",
        "--json",
        "-o",
        str(last_message_path),
        "-C",
        str(repo),
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    cmd.extend(["--disable", "multi_agent"])
    if args.model:
        cmd.extend(["--model", args.model])
    if DEFAULT_CODEX_REASONING_EFFORT:
        cmd.extend(["-c", f'model_reasoning_effort="{DEFAULT_CODEX_REASONING_EFFORT}"'])
    if args.profile:
        cmd.extend(["--profile", args.profile])
    for item in args.codex_arg or []:
        cmd.append(item)
    cmd.append("-")

    (run_dir / "codex.command.json").write_text(
        json.dumps(cmd, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        cmd,
        cwd=repo,
        env=sanitized_env(args.keep_proxy),
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    out_thread = threading.Thread(target=stream_pipe, args=(proc.stdout, stdout_path), daemon=True)
    err_thread = threading.Thread(target=stream_pipe, args=(proc.stderr, stderr_path), daemon=True)
    out_thread.start()
    err_thread.start()
    proc.stdin.write(prompt)
    proc.stdin.close()
    code = proc.wait()
    out_thread.join()
    err_thread.join()
    (run_dir / "exit_code").write_text(str(code) + "\n", encoding="utf-8")
    return int(code)


def git_range(repo: Path, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_head = ((before.get("git") or {}).get("head") or {}).get("stdout") or ""
    after_head = ((after.get("git") or {}).get("head") or {}).get("stdout") or ""
    if not before_head or not after_head or before_head == after_head:
        return {"before": before_head, "after": after_head, "new_commits": "", "changed_files": ""}
    rev = f"{before_head}..{after_head}"
    return {
        "before": before_head,
        "after": after_head,
        "new_commits": run_git(repo, ["log", "--oneline", rev]),
        "changed_files": run_git(repo, ["diff", "--name-only", rev]),
        "stat": run_git(repo, ["diff", "--stat", rev]),
    }


def run_shell(repo: Path, script: str, timeout: int = 30) -> dict[str, Any]:
    cmd = ["bash", "-lc", script]
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - audit should degrade to evidence
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def command_block(title: str, result: dict[str, Any], max_chars: int = 12000) -> list[str]:
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    body = stdout
    if stderr:
        body = (body + "\n" if body else "") + "[stderr]\n" + stderr
    if len(body) > max_chars:
        body = body[: max_chars // 2] + "\n...[truncated]...\n" + body[-max_chars // 2 :]
    return [
        f"### {title}",
        "",
        f"- returncode: `{result.get('returncode')}`",
        "",
        "```",
        body,
        "```",
        "",
    ]


def write_repo_audit(run_dir: Path, before: dict[str, Any], after: dict[str, Any]) -> Path:
    repo = Path(before["repo"])
    diff = git_range(repo, before, after)
    before_head = str(diff.get("before") or "")
    after_head = str(diff.get("after") or "")
    commands: tuple[tuple[str, dict[str, Any], int], ...] = (
        ("git status --short", run_git(repo, ["status", "--short"]), 6000),
        ("recent git log", run_git(repo, ["log", "--oneline", "-12"]), 6000),
        ("new commits", diff.get("new_commits") if isinstance(diff.get("new_commits"), dict) else {}, 6000),
        ("changed files in new commits", diff.get("changed_files") if isinstance(diff.get("changed_files"), dict) else {}, 9000),
        ("diff stat in new commits", diff.get("stat") if isinstance(diff.get("stat"), dict) else {}, 9000),
        ("last commit name-status/stat", run_git(repo, ["show", "--name-status", "--stat", "--oneline", "--decorate", "--no-renames", "HEAD"]), 16000),
        ("tracked project files", run_shell(repo, "git ls-files | sort | sed -n '1,500p'"), 20000),
        ("top-level tree", run_shell(repo, "find . -maxdepth 3 -type f -not -path './.git/*' -not -path './artifacts/tmp/*' | sort | sed -n '1,300p'"), 16000),
        ("TianchenRV specs", run_shell(repo, "find .trellis/spec -maxdepth 3 -type f | sort | sed -n '1,240p'"), 12000),
        ("source package directories", run_shell(repo, "find . -maxdepth 3 -type d \\( -name src -o -name include -o -name lib -o -name python -o -name tianchenrv -o -name tests -o -name examples -o -name tools \\) | sort"), 12000),
        ("MLIR and build tool detection", run_shell(repo, "printf 'local:\\n'; for t in mlir-opt mlir-tblgen clang clang++ cmake ninja python3; do printf '%s=' \"$t\"; command -v \"$t\" || true; done"), 12000),
        ("rvv remote quick probe", run_shell(repo, "ssh -o BatchMode=yes -o ConnectTimeout=6 rvv 'hostname; uname -m; nproc; command -v clang || true; command -v mlir-opt || true; command -v cmake || true; sudo -n true && echo sudo_nopass_ok || echo sudo_needs_password'"), 12000),
    )
    lines = [
        "# Hermes Repo Audit",
        "",
        "This is a read-only audit generated by the supervisor runner from the real repo checkout.",
        "",
        "## Scope",
        "",
        f"- repo: `{repo}`",
        f"- run_dir: `{run_dir}`",
        f"- before_head: `{before_head}`",
        f"- after_head: `{after_head}`",
        "",
        "Hermes should treat this file as stronger evidence than the Codex final message when the two disagree.",
        "",
    ]
    for title, result, max_chars in commands:
        lines.extend(command_block(title, result, max_chars=max_chars))
    path = run_dir / "repo_audit.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def task_status(snapshot: dict[str, Any]) -> str:
    task = ((snapshot.get("trellis") or {}).get("current_task") or {}).get("task_json")
    if isinstance(task, dict):
        return str(task.get("status") or "")
    return ""


def write_review_input(run_dir: Path, before: dict[str, Any], after: dict[str, Any], code: int | None, delta: str) -> None:
    repo = Path(before["repo"])
    diff = git_range(repo, before, after)
    before_task = ((before.get("trellis") or {}).get("current_task") or {})
    after_task = ((after.get("trellis") or {}).get("current_task") or {})
    lines = [
        "# Hermes Supervisor Review Input",
        "",
        "This file is for Hermes. Do not treat it as the Codex final report.",
        "",
        "## Run",
        "",
        f"- run_dir: `{run_dir}`",
        f"- codex_exit_code: `{code}`",
        f"- before_head: `{diff.get('before', '')}`",
        f"- after_head: `{diff.get('after', '')}`",
        f"- before_current_task: `{before_task.get('ref', '')}` status `{task_status(before)}`",
        f"- after_current_task: `{after_task.get('ref', '')}` status `{task_status(after)}`",
        "",
        "## Supervisor Delta Used",
        "",
        delta.strip() or "(none)",
        "",
        "## New Commits",
        "",
        (diff.get("new_commits") or {}).get("stdout", "") if isinstance(diff.get("new_commits"), dict) else "",
        "",
        "## Changed Files In New Commits",
        "",
        (diff.get("changed_files") or {}).get("stdout", "") if isinstance(diff.get("changed_files"), dict) else "",
        "",
        "## Git Status After",
        "",
        ((after.get("git") or {}).get("status_short") or {}).get("stdout", ""),
        "",
        "## Codex Last Message Tail",
        "",
        tail_text(run_dir / "last_message.md", max_chars=6000),
        "",
        "## stderr Tail",
        "",
        tail_text(run_dir / "codex.stderr.log", max_chars=4000),
        "",
        "## Hermes Review Checklist",
        "",
        "Review only these supervision questions:",
        "",
        "- Did Codex stay on the TianchenRV capability-driven MLIR execution-layer mainline and respect the red lines?",
        "- Was the milestone right-bigsize, or did it shrink into metadata/test/status/report-only work?",
        "- If it used TianchenRV Trellis, did it finish/archive the current local task instead of leaving stale completed tasks at `.trellis/tasks/` top level?",
        "- Did it produce active code/schema/build/RVV evidence, not just docs or scaffolding labels?",
        "- Did it keep `tcrv.exec` compute-free and extension-specific behavior plugin-local?",
        "- Did any RVV correctness/performance claim rely on real `ssh rvv` evidence?",
        "- Did it preserve parameter layering: hardware facts / target capability, compile-time variant config, runtime SSA/control values, and descriptor-local boundaries?",
        "- If continuing, Hermes must generate a focused Direction Brief from current evidence; the runner prepends the base prompt.",
    ]
    (run_dir / "review_input.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def command_snapshot(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    snapshot = collect_snapshot(repo)
    if args.output == "-":
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    else:
        write_json(Path(args.output), snapshot)
        print(args.output)
    return 0


def command_prompt(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    run_dir = make_run_dir(repo, args)
    delta = read_delta(args)
    prompt_override = read_prompt_override(args)
    prompt = build_prompt(Path(args.base_prompt), delta, prompt_override=prompt_override)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    write_json(
        run_dir / "prompt_source.json",
        {
            "prompt_source": "override" if prompt_override.strip() else "delta" if delta.strip() else "base",
            "delta_chars": len(delta),
            "prompt_override_chars": len(prompt_override),
        },
    )
    print(run_dir / "prompt.md")
    return 0


def make_run_dir(repo: Path, args: argparse.Namespace) -> Path:
    return artifact_root_path(repo, args.artifact_root) / "runs" / (args.run_id or utc_run_id())


def command_run(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    run_dir = make_run_dir(repo, args)
    lock_path = artifact_root_path(repo, args.artifact_root) / "codex_serial_supervisor.lock"

    with FileLock(lock_path):
        run_dir.mkdir(parents=True, exist_ok=False)
        delta = read_delta(args)
        prompt_override = read_prompt_override(args)
        prompt_source = "override" if prompt_override.strip() else "delta" if delta.strip() else "base"
        prompt = build_prompt(Path(args.base_prompt), delta, prompt_override=prompt_override)
        (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        write_json(
            run_dir / "prompt_source.json",
            {
                "prompt_source": prompt_source,
                "delta_chars": len(delta),
                "prompt_override_chars": len(prompt_override),
                "base_prompt": str(Path(args.base_prompt)),
            },
        )
        before = collect_snapshot(repo)
        write_json(run_dir / "snapshot_before.json", before)
        if args.telegram_event in ("start", "both"):
            send_telegram(
                args,
                run_dir,
                "start",
                telegram_message("started", run_dir, before=before),
            )

        code: int | None
        runner_error = ""
        if args.no_exec:
            code = None
            (run_dir / "exit_code").write_text("not_run\n", encoding="utf-8")
        else:
            try:
                code = run_codex(args, repo, run_dir, prompt)
            except Exception as exc:  # noqa: BLE001 - still package and notify
                code = -1
                runner_error = f"{type(exc).__name__}: {exc}"
                (run_dir / "exit_code").write_text(str(code) + "\n", encoding="utf-8")
                with (run_dir / "codex.stderr.log").open("a", encoding="utf-8") as fh:
                    fh.write(f"\n[runner_error] {runner_error}\n")

        after = collect_snapshot(repo)
        write_json(run_dir / "snapshot_after.json", after)
        repo_audit_path = write_repo_audit(run_dir, before, after)
        write_review_input(run_dir, before, after, code, delta)
        manifest = {
            "run_dir": str(run_dir),
            "repo": str(repo),
            "codex_exit_code": code,
            "prompt": str(run_dir / "prompt.md"),
            "review_input": str(run_dir / "review_input.md"),
            "snapshot_before": str(run_dir / "snapshot_before.json"),
            "snapshot_after": str(run_dir / "snapshot_after.json"),
            "stdout": str(run_dir / "codex.stdout.jsonl"),
            "stderr": str(run_dir / "codex.stderr.log"),
            "last_message": str(run_dir / "last_message.md"),
            "repo_audit": str(repo_audit_path),
            "prompt_source": prompt_source,
        }
        if runner_error:
            manifest["runner_error"] = runner_error
        write_json(run_dir / "manifest.json", manifest)
        if args.telegram_event in ("finish", "both"):
            send_telegram(
                args,
                run_dir,
                "finish",
                telegram_message("finished", run_dir, before=before, after=after, code=code),
            )
        print(run_dir)
    return 1 if runner_error else 0


def latest_run_dirs(root: Path, limit: int = 8) -> list[dict[str, Any]]:
    runs_root = root / "runs"
    if not runs_root.exists():
        return []
    dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    rows: list[dict[str, Any]] = []
    for path in dirs[:limit]:
        manifest = load_json(path / "manifest.json")
        exit_code = read_text(path / "exit_code", max_chars=100).strip()
        rows.append(
            {
                "run_id": path.name,
                "path": str(path),
                "modified": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat(),
                "exit_code": exit_code,
                "manifest": manifest if isinstance(manifest, dict) else None,
                "review_input": str(path / "review_input.md"),
                "last_message": str(path / "last_message.md"),
            }
        )
    return rows


def latest_loop_dirs(root: Path, limit: int = 8) -> list[dict[str, Any]]:
    loops_root = root / "loops"
    if not loops_root.exists():
        return []
    dirs = [p for p in loops_root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    rows: list[dict[str, Any]] = []
    for path in dirs[:limit]:
        manifest = load_json(path / "loop_manifest.json")
        result = load_json(path / "loop_result.json")
        rows.append(
            {
                "loop_id": path.name,
                "path": str(path),
                "modified": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat(),
                "manifest": manifest if isinstance(manifest, dict) else None,
                "result": result if isinstance(result, dict) else None,
                "events": str(path / "events.jsonl"),
            }
        )
    return rows


def pid_is_alive(text: str) -> bool | None:
    try:
        pid = int(text.strip())
    except ValueError:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [fence.group(1)] if fence else []

    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
        start = text.find("{", start + 1)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def bool_from_review(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "stop"}
    if value is None:
        return True
    return bool(value)


def normalize_review(raw_text: str) -> dict[str, Any]:
    data = parse_json_object(raw_text)
    if not data:
        return {
            "continue": True,
            "delta": "",
            "next_prompt": "",
            "base_prompt_edits": [],
            "reason": "Hermes review output was not valid JSON; continuing is blocked until a non-empty next_prompt Direction Brief is available.",
            "telegram_note": "",
            "raw": raw_text[:4000],
            "parse_error": True,
        }
    base_prompt_edits = data.get("base_prompt_edits")
    if not isinstance(base_prompt_edits, list):
        base_prompt_edits = []
    return {
        "continue": bool_from_review(data.get("continue", True)),
        "delta": str(data.get("delta") or "").strip(),
        "next_prompt": str(data.get("next_prompt") or "").strip(),
        "base_prompt_edits": base_prompt_edits,
        "reason": str(data.get("reason") or "").strip(),
        "telegram_note": str(data.get("telegram_note") or "").strip(),
        "raw": raw_text[:4000],
        "parse_error": False,
    }


def extract_hermes_session_id(raw_text: str) -> str:
    match = re.search(r"(?m)^session_id:\s*(\S+)\s*$", raw_text)
    return match.group(1).strip() if match else ""


def latest_hermes_session_id(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    cmd = [args.hermes_bin, "sessions", "list", "--source", args.hermes_source, "--limit", "1"]
    proc = subprocess.run(
        cmd,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    matches = HERMES_SESSION_ID_PATTERN.findall(proc.stdout)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "session_id": matches[-1] if matches else "",
    }


def build_review_prompt(
    run_dir: Path,
    loop_dir: Path,
    base_prompt: Path,
    round_index: int,
    previous_delta: str,
    previous_prompt_mode: str,
    max_chars: int,
    manual_steering: str = "",
) -> str:
    del max_chars  # The review prompt now carries evidence entry points, not bulk evidence.
    before = load_json(run_dir / "snapshot_before.json")
    after = load_json(run_dir / "snapshot_after.json")
    if not isinstance(before, dict):
        before = {}
    if not isinstance(after, dict):
        after = {}
    repo = Path(str(after.get("repo") or before.get("repo") or "/home/kingdom/phdworks/TianchenRV"))
    manifest = load_json(run_dir / "manifest.json")
    if not isinstance(manifest, dict):
        manifest = {}
    diff = git_range(repo, before, after)
    changed_files_result = diff.get("changed_files") if isinstance(diff.get("changed_files"), dict) else {}
    stat_result = diff.get("stat") if isinstance(diff.get("stat"), dict) else {}
    new_commits_result = diff.get("new_commits") if isinstance(diff.get("new_commits"), dict) else {}
    changed_files = compact_text(str(changed_files_result.get("stdout") or ""), max_chars=1800, max_lines=60)
    diff_stat = compact_text(str(stat_result.get("stdout") or ""), max_chars=1800, max_lines=60)
    new_commits = compact_text(str(new_commits_result.get("stdout") or ""), max_chars=1400, max_lines=40)
    git_status = compact_text(str((run_git(repo, ["status", "--short"]).get("stdout") or "")), max_chars=1000, max_lines=40)
    latest_commit = compact_text(
        str(
            run_git(
                repo,
                ["show", "--name-status", "--stat", "--oneline", "--decorate", "--no-renames", "HEAD"],
            ).get("stdout")
            or ""
        ),
        max_chars=2400,
        max_lines=80,
    )
    after_task = ((after.get("trellis") or {}).get("current_task") or {})
    after_task_json = after_task.get("task_json") if isinstance(after_task, dict) else None
    after_task_status = ""
    after_task_title = ""
    if isinstance(after_task_json, dict):
        after_task_status = str(after_task_json.get("status") or "")
        after_task_title = str(after_task_json.get("title") or "")
    codex_exit_code = manifest.get("codex_exit_code", "")
    previous_context = summarize_previous_prompt(previous_delta)
    base_prompt_hash = file_sha256(base_prompt)
    base_prompt_note = (
        f"path: {base_prompt}\n"
        f"sha256: {base_prompt_hash or '(missing)'}\n"
        "The runner prepends this canonical Codex base prompt to your Direction Brief. "
        "Do not repeat the base prompt. Codex will turn your brief into or repair a Trellis PRD."
    )
    return f"""You are Hermes supervisor for the TianChen-RV MLIR repository.

You are a read-only planner and reviewer. Review the Codex worker run that just
finished and produce the next Hermes Direction Brief. You do not edit files and
you do not implement code.

Repository root:

```text
/home/kingdom/phdworks/TianchenRV
```

The runner will prepend the canonical Codex base prompt to your `next_prompt`.
Therefore `next_prompt` must be a focused Direction Brief, not a full PRD, not
a repeated copy of the base prompt, and not a long architecture document.

## Evidence Priority

Default review evidence is the live summary below:

1. changed files
2. diff stat
3. latest commit summary
4. current Trellis task status/title
5. git status summary

Only read more files when that live summary does not answer the owner /
direction question. If more evidence is needed, keep the inspection bounded to
the directly relevant files or directories. Do not rebuild a repo audit, do not
open `repo_audit.md` by default, and do not broad-grep the whole repository.
Artifact paths are entry points, not a mandatory reading checklist.

If Codex says one thing and repository evidence says another, trust repository
evidence. When you need read-only commands, prefer a small bounded set such as:

```bash
git show --name-status --stat --oneline --decorate HEAD
sed -n '1,220p' <changed-file-or-directly-related-spec>
rg -n "<symbol-or-phrase>" <specific-file-or-directly-related-directory>
```

Never modify files.

## Active Human Steering

Manual steering is a control-plane input for choosing the next owner. Treat it
as newer than the base prompt, but not as proof of repository state.

```text
{manual_steering or "(none)"}
```

## Review Job

Codex task completion is normal. Review the completed task's contribution to
the current module direction. Decide exactly one next direction. Do not ask
Codex to choose among candidate tasks. In `reason`, name the case and explain
how the next owner will make a compiler path more real. If you cannot answer
that, the task is too small and you must choose a larger owner.

```text
continue same direction: current owner is still the right bottleneck
expand same direction: previous work was too small and should become one coherent module
switch direction: current module converged, is stale, or is causing repeated stall
redirect direction: previous task drifted from the current architecture or evidence
```

The next owner should be large enough to move a real compiler path or workflow
path, but still bounded enough for one Codex round to complete a coherent
submodule. Good owners make one of these more real:

```text
TianChen-RV MLIR -> selected boundary -> plugin-owned lowering/emission -> artifact/runtime evidence
an IR boundary modeled and consumed by pass/exporter
a selected variant materialized to extension ops
a plugin-owned extension family lowering or EmitC emission path completed
a runtime ABI boundary carried from IR to artifact
a bounded RVV kernel carried from IR to generated artifact and ssh rvv evidence
an extension plugin template that makes future plugin integration clearer
```

Tiny helpers, one-off negative tests, broad smoke matrices, metadata-only
changes, wrapper-only work, status/report work, and standalone evidence
packaging are not valid default owners. Use them only when they are the single
named blocker for the chosen module.

## Wrong Logic Deletion Campaign Gate

This is the highest-priority owner-selection rule. While the live repository
still contains old logic that conflicts with the final TianChen-RV architecture,
choose a deletion-only owner. Do not choose a new feature owner, do not extend
the finite RVV family, do not add helpers, wrappers, compatibility paths,
descriptor tests, or new architecture implementation, and do not mix deletion
with rebuild work in the same round.

During the campaign, your `next_prompt` may only ask Codex to delete, remove
references, delete or rewrite tests that protect old behavior, remove obsolete
docs/comments, remove old routes/exporters, and remove old descriptor
dependencies. If removal exposes build or test failures, Codex should report
the breakage as a missing new-architecture gap and continue deletion/refactor
in the next round. Codex must not restore the wrong path to make checks pass.

Do not preserve old logic for compatibility. Do not quarantine old logic. Do
not add wrappers around old logic. Do not turn old logic into legacy mode. If
it is architecturally wrong, remove it. If removal breaks build/tests, report
the breakage and continue deletion/refactor in the next round. Do not restore
the wrong path.

Deletion Campaign objects:

```text
descriptor-driven computation:
  lowering_descriptor; descriptor decides microkernel, intrinsic route,
  artifact route, C generation, route authority, plugin/template/RAG basis,
  finite descriptor family production paths, i32-vadd-microkernel.v1-style
  descriptor semantics, RVVBinaryDescriptor, RVVBinaryFamilyRegistry,
  descriptor-based dispatch, descriptor-to-C exporter.

direct C semantic exporter:
  any path that bypasses extension family ops / EmitC ops and directly uses
  metadata, descriptors, selected routes, or family registries to synthesize
  C compute semantics. Real EmitC module -> C/C++ emitter paths, generated C
  compile harnesses, artifact readers, and non-semantic helper output may stay.

independent backend/dialect wording:
  specs, prompts, docs, comments, or tests that describe RVV, IME, TensorExt,
  Offload, scalar fallback, or future vendor targets as independent backends
  rather than TCRV extension families.

core extension-specific semantic branch:
  core orchestration that knows RVV intrinsic names, RVV microkernel semantics,
  scalar loop semantics, offload runtime call semantics, TensorExt/IME fragment
  semantics, descriptor family semantics, or if-RVV/if-IME/if-TensorExt compute
  branches instead of common interfaces, registry, and route abstraction.

Python compiler core:
  Python implementations of IR semantics, lowering, plugin registry, codegen,
  route selection, capability model, or compiler-core behavior.
```

Allowed campaign owners:

```text
Descriptor Erasure Owner:
  remove descriptor-driven compute authority, descriptor microkernel selection,
  descriptor artifact authority, descriptor plugin/template/RAG basis, and
  descriptor-protecting tests.

Direct C Semantic Exporter Erasure Owner:
  remove direct C compute-body generation as a semantic path; remaining
  rendering may only render existing EmitC modules or non-semantic packaging.

Independent Backend/Dialect Cleanup Owner:
  remove wording and durable prompt/spec/doc/comment/test residue that treats
  extension families as independent backends or dialect systems.

Core Semantic Branch Erasure Owner:
  remove extension-specific compute branches from core orchestration; core may
  organize capability, variants, dispatch/fallback, ABI envelope, registry, and
  common-interface dispatch only.

Legacy Tests and Artifact Cleanup Owner:
  remove tests, fixtures, artifacts, and negative tests that protect descriptor
  routes, direct C semantic exporters, or descriptor-as-legal-input behavior.
```

Deletion before rebuild is the campaign rule (`deletion before rebuild`).
Deletion Campaign is not the rebuild phase. Do not ask Codex to implement new
general RVV lowering, common lower-to-EmitC pass, executable plugin template,
TensorExt/IME extension, new EmitC route, new capability model features, or new
performance/evidence matrix until the old wrong logic is deleted.

Exit Deletion Campaign only when live repo evidence shows all conditions below
are true:

```text
descriptor is no longer compute semantics;
direct C exporter is no longer a semantic route;
core pass code has no extension-specific compute branch;
specs/prompts/docs do not describe extension families as independent backends;
legacy tests no longer protect old paths;
remaining failures are clearly missing new architecture, not old-path compatibility.
```

After exit, choose a rebuild owner such as Common Extension Interface
Foundation, Common Lower-To-EmitC Pass, Executable Plugin Construction
Template, or General RVV Extension Family Rebuild.

## Structural Migration Review

For migration or refactor tasks, check whether the production/default path
actually changed. Adding helper infrastructure, metadata, evidence, or tests is
not enough if the old path remains the source of compute semantics. If a
previous round built a replacement path, the next owner should usually switch
the default path to it, delete or bypass obsolete code, or make the new route
the production route instead of adding coverage for the old route.

For architecture cleanup, Codex may delete, replace, or rewrite obsolete code
and tests. Do not reward preserving descriptor-driven tests when the active
task is to remove descriptor authority from the default path.

For RVV migration work, prefer extension family ops as the source of truth,
common EmitC route usage, production/default path rewiring, and deletion of
obsolete descriptor-driven behavior. Reject rounds that only add
finite-family coverage, route metadata checks, helper-only tests, smoke tests,
or ssh evidence as a standalone owner when the missing piece is structural
migration. Descriptor-driven computation is invalid as long-term architecture.

## Anti-Stall Rule

Before writing the next brief, ask:

```text
Did the last round move an end-to-end compiler path closer to completion?
Did it make a module behavior available, or only add helper/test/smoke/report work?
Is the current Trellis task too small, stale, or missing a clear PRD?
Would another micro-round repeat the same stall pattern?
```

If several recent rounds did not move an end-to-end path closer, stop refining
the same small surface. Create, repair, or expand a module-level Trellis PRD and
make Codex execute that module instead.

For extension/plugin work, do not treat a checklist, metadata-only manifest, or
documentation-only template as sufficient progress when the missing piece is an
executable extension-family construction template. Prefer owners that make a
family declaration, interface realization, EmitC route mapping, or evidence
profile consumable by code or tests.

## Architecture Audit Surface

Keep the audit concise, but block or redirect if the worker:

```text
implements compiler internals in Python;
bypasses the C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck stack;
adds compute semantics to tcrv.exec or treats tcrv.exec.kernel as a mathematical kernel/hardware IR body;
treats RVV, IME, TensorExt, Offload, scalar fallback, or future vendor targets as independent backend dialects instead of TCRV extension families;
puts extension-specific semantic branches in core passes instead of TCRV common interfaces and plugin hooks;
adds computation semantics through descriptors or direct descriptor-to-C exporters;
routes new executable lowering around extension family ops -> EmitC -> intrinsic/vendor builtin/runtime C/C++;
implements a new plugin without the Extension-Family Plugin Construction Protocol;
omits extension archetype, semantic role graph, common interface realization, EmitC route mapping, or evidence profile for extension/plugin work;
treats the Extension Manifest as metadata-only documentation rather than the machine-readable construction entry point;
modifies core orchestration passes to give one extension a semantic special case;
claims GCC or vendor compilers are the default route instead of clang/LLVM default with compatibility paths;
confuses hardware facts, compile-time variant config, runtime SSA/control values, or descriptor-local parameters;
claims RVV runtime/correctness/performance without ssh rvv evidence;
treats prompt/report/smoke/helper work as the main result.
```

Detailed architecture rules live in `.trellis/spec/`; point Codex to the
relevant specs instead of restating all rules in the Direction Brief.

## Required next_prompt Shape

When `continue=true`, `next_prompt` must be a non-empty Hermes Direction Brief
to append under the base prompt. Codex will create or repair the Trellis PRD
from this brief. Include:

```text
Direction title:
  concise name of the next owner
Module owner:
  the single owner for this round
Why now:
  evidence-based reason this owner is the next bottleneck
Previous completed task contributed:
  one sentence on what the last commit/task made real
Capability to improve next:
  the compiler path, IR boundary, route, artifact, or plugin surface that should become more real
Read first:
  specific specs, task files, and code directories
What Codex should turn into a Trellis PRD:
  module goal, boundary, and acceptance criteria, not detailed implementation steps
Non-goals:
  what Codex must not do
Minimal evidence expected:
  focused checks and evidence only for this module
Final report:
  task id/title, phase, files changed, checks, self-repair, finish/archive, commit
```

The brief must not contain three candidate tasks for Codex to choose from. Do
not write a full PRD or detailed implementation plan. If the PRD is unclear,
tell Codex to repair the PRD first rather than choosing a different direction.
If the module is too large, tell Codex to finish one coherent submodule and
keep the task state truthful for the next Hermes round.

## Stop Conditions

Set `continue=false` only when the user requested stop, evidence is missing or
corrupted enough that continuation is unsafe, credentials or hardware access
require human intervention, or repeated failures make another automatic round
harmful.

## Required Output

Return only one JSON object. No Markdown, no code fence, no text outside JSON.
The JSON must be parseable by Python `json.loads` and contain exactly:

```json
{{
  "continue": true,
  "next_prompt": "Hermes Direction Brief appended under the base prompt; required when continue is true",
  "base_prompt_edits": [],
  "delta": "",
  "reason": "brief audit conclusion and why this module owner was chosen",
  "telegram_note": "very short optional user-facing note"
}}
```

`next_prompt` must be a single JSON string. Encode newlines as `\\n`; do not
place literal unescaped line breaks inside the JSON string.

## Loop Context

round_index: {round_index}
run_dir: {run_dir}
codex_exit_code: {codex_exit_code}
before_head: {diff.get("before", "")}
after_head: {diff.get("after", "")}
current_trellis_task: {after_task.get("ref", "") or "(none)"}
current_trellis_task_status: {after_task_status or "(unknown)"}
current_trellis_task_title: {after_task_title or "(unknown)"}
previous_prompt_mode: {previous_prompt_mode or "base"}
previous_direction_summary:
{previous_context}

## Canonical Codex Base Prompt

{base_prompt_note}

## Live Summary

git status --short:
```text
{git_status or "(clean)"}
```

new commits:
```text
{new_commits or "(none)"}
```

changed files:
```text
{changed_files or "(none)"}
```

diff stat:
```text
{diff_stat or "(none)"}
```

latest commit summary:
```text
{latest_commit or "(missing)"}
```

## Artifact Entry Points

repo_audit.md:
  path: {run_dir / "repo_audit.md"}
  note: cached read-only evidence; inspect the live repo first, open repo_audit only when useful.
review_input.md:
  path: {run_dir / "review_input.md"}
manifest:
  path: {run_dir / "manifest.json"}
last_message.md:
  path: {run_dir / "last_message.md"}
stderr:
  path: {run_dir / "codex.stderr.log"}
snapshot_before:
  path: {run_dir / "snapshot_before.json"}
snapshot_after:
  path: {run_dir / "snapshot_after.json"}
loop_manifest:
  path: {loop_dir / "loop_manifest.json"}
events:
  path: {loop_dir / "events.jsonl"}

## Run Manifest Summary

```json
{json.dumps({k: manifest.get(k) for k in ("run_dir", "repo", "codex_exit_code", "prompt", "review_input", "snapshot_before", "snapshot_after", "stdout", "stderr", "last_message", "repo_audit", "prompt_source", "runner_error") if k in manifest}, ensure_ascii=False, indent=2)}
```
"""


def hermes_review(
    args: argparse.Namespace,
    repo: Path,
    run_dir: Path,
    loop_dir: Path,
    round_index: int,
    previous_delta: str,
    previous_prompt_mode: str,
) -> dict[str, Any]:
    if args.review_no_llm:
        return {
            "continue": True,
            "delta": "",
            "next_prompt": (
                "Review LLM is disabled for this round. Inspect repository evidence, "
                "continue the current Trellis task if one is active, and otherwise "
                "create or repair a module-sized Trellis task before implementation."
            ),
            "base_prompt_edits": [],
            "reason": "review_no_llm was set; using a minimal Direction Brief with the canonical base prompt.",
            "telegram_note": "",
            "raw": "",
            "parse_error": False,
            "cmd": [],
            "returncode": 0,
        }

    prompt = build_review_prompt(
        run_dir=run_dir,
        loop_dir=loop_dir,
        base_prompt=Path(args.base_prompt),
        round_index=round_index,
        previous_delta=previous_delta,
        previous_prompt_mode=previous_prompt_mode,
        max_chars=args.review_max_chars,
        manual_steering=read_review_steering(
            repo,
            args,
            max_chars=max(4000, int(args.review_max_chars * 0.12)),
        ),
    )
    review_prompt_path = loop_dir / f"round_{round_index:04d}_hermes_review_prompt.md"
    review_prompt_path.write_text(prompt, encoding="utf-8")

    def build_cmd(review_prompt: str, max_turns: int) -> list[str]:
        if args.hermes_review_mode == "oneshot":
            cmd = [args.hermes_bin]
            if args.hermes_model:
                cmd.extend(["--model", args.hermes_model])
            if args.hermes_provider:
                cmd.extend(["--provider", args.hermes_provider])
            if args.hermes_ignore_rules:
                cmd.append("--ignore-rules")
            for skill in [s.strip() for s in args.hermes_skills.split(",") if s.strip()]:
                cmd.extend(["--skills", skill])
            cmd.extend(["-z", review_prompt])
            return cmd

        cmd = [
            args.hermes_bin,
            "chat",
            "-q",
            review_prompt,
            "-Q",
            "--source",
            args.hermes_source,
            "--max-turns",
            str(max_turns),
        ]
        if args.hermes_session_id:
            cmd.extend(["--resume", args.hermes_session_id])
        if args.hermes_model:
            cmd.extend(["--model", args.hermes_model])
        if args.hermes_provider:
            cmd.extend(["--provider", args.hermes_provider])
        if args.hermes_ignore_rules:
            cmd.append("--ignore-rules")
        if args.hermes_pass_session_id:
            cmd.append("--pass-session-id")
        for skill in [s.strip() for s in args.hermes_skills.split(",") if s.strip()]:
            cmd.extend(["--skills", skill])
        return cmd

    def run_attempt(attempt_index: int, review_prompt: str, max_turns: int) -> dict[str, Any]:
        cmd = build_cmd(review_prompt, max_turns)
        with BlockingFileLock(artifact_root_path(repo, args.artifact_root) / HERMES_SESSION_LOCK_FILE_NAME):
            proc = subprocess.run(
                cmd,
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.hermes_timeout,
            )
        suffix = "" if attempt_index == 0 else f"_retry{attempt_index}"
        (loop_dir / f"round_{round_index:04d}_hermes{suffix}.stdout").write_text(
            proc.stdout,
            encoding="utf-8",
        )
        (loop_dir / f"round_{round_index:04d}_hermes{suffix}.stderr").write_text(
            proc.stderr,
            encoding="utf-8",
        )
        review = normalize_review(proc.stdout)
        hermes_session_id = extract_hermes_session_id(proc.stdout)
        session_lookup: dict[str, Any] | None = None
        if proc.returncode == 0 and not hermes_session_id and not args.hermes_session_id:
            session_lookup = latest_hermes_session_id(args, repo)
            hermes_session_id = str(session_lookup.get("session_id") or "")
        rename_result: dict[str, Any] | None = None
        if (
            proc.returncode == 0
            and hermes_session_id
            and args.hermes_session_name
            and not args.hermes_session_id
        ):
            rename_proc = subprocess.run(
                [args.hermes_bin, "sessions", "rename", hermes_session_id, args.hermes_session_name],
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            rename_result = {
                "returncode": rename_proc.returncode,
                "stdout": rename_proc.stdout.strip(),
                "stderr": rename_proc.stderr.strip(),
            }
        review.update(
            {
                "attempt": attempt_index,
                "cmd": [*cmd[:3], "<prompt>", *cmd[4:]]
                if args.hermes_review_mode == "chat"
                else cmd[:-1] + ["<prompt>"],
                "returncode": proc.returncode,
                "review_prompt": str(review_prompt_path),
                "hermes_review_mode": args.hermes_review_mode,
                "hermes_max_turns": max_turns,
                "hermes_session_id": hermes_session_id or args.hermes_session_id,
                "hermes_session_name": args.hermes_session_name,
                "hermes_session_lookup": session_lookup,
                "hermes_session_rename": rename_result,
                "manual_steering_once_consumption": {
                    "path": str(manual_steering_once_path(repo, args)),
                    "consumed": False,
                    "reason": "pending_successful_official_review",
                    "review_prompt": str(review_prompt_path),
                },
            }
        )
        if proc.returncode != 0:
            review["continue"] = False
            review["delta"] = ""
            review["next_prompt"] = ""
            review["base_prompt_edits"] = []
            review["parse_error"] = True
            review["reason"] = (
                f"Hermes review failed with exit code {proc.returncode}: {proc.stderr[-1000:]}"
            )
            review["telegram_note"] = "Hermes review failed; supervisor is blocked."
        return review

    attempts: list[dict[str, Any]] = []
    review = run_attempt(0, prompt, args.hermes_max_turns)
    attempts.append(
        {
            "attempt": review.get("attempt", 0),
            "parse_error": review.get("parse_error", False),
            "returncode": review.get("returncode", 0),
            "hermes_max_turns": review.get("hermes_max_turns", args.hermes_max_turns),
            "raw_prefix": str(review.get("raw") or "")[:500],
            "reason": review.get("reason", ""),
        }
    )
    retry_count = max(0, int(getattr(args, "hermes_review_retries", 0)))
    if review.get("parse_error") and retry_count:
        failed_raw = str(review.get("raw") or "")
        raw_lower = failed_raw.lower()
        has_repairable_review = bool(failed_raw.strip()) and (
            "next_prompt" in raw_lower
            or "continue" in raw_lower
            or "module owner" in raw_lower
            or "direction" in raw_lower
        )
        if has_repairable_review:
            retry_prompt = (
                "Convert the previous Hermes supervisor output into exactly one strict JSON object. "
                "Do not perform new repository review and do not add new reasoning. Preserve the intended "
                "decision and direction brief from the raw output when present.\n\n"
                "Required JSON keys exactly:\n"
                "continue, next_prompt, base_prompt_edits, delta, reason, telegram_note\n\n"
                "Rules:\n"
                "- Return only JSON, no markdown and no code fence.\n"
                "- If the raw output does not contain enough information for a non-empty next_prompt, set continue=false.\n"
                "- next_prompt must be a single JSON string when continue=true.\n\n"
                "Raw output prefix:\n"
                "```text\n"
                + failed_raw[:6000]
                + "\n```\n\n"
                "Raw output tail:\n"
                "```text\n"
                + failed_raw[-6000:]
                + "\n```\n"
            )
            retry_prompt_path = loop_dir / f"round_{round_index:04d}_hermes_review_retry_prompt.md"
            retry_prompt_path.write_text(retry_prompt, encoding="utf-8")
            retry_turns = max(4, args.hermes_retry_max_turns)
            for attempt_index in range(1, retry_count + 1):
                review = run_attempt(attempt_index, retry_prompt, retry_turns)
                attempts.append(
                    {
                        "attempt": review.get("attempt", attempt_index),
                        "parse_error": review.get("parse_error", False),
                        "returncode": review.get("returncode", 0),
                        "hermes_max_turns": review.get("hermes_max_turns", retry_turns),
                        "raw_prefix": str(review.get("raw") or "")[:500],
                        "reason": review.get("reason", ""),
                    }
                )
                if not review.get("parse_error"):
                    break
        else:
            attempts.append(
                {
                    "attempt": "repair_skipped",
                    "parse_error": True,
                    "returncode": review.get("returncode", 0),
                    "hermes_max_turns": 0,
                    "raw_prefix": failed_raw[:500],
                    "reason": "Hermes raw output was not repairable without rerunning the full review.",
                }
            )
    review["hermes_attempts"] = attempts
    if review.get("parse_error"):
        review["continue"] = False
        review["delta"] = ""
        review["next_prompt"] = ""
        review["base_prompt_edits"] = []
        review["reason"] = (
            "Hermes review did not return valid strict JSON after official Hermes retry attempts; "
            "supervisor must stop instead of falling back to a Codex-authored Direction Brief."
        )
        review["telegram_note"] = "Hermes review JSON failed after retry; supervisor stopped."
    return review


def read_question(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if getattr(args, "question", ""):
        parts.append(str(args.question).strip())
    if getattr(args, "question_file", ""):
        parts.append(read_text(Path(args.question_file), max_chars=20000).strip())
    return "\n\n".join(p for p in parts if p).strip()


def build_hermes_ask_prompt(
    ask_dir: Path,
    question: str,
    manual_steering: str,
    max_chars: int,
) -> str:
    max_chars = max(max_chars, 12000)
    review_input = read_text(ask_dir / "review_input.md", max_chars=int(max_chars * 0.30))
    repo_audit = read_text(ask_dir / "repo_audit.md", max_chars=int(max_chars * 0.42))
    manifest = read_text(ask_dir / "manifest.json", max_chars=6000)
    question = question or (
        "Perform a read-only supervisor self-check of the current TianChen-RV repository state. "
        "Explain whether the next Codex owner should continue the compiler spine, pause for workspace hygiene, "
        "or stop for human intervention."
    )
    return f"""You are Hermes supervisor for the TianChen-RV MLIR repository.

This is an ask-only self-check. Do not modify the repository. Do not launch Codex. Do not generate or write a next worker prompt unless the user explicitly asks for one in this ask-only question.

Use evidence in this order:

1. Real repository state and file contents from the live checkout.
2. `repo_audit.md` generated by the runner.
3. `review_input.md` and run manifest.
4. Prior chat context.

If prior chat context conflicts with current repository evidence, trust current repository evidence.

Answer in concise Markdown. It is okay to name concrete files, commits, risks, and suggested next owner candidates. Keep the answer read-only.

## User Question

```text
{question}
```

## Active Human Steering

```text
{manual_steering or "(none)"}
```

## Ask Manifest

{manifest or "(missing)"}

## review_input.md

{review_input or "(missing)"}

## repo_audit.md

{repo_audit or "(missing)"}
"""


def command_ask_hermes(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    root = artifact_root_path(repo, args.artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    session_resolution = resolve_hermes_session_id(args, repo)
    ask_dir = root / "asks" / (args.run_id or utc_run_id())
    ask_dir.mkdir(parents=True, exist_ok=False)

    before = collect_snapshot(repo)
    write_json(ask_dir / "snapshot.json", before)
    (ask_dir / "last_message.md").write_text("(ask-hermes: no Codex worker was launched)\n", encoding="utf-8")
    repo_audit_path = write_repo_audit(ask_dir, before, before)
    write_review_input(
        ask_dir,
        before,
        before,
        None,
        "ask-hermes read-only self-check; no Codex worker was launched.",
    )
    question = read_question(args)
    manual_steering = read_manual_steering(repo, args, max_chars=max(4000, int(args.review_max_chars * 0.12)))
    manifest = {
        "ask_dir": str(ask_dir),
        "repo": str(repo),
        "question_chars": len(question),
        "manual_steering_file": str(manual_steering_path(repo, args)),
        "manual_steering_chars": len(manual_steering),
        "repo_audit": str(repo_audit_path),
        "review_input": str(ask_dir / "review_input.md"),
        "hermes_review_mode": "chat",
        "hermes_session_id": args.hermes_session_id,
        "hermes_session_resolution": session_resolution,
        "render_only": bool(args.render_only),
    }
    write_json(ask_dir / "manifest.json", manifest)
    prompt = build_hermes_ask_prompt(
        ask_dir=ask_dir,
        question=question,
        manual_steering=manual_steering,
        max_chars=args.review_max_chars,
    )
    (ask_dir / "hermes_ask_prompt.md").write_text(prompt, encoding="utf-8")

    if args.render_only:
        result = {
            "ask_dir": str(ask_dir),
            "prompt": str(ask_dir / "hermes_ask_prompt.md"),
            "render_only": True,
            "hermes_session_id": args.hermes_session_id,
            "hermes_session_resolution": session_resolution,
        }
        write_json(ask_dir / "result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    cmd = [
        args.hermes_bin,
        "chat",
        "-q",
        prompt,
        "-Q",
        "--source",
        args.hermes_source,
        "--max-turns",
        str(args.hermes_max_turns),
    ]
    if args.hermes_session_id:
        cmd.extend(["--resume", args.hermes_session_id])
    if args.hermes_model:
        cmd.extend(["--model", args.hermes_model])
    if args.hermes_provider:
        cmd.extend(["--provider", args.hermes_provider])
    if args.hermes_ignore_rules:
        cmd.append("--ignore-rules")
    if args.hermes_pass_session_id:
        cmd.append("--pass-session-id")
    for skill in [s.strip() for s in args.hermes_skills.split(",") if s.strip()]:
        cmd.extend(["--skills", skill])

    with BlockingFileLock(root / HERMES_SESSION_LOCK_FILE_NAME):
        proc = subprocess.run(
            cmd,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.hermes_timeout,
        )
    (ask_dir / "hermes.stdout").write_text(proc.stdout, encoding="utf-8")
    (ask_dir / "hermes.stderr").write_text(proc.stderr, encoding="utf-8")
    hermes_session_id = extract_hermes_session_id(proc.stdout) or args.hermes_session_id
    result = {
        "ask_dir": str(ask_dir),
        "prompt": str(ask_dir / "hermes_ask_prompt.md"),
        "stdout": str(ask_dir / "hermes.stdout"),
        "stderr": str(ask_dir / "hermes.stderr"),
        "returncode": proc.returncode,
        "hermes_session_id": hermes_session_id,
        "hermes_session_resolution": session_resolution,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-2000:],
    }
    write_json(ask_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return proc.returncode


def build_run_subprocess_cmd(
    args: argparse.Namespace,
    run_id: str,
    delta_file: Path,
    prompt_override_file: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--repo",
        args.repo,
        "--artifact-root",
        args.artifact_root,
        "--base-prompt",
        args.base_prompt,
        "--run-id",
        run_id,
        "--delta-file",
        str(delta_file),
        "--prompt-override-file",
        str(prompt_override_file),
        "--codex-bin",
        args.codex_bin,
        "--telegram-event",
        args.telegram_event,
        "--telegram-timeout",
        str(args.telegram_timeout),
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.profile:
        cmd.extend(["--profile", args.profile])
    for item in args.codex_arg or []:
        cmd.extend(["--codex-arg", item])
    if args.keep_proxy:
        cmd.append("--keep-proxy")
    if args.no_exec:
        cmd.append("--no-exec")
    if args.no_telegram:
        cmd.append("--no-telegram")
    if args.telegram_chat_id:
        cmd.extend(["--telegram-chat-id", args.telegram_chat_id])
    if args.telegram_bot_token:
        cmd.extend(["--telegram-bot-token", args.telegram_bot_token])
    if args.telegram_proxy:
        cmd.extend(["--telegram-proxy", args.telegram_proxy])
    return cmd


def build_loop_subprocess_cmd(args: argparse.Namespace, loop_id: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "loop",
        "--repo",
        args.repo,
        "--artifact-root",
        args.artifact_root,
        "--base-prompt",
        args.base_prompt,
        "--loop-id",
        loop_id,
        "--max-rounds",
        str(args.max_rounds),
        "--sleep-seconds",
        str(args.sleep_seconds),
        "--codex-bin",
        args.codex_bin,
        "--telegram-event",
        args.telegram_event,
        "--telegram-timeout",
        str(args.telegram_timeout),
        "--review-max-chars",
        str(args.review_max_chars),
        "--hermes-bin",
        args.hermes_bin,
        "--hermes-timeout",
        str(args.hermes_timeout),
        "--hermes-skills",
        args.hermes_skills,
        "--hermes-review-mode",
        args.hermes_review_mode,
        "--hermes-source",
        args.hermes_source,
        "--hermes-max-turns",
        str(args.hermes_max_turns),
        "--hermes-review-retries",
        str(args.hermes_review_retries),
        "--hermes-retry-max-turns",
        str(args.hermes_retry_max_turns),
        "--codex-transient-retries",
        str(args.codex_transient_retries),
    ]
    if args.stop_file:
        cmd.extend(["--stop-file", args.stop_file])
    if getattr(args, "manual_steering_file", ""):
        cmd.extend(["--manual-steering-file", args.manual_steering_file])
    if getattr(args, "manual_steering_once_file", ""):
        cmd.extend(["--manual-steering-once-file", args.manual_steering_once_file])
    if args.initial_delta:
        cmd.extend(["--initial-delta", args.initial_delta])
    if args.initial_delta_file:
        cmd.extend(["--initial-delta-file", args.initial_delta_file])
    if getattr(args, "resume_review_run_dir", ""):
        cmd.extend(["--resume-review-run-dir", args.resume_review_run_dir])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.profile:
        cmd.extend(["--profile", args.profile])
    for item in args.codex_arg or []:
        cmd.extend(["--codex-arg", item])
    if args.keep_proxy:
        cmd.append("--keep-proxy")
    if args.no_exec:
        cmd.append("--no-exec")
    if args.no_telegram:
        cmd.append("--no-telegram")
    if args.telegram_chat_id:
        cmd.extend(["--telegram-chat-id", args.telegram_chat_id])
    if args.telegram_bot_token:
        cmd.extend(["--telegram-bot-token", args.telegram_bot_token])
    if args.telegram_proxy:
        cmd.extend(["--telegram-proxy", args.telegram_proxy])
    if args.review_no_llm:
        cmd.append("--review-no-llm")
    if args.hermes_session_id:
        cmd.extend(["--hermes-session-id", args.hermes_session_id])
    if not getattr(args, "resume_latest_hermes_session", True):
        cmd.append("--no-resume-latest-hermes-session")
    if args.hermes_session_name:
        cmd.extend(["--hermes-session-name", args.hermes_session_name])
    if args.hermes_model:
        cmd.extend(["--hermes-model", args.hermes_model])
    if args.hermes_provider:
        cmd.extend(["--hermes-provider", args.hermes_provider])
    if args.hermes_ignore_rules:
        cmd.append("--hermes-ignore-rules")
    if not args.hermes_pass_session_id:
        cmd.append("--no-hermes-pass-session-id")
    return cmd


def run_worker_subprocess(
    args: argparse.Namespace,
    repo: Path,
    loop_dir: Path,
    round_index: int,
    run_id: str,
    delta: str,
    prompt_override: str,
) -> dict[str, Any]:
    delta_file = loop_dir / f"round_{round_index:04d}_delta.md"
    delta_file.write_text(delta.strip() + ("\n" if delta.strip() else ""), encoding="utf-8")
    prompt_override_file = loop_dir / f"round_{round_index:04d}_prompt_override.md"
    prompt_override_file.write_text(
        prompt_override.strip() + ("\n" if prompt_override.strip() else ""),
        encoding="utf-8",
    )
    cmd = build_run_subprocess_cmd(args, run_id, delta_file, prompt_override_file)
    proc = subprocess.run(
        cmd,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (loop_dir / f"round_{round_index:04d}_run.stdout").write_text(proc.stdout, encoding="utf-8")
    (loop_dir / f"round_{round_index:04d}_run.stderr").write_text(proc.stderr, encoding="utf-8")
    run_dir = artifact_root_path(repo, args.artifact_root) / "runs" / run_id
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "run_id": run_id,
        "run_dir": str(run_dir),
        "prompt_mode": "override" if prompt_override.strip() else "delta" if delta.strip() else "base",
        "prompt_override_file": str(prompt_override_file),
    }


def command_start(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    root = artifact_root_path(repo, args.artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    loop_id = args.loop_id or utc_run_id()
    stop_file = stop_file_path(repo, args)
    if stop_file.exists() and not args.keep_stop_file:
        stop_file.unlink()
    session_resolution = resolve_hermes_session_id(args, repo)
    cmd = build_loop_subprocess_cmd(args, loop_id)
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{loop_id}.stdout.log"
    stderr_path = logs_dir / f"{loop_id}.stderr.log"
    with stdout_path.open("a", encoding="utf-8") as stdout_fh, stderr_path.open("a", encoding="utf-8") as stderr_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=repo,
            stdin=subprocess.DEVNULL,
            stdout=stdout_fh,
            stderr=stderr_fh,
            start_new_session=True,
            close_fds=True,
        )
    result = {
        "loop_id": loop_id,
        "pid": proc.pid,
        "repo": str(repo),
        "loop_dir": str(root / "loops" / loop_id),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stop_file": str(stop_file),
        "manual_steering_file": str(manual_steering_path(repo, args)),
        "manual_steering_once_file": str(manual_steering_once_path(repo, args)),
        "hermes_review_mode": args.hermes_review_mode,
        "hermes_session_id": args.hermes_session_id,
        "hermes_session_resolution": session_resolution,
        "hermes_session_name": args.hermes_session_name,
        "codex_multi_agent_disabled": True,
        "cmd": cmd,
    }
    write_json(root / "starts" / f"{loop_id}.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_loop(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    root = artifact_root_path(repo, args.artifact_root)
    session_resolution = resolve_hermes_session_id(args, repo)
    loop_id = args.loop_id or utc_run_id()
    loop_dir = root / "loops" / loop_id
    loop_dir.mkdir(parents=True, exist_ok=False)
    stop_file = stop_file_path(repo, args)
    manual_steering_file = manual_steering_path(repo, args)
    manual_steering_once_file = manual_steering_once_path(repo, args)
    active_loop_path = root / ACTIVE_LOOP_FILE_NAME
    events_path = loop_dir / "events.jsonl"
    delta = read_initial_delta(args)
    prompt_override = ""
    prompt_mode = "delta" if delta.strip() else "base"
    rounds_completed = 0
    exit_reason = "max_rounds_reached"
    ready_for_worker = True
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        active_loop_path,
        {
            "pid": os.getpid(),
            "loop_id": loop_id,
            "loop_dir": str(loop_dir),
            "repo": str(repo),
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stop_file": str(stop_file),
            "manual_steering_file": str(manual_steering_file),
            "manual_steering_once_file": str(manual_steering_once_file),
            "max_rounds": args.max_rounds,
            "hermes_review_mode": args.hermes_review_mode,
            "hermes_session_id": args.hermes_session_id,
            "hermes_session_resolution": session_resolution,
            "hermes_session_name": args.hermes_session_name,
            "codex_multi_agent_disabled": True,
            "next_prompt_mode": prompt_mode,
        },
    )
    write_json(
        loop_dir / "loop_manifest.json",
        {
            "loop_id": loop_id,
            "repo": str(repo),
            "artifact_root": str(root),
            "stop_file": str(stop_file),
            "manual_steering_file": str(manual_steering_file),
            "manual_steering_once_file": str(manual_steering_once_file),
            "max_rounds": args.max_rounds,
            "review_no_llm": args.review_no_llm,
            "hermes_review_mode": args.hermes_review_mode,
            "hermes_session_id": args.hermes_session_id,
            "hermes_session_resolution": session_resolution,
            "hermes_session_name": args.hermes_session_name,
            "codex_multi_agent_disabled": True,
            "initial_prompt_mode": prompt_mode,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    print(loop_dir, flush=True)

    try:
        if getattr(args, "resume_review_run_dir", ""):
            resume_run_dir = Path(args.resume_review_run_dir)
            if not resume_run_dir.is_absolute():
                resume_run_dir = (repo / resume_run_dir).resolve()
            if not resume_run_dir.exists():
                raise SystemExit(f"resume review run dir does not exist: {resume_run_dir}")
            if not (resume_run_dir / "review_input.md").exists():
                raise SystemExit(f"resume review run dir is missing review_input.md: {resume_run_dir}")

            resume_round = infer_round_index_from_run_dir(resume_run_dir)
            if resume_round <= 0:
                resume_round = 1
            append_jsonl(
                events_path,
                {
                    "event": "resume_review_start",
                    "round": resume_round,
                    "run_dir": str(resume_run_dir),
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            try:
                review = hermes_review(
                    args,
                    repo,
                    resume_run_dir,
                    loop_dir,
                    resume_round,
                    "",
                    "resume-review",
                )
            except Exception as exc:  # noqa: BLE001 - keep loop evidence explicit
                review = {
                    "continue": False,
                    "delta": "",
                    "next_prompt": "",
                    "base_prompt_edits": [],
                    "reason": (
                        f"Hermes resume review exception: {type(exc).__name__}: {exc}; "
                        "supervisor stopped instead of launching Codex before review."
                    ),
                    "telegram_note": "Hermes resume review raised an exception; supervisor stopped.",
                    "parse_error": True,
                }
            base_prompt_text = read_text(Path(args.base_prompt), max_chars=200000).strip()
            edited_prompt, prompt_edit_results = apply_prompt_edits(
                base_prompt_text,
                review.get("base_prompt_edits", []),
            )
            next_prompt = str(review.get("next_prompt") or "").strip()
            if not next_prompt and prompt_edit_results and any(r.get("applied") for r in prompt_edit_results):
                next_prompt = edited_prompt
            if review.get("continue", True) and not next_prompt:
                review["continue"] = False
                if review.get("parse_error"):
                    review["reason"] = (
                        "Hermes resume review did not provide valid JSON/next_prompt after retry; "
                        "supervisor stopped instead of using a human-written delta."
                    )
                else:
                    review["reason"] = "Hermes resume review did not provide required non-empty next_prompt Direction Brief."
            next_prompt_path = loop_dir / f"round_{resume_round:04d}_next_prompt.md"
            next_prompt_path.write_text(next_prompt + ("\n" if next_prompt else ""), encoding="utf-8")
            review["prompt_edit_results"] = prompt_edit_results
            review["next_prompt_path"] = str(next_prompt_path)
            review["next_prompt_chars"] = len(next_prompt)
            review_path = loop_dir / f"round_{resume_round:04d}_review.json"
            write_json(review_path, review)
            if not review.get("parse_error") and (
                not review.get("continue", True) or bool(next_prompt)
            ):
                review_prompt_path = Path(str(review.get("review_prompt") or ""))
                if not review_prompt_path.exists():
                    review_prompt_path = loop_dir / f"round_{resume_round:04d}_hermes_review_prompt.md"
                review["manual_steering_once_consumption"] = consume_manual_steering_once(
                    repo,
                    args,
                    loop_dir,
                    resume_round,
                    review_prompt_path,
                )
                write_json(review_path, review)
            if review.get("hermes_session_id") and not args.hermes_session_id:
                args.hermes_session_id = str(review.get("hermes_session_id"))
            write_json(
                loop_dir / "hermes_session.json",
                {
                    "hermes_review_mode": args.hermes_review_mode,
                    "hermes_session_id": args.hermes_session_id,
                    "hermes_session_name": args.hermes_session_name,
                    "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            write_json(
                active_loop_path,
                {
                    "pid": os.getpid(),
                    "loop_id": loop_id,
                    "loop_dir": str(loop_dir),
                    "repo": str(repo),
                    "latest_run_dir": str(resume_run_dir),
                    "latest_round": resume_round,
                    "stop_file": str(stop_file),
                    "manual_steering_file": str(manual_steering_file),
                    "manual_steering_once_file": str(manual_steering_once_file),
                    "hermes_review_mode": args.hermes_review_mode,
                    "hermes_session_id": args.hermes_session_id,
                    "hermes_session_name": args.hermes_session_name,
                    "codex_multi_agent_disabled": True,
                    "next_prompt_mode": "override" if next_prompt else "delta" if review.get("delta") else "base",
                    "next_prompt_chars": len(next_prompt),
                    "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            append_jsonl(
                events_path,
                {
                    "event": "hermes_review_finished",
                    "round": resume_round,
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "continue": review.get("continue", True),
                    "delta": review.get("delta", ""),
                    "next_prompt_path": str(next_prompt_path),
                    "next_prompt_chars": len(next_prompt),
                    "prompt_edit_results": prompt_edit_results,
                    "reason": review.get("reason", ""),
                    "telegram_note": review.get("telegram_note", ""),
                    "hermes_review_mode": review.get("hermes_review_mode", args.hermes_review_mode),
                    "hermes_session_id": review.get("hermes_session_id", args.hermes_session_id),
                    "review_path": str(review_path),
                    "manual_steering_once_consumption": review.get("manual_steering_once_consumption", {}),
                    "resume_review_run_dir": str(resume_run_dir),
                },
            )

            rounds_completed = resume_round
            if stop_file.exists():
                exit_reason = f"stop_file_present_after_resume_review_round_{resume_round}"
                ready_for_worker = False
            elif not review.get("continue", True):
                exit_reason = str(review.get("reason") or f"hermes_requested_stop_after_resume_review_round_{resume_round}")
                ready_for_worker = False
            elif next_prompt:
                prompt_override = next_prompt
                delta = ""
                prompt_mode = "override"
            else:
                prompt_override = ""
                delta = str(review.get("delta") or "").strip()
                prompt_mode = "delta" if delta else "base"

        while ready_for_worker and (args.max_rounds <= 0 or rounds_completed < args.max_rounds):
            next_round = rounds_completed + 1
            if stop_file.exists():
                exit_reason = f"stop_file_present_before_round_{next_round}"
                break

            run_id = f"{loop_id}-r{next_round:04d}-{utc_run_id()}"
            append_jsonl(
                events_path,
                {
                    "event": "round_start",
                    "round": next_round,
                    "run_id": run_id,
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "delta": delta,
                    "prompt_mode": prompt_mode,
                    "prompt_override_chars": len(prompt_override),
                },
            )
            worker = run_worker_subprocess(args, repo, loop_dir, next_round, run_id, delta, prompt_override)
            write_json(
                active_loop_path,
                {
                    "pid": os.getpid(),
                    "loop_id": loop_id,
                    "loop_dir": str(loop_dir),
                    "repo": str(repo),
                    "latest_run_dir": worker["run_dir"],
                    "latest_round": next_round,
                    "stop_file": str(stop_file),
                    "manual_steering_file": str(manual_steering_file),
                    "manual_steering_once_file": str(manual_steering_once_file),
                    "hermes_review_mode": args.hermes_review_mode,
                    "hermes_session_id": args.hermes_session_id,
                    "hermes_session_name": args.hermes_session_name,
                    "codex_multi_agent_disabled": True,
                    "current_prompt_mode": prompt_mode,
                    "current_prompt_override_chars": len(prompt_override),
                    "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            append_jsonl(
                events_path,
                {
                    "event": "worker_finished",
                    "round": next_round,
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    **worker,
                },
            )

            retry_reason = codex_transient_failure_reason(Path(worker["run_dir"]))
            if retry_reason and args.codex_transient_retries > 0:
                worker_run_dir = Path(worker["run_dir"])
                retry_blocker = codex_retry_blocker(worker_run_dir)
                if retry_blocker:
                    exit_reason = f"codex_transient_retry_skipped_{retry_blocker}"
                    append_jsonl(
                        events_path,
                        {
                            "event": "codex_transient_retry_skipped",
                            "round": next_round,
                            "run_id": run_id,
                            "reason": retry_reason,
                            "blocker": retry_blocker,
                            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        },
                    )
                    write_stop_request(stop_file, exit_reason)
                    append_jsonl(
                        events_path,
                        {
                            "event": "loop_stop",
                            "round": next_round,
                            "reason": exit_reason,
                            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        },
                    )
                    break

                continuation_retry = codex_retry_needs_continuation(worker_run_dir)
                retry_prompt_override = prompt_override
                if continuation_retry:
                    retry_prompt_override = build_codex_continuation_retry_prompt(
                        worker_run_dir,
                        retry_reason,
                        delta,
                        prompt_override,
                    )
                retry_run_id = f"{run_id}-retry1"
                append_jsonl(
                    events_path,
                    {
                        "event": "codex_transient_retry_start",
                        "round": next_round,
                        "attempt": 2,
                        "previous_run_id": run_id,
                        "run_id": retry_run_id,
                        "reason": retry_reason,
                        "continuation_retry": continuation_retry,
                        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                )
                retry_worker = run_worker_subprocess(
                    args,
                    repo,
                    loop_dir,
                    next_round,
                    retry_run_id,
                    delta,
                    retry_prompt_override,
                )
                retry_worker["attempt"] = 2
                write_json(
                    active_loop_path,
                    {
                        "pid": os.getpid(),
                        "loop_id": loop_id,
                        "loop_dir": str(loop_dir),
                        "repo": str(repo),
                        "latest_run_dir": retry_worker["run_dir"],
                        "latest_round": next_round,
                        "stop_file": str(stop_file),
                        "manual_steering_file": str(manual_steering_file),
                        "manual_steering_once_file": str(manual_steering_once_file),
                        "hermes_review_mode": args.hermes_review_mode,
                        "hermes_session_id": args.hermes_session_id,
                        "hermes_session_name": args.hermes_session_name,
                        "codex_multi_agent_disabled": True,
                        "current_prompt_mode": retry_worker["prompt_mode"],
                        "current_prompt_override_chars": len(retry_prompt_override),
                        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                )
                append_jsonl(
                    events_path,
                    {
                        "event": "worker_finished",
                        "round": next_round,
                        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        **retry_worker,
                    },
                )
                retry_again_reason = codex_transient_failure_reason(Path(retry_worker["run_dir"]))
                if retry_again_reason:
                    exit_reason = "codex_transient_retry_failed"
                    append_jsonl(
                        events_path,
                        {
                            "event": "codex_transient_retry_failed",
                            "round": next_round,
                            "first_run_id": run_id,
                            "retry_run_id": retry_run_id,
                            "first_reason": retry_reason,
                            "retry_reason": retry_again_reason,
                            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        },
                    )
                    write_stop_request(stop_file, exit_reason)
                    append_jsonl(
                        events_path,
                        {
                            "event": "loop_stop",
                            "round": next_round,
                            "reason": exit_reason,
                            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        },
                    )
                    break
                worker = retry_worker

            run_dir = Path(worker["run_dir"])
            if not (run_dir / "review_input.md").exists():
                exit_reason = f"review_input_missing_after_round_{next_round}"
                append_jsonl(
                    events_path,
                    {
                        "event": "loop_stop",
                        "round": next_round,
                        "reason": exit_reason,
                        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                )
                break

            try:
                review = hermes_review(args, repo, run_dir, loop_dir, next_round, delta, prompt_mode)
            except Exception as exc:  # noqa: BLE001 - keep loop evidence explicit
                review = {
                    "continue": False,
                    "delta": "",
                    "next_prompt": "",
                    "base_prompt_edits": [],
                    "reason": (
                        f"Hermes review exception: {type(exc).__name__}: {exc}; "
                        "supervisor stopped instead of falling back to a Codex-authored Direction Brief."
                    ),
                    "telegram_note": (
                        "Hermes review raised an exception; supervisor stopped."
                    ),
                    "parse_error": True,
                }
            base_prompt_text = read_text(Path(args.base_prompt), max_chars=200000).strip()
            edited_prompt, prompt_edit_results = apply_prompt_edits(
                base_prompt_text,
                review.get("base_prompt_edits", []),
            )
            next_prompt = str(review.get("next_prompt") or "").strip()
            if not next_prompt and prompt_edit_results and any(r.get("applied") for r in prompt_edit_results):
                next_prompt = edited_prompt
            if review.get("continue", True) and not next_prompt:
                review["continue"] = False
                if review.get("parse_error"):
                    review["reason"] = (
                        "Hermes review did not provide valid JSON/next_prompt after retry; "
                        "supervisor stopped instead of using Codex fallback."
                    )
                else:
                    review["reason"] = "Hermes review did not provide required non-empty next_prompt Direction Brief."
            next_prompt_path = loop_dir / f"round_{next_round:04d}_next_prompt.md"
            next_prompt_path.write_text(next_prompt + ("\n" if next_prompt else ""), encoding="utf-8")
            review["prompt_edit_results"] = prompt_edit_results
            review["next_prompt_path"] = str(next_prompt_path)
            review["next_prompt_chars"] = len(next_prompt)
            review_path = loop_dir / f"round_{next_round:04d}_review.json"
            write_json(review_path, review)
            if not review.get("parse_error") and (
                not review.get("continue", True) or bool(next_prompt)
            ):
                review_prompt_path = Path(str(review.get("review_prompt") or ""))
                if not review_prompt_path.exists():
                    review_prompt_path = loop_dir / f"round_{next_round:04d}_hermes_review_prompt.md"
                review["manual_steering_once_consumption"] = consume_manual_steering_once(
                    repo,
                    args,
                    loop_dir,
                    next_round,
                    review_prompt_path,
                )
                write_json(review_path, review)
            if review.get("hermes_session_id") and not args.hermes_session_id:
                args.hermes_session_id = str(review.get("hermes_session_id"))
            write_json(
                loop_dir / "hermes_session.json",
                {
                    "hermes_review_mode": args.hermes_review_mode,
                    "hermes_session_id": args.hermes_session_id,
                    "hermes_session_name": args.hermes_session_name,
                    "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            write_json(
                active_loop_path,
                {
                    "pid": os.getpid(),
                    "loop_id": loop_id,
                    "loop_dir": str(loop_dir),
                    "repo": str(repo),
                    "latest_run_dir": worker["run_dir"],
                    "latest_round": next_round,
                    "stop_file": str(stop_file),
                    "manual_steering_file": str(manual_steering_file),
                    "manual_steering_once_file": str(manual_steering_once_file),
                    "hermes_review_mode": args.hermes_review_mode,
                    "hermes_session_id": args.hermes_session_id,
                    "hermes_session_name": args.hermes_session_name,
                    "codex_multi_agent_disabled": True,
                    "next_prompt_mode": "override" if next_prompt else "delta" if review.get("delta") else "base",
                    "next_prompt_chars": len(next_prompt),
                    "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            append_jsonl(
                events_path,
                {
                    "event": "hermes_review_finished",
                    "round": next_round,
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "continue": review.get("continue", True),
                    "delta": review.get("delta", ""),
                    "next_prompt_path": str(next_prompt_path),
                    "next_prompt_chars": len(next_prompt),
                    "prompt_edit_results": prompt_edit_results,
                    "reason": review.get("reason", ""),
                    "telegram_note": review.get("telegram_note", ""),
                    "hermes_review_mode": review.get("hermes_review_mode", args.hermes_review_mode),
                    "hermes_session_id": review.get("hermes_session_id", args.hermes_session_id),
                    "review_path": str(review_path),
                    "manual_steering_once_consumption": review.get("manual_steering_once_consumption", {}),
                },
            )

            rounds_completed = next_round
            if stop_file.exists():
                exit_reason = f"stop_file_present_after_round_{next_round}"
                break
            if not review.get("continue", True):
                exit_reason = str(review.get("reason") or f"hermes_requested_stop_after_round_{next_round}")
                break
            if next_prompt:
                prompt_override = next_prompt
                delta = ""
                prompt_mode = "override"
            else:
                prompt_override = ""
                delta = str(review.get("delta") or "").strip()
                prompt_mode = "delta" if delta else "base"
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        else:
            if ready_for_worker:
                exit_reason = "loop_condition_finished"
    finally:
        finished = {
            "event": "loop_finished",
            "loop_id": loop_id,
            "rounds_completed": rounds_completed,
            "reason": exit_reason,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        append_jsonl(events_path, finished)
        write_json(loop_dir / "loop_result.json", finished)
        current_active = load_json(active_loop_path)
        if isinstance(current_active, dict) and current_active.get("pid") == os.getpid():
            try:
                active_loop_path.unlink()
            except FileNotFoundError:
                pass
    return 0


def command_status(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    root = artifact_root_path(repo, args.artifact_root)
    stop_file = stop_file_path(repo, args)
    steering_file = manual_steering_path(repo, args)
    steering_once_file = manual_steering_once_path(repo, args)
    lock_path = root / "codex_serial_supervisor.lock"
    active_loop_path = root / ACTIVE_LOOP_FILE_NAME
    lock_content = read_text(lock_path, max_chars=2000).strip() if lock_path.exists() else ""
    active_loop_data = load_json(active_loop_path) if active_loop_path.exists() else None
    active_loop_pid_alive = None
    if isinstance(active_loop_data, dict):
        pid_value = active_loop_data.get("pid")
        if pid_value is not None:
            active_loop_pid_alive = pid_is_alive(str(pid_value))
    status = {
        "repo": str(repo),
        "artifact_root": str(root),
        "active_loop": active_loop_data,
        "active_loop_pid_alive": active_loop_pid_alive,
        "stop_file": {
            "path": str(stop_file),
            "exists": stop_file.exists(),
            "content": read_text(stop_file, max_chars=2000).strip() if stop_file.exists() else "",
        },
        "manual_steering": {
            "path": str(steering_file),
            "exists": steering_file.exists(),
            "content": read_text(steering_file, max_chars=4000).strip() if steering_file.exists() else "",
        },
        "manual_steering_once": {
            "path": str(steering_once_file),
            "exists": steering_once_file.exists(),
            "content": read_text(steering_once_file, max_chars=4000).strip() if steering_once_file.exists() else "",
        },
        "latest_saved_hermes_session": latest_saved_hermes_session(root),
        "worker_lock": {
            "path": str(lock_path),
            "exists": lock_path.exists(),
            "content": lock_content,
            "pid_alive": pid_is_alive(lock_content) if lock_content else None,
        },
        "latest_loops": latest_loop_dirs(root, limit=args.limit),
        "latest_runs": latest_run_dirs(root, limit=args.limit),
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def command_stop(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    path = stop_file_path(repo, args)
    write_stop_request(path, args.reason)
    print(path)
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", default=".", help="Repository root to run in.")
    parser.add_argument(
        "--artifact-root",
        default=REPO_DEFAULT_ARTIFACT_ROOT,
        help="Artifact root, relative to repo unless absolute.",
    )


def add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-prompt", default=str(DEFAULT_BASE_PROMPT))
    parser.add_argument("--delta", default="", help="Legacy short Hermes steering delta for this run.")
    parser.add_argument("--delta-file", default="", help="File containing a legacy steering delta.")
    parser.add_argument("--prompt-override", default="", help="Transient Direction Brief appended below the base prompt for this run.")
    parser.add_argument("--prompt-override-file", default="", help="File containing a transient Direction Brief appended below the base prompt.")
    parser.add_argument("--run-id", default="", help="Override run id.")


def add_codex_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-arg", action="append", default=[], help="Extra raw arg passed to codex exec.")
    parser.add_argument(
        "--model",
        default=DEFAULT_SUPERVISOR_MODEL,
        help="Codex model override. Defaults to the supervisor-required gpt-5.5.",
    )
    parser.add_argument("--profile", default="", help="Codex profile override.")
    parser.add_argument("--keep-proxy", action="store_true", help="Do not strip proxy env vars for Codex.")
    parser.add_argument("--no-exec", action="store_true", help="Prepare artifacts without launching Codex.")
    parser.add_argument(
        "--telegram-event",
        choices=("finish", "start", "both", "off"),
        default="finish",
        help="Telegram notification timing for each Codex run.",
    )
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram notification.")
    parser.add_argument("--telegram-chat-id", default="", help="Override Telegram chat/channel id.")
    parser.add_argument("--telegram-bot-token", default="", help="Override Telegram bot token.")
    parser.add_argument("--telegram-proxy", default="", help="Override Telegram proxy URL.")
    parser.add_argument("--telegram-timeout", type=int, default=20, help="Telegram send timeout in seconds.")


def add_stop_file_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stop-file",
        default="",
        help="Stop file path. Relative paths are resolved from repo. Default: artifact-root/STOP.",
    )


def add_manual_steering_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manual-steering-file",
        default="",
        help="Durable human steering file included in Hermes reviews. Relative paths are resolved from repo. Default: artifact-root/manual_steering.md.",
    )
    parser.add_argument(
        "--manual-steering-once-file",
        default="",
        help="One-shot human steering file included in the next official Hermes review, then archived and cleared. Relative paths are resolved from repo. Default: artifact-root/manual_steering_once.md.",
    )


def add_resume_latest_hermes_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resume-latest-hermes-session",
        dest="resume_latest_hermes_session",
        action="store_true",
        default=True,
        help="Resume the latest saved Hermes session when --hermes-session-id is not provided.",
    )
    parser.add_argument(
        "--no-resume-latest-hermes-session",
        dest="resume_latest_hermes_session",
        action="store_false",
        help="Start without auto-resuming the latest saved Hermes session.",
    )


def add_loop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-prompt", default=str(DEFAULT_BASE_PROMPT))
    parser.add_argument("--loop-id", default="", help="Override loop id.")
    parser.add_argument("--max-rounds", type=int, default=0, help="0 means run until stopped.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Delay between rounds.")
    parser.add_argument("--initial-delta", default="", help="Optional legacy steering delta for the first round.")
    parser.add_argument("--initial-delta-file", default="", help="Optional file containing a first-round legacy delta.")
    parser.add_argument(
        "--resume-review-run-dir",
        default="",
        help=(
            "Run official Hermes review on an existing Codex run directory "
            "before launching the next worker. Use this to resume from a "
            "paused post-worker review point without a human-written initial delta."
        ),
    )
    add_manual_steering_arg(parser)
    parser.add_argument("--review-no-llm", action="store_true", help="Skip Hermes review and always continue.")
    parser.add_argument("--review-max-chars", type=int, default=30000, help="Max chars included in Hermes review prompt.")
    parser.add_argument("--hermes-bin", default="hermes")
    parser.add_argument(
        "--hermes-review-mode",
        choices=("chat", "oneshot"),
        default="chat",
        help="Use a persistent Hermes chat session or legacy one-shot -z review.",
    )
    parser.add_argument(
        "--hermes-session-id",
        default="",
        help="Hermes session id to resume for supervisor review.",
    )
    add_resume_latest_hermes_args(parser)
    parser.add_argument(
        "--hermes-session-name",
        default=DEFAULT_HERMES_SESSION_NAME,
        help="Hermes session title/name to continue when no session id is provided.",
    )
    parser.add_argument("--hermes-source", default="cli", help="Hermes session source tag.")
    parser.add_argument("--hermes-max-turns", type=int, default=50, help="Hermes review max tool-calling turns.")
    parser.add_argument(
        "--hermes-review-retries",
        type=int,
        default=1,
        help="Retry official Hermes review this many times when it does not return strict JSON.",
    )
    parser.add_argument(
        "--hermes-retry-max-turns",
        type=int,
        default=12,
        help="Hermes max tool-calling turns for the short strict-JSON repair retry.",
    )
    parser.add_argument(
        "--codex-transient-retries",
        type=int,
        default=1,
        help=(
            "Retry a Codex worker once when artifacts indicate a transient "
            "API/stream/model failure. If HEAD or git status changed, retry "
            "with a continuation brief over the live repo state and previous "
            "run artifacts."
        ),
    )
    parser.add_argument(
        "--hermes-model",
        default=DEFAULT_SUPERVISOR_MODEL,
        help="Hermes reviewer model override. Defaults to the supervisor-required gpt-5.5.",
    )
    parser.add_argument("--hermes-provider", default="", help="Hermes reviewer provider override.")
    parser.add_argument("--hermes-timeout", type=int, default=900, help="Hermes review timeout in seconds.")
    parser.add_argument(
        "--hermes-skills",
        default="codex-serial-supervisor",
        help="Comma-separated Hermes skills to preload for review. Empty string disables skill preload.",
    )
    parser.add_argument("--hermes-ignore-rules", action="store_true", help="Pass --ignore-rules to Hermes review.")
    parser.add_argument(
        "--no-hermes-pass-session-id",
        dest="hermes_pass_session_id",
        action="store_false",
        help="Do not pass the Hermes session id into the supervisor prompt.",
    )
    parser.set_defaults(hermes_pass_session_id=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_snapshot = sub.add_parser("snapshot", help="Write a repo/Trellis/git/CSV snapshot.")
    add_common(p_snapshot)
    p_snapshot.add_argument("--output", default="-")
    p_snapshot.set_defaults(func=command_snapshot)

    p_prompt = sub.add_parser("prompt", help="Build the Codex prompt without running Codex.")
    add_common(p_prompt)
    add_prompt_args(p_prompt)
    p_prompt.set_defaults(func=command_prompt)

    p_run = sub.add_parser("run", help="Run one Codex worker turn and package evidence.")
    add_common(p_run)
    add_prompt_args(p_run)
    add_codex_run_args(p_run)
    p_run.set_defaults(func=command_run)

    p_start = sub.add_parser("start", help="Start the continuous loop as a detached background process.")
    add_common(p_start)
    add_codex_run_args(p_start)
    add_stop_file_arg(p_start)
    add_loop_args(p_start)
    p_start.add_argument("--keep-stop-file", action="store_true", help="Do not clear an existing stop file before start.")
    p_start.set_defaults(func=command_start)

    p_loop = sub.add_parser("loop", help="Run the continuous Hermes-reviewed serial Codex loop.")
    add_common(p_loop)
    add_codex_run_args(p_loop)
    add_stop_file_arg(p_loop)
    add_loop_args(p_loop)
    p_loop.set_defaults(func=command_loop)

    p_ask = sub.add_parser("ask-hermes", help="Ask Hermes for a read-only supervisor self-check without launching Codex.")
    add_common(p_ask)
    add_manual_steering_arg(p_ask)
    p_ask.add_argument("--base-prompt", default=str(DEFAULT_BASE_PROMPT))
    p_ask.add_argument("--run-id", default="", help="Override ask artifact id.")
    p_ask.add_argument("--question", default="", help="Ask-only question for Hermes.")
    p_ask.add_argument("--question-file", default="", help="File containing an ask-only question for Hermes.")
    p_ask.add_argument("--render-only", action="store_true", help="Write the Hermes ask prompt but do not call Hermes.")
    p_ask.add_argument("--review-max-chars", type=int, default=30000, help="Max chars included in Hermes ask prompt.")
    p_ask.add_argument("--hermes-bin", default="hermes")
    p_ask.add_argument("--hermes-source", default="cli", help="Hermes session source tag.")
    p_ask.add_argument("--hermes-max-turns", type=int, default=50, help="Hermes ask max tool-calling turns.")
    p_ask.add_argument("--hermes-model", default=DEFAULT_SUPERVISOR_MODEL)
    p_ask.add_argument("--hermes-provider", default="")
    p_ask.add_argument("--hermes-timeout", type=int, default=900)
    p_ask.add_argument("--hermes-skills", default="codex-serial-supervisor")
    p_ask.add_argument("--hermes-ignore-rules", action="store_true")
    p_ask.add_argument("--hermes-session-id", default="", help="Hermes session id to resume for ask-only self-check.")
    add_resume_latest_hermes_args(p_ask)
    p_ask.add_argument(
        "--no-hermes-pass-session-id",
        dest="hermes_pass_session_id",
        action="store_false",
        help="Do not pass the Hermes session id into the ask prompt.",
    )
    p_ask.set_defaults(hermes_pass_session_id=True)
    p_ask.set_defaults(func=command_ask_hermes)

    p_status = sub.add_parser("status", help="Show active loop, stop file, worker lock, and recent runs.")
    add_common(p_status)
    add_stop_file_arg(p_status)
    add_manual_steering_arg(p_status)
    p_status.add_argument("--limit", type=int, default=8)
    p_status.set_defaults(func=command_status)

    p_stop = sub.add_parser("stop", help="Request loop stop after the current Codex run finishes.")
    add_common(p_stop)
    add_stop_file_arg(p_stop)
    p_stop.add_argument("--reason", default="manual stop")
    p_stop.set_defaults(func=command_stop)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
