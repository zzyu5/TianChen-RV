#!/usr/bin/env python3
"""Run serial Codex worker turns and package evidence for Hermes review.

In loop mode the runner supervises the process while a persistent Hermes chat
session reviews each finished Codex run and decides whether the next worker
should receive the canonical base prompt or a transient full prompt rewrite
derived from that base prompt.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
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
ACTIVE_LOOP_FILE_NAME = "active_loop.json"
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


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def csv_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "error": str(exc)}
    summary: dict[str, Any] = {"exists": True, "rows": len(rows)}
    if path.name == "provider29_four_ratio.csv":
        active = [r for r in rows if r.get("active_import") == "true"]
        first_missing: dict[str, int] = {}
        states: dict[str, int] = {}
        for row in rows:
            key = row.get("first_missing_layer") or "<none>"
            first_missing[key] = first_missing.get(key, 0) + 1
            state = row.get("provider_state") or "<none>"
            states[state] = states.get(state, 0) + 1
        summary.update(
            {
                "active_import_rows": len(active),
                "provider_state_counts": states,
                "first_missing_layer_counts": first_missing,
                "active_frontier_rows": [
                    {
                        "provider": r.get("provider"),
                        "kernel": r.get("kernel"),
                        "state": r.get("provider_state"),
                        "first_missing_layer": r.get("first_missing_layer"),
                        "local": r.get("local_paired_status"),
                        "remote": r.get("remote_paired_status"),
                        "four_gate": r.get("four_gate_status"),
                        "blocker": r.get("blocker"),
                    }
                    for r in active
                    if r.get("first_missing_layer") or r.get("blocker")
                ][:20],
            }
        )
    return summary


def json_artifact_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    data = load_json(path)
    if isinstance(data, dict):
        keys = sorted(str(k) for k in data.keys())
        return {
            "exists": True,
            "path": str(path),
            "top_level_keys": keys[:40],
        }
    return {"exists": True, "path": str(path), "parseable_json_object": False}


def collect_snapshot(repo: Path) -> dict[str, Any]:
    workspace_index = repo / ".trellis" / "workspace" / "index.md"
    provider_csv = repo / "experiments" / "provider29_four_ratio.csv"
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
        "experiments": {
            "provider29_four_ratio": csv_summary(provider_csv),
        },
        "artifacts": {
            "provider29_optguide": json_artifact_summary(repo / "artifacts" / "provider29_optguide" / "summary.json"),
            "provider29_four_ratio": json_artifact_summary(repo / "artifacts" / "provider29_four_ratio" / "summary.json"),
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
        return prompt_override.strip() + "\n"
    if not delta.strip():
        return base + "\n"
    return (
        "# Hermes Supervisor Delta\n\n"
        + delta.strip()
        + "\n\n---\n\n"
        + base
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
        ("predoc capability pack", run_shell(repo, "find predoc/tianchen_rv_mlir_capability_pack -maxdepth 1 -type f | sort | sed -n '1,120p'"), 8000),
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
        "## Experiment Summary After",
        "",
        "```json",
        json.dumps(after.get("experiments", {}), ensure_ascii=False, indent=2)[:12000],
        "```",
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
        "- If continuing, Hermes must generate a full transient next prompt from the base prompt and current evidence.",
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
            "reason": "Hermes review output was not valid JSON; continuing is blocked until a full next_prompt is available.",
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
) -> str:
    max_chars = max(max_chars, 12000)
    review_input = read_text(run_dir / "review_input.md", max_chars=int(max_chars * 0.30))
    repo_audit = read_text(run_dir / "repo_audit.md", max_chars=int(max_chars * 0.34))
    last_message = read_text(run_dir / "last_message.md", max_chars=int(max_chars * 0.14))
    canonical_base_prompt = read_text(base_prompt, max_chars=int(max_chars * 0.20))
    manifest = read_text(run_dir / "manifest.json", max_chars=6000)
    loop_history = tail_text(loop_dir / "events.jsonl", max_chars=int(max_chars * 0.08))
    stderr_tail = tail_text(run_dir / "codex.stderr.log", max_chars=int(max_chars * 0.06))
    return f"""你是 Hermes 的串行 supervisor LLM。你不写 TianchenRV 代码，也不替 Codex 完成 implementation。

你的任务是审查刚结束的 Codex worker run，然后决定下一轮发给 Codex worker 的完整 prompt。

核心机制：
- 下面给出 canonical base prompt。它只是第一轮 seed 和改写参考，不要要求落盘修改它。
- 每轮继续执行时，你都必须基于 canonical base prompt 和本轮 evidence 生成下一轮完整 `next_prompt`。
- 你有权改写完整下一轮 prompt；3 选 1 只是 Codex worker 默认防牛角尖机制，不是证据已经指向架构清理时的死规则。
- 不要把上一轮临时 prompt 当作继续变异的来源；每次都从 canonical base prompt 重新改。
- 不要用空 `next_prompt` 表达“继续照旧”；继续时也要返回完整 prompt。
- `base_prompt_edits` 和 `delta` 只保留给 legacy/debug fallback；默认不要用。

重要行为约束：
- 不要调用工具；下面已经给出 review_input.md、repo_audit.md、last_message.md、manifest 和 stderr tail。`repo_audit.md` 是 runner 从真实 `/home/kingdom/phdworks/TianchenRV` checkout 生成的只读审计，若它和 Codex final message 冲突，以 repo_audit 为准。
- 不要输出 pass/repair/drift/blocked 这类状态标签给用户。
- 这个 loop 默认要持续执行；只有出现明确停止请求、runner 产物缺失、外部凭据/环境需要人工介入、或连续失败会让下一轮无意义时，才把 continue 设为 false。
- 如果上一轮基本对齐 TianchenRV 主线，也要返回一个完整 `next_prompt`；可以基本等同 canonical base prompt，但必须由你显式生成。
- 如果上一轮跑偏、任务太小、偏 metadata/test/status/report-only、track 选择失衡、未 finish/archive current TianchenRV Trellis task，修改下一轮 prompt 本身，而不是只追加 delta。
- 如果 evidence 显示项目在堆砌 docs/tests/tools/scaffold 而没有 active code/schema/build/RVV evidence，下一轮 prompt 必须改写为 bounded active-owner milestone，要求落到 capability model、`tcrv.exec` contract、plugin registry、RVV probe/emission 或 variant pipeline 中的真实 owner，并给出最小验证和 clean commit。
- 对 test/tool/harness/trace/provenance/feedback plumbing 反复扩张要严格：没有真实 runtime/ratio/backend implementation 或 evidence-producing performance work 时，不要空放。
- 每轮 Codex worker 必须是 full-access non-TUI `codex exec`，且不得使用 subagent、spawned agent、parallel agent 或 multi-agent workflow；runner 会强制传入 `--disable multi_agent`。
- 第一阶段监督目标：把空 repo 变成可执行、可验证、可持续演进的 TianchenRV MLIR 项目骨架，包含 capability/profile、plugin registry、`tcrv.exec` contract、RVV remote probe 与最小 validation。
- 第二阶段监督目标：长期推进 RVV plugin、variant generation/legality/selection、lowering/emission/runtime glue，并以 `ssh rvv` 作为真实 correctness/performance evidence。
- 绝不能把 TianchenRV 写成新的高层 tensor/tile IR；`tcrv.exec` 必须 compute-free，RVV/IME/offload/future extension 细节必须 plugin-local。
- 远端 RVV evidence 必须来自真实 `ssh rvv`，不能把 local compile-only 或 smoke-only 包装成 RVV runtime 通过。

只返回一个 JSON 对象，不要加 Markdown。返回必须能被 Python `json.loads` 直接解析；`next_prompt` 必须是单个 JSON string，所有换行都必须编码为 `\\n`，不要在字符串内部输出未转义的真实换行，不要代码块，不要前后解释：
{{
  "continue": true,
  "next_prompt": "继续时必须填入下一轮完整 Codex prompt；停止时可留空。",
  "base_prompt_edits": [
    {{"find": "canonical base prompt 中唯一出现的一段原文", "replace": "替换后的文字"}},
    {{"insert_after": "canonical base prompt 中唯一出现的标题或段落", "text": "要插入的临时规则"}}
  ],
  "delta": "",
  "reason": "简短说明为什么需要或不需要临时 prompt rewrite",
  "telegram_note": "可选；给用户看的极短说明，没有就留空"
}}

## Loop Context

- round_index: {round_index}
- run_dir: {run_dir}

## Previous Delta

{previous_delta or "(none)"}

## Previous Prompt Mode

{previous_prompt_mode or "base"}

## Canonical Base Prompt (read-only)

{canonical_base_prompt or "(missing)"}

## Loop History Tail

{loop_history or "(empty)"}

## Run Manifest

{manifest or "(missing)"}

## review_input.md

{review_input or "(missing)"}

## repo_audit.md

{repo_audit or "(missing)"}

## last_message.md

{last_message or "(missing)"}

## stderr tail

{stderr_tail or "(empty)"}
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
        base_prompt_text = read_text(Path(args.base_prompt), max_chars=200000).strip()
        return {
            "continue": True,
            "delta": "",
            "next_prompt": base_prompt_text,
            "base_prompt_edits": [],
            "reason": "review_no_llm was set; using the base prompt as a smoke-test fallback.",
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
    )
    review_prompt_path = loop_dir / f"round_{round_index:04d}_hermes_review_prompt.md"
    review_prompt_path.write_text(prompt, encoding="utf-8")

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
        cmd.extend(["-z", prompt])
    else:
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

    proc = subprocess.run(
        cmd,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.hermes_timeout,
    )
    (loop_dir / f"round_{round_index:04d}_hermes.stdout").write_text(proc.stdout, encoding="utf-8")
    (loop_dir / f"round_{round_index:04d}_hermes.stderr").write_text(proc.stderr, encoding="utf-8")
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
            "cmd": [*cmd[:3], "<prompt>", *cmd[4:]] if args.hermes_review_mode == "chat" else cmd[:-1] + ["<prompt>"],
            "returncode": proc.returncode,
            "review_prompt": str(review_prompt_path),
            "hermes_review_mode": args.hermes_review_mode,
            "hermes_session_id": hermes_session_id or args.hermes_session_id,
            "hermes_session_name": args.hermes_session_name,
            "hermes_session_lookup": session_lookup,
            "hermes_session_rename": rename_result,
        }
    )
    if proc.returncode != 0:
        review["continue"] = True
        review["delta"] = ""
        review["next_prompt"] = ""
        review["base_prompt_edits"] = []
        review["parse_error"] = True
        review["reason"] = (
            f"Hermes review failed with exit code {proc.returncode}: {proc.stderr[-1000:]}"
            " Runner will continue with canonical base prompt fallback."
        )
        review["telegram_note"] = (
            "Hermes review failed; supervisor is continuing with canonical fallback prompt."
        )
    return review


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
    ]
    if args.stop_file:
        cmd.extend(["--stop-file", args.stop_file])
    if args.initial_delta:
        cmd.extend(["--initial-delta", args.initial_delta])
    if args.initial_delta_file:
        cmd.extend(["--initial-delta-file", args.initial_delta_file])
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
        "hermes_review_mode": args.hermes_review_mode,
        "hermes_session_id": args.hermes_session_id,
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
    loop_id = args.loop_id or utc_run_id()
    loop_dir = root / "loops" / loop_id
    loop_dir.mkdir(parents=True, exist_ok=False)
    stop_file = stop_file_path(repo, args)
    active_loop_path = root / ACTIVE_LOOP_FILE_NAME
    events_path = loop_dir / "events.jsonl"
    delta = read_initial_delta(args)
    prompt_override = ""
    prompt_mode = "delta" if delta.strip() else "base"
    rounds_completed = 0
    exit_reason = "max_rounds_reached"
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
            "max_rounds": args.max_rounds,
            "hermes_review_mode": args.hermes_review_mode,
            "hermes_session_id": args.hermes_session_id,
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
            "max_rounds": args.max_rounds,
            "review_no_llm": args.review_no_llm,
            "hermes_review_mode": args.hermes_review_mode,
            "hermes_session_id": args.hermes_session_id,
            "hermes_session_name": args.hermes_session_name,
            "codex_multi_agent_disabled": True,
            "initial_prompt_mode": prompt_mode,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    print(loop_dir, flush=True)

    try:
        while args.max_rounds <= 0 or rounds_completed < args.max_rounds:
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
                    "continue": True,
                    "delta": "",
                    "next_prompt": "",
                    "base_prompt_edits": [],
                    "reason": (
                        f"Hermes review exception: {type(exc).__name__}: {exc}; "
                        "runner will continue with canonical base prompt fallback."
                    ),
                    "telegram_note": (
                        "Hermes review raised an exception; supervisor is continuing with canonical fallback prompt."
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
                if review.get("parse_error"):
                    next_prompt = (
                        base_prompt_text
                        + "\n\n## Supervisor Review Parse Fallback\n\n"
                        + "Hermes returned malformed JSON in the previous review, so the runner is continuing with the canonical base prompt instead of stopping. "
                        + "Before choosing work, inspect the latest `repo_audit.md`, `review_input.md`, git history, TianchenRV specs, predoc, and current code. "
                        + "Do not redo completed scaffolding. Return to the highest-value capability model, tcrv.exec, plugin registry, RVV probe, variant pipeline, or lowering/runtime milestone. "
                        + "Maintain full-access serial execution, no subagents, focused validation, optional TianchenRV Trellis finish/archive when used, and a clean commit.\n"
                    )
                    review["reason"] = (
                        "Hermes review output was malformed JSON; runner used canonical base prompt with parse-fallback steering."
                    )
                else:
                    review["continue"] = False
                    review["reason"] = "Hermes review did not provide required full next_prompt."
            next_prompt_path = loop_dir / f"round_{next_round:04d}_next_prompt.md"
            next_prompt_path.write_text(next_prompt + ("\n" if next_prompt else ""), encoding="utf-8")
            review["prompt_edit_results"] = prompt_edit_results
            review["next_prompt_path"] = str(next_prompt_path)
            review["next_prompt_chars"] = len(next_prompt)
            write_json(loop_dir / f"round_{next_round:04d}_review.json", review)
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
                    "review_path": str(loop_dir / f"round_{next_round:04d}_review.json"),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "requested_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pid": os.getpid(),
        "reason": args.reason,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    parser.add_argument("--prompt-override", default="", help="Full transient prompt override for this run.")
    parser.add_argument("--prompt-override-file", default="", help="File containing a full transient prompt override.")
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


def add_loop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-prompt", default=str(DEFAULT_BASE_PROMPT))
    parser.add_argument("--loop-id", default="", help="Override loop id.")
    parser.add_argument("--max-rounds", type=int, default=0, help="0 means run until stopped.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Delay between rounds.")
    parser.add_argument("--initial-delta", default="", help="Optional legacy steering delta for the first round.")
    parser.add_argument("--initial-delta-file", default="", help="Optional file containing a first-round legacy delta.")
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
    parser.add_argument(
        "--hermes-session-name",
        default=DEFAULT_HERMES_SESSION_NAME,
        help="Hermes session title/name to continue when no session id is provided.",
    )
    parser.add_argument("--hermes-source", default="cli", help="Hermes session source tag.")
    parser.add_argument("--hermes-max-turns", type=int, default=8, help="Hermes review max tool-calling turns.")
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

    p_status = sub.add_parser("status", help="Show active loop, stop file, worker lock, and recent runs.")
    add_common(p_status)
    add_stop_file_arg(p_status)
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
