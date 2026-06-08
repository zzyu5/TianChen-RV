#!/usr/bin/env python3
"""Check supervisor prompt contracts that are hard to cover with MLIR lit."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile


def run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.strip()


def load_supervisor(path: Path):
    spec = importlib.util.spec_from_file_location("codex_serial_supervisor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def make_commit(repo: Path, relpath: str, text: str, message: str) -> str:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    run(["git", "add", relpath], repo)
    run(["git", "commit", "-m", message], repo)
    return run(["git", "rev-parse", "HEAD"], repo)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor", required=True, type=Path)
    parser.add_argument("--worker-prompt", required=True, type=Path)
    args = parser.parse_args()
    supervisor = load_supervisor(args.supervisor.resolve())

    with tempfile.TemporaryDirectory(prefix="tcrv-supervisor-contract-") as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        run(["git", "init"], repo)
        run(["git", "config", "user.email", "test@example.invalid"], repo)
        run(["git", "config", "user.name", "Supervisor Contract Test"], repo)
        make_commit(repo, "README.md", "base\n", "base")

        task_dir = repo / ".trellis/tasks/macro-rvv-production-kernel"
        task_dir.mkdir(parents=True)
        (repo / ".trellis/.current-task").parent.mkdir(parents=True, exist_ok=True)
        (repo / ".trellis/.current-task").write_text(
            ".trellis/tasks/macro-rvv-production-kernel\n",
            encoding="utf-8",
        )
        write_json(
            task_dir / "task.json",
            {
                "id": "macro-rvv-production-kernel",
                "title": "RVV production-kernel capability campaign",
                "status": "in_progress",
            },
        )
        (task_dir / "prd.md").write_text(
            "# RVV production-kernel capability campaign\n\n"
            "This is a macro-task with campaign-level gates.\n\n"
            "- [x] Record current drift.\n"
            "- [ ] Build Gearbox selected-body realization milestone.\n"
            "- [ ] Add low-precision contraction primitive surface.\n",
            encoding="utf-8",
        )
        run(["git", "add", ".trellis"], repo)
        run(["git", "commit", "-m", "task: add macro campaign"], repo)

        artifact_root = repo / "artifacts/tmp/hermes_codex_supervisor"
        loop_id = "20260608T000000Z"
        loop_dir = artifact_root / "loops" / loop_id
        events_path = loop_dir / "events.jsonl"
        latest_run_dir = None
        latest_before = None
        latest_after = None

        for round_index in range(1, 4):
            run_dir = artifact_root / "runs" / f"{loop_id}-r{round_index:04d}-20260608T00000{round_index}Z"
            before = supervisor.collect_snapshot(repo)
            make_commit(
                repo,
                f".trellis/tasks/archive/2026-06/evidence-{round_index}/prd.md",
                f"metadata-only generated-bundle evidence {round_index}\n",
                f"rvv: record generated bundle evidence {round_index}",
            )
            after = supervisor.collect_snapshot(repo)
            write_json(run_dir / "snapshot_before.json", before)
            write_json(run_dir / "snapshot_after.json", after)
            write_json(run_dir / "manifest.json", {"codex_exit_code": 0})
            (run_dir / "last_message.md").write_text("metadata-only evidence closeout\n", encoding="utf-8")
            (run_dir / "codex.stderr.log").write_text("", encoding="utf-8")
            next_prompt_path = loop_dir / f"round_{round_index:04d}_next_prompt.md"
            next_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            next_prompt_path.write_text(
                "Direction title:\n"
                f"  Stage2 adjacent generated-bundle artifact seam {round_index}\n",
                encoding="utf-8",
            )
            append_jsonl(
                events_path,
                {
                    "event": "worker_finished",
                    "round": round_index,
                    "run_dir": str(run_dir),
                },
            )
            append_jsonl(
                events_path,
                {
                    "event": "hermes_review_finished",
                    "round": round_index,
                    "next_prompt_path": str(next_prompt_path),
                    "reason": (
                        "metadata-only evidence closeout: production source did not "
                        "need changes; generated bundle proved on ssh rvv."
                    ),
                },
            )
            latest_run_dir = run_dir
            latest_before = before
            latest_after = after

        assert latest_run_dir is not None
        prompt = supervisor.build_review_prompt(
            latest_run_dir,
            loop_dir,
            args.worker_prompt.resolve(),
            3,
            "",
            "override",
            12000,
            manual_steering="",
        )
        required_prompt_fragments = [
            "Macro-Owner Task Discovery Algorithm",
            "Recent Round Drift Summary",
            "recent_drift_escalation_required: yes",
            "RVV production-kernel capability campaign",
            "Gearbox/resource-aware selected-body realization",
            "low-precision contraction primitive surface",
            "llama.cpp q8/q4",
        ]
        for fragment in required_prompt_fragments:
            assert fragment in prompt, fragment

        supervisor.write_review_input(latest_run_dir, latest_before, latest_after, 0, "")
        review_input = (latest_run_dir / "review_input.md").read_text(encoding="utf-8")
        for fragment in (
            "Current Trellis Task Progress",
            "Recent Round Drift Summary",
            "metadata_only_signal: yes",
            "evidence_closeout_signal: yes",
        ):
            assert fragment in review_input, fragment

    worker_prompt = args.worker_prompt.read_text(encoding="utf-8")
    for fragment in (
        "Macro-Task Continuation Contract",
        "Leave `.trellis/.current-task` active",
        "RVV production-kernel capability campaign",
    ):
        assert fragment in worker_prompt, fragment

    print("supervisor macro-owner prompt contract ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
