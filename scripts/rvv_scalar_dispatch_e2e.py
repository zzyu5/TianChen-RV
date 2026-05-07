#!/usr/bin/env python3
"""Drive bounded RVV+scalar dispatch executable evidence.

This helper is runner/evidence tooling only. It orchestrates existing
TianChen-RV MLIR tools and optional ``ssh rvv`` compile/link/run evidence for
the planned RVV+scalar i32-vadd dispatch self-check slice. It does not
implement compiler IR, plugin decisions, target selection, capability modeling,
lowering, emission, runtime ABI, correctness logic, or performance measurement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


SCRIPT_NAME = "tianchenrv-rvv-scalar-dispatch-e2e"
SCHEMA_VERSION = 1
DEFAULT_INPUT = Path(
    "test/Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-generic-route.mlir"
)
DEFAULT_ARTIFACT_ROOT = Path("artifacts/tmp/rvv_scalar_dispatch_e2e")
DEFAULT_SSH_TARGET = "rvv"
DEFAULT_TIMEOUT_SECONDS = 60
SUCCESS_MARKER = "tcrv_rvv_scalar_i32_vadd_dispatch_self_check_ok"

SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?is)-----BEGIN [A-Z0-9 _-]*PRIVATE KEY-----.*?"
            r"-----END [A-Z0-9 _-]*PRIVATE KEY-----"
        ),
        "[REDACTED PRIVATE KEY]",
    ),
    (
        re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r"(?i)\b((?:[A-Z0-9_]*"
            r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|PRIVATE_KEY)"
            r"[A-Z0-9_]*)\s*=\s*)[^\s]+"
        ),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r"(?i)\b((?:token|secret|password|passwd|api[_-]?key|"
            r"access[_-]?key|private[_ -]?key)\s*[:=]\s*)[^\s]+"
        ),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(?i)\b((?:https?|socks5?)_?proxy\s*=\s*)[^\s]+"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(?i)\bhttps?://[^\s\"']+"),
        "[REDACTED URL]",
    ),
)


class BridgeError(RuntimeError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_run_id(raw_run_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_run_id.strip())
    return sanitized or utc_run_id()


def safe_file_component(raw: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip()).strip("._")
    return sanitized or "command"


def sanitize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    sanitized = str(text).replace("\x00", "\\0")
    for pattern, replacement in SECRET_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def reject_secret_like_text(context: str, text: Any) -> None:
    original = "" if text is None else str(text)
    if sanitize_text(original) != original:
        raise BridgeError(f"{context} contains secret-like, credential, or URL text")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_repo(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def resolve_repo_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def require_under_artifacts_tmp(path: Path, root: Path) -> None:
    artifacts_tmp = (root / "artifacts" / "tmp").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_tmp)
    except ValueError as error:
        raise BridgeError(
            f"artifact path must be nested under artifacts/tmp: {path}"
        ) from error


def prepare_artifact_dir(
    artifact_root: Path, run_id: str, root: Path, overwrite: bool
) -> Path:
    resolved_root = resolve_repo_path(artifact_root, root)
    require_under_artifacts_tmp(resolved_root, root)
    artifact_dir = resolved_root / safe_run_id(run_id)
    require_under_artifacts_tmp(artifact_dir, root)
    if artifact_dir.exists():
        if not overwrite:
            raise BridgeError(
                "artifact directory already exists; pass --overwrite or use a "
                f"new --run-id: {relative_to_repo(artifact_dir, root)}"
            )
        shutil.rmtree(artifact_dir)
    (artifact_dir / "logs").mkdir(parents=True, exist_ok=False)
    return artifact_dir


def resolve_tool(explicit: str, tool_name: str, root: Path) -> str:
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            raise BridgeError(f"{tool_name} path does not exist: {explicit}")
        if not os.access(candidate, os.X_OK):
            raise BridgeError(f"{tool_name} path is not executable: {explicit}")
        return str(candidate)

    if shutil.which(tool_name):
        return tool_name

    for candidate in (
        root / "build" / "bin" / tool_name,
        root / "artifacts" / "tmp" / "tianchenrv-build" / "bin" / tool_name,
    ):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate.relative_to(root))

    raise BridgeError(
        f"could not find {tool_name}; pass --{tool_name.replace('-', '_')} or build the project"
    )


def command_display(command: list[str]) -> str:
    return sanitize_text(shlex.join(command))


def write_command_log(
    artifact_dir: Path,
    index: int,
    name: str,
    command: list[str],
    exit_code: int | None,
    timed_out: bool,
    duration_seconds: float,
    stdout: str,
    stderr: str,
) -> str:
    log_path = artifact_dir / "logs" / f"{index:03d}_{safe_file_component(name)}.log"
    log_path.write_text(
        "\n".join(
            [
                f"name: {name}",
                f"command: {command_display(command)}",
                f"exit_code: {exit_code}",
                f"timed_out: {str(timed_out).lower()}",
                f"duration_seconds: {duration_seconds:.3f}",
                "--- stdout ---",
                sanitize_text(stdout),
                "--- stderr ---",
                sanitize_text(stderr),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(log_path.relative_to(artifact_dir))


def run_command(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    artifact_dir: Path,
    commands: list[dict[str, Any]],
    timeout_seconds: int,
) -> tuple[str, str, int]:
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
        exit_code: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as timeout:
        timed_out = True
        exit_code = None
        stdout = timeout.stdout or ""
        stderr = (timeout.stderr or "") + f"\n[timed out after {timeout_seconds}s]"
    except OSError as error:
        exit_code = None
        stdout = ""
        stderr = str(error)

    duration = time.monotonic() - started
    log_path = write_command_log(
        artifact_dir,
        len(commands),
        name,
        command,
        exit_code,
        timed_out,
        duration,
        stdout,
        stderr,
    )
    commands.append(
        {
            "name": name,
            "command": command_display(command),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_seconds": round(duration, 3),
            "stdout_sha256": sha256_text(sanitize_text(stdout)),
            "stderr_sha256": sha256_text(sanitize_text(stderr)),
            "log_path": log_path,
        }
    )
    if exit_code != 0 or timed_out:
        raise BridgeError(
            f"command failed: {name}; see {log_path} for sanitized stdout/stderr"
        )
    return stdout, stderr, int(exit_code)


def write_generated_text(path: Path, context: str, text: str) -> None:
    reject_secret_like_text(context, text)
    path.write_text(text, encoding="utf-8")


def parse_source_comment(source: str, field: str, *, required: bool) -> str:
    match = re.search(rf"/\*\s*{re.escape(field)}:\s*([^*]+?)\s*\*/", source)
    if not match:
        if required:
            raise BridgeError(f"generated dispatch C source missing comment field: {field}")
        return ""
    value = match.group(1).strip()
    reject_secret_like_text(f"generated dispatch C source field {field}", value)
    return value


def validate_dispatch_manifest(manifest_text: str) -> None:
    reject_secret_like_text("emission manifest", manifest_text)
    required = [
        'selected_surface: dispatch',
        'origin: "rvv-plugin"',
        'role: "dispatch case"',
        'lowering_pipeline: "tcrv-export-rvv-microkernel-c"',
        'runtime_abi_kind: "rvv-runtime-callable-c-abi"',
        'origin: "scalar-plugin"',
        'role: "dispatch fallback"',
        'lowering_pipeline: "tcrv-export-scalar-microkernel-c"',
        'runtime_abi_kind: "scalar-runtime-callable-c-abi"',
    ]
    missing = [snippet for snippet in required if snippet not in manifest_text]
    if missing:
        raise BridgeError(
            "emission manifest missing planned RVV+scalar dispatch handoff snippets: "
            + ", ".join(missing)
        )


def validate_library_dispatch_source(source: str) -> None:
    if not source.strip():
        raise BridgeError("generated library dispatch C source is empty")
    reject_secret_like_text("generated library dispatch C source", source)
    if "int main(void)" in source or SUCCESS_MARKER in source:
        raise BridgeError(
            "generic target source artifact must remain library-style without "
            "a self-check main or success marker"
        )
    required = [
        "/* TianChen-RV RVV+scalar host runtime dispatch C export. */",
        "/* Runtime guard: explicit host-provided rvv_available parameter; no automatic hardware probe is generated. */",
        "__riscv_vadd_vv_i32m1",
        "out[index] = lhs[index] + rhs[index];",
        "void tcrv_dispatch_i32_vadd_",
        "if (rvv_available)",
    ]
    missing = [snippet for snippet in required if snippet not in source]
    if missing:
        raise BridgeError(
            "generated library dispatch C source missing required snippets: "
            + ", ".join(missing)
        )


def validate_self_check_dispatch_source(source: str) -> dict[str, str]:
    if not source.strip():
        raise BridgeError("generated dispatch self-check C source is empty")
    reject_secret_like_text("generated dispatch self-check C source", source)
    required = [
        "/* TianChen-RV RVV+scalar host runtime dispatch C export. */",
        "/* Explicit bounded self-check harness for RVV+scalar dispatch runtime invocation evidence. */",
        "rvv_available = 0",
        "rvv_available = 1",
        "int main(void)",
        SUCCESS_MARKER,
        "__riscv_vadd_vv_i32m1",
        "out[index] = lhs[index] + rhs[index];",
        "tcrv_dispatch_i32_vadd_",
    ]
    missing = [snippet for snippet in required if snippet not in source]
    if missing:
        raise BridgeError(
            "generated dispatch self-check C source missing required snippets: "
            + ", ".join(missing)
        )
    forbidden = ["runtime_success", "throughput", "latency", "artifacts/tmp"]
    leaked = [snippet for snippet in forbidden if snippet in source]
    if leaked:
        raise BridgeError(
            "generated dispatch self-check C source includes forbidden claim text: "
            + ", ".join(leaked)
        )
    selected_march = parse_source_comment(source, "selected_march", required=True)
    selected_mabi = parse_source_comment(source, "selected_mabi", required=False)
    if "v" not in selected_march.lower():
        raise BridgeError("selected_march from generated source must contain RVV vector evidence")
    return {"selected_march": selected_march, "selected_mabi": selected_mabi}


def quote_remote_path(path: str) -> str:
    return shlex.quote(path)


def ssh_base_command(args: argparse.Namespace) -> list[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
    ]
    for option in args.ssh_option:
        reject_secret_like_text("ssh option", option)
        command.extend(["-o", option])
    command.append(args.ssh_target)
    return command


def scp_base_command(args: argparse.Namespace) -> list[str]:
    command = [
        "scp",
        "-q",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
    ]
    for option in args.ssh_option:
        reject_secret_like_text("ssh option", option)
        command.extend(["-o", option])
    return command


def remote_shell_command(args: argparse.Namespace, remote_command: str) -> list[str]:
    return [
        *ssh_base_command(args),
        shlex.join(["sh", "-lc", remote_command]),
    ]


def remote_compile_flags(flags: dict[str, str]) -> list[str]:
    result = ["-O2", f"-march={flags['selected_march']}"]
    if flags.get("selected_mabi"):
        result.append(f"-mabi={flags['selected_mabi']}")
    return result


def build_remote_compile_object_command(remote_dir: str, flags: dict[str, str]) -> str:
    quoted_flags = " ".join(shlex.quote(flag) for flag in remote_compile_flags(flags))
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        f"clang {quoted_flags} -c rvv_scalar_dispatch_self_check.c "
        "-o rvv_scalar_dispatch_self_check.o"
    )


def build_remote_link_executable_command(remote_dir: str, flags: dict[str, str]) -> str:
    quoted_flags = " ".join(shlex.quote(flag) for flag in remote_compile_flags(flags))
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        f"clang {quoted_flags} rvv_scalar_dispatch_self_check.o "
        "-o rvv_scalar_dispatch_self_check"
    )


def remote_sha256_command(remote_dir: str, filename: str) -> str:
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        "if command -v sha256sum >/dev/null 2>&1; then "
        f"set -- $(sha256sum {shlex.quote(filename)}); printf '%s\\n' \"$1\"; "
        "else printf '\\n'; fi"
    )


def run_remote_evidence(
    args: argparse.Namespace,
    *,
    root: Path,
    artifact_dir: Path,
    commands: list[dict[str, Any]],
    source_path: Path,
    flags: dict[str, str],
    run_id: str,
) -> dict[str, Any]:
    reject_secret_like_text("ssh target", args.ssh_target)
    remote_dir = f"/tmp/tianchenrv_rvv_scalar_dispatch_e2e_{safe_run_id(run_id)}"
    remote_source = f"{remote_dir}/rvv_scalar_dispatch_self_check.c"

    setup_command = (
        f"rm -rf {quote_remote_path(remote_dir)} && "
        f"mkdir -p {quote_remote_path(remote_dir)}"
    )
    run_command(
        "ssh_setup_remote_dir",
        remote_shell_command(args, setup_command),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_command(
        "scp_generated_dispatch_source",
        [
            *scp_base_command(args),
            str(source_path),
            f"{args.ssh_target}:{remote_source}",
        ],
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_command(
        "ssh_compile_dispatch_object",
        remote_shell_command(args, build_remote_compile_object_command(remote_dir, flags)),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    object_hash_stdout, _, _ = run_command(
        "ssh_object_sha256",
        remote_shell_command(
            args, remote_sha256_command(remote_dir, "rvv_scalar_dispatch_self_check.o")
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_command(
        "ssh_link_dispatch_executable",
        remote_shell_command(args, build_remote_link_executable_command(remote_dir, flags)),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    executable_hash_stdout, _, _ = run_command(
        "ssh_executable_sha256",
        remote_shell_command(
            args, remote_sha256_command(remote_dir, "rvv_scalar_dispatch_self_check")
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_stdout, _, _ = run_command(
        "ssh_run_dispatch_self_check",
        remote_shell_command(
            args,
            f"cd {quote_remote_path(remote_dir)} && ./rvv_scalar_dispatch_self_check",
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    if SUCCESS_MARKER not in run_stdout:
        raise BridgeError(
            f"remote dispatch self-check stdout missing expected marker: {SUCCESS_MARKER}"
        )

    cleanup_status = "success"
    try:
        run_command(
            "ssh_cleanup_remote_dir",
            remote_shell_command(args, f"rm -rf {quote_remote_path(remote_dir)}"),
            cwd=root,
            artifact_dir=artifact_dir,
            commands=commands,
            timeout_seconds=args.timeout,
        )
    except BridgeError:
        cleanup_status = "failure"

    return {
        "ssh_target": args.ssh_target,
        "remote_dir": remote_dir,
        "compile_flags": remote_compile_flags(flags),
        "object_compile_exit_code": 0,
        "link_exit_code": 0,
        "run_exit_code": 0,
        "expected_stdout_marker": SUCCESS_MARKER,
        "stdout_marker_observed": True,
        "object_sha256": sanitize_text(object_hash_stdout).strip().splitlines()[0]
        if sanitize_text(object_hash_stdout).strip()
        else "",
        "executable_sha256": sanitize_text(executable_hash_stdout)
        .strip()
        .splitlines()[0]
        if sanitize_text(executable_hash_stdout).strip()
        else "",
        "cleanup_status": cleanup_status,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    reject_secret_like_text(path.name, text)
    path.write_text(text, encoding="utf-8")


def run_bridge(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    if args.evidence_note:
        reject_secret_like_text("evidence note", args.evidence_note)

    run_id = safe_run_id(args.run_id or utc_run_id())
    artifact_dir = prepare_artifact_dir(
        Path(args.artifact_root), run_id, root, args.overwrite
    )
    commands: list[dict[str, Any]] = []

    input_path = resolve_repo_path(Path(args.input), root)
    if not input_path.exists():
        raise BridgeError(f"input MLIR does not exist: {args.input}")
    reject_secret_like_text("input MLIR path", str(args.input))

    tcrv_opt = resolve_tool(args.tcrv_opt, "tcrv-opt", root)
    tcrv_translate = resolve_tool(args.tcrv_translate, "tcrv-translate", root)

    post_planning_mlir, _, _ = run_command(
        "tcrv_opt_execution_planning_pipeline",
        [
            tcrv_opt,
            str(input_path),
            "--tcrv-execution-planning-pipeline",
        ],
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    post_planning_path = artifact_dir / "post_planning.mlir"
    write_generated_text(post_planning_path, "post-planning MLIR", post_planning_mlir)

    manifest_text, _, _ = run_command(
        "export_emission_manifest",
        [
            tcrv_translate,
            "--tcrv-export-emission-manifest",
            str(post_planning_path),
        ],
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    validate_dispatch_manifest(manifest_text)
    manifest_path = artifact_dir / "emission_manifest.txt"
    write_generated_text(manifest_path, "emission manifest", manifest_text)

    library_source_text, _, _ = run_command(
        "export_dispatch_library_source",
        [
            tcrv_translate,
            "--tcrv-export-target-source-artifact",
            str(post_planning_path),
        ],
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    validate_library_dispatch_source(library_source_text)
    library_source_path = artifact_dir / "rvv_scalar_dispatch_library.c"
    write_generated_text(
        library_source_path,
        "generated RVV+scalar dispatch library source",
        library_source_text,
    )

    self_check_source_text, _, _ = run_command(
        "export_dispatch_self_check_source",
        [
            tcrv_translate,
            "--tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c",
            str(post_planning_path),
        ],
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    source_flags = validate_self_check_dispatch_source(self_check_source_text)
    self_check_source_path = artifact_dir / "rvv_scalar_dispatch_self_check.c"
    write_generated_text(
        self_check_source_path,
        "generated RVV+scalar dispatch self-check source",
        self_check_source_text,
    )

    hashes = {
        "input_sha256": sha256_file(input_path),
        "post_planning_mlir_sha256": sha256_text(post_planning_mlir),
        "emission_manifest_sha256": sha256_text(manifest_text),
        "dispatch_library_c_sha256": sha256_text(library_source_text),
        "dispatch_self_check_c_sha256": sha256_text(self_check_source_text),
    }
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runner": SCRIPT_NAME,
        "run_id": run_id,
        "mode": "dry-run" if args.dry_run else "ssh",
        "status": "success",
        "input": relative_to_repo(input_path, root),
        "artifact_dir": relative_to_repo(artifact_dir, root),
        "planned_dispatch_pipeline": "tcrv-execution-planning-pipeline",
        "library_source_export_route": "generic-target-source-artifact",
        "self_check_source_export_route": "direct-rvv-scalar-dispatch-self-check",
        "source_export_mode": "self-check-harness",
        "selected_compile_flags": remote_compile_flags(source_flags),
        "hashes": hashes,
        "artifacts": {
            "post_planning_mlir": relative_to_repo(post_planning_path, root),
            "emission_manifest": relative_to_repo(manifest_path, root),
            "dispatch_library_c": relative_to_repo(library_source_path, root),
            "dispatch_self_check_c": relative_to_repo(self_check_source_path, root),
        },
        "commands": commands,
        "ssh_evidence": None,
        "claim_scope": (
            "local dry-run verifies planned dispatch, manifest handoff, and source export only"
            if args.dry_run
            else "bounded RVV+scalar i32-vadd dispatch self-check executable runtime only"
        ),
    }

    if not args.dry_run:
        try:
            evidence["ssh_evidence"] = run_remote_evidence(
                args,
                root=root,
                artifact_dir=artifact_dir,
                commands=commands,
                source_path=self_check_source_path,
                flags=source_flags,
                run_id=run_id,
            )
            evidence["commands"] = commands
        except BridgeError as error:
            evidence["status"] = "failure"
            evidence["error"] = sanitize_text(str(error))
            evidence["commands"] = commands
            write_json(
                artifact_dir / "command_summary.json",
                {"artifact_dir": relative_to_repo(artifact_dir, root), "commands": commands},
            )
            write_json(artifact_dir / "hashes.json", hashes)
            write_json(artifact_dir / "evidence.json", evidence)
            raise

    write_json(
        artifact_dir / "command_summary.json",
        {"artifact_dir": relative_to_repo(artifact_dir, root), "commands": commands},
    )
    write_json(artifact_dir / "hashes.json", hashes)
    write_json(artifact_dir / "evidence.json", evidence)
    return evidence


def assert_self_test(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_self_test() -> None:
    unsafe_text = (
        "Authorization: Bearer abc.def.ghi\n"
        "PASSWORD=hunter2\n"
        "token: live-token\n"
        "https://proxy.example.invalid/path\n"
        "visible=ok\n"
    )
    sanitized = sanitize_text(unsafe_text)
    assert_self_test("abc.def.ghi" not in sanitized, "bearer token was not redacted")
    assert_self_test("hunter2" not in sanitized, "password was not redacted")
    assert_self_test("live-token" not in sanitized, "token was not redacted")
    assert_self_test("proxy.example" not in sanitized, "URL was not redacted")
    assert_self_test("visible=ok" in sanitized, "non-secret text should remain")

    try:
        reject_secret_like_text("self-test", unsafe_text)
    except BridgeError:
        pass
    else:
        raise AssertionError("secret-like metadata was accepted")

    sample_source = """
/* TianChen-RV RVV+scalar host runtime dispatch C export. */
/* Runtime guard: explicit host-provided rvv_available parameter; no automatic hardware probe is generated. */
/* selected_march: rv64gcv */
/* selected_mabi: lp64d */
#include <riscv_vector.h>
void tcrv_dispatch_i32_vadd_self_test(void) {}
void f(void) { __riscv_vadd_vv_i32m1; out[index] = lhs[index] + rhs[index]; }
/* Explicit bounded self-check harness for RVV+scalar dispatch runtime invocation evidence. */
/* Harness scope: calls the generated dispatcher once with rvv_available = 0 and once with rvv_available = 1. */
int main(void) { puts("tcrv_rvv_scalar_i32_vadd_dispatch_self_check_ok"); }
""".strip()
    flags = validate_self_check_dispatch_source(sample_source)
    assert_self_test(flags["selected_march"] == "rv64gcv", "selected march parser failed")
    assert_self_test(flags["selected_mabi"] == "lp64d", "selected mabi parser failed")

    compile_command = build_remote_compile_object_command(
        "/tmp/tianchenrv_rvv_scalar_dispatch_e2e_self_test", flags
    )
    assert_self_test("-march=rv64gcv" in compile_command, "compile march flag missing")
    assert_self_test("-mabi=lp64d" in compile_command, "compile mabi flag missing")
    assert_self_test(" -c rvv_scalar_dispatch_self_check.c " in compile_command, "object compile command missing -c")
    link_command = build_remote_link_executable_command(
        "/tmp/tianchenrv_rvv_scalar_dispatch_e2e_self_test", flags
    )
    assert_self_test("rvv_scalar_dispatch_self_check.o" in link_command, "link command missing object input")
    assert_self_test(" -o rvv_scalar_dispatch_self_check" in link_command, "link command missing executable output")

    root = repo_root()
    try:
        require_under_artifacts_tmp(root / "rvv_scalar_dispatch_self_check.c", root)
    except BridgeError:
        pass
    else:
        raise AssertionError("repository-root artifact path was accepted")

    with tempfile.TemporaryDirectory(dir=root / "artifacts" / "tmp") as temp_dir:
        artifact_dir = Path(temp_dir)
        (artifact_dir / "logs").mkdir()
        commands: list[dict[str, Any]] = []
        run_command(
            "secret_redaction_fixture",
            ["bash", "-lc", "printf 'TOKEN=live-token\\nvisible=ok\\n'"],
            cwd=root,
            artifact_dir=artifact_dir,
            commands=commands,
            timeout_seconds=10,
        )
        summary_text = json.dumps({"commands": commands}, sort_keys=True)
        assert_self_test("live-token" not in summary_text, "command summary leaked token")
        log_text = (artifact_dir / commands[0]["log_path"]).read_text(encoding="utf-8")
        assert_self_test("live-token" not in log_text, "command log leaked token")
        assert_self_test("visible=ok" in log_text, "command log lost non-secret text")

    print("rvv_scalar_dispatch_e2e self-test passed")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tcrv-opt", default="")
    parser.add_argument("--tcrv-translate", default="")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true", help="Do not contact ssh rvv")
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--ssh-option", action="append", default=[])
    parser.add_argument("--evidence-note", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0

    try:
        evidence = run_bridge(args)
    except BridgeError as error:
        print(f"rvv_scalar_dispatch_e2e: {sanitize_text(str(error))}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "artifact_dir": evidence["artifact_dir"],
                "mode": evidence["mode"],
                "status": evidence["status"],
                "planned_dispatch_pipeline": evidence["planned_dispatch_pipeline"],
                "source_sha256": evidence["hashes"]["dispatch_self_check_c_sha256"],
                "source_export_mode": evidence["source_export_mode"],
                "self_check_source_export_route": evidence[
                    "self_check_source_export_route"
                ],
                "ssh_evidence": bool(evidence["ssh_evidence"]),
                "claim_scope": evidence["claim_scope"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
