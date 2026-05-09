#!/usr/bin/env python3
"""Drive bounded explicit RVV microkernel source-export evidence.

This helper is runner/evidence tooling only. It orchestrates existing
TianChen-RV MLIR tools and optional ``ssh rvv`` compile/run evidence for the
explicit ``tcrv_rvv.i32_vadd_microkernel`` first slice. It does not implement
compiler IR, plugin decisions, capability modeling, lowering, emission, runtime
ABI, correctness logic, or performance measurement.
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


SCRIPT_NAME = "tianchenrv-rvv-microkernel-e2e"
SCHEMA_VERSION = 1
DEFAULT_INPUT = Path("test/Target/EmissionManifest/emission-manifest-rvv-microkernel.mlir")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/tmp/rvv_microkernel_e2e")
DEFAULT_SSH_TARGET = "rvv"
DEFAULT_TIMEOUT_SECONDS = 60
SUCCESS_MARKER = "tcrv_rvv_microkernel_ok"
EXTERNAL_ABI_SUCCESS_MARKER = "tcrv_rvv_microkernel_external_abi_ok"

EXPECTED_DATAFLOW_PROVENANCE = {
    "dataflow_abi_roles": [
        "lhs_load.buffer_role=lhs-input-buffer",
        "rhs_load.buffer_role=rhs-input-buffer",
        "store.buffer_role=output-buffer",
        "runtime n remains the target/export-owned runtime element-count ABI parameter",
    ],
    "dataflow_emission_step[0]": [
        "op=tcrv_rvv.i32_load",
        "role=lhs-input-buffer",
        "result=lhs_vec",
    ],
    "dataflow_emission_step[1]": [
        "op=tcrv_rvv.i32_load",
        "role=rhs-input-buffer",
        "result=rhs_vec",
    ],
    "dataflow_emission_step[2]": [
        "op=tcrv_rvv.i32_add",
        "lhs=lhs_vec",
        "rhs=rhs_vec",
        "result=sum_vec",
    ],
    "dataflow_emission_step[3]": [
        "op=tcrv_rvv.i32_store",
        "role=output-buffer",
        "value=sum_vec",
    ],
}

REQUIRED_HANDOFF = {
    "origin": "rvv-plugin",
    "emission_status": "supported",
    "emission_kind": "rvv-explicit-i32-vadd-microkernel-c-source",
    "lowering_pipeline": "tcrv-export-rvv-microkernel-c",
    "lowering_boundary": "tcrv_rvv.lowering_boundary",
    "runtime_abi": "rvv-i32-vadd-runtime-callable-c-abi.v1",
    "runtime_abi_kind": "rvv-runtime-callable-c-abi",
    "runtime_abi_name": "rvv-i32-vadd-runtime-callable-c-function.v1",
    "runtime_glue_role": "runtime-callable-i32-vadd-function",
    "artifact_kind": "runtime-callable-c-source",
}

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
                f"artifact directory already exists; pass --overwrite or use a new --run-id: "
                f"{relative_to_repo(artifact_dir, root)}"
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

    build_candidates: list[Path] = []
    for candidate in (
        root / "artifacts" / "tmp" / "tianchenrv-build" / "bin" / tool_name,
        root / "build" / "bin" / tool_name,
    ):
        if candidate.exists() and os.access(candidate, os.X_OK):
            build_candidates.append(candidate)

    if build_candidates:
        newest = max(build_candidates, key=lambda path: path.stat().st_mtime)
        return str(newest.relative_to(root))

    raise BridgeError(
        f"could not find {tool_name}; pass --{tool_name.replace('-', '_')} or build the project"
    )


def ensure_local_clang_on_path() -> str:
    existing = shutil.which("clang")
    if existing:
        return existing

    for candidate in (
        Path("/usr/lib/llvm-20/bin/clang"),
        Path("/usr/lib/llvm-19/bin/clang"),
        Path("/usr/lib/llvm-18/bin/clang"),
        Path("/usr/lib/llvm-17/bin/clang"),
    ):
        if candidate.exists() and os.access(candidate, os.X_OK):
            os.environ["PATH"] = (
                str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")
            )
            return str(candidate)

    raise BridgeError(
        "could not find clang for local RVV object export; install clang or "
        "put an LLVM clang directory on PATH"
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


def run_command_stdout_to_file(
    name: str,
    command: list[str],
    output_path: Path,
    *,
    cwd: Path,
    artifact_dir: Path,
    commands: list[dict[str, Any]],
    timeout_seconds: int,
) -> int:
    started = time.monotonic()
    timed_out = False
    stdout_note = f"<binary stdout redirected to {relative_to_repo(output_path, cwd)}>"
    try:
        with output_path.open("wb") as stdout_handle:
            completed = subprocess.run(
                command,
                cwd=cwd,
                stdout=stdout_handle,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        exit_code: int | None = completed.returncode
        stderr = completed.stderr.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired as timeout:
        timed_out = True
        exit_code = None
        stderr_bytes = timeout.stderr or b""
        if isinstance(stderr_bytes, str):
            stderr = stderr_bytes
        else:
            stderr = stderr_bytes.decode("utf-8", errors="replace")
        stderr += f"\n[timed out after {timeout_seconds}s]"
    except OSError as error:
        exit_code = None
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
        stdout_note,
        stderr,
    )
    commands.append(
        {
            "name": name,
            "command": command_display(command),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_seconds": round(duration, 3),
            "stdout_sha256": sha256_text(stdout_note),
            "stderr_sha256": sha256_text(sanitize_text(stderr)),
            "log_path": log_path,
        }
    )
    if exit_code != 0 or timed_out:
        raise BridgeError(
            f"command failed: {name}; see {log_path} for sanitized stdout/stderr"
        )
    return int(exit_code)


def write_generated_text(path: Path, context: str, text: str) -> None:
    reject_secret_like_text(context, text)
    path.write_text(text, encoding="utf-8")


def parse_manifest_value(raw: str) -> str:
    value = raw.strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        return value[1:-1].replace(r"\"", '"').replace(r"\\", "\\")
    if value.startswith("@"):
        return value[1:]
    return value


def parse_manifest_paths(manifest_text: str) -> list[dict[str, str]]:
    paths: list[dict[str, str]] = []
    current_kernel = ""
    current_path: dict[str, str] | None = None
    for line in manifest_text.splitlines():
        if line.startswith("kernel @"):
            current_kernel = line.removeprefix("kernel @").strip()
            current_path = None
            continue
        if re.match(r"^  path\[[0-9]+\]:$", line):
            current_path = {"kernel": current_kernel}
            paths.append(current_path)
            continue
        if current_path is None:
            continue
        match = re.match(r"^    ([A-Za-z0-9_]+):\s*(.*)$", line)
        if not match:
            continue
        key, raw_value = match.groups()
        if key == "preference":
            current_path = None
            continue
        current_path[key] = parse_manifest_value(raw_value)
    return paths


def find_supported_handoff(manifest_text: str) -> dict[str, str]:
    reject_secret_like_text("emission manifest", manifest_text)
    matches: list[dict[str, str]] = []
    for path in parse_manifest_paths(manifest_text):
        if all(path.get(key) == value for key, value in REQUIRED_HANDOFF.items()):
            matches.append(path)

    if not matches:
        raise BridgeError(
            "manifest missing supported bounded RVV microkernel source-export handoff"
        )
    if len(matches) != 1:
        raise BridgeError(
            "manifest must contain exactly one supported bounded RVV microkernel handoff"
        )
    return matches[0]


def parse_source_comment(source: str, field: str, *, required: bool) -> str:
    match = re.search(rf"/\*\s*{re.escape(field)}:\s*([^*]+?)\s*\*/", source)
    if not match:
        if required:
            raise BridgeError(f"generated C source missing comment field: {field}")
        return ""
    value = match.group(1).strip()
    reject_secret_like_text(f"generated C source field {field}", value)
    return value


def validate_generated_source(source: str, *, require_harness: bool) -> dict[str, str]:
    if not source.strip():
        raise BridgeError("generated RVV microkernel C source is empty")
    reject_secret_like_text("generated RVV microkernel C source", source)
    required_snippets = [
        "#include <riscv_vector.h>",
        "__riscv_vsetvl_e32m1",
        "__riscv_vle32_v_i32m1",
        "__riscv_vadd_vv_i32m1",
        "__riscv_vse32_v_i32m1",
    ]
    if require_harness:
        required_snippets.extend(
            [
                "int main(void)",
                SUCCESS_MARKER,
                "runtime_counts=",
                "_self_check_one(size_t runtime_n)",
            ]
        )
    elif "int main(void)" in source or SUCCESS_MARKER in source:
        raise BridgeError(
            "default generated RVV microkernel C source must be library-style "
            "runtime-callable source without the self-check harness"
        )
    missing = [snippet for snippet in required_snippets if snippet not in source]
    if missing:
        raise BridgeError(
            "generated RVV microkernel C source missing required snippets: "
            + ", ".join(missing)
        )
    selected_march = parse_source_comment(source, "selected_march", required=True)
    selected_mabi = parse_source_comment(source, "selected_mabi", required=False)
    if "v" not in selected_march.lower():
        raise BridgeError("selected_march from generated source must contain RVV vector evidence")
    provenance = validate_dataflow_provenance(source)
    return {
        "selected_march": selected_march,
        "selected_mabi": selected_mabi,
        "dataflow_provenance": provenance,
    }


def validate_dataflow_provenance(source: str) -> dict[str, str]:
    provenance: dict[str, str] = {}
    for field, required_fragments in EXPECTED_DATAFLOW_PROVENANCE.items():
        value = parse_source_comment(source, field, required=True)
        missing = [fragment for fragment in required_fragments if fragment not in value]
        if missing:
            raise BridgeError(
                "generated RVV microkernel C source field "
                f"{field} is not the expected dataflow-driven exporter provenance; "
                "missing fragments: " + ", ".join(missing)
            )
        provenance[field] = value
    return provenance


def validate_generated_header(header: str) -> str:
    if not header.strip():
        raise BridgeError("generated RVV microkernel C header is empty")
    reject_secret_like_text("generated RVV microkernel C header", header)
    forbidden_snippets = [
        "int main",
        "_self_check",
        "__riscv",
        "riscv_vector",
        "runtime_success",
        "throughput",
        "latency",
        "artifacts/tmp",
        "tcrv_rvv_microkernel_ok",
    ]
    for snippet in forbidden_snippets:
        if snippet in header:
            raise BridgeError(
                f"generated RVV microkernel C header contains forbidden snippet: {snippet}"
            )
    for snippet in ("#ifndef ", "#define ", "#include <stddef.h>", "#include <stdint.h>"):
        if snippet not in header:
            raise BridgeError(
                f"generated RVV microkernel C header missing required snippet: {snippet}"
            )
    prototypes = re.findall(
        r"(?m)^\s*void\s+([A-Za-z_][A-Za-z0-9_]*)\s*"
        r"\(\s*const\s+int32_t\s*\*\s*lhs\s*,\s*"
        r"const\s+int32_t\s*\*\s*rhs\s*,\s*"
        r"int32_t\s*\*\s*out\s*,\s*size_t\s+n\s*\)\s*;\s*$",
        header,
    )
    if len(prototypes) != 1:
        raise BridgeError(
            "generated RVV microkernel C header must contain exactly one "
            "runtime-callable i32-vadd prototype"
        )
    if re.search(r"(?m)^\s*void\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*\)\s*\{", header):
        raise BridgeError("generated RVV microkernel C header must not contain a function body")
    return prototypes[0]


def build_external_caller_source(function_name: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", function_name):
        raise BridgeError("generated header function name is not a valid C identifier")
    return f"""\
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "rvv_microkernel.h"

int main(void) {{
  enum {{ kElements = 16 }};
  const int32_t lhs[kElements] = {{0, 1, 2, 3, 4, 5, 6, 7,
                                  8, 9, 10, 11, 12, 13, 14, 15}};
  const int32_t rhs[kElements] = {{31, 29, 23, 19, 17, 13, 11, 7,
                                  5, 3, 2, 1, -1, -3, -5, -7}};
  int32_t out[kElements] = {{0}};

  {function_name}(lhs, rhs, out, (size_t)kElements);
  for (size_t index = 0; index < (size_t)kElements; ++index) {{
    if (out[index] != lhs[index] + rhs[index]) {{
      fprintf(stderr, "rvv microkernel external ABI mismatch at %zu\\n", index);
      return 3;
    }}
  }}

  printf("{EXTERNAL_ABI_SUCCESS_MARKER} elements=%zu\\n", (size_t)kElements);
  return 0;
}}
"""


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


def build_remote_compile_command(remote_dir: str, flags: dict[str, str]) -> str:
    quoted_flags = " ".join(shlex.quote(part) for part in remote_compile_flags(flags))
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        f"clang {quoted_flags} rvv_microkernel.c -o rvv_microkernel"
    )


def build_remote_external_link_command(remote_dir: str, flags: dict[str, str]) -> str:
    quoted_flags = " ".join(shlex.quote(part) for part in remote_compile_flags(flags))
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        f"clang {quoted_flags} rvv_microkernel_external_caller.c "
        "rvv_microkernel.o -o rvv_microkernel_external_caller"
    )


def remote_sha256_command(remote_dir: str, filename: str) -> str:
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        "if command -v sha256sum >/dev/null 2>&1; then "
        f"set -- $(sha256sum {shlex.quote(filename)}); printf '%s\\n' \"$1\"; "
        "else printf '\\n'; fi"
    )


def first_sanitized_line(text: str) -> str:
    for line in sanitize_text(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def run_remote_self_check_source_evidence(
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
    remote_dir = f"/tmp/tianchenrv_rvv_microkernel_e2e_{safe_run_id(run_id)}"
    remote_source = f"{remote_dir}/rvv_microkernel.c"

    setup_command = f"rm -rf {quote_remote_path(remote_dir)} && mkdir -p {quote_remote_path(remote_dir)}"
    run_command(
        "ssh_setup_remote_dir",
        remote_shell_command(args, setup_command),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_command(
        "scp_generated_microkernel_self_check_source",
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
        "ssh_compile_microkernel_self_check",
        remote_shell_command(args, build_remote_compile_command(remote_dir, flags)),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    executable_hash_stdout, _, _ = run_command(
        "ssh_executable_sha256",
        remote_shell_command(args, remote_sha256_command(remote_dir, "rvv_microkernel")),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_stdout, _, _ = run_command(
        "ssh_run_microkernel_self_check",
        remote_shell_command(args, f"cd {quote_remote_path(remote_dir)} && ./rvv_microkernel"),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    if SUCCESS_MARKER not in run_stdout:
        raise BridgeError(
            f"remote microkernel self-check stdout missing expected marker: {SUCCESS_MARKER}"
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
        "remote_source": "rvv_microkernel.c",
        "remote_executable": "rvv_microkernel",
        "compile_flags": remote_compile_flags(flags),
        "compile_exit_code": 0,
        "run_exit_code": 0,
        "expected_stdout_marker": SUCCESS_MARKER,
        "stdout_marker_observed": True,
        "executable_sha256": first_sanitized_line(executable_hash_stdout),
        "cleanup_status": cleanup_status,
    }


def run_remote_external_abi_evidence(
    args: argparse.Namespace,
    *,
    root: Path,
    artifact_dir: Path,
    commands: list[dict[str, Any]],
    header_path: Path,
    object_path: Path,
    caller_path: Path,
    flags: dict[str, str],
    run_id: str,
) -> dict[str, Any]:
    reject_secret_like_text("ssh target", args.ssh_target)
    remote_dir = f"/tmp/tianchenrv_rvv_microkernel_external_abi_{safe_run_id(run_id)}"

    setup_command = f"rm -rf {quote_remote_path(remote_dir)} && mkdir -p {quote_remote_path(remote_dir)}"
    run_command(
        "ssh_setup_remote_dir",
        remote_shell_command(args, setup_command),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_command(
        "scp_external_abi_inputs",
        [
            *scp_base_command(args),
            str(header_path),
            str(object_path),
            str(caller_path),
            f"{args.ssh_target}:{remote_dir}/",
        ],
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    compile_command = build_remote_external_link_command(remote_dir, flags)
    run_command(
        "ssh_compile_external_header_object_caller",
        remote_shell_command(args, compile_command),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    binary_hash_stdout, _, _ = run_command(
        "ssh_binary_sha256",
        remote_shell_command(
            args,
            f"cd {quote_remote_path(remote_dir)} && "
            "if command -v sha256sum >/dev/null 2>&1; then "
            "set -- $(sha256sum rvv_microkernel_external_caller); printf '%s\\n' \"$1\"; "
            "else printf '\\n'; fi",
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_stdout, _, _ = run_command(
        "ssh_run_external_header_object_caller",
        remote_shell_command(
            args,
            f"cd {quote_remote_path(remote_dir)} && ./rvv_microkernel_external_caller",
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    if EXTERNAL_ABI_SUCCESS_MARKER not in run_stdout:
        raise BridgeError(
            f"remote external caller stdout missing expected marker: {EXTERNAL_ABI_SUCCESS_MARKER}"
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
        "compile_exit_code": 0,
        "run_exit_code": 0,
        "expected_stdout_marker": EXTERNAL_ABI_SUCCESS_MARKER,
        "stdout_marker_observed": True,
        "binary_sha256": first_sanitized_line(binary_hash_stdout),
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
    use_harness = args.self_check_harness
    if args.generic_route and args.self_check_harness:
        raise BridgeError(
            "--self-check-harness is explicit direct-export evidence mode and "
            "cannot be combined with --generic-route"
        )
    if args.generic_route and not args.dry_run:
        raise BridgeError(
            "--generic-route exports the default runtime-callable library "
            "source; ssh evidence uses the generated header plus generated "
            "object external caller"
        )

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
    local_clang = (
        ensure_local_clang_on_path()
        if not args.dry_run and not use_harness
        else ""
    )

    post_planning_mlir, _, _ = run_command(
        "tcrv_opt_materialize_plans",
        [
            tcrv_opt,
            str(input_path),
            "--tcrv-materialize-selected-lowering-boundaries",
            "--tcrv-materialize-emission-plans",
        ],
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    post_planning_path = artifact_dir / "post_planning.mlir"
    write_generated_text(
        post_planning_path, "post-planning MLIR", post_planning_mlir
    )

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
    manifest_path = artifact_dir / "emission_manifest.txt"
    write_generated_text(manifest_path, "emission manifest", manifest_text)
    manifest_handoff = find_supported_handoff(manifest_text)

    if args.generic_route:
        source_export_flag = "--tcrv-export-target-source-artifact"
        source_export_name = "export_target_source_artifact"
    elif use_harness:
        source_export_flag = "--tcrv-export-rvv-microkernel-self-check-c"
        source_export_name = "export_rvv_microkernel_self_check_c"
    else:
        source_export_flag = "--tcrv-export-rvv-microkernel-c"
        source_export_name = "export_rvv_microkernel_c"
    source_text, _, _ = run_command(
        source_export_name,
        [
            tcrv_translate,
            source_export_flag,
            str(post_planning_path),
        ],
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    source_path = artifact_dir / "rvv_microkernel.c"
    source_flags = validate_generated_source(
        source_text, require_harness=use_harness
    )
    write_generated_text(source_path, "generated RVV microkernel source", source_text)

    header_path = artifact_dir / "rvv_microkernel.h"
    object_path = artifact_dir / "rvv_microkernel.o"
    caller_path = artifact_dir / "rvv_microkernel_external_caller.c"
    header_function_name = ""
    object_sha256 = ""
    header_sha256 = ""
    caller_sha256 = ""
    if not args.dry_run and not use_harness:
        header_text, _, _ = run_command(
            "export_target_header_artifact",
            [
                tcrv_translate,
                "--tcrv-export-target-header-artifact",
                str(post_planning_path),
            ],
            cwd=root,
            artifact_dir=artifact_dir,
            commands=commands,
            timeout_seconds=args.timeout,
        )
        header_function_name = validate_generated_header(header_text)
        write_generated_text(
            header_path, "generated RVV microkernel header", header_text
        )
        header_sha256 = sha256_text(header_text)

        run_command_stdout_to_file(
            "export_target_object_artifact",
            [
                tcrv_translate,
                "--tcrv-export-target-artifact",
                str(post_planning_path),
            ],
            object_path,
            cwd=root,
            artifact_dir=artifact_dir,
            commands=commands,
            timeout_seconds=args.timeout,
        )
        if object_path.stat().st_size < 4 or object_path.read_bytes()[:4] != b"\x7fELF":
            raise BridgeError("generated RVV microkernel object must be a non-empty ELF relocatable")
        object_sha256 = sha256_file(object_path)

        caller_text = build_external_caller_source(header_function_name)
        write_generated_text(
            caller_path, "generated RVV microkernel external caller", caller_text
        )
        caller_sha256 = sha256_text(caller_text)

    hashes = {
        "input_sha256": sha256_file(input_path),
        "post_planning_mlir_sha256": sha256_text(post_planning_mlir),
        "emission_manifest_sha256": sha256_text(manifest_text),
        "rvv_microkernel_c_sha256": sha256_text(source_text),
    }
    if not args.dry_run and not use_harness:
        hashes.update(
            {
                "rvv_microkernel_h_sha256": header_sha256,
                "rvv_microkernel_o_sha256": object_sha256,
                "rvv_microkernel_external_caller_c_sha256": caller_sha256,
            }
        )
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runner": SCRIPT_NAME,
        "run_id": run_id,
        "mode": "dry-run" if args.dry_run else "ssh",
        "status": "success",
        "input": relative_to_repo(input_path, root),
        "artifact_dir": relative_to_repo(artifact_dir, root),
        "manifest_handoff": True,
        "manifest_record": manifest_handoff,
        "source_export_flag": source_export_flag,
        "source_export_route": (
            "generic-target-source-artifact"
            if args.generic_route
            else "direct-rvv-microkernel-self-check-harness"
            if use_harness
            else "direct-rvv-microkernel"
        ),
        "source_export_mode": (
            "self-check-harness" if use_harness else "runtime-callable-library"
        ),
        "selected_compile_flags": [
            "-O2",
            f"-march={source_flags['selected_march']}",
            *(
                [f"-mabi={source_flags['selected_mabi']}"]
                if source_flags.get("selected_mabi")
                else []
            ),
        ],
        "source_dataflow_provenance": source_flags["dataflow_provenance"],
        "hashes": hashes,
        "artifacts": {
            "post_planning_mlir": relative_to_repo(post_planning_path, root),
            "emission_manifest": relative_to_repo(manifest_path, root),
            "rvv_microkernel_c": relative_to_repo(source_path, root),
        },
        "commands": commands,
        "ssh_evidence": None,
        "claim_scope": (
            "local dry-run verifies compiler-tool handoff and source export only"
            if args.dry_run
            else "bounded generated RVV i32-vadd self-check executable correctness only"
            if use_harness
            else "bounded generated RVV i32-vadd header plus object external caller correctness only"
        ),
    }
    if not args.dry_run:
        if use_harness:
            evidence["self_check"] = {
                "kind": "generated-c-self-check-main",
                "success_marker": SUCCESS_MARKER,
            }
        else:
            evidence["artifacts"].update(
                {
                    "rvv_microkernel_h": relative_to_repo(header_path, root),
                    "rvv_microkernel_o": relative_to_repo(object_path, root),
                    "rvv_microkernel_external_caller_c": relative_to_repo(
                        caller_path, root
                    ),
                }
            )
            evidence["header_function_name"] = header_function_name
            evidence["local_object_export_clang"] = sanitize_text(local_clang)

    if not args.dry_run:
        try:
            if use_harness:
                evidence["ssh_evidence"] = run_remote_self_check_source_evidence(
                    args,
                    root=root,
                    artifact_dir=artifact_dir,
                    commands=commands,
                    source_path=source_path,
                    flags=source_flags,
                    run_id=run_id,
                )
            else:
                evidence["ssh_evidence"] = run_remote_external_abi_evidence(
                    args,
                    root=root,
                    artifact_dir=artifact_dir,
                    commands=commands,
                    header_path=header_path,
                    object_path=object_path,
                    caller_path=caller_path,
                    flags=source_flags,
                    run_id=run_id,
                )
            evidence["commands"] = commands
        except BridgeError as error:
            evidence["status"] = "failure"
            evidence["error"] = sanitize_text(str(error))
            evidence["commands"] = commands
            write_json(artifact_dir / "command_summary.json", {
                "artifact_dir": relative_to_repo(artifact_dir, root),
                "commands": commands,
            })
            write_json(artifact_dir / "hashes.json", hashes)
            write_json(artifact_dir / "evidence.json", evidence)
            raise

    write_json(artifact_dir / "command_summary.json", {
        "artifact_dir": relative_to_repo(artifact_dir, root),
        "commands": commands,
    })
    write_json(artifact_dir / "hashes.json", hashes)
    write_json(artifact_dir / "evidence.json", evidence)
    return evidence


def assert_self_test(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_self_test() -> None:
    supported_manifest = """
tianchenrv.emission_manifest.version: 1
module: "self_test"
kernel_count: 1
kernel @rvv_microkernel_manifest
  selected_surface: selected-marker
  path[0]:
    selected_variant: @rvv_first_slice
    role: "direct variant"
    origin: "rvv-plugin"
    emission_status: "supported"
    emission_kind: "rvv-explicit-i32-vadd-microkernel-c-source"
    lowering_pipeline: "tcrv-export-rvv-microkernel-c"
    lowering_boundary: "tcrv_rvv.lowering_boundary"
    runtime_abi: "rvv-i32-vadd-runtime-callable-c-abi.v1"
    runtime_abi_kind: "rvv-runtime-callable-c-abi"
    runtime_abi_name: "rvv-i32-vadd-runtime-callable-c-function.v1"
    runtime_glue_role: "runtime-callable-i32-vadd-function"
    artifact_kind: "runtime-callable-c-source"
    required_capabilities: [@rvv]
    explanation: "bounded"
""".strip()
    record = find_supported_handoff(supported_manifest)
    assert_self_test(
        record["kernel"] == "rvv_microkernel_manifest",
        "supported handoff parser lost kernel name",
    )

    missing_manifest = supported_manifest.replace(
        'emission_status: "supported"', 'emission_status: "unsupported"'
    )
    try:
        find_supported_handoff(missing_manifest)
    except BridgeError as error:
        assert_self_test("missing supported bounded" in str(error), "missing handoff error changed")
    else:
        raise AssertionError("manifest without supported handoff was accepted")

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

    root = repo_root()
    try:
        require_under_artifacts_tmp(root / "rvv_microkernel.c", root)
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
        summary = {"artifact_dir": relative_to_repo(artifact_dir, root), "commands": commands}
        summary_text = json.dumps(summary, sort_keys=True)
        assert_self_test("live-token" not in summary_text, "command summary leaked token")
        log_text = (artifact_dir / commands[0]["log_path"]).read_text(encoding="utf-8")
        assert_self_test("live-token" not in log_text, "command log leaked token")
        assert_self_test("visible=ok" in log_text, "command log lost non-secret text")

    sample_source = """\
/* selected_march: rv64gcv */
/* selected_mabi: lp64d */
/* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, rhs_load.buffer_role=rhs-input-buffer, store.buffer_role=output-buffer; runtime n remains the target/export-owned runtime element-count ABI parameter */
/* dataflow_emission_step[0]: op=tcrv_rvv.i32_load, role=lhs-input-buffer, result=lhs_vec */
/* dataflow_emission_step[1]: op=tcrv_rvv.i32_load, role=rhs-input-buffer, result=rhs_vec */
/* dataflow_emission_step[2]: op=tcrv_rvv.i32_add, lhs=lhs_vec, rhs=rhs_vec, result=sum_vec */
/* dataflow_emission_step[3]: op=tcrv_rvv.i32_store, role=output-buffer, value=sum_vec */
#include <riscv_vector.h>
void f(void) {
  __riscv_vsetvl_e32m1;
  __riscv_vle32_v_i32m1;
  __riscv_vadd_vv_i32m1;
  __riscv_vse32_v_i32m1;
}
static int f_self_check_one(size_t runtime_n) { return runtime_n == 0; }
int main(void) { puts("tcrv_rvv_microkernel_ok runtime_counts=7,16"); }
"""
    source_flags = validate_generated_source(sample_source, require_harness=True)
    assert_self_test(
        source_flags["dataflow_provenance"]["dataflow_emission_step[2]"]
        == "op=tcrv_rvv.i32_add, lhs=lhs_vec, rhs=rhs_vec, result=sum_vec",
        "dataflow provenance parser lost add step",
    )
    try:
        validate_generated_source(
            sample_source.replace("dataflow_emission_step[3]", "stale_step[3]"),
            require_harness=True,
        )
    except BridgeError as error:
        assert_self_test(
            "dataflow_emission_step[3]" in str(error),
            "missing dataflow provenance error changed",
        )
    else:
        raise AssertionError("source without complete dataflow provenance was accepted")

    remote_command = build_remote_compile_command(
        "/tmp/tianchenrv_rvv_microkernel_e2e_self_test",
        {"selected_march": "rv64gcv", "selected_mabi": "lp64d"},
    )
    assert_self_test("-march=rv64gcv" in remote_command, "remote march flag missing")
    assert_self_test("-mabi=lp64d" in remote_command, "remote mabi flag missing")

    header = """\
#ifndef TIANCHENRV_RVV_I32_VADD_MICROKERNEL_SELF_TEST_H
#define TIANCHENRV_RVV_I32_VADD_MICROKERNEL_SELF_TEST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void tcrv_rvv_i32_vadd_microkernel_self_test(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* TIANCHENRV_RVV_I32_VADD_MICROKERNEL_SELF_TEST_H */
"""
    function_name = validate_generated_header(header)
    caller = build_external_caller_source(function_name)
    assert_self_test(
        EXTERNAL_ABI_SUCCESS_MARKER in caller,
        "external ABI caller success marker missing",
    )
    external_link_command = build_remote_external_link_command(
        "/tmp/tianchenrv_rvv_microkernel_external_abi_self_test",
        {"selected_march": "rv64gcv", "selected_mabi": "lp64d"},
    )
    assert_self_test(
        "rvv_microkernel_external_caller.c rvv_microkernel.o" in external_link_command,
        "external ABI link command does not consume caller and object",
    )

    print("rvv_microkernel_e2e self-test passed")


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
    parser.add_argument(
        "--generic-route",
        action="store_true",
        help="Export generated C through the generic target source artifact route",
    )
    parser.add_argument(
        "--self-check-harness",
        action="store_true",
        help="Use the explicit RVV microkernel self-check harness export",
    )
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
        print(f"rvv_microkernel_e2e: {sanitize_text(str(error))}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "artifact_dir": evidence["artifact_dir"],
                "mode": evidence["mode"],
                "status": evidence["status"],
                "manifest_handoff": evidence["manifest_handoff"],
                "source_sha256": evidence["hashes"]["rvv_microkernel_c_sha256"],
                "source_export_mode": evidence["source_export_mode"],
                "source_export_route": evidence["source_export_route"],
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
