#!/usr/bin/env python3
"""Drive bounded RVV+scalar dispatch executable evidence.

This helper is runner/evidence tooling only. It orchestrates existing
TianChen-RV MLIR tools and optional ``ssh rvv`` compile/link/run evidence for
the planned RVV+scalar i32 add/sub/mul dispatch slice. It does not
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
DEFAULT_BUNDLE_ARTIFACT_ROOT = Path(
    "artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e"
)
DEFAULT_SSH_TARGET = "rvv"
DEFAULT_TIMEOUT_SECONDS = 60
BUNDLE_INDEX_FILE_NAME = "tianchenrv-target-artifact-bundle.index"
DISPATCH_RUNTIME_ABI_SIGNATURE = [
    {
        "c_name": "lhs",
        "c_type": "const int32_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "rhs",
        "c_type": "const int32_t *",
        "role": "rhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "out",
        "c_type": "int32_t *",
        "role": "output-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "n",
        "c_type": "size_t",
        "role": "runtime-element-count",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "rvv_available",
        "c_type": "int",
        "role": "dispatch-availability-guard",
        "ownership": "target-export-abi-owned",
    },
]

SUPPORTED_RVV_VECTOR_SHAPES: tuple[str, ...] = ("i32m1", "i32m2")

ARITHMETIC_FAMILY_SPECS: dict[str, dict[str, str | Path]] = {
    "i32-vadd": {
        "diagnostic_name": "i32-vadd",
        "function_stem": "i32_vadd",
        "intrinsic": "__riscv_vadd_vv_i32m1",
        "c_operator": "+",
        "success_marker": "tcrv_rvv_scalar_i32_vadd_dispatch_self_check_ok",
        "bundle_success_marker": "tcrv_rvv_scalar_i32_vadd_bundle_external_abi_ok",
        "component_group": "rvv-scalar-i32-vadd-dispatch-external-abi.v1",
        "external_abi_name": "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1",
        "source_route": "tcrv-export-rvv-scalar-i32-vadd-dispatch-c",
        "header_route": "tcrv-export-rvv-scalar-i32-vadd-dispatch-header",
        "object_route": "tcrv-export-rvv-scalar-i32-vadd-dispatch-object",
        "rvv_callable_route": "tcrv-export-rvv-microkernel-c",
        "scalar_callable_route": "tcrv-export-scalar-microkernel-c",
        "self_check_route": "--tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c",
        "default_input": DEFAULT_INPUT,
        "default_plan_and_export_input": Path(
            "test/Target/TargetArtifactBundleExport/plan-linalg-i32-vadd-and-export-target-artifact-bundle.mlir"
        ),
    },
    "i32-vsub": {
        "diagnostic_name": "i32-vsub",
        "function_stem": "i32_vsub",
        "intrinsic": "__riscv_vsub_vv_i32m1",
        "c_operator": "-",
        "success_marker": "tcrv_rvv_scalar_i32_vsub_dispatch_self_check_ok",
        "bundle_success_marker": "tcrv_rvv_scalar_i32_vsub_bundle_external_abi_ok",
        "component_group": "rvv-scalar-i32-vsub-dispatch-external-abi.v1",
        "external_abi_name": "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1",
        "source_route": "tcrv-export-rvv-scalar-i32-vsub-dispatch-c",
        "header_route": "tcrv-export-rvv-scalar-i32-vsub-dispatch-header",
        "object_route": "tcrv-export-rvv-scalar-i32-vsub-dispatch-object",
        "rvv_callable_route": "tcrv-export-rvv-i32-vsub-microkernel-c",
        "scalar_callable_route": "tcrv-export-scalar-i32-vsub-microkernel-c",
        "self_check_route": "--tcrv-export-rvv-scalar-i32-vsub-dispatch-self-check-c",
        "default_input": Path(
            "test/Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir"
        ),
        "i32m2_default_input": Path(
            "test/Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-i32m2-generic-route.mlir"
        ),
        "default_plan_and_export_input": Path(
            "test/Target/TargetArtifactBundleExport/plan-linalg-i32-vsub-and-export-target-artifact-bundle.mlir"
        ),
        "i32m2_default_plan_and_export_input": Path(
            "test/Target/TargetArtifactBundleExport/plan-linalg-i32m2-vsub-and-export-target-artifact-bundle.mlir"
        ),
    },
    "i32-vmul": {
        "diagnostic_name": "i32-vmul",
        "function_stem": "i32_vmul",
        "intrinsic": "__riscv_vmul_vv_i32m1",
        "c_operator": "*",
        "success_marker": "tcrv_rvv_scalar_i32_vmul_dispatch_self_check_ok",
        "bundle_success_marker": "tcrv_rvv_scalar_i32_vmul_bundle_external_abi_ok",
        "component_group": "rvv-scalar-i32-vmul-dispatch-external-abi.v1",
        "external_abi_name": "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1",
        "source_route": "tcrv-export-rvv-scalar-i32-vmul-dispatch-c",
        "header_route": "tcrv-export-rvv-scalar-i32-vmul-dispatch-header",
        "object_route": "tcrv-export-rvv-scalar-i32-vmul-dispatch-object",
        "rvv_callable_route": "tcrv-export-rvv-i32-vmul-microkernel-c",
        "scalar_callable_route": "tcrv-export-scalar-i32-vmul-microkernel-c",
        "self_check_route": "--tcrv-export-rvv-scalar-i32-vmul-dispatch-self-check-c",
        "default_input": Path(
            "test/Target/RVVScalarDispatch/rvv-scalar-i32-vmul-dispatch-generic-route.mlir"
        ),
        "default_plan_and_export_input": Path(
            "test/Target/TargetArtifactBundleExport/plan-linalg-i32-vmul-and-export-target-artifact-bundle.mlir"
        ),
    },
}

ACTIVE_ARITHMETIC_FAMILY = ARITHMETIC_FAMILY_SPECS["i32-vadd"]
ACTIVE_VECTOR_SHAPE = "i32m1"
SUCCESS_MARKER = str(ACTIVE_ARITHMETIC_FAMILY["success_marker"])
BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER = str(
    ACTIVE_ARITHMETIC_FAMILY["bundle_success_marker"]
)
DISPATCH_EXTERNAL_ABI_COMPONENT_GROUP = str(
    ACTIVE_ARITHMETIC_FAMILY["component_group"]
)
DISPATCH_EXTERNAL_ABI_NAME = str(ACTIVE_ARITHMETIC_FAMILY["external_abi_name"])


def make_dispatch_bundle_routes(
    family: dict[str, str | Path],
) -> dict[str, dict[str, str]]:
    component_group = str(family["component_group"])
    external_abi_name = str(family["external_abi_name"])
    return {
        "source": {
            "route": str(family["source_route"]),
            "artifact_kind": "runtime-callable-c-source",
            "component_group": component_group,
            "component_role": "source",
            "external_abi_name": external_abi_name,
            "owner": "rvv-scalar-dispatch-target",
            "runtime_abi_kind": "rvv-scalar-dispatch-runtime-callable-c-abi",
            "runtime_abi_name": external_abi_name,
            "evidence_role": "compiler-artifact",
        },
        "header": {
            "route": str(family["header_route"]),
            "artifact_kind": "runtime-callable-c-header",
            "component_group": component_group,
            "component_role": "header",
            "external_abi_name": external_abi_name,
            "owner": "rvv-scalar-dispatch-target",
            "runtime_abi_kind": "rvv-scalar-dispatch-runtime-callable-c-abi",
            "runtime_abi_name": external_abi_name,
            "evidence_role": "header-declaration",
        },
        "object": {
            "route": str(family["object_route"]),
            "artifact_kind": "riscv-elf-relocatable-object",
            "component_group": component_group,
            "component_role": "object",
            "external_abi_name": external_abi_name,
            "owner": "rvv-scalar-dispatch-target",
            "runtime_abi_kind": "rvv-scalar-dispatch-runtime-callable-c-abi",
            "runtime_abi_name": external_abi_name,
            "evidence_role": "relocatable-object",
        },
    }


DISPATCH_BUNDLE_ROUTES = make_dispatch_bundle_routes(ACTIVE_ARITHMETIC_FAMILY)


def configure_arithmetic_family(family_name: str) -> None:
    global ACTIVE_ARITHMETIC_FAMILY
    global SUCCESS_MARKER
    global BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER
    global DISPATCH_EXTERNAL_ABI_COMPONENT_GROUP
    global DISPATCH_EXTERNAL_ABI_NAME
    global DISPATCH_BUNDLE_ROUTES

    family = ARITHMETIC_FAMILY_SPECS.get(family_name)
    if family is None:
        raise BridgeError(f"unsupported arithmetic family: {family_name}")
    ACTIVE_ARITHMETIC_FAMILY = family
    SUCCESS_MARKER = str(family["success_marker"])
    BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER = str(family["bundle_success_marker"])
    DISPATCH_EXTERNAL_ABI_COMPONENT_GROUP = str(family["component_group"])
    DISPATCH_EXTERNAL_ABI_NAME = str(family["external_abi_name"])
    DISPATCH_BUNDLE_ROUTES = make_dispatch_bundle_routes(family)


def configure_vector_shape(shape_name: str) -> None:
    global ACTIVE_VECTOR_SHAPE

    if shape_name not in SUPPORTED_RVV_VECTOR_SHAPES:
        raise BridgeError(f"unsupported RVV vector shape: {shape_name}")
    ACTIVE_VECTOR_SHAPE = shape_name


def arithmetic_intrinsic_for_family(
    family: dict[str, str | Path], vector_suffix: str
) -> str:
    stem = str(family["function_stem"]).removeprefix("i32_v")
    return "__riscv_v" + stem + "_vv_" + vector_suffix


def load_intrinsic_for_suffix(vector_suffix: str) -> str:
    return "__riscv_vle32_v_" + vector_suffix


def store_intrinsic_for_suffix(vector_suffix: str) -> str:
    return "__riscv_vse32_v_" + vector_suffix


def setvl_intrinsic_for_suffix(setvl_suffix: str) -> str:
    return "__riscv_vsetvl_" + setvl_suffix


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


def git_sha(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BridgeError(f"failed to read git sha: {error}") from error
    if completed.returncode != 0:
        raise BridgeError(
            "failed to read git sha: " + bounded_tail(completed.stderr, limit=400)
        )
    sha = sanitize_text(completed.stdout).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise BridgeError("git rev-parse HEAD returned a malformed sha")
    return sha


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

    if shutil.which(tool_name):
        return tool_name

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
        "could not find clang for local target artifact bundle object export; "
        "install clang or put an LLVM clang directory on PATH"
    )


def command_display(command: list[str]) -> str:
    return sanitize_text(shlex.join(command))


def bounded_tail(text: Any, limit: int = 1200) -> str:
    sanitized = sanitize_text(text)
    if len(sanitized) <= limit:
        return sanitized
    return sanitized[-limit:]


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
            "stdout_tail": bounded_tail(stdout),
            "stderr_tail": bounded_tail(stderr),
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
        'lowering_pipeline: "'
        + str(ACTIVE_ARITHMETIC_FAMILY["rvv_callable_route"])
        + '"',
        'runtime_abi_kind: "rvv-runtime-callable-c-abi"',
        'origin: "scalar-plugin"',
        'role: "dispatch fallback"',
        'lowering_pipeline: "'
        + str(ACTIVE_ARITHMETIC_FAMILY["scalar_callable_route"])
        + '"',
        'runtime_abi_kind: "scalar-runtime-callable-c-abi"',
    ]
    missing = [snippet for snippet in required if snippet not in manifest_text]
    if missing:
        raise BridgeError(
            "emission manifest missing planned RVV+scalar dispatch handoff snippets: "
            + ", ".join(missing)
        )


def parse_comma_key_values(text: str, context: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise BridgeError(f"{context} contains malformed field: {part}")
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise BridgeError(f"{context} contains empty key/value field: {part}")
        reject_secret_like_text(f"{context} field {key}", value)
        values[key] = value
    return values


def parse_control_plane_config(text: str) -> dict[str, str]:
    match = re.search(
        r"sew=([^,]+),\s*lmul=([^,]+),\s*policy=#tcrv_rvv\.policy<tail = ([^,]+), mask = ([^>]+)>",
        text,
    )
    if not match:
        raise BridgeError(
            "generated dispatch C source control_plane_config is malformed"
        )
    return {
        "sew": match.group(1).strip(),
        "lmul": match.group(2).strip(),
        "tail_policy": match.group(3).strip(),
        "mask_policy": match.group(4).strip(),
    }


def parse_rvv_selected_plan_metadata_comments(source: str) -> list[dict[str, str]]:
    metadata: list[dict[str, str]] = []
    pattern = re.compile(
        r"/\*\s*rvv_selected_plan_metadata\[([0-9]+)\]:\s*"
        r"name=([^,]+),\s*value=([^,]+),\s*role=([^,]+),\s*note=([^*]+?)\s*\*/"
    )
    for match in pattern.finditer(source):
        entry = {
            "index": match.group(1).strip(),
            "name": match.group(2).strip(),
            "value": match.group(3).strip(),
            "role": match.group(4).strip(),
            "note": match.group(5).strip(),
        }
        for field, value in entry.items():
            reject_secret_like_text(f"rvv selected_plan_metadata {field}", value)
        metadata.append(entry)
    return metadata


def require_source_snippets(source: str, snippets: list[str], context: str) -> None:
    missing = [snippet for snippet in snippets if snippet not in source]
    if missing:
        raise BridgeError(
            f"{context} missing required snippets: " + ", ".join(missing)
        )


def validate_vector_shape_metadata(source: str) -> dict[str, Any]:
    selected_shape_config = parse_source_comment(
        source, "selected_vector_shape_config", required=True
    )
    selected_shape_capabilities = parse_source_comment(
        source, "selected_vector_shape_capabilities", required=True
    )
    control_config = parse_source_comment(source, "control_plane_config", required=True)
    intrinsic_config = parse_source_comment(source, "intrinsic_config", required=True)

    shape_values = parse_comma_key_values(
        selected_shape_config, "selected_vector_shape_config"
    )
    intrinsic_values = parse_comma_key_values(intrinsic_config, "intrinsic_config")
    control_values = parse_control_plane_config(control_config)
    capability_ids = selected_shape_capabilities.split()
    for capability_id in capability_ids:
        reject_secret_like_text("selected_vector_shape_capability", capability_id)

    required_shape_fields = {
        "shape",
        "sew",
        "lmul",
        "tail_policy",
        "mask_policy",
        "vector_type",
        "vector_suffix",
        "setvl_suffix",
    }
    missing_shape_fields = sorted(required_shape_fields - set(shape_values))
    if missing_shape_fields:
        raise BridgeError(
            "generated dispatch C source selected_vector_shape_config is "
            "missing fields: "
            + ", ".join(missing_shape_fields)
        )

    if shape_values["shape"] != ACTIVE_VECTOR_SHAPE:
        raise BridgeError(
            "generated dispatch C source selected_vector_shape_config has "
            f"shape {shape_values['shape']}, requested vector shape "
            f"{ACTIVE_VECTOR_SHAPE}"
        )

    if control_values["sew"] != shape_values["sew"]:
        raise BridgeError(
            "generated dispatch C source control_plane_config sew does not "
            "match selected_vector_shape_config"
        )
    for field in ("lmul", "tail_policy", "mask_policy"):
        if control_values[field] != shape_values[field]:
            raise BridgeError(
                "generated dispatch C source control_plane_config "
                f"{field} does not match selected_vector_shape_config"
            )

    for field in (
        "vector_type",
        "vector_suffix",
        "setvl_suffix",
        "tail_policy",
        "mask_policy",
    ):
        if intrinsic_values.get(field) != shape_values[field]:
            raise BridgeError(
                "generated dispatch C source intrinsic_config "
                f"{field} does not match selected_vector_shape_config"
            )

    if len(capability_ids) != 4 or not all(
        capability.startswith("rvv.i32_") for capability in capability_ids
    ):
        raise BridgeError(
            "generated dispatch C source selected_vector_shape_capabilities "
            "must contain exactly four bounded RVV i32 capability ids"
        )

    required_intrinsics = [
        setvl_intrinsic_for_suffix(shape_values["setvl_suffix"]),
        load_intrinsic_for_suffix(shape_values["vector_suffix"]),
        arithmetic_intrinsic_for_family(
            ACTIVE_ARITHMETIC_FAMILY, shape_values["vector_suffix"]
        ),
        store_intrinsic_for_suffix(shape_values["vector_suffix"]),
    ]
    require_source_snippets(
        source, required_intrinsics, "generated dispatch C source"
    )

    return {
        "shape": shape_values["shape"],
        "sew_bits": int(shape_values["sew"]),
        "lmul": shape_values["lmul"],
        "tail_policy": shape_values["tail_policy"],
        "mask_policy": shape_values["mask_policy"],
        "vector_type": shape_values["vector_type"],
        "vector_suffix": shape_values["vector_suffix"],
        "setvl_suffix": shape_values["setvl_suffix"],
        "capability_ids": capability_ids,
        "selected_vector_shape_config": selected_shape_config,
        "selected_vector_shape_capabilities": selected_shape_capabilities,
        "control_plane_config": control_config,
        "intrinsic_config": intrinsic_config,
        "selected_plan_metadata": parse_rvv_selected_plan_metadata_comments(source),
        "required_intrinsics": required_intrinsics,
    }


def validate_library_dispatch_source(source: str) -> dict[str, Any]:
    if not source.strip():
        raise BridgeError("generated library dispatch C source is empty")
    reject_secret_like_text("generated library dispatch C source", source)
    if "int main(void)" in source or SUCCESS_MARKER in source:
        raise BridgeError(
            "generic target source artifact must remain library-style without "
            "a self-check main or success marker"
        )
    vector_config = validate_vector_shape_metadata(source)
    required = [
        "/* TianChen-RV RVV+scalar host runtime dispatch C export. */",
        "/* Runtime guard: explicit host-provided rvv_available parameter; no automatic hardware probe is generated. */",
        *vector_config["required_intrinsics"],
        "out[index] = lhs[index] "
        + str(ACTIVE_ARITHMETIC_FAMILY["c_operator"])
        + " rhs[index];",
        "void tcrv_dispatch_"
        + str(ACTIVE_ARITHMETIC_FAMILY["function_stem"])
        + "_",
        "if (rvv_available)",
        "/* selected_vector_shape_config: shape=" + ACTIVE_VECTOR_SHAPE,
        "/* selected_vector_shape_capabilities:",
        "/* control_plane_config: sew=" + str(vector_config["sew_bits"]),
        "/* intrinsic_config: vector_type="
        + str(vector_config["vector_type"]),
    ]
    missing = [snippet for snippet in required if snippet not in source]
    if missing:
        raise BridgeError(
            "generated library dispatch C source missing required snippets: "
            + ", ".join(missing)
        )
    for other_name in SUPPORTED_RVV_VECTOR_SHAPES:
        if other_name == ACTIVE_VECTOR_SHAPE:
            continue
        stale_snippets = [
            "selected_vector_shape_config: shape=" + other_name,
            "vector_suffix=" + other_name,
            "__riscv_vle32_v_" + other_name,
            arithmetic_intrinsic_for_family(ACTIVE_ARITHMETIC_FAMILY, other_name),
            "__riscv_vse32_v_" + other_name,
        ]
        leaked = [snippet for snippet in stale_snippets if snippet in source]
        if leaked:
            raise BridgeError(
                "generated library dispatch C source contains stale "
                + other_name
                + " vector-shape metadata: "
                + ", ".join(leaked)
            )
    return vector_config


def validate_self_check_dispatch_source(source: str) -> dict[str, Any]:
    if not source.strip():
        raise BridgeError("generated dispatch self-check C source is empty")
    reject_secret_like_text("generated dispatch self-check C source", source)
    required = [
        "/* TianChen-RV RVV+scalar host runtime dispatch C export. */",
        "/* Explicit bounded self-check harness for RVV+scalar dispatch runtime invocation evidence. */",
        "rvv_available = 0",
        "rvv_available = 1",
        "runtime_counts=7,16",
        "int main(void)",
        SUCCESS_MARKER,
        "out[index] = lhs[index] "
        + str(ACTIVE_ARITHMETIC_FAMILY["c_operator"])
        + " rhs[index];",
        "tcrv_dispatch_"
        + str(ACTIVE_ARITHMETIC_FAMILY["function_stem"])
        + "_",
        "/* selected_vector_shape_config: shape=" + ACTIVE_VECTOR_SHAPE,
        "/* selected_vector_shape_capabilities:",
    ]
    vector_config = validate_vector_shape_metadata(source)
    required.extend(vector_config["required_intrinsics"])
    required.append("/* control_plane_config: sew=" + str(vector_config["sew_bits"]))
    required.append(
        "/* intrinsic_config: vector_type=" + str(vector_config["vector_type"])
    )
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
    return {
        "selected_march": selected_march,
        "selected_mabi": selected_mabi,
        "vector_config": vector_config,
    }


def parse_bundle_index_value(raw: str) -> str:
    value = raw.strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        inner = value[1:-1]
        return (
            inner.replace(r"\t", "\t")
            .replace(r"\"", '"')
            .replace(r"\\", "\\")
        )
    if value.startswith("@"):
        return value[1:]
    return value


def validate_bundle_file_name(file_name: str) -> None:
    if not file_name:
        raise BridgeError("bundle index artifact file_name must be non-empty")
    reject_secret_like_text("bundle index artifact file_name", file_name)
    if Path(file_name).name != file_name or "/" in file_name or "\\" in file_name:
        raise BridgeError(
            f"bundle index artifact file_name must be a plain file name: {file_name}"
        )


def parse_target_artifact_bundle_index(index_text: str) -> list[dict[str, Any]]:
    reject_secret_like_text("target artifact bundle index", index_text)
    bundle_status = ""
    artifact_count: int | None = None
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_component: dict[str, str] | None = None
    current_runtime_abi_parameter: dict[str, str] | None = None
    current_selected_plan_metadata: dict[str, str] | None = None

    for raw_line in index_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.startswith("bundle_status:"):
            bundle_status = parse_bundle_index_value(line.split(":", 1)[1])
            continue
        if line.startswith("artifact_count:"):
            count_text = line.split(":", 1)[1].strip()
            if not count_text.isdigit():
                raise BridgeError("bundle index artifact_count must be an integer")
            artifact_count = int(count_text)
            continue
        artifact_match = re.match(r"^artifact\[([0-9]+)\]:$", line)
        if artifact_match:
            current = {
                "index": int(artifact_match.group(1)),
                "components": [],
                "runtime_abi_parameters": [],
                "selected_plan_metadata": [],
            }
            records.append(current)
            current_component = None
            current_runtime_abi_parameter = None
            current_selected_plan_metadata = None
            continue
        component_match = re.match(r"^  component\[([0-9]+)\]:$", line)
        if component_match:
            if current is None:
                raise BridgeError("bundle index component appears before artifact")
            current_component = {"index": component_match.group(1)}
            current["components"].append(current_component)
            current_runtime_abi_parameter = None
            current_selected_plan_metadata = None
            continue
        parameter_match = re.match(r"^  runtime_abi_parameter\[([0-9]+)\]:$", line)
        if parameter_match:
            if current is None:
                raise BridgeError(
                    "bundle index runtime_abi_parameter appears before artifact"
                )
            current_runtime_abi_parameter = {"index": parameter_match.group(1)}
            current["runtime_abi_parameters"].append(current_runtime_abi_parameter)
            current_component = None
            current_selected_plan_metadata = None
            continue
        selected_metadata_match = re.match(
            r"^  selected_plan_metadata\[([0-9]+)\]:$", line
        )
        if selected_metadata_match:
            if current is None:
                raise BridgeError(
                    "bundle index selected_plan_metadata appears before artifact"
                )
            current_selected_plan_metadata = {
                "index": selected_metadata_match.group(1)
            }
            current["selected_plan_metadata"].append(current_selected_plan_metadata)
            current_component = None
            current_runtime_abi_parameter = None
            continue

        field_match = re.match(r"^  ([A-Za-z0-9_]+):\s*(.*)$", line)
        if field_match and current is not None:
            key, raw_value = field_match.groups()
            value = parse_bundle_index_value(raw_value)
            reject_secret_like_text(f"bundle index field {key}", value)
            current[key] = value
            current_component = None
            current_runtime_abi_parameter = None
            current_selected_plan_metadata = None
            continue

        component_field_match = re.match(r"^    ([A-Za-z0-9_]+):\s*(.*)$", line)
        if component_field_match and current_component is not None:
            key, raw_value = component_field_match.groups()
            value = parse_bundle_index_value(raw_value)
            reject_secret_like_text(f"bundle index component field {key}", value)
            current_component[key] = value
            continue
        if component_field_match and current_runtime_abi_parameter is not None:
            key, raw_value = component_field_match.groups()
            value = parse_bundle_index_value(raw_value)
            reject_secret_like_text(
                f"bundle index runtime_abi_parameter field {key}", value
            )
            current_runtime_abi_parameter[key] = value
            continue
        if component_field_match and current_selected_plan_metadata is not None:
            key, raw_value = component_field_match.groups()
            value = parse_bundle_index_value(raw_value)
            reject_secret_like_text(
                f"bundle index selected_plan_metadata field {key}", value
            )
            current_selected_plan_metadata[key] = value
            continue

    if bundle_status != "complete":
        raise BridgeError("bundle index must record bundle_status complete")
    if artifact_count is None:
        raise BridgeError("bundle index missing artifact_count")
    if artifact_count != len(records):
        raise BridgeError(
            f"bundle index artifact_count={artifact_count} does not match parsed records={len(records)}"
        )

    required_fields = [
        "file_name",
        "component_role",
        "artifact_kind",
        "route",
        "owner",
        "runtime_abi_kind",
        "runtime_abi_name",
        "evidence_role",
    ]
    for record in records:
        for field in required_fields:
            if not str(record.get(field, "")).strip():
                raise BridgeError(
                    f"bundle index artifact[{record.get('index')}] missing required field {field}"
                )
        validate_bundle_file_name(str(record["file_name"]))
    return records


def require_dispatch_runtime_abi_signature(record: dict[str, Any]) -> list[dict[str, str]]:
    raw_parameters = record.get("runtime_abi_parameters")
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise BridgeError(
            f"bundle record route {record.get('route')} must publish runtime_abi_parameter signature metadata"
        )

    parameters: list[dict[str, str]] = []
    seen_roles: set[str] = set()
    for index, raw_parameter in enumerate(raw_parameters):
        if not isinstance(raw_parameter, dict):
            raise BridgeError(
                f"bundle record route {record.get('route')} runtime_abi_parameter[{index}] must be a dictionary"
            )
        parameter = {
            "c_name": str(raw_parameter.get("c_name", "")),
            "c_type": str(raw_parameter.get("c_type", "")),
            "role": str(raw_parameter.get("role", "")),
            "ownership": str(raw_parameter.get("ownership", "")),
        }
        for field, value in parameter.items():
            if not value.strip():
                raise BridgeError(
                    f"bundle record route {record.get('route')} runtime_abi_parameter[{index}] missing {field}"
                )
            reject_secret_like_text(f"runtime ABI parameter {field}", value)
        if parameter["role"] in seen_roles:
            raise BridgeError(
                f"bundle record route {record.get('route')} has duplicate runtime ABI parameter role {parameter['role']}"
            )
        seen_roles.add(parameter["role"])
        parameters.append(parameter)

    if len(parameters) != len(DISPATCH_RUNTIME_ABI_SIGNATURE):
        raise BridgeError(
            f"bundle record route {record.get('route')} runtime ABI signature must contain {len(DISPATCH_RUNTIME_ABI_SIGNATURE)} parameters"
        )
    for index, (parameter, expected) in enumerate(
        zip(parameters, DISPATCH_RUNTIME_ABI_SIGNATURE)
    ):
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", parameter["c_name"]):
            raise BridgeError(
                f"bundle record route {record.get('route')} runtime_abi_parameter[{index}] c_name is not a valid C identifier"
            )
        for field in ("c_type", "role", "ownership"):
            if parameter[field] != expected[field]:
                raise BridgeError(
                    f"bundle record route {record.get('route')} runtime_abi_parameter[{index}] {field}={parameter[field]!r} does not match expected {expected[field]!r}"
                )
    return parameters


def require_dispatch_component_roles(record: dict[str, Any]) -> None:
    if record.get("selected_surface") != "dispatch":
        raise BridgeError(
            f"bundle record route {record.get('route')} must preserve selected_surface dispatch"
        )
    components = record.get("components")
    if not isinstance(components, list) or len(components) != 2:
        raise BridgeError(
            f"bundle record route {record.get('route')} must preserve two dispatch components"
        )
    roles = {component.get("role") for component in components}
    if roles != {"dispatch case", "dispatch fallback"}:
        raise BridgeError(
            f"bundle record route {record.get('route')} must preserve dispatch case and dispatch fallback roles"
        )
    for component in components:
        if not component.get("selected_variant"):
            raise BridgeError(
                f"bundle record route {record.get('route')} has a component without selected_variant"
            )


def select_dispatch_bundle_records(
    records: list[dict[str, Any]], bundle_dir: Path
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for label, expected in DISPATCH_BUNDLE_ROUTES.items():
        matches = [record for record in records if record.get("route") == expected["route"]]
        if len(matches) != 1:
            raise BridgeError(
                f"bundle index must contain exactly one {label} record for route {expected['route']}; found {len(matches)}"
            )
        record = matches[0]
        for field in (
            "artifact_kind",
            "component_group",
            "component_role",
            "external_abi_name",
            "owner",
            "runtime_abi_kind",
            "runtime_abi_name",
            "evidence_role",
        ):
            if record.get(field) != expected[field]:
                raise BridgeError(
                    f"bundle {label} record field {field}={record.get(field)!r} does not match expected {expected[field]!r}"
                )
        require_dispatch_component_roles(record)
        require_dispatch_runtime_abi_signature(record)
        artifact_path = bundle_dir / str(record["file_name"])
        if not artifact_path.exists() or artifact_path.stat().st_size == 0:
            raise BridgeError(
                f"bundle {label} artifact is missing or empty: {record['file_name']}"
            )
        selected[label] = record
    groups = {record.get("component_group") for record in selected.values()}
    roles = {record.get("component_role") for record in selected.values()}
    external_abi_names = {
        record.get("external_abi_name") for record in selected.values()
    }
    if groups != {DISPATCH_EXTERNAL_ABI_COMPONENT_GROUP}:
        raise BridgeError(
            "selected dispatch bundle records must share the compiler-emitted "
            f"component_group {DISPATCH_EXTERNAL_ABI_COMPONENT_GROUP}"
        )
    if roles != {"source", "header", "object"}:
        raise BridgeError(
            "selected dispatch bundle records must expose source/header/object "
            f"component_role values; got {sorted(str(role) for role in roles)}"
        )
    if external_abi_names != {DISPATCH_EXTERNAL_ABI_NAME}:
        raise BridgeError(
            "selected dispatch bundle records must share the compiler-emitted "
            f"external_abi_name {DISPATCH_EXTERNAL_ABI_NAME}"
        )
    signatures = {
        json.dumps(record.get("runtime_abi_parameters", []), sort_keys=True)
        for record in selected.values()
    }
    if len(signatures) != 1:
        raise BridgeError(
            "selected dispatch bundle records must share the compiler-emitted runtime ABI parameter signature"
        )
    return selected


def bundle_records_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for record in records:
        summary.append(
            {
                "index": record.get("index"),
                "file_name": record.get("file_name"),
                "component_group": record.get("component_group", ""),
                "component_role": record.get("component_role", ""),
                "external_abi_name": record.get("external_abi_name", ""),
                "artifact_kind": record.get("artifact_kind"),
                "route": record.get("route"),
                "owner": record.get("owner"),
                "runtime_abi_kind": record.get("runtime_abi_kind"),
                "runtime_abi_name": record.get("runtime_abi_name"),
                "evidence_role": record.get("evidence_role"),
                "runtime_abi_parameters": record.get("runtime_abi_parameters", []),
                "selected_plan_metadata": record.get("selected_plan_metadata", []),
                "selected_surface": record.get("selected_surface", ""),
                "components": record.get("components", []),
            }
        )
    return summary


def runtime_abi_parameter_c_declaration(parameter: dict[str, str]) -> str:
    c_type = parameter["c_type"]
    c_name = parameter["c_name"]
    separator = "" if c_type.endswith("*") else " "
    return f"{c_type}{separator}{c_name}"


def normalize_c_parameter_list(parameters: str) -> str:
    return re.sub(r"\s+", " ", parameters.strip())


def validate_generated_dispatch_header(
    header: str, signature: list[dict[str, str]]
) -> str:
    if not header.strip():
        raise BridgeError("generated dispatch C header is empty")
    reject_secret_like_text("generated dispatch C header", header)
    forbidden = [
        "int main",
        "_self_check",
        "__riscv",
        "riscv_vector",
        "runtime_success",
        "throughput",
        "latency",
        "artifacts/tmp",
        SUCCESS_MARKER,
        BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER,
    ]
    for snippet in forbidden:
        if snippet in header:
            raise BridgeError(
                f"generated dispatch C header contains forbidden snippet: {snippet}"
            )
    for snippet in ("#ifndef ", "#define ", "#include <stddef.h>", "#include <stdint.h>"):
        if snippet not in header:
            raise BridgeError(
                f"generated dispatch C header missing required snippet: {snippet}"
            )
    prototypes = re.findall(
        r"(?m)^\s*void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;{}]*)\)\s*;\s*$",
        header,
    )
    if len(prototypes) != 1:
        raise BridgeError(
            "generated dispatch C header must contain exactly one runtime-callable dispatch prototype"
        )
    expected_parameters = ", ".join(
        runtime_abi_parameter_c_declaration(parameter) for parameter in signature
    )
    if normalize_c_parameter_list(prototypes[0][1]) != normalize_c_parameter_list(
        expected_parameters
    ):
        raise BridgeError(
            "generated dispatch C header prototype does not match bundle index runtime ABI signature"
        )
    if re.search(r"(?m)^\s*void\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*\)\s*\{", header):
        raise BridgeError("generated dispatch C header must not contain a function body")
    return prototypes[0][0]


def build_dispatch_external_caller_source(
    function_name: str, header_file_name: str, signature: list[dict[str, str]]
) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", function_name):
        raise BridgeError("generated dispatch header function name is not a valid C identifier")
    validate_bundle_file_name(header_file_name)
    call_arguments: list[str] = []
    for parameter in signature:
        role = parameter["role"]
        if role == "lhs-input-buffer":
            call_arguments.append("lhs")
        elif role == "rhs-input-buffer":
            call_arguments.append("rhs")
        elif role == "output-buffer":
            call_arguments.append("out")
        elif role == "runtime-element-count":
            call_arguments.append("runtime_n")
        elif role == "dispatch-availability-guard":
            call_arguments.append("rvv_available")
        else:
            raise BridgeError(f"unsupported runtime ABI parameter role in bundle index: {role}")
    rendered_call_arguments = ", ".join(call_arguments)
    escaped_header = header_file_name.replace("\\", "\\\\").replace('"', '\\"')
    c_operator = str(ACTIVE_ARITHMETIC_FAMILY["c_operator"])
    family_name = str(ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"])
    return f"""\
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "{escaped_header}"

static int run_dispatch_case(size_t runtime_n, int rvv_available) {{
  enum {{ kCapacity = 32 }};
  int32_t lhs[kCapacity];
  int32_t rhs[kCapacity];
  int32_t out[kCapacity];

  for (size_t index = 0; index < (size_t)kCapacity; ++index) {{
    lhs[index] = (int32_t)index;
    rhs[index] = (int32_t)(31 - (int)index);
    out[index] = -12345;
  }}

  {function_name}({rendered_call_arguments});
  for (size_t index = 0; index < runtime_n; ++index) {{
    if (out[index] != lhs[index] {c_operator} rhs[index]) {{
      fprintf(stderr,
              "rvv scalar {family_name} dispatch bundle external ABI mismatch n=%zu guard=%d index=%zu\\n",
              runtime_n, rvv_available, index);
      return rvv_available ? 11 : 10;
    }}
  }}
  for (size_t index = runtime_n; index < (size_t)kCapacity; ++index) {{
    if (out[index] != -12345) {{
      fprintf(stderr,
              "rvv scalar {family_name} dispatch bundle external ABI overrun n=%zu guard=%d index=%zu\\n",
              runtime_n, rvv_available, index);
      return rvv_available ? 13 : 12;
    }}
  }}
  return 0;
}}

int main(void) {{
  if (run_dispatch_case(7, 0))
    return 10;
  if (run_dispatch_case(16, 0))
    return 11;
  if (run_dispatch_case(7, 1))
    return 12;
  if (run_dispatch_case(16, 1))
    return 13;
  printf("{BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER} runtime_counts=7,16 branches=scalar_and_rvv\\n");
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


def first_non_empty_line(text: str) -> str:
    for line in sanitize_text(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


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


def build_remote_bundle_compile_caller_object_command(
    remote_dir: str, flags: dict[str, str]
) -> str:
    quoted_flags = " ".join(shlex.quote(flag) for flag in remote_compile_flags(flags))
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        f"clang {quoted_flags} -c rvv_bundle_dispatch_external_caller.c "
        "-o rvv_bundle_dispatch_external_caller.o"
    )


def build_remote_bundle_compile_dispatch_source_object_command(
    remote_dir: str, source_file_name: str, flags: dict[str, str]
) -> str:
    validate_bundle_file_name(source_file_name)
    quoted_flags = " ".join(shlex.quote(flag) for flag in remote_compile_flags(flags))
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        f"clang {quoted_flags} -c {shlex.quote(source_file_name)} "
        "-o rvv_bundle_dispatch_from_source.o"
    )


def build_remote_bundle_link_executable_command(
    remote_dir: str,
    object_file_name: str,
    executable_file_name: str,
    flags: dict[str, str],
) -> str:
    validate_bundle_file_name(object_file_name)
    validate_bundle_file_name(executable_file_name)
    quoted_flags = " ".join(shlex.quote(flag) for flag in remote_compile_flags(flags))
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        f"clang {quoted_flags} rvv_bundle_dispatch_external_caller.o "
        f"{shlex.quote(object_file_name)} -o {shlex.quote(executable_file_name)}"
    )


def run_remote_bundle_external_abi_evidence(
    args: argparse.Namespace,
    *,
    root: Path,
    artifact_dir: Path,
    commands: list[dict[str, Any]],
    source_path: Path,
    header_path: Path,
    object_path: Path,
    caller_path: Path,
    source_file_name: str,
    object_file_name: str,
    flags: dict[str, str],
    run_id: str,
) -> dict[str, Any]:
    reject_secret_like_text("ssh target", args.ssh_target)
    remote_dir = f"/tmp/tianchenrv_rvv_bundle_e2e_{safe_run_id(run_id)}"

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

    uname_stdout, _, _ = run_command(
        "ssh_uname",
        remote_shell_command(args, "uname -a"),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    arch_stdout, _, _ = run_command(
        "ssh_architecture",
        remote_shell_command(args, "uname -m"),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    clang_path_stdout, _, _ = run_command(
        "ssh_clang_path",
        remote_shell_command(args, "command -v clang || true"),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    clang_version_stdout, _, _ = run_command(
        "ssh_clang_version",
        remote_shell_command(args, "clang --version | head -n 1"),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_command(
        "scp_bundle_source_header_object_and_external_caller",
        [
            *scp_base_command(args),
            relative_to_repo(source_path, root),
            relative_to_repo(header_path, root),
            relative_to_repo(object_path, root),
            relative_to_repo(caller_path, root),
            f"{args.ssh_target}:{remote_dir}/",
        ],
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_command(
        "ssh_compile_bundle_external_caller_object",
        remote_shell_command(
            args, build_remote_bundle_compile_caller_object_command(remote_dir, flags)
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    caller_object_hash_stdout, _, _ = run_command(
        "ssh_bundle_caller_object_sha256",
        remote_shell_command(
            args,
            remote_sha256_command(remote_dir, "rvv_bundle_dispatch_external_caller.o"),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_command(
        "ssh_compile_bundle_dispatch_source_object",
        remote_shell_command(
            args,
            build_remote_bundle_compile_dispatch_source_object_command(
                remote_dir, source_file_name, flags
            ),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    source_object_hash_stdout, _, _ = run_command(
        "ssh_bundle_dispatch_source_object_sha256",
        remote_shell_command(
            args,
            remote_sha256_command(remote_dir, "rvv_bundle_dispatch_from_source.o"),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_command(
        "ssh_link_bundle_source_external_caller",
        remote_shell_command(
            args,
            build_remote_bundle_link_executable_command(
                remote_dir,
                "rvv_bundle_dispatch_from_source.o",
                "rvv_bundle_dispatch_external_caller_from_source",
                flags,
            ),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    source_executable_hash_stdout, _, _ = run_command(
        "ssh_bundle_source_executable_sha256",
        remote_shell_command(
            args,
            remote_sha256_command(
                remote_dir, "rvv_bundle_dispatch_external_caller_from_source"
            ),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    source_run_stdout, _, _ = run_command(
        "ssh_run_bundle_source_external_caller",
        remote_shell_command(
            args,
            f"cd {quote_remote_path(remote_dir)} && "
            "./rvv_bundle_dispatch_external_caller_from_source",
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    if BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER not in source_run_stdout:
        raise BridgeError(
            "remote bundle source-built external caller stdout missing expected marker: "
            + BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER
        )

    run_command(
        "ssh_link_bundle_index_object_external_caller",
        remote_shell_command(
            args,
            build_remote_bundle_link_executable_command(
                remote_dir,
                object_file_name,
                "rvv_bundle_dispatch_external_caller_from_bundle_object",
                flags,
            ),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    bundle_object_executable_hash_stdout, _, _ = run_command(
        "ssh_bundle_index_object_executable_sha256",
        remote_shell_command(
            args,
            remote_sha256_command(
                remote_dir, "rvv_bundle_dispatch_external_caller_from_bundle_object"
            ),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    bundle_object_run_stdout, _, _ = run_command(
        "ssh_run_bundle_index_object_external_caller",
        remote_shell_command(
            args,
            f"cd {quote_remote_path(remote_dir)} && "
            "./rvv_bundle_dispatch_external_caller_from_bundle_object",
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    if BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER not in bundle_object_run_stdout:
        raise BridgeError(
            "remote bundle object external caller stdout missing expected marker: "
            + BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER
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
        "host_facts": {
            "uname": first_non_empty_line(uname_stdout),
            "architecture": first_non_empty_line(arch_stdout),
            "clang_path": first_non_empty_line(clang_path_stdout),
            "clang_version": first_non_empty_line(clang_version_stdout),
        },
        "remote_artifacts": {
            "source_file": source_file_name,
            "header_file": header_path.name,
            "bundle_object_file": object_file_name,
            "caller_file": caller_path.name,
            "caller_object_file": "rvv_bundle_dispatch_external_caller.o",
            "source_built_object_file": "rvv_bundle_dispatch_from_source.o",
            "source_built_executable": (
                "rvv_bundle_dispatch_external_caller_from_source"
            ),
            "bundle_object_executable": (
                "rvv_bundle_dispatch_external_caller_from_bundle_object"
            ),
        },
        "compile_flags": remote_compile_flags(flags),
        "caller_object_compile_exit_code": 0,
        "source_object_compile_exit_code": 0,
        "source_link_exit_code": 0,
        "source_run_exit_code": 0,
        "bundle_object_link_exit_code": 0,
        "bundle_object_run_exit_code": 0,
        "expected_stdout_marker": BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER,
        "source_stdout_marker_observed": True,
        "bundle_object_stdout_marker_observed": True,
        "caller_object_sha256": sanitize_text(caller_object_hash_stdout)
        .strip()
        .splitlines()[0]
        if sanitize_text(caller_object_hash_stdout).strip()
        else "",
        "source_built_object_sha256": sanitize_text(source_object_hash_stdout)
        .strip()
        .splitlines()[0]
        if sanitize_text(source_object_hash_stdout).strip()
        else "",
        "source_built_executable_sha256": sanitize_text(
            source_executable_hash_stdout
        )
        .strip()
        .splitlines()[0]
        if sanitize_text(source_executable_hash_stdout).strip()
        else "",
        "bundle_object_executable_sha256": sanitize_text(
            bundle_object_executable_hash_stdout
        )
        .strip()
        .splitlines()[0]
        if sanitize_text(bundle_object_executable_hash_stdout).strip()
        else "",
        "cleanup_status": cleanup_status,
    }


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


def selected_artifact_root(args: argparse.Namespace) -> Path:
    if (
        args.use_target_artifact_bundle
        and args.artifact_root == str(DEFAULT_ARTIFACT_ROOT)
    ):
        return DEFAULT_BUNDLE_ARTIFACT_ROOT
    return Path(args.artifact_root)


def selected_input_path(args: argparse.Namespace) -> Path:
    if args.input:
        return Path(args.input)
    if ACTIVE_VECTOR_SHAPE != "i32m1":
        if args.use_target_artifact_bundle and args.use_plan_and_export_bundle_front_door:
            key = ACTIVE_VECTOR_SHAPE + "_default_plan_and_export_input"
        else:
            key = ACTIVE_VECTOR_SHAPE + "_default_input"
        default_input = ACTIVE_ARITHMETIC_FAMILY.get(key)
        if default_input is None:
            raise BridgeError(
                "no default dispatch MLIR fixture for arithmetic family "
                f"{ACTIVE_ARITHMETIC_FAMILY['diagnostic_name']} and vector "
                f"shape {ACTIVE_VECTOR_SHAPE}; pass --input only if "
                "the fixture already carries the matching typed compiler path"
            )
        return Path(default_input)
    if args.use_target_artifact_bundle and args.use_plan_and_export_bundle_front_door:
        return Path(ACTIVE_ARITHMETIC_FAMILY["default_plan_and_export_input"])
    return Path(ACTIVE_ARITHMETIC_FAMILY["default_input"])


def execution_planning_command_args(args: argparse.Namespace) -> list[str]:
    command_args: list[str] = []
    if args.lower_linalg_frontend:
        command_args.append("--tcrv-lower-linalg-i32-binary-to-exec")
    command_args.append("--tcrv-execution-planning-pipeline")
    return command_args


def run_bundle_bridge(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    if args.evidence_note:
        reject_secret_like_text("evidence note", args.evidence_note)

    run_id = safe_run_id(args.run_id or utc_run_id())
    artifact_dir = prepare_artifact_dir(
        selected_artifact_root(args), run_id, root, args.overwrite
    )
    commands: list[dict[str, Any]] = []

    input_path = resolve_repo_path(selected_input_path(args), root)
    if not input_path.exists():
        raise BridgeError(f"input MLIR does not exist: {selected_input_path(args)}")
    reject_secret_like_text("input MLIR path", str(selected_input_path(args)))

    tcrv_translate = resolve_tool(args.tcrv_translate, "tcrv-translate", root)
    local_clang = ensure_local_clang_on_path()

    bundle_dir = artifact_dir / "target_artifact_bundle"
    bundle_dir.mkdir()

    post_planning_path: Path | None = None
    post_planning_mlir = ""
    if args.use_plan_and_export_bundle_front_door:
        bundle_stdout, _, _ = run_command(
            "plan_and_export_target_artifact_bundle",
            [
                tcrv_translate,
                "--tcrv-plan-and-export-target-artifact-bundle",
                f"--tcrv-target-artifact-bundle-output-dir={relative_to_repo(bundle_dir, root)}",
                relative_to_repo(input_path, root),
            ],
            cwd=root,
            artifact_dir=artifact_dir,
            commands=commands,
            timeout_seconds=args.timeout,
        )
    else:
        tcrv_opt = resolve_tool(args.tcrv_opt, "tcrv-opt", root)
        post_planning_mlir, _, _ = run_command(
            "tcrv_opt_execution_planning_pipeline",
            [
                tcrv_opt,
                relative_to_repo(input_path, root),
                *execution_planning_command_args(args),
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

        bundle_stdout, _, _ = run_command(
            "export_target_artifact_bundle",
            [
                tcrv_translate,
                "--tcrv-export-target-artifact-bundle",
                f"--tcrv-target-artifact-bundle-output-dir={relative_to_repo(bundle_dir, root)}",
                relative_to_repo(post_planning_path, root),
            ],
            cwd=root,
            artifact_dir=artifact_dir,
            commands=commands,
            timeout_seconds=args.timeout,
        )
    bundle_stdout_path = artifact_dir / "bundle_export_stdout.txt"
    write_generated_text(
        bundle_stdout_path, "target artifact bundle export stdout", bundle_stdout
    )

    index_path = bundle_dir / BUNDLE_INDEX_FILE_NAME
    if not index_path.exists():
        raise BridgeError(f"target artifact bundle index was not emitted: {BUNDLE_INDEX_FILE_NAME}")
    index_text = index_path.read_text(encoding="utf-8")
    records = parse_target_artifact_bundle_index(index_text)
    selected_records = select_dispatch_bundle_records(records, bundle_dir)

    source_path = bundle_dir / str(selected_records["source"]["file_name"])
    header_path = bundle_dir / str(selected_records["header"]["file_name"])
    object_path = bundle_dir / str(selected_records["object"]["file_name"])
    source_text = source_path.read_text(encoding="utf-8")
    header_text = header_path.read_text(encoding="utf-8")
    source_vector_config = validate_library_dispatch_source(source_text)
    source_flags = {
        "selected_march": parse_source_comment(source_text, "selected_march", required=True),
        "selected_mabi": parse_source_comment(source_text, "selected_mabi", required=False),
        "vector_config": source_vector_config,
    }
    if "v" not in source_flags["selected_march"].lower():
        raise BridgeError("selected_march from bundled dispatch source must contain RVV vector evidence")

    dispatch_signature = require_dispatch_runtime_abi_signature(
        selected_records["header"]
    )
    dispatcher_function = validate_generated_dispatch_header(
        header_text, dispatch_signature
    )
    caller_text = build_dispatch_external_caller_source(
        dispatcher_function,
        str(selected_records["header"]["file_name"]),
        dispatch_signature,
    )
    caller_path = artifact_dir / "rvv_bundle_dispatch_external_caller.c"
    write_generated_text(
        caller_path,
        "generated RVV+scalar dispatch bundle external caller",
        caller_text,
    )

    hashes = {
        "input_sha256": sha256_file(input_path),
        "bundle_export_stdout_sha256": sha256_text(bundle_stdout),
        "bundle_index_sha256": sha256_text(index_text),
        "bundle_dispatch_source_sha256": sha256_text(source_text),
        "bundle_dispatch_header_sha256": sha256_text(header_text),
        "bundle_dispatch_object_sha256": sha256_file(object_path),
        "bundle_external_caller_c_sha256": sha256_text(caller_text),
    }
    if post_planning_path is not None:
        hashes["post_planning_mlir_sha256"] = sha256_text(post_planning_mlir)

    bundle_export_mode = (
        "plan-and-export-target-artifact-bundle"
        if args.use_plan_and_export_bundle_front_door
        else "target-artifact-bundle"
    )
    planned_dispatch_pipeline = (
        "tcrv-plan-and-export-target-artifact-bundle"
        if args.use_plan_and_export_bundle_front_door
        else "tcrv-execution-planning-pipeline"
    )
    artifacts = {
        "bundle_export_stdout": relative_to_repo(bundle_stdout_path, root),
        "bundle_dir": relative_to_repo(bundle_dir, root),
        "bundle_index": relative_to_repo(index_path, root),
        "bundle_dispatch_source": relative_to_repo(source_path, root),
        "bundle_dispatch_header": relative_to_repo(header_path, root),
        "bundle_dispatch_object": relative_to_repo(object_path, root),
        "bundle_external_caller_c": relative_to_repo(caller_path, root),
    }
    if post_planning_path is not None:
        artifacts["post_planning_mlir"] = relative_to_repo(
            post_planning_path, root
        )

    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runner": SCRIPT_NAME,
        "run_id": run_id,
        "git_sha": git_sha(root),
        "mode": "dry-run" if args.dry_run else "ssh",
        "status": "success",
        "arithmetic_family": str(ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"]),
        "vector_shape": ACTIVE_VECTOR_SHAPE,
        "input": relative_to_repo(input_path, root),
        "artifact_dir": relative_to_repo(artifact_dir, root),
        "planned_dispatch_pipeline": planned_dispatch_pipeline,
        "bundle_export_mode": bundle_export_mode,
        "rvv_config": source_vector_config,
        "bundle_index": relative_to_repo(index_path, root),
        "bundle_index_summary": bundle_records_summary(records),
        "local_object_export_clang": sanitize_text(local_clang),
        "selected_bundle_records": {
            label: bundle_records_summary([record])[0]
            for label, record in selected_records.items()
        },
        "selected_artifact_paths": {
            "bundle_index": relative_to_repo(index_path, root),
            "source": relative_to_repo(source_path, root),
            "header": relative_to_repo(header_path, root),
            "object": relative_to_repo(object_path, root),
            "external_caller": relative_to_repo(caller_path, root),
        },
        "source_export_mode": "runtime-callable-library",
        "external_caller": {
            "kind": "generated-c-caller",
            "dispatcher_function": dispatcher_function,
            "runtime_abi_signature": dispatch_signature,
            "success_marker": BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER,
            "branches_exercised": ["rvv_available=0", "rvv_available=1"],
            "runtime_element_counts": [7, 16],
        },
        "selected_compile_flags": remote_compile_flags(source_flags),
        "pass_fail_result": "pass",
        "hashes": hashes,
        "artifacts": artifacts,
        "commands": commands,
        "ssh_evidence": None,
        "claim_scope": (
            "local dry-run verifies bundle export, index parsing, file discovery, and external caller construction only"
            if args.dry_run
            else "bounded RVV+scalar "
            + str(ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"])
            + " target-artifact bundle external caller correctness only"
        ),
    }

    if not args.dry_run:
        try:
            evidence["ssh_evidence"] = run_remote_bundle_external_abi_evidence(
                args,
                root=root,
                artifact_dir=artifact_dir,
                commands=commands,
                source_path=source_path,
                header_path=header_path,
                object_path=object_path,
                caller_path=caller_path,
                source_file_name=str(selected_records["source"]["file_name"]),
                object_file_name=str(selected_records["object"]["file_name"]),
                flags=source_flags,
                run_id=run_id,
            )
            evidence["commands"] = commands
        except BridgeError as error:
            evidence["status"] = "failure"
            evidence["pass_fail_result"] = "fail"
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


def run_bridge(args: argparse.Namespace) -> dict[str, Any]:
    if args.use_target_artifact_bundle:
        return run_bundle_bridge(args)

    root = repo_root()
    if args.evidence_note:
        reject_secret_like_text("evidence note", args.evidence_note)

    run_id = safe_run_id(args.run_id or utc_run_id())
    artifact_dir = prepare_artifact_dir(
        Path(args.artifact_root), run_id, root, args.overwrite
    )
    commands: list[dict[str, Any]] = []

    input_path = resolve_repo_path(selected_input_path(args), root)
    if not input_path.exists():
        raise BridgeError(f"input MLIR does not exist: {selected_input_path(args)}")
    reject_secret_like_text("input MLIR path", str(selected_input_path(args)))

    tcrv_opt = resolve_tool(args.tcrv_opt, "tcrv-opt", root)
    tcrv_translate = resolve_tool(args.tcrv_translate, "tcrv-translate", root)

    post_planning_mlir, _, _ = run_command(
        "tcrv_opt_execution_planning_pipeline",
        [
            tcrv_opt,
            str(input_path),
            *execution_planning_command_args(args),
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
    library_vector_config = validate_library_dispatch_source(library_source_text)
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
            str(ACTIVE_ARITHMETIC_FAMILY["self_check_route"]),
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
        "arithmetic_family": str(ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"]),
        "vector_shape": ACTIVE_VECTOR_SHAPE,
        "input": relative_to_repo(input_path, root),
        "artifact_dir": relative_to_repo(artifact_dir, root),
        "planned_dispatch_pipeline": "tcrv-execution-planning-pipeline",
        "library_source_export_route": "generic-target-source-artifact",
        "self_check_source_export_route": "direct-rvv-scalar-dispatch-self-check",
        "source_export_mode": "self-check-harness",
        "rvv_config": source_flags["vector_config"],
        "library_rvv_config": library_vector_config,
        "self_check": {
            "branches_exercised": ["rvv_available=0", "rvv_available=1"],
            "runtime_element_counts": [7, 16],
            "success_marker": SUCCESS_MARKER,
        },
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
            else "bounded RVV+scalar "
            + str(ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"])
            + " dispatch self-check executable runtime only"
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
    configure_arithmetic_family("i32-vadd")
    configure_vector_shape("i32m1")

    def sample_shape_payload(shape_name: str) -> dict[str, Any]:
        if shape_name == "i32m1":
            return {
                "shape": "i32m1",
                "sew_bits": 32,
                "lmul": "m1",
                "tail_policy": "agnostic",
                "mask_policy": "agnostic",
                "vector_type": "vint32m1_t",
                "vector_suffix": "i32m1",
                "setvl_suffix": "e32m1",
                "capability_ids": [
                    "rvv.i32_m1.sew32",
                    "rvv.i32_m1.lmul_m1",
                    "rvv.i32_m1.tail_policy.agnostic",
                    "rvv.i32_m1.mask_policy.agnostic",
                ],
            }
        if shape_name == "i32m2":
            return {
                "shape": "i32m2",
                "sew_bits": 32,
                "lmul": "m2",
                "tail_policy": "agnostic",
                "mask_policy": "agnostic",
                "vector_type": "vint32m2_t",
                "vector_suffix": "i32m2",
                "setvl_suffix": "e32m2",
                "capability_ids": [
                    "rvv.i32_m2.sew32",
                    "rvv.i32_m2.lmul_m2",
                    "rvv.i32_m2.tail_policy.agnostic",
                    "rvv.i32_m2.mask_policy.agnostic",
                ],
            }
        raise AssertionError(f"unsupported self-test shape: {shape_name}")

    def sample_vector_shape_comments(shape_name: str) -> str:
        shape = sample_shape_payload(shape_name)
        return "\n".join(
            [
                "/* selected_vector_shape_config: shape="
                + str(shape["shape"])
                + ", sew="
                + str(shape["sew_bits"])
                + ", lmul="
                + str(shape["lmul"])
                + ", tail_policy="
                + str(shape["tail_policy"])
                + ", mask_policy="
                + str(shape["mask_policy"])
                + ", vector_type="
                + str(shape["vector_type"])
                + ", vector_suffix="
                + str(shape["vector_suffix"])
                + ", setvl_suffix="
                + str(shape["setvl_suffix"])
                + " */",
                "/* selected_vector_shape_capabilities: "
                + " ".join(str(capability) for capability in shape["capability_ids"])
                + " */",
                "/* control_plane_config: sew="
                + str(shape["sew_bits"])
                + ", lmul="
                + str(shape["lmul"])
                + ", policy=#tcrv_rvv.policy<tail = "
                + str(shape["tail_policy"])
                + ", mask = "
                + str(shape["mask_policy"])
                + "> */",
                "/* intrinsic_config: vector_type="
                + str(shape["vector_type"])
                + ", vector_suffix="
                + str(shape["vector_suffix"])
                + ", setvl_suffix="
                + str(shape["setvl_suffix"])
                + ", tail_policy="
                + str(shape["tail_policy"])
                + ", mask_policy="
                + str(shape["mask_policy"])
                + " */",
            ]
        )

    def sample_vector_intrinsics(
        family: dict[str, str | Path], shape_name: str, operator: str
    ) -> str:
        shape = sample_shape_payload(shape_name)
        return (
            "void f(void) {\n"
            f"  {setvl_intrinsic_for_suffix(str(shape['setvl_suffix']))};\n"
            f"  {load_intrinsic_for_suffix(str(shape['vector_suffix']))};\n"
            f"  {arithmetic_intrinsic_for_family(family, str(shape['vector_suffix']))};\n"
            f"  {store_intrinsic_for_suffix(str(shape['vector_suffix']))};\n"
            f"  out[index] = lhs[index] {operator} rhs[index];\n"
            "}"
        )

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

    sample_source = f"""
/* TianChen-RV RVV+scalar host runtime dispatch C export. */
/* Runtime guard: explicit host-provided rvv_available parameter; no automatic hardware probe is generated. */
/* selected_march: rv64gcv */
/* selected_mabi: lp64d */
{sample_vector_shape_comments(ACTIVE_VECTOR_SHAPE)}
#include <riscv_vector.h>
void tcrv_dispatch_i32_vadd_self_test(void) {{}}
{sample_vector_intrinsics(ACTIVE_ARITHMETIC_FAMILY, ACTIVE_VECTOR_SHAPE, "+")}
/* Explicit bounded self-check harness for RVV+scalar dispatch runtime invocation evidence. */
/* Harness scope: calls the generated dispatcher with explicit n values 7 and 16 for rvv_available = 0 and rvv_available = 1. */
int main(void) {{ puts("tcrv_rvv_scalar_i32_vadd_dispatch_self_check_ok runtime_counts=7,16 branches=scalar_and_rvv"); }}
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

    configure_arithmetic_family("i32-vsub")
    sample_vsub_source = f"""
/* TianChen-RV RVV+scalar host runtime dispatch C export. */
/* Runtime guard: explicit host-provided rvv_available parameter; no automatic hardware probe is generated. */
/* selected_march: rv64gcv */
/* selected_mabi: lp64d */
{sample_vector_shape_comments(ACTIVE_VECTOR_SHAPE)}
#include <riscv_vector.h>
void tcrv_dispatch_i32_vsub_self_test(void) {{}}
{sample_vector_intrinsics(ACTIVE_ARITHMETIC_FAMILY, ACTIVE_VECTOR_SHAPE, "-")}
/* Explicit bounded self-check harness for RVV+scalar dispatch runtime invocation evidence. */
/* Harness scope: calls the generated dispatcher with explicit n values 7 and 16 for rvv_available = 0 and rvv_available = 1. */
int main(void) {{ puts("tcrv_rvv_scalar_i32_vsub_dispatch_self_check_ok runtime_counts=7,16 branches=scalar_and_rvv"); }}
""".strip()
    vsub_flags = validate_self_check_dispatch_source(sample_vsub_source)
    assert_self_test(
        vsub_flags["selected_march"] == "rv64gcv",
        "vsub selected march parser failed",
    )
    vsub_caller = build_dispatch_external_caller_source(
        "tcrv_dispatch_i32_vsub_self_test",
        "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vsub-dispatch-header.h",
        DISPATCH_RUNTIME_ABI_SIGNATURE,
    )
    assert_self_test(
        "lhs[index] - rhs[index]" in vsub_caller,
        "vsub bundle caller did not check subtract semantics",
    )
    assert_self_test(
        BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER in vsub_caller,
        "vsub bundle caller success marker missing",
    )

    configure_vector_shape("i32m2")
    sample_vsub_m2_source = f"""
/* TianChen-RV RVV+scalar host runtime dispatch C export. */
/* Runtime guard: explicit host-provided rvv_available parameter; no automatic hardware probe is generated. */
/* selected_march: rv64gcv */
/* selected_mabi: lp64d */
{sample_vector_shape_comments(ACTIVE_VECTOR_SHAPE)}
#include <riscv_vector.h>
void tcrv_dispatch_i32_vsub_self_test(void) {{}}
{sample_vector_intrinsics(ACTIVE_ARITHMETIC_FAMILY, ACTIVE_VECTOR_SHAPE, "-")}
/* Explicit bounded self-check harness for RVV+scalar dispatch runtime invocation evidence. */
/* Harness scope: calls the generated dispatcher with explicit n values 7 and 16 for rvv_available = 0 and rvv_available = 1. */
int main(void) {{ puts("tcrv_rvv_scalar_i32_vsub_dispatch_self_check_ok runtime_counts=7,16 branches=scalar_and_rvv"); }}
""".strip()
    vsub_m2_flags = validate_self_check_dispatch_source(sample_vsub_m2_source)
    assert_self_test(
        vsub_m2_flags["vector_config"]["lmul"] == "m2",
        "m2 vector-shape metadata was not preserved",
    )
    try:
        validate_self_check_dispatch_source(sample_vsub_source)
    except BridgeError as error:
        assert_self_test(
            "requested vector shape i32m2" in str(error)
            or "missing required snippets" in str(error),
            "m2 mode did not reject m1 dispatch source metadata",
        )
    else:
        raise AssertionError("m2 mode accepted m1 generated dispatch source")
    configure_vector_shape("i32m1")

    configure_arithmetic_family("i32-vmul")
    sample_vmul_source = f"""
/* TianChen-RV RVV+scalar host runtime dispatch C export. */
/* Runtime guard: explicit host-provided rvv_available parameter; no automatic hardware probe is generated. */
/* selected_march: rv64gcv */
/* selected_mabi: lp64d */
{sample_vector_shape_comments(ACTIVE_VECTOR_SHAPE)}
#include <riscv_vector.h>
void tcrv_dispatch_i32_vmul_self_test(void) {{}}
{sample_vector_intrinsics(ACTIVE_ARITHMETIC_FAMILY, ACTIVE_VECTOR_SHAPE, "*")}
/* Explicit bounded self-check harness for RVV+scalar dispatch runtime invocation evidence. */
/* Harness scope: calls the generated dispatcher with explicit n values 7 and 16 for rvv_available = 0 and rvv_available = 1. */
int main(void) {{ puts("tcrv_rvv_scalar_i32_vmul_dispatch_self_check_ok runtime_counts=7,16 branches=scalar_and_rvv"); }}
""".strip()
    vmul_flags = validate_self_check_dispatch_source(sample_vmul_source)
    assert_self_test(
        vmul_flags["selected_march"] == "rv64gcv",
        "vmul selected march parser failed",
    )
    vmul_caller = build_dispatch_external_caller_source(
        "tcrv_dispatch_i32_vmul_self_test",
        "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vmul-dispatch-header.h",
        DISPATCH_RUNTIME_ABI_SIGNATURE,
    )
    assert_self_test(
        "lhs[index] * rhs[index]" in vmul_caller,
        "vmul bundle caller did not check multiply semantics",
    )
    assert_self_test(
        BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER in vmul_caller,
        "vmul bundle caller success marker missing",
    )
    configure_arithmetic_family("i32-vadd")

    sample_bundle_index = """
tianchenrv.target_artifact_bundle.version: 1
bundle_status: "complete"
artifact_count: 3
artifact[0]:
  file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c"
  component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
  component_role: "source"
  external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
  selected_surface: "dispatch"
  component[0]:
    selected_variant: @rvv_first_slice
    role: "dispatch case"
  component[1]:
    selected_variant: @scalar_fallback_first_slice
    role: "dispatch fallback"
  artifact_kind: "runtime-callable-c-source"
  route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-c"
  owner: "rvv-scalar-dispatch-target"
  runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
  runtime_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
  runtime_abi_parameter[0]:
    c_name: "lhs"
    c_type: "const int32_t *"
    role: "lhs-input-buffer"
    ownership: "target-export-abi-owned"
  runtime_abi_parameter[1]:
    c_name: "rhs"
    c_type: "const int32_t *"
    role: "rhs-input-buffer"
    ownership: "target-export-abi-owned"
  runtime_abi_parameter[2]:
    c_name: "out"
    c_type: "int32_t *"
    role: "output-buffer"
    ownership: "target-export-abi-owned"
  runtime_abi_parameter[3]:
    c_name: "n"
    c_type: "size_t"
    role: "runtime-element-count"
    ownership: "target-export-abi-owned"
  runtime_abi_parameter[4]:
    c_name: "rvv_available"
    c_type: "int"
    role: "dispatch-availability-guard"
    ownership: "target-export-abi-owned"
  evidence_role: "compiler-artifact"
artifact[1]:
  file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h"
  component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
  component_role: "header"
  external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
  selected_surface: "dispatch"
  component[0]:
    selected_variant: @rvv_first_slice
    role: "dispatch case"
  component[1]:
    selected_variant: @scalar_fallback_first_slice
    role: "dispatch fallback"
  artifact_kind: "runtime-callable-c-header"
  route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-header"
  owner: "rvv-scalar-dispatch-target"
  runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
  runtime_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
  runtime_abi_parameter[0]:
    c_name: "lhs"
    c_type: "const int32_t *"
    role: "lhs-input-buffer"
    ownership: "target-export-abi-owned"
  runtime_abi_parameter[1]:
    c_name: "rhs"
    c_type: "const int32_t *"
    role: "rhs-input-buffer"
    ownership: "target-export-abi-owned"
  runtime_abi_parameter[2]:
    c_name: "out"
    c_type: "int32_t *"
    role: "output-buffer"
    ownership: "target-export-abi-owned"
  runtime_abi_parameter[3]:
    c_name: "n"
    c_type: "size_t"
    role: "runtime-element-count"
    ownership: "target-export-abi-owned"
  runtime_abi_parameter[4]:
    c_name: "rvv_available"
    c_type: "int"
    role: "dispatch-availability-guard"
    ownership: "target-export-abi-owned"
  evidence_role: "header-declaration"
artifact[2]:
  file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o"
  component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
  component_role: "object"
  external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
  selected_surface: "dispatch"
  component[0]:
    selected_variant: @rvv_first_slice
    role: "dispatch case"
  component[1]:
    selected_variant: @scalar_fallback_first_slice
    role: "dispatch fallback"
  artifact_kind: "riscv-elf-relocatable-object"
  route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-object"
  owner: "rvv-scalar-dispatch-target"
  runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
  runtime_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
  runtime_abi_parameter[0]:
    c_name: "lhs"
    c_type: "const int32_t *"
    role: "lhs-input-buffer"
    ownership: "target-export-abi-owned"
  runtime_abi_parameter[1]:
    c_name: "rhs"
    c_type: "const int32_t *"
    role: "rhs-input-buffer"
    ownership: "target-export-abi-owned"
  runtime_abi_parameter[2]:
    c_name: "out"
    c_type: "int32_t *"
    role: "output-buffer"
    ownership: "target-export-abi-owned"
  runtime_abi_parameter[3]:
    c_name: "n"
    c_type: "size_t"
    role: "runtime-element-count"
    ownership: "target-export-abi-owned"
  runtime_abi_parameter[4]:
    c_name: "rvv_available"
    c_type: "int"
    role: "dispatch-availability-guard"
    ownership: "target-export-abi-owned"
  evidence_role: "relocatable-object"
""".strip()
    bundle_records = parse_target_artifact_bundle_index(sample_bundle_index)
    assert_self_test(len(bundle_records) == 3, "bundle index parser lost records")
    assert_self_test(
        bundle_records[0]["components"][0]["role"] == "dispatch case",
        "bundle index parser lost component role",
    )
    assert_self_test(
        bundle_records[0]["component_group"]
        == DISPATCH_EXTERNAL_ABI_COMPONENT_GROUP,
        "bundle index parser lost artifact component_group",
    )
    assert_self_test(
        bundle_records[1]["component_role"] == "header",
        "bundle index parser lost artifact component_role",
    )
    assert_self_test(
        bundle_records[2]["external_abi_name"] == DISPATCH_EXTERNAL_ABI_NAME,
        "bundle index parser lost external_abi_name",
    )
    assert_self_test(
        bundle_records[0]["runtime_abi_parameters"][4]["role"]
        == "dispatch-availability-guard",
        "bundle index parser lost runtime ABI signature",
    )
    root = repo_root()
    with tempfile.TemporaryDirectory(dir=root / "artifacts" / "tmp") as temp_dir:
        bundle_dir = Path(temp_dir)
        for record in bundle_records:
            (bundle_dir / record["file_name"]).write_bytes(b"artifact")
        selected = select_dispatch_bundle_records(bundle_records, bundle_dir)
        assert_self_test(
            selected["object"]["artifact_kind"] == "riscv-elf-relocatable-object",
            "bundle dispatch object record was not selected",
        )
        assert_self_test(
            selected["source"]["component_role"] == "source",
            "bundle dispatch source record did not use explicit component_role",
        )
        assert_self_test(
            require_dispatch_runtime_abi_signature(selected["header"])
            == DISPATCH_RUNTIME_ABI_SIGNATURE,
            "bundle dispatch header record did not use explicit runtime ABI signature",
        )
        try:
            select_dispatch_bundle_records(
                [
                    dict(record, component_group="wrong-group")
                    if record["component_role"] == "object"
                    else record
                    for record in bundle_records
                ],
                bundle_dir=bundle_dir,
            )
        except BridgeError as error:
            assert_self_test("component_group" in str(error), "group mismatch error changed")
        else:
            raise AssertionError("bundle index with mismatched component_group was accepted")
        try:
            select_dispatch_bundle_records(
                [
                    dict(record, runtime_abi_parameters=[])
                    if record["component_role"] == "header"
                    else record
                    for record in bundle_records
                ],
                bundle_dir=bundle_dir,
            )
        except BridgeError as error:
            assert_self_test(
                "runtime_abi_parameter" in str(error),
                "missing runtime ABI signature error changed",
            )
        else:
            raise AssertionError("bundle index with missing runtime ABI signature was accepted")
    try:
        parse_target_artifact_bundle_index(
            sample_bundle_index.replace(
                '  runtime_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"\n',
                "",
                1,
            )
        )
    except BridgeError as error:
        assert_self_test("runtime_abi_name" in str(error), "missing bundle field error changed")
    else:
        raise AssertionError("bundle index with missing runtime_abi_name was accepted")
    try:
        parse_target_artifact_bundle_index(
            sample_bundle_index.replace(
                'file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h"',
                'file_name: "https://example.invalid/header.h"',
            )
        )
    except BridgeError:
        pass
    else:
        raise AssertionError("bundle index with secret-like URL text was accepted")

    dispatch_header = """\
#ifndef TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_SELF_TEST_H
#define TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_SELF_TEST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void tcrv_dispatch_i32_vadd_self_test(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);

#ifdef __cplusplus
}
#endif

#endif /* TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_SELF_TEST_H */
"""
    dispatch_function = validate_generated_dispatch_header(
        dispatch_header, DISPATCH_RUNTIME_ABI_SIGNATURE
    )
    assert_self_test(
        dispatch_function == "tcrv_dispatch_i32_vadd_self_test",
        "dispatch header prototype parser failed",
    )
    caller = build_dispatch_external_caller_source(
        dispatch_function,
        "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h",
        DISPATCH_RUNTIME_ABI_SIGNATURE,
    )
    assert_self_test(
        BUNDLE_EXTERNAL_ABI_SUCCESS_MARKER in caller,
        "bundle external caller success marker missing",
    )
    assert_self_test(
        "run_dispatch_case(7, 0)" in caller
        and "run_dispatch_case(16, 1)" in caller,
        "bundle external caller did not exercise multiple runtime element counts",
    )
    bundle_link_command = build_remote_bundle_link_executable_command(
        "/tmp/tianchenrv_rvv_bundle_e2e_self_test",
        "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o",
        "rvv_bundle_dispatch_external_caller_from_bundle_object",
        flags,
    )
    assert_self_test(
        "rvv_bundle_dispatch_external_caller.o" in bundle_link_command,
        "bundle link command missing caller object",
    )
    source_compile_command = build_remote_bundle_compile_dispatch_source_object_command(
        "/tmp/tianchenrv_rvv_bundle_e2e_self_test",
        "artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c",
        flags,
    )
    assert_self_test(
        "rvv_bundle_dispatch_from_source.o" in source_compile_command,
        "bundle source compile command missing source-built object",
    )

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
    parser.add_argument("--input", default="")
    parser.add_argument(
        "--arithmetic-family",
        choices=sorted(ARITHMETIC_FAMILY_SPECS),
        default="i32-vadd",
        help="Bounded dispatch arithmetic family to validate",
    )
    parser.add_argument(
        "--vector-shape",
        choices=sorted(SUPPORTED_RVV_VECTOR_SHAPES),
        default="i32m1",
        help=(
            "Bounded typed RVV i32 vector shape to validate; i32m2 selects "
            "the existing typed dispatch compiler/export fixture when available"
        ),
    )
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tcrv-opt", default="")
    parser.add_argument("--tcrv-translate", default="")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true", help="Do not contact ssh rvv")
    parser.add_argument(
        "--lower-linalg-frontend",
        action="store_true",
        help=(
            "Run the bounded linalg i32 add/sub/mul frontend lowering pass before "
            "the execution-planning pipeline"
        ),
    )
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--ssh-option", action="append", default=[])
    parser.add_argument("--evidence-note", default="")
    parser.add_argument(
        "--use-target-artifact-bundle",
        action="store_true",
        help="Export and consume the registry-derived target artifact bundle",
    )
    parser.add_argument(
        "--use-plan-and-export-bundle-front-door",
        action="store_true",
        help=(
            "In target artifact bundle mode, call the C++ tcrv-translate "
            "plan-and-export bundle front door instead of orchestrating "
            "tcrv-opt followed by bundle export"
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    configure_arithmetic_family(args.arithmetic_family)
    configure_vector_shape(args.vector_shape)
    if args.self_test:
        run_self_test()
        return 0
    if (
        args.use_plan_and_export_bundle_front_door
        and not args.use_target_artifact_bundle
    ):
        print(
            "rvv_scalar_dispatch_e2e: --use-plan-and-export-bundle-front-door "
            "requires --use-target-artifact-bundle",
            file=sys.stderr,
        )
        return 1

    try:
        evidence = run_bridge(args)
    except BridgeError as error:
        print(f"rvv_scalar_dispatch_e2e: {sanitize_text(str(error))}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "artifact_dir": evidence["artifact_dir"],
                "arithmetic_family": evidence.get("arithmetic_family", ""),
                "bundle_export_mode": evidence.get("bundle_export_mode", ""),
                "mode": evidence["mode"],
                "status": evidence["status"],
                "vector_shape": evidence.get("vector_shape", ""),
                "planned_dispatch_pipeline": evidence["planned_dispatch_pipeline"],
                "source_sha256": evidence["hashes"].get(
                    "dispatch_self_check_c_sha256",
                    evidence["hashes"].get("bundle_external_caller_c_sha256", ""),
                ),
                "source_export_mode": evidence["source_export_mode"],
                "self_check_source_export_route": evidence.get(
                    "self_check_source_export_route", ""
                ),
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
