#!/usr/bin/env python3
"""Prove generated RVV object/header bundle ABI consumption on ``ssh rvv``.

This is evidence tooling only. It invokes the existing MLIR/C++ compiler
front doors, checks the generated target artifact bundle, builds a small
external C ABI consumer, and optionally runs that consumer on the real RVV
target. It does not implement compiler IR, lowering, plugin selection,
emission, descriptors, fallback computation, or runtime glue.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any


SCRIPT_NAME = "rvv_generated_bundle_abi_e2e"
SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_ROOT = Path("artifacts/tmp/rvv_generated_bundle_abi_e2e")
DEFAULT_INPUT = Path("test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir")
DEFAULT_SSH_TARGET = "rvv"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_RUNTIME_COUNTS = (1, 7, 16, 17, 257)

INDEX_FILE_NAME = "tianchenrv-target-artifact-bundle.index"
EXPECTED_SELECTED_VARIANT = "vector_source_rvv_i32_add"
EXPECTED_SELECTED_ROLE = "dispatch case"
EXPECTED_COMPONENT_GROUP = "rvv-i32m1-arithmetic-materialized-emitc-bundle.v1"
EXPECTED_EXTERNAL_ABI_NAME = "rvv-i32m1-add-callable-c-abi.v1"
EXPECTED_RUNTIME_ABI_KIND = "plugin-owned-runtime-abi"
EXPECTED_OBJECT_ROUTE = "rvv-i32m1-arithmetic-emitc-route-family"
EXPECTED_HEADER_ROUTE = "rvv-i32m1-arithmetic-emitc-route-family.header"
EXPECTED_OWNER = "rvv-plugin"
EXPECTED_OBJECT_KIND = "riscv-elf-relocatable-object"
EXPECTED_HEADER_KIND = "runtime-callable-c-header"
EXPECTED_FUNCTION = "tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add"
EXPECTED_PROTOTYPE = (
    "void tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add("
    "const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);"
)
EXPECTED_RUNTIME_PARAMETERS = (
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
)
EXPECTED_METADATA = {
    "rvv_emitc_lowerable_route": "rvv-i32m1-add-emitc-route",
    "rvv_arithmetic_op": "add",
    "tcrv_rvv.config_contract": "rvv-i32m1-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1",
    "tcrv_rvv.runtime_vl_contract": "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1",
    "tcrv_rvv.runtime_avl_source": "runtime_abi:n",
    "tcrv_rvv.vl_def": "tcrv_rvv.setvl",
    "tcrv_rvv.vl_scope": "tcrv_rvv.with_vl",
    "tcrv_rvv.runtime_abi_order": "lhs,rhs,out,n",
    "tcrv_rvv.runtime_avl_abi_parameter": "n",
    "tcrv_rvv.emitc_loop": "emitc.for",
    "tcrv_rvv.loop_induction": "offset",
    "tcrv_rvv.loop_step": "full_chunk_vl",
    "tcrv_rvv.remaining_avl": "n-offset",
    "tcrv_rvv.pointer_advance": "offset",
    "tcrv_rvv.bounded_slice": "multi-vl-i32m1-arithmetic",
    "tcrv_rvv.multi_vl": "supported",
}
FORBIDDEN_HEADER_TOKENS = (
    "__riscv_",
    "vint32m1_t",
    "return;",
    "int main",
    "descriptor",
    "direct-C",
    "direct_c",
    "source-export",
    "source_export",
    "rvv-direct-microkernel",
)

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
            r"(?i)\b((?:token|secret|password|passwd|api[_-]?key|access[_-]?key)"
            r"\s*[:=]\s*)[^\s]+"
        ),
        r"\1[REDACTED]",
    ),
)


class EvidenceError(RuntimeError):
    pass


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_run_id(raw_run_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_run_id.strip())
    return sanitized or utc_run_id()


def sanitize_text(text: Any, limit: int | None = None) -> str:
    if text is None:
        value = ""
    elif isinstance(text, bytes):
        value = text.decode("utf-8", errors="replace")
    else:
        value = str(text)
    value = value.replace("\x00", "\\0")
    for pattern, replacement in SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    if limit is not None and len(value) > limit:
        return value[:limit] + "...<truncated>"
    return value


def command_display(command: list[str]) -> str:
    return sanitize_text(shlex.join(command))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def ensure_tool(path_or_name: str) -> str:
    candidate = Path(path_or_name)
    if candidate.exists():
        return str(candidate)
    resolved = shutil.which(path_or_name)
    if resolved:
        return resolved
    raise EvidenceError(f"required tool not found: {path_or_name}")


def default_readobj() -> str:
    for candidate in ("llvm-readobj", "/usr/lib/llvm-20/bin/llvm-readobj"):
        resolved = shutil.which(candidate) if candidate == "llvm-readobj" else None
        if resolved:
            return resolved
        if Path(candidate).exists():
            return candidate
    return "llvm-readobj"


def run_command(
    command: list[str],
    *,
    input_data: bytes | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    cwd: Path | None = None,
) -> dict[str, Any]:
    started = utc_timestamp()
    try:
        result = subprocess.run(
            command,
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command_display(command),
            "started_at": started,
            "finished_at": utc_timestamp(),
            "exit_code": result.returncode,
            "stdout": sanitize_text(result.stdout, limit=8192),
            "stderr": sanitize_text(result.stderr, limit=8192),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command_display(command),
            "started_at": started,
            "finished_at": utc_timestamp(),
            "exit_code": None,
            "timeout": True,
            "stdout": sanitize_text(exc.stdout, limit=8192),
            "stderr": sanitize_text(exc.stderr, limit=8192),
        }
    except OSError as exc:
        return {
            "command": command_display(command),
            "started_at": started,
            "finished_at": utc_timestamp(),
            "exit_code": None,
            "os_error": sanitize_text(exc),
            "stdout": "",
            "stderr": "",
        }


def require_command_success(record: dict[str, Any], context: str) -> None:
    if record.get("exit_code") == 0 and not record.get("timeout"):
        return
    raise EvidenceError(
        f"{context} failed: command={record.get('command')} "
        f"exit={record.get('exit_code')} stderr={record.get('stderr', '')[:512]}"
    )


def parse_value(raw: str) -> str:
    value = raw.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if value.startswith("@"):
        return value[1:]
    return value


def parse_index_block(block: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "runtime_abi_parameters": [],
        "artifact_metadata": [],
        "components": [],
    }
    current: dict[str, str] | None = None
    current_kind: str | None = None

    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("  runtime_abi_parameter["):
            current = {}
            current_kind = "runtime_abi_parameters"
            record[current_kind].append(current)
            continue
        if line.startswith("  artifact_metadata["):
            current = {}
            current_kind = "artifact_metadata"
            record[current_kind].append(current)
            continue
        if line.startswith("  component["):
            current = {}
            current_kind = "components"
            record[current_kind].append(current)
            continue
        if line.startswith("    ") and current is not None and ":" in line:
            key, value = line.strip().split(":", 1)
            current[key] = parse_value(value)
            continue
        if line.startswith("  ") and ":" in line:
            current = None
            current_kind = None
            key, value = line.strip().split(":", 1)
            record[key] = parse_value(value)
            continue
    return record


def parse_bundle_index(index_text: str) -> dict[str, Any]:
    root: dict[str, Any] = {"records": []}
    for line in index_text.splitlines():
        if line.startswith("tianchenrv.target_artifact_bundle.version:"):
            root["version"] = line.split(":", 1)[1].strip()
        elif line.startswith("bundle_status:"):
            root["bundle_status"] = parse_value(line.split(":", 1)[1])
        elif line.startswith("artifact_count:"):
            root["artifact_count"] = line.split(":", 1)[1].strip()

    matches = list(re.finditer(r"^artifact\[[0-9]+\]:$", index_text, re.MULTILINE))
    for position, match in enumerate(matches):
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(index_text)
        root["records"].append(parse_index_block(index_text[start:end]))
    return root


def metadata_map(record: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in record.get("artifact_metadata", []):
        key = entry.get("key", "")
        value = entry.get("value", "")
        if key:
            result[key] = value
    return result


def require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual == expected:
        return
    raise EvidenceError(f"{context}: expected {expected!r}, got {actual!r}")


def require_contains(text: str, needle: str, context: str) -> None:
    if needle in text:
        return
    raise EvidenceError(f"{context}: missing {needle!r}")


def require_not_contains(text: str, needle: str, context: str) -> None:
    if needle not in text:
        return
    raise EvidenceError(f"{context}: forbidden token {needle!r} present")


def find_record(records: list[dict[str, Any]], component_role: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("component_role") == component_role]
    if len(matches) != 1:
        raise EvidenceError(
            f"expected exactly one {component_role} bundle component, found {len(matches)}"
        )
    return matches[0]


def verify_runtime_parameters(record: dict[str, Any], context: str) -> None:
    require_equal(
        record.get("runtime_abi_parameter_count"),
        str(len(EXPECTED_RUNTIME_PARAMETERS)),
        f"{context} runtime ABI parameter count",
    )
    require_equal(
        record.get("runtime_abi_parameters"),
        list(EXPECTED_RUNTIME_PARAMETERS),
        f"{context} ordered runtime ABI parameters",
    )


def verify_common_record_fields(record: dict[str, Any], context: str) -> None:
    require_equal(record.get("component_group"), EXPECTED_COMPONENT_GROUP, f"{context} component group")
    require_equal(record.get("external_abi_name"), EXPECTED_EXTERNAL_ABI_NAME, f"{context} external ABI")
    require_equal(record.get("selected_variant"), EXPECTED_SELECTED_VARIANT, f"{context} selected variant")
    require_equal(record.get("role"), EXPECTED_SELECTED_ROLE, f"{context} selected role")
    require_equal(record.get("owner"), EXPECTED_OWNER, f"{context} owner")
    require_equal(record.get("runtime_abi"), EXPECTED_EXTERNAL_ABI_NAME, f"{context} runtime ABI")
    require_equal(record.get("runtime_abi_kind"), EXPECTED_RUNTIME_ABI_KIND, f"{context} runtime ABI kind")
    require_equal(record.get("runtime_abi_name"), EXPECTED_EXTERNAL_ABI_NAME, f"{context} runtime ABI name")
    verify_runtime_parameters(record, context)
    components = record.get("components", [])
    require_equal(len(components), 1, f"{context} selected component count")
    require_equal(
        components[0].get("selected_variant"),
        EXPECTED_SELECTED_VARIANT,
        f"{context} selected component variant",
    )
    require_equal(
        components[0].get("role"),
        EXPECTED_SELECTED_ROLE,
        f"{context} selected component role",
    )


def verify_record_metadata(record: dict[str, Any], context: str) -> None:
    metadata = metadata_map(record)
    for key, expected in EXPECTED_METADATA.items():
        require_equal(metadata.get(key), expected, f"{context} metadata {key}")
    for key in metadata:
        lowered = key.lower()
        if "descriptor" in lowered or "element-count" in lowered or "element_count" in lowered:
            raise EvidenceError(f"{context} metadata key {key!r} is descriptor residue")


def verify_header(header_path: Path) -> dict[str, Any]:
    if not header_path.exists():
        raise EvidenceError(f"generated header is missing: {header_path}")
    text = header_path.read_text(encoding="utf-8")
    require_contains(text, "#include <stddef.h>", "generated header")
    require_contains(text, "#include <stdint.h>", "generated header")
    require_contains(text, EXPECTED_PROTOTYPE, "generated header")
    require_contains(text, "tianchenrv.rvv.runtime_avl_source: runtime_abi:n", "generated header")
    require_contains(text, "tianchenrv.rvv.multi_vl: supported", "generated header")
    for token in FORBIDDEN_HEADER_TOKENS:
        require_not_contains(text, token, "generated declaration-only header")
    return {
        "path": str(header_path),
        "size": header_path.stat().st_size,
        "sha256": sha256_file(header_path),
        "prototype": EXPECTED_PROTOTYPE,
    }


def verify_object(object_path: Path, readobj: str | None) -> dict[str, Any]:
    if not object_path.exists():
        raise EvidenceError(f"generated object is missing: {object_path}")
    size = object_path.stat().st_size
    if size == 0:
        raise EvidenceError(f"generated object is empty: {object_path}")
    result = {
        "path": str(object_path),
        "size": size,
        "sha256": sha256_file(object_path),
    }
    if readobj:
        readobj_record = run_command([readobj, "-h", str(object_path)], timeout=30)
        require_command_success(readobj_record, "llvm-readobj header check")
        stdout = str(readobj_record.get("stdout", ""))
        require_contains(stdout, "Format: elf64-littleriscv", "llvm-readobj header check")
        require_contains(stdout, "Arch: riscv64", "llvm-readobj header check")
        require_contains(stdout, "Type: Relocatable", "llvm-readobj header check")
        result["readobj"] = readobj_record
    return result


def verify_bundle(bundle_dir: Path, readobj: str | None) -> dict[str, Any]:
    index_path = bundle_dir / INDEX_FILE_NAME
    if not index_path.exists():
        raise EvidenceError(f"bundle index is missing: {index_path}")
    index_text = index_path.read_text(encoding="utf-8")
    parsed = parse_bundle_index(index_text)
    require_equal(parsed.get("version"), "1", "bundle version")
    require_equal(parsed.get("bundle_status"), "complete", "bundle status")
    require_equal(parsed.get("artifact_count"), "2", "bundle artifact count")
    records = parsed.get("records", [])
    require_equal(len(records), 2, "bundle record count")

    object_record = find_record(records, "object")
    header_record = find_record(records, "header")
    verify_common_record_fields(object_record, "object record")
    verify_common_record_fields(header_record, "header record")
    verify_record_metadata(object_record, "object record")
    verify_record_metadata(header_record, "header record")
    require_equal(object_record.get("artifact_kind"), EXPECTED_OBJECT_KIND, "object artifact kind")
    require_equal(object_record.get("route"), EXPECTED_OBJECT_ROUTE, "object route")
    require_equal(object_record.get("handoff_kind"), "materialized-emitc-cpp-rvv-intrinsic-object", "object handoff")
    require_equal(object_record.get("evidence_role"), "relocatable-object", "object evidence role")
    require_equal(header_record.get("artifact_kind"), EXPECTED_HEADER_KIND, "header artifact kind")
    require_equal(header_record.get("route"), EXPECTED_HEADER_ROUTE, "header route")
    require_equal(header_record.get("handoff_kind"), "materialized-emitc-cpp-rvv-intrinsic-object", "header handoff")
    require_equal(header_record.get("evidence_role"), "header-declaration", "header evidence role")

    object_file = object_record.get("file_name")
    header_file = header_record.get("file_name")
    if not object_file or not header_file:
        raise EvidenceError("bundle records must carry generated file names")
    object_path = bundle_dir / object_file
    header_path = bundle_dir / header_file
    return {
        "index": {
            "path": str(index_path),
            "size": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
            "parsed": parsed,
        },
        "object": verify_object(object_path, readobj),
        "header": verify_header(header_path),
        "object_file": object_file,
        "header_file": header_file,
    }


def harness_source(header_file_name: str, runtime_counts: list[int]) -> str:
    counts = ", ".join(str(count) for count in runtime_counts)
    return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  size_t alloc_n = n == 0 ? 1 : n;
  int32_t *lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!lhs || !rhs || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(lhs);
    free(rhs);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < n; ++index) {{
    lhs[index] = (int32_t)(7 + (int32_t)(index * 3));
    rhs[index] = (int32_t)(1000 - (int32_t)(index * 5));
    out[index] = (int32_t)0x5a5a5a5a;
  }}

  {EXPECTED_FUNCTION}(lhs, rhs, out, n);

  for (size_t index = 0; index < n; ++index) {{
    int32_t expected = lhs[index] + rhs[index];
    if (out[index] != expected) {{
      fprintf(stderr,
              "mismatch n=%zu index=%zu got=%d expected=%d lhs=%d rhs=%d\\n",
              n, index, out[index], expected, lhs[index], rhs[index]);
      free(lhs);
      free(rhs);
      free(out);
      return 12;
    }}
  }}

  free(lhs);
  free(rhs);
  free(out);
  printf("case n=%zu ok\\n", n);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  for (size_t index = 0; index < count_count; ++index) {{
    int status = run_case(counts[index]);
    if (status != 0)
      return status;
  }}
  printf("tcrv_rvv_generated_bundle_abi_ok counts={','.join(str(c) for c in runtime_counts)}\\n");
  return 0;
}}
""".lstrip()


def generate_bundle(
    tcrv_opt: str,
    tcrv_translate: str,
    input_path: Path,
    bundle_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    opt_command = [
        tcrv_opt,
        str(input_path),
        "--tcrv-source-artifact-front-door-pipeline",
    ]
    opt_record = run_command(opt_command, timeout=timeout)
    require_command_success(opt_record, "tcrv-opt source artifact front-door pipeline")

    translate_command = [
        tcrv_translate,
        "--tcrv-export-target-artifact-bundle",
        f"--tcrv-target-artifact-bundle-output-dir={bundle_dir}",
    ]
    translate_record = run_command(
        translate_command,
        input_data=str(opt_record.get("stdout", "")).encode("utf-8"),
        timeout=timeout,
    )
    require_command_success(translate_record, "tcrv-translate target artifact bundle export")
    return {
        "pipeline": f"{command_display(opt_command)} | {command_display(translate_command)}",
        "tcrv_opt": opt_record,
        "tcrv_translate": translate_record,
    }


def ssh_base_command(ssh_target: str, connect_timeout: int) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        ssh_target,
    ]


def scp_base_command(connect_timeout: int) -> list[str]:
    return ["scp", "-q", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_timeout}"]


def remote_quote(value: str) -> str:
    return shlex.quote(value)


def run_remote_shell(
    ssh_target: str,
    connect_timeout: int,
    command: str,
    timeout: int,
) -> dict[str, Any]:
    remote_invocation = shlex.join(["sh", "-lc", command])
    return run_command(
        ssh_base_command(ssh_target, connect_timeout) + [remote_invocation],
        timeout=timeout,
    )


def run_remote_evidence(
    *,
    artifact_dir: Path,
    run_id: str,
    ssh_target: str,
    connect_timeout: int,
    timeout: int,
    object_path: Path,
    header_path: Path,
    harness_path: Path,
) -> dict[str, Any]:
    remote_dir = f"/tmp/tianchenrv_rvv_generated_bundle_abi_{safe_run_id(run_id)}"
    remote_object = f"{remote_dir}/{object_path.name}"
    remote_header = f"{remote_dir}/{header_path.name}"
    remote_harness = f"{remote_dir}/{harness_path.name}"
    remote_binary = f"{remote_dir}/rvv_generated_bundle_abi_harness"

    commands: dict[str, Any] = {"remote_dir": remote_dir}
    setup = run_remote_shell(
        ssh_target,
        connect_timeout,
        f"rm -rf {remote_quote(remote_dir)} && mkdir -p {remote_quote(remote_dir)}",
        timeout,
    )
    commands["setup"] = setup
    require_command_success(setup, "remote setup")

    scp_command = scp_base_command(connect_timeout) + [
        str(object_path),
        str(header_path),
        str(harness_path),
        f"{ssh_target}:{remote_dir}/",
    ]
    scp_record = run_command(scp_command, timeout=timeout)
    commands["scp"] = scp_record
    require_command_success(scp_record, "remote artifact staging")

    compile_command = (
        f"cd {remote_quote(remote_dir)} && "
        "printf 'remote_arch=' && uname -m && "
        "printf 'clang_path=' && command -v clang && "
        "printf 'clang_version=' && clang --version | head -n 1 && "
        f"clang -O2 -march=rv64gcv -mabi=lp64d -I. "
        f"{remote_quote(remote_harness)} {remote_quote(remote_object)} "
        f"-o {remote_quote(remote_binary)}"
    )
    compile_record = run_remote_shell(
        ssh_target, connect_timeout, compile_command, timeout
    )
    commands["compile"] = compile_record
    require_command_success(compile_record, "remote clang compile/link")

    run_record = run_remote_shell(
        ssh_target, connect_timeout, remote_quote(remote_binary), timeout
    )
    commands["run"] = run_record
    require_command_success(run_record, "remote generated bundle ABI harness run")
    require_contains(
        str(run_record.get("stdout", "")),
        "tcrv_rvv_generated_bundle_abi_ok",
        "remote generated bundle ABI harness output",
    )

    cleanup = run_remote_shell(
        ssh_target, connect_timeout, f"rm -rf {remote_quote(remote_dir)}", timeout
    )
    commands["cleanup"] = cleanup
    if cleanup.get("exit_code") != 0:
        commands["cleanup_warning"] = "remote cleanup failed"

    (artifact_dir / "remote_compile_stdout.txt").write_text(
        str(compile_record.get("stdout", "")), encoding="utf-8"
    )
    (artifact_dir / "remote_run_stdout.txt").write_text(
        str(run_record.get("stdout", "")), encoding="utf-8"
    )
    return {
        "ssh_target": ssh_target,
        "remote_dir": remote_dir,
        "remote_object": remote_object,
        "remote_header": remote_header,
        "remote_harness": remote_harness,
        "remote_binary": remote_binary,
        "commands": commands,
        "remote_compile_succeeded": compile_record.get("exit_code") == 0,
        "remote_run_succeeded": run_record.get("exit_code") == 0,
        "remote_output": sanitize_text(run_record.get("stdout", ""), limit=4096),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_artifact_dir(root: Path, run_id: str, overwrite: bool) -> Path:
    artifact_dir = root / safe_run_id(run_id)
    if artifact_dir.exists():
        if not overwrite:
            raise EvidenceError(
                f"artifact directory already exists; pass --overwrite to replace: {artifact_dir}"
            )
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    return artifact_dir


def run_e2e(args: argparse.Namespace) -> int:
    run_id = safe_run_id(args.run_id or utc_run_id())
    artifact_dir = prepare_artifact_dir(args.artifact_root, run_id, args.overwrite)
    bundle_dir = artifact_dir / "generated_bundle"
    bundle_dir.mkdir()
    tcrv_opt = ensure_tool(args.tcrv_opt)
    tcrv_translate = ensure_tool(args.tcrv_translate)
    readobj = ensure_tool(args.llvm_readobj) if args.llvm_readobj else None

    runtime_counts = args.runtime_count or list(DEFAULT_RUNTIME_COUNTS)
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": SCRIPT_NAME,
        "status": "started",
        "created_at": utc_timestamp(),
        "run_id": run_id,
        "dry_run": bool(args.dry_run),
        "input": str(args.input),
        "artifact_dir": str(artifact_dir),
        "runtime_counts": runtime_counts,
    }
    try:
        local = generate_bundle(tcrv_opt, tcrv_translate, args.input, bundle_dir, args.timeout)
        evidence["local_bundle_generation"] = local
        bundle_checks = verify_bundle(bundle_dir, readobj)
        evidence["bundle_checks"] = bundle_checks

        header_path = bundle_dir / bundle_checks["header_file"]
        object_path = bundle_dir / bundle_checks["object_file"]
        harness_path = artifact_dir / "rvv_generated_bundle_abi_harness.c"
        harness_path.write_text(
            harness_source(bundle_checks["header_file"], runtime_counts),
            encoding="utf-8",
        )
        evidence["harness"] = {
            "path": str(harness_path),
            "sha256": sha256_file(harness_path),
            "boundary": "external C ABI consumer of generated header and object only",
        }

        if args.dry_run:
            evidence["status"] = "dry_run_success"
            evidence["ssh_evidence"] = False
        else:
            remote = run_remote_evidence(
                artifact_dir=artifact_dir,
                run_id=run_id,
                ssh_target=args.ssh_target,
                connect_timeout=args.connect_timeout,
                timeout=args.timeout,
                object_path=object_path,
                header_path=header_path,
                harness_path=harness_path,
            )
            evidence["remote"] = remote
            evidence["ssh_evidence"] = True
            evidence["status"] = "success"
        evidence["completed_at"] = utc_timestamp()
        write_json(artifact_dir / "evidence.json", evidence)
        print(f"{SCRIPT_NAME}: {evidence['status']}")
        print(f"artifact_dir: {artifact_dir}")
        if evidence.get("ssh_evidence"):
            print(str(evidence["remote"]["remote_output"]).strip())
        return 0
    except Exception as exc:  # noqa: BLE001 - evidence should record exact blocker.
        evidence["status"] = "blocked" if not args.dry_run else "failed"
        evidence["completed_at"] = utc_timestamp()
        evidence["diagnostic"] = sanitize_text(exc)
        write_json(artifact_dir / "evidence.json", evidence)
        print(f"{SCRIPT_NAME}: {evidence['status']}", file=sys.stderr)
        print(evidence["diagnostic"], file=sys.stderr)
        print(f"artifact_dir: {artifact_dir}", file=sys.stderr)
        return 1


def make_fake_bundle(root: Path) -> Path:
    bundle_dir = root / "bundle"
    bundle_dir.mkdir(parents=True)
    object_name = (
        "artifact-0-riscv-elf-relocatable-object-"
        "rvv-i32m1-arithmetic-emitc-route-family.o"
    )
    header_name = (
        "artifact-1-runtime-callable-c-header-"
        "rvv-i32m1-arithmetic-emitc-route-family.header.h"
    )
    (bundle_dir / object_name).write_bytes(b"\x7fELFfake-riscv-object")
    (bundle_dir / header_name).write_text(
        f"""
#ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
#define TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
#include <stddef.h>
#include <stdint.h>
/* tianchenrv.rvv.runtime_avl_source: runtime_abi:n */
/* tianchenrv.rvv.multi_vl: supported */
{EXPECTED_PROTOTYPE}
#endif
""".lstrip(),
        encoding="utf-8",
    )
    metadata_lines = "\n".join(
        f"""  artifact_metadata[{index}]:
    key: "{key}"
    value: "{value}" """
        for index, (key, value) in enumerate(EXPECTED_METADATA.items())
    )
    parameter_lines = "\n".join(
        f"""  runtime_abi_parameter[{index}]:
    c_name: "{parameter['c_name']}"
    c_type: "{parameter['c_type']}"
    role: "{parameter['role']}"
    ownership: "{parameter['ownership']}" """
        for index, parameter in enumerate(EXPECTED_RUNTIME_PARAMETERS)
    )
    index_text = f"""
tianchenrv.target_artifact_bundle.version: 1
bundle_status: "complete"
artifact_count: 2
artifact[0]:
  file_name: "{object_name}"
  component_group: "{EXPECTED_COMPONENT_GROUP}"
  component_role: "object"
  external_abi_name: "{EXPECTED_EXTERNAL_ABI_NAME}"
  selected_variant: @{EXPECTED_SELECTED_VARIANT}
  role: "{EXPECTED_SELECTED_ROLE}"
  component[0]:
    selected_variant: @{EXPECTED_SELECTED_VARIANT}
    role: "{EXPECTED_SELECTED_ROLE}"
  artifact_kind: "{EXPECTED_OBJECT_KIND}"
  route: "{EXPECTED_OBJECT_ROUTE}"
  owner: "{EXPECTED_OWNER}"
  runtime_abi: "{EXPECTED_EXTERNAL_ABI_NAME}"
  runtime_abi_kind: "{EXPECTED_RUNTIME_ABI_KIND}"
  runtime_abi_name: "{EXPECTED_EXTERNAL_ABI_NAME}"
  runtime_abi_parameter_count: 4
{parameter_lines}
{metadata_lines}
  handoff_kind: "materialized-emitc-cpp-rvv-intrinsic-object"
  evidence_role: "relocatable-object"
artifact[1]:
  file_name: "{header_name}"
  component_group: "{EXPECTED_COMPONENT_GROUP}"
  component_role: "header"
  external_abi_name: "{EXPECTED_EXTERNAL_ABI_NAME}"
  selected_variant: @{EXPECTED_SELECTED_VARIANT}
  role: "{EXPECTED_SELECTED_ROLE}"
  component[0]:
    selected_variant: @{EXPECTED_SELECTED_VARIANT}
    role: "{EXPECTED_SELECTED_ROLE}"
  artifact_kind: "{EXPECTED_HEADER_KIND}"
  route: "{EXPECTED_HEADER_ROUTE}"
  owner: "{EXPECTED_OWNER}"
  runtime_abi: "{EXPECTED_EXTERNAL_ABI_NAME}"
  runtime_abi_kind: "{EXPECTED_RUNTIME_ABI_KIND}"
  runtime_abi_name: "{EXPECTED_EXTERNAL_ABI_NAME}"
  runtime_abi_parameter_count: 4
{parameter_lines}
{metadata_lines}
  handoff_kind: "materialized-emitc-cpp-rvv-intrinsic-object"
  evidence_role: "header-declaration"
""".lstrip()
    (bundle_dir / INDEX_FILE_NAME).write_text(index_text, encoding="utf-8")
    return bundle_dir


def expect_self_test_failure(name: str, fn: Any) -> None:
    try:
        fn()
    except EvidenceError:
        return
    raise AssertionError(f"self-test negative case did not fail: {name}")


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="tcrv-rvv-generated-bundle-self-test-") as tmp_raw:
        tmp = Path(tmp_raw)
        bundle = make_fake_bundle(tmp)
        verify_bundle(bundle, readobj=None)
        harness = harness_source(
            "artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h",
            [1, 17, 257],
        )
        if EXPECTED_FUNCTION not in harness or "tcrv_rvv_generated_bundle_abi_ok" not in harness:
            raise AssertionError("self-test harness generation lost expected ABI call")

        missing_header = make_fake_bundle(tmp / "missing-header")
        header = next(missing_header.glob("*.h"))
        header.unlink()
        expect_self_test_failure("missing header", lambda: verify_bundle(missing_header, None))

        missing_object = make_fake_bundle(tmp / "missing-object")
        obj = next(missing_object.glob("*.o"))
        obj.unlink()
        expect_self_test_failure("missing object", lambda: verify_bundle(missing_object, None))

        bad_order = make_fake_bundle(tmp / "bad-order")
        index_path = bad_order / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace('role: "lhs-input-buffer"', 'role: "rhs-input-buffer"', 1)
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure("stale ABI order", lambda: verify_bundle(bad_order, None))

        missing_metadata = make_fake_bundle(tmp / "missing-metadata")
        index_path = missing_metadata / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace('value: "supported"', 'value: "missing"', 1)
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure("missing multi-VL metadata", lambda: verify_bundle(missing_metadata, None))

        bad_header = make_fake_bundle(tmp / "bad-header")
        header = next(bad_header.glob("*.h"))
        header.write_text(header.read_text(encoding="utf-8") + "\n__riscv_vadd_vv_i32m1\n", encoding="utf-8")
        expect_self_test_failure("header intrinsic body residue", lambda: verify_bundle(bad_header, None))

        mismatched_variant = make_fake_bundle(tmp / "mismatched-variant")
        index_path = mismatched_variant / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace(f"@{EXPECTED_SELECTED_VARIANT}", "@stale_variant", 1)
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure("mismatched selected variant", lambda: verify_bundle(mismatched_variant, None))

    print(f"{SCRIPT_NAME} self-test passed")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run local parser/verifier self-tests")
    parser.add_argument("--dry-run", action="store_true", help="generate and verify local bundle without ssh rvv")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tcrv-opt", default="build/bin/tcrv-opt")
    parser.add_argument("--tcrv-translate", default="build/bin/tcrv-translate")
    parser.add_argument("--llvm-readobj", default=default_readobj())
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--connect-timeout", type=int, default=DEFAULT_CONNECT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--runtime-count",
        action="append",
        type=int,
        default=[],
        help="runtime n value to test; may be repeated",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    if any(count < 0 for count in args.runtime_count):
        print("--runtime-count values must be non-negative", file=sys.stderr)
        return 2
    return run_e2e(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
