#!/usr/bin/env python3
"""Collect bounded RVV hardware/toolchain evidence from ``ssh rvv``.

The probe is evidence tooling only. It does not implement TianChen-RV compiler
IR, plugin logic, lowering, emission, runtime glue, correctness, or
performance measurement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any


PROBE_NAME = "tianchenrv-rvv-remote-probe"
SCHEMA_VERSION = 3
DEFAULT_ARTIFACT_ROOT = Path("artifacts/tmp/rvv_probe")
DEFAULT_SSH_TARGET = "rvv"
DEFAULT_TIMEOUT_SECONDS = 30

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

RVV_PROBE_SOURCE = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <riscv_vector.h>

static size_t tcrv_read_vlenb(void) {
  size_t value = 0;
  __asm__ volatile("csrr %0, vlenb" : "=r"(value));
  return value;
}

int main(void) {
  enum { kElements = 16 };
  int32_t lhs[kElements];
  int32_t rhs[kElements];
  int32_t out[kElements];

  for (int index = 0; index < kElements; ++index) {
    lhs[index] = index + 1;
    rhs[index] = 100 - index;
    out[index] = 0;
  }

  size_t first_vl = __riscv_vsetvl_e32m1(kElements);
  if (first_vl == 0 || first_vl > kElements) {
    fprintf(stderr, "invalid first vl=%zu\n", first_vl);
    return 2;
  }
  size_t vlenb = tcrv_read_vlenb();
  size_t i32_m1_lanes = vlenb / sizeof(int32_t);
  if (vlenb < sizeof(int32_t) || i32_m1_lanes == 0 ||
      i32_m1_lanes * sizeof(int32_t) != vlenb) {
    fprintf(stderr, "invalid vlenb=%zu i32_m1_lanes=%zu\n", vlenb,
            i32_m1_lanes);
    return 4;
  }

  size_t offset = 0;
  while (offset < kElements) {
    size_t vl = __riscv_vsetvl_e32m1(kElements - offset);
    vint32m1_t lhs_vec = __riscv_vle32_v_i32m1(&lhs[offset], vl);
    vint32m1_t rhs_vec = __riscv_vle32_v_i32m1(&rhs[offset], vl);
    vint32m1_t sum_vec = __riscv_vadd_vv_i32m1(lhs_vec, rhs_vec, vl);
    __riscv_vse32_v_i32m1(&out[offset], sum_vec, vl);
    offset += vl;
  }

  for (int index = 0; index < kElements; ++index) {
    int32_t expected = lhs[index] + rhs[index];
    if (out[index] != expected) {
      fprintf(stderr, "mismatch at %d: got %d expected %d\n", index,
              out[index], expected);
      return 3;
    }
  }

  printf("rvv_probe_ok first_vl=%zu vlenb=%zu i32_m1_lanes=%zu elements=%d\n",
         first_vl, vlenb, i32_m1_lanes, kElements);
  return 0;
}
""".lstrip()


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_run_id(raw_run_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_run_id.strip())
    return sanitized or utc_run_id()


def allocate_artifact_dir(artifact_root: Path, base_run_id: str) -> tuple[str, Path]:
    for suffix in range(100):
        run_id = base_run_id if suffix == 0 else f"{base_run_id}-{suffix}"
        artifact_dir = artifact_root / run_id
        if not artifact_dir.exists():
            return run_id, artifact_dir
    raise RuntimeError(f"could not allocate unique artifact directory for {base_run_id}")


def sanitize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    sanitized = str(text).replace("\x00", "\\0")
    for pattern, replacement in SECRET_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def command_display(command: list[str]) -> str:
    return sanitize_text(shlex.join(command))


def parse_first_int(text: str) -> int | None:
    match = re.search(r"\b([0-9]+)\b", text)
    if not match:
        return None
    return int(match.group(1))


def first_non_empty_line(text: str) -> str:
    for line in sanitize_text(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in sanitize_text(text).splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_named_int(text: str, name: str) -> int | None:
    match = re.search(rf"(?:^|[ \t]){re.escape(name)}=([0-9]+)(?:[ \t]|$)", text)
    if not match:
        return None
    return int(match.group(1))


def bounded_lines(text: str, limit: int = 160) -> list[str]:
    lines = [line.rstrip() for line in sanitize_text(text).splitlines()]
    return [line for line in lines if line.strip()][:limit]


def bounded_fact_value(text: Any, limit: int = 512) -> str:
    value = " ; ".join(sanitize_text(text).splitlines()).strip()
    return value[:limit]


def extract_isa_vector_hints(facts: dict[str, Any]) -> str:
    lines = facts.get("cpuinfo_hints", {}).get("lines", [])
    if not isinstance(lines, list):
        return ""
    relevant: list[str] = []
    for line in lines:
        sanitized = bounded_fact_value(line)
        lowered = sanitized.lower()
        if (
            "isa" in lowered
            or "rv64" in lowered
            or "rv32" in lowered
            or "zve" in lowered
            or "zvl" in lowered
            or "zvfh" in lowered
            or "vector" in lowered
        ):
            relevant.append(sanitized)
    return bounded_fact_value("; ".join(relevant[:8]))


def extract_selected_flag(flags: Any, prefix: str) -> str:
    if not isinstance(flags, list):
        return ""
    for flag in flags:
        text = str(flag)
        if text.startswith(prefix):
            return bounded_fact_value(text[len(prefix) :])
    return ""


def build_capability_facts(
    facts: dict[str, Any], compile_run: dict[str, Any]
) -> dict[str, Any]:
    hart_count_value = facts.get("hart_count", {}).get("value") or 0
    if not isinstance(hart_count_value, int):
        hart_count_value = parse_first_int(str(hart_count_value)) or 0
    diagnostic = str(compile_run.get("diagnostic", ""))
    vlenb_bytes = parse_named_int(diagnostic, "vlenb") or 0
    i32_m1_lane_count = parse_named_int(diagnostic, "i32_m1_lanes") or 0
    return {
        "architecture": bounded_fact_value(facts.get("architecture", {}).get("value", "")),
        "hart_count": hart_count_value,
        "vlenb_bytes": vlenb_bytes,
        "i32_m1_lane_count": i32_m1_lane_count,
        "first_slice_sew_bits": 32
        if compile_run.get("status") == "success"
        else 0,
        "first_slice_lmul": "m1"
        if compile_run.get("status") == "success"
        else "",
        "first_slice_tail_policy": "agnostic"
        if compile_run.get("status") == "success"
        else "",
        "first_slice_mask_policy": "agnostic"
        if compile_run.get("status") == "success"
        else "",
        "i64_m1_sew_bits": 64
        if compile_run.get("status") == "success"
        else 0,
        "i64_m1_lmul": "m1"
        if compile_run.get("status") == "success"
        else "",
        "i64_m1_tail_policy": "agnostic"
        if compile_run.get("status") == "success"
        else "",
        "i64_m1_mask_policy": "agnostic"
        if compile_run.get("status") == "success"
        else "",
        "isa_vector_hints": extract_isa_vector_hints(facts),
        "clang_available": bool(facts.get("clang", {}).get("available")),
        "clang_version": bounded_fact_value(facts.get("clang", {}).get("version", "")),
        "cmake_available": bool(facts.get("cmake", {}).get("available")),
        "cmake_version": bounded_fact_value(facts.get("cmake", {}).get("version", "")),
        "minimal_rvv_compile_run_succeeded": compile_run.get("status") == "success",
        "selected_march": extract_selected_flag(
            compile_run.get("selected_compile_flags"), "-march="
        ),
        "selected_mabi": extract_selected_flag(
            compile_run.get("selected_compile_flags"), "-mabi="
        ),
        "source_sha256": bounded_fact_value(compile_run.get("source_sha256", "")),
        "binary_sha256": bounded_fact_value(compile_run.get("binary_sha256", "")),
    }


def run_command(command: list[str], timeout_seconds: int) -> dict[str, Any]:
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
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
    return {
        "command": command_display(command),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": round(duration, 3),
        "stdout": sanitize_text(stdout),
        "stderr": sanitize_text(stderr),
    }


def write_command_log(log_dir: Path, index: int, name: str, result: dict[str, Any]) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "command"
    log_path = log_dir / f"{index:03d}_{safe_name}.log"
    log_path.write_text(
        "\n".join(
            [
                f"name: {name}",
                f"command: {result['command']}",
                f"exit_code: {result['exit_code']}",
                f"timed_out: {str(result['timed_out']).lower()}",
                f"duration_seconds: {result['duration_seconds']}",
                "--- stdout ---",
                result["stdout"],
                "--- stderr ---",
                result["stderr"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return log_path


class RemoteProbe:
    def __init__(
        self,
        ssh_target: str,
        artifact_dir: Path,
        timeout_seconds: int,
        ssh_options: list[str],
    ) -> None:
        self.ssh_target = ssh_target
        self.artifact_dir = artifact_dir
        self.log_dir = artifact_dir / "logs"
        self.timeout_seconds = timeout_seconds
        self.ssh_options = ssh_options
        self.commands: list[dict[str, Any]] = []

    def run_remote(
        self, name: str, remote_command: str, timeout_seconds: int | None = None
    ) -> dict[str, Any]:
        remote_invocation = shlex.join(["sh", "-lc", remote_command])
        command = [
            "ssh",
            *self.ssh_options,
            self.ssh_target,
            remote_invocation,
        ]
        result = run_command(command, timeout_seconds or self.timeout_seconds)
        result["name"] = name
        log_path = write_command_log(self.log_dir, len(self.commands), name, result)
        result["log_path"] = str(log_path.relative_to(self.artifact_dir))
        self.commands.append(result)
        return result


def collect_remote_facts(probe: RemoteProbe) -> dict[str, Any]:
    facts: dict[str, Any] = {}

    uname = probe.run_remote("uname_kernel", "uname -a")
    facts["uname"] = {
        "available": uname["exit_code"] == 0,
        "kernel": first_non_empty_line(uname["stdout"]),
        "command": "uname_kernel",
    }

    architecture = probe.run_remote("architecture", "uname -m")
    facts["architecture"] = {
        "available": architecture["exit_code"] == 0,
        "value": first_non_empty_line(architecture["stdout"]),
        "command": "architecture",
    }

    harts = probe.run_remote(
        "hart_count", "getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || true"
    )
    facts["hart_count"] = {
        "available": harts["exit_code"] == 0 and parse_first_int(harts["stdout"]) is not None,
        "value": parse_first_int(harts["stdout"]),
        "command": "hart_count",
    }

    clang_path = probe.run_remote("clang_path", "command -v clang || true")
    clang_version = probe.run_remote("clang_version", "clang --version")
    facts["clang"] = {
        "available": bool(first_non_empty_line(clang_path["stdout"]))
        and clang_version["exit_code"] == 0,
        "path": first_non_empty_line(clang_path["stdout"]),
        "version": first_non_empty_line(clang_version["stdout"]),
        "path_command": "clang_path",
        "version_command": "clang_version",
    }

    cmake_path = probe.run_remote("cmake_path", "command -v cmake || true")
    cmake_version = probe.run_remote("cmake_version", "cmake --version")
    facts["cmake"] = {
        "available": bool(first_non_empty_line(cmake_path["stdout"]))
        and cmake_version["exit_code"] == 0,
        "path": first_non_empty_line(cmake_path["stdout"]),
        "version": first_non_empty_line(cmake_version["stdout"]),
        "path_command": "cmake_path",
        "version_command": "cmake_version",
    }

    cpuinfo_command = (
        "if [ -r /proc/cpuinfo ]; then "
        "grep -E -i '^(processor|hart|isa|mmu|uarch|model name|cpu|riscv|mvendorid|marchid|mimpid)[[:space:]]*:' "
        "/proc/cpuinfo | head -n 160; "
        "else echo '/proc/cpuinfo unavailable'; fi"
    )
    cpuinfo = probe.run_remote("cpuinfo_riscv_vector_hints", cpuinfo_command)
    facts["cpuinfo_hints"] = {
        "available": cpuinfo["exit_code"] == 0,
        "lines": bounded_lines(cpuinfo["stdout"]),
        "command": "cpuinfo_riscv_vector_hints",
    }

    sudo_command = (
        "if command -v sudo >/dev/null 2>&1; then "
        "sudo -n true; code=$?; "
        "echo sudo_present=true; "
        "if [ \"$code\" -eq 0 ]; then echo sudo_non_interactive=true; "
        "else echo sudo_non_interactive=false; fi; "
        "echo sudo_exit_code=$code; "
        "else echo sudo_present=false; echo sudo_non_interactive=false; echo sudo_exit_code=127; fi"
    )
    sudo = probe.run_remote("sudo_non_interactive_capability", sudo_command)
    sudo_values = parse_key_values(sudo["stdout"])
    facts["sudo"] = {
        "present": sudo_values.get("sudo_present") == "true",
        "non_interactive": sudo_values.get("sudo_non_interactive") == "true",
        "command": "sudo_non_interactive_capability",
    }

    return facts


def compile_flag_candidates(architecture: str) -> list[list[str]]:
    candidates = [["-O2", "-march=native"]]
    if "riscv64" in architecture:
        candidates.extend(
            [
                ["-O2", "-march=rv64gcv", "-mabi=lp64d"],
                ["-O2", "-march=rv64gcv_zvl128b", "-mabi=lp64d"],
            ]
        )
    elif "riscv32" in architecture:
        candidates.extend(
            [
                ["-O2", "-march=rv32gcv", "-mabi=ilp32d"],
                ["-O2", "-march=rv32gcv_zvl128b", "-mabi=ilp32d"],
            ]
        )
    candidates.append(["-O2"])
    return candidates


def quote_remote_path(path: str) -> str:
    return shlex.quote(path)


def run_rvv_compile_probe(
    probe: RemoteProbe, run_id: str, facts: dict[str, Any]
) -> dict[str, Any]:
    source_sha256 = sha256_text(RVV_PROBE_SOURCE)
    compile_run: dict[str, Any] = {
        "attempted": False,
        "status": "skipped",
        "diagnostic": "",
        "compiler_path": facts.get("clang", {}).get("path", ""),
        "compiler_version": facts.get("clang", {}).get("version", ""),
        "source_sha256": source_sha256,
        "binary_sha256": "",
        "selected_compile_flags": [],
        "commands": [],
    }

    if not facts.get("clang", {}).get("available"):
        compile_run["diagnostic"] = "clang unavailable; see clang_path and clang_version command logs"
        compile_run["commands"] = ["clang_path", "clang_version"]
        return compile_run

    remote_dir = f"/tmp/tianchenrv_rvv_probe_{safe_run_id(run_id)}"
    remote_source = f"{remote_dir}/rvv_probe.c"
    remote_binary = f"{remote_dir}/rvv_probe"
    setup_command = (
        f"rm -rf {quote_remote_path(remote_dir)} && "
        f"mkdir -p {quote_remote_path(remote_dir)} && "
        f"cat > {quote_remote_path(remote_source)} <<'TCRV_RVV_PROBE_SOURCE'\n"
        f"{RVV_PROBE_SOURCE}"
        "TCRV_RVV_PROBE_SOURCE\n"
        "if command -v sha256sum >/dev/null 2>&1; then "
        f"sha256sum {quote_remote_path(remote_source)}; "
        "else echo 'sha256sum unavailable'; fi"
    )
    setup = probe.run_remote("rvv_compile_setup_source", setup_command)
    compile_run["commands"].append("rvv_compile_setup_source")
    compile_run["attempted"] = True
    if setup["exit_code"] != 0:
        compile_run["status"] = "failure"
        compile_run["diagnostic"] = "failed to stage RVV probe source on remote target"
        return compile_run

    architecture = facts.get("architecture", {}).get("value") or ""
    for index, flags in enumerate(compile_flag_candidates(str(architecture))):
        flag_text = " ".join(shlex.quote(flag) for flag in flags)
        compile_command = (
            f"cd {quote_remote_path(remote_dir)} && "
            f"clang {flag_text} {quote_remote_path(remote_source)} -o {quote_remote_path(remote_binary)}"
        )
        command_name = f"rvv_compile_attempt_{index}"
        compile_result = probe.run_remote(command_name, compile_command, timeout_seconds=60)
        compile_run["commands"].append(command_name)
        if compile_result["exit_code"] == 0:
            compile_run["selected_compile_flags"] = flags
            break
    else:
        compile_run["status"] = "failure"
        compile_run["diagnostic"] = (
            "RVV compile failed for all candidate flag sets; see rvv_compile_attempt_* logs"
        )
        probe.run_remote("rvv_compile_cleanup", f"rm -rf {quote_remote_path(remote_dir)}")
        compile_run["commands"].append("rvv_compile_cleanup")
        return compile_run

    binary_hash = probe.run_remote(
        "rvv_binary_sha256",
        "if command -v sha256sum >/dev/null 2>&1; then "
        f"set -- $(sha256sum {quote_remote_path(remote_binary)}); "
        "printf '%s\\n' \"$1\"; "
        "else echo ''; fi",
    )
    compile_run["commands"].append("rvv_binary_sha256")
    if binary_hash["exit_code"] == 0:
        compile_run["binary_sha256"] = first_non_empty_line(binary_hash["stdout"])

    run_result = probe.run_remote(
        "rvv_probe_program_run", quote_remote_path(remote_binary), timeout_seconds=60
    )
    compile_run["commands"].append("rvv_probe_program_run")
    cleanup = probe.run_remote("rvv_compile_cleanup", f"rm -rf {quote_remote_path(remote_dir)}")
    compile_run["commands"].append("rvv_compile_cleanup")
    if cleanup["exit_code"] != 0 and not compile_run["diagnostic"]:
        compile_run["diagnostic"] = "remote cleanup failed; see rvv_compile_cleanup log"

    if run_result["exit_code"] == 0:
        compile_run["status"] = "success"
        compile_run["diagnostic"] = first_non_empty_line(run_result["stdout"])
    else:
        compile_run["status"] = "failure"
        compile_run["diagnostic"] = "RVV probe binary failed at runtime; see rvv_probe_program_run log"
    return compile_run


def validate_evidence_artifact(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_top_level = {
        "schema_version": int,
        "probe_name": str,
        "run_id": str,
        "timestamp_utc": str,
        "ssh_target": str,
        "artifact_dir": str,
        "status": str,
        "success": bool,
        "facts": dict,
        "capability_facts": dict,
        "rvv_compile_run": dict,
        "commands": list,
    }
    for key, expected_type in required_top_level.items():
        if key not in artifact:
            errors.append(f"missing top-level key: {key}")
            continue
        if not isinstance(artifact[key], expected_type):
            errors.append(f"top-level key {key} has wrong type")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema version")
    if artifact.get("probe_name") != PROBE_NAME:
        errors.append("unexpected probe name")
    if artifact.get("status") not in ("success", "failure"):
        errors.append("status must be success or failure")
    facts = artifact.get("facts") if isinstance(artifact.get("facts"), dict) else {}
    for key in ("uname", "architecture", "hart_count", "clang", "cmake", "cpuinfo_hints", "sudo"):
        if key not in facts:
            errors.append(f"missing fact section: {key}")
    compile_run = (
        artifact.get("rvv_compile_run")
        if isinstance(artifact.get("rvv_compile_run"), dict)
        else {}
    )
    for key in (
        "attempted",
        "status",
        "diagnostic",
        "compiler_path",
        "compiler_version",
        "source_sha256",
        "binary_sha256",
        "selected_compile_flags",
        "commands",
    ):
        if key not in compile_run:
            errors.append(f"missing compile/run key: {key}")
    if compile_run.get("status") not in ("success", "failure", "skipped"):
        errors.append("compile/run status must be success, failure, or skipped")
    capability_facts = (
        artifact.get("capability_facts")
        if isinstance(artifact.get("capability_facts"), dict)
        else {}
    )
    required_capability_facts = {
        "architecture": str,
        "hart_count": int,
        "vlenb_bytes": int,
        "i32_m1_lane_count": int,
        "first_slice_sew_bits": int,
        "first_slice_lmul": str,
        "first_slice_tail_policy": str,
        "first_slice_mask_policy": str,
        "i64_m1_sew_bits": int,
        "i64_m1_lmul": str,
        "i64_m1_tail_policy": str,
        "i64_m1_mask_policy": str,
        "isa_vector_hints": str,
        "clang_available": bool,
        "clang_version": str,
        "cmake_available": bool,
        "cmake_version": str,
        "minimal_rvv_compile_run_succeeded": bool,
        "selected_march": str,
        "selected_mabi": str,
        "source_sha256": str,
        "binary_sha256": str,
    }
    for key, expected_type in required_capability_facts.items():
        if key not in capability_facts:
            errors.append(f"missing capability fact key: {key}")
            continue
        if not isinstance(capability_facts[key], expected_type):
            errors.append(f"capability fact {key} has wrong type")
    for index, command in enumerate(artifact.get("commands", [])):
        if not isinstance(command, dict):
            errors.append(f"command {index} is not an object")
            continue
        for key in ("name", "command", "exit_code", "stdout", "stderr", "log_path"):
            if key not in command:
                errors.append(f"command {index} missing key: {key}")
    return errors


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    base_run_id = safe_run_id(args.run_id or utc_run_id())
    artifact_root = Path(args.artifact_root)
    run_id, artifact_dir = allocate_artifact_dir(artifact_root, base_run_id)
    log_dir = artifact_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)

    ssh_options = [
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
    ]
    for option in args.ssh_option:
        ssh_options.extend(["-o", option])

    probe = RemoteProbe(args.ssh_target, artifact_dir, args.timeout, ssh_options)
    facts = collect_remote_facts(probe)
    compile_run = run_rvv_compile_probe(probe, run_id, facts)
    capability_facts = build_capability_facts(facts, compile_run)

    fact_failures = [
        name
        for name in ("uname", "architecture", "hart_count", "clang", "cmake", "cpuinfo_hints")
        if not facts.get(name, {}).get("available")
    ]
    success = not fact_failures and compile_run.get("status") == "success"
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "probe_name": PROBE_NAME,
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "ssh_target": args.ssh_target,
        "artifact_dir": str(artifact_dir),
        "status": "success" if success else "failure",
        "success": success,
        "diagnostic": (
            "probe completed successfully"
            if success
            else "probe did not prove full hardware/toolchain availability"
        ),
        "facts": facts,
        "capability_facts": capability_facts,
        "rvv_compile_run": compile_run,
        "commands": probe.commands,
    }
    schema_errors = validate_evidence_artifact(artifact)
    artifact["schema_errors"] = schema_errors
    if schema_errors:
        artifact["status"] = "failure"
        artifact["success"] = False
        artifact["diagnostic"] = "artifact schema validation failed"

    artifact_path = artifact_dir / "rvv_probe_evidence.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "status": artifact["status"],
                "success": artifact["success"],
                "compile_run_status": compile_run.get("status"),
                "diagnostic": artifact["diagnostic"],
            },
            sort_keys=True,
        )
    )
    return artifact


def assert_self_test(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_self_test() -> None:
    unsafe_text = (
        "Authorization: Bearer abc.def.ghi\n"
        "PASSWORD=hunter2\n"
        "token: live-token\n"
        "-----BEGIN OPENSSH PRIVATE KEY-----\nsecret\n"
        "-----END OPENSSH PRIVATE KEY-----\n"
        "visible=ok\n"
    )
    sanitized = sanitize_text(unsafe_text)
    assert_self_test("abc.def.ghi" not in sanitized, "bearer token was not redacted")
    assert_self_test("hunter2" not in sanitized, "password was not redacted")
    assert_self_test("live-token" not in sanitized, "token value was not redacted")
    assert_self_test("OPENSSH PRIVATE KEY" not in sanitized, "private key was not redacted")
    assert_self_test("visible=ok" in sanitized, "non-secret text should remain visible")

    assert_self_test(parse_first_int("64\n") == 64, "hart parser failed")
    assert_self_test(
        parse_named_int("rvv_probe_ok first_vl=4 vlenb=16 i32_m1_lanes=4", "vlenb")
        == 16,
        "named vlenb parser failed",
    )
    assert_self_test(
        parse_named_int("rvv_probe_ok first_vl=4 vlenb=16 i32_m1_lanes=4", "i32_m1_lanes")
        == 4,
        "named lane parser failed",
    )
    sudo_values = parse_key_values(
        "sudo_present=true\nsudo_non_interactive=false\nsudo_exit_code=1\n"
    )
    assert_self_test(sudo_values["sudo_present"] == "true", "sudo present parser failed")
    assert_self_test(
        sudo_values["sudo_non_interactive"] == "false",
        "sudo capability parser failed",
    )
    cpuinfo = bounded_lines(
        "processor\t: 0\nisa\t: rv64imafdcv\nmmu\t: sv39\nunrelated\t: hidden\n",
        limit=3,
    )
    assert_self_test(cpuinfo[:2] == ["processor\t: 0", "isa\t: rv64imafdcv"], "line parser failed")

    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir)
        result = {
            "command": sanitize_text("echo PASSWORD=hunter2"),
            "exit_code": 0,
            "timed_out": False,
            "duration_seconds": 0.001,
            "stdout": sanitized,
            "stderr": "",
        }
        log_path = write_command_log(log_dir, 0, "fixture command", result)
        log_text = log_path.read_text(encoding="utf-8")
        assert_self_test("hunter2" not in log_text, "command log leaked password")
        assert_self_test("abc.def.ghi" not in log_text, "command log leaked bearer token")

    synthetic_facts = {
        "uname": {},
        "architecture": {"available": True, "value": "riscv64"},
        "hart_count": {"available": True, "value": 64},
        "clang": {
            "available": True,
            "version": "clang version 18.1.3 PASSWORD=hunter2",
        },
        "cmake": {"available": True, "version": "cmake version 3.28.3"},
        "cpuinfo_hints": {
            "available": True,
            "lines": [
                "processor\t: 0",
                "isa\t: rv64imafdcv_zve32f_zvfh TOKEN=live-token",
                "unrelated\t: hidden",
            ],
        },
        "sudo": {},
    }
    synthetic_compile_run = {
        "attempted": True,
        "status": "success",
        "diagnostic": "rvv_probe_ok first_vl=4 vlenb=16 i32_m1_lanes=4",
        "compiler_path": "/usr/bin/clang",
        "compiler_version": "fixture clang",
        "source_sha256": sha256_text(RVV_PROBE_SOURCE),
        "binary_sha256": "a" * 64,
        "selected_compile_flags": ["-O2", "-march=rv64gcv", "-mabi=lp64d"],
        "commands": ["rvv_probe_program_run"],
    }
    capability_facts = build_capability_facts(
        synthetic_facts, synthetic_compile_run
    )
    assert_self_test(
        capability_facts["architecture"] == "riscv64",
        "capability facts architecture missing",
    )
    assert_self_test(
        capability_facts["hart_count"] == 64,
        "capability facts hart count missing",
    )
    assert_self_test(
        capability_facts["vlenb_bytes"] == 16,
        "capability facts vlenb missing",
    )
    assert_self_test(
        capability_facts["i32_m1_lane_count"] == 4,
        "capability facts i32 m1 lane count missing",
    )
    assert_self_test(
        capability_facts["first_slice_sew_bits"] == 32,
        "capability facts first-slice SEW missing",
    )
    assert_self_test(
        capability_facts["first_slice_lmul"] == "m1",
        "capability facts first-slice LMUL missing",
    )
    assert_self_test(
        capability_facts["first_slice_tail_policy"] == "agnostic",
        "capability facts first-slice tail policy missing",
    )
    assert_self_test(
        capability_facts["first_slice_mask_policy"] == "agnostic",
        "capability facts first-slice mask policy missing",
    )
    assert_self_test(
        capability_facts["i64_m1_sew_bits"] == 64,
        "capability facts i64m1 SEW missing",
    )
    assert_self_test(
        capability_facts["i64_m1_lmul"] == "m1",
        "capability facts i64m1 LMUL missing",
    )
    assert_self_test(
        capability_facts["i64_m1_tail_policy"] == "agnostic",
        "capability facts i64m1 tail policy missing",
    )
    assert_self_test(
        capability_facts["i64_m1_mask_policy"] == "agnostic",
        "capability facts i64m1 mask policy missing",
    )
    assert_self_test(
        "rv64imafdcv" in capability_facts["isa_vector_hints"],
        "capability facts vector hint missing",
    )
    assert_self_test(
        "hunter2" not in capability_facts["clang_version"],
        "capability facts leaked password",
    )
    assert_self_test(
        "live-token" not in capability_facts["isa_vector_hints"],
        "capability facts leaked token",
    )
    assert_self_test(
        capability_facts["selected_march"] == "rv64gcv",
        "capability facts selected march missing",
    )
    assert_self_test(
        capability_facts["selected_mabi"] == "lp64d",
        "capability facts selected mabi missing",
    )

    synthetic_artifact = {
        "schema_version": SCHEMA_VERSION,
        "probe_name": PROBE_NAME,
        "run_id": "self-test",
        "timestamp_utc": "2026-05-07T00:00:00Z",
        "ssh_target": "rvv",
        "artifact_dir": "artifacts/tmp/rvv_probe/self-test",
        "status": "success",
        "success": True,
        "facts": synthetic_facts,
        "capability_facts": capability_facts,
        "rvv_compile_run": synthetic_compile_run,
        "commands": [
            {
                "name": "fixture",
                "command": "true",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "log_path": "logs/000_fixture.log",
            }
        ],
    }
    errors = validate_evidence_artifact(synthetic_artifact)
    assert_self_test(not errors, "schema validator rejected fixture: " + "; ".join(errors))
    print("rvv_remote_probe self-test passed")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Additional ssh -o option, for example StrictHostKeyChecking=no",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run local schema/parser/sanitizer tests without ssh rvv",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    artifact = run_probe(args)
    return 0 if artifact.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
