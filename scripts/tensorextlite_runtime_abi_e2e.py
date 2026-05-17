#!/usr/bin/env python3
"""Prove TensorExtLite generated runtime ABI bundle consumption locally.

This is evidence tooling only. It invokes the one-command source artifact
bundle front door for the source-input bundle, uses an explicit materialized IR
fixture for lower-level EmitC/header/object exporter checks, builds a small
native ABI consumer from the generated declaration header and materialized
EmitC C++ source, and runs that consumer locally. It does not implement
compiler IR, lowering, plugin selection, emission, descriptors, fallback
computation, or runtime glue.
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
from typing import Any


SCRIPT_NAME = "tensorextlite_runtime_abi_e2e"
SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_ROOT = Path("artifacts/tmp/tensorextlite_runtime_abi_e2e")
DEFAULT_INPUT = Path(
    "test/Transforms/TensorExtLite/tensorext-lite-fragment-mma-source-front-door.mlir"
)
DEFAULT_MATERIALIZED_INPUT = Path(
    "test/Target/TensorExtLite/tensorext-lite-target-artifact-header.mlir"
)
DEFAULT_TIMEOUT_SECONDS = 60

INDEX_FILE_NAME = "tianchenrv-target-artifact-bundle.index"
EXPECTED_SELECTED_VARIANT = "tensorext_lite_tile_mma_first_slice"
EXPECTED_ORIGIN_PLUGIN = "tensorext-lite-plugin"
EXPECTED_CONSTRUCTION_PROTOCOL = "extension-family-construction-protocol.v1"
EXPECTED_EMITC_ROUTE = "tensorext-lite-fragment-mma-emitc-route"
EXPECTED_HEADER_ROUTE = "tensorext-lite-fragment-mma-emitc-route.header"
EXPECTED_RUNTIME_ABI = "tensorext-lite-fragment-mma-runtime-c-abi.v1"
EXPECTED_RUNTIME_ABI_KIND = "plugin-owned-runtime-abi"
EXPECTED_COMPONENT_GROUP = "tensorext-lite-fragment-mma-materialized-emitc-bundle.v1"
EXPECTED_OBJECT_KIND = "riscv-elf-relocatable-object"
EXPECTED_HEADER_KIND = "runtime-callable-c-header"
EXPECTED_FUNCTION = (
    "tcrv_emitc_tensorext_lite_header_export_"
    "tensorext_lite_tile_mma_first_slice"
)
EXPECTED_CALL_TRACE = "configure,load_frag,tile_mma,store_frag"
EXPECTED_BUNDLE_OBJECT = (
    "artifact-0-riscv-elf-relocatable-object-"
    "tensorext-lite-fragment-mma-emitc-route.o"
)
EXPECTED_BUNDLE_HEADER = (
    "artifact-1-runtime-callable-c-header-"
    "tensorext-lite-fragment-mma-emitc-route.header.h"
)

FORBIDDEN_HEADER_TOKENS = (
    "__riscv_",
    "descriptor",
    "direct-C",
    "direct_c",
    "source-export",
    "source_export",
    "int main",
    "return;",
    "tcrv_tensorext_lite_config(",
    "tcrv_tensorext_lite_load_frag(",
    "tcrv_tensorext_lite_tile_mma(",
    "tcrv_tensorext_lite_store_frag(",
)


class EvidenceError(RuntimeError):
    pass


def utc_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_run_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip()) or utc_run_id()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_tool(path_or_name: str, fallback_candidates: tuple[str, ...] = ()) -> str:
    candidate = Path(path_or_name)
    if candidate.exists():
        return str(candidate)
    resolved = shutil.which(path_or_name)
    if resolved:
        return resolved
    for fallback in fallback_candidates:
        fallback_path = Path(fallback)
        if fallback_path.exists():
            return str(fallback_path)
        resolved = shutil.which(fallback)
        if resolved:
            return resolved
    raise EvidenceError(f"required tool not found: {path_or_name}")


def command_display(command: list[str]) -> str:
    return shlex.join(command)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_contains(text: str, needle: str, context: str) -> None:
    if needle not in text:
        raise EvidenceError(f"{context} missing expected text: {needle}")


def require_not_contains(text: str, needle: str, context: str) -> None:
    if needle in text:
        raise EvidenceError(f"{context} contains forbidden text: {needle}")


def require_any_contains(text: str, needles: tuple[str, ...], context: str) -> None:
    if not any(needle in text for needle in needles):
        raise EvidenceError(
            f"{context} missing any expected text: {', '.join(needles)}"
        )


class EvidenceRun:
    def __init__(self, run_dir: Path, timeout: int):
        self.run_dir = run_dir
        self.timeout = timeout
        self.commands: list[dict[str, Any]] = []

    def run(
        self,
        name: str,
        command: list[str],
        *,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
            check=False,
        )
        if stdout_path is not None:
            stdout_path.write_bytes(result.stdout)
        if stderr_path is not None:
            stderr_path.write_bytes(result.stderr)
        self.commands.append(
            {
                "name": name,
                "command": command_display(command),
                "returncode": result.returncode,
                "stdout": str(stdout_path.relative_to(self.run_dir))
                if stdout_path is not None and stdout_path.is_relative_to(self.run_dir)
                else str(stdout_path) if stdout_path is not None else None,
                "stderr": str(stderr_path.relative_to(self.run_dir))
                if stderr_path is not None and stderr_path.is_relative_to(self.run_dir)
                else str(stderr_path) if stderr_path is not None else None,
            }
        )
        if check and result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise EvidenceError(
                f"{name} failed with exit {result.returncode}: {stderr[:1000]}"
            )
        return result


def write_harness_source(path: Path) -> None:
    path.write_text(
        f"""#include \"generated.h\"

#include <cstdio>
#include <cstring>

namespace {{
int call_index = 0;
bool bad_order = false;
char trace[5] = {{}};

void record_role(char marker, int expected_index) {{
  if (call_index != expected_index)
    bad_order = true;
  if (call_index >= 0 && call_index < 4)
    trace[call_index] = marker;
  ++call_index;
}}
}} // namespace

void tcrv_tensorext_lite_config() {{ record_role('C', 0); }}
void tcrv_tensorext_lite_load_frag() {{ record_role('L', 1); }}
void tcrv_tensorext_lite_tile_mma() {{ record_role('M', 2); }}
void tcrv_tensorext_lite_store_frag() {{ record_role('S', 3); }}

int main() {{
  {EXPECTED_FUNCTION}();
  trace[4] = '\\0';
  if (bad_order || call_index != 4 || std::strcmp(trace, \"CLMS\") != 0) {{
    std::printf(\"FAIL call_index=%d trace=%s bad_order=%d\\n\", call_index,
                trace, bad_order ? 1 : 0);
    return 1;
  }}

  std::printf(
      \"PASS tianchenrv.tensorext_lite.runtime_abi_e2e \"
      \"selected_variant={EXPECTED_SELECTED_VARIANT} \"
      \"origin_plugin={EXPECTED_ORIGIN_PLUGIN} \"
      \"construction_protocol={EXPECTED_CONSTRUCTION_PROTOCOL} \"
      \"emitc_route={EXPECTED_EMITC_ROUTE} \"
      \"runtime_abi_name={EXPECTED_RUNTIME_ABI} \"
      \"runtime_abi_parameter_count=0 \"
      \"component_group={EXPECTED_COMPONENT_GROUP} \"
      \"native_call_trace={EXPECTED_CALL_TRACE}\\n\");
  return 0;
}}
""",
        encoding="utf-8",
    )


def validate_generated_header(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    require_contains(text, f"void {EXPECTED_FUNCTION}(void);", "generated header")
    require_contains(text, EXPECTED_ORIGIN_PLUGIN, "generated header")
    require_contains(text, f"@{EXPECTED_SELECTED_VARIANT}", "generated header")
    require_contains(text, EXPECTED_EMITC_ROUTE, "generated header")
    require_contains(text, EXPECTED_RUNTIME_ABI, "generated header")
    require_contains(text, EXPECTED_CONSTRUCTION_PROTOCOL, "generated header")
    for token in FORBIDDEN_HEADER_TOKENS:
        require_not_contains(text, token, "generated header")


def validate_bundle_index(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for expected in (
        'bundle_status: "complete"',
        "artifact_count: 2",
        EXPECTED_BUNDLE_OBJECT,
        EXPECTED_BUNDLE_HEADER,
        EXPECTED_COMPONENT_GROUP,
        EXPECTED_OBJECT_KIND,
        EXPECTED_HEADER_KIND,
        EXPECTED_EMITC_ROUTE,
        EXPECTED_HEADER_ROUTE,
        EXPECTED_ORIGIN_PLUGIN,
        EXPECTED_RUNTIME_ABI,
        EXPECTED_RUNTIME_ABI_KIND,
        "runtime_abi_parameter_count: 0",
        EXPECTED_CONSTRUCTION_PROTOCOL,
    ):
        require_contains(text, expected, "bundle index")
    for token in ("descriptor", "direct-C", "source-export", "int main", "__riscv_"):
        require_not_contains(text, token, "bundle index")


def validate_generated_source(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for expected in (
        f'extern "C" void {EXPECTED_FUNCTION}()',
        "tcrv_tensorext_lite_config();",
        "tcrv_tensorext_lite_load_frag();",
        "tcrv_tensorext_lite_tile_mma();",
        "tcrv_tensorext_lite_store_frag();",
        "role=configure",
        "role=load_frag",
        "role=tile_mma",
        "role=store_frag",
        "op_interface=TCRVEmitCLowerableOpInterface",
    ):
        require_contains(text, expected, "generated C++ source")
    for token in ("descriptor", "direct-C", "source-export", "__riscv_", "int main"):
        require_not_contains(text, token, "generated C++ source")


def copy_header_if_equivalent(source: Path, destination: Path) -> None:
    if source.read_bytes() != destination.read_bytes():
        raise EvidenceError("standalone generated header and bundle header differ")


def expect_failure(
    run: EvidenceRun,
    name: str,
    command: list[str],
    *,
    stderr_path: Path,
    expected: tuple[str, ...],
) -> None:
    result = run.run(name, command, stderr_path=stderr_path, check=False)
    if result.returncode == 0:
        raise EvidenceError(f"{name} unexpectedly succeeded")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    require_any_contains(stderr, expected, name)


def create_evidence(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = Path(args.artifact_root)
    run_id = safe_run_id(args.run_id or utc_run_id())
    run_dir = artifact_root / run_id
    if run_dir.exists():
        raise EvidenceError(f"evidence directory already exists: {run_dir}")
    bundle_dir = run_dir / "bundle"
    negative_dir = run_dir / "negative"
    bundle_dir.mkdir(parents=True)
    negative_dir.mkdir()

    tcrv_translate = ensure_tool(args.tcrv_translate, ("build/bin/tcrv-translate",))
    clangxx = ensure_tool(args.clangxx, ("/usr/lib/llvm-20/bin/clang++",))
    readobj = ensure_tool(args.llvm_readobj, ("/usr/lib/llvm-20/bin/llvm-readobj",))

    input_path = Path(args.input)
    if not input_path.exists():
        raise EvidenceError(f"input MLIR does not exist: {input_path}")
    materialized_input_path = Path(args.materialized_input)
    if not materialized_input_path.exists():
        raise EvidenceError(f"materialized input MLIR does not exist: {materialized_input_path}")

    run = EvidenceRun(run_dir, args.timeout)
    input_copy = run_dir / "source_front_door_input.mlir"
    input_copy.write_bytes(input_path.read_bytes())
    materialized_input_copy = run_dir / "materialized_input.mlir"
    materialized_input_copy.write_bytes(materialized_input_path.read_bytes())

    generated_cpp = run_dir / "generated.cpp"
    generated_header = run_dir / "generated.h"
    target_object = run_dir / "target_object.o"
    bundle_stdout = run_dir / "bundle_stdout.txt"
    bundle_stderr = run_dir / "bundle_stderr.txt"
    target_object_stderr = run_dir / "target_object.stderr.txt"
    header_stderr = run_dir / "header.stderr.txt"
    cpp_stderr = run_dir / "generated_cpp.stderr.txt"

    run.run(
        "export source artifact bundle front door",
        [
            tcrv_translate,
            "--tcrv-source-artifact-bundle-front-door",
            f"--tcrv-target-artifact-bundle-output-dir={bundle_dir}",
            str(input_path),
        ],
        stdout_path=bundle_stdout,
        stderr_path=bundle_stderr,
    )
    run.run(
        "export materialized EmitC C++ source",
        [tcrv_translate, "--tcrv-tensorext-lite-emitc-to-cpp", str(materialized_input_path)],
        stdout_path=generated_cpp,
        stderr_path=cpp_stderr,
    )
    run.run(
        "export declaration header",
        [
            tcrv_translate,
            "--tcrv-export-target-header-artifact",
            str(materialized_input_path),
        ],
        stdout_path=generated_header,
        stderr_path=header_stderr,
    )
    run.run(
        "export standalone target object",
        [tcrv_translate, "--tcrv-export-target-artifact", str(materialized_input_path)],
        stdout_path=target_object,
        stderr_path=target_object_stderr,
    )

    bundle_object = bundle_dir / EXPECTED_BUNDLE_OBJECT
    bundle_header = bundle_dir / EXPECTED_BUNDLE_HEADER
    bundle_index = bundle_dir / INDEX_FILE_NAME
    for path in (bundle_object, bundle_header, bundle_index):
        if not path.exists():
            raise EvidenceError(f"bundle output missing: {path.name}")

    validate_generated_source(generated_cpp)
    validate_generated_header(generated_header)
    validate_generated_header(bundle_header)
    copy_header_if_equivalent(generated_header, bundle_header)
    validate_bundle_index(bundle_index)

    readobj_header = run_dir / "target_object.readobj_header.txt"
    readobj_symbols = run_dir / "target_object.readobj_symbols.txt"
    run.run(
        "inspect target bundle object header",
        [readobj, "-h", str(bundle_object)],
        stdout_path=readobj_header,
        stderr_path=run_dir / "target_object.readobj_header.stderr.txt",
    )
    run.run(
        "inspect target bundle object symbols",
        [readobj, "--symbols", str(bundle_object)],
        stdout_path=readobj_symbols,
        stderr_path=run_dir / "target_object.readobj_symbols.stderr.txt",
    )
    readobj_header_text = readobj_header.read_text(encoding="utf-8")
    readobj_symbols_text = readobj_symbols.read_text(encoding="utf-8")
    require_contains(readobj_header_text, "Format: elf64-littleriscv", "target object")
    require_contains(readobj_header_text, "Arch: riscv64", "target object")
    require_contains(readobj_symbols_text, EXPECTED_FUNCTION, "target object symbols")

    harness_source = run_dir / "harness.cpp"
    native_object = run_dir / "generated.native.o"
    harness_binary = run_dir / "harness"
    harness_stdout = run_dir / "run.stdout.txt"
    harness_stderr = run_dir / "run.stderr.txt"
    write_harness_source(harness_source)
    run.run(
        "compile native proof object from generated C++ source",
        [
            clangxx,
            "-std=c++17",
            "-O0",
            "-g",
            "-c",
            str(generated_cpp),
            "-o",
            str(native_object),
        ],
        stderr_path=run_dir / "native_object.stderr.txt",
    )
    run.run(
        "compile and link local runtime ABI harness",
        [
            clangxx,
            "-std=c++17",
            "-O0",
            "-g",
            str(native_object),
            str(harness_source),
            f"-I{run_dir}",
            "-o",
            str(harness_binary),
        ],
        stderr_path=run_dir / "harness_link.stderr.txt",
    )
    run.run(
        "run local runtime ABI harness",
        [str(harness_binary)],
        stdout_path=harness_stdout,
        stderr_path=harness_stderr,
    )
    harness_output = harness_stdout.read_text(encoding="utf-8")
    require_contains(harness_output, "PASS tianchenrv.tensorext_lite.runtime_abi_e2e", "harness")
    require_contains(harness_output, f"selected_variant={EXPECTED_SELECTED_VARIANT}", "harness")
    require_contains(harness_output, f"native_call_trace={EXPECTED_CALL_TRACE}", "harness")

    stale_stderr = negative_dir / "stale_source_front_door.stderr.txt"
    expect_failure(
        run,
        "negative stale source-front-door metadata is not artifact authority",
        [tcrv_translate, "--tcrv-tensorext-lite-emitc-to-cpp", str(input_path)],
        stderr_path=stale_stderr,
        expected=("stale TensorExtLite source-front-door metadata",),
    )

    unsupported_mlir = negative_dir / "unsupported_artifact_kind.mlir"
    unsupported_mlir.write_text(
        materialized_input_path.read_text(encoding="utf-8").replace(
            'artifact_kind = "riscv-elf-relocatable-object"',
            'artifact_kind = "metadata-diagnostic"',
            1,
        ),
        encoding="utf-8",
    )
    unsupported_stderr = negative_dir / "unsupported_artifact_kind.stderr.txt"
    expect_failure(
        run,
        "negative unsupported artifact kind fails closed",
        [tcrv_translate, "--tcrv-export-target-artifact", str(unsupported_mlir)],
        stderr_path=unsupported_stderr,
        expected=("unsupported artifact_kind", "found none", "metadata-diagnostic"),
    )

    wrong_route_mlir = negative_dir / "wrong_route_identity.mlir"
    wrong_route_mlir.write_text(
        materialized_input_path.read_text(encoding="utf-8").replace(
            f'lowering_pipeline = "{EXPECTED_EMITC_ROUTE}"',
            'lowering_pipeline = "tensorext-lite-fragment-mma-wrong-route"',
            1,
        ),
        encoding="utf-8",
    )
    wrong_route_stderr = negative_dir / "wrong_route_identity.stderr.txt"
    expect_failure(
        run,
        "negative mismatched route identity fails closed",
        [tcrv_translate, "--tcrv-export-target-header-artifact", str(wrong_route_mlir)],
        stderr_path=wrong_route_stderr,
        expected=("found none", "unknown target artifact export route id", "wrong-route"),
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "script": SCRIPT_NAME,
        "created_at": utc_timestamp(),
        "status": "PASS",
        "selected_variant": EXPECTED_SELECTED_VARIANT,
        "origin_plugin": EXPECTED_ORIGIN_PLUGIN,
        "construction_protocol": EXPECTED_CONSTRUCTION_PROTOCOL,
        "emitc_route": EXPECTED_EMITC_ROUTE,
        "runtime_abi_name": EXPECTED_RUNTIME_ABI,
        "runtime_abi_kind": EXPECTED_RUNTIME_ABI_KIND,
        "runtime_abi_parameter_count": 0,
        "bundle_component_group": EXPECTED_COMPONENT_GROUP,
        "target_bundle_object_format": "elf64-littleriscv",
        "target_bundle_object_arch": "riscv64",
        "native_call_trace": EXPECTED_CALL_TRACE,
        "evidence_dir": str(run_dir),
        "artifacts": {
            "source_front_door_input": str(input_copy.relative_to(run_dir)),
            "materialized_input": str(materialized_input_copy.relative_to(run_dir)),
            "generated_cpp": str(generated_cpp.relative_to(run_dir)),
            "generated_header": str(generated_header.relative_to(run_dir)),
            "target_object": str(target_object.relative_to(run_dir)),
            "bundle_index": str(bundle_index.relative_to(run_dir)),
            "bundle_object": str(bundle_object.relative_to(run_dir)),
            "bundle_header": str(bundle_header.relative_to(run_dir)),
            "native_object": str(native_object.relative_to(run_dir)),
            "harness_source": str(harness_source.relative_to(run_dir)),
            "harness_binary": str(harness_binary.relative_to(run_dir)),
            "run_stdout": str(harness_stdout.relative_to(run_dir)),
        },
        "sha256": {
            "generated_cpp": sha256_file(generated_cpp),
            "generated_header": sha256_file(generated_header),
            "target_object": sha256_file(target_object),
            "bundle_object": sha256_file(bundle_object),
            "bundle_header": sha256_file(bundle_header),
            "native_object": sha256_file(native_object),
            "harness_source": sha256_file(harness_source),
        },
    }
    write_json(run_dir / "commands.json", run.commands)
    write_json(run_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--materialized-input", default=str(DEFAULT_MATERIALIZED_INPUT))
    parser.add_argument("--tcrv-translate", default="tcrv-translate")
    parser.add_argument("--clangxx", default="clang++")
    parser.add_argument("--llvm-readobj", default="llvm-readobj")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        summary = create_evidence(parse_args(argv))
    except (EvidenceError, subprocess.TimeoutExpired, OSError) as error:
        print(f"tianchenrv.tensorext_lite.runtime_abi_e2e: FAIL {error}", file=sys.stderr)
        return 1

    print("tianchenrv.tensorext_lite.runtime_abi_e2e: PASS")
    print(f"evidence_dir: {summary['evidence_dir']}")
    print(f"selected_variant: {summary['selected_variant']}")
    print(f"origin_plugin: {summary['origin_plugin']}")
    print(f"construction_protocol: {summary['construction_protocol']}")
    print(f"emitc_route: {summary['emitc_route']}")
    print(f"runtime_abi_name: {summary['runtime_abi_name']}")
    print(f"runtime_abi_parameter_count: {summary['runtime_abi_parameter_count']}")
    print(f"target_bundle_component_group: {summary['bundle_component_group']}")
    print(f"target_bundle_object_format: {summary['target_bundle_object_format']}")
    print(f"target_bundle_object_arch: {summary['target_bundle_object_arch']}")
    print(f"native_call_trace: {summary['native_call_trace']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
