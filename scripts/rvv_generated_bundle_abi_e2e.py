#!/usr/bin/env python3
"""Prove generated RVV object/header bundle ABI consumption on ``ssh rvv``.

This is evidence tooling only. By default it starts from hand-authored explicit
selected ``tcrv.exec`` / ``tcrv_rvv`` body fixtures, materializes selected
emission plans, exports the generated target artifact bundle, checks the
bundle, builds a small external C ABI consumer, and optionally runs that
consumer on the real RVV target. ``--pre-realized-selected-body`` starts from
the bounded pre-realized selected-body fixtures and uses the public selected
lowering-boundary materialization pass before emission planning. The legacy
``--source-seed`` mode is unsupported and exits before bundle generation.
The script does not implement compiler IR, lowering, plugin selection,
emission, descriptors, fallback computation, or runtime glue.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
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
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = Path("artifacts/tmp/rvv_generated_bundle_abi_e2e")
DEFAULT_SSH_TARGET = "rvv"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_RUNTIME_COUNTS = (1, 7, 16, 17, 257)
MIN_RUNTIME_COUNT_CASES = 2
MIN_NON_ONE_VECTOR_SENTINEL_COUNT = 17
DEFAULT_OP_KINDS = ("add", "sub", "mul")
OP_KIND_CHOICES = DEFAULT_OP_KINDS + (
    "cmp_select",
    "reduce_add",
    "masked_add",
    "macc_add",
    "strided_add",
)
REDUCE_ADD_ACCUMULATOR_LAYOUT = "rhs-vector-seed-lane0-per-vl-chunk"
REDUCE_ADD_RESULT_LAYOUT = "store-reduction-lane0-to-output-chunk-base"
REDUCE_ADD_STORE_VL = "1"
MASKED_ADD_MASK_SOURCE = "compare-produced-mask-same-vl-scope"
MASKED_ADD_PASSTHROUGH_LAYOUT = "passthrough-vector-preserves-inactive-lanes"
MACC_ADD_ACCUMULATOR_LAYOUT = "output-buffer-vector-accumulator-input"
MACC_ADD_RESULT_LAYOUT = "store-multiply-accumulate-result-to-output-buffer"
STRIDED_ADD_RUNTIME_ABI_ORDER = "lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride"
STRIDED_ADD_MEMORY_LAYOUT = "element-strided-lhs-rhs-output-runtime-abi"
STRIDED_ADD_LHS_STRIDE_SOURCE = "runtime_abi:lhs_stride"
STRIDED_ADD_RHS_STRIDE_SOURCE = "runtime_abi:rhs_stride"
STRIDED_ADD_OUT_STRIDE_SOURCE = "runtime_abi:out_stride"
OUT_SENTINEL = "(int32_t)0x5a5a5a5a"

INDEX_FILE_NAME = "tianchenrv-target-artifact-bundle.index"
EXPECTED_SELECTED_ROLE = "dispatch case"
EXPECTED_COMPONENT_GROUP = "rvv-generic-typed-body-materialized-emitc-bundle.v1"
EXPECTED_RUNTIME_ABI_KIND = "plugin-owned-runtime-abi"
EXPECTED_OBJECT_ROUTE = "rvv-generic-typed-body-emitc-route-family"
EXPECTED_HEADER_ROUTE = "rvv-generic-typed-body-emitc-route-family.header"
EXPECTED_OWNER = "rvv-plugin"
EXPECTED_OBJECT_KIND = "riscv-elf-relocatable-object"
EXPECTED_HEADER_KIND = "runtime-callable-c-header"


@dataclass(frozen=True)
class OpExpectation:
    kind: str
    input_path: Path
    input_mode: str
    source_seed: bool
    selected_variant: str
    external_abi_name: str
    function_name: str
    emitc_route: str
    typed_compute_op: str
    memory_form: str
    lhs_initializer: str
    rhs_initializer: str
    expected_expression: str
    out_initializer: str = OUT_SENTINEL
    lmul: str = "m1"
    config_contract: str = "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"
    bounded_slice: str = "multi-vl-selected-body-sew32-lmul-m1"

    @property
    def prototype(self) -> str:
        if self.is_strided_add:
            return (
                f"void {self.function_name}(const int32_t *lhs, "
                "const int32_t *rhs, int32_t *out, size_t n, "
                "size_t lhs_stride, size_t rhs_stride, size_t out_stride);"
            )
        return (
            f"void {self.function_name}(const int32_t *lhs, "
            "const int32_t *rhs, int32_t *out, size_t n);"
        )

    @property
    def runtime_parameters(self) -> tuple[dict[str, str], ...]:
        if self.is_strided_add:
            return EXPECTED_STRIDED_RUNTIME_PARAMETERS
        return EXPECTED_RUNTIME_PARAMETERS

    @property
    def runtime_abi_order(self) -> str:
        if self.is_strided_add:
            return STRIDED_ADD_RUNTIME_ABI_ORDER
        return "lhs,rhs,out,n"

    @property
    def pass_marker(self) -> str:
        return f"tcrv_rvv_generated_bundle_abi_{self.kind}_ok"

    @property
    def is_pre_realized(self) -> bool:
        return self.input_mode == "pre-realized-selected-body"

    @property
    def is_rhs_broadcast(self) -> bool:
        return self.input_mode == "rhs-broadcast-selected-body"

    @property
    def is_reduce_add(self) -> bool:
        return self.kind == "reduce_add"

    @property
    def is_masked_add(self) -> bool:
        return self.kind == "masked_add"

    @property
    def is_macc_add(self) -> bool:
        return self.kind == "macc_add"

    @property
    def is_strided_add(self) -> bool:
        return self.kind == "strided_add"


EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS = {
    "add": OpExpectation(
        kind="add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_i32_add",
        external_abi_name="rvv-generic-binary-add-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_add_kernel_explicit_selected_body_rvv_i32_add",
        emitc_route="rvv-generic-binary-add-emitc-route",
        typed_compute_op="tcrv_rvv.binary",
        memory_form="vector-rhs-load",
        lhs_initializer="(int32_t)(7 + (int32_t)(index * 3))",
        rhs_initializer="(int32_t)(1000 - (int32_t)(index * 5))",
        expected_expression="lhs[index] + rhs[index]",
    ),
    "sub": OpExpectation(
        kind="sub",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-sub.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_i32_sub",
        external_abi_name="rvv-generic-binary-sub-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_sub_kernel_explicit_selected_body_rvv_i32_sub",
        emitc_route="rvv-generic-binary-sub-emitc-route",
        typed_compute_op="tcrv_rvv.binary",
        memory_form="vector-rhs-load",
        lhs_initializer="(int32_t)(500 - (int32_t)(index * 2))",
        rhs_initializer="(int32_t)(13 + (int32_t)(index * 5))",
        expected_expression="lhs[index] - rhs[index]",
    ),
    "mul": OpExpectation(
        kind="mul",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-mul.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_i32_mul",
        external_abi_name="rvv-generic-binary-mul-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_mul_kernel_explicit_selected_body_rvv_i32_mul",
        emitc_route="rvv-generic-binary-mul-emitc-route",
        typed_compute_op="tcrv_rvv.binary",
        memory_form="vector-rhs-load",
        lhs_initializer="(int32_t)((int)(index % 13) - 6)",
        rhs_initializer="(int32_t)((int)(index % 17) - 8)",
        expected_expression="lhs[index] * rhs[index]",
    ),
    "cmp_select": OpExpectation(
        kind="cmp_select",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-cmp-select.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_i32_cmp_select",
        external_abi_name="rvv-generic-cmp-select-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_cmp_select_kernel_explicit_selected_body_rvv_i32_cmp_select",
        emitc_route="rvv-generic-cmp-select-emitc-route",
        typed_compute_op="tcrv_rvv.select",
        memory_form="vector-rhs-load",
        lhs_initializer="(int32_t)(41 + (int32_t)(index * 9))",
        rhs_initializer="(int32_t)(-300 - (int32_t)(index * 7))",
        expected_expression="(lhs[index] == rhs[index] ? lhs[index] : rhs[index])",
    ),
    "reduce_add": OpExpectation(
        kind="reduce_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-reduce-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_reduce_add",
        external_abi_name="rvv-generic-reduce-add-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_reduce_add_kernel_explicit_selected_body_rvv_reduce_add",
        emitc_route="rvv-generic-reduce-add-emitc-route",
        typed_compute_op="tcrv_rvv.reduce",
        memory_form="vector-rhs-load",
        lhs_initializer="(int32_t)(1 + (int32_t)(index % 11))",
        rhs_initializer="(int32_t)(1000 + (int32_t)(index * 3))",
        expected_expression="rhs[chunk_start] + sum(lhs[chunk_start:chunk_start+vl])",
    ),
    "masked_add": OpExpectation(
        kind="masked_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-masked-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_i32_masked_add",
        external_abi_name="rvv-generic-masked-add-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_masked_add_kernel_explicit_selected_body_rvv_i32_masked_add",
        emitc_route="rvv-generic-masked-add-emitc-route",
        typed_compute_op="tcrv_rvv.masked_binary",
        memory_form="vector-rhs-load",
        lhs_initializer="(int32_t)(((index % 4) == 0) ? (int32_t)(20 + (int32_t)index) : (int32_t)(3 + (int32_t)index))",
        rhs_initializer="(int32_t)(((index % 4) == 0) ? (int32_t)(20 + (int32_t)index) : (int32_t)(100 + (int32_t)index))",
        expected_expression="(lhs[index] == rhs[index] ? (int32_t)(lhs[index] + rhs[index]) : lhs[index])",
    ),
    "macc_add": OpExpectation(
        kind="macc_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-macc-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_macc_add",
        external_abi_name="rvv-generic-macc-add-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_macc_add_kernel_explicit_selected_body_rvv_macc_add",
        emitc_route="rvv-generic-macc-add-emitc-route",
        typed_compute_op="tcrv_rvv.macc",
        memory_form="vector-rhs-load",
        lhs_initializer="(int32_t)((int)(index % 13) - 6)",
        rhs_initializer="(int32_t)((int)(index % 17) - 8)",
        out_initializer="(int32_t)(17 - (int32_t)(index % 9))",
        expected_expression="(int32_t)((int32_t)(17 - (int32_t)(index % 9)) + (int32_t)(lhs[index] * rhs[index]))",
    ),
    "strided_add": OpExpectation(
        kind="strided_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-strided-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_strided_add",
        external_abi_name="rvv-generic-strided-add-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_strided_add_kernel_explicit_selected_body_rvv_strided_add",
        emitc_route="rvv-generic-strided-add-emitc-route",
        typed_compute_op="tcrv_rvv.binary",
        memory_form="strided-load-store",
        lhs_initializer="(int32_t)(11 + (int32_t)(index * 2))",
        rhs_initializer="(int32_t)(700 - (int32_t)(index * 3))",
        expected_expression="lhs[index * lhs_stride] + rhs[index * rhs_stride]",
    ),
}

RHS_BROADCAST_SELECTED_BODY_OP_EXPECTATIONS = {
    "add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"],
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-broadcast-add.mlir"),
        input_mode="rhs-broadcast-selected-body",
        selected_variant="explicit_selected_body_rvv_i32_broadcast_add",
        function_name="tcrv_emitc_explicit_selected_body_broadcast_add_kernel_explicit_selected_body_rvv_i32_broadcast_add",
        memory_form="rhs-broadcast-load",
        rhs_initializer="(int32_t)(17 - (int32_t)(index % 5))",
        expected_expression="lhs[index] + rhs[0]",
    ),
    "sub": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["sub"],
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-broadcast-sub.mlir"),
        input_mode="rhs-broadcast-selected-body",
        selected_variant="explicit_selected_body_rvv_i32_broadcast_sub",
        function_name="tcrv_emitc_explicit_selected_body_broadcast_sub_kernel_explicit_selected_body_rvv_i32_broadcast_sub",
        memory_form="rhs-broadcast-load",
        rhs_initializer="(int32_t)(-11 + (int32_t)(index % 7))",
        expected_expression="lhs[index] - rhs[0]",
    ),
    "mul": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["mul"],
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-broadcast-mul.mlir"),
        input_mode="rhs-broadcast-selected-body",
        selected_variant="explicit_selected_body_rvv_i32_broadcast_mul",
        function_name="tcrv_emitc_explicit_selected_body_broadcast_mul_kernel_explicit_selected_body_rvv_i32_broadcast_mul",
        memory_form="rhs-broadcast-load",
        rhs_initializer="(int32_t)(3 + (int32_t)(index % 3))",
        expected_expression="lhs[index] * rhs[0]",
    ),
}

LMUL_M2_SELECTED_BODY_OP_EXPECTATIONS = {
    "add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"],
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-m2-add.mlir"),
        input_mode="lmul-m2-selected-body",
        selected_variant="explicit_selected_body_rvv_i32m2_add",
        function_name="tcrv_emitc_explicit_selected_body_m2_add_kernel_explicit_selected_body_rvv_i32m2_add",
        lmul="m2",
        config_contract="rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m2",
    ),
    "sub": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["sub"],
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-m2-sub.mlir"),
        input_mode="lmul-m2-selected-body",
        selected_variant="explicit_selected_body_rvv_i32m2_sub",
        function_name="tcrv_emitc_explicit_selected_body_m2_sub_kernel_explicit_selected_body_rvv_i32m2_sub",
        lmul="m2",
        config_contract="rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m2",
    ),
    "mul": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["mul"],
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-m2-mul.mlir"),
        input_mode="lmul-m2-selected-body",
        selected_variant="explicit_selected_body_rvv_i32m2_mul",
        function_name="tcrv_emitc_explicit_selected_body_m2_mul_kernel_explicit_selected_body_rvv_i32m2_mul",
        lmul="m2",
        config_contract="rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m2",
    ),
}

PRE_REALIZED_SELECTED_BODY_OP_EXPECTATIONS = {
    "add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_i32_add",
        function_name="tcrv_emitc_pre_realized_body_add_kernel_pre_realized_body_rvv_i32_add",
    ),
    "sub": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["sub"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-sub.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_i32_sub",
        function_name="tcrv_emitc_pre_realized_body_sub_kernel_pre_realized_body_rvv_i32_sub",
    ),
    "mul": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["mul"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-mul.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_i32_mul",
        function_name="tcrv_emitc_pre_realized_body_mul_kernel_pre_realized_body_rvv_i32_mul",
    ),
    "strided_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["strided_add"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-strided-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_strided_add",
        function_name="tcrv_emitc_pre_realized_body_strided_add_kernel_pre_realized_body_rvv_strided_add",
    ),
}

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
EXPECTED_STRIDED_RUNTIME_PARAMETERS = EXPECTED_RUNTIME_PARAMETERS + (
    {
        "c_name": "lhs_stride",
        "c_type": "size_t",
        "role": "lhs-input-stride",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "rhs_stride",
        "c_type": "size_t",
        "role": "rhs-input-stride",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "out_stride",
        "c_type": "size_t",
        "role": "output-stride",
        "ownership": "target-export-abi-owned",
    },
)
COMMON_EXPECTED_METADATA = {
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
    "tcrv_rvv.multi_vl": "supported",
}
FORBIDDEN_PUBLIC_RESIDUE_TOKENS = (
    "BinarySelfCheck",
    "binary self-check",
    "self-check",
    "self_check",
    "self check",
    "descriptor",
    "direct-C",
    "direct_c",
    "source-export",
    "source_export",
    "rvv-direct-microkernel",
    "raw log",
    "raw-log",
    "password",
    "passwd",
    "token",
    "secret",
    "private key",
    "authorization:",
    "api_key",
    "access_key",
    "http://",
    "https://",
    "://",
)
FORBIDDEN_HEADER_TOKENS = (
    "__riscv_",
    "vint32m1_t",
    "return;",
    "int main",
    "main(",
) + FORBIDDEN_PUBLIC_RESIDUE_TOKENS

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


def require_no_forbidden_public_residue(text: str, context: str) -> None:
    lowered = text.lower()
    for token in FORBIDDEN_PUBLIC_RESIDUE_TOKENS:
        if token.lower() in lowered:
            raise EvidenceError(
                f"{context}: forbidden public residue token {token!r} present"
            )


def find_record(records: list[dict[str, Any]], component_role: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("component_role") == component_role]
    if len(matches) != 1:
        raise EvidenceError(
            f"expected exactly one {component_role} bundle component, found {len(matches)}"
        )
    return matches[0]


def verify_runtime_parameters(
    record: dict[str, Any], context: str, expectation: OpExpectation
) -> None:
    require_equal(
        record.get("runtime_abi_parameter_count"),
        str(len(expectation.runtime_parameters)),
        f"{context} runtime ABI parameter count",
    )
    require_equal(
        record.get("runtime_abi_parameters"),
        list(expectation.runtime_parameters),
        f"{context} ordered runtime ABI parameters",
    )


def verify_common_record_fields(
    record: dict[str, Any], context: str, expectation: OpExpectation
) -> None:
    require_equal(record.get("component_group"), EXPECTED_COMPONENT_GROUP, f"{context} component group")
    require_equal(record.get("external_abi_name"), expectation.external_abi_name, f"{context} external ABI")
    require_equal(record.get("selected_variant"), expectation.selected_variant, f"{context} selected variant")
    require_equal(record.get("role"), EXPECTED_SELECTED_ROLE, f"{context} selected role")
    require_equal(record.get("owner"), EXPECTED_OWNER, f"{context} owner")
    require_equal(record.get("runtime_abi"), expectation.external_abi_name, f"{context} runtime ABI")
    require_equal(record.get("runtime_abi_kind"), EXPECTED_RUNTIME_ABI_KIND, f"{context} runtime ABI kind")
    require_equal(record.get("runtime_abi_name"), expectation.external_abi_name, f"{context} runtime ABI name")
    verify_runtime_parameters(record, context, expectation)
    components = record.get("components", [])
    require_equal(len(components), 1, f"{context} selected component count")
    require_equal(
        components[0].get("selected_variant"),
        expectation.selected_variant,
        f"{context} selected component variant",
    )
    require_equal(
        components[0].get("role"),
        EXPECTED_SELECTED_ROLE,
        f"{context} selected component role",
    )


def expected_metadata_for(expectation: OpExpectation) -> dict[str, str]:
    common_metadata = dict(COMMON_EXPECTED_METADATA)
    common_metadata["tcrv_rvv.runtime_abi_order"] = expectation.runtime_abi_order
    per_op_metadata = {
        "rvv_emitc_lowerable_route": expectation.emitc_route,
        "rvv_selected_body_operation": expectation.kind,
        "rvv_selected_body_typed_compute_op": expectation.typed_compute_op,
        "tcrv_rvv.config_contract": expectation.config_contract,
        "tcrv_rvv.sew": "32",
        "tcrv_rvv.lmul": expectation.lmul,
        "tcrv_rvv.tail_policy": "agnostic",
        "tcrv_rvv.mask_policy": "agnostic",
        "tcrv_rvv.memory_form": expectation.memory_form,
        "tcrv_rvv.bounded_slice": expectation.bounded_slice,
    }
    if expectation.is_reduce_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.reduction_accumulator_layout": REDUCE_ADD_ACCUMULATOR_LAYOUT,
                "tcrv_rvv.reduction_result_layout": REDUCE_ADD_RESULT_LAYOUT,
                "tcrv_rvv.reduction_store_vl": REDUCE_ADD_STORE_VL,
            }
        )
    if expectation.is_masked_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.mask_source": MASKED_ADD_MASK_SOURCE,
                "tcrv_rvv.masked_passthrough_layout": MASKED_ADD_PASSTHROUGH_LAYOUT,
            }
        )
    if expectation.is_macc_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.macc_accumulator_layout": MACC_ADD_ACCUMULATOR_LAYOUT,
                "tcrv_rvv.macc_result_layout": MACC_ADD_RESULT_LAYOUT,
            }
        )
    if expectation.is_strided_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.strided_memory_layout": STRIDED_ADD_MEMORY_LAYOUT,
                "tcrv_rvv.lhs_stride_source": STRIDED_ADD_LHS_STRIDE_SOURCE,
                "tcrv_rvv.rhs_stride_source": STRIDED_ADD_RHS_STRIDE_SOURCE,
                "tcrv_rvv.out_stride_source": STRIDED_ADD_OUT_STRIDE_SOURCE,
            }
        )
    return {**per_op_metadata, **common_metadata}


def verify_record_metadata(
    record: dict[str, Any], context: str, expectation: OpExpectation
) -> None:
    metadata = metadata_map(record)
    for key, expected in expected_metadata_for(expectation).items():
        require_equal(metadata.get(key), expected, f"{context} metadata {key}")
    for key, value in metadata.items():
        require_no_forbidden_public_residue(
            f"{key}={value}", f"{context} artifact metadata"
        )
        lowered = key.lower()
        if "descriptor" in lowered or "element-count" in lowered or "element_count" in lowered:
            raise EvidenceError(f"{context} metadata key {key!r} is descriptor residue")


def verify_header(header_path: Path, expectation: OpExpectation) -> dict[str, Any]:
    if not header_path.exists():
        raise EvidenceError(f"generated header is missing: {header_path}")
    text = header_path.read_text(encoding="utf-8")
    require_contains(text, "#include <stddef.h>", "generated header")
    require_contains(text, "#include <stdint.h>", "generated header")
    require_contains(text, expectation.prototype, "generated header")
    open_guard = '#ifdef __cplusplus\nextern "C" {\n#endif'
    close_guard = '#ifdef __cplusplus\n} /* extern "C" */\n#endif'
    open_guard_index = text.find(open_guard)
    prototype_index = text.find(expectation.prototype)
    close_guard_index = text.find(close_guard, prototype_index)
    if not (
        open_guard_index >= 0
        and prototype_index >= 0
        and close_guard_index >= 0
        and open_guard_index < prototype_index < close_guard_index
    ):
        raise EvidenceError(
            "generated header public declaration is not wrapped by the "
            'C++ extern "C" guard required for the runtime-callable C ABI'
        )
    require_contains(text, "tianchenrv.rvv.runtime_avl_source: runtime_abi:n", "generated header")
    require_contains(text, "tianchenrv.rvv.multi_vl: supported", "generated header")
    require_no_forbidden_public_residue(text, "generated declaration-only header")
    for token in FORBIDDEN_HEADER_TOKENS:
        require_not_contains(text, token, "generated declaration-only header")
    return {
        "path": str(header_path),
        "size": header_path.stat().st_size,
        "sha256": sha256_file(header_path),
        "prototype": expectation.prototype,
        "extern_c_guard": True,
    }


def verify_object(
    object_path: Path, readobj: str | None, expectation: OpExpectation
) -> dict[str, Any]:
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
        symbols_record = run_command([readobj, "--symbols", str(object_path)], timeout=30)
        require_command_success(symbols_record, "llvm-readobj symbol check")
        symbols_stdout = str(symbols_record.get("stdout", ""))
        require_contains(
            symbols_stdout,
            f"Name: {expectation.function_name}",
            "llvm-readobj symbol check",
        )
        mangled_selected_pattern = re.compile(
            rf"\b_Z[0-9]+{re.escape(expectation.function_name)}"
        )
        if mangled_selected_pattern.search(symbols_stdout):
            raise EvidenceError(
                "generated object exposes a C++-mangled selected function "
                "symbol instead of the runtime-callable C ABI symbol"
            )
        result["symbols"] = {
            "command": symbols_record["command"],
            "selected_symbol": expectation.function_name,
            "unmangled_selected_symbol": True,
        }
    return result


def verify_bundle(
    bundle_dir: Path, readobj: str | None, expectation: OpExpectation
) -> dict[str, Any]:
    index_path = bundle_dir / INDEX_FILE_NAME
    if not index_path.exists():
        raise EvidenceError(f"bundle index is missing: {index_path}")
    index_text = index_path.read_text(encoding="utf-8")
    require_no_forbidden_public_residue(index_text, "bundle index")
    parsed = parse_bundle_index(index_text)
    require_equal(parsed.get("version"), "1", "bundle version")
    require_equal(parsed.get("bundle_status"), "complete", "bundle status")
    require_equal(parsed.get("artifact_count"), "2", "bundle artifact count")
    records = parsed.get("records", [])
    require_equal(len(records), 2, "bundle record count")

    object_record = find_record(records, "object")
    header_record = find_record(records, "header")
    verify_common_record_fields(object_record, "object record", expectation)
    verify_common_record_fields(header_record, "header record", expectation)
    verify_record_metadata(object_record, "object record", expectation)
    verify_record_metadata(header_record, "header record", expectation)
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
        "object": verify_object(object_path, readobj, expectation),
        "header": verify_header(header_path, expectation),
        "object_file": object_file,
        "header_file": header_file,
    }


def verify_materialized_selected_body(
    materialized_path: Path, expectation: OpExpectation
) -> dict[str, Any]:
    if not materialized_path.exists():
        raise EvidenceError(
            f"materialized selected-body MLIR is missing: {materialized_path}"
        )
    text = materialized_path.read_text(encoding="utf-8")
    require_contains(
        text,
        f"@{expectation.selected_variant}",
        "materialized selected-body MLIR selected variant",
    )
    require_contains(
        text,
        "tcrv_rvv.with_vl",
        "materialized selected-body MLIR lowering boundary",
    )
    require_contains(
        text,
        expectation.typed_compute_op,
        "materialized selected-body MLIR typed compute op",
    )
    require_contains(
        text,
        f'lmul = "{expectation.lmul}"',
        "materialized selected-body MLIR LMUL config",
    )
    if expectation.is_rhs_broadcast:
        require_contains(
            text,
            "tcrv_rvv.broadcast_load",
            "materialized selected-body MLIR RHS broadcast load",
        )
    if expectation.is_strided_add:
        require_contains(
            text,
            "tcrv_rvv.strided_load",
            "materialized selected-body MLIR strided load",
        )
        require_contains(
            text,
            "tcrv_rvv.strided_store",
            "materialized selected-body MLIR strided store",
        )
    if expectation.is_pre_realized:
        require_not_contains(
            text,
            "tcrv_rvv.typed_binary_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
    require_no_forbidden_public_residue(text, "materialized selected-body MLIR")
    return {
        "path": str(materialized_path),
        "sha256": sha256_file(materialized_path),
        "selected_variant": expectation.selected_variant,
        "typed_compute_op": expectation.typed_compute_op,
        "memory_form": expectation.memory_form,
        "lmul": expectation.lmul,
        "contains_with_vl": True,
        "pre_realized_body_consumed": expectation.is_pre_realized,
    }


def harness_source(
    header_file_name: str, runtime_counts: list[int], expectation: OpExpectation
) -> str:
    counts = ", ".join(str(count) for count in runtime_counts)
    if expectation.is_strided_add:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  const size_t lhs_stride = 2;
  const size_t rhs_stride = 3;
  const size_t out_stride = 2;
  const size_t max_stride = rhs_stride;
  size_t alloc_n = (n == 0 ? 1 : n) * max_stride + 8;
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

  for (size_t index = 0; index < alloc_n; ++index) {{
    lhs[index] = (int32_t)-1234567;
    rhs[index] = (int32_t)1234567;
    out[index] = {OUT_SENTINEL};
  }}

  for (size_t index = 0; index < n; ++index) {{
    lhs[index * lhs_stride] = {expectation.lhs_initializer};
    rhs[index * rhs_stride] = {expectation.rhs_initializer};
  }}

  {expectation.function_name}(lhs, rhs, out, n, lhs_stride, rhs_stride, out_stride);

  for (size_t index = 0; index < n; ++index) {{
    int32_t expected = {expectation.expected_expression};
    size_t out_index = index * out_stride;
    if (out[out_index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu logical_index=%zu out_index=%zu got=%d expected=%d lhs=%d rhs=%d\\n",
              n, index, out_index, out[out_index], expected,
              lhs[index * lhs_stride], rhs[index * rhs_stride]);
      free(lhs);
      free(rhs);
      free(out);
      return 12;
    }}
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    if ((index % out_stride) == 0 && (index / out_stride) < n)
      continue;
    if (out[index] != {OUT_SENTINEL}) {{
      fprintf(stderr,
              "{expectation.kind} touched non-strided output lane n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, out[index], {OUT_SENTINEL});
      free(lhs);
      free(rhs);
      free(out);
      return 13;
    }}
  }}

  free(lhs);
  free(rhs);
  free(out);
  printf("{expectation.kind} case n=%zu ok\\n", n);
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
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)}\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_reduce_add:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <riscv_vector.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
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
    lhs[index] = {expectation.lhs_initializer};
    rhs[index] = {expectation.rhs_initializer};
    out[index] = {OUT_SENTINEL};
  }}

  {expectation.function_name}(lhs, rhs, out, n);

  if (n > 0) {{
    size_t full_chunk_vl = __riscv_vsetvl_e32m1(n);
    if (full_chunk_vl == 0) {{
      fprintf(stderr, "reduce_add invalid full_chunk_vl=0 for n=%zu\\n", n);
      free(lhs);
      free(rhs);
      free(out);
      return 13;
    }}

    for (size_t chunk_start = 0; chunk_start < n; chunk_start += full_chunk_vl) {{
      size_t vl = __riscv_vsetvl_e32m1(n - chunk_start);
      int32_t expected = rhs[chunk_start];
      for (size_t lane = 0; lane < vl; ++lane)
        expected = (int32_t)(expected + lhs[chunk_start + lane]);

      if (out[chunk_start] != expected) {{
        fprintf(stderr,
                "reduce_add mismatch n=%zu chunk_start=%zu got=%d expected=%d rhs_seed=%d vl=%zu\\n",
                n, chunk_start, out[chunk_start], expected, rhs[chunk_start], vl);
        free(lhs);
        free(rhs);
        free(out);
        return 12;
      }}

      for (size_t lane = 1; lane < vl; ++lane) {{
        size_t index = chunk_start + lane;
        if (out[index] != {OUT_SENTINEL}) {{
          fprintf(stderr,
                  "reduce_add touched non-result lane n=%zu index=%zu got=%d sentinel=%d\\n",
                  n, index, out[index], {OUT_SENTINEL});
          free(lhs);
          free(rhs);
          free(out);
          return 14;
        }}
      }}
    }}
  }}

  free(lhs);
  free(rhs);
  free(out);
  printf("{expectation.kind} case n=%zu ok\\n", n);
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
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)}\\n");
  return 0;
}}
""".lstrip()
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
    lhs[index] = {expectation.lhs_initializer};
    rhs[index] = {expectation.rhs_initializer};
    out[index] = {expectation.out_initializer};
  }}

  {expectation.function_name}(lhs, rhs, out, n);

  for (size_t index = 0; index < n; ++index) {{
    int32_t expected = {expectation.expected_expression};
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d lhs=%d rhs=%d\\n",
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
  printf("{expectation.kind} case n=%zu ok\\n", n);
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
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)}\\n");
  return 0;
}}
""".lstrip()


def generate_bundle(
    tcrv_opt: str,
    tcrv_translate: str,
    expectation: OpExpectation,
    bundle_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    materialized_path = bundle_dir.parent / "materialized_selected_body.mlir"
    materialize_command = [tcrv_opt, str(expectation.input_path)]
    if expectation.is_pre_realized:
        materialize_command.append("--tcrv-materialize-selected-lowering-boundaries")
    materialize_command.extend(
        ["--tcrv-materialize-emission-plans", "-o", str(materialized_path)]
    )
    materialize_record = run_command(materialize_command, timeout=timeout)
    require_command_success(
        materialize_record,
        "tcrv-opt explicit selected-body emission-plan materialization",
    )

    translate_command = [
        tcrv_translate,
        "--tcrv-export-target-artifact-bundle",
        f"--tcrv-target-artifact-bundle-output-dir={bundle_dir}",
        str(materialized_path),
    ]
    translate_record = run_command(translate_command, timeout=timeout)
    require_command_success(
        translate_record,
        "tcrv-translate selected typed-body artifact bundle export",
    )
    result = {
        "input_mode": expectation.input_mode,
        "front_door": "explicit-selected-tcrv-exec-rvv-body",
        "materializer": "none-selected-body-already-explicit",
        "source_seed": expectation.source_seed,
        "target_export": "tcrv-export-target-artifact-bundle",
        "materialized_selected_body": str(materialized_path),
        "pipeline": (
            command_display(materialize_command)
            + " && "
            + command_display(translate_command)
        ),
        "tcrv_opt": materialize_record,
        "tcrv_translate": translate_record,
    }
    if expectation.is_pre_realized:
        result["front_door"] = "pre-realized-selected-tcrv-exec-rvv-body"
        result["materializer"] = "tcrv-materialize-selected-lowering-boundaries"
        result["realization_boundary"] = (
            "public selected lowering-boundary materialization consumed the "
            "pre-realized typed tcrv_rvv body before provider route construction"
        )
    return result


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
    expectation: OpExpectation,
    ssh_target: str,
    connect_timeout: int,
    timeout: int,
    object_path: Path,
    header_path: Path,
    harness_path: Path,
) -> dict[str, Any]:
    remote_dir = (
        f"/tmp/tianchenrv_rvv_generated_bundle_abi_"
        f"{safe_run_id(run_id)}_{expectation.kind}"
    )
    remote_object = f"{remote_dir}/{object_path.name}"
    remote_header = f"{remote_dir}/{header_path.name}"
    remote_harness = f"{remote_dir}/{harness_path.name}"
    remote_binary = f"{remote_dir}/rvv_generated_bundle_abi_{expectation.kind}_harness"

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
        expectation.pass_marker,
        "remote generated bundle ABI harness output",
    )
    require_contains(
        str(run_record.get("stdout", "")),
        f"PASS op={expectation.kind}",
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
        "op_kind": expectation.kind,
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


def selected_expectations(args: argparse.Namespace) -> list[OpExpectation]:
    op_kinds = args.op_kind or list(DEFAULT_OP_KINDS)
    if len(set(op_kinds)) != len(op_kinds):
        raise EvidenceError(f"duplicate --op-kind values are not allowed: {op_kinds}")
    if args.input is not None and len(op_kinds) != 1:
        raise EvidenceError("--input may only be used with exactly one --op-kind")
    if args.source_seed:
        raise EvidenceError(
            "legacy RVV --source-seed evidence mode is unsupported during "
            "Stage1 residue hygiene; use explicit selected generic typed "
            "tcrv_rvv body fixtures instead"
        )
    selected_modes = [
        args.pre_realized_selected_body,
        args.rhs_broadcast_selected_body,
        args.lmul_m2_selected_body,
    ]
    if sum(1 for selected in selected_modes if selected) > 1:
        raise EvidenceError(
            "--pre-realized-selected-body, --rhs-broadcast-selected-body, "
            "and --lmul-m2-selected-body are "
            "mutually exclusive"
        )

    if args.pre_realized_selected_body:
        expectation_table = PRE_REALIZED_SELECTED_BODY_OP_EXPECTATIONS
        mode = "pre-realized-selected-body"
    elif args.rhs_broadcast_selected_body:
        expectation_table = RHS_BROADCAST_SELECTED_BODY_OP_EXPECTATIONS
        mode = "rhs-broadcast-selected-body"
    elif args.lmul_m2_selected_body:
        expectation_table = LMUL_M2_SELECTED_BODY_OP_EXPECTATIONS
        mode = "lmul-m2-selected-body"
    else:
        expectation_table = EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS
        mode = "explicit-selected-body"
    unsupported = [kind for kind in op_kinds if kind not in expectation_table]
    if unsupported:
        raise EvidenceError(
            f"--op-kind values {unsupported} are not supported in {mode} mode"
        )
    expectations = [expectation_table[kind] for kind in op_kinds]
    if args.input is not None:
        expectations = [replace(expectations[0], input_path=args.input)]
    return [
        replace(
            expectation,
            input_path=resolve_repo_relative_path(expectation.input_path),
        )
        for expectation in expectations
    ]


def resolve_repo_relative_path(path: Path) -> Path:
    if path.exists() or path.is_absolute():
        return path
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    return path


def validate_runtime_counts(runtime_counts: list[int]) -> None:
    if len(runtime_counts) < MIN_RUNTIME_COUNT_CASES:
        raise EvidenceError(
            "runtime ABI evidence requires several runtime n counts; "
            f"got {runtime_counts}"
        )
    if len(set(runtime_counts)) != len(runtime_counts):
        raise EvidenceError(
            f"runtime ABI evidence requires distinct runtime n counts: {runtime_counts}"
        )
    if any(count < 0 for count in runtime_counts):
        raise EvidenceError(
            f"runtime ABI evidence requires non-negative runtime n counts: {runtime_counts}"
        )
    if max(runtime_counts) < MIN_NON_ONE_VECTOR_SENTINEL_COUNT:
        raise EvidenceError(
            "runtime ABI evidence must include a bounded non-one-vector stress "
            f"count n >= {MIN_NON_ONE_VECTOR_SENTINEL_COUNT}; got {runtime_counts}"
        )


def runtime_count_contract_summary(runtime_counts: list[int]) -> dict[str, Any]:
    return {
        "minimum_case_count": MIN_RUNTIME_COUNT_CASES,
        "non_one_vector_sentinel_min_n": MIN_NON_ONE_VECTOR_SENTINEL_COUNT,
        "case_count": len(runtime_counts),
        "max_runtime_count": max(runtime_counts),
    }


def run_one_op_e2e(
    *,
    args: argparse.Namespace,
    run_id: str,
    artifact_dir: Path,
    expectation: OpExpectation,
    tcrv_opt: str,
    tcrv_translate: str,
    readobj: str | None,
    runtime_counts: list[int],
) -> dict[str, Any]:
    op_artifact_dir = artifact_dir / expectation.kind
    bundle_dir = op_artifact_dir / "generated_bundle"
    bundle_dir.mkdir(parents=True)

    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": SCRIPT_NAME,
        "status": "started",
        "created_at": utc_timestamp(),
        "run_id": run_id,
        "op_kind": expectation.kind,
        "input_mode": expectation.input_mode,
        "source_seed": expectation.source_seed,
        "dry_run": bool(args.dry_run),
        "input": str(expectation.input_path),
        "artifact_dir": str(op_artifact_dir),
        "runtime_counts": runtime_counts,
        "runtime_count_contract": runtime_count_contract_summary(runtime_counts),
        "expected_selected_variant": expectation.selected_variant,
        "expected_runtime_abi_name": expectation.external_abi_name,
        "expected_function": expectation.function_name,
    }

    try:
        if expectation.is_pre_realized:
            input_copy_name = "pre_realized_selected_body_input.mlir"
        else:
            input_copy_name = "selected_body_input.mlir"
        input_copy = op_artifact_dir / input_copy_name
        shutil.copyfile(expectation.input_path, input_copy)
        evidence["selected_input"] = {
            "path": str(expectation.input_path),
            "copy": str(input_copy),
            "sha256": sha256_file(input_copy),
            "mode": expectation.input_mode,
            "source_seed": expectation.source_seed,
        }

        local = generate_bundle(
            tcrv_opt,
            tcrv_translate,
            expectation,
            bundle_dir,
            args.timeout,
        )
        evidence["local_bundle_generation"] = local
        evidence["materialized_selected_body_checks"] = (
            verify_materialized_selected_body(
                Path(local["materialized_selected_body"]), expectation
            )
        )
        bundle_checks = verify_bundle(bundle_dir, readobj, expectation)
        evidence["bundle_checks"] = bundle_checks

        header_path = bundle_dir / bundle_checks["header_file"]
        object_path = bundle_dir / bundle_checks["object_file"]
        harness_path = (
            op_artifact_dir
            / f"rvv_generated_bundle_abi_{expectation.kind}_harness.c"
        )
        harness_path.write_text(
            harness_source(bundle_checks["header_file"], runtime_counts, expectation),
            encoding="utf-8",
        )
        evidence["harness"] = {
            "path": str(harness_path),
            "sha256": sha256_file(harness_path),
            "pass_marker": expectation.pass_marker,
            "boundary": "external C ABI consumer of generated header and object only",
        }

        if args.dry_run:
            evidence["status"] = "dry_run_success"
            evidence["ssh_evidence"] = False
        else:
            remote = run_remote_evidence(
                artifact_dir=op_artifact_dir,
                run_id=run_id,
                expectation=expectation,
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
        write_json(op_artifact_dir / "evidence.json", evidence)
        return evidence
    except Exception as exc:  # noqa: BLE001 - evidence should record exact blocker.
        evidence["status"] = "blocked" if not args.dry_run else "failed"
        evidence["completed_at"] = utc_timestamp()
        evidence["diagnostic"] = sanitize_text(exc)
        write_json(op_artifact_dir / "evidence.json", evidence)
        raise EvidenceError(
            f"{expectation.kind} generated bundle ABI evidence failed: {exc}"
        ) from exc


def run_e2e(args: argparse.Namespace) -> int:
    run_id = safe_run_id(args.run_id or utc_run_id())
    artifact_dir = prepare_artifact_dir(args.artifact_root, run_id, args.overwrite)
    runtime_counts = args.runtime_count or list(DEFAULT_RUNTIME_COUNTS)
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": SCRIPT_NAME,
        "status": "started",
        "created_at": utc_timestamp(),
        "run_id": run_id,
        "dry_run": bool(args.dry_run),
        "input_mode": "explicit-selected-body",
        "source_seed": bool(args.source_seed),
        "pre_realized_selected_body": bool(args.pre_realized_selected_body),
        "rhs_broadcast_selected_body": bool(args.rhs_broadcast_selected_body),
        "lmul_m2_selected_body": bool(args.lmul_m2_selected_body),
        "artifact_dir": str(artifact_dir),
        "runtime_counts": runtime_counts,
        "op_results": {},
    }
    if args.pre_realized_selected_body:
        evidence["input_mode"] = "pre-realized-selected-body"
    elif args.rhs_broadcast_selected_body:
        evidence["input_mode"] = "rhs-broadcast-selected-body"
    elif args.lmul_m2_selected_body:
        evidence["input_mode"] = "lmul-m2-selected-body"
    try:
        validate_runtime_counts(runtime_counts)
        evidence["runtime_count_contract"] = runtime_count_contract_summary(
            runtime_counts
        )
        expectations = selected_expectations(args)
        evidence["op_kinds"] = [expectation.kind for expectation in expectations]
        tcrv_opt = ensure_tool(args.tcrv_opt)
        tcrv_translate = ensure_tool(args.tcrv_translate)
        readobj = ensure_tool(args.llvm_readobj) if args.llvm_readobj else None

        for expectation in expectations:
            result = run_one_op_e2e(
                args=args,
                run_id=run_id,
                artifact_dir=artifact_dir,
                expectation=expectation,
                tcrv_opt=tcrv_opt,
                tcrv_translate=tcrv_translate,
                readobj=readobj,
                runtime_counts=runtime_counts,
            )
            evidence["op_results"][expectation.kind] = {
                "status": result["status"],
                "artifact_dir": result["artifact_dir"],
                "ssh_evidence": result["ssh_evidence"],
                "pass_marker": result["harness"]["pass_marker"],
                "remote_output": result.get("remote", {}).get("remote_output", ""),
            }

        evidence["ssh_evidence"] = not args.dry_run
        evidence["status"] = "success" if not args.dry_run else "dry_run_success"
        evidence["completed_at"] = utc_timestamp()
        write_json(artifact_dir / "evidence.json", evidence)
        print(f"{SCRIPT_NAME}: {evidence['status']}")
        print(f"artifact_dir: {artifact_dir}")
        if evidence.get("ssh_evidence"):
            for op_kind, result in evidence["op_results"].items():
                print(f"[{op_kind}] {str(result['remote_output']).strip()}")
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


def make_fake_bundle(root: Path, expectation: OpExpectation) -> Path:
    bundle_dir = root / "bundle"
    bundle_dir.mkdir(parents=True)
    object_name = (
        "artifact-0-riscv-elf-relocatable-object-"
        "rvv-generic-typed-body-emitc-route-family.o"
    )
    header_name = (
        "artifact-1-runtime-callable-c-header-"
        "rvv-generic-typed-body-emitc-route-family.header.h"
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
#ifdef __cplusplus
extern "C" {{
#endif
{expectation.prototype}
#ifdef __cplusplus
}} /* extern "C" */
#endif
#endif
""".lstrip(),
        encoding="utf-8",
    )
    expected_metadata = expected_metadata_for(expectation)
    metadata_lines = "\n".join(
        f"""  artifact_metadata[{index}]:
    key: "{key}"
    value: "{value}" """
        for index, (key, value) in enumerate(expected_metadata.items())
    )
    parameter_lines = "\n".join(
        f"""  runtime_abi_parameter[{index}]:
    c_name: "{parameter['c_name']}"
    c_type: "{parameter['c_type']}"
    role: "{parameter['role']}"
    ownership: "{parameter['ownership']}" """
        for index, parameter in enumerate(expectation.runtime_parameters)
    )
    runtime_parameter_count = len(expectation.runtime_parameters)
    index_text = f"""
tianchenrv.target_artifact_bundle.version: 1
bundle_status: "complete"
artifact_count: 2
artifact[0]:
  file_name: "{object_name}"
  component_group: "{EXPECTED_COMPONENT_GROUP}"
  component_role: "object"
  external_abi_name: "{expectation.external_abi_name}"
  selected_variant: @{expectation.selected_variant}
  role: "{EXPECTED_SELECTED_ROLE}"
  component[0]:
    selected_variant: @{expectation.selected_variant}
    role: "{EXPECTED_SELECTED_ROLE}"
  artifact_kind: "{EXPECTED_OBJECT_KIND}"
  route: "{EXPECTED_OBJECT_ROUTE}"
  owner: "{EXPECTED_OWNER}"
  runtime_abi: "{expectation.external_abi_name}"
  runtime_abi_kind: "{EXPECTED_RUNTIME_ABI_KIND}"
  runtime_abi_name: "{expectation.external_abi_name}"
  runtime_abi_parameter_count: {runtime_parameter_count}
{parameter_lines}
{metadata_lines}
  handoff_kind: "materialized-emitc-cpp-rvv-intrinsic-object"
  evidence_role: "relocatable-object"
artifact[1]:
  file_name: "{header_name}"
  component_group: "{EXPECTED_COMPONENT_GROUP}"
  component_role: "header"
  external_abi_name: "{expectation.external_abi_name}"
  selected_variant: @{expectation.selected_variant}
  role: "{EXPECTED_SELECTED_ROLE}"
  component[0]:
    selected_variant: @{expectation.selected_variant}
    role: "{EXPECTED_SELECTED_ROLE}"
  artifact_kind: "{EXPECTED_HEADER_KIND}"
  route: "{EXPECTED_HEADER_ROUTE}"
  owner: "{EXPECTED_OWNER}"
  runtime_abi: "{expectation.external_abi_name}"
  runtime_abi_kind: "{EXPECTED_RUNTIME_ABI_KIND}"
  runtime_abi_name: "{expectation.external_abi_name}"
  runtime_abi_parameter_count: {runtime_parameter_count}
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


def require_self_test_sanitized(name: str, raw: str, forbidden: str) -> None:
    sanitized = sanitize_text(raw)
    if forbidden in sanitized:
        raise AssertionError(f"self-test sanitizer failed to redact {name}")
    if "[REDACTED" not in sanitized:
        raise AssertionError(f"self-test sanitizer produced no marker for {name}")


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="tcrv-rvv-generated-bundle-self-test-") as tmp_raw:
        tmp = Path(tmp_raw)
        validate_runtime_counts([7, 16, 23])
        expect_self_test_failure(
            "single runtime count", lambda: validate_runtime_counts([23])
        )
        expect_self_test_failure(
            "duplicate runtime count", lambda: validate_runtime_counts([7, 23, 23])
        )
        expect_self_test_failure(
            "missing non-one-vector runtime count",
            lambda: validate_runtime_counts([7, 16]),
        )

        for expectation in (
            list(EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS.values())
            + list(RHS_BROADCAST_SELECTED_BODY_OP_EXPECTATIONS.values())
            + list(LMUL_M2_SELECTED_BODY_OP_EXPECTATIONS.values())
            + list(PRE_REALIZED_SELECTED_BODY_OP_EXPECTATIONS.values())
        ):
            bundle = make_fake_bundle(
                tmp / expectation.input_mode / expectation.kind, expectation
            )
            verify_bundle(bundle, readobj=None, expectation=expectation)
            harness = harness_source(
                "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h",
                [1, 17, 257],
                expectation,
            )
            if (
                expectation.function_name not in harness
                or expectation.pass_marker not in harness
                or expectation.expected_expression not in harness
            ):
                raise AssertionError(
                    f"self-test harness generation lost {expectation.kind} ABI call"
                )

        expectation = EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"]
        missing_header = make_fake_bundle(tmp / "missing-header", expectation)
        header = next(missing_header.glob("*.h"))
        header.unlink()
        expect_self_test_failure(
            "missing header", lambda: verify_bundle(missing_header, None, expectation)
        )

        missing_object = make_fake_bundle(tmp / "missing-object", expectation)
        obj = next(missing_object.glob("*.o"))
        obj.unlink()
        expect_self_test_failure(
            "missing object", lambda: verify_bundle(missing_object, None, expectation)
        )

        bad_order = make_fake_bundle(tmp / "bad-order", expectation)
        index_path = bad_order / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace('role: "lhs-input-buffer"', 'role: "rhs-input-buffer"', 1)
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure(
            "stale ABI order", lambda: verify_bundle(bad_order, None, expectation)
        )

        missing_metadata = make_fake_bundle(tmp / "missing-metadata", expectation)
        index_path = missing_metadata / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace('value: "supported"', 'value: "missing"', 1)
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure(
            "missing multi-VL metadata",
            lambda: verify_bundle(missing_metadata, None, expectation),
        )

        sub_expectation = EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["sub"]
        stale_arithmetic = make_fake_bundle(tmp / "stale-arithmetic", sub_expectation)
        index_path = stale_arithmetic / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace('value: "sub"', 'value: "add"', 1)
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure(
            "stale arithmetic metadata",
            lambda: verify_bundle(stale_arithmetic, None, sub_expectation),
        )

        bad_header = make_fake_bundle(tmp / "bad-header", expectation)
        header = next(bad_header.glob("*.h"))
        header.write_text(header.read_text(encoding="utf-8") + "\n__riscv_vadd_vv_i32m1\n", encoding="utf-8")
        expect_self_test_failure(
            "header intrinsic body residue",
            lambda: verify_bundle(bad_header, None, expectation),
        )

        missing_extern_c = make_fake_bundle(tmp / "missing-extern-c", expectation)
        header = next(missing_extern_c.glob("*.h"))
        header.write_text(
            header.read_text(encoding="utf-8").replace('extern "C" {\n', ""),
            encoding="utf-8",
        )
        expect_self_test_failure(
            "missing extern C guard",
            lambda: verify_bundle(missing_extern_c, None, expectation),
        )

        mismatched_variant = make_fake_bundle(tmp / "mismatched-variant", expectation)
        index_path = mismatched_variant / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace(f"@{expectation.selected_variant}", "@stale_variant", 1)
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure(
            "mismatched selected variant",
            lambda: verify_bundle(mismatched_variant, None, expectation),
        )

        stale_object_route = make_fake_bundle(tmp / "stale-object-route", expectation)
        index_path = stale_object_route / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace(
            f'route: "{EXPECTED_OBJECT_ROUTE}"',
            'route: "rvv-stale-object-route"',
            1,
        )
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure(
            "stale object route",
            lambda: verify_bundle(stale_object_route, None, expectation),
        )

        stale_header_route = make_fake_bundle(tmp / "stale-header-route", expectation)
        index_path = stale_header_route / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace(
            f'route: "{EXPECTED_HEADER_ROUTE}"',
            'route: "rvv-stale-header-route"',
            1,
        )
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure(
            "stale header route",
            lambda: verify_bundle(stale_header_route, None, expectation),
        )

        stale_runtime_abi = make_fake_bundle(tmp / "stale-runtime-abi", expectation)
        index_path = stale_runtime_abi / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace(
            expectation.external_abi_name,
            "rvv-generic-stale-callable-c-abi.v1",
            1,
        )
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure(
            "stale runtime ABI",
            lambda: verify_bundle(stale_runtime_abi, None, expectation),
        )

        descriptor_residue = make_fake_bundle(tmp / "descriptor-residue", expectation)
        index_path = descriptor_residue / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace(
            'key: "rvv_emitc_lowerable_route"',
            'key: "descriptor.route"',
            1,
        )
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure(
            "descriptor metadata residue",
            lambda: verify_bundle(descriptor_residue, None, expectation),
        )

        direct_c_residue = make_fake_bundle(tmp / "direct-c-residue", expectation)
        index_path = direct_c_residue / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace(
            'value: "emitc.for"',
            'value: "direct-C source-export residue"',
            1,
        )
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure(
            "direct-C source-export metadata residue",
            lambda: verify_bundle(direct_c_residue, None, expectation),
        )

        self_check_residue = make_fake_bundle(tmp / "self-check-residue", expectation)
        header = next(self_check_residue.glob("*.h"))
        header.write_text(
            header.read_text(encoding="utf-8") + "\n/* self-check helper */\n",
            encoding="utf-8",
        )
        expect_self_test_failure(
            "header self-check residue",
            lambda: verify_bundle(self_check_residue, None, expectation),
        )

        credential_residue = make_fake_bundle(tmp / "credential-residue", expectation)
        header = next(credential_residue.glob("*.h"))
        header.write_text(
            header.read_text(encoding="utf-8") + "\n/* TOKEN=leaked */\n",
            encoding="utf-8",
        )
        expect_self_test_failure(
            "header credential residue",
            lambda: verify_bundle(credential_residue, None, expectation),
        )

        require_self_test_sanitized(
            "private key",
            "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----",
            "abc",
        )
        require_self_test_sanitized(
            "authorization bearer",
            "Authorization: Bearer raw-secret-token",
            "raw-secret-token",
        )
        require_self_test_sanitized(
            "environment token",
            "TCRV_API_TOKEN=raw-secret-token",
            "raw-secret-token",
        )
        require_self_test_sanitized(
            "password key",
            "password: raw-secret-token",
            "raw-secret-token",
        )

    print(f"{SCRIPT_NAME} self-test passed")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run local parser/verifier self-tests")
    parser.add_argument("--dry-run", action="store_true", help="generate and verify local bundle without ssh rvv")
    parser.add_argument(
        "--source-seed",
        action="store_true",
        help=(
            "unsupported legacy RVV source-front-door seed mode; exits before "
            "bundle generation"
        ),
    )
    parser.add_argument(
        "--pre-realized-selected-body",
        action="store_true",
        help=(
            "use the pre-realized selected-body add/sub/mul/strided_add "
            "fixtures and run public selected lowering-boundary "
            "materialization before emission planning; mutually exclusive "
            "with --rhs-broadcast-selected-body and --lmul-m2-selected-body"
        ),
    )
    parser.add_argument(
        "--rhs-broadcast-selected-body",
        action="store_true",
        help=(
            "use explicit selected-body add/sub/mul fixtures where rhs is "
            "produced by tcrv_rvv.broadcast_load; mutually exclusive with "
            "--pre-realized-selected-body and "
            "--lmul-m2-selected-body"
        ),
    )
    parser.add_argument(
        "--lmul-m2-selected-body",
        action="store_true",
        help=(
            "use explicit selected-body add/sub/mul fixtures with "
            "generic !tcrv_rvv.vector<i32, \"m2\"> dataflow and lmul=m2 "
            "config; mutually exclusive "
            "with pre-realized and rhs-broadcast modes"
        ),
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--op-kind",
        choices=OP_KIND_CHOICES,
        action="append",
        default=[],
        help="op kind to prove; may be repeated; defaults to add, sub, and mul",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "override the selected-body MLIR fixture for exactly one --op-kind; "
            "applies to explicit or pre-realized selected-body modes"
        ),
    )
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
