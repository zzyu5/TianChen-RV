#!/usr/bin/env python3
"""Drive bounded explicit RVV microkernel source-export evidence.

This helper is runner/evidence tooling only. It orchestrates existing
TianChen-RV MLIR tools and optional ``ssh rvv`` compile/run evidence for the
explicit RVV binary microkernel first slice. It does not implement
compiler IR, plugin decisions, capability modeling, lowering, emission,
runtime ABI, correctness logic, or performance measurement.
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
DEFAULT_BUNDLE_ARTIFACT_ROOT = Path("artifacts/tmp/rvv_microkernel_bundle_e2e")
DEFAULT_SSH_TARGET = "rvv"
DEFAULT_TIMEOUT_SECONDS = 60
BUNDLE_INDEX_FILE_NAME = "tianchenrv-target-artifact-bundle.index"
DIRECT_EXTERNAL_RUNTIME_COUNTS = [7, 16]
DTYPE_SCALAR_C_TYPES = {
    "i32": "int32_t",
    "i64": "int64_t",
}

ARITHMETIC_FAMILY_SPECS: dict[str, dict[str, str | Path]] = {
    "i32-vadd": {
        "diagnostic_name": "i32-vadd",
        "default_input": DEFAULT_INPUT,
        "default_vector_shape": "i32m1",
        "selected_variant": "rvv_first_slice",
        "microkernel_op_name": "tcrv_rvv.i32_vadd_microkernel",
        "arithmetic_op_name": "tcrv_rvv.i32_add",
        "intrinsic": "__riscv_vadd_vv_i32m1",
        "function_stem": "i32_vadd",
        "result_vec": "sum_vec",
        "c_operator": "+",
        "source_route": "tcrv-export-rvv-microkernel-c",
        "header_route": "tcrv-export-rvv-microkernel-header",
        "object_route": "tcrv-export-rvv-microkernel-object",
        "emission_kind": "rvv-explicit-i32-vadd-microkernel-c-source",
        "runtime_abi": "rvv-i32-vadd-runtime-callable-c-abi.v1",
        "runtime_abi_kind": "rvv-runtime-callable-c-abi",
        "runtime_abi_name": "rvv-i32-vadd-runtime-callable-c-function.v1",
        "runtime_glue_role": "runtime-callable-i32-vadd-function",
        "component_group": "rvv-i32-vadd-microkernel-external-abi.v1",
        "external_abi_name": "rvv-i32-vadd-runtime-callable-c-function.v1",
        "self_check_success_marker": "tcrv_rvv_microkernel_ok",
        "external_abi_success_marker": "tcrv_rvv_microkernel_external_abi_ok",
    },
    "i32-vsub": {
        "diagnostic_name": "i32-vsub",
        "default_input": Path("test/Target/RVVMicrokernel/rvv-microkernel-family-sub.mlir"),
        "default_vector_shape": "i32m1",
        "i32m2_default_input": Path(
            "test/Target/RVVMicrokernel/rvv-microkernel-i32m2-family-sub.mlir"
        ),
        "i32m2_selected_variant": "rvv_first_slice",
        "selected_variant": "rvv_sub_slice",
        "microkernel_op_name": "tcrv_rvv.i32_vsub_microkernel",
        "arithmetic_op_name": "tcrv_rvv.i32_sub",
        "intrinsic": "__riscv_vsub_vv_i32m1",
        "function_stem": "i32_vsub",
        "result_vec": "difference_vec",
        "c_operator": "-",
        "source_route": "tcrv-export-rvv-i32-vsub-microkernel-c",
        "header_route": "tcrv-export-rvv-i32-vsub-microkernel-header",
        "object_route": "tcrv-export-rvv-i32-vsub-microkernel-object",
        "emission_kind": "rvv-explicit-i32-vsub-microkernel-c-source",
        "runtime_abi": "rvv-i32-vsub-runtime-callable-c-abi.v1",
        "runtime_abi_kind": "rvv-runtime-callable-c-abi",
        "runtime_abi_name": "rvv-i32-vsub-runtime-callable-c-function.v1",
        "runtime_glue_role": "runtime-callable-i32-vsub-function",
        "component_group": "rvv-i32-vsub-microkernel-external-abi.v1",
        "external_abi_name": "rvv-i32-vsub-runtime-callable-c-function.v1",
        "self_check_success_marker": "tcrv_rvv_microkernel_ok",
        "external_abi_success_marker": "tcrv_rvv_i32_vsub_microkernel_external_abi_ok",
    },
    "i32-vmul": {
        "diagnostic_name": "i32-vmul",
        "default_input": Path("test/Target/RVVMicrokernel/rvv-microkernel-family-mul.mlir"),
        "default_vector_shape": "i32m1",
        "selected_variant": "rvv_mul_slice",
        "microkernel_op_name": "tcrv_rvv.i32_vmul_microkernel",
        "arithmetic_op_name": "tcrv_rvv.i32_mul",
        "intrinsic": "__riscv_vmul_vv_i32m1",
        "function_stem": "i32_vmul",
        "result_vec": "product_vec",
        "c_operator": "*",
        "source_route": "tcrv-export-rvv-i32-vmul-microkernel-c",
        "header_route": "tcrv-export-rvv-i32-vmul-microkernel-header",
        "object_route": "tcrv-export-rvv-i32-vmul-microkernel-object",
        "emission_kind": "rvv-explicit-i32-vmul-microkernel-c-source",
        "runtime_abi": "rvv-i32-vmul-runtime-callable-c-abi.v1",
        "runtime_abi_kind": "rvv-runtime-callable-c-abi",
        "runtime_abi_name": "rvv-i32-vmul-runtime-callable-c-function.v1",
        "runtime_glue_role": "runtime-callable-i32-vmul-function",
        "component_group": "rvv-i32-vmul-microkernel-external-abi.v1",
        "external_abi_name": "rvv-i32-vmul-runtime-callable-c-function.v1",
        "self_check_success_marker": "tcrv_rvv_microkernel_ok",
        "external_abi_success_marker": "tcrv_rvv_i32_vmul_microkernel_external_abi_ok",
    },
    "i64-vadd": {
        "diagnostic_name": "i64-vadd",
        "default_input": Path("test/Target/RVVMicrokernel/rvv-microkernel-i64-vadd.mlir"),
        "default_frontend_input": Path(
            "test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir"
        ),
        "default_vector_shape": "i64m1",
        "selected_variant": "rvv_i64_slice",
        "microkernel_op_name": "tcrv_rvv.i64_vadd_microkernel",
        "arithmetic_op_name": "tcrv_rvv.i64_add",
        "intrinsic": "__riscv_vadd_vv_i64m1",
        "function_stem": "i64_vadd",
        "result_vec": "sum_vec",
        "c_operator": "+",
        "source_route": "tcrv-export-rvv-i64-vadd-microkernel-c",
        "source_translation_route": "tcrv-export-rvv-microkernel-c",
        "header_route": "tcrv-export-rvv-i64-vadd-microkernel-header",
        "header_translation_route": "tcrv-export-rvv-microkernel-header",
        "object_route": "tcrv-export-rvv-i64-vadd-microkernel-object",
        "object_translation_route": "tcrv-export-rvv-microkernel-object",
        "emission_kind": "rvv-explicit-i64-vadd-microkernel-c-source",
        "runtime_abi": "rvv-i64-vadd-runtime-callable-c-abi.v1",
        "runtime_abi_kind": "rvv-runtime-callable-c-abi",
        "runtime_abi_name": "rvv-i64-vadd-runtime-callable-c-function.v1",
        "runtime_glue_role": "runtime-callable-i64-vadd-function",
        "component_group": "rvv-i64-vadd-microkernel-external-abi.v1",
        "external_abi_name": "rvv-i64-vadd-runtime-callable-c-function.v1",
        "self_check_success_marker": "tcrv_rvv_microkernel_ok",
        "external_abi_success_marker": "tcrv_rvv_i64_vadd_microkernel_external_abi_ok",
    },
    "i64-vsub": {
        "diagnostic_name": "i64-vsub",
        "default_input": Path("test/Target/RVVMicrokernel/rvv-microkernel-i64-vsub.mlir"),
        "default_frontend_input": Path(
            "test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir"
        ),
        "default_vector_shape": "i64m1",
        "selected_variant": "rvv_i64_slice",
        "microkernel_op_name": "tcrv_rvv.i64_vsub_microkernel",
        "arithmetic_op_name": "tcrv_rvv.i64_sub",
        "intrinsic": "__riscv_vsub_vv_i64m1",
        "function_stem": "i64_vsub",
        "result_vec": "difference_vec",
        "c_operator": "-",
        "source_route": "tcrv-export-rvv-i64-vsub-microkernel-c",
        "source_translation_route": "tcrv-export-rvv-microkernel-c",
        "header_route": "tcrv-export-rvv-i64-vsub-microkernel-header",
        "header_translation_route": "tcrv-export-rvv-microkernel-header",
        "object_route": "tcrv-export-rvv-i64-vsub-microkernel-object",
        "object_translation_route": "tcrv-export-rvv-microkernel-object",
        "emission_kind": "rvv-explicit-i64-vsub-microkernel-c-source",
        "runtime_abi": "rvv-i64-vsub-runtime-callable-c-abi.v1",
        "runtime_abi_kind": "rvv-runtime-callable-c-abi",
        "runtime_abi_name": "rvv-i64-vsub-runtime-callable-c-function.v1",
        "runtime_glue_role": "runtime-callable-i64-vsub-function",
        "component_group": "rvv-i64-vsub-microkernel-external-abi.v1",
        "external_abi_name": "rvv-i64-vsub-runtime-callable-c-function.v1",
        "self_check_success_marker": "tcrv_rvv_microkernel_ok",
        "external_abi_success_marker": "tcrv_rvv_i64_vsub_microkernel_external_abi_ok",
    },
    "i64-vmul": {
        "diagnostic_name": "i64-vmul",
        "default_input": Path("test/Target/RVVMicrokernel/rvv-microkernel-i64-vmul.mlir"),
        "default_frontend_input": Path(
            "test/Transforms/LinalgToExec/linalg-i64-vmul-to-rvv-artifact.mlir"
        ),
        "default_vector_shape": "i64m1",
        "selected_variant": "rvv_i64_slice",
        "microkernel_op_name": "tcrv_rvv.i64_vmul_microkernel",
        "arithmetic_op_name": "tcrv_rvv.i64_mul",
        "intrinsic": "__riscv_vmul_vv_i64m1",
        "function_stem": "i64_vmul",
        "result_vec": "product_vec",
        "c_operator": "*",
        "source_route": "tcrv-export-rvv-i64-vmul-microkernel-c",
        "source_translation_route": "tcrv-export-rvv-microkernel-c",
        "header_route": "tcrv-export-rvv-i64-vmul-microkernel-header",
        "header_translation_route": "tcrv-export-rvv-microkernel-header",
        "object_route": "tcrv-export-rvv-i64-vmul-microkernel-object",
        "object_translation_route": "tcrv-export-rvv-microkernel-object",
        "emission_kind": "rvv-explicit-i64-vmul-microkernel-c-source",
        "runtime_abi": "rvv-i64-vmul-runtime-callable-c-abi.v1",
        "runtime_abi_kind": "rvv-runtime-callable-c-abi",
        "runtime_abi_name": "rvv-i64-vmul-runtime-callable-c-function.v1",
        "runtime_glue_role": "runtime-callable-i64-vmul-function",
        "component_group": "rvv-i64-vmul-microkernel-external-abi.v1",
        "external_abi_name": "rvv-i64-vmul-runtime-callable-c-function.v1",
        "self_check_success_marker": "tcrv_rvv_microkernel_ok",
        "external_abi_success_marker": "tcrv_rvv_i64_vmul_microkernel_external_abi_ok",
    },
}

RVV_VECTOR_SHAPE_SPECS: dict[str, dict[str, Any]] = {
    "i32m1": {
        "shape": "i32m1",
        "dtype": "i32",
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
        "planning_pipeline": "tcrv-materialize-selected-lowering-boundaries",
    },
    "i32m2": {
        "shape": "i32m2",
        "dtype": "i32",
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
        "planning_pipeline": "tcrv-execution-planning-pipeline",
    },
    "i64m1": {
        "shape": "i64m1",
        "dtype": "i64",
        "sew_bits": 64,
        "lmul": "m1",
        "tail_policy": "agnostic",
        "mask_policy": "agnostic",
        "vector_type": "vint64m1_t",
        "vector_suffix": "i64m1",
        "setvl_suffix": "e64m1",
        "capability_ids": [
            "rvv.i64_m1.sew64",
            "rvv.i64_m1.lmul_m1",
            "rvv.i64_m1.tail_policy.agnostic",
            "rvv.i64_m1.mask_policy.agnostic",
        ],
        "planning_pipeline": "tcrv-materialize-selected-lowering-boundaries",
    },
}

ACTIVE_ARITHMETIC_FAMILY = ARITHMETIC_FAMILY_SPECS["i32-vadd"]
ACTIVE_VECTOR_SHAPE = RVV_VECTOR_SHAPE_SPECS["i32m1"]
SUCCESS_MARKER = str(ACTIVE_ARITHMETIC_FAMILY["self_check_success_marker"])
EXTERNAL_ABI_SUCCESS_MARKER = str(
    ACTIVE_ARITHMETIC_FAMILY["external_abi_success_marker"]
)
BUNDLE_EXTERNAL_ABI_COMPONENT_GROUP = str(ACTIVE_ARITHMETIC_FAMILY["component_group"])
BUNDLE_EXTERNAL_ABI_NAME = str(ACTIVE_ARITHMETIC_FAMILY["external_abi_name"])


def make_required_handoff(family: dict[str, str | Path]) -> dict[str, str]:
    return {
        "origin": "rvv-plugin",
        "emission_status": "supported",
        "emission_kind": str(family["emission_kind"]),
        "lowering_pipeline": str(family["source_route"]),
        "lowering_boundary": "tcrv_rvv.lowering_boundary",
        "runtime_abi": str(family["runtime_abi"]),
        "runtime_abi_kind": str(family["runtime_abi_kind"]),
        "runtime_abi_name": str(family["runtime_abi_name"]),
        "runtime_glue_role": str(family["runtime_glue_role"]),
        "artifact_kind": "runtime-callable-c-source",
    }


def direct_helper_routes_for_family(
    family: dict[str, str | Path],
) -> dict[str, str]:
    return {
        "source": str(family["source_route"]),
        "header": str(family["header_route"]),
        "object": str(family["object_route"]),
    }


def direct_helper_flag(route_id: str) -> str:
    return "--" + route_id


def family_dtype(family: dict[str, str | Path]) -> str:
    function_stem = str(family["function_stem"])
    dtype = function_stem.split("_", 1)[0]
    if dtype not in DTYPE_SCALAR_C_TYPES:
        raise BridgeError(f"unsupported RVV binary dtype in family: {function_stem}")
    return dtype


def family_scalar_c_type(family: dict[str, str | Path]) -> str:
    return DTYPE_SCALAR_C_TYPES[family_dtype(family)]


def runtime_abi_signature_for_family(
    family: dict[str, str | Path],
) -> list[dict[str, str]]:
    scalar_type = family_scalar_c_type(family)
    return [
        {
            "c_name": "lhs",
            "c_type": f"const {scalar_type} *",
            "role": "lhs-input-buffer",
            "ownership": "target-export-abi-owned",
        },
        {
            "c_name": "rhs",
            "c_type": f"const {scalar_type} *",
            "role": "rhs-input-buffer",
            "ownership": "target-export-abi-owned",
        },
        {
            "c_name": "out",
            "c_type": f"{scalar_type} *",
            "role": "output-buffer",
            "ownership": "target-export-abi-owned",
        },
        {
            "c_name": "n",
            "c_type": "size_t",
            "role": "runtime-element-count",
            "ownership": "target-export-abi-owned",
        },
    ]


def direct_helper_translation_route(
    family: dict[str, str | Path], artifact_role: str
) -> str:
    translation_key = artifact_role + "_translation_route"
    route_key = artifact_role + "_route"
    return str(family.get(translation_key, family[route_key]))


def arithmetic_intrinsic_for_family(
    family: dict[str, str | Path], shape: dict[str, Any]
) -> str:
    function_stem = str(family["function_stem"])
    match = re.match(r"^i[0-9]+_v([A-Za-z0-9]+)$", function_stem)
    if not match:
        raise BridgeError(f"unsupported RVV binary function stem: {function_stem}")
    return "__riscv_v" + match.group(1) + "_vv_" + str(shape["vector_suffix"])


def load_intrinsic_for_shape(shape: dict[str, Any]) -> str:
    return "__riscv_vle" + str(shape["sew_bits"]) + "_v_" + str(shape["vector_suffix"])


def store_intrinsic_for_shape(shape: dict[str, Any]) -> str:
    return "__riscv_vse" + str(shape["sew_bits"]) + "_v_" + str(shape["vector_suffix"])


def setvl_intrinsic_for_shape(shape: dict[str, Any]) -> str:
    return "__riscv_vsetvl_" + str(shape["setvl_suffix"])


def vector_shape_evidence(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape": str(shape["shape"]),
        "dtype": str(shape["dtype"]),
        "sew_bits": int(shape["sew_bits"]),
        "lmul": str(shape["lmul"]),
        "tail_policy": str(shape["tail_policy"]),
        "mask_policy": str(shape["mask_policy"]),
        "vector_type": str(shape["vector_type"]),
        "vector_suffix": str(shape["vector_suffix"]),
        "setvl_suffix": str(shape["setvl_suffix"]),
        "capability_ids": list(shape["capability_ids"]),
    }


def active_selected_variant() -> str:
    key = str(ACTIVE_VECTOR_SHAPE["shape"]) + "_selected_variant"
    return str(
        ACTIVE_ARITHMETIC_FAMILY.get(
            key, ACTIVE_ARITHMETIC_FAMILY["selected_variant"]
        )
    )


def command_name_for_route(route_id: str) -> str:
    name = route_id
    if name.startswith("tcrv-export-"):
        name = name.removeprefix("tcrv-export-")
    return "export_" + safe_file_component(name).replace("-", "_")


def make_rvv_bundle_routes(
    family: dict[str, str | Path],
) -> dict[str, dict[str, str]]:
    component_group = str(family["component_group"])
    external_abi_name = str(family["external_abi_name"])
    runtime_abi_kind = str(family["runtime_abi_kind"])
    return {
        "source": {
            "route": str(family["source_route"]),
            "artifact_kind": "runtime-callable-c-source",
            "component_group": component_group,
            "component_role": "source",
            "external_abi_name": external_abi_name,
            "owner": "rvv-plugin",
            "runtime_abi_kind": runtime_abi_kind,
            "runtime_abi_name": external_abi_name,
            "evidence_role": "compiler-artifact",
        },
        "header": {
            "route": str(family["header_route"]),
            "artifact_kind": "runtime-callable-c-header",
            "component_group": component_group,
            "component_role": "header",
            "external_abi_name": external_abi_name,
            "owner": "rvv-plugin",
            "runtime_abi_kind": runtime_abi_kind,
            "runtime_abi_name": external_abi_name,
            "evidence_role": "header-declaration",
        },
        "object": {
            "route": str(family["object_route"]),
            "artifact_kind": "riscv-elf-relocatable-object",
            "component_group": component_group,
            "component_role": "object",
            "external_abi_name": external_abi_name,
            "owner": "rvv-plugin",
            "runtime_abi_kind": runtime_abi_kind,
            "runtime_abi_name": external_abi_name,
            "evidence_role": "relocatable-object",
        },
    }


RVV_BUNDLE_ROUTES = make_rvv_bundle_routes(ACTIVE_ARITHMETIC_FAMILY)


def make_expected_dataflow_provenance(
    family: dict[str, str | Path],
) -> dict[str, list[str]]:
    result_vec = str(family["result_vec"])
    dtype = family_dtype(family)
    return {
        "dataflow_abi_roles": [
            "lhs_load.buffer_role=lhs-input-buffer",
            "rhs_load.buffer_role=rhs-input-buffer",
            "store.buffer_role=output-buffer",
            "runtime n remains the target/export-owned runtime element-count ABI parameter",
        ],
        "dataflow_emission_step[0]": [
            "op=tcrv_rvv." + dtype + "_load",
            "role=lhs-input-buffer",
            "result=lhs_vec",
        ],
        "dataflow_emission_step[1]": [
            "op=tcrv_rvv." + dtype + "_load",
            "role=rhs-input-buffer",
            "result=rhs_vec",
        ],
        "dataflow_emission_step[2]": [
            "op=" + str(family["arithmetic_op_name"]),
            "lhs=lhs_vec",
            "rhs=rhs_vec",
            "result=" + result_vec,
        ],
        "dataflow_emission_step[3]": [
            "op=tcrv_rvv." + dtype + "_store",
            "role=output-buffer",
            "value=" + result_vec,
        ],
    }


def configure_arithmetic_family(family_name: str) -> None:
    global ACTIVE_ARITHMETIC_FAMILY
    global SUCCESS_MARKER
    global EXTERNAL_ABI_SUCCESS_MARKER
    global BUNDLE_EXTERNAL_ABI_COMPONENT_GROUP
    global BUNDLE_EXTERNAL_ABI_NAME
    global RVV_BUNDLE_ROUTES

    family = ARITHMETIC_FAMILY_SPECS.get(family_name)
    if family is None:
        raise BridgeError(f"unsupported arithmetic family: {family_name}")
    ACTIVE_ARITHMETIC_FAMILY = family
    SUCCESS_MARKER = str(family["self_check_success_marker"])
    EXTERNAL_ABI_SUCCESS_MARKER = str(family["external_abi_success_marker"])
    BUNDLE_EXTERNAL_ABI_COMPONENT_GROUP = str(family["component_group"])
    BUNDLE_EXTERNAL_ABI_NAME = str(family["external_abi_name"])
    RVV_BUNDLE_ROUTES = make_rvv_bundle_routes(family)


def configure_vector_shape(shape_name: str) -> None:
    global ACTIVE_VECTOR_SHAPE

    shape = RVV_VECTOR_SHAPE_SPECS.get(shape_name)
    if shape is None:
        raise BridgeError(f"unsupported RVV vector shape: {shape_name}")
    if str(shape["dtype"]) != family_dtype(ACTIVE_ARITHMETIC_FAMILY):
        raise BridgeError(
            "RVV vector shape "
            + shape_name
            + " has dtype "
            + str(shape["dtype"])
            + " but arithmetic family "
            + str(ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"])
            + " requires dtype "
            + family_dtype(ACTIVE_ARITHMETIC_FAMILY)
        )
    ACTIVE_VECTOR_SHAPE = shape

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


def current_git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    commit = sanitize_text(result.stdout.strip())
    if not re.match(r"^[0-9a-f]{40}$", commit):
        raise BridgeError("git rev-parse HEAD returned an invalid commit hash")
    return commit


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
    required_handoff = make_required_handoff(ACTIVE_ARITHMETIC_FAMILY)
    for path in parse_manifest_paths(manifest_text):
        if all(path.get(key) == value for key, value in required_handoff.items()):
            matches.append(path)

    if not matches:
        for path in parse_manifest_paths(manifest_text):
            for family_name, family in ARITHMETIC_FAMILY_SPECS.items():
                if family is ACTIVE_ARITHMETIC_FAMILY:
                    continue
                other_handoff = make_required_handoff(family)
                if all(path.get(key) == value for key, value in other_handoff.items()):
                    raise BridgeError(
                        "manifest contains stale RVV microkernel handoff for "
                        f"{family_name}; expected "
                        f"{ACTIVE_ARITHMETIC_FAMILY['diagnostic_name']}"
                    )
        raise BridgeError(
            "manifest missing supported bounded RVV "
            f"{ACTIVE_ARITHMETIC_FAMILY['diagnostic_name']} microkernel "
            "source-export handoff"
        )
    if len(matches) != 1:
        raise BridgeError(
            "manifest must contain exactly one supported bounded RVV "
            f"{ACTIVE_ARITHMETIC_FAMILY['diagnostic_name']} microkernel handoff"
        )
    return matches[0]


def parse_source_comment(source: str, field: str, *, required: bool) -> str:
    match = re.search(rf"/\*\s*{re.escape(field)}:\s*(.*?)\s*\*/", source)
    if not match:
        if required:
            raise BridgeError(f"generated C source missing comment field: {field}")
        return ""
    value = match.group(1).strip()
    reject_secret_like_text(f"generated C source field {field}", value)
    return value


def parse_runtime_abi_parameters_from_source(source: str) -> list[dict[str, str]]:
    parameters_by_index: dict[int, dict[str, str]] = {}
    pattern = re.compile(
        r"/\*\s*runtime_abi_parameter\[([0-9]+)\]:\s*"
        r"c_name=([^,]+),\s*c_type=(.*?),\s*role=([^,]+),\s*"
        r"ownership=([^*]+?)\s*\*/"
    )
    for match in pattern.finditer(source):
        index = int(match.group(1))
        parameter = {
            "c_name": match.group(2).strip(),
            "c_type": match.group(3).strip(),
            "role": match.group(4).strip(),
            "ownership": match.group(5).strip(),
        }
        for field, value in parameter.items():
            if not value:
                raise BridgeError(
                    f"generated C source runtime_abi_parameter[{index}] missing {field}"
                )
            reject_secret_like_text(
                f"generated C source runtime_abi_parameter[{index}] {field}",
                value,
            )
        if index in parameters_by_index:
            raise BridgeError(
                f"generated C source duplicates runtime_abi_parameter[{index}]"
            )
        parameters_by_index[index] = parameter

    if not parameters_by_index:
        raise BridgeError(
            "generated C source missing runtime_abi_parameter metadata"
        )

    parameters: list[dict[str, str]] = []
    for index in range(len(parameters_by_index)):
        parameter = parameters_by_index.get(index)
        if parameter is None:
            raise BridgeError(
                f"generated C source missing runtime_abi_parameter[{index}]"
            )
        parameters.append(parameter)
    return parameters


def validate_runtime_abi_signature(
    observed: list[dict[str, str]], family: dict[str, str | Path]
) -> list[dict[str, str]]:
    expected = runtime_abi_signature_for_family(family)
    if observed != expected:
        raise BridgeError(
            "generated C source runtime ABI signature does not match the "
            "selected RVV "
            + str(family["diagnostic_name"])
            + " compiler-emitted descriptor contract"
        )
    return observed


def normalize_symbol_name(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("@"):
        normalized = normalized[1:]
    return normalized


def validate_compiler_path_context(source: str) -> dict[str, str]:
    selected_kernel = normalize_symbol_name(
        parse_source_comment(source, "selected_kernel", required=True)
    )
    selected_variant = normalize_symbol_name(
        parse_source_comment(source, "selected_variant", required=True)
    )
    context = {
        "microkernel_function": parse_source_comment(
            source, "microkernel function", required=True
        ),
        "selected_kernel": selected_kernel,
        "selected_variant": selected_variant,
        "selected_role": parse_source_comment(source, "selected_role", required=True),
        "lowering_boundary": parse_source_comment(
            source, "lowering_boundary", required=True
        ),
        "active_route": parse_source_comment(source, "active_route", required=True),
        "callable_abi_source": parse_source_comment(
            source, "callable_abi_source", required=True
        ),
    }
    for key, value in context.items():
        if not value:
            raise BridgeError(f"generated C source compiler path field {key} is empty")
    return context


def validate_expected_selected_kernel(
    compiler_path_context: dict[str, str], expected_selected_kernel: str
) -> str:
    expected = normalize_symbol_name(expected_selected_kernel)
    if not expected:
        return ""
    reject_secret_like_text("expected selected kernel", expected)
    observed = compiler_path_context.get("selected_kernel", "")
    if observed != expected:
        raise BridgeError(
            "generated RVV microkernel source selected_kernel "
            f"@{observed} does not match expected @{expected}"
        )
    return expected


def validate_generated_source(source: str, *, require_harness: bool) -> dict[str, Any]:
    if not source.strip():
        raise BridgeError("generated RVV microkernel C source is empty")
    reject_secret_like_text("generated RVV microkernel C source", source)
    family_name = str(ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"])
    required_snippets = [
        "#include <riscv_vector.h>",
        setvl_intrinsic_for_shape(ACTIVE_VECTOR_SHAPE),
        load_intrinsic_for_shape(ACTIVE_VECTOR_SHAPE),
        arithmetic_intrinsic_for_family(
            ACTIVE_ARITHMETIC_FAMILY, ACTIVE_VECTOR_SHAPE
        ),
        store_intrinsic_for_shape(ACTIVE_VECTOR_SHAPE),
        "/* executable_microkernel: "
        + str(ACTIVE_ARITHMETIC_FAMILY["microkernel_op_name"])
        + " */",
        "/* arithmetic_c_operator: " + str(ACTIVE_ARITHMETIC_FAMILY["c_operator"]),
        "/* selected_vector_shape_config:",
        "/* selected_vector_shape_capabilities:",
        "/* dataflow_body: tcrv_rvv." + str(ACTIVE_VECTOR_SHAPE["dtype"]) + "_load -> tcrv_rvv." + str(ACTIVE_VECTOR_SHAPE["dtype"]) + "_load -> "
        + str(ACTIVE_ARITHMETIC_FAMILY["arithmetic_op_name"])
        + " -> tcrv_rvv." + str(ACTIVE_VECTOR_SHAPE["dtype"]) + "_store */",
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
    for other_name, other_shape in RVV_VECTOR_SHAPE_SPECS.items():
        if other_shape is ACTIVE_VECTOR_SHAPE:
            continue
        stale_snippets = [
            setvl_intrinsic_for_shape(other_shape),
            load_intrinsic_for_shape(other_shape),
            arithmetic_intrinsic_for_family(ACTIVE_ARITHMETIC_FAMILY, other_shape),
            store_intrinsic_for_shape(other_shape),
            "selected_vector_shape_config: shape="
            + str(other_shape["shape"]),
            "vector_type=" + str(other_shape["vector_type"]),
            "vector_suffix=" + str(other_shape["vector_suffix"]),
            "setvl_suffix=" + str(other_shape["setvl_suffix"]),
        ]
        leaked = [snippet for snippet in stale_snippets if snippet in source]
        if leaked:
            raise BridgeError(
                "generated RVV "
                + family_name
                + " microkernel C source contains stale "
                + other_name
                + " vector-shape metadata: "
                + ", ".join(leaked)
            )
    missing = [snippet for snippet in required_snippets if snippet not in source]
    if missing:
        raise BridgeError(
            "generated RVV "
            + family_name
            + " microkernel C source missing required snippets: "
            + ", ".join(missing)
        )
    for other_name, other_family in ARITHMETIC_FAMILY_SPECS.items():
        if other_family is ACTIVE_ARITHMETIC_FAMILY:
            continue
        other_dtype = family_dtype(other_family)
        stale_snippets = [
            arithmetic_intrinsic_for_family(other_family, ACTIVE_VECTOR_SHAPE),
            "executable_microkernel: " + str(other_family["microkernel_op_name"]),
            "arithmetic_family: " + str(other_family["diagnostic_name"]),
            "op=" + str(other_family["arithmetic_op_name"]),
            str(other_family["runtime_abi_name"]),
            str(other_family["runtime_glue_role"]),
            "dataflow_body: tcrv_rvv." + other_dtype + "_load -> tcrv_rvv." + other_dtype + "_load -> "
            + str(other_family["arithmetic_op_name"]) + " -> tcrv_rvv." + other_dtype + "_store",
        ]
        active_snippets = {
            arithmetic_intrinsic_for_family(
                ACTIVE_ARITHMETIC_FAMILY, ACTIVE_VECTOR_SHAPE
            )
        }
        stale_snippets = [
            snippet for snippet in stale_snippets if snippet not in active_snippets
        ]
        leaked = [snippet for snippet in stale_snippets if snippet in source]
        if leaked:
            raise BridgeError(
                "generated RVV "
                + family_name
                + " microkernel C source contains stale "
                + other_name
                + " metadata: "
                + ", ".join(leaked)
            )
    selected_march = parse_source_comment(source, "selected_march", required=True)
    selected_mabi = parse_source_comment(source, "selected_mabi", required=False)
    arithmetic_operator = parse_source_comment(
        source, "arithmetic_c_operator", required=True
    )
    if arithmetic_operator != str(ACTIVE_ARITHMETIC_FAMILY["c_operator"]):
        raise BridgeError(
            "generated RVV microkernel C source arithmetic_c_operator "
            f"{arithmetic_operator!r} does not match selected family "
            f"{ACTIVE_ARITHMETIC_FAMILY['diagnostic_name']}"
        )
    if "v" not in selected_march.lower():
        raise BridgeError("selected_march from generated source must contain RVV vector evidence")
    vector_config = validate_vector_shape_metadata(source)
    provenance = validate_dataflow_provenance(source)
    compiler_path_context = validate_compiler_path_context(source)
    runtime_abi_parameters = validate_runtime_abi_signature(
        parse_runtime_abi_parameters_from_source(source), ACTIVE_ARITHMETIC_FAMILY
    )
    return {
        "selected_march": selected_march,
        "selected_mabi": selected_mabi,
        "arithmetic_operator": arithmetic_operator,
        "vector_config": vector_config,
        "dataflow_provenance": provenance,
        "compiler_path_context": compiler_path_context,
        "runtime_abi_parameters": runtime_abi_parameters,
    }


def validate_vector_shape_metadata(source: str) -> dict[str, Any]:
    selected_shape_config = parse_source_comment(
        source, "selected_vector_shape_config", required=True
    )
    selected_shape_capabilities = parse_source_comment(
        source, "selected_vector_shape_capabilities", required=True
    )
    control_config = parse_source_comment(
        source, "control_plane_config", required=True
    )
    intrinsic_config = parse_source_comment(
        source, "intrinsic_config", required=True
    )
    expected_control_fragments = [
        "sew=" + str(ACTIVE_VECTOR_SHAPE["sew_bits"]),
        "lmul=" + str(ACTIVE_VECTOR_SHAPE["lmul"]),
        "tail = " + str(ACTIVE_VECTOR_SHAPE["tail_policy"]),
        "mask = " + str(ACTIVE_VECTOR_SHAPE["mask_policy"]),
    ]
    missing_control = [
        fragment for fragment in expected_control_fragments if fragment not in control_config
    ]
    if missing_control:
        raise BridgeError(
            "generated RVV microkernel C source control_plane_config does not "
            "match requested vector shape "
            + str(ACTIVE_VECTOR_SHAPE["shape"])
            + "; missing fragments: "
            + ", ".join(missing_control)
        )

    expected_selected_shape_fragments = [
        "shape=" + str(ACTIVE_VECTOR_SHAPE["shape"]),
        "sew=" + str(ACTIVE_VECTOR_SHAPE["sew_bits"]),
        "lmul=" + str(ACTIVE_VECTOR_SHAPE["lmul"]),
        "tail_policy=" + str(ACTIVE_VECTOR_SHAPE["tail_policy"]),
        "mask_policy=" + str(ACTIVE_VECTOR_SHAPE["mask_policy"]),
        "vector_type=" + str(ACTIVE_VECTOR_SHAPE["vector_type"]),
        "vector_suffix=" + str(ACTIVE_VECTOR_SHAPE["vector_suffix"]),
        "setvl_suffix=" + str(ACTIVE_VECTOR_SHAPE["setvl_suffix"]),
    ]
    if str(ACTIVE_VECTOR_SHAPE["dtype"]) != "i32":
        expected_selected_shape_fragments.insert(
            0, "dtype=" + str(ACTIVE_VECTOR_SHAPE["dtype"])
        )
    missing_selected_shape = [
        fragment
        for fragment in expected_selected_shape_fragments
        if fragment not in selected_shape_config
    ]
    if missing_selected_shape:
        raise BridgeError(
            "generated RVV microkernel C source selected_vector_shape_config "
            "does not match requested vector shape "
            + str(ACTIVE_VECTOR_SHAPE["shape"])
            + "; missing fragments: "
            + ", ".join(missing_selected_shape)
        )

    missing_capabilities = [
        str(capability)
        for capability in ACTIVE_VECTOR_SHAPE["capability_ids"]
        if str(capability) not in selected_shape_capabilities
    ]
    if missing_capabilities:
        raise BridgeError(
            "generated RVV microkernel C source selected_vector_shape_capabilities "
            "does not contain requested vector-shape capability ids: "
            + ", ".join(missing_capabilities)
        )

    expected_intrinsic_fragments = [
        "vector_type=" + str(ACTIVE_VECTOR_SHAPE["vector_type"]),
        "vector_suffix=" + str(ACTIVE_VECTOR_SHAPE["vector_suffix"]),
        "setvl_suffix=" + str(ACTIVE_VECTOR_SHAPE["setvl_suffix"]),
        "tail_policy=" + str(ACTIVE_VECTOR_SHAPE["tail_policy"]),
        "mask_policy=" + str(ACTIVE_VECTOR_SHAPE["mask_policy"]),
    ]
    missing_intrinsic = [
        fragment
        for fragment in expected_intrinsic_fragments
        if fragment not in intrinsic_config
    ]
    if missing_intrinsic:
        raise BridgeError(
            "generated RVV microkernel C source intrinsic_config does not "
            "match requested vector shape "
            + str(ACTIVE_VECTOR_SHAPE["shape"])
            + "; missing fragments: "
            + ", ".join(missing_intrinsic)
        )

    evidence = vector_shape_evidence(ACTIVE_VECTOR_SHAPE)
    evidence["selected_vector_shape_config"] = selected_shape_config
    evidence["selected_vector_shape_capabilities"] = selected_shape_capabilities
    evidence["control_plane_config"] = control_config
    evidence["intrinsic_config"] = intrinsic_config
    evidence["required_intrinsics"] = [
        setvl_intrinsic_for_shape(ACTIVE_VECTOR_SHAPE),
        load_intrinsic_for_shape(ACTIVE_VECTOR_SHAPE),
        arithmetic_intrinsic_for_family(
            ACTIVE_ARITHMETIC_FAMILY, ACTIVE_VECTOR_SHAPE
        ),
        store_intrinsic_for_shape(ACTIVE_VECTOR_SHAPE),
    ]
    return evidence


def validate_dataflow_provenance(source: str) -> dict[str, str]:
    provenance: dict[str, str] = {}
    for field, required_fragments in make_expected_dataflow_provenance(
        ACTIVE_ARITHMETIC_FAMILY
    ).items():
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


def normalize_c_parameter_list(parameter_text: str) -> str:
    normalized = " ".join(parameter_text.strip().split())
    normalized = re.sub(r"\s*\*\s*", "*", normalized)
    normalized = re.sub(r"\s*,\s*", ",", normalized)
    return normalized


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
        r"(?m)^\s*void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;{}]*)\)\s*;\s*$",
        header,
    )
    if len(prototypes) != 1:
        raise BridgeError(
            "generated RVV microkernel C header must contain exactly one "
            f"runtime-callable {ACTIVE_ARITHMETIC_FAMILY['diagnostic_name']} prototype"
        )
    function_name, parameter_text = prototypes[0]
    if str(ACTIVE_ARITHMETIC_FAMILY["function_stem"]) not in function_name:
        raise BridgeError(
            "generated RVV microkernel C header prototype does not match "
            f"selected arithmetic family {ACTIVE_ARITHMETIC_FAMILY['diagnostic_name']}"
        )
    expected_parameters = []
    for parameter in runtime_abi_signature_for_family(ACTIVE_ARITHMETIC_FAMILY):
        expected_parameters.append(parameter["c_type"] + " " + parameter["c_name"])
    observed = normalize_c_parameter_list(parameter_text)
    expected = normalize_c_parameter_list(", ".join(expected_parameters))
    if observed != expected:
        raise BridgeError(
            "generated RVV microkernel C header prototype ABI does not match "
            f"selected arithmetic family {ACTIVE_ARITHMETIC_FAMILY['diagnostic_name']}"
        )
    if re.search(r"(?m)^\s*void\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*\)\s*\{", header):
        raise BridgeError("generated RVV microkernel C header must not contain a function body")
    return function_name


def validate_bundle_file_name(file_name: str) -> None:
    if not file_name:
        raise BridgeError("bundle index artifact file_name must be non-empty")
    reject_secret_like_text("bundle index artifact file_name", file_name)
    if Path(file_name).name != file_name or "/" in file_name or "\\" in file_name:
        raise BridgeError(
            f"bundle index artifact file_name must be a plain file name: {file_name}"
        )


def c_scalar_type_from_abi_type(c_type: str) -> str:
    normalized = " ".join(c_type.strip().split())
    normalized = normalized.removeprefix("const ").strip()
    normalized = normalized.removesuffix("*").strip()
    if normalized not in DTYPE_SCALAR_C_TYPES.values():
        raise BridgeError(f"unsupported RVV caller scalar type: {c_type}")
    return normalized


def build_external_caller_source(
    function_name: str,
    header_file_name: str = "rvv_microkernel.h",
    runtime_abi_parameters: list[dict[str, str]] | None = None,
    arithmetic_operator: str | None = None,
    runtime_counts: list[int] | None = None,
) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", function_name):
        raise BridgeError("generated header function name is not a valid C identifier")
    validate_bundle_file_name(header_file_name)
    parameters = runtime_abi_parameters or runtime_abi_signature_for_family(
        ACTIVE_ARITHMETIC_FAMILY
    )
    if parameters != runtime_abi_signature_for_family(ACTIVE_ARITHMETIC_FAMILY):
        validate_runtime_abi_signature(parameters, ACTIVE_ARITHMETIC_FAMILY)
    scalar_c_type = c_scalar_type_from_abi_type(parameters[0]["c_type"])
    c_operator = arithmetic_operator or str(ACTIVE_ARITHMETIC_FAMILY["c_operator"])
    if c_operator not in {"+", "-", "*"}:
        raise BridgeError(f"unsupported arithmetic operator: {c_operator}")
    counts = runtime_counts or DIRECT_EXTERNAL_RUNTIME_COUNTS
    if len(counts) < 2:
        raise BridgeError("external caller evidence requires at least two runtime counts")
    escaped_header = header_file_name.replace("\\", "\\\\").replace('"', '\\"')
    family_name = str(ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"])
    runtime_counts_list = ", ".join(str(count) for count in counts)
    runtime_counts_csv = ",".join(str(count) for count in counts)
    return f"""\
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "{escaped_header}"

int main(void) {{
  enum {{ kElements = 16 }};
  const {scalar_c_type} lhs[kElements] = {{0, 1, 2, 3, 4, 5, 6, 7,
                                           8, 9, 10, 11, 12, 13, 14, 15}};
  const {scalar_c_type} rhs[kElements] = {{31, 29, 23, 19, 17, 13, 11, 7,
                                           5, 3, 2, 1, -1, -3, -5, -7}};
  {scalar_c_type} out[kElements] = {{0}};
  const size_t runtime_counts[] = {{{runtime_counts_list}}};

  for (size_t runtime_index = 0; runtime_index < sizeof(runtime_counts) / sizeof(runtime_counts[0]); ++runtime_index) {{
    size_t runtime_n = runtime_counts[runtime_index];
    for (size_t index = 0; index < (size_t)kElements; ++index)
      out[index] = 0;

    {function_name}(lhs, rhs, out, runtime_n);
    for (size_t index = 0; index < runtime_n; ++index) {{
      if (out[index] != lhs[index] {c_operator} rhs[index]) {{
        fprintf(stderr, "rvv {family_name} microkernel external ABI mismatch at %zu for n=%zu\\n", index, runtime_n);
        return 3;
      }}
    }}
  }}

  printf("{EXTERNAL_ABI_SUCCESS_MARKER} counts={runtime_counts_csv}\\n");
  return 0;
}}
"""


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
        "component_group",
        "component_role",
        "external_abi_name",
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


def require_rvv_runtime_abi_signature(
    record: dict[str, Any],
) -> list[dict[str, str]]:
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

    if parameters != runtime_abi_signature_for_family(ACTIVE_ARITHMETIC_FAMILY):
        raise BridgeError(
            f"bundle record route {record.get('route')} runtime ABI signature "
            "does not match the RVV "
            f"{ACTIVE_ARITHMETIC_FAMILY['diagnostic_name']} callable ABI"
        )
    return parameters


def require_rvv_component_metadata(record: dict[str, Any]) -> None:
    expected_variant = active_selected_variant()
    if record.get("selected_variant") != expected_variant:
        raise BridgeError(
            f"bundle record route {record.get('route')} must preserve "
            f"selected_variant @{expected_variant}"
        )
    if record.get("role") != "direct variant":
        raise BridgeError(
            f"bundle record route {record.get('route')} must preserve direct variant role"
        )
    components = record.get("components")
    if not isinstance(components, list) or len(components) != 1:
        raise BridgeError(
            f"bundle record route {record.get('route')} must preserve exactly one selected component"
        )
    component = components[0]
    if component.get("selected_variant") != expected_variant:
        raise BridgeError(
            f"bundle record route {record.get('route')} component must "
            f"preserve {expected_variant}"
        )
    if component.get("role") != "direct variant":
        raise BridgeError(
            f"bundle record route {record.get('route')} component must preserve direct variant role"
        )


def select_rvv_bundle_records(
    records: list[dict[str, Any]], bundle_dir: Path
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for label, expected in RVV_BUNDLE_ROUTES.items():
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
        require_rvv_component_metadata(record)
        require_rvv_runtime_abi_signature(record)
        artifact_path = bundle_dir / str(record["file_name"])
        if not artifact_path.exists() or artifact_path.stat().st_size == 0:
            raise BridgeError(
                f"bundle {label} artifact is missing or empty: {record['file_name']}"
            )
        selected[label] = record

    signatures = {
        json.dumps(record.get("runtime_abi_parameters", []), sort_keys=True)
        for record in selected.values()
    }
    if len(signatures) != 1:
        raise BridgeError(
            "selected RVV bundle records must share the compiler-emitted runtime ABI parameter signature"
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
                "selected_variant": record.get("selected_variant", ""),
                "role": record.get("role", ""),
                "components": record.get("components", []),
                "selected_plan_metadata": record.get("selected_plan_metadata", []),
            }
        )
    return summary


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


def remote_compile_flags(flags: dict[str, Any]) -> list[str]:
    result = ["-O2", f"-march={flags['selected_march']}"]
    if flags.get("selected_mabi"):
        result.append(f"-mabi={flags['selected_mabi']}")
    return result


def build_remote_compile_command(remote_dir: str, flags: dict[str, Any]) -> str:
    quoted_flags = " ".join(shlex.quote(part) for part in remote_compile_flags(flags))
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        f"clang {quoted_flags} rvv_microkernel.c -o rvv_microkernel"
    )


def build_remote_external_link_command(remote_dir: str, flags: dict[str, Any]) -> str:
    quoted_flags = " ".join(shlex.quote(part) for part in remote_compile_flags(flags))
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        f"clang {quoted_flags} rvv_microkernel_external_caller.c "
        "rvv_microkernel.o -o rvv_microkernel_external_caller"
    )


def build_remote_bundle_compile_caller_object_command(
    remote_dir: str, flags: dict[str, Any]
) -> str:
    quoted_flags = " ".join(shlex.quote(part) for part in remote_compile_flags(flags))
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        f"clang {quoted_flags} -c rvv_microkernel_external_caller.c "
        "-o rvv_microkernel_external_caller.o"
    )


def build_remote_bundle_compile_source_object_command(
    remote_dir: str, source_file_name: str, flags: dict[str, Any]
) -> str:
    validate_bundle_file_name(source_file_name)
    quoted_flags = " ".join(shlex.quote(part) for part in remote_compile_flags(flags))
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        f"clang {quoted_flags} -c {shlex.quote(source_file_name)} "
        "-o rvv_microkernel_from_source.o"
    )


def build_remote_bundle_link_executable_command(
    remote_dir: str,
    object_file_name: str,
    executable_file_name: str,
    flags: dict[str, Any],
) -> str:
    validate_bundle_file_name(object_file_name)
    validate_bundle_file_name(executable_file_name)
    quoted_flags = " ".join(shlex.quote(part) for part in remote_compile_flags(flags))
    return (
        f"cd {quote_remote_path(remote_dir)} && "
        f"clang {quoted_flags} rvv_microkernel_external_caller.o "
        f"{shlex.quote(object_file_name)} -o {shlex.quote(executable_file_name)}"
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
    flags: dict[str, Any],
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
    source_path: Path,
    header_path: Path,
    object_path: Path,
    caller_path: Path,
    flags: dict[str, Any],
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
            str(source_path),
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

    run_command(
        "ssh_compile_external_caller_object",
        remote_shell_command(
            args, build_remote_bundle_compile_caller_object_command(remote_dir, flags)
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    caller_object_hash_stdout, _, _ = run_command(
        "ssh_external_caller_object_sha256",
        remote_shell_command(
            args, remote_sha256_command(remote_dir, "rvv_microkernel_external_caller.o")
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_command(
        "ssh_compile_external_source_object",
        remote_shell_command(
            args,
            build_remote_bundle_compile_source_object_command(
                remote_dir, "rvv_microkernel.c", flags
            ),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    source_object_hash_stdout, _, _ = run_command(
        "ssh_external_source_object_sha256",
        remote_shell_command(
            args, remote_sha256_command(remote_dir, "rvv_microkernel_from_source.o")
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_command(
        "ssh_link_external_source_caller",
        remote_shell_command(
            args,
            build_remote_bundle_link_executable_command(
                remote_dir,
                "rvv_microkernel_from_source.o",
                "rvv_microkernel_external_caller_from_source",
                flags,
            ),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    source_executable_hash_stdout, _, _ = run_command(
        "ssh_external_source_executable_sha256",
        remote_shell_command(
            args,
            remote_sha256_command(
                remote_dir, "rvv_microkernel_external_caller_from_source"
            ),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    source_run_stdout, _, _ = run_command(
        "ssh_run_external_source_caller",
        remote_shell_command(
            args,
            f"cd {quote_remote_path(remote_dir)} && "
            "./rvv_microkernel_external_caller_from_source",
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    if EXTERNAL_ABI_SUCCESS_MARKER not in source_run_stdout:
        raise BridgeError(
            "remote source-built external caller stdout missing expected marker: "
            + EXTERNAL_ABI_SUCCESS_MARKER
        )

    run_command(
        "ssh_link_external_object_caller",
        remote_shell_command(
            args,
            build_remote_bundle_link_executable_command(
                remote_dir,
                "rvv_microkernel.o",
                "rvv_microkernel_external_caller_from_object",
                flags,
            ),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    object_executable_hash_stdout, _, _ = run_command(
        "ssh_external_object_executable_sha256",
        remote_shell_command(
            args,
            remote_sha256_command(
                remote_dir, "rvv_microkernel_external_caller_from_object"
            ),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    object_run_stdout, _, _ = run_command(
        "ssh_run_external_object_caller",
        remote_shell_command(
            args,
            f"cd {quote_remote_path(remote_dir)} && "
            "./rvv_microkernel_external_caller_from_object",
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    if EXTERNAL_ABI_SUCCESS_MARKER not in object_run_stdout:
        raise BridgeError(
            "remote generated-object external caller stdout missing expected marker: "
            + EXTERNAL_ABI_SUCCESS_MARKER
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
        "remote_artifacts": {
            "source_file": "rvv_microkernel.c",
            "header_file": header_path.name,
            "object_file": object_path.name,
            "caller_file": caller_path.name,
            "caller_object_file": "rvv_microkernel_external_caller.o",
            "source_built_object_file": "rvv_microkernel_from_source.o",
            "source_built_executable": "rvv_microkernel_external_caller_from_source",
            "object_executable": "rvv_microkernel_external_caller_from_object",
        },
        "compile_flags": remote_compile_flags(flags),
        "caller_object_compile_exit_code": 0,
        "source_object_compile_exit_code": 0,
        "source_link_exit_code": 0,
        "source_run_exit_code": 0,
        "object_link_exit_code": 0,
        "object_run_exit_code": 0,
        "expected_stdout_marker": EXTERNAL_ABI_SUCCESS_MARKER,
        "source_stdout_marker_observed": True,
        "object_stdout_marker_observed": True,
        "caller_object_sha256": first_sanitized_line(caller_object_hash_stdout),
        "source_built_object_sha256": first_sanitized_line(source_object_hash_stdout),
        "source_built_executable_sha256": first_sanitized_line(
            source_executable_hash_stdout
        ),
        "object_executable_sha256": first_sanitized_line(
            object_executable_hash_stdout
        ),
        "cleanup_status": cleanup_status,
    }


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
    flags: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    reject_secret_like_text("ssh target", args.ssh_target)
    remote_dir = f"/tmp/tianchenrv_rvv_microkernel_bundle_{safe_run_id(run_id)}"

    setup_command = f"rm -rf {quote_remote_path(remote_dir)} && mkdir -p {quote_remote_path(remote_dir)}"
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
            args, remote_sha256_command(remote_dir, "rvv_microkernel_external_caller.o")
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )

    run_command(
        "ssh_compile_bundle_source_object",
        remote_shell_command(
            args,
            build_remote_bundle_compile_source_object_command(
                remote_dir, source_file_name, flags
            ),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    source_object_hash_stdout, _, _ = run_command(
        "ssh_bundle_source_object_sha256",
        remote_shell_command(
            args, remote_sha256_command(remote_dir, "rvv_microkernel_from_source.o")
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
                "rvv_microkernel_from_source.o",
                "rvv_microkernel_external_caller_from_source",
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
                remote_dir, "rvv_microkernel_external_caller_from_source"
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
            "./rvv_microkernel_external_caller_from_source",
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    if EXTERNAL_ABI_SUCCESS_MARKER not in source_run_stdout:
        raise BridgeError(
            "remote bundle source-built external caller stdout missing expected marker: "
            + EXTERNAL_ABI_SUCCESS_MARKER
        )

    run_command(
        "ssh_link_bundle_index_object_external_caller",
        remote_shell_command(
            args,
            build_remote_bundle_link_executable_command(
                remote_dir,
                object_file_name,
                "rvv_microkernel_external_caller_from_bundle_object",
                flags,
            ),
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    bundle_object_executable_hash_stdout, _, _ = run_command(
        "ssh_bundle_object_executable_sha256",
        remote_shell_command(
            args,
            remote_sha256_command(
                remote_dir, "rvv_microkernel_external_caller_from_bundle_object"
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
            "./rvv_microkernel_external_caller_from_bundle_object",
        ),
        cwd=root,
        artifact_dir=artifact_dir,
        commands=commands,
        timeout_seconds=args.timeout,
    )
    if EXTERNAL_ABI_SUCCESS_MARKER not in bundle_object_run_stdout:
        raise BridgeError(
            "remote bundle object external caller stdout missing expected marker: "
            + EXTERNAL_ABI_SUCCESS_MARKER
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
            "uname": first_sanitized_line(uname_stdout),
            "architecture": first_sanitized_line(arch_stdout),
            "clang_path": first_sanitized_line(clang_path_stdout),
            "clang_version": first_sanitized_line(clang_version_stdout),
        },
        "remote_artifacts": {
            "source_file": source_file_name,
            "header_file": header_path.name,
            "bundle_object_file": object_file_name,
            "caller_file": caller_path.name,
            "caller_object_file": "rvv_microkernel_external_caller.o",
            "source_built_object_file": "rvv_microkernel_from_source.o",
            "source_built_executable": "rvv_microkernel_external_caller_from_source",
            "bundle_object_executable": "rvv_microkernel_external_caller_from_bundle_object",
        },
        "compile_flags": remote_compile_flags(flags),
        "caller_object_compile_exit_code": 0,
        "source_object_compile_exit_code": 0,
        "source_link_exit_code": 0,
        "source_run_exit_code": 0,
        "bundle_object_link_exit_code": 0,
        "bundle_object_run_exit_code": 0,
        "expected_stdout_marker": EXTERNAL_ABI_SUCCESS_MARKER,
        "source_stdout_marker_observed": True,
        "bundle_object_stdout_marker_observed": True,
        "caller_object_sha256": first_sanitized_line(caller_object_hash_stdout),
        "source_built_object_sha256": first_sanitized_line(source_object_hash_stdout),
        "source_built_executable_sha256": first_sanitized_line(
            source_executable_hash_stdout
        ),
        "bundle_object_executable_sha256": first_sanitized_line(
            bundle_object_executable_hash_stdout
        ),
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
    if getattr(args, "lower_linalg_frontend", False):
        frontend_input = ACTIVE_ARITHMETIC_FAMILY.get("default_frontend_input")
        if frontend_input is not None:
            return Path(frontend_input)
    active_shape = str(ACTIVE_VECTOR_SHAPE["shape"])
    default_shape = str(ACTIVE_ARITHMETIC_FAMILY.get("default_vector_shape", "i32m1"))
    if active_shape == default_shape:
        return Path(ACTIVE_ARITHMETIC_FAMILY["default_input"])

    key = active_shape + "_default_input"
    default_input = ACTIVE_ARITHMETIC_FAMILY.get(key)
    if default_input is None:
        raise BridgeError(
            "no default MLIR fixture for arithmetic family "
            f"{ACTIVE_ARITHMETIC_FAMILY['diagnostic_name']} and vector "
            f"shape {ACTIVE_VECTOR_SHAPE['shape']}; pass --input only if "
            "the fixture already carries the matching typed compiler path"
        )
    return Path(default_input)


def selected_planning_pipeline(args: argparse.Namespace) -> tuple[str, list[str]]:
    if getattr(args, "lower_linalg_frontend", False):
        return (
            "tcrv_opt_linalg_frontend_execution_planning_pipeline",
            [
                "--tcrv-lower-linalg-i32-binary-to-exec",
                "--tcrv-execution-planning-pipeline",
            ],
        )
    if ACTIVE_VECTOR_SHAPE["planning_pipeline"] == "tcrv-execution-planning-pipeline":
        return (
            "tcrv_opt_execution_planning_pipeline",
            ["--tcrv-execution-planning-pipeline"],
        )
    return (
        "tcrv_opt_materialize_plans",
        [
            "--tcrv-materialize-selected-lowering-boundaries",
            "--tcrv-materialize-emission-plans",
        ],
    )


def selected_planning_pipeline_label(args: argparse.Namespace) -> str:
    if getattr(args, "lower_linalg_frontend", False):
        return (
            "tcrv-lower-linalg-i32-binary-to-exec + "
            "tcrv-execution-planning-pipeline"
        )
    return str(ACTIVE_VECTOR_SHAPE["planning_pipeline"])


def run_bundle_bridge(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    if args.evidence_note:
        reject_secret_like_text("evidence note", args.evidence_note)

    run_id = safe_run_id(args.run_id or utc_run_id())
    artifact_dir = prepare_artifact_dir(
        selected_artifact_root(args), run_id, root, args.overwrite
    )
    commands: list[dict[str, Any]] = []

    selected_input = selected_input_path(args)
    input_path = resolve_repo_path(selected_input, root)
    if not input_path.exists():
        raise BridgeError(f"input MLIR does not exist: {selected_input}")
    reject_secret_like_text("input MLIR path", str(selected_input))

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
        planning_command_name, planning_args = selected_planning_pipeline(args)
        post_planning_mlir, _, _ = run_command(
            planning_command_name,
            [
                tcrv_opt,
                relative_to_repo(input_path, root),
                *planning_args,
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
    selected_records = select_rvv_bundle_records(records, bundle_dir)

    source_path = bundle_dir / str(selected_records["source"]["file_name"])
    header_path = bundle_dir / str(selected_records["header"]["file_name"])
    object_path = bundle_dir / str(selected_records["object"]["file_name"])
    source_text = source_path.read_text(encoding="utf-8")
    header_text = header_path.read_text(encoding="utf-8")
    source_flags = validate_generated_source(source_text, require_harness=False)
    expected_selected_kernel = validate_expected_selected_kernel(
        source_flags["compiler_path_context"], args.expect_selected_kernel
    )
    header_function_name = validate_generated_header(header_text)
    if object_path.stat().st_size < 4 or object_path.read_bytes()[:4] != b"\x7fELF":
        raise BridgeError("bundled RVV microkernel object must be a non-empty ELF relocatable")

    caller_text = build_external_caller_source(
        header_function_name,
        str(selected_records["header"]["file_name"]),
        source_flags["runtime_abi_parameters"],
        source_flags["arithmetic_operator"],
        DIRECT_EXTERNAL_RUNTIME_COUNTS,
    )
    caller_path = artifact_dir / "rvv_microkernel_external_caller.c"
    write_generated_text(
        caller_path, "generated RVV microkernel bundle external caller", caller_text
    )

    hashes = {
        "input_sha256": sha256_file(input_path),
        "bundle_export_stdout_sha256": sha256_text(bundle_stdout),
        "bundle_index_sha256": sha256_text(index_text),
        "bundle_microkernel_source_sha256": sha256_text(source_text),
        "bundle_microkernel_header_sha256": sha256_text(header_text),
        "bundle_microkernel_object_sha256": sha256_file(object_path),
        "bundle_external_caller_c_sha256": sha256_text(caller_text),
    }
    if post_planning_path is not None:
        hashes["post_planning_mlir_sha256"] = sha256_text(post_planning_mlir)

    bundle_export_mode = (
        "plan-and-export-target-artifact-bundle"
        if args.use_plan_and_export_bundle_front_door
        else "target-artifact-bundle"
    )
    planned_pipeline = (
        "tcrv-plan-and-export-target-artifact-bundle"
        if args.use_plan_and_export_bundle_front_door
        else selected_planning_pipeline_label(args)
    )
    artifacts = {
        "bundle_export_stdout": relative_to_repo(bundle_stdout_path, root),
        "bundle_dir": relative_to_repo(bundle_dir, root),
        "bundle_index": relative_to_repo(index_path, root),
        "bundle_microkernel_source": relative_to_repo(source_path, root),
        "bundle_microkernel_header": relative_to_repo(header_path, root),
        "bundle_microkernel_object": relative_to_repo(object_path, root),
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
        "mode": "dry-run" if args.dry_run else "ssh",
        "status": "success",
        "repo_commit": current_git_commit(root),
        "arithmetic_family": str(ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"]),
        "function_symbol": source_flags["compiler_path_context"]["microkernel_function"],
        "input": relative_to_repo(input_path, root),
        "artifact_dir": relative_to_repo(artifact_dir, root),
        "planned_pipeline": planned_pipeline,
        "bundle_export_mode": bundle_export_mode,
        "bundle_index": relative_to_repo(index_path, root),
        "bundle_index_summary": bundle_records_summary(records),
        "rvv_config": source_flags["vector_config"],
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
        "source_dataflow_provenance": source_flags["dataflow_provenance"],
        "compiler_path_context": source_flags["compiler_path_context"],
        "runtime_abi_signature": source_flags["runtime_abi_parameters"],
        "arithmetic_operator": source_flags["arithmetic_operator"],
        "expected_selected_kernel": expected_selected_kernel,
        "external_caller": {
            "kind": "generated-c-caller",
            "function": header_function_name,
            "runtime_abi_signature": source_flags["runtime_abi_parameters"],
            "success_marker": EXTERNAL_ABI_SUCCESS_MARKER,
            "arithmetic_check": "lhs "
            + source_flags["arithmetic_operator"]
            + " rhs",
            "runtime_element_counts": DIRECT_EXTERNAL_RUNTIME_COUNTS,
        },
        "selected_compile_flags": remote_compile_flags(source_flags),
        "expected_stdout_marker": EXTERNAL_ABI_SUCCESS_MARKER,
        "stdout_marker_observed": False,
        "pass_fail_result": "pass",
        "no_performance_claim": True,
        "hashes": hashes,
        "artifacts": artifacts,
        "commands": commands,
        "ssh_evidence": False,
        "ssh_evidence_details": None,
        "claim_scope": (
            "local dry-run verifies bundle export, index parsing, file discovery, and external caller construction only"
            if args.dry_run
            else "bounded RVV "
            + str(ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"])
            + " target-artifact bundle external caller correctness only"
        ),
    }

    if not args.dry_run:
        try:
            evidence["ssh_evidence_details"] = run_remote_bundle_external_abi_evidence(
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
            evidence["ssh_evidence_details"]["host_artifact_hashes"] = {
                "bundle_microkernel_source_sha256": hashes[
                    "bundle_microkernel_source_sha256"
                ],
                "bundle_microkernel_header_sha256": hashes[
                    "bundle_microkernel_header_sha256"
                ],
                "bundle_microkernel_object_sha256": hashes[
                    "bundle_microkernel_object_sha256"
                ],
                "bundle_external_caller_c_sha256": hashes[
                    "bundle_external_caller_c_sha256"
                ],
            }
            evidence["ssh_evidence"] = True
            evidence["stdout_marker_observed"] = True
            evidence["commands"] = commands
        except BridgeError as error:
            evidence["status"] = "failure"
            evidence["pass_fail_result"] = "fail"
            evidence["ssh_evidence"] = False
            evidence["stdout_marker_observed"] = False
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


def run_bridge(args: argparse.Namespace) -> dict[str, Any]:
    if args.use_target_artifact_bundle:
        return run_bundle_bridge(args)

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

    selected_input = selected_input_path(args)
    input_path = resolve_repo_path(selected_input, root)
    if not input_path.exists():
        raise BridgeError(f"input MLIR does not exist: {selected_input}")
    reject_secret_like_text("input MLIR path", str(selected_input))

    tcrv_opt = resolve_tool(args.tcrv_opt, "tcrv-opt", root)
    tcrv_translate = resolve_tool(args.tcrv_translate, "tcrv-translate", root)
    local_clang = (
        ensure_local_clang_on_path()
        if not args.dry_run and not use_harness
        else ""
    )

    planning_command_name, planning_args = selected_planning_pipeline(args)
    post_planning_mlir, _, _ = run_command(
        planning_command_name,
        [
            tcrv_opt,
            str(input_path),
            *planning_args,
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
        source_export_route = "generic-target-source-artifact"
    elif use_harness:
        source_export_flag = "--tcrv-export-rvv-microkernel-self-check-c"
        source_export_name = "export_rvv_microkernel_self_check_c"
        source_export_route = "direct-rvv-microkernel-self-check-harness"
    else:
        source_export_route = str(ACTIVE_ARITHMETIC_FAMILY["source_route"])
        source_export_flag = direct_helper_flag(
            direct_helper_translation_route(ACTIVE_ARITHMETIC_FAMILY, "source")
        )
        source_export_name = command_name_for_route(
            direct_helper_translation_route(ACTIVE_ARITHMETIC_FAMILY, "source")
        )
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
    expected_selected_kernel = validate_expected_selected_kernel(
        source_flags["compiler_path_context"], args.expect_selected_kernel
    )
    write_generated_text(source_path, "generated RVV microkernel source", source_text)

    header_path = artifact_dir / "rvv_microkernel.h"
    object_path = artifact_dir / "rvv_microkernel.o"
    caller_path = artifact_dir / "rvv_microkernel_external_caller.c"
    header_function_name = ""
    object_sha256 = ""
    header_sha256 = ""
    caller_sha256 = ""
    direct_helper_routes = direct_helper_routes_for_family(ACTIVE_ARITHMETIC_FAMILY)
    direct_helper_translation_routes = {
        role: direct_helper_translation_route(ACTIVE_ARITHMETIC_FAMILY, role)
        for role in ("source", "header", "object")
    }
    uses_direct_family_helpers = not args.generic_route and not use_harness
    direct_helper_artifacts: dict[str, str] = {}
    if uses_direct_family_helpers:
        direct_helper_artifacts["source"] = relative_to_repo(source_path, root)
    should_export_direct_header = uses_direct_family_helpers
    if should_export_direct_header:
        header_route = direct_helper_routes["header"]
        header_translation_route = direct_helper_translation_routes["header"]
        header_text, _, _ = run_command(
            command_name_for_route(header_translation_route),
            [
                tcrv_translate,
                direct_helper_flag(header_translation_route),
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
        direct_helper_artifacts["header"] = relative_to_repo(header_path, root)

    if not args.dry_run and not use_harness:
        object_translation_route = direct_helper_translation_routes["object"]
        run_command_stdout_to_file(
            command_name_for_route(object_translation_route),
            [
                tcrv_translate,
                direct_helper_flag(object_translation_route),
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
        direct_helper_artifacts["object"] = relative_to_repo(object_path, root)

        caller_text = build_external_caller_source(
            header_function_name,
            header_path.name,
            source_flags["runtime_abi_parameters"],
            source_flags["arithmetic_operator"],
            DIRECT_EXTERNAL_RUNTIME_COUNTS,
        )
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
    if header_sha256:
        hashes["rvv_microkernel_h_sha256"] = header_sha256
    if not args.dry_run and not use_harness:
        hashes.update(
            {
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
        "repo_commit": current_git_commit(root),
        "arithmetic_family": str(ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"]),
        "function_symbol": source_flags["compiler_path_context"]["microkernel_function"],
        "input": relative_to_repo(input_path, root),
        "artifact_dir": relative_to_repo(artifact_dir, root),
        "planned_pipeline": selected_planning_pipeline_label(args),
        "manifest_handoff": True,
        "manifest_record": manifest_handoff,
        "rvv_config": source_flags["vector_config"],
        "source_export_flag": source_export_flag,
        "source_export_route": source_export_route,
        "direct_helper_routes": direct_helper_routes,
        "direct_helper_translation_routes": direct_helper_translation_routes,
        "direct_helper_artifacts": direct_helper_artifacts,
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
        "compiler_path_context": source_flags["compiler_path_context"],
        "runtime_abi_signature": source_flags["runtime_abi_parameters"],
        "arithmetic_operator": source_flags["arithmetic_operator"],
        "expected_selected_kernel": expected_selected_kernel,
        "expected_stdout_marker": (
            SUCCESS_MARKER if use_harness else EXTERNAL_ABI_SUCCESS_MARKER
        ),
        "stdout_marker_observed": False,
        "pass_fail_result": "pass",
        "no_performance_claim": True,
        "hashes": hashes,
        "artifacts": {
            "post_planning_mlir": relative_to_repo(post_planning_path, root),
            "emission_manifest": relative_to_repo(manifest_path, root),
            "rvv_microkernel_c": relative_to_repo(source_path, root),
        },
        "commands": commands,
        "ssh_evidence": False,
        "ssh_evidence_details": None,
        "claim_scope": (
            "local dry-run verifies compiler-tool handoff plus direct source/header helper export only"
            if args.dry_run and uses_direct_family_helpers
            else (
                "local dry-run verifies compiler-tool handoff and source export only"
                if args.dry_run
                else (
                    "bounded generated RVV "
                    + str(ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"])
                    + " self-check executable correctness only"
                    if use_harness
                    else "bounded generated RVV "
                    + str(ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"])
                    + " direct helper artifact handoff plus header/object external caller correctness only"
                )
            )
        ),
    }
    if should_export_direct_header:
        evidence["artifacts"]["rvv_microkernel_h"] = relative_to_repo(
            header_path, root
        )
    if not args.dry_run:
        if use_harness:
            evidence["self_check"] = {
                "kind": "generated-c-self-check-main",
                "success_marker": SUCCESS_MARKER,
                "arithmetic_family": str(
                    ACTIVE_ARITHMETIC_FAMILY["diagnostic_name"]
                ),
            }
        else:
            evidence["artifacts"].update(
                {
                    "rvv_microkernel_o": relative_to_repo(object_path, root),
                    "rvv_microkernel_external_caller_c": relative_to_repo(
                        caller_path, root
                    ),
                }
            )
            evidence["header_function_name"] = header_function_name
            evidence["local_object_export_clang"] = sanitize_text(local_clang)
            evidence["external_caller"] = {
                "kind": "generated-c-caller",
                "function": header_function_name,
                "runtime_abi_signature": source_flags["runtime_abi_parameters"],
                "success_marker": EXTERNAL_ABI_SUCCESS_MARKER,
                "arithmetic_check": "lhs "
                + source_flags["arithmetic_operator"]
                + " rhs",
                "runtime_element_counts": DIRECT_EXTERNAL_RUNTIME_COUNTS,
            }

    if not args.dry_run:
        try:
            if use_harness:
                evidence["ssh_evidence_details"] = run_remote_self_check_source_evidence(
                    args,
                    root=root,
                    artifact_dir=artifact_dir,
                    commands=commands,
                    source_path=source_path,
                    flags=source_flags,
                    run_id=run_id,
                )
                evidence["ssh_evidence_details"]["host_artifact_hashes"] = {
                    "rvv_microkernel_c_sha256": hashes["rvv_microkernel_c_sha256"],
                }
            else:
                evidence["ssh_evidence_details"] = run_remote_external_abi_evidence(
                    args,
                    root=root,
                    artifact_dir=artifact_dir,
                    commands=commands,
                    source_path=source_path,
                    header_path=header_path,
                    object_path=object_path,
                    caller_path=caller_path,
                    flags=source_flags,
                    run_id=run_id,
                )
                evidence["ssh_evidence_details"]["host_artifact_hashes"] = {
                    "rvv_microkernel_c_sha256": hashes["rvv_microkernel_c_sha256"],
                    "rvv_microkernel_h_sha256": hashes["rvv_microkernel_h_sha256"],
                    "rvv_microkernel_o_sha256": hashes["rvv_microkernel_o_sha256"],
                    "rvv_microkernel_external_caller_c_sha256": hashes[
                        "rvv_microkernel_external_caller_c_sha256"
                    ],
                }
            evidence["ssh_evidence"] = True
            evidence["stdout_marker_observed"] = True
            evidence["commands"] = commands
        except BridgeError as error:
            evidence["status"] = "failure"
            evidence["pass_fail_result"] = "fail"
            evidence["ssh_evidence"] = False
            evidence["stdout_marker_observed"] = False
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
    configure_arithmetic_family("i32-vadd")
    configure_vector_shape("i32m1")
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

    configure_arithmetic_family("i32-vsub")
    try:
        find_supported_handoff(supported_manifest)
    except BridgeError as error:
        assert_self_test(
            "stale RVV microkernel handoff for i32-vadd" in str(error),
            "stale vadd handoff error changed",
        )
    else:
        raise AssertionError("stale vadd handoff was accepted for vsub")
    configure_arithmetic_family("i32-vadd")

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
/* microkernel function: tcrv_rvv_i32_vadd_microkernel_rvv_microkernel_manifest_rvv_first_slice */
/* selected_kernel: @rvv_microkernel_manifest */
/* selected_variant: @rvv_first_slice */
/* selected_role: direct variant */
/* lowering_boundary: tcrv_rvv.lowering_boundary */
/* active_route: tcrv-export-rvv-microkernel-self-check-c */
/* callable_abi_source: tcrv.exec.mem_window + tcrv.exec.runtime_param */
/* executable_microkernel: tcrv_rvv.i32_vadd_microkernel */
/* arithmetic_family: i32-vadd */
/* dtype: i32 */
/* arithmetic_c_operator: + */
/* dataflow_body: tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_add -> tcrv_rvv.i32_store */
/* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, rhs_load.buffer_role=rhs-input-buffer, store.buffer_role=output-buffer; runtime n remains the target/export-owned runtime element-count ABI parameter */
/* dataflow_emission_step[0]: op=tcrv_rvv.i32_load, role=lhs-input-buffer, result=lhs_vec */
/* dataflow_emission_step[1]: op=tcrv_rvv.i32_load, role=rhs-input-buffer, result=rhs_vec */
/* dataflow_emission_step[2]: op=tcrv_rvv.i32_add, lhs=lhs_vec, rhs=rhs_vec, result=sum_vec */
/* dataflow_emission_step[3]: op=tcrv_rvv.i32_store, role=output-buffer, value=sum_vec */
/* selected_vector_shape_config: shape=i32m1, sew=32, lmul=m1, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint32m1_t, vector_suffix=i32m1, setvl_suffix=e32m1 */
/* selected_vector_shape_capabilities: rvv.i32_m1.sew32 rvv.i32_m1.lmul_m1 rvv.i32_m1.tail_policy.agnostic rvv.i32_m1.mask_policy.agnostic */
/* control_plane_config: sew=32, lmul=m1, policy=#tcrv_rvv.policy<tail = agnostic, mask = agnostic> */
/* intrinsic_config: vector_type=vint32m1_t, vector_suffix=i32m1, setvl_suffix=e32m1, tail_policy=agnostic, mask_policy=agnostic */
/* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[1]: c_name=rhs, c_type=const int32_t *, role=rhs-input-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[2]: c_name=out, c_type=int32_t *, role=output-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
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
    assert_self_test(
        source_flags["compiler_path_context"]["selected_kernel"]
        == "rvv_microkernel_manifest",
        "compiler path context lost selected kernel",
    )
    assert_self_test(
        validate_expected_selected_kernel(
            source_flags["compiler_path_context"], "@rvv_microkernel_manifest"
        )
        == "rvv_microkernel_manifest",
        "expected selected-kernel normalization failed",
    )
    try:
        validate_expected_selected_kernel(
            source_flags["compiler_path_context"], "frontend_i32_vsub"
        )
    except BridgeError as error:
        assert_self_test(
            "does not match expected @frontend_i32_vsub" in str(error),
            "selected-kernel mismatch diagnostic changed",
        )
    else:
        raise AssertionError("mismatched selected-kernel expectation was accepted")
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

    configure_arithmetic_family("i32-vsub")
    vsub_routes = direct_helper_routes_for_family(ACTIVE_ARITHMETIC_FAMILY)
    assert_self_test(
        vsub_routes["source"] == "tcrv-export-rvv-i32-vsub-microkernel-c",
        "vsub direct helper source route changed",
    )
    assert_self_test(
        direct_helper_flag(vsub_routes["header"])
        == "--tcrv-export-rvv-i32-vsub-microkernel-header",
        "vsub direct helper header flag changed",
    )
    assert_self_test(
        command_name_for_route(vsub_routes["object"])
        == "export_rvv_i32_vsub_microkernel_object",
        "vsub direct helper object command name changed",
    )
    sample_vsub_source = """\
/* selected_march: rv64gcv */
/* selected_mabi: lp64d */
/* microkernel function: tcrv_rvv_i32_vsub_microkernel_rvv_sub_kernel_rvv_sub_slice */
/* selected_kernel: @rvv_sub_kernel */
/* selected_variant: @rvv_sub_slice */
/* selected_role: direct variant */
/* lowering_boundary: tcrv_rvv.lowering_boundary */
/* active_route: tcrv-export-rvv-i32-vsub-microkernel-c */
/* callable_abi_source: tcrv.exec.mem_window + tcrv.exec.runtime_param */
/* executable_microkernel: tcrv_rvv.i32_vsub_microkernel */
/* arithmetic_family: i32-vsub */
/* dtype: i32 */
/* arithmetic_c_operator: - */
/* dataflow_body: tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_sub -> tcrv_rvv.i32_store */
/* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, rhs_load.buffer_role=rhs-input-buffer, store.buffer_role=output-buffer; runtime n remains the target/export-owned runtime element-count ABI parameter */
/* dataflow_emission_step[0]: op=tcrv_rvv.i32_load, role=lhs-input-buffer, result=lhs_vec */
/* dataflow_emission_step[1]: op=tcrv_rvv.i32_load, role=rhs-input-buffer, result=rhs_vec */
/* dataflow_emission_step[2]: op=tcrv_rvv.i32_sub, lhs=lhs_vec, rhs=rhs_vec, result=difference_vec */
/* dataflow_emission_step[3]: op=tcrv_rvv.i32_store, role=output-buffer, value=difference_vec */
/* selected_vector_shape_config: shape=i32m1, sew=32, lmul=m1, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint32m1_t, vector_suffix=i32m1, setvl_suffix=e32m1 */
/* selected_vector_shape_capabilities: rvv.i32_m1.sew32 rvv.i32_m1.lmul_m1 rvv.i32_m1.tail_policy.agnostic rvv.i32_m1.mask_policy.agnostic */
/* control_plane_config: sew=32, lmul=m1, policy=#tcrv_rvv.policy<tail = agnostic, mask = agnostic> */
/* intrinsic_config: vector_type=vint32m1_t, vector_suffix=i32m1, setvl_suffix=e32m1, tail_policy=agnostic, mask_policy=agnostic */
/* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[1]: c_name=rhs, c_type=const int32_t *, role=rhs-input-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[2]: c_name=out, c_type=int32_t *, role=output-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
#include <riscv_vector.h>
void f(void) {
  __riscv_vsetvl_e32m1;
  __riscv_vle32_v_i32m1;
  __riscv_vsub_vv_i32m1;
  __riscv_vse32_v_i32m1;
}
"""
    vsub_flags = validate_generated_source(sample_vsub_source, require_harness=False)
    assert_self_test(
        vsub_flags["dataflow_provenance"]["dataflow_emission_step[2]"]
        == "op=tcrv_rvv.i32_sub, lhs=lhs_vec, rhs=rhs_vec, result=difference_vec",
        "dataflow provenance parser lost subtract step",
    )
    vsub_caller = build_external_caller_source(
        "tcrv_rvv_i32_vsub_microkernel_self_test",
        "artifact-1-runtime-callable-c-header-tcrv-export-rvv-i32-vsub-microkernel-header.h",
    )
    assert_self_test(
        "lhs[index] - rhs[index]" in vsub_caller,
        "vsub external caller did not check subtract semantics",
    )
    assert_self_test(
        EXTERNAL_ABI_SUCCESS_MARKER in vsub_caller,
        "vsub external ABI caller success marker missing",
    )

    configure_vector_shape("i32m2")
    sample_vsub_m2_source = """\
/* selected_march: rv64gcv */
/* selected_mabi: lp64d */
/* microkernel function: tcrv_rvv_i32_vsub_microkernel_frontend_i32_vsub_rvv_first_slice */
/* selected_kernel: @frontend_i32_vsub */
/* selected_variant: @rvv_first_slice */
/* selected_role: direct variant */
/* lowering_boundary: tcrv_rvv.lowering_boundary */
/* active_route: tcrv-export-rvv-i32-vsub-microkernel-c */
/* callable_abi_source: tcrv.exec.mem_window + tcrv.exec.runtime_param */
/* executable_microkernel: tcrv_rvv.i32_vsub_microkernel */
/* arithmetic_family: i32-vsub */
/* dtype: i32 */
/* arithmetic_c_operator: - */
/* dataflow_body: tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_sub -> tcrv_rvv.i32_store */
/* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, rhs_load.buffer_role=rhs-input-buffer, store.buffer_role=output-buffer; runtime n remains the target/export-owned runtime element-count ABI parameter */
/* dataflow_emission_step[0]: op=tcrv_rvv.i32_load, role=lhs-input-buffer, result=lhs_vec */
/* dataflow_emission_step[1]: op=tcrv_rvv.i32_load, role=rhs-input-buffer, result=rhs_vec */
/* dataflow_emission_step[2]: op=tcrv_rvv.i32_sub, lhs=lhs_vec, rhs=rhs_vec, result=difference_vec */
/* dataflow_emission_step[3]: op=tcrv_rvv.i32_store, role=output-buffer, value=difference_vec */
/* selected_vector_shape_config: shape=i32m2, sew=32, lmul=m2, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2 */
/* selected_vector_shape_capabilities: rvv.i32_m2.sew32 rvv.i32_m2.lmul_m2 rvv.i32_m2.tail_policy.agnostic rvv.i32_m2.mask_policy.agnostic */
/* control_plane_config: sew=32, lmul=m2, policy=#tcrv_rvv.policy<tail = agnostic, mask = agnostic> */
/* intrinsic_config: vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2, tail_policy=agnostic, mask_policy=agnostic */
/* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[1]: c_name=rhs, c_type=const int32_t *, role=rhs-input-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[2]: c_name=out, c_type=int32_t *, role=output-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
#include <riscv_vector.h>
void f(void) {
  __riscv_vsetvl_e32m2;
  __riscv_vle32_v_i32m2;
  __riscv_vsub_vv_i32m2;
  __riscv_vse32_v_i32m2;
}
"""
    vsub_m2_flags = validate_generated_source(
        sample_vsub_m2_source, require_harness=False
    )
    assert_self_test(
        vsub_m2_flags["vector_config"]["lmul"] == "m2",
        "m2 vector-shape metadata was not preserved",
    )
    assert_self_test(
        vsub_m2_flags["compiler_path_context"]["selected_kernel"]
        == "frontend_i32_vsub",
        "m2 compiler path context lost frontend selected kernel",
    )
    try:
        validate_generated_source(sample_vsub_source, require_harness=False)
    except BridgeError as error:
        assert_self_test(
            "stale i32m1 vector-shape metadata" in str(error),
            "m2 mode did not reject m1 source metadata",
        )
    else:
        raise AssertionError("m2 mode accepted m1 generated source")
    configure_vector_shape("i32m1")

    configure_arithmetic_family("i32-vmul")
    sample_vmul_source = """\
/* selected_march: rv64gcv */
/* selected_mabi: lp64d */
/* microkernel function: tcrv_rvv_i32_vmul_microkernel_rvv_mul_kernel_rvv_mul_slice */
/* selected_kernel: @rvv_mul_kernel */
/* selected_variant: @rvv_mul_slice */
/* selected_role: direct variant */
/* lowering_boundary: tcrv_rvv.lowering_boundary */
/* active_route: tcrv-export-rvv-i32-vmul-microkernel-c */
/* callable_abi_source: tcrv.exec.mem_window + tcrv.exec.runtime_param */
/* executable_microkernel: tcrv_rvv.i32_vmul_microkernel */
/* arithmetic_family: i32-vmul */
/* dtype: i32 */
/* arithmetic_c_operator: * */
/* dataflow_body: tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_mul -> tcrv_rvv.i32_store */
/* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, rhs_load.buffer_role=rhs-input-buffer, store.buffer_role=output-buffer; runtime n remains the target/export-owned runtime element-count ABI parameter */
/* dataflow_emission_step[0]: op=tcrv_rvv.i32_load, role=lhs-input-buffer, result=lhs_vec */
/* dataflow_emission_step[1]: op=tcrv_rvv.i32_load, role=rhs-input-buffer, result=rhs_vec */
/* dataflow_emission_step[2]: op=tcrv_rvv.i32_mul, lhs=lhs_vec, rhs=rhs_vec, result=product_vec */
/* dataflow_emission_step[3]: op=tcrv_rvv.i32_store, role=output-buffer, value=product_vec */
/* selected_vector_shape_config: shape=i32m1, sew=32, lmul=m1, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint32m1_t, vector_suffix=i32m1, setvl_suffix=e32m1 */
/* selected_vector_shape_capabilities: rvv.i32_m1.sew32 rvv.i32_m1.lmul_m1 rvv.i32_m1.tail_policy.agnostic rvv.i32_m1.mask_policy.agnostic */
/* control_plane_config: sew=32, lmul=m1, policy=#tcrv_rvv.policy<tail = agnostic, mask = agnostic> */
/* intrinsic_config: vector_type=vint32m1_t, vector_suffix=i32m1, setvl_suffix=e32m1, tail_policy=agnostic, mask_policy=agnostic */
/* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[1]: c_name=rhs, c_type=const int32_t *, role=rhs-input-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[2]: c_name=out, c_type=int32_t *, role=output-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
#include <riscv_vector.h>
void f(void) {
  __riscv_vsetvl_e32m1;
  __riscv_vle32_v_i32m1;
  __riscv_vmul_vv_i32m1;
  __riscv_vse32_v_i32m1;
}
"""
    vmul_flags = validate_generated_source(sample_vmul_source, require_harness=False)
    assert_self_test(
        vmul_flags["dataflow_provenance"]["dataflow_emission_step[2]"]
        == "op=tcrv_rvv.i32_mul, lhs=lhs_vec, rhs=rhs_vec, result=product_vec",
        "dataflow provenance parser lost multiply step",
    )
    vmul_caller = build_external_caller_source(
        "tcrv_rvv_i32_vmul_microkernel_self_test",
        "artifact-1-runtime-callable-c-header-tcrv-export-rvv-i32-vmul-microkernel-header.h",
    )
    assert_self_test(
        "lhs[index] * rhs[index]" in vmul_caller,
        "vmul external caller did not check multiply semantics",
    )
    assert_self_test(
        EXTERNAL_ABI_SUCCESS_MARKER in vmul_caller,
        "vmul external ABI caller success marker missing",
    )

    configure_arithmetic_family("i64-vadd")
    configure_vector_shape("i64m1")
    sample_i64_source = """\
/* selected_march: rv64gcv */
/* selected_mabi: lp64d */
/* microkernel function: tcrv_rvv_i64_vadd_microkernel_rvv_i64_vadd_kernel_rvv_i64_slice */
/* selected_kernel: @rvv_i64_vadd_kernel */
/* selected_variant: @rvv_i64_slice */
/* selected_role: direct variant */
/* lowering_boundary: tcrv_rvv.lowering_boundary */
/* active_route: tcrv-export-rvv-i64-vadd-microkernel-c */
/* callable_abi_source: tcrv.exec.mem_window + tcrv.exec.runtime_param */
/* executable_microkernel: tcrv_rvv.i64_vadd_microkernel */
/* arithmetic_family: i64-vadd */
/* dtype: i64 */
/* arithmetic_c_operator: + */
/* dataflow_body: tcrv_rvv.i64_load -> tcrv_rvv.i64_load -> tcrv_rvv.i64_add -> tcrv_rvv.i64_store */
/* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, rhs_load.buffer_role=rhs-input-buffer, store.buffer_role=output-buffer; runtime n remains the target/export-owned runtime element-count ABI parameter */
/* dataflow_emission_step[0]: op=tcrv_rvv.i64_load, role=lhs-input-buffer, result=lhs_vec */
/* dataflow_emission_step[1]: op=tcrv_rvv.i64_load, role=rhs-input-buffer, result=rhs_vec */
/* dataflow_emission_step[2]: op=tcrv_rvv.i64_add, lhs=lhs_vec, rhs=rhs_vec, result=sum_vec */
/* dataflow_emission_step[3]: op=tcrv_rvv.i64_store, role=output-buffer, value=sum_vec */
/* selected_vector_shape_config: dtype=i64, shape=i64m1, sew=64, lmul=m1, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint64m1_t, vector_suffix=i64m1, setvl_suffix=e64m1 */
/* selected_vector_shape_capabilities: rvv.i64_m1.sew64 rvv.i64_m1.lmul_m1 rvv.i64_m1.tail_policy.agnostic rvv.i64_m1.mask_policy.agnostic */
/* control_plane_config: sew=64, lmul=m1, policy=#tcrv_rvv.policy<tail = agnostic, mask = agnostic> */
/* intrinsic_config: vector_type=vint64m1_t, vector_suffix=i64m1, setvl_suffix=e64m1, tail_policy=agnostic, mask_policy=agnostic */
/* runtime_abi_parameter[0]: c_name=lhs, c_type=const int64_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[1]: c_name=rhs, c_type=const int64_t *, role=rhs-input-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[2]: c_name=out, c_type=int64_t *, role=output-buffer, ownership=target-export-abi-owned */
/* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
#include <riscv_vector.h>
void f(void) {
  __riscv_vsetvl_e64m1;
  __riscv_vle64_v_i64m1;
  __riscv_vadd_vv_i64m1;
  __riscv_vse64_v_i64m1;
}
"""
    i64_flags = validate_generated_source(sample_i64_source, require_harness=False)
    assert_self_test(
        i64_flags["runtime_abi_parameters"][0]["c_type"] == "const int64_t *",
        "i64 runtime ABI signature was not preserved",
    )
    i64_header = """\
#ifndef TIANCHENRV_RVV_I64_VADD_MICROKERNEL_SELF_TEST_H
#define TIANCHENRV_RVV_I64_VADD_MICROKERNEL_SELF_TEST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void tcrv_rvv_i64_vadd_microkernel_self_test(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* TIANCHENRV_RVV_I64_VADD_MICROKERNEL_SELF_TEST_H */
"""
    i64_function_name = validate_generated_header(i64_header)
    i64_caller = build_external_caller_source(
        i64_function_name,
        "artifact-1-runtime-callable-c-header-tcrv-export-rvv-i64-vadd-microkernel-header.h",
        i64_flags["runtime_abi_parameters"],
        i64_flags["arithmetic_operator"],
        DIRECT_EXTERNAL_RUNTIME_COUNTS,
    )
    assert_self_test(
        "lhs[index] + rhs[index]" in i64_caller,
        "i64 external caller did not check add semantics",
    )
    assert_self_test(
        "counts=7,16" in i64_caller,
        "i64 external caller did not record the expected runtime counts",
    )
    assert_self_test(
        EXTERNAL_ABI_SUCCESS_MARKER in i64_caller,
        "i64 external ABI caller success marker missing",
    )

    configure_arithmetic_family("i32-vadd")
    configure_vector_shape("i32m1")
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
    parser.add_argument("--input", default="")
    parser.add_argument(
        "--arithmetic-family",
        choices=sorted(ARITHMETIC_FAMILY_SPECS),
        default="i32-vadd",
        help="Bounded RVV direct microkernel arithmetic family to validate",
    )
    parser.add_argument(
        "--vector-shape",
        choices=sorted(RVV_VECTOR_SHAPE_SPECS),
        default="",
        help=(
            "Bounded typed RVV vector shape to validate; defaults to the "
            "selected family's descriptor-backed shape"
        ),
    )
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
        "--lower-linalg-frontend",
        action="store_true",
        help="Run bounded linalg frontend lowering before execution planning",
    )
    parser.add_argument(
        "--expect-selected-kernel",
        default="",
        help=(
            "Require the compiler-emitted generated source selected_kernel "
            "comment to match this kernel symbol before accepting evidence"
        ),
    )
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
    configure_vector_shape(
        args.vector_shape
        or str(ACTIVE_ARITHMETIC_FAMILY.get("default_vector_shape", "i32m1"))
    )
    if args.self_test:
        run_self_test()
        return 0
    if (
        args.use_plan_and_export_bundle_front_door
        and not args.use_target_artifact_bundle
    ):
        print(
            "rvv_microkernel_e2e: --use-plan-and-export-bundle-front-door "
            "requires --use-target-artifact-bundle",
            file=sys.stderr,
        )
        return 1
    if args.lower_linalg_frontend and args.use_plan_and_export_bundle_front_door:
        print(
            "rvv_microkernel_e2e: --lower-linalg-frontend is not supported "
            "with --use-plan-and-export-bundle-front-door",
            file=sys.stderr,
        )
        return 1
    if args.use_target_artifact_bundle and (
        args.generic_route or args.self_check_harness
    ):
        print(
            "rvv_microkernel_e2e: --use-target-artifact-bundle cannot be "
            "combined with --generic-route or --self-check-harness",
            file=sys.stderr,
        )
        return 1

    try:
        evidence = run_bridge(args)
    except BridgeError as error:
        print(f"rvv_microkernel_e2e: {sanitize_text(str(error))}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "artifact_dir": evidence["artifact_dir"],
                "arithmetic_family": evidence.get("arithmetic_family", ""),
                "bundle_export_mode": evidence.get("bundle_export_mode", ""),
                "mode": evidence["mode"],
                "status": evidence["status"],
                "manifest_handoff": evidence.get("manifest_handoff", False),
                "planned_pipeline": evidence.get("planned_pipeline", ""),
                "selected_kernel": evidence.get("compiler_path_context", {}).get(
                    "selected_kernel", ""
                ),
                "expected_selected_kernel": evidence.get(
                    "expected_selected_kernel", ""
                ),
                "vector_shape": evidence.get("rvv_config", {}).get("shape", ""),
                "lmul": evidence.get("rvv_config", {}).get("lmul", ""),
                "source_sha256": evidence["hashes"].get(
                    "rvv_microkernel_c_sha256",
                    evidence["hashes"].get("bundle_microkernel_source_sha256", ""),
                ),
                "source_export_mode": evidence["source_export_mode"],
                "source_export_route": evidence.get("source_export_route", ""),
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
