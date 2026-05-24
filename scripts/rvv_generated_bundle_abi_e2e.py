#!/usr/bin/env python3
"""Prove generated RVV object/header bundle ABI consumption on ``ssh rvv``.

This is evidence tooling only. By default it starts from hand-authored explicit
selected ``tcrv.exec`` / ``tcrv_rvv`` body fixtures, materializes selected
emission plans, exports the generated target artifact bundle, checks the
bundle, builds a small external C ABI consumer, and optionally runs that
consumer on the real RVV target. ``--pre-realized-selected-body`` starts from
the bounded pre-realized selected-body fixtures and uses the public selected
lowering-boundary materialization pass before emission planning unless
``--direct-pre-realized-route-entry`` is set for the bounded route-entry
artifact/ABI evidence cases. The legacy ``--source-seed`` mode is unsupported
and exits before bundle generation.
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
DEFAULT_RHS_SCALAR_VALUES = (-37,)
DEFAULT_STRIDED_LOAD_BYTE_STRIDES = (4, 8, 12)
MIN_RUNTIME_COUNT_CASES = 2
MIN_NON_ONE_VECTOR_SENTINEL_COUNT = 17
DEFAULT_OP_KINDS = ("add", "sub", "mul")
MASKED_ELEMENTWISE_OP_KINDS = ("masked_add", "masked_sub", "masked_mul")
SCALAR_BROADCAST_OP_KINDS = (
    "scalar_broadcast_add",
    "scalar_broadcast_sub",
    "scalar_broadcast_mul",
)
OP_KIND_CHOICES = DEFAULT_OP_KINDS + (
    "cmp_select",
    "cmp_select_sle",
    "computed_mask_select",
    "computed_mask_select_sle",
    "runtime_scalar_cmp_select",
    "runtime_scalar_dual_cmp_mask_and_select",
    "runtime_scalar_cmp_masked_store",
    "runtime_scalar_cmp_masked_load_store",
    "reduce_add",
    *MASKED_ELEMENTWISE_OP_KINDS,
    "macc_add",
    "computed_masked_macc_add",
    "runtime_scalar_cmp_masked_macc_add",
    "widening_macc_add",
    "widening_dot_reduce_add",
    "strided_input_widening_dot_reduce_add",
    "computed_masked_widening_dot_reduce_add",
    "computed_masked_strided_input_widening_dot_reduce_add",
    "strided_add",
    "strided_load_unit_store",
    "unit_load_strided_store",
    "indexed_gather_unit_store",
    "indexed_scatter_unit_load",
    "masked_unit_load_store",
    "masked_unit_store",
    "computed_masked_unit_load_store",
    "computed_masked_strided_store",
    "computed_masked_strided_load_unit_store",
    "computed_masked_indexed_gather_load_unit_store",
    "computed_masked_indexed_scatter_store_unit_load",
    "computed_masked_segment2_load_unit_store",
    "computed_masked_segment2_store_unit_load",
    "segment2_deinterleave_unit_store",
    "segment2_interleave_unit_load",
    *SCALAR_BROADCAST_OP_KINDS,
    "standalone_reduce_add",
    "standalone_reduce_min",
    "standalone_reduce_max",
    "computed_mask_standalone_reduce_add",
    "computed_mask_standalone_reduce_min",
    "computed_mask_standalone_reduce_max",
    "runtime_scalar_cmp_masked_standalone_reduce_add",
    "runtime_i32_splat_store",
    "i64_add",
    "lmul_m2_add",
    "widen_i32_to_i64",
    "widen_i16_to_i32",
)
REDUCE_ADD_ACCUMULATOR_LAYOUT = "rhs-vector-seed-lane0-per-vl-chunk"
REDUCE_ADD_RESULT_LAYOUT = "store-reduction-lane0-to-output-chunk-base"
REDUCE_ADD_STORE_VL = "1"
MASKED_ADD_MASK_SOURCE = "compare-produced-mask-same-vl-scope"
MASKED_ADD_MASK_ROLE = "predicate-mask-produced-by-compare"
MASKED_ADD_INACTIVE_LANE_CONTRACT = "masked-off-lanes-preserve-passthrough-vector"
MASKED_ADD_PASSTHROUGH_LAYOUT = "passthrough-vector-preserves-inactive-lanes"
MACC_ADD_ACCUMULATOR_LAYOUT = "separate-i32-vector-accumulator-input"
MACC_ADD_RESULT_LAYOUT = "store-multiply-accumulate-result-to-output-buffer"
MACC_ADD_RUNTIME_ABI_ORDER = "lhs,rhs,acc,out,n"
COMPUTED_MASKED_MACC_ADD_RUNTIME_ABI_ORDER = "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n"
COMPUTED_MASKED_MACC_ADD_MEMORY_LAYOUT = (
    "unit-stride-compare-lhs-rhs-accumulator-masked-macc-output-runtime-abi"
)
COMPUTED_MASKED_MACC_ADD_INACTIVE_LANE_CONTRACT = (
    "masked-macc-false-lanes-preserve-accumulator"
)
COMPUTED_MASKED_MACC_ADD_PASSTHROUGH_LAYOUT = (
    "accumulator-vector-preserves-inactive-lanes"
)
COMPUTED_MASKED_MACC_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-computed-mask-macc-add-leaf-profile.v1"
)
COMPUTED_MASKED_MACC_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-computed-mask-macc-add-plan-validated"
)
COMPUTED_MASKED_MACC_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
COMPUTED_MASKED_MACC_C_TYPE_MAPPING = (
    "vl:size_t,cmp_lhs/cmp_rhs/lhs/rhs/acc:signed-e32m1,mask:b32,"
    "result:signed-e32m1"
)
COMPUTED_MASK_ACCUMULATION_ROUTE_FAMILY_PLAN = (
    "rvv-computed-mask-accumulation-route-family-plan.v1"
)
COMPUTED_MASK_ACCUMULATION_VECTOR_MACC_SUFFIX = "vector-masked-macc-add"
COMPUTED_MASK_ACCUMULATION_STANDALONE_REDUCE_SUFFIX = (
    "scalar-horizontal-masked-standalone-reduce-add"
)
COMPUTED_MASK_ACCUMULATION_VECTOR_PRODUCER_SOURCE = "vector-compare-rhs-load"
COMPUTED_MASK_ACCUMULATION_RUNTIME_SCALAR_PRODUCER_SOURCE = (
    "runtime-scalar-splat-compare-rhs"
)
COMPUTED_MASK_ACCUMULATION_VECTOR_MACC_ACCUMULATOR_CONTRACT = (
    "vector-accumulator-input-preserves-inactive-lanes"
)
COMPUTED_MASK_ACCUMULATION_VECTOR_MACC_RESULT_CONTRACT = (
    "vector-macc-result-stored-to-output-buffer"
)
COMPUTED_MASK_ACCUMULATION_STANDALONE_ACCUMULATOR_CONTRACT = (
    "scalar-seed-input-feeds-masked-horizontal-reduction"
)
COMPUTED_MASK_ACCUMULATION_STANDALONE_RESULT_CONTRACT = (
    "scalar-horizontal-reduction-lane0-stored-to-output"
)
COMPUTED_MASK_ACCUMULATION_STANDALONE_SCALAR_CARRY_CONTRACT = (
    "scalar-result-carries-across-runtime-vl-chunks"
)
RUNTIME_SCALAR_COMPUTED_MASKED_MACC_ADD_RUNTIME_ABI_ORDER = (
    "cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n"
)
RUNTIME_SCALAR_COMPUTED_MASKED_MACC_ADD_MEMORY_LAYOUT = (
    "unit-stride-compare-lhs-runtime-scalar-threshold-lhs-rhs-accumulator-"
    "masked-macc-output-runtime-abi"
)
RUNTIME_SCALAR_COMPUTED_MASKED_MACC_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-runtime-scalar-cmp-masked-macc-add-leaf-profile.v1"
)
RUNTIME_SCALAR_COMPUTED_MASKED_MACC_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-macc-add-plan-"
    "validated"
)
RUNTIME_SCALAR_COMPUTED_MASKED_MACC_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
RUNTIME_SCALAR_COMPUTED_MASKED_MACC_C_TYPE_MAPPING = (
    "vl:size_t,cmp_lhs/lhs/rhs/acc:signed-e32m1,rhs_scalar:i32,mask:b32,"
    "result:signed-e32m1"
)
WIDENING_MACC_ACCUMULATOR_LAYOUT = "separate-i32-vector-accumulator-input"
WIDENING_MACC_RESULT_LAYOUT = (
    "store-widening-multiply-accumulate-result-to-output-buffer"
)
WIDENING_MACC_RELATION = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1"
WIDENING_MACC_RUNTIME_ABI_ORDER = "lhs,rhs,acc,out,n"
WIDENING_DOT_ACCUMULATOR_LAYOUT = "scalar-i32-seed-lane0-from-accumulator-input"
WIDENING_DOT_RESULT_LAYOUT = "store-dot-reduction-lane0-to-output-scalar"
WIDENING_DOT_RELATION = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32"
WIDENING_DOT_REDUCTION_STORE_VL = "1"
WIDENING_DOT_RUNTIME_ABI_ORDER = "lhs,rhs,acc,out,n"
STANDALONE_REDUCE_ACCUMULATOR_LAYOUT = "scalar-i32-seed-lane0-from-accumulator-input"
STANDALONE_REDUCE_RESULT_LAYOUT = "store-standalone-reduction-lane0-to-output-scalar"
STANDALONE_REDUCE_STORE_VL = "1"
STANDALONE_REDUCE_RUNTIME_ABI_ORDER = "lhs,acc,out,n"
COMPUTED_MASK_STANDALONE_REDUCE_RUNTIME_ABI_ORDER = "cmp_lhs,cmp_rhs,src,acc,out,n"
RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_RUNTIME_ABI_ORDER = (
    "cmp_lhs,rhs_scalar,src,acc,out,n"
)
STANDALONE_REDUCTION_ROUTE_FAMILY_PLAN = (
    "rvv-standalone-reduction-route-family-plan.v1"
)
RUNTIME_AVL_VL_CONTROL_PLAN = "rvv-runtime-avl-vl-control-plan.v1"
STANDALONE_REDUCE_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-standalone-reduction-leaf-profile.v1"
)
STANDALONE_REDUCE_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-standalone-reduction-plan-validated"
)
STANDALONE_REDUCE_REQUIRED_HEADER_DECLARATIONS = "stddef.h,stdint.h,riscv_vector.h"
STANDALONE_REDUCE_C_TYPE_MAPPING = "vl:size_t,input:signed-e32m1,seed:i32,result:signed-e32m1"
COMPUTED_MASK_STANDALONE_REDUCE_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-computed-mask-standalone-reduction-leaf-profile.v1"
)
COMPUTED_MASK_STANDALONE_REDUCE_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-computed-mask-standalone-reduction-plan-validated"
)
COMPUTED_MASK_STANDALONE_REDUCE_C_TYPE_MAPPING = (
    "vl:size_t,compare/source:signed-e32m1,mask:b32,seed:i32,result:signed-e32m1"
)
RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-runtime-scalar-cmp-masked-standalone-reduction-leaf-profile.v1"
)
RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-standalone-reduction-plan-validated"
)
RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_C_TYPE_MAPPING = (
    "vl:size_t,cmp_lhs/source:signed-e32m1,rhs_scalar:i32,mask:b32,"
    "seed:i32,result:signed-e32m1"
)
COMPUTED_MASK_STANDALONE_REDUCE_INACTIVE_ZEROING = (
    "masked-standalone-reduction-zero-inactive-lanes-before-reduction"
)
COMPUTED_MASK_STANDALONE_REDUCE_INACTIVE_NEUTRAL = (
    "masked-standalone-reduction-neutral-inactive-lanes-before-reduction"
)
CONTRACTION_TARGET_LEAF_PROFILE = "rvv-v1-i16mf2-i32m1-contraction-leaf-profile.v1"
CONTRACTION_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-contraction-family-plan-validated"
)
CONTRACTION_REQUIRED_HEADER_DECLARATIONS = "stddef.h,stdint.h,riscv_vector.h"
CONTRACTION_C_TYPE_MAPPING = (
    "vl:size_t,source:signed-e16mf2,result:signed-e32m1,mask:b32"
)
CONTRACTION_MASKED_INACTIVE_LANE_ZEROING_REQUIREMENT = (
    "masked-widening-products-zero-inactive-lanes-before-reduction"
)
STRIDED_INPUT_WIDENING_DOT_RUNTIME_ABI_ORDER = (
    "lhs,rhs,acc,out,n,lhs_stride,rhs_stride"
)
STRIDED_INPUT_WIDENING_DOT_MEMORY_LAYOUT = (
    "element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi"
)
STRIDED_INPUT_WIDENING_DOT_LHS_STRIDE_SOURCE = "runtime_abi:lhs_stride"
STRIDED_INPUT_WIDENING_DOT_RHS_STRIDE_SOURCE = "runtime_abi:rhs_stride"
STRIDED_INPUT_WIDENING_DOT_SOURCE_MEMORY_FORM = "strided-load"
STRIDED_INPUT_WIDENING_DOT_DESTINATION_MEMORY_FORM = "unit-stride-store"
STRIDED_INPUT_WIDENING_DOT_STRIDED_LOAD_INTRINSIC = "__riscv_vlse16_v_i16mf2"
COMPUTED_MASK_WIDENING_DOT_RUNTIME_ABI_ORDER = (
    "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n"
)
COMPUTED_MASK_STRIDED_INPUT_WIDENING_DOT_RUNTIME_ABI_ORDER = (
    "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride"
)
COMPUTED_MASK_STRIDED_INPUT_WIDENING_DOT_MEMORY_LAYOUT = (
    "unit-stride-compare-element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi"
)
STRIDED_ADD_RUNTIME_ABI_ORDER = "lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride"
STRIDED_LOAD_UNIT_STORE_RUNTIME_ABI_ORDER = "src,out,n,stride_bytes"
UNIT_LOAD_STRIDED_STORE_RUNTIME_ABI_ORDER = "src,dst,n,dst_stride_bytes"
MACC_ROUTE_OPERAND_BINDING_PLAN = "rvv-route-operand-binding:macc_add.v1"
MACC_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:macc_add.v1;"
    "lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|macc-lhs-call;"
    "rhs=rhs-input-buffer:rhs:runtime-abi-mirror|materialized-load-base|macc-rhs-call;"
    "acc=accumulator-input-buffer:acc:runtime-abi-mirror|materialized-accumulator-load-base|macc-accumulator-call;"
    "out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"
)
COMPUTED_MASKED_MACC_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:computed_masked_macc_add.v1"
)
COMPUTED_MASKED_MACC_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:computed_masked_macc_add.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs|cmp-call|hdr;"
    "lhs=dot-lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;"
    "rhs=dot-rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;"
    "acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;"
    "out=output-buffer:out:abi|store|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop|hdr"
)
RUNTIME_SCALAR_COMPUTED_MASKED_MACC_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1"
)
RUNTIME_SCALAR_COMPUTED_MASKED_MACC_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;"
    "rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr;"
    "lhs=dot-lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;"
    "rhs=dot-rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;"
    "acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;"
    "out=output-buffer:out:abi|store|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop|hdr"
)
WIDENING_MACC_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:widening_macc_add.v1"
)
WIDENING_MACC_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:widening_macc_add.v1;"
    "lhs=lhs-input-buffer:lhs:abi|src-load|wmacc-lhs|src-i16mf2|hdr;"
    "rhs=rhs-input-buffer:rhs:abi|src-load|wmacc-rhs|src-i16mf2|hdr;"
    "acc=accumulator-input-buffer:acc:abi|acc-load|wmacc-acc|acc-i32m1|hdr;"
    "out=output-buffer:out:abi|res-store|res-i32m1|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop|hdr"
)
STRIDED_LOAD_UNIT_STORE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:strided_load_unit_store.v1"
)
STRIDED_LOAD_UNIT_STORE_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:strided_load_unit_store.v1;"
    "src=source-input-buffer:src:runtime-abi-mirror|materialized-strided-load-base|move-source;"
    "out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror;"
    "stride_bytes=source-byte-stride:stride_bytes:runtime-abi-mirror|materialized-strided-load-stride|materialized-byte-address|header-mirror"
)
UNIT_LOAD_STRIDED_STORE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:unit_load_strided_store.v1"
)
UNIT_LOAD_STRIDED_STORE_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:unit_load_strided_store.v1;"
    "src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|move-source;"
    "dst=output-buffer:dst:runtime-abi-mirror|materialized-strided-store-base|header-mirror;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror;"
    "dst_stride_bytes=destination-byte-stride:dst_stride_bytes:runtime-abi-mirror|materialized-strided-store-stride|materialized-byte-address|header-mirror"
)
INDEXED_GATHER_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:indexed_gather_unit_store.v1"
)
INDEXED_GATHER_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:indexed_gather_unit_store.v1;"
    "data=lhs-input-buffer:data:runtime-abi-mirror|materialized-indexed-data-base|indexed-load-base|header-mirror;"
    "index=index-input-buffer:index:runtime-abi-mirror|materialized-index-load-base|index-offset-scale|index-source-mirror|header-mirror;"
    "out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"
)
INDEXED_SCATTER_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:indexed_scatter_unit_load.v1"
)
INDEXED_SCATTER_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:indexed_scatter_unit_load.v1;"
    "src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|move-source|header-mirror;"
    "index=index-input-buffer:index:runtime-abi-mirror|materialized-index-load-base|index-offset-scale|index-source-mirror|header-mirror;"
    "dst=output-buffer:dst:runtime-abi-mirror|materialized-indexed-store-base|header-mirror;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"
)
SEGMENT2_DEINTERLEAVE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:segment2_deinterleave_unit_store.v1"
)
SEGMENT2_DEINTERLEAVE_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:segment2_deinterleave_unit_store.v1;"
    "src=lhs-input-buffer:src:runtime-abi-mirror|seg-load-base|src-mem|header;"
    "out0=segment-field0-output-buffer:out0:runtime-abi-mirror|field0-store-base|field0-role|dst-mem|header;"
    "out1=segment-field1-output-buffer:out1:runtime-abi-mirror|field1-store-base|field1-role|dst-mem|header;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header"
)
SEGMENT2_INTERLEAVE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:segment2_interleave_unit_load.v1"
)
SEGMENT2_INTERLEAVE_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:segment2_interleave_unit_load.v1;"
    "src0=segment-field0-input-buffer:src0:runtime-abi-mirror|field0-load-base|field0-role|src0-mem|tuple-field0|header;"
    "src1=segment-field1-input-buffer:src1:runtime-abi-mirror|field1-load-base|field1-role|src1-mem|tuple-field1|header;"
    "dst=segment-interleaved-output-buffer:dst:runtime-abi-mirror|seg-store-base|dst-mem|header;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header"
)
SCALAR_BROADCAST_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:scalar_broadcast_add.v1"
)
SCALAR_BROADCAST_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:scalar_broadcast_add.v1;"
    "lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|scalar-broadcast-lhs-call|header-mirror;"
    "rhs_scalar=rhs-scalar-value:rhs_scalar:runtime-abi-mirror|scalar-broadcast-rhs-call|header-mirror;"
    "out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"
)
SCALAR_BROADCAST_ELEMENTWISE_ROUTE_FAMILY_PLAN = (
    "rvv-scalar-broadcast-elementwise-route-family-plan.v1"
)
STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_PLAN = "rvv-route-operand-binding:standalone_reduce_add.v1"
STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_OPERANDS_TEMPLATE = (
    "rvv-route-operand-binding:{kind}.v1;"
    "lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|standalone-reduction-input-call;"
    "acc=accumulator-input-buffer:acc:runtime-abi-mirror|standalone-initial-accumulator-call;"
    "out=output-buffer:out:runtime-abi-mirror|standalone-accumulator-state-load|materialized-store-base|header-mirror;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"
)
STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_OPERANDS = (
    STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_OPERANDS_TEMPLATE.format(
        kind="standalone_reduce_add"
    )
)
COMPUTED_MASK_STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:computed_mask_standalone_reduce_add.v1"
)
COMPUTED_MASK_STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_OPERANDS_TEMPLATE = (
    "rvv-route-operand-binding:{kind}.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|cmp-lhs-call|hdr;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|cmp-rhs-call|hdr;"
    "src=source-input-buffer:src:abi|src-load|masked-reduce-input|{inactive_use}|hdr;"
    "acc=accumulator-input-buffer:acc:abi|initial-seed|acc-state|masked-reduce-acc;"
    "out=output-buffer:out:abi|acc-state|store-base|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop|hdr"
)
COMPUTED_MASK_STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_OPERANDS = (
    COMPUTED_MASK_STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_OPERANDS_TEMPLATE.format(
        kind="computed_mask_standalone_reduce_add",
        inactive_use="zero-inactive",
    )
)
RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_add.v1"
)
RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_add.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|cmp-lhs-call|hdr;"
    "rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs-call|hdr;"
    "src=source-input-buffer:src:abi|src-load|masked-reduce-input|zero-inactive|hdr;"
    "acc=accumulator-input-buffer:acc:abi|initial-seed|acc-state|masked-reduce-acc;"
    "out=output-buffer:out:abi|acc-state|store-base|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop|hdr"
)
MASKED_UNIT_LOAD_STORE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:masked_unit_load_store.v1"
)
MASKED_UNIT_LOAD_STORE_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:masked_unit_load_store.v1;"
    "src=lhs-input-buffer:src:runtime-abi-mirror|materialized-masked-load-base|masked-load-source-call|header-mirror;"
    "mask=mask-input-buffer:mask:runtime-abi-mirror|materialized-mask-load-base|masked-load-mask-call;"
    "dst=output-buffer:dst:runtime-abi-mirror|materialized-old-destination-load-base|masked-load-passthrough-call|materialized-store-base|header-mirror;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"
)
MASKED_UNIT_STORE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:masked_unit_store.v1"
)
MASKED_UNIT_STORE_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:masked_unit_store.v1;"
    "src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|masked-store-source-call;"
    "mask=mask-input-buffer:mask:runtime-abi-mirror|materialized-mask-load-base|masked-store-mask-call;"
    "dst=output-buffer:dst:runtime-abi-mirror|materialized-masked-store-base|header-mirror;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"
)
COMPUTED_MASK_UNIT_LOAD_STORE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:computed_masked_unit_load_store.v1"
)
COMPUTED_MASK_UNIT_LOAD_STORE_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:computed_masked_unit_load_store.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi-mirror|cmp-lhs-load|compare-lhs-call;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi-mirror|cmp-rhs-load|compare-rhs-call;"
    "src=source-input-buffer:src:abi-mirror|materialized-masked-load-base|masked-load-source-call;"
    "dst=output-buffer:dst:abi-mirror|old-dst-load|masked-load-passthrough-call|materialized-store-base|header-mirror;"
    "n=runtime-element-count:n:abi-mirror|setvl-avl|loop-control|header-mirror"
)
COMPUTED_MASK_STRIDED_STORE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:computed_masked_strided_store.v1"
)
COMPUTED_MASK_STRIDED_STORE_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:computed_masked_strided_store.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi-mirror|cmp-lhs-load|cmp-lhs-call;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi-mirror|cmp-rhs-load|cmp-rhs-call;"
    "src=source-input-buffer:src:abi-mirror|src-load|mstr-store-src-call;"
    "dst=output-buffer:dst:abi-mirror|mstr-store-base|header-mirror;"
    "n=runtime-element-count:n:abi-mirror|setvl-avl|loop-control|header-mirror;"
    "dst_stride_bytes=destination-byte-stride:dst_stride_bytes:abi-mirror|mstr-store-stride|byte|header-mirror"
)
COMPUTED_MASK_STRIDED_LOAD_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:computed_masked_strided_load_unit_store.v1"
)
COMPUTED_MASK_STRIDED_LOAD_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:computed_masked_strided_load_unit_store.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call;"
    "src=source-input-buffer:src:abi|mstr-base|mstr-load-call;"
    "dst=output-buffer:dst:abi|old-dst-load|passthru-call|store-base|hdr-mirror;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr-mirror;"
    "src_stride_bytes=source-byte-stride:src_stride_bytes:abi|mstr-stride|byte|hdr-mirror"
)
COMPUTED_MASK_INDEXED_GATHER_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:computed_masked_indexed_gather_load_unit_store.v1"
)
COMPUTED_MASK_INDEXED_GATHER_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:computed_masked_indexed_gather_load_unit_store.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call;"
    "src=source-input-buffer:src:abi|midx-base|midx-load-call;"
    "index=index-input-buffer:index:abi|materialized-index-load-base|"
    "index-offset-scale|index-source-mirror|hdr-mirror;"
    "dst=output-buffer:dst:abi|old-dst-load|passthru-call|store-base|hdr-mirror;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr-mirror"
)
COMPUTED_MASK_INDEXED_SCATTER_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:computed_masked_indexed_scatter_store_unit_load.v1"
)
COMPUTED_MASK_INDEXED_SCATTER_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:computed_masked_indexed_scatter_store_unit_load.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call;"
    "src=source-input-buffer:src:abi|src-load|mistore-src-call;"
    "index=index-input-buffer:index:abi|materialized-index-load-base|"
    "index-offset-scale|index-source-mirror|hdr-mirror;"
    "dst=output-buffer:dst:abi|mistore-base|hdr-mirror;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr-mirror"
)
COMPUTED_MASK_SEGMENT2_LOAD_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1"
)
COMPUTED_MASK_SEGMENT2_LOAD_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call;"
    "src=source-input-buffer:src:abi|mseg-base|mseg-call|src-mem;"
    "out0=segment-field0-output-buffer:out0:abi|old0-load|f0-pass|"
    "f0-store|f0-role|dst-mem|hdr;"
    "out1=segment-field1-output-buffer:out1:abi|old1-load|f1-pass|"
    "f1-store|f1-role|dst-mem|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"
)
COMPUTED_MASK_SEGMENT2_STORE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:computed_masked_segment2_store_unit_load.v1"
)
COMPUTED_MASK_SEGMENT2_STORE_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:computed_masked_segment2_store_unit_load.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call;"
    "src0=segment-field0-input-buffer:src0:abi|f0-load|f0-payload|tuple0|f0-role|src0-mem;"
    "src1=segment-field1-input-buffer:src1:abi|f1-load|f1-payload|tuple1|f1-role|src1-mem;"
    "dst=segment-interleaved-output-buffer:dst:abi|mseg-store|dst-mem|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"
)
BINARY_ROUTE_OPERAND_BINDING_OPERANDS = (
    "{plan};"
    "lhs=lhs-input-buffer:lhs:abi|load-base|binary-lhs-call;"
    "rhs=rhs-input-buffer:rhs:abi|load-base|binary-rhs-call;"
    "out=output-buffer:out:abi|store-base|header;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|header"
)
CMP_SELECT_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:cmp_select.v1"
)
CMP_SELECT_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:cmp_select.v1;"
    "lhs=lhs-input-buffer:lhs:abi|load-base|compare-lhs-call|select-true-call;"
    "rhs=rhs-input-buffer:rhs:abi|load-base|compare-rhs-call|select-false-call;"
    "out=output-buffer:out:abi|store-base|header;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|header"
)
PLAIN_COMPARE_SELECT_ROUTE_FAMILY_PLAN = (
    "rvv-plain-compare-select-route-family-plan.v1"
)
PLAIN_COMPARE_SELECT_TARGET_LEAF_PROFILE = (
    "rvv-v1-typed-plain-compare-select-leaf-profile.v1"
)
PLAIN_COMPARE_SELECT_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-plain-compare-select-plan-validated"
)
PLAIN_COMPARE_SELECT_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
PLAIN_COMPARE_SELECT_C_TYPE_MAPPING = (
    "vl:size_t,lhs/rhs:typed-vector,mask:typed-mask,result:typed-vector"
)
PLAIN_COMPARE_SELECT_LAYOUT = "select-lhs-when-mask-else-rhs"
COMPUTED_MASK_SELECT_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:computed_mask_select.v1"
)
COMPUTED_MASK_SELECT_ROUTE_FAMILY_PLAN = (
    "rvv-computed-mask-select-route-family-plan.v1"
)
COMPUTED_MASK_SELECT_VECTOR_COMPARE_PRODUCER_SOURCE = "vector-compare-rhs-load"
COMPUTED_MASK_SELECT_RUNTIME_SCALAR_PRODUCER_SOURCE = (
    "runtime-scalar-splat-compare-rhs"
)
COMPUTED_MASK_SELECT_DUAL_RUNTIME_SCALAR_PRODUCER_SOURCE = (
    "dual-runtime-scalar-splat-compare-rhs-mask-and"
)
COMPUTED_MASK_MEMORY_ROUTE_FAMILY_PLAN = (
    "rvv-computed-mask-memory-route-family-plan.v1"
)
COMPUTED_MASK_MEMORY_VECTOR_COMPARE_PRODUCER_SOURCE = "vector-compare-rhs-load"
COMPUTED_MASK_MEMORY_RUNTIME_SCALAR_PRODUCER_SOURCE = (
    "runtime-scalar-splat-compare-rhs"
)
COMPUTED_MASK_SELECT_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:computed_mask_select.v1;"
    "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;"
    "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs|cmp-call|hdr;"
    "true_value=true-value-input-buffer:true_value:abi|true-load|sel-true|hdr;"
    "false_value=false-value-input-buffer:false_value:abi|false-load|sel-false|hdr;"
    "out=output-buffer:out:abi|store|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop|hdr"
)
RUNTIME_SCALAR_CMP_SELECT_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:runtime_scalar_cmp_select.v1"
)
RUNTIME_SCALAR_CMP_SELECT_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:runtime_scalar_cmp_select.v1;"
    "lhs=lhs-input-buffer:lhs:abi|lhs-load|cmp-lhs|hdr;"
    "rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr;"
    "true_value=true-value-input-buffer:true_value:abi|true-load|sel-true|hdr;"
    "false_value=false-value-input-buffer:false_value:abi|false-load|sel-false|hdr;"
    "out=output-buffer:out:abi|store|hdr;"
    "n=runtime-element-count:n:abi|setvl|loop|hdr"
)
RUNTIME_SCALAR_CMP_MASKED_STORE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:runtime_scalar_cmp_masked_store.v1"
)
RUNTIME_SCALAR_CMP_MASKED_STORE_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:runtime_scalar_cmp_masked_store.v1;"
    "lhs=lhs-input-buffer:lhs:abi|lhs-load|cmp-lhs|hdr;"
    "rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr;"
    "src=source-input-buffer:src:abi|src-load|mstore-src|hdr;"
    "dst=output-buffer:dst:abi|mstore-base|mstore-dst|hdr;"
    "n=runtime-element-count:n:abi|setvl|loop|hdr"
)
RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:runtime_scalar_cmp_masked_load_store.v1"
)
RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:runtime_scalar_cmp_masked_load_store.v1;"
    "lhs=lhs-input-buffer:lhs:abi|lhs-load|cmp-lhs|hdr;"
    "rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr;"
    "src=source-input-buffer:src:abi|mload-base|mload-src|hdr;"
    "dst=output-buffer:dst:abi|old-dst-load|mload-pass|store|hdr;"
    "n=runtime-element-count:n:abi|setvl|loop|hdr"
)
MASKED_ADD_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:masked_add.v1"
)
MASKED_ADD_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:masked_add.v1;"
    "lhs=lhs-input-buffer:lhs:abi|load-base|compare-lhs-call|masked-add-lhs-call|masked-merge-passthrough-call;"
    "rhs=rhs-input-buffer:rhs:abi|load-base|compare-rhs-call|masked-add-rhs-call;"
    "out=output-buffer:out:abi|store-base|header;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|header"
)


def scalar_broadcast_route_operand_binding_plan(kind: str) -> str:
    if kind not in SCALAR_BROADCAST_OP_KINDS:
        raise EvidenceError(f"unsupported scalar broadcast op kind: {kind}")
    return f"rvv-route-operand-binding:{kind}.v1"


def scalar_broadcast_route_operand_binding_operands(kind: str) -> str:
    plan = scalar_broadcast_route_operand_binding_plan(kind)
    return (
        f"{plan};"
        "lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|scalar-broadcast-lhs-call|header-mirror;"
        "rhs_scalar=rhs-scalar-value:rhs_scalar:runtime-abi-mirror|scalar-broadcast-rhs-call|header-mirror;"
        "out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;"
        "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"
    )


RUNTIME_SCALAR_SPLAT_STORE_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:runtime_i32_splat_store.v1"
)
RUNTIME_SCALAR_SPLAT_STORE_ROUTE_FAMILY_PLAN = (
    "rvv-runtime-scalar-splat-store-route-family-plan.v1"
)
WIDENING_CONVERSION_ROUTE_FAMILY_PLAN = (
    "rvv-widening-conversion-route-family-plan.v1"
)
RUNTIME_SCALAR_SPLAT_STORE_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:runtime_i32_splat_store.v1;"
    "rhs_scalar=rhs-scalar-value:rhs_scalar:runtime-abi-mirror|runtime-scalar-splat-call|header-mirror;"
    "out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"
)


def masked_elementwise_route_operand_binding_plan(kind: str) -> str:
    if kind not in MASKED_ELEMENTWISE_OP_KINDS:
        raise EvidenceError(f"unsupported masked elementwise op kind: {kind}")
    return f"rvv-route-operand-binding:{kind}.v1"


def masked_elementwise_materialized_use_prefix(kind: str) -> str:
    if kind not in MASKED_ELEMENTWISE_OP_KINDS:
        raise EvidenceError(f"unsupported masked elementwise op kind: {kind}")
    return kind.replace("_", "-")


def masked_elementwise_route_operand_binding_operands(kind: str) -> str:
    plan = masked_elementwise_route_operand_binding_plan(kind)
    prefix = masked_elementwise_materialized_use_prefix(kind)
    return (
        f"{plan};"
        f"lhs=lhs-input-buffer:lhs:abi|load-base|compare-lhs-call|{prefix}-lhs-call|masked-merge-passthrough-call;"
        f"rhs=rhs-input-buffer:rhs:abi|load-base|compare-rhs-call|{prefix}-rhs-call;"
        "out=output-buffer:out:abi|store-base|header;"
        "n=runtime-element-count:n:abi|setvl-avl|loop-control|header"
    )


REDUCE_ADD_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:reduce_add.v1"
)
REDUCE_ADD_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:reduce_add.v1;"
    "lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|reduction-input-call;"
    "rhs=rhs-input-buffer:rhs:runtime-abi-mirror|materialized-accumulator-load-base|reduction-accumulator-call;"
    "out=output-buffer:out:runtime-abi-mirror|materialized-store-base|reduction-result-store|header-mirror;"
    "n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"
)
STRIDED_ADD_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:strided_add.v1"
)
STRIDED_ADD_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:strided_add.v1;"
    "lhs=lhs-input-buffer:lhs:abi|lhs-load-base|binary-lhs-call;"
    "rhs=rhs-input-buffer:rhs:abi|rhs-load-base|binary-rhs-call;"
    "out=output-buffer:out:abi|store-base|header;"
    "n=runtime-element-count:n:abi|setvl-avl|loop-control|header;"
    "lhs_stride=lhs-input-stride:lhs_stride:abi|lhs-load-stride|lhs-byte-addr|header;"
    "rhs_stride=rhs-input-stride:rhs_stride:abi|rhs-load-stride|rhs-byte-addr|header;"
    "out_stride=output-stride:out_stride:abi|store-stride|out-byte-addr|header"
)
SCALAR_BROADCAST_ADD_RUNTIME_ABI_ORDER = "lhs,rhs_scalar,out,n"
RUNTIME_SCALAR_SPLAT_STORE_RUNTIME_ABI_ORDER = "rhs_scalar,out,n"
RUNTIME_SCALAR_SPLAT_STORE_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-runtime-scalar-splat-store-leaf-profile.v1"
)
RUNTIME_SCALAR_SPLAT_STORE_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-runtime-scalar-splat-store-plan-validated"
)
RUNTIME_SCALAR_SPLAT_STORE_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
RUNTIME_SCALAR_SPLAT_STORE_C_TYPE_MAPPING = (
    "vl:size_t,rhs_scalar:i32,result:signed-e32m1"
)
WIDEN_I32_TO_I64_TARGET_LEAF_PROFILE = (
    "rvv-v1-i32m1-i64m2-widening-conversion-leaf-profile.v1"
)
WIDEN_I16_TO_I32_TARGET_LEAF_PROFILE = (
    "rvv-v1-i16mf2-i32m1-widening-conversion-leaf-profile.v1"
)
WIDEN_I32_TO_I64_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-widen-i32-to-i64-plan-validated"
)
WIDEN_I16_TO_I32_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-widen-i16-to-i32-plan-validated"
)
WIDENING_CONVERSION_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
WIDEN_I32_TO_I64_C_TYPE_MAPPING = (
    "vl:size_t,source:signed-e32m1,result:signed-e64m2"
)
WIDEN_I16_TO_I32_C_TYPE_MAPPING = (
    "vl:size_t,source:signed-e16mf2,result:signed-e32m1"
)
BASE_MEMORY_MOVEMENT_ROUTE_FAMILY_PLAN = (
    "rvv-base-memory-movement-route-family-plan.v1"
)
BASE_MEMORY_REQUIRED_HEADER_DECLARATIONS = "stddef.h,stdint.h,riscv_vector.h"
BASE_MEMORY_TARGET_LEAF_PROFILE_BY_KIND = {
    "strided_load_unit_store": "rvv-v1-e32m1-strided-load-unit-store-leaf-profile.v1",
    "unit_load_strided_store": "rvv-v1-e32m1-unit-load-strided-store-leaf-profile.v1",
    "indexed_gather_unit_store": "rvv-v1-e32m1-indexed-gather-unit-store-leaf-profile.v1",
    "indexed_scatter_unit_load": "rvv-v1-e32m1-indexed-scatter-unit-load-leaf-profile.v1",
    "masked_unit_load_store": "rvv-v1-e32m1-masked-unit-load-store-leaf-profile.v1",
    "masked_unit_store": "rvv-v1-e32m1-masked-unit-store-leaf-profile.v1",
}
BASE_MEMORY_PROVIDER_SUPPORTED_MIRROR_BY_KIND = {
    "strided_load_unit_store": "provider_supported_mirror:rvv-strided-load-unit-store-plan-validated",
    "unit_load_strided_store": "provider_supported_mirror:rvv-unit-load-strided-store-plan-validated",
    "indexed_gather_unit_store": "provider_supported_mirror:rvv-indexed-gather-unit-store-plan-validated",
    "indexed_scatter_unit_load": "provider_supported_mirror:rvv-indexed-scatter-unit-load-plan-validated",
    "masked_unit_load_store": "provider_supported_mirror:rvv-masked-unit-load-store-plan-validated",
    "masked_unit_store": "provider_supported_mirror:rvv-masked-unit-store-plan-validated",
}
BASE_MEMORY_C_TYPE_MAPPING_BY_KIND = {
    "strided_load_unit_store": "vl:size_t,source:byte-strided-e32m1,result:signed-e32m1",
    "unit_load_strided_store": "vl:size_t,source:signed-e32m1,destination:byte-strided-e32m1",
    "indexed_gather_unit_store": "vl:size_t,data:signed-e32m1,index:u32m1,result:signed-e32m1",
    "indexed_scatter_unit_load": "vl:size_t,source:signed-e32m1,index:u32m1,destination:indexed-e32m1",
    "masked_unit_load_store": "vl:size_t,source/passthrough:signed-e32m1,mask:b32,result:masked-load-store",
    "masked_unit_store": "vl:size_t,source:signed-e32m1,mask:b32,destination:masked-store",
}
RUNTIME_SCALAR_CMP_SELECT_RUNTIME_ABI_ORDER = (
    "lhs,rhs_scalar,true_value,false_value,out,n"
)
RUNTIME_SCALAR_CMP_SELECT_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-runtime-scalar-cmp-select-leaf-profile.v1"
)
RUNTIME_SCALAR_CMP_SELECT_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-runtime-scalar-cmp-select-plan-validated"
)
RUNTIME_SCALAR_CMP_SELECT_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
RUNTIME_SCALAR_CMP_SELECT_C_TYPE_MAPPING = (
    "vl:size_t,lhs:signed-e32m1,rhs_scalar:i32,mask:b32,"
    "true_false:signed-e32m1,result:signed-e32m1"
)
RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_RUNTIME_ABI_ORDER = (
    "cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n"
)
RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-runtime-scalar-dual-cmp-mask-and-select-leaf-profile.v1"
)
RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-runtime-scalar-dual-cmp-mask-and-select-plan-validated"
)
RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_C_TYPE_MAPPING = (
    "vl:size_t,cmp_lhs_a:signed-e32m1,rhs_scalar_a:i32,"
    "cmp_lhs_b:signed-e32m1,rhs_scalar_b:i32,mask_a:b32,mask_b:b32,"
    "mask_and:b32,true_false:signed-e32m1,result:signed-e32m1"
)
RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_MEMORY_LAYOUT = (
    "unit-stride-dual-lhs-runtime-scalar-thresholds-mask-and-true-false-select-"
    "output-runtime-abi"
)
RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:runtime_scalar_dual_cmp_mask_and_select.v1"
)
RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_ROUTE_OPERAND_BINDING_OPERANDS = (
    "cmp_lhs_a=abi|load|cmp|and;"
    "rhs_scalar_a=abi|splat|cmp;"
    "cmp_lhs_b=abi|load|cmp|and;"
    "rhs_scalar_b=abi|splat|cmp;"
    "true_value=abi|load|sel;"
    "false_value=abi|load|sel;"
    "out=abi|store|hdr;"
    "n=abi|setvl|loop|hdr"
)
RUNTIME_SCALAR_CMP_MASKED_STORE_RUNTIME_ABI_ORDER = "lhs,rhs_scalar,src,dst,n"
RUNTIME_SCALAR_CMP_MASKED_STORE_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-runtime-scalar-cmp-masked-store-leaf-profile.v1"
)
RUNTIME_SCALAR_CMP_MASKED_STORE_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-store-plan-validated"
)
RUNTIME_SCALAR_CMP_MASKED_STORE_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
RUNTIME_SCALAR_CMP_MASKED_STORE_C_TYPE_MAPPING = (
    "vl:size_t,lhs_payload:signed-e32m1,rhs_scalar:i32,mask:b32,dst:masked-store"
)
RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_RUNTIME_ABI_ORDER = "lhs,rhs_scalar,src,dst,n"
RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-runtime-scalar-cmp-masked-load-store-leaf-profile.v1"
)
RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-load-store-plan-validated"
)
RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_C_TYPE_MAPPING = (
    "vl:size_t,lhs/source/passthrough:signed-e32m1,rhs_scalar:i32,"
    "mask:b32,result:masked-load-store"
)
COMPUTED_MASK_UNIT_LOAD_STORE_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-computed-mask-unit-load-store-leaf-profile.v1"
)
COMPUTED_MASK_UNIT_LOAD_STORE_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-computed-mask-unit-load-store-plan-validated"
)
COMPUTED_MASK_UNIT_LOAD_STORE_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
COMPUTED_MASK_UNIT_LOAD_STORE_C_TYPE_MAPPING = (
    "vl:size_t,compare/source/passthrough:signed-e32m1,mask:b32,"
    "result:masked-load-store"
)
COMPUTED_MASK_STRIDED_STORE_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-computed-mask-strided-store-leaf-profile.v1"
)
COMPUTED_MASK_STRIDED_STORE_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-computed-mask-strided-store-plan-validated"
)
COMPUTED_MASK_STRIDED_STORE_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
COMPUTED_MASK_STRIDED_STORE_C_TYPE_MAPPING = (
    "vl:size_t,compare/source:signed-e32m1,mask:b32,dst:masked-strided-store"
)
COMPUTED_MASK_STRIDED_LOAD_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-computed-mask-strided-load-leaf-profile.v1"
)
COMPUTED_MASK_STRIDED_LOAD_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-computed-mask-strided-load-plan-validated"
)
COMPUTED_MASK_STRIDED_LOAD_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
COMPUTED_MASK_STRIDED_LOAD_C_TYPE_MAPPING = (
    "vl:size_t,compare/source/passthrough:signed-e32m1,mask:b32,"
    "result:masked-strided-load-store"
)
COMPUTED_MASK_INDEXED_GATHER_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-computed-mask-indexed-gather-load-leaf-profile.v1"
)
COMPUTED_MASK_INDEXED_GATHER_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-computed-mask-indexed-gather-load-plan-validated"
)
COMPUTED_MASK_INDEXED_GATHER_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
COMPUTED_MASK_INDEXED_GATHER_C_TYPE_MAPPING = (
    "vl:size_t,compare/source/passthrough:signed-e32m1,index:u32m1,"
    "mask:b32,result:masked-indexed-load-store"
)
COMPUTED_MASK_INDEXED_SCATTER_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-computed-mask-indexed-scatter-store-leaf-profile.v1"
)
COMPUTED_MASK_INDEXED_SCATTER_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-computed-mask-indexed-scatter-store-plan-validated"
)
COMPUTED_MASK_INDEXED_SCATTER_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
COMPUTED_MASK_INDEXED_SCATTER_C_TYPE_MAPPING = (
    "vl:size_t,compare/source:signed-e32m1,index:u32m1,mask:b32,"
    "dst:masked-indexed-store"
)
COMPUTED_MASK_SEGMENT2_LOAD_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-computed-mask-segment2-load-leaf-profile.v1"
)
COMPUTED_MASK_SEGMENT2_LOAD_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-computed-mask-segment2-load-plan-validated"
)
COMPUTED_MASK_SEGMENT2_LOAD_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
COMPUTED_MASK_SEGMENT2_LOAD_C_TYPE_MAPPING = (
    "vl:size_t,compare/source/passthrough-fields:signed-e32m1,mask:b32,"
    "segment2:vint32m1x2,result:masked-segment2-load-store"
)
COMPUTED_MASK_SEGMENT2_STORE_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-computed-mask-segment2-store-leaf-profile.v1"
)
COMPUTED_MASK_SEGMENT2_STORE_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-computed-mask-segment2-store-plan-validated"
)
COMPUTED_MASK_SEGMENT2_STORE_REQUIRED_HEADER_DECLARATIONS = (
    "stddef.h,stdint.h,riscv_vector.h"
)
COMPUTED_MASK_SEGMENT2_STORE_C_TYPE_MAPPING = (
    "vl:size_t,compare/field-payloads:signed-e32m1,mask:b32,"
    "segment2:vint32m1x2,dst:masked-segment2-store"
)
WIDENING_CONVERSION_RUNTIME_ABI_ORDER = "lhs,out,n"
WIDENING_CONVERSION_RELATION = "signed-i32m1-to-i64m2"
WIDEN_I16_TO_I32_CONVERSION_RELATION = "signed-i16mf2-to-i32m1"
WIDEN_I32_TO_I64_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:widen_i32_to_i64.v1"
)
WIDEN_I32_TO_I64_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:widen_i32_to_i64.v1;"
    "lhs=lhs-input-buffer:lhs:abi|src-load|convert-src|src-i32m1|relation-signed-i32m1-to-i64m2|hdr;"
    "out=output-buffer:out:abi|res-store|convert-result|res-i64m2|relation-signed-i32m1-to-i64m2|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop|hdr"
)
WIDEN_I16_TO_I32_ROUTE_OPERAND_BINDING_PLAN = (
    "rvv-route-operand-binding:widen_i16_to_i32.v1"
)
WIDEN_I16_TO_I32_ROUTE_OPERAND_BINDING_OPERANDS = (
    "rvv-route-operand-binding:widen_i16_to_i32.v1;"
    "lhs=lhs-input-buffer:lhs:abi|src-load|convert-src|src-i16mf2|relation-signed-i16mf2-to-i32m1|hdr;"
    "out=output-buffer:out:abi|res-store|convert-result|res-i32m1|relation-signed-i16mf2-to-i32m1|hdr;"
    "n=runtime-element-count:n:abi|setvl-avl|loop|hdr"
)
STRIDED_ADD_MEMORY_LAYOUT = "element-strided-lhs-rhs-output-runtime-abi"
STRIDED_ADD_LHS_STRIDE_SOURCE = "runtime_abi:lhs_stride"
STRIDED_ADD_RHS_STRIDE_SOURCE = "runtime_abi:rhs_stride"
STRIDED_ADD_OUT_STRIDE_SOURCE = "runtime_abi:out_stride"
STRIDED_LOAD_UNIT_STORE_MEMORY_LAYOUT = (
    "byte-strided-source-unit-stride-output-runtime-abi"
)
STRIDED_LOAD_UNIT_STORE_SOURCE_STRIDE_SOURCE = "runtime_abi:stride_bytes"
STRIDED_LOAD_UNIT_STORE_SOURCE_MEMORY_FORM = "strided-load"
STRIDED_LOAD_UNIT_STORE_DESTINATION_MEMORY_FORM = "unit-stride-store"
UNIT_LOAD_STRIDED_STORE_MEMORY_LAYOUT = (
    "unit-stride-source-byte-strided-destination-runtime-abi"
)
UNIT_LOAD_STRIDED_STORE_DESTINATION_STRIDE_SOURCE = (
    "runtime_abi:dst_stride_bytes"
)
COMPUTED_MASK_STRIDED_STORE_DESTINATION_STRIDE_SOURCE = (
    "runtime_abi:dst_stride_bytes"
)
UNIT_LOAD_STRIDED_STORE_SOURCE_MEMORY_FORM = "unit-stride-load"
UNIT_LOAD_STRIDED_STORE_DESTINATION_MEMORY_FORM = "strided-store"
INDEXED_GATHER_RUNTIME_ABI_ORDER = "data,index,out,n"
INDEXED_GATHER_MEMORY_LAYOUT = "element-indexed-data-index-unit-stride-output-runtime-abi"
INDEXED_GATHER_INDEX_SOURCE = "runtime_abi:index"
INDEXED_GATHER_INDEX_EEW = "32"
INDEXED_GATHER_OFFSET_UNIT = "element"
INDEXED_GATHER_DATA_MEMORY_FORM = "indexed-load"
INDEXED_GATHER_DESTINATION_MEMORY_FORM = "unit-stride-store"
INDEXED_SCATTER_RUNTIME_ABI_ORDER = "src,index,dst,n"
INDEXED_SCATTER_MEMORY_LAYOUT = "unit-stride-source-indexed-destination-index-runtime-abi"
INDEXED_SCATTER_INDEX_SOURCE = "runtime_abi:index"
INDEXED_SCATTER_INDEX_EEW = "32"
INDEXED_SCATTER_OFFSET_UNIT = "element"
INDEXED_SCATTER_INDEX_UNIQUENESS = "unique"
INDEXED_SCATTER_SOURCE_MEMORY_FORM = "unit-stride-load"
INDEXED_SCATTER_INDEXED_DESTINATION_MEMORY_FORM = "indexed-store"
INDEXED_SCATTER_DESTINATION_MEMORY_FORM = "indexed-store"
MASKED_MEMORY_RUNTIME_ABI_ORDER = "src,mask,dst,n"
MASKED_MEMORY_LAYOUT = "unit-stride-source-mask-old-destination-runtime-abi"
MASKED_MEMORY_MASK_ROLE = "predicate-mask-input-buffer"
MASKED_MEMORY_MASK_SOURCE = "runtime_abi:mask"
MASKED_MEMORY_MASK_FORM = "unit-stride-mask-load"
MASKED_MEMORY_INACTIVE_LANE_CONTRACT = "masked-off-lanes-preserve-old-destination"
MASKED_MEMORY_PASSTHROUGH_LAYOUT = "old-destination-vector-preserves-inactive-lanes"
MASKED_MEMORY_SOURCE_MEMORY_FORM = "unit-stride-load"
MASKED_MEMORY_DESTINATION_MEMORY_FORM = "unit-stride-store"
MASKED_STORE_LAYOUT = "unit-stride-source-mask-destination-masked-store-runtime-abi"
MASKED_STORE_INACTIVE_LANE_CONTRACT = "masked-store-false-lanes-preserve-output-buffer"
MASKED_STORE_PASSTHROUGH_LAYOUT = "masked-store-has-no-passthrough-load"
MASKED_STORE_DESTINATION_MEMORY_FORM = "masked-unit-store"
RUNTIME_SCALAR_CMP_MASKED_STORE_MEMORY_LAYOUT = (
    "unit-stride-lhs-runtime-scalar-threshold-source-masked-destination-runtime-abi"
)
RUNTIME_SCALAR_CMP_MASKED_STORE_SOURCE_MEMORY_FORM = "unit-stride-load"
RUNTIME_SCALAR_CMP_MASKED_STORE_DESTINATION_MEMORY_FORM = "masked-unit-store"
RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_MEMORY_LAYOUT = (
    "unit-stride-lhs-runtime-scalar-threshold-source-old-destination-runtime-abi"
)
COMPUTED_MASK_MEMORY_RUNTIME_ABI_ORDER = "cmp_lhs,cmp_rhs,src,dst,n"
COMPUTED_MASK_MEMORY_LAYOUT = (
    "unit-stride-compare-source-old-destination-runtime-abi"
)
COMPUTED_MASK_SELECT_RUNTIME_ABI_ORDER = (
    "cmp_lhs,cmp_rhs,true_value,false_value,out,n"
)
COMPUTED_MASK_SELECT_MEMORY_LAYOUT = (
    "unit-stride-compare-true-false-select-output-runtime-abi"
)
COMPUTED_MASK_SELECT_SOURCE_MEMORY_FORM = "unit-stride-load"
COMPUTED_MASK_SELECT_DESTINATION_MEMORY_FORM = "unit-stride-store"
COMPUTED_MASK_SELECT_LAYOUT = "select-true-value-when-mask-else-false-value"
COMPUTED_MASK_STRIDED_STORE_RUNTIME_ABI_ORDER = (
    "cmp_lhs,cmp_rhs,src,dst,n,dst_stride_bytes"
)
COMPUTED_MASK_STRIDED_LOAD_RUNTIME_ABI_ORDER = (
    "cmp_lhs,cmp_rhs,src,dst,n,src_stride_bytes"
)
COMPUTED_MASK_INDEXED_GATHER_RUNTIME_ABI_ORDER = (
    "cmp_lhs,cmp_rhs,src,index,dst,n"
)
COMPUTED_MASK_INDEXED_SCATTER_RUNTIME_ABI_ORDER = (
    "cmp_lhs,cmp_rhs,src,index,dst,n"
)
COMPUTED_MASK_SEGMENT2_LOAD_RUNTIME_ABI_ORDER = (
    "cmp_lhs,cmp_rhs,src,out0,out1,n"
)
COMPUTED_MASK_SEGMENT2_STORE_RUNTIME_ABI_ORDER = (
    "cmp_lhs,cmp_rhs,src0,src1,dst,n"
)
COMPUTED_MASK_STRIDED_STORE_MEMORY_LAYOUT = (
    "unit-stride-compare-source-byte-strided-masked-destination-runtime-abi"
)
COMPUTED_MASK_STRIDED_LOAD_MEMORY_LAYOUT = (
    "unit-stride-compare-byte-strided-masked-source-old-destination-runtime-abi"
)
COMPUTED_MASK_INDEXED_GATHER_MEMORY_LAYOUT = (
    "unit-stride-compare-indexed-masked-source-old-destination-runtime-abi"
)
COMPUTED_MASK_INDEXED_SCATTER_MEMORY_LAYOUT = (
    "unit-stride-compare-source-indexed-masked-destination-runtime-abi"
)
COMPUTED_MASK_SEGMENT2_LOAD_MEMORY_LAYOUT = (
    "unit-stride-compare-segment2-masked-source-old-fields-destination-runtime-abi"
)
COMPUTED_MASK_STRIDED_LOAD_SOURCE_STRIDE_SOURCE = (
    "runtime_abi:src_stride_bytes"
)
COMPUTED_MASK_STRIDED_LOAD_SOURCE_MEMORY_FORM = "masked-strided-load"
COMPUTED_MASK_INDEXED_GATHER_INDEX_SOURCE = "runtime_abi:index"
COMPUTED_MASK_INDEXED_GATHER_INDEX_EEW = "32"
COMPUTED_MASK_INDEXED_GATHER_OFFSET_UNIT = "element"
COMPUTED_MASK_INDEXED_GATHER_SOURCE_MEMORY_FORM = "masked-indexed-load"
COMPUTED_MASK_INDEXED_GATHER_DATA_MEMORY_FORM = "masked-indexed-load"
COMPUTED_MASK_INDEXED_SCATTER_INDEX_SOURCE = "runtime_abi:index"
COMPUTED_MASK_INDEXED_SCATTER_INDEX_EEW = "32"
COMPUTED_MASK_INDEXED_SCATTER_OFFSET_UNIT = "element"
COMPUTED_MASK_INDEXED_SCATTER_INDEX_UNIQUENESS = "unique"
COMPUTED_MASK_INDEXED_SCATTER_SOURCE_MEMORY_FORM = "unit-stride-load"
COMPUTED_MASK_INDEXED_SCATTER_DESTINATION_MEMORY_FORM = "masked-indexed-store"
COMPUTED_MASK_INDEXED_SCATTER_INDEXED_DESTINATION_MEMORY_FORM = (
    "masked-indexed-store"
)
COMPUTED_MASK_SEGMENT2_SOURCE_MEMORY_FORM = "segment2-interleaved-unit-stride-load"
COMPUTED_MASK_SEGMENT2_DESTINATION_MEMORY_FORM = "unit-stride-store"
COMPUTED_MASK_SEGMENT2_STORE_MEMORY_LAYOUT = (
    "unit-stride-compare-field-payloads-segment2-masked-destination-runtime-abi"
)
COMPUTED_MASK_SEGMENT2_STORE_SOURCE_MEMORY_FORM = "unit-stride-load"
COMPUTED_MASK_SEGMENT2_STORE_DESTINATION_MEMORY_FORM = (
    "segment2-interleaved-unit-stride-store"
)
COMPUTED_MASK_SEGMENT2_STORE_INTRINSIC = "__riscv_vsseg2e32_v_i32m1x2_m"
COMPUTED_MASK_SEGMENT2_LOAD_INTRINSIC = "__riscv_vlseg2e32_v_i32m1x2_tumu"
COMPUTED_MASK_SEGMENT2_TUPLE_CREATE_INTRINSIC = "__riscv_vcreate_v_i32m1x2"
COMPUTED_MASK_STRIDED_STORE_INACTIVE_LANE_CONTRACT = (
    "masked-strided-store-false-lanes-preserve-output-buffer"
)
COMPUTED_MASK_STRIDED_STORE_PASSTHROUGH_LAYOUT = (
    "masked-strided-store-has-no-passthrough-load"
)
COMPUTED_MASK_STRIDED_STORE_DESTINATION_MEMORY_FORM = "masked-strided-store"
COMPUTED_MASK_INDEXED_SCATTER_INACTIVE_LANE_CONTRACT = (
    "masked-indexed-store-false-lanes-preserve-output-buffer"
)
COMPUTED_MASK_INDEXED_SCATTER_PASSTHROUGH_LAYOUT = (
    "masked-indexed-store-has-no-passthrough-load"
)
COMPUTED_MASK_MEMORY_MASK_ROLE = "predicate-mask-produced-by-compare"
COMPUTED_MASK_MEMORY_MASK_SOURCE = "compare-produced-mask-same-vl-scope"
COMPUTED_MASK_MEMORY_MASK_FORM = "compare-produced-mask"
SEGMENT2_MEMORY_ROUTE_FAMILY_PLAN = "rvv-segment2-memory-route-family-plan.v1"
SEGMENT2_DEINTERLEAVE_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-segment2-deinterleave-leaf-profile.v1"
)
SEGMENT2_INTERLEAVE_TARGET_LEAF_PROFILE = (
    "rvv-v1-e32m1-segment2-interleave-leaf-profile.v1"
)
SEGMENT2_DEINTERLEAVE_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-segment2-deinterleave-plan-validated"
)
SEGMENT2_INTERLEAVE_PROVIDER_SUPPORTED_MIRROR = (
    "provider_supported_mirror:rvv-segment2-interleave-plan-validated"
)
SEGMENT2_REQUIRED_HEADER_DECLARATIONS = "stddef.h,stdint.h,riscv_vector.h"
SEGMENT2_DEINTERLEAVE_C_TYPE_MAPPING = (
    "vl:size_t,segment2:vint32m1x2,field-outputs:signed-e32m1"
)
SEGMENT2_INTERLEAVE_C_TYPE_MAPPING = (
    "vl:size_t,field-inputs:signed-e32m1,segment2:vint32m1x2"
)
SEGMENT2_RUNTIME_ABI_ORDER = "src,out0,out1,n"
SEGMENT2_MEMORY_LAYOUT = (
    "segment2-interleaved-source-dual-unit-stride-destination-runtime-abi"
)
SEGMENT2_SOURCE_MEMORY_FORM = "segment2-interleaved-unit-stride-load"
SEGMENT2_DESTINATION_MEMORY_FORM = "unit-stride-store"
SEGMENT2_FIELD0_ROLE = "segment-field0-output-buffer"
SEGMENT2_FIELD1_ROLE = "segment-field1-output-buffer"
SEGMENT2_TUPLE_C_TYPE = "vint32m1x2_t"
SEGMENT2_LOAD_INTRINSIC = "__riscv_vlseg2e32_v_i32m1x2"
SEGMENT2_FIELD_EXTRACT_INTRINSIC = "__riscv_vget_v_i32m1x2_i32m1"
SEGMENT2_INTERLEAVE_RUNTIME_ABI_ORDER = "src0,src1,dst,n"
SEGMENT2_INTERLEAVE_MEMORY_LAYOUT = (
    "dual-unit-stride-source-segment2-interleaved-destination-runtime-abi"
)
SEGMENT2_FIELD_SOURCE_MEMORY_FORM = "unit-stride-load"
SEGMENT2_INTERLEAVED_DESTINATION_MEMORY_FORM = (
    "segment2-interleaved-unit-stride-store"
)
SEGMENT2_FIELD0_INPUT_ROLE = "segment-field0-input-buffer"
SEGMENT2_FIELD1_INPUT_ROLE = "segment-field1-input-buffer"
SEGMENT2_STORE_INTRINSIC = "__riscv_vsseg2e32_v_i32m1x2"
SEGMENT2_TUPLE_CREATE_INTRINSIC = "__riscv_vcreate_v_i32m1x2"
OUT_SENTINEL = "(int32_t)0x5a5a5a5a"
I64_OUT_SENTINEL = "(int64_t)0x5a5a5a5a5a5a5a5aLL"

INDEX_FILE_NAME = "tianchenrv-target-artifact-bundle.index"
EXPECTED_SELECTED_ROLE = "dispatch case"
EXPECTED_COMPONENT_GROUP = "rvv-generic-typed-body-materialized-emitc-bundle.v1"
EXPECTED_RUNTIME_ABI_KIND = "plugin-owned-runtime-abi"
EXPECTED_OBJECT_ROUTE = "rvv-generic-typed-body-emitc-route-family"
EXPECTED_HEADER_ROUTE = "rvv-generic-typed-body-emitc-route-family.header"
EXPECTED_OWNER = "rvv-plugin"
EXPECTED_OBJECT_KIND = "riscv-elf-relocatable-object"
EXPECTED_HEADER_KIND = "runtime-callable-c-header"


def is_standalone_reduce_kind(kind: str) -> bool:
    return kind in {
        "standalone_reduce_add",
        "standalone_reduce_min",
        "standalone_reduce_max",
    }


def standalone_reduce_dataflow_kind(kind: str) -> str:
    if not is_standalone_reduce_kind(kind):
        raise ValueError(f"not a standalone reduction kind: {kind}")
    return kind.removeprefix("standalone_reduce_")


def standalone_reduce_route_operand_binding_plan(kind: str) -> str:
    if not is_standalone_reduce_kind(kind):
        raise ValueError(f"not a standalone reduction kind: {kind}")
    return f"rvv-route-operand-binding:{kind}.v1"


def standalone_reduce_route_operand_binding_operands(kind: str) -> str:
    if not is_standalone_reduce_kind(kind):
        raise ValueError(f"not a standalone reduction kind: {kind}")
    return STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_OPERANDS_TEMPLATE.format(
        kind=kind
    )


def is_computed_mask_standalone_reduce_kind(kind: str) -> bool:
    return kind in {
        "computed_mask_standalone_reduce_add",
        "computed_mask_standalone_reduce_min",
        "computed_mask_standalone_reduce_max",
    }


def computed_mask_standalone_reduce_dataflow_kind(kind: str) -> str:
    if not is_computed_mask_standalone_reduce_kind(kind):
        raise ValueError(f"not a computed-mask standalone reduction kind: {kind}")
    return kind.removeprefix("computed_mask_standalone_reduce_")


def computed_mask_standalone_reduce_route_operand_binding_plan(kind: str) -> str:
    if not is_computed_mask_standalone_reduce_kind(kind):
        raise ValueError(f"not a computed-mask standalone reduction kind: {kind}")
    return f"rvv-route-operand-binding:{kind}.v1"


def computed_mask_standalone_reduce_inactive_use(kind: str) -> str:
    reduction_kind = computed_mask_standalone_reduce_dataflow_kind(kind)
    return "zero-inactive" if reduction_kind == "add" else "neutral-inactive"


def computed_mask_standalone_reduce_inactive_contract(kind: str) -> str:
    reduction_kind = computed_mask_standalone_reduce_dataflow_kind(kind)
    return (
        COMPUTED_MASK_STANDALONE_REDUCE_INACTIVE_ZEROING
        if reduction_kind == "add"
        else COMPUTED_MASK_STANDALONE_REDUCE_INACTIVE_NEUTRAL
    )


def computed_mask_standalone_reduce_route_operand_binding_operands(kind: str) -> str:
    if not is_computed_mask_standalone_reduce_kind(kind):
        raise ValueError(f"not a computed-mask standalone reduction kind: {kind}")
    return COMPUTED_MASK_STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_OPERANDS_TEMPLATE.format(
        kind=kind,
        inactive_use=computed_mask_standalone_reduce_inactive_use(kind),
    )


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
    compare_predicate_kind: str = ""
    out_initializer: str = OUT_SENTINEL
    source_initializer: str = "(int32_t)(900 + (int32_t)(index * 13))"
    true_value_initializer: str = "(int32_t)(1300 + (int32_t)(index * 11))"
    false_value_initializer: str = "(int32_t)(-1700 - (int32_t)(index * 13))"
    lmul: str = "m1"
    sew: str = "32"
    element_c_type: str = "int32_t"
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
        if self.is_strided_load_unit_store:
            return (
                f"void {self.function_name}(const int32_t *src, "
                "int32_t *out, size_t n, size_t stride_bytes);"
            )
        if self.is_unit_load_strided_store:
            return (
                f"void {self.function_name}(const int32_t *src, "
                "int32_t *dst, size_t n, size_t dst_stride_bytes);"
            )
        if self.is_indexed_gather_unit_store:
            return (
                f"void {self.function_name}(const int32_t *data, "
                "const uint32_t *index, int32_t *out, size_t n);"
            )
        if self.is_indexed_scatter_unit_load:
            return (
                f"void {self.function_name}(const int32_t *src, "
                "const uint32_t *index, int32_t *dst, size_t n);"
            )
        if self.is_masked_unit_load_store or self.is_masked_unit_store:
            return (
                f"void {self.function_name}(const int32_t *src, "
                "const int32_t *mask, int32_t *dst, size_t n);"
            )
        if self.is_computed_masked_unit_load_store:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "const int32_t *cmp_rhs, const int32_t *src, "
                "int32_t *dst, size_t n);"
            )
        if self.is_computed_mask_select:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "const int32_t *cmp_rhs, const int32_t *true_value, "
                "const int32_t *false_value, int32_t *out, size_t n);"
            )
        if self.is_runtime_scalar_compare_select:
            return (
                f"void {self.function_name}(const int32_t *lhs, "
                "int32_t rhs_scalar, const int32_t *true_value, "
                "const int32_t *false_value, int32_t *out, size_t n);"
            )
        if self.is_runtime_scalar_dual_compare_mask_and_select:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs_a, "
                "int32_t rhs_scalar_a, const int32_t *cmp_lhs_b, "
                "int32_t rhs_scalar_b, const int32_t *true_value, "
                "const int32_t *false_value, int32_t *out, size_t n);"
            )
        if (
            self.is_runtime_scalar_computed_mask_store
            or self.is_runtime_scalar_computed_mask_load_store
        ):
            return (
                f"void {self.function_name}(const int32_t *lhs, "
                "int32_t rhs_scalar, const int32_t *src, "
                "int32_t *dst, size_t n);"
            )
        if self.is_computed_masked_strided_store:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "const int32_t *cmp_rhs, const int32_t *src, "
                "int32_t *dst, size_t n, size_t dst_stride_bytes);"
            )
        if self.is_computed_masked_strided_load_unit_store:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "const int32_t *cmp_rhs, const int32_t *src, "
                "int32_t *dst, size_t n, size_t src_stride_bytes);"
            )
        if self.is_computed_masked_indexed_gather_load_unit_store:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "const int32_t *cmp_rhs, const int32_t *src, "
                "const uint32_t *index, int32_t *dst, size_t n);"
            )
        if self.is_computed_masked_indexed_scatter_store_unit_load:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "const int32_t *cmp_rhs, const int32_t *src, "
                "const uint32_t *index, int32_t *dst, size_t n);"
            )
        if self.is_computed_masked_segment2_load_unit_store:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "const int32_t *cmp_rhs, const int32_t *src, "
                "int32_t *out0, int32_t *out1, size_t n);"
            )
        if self.is_computed_masked_segment2_store_unit_load:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "const int32_t *cmp_rhs, const int32_t *src0, "
                "const int32_t *src1, int32_t *dst, size_t n);"
            )
        if self.is_computed_masked_widening_dot_reduce_add:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "const int32_t *cmp_rhs, const int16_t *lhs, "
                "const int16_t *rhs, const int32_t *acc, "
                "int32_t *out, size_t n);"
            )
        if self.is_computed_masked_strided_input_widening_dot_reduce_add:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "const int32_t *cmp_rhs, const int16_t *lhs, "
                "const int16_t *rhs, const int32_t *acc, "
                "int32_t *out, size_t n, size_t lhs_stride, "
                "size_t rhs_stride);"
            )
        if self.is_strided_input_widening_dot_reduce_add:
            return (
                f"void {self.function_name}(const int16_t *lhs, "
                "const int16_t *rhs, const int32_t *acc, int32_t *out, "
                "size_t n, size_t lhs_stride, size_t rhs_stride);"
            )
        if self.is_segment2_deinterleave_unit_store:
            return (
                f"void {self.function_name}(const int32_t *src, "
                "int32_t *out0, int32_t *out1, size_t n);"
            )
        if self.is_segment2_interleave_unit_load:
            return (
                f"void {self.function_name}(const int32_t *src0, "
                "const int32_t *src1, int32_t *dst, size_t n);"
            )
        if self.is_scalar_broadcast_elementwise:
            return (
                f"void {self.function_name}(const int32_t *lhs, "
                "int32_t rhs_scalar, int32_t *out, size_t n);"
            )
        if self.is_runtime_scalar_splat_store:
            return (
                f"void {self.function_name}(int32_t rhs_scalar, "
                "int32_t *out, size_t n);"
            )
        if self.is_standalone_reduce:
            return (
                f"void {self.function_name}(const int32_t *lhs, "
                "const int32_t *acc, int32_t *out, size_t n);"
            )
        if self.is_computed_mask_standalone_reduce:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "const int32_t *cmp_rhs, const int32_t *src, "
                "const int32_t *acc, int32_t *out, size_t n);"
            )
        if self.is_runtime_scalar_computed_mask_standalone_reduce:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "int32_t rhs_scalar, const int32_t *src, "
                "const int32_t *acc, int32_t *out, size_t n);"
            )
        if self.is_macc_add:
            return (
                f"void {self.function_name}(const int32_t *lhs, "
                "const int32_t *rhs, const int32_t *acc, "
                "int32_t *out, size_t n);"
            )
        if self.is_computed_masked_macc_add:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "const int32_t *cmp_rhs, const int32_t *lhs, "
                "const int32_t *rhs, const int32_t *acc, "
                "int32_t *out, size_t n);"
            )
        if self.is_runtime_scalar_computed_masked_macc_add:
            return (
                f"void {self.function_name}(const int32_t *cmp_lhs, "
                "int32_t rhs_scalar, const int32_t *lhs, "
                "const int32_t *rhs, const int32_t *acc, "
                "int32_t *out, size_t n);"
            )
        if self.is_widen_i32_to_i64:
            return (
                f"void {self.function_name}(const int32_t *lhs, "
                "int64_t *out, size_t n);"
            )
        if self.is_widen_i16_to_i32:
            return (
                f"void {self.function_name}(const int16_t *lhs, "
                "int32_t *out, size_t n);"
            )
        if self.is_widening_macc_add or self.is_widening_dot_reduce_add:
            return (
                f"void {self.function_name}(const int16_t *lhs, "
                "const int16_t *rhs, const int32_t *acc, "
                "int32_t *out, size_t n);"
            )
        return (
            f"void {self.function_name}(const {self.element_c_type} *lhs, "
            f"const {self.element_c_type} *rhs, "
            f"{self.element_c_type} *out, size_t n);"
        )

    @property
    def runtime_parameters(self) -> tuple[dict[str, str], ...]:
        if self.is_strided_add:
            return EXPECTED_STRIDED_RUNTIME_PARAMETERS
        if self.is_strided_load_unit_store:
            return EXPECTED_STRIDED_LOAD_UNIT_STORE_RUNTIME_PARAMETERS
        if self.is_unit_load_strided_store:
            return EXPECTED_UNIT_LOAD_STRIDED_STORE_RUNTIME_PARAMETERS
        if self.is_indexed_gather_unit_store:
            return EXPECTED_INDEXED_GATHER_RUNTIME_PARAMETERS
        if self.is_indexed_scatter_unit_load:
            return EXPECTED_INDEXED_SCATTER_RUNTIME_PARAMETERS
        if self.is_masked_unit_load_store or self.is_masked_unit_store:
            return EXPECTED_MASKED_MEMORY_RUNTIME_PARAMETERS
        if self.is_computed_masked_unit_load_store:
            return EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS
        if self.is_computed_mask_select:
            return EXPECTED_COMPUTED_MASK_SELECT_RUNTIME_PARAMETERS
        if self.is_runtime_scalar_compare_select:
            return EXPECTED_RUNTIME_SCALAR_CMP_SELECT_RUNTIME_PARAMETERS
        if self.is_runtime_scalar_dual_compare_mask_and_select:
            return EXPECTED_RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_RUNTIME_PARAMETERS
        if self.is_runtime_scalar_computed_mask_store:
            return EXPECTED_RUNTIME_SCALAR_CMP_MASKED_STORE_RUNTIME_PARAMETERS
        if self.is_runtime_scalar_computed_mask_load_store:
            return EXPECTED_RUNTIME_SCALAR_CMP_MASKED_STORE_RUNTIME_PARAMETERS
        if self.is_computed_masked_strided_store:
            return EXPECTED_COMPUTED_MASK_STRIDED_STORE_RUNTIME_PARAMETERS
        if self.is_computed_masked_strided_load_unit_store:
            return EXPECTED_COMPUTED_MASK_STRIDED_LOAD_RUNTIME_PARAMETERS
        if self.is_computed_masked_indexed_gather_load_unit_store:
            return EXPECTED_COMPUTED_MASK_INDEXED_GATHER_RUNTIME_PARAMETERS
        if self.is_computed_masked_indexed_scatter_store_unit_load:
            return EXPECTED_COMPUTED_MASK_INDEXED_SCATTER_RUNTIME_PARAMETERS
        if self.is_computed_masked_segment2_load_unit_store:
            return EXPECTED_COMPUTED_MASK_SEGMENT2_LOAD_RUNTIME_PARAMETERS
        if self.is_computed_masked_segment2_store_unit_load:
            return EXPECTED_COMPUTED_MASK_SEGMENT2_STORE_RUNTIME_PARAMETERS
        if self.is_computed_masked_widening_dot_reduce_add:
            return EXPECTED_COMPUTED_MASK_WIDENING_DOT_RUNTIME_PARAMETERS
        if self.is_computed_masked_strided_input_widening_dot_reduce_add:
            return EXPECTED_COMPUTED_MASK_STRIDED_INPUT_WIDENING_DOT_RUNTIME_PARAMETERS
        if self.is_strided_input_widening_dot_reduce_add:
            return EXPECTED_STRIDED_INPUT_WIDENING_DOT_RUNTIME_PARAMETERS
        if self.is_segment2_deinterleave_unit_store:
            return EXPECTED_SEGMENT2_RUNTIME_PARAMETERS
        if self.is_segment2_interleave_unit_load:
            return EXPECTED_SEGMENT2_INTERLEAVE_RUNTIME_PARAMETERS
        if self.is_scalar_broadcast_elementwise:
            return EXPECTED_SCALAR_BROADCAST_RUNTIME_PARAMETERS
        if self.is_runtime_scalar_splat_store:
            return EXPECTED_RUNTIME_SCALAR_SPLAT_STORE_RUNTIME_PARAMETERS
        if self.is_standalone_reduce:
            return EXPECTED_STANDALONE_REDUCE_RUNTIME_PARAMETERS
        if self.is_computed_mask_standalone_reduce:
            return EXPECTED_COMPUTED_MASK_STANDALONE_REDUCE_RUNTIME_PARAMETERS
        if self.is_runtime_scalar_computed_mask_standalone_reduce:
            return EXPECTED_RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_RUNTIME_PARAMETERS
        if self.is_widen_i32_to_i64:
            return EXPECTED_WIDENING_CONVERSION_RUNTIME_PARAMETERS
        if self.is_widen_i16_to_i32:
            return EXPECTED_WIDEN_I16_TO_I32_RUNTIME_PARAMETERS
        if self.is_macc_add:
            return EXPECTED_MACC_RUNTIME_PARAMETERS
        if self.is_computed_masked_macc_add:
            return EXPECTED_COMPUTED_MASKED_MACC_RUNTIME_PARAMETERS
        if self.is_runtime_scalar_computed_masked_macc_add:
            return EXPECTED_RUNTIME_SCALAR_COMPUTED_MASKED_MACC_RUNTIME_PARAMETERS
        if self.is_widening_macc_add or self.is_widening_dot_reduce_add:
            return EXPECTED_WIDENING_MACC_RUNTIME_PARAMETERS
        if self.is_i64_add:
            return EXPECTED_I64_RUNTIME_PARAMETERS
        return EXPECTED_RUNTIME_PARAMETERS

    @property
    def selected_body_operation(self) -> str:
        if self.is_i64_add or self.is_lmul_m2_add:
            return "add"
        if self.kind == "cmp_select_sle":
            return "cmp_select"
        if self.kind == "computed_mask_select_sle":
            return "computed_mask_select"
        return self.kind

    @property
    def runtime_abi_order(self) -> str:
        if self.is_strided_add:
            return STRIDED_ADD_RUNTIME_ABI_ORDER
        if self.is_strided_load_unit_store:
            return STRIDED_LOAD_UNIT_STORE_RUNTIME_ABI_ORDER
        if self.is_unit_load_strided_store:
            return UNIT_LOAD_STRIDED_STORE_RUNTIME_ABI_ORDER
        if self.is_indexed_gather_unit_store:
            return INDEXED_GATHER_RUNTIME_ABI_ORDER
        if self.is_indexed_scatter_unit_load:
            return INDEXED_SCATTER_RUNTIME_ABI_ORDER
        if self.is_masked_unit_load_store or self.is_masked_unit_store:
            return MASKED_MEMORY_RUNTIME_ABI_ORDER
        if self.is_computed_masked_unit_load_store:
            return COMPUTED_MASK_MEMORY_RUNTIME_ABI_ORDER
        if self.is_computed_mask_select:
            return COMPUTED_MASK_SELECT_RUNTIME_ABI_ORDER
        if self.is_runtime_scalar_compare_select:
            return RUNTIME_SCALAR_CMP_SELECT_RUNTIME_ABI_ORDER
        if self.is_runtime_scalar_dual_compare_mask_and_select:
            return RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_RUNTIME_ABI_ORDER
        if self.is_runtime_scalar_computed_mask_store:
            return RUNTIME_SCALAR_CMP_MASKED_STORE_RUNTIME_ABI_ORDER
        if self.is_runtime_scalar_computed_mask_load_store:
            return RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_RUNTIME_ABI_ORDER
        if self.is_computed_masked_strided_store:
            return COMPUTED_MASK_STRIDED_STORE_RUNTIME_ABI_ORDER
        if self.is_computed_masked_strided_load_unit_store:
            return COMPUTED_MASK_STRIDED_LOAD_RUNTIME_ABI_ORDER
        if self.is_computed_masked_indexed_gather_load_unit_store:
            return COMPUTED_MASK_INDEXED_GATHER_RUNTIME_ABI_ORDER
        if self.is_computed_masked_indexed_scatter_store_unit_load:
            return COMPUTED_MASK_INDEXED_SCATTER_RUNTIME_ABI_ORDER
        if self.is_computed_masked_segment2_load_unit_store:
            return COMPUTED_MASK_SEGMENT2_LOAD_RUNTIME_ABI_ORDER
        if self.is_computed_masked_segment2_store_unit_load:
            return COMPUTED_MASK_SEGMENT2_STORE_RUNTIME_ABI_ORDER
        if self.is_segment2_deinterleave_unit_store:
            return SEGMENT2_RUNTIME_ABI_ORDER
        if self.is_segment2_interleave_unit_load:
            return SEGMENT2_INTERLEAVE_RUNTIME_ABI_ORDER
        if self.is_scalar_broadcast_elementwise:
            return SCALAR_BROADCAST_ADD_RUNTIME_ABI_ORDER
        if self.is_runtime_scalar_splat_store:
            return RUNTIME_SCALAR_SPLAT_STORE_RUNTIME_ABI_ORDER
        if self.is_standalone_reduce:
            return STANDALONE_REDUCE_RUNTIME_ABI_ORDER
        if self.is_computed_mask_standalone_reduce:
            return COMPUTED_MASK_STANDALONE_REDUCE_RUNTIME_ABI_ORDER
        if self.is_runtime_scalar_computed_mask_standalone_reduce:
            return RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_RUNTIME_ABI_ORDER
        if self.is_macc_add:
            return MACC_ADD_RUNTIME_ABI_ORDER
        if self.is_computed_masked_macc_add:
            return COMPUTED_MASKED_MACC_ADD_RUNTIME_ABI_ORDER
        if self.is_runtime_scalar_computed_masked_macc_add:
            return RUNTIME_SCALAR_COMPUTED_MASKED_MACC_ADD_RUNTIME_ABI_ORDER
        if self.is_widen_i32_to_i64 or self.is_widen_i16_to_i32:
            return WIDENING_CONVERSION_RUNTIME_ABI_ORDER
        if self.is_widening_macc_add:
            return WIDENING_MACC_RUNTIME_ABI_ORDER
        if self.is_widening_dot_reduce_add:
            return WIDENING_DOT_RUNTIME_ABI_ORDER
        if self.is_strided_input_widening_dot_reduce_add:
            return STRIDED_INPUT_WIDENING_DOT_RUNTIME_ABI_ORDER
        if self.is_computed_masked_widening_dot_reduce_add:
            return COMPUTED_MASK_WIDENING_DOT_RUNTIME_ABI_ORDER
        if self.is_computed_masked_strided_input_widening_dot_reduce_add:
            return COMPUTED_MASK_STRIDED_INPUT_WIDENING_DOT_RUNTIME_ABI_ORDER
        return "lhs,rhs,out,n"

    @property
    def pass_marker(self) -> str:
        return f"tcrv_rvv_generated_bundle_abi_{self.kind}_ok"

    @property
    def is_pre_realized(self) -> bool:
        return self.input_mode == "pre-realized-selected-body"

    @property
    def supports_direct_pre_realized_route_entry(self) -> bool:
        return self.is_pre_realized and (
            self.is_cmp_select or self.is_strided_load_unit_store
        )

    @property
    def is_rhs_broadcast(self) -> bool:
        return self.input_mode == "rhs-broadcast-selected-body"

    @property
    def is_reduce_add(self) -> bool:
        return self.kind == "reduce_add"

    @property
    def is_cmp_select(self) -> bool:
        return self.kind in {"cmp_select", "cmp_select_sle"}

    @property
    def is_computed_mask_select(self) -> bool:
        return self.kind in {"computed_mask_select", "computed_mask_select_sle"}

    @property
    def is_runtime_scalar_compare_select(self) -> bool:
        return self.kind == "runtime_scalar_cmp_select"

    @property
    def is_runtime_scalar_dual_compare_mask_and_select(self) -> bool:
        return self.kind == "runtime_scalar_dual_cmp_mask_and_select"

    @property
    def is_runtime_scalar_computed_mask_store(self) -> bool:
        return self.kind == "runtime_scalar_cmp_masked_store"

    @property
    def is_runtime_scalar_computed_mask_load_store(self) -> bool:
        return self.kind == "runtime_scalar_cmp_masked_load_store"

    def compare_predicate_c_expression(self, lhs: str, rhs: str) -> str:
        if self.compare_predicate_kind == "eq":
            return f"{lhs} == {rhs}"
        if self.compare_predicate_kind == "slt":
            return f"{lhs} < {rhs}"
        if self.compare_predicate_kind == "sle":
            return f"{lhs} <= {rhs}"
        raise EvidenceError(
            f"{self.kind} requires a supported compare predicate kind"
        )

    @property
    def is_masked_add(self) -> bool:
        return self.kind == "masked_add"

    @property
    def is_masked_elementwise(self) -> bool:
        return self.kind in MASKED_ELEMENTWISE_OP_KINDS

    @property
    def is_macc_add(self) -> bool:
        return self.kind == "macc_add"

    @property
    def is_computed_masked_macc_add(self) -> bool:
        return self.kind == "computed_masked_macc_add"

    @property
    def is_runtime_scalar_computed_masked_macc_add(self) -> bool:
        return self.kind == "runtime_scalar_cmp_masked_macc_add"

    @property
    def is_widening_macc_add(self) -> bool:
        return self.kind == "widening_macc_add"

    @property
    def is_widening_dot_reduce_add(self) -> bool:
        return self.kind == "widening_dot_reduce_add"

    @property
    def is_computed_masked_widening_dot_reduce_add(self) -> bool:
        return self.kind == "computed_masked_widening_dot_reduce_add"

    @property
    def is_computed_masked_strided_input_widening_dot_reduce_add(self) -> bool:
        return self.kind == "computed_masked_strided_input_widening_dot_reduce_add"

    @property
    def is_strided_input_widening_dot_reduce_add(self) -> bool:
        return self.kind == "strided_input_widening_dot_reduce_add"

    @property
    def is_strided_add(self) -> bool:
        return self.kind == "strided_add"

    @property
    def is_strided_load_unit_store(self) -> bool:
        return self.kind == "strided_load_unit_store"

    @property
    def is_unit_load_strided_store(self) -> bool:
        return self.kind == "unit_load_strided_store"

    @property
    def is_indexed_gather_unit_store(self) -> bool:
        return self.kind == "indexed_gather_unit_store"

    @property
    def is_indexed_scatter_unit_load(self) -> bool:
        return self.kind == "indexed_scatter_unit_load"

    @property
    def is_masked_unit_load_store(self) -> bool:
        return self.kind == "masked_unit_load_store"

    @property
    def is_masked_unit_store(self) -> bool:
        return self.kind == "masked_unit_store"

    @property
    def is_computed_masked_unit_load_store(self) -> bool:
        return self.kind == "computed_masked_unit_load_store"

    @property
    def is_computed_masked_strided_store(self) -> bool:
        return self.kind == "computed_masked_strided_store"

    @property
    def is_computed_masked_strided_load_unit_store(self) -> bool:
        return self.kind == "computed_masked_strided_load_unit_store"

    @property
    def is_computed_masked_indexed_gather_load_unit_store(self) -> bool:
        return self.kind == "computed_masked_indexed_gather_load_unit_store"

    @property
    def is_computed_masked_indexed_scatter_store_unit_load(self) -> bool:
        return self.kind == "computed_masked_indexed_scatter_store_unit_load"

    @property
    def is_computed_masked_segment2_load_unit_store(self) -> bool:
        return self.kind == "computed_masked_segment2_load_unit_store"

    @property
    def is_computed_masked_segment2_store_unit_load(self) -> bool:
        return self.kind == "computed_masked_segment2_store_unit_load"

    @property
    def is_segment2_deinterleave_unit_store(self) -> bool:
        return self.kind == "segment2_deinterleave_unit_store"

    @property
    def is_segment2_interleave_unit_load(self) -> bool:
        return self.kind == "segment2_interleave_unit_load"

    @property
    def is_scalar_broadcast_add(self) -> bool:
        return self.kind == "scalar_broadcast_add"

    @property
    def is_scalar_broadcast_elementwise(self) -> bool:
        return self.kind in SCALAR_BROADCAST_OP_KINDS

    @property
    def is_runtime_scalar_splat_store(self) -> bool:
        return self.kind == "runtime_i32_splat_store"

    @property
    def is_standalone_reduce_add(self) -> bool:
        return self.kind == "standalone_reduce_add"

    @property
    def is_standalone_reduce_min(self) -> bool:
        return self.kind == "standalone_reduce_min"

    @property
    def is_standalone_reduce_max(self) -> bool:
        return self.kind == "standalone_reduce_max"

    @property
    def is_standalone_reduce(self) -> bool:
        return is_standalone_reduce_kind(self.kind)

    @property
    def standalone_reduction_kind(self) -> str:
        return standalone_reduce_dataflow_kind(self.kind)

    @property
    def is_computed_mask_standalone_reduce_add(self) -> bool:
        return self.kind == "computed_mask_standalone_reduce_add"

    @property
    def is_computed_mask_standalone_reduce_min(self) -> bool:
        return self.kind == "computed_mask_standalone_reduce_min"

    @property
    def is_computed_mask_standalone_reduce_max(self) -> bool:
        return self.kind == "computed_mask_standalone_reduce_max"

    @property
    def is_computed_mask_standalone_reduce(self) -> bool:
        return is_computed_mask_standalone_reduce_kind(self.kind)

    @property
    def computed_mask_standalone_reduction_kind(self) -> str:
        return computed_mask_standalone_reduce_dataflow_kind(self.kind)

    @property
    def is_runtime_scalar_computed_mask_standalone_reduce(self) -> bool:
        return self.kind == "runtime_scalar_cmp_masked_standalone_reduce_add"

    @property
    def is_i64_add(self) -> bool:
        return self.kind == "i64_add"

    @property
    def is_lmul_m2_add(self) -> bool:
        return self.kind == "lmul_m2_add"

    @property
    def is_widen_i32_to_i64(self) -> bool:
        return self.kind == "widen_i32_to_i64"

    @property
    def is_widen_i16_to_i32(self) -> bool:
        return self.kind == "widen_i16_to_i32"


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
    "widen_i32_to_i64": OpExpectation(
        kind="widen_i32_to_i64",
        input_path=Path(
            "test/Target/RVV/explicit-selected-body-artifact-widen-i32-to-i64.mlir"
        ),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_widen_i32_to_i64",
        external_abi_name="rvv-generic-widen-i32-to-i64-callable-c-abi.v1",
        function_name=(
            "tcrv_emitc_explicit_selected_body_widen_i32_to_i64_kernel_"
            "explicit_selected_body_rvv_widen_i32_to_i64"
        ),
        emitc_route="rvv-generic-widen-i32-to-i64-emitc-route",
        typed_compute_op="tcrv_rvv.widening_convert",
        memory_form="unit-stride-conversion",
        lhs_initializer="(int32_t)(((int)(index % 31) - 15) * 65537)",
        rhs_initializer="unused",
        expected_expression="(int64_t)lhs[index]",
        out_initializer=I64_OUT_SENTINEL,
        lmul="m2",
        sew="64",
        element_c_type="int64_t",
        config_contract="rvv-selected-body-sew64-lmul-m2-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew64-lmul-m2",
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
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0) "
            "? (int32_t)(41 + (int32_t)(index * 9)) "
            ": (int32_t)(-300 - (int32_t)(index * 7)))"
        ),
        expected_expression="(lhs[index] == rhs[index] ? lhs[index] : rhs[index])",
        compare_predicate_kind="eq",
    ),
    "cmp_select_sle": OpExpectation(
        kind="cmp_select_sle",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-cmp-select-sle.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_i32_cmp_select_sle",
        external_abi_name="rvv-generic-cmp-select-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_cmp_select_sle_kernel_explicit_selected_body_rvv_i32_cmp_select_sle",
        emitc_route="rvv-generic-cmp-select-emitc-route",
        typed_compute_op="tcrv_rvv.select",
        memory_form="vector-rhs-load",
        lhs_initializer=(
            "(int32_t)((index % 5) == 0 ? 10 : "
            "((index % 5) == 1 ? 11 : "
            "((index % 5) == 2 ? 12 : "
            "((index % 5) == 3 ? 15 : -4))))"
        ),
        rhs_initializer=(
            "(int32_t)((index % 5) == 0 ? 10 : "
            "((index % 5) == 1 ? 13 : "
            "((index % 5) == 2 ? 9 : "
            "((index % 5) == 3 ? 15 : -9))))"
        ),
        expected_expression="(lhs[index] <= rhs[index] ? lhs[index] : rhs[index])",
        compare_predicate_kind="sle",
    ),
    "computed_mask_select_sle": OpExpectation(
        kind="computed_mask_select_sle",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-computed-mask-select-sle.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_computed_mask_select_sle",
        external_abi_name="rvv-generic-computed-mask-select-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_computed_mask_select_sle_kernel_explicit_selected_body_rvv_computed_mask_select_sle",
        emitc_route="rvv-generic-computed-mask-select-emitc-route",
        typed_compute_op="tcrv_rvv.select",
        memory_form="computed-mask-vector-select",
        lhs_initializer=(
            "(int32_t)(((index % 5) == 0) ? 10 : "
            "((index % 5) == 1) ? 9 : "
            "((index % 5) == 2) ? 14 : "
            "((index % 5) == 3) ? -3 : -9)"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 5) == 0) ? 10 : "
            "((index % 5) == 1) ? 12 : "
            "((index % 5) == 2) ? 11 : "
            "((index % 5) == 3) ? -3 : -12)"
        ),
        true_value_initializer="(int32_t)(2100 + (int32_t)(index * 17))",
        false_value_initializer="(int32_t)(-2300 - (int32_t)(index * 19))",
        expected_expression=(
            "(cmp_lhs[index] <= cmp_rhs[index] ? true_value[index] : false_value[index])"
        ),
        compare_predicate_kind="sle",
    ),
    "runtime_scalar_cmp_select": OpExpectation(
        kind="runtime_scalar_cmp_select",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-select.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_runtime_scalar_cmp_select",
        external_abi_name="rvv-generic-runtime-scalar-cmp-select-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_runtime_scalar_cmp_select_kernel_explicit_selected_body_rvv_runtime_scalar_cmp_select",
        emitc_route="rvv-generic-runtime-scalar-cmp-select-emitc-route",
        typed_compute_op="tcrv_rvv.select",
        memory_form="runtime-scalar-compare-select",
        lhs_initializer=(
            "(int32_t)(((index % 5) == 0) ? -100 : "
            "((index % 5) == 1) ? -37 : "
            "((index % 5) == 2) ? 0 : "
            "((index % 5) == 3) ? 91 : 130)"
        ),
        rhs_initializer="rhs_scalar",
        true_value_initializer="(int32_t)(3100 + (int32_t)(index * 23))",
        false_value_initializer="(int32_t)(-4100 - (int32_t)(index * 29))",
        expected_expression=(
            "(lhs[index] <= rhs_scalar ? true_value[index] : false_value[index])"
        ),
        compare_predicate_kind="sle",
    ),
    "runtime_scalar_dual_cmp_mask_and_select": OpExpectation(
        kind="runtime_scalar_dual_cmp_mask_and_select",
        input_path=Path(
            "test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir"
        ),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_rvv_dual_cmp_mask_select",
        external_abi_name="rvv-generic-runtime-scalar-dual-cmp-mask-and-select-callable-c-abi.v1",
        function_name=(
            "tcrv_emitc_explicit_dual_cmp_mask_select_kernel_"
            "explicit_rvv_dual_cmp_mask_select"
        ),
        emitc_route="rvv-generic-runtime-scalar-dual-cmp-mask-and-select-emitc-route",
        typed_compute_op="tcrv_rvv.select",
        memory_form="runtime-scalar-dual-cmp-mask-and-select",
        lhs_initializer=(
            "(int32_t)(((index % 6) == 0) ? -100 : "
            "((index % 6) == 1) ? -37 : "
            "((index % 6) == 2) ? -12 : "
            "((index % 6) == 3) ? 0 : "
            "((index % 6) == 4) ? 91 : 130)"
        ),
        rhs_initializer="rhs_scalar_a",
        true_value_initializer="(int32_t)(5100 + (int32_t)(index * 31))",
        false_value_initializer="(int32_t)(-6100 - (int32_t)(index * 37))",
        expected_expression=(
            "((cmp_lhs_a[index] <= rhs_scalar_a && "
            "cmp_lhs_b[index] <= rhs_scalar_b) ? true_value[index] : "
            "false_value[index])"
        ),
        compare_predicate_kind="sle",
    ),
    "runtime_scalar_cmp_masked_store": OpExpectation(
        kind="runtime_scalar_cmp_masked_store",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-store.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_body_rvv_runtime_scalar_cmp_masked_store",
        external_abi_name="rvv-generic-runtime-scalar-cmp-masked-store-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_body_runtime_scalar_cmp_masked_store_kernel_explicit_body_rvv_runtime_scalar_cmp_masked_store",
        emitc_route="rvv-generic-runtime-scalar-cmp-masked-store-emitc-route",
        typed_compute_op="tcrv_rvv.masked_store",
        memory_form="runtime-scalar-computed-mask-store",
        lhs_initializer=(
            "(int32_t)(((index % 5) == 0) ? -120 : "
            "((index % 5) == 1) ? -37 : "
            "((index % 5) == 2) ? 0 : "
            "((index % 5) == 3) ? 91 : 130)"
        ),
        rhs_initializer="rhs_scalar",
        source_initializer="(int32_t)(5100 + (int32_t)(index * 31))",
        out_initializer="(int32_t)(-15000 - (int32_t)(index * 43))",
        expected_expression="(lhs[index] <= rhs_scalar ? src[index] : old_dst[index])",
        compare_predicate_kind="sle",
        config_contract="rvv-selected-body-sew32-lmul-m1-tail-undisturbed-mask-undisturbed.v1",
    ),
    "runtime_scalar_cmp_masked_load_store": OpExpectation(
        kind="runtime_scalar_cmp_masked_load_store",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-load-store.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_body_rvv_runtime_scalar_cmp_masked_load_store",
        external_abi_name="rvv-generic-runtime-scalar-cmp-masked-load-store-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_body_runtime_scalar_cmp_masked_load_store_kernel_explicit_body_rvv_runtime_scalar_cmp_masked_load_store",
        emitc_route="rvv-generic-runtime-scalar-cmp-masked-load-store-emitc-route",
        typed_compute_op="tcrv_rvv.masked_load",
        memory_form="runtime-scalar-computed-mask-load-store",
        lhs_initializer=(
            "(int32_t)(((index % 5) == 0) ? -120 : "
            "((index % 5) == 1) ? -37 : "
            "((index % 5) == 2) ? 0 : "
            "((index % 5) == 3) ? 91 : 130)"
        ),
        rhs_initializer="rhs_scalar",
        source_initializer="(int32_t)(6100 + (int32_t)(index * 37))",
        out_initializer="(int32_t)(-17000 - (int32_t)(index * 47))",
        expected_expression="(lhs[index] <= rhs_scalar ? src[index] : old_dst[index])",
        compare_predicate_kind="sle",
    ),
    "computed_mask_standalone_reduce_add": OpExpectation(
        kind="computed_mask_standalone_reduce_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-computed-mask-standalone-reduce-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="rvv_cm_standalone_reduce",
        external_abi_name="rvv-generic-computed-mask-standalone-reduce-add-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_cm_standalone_reduce_kernel_rvv_cm_standalone_reduce",
        emitc_route="rvv-generic-computed-mask-standalone-reduce-add-emitc-route",
        typed_compute_op="tcrv_rvv.masked_standalone_reduce",
        memory_form="computed-mask-unit-stride-standalone-reduction",
        lhs_initializer=(
            "(int32_t)((index % 5) == 0 ? 10 : "
            "((index % 5) == 1 ? 7 : "
            "((index % 5) == 2 ? 15 : "
            "((index % 5) == 3 ? -4 : -12))))"
        ),
        rhs_initializer=(
            "(int32_t)((index % 5) == 0 ? 10 : "
            "((index % 5) == 1 ? 9 : "
            "((index % 5) == 2 ? 12 : "
            "((index % 5) == 3 ? -4 : -13))))"
        ),
        source_initializer="(int32_t)(((index % 3) == 0) ? (int32_t)(index + 5) : (int32_t)(-((int32_t)index + 3)))",
        expected_expression=(
            "(int32_t)(acc[0] + sum_i(cmp_lhs[i] <= cmp_rhs[i] ? src[i] : 0))"
        ),
        compare_predicate_kind="sle",
    ),
    "runtime_scalar_cmp_masked_standalone_reduce_add": OpExpectation(
        kind="runtime_scalar_cmp_masked_standalone_reduce_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="rvv_rt_scalar_cm_standalone_reduce",
        external_abi_name="rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-add-callable-c-abi.v1",
        function_name="tcrv_emitc_rt_scalar_cm_standalone_reduce_kernel_rvv_rt_scalar_cm_standalone_reduce",
        emitc_route="rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-add-emitc-route",
        typed_compute_op="tcrv_rvv.masked_standalone_reduce",
        memory_form="runtime-scalar-computed-mask-unit-stride-standalone-reduction",
        lhs_initializer=(
            "(int32_t)(((index % 6) == 0) ? -120 : "
            "((index % 6) == 1) ? -37 : "
            "((index % 6) == 2) ? -11 : "
            "((index % 6) == 3) ? 0 : "
            "((index % 6) == 4) ? 91 : 140)"
        ),
        rhs_initializer="rhs_scalar",
        source_initializer=(
            "(int32_t)(((index % 4) == 0) ? (int32_t)(index + 5) : "
            "((index % 4) == 1) ? (int32_t)(-((int32_t)index + 7)) : "
            "((index % 4) == 2) ? (int32_t)(37 + (int32_t)index) : "
            "(int32_t)(-19 - (int32_t)index))"
        ),
        expected_expression=(
            "(int32_t)(acc[0] + sum_i(cmp_lhs[i] <= rhs_scalar ? src[i] : 0))"
        ),
        compare_predicate_kind="sle",
    ),
    "computed_mask_standalone_reduce_min": OpExpectation(
        kind="computed_mask_standalone_reduce_min",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-computed-mask-standalone-reduce-min.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="rvv_cm_standalone_reduce_min",
        external_abi_name="rvv-generic-computed-mask-standalone-reduce-min-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_cm_standalone_reduce_min_kernel_rvv_cm_standalone_reduce_min",
        emitc_route="rvv-generic-computed-mask-standalone-reduce-min-emitc-route",
        typed_compute_op="tcrv_rvv.masked_standalone_reduce",
        memory_form="computed-mask-unit-stride-standalone-reduction",
        lhs_initializer=(
            "(int32_t)((index % 5) == 0 ? 10 : "
            "((index % 5) == 1 ? 7 : "
            "((index % 5) == 2 ? 15 : "
            "((index % 5) == 3 ? -4 : -12))))"
        ),
        rhs_initializer=(
            "(int32_t)((index % 5) == 0 ? 10 : "
            "((index % 5) == 1 ? 9 : "
            "((index % 5) == 2 ? 12 : "
            "((index % 5) == 3 ? -4 : -13))))"
        ),
        source_initializer="(int32_t)(((index % 4) == 0) ? (int32_t)(index + 9) : ((index % 4) == 1) ? (int32_t)(-((int32_t)index + 21)) : ((index % 4) == 2) ? (int32_t)37 : (int32_t)-5)",
        expected_expression=(
            "(int32_t)(min_i(acc[0], src[i] where cmp_lhs[i] <= cmp_rhs[i]))"
        ),
        compare_predicate_kind="sle",
    ),
    "computed_mask_standalone_reduce_max": OpExpectation(
        kind="computed_mask_standalone_reduce_max",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-computed-mask-standalone-reduce-max.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="rvv_cm_standalone_reduce_max",
        external_abi_name="rvv-generic-computed-mask-standalone-reduce-max-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_cm_standalone_reduce_max_kernel_rvv_cm_standalone_reduce_max",
        emitc_route="rvv-generic-computed-mask-standalone-reduce-max-emitc-route",
        typed_compute_op="tcrv_rvv.masked_standalone_reduce",
        memory_form="computed-mask-unit-stride-standalone-reduction",
        lhs_initializer=(
            "(int32_t)((index % 5) == 0 ? 10 : "
            "((index % 5) == 1 ? 7 : "
            "((index % 5) == 2 ? 15 : "
            "((index % 5) == 3 ? -4 : -12))))"
        ),
        rhs_initializer=(
            "(int32_t)((index % 5) == 0 ? 10 : "
            "((index % 5) == 1 ? 9 : "
            "((index % 5) == 2 ? 12 : "
            "((index % 5) == 3 ? -4 : -13))))"
        ),
        source_initializer="(int32_t)(((index % 4) == 0) ? (int32_t)(-((int32_t)index + 9)) : ((index % 4) == 1) ? (int32_t)(index + 21) : ((index % 4) == 2) ? (int32_t)-37 : (int32_t)5)",
        expected_expression=(
            "(int32_t)(max_i(acc[0], src[i] where cmp_lhs[i] <= cmp_rhs[i]))"
        ),
        compare_predicate_kind="sle",
    ),
    "standalone_reduce_add": OpExpectation(
        kind="standalone_reduce_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-standalone-reduce-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_standalone_reduce_add",
        external_abi_name="rvv-generic-standalone-reduce-add-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_standalone_reduce_add_kernel_explicit_selected_body_rvv_standalone_reduce_add",
        emitc_route="rvv-generic-standalone-reduce-add-emitc-route",
        typed_compute_op="tcrv_rvv.standalone_reduce",
        memory_form="unit-stride-standalone-reduction",
        lhs_initializer="(int32_t)(((index % 5) < 2) ? -((int32_t)(index % 29) + 1) : ((int32_t)(index % 31) + 3))",
        rhs_initializer="unused",
        source_initializer="(int32_t)-11",
        expected_expression="(int32_t)(acc[0] + sum_i(lhs[i]))",
    ),
    "standalone_reduce_min": OpExpectation(
        kind="standalone_reduce_min",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-standalone-reduce-min.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_standalone_reduce_min",
        external_abi_name="rvv-generic-standalone-reduce-min-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_standalone_reduce_min_kernel_explicit_selected_body_rvv_standalone_reduce_min",
        emitc_route="rvv-generic-standalone-reduce-min-emitc-route",
        typed_compute_op="tcrv_rvv.standalone_reduce",
        memory_form="unit-stride-standalone-reduction",
        lhs_initializer="(int32_t)(((index % 5) == 0) ? (int32_t)(index + 4) : ((index % 5) == 1) ? (int32_t)(-((int32_t)index + 13)) : ((index % 5) == 2) ? (int32_t)37 : ((index % 5) == 3) ? (int32_t)-5 : (int32_t)(index + 19))",
        rhs_initializer="unused",
        source_initializer="(int32_t)-11",
        expected_expression="(int32_t)(min_i(acc[0], lhs[i]))",
    ),
    "standalone_reduce_max": OpExpectation(
        kind="standalone_reduce_max",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-standalone-reduce-max.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_standalone_reduce_max",
        external_abi_name="rvv-generic-standalone-reduce-max-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_standalone_reduce_max_kernel_explicit_selected_body_rvv_standalone_reduce_max",
        emitc_route="rvv-generic-standalone-reduce-max-emitc-route",
        typed_compute_op="tcrv_rvv.standalone_reduce",
        memory_form="unit-stride-standalone-reduction",
        lhs_initializer="(int32_t)(((index % 5) == 0) ? (int32_t)(-((int32_t)index + 4)) : ((index % 5) == 1) ? (int32_t)(index + 13) : ((index % 5) == 2) ? (int32_t)-37 : ((index % 5) == 3) ? (int32_t)5 : (int32_t)(-((int32_t)index + 19)))",
        rhs_initializer="unused",
        source_initializer="(int32_t)-11",
        expected_expression="(int32_t)(max_i(acc[0], lhs[i]))",
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
    "masked_sub": OpExpectation(
        kind="masked_sub",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-masked-sub.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_i32_masked_sub",
        external_abi_name="rvv-generic-masked-sub-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_masked_sub_kernel_explicit_selected_body_rvv_i32_masked_sub",
        emitc_route="rvv-generic-masked-sub-emitc-route",
        typed_compute_op="tcrv_rvv.masked_binary",
        memory_form="vector-rhs-load",
        lhs_initializer="(int32_t)(((index % 4) == 0) ? (int32_t)(200 + (int32_t)index) : (int32_t)(30 + (int32_t)index))",
        rhs_initializer="(int32_t)(((index % 4) == 0) ? (int32_t)(200 + (int32_t)index) : (int32_t)(100 + (int32_t)index))",
        expected_expression="(lhs[index] == rhs[index] ? (int32_t)(lhs[index] - rhs[index]) : lhs[index])",
    ),
    "masked_mul": OpExpectation(
        kind="masked_mul",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-masked-mul.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_i32_masked_mul",
        external_abi_name="rvv-generic-masked-mul-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_masked_mul_kernel_explicit_selected_body_rvv_i32_masked_mul",
        emitc_route="rvv-generic-masked-mul-emitc-route",
        typed_compute_op="tcrv_rvv.masked_binary",
        memory_form="vector-rhs-load",
        lhs_initializer="(int32_t)(((index % 4) == 0) ? (int32_t)(5 + (int32_t)(index % 7)) : (int32_t)(3 + (int32_t)index))",
        rhs_initializer="(int32_t)(((index % 4) == 0) ? (int32_t)(5 + (int32_t)(index % 7)) : (int32_t)(100 + (int32_t)index))",
        expected_expression="(lhs[index] == rhs[index] ? (int32_t)(lhs[index] * rhs[index]) : lhs[index])",
    ),
    "scalar_broadcast_add": OpExpectation(
        kind="scalar_broadcast_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_scalar_broadcast_add",
        external_abi_name="rvv-generic-scalar-broadcast-add-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_scalar_broadcast_add_kernel_explicit_selected_body_rvv_scalar_broadcast_add",
        emitc_route="rvv-generic-scalar-broadcast-add-emitc-route",
        typed_compute_op="tcrv_rvv.binary",
        memory_form="rhs-scalar-broadcast",
        lhs_initializer="(int32_t)(7 + (int32_t)(index * 3))",
        rhs_initializer="(int32_t)-37",
        expected_expression="lhs[index] + rhs_scalar",
    ),
    "scalar_broadcast_sub": OpExpectation(
        kind="scalar_broadcast_sub",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-sub.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_scalar_broadcast_sub",
        external_abi_name="rvv-generic-scalar-broadcast-sub-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_scalar_broadcast_sub_kernel_explicit_selected_body_rvv_scalar_broadcast_sub",
        emitc_route="rvv-generic-scalar-broadcast-sub-emitc-route",
        typed_compute_op="tcrv_rvv.binary",
        memory_form="rhs-scalar-broadcast",
        lhs_initializer="(int32_t)(500 - (int32_t)(index * 2))",
        rhs_initializer="(int32_t)17",
        expected_expression="lhs[index] - rhs_scalar",
    ),
    "scalar_broadcast_mul": OpExpectation(
        kind="scalar_broadcast_mul",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-mul.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_scalar_broadcast_mul",
        external_abi_name="rvv-generic-scalar-broadcast-mul-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_scalar_broadcast_mul_kernel_explicit_selected_body_rvv_scalar_broadcast_mul",
        emitc_route="rvv-generic-scalar-broadcast-mul-emitc-route",
        typed_compute_op="tcrv_rvv.binary",
        memory_form="rhs-scalar-broadcast",
        lhs_initializer="(int32_t)((int)(index % 13) - 6)",
        rhs_initializer="(int32_t)-3",
        expected_expression="lhs[index] * rhs_scalar",
    ),
    "runtime_i32_splat_store": OpExpectation(
        kind="runtime_i32_splat_store",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-runtime-i32-splat-store.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_runtime_i32_splat_store",
        external_abi_name="rvv-generic-runtime-i32-splat-store-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_runtime_i32_splat_store_kernel_explicit_selected_body_rvv_runtime_i32_splat_store",
        emitc_route="rvv-generic-runtime-i32-splat-store-emitc-route",
        typed_compute_op="tcrv_rvv.splat",
        memory_form="runtime-scalar-splat-store",
        lhs_initializer="(int32_t)0",
        rhs_initializer="(int32_t)-37",
        expected_expression="rhs_scalar",
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
        lhs_initializer="(int32_t)(((index % 2) == 0) ? ((int32_t)(index % 7) + 1) : -((int32_t)(index % 7) + 1))",
        rhs_initializer="(int32_t)(((index % 3) == 0) ? -((int32_t)(index % 5) + 2) : ((int32_t)(index % 5) + 2))",
        source_initializer="(int32_t)(((index % 2) == 0) ? (17 - (int32_t)(index % 9)) : -(17 - (int32_t)(index % 9)))",
        expected_expression="(int32_t)(acc[index] + (int32_t)(lhs[index] * rhs[index]))",
    ),
    "computed_masked_macc_add": OpExpectation(
        kind="computed_masked_macc_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-computed-masked-macc-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_computed_masked_macc_add",
        external_abi_name="rvv-generic-computed-masked-macc-add-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_computed_masked_macc_add_kernel_explicit_selected_body_rvv_computed_masked_macc_add",
        emitc_route="rvv-generic-computed-masked-macc-add-emitc-route",
        typed_compute_op="tcrv_rvv.masked_macc",
        memory_form="computed-mask-unit-stride-macc",
        lhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(10 + (int32_t)index) "
            ": (int32_t)(100 + (int32_t)index))"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(50 + (int32_t)index) "
            ": (int32_t)(20 + (int32_t)index))"
        ),
        source_initializer="(int32_t)(((index % 2) == 0) ? (23 - (int32_t)(index % 11)) : -(23 - (int32_t)(index % 11)))",
        true_value_initializer="(int32_t)(((index % 2) == 0) ? ((int32_t)(index % 7) + 2) : -((int32_t)(index % 7) + 2))",
        false_value_initializer="(int32_t)(((index % 3) == 0) ? -((int32_t)(index % 5) + 3) : ((int32_t)(index % 5) + 3))",
        expected_expression=(
            "(cmp_lhs[index] < cmp_rhs[index] ? "
            "(int32_t)(acc[index] + (int32_t)(lhs[index] * rhs[index])) "
            ": acc[index])"
        ),
        compare_predicate_kind="slt",
    ),
    "runtime_scalar_cmp_masked_macc_add": OpExpectation(
        kind="runtime_scalar_cmp_masked_macc_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="rvv_rt_scalar_masked_macc",
        external_abi_name="rvv-generic-runtime-scalar-cmp-masked-macc-add-callable-c-abi.v1",
        function_name="tcrv_emitc_rt_scalar_masked_macc_kernel_rvv_rt_scalar_masked_macc",
        emitc_route="rvv-generic-runtime-scalar-cmp-masked-macc-add-emitc-route",
        typed_compute_op="tcrv_rvv.masked_macc",
        memory_form="runtime-scalar-computed-mask-unit-stride-macc",
        lhs_initializer=(
            "(int32_t)(((index % 5) == 0) ? -120 : "
            "((index % 5) == 1) ? -37 : "
            "((index % 5) == 2) ? 0 : "
            "((index % 5) == 3) ? 91 : 130)"
        ),
        rhs_initializer="rhs_scalar",
        source_initializer=(
            "(int32_t)(((index % 2) == 0) ? "
            "(29 - (int32_t)(index % 13)) : "
            "-(29 - (int32_t)(index % 13)))"
        ),
        true_value_initializer=(
            "(int32_t)(((index % 2) == 0) ? "
            "((int32_t)(index % 7) + 2) : "
            "-((int32_t)(index % 7) + 2))"
        ),
        false_value_initializer=(
            "(int32_t)(((index % 3) == 0) ? "
            "-((int32_t)(index % 5) + 3) : "
            "((int32_t)(index % 5) + 3))"
        ),
        expected_expression=(
            "(cmp_lhs[index] <= rhs_scalar ? "
            "(int32_t)(acc[index] + (int32_t)(lhs[index] * rhs[index])) "
            ": acc[index])"
        ),
        compare_predicate_kind="sle",
    ),
    "widening_macc_add": OpExpectation(
        kind="widening_macc_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-widening-macc-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_widening_macc_add",
        external_abi_name="rvv-generic-widening-macc-add-callable-c-abi.v1",
        function_name=(
            "tcrv_emitc_explicit_selected_body_widening_macc_add_kernel_"
            "explicit_selected_body_rvv_widening_macc_add"
        ),
        emitc_route="rvv-generic-widening-macc-add-emitc-route",
        typed_compute_op="tcrv_rvv.widening_macc",
        memory_form="vector-rhs-load",
        lhs_initializer="(int16_t)(((index % 4) < 2) ? -((int)(index % 97) + 2) : ((int)(index % 97) + 3))",
        rhs_initializer="(int16_t)(((index % 3) == 0) ? -((int)(index % 41) + 5) : ((int)(index % 41) + 7))",
        source_initializer="(int32_t)(300 + (int32_t)(index * 11))",
        expected_expression="(int32_t)(acc[index] + (int32_t)lhs[index] * (int32_t)rhs[index])",
        out_initializer=OUT_SENTINEL,
        lmul="m1",
        sew="32",
        element_c_type="int32_t",
        config_contract="rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m1",
    ),
    "widening_dot_reduce_add": OpExpectation(
        kind="widening_dot_reduce_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-widening-dot-reduce-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_widening_dot_reduce_add",
        external_abi_name="rvv-generic-widening-dot-reduce-add-callable-c-abi.v1",
        function_name=(
            "tcrv_emitc_explicit_selected_body_widening_dot_reduce_add_kernel_"
            "explicit_selected_body_rvv_widening_dot_reduce_add"
        ),
        emitc_route="rvv-generic-widening-dot-reduce-add-emitc-route",
        typed_compute_op="tcrv_rvv.widening_dot_reduce",
        memory_form="vector-rhs-load",
        lhs_initializer="(int16_t)(((index % 4) < 2) ? -((int)(index % 53) + 2) : ((int)(index % 53) + 5))",
        rhs_initializer="(int16_t)(((index % 5) == 0) ? -((int)(index % 37) + 3) : ((int)(index % 37) + 7))",
        source_initializer="(int32_t)17",
        expected_expression="(int32_t)(acc[0] + sum_i((int32_t)lhs[i] * (int32_t)rhs[i]))",
        out_initializer=OUT_SENTINEL,
        lmul="m1",
        sew="32",
        element_c_type="int32_t",
        config_contract="rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m1",
    ),
    "strided_input_widening_dot_reduce_add": OpExpectation(
        kind="strided_input_widening_dot_reduce_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-strided-input-widening-dot-reduce-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="rvv_explicit_strided_input_dot",
        external_abi_name="rvv-generic-strided-input-widening-dot-reduce-add-callable-c-abi.v1",
        function_name=(
            "tcrv_emitc_explicit_strided_dot_kernel_"
            "rvv_explicit_strided_input_dot"
        ),
        emitc_route="rvv-generic-strided-input-widening-dot-reduce-add-emitc-route",
        typed_compute_op="tcrv_rvv.widening_dot_reduce",
        memory_form="strided-input-widening-dot-reduce",
        lhs_initializer="(int16_t)(((index % 4) < 2) ? -((int)(index % 59) + 3) : ((int)(index % 59) + 6))",
        rhs_initializer="(int16_t)(((index % 5) == 0) ? -((int)(index % 43) + 4) : ((int)(index % 43) + 9))",
        source_initializer="(int32_t)31",
        expected_expression=(
            "(int32_t)(acc[0] + sum_i((int32_t)lhs[i * lhs_stride] * "
            "(int32_t)rhs[i * rhs_stride]))"
        ),
        out_initializer=OUT_SENTINEL,
        lmul="m1",
        sew="32",
        element_c_type="int32_t",
        config_contract="rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m1",
    ),
    "computed_masked_widening_dot_reduce_add": OpExpectation(
        kind="computed_masked_widening_dot_reduce_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="rvv_explicit_masked_wdot",
        external_abi_name="rvv-generic-computed-masked-widening-dot-reduce-add-callable-c-abi.v1",
        function_name=(
            "tcrv_emitc_explicit_masked_wdot_kernel_"
            "rvv_explicit_masked_wdot"
        ),
        emitc_route="rvv-generic-computed-masked-widening-dot-reduce-add-emitc-route",
        typed_compute_op="tcrv_rvv.masked_widening_dot_reduce",
        memory_form="computed-mask-unit-stride-widening-dot-reduce",
        lhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(10 + (int32_t)index) "
            ": (int32_t)(100 + (int32_t)index))"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(50 + (int32_t)index) "
            ": (int32_t)(20 + (int32_t)index))"
        ),
        source_initializer="(int32_t)29",
        expected_expression=(
            "(int32_t)(acc[0] + sum_i(cmp_lhs[i] < cmp_rhs[i] ? "
            "(int32_t)dot_lhs[i] * (int32_t)dot_rhs[i] : 0))"
        ),
        out_initializer=OUT_SENTINEL,
        lmul="m1",
        sew="32",
        element_c_type="int32_t",
        config_contract="rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m1",
    ),
    "computed_masked_strided_input_widening_dot_reduce_add": OpExpectation(
        kind="computed_masked_strided_input_widening_dot_reduce_add",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="rvv_explicit_masked_strided_dot",
        external_abi_name="rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-callable-c-abi.v1",
        function_name=(
            "tcrv_emitc_explicit_masked_strided_dot_kernel_"
            "rvv_explicit_masked_strided_dot"
        ),
        emitc_route="rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-emitc-route",
        typed_compute_op="tcrv_rvv.masked_widening_dot_reduce",
        memory_form="computed-mask-strided-input-widening-dot-reduce",
        lhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(10 + (int32_t)index) "
            ": (int32_t)(100 + (int32_t)index))"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(50 + (int32_t)index) "
            ": (int32_t)(20 + (int32_t)index))"
        ),
        source_initializer="(int32_t)37",
        expected_expression=(
            "(int32_t)(acc[0] + sum_i(cmp_lhs[i] < cmp_rhs[i] ? "
            "(int32_t)dot_lhs[i * lhs_stride] * "
            "(int32_t)dot_rhs[i * rhs_stride] : 0))"
        ),
        out_initializer=OUT_SENTINEL,
        lmul="m1",
        sew="32",
        element_c_type="int32_t",
        config_contract="rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m1",
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
    "strided_load_unit_store": OpExpectation(
        kind="strided_load_unit_store",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-strided-load-unit-store.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_strided_load_unit_store",
        external_abi_name="rvv-generic-strided-load-unit-store-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_strided_load_unit_store_kernel_explicit_selected_body_rvv_strided_load_unit_store",
        emitc_route="rvv-generic-strided-load-unit-store-emitc-route",
        typed_compute_op="tcrv_rvv.move",
        memory_form="strided-load-unit-store",
        lhs_initializer="(int32_t)(31 + (int32_t)(index * 7))",
        rhs_initializer="unused",
        expected_expression=(
            "*(const int32_t *)((const uint8_t *)src + "
            "index * stride_bytes)"
        ),
    ),
    "unit_load_strided_store": OpExpectation(
        kind="unit_load_strided_store",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-unit-load-strided-store.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_unit_load_strided_store",
        external_abi_name="rvv-generic-unit-load-strided-store-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_unit_load_strided_store_kernel_explicit_selected_body_rvv_unit_load_strided_store",
        emitc_route="rvv-generic-unit-load-strided-store-emitc-route",
        typed_compute_op="tcrv_rvv.move",
        memory_form="unit-load-strided-store",
        lhs_initializer="(int32_t)(131 + (int32_t)(index * 11))",
        rhs_initializer="unused",
        expected_expression="src[index]",
    ),
    "computed_masked_strided_load_unit_store": OpExpectation(
        kind="computed_masked_strided_load_unit_store",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-load.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_computed_masked_strided_load",
        external_abi_name="rvv-generic-computed-masked-strided-load-unit-store-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_computed_masked_strided_load_kernel_explicit_selected_body_rvv_computed_masked_strided_load",
        emitc_route="rvv-generic-computed-masked-strided-load-unit-store-emitc-route",
        typed_compute_op="tcrv_rvv.masked_strided_load",
        memory_form="computed-mask-strided-load-unit-store",
        lhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(10 + (int32_t)index) "
            ": (int32_t)(100 + (int32_t)index))"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(50 + (int32_t)index) "
            ": (int32_t)(20 + (int32_t)index))"
        ),
        source_initializer="(int32_t)(3100 + (int32_t)(index * 31))",
        out_initializer="(int32_t)(-11000 - (int32_t)(index * 37))",
        expected_expression=(
            "(cmp_lhs[index] < cmp_rhs[index] ? source_slot : old_dst[index])"
        ),
    ),
    "computed_masked_indexed_gather_load_unit_store": OpExpectation(
        kind="computed_masked_indexed_gather_load_unit_store",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-computed-masked-indexed-gather-load.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_cmidx_load",
        external_abi_name="rvv-generic-computed-masked-indexed-gather-load-unit-store-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_cmidx_load_kernel_explicit_selected_body_rvv_cmidx_load",
        emitc_route="rvv-generic-computed-masked-indexed-gather-load-unit-store-emitc-route",
        typed_compute_op="tcrv_rvv.masked_indexed_load",
        memory_form="computed-mask-indexed-gather-load-unit-store",
        lhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(10 + (int32_t)index) "
            ": (int32_t)(100 + (int32_t)index))"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(50 + (int32_t)index) "
            ": (int32_t)(20 + (int32_t)index))"
        ),
        source_initializer="(int32_t)(4100 + (int32_t)(index * 43))",
        out_initializer="(int32_t)(-13000 - (int32_t)(index * 41))",
        expected_expression=(
            "(cmp_lhs[index] < cmp_rhs[index] ? src[indices[index]] : old_dst[index])"
        ),
        compare_predicate_kind="slt",
    ),
    "computed_masked_indexed_scatter_store_unit_load": OpExpectation(
        kind="computed_masked_indexed_scatter_store_unit_load",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-computed-masked-indexed-scatter-store.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_cmidx_store",
        external_abi_name="rvv-generic-computed-masked-indexed-scatter-store-unit-load-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_cmidx_store_kernel_explicit_selected_body_rvv_cmidx_store",
        emitc_route="rvv-generic-computed-masked-indexed-scatter-store-unit-load-emitc-route",
        typed_compute_op="tcrv_rvv.masked_indexed_store",
        memory_form="computed-mask-unit-load-indexed-scatter-store",
        lhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(10 + (int32_t)index) "
            ": (int32_t)(100 + (int32_t)index))"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(50 + (int32_t)index) "
            ": (int32_t)(20 + (int32_t)index))"
        ),
        source_initializer="(int32_t)(5100 + (int32_t)(index * 47))",
        out_initializer="(int32_t)(-15000 - (int32_t)(index * 43))",
        expected_expression=(
            "(cmp_lhs[index] < cmp_rhs[index] ? src[index] : old_dst[indices[index]])"
        ),
        compare_predicate_kind="slt",
    ),
    "computed_masked_segment2_load_unit_store": OpExpectation(
        kind="computed_masked_segment2_load_unit_store",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-load.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_cmseg_load",
        external_abi_name="rvv-generic-computed-masked-segment2-load-unit-store-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_cmseg_load_kernel_explicit_selected_body_rvv_cmseg_load",
        emitc_route="rvv-generic-computed-masked-segment2-load-unit-store-emitc-route",
        typed_compute_op="tcrv_rvv.masked_segment2_load",
        memory_form="computed-mask-segment2-load-unit-store",
        lhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(10 + (int32_t)index) "
            ": (int32_t)(100 + (int32_t)index))"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(50 + (int32_t)index) "
            ": (int32_t)(20 + (int32_t)index))"
        ),
        source_initializer="(int32_t)(6100 + (int32_t)(index * 53))",
        out_initializer="(int32_t)(-17000 - (int32_t)(index * 47))",
        expected_expression=(
            "cmp_lhs[index] < cmp_rhs[index] ? src[2 * index + field] : old_field[index]"
        ),
        compare_predicate_kind="slt",
    ),
    "computed_masked_segment2_store_unit_load": OpExpectation(
        kind="computed_masked_segment2_store_unit_load",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-store.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_cmseg_store",
        external_abi_name="rvv-generic-computed-masked-segment2-store-unit-load-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_cmseg_store_kernel_explicit_selected_body_rvv_cmseg_store",
        emitc_route="rvv-generic-computed-masked-segment2-store-unit-load-emitc-route",
        typed_compute_op="tcrv_rvv.masked_segment2_store",
        memory_form="computed-mask-unit-load-segment2-store",
        lhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(10 + (int32_t)index) "
            ": (int32_t)(100 + (int32_t)index))"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(50 + (int32_t)index) "
            ": (int32_t)(20 + (int32_t)index))"
        ),
        source_initializer="(int32_t)(7100 + (int32_t)(index * 59))",
        out_initializer="(int32_t)(-19000 - (int32_t)(index * 41))",
        expected_expression=(
            "cmp_lhs[index] < cmp_rhs[index] ? src_field[index] : old_dst[2 * index + field]"
        ),
        compare_predicate_kind="slt",
    ),
    "segment2_deinterleave_unit_store": OpExpectation(
        kind="segment2_deinterleave_unit_store",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-segment2-deinterleave-unit-store.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_rvv_seg2_deinterleave",
        external_abi_name="rvv-generic-segment2-deinterleave-unit-store-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_seg2_deinterleave_kernel_explicit_rvv_seg2_deinterleave",
        emitc_route="rvv-generic-segment2-deinterleave-unit-store-emitc-route",
        typed_compute_op="tcrv_rvv.move",
        memory_form="segment2-load-unit-store",
        lhs_initializer="unused",
        rhs_initializer="unused",
        source_initializer="(int32_t)(1700 + (int32_t)(index * 23))",
        expected_expression="out0[index] == src[2 * index] && out1[index] == src[2 * index + 1]",
    ),
    "segment2_interleave_unit_load": OpExpectation(
        kind="segment2_interleave_unit_load",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-segment2-interleave-unit-load.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_rvv_seg2_interleave",
        external_abi_name="rvv-generic-segment2-interleave-unit-load-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_seg2_interleave_kernel_explicit_rvv_seg2_interleave",
        emitc_route="rvv-generic-segment2-interleave-unit-load-emitc-route",
        typed_compute_op="tcrv_rvv.segment2_store",
        memory_form="unit-load-segment2-store",
        lhs_initializer="(int32_t)(1900 + (int32_t)(index * 29))",
        rhs_initializer="(int32_t)(-2300 - (int32_t)(index * 31))",
        expected_expression="dst[2 * index] == src0[index] && dst[2 * index + 1] == src1[index]",
    ),
    "indexed_gather_unit_store": OpExpectation(
        kind="indexed_gather_unit_store",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-indexed-gather-unit-store.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_indexed_gather_unit_store",
        external_abi_name="rvv-generic-indexed-gather-unit-store-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_indexed_gather_unit_store_kernel_explicit_selected_body_rvv_indexed_gather_unit_store",
        emitc_route="rvv-generic-indexed-gather-unit-store-emitc-route",
        typed_compute_op="tcrv_rvv.move",
        memory_form="indexed-load-unit-store",
        lhs_initializer="(int32_t)(101 + (int32_t)(index * 9))",
        rhs_initializer="unused",
        expected_expression="data[indices[index]]",
    ),
    "indexed_scatter_unit_load": OpExpectation(
        kind="indexed_scatter_unit_load",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-indexed-scatter-unit-load.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_indexed_scatter_unit_load",
        external_abi_name="rvv-generic-indexed-scatter-unit-load-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_indexed_scatter_unit_load_kernel_explicit_selected_body_rvv_indexed_scatter_unit_load",
        emitc_route="rvv-generic-indexed-scatter-unit-load-emitc-route",
        typed_compute_op="tcrv_rvv.move",
        memory_form="unit-load-indexed-store",
        lhs_initializer="(int32_t)(503 + (int32_t)(index * 11))",
        rhs_initializer="unused",
        expected_expression="src[logical_index]",
    ),
    "masked_unit_load_store": OpExpectation(
        kind="masked_unit_load_store",
        input_path=Path("test/Target/RVV/explicit-selected-body-artifact-masked-unit-load-store.mlir"),
        input_mode="explicit-selected-body",
        source_seed=False,
        selected_variant="explicit_selected_body_rvv_masked_unit_load_store",
        external_abi_name="rvv-generic-masked-unit-load-store-callable-c-abi.v1",
        function_name="tcrv_emitc_explicit_selected_body_masked_unit_load_store_kernel_explicit_selected_body_rvv_masked_unit_load_store",
        emitc_route="rvv-generic-masked-unit-load-store-emitc-route",
        typed_compute_op="tcrv_rvv.masked_load",
        memory_form="masked-unit-load-store",
        lhs_initializer="(int32_t)(900 + (int32_t)(index * 13))",
        rhs_initializer="(int32_t)(((index % 5) == 0 || (index % 5) == 2) ? 1 : 0)",
        out_initializer="(int32_t)(-7000 - (int32_t)(index * 17))",
        expected_expression="(mask[index] != 0 ? src[index] : old_dst[index])",
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
    "cmp_select": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["cmp_select"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-cmp-select.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_cmp_select",
        function_name="tcrv_emitc_pre_realized_body_cmp_select_kernel_pre_realized_body_rvv_cmp_select",
    ),
    "cmp_select_sle": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["cmp_select_sle"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-cmp-select-sle.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_cmp_select_sle",
        function_name="tcrv_emitc_pre_realized_body_cmp_select_sle_kernel_pre_realized_body_rvv_cmp_select_sle",
    ),
    "computed_mask_select": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["cmp_select"],
        kind="computed_mask_select",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_computed_mask_select",
        external_abi_name="rvv-generic-computed-mask-select-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_body_computed_mask_select_kernel_pre_realized_body_rvv_computed_mask_select",
        emitc_route="rvv-generic-computed-mask-select-emitc-route",
        typed_compute_op="tcrv_rvv.select",
        memory_form="computed-mask-vector-select",
        lhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(10 + (int32_t)index) "
            ": (int32_t)(100 + (int32_t)index))"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(50 + (int32_t)index) "
            ": (int32_t)(20 + (int32_t)index))"
        ),
        true_value_initializer="(int32_t)(3000 + (int32_t)(index * 17))",
        false_value_initializer="(int32_t)(-4100 - (int32_t)(index * 19))",
        expected_expression=(
            "(cmp_lhs[index] < cmp_rhs[index] ? true_value[index] : false_value[index])"
        ),
        compare_predicate_kind="slt",
    ),
    "computed_mask_select_sle": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["computed_mask_select_sle"],
        kind="computed_mask_select_sle",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select-sle.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_computed_mask_select_sle",
        external_abi_name="rvv-generic-computed-mask-select-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_body_computed_mask_select_sle_kernel_pre_realized_body_rvv_computed_mask_select_sle",
        emitc_route="rvv-generic-computed-mask-select-emitc-route",
        typed_compute_op="tcrv_rvv.select",
        memory_form="computed-mask-vector-select",
        lhs_initializer=(
            "(int32_t)(((index % 5) == 0) ? 10 : "
            "((index % 5) == 1) ? 9 : "
            "((index % 5) == 2) ? 14 : "
            "((index % 5) == 3) ? -3 : -9)"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 5) == 0) ? 10 : "
            "((index % 5) == 1) ? 12 : "
            "((index % 5) == 2) ? 11 : "
            "((index % 5) == 3) ? -3 : -12)"
        ),
        true_value_initializer="(int32_t)(2100 + (int32_t)(index * 17))",
        false_value_initializer="(int32_t)(-2300 - (int32_t)(index * 19))",
        expected_expression=(
            "(cmp_lhs[index] <= cmp_rhs[index] ? true_value[index] : false_value[index])"
        ),
        compare_predicate_kind="sle",
    ),
    "runtime_scalar_cmp_select": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["runtime_scalar_cmp_select"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-select.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_runtime_scalar_cmp_select",
        function_name="tcrv_emitc_pre_realized_body_runtime_scalar_cmp_select_kernel_pre_realized_body_rvv_runtime_scalar_cmp_select",
    ),
    "runtime_scalar_dual_cmp_mask_and_select": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "runtime_scalar_dual_cmp_mask_and_select"
        ],
        input_path=Path(
            "test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir"
        ),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_rvv_dual_cmp_mask_select",
        function_name=(
            "tcrv_emitc_pre_dual_cmp_mask_select_kernel_"
            "pre_rvv_dual_cmp_mask_select"
        ),
    ),
    "runtime_scalar_cmp_masked_store": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["runtime_scalar_cmp_masked_store"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-store.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_runtime_scalar_cmp_masked_store",
        function_name="tcrv_emitc_pre_realized_body_runtime_scalar_cmp_masked_store_kernel_pre_realized_body_rvv_runtime_scalar_cmp_masked_store",
    ),
    "runtime_scalar_cmp_masked_load_store": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "runtime_scalar_cmp_masked_load_store"
        ],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-load-store.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pr_rvv_cmp_mload",
        function_name="tcrv_emitc_pr_rt_cmp_mload_kernel_pr_rvv_cmp_mload",
    ),
    "masked_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["masked_add"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-masked-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_masked_add",
        function_name="tcrv_emitc_pre_realized_body_masked_add_kernel_pre_realized_body_rvv_masked_add",
    ),
    "masked_sub": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["masked_sub"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-masked-sub.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_masked_sub",
        function_name="tcrv_emitc_pre_realized_body_masked_sub_kernel_pre_realized_body_rvv_masked_sub",
    ),
    "masked_mul": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["masked_mul"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-masked-mul.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_masked_mul",
        function_name="tcrv_emitc_pre_realized_body_masked_mul_kernel_pre_realized_body_rvv_masked_mul",
    ),
    "reduce_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["reduce_add"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-reduce-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_reduce_add",
        function_name="tcrv_emitc_pre_realized_body_reduce_add_kernel_pre_realized_body_rvv_reduce_add",
    ),
    "macc_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["macc_add"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-macc-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_macc_add",
        function_name="tcrv_emitc_pre_realized_body_macc_add_kernel_pre_realized_body_rvv_macc_add",
    ),
    "computed_masked_macc_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["computed_masked_macc_add"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-macc-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_computed_masked_macc_add",
        function_name="tcrv_emitc_pre_realized_body_computed_masked_macc_add_kernel_pre_realized_body_rvv_computed_masked_macc_add",
    ),
    "runtime_scalar_cmp_masked_macc_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "runtime_scalar_cmp_masked_macc_add"
        ],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="rvv_pr_rt_scalar_masked_macc",
        function_name="tcrv_emitc_pr_rt_scalar_masked_macc_kernel_rvv_pr_rt_scalar_masked_macc",
    ),
    "strided_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["strided_add"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-strided-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_strided_add",
        function_name="tcrv_emitc_pre_realized_body_strided_add_kernel_pre_realized_body_rvv_strided_add",
    ),
    "strided_load_unit_store": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["strided_load_unit_store"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-strided-load-unit-store.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_strided_load_unit_store",
        function_name="tcrv_emitc_pre_realized_body_strided_load_unit_store_kernel_pre_realized_body_rvv_strided_load_unit_store",
    ),
    "unit_load_strided_store": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["unit_load_strided_store"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-unit-load-strided-store.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_unit_load_strided_store",
        function_name="tcrv_emitc_pre_realized_body_unit_load_strided_store_kernel_pre_realized_body_rvv_unit_load_strided_store",
    ),
    "indexed_gather_unit_store": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["indexed_gather_unit_store"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-indexed-gather-unit-store.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_indexed_gather_unit_store",
        function_name="tcrv_emitc_pre_realized_body_indexed_gather_unit_store_kernel_pre_realized_body_rvv_indexed_gather_unit_store",
    ),
    "indexed_scatter_unit_load": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["indexed_scatter_unit_load"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-indexed-scatter-unit-load.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_indexed_scatter_unit_load",
        function_name="tcrv_emitc_pre_realized_body_indexed_scatter_unit_load_kernel_pre_realized_body_rvv_indexed_scatter_unit_load",
    ),
    "masked_unit_load_store": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["masked_unit_load_store"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-masked-unit-load-store.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_masked_unit_load_store",
        function_name="tcrv_emitc_pre_realized_body_masked_unit_load_store_kernel_pre_realized_body_rvv_masked_unit_load_store",
    ),
    "masked_unit_store": OpExpectation(
        kind="masked_unit_store",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-masked-unit-store.mlir"),
        input_mode="pre-realized-selected-body",
        source_seed=False,
        selected_variant="pre_realized_body_rvv_masked_unit_store",
        external_abi_name="rvv-generic-masked-unit-store-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_body_masked_unit_store_kernel_pre_realized_body_rvv_masked_unit_store",
        emitc_route="rvv-generic-masked-unit-store-emitc-route",
        typed_compute_op="tcrv_rvv.masked_store",
        memory_form="masked-unit-store",
        lhs_initializer="(int32_t)(900 + (int32_t)(index * 13))",
        rhs_initializer="(int32_t)(((index % 5) == 0 || (index % 5) == 2) ? 1 : 0)",
        out_initializer="(int32_t)(-7000 - (int32_t)(index * 17))",
        expected_expression="(mask[index] != 0 ? src[index] : old_dst[index])",
        config_contract="rvv-selected-body-sew32-lmul-m1-tail-undisturbed-mask-undisturbed.v1",
    ),
    "computed_masked_unit_load_store": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["masked_unit_load_store"],
        kind="computed_masked_unit_load_store",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-unit-load-store.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_computed_masked_unit_load_store",
        external_abi_name="rvv-generic-computed-masked-unit-load-store-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_body_computed_masked_unit_load_store_kernel_pre_realized_body_rvv_computed_masked_unit_load_store",
        emitc_route="rvv-generic-computed-masked-unit-load-store-emitc-route",
        typed_compute_op="tcrv_rvv.masked_load",
        memory_form="computed-mask-unit-load-store",
        lhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(10 + (int32_t)index) "
            ": (int32_t)(100 + (int32_t)index))"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(50 + (int32_t)index) "
            ": (int32_t)(20 + (int32_t)index))"
        ),
        source_initializer="(int32_t)(1200 + (int32_t)(index * 19))",
        out_initializer="(int32_t)(-7000 - (int32_t)(index * 17))",
        expected_expression=(
            "(cmp_lhs[index] < cmp_rhs[index] ? src[index] : old_dst[index])"
        ),
        compare_predicate_kind="slt",
    ),
    "computed_masked_strided_store": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["masked_unit_load_store"],
        kind="computed_masked_strided_store",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-store.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_computed_masked_strided_store",
        external_abi_name="rvv-generic-computed-masked-strided-store-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_body_computed_masked_strided_store_kernel_pre_realized_body_rvv_computed_masked_strided_store",
        emitc_route="rvv-generic-computed-masked-strided-store-emitc-route",
        typed_compute_op="tcrv_rvv.masked_strided_store",
        memory_form="computed-mask-unit-load-strided-store",
        lhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(10 + (int32_t)index) "
            ": (int32_t)(100 + (int32_t)index))"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(50 + (int32_t)index) "
            ": (int32_t)(20 + (int32_t)index))"
        ),
        source_initializer="(int32_t)(2200 + (int32_t)(index * 23))",
        out_initializer="(int32_t)(-9000 - (int32_t)(index * 29))",
        expected_expression=(
            "(cmp_lhs[index] < cmp_rhs[index] ? src[index] : old_slot)"
        ),
        compare_predicate_kind="slt",
    ),
    "computed_masked_strided_load_unit_store": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "computed_masked_strided_load_unit_store"
        ],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-load.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_computed_masked_strided_load",
        function_name="tcrv_emitc_pre_realized_body_computed_masked_strided_load_kernel_pre_realized_body_rvv_computed_masked_strided_load",
    ),
    "computed_masked_indexed_gather_load_unit_store": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "computed_masked_indexed_gather_load_unit_store"
        ],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-indexed-gather-load.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_cmidx_load",
        function_name="tcrv_emitc_pre_realized_body_cmidx_load_kernel_pre_realized_body_rvv_cmidx_load",
    ),
    "computed_masked_indexed_scatter_store_unit_load": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "computed_masked_indexed_scatter_store_unit_load"
        ],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-indexed-scatter-store.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_cmidx_store",
        function_name="tcrv_emitc_pre_realized_body_cmidx_store_kernel_pre_realized_body_rvv_cmidx_store",
    ),
    "computed_masked_segment2_load_unit_store": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "computed_masked_segment2_load_unit_store"
        ],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-load.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_cmseg_load",
        function_name="tcrv_emitc_pre_realized_body_cmseg_load_kernel_pre_realized_body_rvv_cmseg_load",
    ),
    "computed_masked_segment2_store_unit_load": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "computed_masked_segment2_store_unit_load"
        ],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-store.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_cmseg_store",
        function_name="tcrv_emitc_pre_realized_body_cmseg_store_kernel_pre_realized_body_rvv_cmseg_store",
    ),
    "segment2_deinterleave_unit_store": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["strided_load_unit_store"],
        kind="segment2_deinterleave_unit_store",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-segment2-deinterleave-unit-store.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_segment2_deinterleave_unit_store",
        external_abi_name="rvv-generic-segment2-deinterleave-unit-store-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_body_segment2_deinterleave_unit_store_kernel_pre_realized_body_rvv_segment2_deinterleave_unit_store",
        emitc_route="rvv-generic-segment2-deinterleave-unit-store-emitc-route",
        typed_compute_op="tcrv_rvv.move",
        memory_form="segment2-load-unit-store",
        source_initializer="(int32_t)(1700 + (int32_t)(index * 23))",
        expected_expression="out0[index] == src[2 * index] && out1[index] == src[2 * index + 1]",
    ),
    "segment2_interleave_unit_load": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["strided_load_unit_store"],
        kind="segment2_interleave_unit_load",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_segment2_interleave_unit_load",
        external_abi_name="rvv-generic-segment2-interleave-unit-load-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_body_segment2_interleave_unit_load_kernel_pre_realized_body_rvv_segment2_interleave_unit_load",
        emitc_route="rvv-generic-segment2-interleave-unit-load-emitc-route",
        typed_compute_op="tcrv_rvv.segment2_store",
        memory_form="unit-load-segment2-store",
        lhs_initializer="(int32_t)(1900 + (int32_t)(index * 29))",
        rhs_initializer="(int32_t)(-2300 - (int32_t)(index * 31))",
        expected_expression="dst[2 * index] == src0[index] && dst[2 * index + 1] == src1[index]",
    ),
    "scalar_broadcast_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"],
        kind="scalar_broadcast_add",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_scalar_broadcast_add",
        external_abi_name="rvv-generic-scalar-broadcast-add-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_body_scalar_broadcast_add_kernel_pre_realized_body_rvv_scalar_broadcast_add",
        emitc_route="rvv-generic-scalar-broadcast-add-emitc-route",
        typed_compute_op="tcrv_rvv.binary",
        memory_form="rhs-scalar-broadcast",
        rhs_initializer="(int32_t)-37",
        expected_expression="lhs[index] + rhs_scalar",
    ),
    "scalar_broadcast_sub": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["scalar_broadcast_sub"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-sub.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_scalar_broadcast_sub",
        function_name="tcrv_emitc_pre_realized_body_scalar_broadcast_sub_kernel_pre_realized_body_rvv_scalar_broadcast_sub",
    ),
    "scalar_broadcast_mul": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["scalar_broadcast_mul"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-mul.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_scalar_broadcast_mul",
        function_name="tcrv_emitc_pre_realized_body_scalar_broadcast_mul_kernel_pre_realized_body_rvv_scalar_broadcast_mul",
    ),
    "runtime_i32_splat_store": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["runtime_i32_splat_store"],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-runtime-i32-splat-store.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_runtime_i32_splat_store",
        function_name="tcrv_emitc_pre_realized_body_runtime_i32_splat_store_kernel_pre_realized_body_rvv_runtime_i32_splat_store",
    ),
    "standalone_reduce_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["standalone_reduce_add"],
        kind="standalone_reduce_add",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-standalone-reduce-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_standalone_reduce_add",
        function_name="tcrv_emitc_pre_realized_body_standalone_reduce_add_kernel_pre_realized_body_rvv_standalone_reduce_add",
    ),
    "standalone_reduce_min": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["standalone_reduce_min"],
        kind="standalone_reduce_min",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-standalone-reduce-min.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_standalone_reduce_min",
        function_name="tcrv_emitc_pre_realized_body_standalone_reduce_min_kernel_pre_realized_body_rvv_standalone_reduce_min",
    ),
    "standalone_reduce_max": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["standalone_reduce_max"],
        kind="standalone_reduce_max",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-standalone-reduce-max.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_standalone_reduce_max",
        function_name="tcrv_emitc_pre_realized_body_standalone_reduce_max_kernel_pre_realized_body_rvv_standalone_reduce_max",
    ),
    "computed_mask_standalone_reduce_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "computed_mask_standalone_reduce_add"
        ],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-standalone-reduce-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="rvv_pre_cm_standalone_reduce",
        function_name="tcrv_emitc_pre_cm_standalone_reduce_kernel_rvv_pre_cm_standalone_reduce",
    ),
    "runtime_scalar_cmp_masked_standalone_reduce_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "runtime_scalar_cmp_masked_standalone_reduce_add"
        ],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="rvv_pre_rt_scalar_cm_standalone_reduce",
        function_name="tcrv_emitc_pre_rt_scalar_cm_standalone_reduce_kernel_rvv_pre_rt_scalar_cm_standalone_reduce",
    ),
    "computed_mask_standalone_reduce_min": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "computed_mask_standalone_reduce_min"
        ],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-standalone-reduce-min.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="rvv_pre_cm_standalone_reduce_min",
        function_name="tcrv_emitc_pre_cm_standalone_reduce_min_kernel_rvv_pre_cm_standalone_reduce_min",
    ),
    "computed_mask_standalone_reduce_max": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "computed_mask_standalone_reduce_max"
        ],
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-standalone-reduce-max.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="rvv_pre_cm_standalone_reduce_max",
        function_name="tcrv_emitc_pre_cm_standalone_reduce_max_kernel_rvv_pre_cm_standalone_reduce_max",
    ),
    "i64_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"],
        kind="i64_add",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-i64-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_i64_add",
        function_name="tcrv_emitc_pre_realized_body_i64_add_kernel_pre_realized_body_rvv_i64_add",
        lhs_initializer="(int64_t)(7000000000LL + (int64_t)(index * 3000003LL))",
        rhs_initializer="(int64_t)(-2000000000LL + (int64_t)(index * 5000005LL))",
        expected_expression="lhs[index] + rhs[index]",
        out_initializer="(int64_t)0x5a5a5a5a5a5a5a5aLL",
        sew="64",
        element_c_type="int64_t",
        config_contract="rvv-selected-body-sew64-lmul-m1-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew64-lmul-m1",
    ),
    "lmul_m2_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"],
        kind="lmul_m2_add",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-lmul-m2-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_lmul_m2_add",
        function_name="tcrv_emitc_pre_realized_body_lmul_m2_add_kernel_pre_realized_body_rvv_lmul_m2_add",
        lmul="m2",
        config_contract="rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m2",
    ),
    "widen_i32_to_i64": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"],
        kind="widen_i32_to_i64",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-widen-i32-to-i64.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_widen_i32_to_i64",
        external_abi_name="rvv-generic-widen-i32-to-i64-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_body_widen_i32_to_i64_kernel_pre_realized_body_rvv_widen_i32_to_i64",
        emitc_route="rvv-generic-widen-i32-to-i64-emitc-route",
        typed_compute_op="tcrv_rvv.widening_convert",
        memory_form="unit-stride-conversion",
        lhs_initializer="(int32_t)((int)(index % 29) - 14)",
        rhs_initializer="unused",
        expected_expression="(int64_t)lhs[index]",
        out_initializer=I64_OUT_SENTINEL,
        lmul="m2",
        sew="64",
        element_c_type="int64_t",
        config_contract="rvv-selected-body-sew64-lmul-m2-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew64-lmul-m2",
    ),
    "widen_i16_to_i32": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"],
        kind="widen_i16_to_i32",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-widen-i16-to-i32.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_widen_i16_to_i32",
        external_abi_name="rvv-generic-widen-i16-to-i32-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_body_widen_i16_to_i32_kernel_pre_realized_body_rvv_widen_i16_to_i32",
        emitc_route="rvv-generic-widen-i16-to-i32-emitc-route",
        typed_compute_op="tcrv_rvv.widening_convert",
        memory_form="unit-stride-conversion",
        lhs_initializer="(int16_t)(((index % 2) == 0) ? -((int)(index % 127) + 1) : ((int)(index % 127) + 1))",
        rhs_initializer="unused",
        expected_expression="(int32_t)lhs[index]",
        out_initializer=OUT_SENTINEL,
        lmul="m1",
        sew="32",
        element_c_type="int32_t",
        config_contract="rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m1",
    ),
    "widening_macc_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"],
        kind="widening_macc_add",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-widening-macc-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_widening_macc_add",
        external_abi_name="rvv-generic-widening-macc-add-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_body_widening_macc_add_kernel_pre_realized_body_rvv_widening_macc_add",
        emitc_route="rvv-generic-widening-macc-add-emitc-route",
        typed_compute_op="tcrv_rvv.widening_macc",
        memory_form="vector-rhs-load",
        lhs_initializer="(int16_t)(((index % 4) < 2) ? -((int)(index % 97) + 2) : ((int)(index % 97) + 3))",
        rhs_initializer="(int16_t)(((index % 3) == 0) ? -((int)(index % 41) + 5) : ((int)(index % 41) + 7))",
        source_initializer="(int32_t)(300 + (int32_t)(index * 11))",
        expected_expression="(int32_t)(acc[index] + (int32_t)lhs[index] * (int32_t)rhs[index])",
        out_initializer=OUT_SENTINEL,
        lmul="m1",
        sew="32",
        element_c_type="int32_t",
        config_contract="rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m1",
    ),
    "widening_dot_reduce_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"],
        kind="widening_dot_reduce_add",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-widening-dot-reduce-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_widening_dot_reduce_add",
        external_abi_name="rvv-generic-widening-dot-reduce-add-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_body_widening_dot_reduce_add_kernel_pre_realized_body_rvv_widening_dot_reduce_add",
        emitc_route="rvv-generic-widening-dot-reduce-add-emitc-route",
        typed_compute_op="tcrv_rvv.widening_dot_reduce",
        memory_form="vector-rhs-load",
        lhs_initializer="(int16_t)(((index % 4) < 2) ? -((int)(index % 53) + 2) : ((int)(index % 53) + 5))",
        rhs_initializer="(int16_t)(((index % 5) == 0) ? -((int)(index % 37) + 3) : ((int)(index % 37) + 7))",
        source_initializer="(int32_t)17",
        expected_expression="(int32_t)(acc[0] + sum_i((int32_t)lhs[i] * (int32_t)rhs[i]))",
        out_initializer=OUT_SENTINEL,
        lmul="m1",
        sew="32",
        element_c_type="int32_t",
        config_contract="rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m1",
    ),
    "strided_input_widening_dot_reduce_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"],
        kind="strided_input_widening_dot_reduce_add",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-strided-input-widening-dot-reduce-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="rvv_strided_input_dot",
        external_abi_name="rvv-generic-strided-input-widening-dot-reduce-add-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_strided_dot_kernel_rvv_strided_input_dot",
        emitc_route="rvv-generic-strided-input-widening-dot-reduce-add-emitc-route",
        typed_compute_op="tcrv_rvv.widening_dot_reduce",
        memory_form="strided-input-widening-dot-reduce",
        lhs_initializer="(int16_t)(((index % 4) < 2) ? -((int)(index % 59) + 3) : ((int)(index % 59) + 6))",
        rhs_initializer="(int16_t)(((index % 5) == 0) ? -((int)(index % 43) + 4) : ((int)(index % 43) + 9))",
        source_initializer="(int32_t)31",
        expected_expression=(
            "(int32_t)(acc[0] + sum_i((int32_t)lhs[i * lhs_stride] * "
            "(int32_t)rhs[i * rhs_stride]))"
        ),
        out_initializer=OUT_SENTINEL,
        lmul="m1",
        sew="32",
        element_c_type="int32_t",
        config_contract="rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m1",
    ),
    "computed_masked_widening_dot_reduce_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"],
        kind="computed_masked_widening_dot_reduce_add",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="pre_realized_body_rvv_masked_widening_dot_reduce_add",
        external_abi_name="rvv-generic-computed-masked-widening-dot-reduce-add-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_body_masked_widening_dot_reduce_add_kernel_pre_realized_body_rvv_masked_widening_dot_reduce_add",
        emitc_route="rvv-generic-computed-masked-widening-dot-reduce-add-emitc-route",
        typed_compute_op="tcrv_rvv.masked_widening_dot_reduce",
        memory_form="computed-mask-unit-stride-widening-dot-reduce",
        lhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(10 + (int32_t)index) "
            ": (int32_t)(100 + (int32_t)index))"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(50 + (int32_t)index) "
            ": (int32_t)(20 + (int32_t)index))"
        ),
        source_initializer="(int32_t)29",
        expected_expression=(
            "(int32_t)(acc[0] + sum_i(cmp_lhs[i] < cmp_rhs[i] ? "
            "(int32_t)dot_lhs[i] * (int32_t)dot_rhs[i] : 0))"
        ),
        out_initializer=OUT_SENTINEL,
        lmul="m1",
        sew="32",
        element_c_type="int32_t",
        config_contract="rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m1",
    ),
    "computed_masked_strided_input_widening_dot_reduce_add": replace(
        EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS["add"],
        kind="computed_masked_strided_input_widening_dot_reduce_add",
        input_path=Path("test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir"),
        input_mode="pre-realized-selected-body",
        selected_variant="rvv_computed_mask_strided_input_dot",
        external_abi_name="rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-callable-c-abi.v1",
        function_name="tcrv_emitc_pre_realized_masked_strided_dot_kernel_rvv_computed_mask_strided_input_dot",
        emitc_route="rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-emitc-route",
        typed_compute_op="tcrv_rvv.masked_widening_dot_reduce",
        memory_form="computed-mask-strided-input-widening-dot-reduce",
        lhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(10 + (int32_t)index) "
            ": (int32_t)(100 + (int32_t)index))"
        ),
        rhs_initializer=(
            "(int32_t)(((index % 4) == 0 || (index % 4) == 3) "
            "? (int32_t)(50 + (int32_t)index) "
            ": (int32_t)(20 + (int32_t)index))"
        ),
        source_initializer="(int32_t)37",
        expected_expression=(
            "(int32_t)(acc[0] + sum_i(cmp_lhs[i] < cmp_rhs[i] ? "
            "(int32_t)dot_lhs[i * lhs_stride] * "
            "(int32_t)dot_rhs[i * rhs_stride] : 0))"
        ),
        out_initializer=OUT_SENTINEL,
        lmul="m1",
        sew="32",
        element_c_type="int32_t",
        config_contract="rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1",
        bounded_slice="multi-vl-selected-body-sew32-lmul-m1",
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
EXPECTED_I64_RUNTIME_PARAMETERS = (
    {
        "c_name": "lhs",
        "c_type": "const int64_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "rhs",
        "c_type": "const int64_t *",
        "role": "rhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "out",
        "c_type": "int64_t *",
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
EXPECTED_STRIDED_LOAD_UNIT_STORE_RUNTIME_PARAMETERS = (
    {
        "c_name": "src",
        "c_type": "const int32_t *",
        "role": "source-input-buffer",
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
        "c_name": "stride_bytes",
        "c_type": "size_t",
        "role": "source-byte-stride",
        "ownership": "target-export-abi-owned",
    },
)
EXPECTED_UNIT_LOAD_STRIDED_STORE_RUNTIME_PARAMETERS = (
    {
        "c_name": "src",
        "c_type": "const int32_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "dst",
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
        "c_name": "dst_stride_bytes",
        "c_type": "size_t",
        "role": "destination-byte-stride",
        "ownership": "target-export-abi-owned",
    },
)
EXPECTED_INDEXED_GATHER_RUNTIME_PARAMETERS = (
    {
        "c_name": "data",
        "c_type": "const int32_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "index",
        "c_type": "const uint32_t *",
        "role": "index-input-buffer",
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
EXPECTED_INDEXED_SCATTER_RUNTIME_PARAMETERS = (
    {
        "c_name": "src",
        "c_type": "const int32_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "index",
        "c_type": "const uint32_t *",
        "role": "index-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "dst",
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
EXPECTED_MASKED_MEMORY_RUNTIME_PARAMETERS = (
    {
        "c_name": "src",
        "c_type": "const int32_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "mask",
        "c_type": "const int32_t *",
        "role": "mask-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "dst",
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
EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS = (
    {
        "c_name": "cmp_lhs",
        "c_type": "const int32_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "cmp_rhs",
        "c_type": "const int32_t *",
        "role": "rhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "src",
        "c_type": "const int32_t *",
        "role": "source-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "dst",
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
EXPECTED_COMPUTED_MASK_SELECT_RUNTIME_PARAMETERS = (
    {
        "c_name": "cmp_lhs",
        "c_type": "const int32_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "cmp_rhs",
        "c_type": "const int32_t *",
        "role": "rhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "true_value",
        "c_type": "const int32_t *",
        "role": "true-value-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "false_value",
        "c_type": "const int32_t *",
        "role": "false-value-input-buffer",
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
EXPECTED_RUNTIME_SCALAR_CMP_SELECT_RUNTIME_PARAMETERS = (
    EXPECTED_RUNTIME_PARAMETERS[0],
    {
        "c_name": "rhs_scalar",
        "c_type": "int32_t",
        "role": "rhs-scalar-value",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_COMPUTED_MASK_SELECT_RUNTIME_PARAMETERS[2],
    EXPECTED_COMPUTED_MASK_SELECT_RUNTIME_PARAMETERS[3],
    EXPECTED_RUNTIME_PARAMETERS[2],
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_RUNTIME_PARAMETERS = (
    {
        "c_name": "cmp_lhs_a",
        "c_type": "const int32_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "rhs_scalar_a",
        "c_type": "int32_t",
        "role": "rhs-scalar-value",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "cmp_lhs_b",
        "c_type": "const int32_t *",
        "role": "rhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "rhs_scalar_b",
        "c_type": "int32_t",
        "role": "rhs-secondary-scalar-value",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_COMPUTED_MASK_SELECT_RUNTIME_PARAMETERS[2],
    EXPECTED_COMPUTED_MASK_SELECT_RUNTIME_PARAMETERS[3],
    EXPECTED_RUNTIME_PARAMETERS[2],
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_RUNTIME_SCALAR_CMP_MASKED_STORE_RUNTIME_PARAMETERS = (
    EXPECTED_RUNTIME_PARAMETERS[0],
    {
        "c_name": "rhs_scalar",
        "c_type": "int32_t",
        "role": "rhs-scalar-value",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "src",
        "c_type": "const int32_t *",
        "role": "source-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "dst",
        "c_type": "int32_t *",
        "role": "output-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_COMPUTED_MASK_STRIDED_STORE_RUNTIME_PARAMETERS = (
    *EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS,
    {
        "c_name": "dst_stride_bytes",
        "c_type": "size_t",
        "role": "destination-byte-stride",
        "ownership": "target-export-abi-owned",
    },
)
EXPECTED_COMPUTED_MASK_STRIDED_LOAD_RUNTIME_PARAMETERS = (
    *EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS,
    {
        "c_name": "src_stride_bytes",
        "c_type": "size_t",
        "role": "source-byte-stride",
        "ownership": "target-export-abi-owned",
    },
)
EXPECTED_COMPUTED_MASK_INDEXED_GATHER_RUNTIME_PARAMETERS = (
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[0],
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[1],
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[2],
    {
        "c_name": "index",
        "c_type": "const uint32_t *",
        "role": "index-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[3],
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[4],
)
EXPECTED_COMPUTED_MASK_INDEXED_SCATTER_RUNTIME_PARAMETERS = (
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[0],
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[1],
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[2],
    {
        "c_name": "index",
        "c_type": "const uint32_t *",
        "role": "index-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[3],
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[4],
)
EXPECTED_COMPUTED_MASK_SEGMENT2_LOAD_RUNTIME_PARAMETERS = (
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[0],
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[1],
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[2],
    {
        "c_name": "out0",
        "c_type": "int32_t *",
        "role": "segment-field0-output-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "out1",
        "c_type": "int32_t *",
        "role": "segment-field1-output-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[4],
)
EXPECTED_COMPUTED_MASK_SEGMENT2_STORE_RUNTIME_PARAMETERS = (
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[0],
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[1],
    {
        "c_name": "src0",
        "c_type": "const int32_t *",
        "role": "segment-field0-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "src1",
        "c_type": "const int32_t *",
        "role": "segment-field1-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "dst",
        "c_type": "int32_t *",
        "role": "segment-interleaved-output-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[4],
)
EXPECTED_SEGMENT2_RUNTIME_PARAMETERS = (
    {
        "c_name": "src",
        "c_type": "const int32_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "out0",
        "c_type": "int32_t *",
        "role": "segment-field0-output-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "out1",
        "c_type": "int32_t *",
        "role": "segment-field1-output-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "n",
        "c_type": "size_t",
        "role": "runtime-element-count",
        "ownership": "target-export-abi-owned",
    },
)
EXPECTED_SEGMENT2_INTERLEAVE_RUNTIME_PARAMETERS = (
    {
        "c_name": "src0",
        "c_type": "const int32_t *",
        "role": "segment-field0-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "src1",
        "c_type": "const int32_t *",
        "role": "segment-field1-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "dst",
        "c_type": "int32_t *",
        "role": "segment-interleaved-output-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "n",
        "c_type": "size_t",
        "role": "runtime-element-count",
        "ownership": "target-export-abi-owned",
    },
)
EXPECTED_SCALAR_BROADCAST_RUNTIME_PARAMETERS = (
    EXPECTED_RUNTIME_PARAMETERS[0],
    {
        "c_name": "rhs_scalar",
        "c_type": "int32_t",
        "role": "rhs-scalar-value",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[2],
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_RUNTIME_SCALAR_SPLAT_STORE_RUNTIME_PARAMETERS = (
    {
        "c_name": "rhs_scalar",
        "c_type": "int32_t",
        "role": "rhs-scalar-value",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[2],
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_STANDALONE_REDUCE_RUNTIME_PARAMETERS = (
    EXPECTED_RUNTIME_PARAMETERS[0],
    {
        "c_name": "acc",
        "c_type": "const int32_t *",
        "role": "accumulator-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[2],
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_COMPUTED_MASK_STANDALONE_REDUCE_RUNTIME_PARAMETERS = (
    EXPECTED_COMPUTED_MASK_SELECT_RUNTIME_PARAMETERS[0],
    EXPECTED_COMPUTED_MASK_SELECT_RUNTIME_PARAMETERS[1],
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[2],
    {
        "c_name": "acc",
        "c_type": "const int32_t *",
        "role": "accumulator-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[2],
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_RUNTIME_PARAMETERS = (
    {
        "c_name": "cmp_lhs",
        "c_type": "const int32_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_SCALAR_CMP_SELECT_RUNTIME_PARAMETERS[1],
    EXPECTED_COMPUTED_MASK_MEMORY_RUNTIME_PARAMETERS[2],
    {
        "c_name": "acc",
        "c_type": "const int32_t *",
        "role": "accumulator-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[2],
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_MACC_RUNTIME_PARAMETERS = (
    EXPECTED_RUNTIME_PARAMETERS[0],
    EXPECTED_RUNTIME_PARAMETERS[1],
    {
        "c_name": "acc",
        "c_type": "const int32_t *",
        "role": "accumulator-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[2],
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_COMPUTED_MASKED_MACC_RUNTIME_PARAMETERS = (
    EXPECTED_COMPUTED_MASK_SELECT_RUNTIME_PARAMETERS[0],
    EXPECTED_COMPUTED_MASK_SELECT_RUNTIME_PARAMETERS[1],
    {
        "c_name": "lhs",
        "c_type": "const int32_t *",
        "role": "dot-lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "rhs",
        "c_type": "const int32_t *",
        "role": "dot-rhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "acc",
        "c_type": "const int32_t *",
        "role": "accumulator-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[2],
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_RUNTIME_SCALAR_COMPUTED_MASKED_MACC_RUNTIME_PARAMETERS = (
    {
        "c_name": "cmp_lhs",
        "c_type": "const int32_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_SCALAR_CMP_SELECT_RUNTIME_PARAMETERS[1],
    {
        "c_name": "lhs",
        "c_type": "const int32_t *",
        "role": "dot-lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "rhs",
        "c_type": "const int32_t *",
        "role": "dot-rhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "acc",
        "c_type": "const int32_t *",
        "role": "accumulator-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[2],
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_WIDENING_CONVERSION_RUNTIME_PARAMETERS = (
    EXPECTED_RUNTIME_PARAMETERS[0],
    {
        "c_name": "out",
        "c_type": "int64_t *",
        "role": "output-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_WIDEN_I16_TO_I32_RUNTIME_PARAMETERS = (
    {
        "c_name": "lhs",
        "c_type": "const int16_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "out",
        "c_type": "int32_t *",
        "role": "output-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_WIDENING_MACC_RUNTIME_PARAMETERS = (
    {
        "c_name": "lhs",
        "c_type": "const int16_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "rhs",
        "c_type": "const int16_t *",
        "role": "rhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "acc",
        "c_type": "const int32_t *",
        "role": "accumulator-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "out",
        "c_type": "int32_t *",
        "role": "output-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_STRIDED_INPUT_WIDENING_DOT_RUNTIME_PARAMETERS = (
    {
        "c_name": "lhs",
        "c_type": "const int16_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "rhs",
        "c_type": "const int16_t *",
        "role": "rhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "acc",
        "c_type": "const int32_t *",
        "role": "accumulator-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "out",
        "c_type": "int32_t *",
        "role": "output-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[3],
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
)
EXPECTED_COMPUTED_MASK_WIDENING_DOT_RUNTIME_PARAMETERS = (
    {
        "c_name": "cmp_lhs",
        "c_type": "const int32_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "cmp_rhs",
        "c_type": "const int32_t *",
        "role": "rhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "lhs",
        "c_type": "const int16_t *",
        "role": "dot-lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "rhs",
        "c_type": "const int16_t *",
        "role": "dot-rhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "acc",
        "c_type": "const int32_t *",
        "role": "accumulator-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "out",
        "c_type": "int32_t *",
        "role": "output-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[3],
)
EXPECTED_COMPUTED_MASK_STRIDED_INPUT_WIDENING_DOT_RUNTIME_PARAMETERS = (
    {
        "c_name": "cmp_lhs",
        "c_type": "const int32_t *",
        "role": "lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "cmp_rhs",
        "c_type": "const int32_t *",
        "role": "rhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "lhs",
        "c_type": "const int16_t *",
        "role": "dot-lhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "rhs",
        "c_type": "const int16_t *",
        "role": "dot-rhs-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "acc",
        "c_type": "const int32_t *",
        "role": "accumulator-input-buffer",
        "ownership": "target-export-abi-owned",
    },
    {
        "c_name": "out",
        "c_type": "int32_t *",
        "role": "output-buffer",
        "ownership": "target-export-abi-owned",
    },
    EXPECTED_RUNTIME_PARAMETERS[3],
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


def require_no_op_invocation(text: str, op_name: str, context: str) -> None:
    pattern = re.compile(rf"^\s*(?:%[^\s]+ = )?{re.escape(op_name)}\b", re.MULTILINE)
    if not pattern.search(text):
        return
    raise EvidenceError(f"{context}: forbidden op invocation {op_name!r} present")


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
        "rvv_selected_body_operation": expectation.selected_body_operation,
        "rvv_selected_body_typed_compute_op": expectation.typed_compute_op,
        "tcrv_rvv.config_contract": expectation.config_contract,
        "tcrv_rvv.sew": expectation.sew,
        "tcrv_rvv.lmul": expectation.lmul,
        "tcrv_rvv.tail_policy": "agnostic",
        "tcrv_rvv.mask_policy": "agnostic",
        "tcrv_rvv.memory_form": expectation.memory_form,
        "tcrv_rvv.bounded_slice": expectation.bounded_slice,
    }
    if expectation.is_masked_unit_store:
        per_op_metadata["tcrv_rvv.tail_policy"] = "undisturbed"
        per_op_metadata["tcrv_rvv.mask_policy"] = "undisturbed"
    if expectation.kind in BASE_MEMORY_TARGET_LEAF_PROFILE_BY_KIND:
        per_op_metadata.update(
            {
                "tcrv_rvv.runtime_control_plan": RUNTIME_AVL_VL_CONTROL_PLAN,
                "tcrv_rvv.base_memory_movement_route_family_plan": (
                    BASE_MEMORY_MOVEMENT_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.target_leaf_profile": (
                    BASE_MEMORY_TARGET_LEAF_PROFILE_BY_KIND[expectation.kind]
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    BASE_MEMORY_PROVIDER_SUPPORTED_MIRROR_BY_KIND[
                        expectation.kind
                    ]
                ),
                "tcrv_rvv.required_header_declarations": (
                    BASE_MEMORY_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    BASE_MEMORY_C_TYPE_MAPPING_BY_KIND[expectation.kind]
                ),
            }
        )
    if expectation.is_reduce_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.reduction_accumulator_layout": REDUCE_ADD_ACCUMULATOR_LAYOUT,
                "tcrv_rvv.reduction_result_layout": REDUCE_ADD_RESULT_LAYOUT,
                "tcrv_rvv.reduction_store_vl": REDUCE_ADD_STORE_VL,
                "tcrv_rvv.route_operand_binding_plan": (
                    REDUCE_ADD_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    REDUCE_ADD_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_standalone_reduce:
        per_op_metadata.update(
            {
                "tcrv_rvv.reduction_accumulator_layout": (
                    STANDALONE_REDUCE_ACCUMULATOR_LAYOUT
                ),
                "tcrv_rvv.reduction_result_layout": (
                    STANDALONE_REDUCE_RESULT_LAYOUT
                ),
                "tcrv_rvv.reduction_store_vl": STANDALONE_REDUCE_STORE_VL,
                "tcrv_rvv.standalone_reduction_route_family_plan": (
                    STANDALONE_REDUCTION_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.target_leaf_profile": (
                    STANDALONE_REDUCE_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    STANDALONE_REDUCE_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    STANDALONE_REDUCE_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": STANDALONE_REDUCE_C_TYPE_MAPPING,
                "tcrv_rvv.route_operand_binding_plan": (
                    standalone_reduce_route_operand_binding_plan(expectation.kind)
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    standalone_reduce_route_operand_binding_operands(
                        expectation.kind
                    )
                ),
            }
        )
    if expectation.is_computed_mask_standalone_reduce:
        per_op_metadata.update(
            {
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_zeroing_requirement": (
                    computed_mask_standalone_reduce_inactive_contract(
                        expectation.kind
                    )
                ),
                "tcrv_rvv.reduction_accumulator_layout": (
                    STANDALONE_REDUCE_ACCUMULATOR_LAYOUT
                ),
                "tcrv_rvv.reduction_result_layout": (
                    STANDALONE_REDUCE_RESULT_LAYOUT
                ),
                "tcrv_rvv.reduction_store_vl": STANDALONE_REDUCE_STORE_VL,
                "tcrv_rvv.standalone_reduction_route_family_plan": (
                    STANDALONE_REDUCTION_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.target_leaf_profile": (
                    COMPUTED_MASK_STANDALONE_REDUCE_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    COMPUTED_MASK_STANDALONE_REDUCE_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    STANDALONE_REDUCE_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    COMPUTED_MASK_STANDALONE_REDUCE_C_TYPE_MAPPING
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    computed_mask_standalone_reduce_route_operand_binding_plan(
                        expectation.kind
                    )
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    computed_mask_standalone_reduce_route_operand_binding_operands(
                        expectation.kind
                    )
                ),
            }
        )
        if expectation.is_computed_mask_standalone_reduce_add:
            per_op_metadata.update(
                {
                    "tcrv_rvv.accumulation_route_family_plan": (
                        COMPUTED_MASK_ACCUMULATION_ROUTE_FAMILY_PLAN
                    ),
                    "tcrv_rvv.accumulation_compute_suffix": (
                        COMPUTED_MASK_ACCUMULATION_STANDALONE_REDUCE_SUFFIX
                    ),
                    "tcrv_rvv.accumulation_mask_producer_source": (
                        COMPUTED_MASK_ACCUMULATION_VECTOR_PRODUCER_SOURCE
                    ),
                    "tcrv_rvv.accumulation_accumulator_contract": (
                        COMPUTED_MASK_ACCUMULATION_STANDALONE_ACCUMULATOR_CONTRACT
                    ),
                    "tcrv_rvv.accumulation_result_contract": (
                        COMPUTED_MASK_ACCUMULATION_STANDALONE_RESULT_CONTRACT
                    ),
                    "tcrv_rvv.accumulation_scalar_carry_contract": (
                        COMPUTED_MASK_ACCUMULATION_STANDALONE_SCALAR_CARRY_CONTRACT
                    ),
                }
            )
    if expectation.is_runtime_scalar_computed_mask_standalone_reduce:
        per_op_metadata.update(
            {
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_zeroing_requirement": (
                    COMPUTED_MASK_STANDALONE_REDUCE_INACTIVE_ZEROING
                ),
                "tcrv_rvv.reduction_accumulator_layout": (
                    STANDALONE_REDUCE_ACCUMULATOR_LAYOUT
                ),
                "tcrv_rvv.reduction_result_layout": (
                    STANDALONE_REDUCE_RESULT_LAYOUT
                ),
                "tcrv_rvv.reduction_store_vl": STANDALONE_REDUCE_STORE_VL,
                "tcrv_rvv.standalone_reduction_route_family_plan": (
                    STANDALONE_REDUCTION_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.target_leaf_profile": (
                    RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    STANDALONE_REDUCE_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_C_TYPE_MAPPING
                ),
                "tcrv_rvv.accumulation_route_family_plan": (
                    COMPUTED_MASK_ACCUMULATION_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.accumulation_compute_suffix": (
                    COMPUTED_MASK_ACCUMULATION_STANDALONE_REDUCE_SUFFIX
                ),
                "tcrv_rvv.accumulation_mask_producer_source": (
                    COMPUTED_MASK_ACCUMULATION_RUNTIME_SCALAR_PRODUCER_SOURCE
                ),
                "tcrv_rvv.accumulation_accumulator_contract": (
                    COMPUTED_MASK_ACCUMULATION_STANDALONE_ACCUMULATOR_CONTRACT
                ),
                "tcrv_rvv.accumulation_result_contract": (
                    COMPUTED_MASK_ACCUMULATION_STANDALONE_RESULT_CONTRACT
                ),
                "tcrv_rvv.accumulation_scalar_carry_contract": (
                    COMPUTED_MASK_ACCUMULATION_STANDALONE_SCALAR_CARRY_CONTRACT
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    RUNTIME_SCALAR_COMPUTED_MASK_STANDALONE_REDUCE_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_scalar_broadcast_elementwise:
        per_op_metadata.update(
            {
                "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan": (
                    SCALAR_BROADCAST_ELEMENTWISE_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    scalar_broadcast_route_operand_binding_plan(expectation.kind)
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    scalar_broadcast_route_operand_binding_operands(
                        expectation.kind
                    )
                ),
            }
        )
    if expectation.is_runtime_scalar_splat_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.runtime_scalar_splat_store_route_family_plan": (
                    RUNTIME_SCALAR_SPLAT_STORE_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.target_leaf_profile": (
                    RUNTIME_SCALAR_SPLAT_STORE_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    RUNTIME_SCALAR_SPLAT_STORE_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    RUNTIME_SCALAR_SPLAT_STORE_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    RUNTIME_SCALAR_SPLAT_STORE_C_TYPE_MAPPING
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    RUNTIME_SCALAR_SPLAT_STORE_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    RUNTIME_SCALAR_SPLAT_STORE_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.kind in {"add", "sub", "mul"}:
        plan = f"rvv-route-operand-binding:{expectation.kind}.v1"
        per_op_metadata.update(
            {
                "tcrv_rvv.route_operand_binding_plan": plan,
                "tcrv_rvv.route_operand_binding_operands": (
                    BINARY_ROUTE_OPERAND_BINDING_OPERANDS.format(plan=plan)
                ),
            }
        )
    if (
        expectation.is_scalar_broadcast_elementwise
        or expectation.is_runtime_scalar_splat_store
        or expectation.is_standalone_reduce
        or expectation.is_computed_mask_standalone_reduce
        or expectation.is_runtime_scalar_computed_mask_standalone_reduce
    ):
        per_op_metadata["tcrv_rvv.runtime_control_plan"] = (
            RUNTIME_AVL_VL_CONTROL_PLAN
        )
    if expectation.is_masked_elementwise:
        per_op_metadata.update(
            {
                "tcrv_rvv.mask_role": MASKED_ADD_MASK_ROLE,
                "tcrv_rvv.mask_source": MASKED_ADD_MASK_SOURCE,
                "tcrv_rvv.inactive_lane_contract": MASKED_ADD_INACTIVE_LANE_CONTRACT,
                "tcrv_rvv.masked_passthrough_layout": MASKED_ADD_PASSTHROUGH_LAYOUT,
                "tcrv_rvv.route_operand_binding_plan": (
                    masked_elementwise_route_operand_binding_plan(
                        expectation.kind
                    )
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    masked_elementwise_route_operand_binding_operands(
                        expectation.kind
                    )
                ),
            }
        )
    if expectation.is_cmp_select:
        per_op_metadata.update(
            {
                "tcrv_rvv.runtime_control_plan": RUNTIME_AVL_VL_CONTROL_PLAN,
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.source_memory_form": (
                    COMPUTED_MASK_SELECT_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    COMPUTED_MASK_SELECT_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.select_layout": PLAIN_COMPARE_SELECT_LAYOUT,
                "tcrv_rvv.plain_compare_select_route_family_plan": (
                    PLAIN_COMPARE_SELECT_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.target_leaf_profile": (
                    PLAIN_COMPARE_SELECT_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    PLAIN_COMPARE_SELECT_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    PLAIN_COMPARE_SELECT_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    PLAIN_COMPARE_SELECT_C_TYPE_MAPPING
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    CMP_SELECT_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    CMP_SELECT_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_macc_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.macc_accumulator_layout": MACC_ADD_ACCUMULATOR_LAYOUT,
                "tcrv_rvv.macc_result_layout": MACC_ADD_RESULT_LAYOUT,
                "tcrv_rvv.route_operand_binding_plan": (
                    MACC_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    MACC_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_computed_masked_macc_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.target_leaf_profile": (
                    COMPUTED_MASKED_MACC_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    COMPUTED_MASKED_MACC_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    COMPUTED_MASKED_MACC_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    COMPUTED_MASKED_MACC_C_TYPE_MAPPING
                ),
                "tcrv_rvv.accumulation_route_family_plan": (
                    COMPUTED_MASK_ACCUMULATION_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.accumulation_compute_suffix": (
                    COMPUTED_MASK_ACCUMULATION_VECTOR_MACC_SUFFIX
                ),
                "tcrv_rvv.accumulation_mask_producer_source": (
                    COMPUTED_MASK_ACCUMULATION_VECTOR_PRODUCER_SOURCE
                ),
                "tcrv_rvv.accumulation_accumulator_contract": (
                    COMPUTED_MASK_ACCUMULATION_VECTOR_MACC_ACCUMULATOR_CONTRACT
                ),
                "tcrv_rvv.accumulation_result_contract": (
                    COMPUTED_MASK_ACCUMULATION_VECTOR_MACC_RESULT_CONTRACT
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_contract": (
                    COMPUTED_MASKED_MACC_ADD_INACTIVE_LANE_CONTRACT
                ),
                "tcrv_rvv.masked_passthrough_layout": (
                    COMPUTED_MASKED_MACC_ADD_PASSTHROUGH_LAYOUT
                ),
                "tcrv_rvv.source_memory_form": MASKED_MEMORY_SOURCE_MEMORY_FORM,
                "tcrv_rvv.destination_memory_form": (
                    MASKED_MEMORY_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.indexed_memory_layout": (
                    COMPUTED_MASKED_MACC_ADD_MEMORY_LAYOUT
                ),
                "tcrv_rvv.macc_accumulator_layout": MACC_ADD_ACCUMULATOR_LAYOUT,
                "tcrv_rvv.macc_result_layout": MACC_ADD_RESULT_LAYOUT,
                "tcrv_rvv.route_operand_binding_plan": (
                    COMPUTED_MASKED_MACC_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    COMPUTED_MASKED_MACC_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_runtime_scalar_computed_masked_macc_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.runtime_control_plan": RUNTIME_AVL_VL_CONTROL_PLAN,
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.target_leaf_profile": (
                    RUNTIME_SCALAR_COMPUTED_MASKED_MACC_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    RUNTIME_SCALAR_COMPUTED_MASKED_MACC_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    RUNTIME_SCALAR_COMPUTED_MASKED_MACC_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    RUNTIME_SCALAR_COMPUTED_MASKED_MACC_C_TYPE_MAPPING
                ),
                "tcrv_rvv.accumulation_route_family_plan": (
                    COMPUTED_MASK_ACCUMULATION_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.accumulation_compute_suffix": (
                    COMPUTED_MASK_ACCUMULATION_VECTOR_MACC_SUFFIX
                ),
                "tcrv_rvv.accumulation_mask_producer_source": (
                    COMPUTED_MASK_ACCUMULATION_RUNTIME_SCALAR_PRODUCER_SOURCE
                ),
                "tcrv_rvv.accumulation_accumulator_contract": (
                    COMPUTED_MASK_ACCUMULATION_VECTOR_MACC_ACCUMULATOR_CONTRACT
                ),
                "tcrv_rvv.accumulation_result_contract": (
                    COMPUTED_MASK_ACCUMULATION_VECTOR_MACC_RESULT_CONTRACT
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_contract": (
                    COMPUTED_MASKED_MACC_ADD_INACTIVE_LANE_CONTRACT
                ),
                "tcrv_rvv.masked_passthrough_layout": (
                    COMPUTED_MASKED_MACC_ADD_PASSTHROUGH_LAYOUT
                ),
                "tcrv_rvv.source_memory_form": MASKED_MEMORY_SOURCE_MEMORY_FORM,
                "tcrv_rvv.destination_memory_form": (
                    MASKED_MEMORY_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.indexed_memory_layout": (
                    RUNTIME_SCALAR_COMPUTED_MASKED_MACC_ADD_MEMORY_LAYOUT
                ),
                "tcrv_rvv.macc_accumulator_layout": MACC_ADD_ACCUMULATOR_LAYOUT,
                "tcrv_rvv.macc_result_layout": MACC_ADD_RESULT_LAYOUT,
                "tcrv_rvv.route_operand_binding_plan": (
                    RUNTIME_SCALAR_COMPUTED_MASKED_MACC_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    RUNTIME_SCALAR_COMPUTED_MASKED_MACC_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_strided_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.strided_memory_layout": STRIDED_ADD_MEMORY_LAYOUT,
                "tcrv_rvv.lhs_stride_source": STRIDED_ADD_LHS_STRIDE_SOURCE,
                "tcrv_rvv.rhs_stride_source": STRIDED_ADD_RHS_STRIDE_SOURCE,
                "tcrv_rvv.out_stride_source": STRIDED_ADD_OUT_STRIDE_SOURCE,
                "tcrv_rvv.route_operand_binding_plan": (
                    STRIDED_ADD_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    STRIDED_ADD_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_strided_load_unit_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.strided_memory_layout": (
                    STRIDED_LOAD_UNIT_STORE_MEMORY_LAYOUT
                ),
                "tcrv_rvv.source_stride_source": (
                    STRIDED_LOAD_UNIT_STORE_SOURCE_STRIDE_SOURCE
                ),
                "tcrv_rvv.source_memory_form": (
                    STRIDED_LOAD_UNIT_STORE_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    STRIDED_LOAD_UNIT_STORE_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    STRIDED_LOAD_UNIT_STORE_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    STRIDED_LOAD_UNIT_STORE_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_unit_load_strided_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.strided_memory_layout": (
                    UNIT_LOAD_STRIDED_STORE_MEMORY_LAYOUT
                ),
                "tcrv_rvv.destination_stride_source": (
                    UNIT_LOAD_STRIDED_STORE_DESTINATION_STRIDE_SOURCE
                ),
                "tcrv_rvv.source_memory_form": (
                    UNIT_LOAD_STRIDED_STORE_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    UNIT_LOAD_STRIDED_STORE_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    UNIT_LOAD_STRIDED_STORE_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    UNIT_LOAD_STRIDED_STORE_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_indexed_gather_unit_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.indexed_memory_layout": INDEXED_GATHER_MEMORY_LAYOUT,
                "tcrv_rvv.index_source": INDEXED_GATHER_INDEX_SOURCE,
                "tcrv_rvv.index_eew": INDEXED_GATHER_INDEX_EEW,
                "tcrv_rvv.offset_unit": INDEXED_GATHER_OFFSET_UNIT,
                "tcrv_rvv.indexed_data_memory_form": INDEXED_GATHER_DATA_MEMORY_FORM,
                "tcrv_rvv.destination_memory_form": (
                    INDEXED_GATHER_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    INDEXED_GATHER_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    INDEXED_GATHER_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_indexed_scatter_unit_load:
        per_op_metadata.update(
            {
                "tcrv_rvv.indexed_memory_layout": INDEXED_SCATTER_MEMORY_LAYOUT,
                "tcrv_rvv.index_source": INDEXED_SCATTER_INDEX_SOURCE,
                "tcrv_rvv.index_eew": INDEXED_SCATTER_INDEX_EEW,
                "tcrv_rvv.offset_unit": INDEXED_SCATTER_OFFSET_UNIT,
                "tcrv_rvv.index_uniqueness": INDEXED_SCATTER_INDEX_UNIQUENESS,
                "tcrv_rvv.source_memory_form": (
                    INDEXED_SCATTER_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.indexed_destination_memory_form": (
                    INDEXED_SCATTER_INDEXED_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    INDEXED_SCATTER_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    INDEXED_SCATTER_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    INDEXED_SCATTER_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_masked_unit_load_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.masked_memory_layout": MASKED_MEMORY_LAYOUT,
                "tcrv_rvv.mask_role": MASKED_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": MASKED_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": MASKED_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_contract": (
                    MASKED_MEMORY_INACTIVE_LANE_CONTRACT
                ),
                "tcrv_rvv.masked_passthrough_layout": (
                    MASKED_MEMORY_PASSTHROUGH_LAYOUT
                ),
                "tcrv_rvv.source_memory_form": MASKED_MEMORY_SOURCE_MEMORY_FORM,
                "tcrv_rvv.destination_memory_form": (
                    MASKED_MEMORY_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    MASKED_UNIT_LOAD_STORE_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    MASKED_UNIT_LOAD_STORE_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_masked_unit_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.masked_memory_layout": MASKED_STORE_LAYOUT,
                "tcrv_rvv.mask_role": MASKED_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": MASKED_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": MASKED_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_contract": (
                    MASKED_STORE_INACTIVE_LANE_CONTRACT
                ),
                "tcrv_rvv.masked_passthrough_layout": (
                    MASKED_STORE_PASSTHROUGH_LAYOUT
                ),
                "tcrv_rvv.source_memory_form": MASKED_MEMORY_SOURCE_MEMORY_FORM,
                "tcrv_rvv.destination_memory_form": (
                    MASKED_STORE_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    MASKED_UNIT_STORE_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    MASKED_UNIT_STORE_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_computed_masked_unit_load_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.runtime_control_plan": RUNTIME_AVL_VL_CONTROL_PLAN,
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.masked_memory_layout": COMPUTED_MASK_MEMORY_LAYOUT,
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_contract": (
                    MASKED_MEMORY_INACTIVE_LANE_CONTRACT
                ),
                "tcrv_rvv.masked_passthrough_layout": (
                    MASKED_MEMORY_PASSTHROUGH_LAYOUT
                ),
                "tcrv_rvv.source_memory_form": MASKED_MEMORY_SOURCE_MEMORY_FORM,
                "tcrv_rvv.destination_memory_form": (
                    MASKED_MEMORY_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    COMPUTED_MASK_UNIT_LOAD_STORE_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    COMPUTED_MASK_UNIT_LOAD_STORE_ROUTE_OPERAND_BINDING_OPERANDS
                ),
                "tcrv_rvv.computed_mask_memory_route_family_plan": (
                    COMPUTED_MASK_MEMORY_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.computed_mask_memory_mask_producer_source": (
                    COMPUTED_MASK_MEMORY_VECTOR_COMPARE_PRODUCER_SOURCE
                ),
                "tcrv_rvv.target_leaf_profile": (
                    COMPUTED_MASK_UNIT_LOAD_STORE_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    COMPUTED_MASK_UNIT_LOAD_STORE_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    COMPUTED_MASK_UNIT_LOAD_STORE_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    COMPUTED_MASK_UNIT_LOAD_STORE_C_TYPE_MAPPING
                ),
            }
        )
    if expectation.is_computed_mask_select:
        per_op_metadata.update(
            {
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.select_layout": COMPUTED_MASK_SELECT_LAYOUT,
                "tcrv_rvv.route_operand_binding_plan": (
                    COMPUTED_MASK_SELECT_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    COMPUTED_MASK_SELECT_ROUTE_OPERAND_BINDING_OPERANDS
                ),
                "tcrv_rvv.computed_mask_select_route_family_plan": (
                    COMPUTED_MASK_SELECT_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.computed_mask_select_mask_producer_source": (
                    COMPUTED_MASK_SELECT_VECTOR_COMPARE_PRODUCER_SOURCE
                ),
            }
        )
    if expectation.is_runtime_scalar_compare_select:
        per_op_metadata.update(
            {
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.source_memory_form": (
                    COMPUTED_MASK_SELECT_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    COMPUTED_MASK_SELECT_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.select_layout": COMPUTED_MASK_SELECT_LAYOUT,
                "tcrv_rvv.route_operand_binding_plan": (
                    RUNTIME_SCALAR_CMP_SELECT_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    RUNTIME_SCALAR_CMP_SELECT_ROUTE_OPERAND_BINDING_OPERANDS
                ),
                "tcrv_rvv.computed_mask_select_route_family_plan": (
                    COMPUTED_MASK_SELECT_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.computed_mask_select_mask_producer_source": (
                    COMPUTED_MASK_SELECT_RUNTIME_SCALAR_PRODUCER_SOURCE
                ),
                "tcrv_rvv.target_leaf_profile": (
                    RUNTIME_SCALAR_CMP_SELECT_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    RUNTIME_SCALAR_CMP_SELECT_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    RUNTIME_SCALAR_CMP_SELECT_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    RUNTIME_SCALAR_CMP_SELECT_C_TYPE_MAPPING
                ),
            }
        )
    if expectation.is_runtime_scalar_dual_compare_mask_and_select:
        per_op_metadata.update(
            {
                "tcrv_rvv.runtime_control_plan": RUNTIME_AVL_VL_CONTROL_PLAN,
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.secondary_compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.mask_role": "predicate-mask-produced-by-mask-and",
                "tcrv_rvv.mask_source": (
                    "mask-and-of-two-runtime-scalar-compare-produced-masks"
                ),
                "tcrv_rvv.mask_memory_form": "composed-compare-produced-mask",
                "tcrv_rvv.mask_composition": "and",
                "tcrv_rvv.source_memory_form": (
                    COMPUTED_MASK_SELECT_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    COMPUTED_MASK_SELECT_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.select_layout": COMPUTED_MASK_SELECT_LAYOUT,
                "tcrv_rvv.route_operand_binding_plan": (
                    RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_ROUTE_OPERAND_BINDING_OPERANDS
                ),
                "tcrv_rvv.computed_mask_select_route_family_plan": (
                    COMPUTED_MASK_SELECT_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.computed_mask_select_mask_producer_source": (
                    COMPUTED_MASK_SELECT_DUAL_RUNTIME_SCALAR_PRODUCER_SOURCE
                ),
                "tcrv_rvv.target_leaf_profile": (
                    RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    RUNTIME_SCALAR_DUAL_CMP_MASK_AND_SELECT_C_TYPE_MAPPING
                ),
            }
        )
    if expectation.is_runtime_scalar_computed_mask_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.tail_policy": "undisturbed",
                "tcrv_rvv.mask_policy": "undisturbed",
                "tcrv_rvv.runtime_control_plan": RUNTIME_AVL_VL_CONTROL_PLAN,
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.masked_memory_layout": (
                    RUNTIME_SCALAR_CMP_MASKED_STORE_MEMORY_LAYOUT
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_contract": (
                    MASKED_STORE_INACTIVE_LANE_CONTRACT
                ),
                "tcrv_rvv.masked_passthrough_layout": (
                    MASKED_STORE_PASSTHROUGH_LAYOUT
                ),
                "tcrv_rvv.source_memory_form": (
                    RUNTIME_SCALAR_CMP_MASKED_STORE_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    RUNTIME_SCALAR_CMP_MASKED_STORE_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    RUNTIME_SCALAR_CMP_MASKED_STORE_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    RUNTIME_SCALAR_CMP_MASKED_STORE_ROUTE_OPERAND_BINDING_OPERANDS
                ),
                "tcrv_rvv.computed_mask_memory_route_family_plan": (
                    COMPUTED_MASK_MEMORY_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.computed_mask_memory_mask_producer_source": (
                    COMPUTED_MASK_MEMORY_RUNTIME_SCALAR_PRODUCER_SOURCE
                ),
                "tcrv_rvv.target_leaf_profile": (
                    RUNTIME_SCALAR_CMP_MASKED_STORE_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    RUNTIME_SCALAR_CMP_MASKED_STORE_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    RUNTIME_SCALAR_CMP_MASKED_STORE_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    RUNTIME_SCALAR_CMP_MASKED_STORE_C_TYPE_MAPPING
                ),
            }
        )
    if expectation.is_runtime_scalar_computed_mask_load_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.runtime_control_plan": RUNTIME_AVL_VL_CONTROL_PLAN,
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.masked_memory_layout": (
                    RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_MEMORY_LAYOUT
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_contract": (
                    MASKED_MEMORY_INACTIVE_LANE_CONTRACT
                ),
                "tcrv_rvv.masked_passthrough_layout": (
                    MASKED_MEMORY_PASSTHROUGH_LAYOUT
                ),
                "tcrv_rvv.source_memory_form": MASKED_MEMORY_SOURCE_MEMORY_FORM,
                "tcrv_rvv.destination_memory_form": (
                    MASKED_MEMORY_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_ROUTE_OPERAND_BINDING_OPERANDS
                ),
                "tcrv_rvv.computed_mask_memory_route_family_plan": (
                    COMPUTED_MASK_MEMORY_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.computed_mask_memory_mask_producer_source": (
                    COMPUTED_MASK_MEMORY_RUNTIME_SCALAR_PRODUCER_SOURCE
                ),
                "tcrv_rvv.target_leaf_profile": (
                    RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    RUNTIME_SCALAR_CMP_MASKED_LOAD_STORE_C_TYPE_MAPPING
                ),
            }
        )
    if expectation.is_computed_masked_strided_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.runtime_control_plan": RUNTIME_AVL_VL_CONTROL_PLAN,
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.masked_memory_layout": (
                    COMPUTED_MASK_STRIDED_STORE_MEMORY_LAYOUT
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_contract": (
                    COMPUTED_MASK_STRIDED_STORE_INACTIVE_LANE_CONTRACT
                ),
                "tcrv_rvv.masked_passthrough_layout": (
                    COMPUTED_MASK_STRIDED_STORE_PASSTHROUGH_LAYOUT
                ),
                "tcrv_rvv.source_memory_form": MASKED_MEMORY_SOURCE_MEMORY_FORM,
                "tcrv_rvv.destination_memory_form": (
                    COMPUTED_MASK_STRIDED_STORE_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.strided_memory_layout": (
                    COMPUTED_MASK_STRIDED_STORE_MEMORY_LAYOUT
                ),
                "tcrv_rvv.destination_stride_source": (
                    COMPUTED_MASK_STRIDED_STORE_DESTINATION_STRIDE_SOURCE
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    COMPUTED_MASK_STRIDED_STORE_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    COMPUTED_MASK_STRIDED_STORE_ROUTE_OPERAND_BINDING_OPERANDS
                ),
                "tcrv_rvv.computed_mask_memory_route_family_plan": (
                    COMPUTED_MASK_MEMORY_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.computed_mask_memory_mask_producer_source": (
                    COMPUTED_MASK_MEMORY_VECTOR_COMPARE_PRODUCER_SOURCE
                ),
                "tcrv_rvv.target_leaf_profile": (
                    COMPUTED_MASK_STRIDED_STORE_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    COMPUTED_MASK_STRIDED_STORE_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    COMPUTED_MASK_STRIDED_STORE_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    COMPUTED_MASK_STRIDED_STORE_C_TYPE_MAPPING
                ),
            }
        )
    if expectation.is_computed_masked_strided_load_unit_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.masked_memory_layout": (
                    COMPUTED_MASK_STRIDED_LOAD_MEMORY_LAYOUT
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_contract": (
                    MASKED_MEMORY_INACTIVE_LANE_CONTRACT
                ),
                "tcrv_rvv.masked_passthrough_layout": (
                    MASKED_MEMORY_PASSTHROUGH_LAYOUT
                ),
                "tcrv_rvv.source_memory_form": (
                    COMPUTED_MASK_STRIDED_LOAD_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    MASKED_MEMORY_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.strided_memory_layout": (
                    COMPUTED_MASK_STRIDED_LOAD_MEMORY_LAYOUT
                ),
                "tcrv_rvv.source_stride_source": (
                    COMPUTED_MASK_STRIDED_LOAD_SOURCE_STRIDE_SOURCE
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    COMPUTED_MASK_STRIDED_LOAD_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    COMPUTED_MASK_STRIDED_LOAD_ROUTE_OPERAND_BINDING_OPERANDS
                ),
                "tcrv_rvv.computed_mask_memory_route_family_plan": (
                    COMPUTED_MASK_MEMORY_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.computed_mask_memory_mask_producer_source": (
                    COMPUTED_MASK_MEMORY_VECTOR_COMPARE_PRODUCER_SOURCE
                ),
                "tcrv_rvv.target_leaf_profile": (
                    COMPUTED_MASK_STRIDED_LOAD_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    COMPUTED_MASK_STRIDED_LOAD_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    COMPUTED_MASK_STRIDED_LOAD_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    COMPUTED_MASK_STRIDED_LOAD_C_TYPE_MAPPING
                ),
            }
        )
    if expectation.is_computed_masked_indexed_gather_load_unit_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.masked_memory_layout": (
                    COMPUTED_MASK_INDEXED_GATHER_MEMORY_LAYOUT
                ),
                "tcrv_rvv.indexed_memory_layout": (
                    COMPUTED_MASK_INDEXED_GATHER_MEMORY_LAYOUT
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_contract": (
                    MASKED_MEMORY_INACTIVE_LANE_CONTRACT
                ),
                "tcrv_rvv.masked_passthrough_layout": (
                    MASKED_MEMORY_PASSTHROUGH_LAYOUT
                ),
                "tcrv_rvv.index_source": (
                    COMPUTED_MASK_INDEXED_GATHER_INDEX_SOURCE
                ),
                "tcrv_rvv.index_eew": COMPUTED_MASK_INDEXED_GATHER_INDEX_EEW,
                "tcrv_rvv.offset_unit": (
                    COMPUTED_MASK_INDEXED_GATHER_OFFSET_UNIT
                ),
                "tcrv_rvv.indexed_data_memory_form": (
                    COMPUTED_MASK_INDEXED_GATHER_DATA_MEMORY_FORM
                ),
                "tcrv_rvv.source_memory_form": (
                    COMPUTED_MASK_INDEXED_GATHER_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    MASKED_MEMORY_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    COMPUTED_MASK_INDEXED_GATHER_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    COMPUTED_MASK_INDEXED_GATHER_ROUTE_OPERAND_BINDING_OPERANDS
                ),
                "tcrv_rvv.computed_mask_memory_route_family_plan": (
                    COMPUTED_MASK_MEMORY_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.computed_mask_memory_mask_producer_source": (
                    COMPUTED_MASK_MEMORY_VECTOR_COMPARE_PRODUCER_SOURCE
                ),
                "tcrv_rvv.target_leaf_profile": (
                    COMPUTED_MASK_INDEXED_GATHER_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    COMPUTED_MASK_INDEXED_GATHER_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    COMPUTED_MASK_INDEXED_GATHER_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    COMPUTED_MASK_INDEXED_GATHER_C_TYPE_MAPPING
                ),
            }
        )
    if expectation.is_computed_masked_indexed_scatter_store_unit_load:
        per_op_metadata.update(
            {
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.masked_memory_layout": (
                    COMPUTED_MASK_INDEXED_SCATTER_MEMORY_LAYOUT
                ),
                "tcrv_rvv.indexed_memory_layout": (
                    COMPUTED_MASK_INDEXED_SCATTER_MEMORY_LAYOUT
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_contract": (
                    COMPUTED_MASK_INDEXED_SCATTER_INACTIVE_LANE_CONTRACT
                ),
                "tcrv_rvv.masked_passthrough_layout": (
                    COMPUTED_MASK_INDEXED_SCATTER_PASSTHROUGH_LAYOUT
                ),
                "tcrv_rvv.index_source": (
                    COMPUTED_MASK_INDEXED_SCATTER_INDEX_SOURCE
                ),
                "tcrv_rvv.index_eew": COMPUTED_MASK_INDEXED_SCATTER_INDEX_EEW,
                "tcrv_rvv.offset_unit": (
                    COMPUTED_MASK_INDEXED_SCATTER_OFFSET_UNIT
                ),
                "tcrv_rvv.index_uniqueness": (
                    COMPUTED_MASK_INDEXED_SCATTER_INDEX_UNIQUENESS
                ),
                "tcrv_rvv.source_memory_form": (
                    COMPUTED_MASK_INDEXED_SCATTER_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    COMPUTED_MASK_INDEXED_SCATTER_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.indexed_destination_memory_form": (
                    COMPUTED_MASK_INDEXED_SCATTER_INDEXED_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    COMPUTED_MASK_INDEXED_SCATTER_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    COMPUTED_MASK_INDEXED_SCATTER_ROUTE_OPERAND_BINDING_OPERANDS
                ),
                "tcrv_rvv.computed_mask_memory_route_family_plan": (
                    COMPUTED_MASK_MEMORY_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.computed_mask_memory_mask_producer_source": (
                    COMPUTED_MASK_MEMORY_VECTOR_COMPARE_PRODUCER_SOURCE
                ),
                "tcrv_rvv.target_leaf_profile": (
                    COMPUTED_MASK_INDEXED_SCATTER_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    COMPUTED_MASK_INDEXED_SCATTER_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    COMPUTED_MASK_INDEXED_SCATTER_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    COMPUTED_MASK_INDEXED_SCATTER_C_TYPE_MAPPING
                ),
            }
        )
    if expectation.is_computed_masked_segment2_load_unit_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.masked_memory_layout": (
                    COMPUTED_MASK_SEGMENT2_LOAD_MEMORY_LAYOUT
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_contract": (
                    MASKED_MEMORY_INACTIVE_LANE_CONTRACT
                ),
                "tcrv_rvv.masked_passthrough_layout": (
                    MASKED_MEMORY_PASSTHROUGH_LAYOUT
                ),
                "tcrv_rvv.source_memory_form": (
                    COMPUTED_MASK_SEGMENT2_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    COMPUTED_MASK_SEGMENT2_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.segment_memory_layout": (
                    COMPUTED_MASK_SEGMENT2_LOAD_MEMORY_LAYOUT
                ),
                "tcrv_rvv.segment_count": "2",
                "tcrv_rvv.segment_tuple_c_type": SEGMENT2_TUPLE_C_TYPE,
                "tcrv_rvv.segment_load_intrinsic": (
                    COMPUTED_MASK_SEGMENT2_LOAD_INTRINSIC
                ),
                "tcrv_rvv.segment_store_intrinsic": (
                    COMPUTED_MASK_SEGMENT2_TUPLE_CREATE_INTRINSIC
                ),
                "tcrv_rvv.segment_field_extract_intrinsic": (
                    SEGMENT2_FIELD_EXTRACT_INTRINSIC
                ),
                "tcrv_rvv.field0_role": SEGMENT2_FIELD0_ROLE,
                "tcrv_rvv.field1_role": SEGMENT2_FIELD1_ROLE,
                "tcrv_rvv.field0_name": "masked_segment2_field0_vec",
                "tcrv_rvv.field1_name": "masked_segment2_field1_vec",
                "tcrv_rvv.field0_destination_memory_form": (
                    COMPUTED_MASK_SEGMENT2_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.field1_destination_memory_form": (
                    COMPUTED_MASK_SEGMENT2_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    COMPUTED_MASK_SEGMENT2_LOAD_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    COMPUTED_MASK_SEGMENT2_LOAD_ROUTE_OPERAND_BINDING_OPERANDS
                ),
                "tcrv_rvv.computed_mask_memory_route_family_plan": (
                    COMPUTED_MASK_MEMORY_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.computed_mask_memory_mask_producer_source": (
                    COMPUTED_MASK_MEMORY_VECTOR_COMPARE_PRODUCER_SOURCE
                ),
                "tcrv_rvv.target_leaf_profile": (
                    COMPUTED_MASK_SEGMENT2_LOAD_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    COMPUTED_MASK_SEGMENT2_LOAD_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    COMPUTED_MASK_SEGMENT2_LOAD_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    COMPUTED_MASK_SEGMENT2_LOAD_C_TYPE_MAPPING
                ),
            }
        )
    if expectation.is_computed_masked_segment2_store_unit_load:
        per_op_metadata.update(
            {
                "tcrv_rvv.compare_predicate_kind": (
                    expectation.compare_predicate_kind
                ),
                "tcrv_rvv.masked_memory_layout": (
                    COMPUTED_MASK_SEGMENT2_STORE_MEMORY_LAYOUT
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_contract": (
                    MASKED_STORE_INACTIVE_LANE_CONTRACT
                ),
                "tcrv_rvv.masked_passthrough_layout": (
                    MASKED_STORE_PASSTHROUGH_LAYOUT
                ),
                "tcrv_rvv.source_memory_form": (
                    COMPUTED_MASK_SEGMENT2_STORE_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    COMPUTED_MASK_SEGMENT2_STORE_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.segment_memory_layout": (
                    COMPUTED_MASK_SEGMENT2_STORE_MEMORY_LAYOUT
                ),
                "tcrv_rvv.segment_count": "2",
                "tcrv_rvv.segment_tuple_c_type": SEGMENT2_TUPLE_C_TYPE,
                "tcrv_rvv.segment_store_intrinsic": (
                    COMPUTED_MASK_SEGMENT2_STORE_INTRINSIC
                ),
                "tcrv_rvv.segment_tuple_create_intrinsic": (
                    SEGMENT2_TUPLE_CREATE_INTRINSIC
                ),
                "tcrv_rvv.field0_role": SEGMENT2_FIELD0_INPUT_ROLE,
                "tcrv_rvv.field1_role": SEGMENT2_FIELD1_INPUT_ROLE,
                "tcrv_rvv.field0_name": "masked_segment2_store_field0_vec",
                "tcrv_rvv.field1_name": "masked_segment2_store_field1_vec",
                "tcrv_rvv.field0_source_memory_form": (
                    SEGMENT2_FIELD_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.field1_source_memory_form": (
                    SEGMENT2_FIELD_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    COMPUTED_MASK_SEGMENT2_STORE_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    COMPUTED_MASK_SEGMENT2_STORE_ROUTE_OPERAND_BINDING_OPERANDS
                ),
                "tcrv_rvv.computed_mask_memory_route_family_plan": (
                    COMPUTED_MASK_MEMORY_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.computed_mask_memory_mask_producer_source": (
                    COMPUTED_MASK_MEMORY_VECTOR_COMPARE_PRODUCER_SOURCE
                ),
                "tcrv_rvv.target_leaf_profile": (
                    COMPUTED_MASK_SEGMENT2_STORE_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    COMPUTED_MASK_SEGMENT2_STORE_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    COMPUTED_MASK_SEGMENT2_STORE_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    COMPUTED_MASK_SEGMENT2_STORE_C_TYPE_MAPPING
                ),
            }
        )
    if expectation.is_segment2_deinterleave_unit_store:
        per_op_metadata.update(
            {
                "tcrv_rvv.runtime_control_plan": RUNTIME_AVL_VL_CONTROL_PLAN,
                "tcrv_rvv.segment2_memory_route_family_plan": (
                    SEGMENT2_MEMORY_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.target_leaf_profile": (
                    SEGMENT2_DEINTERLEAVE_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    SEGMENT2_DEINTERLEAVE_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    SEGMENT2_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    SEGMENT2_DEINTERLEAVE_C_TYPE_MAPPING
                ),
                "tcrv_rvv.segment_memory_layout": SEGMENT2_MEMORY_LAYOUT,
                "tcrv_rvv.segment_count": "2",
                "tcrv_rvv.segment_tuple_c_type": SEGMENT2_TUPLE_C_TYPE,
                "tcrv_rvv.segment_load_intrinsic": SEGMENT2_LOAD_INTRINSIC,
                "tcrv_rvv.segment_field_extract_intrinsic": (
                    SEGMENT2_FIELD_EXTRACT_INTRINSIC
                ),
                "tcrv_rvv.source_memory_form": SEGMENT2_SOURCE_MEMORY_FORM,
                "tcrv_rvv.destination_memory_form": (
                    SEGMENT2_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.field0_role": SEGMENT2_FIELD0_ROLE,
                "tcrv_rvv.field1_role": SEGMENT2_FIELD1_ROLE,
                "tcrv_rvv.field0_name": "field0_vec",
                "tcrv_rvv.field1_name": "field1_vec",
                "tcrv_rvv.field0_destination_memory_form": (
                    SEGMENT2_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.field1_destination_memory_form": (
                    SEGMENT2_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    SEGMENT2_DEINTERLEAVE_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    SEGMENT2_DEINTERLEAVE_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_segment2_interleave_unit_load:
        per_op_metadata.update(
            {
                "tcrv_rvv.runtime_control_plan": RUNTIME_AVL_VL_CONTROL_PLAN,
                "tcrv_rvv.segment2_memory_route_family_plan": (
                    SEGMENT2_MEMORY_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.target_leaf_profile": (
                    SEGMENT2_INTERLEAVE_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    SEGMENT2_INTERLEAVE_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    SEGMENT2_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": (
                    SEGMENT2_INTERLEAVE_C_TYPE_MAPPING
                ),
                "tcrv_rvv.segment_memory_layout": (
                    SEGMENT2_INTERLEAVE_MEMORY_LAYOUT
                ),
                "tcrv_rvv.segment_count": "2",
                "tcrv_rvv.segment_tuple_c_type": SEGMENT2_TUPLE_C_TYPE,
                "tcrv_rvv.segment_store_intrinsic": SEGMENT2_STORE_INTRINSIC,
                "tcrv_rvv.segment_tuple_create_intrinsic": (
                    SEGMENT2_TUPLE_CREATE_INTRINSIC
                ),
                "tcrv_rvv.source_memory_form": (
                    SEGMENT2_FIELD_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    SEGMENT2_INTERLEAVED_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.field0_role": SEGMENT2_FIELD0_INPUT_ROLE,
                "tcrv_rvv.field1_role": SEGMENT2_FIELD1_INPUT_ROLE,
                "tcrv_rvv.field0_name": "field0_vec",
                "tcrv_rvv.field1_name": "field1_vec",
                "tcrv_rvv.field0_source_memory_form": (
                    SEGMENT2_FIELD_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.field1_source_memory_form": (
                    SEGMENT2_FIELD_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.route_operand_binding_plan": (
                    SEGMENT2_INTERLEAVE_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    SEGMENT2_INTERLEAVE_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_widen_i32_to_i64:
        per_op_metadata.update(
            {
                "tcrv_rvv.runtime_control_plan": RUNTIME_AVL_VL_CONTROL_PLAN,
                "tcrv_rvv.widening_conversion_route_family_plan": (
                    WIDENING_CONVERSION_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.target_leaf_profile": (
                    WIDEN_I32_TO_I64_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    WIDEN_I32_TO_I64_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    WIDENING_CONVERSION_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": WIDEN_I32_TO_I64_C_TYPE_MAPPING,
                "tcrv_rvv.source_sew": "32",
                "tcrv_rvv.source_lmul": "m1",
                "tcrv_rvv.dest_sew": "64",
                "tcrv_rvv.dest_lmul": "m2",
                "tcrv_rvv.conversion_relation": WIDENING_CONVERSION_RELATION,
                "tcrv_rvv.route_operand_binding_plan": (
                    WIDEN_I32_TO_I64_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    WIDEN_I32_TO_I64_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_widen_i16_to_i32:
        per_op_metadata.update(
            {
                "tcrv_rvv.runtime_control_plan": RUNTIME_AVL_VL_CONTROL_PLAN,
                "tcrv_rvv.widening_conversion_route_family_plan": (
                    WIDENING_CONVERSION_ROUTE_FAMILY_PLAN
                ),
                "tcrv_rvv.target_leaf_profile": (
                    WIDEN_I16_TO_I32_TARGET_LEAF_PROFILE
                ),
                "tcrv_rvv.provider_supported_mirror": (
                    WIDEN_I16_TO_I32_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    WIDENING_CONVERSION_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": WIDEN_I16_TO_I32_C_TYPE_MAPPING,
                "tcrv_rvv.source_sew": "16",
                "tcrv_rvv.source_lmul": "mf2",
                "tcrv_rvv.dest_sew": "32",
                "tcrv_rvv.dest_lmul": "m1",
                "tcrv_rvv.conversion_relation": WIDEN_I16_TO_I32_CONVERSION_RELATION,
                "tcrv_rvv.route_operand_binding_plan": (
                    WIDEN_I16_TO_I32_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    WIDEN_I16_TO_I32_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if (
        expectation.is_widening_macc_add
        or expectation.is_widening_dot_reduce_add
        or expectation.is_strided_input_widening_dot_reduce_add
        or expectation.is_computed_masked_widening_dot_reduce_add
        or expectation.is_computed_masked_strided_input_widening_dot_reduce_add
    ):
        per_op_metadata.update(
            {
                "tcrv_rvv.target_leaf_profile": CONTRACTION_TARGET_LEAF_PROFILE,
                "tcrv_rvv.provider_supported_mirror": (
                    CONTRACTION_PROVIDER_SUPPORTED_MIRROR
                ),
                "tcrv_rvv.required_header_declarations": (
                    CONTRACTION_REQUIRED_HEADER_DECLARATIONS
                ),
                "tcrv_rvv.c_type_mapping": CONTRACTION_C_TYPE_MAPPING,
                "tcrv_rvv.contraction_route_family_plan": (
                    "rvv-contraction-route-family-plan.v1"
                ),
            }
        )
    if expectation.is_widening_macc_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.source_sew": "16",
                "tcrv_rvv.source_lmul": "mf2",
                "tcrv_rvv.accumulator_sew": "32",
                "tcrv_rvv.accumulator_lmul": "m1",
                "tcrv_rvv.result_sew": "32",
                "tcrv_rvv.result_lmul": "m1",
                "tcrv_rvv.widening_macc_accumulator_layout": (
                    WIDENING_MACC_ACCUMULATOR_LAYOUT
                ),
                "tcrv_rvv.widening_macc_result_layout": (
                    WIDENING_MACC_RESULT_LAYOUT
                ),
                "tcrv_rvv.widening_macc_relation": WIDENING_MACC_RELATION,
                "tcrv_rvv.route_operand_binding_plan": (
                    WIDENING_MACC_ROUTE_OPERAND_BINDING_PLAN
                ),
                "tcrv_rvv.route_operand_binding_operands": (
                    WIDENING_MACC_ROUTE_OPERAND_BINDING_OPERANDS
                ),
            }
        )
    if expectation.is_widening_dot_reduce_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.source_sew": "16",
                "tcrv_rvv.source_lmul": "mf2",
                "tcrv_rvv.accumulator_sew": "32",
                "tcrv_rvv.accumulator_lmul": "m1",
                "tcrv_rvv.result_sew": "32",
                "tcrv_rvv.result_lmul": "m1",
                "tcrv_rvv.widening_dot_accumulator_layout": (
                    WIDENING_DOT_ACCUMULATOR_LAYOUT
                ),
                "tcrv_rvv.widening_dot_result_layout": WIDENING_DOT_RESULT_LAYOUT,
                "tcrv_rvv.widening_dot_relation": WIDENING_DOT_RELATION,
                "tcrv_rvv.widening_product_intrinsic": "__riscv_vwmul_vv_i32m1",
                "tcrv_rvv.widening_dot_reduction_store_vl": (
                    WIDENING_DOT_REDUCTION_STORE_VL
                ),
            }
        )
    if expectation.is_strided_input_widening_dot_reduce_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.source_sew": "16",
                "tcrv_rvv.source_lmul": "mf2",
                "tcrv_rvv.accumulator_sew": "32",
                "tcrv_rvv.accumulator_lmul": "m1",
                "tcrv_rvv.result_sew": "32",
                "tcrv_rvv.result_lmul": "m1",
                "tcrv_rvv.strided_memory_layout": (
                    STRIDED_INPUT_WIDENING_DOT_MEMORY_LAYOUT
                ),
                "tcrv_rvv.lhs_stride_source": (
                    STRIDED_INPUT_WIDENING_DOT_LHS_STRIDE_SOURCE
                ),
                "tcrv_rvv.rhs_stride_source": (
                    STRIDED_INPUT_WIDENING_DOT_RHS_STRIDE_SOURCE
                ),
                "tcrv_rvv.source_memory_form": (
                    STRIDED_INPUT_WIDENING_DOT_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    STRIDED_INPUT_WIDENING_DOT_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.widening_dot_accumulator_layout": (
                    WIDENING_DOT_ACCUMULATOR_LAYOUT
                ),
                "tcrv_rvv.widening_dot_result_layout": WIDENING_DOT_RESULT_LAYOUT,
                "tcrv_rvv.widening_dot_relation": WIDENING_DOT_RELATION,
                "tcrv_rvv.widening_product_intrinsic": "__riscv_vwmul_vv_i32m1",
                "tcrv_rvv.strided_load_intrinsic": (
                    STRIDED_INPUT_WIDENING_DOT_STRIDED_LOAD_INTRINSIC
                ),
                "tcrv_rvv.widening_dot_reduction_store_vl": (
                    WIDENING_DOT_REDUCTION_STORE_VL
                ),
            }
        )
    if expectation.is_computed_masked_widening_dot_reduce_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.source_sew": "16",
                "tcrv_rvv.source_lmul": "mf2",
                "tcrv_rvv.accumulator_sew": "32",
                "tcrv_rvv.accumulator_lmul": "m1",
                "tcrv_rvv.result_sew": "32",
                "tcrv_rvv.result_lmul": "m1",
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_zeroing_requirement": (
                    CONTRACTION_MASKED_INACTIVE_LANE_ZEROING_REQUIREMENT
                ),
                "tcrv_rvv.widening_dot_accumulator_layout": (
                    WIDENING_DOT_ACCUMULATOR_LAYOUT
                ),
                "tcrv_rvv.widening_dot_result_layout": WIDENING_DOT_RESULT_LAYOUT,
                "tcrv_rvv.widening_dot_relation": WIDENING_DOT_RELATION,
                "tcrv_rvv.widening_product_intrinsic": "__riscv_vwmul_vv_i32m1",
                "tcrv_rvv.masked_widening_product_intrinsic": (
                    "__riscv_vwmul_vv_i32m1_m"
                ),
                "tcrv_rvv.widening_dot_reduction_store_vl": (
                    WIDENING_DOT_REDUCTION_STORE_VL
                ),
            }
        )
    if expectation.is_computed_masked_strided_input_widening_dot_reduce_add:
        per_op_metadata.update(
            {
                "tcrv_rvv.source_sew": "16",
                "tcrv_rvv.source_lmul": "mf2",
                "tcrv_rvv.accumulator_sew": "32",
                "tcrv_rvv.accumulator_lmul": "m1",
                "tcrv_rvv.result_sew": "32",
                "tcrv_rvv.result_lmul": "m1",
                "tcrv_rvv.strided_memory_layout": (
                    COMPUTED_MASK_STRIDED_INPUT_WIDENING_DOT_MEMORY_LAYOUT
                ),
                "tcrv_rvv.lhs_stride_source": (
                    STRIDED_INPUT_WIDENING_DOT_LHS_STRIDE_SOURCE
                ),
                "tcrv_rvv.rhs_stride_source": (
                    STRIDED_INPUT_WIDENING_DOT_RHS_STRIDE_SOURCE
                ),
                "tcrv_rvv.source_memory_form": (
                    STRIDED_INPUT_WIDENING_DOT_SOURCE_MEMORY_FORM
                ),
                "tcrv_rvv.destination_memory_form": (
                    STRIDED_INPUT_WIDENING_DOT_DESTINATION_MEMORY_FORM
                ),
                "tcrv_rvv.mask_role": COMPUTED_MASK_MEMORY_MASK_ROLE,
                "tcrv_rvv.mask_source": COMPUTED_MASK_MEMORY_MASK_SOURCE,
                "tcrv_rvv.mask_memory_form": COMPUTED_MASK_MEMORY_MASK_FORM,
                "tcrv_rvv.inactive_lane_zeroing_requirement": (
                    CONTRACTION_MASKED_INACTIVE_LANE_ZEROING_REQUIREMENT
                ),
                "tcrv_rvv.widening_dot_accumulator_layout": (
                    WIDENING_DOT_ACCUMULATOR_LAYOUT
                ),
                "tcrv_rvv.widening_dot_result_layout": WIDENING_DOT_RESULT_LAYOUT,
                "tcrv_rvv.widening_dot_relation": WIDENING_DOT_RELATION,
                "tcrv_rvv.widening_product_intrinsic": "__riscv_vwmul_vv_i32m1",
                "tcrv_rvv.masked_widening_product_intrinsic": (
                    "__riscv_vwmul_vv_i32m1_m"
                ),
                "tcrv_rvv.strided_load_intrinsic": (
                    STRIDED_INPUT_WIDENING_DOT_STRIDED_LOAD_INTRINSIC
                ),
                "tcrv_rvv.widening_dot_reduction_store_vl": (
                    WIDENING_DOT_REDUCTION_STORE_VL
                ),
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
    require_contains(
        text,
        f"sew = {expectation.sew} : i64",
        "materialized selected-body MLIR SEW config",
    )
    if expectation.is_i64_add:
        require_contains(
            text,
            '!tcrv_rvv.vector<i64, "m1">',
            "materialized selected-body MLIR i64 vector type",
        )
    if expectation.is_lmul_m2_add:
        require_contains(
            text,
            '!tcrv_rvv.vector<i32, "m2">',
            "materialized selected-body MLIR i32 LMUL m2 vector type",
        )
    if expectation.is_widen_i32_to_i64:
        require_contains(
            text,
            '!tcrv_rvv.vector<i32, "m1">',
            "materialized selected-body MLIR widening conversion source vector type",
        )
        require_contains(
            text,
            '!tcrv_rvv.vector<i64, "m2">',
            "materialized selected-body MLIR widening conversion result vector type",
        )
        require_contains(
            text,
            "tcrv_rvv.widening_convert",
            "materialized selected-body MLIR widening conversion compute op",
        )
    if expectation.is_widen_i16_to_i32:
        require_contains(
            text,
            '!tcrv_rvv.vector<i16, "mf2">',
            "materialized selected-body MLIR i16mf2 widening conversion source vector type",
        )
        require_contains(
            text,
            '!tcrv_rvv.vector<i32, "m1">',
            "materialized selected-body MLIR i32m1 widening conversion result vector type",
        )
        require_contains(
            text,
            'kind = "sign_extend_widen_vf2"',
            "materialized selected-body MLIR sign-extend widening conversion kind",
        )
    if expectation.is_widening_macc_add:
        require_contains(
            text,
            'role = "accumulator-input-buffer"',
            "materialized selected-body MLIR widening macc accumulator ABI role",
        )
        require_contains(
            text,
            '!tcrv_rvv.vector<i16, "mf2">',
            "materialized selected-body MLIR widening macc source vector type",
        )
        require_contains(
            text,
            '!tcrv_rvv.vector<i32, "m1">',
            "materialized selected-body MLIR widening macc accumulator/result vector type",
        )
        require_contains(
            text,
            "tcrv_rvv.widening_macc",
            "materialized selected-body MLIR widening macc compute op",
        )
        require_contains(
            text,
            'kind = "signed_widening_macc_add"',
            "materialized selected-body MLIR widening macc kind",
        )
        require_contains(
            text,
            f'macc_relation = "{WIDENING_MACC_RELATION}"',
            "materialized selected-body MLIR widening macc relation",
        )
    if expectation.is_widening_dot_reduce_add:
        require_contains(
            text,
            'role = "accumulator-input-buffer"',
            "materialized selected-body MLIR widening dot accumulator seed ABI role",
        )
        require_contains(
            text,
            '!tcrv_rvv.vector<i16, "mf2">',
            "materialized selected-body MLIR widening dot source vector type",
        )
        require_contains(
            text,
            '!tcrv_rvv.vector<i32, "m1">',
            "materialized selected-body MLIR widening dot result vector type",
        )
        require_contains(
            text,
            "tcrv_rvv.widening_dot_reduce",
            "materialized selected-body MLIR widening dot compute op",
        )
        require_contains(
            text,
            'kind = "signed_widening_dot_reduce_add"',
            "materialized selected-body MLIR widening dot kind",
        )
        require_contains(
            text,
            f'dot_product_relation = "{WIDENING_DOT_RELATION}"',
            "materialized selected-body MLIR widening dot relation",
        )
    if expectation.is_strided_input_widening_dot_reduce_add:
        require_contains(
            text,
            'role = "lhs-input-stride"',
            "materialized selected-body MLIR strided dot lhs stride ABI role",
        )
        require_contains(
            text,
            'role = "rhs-input-stride"',
            "materialized selected-body MLIR strided dot rhs stride ABI role",
        )
        require_contains(
            text,
            '!tcrv_rvv.vector<i16, "mf2">',
            "materialized selected-body MLIR strided dot source vector type",
        )
        require_contains(
            text,
            "tcrv_rvv.strided_load",
            "materialized selected-body MLIR strided dot source loads",
        )
        require_contains(
            text,
            "tcrv_rvv.widening_dot_reduce",
            "materialized selected-body MLIR strided dot compute op",
        )
        require_contains(
            text,
            f'dot_product_relation = "{WIDENING_DOT_RELATION}"',
            "materialized selected-body MLIR strided dot relation",
        )
    if expectation.is_computed_masked_widening_dot_reduce_add:
        require_contains(
            text,
            'role = "dot-lhs-input-buffer"',
            "materialized selected-body MLIR masked widening dot lhs ABI role",
        )
        require_contains(
            text,
            'role = "dot-rhs-input-buffer"',
            "materialized selected-body MLIR masked widening dot rhs ABI role",
        )
        require_contains(
            text,
            'role = "accumulator-input-buffer"',
            "materialized selected-body MLIR masked widening dot accumulator seed ABI role",
        )
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR masked widening dot compare producer",
        )
        require_contains(
            text,
            '!tcrv_rvv.mask<i32, "m1">',
            "materialized selected-body MLIR masked widening dot mask type",
        )
        require_contains(
            text,
            '!tcrv_rvv.vector<i16, "mf2">',
            "materialized selected-body MLIR masked widening dot source vector type",
        )
        require_contains(
            text,
            '!tcrv_rvv.vector<i32, "m1">',
            "materialized selected-body MLIR masked widening dot result vector type",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_widening_dot_reduce",
            "materialized selected-body MLIR masked widening dot compute op",
        )
        require_contains(
            text,
            'kind = "signed_masked_widening_dot_reduce_add"',
            "materialized selected-body MLIR masked widening dot kind",
        )
        require_contains(
            text,
            f'mask_source = "{COMPUTED_MASK_MEMORY_MASK_SOURCE}"',
            "materialized selected-body MLIR masked widening dot mask source",
        )
        require_contains(
            text,
            f'dot_product_relation = "{WIDENING_DOT_RELATION}"',
            "materialized selected-body MLIR masked widening dot relation",
        )
    if expectation.is_computed_masked_strided_input_widening_dot_reduce_add:
        require_contains(
            text,
            'role = "dot-lhs-input-buffer"',
            "materialized selected-body MLIR masked strided dot lhs ABI role",
        )
        require_contains(
            text,
            'role = "dot-rhs-input-buffer"',
            "materialized selected-body MLIR masked strided dot rhs ABI role",
        )
        require_contains(
            text,
            'role = "lhs-input-stride"',
            "materialized selected-body MLIR masked strided dot lhs stride ABI role",
        )
        require_contains(
            text,
            'role = "rhs-input-stride"',
            "materialized selected-body MLIR masked strided dot rhs stride ABI role",
        )
        require_contains(
            text,
            'role = "accumulator-input-buffer"',
            "materialized selected-body MLIR masked strided dot accumulator seed ABI role",
        )
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR masked strided dot compare producer",
        )
        require_contains(
            text,
            '!tcrv_rvv.mask<i32, "m1">',
            "materialized selected-body MLIR masked strided dot mask type",
        )
        require_contains(
            text,
            '!tcrv_rvv.vector<i16, "mf2">',
            "materialized selected-body MLIR masked strided dot source vector type",
        )
        require_contains(
            text,
            "tcrv_rvv.strided_load",
            "materialized selected-body MLIR masked strided dot source loads",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_widening_dot_reduce",
            "materialized selected-body MLIR masked strided dot compute op",
        )
        require_contains(
            text,
            'kind = "signed_masked_widening_dot_reduce_add"',
            "materialized selected-body MLIR masked strided dot kind",
        )
        require_contains(
            text,
            f'mask_source = "{COMPUTED_MASK_MEMORY_MASK_SOURCE}"',
            "materialized selected-body MLIR masked strided dot mask source",
        )
        require_contains(
            text,
            f'dot_product_relation = "{WIDENING_DOT_RELATION}"',
            "materialized selected-body MLIR masked strided dot relation",
        )
    if expectation.is_rhs_broadcast:
        require_contains(
            text,
            "tcrv_rvv.broadcast_load",
            "materialized selected-body MLIR RHS broadcast load",
        )
    if expectation.is_scalar_broadcast_elementwise:
        require_contains(
            text,
            "tcrv_rvv.splat",
            "materialized selected-body MLIR RHS scalar splat",
        )
        require_contains(
            text,
            'role = "rhs-scalar-value"',
            "materialized selected-body MLIR RHS scalar ABI role",
        )
    if expectation.is_runtime_scalar_splat_store:
        require_contains(
            text,
            "tcrv_rvv.splat",
            "materialized selected-body MLIR runtime scalar splat",
        )
        require_contains(
            text,
            "tcrv_rvv.store",
            "materialized selected-body MLIR runtime scalar splat store",
        )
        require_contains(
            text,
            'role = "rhs-scalar-value"',
            "materialized selected-body MLIR runtime scalar ABI role",
        )
        require_not_contains(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR runtime scalar splat-store",
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
    if expectation.is_strided_load_unit_store:
        require_contains(
            text,
            'role = "source-input-buffer"',
            "materialized selected-body MLIR source buffer ABI role",
        )
        require_contains(
            text,
            "tcrv_rvv.strided_load",
            "materialized selected-body MLIR source strided load",
        )
        require_contains(
            text,
            "tcrv_rvv.move",
            "materialized selected-body MLIR strided load movement op",
        )
        require_contains(
            text,
            "tcrv_rvv.store",
            "materialized selected-body MLIR unit-stride store",
        )
        require_contains(
            text,
            'role = "source-byte-stride"',
            "materialized selected-body MLIR source byte-stride ABI role",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.strided_store",
            "materialized selected-body MLIR strided load unit store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR strided load unit store",
        )
    if expectation.is_unit_load_strided_store:
        require_contains(
            text,
            "tcrv_rvv.load",
            "materialized selected-body MLIR unit-stride source load",
        )
        require_contains(
            text,
            "tcrv_rvv.move",
            "materialized selected-body MLIR unit load movement op",
        )
        require_contains(
            text,
            "tcrv_rvv.strided_store",
            "materialized selected-body MLIR destination strided store",
        )
        require_contains(
            text,
            'role = "destination-byte-stride"',
            "materialized selected-body MLIR destination byte-stride ABI role",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.strided_load",
            "materialized selected-body MLIR unit load strided store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR unit load strided store",
        )
    if expectation.is_indexed_gather_unit_store:
        require_contains(
            text,
            "tcrv_rvv.index_load",
            "materialized selected-body MLIR index vector load",
        )
        require_contains(
            text,
            "tcrv_rvv.indexed_load",
            "materialized selected-body MLIR indexed data load",
        )
        require_contains(
            text,
            "tcrv_rvv.move",
            "materialized selected-body MLIR indexed gather movement op",
        )
        require_contains(
            text,
            "tcrv_rvv.store",
            "materialized selected-body MLIR unit-stride store",
        )
        require_contains(
            text,
            'role = "index-input-buffer"',
            "materialized selected-body MLIR index ABI role",
        )
        require_contains(
            text,
            'offset_unit = "element"',
            "materialized selected-body MLIR indexed gather offset unit",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR indexed gather unit store",
        )
    if expectation.is_indexed_scatter_unit_load:
        require_contains(
            text,
            "tcrv_rvv.load",
            "materialized selected-body MLIR unit-stride source load",
        )
        require_contains(
            text,
            "tcrv_rvv.index_load",
            "materialized selected-body MLIR index vector load",
        )
        require_contains(
            text,
            "tcrv_rvv.move",
            "materialized selected-body MLIR indexed scatter movement op",
        )
        require_contains(
            text,
            "tcrv_rvv.indexed_store",
            "materialized selected-body MLIR indexed destination store",
        )
        require_contains(
            text,
            'role = "index-input-buffer"',
            "materialized selected-body MLIR index ABI role",
        )
        require_contains(
            text,
            'index_uniqueness = "unique"',
            "materialized selected-body MLIR indexed scatter uniqueness",
        )
        require_contains(
            text,
            'offset_unit = "element"',
            "materialized selected-body MLIR indexed scatter offset unit",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.indexed_load",
            "materialized selected-body MLIR indexed scatter unit load",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR indexed scatter unit load",
        )
    if expectation.is_masked_unit_load_store:
        require_contains(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR mask load",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_load",
            "materialized selected-body MLIR masked load op",
        )
        require_contains(
            text,
            'role = "mask-input-buffer"',
            "materialized selected-body MLIR mask ABI role",
        )
        require_contains(
            text,
            'mask_role = "predicate-mask-input-buffer"',
            "materialized selected-body MLIR mask role",
        )
        require_contains(
            text,
            'mask_memory_form = "unit-stride-mask-load"',
            "materialized selected-body MLIR mask memory form",
        )
        require_contains(
            text,
            'memory_form = "masked-unit-load"',
            "materialized selected-body MLIR masked load memory form",
        )
        require_contains(
            text,
            'inactive_lane_policy = "preserve-passthrough-on-false-lanes"',
            "materialized selected-body MLIR masked load inactive lane policy",
        )
        require_contains(
            text,
            "tcrv_rvv.store",
            "materialized selected-body MLIR unit-stride store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR masked memory movement",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_move",
            "materialized selected-body MLIR masked memory movement",
        )
    if expectation.is_masked_unit_store:
        require_contains(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR masked-store mask load",
        )
        require_contains(
            text,
            'role = "mask-input-buffer"',
            "materialized selected-body MLIR masked-store mask ABI role",
        )
        require_contains(
            text,
            'mask_role = "predicate-mask-input-buffer"',
            "materialized selected-body MLIR masked-store mask role",
        )
        require_contains(
            text,
            'mask_memory_form = "unit-stride-mask-load"',
            "materialized selected-body MLIR masked-store mask memory form",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_store",
            "materialized selected-body MLIR masked-store op",
        )
        require_contains(
            text,
            'memory_form = "masked-unit-store"',
            "materialized selected-body MLIR masked-store memory form",
        )
        require_contains(
            text,
            'inactive_lane_policy = "preserve-output-on-false-lanes"',
            "materialized selected-body MLIR masked-store inactive policy",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_move",
            "materialized selected-body MLIR masked-store route",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.store",
            "materialized selected-body MLIR masked-store route",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR masked-store route",
        )
    if expectation.is_computed_masked_unit_load_store:
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR computed mask compare",
        )
        require_contains(
            text,
            'kind = "slt"',
            "materialized selected-body MLIR computed mask predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_load",
            "materialized selected-body MLIR computed mask masked load op",
        )
        require_contains(
            text,
            'role = "source-input-buffer"',
            "materialized selected-body MLIR active source ABI role",
        )
        require_contains(
            text,
            'memory_form = "masked-unit-load"',
            "materialized selected-body MLIR computed mask masked load memory form",
        )
        require_contains(
            text,
            'inactive_lane_policy = "preserve-passthrough-on-false-lanes"',
            "materialized selected-body MLIR computed mask inactive lane policy",
        )
        require_contains(
            text,
            "tcrv_rvv.store",
            "materialized selected-body MLIR computed mask unit-stride store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR computed mask memory movement",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR computed mask memory movement",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_move",
            "materialized selected-body MLIR computed mask memory movement",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.indexed_store",
            "materialized selected-body MLIR computed mask memory movement",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.indexed_store",
            "materialized selected-body MLIR masked memory movement",
        )
    if expectation.is_computed_mask_select:
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR computed-mask select compare",
        )
        require_contains(
            text,
            f'kind = "{expectation.compare_predicate_kind}"',
            "materialized selected-body MLIR computed-mask select predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.select",
            "materialized selected-body MLIR computed-mask select op",
        )
        require_contains(
            text,
            'role = "true-value-input-buffer"',
            "materialized selected-body MLIR true-value ABI role",
        )
        require_contains(
            text,
            'role = "false-value-input-buffer"',
            "materialized selected-body MLIR false-value ABI role",
        )
        require_contains(
            text,
            "tcrv_rvv.store",
            "materialized selected-body MLIR computed-mask select store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR computed-mask select",
        )
    if expectation.is_runtime_scalar_compare_select:
        require_contains(
            text,
            "tcrv_rvv.splat",
            "materialized selected-body MLIR runtime scalar threshold splat",
        )
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR runtime scalar compare",
        )
        require_contains(
            text,
            f'kind = "{expectation.compare_predicate_kind}"',
            "materialized selected-body MLIR runtime scalar compare predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.select",
            "materialized selected-body MLIR runtime scalar select op",
        )
        require_contains(
            text,
            'role = "rhs-scalar-value"',
            "materialized selected-body MLIR runtime scalar threshold ABI role",
        )
        require_contains(
            text,
            'role = "true-value-input-buffer"',
            "materialized selected-body MLIR runtime scalar true-value ABI role",
        )
        require_contains(
            text,
            'role = "false-value-input-buffer"',
            "materialized selected-body MLIR runtime scalar false-value ABI role",
        )
        require_contains(
            text,
            "tcrv_rvv.store",
            "materialized selected-body MLIR runtime scalar compare/select store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR runtime scalar compare/select",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR runtime scalar compare/select",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_move",
            "materialized selected-body MLIR computed-mask select",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR computed-mask select",
        )
    if expectation.is_runtime_scalar_dual_compare_mask_and_select:
        require_contains(
            text,
            'role = "rhs-scalar-value"',
            "materialized selected-body MLIR primary runtime scalar threshold role",
        )
        require_contains(
            text,
            'role = "rhs-secondary-scalar-value"',
            "materialized selected-body MLIR secondary runtime scalar threshold role",
        )
        require_contains(
            text,
            "tcrv_rvv.splat",
            "materialized selected-body MLIR dual runtime scalar threshold splats",
        )
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR dual runtime scalar compare producers",
        )
        require_contains(
            text,
            "tcrv_rvv.mask_and",
            "materialized selected-body MLIR composed mask-and op",
        )
        require_contains(
            text,
            'kind = "and"',
            "materialized selected-body MLIR mask-and kind",
        )
        require_contains(
            text,
            "tcrv_rvv.select",
            "materialized selected-body MLIR mask-and select op",
        )
        require_contains(
            text,
            'role = "true-value-input-buffer"',
            "materialized selected-body MLIR dual compare true-value ABI role",
        )
        require_contains(
            text,
            'role = "false-value-input-buffer"',
            "materialized selected-body MLIR dual compare false-value ABI role",
        )
        require_contains(
            text,
            "tcrv_rvv.store",
            "materialized selected-body MLIR dual compare select store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR dual compare mask composition",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_store",
            "materialized selected-body MLIR dual compare mask composition",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_load",
            "materialized selected-body MLIR dual compare mask composition",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_macc",
            "materialized selected-body MLIR dual compare mask composition",
        )
    if expectation.is_runtime_scalar_computed_mask_store:
        require_contains(
            text,
            "tcrv_rvv.splat",
            "materialized selected-body MLIR runtime scalar masked-store splat",
        )
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR runtime scalar masked-store compare",
        )
        require_contains(
            text,
            f'kind = "{expectation.compare_predicate_kind}"',
            "materialized selected-body MLIR runtime scalar masked-store predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_store",
            "materialized selected-body MLIR runtime scalar computed-mask store op",
        )
        require_contains(
            text,
            'memory_form = "masked-unit-store"',
            "materialized selected-body MLIR runtime scalar computed-mask store leaf",
        )
        require_contains(
            text,
            'inactive_lane_policy = "preserve-output-on-false-lanes"',
            "materialized selected-body MLIR runtime scalar masked-store inactive policy",
        )
        require_contains(
            text,
            'role = "rhs-scalar-value"',
            "materialized selected-body MLIR runtime scalar masked-store threshold ABI role",
        )
        require_contains(
            text,
            'role = "source-input-buffer"',
            "materialized selected-body MLIR runtime scalar masked-store payload ABI role",
        )
        require_contains(
            text,
            'role = "output-buffer"',
            "materialized selected-body MLIR runtime scalar masked-store destination ABI role",
        )
        require_not_contains(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR runtime scalar computed-mask store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.select",
            "materialized selected-body MLIR runtime scalar computed-mask store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.store",
            "materialized selected-body MLIR runtime scalar computed-mask store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR runtime scalar computed-mask store",
        )
    if expectation.is_runtime_scalar_computed_mask_load_store:
        require_contains(
            text,
            "tcrv_rvv.splat",
            "materialized selected-body MLIR runtime scalar masked-load splat",
        )
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR runtime scalar masked-load compare",
        )
        require_contains(
            text,
            f'kind = "{expectation.compare_predicate_kind}"',
            "materialized selected-body MLIR runtime scalar masked-load predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_load",
            "materialized selected-body MLIR runtime scalar computed-mask load op",
        )
        require_contains(
            text,
            'memory_form = "masked-unit-load"',
            "materialized selected-body MLIR runtime scalar computed-mask load leaf",
        )
        require_contains(
            text,
            'inactive_lane_policy = "preserve-passthrough-on-false-lanes"',
            "materialized selected-body MLIR runtime scalar masked-load inactive policy",
        )
        require_contains(
            text,
            'role = "rhs-scalar-value"',
            "materialized selected-body MLIR runtime scalar masked-load threshold ABI role",
        )
        require_contains(
            text,
            'role = "source-input-buffer"',
            "materialized selected-body MLIR runtime scalar masked-load source ABI role",
        )
        require_contains(
            text,
            'role = "output-buffer"',
            "materialized selected-body MLIR runtime scalar masked-load destination ABI role",
        )
        require_not_contains(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR runtime scalar computed-mask load",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.select",
            "materialized selected-body MLIR runtime scalar computed-mask load",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_store",
            "materialized selected-body MLIR runtime scalar computed-mask load",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR runtime scalar computed-mask load",
        )
    if expectation.is_computed_mask_standalone_reduce:
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR computed-mask standalone reduction compare",
        )
        require_contains(
            text,
            f'kind = "{expectation.compare_predicate_kind}"',
            "materialized selected-body MLIR computed-mask standalone reduction predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_standalone_reduce",
            "materialized selected-body MLIR masked standalone reduction op",
        )
        require_contains(
            text,
            f'kind = "{expectation.computed_mask_standalone_reduction_kind}"',
            "materialized selected-body MLIR computed-mask standalone reduction kind",
        )
        require_contains(
            text,
            'role = "source-input-buffer"',
            "materialized selected-body MLIR computed-mask standalone source role",
        )
        require_contains(
            text,
            'role = "accumulator-input-buffer"',
            "materialized selected-body MLIR computed-mask standalone seed role",
        )
        require_contains(
            text,
            'role = "output-buffer"',
            "materialized selected-body MLIR computed-mask standalone scalar output role",
        )
        require_contains(
            text,
            f'mask_role = "{COMPUTED_MASK_MEMORY_MASK_ROLE}"',
            "materialized selected-body MLIR computed-mask standalone mask role",
        )
        require_contains(
            text,
            f'mask_source = "{COMPUTED_MASK_MEMORY_MASK_SOURCE}"',
            "materialized selected-body MLIR computed-mask standalone mask source",
        )
        require_contains(
            text,
            f'mask_memory_form = "{COMPUTED_MASK_MEMORY_MASK_FORM}"',
            "materialized selected-body MLIR computed-mask standalone mask memory form",
        )
        require_contains(
            text,
            f'accumulator_layout = "{STANDALONE_REDUCE_ACCUMULATOR_LAYOUT}"',
            "materialized selected-body MLIR computed-mask standalone accumulator layout",
        )
        require_contains(
            text,
            f'result_layout = "{STANDALONE_REDUCE_RESULT_LAYOUT}"',
            "materialized selected-body MLIR computed-mask standalone result layout",
        )
        require_contains(
            text,
            "tcrv_rvv.store",
            "materialized selected-body MLIR computed-mask standalone store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR computed-mask standalone reduction",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.standalone_reduce",
            "materialized selected-body MLIR computed-mask standalone reduction",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR computed-mask standalone reduction",
        )
    if expectation.is_runtime_scalar_computed_mask_standalone_reduce:
        require_contains(
            text,
            'role = "rhs-scalar-value"',
            "materialized selected-body MLIR runtime scalar standalone reduction threshold ABI role",
        )
        require_contains(
            text,
            "tcrv_rvv.splat",
            "materialized selected-body MLIR runtime scalar standalone reduction threshold splat",
        )
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR runtime scalar standalone reduction compare",
        )
        require_contains(
            text,
            f'kind = "{expectation.compare_predicate_kind}"',
            "materialized selected-body MLIR runtime scalar standalone reduction predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_standalone_reduce",
            "materialized selected-body MLIR runtime scalar masked standalone reduction op",
        )
        require_contains(
            text,
            'role = "source-input-buffer"',
            "materialized selected-body MLIR runtime scalar standalone source role",
        )
        require_contains(
            text,
            'role = "accumulator-input-buffer"',
            "materialized selected-body MLIR runtime scalar standalone seed role",
        )
        require_contains(
            text,
            'role = "output-buffer"',
            "materialized selected-body MLIR runtime scalar standalone scalar output role",
        )
        require_contains(
            text,
            f'mask_source = "{COMPUTED_MASK_MEMORY_MASK_SOURCE}"',
            "materialized selected-body MLIR runtime scalar standalone mask source",
        )
        require_contains(
            text,
            f'accumulator_layout = "{STANDALONE_REDUCE_ACCUMULATOR_LAYOUT}"',
            "materialized selected-body MLIR runtime scalar standalone accumulator layout",
        )
        require_contains(
            text,
            f'result_layout = "{STANDALONE_REDUCE_RESULT_LAYOUT}"',
            "materialized selected-body MLIR runtime scalar standalone result layout",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.standalone_reduce",
            "materialized selected-body MLIR runtime scalar standalone reduction",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.select",
            "materialized selected-body MLIR runtime scalar standalone reduction",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_store",
            "materialized selected-body MLIR runtime scalar standalone reduction",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_macc",
            "materialized selected-body MLIR runtime scalar standalone reduction",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR runtime scalar standalone reduction",
        )
    if expectation.is_computed_masked_strided_store:
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR computed-mask strided-store compare",
        )
        require_contains(
            text,
            'kind = "slt"',
            "materialized selected-body MLIR computed-mask strided-store predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_strided_store",
            "materialized selected-body MLIR computed-mask masked strided-store op",
        )
        require_contains(
            text,
            'memory_form = "masked-strided-store"',
            "materialized selected-body MLIR computed-mask destination memory form",
        )
        require_contains(
            text,
            'inactive_lane_policy = "preserve-output-on-false-lanes"',
            "materialized selected-body MLIR computed-mask inactive lane policy",
        )
        require_contains(
            text,
            'role = "destination-byte-stride"',
            "materialized selected-body MLIR destination stride ABI role",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_move",
            "materialized selected-body MLIR computed-mask strided-store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.strided_load",
            "materialized selected-body MLIR computed-mask strided-store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.strided_store",
            "materialized selected-body MLIR computed-mask strided-store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR computed-mask strided-store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR computed-mask strided-store",
        )
    if expectation.is_computed_masked_strided_load_unit_store:
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR computed-mask strided-load compare",
        )
        require_contains(
            text,
            'kind = "slt"',
            "materialized selected-body MLIR computed-mask strided-load predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_strided_load",
            "materialized selected-body MLIR computed-mask masked strided-load op",
        )
        require_contains(
            text,
            'memory_form = "masked-strided-load"',
            "materialized selected-body MLIR computed-mask source memory form",
        )
        require_contains(
            text,
            'inactive_lane_policy = "preserve-passthrough-on-false-lanes"',
            "materialized selected-body MLIR computed-mask inactive lane policy",
        )
        require_contains(
            text,
            'role = "source-byte-stride"',
            "materialized selected-body MLIR source stride ABI role",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_move",
            "materialized selected-body MLIR computed-mask strided-load",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.strided_load",
            "materialized selected-body MLIR computed-mask strided-load",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.strided_store",
            "materialized selected-body MLIR computed-mask strided-load",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR computed-mask strided-load",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR computed-mask strided-load",
        )
    if expectation.is_computed_masked_indexed_gather_load_unit_store:
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR computed-mask indexed-gather compare",
        )
        require_contains(
            text,
            'kind = "slt"',
            "materialized selected-body MLIR computed-mask indexed-gather predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.index_load",
            "materialized selected-body MLIR computed-mask indexed-gather index load",
        )
        require_contains(
            text,
            'role = "index-input-buffer"',
            "materialized selected-body MLIR computed-mask indexed-gather index ABI role",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_indexed_load",
            "materialized selected-body MLIR computed-mask masked indexed-load op",
        )
        require_contains(
            text,
            'memory_form = "masked-indexed-load"',
            "materialized selected-body MLIR computed-mask indexed source memory form",
        )
        require_contains(
            text,
            'offset_unit = "element"',
            "materialized selected-body MLIR computed-mask indexed offset unit",
        )
        require_contains(
            text,
            'index_eew = 32 : i64',
            "materialized selected-body MLIR computed-mask indexed EEW",
        )
        require_contains(
            text,
            'inactive_lane_policy = "preserve-passthrough-on-false-lanes"',
            "materialized selected-body MLIR computed-mask indexed inactive lane policy",
        )
        require_contains(
            text,
            "tcrv_rvv.store",
            "materialized selected-body MLIR computed-mask indexed unit-stride store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.indexed_load",
            "materialized selected-body MLIR computed-mask indexed-gather",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.strided_load",
            "materialized selected-body MLIR computed-mask indexed-gather",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR computed-mask indexed-gather",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_move",
            "materialized selected-body MLIR computed-mask indexed-gather",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR computed-mask indexed-gather",
        )
    if expectation.is_computed_masked_indexed_scatter_store_unit_load:
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR computed-mask indexed-scatter compare",
        )
        require_contains(
            text,
            'kind = "slt"',
            "materialized selected-body MLIR computed-mask indexed-scatter predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.index_load",
            "materialized selected-body MLIR computed-mask indexed-scatter index load",
        )
        require_contains(
            text,
            'role = "index-input-buffer"',
            "materialized selected-body MLIR computed-mask indexed-scatter index ABI role",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_indexed_store",
            "materialized selected-body MLIR computed-mask masked indexed-store op",
        )
        require_contains(
            text,
            'memory_form = "masked-indexed-store"',
            "materialized selected-body MLIR computed-mask indexed destination memory form",
        )
        require_contains(
            text,
            'offset_unit = "element"',
            "materialized selected-body MLIR computed-mask indexed scatter offset unit",
        )
        require_contains(
            text,
            'index_eew = 32 : i64',
            "materialized selected-body MLIR computed-mask indexed scatter EEW",
        )
        require_contains(
            text,
            'index_uniqueness = "unique"',
            "materialized selected-body MLIR computed-mask indexed scatter uniqueness",
        )
        require_contains(
            text,
            'inactive_lane_policy = "preserve-output-on-false-lanes"',
            "materialized selected-body MLIR computed-mask indexed scatter inactive lane policy",
        )
        require_contains(
            text,
            "tcrv_rvv.load",
            "materialized selected-body MLIR computed-mask indexed scatter unit source load",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_indexed_load",
            "materialized selected-body MLIR computed-mask indexed-scatter",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.indexed_store",
            "materialized selected-body MLIR computed-mask indexed-scatter",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.strided_store",
            "materialized selected-body MLIR computed-mask indexed-scatter",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR computed-mask indexed-scatter",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_move",
            "materialized selected-body MLIR computed-mask indexed-scatter",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR computed-mask indexed-scatter",
        )
    if expectation.is_computed_masked_segment2_load_unit_store:
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR computed-mask segment2 compare",
        )
        require_contains(
            text,
            'kind = "slt"',
            "materialized selected-body MLIR computed-mask segment2 predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_segment2_load",
            "materialized selected-body MLIR computed-mask masked segment2 load op",
        )
        require_contains(
            text,
            'source_memory_form = "segment2-interleaved-unit-stride-load"',
            "materialized selected-body MLIR computed-mask segment2 source memory form",
        )
        require_contains(
            text,
            'inactive_lane_policy = "preserve-passthrough-on-false-lanes"',
            "materialized selected-body MLIR computed-mask segment2 inactive lane policy",
        )
        require_contains(
            text,
            'field0_role = "segment-field0-output-buffer"',
            "materialized selected-body MLIR computed-mask segment2 field0 role",
        )
        require_contains(
            text,
            'field1_role = "segment-field1-output-buffer"',
            "materialized selected-body MLIR computed-mask segment2 field1 role",
        )
        require_contains(
            text,
            'role = "segment-field0-output-buffer"',
            "materialized selected-body MLIR computed-mask segment2 field0 ABI role",
        )
        require_contains(
            text,
            'role = "segment-field1-output-buffer"',
            "materialized selected-body MLIR computed-mask segment2 field1 ABI role",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.segment2_load",
            "materialized selected-body MLIR computed-mask segment2",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR computed-mask segment2",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_move",
            "materialized selected-body MLIR computed-mask segment2",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.strided_load",
            "materialized selected-body MLIR computed-mask segment2",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_indexed_load",
            "materialized selected-body MLIR computed-mask segment2",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR computed-mask segment2",
        )
    if expectation.is_computed_masked_segment2_store_unit_load:
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR computed-mask segment2-store compare",
        )
        require_contains(
            text,
            'kind = "slt"',
            "materialized selected-body MLIR computed-mask segment2-store predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_segment2_store",
            "materialized selected-body MLIR computed-mask masked segment2 store op",
        )
        require_contains(
            text,
            'destination_memory_form = "segment2-interleaved-unit-stride-store"',
            "materialized selected-body MLIR computed-mask segment2-store destination memory form",
        )
        require_contains(
            text,
            'inactive_lane_policy = "preserve-output-on-false-lanes"',
            "materialized selected-body MLIR computed-mask segment2-store inactive lane policy",
        )
        require_contains(
            text,
            'field0_role = "segment-field0-input-buffer"',
            "materialized selected-body MLIR computed-mask segment2-store field0 role",
        )
        require_contains(
            text,
            'field1_role = "segment-field1-input-buffer"',
            "materialized selected-body MLIR computed-mask segment2-store field1 role",
        )
        require_contains(
            text,
            'role = "segment-interleaved-output-buffer"',
            "materialized selected-body MLIR computed-mask segment2-store destination ABI role",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.segment2_store",
            "materialized selected-body MLIR computed-mask segment2-store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.mask_load",
            "materialized selected-body MLIR computed-mask segment2-store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_move",
            "materialized selected-body MLIR computed-mask segment2-store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.strided_load",
            "materialized selected-body MLIR computed-mask segment2-store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_indexed_store",
            "materialized selected-body MLIR computed-mask segment2-store",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR computed-mask segment2-store",
        )
    if expectation.is_segment2_deinterleave_unit_store:
        require_contains(
            text,
            "tcrv_rvv.segment2_load",
            "materialized selected-body MLIR segment2 load",
        )
        require_contains(
            text,
            'segment_count = 2',
            "materialized selected-body MLIR segment count",
        )
        require_contains(
            text,
            'source_memory_form = "segment2-interleaved-unit-stride-load"',
            "materialized selected-body MLIR segment source memory form",
        )
        require_contains(
            text,
            'field0_role = "segment-field0-output-buffer"',
            "materialized selected-body MLIR field0 role",
        )
        require_contains(
            text,
            'field1_role = "segment-field1-output-buffer"',
            "materialized selected-body MLIR field1 role",
        )
        require_contains(
            text,
            'role = "segment-field0-output-buffer"',
            "materialized selected-body MLIR field0 ABI role",
        )
        require_contains(
            text,
            'role = "segment-field1-output-buffer"',
            "materialized selected-body MLIR field1 ABI role",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR segment2 memory movement",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.indexed_store",
            "materialized selected-body MLIR segment2 memory movement",
        )
    if expectation.is_segment2_interleave_unit_load:
        require_contains(
            text,
            "tcrv_rvv.segment2_store",
            "materialized selected-body MLIR segment2 store",
        )
        require_contains(
            text,
            'segment_count = 2',
            "materialized selected-body MLIR segment count",
        )
        require_contains(
            text,
            'destination_memory_form = "segment2-interleaved-unit-stride-store"',
            "materialized selected-body MLIR segment destination memory form",
        )
        require_contains(
            text,
            'field0_role = "segment-field0-input-buffer"',
            "materialized selected-body MLIR field0 role",
        )
        require_contains(
            text,
            'field1_role = "segment-field1-input-buffer"',
            "materialized selected-body MLIR field1 role",
        )
        require_contains(
            text,
            'role = "segment-field0-input-buffer"',
            "materialized selected-body MLIR field0 source ABI role",
        )
        require_contains(
            text,
            'role = "segment-field1-input-buffer"',
            "materialized selected-body MLIR field1 source ABI role",
        )
        require_contains(
            text,
            'role = "segment-interleaved-output-buffer"',
            "materialized selected-body MLIR interleaved destination ABI role",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.segment2_load",
            "materialized selected-body MLIR segment2 interleave movement",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR segment2 interleave movement",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.indexed_store",
            "materialized selected-body MLIR segment2 interleave movement",
        )
    if expectation.is_cmp_select:
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR compare mask producer",
        )
    if expectation.is_reduce_add:
        require_contains(
            text,
            'accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk"',
            "materialized selected-body MLIR reduce accumulator layout",
        )
        require_contains(
            text,
            'result_layout = "store-reduction-lane0-to-output-chunk-base"',
            "materialized selected-body MLIR reduce result layout",
        )
    if expectation.is_standalone_reduce:
        require_contains(
            text,
            "tcrv_rvv.standalone_reduce",
            "materialized selected-body MLIR standalone reduction op",
        )
        require_contains(
            text,
            f'kind = "{expectation.standalone_reduction_kind}"',
            "materialized selected-body MLIR standalone reduction kind",
        )
        require_contains(
            text,
            f'accumulator_layout = "{STANDALONE_REDUCE_ACCUMULATOR_LAYOUT}"',
            "materialized selected-body MLIR standalone reduction accumulator layout",
        )
        require_contains(
            text,
            f'result_layout = "{STANDALONE_REDUCE_RESULT_LAYOUT}"',
            "materialized selected-body MLIR standalone reduction result layout",
        )
        require_contains(
            text,
            'role = "accumulator-input-buffer"',
            "materialized selected-body MLIR standalone seed ABI role",
        )
        require_contains(
            text,
            'role = "output-buffer"',
            "materialized selected-body MLIR standalone scalar output role",
        )
        require_not_contains(
            text,
            "rhs-vector-seed-lane0-per-vl-chunk",
            "materialized selected-body MLIR standalone reduction",
        )
    if expectation.is_macc_add:
        require_contains(
            text,
            'role = "accumulator-input-buffer"',
            "materialized selected-body MLIR macc accumulator ABI role",
        )
        require_contains(
            text,
            f'accumulator_layout = "{MACC_ADD_ACCUMULATOR_LAYOUT}"',
            "materialized selected-body MLIR macc accumulator layout",
        )
        require_contains(
            text,
            'result_layout = "store-multiply-accumulate-result-to-output-buffer"',
            "materialized selected-body MLIR macc result layout",
        )
    if expectation.is_computed_masked_macc_add:
        require_contains(
            text,
            'role = "dot-lhs-input-buffer"',
            "materialized selected-body MLIR computed-mask macc lhs payload ABI role",
        )
        require_contains(
            text,
            'role = "dot-rhs-input-buffer"',
            "materialized selected-body MLIR computed-mask macc rhs payload ABI role",
        )
        require_contains(
            text,
            'role = "accumulator-input-buffer"',
            "materialized selected-body MLIR computed-mask macc accumulator ABI role",
        )
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR computed-mask macc compare producer",
        )
        require_contains(
            text,
            'kind = "slt"',
            "materialized selected-body MLIR computed-mask macc predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_macc",
            "materialized selected-body MLIR computed-mask macc op",
        )
        require_contains(
            text,
            f'mask_source = "{COMPUTED_MASK_MEMORY_MASK_SOURCE}"',
            "materialized selected-body MLIR computed-mask macc mask source",
        )
        require_contains(
            text,
            f'mask_memory_form = "{COMPUTED_MASK_MEMORY_MASK_FORM}"',
            "materialized selected-body MLIR computed-mask macc mask memory form",
        )
        require_contains(
            text,
            f'accumulator_layout = "{MACC_ADD_ACCUMULATOR_LAYOUT}"',
            "materialized selected-body MLIR computed-mask macc accumulator layout",
        )
        require_contains(
            text,
            f'result_layout = "{MACC_ADD_RESULT_LAYOUT}"',
            "materialized selected-body MLIR computed-mask macc result layout",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.macc",
            "materialized selected-body MLIR computed-mask macc route",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR computed-mask macc route",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_move",
            "materialized selected-body MLIR computed-mask macc route",
        )
    if expectation.is_runtime_scalar_computed_masked_macc_add:
        require_contains(
            text,
            'role = "rhs-scalar-value"',
            "materialized selected-body MLIR runtime scalar macc threshold ABI role",
        )
        require_contains(
            text,
            "tcrv_rvv.splat",
            "materialized selected-body MLIR runtime scalar macc threshold splat",
        )
        require_contains(
            text,
            'role = "dot-lhs-input-buffer"',
            "materialized selected-body MLIR runtime scalar macc lhs payload ABI role",
        )
        require_contains(
            text,
            'role = "dot-rhs-input-buffer"',
            "materialized selected-body MLIR runtime scalar macc rhs payload ABI role",
        )
        require_contains(
            text,
            'role = "accumulator-input-buffer"',
            "materialized selected-body MLIR runtime scalar macc accumulator ABI role",
        )
        require_contains(
            text,
            "tcrv_rvv.compare",
            "materialized selected-body MLIR runtime scalar macc compare producer",
        )
        require_contains(
            text,
            f'kind = "{expectation.compare_predicate_kind}"',
            "materialized selected-body MLIR runtime scalar macc predicate",
        )
        require_contains(
            text,
            "tcrv_rvv.masked_macc",
            "materialized selected-body MLIR runtime scalar macc op",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.macc",
            "materialized selected-body MLIR runtime scalar macc route",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.binary",
            "materialized selected-body MLIR runtime scalar macc route",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.select",
            "materialized selected-body MLIR runtime scalar macc route",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_store",
            "materialized selected-body MLIR runtime scalar macc route",
        )
        require_no_op_invocation(
            text,
            "tcrv_rvv.masked_load",
            "materialized selected-body MLIR runtime scalar macc route",
        )
    if expectation.is_widening_macc_add:
        require_contains(
            text,
            f'accumulator_layout = "{WIDENING_MACC_ACCUMULATOR_LAYOUT}"',
            "materialized selected-body MLIR widening macc accumulator layout",
        )
        require_contains(
            text,
            f'result_layout = "{WIDENING_MACC_RESULT_LAYOUT}"',
            "materialized selected-body MLIR widening macc result layout",
        )
    if expectation.is_widening_dot_reduce_add:
        require_contains(
            text,
            f'accumulator_layout = "{WIDENING_DOT_ACCUMULATOR_LAYOUT}"',
            "materialized selected-body MLIR widening dot accumulator layout",
        )
        require_contains(
            text,
            f'result_layout = "{WIDENING_DOT_RESULT_LAYOUT}"',
            "materialized selected-body MLIR widening dot result layout",
        )
    if expectation.is_computed_masked_widening_dot_reduce_add:
        require_contains(
            text,
            f'accumulator_layout = "{WIDENING_DOT_ACCUMULATOR_LAYOUT}"',
            "materialized selected-body MLIR masked widening dot accumulator layout",
        )
        require_contains(
            text,
            f'result_layout = "{WIDENING_DOT_RESULT_LAYOUT}"',
            "materialized selected-body MLIR masked widening dot result layout",
        )
    if expectation.is_computed_masked_strided_input_widening_dot_reduce_add:
        require_contains(
            text,
            f'accumulator_layout = "{WIDENING_DOT_ACCUMULATOR_LAYOUT}"',
            "materialized selected-body MLIR masked strided dot accumulator layout",
        )
        require_contains(
            text,
            f'result_layout = "{WIDENING_DOT_RESULT_LAYOUT}"',
            "materialized selected-body MLIR masked strided dot result layout",
        )
    if expectation.is_pre_realized:
        require_not_contains(
            text,
            "tcrv_rvv.typed_binary_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_masked_binary_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_compare_select_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_computed_mask_select_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_runtime_scalar_compare_select_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_reduce_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_standalone_reduce_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_computed_mask_standalone_reduce_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_macc_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_computed_mask_macc_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_widening_conversion_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_widening_macc_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_widening_dot_reduce_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_computed_mask_widening_dot_reduce_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_computed_mask_strided_input_widening_dot_reduce_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_strided_memory_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_indexed_gather_memory_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_indexed_scatter_memory_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_masked_memory_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_computed_mask_memory_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_segment2_deinterleave_memory_pre_realized_body",
            "materialized pre-realized selected-body MLIR",
        )
        require_not_contains(
            text,
            "tcrv_rvv.typed_segment2_interleave_memory_pre_realized_body",
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
    header_file_name: str,
    runtime_counts: list[int],
    expectation: OpExpectation,
    rhs_scalar_values: list[int] | None = None,
    stride_bytes_values: list[int] | None = None,
) -> str:
    counts = ", ".join(str(count) for count in runtime_counts)
    scalar_values = list(
        DEFAULT_RHS_SCALAR_VALUES if rhs_scalar_values is None else rhs_scalar_values
    )
    scalar_values_literal = ", ".join(f"(int32_t){value}" for value in scalar_values)
    scalar_values_summary = ",".join(str(value) for value in scalar_values)
    stride_values = list(
        DEFAULT_STRIDED_LOAD_BYTE_STRIDES
        if stride_bytes_values is None
        else stride_bytes_values
    )
    stride_values_literal = ", ".join(str(value) for value in stride_values)
    stride_values_summary = ",".join(str(value) for value in stride_values)
    if expectation.is_computed_masked_segment2_load_unit_store:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t src_alloc_n = alloc_n * 2 + 8;
  size_t out_alloc_n = alloc_n + 8;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src = (int32_t *)malloc(sizeof(int32_t) * src_alloc_n);
  int32_t *src_before = (int32_t *)malloc(sizeof(int32_t) * src_alloc_n);
  int32_t *out0 = (int32_t *)malloc(sizeof(int32_t) * out_alloc_n);
  int32_t *out1 = (int32_t *)malloc(sizeof(int32_t) * out_alloc_n);
  int32_t *old0 = (int32_t *)malloc(sizeof(int32_t) * out_alloc_n);
  int32_t *old1 = (int32_t *)malloc(sizeof(int32_t) * out_alloc_n);
  int status = 0;
  if (!cmp_lhs || !cmp_rhs || !src || !src_before || !out0 || !out1 || !old0 || !old1) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    status = 11;
    goto cleanup;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs[index] = {expectation.lhs_initializer};
    cmp_rhs[index] = {expectation.rhs_initializer};
  }}
  for (size_t index = 0; index < src_alloc_n; ++index) {{
    src[index] = {expectation.source_initializer};
    src_before[index] = src[index];
  }}
  for (size_t index = 0; index < out_alloc_n; ++index) {{
    out0[index] = {expectation.out_initializer};
    out1[index] = (int32_t)({expectation.out_initializer} - 97);
    old0[index] = out0[index];
    old1[index] = out1[index];
  }}

  {expectation.function_name}(cmp_lhs, cmp_rhs, src, out0, out1, n);

  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t inactive_preserved_lanes = 0;
  size_t field_distinguishing_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int active = cmp_lhs[index] < cmp_rhs[index];
    if (active)
      ++active_lanes;
    else
      ++inactive_lanes;
    int32_t expected0 = active ? src[2 * index] : old0[index];
    int32_t expected1 = active ? src[2 * index + 1] : old1[index];
    if (expected0 != expected1)
      ++field_distinguishing_lanes;
    if (out0[index] != expected0 || out1[index] != expected1) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got0=%d expected0=%d got1=%d expected1=%d cmp_lhs=%d cmp_rhs=%d src0=%d src1=%d old0=%d old1=%d\\n",
              n, index, out0[index], expected0, out1[index], expected1,
              cmp_lhs[index], cmp_rhs[index], src[2 * index],
              src[2 * index + 1], old0[index], old1[index]);
      status = 12;
      goto cleanup;
    }}
    if (!active) {{
      if (out0[index] != old0[index] || out1[index] != old1[index]) {{
        fprintf(stderr,
                "{expectation.kind} inactive lane failed to preserve old fields n=%zu index=%zu got0=%d old0=%d got1=%d old1=%d\\n",
                n, index, out0[index], old0[index], out1[index], old1[index]);
        status = 13;
        goto cleanup;
      }}
      ++inactive_preserved_lanes;
    }}
  }}

  for (size_t index = n; index < out_alloc_n; ++index) {{
    if (out0[index] != old0[index] || out1[index] != old1[index]) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu index=%zu got0=%d old0=%d got1=%d old1=%d\\n",
              n, index, out0[index], old0[index], out1[index], old1[index]);
      status = 14;
      goto cleanup;
    }}
  }}
  for (size_t index = 0; index < src_alloc_n; ++index) {{
    if (src[index] != src_before[index]) {{
      fprintf(stderr,
              "{expectation.kind} source buffer mutated n=%zu index=%zu got=%d before=%d\\n",
              n, index, src[index], src_before[index]);
      status = 15;
      goto cleanup;
    }}
  }}
  if (n > 1 && (active_lanes == 0 || inactive_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} compare mask coverage missing n=%zu active_lanes=%zu inactive_lanes=%zu\\n",
            n, active_lanes, inactive_lanes);
    status = 16;
    goto cleanup;
  }}
  if (n > 1 && field_distinguishing_lanes == 0) {{
    fprintf(stderr,
            "{expectation.kind} field-distinguishing coverage missing n=%zu\\n",
            n);
    status = 17;
    goto cleanup;
  }}
  if (inactive_lanes != inactive_preserved_lanes) {{
    fprintf(stderr,
            "{expectation.kind} inactive preservation coverage mismatch n=%zu inactive_lanes=%zu preserved_lanes=%zu\\n",
            n, inactive_lanes, inactive_preserved_lanes);
    status = 18;
    goto cleanup;
  }}

  printf("{expectation.kind} case n=%zu ok computed_mask segment2_load active_lanes=%zu inactive_lanes=%zu inactive_preserved_lanes=%zu field_distinguishing_lanes=%zu source_preserved tail_preserved\\n",
         n, active_lanes, inactive_lanes, inactive_preserved_lanes,
         field_distinguishing_lanes);

cleanup:
  free(cmp_lhs);
  free(cmp_rhs);
  free(src);
  free(src_before);
  free(out0);
  free(out1);
  free(old0);
  free(old1);
  return status;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  for (size_t index = 0; index < sizeof(counts) / sizeof(counts[0]); ++index) {{
    int status = run_case(counts[index]);
    if (status != 0)
      return status;
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)}\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_computed_masked_segment2_store_unit_load:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t dst_alloc_n = alloc_n * 2 + 8;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src0 = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src1 = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src0_before = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src1_before = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  int32_t *old_dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  int status = 0;
  if (!cmp_lhs || !cmp_rhs || !src0 || !src1 || !src0_before ||
      !src1_before || !dst || !old_dst) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    status = 11;
    goto cleanup;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs[index] = {expectation.lhs_initializer};
    cmp_rhs[index] = {expectation.rhs_initializer};
    src0[index] = {expectation.source_initializer};
    src1[index] = (int32_t)({expectation.source_initializer} - 211);
    src0_before[index] = src0[index];
    src1_before[index] = src1[index];
  }}
  for (size_t index = 0; index < dst_alloc_n; ++index) {{
    dst[index] = {expectation.out_initializer};
    old_dst[index] = dst[index];
  }}

  {expectation.function_name}(cmp_lhs, cmp_rhs, src0, src1, dst, n);

  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t inactive_preserved_lanes = 0;
  size_t field_distinguishing_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int active = cmp_lhs[index] < cmp_rhs[index];
    if (active)
      ++active_lanes;
    else
      ++inactive_lanes;
    int32_t expected0 = active ? src0[index] : old_dst[2 * index];
    int32_t expected1 = active ? src1[index] : old_dst[2 * index + 1];
    if (expected0 != expected1)
      ++field_distinguishing_lanes;
    if (dst[2 * index] != expected0 || dst[2 * index + 1] != expected1) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got0=%d expected0=%d got1=%d expected1=%d cmp_lhs=%d cmp_rhs=%d src0=%d src1=%d old0=%d old1=%d\\n",
              n, index, dst[2 * index], expected0, dst[2 * index + 1],
              expected1, cmp_lhs[index], cmp_rhs[index], src0[index],
              src1[index], old_dst[2 * index], old_dst[2 * index + 1]);
      status = 12;
      goto cleanup;
    }}
    if (!active) {{
      if (dst[2 * index] != old_dst[2 * index] ||
          dst[2 * index + 1] != old_dst[2 * index + 1]) {{
        fprintf(stderr,
                "{expectation.kind} inactive lane failed to preserve interleaved destination n=%zu index=%zu got0=%d old0=%d got1=%d old1=%d\\n",
                n, index, dst[2 * index], old_dst[2 * index],
                dst[2 * index + 1], old_dst[2 * index + 1]);
        status = 13;
        goto cleanup;
      }}
      ++inactive_preserved_lanes;
    }}
  }}

  for (size_t index = n * 2; index < dst_alloc_n; ++index) {{
    if (dst[index] != old_dst[index]) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d old=%d\\n",
              n, index, dst[index], old_dst[index]);
      status = 14;
      goto cleanup;
    }}
  }}
  for (size_t index = 0; index < alloc_n; ++index) {{
    if (src0[index] != src0_before[index] || src1[index] != src1_before[index]) {{
      fprintf(stderr,
              "{expectation.kind} source buffer mutated n=%zu index=%zu got0=%d before0=%d got1=%d before1=%d\\n",
              n, index, src0[index], src0_before[index], src1[index],
              src1_before[index]);
      status = 15;
      goto cleanup;
    }}
  }}
  if (n > 1 && (active_lanes == 0 || inactive_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} compare mask coverage missing n=%zu active_lanes=%zu inactive_lanes=%zu\\n",
            n, active_lanes, inactive_lanes);
    status = 16;
    goto cleanup;
  }}
  if (n > 1 && field_distinguishing_lanes == 0) {{
    fprintf(stderr,
            "{expectation.kind} field-distinguishing coverage missing n=%zu\\n",
            n);
    status = 17;
    goto cleanup;
  }}
  if (inactive_lanes != inactive_preserved_lanes) {{
    fprintf(stderr,
            "{expectation.kind} inactive preservation coverage mismatch n=%zu inactive_lanes=%zu preserved_lanes=%zu\\n",
            n, inactive_lanes, inactive_preserved_lanes);
    status = 18;
    goto cleanup;
  }}

  printf("{expectation.kind} case n=%zu ok computed_mask segment2_store active_lanes=%zu inactive_lanes=%zu inactive_preserved_lanes=%zu field_distinguishing_lanes=%zu source_preserved tail_preserved\\n",
         n, active_lanes, inactive_lanes, inactive_preserved_lanes,
         field_distinguishing_lanes);

cleanup:
  free(cmp_lhs);
  free(cmp_rhs);
  free(src0);
  free(src1);
  free(src0_before);
  free(src1_before);
  free(dst);
  free(old_dst);
  return status;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  for (size_t index = 0; index < sizeof(counts) / sizeof(counts[0]); ++index) {{
    int status = run_case(counts[index]);
    if (status != 0)
      return status;
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)}\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_segment2_deinterleave_unit_store:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t src_alloc_n = alloc_n * 2 + 8;
  size_t out_alloc_n = alloc_n + 8;
  int32_t *src = (int32_t *)malloc(sizeof(int32_t) * src_alloc_n);
  int32_t *out0 = (int32_t *)malloc(sizeof(int32_t) * out_alloc_n);
  int32_t *out1 = (int32_t *)malloc(sizeof(int32_t) * out_alloc_n);
  if (!src || !out0 || !out1) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(src);
    free(out0);
    free(out1);
    return 11;
  }}
  for (size_t index = 0; index < src_alloc_n; ++index)
    src[index] = (int32_t)(1700 + (int32_t)(index * 23));
  for (size_t index = 0; index < out_alloc_n; ++index) {{
    out0[index] = {OUT_SENTINEL};
    out1[index] = {OUT_SENTINEL};
  }}

  {expectation.function_name}(src, out0, out1, n);

  size_t field_order_distinguishing_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int32_t expected0 = src[2 * index];
    int32_t expected1 = src[2 * index + 1];
    if (expected0 != expected1)
      ++field_order_distinguishing_lanes;
    if (out0[index] != expected0 || out1[index] != expected1) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got0=%d expected0=%d got1=%d expected1=%d src_even=%d src_odd=%d\\n",
              n, index, out0[index], expected0, out1[index], expected1,
              src[2 * index], src[2 * index + 1]);
      free(src);
      free(out0);
      free(out1);
      return 13;
    }}
  }}
  for (size_t index = n; index < out_alloc_n; ++index) {{
    if (out0[index] != {OUT_SENTINEL} || out1[index] != {OUT_SENTINEL}) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got0=%d got1=%d sentinel=%d\\n",
              n, index, out0[index], out1[index], {OUT_SENTINEL});
      free(src);
      free(out0);
      free(out1);
      return 17;
    }}
  }}
  if (n > 1 && field_order_distinguishing_lanes == 0) {{
    fprintf(stderr,
            "{expectation.kind} field-order coverage missing n=%zu\\n", n);
    free(src);
    free(out0);
    free(out1);
    return 19;
  }}
  printf("{expectation.kind} case n=%zu ok field_order_distinguishing_lanes=%zu tail_preserved\\n",
         n, field_order_distinguishing_lanes);
  free(src);
  free(out0);
  free(out1);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {{
    int rc = run_case(counts[i]);
    if (rc != 0)
      return rc;
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)}\\n");
  return 0;
}}
"""
    if expectation.is_segment2_interleave_unit_load:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t dst_alloc_n = alloc_n * 2 + 8;
  int32_t *src0 = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src1 = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  if (!src0 || !src1 || !dst) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(src0);
    free(src1);
    free(dst);
    return 11;
  }}
  for (size_t index = 0; index < alloc_n; ++index) {{
    src0[index] = (int32_t)(1900 + (int32_t)(index * 29));
    src1[index] = (int32_t)(-2300 - (int32_t)(index * 31));
  }}
  for (size_t index = 0; index < dst_alloc_n; ++index)
    dst[index] = {OUT_SENTINEL};

  {expectation.function_name}(src0, src1, dst, n);

  size_t field_order_distinguishing_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int32_t expected0 = src0[index];
    int32_t expected1 = src1[index];
    if (expected0 != expected1)
      ++field_order_distinguishing_lanes;
    if (dst[2 * index] != expected0 || dst[2 * index + 1] != expected1) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got_even=%d expected_even=%d got_odd=%d expected_odd=%d src0=%d src1=%d\\n",
              n, index, dst[2 * index], expected0,
              dst[2 * index + 1], expected1, src0[index], src1[index]);
      free(src0);
      free(src1);
      free(dst);
      return 13;
    }}
  }}
  for (size_t index = n * 2; index < dst_alloc_n; ++index) {{
    if (dst[index] != {OUT_SENTINEL}) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, dst[index], {OUT_SENTINEL});
      free(src0);
      free(src1);
      free(dst);
      return 17;
    }}
  }}
  if (n > 1 && field_order_distinguishing_lanes == 0) {{
    fprintf(stderr,
            "{expectation.kind} field-order coverage missing n=%zu\\n", n);
    free(src0);
    free(src1);
    free(dst);
    return 19;
  }}
  printf("{expectation.kind} case n=%zu ok field_order_distinguishing_lanes=%zu tail_preserved\\n",
         n, field_order_distinguishing_lanes);
  free(src0);
  free(src1);
  free(dst);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {{
    int rc = run_case(counts[i]);
    if (rc != 0)
      return rc;
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)}\\n");
  return 0;
}}
"""
    if expectation.is_computed_masked_strided_store:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n, size_t dst_stride_bytes) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t dst_alloc_bytes = alloc_n * dst_stride_bytes + 32;
  size_t dst_alloc_n = (dst_alloc_bytes + sizeof(int32_t) - 1) / sizeof(int32_t);
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  int32_t *old_dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  uint8_t *dst_raw = (uint8_t *)dst;
  uint8_t *old_dst_raw = (uint8_t *)old_dst;
  if (!cmp_lhs || !cmp_rhs || !src || !dst || !old_dst) {{
    fprintf(stderr, "allocation failed for n=%zu dst_stride_bytes=%zu\\n",
            n, dst_stride_bytes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(dst);
    free(old_dst);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs[index] = {expectation.lhs_initializer};
    cmp_rhs[index] = {expectation.rhs_initializer};
    src[index] = {expectation.source_initializer};
  }}
  for (size_t index = 0; index < dst_alloc_n; ++index) {{
    dst[index] = {expectation.out_initializer};
    old_dst[index] = dst[index];
  }}

  {expectation.function_name}(cmp_lhs, cmp_rhs, src, dst, n, dst_stride_bytes);

  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t inactive_preserved_lanes = 0;
  size_t skipped_preserved_slots = 0;
  for (size_t index = 0; index < n; ++index) {{
    size_t dst_byte_offset = index * dst_stride_bytes;
    int32_t got = *(int32_t *)(dst_raw + dst_byte_offset);
    int32_t old_slot = *(int32_t *)(old_dst_raw + dst_byte_offset);
    int active = cmp_lhs[index] < cmp_rhs[index];
    if (active)
      ++active_lanes;
    else
      ++inactive_lanes;

    int32_t expected = {expectation.expected_expression};
    if (got != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu dst_byte_offset=%zu got=%d expected=%d cmp_lhs=%d cmp_rhs=%d src=%d old_dst=%d dst_stride_bytes=%zu\\n",
              n, index, dst_byte_offset, got, expected, cmp_lhs[index],
              cmp_rhs[index], src[index], old_slot, dst_stride_bytes);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(dst);
      free(old_dst);
      return 12;
    }}

    if (!active) {{
      if (got != old_slot) {{
        fprintf(stderr,
                "{expectation.kind} inactive byte-strided lane did not preserve old destination n=%zu index=%zu dst_byte_offset=%zu got=%d old_dst=%d dst_stride_bytes=%zu\\n",
                n, index, dst_byte_offset, got, old_slot, dst_stride_bytes);
        free(cmp_lhs);
        free(cmp_rhs);
        free(src);
        free(dst);
        free(old_dst);
        return 13;
      }}
      ++inactive_preserved_lanes;
    }}
  }}

  for (size_t raw = 0; raw < dst_alloc_n; ++raw) {{
    int is_logical_strided_slot = 0;
    for (size_t lane = 0; lane < n; ++lane) {{
      if (raw * sizeof(int32_t) == lane * dst_stride_bytes) {{
        is_logical_strided_slot = 1;
        break;
      }}
    }}
    if (!is_logical_strided_slot && dst[raw] != old_dst[raw]) {{
      fprintf(stderr,
              "{expectation.kind} touched skipped/tail destination slot n=%zu raw_index=%zu got=%d old_dst=%d dst_stride_bytes=%zu\\n",
              n, raw, dst[raw], old_dst[raw], dst_stride_bytes);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(dst);
      free(old_dst);
      return 14;
    }}
    if (!is_logical_strided_slot)
      ++skipped_preserved_slots;
  }}

  if (n > 1 && dst_stride_bytes > sizeof(int32_t) && dst[1] != old_dst[1]) {{
    fprintf(stderr,
            "{expectation.kind} vacuous byte-strided-store check failed n=%zu dst_stride_bytes=%zu dst[1]=%d old_dst[1]=%d\\n",
            n, dst_stride_bytes, dst[1], old_dst[1]);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(dst);
    free(old_dst);
    return 15;
  }}

  if (n > 1 && (active_lanes == 0 || inactive_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} compare mask coverage missing n=%zu active_lanes=%zu inactive_lanes=%zu\\n",
            n, active_lanes, inactive_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(dst);
    free(old_dst);
    return 16;
  }}
  if (inactive_lanes != inactive_preserved_lanes) {{
    fprintf(stderr,
            "{expectation.kind} inactive preservation coverage mismatch n=%zu inactive_lanes=%zu preserved_lanes=%zu\\n",
            n, inactive_lanes, inactive_preserved_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(dst);
    free(old_dst);
    return 17;
  }}

  free(cmp_lhs);
  free(cmp_rhs);
  free(src);
  free(dst);
  free(old_dst);
  printf("{expectation.kind} case n=%zu ok dst_stride_bytes=%zu computed_mask byte_strided_store selected_destination_writes false_lanes_preserved sentinel_preserved tail_preserved compare_active_lanes=%zu compare_inactive_lanes=%zu inactive_preserved_lanes=%zu skipped_preserved_slots=%zu\\n",
         n, dst_stride_bytes, active_lanes, inactive_lanes,
         inactive_preserved_lanes, skipped_preserved_slots);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t stride_bytes_values[] = {{{stride_values_literal}}};
  const size_t stride_count = sizeof(stride_bytes_values) / sizeof(stride_bytes_values[0]);
  for (size_t index = 0; index < count_count; ++index) {{
    for (size_t stride_index = 0; stride_index < stride_count; ++stride_index) {{
      int status = run_case(counts[index], stride_bytes_values[stride_index]);
      if (status != 0)
        return status;
    }}
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} stride_bytes={stride_values_summary}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} stride_bytes={stride_values_summary}\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_computed_masked_strided_load_unit_store:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n, size_t src_stride_bytes) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t src_alloc_bytes = alloc_n * src_stride_bytes + 32;
  size_t src_alloc_n = (src_alloc_bytes + sizeof(int32_t) - 1) / sizeof(int32_t);
  size_t dst_alloc_n = alloc_n + 8;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src = (int32_t *)malloc(sizeof(int32_t) * src_alloc_n);
  int32_t *src_before = (int32_t *)malloc(sizeof(int32_t) * src_alloc_n);
  int32_t *dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  int32_t *old_dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  uint8_t *src_raw = (uint8_t *)src;
  if (!cmp_lhs || !cmp_rhs || !src || !src_before || !dst || !old_dst) {{
    fprintf(stderr, "allocation failed for n=%zu src_stride_bytes=%zu\\n",
            n, src_stride_bytes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(src_before);
    free(dst);
    free(old_dst);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs[index] = {expectation.lhs_initializer};
    cmp_rhs[index] = {expectation.rhs_initializer};
  }}
  for (size_t index = 0; index < src_alloc_n; ++index)
    src[index] = (int32_t)(-21000 - (int32_t)(index * 41));
  for (size_t index = 0; index < alloc_n; ++index) {{
    int32_t *slot = (int32_t *)(src_raw + index * src_stride_bytes);
    *slot = {expectation.source_initializer};
  }}
  for (size_t index = 0; index < src_alloc_n; ++index)
    src_before[index] = src[index];
  for (size_t index = 0; index < dst_alloc_n; ++index) {{
    dst[index] = {expectation.out_initializer};
    old_dst[index] = dst[index];
  }}

  {expectation.function_name}(cmp_lhs, cmp_rhs, src, dst, n, src_stride_bytes);

  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t inactive_preserved_lanes = 0;
  size_t source_gap_preserved_slots = 0;
  for (size_t index = 0; index < n; ++index) {{
    int32_t source_slot = *(const int32_t *)(src_raw + index * src_stride_bytes);
    int active = cmp_lhs[index] < cmp_rhs[index];
    if (active)
      ++active_lanes;
    else
      ++inactive_lanes;
    int32_t expected = {expectation.expected_expression};
    if (dst[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d cmp_lhs=%d cmp_rhs=%d source_slot=%d old_dst=%d src_stride_bytes=%zu\\n",
              n, index, dst[index], expected, cmp_lhs[index], cmp_rhs[index],
              source_slot, old_dst[index], src_stride_bytes);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(src_before);
      free(dst);
      free(old_dst);
      return 12;
    }}
    if (!active) {{
      if (dst[index] != old_dst[index]) {{
        fprintf(stderr,
                "{expectation.kind} inactive lane did not preserve old destination n=%zu index=%zu got=%d old_dst=%d src_stride_bytes=%zu\\n",
                n, index, dst[index], old_dst[index], src_stride_bytes);
        free(cmp_lhs);
        free(cmp_rhs);
        free(src);
        free(src_before);
        free(dst);
        free(old_dst);
        return 13;
      }}
      ++inactive_preserved_lanes;
    }}
  }}

  for (size_t index = n; index < dst_alloc_n; ++index) {{
    if (dst[index] != old_dst[index]) {{
      fprintf(stderr,
              "{expectation.kind} touched tail destination n=%zu index=%zu got=%d old_dst=%d\\n",
              n, index, dst[index], old_dst[index]);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(src_before);
      free(dst);
      free(old_dst);
      return 14;
    }}
  }}
  for (size_t raw = 0; raw < src_alloc_n; ++raw) {{
    if (src[raw] != src_before[raw]) {{
      fprintf(stderr,
              "{expectation.kind} source buffer mutated n=%zu raw_index=%zu got=%d before=%d src_stride_bytes=%zu\\n",
              n, raw, src[raw], src_before[raw], src_stride_bytes);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(src_before);
      free(dst);
      free(old_dst);
      return 15;
    }}
    int is_logical_strided_slot = 0;
    for (size_t lane = 0; lane < n; ++lane) {{
      if (raw * sizeof(int32_t) == lane * src_stride_bytes) {{
        is_logical_strided_slot = 1;
        break;
      }}
    }}
    if (!is_logical_strided_slot)
      ++source_gap_preserved_slots;
  }}

  if (n > 1 && (active_lanes == 0 || inactive_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} compare mask coverage missing n=%zu active_lanes=%zu inactive_lanes=%zu\\n",
            n, active_lanes, inactive_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(src_before);
    free(dst);
    free(old_dst);
    return 16;
  }}
  if (inactive_lanes != inactive_preserved_lanes) {{
    fprintf(stderr,
            "{expectation.kind} inactive preservation coverage mismatch n=%zu inactive_lanes=%zu preserved_lanes=%zu\\n",
            n, inactive_lanes, inactive_preserved_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(src_before);
    free(dst);
    free(old_dst);
    return 17;
  }}

  free(cmp_lhs);
  free(cmp_rhs);
  free(src);
  free(src_before);
  free(dst);
  free(old_dst);
  printf("{expectation.kind} case n=%zu ok src_stride_bytes=%zu computed_mask byte_strided_load active_lanes=%zu inactive_lanes=%zu inactive_preserved_lanes=%zu source_gap_preserved_slots=%zu tail_preserved\\n",
         n, src_stride_bytes, active_lanes, inactive_lanes,
         inactive_preserved_lanes, source_gap_preserved_slots);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t stride_bytes_values[] = {{{stride_values_literal}}};
  const size_t stride_count = sizeof(stride_bytes_values) / sizeof(stride_bytes_values[0]);
  for (size_t index = 0; index < count_count; ++index) {{
    for (size_t stride_index = 0; stride_index < stride_count; ++stride_index) {{
      int status = run_case(counts[index], stride_bytes_values[stride_index]);
      if (status != 0)
        return status;
    }}
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} stride_bytes={stride_values_summary}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} stride_bytes={stride_values_summary}\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_computed_masked_indexed_gather_load_unit_store:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static uint32_t make_index(size_t logical_index, size_t n) {{
  if (n == 0)
    return 0;
  size_t mixed = (logical_index * 5 + 3) % n;
  if ((logical_index % 2) == 0)
    mixed = (n - 1) - mixed;
  return (uint32_t)mixed;
}}

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t src_alloc_n = alloc_n + 8;
  size_t dst_alloc_n = alloc_n + 8;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src = (int32_t *)malloc(sizeof(int32_t) * src_alloc_n);
  int32_t *src_before = (int32_t *)malloc(sizeof(int32_t) * src_alloc_n);
  uint32_t *indices = (uint32_t *)malloc(sizeof(uint32_t) * alloc_n);
  int32_t *dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  int32_t *old_dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  if (!cmp_lhs || !cmp_rhs || !src || !src_before || !indices || !dst || !old_dst) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(src_before);
    free(indices);
    free(dst);
    free(old_dst);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs[index] = {expectation.lhs_initializer};
    cmp_rhs[index] = {expectation.rhs_initializer};
    indices[index] = make_index(index, alloc_n);
  }}
  for (size_t index = 0; index < src_alloc_n; ++index) {{
    src[index] = {expectation.source_initializer};
    src_before[index] = src[index];
  }}
  for (size_t index = 0; index < dst_alloc_n; ++index) {{
    dst[index] = {expectation.out_initializer};
    old_dst[index] = dst[index];
  }}

  {expectation.function_name}(cmp_lhs, cmp_rhs, src, indices, dst, n);

  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t inactive_preserved_lanes = 0;
  size_t noncontiguous_index_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    if ((size_t)indices[index] >= n) {{
      fprintf(stderr,
              "{expectation.kind} generated out-of-range index n=%zu index=%zu gather_index=%u\\n",
              n, index, indices[index]);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(src_before);
      free(indices);
      free(dst);
      free(old_dst);
      return 12;
    }}
    if ((size_t)indices[index] != index)
      ++noncontiguous_index_lanes;
    int active = cmp_lhs[index] < cmp_rhs[index];
    if (active)
      ++active_lanes;
    else
      ++inactive_lanes;
    int32_t expected = {expectation.expected_expression};
    if (dst[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu gather_index=%u got=%d expected=%d cmp_lhs=%d cmp_rhs=%d src_at_index=%d old_dst=%d\\n",
              n, index, indices[index], dst[index], expected, cmp_lhs[index],
              cmp_rhs[index], src[indices[index]], old_dst[index]);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(src_before);
      free(indices);
      free(dst);
      free(old_dst);
      return 13;
    }}
    if (!active) {{
      if (dst[index] != old_dst[index]) {{
        fprintf(stderr,
                "{expectation.kind} inactive lane did not preserve old destination n=%zu index=%zu got=%d old_dst=%d\\n",
                n, index, dst[index], old_dst[index]);
        free(cmp_lhs);
        free(cmp_rhs);
        free(src);
        free(src_before);
        free(indices);
        free(dst);
        free(old_dst);
        return 14;
      }}
      ++inactive_preserved_lanes;
    }}
  }}

  for (size_t index = n; index < dst_alloc_n; ++index) {{
    if (dst[index] != old_dst[index]) {{
      fprintf(stderr,
              "{expectation.kind} touched tail destination n=%zu index=%zu got=%d old_dst=%d\\n",
              n, index, dst[index], old_dst[index]);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(src_before);
      free(indices);
      free(dst);
      free(old_dst);
      return 15;
    }}
  }}
  for (size_t index = 0; index < src_alloc_n; ++index) {{
    if (src[index] != src_before[index]) {{
      fprintf(stderr,
              "{expectation.kind} source buffer mutated n=%zu index=%zu got=%d before=%d\\n",
              n, index, src[index], src_before[index]);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(src_before);
      free(indices);
      free(dst);
      free(old_dst);
      return 16;
    }}
  }}

  if (n > 1 && (active_lanes == 0 || inactive_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} compare mask coverage missing n=%zu active_lanes=%zu inactive_lanes=%zu\\n",
            n, active_lanes, inactive_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(src_before);
    free(indices);
    free(dst);
    free(old_dst);
    return 17;
  }}
  if (n > 3 && noncontiguous_index_lanes == 0) {{
    fprintf(stderr,
            "{expectation.kind} vacuous indexed gather check failed n=%zu first_indices=[%u,%u,%u]\\n",
            n, indices[0], indices[1], indices[2]);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(src_before);
    free(indices);
    free(dst);
    free(old_dst);
    return 18;
  }}
  if (inactive_lanes != inactive_preserved_lanes) {{
    fprintf(stderr,
            "{expectation.kind} inactive preservation coverage mismatch n=%zu inactive_lanes=%zu preserved_lanes=%zu\\n",
            n, inactive_lanes, inactive_preserved_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(src_before);
    free(indices);
    free(dst);
    free(old_dst);
    return 19;
  }}

  free(cmp_lhs);
  free(cmp_rhs);
  free(src);
  free(src_before);
  free(indices);
  free(dst);
  free(old_dst);
  printf("{expectation.kind} case n=%zu ok computed_mask indexed_gather active_lanes=%zu inactive_lanes=%zu inactive_preserved_lanes=%zu noncontiguous_index_lanes=%zu source_preserved tail_preserved\\n",
         n, active_lanes, inactive_lanes, inactive_preserved_lanes,
         noncontiguous_index_lanes);
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
    if expectation.is_computed_masked_indexed_scatter_store_unit_load:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static uint32_t make_index(size_t logical_index, size_t n) {{
  if (n == 0)
    return 0;
  size_t mixed = (n - 1) - (logical_index % n);
  if (n > 3)
    mixed = (mixed + 3) % n;
  return (uint32_t)mixed;
}}

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t src_alloc_n = alloc_n + 8;
  size_t dst_alloc_n = alloc_n + 8;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src = (int32_t *)malloc(sizeof(int32_t) * src_alloc_n);
  int32_t *src_before = (int32_t *)malloc(sizeof(int32_t) * src_alloc_n);
  uint32_t *indices = (uint32_t *)malloc(sizeof(uint32_t) * alloc_n);
  int32_t *dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  int32_t *old_dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  if (!cmp_lhs || !cmp_rhs || !src || !src_before || !indices || !dst || !old_dst) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(src_before);
    free(indices);
    free(dst);
    free(old_dst);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs[index] = {expectation.lhs_initializer};
    cmp_rhs[index] = {expectation.rhs_initializer};
    src[index] = {expectation.source_initializer};
    src_before[index] = src[index];
    indices[index] = make_index(index, alloc_n);
  }}
  for (size_t index = alloc_n; index < src_alloc_n; ++index) {{
    src[index] = {expectation.source_initializer};
    src_before[index] = src[index];
  }}
  for (size_t index = 0; index < dst_alloc_n; ++index) {{
    dst[index] = {expectation.out_initializer};
    old_dst[index] = dst[index];
  }}

  for (size_t index = 0; index < n; ++index) {{
    if ((size_t)indices[index] >= n) {{
      fprintf(stderr,
              "{expectation.kind} generated out-of-range index n=%zu index=%zu scatter_index=%u\\n",
              n, index, indices[index]);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(src_before);
      free(indices);
      free(dst);
      free(old_dst);
      return 12;
    }}
    for (size_t prior = 0; prior < index; ++prior) {{
      if (indices[prior] == indices[index]) {{
        fprintf(stderr,
                "{expectation.kind} duplicate scatter index n=%zu prior=%zu index=%zu scatter_index=%u\\n",
                n, prior, index, indices[index]);
        free(cmp_lhs);
        free(cmp_rhs);
        free(src);
        free(src_before);
        free(indices);
        free(dst);
        free(old_dst);
        return 13;
      }}
    }}
  }}

  {expectation.function_name}(cmp_lhs, cmp_rhs, src, indices, dst, n);

  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t inactive_preserved_lanes = 0;
  size_t noncontiguous_index_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    size_t dst_index = (size_t)indices[index];
    if (dst_index != index)
      ++noncontiguous_index_lanes;
    int active = cmp_lhs[index] < cmp_rhs[index];
    if (active)
      ++active_lanes;
    else
      ++inactive_lanes;
    int32_t expected = {expectation.expected_expression};
    if (dst[dst_index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu scatter_index=%u got=%d expected=%d cmp_lhs=%d cmp_rhs=%d src=%d old_dst_at_scatter=%d\\n",
              n, index, indices[index], dst[dst_index], expected,
              cmp_lhs[index], cmp_rhs[index], src[index], old_dst[dst_index]);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(src_before);
      free(indices);
      free(dst);
      free(old_dst);
      return 14;
    }}
    if (!active) {{
      if (dst[dst_index] != old_dst[dst_index]) {{
        fprintf(stderr,
                "{expectation.kind} inactive lane did not preserve destination n=%zu index=%zu scatter_index=%u got=%d old_dst=%d\\n",
                n, index, indices[index], dst[dst_index], old_dst[dst_index]);
        free(cmp_lhs);
        free(cmp_rhs);
        free(src);
        free(src_before);
        free(indices);
        free(dst);
        free(old_dst);
        return 15;
      }}
      ++inactive_preserved_lanes;
    }}
  }}

  for (size_t index = n; index < dst_alloc_n; ++index) {{
    if (dst[index] != old_dst[index]) {{
      fprintf(stderr,
              "{expectation.kind} touched tail destination n=%zu index=%zu got=%d old_dst=%d\\n",
              n, index, dst[index], old_dst[index]);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(src_before);
      free(indices);
      free(dst);
      free(old_dst);
      return 16;
    }}
  }}
  for (size_t index = 0; index < src_alloc_n; ++index) {{
    if (src[index] != src_before[index]) {{
      fprintf(stderr,
              "{expectation.kind} source buffer mutated n=%zu index=%zu got=%d before=%d\\n",
              n, index, src[index], src_before[index]);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(src_before);
      free(indices);
      free(dst);
      free(old_dst);
      return 17;
    }}
  }}

  if (n > 1 && (active_lanes == 0 || inactive_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} compare mask coverage missing n=%zu active_lanes=%zu inactive_lanes=%zu\\n",
            n, active_lanes, inactive_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(src_before);
    free(indices);
    free(dst);
    free(old_dst);
    return 18;
  }}
  if (n > 3 && noncontiguous_index_lanes == 0) {{
    fprintf(stderr,
            "{expectation.kind} vacuous indexed scatter check failed n=%zu first_indices=[%u,%u,%u]\\n",
            n, indices[0], indices[1], indices[2]);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(src_before);
    free(indices);
    free(dst);
    free(old_dst);
    return 19;
  }}
  if (inactive_lanes != inactive_preserved_lanes) {{
    fprintf(stderr,
            "{expectation.kind} inactive preservation coverage mismatch n=%zu inactive_lanes=%zu preserved_lanes=%zu\\n",
            n, inactive_lanes, inactive_preserved_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(src_before);
    free(indices);
    free(dst);
    free(old_dst);
    return 20;
  }}

  free(cmp_lhs);
  free(cmp_rhs);
  free(src);
  free(src_before);
  free(indices);
  free(dst);
  free(old_dst);
  printf("{expectation.kind} case n=%zu ok computed_mask indexed_scatter active_lanes=%zu inactive_lanes=%zu inactive_preserved_lanes=%zu noncontiguous_index_lanes=%zu source_preserved tail_preserved\\n",
         n, active_lanes, inactive_lanes, inactive_preserved_lanes,
         noncontiguous_index_lanes);
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
    if expectation.is_runtime_scalar_computed_masked_macc_add:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n, int32_t rhs_scalar) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n + 8;
  if (alloc_n == 8 && n == 0)
    alloc_n = 9;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!cmp_lhs || !lhs || !rhs || !acc || !out) {{
    fprintf(stderr, "allocation failed for n=%zu rhs_scalar=%d\\n", n, rhs_scalar);
    free(cmp_lhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs[index] = {expectation.lhs_initializer};
    lhs[index] = {expectation.true_value_initializer};
    rhs[index] = {expectation.false_value_initializer};
    acc[index] = {expectation.source_initializer};
    out[index] = {expectation.out_initializer};
  }}

  {expectation.function_name}(cmp_lhs, rhs_scalar, lhs, rhs, acc, out, n);

  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t inactive_acc_preserved = 0;
  size_t add_only_distinguishing = 0;
  size_t mul_only_distinguishing = 0;
  size_t signed_product_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int predicate = {expectation.compare_predicate_c_expression("cmp_lhs[index]", "rhs_scalar")};
    int32_t product = (int32_t)lhs[index] * (int32_t)rhs[index];
    int32_t expected = {expectation.expected_expression};
    int32_t add_only = (int32_t)(acc[index] + lhs[index] + rhs[index]);
    int32_t mul_only = product;
    if (predicate)
      ++active_lanes;
    else
      ++inactive_lanes;
    if (!predicate && out[index] == acc[index])
      ++inactive_acc_preserved;
    if (predicate && expected != add_only)
      ++add_only_distinguishing;
    if (predicate && expected != mul_only)
      ++mul_only_distinguishing;
    if (predicate && product != 0 && acc[index] != 0)
      ++signed_product_lanes;
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu rhs_scalar=%d index=%zu got=%d expected=%d cmp_lhs=%d lhs=%d rhs=%d acc=%d predicate=%d product=%d\\n",
              n, rhs_scalar, index, out[index], expected, cmp_lhs[index],
              lhs[index], rhs[index], acc[index], predicate, product);
      free(cmp_lhs);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 12;
    }}
  }}

  for (size_t index = n; index < alloc_n; ++index) {{
    if (out[index] != {expectation.out_initializer}) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu rhs_scalar=%d raw_index=%zu got=%d sentinel=%d\\n",
              n, rhs_scalar, index, out[index], {expectation.out_initializer});
      free(cmp_lhs);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 13;
    }}
  }}

  if (n > 4 && (active_lanes == 0 || inactive_lanes == 0 ||
                inactive_acc_preserved != inactive_lanes ||
                add_only_distinguishing == 0 ||
                mul_only_distinguishing == 0 ||
                signed_product_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} coverage missing n=%zu rhs_scalar=%d active=%zu inactive=%zu inactive_preserved=%zu add_dist=%zu mul_dist=%zu signed_products=%zu\\n",
            n, rhs_scalar, active_lanes, inactive_lanes,
            inactive_acc_preserved, add_only_distinguishing,
            mul_only_distinguishing, signed_product_lanes);
    free(cmp_lhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 14;
  }}

  free(cmp_lhs);
  free(lhs);
  free(rhs);
  free(acc);
  free(out);
  printf("{expectation.kind} case n=%zu rhs_scalar=%d ok runtime_scalar_computed_mask_macc active_lanes=%zu inactive_lanes=%zu inactive_acc_preserved=%zu add_only_distinguishing=%zu mul_only_distinguishing=%zu tail_preserved\\n",
         n, rhs_scalar, active_lanes, inactive_lanes,
         inactive_acc_preserved, add_only_distinguishing,
         mul_only_distinguishing);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const int32_t rhs_scalar_values[] = {{{scalar_values_literal}}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t scalar_count = sizeof(rhs_scalar_values) / sizeof(rhs_scalar_values[0]);
  for (size_t index = 0; index < count_count; ++index) {{
    for (size_t scalar_index = 0; scalar_index < scalar_count; ++scalar_index) {{
      int status = run_case(counts[index], rhs_scalar_values[scalar_index]);
      if (status != 0)
        return status;
    }}
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} rhs_scalars={scalar_values_summary}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} rhs_scalars={scalar_values_summary}\\n");
  return 0;
}}
""".lstrip()
    if (
        expectation.is_runtime_scalar_computed_mask_store
        or expectation.is_runtime_scalar_computed_mask_load_store
    ):
        runtime_masked_memory_label = (
            "runtime_scalar_computed_mask_load_store"
            if expectation.is_runtime_scalar_computed_mask_load_store
            else "runtime_scalar_computed_mask_store"
        )
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n, int32_t rhs_scalar) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t dst_alloc_n = alloc_n + 8;
  int32_t *lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src_before = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  int32_t *old_dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  if (!lhs || !src || !src_before || !dst || !old_dst) {{
    fprintf(stderr, "allocation failed for n=%zu rhs_scalar=%d\\n",
            n, rhs_scalar);
    free(lhs);
    free(src);
    free(src_before);
    free(dst);
    free(old_dst);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    lhs[index] = {expectation.lhs_initializer};
    src[index] = {expectation.source_initializer};
    src_before[index] = src[index];
  }}
  for (size_t index = 0; index < dst_alloc_n; ++index) {{
    dst[index] = {expectation.out_initializer};
    old_dst[index] = dst[index];
  }}

  {expectation.function_name}(lhs, rhs_scalar, src, dst, n);

  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t inactive_preserved_lanes = 0;
  size_t payload_distinguishing_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int active = lhs[index] <= rhs_scalar;
    if (active)
      ++active_lanes;
    else
      ++inactive_lanes;
    int32_t expected = {expectation.expected_expression};
    if (active && src[index] != old_dst[index])
      ++payload_distinguishing_lanes;
    if (dst[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu rhs_scalar=%d index=%zu got=%d expected=%d lhs=%d src=%d old_dst=%d\\n",
              n, rhs_scalar, index, dst[index], expected, lhs[index],
              src[index], old_dst[index]);
      free(lhs);
      free(src);
      free(src_before);
      free(dst);
      free(old_dst);
      return 12;
    }}
    if (!active) {{
      if (dst[index] != old_dst[index]) {{
        fprintf(stderr,
                "{expectation.kind} inactive lane did not preserve old destination n=%zu rhs_scalar=%d index=%zu got=%d old_dst=%d\\n",
                n, rhs_scalar, index, dst[index], old_dst[index]);
        free(lhs);
        free(src);
        free(src_before);
        free(dst);
        free(old_dst);
        return 13;
      }}
      ++inactive_preserved_lanes;
    }}
  }}

  for (size_t index = n; index < dst_alloc_n; ++index) {{
    if (dst[index] != old_dst[index]) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu rhs_scalar=%d raw_index=%zu got=%d old_dst=%d\\n",
              n, rhs_scalar, index, dst[index], old_dst[index]);
      free(lhs);
      free(src);
      free(src_before);
      free(dst);
      free(old_dst);
      return 14;
    }}
  }}
  for (size_t index = 0; index < alloc_n; ++index) {{
    if (src[index] != src_before[index]) {{
      fprintf(stderr,
              "{expectation.kind} source payload buffer mutated n=%zu rhs_scalar=%d index=%zu got=%d before=%d\\n",
              n, rhs_scalar, index, src[index], src_before[index]);
      free(lhs);
      free(src);
      free(src_before);
      free(dst);
      free(old_dst);
      return 15;
    }}
  }}

  if (n > 1 && (active_lanes == 0 || inactive_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} runtime scalar mask coverage missing n=%zu rhs_scalar=%d active_lanes=%zu inactive_lanes=%zu\\n",
            n, rhs_scalar, active_lanes, inactive_lanes);
    free(lhs);
    free(src);
    free(src_before);
    free(dst);
    free(old_dst);
    return 16;
  }}
  if (inactive_lanes != inactive_preserved_lanes) {{
    fprintf(stderr,
            "{expectation.kind} inactive preservation coverage mismatch n=%zu rhs_scalar=%d inactive_lanes=%zu preserved_lanes=%zu\\n",
            n, rhs_scalar, inactive_lanes, inactive_preserved_lanes);
    free(lhs);
    free(src);
    free(src_before);
    free(dst);
    free(old_dst);
    return 17;
  }}
  if (n > 1 && active_lanes != 0 && payload_distinguishing_lanes == 0) {{
    fprintf(stderr,
            "{expectation.kind} payload-distinguishing coverage missing n=%zu rhs_scalar=%d\\n",
            n, rhs_scalar);
    free(lhs);
    free(src);
    free(src_before);
    free(dst);
    free(old_dst);
    return 18;
  }}

  free(lhs);
  free(src);
  free(src_before);
  free(dst);
  free(old_dst);
  printf("{expectation.kind} case n=%zu rhs_scalar=%d ok {runtime_masked_memory_label} active_lanes=%zu inactive_lanes=%zu inactive_preserved_lanes=%zu payload_distinguishing_lanes=%zu source_preserved tail_preserved\\n",
         n, rhs_scalar, active_lanes, inactive_lanes,
         inactive_preserved_lanes, payload_distinguishing_lanes);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const int32_t rhs_scalar_values[] = {{{scalar_values_literal}}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t rhs_scalar_count = sizeof(rhs_scalar_values) / sizeof(rhs_scalar_values[0]);
  for (size_t count_index = 0; count_index < count_count; ++count_index) {{
    for (size_t scalar_index = 0; scalar_index < rhs_scalar_count; ++scalar_index) {{
      int status = run_case(counts[count_index], rhs_scalar_values[scalar_index]);
      if (status != 0)
        return status;
    }}
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} rhs_scalars={scalar_values_summary}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} rhs_scalars={scalar_values_summary}\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_computed_masked_unit_load_store:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t dst_alloc_n = alloc_n + 8;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  int32_t *old_dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  if (!cmp_lhs || !cmp_rhs || !src || !dst || !old_dst) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(dst);
    free(old_dst);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs[index] = {expectation.lhs_initializer};
    cmp_rhs[index] = {expectation.rhs_initializer};
    src[index] = {expectation.source_initializer};
    dst[index] = {expectation.out_initializer};
    old_dst[index] = dst[index];
  }}
  for (size_t index = alloc_n; index < dst_alloc_n; ++index) {{
    dst[index] = {OUT_SENTINEL};
    old_dst[index] = dst[index];
  }}

  {expectation.function_name}(cmp_lhs, cmp_rhs, src, dst, n);

  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t inactive_preserved_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int active = cmp_lhs[index] < cmp_rhs[index];
    if (active)
      ++active_lanes;
    else
      ++inactive_lanes;

    int32_t expected = {expectation.expected_expression};
    if (dst[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d cmp_lhs=%d cmp_rhs=%d src=%d old_dst=%d\\n",
              n, index, dst[index], expected, cmp_lhs[index], cmp_rhs[index],
              src[index], old_dst[index]);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(dst);
      free(old_dst);
      return 12;
    }}

    if (!active) {{
      if (dst[index] != old_dst[index]) {{
        fprintf(stderr,
                "{expectation.kind} inactive lane did not preserve old destination n=%zu index=%zu got=%d old_dst=%d\\n",
                n, index, dst[index], old_dst[index]);
        free(cmp_lhs);
        free(cmp_rhs);
        free(src);
        free(dst);
        free(old_dst);
        return 13;
      }}
      ++inactive_preserved_lanes;
    }}
  }}

  for (size_t index = n; index < dst_alloc_n; ++index) {{
    if (dst[index] != old_dst[index]) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d old_dst=%d\\n",
              n, index, dst[index], old_dst[index]);
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      free(dst);
      free(old_dst);
      return 14;
    }}
  }}

  if (n > 1 && (active_lanes == 0 || inactive_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} compare mask coverage missing n=%zu active_lanes=%zu inactive_lanes=%zu\\n",
            n, active_lanes, inactive_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(dst);
    free(old_dst);
    return 15;
  }}
  if (inactive_lanes != inactive_preserved_lanes) {{
    fprintf(stderr,
            "{expectation.kind} inactive preservation coverage mismatch n=%zu inactive_lanes=%zu preserved_lanes=%zu\\n",
            n, inactive_lanes, inactive_preserved_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    free(dst);
    free(old_dst);
    return 16;
  }}

  free(cmp_lhs);
  free(cmp_rhs);
  free(src);
  free(dst);
  free(old_dst);
  printf("{expectation.kind} case n=%zu ok compare_active_lanes=%zu compare_inactive_lanes=%zu inactive_preserved_lanes=%zu tail_preserved\\n",
         n, active_lanes, inactive_lanes, inactive_preserved_lanes);
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
    if expectation.is_masked_unit_load_store or expectation.is_masked_unit_store:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t dst_alloc_n = alloc_n + 8;
  int32_t *src = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *mask = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  int32_t *old_dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  if (!src || !mask || !dst || !old_dst) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(src);
    free(mask);
    free(dst);
    free(old_dst);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    src[index] = {expectation.lhs_initializer};
    mask[index] = {expectation.rhs_initializer};
    dst[index] = {expectation.out_initializer};
    old_dst[index] = dst[index];
  }}
  for (size_t index = alloc_n; index < dst_alloc_n; ++index) {{
    dst[index] = {OUT_SENTINEL};
    old_dst[index] = dst[index];
  }}

  {expectation.function_name}(src, mask, dst, n);

  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t inactive_preserved_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int active = mask[index] != 0;
    if (active)
      ++active_lanes;
    else
      ++inactive_lanes;

    int32_t expected = {expectation.expected_expression};
    if (dst[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d src=%d mask=%d old_dst=%d\\n",
              n, index, dst[index], expected, src[index], mask[index],
              old_dst[index]);
      free(src);
      free(mask);
      free(dst);
      free(old_dst);
      return 12;
    }}

    if (!active) {{
      if (dst[index] != old_dst[index]) {{
        fprintf(stderr,
                "{expectation.kind} inactive lane did not preserve old destination n=%zu index=%zu got=%d old_dst=%d\\n",
                n, index, dst[index], old_dst[index]);
        free(src);
        free(mask);
        free(dst);
        free(old_dst);
        return 13;
      }}
      ++inactive_preserved_lanes;
    }}
  }}

  for (size_t index = n; index < dst_alloc_n; ++index) {{
    if (dst[index] != old_dst[index]) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d old_dst=%d\\n",
              n, index, dst[index], old_dst[index]);
      free(src);
      free(mask);
      free(dst);
      free(old_dst);
      return 14;
    }}
  }}

  if (n > 1 && (active_lanes == 0 || inactive_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} mask coverage missing n=%zu active_lanes=%zu inactive_lanes=%zu\\n",
            n, active_lanes, inactive_lanes);
    free(src);
    free(mask);
    free(dst);
    free(old_dst);
    return 15;
  }}
  if (inactive_lanes != inactive_preserved_lanes) {{
    fprintf(stderr,
            "{expectation.kind} inactive preservation coverage mismatch n=%zu inactive_lanes=%zu preserved_lanes=%zu\\n",
            n, inactive_lanes, inactive_preserved_lanes);
    free(src);
    free(mask);
    free(dst);
    free(old_dst);
    return 16;
  }}

  free(src);
  free(mask);
  free(dst);
  free(old_dst);
  printf("{expectation.kind} case n=%zu ok active_lanes=%zu inactive_lanes=%zu inactive_preserved_lanes=%zu tail_preserved\\n",
         n, active_lanes, inactive_lanes, inactive_preserved_lanes);
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
    if expectation.is_indexed_scatter_unit_load:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static uint32_t make_unique_index(size_t logical_index, size_t n) {{
  if (n == 0)
    return 0;
  return (uint32_t)((logical_index * 5 + 3) % n);
}}

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t dst_alloc_n = alloc_n + 8;
  int32_t *src = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  uint32_t *indices = (uint32_t *)malloc(sizeof(uint32_t) * alloc_n);
  int32_t *dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  uint8_t *seen = (uint8_t *)calloc(alloc_n, sizeof(uint8_t));
  if (!src || !indices || !dst || !seen) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(src);
    free(indices);
    free(dst);
    free(seen);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    src[index] = {expectation.lhs_initializer};
    indices[index] = make_unique_index(index, alloc_n);
    dst[index] = {OUT_SENTINEL};
  }}
  for (size_t index = alloc_n; index < dst_alloc_n; ++index)
    dst[index] = {OUT_SENTINEL};

  for (size_t logical_index = 0; logical_index < n; ++logical_index) {{
    uint32_t dst_index = indices[logical_index];
    if ((size_t)dst_index >= n) {{
      fprintf(stderr,
              "{expectation.kind} generated out-of-range index n=%zu logical_index=%zu dst_index=%u\\n",
              n, logical_index, dst_index);
      free(src);
      free(indices);
      free(dst);
      free(seen);
      return 12;
    }}
    if (seen[dst_index] != 0) {{
      fprintf(stderr,
              "{expectation.kind} duplicate index unsupported n=%zu logical_index=%zu dst_index=%u\\n",
              n, logical_index, dst_index);
      free(src);
      free(indices);
      free(dst);
      free(seen);
      return 13;
    }}
    seen[dst_index] = 1;
  }}

  {expectation.function_name}(src, indices, dst, n);

  for (size_t logical_index = 0; logical_index < n; ++logical_index) {{
    uint32_t dst_index = indices[logical_index];
    int32_t expected = {expectation.expected_expression};
    if (dst[dst_index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu logical_index=%zu scatter_index=%u got=%d expected=%d src=%d\\n",
              n, logical_index, dst_index, dst[dst_index], expected,
              src[logical_index]);
      free(src);
      free(indices);
      free(dst);
      free(seen);
      return 14;
    }}
  }}

  for (size_t index = n; index < dst_alloc_n; ++index) {{
    if (dst[index] != {OUT_SENTINEL}) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, dst[index], {OUT_SENTINEL});
      free(src);
      free(indices);
      free(dst);
      free(seen);
      return 15;
    }}
  }}

  if (n > 3 && indices[0] == 0 && indices[1] == 1 && indices[2] == 2) {{
    fprintf(stderr,
            "{expectation.kind} vacuous indexed scatter check failed n=%zu indices=[%u,%u,%u]\\n",
            n, indices[0], indices[1], indices[2]);
    free(src);
    free(indices);
    free(dst);
    free(seen);
    return 16;
  }}

  free(src);
  free(indices);
  free(dst);
  free(seen);
  printf("{expectation.kind} case n=%zu ok unique_non_monotonic_indexed_scatter\\n", n);
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
    if expectation.is_indexed_gather_unit_store:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static uint32_t make_index(size_t logical_index, size_t n) {{
  if (n == 0)
    return 0;
  size_t mixed = (logical_index * 5 + 3) % n;
  if ((logical_index % 2) == 0)
    mixed = (n - 1) - mixed;
  return (uint32_t)mixed;
}}

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  int32_t *data = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  uint32_t *indices = (uint32_t *)malloc(sizeof(uint32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!data || !indices || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(data);
    free(indices);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    data[index] = {expectation.lhs_initializer};
    indices[index] = make_index(index, alloc_n);
    out[index] = {OUT_SENTINEL};
  }}

  {expectation.function_name}(data, indices, out, n);

  for (size_t index = 0; index < n; ++index) {{
    int32_t expected = {expectation.expected_expression};
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu gather_index=%u got=%d expected=%d data_at_index=%d\\n",
              n, index, indices[index], out[index], expected,
              data[indices[index]]);
      free(data);
      free(indices);
      free(out);
      return 12;
    }}
  }}

  if (n > 3 && indices[0] == 0 && indices[1] == 1 && indices[2] == 2) {{
    fprintf(stderr,
            "{expectation.kind} vacuous indexed gather check failed n=%zu indices=[%u,%u,%u]\\n",
            n, indices[0], indices[1], indices[2]);
    free(data);
    free(indices);
    free(out);
    return 13;
  }}

  free(data);
  free(indices);
  free(out);
  printf("{expectation.kind} case n=%zu ok non_monotonic_indexed_gather\\n", n);
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
    if expectation.is_strided_load_unit_store:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n, size_t stride_bytes) {{
  /* expected: {expectation.expected_expression} */
  size_t src_alloc_bytes = (n == 0 ? 1 : n) * stride_bytes + 32;
  size_t out_alloc_n = (n == 0 ? 1 : n) + 8;
  uint8_t *src_raw = (uint8_t *)malloc(src_alloc_bytes);
  int32_t *src = (int32_t *)src_raw;
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * out_alloc_n);
  if (!src_raw || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(src_raw);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < src_alloc_bytes; ++index)
    src_raw[index] = (uint8_t)0xA5;
  for (size_t index = 0; index < out_alloc_n; ++index)
    out[index] = {OUT_SENTINEL};

  for (size_t index = 0; index < n; ++index)
    *(int32_t *)(src_raw + index * stride_bytes) = {expectation.lhs_initializer};

  {expectation.function_name}(src, out, n, stride_bytes);

  for (size_t index = 0; index < n; ++index) {{
    int32_t expected = *(int32_t *)(src_raw + index * stride_bytes);
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d stride_bytes=%zu\\n",
              n, index, out[index], expected, stride_bytes);
      free(src_raw);
      free(out);
      return 12;
    }}
  }}

  for (size_t index = n; index < out_alloc_n; ++index) {{
    if (out[index] != {OUT_SENTINEL}) {{
      fprintf(stderr,
              "{expectation.kind} touched unit-store tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, out[index], {OUT_SENTINEL});
      free(src_raw);
      free(out);
      return 13;
    }}
  }}

  if (n > 1 && stride_bytes > sizeof(int32_t) &&
      *(int32_t *)(src_raw + sizeof(int32_t)) == out[1]) {{
    fprintf(stderr,
            "{expectation.kind} vacuous byte-stride check failed n=%zu stride_bytes=%zu contiguous_slot=%d out[1]=%d\\n",
            n, stride_bytes, *(int32_t *)(src_raw + sizeof(int32_t)), out[1]);
    free(src_raw);
    free(out);
    return 14;
  }}

  free(src_raw);
  free(out);
  printf("{expectation.kind} case n=%zu ok stride_bytes=%zu byte_strided_load contiguous_output tail_preserved\\n", n, stride_bytes);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t stride_bytes_values[] = {{{stride_values_literal}}};
  const size_t stride_count = sizeof(stride_bytes_values) / sizeof(stride_bytes_values[0]);
  for (size_t index = 0; index < count_count; ++index) {{
    for (size_t stride_index = 0; stride_index < stride_count; ++stride_index) {{
      int status = run_case(counts[index], stride_bytes_values[stride_index]);
      if (status != 0)
        return status;
    }}
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} stride_bytes={stride_values_summary}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} stride_bytes={stride_values_summary}\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_unit_load_strided_store:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n, size_t dst_stride_bytes) {{
  /* expected: {expectation.expected_expression} */
  size_t src_alloc_n = (n == 0 ? 1 : n) + 8;
  size_t dst_alloc_bytes = (n == 0 ? 1 : n) * dst_stride_bytes + 32;
  size_t dst_alloc_n = (dst_alloc_bytes + sizeof(int32_t) - 1) / sizeof(int32_t);
  int32_t *src = (int32_t *)malloc(sizeof(int32_t) * src_alloc_n);
  int32_t *dst = (int32_t *)malloc(sizeof(int32_t) * dst_alloc_n);
  uint8_t *dst_raw = (uint8_t *)dst;
  if (!src || !dst) {{
    fprintf(stderr, "allocation failed for n=%zu dst_stride_bytes=%zu\\n",
            n, dst_stride_bytes);
    free(src);
    free(dst);
    return 11;
  }}

  for (size_t index = 0; index < src_alloc_n; ++index)
    src[index] = (int32_t)(-900000 - (int32_t)index);
  for (size_t index = 0; index < dst_alloc_n; ++index)
    dst[index] = {OUT_SENTINEL};

  for (size_t index = 0; index < n; ++index)
    src[index] = {expectation.lhs_initializer};

  {expectation.function_name}(src, dst, n, dst_stride_bytes);

  for (size_t index = 0; index < n; ++index) {{
    int32_t expected = {expectation.expected_expression};
    size_t dst_byte_offset = index * dst_stride_bytes;
    int32_t got = *(int32_t *)(dst_raw + dst_byte_offset);
    if (got != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu dst_byte_offset=%zu got=%d expected=%d src=%d dst_stride_bytes=%zu\\n",
              n, index, dst_byte_offset, got, expected, src[index],
              dst_stride_bytes);
      free(src);
      free(dst);
      return 12;
    }}
  }}

  for (size_t index = 0; index < dst_alloc_n; ++index) {{
    int is_selected_slot = 0;
    for (size_t lane = 0; lane < n; ++lane) {{
      if (index * sizeof(int32_t) == lane * dst_stride_bytes) {{
        is_selected_slot = 1;
        break;
      }}
    }}
    if (is_selected_slot)
        continue;
    if (dst[index] != {OUT_SENTINEL}) {{
      fprintf(stderr,
              "{expectation.kind} touched skipped/tail destination sentinel n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, dst[index], {OUT_SENTINEL});
      free(src);
      free(dst);
      return 13;
    }}
  }}

  if (n > 1 && dst_stride_bytes > sizeof(int32_t) && dst[1] != {OUT_SENTINEL}) {{
    fprintf(stderr,
            "{expectation.kind} vacuous byte-strided-store check failed n=%zu dst_stride_bytes=%zu dst[1]=%d sentinel=%d\\n",
            n, dst_stride_bytes, dst[1], {OUT_SENTINEL});
    free(src);
    free(dst);
    return 14;
  }}

  free(src);
  free(dst);
  printf("{expectation.kind} case n=%zu ok dst_stride_bytes=%zu byte_strided_store selected_destination_writes sentinel_preserved tail_preserved\\n",
         n, dst_stride_bytes);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t stride_bytes_values[] = {{{stride_values_literal}}};
  const size_t stride_count = sizeof(stride_bytes_values) / sizeof(stride_bytes_values[0]);
  for (size_t index = 0; index < count_count; ++index) {{
    for (size_t stride_index = 0; stride_index < stride_count; ++stride_index) {{
      int status = run_case(counts[index], stride_bytes_values[stride_index]);
      if (status != 0)
        return status;
    }}
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} stride_bytes={stride_values_summary}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} stride_bytes={stride_values_summary}\\n");
  return 0;
}}
""".lstrip()
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
  {expectation.element_c_type} *lhs = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
  {expectation.element_c_type} *rhs = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
  {expectation.element_c_type} *out = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
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
    if expectation.is_runtime_scalar_splat_store:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n, int32_t rhs_scalar) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = (n == 0 ? 1 : n) + 8;
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index)
    out[index] = {expectation.out_initializer};

  {expectation.function_name}(rhs_scalar, out, n);

  for (size_t index = 0; index < n; ++index) {{
    int32_t expected = {expectation.expected_expression};
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d rhs_scalar=%d\\n",
              n, index, out[index], expected, rhs_scalar);
      free(out);
      return 12;
    }}
  }}

  for (size_t index = n; index < alloc_n; ++index) {{
    if (out[index] != {expectation.out_initializer}) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d rhs_scalar=%d\\n",
              n, index, out[index], {expectation.out_initializer}, rhs_scalar);
      free(out);
      return 13;
    }}
  }}

  free(out);
  printf("{expectation.kind} case n=%zu ok rhs_scalar=%d tail_preserved\\n",
         n, rhs_scalar);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const int32_t rhs_scalar_values[] = {{{scalar_values_literal}}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t scalar_count = sizeof(rhs_scalar_values) / sizeof(rhs_scalar_values[0]);
  for (size_t scalar_index = 0; scalar_index < scalar_count; ++scalar_index) {{
    for (size_t index = 0; index < count_count; ++index) {{
      int status = run_case(counts[index], rhs_scalar_values[scalar_index]);
      if (status != 0)
        return status;
    }}
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} rhs_scalars={scalar_values_summary}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} rhs_scalars={scalar_values_summary}\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_scalar_broadcast_elementwise:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n, int32_t rhs_scalar) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = (n == 0 ? 1 : n) + 8;
  {expectation.element_c_type} *lhs = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
  {expectation.element_c_type} *out = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
  if (!lhs || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(lhs);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    lhs[index] = {expectation.lhs_initializer};
    out[index] = {expectation.out_initializer};
  }}

  {expectation.function_name}(lhs, rhs_scalar, out, n);

  for (size_t index = 0; index < n; ++index) {{
    int32_t expected = {expectation.expected_expression};
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d lhs=%d rhs_scalar=%d\\n",
              n, index, out[index], expected, lhs[index], rhs_scalar);
      free(lhs);
      free(out);
      return 12;
    }}
  }}

  for (size_t index = n; index < alloc_n; ++index) {{
    if (out[index] != {expectation.out_initializer}) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d rhs_scalar=%d\\n",
              n, index, out[index], {expectation.out_initializer}, rhs_scalar);
      free(lhs);
      free(out);
      return 13;
    }}
  }}

  free(lhs);
  free(out);
  printf("{expectation.kind} case n=%zu ok rhs_scalar=%d tail_preserved\\n",
         n, rhs_scalar);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const int32_t rhs_scalar_values[] = {{{scalar_values_literal}}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t scalar_count = sizeof(rhs_scalar_values) / sizeof(rhs_scalar_values[0]);
  for (size_t scalar_index = 0; scalar_index < scalar_count; ++scalar_index) {{
    for (size_t index = 0; index < count_count; ++index) {{
      int status = run_case(counts[index], rhs_scalar_values[scalar_index]);
      if (status != 0)
        return status;
    }}
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} rhs_scalars={scalar_values_summary}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} rhs_scalars={scalar_values_summary}\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_standalone_reduce:
        if expectation.is_standalone_reduce_min:
            expected_update = (
                "expected = (expected < lhs[index]) ? expected : lhs[index];"
            )
        elif expectation.is_standalone_reduce_max:
            expected_update = (
                "expected = (expected > lhs[index]) ? expected : lhs[index];"
            )
        else:
            expected_update = "expected = (int32_t)(expected + lhs[index]);"
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int32_t make_lhs_value(size_t index) {{
  return (int32_t)(((index % 7) < 3)
                       ? -((int32_t)(index % 31) + 1)
                       : ((int32_t)(index % 37) + 2));
}}

static int run_case(size_t n, int32_t seed) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  int32_t *lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t acc[1];
  int32_t out[4];
  if (!lhs) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    return 11;
  }}

  int32_t expected = seed;
  for (size_t index = 0; index < alloc_n; ++index) {{
    lhs[index] = make_lhs_value(index);
    if (index < n)
      {expected_update}
  }}
  acc[0] = seed;
  for (size_t index = 0; index < sizeof(out) / sizeof(out[0]); ++index)
    out[index] = {OUT_SENTINEL};

  {expectation.function_name}(lhs, acc, out, n);

  if (out[0] != expected) {{
    fprintf(stderr,
            "{expectation.kind} mismatch n=%zu seed=%d got=%d expected=%d\\n",
            n, seed, out[0], expected);
    free(lhs);
    return 12;
  }}
  if (acc[0] != seed) {{
    fprintf(stderr,
            "{expectation.kind} mutated seed input n=%zu got=%d expected=%d\\n",
            n, acc[0], seed);
    free(lhs);
    return 13;
  }}
  for (size_t index = 1; index < sizeof(out) / sizeof(out[0]); ++index) {{
    if (out[index] != {OUT_SENTINEL}) {{
      fprintf(stderr,
              "{expectation.kind} touched scalar-output sentinel n=%zu seed=%d index=%zu got=%d sentinel=%d\\n",
              n, seed, index, out[index], {OUT_SENTINEL});
      free(lhs);
      return 14;
    }}
  }}

  free(lhs);
  printf("{expectation.kind} case n=%zu seed=%d ok scalar_out=%d tail_preserved\\n",
         n, seed, out[0]);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const int32_t seeds[] = {{(int32_t)-11, (int32_t)17}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t seed_count = sizeof(seeds) / sizeof(seeds[0]);
  for (size_t seed_index = 0; seed_index < seed_count; ++seed_index) {{
    for (size_t count_index = 0; count_index < count_count; ++count_index) {{
      int status = run_case(counts[count_index], seeds[seed_index]);
      if (status != 0)
        return status;
    }}
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} seeds=-11,17\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} seeds=-11,17\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_computed_mask_standalone_reduce:
        active_expression = expectation.compare_predicate_c_expression(
            "cmp_lhs[index]", "cmp_rhs[index]"
        )
        if expectation.is_computed_mask_standalone_reduce_min:
            expected_update = (
                "expected = (expected < src[index]) ? expected : src[index];"
            )
        elif expectation.is_computed_mask_standalone_reduce_max:
            expected_update = (
                "expected = (expected > src[index]) ? expected : src[index];"
            )
        else:
            expected_update = "expected = (int32_t)(expected + src[index]);"
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int32_t make_cmp_lhs_value(size_t index) {{
  return {expectation.lhs_initializer};
}}

static int32_t make_cmp_rhs_value(size_t index) {{
  return {expectation.rhs_initializer};
}}

static int32_t make_src_value(size_t index) {{
  return {expectation.source_initializer};
}}

static int run_case(size_t n, int32_t seed) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t acc[1];
  int32_t out[4];
  if (!cmp_lhs || !cmp_rhs || !src) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    return 11;
  }}

  int32_t expected = seed;
  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs[index] = make_cmp_lhs_value(index);
    cmp_rhs[index] = make_cmp_rhs_value(index);
    src[index] = make_src_value(index);
    if (index < n) {{
      int active = {active_expression};
      if (active) {{
        {expected_update}
        ++active_lanes;
      }} else {{
        ++inactive_lanes;
      }}
    }}
  }}
  acc[0] = seed;
  for (size_t index = 0; index < sizeof(out) / sizeof(out[0]); ++index)
    out[index] = {OUT_SENTINEL};

  {expectation.function_name}(cmp_lhs, cmp_rhs, src, acc, out, n);

  if (out[0] != expected) {{
    fprintf(stderr,
            "{expectation.kind} mismatch n=%zu seed=%d got=%d expected=%d active_lanes=%zu inactive_lanes=%zu\\n",
            n, seed, out[0], expected, active_lanes, inactive_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    return 12;
  }}
  if (acc[0] != seed) {{
    fprintf(stderr,
            "{expectation.kind} mutated seed input n=%zu got=%d expected=%d\\n",
            n, acc[0], seed);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    return 13;
  }}
  for (size_t index = 1; index < sizeof(out) / sizeof(out[0]); ++index) {{
    if (out[index] != {OUT_SENTINEL}) {{
      fprintf(stderr,
              "{expectation.kind} touched scalar-output sentinel n=%zu seed=%d index=%zu got=%d sentinel=%d\\n",
              n, seed, index, out[index], {OUT_SENTINEL});
      free(cmp_lhs);
      free(cmp_rhs);
      free(src);
      return 14;
    }}
  }}
  if (n > 1 && (active_lanes == 0 || inactive_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} compare mask coverage missing n=%zu active_lanes=%zu inactive_lanes=%zu\\n",
            n, active_lanes, inactive_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(src);
    return 15;
  }}

  free(cmp_lhs);
  free(cmp_rhs);
  free(src);
  printf("{expectation.kind} case n=%zu seed=%d ok scalar_out=%d active_lanes=%zu inactive_lanes=%zu tail_preserved\\n",
         n, seed, out[0], active_lanes, inactive_lanes);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const int32_t seeds[] = {{(int32_t)-11, (int32_t)17}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t seed_count = sizeof(seeds) / sizeof(seeds[0]);
  for (size_t seed_index = 0; seed_index < seed_count; ++seed_index) {{
    for (size_t count_index = 0; count_index < count_count; ++count_index) {{
      int status = run_case(counts[count_index], seeds[seed_index]);
      if (status != 0)
        return status;
    }}
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} seeds=-11,17\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} seeds=-11,17\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_runtime_scalar_computed_mask_standalone_reduce:
        active_expression = expectation.compare_predicate_c_expression(
            "cmp_lhs[index]", "rhs_scalar"
        )
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int32_t make_cmp_lhs_value(size_t index) {{
  return {expectation.lhs_initializer};
}}

static int32_t make_src_value(size_t index) {{
  return {expectation.source_initializer};
}}

static int run_case(size_t n, int32_t rhs_scalar, int32_t seed) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *src = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t acc[1];
  int32_t out[4];
  if (!cmp_lhs || !src) {{
    fprintf(stderr, "allocation failed for n=%zu rhs_scalar=%d\\n",
            n, rhs_scalar);
    free(cmp_lhs);
    free(src);
    return 11;
  }}

  int32_t expected = seed;
  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t active_positive = 0;
  size_t active_negative = 0;
  size_t inactive_nonzero = 0;
  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs[index] = make_cmp_lhs_value(index);
    src[index] = make_src_value(index);
    if (index < n) {{
      int active = {active_expression};
      if (active) {{
        expected = (int32_t)(expected + src[index]);
        ++active_lanes;
        if (src[index] > 0)
          ++active_positive;
        if (src[index] < 0)
          ++active_negative;
      }} else {{
        ++inactive_lanes;
        if (src[index] != 0)
          ++inactive_nonzero;
      }}
    }}
  }}
  acc[0] = seed;
  for (size_t index = 0; index < sizeof(out) / sizeof(out[0]); ++index)
    out[index] = {OUT_SENTINEL};

  {expectation.function_name}(cmp_lhs, rhs_scalar, src, acc, out, n);

  if (out[0] != expected) {{
    fprintf(stderr,
            "{expectation.kind} mismatch n=%zu rhs_scalar=%d seed=%d got=%d expected=%d active_lanes=%zu inactive_lanes=%zu\\n",
            n, rhs_scalar, seed, out[0], expected, active_lanes,
            inactive_lanes);
    free(cmp_lhs);
    free(src);
    return 12;
  }}
  if (acc[0] != seed) {{
    fprintf(stderr,
            "{expectation.kind} mutated seed input n=%zu rhs_scalar=%d got=%d expected=%d\\n",
            n, rhs_scalar, acc[0], seed);
    free(cmp_lhs);
    free(src);
    return 13;
  }}
  for (size_t index = 1; index < sizeof(out) / sizeof(out[0]); ++index) {{
    if (out[index] != {OUT_SENTINEL}) {{
      fprintf(stderr,
              "{expectation.kind} touched scalar-output sentinel n=%zu rhs_scalar=%d seed=%d index=%zu got=%d sentinel=%d\\n",
              n, rhs_scalar, seed, index, out[index], {OUT_SENTINEL});
      free(cmp_lhs);
      free(src);
      return 14;
    }}
  }}
  if (n > 1 && (active_lanes == 0 || inactive_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} runtime scalar mask coverage missing n=%zu rhs_scalar=%d active_lanes=%zu inactive_lanes=%zu\\n",
            n, rhs_scalar, active_lanes, inactive_lanes);
    free(cmp_lhs);
    free(src);
    return 15;
  }}
  if (n > 1 && (active_positive == 0 || active_negative == 0)) {{
    fprintf(stderr,
            "{expectation.kind} payload-sign coverage missing n=%zu rhs_scalar=%d active_positive=%zu active_negative=%zu\\n",
            n, rhs_scalar, active_positive, active_negative);
    free(cmp_lhs);
    free(src);
    return 16;
  }}
  if (n > 1 && inactive_nonzero == 0) {{
    fprintf(stderr,
            "{expectation.kind} inactive exclusion coverage missing n=%zu rhs_scalar=%d\\n",
            n, rhs_scalar);
    free(cmp_lhs);
    free(src);
    return 17;
  }}

  free(cmp_lhs);
  free(src);
  printf("{expectation.kind} case n=%zu rhs_scalar=%d seed=%d ok runtime_scalar_computed_mask_standalone_reduce scalar_out=%d active_lanes=%zu inactive_lanes=%zu active_positive=%zu active_negative=%zu inactive_nonzero=%zu tail_preserved\\n",
         n, rhs_scalar, seed, out[0], active_lanes, inactive_lanes,
         active_positive, active_negative, inactive_nonzero);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const int32_t rhs_scalar_values[] = {{{scalar_values_literal}}};
  const int32_t seeds[] = {{(int32_t)-11, (int32_t)17}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t scalar_count = sizeof(rhs_scalar_values) / sizeof(rhs_scalar_values[0]);
  const size_t seed_count = sizeof(seeds) / sizeof(seeds[0]);
  for (size_t scalar_index = 0; scalar_index < scalar_count; ++scalar_index) {{
    for (size_t seed_index = 0; seed_index < seed_count; ++seed_index) {{
      for (size_t count_index = 0; count_index < count_count; ++count_index) {{
        int status = run_case(counts[count_index], rhs_scalar_values[scalar_index],
                              seeds[seed_index]);
        if (status != 0)
          return status;
      }}
    }}
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} rhs_scalars={scalar_values_summary} seeds=-11,17\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} rhs_scalars={scalar_values_summary} seeds=-11,17\\n");
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
  {expectation.element_c_type} *lhs = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
  {expectation.element_c_type} *rhs = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
  {expectation.element_c_type} *out = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
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
    if expectation.is_widen_i32_to_i64:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  int32_t *lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int64_t *out = (int64_t *)malloc(sizeof(int64_t) * alloc_n);
  if (!lhs || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(lhs);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < n; ++index) {{
    lhs[index] = {expectation.lhs_initializer};
    out[index] = {expectation.out_initializer};
  }}

  {expectation.function_name}(lhs, out, n);

  for (size_t index = 0; index < n; ++index) {{
    int64_t expected = {expectation.expected_expression};
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%lld expected=%lld lhs=%d\\n",
              n, index, (long long)out[index], (long long)expected, lhs[index]);
      free(lhs);
      free(out);
      return 12;
    }}
  }}

  free(lhs);
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
    if expectation.is_widen_i16_to_i32:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n + 5;
  if (alloc_n == 5 && n == 0)
    alloc_n = 6;
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!lhs || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(lhs);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    lhs[index] = {expectation.lhs_initializer};
    out[index] = {expectation.out_initializer};
  }}

  {expectation.function_name}(lhs, out, n);

  size_t negative_lanes = 0;
  size_t positive_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int32_t expected = {expectation.expected_expression};
    if (lhs[index] < 0)
      ++negative_lanes;
    if (lhs[index] > 0)
      ++positive_lanes;
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d lhs=%d\\n",
              n, index, out[index], expected, lhs[index]);
      free(lhs);
      free(out);
      return 12;
    }}
    if (lhs[index] < 0 && out[index] >= 0) {{
      fprintf(stderr,
              "{expectation.kind} sign-extension failed n=%zu index=%zu lhs=%d out=%d\\n",
              n, index, lhs[index], out[index]);
      free(lhs);
      free(out);
      return 13;
    }}
  }}

  for (size_t index = n; index < alloc_n; ++index) {{
    if (out[index] != {expectation.out_initializer}) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, out[index], {expectation.out_initializer});
      free(lhs);
      free(out);
      return 14;
    }}
  }}

  if (n > 1 && (negative_lanes == 0 || positive_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} sign-extension coverage missing n=%zu negative_lanes=%zu positive_lanes=%zu\\n",
            n, negative_lanes, positive_lanes);
    free(lhs);
    free(out);
    return 15;
  }}

  free(lhs);
  free(out);
  printf("{expectation.kind} case n=%zu ok sign_extension_checked tail_preserved\\n", n);
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
    if expectation.is_computed_masked_strided_input_widening_dot_reduce_add:
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
  size_t cmp_alloc = n + 5;
  size_t lhs_alloc = n * lhs_stride + 8;
  size_t rhs_alloc = n * rhs_stride + 8;
  size_t out_alloc = n + 5;
  if (cmp_alloc == 5 && n == 0)
    cmp_alloc = 6;
  if (out_alloc == 5 && n == 0)
    out_alloc = 6;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * cmp_alloc);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * cmp_alloc);
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * lhs_alloc);
  int16_t *rhs = (int16_t *)malloc(sizeof(int16_t) * rhs_alloc);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * out_alloc);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * out_alloc);
  if (!cmp_lhs || !cmp_rhs || !lhs || !rhs || !acc || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(cmp_lhs);
    free(cmp_rhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < cmp_alloc; ++index) {{
    cmp_lhs[index] = {expectation.lhs_initializer};
    cmp_rhs[index] = {expectation.rhs_initializer};
  }}
  for (size_t index = 0; index < lhs_alloc; ++index)
    lhs[index] = (int16_t)(((index % 4) < 2)
                               ? -((int)(index % 59) + 3)
                               : ((int)(index % 59) + 6));
  for (size_t index = 0; index < rhs_alloc; ++index)
    rhs[index] = (int16_t)(((index % 5) == 0)
                               ? -((int)(index % 43) + 4)
                               : ((int)(index % 43) + 9));
  for (size_t index = 0; index < out_alloc; ++index) {{
    acc[index] = {expectation.source_initializer};
    out[index] = {expectation.out_initializer};
  }}

  {expectation.function_name}(cmp_lhs, cmp_rhs, lhs, rhs, acc, out, n,
                              lhs_stride, rhs_stride);

  int32_t expected = acc[0];
  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t active_positive_products = 0;
  size_t active_negative_products = 0;
  size_t inactive_nonzero_products = 0;
  size_t lhs_skipped_nonzero = 0;
  size_t rhs_skipped_nonzero = 0;
  for (size_t index = 0; index < n; ++index) {{
    int active = cmp_lhs[index] < cmp_rhs[index];
    size_t lhs_index = index * lhs_stride;
    size_t rhs_index = index * rhs_stride;
    int32_t product = (int32_t)lhs[lhs_index] * (int32_t)rhs[rhs_index];
    if (active) {{
      ++active_lanes;
      if (product > 0)
        ++active_positive_products;
      if (product < 0)
        ++active_negative_products;
      expected = (int32_t)(expected + product);
    }} else {{
      ++inactive_lanes;
      if (product != 0)
        ++inactive_nonzero_products;
    }}
  }}
  for (size_t index = 0; index < n * lhs_stride; ++index)
    if ((index % lhs_stride) != 0 && lhs[index] != 0)
      ++lhs_skipped_nonzero;
  for (size_t index = 0; index < n * rhs_stride; ++index)
    if ((index % rhs_stride) != 0 && rhs[index] != 0)
      ++rhs_skipped_nonzero;

  if (out[0] != expected) {{
    fprintf(stderr,
            "{expectation.kind} scalar mismatch n=%zu got=%d expected=%d seed=%d lhs_stride=%zu rhs_stride=%zu active=%zu inactive=%zu active_pos=%zu active_neg=%zu inactive_nonzero=%zu lhs_skipped_nonzero=%zu rhs_skipped_nonzero=%zu\\n",
            n, out[0], expected, acc[0], lhs_stride, rhs_stride,
            active_lanes, inactive_lanes, active_positive_products,
            active_negative_products, inactive_nonzero_products,
            lhs_skipped_nonzero, rhs_skipped_nonzero);
    free(cmp_lhs);
    free(cmp_rhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 12;
  }}

  for (size_t index = 1; index < out_alloc; ++index) {{
    if (out[index] != {expectation.out_initializer}) {{
      fprintf(stderr,
              "{expectation.kind} touched non-scalar/tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, out[index], {expectation.out_initializer});
      free(cmp_lhs);
      free(cmp_rhs);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 13;
    }}
  }}

  if (n > 4 && (active_lanes == 0 || inactive_lanes == 0 ||
                active_positive_products == 0 ||
                active_negative_products == 0 ||
                inactive_nonzero_products == 0 ||
                lhs_skipped_nonzero == 0 || rhs_skipped_nonzero == 0 ||
                acc[0] == 0)) {{
    fprintf(stderr,
            "{expectation.kind} coverage missing n=%zu active=%zu inactive=%zu active_pos=%zu active_neg=%zu inactive_nonzero=%zu lhs_skipped_nonzero=%zu rhs_skipped_nonzero=%zu seed=%d\\n",
            n, active_lanes, inactive_lanes, active_positive_products,
            active_negative_products, inactive_nonzero_products,
            lhs_skipped_nonzero, rhs_skipped_nonzero, acc[0]);
    free(cmp_lhs);
    free(cmp_rhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 14;
  }}

  free(cmp_lhs);
  free(cmp_rhs);
  free(lhs);
  free(rhs);
  free(acc);
  free(out);
  printf("{expectation.kind} case n=%zu ok compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=2,3 skipped_source_elements_ignored scalar_output tail_preserved\\n", n);
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
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} lhs_stride=2 rhs_stride=3\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} lhs_stride=2 rhs_stride=3\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_strided_input_widening_dot_reduce_add:
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
  size_t lhs_alloc = n * lhs_stride + 8;
  size_t rhs_alloc = n * rhs_stride + 8;
  size_t out_alloc = n + 5;
  if (out_alloc == 5 && n == 0)
    out_alloc = 6;
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * lhs_alloc);
  int16_t *rhs = (int16_t *)malloc(sizeof(int16_t) * rhs_alloc);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * out_alloc);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * out_alloc);
  if (!lhs || !rhs || !acc || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < lhs_alloc; ++index)
    lhs[index] = {expectation.lhs_initializer};
  for (size_t index = 0; index < rhs_alloc; ++index)
    rhs[index] = {expectation.rhs_initializer};
  for (size_t index = 0; index < out_alloc; ++index) {{
    acc[index] = {expectation.source_initializer};
    out[index] = {expectation.out_initializer};
  }}

  {expectation.function_name}(lhs, rhs, acc, out, n, lhs_stride, rhs_stride);

  int32_t expected = acc[0];
  size_t positive_products = 0;
  size_t negative_products = 0;
  size_t lhs_skipped_nonzero = 0;
  size_t rhs_skipped_nonzero = 0;
  for (size_t index = 0; index < n; ++index) {{
    size_t lhs_index = index * lhs_stride;
    size_t rhs_index = index * rhs_stride;
    int32_t product = (int32_t)lhs[lhs_index] * (int32_t)rhs[rhs_index];
    if (product > 0)
      ++positive_products;
    if (product < 0)
      ++negative_products;
    expected = (int32_t)(expected + product);
  }}
  for (size_t index = 0; index < n * lhs_stride; ++index)
    if ((index % lhs_stride) != 0 && lhs[index] != 0)
      ++lhs_skipped_nonzero;
  for (size_t index = 0; index < n * rhs_stride; ++index)
    if ((index % rhs_stride) != 0 && rhs[index] != 0)
      ++rhs_skipped_nonzero;

  if (out[0] != expected) {{
    fprintf(stderr,
            "{expectation.kind} scalar mismatch n=%zu got=%d expected=%d seed=%d lhs_stride=%zu rhs_stride=%zu positive_products=%zu negative_products=%zu lhs_skipped_nonzero=%zu rhs_skipped_nonzero=%zu\\n",
            n, out[0], expected, acc[0], lhs_stride, rhs_stride,
            positive_products, negative_products, lhs_skipped_nonzero,
            rhs_skipped_nonzero);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 12;
  }}

  for (size_t index = 1; index < out_alloc; ++index) {{
    if (out[index] != {expectation.out_initializer}) {{
      fprintf(stderr,
              "{expectation.kind} touched non-scalar/tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, out[index], {expectation.out_initializer});
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 13;
    }}
  }}

  if (n > 3 && (positive_products == 0 || negative_products == 0 ||
                lhs_skipped_nonzero == 0 || rhs_skipped_nonzero == 0 ||
                acc[0] == 0)) {{
    fprintf(stderr,
            "{expectation.kind} coverage missing n=%zu positive_products=%zu negative_products=%zu lhs_skipped_nonzero=%zu rhs_skipped_nonzero=%zu seed=%d\\n",
            n, positive_products, negative_products, lhs_skipped_nonzero,
            rhs_skipped_nonzero, acc[0]);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 14;
  }}

  free(lhs);
  free(rhs);
  free(acc);
  free(out);
  printf("{expectation.kind} case n=%zu ok strided_signed_horizontal_dot seed_added source_strides=2,3 skipped_source_elements_ignored scalar_output tail_preserved\\n", n);
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
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} lhs_stride=2 rhs_stride=3\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} lhs_stride=2 rhs_stride=3\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_computed_masked_widening_dot_reduce_add:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n + 5;
  if (alloc_n == 5 && n == 0)
    alloc_n = 6;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int16_t *rhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!cmp_lhs || !cmp_rhs || !lhs || !rhs || !acc || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(cmp_lhs);
    free(cmp_rhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs[index] = {expectation.lhs_initializer};
    cmp_rhs[index] = {expectation.rhs_initializer};
    lhs[index] = (int16_t)(((index % 4) < 2)
                               ? -((int)(index % 53) + 2)
                               : ((int)(index % 53) + 5));
    rhs[index] = (int16_t)(((index % 5) == 0)
                               ? -((int)(index % 37) + 3)
                               : ((int)(index % 37) + 7));
    acc[index] = {expectation.source_initializer};
    out[index] = {expectation.out_initializer};
  }}

  {expectation.function_name}(cmp_lhs, cmp_rhs, lhs, rhs, acc, out, n);

  int32_t expected = acc[0];
  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t active_positive_products = 0;
  size_t active_negative_products = 0;
  size_t inactive_nonzero_products = 0;
  for (size_t index = 0; index < n; ++index) {{
    int active = cmp_lhs[index] < cmp_rhs[index];
    int32_t product = (int32_t)lhs[index] * (int32_t)rhs[index];
    if (active) {{
      ++active_lanes;
      if (product > 0)
        ++active_positive_products;
      if (product < 0)
        ++active_negative_products;
      expected = (int32_t)(expected + product);
    }} else {{
      ++inactive_lanes;
      if (product != 0)
        ++inactive_nonzero_products;
    }}
  }}

  if (out[0] != expected) {{
    fprintf(stderr,
            "{expectation.kind} scalar mismatch n=%zu got=%d expected=%d seed=%d active=%zu inactive=%zu active_pos=%zu active_neg=%zu inactive_nonzero=%zu\\n",
            n, out[0], expected, acc[0], active_lanes, inactive_lanes,
            active_positive_products, active_negative_products,
            inactive_nonzero_products);
    free(cmp_lhs);
    free(cmp_rhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 12;
  }}

  for (size_t index = 1; index < alloc_n; ++index) {{
    if (out[index] != {expectation.out_initializer}) {{
      fprintf(stderr,
              "{expectation.kind} touched non-scalar/tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, out[index], {expectation.out_initializer});
      free(cmp_lhs);
      free(cmp_rhs);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 13;
    }}
  }}

  if (n > 4 && (active_lanes == 0 || inactive_lanes == 0 ||
                active_positive_products == 0 ||
                active_negative_products == 0 ||
                inactive_nonzero_products == 0 || acc[0] == 0)) {{
    fprintf(stderr,
            "{expectation.kind} coverage missing n=%zu active=%zu inactive=%zu active_pos=%zu active_neg=%zu inactive_nonzero=%zu seed=%d\\n",
            n, active_lanes, inactive_lanes, active_positive_products,
            active_negative_products, inactive_nonzero_products, acc[0]);
    free(cmp_lhs);
    free(cmp_rhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 14;
  }}

  free(cmp_lhs);
  free(cmp_rhs);
  free(lhs);
  free(rhs);
  free(acc);
  free(out);
  printf("{expectation.kind} case n=%zu ok compare_masked_signed_horizontal_dot seed_added inactive_lanes_skipped scalar_output tail_preserved\\n", n);
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
    if expectation.is_computed_masked_macc_add:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n + 8;
  if (alloc_n == 8 && n == 0)
    alloc_n = 9;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!cmp_lhs || !cmp_rhs || !lhs || !rhs || !acc || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(cmp_lhs);
    free(cmp_rhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs[index] = {expectation.lhs_initializer};
    cmp_rhs[index] = {expectation.rhs_initializer};
    lhs[index] = {expectation.true_value_initializer};
    rhs[index] = {expectation.false_value_initializer};
    acc[index] = {expectation.source_initializer};
    out[index] = {expectation.out_initializer};
  }}

  {expectation.function_name}(cmp_lhs, cmp_rhs, lhs, rhs, acc, out, n);

  size_t active_lanes = 0;
  size_t inactive_lanes = 0;
  size_t inactive_acc_preserved = 0;
  size_t add_only_distinguishing = 0;
  size_t mul_only_distinguishing = 0;
  size_t signed_product_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int predicate = {expectation.compare_predicate_c_expression("cmp_lhs[index]", "cmp_rhs[index]")};
    int32_t product = (int32_t)lhs[index] * (int32_t)rhs[index];
    int32_t expected = {expectation.expected_expression};
    int32_t add_only = (int32_t)(acc[index] + lhs[index] + rhs[index]);
    int32_t mul_only = product;
    if (predicate)
      ++active_lanes;
    else
      ++inactive_lanes;
    if (!predicate && out[index] == acc[index])
      ++inactive_acc_preserved;
    if (predicate && expected != add_only)
      ++add_only_distinguishing;
    if (predicate && expected != mul_only)
      ++mul_only_distinguishing;
    if (predicate && product != 0 && acc[index] != 0)
      ++signed_product_lanes;
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d cmp_lhs=%d cmp_rhs=%d lhs=%d rhs=%d acc=%d predicate=%d product=%d\\n",
              n, index, out[index], expected, cmp_lhs[index], cmp_rhs[index],
              lhs[index], rhs[index], acc[index], predicate, product);
      free(cmp_lhs);
      free(cmp_rhs);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 12;
    }}
  }}

  for (size_t index = n; index < alloc_n; ++index) {{
    if (out[index] != {expectation.out_initializer}) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, out[index], {expectation.out_initializer});
      free(cmp_lhs);
      free(cmp_rhs);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 13;
    }}
  }}

  if (n > 3 && (active_lanes == 0 || inactive_lanes == 0 ||
                inactive_acc_preserved == 0 ||
                add_only_distinguishing == 0 ||
                mul_only_distinguishing == 0 ||
                signed_product_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} coverage missing n=%zu active=%zu inactive=%zu inactive_acc_preserved=%zu add_only_distinguishing=%zu mul_only_distinguishing=%zu signed_product_lanes=%zu\\n",
            n, active_lanes, inactive_lanes, inactive_acc_preserved,
            add_only_distinguishing, mul_only_distinguishing,
            signed_product_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 14;
  }}

  free(cmp_lhs);
  free(cmp_rhs);
  free(lhs);
  free(rhs);
  free(acc);
  free(out);
  printf("{expectation.kind} case n=%zu ok computed_mask macc active_lanes=%zu inactive_lanes=%zu inactive_acc_preserved=%zu add_only_distinguishing=%zu mul_only_distinguishing=%zu tail_preserved\\n",
         n, active_lanes, inactive_lanes, inactive_acc_preserved,
         add_only_distinguishing, mul_only_distinguishing);
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
    if expectation.is_macc_add:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n + 5;
  if (alloc_n == 5 && n == 0)
    alloc_n = 6;
  int32_t *lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!lhs || !rhs || !acc || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    lhs[index] = {expectation.lhs_initializer};
    rhs[index] = {expectation.rhs_initializer};
    acc[index] = {expectation.source_initializer};
    out[index] = {expectation.out_initializer};
  }}

  {expectation.function_name}(lhs, rhs, acc, out, n);

  size_t positive_products = 0;
  size_t negative_products = 0;
  size_t positive_lhs = 0;
  size_t negative_lhs = 0;
  size_t positive_rhs = 0;
  size_t negative_rhs = 0;
  size_t positive_accumulators = 0;
  size_t negative_accumulators = 0;
  size_t nonzero_accumulators = 0;
  for (size_t index = 0; index < n; ++index) {{
    int32_t product = (int32_t)lhs[index] * (int32_t)rhs[index];
    int32_t expected = {expectation.expected_expression};
    if (product > 0)
      ++positive_products;
    if (product < 0)
      ++negative_products;
    if (lhs[index] > 0)
      ++positive_lhs;
    if (lhs[index] < 0)
      ++negative_lhs;
    if (rhs[index] > 0)
      ++positive_rhs;
    if (rhs[index] < 0)
      ++negative_rhs;
    if (acc[index] > 0)
      ++positive_accumulators;
    if (acc[index] < 0)
      ++negative_accumulators;
    if (acc[index] != 0)
      ++nonzero_accumulators;
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d lhs=%d rhs=%d acc=%d product=%d\\n",
              n, index, out[index], expected, lhs[index], rhs[index], acc[index], product);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 12;
    }}
  }}

  for (size_t index = n; index < alloc_n; ++index) {{
    if (out[index] != {expectation.out_initializer}) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, out[index], {expectation.out_initializer});
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 13;
    }}
  }}

  if (n > 3 && (positive_products == 0 || negative_products == 0 ||
                positive_lhs == 0 || negative_lhs == 0 ||
                positive_rhs == 0 || negative_rhs == 0 ||
                positive_accumulators == 0 || negative_accumulators == 0 ||
                nonzero_accumulators == 0)) {{
    fprintf(stderr,
            "{expectation.kind} coverage missing n=%zu positive_products=%zu negative_products=%zu positive_lhs=%zu negative_lhs=%zu positive_rhs=%zu negative_rhs=%zu positive_accumulators=%zu negative_accumulators=%zu nonzero_accumulators=%zu\\n",
            n, positive_products, negative_products, positive_lhs, negative_lhs,
            positive_rhs, negative_rhs, positive_accumulators,
            negative_accumulators, nonzero_accumulators);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 14;
  }}

  free(lhs);
  free(rhs);
  free(acc);
  free(out);
  printf("{expectation.kind} case n=%zu ok explicit_accumulator signed_products tail_preserved\\n", n);
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
    if expectation.is_widening_dot_reduce_add:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n + 5;
  if (alloc_n == 5 && n == 0)
    alloc_n = 6;
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int16_t *rhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!lhs || !rhs || !acc || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    lhs[index] = {expectation.lhs_initializer};
    rhs[index] = {expectation.rhs_initializer};
    acc[index] = {expectation.source_initializer};
    out[index] = {expectation.out_initializer};
  }}

  {expectation.function_name}(lhs, rhs, acc, out, n);

  int32_t expected = acc[0];
  size_t positive_products = 0;
  size_t negative_products = 0;
  for (size_t index = 0; index < n; ++index) {{
    int32_t product = (int32_t)lhs[index] * (int32_t)rhs[index];
    if (product > 0)
      ++positive_products;
    if (product < 0)
      ++negative_products;
    expected = (int32_t)(expected + product);
  }}

  if (out[0] != expected) {{
    fprintf(stderr,
            "{expectation.kind} scalar mismatch n=%zu got=%d expected=%d seed=%d positive_products=%zu negative_products=%zu\\n",
            n, out[0], expected, acc[0], positive_products, negative_products);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 12;
  }}

  for (size_t index = 1; index < alloc_n; ++index) {{
    if (out[index] != {expectation.out_initializer}) {{
      fprintf(stderr,
              "{expectation.kind} touched non-scalar/tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, out[index], {expectation.out_initializer});
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 13;
    }}
  }}

  if (n > 3 && (positive_products == 0 || negative_products == 0 ||
                acc[0] == 0)) {{
    fprintf(stderr,
            "{expectation.kind} coverage missing n=%zu positive_products=%zu negative_products=%zu seed=%d\\n",
            n, positive_products, negative_products, acc[0]);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 14;
  }}

  free(lhs);
  free(rhs);
  free(acc);
  free(out);
  printf("{expectation.kind} case n=%zu ok signed_horizontal_dot seed_added scalar_output tail_preserved\\n", n);
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
    if expectation.is_widening_macc_add:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n + 5;
  if (alloc_n == 5 && n == 0)
    alloc_n = 6;
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int16_t *rhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!lhs || !rhs || !acc || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    lhs[index] = {expectation.lhs_initializer};
    rhs[index] = {expectation.rhs_initializer};
    acc[index] = {expectation.source_initializer};
    out[index] = {expectation.out_initializer};
  }}

  {expectation.function_name}(lhs, rhs, acc, out, n);

  size_t positive_products = 0;
  size_t negative_products = 0;
  size_t nonzero_accumulators = 0;
  for (size_t index = 0; index < n; ++index) {{
    int32_t product = (int32_t)lhs[index] * (int32_t)rhs[index];
    int32_t expected = {expectation.expected_expression};
    if (product > 0)
      ++positive_products;
    if (product < 0)
      ++negative_products;
    if (acc[index] != 0)
      ++nonzero_accumulators;
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d lhs=%d rhs=%d acc=%d product=%d\\n",
              n, index, out[index], expected, lhs[index], rhs[index], acc[index], product);
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 12;
    }}
  }}

  for (size_t index = n; index < alloc_n; ++index) {{
    if (out[index] != {expectation.out_initializer}) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, out[index], {expectation.out_initializer});
      free(lhs);
      free(rhs);
      free(acc);
      free(out);
      return 13;
    }}
  }}

  if (n > 3 && (positive_products == 0 || negative_products == 0 ||
                nonzero_accumulators == 0)) {{
    fprintf(stderr,
            "{expectation.kind} coverage missing n=%zu positive_products=%zu negative_products=%zu nonzero_accumulators=%zu\\n",
            n, positive_products, negative_products, nonzero_accumulators);
    free(lhs);
    free(rhs);
    free(acc);
    free(out);
    return 14;
  }}

  free(lhs);
  free(rhs);
  free(acc);
  free(out);
  printf("{expectation.kind} case n=%zu ok signed_products accumulation tail_preserved\\n", n);
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
    if expectation.is_runtime_scalar_dual_compare_mask_and_select:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n, int32_t rhs_scalar_a, int32_t rhs_scalar_b) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t out_alloc_n = alloc_n + 8;
  int32_t *cmp_lhs_a = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_lhs_b = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *true_value = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *false_value = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * out_alloc_n);
  if (!cmp_lhs_a || !cmp_lhs_b || !true_value || !false_value || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(cmp_lhs_a);
    free(cmp_lhs_b);
    free(true_value);
    free(false_value);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs_a[index] = {expectation.lhs_initializer};
    cmp_lhs_b[index] = (int32_t)(((index % 6) == 0) ? -50 :
                         ((index % 6) == 1) ? -10 :
                         ((index % 6) == 2) ? -60 :
                         ((index % 6) == 3) ? 100 :
                         ((index % 6) == 4) ? 50 : 120);
    true_value[index] = {expectation.true_value_initializer};
    false_value[index] = {expectation.false_value_initializer};
    out[index] = {expectation.out_initializer};
  }}
  for (size_t index = alloc_n; index < out_alloc_n; ++index)
    out[index] = {OUT_SENTINEL};

  {expectation.function_name}(cmp_lhs_a, rhs_scalar_a, cmp_lhs_b, rhs_scalar_b,
                              true_value, false_value, out, n);

  size_t mask_a_true = 0;
  size_t mask_a_false = 0;
  size_t mask_b_true = 0;
  size_t mask_b_false = 0;
  size_t composed_true = 0;
  size_t composed_false = 0;
  size_t single_mask_only = 0;
  size_t true_payload_lanes = 0;
  size_t false_payload_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int predicate_a = cmp_lhs_a[index] <= rhs_scalar_a;
    int predicate_b = cmp_lhs_b[index] <= rhs_scalar_b;
    int composed = predicate_a && predicate_b;
    if (predicate_a)
      ++mask_a_true;
    else
      ++mask_a_false;
    if (predicate_b)
      ++mask_b_true;
    else
      ++mask_b_false;
    if (composed)
      ++composed_true;
    else
      ++composed_false;
    if (predicate_a != predicate_b)
      ++single_mask_only;

    int32_t expected = {expectation.expected_expression};
    if (expected == true_value[index])
      ++true_payload_lanes;
    if (expected == false_value[index])
      ++false_payload_lanes;
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d cmp_lhs_a=%d rhs_scalar_a=%d cmp_lhs_b=%d rhs_scalar_b=%d true=%d false=%d mask_a=%d mask_b=%d composed=%d\\n",
              n, index, out[index], expected, cmp_lhs_a[index], rhs_scalar_a,
              cmp_lhs_b[index], rhs_scalar_b, true_value[index],
              false_value[index], predicate_a, predicate_b, composed);
      free(cmp_lhs_a);
      free(cmp_lhs_b);
      free(true_value);
      free(false_value);
      free(out);
      return 12;
    }}
  }}

  for (size_t index = n; index < out_alloc_n; ++index) {{
    if (out[index] != {OUT_SENTINEL}) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d rhs_scalar_a=%d rhs_scalar_b=%d\\n",
              n, index, out[index], {OUT_SENTINEL}, rhs_scalar_a,
              rhs_scalar_b);
      free(cmp_lhs_a);
      free(cmp_lhs_b);
      free(true_value);
      free(false_value);
      free(out);
      return 13;
    }}
  }}

  if (n > 1 && (mask_a_true == 0 || mask_a_false == 0 ||
                mask_b_true == 0 || mask_b_false == 0 ||
                composed_true == 0 || composed_false == 0)) {{
    fprintf(stderr,
            "{expectation.kind} mask coverage missing n=%zu rhs_scalar_a=%d rhs_scalar_b=%d mask_a_true=%zu mask_a_false=%zu mask_b_true=%zu mask_b_false=%zu composed_true=%zu composed_false=%zu\\n",
            n, rhs_scalar_a, rhs_scalar_b, mask_a_true, mask_a_false,
            mask_b_true, mask_b_false, composed_true, composed_false);
    free(cmp_lhs_a);
    free(cmp_lhs_b);
    free(true_value);
    free(false_value);
    free(out);
    return 14;
  }}
  if (n > 1 && single_mask_only == 0) {{
    fprintf(stderr,
            "{expectation.kind} mask-and distinction missing n=%zu rhs_scalar_a=%d rhs_scalar_b=%d\\n",
            n, rhs_scalar_a, rhs_scalar_b);
    free(cmp_lhs_a);
    free(cmp_lhs_b);
    free(true_value);
    free(false_value);
    free(out);
    return 15;
  }}
  if (n > 1 && (true_payload_lanes == 0 || false_payload_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} select payload coverage missing n=%zu rhs_scalar_a=%d rhs_scalar_b=%d true_lanes=%zu false_lanes=%zu\\n",
            n, rhs_scalar_a, rhs_scalar_b, true_payload_lanes,
            false_payload_lanes);
    free(cmp_lhs_a);
    free(cmp_lhs_b);
    free(true_value);
    free(false_value);
    free(out);
    return 16;
  }}

  free(cmp_lhs_a);
  free(cmp_lhs_b);
  free(true_value);
  free(false_value);
  free(out);
  printf("{expectation.kind} case n=%zu ok rhs_scalar_a=%d rhs_scalar_b=%d mask_a_true=%zu mask_b_true=%zu composed_true=%zu single_mask_only=%zu tail_preserved\\n",
         n, rhs_scalar_a, rhs_scalar_b, mask_a_true, mask_b_true,
         composed_true, single_mask_only);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const int32_t rhs_scalar_a_values[] = {{{scalar_values_literal}}};
  const int32_t rhs_scalar_b_values[] = {{(int32_t)-37, (int32_t)91}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t scalar_a_count = sizeof(rhs_scalar_a_values) / sizeof(rhs_scalar_a_values[0]);
  const size_t scalar_b_count = sizeof(rhs_scalar_b_values) / sizeof(rhs_scalar_b_values[0]);
  for (size_t scalar_a_index = 0; scalar_a_index < scalar_a_count; ++scalar_a_index) {{
    for (size_t scalar_b_index = 0; scalar_b_index < scalar_b_count; ++scalar_b_index) {{
      for (size_t index = 0; index < count_count; ++index) {{
        int status = run_case(counts[index], rhs_scalar_a_values[scalar_a_index],
                              rhs_scalar_b_values[scalar_b_index]);
        if (status != 0)
          return status;
      }}
    }}
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} rhs_scalar_a_values={scalar_values_summary} rhs_scalar_b_values=-37,91\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} rhs_scalar_a_values={scalar_values_summary} rhs_scalar_b_values=-37,91\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_runtime_scalar_compare_select:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n, int32_t rhs_scalar) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t out_alloc_n = alloc_n + 8;
  int32_t *lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *true_value = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *false_value = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * out_alloc_n);
  if (!lhs || !true_value || !false_value || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(lhs);
    free(true_value);
    free(false_value);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    lhs[index] = {expectation.lhs_initializer};
    true_value[index] = {expectation.true_value_initializer};
    false_value[index] = {expectation.false_value_initializer};
    out[index] = {expectation.out_initializer};
  }}
  for (size_t index = alloc_n; index < out_alloc_n; ++index)
    out[index] = {OUT_SENTINEL};

  {expectation.function_name}(lhs, rhs_scalar, true_value, false_value, out, n);

  size_t true_lanes = 0;
  size_t false_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int predicate = {expectation.compare_predicate_c_expression("lhs[index]", "rhs_scalar")};
    if (predicate)
      ++true_lanes;
    else
      ++false_lanes;

    int32_t expected = {expectation.expected_expression};
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d lhs=%d rhs_scalar=%d true=%d false=%d predicate=%d\\n",
              n, index, out[index], expected, lhs[index], rhs_scalar,
              true_value[index], false_value[index], predicate);
      free(lhs);
      free(true_value);
      free(false_value);
      free(out);
      return 12;
    }}
  }}

  for (size_t index = n; index < out_alloc_n; ++index) {{
    if (out[index] != {OUT_SENTINEL}) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d rhs_scalar=%d\\n",
              n, index, out[index], {OUT_SENTINEL}, rhs_scalar);
      free(lhs);
      free(true_value);
      free(false_value);
      free(out);
      return 13;
    }}
  }}

  if (n > 1 && (true_lanes == 0 || false_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} select coverage missing n=%zu rhs_scalar=%d true_lanes=%zu false_lanes=%zu\\n",
            n, rhs_scalar, true_lanes, false_lanes);
    free(lhs);
    free(true_value);
    free(false_value);
    free(out);
    return 14;
  }}

  free(lhs);
  free(true_value);
  free(false_value);
  free(out);
  printf("{expectation.kind} case n=%zu ok rhs_scalar=%d select_true_lanes=%zu select_false_lanes=%zu tail_preserved\\n",
         n, rhs_scalar, true_lanes, false_lanes);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{{counts}}};
  const int32_t rhs_scalar_values[] = {{{scalar_values_literal}}};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t scalar_count = sizeof(rhs_scalar_values) / sizeof(rhs_scalar_values[0]);
  for (size_t scalar_index = 0; scalar_index < scalar_count; ++scalar_index) {{
    for (size_t index = 0; index < count_count; ++index) {{
      int status = run_case(counts[index], rhs_scalar_values[scalar_index]);
      if (status != 0)
        return status;
    }}
  }}
  printf("{expectation.pass_marker} counts={','.join(str(c) for c in runtime_counts)} rhs_scalars={scalar_values_summary}\\n");
  printf("PASS op={expectation.kind} counts={','.join(str(c) for c in runtime_counts)} rhs_scalars={scalar_values_summary}\\n");
  return 0;
}}
""".lstrip()
    if expectation.is_computed_mask_select:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t out_alloc_n = alloc_n + 8;
  int32_t *cmp_lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *cmp_rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *true_value = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *false_value = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * out_alloc_n);
  if (!cmp_lhs || !cmp_rhs || !true_value || !false_value || !out) {{
    fprintf(stderr, "allocation failed for n=%zu\\n", n);
    free(cmp_lhs);
    free(cmp_rhs);
    free(true_value);
    free(false_value);
    free(out);
    return 11;
  }}

  for (size_t index = 0; index < alloc_n; ++index) {{
    cmp_lhs[index] = {expectation.lhs_initializer};
    cmp_rhs[index] = {expectation.rhs_initializer};
    true_value[index] = {expectation.true_value_initializer};
    false_value[index] = {expectation.false_value_initializer};
    out[index] = {expectation.out_initializer};
  }}
  for (size_t index = alloc_n; index < out_alloc_n; ++index)
    out[index] = {OUT_SENTINEL};

  {expectation.function_name}(cmp_lhs, cmp_rhs, true_value, false_value, out, n);

  size_t true_lanes = 0;
  size_t false_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int predicate = {expectation.compare_predicate_c_expression("cmp_lhs[index]", "cmp_rhs[index]")};
    if (predicate)
      ++true_lanes;
    else
      ++false_lanes;

    int32_t expected = {expectation.expected_expression};
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d cmp_lhs=%d cmp_rhs=%d true=%d false=%d predicate=%d\\n",
              n, index, out[index], expected, cmp_lhs[index], cmp_rhs[index],
              true_value[index], false_value[index], predicate);
      free(cmp_lhs);
      free(cmp_rhs);
      free(true_value);
      free(false_value);
      free(out);
      return 12;
    }}
  }}

  for (size_t index = n; index < out_alloc_n; ++index) {{
    if (out[index] != {OUT_SENTINEL}) {{
      fprintf(stderr,
              "{expectation.kind} touched tail sentinel n=%zu raw_index=%zu got=%d sentinel=%d\\n",
              n, index, out[index], {OUT_SENTINEL});
      free(cmp_lhs);
      free(cmp_rhs);
      free(true_value);
      free(false_value);
      free(out);
      return 13;
    }}
  }}

  if (n > 1 && (true_lanes == 0 || false_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} select coverage missing n=%zu true_lanes=%zu false_lanes=%zu\\n",
            n, true_lanes, false_lanes);
    free(cmp_lhs);
    free(cmp_rhs);
    free(true_value);
    free(false_value);
    free(out);
    return 14;
  }}

  free(cmp_lhs);
  free(cmp_rhs);
  free(true_value);
  free(false_value);
  free(out);
  printf("{expectation.kind} case n=%zu ok select_true_lanes=%zu select_false_lanes=%zu tail_preserved\\n",
         n, true_lanes, false_lanes);
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
    if expectation.is_cmp_select:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  size_t alloc_n = n == 0 ? 1 : n;
  {expectation.element_c_type} *lhs = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
  {expectation.element_c_type} *rhs = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
  {expectation.element_c_type} *out = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
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

  size_t predicate_true_lanes = 0;
  size_t predicate_false_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int predicate = {expectation.compare_predicate_c_expression("lhs[index]", "rhs[index]")};
    if (predicate)
      ++predicate_true_lanes;
    else
      ++predicate_false_lanes;

    int32_t expected = {expectation.expected_expression};
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d lhs=%d rhs=%d predicate=%d\\n",
              n, index, out[index], expected, lhs[index], rhs[index], predicate);
      free(lhs);
      free(rhs);
      free(out);
      return 12;
    }}
  }}

  if (n > 1 && (predicate_true_lanes == 0 || predicate_false_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} predicate coverage missing n=%zu true_lanes=%zu false_lanes=%zu\\n",
            n, predicate_true_lanes, predicate_false_lanes);
    free(lhs);
    free(rhs);
    free(out);
    return 13;
  }}

  free(lhs);
  free(rhs);
  free(out);
  printf("{expectation.kind} case n=%zu ok predicate_true_lanes=%zu predicate_false_lanes=%zu\\n",
         n, predicate_true_lanes, predicate_false_lanes);
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
    if expectation.is_masked_elementwise:
        return f"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "{header_file_name}"

static int run_case(size_t n) {{
  /* expected: {expectation.expected_expression} */
  size_t alloc_n = n == 0 ? 1 : n;
  {expectation.element_c_type} *lhs = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
  {expectation.element_c_type} *rhs = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
  {expectation.element_c_type} *out = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
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

  size_t mask_true_lanes = 0;
  size_t mask_false_lanes = 0;
  size_t passthrough_preserved_lanes = 0;
  for (size_t index = 0; index < n; ++index) {{
    int mask = lhs[index] == rhs[index];
    if (mask)
      ++mask_true_lanes;
    else
      ++mask_false_lanes;

    int32_t expected = {expectation.expected_expression};
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%d expected=%d lhs=%d rhs=%d mask=%d\\n",
              n, index, out[index], expected, lhs[index], rhs[index], mask);
      free(lhs);
      free(rhs);
      free(out);
      return 12;
    }}

    if (!mask) {{
      if (out[index] != lhs[index]) {{
        fprintf(stderr,
                "{expectation.kind} inactive lane did not preserve passthrough n=%zu index=%zu got=%d passthrough=%d rhs=%d\\n",
                n, index, out[index], lhs[index], rhs[index]);
        free(lhs);
        free(rhs);
        free(out);
        return 13;
      }}
      ++passthrough_preserved_lanes;
    }}
  }}

  if (n > 1 && (mask_true_lanes == 0 || mask_false_lanes == 0)) {{
    fprintf(stderr,
            "{expectation.kind} mask coverage missing n=%zu true_lanes=%zu false_lanes=%zu\\n",
            n, mask_true_lanes, mask_false_lanes);
    free(lhs);
    free(rhs);
    free(out);
    return 14;
  }}
  if (mask_false_lanes != passthrough_preserved_lanes) {{
    fprintf(stderr,
            "{expectation.kind} inactive lane passthrough coverage mismatch n=%zu false_lanes=%zu preserved_lanes=%zu\\n",
            n, mask_false_lanes, passthrough_preserved_lanes);
    free(lhs);
    free(rhs);
    free(out);
    return 15;
  }}

  free(lhs);
  free(rhs);
  free(out);
  printf("{expectation.kind} case n=%zu ok mask_true_lanes=%zu mask_false_lanes=%zu passthrough_preserved_lanes=%zu\\n",
         n, mask_true_lanes, mask_false_lanes, passthrough_preserved_lanes);
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
  {expectation.element_c_type} *lhs = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
  {expectation.element_c_type} *rhs = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
  {expectation.element_c_type} *out = ({expectation.element_c_type} *)malloc(sizeof({expectation.element_c_type}) * alloc_n);
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
    {expectation.element_c_type} expected = {expectation.expected_expression};
    if (out[index] != expected) {{
      fprintf(stderr,
              "{expectation.kind} mismatch n=%zu index=%zu got=%lld expected=%lld lhs=%lld rhs=%lld\\n",
              n, index, (long long)out[index], (long long)expected,
              (long long)lhs[index], (long long)rhs[index]);
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
    direct_pre_realized_route_entry: bool,
) -> dict[str, Any]:
    materialized_path = bundle_dir.parent / "materialized_selected_body.mlir"
    materialize_command = [tcrv_opt, str(expectation.input_path)]
    if expectation.is_pre_realized and not direct_pre_realized_route_entry:
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
        if direct_pre_realized_route_entry:
            result["materializer"] = "rvv-route-entry-selected-body-realization"
            result["route_entry_realization"] = True
            result["realization_boundary"] = (
                "RVV production emission-plan route-entry consumed the "
                "pre-realized typed tcrv_rvv body before provider route "
                "construction"
            )
        else:
            result["materializer"] = "tcrv-materialize-selected-lowering-boundaries"
            result["route_entry_realization"] = False
            result["realization_boundary"] = (
                "public selected lowering-boundary materialization consumed the "
                "pre-realized typed tcrv_rvv body before provider route "
                "construction"
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
    if args.direct_pre_realized_route_entry:
        if not args.pre_realized_selected_body:
            raise EvidenceError(
                "--direct-pre-realized-route-entry requires "
                "--pre-realized-selected-body"
            )
        unsupported_direct = [
            expectation.kind
            for expectation in expectations
            if not expectation.supports_direct_pre_realized_route_entry
        ]
        if unsupported_direct:
            raise EvidenceError(
                "--direct-pre-realized-route-entry is bounded to "
                "pre-realized cmp_select/cmp_select_sle and "
                f"strided_load_unit_store fixtures; got {unsupported_direct}"
            )
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


def validate_rhs_scalar_values(rhs_scalar_values: list[int]) -> None:
    if not rhs_scalar_values:
        raise EvidenceError(
            "runtime scalar-broadcast evidence requires at least one RHS scalar value"
        )
    if len(set(rhs_scalar_values)) != len(rhs_scalar_values):
        raise EvidenceError(
            "runtime scalar-broadcast evidence requires distinct RHS scalar "
            f"values: {rhs_scalar_values}"
        )
    min_i32 = -(2**31)
    max_i32 = 2**31 - 1
    if any(value < min_i32 or value > max_i32 for value in rhs_scalar_values):
        raise EvidenceError(
            "runtime scalar-broadcast evidence requires int32 RHS scalar "
            f"values: {rhs_scalar_values}"
        )


def validate_strided_load_byte_strides(stride_bytes_values: list[int]) -> None:
    if not stride_bytes_values:
        raise EvidenceError(
            "runtime byte-strided memory evidence requires at least one byte "
            "stride"
        )
    if len(set(stride_bytes_values)) != len(stride_bytes_values):
        raise EvidenceError(
            "runtime byte-strided memory evidence requires distinct byte "
            f"strides: {stride_bytes_values}"
        )
    for stride in stride_bytes_values:
        if stride <= 0:
            raise EvidenceError(
                "runtime byte-strided memory evidence requires positive byte "
                f"strides: {stride_bytes_values}"
            )
        if stride % 4 != 0:
            raise EvidenceError(
                "runtime byte-strided memory evidence requires byte strides "
                "to be multiples of sizeof(int32_t): "
                f"{stride_bytes_values}"
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
    rhs_scalar_values: list[int],
    stride_bytes_values: list[int],
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
    if (
        expectation.is_scalar_broadcast_elementwise
        or expectation.is_runtime_scalar_splat_store
        or expectation.is_runtime_scalar_compare_select
        or expectation.is_runtime_scalar_dual_compare_mask_and_select
        or expectation.is_runtime_scalar_computed_mask_store
        or expectation.is_runtime_scalar_computed_mask_load_store
        or expectation.is_runtime_scalar_computed_masked_macc_add
    ):
        evidence["rhs_scalar_values"] = rhs_scalar_values
    if (
        expectation.is_strided_load_unit_store
        or expectation.is_unit_load_strided_store
        or expectation.is_computed_masked_strided_store
        or expectation.is_computed_masked_strided_load_unit_store
    ):
        evidence["stride_bytes_values"] = stride_bytes_values

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
            args.direct_pre_realized_route_entry,
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
            harness_source(
                bundle_checks["header_file"],
                runtime_counts,
                expectation,
                rhs_scalar_values,
                stride_bytes_values,
            ),
            encoding="utf-8",
        )
        evidence["harness"] = {
            "path": str(harness_path),
            "sha256": sha256_file(harness_path),
            "pass_marker": expectation.pass_marker,
            "boundary": "external C ABI consumer of generated header and object only",
        }
        if (
            expectation.is_scalar_broadcast_elementwise
            or expectation.is_runtime_scalar_splat_store
            or expectation.is_runtime_scalar_compare_select
            or expectation.is_runtime_scalar_dual_compare_mask_and_select
            or expectation.is_runtime_scalar_computed_mask_store
            or expectation.is_runtime_scalar_computed_mask_load_store
            or expectation.is_runtime_scalar_computed_masked_macc_add
        ):
            evidence["harness"]["rhs_scalar_values"] = rhs_scalar_values
            evidence["harness"]["rhs_scalar_coverage_contract"] = (
                "runtime scalar operand cases must execute the same generated "
                "artifact with explicit runtime RHS scalar values"
            )
        if (
            expectation.is_strided_load_unit_store
            or expectation.is_unit_load_strided_store
            or expectation.is_computed_masked_strided_store
            or expectation.is_computed_masked_strided_load_unit_store
        ):
            evidence["harness"]["stride_bytes_values"] = stride_bytes_values
            if expectation.is_unit_load_strided_store:
                evidence["harness"]["byte_stride_coverage_contract"] = (
                    "runtime unit_load_strided_store cases must execute the "
                    "same generated artifact with explicit runtime destination "
                    "byte strides"
                )
            elif expectation.is_computed_masked_strided_store:
                evidence["harness"]["byte_stride_coverage_contract"] = (
                    "runtime computed_masked_strided_store cases must execute "
                    "the same generated artifact with explicit runtime "
                    "destination byte strides"
                )
            elif expectation.is_computed_masked_strided_load_unit_store:
                evidence["harness"]["byte_stride_coverage_contract"] = (
                    "runtime computed_masked_strided_load_unit_store cases "
                    "must execute the same generated artifact with explicit "
                    "runtime source byte strides"
                )
            else:
                evidence["harness"]["byte_stride_coverage_contract"] = (
                    "runtime strided_load_unit_store cases must execute the "
                    "same generated artifact with explicit runtime byte strides"
                )
        if expectation.is_cmp_select:
            evidence["harness"][
                "predicate_coverage_contract"
            ] = "multi-lane cmp_select cases require predicate true and false lanes"
        if expectation.is_computed_mask_select:
            evidence["harness"]["select_coverage_contract"] = (
                "multi-lane computed_mask_select cases require compare-produced "
                "true and false select lanes"
            )
            evidence["harness"]["tail_lane_contract"] = (
                "runtime n must be honored and tail sentinel lanes must be "
                "preserved"
            )
        if expectation.is_runtime_scalar_compare_select:
            evidence["harness"]["select_coverage_contract"] = (
                "multi-lane runtime_scalar_cmp_select cases require runtime "
                "scalar threshold true and false select lanes"
            )
            evidence["harness"]["tail_lane_contract"] = (
                "runtime n/AVL must be honored and tail sentinel lanes must be "
                "preserved"
            )
        if expectation.is_runtime_scalar_dual_compare_mask_and_select:
            evidence["harness"]["mask_composition_coverage_contract"] = (
                "multi-lane runtime_scalar_dual_cmp_mask_and_select cases "
                "require both input masks, composed mask-and true/false lanes, "
                "and single-mask-only lanes that prove intersection semantics"
            )
            evidence["harness"]["tail_lane_contract"] = (
                "runtime n/AVL must be honored and tail sentinel lanes must be "
                "preserved"
            )
        if expectation.is_runtime_scalar_computed_mask_store:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane runtime_scalar_cmp_masked_store cases require "
                "runtime scalar threshold active and inactive mask lanes"
            )
            evidence["harness"]["inactive_lane_contract"] = (
                "compare-false lanes and tail lanes must preserve old "
                "destination sentinel values"
            )
            evidence["harness"]["payload_contract"] = (
                "compare-true lanes store the source payload vector, not the "
                "lhs compare vector or a select fallback"
            )
        if expectation.is_runtime_scalar_computed_mask_load_store:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane runtime_scalar_cmp_masked_load_store cases require "
                "runtime scalar threshold active and inactive masked-load lanes"
            )
            evidence["harness"]["inactive_lane_contract"] = (
                "compare-false lanes and tail lanes must preserve old "
                "destination passthrough values"
            )
            evidence["harness"]["payload_contract"] = (
                "compare-true lanes load the source payload through masked_load, "
                "not the lhs compare vector, masked_store, or select fallback"
            )
        if expectation.is_masked_elementwise:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane masked elementwise cases require true and false mask lanes"
            )
            evidence["harness"]["inactive_lane_contract"] = (
                "masked-off lanes must preserve the explicit passthrough vector"
            )
        if expectation.is_macc_add:
            evidence["harness"]["accumulator_contract"] = (
                "macc_add cases use an explicit accumulator input buffer and "
                "a separate output destination"
            )
            evidence["harness"]["tail_lane_contract"] = (
                "runtime n controls active lanes and output tail sentinels are "
                "preserved"
            )
        if expectation.is_computed_masked_macc_add:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane computed_masked_macc_add cases require "
                "compare-produced active and inactive mask lanes"
            )
            evidence["harness"]["accumulator_contract"] = (
                "inactive lanes preserve the explicit accumulator input while "
                "active lanes compute accumulator + lhs * rhs"
            )
            evidence["harness"]["macc_distinguishing_contract"] = (
                "active-lane checks distinguish fused multiply-add "
                "accumulation from add-only or multiply-only behavior"
            )
            evidence["harness"]["tail_lane_contract"] = (
                "runtime n controls active lanes and output tail sentinels are "
                "preserved"
            )
        if expectation.is_runtime_scalar_computed_masked_macc_add:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane runtime_scalar_cmp_masked_macc_add cases require "
                "runtime scalar threshold active and inactive mask lanes"
            )
            evidence["harness"]["accumulator_contract"] = (
                "inactive lanes preserve the explicit accumulator input while "
                "active lanes compute accumulator + lhs * rhs"
            )
            evidence["harness"]["macc_distinguishing_contract"] = (
                "active-lane checks distinguish fused multiply-add "
                "accumulation from add-only or multiply-only behavior"
            )
            evidence["harness"]["tail_lane_contract"] = (
                "runtime n/AVL controls active lanes and output tail sentinels "
                "are preserved"
            )
        if expectation.is_masked_unit_load_store or expectation.is_masked_unit_store:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane masked unit-store/load-store cases require active and "
                "inactive mask lanes"
            )
            evidence["harness"]["inactive_lane_contract"] = (
                "masked-off lanes and tail lanes must preserve old destination "
                "sentinel values"
            )
        if expectation.is_computed_masked_unit_load_store:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane computed_masked_unit_load_store cases require "
                "compare-produced active and inactive mask lanes"
            )
            evidence["harness"]["inactive_lane_contract"] = (
                "compare-false lanes and tail lanes must preserve old "
                "destination sentinel values"
            )
        if expectation.is_computed_masked_strided_store:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane computed_masked_strided_store cases require "
                "compare-produced active and inactive mask lanes"
            )
            evidence["harness"]["inactive_lane_contract"] = (
                "compare-false strided lanes must preserve old destination "
                "sentinel values"
            )
            evidence["harness"]["stride_preservation_contract"] = (
                "active writes land at byte offset index * dst_stride_bytes "
                "while compare-false lanes, skipped destination slots, and "
                "tail slots preserve sentinels"
            )
        if expectation.is_computed_masked_strided_load_unit_store:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane computed_masked_strided_load_unit_store cases "
                "require compare-produced active and inactive mask lanes"
            )
            evidence["harness"]["inactive_lane_contract"] = (
                "compare-false lanes must preserve old destination "
                "passthrough values"
            )
            evidence["harness"]["stride_preservation_contract"] = (
                "active lanes load from byte offset index * src_stride_bytes "
                "while compare-false lanes, source gaps, and destination tail "
                "slots preserve sentinels"
            )
        if expectation.is_computed_masked_indexed_gather_load_unit_store:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane computed_masked_indexed_gather_load_unit_store "
                "cases require compare-produced active and inactive mask lanes"
            )
            evidence["harness"]["inactive_lane_contract"] = (
                "compare-false lanes must preserve old destination "
                "passthrough values"
            )
            evidence["harness"]["indexed_gather_contract"] = (
                "active lanes load from source[index[i]] with noncontiguous "
                "permuted indices while source buffers and destination tail "
                "slots preserve sentinels"
            )
        if expectation.is_computed_masked_indexed_scatter_store_unit_load:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane computed_masked_indexed_scatter_store_unit_load "
                "cases require compare-produced active and inactive mask lanes"
            )
            evidence["harness"]["inactive_lane_contract"] = (
                "compare-false indexed destination lanes must preserve old "
                "destination values"
            )
            evidence["harness"]["indexed_scatter_contract"] = (
                "active lanes store source[i] to destination[index[i]] with "
                "unique noncontiguous permuted indices while inactive indexed "
                "lanes, source buffers, and destination tail slots preserve "
                "sentinels"
            )
        if expectation.is_computed_masked_segment2_load_unit_store:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane computed_masked_segment2_load_unit_store cases "
                "require compare-produced active and inactive mask lanes"
            )
            evidence["harness"]["inactive_lane_contract"] = (
                "compare-false lanes must preserve both old destination field "
                "passthrough vectors"
            )
            evidence["harness"]["field_order_contract"] = (
                "active lanes load distinguishable even/odd interleaved "
                "segment fields into out0/out1 while source and tail slots "
                "preserve sentinels"
            )
        if expectation.is_computed_masked_segment2_store_unit_load:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane computed_masked_segment2_store_unit_load cases "
                "require compare-produced active and inactive mask lanes"
            )
            evidence["harness"]["inactive_lane_contract"] = (
                "compare-false lanes must preserve both old interleaved "
                "destination field slots"
            )
            evidence["harness"]["field_order_contract"] = (
                "active lanes store distinguishable field0/field1 payload "
                "vectors into even/odd interleaved destination slots while "
                "source buffers and tail slots preserve sentinels"
            )
        if expectation.is_computed_mask_standalone_reduce:
            evidence["harness"]["mask_coverage_contract"] = (
                f"multi-lane {expectation.kind} cases require "
                "compare-produced active and inactive mask lanes"
            )
            evidence["harness"]["reduction_contract"] = (
                f"active lanes contribute signed i32 source values to the signed "
                f"{expectation.computed_mask_standalone_reduction_kind} reduction "
                "with the i32 scalar seed while inactive lanes use the "
                "operation-specific neutral element"
            )
            evidence["harness"]["scalar_result_contract"] = (
                "only out[0] is written and non-scalar output slots preserve "
                "sentinels"
            )
        if expectation.is_runtime_scalar_computed_mask_standalone_reduce:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane runtime_scalar_cmp_masked_standalone_reduce_add "
                "cases require runtime scalar threshold active and inactive "
                "mask lanes"
            )
            evidence["harness"]["reduction_contract"] = (
                "only active masked lanes contribute signed i32 source payload "
                "values to the scalar add reduction with the i32 scalar seed"
            )
            evidence["harness"]["inactive_lane_contract"] = (
                "inactive lanes must be excluded from the horizontal sum even "
                "when their payload values are nonzero"
            )
            evidence["harness"]["scalar_result_contract"] = (
                "only out[0] is written and non-scalar output slots preserve "
                "sentinels"
            )
        if expectation.is_standalone_reduce:
            evidence["harness"]["reduction_contract"] = (
                f"signed i32 {expectation.standalone_reduction_kind} reduction "
                "combines input lanes with the scalar accumulator seed"
            )
            evidence["harness"]["scalar_result_contract"] = (
                "only out[0] is written and non-scalar output slots preserve "
                "sentinels"
            )
        if expectation.is_computed_masked_widening_dot_reduce_add:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane computed_masked_widening_dot_reduce_add cases "
                "require compare-produced active and inactive mask lanes"
            )
            evidence["harness"]["dot_reduction_contract"] = (
                "active lanes contribute signed i16*i16 widening products to "
                "the i32 scalar seed while inactive nonzero products do not"
            )
            evidence["harness"]["scalar_result_contract"] = (
                "only out[0] is written and tail/non-scalar output slots "
                "preserve sentinels"
            )
        if expectation.is_computed_masked_strided_input_widening_dot_reduce_add:
            evidence["harness"]["mask_coverage_contract"] = (
                "multi-lane computed_masked_strided_input_widening_dot_reduce_add "
                "cases require compare-produced active and inactive mask lanes"
            )
            evidence["harness"]["stride_coverage_contract"] = (
                "runtime lhs_stride=2 and rhs_stride=3 source elements "
                "contribute only at indexed source positions"
            )
            evidence["harness"]["dot_reduction_contract"] = (
                "active lanes contribute signed strided i16*i16 widening "
                "products to the i32 scalar seed while inactive nonzero "
                "products do not"
            )
            evidence["harness"]["scalar_result_contract"] = (
                "only out[0] is written and tail/non-scalar output slots "
                "preserve sentinels"
            )
        if expectation.is_segment2_deinterleave_unit_store:
            evidence["harness"]["field_order_contract"] = (
                "multi-lane segment2_deinterleave_unit_store cases require "
                "even and odd interleaved source fields to be distinguishable"
            )
            evidence["harness"]["tail_lane_contract"] = (
                "field0 and field1 output tail lanes must preserve sentinel "
                "values past runtime n"
            )
        if expectation.is_segment2_interleave_unit_load:
            evidence["harness"]["field_order_contract"] = (
                "multi-lane segment2_interleave_unit_load cases require "
                "field0 and field1 source lanes to interleave into even and "
                "odd destination lanes"
            )
            evidence["harness"]["tail_lane_contract"] = (
                "interleaved destination tail lanes must preserve sentinel "
                "values past runtime 2*n"
            )

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
    rhs_scalar_values = args.rhs_scalar or list(DEFAULT_RHS_SCALAR_VALUES)
    stride_bytes_values = args.stride_bytes or list(DEFAULT_STRIDED_LOAD_BYTE_STRIDES)
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
        if args.direct_pre_realized_route_entry:
            evidence["pre_realized_route_entry_mode"] = "direct"
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
        has_runtime_scalar_operand = any(
            expectation.is_scalar_broadcast_elementwise
            or expectation.is_runtime_scalar_splat_store
            or expectation.is_runtime_scalar_compare_select
            or expectation.is_runtime_scalar_dual_compare_mask_and_select
            or expectation.is_runtime_scalar_computed_mask_store
            or expectation.is_runtime_scalar_computed_mask_load_store
            or expectation.is_runtime_scalar_computed_masked_macc_add
            or expectation.is_runtime_scalar_computed_mask_standalone_reduce
            for expectation in expectations
        )
        has_strided_load_unit_store = any(
            expectation.is_strided_load_unit_store for expectation in expectations
        )
        has_unit_load_strided_store = any(
            expectation.is_unit_load_strided_store for expectation in expectations
        )
        has_computed_masked_strided_store = any(
            expectation.is_computed_masked_strided_store
            for expectation in expectations
        )
        has_computed_masked_strided_load = any(
            expectation.is_computed_masked_strided_load_unit_store
            for expectation in expectations
        )
        if args.rhs_scalar and not has_runtime_scalar_operand:
            raise EvidenceError(
                "--rhs-scalar may only be used when an op kind includes "
                "scalar_broadcast_add/sub/mul, runtime_i32_splat_store, or "
                "runtime_scalar_cmp_select/runtime_scalar_cmp_masked_store/"
                "runtime_scalar_dual_cmp_mask_and_select/"
                "runtime_scalar_cmp_masked_load_store/"
                "runtime_scalar_cmp_masked_macc_add/"
                "runtime_scalar_cmp_masked_standalone_reduce_add"
            )
        if args.stride_bytes and not (
            has_strided_load_unit_store
            or has_unit_load_strided_store
            or has_computed_masked_strided_store
            or has_computed_masked_strided_load
        ):
            raise EvidenceError(
                "--stride-bytes may only be used when an op kind includes "
                "strided_load_unit_store, unit_load_strided_store, or "
                "computed_masked_strided_store/"
                "computed_masked_strided_load_unit_store"
            )
        validate_rhs_scalar_values(rhs_scalar_values)
        if has_runtime_scalar_operand:
            evidence["rhs_scalar_values"] = rhs_scalar_values
        if (
            has_strided_load_unit_store
            or has_unit_load_strided_store
            or has_computed_masked_strided_store
            or has_computed_masked_strided_load
        ):
            validate_strided_load_byte_strides(stride_bytes_values)
            evidence["stride_bytes_values"] = stride_bytes_values
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
                rhs_scalar_values=rhs_scalar_values,
                stride_bytes_values=stride_bytes_values,
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
        validate_rhs_scalar_values([-37, 91])
        expect_self_test_failure(
            "duplicate scalar values", lambda: validate_rhs_scalar_values([-37, -37])
        )
        expect_self_test_failure(
            "out-of-range scalar value",
            lambda: validate_rhs_scalar_values([-(2**31) - 1]),
        )
        validate_strided_load_byte_strides([4, 8, 12])
        expect_self_test_failure(
            "zero byte stride", lambda: validate_strided_load_byte_strides([0])
        )
        expect_self_test_failure(
            "unaligned byte stride", lambda: validate_strided_load_byte_strides([6])
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
            runtime_scalar_coverage = (
                "rhs_scalar_values" not in harness
                or "(int32_t)-37" not in harness
            )
            if expectation.is_runtime_scalar_dual_compare_mask_and_select:
                runtime_scalar_coverage = (
                    "rhs_scalar_a_values" not in harness
                    or "rhs_scalar_b_values" not in harness
                    or "(int32_t)-37" not in harness
                )
            if (
                expectation.is_scalar_broadcast_elementwise
                or expectation.is_runtime_scalar_splat_store
                or expectation.is_runtime_scalar_compare_select
                or expectation.is_runtime_scalar_dual_compare_mask_and_select
                or expectation.is_runtime_scalar_computed_mask_store
                or expectation.is_runtime_scalar_computed_mask_load_store
                or expectation.is_runtime_scalar_computed_masked_macc_add
                or expectation.is_runtime_scalar_computed_mask_standalone_reduce
            ) and runtime_scalar_coverage:
                raise AssertionError(
                    "self-test harness generation lost runtime scalar "
                    "runtime scalar coverage"
                )
            if expectation.is_strided_load_unit_store and (
                "stride_bytes_values" not in harness
                or "byte_strided_load" not in harness
            ):
                raise AssertionError(
                    "self-test harness generation lost runtime byte-stride "
                    "coverage"
                )
            if expectation.is_unit_load_strided_store and (
                "stride_bytes_values" not in harness
                or "byte_strided_store" not in harness
                or "dst_stride_bytes" not in harness
            ):
                raise AssertionError(
                    "self-test harness generation lost runtime destination "
                    "byte-stride coverage"
                )
            if expectation.is_computed_masked_strided_store and (
                "stride_bytes_values" not in harness
                or "byte_strided_store" not in harness
                or "dst_stride_bytes" not in harness
                or "old_slot" not in harness
            ):
                raise AssertionError(
                    "self-test harness generation lost computed-mask runtime "
                    "destination byte-stride coverage"
                )
            if expectation.is_computed_masked_strided_load_unit_store and (
                "stride_bytes_values" not in harness
                or "byte_strided_load" not in harness
                or "src_stride_bytes" not in harness
                or "source_gap_preserved_slots" not in harness
            ):
                raise AssertionError(
                    "self-test harness generation lost computed-mask runtime "
                    "source byte-stride coverage"
                )
            if expectation.is_computed_masked_indexed_gather_load_unit_store and (
                "computed_mask indexed_gather" not in harness
                or "src[indices[index]]" not in harness
                or "inactive_preserved_lanes" not in harness
                or "noncontiguous_index_lanes" not in harness
            ):
                raise AssertionError(
                    "self-test harness generation lost computed-mask indexed "
                    "gather mask, passthrough, or index coverage"
                )
            if expectation.is_computed_masked_indexed_scatter_store_unit_load and (
                "computed_mask indexed_scatter" not in harness
                or "dst[dst_index]" not in harness
                or "src[index]" not in harness
                or "inactive lane did not preserve destination" not in harness
                or "source buffer mutated" not in harness
                or "noncontiguous_index_lanes" not in harness
            ):
                raise AssertionError(
                    "self-test harness generation lost computed-mask indexed "
                    "scatter mask, no-write, or index coverage"
                )
            if expectation.is_computed_masked_segment2_load_unit_store and (
                "computed_mask segment2_load" not in harness
                or "src[2 * index]" not in harness
                or "src[2 * index + 1]" not in harness
                or "inactive lane failed to preserve old fields" not in harness
                or "field_distinguishing_lanes" not in harness
            ):
                raise AssertionError(
                    "self-test harness generation lost computed-mask segment2 "
                    "mask, passthrough, or field-order coverage"
                )
            if expectation.is_computed_masked_segment2_store_unit_load and (
                "computed_mask segment2_store" not in harness
                or "dst[2 * index]" not in harness
                or "dst[2 * index + 1]" not in harness
                or "inactive lane failed to preserve interleaved destination"
                not in harness
                or "source buffer mutated" not in harness
                or "field_distinguishing_lanes" not in harness
            ):
                raise AssertionError(
                    "self-test harness generation lost computed-mask segment2 "
                    "store mask, no-write, or field-order coverage"
                )
            if expectation.is_computed_masked_macc_add and (
                "computed_mask macc" not in harness
                or "active_lanes" not in harness
                or "inactive_acc_preserved" not in harness
                or "add_only_distinguishing" not in harness
                or "mul_only_distinguishing" not in harness
            ):
                raise AssertionError(
                    "self-test harness generation lost computed-mask macc "
                    "mask, accumulator passthrough, or arithmetic coverage"
                )
            if expectation.is_runtime_scalar_computed_masked_macc_add and (
                "runtime_scalar_computed_mask_macc" not in harness
                or "rhs_scalar_values" not in harness
                or "active_lanes" not in harness
                or "inactive_acc_preserved" not in harness
                or "add_only_distinguishing" not in harness
                or "mul_only_distinguishing" not in harness
            ):
                raise AssertionError(
                    "self-test harness generation lost runtime scalar "
                    "computed-mask macc threshold, accumulator passthrough, "
                    "or arithmetic coverage"
                )
            if expectation.is_runtime_scalar_computed_mask_standalone_reduce and (
                "runtime_scalar_computed_mask_standalone_reduce" not in harness
                or "rhs_scalar_values" not in harness
                or "active_lanes" not in harness
                or "inactive_lanes" not in harness
                or "inactive_nonzero" not in harness
                or "acc[0] != seed" not in harness
            ):
                raise AssertionError(
                    "self-test harness generation lost runtime scalar "
                    "computed-mask standalone reduction threshold, inactive "
                    "exclusion, or scalar seed coverage"
                )
            if expectation.is_computed_mask_standalone_reduce and (
                "active_lanes" not in harness
                or "inactive_lanes" not in harness
                or "acc[0] != seed" not in harness
            ):
                raise AssertionError(
                    "self-test harness generation lost computed-mask "
                    "standalone reduction seed and mask coverage"
                )
            if expectation.is_standalone_reduce and (
                "acc[0] != seed" not in harness
                or "tail_preserved" not in harness
                or expectation.standalone_reduction_kind not in harness
            ):
                raise AssertionError(
                    "self-test harness generation lost standalone reduction "
                    "seed, kind, or scalar-tail coverage"
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

        accumulation_expectation = EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "computed_masked_macc_add"
        ]
        stale_accumulation_producer = make_fake_bundle(
            tmp / "stale-accumulation-producer", accumulation_expectation
        )
        index_path = stale_accumulation_producer / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace(
            COMPUTED_MASK_ACCUMULATION_VECTOR_PRODUCER_SOURCE,
            COMPUTED_MASK_ACCUMULATION_RUNTIME_SCALAR_PRODUCER_SOURCE,
            1,
        )
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure(
            "wrong computed-mask accumulation producer source",
            lambda: verify_bundle(
                stale_accumulation_producer, None, accumulation_expectation
            ),
        )

        stale_accumulation_suffix = make_fake_bundle(
            tmp / "stale-accumulation-suffix", accumulation_expectation
        )
        index_path = stale_accumulation_suffix / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace(
            COMPUTED_MASK_ACCUMULATION_VECTOR_MACC_SUFFIX,
            COMPUTED_MASK_ACCUMULATION_STANDALONE_REDUCE_SUFFIX,
            1,
        )
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure(
            "wrong computed-mask accumulation suffix",
            lambda: verify_bundle(
                stale_accumulation_suffix, None, accumulation_expectation
            ),
        )

        runtime_scalar_accumulation_expectation = EXPLICIT_SELECTED_BODY_OP_EXPECTATIONS[
            "runtime_scalar_cmp_masked_macc_add"
        ]
        stale_runtime_scalar_producer = make_fake_bundle(
            tmp / "stale-runtime-scalar-accumulation-producer",
            runtime_scalar_accumulation_expectation,
        )
        index_path = stale_runtime_scalar_producer / INDEX_FILE_NAME
        text = index_path.read_text(encoding="utf-8")
        text = text.replace(
            COMPUTED_MASK_ACCUMULATION_RUNTIME_SCALAR_PRODUCER_SOURCE,
            COMPUTED_MASK_ACCUMULATION_VECTOR_PRODUCER_SOURCE,
            1,
        )
        index_path.write_text(text, encoding="utf-8")
        expect_self_test_failure(
            "wrong runtime-scalar accumulation producer source",
            lambda: verify_bundle(
                stale_runtime_scalar_producer,
                None,
                runtime_scalar_accumulation_expectation,
            ),
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
            "use the pre-realized selected-body add/sub/mul/masked_add/"
            "reduce_add/macc_add/strided_add/"
            "strided_load_unit_store/"
            "unit_load_strided_store/indexed_gather_unit_store/"
            "indexed_scatter_unit_load/"
            "masked_unit_load_store/masked_unit_store/"
            "computed_masked_unit_load_store/"
            "computed_masked_strided_store/"
            "computed_masked_strided_load_unit_store/"
            "computed_masked_indexed_gather_load_unit_store/"
            "computed_masked_indexed_scatter_store_unit_load/"
            "runtime_scalar_cmp_masked_store/"
            "runtime_scalar_cmp_masked_load_store/"
            "computed_masked_macc_add/"
            "segment2_deinterleave_unit_store/"
            "segment2_interleave_unit_load/"
            "lmul_m2_add/widen_i32_to_i64/widen_i16_to_i32/"
            "widening_macc_add/widening_dot_reduce_add/"
            "strided_input_widening_dot_reduce_add/"
            "computed_masked_widening_dot_reduce_add/"
            "computed_masked_strided_input_widening_dot_reduce_add "
            "fixtures and run public selected lowering-boundary "
            "materialization before emission planning; mutually exclusive "
            "with --rhs-broadcast-selected-body and --lmul-m2-selected-body"
        ),
    )
    parser.add_argument(
        "--direct-pre-realized-route-entry",
        action="store_true",
        help=(
            "with --pre-realized-selected-body, skip the public selected "
            "lowering-boundary materializer and require the RVV production "
            "emission-plan route-entry bridge to realize bounded cmp_select/"
            "cmp_select_sle or strided_load_unit_store fixtures before "
            "target bundle export"
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
    parser.add_argument(
        "--rhs-scalar",
        action="append",
        type=int,
        default=[],
        help=(
            "runtime RHS scalar value for scalar_broadcast_add/sub/mul, "
            "runtime_i32_splat_store, runtime_scalar_cmp_select, or "
            "runtime_scalar_dual_cmp_mask_and_select, or "
            "runtime_scalar_cmp_masked_store/"
            "runtime_scalar_cmp_masked_load_store; may be "
            "repeated to prove the same generated artifact consumes multiple "
            "runtime scalar values"
        ),
    )
    parser.add_argument(
        "--stride-bytes",
        action="append",
        type=int,
        default=[],
        help=(
            "runtime byte stride for byte-strided RVV memory slices; may be repeated "
            "to prove the same generated artifact consumes multiple byte "
            "strides"
        ),
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
