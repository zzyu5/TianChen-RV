// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/vector_lane0_to_scalar_i32_extract/plain_vector_scalar_extract/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/i32m1-lane0-to-scalar-i32/i32m1-lane1-to-scalar-i32/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADRELATION
// RUN: sed 's/!weft_rvv.vl -> i32/!weft_rvv.vl -> !weft_rvv.vector<i32, "m1">/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADRESULT

// M-FLAT step 2/6 -- the scalar-lane-extract BRIDGE (dismantling blocker B of
// the loop-scaffold analysis: the dialect carried NO typed scalar-extract op;
// the vwredsum lane0 -> scalar i32 jump lived only as an opaque
// __riscv_vmv_x_s_i32m1_i32 call inside the monolithic emitFlatBlockDot).
// weft_rvv.typed_vector_lane0_to_scalar_extract takes a typed i32 LMUL m1
// vector token whose lane 0 is the scalar output boundary (the standalone-
// reduce / vwredsum output shape) plus the active !weft_rvv.vl token, and
// extracts lane 0 into ONE scalar i32. This is the integer-core -> scalar-fold
// handoff the flat loop body needs: brick 2 (block_computed_scale_dequant)
// hard-requires a SCALAR i32 sumi and ggml's byte-exact `(float)sumi` is a
// scalar cast, so under the zero-opaque + byte-exact constraints this
// vector->scalar jump must become a typed primitive.
//
// This is an emit-consistency ("CORE == emission-plans") lit, NOT a numerical
// oracle: the op lowers BYTE-IDENTICAL at the operation-spelling level to the
// SAME __riscv_vmv_x_s_i32m1_i32 call_opaque the monolithic block-dot emits at
// RVVToEmitCBlockQuantLinear.cpp:5653-5655. Emitting an RVV intrinsic via
// call_opaque is NOT the monolithic hand-written block-dot helper -- it is the
// SAME typed-op-lowers-to-an-intrinsic shape as widening_product -> vwmul and
// standalone_reduce -> vwredsum. The i32 range fully covers the lane0 value, so
// the extract is byte-exact by construction; bit-exact-vs-ggml is
// pending-hardware (ssh rvv), not tested here.
//
// The %reduced input here is a plain weft_rvv.load i32m1 used as a STAND-IN for
// the vwredsum lane0 output (the op's verifier enforces the i32m1 input shape /
// scalar i32 result / vl boundary; a structural "came from a reduce" check is a
// later validator step). The scalar i32 result is left unused -- a typed-op
// result with no consumer still lowers (no DCE in the body walk).

module {
  weft.exec.kernel @rvv_typed_vector_lane0_to_scalar_extract_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_typed_vector_lane0_to_scalar_extract attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %red = weft_rvv.runtime_abi_value {c_name = "vred", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "lane0-extract:reduced", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "lane0-extract:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // The i32m1 reduce-lane0 boundary vector (stand-in: a plain load).
        %reduced = weft_rvv.load %red, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        // The scalar-lane-extract bridge: i32m1 lane0 -> scalar i32.
        %sumi = weft_rvv.typed_vector_lane0_to_scalar_extract %reduced, %vl {kind = "vector_lane0_to_scalar_i32_extract", extract_relation = "i32m1-lane0-to-scalar-i32"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> i32
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_typed_vector_lane0_to_scalar_extract_kernel_rvv_typed_vector_lane0_to_scalar_extract(
// The i32m1 reduce-lane0 boundary vector load.
// CHECK: %[[RED:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// The lane0 extract -> scalar i32: BYTE-IDENTICAL to emitFlatBlockDot:5653-5655.
// The call takes ONLY the i32m1 vector (lane 0 is extracted regardless of vl).
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"(%[[RED]]) : (!emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"int32_t">
// CHECK: return

// The bounded surface is fail-closed on the extract kind + relation facts (I7).
// BADKIND: currently supports only kind "vector_lane0_to_scalar_i32_extract"
// BADRELATION: currently supports only extract_relation "i32m1-lane0-to-scalar-i32"
// Vector->scalar contrast: a vector result (a no-op passthrough) is rejected
// fail-closed -- this is exactly the vector -> scalar jump the bridge exists for.
// BADRESULT: requires a scalar i32 result
