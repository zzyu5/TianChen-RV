// RUN: not weft-opt %s --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=IMPORTEDSCALE

// M-FLAT milestone brick 2/5 -- the LOAD-BEARING wall-2 negative. This proves
// the distinguishing contract of weft_rvv.block_computed_scale_dequant vs
// weft_rvv.dequantize: the new op fail-closed REJECTS an IMPORTED runtime ABI
// scale. Here the computed_scale operand is fed the EXACT shape
// weft_rvv.dequantize hard-requires -- an imported
// weft_rvv.runtime_abi_value with role "dequant-scale-value" and C type
// "float" (RVVDialectWideningOps.cpp DequantizeOp::verify) -- and the verifier
// rejects it because block_computed_scale_dequant requires a COMPUTED f32 SSA
// scale (the brick-1 weft_rvv.block_fp16_scale_product output), not an
// imported ABI value. The imported !weft_rvv.runtime_abi_value type fails the
// f32 check (I7). This is wall-2 DEMONSTRATED, not merely asserted in a
// comment: weft_rvv.dequantize can only consume this imported scale; brick 2
// can only consume the computed one -- the two contracts are disjoint and
// DequantizeOp's contract is untouched.

module {
  weft.exec.kernel @rvv_block_computed_scale_dequant_imported_reject_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_block_computed_scale_dequant_imported_reject attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %sumi = weft_rvv.runtime_abi_value {c_name = "sumi", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "block-computed-scale-dequant-imported-reject:sumi", role = "rhs-scalar-value"} : i32
      // The EXACT imported scale shape weft_rvv.dequantize requires: imported
      // runtime ABI float, role dequant-scale-value.
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "block-computed-scale-dequant-imported-reject:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "block-computed-scale-dequant-imported-reject:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
        %term = weft_rvv.block_computed_scale_dequant %sumi, %scale {kind = "computed_scale_sumi_dequant", dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32"} : i32, !weft_rvv.runtime_abi_value -> f32
      } : !weft_rvv.vl
    }
  }
}

// IMPORTEDSCALE: requires the computed_scale operand to be a COMPUTED f32 SSA value
