// RUN: not tcrv-opt %s --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=IMPORTEDSCALE

// M-FLAT milestone brick 2/5 -- the LOAD-BEARING wall-2 negative. This proves
// the distinguishing contract of tcrv_rvv.block_computed_scale_dequant vs
// tcrv_rvv.dequantize: the new op fail-closed REJECTS an IMPORTED runtime ABI
// scale. Here the computed_scale operand is fed the EXACT shape
// tcrv_rvv.dequantize hard-requires -- an imported
// tcrv_rvv.runtime_abi_value with role "dequant-scale-value" and C type
// "float" (RVVDialectWideningOps.cpp DequantizeOp::verify) -- and the verifier
// rejects it because block_computed_scale_dequant requires a COMPUTED f32 SSA
// scale (the brick-1 tcrv_rvv.block_fp16_scale_product output), not an
// imported ABI value. The imported !tcrv_rvv.runtime_abi_value type fails the
// f32 check (I7). This is wall-2 DEMONSTRATED, not merely asserted in a
// comment: tcrv_rvv.dequantize can only consume this imported scale; brick 2
// can only consume the computed one -- the two contracts are disjoint and
// DequantizeOp's contract is untouched.

module {
  tcrv.exec.kernel @rvv_block_computed_scale_dequant_imported_reject_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_block_computed_scale_dequant_imported_reject attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %sumi = tcrv_rvv.runtime_abi_value {c_name = "sumi", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "block-computed-scale-dequant-imported-reject:sumi", role = "rhs-scalar-value"} : i32
      // The EXACT imported scale shape tcrv_rvv.dequantize requires: imported
      // runtime ABI float, role dequant-scale-value.
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "block-computed-scale-dequant-imported-reject:scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "block-computed-scale-dequant-imported-reject:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_block_computed_scale_dequant_imported_reject, sew = 16 : i64, source_kernel = "rvv_block_computed_scale_dequant_imported_reject_kernel", status = "selected-lowering-boundary"} {
        %term = tcrv_rvv.block_computed_scale_dequant %sumi, %scale {kind = "computed_scale_sumi_dequant", dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32"} : i32, !tcrv_rvv.runtime_abi_value -> f32
      } : !tcrv_rvv.vl
    }
  }
}

// IMPORTEDSCALE: requires the computed_scale operand to be a COMPUTED f32 SSA value
