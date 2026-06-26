// The byte-anchor NARROW low-precision dequant body -- the hand-authored target
// the Track B dequant front door auto-constructs
// (Transforms/RVV/rvv-widening-dot-reduce-dequantize-source-front-door.mlir).
// This is the DIRECT dialect-level byte-exact reference: the SAME single-scope
// load/widening_product/standalone_reduce/dequantize/store chain, hand-authored
// at the SEW8 byte anchor, lowering through the existing isLowPrecisionDequantBody
// RVV->emitc sink. It proves the DequantizeOp + StoreOp narrow byte-anchor verifier
// branches (SEW8 LMUL m1/m2, source reduce <- widening_product) admit the dequant
// chain on the byte-anchor strip scope, and that the emitted intrinsics FLIP with
// the anchor -- the EMITC the front door reproduces byte-for-byte.
//
// This is the byte-anchor sibling of the SEW32/m1 grouped fixture
// (rvv-to-emitc-widening-product-reduce-dequantize-grouped.mlir) and the SEW8/m2
// deferred-wide fixture (rvv-to-emitc-widening-product-reduce-dequantize-wide-lmul.mlir):
// the SAME i32m1->f32m1 runtime-f32-scale dequant epilogue, here on the NARROW
// per-iteration-vwredsum byte-anchor path the front door selects.

// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=M2
// RUN: sed -e 's/"m2"/"m1"/g; s/i16, "m4"/i16, "m2"/g; s/i8m2xi8m2-to-i16m4/i8m1xi8m1-to-i16m2/g' %s | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=M1

module {
  tcrv.exec.kernel @byte_anchor_dequant_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @byte_anchor_dequant attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], selected_path_role = "dispatch case", selected_variant = @byte_anchor_dequant, sew = 8 : i64, source_kernel = "byte_anchor_dequant_kernel", status = "selected-lowering-boundary", unroll_factor = 1 : i64} {
        %l0 = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
        %r0 = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
        %p0 = tcrv_rvv.widening_product %l0, %r0, %vl {kind = "signed_widening_product", product_relation = "signed-i8m2xi8m2-to-i16m4"} : !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
        %red0 = tcrv_rvv.standalone_reduce %p0, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %deq = tcrv_rvv.dequantize %red0, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        tcrv_rvv.store %out, %deq, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// ===================== VLEN128 byte anchor (SEW8/m2 -> i16m4) ===============
// M2-LABEL: emitc.func @tcrv_emitc_byte_anchor_dequant_kernel_byte_anchor_dequant
// M2: call_opaque "__riscv_vle8_v_i8m2"
// M2: call_opaque "__riscv_vwmul_vv_i16m4"
// M2: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// M2-NOT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// The i32->f32 runtime-scale dequant epilogue (scalar-extract -> *scale -> store).
// M2: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// M2: call_opaque "__riscv_vfmv_v_f_f32m1"
// M2: call_opaque "__riscv_vse32_v_f32m1"

// ===================== VLEN256 byte anchor FLIP (SEW8/m1 -> i16m2) ==========
// M1-LABEL: emitc.func @tcrv_emitc_byte_anchor_dequant_kernel_byte_anchor_dequant
// M1: call_opaque "__riscv_vle8_v_i8m1"
// M1: call_opaque "__riscv_vwmul_vv_i16m2"
// M1: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// M1-NOT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// M1: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// M1: call_opaque "__riscv_vfmv_v_f_f32m1"
// M1: call_opaque "__riscv_vse32_v_f32m1"
