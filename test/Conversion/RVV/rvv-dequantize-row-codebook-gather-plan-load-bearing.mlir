// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/quant_byte_offset = 1 : i64/quant_byte_offset = 3 : i64/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADG
// RUN: sed 's/qk = 32 : i64/qk = 44 : i64/g' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADG
// RUN: sed 's/weight_block_stride = 17 : i64/weight_block_stride = 19 : i64/g' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADG
// RUN: sed 's/scale_byte_offset = 0 : i64/scale_byte_offset = 99 : i64/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADG

// A3 load-bearing boundary: construction owns canonical mechanism geometry g;
// backend preparation runs the typed formula and writes a complete selected stamp;
// emission only consumes that stamp. The four production layouts are exact, so qk,
// stride, scale offset and quant offset are not mutation knobs. Any stale/forged g
// must fail before emission rather than creating a new accidental format.

module {
  weft.exec.kernel @dequant_mxfp4_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available", minimum_vlen = 128 : i64, rvv_version = "1.0", supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8", supported_sew = "8,16,32,64"}
    weft.exec.variant @dequant_mxfp4 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_mxfp4, sew = 32 : i64, source_kernel = "dequant_mxfp4_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_dequantize_row_loop_body %x, %y, %k attributes {decode_model = "mxfp4", kind = "typed_dequantize_row_loop_body", qk = 32 : i64, weight_block_stride = 17 : i64} {
        ^bb0(%block_index: index):
          weft_rvv.dequantize_row_decode_core %x, %y, %block_index {codebook_scale_model = "e8m0-shared-exp", decode_model = "mxfp4", dequant_mechanism = "codebook-gather", qk = 32 : i64, quant_byte_offset = 1 : i64, scale_byte_offset = 0 : i64, weight_block_stride = 17 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
          weft_rvv.typed_dequantize_row_loop_yield
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// NORMAL (quant_byte_offset = 1, qk = 32): the plan's codebookByteOffset (1) is the qs
// pointer offset, and the plan's stripLanes (qk/2 = 16) is the per-strip vl the codebook
// nibble load (vle8_v_u8m1) uses; the codebook TABLE load (vle8_v_i8m1) uses
// codebookEntries (16).
// CHECK: emitc.func @weft_emitc_dequant_mxfp4_kernel_dequant_mxfp4(
// CHECK: %[[ENTRIES:.*]] = literal "16" : !emitc.opaque<"size_t">
// CHECK: call_opaque "__riscv_vle8_v_i8m1"(%{{.*}}, %[[ENTRIES]])
// Anchor past the E8M0 scale decode so the qs offset "1" (size_t) is not confused with the
// block-loop step; the FIRST size_t "1" after mxfp4_decode is plan.codebookByteOffset.
// CHECK: verbatim{{.*}}callee=mxfp4_decode
// CHECK: %[[QSOFF:.*]] = literal "1" : !emitc.opaque<"size_t">
// CHECK-NEXT: add %{{.*}}, %[[QSOFF]]
// CHECK: %[[STRIP:.*]] = literal "16" : !emitc.opaque<"size_t">
// CHECK-NEXT: verbatim{{.*}}callee=__riscv_vle8_v_u8m1
// CHECK-NEXT: call_opaque "__riscv_vle8_v_u8m1"(%{{.*}}, %[[STRIP]])

// BADG: codebook gather decision rejected domain 'rvv.dequant.small-codebook': rejected-invalid-geometry
