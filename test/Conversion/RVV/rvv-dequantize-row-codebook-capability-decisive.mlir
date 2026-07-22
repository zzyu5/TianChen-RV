// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=V128 --implicit-check-not=__riscv_vle8_v_i8mf2 --implicit-check-not=__riscv_vsext_vf4_i32m2 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m2 --implicit-check-not=__riscv_vle8_v_i8m2 --implicit-check-not=__riscv_vsext_vf4_i32m8 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m8
// RUN: sed 's/minimum_vlen = 128 : i64/minimum_vlen = 256 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=V256 --implicit-check-not=__riscv_vle8_v_i8m1 --implicit-check-not=__riscv_vsext_vf4_i32m4 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m4 --implicit-check-not=__riscv_vle8_v_i8m2 --implicit-check-not=__riscv_vsext_vf4_i32m8 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m8
// RUN: weft-opt %s --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=REG128 --implicit-check-not=__riscv_vle8_v_i8mf2 --implicit-check-not=__riscv_vsext_vf4_i32m2 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m2 --implicit-check-not=__riscv_vle8_v_i8m2 --implicit-check-not=__riscv_vsext_vf4_i32m8 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m8
// RUN: sed 's/minimum_vlen = 128 : i64/minimum_vlen = 256 : i64/' %s | weft-opt --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=REG256 --implicit-check-not=__riscv_vle8_v_i8m1 --implicit-check-not=__riscv_vsext_vf4_i32m4 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m4 --implicit-check-not=__riscv_vle8_v_i8m2 --implicit-check-not=__riscv_vsext_vf4_i32m8 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m8
// RUN: weft-opt %s --weft-rvv-materialize-dequantize-row-stream-front-door | FileCheck %s --check-prefix=PRE
// RUN: sed -e 's/, supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8"//' -e 's/, supported_sew = "8,16,32,64"//' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=BASE128 --implicit-check-not=__riscv_vle8_v_i8mf2 --implicit-check-not=__riscv_vsext_vf4_i32m2 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m2 --implicit-check-not=__riscv_vle8_v_i8m2 --implicit-check-not=__riscv_vsext_vf4_i32m8 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m8
// RUN: sed -e 's/minimum_vlen = 128 : i64/minimum_vlen = 256 : i64/' -e 's/, rvv_version = "1.0"//' -e 's/, supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8"//' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=BASE256 --implicit-check-not=__riscv_vle8_v_i8mf2 --implicit-check-not=__riscv_vsext_vf4_i32m2 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m2 --implicit-check-not=__riscv_vle8_v_i8m2 --implicit-check-not=__riscv_vsext_vf4_i32m8 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m8
// RUN: sed -e 's/@rvv/@zve32f/g' -e 's/id = "rvv", kind = "isa-vector", status = "available", minimum_vlen = 128 : i64/id = "rvv.profile.zve32f-vlen64", kind = "profile", status = "available", relations = #weft.capability_relations<provides = ["rvv"]>, minimum_vlen = 64 : i64/' -e 's/supported_sew = "8,16,32,64"/supported_sew = "8,16,32"/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=M2 --implicit-check-not=__riscv_vle8_v_i8m1 --implicit-check-not=__riscv_vle8_v_i8mf2 --implicit-check-not=__riscv_vsext_vf4_i32m4 --implicit-check-not=__riscv_vsext_vf4_i32m2 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m4 --implicit-check-not=__riscv_vfcvt_f_x_v_f32m2

// Production-path A3 witness. The abstract small-codebook row is constructed,
// Formula constructs the complete typed body before emission, and the mechanical
// emitter consumes it. Changing only selected
// capability c changes every intrinsic in the actual direct-vf4 chain:
//   VLEN128: i8m1  -> i32m4/f32m4
//   VLEN256: i8mf2 -> i32m2/f32m2
//   Zve32f VLEN64 profile: i8m2 -> i32m8/f32m8
// Both the public RVV wrapper and registry clone entry must run that preparation.

module {
  weft.exec.kernel @codebook_capability_decisive {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available", minimum_vlen = 128 : i64, rvv_version = "1.0", supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8", supported_sew = "8,16,32,64"}
    weft.exec.variant @mxfp4 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @mxfp4, sew = 32 : i64, source_kernel = "codebook_capability_decisive", status = "selected-lowering-boundary"} {
        %r = weft_rvv.dequantize_row %x, %y, %k, %vl {format = "mxfp4"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// V128: emitc.func @weft_emitc_codebook_capability_decisive_mxfp4
// V128: call_opaque "__riscv_vle8_v_i8m1"
// V128: call_opaque "__riscv_vle8_v_u8m1"
// V128: call_opaque "__riscv_vrgather_vv_i8m1"
// V128: call_opaque "__riscv_vsext_vf4_i32m4"
// V128: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// V128: call_opaque "__riscv_vfmul_vf_f32m4"
// V128: call_opaque "__riscv_vse32_v_f32m4"

// V256: emitc.func @weft_emitc_codebook_capability_decisive_mxfp4
// V256: call_opaque "__riscv_vle8_v_i8mf2"
// V256: call_opaque "__riscv_vle8_v_u8mf2"
// V256: call_opaque "__riscv_vrgather_vv_i8mf2"
// V256: call_opaque "__riscv_vsext_vf4_i32m2"
// V256: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// V256: call_opaque "__riscv_vfmul_vf_f32m2"
// V256: call_opaque "__riscv_vse32_v_f32m2"

// M2: emitc.func @weft_emitc_codebook_capability_decisive_mxfp4
// M2: call_opaque "__riscv_vle8_v_i8m2"
// M2: call_opaque "__riscv_vle8_v_u8m2"
// M2: call_opaque "__riscv_vrgather_vv_i8m2"
// M2: call_opaque "__riscv_vsext_vf4_i32m8"
// M2: call_opaque "__riscv_vfcvt_f_x_v_f32m8"
// M2: call_opaque "__riscv_vfmul_vf_f32m8"
// M2: call_opaque "__riscv_vse32_v_f32m8"

// REG128: emitc.func @weft_emitc_codebook_capability_decisive_mxfp4
// REG128: call_opaque "__riscv_vle8_v_i8m1"
// REG128: call_opaque "__riscv_vsext_vf4_i32m4"
// REG128: call_opaque "__riscv_vse32_v_f32m4"

// REG256: emitc.func @weft_emitc_codebook_capability_decisive_mxfp4
// REG256: call_opaque "__riscv_vle8_v_i8mf2"
// REG256: call_opaque "__riscv_vsext_vf4_i32m2"
// REG256: call_opaque "__riscv_vse32_v_f32m2"

// Optional restriction lists absent: selected @rvv still carries the base-V
// whole-rung and SEW contract. Fractional mf2 needs positive generation/token
// evidence, so VLEN256 with both version and LMUL-list absent stays on m1.
// BASE128: call_opaque "__riscv_vle8_v_i8m1"
// BASE128: call_opaque "__riscv_vsext_vf4_i32m4"
// BASE256: call_opaque "__riscv_vle8_v_i8m1"
// BASE256: call_opaque "__riscv_vsext_vf4_i32m4"

// The explicit construction front door produces the same final formula body as
// the mandatory lowering path.
// PRE: weft_rvv.dequantize_row_decode_core
// PRE-SAME: codebook_gather_entries = 16 : i64
// PRE-SAME: codebook_gather_table = "fp4-e2m1"
// PRE-SAME: codebook_scale_model = "e8m0-shared-exp"
// PRE-SAME: dequant_load_lmul = "m1"
// PRE-SAME: dequant_mechanism = "codebook-gather"
// PRE-SAME: dequant_strip_lanes = 16 : i64
