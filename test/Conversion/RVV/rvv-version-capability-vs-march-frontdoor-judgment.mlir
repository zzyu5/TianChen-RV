// JUDGMENT — the front-door RVV ISA-generation fact is PULLED off the in-IR
// capability provider op (resolveRVVVersion reads the materialized `rvv_version`),
// NOT re-parsed from -march. This is the version-axis sibling of the pulled VLEN
// pipe (resolveRVVMinimumVLEN): the LOAD-BEARING generation flows through the typed
// capability object, and a capability file WINS over a conflicting -march.
//
// The SAME pass, the SAME -march (rv64gcv => the ratified RVV1.0 generation), the
// SAME q5_1 decode repack-GEVM construction path -- differing ONLY in whether the
// capability provider op declares rvv_version = "0.7":
//   * NO provider rvv_version  -> resolveRVVVersion falls back to deriveRVVVersion
//                                 (rv64gcv) = RVV1.0 => isRVV0p7 = false => the
//                                 fractional mf2 default (half_lanes 8, reason
//                                 "capability-default-mf2").
//   * provider rvv_version="0.7" (a C920-class capability file that CONFLICTS with
//                                 the -march rv64gcv 1.0 string) -> the provider
//                                 fact WINS => isRVV0p7 = true => the mandatory
//                                 whole-LMUL m1 chain (half_lanes 16,
//                                 integer_core_lmul "m1", reason "correctness-rvv0p7").
//
// The accumulator LMUL flips PURELY on the capability `rvv_version` fact while
// -march is held at rv64gcv (1.0). Were the front door still re-parsing -march the
// output could NOT change (rv64gcv is 1.0 either way) -- so the flip proves the
// generation follows the CAPABILITY object, not the -march bypass (翻 march ≠ 假绿;
// core-invariants I1/I4).
//
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv \
// RUN:   | FileCheck %s --check-prefix=MARCH-DEFAULT
// RUN: sed 's/status = "available"}/status = "available", rvv_version = "0.7"}/' %s \
// RUN:   | weft-opt --weft-rvv-lower-quant-contraction=march=rv64gcv \
// RUN:   | FileCheck %s --check-prefix=CAP-RVV07
// RUN: sed 's/status = "available"}/status = "available", minimum_vlen = 128 : i64}/' %s \
// RUN:   | not weft-opt --weft-rvv-lower-quant-contraction 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNKNOWN-VERSION

module {
  weft.exec.kernel @ggml_vec_dot_q5_1_q8_1_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q5_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q5_1_q8_1, sew = 32 : i64, source_kernel = "ggml_vec_dot_q5_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q5_1", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min-five-bit", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 24 : i64, activation_block_stride = 36 : i64, quant_byte_offset = 4 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// -march rv64gcv (1.0) with NO capability rvv_version: the un-probed fallback
// derives RVV1.0 => the fractional mf2 default holds.
// MARCH-DEFAULT: weft_rvv.typed_repack_gemv_loop_body
// MARCH-DEFAULT-SAME: half_lanes = 8 : i64
// MARCH-DEFAULT-SAME: integer_core_lmul = "mf2"

// The capability file declares rvv_version = "0.7" while -march stays rv64gcv (1.0):
// the provider fact WINS => isRVV0p7 => the mandatory whole-LMUL m1 chain. The
// generation followed the CAPABILITY object, not the -march string.
// CAP-RVV07: weft_rvv.typed_repack_gemv_loop_body
// CAP-RVV07-SAME: half_lanes = 16 : i64
// CAP-RVV07-SAME: integer_core_lmul = "m1"

// A VLEN-only provider can make repack structurally reachable, but it does not
// prove whether fractional LMUL exists.  With no rvv_version and no -march
// fallback, A2 keeps the capability projection unknown and fails closed instead
// of silently treating Unknown as RVV1.0.
// UNKNOWN-VERSION: repack accumulator-LMUL decision failed closed: rejected-missing-capability
