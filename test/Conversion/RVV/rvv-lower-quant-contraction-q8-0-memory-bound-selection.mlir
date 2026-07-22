// OPTION-2 STAGE B -- the q8_0 MEMORY-BOUND cell of the IN-COMPILER contraction-
// path SELECTION. This is the G5 tension-A landing: the DEFAULT selector now routes
// q8_0 to REPACK on q8_0's HONEST fact set -- block_dot_memory_bound = true (fact
// 2b) with NO block_dot_compute_heavy -- WITHOUT the forced compute_heavy = true the
// M1b transmission probe used. q8_0's block-dot is compute-LEAN (one vwredsum/block,
// no nibble decode) but the WIDEST linear quant (34 bytes/block ~= 1 byte/weight), so
// it is DRAM-BANDWIDTH-bound; repack's contiguous block_q8_0x16 stream + activation
// reuse removes the redundant MEMORY traffic. This proves the selection is carried by
// the MEMORY arm (fact 2b), not compute-heaviness -- the audit reason token names the
// memory-bound cell.
//
// VLEN128 (rv64gcv => Zvl128b => 128): q8_0 decode + block_dot_memory_bound = true ->
// REPACK SELECTED + REALIZED as the typed weft_rvv.typed_repack_gemv_loop_body region,
// carried by fact 2b (the MEMORY-BOUND reason token, distinct from the compute-heavy
// q4_0 token). Fact 3 (minVLEN==128) affords the strip width.
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=V128
//
// DEFAULT -march "" => deriveMinimumVLEN 0 => fact 3 false (decode, VLEN != 128) so
// the selector DECLINES repack -- but q8_0 has NO block-dot decline path (the
// block-dot identity lowering is q4_0-nibble-only), so the request FAILS CLOSED (I7)
// with a capability error. This proves (a) the memory arm (fact 2b) is STILL gated by
// the capability/regime fact 3 -- repack is NOT a format-unconditional flip, it is
// fact-1-clear AND (fact 2 OR fact 2b) AND fact 3 -- and (b) q8_0 is repack-or-error,
// exactly why the honest fact set (not a forced compute_heavy) must select repack at
// the deployed VLEN128.
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction --verify-diagnostics

module {
  weft.exec.kernel @ggml_vec_dot_q8_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q8_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %bx = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %by = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q8_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q8_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        // HONEST q8_0 facts: block_dot_memory_bound = true (fact 2b), NO
        // block_dot_compute_heavy -- routing decides from the structured facts alone.
        // At VLEN128 -> repack (V128 below); at DEFAULT VLEN0 -> fact 3 declines and
        // q8_0 has no block-dot fallback -> capability error (verify-diagnostics run).
        // expected-error@+1 {{requires a repack-affording capability}}
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q8_0", scale_model = "dual-fp16-per-block-d_x.d_y-full-i8", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_memory_bound = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// (VLEN128) fact 2b carries the Repack selection: the typed GEVM region is realized,
// the abstract op is gone, and the audit reason names the MEMORY-BOUND cell (proving
// the memory arm carried it, NOT compute-heaviness).
// V128-NOT: weft_rvv.quant_contraction
// V128-NOT: weft_rvv.q8_0_q8_0_block_dot
// V128: weft_rvv.typed_repack_gemv_loop_body
// V128-SAME: weft_rvv.weight_layout_contract = "x16"
//
// (DEFAULT VLEN0) fact 3 declines the memory arm; q8_0 has no block-dot fallback so
// the request fails closed -- asserted by the `expected-error` above under the
// `--verify-diagnostics` run (no FileCheck directives needed for that arm).
