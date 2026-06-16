// The MEASUREMENT-BACKED GEMM M-block autotuner (the genuine N3 "实测胜出" for the
// GEMM front, INC-26 / G3): the COMPILER selects the ggml Q4_0 x Q8_0 FULL GEMM's
// M (the activation-column block) by ACTUAL on-board measurement, READ from a
// cached tuning record, NOT a static cost-model guess. It REUSES the EXACT
// tune-once -> cache -> read mechanism the block-dot measurement tuner (INC-10)
// established (the record reader, the fail-closed revalidation, the static
// fallback) -- the only difference is the knob: a single M (the cache/vreg
// column-block) instead of the LMUL/factor/elision triple.
//
// The headline: the fair interleaved on-board ladder picks M=6 (~1.19x vs
// per-(row,col) vec_dot at K=4096), which OVERTURNS the naive static default
// (M=4). M is NOT capability-gated -- the cache/vreg M optimum is noisy +
// analytically unpredictable, so MEASUREMENT decides (it overturns the static
// default), exactly the measurement-overturns-prediction pattern.
//
// This test pins the three behaviours of the seam (the SAME op + same --march;
// only the tune-record changes the decision):
//   (1) record PRESENT + the recorded M still legal -> stamp the MEASURED M=6.
//   (2) record ABSENT                               -> the STATIC default M=4.
//   (3) record STALE (names a now-illegal M, I7)    -> fail-closed to STATIC M=4.

// (1) Record present + recorded M legal -- the pass stamps the measured-fastest
// M (=6), NOT the static default (=4), and tags the provenance distinctly with
// the recorded ns. measured(M=6) != default(M=4) is a VISIBLE proof the record
// drove the choice.
// RUN: tcrv-opt %s "--tcrv-rvv-materialize-gemm-schedule=march=rv64gcv tune-record=%S/Inputs/gemm-measurement-tuning-record.txt" | FileCheck %s --check-prefix=MEASURED

// (2) No record -- the SAME pass falls back to the static default M (=4, the safe
// cache-friendly tile), with no measured_ns.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gemm-schedule=march=rv64gcv | FileCheck %s --check-prefix=FALLBACK

// (3) Stale record (fail-closed, I7) -- a record naming M=16 (outside the legal
// band, vreg ceiling 8) must NOT be stamped; the pass revalidates the recorded M
// against the current band, finds it illegal, and falls back to the static
// default M (=4).
// RUN: tcrv-opt %s "--tcrv-rvv-materialize-gemm-schedule=march=rv64gcv tune-record=%S/Inputs/gemm-stale-tuning-record.txt" | FileCheck %s --check-prefix=STALE

module {
  tcrv.exec.kernel @gemm {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @gemm attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %by = tcrv_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bx = tcrv_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @gemm, sew = 32 : i64, source_kernel = "gemm", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.q4_0_q8_0_gemm %vx, %vy, %by, %s, %n, %nr, %nc, %bx, %bs, %vl {kind = "ggml_q4_0_q8_0_gemm", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.runtime_abi_value, index, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// =================== (1) MEASURED (record present, M legal) ==================
// The compiler stamped the MEASURED-fastest M (=6), overturning the static
// default (=4), with measured-fastest provenance + the recorded ns.
// MEASURED: tcrv_rvv.q4_0_q8_0_gemm
// MEASURED-SAME: activation_cols = 6 : i64
// MEASURED-SAME: tcrv_rvv.q4_0_gemm_schedule.measured_ns = 1.243800e+03
// MEASURED-SAME: tcrv_rvv.q4_0_gemm_schedule.selection_reason = "measured-fastest

// =================== (2) FALLBACK (no record -> static default) ==============
// With no record the SAME op gets the STATIC default M (=4, the safe
// cache-friendly tile), with NO measured_ns.
// FALLBACK: tcrv_rvv.q4_0_q8_0_gemm
// FALLBACK-SAME: activation_cols = 4 : i64
// FALLBACK-SAME: tcrv_rvv.q4_0_gemm_schedule.selection_reason = "static default
// FALLBACK-NOT: tcrv_rvv.q4_0_gemm_schedule.measured_ns

// =================== (3) STALE -> FAIL-CLOSED (static default) ===============
// The stale record named M=16 (outside the legal band), so the pass revalidated,
// found it illegal, and fell back to the static default M (=4) -- it never
// stamped the stale illegal M.
// STALE: tcrv_rvv.q4_0_q8_0_gemm
// STALE-SAME: activation_cols = 4 : i64
// STALE-SAME: tcrv_rvv.q4_0_gemm_schedule.selection_reason = "static default
// STALE-NOT: tcrv_rvv.q4_0_gemm_schedule.measured_ns
