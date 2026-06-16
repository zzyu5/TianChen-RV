// The MEASUREMENT-BACKED autotuner (the genuine N3 "实测胜出"): the COMPILER selects
// the q4_1 block-dot shape by ACTUAL on-board measurement, READ from a cached
// tuning record, NOT a static cost-model guess. q4_1 is the kernel the static cost
// model MIS-RANKS (its argmin is m1/factor=4/elided, the SLOWEST legal shape, a
// measured loss); the measurement record fixes the pick to m1/factor=1/elided (the
// measured-fastest legal shape, a measured win over ggml). The static cost model is
// NOT deleted -- it stays as the candidate PRUNER (it still defines the legal set
// the tuner measures) AND the offline FALLBACK (used when no record is present).
//
// This test pins the three behaviours of the seam:
//   (1) record PRESENT + the recorded shape still legal  -> stamp the MEASURED shape.
//   (2) record ABSENT                                     -> fall back to the STATIC argmin.
//   (3) record STALE (names a now-illegal shape, I7)      -> fail-closed to the STATIC argmin.
// The same op + same --march; only the tune-record changes the decision.

// (1) Record present + recorded shape legal -- the full-V pass stamps the
// measured-fastest shape (m1, factor=1, elided), NOT the static argmin (m1,
// factor=4, elided), and tags the provenance distinctly (the recorded ns).
// RUN: tcrv-opt %s "--tcrv-rvv-materialize-q4-1-schedule=march=rv64gcv tune-record=%S/Inputs/q4-1-measurement-tuning-record.txt" | FileCheck %s --check-prefix=MEASURED

// (2) No record -- the SAME pass falls back to the static cost-model argmin (m1,
// factor=4, elided), the static mis-pick the measurement fixes. This is exactly
// the behaviour the autotuner-divergence test pins, preserved as the fallback.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q4-1-schedule=march=rv64gcv | FileCheck %s --check-prefix=FALLBACK

// (3) Stale record (fail-closed, I7) -- a record naming a strip-elided shape for
// the zve32x capability-march (where elided is capability-pruned) must NOT be
// stamped; the pass revalidates the recorded shape against the current legal set,
// finds it pruned, and falls back to the static argmin (m1, factor=2, robust).
// RUN: tcrv-opt %s "--tcrv-rvv-materialize-q4-1-schedule=march=rv64gc_zve32x tune-record=%S/Inputs/q4-1-stale-tuning-record.txt" | FileCheck %s --check-prefix=STALE

module {
  tcrv.exec.kernel @ggml_vec_dot_q4_1_q8_1_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q4_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_1_q8_1, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q4_1_q8_1_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q4_1_q8_1_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min", qk = 32 : i64, weight_block_stride = 20 : i64, activation_block_stride = 36 : i64, quant_byte_offset = 4 : i64, activation_high_byte_offset = 16 : i64, weight_min_byte_offset = 2 : i64, activation_sum_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// =================== (1) MEASURED (record present, shape legal) ==============
// The compiler stamped the MEASURED-fastest shape (m1, factor=1, elided), the FIX
// of the static mis-pick, with measured-fastest provenance + the recorded ns.
// MEASURED: tcrv_rvv.q4_1_q8_1_block_dot
// MEASURED-SAME: integer_core_lmul = "m1"
// MEASURED-SAME: multi_block_factor = 1 : i64
// MEASURED-SAME: strip_elision = "elided"
// MEASURED-SAME: tcrv_rvv.q4_1_schedule.measured_ns = 1.262800e+03
// MEASURED-SAME: tcrv_rvv.q4_1_schedule.selection_reason = "measured-fastest
// MEASURED-SAME: FIXES the static mis-pick

// =================== (2) FALLBACK (no record -> static argmin) ===============
// With no record the SAME op gets the STATIC cost-model argmin (m1, factor=4,
// elided) -- the slowest legal shape, the very mis-pick measurement fixes.
// FALLBACK: tcrv_rvv.q4_1_q8_1_block_dot
// FALLBACK-SAME: integer_core_lmul = "m1"
// FALLBACK-SAME: multi_block_factor = 4 : i64
// FALLBACK-SAME: strip_elision = "elided"
// FALLBACK-SAME: tcrv_rvv.q4_1_schedule.selection_reason = "min-cost legal Q4_1
// FALLBACK-NOT: tcrv_rvv.q4_1_schedule.measured_ns

// =================== (3) STALE -> FAIL-CLOSED (static argmin) ================
// The stale record named an elided shape for zve32x (capability-pruned), so the
// pass revalidated, found it illegal, and fell back to the static argmin
// (m1, factor=2, robust) -- it never stamped the stale illegal shape.
// STALE: tcrv_rvv.q4_1_q8_1_block_dot
// STALE-SAME: integer_core_lmul = "m1"
// STALE-SAME: multi_block_factor = 2 : i64
// STALE-SAME: strip_elision = "robust"
// STALE-SAME: tcrv_rvv.q4_1_schedule.selection_reason = "min-cost legal Q4_1
// STALE-NOT: tcrv_rvv.q4_1_schedule.measured_ns
