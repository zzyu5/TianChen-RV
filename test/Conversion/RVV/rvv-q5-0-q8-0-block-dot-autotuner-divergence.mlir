// The Family-A 5-bit breadth proof: the COMPILER SELECTS the ggml Q5_0 x Q8_0
// block-dot shape (single-scale, 5-bit weight), and the selection DIVERGES by
// capability from the SAME attr-less input -- the autotuner generalizes to a
// structurally NEW (5-bit) real llama.cpp kernel.
//
// The kernel below carries NO shape knobs -- the compiler must compute them. The
// capability-aware autotuner pass (--tcrv-rvv-materialize-q5-0-schedule) derives
// the Zvl128b capability fact from the selected -march, enumerates + prunes +
// ranks (the SAME capability-blind structural cost the q4_0 autotuner uses, fed
// q5_0's DERIVED latency depth -- the LONGEST dependent decode chain because the
// unsigned-nibble unpack + 5th-bit injection + offset-binary `-16` bias is the
// deepest of the block-dot formats) + selects + stamps; the lowering then emits
// that shape. Capability enters ONLY through the legality prune, so the SAME
// argmin diverges:
//
//   * --march=rv64gcv      (Zvl128b): strip-elided shapes LEGAL; min-cost legal
//     shape is (m1, factor=4, elided).
//   * --march=rv64gc_zve32x (no Zvl128b): strip-elided shapes PRUNED; same argmin
//     selects (m1, factor=2, robust).
//
// One capability fact (Zvl128b) -> N1 legality divergence (elided vs robust), on a
// structurally NEW (5-bit weight) real llama.cpp kernel.

// The DECISION-LEVEL proof: the autotuner stamps DIFFERENT shape knobs onto the
// SAME attr-less op purely by capability.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q5-0-schedule=march=rv64gcv | FileCheck %s --check-prefix=STAMP-FULLV
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q5-0-schedule=march=rv64gc_zve32x | FileCheck %s --check-prefix=STAMP-ZVE32X
//
// The EMISSION-LEVEL proof: the selected shape carries through the lowering.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q5-0-schedule=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=FULLV
// RUN: tcrv-opt %s --tcrv-rvv-materialize-q5-0-schedule=march=rv64gc_zve32x --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=ZVE32X

module {
  tcrv.exec.kernel @ggml_vec_dot_q5_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q5_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q5_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q5_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q5_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q5_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 22 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 6 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, weight_qh_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// ================= STAMPED SHAPE KNOBS (the SELECTION decision) =============
// rv64gcv (Zvl128b): the compiler SELECTED (m1, factor=4, elided) and recorded
// the resource provenance in the q5_0_schedule.* namespace.
// STAMP-FULLV: tcrv_rvv.q5_0_q8_0_block_dot
// STAMP-FULLV-SAME: integer_core_lmul = "m1"
// STAMP-FULLV-SAME: multi_block_factor = 4 : i64
// STAMP-FULLV-SAME: strip_elision = "elided"
// STAMP-FULLV-SAME: tcrv_rvv.q5_0_schedule.has_zvl128b = true
// STAMP-FULLV-SAME: tcrv_rvv.q5_0_schedule.producer = "rvv-q5-0-autotuner"
//
// rv64gc_zve32x (no Zvl128b): the SAME op gets (m1, factor=2, robust) -- the
// elided shapes were pruned. The shape DIVERGES purely by the capability fact.
// STAMP-ZVE32X: tcrv_rvv.q5_0_q8_0_block_dot
// STAMP-ZVE32X-SAME: integer_core_lmul = "m1"
// STAMP-ZVE32X-SAME: multi_block_factor = 2 : i64
// STAMP-ZVE32X-SAME: strip_elision = "robust"
// STAMP-ZVE32X-SAME: tcrv_rvv.q5_0_schedule.has_zvl128b = false

// =============================== FULL-V (rv64gcv) ===========================
// The compiler SELECTED (m1, factor=4, elided): the by-4 outer loop, FOUR
// adjacent elided integer cores (each ONE vsetvl_e8m1 + ONE vwredsum, NO inner
// strip for-loop), then the four folds, plus an nb%4 robust scalar tail.
// FULLV: emitc.func @tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(
// FULLV: %[[REM:.*]] = rem %{{.*}}, %{{.*}}
// FULLV: %[[MAIN:.*]] = sub %{{.*}}, %[[REM]]
// FULLV: for %[[IB:.*]] = %{{.*}} to %[[MAIN]] step
// The four elided cores: ONE vsetvl(16) + ONE vwredsum each, NO inner strip loop.
// FULLV: call_opaque "__riscv_vsetvl_e8m1"
// FULLV-NOT: for %{{.*}} = %{{.*}} to %{{.*}} step
// FULLV: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// FULLV: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// FULLV: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// FULLV: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// The nb % 4 ROBUST scalar tail keeps the strip loop.
// FULLV: for %{{.*}} = %[[MAIN]] to %{{.*}} step
// FULLV: call_opaque "__riscv_vsetvl_e8m1"
// FULLV: for %{{.*}} = %{{.*}} to %{{.*}} step
// FULLV: return

// =============================== ZVE32X (no Zvl128b) ========================
// The compiler SELECTED (m1, factor=2, robust): the by-2 outer loop, TWO per-block
// ROBUST integer cores (each keeps the inner strip for-loop carrying sumi), then
// the two folds, plus an nb%2 robust scalar tail. The strip-elided shape was
// PRUNED (illegal without Zvl128b).
// ZVE32X: emitc.func @tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(
// ZVE32X: %[[REM:.*]] = rem %{{.*}}, %{{.*}}
// ZVE32X: %[[MAIN:.*]] = sub %{{.*}}, %[[REM]]
// ZVE32X: for %[[IB:.*]] = %{{.*}} to %[[MAIN]] step
// Each of the two robust cores KEEPS the inner strip for-loop (VLEN-robust).
// ZVE32X: call_opaque "__riscv_vsetvl_e8m1"
// ZVE32X: for %{{.*}} = %{{.*}} to %{{.*}} step
// ZVE32X: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// ZVE32X: return
