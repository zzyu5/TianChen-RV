// The headline N1+N3 proof: the COMPILER SELECTS the ggml Q4_0 x Q8_0 block-dot
// shape, and the selection DIVERGES by capability from the SAME attr-less input.
//
// The kernel below carries NO shape knobs (no integer_core_lmul /
// multi_block_factor / strip_elision) -- the compiler must compute them. The
// capability-aware schedule formula derives
// the Zvl128b capability fact from the selected -march, enumerates + prunes +
// ranks + selects the resource-best legal shape, and stamps it; the lowering
// then emits that shape. Capability enters ONLY through the legality prune (the
// cost model is capability-blind), so the SAME argmin diverges:
//
//   * --march=rv64gcv      (full V => Zvl128b => VLEN >= 128): the strip-elided
//     shapes are LEGAL, and the min-cost legal shape is (m1, factor=4, elided)
//     -- the ~13% ggml-beating shape (NO inner strip loop; 4 adjacent
//     vwredsums + the by-4 outer loop). Correct for the full-V capability class
//     (the strip loop is dead code at VLEN >= 128).
//   * --march=rv64gc_zve32x (no Zvl128b: VLEN may be < 128): the strip-elided
//     shapes are PRUNED (illegal -- they would silently drop nibble bytes at
//     VLEN < 128), so the same argmin selects (m1, factor=2, robust) -- the
//     VLEN-robust strip-loop shape (the robust optimum).
//
// One capability fact (Zvl128b) -> N1 legality divergence (elided vs robust) ->
// N3 win (the elided shape beats ggml on a full-V board). This is the compiler
// SELECTING the shape, not a hand-set attr.

// First, the DECISION-LEVEL proof: the autotuner stamps DIFFERENT shape knobs
// onto the SAME attr-less op purely by capability (no lowering involved). This
// asserts "the compiler PROVABLY SELECTED a different shape" at the selection
// boundary, independent of the emission detail.
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gcv | FileCheck %s --check-prefix=STAMP-FULLV
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gc_zve32x | FileCheck %s --check-prefix=STAMP-ZVE32X
//
// Then the EMISSION-LEVEL proof: the selected shape carries through the lowering.
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=FULLV
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gc_zve32x --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=ZVE32X

module {
  weft.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %bx = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %by = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %dot = weft_rvv.q4_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q4_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ================= STAMPED SHAPE KNOBS (the SELECTION decision) =============
// rv64gcv (Zvl128b): the compiler SELECTED (m1, factor=4, elided) and recorded
// the resource provenance (capability fact + candidate counts + cost).
// STAMP-FULLV: weft_rvv.q4_0_q8_0_block_dot
// STAMP-FULLV-SAME: integer_core_lmul = "m1"
// STAMP-FULLV-SAME: multi_block_factor = 4 : i64
// STAMP-FULLV-SAME: strip_elision = "elided"
// STAMP-FULLV-NOT: weft_rvv.q4_0_schedule.
//
// rv64gc_zve32x (no Zvl128b): the SAME op gets (m1, factor=2, robust) -- the
// elided shapes were pruned. The shape DIVERGES purely by the capability fact.
// STAMP-ZVE32X: weft_rvv.q4_0_q8_0_block_dot
// STAMP-ZVE32X-SAME: integer_core_lmul = "m1"
// STAMP-ZVE32X-SAME: multi_block_factor = 2 : i64
// STAMP-ZVE32X-SAME: strip_elision = "robust"
// STAMP-ZVE32X-NOT: weft_rvv.q4_0_schedule.

// =============================== FULL-V (rv64gcv) ===========================
// The compiler SELECTED (m1, factor=4, elided): the by-4 outer loop, FOUR
// adjacent elided integer cores (each ONE vsetvl_e8m1 + ONE vwredsum, NO inner
// strip for-loop), then the four folds in ascending block order, plus an nb%4
// robust scalar tail.
// FULLV: emitc.func @weft_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
// The by-4 main loop bound nb_main = nb - nb % 4.
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
// The compiler SELECTED (m1, factor=2, robust): the by-2 outer loop, TWO
// per-block ROBUST integer cores (each keeps the inner strip for-loop carrying
// sumi), then the two folds in ascending order, plus an nb%2 robust scalar tail.
// The strip-elided shape was PRUNED (illegal without Zvl128b).
// ZVE32X: emitc.func @weft_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
// The by-2 main loop bound nb_main = nb - nb % 2.
// ZVE32X: %[[REM:.*]] = rem %{{.*}}, %{{.*}}
// ZVE32X: %[[MAIN:.*]] = sub %{{.*}}, %[[REM]]
// ZVE32X: for %[[IB:.*]] = %{{.*}} to %[[MAIN]] step
// Each of the two robust cores KEEPS the inner strip for-loop (VLEN-robust).
// ZVE32X: call_opaque "__riscv_vsetvl_e8m1"
// ZVE32X: for %{{.*}} = %{{.*}} to %{{.*}} step
// ZVE32X: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// ZVE32X: return
