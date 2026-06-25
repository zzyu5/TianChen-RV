// The CODEBOOK-class breadth proof (Win-A brick #2b): the COMPILER SELECTS the
// ggml IQ4_NL x Q8_0 block-dot shape (the FP4 codebook family, a THIRD real
// llama.cpp kernel class) and the selection DIVERGES by the REAL VLEN capability
// fact -- via the SAME walk-all autotuner machinery the linear-decode siblings
// use (tcrv-rvv-materialize-schedule auto-discovers the op the moment it adopts
// TunableScheduleOpInterface + a registry descriptor; no new pass).
//
// The kernel below carries NO shape knobs (no integer_core_lmul /
// multi_block_factor / strip_elision) -- the compiler must compute them. The
// codebook integer core GATHERS each nibble through a broadcast 16-entry table, so
// its legal i8 anchor is the VLEN-CAPABILITY fact "the strip VLMAX spans the whole
// 16-entry table-index range": at VLEN128 only m1 reaches VLMAX 16 (mf2 -> 8 < 16,
// PRUNED); at VLEN256 mf2 ALSO reaches VLMAX 16 (the ggml `_vl256` shape). So the
// SAME attr-less op selects the m1 anchor at VLEN128 but FLIPS to mf2 at VLEN256 --
// a byte-different emitted kernel (the Win-A capability-driven lowering, NOT a
// structural NULL). The factor is PINNED to 1 (the codebook descriptor caps the
// multi_block_factor unroll AXIS at 1, isolating THIS brick to the anchor-flip
// novelty; the deep codebook-decode chain WOULD saturate the {1,2,4} unroll at 4
// like q4_0, but that factor-4 unroll is a legal Win-A axis verified as its OWN
// brick, not folded in here). So the divergence is purely the coreLmul anchor, and
// the VLEN128 default emit is the already-board-validated m1/factor-1/elided form.

// The DECISION-LEVEL proof: the autotuner stamps DIFFERENT shape knobs onto the
// SAME attr-less op purely by the REAL VLEN fact (no lowering involved).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv | FileCheck %s --check-prefix=STAMP-VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=STAMP-VLEN256
//
// The EMISSION-LEVEL non-NULL proof: the selected anchor carries through the
// lowering, and the two VLENs emit BYTE-DIFFERENT codebook cores.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT-VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT-VLEN256

module {
  tcrv.exec.kernel @ggml_vec_dot_iq4_nl_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_iq4_nl_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_iq4_nl_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_iq4_nl_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.iq4_nl_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_iq4_nl_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// =============== STAMPED SHAPE KNOBS (the SELECTION decision) ===============
// rv64gcv (VLEN128): the compiler SELECTED the m1 anchor -- the ONLY legal anchor
// (mf2 was PRUNED: its strip VLMAX 8 < 16 cannot index the codebook table). The
// SEMANTIC minimum_vlen = 128 is stamped (the verifier recomputes the gather
// legality from it). The multi_block_factor unroll axis is CAPPED at 1 (the
// codebook factorCap), so 2 of 4 candidates are legal (the m1 anchor's {robust,
// elided} at factor 1) and the default is the board-validated m1/factor-1/elided
// pinned form.
// STAMP-VLEN128: tcrv_rvv.iq4_nl_q8_0_block_dot
// STAMP-VLEN128-SAME: integer_core_lmul = "m1"
// STAMP-VLEN128-SAME: minimum_vlen = 128 : i64
// STAMP-VLEN128-SAME: multi_block_factor = 1 : i64
// STAMP-VLEN128-SAME: strip_elision = "elided"
// STAMP-VLEN128-SAME: tcrv_rvv.iq4_nl_schedule.legal_candidate_count = 2 : i64
// STAMP-VLEN128-SAME: tcrv_rvv.iq4_nl_schedule.peak_live_vector_registers = 6 : i64
// STAMP-VLEN128-SAME: tcrv_rvv.iq4_nl_schedule.producer = "rvv-iq4-nl-autotuner"
//
// rv64gcv_zvl256b (VLEN256): the SAME attr-less op FLIPS to the mf2 anchor. The
// REAL VLEN fact -- stamped as minimum_vlen = 256 -- makes mf2's strip VLMAX reach
// 16 (it can now index the whole codebook table), so mf2 (the ggml `_vl256` shape)
// becomes LEGAL; all 4 candidates (factor capped at 1) are legal, mf2 TIES m1 on the
// capability-blind cost, and the LIGHTER peak-live footprint (mf2's i16 product is
// m1 = 5 vregs < m1's i16m2 = 6) breaks the tie to mf2. The anchor MOVES with VLEN
// -- the Win-A brick (1-bit Zvl128b boolean -> real VLEN-bits-driven codebook
// lowering); the factor stays 1 at BOTH VLENs (the anchor-flip is the whole novelty).
// STAMP-VLEN256: tcrv_rvv.iq4_nl_q8_0_block_dot
// STAMP-VLEN256-SAME: integer_core_lmul = "mf2"
// STAMP-VLEN256-SAME: minimum_vlen = 256 : i64
// STAMP-VLEN256-SAME: multi_block_factor = 1 : i64
// STAMP-VLEN256-SAME: strip_elision = "elided"
// STAMP-VLEN256-SAME: tcrv_rvv.iq4_nl_schedule.legal_candidate_count = 4 : i64
// STAMP-VLEN256-SAME: tcrv_rvv.iq4_nl_schedule.peak_live_vector_registers = 5 : i64

// ===================== EMISSION (VLEN128 — the m1 form) =====================
// The compiler-selected m1 anchor carries through the lowering: the codebook table
// + nibble gather + strip vsetvl are spelled at e8m1 (VLMAX 16 = the whole 16-lane
// half-block in ONE cover), the widened i16 product is m2. The factor is PINNED to 1
// (the codebook factorCap), so the emit is the SINGLE-block core (NO by-4 outer
// loop, NO nb % 4 scalar tail) -- BYTE-IDENTICAL to the already-board-validated
// m1/factor-1/elided pinned form (rvv-to-emitc-iq4-nl-q8-0-block-dot.mlir, ssh-rvv
// byte-exact vs ggml, artifacts/inc30-iq4_nl). So VLEN128 needs ZERO new board work.
// EMIT-VLEN128: emitc.func @tcrv_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_ggml_vec_dot_iq4_nl_q8_0(
// The single-block core: NO by-4 main-loop split (nb_main = nb - nb % 4 is absent).
// EMIT-VLEN128-NOT: rem %{{.*}}, %{{.*}}
// EMIT-VLEN128: call_opaque "__riscv_vsetvl_e8m1"
// EMIT-VLEN128: call_opaque "__riscv_vrgather_vv_i8m1"
// EMIT-VLEN128-NOT: call_opaque "__riscv_vsetvl_e8mf2"
// EMIT-VLEN128-NOT: call_opaque "__riscv_vrgather_vv_i8mf2"
// EMIT-VLEN128: return

// ============ EMISSION (VLEN256 — the mf2 FLIP, byte-different) =============
// The compiler-selected mf2 anchor emits a BYTE-DIFFERENT codebook core: the table
// load / nibble gather / strip vsetvl narrow to e8mf2 (a FULL mf2 register at
// VLEN256, VLMAX 16 = the ggml `_vl256` shape), and the widened i16 product is m1
// (the FLIP from i16m2). This is the NON-NULL proof: the two VLENs do NOT emit the
// same bytes -- a capability FACT changes the lowering.
// EMIT-VLEN256: emitc.func @tcrv_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_ggml_vec_dot_iq4_nl_q8_0(
// EMIT-VLEN256: call_opaque "__riscv_vsetvl_e8mf2"
// EMIT-VLEN256: call_opaque "__riscv_vrgather_vv_i8mf2"
// EMIT-VLEN256-NOT: call_opaque "__riscv_vsetvl_e8m1"
// EMIT-VLEN256-NOT: call_opaque "__riscv_vrgather_vv_i8m1"
// EMIT-VLEN256: return
