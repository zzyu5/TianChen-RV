// The CODEBOOK-class breadth proof (Win-A brick #2b), MXFP4 sibling: the COMPILER
// SELECTS the ggml MXFP4 x Q8_0 block-dot shape (the FP4 e2m1 codebook) and the
// selection DIVERGES by the REAL VLEN fact via the SAME walk-all autotuner
// (tcrv-rvv-materialize-schedule). mxfp4 shares the codebook enumeration with
// iq4_nl VERBATIM (they differ only in block strides + codebook values, neither
// touching the shape axes), so the SAME m1->mf2 VLEN flip holds.
//
// The attr-less op's codebook gather pins its i8 anchor to "the strip VLMAX spans
// the 16-entry table": m1 at VLEN128 (mf2 -> VLMAX 8 < 16, PRUNED), mf2 ALSO at
// VLEN256 (the ggml `_vl256` shape). Byte-different emitted cores; factor is PINNED
// to 1 (the codebook factorCap isolates THIS brick to the anchor-flip; the factor-4
// unroll is a legal Win-A axis verified separately). Win-A capability-driven lowering.

// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv | FileCheck %s --check-prefix=STAMP-VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=STAMP-VLEN256
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT-VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT-VLEN256

module {
  tcrv.exec.kernel @ggml_vec_dot_mxfp4_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_mxfp4_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_mxfp4_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_mxfp4_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.mxfp4_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_mxfp4_q8_0_block_dot", scale_model = "e8m0-half-shared-exponent-per-block", qk = 32 : i64, weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 1 : i64, activation_quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// =============== STAMPED SHAPE KNOBS (the SELECTION decision) ===============
// rv64gcv (VLEN128): the compiler SELECTED the m1 anchor -- the ONLY legal anchor
// (mf2 PRUNED: strip VLMAX 8 < 16). minimum_vlen = 128 stamped; the factor axis is
// CAPPED at 1 (the codebook factorCap), so 2 of 4 legal and the default is the
// board-validated m1/factor-1/elided pinned form.
// STAMP-VLEN128: tcrv_rvv.mxfp4_q8_0_block_dot
// STAMP-VLEN128-SAME: integer_core_lmul = "m1"
// STAMP-VLEN128-SAME: minimum_vlen = 128 : i64
// STAMP-VLEN128-SAME: multi_block_factor = 1 : i64
// STAMP-VLEN128-SAME: strip_elision = "elided"
// STAMP-VLEN128-SAME: tcrv_rvv.mxfp4_schedule.legal_candidate_count = 2 : i64
// STAMP-VLEN128-SAME: tcrv_rvv.mxfp4_schedule.peak_live_vector_registers = 6 : i64
// STAMP-VLEN128-SAME: tcrv_rvv.mxfp4_schedule.producer = "rvv-mxfp4-autotuner"
//
// rv64gcv_zvl256b (VLEN256): the SAME op FLIPS to mf2 -- minimum_vlen = 256 makes
// mf2's VLMAX reach 16 (legal, the ggml `_vl256` shape), so mf2 TIES m1 on cost and
// the lighter footprint (5 < 6) breaks the tie. All 4 candidates (factor capped at
// 1) legal; the factor stays 1 at BOTH VLENs.
// STAMP-VLEN256: tcrv_rvv.mxfp4_q8_0_block_dot
// STAMP-VLEN256-SAME: integer_core_lmul = "mf2"
// STAMP-VLEN256-SAME: minimum_vlen = 256 : i64
// STAMP-VLEN256-SAME: multi_block_factor = 1 : i64
// STAMP-VLEN256-SAME: strip_elision = "elided"
// STAMP-VLEN256-SAME: tcrv_rvv.mxfp4_schedule.legal_candidate_count = 4 : i64
// STAMP-VLEN256-SAME: tcrv_rvv.mxfp4_schedule.peak_live_vector_registers = 5 : i64

// ===================== EMISSION (VLEN128 — the m1 form) =====================
// The factor is PINNED to 1 (the codebook factorCap), so the emit is the SINGLE-block
// core (NO by-4 outer loop, NO nb % 4 tail) -- BYTE-IDENTICAL to the already-board-
// validated m1/factor-1/elided pinned form (rvv-to-emitc-mxfp4-q8-0-block-dot.mlir).
// VLEN128 needs ZERO new board work.
// EMIT-VLEN128: emitc.func @tcrv_emitc_ggml_vec_dot_mxfp4_q8_0_kernel_ggml_vec_dot_mxfp4_q8_0(
// The single-block core: NO by-4 main-loop split (nb_main = nb - nb % 4 is absent).
// EMIT-VLEN128-NOT: rem %{{.*}}, %{{.*}}
// EMIT-VLEN128: call_opaque "__riscv_vsetvl_e8m1"
// EMIT-VLEN128: call_opaque "__riscv_vrgather_vv_i8m1"
// EMIT-VLEN128-NOT: call_opaque "__riscv_vsetvl_e8mf2"
// EMIT-VLEN128-NOT: call_opaque "__riscv_vrgather_vv_i8mf2"
// EMIT-VLEN128: return

// ============ EMISSION (VLEN256 — the mf2 FLIP, byte-different) =============
// EMIT-VLEN256: emitc.func @tcrv_emitc_ggml_vec_dot_mxfp4_q8_0_kernel_ggml_vec_dot_mxfp4_q8_0(
// EMIT-VLEN256: call_opaque "__riscv_vsetvl_e8mf2"
// EMIT-VLEN256: call_opaque "__riscv_vrgather_vv_i8mf2"
// EMIT-VLEN256-NOT: call_opaque "__riscv_vsetvl_e8m1"
// EMIT-VLEN256-NOT: call_opaque "__riscv_vrgather_vv_i8m1"
// EMIT-VLEN256: return
