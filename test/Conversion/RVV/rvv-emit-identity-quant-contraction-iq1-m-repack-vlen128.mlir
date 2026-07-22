// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// C4a-3 iq1_m GRID CONSTRUCTION -- the FIFTH row of the weft::GridDecodePlan registry and
// iq1_s's ternary-grid sibling. Landed the same four ways C4a-2 landed iq1_s (registry row +
// front door + BOTH emitter leaves + a byte-exact oracle), because a registry row the front
// door cannot construct and no emitter leaf can lower would ship a "verifier accepts but
// nobody can lower" inconsistency.
//
// iq1_m REUSES four of five axis values (I64x8 entry, TernaryDelta sign plane, 2048-entry
// grid, and -- from iq2_xs/iq2_s -- Dual ls arity). Its ONE new axis value is the fold,
// GridFoldArith::DeltaGridGroupSum, and this test is where the three structural deltas from
// iq1_s become checkable:
//
//   (a) NO inline d. block_iq1_m is 56 B of qs[32]+qh[16]+scales[8] with NO ggml_half member
//       at all -- hence quant_byte_offset = 0 here, unique among the grid rows (every other
//       one is 2, "the qs follow the inline fp16 d"). The fp16 super-block d is ASSEMBLED
//       from four nibbles scattered across the scales words at REPACK time, so the kernel
//       loads an ordinary inline fp16 strip.
//   (b) DUAL ls (the iq2_xs shape, reused).
//   (c) PER-GROUP delta over IN-KERNEL group-of-8 activation sums, and therefore NO bsums.
//       block_q8_K's bsums are sums over groups of SIXTEEN; iq1_m's four deltas per
//       sub-block are INDEPENDENT, so the two 8-groups inside one bsums entry can carry
//       opposite signs and a bsums entry CANNOT express the term. The absence of
//       activation_bsums_byte_offset below is a FACT about the format, not an omission --
//       the loop-body verifier rejects the attr for this fold, and the NOWALL block at the
//       bottom pins that the emitted C contains no int16 activation read at all.
//
// The abstract, algorithm-UNCOMMITTED weft_rvv.quant_contraction op (iq1_m grid / decode) is
// AUTO-LOWERED by --weft-rvv-lower-quant-contraction at rv64gcv: selection reads the
// STRUCTURED opponent facts (block_dot_compute_heavy = true) + the derived capability VLEN
// (rv64gcv => 128) and picks REPACK; the decode FAMILY is keyed off the committed iq1_m
// scale_model WHAT, so lowerToRepackGemvGrid CONSTRUCTS the typed
// weft_rvv.typed_repack_gemv_loop_body REGION (fold_model
// "grid_ternary_delta_groupsum_eighth") carrying the SINGLE weft_rvv.repack_gemv_grid_core
// brick (decode_model "iq1_m"), reconstructing the block_iq1_mx16 x16 weight facts
// 1824/800/32/288 + the plain block_q8_K activation facts 292/4, half_lanes=8 => mf2,
// numHalves==2.
//
// The FIXED 2048-entry TERNARY grid the abstract request DOES NOT carry is RECONSTRUCTED at
// emit as a static const decl. There is NO sign plane to reconstruct -- that is what
// TernaryDelta means.
//
// NO perf/e2e claim of any kind: C4a-3 is a CONSTRUCTION line and never touched a board.
// Numeric correctness is gated separately + byte-exactly by
// tools/oracle-repack/oracle_repack_iq1_m.cpp (ZERO-MODEL, x86-native).

module {
  weft.exec.kernel @ggml_repack_gemv_iq1_m_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_iq1_m_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // The ABSTRACT ternary-delta-groupsum grid request: PLAIN block_iq1_m (stride 56 =
        // 32 uint8 qs + 16 uint8 qh + 8 uint8 scales, qs @0 -- there is NO inline d) x PLAIN
        // block_q8_K (stride 292), qk 256, scale_model = the iq1_m WHAT,
        // block_dot_compute_heavy = true (routes REPACK). It carries NO grid -- the compiler
        // RECONSTRUCTS it. The result is dead (the repacked lane-wise grid GEVM sinks
        // through the output pointer).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq1_m", scale_model = "superblock-d.fp16-grid-ternary-delta-groupsum-dualscale-nomin-eighth", m_regime = "decode", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 56 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 0 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the ternary-delta-groupsum grid
// typed_repack GEVM region (fold_model grid_ternary_delta_groupsum_eighth) carrying the grid
// core brick (decode_model iq1_m), NOT a hand-authored region.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// CONSTRUCT: weft_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: fold_model = "grid_ternary_delta_groupsum_eighth"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: scale_model = "superblock-d.fp16-grid-ternary-delta-groupsum-dualscale-nomin-eighth"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 1824
// The in-region block_index-tied grid integer-core BRICK, decode_model iq1_m + the
// RECONSTRUCTED grid / dual-ls / DELTA byte offsets (the abstract request carried plain
// facts). The weight_sign_byte_offset SLOT carries the PER-GROUP +-1 DELTA strip @288: a
// TernaryDelta row has no sign plane, so the slot addresses the delta strip instead.
// CONSTRUCT: weft_rvv.repack_gemv_grid_core
// CONSTRUCT-SAME: decode_model = "iq1_m"
// CONSTRUCT-SAME: weight_ls_byte_offset = 32
// CONSTRUCT-SAME: weight_quant_byte_offset = 800
// CONSTRUCT-SAME: weight_sign_byte_offset = 288
// The loop body must NOT carry activation_bsums_byte_offset. This is (c) above made
// checkable: block_q8_K's per-16 bsums cannot express iq1_m's per-8 delta group sum, so
// unlike iq1_s (whose fixture CHECKs activation_bsums_byte_offset = 260) iq1_m reads none.
// CONSTRUCT-NOT: activation_bsums_byte_offset

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to a REAL iq1_m GEVM body -- the half that makes the
// registry row honest rather than an advertised-but-unlowerable capability.
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_grid_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_iq1_m_q8_K_kernel_ggml_repack_gemv_iq1_m_q8_K(
// The FIXED 2048-entry TERNARY grid decl (RECONSTRUCTED by the compiler), emitted from the
// SAME canonical kIQ1MGrid the iq1_m BLOCK-DOT path emits, so the two paths cannot disagree.
// CHECK: verbatim "static const uint64_t weft_iq1m_grid[2048] = {0xffffffffffffffffULL
// The body's grid table POINTER is the registry row's gridArrayName (pure DATA), NOT a
// hard-coded emitter string -- and it must name the SAME table the decl above declares.
// This CHECK is load-bearing for the SAME reason the iq1_s fixture's is: the decl comes from
// a canonical helper, so WITHOUT this line a corrupted registry gridArrayName emits a body
// referencing an UNDECLARED table (a dangling reference that would not compile) while the
// test stayed green. Verified by injecting gridArrayName -> "weft_iq1m_grid_BOGUS": with this
// CHECK the test goes RED; without it, it passed. The trailing quote makes the match exact,
// so a _BOGUS suffix cannot satisfy it.
// CHECK: literal "weft_iq1m_grid"
// Per-group weight base vx + x*nb*1824 (block_iq1_mx16 stride 1824).
// CHECK: literal "1824"
// The plain q8_K activation stride 292.
// CHECK: literal "292"
// The REAL memory ternary-grid GATHER: a u16 index strip (vle16, NO vzext) -> vsll -> vluxei16.
// CHECK: call_opaque "__riscv_vle16_v_u16m1"
// CHECK: call_opaque "__riscv_vsll_vx_u16m1"
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// The i32 in-block dot + the DUAL per-sub-block ls-scale vmacc. NOTE there is NO sign gather
// and NO vmul_vv_i8 sign fold: the gathered ternary grid byte IS the signed weight.
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vv_i32m2"
// The DELTA term: ls*delta (vmul_vv) then += groupSum*(ls*delta) (vmacc_vx, the group-of-8
// activation sum being the SCALAR term). Same two instructions as iq1_s's delta term -- the
// difference is entirely in where the scalar comes from, which the NOWALL block pins.
// CHECK: call_opaque "__riscv_vmul_vv_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vx_i32m2"
// The DeltaGridGroupSum end-of-block fold: cvt(sumi1) + 0.125f*cvt(sumi2), then vfmacc by
// d_x*d_y.
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: literal "0.125f"
// CHECK: call_opaque "__riscv_vfadd_vv_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The grid decode is a MEMORY gather, NOT a register vrgather; the block-as-lane repack
// erases the cross-lane reduction wall; the ternary-delta fold has NO min.
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// iq1_m has NO sign plane at all: neither an iq2-style signs table nor the vmul-onto-grid
// sign fold may appear. If either showed up, the emitter would be silently running the iq2
// dual-ls leaf -- which is a LIVE hazard for iq1_m specifically, because its ls arity IS
// Dual, so only the fold-arith-first dispatch order keeps it off that leaf.
// NOWALL-NOT: signs64
// NOWALL-NOT: signs256
// NOWALL-NOT: __riscv_vmul_vv_i8mf2
// iq1_m reads NO bsums: there is no int16 scalar activation read anywhere in the body. This
// is the load-bearing (c) check and the one that separates this leaf from iq1_s's, whose
// delta term is exactly `*(const int16_t *)` bsums reads. A body that grew a bsums read
// would be either the iq1_s leaf mis-selected or a per-16 term that cannot be correct here.
// NOWALL-NOT: const int16_t
