// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// C4a-2 iq1_s GRID CONSTRUCTION -- the TERNARY-DELTA grid sibling, the FOURTH row of the
// weft::GridDecodePlan registry and the first NON-iq2 one. C4a deliberately REFUSED to
// register iq1_s because a registry row the front door cannot construct and no emitter leaf
// can lower would ship a "verifier accepts but nobody can lower" inconsistency. C4a-2 lands
// all of it at once, and THIS test is the front-door half of that proof.
//
// The abstract, algorithm-UNCOMMITTED weft_rvv.quant_contraction op (iq1_s grid / decode) is
// AUTO-LOWERED by --weft-rvv-lower-quant-contraction at rv64gcv: the in-compiler selection
// reads the STRUCTURED opponent facts (block_dot_compute_heavy = true) + the derived
// capability VLEN (rv64gcv => 128) and picks REPACK; the decode FAMILY is keyed off the
// committed iq1_s ternary-delta scale_model WHAT, so lowerToRepackGemvGrid CONSTRUCTS the
// typed weft_rvv.typed_repack_gemv_loop_body REGION (fold_model "grid_ternary_delta_eighth")
// carrying the SINGLE weft_rvv.repack_gemv_grid_core integer-core brick (decode_model
// "iq1_s"), reconstructing the block_iq1_sx16 x16 weight facts 1312/288/32/160 + the plain
// block_q8_K activation facts 292/4 + the bsums plane @260, half_lanes=8 => mf2, numHalves==2.
//
// The FIXED 2048-entry TERNARY iq1s_grid the abstract request DOES NOT carry is RECONSTRUCTED
// at emit as a static const decl (a true CONSTRUCT-from-abstract). There is NO sign plane to
// reconstruct -- that is what TernaryDelta means.
//
// NO perf/e2e claim of any kind: C4a-2 is a CONSTRUCTION line and never touched a board.
// Numeric correctness is gated separately + byte-exactly by
// tools/oracle-repack/oracle_repack_iq1_s.cpp (ZERO-MODEL, x86-native).

module {
  weft.exec.kernel @ggml_repack_gemv_iq1_s_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_iq1_s_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // The ABSTRACT ternary-delta grid request: PLAIN block_iq1_s (stride 50 = fp16 d +
        // 32 uint8 qs + 8 uint16 qh, qs @2) x PLAIN block_q8_K (stride 292), qk 256,
        // scale_model = the iq1_s ternary-delta WHAT, block_dot_compute_heavy = true (routes
        // REPACK). It carries NO grid -- the compiler RECONSTRUCTS it. The result is dead
        // (the repacked lane-wise grid GEVM sinks through the output pointer).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq1_s", scale_model = "superblock-d.fp16-grid-ternary-delta-singlescale-nomin-eighth", m_regime = "decode", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 50 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the ternary-delta grid typed_repack
// GEVM region (fold_model grid_ternary_delta_eighth) carrying the grid core brick
// (decode_model iq1_s), NOT a hand-authored region.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// CONSTRUCT: weft_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: activation_bsums_byte_offset = 260
// CONSTRUCT-SAME: fold_model = "grid_ternary_delta_eighth"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: scale_model = "superblock-d.fp16-grid-ternary-delta-singlescale-nomin-eighth"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 1312
// The in-region block_index-tied grid integer-core BRICK, decode_model iq1_s + the
// RECONSTRUCTED grid / ls / DELTA byte offsets (the abstract request carried plain facts).
// The weight_sign_byte_offset SLOT carries the +-1 DELTA strip @160: a TernaryDelta row has
// no sign plane, so the slot addresses the delta strip instead of a sign selector strip.
// CONSTRUCT: weft_rvv.repack_gemv_grid_core
// CONSTRUCT-SAME: decode_model = "iq1_s"
// CONSTRUCT-SAME: weight_ls_byte_offset = 32
// CONSTRUCT-SAME: weight_sign_byte_offset = 160

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to a REAL iq1_s GEVM body -- this is the half C4a could not
// have shipped, and the reason it refused the registry row.
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_grid_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_iq1_s_q8_K_kernel_ggml_repack_gemv_iq1_s_q8_K(
// The FIXED 2048-entry TERNARY grid decl (RECONSTRUCTED by the compiler), emitted from the
// SAME canonical kIQ1SGrid the iq1_s BLOCK-DOT path emits, so the two paths cannot disagree.
// CHECK: verbatim "static const uint64_t weft_iq1s_grid[2048] = {0xffffffffffffffffULL
// The body's grid table POINTER is the registry row's gridArrayName (pure DATA), NOT a
// hard-coded emitter string -- and it must name the SAME table the decl above declares.
// This CHECK is load-bearing: the decl is emitted by a canonical helper, so WITHOUT it a
// corrupted registry gridArrayName emits a body referencing an UNDECLARED table (a dangling
// reference that would not compile) while this test stayed green. Verified by injecting
// gridArrayName -> "weft_iq1s_grid_BOGUS": with this CHECK the test goes RED; without it, it
// passed. The trailing quote makes the match exact, so a _BOGUS suffix cannot satisfy it.
// CHECK: literal "weft_iq1s_grid"
// Per-group weight base vx + x*nb*1312 (block_iq1_sx16 stride 1312).
// CHECK: literal "1312"
// The plain q8_K activation stride 292.
// CHECK: literal "292"
// The REAL memory ternary-grid GATHER: a u16 index strip (vle16, NO vzext) -> vsll -> vluxei16.
// CHECK: call_opaque "__riscv_vle16_v_u16m1"
// CHECK: call_opaque "__riscv_vsll_vx_u16m1"
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// The i32 in-block dot + the SINGLE per-sub-block ls-scale vmacc. NOTE there is NO sign
// gather and NO vmul_vv_i8 sign fold: the gathered ternary grid byte IS the signed weight.
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vv_i32m2"
// The DELTA term: ls*delta (vmul_vv) then += bsumPair*(ls*delta) (vmacc_vx, the bsum pair
// being the SCALAR activation term). This is the mechanism no iq2 row has.
// CHECK: call_opaque "__riscv_vmul_vv_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vx_i32m2"
// The DeltaGrid end-of-block fold: cvt(sumi) + 0.125f*cvt(sumi1), then vfmacc by d_x*d_y.
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
// iq1_s has NO sign plane at all: neither an iq2-style signs table nor the vmul-onto-grid
// sign fold may appear. This is the load-bearing TernaryDelta distinction -- if a sign table
// or sign fold showed up, the emitter would be silently running the iq2 leaf.
// NOWALL-NOT: signs64
// NOWALL-NOT: signs256
// NOWALL-NOT: __riscv_vmul_vv_i8mf2
