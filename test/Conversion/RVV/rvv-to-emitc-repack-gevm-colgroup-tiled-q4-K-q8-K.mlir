// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL
// RUN: weft-opt %s | weft-opt | FileCheck %s --check-prefix=ROUNDTRIP

// G7 L1 P1: the NEW, INDEPENDENT q4_K@rvv GEVM Emission Plan ([K-10] structural-level ·
// [PAT-2] P9 GEMV-regime-plan-streaming). This is a DISTINCT plan op
// (weft_rvv.typed_repack_gemv_colgroup_tiled_loop_body) -- NOT a knob on
// weft_rvv.typed_repack_gemv_loop_body -- because the iteration-space topology is
// STRUCTURALLY different ([K-10] 硬禁: 结构级差异禁实现为 plan 内旋钮). Where the sibling
// GEVM plan is column-group-OUTER / block-INNER (activation re-addressed per column-group),
// THIS plan: (1) TILES the weight-column-group loop by column_group_tile (TG=2 here);
// (2) makes the contraction-BLOCK loop the SHARED MIDDLE loop -- ONE q8_K activation block
// base a+l*292 + delta d_y is computed ONCE per block and REUSED across the TG groups;
// (3) keeps a REGISTER-RESIDENT f32 accumulator BANK (TG*numHalves = 4 strips) live across
// the block stream; (4) carries a weight-strip PREFETCH cadence in the structure. The
// per-block q4_K super-block decode + fold leaf is UNCHANGED and byte-exact to the sibling
// GEVM / GEMM-M=1 body (the host ZERO-MODEL oracle
// test/Target/RVV/q4-K-q8-K-repack-gevm-colgroup-tiled-oracle.c proves mismatch=0). The
// front door carries the block_index-tied (anti-bypass) repack_gemv_kquant_core brick
// (decode_model "q4_K"); the lowering RE-EMITS the byte-exact q4_K body via
// emitTypedRepackGemvColgroupTiledLoopBody -> emitRepackKQuantGemvColgroupTiledBodyQ4K.
// VLEN=128 => TWO disjoint 8-lane strips per group => 4 resident accumulators for TG=2.

module {
  weft.exec.kernel @ggml_repack_gevm_ct_q4_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gevm_ct_q4_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gevm_ct_q4_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gevm_ct_q4_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The DISTINGUISHING structural attr: column_group_tile = 2 (TG groups per tile).
        weft_rvv.typed_repack_gemv_colgroup_tiled_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_colgroup_tiled_loop_body", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks", qk = 256 : i64, weight_block_stride = 2304 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 256 : i64, activation_quant_byte_offset = 4 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, activation_bsums_byte_offset = 260 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, fold_model = "kquant_dmin_bsums_min", column_group_tile = 2 : i64} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          %sumi:2 = weft_rvv.repack_gemv_kquant_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_kquant_core", decode_model = "q4_K", weight_quant_byte_offset = 256 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// The independent plan op survives an IR round-trip (parses + verifies + prints).
// ROUNDTRIP: weft_rvv.typed_repack_gemv_colgroup_tiled_loop_body
// ROUNDTRIP: column_group_tile = 2

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemv_kquant_core %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gevm_ct_q4_K_q8_K_kernel_ggml_repack_gevm_ct_q4_K_q8_K(
// The block count nb = n / 256 (QK_K).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The column-group count nc/16 (%arg4 = nc).
// CHECK: div %arg4, %[[G:.*]] : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The DISTINGUISHING structural division: the number of column-group TILES = ncGroups
// / column_group_tile (TG). This is the plan's [K-10] iteration-space-topology fact
// that the sibling per-column GEVM plan does NOT emit.
// CHECK: div %[[NCG:.*]], %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The REGISTER-RESIDENT accumulator BANK: TG(2) column-groups * numHalves(2) strips = 4
// f32 accumulators seeded vfmv_v_f(0.0f) BEFORE the shared block stream.
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The shared contraction-BLOCK stream: ONE q8_K activation super-block delta d_y read
// (*(const float *)) per block -- REUSED across the TG column-groups of the tile (the
// [PAT-2] "交织组多列共享输入向量加载" the per-column plan lacks).
// CHECK: call_opaque "*(const float *)"
// The PREFETCH cadence in the structure ([PAT-2] P9): the next block's weight strips are
// hinted for each column-group of the tile.
// CHECK: call_opaque "__builtin_prefetch"
// The byte-exact q4_K super-block leaf is REUSED unchanged: the 6-bit scale/min lane-wise
// unpack (vle8 / vand / vsrl / vsll / vor / vzext), the bsums-min correction (vwmacc_vx),
// the split-32 main dot (vwmacc_vv), and the dual d/dmin fp16 fold (vfmacc then vfnmsac).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfnmsac_vv_f32m2"
// The per-tile store: TG(2) groups * numHalves(2) strips = 4 lane-wise vse32.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall: the dot
// accumulates LANE-WISE via vwmacc, so NO vredsum / vwredsum appears (same as the
// sibling per-column GEVM plan; the tiling never reintroduces a horizontal fold).
// NOWALL-NOT: redsum
