// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=SEL1
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=REALIZE

// [G3 主线C / SEL-1] T3 gate-(7) capability-attribution proof: the SP4
// (tiled-vs-plain) output-tiling choice for q4_K is no longer a COMPILE-TIME per-format
// hardcode inside emitTypedRepackGemmLoopBody -- it is a RUNTIME CAPABILITY-KEYED
// selection. The front-door pass, at rv64gcv (derived VLEN 128, 32 architectural
// vregs), classifies the q4_K loop-body's fold_model "kquant_dmin_bsums_min" as the
// min-fold register-cliff BOTTLENECK SHAPE (the KEY is the shape, NOT the format name),
// intersects the implemented-body set with target capability and obtains the singleton
// {S6Tiled} (reason=only_feasible). Historical SP4 A/B rows remain evidence but do not
// rank S6 against an unrealized Plain body. The chosen variant + its reason are stamped for the
// emitter's pure realize; the full [D-4] attribution record (reusing the SAME
// computeDeclaredInstanceHash) rides an inert in-IR attr. The realize path emits the
// byte-exact S6-tiled q4_K GEMM body (identical to the retired direct emitter). The
// Plain singleton paths are proven in rvv-sel1-t3-tiling-rollout-gate7.mlir.

module {
  weft.exec.kernel @ggml_repack_gemm_q4_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q4_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q4_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q4_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 16 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ===================== gate-(7) capability-keyed attribution =====================
// The abstract op is GONE; the compiler CONSTRUCTED the q4_K min-fold GEMM region and
// STAMPED the SP4 tiling decision (the loop-body op's attrs print alphabetically).
// The loop-body op prints its attrs alphabetically, so the SEL1-SAME chain tracks
// that order: fold_model (the KEY) < weft_rvv.tiling_selection_reason < ...record <
// ...variant.
// SEL1-NOT: weft_rvv.quant_contraction
// SEL1: weft_rvv.typed_repack_gemm_loop_body
// The fold_model that KEYED the selection (the bottleneck SHAPE, not the format name).
// SEL1-SAME: fold_model = "kquant_dmin_bsums_min"
// [M1c] the loop-order schedule axis (attrs print alphabetically, so loop_order < tiling):
// the M1b col-outer decision, lifted from emitter-inline to the SEL-1 selector, keyed on
// the repack layout STRIDE fact (weight x16 panel 2304 >= activation x4 panel 1168) +
// the prefill regime. q4_K carries the M1b-board offline A/B seed => reason=measured.
// SEL1-SAME: weft_rvv.loop_order = "col_outer"
// SEL1-SAME: weft_rvv.loop_order_selection_reason = "measured"
// SEL1-SAME: weft_rvv.loop_order_selection_record = "{{.*}}candidates{{.*}}row_outer{{.*}}col_outer
// Legality leaves the one implemented S6Tiled body.
// SEL1-SAME: weft_rvv.tiling_selection_reason = "only_feasible"
// The bounded candidate record contains that singleton; the [D-4] JSONL
// attribution record (reused FORM + declared_instance_hash) is inert + in-IR.
// SEL1-SAME: weft_rvv.tiling_selection_record = "{{.*}}candidates{{.*}}s6_tiled{{.*}}reason{{.*}}only_feasible
// SEL1-SAME: weft_rvv.tiling_variant = "s6_tiled"
// The decision is capability/measurement-DERIVED, never the capability-blind fallback.
// SEL1-NOT: static_order

// ===================== pure realize -> byte-exact S6-tiled body ==================
// The emitter degenerates to a pure realize: it reads weft_rvv.tiling_variant =
// "s6_tiled" and emits the byte-exact S6-tiled q4_K GEMM body (the same emit as
// rvv-emit-quant-contraction-q4-K-repack-gemm-prefill-vlen128).
// REALIZE-NOT: weft_rvv.quant_contraction
// REALIZE-NOT: weft_rvv.typed_repack_gemm_loop_body
// REALIZE: emitc.func @weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
// REALIZE: literal "2304"
// REALIZE: call_opaque "__riscv_vfmv_v_f_f32m2"
// REALIZE: call_opaque "__riscv_vwmacc_vv_i32m2"
// REALIZE: call_opaque "__riscv_vfnmsac_vv_f32m2"
// REALIZE: return
