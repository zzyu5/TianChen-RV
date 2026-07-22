// FULL production-export CLOSURE for the q4_0 16x1-REPACKED GEVM -- the option-2
// quant_contraction BRIDGE route brought through the SAME "source function ->
// object export" front door the flat / super-block block-dots use. Unlike those
// SOURCE-FUNC front doors, the repacked GEVM is CONSTRUCTED by
// --weft-rvv-lower-quant-contraction (it lowers the abstract
// weft_rvv.quant_contraction inside the selected variant's with_vl into the typed
// weft_rvv.typed_repack_gemv_loop_body REGION; the monolithic
// weft_rvv.repack_gemv_q4_0_q8_0 op is retired). That single plugin-owned typed
// body then flows through the COMPLETE weft-source-artifact-front-door-pipeline
// (materialize-emission-plans PLUS --weft-check-execution-plan-coherence) AND
// exports a real RISC-V RVV relocatable object through weft-translate
// --weft-export-target-artifact.
//
// This is the LINE-B wiring: the repacked GEVM is a GEVM (writes through the
// output pointer, internalizes the N loop), NOT a block-dot, so it takes its OWN
// honest monolithic route family id
// 'rvv-ggml-repack-gemv-monolithic-emitc-route-family' (NOT the flat/super-block
// block-dot route ids) and carries the 5-role GEVM ABI (n, s, bs, vx, vy). It
// shares the shared monolithic emission-plan + object-export MECHANISM
// (RVVMonolithicBlockDotFamily.h) -- the flat/super-block routes stay byte-exact
// (they never carry this route id and never reach the RepackGemv exporter).
//
// BYTE-EXACT: the object is packaged from the exact CORE EmitC (the same
// tryConvertConstructedModuleWithRegisteredBackend / convertRVVModuleToEmitC lowering the
// direct --weft-rvv-lower-to-emitc path uses), so the production-export emit is
// byte-identical to the CORE == emission-plans emit (asserted below by diff). The
// exported function symbol is the kernel+variant handoff name. NO board / NO perf
// claim -- coverage/wiring maturity; numbers pending-hardware (ssh rvv). The
// repacked GEVM's dual-fp16 scale fold loads the per-strip fp16 scales as a VECTOR
// (vle16_v_f16m2 -- the repack locality win), a Zvfh op, so the RepackGemv object
// packages under -march=rv64gcv_zvfh (its own packager; the block-dot families
// keep the baseline rv64gcv packager byte-identical).
//
// clang for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: weft-local-rvv-object-clang

// CONSTRUCTION: the front door (VLEN128 => Zvl128b => repack SELECTED) constructs
// the typed region + its decomposed bricks from the REAL quant_contraction request
// (NOT test-authored).
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT

// CORE emit: the constructed region lowers to the m1 / half_lanes=16 single-16-lane-
// strip repacked GEVM kernel (integer_core_lmul="m1", measured accumulator-LMUL gate).
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT

// FULL pipeline: after construction, the weft-source-artifact-front-door-pipeline
// materializes the emission plan AND passes --weft-check-execution-plan-coherence
// (the RepackGemv monolithic route id is a registered target-artifact export
// route).
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

// BYTE-EXACT: --weft-materialize-emission-plans only APPENDS the emission-plan
// diagnostic mirror; the typed repack body is untouched, so the production-export
// EmitC is byte-for-byte the CORE --weft-rvv-lower-to-emitc emit.
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc > %t.core.mlir
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-materialize-emission-plans --weft-rvv-lower-to-emitc > %t.prod.mlir
// RUN: diff %t.core.mlir %t.prod.mlir

// Target-artifact OBJECT export: the RepackGemv monolithic emission plan exports a
// real RISC-V RVV relocatable object through the registered peer object exporter.
// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL

module {
  weft.exec.kernel @ggml_vec_dot_q4_0_q8_0_repack_gemv_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @ggml_vec_dot_q4_0_q8_0_repack_gemv attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_0_q8_0_repack_gemv, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_0_q8_0_repack_gemv_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q4_0", scale_model = "dual-fp16-per-block-d_x.d_y", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
    weft.exec.variant @ggml_vec_dot_q4_0_q8_0_repack_gemv_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @ggml_vec_dot_q4_0_q8_0_repack_gemv {origin = "rvv-plugin", policy = "rvv-q4-0-q8-0-repack-gemv-bridge-case"}
      weft.exec.fallback @ggml_vec_dot_q4_0_q8_0_repack_gemv_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin"}
    }
  }
}

// ===================== CONSTRUCTION (post-bridge IR) =========================
// The abstract quant_contraction op is GONE; the repack-SELECTED cell is
// CONSTRUCTED as the typed region + its decomposed bricks, carrying the
// block_q4_0x16 x16 facts + the DECLARED OUTPUT CONTRACT. NO monolithic repack op,
// NO block-dot op.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// CONSTRUCT-NOT: weft_rvv.repack_gemv_q4_0_q8_0
// CONSTRUCT-NOT: weft_rvv.q4_0_q8_0_block_dot
// CONSTRUCT: weft_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: half_lanes = 16 : i64
// CONSTRUCT-SAME: integer_core_lmul = "m1"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 288 : i64
// CONSTRUCT: weft_rvv.repack_lane_wise_q4_x_i8_dot
// CONSTRUCT: weft_rvv.repack_dual_fp16_scale_fold
// CONSTRUCT: weft_rvv.typed_repack_gemv_loop_yield

// ===================== CORE EmitC =============================================
// The single 16-lane f32m4 accumulator, the single contiguous repacked i8m1 sub-load,
// the lane-wise vwmacc accumulate (NO cross-lane reduction wall), and the single
// per-strip vector store.
// EMIT-NOT: unrealized_conversion_cast
// EMIT: emitc.func @weft_emitc_ggml_vec_dot_q4_0_q8_0_repack_gemv_kernel_ggml_vec_dot_q4_0_q8_0_repack_gemv(
// EMIT: literal "288"
// EMIT: call_opaque "__riscv_vfmv_v_f_f32m4"
// EMIT: call_opaque "__riscv_vle8_v_i8m1"
// EMIT: call_opaque "__riscv_vwmacc_vx_i16m2"
// EMIT: call_opaque "__riscv_vwmacc_vx_i16m2"
// EMIT: call_opaque "__riscv_vse32_v_f32m4"
// EMIT: return

// ===================== FULL-PIPELINE COHERENCE (post-coherence IR) ============
// The kernel survived coherence with exactly the supported monolithic
// emission-plan diagnostic naming the RepackGemv monolithic route id + object
// kind.
// PLAN: weft.exec.kernel @ggml_vec_dot_q4_0_q8_0_repack_gemv_kernel
// The constructed body is the typed repacked-GEVM LOOP body op; it exports through
// the RepackGemv monolithic plan (its own route id / kind / 5-role GEVM ABI).
// PLAN: weft_rvv.typed_repack_gemv_loop_body
// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// The repacked-GEVM carries the RepackGemv op-derived metadata keys (rendered
// inside artifact_metadata, ahead of lowering_pipeline).
// PLAN-SAME: rvv_ggml_repack_gemv_kind
// The honest RepackGemv monolithic-body route id (NOT the flat/super-block
// block-dot routes, NOT the decomposed generic-typed-body route) is the
// coherence-recognized export route.
// PLAN-SAME: lowering_pipeline = "rvv-ggml-repack-gemv-monolithic-emitc-route-family"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @ggml_vec_dot_q4_0_q8_0_repack_gemv
// The repacked-GEVM honestly carries NO decomposed-route slice config metadata,
// and never claims a block-dot route.
// PLAN-NOT: rvv_selected_body_operation
// PLAN-NOT: value = "rvv-generic-typed-body-emitc-route-family"
// PLAN-NOT: rvv-ggml-super-block-block-dot-monolithic-emitc-route-family
// PLAN-NOT: rvv-ggml-flat-block-dot-monolithic-emitc-route-family

// ===================== EXPORTED RISC-V RVV OBJECT ============================
// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// The exported function symbol is the kernel+variant handoff name -- the same name
// the CORE EmitC emit carries (weft_emitc_<kernel>_<variant>).
// SYMBOL: Name: weft_emitc_ggml_vec_dot_q4_0_q8_0_repack_gemv_kernel_ggml_vec_dot_q4_0_q8_0_repack_gemv
