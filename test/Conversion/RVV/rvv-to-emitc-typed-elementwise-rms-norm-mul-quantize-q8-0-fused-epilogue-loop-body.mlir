// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOSTORE --implicit-check-not={{"call_opaque \"__riscv_vse32"}} --implicit-check-not={{"call_opaque \"__riscv_vse16"}}
// RUN: sed 's/kind = "elementwise_quantize_q8_0_map"/kind = "elementwise_bogus_q8_0"/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADQUANTKIND
// RUN: sed 's/elementwise_quantize_q8_0_map %vz, %yq/elementwise_quantize_q8_0_map %vz, %z/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADQUANTOUT
// RUN: sed 's/strip_lmul = "m8"/strip_lmul = "m4"/; s/f32, "m8"/f32, "m4"/g' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADQUANTSTRIPLMUL

// G2 [FMT-PROP] tracer — the CONSTRUCTED fused rms_norm->mul->QUANTIZE chain: the
// llama attn_norm / ffn_norm epilogue (`ggml_rms_norm` -> `ggml_mul` against the
// learned weight) FUSED with the downstream activation quantize (`quantize_row_q8_0`
// producing the block_q8_0 our q4_0_q8_0 / q8_0_q8_0 block-dot kernels consume). It
// BUILDS the [FMT-PROP] FORMAT-PROPAGATION chain on TOP of the fused rms_norm->mul
// epilogue: the mul core brick (tcrv_rvv.elementwise_mul_map) now carries an OPTIONAL
// single-block $quant_epilogue region whose entry argument is the per-block WEIGHTED
// vector `vz` and which holds ONE tcrv_rvv.elementwise_quantize_q8_0_map consumer
// brick. The reduce-body emitter SPLICES the quantize into a SINGLE fused block loop:
// the normalized+weighted vz (a register-resident vfmul_vv result) flows STRAIGHT
// into the REUSED per-block ggml amax/scale/narrow q8_0 body and writes block_q8_0
// once.
//
// THE FUSION FACT (mechanically checked below): the f32 z[] intermediate is NEVER
// stored and NEVER reloaded. There is ZERO f32 vector store (no vse32/vse16 anywhere
// -- the NOSTORE run asserts it globally); the ONLY vector store is the single
// block_q8_0 vse8. The register-kept vz is the FIRST operand of BOTH the amax vfabs
// AND the scale vfmul_vf, so it is consumed in place -- the downstream INDEPENDENT
// quantize_row_q8_0 pass (the f32 activation store n*4 + the quantize reload n*4 =
// 2*n*4 bytes of DRAM round-trip) is ELIDED (honest STATIC byte accounting, not a
// measured bandwidth beat [NG-4]).
//
// BYTE-EXACT (strict, editable-golden argument): the scalar-double Sx^2 fold, the
// mean, and the 1/sqrtf(mean+eps) scale are IDENTICAL to plain rms_norm. The
// register-kept vz is bit-identical to a store-then-reload of the same f32 vector
// (IEEE binary32 round-trips losslessly through memory), so the amax reduction /
// scalar scale / f32->i16->i8 narrow (vfncvt = round-to-nearest-EVEN) run on the
// IDENTICAL block values the two-kernel path would read from z[] -- the fused
// block_q8_0 is byte-exact to rms_norm->mul (writing z[]) followed by a separate
// quantize_row_q8_0 (reading z[]), modulo ONLY the eliminated store/reload. This is
// a kernel-level equivalence [NG-2]; numerical bit-exact-vs-ggml is pending-hardware.

module {
  tcrv.exec.kernel @ggml_rms_norm_mul_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_rms_norm_mul_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "ne00", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %w = tcrv_rvv.runtime_abi_value {c_name = "w", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      // The f32 z[] intermediate the two-kernel path would materialize -- the fused
      // chain declares it (the mul epilogue ABI) but NEVER stores through it.
      %z = tcrv_rvv.runtime_abi_value {c_name = "z", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      // The block_q8_0 AoS byte buffer -- the actual fused-quant sink.
      %yq = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "uint8_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %eps = tcrv_rvv.runtime_abi_value {c_name = "eps", c_type = "float", ownership = "target-export-abi-owned", purpose = "eps", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_rms_norm_mul_q8_0, sew = 32 : i64, source_kernel = "ggml_rms_norm_mul_q8_0_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_elementwise_loop_body %x, %z, %n attributes {kind = "typed_elementwise_loop_body", reduce_map_model = "reduce", element_sew = 32 : i64} {
        ^bb0(%strip_index: index, %acc: f64):
          // The rms_norm reduce core PRODUCER carries the fused mul EPILOGUE, which
          // in turn carries the [FMT-PROP] fused-quant EPILOGUE: the mul's
          // $quant_epilogue region entry argument %vz is the per-block WEIGHTED
          // vector (the register-kept vfmul_vv result), and the single
          // tcrv_rvv.elementwise_quantize_q8_0_map consumer quantizes it to
          // block_q8_0 in the byte buffer %yq. anti-bypass: every brick's
          // strip_index is the loop induction variable (region arg 0); the mul
          // chain is %vy (rms epilogue arg 0); the quant chain is %vz (mul quant
          // epilogue arg 0). strip_lmul = "m8" pins the normalize + mul + quantize
          // block (the ggml QK8_0 e32m8 anchor).
          %acc_next = tcrv_rvv.elementwise_rms_norm_reduce_core %x, %z, %eps, %n strip %strip_index acc %acc {kind = "elementwise_rms_norm_reduce_core", strip_lmul = "m8"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, f64 -> f64 epilogue {
          ^bb0(%vy: !tcrv_rvv.vector<f32, "m8">):
            tcrv_rvv.elementwise_mul_map %vy, %w, %z, %n strip %strip_index {kind = "elementwise_mul_map"} : !tcrv_rvv.vector<f32, "m8">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index quant_epilogue {
            ^bb0(%vz: !tcrv_rvv.vector<f32, "m8">):
              tcrv_rvv.elementwise_quantize_q8_0_map %vz, %yq, %n strip %strip_index {kind = "elementwise_quantize_q8_0_map", qk = 32 : i64, block_stride = 34 : i64, scale_byte_offset = 0 : i64, quant_byte_offset = 2 : i64} : !tcrv_rvv.vector<f32, "m8">, !tcrv_rvv.runtime_abi_value, index, index
            }
          }
          tcrv_rvv.typed_elementwise_loop_yield %acc_next : f64
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// The fused TU still calls scalar libm (1/sqrtf(mean+eps)), so it self-includes
// <math.h> (keyed on the reduce core brick, unchanged by the quant epilogue).
// CHECK: emitc.include <"math.h">
// The kernel signature carries the block_q8_0 uint8_t* buffer (the fused-quant sink).
// CHECK: emitc.func @tcrv_emitc_ggml_rms_norm_mul_q8_0_kernel_ggml_rms_norm_mul_q8_0(
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"uint8_t">>
//
// ===== rms_norm reduce is BYTE-IDENTICAL to the unfused path =====
// The scalar-double accumulator + ascending fold (step is the literal 1, NOT a
// vlmax -- no vectorized vfredusum), then mean / sqrtf / scale.
// CHECK: %[[SUM:.*]] = "emitc.variable"{{.*}}lvalue<!emitc.opaque<"double">>
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: %[[PROD:.*]] = mul %{{.*}}, %{{.*}}{{.*}} -> !emitc.opaque<"float">
// CHECK: cast %[[PROD]] : !emitc.opaque<"float"> to !emitc.opaque<"double">
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"double">, !emitc.opaque<"double">)
// CHECK: div %{{.*}}, %{{.*}} : (!emitc.opaque<"double">, !emitc.opaque<"double">)
// CHECK: call_opaque "sqrtf"
// CHECK: %[[SCALE:.*]] = div %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
//
// ===== the FUSED normalize+mul+QUANTIZE block loop (the [FMT-PROP] splice) =====
// The AoS block count nb = n / 32 and the SINGLE fused block loop (one loop does
// normalize + mul + quantize per QK8_0 block -- no separate quantize pass).
// CHECK: route_source_op=tcrv_rvv.elementwise_quantize_q8_0_map
// CHECK: %[[C32:.*]] = literal "32" : !emitc.opaque<"size_t">
// CHECK: div %{{.*}}, %[[C32]] : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">)
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// The x[] block load + the normalize vfmul_vf(vx, scale) -> the register-kept vy.
// CHECK: call_opaque "__riscv_vle32_v_f32m8"
// CHECK: %[[VY:.*]] = call_opaque "__riscv_vfmul_vf_f32m8"(%{{.*}}, %[[SCALE]], %{{.*}})
// The mul epilogue provenance, the w[] block load, then the FUSED vfmul_vv whose
// FIRST operand is the register-kept vy (vy flows straight in -- no norm[] store).
// CHECK: route_source_op=tcrv_rvv.elementwise_mul_map
// CHECK: call_opaque "__riscv_vle32_v_f32m8"
// CHECK: %[[VZ:.*]] = call_opaque "__riscv_vfmul_vv_f32m8"(%[[VY]], %{{.*}}, %{{.*}})
// The quant epilogue provenance + the REUSED per-block amax/scale/narrow body on
// the register-kept vz. vz is the FIRST operand of the amax vfabs -- it flows
// straight into the quantizer with NO f32 z[] round-trip.
// CHECK: route_source_op=tcrv_rvv.elementwise_quantize_q8_0_map
// CHECK: call_opaque "__riscv_vfabs_v_f32m8"(%[[VZ]], %{{.*}})
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m1"
// CHECK: call_opaque "__riscv_vfredmax_vs_f32m8_f32m1"
// CHECK: call_opaque "__riscv_vfmv_f_s_f32m1_f32"
// The scalar d = amax / 127.0f and the load-bearing id = d ? 1/d : 0 conditional.
// CHECK: literal "127.0f"
// CHECK: cmp ne, %{{.*}}, %{{.*}}
// CHECK: if %{{.*}} {
// The native (_Float16)d AoS store at byte 0 (cast the uint8_t* cursor to
// _Float16 *, subscript [0], cast d to _Float16, assign).
// CHECK: cast %{{.*}} : !emitc.ptr<!emitc.opaque<"uint8_t">> to !emitc.ptr<!emitc.opaque<"_Float16">>
// CHECK: subscript
// CHECK: cast %{{.*}} : !emitc.opaque<"float"> to !emitc.opaque<"_Float16">
// CHECK: assign
// The scale by id consumes the SAME register-kept vz (vz is the first operand of
// the vfmul_vf), then the f32->i16->i8 narrow (vfncvt = round-to-nearest-even).
// CHECK: call_opaque "__riscv_vfmul_vf_f32m8"(%[[VZ]], %{{.*}}, %{{.*}})
// CHECK: call_opaque "__riscv_vfncvt_x_f_w_i16m4"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_i8m2"
// The SINGLE block_q8_0 store: the 32 int8 qs at AoS byte 2. This is the ONLY
// vector store -- the f32 z[] normalize/mul store is GONE (the NOSTORE run asserts
// no vse32/vse16 anywhere), and the independent quantize_row reload is elided.
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// CHECK: return
//
// The whole body is structured emitc nodes (variable/load/mul/cast/cmp/if/for/
// call_opaque), not a raw C blob; the provenance verbatims carry ALL THREE
// constructed bricks (the reduce core, the mul epilogue, AND the quantize
// epilogue), and NO opaque C blob leaks.
// CHECK-NOT: emitc.verbatim {{.*}}__riscv_v{{.*}};

// ===== the f32 z[] intermediate never lands in memory =====
// The block_q8_0 vse8 IS the only vector store; the --implicit-check-not bans
// vse32/vse16 globally, so no f32/f16 store exists anywhere in the fused kernel.
// NOSTORE: emitc.func @tcrv_emitc_ggml_rms_norm_mul_q8_0_kernel_ggml_rms_norm_mul_q8_0(
// NOSTORE: call_opaque "__riscv_vse8_v_i8m2"

// The [FMT-PROP] quant epilogue brick is fail-closed on its bounded kind, the
// block_q8_0 uint8_t* output binding, and the m8 QK8_0 strip requirement (the
// quantize block rides ggml's e32m8 anchor), enforced by the verifiers.
// BADQUANTKIND: currently supports only kind "elementwise_quantize_q8_0_map"
// BADQUANTOUT: requires the output operand to bind a runtime ABI value of C type 'uint8_t *'
// BADQUANTSTRIPLMUL: the fused quant epilogue requires the producer normalize strip LMUL to be "m8"
