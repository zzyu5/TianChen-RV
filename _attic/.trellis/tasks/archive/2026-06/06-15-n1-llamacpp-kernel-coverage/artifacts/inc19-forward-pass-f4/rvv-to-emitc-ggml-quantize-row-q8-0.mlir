// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// INC-19 F4 — the COMPLETE ggml `quantize_row_q8_0` RVV-path forward-pass op
// (the f32 -> block_q8_0 ACTIVATION QUANTIZER; riscv/quants.c:32-71) as
// STRUCTURED emitc IR (I5; ZERO raw() strings). The single typed op
// tcrv_rvv.quantize_row_q8_0 lowers to an AoS block loop (nb = n/32) whose body
// per block: loads the 32 f32 lanes in ONE e32m8 strip (vl=32), takes amax via
// vfabs + vfredmax, computes the scalar d = amax/127 and the load-bearing
// id = d ? 1/d : 0 (a STRUCTURED emitc.cmp + emitc.if), stores the native
// (_Float16)d at AoS byte 0, scales by id with vfmul_vf, narrows f32->i16->i8
// with vfncvt (round-to-nearest-EVEN) + vncvt, and stores the 32 int8 qs at AoS
// byte 2. This is the f32 -> QUANT BRIDGE: it produces the block_q8_0 activation
// our committed q4_0_q8_0 / q8_0_q8_0 block-dot kernels CONSUME.
//
// BYTE-EXACTNESS matches ggml's EXACT RVV method (vfncvt = rne + the native
// _Float16 conversion), NOT the scalar `_ref` (roundf). Every emitted value is a
// NODE in the IR graph -- no emitc.verbatim with C control flow, no raw string
// blob. Pinned by the ssh-rvv artifact under
// .trellis/tasks/.../artifacts/inc19-forward-pass-f4/.

module {
  tcrv.exec.kernel @quantize_row_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @quantize_row_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "uint8_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @quantize_row_q8_0, sew = 32 : i64, source_kernel = "quantize_row_q8_0_kernel", status = "selected-lowering-boundary"} {
        %q = tcrv_rvv.quantize_row_q8_0 %x, %vy, %n, %vl {kind = "ggml_quantize_row_q8_0", qk = 32 : i64, block_stride = 34 : i64, scale_byte_offset = 0 : i64, quant_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_quantize_row_q8_0_kernel_quantize_row_q8_0(
// The AoS block count nb = n / 32 and the block loop.
// CHECK: div {{.*}}, %{{.*}}
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// The 32 f32 lanes in one e32m8 strip; vfabs + vfredmax amax reduction.
// CHECK: call_opaque "__riscv_vle32_v_f32m8"
// CHECK: call_opaque "__riscv_vfabs_v_f32m8"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m1"
// CHECK: call_opaque "__riscv_vfredmax_vs_f32m8_f32m1"
// CHECK: call_opaque "__riscv_vfmv_f_s_f32m1_f32"
// The scalar d = amax / 127.0f and the load-bearing id = d ? 1/d : 0 conditional.
// CHECK: literal "127.0f"
// CHECK: div %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: cmp ne, %{{.*}}, %{{.*}}
// CHECK: if %{{.*}} {
// The native (_Float16)d AoS store at byte 0 (cast to _Float16 *, subscript [0],
// cast d to _Float16, assign).
// CHECK: cast %{{.*}} : !emitc.ptr<!emitc.opaque<"uint8_t">> to !emitc.ptr<!emitc.opaque<"_Float16">>
// CHECK: subscript
// CHECK: cast %{{.*}} : !emitc.opaque<"float"> to !emitc.opaque<"_Float16">
// CHECK: assign
// The scale + f32->i16->i8 narrowing convert (vfncvt = round-to-nearest-even).
// CHECK: call_opaque "__riscv_vfmul_vf_f32m8"
// CHECK: call_opaque "__riscv_vfncvt_x_f_w_i16m4"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_i8m2"
// The 32 int8 qs store at AoS byte 2.
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// The whole kernel is structured emitc nodes -- the block loop is emitc.for, the
// id conditional emitc.if, the intrinsics emitc.call_opaque. No raw C blob: the
// provenance verbatims are comment lines only.
// CHECK-NOT: emitc.verbatim {{.*}}__riscv
