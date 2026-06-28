// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml q8_0 16x1-REPACKED GEMV (decode) hot kernel as STRUCTURED emitc IR
// (I5; ZERO raw() strings). The single typed op tcrv_rvv.repack_gemv_q8_0_q8_0
// lowers to the BLOCK-AS-LANE single-output-column matmul: the 16 interleaved
// weight rows of a group occupy 16 vector lanes, the dot accumulates LANE-WISE
// (NO cross-lane vredsum wall), and at VLEN=128 each 16-block group is processed
// as TWO disjoint contiguous 8-lane halves (rows 0..7 then rows 8..15) inline
// (NO half loop, NO nr/4 row loop). The activation is ONE plain block_q8_0 stream
// (stride 34, quants at +2) -- a single activation column.
//
// q8_0 weights are FULL int8 (block_q8_0x16: 16 fp16 scales = 32 bytes + 16*32 =
// 512 int8 quant bytes, stride 544), so the weight lane is read DIRECTLY (NO
// nibble vsll/vsra sign-extend, NO ^0x88, NO lo/hi split). The contraction is 32
// positions (one int8 weight byte each), and because full int8 products overflow
// i16 (127*127*3 > 32767) the in-block accumulation is i32: vwmul_vx (i8xi8 ->
// i16 product) then vwadd_wv (i32_acc += widened i16). The dual-fp16 scale fold
// (d_x*d_y) is IDENTICAL to q4_0. Integer accumulation is order-independent so
// the dot is byte-exact vs a scalar/ggml q8_0 reference.
//
// Every emitted value is a NODE in the IR graph -- no emitc.verbatim with C
// control flow, no raw string blob.

module {
  tcrv.exec.kernel @ggml_repack_gemv_q8_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_q8_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q8_0_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q8_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_q8_0_q8_0 %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q8_0_q8_0", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 544 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemv_q8_0_q8_0 %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_q8_0_q8_0_kernel_ggml_repack_gemv_q8_0_q8_0(
// The block count nb = n / 32.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The column-group count nc/16 (%arg4 = nc). There is NO nr/4 row loop.
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*544 (block_q8_0x16 stride 544).
// CHECK: mul %[[X]], %{{.*}}
// CHECK: literal "544"
// The two 8-lane f32 accumulators (rows 0..7, 8..15): two vfmv_v_f_f32m2(0.0f, 8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The per-block plain q8_0 activation base al = a + l*34 (stride 34).
// CHECK: literal "34"
// ONE i32 accumulator per strip seeded vmv_v_x_i32m2(0, 8) (sumi_a, sumi_b) --
// NOT q4_0's i16 lo/hi seeds (full int8 products overflow i16).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// The 32-position contraction loop over the 32 int8 weight bytes per block.
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// Two disjoint repacked int8 sub-loads (rows 0..7 at qs[i*16+0], rows 8..15 at
// qs[i*16+8]) -- read DIRECTLY as int8, NO vsll/vsra nibble decode.
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// The lane-wise vwmul product (i8 x i8 -> i16) against the SINGLE-column scalar
// q8 quant qs[i] (NO lo/hi, NO cross-lane vredsum), then the i32 in-block
// accumulate vwadd_wv (i32 += widened i16), for both 8-lane halves.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// The vector scale fold: vle16 the two 8-lane weight scale halves (b[l].d[0]
// and b[l].d[8]), read the SINGLE raw _Float16 activation scale once, then per
// half vfwmul / vfcvt the i32 accumulator / vfmacc into the f32 accumulator.
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// The single-column two-half vector store vse32_v_f32m2 (s+x*16+0, s+x*16+8;
// NO bs row stride, NO per-block scalar *s store).
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall: the
// dot accumulates LANE-WISE (vwmul + vwadd_wv), so NO vredsum / vwredsum (any
// SEW or signedness) appears anywhere in the emitted GEMV.
// NOWALL-NOT: redsum
// q8_0 weights are FULL int8, so the nibble-decode machinery the q4_0 sibling
// emits is ABSENT by construction: NO vsll/vsra sign-extend, NO offset-binary
// vxor, NO i16-accumulating vwmacc, and NO end-of-block vwadd_vv lo/hi combine
// (q8_0 accumulates in-block via vwadd_wv into i32, never i16). A regression that
// reintroduced the nibble path would re-emit one of these.
// NOWALL-NOT: vsll
// NOWALL-NOT: vsra
// NOWALL-NOT: vxor
// NOWALL-NOT: vwmacc
// NOWALL-NOT: vwadd_vv
