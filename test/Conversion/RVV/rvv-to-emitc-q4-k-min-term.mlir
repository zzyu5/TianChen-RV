// RUN: tcrv-opt %s --split-input-file --tcrv-rvv-lower-to-emitc | FileCheck %s

// Track B q4_K BRICK 4 -- the q4_K/q5_K super-block MIN term (sumf -= dmin *
// sum_j(mins[j/2] * bsums[j])), as a FIRST-CLASS generic op
// (tcrv_rvv.q4_k_min_term), auto-constructing the offset-binary min subtraction
// instead of carrying it inside the monolithic q4_K block dot. ONE super-block:
// NO nb = n/256 loop, NO plain 4-bit nibble unpack (BRICK 1), NO 6-bit scale/min
// bit-dance (BRICK 2), NO scaled dot (BRICK 3), NO fp32 positive fold (deferred
// bricks). The op's lowering declares its OWN scalar sumf + a local sink, loads
// the fp32 activation scale dy once, runs the SCALAR integer reduction sumi =
// sum(bsums * mins) + the single fp `sumf -= dmin * sumi` emitc.expression, and
// stores the resulting sumf into the sink as the observable.
//
// The emitted MIN-term sequence is BYTE-IDENTICAL to the monolithic q4_K block
// dot's MIN term (the SAME emitQ4_KMinTermBsumsDot + emitQ4_KMinTermSubtract
// helpers emit both), modulo only the standalone-vs-in-loop construction
// differences (the op's own sumf decl + sink observable, the ABI input pointers
// vs the strided super-block bases). The MIN term is a SCALAR sequence -- the
// only opaque call is the fp16 dmin read.

module {
  tcrv.exec.kernel @q4_k_min_term_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @q4_k_min_term attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scales = tcrv_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q4_k_min_term, sew = 32 : i64, source_kernel = "q4_k_min_term_kernel", status = "selected-lowering-boundary"} {
        %mt = tcrv_rvv.q4_k_min_term %vx, %scales, %vy, %vl {kind = "q4_k_min_term", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, bsums_byte_offset = 260 : i64, weight_dmin_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-LABEL: emitc.func @tcrv_emitc_q4_k_min_term_kernel_q4_k_min_term(
// The op declares its OWN scalar float sumf accumulator (zeroed) + a function-
// scoped float sumf_out[1] sink it stores into.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: assign %{{.*}} : !emitc.opaque<"float"> to %[[SUMF]]
// CHECK: %[[SUMFOUT:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<1x!emitc.opaque<"float">>
// The fp32 activation scale dy = *(const float *)vy: a const float * cast + a
// subscript + a load (NO add -- activation d is at byte offset 0).
// CHECK: cast %{{.*}} : {{.*}} to !emitc.ptr<!emitc.opaque<"const float">>
// CHECK: subscript
// CHECK: load
// The MIN term FIRST half: bsums = (const int16_t *)(vy + 260), int sumi = 0,
// then the SCALAR integer reduction. NO RVV vector intrinsic appears for the
// reduction (it is a scalar subscript/load/cast/mul/add loop).
// CHECK: add %{{.*}} : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">)
// CHECK: cast %{{.*}} : {{.*}} to !emitc.ptr<!emitc.opaque<"const int16_t">>
// CHECK: %[[SUMI:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int">>
// CHECK: load %{{.*}} : <!emitc.opaque<"const int16_t">>
// CHECK: mul %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// CHECK: add %{{.*}} : (!emitc.opaque<"int">, !emitc.opaque<"int">)
// CHECK: assign %{{.*}} : !emitc.opaque<"int"> to %[[SUMI]]
// The MIN term SECOND half: the fp16 dmin read (the ONLY opaque call), the
// dmin = dmx * dy product (dy is a const-float load), then the `sumf -= dmin *
// sumi` as ONE emitc.expression (a cast + a mul + a sub), assigned back into sumf.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: mul %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"const float">)
// CHECK: expression : !emitc.opaque<"float">
// CHECK: cast %{{.*}} : !emitc.opaque<"int"> to !emitc.opaque<"float">
// CHECK: mul %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: sub %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: yield
// CHECK: assign %{{.*}} : !emitc.opaque<"float"> to %[[SUMF]]
// The store_sumf observable: sumf_out[0] = sumf.
// CHECK: subscript %[[SUMFOUT]]
// CHECK: assign
// CHECK: return
