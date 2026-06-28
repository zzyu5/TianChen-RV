// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=PURE

// option-2 stage-C1b: the ISOLATED bit-exact PACKER. The single typed op
// tcrv_rvv.pack_q4_0_to_q4_0x16 lowers to ggml's make_block_q4_0x16 PACK
// (plain block_q4_0 stride 18 -> block_q4_0x16 stride 288) as a PURE scalar
// byte gather + ^0x88 XOR (the live blck_size_interleave==1 branch): NO vl/LMUL,
// NO vector intrinsics on the pack data path, NO fp arithmetic. It is the
// PRODUCER sibling of the block-as-lane CONSUMER repack-GEMV/GEMM emitters --
// the compiler PRODUCES the x16 layout it DECLARES. Host byte-exact vs ggml
// (memcmp==0); e2e-REDUNDANT (ggml packs at load); NEVER a kernel/perf/e2e win.
//
// The pack body is structured emitc (emitc.for / subscript / bitwise_xor) --
// every value is a NODE, NO emitc.verbatim with C control flow.

module {
  tcrv.exec.kernel @ggml_pack_q4_0_to_q4_0x16_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_pack_q4_0_to_q4_0x16 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %nblocks = tcrv_rvv.runtime_abi_value {c_name = "nblocks", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nblocks", role = "runtime-element-count"} : index
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-plain-src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "uint8_t *", ownership = "target-export-abi-owned", purpose = "q4x16-dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %nblocks {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_pack_q4_0_to_q4_0x16, sew = 32 : i64, source_kernel = "ggml_pack_q4_0_to_q4_0x16_kernel", status = "selected-lowering-boundary"} {
        %p = tcrv_rvv.pack_q4_0_to_q4_0x16 %src, %dst, %nblocks, %vl {kind = "ggml_pack_q4_0_to_q4_0x16", qk = 32 : i64, src_block_stride = 18 : i64, dst_block_stride = 288 : i64, src_quant_byte_offset = 2 : i64, dst_quant_byte_offset = 32 : i64, weight_interleave = 16 : i64, xor_mask = 136 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.pack_q4_0_to_q4_0x16 %
// CHECK-NOT: unrealized_conversion_cast
// The emitted pack function. ABI follows the runtime_abi_value definition order
// (nblocks: size_t, src: const uint8_t*, dst: uint8_t*).
// CHECK: emitc.func @tcrv_emitc_ggml_pack_q4_0_to_q4_0x16_kernel_ggml_pack_q4_0_to_q4_0x16(%arg0: !emitc.opaque<"size_t">, %arg1: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg2: !emitc.ptr<!emitc.opaque<"uint8_t">>)
// The outer pack-block loop over nblocks (%arg0).
// CHECK: for %[[B:.*]] = %{{.*}} to %arg0 step
// Per-block source/dest bases: b * (16*18) and b * 288 (== 288 each).
// CHECK: literal "288"
// CHECK: mul %[[B]], %{{.*}}
// CHECK: literal "288"
// CHECK: mul %[[B]], %{{.*}}
// The scale loop: 16 fp16 d (src stride 18) copied VERBATIM, NO xor.
// CHECK: for %[[J:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "18"
// CHECK: subscript %arg1
// CHECK: load
// CHECK: subscript %arg2
// CHECK: assign
// The ^0x88 (136) bias literal and the quant base dbase + 32 (16 fp16 scales
// precede the 256 interleaved nibble bytes).
// CHECK: literal "136"
// CHECK: literal "32"
// The 16-way interleave loops (nibble offset 0..16 outer, block 0..16 inner).
// CHECK: for %[[OFF:.*]] = %{{.*}} to %{{.*}} step
// CHECK: for %[[BLK:.*]] = %{{.*}} to %{{.*}} step
// The per-block src gather (stride 18) then ^0x88 then dst store.
// CHECK: literal "18"
// CHECK: subscript %arg1
// CHECK: load
// CHECK: bitwise_xor
// CHECK: subscript %arg2
// CHECK: assign

// PURITY: the pack data path is PURE scalar -- NO vector intrinsics, NO fp, NO
// vredsum. The ONLY __riscv_ token allowed is the inert with_vl setvl wrapper
// (its result is unused by the scalar pack body). bitwise_xor / subscript /
// load / assign carry the whole transform.
// PURE-NOT: vwmacc
// PURE-NOT: vredsum
// PURE-NOT: vfwmul
// PURE-NOT: vle8
// PURE-NOT: vsra
// PURE-NOT: vsll
