// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml IQ3_XXS x Q8_K GRID-codebook super-block dot-product
// (tcrv_rvv.iq3_xxs_q8_k_block_dot, the iq3 GRID-of-4 variant) lowers to the COMPLETE
// structured kernel (I5; zero raw(), every value an emitc node): the outer super-block
// loop over nb = n / 256, the per-super-block address arithmetic (weight stride 98,
// activation stride 292), the d = fp16(x.d) * fp32(y.d) scale, an int32 bsum reset per
// super-block, a FLAT loop over 8 sub-blocks each reassembling aux32 from 4 little-
// endian byte loads of the SEPARATE gas region (xb + 66), the 4-bit scale ls =
// 2*(aux32>>28)+1, and a loop over 4 sign groups each doing TWO grid-of-4 passes (the
// iq3 delta): each pass an indexed vle8(4) over grid_i8 + idx*4, the SIGN-plane
// application (broadcast signs / vand kmaskLo-or-kmaskHi / vmsne / vmerge with vneg
// grid), the signed widening product, and the chained vwredsum into sumi. Then bsum +=
// sumi*ls (integer domain), the per-super-block fold sumf += d*(float)bsum (ONE
// expression), and *s = 0.25f*sumf. The grid + ksigns tables are emitted as structured
// static const decls (NOT vrgather: the grid is byte-viewed via (const int8_t *)).

module {
  tcrv.exec.kernel @ggml_vec_dot_iq3_xxs_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_iq3_xxs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq3xxs-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8k-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_iq3_xxs_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_iq3_xxs_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.iq3_xxs_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_iq3_xxs_q8_k_block_dot", scale_model = "per-group-int4-grid-of-4-codebook-scale-int-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 98 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_qs_byte_offset = 2 : i64, weight_gas_byte_offset = 66 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, grid = array<i32: 67372036, 67372052, 67372068, 67374092, 67374108, 67374142, 67376132, 67376148, 67378188, 67380244, 67386908, 67386924, 67896332, 67896348, 67898372, 67898388, 67900428, 67900460, 67902468, 67902484, 67904524, 67906596, 67911172, 68420612, 68420628, 68420644, 68422668, 68424708, 68424724, 68426764, 68426780, 68426814, 68430860, 68430910, 68435500, 68944908, 68944958, 68946948, 68946964, 68949036, 68959748, 69471260, 69475390, 69477412, 69479486, 69484060, 69484076, 69993484, 69993534, 69999636, 70003732, 70523948, 70530084, 71175172, 71175204, 71175220, 71181340, 71185420, 201589772, 201589788, 201591812, 201591828, 201593868, 201593884, 201595908, 201595924, 201595940, 201598014, 201600004, 202114052, 202114068, 202116108, 202118148, 202118164, 202638348, 202638364, 202640388, 202640404, 202642444, 202644484, 202653204, 203162628, 203162644, 203166724, 203168780, 203170868, 203174964, 203686924, 203686956, 203697156, 204215300, 204215332, 204219444, 204226060, 204735532, 205394964, 205399044, 335807492, 335807508, 335809548, 335809564, 335811588, 335811604, 335811636, 335813644, 335815700, 336331788, 336331804, 336331820, 336333828, 336333844, 336335884, 336337924, 336344092, 336344126, 336346628, 336856068, 336856084, 336858124, 336858174, 336860164, 336860180, 336862270, 336864260, 336866348, 337380364, 337382404, 337382436, 337395204, 337395236, 337910828, 337914908, 338428956, 338433086, 338437132, 338443812, 339608588, 339608604, 339610676, 339616812, 470025228, 470027268, 470027284, 470029324, 470029340, 470035460, 470037548, 470040084, 470549508, 470549524, 470553604, 470555660, 470557732, 470557748, 471073804, 471073820, 471075844, 471077932, 471084052, 471088660, 471600140, 471604252, 472128516, 472130622, 472137236, 472646660, 472646708, 472650772, 472656940, 473173028, 473177140, 473183260, 473832476, 473838596, 604242980, 604245054, 604249132, 604249150, 604253212, 604253246, 604782116, 605295620, 605297726, 605299716, 605303812, 605303860, 605815870, 605824044, 606340132, 606350348, 606352420, 606868524, 606872604, 606879236, 608044076, 608046084, 608046100, 608050180, 738462740, 738468876, 738475524, 738984964, 738985012, 738989108, 738995244, 739511332, 739515412, 739524116, 740033556, 740043804, 740559876, 740561948, 740561982, 740572692, 741082132, 741088268, 741616644, 742265892, 742269972, 872682532, 872686628, 872686644, 872690724, 873206796, 873214988, 873729086, 873739300, 874257412, 874257460, 874783780, 875299884, 875310100, 875830300, 876479516, 876483596, 1040450588, 1040450604, 1040450622, 1040452612, 1040456724, 1040460820, 1040978996, 1040983044, 1041501204, 1041507372, 1041509396, 1042023428, 1042025516, 1042029596, 1042035716, 1042551820, 1042555916, 1043072004, 1043072020, 1043076132, 1043602436>, ksigns = array<i32: 0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.{{[a-z]}}
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_iq3_xxs_q8_K_kernel_ggml_vec_dot_iq3_xxs_q8_K(
// The GRID-of-4 codebook + sign plane emitted as structured static const decls.
// CHECK: verbatim "static const uint32_t tcrv_iq3xxs_grid[256] = {0x04040404U,
// CHECK: verbatim "static const uint8_t tcrv_iq3xxs_ksigns[128] = {0, 129, 130, 3,
// CHECK: verbatim "static const uint8_t tcrv_iq3xxs_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};"
// The function-scoped fp32 accumulator + the super-block count nb = n / 256.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The FULL 8-bit kmask broadcast-loaded ONCE (above the super-block loop), u8m1
// (the vluxei16 revectorization uses a single 8-lane kmask, not two 4-lane halves).
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// The outer super-block loop.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// The fp16 weight d read + the fp32 activation d load -> d mul.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"const float">)
// The int32 bsum accumulator.
// CHECK: "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// aux32 reassembled from 4 little-endian byte loads (alignment-safe, no *(uint32_t*)).
// CHECK: bitwise_left_shift
// CHECK: bitwise_or
// ls = 2*(aux32>>28)+1.
// CHECK: bitwise_right_shift
// The GRID-of-4 GROUP gather: the uint16_t tmp[2] idx*4 byte-offsets, vle16 index,
// then the HARDWARE __riscv_vluxei16_v_i32m1 over the i32 grid base, reinterpreted to
// i8m1 grid bytes, then the 8-lane q8 load.
// CHECK: "emitc.variable"() {{.*}} -> !emitc.array<2x!emitc.opaque<"uint16_t">>
// CHECK: call_opaque "__riscv_vle16_v_u16mf2"
// CHECK: call_opaque "__riscv_vluxei16_v_i32m1"
// CHECK: call_opaque "__riscv_vreinterpret_v_i32m1_i8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// The SIGN plane: broadcast signs / vand kmask / vmsne / vneg / vmerge.
// CHECK: call_opaque "__riscv_vmv_v_x_u8m1"
// CHECK: call_opaque "__riscv_vand_vv_u8m1"
// CHECK: call_opaque "__riscv_vmsne_vx_u8m1_b8"
// CHECK: call_opaque "__riscv_vneg_v_i8m1"
// CHECK: call_opaque "__riscv_vmerge_vvm_i8m1"
// The signed widening product + the chained vwredsum + scalar extract.
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// bsum += sumi*ls (integer domain).
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"int32_t">, !emitc.opaque<"int32_t">)
// The per-super-block fold: ONE expression d*(float)bsum then sumf + that.
// CHECK: expression : !emitc.opaque<"float">
// CHECK: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// The trailing 0.25f factor + the *s store.
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: subscript
// CHECK: assign
