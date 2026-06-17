// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml IQ2_XXS x Q8_K GRID-codebook super-block dot-product
// (tcrv_rvv.iq2_xxs_q8_k_block_dot, the FIRST member of the deep IQ tail) lowers to
// the COMPLETE structured kernel (I5; zero raw(), every value an emitc node): the
// outer super-block loop over nb = n / 256, the per-super-block address arithmetic
// (weight stride 66, activation stride 292), the d = fp16(x.d) * fp32(y.d) scale, an
// int32 bsum reset per super-block, a FLAT loop over 8 sub-blocks each reassembling
// aux1 from 4 little-endian byte loads, the 4-bit scale ls = 2*(aux1>>28)+1, and a
// loop over 4 groups of 8 elements doing the GRID lookup (indexed vle8 over grid_i8 +
// idx*8), the SIGN-plane application (broadcast signs / vand kmask / vmsne / vmerge
// with vneg grid), the signed widening product, and the chained vwredsum into sumi.
// Then bsum += sumi*ls (integer domain), the per-super-block fold sumf += d*(float)bsum
// (ONE expression), and *s = 0.125f*sumf. The grid + ksigns tables are emitted as
// structured static const decls (NOT vrgather: the 2048-byte grid cannot broadcast).

module {
  tcrv.exec.kernel @ggml_vec_dot_iq2_xxs_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_iq2_xxs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2xxs-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8k-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_iq2_xxs_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_iq2_xxs_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.iq2_xxs_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_iq2_xxs_q8_k_block_dot", scale_model = "per-group-int4-grid-codebook-scale-int-domain", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_qs_byte_offset = 2 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, grid = array<i64: 578721382704613384, 578721382704613419, 578721382704617753, 578721382704622344, 578721382704622379, 578721382705727513, 578721382705731848, 578721382706907144, 578721382706907179, 578721382706916104, 578721382706916139, 578721382989826073, 578721382989830408, 578721382990940168, 578721382990949128, 578721382992119833, 578721382992124168, 578721383291815944, 578721383291815979, 578721383291824939, 578721383294109739, 578721455719057433, 578721455719061768, 578721455720171528, 578721455720175897, 578721456004270088, 578721456306264328, 578721456307383048, 578721533028468744, 578721533028468779, 578721533030762539, 578721533615671339, 578740074402285593, 578740074402289928, 578740074403399688, 578740074404579353, 578740074404583688, 578740074687498248, 578740074687498283, 578740074687507208, 578740074689792008, 578740074989488153, 578740074989492488, 578740074990602248, 578740074991786248, 578740147416729608, 578740147416729643, 578740147416738568, 578740147419023368, 578740147701946667, 578740147704245017, 578740148003932168, 578740148005046297, 578740224726149913, 578740224727255048, 578740225011353608, 578740225313347848, 578740225315641608, 578759865611585544, 578759865611589913, 578759865611594504, 578759865612704008, 578759865613888264, 578759865896798233, 578759865896802568, 578759865897912328, 578759865897912363, 578759866198797064, 578759938626033928, 578759938911242248, 578760015935440939, 578760015936559368, 583506457308694553, 583506457308698888, 583506457309808648, 583506457310988313, 583506457593907208, 583506457596200968, 583506457895901448, 583506457897011208, 583506457897015577, 583506530323138568, 583506530323147528, 583506530325432328, 583506530609465352, 583506530609474347, 583506530910341128, 583506607634848008, 583506607917766937, 583525149006366728, 583525149006375688, 583525149008660488, 583525149008664857, 583525149291588377, 583525149593569288, 583525222021933832, 583525222308317227, 583525299330222088, 583525299331340587, 583544940215666713, 583544940215671048, 583544940216780808, 583544940500879368, 583544940802869273, 583545013230110728, 583545013230115097, 583545013819607048, 583545090825848857, 588573006889486344, 588573006889486379, 588573006889495339, 588573007174703368, 588573007176992793, 588573007476688904, 588573007476688939, 588573079906233113, 588573080189152008, 588573157213341704, 588573157213341739, 588591698587158553, 588591698587162888, 588591698588272648, 588591698872371208, 588591698873489707, 588591771601602568, 588591771886815257, 588591771889113352, 588591849499330568, 588611489796467464, 588611489798752264, 588611490384779528, 588611640405530888, 1803700481349388313, 1803700481349392648, 1803700481350502408, 1803700481350511368, 1803700481351682073, 1803700481351686408, 1803700481634600968, 1803700481634609928, 1803700481635719467, 1803700481636894728, 1803700481936590873, 1803700481936595208, 1803700481937704968, 1803700554363832328, 1803700554366126088, 1803700554651338777, 1803700554951034888, 1803700554951039257, 1803700631673243673, 1803700631674357768, 1803700631958465288, 1803700631959574827, 1803700631960759048, 1803719173047060488, 1803719173047069448, 1803719173049354248, 1803719173634263048, 1803719173635386137, 1803719246062618667, 1803719246063802632, 1803719323370915848, 1803738964256360473, 1803738964256364808, 1803738964257474568, 1803738964541573128, 1803738964541577497, 1803739037270804488, 1803739037557140232, 1803739037558310937, 1803739037858007083, 1803739114865432857, 1803739115168532488, 1808485555953469448, 1808485555953478408, 1808485555954583577, 1808485555954592537, 1808485555955763208, 1808485556540672008, 1808485556540680968, 1808485628967917832, 1808485629253126187, 1808485629557414152, 1808485706865641497, 1808504248239458312, 1808504248239458347, 1808504320665594667, 1808504397974997017, 1808504398261328136, 1808524038860441608, 1808524038861555737, 1808524038861564697, 1808524039147952392, 1808524112160098312, 1808524189184305928, 1813552105534265608, 1813552105535375368, 1813552105819473928, 1813552105821776648, 1813552178548705288, 1813552178835036441, 1813552255859239688, 1813552256145623048, 1813570797231933448, 1813570797231937817, 1813570870247491592, 1813570870247491627, 1813570870833584392, 1813590588726446123, 3100737174032091144, 3100737174032091179, 3100737174032100139, 3100737174317303833, 3100737174619293739, 3100737247046539528, 3100737247047658248, 3100737247331747848, 3100737324357060633, 3100755865729763353, 3100755865729767688, 3100755865730877448, 3100755865730881817, 3100755866014976008, 3100755866017269768, 3100755938744207368, 3100755939029424427, 3100755939332528392, 3100756016053627673, 3100756016338831368, 3100756016341125128, 3100775656939063339, 3100775729953511688, 3100775807264032793, 3105522248636176648, 3105522248637286408, 3105522248638470408, 3105522248921384968, 3105522249225668633, 3105522321651734827, 3105522322237818888, 3105522399245244697, 3105540940333844488, 3105540940336138283, 3105540940619061512, 3105541013634615321, 3105560732130347033, 3105560804559882248, 3110588798216964139, 3110588798503290888, 3110588798804171033, 3110588871231417113, 3110588948540819464, 3110607489915759368, 3110627281410263048, 3110627354138384648>, ksigns = array<i32: 0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.{{[a-z]}}
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_ggml_vec_dot_iq2_xxs_q8_K(
// The GRID codebook + sign plane emitted as structured static const decls.
// CHECK: verbatim "static const int64_t tcrv_iq2xxs_grid[256] = {0x0808080808080808ULL,
// CHECK: verbatim "static const uint8_t tcrv_iq2xxs_ksigns[128] = {0, 129, 130, 3,
// CHECK: verbatim "static const uint8_t tcrv_iq2xxs_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};"
// The function-scoped fp32 accumulator + the super-block count nb = n / 256.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The kmask broadcast-loaded ONCE (above the super-block loop), u8m1.
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// The outer super-block loop.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// The fp16 weight d read + the fp32 activation d load -> d mul.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"const float">)
// The int32 bsum accumulator.
// CHECK: "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// aux1 reassembled from 4 little-endian byte loads (alignment-safe, no *(uint32_t*)).
// CHECK: bitwise_left_shift
// CHECK: bitwise_or
// ls = 2*(aux1>>28)+1.
// CHECK: bitwise_right_shift
// The GRID lookup: indexed pointer arith (grid_i8 + idx*8) then vle8.
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
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
// The trailing 0.125f factor + the *s store.
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: subscript
// CHECK: assign
