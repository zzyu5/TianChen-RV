// The iq2_xxs Win-A proof: the COMPILER SELECTS the ggml IQ2_XXS x Q8_K
// GRID-codebook block-dot integer-core anchor, and the selection DIVERGES by
// capability from the SAME attr-less input -- via the UNIFIED schedule autotuner
// (the SAME walk-all pass that auto-discovers every TunableScheduleOpInterface op,
// NO per-iq2_xxs pass).
//
// The kernel below carries NO integer_core_lmul knob -- the compiler must compute
// it. iq2_xxs's grid+sign vluxei16 gather + dot is ALWAYS one 32-lane sub-block
// body (gather 4 u64 grid + 4 u64 sign entries as i64<anchor>, reinterpret to
// i8<anchor>, fold the per-lane sign, ONE vwmul_vv_i16<2*anchor> + ONE vwredsum per
// sub-block). The single vsetvl/vluxei16_v_i64<anchor> cover is correct ONLY at the
// whole-LMUL anchor whose i8 strip VLMAX spans the 32-element sub-block at the
// derived minimum VLEN (a single i64 is 8 bytes, so the 4-entry gather needs the
// i64 anchor whose VLMAX reaches 4 = the i8 view reaches 32). WHICH anchor that is
// MOVES with VLEN:
//
//   * at VLEN 128: only m2 spans it (e8m1 VLMAX 16 < 32) -> integer_core_lmul "m2"
//     (i64m2 gather, i8m2 view, wide accumulator i16m4, u16mf2 index).
//   * at VLEN 256: m1 also reaches 32 (and i64m1 VLMAX reaches the 4 grid entries);
//     m1 TIES m2 on the capability-blind cost and the lighter footprint breaks the
//     tie -> integer_core_lmul "m1" (i64m1 gather, i8m1 view, i16m2 acc, u16mf4
//     index) -- exactly the m1-32-lane shape ggml's shipped _vl256 kernel uses.
//
// One capability fact (the REAL VLEN bits) -> the anchor FLIPS m2->m1. This is the
// compiler SELECTING the shape from a capability fact, not a hand-set attr -- the
// Win-A enrichment that right-sizes iq2_xxs's gather (note 10's gather-LMUL gap:
// ggml's _vl256 runs the gather at m1-32-lane while our default m2 leaves the
// register half-empty at VLEN256). The grid+sign decode and the i32 reduction are
// byte-exact for either anchor; only the LMUL/strip footprint changes.

// First, the DECISION-LEVEL proof: the unified autotuner stamps DIFFERENT anchors
// onto the SAME attr-less op purely by the VLEN capability fact (no lowering).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv | FileCheck %s --check-prefix=STAMP-VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=STAMP-VLEN256
//
// Then the EMISSION-LEVEL non-NULL proof: VLEN256 emits a BYTE-DIFFERENT kernel
// from VLEN128 (vluxei16_v_i64m1 / vreinterpret i8m1 / vmul_vv_i8m1 / vle8_v_i8m1 /
// vwmul_vv_i16m2 / vwredsum_i16m2 / vle16_v_u16mf4 vs the m2/i16m4/mf2 shape). A
// capability FACT changes the lowering -- NOT a structural NULL.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN256

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
// The GRID codebook + DERIVED signs64 sign table emitted as structured static const
// decls (signs64 = keven_signs_q2xs, expanded in-emitter from the ksigns selector).
// CHECK: verbatim "static const int64_t tcrv_iq2xxs_grid[256] = {0x0808080808080808ULL,
// CHECK: verbatim "static const int8_t tcrv_iq2xxs_signs64[1024] = {1, 1, 1, 1, 1, 1, 1, 1, -1,
// The function-scoped fp32 accumulator + the super-block count nb = n / 256.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The i64 grid + signs64 views set up ONCE (above the super-block loop).
// CHECK: cast %{{.*}} : !emitc.ptr<!emitc.opaque<"const int8_t">> to !emitc.ptr<!emitc.opaque<"const int64_t">>
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
// The vluxei16 sub-block body: u16 index load, TWO i64m2 indexed gathers (grid64 +
// signs64), each reinterpreted to i8m2, then vmul-fold the +-1 signs onto the GRID.
// CHECK: call_opaque "__riscv_vle16_v_u16mf2"
// CHECK: call_opaque "__riscv_vluxei16_v_i64m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_i64m2_i8m2"
// CHECK: call_opaque "__riscv_vluxei16_v_i64m2"
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vmul_vv_i8m2"
// The signed widening product + ONE vwredsum per sub-block + scalar extract.
// CHECK: call_opaque "__riscv_vwmul_vv_i16m4"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
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

// ================= STAMPED ANCHOR (the SELECTION decision) ==================
// rv64gcv (VLEN128): the compiler SELECTED m2 (the ONLY anchor whose e8 VLMAX 32
// spans the 32-element sub-block at VLEN128) + the SEMANTIC minimum_vlen = 128 the
// verifier recomputes legality from.
// STAMP-VLEN128: tcrv_rvv.iq2_xxs_q8_k_block_dot
// STAMP-VLEN128-SAME: integer_core_lmul = "m2"
// STAMP-VLEN128-SAME: minimum_vlen = 128 : i64
// STAMP-VLEN128-SAME: tcrv_rvv.iq2_xxs_schedule.has_zvl128b = true
// STAMP-VLEN128-SAME: tcrv_rvv.iq2_xxs_schedule.producer = "rvv-iq2-xxs-autotuner"
//
// rv64gcv_zvl256b (VLEN256): the SAME op FLIPS to m1 -- at VLEN256 m1's e8 VLMAX
// reaches 32 and i64m1 VLMAX reaches the 4 grid entries, so it spans the sub-block
// in ONE gather, TIES m2 on the capability-blind cost, and the lighter footprint
// breaks the tie to m1. The anchor MOVES with VLEN (the headline Win-A enrichment,
// matching ggml's _vl256 m1-32-lane shape). minimum_vlen = 256 is stamped.
// STAMP-VLEN256: tcrv_rvv.iq2_xxs_q8_k_block_dot
// STAMP-VLEN256-SAME: integer_core_lmul = "m1"
// STAMP-VLEN256-SAME: minimum_vlen = 256 : i64

// ===================== VLEN128 (rv64gcv) — the m2 anchor ====================
// The compiler SELECTED m2: vluxei16_v_i64m2 grid+sign gather, vreinterpret i8m2,
// vmul_vv_i8m2 sign fold, vle8_v_i8m2 q8, vwmul_vv_i16m4, vwredsum_vs_i16m4_i32m1,
// vle16_v_u16mf2 index.
// VLEN128: emitc.func @tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_ggml_vec_dot_iq2_xxs_q8_K(
// VLEN128: call_opaque "__riscv_vle16_v_u16mf2"
// VLEN128: call_opaque "__riscv_vluxei16_v_i64m2"
// VLEN128: call_opaque "__riscv_vreinterpret_v_i64m2_i8m2"
// VLEN128: call_opaque "__riscv_vmul_vv_i8m2"
// VLEN128: call_opaque "__riscv_vle8_v_i8m2"
// VLEN128: call_opaque "__riscv_vwmul_vv_i16m4"
// VLEN128: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// VLEN128-NOT: call_opaque "__riscv_vluxei16_v_i64m1"
// VLEN128-NOT: call_opaque "__riscv_vwmul_vv_i16m2"
// VLEN128-NOT: call_opaque "__riscv_vle16_v_u16mf4"
// VLEN128: return

// ===================== VLEN256 (rv64gcv_zvl256b) — the FLIP ==================
// The compiler SELECTED m1: a BYTE-DIFFERENT kernel from the VLEN128 m2 shape. The
// gather narrows to i64m1, the view to i8m1, the sign fold to vmul_vv_i8m1, the q8
// load to i8m1, the product to vwmul_vv_i16m2, the reduce to vwredsum_vs_i16m2_i32m1,
// and the index load to u16mf4. This is the NON-NULL proof: the two VLENs do NOT
// emit the same bytes.
// VLEN256: emitc.func @tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_ggml_vec_dot_iq2_xxs_q8_K(
// VLEN256: call_opaque "__riscv_vle16_v_u16mf4"
// VLEN256: call_opaque "__riscv_vluxei16_v_i64m1"
// VLEN256: call_opaque "__riscv_vreinterpret_v_i64m1_i8m1"
// VLEN256: call_opaque "__riscv_vmul_vv_i8m1"
// VLEN256: call_opaque "__riscv_vle8_v_i8m1"
// VLEN256: call_opaque "__riscv_vwmul_vv_i16m2"
// VLEN256: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// VLEN256-NOT: call_opaque "__riscv_vluxei16_v_i64m2"
// VLEN256-NOT: call_opaque "__riscv_vwmul_vv_i16m4"
// VLEN256-NOT: call_opaque "__riscv_vle16_v_u16mf2"
// VLEN256: return
