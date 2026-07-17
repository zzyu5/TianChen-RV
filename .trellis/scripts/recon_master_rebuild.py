#!/usr/bin/env python3
"""G8 主表重铸与对手档位自查令 · recon (机算·禁手写小计·纯案头).

唯一主表 = (op, format, engine) · 板=属性列 · q4_0 gemm decode/prefill=同行双子行不增行([K-10]).
对手三档法 {手调|通用向量|标量类|UNRESOLVED} · autovec归标量类(用户裁·§〇.1).
全员有对手(§〇.2·ggml标量参考兜底) · 唯一域外 q1_0.
两板对称法(§〇.3+补充令) · N/A-hw机判(能力谓词×板实例·ime.present) · 禁手标.
DEQ并表(§一.3·标量仗区·独立分账废止为过滤器) · 15 autovec dequant降标量类(§〇.3).

溯源:
  schema/coverage-roster.v1.json (93 roster / 91 denom)
  experiments/active/g8-stage3-opponent-reparse/evidence.md (符号级对手判据·双板)
  experiments/active/g7-census/bclass-forward-ops/opponent_ggml.cpp (forward源归属)
  v2报告§3 (19 measured cold: 14 vec_dot + 5 K-quant gemm/板)
只重排既有数据·零新测量·零新计时."""
import json, csv, sys
from collections import OrderedDict, Counter

ROOT = "/home/kingdom/phdworks/TianchenRV"
ROSTER = ROOT + "/schema/coverage-roster.v1.json"
T3A = ROOT + "/experiments/master/T3_A_board_A_rvv1.0_vlen128.csv"
T3B = ROOT + "/experiments/master/T3_B_board_B_rvv1.0_vlen256.csv"
OUT = ROOT + "/experiments/master/T3_master_rebuild.csv"
CLUE = ROOT + "/experiments/master/T3_master_rowclue.txt"
SNAPSHOT = "g8-master-final-clang-world"
FWD_OPS = {"add","cpy","gelu","mul","rms_norm","rope","scale","silu","softmax"}

def parse_t3(path):
    """(op,format) -> {'cold':float|None, 'cold_raw':str, 'disp':str} from canonical 36-col rows.
    disp derived from field[35] verdict token (not whole-row substring)."""
    import re
    out={}
    for ln in open(path):
        if ln.startswith("#") or not ln.strip(): continue
        f=ln.rstrip("\n").split(",")
        if len(f)!=36: continue
        parts=f[0].split("|")
        op0=parts[0]
        if op0 in ("measurement_row_key","axis"): continue
        if op0=="forward": op, fmt = parts[1], "f32"
        elif op0 in FWD_OPS: op, fmt = op0, "f32"
        elif op0=="gemm": op, fmt = "gemm_tile", parts[1]
        elif op0=="dequant": op, fmt = "dequantize_row", parts[1]
        elif op0=="quantize": op, fmt = "quantize_row", parts[1]
        elif op0=="vec_dot": op, fmt = "vec_dot", parts[1]
        else: op, fmt = op0, (parts[1] if len(parts)>1 else "")
        v=f[35]
        # ★按 in-denom[*;] 后的 verdict TOKEN 分类·非全串子串(成色注记可能含 '具名-X'/'PASS' 污染·前科)
        m=re.search(r'in-denom[*;]([A-Za-z0-9._-]+)', v)
        tok=m.group(1) if m else ""
        if "test-only-not-in-denom" in v: d="DEQ-照测-not-in-denom"
        elif tok.startswith("JUDGMENT-SUSPENDED") or (not tok and "JUDGMENT-SUSPENDED" in v): d="JUDGMENT-SUSPENDED"
        elif tok.startswith("PASS"): d="PASS"
        elif tok.startswith("FAIL") or "NAMED-X" in tok: d="具名-X"
        # fallback (旧行无 in-denom* 前缀时)
        elif "JUDGMENT-SUSPENDED" in v: d="JUDGMENT-SUSPENDED"
        elif re.search(r'(in-denom|;|\*)PASS',v): d="PASS"
        elif "NAMED-X" in v or "FAIL" in v: d="具名-X"
        elif "PASS" in v: d="PASS"
        else: d="?"
        craw=f[29]; m=re.match(r'\s*([0-9]*\.?[0-9]+)',craw); cold=float(m.group(1)) if m else None
        key=(op,fmt)
        if key in out: continue
        out[key]={"cold":cold,"cold_raw":craw,"disp":d}
    return out

# ---- board capability schema-fact instances (for N/A-hw machine derivation) ----
# rvv = openEuler VLEN128 RVV1.0 (no IME); k1 = SpacemiT X60 VLEN256 RVV1.0 + IME(xsmtvdotii)
BOARD_CAP = {
    "rvv": {"rvv1.0": True, "vlen": 128, "ime.present": False, "zvfh": True},
    "k1":  {"rvv1.0": True, "vlen": 256, "ime.present": True,  "zvfh": True},
}
def required_cap(op, fmt, engine):
    """capability predicate a row needs; returns set of required cap keys."""
    if engine == "ime":
        return {"ime.present"}
    return {"rvv1.0"}  # all base-RVV ops need only RVV1.0 (both boards satisfy)
def na_hw(op, fmt, engine, board):
    for cap in required_cap(op, fmt, engine):
        if not BOARD_CAP[board].get(cap, False):
            return True   # N/A-hw: predicate unsatisfiable on this board
    return False

# ---- opponent 3-tier map (symbol-level·from evidence.md §1/§3 + forward source) ----
# tier ∈ {手调, 通用向量, 标量类, UNRESOLVED, 域外}
# per (op, format): {rvv:(tier,sym,note), k1:(tier,sym,note)}
H, V, S, U, D = "手调", "通用向量", "标量类", "UNRESOLVED", "域外"
TIER = {
 # ---- FLAT vec_dot (block-dot / native-vec = human intrinsic = 通用向量) ----
 ("vec_dot","q4_0"): {"rvv":(V,"ggml_vec_dot_q4_0_q8_0","light-vec block-dot 弱"),      "k1":(V,"ggml_vec_dot_q4_0_q8_0","light-vec 弱")},
 ("vec_dot","q4_1"): {"rvv":(V,"ggml_vec_dot_q4_1_q8_1","light-vec block-dot 弱"),      "k1":(V,"ggml_vec_dot_q4_1_q8_1","light-vec 弱")},
 ("vec_dot","q5_0"): {"rvv":(V,"ggml_vec_dot_q5_0_q8_0","native-vec 中·qh5bit VLEN-adaptive"),"k1":(V,"ggml_vec_dot_q5_0_q8_0","light-vec 弱·板异塌轻")},
 ("vec_dot","q5_1"): {"rvv":(V,"ggml_vec_dot_q5_1_q8_1","native-vec 中·qh5bit"),        "k1":(V,"ggml_vec_dot_q5_1_q8_1","light-vec 弱·板异")},
 ("vec_dot","q8_0"): {"rvv":(V,"ggml_vec_dot_q8_0_q8_0","light-vec 弱·上游VLEN128破损"),"k1":(V,"ggml_vec_dot_q8_0_q8_0","light-vec 弱")},
 # ---- K-quant vec_dot ----
 ("vec_dot","q2_K"): {"rvv":(H,"ggml_vec_dot_q2_K_q8_K_vl128","SpacemiT vl128 full-unroll STRONG"),"k1":(H,"ggml_vec_dot_q2_K_q8_K_vl256","SpacemiT vl256 STRONG")},
 ("vec_dot","q3_K"): {"rvv":(H,"ggml_vec_dot_q3_K_q8_K_vl128","SpacemiT vl128 STRONG"),  "k1":(H,"ggml_vec_dot_q3_K_q8_K_vl256","SpacemiT vl256 STRONG")},
 ("vec_dot","q4_K"): {"rvv":(H,"ggml_vec_dot_q4_K_q8_K_vl128","SpacemiT vl128 STRONG"),  "k1":(H,"ggml_vec_dot_q4_K_q8_K_vl256","SpacemiT vl256 STRONG")},
 ("vec_dot","q5_K"): {"rvv":(V,"ggml_vec_dot_q5_K_q8_K","native-vec 中·NO vl-spec 唯一未手调"),"k1":(V,"ggml_vec_dot_q5_K_q8_K","native-vec-heavy·NO vl-spec 唯一未手调")},
 ("vec_dot","q6_K"): {"rvv":(H,"ggml_vec_dot_q6_K_q8_K_vl128","SpacemiT vl128 STRONG"),  "k1":(H,"ggml_vec_dot_q6_K_q8_K_vl256","SpacemiT vl256 STRONG")},
 # ---- iq/fp4 vec_dot ----
 ("vec_dot","iq1_s"):{"rvv":(H,"ggml_vec_dot_iq1_s_q8_K_vl128","SpacemiT vl128 grid-gather STRONG"),"k1":(H,"ggml_vec_dot_iq1_s_q8_K_vl256","SpacemiT vl256 STRONG")},
 ("vec_dot","iq1_m"):{"rvv":(H,"ggml_vec_dot_iq1_m_q8_K_vl128","SpacemiT vl128 STRONG"),  "k1":(H,"ggml_vec_dot_iq1_m_q8_K_vl256","SpacemiT vl256 STRONG")},
 ("vec_dot","iq4_nl"):{"rvv":(H,"ggml_vec_dot_iq4_nl_q8_0_vl128","hand-vl128 codebook-gather LIGHT〔V·手调〕"),"k1":(H,"ggml_vec_dot_iq4_nl_q8_0_vl256","hand-vl256 codebook-gather LIGHT〔V-纠·板一致·chip-tuned〕")},
 ("vec_dot","nvfp4"):{"rvv":(S,"ggml_vec_dot_nvfp4_q8_0","generic-C autovec NO-riscv-spec→标量类(§〇.1)〔V〕"),"k1":(S,"ggml_vec_dot_nvfp4_q8_0","generic-C autovec→标量类〔V〕")},
 # ---- iq/tq vec_dot ★UNRESOLVED 清偿(收口令二·objdump 反汇编 _vlNNN 静态符号·三判据齐→手调) ----
 #   源=arch/riscv/quants.c 手写 __riscv_v(1788处)·全格有 _vl128(rvv)/_vl256(k1) 专化·objdump 实向量op → 手调
 ("vec_dot","iq2_xxs"):{"rvv":(H,"ggml_vec_dot_iq2_xxs_q8_K_vl128.isra.0","objdump rvv:121/vec36/mac6/gather4·vl128手写gather→手调〔清偿〕"),"k1":(H,"..._vl256","objdump k1:333/vec122/mac24/gather16→手调〔清偿〕")},
 ("vec_dot","iq2_xs"): {"rvv":(H,"ggml_vec_dot_iq2_xs_q8_K_vl128.isra.0","objdump rvv:226/vec73/mac24/gather8·heavy→手调〔清偿〕"),"k1":(H,"..._vl256","objdump k1:216/vec53/mac20→手调〔清偿〕")},
 ("vec_dot","iq2_s"):  {"rvv":(H,"ggml_vec_dot_iq2_s_q8_K_vl128.isra.0","objdump rvv:102/vec27/mac3/gather2→手调〔清偿〕"),"k1":(H,"..._vl256","objdump k1:344/vec105/mac20/gather12→手调〔清偿〕")},
 ("vec_dot","iq3_xxs"):{"rvv":(H,"ggml_vec_dot_iq3_xxs_q8_K_vl128.isra.0","objdump rvv:129/vec26/mac5/gather3→手调〔清偿〕"),"k1":(H,"..._vl256","objdump k1:314/vec71/mac20/gather12→手调〔清偿〕")},
 ("vec_dot","iq3_s"):  {"rvv":(H,"ggml_vec_dot_iq3_s_q8_K_vl128.isra.0","objdump rvv:102/vec26/mac2→手调〔清偿〕"),"k1":(H,"..._vl256","objdump k1:232/vec84/mac12/gather8→手调〔清偿〕")},
 ("vec_dot","iq4_xs"): {"rvv":(H,"ggml_vec_dot_iq4_xs_q8_K_vl128.isra.0","objdump rvv:84/vec16/mac3·LIGHT→手调〔清偿〕"),"k1":(H,"..._vl256","objdump k1:144/vec18/mac5→手调〔清偿〕")},
 ("vec_dot","mxfp4"):  {"rvv":(H,"ggml_vec_dot_mxfp4_q8_0_vl128","objdump rvv:118/vec18/mac4/gather2·hand-vl128 LIGHT→手调〔清偿〕"),"k1":(H,"ggml_vec_dot_mxfp4_q8_0_vl256","objdump k1:96/vec24/mac6/gather4→手调〔清偿·k1原UNRESOLVED已清〕")},
 ("vec_dot","tq1_0"):  {"rvv":(H,"ggml_vec_dot_tq1_0_q8_K_vl128.isra.0","objdump rvv:114/vec45/mac14·ternary vl128手写→手调〔清偿〕"),"k1":(H,"..._vl256","objdump k1:172/vec93/mac32→手调〔清偿〕")},
 ("vec_dot","tq2_0"):  {"rvv":(H,"ggml_vec_dot_tq2_0_q8_K_vl128.isra.0","objdump rvv:85/vec46/mac10→手调〔清偿〕"),"k1":(H,"..._vl256","objdump k1:88/vec46/mac10→手调〔清偿〕")},
 ("vec_dot","q1_0"):   {"rvv":(D,"ggml_vec_dot_q1_0_q8_0_vl128(Weft-internal)","internal-A/B·自己当自己靶禁·永久域外"),"k1":(D,"(Weft-internal)","永久域外")},
 # ---- K-quant GEMM (rvv 全 cross-op vs block-dot·k1 4 real repack + q3_K cross-op) ----
 ("gemm_tile","q2_K"):{"rvv":(H,"ggml_vec_dot_q2_K_q8_K_vl128(CROSSOP)","our-gemm vs vl128 手调block-dot·cross-op(rvv零K-quant repack)"),"k1":(H,"ggml_gemm_q2_K_8x8_q8_K(REAL)","真16x1/8x8 hand-brick repack STRONG")},
 ("gemm_tile","q3_K"):{"rvv":(H,"ggml_vec_dot_q3_K_q8_K_vl128(CROSSOP)","cross-op vs vl128 手调block-dot"),"k1":(H,"ggml_vec_dot_q3_K_q8_K_vl256(CROSSOP)","q3_K唯一无repack·cross-op vs vl256 手调block-dot")},
 ("gemm_tile","q4_K"):{"rvv":(H,"ggml_vec_dot_q4_K_q8_K_vl128(CROSSOP)","cross-op vs vl128 手调block-dot"),"k1":(H,"ggml_gemm_q4_K_16x1_q8_K(REAL)","真16x1 hand-brick byte-drop-in STRONG(Win-K1-VLEN)")},
 ("gemm_tile","q5_K"):{"rvv":(V,"ggml_vec_dot_q5_K_q8_K(CROSSOP)","cross-op vs native-vec-moderate(q5_K无vl-spec)"),"k1":(H,"ggml_gemm_q5_K_8x4_q8_K(repack)","repack opponent·★成色升级 DEFERRED(q5_K 8x8 前提证伪·非 verified hand-brick·F-2 纠·真硬赢锁 2=q4_K/q2_K)")},
 ("gemm_tile","q6_K"):{"rvv":(H,"ggml_vec_dot_q6_K_q8_K_vl128(CROSSOP)","cross-op vs vl128 手调block-dot"),"k1":(H,"ggml_gemm_q6_K_16x1_q8_K(repack)","repack opponent·★成色升级 DEFERRED(非 verified hand-brick·F-2 纠)")},
 # ---- iq/tq/fp4 GEMM (§〇.2 全员有对手·ggml标量参考兜底·去向=待补标量仗·pending-真) ----
 ("gemm_tile","iq1_s"):{"rvv":(S,"ggml scalar-ref(fallback)","1-bit grid·无 repack GEMM→标量参考兜底(§〇.2)·待补标量仗"),"k1":(S,"ggml scalar-ref(fallback)","待补标量仗")},
 ("gemm_tile","iq1_m"):{"rvv":(S,"ggml scalar-ref(fallback)","无 repack GEMM→标量参考兜底·待补标量仗"),"k1":(S,"ggml scalar-ref(fallback)","待补标量仗")},
 ("gemm_tile","iq2_xxs"):{"rvv":(S,"ggml scalar-ref(fallback)","same-op repack absent→标量参考兜底(§〇.2)·待补标量仗"),"k1":(S,"ggml scalar-ref(fallback)","待补标量仗")},
 ("gemm_tile","iq2_xs"): {"rvv":(S,"ggml scalar-ref(fallback)","待补标量仗"),"k1":(S,"ggml scalar-ref(fallback)","待补标量仗")},
 ("gemm_tile","iq2_s"):  {"rvv":(S,"ggml scalar-ref(fallback)","待补标量仗"),"k1":(S,"ggml scalar-ref(fallback)","待补标量仗")},
 ("gemm_tile","iq3_xxs"):{"rvv":(S,"ggml scalar-ref(fallback)","待补标量仗"),"k1":(S,"ggml scalar-ref(fallback)","待补标量仗")},
 ("gemm_tile","iq3_s"):  {"rvv":(S,"ggml scalar-ref(fallback)","待补标量仗"),"k1":(S,"ggml scalar-ref(fallback)","待补标量仗")},
 ("gemm_tile","iq4_nl"): {"rvv":(V,"ggml_vec_dot_iq4_nl_q8_0_vl128(CROSSOP)","measured anchor 0.217×·pending-fold·native-vec中"),"k1":(V,"ggml_vec_dot_iq4_nl_q8_0_vl256(CROSSOP)","pending-fold")},
 ("gemm_tile","iq4_xs"): {"rvv":(S,"ggml scalar-ref(fallback)","待补标量仗"),"k1":(S,"ggml scalar-ref(fallback)","待补标量仗")},
 ("gemm_tile","mxfp4"):  {"rvv":(S,"ggml scalar-ref(fallback)","待补标量仗"),"k1":(S,"ggml scalar-ref(fallback)","待补标量仗")},
 ("gemm_tile","nvfp4"):  {"rvv":(S,"ggml scalar-ref(fallback·generic-C)","generic-C autovec→标量类·待补标量仗"),"k1":(S,"ggml scalar-ref(fallback)","待补标量仗")},
 ("gemm_tile","tq1_0"):  {"rvv":(S,"ggml scalar-ref(fallback)","待补标量仗"),"k1":(S,"ggml scalar-ref(fallback)","待补标量仗")},
 ("gemm_tile","tq2_0"):  {"rvv":(S,"ggml scalar-ref(fallback)","待补标量仗"),"k1":(S,"ggml scalar-ref(fallback)","待补标量仗")},
 ("gemm_tile","q1_0"):   {"rvv":(D,"(Weft-internal)","永久域外"),"k1":(D,"(Weft-internal)","永久域外")},
 # FLAT gemm (pending-fold·e2e绿·kernel-sym未折入clang18 T3) — opponent = 上游 repack / FLAT block-dot
 ("gemm_tile","q4_0"):{"rvv":(V,"ggml_vec_dot_q4_0_q8_0(block-dot)","★二义解〔清偿〕:stock 有 repack(ggml_gemm_q4_0_8x8/16x1)但 VLEN128 gate-off·真派发=block-dot→通用向量·我方win=routing白嫖非beat-repack·decode+prefill·e2e 5.92×/1.91×"),"k1":(V,"block-dot","pending-fold")},
 ("gemm_tile","q4_1"):{"rvv":(V,"ggml_vec_dot_q4_1_q8_1(CROSSOP)","block-dot vectorized·nibble-unpack非autovec-able→通用向量〔V-纠〕·e2e 3.68×"),"k1":(V,"ggml_vec_dot_q4_1_q8_1(CROSSOP)","block-dot→通用向量〔V-纠〕·pending-fold")},
 ("gemm_tile","q5_0"):{"rvv":(V,"FLAT block-dot","e2e prefill 1.21×·pending-fold"),"k1":(V,"FLAT block-dot","pending-fold")},
 ("gemm_tile","q5_1"):{"rvv":(V,"FLAT block-dot","e2e prefill 1.09×·pending-fold"),"k1":(V,"FLAT block-dot","pending-fold")},
 ("gemm_tile","q8_0"):{"rvv":(V,"FLAT block-dot","e2e 4.35× correctness-carrier·pending-fold"),"k1":(V,"FLAT block-dot","pending-fold")},
 # ---- forward (源码归属·§一.B + opponent_ggml.cpp) ----
 ("forward","softmax"):{"rvv":(V,"ggml_vec_soft_max_f32","exported ggml手写RVV intrinsic·expf reduce"),"k1":(V,"ggml_vec_soft_max_f32","leaf 手写intrinsic·wrapper scalar-setup")},
 ("forward","rms_norm"):{"rvv":(V,"ggml_compute_forward_rms_norm_f32","native RVV m8 vec_scale + scalar-dbl reduce·HYBRID·BORDERLINE"),"k1":(V,"ggml_compute_forward_rms_norm_f32","HYBRID·native-vec-light")},
 ("forward","rope"):   {"rvv":(S,"ggml_compute_forward_rope_f32","scalar cos/sin cache 2-pass + autovec rotate→标量类"),"k1":(S,"ggml_compute_forward_rope_f32","mostly-scalar→标量类")},
 ("forward","silu"):   {"rvv":(V,"ggml_vec_silu_f32","exported ggml手写RVV intrinsic·sigmoid/expf 向量化"),"k1":(V,"ggml_vec_silu_f32","手写intrinsic")},
 ("forward","gelu"):   {"rvv":(S,"ggml_table_gelu_f16(BSS-LUT)","A2 f16-LUT同档重比done·WIN 1.116×·便宜档表查·同档调度胜·非硬赢·0ULP"),"k1":(S,"ggml_table_gelu_f16(BSS-LUT)","A2同档重比done·PARITY 0.957×·memory-bound near-parity·便宜档")},
 ("forward","add"):    {"rvv":(S,"ggml_vec_add_f32(vec.h:89)","AVX2-only vec path·RV落scalar loop→autovec→标量类"),"k1":(S,"ggml_vec_add_f32","scalar loop autovec→标量类")},
 ("forward","mul"):    {"rvv":(S,"ggml_vec_mul_f32(vec.h:127)","pure scalar loop→autovec→标量类"),"k1":(S,"ggml_vec_mul_f32","autovec→标量类")},
 ("forward","scale"):  {"rvv":(V,"ggml_vec_scale_f32(vec.h:703)","NATIVE RVV m8 vfmul_vf 手写intrinsic·byte-identical algo"),"k1":(V,"ggml_vec_scale_f32","native RVV m8·板异emit较轻但同源intrinsic")},
 ("forward","cpy"):    {"rvv":(S,"ggml_vec_cpy_f32(vec.h:119)","pure scalar loop→autovec/memcpy→标量类"),"k1":(S,"ggml_compute_forward_dup_cpy","scalar/memcpy-stream→标量类")},
}
# dequant (24) — 全 标量类: 源=scalar dequantize_row_*·任何向量化=autovec(§〇.1)·clang18 per-format autovec·单世界clang(PR-17终裁)
# quantize (3) — 标量类(scalar quantize_row_* autovec)·flag verify
# product_reduce (3) — §〇.2 ggml scalar-ref fallback·internal sub-primitive·待补标量仗
DEQ_NOTE_RVV = {"q4_K":"clang18 rvv0 真SCALAR·autovec弱(单世界clang)"}

def op_group(op):
    if op in ("gemm_tile","vec_dot"): return "matmul"
    if op in FWD_OPS: return "forward"
    if op == "dequantize_row": return "dequant"
    if op == "quantize_row": return "quantize"
    if op == "product_reduce": return "product_reduce"
    return "other"

# ---- measured cold ratios (19/board = 14 vec_dot + 5 K-quant gemm·v2报告§3·clang18对称) ----
COLD = {  # (op,format): (rvv, k1)
 ("vec_dot","q4_0"):(0.938,0.986),("vec_dot","q4_1"):(1.027,1.008),("vec_dot","q5_0"):(0.451,0.384),
 ("vec_dot","q5_1"):(0.446,0.398),("vec_dot","q8_0"):(0.965,0.984),("vec_dot","q2_K"):(0.342,0.684),
 ("vec_dot","q3_K"):(0.258,0.522),("vec_dot","q4_K"):(0.189,0.592),("vec_dot","q5_K"):(0.843,1.065),
 ("vec_dot","q6_K"):(0.180,0.549),("vec_dot","iq1_s"):(0.365,0.361),("vec_dot","iq1_m"):(0.152,0.250),
 ("vec_dot","iq4_nl"):(1.150,0.697),("vec_dot","nvfp4"):(0.565,1.086),
 ("gemm_tile","q2_K"):(1.112,1.364),("gemm_tile","q3_K"):(1.399,2.920),("gemm_tile","q4_K"):(1.114,1.187),
 ("gemm_tile","q5_K"):(1.067,4.448),("gemm_tile","q6_K"):(0.960,1.769),
}
# q5@k1 deployed absorb (§一.1 anti-gate 自证): q5_0/q5_1 k1 gemm-decode repack deploy path
Q5K1_DEPLOY = {("gemm_tile","q5_0"):("k1",1.760),("gemm_tile","q5_1"):("k1",2.241)}
# ★A2-batch4 M=1 GEVM decode 实测(禁串行·T3 gemm 行是 prefill·decode 走此独立测量)
# (op,format): {board:(cold, disp-token, 成色)}·rvv=clang-world(PR-17终裁)·q5x@k1 已 PASS-DEPLOYED
GEMM_DECODE = {
 ("gemm_tile","q4_0"):{"rvv":(6.909,"PASS","便宜档-weak-q4_0-block-dot-VLEN128-gate-off·big-multiple≠hard-win"),
                       "k1":(2.575,"PASS","便宜档-what-if(k1 front-door DECLINE repack@decode·force-construct)")},
 ("gemm_tile","q5_0"):{"rvv":(0.947,"PASS","near-parity-memory-leaning-compiler-sensitive(clang0.794 named-X)"),
                       "k1":(1.205,"PASS-DEPLOYED","C1-deployed·batch4 免测确认1.205×")},
 ("gemm_tile","q5_1"):{"rvv":(1.090,"PASS","near-parity"),
                       "k1":(1.317,"PASS-DEPLOYED","C1-deployed·batch4 免测确认1.317×")},
 ("gemm_tile","q4_K"):{"rvv":(0.361,"具名-X","★genuine-C3′-negative·super-block-fold@M=1不amortize·对手结构优势具名(rvv q4_K native-vec block-dot 3.6×快·gap=opp-strength非我方核·vsetvl1387 spill=指令数内禀墙·单世界clang也输)"),
                       "k1":(1.535,"PASS","手调-REAL·k1 WIN vs 弱opp·genuine(k1 real repack hand-brick域·batch4·8-sub-block不泛化到16-sub)")},
 # ★A2-batch5: K-quant decode C3′ 负结果(super-block fold@M=1 不amortize·format-keyed 适用边界·11/12 具名-X·batch4 q4_K@k1 win 不泛化·判别键=sub-block数16 vs 8)
 ("gemm_tile","q2_K"):{"rvv":(0.0685,"具名-X","指令数内禀-fold@M=1不amortize(16-sub-block双fold)·vsetvl922 spill·单世界clang也输0.36(wall非compiler·结构墙)·C3′负"),
                       "k1":(0.9585,"PASS","★near-parity非win(ratio<1.0·弱opp·q2_K 2-bit但16-sub-block最高vsetvl81)·唯一非-loss·成色低不称赢")},
 ("gemm_tile","q3_K"):{"rvv":(0.0830,"具名-X","指令数内禀-fold@M=1·vsetvl2225 spill(结构墙·单世界clang)·C3′负"),
                       "k1":(0.4252,"具名-X","指令数内禀-fold@M=1不amortize(16-sub-block)·C3′负")},
 ("gemm_tile","q5_K"):{"rvv":(0.1273,"具名-X","指令数内禀-fold@M=1·vsetvl2952 spill(结构墙·单世界clang)·C3′负"),
                       "k1":(0.6806,"具名-X","指令数内禀-fold@M=1不amortize·C3′负")},
 ("gemm_tile","q6_K"):{"rvv":(0.0535,"具名-X","指令数内禀-fold@M=1·vsetvl2461 spill(结构墙·单世界clang)·C3′负"),
                       "k1":(0.3771,"具名-X","指令数内禀-fold@M=1不amortize(16-sub-block)·C3′负")},
 # ★A2-batch6: iq/tq/fp4 gemm decode(scalar-ref兜底·便宜档·单世界clang·PR-17)·memory-bound parity-leaning
 ("gemm_tile","iq4_xs"):{"rvv":(0.63,"具名-X","decode near-parity·rvv named-X(单世界clang)·便宜档-scalar-ref·PR-17"),"k1":(0.93,"PASS","decode near-parity·便宜档-scalar-ref·禁称硬赢")},
 ("gemm_tile","iq2_xxs"):{"rvv":(0.60,"具名-X","decode·rvv named-X(单世界clang)·便宜档"),"k1":(1.13,"PASS","decode·便宜档-scalar-ref·禁称硬赢")},
 ("gemm_tile","iq2_xs"):{"rvv":(1.00,"PASS","decode·便宜档-scalar-ref·禁称硬赢"),"k1":(3.81,"PASS","decode·便宜档·k1 opp clang-bloat假象·禁称硬赢")},
 ("gemm_tile","iq2_s"):{"rvv":(1.03,"PASS","decode·便宜档"),"k1":(3.55,"PASS","decode·便宜档·clang-bloat假象·禁称硬赢")},
 ("gemm_tile","mxfp4"):{"rvv":(1.51,"PASS","decode·便宜档-scalar-ref·禁称硬赢"),"k1":(0.94,"PASS","decode near-parity·便宜档")},
 ("gemm_tile","tq1_0"):{"rvv":(1.67,"PASS","decode·便宜档(rvv-clang boundary 0.79 X footnote)"),"k1":(1.99,"PASS","decode·便宜档·禁称硬赢")},
 ("gemm_tile","tq2_0"):{"rvv":(1.19,"PASS","decode·便宜档"),"k1":(1.59,"PASS","decode·便宜档·禁称硬赢")},
 # ★P1-backfill7 (2026-07-17·独立 M=1 GEVM 实测·禁继承 prefill nr16 / vec_dot M=1 / 历史锚 0.2487)
 #   raw = experiments/active/g8-stage3-attack/P1-backfill7-raw/ · prereg = ../P1-backfill7-prereg.md
 #   rvv 板【不入表】= VOID-NOISE(我方 leaf 内生波动 relIQR 0.79-14.50% vs 对手恒 0.11-0.38%=机器安静)·维持 pending·0 样本不造数
 ("gemm_tile","iq4_nl"):{"k1":(0.2493,"具名-X",
   "★CROSSOP(我方 repack-GEVM vs 对手 per-column block-dot·非同算子硬赢)·s1 0.2493/s2 0.2490·N=25·2-seed·CI[0.2491—0.2498]"
   "·同算子 OPP-S ggml_gemv_iq4_nl_16x1_q8_0 = 0.1469(亦输)·ZERO-MODEL ours0/512 oppX0/512 oppS0/512 = 正确核(输性能非正确性)"
   "·★墙(完整环走完·机制级)=codebook-gather-bound: 我 64×vluxei16.v(走内存 codebook 数组 indexed gather) vs OPP-S 32×vrgather.vv"
   "(16 项 codebook 常驻向量寄存器的寄存器置换)·16 项平凡装入单寄存器→vrgather 是正确原语·vluxei16 每元素付访存端口往返且无法外提"
   "·叠加 vsetvli storm(我 260 vs 51)+2.5×指令量(724 vs 293)·双方 true_sp_spill=0(排除 spill 病)"
   "·可修(emitter 成熟度缺口): 需 codebook-size 键控 gather 原语选择(≤VLMAX 项走寄存器 vrgather·大 grid 才走 vluxei16)"
   "——与 [emitter-maturity-vluxei16] 方向相反(vluxei16 对 iq1_s 大 grid 是 win·对 16 项 tiny codebook 是 loss)→ 判别键=codebook 尺寸(C3′ 能力键控正例)"
   "·★部署成色=what-if(selector=block-dot-decline-vlen256-decode-measured-negative·出货走 block-dot·禁写成 k1 decode 部署输)"
   "·循环论证防线: registry DECLINE 由 0.248x 驱动·本轮独立复测 0.2493/0.2490 与 registry Negative 一致(未证伪)→ 无 canon 触发·selector/registry 不动")},
}
# ★A2-batch7 IME kernel-sym(k1·vendor 真手调 IME 核·非便宜档·honest 负结果·赛道≠e2e perf-covered 绿·禁互推)
IME_KERNELSYM = {
 ("gemm_tile","q4_0"):(0.196,"具名-X","IME kernel-sym vs vendor手调IME核 gemm_kernel_i8i4(真硅vmadot)·输5.1×·墙=scale-fold epilogue未融进vmadot MAC·真硬件vendor正面度量负结果"),
 ("gemm_tile","q8_0"):(None,"该板无合法对手","q8_0不进vendor IME(dispatch ime.cpp:317仅q4_0/q4_1/q4_K)·落RVV-repack回退·结构无IME对手·探针证据·★坐实e2e q8_0@ime beat 0.984×=赢RVV-repack非赢IME"),
 ("gemm_tile","q4_K"):(0.049,"具名-X","IME kernel-sym vs vendor IME·输20×·★C3′[PAT-1]format-keyed边界最锋利·gap随格式复杂度扩张(q4_0 5×→q4_K 20×·我方fold随super-block暴涨25→110ms·vendor单核塞所有格式4.9→5.5ms)"),
}
# ★线A·A1 单世界 clang-18 override(PR-17 终裁·「用 clang 就用 clang-18 全部都用」·gcc 视为不存在)
#   cold 引 experiments/active/g8-stage3-attack/A1-rvv-clang18-unify/evidence.md(真-对称 ours-clang18 vs opp-clang18 重编·s1)
#   承 gelu/GEMM_DECODE/IME_KERNELSYM stale-source 更正范式·rvv board only·便宜档禁称硬赢·0 造数
# (op,format,regime) -> (rvv_cold, clang-world note)
CLANG_WORLD = {
 # ── 7 iq/tq/fp4 gemm PREFILL @rvv(A1 §1·K=2048 nr16 nc512·标量类·opp=ggml scalar-ref 便宜档·byte-exact ZERO-MODEL nbad=0)──
 ("gemm_tile","iq4_xs","prefill"): (3.15,"clang世界(单一编译器世界)·opp=ggml scalar-ref(便宜档)·具名-X→PASS·便宜档禁称硬赢(A1§1·s1 3.15/s2 3.11)"),
 ("gemm_tile","iq2_xxs","prefill"):(2.32,"clang世界·opp=ggml scalar-ref(便宜档)·具名-X→PASS·便宜档禁称硬赢(A1§1·s1 2.32/s2 2.34)"),
 ("gemm_tile","iq2_xs","prefill"): (12.32,"clang世界·opp=ggml scalar-ref(便宜档)·具名-X→PASS·大倍数=opp-clang编巨胖伪影(160ms)·便宜档禁称硬赢(A1§1·s1 12.32/s2 12.29)"),
 ("gemm_tile","iq2_s","prefill"):  (11.49,"clang世界·opp=ggml scalar-ref(便宜档)·具名-X→PASS·大倍数=opp-clang编胖伪影(150ms)·便宜档禁称硬赢(A1§1·s1 11.49/s2 11.46)"),
 ("gemm_tile","mxfp4","prefill"):  (3.65,"clang世界·opp=ggml scalar-ref(便宜档)·具名-X→PASS·便宜档禁称硬赢(A1§1·s1 3.65/s2 3.66)"),
 ("gemm_tile","tq1_0","prefill"):  (7.90,"clang世界·opp=ggml scalar-ref(便宜档)·具名-X→PASS·便宜档禁称硬赢(A1§1·s1 7.90/s2 7.93)"),
 ("gemm_tile","tq2_0","prefill"):  (9.91,"clang世界·opp=ggml scalar-ref(便宜档)·具名-X→PASS·ours vsetvl=8最简·便宜档禁称硬赢(A1§1·s1 9.91/s2 10.10)"),
 # ── 3 DEQ @rvv 翻转(A1 §2·标量类·opp=dequantize_row_* autovec 便宜档·byte-exact 0mism/0ULP)·q4_1/q8_0 本已 PASS ──
 ("dequantize_row","q4_0",""): (2.51,"clang世界(单一编译器世界)·opp=dequantize_row autovec(便宜档)·具名-X→PASS·便宜档禁称硬赢(A1§2·s1 2.51/s2 2.90)"),
 ("dequantize_row","q5_0",""): (1.007,"clang世界·opp=autovec·具名-X→PASS·★parity 编译器中性真结果(双方 memory-bound 0.60GB/s·qh5bit DRAM墙)(A1§2·s1/s2 1.007)"),
 ("dequantize_row","q5_1",""): (1.02,"clang世界·opp=autovec·具名-X→PASS·★parity 编译器中性真结果(双方 memory-bound 0.60GB/s·qh5bit DRAM墙)(A1§2·s1/s2 1.02)"),
}

def disp(op, fmt, engine, board, tier, cold, na):
    if na: return "N/A-hw"
    if tier == D: return "域外-永久(q1_0)"
    if cold is not None:
        return "PASS" if cold >= 0.8 else "具名-X"
    # no cold: pending
    if op == "gemm_tile" and tier == S:  # iq/tq scalar-ref fallback
        return "pending-真(待补标量仗)"
    if op == "product_reduce": return "pending-真(待补标量仗)"
    if op == "quantize_row": return "pending-真(待补向量仗)"  # V-纠: arch/riscv/quants.c 手写intrinsic
    if op == "dequantize_row": return "DEQ-历史(pre-clang18 stale·照测不进头条)"
    return "pending-fold" if op in ("gemm_tile","vec_dot") else "pending"

def main():
    roster = json.load(open(ROSTER))["kernels"]
    t3 = {"rvv":parse_t3(T3A), "k1":parse_t3(T3B)}
    # build master rows: (op,format,engine,regime). ★单分母制(数字字典v2令): 全量口径
    # q4_0 gemm decode/prefill = 2 独立行([K-10] 结构分立·不再折·上轮折成1行=错·85/88 全量分母要求)
    seen = set(); rows = []
    for k in roster:
        op, fmt, eng = k["op"], k["format"], k.get("engine","")
        if op in ("flash_attn","bf16"): continue          # class C OOD (not in 5 groups)
        regime = k.get("regime","")
        key = (op, fmt, eng, regime)
        if key in seen: continue
        seen.add(key)
        rows.append({"op":op,"format":fmt,"engine":eng,"regime":regime})

    master = []
    for r in rows:
        op, fmt, eng, regime = r["op"], r["format"], r["engine"], r.get("regime","")
        isfwd = op in FWD_OPS
        tkey = ("forward", op) if isfwd else (op, fmt)
        tent = TIER.get(tkey)
        rec = {"op":op,"format":fmt,"engine":eng,"regime":regime,"group":op_group(op)}
        for board in ("rvv","k1"):
            na = na_hw(op,fmt,eng,board)
            if fmt == "q1_0":            # q1_0 永久域外(§〇.2 唯一例外)·所有 op(vec_dot/dequant/gemm)
                tier=D; sym="(Weft-internal)"; note="q1_0 internal-A/B·永久域外·非分母"
            elif op == "dequantize_row":
                tier = S; sym = "dequantize_row_%s"%fmt
                note = DEQ_NOTE_RVV.get(fmt,"") if board=="rvv" else ""
                note = (note+" · " if note else "")+"scalar源 autovec→标量类(§〇.1)·per-format clang18 autovec·单世界clang(PR-17)"
            elif op == "quantize_row":
                tier=V; sym="quantize_row_%s"%fmt; note="★ggml-cpu/arch/riscv/quants.c 手写__riscv_v intrinsic(非scalar autovec)→通用向量〔V-纠·source铁证·objdump待§六补〕"
            elif op == "product_reduce":
                tier=S; sym="ggml scalar-ref(fallback)"; note="internal sub-primitive·§〇.2兜底·待补标量仗"
            elif eng == "ime":
                # IME 行对手 = stock vendor IME(chip-specific dispatch)→手调(k1)·rvv=N/A-hw
                tier = H
                sym = "stock IME %s (vendor)"%fmt
                note = "vendor IME dispatch chip-specific→手调·成色:"+{"q4_0":"tie-stock 1.0088×(未真beat)","q8_0":"beat-stock 2.233×(赢弱vendor)","q4_K":"黄0.909× super-block 传导稀释"}.get(fmt,"")
            elif tent:
                tier, sym, note = tent[board]
            else:
                tier, sym, note = U, "(no-map)", "UNRESOLVED"
            # cold/disp: prefer T3-measured verdict; else pending
            # ★串行 bug 双修:
            #  (1) IME 行(engine=ime)禁用 native (op,fmt) T3 cold (q4_K@ime 前科)
            #  (2) ★regime-split: T3 "gemm" 测量 = nr16 GEMM(prefill·M>1)·非 decode(GEVM·M=1)
            #      →只 prefill regime 拿 T3 cold·decode regime pending(GEVM 未测·除 q5x deploy)
            skip_t3 = (eng=="ime") or (op=="gemm_tile" and regime=="decode")
            t3rec = None if skip_t3 else t3[board].get((op,fmt))
            if t3rec and t3rec["disp"]!="?":
                c = t3rec["cold"]
                td = t3rec["disp"]
                if op=="dequantize_row":
                    # ★数字字典v2令: DEQ 计入所属档(标量类)头条·"照测子账/不进头条"废止(§一.3)
                    if td=="PASS": d="PASS"
                    elif td=="具名-X": d="具名-X"
                    elif td=="DEQ-照测-not-in-denom": d="pending(照测未定verdict)"
                    else: d="pending(DEQ)"
                elif td=="JUDGMENT-SUSPENDED":
                    # ★A2 gelu f16-LUT 同档重比清偿(2026-07-16·JUDGMENT-SUSPENDED解除·evidence gelu-f16lut-rematch/·commit f9424d1cb)
                    # rvv WIN 1.116×/k1 PARITY 0.957×·0ULP·便宜档表查·非硬赢(同档调度胜/memory-bound near-parity)→终态 PASS(0.8门)
                    # gelu 为唯一 JS 格·T3 opponent-reparse 行(数值档不对等)superseded by A2 同档重比·此 override 承 GEMM_DECODE/IME_KERNELSYM stale-source 更正范式
                    d = "PASS"
                elif td=="PASS": d="PASS"
                elif td=="具名-X": d="具名-X"
                elif td=="DEQ-照测-not-in-denom": d="DEQ照测(not-in-denom·标量仗)"
                else: d = disp(op,fmt,eng,board,tier,c,na)
            else:
                c = None
                d = disp(op,fmt,eng,board,tier,c,na)
            # ★gemm decode: A2-batch4 M=1 GEVM 实测(禁串行·T3 gemm=prefill·decode 走此独立测量)
            if op=="gemm_tile" and regime=="decode":
                gd = GEMM_DECODE.get((op,fmt))
                if gd and board in gd:
                    c, dtok, cx = gd[board]
                    if dtok=="PASS": d="PASS(decode-M1-GEVM)"
                    elif "DEPLOY" in dtok: d="PASS-DEPLOYED(decode-GEVM·C1)"
                    else: d="具名-X(decode-M1)"
                    note = note + " ·[decode-M1: "+cx+"]"
            # ★A2-batch7 IME kernel-sym(k1·vendor 真手调 IME·非便宜档·honest 负结果)
            if eng=="ime" and board=="k1":
                ik = IME_KERNELSYM.get((op,fmt))
                if ik:
                    c, dtok, cx = ik
                    d = "pending(IME结构·该板无合法对手·q8_0落RVV-repack回退·缺vendor-IME对手·探针证据)" if "无合法对手" in dtok else "具名-X(IME-kernel-sym·vendor手调IME)"
                    note = note + " ·[IME-kernel-sym: "+cx+"]"
            # q5@k1 deploy absorb (裁决④·decode-GEVM 部署赢·只落 decode regime·非 prefill)
            dep = Q5K1_DEPLOY.get((op,fmt))
            depnote=""
            if dep and dep[0]==board and regime in ("decode",""):
                depnote = " ·[★裁决④吸纳: deploy k1 repack-GEVM(decode) %.3f× kernel + e2e 2×(独立验证 a6fdf1a3/w91jl99ia)·anti-gate: 真实部署路径新格(C1 per-format measured-gate·d109d6ed2)·成色 beat-weak-baseline(stock q5 block-dot compute-bound·非 beat-hand-tuned)]"%dep[1]
                if d.startswith("具名-X") or d.startswith("pending"): d="PASS-DEPLOYED(部署路·k1 repack decode-GEVM)"
            # ★线A·A1 单世界 clang-18 override(PR-17 终裁·gcc 视为不存在·rvv board·终态优先·仅次于 na)
            #   承 gelu/GEMM_DECODE/IME_KERNELSYM stale-source 更正范式·12 翻转格按 clang 世界入账
            cw = CLANG_WORLD.get((op,fmt,regime))
            if cw and board=="rvv":
                c, cwnote = cw
                d = "PASS" if c>=0.8 else "具名-X"
                note = cwnote
            if na: tier="N/A-hw"; sym="—"; note="ime.present unsatisfiable on %s(机判)"%board; d="N/A-hw"
            rec[board] = {"tier":tier,"sym":sym,"note":note+depnote,"cold":c,"disp":d,"na":na}
        master.append(rec)

    # ---- group subtotal recon ----
    grp = Counter(r["group"] for r in master)
    # ---- N/A-hw legal-asymmetry scan ----
    asym = []; rev_asym = []
    for r in master:
        rk=f"{r['op']}|{r['format']}|{r['engine']}"+(f"|{r['regime']}" if r['regime'] else "")
        if r["rvv"]["na"] and not r["k1"]["na"]: asym.append((rk,"rvv=N/A-hw"))
        if r["k1"]["na"] and not r["rvv"]["na"]: rev_asym.append((rk,"k1=N/A-hw"))

    def row_state(d):
        if d.startswith("PASS"): return "PASS"
        if d.startswith("具名-X"): return "具名-X"
        if "挂起" in d: return "挂起"
        return "pending"
    TIERS=("手调","通用向量","标量类","UNRESOLVED")
    # ★单分母制(数字字典v2令): 板分母 = 主表 非N/A-hw ∧ 非q1_0(域外) 全部行. 四档穷尽互斥·Σ=分母.
    def board_denom_rows(board):
        return [r for r in master if (not r[board]["na"]) and r[board]["tier"]!="域外"]
    def tier_stats(board):
        st={t:{"full":0,"pass":0,"measured":0,"pending":0,"susp":0,"x":0,"rows":[]} for t in TIERS}
        for r in board_denom_rows(board):
            t=r[board]["tier"]
            if t not in st: t="UNRESOLVED"
            d=r[board]["disp"]; c=r[board]["cold"]; stt=row_state(d)
            s=st[t]; s["full"]+=1
            if stt=="PASS": s["pass"]+=1; s["measured"]+=1
            elif stt=="具名-X": s["x"]+=1; s["measured"]+=1
            elif stt=="挂起": s["susp"]+=1
            else: s["pending"]+=1
            rk=f"{r['op'].replace('_tile','').replace('ntize_row','ntize').replace('quantize_row','quant').replace('uct_reduce','_reduce')}|{r['format']}"+(f"@{r['engine']}" if r['engine'] else "")+(f"/{r['regime']}" if r['regime'] else "")
            s["rows"].append((rk,stt,c))
        return st

    # ---- write master CSV (含 regime 列) ----
    with open(OUT,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["op","format","engine","regime","group",
                    "rvv_tier","rvv_disp","rvv_cold","rvv_opp_sym","rvv_note",
                    "k1_tier","k1_disp","k1_cold","k1_opp_sym","k1_note"])
        for r in master:
            w.writerow([r["op"],r["format"],r["engine"],r["regime"],r["group"],
                r["rvv"]["tier"],r["rvv"]["disp"],r["rvv"]["cold"],r["rvv"]["sym"],r["rvv"]["note"],
                r["k1"]["tier"],r["k1"]["disp"],r["k1"]["cold"],r["k1"]["sym"],r["k1"]["note"]])

    # ---- row-clue 工件 (每口径数字附机打行清单·禁无清单出数·数字字典v2令一.4) ----
    stats={b:tier_stats(b) for b in ("rvv","k1")}
    denom={b:len(board_denom_rows(b)) for b in ("rvv","k1")}
    with open(CLUE,"w") as cf:
        cf.write(f"# 行清单工件 (数字字典v2·每口径行清单+格内状态·机算·snapshot {SNAPSHOT})\n")
        cf.write(f"# 单分母制: 板分母 = 主表非N/A-hw ∧ 非q1_0 全部行. 四档穷尽互斥 Σ=分母.\n\n")
        for board in ("rvv","k1"):
            cf.write(f"===== {board} (板分母={denom[board]}) =====\n")
            tot=0
            for t in TIERS:
                s=stats[board][t]; tot+=s["full"]
                cf.write(f"\n[{t}] 头条 PASS {s['pass']}/{s['full']} (全量) · 已测胜率 {s['pass']}/{s['measured']} (派生) · 格内: PASS {s['pass']}/具名-X {s['x']}/挂起 {s['susp']}/pending {s['pending']}\n")
                for rk,stt,c in s["rows"]:
                    cf.write(f"    {stt:7s} {rk}"+(f" cold={c}" if c is not None else "")+"\n")
            cf.write(f"\n  Σ四档 = {tot} == 板分母 {denom[board]} : {'✓' if tot==denom[board] else '✗MISMATCH'}\n\n")

    # ---- print recon ----
    print("="*76)
    print(f"G8 主表重铸 RECON v2·单分母制 (机算·snapshot {SNAPSHOT}) — {len(master)} 行 master")
    print("="*76)
    print("\n[分组行数对账] (matmul/forward/dequant/quantize/product_reduce·Σ=主表总行数)")
    for g in ("matmul","forward","dequant","quantize","product_reduce"):
        print(f"    {g:16s} {grp[g]}")
    print(f"    {'─'*24}\n    Σ = {sum(grp.values())}  (roster in-domain 91·q4_0-gemm decode/prefill=2行不折·[K-10])")
    print(f"    matmul 细分: gemm {sum(1 for r in master if r['op']=='gemm_tile')} + vec_dot {sum(1 for r in master if r['op']=='vec_dot')}")

    print("\n[合法不对称清单] (N/A-hw·机判 ime.present×板实例·禁手标):")
    for rk,tag in asym: print(f"    {rk}  {tag}")
    print(f"    ---- N/A-hw 行数 = {len(asym)} (预期 IME 3 @rvv) ----")
    print(f"    反向确认(rvv合法∧k1缺席): {len(rev_asym)} 行 {'✓无' if not rev_asym else rev_asym}")

    print("\n[★单分母头条·四档穷尽互斥·PASS/全档量·Σ=板分母] (便宜档禁称硬赢):")
    for board in ("rvv","k1"):
        St=stats[board]; tot=sum(St[t]['full'] for t in TIERS)
        print(f"  ── {board} (板分母={denom[board]}·非N/A-hw∧非q1_0) ──")
        for t in TIERS:
            s=St[t]
            wr = f"已测胜率{s['pass']}/{s['measured']}" if s['measured'] else "已测胜率 n/a"
            print(f"      {t:10s} 头条 PASS {s['pass']:2d}/{s['full']:2d} (全量) · {wr} · [PASS{s['pass']}/X{s['x']}/挂{s['susp']}/pend{s['pending']}]"+("  ★禁称硬赢" if t=="标量类" else ""))
        print(f"      Σ四档 = {tot} == 分母 {denom[board]} : {'✓' if tot==denom[board] else '✗MISMATCH!!'}")

    # ---- 数字字典 v2 (交付物 B·单分母·每数=过滤器) ----
    print("\n" + "="*76)
    print("数字字典 v2 (单分母制·每数=主表过滤器·机算·snapshot "+SNAPSHOT+"·行清单见 "+CLUE.split('/')[-1]+")")
    print("="*76)
    rows_dict=[("master 总行数","(op,format,engine,regime)·in-domain·q4_0-gemm不折·flash_attn/bf16 OOD除",len(master),"主表口径"),
     ("板分母-rvv","非N/A-hw ∧ 非q1_0(域外) 全部行",denom["rvv"],"单分母"),
     ("板分母-k1","同上",denom["k1"],"单分母")]
    for board in ("rvv","k1"):
        for t in TIERS:
            s=stats[board][t]
            rows_dict.append((f"{t}-{board}",f"tier={t} ∧ 板分母·PASS/全档量"+("·禁称硬赢" if t=="标量类" else ""),f"{s['pass']}/{s['full']}",f"头条·{board}"))
    rows_dict += [
     ("N/A-hw","ime.present谓词×板实例不满足·机判",len(asym),"两板对称"),
     ("certified","(op,format[,shape,regime])构造轴·byte-exact·【coverage_metrics.py 机算·非本recon·regime-split后】","101/108","构造轴(外·F-1纠)"),
     ("perf-covered","(op,format,engine)系统账e2e·perf_covered_metrics.py·【收口令〇.1终裁维持·any-board只升成色】","9/83","系统账(外)"),
     ("真硬赢 hand-brick","手调-REAL ∧ PASS ∧ k1·byte-verified·【判据锁 q4_K/q2_K@k1·q5_K/q6_K DEFERRED非verified·非本recon机算】","2","成色定性(F-3判据守护)")]
    for name,filt,val,track in rows_dict:
        print(f"  {name:16s} = {str(val):9s} | {track:12s} | {filt}")
    print("\n  [分组行数对账] Σ = "+" + ".join(f"{g}:{grp[g]}" for g in ('matmul','forward','dequant','quantize','product_reduce'))+f" = {sum(grp.values())}")
    print(f"\n★ master CSV: {OUT}\n★ 行清单工件: {CLUE}")

if __name__=="__main__":
    main()
