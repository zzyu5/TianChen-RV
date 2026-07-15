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
T3A = ROOT + "/experiments/active/result-tables/T3_A_board_A_rvv1.0_vlen128.csv"
T3B = ROOT + "/experiments/active/result-tables/T3_B_board_B_rvv1.0_vlen256.csv"
OUT = ROOT + "/experiments/active/result-tables/T3_master_rebuild.csv"
SNAPSHOT = "g8-master-rebuild-desk-round"
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
        if "JUDGMENT-SUSPENDED" in v: d="JUDGMENT-SUSPENDED"
        elif "test-only-not-in-denom" in v: d="DEQ-照测-not-in-denom"
        elif "NAMED-X" in v or "具名" in v: d="具名-X"
        elif re.search(r'(in-denom|;|\*)PASS',v): d="PASS"
        elif "FAIL" in v: d="具名-X"
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
 # ---- iq/tq vec_dot (pending·vec_dot _vlNNN 符号【从未反汇编】·仅 dequant 反了·禁猜→UNRESOLVED〔V-纠〕·§六补探针) ----
 ("vec_dot","iq2_xxs"):{"rvv":(U,"ggml_vec_dot_iq2_xxs_q8_K(vec_dot符号未反汇编)","仅dequant反了·手调=类推非符号级→UNRESOLVED〔V〕"),"k1":(U,"(未反汇编)","UNRESOLVED〔V〕")},
 ("vec_dot","iq2_xs"): {"rvv":(U,"ggml_vec_dot_iq2_xs_q8_K(未反汇编)","UNRESOLVED〔V〕"),"k1":(U,"(未反汇编)","UNRESOLVED〔V〕")},
 ("vec_dot","iq2_s"):  {"rvv":(U,"ggml_vec_dot_iq2_s_q8_K(未反汇编)","UNRESOLVED〔V〕"),"k1":(U,"(未反汇编)","UNRESOLVED〔V〕")},
 ("vec_dot","iq3_xxs"):{"rvv":(U,"ggml_vec_dot_iq3_xxs_q8_K(未反汇编)","UNRESOLVED〔V〕"),"k1":(U,"(未反汇编)","UNRESOLVED〔V〕")},
 ("vec_dot","iq3_s"):  {"rvv":(U,"ggml_vec_dot_iq3_s_q8_K(未反汇编)","UNRESOLVED〔V〕"),"k1":(U,"(未反汇编)","UNRESOLVED〔V〕")},
 ("vec_dot","iq4_xs"): {"rvv":(U,"ggml_vec_dot_iq4_xs_q8_K(未反汇编)","pending cold·符号未反汇编·UNRESOLVED〔V〕"),"k1":(U,"(未反汇编)","UNRESOLVED〔V〕")},
 ("vec_dot","mxfp4"):  {"rvv":(H,"ggml_vec_dot_mxfp4_q8_0_vl128","★实已反汇编 rvv=18/mac4/gather2·同iq4_nl profile·hand-vl128 LIGHT→手调〔V-纠〕"),"k1":(U,"(k1未反汇编)","UNRESOLVED〔V〕")},
 ("vec_dot","tq1_0"):  {"rvv":(U,"ggml_vec_dot_tq1_0_q8_K(vec_dot未反汇编)","仅dequant反了·UNRESOLVED〔V〕"),"k1":(U,"(未反汇编)","UNRESOLVED〔V〕")},
 ("vec_dot","tq2_0"):  {"rvv":(U,"ggml_vec_dot_tq2_0_q8_K(未反汇编)","UNRESOLVED〔V〕"),"k1":(U,"(未反汇编)","UNRESOLVED〔V〕")},
 ("vec_dot","q1_0"):   {"rvv":(D,"ggml_vec_dot_q1_0_q8_0_vl128(Weft-internal)","internal-A/B·自己当自己靶禁·永久域外"),"k1":(D,"(Weft-internal)","永久域外")},
 # ---- K-quant GEMM (rvv 全 cross-op vs block-dot·k1 4 real repack + q3_K cross-op) ----
 ("gemm_tile","q2_K"):{"rvv":(H,"ggml_vec_dot_q2_K_q8_K_vl128(CROSSOP)","our-gemm vs vl128 手调block-dot·cross-op(rvv零K-quant repack)"),"k1":(H,"ggml_gemm_q2_K_8x8_q8_K(REAL)","真16x1/8x8 hand-brick repack STRONG")},
 ("gemm_tile","q3_K"):{"rvv":(H,"ggml_vec_dot_q3_K_q8_K_vl128(CROSSOP)","cross-op vs vl128 手调block-dot"),"k1":(H,"ggml_vec_dot_q3_K_q8_K_vl256(CROSSOP)","q3_K唯一无repack·cross-op vs vl256 手调block-dot")},
 ("gemm_tile","q4_K"):{"rvv":(H,"ggml_vec_dot_q4_K_q8_K_vl128(CROSSOP)","cross-op vs vl128 手调block-dot"),"k1":(H,"ggml_gemm_q4_K_16x1_q8_K(REAL)","真16x1 hand-brick byte-drop-in STRONG(Win-K1-VLEN)")},
 ("gemm_tile","q5_K"):{"rvv":(V,"ggml_vec_dot_q5_K_q8_K(CROSSOP)","cross-op vs native-vec-moderate(q5_K无vl-spec)"),"k1":(H,"ggml_gemm_q5_K_8x4_q8_K(REAL)","真 hand-brick repack STRONG")},
 ("gemm_tile","q6_K"):{"rvv":(H,"ggml_vec_dot_q6_K_q8_K_vl128(CROSSOP)","cross-op vs vl128 手调block-dot"),"k1":(H,"ggml_gemm_q6_K_16x1_q8_K(REAL)","真 hand-brick repack STRONG")},
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
 ("gemm_tile","q4_0"):{"rvv":(U,"upstream repack / block-dot(rvv objdump零gemm符号)","对手候选二义:block-dot(通用向量·repack-off) vs repack(手调/标量未验)→UNRESOLVED〔V-纠〕·decode+prefill同行双子行·e2e 5.92×/1.91×"),"k1":(V,"block-dot","pending-fold")},
 ("gemm_tile","q4_1"):{"rvv":(V,"ggml_vec_dot_q4_1_q8_1(CROSSOP)","block-dot vectorized·nibble-unpack非autovec-able→通用向量〔V-纠〕·e2e 3.68×"),"k1":(V,"ggml_vec_dot_q4_1_q8_1(CROSSOP)","block-dot→通用向量〔V-纠〕·pending-fold")},
 ("gemm_tile","q5_0"):{"rvv":(V,"FLAT block-dot","e2e prefill 1.21×·pending-fold"),"k1":(V,"FLAT block-dot","pending-fold")},
 ("gemm_tile","q5_1"):{"rvv":(V,"FLAT block-dot","e2e prefill 1.09×·pending-fold"),"k1":(V,"FLAT block-dot","pending-fold")},
 ("gemm_tile","q8_0"):{"rvv":(V,"FLAT block-dot","e2e 4.35× correctness-carrier·pending-fold"),"k1":(V,"FLAT block-dot","pending-fold")},
 # ---- forward (源码归属·§一.B + opponent_ggml.cpp) ----
 ("forward","softmax"):{"rvv":(V,"ggml_vec_soft_max_f32","exported ggml手写RVV intrinsic·expf reduce"),"k1":(V,"ggml_vec_soft_max_f32","leaf 手写intrinsic·wrapper scalar-setup")},
 ("forward","rms_norm"):{"rvv":(V,"ggml_compute_forward_rms_norm_f32","native RVV m8 vec_scale + scalar-dbl reduce·HYBRID·BORDERLINE"),"k1":(V,"ggml_compute_forward_rms_norm_f32","HYBRID·native-vec-light")},
 ("forward","rope"):   {"rvv":(S,"ggml_compute_forward_rope_f32","scalar cos/sin cache 2-pass + autovec rotate→标量类"),"k1":(S,"ggml_compute_forward_rope_f32","mostly-scalar→标量类")},
 ("forward","silu"):   {"rvv":(V,"ggml_vec_silu_f32","exported ggml手写RVV intrinsic·sigmoid/expf 向量化"),"k1":(V,"ggml_vec_silu_f32","手写intrinsic")},
 ("forward","gelu"):   {"rvv":(S,"ggml_table_gelu_f16(BSS-LUT)","f16 LUT 查表·结构差·数值档不对等·待f16-LUT同档重比(§一.4)"),"k1":(S,"ggml_table_gelu_f16(BSS-LUT)","f16 LUT·待同档重比")},
 ("forward","add"):    {"rvv":(S,"ggml_vec_add_f32(vec.h:89)","AVX2-only vec path·RV落scalar loop→autovec→标量类"),"k1":(S,"ggml_vec_add_f32","scalar loop autovec→标量类")},
 ("forward","mul"):    {"rvv":(S,"ggml_vec_mul_f32(vec.h:127)","pure scalar loop→autovec→标量类"),"k1":(S,"ggml_vec_mul_f32","autovec→标量类")},
 ("forward","scale"):  {"rvv":(V,"ggml_vec_scale_f32(vec.h:703)","NATIVE RVV m8 vfmul_vf 手写intrinsic·byte-identical algo"),"k1":(V,"ggml_vec_scale_f32","native RVV m8·板异emit较轻但同源intrinsic")},
 ("forward","cpy"):    {"rvv":(S,"ggml_vec_cpy_f32(vec.h:119)","pure scalar loop→autovec/memcpy→标量类"),"k1":(S,"ggml_compute_forward_dup_cpy","scalar/memcpy-stream→标量类")},
}
# dequant (24) — 全 标量类: 源=scalar dequantize_row_*·任何向量化=autovec(§〇.1)·gcc15.2/clang18 per-format autovec·pre-clang18 stale
# quantize (3) — 标量类(scalar quantize_row_* autovec)·flag verify
# product_reduce (3) — §〇.2 ggml scalar-ref fallback·internal sub-primitive·待补标量仗
DEQ_NOTE_RVV = {"q4_K":"clang18 rvv0 真SCALAR·deploy gcc15.2 autovec footnote"}

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
    # build master rows: (op,format,engine), fold q4_0 gemm decode/prefill -> 1 row
    seen = set(); rows = []
    for k in roster:
        op, fmt, eng = k["op"], k["format"], k.get("engine","")
        if op in ("flash_attn","bf16"): continue          # class C OOD (not in 5 groups)
        key = (op, fmt, eng)
        if op == "gemm_tile" and fmt == "q4_0" and eng == "rvv":
            key = ("gemm_tile","q4_0","rvv")               # decode+prefill fold to 1
        if key in seen: continue
        seen.add(key)
        rows.append({"op":op,"format":fmt,"engine":eng})

    master = []
    for r in rows:
        op, fmt, eng = r["op"], r["format"], r["engine"]
        isfwd = op in FWD_OPS
        tkey = ("forward", op) if isfwd else (op, fmt)
        tent = TIER.get(tkey)
        rec = {"op":op,"format":fmt,"engine":eng,"group":op_group(op)}
        for board in ("rvv","k1"):
            na = na_hw(op,fmt,eng,board)
            if op == "dequantize_row":
                tier = S; sym = "dequantize_row_%s"%fmt
                note = DEQ_NOTE_RVV.get(fmt,"") if board=="rvv" else ""
                note = (note+" · " if note else "")+"scalar源 autovec→标量类(§〇.1)·per-format gcc15.2/clang18·pre-clang18 stale"
            elif op == "quantize_row":
                tier=V; sym="quantize_row_%s"%fmt; note="★ggml-cpu/arch/riscv/quants.c 手写__riscv_v intrinsic(非scalar autovec)→通用向量〔V-纠·source铁证·objdump待§六补〕"
            elif op == "product_reduce":
                tier=S; sym="ggml scalar-ref(fallback)"; note="internal sub-primitive·§〇.2兜底·待补标量仗"
            elif tent:
                tier, sym, note = tent[board]
            else:
                tier, sym, note = U, "(no-map)", "UNRESOLVED"
            # cold/disp: prefer T3-measured verdict; else pending
            t3rec = t3[board].get((op,fmt))
            if t3rec and t3rec["disp"]!="?":
                c = t3rec["cold"]
                td = t3rec["disp"]
                if op=="dequantize_row":
                    # DEQ 照测·标量仗·不进头条(§一.3 filter view)
                    if td=="PASS": d="PASS(照测·标量仗·不进头条0.8)"
                    elif td=="具名-X": d="具名-X(照测·标量仗)"
                    elif td=="DEQ-照测-not-in-denom": d="DEQ照测(not-in-denom·标量仗)"
                    else: d="DEQ照测"
                elif td=="JUDGMENT-SUSPENDED":
                    d = "例外-数值档挂起(gelu·待f16-LUT同档重比§一.4)"
                elif td=="PASS": d="PASS"
                elif td=="具名-X": d="具名-X"
                elif td=="DEQ-照测-not-in-denom": d="DEQ照测(not-in-denom·标量仗)"
                else: d = disp(op,fmt,eng,board,tier,c,na)
            else:
                c = None
                d = disp(op,fmt,eng,board,tier,c,na)
            # q5@k1 deploy absorb
            dep = Q5K1_DEPLOY.get((op,fmt))
            depnote=""
            if dep and dep[0]==board:
                depnote = " ·[deploy k1 repack %.3f× PASS-DEPLOYED·anti-gate:真部署路新格·beat-weak-baseline]"%dep[1]
                if d.startswith("具名-X"): d="PASS-DEPLOYED(部署路·k1 repack)"
            if na: tier="N/A-hw"; sym="—"; note="ime.present unsatisfiable on %s(机判)"%board; d="N/A-hw"
            rec[board] = {"tier":tier,"sym":sym,"note":note+depnote,"cold":c,"disp":d,"na":na}
        master.append(rec)

    # ---- group subtotal recon ----
    grp = Counter(r["group"] for r in master)
    # ---- N/A-hw legal-asymmetry scan ----
    asym = []
    rev_asym = []  # rvv-legal but k1-absent (expect none)
    for r in master:
        if r["rvv"]["na"] and not r["k1"]["na"]:
            asym.append((r["op"],r["format"],r["engine"],"rvv=N/A-hw"))
        if r["k1"]["na"] and not r["rvv"]["na"]:
            rev_asym.append((r["op"],r["format"],r["engine"],"k1=N/A-hw"))
    def is_pass(d): return d.startswith("PASS")
    def is_x(d): return d.startswith("具名-X")
    def contested(d): return is_pass(d) or is_x(d)
    # ---- three-headline: 硬仗(手调)/向量仗(通用向量)/标量仗(标量类)·per board·PASS/局数 ----
    # 头条 = matmul + forward 主 0.8 对局(DEQ照测不进头条·§一.3·单列子账)
    def board_stats(board):
        buckets = {"手调":[0,0],"通用向量":[0,0],"标量类":[0,0],"UNRESOLVED":[0,0]}  # [pass, contest]
        na=oob=pend=susp=0; deq=[0,0]
        for r in master:
            b=r[board]; t=b["tier"]; d=b["disp"]
            if b["na"]: na+=1; continue
            if t=="域外" or d.startswith("域外"): oob+=1; continue
            if r["op"]=="dequantize_row":
                if contested(d): deq[1]+=1;  deq[0]+= (1 if is_pass(d) else 0)
                continue
            if "挂起" in d: susp+=1; continue
            if contested(d):
                if t in buckets:
                    buckets[t][1]+=1
                    if is_pass(d): buckets[t][0]+=1
            else:
                pend+=1
        return buckets, na, oob, pend, susp, deq
    # ---- per-board 局数分母(补充令.5): 该板列中 非N/A-hw ∧ 有对局(matmul+forward头条) 的行数 ----
    def board_contest_denom(board):
        return sum(1 for r in master if (not r[board]["na"]) and r["op"]!="dequantize_row"
                   and contested(r[board]["disp"]))

    # ---- write master CSV ----
    with open(OUT,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["op","format","engine","group",
                    "rvv_tier","rvv_disp","rvv_cold","rvv_opp_sym","rvv_note",
                    "k1_tier","k1_disp","k1_cold","k1_opp_sym","k1_note"])
        for r in master:
            w.writerow([r["op"],r["format"],r["engine"],r["group"],
                r["rvv"]["tier"],r["rvv"]["disp"],r["rvv"]["cold"],r["rvv"]["sym"],r["rvv"]["note"],
                r["k1"]["tier"],r["k1"]["disp"],r["k1"]["cold"],r["k1"]["sym"],r["k1"]["note"]])

    # ---- print recon ----
    print("="*76)
    print(f"G8 主表重铸 RECON (机算·snapshot {SNAPSHOT}) — {len(master)} 行 master")
    print("="*76)
    print("\n[分组行数对账] (matmul/forward/dequant/quantize/product_reduce·Σ=主表总行数)")
    for g in ("matmul","forward","dequant","quantize","product_reduce"):
        print(f"    {g:16s} {grp[g]}")
    print(f"    {'─'*24}\n    Σ = {sum(grp.values())}  (roster in-domain 91 − q4_0-gemm-regime-fold 1 = 90)")
    print(f"    matmul 细分: gemm {sum(1 for r in master if r['op']=='gemm_tile')} (24 rvv格+3 ime·q4_0 regime折) + vec_dot {sum(1 for r in master if r['op']=='vec_dot')}")

    print("\n[合法不对称清单] (N/A-hw·机判 ime.present×板实例·禁手标):")
    for a in asym: print(f"    {a[0]}|{a[1]}|{a[2]}  {a[3]}")
    print(f"    ---- N/A-hw 行数 = {len(asym)} (预期 IME 3 @rvv) ----")
    print(f"    反向确认(rvv合法∧k1缺席): {len(rev_asym)} 行 {'✓无' if not rev_asym else rev_asym}")

    print("\n[三口径头条·per board·PASS/局数] (硬仗=手调·向量仗=通用向量·标量仗=标量类·DEQ照测不进头条):")
    for board in ("rvv","k1"):
        bk,na,oob,pend,susp,deq = board_stats(board)
        denom = board_contest_denom(board)
        print(f"  ── {board} (头条局数分母={denom}·非N/A-hw∧有对局∧非DEQ照测) ──")
        for t in ("手调","通用向量","标量类","UNRESOLVED"):
            p,c = bk[t]
            print(f"      {t:10s} PASS {p}/{c}"+("  ★禁称硬赢(便宜档)" if t=="标量类" and c else ""))
        print(f"      [DEQ照测子账(标量仗·不进头条) PASS {deq[0]}/{deq[1]} · gelu挂起 {susp} · N/A-hw {na} · 域外 {oob} · pending {pend}]")

    # ---- disposition census ----
    print("\n[disposition 普查·双板]:")
    for board in ("rvv","k1"):
        dc=Counter(r[board]["disp"] for r in master)
        print(f"  {board}: "+" · ".join(f"{k}:{v}" for k,v in sorted(dc.items(), key=lambda x:-x[1])))

    # ---- tier census ----
    print("\n[对手三档普查·双板·非N/A-hw]:")
    for board in ("rvv","k1"):
        tc=Counter(r[board]["tier"] for r in master if not r[board]["na"])
        print(f"  {board}: "+" · ".join(f"{k}:{v}" for k,v in sorted(tc.items(), key=lambda x:-x[1])))

    # ---- 数字字典 (交付物 B·每头条数=主表一个过滤器) ----
    def cnt(pred): return sum(1 for r in master if pred(r))
    rvvS = board_stats("rvv"); k1S = board_stats("k1")
    print("\n" + "="*76)
    print("数字字典 (交付物 B·每数=主表过滤器·机算·snapshot "+SNAPSHOT+")")
    print("="*76)
    dictrows = [
     ("master 总行数","(op,format,engine)·in-domain·q4_0-gemm-regime折·flash_attn/bf16 OOD除",len(master),"主表口径"),
     ("硬仗-rvv","tier=手调 ∧ 头条(matmul+forward) ∧ 非N/A-hw·PASS/局数",f"{rvvS[0]['手调'][0]}/{rvvS[0]['手调'][1]}","0.8硬仗"),
     ("硬仗-k1","同上",f"{k1S[0]['手调'][0]}/{k1S[0]['手调'][1]}","0.8硬仗"),
     ("向量仗-rvv","tier=通用向量 ∧ 头条·PASS/局数",f"{rvvS[0]['通用向量'][0]}/{rvvS[0]['通用向量'][1]}","0.8向量仗"),
     ("向量仗-k1","同上",f"{k1S[0]['通用向量'][0]}/{k1S[0]['通用向量'][1]}","0.8向量仗"),
     ("标量仗-rvv","tier=标量类 ∧ 头条·PASS/局数·★禁称硬赢",f"{rvvS[0]['标量类'][0]}/{rvvS[0]['标量类'][1]}","0.8标量仗"),
     ("标量仗-k1","同上",f"{k1S[0]['标量类'][0]}/{k1S[0]['标量类'][1]}","0.8标量仗"),
     ("DEQ照测-rvv","op=dequant·标量仗·不进头条·PASS/局数",f"{rvvS[5][0]}/{rvvS[5][1]}","DEQ子账"),
     ("DEQ照测-k1","同上(k1 0 promoted·§〇.1下与rvv同为标量仗照测)",f"{k1S[5][0]}/{k1S[5][1]}","DEQ子账"),
     ("头条局数分母-rvv","该板非N/A-hw ∧ 有对局(matmul+forward) 行数",board_contest_denom("rvv"),"补充令.5"),
     ("头条局数分母-k1","同上",board_contest_denom("k1"),"补充令.5"),
     ("N/A-hw(合法不对称)","ime.present谓词×板实例不满足·机判",len(asym),"两板对称"),
     ("pending-总-rvv","pending-fold + pending-真",cnt(lambda r:r['rvv']['disp'].startswith('pending')),"待补战线"),
     ("certified","(op,format[,shape])构造轴·byte-exact/emit-golden·coverage_metrics.py","84/91","构造轴(外·非0.8)"),
     ("perf-covered","(op,format,engine)系统账e2e·perf_covered_metrics.py","9/83","系统账(外·非kernel-sym)"),
     ("真硬赢 hand-brick","手调-REAL ∧ PASS ∧ k1·byte-verified","2","成色定性(q4_K/q2_K@k1)"),
    ]
    for name,filt,val,track in dictrows:
        print(f"  {name:18s} = {str(val):9s} | {track:16s} | 过滤器: {filt}")
    print("\n  [分组行数对账] Σ = "+" + ".join(f"{g}:{grp[g]}" for g in ('matmul','forward','dequant','quantize','product_reduce'))+f" = {sum(grp.values())}")
    print(f"\n★ master CSV: {OUT}")

if __name__=="__main__":
    main()
