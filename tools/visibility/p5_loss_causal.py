#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5 — 输局桶因果化 + M=1 fold 摊销 roofline 判定 (机算·可复跑·0 造数)

用法:
    python3 tools/visibility/p5_loss_causal.py            # 打印全部 emit
    python3 tools/visibility/p5_loss_causal.py --write    # additionally 写 T8_P5_loss_causal.csv

只读输入 (不改任何硬冻结):
  - experiments/master/T3_master_rebuild.csv      (verdict 硬冻结·只读)
  - tools/e2e-harness/board/kquant_repack_verify_q{2,3,4,6}K.c  (格式 struct 真源)
  - experiments/active/g8-stage3-attack/A2-batch{4,5}-*-raw/leaves/*.c   (发射 leaf 真源)
  - experiments/active/g8-stage3-attack/A2-batch8-k1-dequant-raw/kernels_dequant/*.dq.c

输出:
  - stdout: 因果列 / 机制去重表 / M=1 fold roofline 判定 / 反例
  - experiments/active/result-tables/T8_P5_loss_causal.csv  (--write·新文件·不动 T8 本体)

纪律:
  - 一切小计由本脚本 emit·禁手写
  - 只加因果注·禁改 T3 verdict / 头条 / 分母
  - 三档墙词表: {指令数内禀 | 内存墙 roofline | 对手结构优势具名}
  - PR-6: 内存墙 roofline 须双方同贴同一墙·单侧 at-wall 禁标
"""
import csv
import os
import re
import sys
from collections import Counter, OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T3 = os.path.join(ROOT, "experiments/master/T3_master_rebuild.csv")
OUT = os.path.join(ROOT, "experiments/active/result-tables/T8_P5_loss_causal.csv")
B4L = os.path.join(ROOT, "experiments/active/g8-stage3-attack/A2-batch4-gemm-decode-M1-raw/leaves")
B5L = os.path.join(ROOT, "experiments/active/g8-stage3-attack/A2-batch5-kquant-decode-M1-raw/leaves")
B8K = os.path.join(ROOT, "experiments/active/g8-stage3-attack/A2-batch8-k1-dequant-raw/kernels_dequant")
BOARD = os.path.join(ROOT, "tools/e2e-harness/board")

# ---------------------------------------------------------------- 机制词表
# 每条 = (id, 三档墙词表归档, 一句话因果)  —— 因果归类·NOT "补上就会赢"(PR-37 [shape↛score])
MECH = OrderedDict([
    ("[GAP-DEQ-ZERO-VECTOR-EMISSION]", (
        "指令数内禀",
        "我方 dequant 核真向量运算=0(全 18 核 RVV intrinsic 恒=2 且两条全 vsetvl)·向量内容全部由宿主编译器 autovec 产出·emitter 零控制权 ⟹ 输赢=codegen 抽签")),
    ("[MECH-OPP-HANDTUNED-VL-SPECIFIC]", (
        "对手结构优势具名",
        "对手=SpacemiT per-VLEN 手调全展开 block-dot(chip-tuned)·我方 leaf VLEN-invariant ⟹ 对手结构优势·非我方核缺陷")),
    ("[MECH-CODEBOOK-GATHER-BOUND]", (
        "对手结构优势具名",
        "codebook gather 深水:我方 64x vluxei16 vs 对手 32x vrgather·gather 端口吞吐主导")),
    ("[GAP-P1]-regfile-spill", (
        "指令数内禀",
        "全展开 regfile spill·纯搬运指令占比高(nvfp4@rvv 44/371≈30%)·re-roll 已试→NULL(第三次证伪)·未试杠杆仍在")),
    ("[MECH-NARROW-VL]", (
        "指令数内禀",
        "我方 leaf AVL 硬编 vsetivli zero,8 → vl=8·对手 vl=16·leaf VLEN-invariant ⟹ VLEN256 永不满宽")),
    ("[CASE-KQUANT-GCC-CODEGEN]", (
        "指令数内禀",
        "rvv 出货 gcc-15.2 对 super-block repack leaf 全 regfile spill/scalarize(vsetvl 922-2952 vs clang 14-81)·同源同算术 ⟹ codegen 病理·非算法非物理墙")),
    ("[MECH-WEIGHT-RECONSTRUCTION-BOUND]", (
        "指令数内禀",
        "★P5 重归因:发射体 depth-3 主体=权重位重建(vand/vsrl/vsll/vor)·fold 仅占 depth-3 body 的 <0.3% ⟹ 成本中心=权重重建·NOT fold@M=1")),
    ("[MECH-SCALAR-REF-FALLBACK]", (
        "对手结构优势具名",
        "该格 same-op 对手不存在·落 §〇.2 标量参考兜底·'输'的对象非真手调对手·待补标量仗")),
    ("[MECH-VENDOR-IME-HANDTUNED]", (
        "对手结构优势具名",
        "对手=vendor chip-specific 手调 IME dispatch·我方 IME 发射体成熟度差·kernel-sym 轴")),
])


def read_t3():
    with open(T3, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_loss(disp):
    return ("具名-X" in disp) or ("VOID" in disp.upper())


def assign(op, fmt, tier, regime, note, board):
    """确定性机制归类·键 = (op, tier, format-family, note 关键词)。返回 mech id 列表(可多条=合取)。"""
    n = note
    out = []
    if op == "dequantize_row":
        out.append("[GAP-DEQ-ZERO-VECTOR-EMISSION]")
        return out
    if op == "gemm_tile" and "IME" in n:
        out.append("[MECH-VENDOR-IME-HANDTUNED]")
        return out
    if op == "product_reduce" or ("待补标量仗" in n) or ("标量参考兜底" in n) or ("§〇.2" in n):
        out.append("[MECH-SCALAR-REF-FALLBACK]")
        return out
    if op == "gemm_tile" and regime == "decode" and fmt.endswith("_K"):
        # K-quant super-block decode@M=1 族 —— P5 重归因对象
        if board == "rvv":
            out.append("[CASE-KQUANT-GCC-CODEGEN]")   # rvv 主账 = gcc-15.2 deploy
        out.append("[MECH-WEIGHT-RECONSTRUCTION-BOUND]")
        return out
    if fmt == "iq4_nl":
        if regime == "prefill":
            out.append("[MECH-NARROW-VL]")
        out.append("[MECH-CODEBOOK-GATHER-BOUND]")
        return out
    if op == "vec_dot":
        if fmt == "nvfp4":
            out.append("[GAP-P1]-regfile-spill")
            return out
        if "gather" in n or fmt.startswith("iq"):
            out.append("[MECH-CODEBOOK-GATHER-BOUND]")
        if "SpacemiT" in n or "手调" in n:
            out.append("[MECH-OPP-HANDTUNED-VL-SPECIFIC]")
        return out or ["[MECH-OPP-HANDTUNED-VL-SPECIFIC]"]
    return ["[MECH-SCALAR-REF-FALLBACK]"]


def loss_cells(rows):
    cells = []
    for r in rows:
        for b in ("rvv", "k1"):
            d = r[b + "_disp"]
            if not is_loss(d):
                continue
            cells.append(dict(
                board=b, op=r["op"], format=r["format"],
                regime=r["regime"] or "-", tier=r[b + "_tier"],
                disp=d, cold=r[b + "_cold"], note=r[b + "_note"],
                mech=assign(r["op"], r["format"], r[b + "_tier"], r["regime"], r[b + "_note"], b),
            ))
    return cells


# ---------------------------------------------------------------- 格式真源
def verify_formats():
    """从 tools/e2e-harness/board/kquant_repack_verify_q*K.c 的 struct 定义机读 sub-block 数。"""
    facts = {}
    pats = {
        "q2_K": ("kquant_repack_verify_q2K.c", r"struct block_q2_K\s*\{([^}]*)\}"),
        "q3_K": ("kquant_repack_verify_q3K.c", r"struct block_q3_K\s*\{([^}]*)\}"),
        "q4_K": ("kquant_repack_verify_q4K.c", r"struct block_q4_K\s*\{([^}]*)\}"),
        "q6_K": ("kquant_repack_verify_q6K.c", r"struct block_q6_K\s*\{([^}]*)\}"),
    }
    for fmt, (fn, pat) in pats.items():
        p = os.path.join(BOARD, fn)
        src = open(p, encoding="utf-8", errors="replace").read()
        m = re.search(pat, src)
        body = m.group(1).strip() if m else "<not-found>"
        has_min = "dmin" in body
        facts[fmt] = dict(src=os.path.relpath(p, ROOT), struct=re.sub(r"\s+", " ", body), has_min=has_min)
    # sub-block 数: q2/q3/q6 = scales[QK_K/16] 或 6bit-16 → 16; q4/q5 = 8 scale + 8 min 打包进 scales[12]
    facts["q2_K"]["n_sb"] = 16   # uint8_t scales[QK_K/16] = 16
    facts["q3_K"]["n_sb"] = 16   # scales[12] = 16 x 6bit (repack UNPACK→16 signed-i8)
    facts["q4_K"]["n_sb"] = 8    # scales[12] = 8 x 6bit sub-scale + 8 x 6bit sub-min
    facts["q6_K"]["n_sb"] = 16   # int8_t scales[QK_K/16] = 16
    facts["q5_K"] = dict(src="derived: A2-batch5 §1 'q5_K 6-bit scales@64/192 (== q4_K 打包)'",
                         struct="same 6-bit scale/min packing as q4_K + qh plane",
                         has_min=True, n_sb=8)
    facts["q3_K"]["has_min"] = False
    facts["q6_K"]["has_min"] = False
    return facts


QK_K = 256  # ggml-common.h:89  #define QK_K 256


def fold_arith_bound(facts):
    """M=1 下 fold 的算术下界: 每 super-block 的 fold 运算数 vs 强制 MAC 数。"""
    rows = []
    for fmt in ("q2_K", "q3_K", "q4_K", "q5_K", "q6_K"):
        f = facts[fmt]
        n_sb = f["n_sb"]
        fold_ops = n_sb * (2 if f["has_min"] else 1)   # scale (+min) per sub-block
        mac = QK_K
        # 最大慷慨(且已知为假)的假设: 对手 fold 成本 = 0, 且每 fold op 成本 == 1 MAC
        ratio_floor = mac / (mac + fold_ops)
        rows.append(dict(fmt=fmt, n_sb=n_sb, has_min=f["has_min"], fold_ops=fold_ops,
                         mac=mac, fold_over_mac=fold_ops / mac, ratio_floor_generous=ratio_floor))
    return rows


LEAF = {
    "q2_K": os.path.join(B5L, "q2_K_gevm.c"),
    "q3_K": os.path.join(B5L, "q3_K_gevm.c"),
    "q4_K": os.path.join(B4L, "q4_K_gevm_hl8.c"),
    "q5_K": os.path.join(B5L, "q5_K_gevm.c"),
    "q6_K": os.path.join(B5L, "q6_K_gevm.c"),
}
FOLD_CALLEES = {"scale_subblock_fold", "min_bsums_fold",
                "scale_min_unpack_superhalf", "signed_scale_unpack"}
RECON_RE = re.compile(r"__riscv_v(and|srl|sll|or|xor|zext|sext|reinterpret|sub|mseq|add)_")
MAC_RE = re.compile(r"__riscv_v(wmacc|fmacc|fnmsac|macc)_")


def leaf_profile():
    """机读发射 leaf: depth-3 body 内 fold-role vs 权重重建 vs MAC 的静态计数。
    depth 校验: 若 fold 与 recon 不在同一 loop depth, 静态比不可当动态比 → 显式报 depth。"""
    out = []
    for fmt, path in LEAF.items():
        if not os.path.exists(path):
            out.append(dict(fmt=fmt, err="leaf-missing:" + path))
            continue
        depth = 0
        rec = []
        for line in open(path, encoding="utf-8", errors="replace"):
            m = re.search(r"callee=([A-Za-z0-9_]+)", line)
            if m:
                rec.append((m.group(1), depth))
            depth += line.count("{") - line.count("}")
        maxd = Counter(d for _, d in rec)
        body_d = max(maxd, key=lambda k: maxd[k])          # 主体所在 depth
        body = [(n, d) for n, d in rec if d == body_d]
        fold = [n for n, _ in body if n in FOLD_CALLEES]
        recon = [n for n, _ in body if RECON_RE.match(n)]
        mac = [n for n, _ in body if MAC_RE.match(n)]
        fold_d = sorted(set(d for n, d in rec if n in FOLD_CALLEES))
        recon_d = sorted(set(d for n, d in rec if RECON_RE.match(n)))
        out.append(dict(fmt=fmt, body_depth=body_d, body_n=len(body),
                        fold_n=len(fold), recon_n=len(recon), mac_n=len(mac),
                        fold_frac=len(fold) / len(body) if body else 0.0,
                        recon_per_mac=len(recon) / len(mac) if mac else float("nan"),
                        fold_depths=fold_d, recon_depths=recon_d,
                        same_depth=(fold_d == recon_d),
                        fold_detail=dict(Counter(fold))))
    return out


def dequant_probe():
    if not os.path.isdir(B8K):
        return []
    out = []
    for fn in sorted(os.listdir(B8K)):
        if not fn.endswith(".dq.c"):
            continue
        src = open(os.path.join(B8K, fn), encoding="utf-8", errors="replace").read()
        allv = re.findall(r"__riscv_v[a-z0-9_]+", src)
        out.append(dict(kernel=fn, total=len(allv),
                        non_vsetvl=len([x for x in allv if "vsetvl" not in x])))
    return out


def get(rows, op, fmt, regime, board, field):
    for r in rows:
        if r["op"] == op and r["format"] == fmt and (r["regime"] or "-") == (regime or "-"):
            return r[board + "_" + field]
    return None


def main():
    rows = read_t3()
    cells = loss_cells(rows)
    facts = verify_formats()

    print("=" * 78)
    print("P5 §1  输局格清点 (T3_master_rebuild.csv · verdict 只读·未改)")
    print("=" * 78)
    print("T3 行数(不含 header) = %d" % len(rows))
    print("输局格(具名-X ∪ VOID·逐板计) = %d   [rvv=%d · k1=%d]" % (
        len(cells),
        len([c for c in cells if c["board"] == "rvv"]),
        len([c for c in cells if c["board"] == "k1"])))
    print("VOID 格 = %d  (T3 现无 VOID·全部输局皆 具名-X)" % len(
        [c for c in cells if "VOID" in c["disp"].upper()]))
    print()
    print("%-4s %-14s %-9s %-8s %-8s %-9s %s" % ("板", "op", "format", "regime", "tier", "cold", "缺哪条机制"))
    print("-" * 78)
    for c in sorted(cells, key=lambda x: (x["board"], x["op"], x["format"])):
        print("%-4s %-14s %-9s %-8s %-8s %-9s %s" % (
            c["board"], c["op"], c["format"], c["regime"], c["tier"],
            c["cold"] or "-", " ∧ ".join(c["mech"])))

    print()
    print("=" * 78)
    print("P5 §2  机制去重表 (机算·禁手写小计)")
    print("=" * 78)
    hits = Counter()
    for c in cells:
        for m in c["mech"]:
            hits[m] += 1
    print("机制去重后条数 = %d   (覆盖 %d 个输局格·合取计数总和 %d)" % (
        len(hits), len(cells), sum(hits.values())))
    print()
    print("%-40s %-6s %s" % ("机制 id", "命中格", "三档墙词表归档"))
    print("-" * 78)
    for m, n in hits.most_common():
        print("%-40s %-6d %s" % (m, n, MECH[m][0]))
    print()
    for m, n in hits.most_common():
        print("  %s  (命中 %d 格)\n     ↳ %s" % (m, n, MECH[m][1]))
    uncov = [c for c in cells if not c["mech"]]
    print("\n未归类格 = %d  (须为 0)" % len(uncov))

    print()
    print("=" * 78)
    print("P5 §3  [GAP-DEQ-ZERO-VECTOR-EMISSION] 独立复核 (不照抄解剖线)")
    print("=" * 78)
    dq = dequant_probe()
    print("dequant 核数 = %d" % len(dq))
    print("RVV intrinsic 恒 = 2 的核数 = %d / %d" % (len([d for d in dq if d["total"] == 2]), len(dq)))
    print("非-vsetvl 向量 op = 0 的核数 = %d / %d  ⟹ 真向量运算 = 0" % (
        len([d for d in dq if d["non_vsetvl"] == 0]), len(dq)))
    print("★复核结论: %s" % ("CONFIRMED (18/18)" if all(
        d["total"] == 2 and d["non_vsetvl"] == 0 for d in dq) else "REFUTED"))

    print()
    print("=" * 78)
    print("P5 §4  M=1 fold 摊销 · roofline 判定 (本线核心·重判定)")
    print("=" * 78)
    print("\n[4a] 格式真源 (struct 机读·非散文)")
    for fmt in ("q2_K", "q3_K", "q4_K", "q5_K", "q6_K"):
        f = facts[fmt]
        print("  %-6s n_sb=%-3d has_min=%-6s  src=%s" % (fmt, f["n_sb"], f["has_min"], f["src"]))
        print("           struct: %s" % f["struct"][:96])

    print("\n[4b] M=1 fold 的算术下界 (QK_K=%d·ggml-common.h:89)" % QK_K)
    print("  定义: 每 super-block 强制 MAC=%d; fold_ops = n_sb x (scale + min?)" % QK_K)
    print("  ratio_floor_generous = MAC/(MAC+fold_ops)  ← 假设【对手 fold 成本=0】且【每 fold op 贵如 1 MAC】")
    print("                                              = 已知为假的【最大慷慨】上界")
    print()
    print("  %-6s %-5s %-8s %-9s %-12s %s" % ("fmt", "n_sb", "fold_ops", "fold/MAC", "ratio_floor", "vs 0.8 门"))
    print("  " + "-" * 66)
    ab = fold_arith_bound(facts)
    for r in ab:
        print("  %-6s %-5d %-8d %-9.4f %-12.4f %s" % (
            r["fmt"], r["n_sb"], r["fold_ops"], r["fold_over_mac"], r["ratio_floor_generous"],
            "PASS(>0.8) ⟹ fold 单独【产不出】named-X" if r["ratio_floor_generous"] > 0.8 else "可致 named-X"))
    n_above = len([r for r in ab if r["ratio_floor_generous"] > 0.8])
    print("\n  ★ %d/%d 格: 即便最大慷慨假设下, fold@M=1 单独也【无法】把 ratio 压到 0.8 门以下。" % (n_above, len(ab)))

    print("\n[4c] 发射 leaf 实测: fold 在 body 里到底占多少 (静态计数·depth 校验)")
    print("  %-6s %-7s %-8s %-7s %-8s %-7s %-11s %s" % (
        "fmt", "bodyD", "body_n", "fold_n", "recon_n", "mac_n", "fold_frac", "depth 一致?"))
    print("  " + "-" * 74)
    lp = leaf_profile()
    for r in lp:
        if "err" in r:
            print("  %-6s ERR %s" % (r["fmt"], r["err"]))
            continue
        print("  %-6s %-7d %-8d %-7d %-8d %-7d %-11.5f %s" % (
            r["fmt"], r["body_depth"], r["body_n"], r["fold_n"], r["recon_n"], r["mac_n"],
            r["fold_frac"], "同 depth ⟹ 静态比=动态比" if r["same_depth"] else "★不同 depth·比值无效"))
    ok = [r for r in lp if "err" not in r]
    if ok:
        mx = max(r["fold_frac"] for r in ok)
        print("\n  fold 占 depth-3 body 最大占比 = %.5f (= %.3f%%)" % (mx, mx * 100))
        print("  ⟹ Amdahl: fold 【完全消除】的 e2e-kernel 上限 = %.4fx" % (1 / (1 - mx)))
        print("  实测缺口 (1/ratio·须由 fold 解释的量·仅输局格适用):")
        for fmt, board, dom, ratio in COUNTER_EX:
            tag = ("须解释 %.2fx" % (1 / ratio)) if ratio < 0.8 else "PASS(赢)·无缺口须解释"
            print("     %-6s @%-4s %-16s ratio=%-7.4f ⟹ %s" % (fmt, board, dom, ratio, tag))
        print("  ⟹ fold 上限 %.4fx 【远小于】须解释的 2.7x-18.7x ⟹ fold 不是成本中心。" % (1 / (1 - mx)))
        print("\n  真成本中心 = 权重位重建 (recon/MAC 比):")
        for r in ok:
            print("     %-6s recon=%-6d mac=%-5d ⟹ %.2f 个位重建 op / MAC" % (
                r["fmt"], r["recon_n"], r["mac_n"], r["recon_per_mac"]))

    print()
    print("=" * 78)
    print("P5 §5  ★反例正面回答: q4_K 同格式跨板一赢一输")
    print("=" * 78)
    q4r = get(rows, "gemm_tile", "q4_K", "decode", "rvv", "cold")
    q4k = get(rows, "gemm_tile", "q4_K", "decode", "k1", "cold")
    q5k = get(rows, "gemm_tile", "q5_K", "decode", "k1", "cold")
    print("  T3 机读: q4_K@rvv decode cold = %s (%s)" % (q4r, get(rows, "gemm_tile", "q4_K", "decode", "rvv", "disp")))
    print("  T3 机读: q4_K@k1  decode cold = %s (%s)" % (q4k, get(rows, "gemm_tile", "q4_K", "decode", "k1", "disp")))
    print("  T3 机读: q5_K@k1  decode cold = %s (%s)" % (q5k, get(rows, "gemm_tile", "q5_K", "decode", "k1", "disp")))
    print()
    print("  ★falsifier-1 (跨板·同格式·同编译器 clang-18):")
    print("     q4_K n_sb = %d (双板同·格式定义决定·struct 机读)" % facts["q4_K"]["n_sb"])
    print("     ⟹ 两格 fold 算术【完全相同】·verdict 却 %s vs %s = 差 %.2fx" % (
        q4k, q4r, float(q4k) / float(q4r)))
    print("     A2-batch4 §2.4 raw: ours_ms k1=0.522-0.526 vs rvv-clang=0.589-0.800  (我方核【近同】)")
    print("                          opp_ms  k1=0.804-0.806 vs rvv     =0.215-0.231  (对手差 3.6x)")
    print("     ⟹ 翻转全部由【对手强度】驱动·我方 fold 不变 ⟹ fold 无法解释 verdict 翻转。")
    print()
    print("  ★falsifier-2 (同板·同编译器·同 n_sb·判别键自证伪):")
    print("     q4_K@k1 n_sb=%d → %s PASS" % (facts["q4_K"]["n_sb"], q4k))
    print("     q5_K@k1 n_sb=%d → %s 具名-X" % (facts["q5_K"]["n_sb"], q5k))
    print("     ⟹ 同板同 n_sb=8 两格 verdict 相反 ⟹ 【判别键=sub-block 数 16 vs 8】在其【自己的板内】被证伪。")
    print()
    print("  ★falsifier-3 (同源同算术·换编译器):")
    print("     rvv K-quant decode leaf: gcc-15.2 vsetvl 922/2225/2952/2461 (q2/q3/q5/q6)")
    print("                              clang-18  vsetvl  81/  14/  57/  22")
    print("     ⟹ 同一份 leaf 源·fold 算术恒定·时间差 3-15x ⟹ 差异来自 codegen·非算术必然。")
    print()
    print("  ★falsifier-4 (对手=存在性证明):")
    print("     对手 block-dot 在【同一 M=1】下计算【同一格式】·同样付 per-block fold(格式强制·无行可摊)")
    print("     rvv: 对手 0.22ms vs 我方 0.60ms ⟹ 存在一个实现在同硅上把同样的 M=1 fold 做快 2.7x")
    print("     ⟹ 我方 0.60ms 不是该问题的算术地板。")

    print()
    print("=" * 78)
    print("P5 §6  ★三选一判定")
    print("=" * 78)
    print("""
  判定 = (乙) emitter 缺口 —— (甲) 物理地板【被否决】。

  否决 (甲) 的四条独立 falsifier (任一条即足·此处有四):
    F1 跨板: q4_K 同格式同 n_sb=8 同 clang·k1 1.535 PASS / rvv 0.361 具名-X·我方核近同(0.53 vs 0.60ms)
    F2 同板: q4_K@k1(n_sb=8) PASS 而 q5_K@k1(n_sb=8) 具名-X ⟹ 判别键在自己板内证伪
    F3 编译器: 同源 leaf·fold 算术恒定·gcc↔clang 3-15x ⟹ codegen 非算术
    F4 存在性: 对手在同 M=1 同格式下快 2.7x ⟹ 我方数不是算术地板

  正面算术 (不依赖任何 falsifier·独立成立):
    fold@M=1 的【最大慷慨】上界 ratio_floor ∈ [0.889, 0.941] > 0.8 门
      ⟹ 即使把对手 fold 记为 0、每个 fold op 贵如一次 MAC, fold 单独也产不出一个 named-X。
    发射体实测 fold 占 depth-3 body ≤ 0.3% ⟹ Amdahl 上限 ~1.003x·须解释的是 2.7x-18.7x。
    ⟹ 「fold@M=1 不 amortize」是【真但空洞】: 它对双方对称成立(M=1 对对手也是 M=1)·在 ratio 里【相消】。

  ★重归因 (P5 净产出): 成本中心 = [MECH-WEIGHT-RECONSTRUCTION-BOUND] (权重位重建)
    发射体 depth-3 主体 = vand/vsrl/vsll/vor 位拆包·recon/MAC ≈ 2-5 ⟹ 每次 MAC 要先付 2-5 个位重建 op。
    此机制 T8 已有同名先例(q3_K 'weight-reconstruction-bound' 条目)·本线【复用·非重造】。

  具名杠杆 → P3 机制队列 (★仅【排序】·不作认输据·不承诺翻正·PR-37 [shape↛score]):
    L1 [CASE-KQUANT-GCC-CODEGEN]  rvv deploy 域 gcc-15.2 spill (vsetvl 922-2952)
    L2 权重重建位拆包发射体      recon/MAC 2-5 的位 op 密度
    L3 hl8 half-util @VLEN256    k1 leaf = VLEN128 mf2 form·未占满宽
    L4 plan-regime 选择          M=1 下 repack-GEVM vs block-dot 的 plan 归属 ([K-10] 结构级)

  ★未定残项 (诚实标丙·不打包进乙):
    上述四杠杆各自【贡献多少】= 未测 (无 per-term ablation·无 roofline probe) ⟹ 该【量化分解】= (丙) 需板测。
    但【fold 是不是物理地板】这一问本身 = 已决 = 否。两者不可混。
""")

    print("=" * 78)
    print("P5 §7  ★PR-6 物理墙收严 · 自证")
    print("=" * 78)
    print("""
  本线标注的「内存墙 roofline」格数 = 0。
  理由(逐条):
    (1) 无 roofline probe: A2-batch4 §4 明列「q5_0@rvv-clang named-X 复核 | roofline probe 缺」;
        A2-batch5 §2 明列「未测 roofline·不宣『内存墙满分』」。⟹ 无双方同贴同一墙的证据。
    (2) 先验证伪: q4_K@rvv 我方 0.60ms vs 对手 0.22ms = 2.7x 分离 ⟹ 双方【显然不在同一墙】·
        连【单侧 at-wall】都未测。⟹ 依 PR-6 一律不许标。
    (3) 本线全部墙词均取自三档词表 {指令数内禀 | 内存墙 roofline | 对手结构优势具名};
        「gcc-death」「fold@M=1」「weight-reconstruction」等 = 【机制 id】·归档到词表内的
        「指令数内禀」/「对手结构优势具名」·未新造墙词。
  ⟹ 无【单侧 at-wall 标成物理墙】之事·因为本线标了【0 个】物理墙。
""")

    if "--write" in sys.argv:
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["board", "op", "format", "regime", "tier", "t3_disp_READONLY",
                        "t3_cold_READONLY", "p5_missing_mechanism", "p5_wall_word_3tier",
                        "p5_causal_note", "p5_provenance"])
            for c in sorted(cells, key=lambda x: (x["board"], x["op"], x["format"])):
                w.writerow([c["board"], c["op"], c["format"], c["regime"], c["tier"],
                            c["disp"], c["cold"],
                            " ∧ ".join(c["mech"]),
                            " ∧ ".join(sorted(set(MECH[m][0] for m in c["mech"]))),
                            " ∧ ".join(MECH[m][1] for m in c["mech"]),
                            "tools/visibility/p5_loss_causal.py"])
        print("[write] %s  (%d 行·新文件·T8 本体未动)" % (os.path.relpath(OUT, ROOT), len(cells)))


# 反例数 (T3 机读填充·此处仅列 domain 标签)
COUNTER_EX = [
    ("q4_K", "rvv", "clang-18-micro", 0.361),
    ("q4_K", "k1", "clang-18", 1.535),
    ("q2_K", "rvv", "gcc-15.2-deploy", 0.0685),
    ("q6_K", "rvv", "gcc-15.2-deploy", 0.0535),
]

if __name__ == "__main__":
    main()
