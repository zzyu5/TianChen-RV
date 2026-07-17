#!/usr/bin/env python3
"""AUDIT · gcc-lane 标签 vs 数据 (2026-07-17 · 纯只读 · 零改数据).

问题: PR-17 终裁「单世界 clang · gcc 视为不存在」, 验收 =「recon 输出零 gcc(grep=0)」.
本工具查: grep=0 验的是【标签字样】还是【数据来源】?

方法(机算·禁手写小计):
  对两块板 CSV 的全部 canonical 36-col 行:
    LABEL 域 = f[33] compiler_axis 声称的编译器域
    DATA  域 = f[35] verdict / f[29] cold_ratio / f[24] board_fp 里实际暴露的测量来源域
  再叠加 recon 的 CLANG_WORLD override(= PR-17 真重测的格) 判定:
    该格的 verdict 到底靠 gcc 数还是 clang 数.

污染格判据(与 PR-17 冲突):
  DATA 域 = gcc  ∧  该格【未】被 CLANG_WORLD 以 clang 重测数覆盖  ∧  该格在分母内
  → 该格 verdict 依赖 gcc 测量数据, 却在「单世界 clang」快照下入账.

输出纯事实, 不裁去向(改数据/重测 = 硬冻结 + 必问).
用法: python3 tools/visibility/audit_gcc_lane.py
"""
import re
import sys

ROOT = "/home/kingdom/phdworks/TianchenRV"
T3A = ROOT + "/experiments/active/result-tables/T3_A_board_A_rvv1.0_vlen128.csv"
T3B = ROOT + "/experiments/active/result-tables/T3_B_board_B_rvv1.0_vlen256.csv"
MASTER = ROOT + "/experiments/active/result-tables/T3_master_rebuild.csv"
RECON = ROOT + "/.trellis/scripts/recon_master_rebuild.py"

FWD_OPS = {"add", "cpy", "gelu", "mul", "rms_norm", "rope", "scale", "silu", "softmax"}

# ── recon CLANG_WORLD override 键 (recon_master_rebuild.py:229-253 · 逐条抄键·值不抄) ──
#    这 10 格 = PR-17 真在 clang 世界重测过 → 其 verdict 不再依赖 gcc 数.
CLANG_WORLD_KEYS = {
    ("gemm_tile", "iq4_xs", "prefill"), ("gemm_tile", "iq2_xxs", "prefill"),
    ("gemm_tile", "iq2_xs", "prefill"), ("gemm_tile", "iq2_s", "prefill"),
    ("gemm_tile", "mxfp4", "prefill"), ("gemm_tile", "tq1_0", "prefill"),
    ("gemm_tile", "tq2_0", "prefill"),
    ("dequantize_row", "q4_0", ""), ("dequantize_row", "q5_0", ""),
    ("dequantize_row", "q5_1", ""),
}

# gcc 数据来源指纹(出现在 verdict/cold/board_fp 里 = 该格的数来自 gcc 车道)
GCC_DATA_PAT = re.compile(
    r"gcc15\.2-lane|domain=gcc-15\.2-deploy-lane|gcc-15-deploy=MAIN|"
    r"gcc15\.2-deploy-MAIN|gcc15\.2-deploy-lane|gcc-deploy",
    re.I)
# gcc 仅作脚注/对照(非 MAIN 数据源)
GCC_FOOTNOTE_PAT = re.compile(r"gcc[- ]?15\s*=\s*deploy-footnote|gcc=deploy-footnote", re.I)
CLANG_DATA_PAT = re.compile(r"clang-?18|clang-micro|clang世界|clang18-sym", re.I)


def norm_key(rowkey):
    parts = rowkey.split("|")
    op0 = parts[0]
    if op0 == "forward":
        return parts[1], "f32"
    if op0 in FWD_OPS:
        return op0, "f32"
    if op0 == "gemm":
        return "gemm_tile", parts[1]
    if op0 == "dequant":
        return "dequantize_row", parts[1]
    if op0 == "quantize":
        return "quantize_row", parts[1]
    if op0 == "vec_dot":
        return "vec_dot", parts[1]
    return op0, (parts[1] if len(parts) > 1 else "")


def classify_label(f33):
    """f[33] compiler_axis 声称的域."""
    has_gcc = "gcc" in f33.lower()
    has_clang = bool(CLANG_DATA_PAT.search(f33))
    if not has_gcc and has_clang:
        return "CLANG-ONLY"          # 标签: 纯 clang 世界
    if has_gcc and GCC_FOOTNOTE_PAT.search(f33):
        return "CLANG-MAIN|gcc-脚注"  # 标签: 诚实双标(主 clang·gcc 仅部署脚注)
    if has_gcc and re.search(r"gcc[^|]*(MAIN|deploy-lane)", f33, re.I):
        return "GCC-MAIN(诚实)"       # 标签: 诚实自认 gcc 是主 verdict 源
    if has_gcc:
        return "GCC-其他"
    return "无编译器标注"


def verdict_token_region(f35):
    """f[35] 的 verdict TOKEN 区 = '*成色' 之前.
    '*成色:' 之后是成色注记(可含跨板脚注, 如 k1 行里引用 'rvv gcc-deploy6.224')
    → 注记【不是】本行数据来源, 不得当指纹(否则 k1 全线假阳性·已实测)."""
    return re.split(r"\*成色|\*★", f35)[0]


def classify_data(board, f29, f35, f33):
    """本行数字的实际来源域.

    只认【描述本行自身测量】的三处:
      (a) f[33] compiler_axis 的 MAIN 声明 (该列定义即"本行 verdict 的编译器轴")
      (b) f[35] verdict TOKEN 区 (成色注记区排除·见 verdict_token_region)
      (c) f[29] cold_ratio
    刻意【不】用 f[24] board_fp: 它是 provenance 指针, 两板共用同一串
    'T9-668f1d45(rvv-gcc-deploy)' → 会把 k1 行误判成 gcc (已实测假阳性 5 格).
    """
    tok_region = verdict_token_region(f35)
    hits = []
    # (a) 标签自认 gcc 是 MAIN / 本行走 gcc deploy-lane
    for m in re.finditer(r"gcc[0-9.]*-?(?:15)?[- ]?(?:deploy)?[-=]?(?:MAIN|deploy-lane)", f33, re.I):
        hits.append(m.group(0))
    # (b)(c) verdict token 区 / cold 里的 gcc 车道指纹
    for src in (tok_region, f29):
        for m in GCC_DATA_PAT.finditer(src):
            hits.append(m.group(0))
    if hits:
        return "gcc", sorted(set(hits))
    if CLANG_DATA_PAT.search(f33 + tok_region):
        return "clang", []
    return "无标注", []


def parse_board(path, board):
    rows = []
    for lineno, ln in enumerate(open(path), 1):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.rstrip("\n").split(",")
        if len(f) != 36:
            continue
        if f[0].split("|")[0] in ("measurement_row_key", "axis"):
            continue
        op, fmt = norm_key(f[0])
        label = classify_label(f[33])
        data, hits = classify_data(board, f[29], f[35], f[33])
        # verdict token(照 recon parse_t3 同法·f[35] in-denom* 后的 token)
        m = re.search(r"in-denom[*;]([A-Za-z0-9._-]+)", f[35])
        tok = m.group(1) if m else ""
        in_denom = "test-only-not-in-denom" not in f[35]
        rows.append(dict(board=board, lineno=lineno, rowkey=f[0], op=op, fmt=fmt,
                         label=label, data=data, hits=hits, tok=tok,
                         in_denom=in_denom, f24=f[24], f29=f[29],
                         f33=f[33], f35=f[35]))
    return rows


def overridden(op, fmt):
    """该 (op,fmt) 是否被 CLANG_WORLD 以 clang 重测数覆盖(任一 regime)."""
    return any(k[0] == op and k[1] == fmt for k in CLANG_WORLD_KEYS)


def master_state(op, fmt):
    """master CSV 里【真正消费该板行】的格 · rvv verdict/cold/note(机读·非手抄).

    ★必须复刻 recon 自己的 skip_t3 门(recon_master_rebuild.py:309 · 已 sed 确认):
        skip_t3 = (eng=="ime") or (op=="gemm_tile" and regime=="decode")
    这两类格【不】读板 CSV 的 t3 行(ime 走 IME_KERNELSYM · gemm decode 走 GEMM_DECODE),
    所以它们不因该板行的 gcc 数而受污染. 不复刻此门 = 把 ime/decode 误算进污染 (前科 8:
    数 generator 的 INTENT 而非真消费).
    master 是头条与报告的直接上游 → 它的 note 才是对外的【标签】.
    """
    out = []
    for ln in open(MASTER):
        f = ln.rstrip("\n").split(",")
        if len(f) < 15 or f[0] == "op":
            continue
        if f[0] == op and f[1] == fmt:
            eng, regime = f[2], f[3]
            if eng == "ime" or (op == "gemm_tile" and regime == "decode"):
                continue   # 不消费板行 → 非本审计范围
            note = f[9]
            if "单世界clang" in note or "clang世界" in note:
                claim = "★声称单世界clang"
            elif "gcc" in note.lower():
                claim = "提及gcc"
            else:
                claim = "对编译器沉默"
            out.append(dict(regime=f[3], rvv_tier=f[5], rvv_disp=f[6],
                            rvv_cold=f[7], note=note, claim=claim))
    return out


def main():
    board_rows = parse_board(T3A, "rvv") + parse_board(T3B, "k1")

    print("=" * 100)
    print("AUDIT · gcc-lane 标签 vs 数据 · PR-17「单世界 clang」核查")
    print("源: T3_A(rvv) / T3_B(k1) 板 CSV · recon CLANG_WORLD override · T3_master_rebuild.csv")
    print("=" * 100)

    # ── 表 1: 全表 标签域 × 数据域 交叉计数 ──
    print("\n[表1] 全部 canonical 行 · LABEL域(f33) × DATA域(f33-MAIN/f35-token/f29) 交叉表")
    print("-" * 100)
    cross = {}
    for r in board_rows:
        cross.setdefault((r["board"], r["label"], r["data"]), []).append(r)
    print("%-6s %-22s %-10s %6s" % ("board", "LABEL域(f33)", "DATA域", "行数"))
    for k in sorted(cross):
        print("%-6s %-22s %-10s %6d" % (k[0], k[1], k[2], len(cross[k])))
    print("  Σ canonical 行 = %d (rvv %d + k1 %d)" % (
        len(board_rows),
        sum(1 for r in board_rows if r["board"] == "rvv"),
        sum(1 for r in board_rows if r["board"] == "k1")))

    # ── 表 2: 逐格 · 凡 DATA 域 = gcc ──
    gcc_rows = [r for r in board_rows if r["data"] == "gcc"]
    print("\n[表2] 逐格 · DATA域 = gcc 的全部行 (%d 行)" % len(gcc_rows))
    print("-" * 100)
    print("%-5s %-26s %-15s %-20s %-6s %-11s %s" % (
        "board", "op|format", "组", "LABEL域(f33声称)", "DATA域", "CLANG_WORLD", "判定"))
    print("-" * 100)
    contaminated = []
    for r in sorted(gcc_rows, key=lambda x: (x["board"], grp(x["op"]), x["op"], x["fmt"])):
        ovr = overridden(r["op"], r["fmt"])
        # 污染 = 数据是 gcc ∧ 未被 clang 重测覆盖 ∧ 在分母内
        is_contam = (not ovr) and r["in_denom"]
        # 矛盾 = 标签说纯 clang 世界, 数据却是 gcc
        contradiction = r["label"] == "CLANG-ONLY"
        mark = "★标签矛盾" if (contradiction and is_contam) else ("○标签诚实" if is_contam else "—已 clang 重测")
        print("%-5s %-26s %-15s %-20s %-6s %-11s %s" % (
            r["board"], r["op"] + "|" + r["fmt"], grp(r["op"]),
            r["label"], r["data"], "覆盖" if ovr else "未覆盖", mark))
        if is_contam:
            r["contradiction"] = contradiction
            contaminated.append(r)

    # ── 表 3: 污染格清单 + 证据行 + master 状态 ──
    print("\n[表3] ★污染格清单 = DATA=gcc ∧ 未被 CLANG_WORLD 重测 ∧ 在分母内 → 共 %d 格"
          % len(contaminated))
    print("-" * 100)
    for r in sorted(contaminated, key=lambda x: (grp(x["op"]), x["op"], x["fmt"])):
        ms = master_state(r["op"], r["fmt"])
        print("★ %s|%s  [%s]  %s:%d" % (
            r["op"], r["fmt"], grp(r["op"]),
            "T3_A" if r["board"] == "rvv" else "T3_B", r["lineno"]))
        print("    标签 f33 : %s" % r["f33"][:96])
        print("    数据指纹 : %s" % ", ".join(r["hits"]))
        print("    verdict  : %s" % r["f35"][:96])
        print("    板标签矛盾: %s" % ("★是(f33 称 clang18-sym·数据是 gcc)" if r["contradiction"]
                                    else "否(f33 诚实自认 gcc·矛盾在 master 层引入)"))
        for m in ms:
            print("    master   : regime=%-8s rvv_disp=%-26s rvv_cold=%-8s → note %s" % (
                m["regime"] or "-", m["rvv_disp"][:26], m["rvv_cold"] or "(空)", m["claim"]))
        print()

    # ── 表 3b: master 层(=头条上游)的 标签 vs 数据 ──
    print("[表3b] ★master 层 · 对外【标签】 vs 数字【来源】 (master note = 报告直接上游)")
    print("-" * 100)
    print("%-32s %-9s %-20s %-8s %s" % ("op|format@regime", "板f33标签", "master note 标签", "数据来源", "verdict"))
    print("-" * 100)
    mcross = {}
    for r in sorted(contaminated, key=lambda x: (grp(x["op"]), x["op"], x["fmt"])):
        for m in master_state(r["op"], r["fmt"]):
            if m["rvv_disp"] in ("N/A-hw",):
                continue
            nm = r["op"] + "|" + r["fmt"] + (("@" + m["regime"]) if m["regime"] else "")
            blab = "clang18-sym" if r["contradiction"] else "gcc(诚实)"
            print("%-32s %-9s %-20s %-8s %s" % (
                nm[:32], blab, m["claim"], "gcc", m["rvv_disp"][:24]))
            mcross[m["claim"]] = mcross.get(m["claim"], 0) + 1
    print("\n  master 层小计(机算): " + " | ".join(
        "%s=%d" % (k, v) for k, v in sorted(mcross.items())))

    # ── 表 4: 头条影响 · 污染格按 verdict 状态分组 ──
    #
    # ★零预处理机算 (2026-07-17 用户裁 · measurement §3.6 值域登记铁律):
    #     「值域登记只许机算枚举, 禁任何预处理归并; 归并即判据, 判据即裁.」
    # 本行原为 `st = m["rvv_disp"].split("(")[0]` —— 那句剥括注, 把 `PASS(decode-M1-GEVM)`
    # 无声并进 `PASS`, 于是这张【自称"从 master 机读"】的表其实报的是归并后的产物.
    # 用户裁: 括注是【判定谱系】(decode-M1 等), 不是噪音 ⟹ 原样取值, 一个字都不剥.
    # 合法值域 = measurement §3.3.1.1 的 12 值 (∪ VOID 四值); 本工具只读不判.
    print("\n[表4] 头条影响 · 污染格现计入的状态 (从 master 机读·零预处理·禁预测重测结果)")
    print("-" * 100)
    bucket = {}
    for r in contaminated:
        for m in master_state(r["op"], r["fmt"]):
            if r["board"] == "rvv":
                st = m["rvv_disp"]  # 原样 · 不 split · 不剥括注 · 不归并
                bucket.setdefault((grp(r["op"]), st), []).append(
                    r["op"] + "|" + r["fmt"] + (("@" + m["regime"]) if m["regime"] else ""))
    for k in sorted(bucket):
        items = sorted(set(bucket[k]))
        print("  %-14s %-26s %2d 格: %s" % (k[0], k[1], len(items), ", ".join(items)))
    print("  ※ 状态列 = rvv_disp 原样值(含括注). 括注 = 判定谱系, 不是噪音 —— 禁剥.")

    # ── 表 4b: 按【四档】归口 = 头条口径 ──
    #
    # ★本表的三桶塌缩(PASS / 具名-X / pending)是【本工具做的一次归并】= 判据, 不是实况:
    # 它把 §3.3.1.1 的 12 个 raw 值按 startswith 压成 3 个, 只为对上 recon 头条的 tier×PASS 口径.
    # 按 measurement §3.6「归并即判据」—— 故此处【明标为投影】, 实况值域一律以上面的表4(零预处理)为准.
    # 该归并是否该留 = 用户裁的面, 本工具不自行扩大也不自行取消.
    print("\n[表4b] ★按四档归口(= recon 头条口径 tier×PASS/全档量) · rvv 板")
    print("  ※ 【投影·非实况】下面的 PASS/具名-X/pending 三桶 = 本工具按 startswith 归并出来的,")
    print("     只为对上 recon 头条口径. 实况值域(12 raw 值·含括注谱系)见上面的[表4]. 归并即判据.")
    print("-" * 100)
    tb = {}
    for r in contaminated:
        if r["board"] != "rvv":
            continue
        for m in master_state(r["op"], r["fmt"]):
            st = "PASS" if m["rvv_disp"].startswith("PASS") else (
                "具名-X" if m["rvv_disp"].startswith("具名-X") else "pending")
            tb.setdefault((m["rvv_tier"], st), []).append(
                r["op"] + "|" + r["fmt"] + (("@" + m["regime"]) if m["regime"] else ""))
    print("%-10s %-8s %5s  %s" % ("四档", "状态", "格数", "格"))
    for k in sorted(tb):
        items = sorted(set(tb[k]))
        print("%-10s %-8s %5d  %s" % (k[0], k[1], len(items), ", ".join(items)[:64]))
    print("\n  ※ 参照 recon 现头条(本次运行·rvv): 标量类 36/51 · 通用向量 22/27 · 手调 8/24 · Σ=102")
    print("  ※ 若按单世界 clang 重测 → 结果【未测】. 本工具不预测翻转方向, 亦不预测头条增减(PR-37).")

    # ── 表 5: grep 门空心性测试 ──
    print("\n[表5] ★PR-17 验收「recon 输出零 gcc·grep=0」空心性测试")
    print("-" * 100)
    hollow_test()


def grp(op):
    if op in ("gemm_tile", "vec_dot"):
        return "matmul"
    if op in FWD_OPS:
        return "forward"
    if op == "dequantize_row":
        return "dequant"
    if op == "quantize_row":
        return "quantize"
    if op == "product_reduce":
        return "product_reduce"
    return "other"


MUT_DIR = "/tmp/claude-1006/-home-kingdom-phdworks-TianchenRV/d3e66624-0ee9-47b9-a28f-6e19468d8e17/scratchpad/hollow"


def hollow_test():
    """★证伪实验: grep 门到底观测【字样】还是【数据来源】?

    实验设计(变异测试·全在 scratchpad 副本上跑·真仓库零改动):
      对照组 A = 现状 recon 输出                       → 记 grep 读数 + master 指纹
      实验组 B = 把两块板 CSV 【每一行】的 compiler_axis(f33) 改写成尖叫式 gcc,
                 并在 verdict(f35) 末尾追加 gcc 溯源标记; 数字/verdict token 一字不改.
                 = 构造一个「100% 数据自认 gcc」的世界.
      判据:
        若 grep 门验的是【数据来源】 → B 组 grep 必 >0 (门应炸).
        若 grep 门验的只是【字样】   → B 组 grep 仍 =0, 且 master 与 A 组【逐字节相同】.
                                       ⇒ 门对编译器来源恒盲 = 恒真 = 空心.
    """
    import subprocess
    import hashlib
    import os
    import shutil

    # ── 对照组 A: 现状 ──
    print("  [对照组 A · 现状]")
    outs = {
        "T3_master_rebuild.csv": ROOT + "/experiments/active/result-tables/T3_master_rebuild.csv",
        "T3_master_rowclue.txt": ROOT + "/experiments/active/result-tables/T3_master_rowclue.txt",
    }
    for name, p in outs.items():
        n = subprocess.run(["grep", "-ci", "gcc", p], capture_output=True, text=True).stdout.strip()
        print("    grep -ci gcc %-24s = %s" % (name, n))
    rA = subprocess.run([sys.executable, RECON], capture_output=True, text=True)
    nA = sum(1 for ln in rA.stdout.splitlines() if "gcc" in ln.lower())
    print("    grep -ci gcc %-24s = %d" % ("recon stdout", nA))
    hA = hashlib.sha256(open(outs["T3_master_rebuild.csv"], "rb").read()).hexdigest()[:16]
    print("    master sha256[:16] = %s" % hA)

    # ── 实验组 B: 100% gcc 自认的输入 ──
    os.makedirs(MUT_DIR, exist_ok=True)
    mutated = 0
    for src, dst in ((T3A, MUT_DIR + "/T3_A.csv"), (T3B, MUT_DIR + "/T3_B.csv")):
        lines = []
        for ln in open(src):
            f = ln.rstrip("\n").split(",")
            if (not ln.startswith("#") and ln.strip() and len(f) == 36
                    and f[0].split("|")[0] not in ("measurement_row_key", "axis")):
                # 只改【溯源字样】· 不动 f[29] 数字 · 不动 f[35] 的 in-denom* token
                f[33] = "gcc-15.2-EVERY-NUMBER-IN-THIS-ROW-IS-GCC-MEASURED"
                f[35] = f[35] + "·PROVENANCE=gcc-15.2-lane-ALL"
                lines.append(",".join(f) + "\n")
                mutated += 1
            else:
                lines.append(ln)
        open(dst, "w").write("".join(lines))
    # 打补丁的 recon 副本: 只改 4 个路径常量 → 绝不写真仓库
    src = open(RECON).read()
    src = src.replace('T3A = ROOT + "/experiments/active/result-tables/T3_A_board_A_rvv1.0_vlen128.csv"',
                      'T3A = "%s/T3_A.csv"' % MUT_DIR)
    src = src.replace('T3B = ROOT + "/experiments/active/result-tables/T3_B_board_B_rvv1.0_vlen256.csv"',
                      'T3B = "%s/T3_B.csv"' % MUT_DIR)
    src = src.replace('OUT = ROOT + "/experiments/active/result-tables/T3_master_rebuild.csv"',
                      'OUT = "%s/master_MUT.csv"' % MUT_DIR)
    src = src.replace('CLUE = ROOT + "/experiments/active/result-tables/T3_master_rowclue.txt"',
                      'CLUE = "%s/rowclue_MUT.txt"' % MUT_DIR)
    open(MUT_DIR + "/recon_MUT.py", "w").write(src)

    print("\n  [实验组 B · 把两板全部 %d 行的溯源字样改成尖叫式 gcc(数字与 verdict token 不动)]" % mutated)
    rB = subprocess.run([sys.executable, MUT_DIR + "/recon_MUT.py"], capture_output=True, text=True)
    if rB.returncode != 0:
        print("    ✗ 变异 recon 跑失败, 测试作废:\n" + rB.stderr[:400])
        return
    for name, p in (("master_MUT.csv", MUT_DIR + "/master_MUT.csv"),
                    ("rowclue_MUT.txt", MUT_DIR + "/rowclue_MUT.txt")):
        n = subprocess.run(["grep", "-ci", "gcc", p], capture_output=True, text=True).stdout.strip()
        print("    grep -ci gcc %-24s = %s" % (name, n))
    nB = sum(1 for ln in rB.stdout.splitlines() if "gcc" in ln.lower())
    print("    grep -ci gcc %-24s = %d" % ("recon stdout", nB))
    hB = hashlib.sha256(open(MUT_DIR + "/master_MUT.csv", "rb").read()).hexdigest()[:16]
    print("    master sha256[:16] = %s" % hB)

    # ── 判读 ──
    print("\n  [判读]")
    print("    输入侧: 100%% 的行自认 gcc  → 门读数 B = %s (A = %s)" % (nB, nA))
    print("    master 逐字节: A %s  vs  B %s  → %s" % (
        hA, hB, "★完全相同" if hA == hB else "不同"))
    if nB == 0 and hA == hB:
        print("\n    ⇒ 判定: ★门是【空心】的.")
        print("      即便输入 100% 自认 gcc, 门仍读 0, 且账面输出【一字节不变】.")
        print("      ∴ 门的读数与数据的编译器来源【无任何函数依赖】= 恒真检查.")
        print("      机制: recon 的 note 全部来自脚本内【硬编码字面量】(:286-303/:340-360),")
        print("      从板 CSV 只取 f[29] 数字 与 f[35] 的 verdict TOKEN 分类 —— 文本不透传,")
        print("      所以 gcc 字样【结构上不可能】出现在输出里. grep=0 是恒等式, 不是证据.")
    else:
        print("\n    ⇒ 判定: 门对来源有敏感性(与预期相反·需复核本测试).")
    print("\n    ※ 本测试全程在 %s 的副本上进行; 真仓库 CSV/recon 零改动(git status 可验)." % MUT_DIR)


if __name__ == "__main__":
    main()
