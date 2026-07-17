#!/usr/bin/env python3
# tools/gates/gen_tools_index.py — TOOLS.md 的可达性机核（总令 §4.2.8 的尺子本体）。
#
# 总令 §4.2.8：`tools/` 只留真实在用；每个留下的在 tools/TOOLS.md 写一行
# {用途 + 被谁调用}，**写不出即 attic**。本脚本机算「被谁调用」那一列，使
# TOOLS.md 的可达性断言【可复跑复核】，而不是靠人读。
#
# 立此存照（governance/决策权限卡 §〇 来历）：
#   「『机算』的可信度 = 那段提取代码的【可复核性】，不是"用了机器"这件事。
#     即席代码是可复核性最低的形态。」
# ⟹ 本文件存在的理由 = 把这段扫描从 heredoc 里的一次性 python 变成入库、可复跑、可负控的代码。
#
# 用法：
#   python3 tools/gates/gen_tools_index.py            # 打印全量 {工具 → 引用者} 表
#   python3 tools/gates/gen_tools_index.py --zero      # 只打印【零引用】者（= attic 候选）
#   python3 tools/gates/gen_tools_index.py --check     # TOOLS.md 覆盖率自检：exit 1 = 有工具缺行
#
# ★三类【已知假阴性】（每条都有前科，勿删本注释）：
#   ① python import 不带扩展名：`import _manifest_common` ≠ 字面 `_manifest_common.py`
#      —— 本役实测：漏判 tools/gates/_manifest_common.py 为零引用，而它有 3 个真 importer。
#      ⟹ 对 .py 额外扫 `import <stem>` / `from <stem>`。
#   ② 花括号展开引用：`{board_stage1_compile_seal,board_stage1_selfcheck}.sh`
#      —— 本役实测：漏判 case-compiler-asymmetry/stage1-rvv-remeasure/ 三脚本为零引用，
#      而 experiments/active/result-tables/T-VALIDITY-STAGE1_rvv_symmetric_remeasure.md 真引用之。
#      ⟹ 对 .sh/.py 额外扫【去扩展名词干】。
#   ③ 散文提及 ≠ 真引用：`grep touch-set` 会匹到 "3-file touch-set"（trellis卫生 §七 明载）。
#      ⟹ 本脚本【不判】真引用 vs 散文，只产候选；判读由人做，结论写进 TOOLS.md。
#
# ★【假阳性】（与上三条方向相反 · 同样有前科）：词干匹配会把**同名标识符**当成引用 ——
#   本役实测：`tools/e2e-harness/g3-lode-flat-q41/build_and_run.sh` 的词干 `build_and_run`
#   匹到了 `scripts/rvv_accumulator_sweep_measure.py` 里的**函数名 `build_and_run_remote`**，
#   据此曾错写「该 py 是 g3-lode 的真调用方」——**真调用边根本不存在**。
#   ⟹ **`引用者=N` 是【候选数】，不是【调用者数】。** 词干命中必须回看上下文再下结论；
#      「X 被 Y 调用」这类断言**禁**直接引本脚本的计数交差。
#
# 本脚本【不】自动 attic 任何东西：零引用 = 候选，不是判决（§6.3「按需工具有正当性」）。

import os
import subprocess
import sys

REPO = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                      capture_output=True, text=True, check=True).stdout.strip()
TOOLS_MD = os.path.join(REPO, "tools", "TOOLS.md")


def tracked():
    """已跟踪 + 未跟踪未忽略（`--others --exclude-standard`）。
    ★用 `git ls-files` 单独收口是错的：它看不见【尚未 stage 的新文件】，
    会让新增工具在自己入库前隐形（本役实测：本脚本自己就查不到自己）。
    `--exclude-standard` 同时把 git-ignored 的 `_attic/` 天然排除。"""
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    return out.split("\n")


# ★收录范围 = 总令 §4.2.8 的原文两目录：「**`tools/`、`scripts/`**」。
# 本役实测：只扫 tools/ 会漏 scripts/ 8 件，其中 scripts/rvv_fair_three_way_measure.py
# 等正是 tools/e2e-harness/g3-lode-flat-q*/build_and_run.sh 的【调用方】——
# 漏扫 scripts/ 会把这 4 件 g3-lode 误判成"无调用者"。
SCOPE = ("tools/", "scripts/")


def tool_files():
    """全量工具 = SCOPE 下的 .py/.sh + 无扩展名可执行件（如 tools/bench/bench,
    tools/hooks/pre-commit）。★勿用 `find -name '*.py' -o -name '*.sh'` 单独收口：
    本役实测该式漏掉 tools/bench/bench（runner 本体）与 tools/hooks/pre-commit（现役执法者）。"""
    res = []
    for f in tracked():
        if not f or not f.startswith(SCOPE):
            continue
        # ★须落盘存在：`git ls-files --cached` 仍会列出【已从工作树删除但尚未 commit】者
        # （如刚 `mv` 进 _attic 的工具），不滤会把已归档件算进总数。
        if not os.path.exists(os.path.join(REPO, f)):
            continue
        if f.endswith((".py", ".sh")):
            res.append(f)
        elif "." not in os.path.basename(f) and os.access(os.path.join(REPO, f), os.X_OK):
            res.append(f)
    return sorted(res)


def read(p):
    try:
        with open(os.path.join(REPO, p), encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except (OSError, IsADirectoryError):
        return ""


# ★【登记簿不是调用者】——否则本尺自废。
# 实证（本役当场踩到）：TOOLS.md 建成后，因其逐条列出每个工具的路径，扫描把 TOOLS.md 自己
# 算成了每个工具的引用者 ⟹ `--zero` 从 12 变 0，「写不出被谁调用即 attic」这把尺子**永远再也
# 量不出东西**。同理 `_attic/ATTIC_INDEX.md`：归档索引记的是"它死了"，不是"它被调用"。
# ⟹ 索引/登记簿一律排除在【调用者集合】之外。
INDEX_NOT_CALLER = ("_attic/", "tools/TOOLS.md")


def scan():
    tools = tool_files()
    files = [f for f in tracked() if f and not f.startswith(INDEX_NOT_CALLER)]
    cache = {f: read(f) for f in files}
    result = {}
    for t in tools:
        base = os.path.basename(t)
        stem = os.path.splitext(base)[0]
        refs = []
        for f in files:
            if f == t:
                continue
            c = cache[f]
            if t in c:
                refs.append((f, "path"))
            elif base in c:
                refs.append((f, "base"))
            elif stem and stem in c:
                # 覆盖假阴性 ①（import 词干）与 ②（花括号展开）
                refs.append((f, "stem"))
        result[t] = refs
    return result


def main():
    argv = sys.argv[1:]
    res = scan()

    if "--check" in argv:
        md = read("tools/TOOLS.md")
        if not md:
            print("RED: tools/TOOLS.md 不存在")
            return 1
        # 覆盖判据 = 全路径出现，或其 **cell 目录**出现。
        # ★后一支的【已知弱点·诚实标注】：`tools/e2e-harness/` 的 206 件按 **cell 粒度**登记
        # （49 个战役 cell），非逐文件一行 —— 因其去留【整体待裁】于 ISSUE-090（见 TOOLS.md §七①），
        # 逐文件写行等于预设"留"。⟹ 本门对该目录只验【cell 覆盖】，**不验逐文件行**。
        # 裁决落地后（判「留」）应收紧为逐文件行，届时删掉 dirname 这一支。
        missing = []
        for t in res:
            if t in md or os.path.dirname(t) in md:
                continue
            missing.append(t)
        for t in missing:
            print(f"  ! 缺行: {t}")
        print(f"工具总数 = {len(res)} · TOOLS.md 缺行 = {len(missing)}")
        return 1 if missing else 0

    zero = {t: r for t, r in res.items() if not r}
    if "--zero" in argv:
        for t in sorted(zero):
            print(f"ZERO\t{t}")
        print(f"# 零引用 = {len(zero)} / 工具总数 = {len(res)}")
        return 0

    for t in sorted(res):
        refs = res[t]
        shown = ", ".join(f"{f}[{k}]" for f, k in refs[:4])
        print(f"{t}\t引用者={len(refs)}\t{shown}")
    print(f"# 工具总数 = {len(res)} · 零引用 = {len(zero)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
