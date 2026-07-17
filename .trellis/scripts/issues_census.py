#!/usr/bin/env python3
"""issues_census.py — 登记簿机算普查（§六「登记簿计数一律机算」·决策权限卡「机算断言连提取代码」）。

计数谓词（与 issues/index.md §编号 明文谓词一致）：
  按 `### ISSUE-NNN` 切块（扫 issues/*.md，排除 index.md）；每块取【最后一个】
  `- **状态**：` 行；状态 = 该行去 `**` 后的 leading token（首个 （/(/·/（ 或换行前）。
  leading-token 分组 = index 计数行既有约定（`阻塞（前置X）`→`阻塞`）。
  同时输出 RAW 尾串 distinct 枚举（禁隐性归并·透明）。

用法：python3 .trellis/scripts/issues_census.py
"""
import re, glob, os, sys
from collections import Counter, defaultdict

ISSUES_DIR = os.path.join(os.path.dirname(__file__), "..", "spec", "issues")
BLOCK_RE = re.compile(r'^### +ISSUE-(\d+)\b', re.M)
STATUS_RE = re.compile(r'^- \*\*状态\*\*[：:]\s*(.*)$', re.M)

def leading_token(s):
    s = s.replace('**', '').strip()
    # 截到首个分隔符（中文/英文括号、间隔号、逗号、句号、空格）
    m = re.split(r'[（(·，,。\s]', s, maxsplit=1)
    return m[0].strip() if m and m[0].strip() else s

def main():
    files = sorted(f for f in glob.glob(os.path.join(ISSUES_DIR, "*.md"))
                   if os.path.basename(f) != "index.md")
    by_num = {}       # num -> (file, raw_status_tail)
    dupes = defaultdict(list)
    for f in files:
        txt = open(f, encoding="utf-8").read()
        # 切块：用标题位置切
        marks = [(m.start(), m.group(1)) for m in BLOCK_RE.finditer(txt)]
        for i, (pos, num) in enumerate(marks):
            end = marks[i+1][0] if i+1 < len(marks) else len(txt)
            block = txt[pos:end]
            sts = STATUS_RE.findall(block)
            tail = sts[-1].strip() if sts else "(无状态行)"
            base = os.path.basename(f)
            if num in by_num:
                dupes[num].append(base)
            else:
                by_num[num] = (base, tail)
    nums = sorted(int(n) for n in by_num)
    total = len(by_num)
    lo, hi = (nums[0], nums[-1]) if nums else (0, 0)
    missing = [n for n in range(lo, hi+1) if n not in nums]
    lead = Counter(leading_token(by_num[f'{n:03d}' if f'{n:03d}' in by_num else str(n)][1])
                   if (f'{n:03d}' in by_num or str(n) in by_num) else '?'
                   for n in nums)
    # 上面 key 形态不定，重算干净：
    lead = Counter(leading_token(v[1]) for v in by_num.values())
    raw = Counter(v[1] for v in by_num.values())

    print(f"总条数 = {total}  范围 = ISSUE-{lo:03d}..ISSUE-{hi:03d}")
    print(f"缺号 = {missing if missing else '无'}  重号 = {dict(dupes) if dupes else '无'}")
    print("\n=== leading-token 分布（index 计数行口径）===")
    for k, c in lead.most_common():
        print(f"  {c:3d}  {k}")
    print("\n=== RAW 状态尾串 distinct（禁隐性归并·透明）===")
    for k, c in raw.most_common():
        print(f"  {c:3d}  {k[:70]}")
    # 缺 index 指针检查
    idx = open(os.path.join(ISSUES_DIR, "index.md"), encoding="utf-8").read()
    no_ptr = [f'{n:03d}' for n in nums if f'ISSUE-{n:03d}' not in idx and f'ISSUE-{n}' not in idx]
    print(f"\n=== 本体存在但 index.md 无指针 = {no_ptr if no_ptr else '无'}")

if __name__ == "__main__":
    main()
