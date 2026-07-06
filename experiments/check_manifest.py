#!/usr/bin/env python3
# experiments/check_manifest.py — MANIFEST 漂移校验(CI 门,fail-closed)
#
# 规则:experiments/ 的【耐久内容】(= git 已跟踪 + 未跟踪-未忽略,即一个 fresh clone
# 会拿到的全部文件)必须与 MANIFEST.md 的 REGISTRY 区【逐路径一一相等】。
#   - 树里有、REGISTRY 没登记的文件  → 漂移(未登记残留/新工件未入账) → 红。
#   - REGISTRY 登记了、树里缺失的文件 → 漂移(引用悬空/被误删)         → 红。
# 退出码非零 = 红。当前树自审应绿。
#
# 口径说明:
#   - 校验对象 = 耐久内容(git 视角),【不含】gitignore 的 on-device scratch
#     (根 .gitignore 的 `experiments/ondevice-*/`;ondevice-q8_0 是历史 force-add,已跟踪)。
#     scratch 内的被引用工件另在 MANIFEST 的 "REFERENCED-BUT-GITIGNORED" 区登记(纯文档、不入本校验),
#     因其易失、无法对易失 scratch 设 pass/fail。
#   - REGISTRY 解析:取 `<!-- REGISTRY:BEGIN -->`..`<!-- REGISTRY:END -->` 之间每行
#     第一个以 `experiments/` 开头的反引号 token 作为登记路径(referenced-by 用纯文本写、不加反引号)。

import os
import re
import subprocess
import sys

SELF_DIR = os.path.dirname(os.path.abspath(__file__))


def repo_root():
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=SELF_DIR, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def durable_set(root):
    # tracked + untracked-not-ignored under experiments/ (NUL-separated -> no path escaping)
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "experiments/"],
        cwd=root, capture_output=True, check=True,
    )
    paths = out.stdout.decode("utf-8").split("\0")
    return {p for p in paths if p}


def registered_set(manifest_path):
    with open(manifest_path, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"<!-- REGISTRY:BEGIN -->(.*?)<!-- REGISTRY:END -->", text, re.S)
    if not m:
        print("FAIL: REGISTRY:BEGIN/END markers not found in MANIFEST.md", file=sys.stderr)
        sys.exit(2)
    reg = set()
    for line in m.group(1).splitlines():
        for tok in re.findall(r"`([^`]+)`", line):
            if tok.startswith("experiments/"):
                reg.add(tok)
                break  # first experiments/ token per line = the registered path
    return reg


def main():
    root = repo_root()
    manifest = os.path.join(root, "experiments", "MANIFEST.md")
    if not os.path.exists(manifest):
        print("FAIL: experiments/MANIFEST.md missing", file=sys.stderr)
        return 2

    durable = durable_set(root)
    registered = registered_set(manifest)

    unregistered = sorted(durable - registered)
    missing = sorted(registered - durable)

    if not unregistered and not missing:
        print(f"OK: experiments/ durable content == MANIFEST REGISTRY ({len(durable)} files).")
        return 0

    if unregistered:
        print(f"DRIFT: {len(unregistered)} file(s) on tree but NOT in MANIFEST REGISTRY:")
        for p in unregistered:
            print(f"  + {p}")
    if missing:
        print(f"DRIFT: {len(missing)} file(s) in MANIFEST REGISTRY but MISSING from tree:")
        for p in missing:
            print(f"  - {p}")
    print("RED: experiments/ drifted from MANIFEST. Update MANIFEST.md REGISTRY or the tree.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
