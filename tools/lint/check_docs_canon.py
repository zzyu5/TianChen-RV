#!/usr/bin/env python3
# tools/lint/check_docs_canon.py — org STAGE2 Lint ② (CI, fail-closed).
#
# Two invariants over docs/ (durable lens = git-tracked + untracked-not-ignored):
#
#   (A) canon/ WHITELIST.  docs/canon/ holds ONLY the user-sovereign 总纲 (charter) docs
#       (实验总纲 / 执行总纲 / 科研目标总纲 / 定位). An agent-generated artifact (report /
#       template / dated snapshot / ledger) mixed into canon/ -> RED.  A canon file is recognised
#       by a charter marker in its basename ("总纲" / "定位" / "canon" / "charter").
#
#   (B) reports/ APPEND-ONLY.  docs/reports/ is an append-only stream: every file name must be
#       date-stamped (`YYYY-MM-DD-…`) so a report is never silently overwritten. The only
#       exception is a declared continuous append-only ledger (allowlist below), which grows
#       in place and therefore carries no date prefix.
#
# docs/method/ is intentionally unconstrained here (framework/method docs, machine-referenced).
#
#   check_docs_canon.py             # audit the tree
#   check_docs_canon.py --self-test # exercise the pure classifiers

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _manifest_common as mc  # noqa: E402

# RFC: 门过严误报修正 (2026-07-11 用户裁). "定位" added as a first-class charter marker so the
# authoritative one-page positioning charter (docs/canon/TianChen-RV_定位-v2.md, referenced by
# CLAUDE.md/ROADMAP) is recognised as a legitimate charter — it IS user-sovereign canon, the
# checker just lacked its marker (fail-closed误报, not a misplaced file). Rename-to-add-总纲 was
# rejected (breaks全仓 pointers); extending the marker set is the minimal-churn fix.
CANON_MARKER_RE = re.compile(r"(总纲|定位|canon|charter)", re.IGNORECASE)
DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-")

# Declared continuous append-only ledgers that grow in place (no per-snapshot date prefix).
# RFC: 门过严误报修正 (2026-07-11 用户裁). SEALED-WIN-REGISTRY.md is a declarative continuous
# append-only Win ledger (same KIND as travel-decision-ledger.md) — it grows in place and is
# never a silently-overwritten snapshot, so the date-prefix rule does not apply. Allowlisting it
# fixes a fail-closed误报, not a naming violation.
REPORTS_LEDGER_ALLOWLIST = {"travel-decision-ledger.md", "SEALED-WIN-REGISTRY.md"}


def classify_canon(basename):
    """docs/canon/ membership. Returns (ok, reason)."""
    if CANON_MARKER_RE.search(basename):
        return True, "canon 总纲/charter"
    return False, "non-charter doc in canon/ (agent-generated? -> docs/reports/ or docs/method/)"


def classify_report(basename):
    """docs/reports/ append-only naming. Returns (ok, reason)."""
    if DATE_PREFIX_RE.match(basename):
        return True, "date-stamped snapshot"
    if basename in REPORTS_LEDGER_ALLOWLIST:
        return True, "declared continuous append-only ledger"
    return False, "reports/ file lacks a YYYY-MM-DD- prefix (append-only violation)"


def self_test():
    ok_all = True
    print("-- (A) canon/ whitelist --")
    canon_cases = [
        ("TianChen-RV_实验总纲v1.md", True),
        ("TianChen-RV_执行总纲v2.md", True),
        ("TianChen-RV_科研目标总纲v2.md", True),
        ("TianChen-RV_定位-v2.md", True),     # 定位 charter (RFC 2026-07-11)
        ("project-charter.md", True),
        ("CANON-overview.md", True),
        ("2026-07-06-T3-report.md", False),   # dated agent report leaked in
        ("CADENCE-LAW.md", False),            # method doc, not charter
        ("travel-decision-ledger.md", False), # ledger, not charter
    ]
    for base, expect in canon_cases:
        got, reason = classify_canon(base)
        mark = "PASS" if got == expect else "FAIL"
        ok_all = ok_all and got == expect
        print(f"  [{mark}] {base} -> ok={got} (expect {expect}) : {reason}")

    print("-- (B) reports/ append-only --")
    report_cases = [
        ("2026-07-06-T3-kernel-micro-report-template.md", True),
        ("2026-07-06-并行线纪律-worktree-与触碰集.md", True),
        ("travel-decision-ledger.md", True),   # allowlisted continuous ledger
        ("SEALED-WIN-REGISTRY.md", True),      # allowlisted append-only Win ledger (RFC 2026-07-11)
        ("T6-e2e-report-template.md", False),  # undated -> would overwrite
        ("notes.md", False),
        ("2026-7-6-bad.md", False),            # not zero-padded ISO
    ]
    for base, expect in report_cases:
        got, reason = classify_report(base)
        mark = "PASS" if got == expect else "FAIL"
        ok_all = ok_all and got == expect
        print(f"  [{mark}] {base} -> ok={got} (expect {expect}) : {reason}")

    print("SELF-TEST:", "GREEN" if ok_all else "RED")
    return 0 if ok_all else 1


def main(argv):
    if "--self-test" in argv:
        return self_test()

    root = mc.repo_root()
    viol = []

    for rel in sorted(mc.durable_paths(root, "docs/canon/")):
        if os.path.dirname(rel) != "docs/canon":
            continue  # only direct members of canon/ are charter-gated
        ok, reason = classify_canon(os.path.basename(rel))
        if not ok:
            viol.append((rel, reason))

    for rel in sorted(mc.durable_paths(root, "docs/reports/")):
        if os.path.dirname(rel) != "docs/reports":
            continue
        ok, reason = classify_report(os.path.basename(rel))
        if not ok:
            viol.append((rel, reason))

    if not viol:
        print("OK: docs/canon/ is charter-only and docs/reports/ is append-only (date-stamped).")
        return 0

    print(f"RED: docs/ policy violated ({len(viol)} file(s)):")
    for rel, reason in viol:
        print(f"  ! {rel}  <-  {reason}")
    print("Fix: keep only 总纲/charter docs in docs/canon/ (move reports to docs/reports/ with a "
          "YYYY-MM-DD- prefix); method/framework docs go to docs/method/.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
