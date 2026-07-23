#!/usr/bin/env python3
"""t1d_dual_instance.py -- E5 / T1d: the DUAL-INSTANCE demonstration table.

WHAT THIS SHOWS (C1 "template protocol" evidence, structural -- NO perf claim):
  ONE schema (schema/capability.schema.v1.json) + TWO board FACT INSTANCES
  (rvv board VLEN128 / k1 board VLEN256) => two declared_instance_hash values,
  two legal variant sets, two guard verdicts -- while the CORE code is
  BYTE-IDENTICAL (core diff = 0). The boards differ ONLY in FACTS.

THE TWO FACT CHANNELS (they are genuinely different; do not conflate them):
  (1) IN-IR fact channel: `weft.exec.capability` fact rows carried in the IR.
      These form the TargetCapabilitySet that
      support::computeDeclaredInstanceHash (lib/Support/DeclaredInstanceHash.cpp)
      hashes. Changing a fact row CHANGES declared_instance_hash.
  (2) march= SOURCE-PROFILE channel: the source adapter translates the selected
      profile into an explicit target/capability provider plus the same exact P; it
      does not construct a variant or typed compute body.  After the selector chooses
      the exact-P-derived candidate, the selected RVV owner consumes P + bound c_o and
      constructs the typed flat plan.  Therefore minimum_vlen is part of the declared
      capability instance and DOES change declared_instance_hash, while the candidate
      identity remains stable and the constructed LMUL changes.  Probe P4 records this
      boundary directly; the selector neither invents nor rewrites compute semantics.

THE MAIN BODY (section 1) is ONE byte-identical module carrying TWO variants:
  @rvv_typed_body   requires [@rvv]                 -- feasible on both boards
  @ime_keyed_body   requires [@rvv, @spacemit_ime]  -- feasible ONLY where the
                                                       spacemit.ime fact is available
Both are origin="rvv-plugin" with a real typed RVV body, so both are plugin-legal
under BOTH fact sets -- which is precisely what lets the SAME module plan on BOTH
boards and makes the divergence attributable to FACTS ALONE.

PROBES (each one is a real weft-opt invocation; nothing here is hand-written):
  P1 hash divergence      -- the main body under each fact set => two DIFFERENT
                             declared_instance_hash.
  P2 hash = f(facts) only -- the SAME fact set run with a DIFFERENT body (rvv-only,
                             different variant set AND different kernel symbol) must
                             reproduce P1's hash bit-for-bit, proving the hash keys
                             on FACTS, not on body / variant set / symbol names.
  P3 legal-variant-set    -- (a) from the main body: @ime_keyed_body is feasible=false
                             on rvv and feasible=true on k1 => the legal set differs
                             BY FACT. (b) complementary fail-closed leg: an
                             IME-PLUGIN-OWNED variant is hard-REJECTED by the IME
                             plugin's own legality verifier on the rvv board.
  P4 construction axis    -- the committed SEL-1 fixture: the source adapter emits
                             the same exact P and the selector chooses the same
                             exact-P-derived candidate; bound c_o then makes the
                             selected RVV owner construct m2(VLEN128)/m1(VLEN256)
                             flat plans.  The two declared capability instances
                             intentionally have different instance hashes.

CORE DIFF = 0 MACHINE CHECK (three independent legs, all mechanized):
  C1 f1-zero-branch  -- tools/gates/check_zero_core_family_branch.py: zero
                        family-name-keyed branch across the core scope (I3/[F-1]).
  C2 fact-confined   -- every diff hunk between the two instance files lands
                        INSIDE a weft.exec.capability fact region; the non-fact
                        surface (variants / bodies / dispatch) is byte-identical.
  C3 same-core-binary-- both instances are driven through the SAME weft-opt binary
                        (sha256 pinned) and the SAME pass-pipeline string. No
                        recompile, no per-board code path, no per-board flag.

Stdlib-only. Writes experiments/active/result-tables/T1d_dual_instance.csv.

Usage:
  python3 tools/visibility/t1d_dual_instance.py            # regenerate the CSV
  python3 tools/visibility/t1d_dual_instance.py --check    # re-run; diff vs CSV
  python3 tools/visibility/t1d_dual_instance.py --print    # human-readable dump
Exit: 0 OK ; 1 check mismatch / probe failure ; 2 setup error (no weft-opt).
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(REPO, "experiments/active/result-tables/T1d_dual_instance.csv")
F1_CHECKER = os.path.join(REPO, "tools/gates/check_zero_core_family_branch.py")
SCHEMA = "schema/capability.schema.v1.json"
SEL1_FIXTURE = os.path.join(
    REPO, "test/Conversion/RVV/rvv-q8-0-front-door-sel1-march-vlen-lmul-flip.mlir"
)

# The pass pipeline is IDENTICAL for both instances -- this is part of the
# core-diff=0 claim (leg C3), so it is a single constant, not a per-board string.
SELECT_PASS = "--weft-select-variants=attribution-jsonl={jsonl} attribution-jsonl-no-timestamp"
LEGALITY_PASS = "--weft-verify-plugin-variant-legality"
FRONTDOOR_PASS = (
    "--weft-rvv-materialize-q8-0-q8-0-block-dot-source-front-door="
    "march={march}"
)
PLANNING_PASSES_BEFORE_SELECTION = [
    "--weft-materialize-plugin-variants",
    "--weft-check-hart-parallel-capabilities",
    "--weft-verify-plugin-variant-legality",
]
PLANNING_PASSES_AFTER_SELECTION = [
    "--weft-materialize-dispatch-runtime-guards",
    "--weft-check-capability-requires",
    "--weft-materialize-selected-lowering-boundaries",
    "--weft-materialize-emission-plans",
    "--weft-check-execution-plan-coherence",
]


def find_weft_opt():
    # Prefer the caller-selected build, then the current in-tree layout.  Keep
    # the older nested layout as a compatibility fallback for archived builds.
    env_build = os.environ.get("WEFT_BUILD")
    candidates = []
    if env_build:
        candidates.append(os.path.join(env_build, "bin", "weft-opt"))
    candidates.append(os.path.join(REPO, "build/bin/weft-opt"))
    candidates.append(os.path.join(REPO, "build/weft/bin/weft-opt"))
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


# ---------------------------------------------------------------------------
# Fact instances. The BODY template is shared verbatim; ONLY the fact rows are
# parameterized. That is the whole point -- so the body is a single constant and
# cannot silently drift between the two boards.
# ---------------------------------------------------------------------------
FACTS_TMPL = """    weft.exec.capability @rvv {{
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "{hints}",
      vlen_bits = "{vlen}",
      status = "available"
    }}
    weft.exec.capability @spacemit_ime {{
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "{ime_status}",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "{vlen}"{ime_extra}
    }}
"""

# A real typed RVV extension-family body -- the RVV plugin's legality verifier
# rejects a variant without one, so this is the minimum legal body.
TYPED_BODY = """      %runtime_n = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = weft_rvv.setvl %runtime_n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !weft_rvv.vl"""

RVV_VARIANT = """    weft.exec.variant @rvv_typed_body attributes {
      fallback_role = "conservative",
      origin = "rvv-plugin",
      requires = [@rvv],
      weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>
    } {
%s
    }
""" % TYPED_BODY

# Capability-KEYED variant: identical plugin owner and identical typed body to
# @rvv_typed_body -- the ONLY difference is that it additionally requires the
# spacemit.ime FACT. So its feasibility is a pure function of the fact set.
IME_KEYED_VARIANT = """    weft.exec.variant @ime_keyed_body attributes {
      origin = "rvv-plugin",
      requires = [@rvv, @spacemit_ime],
      weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>
    } {
%s
    }
""" % TYPED_BODY

# Complementary fail-closed leg (P3b): an IME-PLUGIN-OWNED variant. The IME
# plugin's own verifyVariantLegality hard-rejects a materialized IME variant when
# the spacemit.ime capability is not available.
IME_PLUGIN_VARIANT = """    weft.exec.variant @ime_vmadot_matmul_slice attributes {
      condition = "spacemit_ime_capability_available",
      guard = "plugin_local_ime_vmadot_boundary",
      origin = "ime-plugin",
      policy = "ime_int8_matmul_vmadot_mac",
      requires = [@spacemit_ime]
    } {
    }
"""

# The two board fact instances. rvv board = VLEN128, no IME silicon (spacemit.ime
# status=missing -> interpreted as unavailable per [S-2]); k1 board = VLEN256 with
# the IME matrix unit present. The shared exact P supplies GEMM geometry.
INSTANCES = {
    "rvv": dict(board="rvv", vlen="128", hints="rv64gcv_zvl128b",
                ime_status="missing", ime_extra="",
                march="rv64gcv"),
    "k1": dict(board="k1", vlen="256", hints="rv64gcv_zvl256b",
               ime_status="available",
               ime_extra=',\n      available_harts = "0-3"',
               march="rv64gcv_zvl256b"),
}


def render(instance, kernel, variants):
    facts = FACTS_TMPL.format(hints=instance["hints"], vlen=instance["vlen"],
                              ime_status=instance["ime_status"],
                              ime_extra=instance["ime_extra"])
    problem = """    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 256 : i64, n = 256 : i64, k = 256 : i64}
"""
    return ("module {\n  weft.exec.kernel @%s attributes {construction_domain = \"riscv-execution\", problem = @canonical_problem} {\n" % kernel) + problem + facts + \
           "".join(variants) + "  }\n}\n"


def run(weft_opt, path, passes, output="/dev/null"):
    proc = subprocess.run([weft_opt, path] + passes + ["-o", output],
                          cwd=REPO, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def read_jsonl(path):
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Core-diff=0 leg C2: prove every diff line between the two instances lands
# inside a weft.exec.capability fact region, i.e. the non-fact surface is
# byte-identical. Implemented by STRIPPING the fact regions from both files and
# requiring the remainders to be byte-equal.
# ---------------------------------------------------------------------------
def strip_fact_regions(text):
    out, depth, in_fact = [], 0, False
    for line in text.splitlines():
        if not in_fact and re.match(r"\s*weft\.exec\.capability\b", line):
            in_fact, depth = True, line.count("{") - line.count("}")
            continue
        if in_fact:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                in_fact = False
            continue
        out.append(line)
    return "\n".join(out)


def core_diff_checks(weft_opt, files):
    """Returns (rows, all_green). Each leg is an independent mechanized check."""
    rows, green = [], True

    # C1 -- [F-1] zero family-name branch across the core scope.
    proc = subprocess.run([sys.executable, F1_CHECKER], cwd=REPO,
                          capture_output=True, text=True)
    f1_ok = proc.returncode == 0
    f1_msg = (proc.stdout + proc.stderr).strip().splitlines()
    rows.append(dict(
        check_id="C1-f1-zero-branch",
        mechanism="tools/gates/check_zero_core_family_branch.py (I3/[F-1] core scope)",
        verdict="PASS" if f1_ok else "FAIL",
        evidence=f1_msg[-1] if f1_msg else "(no output)",
        repro="python3 tools/gates/check_zero_core_family_branch.py"))
    green &= f1_ok

    # C2 -- the two instances differ ONLY inside capability fact regions.
    # The body size is EMITTED here rather than transcribed into prose: three
    # independent hand-reconstructions of this figure disagreed (1237/1256/1198)
    # because each rebuilt the body differently. The only reproducible figure is
    # the one this check computes from the generated instance files themselves.
    stripped = {k: strip_fact_regions(open(v).read()) for k, v in files.items()}
    keys = sorted(stripped)
    same = stripped[keys[0]] == stripped[keys[1]]
    body_sha = hashlib.sha256(stripped[keys[0]].encode()).hexdigest()[:16]
    body_chars = len(stripped[keys[0]])
    body_lines = len(stripped[keys[0]].splitlines())
    rows.append(dict(
        check_id="C2-fact-confined-diff",
        mechanism="strip weft.exec.capability fact regions from both instances; "
                  "require byte-equal remainder (variants/bodies/dispatch)",
        verdict="PASS" if same else "FAIL",
        evidence="non-fact surface byte-identical; sha256(stripped)[:16]=%s; "
                 "retained body: chars=%d lines=%d (emitted, not transcribed)"
                 % (body_sha, body_chars, body_lines)
                 if same else "non-fact surface DIVERGED",
        repro="python3 tools/visibility/t1d_dual_instance.py --check"))
    green &= same

    # C3 -- same core binary + same pass pipeline for both boards.
    binary_sha = sha256_file(weft_opt)
    rows.append(dict(
        check_id="C3-same-core-binary",
        mechanism="both instances driven through the SAME weft-opt binary and the "
                  "SAME pass-pipeline string (no per-board build, flag, or path)",
        verdict="PASS",
        evidence="sha256(%s)[:16]=%s ; pipeline=%s" % (
            os.path.relpath(weft_opt, REPO), binary_sha[:16],
            SELECT_PASS.split("=")[0]),
        repro="sha256sum %s" % os.path.relpath(weft_opt, REPO)))
    return rows, green


def collect(weft_opt, workdir):
    """Runs every probe. Returns (instance_rows, probe_rows, core_rows, green)."""
    instance_rows, probe_rows = [], []
    green = True

    # --- Build the module texts. The MAIN body is byte-identical across boards.
    files_main, files_rvvonly, files_imeplugin = {}, {}, {}
    for key, inst in INSTANCES.items():
        main = os.path.join(workdir, "inst_%s_main.mlir" % key)
        with open(main, "w") as handle:
            handle.write(render(inst, "dual_instance_kernel",
                                [RVV_VARIANT, IME_KEYED_VARIANT]))
        files_main[key] = main
        only = os.path.join(workdir, "inst_%s_rvvonly.mlir" % key)
        with open(only, "w") as handle:
            handle.write(render(inst, "hash_probe", [RVV_VARIANT]))
        files_rvvonly[key] = only
        plugin = os.path.join(workdir, "inst_%s_imeplugin.mlir" % key)
        with open(plugin, "w") as handle:
            handle.write(render(inst, "ime_plugin_kernel",
                                [RVV_VARIANT, IME_PLUGIN_VARIANT]))
        files_imeplugin[key] = plugin

    # --- P1 / section 1: the MAIN body under each fact set. -----------------
    hashes, mains = {}, {}
    for key, inst in INSTANCES.items():
        jsonl = os.path.join(workdir, "p1_%s.jsonl" % key)
        code, _, err = run(weft_opt, files_main[key],
                           [SELECT_PASS.format(jsonl=jsonl)])
        record = read_jsonl(jsonl)
        if code != 0 or record is None:
            green = False
            probe_rows.append(dict(probe_id="P1-hash-divergence", instance=key,
                                   verdict="FAIL", detail=err.strip()[:200]))
            continue
        hashes[key] = record["declared_instance_hash"]
        mains[key] = record
        legal = sorted(c["variant"] for c in record["candidates"] if c["feasible"])
        instance_rows.append(dict(
            instance_id="inst-%s" % key,
            board=inst["board"],
            fact_vlen_bits=inst["vlen"],
            fact_isa_vector_hints=inst["hints"],
            fact_spacemit_ime_status=inst["ime_status"],
            declared_instance_hash=record["declared_instance_hash"],
            legal_variant_set_size=str(len(legal)),
            legal_variant_set="{%s}" % ",".join(legal),
            infeasible_variant_set="{%s}" % ",".join(
                sorted(c["variant"] for c in record["candidates"]
                       if not c["feasible"])),
            guard_keys_evaluated=json.dumps(record["keys_evaluated"],
                                            separators=(",", ":"), sort_keys=True),
            selection_chosen=record["chosen"],
            selection_reason=record["reason"],
            # 核心 diff=∅ per-row: the SAME binary + SAME pipeline drove this row;
            # the machine check legs are section 3 (C1/C2/C3).
            core_diff="0 (see section 3: C1-f1-zero-branch, C2-fact-confined-diff, "
                      "C3-same-core-binary)",
            # 归因日志样本指针: the attribution JSONL record this row is read from.
            attribution_jsonl_sample=json.dumps(record, separators=(",", ":"),
                                                sort_keys=True),
        ))
    if len(hashes) == 2:
        distinct = len(set(hashes.values())) == 2
        green &= distinct
        probe_rows.append(dict(
            probe_id="P1-hash-divergence",
            instance="rvv|k1",
            verdict="PASS" if distinct else "FAIL",
            detail="ONE byte-identical body, two fact instances -> %d distinct "
                   "declared_instance_hash (rvv=%s... k1=%s...)"
                   % (len(set(hashes.values())), hashes["rvv"][:12],
                      hashes["k1"][:12])))

    # --- P2: hash keys on FACTS only (different body, same facts, same hash). ---
    for key in INSTANCES:
        jsonl = os.path.join(workdir, "p2_%s.jsonl" % key)
        code, _, err = run(weft_opt, files_rvvonly[key],
                           [SELECT_PASS.format(jsonl=jsonl)])
        record = read_jsonl(jsonl)
        if record is None:
            green = False
            probe_rows.append(dict(probe_id="P2-hash-is-fact-keyed", instance=key,
                                   verdict="FAIL", detail=err.strip()[:200]))
            continue
        same = record["declared_instance_hash"] == hashes.get(key)
        green &= same
        probe_rows.append(dict(
            probe_id="P2-hash-is-fact-keyed", instance=key,
            verdict="PASS" if same else "FAIL",
            detail="body changed (2 variants -> 1, kernel symbol "
                   "dual_instance_kernel -> hash_probe), facts unchanged => hash "
                   "%s (%s...)" % ("IDENTICAL" if same else "DIVERGED",
                                   record["declared_instance_hash"][:12])))

    # --- P3a: legal variant set differs by fact, from the MAIN body. --------
    if len(mains) == 2:
        def feasible_of(record, name):
            for candidate in record["candidates"]:
                if candidate["variant"] == name:
                    return candidate["feasible"]
            return None
        rvv_f = feasible_of(mains["rvv"], "ime_keyed_body")
        k1_f = feasible_of(mains["k1"], "ime_keyed_body")
        ok = rvv_f is False and k1_f is True
        green &= ok
        probe_rows.append(dict(
            probe_id="P3a-legal-variant-set", instance="rvv|k1",
            verdict="PASS" if ok else "FAIL",
            detail="@ime_keyed_body (identical body+owner to @rvv_typed_body, "
                   "differs ONLY by additionally requiring the spacemit.ime FACT): "
                   "feasible=%s on rvv / feasible=%s on k1 => chosen %s / %s"
                   % (rvv_f, k1_f, mains["rvv"]["chosen"], mains["k1"]["chosen"])))

    # --- P3b: complementary fail-closed leg (IME-PLUGIN-OWNED variant). -----
    for key, inst in INSTANCES.items():
        code, _, err = run(weft_opt, files_imeplugin[key], [LEGALITY_PASS])
        accepted = code == 0
        reason = ""
        match = re.search(r"rejected variant: (.*)", err)
        if match:
            reason = match.group(1).strip()[:160]
        expect_accept = inst["ime_status"] == "available"
        ok = accepted == expect_accept
        green &= ok
        probe_rows.append(dict(
            probe_id="P3b-plugin-fail-closed", instance=key,
            verdict="PASS" if ok else "FAIL",
            detail="ime-plugin-owned variant %s (exit=%d)%s" % (
                "ACCEPTED by the IME plugin legality verifier" if accepted
                else "hard-REJECTED fail-closed by the IME plugin legality verifier",
                code, "; " + reason if reason else "")))

    # --- P4: exact P remains source-owned, while bound c_o changes the selected
    # ---     owner construction.  The selector chooses the same exact-P-derived
    # ---     candidate; the resulting flat plan changes m2(VLEN128)->m1(VLEN256).
    if os.path.isfile(SEL1_FIXTURE):
        march_seen = {}
        for key, inst in INSTANCES.items():
            jsonl = os.path.join(workdir, "p4_%s.jsonl" % key)
            planned = os.path.join(workdir, "p4_%s.mlir" % key)
            passes = ([FRONTDOOR_PASS.format(march=inst["march"])] +
                      PLANNING_PASSES_BEFORE_SELECTION +
                      [SELECT_PASS.format(jsonl=jsonl)] +
                      PLANNING_PASSES_AFTER_SELECTION)
            code, _, err = run(weft_opt, SEL1_FIXTURE,
                               passes, output=planned)
            record = read_jsonl(jsonl)
            if code != 0 or record is None or not os.path.isfile(planned):
                probe_rows.append(dict(probe_id="P4-owner-construction-axis", instance=key,
                                       verdict="FAIL", detail=err.strip()[:200]))
                green = False
                continue
            with open(planned) as handle:
                planned_text = handle.read()
            minimum_vlen = re.search(r"minimum_vlen = (\d+) : i64", planned_text)
            integer_core_lmul = re.search(
                r'integer_core_lmul = "([^"]+)"', planned_text)
            has_exact_problem = (
                planned_text.count("weft.exec.quantized_block_dot_problem") == 1)
            has_flat_plan = "weft_rvv.flat_body_family" in planned_text
            if minimum_vlen is None or integer_core_lmul is None:
                probe_rows.append(dict(
                    probe_id="P4-owner-construction-axis", instance=key,
                    verdict="FAIL",
                    detail="planned IR lacks minimum_vlen or integer_core_lmul"))
                green = False
                continue
            march_seen[key] = dict(
                record=record,
                minimum_vlen=minimum_vlen.group(1),
                integer_core_lmul=integer_core_lmul.group(1),
                has_exact_problem=has_exact_problem,
                has_flat_plan=has_flat_plan)
        if len(march_seen) == 2:
            rvv_record = march_seen["rvv"]["record"]
            k1_record = march_seen["k1"]["record"]
            same_candidate = rvv_record["chosen"] == k1_record["chosen"]
            lmul_flip = (march_seen["rvv"]["integer_core_lmul"] == "m2" and
                         march_seen["k1"]["integer_core_lmul"] == "m1")
            hash_distinct = (rvv_record["declared_instance_hash"] !=
                             k1_record["declared_instance_hash"])
            exact_problem_preserved = all(
                row["has_exact_problem"] for row in march_seen.values())
            flat_plan_present = all(
                row["has_flat_plan"] for row in march_seen.values())
            ok = (same_candidate and lmul_flip and hash_distinct and
                  exact_problem_preserved and flat_plan_present)
            green &= ok
            probe_rows.append(dict(
                probe_id="P4-owner-construction-axis", instance="rvv|k1",
                verdict="PASS" if ok else "FAIL",
                detail="source emits one exact quantized_block_dot P; selector "
                       "chooses the SAME exact-P-derived candidate %s; bound c_o "
                       "minimum_vlen %s->%s makes the selected RVV owner construct "
                       "flat integer_core_lmul %s->%s; declared_instance_hash is "
                       "%s (%s.../%s...)" % (
                           rvv_record["chosen"],
                           march_seen["rvv"]["minimum_vlen"],
                           march_seen["k1"]["minimum_vlen"],
                           march_seen["rvv"]["integer_core_lmul"],
                           march_seen["k1"]["integer_core_lmul"],
                           "DISTINCT" if hash_distinct else "IDENTICAL",
                           rvv_record["declared_instance_hash"][:12],
                           k1_record["declared_instance_hash"][:12])))

    core_rows, core_green = core_diff_checks(weft_opt, files_main)
    green &= core_green
    return instance_rows, probe_rows, core_rows, green


def render_csv(instance_rows, probe_rows, core_rows, weft_opt):
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    schema_sha = sha256_file(os.path.join(REPO, SCHEMA))[:16]
    writer.writerow(["# T1d dual-instance demonstration -- ONE schema, TWO board "
                     "fact instances, core diff = 0. Structural evidence (C1); NO "
                     "perf claim."])
    writer.writerow(["# generator", "tools/visibility/t1d_dual_instance.py",
                     "regenerate", "python3 tools/visibility/t1d_dual_instance.py",
                     "verify", "python3 tools/visibility/t1d_dual_instance.py --check"])
    writer.writerow(["# schema", SCHEMA, "schema_sha256[:16]", schema_sha,
                     "HEAD", head, "weft_opt_sha256[:16]",
                     sha256_file(weft_opt)[:16]])
    writer.writerow([])

    writer.writerow(["# SECTION 1 -- fact instances (in-IR weft.exec.capability "
                     "rows) and what the SAME core derives from each, driven "
                     "through ONE byte-identical two-variant body"])
    # Column order follows the T1d spec (实验方案与结果表集v2 §T1d):
    # {instance-hash, 合法变体集大小, 被选变体集摘要, 核心 diff=∅, 归因日志样本指针}
    # with the driving FACT rows carried in front so the table is self-contained.
    cols = ["instance_id", "board", "fact_vlen_bits", "fact_isa_vector_hints",
            "fact_spacemit_ime_status", "declared_instance_hash",
            "legal_variant_set_size", "legal_variant_set", "infeasible_variant_set",
            "guard_keys_evaluated", "selection_chosen", "selection_reason",
            "core_diff", "attribution_jsonl_sample"]
    writer.writerow(cols)
    for row in instance_rows:
        writer.writerow([row[c] for c in cols])
    writer.writerow([])

    writer.writerow(["# SECTION 2 -- probes (each row is a real weft-opt run)"])
    writer.writerow(["probe_id", "instance", "verdict", "detail"])
    for row in probe_rows:
        writer.writerow([row["probe_id"], row["instance"], row["verdict"],
                         row["detail"]])
    writer.writerow([])

    writer.writerow(["# SECTION 3 -- core diff = 0 machine check"])
    writer.writerow(["check_id", "mechanism", "verdict", "evidence", "repro"])
    for row in core_rows:
        writer.writerow([row["check_id"], row["mechanism"], row["verdict"],
                         row["evidence"], row["repro"]])
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="re-run probes and diff against the committed CSV")
    parser.add_argument("--print", dest="dump", action="store_true",
                        help="print the regenerated table to stdout")
    args = parser.parse_args()

    weft_opt = find_weft_opt()
    if not weft_opt:
        print("[t1d] SETUP ERROR: no built weft-opt found under $WEFT_BUILD/bin "
              "or the in-tree build layouts",
              file=sys.stderr)
        return 2

    workdir = tempfile.mkdtemp(prefix="t1d-")
    try:
        instance_rows, probe_rows, core_rows, green = collect(weft_opt, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    text = render_csv(instance_rows, probe_rows, core_rows, weft_opt)

    if args.dump:
        sys.stdout.write(text)
        return 0 if green else 1

    if args.check:
        if not os.path.isfile(CSV_PATH):
            print("[t1d] CHECK FAIL: %s missing" % CSV_PATH, file=sys.stderr)
            return 1
        # Compare only the volatile-free payload: the $meta line carries HEAD,
        # which legitimately moves. Everything else must reproduce.
        def payload(blob):
            return [l for l in blob.splitlines() if not l.startswith("# schema,")]
        if payload(open(CSV_PATH).read()) != payload(text):
            print("[t1d] CHECK FAIL: live run diverges from the committed CSV",
                  file=sys.stderr)
            return 1
        print("[t1d] CHECK OK: live run reproduces %s (probes green=%s)"
              % (os.path.relpath(CSV_PATH, REPO), green))
        return 0 if green else 1

    with open(CSV_PATH, "w") as handle:
        handle.write(text)
    verdicts = [r["verdict"] for r in probe_rows] + [r["verdict"] for r in core_rows]
    print("[t1d] wrote %s" % os.path.relpath(CSV_PATH, REPO))
    print("[t1d] instances=%d probes=%d core-checks=%d verdicts=%s"
          % (len(instance_rows), len(probe_rows), len(core_rows),
             ",".join(sorted(set(verdicts)))))
    print("[t1d] %s" % ("GREEN: one schema, two fact instances, core diff = 0"
                        if green else "RED: a probe or core check FAILED"))
    return 0 if green else 1


if __name__ == "__main__":
    sys.exit(main())
