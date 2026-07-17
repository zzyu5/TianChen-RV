#!/usr/bin/env python3
# [SEL-3] T-SEL3-1 (schema landing) + T-SEL3-2 (ingest transcription)
# READ-ONLY transcription of EXISTING cold. NO new timing. 0 fabricated numbers.
# Touch set: writes ONLY schema/measurement-memory.v1.json (new). Does NOT modify
# schema/tiling-measurements.v1.json (live) or any lib/canon/roster.
#
# Sources:
#   - schema/tiling-measurements.v1.json  (12 live L2 rows: 10 sp4_tiling + 2 loop_order)  -> MIGRATE
#   - experiments/master/T3_master_rebuild.csv                                -> INGEST (deployed_point seeds)
#   - T9_kernel_sym_ledger.md  (opponent caliber classes; consulted for classifier, not re-parsed row-by-row)
#   - onw2 raw: SKIPPED (single aggregate e2e profile, no per-sample distribution => no IQR without new measurement)
import csv, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TM   = os.path.join(ROOT, "schema", "tiling-measurements.v1.json")
T3   = os.path.join(ROOT, "experiments", "active", "result-tables", "T3_master_rebuild.csv")
OUT  = os.path.join(ROOT, "schema", "measurement-memory.v1.json")

tm = json.load(open(TM, encoding="utf-8"))

# ---------- board snapshot locks ----------
def snap(board, march=None, extra=None):
    if board == "rvv":
        s = {"vlen": 128, "vreg_count": 32, "march": march or "rv64gcv",
             "toolchain": {"ours": "clang-17", "opp_shipped": "gcc-15.2"},
             "board_id": "openEuler-VLEN128"}
    else:  # k1
        s = {"vlen": 256, "vreg_count": 32, "march": march or "rv64gcv",
             "toolchain": {"ours": "clang-18", "opp_shipped": "clang-18"},
             "board_id": "SpacemiT-K1-VLEN256"}
    if extra:
        s.update(extra)
    return s

rows = []

# =========================================================================
# T-SEL3-1: MIGRATE the 12 live L2 rows (byte-exact-gated, ab_paired, selection-valid)
# =========================================================================
# --- sp4_tiling (10 rows) ---
for m in tm["measurements"]:
    bi = m["board_identity"]  # verbatim: {"vlen":128,"vreg_count":32,"march":"rv64gcv","toolchain":{"ours":"clang-17","opp_shipped":"gcc-15"}}
    sn = {"vlen": bi["vlen"], "vreg_count": bi["vreg_count"], "march": bi["march"],
          "toolchain": dict(bi["toolchain"]), "board_id": "openEuler-VLEN128"}
    rows.append({
        "declared_instance_hash": m["declared_instance_hash"],
        "instance_hash_seed": False,
        "kernel": m["kernel"],
        "variant": m["variant"],
        "variant_axis": "sp4_tiling",
        "selected": m["selected"],
        # cold_median = the compiler-SYMMETRIC group A/B ratio (s6_tiled/plain, both ours-clang) = the
        # SELECTION-VALID input; direction encodes winner, `selected` is authoritative. Group-mirrored.
        "cold_median": m["ab_wall_ratio_tiled_over_untiled"],
        "cold_iqr": None,
        "ratio_semantics": "ab_paired",
        "selection_valid_input": True,
        "opponent_symbol": {"symbol": "self A/B: s6_tiled vs plain (both ours-clang-17, byte-exact hot core)",
                            "caliber": "self"},
        "snapshot": sn,
        "byte_exact_gate": m["byte_exact_gate"],
        "source": m["source"],
        "ts": m["ts"],
        "axis_extras": {
            "register_cliff_reached": m["register_cliff_reached"],
            "vs_ggml_blockdot_ratio": m["vs_ggml_blockdot_ratio"],
            "vs_ggml_blockdot_opponent": {"symbol": "ggml factory block-dot (as-shipped)", "caliber": "block-dot"},
            "vs_ggml_blockdot_ratio_semantics": "vs_shipped_opp",
            "vs_ggml_blockdot_selection_valid_input": False,
            "vs_ggml_blockdot_note": "[CASE-COMPILER-ASYMMETRY] clang-ours-vs-gcc-shipped artifact; DISCLOSURE-only, selector never keys on it.",
            "ab_wall_ratio_note": "cold_median = ab_wall s6_tiled/plain, group-mirrored on both rows; >1 = s6_tiled faster; `selected` authoritative."
        }
    })

# --- loop_order (2 rows) ---
for lo in tm["loop_order_measurements"]:
    bi = lo["board_identity"]
    sn = {"vlen": bi["vlen"], "vreg_count": bi["vreg_count"], "march": bi["march"],
          "toolchain": dict(bi["toolchain"]), "board_id": "openEuler-VLEN128",
          "taskset": bi.get("taskset"), "threads": bi.get("threads")}
    rows.append({
        "declared_instance_hash": lo["declared_instance_hash"],
        "instance_hash_seed": False,
        "kernel": lo["kernel"],
        "variant": lo["order"],
        "variant_axis": "loop_order",
        "selected": lo["selected"],
        "cold_median": lo["ab_schedule_ratio_colouter_over_rowouter"],
        "cold_iqr": "0.19%" if lo["order"] == "col_outer" else None,  # source: "3-round interleaved median, IQR 0.19%"
        "ratio_semantics": "ab_paired",
        "selection_valid_input": True,
        "opponent_symbol": {"symbol": "self A/B: col_outer vs row_outer (both ours-clang-17, byte-exact hot core)",
                            "caliber": "self"},
        "snapshot": sn,
        "byte_exact_gate": lo["byte_exact_gate"],
        "source": lo["source"],
        "ts": lo["ts"],
        "axis_extras": {
            "weight_panel_larger_stream": lo["weight_panel_larger_stream"],
            "llc_load_miss_ratio_rowouter_over_colouter": lo["llc_load_miss_ratio_rowouter_over_colouter"],
            "backend_idle_pct": lo["backend_idle_pct"],
            "vs_blockdot_shipped_ratio": lo["vs_blockdot_shipped_ratio"],
            "vs_upstream_stock_ratio": lo["vs_upstream_stock_ratio"],
            "vs_blockdot_selection_valid_input": False,
            "system_account_note": "vs_blockdot_shipped / vs_upstream_stock = SYSTEM-account vs gcc-15 shipped; DISCLOSURE-only, never a selection input.",
            "account": lo["account"]
        }
    })

MIGRATED = len(rows)  # must == 12

# =========================================================================
# T-SEL3-2: INGEST T3 existing cold (deployed_point seeds; byte_exact_gate=pending; selection_valid_input=false)
# =========================================================================
def is_num(x):
    if x is None: return False
    x = x.strip()
    if x == "": return False
    try:
        float(x); return True
    except ValueError:
        return False

def caliber(opp, note):
    t = (opp or "") + " || " + (note or "")
    tl = t.lower()
    if "weft-internal" in tl:            return "self-internal"
    if "scalar-ref" in tl or "generic-c" in tl or "(fallback" in tl or "fallback)" in tl:
        return "generic-scalar-ref"
    if "vendor" in tl or "stock ime" in tl:                                return "vendor-ime"
    if "real)" in tl or "hand-brick" in tl or "hand-tuned" in tl or "spacemit" in tl or "STRONG" in t:
        return "hand-brick"
    if "parity-control" in tl:           return "parity-control"
    if "better-vec" in tl:               return "better-vec-blockdot"
    if "light-vec" in tl:                return "light-vec-blockdot"
    if "native-vec" in tl:               return "native-vec-blockdot"
    if "autovec" in tl and "dequantize_row" in tl:  return "scalar-autovec-shipped"
    if "block-dot" in tl:                return "block-dot"
    if "intrinsic" in tl:                return "handwritten-intrinsic-shipped"
    if "lut" in tl or "table" in tl:     return "lut-shipped"
    return "as-shipped-unclassified"

def semantics(opp, note):
    t = (opp or "") + " || " + (note or "")
    tl = t.lower()
    if "scalar-ref" in tl or "generic-c" in tl or "(fallback" in tl or "fallback)" in tl:
        return "vs_generic"
    return "vs_shipped_opp"

ingest = 0
with open(T3, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        op, fmt, engine = r["op"], r["format"], r["engine"]
        regime = r["regime"]
        group = r["group"]
        for board in ("rvv", "k1"):
            cold = r[f"{board}_cold"]
            if not is_num(cold):
                continue
            opp = r[f"{board}_opp_sym"]
            note = r[f"{board}_note"]
            tier = r[f"{board}_tier"]
            disp = r[f"{board}_disp"]
            if "Weft-internal" in (opp or ""):   # 域外/q1_0 internal-A/B: non-denominator, skip
                continue
            sem = semantics(opp, note)
            cal = caliber(opp, note)
            march = "rv64gcv_xsmtvdotii1p0" if (board == "k1" and engine == "ime") else "rv64gcv"
            key_parts = ["SEED", board, op, fmt]
            if engine: key_parts.append(engine)
            if regime: key_parts.append(regime)
            seed_hash = "-".join(key_parts)
            rows.append({
                "declared_instance_hash": seed_hash,
                "instance_hash_seed": True,
                "kernel": fmt,
                "variant": "deployed",
                "variant_axis": "deployed_point",
                "selected": True,  # single-candidate degenerate group (no sibling variant recorded); regret 0, trivial
                "cold_median": float(cold),
                "cold_iqr": None,  # T3 is a closure ledger of single-point ratios; no distribution => sentinel
                "ratio_semantics": sem,
                # CONSERVATIVE default: single-point vs-opponent seed, byte-exact gate NOT individually attested
                # in T3, no paired sibling variant => NOT a memoized-argmin selection input. false regardless of
                # vs_generic/vs_shipped_opp. (Design honesty boundary + PENDING [SEL-3-SELVALID].)
                "selection_valid_input": False,
                "opponent_symbol": {"symbol": opp, "caliber": cal},
                "snapshot": snap(board, march=march),
                "byte_exact_gate": "pending",  # gate not individually attested in T3; upgradeable by T-SEL3-4 writeback
                "source": f"T3_master_rebuild.csv op={op} format={fmt} engine={engine or '-'} regime={regime or '-'} board={board}",
                "ts": "0",
                "axis_extras": {
                    "op": op, "engine": engine or None, "regime": regime or None,
                    "group": group, "tier": tier, "disp": disp, "t3_note": note
                }
            })
            ingest += 1

# =========================================================================
# Assemble $meta + full document
# =========================================================================
meta = {
    "schema_version": "v1.0.0",
    "status": "ACTIVE-SCHEMA (data model landed + seeded). NOT YET wired to any lib/ loader: the in-memory VIEW generalization (lookupMeasurement) is 施工-period task T-SEL3-3, gated on selector-core front-door + [SEL-2] falsifier. This file changes NO core code and is consumed by NO lib/ code today.",
    "declares": "[SEL-3] measurement-memory layer: the general offline-profile argmin cache the capability-keyed selector consults BEFORE its [SEL-1] cold-start prior. Generalizes schema/tiling-measurements.v1.json (SP schedule axis only) to a single rows[] table keyed by (declared_instance_hash, kernel, variant) + variant_axis, covering all bounded variant axes. EVIDENCE-axis mirror data (core-invariants I4): NEVER chooses route/compute/dtype, NEVER a correctness or cost authority in lib/. Authority order: hardware measurement > this cache > static prior.",
    "provenance": "v1.0.0 landed from schema/tiling-measurements.v1.json (v1.1.0); the live schema is UNCHANGED (this is a copy-transform migration, not a move). All cold/parity/spill numbers transcribed from EXISTING cells (T3 / T8-via-tiling-measurements / T9 caliber). ZERO new timing, ZERO fabricated numbers.",
    "authority": [
        "docs/ROADMAP.md#〇.4 (SEL-3 立项: offline-profile argmin + byte-exact gate)",
        "experiments/active/g8-stage3-attack/SEL3-measurement-memory-design.md (design; §2 schema / §3 ingest / §4 iron lines)",
        "schema/tiling-measurements.v1.json (the live SEL-1 schedule-axis instance this generalizes; UNCHANGED)",
        "docs/canon/Weft-RV_科研目标总纲v2.md#[SEL-3]",
        ".trellis/spec/architecture/core-invariants.md#I4",
        "experiments/master/T3_master_rebuild.csv (T3 ingest source)",
        "experiments/active/result-tables/T9_kernel_sym_ledger.md (opponent caliber classes)"
    ],
    "primary_key": ["declared_instance_hash", "kernel", "variant"],
    "honesty_boundary_L2_only": {
        "claim": "The ONLY existing rows that are genuine memoized-argmin SELECTION INPUTS are the ab_paired same-compiler A/B L2 rows (variant_axis in {sp4_tiling, loop_order}) migrated from T8-via-tiling-measurements. These carry selection_valid_input=true. EVERYTHING ELSE is coverage/disclosure seed data.",
        "L1_degenerate": "codegen_lmul (mf2 vs m1) is a DEGENERATE single-candidate axis on both boards ({mf2} only legal; m1 = RVV0.7 cross-ISA non-same-board). Memory column == prior column, regret 0. NO memoized novelty. Not seeded as a live selection axis.",
        "L2_real": "sp4_tiling + loop_order = the real memoized content (q4_K/q2_K/q5_K S6 tiling; q4_K col_outer 2.47x). 12 rows, all selection_valid_input=true, all byte_exact_gate=pass.",
        "L3_structural": "paradigm / plan-shape dispatch is STRUCTURAL keying ([SEL-2]), not measured argmin. Memory does NOT claim L3; it lives at the prior/structural layer.",
        "T3_ingest_is_coverage_not_selection": "The deployed_point rows ingested from T3 are single-point vs-opponent speedup ratios: no paired sibling variant, no IQR distribution, byte-exact gate not individually attested. They are ALL selection_valid_input=false, byte_exact_gate=pending. They SEED the library (source-provenanced, upgradeable by the T-SEL3-4 writeback harness) but drive NO argmin today. This is the honest scope, not a bug: 'the cells memory can actually flip are few, and they are all L2.'",
        "count": {"selection_valid_input_true (all L2 ab_paired)": MIGRATED, "coverage_seeds (T3 deployed_point, selection_valid_input=false)": ingest}
    },
    "key_semantics": {
        "declared_instance_hash": "SHA-256 (lowercase hex) of the EXPANDED normalized capability fact set = the SAME support::computeDeclaredInstanceHash the [D-4](1) exec/schedule attribution sinks compute. march/vlen/vreg are SUBSUMED. Migrated L2 rows carry the REAL @rvv fixture hash (3cd23a4e...). T3-ingested rows carry a SEED SENTINEL ('SEED-<board>-<op>-<fmt>[-<engine>][-<regime>]', instance_hash_seed=true) because the real hash requires computeDeclaredInstanceHash at 施工-period T-SEL3-3; the sentinel is a PLACEHOLDER, not a real hash.",
        "kernel": "the decode_model (measurement-row identity, e.g. 'q4_K'). The SELECTION keys on the bottleneck SHAPE (fold_model), NEVER this format name; kernel is measurement-row identity only, not a dispatch key.",
        "variant": "a bounded enum member within its variant_axis (compile-time hardcoded, closed per axis). 'deployed' is the single-candidate placeholder used by deployed_point coverage seeds."
    },
    "decision_rule": "The memoized winner of a (declared_instance_hash, kernel, variant_axis) group = the candidate with byte_exact_gate==pass AND selection_valid_input==true AND still-feasible, with the best cold_median; on a tie / no decisive difference the SIMPLER variant wins (plain / no-tiling / lower-LMUL). A marginal non-cliff shave does NOT flip the winner (inherits tiling-measurements: q3_K tiled +8% non-cliff => memoized winner stays plain, 'measurement itself says do not tile here'). Structural gates (e.g. sp4_tiling.register_cliff_reached) live in axis_extras, not the common value.",
    "iron_lines_NG1_L4_I4": {
        "no_search": "variant set = bounded compile-time enum; refinement = offline-profile measurement (batch), NOT online learning / autotune search / curve-fit cost model.",
        "no_learned_cost_model": "the library holds NO learnable parameters; static cost formula is only a Stage-1 pruner + no-record fallback, NEVER the cold-start ranking authority (that is [SEL-1] capability prior).",
        "no_external_tuner_benchmark_L4": "oracle is always SELF (per-cell own-variant enumeration micro-best); never cross-cell/cross-board extrapolation, never benchmarked against TVM/AutoTVM/Triton-autotune. external_tuner_hook is DOCUMENTED ONLY (default empty, not implemented).",
        "selector_core_front_door": "landing the VIEW (T-SEL3-3) requires selector-core changes to pass [SEL-2] hard-timing falsifier + T4b ablation discriminative mutation tests + full byte-exact regression + [PERF-1] eight-gate. THIS SCHEMA changes no core.",
        "measurement_is_cache_fact_I4": "authority order hardware measurement > this cache > static prior; the library NEVER chooses route/compute/dtype and is NEVER a lib/ correctness or cost authority.",
        "compiler_symmetry_required_for_selection_input": "a vs_shipped_opp CROSS-compiler cold_median has selection_valid_input=false (成色/disclosure note only); only ab_paired same-compiler A/B rows drive memoized argmin ([CASE-COMPILER-ASYMMETRY]). Single-point vs-opponent seeds (even same-compiler on k1) are also selection_valid_input=false: they lack a paired sibling variant to select among (see PENDING [SEL-3-SELVALID])."
    },
    "fail_closed_revalidate": "The selector discards a cached winner no longer feasible under current capability facts (Stage-1 legality re-check) and falls back to the [SEL-1] prior. A stale cache never forces an infeasible variant.",
    "byte_exact_gate_policy": "A row is SELECTION-ELIGIBLE only past the byte-exact gate (fail-closed): the variant product is bit-identical to the _generic construction oracle (ZERO-MODEL recompute; cert three-requirements corpus-complete / same input path / independent oracle). Enforced at BOTH ingest and writeback. The 12 migrated L2 rows carry byte_exact_gate=pass (T8-attested). T3 coverage seeds carry byte_exact_gate=pending (gate not individually attested in the T3 closure ledger) and are NOT selection-eligible until a writeback attests them.",
    "variant_axis_registry": {
        "note": "closed set of bounded selection axes this superschema unifies; each row's variant_axis names which axis its variant belongs to; axis_extras carries axis-specific structural gates.",
        "sp4_tiling":   {"variants": ["plain", "s6_tiled"], "axis_extras": ["register_cliff_reached", "vs_ggml_blockdot_ratio"], "migrated_from": "tiling-measurements.v1.json#measurements", "layer": "L2"},
        "loop_order":   {"variants": ["col_outer", "row_outer"], "axis_extras": ["weight_panel_larger_stream", "llc_load_miss_ratio_rowouter_over_colouter", "backend_idle_pct", "vs_blockdot_shipped_ratio", "vs_upstream_stock_ratio"], "migrated_from": "tiling-measurements.v1.json#loop_order_measurements", "layer": "L2"},
        "strip_width":  {"variants": ["vl8", "vl16"], "axis_extras": ["vlen_gate"], "layer": "L2", "seed": "q4_K@k1 vl16 1.197x>vl8 (Win-K1-VLEN); NOT transcribed here (sealed vl16 kernel data lives in perf-covered/sealed, not tiling-measurements; a future writeback seeds it)."},
        "codegen_lmul": {"variants": ["mf2", "m1"], "axis_extras": [], "layer": "L1", "note": "DEGENERATE double-board: {mf2} single legal (m1=RVV0.7 cross-ISA non-same-board). Memory column == prior column, regret 0. NO memoized novelty (single candidate). Not seeded."},
        "deployed_point": {"variants": ["deployed"], "axis_extras": ["op", "engine", "regime", "group", "tier", "disp", "t3_note"], "layer": "coverage", "note": "NOT a selection axis. A single deployed-kernel snapshot ingested from T3 (no paired sibling variant recorded). Coverage/disclosure seed only; selection_valid_input=false, byte_exact_gate=pending. Exists so the T-SEL3-4 writeback harness has source-provenanced anchors to upgrade."}
    },
    "value_shape": {
        "declared_instance_hash": "see key_semantics. Real hash on migrated L2 rows; SEED SENTINEL on T3 coverage rows.",
        "instance_hash_seed": "bool -- true iff declared_instance_hash is a placeholder SEED sentinel (T3 rows) rather than a real computeDeclaredInstanceHash value (migrated L2 rows).",
        "kernel": "decode_model / measurement-row identity.",
        "variant": "bounded enum member within variant_axis.",
        "variant_axis": "which selection axis this variant belongs to (see variant_axis_registry).",
        "selected": "true on the group argmin the in-memory view returns (authoritative, stored). On single-candidate groups (codegen_lmul / deployed_point) trivially true (regret 0).",
        "cold_median": "cold-measurement median. Semantics per ratio_semantics. For L2 ab_paired rows = the compiler-symmetric group A/B ratio (>1 = tiled/col_outer faster; group-mirrored, `selected` authoritative). For T3 deployed_point rows = the T3 single-point speedup ratio (>1 = ours faster, per T3 convention).",
        "cold_iqr": "cold-measurement IQR / dispersion guard (oversized IQR => memory untrustworthy => fall to prior). SENTINEL null when the existing source carries no distribution (all migrated SP4 rows; all T3 rows). Only loop_order col_outer carries '0.19%' (source: 3-round interleaved median).",
        "ratio_semantics": "denominator + opponent identity of cold_median: 'vs_generic' (vs _generic / scalar-ref, same-compiler) | 'vs_shipped_opp' (vs board-shipped opponent; [CASE-COMPILER-ASYMMETRY] system-account on rvv) | 'ab_paired' (same-compiler A/B of our own variants; selection input) | 'ns' (absolute).",
        "opponent_symbol": "{symbol, caliber}; caliber in {hand-brick, block-dot, better-vec-blockdot, light-vec-blockdot, native-vec-blockdot, scalar-autovec-shipped, handwritten-intrinsic-shipped, lut-shipped, generic-scalar-ref, vendor-ime, parity-control, self, self-internal, as-shipped-unclassified}.",
        "snapshot": "board identity: {vlen, vreg_count, march, toolchain{ours, opp_shipped}, board_id[, taskset, threads]}. Board lock: rvv={128, rv64gcv, ours clang-17, opp gcc-15.2, openEuler-VLEN128}; k1={256, rv64gcv[_xsmtvdotii1p0 for ime], ours clang-18, opp clang-18, SpacemiT-K1-VLEN256}. NOTE migrated SP4/loop_order rows preserve the source's verbatim opp_shipped 'gcc-15' (== gcc-15.2 point release) rather than re-stamping, per lossless migration.",
        "byte_exact_gate": "'pass' (migrated L2, T8-attested, selection-eligible) | 'pending' (T3 coverage seed, gate not individually attested, NOT selection-eligible).",
        "selection_valid_input": "bool -- true iff the row is a byte-exact-gated, compiler-symmetric, paired-variant memoized-argmin selection input. TRUE only on the 12 migrated ab_paired L2 rows. FALSE on all T3 coverage seeds (single-point, pending-gate, no sibling) and on cross-compiler vs_shipped_opp disclosure numbers.",
        "source": "existing provenance (T8 ledger row + commit / T3 cell coordinates).",
        "ts": "ISO-8601 UTC; sentinel '0' for transcribed seed rows (all rows here: transcription, no new timing).",
        "axis_extras": "object of axis-specific fields keyed per variant_axis_registry."
    },
    "ingest_sources": {
        "T8_via_tiling_measurements": {"file": "schema/tiling-measurements.v1.json", "provides": "12 SP4 + loop-order ab_paired L2 seed rows", "action": "MIGRATED verbatim (copy-transform) into rows[] with variant_axis in {sp4_tiling, loop_order}; register_cliff_reached/weight_panel_larger_stream/vs_* -> axis_extras. Live schema UNCHANGED."},
        "T3": {"file": "experiments/master/T3_master_rebuild.csv", "provides": "dual-board single-point cold speedups + opponent symbols + board identity", "action": "INGESTED as deployed_point coverage seeds; cold_median<-{board}_cold; opponent_symbol<-{board}_opp_sym + T9-style caliber; snapshot<-board lock. IQR absent => sentinel null. selection_valid_input=false, byte_exact_gate=pending.", "skipped": "empty-cold cells (pending-fold / N/A-hw) and Weft-internal q1_0 (域外, non-denominator) are NOT ingested (0 fabricated numbers)."},
        "T9": {"file": "experiments/active/result-tables/T9_kernel_sym_ledger.md", "provides": "opponent caliber classes (hand-brick / block-dot / better-vec / light-vec / native-vec / generic-scalar-ref / vendor)", "action": "consulted for the caliber classifier; applied per-row from the T3 opponent symbol + note text."},
        "onw2": {"file": "tools/e2e-harness/board/g6-m7-ime-vmadot-tiling/raw/{prof_onw2_full.err,c_onw2.out}", "provides": "a SINGLE aggregate e2e IME profile (total ns per phase), NOT a per-sample distribution", "action": "SKIPPED -- no IQR is computable without new measurement (vacation iron line: no new timing). The two k1 IME cold points enter instead via T3 (q4_0@ime 0.196, q4_K@ime 0.049) as vendor-caliber coverage seeds without IQR."}
    },
    "external_tuner_hook": {
        "status": "DOCUMENTED ONLY (default empty, NOT implemented)",
        "note": "an optional offline-profile ingest point for an external tuner's output, PROVIDED it passes the same byte-exact gate and is transcribed as ab_paired/vs_generic rows. Documented插点 per [SEL-3] canon ('留外部 tuner 插点，文档化即可'); it does NOT benchmark against or import an external tuner's cost model ([L-4]). No live consumer.",
        "entries": []
    },
    "case_compiler_asymmetry": tm["$meta"]["case_compiler_asymmetry"],
    "acceptance_T4b": {
        "note": "post-[SEL-3]-施工 fills the T4b '仅记忆'/'先验+记忆' columns (currently N/A(gated)) with HONEST numbers. See design §5. This schema (T-SEL3-1/2) does NOT fill T4b; that is T-SEL3-5.",
        "L1_codegen": "memory column == prior column, regret 0, escape 100% -- trivial NULL (codegen_lmul {mf2} single candidate).",
        "L2_tile_form": "memoized argmin top-1 on cliff-reached / weight-panel-larger cells (sp4_tiling + loop_order) -- the ONLY axis where memory columns differ non-trivially.",
        "L3_paradigm": "memory does NOT claim L3 (shape dispatch = structural keying, [SEL-2] prior layer).",
        "co_bid_regret_zero": "co-bidding row: memory verdict == prior verdict; a stale/cross-compiler false vector-win row is caught by fail-closed-revalidate + selection_valid_input=false => falls to prior => regret 0. Mutation test: inject a selection_valid_input=false fake vector-win row, assert selector still emits matrix.",
        "honesty": "non-trivial real numbers appear ONLY in L2. Do NOT dress L1-trivial / L3-prior / T3-coverage as memoized novelty."
    },
    "pending_rulings": [
        {"id": "[SEL-3-KEY]", "registered": "docs/PENDING_RULINGS.md PR-5", "issue": "执行总纲v2.md:69 says key=kernel+march; ROADMAP令〇.4 + this schema + tiling-measurements use (declared_instance_hash, kernel, variant). Semantically reconcilable (instance-hash SUBSUMES march/vlen/vreg) but canon wording conflicts. Designed to the operative directive (instance-hash); canon wording unification needs user ruling.", "conservative_default": "instance-hash key (per 令〇.4 + live schema); canon [SEL-3] line gets a correction NOTE, not a wording change (hard-freeze)."},
        {"id": "[SEL-3-SELVALID]", "registered": "docs/PENDING_RULINGS.md PR-15", "issue": "design §2.2 lists vs_generic (same-compiler) as a selection-valid input; but design §1 says only ab_paired paired-A/B L2 rows are真正能作 memoized-argmin 选择输入. Tension: is selection_valid_input a pure compiler-symmetry flag (=> vs_generic single-points TRUE) or does it additionally require a paired sibling variant (=> single-point vs-opponent seeds FALSE)?", "conservative_default": "the stricter paired-variant reading: ALL single-point T3 coverage seeds (vs_generic AND vs_shipped_opp) are selection_valid_input=false; only the 12 ab_paired L2 rows are true. This matches the design §1 small-scope thesis and the task honesty guardrail ('T3 多数 selection_valid_input=false 或仅成色')."}
    ],
    "touch_set_discipline": "NEW file schema/measurement-memory.v1.json + transcription script (experiments/active/g8-stage3-attack/sel3_transcribe.py) + PENDING_RULINGS.md PR-15. Does NOT modify tiling-measurements.v1.json (live), lib/, RVVRepackTilingSelection.h, VariantSelection.cpp, or any canon denominator / roster $meta / perf-covered count / headline caliber. Not wired to any schema loader (VERSIONLOG entry is a data-model landing note, not a loader registration). NO git commit. NO new numbers (all引自 T3 / T8-via-tiling-measurements / T9 existing cells).",
    "VERSIONLOG": [
        {"version": "v1.0.0", "date": "2026-07-16", "author": "案头执行员 (度假自治·G8 SEL-3 T-SEL3-1/2)",
         "change": "SEL-3 measurement-memory superschema LANDED + SEEDED. (T-SEL3-1) Generalized schema/tiling-measurements.v1.json v1.1.0 (two arrays measurements[]/loop_order_measurements[]) into a single rows[] keyed by variant_axis; migrated all 12 live L2 rows verbatim (10 sp4_tiling + 2 loop_order) with register_cliff_reached/vs_*/weight_panel/llc/backend_idle -> axis_extras. Preserved case_compiler_asymmetry iron-law block, external_tuner_hook (empty), decision_rule (更简单者胜/non-cliff). (T-SEL3-2) Ingested T3_master_rebuild.csv existing cold as deployed_point coverage seeds (byte_exact_gate=pending, selection_valid_input=false, SEED-sentinel instance hashes); skipped empty-cold + Weft-internal q1_0 + onw2 (no per-sample distribution). L2-only honesty boundary declared at $meta. NO core code, NO canon, NO new timing. tiling-measurements.v1.json UNCHANGED (copy-transform migration).",
         "migrated_from": "schema/tiling-measurements.v1.json v1.1.0"}
    ]
}

doc = {"$meta": meta, "rows": rows}

json.dump(doc, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# ---------- round-trip validate + stats ----------
back = json.load(open(OUT, encoding="utf-8"))
assert back == doc, "ROUND-TRIP MISMATCH"

from collections import Counter
axis_dist = Counter(r["variant_axis"] for r in rows)
sem_dist  = Counter(r["ratio_semantics"] for r in rows)
sv_true   = sum(1 for r in rows if r["selection_valid_input"])
vshipped  = sum(1 for r in rows if r["ratio_semantics"] == "vs_shipped_opp")
noiqr     = sum(1 for r in rows if r["cold_iqr"] is None)
seedhash  = sum(1 for r in rows if r["instance_hash_seed"])
gate_pass = sum(1 for r in rows if r["byte_exact_gate"] == "pass")
gate_pend = sum(1 for r in rows if r["byte_exact_gate"] == "pending")

print("ROUND-TRIP: clean")
print("total rows          :", len(rows))
print("  migrated (L2)      :", MIGRATED, "(expected 12)")
print("  ingested (T3 seeds):", ingest)
print("variant_axis dist    :", dict(axis_dist))
print("ratio_semantics dist :", dict(sem_dist))
print("selection_valid=true :", sv_true, "(expected 12, all L2 ab_paired)")
print("vs_shipped_opp rows  :", vshipped)
print("cold_iqr=null(sent.) :", noiqr)
print("seed-hash rows       :", seedhash)
print("byte_exact_gate pass :", gate_pass, "| pending:", gate_pend)
