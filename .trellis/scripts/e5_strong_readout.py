#!/usr/bin/env python3
"""e5_strong_readout.py — E5 增量① strong-side [L-8] provenance auto-readout.

Machine-derives the STRONG ("constructed") six-state verdict for the 3 existing
strong N-operand contraction routes by WALKING THE ACTUAL REALIZED BODY, then
verifies the verdict reproduces the current hand-label and, via one weak negative
control, that the check discriminates strong from weak (is not vacuously true).

Why this tool exists (contrast with coverage_metrics.py / E6)
------------------------------------------------------------
E6's coverage_metrics.py is stdlib-only and explicitly "does NOT derive
strong-vs-weak from code". That derivation is E5's job. This tool does it the
ONE I4-legal way: it runs the real compiler (tcrv-opt = runner) and walks the
op-identity of the ACTUAL realized `tcrv_rvv.with_vl` body it emits (IR = artifact
parsing). Python is tooling here, never the compiler stack (I6).

The [L-8] machine rule (实验总纲 line 27 / core-invariants [L-8]/[K-4])
----------------------------------------------------------------------
A selected-body path is `constructed` (STRONG) iff:

    manifest non-empty  ∧  no opaque hand-written helper  ∧  decomposed

- manifest = the ordered op-identity list of the ops in the realized `with_vl`
  body (the pattern-library primitives: load / widening_product / *_x_i8_product
  / standalone_reduce / dequantize / store / ...).
- opaque hand-written helper = a MONOLITHIC block-dot op: mnemonic ends `_block_dot`
  AND carries `kind = "ggml_…block_dot"` (the descriptor-selected hand helper that
  lowers to emitFlatBlockDot). The two signals must AGREE, so it is not one fragile
  string match.
- decomposed = the body realizes a REAL dot-product-family primitive
  (widening_product / *_x_i8_product / *_unpack_product / *product_reduce) AND a
  reduce-family primitive (standalone_reduce / *dot_reduce / ...). A per-block fp16
  `block_fp16_scale_product` is a scale multiply, NOT a contraction, so it does not
  satisfy this conjunct — a bare "product" substring would wrongly admit it. This
  conjunct was report-only; it is now a GATE (E5 gate-hole fix).

I4 red line: the manifest is the ACTUAL realized body's op-identity (CORE oracle),
NEVER the `tcrv_rvv.low_precision_resource.*` mirror attributes (which ride on the
with_vl op) nor the emission-plan `rvv_selected_body_typed_compute_op` mirror. The
parser extracts mnemonics from OPERATION position only and asserts no mirror /
attribute-name token ever leaks into a manifest.

Stage discipline: the walk stops at the constructed with_vl body, BEFORE
`--tcrv-rvv-lower-to-emitc` (after that lowering both strong and weak collapse to
indistinguishable `emitc`/`call_opaque` and the discriminator vanishes).

Scope (E5 增量①, bounded): strong side only. This is a standalone read-out +
optional auto_readout write-back. No CI gate, no enforcement, no coverage_metrics.py
edit, no committed JSONL sink — those are later increments. State values never
change (zero flip); only the `auto_readout` field of the strong rows is written.

Subcommands
-----------
report                 run the machine-check; print per-path {manifest, has_opaque,
                       derived_state} + PASS/FAIL vs expected. Exit non-zero on any
                       mismatch or any tcrv-opt failure (fail-closed).
update-sixstate        run report, then (only if all rows pass) rewrite the strong
                       rows' `auto_readout` in schema/coverage-sixstate.v1.json with
                       the machine result + manifest summary. Never touches `state`.
--self-test            hermetic parser test over the 4 captured ground-truth bodies.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# --- locations -------------------------------------------------------------
# This file lives at <repo>/.trellis/scripts/e5_strong_readout.py
REPO_ROOT = Path(__file__).resolve().parents[2]
TCRV_OPT = REPO_ROOT / "build" / "bin" / "tcrv-opt"
SIXSTATE_JSON = REPO_ROOT / "schema" / "coverage-sixstate.v1.json"
TEST_RVV = REPO_ROOT / "test" / "Target" / "RVV"

# --- the 4 strong routes + 1 weak negative control -------------------------
# Each entry: the six-state (op, format) key, the source-op test input, and the
# single front-door pass that CONSTRUCTS/REALIZES the typed with_vl body. NO
# --tcrv-rvv-lower-to-emitc (stage discipline). Row<->test mapping per research
# emission-paths-map.md §1b.
PATHS = [
    {
        "op": "product_reduce", "format": "q4_0_nibble", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "non-deferred-wide-product-reduce-dequantize-f32-front-door-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-widening-dot-reduce-dequantize-source-front-door=march=rv64gcv",
        "front_door_id": "RVVDequantDotSourceFrontDoor",
    },
    {
        "op": "product_reduce", "format": "offset_binary_n3", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "packed-i4-offset-binary-dot-product-reduce-front-door-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-packed-i4-offset-binary-dot-source-front-door=march=rv64gcv",
        "front_door_id": "RVVPackedI4DotSourceFrontDoor",
    },
    {
        "op": "product_reduce", "format": "codebook_n3", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "codebook-gather-dot-product-reduce-front-door-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-codebook-gather-dot-source-front-door=march=rv64gcv",
        "front_door_id": "RVVCodebookDotSourceFrontDoor",
    },
    # q8_0 vec_dot: STRONG. Its front door was upgraded to construct the typed flat
    # block-dot LOOP body (tcrv_rvv.typed_flat_block_dot_loop_body) out of decomposed
    # pattern-library primitives (load + widening_product + standalone_reduce), NOT the
    # opaque emitFlatBlockDot hand helper. The SAME front-door pass now realizes a typed
    # body, so update-sixstate machine-reads the REAL constructor output (no hand .mlir).
    {
        "op": "vec_dot", "format": "q8_0", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q8-0-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door",
        "front_door_id": "createTypedFlatBlockDotLoopChain (typed flat block-dot loop body)",
    },
    # q4_0 vec_dot: STRONG. Same typed flat block-dot LOOP construction as q8_0, but the
    # decomposed body realizes the packed-i4 offset-binary dot primitive
    # (tcrv_rvv.packed_i4_offset_binary_x_i8_product) instead of the plain widening_product,
    # since q4_0 weights are packed nibbles. Its front door constructs the typed
    # tcrv_rvv.typed_flat_block_dot_loop_body (NOT the opaque emitFlatBlockDot hand helper),
    # so update-sixstate machine-reads the REAL constructor output (no hand .mlir).
    {
        "op": "vec_dot", "format": "q4_0", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q4-0-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door",
        "front_door_id": "createTypedFlatBlockDotLoopChain (typed flat block-dot loop body)",
    },
    # q4_1 vec_dot: STRONG. Same typed flat block-dot LOOP construction as q4_0, but the
    # decomposed body realizes q4_1's Family-B scale+MIN structure: the integer core is the
    # unsigned-nibble dot primitive (tcrv_rvv.unsigned_nibble_x_i8_product), and the per-block
    # dequant is a dual-fp16 scale+min fold (block_fp16_scale_product d_x.d_y + block_fp16_min_product
    # m_x.s_y, BOTH operand-derived from the block data, feeding block_computed_scale_dequant).
    # Its front door constructs the typed tcrv_rvv.typed_flat_block_dot_loop_body (NOT the opaque
    # emitFlatBlockDot hand helper), so update-sixstate machine-reads the REAL constructor output.
    {
        "op": "vec_dot", "format": "q4_1", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q4-1-q8-1-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q4-1-q8-1-block-dot-source-front-door",
        "front_door_id": "createTypedFlatBlockDotLoopChain (typed flat block-dot loop body)",
    },
    # q5_0 vec_dot: STRONG. Its front door was upgraded (isQ50TypedFlat in the
    # typedFlatLoopPath gate) to construct the typed flat block-dot LOOP body, whose
    # decomposed integer core realizes q5_0's five-bit offset-binary dot primitive
    # (tcrv_rvv.five_bit_offset_binary_x_i8_product) fed by a per-block qh 5th-bit
    # source brick (block_five_bit_qh_source), plus standalone_reduce and a per-block
    # fp16 scale dequant. NO opaque emitFlatBlockDot hand helper, so [L-8] derives
    # constructed (STRONG). update-sixstate machine-reads the REAL constructor output.
    {
        "op": "vec_dot", "format": "q5_0", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q5-0-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q5-0-q8-0-block-dot-source-front-door",
        "front_door_id": "createTypedFlatBlockDotLoopChain (typed flat block-dot loop body)",
    },
    # q5_1 vec_dot: STRONG (cohort 5/5, LAST). Same typed flat block-dot LOOP body via the
    # front-door gate; decomposed integer core = q5_1's five-bit offset-binary dot
    # (five_bit_offset_binary_x_i8_product, bias-off) + qh 5th-bit source
    # (block_five_bit_qh_source) + the min term (block_fp16_min_product), ScalePlusMin fold.
    # NO opaque emitFlatBlockDot hand helper, so [L-8] derives constructed (STRONG).
    {
        "op": "vec_dot", "format": "q5_1", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q5-1-q8-1-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q5-1-q8-1-block-dot-source-front-door",
        "front_door_id": "createTypedFlatBlockDotLoopChain (typed flat block-dot loop body)",
    },
    # q4_K vec_dot: STRONG (the FIRST K-quant super-block flipped, M-FLAT milestone-3).
    # Its front door constructs the typed SUPER-BLOCK dual-accumulator loop body
    # (tcrv_rvv.typed_super_block_block_dot_loop_body) out of the 5 decomposed q4_K
    # bricks (q4_k_nibble_unpack -> q4_k_scale_min_bit_dance -> q4_k_scaled_dot ->
    # q4_k_min_term -> q4_k_sums_fold_scale_d), NOT the opaque emitQ4_KQ8_KBlockDot
    # hand helper. The contraction+reduction is the fused per-sub-block integer-MAC
    # q4_k_scaled_dot (vwmacc into aux32); NO opaque *_block_dot op, so [L-8] derives
    # constructed (STRONG). update-sixstate machine-reads the REAL constructor output.
    {
        "op": "vec_dot", "format": "q4_K", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q4-k-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q4-k-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockBlockDotLoopChain (typed super-block block-dot loop body)",
    },
    # q5_K vec_dot: STRONG (the SECOND K-quant super-block flipped). q5_K == q4_K + the
    # qh 5th-bit plane: it REUSES the SAME typed super-block dual-accumulator loop body
    # (tcrv_rvv.typed_super_block_block_dot_loop_body) out of the SAME 5 decomposed
    # bricks (q4_k_nibble_unpack -> q4_k_scale_min_bit_dance -> q4_k_scaled_dot ->
    # q4_k_min_term -> q4_k_sums_fold_scale_d), the ONLY net-new work being BRICK 1's
    # weight_qh_byte_offset attr (the emitter injects the 5th bit under cx.hasQh). NOT
    # the opaque emitQ5_KQ8_KBlockDot hand helper (retired same action as the flip). The
    # contraction+reduction is the fused per-sub-block integer-MAC q4_k_scaled_dot
    # (vwmacc into aux32); NO opaque *_block_dot op, so [L-8] derives constructed
    # (STRONG). The manifest is byte-identical to q4_K's (the qh inject is intra-brick,
    # not a new op), resolved to q5_K's OWN export entry by weight_block_stride 176.
    {
        "op": "vec_dot", "format": "q5_K", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q5-k-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q5-k-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockBlockDotLoopChain (typed super-block block-dot loop body; q5_K stamps BRICK 1 qh offset)",
    },
    # Negative control (weak descriptor-selected block-dot). iq4_nl is NOT in the front
    # door's typedFlatLoopPath gate (only q8_0/q4_0/q4_1/q5_0/q5_1 now), so its front door
    # auto-constructs the MONOLITHIC
    # tcrv_rvv.iq4_nl_q8_0_block_dot op (kind "ggml_iq4_nl_q8_0_block_dot"), which
    # is_opaque_hand_helper matches -> [L-8] derives NOT-strong (constructed-weak). This
    # keeps the check proven discriminating (not vacuously true) now that q5_0 is strong.
    {
        "op": "vec_dot", "format": "iq4_nl", "engine": "",
        "kind": "negative", "expected_state": "constructed-weak",
        "input": "iq4-nl-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door",
        "front_door_id": "emitFlatBlockDot (descriptor-selected hand helper)",
    },
]

# --- op-identity parse (position-anchored; I4-safe) ------------------------
# Match a tcrv_rvv op mnemonic ONLY in operation position: line-leading (after
# indent), optional `%result = ` prefix. This never matches attribute-name tokens
# (`tcrv_rvv.low_precision_resource.*`, `tcrv_rvv.gearbox.*`) or type tokens
# (`!tcrv_rvv.vector`, `!tcrv_rvv.vl`), which never start a line in op position.
_OP_RE = re.compile(r"^\s*(?:%[A-Za-z0-9_#]+\s*=\s*)?(tcrv_rvv\.[A-Za-z0-9_]+)\b")
_KIND_RE = re.compile(r'\bkind\s*=\s*"([^"]+)"')
_WITHVL_TERMINATOR = re.compile(r"^(\s*)\}\s*:\s*!tcrv_rvv\.vl\s*$")
# Defense-in-depth I4 guard. The REAL protection is _OP_RE (position-anchored, so
# it can only capture operation mnemonics, never the mid-line attribute tokens
# `tcrv_rvv.low_precision_resource.*` / `tcrv_rvv.gearbox.*` nor the `!tcrv_rvv.*`
# type keywords) plus skipping the with_vl opener line. This guard additionally
# raises if a bare mirror-namespace token ever slipped through as a mnemonic. It is
# anchored to the EXACT namespace token _OP_RE would yield (its `\w+` class stops at
# the `.`, so a leaked attr collapses to `tcrv_rvv.low_precision_resource` /
# `tcrv_rvv.gearbox`). Anchoring with `$` is deliberate: it must NOT fire on the
# legitimate pattern-library primitive `tcrv_rvv.gearbox_cross_region_handoff`.
_MIRROR_GUARD = re.compile(r"^tcrv_rvv\.(low_precision_resource|gearbox)$")

# [L-8] decomposed-gate whitelist. A REAL dot-product-family primitive is one of:
# widening_product (+ *_widening_product_intrinsic), *_x_i8_product (the mixed
# codebook / packed-i4 dot forms routes 2&3 realize), *_unpack_product, a fused
# *product_reduce, or the q4_K/q5_K super-block per-sub-block integer-MAC
# `q4_k_scaled_dot` (which BOTH multiplies the unpacked aux8 weights by the q8
# activation AND reduces the 32 products into the aux32 accumulator via vwmacc --
# a fused dot-reduce, see _FUSED_DOT_REDUCE_RE). This is a WHITELIST on purpose:
# `block_fp16_scale_product` is a per-block fp16 SCALE multiply — not a
# contraction — and carries none of these tokens, so it is excluded. A bare
# "product" substring would wrongly admit it.
_DOT_PRODUCT_RE = re.compile(r"(widening_product|_x_i8_product|_unpack_product|product_reduce|scaled_dot)")

# Fused dot-reduce primitives that carry the reduction INSIDE the product op (no
# separate standalone_reduce in the manifest): the q4_K/q5_K super-block
# `q4_k_scaled_dot` runs a per-sub-block vwmacc that accumulates the 32 products
# into the running aux32, so it satisfies BOTH the product AND the reduce conjunct
# of the decomposed gate. This is deliberately NARROW (a scale-only body has no
# scaled_dot; an opaque *_block_dot still trips the opaque gate), so the check
# stays discriminating.
_FUSED_DOT_REDUCE_RE = re.compile(r"scaled_dot")


def _leading_ws(line):
    return len(line) - len(line.lstrip(" "))


def parse_realized_body(ir_text):
    """Walk the realized `tcrv_rvv.with_vl` body and return its op-identity manifest.

    Returns list of {"mnemonic": str, "kind": str|None} for each op directly
    printed inside the with_vl region (nested region ops included). Bounds the
    region by the with_vl opener line and its indent-matched `} : !tcrv_rvv.vl`
    terminator. Raises on a leaked mirror/type token (parser sanity).
    """
    lines = ir_text.splitlines()
    # Find the with_vl opener: an op-position `tcrv_rvv.with_vl` whose line opens a
    # region (ends with `{`).
    opener_idx = None
    opener_indent = None
    for i, line in enumerate(lines):
        m = _OP_RE.match(line)
        if m and m.group(1) == "tcrv_rvv.with_vl" and line.rstrip().endswith("{"):
            opener_idx = i
            opener_indent = _leading_ws(line)
            break
    if opener_idx is None:
        raise RuntimeError("no realized tcrv_rvv.with_vl body found in emitted IR")

    manifest = []
    closed = False
    for line in lines[opener_idx + 1:]:
        term = _WITHVL_TERMINATOR.match(line)
        if term is not None and len(term.group(1)) == opener_indent:
            closed = True
            break
        m = _OP_RE.match(line)
        if not m:
            continue
        mnemonic = m.group(1)
        if mnemonic == "tcrv_rvv.with_vl":
            continue
        if _MIRROR_GUARD.search(mnemonic):
            raise RuntimeError(
                "parser leaked a non-op / mirror token into the manifest: "
                f"{mnemonic!r} (I4 violation guard tripped)"
            )
        kind_m = _KIND_RE.search(line)
        manifest.append({"mnemonic": mnemonic, "kind": kind_m.group(1) if kind_m else None})
    if not closed:
        raise RuntimeError("with_vl region terminator `} : !tcrv_rvv.vl` not found")
    return manifest


def is_opaque_hand_helper(op):
    """[L-8] opaque hand-written helper: a MONOLITHIC block-dot op.

    Two signals must AGREE (not one fragile match): op mnemonic ends `_block_dot`
    AND its `kind` names a ggml block-dot hand helper (`ggml_…block_dot`).
    """
    mnem = op["mnemonic"]
    kind = op["kind"] or ""
    name_signal = mnem.endswith("_block_dot")
    kind_signal = bool(re.match(r"ggml_.*block_dot", kind))
    return name_signal and kind_signal


def derive(manifest):
    """Apply the [L-8] rule: constructed iff
    (manifest non-empty ∧ no opaque *_block_dot helper ∧ decomposed).

    `decomposed` is the promoted GATE conjunct (was report-only): the body must
    realize a REAL dot-product-family primitive AND a separate reduce-family
    primitive, so a body that merely carries a per-block fp16 scale_product — or no
    contraction at all — does not vacuously read as constructed.
    """
    mnemonics = [op["mnemonic"] for op in manifest]
    opaque_ops = [op["mnemonic"] for op in manifest if is_opaque_hand_helper(op)]
    has_opaque = len(opaque_ops) > 0
    non_empty = len(mnemonics) > 0
    # GATE conjunct: real dot-product-family primitive (whitelisted, so the fp16
    # scale_product is excluded) AND a reduce-family primitive.
    has_product = any(_DOT_PRODUCT_RE.search(m) for m in mnemonics)
    # A separate reduce-family primitive (standalone_reduce / *dot_reduce / ...) OR
    # a fused dot-reduce primitive (q4_K scaled_dot: the vwmacc reduction is fused
    # into the product op), so the super-block route — whose contraction+reduction
    # live in ONE brick — is not wrongly demoted.
    has_reduce = any("reduce" in m for m in mnemonics) or any(
        _FUSED_DOT_REDUCE_RE.search(m) for m in mnemonics)
    decomposed = has_product and has_reduce
    derived_state = (
        "constructed" if (non_empty and not has_opaque and decomposed)
        else "constructed-weak"
    )
    return {
        "manifest": mnemonics,
        "has_opaque": has_opaque,
        "opaque_ops": opaque_ops,
        "has_product": has_product,
        "has_reduce": has_reduce,
        "decomposed": decomposed,
        "derived_state": derived_state,
    }


def run_tcrv_opt(input_path, front_door):
    if not TCRV_OPT.exists():
        raise RuntimeError(f"tcrv-opt not built at {TCRV_OPT} (build first)")
    cmd = [str(TCRV_OPT), str(input_path), front_door]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Fail-closed: never silently drop a strong row.
        raise RuntimeError(
            f"tcrv-opt failed (rc={proc.returncode}) for {input_path.name}\n"
            f"cmd: {' '.join(cmd)}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def check_path(entry):
    input_path = TEST_RVV / entry["input"]
    ir = run_tcrv_opt(input_path, entry["front_door"])
    result = derive(parse_realized_body(ir))
    result["op"] = entry["op"]
    result["format"] = entry["format"]
    result["engine"] = entry["engine"]
    result["kind"] = entry["kind"]
    result["expected_state"] = entry["expected_state"]
    result["front_door_id"] = entry["front_door_id"]
    result["input"] = entry["input"]
    result["match"] = result["derived_state"] == entry["expected_state"]
    return result


def cmd_report(_args):
    results = [check_path(e) for e in PATHS]
    all_pass = True
    for r in results:
        status = "PASS" if r["match"] else "FAIL"
        if not r["match"]:
            all_pass = False
        print(f"[{status}] {r['op']}/{r['format']} ({r['kind']}) "
              f"via {r['front_door_id']}")
        print(f"        manifest      = {r['manifest']}")
        print(f"        has_opaque    = {r['has_opaque']}"
              + (f"  {r['opaque_ops']}" if r["opaque_ops"] else ""))
        print(f"        decomposed    = {r['decomposed']} (GATE: real dot-product + reduce)")
        print(f"        derived_state = {r['derived_state']}"
              f"  (expected {r['expected_state']})")
    print()
    strong = [r for r in results if r["kind"] == "strong"]
    neg = [r for r in results if r["kind"] == "negative"]
    print(f"strong reproduce hand-label: "
          f"{sum(1 for r in strong if r['match'])}/{len(strong)} = constructed")
    print(f"negative discriminates (not-strong): "
          f"{'yes' if all(not r['manifest'] or r['derived_state'] != 'constructed' for r in neg) else 'NO'}")
    print(f"overall: {'PASS' if all_pass else 'FAIL'}")
    return results, all_pass


def _manifest_summary(mnemonics):
    """Short, order-preserving manifest summary for the auto_readout field."""
    short = [m.replace("tcrv_rvv.", "") for m in mnemonics]
    return "+".join(short)


def cmd_update_sixstate(_args):
    results, all_pass = cmd_report(_args)
    if not all_pass:
        print("\nrefusing to update schema: machine-check did not fully pass", file=sys.stderr)
        return 1
    by_key = {(r["op"], r["format"], r["engine"]): r for r in results if r["kind"] == "strong"}
    doc = json.loads(SIXSTATE_JSON.read_text())
    updated = 0
    for row in doc["states"]:
        key = (row.get("op"), row.get("format"), row.get("engine", ""))
        if key in by_key and row.get("auto_readout") == "pending-E5":
            r = by_key[key]
            # ZERO FLIP: state is unchanged; only auto_readout is written.
            assert row["state"] == "constructed", row
            row["auto_readout"] = (
                f"E5-increment1-auto: constructed (STRONG); realized-body manifest="
                f"{_manifest_summary(r['manifest'])}; opaque_helper=false"
            )
            updated += 1
    # Idempotent: the E5 增量① paragraph is appended only once. Re-running
    # update-sixstate must not duplicate it (the JSON already carries the corrected,
    # count-accurate paragraph), so guard on its marker.
    if "E5 增量①" not in doc["$meta"]["labeling"]:
        doc["$meta"]["labeling"] = (
            doc["$meta"]["labeling"]
            + " | E5 增量① (strong-side auto): the 7 STRONG rows (3 product_reduce "
              "N-operand routes + q8_0 vec_dot + q4_0 vec_dot + q4_1 vec_dot + q5_0 vec_dot, all four typed_flat_block_dot_loop_body) carry a "
              "MACHINE-CHECKED auto_readout derived by e5_strong_readout.py, which walks "
              "the actual realized tcrv_rvv.with_vl body op-identity (CORE oracle, not the "
              "low_precision_resource.* mirror) and applies [L-8] (manifest non-empty ∧ no "
              "opaque *_block_dot hand helper). Reproduces the hand-label; an iq4_nl block-dot "
              "negative control derives NOT-strong. Weak rows' auto_readout stays pending-E5 "
              "(later increment). State values are unchanged (zero flip)."
        )
    # ensure_ascii=False preserves the existing UTF-8 (实验总纲 etc.) so the diff
    # stays minimal and this file's byte content is not needlessly re-escaped.
    SIXSTATE_JSON.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"\nupdated auto_readout on {updated} strong rows in {SIXSTATE_JSON}")
    return 0


# --- hermetic parser self-test over captured ground truth ------------------
_GT_STRONG = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "lhs"} : !tcrv_rvv.runtime_abi_value
      %6 = tcrv_rvv.setvl %5 {lmul = "m2"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %6 attributes {lmul = "m2", tcrv_rvv.low_precision_resource.source_lmul = "m2", tcrv_rvv.gearbox.producer_scope = "x"} {
        %7 = tcrv_rvv.load %0, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
        %8 = tcrv_rvv.load %1, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
        %9 = tcrv_rvv.widening_product %7, %8, %6 {kind = "signed_widening_product"} : !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
        %10 = tcrv_rvv.standalone_reduce %9, %2, %6 {kind = "signed_widening_reduce_add"} : !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %11 = tcrv_rvv.dequantize %10, %3, %6 {kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        tcrv_rvv.store %4, %11, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
"""

_GT_WEAK = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %3 = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %8 = tcrv_rvv.setvl %0 {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %8 attributes {lmul = "m1", tcrv_rvv.low_precision_resource.source_lmul = "m1"} {
        %9 = tcrv_rvv.q8_0_q8_0_block_dot %3, %5, %1, %0, %8 {kind = "ggml_q8_0_q8_0_block_dot", qk = 32 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
"""

# Scale-only ground truth: NOT opaque (no *_block_dot), but the only "product" op is
# the per-block fp16 SCALE product and there is NO reduce. This is exactly the hole
# the decomposed gate closes: it must derive NOT-strong.
_GT_SCALE_ONLY = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "lhs"} : !tcrv_rvv.runtime_abi_value
      %6 = tcrv_rvv.setvl %5 {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %6 attributes {lmul = "m1"} {
        %7 = tcrv_rvv.load %0, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f16, "m1">
        %8 = tcrv_rvv.block_fp16_scale_product %7, %2, %6 {kind = "block_fp16_scale_product"} : !tcrv_rvv.vector<f16, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        tcrv_rvv.store %4, %8, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
"""

# Mixed-dot ground truth: routes 2 & 3 realize a `*_x_i8_product` primitive (NOT
# widening_product). Guards the gate against a "simplify to widening_product only"
# regression that would silently demote two-thirds of the strong routes.
_GT_XI8 = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "lhs"} : !tcrv_rvv.runtime_abi_value
      %6 = tcrv_rvv.setvl %5 {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %6 attributes {lmul = "m1"} {
        %7 = tcrv_rvv.load %0, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
        %8 = tcrv_rvv.load %1, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
        %9 = tcrv_rvv.codebook_gather_x_i8_product %7, %8, %6 {kind = "codebook_gather_x_i8_product"} : !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m2">
        %10 = tcrv_rvv.standalone_reduce %9, %2, %6 {kind = "signed_widening_reduce_add"} : !tcrv_rvv.vector<i16, "m2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %4, %10, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Super-block ground truth (q4_K milestone-3): the typed SUPER-BLOCK dual-accumulator
# loop body decomposes into the 5 q4_K bricks. The contraction+reduction is the
# FUSED per-sub-block integer-MAC `q4_k_scaled_dot` (vwmacc into aux32) -- NO
# separate standalone_reduce and NO opaque *_block_dot op -- so the decomposed gate
# must derive constructed via the fused dot-reduce path.
_GT_SUPERBLOCK = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", fold_model = "super_block_two_level_scale_min"} {
        ^bb0(%ib: index, %sums: !tcrv_rvv.vector<f32, "m2">, %sumf: f32):
          %b1 = tcrv_rvv.q4_k_nibble_unpack %vx, %vl block %ib : index {kind = "q4_k_nibble_unpack"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %b2 = tcrv_rvv.q4_k_scale_min_bit_dance %vx, %vl block %ib : index {kind = "q4_k_scale_min_bit_dance"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %b3 = tcrv_rvv.q4_k_scaled_dot %vx, %vx, %vy, %vl block %ib : index {kind = "q4_k_scaled_dot"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %b4 = tcrv_rvv.q4_k_min_term %vx, %vx, %vy, %vl block %ib : index {kind = "q4_k_min_term"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %b6 = tcrv_rvv.q4_k_sums_fold_scale_d %vx, %vx, %vy, %vl block %ib : index {kind = "q4_k_sums_fold_scale_d"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sums, %sumf : !tcrv_rvv.vector<f32, "m2">, f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


def cmd_self_test(_args):
    # Strong ground truth: decomposed primitives, no opaque, no mirror leak.
    strong = derive(parse_realized_body(_GT_STRONG))
    assert strong["manifest"] == [
        "tcrv_rvv.load", "tcrv_rvv.load", "tcrv_rvv.widening_product",
        "tcrv_rvv.standalone_reduce", "tcrv_rvv.dequantize", "tcrv_rvv.store",
    ], strong["manifest"]
    assert strong["has_opaque"] is False, strong
    assert strong["has_product"] is True, strong
    assert strong["has_reduce"] is True, strong
    assert strong["decomposed"] is True, strong
    assert strong["derived_state"] == "constructed", strong
    # No mirror token leaked (guard would have raised; double-check explicitly).
    assert all(not _MIRROR_GUARD.search(m) for m in strong["manifest"]), strong
    # The guard must NOT false-fire on the legitimate pattern-library primitive
    # tcrv_rvv.gearbox_cross_region_handoff (a real strong-body op in other routes),
    # but MUST fire on a bare leaked mirror-namespace token.
    assert not _MIRROR_GUARD.search("tcrv_rvv.gearbox_cross_region_handoff")
    assert _MIRROR_GUARD.search("tcrv_rvv.low_precision_resource")
    assert _MIRROR_GUARD.search("tcrv_rvv.gearbox")

    # Mixed-dot ground truth: a `*_x_i8_product` primitive (routes 2 & 3) must ALSO
    # pass the whitelist gate, not just widening_product.
    xi8 = derive(parse_realized_body(_GT_XI8))
    assert xi8["manifest"] == [
        "tcrv_rvv.load", "tcrv_rvv.load", "tcrv_rvv.codebook_gather_x_i8_product",
        "tcrv_rvv.standalone_reduce", "tcrv_rvv.store",
    ], xi8["manifest"]
    assert xi8["has_product"] is True, xi8
    assert xi8["decomposed"] is True, xi8
    assert xi8["derived_state"] == "constructed", xi8

    # Weak ground truth: single monolithic block-dot, opaque, not-strong.
    weak = derive(parse_realized_body(_GT_WEAK))
    assert weak["manifest"] == ["tcrv_rvv.q8_0_q8_0_block_dot"], weak["manifest"]
    assert weak["has_opaque"] is True, weak
    assert weak["opaque_ops"] == ["tcrv_rvv.q8_0_q8_0_block_dot"], weak
    assert weak["derived_state"] == "constructed-weak", weak

    # Scale-only ground truth: NOT opaque, but the decomposed GATE must reject it —
    # `block_fp16_scale_product` is a scale multiply, not a contraction, and there is
    # no reduce. This is the exact hole the gate closes.
    scale = derive(parse_realized_body(_GT_SCALE_ONLY))
    assert scale["manifest"] == [
        "tcrv_rvv.load", "tcrv_rvv.block_fp16_scale_product", "tcrv_rvv.store",
    ], scale["manifest"]
    assert scale["has_opaque"] is False, scale
    assert scale["has_product"] is False, scale
    assert scale["has_reduce"] is False, scale
    assert scale["decomposed"] is False, scale
    assert scale["derived_state"] == "constructed-weak", scale

    # Super-block ground truth (q4_K milestone-3): the 5 q4_K bricks decompose the
    # super-block dot; the FUSED per-sub-block q4_k_scaled_dot satisfies BOTH the
    # product AND reduce conjunct (the vwmacc reduction is fused into the product),
    # and no opaque *_block_dot appears, so it derives constructed.
    superblock = derive(parse_realized_body(_GT_SUPERBLOCK))
    assert superblock["manifest"] == [
        "tcrv_rvv.typed_super_block_block_dot_loop_body",
        "tcrv_rvv.q4_k_nibble_unpack", "tcrv_rvv.q4_k_scale_min_bit_dance",
        "tcrv_rvv.q4_k_scaled_dot", "tcrv_rvv.q4_k_min_term",
        "tcrv_rvv.q4_k_sums_fold_scale_d",
        "tcrv_rvv.typed_super_block_block_dot_loop_yield",
    ], superblock["manifest"]
    assert superblock["has_opaque"] is False, superblock
    assert superblock["has_product"] is True, superblock
    assert superblock["has_reduce"] is True, superblock
    assert superblock["decomposed"] is True, superblock
    assert superblock["derived_state"] == "constructed", superblock

    # Discrimination: same rule, opposite verdicts — including the non-opaque hole.
    assert strong["derived_state"] != weak["derived_state"]
    assert strong["derived_state"] != scale["derived_state"]
    assert superblock["derived_state"] != weak["derived_state"]
    assert superblock["derived_state"] != scale["derived_state"]
    print("self-test PASS: parser position-anchored, no mirror leak; "
          "strong(widening_product)=constructed / strong(x_i8_product)=constructed / "
          "strong(super-block scaled_dot fused reduce)=constructed / "
          "weak(block-dot)=constructed-weak / scale-only=constructed-weak (decomposed gate)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="hermetic parser test over captured ground-truth bodies")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("report", help="run the machine-check and print the read-out")
    sub.add_parser("update-sixstate",
                   help="write the strong rows' auto_readout (only if all pass)")
    args = ap.parse_args()
    if args.self_test:
        return cmd_self_test(args)
    if args.cmd == "update-sixstate":
        return cmd_update_sixstate(args)
    # default: report
    _results, all_pass = cmd_report(args)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
