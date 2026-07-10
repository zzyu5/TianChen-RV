#!/usr/bin/env python3
# tools/lint/check_construction_manifest_regex.py — F-1 construction-manifest shape gate.
#
# C1 (合取存在性) fail-closed STRUCTURAL machine-check. For every `constructed` (== STRONG)
# cell in the six-state schema, the E5 `auto_readout` carries a MACHINE-DERIVED realized-body
# manifest (from .trellis/scripts/e5_strong_readout.py, which walks the actual realized
# tcrv_rvv body op-identity and applies [L-8]). This gate REGEXES that manifest string into a
# small set of legal typed-primitive SHAPES and rejects anything else — an opaque hand helper,
# an empty manifest, a weak label, or an unrecognized body/yield wrapper all go RED. It is the
# fail-closed structural evidence件 for C1: "constructed" is a checkable shape, not a hand wave.
#
# READS THE SCHEMA FROM `git show <ref>:schema/coverage-sixstate.v1.json` (default ref = HEAD),
# NOT the working tree — the working-tree schema may be mid-edit by a parallel line; C_construct
# and the constructed roster are pinned at the committed HEAD.
#
# Legal shapes (each an ordered-token predicate over the '+'-joined manifest):
#   flat_loop        typed_flat_block_dot_loop_body … standalone_reduce … <product core> …
#                    typed_flat_block_dot_loop_yield
#   super_block_loop typed_super_block_block_dot_loop_body … <non-wrapper core> …
#                    typed_super_block_block_dot_loop_yield
#   repack_gemv/gemm typed_repack_gem{v,m}_loop_body … <_dot core> … typed_repack_gem{v,m}_loop_yield
#   noperand_route   load|codebook_table_broadcast … standalone_reduce … store   (N-operand route,
#                    no loop wrapper)
# Global invariants (all shapes): E5 STRONG envelope, opaque_helper=false, manifest non-empty,
#   and NO opaque hand-helper token ([L-8]: no bare *_block_dot / opaque* / emit* token).
#
#   check_construction_manifest_regex.py             # audit HEAD's constructed roster
#   check_construction_manifest_regex.py --ref <rev> # audit a different committed snapshot
#   check_construction_manifest_regex.py --self-test # exercise the pure classifier

import json
import os
import re
import subprocess
import sys

SCHEMA_REL = "schema/coverage-sixstate.v1.json"

# The exact E5 strong envelope emitted by e5_strong_readout.py.
ENVELOPE_RE = re.compile(
    r"^E5-increment1-auto: constructed \(STRONG\); "
    r"realized-body manifest=(?P<manifest>[^;]+); "
    r"opaque_helper=(?P<opaque>true|false)$"
)

# [F-EMIT] front-door repack DUAL-manifest envelope (G3-lode-flat family). User
# ruling (2026-07-09): the [F-EMIT] front-door construction is a legitimate cert
# envelope — these gemm_tile cells are byte-exact-verified REAL front-door
# constructions (RVVLowerQuantContraction lowerToRepackGem{v,m}*, NOT the
# emit-bypass direct-emitter; the 4 active bypass entries are iq2_xxs/iq2_xs/iq2_s/
# mxfp4, none of these). One gemm_tile row carries BOTH the GEVM (decode) and GEMM
# (prefill) sub-manifests. Certification STRICTLY parses each sub-manifest and
# requires BOTH to be a legal repack shape (repack_gemv for GEVM, repack_gemm for
# GEMM) with opaque_helper=false and no opaque hand-helper token, so the
# envelope-consistency (both regimes present, both legal, no opaque monolith) is
# enforced — [F-EMIT] is NOT a rubber stamp.
FEMIT_ENVELOPE_RE = re.compile(
    r"^\[F-EMIT\][^:]*: constructed \(STRONG\); "
    r"realized-body manifest \(GEVM decode\)=(?P<gevm>[^;]+); "
    r"\(GEMM prefill\)=(?P<gemm>[^;]+); "
    r"(?P<tail>.*)$"
)
# Strip a trailing/inline `[…]` provenance annotation off a manifest token
# (e.g. `repack_lane_wise_q4_x_i8_dot[weight_nibble_unsigned]` → the bare op).
_ANNOTATION_RE = re.compile(r"\[[^\]]*\]")

# loop-body / loop-yield wrappers that are legal (typed pattern-library primitives, not helpers).
ALLOWED_WRAPPERS = {
    "typed_flat_block_dot_loop_body", "typed_flat_block_dot_loop_yield",
    "typed_super_block_block_dot_loop_body", "typed_super_block_block_dot_loop_yield",
    "typed_repack_gemv_loop_body", "typed_repack_gemv_loop_yield",
    "typed_repack_gemm_loop_body", "typed_repack_gemm_loop_yield",
}

REPACK_BODY_RE = re.compile(r"^typed_repack_(gemv|gemm)_loop_body$")
REPACK_YIELD_RE = re.compile(r"^typed_repack_(gemv|gemm)_loop_yield$")

# Compact fused-core dot-reduce primitives (MIRRORS e5_strong_readout.py
# _FUSED_DOT_REDUCE_RE): a SINGLE brick that carries BOTH the product AND the
# reduction, so a compact flat/super-block body needs no separate standalone_reduce
# + *_product pair. The super_block_loop shape already admits such a compact core;
# the flat_loop compact-core branch below uses this to admit the two compact FLAT
# cells (q1_0 binary_sign_core, nvfp4 codebook_core) whose front doors machine-
# derive a `body + <one fused core> + yield` manifest exactly like the super-block
# compact cells. Kept NARROW (a scale-only fp16 body carries none of these tokens),
# so the check stays discriminating.
FUSED_CORE_RE = re.compile(
    r"(scaled_dot|aux32_partial|integer_core|grid_core|codebook_core|ternary_core|"
    r"binary_sign_core|kquant_core|repack_lane_wise_q4_x_i8_dot|"
    r"repack_gemm_lane_wise_q4_x_i8_dot)")


# ------------------------------------------------------------ pure classifier
def is_opaque_helper_token(tok):
    """[L-8]: reject a bare *_block_dot hand helper or an opaque/emit helper token.

    The typed loop wrappers legitimately contain 'block_dot' (…block_dot_loop_body); every OTHER
    token carrying 'block_dot', or starting with 'opaque'/'emit', is an opaque hand helper.
    """
    if tok in ALLOWED_WRAPPERS:
        return False
    if tok.startswith("opaque") or tok.startswith("emit"):
        return True
    if "block_dot" in tok:
        return True
    return False


def classify_shape(tokens):
    """Return (shape_name, reason). shape_name is None when no legal shape matches."""
    first, last = tokens[0], tokens[-1]

    if first == "typed_flat_block_dot_loop_body" and last == "typed_flat_block_dot_loop_yield":
        # (a) DECOMPOSED flat form: an explicit standalone_reduce + a *_product core
        # token (q8_0 widening_product, q4_0 packed_i4_offset_binary_x_i8_product,
        # iq4_nl codebook_gather_x_i8_product, ...).
        if "standalone_reduce" in tokens and any(t.endswith("_product") for t in tokens):
            return "flat_loop", "typed flat block-dot loop body/yield (decomposed)"
        # (b) COMPACT fused-core flat form (mirrors the super_block_loop compact
        # core): a SINGLE fused dot-reduce brick carries BOTH the product and the
        # reduction (q1_0 q1_0_q8_0_binary_sign_core, nvfp4 nvfp4_q8_0_codebook_core),
        # so there is no separate standalone_reduce / *_product token — exactly as
        # the super-block compact cells (iq1_s grid_core, tq*_ternary_core, ...). The
        # core must be a real fused dot-reduce primitive (FUSED_CORE_RE), so a
        # scale-only or empty flat body is still rejected.
        core = [t for t in tokens if t not in ALLOWED_WRAPPERS]
        if any(FUSED_CORE_RE.search(t) for t in core):
            return "flat_loop", "typed flat block-dot loop body/yield (compact fused core)"
        if "standalone_reduce" not in tokens:
            return None, "flat_loop missing standalone_reduce"
        return None, "flat_loop missing a *_product core token (and no fused dot-reduce core)"

    if (first == "typed_super_block_block_dot_loop_body"
            and last == "typed_super_block_block_dot_loop_yield"):
        core = [t for t in tokens if t not in ALLOWED_WRAPPERS]
        if not core:
            return None, "super_block_loop empty core (body/yield only)"
        return "super_block_loop", "typed super-block block-dot loop body/yield"

    mb, my = REPACK_BODY_RE.match(first), REPACK_YIELD_RE.match(last)
    if mb and my and mb.group(1) == my.group(1):
        # The core brick is EITHER a lane-wise `*_dot` (q4_0/q4_1/q5_0/q5_1/q8_0) OR a
        # fused `*_core` repack brick (codebook_core iq4_nl/iq4_xs, ternary_core
        # tq1_0/tq2_0, kquant_core q2_K/q3_K/q4_K/q5_K/q6_K) — all constructed by the
        # SAME repack front door, all non-opaque fused dot-reduce primitives.
        core = [t for t in tokens if t not in ALLOWED_WRAPPERS]
        if not any(("_dot" in t) or FUSED_CORE_RE.search(t) for t in core):
            return None, f"repack_{mb.group(1)}_loop missing a *_dot / fused *_core token"
        return f"repack_{mb.group(1)}_loop", "typed repack GEM{V,M} loop body/yield"

    # N-operand product_reduce route: straight-line op list, no loop wrapper, terminates in store.
    if last == "store" and not any(t in ALLOWED_WRAPPERS for t in tokens):
        if "standalone_reduce" not in tokens:
            return None, "noperand_route missing standalone_reduce"
        if first not in ("load", "codebook_table_broadcast"):
            return None, f"noperand_route unexpected first token '{first}'"
        return "noperand_route", "N-operand product_reduce route (load…standalone_reduce…store)"

    return None, "no legal shape (unrecognized body/yield wrapper or terminal token)"


def classify_femit_repack(auto_readout):
    """Strict [F-EMIT] dual-manifest repack gate. Both the GEVM (decode) and GEMM
    (prefill) sub-manifests must classify as their respective legal repack shape."""
    m = FEMIT_ENVELOPE_RE.match(auto_readout)
    if not m:
        return False, "[F-EMIT] envelope mismatch (not the dual GEVM/GEMM repack form)", None
    if "opaque_helper=false" not in m.group("tail"):
        return False, "[F-EMIT] opaque_helper not declared false", None
    for label, want_shape, raw in (
        ("GEVM", "repack_gemv_loop", m.group("gevm")),
        ("GEMM", "repack_gemm_loop", m.group("gemm")),
    ):
        tokens = [_ANNOTATION_RE.sub("", t) for t in raw.split("+")]
        tokens = [t for t in tokens if t != ""]
        if not tokens:
            return False, f"[F-EMIT] {label} sub-manifest empty", None
        for t in tokens:
            if is_opaque_helper_token(t):
                return False, f"[F-EMIT] {label} opaque hand-helper token '{t}' ([L-8])", None
        shape, why = classify_shape(tokens)
        if shape != want_shape:
            return False, (f"[F-EMIT] {label} shape {shape or 'none'} != {want_shape} "
                           f"({why})"), None
    return True, "compliant ([F-EMIT] repack dual-manifest GEVM+GEMM)", "repack_femit_dual"


def classify_auto_readout(auto_readout):
    """Pure gate. Returns (ok: bool, reason: str, shape: str|None)."""
    if not isinstance(auto_readout, str):
        return False, f"auto_readout is {type(auto_readout).__name__}, not a string", None
    if auto_readout.startswith("[F-EMIT]"):
        return classify_femit_repack(auto_readout)
    m = ENVELOPE_RE.match(auto_readout)
    if not m:
        return False, "envelope mismatch (not the E5 STRONG-constructed form)", None
    if m.group("opaque") != "false":
        return False, "opaque_helper=true ([L-8] opaque hand helper)", None
    manifest = m.group("manifest")
    tokens = manifest.split("+")
    if not manifest or any(t == "" for t in tokens):
        return False, "empty/degenerate realized-body manifest ([L-8])", None
    for t in tokens:
        if is_opaque_helper_token(t):
            return False, f"opaque hand-helper token '{t}' ([L-8] no opaque *_block_dot)", None
    shape, why = classify_shape(tokens)
    if shape is None:
        return False, f"shape non-compliant: {why}", None
    return True, f"compliant (shape={shape})", shape


# ------------------------------------------------------------------ self-test
def self_test():
    ok_all = True
    P = "E5-increment1-auto: constructed (STRONG); realized-body manifest="
    S = "; opaque_helper=false"

    def strong(manifest, opaque="false"):
        return f"{P}{manifest}; opaque_helper={opaque}"

    flat = ("typed_flat_block_dot_loop_body+block_fp16_scale_product+load+load+widening_product+"
            "standalone_reduce+typed_flat_block_dot_loop_yield")
    superb = ("typed_super_block_block_dot_loop_body+iq1_s_q8_k_grid_core+"
              "typed_super_block_block_dot_loop_yield")
    gemv = ("typed_repack_gemv_loop_body+repack_lane_wise_q4_x_i8_dot+repack_dual_fp16_scale_fold+"
            "typed_repack_gemv_loop_yield")
    gemm = ("typed_repack_gemm_loop_body+repack_gemm_lane_wise_q4_x_i8_dot+"
            "repack_gemm_dual_fp16_scale_fold+typed_repack_gemm_loop_yield")
    nroute = "load+load+widening_product+standalone_reduce+dequantize+store"

    flat_compact_binary = ("typed_flat_block_dot_loop_body+q1_0_q8_0_binary_sign_core+"
                           "typed_flat_block_dot_loop_yield")
    flat_compact_codebook = ("typed_flat_block_dot_loop_body+nvfp4_q8_0_codebook_core+"
                             "typed_flat_block_dot_loop_yield")
    flat_bad_core = ("typed_flat_block_dot_loop_body+some_scale_only_brick+"
                     "typed_flat_block_dot_loop_yield")

    cases = [
        # (label, auto_readout, expected_ok)
        ("flat loop", strong(flat), True),
        ("flat loop compact fused core (q1_0 binary_sign_core)", strong(flat_compact_binary), True),
        ("flat loop compact fused core (nvfp4 codebook_core)", strong(flat_compact_codebook), True),
        ("flat loop compact non-fused core (no reduce, rejected)", strong(flat_bad_core), False),
        ("super-block grid loop", strong(superb), True),
        ("repack gemv loop", strong(gemv), True),
        ("repack gemm loop", strong(gemm), True),
        ("N-operand product_reduce route", strong(nroute), True),
        ("opaque_helper=true", strong(flat, opaque="true"), False),
        ("empty manifest", strong(""), False),
        ("opaque hand-helper token injected", strong(
            "typed_flat_block_dot_loop_body+emit_flat_block_dot+standalone_reduce+"
            "typed_flat_block_dot_loop_yield"), False),
        ("bare block_dot helper token", strong(
            "typed_flat_block_dot_loop_body+q4_0_block_dot+standalone_reduce+"
            "typed_flat_block_dot_loop_yield"), False),
        ("mismatched body/yield (flat body, super yield)", strong(
            "typed_flat_block_dot_loop_body+load+standalone_reduce+"
            "typed_super_block_block_dot_loop_yield"), False),
        ("flat missing standalone_reduce", strong(
            "typed_flat_block_dot_loop_body+widening_product+typed_flat_block_dot_loop_yield"),
         False),
        ("super-block body/yield only (empty core)", strong(
            "typed_super_block_block_dot_loop_body+typed_super_block_block_dot_loop_yield"), False),
        ("repack gemv body / gemm yield mismatch", strong(
            "typed_repack_gemv_loop_body+repack_lane_wise_q4_x_i8_dot+"
            "typed_repack_gemm_loop_yield"), False),
        ("unknown free-form shape", strong("foo+bar+baz"), False),
        ("weak label (constructed-weak envelope)", "pending-E5", False),
        ("dispatch-wired (empty auto_readout)", "", False),
        ("mxfp4 opaque block-dot negative control", strong(
            "opaque_mxfp4_block_dot_helper"), False),
        # [F-EMIT] dual-manifest repack (FIX-3): q4_1-shaped GEVM+GEMM, annotations stripped.
        ("[F-EMIT] repack dual-manifest (q4_1 shape)",
         "[F-EMIT] G3-lode-flat: constructed (STRONG); realized-body manifest "
         "(GEVM decode)=typed_repack_gemv_loop_body+repack_lane_wise_q4_x_i8_dot[weight_nibble_unsigned]"
         "+repack_dual_fp16_scale_fold[weight_min_byte_offset=32/activation_sum_byte_offset=2]"
         "+typed_repack_gemv_loop_yield; (GEMM prefill)=typed_repack_gemm_loop_body+"
         "repack_gemm_lane_wise_q4_x_i8_dot[weight_nibble_unsigned]+repack_gemm_dual_fp16_scale_fold"
         "[weight_min_byte_offset=32/activation_sum_byte_offset=8]+typed_repack_gemm_loop_yield; "
         "fold_model=lane_wise_vector_scale_min; opaque_helper=false. q4_1 notes...", True),
        ("[F-EMIT] repack missing GEMM regime (rejected)",
         "[F-EMIT] G3-lode-flat: constructed (STRONG); realized-body manifest "
         "(GEVM decode)=typed_repack_gemv_loop_body+repack_lane_wise_q4_x_i8_dot+"
         "typed_repack_gemv_loop_yield; fold_model=x; opaque_helper=false.", False),
        ("[F-EMIT] repack with opaque GEMM core (rejected)",
         "[F-EMIT] G3-lode-flat: constructed (STRONG); realized-body manifest "
         "(GEVM decode)=typed_repack_gemv_loop_body+repack_lane_wise_q4_x_i8_dot+"
         "typed_repack_gemv_loop_yield; (GEMM prefill)=typed_repack_gemm_loop_body+"
         "q4_0_block_dot+typed_repack_gemm_loop_yield; fold_model=x; opaque_helper=false.", False),
    ]
    print("-- F-1 construction-manifest shape classifier --")
    for label, ar, expect in cases:
        got, reason, _shape = classify_auto_readout(ar)
        mark = "PASS" if got == expect else "FAIL"
        ok_all = ok_all and got == expect
        print(f"  [{mark}] {label}: ok={got} (expect {expect}) : {reason}")

    print("SELF-TEST:", "GREEN" if ok_all else "RED")
    return 0 if ok_all else 1


# --------------------------------------------------------------------- runner
def repo_root():
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def load_schema_from_ref(root, ref):
    """Read schema/coverage-sixstate.v1.json from a committed ref (NOT the working tree)."""
    out = subprocess.run(
        ["git", "show", f"{ref}:{SCHEMA_REL}"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def main(argv):
    if "--self-test" in argv:
        return self_test()

    ref = "HEAD"
    if "--ref" in argv:
        ref = argv[argv.index("--ref") + 1]

    root = repo_root()
    doc = load_schema_from_ref(root, ref)
    states = doc.get("states", [])
    head = subprocess.run(["git", "rev-parse", "--short", ref], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()

    constructed = [s for s in states if s.get("state") == "constructed"]
    viol = []
    shapes = {}
    for s in constructed:
        ok, reason, shape = classify_auto_readout(s.get("auto_readout"))
        if not ok:
            viol.append((f"{s.get('op')}/{s.get('format')}", reason))
        else:
            shapes[shape] = shapes.get(shape, 0) + 1

    # secondary fail-closed cross-check: a NON-constructed state must NOT carry a
    # CERTIFIABLE envelope. Uses the full classify predicate (not just ENVELOPE_RE),
    # so it also catches a mislabeled row wearing an [F-EMIT] / repack-dual /
    # compact-flat manifest, not only the E5 form — a weak/dispatch row wearing any
    # certifiable manifest would otherwise slip C_construct.
    mislabel = []
    for s in states:
        if s.get("state") == "constructed":
            continue
        ok, _reason, _shape = classify_auto_readout(s.get("auto_readout"))
        if ok:
            mislabel.append((f"{s.get('op')}/{s.get('format')}",
                             f"state={s.get('state')} wears a certifiable envelope"))

    rc = 0
    if not viol:
        shape_summary = ", ".join(f"{k}={v}" for k, v in sorted(shapes.items()))
        print(f"OK: {len(constructed)} constructed cell(s) @ {ref} ({head}) carry a compliant "
              f"realized-body manifest shape ({shape_summary}).")
    else:
        rc = 1
        print(f"RED: {len(viol)}/{len(constructed)} constructed cell(s) @ {ref} ({head}) "
              "have a non-compliant realized-body manifest:")
        for cell, reason in viol:
            print(f"  ! {cell}  <-  {reason}")

    if mislabel:
        rc = 1
        print(f"RED: {len(mislabel)} non-constructed cell(s) wear a STRONG envelope (mislabel):")
        for cell, reason in mislabel:
            print(f"  ! {cell}  <-  {reason}")
    else:
        print("OK: no weak/dispatch-wired cell wears the STRONG envelope.")

    if rc:
        print("Fix: a 'constructed' (STRONG) cell must realize a typed pattern-library body "
              "(flat/super-block/repack loop or an N-operand product_reduce route) with "
              "opaque_helper=false and no opaque *_block_dot hand helper.")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
