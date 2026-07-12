#!/usr/bin/env python3
# tools/lint/check_construction_manifest_regex.py — [C1-SHAPE] construction-manifest shape gate.
#
# C1 (合取存在性) fail-closed STRUCTURAL machine-check. For every `constructed` (== STRONG)
# cell in the six-state schema, the E5 `auto_readout` carries a MACHINE-DERIVED realized-body
# manifest (from .trellis/scripts/e5_strong_readout.py, which walks the actual realized
# weft_rvv body op-identity and applies [L-8]). This gate REGEXES that manifest string into a
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
# [IME-SEAL] board-sealed envelope (G4 IME family). User-authorized (G4 M1b relay):
# the IME q4_0_matmul_tile region is a REAL front-door construction on the common
# pipeline (weft.ime.q4_0_matmul_tile OWNS the typed region carrying the decomposed
# q4_0_dequant_core + vmadot_mac_leaf bricks + the q4_0_matmul_tile_yield terminator,
# read by OP-IDENTITY at emission -- IMEBackendEmissionDriver.cpp
# IMEQ40MatMulTileToEmitCFunc). It is NOT an RVV typed loop, so it carries its OWN
# envelope + shape rather than loosening the RVV side. Certification STRICTLY parses
# the manifest into the ime_matmul_tile shape (wrapper + BOTH decode/MAC bricks + the
# yield terminator, opaque_helper=false, no opaque token) AND requires the board_seal
# pointer to name the k1 int32 0-diff evidence + the objdump vmadot golden encoding --
# so an IME cert is a checkable BOARD-SEALED shape, not a hand wave (NOT a rubber stamp).
# board_seal is a free-form evidence blurb that may itself contain '; ' separators
# (e.g. the 2026-07-11 re-seal note "…store-once; re-sealed by 线甲b …"), so it is
# captured GREEDILY up to the FINAL '; opaque_helper=<bool>' at the end-anchor —
# NOT with [^;]+, which would stop at the first internal semicolon and spuriously
# reject the envelope. The manifest field stays [^;]+ ('+'-joined tokens, never a
# semicolon), and the seal-content checks (0xe210312b + 0-diff) below are unchanged,
# so this is a parse fix, not a loosening of the gate.
IME_ENVELOPE_RE = re.compile(
    r"^\[IME-SEAL\][^:]*: constructed \(STRONG\); "
    r"realized-body manifest=(?P<manifest>[^;]+); "
    r"board_seal=(?P<seal>.+); "
    r"opaque_helper=(?P<opaque>true|false)$"
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
    # CERT-FD首族 (FIX-5): the streaming dequantize_row loop wrappers.
    "typed_dequantize_row_loop_body", "typed_dequantize_row_loop_yield",
    # CERT-FD次族: the streaming quantize_row loop wrappers (f32->QUANT mirror).
    "typed_quantize_row_loop_body", "typed_quantize_row_loop_yield",
    # CERT-FD殿后族: the streaming forward-elementwise loop wrappers (the forward-pass
    # map/reduce/rotate sibling of the dequant/quant streams).
    "typed_elementwise_loop_body", "typed_elementwise_loop_yield",
}

# The 5 CONSTRUCTED forward-elementwise CORE bricks (the map/reduce/rotate primitive
# carried inside a typed_elementwise_loop_body). A forward operator is NEITHER a
# contraction NOR a plain dot, so it carries none of the flat/super/repack dot tokens;
# its decomposed evidence is ONE of these forward map/reduce/rotate bricks. Kept NARROW
# (a scale-only fp16 body carries none of them), so the check stays discriminating.
FORWARD_CORE_RE = re.compile(
    r"(elementwise_scale_map|elementwise_silu_map|"
    r"elementwise_rms_norm_reduce_core|elementwise_soft_max_reduce_core|"
    r"elementwise_rope_rotate_core|"
    # The 3 forward SUPPORT-op MAP core bricks (add/mul BINARY, cpy COPY, gelu
    # scalar) widened into the SAME elementwise_stream_loop shape.
    r"elementwise_binary_map|elementwise_copy_map|elementwise_gelu_map)")

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

    if (first == "typed_dequantize_row_loop_body"
            and last == "typed_dequantize_row_loop_yield"):
        # CERT-FD首族 streaming shape (FIX-5): the CONSTRUCTED dequantize_row family
        # (block_qX -> f32 row) is a PURE DECODE -- the region carries ONE per-block
        # dequantize_row_decode_core brick and stores straight through the output
        # pointer (NO product/reduce/accumulator, so NOT a flat/super/repack dot).
        # The core must be exactly the decode brick (kept NARROW: an empty body or a
        # stray non-decode core is rejected). The wrappers/core carry no opaque
        # *_block_dot token (is_opaque_helper_token already gated it), so a decode
        # front door is checkable, not a hand wave.
        core = [t for t in tokens if t not in ALLOWED_WRAPPERS]
        if not core:
            return None, "dequant_stream_loop empty core (body/yield only)"
        if "dequantize_row_decode_core" not in core:
            return None, "dequant_stream_loop missing dequantize_row_decode_core brick"
        return "dequant_stream_loop", "typed dequantize-row streaming loop body/yield"

    if (first == "typed_quantize_row_loop_body"
            and last == "typed_quantize_row_loop_yield"):
        # CERT-FD次族 streaming shape (the f32->QUANT MIRROR of dequant_stream_loop):
        # the CONSTRUCTED quantize_row family (f32 row -> block_qX) is a PURE ENCODE --
        # the region carries ONE per-block quantize_row_encode_core brick and stores
        # straight through the output byte pointer (NO product/reduce/accumulator, so
        # NOT a flat/super/repack dot). The core must be exactly the encode brick (kept
        # NARROW: an empty body or a stray non-encode core is rejected). The
        # wrappers/core carry no opaque *_block_dot token (is_opaque_helper_token
        # already gated it), so an encode front door is checkable, not a hand wave.
        core = [t for t in tokens if t not in ALLOWED_WRAPPERS]
        if not core:
            return None, "quant_stream_loop empty core (body/yield only)"
        if "quantize_row_encode_core" not in core:
            return None, "quant_stream_loop missing quantize_row_encode_core brick"
        return "quant_stream_loop", "typed quantize-row streaming loop body/yield"

    if (first == "typed_elementwise_loop_body"
            and last == "typed_elementwise_loop_yield"):
        # CERT-FD殿后族 streaming shape (the forward-pass sibling of the dequant/quant
        # streams): the 5 CONSTRUCTED forward operators (scale/silu = MAP,
        # rms_norm/soft_max = REDUCE, rope = ROTATE) realize a typed
        # weft_rvv.typed_elementwise_loop_body carrying ONE forward map/reduce/rotate
        # CORE brick + the yield. A forward operator is NEITHER a contraction NOR a
        # dot (no product/reduce/accumulator token), so it is NOT a flat/super/repack
        # dot; the core must be exactly one of the 5 forward bricks (FORWARD_CORE_RE,
        # kept NARROW: an empty body or a stray non-forward core is rejected). The
        # wrappers/core carry no opaque *_block_dot token (is_opaque_helper_token
        # already gated it), so a forward front door is checkable, not a hand wave.
        core = [t for t in tokens if t not in ALLOWED_WRAPPERS]
        if not core:
            return None, "elementwise_stream_loop empty core (body/yield only)"
        if not any(FORWARD_CORE_RE.search(t) for t in core):
            return None, ("elementwise_stream_loop missing a forward map/reduce/rotate "
                          "core brick")
        return "elementwise_stream_loop", "typed forward-elementwise streaming loop body/yield"

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

    # IME matmul-tile shape (G4/G4-M2/G4-M2b): a weft.ime.<fmt>_matmul_tile region
    # OWNS the body (first token) + is terminated by weft.ime.<fmt>_matmul_tile_yield
    # (last token), and carries its decomposed bricks by op-identity. The body-op
    # token and the yield token must name the SAME format prefix (kept NARROW).
    #
    # FLAT tiles (q4_0 offset-binary nibble / q8_0 direct int8): the single-decode
    # copy-adapt shape -- the format-keyed <fmt>_dequant_core decode AND the
    # vmadot_mac_leaf (the FOUNDATION int8->int32 MAC, reused across formats). Two
    # required bricks.
    IME_FLAT_TILE_FORMATS = ("q4_0", "q8_0")
    for fmt in IME_FLAT_TILE_FORMATS:
        if first == fmt + "_matmul_tile" and last == fmt + "_matmul_tile_yield":
            if fmt + "_dequant_core" not in tokens:
                return None, (f"ime_matmul_tile missing {fmt}_dequant_core "
                              "decode brick")
            if "vmadot_mac_leaf" not in tokens:
                return None, "ime_matmul_tile missing vmadot_mac_leaf MAC brick"
            return "ime_matmul_tile", f"typed IME {fmt} matmul-tile region body/yield"

    # SUPER-BLOCK K-quant tile (q4_K): the DEDICATED two-level-fold shape. The bare
    # vmadot nibble MAC alone is a HOLLOW q4_K representation (it drops the
    # per-sub-block scale weighting + the min bias that DEFINE q4_K), so the q4_K
    # manifest is REQUIRED to carry ALL of: the raw-nibble decode
    # (q4_K_dequant_core), the 6-bit scale/min unpack (q4_K_scale_min_unpack_core),
    # the vmadot MAC leaf, the scale-weighted accum (q4_K_scale_weighted_accum ->
    # S_scale), AND the min-bias accum (q4_K_min_bias_accum -> S_min). A q4_K IME
    # manifest missing the scale-weighted or min-bias accum is REJECTED (anti-hollow).
    if first == "q4_K_matmul_tile" and last == "q4_K_matmul_tile_yield":
        if "q4_K_dequant_core" not in tokens:
            return None, "ime_matmul_tile missing q4_K_dequant_core decode brick"
        if "q4_K_scale_min_unpack_core" not in tokens:
            return None, ("ime_matmul_tile missing q4_K_scale_min_unpack_core "
                          "6-bit unpack brick")
        if "vmadot_mac_leaf" not in tokens:
            return None, "ime_matmul_tile missing vmadot_mac_leaf MAC brick"
        if "q4_K_scale_weighted_accum" not in tokens:
            return None, ("ime_matmul_tile missing q4_K_scale_weighted_accum brick "
                          "(HOLLOW bare-MAC q4_K: no per-sub-block scale weighting)")
        if "q4_K_min_bias_accum" not in tokens:
            return None, ("ime_matmul_tile missing q4_K_min_bias_accum brick "
                          "(HOLLOW bare-MAC q4_K: no min bias)")
        return "ime_matmul_tile", "typed IME q4_K super-block matmul-tile region body/yield"

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


def classify_ime_seal(auto_readout):
    """Strict [IME-SEAL] board-sealed gate (G4). The manifest must classify as the
    ime_matmul_tile shape AND the board_seal pointer must name the k1 int32 0-diff
    evidence + the objdump vmadot golden encoding (0xe210312b) -- so the IME cert is
    tied to the real silicon seal, not just a structural claim."""
    m = IME_ENVELOPE_RE.match(auto_readout)
    if not m:
        return False, "[IME-SEAL] envelope mismatch (not the IME matmul-tile seal form)", None
    if m.group("opaque") != "false":
        return False, "[IME-SEAL] opaque_helper=true ([L-8] opaque hand helper)", None
    seal = m.group("seal")
    if "0xe210312b" not in seal or "0-diff" not in seal:
        return False, ("[IME-SEAL] board_seal missing the k1 int32 0-diff evidence / "
                       "vmadot 0xe210312b golden encoding"), None
    tokens = [_ANNOTATION_RE.sub("", t) for t in m.group("manifest").split("+")]
    tokens = [t for t in tokens if t != ""]
    if not tokens:
        return False, "[IME-SEAL] empty realized-body manifest", None
    for t in tokens:
        if is_opaque_helper_token(t):
            return False, f"[IME-SEAL] opaque hand-helper token '{t}' ([L-8])", None
    shape, why = classify_shape(tokens)
    if shape != "ime_matmul_tile":
        return False, f"[IME-SEAL] shape {shape or 'none'} != ime_matmul_tile ({why})", None
    return True, f"compliant ([IME-SEAL] {why})", "ime_matmul_tile"


def classify_auto_readout(auto_readout):
    """Pure gate. Returns (ok: bool, reason: str, shape: str|None)."""
    if not isinstance(auto_readout, str):
        return False, f"auto_readout is {type(auto_readout).__name__}, not a string", None
    if auto_readout.startswith("[F-EMIT]"):
        return classify_femit_repack(auto_readout)
    if auto_readout.startswith("[IME-SEAL]"):
        return classify_ime_seal(auto_readout)
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

    # CERT-FD首族 streaming dequantize_row shape (FIX-5).
    deq_stream = ("typed_dequantize_row_loop_body+dequantize_row_decode_core+"
                  "typed_dequantize_row_loop_yield")
    deq_stream_empty = ("typed_dequantize_row_loop_body+"
                        "typed_dequantize_row_loop_yield")
    deq_stream_nocore = ("typed_dequantize_row_loop_body+some_scale_only_brick+"
                         "typed_dequantize_row_loop_yield")

    # CERT-FD次族 streaming quantize_row shape (f32->QUANT mirror).
    qnt_stream = ("typed_quantize_row_loop_body+quantize_row_encode_core+"
                  "typed_quantize_row_loop_yield")
    qnt_stream_empty = ("typed_quantize_row_loop_body+"
                        "typed_quantize_row_loop_yield")
    qnt_stream_nocore = ("typed_quantize_row_loop_body+some_scale_only_brick+"
                         "typed_quantize_row_loop_yield")

    # CERT-FD殿后族 streaming forward-elementwise shapes (MAP / REDUCE / ROTATE).
    fwd_scale = ("typed_elementwise_loop_body+elementwise_scale_map+"
                 "typed_elementwise_loop_yield")
    fwd_silu = ("typed_elementwise_loop_body+elementwise_silu_map+"
                "typed_elementwise_loop_yield")
    fwd_rms = ("typed_elementwise_loop_body+elementwise_rms_norm_reduce_core+"
               "typed_elementwise_loop_yield")
    fwd_softmax = ("typed_elementwise_loop_body+elementwise_soft_max_reduce_core+"
                   "typed_elementwise_loop_yield")
    fwd_rope = ("typed_elementwise_loop_body+elementwise_rope_rotate_core+"
                "typed_elementwise_loop_yield")
    fwd_empty = "typed_elementwise_loop_body+typed_elementwise_loop_yield"
    fwd_nocore = ("typed_elementwise_loop_body+some_scale_only_brick+"
                  "typed_elementwise_loop_yield")

    # IME q4_0 matmul-tile board-sealed shape (G4 M1b).
    IME_MANIFEST = ("q4_0_matmul_tile+q4_0_dequant_core+vmadot_mac_leaf+"
                    "q4_0_matmul_tile_yield")
    IME_SEAL_OK = "k1 taskset -c 0-3 int32 0-diff 64/64 tiles, objdump vmadot=0xe210312b at leaf"

    def ime_seal(manifest=IME_MANIFEST, seal=IME_SEAL_OK, opaque="false"):
        return (f"[IME-SEAL] G4 M1b k1 silicon seal: constructed (STRONG); "
                f"realized-body manifest={manifest}; board_seal={seal}; "
                f"opaque_helper={opaque}")

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
        ("dequant-stream loop (CERT-FD首族)", strong(deq_stream), True),
        ("dequant-stream body/yield only (empty core, rejected)", strong(deq_stream_empty), False),
        ("dequant-stream missing decode_core (rejected)", strong(deq_stream_nocore), False),
        ("quant-stream loop (CERT-FD次族)", strong(qnt_stream), True),
        ("quant-stream body/yield only (empty core, rejected)", strong(qnt_stream_empty), False),
        ("quant-stream missing encode_core (rejected)", strong(qnt_stream_nocore), False),
        ("forward-stream MAP scale (CERT-FD殿后族)", strong(fwd_scale), True),
        ("forward-stream MAP silu", strong(fwd_silu), True),
        ("forward-stream REDUCE rms_norm", strong(fwd_rms), True),
        ("forward-stream REDUCE soft_max", strong(fwd_softmax), True),
        ("forward-stream ROTATE rope", strong(fwd_rope), True),
        ("forward-stream body/yield only (empty core, rejected)", strong(fwd_empty), False),
        ("forward-stream non-forward core (rejected)", strong(fwd_nocore), False),
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
        # [IME-SEAL] board-sealed IME q4_0 matmul-tile (G4 M1b).
        ("[IME-SEAL] q4_0 matmul-tile (board-sealed)", ime_seal(), True),
        # REGRESSION (2026-07-12): a board_seal blurb that itself contains an internal
        # '; ' separator (the re-seal note) must STILL parse — the seal is captured
        # greedily to the final '; opaque_helper=', not truncated at the first ';'.
        ("[IME-SEAL] q4_0 matmul-tile (seal w/ internal semicolon)",
         ime_seal(seal="k1 X60 batched vmadot leaf (single vsetvli + store-once; "
                       "re-sealed 2026-07-11 encoding UNCHANGED), int32 0-diff 64/64, "
                       "objdump vmadot=0xe210312b at leaf"), True),
        ("[IME-SEAL] missing decode brick (rejected)",
         ime_seal("q4_0_matmul_tile+vmadot_mac_leaf+q4_0_matmul_tile_yield"), False),
        ("[IME-SEAL] missing MAC brick (rejected)",
         ime_seal("q4_0_matmul_tile+q4_0_dequant_core+q4_0_matmul_tile_yield"), False),
        ("[IME-SEAL] body/yield-only, no bricks (rejected)",
         ime_seal("q4_0_matmul_tile+q4_0_matmul_tile_yield"), False),
        ("[IME-SEAL] board_seal missing 0diff/encoding (rejected)",
         ime_seal(seal="ran on k1, looked fine"), False),
        ("[IME-SEAL] opaque hand-helper token (rejected)",
         ime_seal("q4_0_matmul_tile+emit_ime_q4_0_body+vmadot_mac_leaf+"
                  "q4_0_matmul_tile_yield"), False),
        ("[IME-SEAL] opaque_helper=true (rejected)", ime_seal(opaque="true"), False),
        ("[IME-SEAL] wrong wrapper (RVV loop, rejected)",
         ime_seal("typed_flat_block_dot_loop_body+q4_0_dequant_core+vmadot_mac_leaf+"
                  "typed_flat_block_dot_loop_yield"), False),
        # [IME-SEAL] board-sealed IME q8_0 matmul-tile (G4 M2, the FLAT-int8 sibling).
        ("[IME-SEAL] q8_0 matmul-tile (board-sealed)",
         ime_seal("q8_0_matmul_tile+q8_0_dequant_core+vmadot_mac_leaf+"
                  "q8_0_matmul_tile_yield"), True),
        ("[IME-SEAL] q8_0 missing decode brick (rejected)",
         ime_seal("q8_0_matmul_tile+vmadot_mac_leaf+q8_0_matmul_tile_yield"), False),
        ("[IME-SEAL] q8_0 missing MAC brick (rejected)",
         ime_seal("q8_0_matmul_tile+q8_0_dequant_core+q8_0_matmul_tile_yield"), False),
        ("[IME-SEAL] q8_0 body/yield-only, no bricks (rejected)",
         ime_seal("q8_0_matmul_tile+q8_0_matmul_tile_yield"), False),
        ("[IME-SEAL] q8_0 mismatched format decode brick (rejected)",
         ime_seal("q8_0_matmul_tile+q4_0_dequant_core+vmadot_mac_leaf+"
                  "q8_0_matmul_tile_yield"), False),
        ("[IME-SEAL] q8_0 mismatched body/yield prefix (rejected)",
         ime_seal("q8_0_matmul_tile+q8_0_dequant_core+vmadot_mac_leaf+"
                  "q4_0_matmul_tile_yield"), False),
        # [IME-SEAL] board-sealed IME q4_K matmul-tile (G4 M2b, the SUPER-BLOCK
        # K-quant two-level-fold tile). The FULL six-brick manifest is admitted.
        ("[IME-SEAL] q4_K matmul-tile (board-sealed)",
         ime_seal("q4_K_matmul_tile+q4_K_dequant_core+q4_K_scale_min_unpack_core+"
                  "vmadot_mac_leaf+q4_K_scale_weighted_accum+q4_K_min_bias_accum+"
                  "q4_K_matmul_tile_yield"), True),
        # ANTI-HOLLOW: q4_K missing the scale-weighted accum (S_scale) is REJECTED
        # (the bare-nibble-MAC shape M2 refused -- no per-sub-block scale weighting).
        ("[IME-SEAL] q4_K HOLLOW missing scale-weighted accum (rejected)",
         ime_seal("q4_K_matmul_tile+q4_K_dequant_core+q4_K_scale_min_unpack_core+"
                  "vmadot_mac_leaf+q4_K_min_bias_accum+q4_K_matmul_tile_yield"),
         False),
        # ANTI-HOLLOW: q4_K missing the min-bias accum (S_min) is REJECTED.
        ("[IME-SEAL] q4_K HOLLOW missing min-bias accum (rejected)",
         ime_seal("q4_K_matmul_tile+q4_K_dequant_core+q4_K_scale_min_unpack_core+"
                  "vmadot_mac_leaf+q4_K_scale_weighted_accum+"
                  "q4_K_matmul_tile_yield"), False),
        # ANTI-HOLLOW: q4_K bare-MAC only (the copy-adapt flat shape) is REJECTED --
        # this is EXACTLY the hollow representation M2 refused for q4_K.
        ("[IME-SEAL] q4_K bare-nibble-MAC only (hollow; rejected)",
         ime_seal("q4_K_matmul_tile+q4_K_dequant_core+vmadot_mac_leaf+"
                  "q4_K_matmul_tile_yield"), False),
        # q4_K missing the 6-bit scale/min unpack is REJECTED.
        ("[IME-SEAL] q4_K missing scale/min unpack (rejected)",
         ime_seal("q4_K_matmul_tile+q4_K_dequant_core+vmadot_mac_leaf+"
                  "q4_K_scale_weighted_accum+q4_K_min_bias_accum+"
                  "q4_K_matmul_tile_yield"), False),
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
