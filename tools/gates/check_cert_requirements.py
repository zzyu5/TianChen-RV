#!/usr/bin/env python3
"""check_cert_requirements.py -- G3-cert-hardening: the CERT THREE-REQUIREMENTS gate.

防复发入宪. A NUMERICAL-CORRECTNESS certificate (fold-numeric / kernel byte-exact /
repack-numeric) may assert claim='full' (byte-exact / 0-mismatch) ONLY if its declared
lineage satisfies all THREE requirements -- the ones the M2c 误诊 violated:

  ① CORPUS COMPLETE   -- the corpus EXERCISED every arithmetic term of the fold
                         (dmin!=0, mins!=0, non-degenerate scale, sign coverage ...);
                         a term merely declared-required but NOT exercised means the
                         cert must lower its claim to 'partial'. A cert may not narrow
                         away a fold's mandatory arithmetic terms from its 'required'
                         set (that is how the min-term hollow certs escaped: they never
                         listed dmin_nonzero among what they exercised).
  ② INPUT PATH SAME-ORIGIN -- the oracle's activation quantizer == the tested-dispatch
                         quantizer (mat-quant vs row-quant), AND the tested path is the
                         real production dispatch, link-by-link (dispatch_faithful). The
                         M2c 误诊 was a row-quant oracle against a mat-quant/interleaved
                         dispatch.
  ③ ORACLE INDEPENDENT -- the reference does NOT share the被测物's read implementation.
                         self-compare (PRE≡POST / untiled≡tiled / S1≡S6) proves an
                         invariant, not correctness -- both sides can carry the SAME
                         defect. shared-impl ('independent' harness reusing the被测's
                         bsums/fold read) is likewise HOLLOW. Only 'independent' backs a
                         claim=full.

The lineage of every such cert is declared in schema/cert-lineage.v1.json; this gate
reads it and fails CLOSED:
  * an entry missing a required field, or naming an unknown axis / claim / fold_model /
    corpus term / oracle-independence value -> RED (undeclared / typo guard);
  * corpus_terms.required not a superset of the fold's mandatory terms -> RED (UNDER-DECLARED);
  * claim='full' with any required term un-exercised -> RED (CORPUS-INCOMPLETE: mark partial);
  * claim='full' with oracle_quantizer != tested_quantizer -> RED (INPUT-PATH-NOT-SAME-ORIGIN);
  * claim='full' with dispatch_faithful=false -> RED (INPUT-PATH-NOT-DISPATCH-FAITHFUL);
  * claim='full' with oracle_independence != 'independent' -> RED (HOLLOW-CERT).
A 'partial' cert that HONESTLY declares its gaps, and a 'failure' (falsification) cert,
are GREEN -- but both must still declare the fold's full required-term set so the gap
stays visible.

This gate changes NO kernel and NO numerics; it is a prevention framework (rule + CI).
The full back-fill of the registry is done by the parallel accounting line; this gate
validates whatever is declared (it does NOT scan the tree for un-declared certs).

Usage:  python3 tools/gates/check_cert_requirements.py [--self-test] [-v]
Exit:   0 GREEN ; 1 RED (requirement / declaration violation) ; 2 setup error.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY = os.path.join(REPO, "schema/cert-lineage.v1.json")

_ENTRY_STR_FIELDS = ("id", "cert_path", "axis", "op_path", "fold_model", "claim")
_LINEAGE_STR_FIELDS = ("tested_quantizer", "oracle_quantizer", "weight_layout")


# ---------------------------------------------------------------------------
# Pure classifier core (fed synthetic inputs by --self-test; real files at runtime).
# ---------------------------------------------------------------------------
def validate_meta_shape(meta):
    """Structural validation of the registry $meta controlled vocabularies.
    Returns (errors, vocab) where vocab is the resolved lookup sets/maps."""
    errors = []
    axes = set(meta.get("axes") or [])
    claims = set(meta.get("claim_vocab") or [])
    oracle_vocab = set(meta.get("oracle_independence_vocab") or [])
    term_vocab = set(meta.get("corpus_term_vocab") or [])
    fold_req = meta.get("fold_model_required_terms")
    if not axes:
        errors.append("$meta.axes missing or empty")
    if not claims:
        errors.append("$meta.claim_vocab missing or empty")
    if not oracle_vocab:
        errors.append("$meta.oracle_independence_vocab missing or empty")
    if not term_vocab:
        errors.append("$meta.corpus_term_vocab missing or empty")
    if not isinstance(fold_req, dict) or not fold_req:
        errors.append("$meta.fold_model_required_terms missing or not a non-empty object")
        fold_req = {}
    else:
        # every mandatory term named by a fold must itself be in the corpus vocab.
        for fm, terms in fold_req.items():
            if not isinstance(terms, list):
                errors.append(f"$meta.fold_model_required_terms['{fm}'] must be a list")
                continue
            for t in terms:
                if t not in term_vocab:
                    errors.append(
                        f"$meta.fold_model_required_terms['{fm}'] names '{t}' which is "
                        f"not in corpus_term_vocab")
    vocab = {"axes": axes, "claims": claims, "oracle": oracle_vocab,
             "terms": term_vocab, "fold_req": fold_req}
    return errors, vocab


def _lineage_errors(tag, lin):
    """Shape-check input_path_lineage. Returns list of errors."""
    errs = []
    if not isinstance(lin, dict):
        return [f"{tag}: input_path_lineage missing or not an object"]
    for f in _LINEAGE_STR_FIELDS:
        v = lin.get(f)
        if not isinstance(v, str) or not v.strip():
            errs.append(f"{tag}: input_path_lineage.{f} missing or empty")
    if not isinstance(lin.get("dispatch_faithful"), bool):
        errs.append(f"{tag}: input_path_lineage.dispatch_faithful missing or not a bool")
    return errs


def evaluate_entry(e, vocab):
    """Verdict for one cert entry against the resolved vocab. Returns list of errors.
    Empty list == this entry is GREEN."""
    errors = []
    cid = e.get("id") if isinstance(e.get("id"), str) else "?"
    tag = f"cert[{cid}]"

    # --- shape: required string fields present + non-empty ---
    for f in _ENTRY_STR_FIELDS:
        v = e.get(f)
        if not isinstance(v, str) or not v.strip():
            errors.append(f"{tag}: missing/empty field '{f}' (UNDECLARED)")

    axis, claim, fold = e.get("axis"), e.get("claim"), e.get("fold_model")
    if axis is not None and axis not in vocab["axes"]:
        errors.append(f"{tag}: unknown axis '{axis}'")
    if claim is not None and claim not in vocab["claims"]:
        errors.append(f"{tag}: unknown claim '{claim}'")
    if fold is not None and fold not in vocab["fold_req"]:
        errors.append(f"{tag}: unknown fold_model '{fold}' (not in "
                      f"$meta.fold_model_required_terms)")

    oi = e.get("oracle_independence")
    if oi not in vocab["oracle"]:
        errors.append(f"{tag}: oracle_independence '{oi}' not in "
                      f"{sorted(vocab['oracle'])} (UNDECLARED)")

    ct = e.get("corpus_terms")
    required = exercised = None
    if not isinstance(ct, dict):
        errors.append(f"{tag}: corpus_terms missing or not an object")
    else:
        required = ct.get("required")
        exercised = ct.get("exercised")
        if not isinstance(required, list):
            errors.append(f"{tag}: corpus_terms.required missing or not a list")
            required = None
        if not isinstance(exercised, list):
            errors.append(f"{tag}: corpus_terms.exercised missing or not a list")
            exercised = None

    errors.extend(_lineage_errors(tag, e.get("input_path_lineage")))

    # If the shape is broken enough that the requirement checks can't run, stop here
    # (already RED). The requirement checks below need the vocab + lists resolved.
    if required is not None:
        req_set = set(required)
        for t in req_set:
            if t not in vocab["terms"]:
                errors.append(f"{tag}: corpus_terms.required names unknown term '{t}'")
        if exercised is not None:
            ex_set = set(exercised)
            for t in ex_set:
                if t not in vocab["terms"]:
                    errors.append(f"{tag}: corpus_terms.exercised names unknown term '{t}'")
                if t not in req_set:
                    errors.append(f"{tag}: corpus_terms.exercised term '{t}' is not in "
                                  f"'required' (exercised must be a subset of required)")

        # ① CORPUS COMPLETE -- (a) fold's mandatory terms must be DECLARED-required.
        if fold in vocab["fold_req"]:
            mandatory = set(vocab["fold_req"][fold])
            missing = mandatory - req_set
            if missing:
                errors.append(
                    f"{tag}: CORPUS-UNDER-DECLARED: fold '{fold}' requires arithmetic "
                    f"terms {sorted(mandatory)} but corpus_terms.required omits "
                    f"{sorted(missing)}. A cert may not narrow away a fold's mandatory "
                    f"terms (this is how the min-term hollow certs escaped).")

    # --- the strong three-requirement gate: enforced only for claim='full' ---
    if claim == "full" and exercised is not None and required is not None:
        # ① every required term must be EXERCISED, not merely declared.
        unexercised = set(required) - set(exercised)
        if unexercised:
            errors.append(
                f"{tag}: CORPUS-INCOMPLETE: claim='full' but required terms "
                f"{sorted(unexercised)} are NOT exercised. Lower the claim to 'partial' "
                f"and declare the gap, or extend the corpus.")

    if claim == "full":
        lin = e.get("input_path_lineage")
        if isinstance(lin, dict):
            tq, oq = lin.get("tested_quantizer"), lin.get("oracle_quantizer")
            if isinstance(tq, str) and isinstance(oq, str) and tq.strip() and oq.strip() \
                    and tq != oq:
                # ② INPUT PATH SAME-ORIGIN
                errors.append(
                    f"{tag}: INPUT-PATH-NOT-SAME-ORIGIN: claim='full' but oracle_quantizer "
                    f"'{oq}' != tested_quantizer '{tq}'. The reference and the被测 dispatch "
                    f"must share the SAME activation quantizer (the M2c 误诊 = row-quant "
                    f"oracle vs mat-quant dispatch).")
            if lin.get("dispatch_faithful") is False:
                errors.append(
                    f"{tag}: INPUT-PATH-NOT-DISPATCH-FAITHFUL: claim='full' but "
                    f"dispatch_faithful=false. A full numeric cert must run the被测物 "
                    f"through the REAL production dispatch, not an isolated harness.")
        # ③ ORACLE INDEPENDENT
        if oi in ("self-compare", "shared-impl"):
            errors.append(
                f"{tag}: HOLLOW-CERT: claim='full' but oracle_independence='{oi}'. "
                f"self-compare / shared-impl cannot back a full correctness claim -- both "
                f"sides can carry the SAME defect (M2c proved S1 and S6 were both wrong "
                f"yet self-compared 0-mismatch). Use an independent oracle or mark 'partial'.")

    return errors


def evaluate(meta, certs):
    """Core verdict over the whole registry. Returns (ok, errors)."""
    errors, vocab = validate_meta_shape(meta)
    if not isinstance(certs, list):
        errors.append("registry 'certs' missing or not a list")
        return (len(errors) == 0), errors
    seen_ids = set()
    for e in certs:
        if not isinstance(e, dict):
            errors.append("certs[]: entry is not an object")
            continue
        cid = e.get("id")
        if isinstance(cid, str):
            if cid in seen_ids:
                errors.append(f"cert[{cid}]: duplicate id")
            seen_ids.add(cid)
        errors.extend(evaluate_entry(e, vocab))
    return (len(errors) == 0), errors


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_real(verbose):
    if not os.path.isfile(REGISTRY):
        print(f"[setup-error] missing {REGISTRY}", file=sys.stderr)
        return 2
    try:
        doc = json.load(open(REGISTRY, encoding="utf-8"))
    except Exception as ex:  # noqa: BLE001
        print(f"[setup-error] {ex}", file=sys.stderr)
        return 2

    meta = doc.get("$meta") or {}
    certs = doc.get("certs")
    ok, errors = evaluate(meta, certs)

    if verbose or not ok:
        n = len(certs) if isinstance(certs, list) else 0
        print(f"[cert-requirements] cert-lineage registry: {n} declared cert(s)")
        if isinstance(certs, list):
            for e in certs:
                if isinstance(e, dict):
                    print(f"    {str(e.get('claim')):8s} {str(e.get('axis')):16s} "
                          f"{e.get('id')}")

    if ok:
        print("[cert-requirements] GREEN: every declared numeric-correctness cert satisfies "
              "the three requirements for its claim (corpus complete + input same-origin + "
              "oracle independent for claim=full; honest declaration for partial/failure).")
        return 0
    print("[cert-requirements] RED:", file=sys.stderr)
    for e in errors:
        print(f"  ::error:: {e}", file=sys.stderr)
    return 1


def run_self_test():
    """Prove the classifier DISCRIMINATES before it judges the committed registry:
    a compliant entry -> GREEN, each violation class -> RED."""
    fails = []

    def check(label, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            fails.append(label)

    meta = {
        "axes": ["fold-numeric", "kernel-byte-exact", "repack-numeric"],
        "claim_vocab": ["full", "partial", "failure"],
        "oracle_independence_vocab": ["independent", "self-compare", "shared-impl"],
        "corpus_term_vocab": ["dmin_nonzero", "mins_nonzero", "nondegenerate_scale",
                              "sign_coverage_signed", "scale_nonzero"],
        "fold_model_required_terms": {
            "kquant_dmin_bsums_min": ["dmin_nonzero", "mins_nonzero", "nondegenerate_scale"],
            "flat_scale": ["scale_nonzero"],
        },
    }

    def base(claim="full", **over):
        e = {
            "id": "T", "cert_path": "x/", "axis": "repack-numeric",
            "op_path": "q4_K repack GEMM", "fold_model": "kquant_dmin_bsums_min",
            "claim": claim,
            "corpus_terms": {
                "required": ["dmin_nonzero", "mins_nonzero", "nondegenerate_scale",
                             "sign_coverage_signed"],
                "exercised": ["dmin_nonzero", "mins_nonzero", "nondegenerate_scale",
                              "sign_coverage_signed"],
            },
            "input_path_lineage": {
                "tested_quantizer": "ggml_quantize_mat_q8_K_4x1",
                "oracle_quantizer": "ggml_quantize_mat_q8_K_4x1",
                "weight_layout": "interleaved-q8_Kx4", "dispatch_faithful": True,
            },
            "oracle_independence": "independent",
        }
        e.update(over)
        return e

    _, vocab = validate_meta_shape(meta)
    check("$meta shape valid (no meta errors)", not _)

    # --- a compliant claim=full entry -> GREEN ---
    check("compliant claim=full -> GREEN", evaluate_entry(base(), vocab) == [])

    # --- an honest partial (gaps declared, non-independent oracle) -> GREEN ---
    partial = base(claim="partial", oracle_independence="self-compare")
    partial["corpus_terms"]["exercised"] = ["nondegenerate_scale"]  # dmin/mins un-exercised
    partial["input_path_lineage"]["dispatch_faithful"] = False
    partial["input_path_lineage"]["oracle_quantizer"] = "quantize_row_q8_K"
    check("honest partial with declared gaps -> GREEN", evaluate_entry(partial, vocab) == [])

    # --- a failure (falsification) cert on a narrow corpus -> GREEN ---
    fail_cert = base(claim="failure")
    fail_cert["corpus_terms"]["exercised"] = ["dmin_nonzero", "mins_nonzero",
                                              "nondegenerate_scale"]
    check("failure cert -> GREEN (falsification, not hard-enforced)",
          evaluate_entry(fail_cert, vocab) == [])

    # --- ① corpus incomplete: claim=full but a required term un-exercised -> RED ---
    inc = base()
    inc["corpus_terms"]["exercised"] = ["mins_nonzero", "nondegenerate_scale",
                                        "sign_coverage_signed"]  # drop dmin_nonzero
    errs = evaluate_entry(inc, vocab)
    check("claim=full with un-exercised required term -> RED (CORPUS-INCOMPLETE)",
          any("CORPUS-INCOMPLETE" in e for e in errs))

    # --- ① under-declared: fold mandatory term omitted from required -> RED ---
    ud = base()
    ud["corpus_terms"]["required"] = ["nondegenerate_scale", "sign_coverage_signed"]
    ud["corpus_terms"]["exercised"] = ["nondegenerate_scale", "sign_coverage_signed"]
    errs = evaluate_entry(ud, vocab)
    check("fold mandatory term omitted from required -> RED (CORPUS-UNDER-DECLARED)",
          any("CORPUS-UNDER-DECLARED" in e for e in errs))

    # --- ② input path not same-origin: row-quant oracle vs mat-quant dispatch -> RED ---
    xo = base()
    xo["input_path_lineage"]["oracle_quantizer"] = "quantize_row_q8_K"
    errs = evaluate_entry(xo, vocab)
    check("claim=full row-quant oracle vs mat-quant dispatch -> RED (INPUT-PATH-NOT-SAME-ORIGIN)",
          any("INPUT-PATH-NOT-SAME-ORIGIN" in e for e in errs))

    # --- ② input path not dispatch-faithful -> RED ---
    nf = base()
    nf["input_path_lineage"]["dispatch_faithful"] = False
    errs = evaluate_entry(nf, vocab)
    check("claim=full but dispatch_faithful=false -> RED (INPUT-PATH-NOT-DISPATCH-FAITHFUL)",
          any("INPUT-PATH-NOT-DISPATCH-FAITHFUL" in e for e in errs))

    # --- ③ hollow: claim=full on a self-compare oracle -> RED ---
    hc = base(oracle_independence="self-compare")
    errs = evaluate_entry(hc, vocab)
    check("claim=full on self-compare oracle -> RED (HOLLOW-CERT)",
          any("HOLLOW-CERT" in e for e in errs))

    # --- ③ hollow: claim=full on a shared-impl 'independent' harness -> RED ---
    sh = base(oracle_independence="shared-impl")
    errs = evaluate_entry(sh, vocab)
    check("claim=full on shared-impl oracle -> RED (HOLLOW-CERT)",
          any("HOLLOW-CERT" in e for e in errs))

    # --- undeclared: missing a required field -> RED ---
    md = base()
    del md["oracle_independence"]
    errs = evaluate_entry(md, vocab)
    check("missing oracle_independence -> RED (UNDECLARED)",
          any("UNDECLARED" in e for e in errs))

    # --- typo guard: unknown corpus term -> RED ---
    tp = base()
    tp["corpus_terms"]["required"] = tp["corpus_terms"]["required"] + ["dmin_nonzeo"]
    errs = evaluate_entry(tp, vocab)
    check("unknown corpus term -> RED (typo guard)",
          any("unknown term" in e for e in errs))

    # --- unknown fold_model -> RED ---
    uf = base(fold_model="kquant_bogus_fold")
    errs = evaluate_entry(uf, vocab)
    check("unknown fold_model -> RED",
          any("unknown fold_model" in e for e in errs))

    # --- duplicate id at the registry level -> RED ---
    ok, errs = evaluate(meta, [base(), base()])
    check("duplicate cert id -> RED", (not ok) and any("duplicate id" in e for e in errs))

    # --- registry-level: a fully compliant single-entry registry -> GREEN ---
    ok, _ = evaluate(meta, [base(claim="failure",
                                 corpus_terms={"required": ["dmin_nonzero", "mins_nonzero",
                                                            "nondegenerate_scale"],
                                               "exercised": ["dmin_nonzero", "mins_nonzero",
                                                            "nondegenerate_scale"]})])
    check("compliant single-entry registry -> GREEN", ok)

    if fails:
        print(f"[cert-requirements --self-test] RED: {len(fails)} discrimination(s) failed")
        return 2
    print("[cert-requirements --self-test] GREEN: classifier discriminates all cases")
    return 0


def main(argv):
    verbose = "-v" in argv or "--verbose" in argv
    if "--self-test" in argv:
        return run_self_test()
    return run_real(verbose)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
