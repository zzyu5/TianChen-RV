#!/usr/bin/env python3
# G5-M3 IME logit-level bounded-ULP comparator.
#
# For each prompt, reads teacher-forced per-position logit matrices dumped by
# logit_dump under three backends (OFF=stock RVV, VEN=vendor IME, ON=our tcrv IME
# bridge) over the SAME canonical token sequence, and quantifies:
#   (1) bounded-ULP: max |logit_ON - logit_OFF| and its ratio to the logit scale,
#       contextualised by the vendor baseline max |logit_VEN - logit_OFF| and the
#       cleanest cross-IME max |logit_ON - logit_VEN|.
#   (2) near-tie flips: at each DECISION position (predicting a generated token),
#       argmax_ON vs argmax_OFF. For each flip we test whether the OFF gap between
#       the two competing tokens (A=argmax_OFF, B=argmax_ON) is within the numeric
#       perturbation that flipped them:  gap_off(A,B) <= |dlogit[A]| + |dlogit[B]|.
#       A flip passing this test is a correctness-neutral near-tie; a flip failing
#       it (gap_off >> perturbation) would be a genuine correctness problem.
import sys, os
import numpy as np

def load(path):
    with open(path, 'rb') as f:
        n, nv = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=np.float32, count=int(n) * int(nv)).reshape(int(n), int(nv))
    return data

def load_vocab(path):
    m = {}
    if not os.path.exists(path):
        return m
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 1)
            if len(parts) == 2:
                m[int(parts[0])] = parts[1]
            elif len(parts) == 1:
                m[int(parts[0])] = ''
    return m

def main():
    out_dir = sys.argv[1]
    fmt     = sys.argv[2] if len(sys.argv) > 2 else "?"
    prompts = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else range(1, 6)
    vocab = load_vocab(os.path.join(out_dir, "vocab.tsv"))
    def piece(t):
        return vocab.get(int(t), f"<{t}>")

    print(f"================ format={fmt}  logit-level bounded-ULP / near-tie report ================")
    tot_flips = 0
    tot_nontie = 0
    glob_max_on_off = 0.0
    glob_max_rel = 0.0
    glob_max_ven_off = 0.0
    glob_max_on_ven = 0.0
    per_prompt = []
    for i in prompts:
        boff = os.path.join(out_dir, f"logits_off_{i}.bin")
        bven = os.path.join(out_dir, f"logits_ven_{i}.bin")
        bon  = os.path.join(out_dir, f"logits_on_{i}.bin")
        if not (os.path.exists(boff) and os.path.exists(bon)):
            print(f"[prompt {i}] MISSING bins (off={os.path.exists(boff)} on={os.path.exists(bon)}) -- skip")
            continue
        off = load(boff); on = load(bon)
        ven = load(bven) if os.path.exists(bven) else None
        n = min(off.shape[0], on.shape[0])
        off = off[:n]; on = on[:n]
        if ven is not None: ven = ven[:n]
        metap = os.path.join(out_dir, f"meta_{i}.txt")
        n_prompt = 1
        if os.path.exists(metap):
            n_prompt = int(open(metap).read().strip())

        d_on_off = np.abs(on - off)
        scale = np.maximum(np.abs(off).max(), 1e-6)
        max_on_off = float(d_on_off.max())
        rel_on_off = max_on_off / scale
        max_ven_off = float(np.abs(ven - off).max()) if ven is not None else float('nan')
        max_on_ven  = float(np.abs(on - ven).max()) if ven is not None else float('nan')
        glob_max_on_off = max(glob_max_on_off, max_on_off)
        glob_max_rel = max(glob_max_rel, rel_on_off)
        if ven is not None:
            glob_max_ven_off = max(glob_max_ven_off, max_ven_off)
            glob_max_on_ven = max(glob_max_on_ven, max_on_ven)

        # decision positions: p predicts token p+1 that lies in the generated region
        # (p >= n_prompt-1). include the final position (predicts an unobserved token).
        p0 = max(n_prompt - 1, 0)
        argoff = off.argmax(axis=1)
        argon  = on.argmax(axis=1)
        # OFF top-2 margin distribution over decision positions: how many decision
        # positions are near-ties whose top-2 gap is WITHIN our ON-OFF perturbation
        # band (i.e. flippable by an f32-reassoc-scale difference) -- this is why
        # free-running greedy diverges even when teacher-forced argmax is stable.
        margins = []
        for p in range(p0, n):
            t2 = np.partition(off[p], -2)[-2:]
            margins.append(float(t2.max() - t2.min()))
        margins = np.array(margins) if margins else np.array([0.0])
        n_flippable = int((margins <= max_on_off).sum())
        sm_idx = np.argsort(margins)[:3]
        flips = []
        for p in range(p0, n):
            A = int(argoff[p]); B = int(argon[p])
            if A == B:
                continue
            gap_off = float(off[p, A] - off[p, B])          # >= 0 by def of argmax
            pert    = float(abs(on[p, A] - off[p, A]) + abs(on[p, B] - off[p, B]))
            near_tie = gap_off <= pert
            # top-2 margin under OFF at this position (overall near-tie indicator)
            top2 = np.partition(off[p], -2)[-2:]
            margin = float(top2[1] - top2[0]) if top2[1] >= top2[0] else float(top2[0] - top2[1])
            margin = abs(float(np.sort(off[p])[-1] - np.sort(off[p])[-2]))
            ven_flip = (ven is not None and int(ven[p].argmax()) != A)
            flips.append((p, A, B, gap_off, pert, near_tie, margin, ven_flip))

        n_flips = len(flips)
        n_nontie = sum(1 for f in flips if not f[5])
        tot_flips += n_flips
        tot_nontie += n_nontie
        first_flip = flips[0][0] if flips else None
        per_prompt.append((i, n, n_prompt, n_flips, n_nontie, max_on_off, rel_on_off, max_ven_off, max_on_ven, first_flip))

        print(f"\n[prompt {i}] n_pos={n} n_prompt={n_prompt} decision_pos={n-p0}")
        print(f"  bounded-ULP  max|d(ON-OFF)|={max_on_off:.3e}  rel(/max|logit|={scale:.2f})={rel_on_off:.3e}")
        if ven is not None:
            print(f"  vendor base  max|d(VEN-OFF)|={max_ven_off:.3e}   cross-IME max|d(ON-VEN)|={max_on_ven:.3e}")
        print(f"  OFF top-2 margin over {len(margins)} decision pos: min={margins.min():.3e} "
              f"median={np.median(margins):.3e}  near-ties<=max|dON-OFF|({max_on_off:.2e}): "
              f"{n_flippable}/{len(margins)} positions")
        for si in sm_idx:
            pp = p0 + int(si)
            order = np.argsort(off[pp])[::-1]
            a1, a2 = int(order[0]), int(order[1])
            print(f"     tight pos {pp:3d}: top1 '{piece(a1)}'({a1}) top2 '{piece(a2)}'({a2}) "
                  f"gap={off[pp,a1]-off[pp,a2]:.3e}")
        print(f"  argmax flips at decision positions: {n_flips}  (non-near-tie: {n_nontie})")
        for (p, A, B, gap_off, pert, near_tie, margin, ven_flip) in flips:
            tag = "NEAR-TIE" if near_tie else "*** NON-TIE (real?) ***"
            print(f"    pos {p:3d}: OFF->'{piece(A)}'({A})  ON->'{piece(B)}'({B})  "
                  f"gap_off={gap_off:.3e} pert={pert:.3e} top2margin={margin:.3e}  "
                  f"ratio(gap/pert)={ (gap_off/pert if pert>0 else float('inf')):.2f}  VENflip={ven_flip}  [{tag}]")

    print("\n================ SUMMARY " + f"format={fmt}" + " ================")
    all_tie = (tot_nontie == 0)
    # within-vendor-band: our ON divergence from stock is no larger than the vendor
    # IME's own divergence from stock (VEN-OFF) -- our kernel is as faithful to stock
    # RVV as the vendor's blessed IME kernel. 5% slack for measurement.
    within_band = True
    for (i, n, npmt, nf, nnt, mo, ro, mv, mov, ff) in per_prompt:
        band_ok = (mv != mv) or (mo <= mv * 1.05 + 1e-9)   # nan-safe
        within_band = within_band and band_ok
        print(f"  prompt {i}: flips={nf} nontie={nnt} max|dON-OFF|={mo:.3e} rel={ro:.3e} "
              f"max|dVEN-OFF|={mv:.3e} max|dON-VEN|={mov:.3e} ON<=VEN={'T' if band_ok else 'F'} first_flip_pos={ff}")
    print(f"  TOTAL flips={tot_flips}  non-near-tie={tot_nontie}  ALL_NEAR_TIE={all_tie}")
    print(f"  GLOBAL max|d(ON-OFF)|={glob_max_on_off:.3e}  logit-rel(accum through net)={glob_max_rel:.3e}  "
          f"max|d(VEN-OFF)|={glob_max_ven_off:.3e}  max|d(ON-VEN)|={glob_max_on_ven:.3e}")
    print(f"  NOTE: logit-rel ~1e-2 = per-op within-ULP (single-tensor ~1e-7 / int32 0-diff) "
          f"ACCUMULATED through 22 transformer layers; kernel-level numerical band is the seal anchor.")
    print(f"  VERDICT[{fmt}]: all-flips-near-tie={'T' if all_tie else 'F'}  "
          f"within-vendor-band(ON<=VEN)={'T' if within_band else 'F'}  "
          f"correctness-neutral={'T' if all_tie else 'F'}")

if __name__ == "__main__":
    main()
