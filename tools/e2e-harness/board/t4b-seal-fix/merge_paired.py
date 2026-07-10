#!/usr/bin/env python3
# [G3 T4b seal-fix] Merge paired round-files -> one JSON per variant (concat samples_ns per
# (n_prompt,n_gen)); then print median/IQR/95%CI per variant/size + bootstrap ratio CI A/opp.
# usage: merge_paired.py <outdir>
import json, glob, sys, os, statistics as st, random
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tf_paired"
VARS = ["A", "Bstock", "Bq4kOFF"]

def load_variant(var):
    ents = {}  # (npr,ngen) -> {n_prompt,n_gen,samples_ns:[...]}
    for f in sorted(glob.glob(os.path.join(OUT, f"{var}_*_r*.json"))):
        if os.path.basename(f).startswith("merged_"): continue
        try: d = json.load(open(f))
        except Exception: continue
        for e in d:
            key = (e.get("n_prompt", 0) or 0, e.get("n_gen", 0) or 0)
            ents.setdefault(key, {"n_prompt": key[0], "n_gen": key[1], "samples_ns": []})
            ents[key]["samples_ns"] += [s for s in e.get("samples_ns", []) if s > 0]
    merged = list(ents.values())
    json.dump(merged, open(os.path.join(OUT, f"merged_{var}.json"), "w"))
    return merged

def ts_list(merged, npr, ngen):
    n = npr if ngen == 0 else ngen
    for e in merged:
        if (e["n_prompt"], e["n_gen"]) == (npr, ngen):
            return [n / (s / 1e9) for s in e["samples_ns"] if s > 0]
    return []

def summ(ts):
    ts = sorted(ts); n = len(ts)
    if n == 0: return None
    med = st.median(ts); q1 = ts[n//4]; q3 = ts[min(n-1,(3*n)//4)]
    mean = st.mean(ts); sd = st.pstdev(ts) if n > 1 else 0.0
    ci = 1.96*sd/(n**0.5) if n > 1 else 0.0
    return dict(med=med, q1=q1, q3=q3, mean=mean, ci=ci, n=n)

def boot_ratio_ci(a, b, iters=20000):
    if not a or not b: return None
    random.seed(1234)
    rs = []
    for _ in range(iters):
        ma = st.median(random.choices(a, k=len(a)))
        mb = st.median(random.choices(b, k=len(b)))
        if mb > 0: rs.append(ma/mb)
    rs.sort()
    lo = rs[int(0.025*len(rs))]; hi = rs[int(0.975*len(rs))]
    return st.median(a)/st.median(b), lo, hi

data = {v: load_variant(v) for v in VARS}
# gather all (npr,ngen) keys present
keys = sorted({(e["n_prompt"], e["n_gen"]) for v in VARS for e in data[v]})
for (npr, ngen) in keys:
    lbl = f"pp{npr}" if ngen == 0 else f"tg{ngen}"
    print(f"\n==== {lbl} ====")
    tsv = {}
    for v in VARS:
        ts = ts_list(data[v], npr, ngen); tsv[v] = ts
        s = summ(ts)
        if not s: print(f"  {v:9s}: (missing)"); continue
        print(f"  {v:9s}: median={s['med']:8.3f}  IQR[{s['q1']:.3f},{s['q3']:.3f}]  "
              f"mean={s['mean']:.3f}+/-{s['ci']:.3f}(95%CI)  n={s['n']} tok/s")
    for opp in ("Bstock", "Bq4kOFF"):
        r = boot_ratio_ci(tsv.get("A", []), tsv.get(opp, []))
        if r:
            pt, lo, hi = r
            cross = "  <== CI CROSSES 1.0" if (lo <= 1.0 <= hi) else ""
            print(f"  S(A/{opp}) = {pt:.3f}x  boot95%CI[{lo:.3f},{hi:.3f}]{cross}")
