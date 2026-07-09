#!/usr/bin/env python3
# [G3 T4b seal-fix] median + IQR + 95% CI (tok/s) from llama-bench JSON samples_ns.
# usage: analyze_bench.py <label>=<json> [<label>=<json> ...]
import json, sys, statistics as st

def rows(path):
    d = json.load(open(path))
    out = []
    for e in d:
        npr = e.get("n_prompt", 0) or 0
        ngen = e.get("n_gen", 0) or 0
        kind = "pp" if (npr > 0 and ngen == 0) else ("tg" if ngen > 0 else "?")
        n = npr if kind == "pp" else ngen
        # per-rep tok/s from samples_ns (ns for n tokens)
        ts = [n / (s / 1e9) for s in e.get("samples_ns", []) if s > 0]
        out.append((f"{kind}{n}", ts))
    return out

data = {}  # label -> {name: [ts...]}
for arg in sys.argv[1:]:
    lab, path = arg.split("=", 1)
    data[lab] = dict(rows(path))

def summ(ts):
    ts = sorted(ts); n = len(ts)
    med = st.median(ts)
    q1 = ts[n//4]; q3 = ts[(3*n)//4]
    mean = st.mean(ts); sd = st.pstdev(ts) if n>1 else 0.0
    ci = 1.96*sd/(n**0.5) if n>1 else 0.0
    return med, q1, q3, mean, ci, n

names = sorted({k for v in data.values() for k in v})
labs = list(data)
for name in names:
    print(f"== {name} ==")
    med = {}
    for lab in labs:
        ts = data[lab].get(name)
        if not ts:
            print(f"  {lab:10s}: (missing)"); continue
        m,q1,q3,mean,ci,n = summ(ts)
        med[lab] = m
        print(f"  {lab:10s}: median={m:8.3f}  IQR[{q1:.3f},{q3:.3f}]  mean={mean:.3f}±{ci:.3f}(95%CI) n={n} tok/s")
    if len(med) >= 2:
        base = labs[-1] if labs[-1] in med else list(med)[-1]
        for lab in labs[:-1] if labs[-1] in med else list(med)[:-1]:
            if lab in med and med[base]:
                print(f"  S({lab}/{base}) = {med[lab]/med[base]:.3f}x  (>1 => {lab} faster)")
