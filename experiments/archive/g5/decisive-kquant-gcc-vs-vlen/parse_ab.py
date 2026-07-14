#!/usr/bin/env python3
# Parse llama-bench JSON A/B (ours=q4kON repack vs stock=q4kOFF block-dot).
# Splits prefill (n_gen==0) vs decode (n_prompt==0). Aggregates across passes.
import json, sys, glob, statistics as st

def load(files):
    pf=[]; dc=[]
    for f in sorted(files):
        try: data=json.load(open(f))
        except Exception as e: print(f"  !! {f}: {e}"); continue
        for e in data:
            ts=e.get("avg_ts"); npr=e.get("n_prompt",0); ngen=e.get("n_gen",0)
            if ts is None: continue
            if ngen==0 and npr>0: pf.append(ts)
            elif npr==0 and ngen>0: dc.append(ts)
    return pf,dc

def agg(v):
    if not v: return None
    med=st.median(v); mn=st.mean(v)
    iqr=(st.quantiles(v,n=4)[2]-st.quantiles(v,n=4)[0]) if len(v)>=4 else 0
    return med,mn,min(v),max(v),len(v),(iqr/med*100 if med else 0)

d=sys.argv[1] if len(sys.argv)>1 else "."
op,od = load(glob.glob(f"{d}/ours_p*.json"))
sp,sd = load(glob.glob(f"{d}/stock_p*.json"))
print(f"dir={d}")
for phase,ov,sv in [("prefill(pp)",op,sp),("decode(tg)",od,sd)]:
    ao=agg(ov); as_=agg(sv)
    if not ao or not as_: print(f"  {phase}: MISSING ours={bool(ov)} stock={bool(sv)}"); continue
    ratio=ao[0]/as_[0]
    print(f"  {phase}:")
    print(f"    ours (q4kON repack)   median={ao[0]:.4f} t/s mean={ao[1]:.4f} min={ao[2]:.4f} max={ao[3]:.4f} n={ao[4]} relIQR={ao[5]:.2f}%")
    print(f"    stock(q4kOFF blockdot) median={as_[0]:.4f} t/s mean={as_[1]:.4f} min={as_[2]:.4f} max={as_[3]:.4f} n={as_[4]} relIQR={as_[5]:.2f}%")
    verdict = ">=PARITY (WIN/tie)" if ratio>=0.98 else "LOSS"
    print(f"    ==> ratio ours/stock = {ratio:.4f}x   [{verdict}]")
