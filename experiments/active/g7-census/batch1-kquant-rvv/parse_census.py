#!/usr/bin/env python3
# parse_census.py — turn raw CENSUS lines into a CSV + markdown table.
import re, sys
raw = open(sys.argv[1]).read()

def kvseg(seg):  # parse "k=v k=v" of ONE segment
    return dict(re.findall(r"(\w+)=(-?[\d.]+)", seg))

rows = []
for ln in raw.splitlines():
    if not ln.startswith("CENSUS"):
        continue
    segs = ln.split("|")
    head = kvseg(segs[0])          # fmt,VLEN,K,nr,nc,hiters,reps,nf_ns
    hot  = kvseg(segs[1])          # HOT: ours_ns,opp_ns,ours_gmacs,opp_gmacs,ratio_hot
    cold = kvseg(segs[2])          # COLD: ours_med_ns,ours_iqrpct,opp_med_ns,opp_iqrpct,ours_gmacs,opp_gmacs,ours_wGBs,opp_wGBs,ratio_cold_med,ratio_cold_best
    fmt = re.search(r"fmt=(\S+)", segs[0]).group(1)
    rows.append(dict(fmt=fmt, nr=head["nr"], K=head["K"], nc=head["nc"], nf=head["nf_ns"],
                     rhot=hot["ratio_hot"], ohg=hot["ours_gmacs"], phg=hot["opp_gmacs"],
                     rcm=cold["ratio_cold_med"], rcb=cold["ratio_cold_best"],
                     oiqr=cold["ours_iqrpct"], piqr=cold["opp_iqrpct"],
                     owgb=cold["ours_wGBs"], pwgb=cold["opp_wGBs"]))

# CSV
cols = ["fmt","nr","K","nc","rhot","rcm","rcb","ohg","phg","oiqr","piqr","owgb","pwgb","nf"]
print(",".join(cols))
for r in rows:
    print(",".join(str(r[c]) for c in cols))

# opponent machine-probe
opp = {}
for ln in raw.splitlines():
    m = re.search(r"# OPP_(q\d_K): main=(\S+) vl128=(\S+) unused_repack_gemm=(\S+)", ln)
    if m: opp[m.group(1)] = (m.group(2), m.group(3), m.group(4))

print("\n## census (all nr) — ours/opp ratio, gcc-15.2 symmetric, VLEN128, core8\n")
print("| fmt@rvv | nr(M) | hot | cold_med | cold_best | ours_hot GMAC/s | opp_hot GMAC/s | cold iqr% o/opp | verdict |")
print("|---|--:|--:|--:|--:|--:|--:|--:|---|")
for r in rows:
    rh=float(r["rhot"]); rc=float(r["rcm"])
    v = "LOSS" if max(rh,rc)<0.95 else ("~parity" if max(rh,rc)<1.05 else "WIN")
    print(f"| {r['fmt']} | {r['nr']} | {rh:.3f}× | {rc:.3f}× | {float(r['rcb']):.3f}× | {r['ohg']} | {r['phg']} | {r['oiqr']}/{r['piqr']} | {v} |")

print("\n## opponent machine-probe (VLEN128 dispatched = block-dot; repack-gemm VLEN256-gated-off)\n")
for k,v in opp.items():
    print(f"- {k}: block-dot main@{v[0]} vl128={v[1]} | unused_repack_gemm={v[2]}")
