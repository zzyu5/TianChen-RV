#!/usr/bin/env python3
# parse_msweep.py — parse M-sweep CENSUS logs (vecdot block-dot both boards + kquant repack-GEMM rvv)
# into one summary CSV + roofline signal. NO board contact; pure log parse.
import re, sys, glob, os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "board", "raw")

# plain-block weight bytes per format (for arithmetic-intensity / weight-GB/s)
WBLK = {"q4_0":18,"q4_1":20,"q5_0":22,"q5_1":24,"q8_0":34,
        "q2_K":84,"q3_K":110,"q4_K":144,"q5_K":176,"q6_K":210}
BLKK = {f:(32 if f in ("q4_0","q4_1","q5_0","q5_1","q8_0") else 256) for f in WBLK}

rows = []  # dict per (lane,board,fmt,M)

# --- vecdot CENSUS: "CENSUS fmt=q4_0 VLEN=128 K=2048 M=4 nc=512 hiters=8 reps=12 nf_ns=.. | HOT ours_ns=.. opp_ns=.. ratio_hot=.. | COLD ours_med_ns=.. o_iqr=.. opp_med_ns=.. p_iqr=.. ratio_cold_med=.. ratio_cold_best=.. | byte_mismatch=.. worst_ulp=.."
def parse_vecdot(path, board):
    if not os.path.exists(path): return
    for ln in open(path):
        if not ln.startswith("CENSUS"): continue
        g = dict(re.findall(r'(\w+)=(-?[\d.]+)', ln))
        fm = re.search(r'fmt=(\S+)', ln)
        if not fm: continue
        fmt = fm.group(1)
        M = int(float(g["M"])); K = int(float(g["K"])); nc = int(float(g["nc"]))
        om = float(g["ours_med_ns"]); pm = float(g["opp_med_ns"])
        macs = M*nc*K
        rows.append(dict(lane="blockdot", board=board, fmt=fmt, M=M, K=K, nc=nc,
            ours_ns=om, opp_ns=pm, ratio_cold_med=float(g["ratio_cold_med"]),
            ratio_cold_best=float(g.get("ratio_cold_best",0)), ratio_hot=float(g.get("ratio_hot",0)),
            ours_gmacs=macs/om, opp_gmacs=macs/pm,
            o_iqr=float(g.get("o_iqr",0)), p_iqr=float(g.get("p_iqr",0)),
            byte_mismatch=int(float(g.get("byte_mismatch",0)))))

# --- kquant CENSUS: "... K=.. nr=.. nc=.. ... COLD ours_med_ns=.. ours_iqrpct=.. opp_med_ns=.. opp_iqrpct=.. ours_gmacs=.. opp_gmacs=.. ours_wGBs=.. opp_wGBs=.. ratio_cold_med=.. ratio_cold_best=.."
def parse_kquant(path, board):
    if not os.path.exists(path): return
    for ln in open(path):
        if not ln.startswith("CENSUS"): continue
        fm = re.search(r'fmt=(\S+)', ln);
        if not fm: continue
        fmt = fm.group(1)
        nr = int(re.search(r'nr=(\d+)', ln).group(1))
        K = int(re.search(r'K=(\d+)', ln).group(1)); nc = int(re.search(r'nc=(\d+)', ln).group(1))
        om = float(re.search(r'COLD ours_med_ns=([\d.]+)', ln).group(1))
        pm = float(re.search(r'opp_med_ns=([\d.]+)', ln).group(1))
        og = float(re.search(r'ours_gmacs=([\d.]+) opp_gmacs', ln).group(1))
        pg = float(re.search(r'opp_gmacs=([\d.]+) ours_wGBs', ln).group(1))
        owg = float(re.search(r'ours_wGBs=([\d.]+)', ln).group(1))
        pwg = float(re.search(r'opp_wGBs=([\d.]+)', ln).group(1))
        rcm = float(re.search(r'ratio_cold_med=([\d.]+)', ln).group(1))
        rcb = float(re.search(r'ratio_cold_best=([\d.]+)', ln).group(1))
        oiq = float(re.search(r'ours_iqrpct=([\d.]+)', ln).group(1))
        piq = float(re.search(r'opp_iqrpct=([\d.]+)', ln).group(1))
        rows.append(dict(lane="repackGEMM", board=board, fmt=fmt, M=nr, K=K, nc=nc,
            ours_ns=om, opp_ns=pm, ratio_cold_med=rcm, ratio_cold_best=rcb, ratio_hot=0,
            ours_gmacs=og, opp_gmacs=pg, ours_wGBs=owg, opp_wGBs=pwg,
            o_iqr=oiq, p_iqr=piq, byte_mismatch=0))

# --- q4_K@k1 handbrick COLD: "COLD VLEN=256 K=2048 nr=32 nc=512 pool=8 rounds=12 ours_ns=.. ours_iqr=..% ours_gmacs=.. opp_ns=.. opp_iqr=..% opp_gmacs=.. ratio_ours_over_opp=.."
def parse_handbrick(path, board):
    if not os.path.exists(path): return
    for ln in open(path):
        if not ln.startswith("COLD"): continue
        nr = int(re.search(r'nr=(\d+)', ln).group(1))
        K = int(re.search(r'K=(\d+)', ln).group(1)); nc = int(re.search(r'nc=(\d+)', ln).group(1))
        om = float(re.search(r'ours_ns=([\d.]+)', ln).group(1)); pm = float(re.search(r'opp_ns=([\d.]+)', ln).group(1))
        og = float(re.search(r'ours_gmacs=([\d.]+)', ln).group(1)); pg = float(re.search(r'opp_gmacs=([\d.]+)', ln).group(1))
        rr = float(re.search(r'ratio_ours_over_opp=([\d.]+)', ln).group(1))
        oiq = float(re.search(r'ours_iqr=([\d.]+)', ln).group(1)); piq = float(re.search(r'opp_iqr=([\d.]+)', ln).group(1))
        rows.append(dict(lane="repackGEMM-vs-handbrick", board=board, fmt="q4_K", M=nr, K=K, nc=nc,
            ours_ns=om, opp_ns=pm, ratio_cold_med=rr, ratio_cold_best=rr, ratio_hot=0,
            ours_gmacs=og, opp_gmacs=pg, o_iqr=oiq, p_iqr=piq, byte_mismatch=0))

parse_vecdot(os.path.join(RAW,"rvv_vecdot_msweep.log"), "rvv")
parse_vecdot(os.path.join(RAW,"k1_vecdot_msweep.log"), "k1")
parse_kquant(os.path.join(RAW,"rvv_kquant_gemm_msweep.log"), "rvv")
parse_handbrick(os.path.join(RAW,"k1_q4k_handbrick_msweep.log"), "k1")

# write CSV
cols = ["lane","board","fmt","M","K","nc","ratio_cold_med","ratio_cold_best","ratio_hot",
        "ours_gmacs","opp_gmacs","ours_wGBs","opp_wGBs","o_iqr","p_iqr","ours_ns","opp_ns","byte_mismatch"]
out = os.path.join(HERE,"summary_msweep.csv")
with open(out,"w") as f:
    f.write(",".join(cols)+"\n")
    for r in sorted(rows, key=lambda r:(r["lane"],r["board"],r["fmt"],r["M"])):
        f.write(",".join(str(r.get(c,"")) for c in cols)+"\n")
print("wrote", out, "rows=", len(rows))

# --- roofline signal: per (lane,board,fmt) print ratio & ours_gmacs vs M ---
from collections import defaultdict
groups = defaultdict(list)
for r in rows: groups[(r["lane"],r["board"],r["fmt"])].append(r)
print("\n=== M-sweep: ratio_cold_med (ours/opp) and ours GMAC/s vs M ===")
print(f"{'lane':11} {'brd':4} {'fmt':5} | ratio@M and ours_gmacs@M (M asc)")
for k in sorted(groups):
    rs = sorted(groups[k], key=lambda r:r["M"])
    ratios = " ".join(f"M{r['M']}={r['ratio_cold_med']:.3f}" for r in rs)
    gm = " ".join(f"M{r['M']}={r['ours_gmacs']:.2f}" for r in rs)
    # roofline heuristic: ours_gmacs slope over M (flat => compute-bound)
    g0, g1 = rs[0]["ours_gmacs"], rs[-1]["ours_gmacs"]
    slope = (g1-g0)/g0*100 if g0 else 0
    r0, r1 = rs[0]["ratio_cold_med"], rs[-1]["ratio_cold_med"]
    print(f"{k[0]:11} {k[1]:4} {k[2]:5} | ratio {ratios}  Δratio={r1-r0:+.3f}")
    print(f"{'':11} {'':4} {'':5} | gmac  {gm}  Δgmac={slope:+.1f}%  ({'COMPUTE-bound(flat)' if abs(slope)<10 else 'gmac-rises→mem-slope'})")
