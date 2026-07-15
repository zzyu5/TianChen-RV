#!/usr/bin/env python3
import re, sys, collections
log = open(sys.argv[1]).read()
# CENSUS fmt=q2_K VLEN=256 K=1048576 qk=256 WBLK=84 hiters=8 reps=24 nf_ns=... | HOT ours_ns=.. opp_ns=.. ratio_hot=X | COLD ours_med_ns=.. o_iqr=.. opp_med_ns=.. p_iqr=.. ratio_cold_med=X ratio_cold_best=Y our_GBs=.. opp_GBs=.. | byte_mismatch=.. worst_ulp=..
pat = re.compile(r'CENSUS fmt=(\S+) VLEN=(\d+) K=(\d+) qk=(\d+) WBLK=(\d+).*?ratio_hot=([\d.]+) \| COLD ours_med_ns=([\d.]+) o_iqr=([\d.]+) opp_med_ns=([\d.]+) p_iqr=([\d.]+) ratio_cold_med=([\d.]+) ratio_cold_best=([\d.]+) our_GBs=([\d.]+) opp_GBs=([\d.]+) \| byte_mismatch=(-?\d+) worst_ulp=(-?\d+)')
rows = collections.defaultdict(list)
for m in pat.finditer(log):
    fmt=m.group(1)
    rows[fmt].append(dict(vlen=m.group(2),wblk=m.group(5),ratio_hot=float(m.group(6)),
        oq_med=float(m.group(7)),o_iqr=float(m.group(8)),pp_med=float(m.group(9)),p_iqr=float(m.group(10)),
        rcm=float(m.group(11)),rcb=float(m.group(12)),our_gbs=float(m.group(13)),opp_gbs=float(m.group(14)),
        mism=int(m.group(15)),ulp=int(m.group(16))))
order=["q2_K","q3_K","q4_K","q5_K","q6_K","iq1_s","iq1_m","iq2_xxs","iq2_xs","iq2_s","iq3_xxs","iq3_s","iq4_nl","iq4_xs","mxfp4","nvfp4","tq1_0","tq2_0"]
print(f"{'fmt':8} {'s1_cold':>8} {'s2_cold':>8} {'s1_iqr':>7} {'s2_iqr':>7} {'best1/2':>13} {'ourGBs':>7} {'oppGBs':>7} {'mism':>5} {'verdict':>10}")
npass=0; nx=0
for f in order:
    r=rows.get(f)
    if not r or len(r)<2:
        print(f"{f:8} MISSING ({len(r) if r else 0} seeds)"); continue
    s1,s2=r[0],r[1]
    cold_min=min(s1['rcm'],s2['rcm'])
    verdict="PASS" if cold_min>=0.8 else "namedX"
    if verdict=="PASS": npass+=1
    else: nx+=1
    print(f"{f:8} {s1['rcm']:8.4f} {s2['rcm']:8.4f} {s1['o_iqr']:7.2f} {s2['o_iqr']:7.2f} {s1['rcb']:.3f}/{s2['rcb']:.3f}   {s1['our_gbs']:7.2f} {s1['opp_gbs']:7.2f} {s1['mism']+s2['mism']:5} {verdict:>10}")
print(f"\nTALLY (>=0.8 both seeds=PASS): PASS={npass} namedX={nx}")
