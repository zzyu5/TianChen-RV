#!/usr/bin/env python3
"""Bootstrap CI for P1 #4/#6 iq4_nl decode M=1 GEVM. ratio=opp_med/ours_med (>1=ours faster).
Paired within-proc A/B/C => paired bootstrap over rep indices. 10k resamples, 95% pct CI."""
import re,sys,statistics,random
log=open(sys.argv[1]).read()
d={}
for tag in ('OURS','OPPX','OPPS'):
    for m in re.finditer(r'REPS_'+tag+r'\s+seed=(\S+) :([\d ]+)',log):
        d.setdefault(m.group(1),{})[tag]=[float(x) for x in m.group(2).split()]
random.seed(20260717)
def boot(o,p,n=10000):
    N=len(o);rs=[]
    for _ in range(n):
        i=[random.randrange(N) for _ in range(N)]
        rs.append(statistics.median([p[j] for j in i])/statistics.median([o[j] for j in i]))
    rs.sort();return rs[int(.025*n)],rs[int(.975*n)]
def riqr(v):
    s=sorted(v);n=len(s);m=s[n//2];return 100*(s[3*n//4]-s[n//4])/m
print(f"{'seed':8s} {'N':>3s} {'ours_med':>10s} {'oppX_med':>10s} {'ratio_X':>8s} {'CI95_X':>18s} {'oppS_med':>10s} {'ratio_S':>8s} {'CI95_S':>18s}")
rx=[];rs_=[]
for s,v in d.items():
    o,x,sv=v['OURS'],v['OPPX'],v['OPPS']
    om,xm,sm=statistics.median(o),statistics.median(x),statistics.median(sv)
    lx,hx=boot(o,x); ls,hs=boot(o,sv)
    rx.append(xm/om); rs_.append(sm/om)
    print(f"{s:8s} {len(o):3d} {om:10.0f} {xm:10.0f} {xm/om:8.4f} [{lx:6.4f},{hx:6.4f}] {sm:10.0f} {sm/om:8.4f} [{ls:6.4f},{hs:6.4f}]")
    print(f"{'':8s}     relIQR ours={riqr(o):.2f}% oppX={riqr(x):.2f}% oppS={riqr(sv):.2f}%")
print("\n=== PREREG §4 VERDICT (both seeds >=0.8 => PASS; any <0.8 => 具名-X) ===")
for nm,r in (('OPP-X (CROSSOP)',rx),('OPP-S (same-op)',rs_)):
    print(f"{nm:18s} seeds={[f'{v:.4f}' for v in r]} -> {'PASS[0.8]' if all(v>=0.8 for v in r) else '具名-X[0.8]'}")
