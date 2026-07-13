#!/usr/bin/env python3
# Aggregate a run-m6-q40-board.sh run.log: compute-vs-feed split + B-feed-locality interchange.
# Profile configs (ns_matmul = parallel critical path): onepivec_full (M5 mi-outer vec) /
# onepivec_noepi (mi-outer vmadot+feed) / compute_l1 (vmadot compute only) / onjout_full
# (M6 nj-outer vec) / onjout_noepi (nj-outer vmadot+feed). Perf sides: off/ven/onepivec/onjout.
# Usage: agg_m6.py run.log
import sys, json, re, statistics as st

def med_iqr(xs):
    xs = sorted(xs); n = len(xs)
    if n == 0: return (float('nan'),)*4 + (0,)
    med = st.median(xs)
    def pct(p):
        k=p*(n-1); lo=int(k); hi=min(lo+1,n-1); return xs[lo]+(xs[hi]-xs[lo])*(k-lo)
    q1,q3 = pct(0.25), pct(0.75)
    reliqr = (q3-q1)/med if med else float('nan')
    return (med,q1,q3,reliqr,n)

def test_label(o):
    npp=o.get("n_prompt",0); ng=o.get("n_gen",0)
    if ng and not npp: return f"tg{ng}"
    if npp and not ng: return f"pp{npp}"
    return f"pp{npp}_tg{ng}"

def main():
    log=open(sys.argv[1],encoding="utf-8",errors="replace").read()

    # ---- compute-vs-feed + interchange decomposition (X-0) ----
    print("=== [X-0] COMPUTE-vs-FEED + B-locality interchange (4-hart; ns_matmul = parallel crit path) ===")
    prof={}
    for m in re.finditer(r'\[TCRV-Q40-PROF\] (.+)', log):
        d=dict(re.findall(r'(\w+)=([0-9.]+)', m.group(1)))
        l1=d.get('mmprof_l1','0'); ne=d.get('mmprof_noepi','0'); nj=d.get('njouter','0'); ev=d.get('epivec','0')
        if   l1=='1':               side='compute_l1'
        elif ne=='1' and nj=='1':   side='onjout_noepi'
        elif ne=='1':               side='onepivec_noepi'
        elif nj=='1' and ev=='1':   side='onjout_full'
        elif ev=='1':               side='onepivec_full'
        else:                       side='other'
        prof[side]=d
        def g(k): return float(d.get(k,0))/1e9
        print(f"  {side:>14}: ns_matmul={g('ns_matmul'):8.3f}s  ns_quant={g('ns_quant'):6.3f}s  "
              f"ns_copyback={g('ns_copyback'):6.3f}s  (calls={d.get('calls','-')})")
    def gm(side): return float(prof.get(side,{}).get('ns_matmul',0))/1e9
    m5=gm('onepivec_full'); noepi_mi=gm('onepivec_noepi'); comp=gm('compute_l1')
    m6=gm('onjout_full'); noepi_nj=gm('onjout_noepi')
    print()
    if noepi_mi and comp:
        feed_mi = noepi_mi - comp
        print(f"  vmadot COMPUTE (fixed L1 scratch)      = {comp:.3f}s")
        print(f"  noepi mi-outer (vmadot + real feed)    = {noepi_mi:.3f}s")
        print(f"  -> FEED share (mi-outer) = noepi_mi - compute = {feed_mi:.3f}s  ({100*feed_mi/noepi_mi:.1f}% of vmadot+feed; "
              f"compute {100*comp/noepi_mi:.1f}%)")
    if noepi_nj and comp:
        feed_nj = noepi_nj - comp
        print(f"  noepi nj-outer (vmadot + real feed·B-locality) = {noepi_nj:.3f}s  -> FEED share = {feed_nj:.3f}s ({100*feed_nj/noepi_nj:.1f}%)")
    if noepi_mi and noepi_nj:
        print(f"  -> interchange FEED saving = noepi_mi - noepi_nj = {noepi_mi-noepi_nj:.3f}s  "
              f"({100*(noepi_mi-noepi_nj)/noepi_mi:.1f}% of mi-outer vmadot+feed·feed {noepi_mi-comp:.3f}->{noepi_nj-comp:.3f}s)")
    if m5 and m6:
        print(f"\n  M5 mi-outer matmul (full vec)          = {m5:.3f}s")
        print(f"  M6 nj-outer matmul (full vec·B-loc)    = {m6:.3f}s")
        print(f"  [M6] matmul saving = M5 - M6           = {m5-m6:.3f}s  ({100*(m5-m6)/m5:.1f}% of M5 matmul)")
        print(f"  matmul speedup M5/M6                   = {m5/m6:.3f}x" if m6 else "")
    print()

    # ---- perf pooling ----
    lines=log.splitlines()
    data={}; banners={}; freqs={}; i=0
    while i<len(lines):
        m=re.match(r'###AB pass=(\S+) side=(\S+) freq_khz=(\S+)',lines[i])
        if not m: i+=1; continue
        side=m.group(2); freqs.setdefault(side,set()).add(m.group(3))
        j=i+1; buf=[]
        while j<len(lines) and not lines[j].startswith('###BANNER') and not lines[j].startswith('###END'):
            buf.append(lines[j]); j+=1
        bcount=0
        if j<len(lines) and lines[j].startswith('###BANNER'):
            bm=re.search(r'count=(\d+)',lines[j]); bcount=int(bm.group(1)) if bm else 0
        banners[side]=banners.get(side,0)+bcount
        try: arr=json.loads("\n".join(buf).strip())
        except Exception: arr=[]
        for o in arr:
            lbl=test_label(o)
            samples=o.get("samples_ts") or [o.get("avg_ts")]
            samples=[float(x) for x in samples if x is not None]
            data.setdefault(side,{}).setdefault(lbl,[]).extend(samples)
        i=j+1
    sides=[s for s in ("off","ven","onepivec","onjout") if s in data]
    tests=[]
    for s in sides:
        for t in data[s]:
            if t not in tests: tests.append(t)
    print("side-freqs:",{s:sorted(freqs.get(s,[])) for s in sides})
    print("banner-totals:",{s:banners.get(s,0) for s in sides},"(onepivec/onjout>0 routed; ven/off==0)")
    print()
    hdr=f"{'test':>10} | "+" | ".join(f"{s+' med[iqr] relIQR n':>34}" for s in sides)
    print(hdr); print("-"*len(hdr))
    stats={}
    for t in tests:
        row=f"{t:>10} | "; cells=[]
        for s in sides:
            med,q1,q3,rel,n=med_iqr(data.get(s,{}).get(t,[]))
            stats[(s,t)]=med
            cells.append(f"{med:8.4f} [{q1:7.4f},{q3:7.4f}] {rel*100:5.2f}% n={n:<2}")
        print(row+" | ".join(cells))
    print()
    print("=== M6 BENEFIT (throughput t/s; >1 faster) ===")
    for t in tests:
        onv=stats.get(("onepivec",t)); onj=stats.get(("onjout",t)); ven=stats.get(("ven",t)); off=stats.get(("off",t))
        def r(a,b): return a/b if (a and b) else float('nan')
        print(f"[{t}] M6 onjout/onepivec={r(onj,onv):.4f}x  |  onjout/off={r(onj,off):.4f}x (stock)  |  onjout/ven={r(onj,ven):.4f}x")
        print(f"      M5 baseline onepivec/off={r(onv,off):.4f}x  |  vendor VEN/OFF={r(ven,off):.3f}x ceiling ref (same -t)")
        print(f"      slowdown vs stock: off/onjout={r(off,onj):.2f}x   vs vendor: ven/onjout={r(ven,onj):.2f}x")

if __name__=="__main__":
    main()
