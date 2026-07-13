#!/usr/bin/env python3
# Aggregate a run-m7-q40-board.sh run.log: vmadot array-utilization (WIDE output tiling).
# Profile configs (ns_matmul = parallel critical path): compute_l1 (baseline width-1 vmadot compute
# only) / compute_l1_w2 / compute_l1_w4 (WIDE vmadot compute only) / onjout_noepi (M6 vmadot+feed) /
# onw2_noepi (wide vmadot+feed). Perf sides: off / ven / onjout (M6 baseline) / onw2 / onw4.
# array-util win = compute_l1 - compute_l1_wN. Usage: agg_m7.py run.log
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

    # ---- vmadot array-utilization decomposition (X-0) ----
    print("=== [X-0] vmadot ARRAY-UTIL: WIDE output tiling (4-hart; ns_matmul = parallel crit path) ===")
    prof={}
    for m in re.finditer(r'\[TCRV-Q40-PROF\] (.+)', log):
        d=dict(re.findall(r'(\w+)=([0-9.]+)', m.group(1)))
        l1=d.get('mmprof_l1','0'); ne=d.get('mmprof_noepi','0'); nj=d.get('njouter','0')
        ev=d.get('epivec','0'); tw=d.get('tilew','0')
        if   l1=='1' and tw=='2':          side='compute_l1_w2'
        elif l1=='1' and tw=='4':          side='compute_l1_w4'
        elif l1=='1':                      side='compute_l1'
        elif ne=='1' and nj=='1' and tw=='2': side='onw2_noepi'
        elif ne=='1' and nj=='1':          side='onjout_noepi'
        elif ne=='1':                      side='onepivec_noepi'
        elif nj=='1' and ev=='1' and tw=='2': side='onw2_full'
        elif nj=='1' and ev=='1' and tw=='4': side='onw4_full'
        elif nj=='1' and ev=='1':          side='onjout_full'
        elif ev=='1':                      side='onepivec_full'
        else:                              side='other'
        prof[side]=d
        def g(k): return float(d.get(k,0))/1e9
        print(f"  {side:>16}: ns_matmul={g('ns_matmul'):8.3f}s  ns_quant={g('ns_quant'):6.3f}s  "
              f"ns_copyback={g('ns_copyback'):6.3f}s  (calls={d.get('calls','-')})")
    def gm(side): return float(prof.get(side,{}).get('ns_matmul',0))/1e9
    comp=gm('compute_l1'); comp2=gm('compute_l1_w2'); comp4=gm('compute_l1_w4')
    m6=gm('onjout_full'); w2=gm('onw2_full'); w4=gm('onw4_full')
    noepi_nj=gm('onjout_noepi'); noepi_w2=gm('onw2_noepi')
    print()
    if comp:
        print(f"  vmadot COMPUTE width-1 (fixed L1) = {comp:.3f}s   [M6 residual = matmul 33.8%]")
        if comp2: print(f"  vmadot COMPUTE width-2 (fixed L1) = {comp2:.3f}s   -> array-util win = {comp-comp2:.3f}s ({comp/comp2:.3f}x compute)" if comp2 else "")
        if comp4: print(f"  vmadot COMPUTE width-4 (fixed L1) = {comp4:.3f}s   -> array-util win = {comp-comp4:.3f}s ({comp/comp4:.3f}x compute)" if comp4 else "")
    if noepi_nj:
        print(f"  vmadot+feed width-1 (nj-outer)    = {noepi_nj:.3f}s")
        if noepi_w2: print(f"  vmadot+feed width-2 (nj-outer)    = {noepi_w2:.3f}s   -> {noepi_nj-noepi_w2:.3f}s saved")
    if m6:
        print(f"\n  M6 matmul (full vec width-1)      = {m6:.3f}s")
        if w2: print(f"  M7 matmul width-2 (full vec)      = {w2:.3f}s   [M6-M7] saving = {m6-w2:.3f}s ({100*(m6-w2)/m6:.1f}%)  speedup {m6/w2:.3f}x" if w2 else "")
        if w4: print(f"  M7 matmul width-4 (full vec)      = {w4:.3f}s   [M6-M7] saving = {m6-w4:.3f}s ({100*(m6-w4)/m6:.1f}%)  speedup {m6/w4:.3f}x" if w4 else "")
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
    sides=[s for s in ("off","ven","onjout","onw2","onw4") if s in data]
    tests=[]
    for s in sides:
        for t in data[s]:
            if t not in tests: tests.append(t)
    print("side-freqs:",{s:sorted(freqs.get(s,[])) for s in sides})
    print("banner-totals:",{s:banners.get(s,0) for s in sides},"(onjout/onw2/onw4>0 routed; ven/off==0)")
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
    print("=== M7 BENEFIT (throughput t/s; >1 faster) ===")
    for t in tests:
        onj=stats.get(("onjout",t)); ow2=stats.get(("onw2",t)); ow4=stats.get(("onw4",t))
        ven=stats.get(("ven",t)); off=stats.get(("off",t))
        def r(a,b): return a/b if (a and b) else float('nan')
        print(f"[{t}] M6 onjout/off={r(onj,off):.4f}x (stock)")
        print(f"      M7-w2 onw2/off={r(ow2,off):.4f}x (stock) | onw2/onjout={r(ow2,onj):.4f}x (vs M6) | onw2/ven={r(ow2,ven):.4f}x")
        print(f"      M7-w4 onw4/off={r(ow4,off):.4f}x (stock) | onw4/onjout={r(ow4,onj):.4f}x (vs M6) | onw4/ven={r(ow4,ven):.4f}x")
        print(f"      vendor VEN/OFF={r(ven,off):.3f}x ceiling ref (same -t) | best-M7 slowdown vs stock: "
              f"off/best={min([x for x in [r(off,ow2),r(off,ow4)] if x==x] or [float('nan')]):.2f}x")

if __name__=="__main__":
    main()
