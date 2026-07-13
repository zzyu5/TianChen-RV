#!/usr/bin/env python3
# Aggregate a run-m3-q40-board.sh run.log: parse ###AB..###END blocks (llama-bench -o json),
# pool per-rep samples_ts per (side,test) across passes, report median/IQR/relIQR/n, and the
# M3 bundle-benefit ratios + de-reference-form profile split. Sides: off (stock RVV),
# ven (vendor IME), onnc (per-call repack), oncache (M1 cached), onmt (M2 cached+multithread),
# onderef (M3 cached-dequant-B + register epilogue). Usage: agg_m3.py run.log
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

    # ---- profile split (X-0 / Amdahl) ----
    print("=== [X-0] PROFILE split (4-hart; ns_matmul = parallel critical path) ===")
    prof={}
    for m in re.finditer(r'\[TCRV-Q40-PROF\] (.+)', log):
        line=m.group(1)
        d=dict(re.findall(r'(\w+)=([0-9.]+)', line))
        # identify side by gate combo
        c=d.get('cache','0'); th=d.get('threads','0'); dr=d.get('deref','0'); de=d.get('deref_epi','0')
        if dr=='1' and de=='1': side='onderef'
        elif dr=='1': side='onderef_dq'
        elif th=='1': side='onmt'
        elif c=='1': side='oncache'
        else: side='onnc'
        prof[side]=d
        nm=float(d.get('ns_matmul',0))/1e9; nr=float(d.get('ns_repack',0))/1e9
        nd=float(d.get('ns_dequant',0))/1e9; nq=float(d.get('ns_quant',0))/1e9
        print(f"  {side:>11}: ns_matmul={nm:8.3f}s  ns_dequant(populate)={nd:8.3f}s  ns_repack={nr:7.3f}s  ns_quant={nq:6.3f}s  "
              f"(repack_runs={d.get('repack_runs','-')} dequant_runs={d.get('dequant_runs','-')} hits={d.get('cache_hits','-')})")
    if 'onmt' in prof and 'onderef_dq' in prof and 'onderef' in prof:
        mm=float(prof['onmt']['ns_matmul'])/1e9
        dq=float(prof['onderef_dq']['ns_matmul'])/1e9
        fu=float(prof['onderef']['ns_matmul'])/1e9
        print(f"\n  matmul critical-path (4-hart): onmt={mm:.3f}s -> onderef_dq={dq:.3f}s -> onderef={fu:.3f}s")
        print(f"  [M3-a] dequant saving   = onmt - onderef_dq   = {mm-dq:7.3f}s  ({100*(mm-dq)/mm:5.1f}% of onmt ns_matmul)")
        print(f"  [M3-b] epilogue saving  = onderef_dq - onderef = {dq-fu:7.3f}s  ({100*(dq-fu)/mm:5.1f}% of onmt ns_matmul)")
        print(f"  [M3]   total matmul saving = onmt - onderef     = {mm-fu:7.3f}s  ({100*(mm-fu)/mm:5.1f}% of onmt ns_matmul)")
        print(f"  matmul speedup onmt/onderef = {mm/fu:.3f}x  (Amdahl matmul-internal ceiling for M3)")
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
    sides=[s for s in ("off","ven","onnc","oncache","onmt","onderef") if s in data]
    tests=[]
    for s in sides:
        for t in data[s]:
            if t not in tests: tests.append(t)
    print("side-freqs:",{s:sorted(freqs.get(s,[])) for s in sides})
    print("banner-totals:",{s:banners.get(s,0) for s in sides},"(onnc/oncache/onmt/onderef>0 routed; ven/off==0)")
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
    print("=== M3 BUNDLE-BENEFIT (throughput t/s; >1 faster) ===")
    for t in tests:
        onnc=stats.get(("onnc",t)); onc=stats.get(("oncache",t)); onmt=stats.get(("onmt",t))
        ondr=stats.get(("onderef",t)); ven=stats.get(("ven",t)); off=stats.get(("off",t))
        def r(a,b): return a/b if (a and b) else float('nan')
        print(f"[{t}] M1 cache oncache/onnc={r(onc,onnc):.3f}x | M2 mt onmt/oncache={r(onmt,onc):.3f}x | "
              f"M3 deref onderef/onmt={r(ondr,onmt):.3f}x | combined onderef/onnc={r(ondr,onnc):.3f}x")
        print(f"      bundle-benefit curve vs stock: onnc/off={r(onnc,off):.4f}x -> oncache/off={r(onc,off):.4f}x -> "
              f"onmt/off={r(onmt,off):.4f}x -> onderef/off={r(ondr,off):.4f}x")
        print(f"      vs vendor : onderef/ven={r(ondr,ven):.4f}x   (vendor VEN/OFF={r(ven,off):.3f}x ceiling ref; same -t matchup)")
        print(f"      slowdown vs stock : off/onderef={r(off,ondr):.2f}x slower   vs vendor : ven/onderef={r(ven,ondr):.2f}x slower")

if __name__=="__main__":
    main()
