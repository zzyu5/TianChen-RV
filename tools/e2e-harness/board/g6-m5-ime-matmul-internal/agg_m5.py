#!/usr/bin/env python3
# Aggregate a run-m5-q40-board.sh run.log: parse ###AB..###END blocks (llama-bench -o json),
# pool per-rep samples_ts per (side,test) across passes, report median/IQR/relIQR/n, and the
# M5 matmul-internal decomposition + epilogue-vectorization benefit. Perf sides: off (stock RVV),
# ven (vendor IME), onpar (M4 scalar epilogue), onepivec (M5 vectorized epilogue).
# Profile configs (ns_matmul = parallel critical path): onpar_full / onpar_noepi (vmadot+feed) /
# onpar_nomad (scalar epilogue) / onepivec (vmadot+feed+vec-fold).
# Usage: agg_m5.py run.log
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

    # ---- matmul-internal decomposition (X-0) ----
    print("=== [X-0] MATMUL-INTERNAL decomposition (4-hart; ns_matmul = parallel critical path) ===")
    prof={}
    for m in re.finditer(r'\[TCRV-Q40-PROF\] (.+)', log):
        d=dict(re.findall(r'(\w+)=([0-9.]+)', m.group(1)))
        ne=d.get('mmprof_noepi','0'); nm=d.get('mmprof_nomadot','0'); ev=d.get('epivec','0')
        if   ne=='1': side='onpar_noepi'   # vmadot + feed (fold skipped)
        elif nm=='1': side='onpar_nomad'   # scalar epilogue (vmadot skipped)
        elif ev=='1': side='onepivec'      # vmadot + feed + VEC fold
        else:         side='onpar_full'    # vmadot + feed + scalar fold  (M4)
        prof[side]=d
        def g(k): return float(d.get(k,0))/1e9
        print(f"  {side:>12}: ns_matmul={g('ns_matmul'):8.3f}s  ns_quant={g('ns_quant'):6.3f}s  "
              f"ns_alloc={g('ns_alloc'):6.3f}s  ns_copyback={g('ns_copyback'):6.3f}s  (calls={d.get('calls','-')})")
    def gm(side): return float(prof.get(side,{}).get('ns_matmul',0))/1e9
    full=gm('onpar_full'); noepi=gm('onpar_noepi'); nomad=gm('onpar_nomad'); vec=gm('onepivec')
    print()
    if full and noepi:
        epi_marg = full - noepi
        print(f"  full (M4 scalar epilogue) ns_matmul   = {full:.3f}s")
        print(f"  noepi (vmadot + B/A feed)             = {noepi:.3f}s   ({100*noepi/full:.1f}% of full)")
        print(f"  epilogue-marginal = full - noepi      = {epi_marg:.3f}s   ({100*epi_marg/full:.1f}% of full)")
        print(f"  nomad (scalar epilogue standalone)    = {nomad:.3f}s   (fold + dA/dW feed, no B stream)")
    if full and vec:
        print(f"\n  onepivec (M5 vec epilogue) ns_matmul  = {vec:.3f}s")
        print(f"  [M5] matmul saving = full - vec       = {full-vec:.3f}s   ({100*(full-vec)/full:.1f}% of full)")
        print(f"  matmul speedup full/vec               = {full/vec:.3f}x" if vec else "")
        if noepi:
            vec_epi = vec - noepi
            print(f"  vec-epilogue-marginal = vec - noepi   = {vec_epi:.3f}s  (vs scalar epi-marginal {full-noepi:.3f}s "
                  f"-> epilogue {(full-noepi)/vec_epi:.2f}x tighter)" if vec_epi>0 else
                  f"  vec-epilogue-marginal = vec - noepi   = {vec_epi:.3f}s  (<=0: vec fold hides under feed)")
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
    sides=[s for s in ("off","ven","onpar","onepivec") if s in data]
    tests=[]
    for s in sides:
        for t in data[s]:
            if t not in tests: tests.append(t)
    print("side-freqs:",{s:sorted(freqs.get(s,[])) for s in sides})
    print("banner-totals:",{s:banners.get(s,0) for s in sides},"(onpar/onepivec>0 routed; ven/off==0)")
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
    print("=== M5 BENEFIT (throughput t/s; >1 faster) ===")
    for t in tests:
        onp=stats.get(("onpar",t)); onv=stats.get(("onepivec",t)); ven=stats.get(("ven",t)); off=stats.get(("off",t))
        def r(a,b): return a/b if (a and b) else float('nan')
        print(f"[{t}] M5 onepivec/onpar={r(onv,onp):.4f}x  |  onepivec/off={r(onv,off):.4f}x (stock)  |  "
              f"onepivec/ven={r(onv,ven):.4f}x")
        print(f"      M4 baseline onpar/off={r(onp,off):.4f}x  |  vendor VEN/OFF={r(ven,off):.3f}x ceiling ref (same -t)")
        print(f"      slowdown vs stock: off/onepivec={r(off,onv):.2f}x   vs vendor: ven/onepivec={r(ven,onv):.2f}x")

if __name__=="__main__":
    main()
