#!/usr/bin/env python3
# Aggregate a run-m4-q40-board.sh run.log: parse ###AB..###END blocks (llama-bench -o json),
# pool per-rep samples_ts per (side,test) across passes, report median/IQR/relIQR/n, and the
# M4 bundle-benefit ratios + parallel-serial-setup profile split. Sides: off (stock RVV),
# ven (vendor IME), onnc (per-call repack), oncache (M1 cached), onmt (M2 cached+multithread),
# onderef (M3 cached-dequant-B + register epilogue), onpar (M4 + parallel quant + light alloc).
# Usage: agg_m4.py run.log
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

    # ---- profile split (X-0 / Amdahl): serial-setup attribution ----
    print("=== [X-0] PROFILE split (4-hart; ns_matmul = parallel critical path; ns_quant/ns_alloc/ns_copyback = serial setup) ===")
    prof={}
    for m in re.finditer(r'\[TCRV-Q40-PROF\] (.+)', log):
        line=m.group(1)
        d=dict(re.findall(r'(\w+)=([0-9.]+)', line))
        c=d.get('cache','0'); th=d.get('threads','0'); dr=d.get('deref','0'); de=d.get('deref_epi','0'); ps=d.get('parsetup','0')
        if ps=='1': side='onpar'
        elif dr=='1' and de=='1': side='onderef'
        elif dr=='1': side='onderef_dq'
        elif th=='1': side='onmt'
        elif c=='1': side='oncache'
        else: side='onnc'
        prof[side]=d
        def g(k): return float(d.get(k,0))/1e9
        print(f"  {side:>9}: ns_matmul={g('ns_matmul'):8.3f}s  ns_quant={g('ns_quant'):6.3f}s  ns_alloc={g('ns_alloc'):6.3f}s  "
              f"ns_copyback={g('ns_copyback'):6.3f}s  ns_dequant(pop)={g('ns_dequant'):7.3f}s  "
              f"(dequant_runs={d.get('dequant_runs','-')} hits={d.get('cache_hits','-')})")
    if 'onderef' in prof and 'onpar' in prof:
        def g(side,k): return float(prof[side].get(k,0))/1e9
        mm_d=g('onderef','ns_matmul'); q_d=g('onderef','ns_quant'); a_d=g('onderef','ns_alloc'); cb_d=g('onderef','ns_copyback')
        mm_p=g('onpar','ns_matmul'); q_p=g('onpar','ns_quant'); a_p=g('onpar','ns_alloc'); cb_p=g('onpar','ns_copyback')
        # ith==0 critical path = matmul + quant + alloc + copyback (serial setup + parallel matmul wall)
        crit_d=mm_d+q_d+a_d+cb_d; crit_p=mm_p+q_p+a_p+cb_p
        print(f"\n  onderef (M3) ith==0 critical path: matmul={mm_d:.3f} + quant={q_d:.3f} + alloc={a_d:.3f} + copyback={cb_d:.3f} = {crit_d:.3f}s")
        print(f"  onpar   (M4) ith==0 critical path: matmul={mm_p:.3f} + quant={q_p:.3f} + alloc={a_p:.3f} + copyback={cb_p:.3f} = {crit_p:.3f}s")
        print(f"  [M4-a] quant saving   = onderef.ns_quant - onpar.ns_quant = {q_d-q_p:7.3f}s  (serial {q_d:.3f}s -> parallel-crit {q_p:.3f}s)")
        print(f"  [M4-b] alloc saving   = onderef.ns_alloc - onpar.ns_alloc = {a_d-a_p:7.3f}s")
        print(f"  [M4]   crit-path saving = onderef - onpar = {crit_d-crit_p:7.3f}s  ({100*(crit_d-crit_p)/crit_d:5.1f}% of onderef crit)")
        print(f"  crit-path speedup onderef/onpar = {crit_d/crit_p:.3f}x  (Amdahl kernel-region ceiling for M4)")
        print(f"  quant share of crit: onderef {100*q_d/crit_d:.1f}% -> onpar {100*q_p/crit_p:.1f}%")
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
    sides=[s for s in ("off","ven","onnc","oncache","onmt","onderef","onpar") if s in data]
    tests=[]
    for s in sides:
        for t in data[s]:
            if t not in tests: tests.append(t)
    print("side-freqs:",{s:sorted(freqs.get(s,[])) for s in sides})
    print("banner-totals:",{s:banners.get(s,0) for s in sides},"(onnc..onpar>0 routed; ven/off==0)")
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
    print("=== M4 BUNDLE-BENEFIT (throughput t/s; >1 faster) ===")
    for t in tests:
        onnc=stats.get(("onnc",t)); onc=stats.get(("oncache",t)); onmt=stats.get(("onmt",t))
        ondr=stats.get(("onderef",t)); onp=stats.get(("onpar",t)); ven=stats.get(("ven",t)); off=stats.get(("off",t))
        def r(a,b): return a/b if (a and b) else float('nan')
        print(f"[{t}] M1 oncache/onnc={r(onc,onnc):.3f}x | M2 onmt/oncache={r(onmt,onc):.3f}x | "
              f"M3 onderef/onmt={r(ondr,onmt):.3f}x | M4 onpar/onderef={r(onp,ondr):.3f}x | combined onpar/onnc={r(onp,onnc):.3f}x")
        print(f"      bundle-benefit curve vs stock: onnc/off={r(onnc,off):.4f}x -> oncache/off={r(onc,off):.4f}x -> "
              f"onmt/off={r(onmt,off):.4f}x -> onderef/off={r(ondr,off):.4f}x -> onpar/off={r(onp,off):.4f}x")
        print(f"      vs vendor : onpar/ven={r(onp,ven):.4f}x   (vendor VEN/OFF={r(ven,off):.3f}x ceiling ref; same -t matchup)")
        print(f"      slowdown vs stock : off/onpar={r(off,onp):.2f}x slower   vs vendor : ven/onpar={r(ven,onp):.2f}x slower")

if __name__=="__main__":
    main()
