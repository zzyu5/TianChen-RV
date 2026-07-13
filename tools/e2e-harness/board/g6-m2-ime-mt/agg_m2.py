#!/usr/bin/env python3
# Aggregate a run-m2-q40-board.sh run.log: parse ###AB..###END blocks (llama-bench -o json),
# pool per-rep samples_ts per (side,test) across passes, report median/IQR/relIQR/n, and the
# M2 bundle-benefit ratios. Sides: off (stock RVV), ven (vendor IME), onnc (per-call repack),
# oncache (M1 cached), onmt (M2 cached+multithread). Usage: agg_m2.py run.log
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
    sides=[s for s in ("off","ven","onnc","oncache","onmt") if s in data]
    tests=[]
    for s in sides:
        for t in data[s]:
            if t not in tests: tests.append(t)
    print("side-freqs:",{s:sorted(freqs.get(s,[])) for s in sides})
    print("banner-totals:",{s:banners.get(s,0) for s in sides},"(onnc/oncache/onmt>0 routed; ven/off==0)")
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
    print("=== M2 BUNDLE-BENEFIT (throughput t/s; >1 faster) ===")
    for t in tests:
        onnc=stats.get(("onnc",t)); onc=stats.get(("oncache",t)); onmt=stats.get(("onmt",t))
        ven=stats.get(("ven",t)); off=stats.get(("off",t))
        def r(a,b): return a/b if (a and b) else float('nan')
        print(f"[{t}] M1 cache oncache/onnc = {r(onc,onnc):.3f}x   |   M2 multithread onmt/oncache = {r(onmt,onc):.3f}x   |   combined onmt/onnc = {r(onmt,onnc):.3f}x")
        print(f"      e2e vs stock  : onnc/off={r(onnc,off):.4f}x -> oncache/off={r(onc,off):.4f}x -> onmt/off={r(onmt,off):.4f}x   (bundle-benefit curve)")
        print(f"      e2e vs vendor : onmt/ven={r(onmt,ven):.4f}x   (vendor VEN/OFF={r(ven,off):.3f}x ceiling ref; same -t matchup)")

if __name__=="__main__":
    main()
