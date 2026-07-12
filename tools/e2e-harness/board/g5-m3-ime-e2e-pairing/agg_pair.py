#!/usr/bin/env python3
# Aggregate a run_pair_board.sh run.log: parse ###AB..###END blocks (llama-bench -o json),
# pool per-rep samples_ts per (side, test) across passes, report median/IQR/n, and the
# key ratios ON/OFF (tcrv-IME vs stock-RVV) and ON/VEN (tcrv-IME vs vendor-IME).
# Usage: agg_pair.py run.log
import sys, json, re, statistics as st

def med_iqr(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0: return (float('nan'), float('nan'), float('nan'), 0)
    med = st.median(xs)
    def pct(p):
        k = p*(n-1); lo=int(k); hi=min(lo+1,n-1); return xs[lo]+(xs[hi]-xs[lo])*(k-lo)
    return (med, pct(0.25), pct(0.75), n)

def test_label(o):
    npp = o.get("n_prompt",0); ng = o.get("n_gen",0)
    if ng and not npp: return f"tg{ng}"
    if npp and not ng: return f"pp{npp}"
    return f"pp{npp}_tg{ng}"

def main():
    log = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    lines = log.splitlines()
    # data[side][test] = list of samples (pooled across passes)
    data = {}
    banners = {}
    freqs = {}
    i = 0
    while i < len(lines):
        m = re.match(r'###AB pass=(\S+) side=(\S+) freq_khz=(\S+)', lines[i])
        if not m:
            i += 1; continue
        side = m.group(2); freq = m.group(3)
        freqs.setdefault(side, set()).add(freq)
        # collect json text until ###BANNER or ###END
        j = i+1; buf = []
        while j < len(lines) and not lines[j].startswith('###BANNER') and not lines[j].startswith('###END'):
            buf.append(lines[j]); j += 1
        bcount = 0
        if j < len(lines) and lines[j].startswith('###BANNER'):
            bm = re.search(r'count=(\d+)', lines[j]); bcount = int(bm.group(1)) if bm else 0
        banners.setdefault(side, 0)
        banners[side] += bcount
        txt = "\n".join(buf).strip()
        try:
            arr = json.loads(txt)
        except Exception:
            arr = []
        for o in arr:
            lbl = test_label(o)
            samples = o.get("samples_ts") or [o.get("avg_ts")]
            samples = [float(x) for x in samples if x is not None]
            data.setdefault(side, {}).setdefault(lbl, []).extend(samples)
        i = j+1

    sides = [s for s in ("on","ven","off") if s in data]
    tests = []
    for s in sides:
        for t in data[s]:
            if t not in tests: tests.append(t)
    def sortkey(t):
        mm = re.match(r'(pp|tg)(\d+)', t); return (0 if t.startswith('pp') else 1, int(mm.group(2)) if mm else 0)
    tests.sort(key=sortkey)

    print("side-freqs:", {s: sorted(freqs.get(s,[])) for s in sides})
    print("banner-totals:", {s: banners.get(s,0) for s in sides}, " (on>0 routed; ven/off==0 expected)")
    print()
    hdr = f"{'test':>10} | " + " | ".join(f"{s+' med(iqr,n)':>26}" for s in sides)
    print(hdr); print("-"*len(hdr))
    stats = {}
    for t in tests:
        row = f"{t:>10} | "
        cells = []
        for s in sides:
            xs = data.get(s,{}).get(t,[])
            med,lo,hi,n = med_iqr(xs)
            stats[(s,t)] = med
            cells.append(f"{med:8.3f} [{lo:6.2f},{hi:6.2f}] n={n:<3}")
        print(row + " | ".join(cells))
    print()
    print("=== RATIOS (throughput t/s; >1 = ON faster) ===")
    print(f"{'test':>10} | {'phase':>8} | {'ON/OFF(tcrvIME/stockRVV)':>26} | {'ON/VEN(tcrvIME/vendorIME)':>26} | {'VEN/OFF(vendorIME/stockRVV)':>28}")
    for t in tests:
        phase = "prefill" if t.startswith("pp") else "decode"
        on = stats.get(("on",t)); ven = stats.get(("ven",t)); off = stats.get(("off",t))
        r_on_off = on/off if (on and off) else float('nan')
        r_on_ven = on/ven if (on and ven) else float('nan')
        r_ven_off = ven/off if (ven and off) else float('nan')
        print(f"{t:>10} | {phase:>8} | {r_on_off:>26.4f} | {r_on_ven:>26.4f} | {r_ven_off:>28.4f}")

if __name__ == "__main__":
    main()
