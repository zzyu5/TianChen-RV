#!/usr/bin/env python3
# aggregate.py -- turn board_ab.sh raw AB/WINA lines into constitutional stats:
# median, IQR%, bootstrap 95% CI of the ratio, 2x-noise-floor gate.
import sys, re, statistics, random

def parse(path):
    ab = {}       # (pass, side) -> [ns...]
    wina = {}     # side -> [ns...]
    freqs = []
    for ln in open(path):
        m = re.search(r'AB pass=(\d+) round=\d+ side=(\S+)\s+kernel_ns=([\d.]+) freq_khz=(\S+)', ln)
        if m:
            p, side, ns, fk = int(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
            ab.setdefault((p, side), []).append(ns)
            if fk.isdigit(): freqs.append(int(fk))
            continue
        m = re.search(r'WINA round=\d+ side=(\S+)\s+kernel_ns=([\d.]+)', ln)
        if m:
            wina.setdefault(m.group(1), []).append(float(m.group(2)))
    return ab, wina, freqs

def iqr_pct(xs):
    if len(xs) < 2: return 0.0
    xs = sorted(xs); n = len(xs)
    q1 = xs[max(0,int(0.25*(n-1)))]; q3 = xs[min(n-1,int(round(0.75*(n-1))))]
    med = statistics.median(xs)
    return 100.0*(q3-q1)/med if med else 0.0

def boot_ci_ratio(a, b, iters=20000):
    # ratio = median(b)/median(a) ; 95% CI via paired-independent bootstrap
    rs = []
    for _ in range(iters):
        ra = statistics.median(random.choices(a, k=len(a)))
        rb = statistics.median(random.choices(b, k=len(b)))
        rs.append(rb/ra)
    rs.sort()
    return rs[int(0.025*iters)], rs[int(0.975*iters)]

def report(path, our_label, opp_label='factory'):
    ab, wina, freqs = parse(path)
    print(f"=== {path} ===")
    if freqs:
        print(f"freq_khz: min={min(freqs)} max={max(freqs)} span={100*(max(freqs)-min(freqs))/max(freqs):.2f}% (DVFS guard)")
    # pool both passes for the A/B estimate; use pass1 vs pass2 for T-N floor
    ours = ab.get((1,'ours'),[])+ab.get((2,'ours'),[])
    opp  = ab.get((1,'factory'),[])+ab.get((2,'factory'),[])
    if ours and opp:
        mo, mf = statistics.median(ours), statistics.median(opp)
        ratio = mf/mo   # >1 => our kernel faster than factory
        lo, hi = boot_ci_ratio(ours, opp)
        # T-N between-run floor: per-side pass1-median vs pass2-median
        floor = 0.0
        for side,key in [('ours','ours'),('factory','factory')]:
            p1, p2 = ab.get((1,key),[]), ab.get((2,key),[])
            if p1 and p2:
                m1,m2 = statistics.median(p1), statistics.median(p2)
                f = 100*abs(m1-m2)/min(m1,m2)
                floor = max(floor, f)
        within_iqr = max(iqr_pct(ours), iqr_pct(opp))
        eff_floor = max(floor, within_iqr)
        delta_pct = 100*abs(mo-mf)/min(mo,mf)
        passes = (delta_pct > 2*eff_floor) and not (lo <= 1.0 <= hi)
        print(f"OUR {our_label}: median={mo:.1f}ns  IQR%={iqr_pct(ours):.2f}  n={len(ours)}")
        print(f"FACTORY:      median={mf:.1f}ns  IQR%={iqr_pct(opp):.2f}  n={len(opp)}")
        print(f"micro_vs_factory (factory_ns/our_ns) = {ratio:.4f}x   [95% CI {lo:.4f}, {hi:.4f}]")
        print(f"|delta|={delta_pct:.2f}%   between-run floor={floor:.2f}%   within-IQR={within_iqr:.2f}%   2x-eff-floor={2*eff_floor:.2f}%")
        print(f"VERDICT: {'DIFFERENCE (passes 2x floor & CI excl 1.0)' if passes else 'PARITY (within noise / CI includes 1.0) => zero-hypothesis confirmed'}")
    if wina:
        print("--- WINA (pure LMUL, both our pipeline) ---")
        keys = list(wina.keys())
        if len(keys)==2:
            a,b = keys
            ma,mb = statistics.median(wina[a]), statistics.median(wina[b])
            # report faster/slower
            print(f"{a}: median={ma:.1f}ns IQR%={iqr_pct(wina[a]):.2f}   {b}: median={mb:.1f}ns IQR%={iqr_pct(wina[b]):.2f}")
            lo,hi = boot_ci_ratio(wina[a], wina[b])
            print(f"winA_ratio ({b}_ns/{a}_ns) = {mb/ma:.4f}x  [95% CI {lo:.4f},{hi:.4f}]")

if __name__ == '__main__':
    report(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else 'ours')
