#!/usr/bin/env python3
"""Bootstrap CI for P1 #1-3 dequant cold ratios, from REAL per-rep samples (REPS_OURS/REPS_OPP).
ratio_cold = opp_med/ours_med (>1 = ours faster). Paired within-proc A/B => paired bootstrap:
resample rep indices WITH replacement, recompute median(opp)/median(ours). 10k resamples, 95% pct CI.
Prereg §4.0: N>=10 median + bootstrap CI, 2-seed; both seeds >=0.8 => PASS."""
import re, sys, statistics, random

log = open(sys.argv[1]).read()
ours, opp = {}, {}
for m in re.finditer(r'REPS_OURS fmt=(\S+) seed=(\S+) :([\d ]+)', log):
    ours.setdefault((m.group(1), m.group(2)), []).extend(float(x) for x in m.group(3).split())
for m in re.finditer(r'REPS_OPP\s+fmt=(\S+) seed=(\S+) :([\d ]+)', log):
    opp.setdefault((m.group(1), m.group(2)), []).extend(float(x) for x in m.group(3).split())

random.seed(20260717)
def boot(o, p, n=10000):
    N = len(o); rs = []
    for _ in range(n):
        idx = [random.randrange(N) for _ in range(N)]
        rs.append(statistics.median([p[i] for i in idx]) / statistics.median([o[i] for i in idx]))
    rs.sort()
    return rs[int(.025 * n)], rs[int(.975 * n)]

def riqr(v):
    s = sorted(v); n = len(s); m = s[n // 2]
    return 100 * (s[3 * n // 4] - s[n // 4]) / m

print(f"{'cell':10s} {'seed':8s} {'N':>3s} {'ours_med_ns':>12s} {'opp_med_ns':>12s} {'ratio':>7s} {'CI95':>18s} {'o_relIQR':>9s} {'p_relIQR':>9s}")
res = {}
for k in sorted(ours):
    o, p = ours[k], opp[k]
    om, pm = statistics.median(o), statistics.median(p)
    r = pm / om
    lo, hi = boot(o, p)
    res.setdefault(k[0], []).append((k[1], r, lo, hi))
    print(f"{k[0]:10s} {k[1]:8s} {len(o):3d} {om:12.0f} {pm:12.0f} {r:7.4f}  [{lo:6.4f},{hi:6.4f}] {riqr(o):8.2f}% {riqr(p):8.2f}%")

print("\n=== PREREG §4 VERDICT (both seeds >=0.8 => PASS; any <0.8 => 具名-X; split => 具名-X+seed-split) ===")
for f, v in res.items():
    rs = [x[1] for x in v]
    ge = [x >= 0.8 for x in rs]
    verdict = "PASS[0.8]" if all(ge) else ("具名-X[0.8] (seed-split)" if any(ge) else "具名-X[0.8]")
    print(f"{f:10s} seeds={[f'{x:.4f}' for x in rs]} -> {verdict}")
