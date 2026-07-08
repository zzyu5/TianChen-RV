#!/usr/bin/env python3
"""Aggregate G2 fusion e2e A/B (fusion ON vs OFF) from g2_fuse_e2e_ab.sh output.

Parses ###FAB pass=P arm=on|off ... <llama-bench json> ... ###END blocks.
Each json array has one prefill object (n_prompt>0) and one decode object
(n_gen>0), each with samples_ts[]. We pool samples across passes per (arm,phase),
report median / IQR / min, the ON-vs-OFF speedup, and a T-N effect verdict.
"""
import sys, json, re, statistics as st

def parse(path):
    txt = open(path).read()
    blocks = re.findall(r'###FAB pass=(\d+) arm=(on|off)[^\n]*\n(.*?)###END', txt, re.S)
    # pooled[arm][phase] = list of tok/s samples
    pooled = {'on': {'pp': [], 'tg': []}, 'off': {'pp': [], 'tg': []}}
    for _pass, arm, body in blocks:
        m = re.search(r'\[.*\]', body, re.S)
        if not m:
            continue
        arr = json.loads(m.group(0))
        for obj in arr:
            phase = 'pp' if obj.get('n_prompt', 0) > 0 else 'tg'
            s = obj.get('samples_ts') or [obj.get('avg_ts')]
            pooled[arm][phase] += [float(x) for x in s if x is not None]
    return pooled

def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    med = st.median(xs)
    q1 = xs[max(0, int(0.25*(n-1)))]
    q3 = xs[min(n-1, int(round(0.75*(n-1))))]
    return dict(n=n, median=med, q1=q1, q3=q3, iqr=q3-q1, mn=min(xs), mx=max(xs))

def main():
    pooled = parse(sys.argv[1])
    print("phase | arm | N | median t/s | IQR t/s | IQR% | min | max")
    res = {}
    for phase in ('pp', 'tg'):
        for arm in ('on', 'off'):
            xs = pooled[arm][phase]
            if not xs:
                print(f"{phase} | {arm} | 0 | (no data)")
                continue
            s = stats(xs)
            res[(phase, arm)] = s
            iqrp = 100*s['iqr']/s['median'] if s['median'] else 0
            print(f"{phase} | {arm} | {s['n']} | {s['median']:.4f} | {s['iqr']:.4f} | {iqrp:.3f}% | {s['mn']:.4f} | {s['mx']:.4f}")
    print("\n== FUSION EFFECT (ON vs OFF; tok/s higher=faster) ==")
    for phase in ('pp', 'tg'):
        if (phase,'on') in res and (phase,'off') in res:
            on, off = res[(phase,'on')], res[(phase,'off')]
            # speedup in throughput: on/off (>1 => fusion faster)
            sp = on['median']/off['median']
            # noise floor: max IQR% of the two bands
            floor = max(100*on['iqr']/on['median'], 100*off['iqr']/off['median'])
            delta_pct = 100*(sp-1)
            # non-overlap of IQR bands?
            overlap = not (on['q1'] > off['q3'] or off['q1'] > on['q3'])
            verdict = "SIGNIFICANT" if (abs(delta_pct) > 2*floor and abs(delta_pct) > 2.0 and not overlap) else "NOT-SIGNIFICANT (within noise)"
            print(f"{phase}: ON={on['median']:.4f} OFF={off['median']:.4f} t/s | speedup(on/off)={sp:.4f} ({delta_pct:+.2f}%) | floor={floor:.3f}% | IQR_overlap={overlap} | {verdict}")

if __name__ == '__main__':
    main()
