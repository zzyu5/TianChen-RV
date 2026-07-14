#!/usr/bin/env python3
# Parse tagged batched-bench jsonl:  "G7BB side=OFF round=1 {json...}"
# Emit per-pl (M) median speed_pp / speed_tg for OFF & ON and the ON/OFF ratio (the crossover).
import sys, json, re, statistics as st
lines = open(sys.argv[1]).read().splitlines()
# side -> pl -> metric -> [values]
D = {}
pat = re.compile(r'side=(\w+)\s+round=(\d+)\s+(\{.*\})')
for ln in lines:
    m = pat.search(ln)
    if not m: continue
    side, rnd, js = m.group(1), int(m.group(2)), m.group(3)
    try: rec = json.loads(js)
    except Exception: continue
    pl = rec['pl']
    D.setdefault(side, {}).setdefault(pl, {'speed_pp':[], 'speed_tg':[]})
    D[side][pl]['speed_pp'].append(rec['speed_pp'])
    D[side][pl]['speed_tg'].append(rec['speed_tg'])

def med(xs): return st.median(xs) if xs else float('nan')
def iqr(xs):
    if len(xs) < 2: return 0.0
    xs = sorted(xs); n = len(xs)
    q1 = xs[n//4]; q3 = xs[(3*n)//4]; return q3-q1

pls = sorted({pl for s in D for pl in D[s]})
print(f"N rounds captured: OFF={ {pl:len(D.get('OFF',{}).get(pl,{}).get('speed_tg',[])) for pl in pls} }  ON={ {pl:len(D.get('ON',{}).get(pl,{}).get('speed_tg',[])) for pl in pls} }")
print()
for metric in ('speed_tg','speed_pp'):
    label = 'DECODE speed_tg (tok/s, M=pl batched)' if metric=='speed_tg' else 'PREFILL speed_pp (tok/s)'
    print(f"=== {label} ===")
    print(f"{'M(pl)':>6} | {'OFF med':>9} {'OFF iqr':>8} | {'ON med':>9} {'ON iqr':>8} | {'ON/OFF':>7} | verdict")
    for pl in pls:
        off = D.get('OFF',{}).get(pl,{}).get(metric,[])
        on  = D.get('ON',{}).get(pl,{}).get(metric,[])
        om, im = med(off), iqr(off); nm, inq = med(on), iqr(on)
        r = nm/om if om else float('nan')
        v = 'WIN' if r>=1.0 else 'loss'
        print(f"{pl:>6} | {om:>9.3f} {im:>8.3f} | {nm:>9.3f} {inq:>8.3f} | {r:>7.4f} | {v}")
    print()
