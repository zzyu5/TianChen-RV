#!/usr/bin/env python3
# [WORK-ITEM-K1-KQUANT-E2E] Parse phase_split_ab.sh output (###AB..###END blocks each
# containing a llama-bench JSON array) -> per-variant per-test (pp/tg) t/s samples,
# median, IQR, relIQR, and REPACK/VECDOT ratio (prefill + decode).
import sys, json, re, statistics as st
txt = open(sys.argv[1]).read()
# split into AB blocks
blocks = re.findall(r'###AB ([^\n]*)\n(.*?)\n###END', txt, re.S)
# samples[variant][test] = list of t/s
samples = {}
for hdr, body in blocks:
    m = re.search(r'variant=(\w+)', hdr)
    var = m.group(1) if m else '?'
    body = body.strip()
    try:
        arr = json.loads(body)
    except Exception:
        # find the json array within body
        s = body.find('['); e = body.rfind(']')
        if s<0 or e<0: continue
        arr = json.loads(body[s:e+1])
    for row in arr:
        # test name: pp<N> or tg<N>; avg_ts = tokens/s
        n_prompt = row.get('n_prompt',0); n_gen = row.get('n_gen',0)
        test = 'prefill' if n_prompt>0 and n_gen==0 else ('decode' if n_gen>0 and n_prompt==0 else f"pp{n_prompt}_tg{n_gen}")
        ts = row.get('avg_ts')
        if ts is None and 'samples_ts' in row:
            ts = st.mean(row['samples_ts'])
        samples.setdefault(var,{}).setdefault(test,[])
        if 'samples_ts' in row and row['samples_ts']:
            samples[var][test].extend(row['samples_ts'])
        elif ts is not None:
            samples[var][test].append(ts)

def stats(xs):
    xs=sorted(xs); n=len(xs)
    med=st.median(xs)
    if n>=4:
        q1=xs[n//4]; q3=xs[(3*n)//4]; iqr=q3-q1
    else:
        iqr=(max(xs)-min(xs)) if n>1 else 0.0
    rel=100*iqr/med if med else 0
    return med, iqr, rel, n

print("=== per-variant per-test t/s (median [reliqr%] n) ===")
res={}
for var in sorted(samples):
    for test in sorted(samples[var]):
        med,iqr,rel,n = stats(samples[var][test])
        res[(var,test)]=med
        print(f"  {var:8s} {test:8s} median={med:8.3f} t/s  IQR={iqr:6.3f}  relIQR={rel:5.2f}%  n={n}")
print("=== RATIO REPACK / VECDOT (>=1.0 => repack transduces on clang) ===")
for test in ['prefill','decode']:
    r=res.get(('REPACK',test)); v=res.get(('VECDOT',test))
    if r and v:
        print(f"  {test:8s}: REPACK {r:.3f} / VECDOT {v:.3f} = {r/v:.4f}x")
