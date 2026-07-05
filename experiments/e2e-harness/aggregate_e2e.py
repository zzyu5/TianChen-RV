#!/usr/bin/env python3
# aggregate_e2e.py -- turn phase_split_ab.sh raw llama-bench JSON blobs into
# constitutional e2e stats: per-phase (prefill/decode) median, IQR%, bootstrap
# 95% CI of the A/B ratio, T-N between-pass noise floor, DVFS guard, and a
# PARITY/DIFFERENCE verdict (实验总纲v1 §1 第3/5条).
#
# INPUT (stdin or file): the raw board log produced by phase_split_ab.sh, which
# interleaves blocks of the form:
#     ###AB pass=<1|2> side=<ours|stock> freq_khz=<k>
#     <llama-bench `-o json` array for one `-p PP -n TG` call>
#     ###END
# Each JSON array has a pp<PP> (prefill) entry and a tg<TG> (decode) entry, each
# carrying a "samples" list of per-rep throughput in tokens/s (higher = better).
#
# METRIC POLARITY: llama-bench reports tokens/s, so HIGHER is faster. The A/B
# ratio here is median(ours)/median(stock); >1 means our kernel is faster.
#
# CORRECTNESS FRAMING (do NOT confuse with this perf aggregate): e2e numeric
# validity is greedy-token consistency + logits sanity, produced separately by
# correctness_gate.sh. This script NEVER asserts bit-exact vs ggml.
import sys, re, json, statistics, random

def load_blocks(text):
    """Yield (pass_no, side, freq_khz, json_obj) for each ###AB ... ###END block."""
    blocks = []
    cur = None
    buf = []
    for ln in text.splitlines():
        m = re.match(r'###AB pass=(\d+) side=(\S+)(?:\s+freq_khz=(\S+))?', ln)
        if m:
            cur = (int(m.group(1)), m.group(2), m.group(3))
            buf = []
            continue
        if ln.strip() == '###END' and cur is not None:
            raw = '\n'.join(buf).strip()
            obj = None
            if raw:
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    # tolerate leading/trailing noise: extract the [ ... ] array
                    a = raw.find('['); b = raw.rfind(']')
                    if a >= 0 and b > a:
                        try: obj = json.loads(raw[a:b+1])
                        except json.JSONDecodeError: obj = None
            blocks.append((cur[0], cur[1], cur[2], obj))
            cur = None
            buf = []
            continue
        if cur is not None:
            buf.append(ln)
    return blocks

def phase_key(entry):
    """Map a llama-bench json entry to 'prefill'/'decode' via n_prompt/n_gen."""
    npr = entry.get('n_prompt', 0)
    ngn = entry.get('n_gen', 0)
    if npr and not ngn: return 'prefill'
    if ngn and not npr: return 'decode'
    return None

def samples_of(entry):
    # llama-bench `-o json` names per-rep throughput "samples_ts" (tokens/s)
    for key in ('samples_ts', 'samples'):
        s = entry.get(key)
        if isinstance(s, list) and s:
            return [float(x) for x in s]
    # fall back to avg_ts (single sample) if per-rep samples absent
    if 'avg_ts' in entry:
        return [float(entry['avg_ts'])]
    return []

def iqr_pct(xs):
    if len(xs) < 2: return 0.0
    xs = sorted(xs); n = len(xs)
    q1 = xs[max(0, int(0.25*(n-1)))]
    q3 = xs[min(n-1, int(round(0.75*(n-1))))]
    med = statistics.median(xs)
    return 100.0*(q3-q1)/med if med else 0.0

def boot_ci_ratio(ours, stock, iters=20000):
    # ratio = median(ours)/median(stock); 95% CI via independent bootstrap
    rs = []
    for _ in range(iters):
        ro = statistics.median(random.choices(ours, k=len(ours)))
        rs_ = statistics.median(random.choices(stock, k=len(stock)))
        rs.append(ro/rs_ if rs_ else float('nan'))
    rs = [r for r in rs if r == r]
    rs.sort()
    if not rs: return (float('nan'), float('nan'))
    return rs[int(0.025*len(rs))], rs[int(0.975*len(rs))]

def collect(blocks):
    """side -> phase -> {pass_no -> [samples]}"""
    out = {}
    freqs = []
    for (p, side, fk, obj) in blocks:
        if fk and fk.isdigit(): freqs.append(int(fk))
        if not obj: continue
        for entry in obj:
            ph = phase_key(entry)
            if not ph: continue
            out.setdefault(side, {}).setdefault(ph, {}).setdefault(p, []).extend(samples_of(entry))
    return out, freqs

def report(text, out_json=None, board='?', model='?', quant='?'):
    blocks = load_blocks(text)
    data, freqs = collect(blocks)
    lines = []
    def pr(s): lines.append(s); print(s)

    pr(f"=== e2e phase-split aggregate  board={board} model={model} quant={quant} ===")
    if freqs:
        span = 100*(max(freqs)-min(freqs))/max(freqs) if max(freqs) else 0.0
        pr(f"freq_khz: min={min(freqs)} max={max(freqs)} span={span:.2f}%  (DVFS guard; >2% span => INVALIDATE run)")

    ev = {'board': board, 'model': model, 'quant': quant, 'phases': {}}
    for phase in ('prefill', 'decode'):
        ours_by_pass = data.get('ours', {}).get(phase, {})
        stock_by_pass = data.get('stock', {}).get(phase, {})
        ours = [x for p in ours_by_pass.values() for x in p]
        stock = [x for p in stock_by_pass.values() for x in p]
        if not ours or not stock:
            pr(f"\n[{phase}] insufficient data (ours n={len(ours)}, stock n={len(stock)})")
            continue
        mo, ms = statistics.median(ours), statistics.median(stock)
        ratio = mo/ms  # >1 => ours faster (tokens/s, higher better)
        lo, hi = boot_ci_ratio(ours, stock)
        # T-N between-pass floor: per side, pass1-median vs pass2-median
        floor = 0.0
        for by_pass in (ours_by_pass, stock_by_pass):
            p1, p2 = by_pass.get(1, []), by_pass.get(2, [])
            if p1 and p2:
                m1, m2 = statistics.median(p1), statistics.median(p2)
                floor = max(floor, 100*abs(m1-m2)/min(m1, m2))
        within = max(iqr_pct(ours), iqr_pct(stock))
        eff_floor = max(floor, within)
        delta_pct = 100*abs(mo-ms)/min(mo, ms)
        ci_excludes_1 = not (lo <= 1.0 <= hi)
        passes = (delta_pct > 2*eff_floor) and ci_excludes_1
        verdict = ('DIFFERENCE (passes 2x floor & CI excl 1.0)' if passes
                   else 'PARITY (within noise / CI includes 1.0) => zero-hypothesis')
        pr(f"\n[{phase}]  ours median={mo:.2f} t/s (IQR%={iqr_pct(ours):.2f}, n={len(ours)})"
           f"   stock median={ms:.2f} t/s (IQR%={iqr_pct(stock):.2f}, n={len(stock)})")
        pr(f"  A/B ratio (ours/stock) = {ratio:.4f}x   95% CI [{lo:.4f}, {hi:.4f}]")
        pr(f"  |delta|={delta_pct:.2f}%   between-pass floor={floor:.2f}%   within-IQR={within:.2f}%   2x-eff-floor={2*eff_floor:.2f}%")
        pr(f"  VERDICT: {verdict}")
        ev['phases'][phase] = {
            'ours_median_tps': round(mo, 3), 'stock_median_tps': round(ms, 3),
            'ratio_ours_over_stock': round(ratio, 4), 'ci95': [round(lo, 4), round(hi, 4)],
            'iqr_pct_ours': round(iqr_pct(ours), 3), 'iqr_pct_stock': round(iqr_pct(stock), 3),
            'between_pass_floor_pct': round(floor, 3), 'n_ours': len(ours), 'n_stock': len(stock),
            'verdict': 'DIFFERENCE' if passes else 'PARITY',
        }
    if freqs:
        ev['dvfs_span_pct'] = round(100*(max(freqs)-min(freqs))/max(freqs), 3) if max(freqs) else 0.0
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(ev, f, indent=2)
        pr(f"\nwrote {out_json}")
    return ev

if __name__ == '__main__':
    args = sys.argv[1:]
    kw = {}
    pos = []
    i = 0
    while i < len(args):
        if args[i].startswith('--') and i+1 < len(args):
            kw[args[i][2:]] = args[i+1]; i += 2
        else:
            pos.append(args[i]); i += 1
    path = pos[0] if pos else None
    text = open(path).read() if path and path != '-' else sys.stdin.read()
    report(text, out_json=kw.get('out'), board=kw.get('board', '?'),
           model=kw.get('model', '?'), quant=kw.get('quant', '?'))
