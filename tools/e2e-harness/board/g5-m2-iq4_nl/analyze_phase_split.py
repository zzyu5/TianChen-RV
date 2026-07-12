#!/usr/bin/env python3
# [G5 M2 q4_K] Parse ###AB..###END phase-split blocks (llama-bench -o json) into
# prefill(pp)/decode(tg) throughput per side (ours=q4kON / stock=q4kOFF), take the
# median across passes+reps, compute ratio ours/stock, and a crude IQR hygiene flag.
import sys, json, statistics, re

def parse(path):
    blocks = []
    cur = None
    buf = []
    for line in open(path):
        m = re.match(r'###AB pass=(\S+) side=(\S+) freq_khz=(\S+)', line)
        if m:
            cur = {"pass": m.group(1), "side": m.group(2), "freq": m.group(3)}
            buf = []
            continue
        if line.startswith('###END'):
            if cur is not None:
                txt = "".join(buf).strip()
                try:
                    cur["json"] = json.loads(txt)
                except Exception as e:
                    cur["json"] = None; cur["err"] = str(e)
                blocks.append(cur); cur = None
            continue
        if cur is not None:
            buf.append(line)
    return blocks

def ts_of(entry):
    # llama-bench: pp entry has n_gen==0/n_prompt>0; tg entry n_prompt==0/n_gen>0
    return entry

def collect(blocks):
    # side -> {'pp':[ts...], 'tg':[ts...]}
    agg = {}
    for b in blocks:
        if not b.get("json"): continue
        side = b["side"]
        d = agg.setdefault(side, {"pp": [], "tg": []})
        for e in b["json"]:
            npr = int(e.get("n_prompt", 0)); ngn = int(e.get("n_gen", 0))
            # avg_ts is tokens/sec; samples_ts is per-rep list if present
            samples = e.get("samples_ts") or [e.get("avg_ts")]
            samples = [float(x) for x in samples if x is not None]
            if ngn == 0 and npr > 0:
                d["pp"] += samples
            elif npr == 0 and ngn > 0:
                d["tg"] += samples
    return agg

def med_iqr(xs):
    xs = sorted(xs)
    if not xs: return (None, None, None)
    med = statistics.median(xs)
    if len(xs) >= 4:
        q1 = statistics.median(xs[:len(xs)//2])
        q3 = statistics.median(xs[(len(xs)+1)//2:])
        iqr = q3 - q1
    else:
        iqr = (max(xs)-min(xs))
    return (med, iqr, len(xs))

def main():
    blocks = parse(sys.argv[1])
    agg = collect(blocks)
    print("== per-side sample counts ==")
    for side, d in agg.items():
        print(f"  {side}: pp_n={len(d['pp'])} tg_n={len(d['tg'])}")
    res = {}
    for phase in ("pp", "tg"):
        o = agg.get("ours", {}).get(phase, [])
        s = agg.get("stock", {}).get(phase, [])
        om, oi, on = med_iqr(o)
        sm, si, sn = med_iqr(s)
        ratio = (om/sm) if (om and sm) else None
        # IQR hygiene: relative IQR <= 15% (floor x1.5 proxy)
        orel = (oi/om) if (om and oi is not None) else None
        srel = (si/sm) if (sm and si is not None) else None
        res[phase] = dict(ours_med=om, stock_med=sm, ratio=ratio,
                          ours_relIQR=orel, stock_relIQR=srel, on=on, sn=sn)
        pname = "PREFILL(pp)" if phase=="pp" else "DECODE(tg)"
        print(f"== {pname} ==")
        print(f"  ours(q4kON)  median={om} t/s  relIQR={orel}  n={on}")
        print(f"  stock(q4kOFF) median={sm} t/s  relIQR={srel}  n={sn}")
        if ratio: print(f"  RATIO ours/stock = {ratio:.4f}x  ({'>=parity' if ratio>=0.98 else 'DILUTED/<parity'})")
    print("== JSON ==")
    print(json.dumps(res))

if __name__ == "__main__":
    main()
