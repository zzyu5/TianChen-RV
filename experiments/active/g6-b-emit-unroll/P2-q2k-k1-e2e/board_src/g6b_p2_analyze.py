#!/usr/bin/env python3
# [G6-B Phase-2 q2_K] Parse ###AB..###END 3-way phase-split (rolled/unrolled/stock) llama-bench json,
# median across passes+reps per side, ratios rolled/stock (headline), unrolled/stock (baseline control),
# rolled/unrolled (isolated emit delta), + relIQR hygiene.
import sys, json, statistics, re

def parse(path):
    blocks=[]; cur=None; buf=[]
    for line in open(path):
        m=re.match(r'###AB pass=(\S+) side=(\S+).*freq_khz=(\S+)', line)
        if m: cur={"pass":m.group(1),"side":m.group(2),"freq":m.group(3)}; buf=[]; continue
        if line.startswith('###END'):
            if cur is not None:
                txt="".join(buf).strip()
                try: cur["json"]=json.loads(txt)
                except Exception as e: cur["json"]=None; cur["err"]=str(e)
                blocks.append(cur); cur=None
            continue
        if cur is not None: buf.append(line)
    return blocks

def collect(blocks):
    agg={}
    for b in blocks:
        if not b.get("json"): continue
        d=agg.setdefault(b["side"], {"pp":[], "tg":[]})
        for e in b["json"]:
            npr=int(e.get("n_prompt",0)); ngn=int(e.get("n_gen",0))
            samples=e.get("samples_ts") or [e.get("avg_ts")]
            samples=[float(x) for x in samples if x is not None]
            if ngn==0 and npr>0: d["pp"]+=samples
            elif npr==0 and ngn>0: d["tg"]+=samples
    return agg

def med_iqr(xs):
    xs=sorted(xs)
    if not xs: return (None,None,None)
    med=statistics.median(xs)
    if len(xs)>=4:
        q1=statistics.median(xs[:len(xs)//2]); q3=statistics.median(xs[(len(xs)+1)//2:]); iqr=q3-q1
    else: iqr=(max(xs)-min(xs))
    return (med,iqr,len(xs))

def main():
    blocks=parse(sys.argv[1]); agg=collect(blocks)
    print("== per-side sample counts ==")
    for side,d in agg.items(): print(f"  {side}: pp_n={len(d['pp'])} tg_n={len(d['tg'])}")
    res={}
    for phase in ("pp","tg"):
        pname="PREFILL(pp)" if phase=="pp" else "DECODE(tg)"
        print(f"== {pname} ==")
        meds={}
        for side in ("rolled","unrolled","stock"):
            m,i,n=med_iqr(agg.get(side,{}).get(phase,[]))
            meds[side]=m
            rel=(i/m) if (m and i is not None) else None
            relstr=f"{rel*100:.2f}%" if rel is not None else "NA"
            print(f"  {side:8s} median={m} t/s  relIQR={relstr}  n={n}")
        def rat(a,b):
            return (meds[a]/meds[b]) if (meds.get(a) and meds.get(b)) else None
        r_rs=rat("rolled","stock"); r_us=rat("unrolled","stock"); r_ru=rat("rolled","unrolled")
        def verdict(r): return (">=parity WIN" if r>=0.98 else "<parity LOSS") if r else "NA"
        if r_rs: print(f"  ROLLED/STOCK     = {r_rs:.4f}x  ({verdict(r_rs)})   <- Phase-2 headline")
        if r_us: print(f"  UNROLLED/STOCK   = {r_us:.4f}x  ({verdict(r_us)})   <- baseline control (g5=.873x)")
        if r_ru: print(f"  ROLLED/UNROLLED  = {r_ru:.4f}x  (emit-quality delta in isolation)")
        res[phase]=dict(rolled=meds["rolled"],unrolled=meds["unrolled"],stock=meds["stock"],
                        rolled_over_stock=r_rs,unrolled_over_stock=r_us,rolled_over_unrolled=r_ru)
    print("== JSON =="); print(json.dumps(res))

if __name__=="__main__": main()
