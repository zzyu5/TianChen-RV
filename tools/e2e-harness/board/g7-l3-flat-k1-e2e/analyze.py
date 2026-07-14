#!/usr/bin/env python3
# [G7-L3 FLAT@k1] Parse ###AB..###END phase-split blocks (llama-bench -o json).
# side=repack (A, stock as-shipped FLAT repack) vs side=vecdot (B, block-dot).
# median across passes+reps, ratio = repack/vecdot, relIQR hygiene.
import sys, json, statistics, re

def parse(path):
    blocks=[]; cur=None; buf=[]
    for line in open(path):
        m=re.match(r'###AB pass=(\S+) side=(\S+).*freq_khz=(\S+)', line)
        if m:
            cur={"pass":m.group(1),"side":m.group(2),"freq":m.group(3)}; buf=[]; continue
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
        d=agg.setdefault(b["side"], {"pp":[], "tg":[], "freq":set()})
        d["freq"].add(b["freq"])
        for e in b["json"]:
            npr=int(e.get("n_prompt",0)); ngn=int(e.get("n_gen",0))
            samples=e.get("samples_ts") or [e.get("avg_ts")]
            samples=[float(x) for x in samples if x is not None]
            if ngn==0 and npr>0: d["pp"]+=samples
            elif npr==0 and ngn>0: d["tg"]+=samples
    return agg

def med_iqr(xs):
    xs=sorted(xs)
    if not xs: return (None,None,0)
    med=statistics.median(xs)
    if len(xs)>=4:
        q1=statistics.median(xs[:len(xs)//2]); q3=statistics.median(xs[(len(xs)+1)//2:]); iqr=q3-q1
    else: iqr=(max(xs)-min(xs))
    return (med,iqr,len(xs))

def main():
    blocks=parse(sys.argv[1]); agg=collect(blocks)
    print("== per-side sample counts / freq ==");
    for side,d in agg.items(): print(f"  {side}: pp_n={len(d['pp'])} tg_n={len(d['tg'])} freq_khz={sorted(d['freq'])}")
    res={}
    for phase in ("pp","tg"):
        A=agg.get("repack",{}).get(phase,[]); B=agg.get("vecdot",{}).get(phase,[])
        am,ai,an=med_iqr(A); bm,bi,bn=med_iqr(B)
        ratio=(am/bm) if (am and bm) else None
        arel=(ai/am) if (am and ai is not None) else None
        brel=(bi/bm) if (bm and bi is not None) else None
        res[phase]=dict(repack_med=am, vecdot_med=bm, ratio=ratio, repack_relIQR=arel, vecdot_relIQR=brel, n_repack=an, n_vecdot=bn)
        pname="PREFILL(pp)" if phase=="pp" else "DECODE(tg)"
        print(f"== {pname} ==")
        print(f"  A repack(stock FLAT repack) median={am} t/s relIQR={arel} n={an}")
        print(f"  B vecdot(block-dot)         median={bm} t/s relIQR={bn and brel} n={bn}")
        if ratio: print(f"  RATIO repack/vecdot = {ratio:.4f}x  ({'>=parity' if ratio>=0.98 else '<parity'})")
    print("== JSON =="); print(json.dumps(res))

if __name__=="__main__": main()
