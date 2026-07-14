#!/usr/bin/env python3
# Parse ###AB..###END (side in {new,old,stock}) llama-bench -o json blocks.
# Report median pp(prefill)/tg(decode) throughput per side + ratios new/stock, new/old, old/stock.
import sys, json, statistics, re
def parse(path):
    blocks=[]; cur=None; buf=[]
    for line in open(path):
        m=re.match(r'###AB pass=(\S+) side=(\S+) freq_khz=(\S+)', line)
        if m: cur={"pass":m.group(1),"side":m.group(2),"freq":m.group(3)}; buf=[]; continue
        if line.startswith('###END'):
            if cur is not None:
                try: cur["json"]=json.loads("".join(buf).strip())
                except Exception as e: cur["json"]=None; cur["err"]=str(e)
                blocks.append(cur); cur=None
            continue
        if cur is not None: buf.append(line)
    return blocks
def collect(blocks):
    agg={}
    for b in blocks:
        if not b.get("json"): continue
        d=agg.setdefault(b["side"],{"pp":[],"tg":[]})
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
    else: iqr=max(xs)-min(xs)
    return (med,iqr,len(xs))
blocks=parse(sys.argv[1]); agg=collect(blocks)
print("== per-side sample counts ==")
for s,d in agg.items(): print(f"  {s}: pp_n={len(d['pp'])} tg_n={len(d['tg'])}")
stat={}
for phase in ("pp","tg"):
    stat[phase]={}
    for side in ("new","old","stock"):
        m,i,n=med_iqr(agg.get(side,{}).get(phase,[]))
        stat[phase][side]=dict(med=m,relIQR=(i/m if (m and i is not None) else None),n=n)
    pn="PREFILL(pp)" if phase=="pp" else "DECODE(tg)"
    print(f"== {pn} ==")
    for side in ("new","old","stock"):
        s=stat[phase][side]; print(f"  {side:5s} median={s['med']} t/s  relIQR={s['relIQR']}  n={s['n']}")
    nw=stat[phase]["new"]["med"]; ol=stat[phase]["old"]["med"]; st=stat[phase]["stock"]["med"]
    def r(a,b): return (a/b) if (a and b) else None
    rns=r(nw,st); rno=r(nw,ol); ros=r(ol,st)
    def tag(x): return ">=parity" if (x and x>=0.98) else ("<parity" if x else "NA")
    print(f"  RATIO new/stock = {rns:.4f}x ({tag(rns)})  [REDESIGN-B vs stock block-dot, system account]" if rns else "  new/stock NA")
    print(f"  RATIO new/old   = {rno:.4f}x ({tag(rno)})  [REDESIGN-B 削重建 net contribution]" if rno else "  new/old NA")
    print(f"  RATIO old/stock = {ros:.4f}x ({tag(ros)})  [OLD scaffold vs stock, path win]" if ros else "  old/stock NA")
print("== JSON =="); print(json.dumps(stat))
