#!/usr/bin/env python3
import csv, glob, sys, statistics
d = sys.argv[1]      # dir
tag = sys.argv[2]    # q50 / q51
rounds = int(sys.argv[3])
def read(path):
    pp=tg=None
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                if row['n_gen']=='0' and row['n_prompt']!='0': pp=float(row['avg_ts'])
                if row['n_prompt']=='0' and row['n_gen']!='0': tg=float(row['avg_ts'])
    except Exception as e:
        return None,None
    return pp,tg
print(f"{'round':>5} | {'pp_stock':>9} {'pp_weft':>9} {'pp_ratio':>8} | {'tg_stock':>9} {'tg_weft':>9} {'tg_ratio':>8}")
pp_r=[]; tg_r=[]; pps=[];ppw=[];tgs=[];tgw=[]
for r in range(1,rounds+1):
    pps_,tgs_ = read(f"{d}/{tag}_stock_r{r}.csv")
    ppw_,tgw_ = read(f"{d}/{tag}_weft_r{r}.csv")
    if None in (pps_,tgs_,ppw_,tgw_):
        print(f"{r:>5} | incomplete"); continue
    ppr=ppw_/pps_; tgr=tgw_/tgs_
    pp_r.append(ppr); tg_r.append(tgr)
    pps.append(pps_);ppw.append(ppw_);tgs.append(tgs_);tgw.append(tgw_)
    print(f"{r:>5} | {pps_:>9.3f} {ppw_:>9.3f} {ppr:>8.3f} | {tgs_:>9.3f} {tgw_:>9.3f} {tgr:>8.3f}")
def med(x): return statistics.median(x) if x else float('nan')
print("-"*72)
print(f"MEDIAN paired ratio: prefill(pp) weft/stock = {med(pp_r):.4f}   decode(tg) weft/stock = {med(tg_r):.4f}")
print(f"MEDIAN abs t/s: pp_stock={med(pps):.3f} pp_weft={med(ppw):.3f} | tg_stock={med(tgs):.3f} tg_weft={med(tgw):.3f}")
print(f"decode(tg) ratio range: {min(tg_r):.3f}..{max(tg_r):.3f}  prefill(pp) range: {min(pp_r):.3f}..{max(pp_r):.3f}")
