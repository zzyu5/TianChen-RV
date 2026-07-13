import re,json,sys,statistics
raw=open(sys.argv[1]).read()
blocks=re.findall(r"###AB (.*?)\n(.*?)###END", raw, re.S)
agg={}
for hdr,body in blocks:
    m=dict(re.findall(r"(\w+)=(\S+)", hdr))
    side=m.get("side")
    arr=json.loads(body[body.index("["):body.rindex("]")+1])
    print("-- pass=%s side=%s freq=%s"%(m.get("pass"),side,m.get("freq_khz")))
    for r in arr:
        ph = "prefill" if r.get("n_gen",0)==0 else "decode"
        print("     n_prompt=%s n_gen=%s avg_ts=%.4f sd=%.4f"%(r.get("n_prompt"),r.get("n_gen"),r["avg_ts"],r.get("stddev_ts",0)))
        agg.setdefault((ph,side),[]).append(r["avg_ts"])
print("==== SUMMARY ====")
for ph in ["prefill","decode"]:
    on=agg.get((ph,"on"),[]); off=agg.get((ph,"off"),[])
    if on and off:
        r=statistics.mean(on)/statistics.mean(off)
        wc=min(on)/max(off)
        print("%-8s ON=%.4f OFF=%.4f  ON/OFF=%.4fx  worst=%.4fx"%(ph,statistics.mean(on),statistics.mean(off),r,wc))
