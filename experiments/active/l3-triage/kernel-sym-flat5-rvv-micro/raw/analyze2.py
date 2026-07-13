import re, statistics as st
def parse(path):
    d={}
    for ln in open(path):
        m=re.search(r'fmt=(\S+).*?ours_gmacs=([\d.]+).*?opp_gmacs=([\d.]+).*?ratio_ours_over_opp=([\d.]+)', ln)
        if m:
            f=m.group(1); d.setdefault(f,{'ours':[],'opp':[],'ratio':[]})
            d[f]['ours'].append(float(m.group(2))); d[f]['opp'].append(float(m.group(3))); d[f]['ratio'].append(float(m.group(4)))
    return d
def stat(v):
    v=sorted(v); n=len(v)
    def q(p):
        idx=p*(n-1); lo=int(idx); fr=idx-lo
        return v[lo] if lo+1>=n else v[lo]*(1-fr)+v[lo+1]*fr
    med=st.median(v); return med,(q(0.75)-q(0.25))/med*100
order=['q4_0','q4_1','q5_0','q5_1','q8_0']
d=parse('raw_gcc_O3.txt')
print("gcc-O3 (kernel-sym primary): ours-side vs opp-side vs ratio relIQR")
print(f"{'fmt':6} {'ours_relIQR%':>12} {'opp_relIQR%':>12} {'ratio_relIQR%':>13}")
for f in order:
    _,oq=stat(d[f]['ours']); _,pq=stat(d[f]['opp']); _,rq=stat(d[f]['ratio'])
    print(f"{f:6} {oq:>12.3f} {pq:>12.3f} {rq:>13.3f}")
