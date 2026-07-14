import re, statistics as st, sys
def parse(path):
    d={}
    for ln in open(path):
        m=re.search(r'fmt=(\S+).*?ours_gmacs=([\d.]+).*?opp_gmacs=([\d.]+).*?ratio_ours_over_opp=([\d.]+)', ln)
        if m:
            fmt=m.group(1); d.setdefault(fmt,{'ours':[],'opp':[],'ratio':[]})
            d[fmt]['ours'].append(float(m.group(2))); d[fmt]['opp'].append(float(m.group(3))); d[fmt]['ratio'].append(float(m.group(4)))
    return d
def reliqr(v):
    v=sorted(v); n=len(v)
    # linear interpolation quartiles
    def q(p):
        idx=p*(n-1); lo=int(idx); frac=idx-lo
        return v[lo] if lo+1>=n else v[lo]*(1-frac)+v[lo+1]*frac
    q1,q3=q(0.25),q(0.75); med=st.median(v)
    return med, (q3-q1)/med*100
order=['q4_0','q4_1','q5_0','q5_1','q8_0']
for cfg in ['raw_gcc_O3','raw_gcc_O2','raw_clang_O3']:
    d=parse(cfg+'.txt')
    print(f"\n===== {cfg} (N={len(d[order[0]]['ratio'])}) =====")
    print(f"{'fmt':6} {'N':>3} {'ratio_med':>9} {'ratio_relIQR%':>13} {'ratio_min':>9} {'ratio_max':>9} {'ours_gmacs_med':>14} {'opp_gmacs_med':>13} {'verdict':>9}")
    for fmt in order:
        r=d[fmt]['ratio']; med,riqr=reliqr(r)
        om,_=reliqr(d[fmt]['ours']); pm,_=reliqr(d[fmt]['opp'])
        verdict='>=parity' if med>=1.0 else '<parity'
        print(f"{fmt:6} {len(r):>3} {med:>9.4f} {riqr:>13.3f} {min(r):>9.4f} {max(r):>9.4f} {om:>14.4f} {pm:>13.4f} {verdict:>9}")
