#!/usr/bin/env python3
"""G8 冻结整顿令一.3 · kernel-unit T3 重铸 + recon (机算·禁手写小计).
唯一计数单位 = kernel(算子×格式). 板=属性列. 溯源: [COV-1] 83 schema + T3_A(rvv 0.8-gate) + T3_B(k1 0.8-gate).
只重排既有数据·零新测量. 输出: T3_kernel_unit.csv + 机算计数."""
import json, csv, sys, re
from collections import Counter, OrderedDict

ROOT = "/home/kingdom/phdworks/TianchenRV"
COV = ROOT + "/schema/perf-covered-category.v1.json"
T3A = ROOT + "/experiments/master/T3_A_board_A_rvv1.0_vlen128.csv"
T3B = ROOT + "/experiments/master/T3_B_board_B_rvv1.0_vlen256.csv"
OUT = ROOT + "/experiments/master/T3_kernel_unit.csv"

# op-name map: T3 op -> [COV-1] op
OPMAP = {"gemm": "gemm_tile", "dequant": "dequantize_row", "vec_dot": "vec_dot",
         "forward": None,  # forward rows carry the real op in format position? handled below
         "quantize": "quantize_row"}
FWD_OPS = {"add","cpy","gelu","mul","rms_norm","rope","scale","silu","softmax","rms_norm_mul"}

def classify_verdict(vfield, full):
    """classify by the VERDICT TOKEN right after 'in-denom[*;]', not the whole field
    (notes contain 'GEMM-L32-PASS' cross-refs and '§六-FAIL-VOID' text that fool substring match)."""
    v = vfield
    if "JUDGMENT-SUSPENDED" in v: return "例外-数值档挂起", False, False
    if "去向=DEQ-AXIS" in v or "test-only-not-in-denom" in v: return "DEQ-AXIS", False, False
    if "去向=内部格式" in v or "内部格式" in v: return "内部格式", False, False
    if "域外" in v or "legacy" in v: return "域外-legacy", False, False
    m = re.search(r'in-denom[*;]([A-Za-z0-9._-]+)', v)
    tok = m.group(1) if m else ""
    marg = tok.startswith("PASS-MARGINAL")
    cond = tok.startswith("PASS-DEPLOYED") or tok.startswith("PASS-CONDITIONAL")
    if tok.startswith("PASS"): d="PASS"
    elif tok.startswith("FAIL") or tok.startswith("PENDING"):
        d = "具名-X" if "NAMED-X" in v else ("FAIL" if tok.startswith("FAIL") else "pending未测")
    else: d="?"
    return d, marg, cond

def parse_t3(path, board):
    """-> {(op,format): {disp, marginal, conditional, raw_key}}"""
    out = {}
    for ln in open(path):
        if ln.startswith("#") or not ln.strip(): continue
        fields = ln.rstrip("\n").split(",")
        c1 = fields[0]
        parts = c1.split("|")
        if len(parts) < 2 or parts[0] in ("measurement_row_key","axis"): continue
        # only 36-col canonical §2.1 rows carry a real 0.8-gate disposition; legacy short rows skip
        if len(fields) != 36: continue
        op_raw, fmt = parts[0], parts[1]
        if op_raw == "forward":
            op = fmt; fmt = "f32"          # forward|gelu -> op=gelu
        elif op_raw in FWD_OPS:
            op = op_raw; fmt = "f32"
        else:
            op = OPMAP.get(op_raw, op_raw)  # gemm->gemm_tile, dequant->dequantize_row
        if op is None: continue
        vfield = fields[35]                 # col36 = verdict/disposition
        d, marg, cond = classify_verdict(vfield, ln)
        if d == "?": continue
        key = (op, fmt)
        if key in out: continue             # first canonical row wins
        out[key] = {"disp": d, "marginal": marg, "conditional": cond}
    return out

def main():
    cov = json.load(open(COV))["cells"]
    ta = parse_t3(T3A, "rvv")
    tb = parse_t3(T3B, "k1")

    # kernel-unit rows keyed by (op,format); engine variants (ime) folded as note
    rows = OrderedDict()
    for c in cov:
        k = c["key"]; op, fmt, eng = k.get("op"), k.get("format"), k.get("engine") or ""
        kk = (op, fmt)
        r = rows.setdefault(kk, {"op":op,"format":fmt,"perf_cats":{}, "engines":set(),
                                 "rvv":None,"k1":None})
        r["perf_cats"][eng or "base"] = c.get("category")
        if c.get("green"): r["perf_cats"][eng or "base"] += "(GREEN)"
        r["engines"].add(eng or "base")
    # attach T3 0.8-gate dispositions
    for kk, r in rows.items():
        r["rvv"] = ta.get(kk)
        r["k1"]  = tb.get(kk)

    # ---- recon counts (kernel-unit) ----
    def has_contest(d):  # kernel is "in the 0.8 hard-gate contest" on that board
        return d is not None and d["disp"] in ("PASS","FAIL","具名-X")
    def is_pass(d): return d is not None and d["disp"]=="PASS"

    rvv_contest = [kk for kk,r in rows.items() if has_contest(r["rvv"])]
    k1_contest  = [kk for kk,r in rows.items() if has_contest(r["k1"])]
    rvv_pass = [kk for kk in rvv_contest if is_pass(rows[kk]["rvv"])]
    k1_pass  = [kk for kk in k1_contest if is_pass(rows[kk]["k1"])]
    both_contest = [kk for kk in rows if has_contest(rows[kk]["rvv"]) and has_contest(rows[kk]["k1"])]
    both_pass = [kk for kk in both_contest if is_pass(rows[kk]["rvv"]) and is_pass(rows[kk]["k1"])]

    # 去向 buckets (kernel-level, using either board's disposition + perf cat)
    def kernel_disposition(r):
        rv, k = r["rvv"], r["k1"]
        disps = {d["disp"] for d in (rv,k) if d}
        if any(has_contest(d) for d in (rv,k)): return "进硬门对局"
        if "DEQ-AXIS" in disps: return "DEQ-分账"
        if "内部格式" in disps: return "内部格式"
        if "例外-数值档挂起" in disps: return "例外-数值档挂起"
        # not in T3 census at all -> classify by perf-covered category / op
        pc = list(r["perf_cats"].values())
        if r["op"] in ("quantize_row","product_reduce"): return "非对局-quantize/product_reduce(perf账)"
        if r["op"]=="dequantize_row": return "DEQ-分账"
        if r["op"]=="vec_dot": return "非对局-vec_dot-perf声明例外(0.8轴未接/Exit)"
        if r["op"]=="gemm_tile": return "非对局-gemm-perf账(iq/ternary·未入0.8census)"
        if r["op"] in FWD_OPS: return "进硬门对局" if any(has_contest(d) for d in (rv,k)) else "forward-op-perf账"
        return "未分类-待核"

    buckets = Counter()
    for kk,r in rows.items():
        buckets[kernel_disposition(r)] += 1

    print("="*70)
    print(f"KERNEL-UNIT RECON (机算·单位=算子×格式) — {len(rows)} distinct kernels (COV-1 83 = {len(cov)} cells)")
    print("="*70)
    print("\n[Q1] 去向筐 (kernel-level·零悬空):")
    for b,n in buckets.most_common(): print(f"    {b:52s} {n}")
    print(f"    ---- Σ = {sum(buckets.values())} ----")
    print("\n[Q2] 硬门双板战况 (0.8-gate·kernel-sym·kernel-unit):")
    print(f"    rvv: PASS {len(rvv_pass)} / 有对局 {len(rvv_contest)}")
    print(f"    k1 : PASS {len(k1_pass)} / 有对局 {len(k1_contest)}")
    print(f"    双板均有对局: {len(both_contest)} ; 双板全 PASS: {len(both_pass)}")
    print(f"    双板全 PASS kernels: {sorted('|'.join(k) for k in both_pass)}")

    # write kernel-unit table
    with open(OUT,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel(op|format)","rvv_0p8_disp","rvv_flags","k1_0p8_disp","k1_flags",
                    "perf_covered_cat(by-engine)","engines","kernel_去向"])
        for kk,r in rows.items():
            rv,k = r["rvv"],r["k1"]
            def fl(d):
                if not d: return ""
                x=[]
                if d.get("marginal"): x.append("marginal")
                if d.get("conditional"): x.append("cond/deployed")
                return "+".join(x)
            w.writerow(["|".join(kk),
                        rv["disp"] if rv else "(不在T3)", fl(rv),
                        k["disp"] if k else "(不在T3)", fl(k),
                        ";".join(f"{e}:{c}" for e,c in r["perf_cats"].items()),
                        ",".join(sorted(r["engines"])),
                        kernel_disposition(r)])
    print(f"\n★ kernel-unit table written: {OUT}")

if __name__=="__main__":
    main()
