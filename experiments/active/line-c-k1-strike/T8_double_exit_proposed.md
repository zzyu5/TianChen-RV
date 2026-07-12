# T8 双出口登记 (proposed rows) — q8_0-k1 VLEN256 correctness · main-session merge

> **Line C · 1a T8 deliverable**. Proposed T8 rows for the q8_0-k1 VLEN256 correctness double-exit + the k1
> opponent-tier findings. **NOT written to the live `T8_winloss_gap_ledger.csv`** (main-session-owned per t3p
> precedent; this agent ran no git). Main session may merge these rows. Snapshot HEAD = working-tree at probe time.

## q8_0-k1 DOUBLE-EXIT (both simultaneously TRUE — same non-portable body)

### row A — EXIT-1 · 健康 (硬对拼格)
```
key: q8_0-k1-vlen256-upstream-repack-HEALTHY
board: k1 (SpacemiT X60 VLEN256, stock clang-18)
opponent: ggml_gemm/gemv_q8_0_16x1_q8_0 (upstream riscv-16x1 tuned repack, dispatched @case256)
finding: vsetivli zero,16,e32,m2 == VLMAX@256 (256*2/32=16) => runs FULL WIDTH = correct/healthy;
         gemv ~65 rvv insns (well-vectorized). q8_0-k1 = 硬对拼格 (opponent is a runnable tuned kernel;
         parity is the good result, not a weak fallback).
class: healthy-opponent / parity-target
evidence: opponent_map_k1.md P4; raw_probe_evidence.txt
```

### row B — EXIT-2 · 破损@VLEN128 = 碎片化第二证据 ([GAP-Q8_0-VLEN128-KERNEL])
```
key: GAP-Q8_0-VLEN128-KERNEL-frag-2nd-evidence
finding: the SAME hardcoded-AVL=16 body is VLEN-NON-PORTABLE. At VLEN128 VLMAX(e32,m2)=8 => AVL=16 clamps to 8
         => half the 16-interleave columns skipped => garbage. PROVEN in G5-M1b (裸翻 gate = "olta" garbage RED;
         our emitted vl=8 carrier = coherent GREEN). Healthy@256 ∧ broken@128 = one non-portable hand-written body.
claim: 手写库 VLEN 不可移植 = C1 对手侧碎片化供弹 (a library kernel correct on exactly ONE VLEN, not capability-keyed).
sibling: [GAP-Q4K-VLEN128] (q4_K arch body also hardcodes AVL=16, PPL 822057 garbage@128) = same class.
evidence: opponent_map_k1.md P4; g5-wiring/M1b-q8_0/MANIFEST.md; g5-wiring/M2-q4_K/evidence.md
```

## supporting k1 opponent-tier rows (dispatch map, kernel-axis)
```
q4_0/q4_K/q2_K/iq4_nl/q8_0-k1 : riscv-16x1 tuned repack (all vsetivli-16 healthy@256) = 硬对拼格 tier-A
q5_K/q6_K/q3_K/...-k1         : block-dot vec_dot fallback tier-B (q5_K vec_dot VECTORIZED 13 rvv/15 vsetvli;
                                q6_K vec_dot SCALAR 0 rvv insns = weak opponent)
k1 compiler-symmetry          : stock=clang-18 (CMakeCache) => kernel-axis compiler-symmetric on k1 (UNLIKE rvv);
                                t4a k1 ratios (q4_K 3.106× / q5_K 1.916×) re-labeled compiler-symmetric-VALID
                                (NOT clang-ours-vs-gcc artifact). SpacemiT/IME vendor path ABSENT in stock (sym=0).
```

## q5_K-k1 e2e row (声明例外)
```
key: q5_K-k1-e2e-DECLARED-EXCEPTION
verdict: 声明例外 (weight-reconstruction-bound; deployed-domain e2e upper bound <= parity by sibling q4_K 0.42×;
         Amdahl proxy-input not deployed-domain-valid). kernel-axis micro 1.916× compiler-symmetric-VALID (k1) but
         micro↛e2e (K-quant instance of kernel-wins-dont-transplant-to-e2e). perf-covered unchanged 6/84.
reopen: [WORK-ITEM-K1-KQUANT-E2E] (measure q4_K e2e on k1-clang to resolve gcc-death vs intrinsic-recon-overhead).
evidence: q5_K_X0_amdahl_verdict.md
```
