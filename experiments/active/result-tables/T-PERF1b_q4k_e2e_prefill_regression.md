# T-PERF1b — q4_K e2e prefill REGRESSION (measured, clean-salvaged) + compiler-asymmetry root cause

- **board**: `ssh rvv` (openEuler, VLEN128, 64c, gov=performance 2.6GHz), `taskset -c 8-15 -t 8`
- **model**: DeepSeek-R1-Distill-Llama-8B-Q4_K_M (sha256 f8eba201)
- **date**: 2026-07-10 (salvaged clean subset; earlier 02:01 run VOID by SPEC+vllm contention)
- **A** = deployed `.A-q4kON` (md5 a5e927bd) OUR vl=8 emitted q4_K repack GEMM/GEVM (banner ENGAGED confirmed)
- **Bstock** = upstream stock block-dot; **Bq4kOFF** = same A-build, q4_K gate OFF → block-dot (cleanest control)
- source raw: board `/tmp/tf_paired/{RUN.log, clean/*.json, neighbor_snapshots.log}`
- **status**: measured NEGATIVE result (clean). NOT a Win. See mechanism audit
  `docs/reports/2026-07-10-q4k-micro-vs-e2e-regression-mechanism.md`.

## pp128 (n=4 clean rounds), avg_ts (tok/s)
| variant | r1 | r2 | r3 | r4 | median | mean | CI(±) | wall/round |
|---|---|---|---|---|---|---|---|---|
| A (ours vl=8, gcc-15) | 1.229266 | 1.231037 | 1.230542 | 1.228942 | 1.2299 | 1.22995 | 0.001 | ~120–121s |
| Bstock (upstream, gcc-15) | 5.191671 | 5.180183 | 5.178864 | 5.196600 | 5.1859 | 5.18683 | 0.009 | ~27s |
| Bq4kOFF (A-build block-dot, gcc-15) | 3.690114 | 3.675117 | 3.678304 | 3.677449 | 3.6779 | 3.68025 | 0.007 | ~38s |

- **S(A/Bq4kOFF) = 0.334×**  (ours ~3.0× slower than the block-dot it replaces — primary number)
- S(A/Bstock) = 0.237×
- S(Bq4kOFF/Bstock) = 0.709× (A-build integration overhead, separate small effect)

## pp512 (n=1 clean)
| variant | avg_ts | wall_ns |
|---|---|---|
| A | 1.227573 | 417082997313 |
| Bstock | 5.126412 | 99874917262 |
| Bq4kOFF | 3.668865 | 139552708512 |

- **S(A/Bq4kOFF) = 0.335×** (confirms pp128); S(A/Bstock) = 0.239×

## cleanliness / contamination boundary
- paired same-core A/B back-to-back → shared contention common-mode, cancels in ratio.
- each round `pre{busy=0%}` on pinned cores 8-15 (system loadavg 8–10 from neighbors on OTHER cores).
- determinism (NOT contention): A恒~120s / Bstock~27s / Bq4kOFF~38s, 4-round cv<1% → deterministic kernel-speed gap.
- **09:54 boundary**: neighbor_snapshots.log 09:54:42 loadavg→10.74; 09:55:43 two VLLM::EngineCore python3
  @2957%/2384% CPU; pp512 r2 truncated. Clean subset = pre-09:54 (pp128×4 + pp512×1).

## ROOT CAUSE (objdump, contention-immune): compiler asymmetry on identical source
Deployed q4_K GEMM source `/tmp/t4b_seal_fix/gemm_q4_K.inc` md5 **90d454da** (== micro "s6_q4K.c md5 90d454da").
Same board, same march, byte-identical source:

| compiler | insns | vsetvli | vector-spill | objsz | note |
|---|---:|---:|---:|---:|---|
| gcc-15.2.0 -O3 (DEPLOY / e2e) | 14817 | 820 | 742 | 58448 | == deployed .so objdump |
| clang-17.0.6 -O2 (MICRO / 1.884×) | 6430 | 71 | 3 | 27648 | == micro seal (71/3) |

- gcc emits 11.5× vsetvli, 247× spills, 2.3× insns vs clang → clang(ours)/gcc(ours) ≈ 5.6×.
- micro 1.884× = clang(ours) / gcc(block-dot) = **compiler-asymmetric INVALID**; compiler-symmetric (both gcc, shipped) = 0.334× LOSS.
- 1.884× beat/kernel-win wording **RETRACTED** [NG-4]. Only honest residue: LLVM-17 RVV backend ≫ GCC-15 on this mixed-SEW unrolled kernel.
