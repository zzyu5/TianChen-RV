# G8 §六.3 · 具名-X 认输体例编纂 · rvv M=1 vec_dot 输格 (2026-07-15)

> 编纂 agent acfb3aff · host-side · 决令二体例四件齐 · 决令一纪律(证据只取该格自己的解剖+构造+G1/G2·禁引预判作认输证据)。
> **needs-fresh-attack = 0**(六格 prior 证据全非薄 census·均多 commit byte-exact 板/silicon 实测)。具名-X 认输 **4**(q2/q3/q4/q6_K vec_dot·Exit-C·四格 GEMM 行皆 PASS)·次路 **2**(q5_0/q5_1·部署路 REDESIGN-B)。

## 处置总览

| 格@rvv | T3_A 行 | 冷启 M=1 | 对手真派发符号 | disposition |
|---|:--:|:--:|---|---|
| vec_dot/q2_K | L27 | 0.342× | ggml_vec_dot_q2_K_q8_K_vl128 (hand-tuned tot235/rvv125/mac66/vset9) | 具名-X 认输 · GEMM L32 **PASS 1.112** |
| vec_dot/q3_K | L28 | 0.258× | ggml_vec_dot_q3_K_q8_K_vl128 (hand-tuned tot238/rvv102/mac36/vset14) | 具名-X 认输 · GEMM L33 **PASS 1.399**(S6 rolled) |
| vec_dot/q4_K | L29 | 0.189× | ggml_vec_dot_q4_K_q8_K_vl128 (hand-tuned tot220/rvv105/mac35/vset7) | 具名-X 认输 · GEMM L34 **PASS 1.114** |
| vec_dot/q6_K | L31 | 0.180× | ggml_vec_dot_q6_K_q8_K_vl128 (hand-tuned tot232/rvv84/mac36/vset10) | 具名-X 认输 · GEMM L36 **PASS 0.960**(S6 unrolled) |
| vec_dot/q5_0 | L24 | 0.451× | ggml_vec_dot_q5_0_q8_0 (native-vec qh5bit tot112/rvv22/mac4) | 次路 · 部署路 REDESIGN-B 40c21de0 · **PASS-pending-deployed-kernel-sym-verify** |
| vec_dot/q5_1 | L25 | 0.446× | ggml_vec_dot_q5_1_q8_1 (native-vec qh5bit tot120/rvv20/mac4) | 次路 · 部署路 REDESIGN-B 40c21de0(共 leaf) · **PASS-pending-verify** |

## A 组 · K-quant super-block vec_dot（决令二·四件齐·共享根据）

**共享根据**: K-quant super-block vec_dot = M=1 GEVM·weight-recon 每输出恒定·nr=1 无 `[c]` 累加器数组 → GEMM 轴的 S6-strip-outer 摊销在 M=1 **不适用**。真 beat 归 GEMM 轴(Exit-C)。**决令一遵守**: 认输锚在实测 G2 对照(同一 recon 代码·GEMM nr≥16 PASS / GEVM nr=1 NOT-MET)+ 逐符号反汇编·**非** T-CENSUS "weight-recon floor 强先验" 预判措辞。

- **q2_K L27** (0.342×): 差距=mac66/rvv125 手调 full-unroll vs 我方 generic aux32 core(未 VLEN128 专化)。等价构造: block-dot emit(byte-exact 0)·KNEST 家族 G1 05f96739(byte-exact·G2 per-output **1.62× NOT MET**)·whole-K roll 2bbee909(v-insn −86% 零 perf)·P0 字节审计 95a8a768(布局之罪 REJECTED)。残余 X = **M=1-weight-recon-amortization-floor**(GEMM L32 PASS 1.112 vs GEVM NOT-MET·delta=摊销可得性)。X'辅=对手 _vl128 专化 [GAP-KQUANT-VECDOT-VS-NATIVE-RVV]。
- **q3_K L28** (0.258×): 等价构造: **native-mask 削重建 DEPLOYED df7dae7d**(byte-exact be_q3k 0/8·lit242·v-insn −19.5%)·KNEST 05f96739(G2 per-output **4.77× NOT MET**)·**★GEMM 轴翻正 38abf20eb**(PLAIN 0.17→S6 rolled 1.399 PASS·byte-exact 27/27·顺带证伪旧"6-bit recon 32× 内禀"断言)。残余 X = M=1-weight-recon-amortization-floor。
- **q4_K L29** (0.189×·family 最厚): 等价构造: **P1 首个独立 GEVM Emission Plan 806a7cc1**(byte-exact 三证·lit242)·**P2 板 IPC NEGATIVE 干净证伪 67d49316**(colgroup TG=2 M=1 结构回归·IPC 0.294→0.214·scalar spill 534 vs OLD 10=53×·加宽-bank M=1 无算力藏延迟)·GEMM L34 PASS 1.114。残余 X = M=1-weight-recon-amortization-floor + register-budget-fit(加宽-bank M=1 spill 53× 反噬=结构对齐必要非充分)。
- **q6_K L31** (0.180×): 等价构造: KNEST 05f96739(NO-MIN envelope·A-priori 反转正结果[native-mask 靶落 q3_K 非 q6_K]·G2 per-output **4.92× NOT MET**)·**★GEMM 轴翻正 38abf20eb**(PLAIN 0.15→S6 unrolled 0.960/1.137 PASS·spill 2212→97)·roll-GEMM a98cb8fe(6.7× kernel recovery 但 sub-parity 0.26× density floor·归 GEMM 轴 icache 杠杆非 M=1)。残余 X = M=1-weight-recon-amortization-floor。native-mask 不适用(qh=2-bit field 非单-bit plane)。

## B 组 · FLAT vec_dot（qh 5-bit·次路·PASS-pending-verify）

**共享根据**: qh 5-bit 平面重建已在**部署路**(repack-GEVM REDESIGN-B)解决并板验; block-dot vec_dot 是**非部署 secondary**。
- **q5_0 L24 / q5_1 L25** (0.451/0.446×): block-dot emit 的 qh 5-bit 平面重建发射发散(per-lane splat/vid/vsrl/vand/vsll/vncvt/vor ≈8 op) vs 对手 VLEN-adaptive native-vec → 归因 `qh-blockdot-emit-diverge [GAP-Q5x]`(发散在我方 qh 发射路)。部署路 = **REDESIGN-B repack-GEVM 削重建 40c21de0**(transposed-qh masked-vsub 4-op·重建 −43%·**G1 GREEN** lit242+板 oracle 0/256000·**G2 kernel NEW vs stock 2.2-2.5× / vs OLD 1.44-1.67×**·prefill e2e 1.21× 绿·**decode e2e 0.929×<parity 如实标**·routing deployed==proven 88 v-insn 签名)。
  - **★开门张力(主会话须裁·禁赛道躲门)**: 0.8 gate 应测**部署核**(REDESIGN-B·kernel-axis 2.2-2.5× vs stock)非非部署 block-dot(0.451)。若部署 REDESIGN-B kernel-sym M=1 cold clang-18 ≥0.8 vs 真对手 → q5_0/q5_1 应 **PASS**(rvv 18→20/28)。§六 baseline 测了非部署 block-dot 路=可能测错核。**须一次板测(clang-18 M=1 cold·部署 REDESIGN-B GEVM vs 真派发对手)定夺**·未测前保守留 FAIL(不盲升不盲留)。

## 编译器域诚实标注
- 主表对称域 = clang-18.1.8(board 内 ours==opp·-fno-integrated-as)。vec_dot ratio 与 GEMM S6 数均此域。
- gcc-15.2 部署域附注: rvv shipped=gcc。GEMM S6 翻正机制(消 spill)编译器无关·gcc 域预计亦改善待复测。REDESIGN-B e2e 部署=gcc-15.2 对称。

## 指针可核性
- commit git-verified: 806a7cc1/67d49316/92c3968d/40c21de0/7a45a2fe/05f96739/df7dae7d/2bbee909/38abf20eb/a98cb8fe/95a8a768/924dc31f/40521445。
- **⚠ a733e7d5 非 git commit**(= 独立复核 verify-agent artifact·rev-parse unknown)。GEMM S6 翻正真 commit = **38abf20eb**。T3_A 措辞已订正勿把 a733e7d5 当 commit 引。
- casefile: g8-stage3-opponent-reparse/(对手逐符号反汇编)·g8-stage3-attack/kquant-gemm-s6-q3q6-rvv/·g7-census/vecdot-rvv/·kquant-l1-q6q2q3-repack/ + l1-t3-q{2,3,5,6}k-repack-gemm/。
