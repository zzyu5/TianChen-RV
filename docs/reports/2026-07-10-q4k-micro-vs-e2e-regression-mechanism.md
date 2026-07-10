# q4_K micro 1.884× vs e2e 0.334× 回归 — 机制审计 (2026-07-10)

> 性能宪章 rule 1「修性能前先反汇编认瓶颈」的执行。板被邻居(SPEC CPU2006 + vllm)争用中，
> **本审计禁 perf 测量**：只用 objdump / 静态反汇编 / re-emit-compile-disasm（皆瞬时、不受争用影响）。
> A-tree 已复原（live .so md5 75f20b5f = stock），本审计未动 A-tree、未 git commit/add/stash/rm。

## VERDICT（一句话）

e2e prefill 回归的**主导机制 = H2「部署变体≠证过变体」，且是编译器层的**：micro 1.884× 与 e2e
0.334× 跑的是**字节同源（md5 90d454da）但两个编译器编出的两个不同二进制**——micro 用 **clang-17.0.6**
（71 vsetvli / 3 vector-spill / 27KB），e2e 部署用 **gcc-15.2.0 -O3**（**820 vsetvli / 742 vector-spill /
58KB**）。对手 block-dot 两侧都是 gcc-15（common-mode）。同源 gcc 编出的 kernel 比 clang 编出的慢
≈5.6×，方向反转正是这条 clang-vs-gcc codegen gap。**micro 1.884× 不作为 kernel-轴 honest win 存活**——
它是 clang(ours)-vs-gcc(block-dot) 的**编译器不对称**测量（g1-closure canon 已判此类=INVALID）。这也正是
宪章 rule 1「K-quant repack 慢 = 全展开 regfile spill」的病理，且**病理是 GCC-specific，micro 用 clang 从未看见**。

---

## 0. Salvaged 原始测量（耐久证据 — clean 子集，勿丢）

板 `ssh rvv`(openEuler/VLEN128/64c/gov=performance 2.6GHz)，模型 DeepSeek-R1-Distill-Llama-8B-Q4_K_M
(sha256 f8eba201)，`taskset -c 8-15 -t 8`，llama-bench，warmup-dropped。**A** = 部署 `.A-q4kON`(md5
a5e927bd) 我方 vl=8 emitted q4_K repack GEMM/GEVM（banner ENGAGED 9–16 次/round 确认真调用）；
**Bstock** = 上游 stock block-dot；**Bq4kOFF** = 同 A-build 但 q4_K 关闸回退 block-dot（最干净对照：
同二进制、只差 q4_K dispatch）。源 `/tmp/tf_paired/clean/*.json` + `RUN.log`。

### pp128（n=4，clean rounds，avg_ts tok/s）
| variant | r1 | r2 | r3 | r4 | median | mean | CI(±) |
|---|---|---|---|---|---|---|---|
| A (ours vl=8) | 1.229266 | 1.231037 | 1.230542 | 1.228942 | **1.2299** | 1.22995 | ±0.001 |
| Bstock | 5.191671 | 5.180183 | 5.178864 | 5.196600 | **5.1859** | 5.18683 | ±0.009 |
| Bq4kOFF | 3.690114 | 3.675117 | 3.678304 | 3.677449 | **3.6779** | 3.68025 | ±0.007 |

- **S(A / Bq4kOFF) = 1.2299 / 3.6779 = 0.334×**（我方 kernel 比它替代的 block-dot 慢 ~3.0×）。
- S(A / Bstock) = 1.2299 / 5.1859 = 0.237×。
- S(Bq4kOFF / Bstock) = 3.6779 / 5.1859 = 0.709×（A-build 的 block-dot 回退本身 = 上游 0.71×，即
  集成 harness 引入 ~29% 额外开销——一个**独立的小效应**，非主线；主线是 A/Bq4kOFF 的 0.334×）。

### pp512（n=1，clean，前污染窗口）
| variant | avg_ts | wall_ns |
|---|---|---|
| A | 1.227573 | 417082997313 |
| Bstock | 5.126412 | 99874917262 |
| Bq4kOFF | 3.668865 | 139552708512 |

- **S(A / Bq4kOFF) = 0.335×**（确认 pp128 的 0.334×，长 prompt 不救）；S(A / Bstock) = 0.239×。

### 洁净证据 + 污染边界
- **paired 同核 A/B 背靠背**：任何共享内存/调度争用 common-mode，在 ratio 里抵消。
- 每 round 起点记 `pre{busy=0%}`——我方 pin 核 8-15 在 round 起点 quiescent（尽管系统 loadavg 因
  邻居在**其它核**升到 8–10）。
- **确定性证据（非争用 artifact）**：A 每 pp128 round 恒 ~120–121s，Bstock ~27s，Bq4kOFF ~38s，
  4 round cv<1%——若是争用，wall 会剧烈抖动；恒定的 ~3–4× wall 差 = 确定性 kernel 速度差。
- **09:54 污染边界**：`neighbor_snapshots.log` 09:54:42 loadavg→10.74，09:55:43 两个 VLLM::EngineCore
  python3 @2957%/2384% CPU 涌入；pp512 r2 被击中、run exit-restore 截断。clean 子集 = 09:54 前完成的
  pp128×4 + pp512×1。**此前 02:01 的 `t4b_prefill_clean.log` run 已因 SPEC+vllm 于 02:10 VOID**（预注册
  loadavg<4 gate），本 salvaged 集是重跑的洁净子集。

---

## 1. 反汇编认瓶颈 — 部署 gcc-15 kernel（e2e 路径）

部署 `.A-q4kON` 里 `ggml_gemm_q4_K_16x1_q8_K` 是 **VLEN-dispatch 壳**：`if (vlenb*8==128)`（=VLEN128）
打 banner 并 tail-jump（PLT）到真 emitted kernel `tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_...`；
壳体内联的 vl=16 e32/m2 body 是 **VLEN≠128 回退路**（勿当部署 kernel，初读险误认）。真部署 kernel：

| symbol | 地址 | size | insns | back-edges | vsetvli | vector-spill(vs*r.v/vl*re) | scalar sd/ld(sp) |
|---|---|---:|---:|---:|---:|---:|---:|
| GEMM `tcrv_emitc_..._gemm_q4_K_q8_K` | 0x944cc | **58798 B** | 16079 | **14** | **820** | **742** | 57/111 |
| GEVM `tcrv_emitc_..._gevm_q4_K_q8_K` | 0xa2a7a | **38866 B** | 10824 | **6** | **1387 (12.8%!)** | 217 | 227/339 |
| （对比）q4_0 GEMM emitted | 0x93d8c | 566 B | — | rolled | — | — | — |

- **全展开**：GEMM 16079 条指令仅 14 back-edge，GEVM 6 条 ⇒ 几乎直线码。寄存器压力/spill 是**焊死
  在码里的、与 M/N/K shape 无关**。
- **vsetvli 风暴**：GEMM 820（5.1% of insns），GEVM 1387（**12.8% of insns**）。前 5 形态：
  `vsetivli zero,8,e8,mf2` ×311、`vsetvli zero,zero,e16,m1` ×104、`vsetvli zero,zero,e8,mf2` ×86、
  外加大量 `vsetvli sN,zero,e8,mf2`（gcc 反复把 VLMAX 查进标量寄存器）。
- **regfile spill**：GEMM 742 个 `vs2r.v`/`vl2re32.v` 整寄存器进出栈（+ 43–111 scalar sd/ld）。
- 热区实证（GEMM 0x9b6fc–0x9b784，fold/panel 段）：连串 `vs2r.v v22/v18/v6/v2,(a5)` 存栈，每次
  `li a5,89; mul a5,a5,a4; csrr a4,vlenb; srl` **重算 VLA 帧偏移 + 重读 vlenb**——gcc 未 hoist。S6 设计的
  "stack panel"（本意减寄存器压力）在 gcc 下退化成**整寄存器访存流 + 重复 csrr vlenb 地址算术**。

## 2. 直接实验 — 同源 md5 90d454da，两编译器（决定性）

部署 q4_K GEMM 源 = `/tmp/t4b_seal_fix/gemm_q4_K.inc`，**md5 90d454da655f2fc1f88435d2d5826942**
——**与 micro findings 声称的 "s6_q4K.c md5 90d454da" 字节同一**。源仅 2 个 `__riscv_vsetvl`（vl=8 靠
intrinsic 立即数携带，如 `__riscv_vfmv_v_f_f32m2(0.0f, 8)`）⇒ 部署二进制的 820 vsetvli **全是 gcc
插入的**。同板、同 march、同源编译：

| 编译 | insns | vsetvli | vector-spill | objsz | 对齐验证 |
|---|---:|---:|---:|---:|---|
| **gcc-15.2.0 -O3**（DEPLOY） | 14817 | **820** | 742 | 58448 B | == 部署 .so objdump（820/742/58KB）✓ |
| gcc-15.2.0 -O2 | 14810 | 820 | 742 | 58648 B | 同 |
| **clang-17.0.6 -O2**（MICRO） | 6430 | **71** | 3 | 27648 B | == micro seal 表（vsetvli 71 / spill 3）✓ |
| clang-17.0.6 -O3 | 6430 | 71 | 3 | 27608 B | 同 |

- **gcc 比 clang：vsetvli 11.5× 多、vector-spill 247× 多、指令 2.3× 多、码 2.1× 大。**
- gcc 的 codegen 病：`vsetivli zero,8,e8,mf2` ×311（每 8-元素组前重建 vtype）+ `vsetvli sN,zero,...`
  ×百余（反复查 VLMAX）；clang 每区一次 vtype、mixed-SEW 转换只留必要的 71 个，工作集全留寄存器
  （3 spill = bsums-panel init，非热 loop live-value）。
- 结论：**1.884×(clang) ↔ 0.334×(gcc) 反转 100% 是编译器 codegen artifact**，不是 kernel 质量、不是
  shape、不是 overhead、不是访存。

## 3. 四假设裁断

| # | 假设 | 裁断 | 证据 |
|---|---|---|---|
| **H1** | 瓶颈形状：micro nr=64 落在 spill-不触发甜点，e2e shape 触发 spill | **REFUTED** | kernel 全展开（14/6 back-edge）⇒ spill 数**焊死、shape-无关**：gcc 742 / clang 3 与 M/N/K 无关，恒发。micro 用**同 shape**（K=2048, nr=64, nc=512, prefill-like）。区分轴是**编译器非 shape**。（另有真 shape 依赖：1.884× 是 M>1 output-tiling 增益、M=1 decode 塌回 NULL——但那与 prefill 回归正交，本回归在 M>1 仍输。） |
| **H2** | 部署变体 ≠ 证过变体 | **CONFIRMED（主导）** | §2：同源 md5 90d454da，micro=clang-17（71/3/27KB），deploy=gcc-15（820/742/58KB）。对手 block-dot 两侧皆 gcc-15。clang(ours)/gcc(ours) ≈ 1.884/0.334 = **5.64×** = 编译器 gap。**"部署≠证过"第三次**（① e2e-seal 部署 vl=16 非 vl=8；② M4 standalone 不覆盖集成调用；③ 本次：micro clang 不代表 deploy gcc）。 |
| **H3** | overhead 未计：weight repack + q8_K 激活量化 | **REFUTED（非主导）** | (a) weight repack = load-time 一次性、非 per-mul_mat。(b) q8_K 激活量化 common-mode：A 与 Bq4kOFF **都**量化激活成 q8_K（block-dot 也要），在 A/Bq4kOFF ratio 里抵消；profile `quantize_row_q8_K`=0.33%。产生不了 3× gap。 |
| **H4** | memory-bound：整矩阵尺度 weight streaming，repack 布局访存更差 | **REFUTED（prefill）** | prefill 实测 97.3% matmul、**强 compute-bound**（profile + 全仓 finding）。两 kernel 流同样 q4_K 权重字节；回归 = gcc-ours 每权重字节多执行 ~2.3× 指令 + 742 spill 往返 = **指令/compute 瓶颈非访存 pattern**。(decode/GEVM 才 memory-bound，但那条 projection 早已 NULL，且 0.334× 测的是 prefill pp128/pp512。) |

## 4. Amdahl 事后账（projection 建在沙上）

原 1.59× projection = `1 / (0.7908/g₄ + 0.1818/g₆ + 0.0274)`，代入 **g₄=1.884（clang micro 数）**得 1.59×。
代入部署 gcc kernel 的真 per-format 因子 **g₄≈0.334**（A/Bq4kOFF，prefill 97% matmul ⇒ ≈纯 kernel 比）：

```
S = 1 / (0.7908/0.334 + 0.1818 + 0.0274) = 1 / (2.368 + 0.1818 + 0.0274) = 1 / 2.577 = 0.388×
```

Amdahl **正确预测 prefill 回归**（0.388×，与实测 A/Bq4kOFF 0.334× 同向同量级；残差来自部署路对
q6_K/其它张量的集成开销）。**公式无错，输入错**：1.59× 喂了一个非代表性的 clang micro 因子。
**方法学教训：projection 的 kernel 因子必须在部署目标同一编译器下测得**，否则 garbage-in(clang micro)→
garbage-out(1.59×)。

## 5. micro 1.884× 成色裁定 → **不作为 kernel-轴 honest win 存活**

- micro 1.884× = **clang-17(ours) / gcc-15(block-dot)** 的**编译器不对称**测量（ours 侧 clang-17 -O2，
  对手侧 gcc-15）。g1-closure canon：vs-厂商编译器不对称 = **INVALID**。
- 编译器对称（都 gcc-15、= 真出货口径）下，ours = **0.334×（输）**。
- 整条 S1→S6 tile 阶梯（spill 84→23→3、"register cliff reached"、+27.9%、1.473→1.884）是**对着
  clang-17 的 register allocator 验的**，**不传导到 gcc-15 出货二进制**（同源 gcc 下 spill 742、churn 820）。
  S6 的 "stack panel" 在 gcc 下甚至**反效果**（整寄存器访存流 + 重复 csrr vlenb）。
- **可诚实保留的唯一残留**：「clang-17 的 RVV 后端把我方 emitted q4_K EmitC 编得比 gcc-15 好 ~5.6×」
  ——这是 **LLVM-vs-GCC RVV 成熟度**陈述，**不是**我方 kernel/编译器 beat ggml block-dot 的陈述。
  1.884× 作为 beat/kernel-win 措辞应**撤回**（[NG-4]）。

## 6. 措辞锁（[NG-4]）

不得声称 q4_K kernel/e2e Win 或 1.884× kernel-win。可引句：「同一 md5 90d454da EmitC 源，clang-17
编出 71 vsetvli/3 spill、gcc-15 编出 820 vsetvli/742 spill；e2e prefill 部署走 gcc-15，我方 emitted
q4_K repack GEMM = 它替代的 gcc-15 block-dot 的 0.334×(pp128)/0.335×(pp512)，即慢 ~3×；micro 曾报的
1.884× 是 clang-vs-gcc 编译器不对称测量、非 kernel 质量、已撤回。回归根因 = gcc-15 RVV codegen
(vsetvli 插入 + 寄存器分配) 在此 mixed-SEW 全展开 kernel 上远弱于 clang-17，宪章 rule 1『全展开 regfile
spill』病理的 GCC-specific 实证。」

## 附：一手证据指针
- 部署反汇编：board `.A-q4kON`(a5e927bd) `tcrv_emitc_..._gemm/gevm_q4_K_q8_K`（本地 /tmp/q4k_gemm_REAL.txt, q4k_gevm_REAL.txt）。
- 同源双编译器：board `/tmp/t4b_seal_fix/gemm_q4_K.inc`(md5 90d454da) × {gcc-15.2.0, clang-17.0.6}（本地 /tmp/dis_*.txt）。
- micro 对照数：`experiments/active/l1-tile-s6-q4k-repack-gemm/tile_s6_findings.md`（clang-17 71/3）。
- Amdahl 原账：`experiments/active/kquant-family-closure/transmission_account.md`（1.59× 用 g₄=1.884）。
- salvaged 原始 e2e：board `/tmp/tf_paired/{RUN.log,clean/*.json,neighbor_snapshots.log}`；本地耐久副本见
  `experiments/active/result-tables/T-PERF1b_q4k_e2e_prefill_regression.md`。
- seal-fix 部署流程：`tools/e2e-harness/board/t4b-seal-fix/board_deploy.sh`（gcc-15 -O3 build，"vl=8 验封"仅查
  e32/m2 fold 正确、从未查 vsetvli/spill 计数）。
