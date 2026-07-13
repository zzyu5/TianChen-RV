# G7 L1 P2 — GEVM plan 板 IPC 验证 (q4_K@rvv · decode M=1) · ★成功信号 NEGATIVE + 反汇编归因

> board `ssh rvv` openEuler VLEN128 64c · cores **8-15 disjoint-pin** · co-tenant vLLM(cores 0-7)未重启 · load 8-11(含我方 8-thread 自负荷)
> NO git · swap-only · **live .so 测后 byte-exact restore = 05a62e6a stock-pristine (CONFIRMED·md5 双证)**
> 承 P0(H1 结构损·字节触底 roofline) + P1(colgroup-tiled GEVM plan·byte-exact·commit 806a7cc1) · 依据 [K-10]/[PAT-2]/性能宪章规则1-2/[GAP-P1] re-roll trap
> **裁决：★成功信号 NEGATIVE** — colgroup-tiled GEVM plan 在 M=1 令 IPC **反向下降** (0.294→0.214)·cycles +45%·非布局税(字节仍 roofline)·= **register-resident bank 溢出反噬**(反汇编 534 spill vs 10·53×)。

## 0. 部署与口径 (承 P0·三 .so 交织净测)

| 变体 | .so (md5) | 编译 | GEVM(decode) plan | GEMM(prefill) plan |
|---|---|---|---|---|
| **STOCK** | q4kOFF.gcc (05a62e6a) | gcc-15 as-shipped | generic block-dot (roofline 参照) | block-dot |
| **OLD** | q4kON.clangrepack (915e6733) | clang-18 repack | **per-column** GEVM (P0·IPC 0.29) | 旧 GEMM |
| **NEW** | q4kON.gevmct.clangrepack (**8f48c116**) | clang-18 repack | **colgroup-tiled TG=2** (P1 806a7cc1) | 旧 GEMM (不变) |

- **NEW 构建 = 外科 mixed build**：gcc-15 base tree + 仅 2 个 repack TU(arch/riscv/repack.cpp 含 emitted kernel + repack.cpp GEN 门)以 clang-18 重编,g++ -shared relink → **clang-symmetric 于 OLD 915e6733**(隔离 GEVM plan 结构变·编译器对称)。GEMM(prefill)路径**逐字不变**=控制变量。(full clang tree build 因 `arch/riscv/quants.c` clang-18 汇编器拒 vsetvli 失败·与本变更无关·故沿用 P0 mixed 路径。)
- **口径承 P0**：`perf stat -e instructions,cycles,cache-misses` 包 `llama-bench -p 0 -n N -r 1 --no-warmup -t 8`·taskset **8-15**·两点差分 (N64−N16)/48 抵消模型装载·DRAM bytes = cache-misses×64(校准 +0.05%·进程私有)。3 pass 交织(STOCK/NEW/OLD 同 load 窗)。

## 1. correctness 硬门 (ZERO-MODEL byte-exact·承 P1 三证·板上复核) — **GREEN**

greedy `llama-completion --temp 0 --top-k 1 --seed 1 -n 24` · 3 prompts · A=NEW(colgroup-tiled) vs B=STOCK(block-dot):
- **3/3 prompt A==B 逐字节等** (byte-identical greedy decode)。
- engage: `TCRV G7-P2 EMITTED GEVM-CT(...TG=2) ENGAGED`(decode) + `TCRV G5-M2 EMITTED GEMM`(prefill) 均 fire · **nan=0**。
- **CORRECTNESS_GATE: GREEN** — P1 host ZERO-MODEL oracle(mismatch=0·证 A envelope 不变性 + 证 B 独立重算)在**部署集成级**复核通过。

## 2. ★perf 三测 (decode M=1 · per-token 两点差分 · 3-pass median)

| 变体 | bytes/tok | insn/tok | cyc/tok | **IPC** | decode t/s(实测 avg_ts) | vs stock |
|---|---|---|---|---|---|---|
| **STOCK** (roofline) | 4.406 GiB | 5.14 G | 9.95 G | **0.517** | **2.097** | 1.00× |
| **OLD** (per-column) | 4.389 GiB | 5.86 G | 19.96 G | **0.294** | **1.033** | 0.49× (=P0 0.50×) |
| **NEW** (colgroup-tiled) | 4.398 GiB | 6.18 G | 28.93 G | **0.214** | **0.705** | **0.34×** |

- **① e2e decode on/off**：NEW 0.705 t/s / STOCK 2.097 t/s = **0.336×**(< parity)。NEW **比 OLD per-column(0.49×)更慢** → GEVM plan **回归**。
- **② IPC + cycles**：IPC OLD 0.294 → NEW **0.214**(**反向·未向 roofline 0.517 收敛**)。cyc/tok NEW 28.93G vs OLD 19.96G = **+45%**(vs stock 2.91×)。
- **③ per-token DRAM 字节**：NEW 4.398 / stock 4.406 = **0.998×**(≈ roofline·**字节未回升**)→ **H2 布局税仍 REJECTED**·纯结构回归非字节。

**relIQR(cyc)**：STOCK 0.26% / OLD 0.53% / NEW 2.83% — 全稳。

## 3. ★成功信号判读 = **NEGATIVE**（触发预注册"<parity 反汇编归因"分支）

预注册成功信号 = 「IPC 0.29→≥0.52 收敛 ∧ 字节不回升」。实测：**字节不回升✓（0.998×）· 但 IPC 反向下降 0.294→0.214✗**。故 **GEVM plan 结构成功 = 否**。预注册反面动作：反汇编归因（结构没打中）。

## 4. 反汇编归因 (为何 colgroup-tiled 令 cycles 反增·objdump NEW vs OLD)

| 静态指令形态 (objdump) | NEW (colgroup-tiled TG=2) | OLD (per-column) | 比 |
|---|---|---|---|
| total insn lines | 5179 | 2328 | **2.22×** |
| **scalar stack ld/sd (sp-rel · SPILL)** | **534** | **10** | **★53×** |
| vector spill (vs*r.v/vl*re) | 12 | 8 | 1.5× |
| vsetvli | 112 | 57 | 2.0× |
| vle (vector loads) | 568 | 284 | 2.0× |
| vwmacc | 1120 | 560 | 2.0× |
| prefetch | 2 | 0 | — |
| branches | 7 | 7 | 1.0× |

**根因 = 候选① register-resident bank 溢出反噬（[GAP-P1] re-roll/register-pressure trap 同型）**：
- colgroup-tiled TG=2 把 2 个列组的 state（4 个 f32m2 累加器 bank + 2× 权重条带指针/scale + 共享激活）**同时常驻**跨 shared block body。叠加 q4_K 超块叶本身寄存器重（6-bit scale/min 解包 + split-32 dot + dual d/dmin fold）→ **爆 32-vreg 预算** → 编译器把 loop-invariant 标量/指针**每迭代 spill 回栈重载**：**534 sp-relative ld/sd vs OLD 10（53×）**。
- 静态核 TG=2 全展开 2.22×（vle/vwmacc/vsetvli 全翻倍）；**dynamic per-token insn 仅 +5.5%**（6.18 vs 5.86 G·迭代数减半×每迭代翻倍≈持平·+5.5% = spill 重载）。**杀伤全在 IPC**：M=1 memory-latency-bound、**无算力藏延迟**（P0 已证），534 spill 往返**串行化**流水 → IPC 0.294→0.214。
- **候选③ 激活共享空转坐实为"为何成本收不回"**：plan headline 收益 = 每 block 省 1 次 activation delta 读（"交织组多列共享输入"）。M=1 下该收益≈0（激活极小），却**付出 2× 寄存器压力 + 53× spill**成本 → 净负。**prefetch(2 hint) 可忽略**（非瓶颈·[GAP-REPACK-GEVM] 字节非墙已定）。
- **对齐性能宪章**：规则1（修性能前反汇编认瓶颈）——plan **没打中 M=1 memory-latency 瓶颈**·反而加寄存器压力；规则2（直觉投影不可信）——"共享激活+register-resident bank+prefetch→抬 IPC" 的**方向投影被板证伪**（TG=2 = 未 per-format 板测定的猜测）。

## 5. M 扫描 → selector 分流 + M* 实测事实 (供 P3·禁 if(M==1))

- **M=1 decode** (`-p0 -n8`)：`gevmct_banner=4 · gemm_banner=0` → **纯 GEVM plan·零 GEMM**。
- **M=8 prefill** (`-p8 -n0`)：`gemm_banner=8 · gevmct_banner=5` → **GEMM 主导**（8 组，4-行组走 gemm）**+ GEVM 尾余**（leftover 单行走 gevm）。
- **M* 实测事实**：ggml **自有** gemv-vs-gemm dispatch 在 **M=1↔M≥2 之间切换**（M=1 全 gemv/GEVM plan；M≥2 主 gemm + 尾 gevm）。P3 selector 写回**键控 ggml 形状 dispatch（nr/M）**·非硬编码 `if(M==1)`（[K-10] 选择层=形状事实∧能力事实）。⚠ 因 GEVM plan = decode 回归，**分流本身正确但被路由的 GEVM plan 需重设计**。

## 6. 污染判 (insn-based · 硬信号)

- **pin 确认**：全程 `taskset -c 8-15`（disjoint·vLLM 在 cores 0-7）。
- **insn/tok（进程私有·不受带宽争用）**：NEW 6.18G > OLD 5.86G（+5.5%）·跨 3 pass 一致 → **结构损真实**（非仅 cyc 膨胀·insn 也升）。
- **bytes/tok（进程私有 cache-misses）**：NEW 4.398 ≈ OLD 4.389 ≈ STOCK 4.406（全 roofline）→ NEW **未多流 DRAM** → 额外 cycles **非 DRAM 带宽争用**·是 spill/stall（L1 栈往返）。
- **交织控制**：三变体每 pass 同 load 窗（8-11）交织；OLD（同 clang·同 2× 工作量）同窗测得 0.294 稳 → load 对 OLD/NEW 对称。**NEW-vs-OLD delta（both clang-18·交织·同 load）= 干净结构信号**。relIQR 全 <3%。→ **回归 = 结构·非污染**。

## 7. schema label 建议 (主会话裁 · 我方仅建议)

- **q4_K@rvv decode 仍 LOSS·不转正**：colgroup-tiled GEVM plan(TG=2) 板证 = decode **回归**(0.50×→0.34×)·**不得部署**。
- **[PAT-2] P9 GEVM regime-plan** 的 colgroup-tiled/register-resident-bank 变体应标 **board-falsified@M=1(register-pressure spill)**·非"待测"。
- **P1 GEVM plan 构造(byte-exact·mechanized·[K-10] 结构级独立 plan·novelty)仍成立** — P2 NEGATIVE = **首版结构假设 perf 未兑现·需迭代**·非构造失败。**重设计方向（反 register-pressure）**：TG=1（不加宽 bank·只软件流水权重装载 + 深 prefetch·减 per-iter live set），或攻"削 M=1 重建指令"而非"加宽 bank"（bank 假设已证伪）。

## 8. 污染 + restore (板可逆·md5 双证)

- **restore = TRUE**：live md5 = **05a62e6a82857cdf247ba1933c695849**（== stock-pristine·systemacct cert-1）· source 逐字节 pristine（GEN=deb61a29·ARCH=99131cf7）· live q4_K/gevm_ct emitted 符号 = **0**· ARCHDIR 无 stray .inc · 无遗留 bench/build/measure 进程。**NO git · 无 source patch 残留 · mixed build 后 pristine gcc rebuild 复原**。
