# evidence — kernel-sym 台账首轮立账：FLAT 5 gemm@rvv/VLEN128 对称 kernel-axis micro A/B

> **赛道**：第二赛道 **kernel-sym**（kernel-axis MICRO·同编译器/flags/march 对称）。**NOT e2e·NOT perf-covered·与 perf-covered 系统账（9/83）永不混算**。
> **对手身份**：**factory block-dot（as-shipped·real·gcc-15.2 built libggml-cpu.so md5 `d1adc634…`）** = ggml 真 dispatched `ggml_vec_dot_qX_qY`（NOT `_generic`·NOT hand-tuned asm·NOT SELF/internal-A/B）。
> **补测目标**：T9 §1.3 「≥parity 待板批对称 micro 补测」5 格（q4_0/q4_1/q5_0/q5_1/q8_0）逐格独立 N≥10 对称 kernel-axis micro（此前部分缺：q8_0 有 item4/q4_0 有 prefill parity·q4_1/q5_0/q5_1 待补）。
> **状态**：BOARD-MEASURED 2026-07-13·rvv（localhost.localdomain·openEuler·VLEN128·64c·core 14 perf-gov 2.6GHz·vLLM co-tenant pinned 0,1·disjoint）。**禁 git·数据给主会话**。

---

## 0. 净结论（★逐格全 ≥parity·kernel-sym ≥parity 计数建议 4 → 9）

| 格·板 (VLEN128 gemm) | 对手身份(成色) | **kernel-sym ratio (gcc-15 对称·-O3·N=12)** | ratio_relIQR | ≥parity? |
|---|---|---:|---:|:--:|
| **q4_0 @rvv** | factory block-dot（VLEN128 upstream repack=nullptr→block-dot·light-vec 10 rvv-insn·弱对手） | **6.707×** | 0.51% | ✅ ≥parity |
| **q4_1 @rvv** | factory block-dot（无 stock repack any-VLEN→generic·light-vec 8 rvv-insn·弱对手） | **6.829×** | 0.84% | ✅ ≥parity |
| **q5_0 @rvv** | factory block-dot（无 stock repack→generic·**better-vec 26 rvv-insn·较强对手**） | **1.221×** | 5.35%¹ | ✅ ≥parity |
| **q5_1 @rvv** | factory block-dot（无 stock repack→generic·better-vec 24 rvv-insn·较强对手） | **1.407×** | 4.06%¹ | ✅ ≥parity |
| **q8_0 @rvv** | factory block-dot（VLEN128 upstream repack 破损→block-dot·light-vec 7 rvv-insn·弱对手） | **4.077×** | 0.52% | ✅ ≥parity |

**5/5 逐格独立 ≥parity·对称 gcc-15 双侧·N=12·correctness ZERO-MODEL GATE=PASS(bit-exact)·对手身份 = factory block-dot as-shipped（NOT SELF）。**

¹ q5_0/q5_1 的 ratio relIQR（4–5%）**全部来自对手侧方差**（opp_relIQR q5_0=5.32% / q5_1=4.16%），**我方 kernel 侧 relIQR <1%（0.25–0.57%·见 §3）** → 非板污染，是 better-vectorized q5 block-dot 对手固有 timing 抖动。即便取 **min ratio**（q5_0=1.206×·q5_1=1.350×）仍 comfortably ≥parity。

> **★kernel-sym ≥parity 计数建议**：现硬值 **4**（q4_K@k1 / q5_K@k1 / q4_0@k1-gemm-prefill / q8_0@k1）→ **补 FLAT 5 @rvv/VLEN128 = 9**（上限达成）。**成色分层必标**：本 5 格对手皆 **factory block-dot**（q4_0/q4_1/q8_0=light-vec 弱对手·q5_0/q5_1=better-vec 较强 block-dot），**均非 hand-brick 强对手**（唯一 hand-brick ≥parity 仍是 q4_K@k1）。计数入账改留主会话。

---

## 1. 三账本交叉稳健（verdict 不因 opt-level / 编译器身份翻转）

同一 5 kernel·同板·同 core·N=12·三配置全 GATE=PASS·全 ≥parity：

| 格 | **kernel-sym gcc-15 -O3**(主) | gcc-15 -O2(稳健复现) | deploy clang-18 -O3 | batch1(2be7f3d2·-O2·best-of-7) |
|---|---:|---:|---:|---:|
| q4_0 | 6.707× (IQR0.51%) | 6.725× (0.37%) | 6.629× (0.54%) | 6.76× |
| q4_1 | 6.829× (0.84%) | 6.814× (0.17%) | 7.158× (0.43%) | 6.86× |
| q5_0 | 1.221× (5.35%) | 1.288× (4.08%) | 1.251× (2.70%) | 1.23× |
| q5_1 | 1.407× (4.06%) | 1.393× (3.83%) | 1.397× (1.73%) | 1.41× |
| q8_0 | 4.077× (0.52%) | 4.073× (0.85%) | 3.470× (0.36%) | 4.10× |

- **主账 = gcc-15 -O3 对称**（rvv shipped=gcc-15 → kernel-axis==system-axis·[CASE-COMPILER-ASYMMETRY] **not triggered**·both KERNEL_CC & LINK_CC = gcc-15.2.0，见每行 harness header）。
- **gcc -O2 复现 batch1**（差 <2%）·**verdict 对 opt-level 稳健**（-O2/-O3 均 ≥parity）。
- **deploy(clang) 双账本**：q4_1 clang-ours 更快(7.16×)·q8_0 clang-ours 更慢(3.47×<gcc 4.08× — [CASE-COMPILER-ASYMMETRY] 方向·与 batch1 q8_0 clang 3.50 一致)；**两账本对全 5 格均 ≥parity**。

---

## 2. 对手身份 + opponent-absence（为何 block-dot 是 fair 对手）

- **对手 .so** = 板自有 `libggml-cpu.so` md5 `d1adc634c2ca04ffc389536b30d9d9c4`（gcc-15 rv64gcv build·= batch1 同一 .so，测前测后 md5 双证 UNCHANGED）。对手 symbol = arch-dispatched `ggml_vec_dot_qX_qY`（**public T·NOT `_generic` scalar·NOT hand-tuned asm**）。
- **opponent-absence（结构 L1 path 基础·板自有 repack.cpp 证）**：q4_0(:4589)/q8_0(:4710) `if riscv_v: switch(vlenb*8){case 128:{break;}//TODO; case 256:return &qX_16x1;}` → **@VLEN128 repack=nullptr→block-dot fallback**（VLEN256/k1 only）；q4_1/q5_0/q5_1 **无 riscv repack 分支** → 任何 VLEN 皆 block-dot。⇒ VLEN128 下 5 格 prefill mul_mat 全 dispatch block-dot = 本 cell 计时对手。
- **对手 gcc-15 provenance**：.so 位于 `build-gcc15-rv64gcv/`·md5 = batch1 已存证的 gcc-15 build（.comment 段已 strip·provenance 由路径+md5 identity+batch1 MANIFEST 继承）。block-dot 编译器不敏感（[CASE-COMPILER-ASYMMETRY] ~1.002× clang-vs-gcc）→ 对两账本皆为有效固定基线。
- **objdump 对称双证（rvv-insn·see raw/objdump_symmetry.txt）**：OURS(gcc-15 O3) q4_0=31/q4_1=46/q5_0=49/q5_1=63/q8_0=26 rvv-insn；OPP(gcc-15 .so) q4_0=10/q4_1=8/q5_0=26/q5_1=24/q8_0=7。**light-vec 对手(7–10)→big win(4–7×)·better-vec 对手(24–26)→modest win(1.2–1.4×)** — 解释 ratio spread。

---

## 3. correctness + 污染纪律（clean·disjoint-pin）

- **ZERO-MODEL 数值门（timing 前·in-driver）**：每 rep 独立 seed·随机 f16-scaled PLAIN 权重/激活块 → (a) 独立 fp64 scalar decoder（零复用我方 intermediate）(b) 对手 ggml block-dot (c) 我方 repack kernel；`relerr_ours ≈ relerr_opp`（q4_0 1.5e-5 / q8_0 1e-4 级·`nbad=0`·GATE=PASS 全 60 reps）= 我方与 ggml 自己的 block-dot **等精度 bit-exact**。
- **fp16-libcall-free OK** 全 5 格（无 `__truncsfhf2/__extendhfsf2` soft-fp confound）。
- **污染纪律（rvv 共享板）**：disjoint-pin core 14（vLLM co-tenant pinned 0,1·测前测后 `taskset` 双证 vLLM 仍在 0,1·**未重启/未触碰他人 vLLM job**）；**我方 kernel 侧 relIQR <1% 全 5 格**（q4_0 0.48/q4_1 0.85/q5_0 0.57/q5_1 0.26/q8_0 0.25%）→ core 14 measurement clean·无 co-tenant 交织；q5 的 ratio 抖动隔离为对手侧（§0 脚注¹）。governor=performance 锁 2.6GHz。
- **★load 巡检**：run 期 loadavg 4.2→5.2（+~1 = 我方单核 bench 自身负荷·taskset -c 14·非 co-tenant 污染·符合 load-12 教训判别）。

---

## 4. 纪律边界（[NG-4]·令六 lint·永不混算）

- **kernel-sym = 内核轴覆盖面（C3′ 证据·T3/T9 行）·NOT e2e 拉绿杠杆**。本 5 格 ≥parity **禁表述为系统收益·禁写「加速了 N 个 kernel」于 e2e 语境**。
- 八门口径：micro∧e2e **未过**（本为 MICRO/kernel-axis prefill GEMM·无 whole-model llama-bench）+ selector-e2e-routing 未过（front-door CONSTRUCT+select 但真 llama routing 未 exercise·q8_0 selector 历史 self-DECLINE）→ 系统账仍属 [GAP-FLAT-E2E] 黄格（T6 whole-model phase-split 是 high-value 后续）。**本案头只立 kernel-sym 第二赛道账，不动系统账 9/83。**
- 与 batch1（flat-covering-batch1·同 5 格·best-of-7）关系：本案 = **N=12 逐格独立 per-round+median+relIQR 对称 micro 补测**（batch1 是 headline best-of-7 双账本；本案补齐 T9 §1.3「逐格独立对称 kernel-axis micro 数」的 protocol 空缺，并加 gcc-O2 稳健复现 + 我方/对手侧 relIQR 分离证污染-free）。

---

## 5. per-cell 返回摘要（主会话入账用·9 字段）

| 字段 | q4_0 | q4_1 | q5_0 | q5_1 | q8_0 |
|---|---|---|---|---|---|
| emitted kernel 指针 | 净新 repack GEMM（XOR-0x88 signed-nibble·d[0:32]+nibbles@32） | 净新 repack GEMM（q8_1 家族·d+m[32:64]+nibbles@64） | 净新 repack GEMM（make_block_q5_0x16·transposed-qh 32×u16@288） | 净新 repack GEMM（block_q8_1x4·q5_1 transposed-qh@320） | 净新 repack GEVM（.inc=M1b·i8 quants@32） |
| 对手身份 | factory block-dot as-shipped | 同 | 同 | 同 | 同 |
| 对手成色 | light-vec(10) 弱 | light-vec(8) 弱 | better-vec(26) 较强 | better-vec(24) 较强 | light-vec(7) 弱 |
| 对称编译双证 | gcc-15 both·objdump✓ | gcc-15✓ | gcc-15✓ | gcc-15✓ | gcc-15✓ |
| micro A/B N | 12 | 12 | 12 | 12 | 12 |
| ratio_med (gcc-15 -O3) | 6.707× | 6.829× | 1.221× | 1.407× | 4.077× |
| ratio_relIQR | 0.51% | 0.84% | 5.35%(opp-driven) | 4.06%(opp-driven) | 0.52% |
| correctness | ZERO-MODEL PASS bit-exact | 同 | 同 | 同 | 同 |
| **verdict** | **≥parity** | **≥parity** | **≥parity** | **≥parity** | **≥parity** |

**kernel-sym ≥parity 计数建议：4 → 9**（入账改留主会话·成色须标 factory block-dot 非 hand-brick）。
