# G7 L1 货架A 全量 — kernel-sym FLAT-5 hot/cold 双态 + nr-shape 变体穷举 @rvv/VLEN128

> **赛道**：第二赛道 **kernel-sym**（kernel-axis MICRO·同编译器/flags/march 对称）。**NOT e2e·NOT perf-covered·与 perf-covered 系统账（9/83）永不混算**。[NG-4] 纪律·e2e 冻结令期间未开 e2e。
> **验证阶梯层**：[VERIFY-LADDER] **G2 同形状 cold micro**（GEMM 格测 GEMM 形·**禁 cold-GEMM 冒充 decode 预测**——本案只作 prefill-GEMM cell 的 cold 覆盖·NOT decode 预测器）。
> **板**：rvv / localhost.localdomain / openEuler / VLEN128 (vlenb=16) / **core 8**（disjoint-pin 8-15·co-tenant 未在 8-15·未重启）/ gcc-15.2.0 出货对称域（rvv shipped=gcc-15 → kernel-axis==system-axis·[CASE-COMPILER-ASYMMETRY] not triggered）。
> **★板事实（cold 协议锚）**：rvv **L1d=64K · L2=2MiB · L3=64MiB**（k1 无 L3·仅 512K L2）→ cold pool footprint **必须 >64MiB LLC**。本案 pool=140·ws 82.6–110.1MiB（1.3–1.7× LLC）→ 真 cold。
> **测于**：2026-07-13 · loadavg ~2.7（co-tenant vLLM + kernspan 在别的核·within-proc paired ratio 抵消共同 contention）。**禁 git·数据给主会话**。

---

## 0. 净结论（★ FLAT-5 × nr{4,16,64} = 15 cell 全 ≥parity·hot∧cold 双态·双账本）

**gcc-15 对称主账（kernel-axis·N=12·median·relIQR）— 同形状锚 = nr=16（= FLAT-5 hot 立账 shape）：**

| 格·板 (VLEN128 gemm) | 对手符号(机判) | 成色 | **HOT ratio (nr16)** | **COLD ratio (nr16)** | cold regime | ≥parity? |
|---|---|---|---:|---:|---|:--:|
| **q4_0 @rvv** | `ggml_vec_dot_q4_0_q8_0`（0x91854·public T·block-dot） | light-vec(opp 15 insn·弱) | **6.780×** | **6.224×** | compute-bound（cold 0.27 GB/s） | ✅ |
| **q4_1 @rvv** | `ggml_vec_dot_q4_1_q8_1`（0x918e4·block-dot） | light-vec(opp 13·弱) | **6.747×** | **6.105×** | compute-bound（0.30） | ✅ |
| **q5_0 @rvv** | `ggml_vec_dot_q5_0_q8_0`（0x91984·block-dot） | **better-vec(opp 36·较强)** | **1.228×** | **1.232×** | compute-bound（0.17·cold≈hot） | ✅ |
| **q5_1 @rvv** | `ggml_vec_dot_q5_1_q8_1`（0x91abe·block-dot） | **better-vec(opp 33·较强)** | **1.499×** | **1.412×** | compute-bound（0.18） | ✅ |
| **q8_0 @rvv (gevm-native i8@32)** | `ggml_vec_dot_q8_0_q8_0`（0x91c12·block-dot） | light-vec(opp 9·弱) | **4.193×** | **4.752×** | **memory-leaning（cold 1.23 GB/s @nr4·cold>hot）** | ✅ |

- **15/15 cell（5 fmt × nr{4,16,64}）全 ≥parity·hot∧cold·GATE=PASS(ZERO-MODEL bit-exact·relerr 1e-5–1e-4·nbad=0 全 15)。**
- **对手身份 = 机判探针**（nm -D → `ggml_vec_dot_qX_qY` public T 符号·NOT `_generic`·NOT hand-brick·符号映射机判·重编令二.2 稻草人事故修复合规）。
- **对手成色 objdump 对称双证**（rvv-insn·OURS gcc-15 .o vs OPP gcc-15 .so 符号切片）：OURS q4_0=46/q4_1=58/q5_0=66/q5_1=77/q8_0=37；OPP q4_0=15/q4_1=13/q5_0=36/q5_1=33/q8_0=9。**light-vec 对手(9–15)→big win(4–7×)·better-vec 对手(33–36)→modest win(1.2–1.5×)**（与 FLAT-5 hot 立账 ordering 一致）。

---

## 1. cold 协议口径（如何 cache-cold·rvv L3=64MiB 特有）

- **hot（对照）** = 单 weight tile（tile 0）·warmup 3·best-of-5 × hiters=6。
- **cold（新）** = **P=140 独立 weight tile POOL**·每 tile plain 足迹 0.58–0.79MiB·**pool ws 82.6–110.1 MiB >> 64MiB LLC**。每 round 先扫全池 OURS（P 次 call）再扫全池 OPP → 下一 round 回 tile0 时它已被 L3 驱逐 → **每 weight tile 从 DRAM cold-read**。activation 单份复用（decode 的"暖激活/冷权重"精确建模）。median-of-**ROUNDS=12**（≥10·T-N relIQR 见 §4）。
- **同编译器对称**：OURS kernel + driver + link 全 gcc-15（主账·rvv 出货编译器）·opponent = 板自有 `libggml-cpu.so` md5 `d1adc634…`（gcc-15 rv64gcv build·= FLAT-5/batch1 同一 .so·测前测后 md5 UNCHANGED = restore 双证）。
- **量全 mem-chain**：读权重(cold)→dequant→compute→写回·报 achieved **weight-GB/s**（plain 足迹分母）供 roofline 分类。

---

## 2. ★变体穷举（令 §二.4 一鱼三吃 → T4b selector 自体 oracle·[L-4] 合规）

### 2.1 codegen 变体候选集（LMUL/tile 宽）— VLEN128 rv64gcv = **{mf2} 单一合法**
- FLAT repack GEMM 的 codegen 形由 `--weft-rvv-lower-quant-contraction` **自选**：VLEN128 rv64gcv → half_lanes=8 → **mf2**·columnsPerPass==4·**NO integer_core_lmul**（MLIR 头明证）。
- **m1 whole-LMUL core 是 RVV0.7.1(`xtheadvector`) 专属分支**（passes.td:386-394·"RVV0.7.1 has NO fractional LMUL·whole-LMUL core is ONE 16-lane strip"）·**非本板(rv64gcv/RVV1.0) 合法 same-board 候选**（跨-ISA·板不跑 xtheadvector 码）→ 与 memory [repack-winA-always-mf2] 一致（"RVV1.0 repack gearbox 恒发 mf2·m1 只在 isRVV0p7 分支"）。
- ⇒ **codegen 候选集 = {mf2} 单一合法**·selector 平凡选中（min-cost-legal=唯一 legal）·**是否 selector 现选中 = ✅（mf2·唯一候选）**。tile-宽/columnsPerPass=4 亦 C1 bridge 固化（region 构造期 baked·非 export pipeline 旋钮）。

### 2.2 shape 变体穷举（nr-block M {4,16,64}）— 实测轴·T4b selector 自体 oracle
每格同批跑 nr∈{4,16,64}（K=2048 nc=512 固定）·**mf2-repack 在全部 shape 皆 ≥parity（min-cost-legal 跨 shape 验证）**：

| 格 | nr=4 HOT/COLD | nr=16 HOT/COLD（锚·同形状） | nr=64 HOT/COLD | selector 现选中 | T4b 判 |
|---|---|---|---|:---:|---|
| q4_0 | 6.672 / 5.091 | **6.780 / 6.224** | 6.670 / 6.558 | mf2(唯一) | ✅ mf2 每 shape ≥parity |
| q4_1 | 6.699 / 5.043 | **6.747 / 6.105** | 6.701 / 6.504 | mf2(唯一) | ✅ |
| q5_0 | 1.215 / 1.216 | **1.228 / 1.232** | 1.408 / 1.418 | mf2(唯一) | ✅ **win↗nr**（better-vec opp 高 nr 掉更多） |
| q5_1 | 1.356 / 1.124 | **1.499 / 1.412** | 1.427 / 1.435 | mf2(唯一) | ✅ |
| q8_0 | 4.679 / **6.483** | **4.193 / 4.752** | 4.141 / 4.267 | mf2(唯一) | ✅ **cold>hot**（见 §4） |

> **★T4b oracle 结论**：codegen 候选集 {mf2} 单一·selector 无可翻盘（对手轴才是 win 大小的决定项·非我方 codegen 变体选择）。**变体穷举的净信息 = shape 轴（nr）**：mf2-repack 跨全 shape ≥parity·且 q5_0/q5_1 的 win **随 nr 增长**（nr4→64: q5_0 1.22→1.42·better-vec block-dot 在高 nr throughput 掉 3.17→2.72 gmacs·我方 stable 3.83）→ mf2-repack 对 nr 的 scaling 优于 better-vec block-dot。

---

## 3. 对手折中状态（供立柱一对账·T-PILLARS）

对手 @VLEN128 全 = factory block-dot（板 repack.cpp 证·FLAT-5 立账继承）·**折中状态逐格**：

| 格 | 对手符号(机判) | 对手折中状态 | 依据 |
|---|---|---|---|
| q4_0 | `ggml_vec_dot_q4_0_q8_0` | **单实现折中** | 上游有 VLEN256 repack(`q4_0_16x1`) 但 VLEN128 `case128:{break;}//TODO`→nullptr→block-dot（repack 存在别处·此 VLEN 未部署） |
| q8_0 | `ggml_vec_dot_q8_0_q8_0` | **破损** | 上游 VLEN128 repack 破损（FLAT-5 立账 "upstream VLEN128 repack 破损→block-dot"）→ opponent 意图 repack 不可用·退 block-dot |
| q4_1 | `ggml_vec_dot_q4_1_q8_1` | **单实现折中** | 无 stock repack any-VLEN → generic block-dot 唯一实现 |
| q5_0 | `ggml_vec_dot_q5_0_q8_0` | **单实现折中** | 无 stock repack → generic block-dot（但 better-vectorized·较强） |
| q5_1 | `ggml_vec_dot_q5_1_q8_1` | **单实现折中** | 无 stock repack → generic block-dot（better-vec·较强） |

> **无一格对手 = 无折中(full-effort hand-brick)**。全 5 格 opponent 是 block-dot 折中态（单实现或破损）→ **成色分层：light-vec 弱(q4_0/q4_1/q8_0)·better-vec 较强但仍 block-dot(q5_0/q5_1)·均非 hand-brick**（与 T9 §1.1 成色定稿一致·kernel-sym 9 格 0 verified hand-brick）。

---

## 4. roofline / cold regime + ★cold-penalty 机制发现（rvv vs k1 差异）

| 格 (nr16 锚) | cold ours GB/s | cold opp GB/s | 判定 |
|---|---:|---:|---|
| q4_0 | 0.269 | 0.043 | compute(dequant)-bound（远低 DRAM 墙 ~10-20GB/s） |
| q4_1 | 0.301 | 0.049 | compute-bound |
| q5_0 | 0.165 | 0.134 | compute-bound（两侧皆低·qh gather+better-vec） |
| q5_1 | 0.178 | 0.126 | compute-bound |
| q8_0 | 0.330 | 0.069 | memory-leaning（4 格最高·nr4 达 1.23 GB/s） |

### ★发现（rvv L3 特有·与 k1 L2 cold≈hot 对比）
1. **cold-penalty 键控 memory-exposure（nr↓ → penalty↑）**：q4_0/q4_1/q5_1 在 **nr=4 cold penalty 最大**（q4_0 6.67→5.09·q4_1 6.70→5.04·q5_1 1.36→1.12）·**nr↑ penalty→0**（nr=64: q4_0 6.67→6.56·几近 cold≈hot）。机制：低 nr = 低 weight-reuse → 快的赢家(ours)更 memory-exposed（cold achieved BW nr4 0.88 vs nr64 0.071 GB/s）→ cold DRAM 读咬 ours 更甚·而 compute-bound 的 light-vec opp 几乎无 cold 变化 → ratio 压缩。高 nr = weight decode 摊薄到多列 → compute-bound → cold≈hot。
2. **q8_0 反常 cold>hot**（nr4 HOT 4.68→COLD 6.48·nr16 4.19→4.75）：q8_0 weight 足迹最大（34B/block·tile 1.06MiB）→ **对手 block-dot（重复读 plain 权重）cold 退化 > ours** → 我方 ratio cold **上升**。唯一 cold 帮 ratio 的格。
3. **q5_0 cold≈hot 全 nr**（compute-bound·两侧带宽都低）——与 k1 L2 finding 同型（compute-bound → cache 态二阶）。
- **⇒ rvv 有真 L3(64MiB)·同形状 cold 揭示了 k1(无 L3)看不到的 nr-keyed cold-penalty 结构**·但 **15/15 全 ≥parity 不翻转**（cold 最严 penalty q5_1 nr4 仍 1.124× ≥parity）。

---

## 5. clang deploy 双账本（nr16 锚·[CASE-COMPILER-ASYMMETRY] 稳健性·verdict 不翻转）

| 格 | gcc-15 HOT/COLD（主·kernel-axis） | clang-18 HOT/COLD（deploy） | verdict 一致? |
|---|---|---|:---:|
| q4_0 | 6.780 / 6.224 | 6.621 / 5.852 | ✅ both ≥parity |
| q4_1 | 6.747 / 6.105 | 7.200 / 6.325（clang-ours 更快·CASE 方向） | ✅ |
| q5_0 | 1.228 / 1.232 | 1.267 / 1.199 | ✅ |
| q5_1 | 1.499 / 1.412 | 1.441 / 1.402 | ✅ |
| q8_0 | 4.193 / 4.752 | 3.533 / 4.059（clang-ours 更慢·与 FLAT-5 hot 3.47< gcc 4.08 同向） | ✅ |

- **两账本对全 5 格 hot∧cold 均 ≥parity**（10/10 cold-cell·verdict 对编译器身份稳健）。block-dot 编译器不敏感（[CASE-COMPILER-ASYMMETRY] ~1.002×）→ 差异全在 ours-side codegen。q8_0 clang-slower 与 FLAT-5 hot 立账 3.47×(clang) vs 4.08×(gcc) 同向再证。**q8_0 cold>hot 反常两账本皆存 = 结构性(对手 cold 退化)·非编译器 artifact**。

---

## 6. §1.2 <parity cells cold — **PENDING（板批第二 batch·fallback 令）**

任务 §1.2（q5_K/q2_K/q3_K/q6_K/iq4_nl @rvv gemm + iq/tq vec_dot 族）**本 batch 标 pending**（fallback 令："若格太多 timeout→先 §1.1 FLAT 5 cold+穷举·§1.2 标 pending"）。理由：这些 cell 需**独立 kernel + 独立 cold driver**（KQuant super-block repack GEMM · Codebook repack · vec_dot block-dot emit）·correctness-gate 复杂度高·与 FLAT-5 共享 driver 不可复用。

**regime 确认（从既有 T8/T9 数据·非本 batch 实测·cold 预期不改 verdict）**：
- q5_K@rvv 0.120× / q2_K@rvv 0.386× / q3_K@rvv ~0.18× / q6_K@rvv ~0.18× / iq4_nl@rvv prefill 0.217× = 全 **weight-recon/spill-bound LOSS**（T8 已分类·compute/weight-bound）→ cold 预期 **≈hot 或略差**（compute-bound → cache 态二阶·同 §4 发现①机制）·**不预期 regime 翻转**。iq/tq vec_dot 族 0.16–0.78× 皆 fraclmul-2elem scalarization + vsetvli storm（compute-bound）。
- **⇒ §1.2 cold micro 实测排队第二 batch**（每格独立 driver）·主会话可据既有 hot LOSS + compute-bound 分类先记 "cold≈hot 预期·LOSS 维持"·实测补齐后 upgrade。

---

## 7. correctness + 污染纪律 + restore

- **ZERO-MODEL 数值门（timing 前·in-driver·tile 0）**：独立 fp64 scalar decoder（零复用我方 intermediate）vs ggml block-dot vs 我方 repack kernel；`relerr_ours ≈ relerr_opp`（1e-5–1e-4 级·nbad=0）·GATE=PASS **全 15 cell + 5 clang cell**。integer dot bit-identical·仅 f32 跨块 reduction ORDER 差 → ~1e-5–1e-4 rel（正确 kernel 特征）。
- **污染纪律（rvv 共享板）**：disjoint-pin **core 8**（co-tenant vLLM/kernspan 在别核·测前测后 8-15 仅系统 daemon·未触碰/未重启）；**我方 kernel 侧 cold relIQR**：q4_0 0.54–1.43% / q4_1 0.31–1.70% / q5_0 0.07–0.14% / q5_1 1.63–2.30% / q8_0 0.13–1.29% → 全 <2.5%·分布收紧·非 co-tenant 交织。opp 侧 relIQR <0.4% 全格。**load 巡检**：loadavg ~2.7（我方单核 bench 自身负荷·taskset -c 8·非污染·load-gate 判别符合）。
- **restore**：board 仅写 scratch `/tmp/g7_l1_cold/`（driver + 5 kernel.c + .o + 2 bin + log·ephemeral）。**主树 / build / stock ggml .so / governor 全未改**（ggml .so md5 `d1adc634…` 测前测后 UNCHANGED = 双证）→ **无需 restore**（纯 micro）。**无 git add/commit**。
- **stray proc**：测后 `pgrep flatcold` = 空（无 hung llama-cli·无遗留 bench proc）。

---

## 8. per-cell 返回摘要（主会话入账用）+ T9 更新建议

**每格 {hot | cold(同形状 nr16) | regime | 对手符号(机判) | 折中状态 | 变体穷举}：**

| 格 | hot(nr16) | cold(nr16) | cold regime | 对手符号(机判) | 折中状态 | codegen 候选 | shape 变体(nr4/16/64 cold) |
|---|---:|---:|---|---|---|---|---|
| q4_0@rvv | 6.780× | 6.224× | compute-bound | `ggml_vec_dot_q4_0_q8_0` | 单实现折中 | {mf2}单一·selected✓ | 5.091/6.224/6.558 |
| q4_1@rvv | 6.747× | 6.105× | compute-bound | `ggml_vec_dot_q4_1_q8_1` | 单实现折中 | {mf2}单一·selected✓ | 5.043/6.105/6.504 |
| q5_0@rvv | 1.228× | 1.232× | compute-bound | `ggml_vec_dot_q5_0_q8_0` | 单实现折中 | {mf2}单一·selected✓ | 1.216/1.232/1.418 |
| q5_1@rvv | 1.499× | 1.412× | compute-bound | `ggml_vec_dot_q5_1_q8_1` | 单实现折中 | {mf2}单一·selected✓ | 1.124/1.412/1.435 |
| q8_0@rvv | 4.193× | 4.752× | memory-leaning | `ggml_vec_dot_q8_0_q8_0` | **破损** | {mf2}单一·selected✓ | 6.483/4.752/4.267 |

**T9 §1.1 cold 双态子列建议**（留主会话改·本 agent 不动 T9）——每行加 `{cold ratio(nr16 同形状) | cold regime | cold≥parity}`：

| 格·板 | hot(已有) | **cold(nr16·本 batch)** | cold regime | cold≥parity |
|---|---:|---:|---|:---:|
| q4_0@rvv | 6.707× | **6.224×** | compute-bound(0.27GB/s) | ✅ |
| q4_1@rvv | 6.829× | **6.105×** | compute-bound(0.30) | ✅ |
| q5_0@rvv | 1.221× | **1.232×** | compute-bound(0.17·cold≈hot) | ✅ |
| q5_1@rvv | 1.407× | **1.412×** | compute-bound(0.18) | ✅ |
| q8_0@rvv | 4.077× | **4.752×** | memory-leaning(cold>hot·对手退化) | ✅ |

**T4b oracle 建议**：FLAT-5@rvv codegen 候选集 = {mf2} 单一合法（m1=RVV0.7-only 非 same-board 候选）→ selector 平凡最优·每格 micro-最优=mf2·[L-4] 合规。**变体穷举净信息 = nr-shape 轴**（mf2-repack 跨 nr{4,16,64} 全 ≥parity）。

**进度分数**：rvv-半 §1.1 = **5/5 格 filled**（cold 双态 + nr-shape 变体穷举 + clang 双账本）。rvv-半 §1.2 = **0/~14 格**（pending·第二 batch·regime 已从既有数据确认 compute/weight-bound LOSS）。**rvv-半可填总 = 5(§1.1) DONE + 14(§1.2) pending**。

**纪律边界**：本 15 cell ≥parity **仅 kernel-axis 覆盖面·NOT e2e·NOT perf-covered 9/83**·禁写「加速 N kernel」于 e2e 语境·系统账仍 [GAP-FLAT-E2E] 黄格。第二赛道 kernel-sym ≥parity 计数不因本 batch 变（cold 是 hot 的 robustness 补强·非新格）。
