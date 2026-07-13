# G7 §L1 货架A 全量 — k1-半 kernel-sym FLAT q4_1/q5_0/q5_1 @k1/VLEN256 (hot/cold·nr 穷举)

> **性质**：**第二赛道 kernel-axis** 对称 micro A/B（T9 §1.1 kernel-sym 台账派生·C3′ 证据·覆盖面）。
> **★区别 e2e dual-board**：q4_1/q5_0/q5_1@k1 的 **e2e dual-board 已做**（`g7-l3-flat-k1-e2e/*-netnew-deploy`·prefill 4.99/2.21/2.45× WIN·winner=我方 VLEN256 vl=16 kernel）。本任务补 **k1-半 kernel-axis micro**（T9 §1.1 此前只有 q4_0/q8_0@k1·§5 是 e2e 非 kernel-sym）→ **完成 k1-半全量表缺口**。
> **与 perf-covered 系统账（9/83·recon 权威）永不混算**·**非 e2e beat·非 sealed 8-gate Win**·[NG-4] 纪律。
> **板**：k1 / SpacemiT-X60 / VLEN256 (vlenb=32) / **core3 (L2 512KiB shared 0-3·无 L3)** / **clang-18.1.8 出货对称域** / gov=performance 1.6GHz。
> **约束遵守**：无 git · 无 schema/T8/ROADMAP 改动 · 无 rvv · board 纯 micro scratch（`/tmp/g7_l1_k1cold/`·已删）· **stock lib 只读**（md5 双证 UNCHANGED）· e2e 冻结令期间**只做 kernel A/B·未开 e2e**。

---

## 0. 净结论（★ 3 fmt × nr{4,16,64} = 9 cell 全 ≥parity·hot∧cold 双态）

**clang-18 对称主账（kernel-axis·N=12 rounds·median·relIQR）— 同形状锚 = nr=16（= FLAT 立账 shape）：**

| 格·板 (VLEN256 gemm) | 对手符号(机判) | 成色 | 折中态 | **HOT (nr16)** | **COLD (nr16)** | cold regime | ≥parity? |
|---|---|---|---|---:|---:|---|:--:|
| **q4_1 @k1** | `ggml_vec_dot_q4_1_q8_1`（0x9f774·public T·block-dot） | light-vec 弱对手 | **单实现折中**（k1 ships 0 q4_1 repack→block-dot） | **7.188×** | **7.109×** | compute-bound（0.31 GB/s） | ✅ |
| **q5_0 @k1** | `ggml_vec_dot_q5_0_q8_0`（0x9f81c·block-dot） | block-dot 中对手 | **单实现折中**（k1 ships 0 q5_0 repack） | **2.329×** | **2.326×** | compute-bound（0.09） | ✅ |
| **q5_1 @k1** | `ggml_vec_dot_q5_1_q8_1`（0x9f942·block-dot） | block-dot 中对手 | **单实现折中**（k1 ships 0 q5_1 repack） | **2.574×** | **2.571×** | compute-bound（0.11） | ✅ |

- **9/9 cell（3 fmt × nr{4,16,64}）全 ≥parity·hot∧cold·GATE=PASS**（ZERO-MODEL bit-exact·nbad=0 全 9·relerr_ours≈relerr_opp ~1e-5–1e-4 = f32 跨块 reduction ORDER 差·正确 kernel 特征）。
- **对手身份 = 机判探针**（`nm -D` → `ggml_vec_dot_qX_qY` public T 符号·NOT `_generic`·NOT hand-brick）。
- **折中态逐格 = 单实现折中**：k1 出货 **零 q4_1/q5_0/q5_1 repack**（build_seal `OPP_REPACK_SYMS=0`）→ generic block-dot 是唯一实现（同 rvv FLAT 立账·"k1 无 stock repack→我方 emitted repack = 唯一 repack→预期 ≥parity"兑现）。

---

## 1. 复用已部署 VLEN256 vl=16 emitted repack（md5 双证·区别 rvv vl=8）

- ours = **e2e dual-board 已部署的同一 .inc**（`g7-l3-flat-k1-e2e/*-netnew-deploy/weft_emitted_gemm_qX.inc`·md5 逐字核对·板上 md5 == casefile）：
  - q4_1 = `736a716c…` · q5_0 = `7b1ac4e6…` · q5_1 = `03866f50…`（build_seal `OURS_SRC` 逐行）。
- **vl=16 源级/objdump seal（部署==证过·VLEN256-native full-width 16-lane strip·非 rvv vl=8 半宽）**：AVL literal `, 16)`×53(q4_1) / `, 8)`×0。objdump 结构（clang-18 -O3 .o·部署编译器 spill 轴·G1 精化②）：

| ours 核 -O3 | vsetvl | vwmacc | spill-ish | fp16 | size |
|---|--:|--:|--:|---|--:|
| q4_1 | 18 | 8 | 52 | **cleanfp-native-zfh**（0 `__extendhfsf2`） | 2864B |
| q5_0 | 13 | 8 | 60 | cleanfp-native-zfh | 2872B |
| q5_1 | 21 | 8 | 64 | cleanfp-native-zfh | 2960B |

- **libcall-free**（zfh/zvfh native fp16·march=`rv64gcv_zfh_zvfh_zicbop_zihintpause` == stock ggml-cpu compile_commands march·对称）。vwmacc=8/核 = 紧凑 vl=16（对照 K-quant PLAIN full-unroll vwmacc 2176-2304/spill 627-1058 爆·[GAP-P1]·本 FLAT 5 格无 spill 病）。
- **★layout 同构证**：driver `repack_w()`（block_qX_1x16·16-col group·d@0/m@32/nibble@64·q5_0/q5_1 transposed-qh@288/320）**字节等价 rvv VLEN128 与 k1 VLEN256**——仅 emitted kernel 的 strip-read-width 差（vl=8 两半 vs vl=16 一条）·内存布局同。ZERO-MODEL 数值门（fp64 独立 scalar decoder vs ours vs opp·nbad=0）是 byte-exact 安全网（若 layout 失配 gate 必 FAIL）。

---

## 2. hot/cold + nr 穷举（板测·N=12 rounds·T-N·load-gate·loadavg 2.21→2.73·pin core3）

**同形状 GEMM·K=2048·nc=512·pool P=8（ws 5.2–6.3 MB >> k1 L2 512KiB·无 L3→真 cold）·median-of-12·relIQR：**

| 格@k1 | 对手符号(机判) | 折中态 | HOT nr16 | COLD nr16 | nr4 H/C | nr64 H/C | cold≥parity | regime |
|---|---|---|--:|--:|--:|--:|:--:|---|
| **q4_1** | ggml_vec_dot_q4_1_q8_1 | 单实现折中 | **7.188×** | 7.109× | 7.136 / 7.008 | 7.092 / 7.061 | ✅ | compute-bound |
| **q5_0** | ggml_vec_dot_q5_0_q8_0 | 单实现折中 | **2.329×** | 2.326× | 2.312 / 2.302 | 2.312 / 2.308 | ✅ | compute-bound |
| **q5_1** | ggml_vec_dot_q5_1_q8_1 | 单实现折中 | **2.574×** | 2.571× | 2.575 / 2.571 | 2.552 / 2.549 | ✅ | compute-bound |

- **全 9 cell relIQR < 0.72%**（ours 侧 0.05–0.71% · opp 侧 0.03–0.52%）→ 非重叠分布·min-A/B > 1·决定性 ≥parity·非噪声。
- **★hot≈cold = NULL 区分**（匹配 k1-batch §6 铁律）：k1 L2=512KiB < weight tile 640–768KiB → "hot" 本非真 warm·compute-bound（dequant+q8_1 recon）→ 缓存态仅二阶。**无 rvv 的 nr-keyed cold-penalty 结构**（rvv 有 64MiB L3·q4_1 nr4 6.70→5.04 压缩；k1 无 L3·q4_1 nr4 7.14→7.01 几乎平）——**跨板机制差**：k1 无 L3 → 全 nr cold≈hot flat。
- **nr↓ 未翻**（最"memory-exposed" q4_1 nr4 cold wGBs=1.22·仍 7.008× ≥parity）。q5_0/q5_1 win 对 nr 稳（2.30–2.33 / 2.55–2.58·跨 nr flat）。

---

## 3. 变体穷举（令 §二.4·T4b selector 自体 oracle）

- **合法 codegen 候选集（VLEN256 rv64gcv）= VLA mf/mf2 单一**（VLEN-invariant·t4a 证 k1 vtype 描述符 == rvv128·[repack-winA-always-mf2]·m1=RVV0.7-only 非 same-board 候选）→ **selector 平凡最优**·无宽度旋钮可扫。
- **可枚举变体轴 = nr-shape {4,16,64}**（本测·全 9 cell hot∧cold ≥parity·[L-4] 合规）。**变体穷举净信息 = shape 轴**：vl=16 repack 跨 nr 全 ≥parity·win 对 nr 稳（区别 rvv q5_0/q5_1 的 win↗nr）。

---

## 4. 双账本 + 诚实 caveat

- **双账本**：k1 kernel-axis compiler = clang-18；stock ggml-cpu = clang-18（compile_commands 416 clang refs·0 gcc）⇒ **kernel账 == system账（对称·同域）**·[CASE-COMPILER-ASYMMETRY] not triggered。
- **★诚实 caveat（成色·同 rvv FLAT / e2e）**：对手 = **stock generic block-dot 折中态**（k1 无 riscv repack）·非 hand-brick。**7.19× / 2.33× / 2.57× = 我方 VLEN256 vl=16 repack vs 未优化 stock block-dot**（一个 kernel-axis L1 path-win·成色分层 = q4_1 light-vec 弱对手·q5_0/q5_1 block-dot 中对手·**均非 verified hand-brick win**）。verdict 立于 kernel-axis ≥parity（覆盖面/C3′ 证据）。
- **★kernel-axis ≠ e2e ≠ perf-covered**：本 9 cell 是 kernel-axis 对称覆盖面（T9 §1.1）·**禁写"加速 N kernel"于 e2e 语境**·perf-covered 系统账仍 9/83 不变（[GAP-FLAT-E2E] 黄格·T6 whole-model 后续）。
- **fair-play note**（同 e2e）：k1 tree 含 dormant `spacemit/repack.cpp` q4_1 变体·**NOT compiled**（stock lib 0 q4_1 repack syms）——若 SpacemiT 启用对手会更强；本测对手 = as-shipped compiled 路径 = block-dot（诚实基线）。

---

## 5. 污染 / restore（md5 双证 clean）

- **board 仅写 scratch** `/tmp/g7_l1_k1cold/`（driver + 3 .inc + 3 .o + bin + log·ephemeral·**已 rm 清**·`SCRATCH_CLEAN`）。**主树 / build / stock ggml .so / governor 全未改**。
- **stock lib md5 双证 UNCHANGED**：`/data/k1build-stock/bin/libggml-cpu.so` = `871169a0…`（before==after）· `/data/k1build/bin/libggml-cpu.so` = `14b6add6…`（== e2e OFF baseline·before==after）→ **只读·无需 restore**。
- **stray procs = 0**（`pgrep flatcold`=0·`pgrep llama`=0·无 hung llama-cli 遗留[前科已避]）。loadavg_final 2.54（系统 daemon 基线·同 begin 2.21）。

---

## 6. verdict + T9 更新建议（留主会话裁·本 agent 不改 T9）

**★结论**：q4_1/q5_0/q5_1 @k1/VLEN256 kernel-sym = **3 格全 ≥parity（hot∧cold·nr 穷举 9/9 cell·clang-18 对称·ZERO-MODEL PASS·对手=stock block-dot 折中态）** → **完成 T9 §1.1 k1-半 FLAT 缺口收口**（此前 k1-半只有 q4_0/q8_0·§5 是 e2e dual-board 非 kernel-sym）。

**kernel-sym ≥parity 计数更新建议**：**9 → 12**（+3：q4_1@k1 · q5_0@k1 · q5_1@k1·kernel-axis）。
- 成色旁标（令六 lint·计数 ≠ 成色）：+3 全 **block-dot/light 弱—中对手**（q4_1 light-vec·q5_0/q5_1 block-dot 中对手·**0 新 verified hand-brick**）。计数=第二赛道 kernel-axis 对称覆盖面（C3′ 证据）·**非 e2e·非 perf-covered 9/83·禁互推**。
- **★与 e2e dual-board 区别标注**：这 3 格 e2e dual-board 已绿（§5·winner=我方 kernel）；本补的是 **kernel-axis micro**（第二赛道·kernel-sym 计数）——两账并存不混（e2e≠kernel-sym）。

**建议 T9 §1.1 新增 3 行**（本 agent 不动 T9·主会话填）：
| 格·板 | emitted kernel 指针 | 对手 kernel 身份（成色） | 对称编译 | micro A/B |
|---|---|---|---|---|
| q4_1 @k1/VLEN256 (gemm) | VLEN256 vl=16 emitted repack（q8_1 家族·736a716c） | `ggml_vec_dot_q4_1_q8_1` block-dot（light-vec 弱·折中态单实现） | ✓ clang-18 双侧对称 | ✓ **7.188× hot / 7.109× cold ≥parity**（N=12·relIQR<0.72%） |
| q5_0 @k1/VLEN256 (gemm) | VLEN256 vl=16 emitted repack（make_block_q5_0x16·transposed-qh·7b1ac4e6） | `ggml_vec_dot_q5_0_q8_0` block-dot（中对手·折中态） | ✓ clang-18 对称 | ✓ **2.329× hot / 2.326× cold ≥parity** |
| q5_1 @k1/VLEN256 (gemm) | VLEN256 vl=16 emitted repack（block_q8_1x4·transposed-qh·03866f50） | `ggml_vec_dot_q5_1_q8_1` block-dot（中对手·折中态） | ✓ clang-18 对称 | ✓ **2.574× hot / 2.571× cold ≥parity** |

---

## 7. 进度分数（货架A 全量·§六.1 分数化）

- **k1-半 §1.1 kernel-sym**：此前 = q4_0/q8_0@k1 (2 存量·self/internal-control 口径) + q4_K/q5_K@k1 (2) = 4 filled；**本 batch +3（q4_1/q5_0/q5_1@k1·vs-block-dot 干净对·kernel-axis micro·hot/cold+nr 穷举）** → **k1-半 FLAT 缺口 CLOSED**。
- **kernel-sym ≥parity 计数（第二常驻·跨板）**：9 → **12（建议·主会话裁）**。
- rvv-半 §1.1 = 5/5 DONE（`f8da2f5c`/`668f1d45`）· k1-半 FLAT = 本 batch DONE。**FLAT 家族 kernel-sym 双板全量收口**（rvv 5 + k1 5[q4_0/q8_0 存量 + q4_1/q5_0/q5_1 本 batch]）。
- **纪律边界**：本 9 cell ≥parity **仅 kernel-axis 覆盖面·NOT e2e·NOT perf-covered 9/83**·系统账仍 [GAP-FLAT-E2E] 黄。

## durable files
- `evidence.md`（本文）· `flat_gemm_cold_driver_k1.c`（3-fmt port·repack_w 同构·ZERO-MODEL gate）· `k1_build.sh` · `k1_measure.sh`
- `raw/run.log`（build+9 cell hot/cold JSON·loadavg·gate）· `raw/build_seal.txt`（objdump vl=16 seal·opponent 机判·stock md5 双证·folded 到 run 已删的 board scratch）
- ours .inc 复用 `../../g7-l3-flat-k1-e2e/{q4_1,q5_0,q5_1}-netnew-deploy/weft_emitted_gemm_qX.inc`（md5 736a716c/7b1ac4e6/03866f50·未拷入本 casefile 避重·板上 md5 双证 == casefile）
