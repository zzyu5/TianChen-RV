# G8 §六.3 攻坚 · k1(VLEN256) M=1 GEVM 输格真攻坚 3 组 (2026-07-15)

> **战果（如实）**：三组皆 **真解剖 + 真构造 + 真测**。**无 deployed-cell 翻正**（三组的部署核在 k1 VLEN256 全 < 0.8）。
> 但 **q5_0/q5_1 出一个强 named-X / 条件翻正**：部署核=block-dot（编译器前门在 VLEN256 decode **主动 decline repack**）→ §六 测对、FAIL 立；**然而** 被 decline 的 repack-GEVM leaf **byte-exact 且 cold 1.19×/1.31× 赢对手**——翻正被一个 **q4_0-keyed 的 selector decline gate 挡住，非硬件墙**（可检验、一处 capability-key 即达）。
> iq1_s/iq1_m/iq4_nl 三格 = **真具名-X（体例四件齐）**，无 flip：iq1_s 的 rvv gather-widening lever **不 port 到 VLEN256（byte-exact 失败）**；iq1_m lever port 但仅 +6%；iq4_nl repack decline **被证成立**（repack 0.25× 比 block-dot 更差）。
>
> **口径**：clang-18.1.8 Bianbu 对称域（ours 与 stock opp 同 clang-18；ours 用 decree march `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zvl256b` -O3 -fno-integrated-as）· k1/X60/VLEN256 · core1 pin（idle 100%, gov=performance 1.6GHz）· 32MiB flush（64× L2, k1 无 L3, 无 cache-miss PMU）· median + relIQR · N≥10 冷态 · opp = stock `libggml-cpu.so`(md5 871169a0, 只读) 前门 dispatched vec_dot（真派发对手，非稻草人）。
> **byte-exact**：ours vs stock-opp AND vs 独立标量 oracle。**禁 git**（主会审后 commit）· 未改 T3_B/T8 · 主树/build/stock-.so/governor 未动 · scratch 已清。

---

## 0. Headline 结果表 (cold median · K=2048 nc=512 · N≥10 冷态 trials)

| 格 | 部署核 (deployed) | cold ratio (部署) | byte-exact | 被 decline/未部署 变体 | cold ratio (变体) | byte-exact | 裁决 |
|---|---|--:|:--:|---|--:|:--:|---|
| **q5_0@k1** | block-dot vec_dot | **0.381×** (§六 复现 0.383) | 0/512 | repack-mf2 GEVM leaf (front-door DECLINED) | **1.190×** | 0/512 | 部署 FAIL·**repack 赢但被 decline** → named-X/条件翻正 |
| **q5_1@k1** | block-dot vec_dot | **0.396×** (§六 复现 0.400) | 0/512¹ | repack-mf2 GEVM leaf (DECLINED) | **1.306×** | 0/512¹ | 同上 |
| **iq1_s@k1** | block-dot grid-gather | **0.394×** (census 0.367²) | 0/512 | wide (rvv gather-widen lever) | 0.483× **INVALID** | **512/512 mism** | LOSS·**lever 不 port VLEN256** → named-X |
| **iq1_m@k1** | block-dot grid-gather | **0.383×** (census 0.247²) | 0/512 | wide (rvv grid-gather-widen) | **0.405×** (+6%) | 0/512 | LOSS·lever port 但不足 → named-X |
| **iq4_nl@k1** | block-dot codebook | **0.693×** (census 0.694 复现) | 0/512 | repack-mf2 GEVM leaf (DECLINED) | **0.248×** (更差) | 0/512 | LOSS·**repack decline 被证成立** → named-X |

¹ q5_1 随机 scale 下 rel>1e-3 判据 0 mism（max_rel≤5.6e-4 = 标量-vs-向量 FMA 结合序良性噪声，与 rvv casefile 同）；vs 独立 oracle 亦 0 mism。
² census(§六/G7 vecdot-k1) 用 stock-matching march（`...zicbop_zihintpause`，对称）；本役用 decree march（含 zba/zbb/zbc/zbs bit-manip）—— 该 march **favors ours**（bit-manip 加速标量 index，iq1_m 0.247→0.383），**ours 仍 LOSS** → LOSS 判决对 march 选择 robust（对称域会更差）。

---

## 1. ★ 决定性结构发现 — k1 VLEN256 decode 编译器【主动 decline repack】(与 rvv 相反)

**源级证据（`lib/Plugin/RVV/Selection/RVVContractionPathSelection.cpp:84-85, 143`）**：
```cpp
bool vlenOrPrefillFavorsRepack(int64_t minVLEN, MRegime m){ return minVLEN==128 || m==Prefill; }
// ... 否则:
return {ContractionAlgorithm::BlockDot, "block-dot-decline-q4_0-vlen256-decode-k1-loss"};
```
- **rvv(VLEN128) decode**：`vlenOrPrefillFavorsRepack(128,decode)=true` → 选 repack → 部署核=REDESIGN-B GEVM leaf（rvv q5x casefile 结论：§六 测错核）。
- **k1(VLEN256) decode**：`vlenOrPrefillFavorsRepack(256,decode)=false` → **decline repack** → 部署核=block-dot。decline 理由字面 `block-dot-decline-q4_0-vlen256-decode-k1-loss`，源注（同文件 :80-81）载「q4_0 @ VLEN256 decode measured 0.74x LOSS, so that decode cell is declined」——**一个 q4_0-keyed 的 blanket VLEN256-decode-decline**。
- 三向验证部署=block-dot：① 前门 `weft-opt ... march=rv64gcv_zvl256b` 对 q5_0/q5_1/iq4_nl 的抽象 quant_contraction decode op **fail-closed（selection 返回 BlockDot，RVVLowerQuantContraction 无 q5/iq4 block-dot decline path 故报错）**；② §六 k1 实测核符号 = `weft_emitc_..._rvv_q5_0_q8_0_block_dot`（block-dot）；③ 本役 block-dot cold **复现 §六**（q5_0 0.381 vs §六 0.383；iq4_nl 0.693 vs 0.694）= harness 自证。

**含义**：k1 的 q5/iq4_nl decode **部署核确是 block-dot（§六 测对，与 rvv 相反）**。但 decline 的正当性 **per-format 分裂**：q5 的 repack **赢**（decline 误伤）、iq4_nl 的 repack **输**（decline 正确）。blanket q4_0-key 过粗。

---

## 2. 组1 · q5_0/q5_1 — 部署 FAIL 立 + repack 赢被 decline (named-X / 条件翻正 · 体例四件)

### 2.1 对手符号 + 反汇编
`ggml_vec_dot_q5_0_q8_0` @0x9f81c / `ggml_vec_dot_q5_1_q8_1` @0x9f942（stock .so, native-RVV inline, §六机判 80/90 ins·32/30 rvv·csrr; 本役反汇编 `raw/opp_ggml_vec_dot_q5_*.s`）。5-bit qh 处理 + 紧凑 native block-dot。

### 2.2 差距构成逐项量化 (deployed block-dot vs opp)
ours block-dot（`raw/q5_0_blockdot_deployed.objdump.txt`, vsetvl=9）= **发散的 5-bit qh block-dot 变体**（≈2.6× 慢 = §六 [GAP-Q5x-QH-BLOCKDOT-EMIT]）：非 q4_0/q8_0 那种 parity-by-adoption（那里 ours 发 ggml 自家紧凑 block-dot），q5 的 qh emit 与对手紧凑 native 路径**发散**。→ 部署 block-dot 0.38-0.40× LOSS，§六 数字正确、立。

### 2.3 我方等价构造实际尝试 (force-construct 被 decline 的 repack leaf)
- 构造法：`weft-opt HEAD` 前门在 VLEN256 decline，故手工喂已过 selection 的 typed op（`raw/q5_*_typed_vlen128.mlir`, half_lanes=8 mf2）→ `--weft-rvv-lower-to-emitc` → `mlir-translate-20` → `kernels/q5_*_gevm_repack_mf2.c`（REDESIGN-B 签名 vlm_v_b16/vmnand·vsetvl 7/11）。
- **G1 byte-exact**：repack leaf vs stock-opp AND vs 独立 oracle **0/512 mism**（q5_0 bitwise, q5_1 rel<1e-3 max_rel 5.6e-4 良性 FMA 噪声）· K=2048/4096 全过。
- **G2 cold**（12 trials K2048N512 + 4 trials K4096N256, paired 32MiB-flush, core1）：**q5_0 repack median 1.190×（min 1.156 max 1.218, 12/12 >1.15）·K4096 1.18-1.28×**；**q5_1 repack median 1.306×（min 1.298 max 1.324）·K4096 1.37-1.39×**。IQR<12%。
- 物理机理：VLEN256 下 repack 的 contiguous x16 权重流 + 宽 SIMD 归约（1 fused GEVM call）比 block-dot 的 per-column 分散 5-bit 解码（512 vec_dot calls）**更吃 VLEN**——故 rvv(VLEN128) 只 near-parity(0.65-1.01, rvv q5x)、k1(VLEN256) 转决定性 win。**注**：measured 是 mf2(half_lanes=8) 次优形；VLEN256 前门本应发 half_lanes=16/m1（whole-LMUL），故 1.19/1.31 是 **下界**。

### 2.4 残余具名 X (可检验)
**X = q4_0-keyed 的 blanket VLEN256-decode-decline gate 误伤 q5**（`RVVContractionPathSelection.cpp:84-85`）。**非硬件墙**：赢的 kernel 已存在且 byte-exact（1.19/1.31 measured）。可检验：把 decline 改 per-format（q5_0/q5_1 在 VLEN256 decode measured WIN → 应 select repack），即翻正。**主会裁**：deployed-cell 严格判 = 维持 FAIL（部署核仍 block-dot）；但这是「一处 capability-key 即达的条件翻正」，**非硬件天花板 / 非靠次路躲门**——赢面已 byte-exact 证。翻正需 selector-key 改（主树改动，本役禁 commit）。

---

## 3. 组2 · iq1_s/iq1_m gather — 真具名-X (体例四件 · 无 flip)

### 3.1 对手符号 + 反汇编
`ggml_vec_dot_iq1_s_q8_K` @0xa2c8c / `ggml_vec_dot_iq1_m_q8_K` @0xa361e（§六机判：**VLEN-adaptive dispatcher → vectorized RVV gather**，iq1_s generic 149 rvv/68 gather——强对手非 scalar；入口为短 dispatcher，`raw/opp_ggml_vec_dot_iq1_*.s`）。

### 3.2 差距构成逐项量化 (baseline)
ours block-dot grid-gather（`raw/iq1_s_baseline.objdump.txt` vsetvl=79 gather=8; iq1_m vsetvl=171 gather=17）= §六 结论：标量 grid-index 合成 + 栈往返 + 窄 gather + 每组 tiny reduction(iq1_m)。cold LOSS。

### 3.3 我方等价构造实际尝试 (rvv gather-widening lever @VLEN256)
采 rvv acfdf18b 的 gather-widening kernels（`kernels/iq1_*.wide.kernel.c`，rvv 上 byte-exact 提升过）直接 VLEN256 重测：
- **iq1_s wide**：objdump vsetvl 79→45（gather 仍 8，**未如 rvv VLEN128 的 8→4**）· cold 0.394→0.483 · **但 G1 byte-exact 失败 512/512 mism, worst_ulp 2.3e9** → **INVALID**。根因：rvv wide 的 `vluxei16 i64m4` + `vget_v_i16m8_i16m4` 半拆逻辑 **烘死 VLEN128 元素数**；VLEN256 下 i64m4=16 元素(非8) → 越界读、半拆错位 → 算错。**rvv lever 不 port VLEN256**。
- **iq1_m wide**：grid-gather i64m1→i64m2（gather 17→9, vsetvl 171→155）· **G1 byte-exact 0/512** · cold 0.383→**0.405×（+6%）**——port 成功但**远不足** 0.8。

### 3.4 残余具名 X (可检验)
- **iq1_s X = ① rvv gather-widening lever VLEN128-baked（半拆假设烘死元素数）不 port VLEN256（byte-exact 隔离证明）+ ② 真瓶颈仍 = 标量 grid-index 合成**（与 rvv 同）。翻正需 **VLEN256-correct 的 gather-widening + 向量 grid-index 合成原语**（结构级发射器 follow-up，非硬件墙——对手同板已向量化证可快）。
- **iq1_m X = grid-gather-widening 真但不足(+6%)；真瓶颈 = 每组 8-element tiny reduction ×N + 标量 index**（与 rvv 同；grid gather 减半仅 +6% = 隔离证明瓶颈不在 grid gather）。发射器成熟度-X。
- **march 披露**：decree march 的 bit-manip **favors ours**（iq1_m 0.247→0.383），ours 仍 LOSS → 判决 robust。

---

## 4. 组3 · iq4_nl — 真具名-X (bounded · repack decline 被证成立)

### 4.1 对手符号 + 反汇编 + 成色
`ggml_vec_dot_iq4_nl_q8_0` @0xa7b44（§六机判：**dispatcher → `_vl256` RVV native-vec, 79 ins, gather-4**）。**对手成色 = native-vec-mid**（真 vl256 specialization，非最强 hand-brick、非 scalar；另注 ggml 另出 `ggml_gemm_iq4_nl_16x1/8x8/4x4_q8_0` repack-GEMM 家族 = format 真 beat 在 GEMM/prefill 轴，非本 decode vec_dot 轴）。

### 4.2 差距构成逐项量化
ours block-dot（`raw/iq4_nl_blockdot.objdump.txt` vsetvl=5, 紧凑）cold **0.693×**（复现 census 0.694, byte-exact）· K4096 亦 0.69。gap = 16-entry codebook gather 效率 vs 对手 `_vl256` codebook 路径。

### 4.3 我方等价构造实际尝试 (repack leaf what-if · 一轮)
iq4_nl repack decode 在 VLEN256 **同样被 decline**（同 gate）；force-construct repack-mf2 leaf（`kernels/iq4_nl_repack_mf2.c`, entry `..._repack_gemv_...`, vsetvl=260 **gather=64**）：
- **G1 byte-exact**：repack leaf vs opp AND vs 独立 codebook oracle **0/512 mism max_rel=0**（q5 之外再证 harness 正确性）。
- **G2 cold**：repack leaf **0.248×**（K2048/K4096 稳定）—— **比部署 block-dot(0.693) 更差 2.8×**。

### 4.4 残余具名 X (可检验)
**X = iq4_nl repack-GEVM = codebook-gather-bound（vsetvl 260 / 64 gather），repack 的 contiguous 权重流【不敌】codebook gather 开销 → repack decline 对 iq4_nl 【正确】**。与 q5 **决定性对照**：同 blanket q4_0-key gate，q5 误伤(repack 赢)、iq4_nl 命中(repack 输)——**证 gate 应 per-format 而非 blanket**。iq4_nl deployed block-dot 0.69× LOSS 是 codebook-gather vs `_vl256` 的成熟度 gap（bounded，低价值，未深潜；真 beat 在 iq4_nl GEMM 轴 = 另轴）。

---

## 5. 综判 (供主会整合 · 诚实 · 翻正不承诺)

- **无 deployed-cell 翻正**（三组部署核 k1 VLEN256 全 <0.8）。§六 k1 的 q5/iq4_nl block-dot FAIL **测对**（与 rvv 的「测错核」相反——k1 VLEN256 decode 编译器 decline repack）。
- **q5_0/q5_1 = 最强结果**：deployed block-dot FAIL 立；**repack-GEVM leaf byte-exact 且 cold 1.19×/1.31× 赢真对手**，翻正被 **q4_0-keyed selector decline** 挡（一处 capability-key 即达、赢面已证、非硬件墙）。主会可判 = 「deployed 维持 FAIL + 记 named-X『repack 赢/被 decline/per-format-key 修法可翻正』」或「条件翻正 pending selector-key 改」。**不得**写成无条件 deployed-PASS。
- **iq1_s/iq1_m/iq4_nl = 真具名-X 无 flip**：iq1_s(lever 不 port VLEN256, byte-fail)、iq1_m(lever +6% 不足)、iq4_nl(repack decline 被证成立)。皆发射器成熟度/轴转移，非硬件天花板；对手成色如实标（iq1 强 vectorized RVV gather / iq4_nl native-vec-mid）。
- **★per-board 别外推**：本 = k1 VLEN256 数字。rvv(VLEN128) q5 repack 是部署核且 near-parity(0.65-1.01)、iq1 lever 在 VLEN128 byte-exact——**均不外推**。

---

## 6. 板卫生
- Board k1/X60/VLEN256；core1 pin（idle 100%, gov=performance 1.6GHz 未改）；loadavg 2.0-2.8（持续 co-tenant baseline·paired ratio 吸收，`top` 空闲）。
- stock `/data/k1build-stock/bin/libggml-cpu.so` md5 871169a0（前后一致、只读）；主树/build/governor 未动。
- scratch `/tmp/g8k1_q5x`, `/tmp/g8k1_iq`, `/tmp/g8k1_iq4nl` 抽取后清；无 stray 进程。
- 未 git commit；未改 T3_B/T8。casefile = 本目录（evidence + kernels/ + raw/{driver, run.log, build_seal, objdump, opp .s, typed mlir}）。
