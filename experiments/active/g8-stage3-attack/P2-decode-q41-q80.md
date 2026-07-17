# P2 decode-debt — q4_1 / q8_0 DECODE (M=1 GEVM) 构造 + 双板判定

> **任务**：P2 剩余构造/测量缺口中的 2 个 gemm_tile decode 格（q4_1 / q8_0 × rvv+k1 = 4 board-cells）。
> **赛道**：**GEMM-decode 轴 kernel-sym**（M=1 GEVM·kernel-axis MICRO）。**NOT e2e·NOT perf-covered·不入系统账**。[NG-4]。
> **口径**：cold 唯一·N=25 median+relIQR·2-seed·单世界 clang-18（PR-17）·对手身份反汇编探针（PR-49 gate）·ZERO-MODEL 独立 oracle·预注册判读（cold≥0.8=PASS / <0.8=具名-X+墙）·**0 样本不造数·便宜档禁称硬赢**。
> **测于**：2026-07-17 · rvv(VLEN128·64c·core8 idle=100%·gov=performance)。主树/build/stock `.so` 未改（cpu_md5 before==after `e85fceda`）·STRAY=0·无 git commit。
> **测量树状态**：`git HEAD=554b47649` branch `refactor/full-refactor-m1`；`ninja -C build-weft weft-opt` **EXIT=0 且 up-to-date**（weft-opt md5 `9491ff22` ts 2026-07-17 09:57:41）。**本役未改任何 `lib/` / `include/` 编译器源**，故未触发重链 —— 非 forced-rebuild，而是"无源变更 ⇒ 无需重建"。ODS `.inc` 每次重生属本树已知行为（memory: build-incremental-unreliable），不影响本役（无源变更）。

---

## 0. ★净结论

**★A2-batch4 §4 的 "q4_1 / q8_0 decode = BLOCKED-ON-CONSTRUCTION" 判定【已过时·本役证伪】。**
batch4 §4 原文载：*"q4_1 decode：无 clean weft-opt leaf 源，dataflow 用 retired monolithic op `repack_gemv_q4_1_q8_1`（lower-to-emitc 拒 exec.variant），front-door 输入缺"*；*"q8_0 decode：无 clean GEVM leaf，仅 dataflow(multi-module)，须补 q8_0 typed-region single-module 源"*。
**实测**：前门**现已**携带 `kNibbleQ41ScaleModel`（`"dual-fp16-per-block-d_x.d_y-plus-min"`）与 `kNibbleQ80ScaleModel`（`"dual-fp16-per-block-d_x.d_y-full-i8"`），`m_regime="decode"` 的 `quant_contraction` 请求**直接构造** `typed_repack_gemv_loop_body`。零 retired emitter、零 archived `.inc`。两个前门输入已入库为 lit（**2/2 PASS**）。

| 格 | rvv(VLEN128) | k1(VLEN256) |
|---|---|---|
| **q4_1** decode | **1.381×** PASS（前门真构造·genuine 部署候选路） | **构造 BLOCKED·具名·fail-closed**（见 §3） |
| **q8_0** decode | **7.86×** PASS（**便宜档**·对手病理·非硬赢） | **构造 BLOCKED·具名·fail-closed**（见 §3） |

**★成色诚实**：
- **q4_1 1.381× = 相对干净的中等胜**：对手 `ggml_vec_dot_q4_1_q8_1` = 真 kernel（44 插/14 向量·nibble unpack `vand`+`vsrl`、`vwmul`+`vwmacc`、`vwredsum`），tier=通用向量（与 recon 现值相符）。**非 hand-brick**。噪声干净（ours relIQR 0.5–0.6%）。
- **q8_0 7.86× = 便宜档（cheap-tier）·【禁称硬赢】**：对手 `ggml_vec_dot_q8_0_q8_0` = 真 kernel 但**极弱**（34 插/10 向量）。**病理证据（同形状 K=2048 N=512 对照）**：weight 字节比 q8_0:q4_1 = 34:20 = **1.70×**；**我方** 0.204→0.437ms = **2.14×**（≈1.70×，memory-bound，正常）；**对手** 0.282→3.430ms = **12.16×**（远超 1.70× = 病理）。⇒ **7.86× 由对手病理主导，非我方核质量**。与 memory `[vec_dot q8_0 rvv: 上游 VLEN128 破损]` 及 batch4 q4_0 6.909×=便宜档 同构。
- **0 verified hand-brick**（本役两格对手均为 stock block-dot 单实现折中/破损）。

---

## 1. ★对手身份反汇编实证（PR-49 gate·先反汇编再测）

反汇编源 = 各板 stock `libggml-cpu.so`（rvv: `/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin`），TAB-field-aware awk 抽**完整函数体**（禁 `grep -A<N>` 窗口截断·PR-49 前科）。

| 符号 | 地址 | 插/向量 | 判定 |
|---|---|---|---|
| `ggml_vec_dot_q4_1_q8_1` | `0xb92a4` | 44 / 14 | **REAL-KERNEL**（非 thunk） |
| `ggml_vec_dot_q4_1_q8_1_generic` | `0x6ca58` | 367 / 211 | REAL（fallback·未派发） |
| `ggml_vec_dot_q8_0_q8_0` | `0xb95bc` | 34 / 10 | **REAL-KERNEL**（非 thunk·极弱） |
| `ggml_vec_dot_q8_0_q8_0_generic` | `0x6d7d0` | 291 / 183 | REAL（fallback·未派发） |
| `ggml_gemv_q8_0_16x1_q8_0` | `0xc3ef0` | 271 / 171 | **REAL-KERNEL**（重型手调·同算子·见 §2） |
| OURS q4_1 leaf | (leaf.o) | 301 / 199 | 真向量·`fp16_libcall=0` |
| OURS q8_0 leaf | (leaf.o) | 92 / 32 | 真向量·`fp16_libcall=0` |

**★PR-49 同型陷阱在本域【确认存在但不在本 2 格】**：`ggml_vec_dot_mxfp4_q8_0` = **9 插 / 0 向量的 VLEN 分发 thunk**（`csrr a2,vlenb` → `vlenb==16 ? j _vl128 : j _vl256`）。recon 对 mxfp4 已正确指向 `_vl128`/`_vl256`（**recon 无误**）。q4_1/q8_0 **无** `_vl` 专化（仅 `_generic` 兜底），其无后缀符号即真派发核 —— 本役所测即真派发核。

**tier 与 recon 现值核对（只报不改·分档口径硬冻结）**：
- `gemm_tile|q4_1` recon = `ggml_vec_dot_q4_1_q8_1(CROSSOP)` / tier=通用向量 → **相符**。
- `gemm_tile|q8_0` recon = `FLAT block-dot` / tier=通用向量 → **相符**（符号 = `ggml_vec_dot_q8_0_q8_0`）。

---

## 2. ★recon 未覆盖的同算子对手（OPP-S）— 报告，不改 recon

`ggml_gemv_q8_0_16x1_q8_0`（+ `_4x4` / `_4x8`）**存在且是真核**（271 插/171 向量）。recon 对 `gemm_tile|q8_0` 只名 CROSSOP block-dot。按 P1-backfill7 对 iq4_nl 的 **OPP-S 先例**，同算子对手存在时必须报。
**本役实测判定 = OPP-S 在此 harness 内【不可测】（具名）**：driver 把**我方** x16 packed buffer 喂给 OPP-S，由 **ZERO-MODEL oracle 裁决布局兼容性** —— 结果 **mism=256/512（恰半数）· maxrel=1.00** ⇒ **ggml 的 16x1 repack 布局与我方 x16 布局字节不兼容**（非"我方核错"：同一 buffer 我方核 oracle 0/512）。
⇒ **拒绝对错误布局计时**（设计即 fail-closed，非事后解释）。要测 OPP-S 须先接 ggml 自己的 repack（`ggml_cpu_repack` 路径）产出其 16x1 buffer —— **剩余构造·具名**。
⇒ **纪律**：q8_0 decode 的 7.86× 是 **vs CROSSOP 弱对手**，**不得**外推为"赢同算子 16x1 手调核"。
**q4_1 无任何 `ggml_gemv_q4_1_*` 符号** ⇒ CROSSOP 是其唯一合法对手，recon 正确。

---

## 3. ★k1(VLEN256) 两格构造 BLOCKED — 具名机制（非"未攻认输"）

前门在 VLEN256 decode 对两格**均 fail-closed**，机制**各异**且**都不是我方懒惰**：

| 格 | march=rv64gcv_zvl256b 行为 | 机制 |
|---|---|---|
| **q4_1** | 选择器 DECLINE repack（`path_selection_reason="block-dot-decline-vlen256-decode-unmeasured"`·registry 无 q4_1 decode 实测），decline 分支构造 `weft_rvv.q4_0_q8_0_block_dot` 但**仍携带 q4_1 的 `...-plus-min` scale_model** → **q4_0 block-dot verifier 正确拒绝**：`error: 'weft_rvv.q4_0_q8_0_block_dot' op requires scale_model "dual-fp16-per-block-d_x.d_y"` | **decline 路径是 q4_0-only**。若放行将**静默丢弃 `m_x*s_y` MIN 项** = q4_1 miscompile。**verifier fail-closed = 正确行为**（[K-5] 纪律成立） |
| **q8_0** | 更早失败于能力检查：`error: ... q8_0 quant_contraction requires a repack-affording capability ...; there is no ... q8_0 block-dot decline path (the block-dot identity lowering is q4_0-nibble-only, which would MISCOMPILE ...)` | **q8_0 无 block-dot decline 路径**。若放行将把 full-int8 权重当 nibble 解 = miscompile |

**★结构级发现（C1/C3′ 素材）**：**VLEN256-decode 的 "decline to block-dot" 路径只对 q4_0 可表达**。任何非 q4_0 的 flat 格式在该 cell 无法 decline。q5_0/q5_1 之所以不撞此墙，**仅因它们 measured-BENEFICIAL 从而根本不走 decline 分支**（`kRepackVlen256DecodeMeasurements`）。⇒ 这不是 q4_1/q8_0 的偶发缺陷，是 decline 分支的**结构缺口**。
**两条 fail-closed 契约已入 lit 固化**（两个新测试的 VLEN256 `RUN: not weft-opt ... | FileCheck --check-prefix=VLEN256`）。

**k1 两格去向（具名·未做）**：唯一诚实路径 = **force-constructed what-if leaf**（承 batch4 q4_0@k1 2.575×=what-if 先例）：直接 author VLEN256 typed leaf（`half_lanes=16`），绕开 declining selector。**本役未做**（预算）。**禁**用 VLEN128 半宽 leaf 在 k1 上跑充数（那是 half-util what-if 的 what-if，非部署形状）。**registry 不动**（加 measurement 会驱动部署 = 循环论证 + 必问级）。

---

## 4. 逐格 cold / 判读（rvv·VLEN128·clang-18 单世界）

`ratio = opp_med / ours_med`（≥0.8=PASS）。K=2048·nc=512·N=25·2-seed·flush=224MiB(>rvv L3)·core8 pinned·gov=performance。

### q4_1 decode M=1 GEVM
| seed | ours_ms(relIQR) | oppX_ms(relIQR) | ratio | predreg |
|---|---:|---:|---:|:--:|
| 0xC0FFEE1 | 0.2042 (0.5%) | 0.2834 (1.5%) | **1.3879** | PASS |
| 0x1357ACE | 0.2045 (0.6%) | 0.2810 (0.1%) | **1.3740** | PASS |
| **median-of-seeds** | | | **1.381** | **PASS** |
- **正确门**：vs ORACLE mism=**0/512**（maxrel 1.56e-05 / 9.03e-05）；vs oppX mism=0/512。
- **成色**：对手 = 真 native block-dot（44/14·nibble-unpack 非 autovec-able）·**非 hand-brick**·**非便宜档但对手中等**。噪声干净、2-seed 一致（1.388/1.374 差 1%）。

### q8_0 decode M=1 GEVM
| seed | ours_ms(relIQR) | oppX_ms(relIQR) | ratio | predreg |
|---|---:|---:|---:|:--:|
| 0xC0FFEE1 | 0.4553 (10.2%) | 3.4361 (1.4%) | **7.5468** | PASS |
| 0x1357ACE | 0.4185 (6.3%) | 3.4238 (1.0%) | **8.1804** | PASS |
| **median-of-seeds** | | | **7.86** | **PASS** |
- **正确门**：vs ORACLE mism=**0/512**（maxrel 5.45e-06 / 1.24e-04）；vs oppX mism=**0/512 maxrel=0.00e+00**（**bit-exact**）。
- **★成色 = 便宜档·禁称硬赢**（§0 病理对照：对手 12.16× vs 我方 2.14× @1.70× 字节比）。
- **★噪声诚实**：ours relIQR **10.2%/6.3%** vs opp 1.0–1.4% = **我方 leaf 内生波动显著高于对手**，与 P1-backfill7 判 rvv VOID-NOISE 的现象**同型**（该役 ours 0.79–14.50% vs opp 0.11–0.38%）。**本役仍入账的理由（具名）**：ratio ≈7.9 距 0.8 门 **~10×**，10% 噪声在数学上无法翻转 verdict；且 2-seed(7.55/8.18) 同向。**若将来要把 q8_0 当"近门"数用，须先复现噪声**。

---

## 5. 反空心 / 污染纪律

- **3-arm anti-hollow（每格·verify 模式）**：
  - `INJECT=0` clean：ORACLE mism=0/512 ✓
  - `INJECT=1` oracle-input fault（`wb.qs[0]^=0x01`）：ORACLE **RED 1/512** ✓（门会咬）
  - `INJECT=2` DUT-output fault（`ours[N/2]+=1.0f`）：ORACLE+oppX **RED 1/512** ✓（门会咬）
  - ⇒ 证书非空心（承 cert-hardening 三要件：语料完备/输入路径同源/oracle 独立）。
- **oracle 独立性**：ZERO-MODEL 从 **PLAIN** 输入零复用重算（q4_1 = RAW unsigned nibble [0,15] + `d_x*d_y*sumi + m_x*s_y`；q8_0 = full int8 + `d_x*d_y*sumi`），**非**捕获 leaf 中间量、**非** vs-ggml-agreement。
- **perf gate**：oracle RED ⇒ **PERF SKIPPED**（driver 内硬编码·0 造数）。
- **repack 布局非猜测**：x16 layout 逐字节读自**生成的 C leaf** 地址算术（q4_1: qs@`64+j*16`、d@0/+16、m@+32/+48、act q8_1 qs@4/@20；q8_0: qs@`32+p*16`、d@0/+16、act qs@2）。**q4_1 无 `^0x88` XOR**（前门 stamp `weight_nibble_unsigned` = RAW [0,15]，偏置在独立 MIN 项）——若误抄 q4_0 的 XOR 会被 oracle 抓（未发生）。
- **restore**：rvv stock `libggml-cpu.so` md5 `e85fceda47606a115c5fbb0021817cdc` **before==after UNCHANGED**（只读·未触 .so）。STRAY=0。load-gate core8 idle=100% gov=performance。未 git add/commit。

---

## 6. T3 回填清单（★留主会话机算入库·本役不动 T3/recon）

**禁改**：`T3_master_rebuild.csv`（生成物）· recon 的 tier/分母 · ROADMAP · sealed 资产。下表仅供主会话按 `GEMM_DECODE` dict 机算入库。
recon 机制说明（本役查证）：`parse_t3` 按 `(op,fmt)` 建键**不含 regime**；`skip_t3` 显式排除 `gemm_tile+decode`；decode 值**只**来自硬编码 `GEMM_DECODE` dict（该 dict 现有 q4_0/q5_0/q5_1/q4_K/q2_K/q3_K/q5_K/q6_K/iq4_xs/iq2_xxs/iq2_xs/iq2_s/mxfp4/tq1_0/tq2_0/iq4_nl(k1) —— **独缺 q4_1 与 q8_0**，即本役 4 格 pending 的**唯一**成因)。

| op | format | regime | rvv_cold | rvv_disp | k1_cold | k1_disp | 成色 tag |
|---|---|---|---:|---|---:|---|---|
| gemm_tile | q4_1 | decode | **1.381** | PASS | — | **维持 pending** | 前门真构造·对手 native block-dot 中等·非 hand-brick·2-seed 1.388/1.374 |
| gemm_tile | q8_0 | decode | **7.86** | PASS | — | **维持 pending** | **便宜档**·对手病理(12.16× vs 我方 2.14× @1.70×字节)·禁称硬赢·ours relIQR 10.2%/6.3% 具名 |

- k1 两格 **维持 pending·禁填**（§3 fail-closed·0 造数）。
- provenance：`P2-decode-q41-q80-raw/{rvv_q4_1,rvv_q8_0}_{verify,measure}.log` + `{...}_seal.txt` + `GEN_SEAL.txt`。
- 赛道 tag：`matmul-kernel-sym`·**regime=decode**（≠prefill 行·禁混）。**NOT e2e·NOT perf-covered·禁互推**。

## durable files
- `test/Conversion/RVV/rvv-lower-quant-contraction-q4-1-decode-repack-gevm.mlir`（net-new·lit PASS）
- `test/Conversion/RVV/rvv-lower-quant-contraction-q8-0-decode-repack-gevm.mlir`（net-new·lit PASS）
- `experiments/active/g8-stage3-attack/P2-decode-q41-q80-raw/`（driver / runner / leaves / GEN_SEAL / logs / seals）
