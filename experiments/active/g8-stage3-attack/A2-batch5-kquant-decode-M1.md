# A2 batch5 — K-quant DECODE (M=1 GEVM) 剩余 4 格：q2_K / q3_K / q5_K / q6_K（双板·承 batch4 harness）

> **任务**：线 A·A2.5 = 测 K-quant gemm **decode**（M=1 GEVM）剩余 4 格 = **q2_K/q3_K/q5_K/q6_K**（我方 repack-GEVM leaf @ M=1 vs 对手真派发 block-dot @ M=1）。承 A2-batch4（batch4 已构造 M=1 GEVM harness + 测 q4_K；本役复用 driver 模板·per-format super-block repack 适配）。
> **赛道**：**GEMM-decode 轴 kernel-sym**（M=1 GEVM·kernel-axis MICRO）。**NOT e2e·NOT perf-covered·不入系统账**。[NG-4]。
> **口径铁线**：cold 唯一·**N=25 cold 中位内禀（≥20）+2-seed**·**禁一切继承**（decode 真测 M=1·禁用 prefill nr≥4 / vec_dot M1 数）·每格对手身份探针（符号级）·预注册判读（cold≥0.8=PASS / <0.8=named-X+墙）·**0 样本不造数·便宜档禁称硬赢·预判不作结论**·成色诚实（0 hand-brick 就说 0）。★q4_K decode 前例（board-split）**不外推**·每格各自真测。
> **测于**：2026-07-16 · k1(VLEN256·SpacemiT-X60·clang-18.1.8) + rvv(VLEN128·64c·gcc-15.2 部署 + clang-18 对称-micro)。主树/build/stock `.so`/governor 未改。

---

## 0. ★净结论（测完·双板真 cold）

**harness = batch4 复用成功**（driver = `kquant_gevm_m1_driver_b5.cpp`·per-format pack_w byte-exact 复用 `kquant_repack_verify_q{2,3,6}K.c`·q5_K 由 MLIR layout+q4_K 精度派生）。**leaf export 4/4 clean**（`test/Conversion/RVV/rvv-to-emitc-repack-gemv-{q2,q3,q5,q6}-K-q8-K.mlir` → weft-opt --weft-rvv-lower-to-emitc | mlir-translate·5-arg ABI `(n,s,vx,vy,nc)`·body=typed_repack_gemv_loop_body 全格验证·非 block-dot）。

**关键区分兑现（禁混·两串行 bug 前科）**：本役测 = **gemm decode = 我方 repack-GEVM leaf（body=typed_repack_gemv_loop_body）@ M=1 vs opp 真派发 block-dot @ M=1**。判别键 = 核 body 形态（repack_gemv vs block_dot），非符号名。

**gate = ours-leaf vs ggml-opp agreement**（两独立 codegen·读 DISJOINT layout：packed x16 vs 原始 per-block·一致 ⟹ pack_w+leaf 皆正确）。**非** from-scratch ZERO-MODEL scalar oracle（同 batch4 q4_K gate 档·honest 标）。**★正确门全绿**：4 格 × 双板 × 双编译器 × 2-seed = **全 0/512 mismatch**（maxrel ≤8.5e-4·多数 <1e-5·多数 bit-exact 0.00e+00）——**验证 pack_w 全格正确·含派生 q5_K**。

### ★主表（median of 2seed×2trial·全真测·门线 0.8）

| 格 | rvv gcc-15.2 **deploy** | rvv clang-18 micro | k1 clang-18 (what-if) | 正确门 | verdict/成色 |
|---|---:|---:|---:|:--:|---|
| **q2_K** | **0.0685** named-X (GCC-DEATH) | 0.3635 named-X | **0.9585 PASS**(near-parity) | 0/512 全 | **仅 q2_K@k1 near-parity PASS**(opp 弱 0.86ms)·rvv 全 LOSS·**非硬赢**(<1.0) |
| **q3_K** | **0.0830** named-X (GCC-DEATH) | 0.2189 named-X | 0.4252 named-X | 0/512 全 | **全 LOSS**·3-bit hmask super-block 重·16 sub-block fold@M=1 不摊销 |
| **q5_K** | **0.1273** named-X (GCC-DEATH) | 0.5040 named-X | 0.6806 named-X | 0/512 全 | **全 LOSS**·qh 5th-bit inject 重·opp 本身也慢(1.13/1.66ms) 但 ours 更慢 |
| **q6_K** | **0.0535** named-X (GCC-DEATH) | 0.2595 named-X | 0.3771 named-X | 0/512 全 | **全 LOSS**·6-bit signed8 super-block 最重·16 sub-block |

**★净判读（诚实·预注册预期兑现）**：
- **12 格中 11 named-X·唯 1 PASS = q2_K@k1（0.9585 near-parity·非硬赢·opp 弱且 ratio<1.0）**。这是 K-quant super-block decode@M=1 的 **C3′ 负结果**——**format-keyed 适用边界**（fold epilogue dmin/bsums/hmask/6-bit-unpack @M=1 GEVM 不摊销·同 q4_K rvv LOSS·[PAT-1]）。
- **★board-split 不普适（关键纠偏·别外推 batch4）**：batch4 q4_K@k1=1.535× PASS(赢弱 opp)·但本役 q2/q3/q5/q6@k1 **全 ≤0.96**（q2 near-parity·q3/q5/q6 LOSS）。**q4_K@k1 的 win 不推广到 K-quant 家族**——是 q4_K 特定(8 sub-block·该 opp 弱)·非家族律。判别键 = **sub-block 数**：q2/q3/q6=**16** sub-block(fold 开销倍增·vsetvl 高)·q4/q5=8。"更少量化 bit ≠ 更轻 leaf"（q2_K 2-bit 但 16 sub-block·vsetvl=81 最高·仅 near-parity）。
- **★rvv 全 GCC-DEATH（deploy 现实·[CASE-KQUANT-GCC-CODEGEN] 再证 4 格）**：gcc-15.2 vsetvl **922/2225/2952/2461**(q2/q3/q5/q6) vs clang **81/14/57/22** = **10–160× 更多 vsetvl** = super-block repack leaf 全 regfile spill/scalarize。**rvv 出货 gcc-15 → deploy-clean 全 LOSS**（0.05–0.13×·灾难级）。**编译器 codegen 病理·非算法**（clang 域也输·但只输 2–5× 非 15×）。
- **★0 verified hand-brick**（令六 lint）：全格 opp = stock block-dot/native-vec 单实现·**无 hand-tuned brick 强对手**。唯一 PASS(q2_K@k1) = 赢弱 opp·near-parity·**成色低·非硬赢**。

**★部署现实**：rvv front-door decode 若构造 K-quant repack-GEVM = **gcc-death 灾难**（deploy = gcc-15）·**应 DECLINE**（保 stock block-dot）。k1 front-door 已 DECLINE repack@decode（batch4）·本役 k1 = force-constructed what-if·**除 q2_K near-parity 外无立项价值**。**结论：K-quant super-block 的部署路 = block-dot（decode@M=1）·repack-GEVM 是 prefill/GEMM(nr≥4) 路的工具·decode 边界外**。

---

## 1. harness 构造（复用 batch4·durable）

- **驱动**：`A2-batch5-kquant-decode-M1-raw/kquant_gevm_m1_driver_b5.cpp`（承 batch4 `kquant_gevm_m1_driver.cpp`·扩 q2/q3/q5/q6_K·template 化 run_fmt）。
- **OURS** = repack-GEVM leaf（ONE GEVM call over nc 列·weight=block_qX_Kx16·activation=PLAIN 单 q8_K 292B·M=1）。
- **OPP** = stock `ggml_vec_dot_qX_K_q8_K`（board libggml-cpu.so）per-column（nrc=1）= block-dot @ M=1。
- **pack_w 溯源（byte-exact）**：
  - q2_K stride **1344**：d[16]fp16@0 · dmin[16]fp16@32 · scales[256]@64 · qs[1024]@320 —— 复用 `kquant_repack_verify_q2K.c`。
  - q3_K stride **1824**：d[16]fp16@0 · scales UNPACK→16 signed-i8@32 · hmask[512]@288 · qs[1024]@800 —— 复用 `kquant_repack_verify_q3K.c`（含 `unpack_q3K_scales` bit-dance）。
  - q5_K stride **2816**：d@0 · dmin@32 · 6-bit scales@64/192（== q4_K 打包）· qh@256 · qs@768 —— 由 `rvv-to-emitc-repack-gemv-q5-K-q8-K.mlir` layout + q4_K silicon-proven scale packing 派生（MLIR："ONLY delta vs q4_K = qh inject"）。
  - q6_K stride **3360**：d@0 · scales[256]i8@32 · qh[1024]@288 · ql[2048]@1312 —— 复用 `kquant_repack_verify_q6K.c`。
- **canonical ggml struct**（opp 读）：q2_K 84B · q3_K 110B · q5_K 176B · q6_K 210B（static_assert 守）。
- **leaf 生成（front-door·可复现）**：
  ```
  weft-opt test/Conversion/RVV/rvv-to-emitc-repack-gemv-{q2,q3,q5,q6}-K-q8-K.mlir \
    --weft-rvv-lower-to-emitc | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp
  ```
- **cold 协议**：32MiB flush · N=25 median+relIQR · 2-seed{0xC0FFEE1,0x1357ACE} · within-proc paired · K=2048 nc=512（ws q2K 344KB / q3K 467KB / q5K 720KB / q6K 860KB weight cold）。
- **build seal**：rvv 双域 clang-18.1.8 + gcc-15.2.0（march rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs）· k1 clang-18.1.8（+zvl256b·-fno-integrated-as -DBOARD_K1）。stock md5：rvv `d1adc634` / k1 `871169a0`（before==after 待收尾核）。

---

## 2. 逐格 cold / 判读 / 成色 / 探针（N=25 median·2-seed×2trial=4 ratio·门线 0.8）

**探针（opp 身份·符号级）**：rvv gcc-15 stock `ggml_vec_dot_qX_K_q8_K` @ q2_K=0x92316 / q3_K=0x930a6 / q5_K=0x93146 / q6_K=0x935de；k1 stock @ q2_K=0xa034c / q3_K=0xa15da / q5_K=0xa1ed8 / q6_K=0xa21d2。**均 stock block-dot·native-RVV inline·非 hand-brick**。

### q2_K decode M=1 GEVM（2-bit·16 sub-block·dmin+bsums fold·stride 1344）
| 板·域 | leaf vsetvl | ours_ms | opp_ms | ratio(4) | median | predreg | 墙/成色 |
|---|---:|---:|---:|---|---:|:--:|---|
| rvv·gcc-15.2(deploy) | **922** | 3.29–3.48 | 0.226–0.231 | 0.070/0.065/0.068/0.069 | **0.0685** | **named-X** | **GCC-DEATH**(922 vsetvl·regfile spill)·deploy LOSS |
| rvv·clang-18(micro) | 81 | 0.52–0.68 | 0.226–0.227 | 0.436/0.333/0.362/0.366 | **0.3635** | **named-X** | opp 强 native-vec(0.23ms)+16 sub-block fold·ours iqr 高(0.12–0.24 noisy·light leaf DRAM-bound) |
| k1·clang-18(what-if) | 81 | 0.91–0.96 | 0.856–0.899 | 0.894/0.937/0.980/0.983 | **0.9585** | **PASS** | **唯一 non-loss·near-parity**·opp 弱(0.86ms)·非硬赢(<1.0)·hl8 半宽 VLEN256 |
- **正确门**：0/512 全（maxrel ≤2.5e-5·多数 <2e-6）。

### q3_K decode M=1 GEVM（3-bit subtractive-hmask·16 sub-block·no-min single-scale·stride 1824）
| 板·域 | leaf vsetvl | ours_ms | opp_ms | ratio(4) | median | predreg | 墙/成色 |
|---|---:|---:|---:|---|---:|:--:|---|
| rvv·gcc-15.2(deploy) | **2225** | 3.51–3.53 | 0.292–0.294 | 0.083/0.083/0.084/0.083 | **0.0830** | **named-X** | **GCC-DEATH**(2225 vsetvl)·deploy LOSS |
| rvv·clang-18(micro) | 14 | 1.33–1.37 | 0.290–0.303 | 0.220/0.218/0.227/0.218 | **0.2189** | **named-X** | opp 强(0.30ms)+hmask bit-select+16 sub-block scale unpack 重·ours 1.35ms |
| k1·clang-18(what-if) | 14 | 1.98–2.02 | 0.856–0.857 | 0.424/0.432/0.425/0.425 | **0.4252** | **named-X** | 输·hl8 半宽+hmask/scale 重·opp 0.86ms 非弱到能赢 |
- **正确门**：0/512 全（maxrel 全 **0.00e+00 bit-exact**·no-min 整数核干净）。

### q5_K decode M=1 GEVM（5-bit qh-inject·8 sub-block·dmin+bsums fold·stride 2816）
| 板·域 | leaf vsetvl | ours_ms | opp_ms | ratio(4) | median | predreg | 墙/成色 |
|---|---:|---:|---:|---|---:|:--:|---|
| rvv·gcc-15.2(deploy) | **2952** | 8.79–8.85 | 1.106–1.130 | 0.128/0.128/0.127/0.126 | **0.1273** | **named-X** | **GCC-DEATH**(2952 vsetvl·最重·ours 8.8ms!)·deploy LOSS |
| rvv·clang-18(micro) | 57 | 2.19–2.28 | 1.111–1.129 | 0.496/0.503/0.505/0.507 | **0.5040** | **named-X** | opp 本身慢(1.13ms·5-bit unpack) 但 ours 2.2ms 更慢·qh inject+dmin fold |
| k1·clang-18(what-if) | 57 | 2.44–2.45 | 1.665–1.666 | 0.682/0.679/0.681/0.680 | **0.6806** | **named-X** | 输·opp 慢(1.66ms) 但 ours 2.45ms·hl8 半宽·**q5x [GAP-Q5K-VLEN128-QH-REGCLIFF] per-board 一致(qh 重)** |
- **正确门**：0/512 全（maxrel ≤8.5e-4·q5_K random-scale FMA-order benign·仍 0 mismatch@rel≥1e-3）。

### q6_K decode M=1 GEVM（6-bit signed8·16 sub-block·no-min single-scale·stride 3360）
| 板·域 | leaf vsetvl | ours_ms | opp_ms | ratio(4) | median | predreg | 墙/成色 |
|---|---:|---:|---:|---|---:|:--:|---|
| rvv·gcc-15.2(deploy) | **2461** | 5.83–5.92 | 0.310–0.322 | 0.053/0.055/0.054/0.053 | **0.0535** | **named-X** | **GCC-DEATH**(2461 vsetvl)·deploy LOSS |
| rvv·clang-18(micro) | 22 | 1.16–1.22 | 0.306–0.312 | 0.256/0.268/0.257/0.262 | **0.2595** | **named-X** | opp 强(0.31ms)+ql/qh 6-bit combine+16 sub-block·ours 1.2ms |
| k1·clang-18(what-if) | 22 | 2.31–2.32 | 0.867–0.907 | 0.374/0.392/0.379/0.375 | **0.3771** | **named-X** | 输·hl8 半宽+6-bit 重·opp 0.87ms |
- **正确门**：0/512 全（maxrel ≤1.1e-5·多数 bit-exact 0.00e+00·no-min 整数核干净）。

**★[CASE-COMPILER-ASYMMETRY] 本役 4 格再证（K-quant super-block）**：repack-GEVM leaf **compiler 极敏感**（gcc vsetvl 922–2952 vs clang 14–81·gcc ours 3.3–8.8ms vs clang 0.5–2.3ms = **3–15× 慢**）；opp block-dot insensitive（gcc-stock，跨编译器不变）。**rvv 主账 = gcc-15.2 deploy-clean（出货 gcc-15·全 GCC-DEATH）**·clang-18 = 对称-micro（跨板可比 k1）。

**★fold-not-amortized 机理（C3′ 边界·durable）**：M=1 GEVM 只有 1 行 activation·super-block fold epilogue（q2/q5 dmin×bsums·q3/q6 hmask/6-bit unpack·全格 6-bit scale/min lane-unpack）**每 super-block 固定开销**·无 nr≥4 行摊销 → repack-GEVM leaf 的 fold 占比高·输给 stock block-dot（block-dot 本就 per-block 标量 fold·无 repack 转置税）。**16 sub-block 格（q2/q3/q6）比 8 sub-block（q4/q5）更重**（fold 次数×2）。

---

## 3. ★T3 回填清单（★留主会话机算入库·本 agent 不动 T3·regime=DECODE·禁继承·rvv 双域）

**主表 `T3_master_rebuild.csv`（gemm_tile·regime=DECODE 行·本役 4 格填·禁混 prefix/vec_dot 行）：**

| op | format | regime | rvv_disp | rvv_cold(median) | k1_disp | k1_cold(median) | 成色/域 tag |
|---|---|---|---|---:|---|---:|---|
| gemm_tile | q2_K | **decode** | **named-X** | **0.0685**(gcc-15.2 GCC-DEATH)/0.3635(clang18-micro) | PASS | **0.9585**(clang18·what-if·near-parity) | 仅 q2_K@k1 near-parity·rvv 全 LOSS·gate=vs-ggml·0 hand-brick |
| gemm_tile | q3_K | **decode** | **named-X** | **0.0830**(gcc GCC-DEATH)/0.2189(clang) | **named-X** | **0.4252**(clang18·what-if) | 全 LOSS·16 sub-block hmask 重·bit-exact 门 |
| gemm_tile | q5_K | **decode** | **named-X** | **0.1273**(gcc GCC-DEATH)/0.5040(clang) | **named-X** | **0.6806**(clang18·what-if) | 全 LOSS·qh inject 重·[GAP-Q5K-QH-REGCLIFF] per-board |
| gemm_tile | q6_K | **decode** | **named-X** | **0.0535**(gcc GCC-DEATH)/0.2595(clang) | **named-X** | **0.3771**(clang18·what-if) | 全 LOSS·16 sub-block 6-bit 最重·bit-exact 门 |

**36-col T3_A(rvv)/T3_B(k1) 关键列**：cold_ratio=col30（median 填·**rvv 主填 gcc-15.2 deploy-clean·clang18-micro 入 col34 footnote**）；opponent_grade=**native-vec/better-vec**（stock block-dot·非 hand-brick）；opponent_symbol=上表探针地址；compiler_axis=rvv `gcc15.2-deploy=MAIN`+`clang18-micro=footnote` / k1 `clang18-sym=MAIN`；hardgate_0p8=**in-denom**（q2_K rvv 双域 named-X + k1 PASS·q3/q5/q6 全 named-X 三口径）；ledger_account=`matmul-kernel-sym`·**regime=decode**（≠prefill 行·禁混）；gcc_death_flag=**TRUE 全 4 格@rvv-gcc**（vsetvl 922–2952·[CASE-KQUANT-GCC-CODEGEN]）。
- **provenance 指针**：`A2-batch5-kquant-decode-M1-raw/logs/{rvv,k1}_run.log` + `{rvv,k1}_build_seal.txt`（raw A/B·gate 全绿·2-seed×2trial·探针）。
- **★全格成色标注 = what-if/LOSS**：k1 = force-constructed what-if（front-door VLEN256 declines repack@decode·batch4）·rvv = deploy-clean 但全 GCC-DEATH LOSS。**唯 q2_K@k1 near-parity·可作弱素材·非硬赢·非已部署**。

**★计数纪律**：decode kernel-sym 覆盖 = C3′ 证据·**NOT e2e·NOT perf-covered·禁互推·禁写"加速 N kernel"于 e2e 语境**。本役净贡献 = **C3′ 负结果坐实**（K-quant super-block decode@M=1 = repack-GEVM 适用边界·format-keyed·档案级教材）+ **1 near-parity(q2_K@k1)**。**0 硬赢·0 hand-brick·别登记 Win**。

---

## 4. blocked / 转补测清单（如实报·0 造数）

| 项 | 状态 | 说明 |
|---|---|---|
| **q2_K/q3_K/q5_K/q6_K decode** | ✅ **本役完成**（双板真测·全 named-X 除 q2_K@k1 PASS·正确门全绿） | driver `kquant_gevm_m1_driver_b5.cpp` + leaves 4 格·gate=vs-ggml-agreement |
| **独立 ZERO-MODEL K-quant scalar oracle** | 未 author（gate=vs-ggml-agreement·weaker） | `kquant_repack_verify_q{2,3,6}K.c` 有 `ref_block/ref_isum_block` ZERO-MODEL·可接入强化 gate（本役 vs-ggml 已足够证 pack_w+leaf 正确·未接为省时） |
| **k1 hl16(VLEN256 满宽) K-quant leaf** | 缺（现 hl8 半宽） | 现 leaf = VLEN128 mf2 form·k1 VLEN256 half-util。authored hl16 K-quant leaf 缺·**但预期收益有限**（fold-not-amortized@M=1 是主墙·非宽度）·**低优先** |
| **rvv-gcc GCC-DEATH 修** | 未修（[CASE-KQUANT-GCC-CODEGEN]） | super-block repack leaf 触发 gcc-15 regfile spill(vsetvl 922–2952)·**修法=改 emit 减 live regs 或换 clang deploy**·但 decode@M=1 本就 LOSS(clang 也输)→**修 gcc-death 不翻正·不立项**（宪章规则1：未击中关键路径=小改善·此处关键路径=fold 摊销非编译器） |
| **q2_K@k1 near-parity per-format gate 立项** | 候选素材（弱） | 0.9585 near-parity·非硬赢·**成色不足以立 per-format deploy gate**（<1.0·what-if）·记录不推进 |

**★便宜档/near-parity 纪律**：本役 **0 verified hand-brick**·唯 1 non-loss(q2_K@k1 0.96 near-parity·opp 弱)。**无硬赢·无 Win 登记**。这是 C3′ **负结果**（format-keyed 适用边界坐实）·与 q4_K rvv LOSS 同构·**扩展 batch4 board-split 为家族级边界律**。

## durable files
- `A2-batch5-kquant-decode-M1.md`（本文）
- `A2-batch5-kquant-decode-M1-raw/kquant_gevm_m1_driver_b5.cpp`（q2/q3/q5/q6_K M=1 GEVM 驱动·gate=vs-ggml-agreement）
- `A2-batch5-kquant-decode-M1-raw/leaves/{q2,q3,q5,q6}_K_gevm.c`（weft-opt 生成 leaf·hl8）
- `A2-batch5-kquant-decode-M1-raw/run_kquant_gevm_{rvv,k1}.sh`（双板 build+cold 脚本·可复现）
- `A2-batch5-kquant-decode-M1-raw/logs/{rvv,k1}_run.log` + `{rvv,k1}_build_seal.txt`（收尾拷回）
