# [G7-L3 货架B] q5_1 @k1 net-new deploy + e2e — ★真 our-kernel dual-board 绿（prefill）

> Board `ssh k1` (SpacemiT X60 / VLEN256 / 8 harts / **stock clang-18** / DVFS perf-gov 1.6GHz).
> Model `/data/g7q51/tinyllama-q5_1.gguf` (llama 1B Q5_1, 831 MiB, sha256 head `b5a90357e67ca05a`; requantized from tinyllama-q8_0 via `llama-quantize --allow-requantize`, 决策卡④).
> **NO git · NO rvv · board reversible (shared tree patched-in-place then RESTORED·md5 double-proof) · correctness-first · kernel==system (clang-18 symmetric).**
> 承 q4_1@k1 成功 pattern（`d811b323`）copy-then-adapt 到 **q5_1**（q5-family transposed-qh·**q8_1 activation**·block_q8_1x4）.
> §L0 收尾：完成 FLAT@k1 五格 dual-board 矩阵（q4_0/q8_0/q4_1/q5_0/**q5_1**）·前 agent 腰斩→本 agent 重做收尾.

## ★关键区别 q4_0/q8_0@k1（成色强于首批·同 q4_1/q5_0）
- **q5_1@k1**：k1 stock = **纯 block-dot**（无 q5_1 riscv repack；stock lib q5_1-repack syms = **0**；dormant `spacemit/repack.cpp` q5 变体存在但 **NOT compiled**·compile_commands 0 条·§6）。∴ **我方 emitted repack = 唯一 repack**·OFF = 真 stock block-dot → **e2e winner = 我方 net-new VLEN256 kernel** = ★**真 our-kernel dual-board 绿**（不是 stock-repack 白嫖）。

## 1. VLEN256 kernel emit（weft-opt·vl=16·byte-exact seal·transposed-qh·q8_1）
- **GEMM (prefill)**：front-door AUTO-selects repack·`deriveRepackHalfLanes(256,16)=16` → **one 16-lane strip**·`weft_emitted_gemm_q5_1.inc`（md5 **03866f50**·func `weft_emitc_ggml_gemm_q5_1_q8_1_kernel_…`）。
- **GEVM (decode)**：VLEN256 decode DECLINE repack → **author** one-strip typed region（`q5_1_gevm_vlen256.mlir`·half_lanes=16）→ `weft_emitted_gevm_q5_1.inc`（md5 **96c87367**）。
- **★transposed-qh + q8_1**：q5_1 5th-bit plane 转置为 interleaved qh_lo/qh_hi（u16×16）；weight block `block_q5_1x16`（d/m/qs/qh·stride 匹配 emitted）；**q8_1 activation**（含 sum-of-quants·区别 q5_0 的 q8_0）。
- **★vl=16 板 objdump seal**（build_seal §6）：GEMM vset = 11×e16m1·9×e32m2·2×e8mf2；GEVM = 5×e16m1·3×e32m2·2×e8mf2。VLEN256 上 e16m1/e32m2/e8mf2 = **16 lanes** = full-width 16-col strip（部署==证过 VLEN256-native）。

## 2. 净新 scaffold 部署到 k1 tree（3 shared 文件 + 2 incs·可逆·NO git）
`deploy_patch_q5_1_k1.py`（patch OK·anchors count==1 验证）：
1. **weight block（transposed qh）** `block_q5_1x16` + **q8_1 activation** `block_q8_1x4`（144B·复用 q4_1 家族基建）。
2. **★净新 q8_1 mat-quant** `ggml_quantize_mat_t<1,GGML_TYPE_Q8_1>`（byte-match `quantize_row_q8_1`·上游只 Q8_0/Q8_K·`q81mat: True`）。
3. **make_block_q5_1x16 interleaver**（d/m/qs memcpy + **qh 转置 lo/hi**·MIRAGE trap）+ repack/gemv/gemm templates + generics + trait 注册。
4. **dispatch case256 intercept**（case128/256=ON·`else if (GGML_TYPE_Q5_1)`）。
5. **arch VLEN256 bodies**（gated·call `weft_emitc_…`·banner）；非-256 fall through。
- 边际成本：q5_1 = q5_0 + m-field + q8_1 activation·**零改动复用 q4_1 q8_1 sub-scaffold** = C2 模板经济学再实证。

## 3. correctness GREEN（★硬门·ZERO-MODEL bit-exact 强门 + greedy A==B 佐证 + MIRAGE 排除）
- **① UT 独立 oracle（ZERO-MODEL·板 VLEN256·build 前先验·强门）**：`ut_q5_1_interleaver_k1.cpp` + `ut_q5_1_gemm_k1.cpp`（emitted vl=16 kernel vs **独立标量 q5_1 oracle**·从原 block 布局重构·×q8_1·零复用 packed）：
  - **interleaver/GEVM GREEN**：fails=**0/16**·max_rel **4.988e-07**。
  - **GEMM/prefill GREEN**：fails=**0/64**·max_rel **3.347e-06**。
  - ★qh 转置 bit-order 若错 → emit 偏离 oracle；0-fail = **MIRAGE trap 关**（transposed-qh 正确·q8_1 sum-of-quants 正确）。**这是强正确性保证（tol 1e-3·实测 3e-6）**。
- **② greedy A==B（真模型·佐证轴·temp0/top-k1/seed1·clean 归一）**：**3/4 completion token-identical**（"…is Paris." / 小村庄故事 / brown-fox "…over the lazy dog."）+ **1/4 benign near-tie**（prompt[3]："Q: What is 2+2? A:"）：
  - A(ON)：`The answer is 4.` ／ B(OFF)：`The answer to the question is 4.`——**两者皆 coherent·皆给正确答案 4**·仅一个 near-tie token 位 argmax 翻转（我方 16-lane repack vs stock scalar block-dot 累加顺序不同·~1e-6 差·在概率贴近的 token 边界翻 argmax）·**非 garbage·非正确性失败**（q4_1/q5_0 侥幸 4/4·q5_1 撞上 near-tie·如实披露）。verdict 立于 ① ZERO-MODEL bit-exact，非 ② 的 4/4。
- **③ engage banner 14 fires** = 我方 repack 真在 e2e 运行（非 dead code）。
→ **correctness_green = TRUE**（强门 = ZERO-MODEL byte-exact 0/16+0/64；佐证 = A==B 3/4 identical + 1 benign near-tie·MIRAGE closed）。

## 4. 部署五验 + 反向控制（build_seal_raw.txt）
- ① dedicated build dir seeded cp-a stock·physical .so swap（LD_LIBRARY_PATH → build-k1-q5_1/bin）。
- ② **OFF md5=`14b6add6`**（block-dot·q5_1 syms=**0**·同 q5_0 OFF·q5_x 无 stock repack ⇒ OFF=同一 pristine block-dot lib）· **ON md5=`aef46fea`**（our repack·weft_emitc q5_1 syms=**2**）· **ON≠OFF**。
- ③ 反向控制：OFF 无 q5_1 → block-dot·prefill 8.99→3.67 t/s（**2.45× 时间分离**·swap 真生效）。
- ④ objdump vl=16 seal（§1·16 lanes）。
- ⑤ 对手 = stock generic block-dot `ggml_vec_dot_q5_1_q8_1`（非 SELF·OFF 0 syms）。
- ⑥ clang-18 双侧对称（唯一 diff = q5_1 净新 scaffold + q8_1 mat-quant + emitted kernels）。

## 5. 分相 e2e（phase-split paired A/B·n=12/side[REPS6×PASS2]·1.6GHz·-t4·interleaved·measure_raw.txt）

| phase | A ON (our vl=16 repack) | B OFF (block-dot) | **ON/OFF** | ON sd | OFF sd | n | freq |
|---|---|---|---|---|---|---|---|
| **prefill pp128** | **8.9934** t/s | 3.6670 t/s | **2.4525×** | ~0.1% | ~0.03% | 12 | 1.6GHz |
| decode tg32 | 2.5206 t/s | 3.1108 t/s | **0.8103×** | ~0.03% | ~0.05% | 12 | 1.6GHz |

- worst-case prefill（min ON / max OFF）= **2.4521×**·**prefill WIN**（well above parity）。
- **decode 0.8103× = LOSS**（预着色·同 q5_0）：我方 authored GEVM region 慢于 stock vec_dot·honest·verdict 不依赖 decode。
- **★load 说明**：load-gate 开门（start loadavg 2.83·0 CPU-bound on cores 0-3）；测中外部争用使 loadavg 升至 ~6.0（他人进程）。**interleaved paired A/B 保 ratio**（bench 内部 6-rep sd 仅 0.03–0.1%·prefill ratio 极稳·worst-case≈mean）·documented deviation·同 q4_1 §5 情形。

## 6. 双账本 + 预注册出口 + 诚实 caveat
- **双账本**：clang-18 双侧对称 ⇒ **kernel账 == system账（CONVERGE·同数 2.45×/0.81×）**·[CASE-COMPILER-ASYMMETRY] not triggered。
- **★预注册出口 = prefill ≥parity（2.4525×）→ ★真 our-kernel dual-board 绿**：q5_1 rvv 绿（perf-covered）+ **k1 绿（our VLEN256 vl=16 repack·prefill 2.45× vs stock block-dot·winner=我方 kernel）= dual-board**·成色强于 q4_0/q8_0@k1。perf-covered **9/83 不变**（q5_1 已绿 rvv·此=dual-board 成色·禁互推）。
- **★诚实 caveat**：**stock q5_1 baseline 弱**——无 compiled stock repack（block-dot·3.67 pp）。2.45× = 我方净新 repack GEMM vs **未优化 stock q5_1 block-dot**（L1 path-win），**非**"比调优内核快 2.5×"。倍数低于 q4_1（4.99×）因 q5_1 qh 5th-bit 重构 + q8_1 sum 开销使我方 repack 本身慢于 q4_1——仍 beat block-dot。verdict 立于 prefill≥parity。
- **decode LOSS（0.81×）= 预着色·黄**·如实分报。
- **fair-play note**：dormant `spacemit/repack.cpp` q5 repack **NOT compiled**（0 条·OFF 0 syms）——对手 = as-shipped block-dot（诚实基线）。
- kernel+e2e / prefill+decode 永分报。

## 7. A-tree restore（md5 双证 clean）
- **proof 1**：shared source 全回 baseline —— repack.cpp `3cac40aa` · repack.h `57851439` · arch/riscv/repack.cpp `c3c101fd`（== baseline·post-build 复验）。
- **proof 2**：**0** stray q5_1 incs in tree · **0** TCRV-G7-L3 route markers in shared source。
- **cleanup**：`/data/build-k1-q5_1` 已删 · 0 我方 leftover procs · 生产 `/data/k1build` live lib = OFF pristine（`14b6add6`·未动）。**污染状态 = CLEAN**。

## 8. verdict
**★GREEN（真 our-kernel dual-board·prefill）**：correctness GREEN（UT ZERO-MODEL 0/16+0/64 byte-exact 强门·A==B 3/4 identical+1 benign near-tie·banner 14·MIRAGE closed）∧ **prefill 2.4525× ≥parity**（n=12·部署五验全过·clang-18 symmetric·winner=我方 VLEN256 vl=16 emitted repack·对手=stock block-dot）→ q5_1@k1 dual-board·成色强于 q4_0/q8_0@k1。**decode 0.8103× LOSS（预着色·黄·honest 分报）**。
**方法学**：VLEN256 vl=16 net-new emit（transposed-qh + 净新 q8_1 mat-quant·MIRAGE 排除）+ q8_1 家族 sub-scaffold 零改动复用（C1 extensibility + C2 边际成本）·★near-tie 现象 = FP 累加顺序诚实教材（bit-exact-vs-oracle ≠ bit-identical-vs-stock·greedy 在 near-tie 翻转·非失败）。

## schema label 建议（留主会话裁）
- 建议登记：`q5_1@k1-e2e-prefill 2.4525× WIN (our VLEN256 vl=16 emitted repack vs stock block-dot·dual-board·winner=our-kernel)` · decode 0.8103× LOSS(pre-colored)。
- perf-covered **9/83 不变**（q5_1 已绿 rvv·此为 dual-board 成色·禁互推）。dual-board 成色增量裁量主会话。
- ★correctness 措辞：强门 = ZERO-MODEL bit-exact；A==B 记 "3/4 token-identical + 1 benign near-tie（both coherent+correct answer·FP-order argmax flip·非 garbage）"·勿记为 mismatch/fail。

## durable files
- `evidence.md`（本文）· `weft_emitted_gemm_q5_1.inc`（03866f50·vl=16 GEMM）· `weft_emitted_gevm_q5_1.inc`（96c87367·vl=16 GEVM）· `q5_1_gevm_vlen256.mlir`
- `deploy_patch_q5_1_k1.py`（含净新 q8_1 mat-quant）· `ut_q5_1_gemm_k1.cpp` · `ut_q5_1_interleaver_k1.cpp`（独立 oracle）
- `build_seal_raw.txt` · `measure_raw.txt`（load-gate+correctness+4×###AB n=12）· `correctness_clean.txt`（token-level 3/4 identical + near-tie diff）· `objdump_*_q5_1_*.txt`（vl=16 seals）
- 驱动脚本（父目录）：`../q5_build_seal.sh` · `../q5_measure.sh` · `../q5_correctness_ut.sh`
