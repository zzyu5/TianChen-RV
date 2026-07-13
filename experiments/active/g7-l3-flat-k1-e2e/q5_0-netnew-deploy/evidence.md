# [G7-L3 货架B] q5_0 @k1 net-new deploy + e2e — ★真 our-kernel dual-board 绿（prefill）

> Board `ssh k1` (SpacemiT X60 / VLEN256 / 8 harts / **stock clang-18** / DVFS perf-gov 1.6GHz).
> Model `/data/g7q50/tinyllama-q5_0.gguf` (llama 1B Q5_0, 767 MiB, sha256 head `2828d7681fff4cbf`; requantized from tinyllama-q8_0 via `llama-quantize --allow-requantize`, 决策卡④).
> **NO git · NO rvv · board reversible (shared tree patched-in-place then RESTORED·md5 double-proof) · correctness-first · kernel==system (clang-18 symmetric).**
> 承 q4_1@k1 成功 pattern（`d811b323`·casefile `q4_1-netnew-deploy/`）copy-then-adapt 到 **q5_0**（q5-family transposed-qh·q8_0 activation）.
> §L0 收尾：完成 FLAT@k1 五格 dual-board 矩阵（q4_0/q8_0/q4_1/**q5_0**/q5_1）·前 agent 腰斩→本 agent 重做收尾（腰斩遗留 build dir 已核干净·生产树未动）.

## ★关键区别 q4_0/q8_0@k1（成色强于首批·同 q4_1）
- **q4_0/q8_0@k1**：k1 stock **出货 16x1 repack**（case256 ON）→ e2e winner = **stock 自己的 repack**（非我方 kernel）→ dual-board 成色·**不增 perf-covered**。
- **q5_0@k1**：k1 stock = **纯 block-dot**（无 q5_0 riscv repack；stock lib q5_0-repack syms = **0**；dormant `spacemit/repack.cpp` q5 变体存在但 **NOT compiled**·compile_commands 0 条·见 §6 fair-play）。∴ **我方 emitted repack = 唯一 repack**·OFF = 真 stock block-dot → **e2e winner = 我方 net-new VLEN256 kernel** = ★**真 our-kernel dual-board 绿**（不是 stock-repack 白嫖）。

## 1. VLEN256 kernel emit（weft-opt·vl=16·byte-exact seal·transposed-qh）
两 .inc 由 host weft-opt + `mlir-translate --mlir-to-cpp` 生成（fixtures = rvv q5_0 VLEN128 素材·march 切 VLEN256）：
- **GEMM (prefill)**：front-door AUTO-selects repack·strip-width `deriveRepackHalfLanes(256,16)=16` → **one 16-lane strip**·`weft_emitted_gemm_q5_0.inc`（md5 **7b1ac4e6**·func `weft_emitc_ggml_gemm_q5_0_q8_0_kernel_…`）。
- **GEVM (decode)**：front-door 在 VLEN256 decode 主动 DECLINE repack GEVM → **直接 author** one-strip typed region（`q5_0_gevm_vlen256.mlir`·half_lanes=16）→ lower-to-emitc·`weft_emitted_gevm_q5_0.inc`（md5 **2db3c0c2**）。
- **★transposed-qh 处理**：q5_0 5th-bit plane（原 qh 每列 uint32）转置为 interleaved qh_lo/qh_hi（u16×16·bit c = col c 的第 k / k+16 位）；weight block `block_q5_0x16`（d[16]@0·qs[256]@32·qh[64]@288·stride **352**）。unsigned nibble+min-fold offsets（−16）·q8_0 activation。
- **★vl=16 板 objdump seal**（build_seal §6）：GEMM vset = 7×e16m1·5×e32m2·2×e8mf2；GEVM = 4×e16m1·2×e32m2·2×e8mf2。VLEN256 上 e16m1/e32m2/e8mf2 = **16 lanes** = full-width 16-col strip（**部署==证过 VLEN256-native**·非 rvv vl=8 半宽）。

## 2. 净新 scaffold 部署到 k1 tree（3 shared 文件 + 2 incs·可逆·NO git）
`deploy_patch_q5_0_k1.py`（port 自 rvv·anchors 逐一验 count==1 于 k1 baseline·patch OK）：
1. **weight block（transposed qh）** `block_q5_0x16`（352B）+ **q8_0 activation**（复用 q4_0/q8_0 家族基建）。
2. **make_block_q5_0x16 interleaver**（d/qs byte-offset memcpy + **qh 转置 lo/hi**·correctness-critical MIRAGE trap）+ repack/gemv/gemm templates + scalar generics + trait 注册。
3. **dispatch case256 intercept**（case128/256=ON·append `else if (GGML_TYPE_Q5_0)`）。
4. **arch VLEN256 bodies**（gated `__riscv_vlenb()*8==256`·call `weft_emitc_…`·banner）；非-256 fall through generic。
- 边际成本：q5_0 = q4_0 + qh-transpose plane·**零改动复用 q8_0 activation sub-scaffold** = C2 模板经济学再实证（家族第 N 成员）。

## 3. correctness GREEN（★硬门·ZERO-MODEL bit-exact + greedy A==B + MIRAGE 排除）
- **① UT 独立 oracle（ZERO-MODEL·板 VLEN256·build 前先验）**：`ut_q5_0_interleaver_k1.cpp` + `ut_q5_0_gemm_k1.cpp`（make_block + emitted vl=16 kernel vs **独立标量 q5_0 oracle**·从原 block 布局重构 w0/w1=nibble|qh-5th-bit−16·×q8_0·×d·a.d·零复用 packed）：
  - **interleaver/GEVM GREEN**：fails=**0/16**·max_rel **1.454e-07**。
  - **GEMM/prefill GREEN**：fails=**0/64**·max_rel **8.114e-06**。
  - ★qh 转置 bit-order 若错 → emit 偏离 oracle；0-fail = **MIRAGE trap 关**（transposed-qh 正确）。
- **② greedy A==B（真模型·A=ON our-repack / B=OFF block-dot·同权重·temp0/top-k1/seed1）**：clean 归一（剥离 load-spinner + 内联 perf-stats 行）后 **4/4 completion token-identical**（"…is Paris." / 小村庄故事 / "…2 + 2 is 4." / brown-fox idiom）·all coherent English·NO NaN/garbage。（唯一 raw diff = spinner 动画长度 + `[ Prompt: X ts Generation: Y ts ]` perf 行——该数字**正是我们测的吞吐差**·非 token·见 correctness_clean.txt）。
- **③ engage banner 17 fires**（`TCRV G7-L3 EMITTED GEMM/GEVM … ENGAGED`）= 我方 repack 真在 e2e 运行（非 dead code）。
→ **correctness_green = TRUE**（ZERO-MODEL byte-exact ∧ A==B 4/4 ∧ MIRAGE closed）。

## 4. 部署五验 + 反向控制（build_seal_raw.txt）
- ① dedicated build dir seeded cp-a stock·physical .so swap（LD_LIBRARY_PATH → build-k1-q5_0/bin）。
- ② **OFF md5=`14b6add6`**（block-dot·q5_0 syms=**0**）· **ON md5=`8be6ee1d`**（our repack·weft_emitc q5_0 syms=**2**）· **ON≠OFF**。
- ③ 反向控制真熄灭：OFF dispatch 无 q5_0 → block-dot·行为证 = prefill 8.49→3.84 t/s（**2.21× 时间分离**·swap 真生效）。
- ④ objdump vl=16 seal（§1·VLEN256-native e16m1/e32m2/e8mf2 = 16 lanes）。
- ⑤ 对手 = stock generic block-dot `ggml_vec_dot_q5_0_q8_0`（非 SELF·非 hand-brick·stock q5_0 无 compiled repack·OFF 0 syms）。
- ⑥ clang-18 双侧对称（唯一 diff = q5_0 净新 scaffold + emitted kernels）。

## 5. 分相 e2e（phase-split paired A/B·n=12/side[REPS6×PASS2]·1.6GHz·-t4·interleaved·measure_raw.txt）

| phase | A ON (our vl=16 repack) | B OFF (block-dot) | **ON/OFF** | ON sd | OFF sd | n | freq |
|---|---|---|---|---|---|---|---|
| **prefill pp128** | **8.4923** t/s | 3.8434 t/s | **2.2096×** | ~0.03% | ~0.02% | 12 | 1.6GHz |
| decode tg32 | 2.3549 t/s | 3.2839 t/s | **0.7171×** | ~0.06% | ~0.05% | 12 | 1.6GHz |

- worst-case prefill（min ON / max OFF）= **2.2092×**·**prefill WIN**（well above parity）。
- **decode 0.7171× = LOSS**（预着色·同 rvv q5_0/q5_1 decode loss 定式）：我方 authored GEVM one-strip region 慢于 stock generic vec_dot·honest·verdict 不依赖 decode（§6）。
- load-gate 开门（start loadavg~2.4·0 CPU-bound on cores 0-3）·测中稳定·极低 sd 证 ratio robust。

## 6. 双账本 + 预注册出口 + 诚实 caveat
- **双账本**：k1 kernel-axis compiler = clang-18；system/deploy = clang-18 ⇒ **kernel账 == system账（CONVERGE·同数 2.21×/0.72×）**·[CASE-COMPILER-ASYMMETRY] not triggered。
- **★预注册出口 = prefill ≥parity（2.2096×）→ ★真 our-kernel dual-board 绿**：q5_0 rvv 绿（perf-covered）+ **k1 绿（our VLEN256 vl=16 repack·prefill 2.21× vs stock block-dot·winner=我方 kernel）= dual-board**·**成色强于 q4_0/q8_0@k1**（winner=our-kernel 非 stock-repack）。perf-covered **9/83 不变**（q5_0 已绿 rvv·此=dual-board 成色·禁互推）。
- **★诚实 caveat（大倍数解读·同 q4_1/q4_0 WinB）**：**stock q5_0 baseline 弱**——q5_0 无 compiled stock riscv repack（block-dot·3.84 pp·明显慢）。2.21× = 我方净新 repack GEMM vs **未优化 stock q5_0 block-dot**（一个 L1 path-win），**非**"比调优内核快 2×"。★倍数低于 q4_1（4.99×）因 q5_0 **qh 5th-bit 重构开销**使我方 repack 本身慢于 q4_1 plain-nibble repack——仍 beat block-dot。verdict 立于 prefill≥parity。
- **decode LOSS（0.72×）= 预着色·黄**：与 q4_1@k1 both-phase-win 相反·我方 GEVM region 慢于 stock vec_dot·如实分相（reserve for scrutiny·非 verdict）。
- **fair-play note**：k1 tree 含 dormant `spacemit/repack.cpp` q5 repack（vendor·**NOT compiled**·compile_commands 0 条·OFF lib 0 q5_0 syms）——若 SpacemiT 启用，对手更强；本测对手 = **as-shipped compiled 路径 = block-dot**（诚实基线）。
- kernel+e2e / prefill+decode 永分报（[[kernel-wins-dont-transplant-to-e2e]]）。

## 7. A-tree restore（md5 双证 clean）
- **proof 1**：shared source 全回 baseline —— repack.cpp `3cac40aa` · repack.h `57851439` · arch/riscv/repack.cpp `c3c101fd`（== baseline byte-exact·post-build 复验）。
- **proof 2**：**0** stray q5_0 incs in tree · **0** TCRV-G7-L3 route markers in shared source。
- **cleanup**：`/data/build-k1-q5_0` 已删 · 0 我方 leftover llama/bench procs · 生产 `/data/k1build` live lib = OFF pristine（`14b6add6`·未动·measure 用 LD_LIBRARY_PATH override）。**污染状态 = CLEAN**。

## 8. verdict
**★GREEN（真 our-kernel dual-board·prefill）**：correctness GREEN（UT ZERO-MODEL 0/16+0/64 byte-exact·A==B 4/4·banner 17·MIRAGE closed）∧ **prefill 2.2096× ≥parity**（n=12·部署五验全过·clang-18 symmetric·winner=我方 VLEN256 vl=16 emitted repack·对手=stock block-dot）→ q5_0@k1 dual-board·**成色强于 q4_0/q8_0@k1**（winner=our-kernel 非 stock-repack）。**decode 0.7171× LOSS（预着色·黄·honest 分报）**。
**方法学**：VLEN256 vl=16 net-new emit（GEMM front-door auto·GEVM authored one-strip·transposed-qh interleaver MIRAGE 排除）+ q8_0 activation 家族 sub-scaffold 零改动复用（C1 extensibility + C2 边际成本）。

## schema label 建议（留主会话裁）
- 建议登记：`q5_0@k1-e2e-prefill 2.2096× WIN (our VLEN256 vl=16 emitted repack vs stock block-dot·dual-board·winner=our-kernel)` · decode 0.7171× LOSS(pre-colored)。
- perf-covered **9/83 不变**（q5_0 已绿 rvv·此为 dual-board 成色·winner=我方 kernel·禁互推）。dual-board 成色增量裁量主会话。

## durable files
- `evidence.md`（本文）· `weft_emitted_gemm_q5_0.inc`（7b1ac4e6·vl=16 GEMM）· `weft_emitted_gevm_q5_0.inc`（2db3c0c2·vl=16 GEVM）· `q5_0_gevm_vlen256.mlir`（authored one-strip fixture）
- `deploy_patch_q5_0_k1.py` · `ut_q5_0_gemm_k1.cpp` · `ut_q5_0_interleaver_k1.cpp`（独立 oracle）
- `build_seal_raw.txt`（OFF/ON build+patch+objdump vl=16 seal+restore double-proof）· `measure_raw.txt`（load-gate+correctness+4×###AB n=12 JSON）· `correctness_clean.txt`（token-level A==B 4/4）· `objdump_*_q5_0_*.txt`（vl=16 seals）
- 驱动脚本（父目录·FMT-parametrized）：`../q5_build_seal.sh` · `../q5_measure.sh` · `../q5_correctness_ut.sh`
