# [G7-L3 货架B] q4_1 @k1 net-new deploy + e2e — ★真 our-kernel dual-board 绿

> Board `ssh k1` (SpacemiT X60 / VLEN256 / 8 harts / **stock clang-18** / DVFS perf-gov 1.6GHz).
> Model `/data/g7q41/tinyllama-q4_1.gguf` (llama 1B Q4_1, 702 MiB, sha256 `09c1fb0e95c92710`; requantized from tinyllama-q8_0 via `llama-quantize --allow-requantize`, 决策卡④).
> **NO git · NO rvv · board reversible (shared tree patched-in-place then RESTORED·md5 double-proof) · correctness-first · kernel==system (clang-18 symmetric).**
> 承 rvv G5-M2 q4_1 净新 scaffold（`1d818621`·VLEN128 vl=8）→ copy-then-adapt 到 **k1/VLEN256 vl=16**.

## ★关键区别 q4_0/q8_0@k1（成色强于首批）
- **q4_0/q8_0@k1**：k1 stock **出货 16x1 repack**（case256 ON）→ e2e winner = **stock 自己的 repack**（非我方 kernel）→ dual-board 成色·**不增 perf-covered**。
- **q4_1@k1**：k1 stock = **纯 block-dot**（`ggml_repack_get_optimal_repack_type` 无 q4_1 branch → nullptr → generic `ggml_vec_dot_q4_1_q8_1`；stock lib q4_1-repack syms = **0**；一个 dormant `spacemit/repack.cpp` q4_1 变体存在但 **NOT compiled**，compile_commands 0 条·见 §6 fair-play）。∴ **我方 emitted repack = 唯一 repack**·OFF = 真 stock block-dot → **e2e winner = 我方 net-new VLEN256 kernel** = ★**真 our-kernel dual-board 绿**（不是 stock-repack 白嫖）。

## 1. VLEN256 kernel emit（weft-opt·vl=16·区别 rvv vl=8 VLEN128·byte-exact seal）
两 .inc 由 host `build-weft/bin/weft-opt` + `mlir-translate --mlir-to-cpp` 生成（fixtures = rvv q4_1 VLEN128 素材·march 切 VLEN256）：
- **GEMM (prefill)**：`weft-opt <q4_1-gemm-prefill-vlen128.mlir> --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b --weft-rvv-lower-to-emitc` → **front-door AUTO-selects repack**·strip-width materialize `deriveRepackHalfLanes(256,16)=16` → **one 16-lane strip**（vs VLEN128 two 8-lane halves）。`weft_emitted_gemm_q4_1.inc`（md5 **736a716c**·func `weft_emitc_ggml_gemm_q4_1_q8_1_kernel_...`·ABI (nr,bs,n,s,nc,vx,vy)）。
- **GEVM (decode)**：front-door 在 **VLEN256 decode 主动 DECLINE repack GEVM → block-dot**（selector reason `block-dot-decline-q4_0-vlen256-decode-k1-loss`·capability-keyed 决策）。∴ 同 q4_0-vlen256-gemv fixture 做法**直接 author** one-strip typed region（`q4_1_gevm_vlen256.mlir`·half_lanes=16·`fold_model=lane_wise_vector_scale_min`·unsigned nibble·min-fold offsets 32/2）→ lower-to-emitc。`weft_emitted_gevm_q4_1.inc`（md5 **c0e69a28**·ABI (n,s,nc,vx,bx,vy,by,nrc) 同 VLEN128）。
- **★vl=16 源级 seal**：GEMM AVL literal `, 16)`×**53** / `, 8)`×**0**；GEVM `, 16)`×**17** / `, 8)`×**0**·unsigned nibble（vand/vsrl，无 vsll/vsra）·min-fold（vfwmul+vfadd）。
- **★vl=16 板 objdump seal**（build_seal §6）：GEMM vset = 9×e32m2·9×e16m1·1×e8mf2；GEVM = 3×e32m2·3×e16m1·1×e8mf2。VLEN256 上 e32m2/e16m1/e8mf2 = **16 lanes** = full-width 16-col strip（**部署==证过 VLEN256-native**·非 rvv 的 vl=8 半宽跑 VLEN256）。

## 2. 净新 scaffold 部署到 k1 tree（3 shared 文件 + 2 incs·可逆·NO git）
`deploy_patch_q4_1_k1.py`（port 自 rvv `deploy_patch_q4_1_emitted.py`·anchors 逐一验 count==1 于 k1 baseline）：
1. **weight block（无 qh）** `block_q4_1x16`（320B·d@0 m@32 nibbles@64）+ **q8_1 activation** `block_q8_1x4`（144B·复用 q5_1/q4_1 家族基建）。
2. **净新 q8_1 mat-quant** `ggml_quantize_mat_t<1,GGML_TYPE_Q8_1>`（byte-match `quantize_row_q8_1`·上游只 Q8_0/Q8_K）。
3. **make_block_q4_1x16 interleaver**（d/m byte-offset memcpy·unsigned nibble·无 xor·无 qh 转置）+ repack/gemv/gemm templates + scalar generics + trait 注册。
4. **dispatch case256 intercept**（k1 q8_0 block 后 append `else if (GGML_TYPE_Q4_1)`·case128/256=ON）。
5. **arch VLEN256 bodies**（gated `__riscv_vlenb()*8==256`·call `weft_emitc_...`·banner）；非-256 fall through generic。
- 边际成本：q4_1 = q5_1 减 qh·**零改动复用 q8_1 sub-scaffold** = C2 模板经济学再实证（家族第 N 成员）。

## 3. correctness GREEN（★硬门·ZERO-MODEL bit-exact + greedy A==B + MIRAGE 排除）
- **① UT 独立 oracle（ZERO-MODEL·板 VLEN256·build 前先验）**：`ut_q4_1_gemm_k1.cpp` + `ut_q4_1_interleaver_k1.cpp`（make_block + emitted vl=16 kernel vs **独立标量 q4_1 oracle**·unsigned nibble+min+q8_1 s·从实际输入零复用重算）：
  - **GEMM/prefill GREEN**：fails=**0/64**·max_rel **6.364e-06**。
  - **GEVM/interleaver GREEN**：fails=**0/16**·max_rel **1.974e-06**。
  - ★两数**与 rvv VLEN128 逐位相同**（6.364e-06 / 1.974e-06）→ strip-width 8→16 不改数值·MIRAGE trap 关。
- **② greedy A==B（真模型·A=ON our-repack / B=OFF block-dot·同权重）**：**4/4 生成文本 byte-identical**（"…is Paris." / Lily 故事 / "…is 4." / "…fox jumps."）·all coherent English·NO NaN/garbage。（唯一 raw diff = model-loading 旋转动画长度 + llama-cli 内联 perf-stats 行·非 token·见 correctness_clean.txt / measure_raw）。
- **③ engage banner 23 fires**（`TCRV G7-L3 EMITTED GEMM/GEVM ... ENGAGED`）= 我方 repack 真在 e2e 运行（非 dead code）。
→ **correctness_green = TRUE**（ZERO-MODEL byte-exact ∧ A==B ∧ MIRAGE closed）。

## 4. 部署五验 + 反向控制（build_seal_raw.txt）
- ① ONE 真 ELF `/data/k1build/bin/llama-bench`·physical .so swap（LD_LIBRARY_PATH=/data/build-k1-q41/bin·ldd 证 libggml-cpu 从此解析）。
- ② **OFF md5=`14b6add6`**（block-dot·q4_1 syms=**0**）· **ON md5=`24be4472`**（our repack·weft_emitc q4_1 syms=**2**）· **ON≠OFF**。
- ③ 反向控制真熄灭：OFF dispatch 无 q4_1 → block-dot·行为证 = prefill 23.94→4.79 t/s（**5.0× 时间分离**·swap 真生效）。
- ④ objdump vl=16 seal（§1·VLEN256-native e32m2/e16m1/e8mf2 = 16 lanes）。
- ⑤ 对手 = stock generic block-dot `ggml_vec_dot_q4_1_q8_1`（非 SELF·非 hand-brick·stock q4_1 无 repack）。
- ⑥ clang-18 双侧对称（唯一 diff = q4_1 净新 scaffold + emitted kernels）。

## 5. 分相 e2e（phase-split paired A/B·n=12/side·1.6GHz·-t4·interleaved·relIQR·measure_raw.txt）

| phase | A ON (our vl=16 repack) | B OFF (block-dot) | **ON/OFF** | ON spread | OFF spread | n | freq |
|---|---|---|---|---|---|---|---|
| **prefill pp128** | **23.943** t/s | 4.7930 t/s | **4.9955×** | ~3–6%† | 0.017% | 12 | 1.6GHz |
| decode tg32 | 7.6402 t/s | 3.9781 t/s | **1.9205×** | 0.014% | 0.270% | 12 | 1.6GHz |

- worst-case prefill（min ON / max OFF）= **4.92×**·两相皆 WIN（well above parity）。
- †ON prefill spread（stddev 1.55/1.04·CV ~6%）来自**外部争用**：一 foreign q5_K llama-cli hung 90min@100% on core 1（另一线·未杀）→ 本测**改 pin 0-3→4-7**（同构 X60·ratio cluster-independent·documented deviation）；测中 2 competitors 短暂上 4-7（loadavg 峰 6.25）。**interleaved paired A/B 保 ratio**（OFF 极稳 0.017%·ratio 5.0× 远超噪声）。

## 6. 双账本 + 预注册出口 + 诚实 caveat
- **双账本**：k1 kernel-axis compiler = clang-18；system/deploy = clang-18 ⇒ **kernel账 == system账（CONVERGE·同数 5.00×/1.92×）**·[CASE-COMPILER-ASYMMETRY] not triggered。
- **★预注册出口 = prefill ≥parity（4.9955×）→ ★真 our-kernel dual-board 绿**：q4_1 rvv 绿（perf-covered·`1d818621`·prefill 3.68×）+ **k1 绿（our VLEN256 vl=16 repack·prefill 5.00× / decode 1.92× vs stock block-dot）= dual-board**·**winner=我方 kernel**（≠ q4_0/q8_0@k1 的 stock-repack）→ **成色强于首批·可能增 perf-covered dual-board（裁量主会话）**。decode 亦 WIN（both-phase·未触发"decode→黄"）。
- **★诚实 caveat（大倍数解读·同 rvv q4_1 / q4_0 WinB）**：**stock q4_1 baseline 弱**——q4_1 无 stock riscv repack（block-dot·4.79 pp / 3.98 tg·明显慢）。5.00× = 我方净新 repack GEMM vs **未优化 stock q4_1 block-dot**（一个 L1 path-win·同 q4_0 WinB 5.9×），**非**"比调优内核快 5×"。verdict 立于 prefill≥parity。
- **decode 亦 WIN（1.92×）= atypical both-phase**：与 q5_x prefill-win/decode-wash 定式相反·主因 = 弱 stock q4_1 decode baseline（同 rvv q4_1 1.67× both-win 现象·flag for scrutiny·verdict 不依赖 decode）。
- **fair-play note**：k1 tree 含 dormant `spacemit/repack.cpp` q4_1 repack（vendor 变体·**NOT compiled**·compile_commands 0 条·stock lib 0 q4_1 syms）——若 SpacemiT 启用该变体，对手会更强；本测对手 = **as-shipped compiled 路径 = block-dot**（诚实基线）。
- kernel+e2e / prefill+decode 永分报（[[kernel-wins-dont-transplant-to-e2e]]）。

## 7. A-tree restore（md5 双证 clean）
- **proof 1**：shared source 全回 baseline —— repack.cpp `3cac40aa` · repack.h `57851439` · arch/riscv/repack.cpp `c3c101fd`（== baseline byte-exact·post-measurement 复验）。
- **proof 2**：**0** stray q4_1 incs in tree · **0** TCRV-G7-L3 route markers in shared source · q4_0 stock incs UNTOUCHED（`spacemit/repack.cpp` 的 block_q4_1x16 是 pre-existing vendor·0 我方 marker·mtime 09:59 早于本 session）。
- **cleanup**：`/data/build-k1-q41` 已删 · 0 我方 leftover llama procs（foreign q5_K 未动·已自行退出）· live lib = OFF pristine。**污染状态 = CLEAN**。

## 8. verdict
**★GREEN（真 our-kernel dual-board）**：correctness GREEN（UT ZERO-MODEL 0/64+0/16 byte-exact·A==B 4/4·banner 23·MIRAGE closed）∧ **prefill 4.9955× ≥parity**（n=12·八门全过·clang-18 symmetric·winner=我方 VLEN256 vl=16 emitted repack·对手=stock block-dot）→ q4_1@k1 dual-board·**成色强于 q4_0/q8_0@k1**（winner=our-kernel 非 stock-repack）。decode 1.92× 亦 WIN（atypical·weak-baseline 驱动·§6 诚实披露）。
**方法学**：VLEN256 vl=16 net-new emit（GEMM front-door auto·GEVM authored one-strip·selector VLEN256-decode-decline 实证）+ q8_1 家族 sub-scaffold 零改动复用（C1 extensibility + C2 边际成本）。

## schema label 建议（留主会话裁）
- 建议 T8/登记：`q4_1@k1-e2e-prefill 4.9955× WIN (our VLEN256 vl=16 emitted repack vs stock block-dot·dual-board·winner=our-kernel)` · decode 1.9205× WIN(atypical)。
- perf-covered dual-board 成色增量 = **裁量主会话**（winner=我方 kernel·区别 q4_0/q8_0@k1 的 stock-repack 白嫖·可能是 q4_1 的第二板 our-kernel 证据）。

## durable files
- `evidence.md`（本文）· `weft_emitted_gemm_q4_1.inc`（736a716c·vl=16 GEMM）· `weft_emitted_gevm_q4_1.inc`（c0e69a28·vl=16 GEVM）· `q4_1_gevm_vlen256.mlir`（authored one-strip fixture）
- `deploy_patch_q4_1_k1.py` · `q41_k1_build_seal.sh` · `q41_k1_measure.sh` · `ut_q4_1_gemm_k1.cpp` · `ut_q4_1_interleaver_k1.cpp`
- `build_seal_raw.txt`（OFF/ON build+patch+objdump vl=16 seal+restore double-proof）· `measure_raw.txt`（load-gate+correctness+4×###AB n=12 JSON）· `correctness_clean.txt` · `objdump_*_q4_1_*.txt`（vl=16 seals）
