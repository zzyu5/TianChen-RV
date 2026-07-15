# A2 batch4 — DECODE (M=1 GEVM) 转补测：repack-GEVM leaf vs opp block-dot @ M=1（双板·构造 harness）

> **任务**：线 A·A2.5 = 构造 **M=1 GEVM microbench harness** + 测 gemm **decode** cells（我方 repack-GEVM leaf @ M=1 vs 对手真派发 block-dot @ M=1）。承 A2-batch2（batch2 DECODE 全格转补测·**无 M=1 GEVM harness**·本役构造之）。
> **赛道**：**GEMM-decode 轴 kernel-sym**（M=1 GEVM·kernel-axis MICRO）。**NOT e2e·NOT perf-covered·不入系统账**。[NG-4]。
> **口径铁线**：cold 唯一·**N=25 cold 中位内禀（≥20）+2-seed**·**禁一切继承**（decode 真测 M=1·禁用 prefill nr≥4 / vec_dot M1 数）·每格对手身份探针（符号级）·ZERO-MODEL 独立 oracle 正确门·预注册判读（cold≥0.8=PASS / <0.8=named-X+墙）·**0 样本不造数·便宜档禁称硬赢·接线≠转绿·预判不作结论**·成色诚实（0 hand-brick 就说 0）。
> **测于**：2026-07-16 · k1(VLEN256·SpacemiT-X60·clang-18.1.8 native·gov=performance 1.6GHz·core0 idle=97%) + rvv(VLEN128·64c·gov=performance 2.6GHz·core32)。主树/build/stock `.so`/governor 未改·stock md5 before==after·无 git。

---

## 0. ★净结论

**★关键区分兑现（禁混·两串行 bug 前科）**：本役测的是 **gemm decode = 我方 repack-GEVM leaf（body=`typed_repack_gemv_loop_body`）@ M=1 vs opp 真派发 block-dot @ M=1**。leaf 符号串虽含 "vec_dot" 但 **BODY = typed_repack_gemv_loop_body**（front-door `--weft-rvv-lower-quant-contraction` 产物），**≠** block-dot vec_dot 路（T3 vec_dot 行·已在别处）。判别键=**核 body 形态**（repack_gemv vs block_dot），非符号名。

**harness = 构造成功**（§1）。**FLAT 3 格 + K-quant 1 格（q4_K）× 双板真测**（q4_0/q4_K net-new·q5_0/q5_1@k1 免测确认·q5_0/q5_1@rvv net-new）：

| 格 | rvv(VLEN128) gcc-15.2 部署-clean | rvv clang-18 对称-micro | k1(VLEN256) clang-18 对称 | 正确门 | 成色 |
|---|---:|---:|---:|:--:|---|
| **q4_0** | **6.909×** PASS | 4.071× PASS | **2.575×** PASS(what-if) | byte-exact 0/512 | **便宜档**·light-vec 弱 opp（rvv q4_0 block-dot VLEN128 gate-off/破损·opp_ms 0.85 >> q5_0 0.34）·big multiple **非硬赢** |
| **q5_0** | 0.947× PASS(near-parity) | **0.794× named-X** | 1.205× PASS(免测确认) | 0/512 | near-parity·memory-leaning·better-vec 中 opp·**repack compiler-sensitive**(clang 0.79↔gcc 0.95) |
| **q5_1** | 1.090× PASS(near-parity) | 0.928× PASS | 1.317× PASS(免测确认) | 0/512 | near-parity·better-vec 中 opp·compiler-sensitive(clang 0.93↔gcc 1.09) |
| **q4_K** | **0.066× named-X**(GCC-DEATH) | **0.361× named-X** | **1.535× PASS** | byte-exact 0/512(vs ggml) | ★**board+opp-strength 分裂**：k1 赢**弱 opp**(0.80ms)·rvv 输**强 opp**(0.22ms native-vec)·super-block fold@M=1 不摊销·rvv-gcc = **[CASE-KQUANT-GCC-CODEGEN] gcc-death**(vsetvl 1387 vs clang 57) |

**★成色诚实（令六 lint·0 verified hand-brick）**：
- **q4_0 大倍数 = 便宜档**：opp = stock q4_0 block-dot（rvv VLEN128 gate-off / light-vec 弱·opp_ms 0.85ms vs q5_0 opp 0.34ms = q4_0 opp ~2.5× 慢=弱）。4–7× multiple **非硬赢**·成色低（同 batch2 prefill q4_0=便宜档 一致）。
- **q5_0/q5_1 = near-parity**（memory-leaning·M=1 decode 常 memory-bound）。opp = better-vec 中对手（block-dot ~0.34–0.36ms 快）。q5_0@rvv-clang **named-X 0.794**（墙见 §2）。**未测 roofline·不宣"内存墙满分"**（roofline probe = queue）。
- **0 verified hand-brick**：全格 opp = stock block-dot 单实现折中/破损（k1 `OPP_REPACK_SYMS=0`·rvv VLEN128 gate-off）。无 hand-brick 强对手。

**★部署现实（关键 honesty）**：
- **rvv(VLEN128)**：front-door decode **主动构造** hl8 repack-GEVM leaf（`path_selection_reason="repack-kept-q4_0-vlen128-decode"`）→ 本役测的是 **真部署-候选 decode 路**（genuine）。
- **k1(VLEN256)**：front-door decode **主动 DECLINE** repack（`"block-dot-decline-vlen256-decode-measured-negative"`）→ 出货 block-dot。本役 k1 repack-GEVM leaf = **force-constructed what-if**（除 q5_0/q5_1 已由 C1 per-format measured-gate 部署·`d109d6ed2`）。q4_0@k1 2.575× = **what-if 候选**（可作 q4_0@k1 per-format gate 立项素材·非已部署）。

**转补测/blocked（§4）**：q4_1 / q8_0 @双板 · K-quant q2_K/q3_K/q4_K/q5_K/q6_K @双板 = **BLOCKED-ON-CONSTRUCTION**（leaf 或 driver-side oracle 缺·如实报·**0 造数**）。

---

## 1. ★M=1 GEVM harness 构造实录（durable·可复现）

**驱动**：`A2-batch4-gemm-decode-M1-raw/flat_gevm_m1_driver.cpp`（新造·模型 = 证实的 `k1-gevm-sweep/raw/q5x_driver.cpp` + `flat_gemm_paired_driver.c` 的 oracle/repack）。
- **OURS** = repack-GEVM leaf：**ONE** GEVM call over nc 列·weight = x16-interleaved（block_qX_0x16·stride q4_0=288/q5_0=352/q5_1=384）· activation = **PLAIN 单 q8 向量**（M=1·无行交织·activation_block_stride=34/36）。
- **OPP** = 该板真派发 `ggml_vec_dot_<fmt>`（stock libggml-cpu.so）·per-column（nrc=1）over 同 plain blocks = **block-dot @ M=1**。
- **GATE** = ZERO-MODEL 独立 scalar oracle（从 plain 输入重算·零复用 leaf 中间量）·per column。判据 = oracle mism（rel≥1e-3）。
- **ratio_cold = opp_med / ours_med**（≥0.8=PASS / <0.8=named-X）。cold = 32MiB flush（k1 64× L2 无 L3 / rvv >> L2）· CLOCK_MONOTONIC · median+relIQR · core-pinned。

**leaf 生成（front-door via weft-opt·可复现·非 stale-archive）**：
```
# rvv hl8（VLEN128·front-door 真构造 decode 路）:
weft-opt test/Target/RVV/q4-0-q8-0-repack-gemv-full-pipeline-export-e2e.mlir \
  --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp
weft-opt k1-gevm-sweep/raw/q5_{0,1}_typed_vlen128.mlir --weft-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp
# k1 hl16（VLEN256·authored one-strip·front-door declines→force-construct）:
weft-opt test/Conversion/RVV/rvv-to-emitc-repack-gemv-q4-0-q8-0-vlen256.mlir --weft-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp
# q5_{0,1}@k1 = k1-gevm-sweep/kernels/q5_{0,1}_gevm_repack_mf2.c（已部署核·C1）
```
- **leaf ABI（两族·driver 按板/格声明）**：q4_0-rvv=`(n,s,nc,vx,vy)` 5-arg；q4_0-k1=`(n,s,vx,vy,nc)` 5-arg；q5_0/q5_1=`(n,s,nc,vx,bx,vy,by,nrc)` 8-arg（nc@arg3·两板同）。
- **build seal**：k1 clang-18.1.8 `-O3 -march=rv64gcv_zfh_zvfh_..._zvl256b -fno-integrated-as -DBOARD_K1`·stock md5 `871169a0` before==after。rvv 双域 = clang-18.1.8（`/opt/tcrv-toolchains/llvm-18.1.8`·`-fno-integrated-as --gcc-toolchain=…/gcc-15.2.0`）+ gcc-15.2.0（`/opt/tcrv-toolchains/gcc-15.2.0`）·march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs·stock md5 `d1adc634` before==after。全 leaf **fp16-libcall-free**（vsetvl q4_0=5-7 / q5_0=7 / q5_1=9-11·无 spill 病）。

---

## 2. 逐格 cold / 判读 / 成色 / 探针（N=25 median · 2-seed · 2 trials）

### q4_0 decode M=1 GEVM
| 板·域 | opp_sym（探针） | ours_ms | opp_ms | ratio(2seed×2trial) | median | predreg | 成色 |
|---|---|---:|---:|---|---:|:--:|---|
| rvv·gcc-15.2(部署-clean) | `ggml_vec_dot_q4_0_q8_0@0x91854` | 0.121–0.126 | 0.851 | 7.01/6.96/6.77/6.86 | **6.909** | **PASS** | 便宜·light-vec 弱(gate-off) |
| rvv·clang-18(对称-micro) | 同上(compiler-insensitive) | 0.208–0.214 | 0.849–0.851 | 3.98/4.07/4.07/4.08 | **4.071** | PASS | 便宜 |
| k1·clang-18(对称·what-if) | `ggml_vec_dot_q4_0_q8_0@0x9f6dc` | 0.351–0.359 | 0.918–0.921 | 2.62/2.58/2.57/2.57 | **2.575** | PASS | 便宜·what-if(front-door declines) |
- **byte-exact**：repack_leaf vs opp mism=**0/512**（bit-f32）·vs oracle mism=0/512（maxrel 1.3–2.9e-4·f32 跨块 order·nbad=0=正确核）。
- **成色**：big 4–7× **便宜档**（opp q4_0 block-dot light-vec 弱·rvv VLEN128 gate-off·opp_ms 0.85 >> q5_0 0.34）。**非硬赢**。

### q5_0 decode M=1 GEVM
| 板·域 | ours_ms | opp_ms | ratio | median | predreg | 墙/成色 |
|---|---:|---:|---|---:|:--:|---|
| rvv·gcc-15.2 | 0.360–0.364 | 0.342 | 0.948/0.945/0.952/0.938 | **0.947** | PASS(near-parity) | better-vec 中 opp·near-parity |
| rvv·clang-18 | 0.428–0.476 | 0.340–0.344 | 0.794/0.723/0.795/0.794 | **0.794** | **named-X** | 墙=**repack compiler-sensitive**（gcc 0.947 恢复 parity）+ memory-leaning near-parity·opp block-dot 编译器不敏感(0.340↔0.342) |
| k1·clang-18(免测确认·C1 已部署) | 0.895–0.955 | 1.134–1.190 | 1.187/1.195/1.218/1.215 | **1.205** | PASS | C1 per-format gate deployed(`d109d6ed2`)·本役独立复算确认 |
- **byte-exact** vs opp mism=0/512·vs oracle 0/512（maxrel ≤6.9e-5）。

### q5_1 decode M=1 GEVM
| 板·域 | ratio | median | predreg | 成色 |
|---|---|---:|:--:|---|
| rvv·gcc-15.2 | 1.101/1.078/0.891/1.107 | **1.090** | PASS(near-parity) | better-vec 中 opp |
| rvv·clang-18 | 0.885/0.935/0.927/0.930 | **0.928** | PASS(near-parity) | compiler-sensitive(gcc 1.090) |
| k1·clang-18(免测确认·C1) | 1.310/1.323/1.324/1.282 | **1.317** | PASS | C1 deployed 独立复算确认 |
- vs oracle mism=0/512（maxrel ≤1.5e-4·q5_1 random-scale FMA-order benign）。

**★[CASE-COMPILER-ASYMMETRY] 本役再证**：repack-GEVM leaf **compiler-sensitive**（q4_0 clang 4.07↔gcc 6.91·q5_0 clang 0.79↔gcc 0.95）；opp block-dot **insensitive**（q4_0 opp_ms 0.849↔0.851·q5_0 0.340↔0.342）。故 **rvv 主账=gcc-15.2 部署-clean**（rvv 出货 gcc-15）·**clang-18=对称-micro**（跨板可比 k1·带 opp-insensitive caveat）。

### q4_K decode M=1 GEVM（K-quant·gate=repack_leaf vs ggml agreement·非独立 ZERO-MODEL）
| 板·域 | leaf codegen | ours_ms | opp_ms(ggml q4_K block-dot) | ratio(2seed×2trial) | median | predreg | 墙/成色 |
|---|---|---:|---:|---|---:|:--:|---|
| k1·clang-18(what-if) | vsetvl=57 clean | 0.522–0.526 | 0.804–0.806 | 1.53/1.53/1.54/1.54 | **1.535** | **PASS** | 赢·但 opp=k1 q4_K block-dot **偏弱**(0.80ms)·非 hand-brick·hl8 leaf VLEN256 half-util(未占满宽·真部署 hl16 更快)·front-door declines(what-if) |
| rvv·clang-18 | vsetvl=57 clean | 0.589–0.800 | 0.215–0.231 | 0.29/0.36/0.36/0.37 | **0.361** | **named-X** | 输·墙=**opp 强 native-vec**(0.22ms fast)+**super-block fold@M=1 不摊销**(dmin/bsums/6-bit unpack epilogue·[PAT-1] format-keyed 适用边界·同 q4_K@ime 0.909× 稀释) |
| rvv·gcc-15.2 | **vsetvl=1387** | 3.34–3.38 | 0.216–0.229 | 0.066/0.069/0.065/0.064 | **0.066** | **named-X** | 输·墙=**[CASE-KQUANT-GCC-CODEGEN] gcc-death**(super-block repack leaf 全 regfile spill/scalarize·1387 vsetvl vs clang 57·**编译器 codegen 病理·非算法**) |
- **正确门**：repack_leaf vs opp(ggml) mism=**0/512**（rel≥1e-3·maxrel ≤1.7e-5·多数 0.00e+00 bit-exact）。**注**：gate = 我方 repack leaf vs ggml block-dot **两独立 codegen 一致**（强正确性证据）·**非** from-scratch 独立 ZERO-MODEL scalar oracle（本役未 author q4_K super-block scalar decode·如实标·weaker gate）。weight repack `pack_w`(stride 2304·scale-region custom layout) byte-exact 复用 `kquant_repack_verify_q4K.c`。
- **★board+opp-strength 分裂（honest 关键）**：verdict 由 **opp 强度** + **board** 定：rvv q4_K block-dot(gcc-15 stock·0.22ms) 比 k1(clang-18 stock·0.80ms) **快 3.6×**（rvv native-vec 强对手）→ k1 赢弱 opp·rvv 输强 opp。**非我方 kernel 双板异**(k1 leaf 0.53ms ≈ rvv-clang leaf 0.60ms·近同)·**是 opp 强度异**。**成色**：k1 1.535× = **beat 弱 stock block-dot**（非 hand-brick·非便宜档但 opp 偏弱）；rvv = **genuine LOSS**（C3′ 负结果·super-block decode@M=1 format-keyed 边界·档案级教材）。

---

## 3. ★T3 回填清单（★留主会话机算入库·本 agent 不动 T3·区分 decode·禁继承）

**主表 `T3_master_rebuild.csv`（gemm_tile·regime=DECODE 行·仅下列已真测格填·余保持 pending）：**

| op | format | regime | rvv_disp | rvv_cold | k1_disp | k1_cold | 成色/域 tag |
|---|---|---|---|---:|---|---:|---|
| gemm_tile | q4_0 | **decode** | PASS | **6.909**(gcc-15.2 deploy-clean)/4.071(clang18-micro) | PASS | **2.575**(clang18·**what-if**) | 便宜·light-vec·rvv=真front-door路·k1=force-construct(front-door declines) |
| gemm_tile | q5_0 | **decode** | PASS | **0.947**(gcc deploy)/0.794-X(clang micro) | PASS(**deployed**) | **1.205**(clang18·C1 `d109d6ed2`) | near-parity·better-vec·rvv compiler-sensitive·k1=C1 已部署独立复算确认 |
| gemm_tile | q5_1 | **decode** | PASS | **1.090**(gcc deploy)/0.928(clang micro) | PASS(**deployed**) | **1.317**(clang18·C1) | near-parity·better-vec·k1=C1 已部署确认 |
| gemm_tile | q4_K | **decode** | **named-X** | **0.066**(gcc·GCC-DEATH)/0.361(clang) | PASS | **1.535**(clang18·what-if) | ★board-split·rvv genuine LOSS(强 opp+super-block fold@M=1)·k1 win(弱 opp·hl8 half-util)·gate=vs-ggml-agreement(非 ZERO-MODEL) |

**36-col T3_A(rvv)/T3_B(k1) 关键列**：cold_ratio=col30（上表 median 填 col30·rvv 主填 gcc-15.2 deploy-clean·clang18-micro 入 col34 footnote）；opponent_grade=**light-vec**(q4_0)/**better-vec**(q5_0/q5_1)；opponent_symbol=`ggml_vec_dot_<fmt>`（rvv q4_0@0x91854/q5_0@0x91984/q5_1@0x91abe·k1 q4_0@0x9f6dc/q5_0@0x9f81c/q5_1@0x9f942）；compiler_axis=rvv `gcc15.2-deploy=MAIN`+`clang18-micro=footnote(opp-insensitive)` / k1 `clang18-sym=MAIN`；hardgate_0p8=**in-denom**（q4_0 PASS双板·q5_0/q5_1 rvv near-parity[q5_0-clang single named-X 0.794·gcc 恢复]·k1 PASS）；ledger_account=`matmul-kernel-sym`·**regime=decode**（≠prefill 行·禁混）。
- **provenance 指针**：`A2-batch4-gemm-decode-M1-raw/logs/{k1,rvv}_run.log` + `{k1,rvv}_build_seal.txt`（raw A/B·byte-exact·2-seed）。
- **★q4_0@k1 decode 成色标注 = what-if**（front-door VLEN256 declines repack·非已部署·可作 per-format gate 立项素材）。q5_0/q5_1@k1 = **C1 已部署**（本役独立复算 confirm·非新登记）。

**★DECODE 行余格（q4_1/q8_0/iq4_nl @双板·K-quant q2_K..q6_K @双板）= 保持 pending·禁填**（§4 blocked）。

---

## 4. blocked / 转补测清单（如实报·0 造数·下一役承接）

| 项 | 缺（精确） | 已就绪 | 剩余构造 | 板 |
|---|---|---|---|---|
| **q4_1 decode** | 无 clean weft-opt leaf 源 | driver q8_1 机制已在·oracle 易加 | q4_1 GEVM leaf：dataflow 用 **retired monolithic op**（`repack_gemv_q4_1_q8_1`·lower-to-emitc 拒 exec.variant）·front-door 输入缺·archived inc(g5/g7·tcrv_/weft_)可复用但非 weft-opt 现生。须补 q4_1 front-door quant_contraction 输入或授权复用 archived inc | rvv+k1 |
| **q8_0 decode** | 无 clean GEVM leaf | — | q8_0 repack-gemv 仅 dataflow(multi-module)·`gevm_q8_timing_driver` 用 retired monolithic `repack_gemv_q8_0_q8_0`。须补 q8_0 typed-region single-module 源 | rvv+k1 |
| **q4_K decode** | ✅ **本役完成**（§2.4·双板真测·k1 1.535× PASS·rvv 0.361/0.066 named-X） | driver `kquant_gevm_m1_driver.cpp` + leaf `q4_K_gevm_hl8.c` | 剩：独立 ZERO-MODEL q4_K scalar oracle（现 gate=vs-ggml-agreement）· k1 hl16 leaf（现 hl8 half-util at VLEN256） | done |
| **K-quant q2_K/q3_K/q5_K/q6_K decode** | driver-side repack+q8_K per-format 缺 | **leaf export PROVEN**（`test/Conversion/RVV/rvv-to-emitc-repack-gemv-{q2,q3,q5,q6}-K-q8-K.mlir` → weft-opt lower-to-emitc → mlir-translate·同 q4_K 路 clean·5-arg ABI `(n,s,vx,vy,nc)`）·driver 模板 = 本役 `kquant_gevm_m1_driver.cpp`（换 pack_w+block struct 即可） | driver-side：per-format super-block x16 repack + q8_K 单向量 act + gate。q3_K/q6_K repack-verify 有参考(`kquant_repack_verify_q{3,6}K.c`)。**预注册预期**：super-block fold@M=1 rvv 强 opp 大概率 LOSS（同 q4_K·format-keyed 边界）·**未测不作结论·0 造数** | rvv+k1 |
| **q5_0@rvv-clang named-X 复核** | roofline probe 缺 | ratio 0.794(clang)/0.947(gcc) | roofline probe 验双方是否同贴 DRAM（memory-wall 满分判据）·或接受 near-parity(compiler-sensitive·gcc 恢复 parity) | rvv |
| **rvv 严格 clang-18 clean-opp** | opp=gcc-15 stock | clang18-micro 已测(opp-insensitive) | opp libggml `-fno-integrated-as` clang-18 重编→严格双域(同 batch2 §6) | rvv |

**★便宜档/near-parity 纪律**：本役 0 verified hand-brick·q4_0 big-multiple=便宜（弱 opp）·q5_0/q5_1=near-parity（memory-leaning·未验 roofline）。**计数纪律**：decode kernel-sym 覆盖=C3′ 证据·**NOT e2e·NOT perf-covered·禁互推·禁写"加速 N kernel"于 e2e 语境**。

---

## 5. 污染纪律 + restore（双板）

- **cold 协议**：32MiB flush（k1 64× L2 无 L3=真 DRAM cold / rvv >> L2）·N=25 median+relIQR·2-seed{0xC0FFEE1,0x1357ACE}·within-proc paired·warmup dropped·K=2048 nc=512（ws 720KB weight cold）。
- **restore 双证**：k1 stock md5 `871169a0` before==after UNCHANGED·rvv stock md5 `d1adc634` before==after UNCHANGED（只读·仅 build ours .o + link·未触 .so·未 git add/commit）。
- **stray**：k1/rvv `pgrep bench`=0·scratch `/tmp/g8_a2b4_{k1,rvv}` 留 raw（logs 已拷回 raw/logs）。load-gate：k1 core0 idle=97%·rvv core32 gov=performance 2.6GHz。
- **disjoint-pin**：k1 taskset -c 0·rvv taskset -c 32。

## durable files
- `A2-batch4-gemm-decode-M1.md`（本文）
- `A2-batch4-gemm-decode-M1-raw/flat_gevm_m1_driver.cpp`（FLAT M=1 GEVM paired 驱动·新造·ZERO-MODEL 独立 oracle gate）
- `A2-batch4-gemm-decode-M1-raw/kquant_gevm_m1_driver.cpp`（K-quant M=1 GEVM 驱动·q4_K·gate=vs-ggml-agreement·可扩 q2/q3/q5/q6_K）
- `A2-batch4-gemm-decode-M1-raw/run_flat_gevm_{k1,rvv}.sh`（双板 build+cold 脚本·可复现）
- `A2-batch4-gemm-decode-M1-raw/leaves/*.c`（weft-opt 生成 leaf：q4_0 hl8-rvv/hl16-k1·q5_0/q5_1 hl8-rvv·q5_0/q5_1 k1·q4_K hl8）
- `A2-batch4-gemm-decode-M1-raw/logs/{k1,rvv}_run.log` + `{k1,rvv}_build_seal.txt` + `q4_K_raw.txt`（raw A/B·byte-exact·2-seed·探针）
