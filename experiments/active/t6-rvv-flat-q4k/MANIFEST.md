# T6 batch-rvv — 整模型 e2e 传导（FLAT 5 + K-quant q4_K）· rvv 板

- **campaign**: 测量总攻 / T6（内核轴对称候选 → perf-covered 绿格的 e2e 传导验证）· rvv 批
- **role**: 用整模型 llama-bench 分相（prefill/decode）A/B 验 flat-covering-batch1（f78ad2da）证得的
  FLAT 5 格 kernel-轴对称 micro 赢是否**传导**进 whole-model forward；对手 = 板自有 stock ggml block-dot。
- **status**: BOARD-MEASURED 2026-07-11 · rvv (openEuler 24.03 LTS, localhost.localdomain, VLEN128, 64c,
  core40-43 perf-gov 2.6 GHz) · HEAD 保持只读（无 git，本 agent 零提交；主会话另提交了 batch-k1）
- **★核心裁决（决定性·高价值 negative）**: **VLEN128 板上仅 q4_0 被 wire 进 forward**；q4_1/q5_0/q5_1/q8_0/q4_K
  在 VLEN128 **未路由** → 内核轴 micro 赢**无法 e2e 传导** = 具名 **[GAP-FLAT-E2E-ROUTING]**。

## board identity fingerprint
- host=localhost.localdomain / openEuler 24.03 LTS / kernel 6.12.66 riscv64 / VLEN128 /
  rv64imafdcv+zfh+zvfh+zba+zbb+zbc+zbs / 64c / core40-43 perf-gov 2.6 GHz / DVFS span 0.00%
- toolchains: A(ours)+B(stock) 都 **gcc-15.2.0**（= 对手 .so 编译器；本批 = **kernel-symmetric 账**，
  非 clang-deploy 账）；link/harness gcc-15.2.0 neutral。
- A(ours) = `/home/ubuntu/tcrv-llamacpp/build-gcc15-rv64gcv`，deployed `libggml-cpu.so` md5
  **75f20b5fddf2faf2b18d7a88b1bfb4b5**（含 `tcrv_emitc_ggml_repack_gemm/gemv_q4_0_q8_0` 符号 +
  repack.cpp:4592 `TCRV-WINB-ON-TOGGLE`）。
- B(stock) = `/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv`，`libggml-cpu.so` md5
  **d1adc634c2ca04ffc389536b30d9d9c4**（tag b9692 / f3e1828，与 batch1 对手同一）。
- ★两 .so md5 测前=测后不变 = **零 stock 改动**；board /tmp scratch 测后清空（仅遗留他会话 7/6 的
  `/tmp/tcrv_dboard`，非本批，未动）。

## model registration
| model | 来源 | quant | sha256(16) | 用途 |
|---|---|---|---|---|
| tinyllama-q4_0.gguf (606.5 MiB, 1.10B) | 板既有 `/home/ubuntu/tcrv-llamacpp/models/` | Q4_0 | da3087fb14aede55 | q4_0 e2e A/B（wired 格）|
| tinyllama-q8_0.gguf (/tmp, 测后删) | 本批 `llama-quantize --allow-requantize` 自 q4_0（保真无关：A/B 同模型只验路由）| Q8_0 | — | routing-gap 对照（banner-absent + A/B parity）|
| tinyllama-q4_1.gguf (/tmp, 测后删) | 同上 requantize | Q4_1 | — | routing-gap 对照（banner-absent）|
> 无 f16/f32 tinyllama 源，只 q4_0；requantize 产物仅供 A-vs-B **路由**对照（两侧同模型、保真不影响路由结论）。
> q4_K e2e 未新测：板既有 DeepSeek-8B-Q4_K_M 存在，但 q4_K 在 VLEN128 未路由（结构已定，见下），无 A/B 旋钮。

## harness（复用既有 T6 骨架 tools/e2e-harness/{run_e2e.sh,board/*}，本批未改 board 脚本）
- `run_e2e.sh` → preflight（fail-closed 4 门：同编译器/march-complete/libcall-free/VLEN-fingerprint）→
  `phase_split_ab.sh`（llama-bench 原生 -p PP 分相 prefill + -n TG 分相 decode，配对交替 2 pass × 5 rep，
  taskset pin，JSON 逐 rep 样本）→ `correctness_gate.sh`（greedy-token A==B 一致 + logits sanity）→
  `aggregate_e2e.py`（分相中位 + IQR% + bootstrap 95%CI + T-N between-pass floor + DVFS 守卫 + PARITY/DIFFERENCE 判决）。
- preflight: **4/4 GREEN**（A/B 同 gcc-15.2.0 同 flags Release；march 覆盖板关键扩展；两树 ggml-cpu.o
  无 `__extendhfsf2`-类 fp16 softfloat libcall；board VLEN=128==target）。

## 逐格分相 e2e + 传导会计（数据: transmission_accounting.csv + results/）
| fmt | micro kernel-轴（gcc-sym·batch1）| wired@VLEN128 | e2e-prefill | e2e-decode | 稀释 | 判读 |
|-----|-----|-----|-----|-----|-----|-----|
| **q4_0** | 6.76× | **YES**（TCRV-WINB-ON-TOGGLE·banner FIRES）| **5.755× DIFFERENCE** | **1.910× DIFFERENCE** | 无（两相传导）| **perf-covered 维持+强化** |
| q4_1 | 6.86× | NO（无 riscv repack 分支·banner absent）| — 不可测（A==B）| — | 100%（未路由）| kernel-axis-only 黄 |
| q5_0 | 1.23× | NO（无 riscv repack 分支）| — 不可测 | — | 100% | kernel-axis-only 黄 |
| q5_1 | 1.41× | NO（无 riscv repack 分支）| — 不可测 | — | 100% | kernel-axis-only 黄 |
| q8_0 | 4.10× | NO（case128 break·DECLINE·banner absent）| **1.049× PARITY**（实测）| **1.051× PARITY** | 100%（micro4.10→e2e1.05）| kernel-axis-only 黄 |
| q4_K | S6撤回/kernel-sym-null | NO（case128 break·K-quant 无 toggle）| — 不可测（无 A/B 旋钮）| — | — | 非 perf-covered |

### q4_0 e2e 细节（唯一 wired 格·干净）
- **clean warm 3-pass**（重测·model 已入 page-cache·heavy warmup 消除冷启）: prefill ours 30.17 vs stock 5.24 t/s
  = **5.755×**（pass-spread 0.08%）; decode ours 9.02 vs stock 4.72 t/s = **1.910×**（pass-spread 0.51%，CI[1.894,1.913]）。
  DVFS span 0%。两相均 **DIFFERENCE**（过 2× floor + CI 排除 1.0），符测量卫生 SOP。
- **首跑（冷+暖混）**: prefill 组合中位 4.27× 但 ours-side IQR 70% / between-pass floor 108%
  （冷 pass1=14.5 vs 暖 pass2=30.1 = 2× 冷启摆动，stock 侧稳 5.23/5.24）→ 该相**作废重跑**（上表用干净暖值）。
  decode 首跑已干净 1.903×（与重测 1.910× 一致）。
- **correctness**: GREEN（3/3 prompt greedy-token A==B 一致 + 无 NaN/Inf）。注：normalize 后可见文本多为
  llama-cli banner（tinyllama chat 单轮 -st 生成体被裁），A==B 成立但内容主要是 banner；**真正确性基座 =
  prior-seal 的 prefill kernel A==B 字节相同**（memory q4-0-e2e-is-routing-not-kernel）。
- **★披露（memory q4-0-e2e-is-routing-not-kernel）**: q4_0 e2e 赢 = **routing/dispatch 白嫖 + decode 内存局部性**，
  **非 kernel-质量-vs-手调 asm 赢**。上游 f3e1828 已建整条 q4_0 repack 流水线，我方仅翻 repack.cpp:4592 一行
  VLEN128 gate（prefill kernel A==B 字节相同）。措辞层级 = 能力键控构造+路由，不外推 kernel 微质量。

### routing-gap 实证（4 重确认·决定性）
1. **源码**: repack.cpp — q4_0 case128=`return &q4_0_16x1_q8_0`（TCRV-WINB-ON-TOGGLE）; q8_0/q4_K/q2_K/iq4_nl
   case128=`break`//TODO（仅 case256 路由）; q4_1/q5_0/q5_1 **零 riscv repack 分支**。
2. **deployed .so 符号**: 仅 `tcrv_emitc_...q4_0` 存在，无 q8_0/q4_1/q5_x TCRV 符号。
3. **独立佐证**: 既有 `kquant_transmission_amdahl.sh` 头注明文 "q4_K not wired into ggml mul_mat... no A/B knob exists"。
4. **本批实证 engage-banner 对照**: A-tree llama-bench 对 q8_0 / q4_1 模型 **无 TCRV banner**（block-dot 回退）;
   对 q4_0 模型 banner FIRES（GEMM+GEVM ENGAGED）; q8_0 A/B e2e = **1.05× PARITY**（micro 4.10× 完全被非路由稀释）。

## 判读（预注册·照判照走·零未定义）
- **q4_0**: e2e ≥parity 两相（prefill 5.76× / decode 1.91×·DIFFERENCE·correctness GREEN·卫生干净）→
  micro∧e2e + selector-routing 两门对 q4_0 **CLOSED**。但 q4_0 **本已 perf-covered**（ROADMAP 系统账 routing-win 5/8 disclosed）→
  本批 = **perf-covered 维持+强化**（补干净分相 e2e 证据），**非新增绿格**；routing-win 披露不变。
- **q4_1/q5_0/q5_1/q8_0**（FLAT 4 格）: VLEN128 未路由 → e2e 无法传导（q8_0 实测 1.05× parity 佐证）→
  **kernel-axis-only 黄格** · 具名 **[GAP-FLAT-E2E-ROUTING]**（selector/repack VLEN128 gate 未接这 4 格）。
- **q4_K**: VLEN128 未路由（K-quant 无 toggle）→ 无 A/B 旋钮 → e2e 不可测；且 kernel-轴 S6 已撤回（clang-vs-gcc
  artifact）→ **非 perf-covered**（构造净新在案，perf 立不住）· 具名 **[GAP-FLAT-E2E-ROUTING]（K-quant 变体：需整条链路 wire）**。

## perf-covered 对本批贡献
- **新增绿格: 0**。5 格（q4_1/q5_0/q5_1/q8_0/q4_K）全被 **routing/接线缺** 挡在 e2e 门外 = 高价值 negative。
- **q4_0**: 已绿·本批补充干净分相 e2e 传导证据（prefill 5.76× / decode 1.91×），两门对 q4_0 关闭；
  SEALED-WIN 候选措辞（若主会话据此登记）见下。
- 结论与并行 batch-k1 同律汇合（主会话 ROADMAP afb1899c）: **perf-covered 瓶颈 = 接线（routing/wiring）+ micro↛e2e 封顶**。

### SEALED-WIN 候选措辞（q4_0·供主会话据数据裁·八门口径·routing-win 披露）
> q4_0 whole-model e2e 传导（rvv-openEuler-VLEN128·tinyllama-1B-Q4_0·kernel-symmetric gcc-15.2.0 双树账·
> 对手 = stock ggml block-dot md5 d1adc634 dispatched VLEN128 路径）: **prefill 5.755× / decode 1.910×**
> （DIFFERENCE·correctness GREEN·DVFS 0%·pass-spread<0.6%）。机制 = **能力键控前门构造 + repack 路由**
> （上游具等效 repack 路径但 VLEN128 gate OFF；我方翻一行接入）+ **decode 内存局部性**；**非 kernel-质量-vs-手调赢**。
> 八门: byte-exact ✓ / 对手对称(gcc-sym) ✓ / 双账本 ✓ / 对手身份探针 ✓ / micro∧e2e ✓(本批补) /
> selector-routing ✓(banner FIRES) / 纪律 ✓ / 措辞 ✓(routing-win 披露)。

## [NG-4] 纪律
本批 beat 只对 q4_0（已绿·routing-win 披露）成立；4+1 格全 kernel-axis-only 黄格 + 具名 GAP，不冒绿、不硬造传导。
数字全 board-measured，双账本 = kernel-symmetric（A/B 同 gcc-15）。

## data-only cell · files
- `transmission_accounting.csv` — 6 格传导会计四列（micro / wired / e2e-prefill / e2e-decode / 稀释 / 判读 / GAP）
- `results/q4_0-tinyllama-vlen128/` — run_e2e.sh 产出（preflight/phase_split_raw/correctness/aggregate/evidence.json）
- `results/q4_0_reprefill_raw.txt` — 干净暖 3-pass 重测（prefill 5.755× / decode 1.910×）
- `results/routing_gap_probe.txt` — engage-banner 对照 + q8_0 A/B parity 实证
- `results/q4_0_harness_stdout.log` — 首跑全日志（含冷启诊断）
