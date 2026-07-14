# G5-M3 L-接线③ — IME e2e forward bridge 侦察（deploy-readiness · IME 桥）

> **campaign**: G5 接线战役 · **M3-ime-bridge-recon = IME e2e 桥部署就绪度侦察**（用户 ③ 优先级 · `[GAP-IME-E2E-INTEGRATION]` 的解）
> **role**: 纯 scout/docs — 本仓 read-only 静态分析 + k1 板 **read-only 核**（禁执行集成 · 禁 build/deploy · 禁 git）。零 lib/schema/ROADMAP 触碰。
> **HEAD (TianChen-RV)**: `61596dcc` 未变。
> **定位（M3 SOP §1 L-接线③ · ROADMAP 排②之后）**: IME 桥 = **最重接线层**（跨框架 forward bridge · 矩阵范式 · 换板 k1）。评估口径 = **N2 结构价值（IME 家族接入 e2e · C1 跨范式接线闭环）为主 · perf 为辅（诚实低预期 · micro↛e2e 重核）**。
> **来源（全只读）**: 板 `/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu/spacemit/{ime.cpp,ime_kernels.h,repack.cpp}`（vendor IME 集成机制活核）· 本仓 `lib/Plugin/IME/IMEBackendEmissionDriver.cpp`（emitter）· `test/Target/IME/*.c`（seal harness · Apack/Bq4 布局活核）· `test/Conversion/EmitC/ime-*.mlir`（lit）· `experiments/active/{t6-k1-ime-q4k-e2e,g4-m3-ime-paradigm-t5b}/MANIFEST.md`（`[GAP-IME-E2E-INTEGRATION]` HEADLINE + T5b 范式杠杆数）· `M0-接线机制解剖.md` 乙.2 · M3 SOP `docs/reports/2026-07-12-G5-接线机制方法学-SOP.md` §1。
> **canon 锚（引用不重述）**: `kernel-wins-dont-transplant-to-e2e`（micro↛e2e 铁律·IME 5.51× kernel→0.86× decode）· `k1-ime-n2-hardware-candidate`（N2 已证·per-hart IME 0–3·march xsmtvdotii）· `zero-model-adjudication-cert-hardening`（ZERO-MODEL int32-exact · 部署变体≠证过变体）· core-invariants `[NG-2]`（不改 ggml 图 schedule）·`[NG-4]`（接线≠自动转绿）。

---

## 0. 一页速览（判读口诀）

| 问题 | 结论 |
|---|---|
| IME 3 格 emitted 状态？ | q4_0/q8_0/q4_K@ime 全 **silicon-sealed standalone**（k1 硅 `vmadot 0xe210312b` int32 0-diff）·但 **未接任何 ggml forward**（板 grep tcrv IME 符号 = **0 hit·live 复核 grep-exit=1**）= `[GAP-IME-E2E-INTEGRATION]` 确证。 |
| IME 接线与 RVV repack 挂点差别？ | **根本不同构**。RVV = 翻一行 gate + `#include` + intercept（3 挂点·2 文件）。IME = 无"可翻的 gate"——emitted kernel 是 **int32 MAC 微 tile 叶子**（固定 M/N/K），须**包成完整 GEMM**（scale-fold epilogue + shape driver + 激活/权重布局桥 + hook `forward_mul_mat`），且**换板 k1**。 |
| emitted kernel 直接可用于真 forward？ | **否**。三层缺口：① 输出 = **单个 full-K int32 累加·无 per-block scale**（真 q4_0 须逐 32-block d_a×d_w 折叠 f32）② M/N/K = **compile-time 常量固定微 tile**（真 forward runtime 变形）③ Apack/Bq4 = **tcrv fragment-major tile 布局**（≠ ggml native）。 |
| perf 预期？ | **诚实近零 e2e green**。vendor 全接线 IME GEMM（本模型·同板·同工具链）已证 **无干净 IME-unit e2e 赢**（prefill 1.49–1.61× 大半是 kernel-family swap·decode floor 1.46× = M=1 阵列跑不起）→ tcrv ~2.09× **compute-account 不传导**。 |
| 建议？ | **perf 名义不值当下做**（vendor toggle 已是 ceiling proxy·近零 payoff）。**N2 结构名义**若为 C1 跨范式完整性做 → 曳光弹首格 = **q4_0@ime**（板已有 q4_0 模型·零 provisioning·最简 scale fold·seal-proven tile）· correctness-first · 预注册 expecting parity/非绿。 |

---

## 1. 现状盘点

### 1.1 IME 3 格 emitted kernel 状态（silicon-sealed · 未接 forward）

| 格 | emitter（`lib/Plugin/IME/IMEBackendEmissionDriver.cpp`） | emitted 函数 / ABI | seal 证据 | forward-wired? |
|---|---|---|---|---|
| **q4_0@ime** | `IMEQ40MatMulTileToEmitCFunc`（:1116）· 由 `Q40DequantCoreOp` + `VmadotMacLeafOp` 两 brick 按 op-identity keying | `tcrv_emitc_<sourceKernel>_<variant>`（例 `tcrv_emitc_ime_q4_0_matmul_kernel_ime_vmadot_matmul_slice`）· ABI = `void(const int8_t* Apack, const uint8_t* Bq4, int32_t* C)`·**M/N/K = compile-time baked**（`mat_m/mat_n/mat_k` FACTS） | `test/Target/IME/q4-0-matmul-tile-int32-{k1seal,oracle}.c`（微 tile M=8,N=8,K=64）+ lit `ime-q4-0-matmul-tile-materialization.mlir` | **NO**（仅 seal+lit 消费） |
| **q8_0@ime** | `IMEQ80MatMulTileToEmitCFunc`（:1237）· `Q80DequantCoreOp`（DIRECT int8 read）+ 复用 `VmadotMacLeafOp` | 同上 ABI（int8 直读·无 nibble 解码） | `q8-0-matmul-tile-int32-{k1seal,oracle}.c` + lit | **NO** |
| **q4_K@ime** | `IMEQ4KMatMulTileToEmitCFunc`（:1358）· **SIX bricks**（`q4_K_scale_min_unpack_core` + `vmadot_mac_leaf` + K-quant 两级 fold） | 同上 ABI + K-quant super-block 两级 scale/min | `q4-K-matmul-tile-int32-{k1seal,oracle}.c` + lit | **NO** |

**共性事实（三格同）**：
- **int32-EXACT 累加器输出**（M1b K1-seal bit-exact 契约）；**per-block fp16 scale fold = separate downstream epilogue**（emitter 注释 `IMEBackendEmissionDriver.cpp:1113-1115` 明示：scale fold 是下游 epilogue·**当前不 emit**）。
- **[GAP-IME-LEAF-PIPELINE] 已闭**：emitter 现发 **register-resident batched MAC leaf**（`vmadotMacKloopHelperBody()` :217·单 vsetvli/K/8 loop v2/v3 驻留/只存一次）= T5b Cell1b ~2.09× compute-account 的使能件（as-emitted un-batched leaf 曾 4× LOSS·已翻正）。
- **消费面**：**仅** `test/Target/IME/*.c` seal + `test/Conversion/EmitC/ime-*.mlir` lit。**无 tcrv→llama IME 桥存在**。板 `/home/bianbu/tcrv-k1-llama/ggml` grep `vmadot_mac_kloop|tcrv_ime|q4_0_matmul_tile` = **0 hit**（live 2026-07-12 复核）。

### 1.2 k1 板 ggml 的 mul_mat 派发链 + vendor IME 挂点（= tcrv 桥的对照锚）

板 `/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu/spacemit/`（vendor SpacemiT ggml-cpu IME backend·活核 read-only）：

- **挂点机制 = `extra_buffer_type` / `tensor_traits`**（与 RVV repack 同族·但更深）：
  1. `ggml_riscv64_spacemit_get_optimal_repack_type(cur)`（`ime.cpp:1257`）按 weight `type` 返回一个 repack `tensor_traits<BLOC_TYPE, INTER_SIZE, NB_COLS>`（q4_0 → `q4_0_16x32_q8_0` 等·`ime.cpp:1235-1251` 注册 14 种 typed traits），经 `ime.cpp:1398` 挂到 weight tensor 的 extra。
  2. `tensor_traits::compute_forward`（switch `ime.cpp:193` `case GGML_TYPE_Q4_0`）→ `forward_mul_mat(params, op)`（`ime.cpp:234`）。
- **`forward_mul_mat` 体（`ime.cpp:234–420`·= vendor 的 RVV 挂点③ 类比·tcrv 桥须对应此层）**：
  - `gemm_m = ne11*ne12*ne13`（= M·激活行/token 数）· `gemm_k = ne10` · `gemm_n = ne01`。
  - **regime split（非硬 M* 阈值·两路都调同一 gemm_kernel）**：`if (gemm_m == 1)`（decode）→ `quantize_a_row_i8` 单行量化；`else`（prefill）→ `quantize_a_4row_i8` 4-row blocking。二者**都**调 `gemm_kernel`（q4_0/q4_1/q4_K → IME1 `ime1::gemm_kernel_i8i4` `:320`；IME2 变体 `:276-282` `gemm_kernel_i8i4[_hp]`）。**decode 也调 IME kernel·但阵列跑不起**（vendor toggle 实证 decode 1.46× = 227-symbol kernel-family swap·非阵列）。
  - `gemm_kernel_i8i4`（`ime_kernels.h:72/109`）= **完整 workspace-aware GEMM**：激活 i8 量化 + 权重 repack 布局消费 + **per-block scale fold → f32 直出** + TCM buffer + `ggml_barrier` 多线程 tiling（`row_align=4` / `NB_COLS` / `gemm_n_stride`）。
- **board fingerprint**：VLEN=256 · 8 harts · **IME on harts 0–3**（`taskset -c 0-3` 必须·per-hart 非对称）· `build-ime`（SPACEMIT=ON·32 `vmadot`·clang-18 `-fno-integrated-as`）/ `build-off`（RVV fallback·0 `vmadot`）· 模型 = **仅 `tinyllama-q4_0.gguf`（637 MB）·无 q8_0/q4_K gguf·无 `llama-quantize`**。tcrv emit 复现工具链 = stock gcc-13 + token `rv64gcv_xsmtvdotii1p0`（或 SpacemiT GCC15.2 fork）。

> **对照锚精化**：vendor `ime1::gemm_kernel_i8i4` IS wired into forward（这就是 tcrv 桥要对标的集成形态）。项目**已有** tcrv→forward 集成机制（RVV q4_0 repack：`ggml_gemv_q4_0_16x1_q8_0`→`tcrv_emitc_...`），只是**从未为 IME 矩阵路径建过**。这是 **integration gap·非 fundamental barrier**（t6-k1-ime-q4k-e2e HEADLINE）。

---

## 2. 集成方案（IME e2e 桥怎么建 · 与 RVV repack 挂点对比）

### 2.1 核心不对称：tcrv IME emitted = MAC 微 tile 叶子 ≠ vendor 完整 GEMM

| 维度 | tcrv IME emitted kernel | vendor `gemm_kernel_i8i4`（真 forward 消费） | 桥需补的缺口 |
|---|---|---|---|
| 输出类型 | **int32 累加器**（单个 full-K·无 scale） | **f32**（per-block d_a×d_w 折叠内建） | **scale-fold epilogue（correctness-critical #1）** |
| per-block scale | **无**（seal harness 全 K 单次 mac_kloop·仅 int32-exact MAC 契约） | 逐 32-block（真 q4_0 每 block 独立 fp16 scale·激活 q8 每 block d_a） | **块级 scale 交织**（须逐 block 调 kernel K=32 + 块间折叠·correctness-critical #1'·比"int32→f32 单 scale"重） |
| 形状 | **compile-time M/N/K 固定微 tile**（seal M=8,N=8,K=64） | runtime `gemm_m/gemm_n/gemm_k`（变 batch/seq） | **shape driver**（把固定微 tile loop over 真问题·或 emit shape-参数化变体·correctness-critical #2） |
| 激活布局 | **Apack fragment-major int8 tile**（`Apack[mi*4*K + kf*32 + ml*8 + kl]`·4×8 fragment·seal harness :133-172 活核） | ggml native f32 行 → vendor `quantize_a_row_i8` 产 vendor 自有 packed 布局 | **激活量化 + Apack 打包桥**（f32→int8→tcrv fragment-major·correctness-critical #3） |
| 权重布局 | **Bq4 fragment-major**（native 18B q4_0 block·但 col-tile n/4·block k/8·nibble→frame idx `(n%4)*8+(k%8)`·seal :121-133 活核） | ggml native q4_0 mul_mat 布局 → vendor repack trait | **权重 repack 桥**（native q4_0 → tcrv fragment-major pack·correctness-critical #4） |
| 挂点 | 无（standalone） | `forward_mul_mat` gemm_kernel 调用点 | **patch `forward_mul_mat` prefill 分支**（gemm_m≥M*）或建平行 tcrv `tensor_traits` |

### 2.2 与 RVV repack 挂点对比（IME 更深·无白嫖）

| | RVV repack（L-接线①·q4_0/q8_0 先例） | IME（L-接线③·本 recon） |
|---|---|---|
| 挂点层 | `arch/riscv/repack.cpp` 的 `ggml_gemv/gemm_<fmt>` body（既有 dispatch slot） | ggml `mul_mat` 的 `forward_mul_mat`（`spacemit/ime.cpp`·矩阵范式·**无既有 tcrv slot**） |
| 净改动 | **翻一行 gate + `#include` + intercept·return**（3 挂点·2 tracked 文件） | **无 gate 可翻**——须包 MAC 叶子成完整 GEMM（scale epilogue + shape driver + 激活/权重桥 + hook·多件净新） |
| correctness 承载 | emitted vl=8 kernel 拦截破损上游 body（1 处） | emitted int32 叶子 + **4 处布局桥 + scale fold**（每处 MIRAGE trap） |
| 换板 | rvv（openEuler VLEN128） | **k1（SpacemiT X60·VLEN256·per-hart 0–3）** |
| 白嫖可能 | q4_0 routing-freebie 存在 | **无**（IME 无上游 tcrv 路径·纯净新桥） |

### 2.3 两条集成路径

- **路径 A（最深·最干净 N2·最重）**：在 `forward_mul_mat` 调用点用 tcrv-emitted 等价物**替换** `ime1::gemm_kernel_i8i4`。须把 tcrv 固定微 tile 叶子 wrap 成匹配 vendor `blk_len/workspace/NB_COLS` ABI 的 runtime-shape f32 GEMM + 补 per-block scale fold。**工作量大**（重造 vendor GEMM 的编排层）。
- **路径 B（曳光弹·薄·correctness-first·推荐若做）**：为 **q4_0 单格**建**最小 tcrv `tensor_traits`-style hook**（或直接 patch `forward_mul_mat` 的 prefill `gemm_m≥M*` 分支），内部：native q4_0 → tcrv Bq4 fragment-major repack → f32 激活 → tcrv Apack 打包 → **逐 32-block 调 tcrv `tcrv_emitc_..._matmul` + 手写 per-block fp16 scale fold** → f32 输出。仅 prefill（阵列有物理意义的唯一 regime）。correctness-first（ZERO-MODEL / greedy A==B vs stock RVV forward）。

---

## 3. ★工作量级 + 分批（诚实 · 多 session）

**净新桥件数（路径 B 曳光弹·q4_0@ime）**：

| 阶段 | 件 | correctness-critical? | 备注 |
|---|---|---|---|
| P0 | board provisioning + host emit `.inc`（tcrv-opt recipe 产 q4_0@ime kernel `.inc`） | — | **q4_0 模型板已有·零 quantize**（vs q8_0/q4_K 缺 gguf）。emit 复现 = SpacemiT/stock gcc token。 |
| P1 | **per-block scale-fold epilogue**（int32 partial × d_a × d_w → f32·逐 32-block） | ★是 #1 | 当前 emitter **不 emit**·须手写或新 emitter brick。**比 RVV carrier 的"int32→f32"重**（块级交织·非单终 scale）。 |
| P2 | **shape driver**（固定微 tile loop over runtime gemm_m/n/k·块级粒度） | ★是 #2 | 固定 M/N/K 微 tile 是 leaf·非 GEMM。 |
| P3 | **激活量化 + Apack fragment-major 打包**（f32→int8→`[mi*4*K+kf*32+ml*8+kl]`） | ★是 #3 | 须 byte-match seal harness Apack 布局。 |
| P4 | **权重 repack 桥**（native ggml q4_0 → tcrv Bq4 fragment-major col-tile pack） | ★是 #4 | nibble→frame idx `(n%4)*8+(k%8)`·MIRAGE trap。 |
| P5 | hook `forward_mul_mat` prefill 分支（gemm_m≥M*·A-tree cp `.ORIG`·可逆） | — | `[NG-2]` 不改图 schedule·仅 kernel 调用点 patch。 |
| P6 | correctness 门（ZERO-MODEL int32-exact 复用 seal harness + greedy A==B vs stock RVV forward + logits sanity）+ objdump `vmadot` engage seal + restore byte-exact | ★门 | 4 布局桥各须 board UT vs 独立 oracle byte-match（build 前 de-risk）。 |

**工作量诚实定级 = 多 session（≥3·大概率更多）**。理由：**4 处 correctness-critical 布局桥 + per-block scale fold（重于 RVV carrier 的单 intercept）+ 无"翻 gate"白嫖 + 换板 provisioning + 固定微 tile 非 GEMM**。对比：RVV q8_0 carrier（L-接线①b·最重的向量格）= 单 session 量级；IME 桥 ≥ 数倍。

**k1 board provisioning**：q4_0 gguf 已在板（零下载）；q4_0 **无需 `llama-quantize`**（vs q8_0/q4_K 缺 gguf+缺 quantize = 额外 provisioning）。march = `rv64gcv_..._xsmtvdotii` / stock gcc-13 token `xsmtvdotii1p0`；`taskset -c 0-3`（per-hart IME）。

**A-tree 可逆规程**（套 M3 SOP §7.3）：patch `forward_mul_mat` call site + `#include` tcrv `.inc`·`cp *.ORIG`·对称 ON/OFF build·测后 `restore` + rebuild + md5 证零 stock 改动·**无 git**·HEAD 不变。

---

## 4. ★perf 预期（诚实 · Amdahl 同域 · micro↛e2e 重核）

**★Amdahl 同域输入铁律**：估算的 kernel 因子必须与目标部署**同域**（同板/同 deployed variant/同输入路径）。IME 桥的**唯一同域已测 proxy = vendor 全接线 IME GEMM 的 e2e toggle**（`t6-k1-ime-q4k-e2e`·同板 k1·同模型 tinyllama-Q4_0·同工具链对称 SPACEMIT ON/OFF），**不是** T5b compute-isolated 的 2.09×（那是 L2-resident 计算隔离 micro·跨域喂进 e2e = garbage-in）。

**vendor ceiling proxy 实测（2026-07-11 `vendor_ime_toggle.csv`·= "若接好 IME GEMM·2× compute 能否传导 e2e" 的最强上界）**：

| regime | OFF（RVV） | ON（vendor IME） | ON/OFF | 判读 |
|---|--:|--:|--:|---|
| pp256（prefill） | 24.90 | 40.06 | **1.61×** | 大半 = SpacemiT kernel-family swap·非纯阵列 |
| pp512（prefill） | 24.26 | 36.18 | **1.49×** | 同 |
| **tg64（decode M=1·CONTROL·阵列跑不起）** | 5.66 | 8.26 | **1.46×** | = 227-symbol kernel-FAMILY swap floor·**非 IME 阵列** |

**读法（load-bearing）**：decode control（M=1·`vmadot` 物理跑不起）= **1.46×** → 意味着 prefill 1.49–1.61× 里 **~1.46× 是 kernel-family swap**（RVV kernel 族整体换代）、**IME-unit 增量仅 ~+0.03…+0.15**·且与 GEMM-vs-GEVM 混淆**不可干净隔离**。→ **即便 vendor 全接线 IME GEMM·本 memory-bound 1B 模型上无干净 IME-unit e2e 赢**。

**传导预期（分相·prefill/decode 永远分开）**：
- **prefill（M≥M*）**：tcrv IME 桥**可能**显 prefill delta（若真路由到阵列）·但**上界 = 同域 vendor 的 ~1.5×·且其中大半非阵列**。且对手身份两难：vs stock RVV（build-off）= 测 kernel-family swap；vs vendor IME = tcrv-vs-vendor（同 IME·测成熟度非范式）。**两者都不是干净 perf-covered 绿格**。
- **decode（M=1）**：阵列跑不起 → **parity/LOSS**（内存墙·q4_0 4-bit 权重·memory-bandwidth-bound）。
- **verdict = likely NO e2e green**。与 `kernel-wins-dont-transplant-to-e2e`（IME 5.51× kernel→0.86× decode）+ vendor-toggle 已测 ceiling 一致。tcrv ~2.09× compute-account **不 e2e 传导**。

**★N2 结构价值（独立于 perf·分开报）**：
- 闭 `[GAP-IME-E2E-INTEGRATION]` = tcrv IME 矩阵家族（N2 non-RVV 第二 family）经**同一 template 接线方法学**（RVV q4_0 repack 部署的跨范式类比）抵达 ggml 真 forward = **C1 模板协议本体在家族#2/跨范式的 extensibility 铁证**（矩阵范式·非向量 repack·最深一层扩展）。
- **此价值不需 perf green 成立**：证接线方法学**跨范式泛化**（向量 repack → 矩阵 GEMM）= C1 headline。诚实 framing = 机制/结构闭环·`[NG-4]` 全程·**非 perf 绿格**。

---

## 5. 建议

### 5.1 是否值当下做

- **perf 名义 = 不值当下做**。理由：vendor 全接线 IME GEMM toggle **已是 ceiling proxy**·已证本模型无干净 IME-unit e2e 赢（decode floor 1.46× = kernel-family·非阵列）。tcrv 桥即便建成·perf payoff 诚实近零·且已被 `t6-k1-ime-q4k-e2e` 的"高价值 negative + vendor-ceiling 参照"**捕获**（欠账已有诚实交付）。**高成本（≥3 session·4 correctness-critical 桥）/ 低 perf payoff**。
- **N2 结构名义 = 有真价值·但边际**。结构价值主要**已由** standalone silicon-sealed cert（三格 int32 0-diff）+ vendor-toggle ceiling 参照交付。桥的**净增** = "tcrv IME 真进 forward"的 C1 跨范式 extensibility demo（对论文 C1 headline 有增量·但非 perf）。
- **优先级建议**：**排在其它更高 payoff 面之后**（② q5_x 已完成；FLAT 剩面/K-quant e2e/[TEMPLATE-AUDIT] 等 perf-covered 直接杠杆优先）。IME 桥作为 **C1 跨范式完整性的可选收尾**·非测量总攻主线。

### 5.2 若做 · 曳光弹路径（首格 = q4_0@ime）

**首格建议 = q4_0@ime**（不是 q8_0/q4_K@ime），理由：
1. **零 board provisioning**：q4_0 gguf 板已有（637 MB）·无需 `llama-quantize`（q8_0/q4_K 缺 gguf+缺 quantize）。
2. **最简 scale fold**：q4_0 单 fp16 scale/block·**无 min**（vs q4_K 两级 scale/min super-block·6 bricks·scale-fold epilogue 复杂度数倍）。
3. **seal-proven tile**：`q4-0-matmul-tile-int32-k1seal.c` 已 int32 0-diff·ZERO-MODEL 门可直接复用为 correctness harness。
4. **founding IME format**·N2 首格。

**曳光弹协议**：路径 B（薄 hook）· 仅 prefill `gemm_m≥M*` 分支（阵列唯一物理意义 regime·M* 参照 vendor row_align=4·建议 M≥4）· correctness-first：
- **ZERO-MODEL int32-exact 门**（复用 seal harness）在任何 e2e 前 · 4 布局桥（scale fold / shape tiling / 激活 pack / 权重 repack）**各**board UT vs 独立 oracle byte-match（build 前 de-risk·MIRAGE trap 防线）。
- **greedy A==B vs stock RVV forward**（ON=tcrv IME / OFF=stock）+ logits sanity·correctness RED ⇒ perf VOID。
- objdump `vmadot` engage seal（ON 含 `vmadot`·OFF=0）+ banner FIRES + A-tree restore byte-exact + 测后 md5 零 stock 改动。
- **措辞预注册（诚实·非绿）**：结果 framing = **N2 结构闭环 / 机制**（`[GAP-IME-E2E-INTEGRATION]` closed·C1 跨范式 extensibility）·prefill 数**绑 vendor-ceiling 同域披露 + 对手身份 + kernel-family-vs-array 分解**·**预注册 expecting parity/非 perf-covered 绿格**（micro↛e2e·`[NG-4]`）。decode 预注册 parity/LOSS（内存墙）。

### 5.3 correctness-first 铁律套用（破损上游/MIRAGE 类比）

IME 桥无"破损上游"（无上游 tcrv IME kernel）·但**4 布局桥是等价 MIRAGE 源**（layout 失配 = 假绿 garbage·类比 RVV vl=16 死宽度）。铁律：
1. 4 布局桥 board UT vs 独立 oracle byte-match·build 前 de-risk。
2. ZERO-MODEL int32-exact（seal harness）作 MAC 层门·greedy A==B 作全链门。
3. per-block scale fold 是**最易错点**（int32-exact 不覆盖 scale·seal harness 全 K 单 scale·真 forward 逐 block）·须**独立 oracle 折叠对照**。
4. correctness 门绿前 perf 一律不入台账·任何 e2e 数绑 micro↛e2e + vendor-ceiling 同域披露。

---

## ★一句话结论

**IME e2e 桥 = 最重 L-接线③**（4 处 correctness-critical 布局桥 + int32→f32 per-block scale-fold epilogue + 固定微 tile 的 shape driver + 无"翻 gate"白嫖 + 换板 k1·**多 session ≥3**）；**N2 结构价值真实**（闭 `[GAP-IME-E2E-INTEGRATION]` = C1 模板协议跨范式 extensibility 铁证·矩阵范式抵达真 forward）**但 perf payoff 诚实近零**（vendor 全接线 IME GEMM 同域 toggle 已证本 memory-bound 1B 模型无干净 IME-unit e2e 赢·decode floor 1.46× = kernel-family swap 非阵列·tcrv 2.09× compute-account 不传导·micro↛e2e）；**建议 = perf 名义不值当下做**（vendor toggle 已是 ceiling proxy·欠账已诚实捕获），若为 C1 跨范式结构完整性做则**曳光弹首格 = q4_0@ime**（板已有 q4_0 模型零 provisioning·最简单 scale fold·seal-proven tile·仅 prefill M≥M* 分支·correctness-first ZERO-MODEL + greedy A==B vs stock RVV·预注册 expecting parity/非绿·`[NG-4]` 全程）。

## durable files
- `deploy-readiness.md`
