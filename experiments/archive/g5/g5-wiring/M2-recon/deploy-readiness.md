# G5-M2 铺线 — 部署就绪度盘点（deploy-readiness inventory）

> **campaign**: G5 接线战役 · **M2-recon = 部署就绪地图**（M2 fan-out 排序依据）
> **role**: 纯本仓 read-only 静态分析 · 零 board / 零 build / 零 git（HEAD `92c27389` 未变）
> **上游 approach 锁（M1 发现）**: 接线 ≠ routing-freebie 白嫖。**通用接线 = 部署我方 emitted kernel（correctness-carrier）**——上游 VLEN128 repack kernel 多数硬编码 vl=16、在 VLEN128 破损（q8_0 已确证 `[GAP-Q8_0-VLEN128-KERNEL]`）；翻 gate 只 route 到破损上游 body → e2e garbage。q4_0 之所以在 VLEN128 正确，是我方 emitted vl=8 kernel **拦截**在破损上游 vl=16 body 之前。**M2 每格 = 部署 emitted（挂点③），非翻 gate 白嫖。**
> **来源（全只读）**: `M0-接线机制解剖.md`（两挂点+通用方案）· `M1-q8_0/MANIFEST.md`（[GAP-Q8_0-VLEN128-KERNEL] 根因）· `docs/ROADMAP.md`（perf-covered 定义/T6 关键发现）· `schema/coverage-sixstate.v1.json`（93 roster·84 certified·3 IME certified·7 declared-exception·2 out-of-domain）· 本仓 emit 路径 grep（`test/Conversion/RVV`、`test/Target/RVV`、`test/Conversion/EmitC/ime-*`、`lib/**/emitRepackGem*` / `IME*ToEmitCFunc`、`tools/e2e-harness/board/*`）。
> **★行号/挂点 provenance 锁**: dispatch-gate（GEN `repack.cpp`）+ kernel 挂点（`arch/riscv/repack.cpp`）行号**全部来自板 A-tree `ssh rvv:/home/ubuntu/tcrv-llamacpp/ggml` HEAD `f3e1828`（M0/M1 已记）**，**不在本仓**。凡上游 VLEN128 kernel 破损/存在状态本仓查不到者，一律标 **"待 board 核"**（禁 ssh，M2 落地取）。

---

## 就绪度分级定义（本战役口径）

| 级 | 判据（本 repack-wiring 战役） |
|----|------|
| **🟢 就绪** | 我方 emitted repack kernel 已构造 **∧** 上游 repack scaffold（trait 注册 + `get_tensor_traits` case + `arch/riscv` gemv/gemm body）已存在 **∧** 三挂点行号明确 → 部署 = 翻 case128 gate + `#include` emitted `.inc` 拦截（**M1b 模板可直接套**）。 |
| **🟡 半就绪** | 我方 emitted repack kernel 已构造，但缺一环：① 上游 repack scaffold 全无、需净新 trait+arch skeleton（无硬阻塞）；**或** ② 整链已建但缺 board provisioning（gguf/quantize）需 redeploy。 |
| **🔴 未就绪** | repack kernel 未构造（gemm_tile `absent`/declared-exception）**或** 缺整条上游链路 + 硬阻塞（ABI / q8_K 激活胶水 / 换板无桥）。 |

**correctness-carrier 列**：标注该格上游 VLEN128 kernel 是否**已知破损**（→ 部署 emitted 是正确性必要条件，非仅 perf）。本仓查不到者标"待 board 核"。

---

## 一、FLAT 类（typed-flat weight 格 · 直类比 q4_0）

| 格 | 已构造 kernel? | emit 路径（.inc/C 怎么产 · 现成 lit/脚本） | dispatch-gate 挂点（GEN `repack.cpp`·板 A-tree） | kernel 挂点（`arch/riscv/repack.cpp`·板 A-tree） | correctness-carrier（上游 VLEN128 破损?） | 就绪 |
|---|---|---|---|---|---|---|
| **q4_0** | ✅ GEVM+GEMM（**dedicated**：`emitRepackGemvQ4_0Q8_0` / `emitRepackGemmQ4_0Q8_0` + `TypedRepackGem{v,m}LoopBody`） | **现成·deployable**：`test/Target/RVV/q4-0-q8-0-repack-{gemv,gemm}-full-pipeline-export-e2e.mlir`（含 `--tcrv-materialize-emission-plans` prod 变体 + `--tcrv-rvv-lower-to-emitc` core diff）；recipe = `tcrv-opt --tcrv-rvv-lower-quant-contraction=march=… --tcrv-rvv-lower-to-emitc` → C | **:4592** `case 128 {…return &q4_0_16x1_q8_0;}` `TCRV-WINB-ON-TOGGLE`（区 :4589–4597） | `ggml_gemv/gemm_q4_0_16x1_q8_0` body :234–244（VLEN128 分支 + banner + emitted call + `return`） | **是**（上游 vl=16 body 同破损 · 我方 emitted vl=8 拦截 = VLEN128 correctness-carrier · M1 canon 精化） | **🟢 WIRED**（唯一成熟先例 · deployed `.so` md5 `75f20b5f` 仅含 q4_0 tcrv 符号） |
| **q8_0** | ✅ GEVM（**dedicated** `emitRepackGemvQ8_0Q8_0` · `RVVOps.td:5010` `GgmlRepackGemvQ80Q80Op`）+ GEMM（**generic** quant-contraction `TypedRepackGemmLoopBody`） | **现成**：GEVM `rvv-emit-identity-quant-contraction-q8-0-repack-vlen128.mlir`；GEMM `rvv-emit-quant-contraction-q8-0-repack-gemm-prefill-vlen128.mlir`；**已 emit 落地** `M1b-q8_0/q8_0_gevm_raw.c` + `q8_0_gemm_raw.c`（vl=8 raw C 两核齐）；**deploy 脚本已写** `tools/e2e-harness/board/g5-m1-q8/deploy_patch_q8_emitted.py`（改 2 tracked 文件 + `#include tcrv_emitted_q8_0.inc`） | **:4713** `case 128 {break;}//TODO`→flip；trait 注册 :4569；case256 route :4714 | `ggml_gemv_q8_0_16x1_q8_0` :518（marker :535）/ `ggml_gemm_q8_0_16x1_q8_0` :1426（marker :1444） | **是·已确证** `[GAP-Q8_0-VLEN128-KERNEL]`（上游每 intrinsic 硬编码 AVL=16→VLEN128 钳 8→半列 garbage · M1 source-confirmed）→ **correctness-carrier 必要** | **🟢**（M1b pilot 在飞 · **M2 模板源**·套 M1b 即可） |
| **q4_1** | ✅ GEVM+GEMM（dedicated `GgmlRepackGem{v,m}Q41Q81Op` · `emitRepackGem{v,m}Q4_1Q8_1`） | 现成 emit：`rvv-emit-identity-quant-contraction-q4-1-repack-vlen128.mlir` + `rvv-emit-quant-contraction-q4-1-repack-gemm-prefill-vlen128.mlir` | **零 riscv repack 分支（任何 VLEN）**→需净新 trait+case | **零 arch 函数**→需净新 skeleton | 待 board 核（上游无 kernel = 无"破损 body"，正确性全落我方 emitted） | **🔴**（上游 scaffold 全无 **+ 硬阻塞**：上游 q8_1×4 量化器 + int-zero-point ABI，repack-terrain 记 BLOCKED） |
| **q5_0** | ✅ GEVM（dedicated `GgmlRepackGemvQ50Q80Op` · `emitRepackGemvQ5_0Q8_0`）+ GEMM（generic quant-contraction） | 现成 emit：`rvv-emit-identity-quant-contraction-q5-0-repack-vlen128.mlir` + `rvv-emit-quant-contraction-q5-0-repack-gemm-prefill-vlen128.mlir`；micro 佐证 `tools/e2e-harness/g3-lode-flat-q50/` | 零 riscv repack 分支→需净新 trait+case | 零 arch 函数→需净新 skeleton | 待 board 核（无上游 kernel） | **🟡**（emitted kernel 就绪 · q8_0 激活可复用 · 无 ABI 阻塞 · 缺上游 scaffold = 净新 plumbing） |
| **q5_1** | ✅ GEVM（dedicated `GgmlRepackGemvQ51Q81Op` · `emitRepackGemvQ5_1Q8_1`）+ GEMM（generic） | 现成 emit：`rvv-emit-identity-quant-contraction-q5-1-repack-vlen128.mlir` + `rvv-emit-quant-contraction-q5-1-repack-gemm-prefill-vlen128.mlir`；micro `g3-lode-flat-q51/` | 零 riscv repack 分支→需净新 | 零 arch 函数→需净新 | 待 board 核 | **🟡**（同 q5_0 · q8_1 激活 · 缺上游 scaffold） |
| **q1_0** | ❌ gemm_tile `absent`（declared-exception）· 仅 block-dot + dequant（dequant byte-exact 已跑） | 仅 block-dot：`test/Target/RVV/q1-0-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir`；**无 repack emitter** | 无 | 无 | N/A | **🔴**（custom 格 · 无 repack kernel） |
| **tq1_0** | ✅ GEVM+GEMM（`emitRepackGem{v,m}TQ10Q8K`·q8_K 激活·ternary） | 现成 emit：`rvv-to-emitc-repack-{gemv,gemm}-tq1-0-q8-K.mlir` + `rvv-emit-quant-contraction-tq1-0-repack-gemm-prefill-vlen128.mlir` + `test/Target/RVV/tq1-0-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir` | 待 board 核（上游 ternary repack 很可能缺） | 待 board 核 | 待 board 核 | **🔴**（无上游链路 · q8_K 激活胶水 · perf 现状 LOSS · 殿后） |
| **tq2_0** | ✅ GEVM+GEMM（`emitRepackGem{v,m}TQ20Q8K`） | 现成 emit：`rvv-to-emitc-repack-{gemv,gemm}-tq2-0-q8-K.mlir` + `…prefill-vlen128` + `tq2-0-…full-pipeline-export-e2e.mlir` | 待 board 核 | 待 board 核 | 待 board 核 | **🔴**（同 tq1_0） |

---

## 二、K-quant 类（q2_K..q6_K · super-block · 缺整条链路 `[GAP-KQUANT-E2E-INTEGRATION]`）

> **通则（M0 乙.3）**：K-quant VLEN128 全 `case128 {break;}//TODO`（q4_K :4619 / q2_K :4636），q5_K/q6_K/q3_K **连 riscv repack 分支都无**。repack 布局构造器、q8_K 激活量化、kernel body、dispatch 接线**全缺**，须编译器从头全构造（≠ q4_0 免费 5×）。板另**无 q4_K gguf、无 `llama-quantize`** → e2e 不可跑。**我方 emit 侧 5 超块全齐**（下表），瓶颈 = 上游链路 + board provisioning，非我方 kernel。

| 格 | 已构造 kernel? | emit 路径 | dispatch-gate（GEN`repack.cpp`·板 A-tree） | kernel 挂点（`arch/riscv`·板 A-tree） | correctness-carrier | 就绪 |
|---|---|---|---|---|---|---|
| **q4_K** | ✅ GEVM+GEMM（**dedicated** `emitRepackGemvQ4KQ8K` / `emitRepackGemmQ4KQ8K` + lane-wise integer core） | 现成：`rvv-to-emitc-repack-{gemv,gemm}-q4-K-q8-K.mlir` + `rvv-emit-quant-contraction-q4-K-repack-gemm-prefill-vlen128.mlir` + `q4-k-…-{front-door,full-pipeline}-export-e2e.mlir` + `rvv-to-emitc-q4-k-min-term.mlir`；**★deploy harness 已存** `tools/e2e-harness/board/t4b-seal-fix/{deploy_patch.py,board_deploy.sh}` | **:4619** `case128 {break;}//TODO` | 待 board 核（seal-fix 建过整链 route+repack+q8_K+dispatch） | 待 board 核（seal-fix 教训：部署变体须 == 证过变体 vl=8·误挂 vl=16→PPL 822057 garbage） | **🟡**（**K-quant 最就绪**：整链已建 + 证正确 PPL 11.97 vs stock 12.05[seal-fix `98717158`] + deploy harness 已写；缺 = board gguf/quantize provisioning + redeploy · 非 live-wired） |
| **q2_K** | ✅ GEVM+GEMM（`emitRepackGem{v,m}Q2KQ8K`） | 现成：`rvv-to-emitc-repack-{gemv,gemm}-q2-K-q8-K.mlir` + `…prefill-vlen128` + `q2-k-…full-pipeline-export-e2e.mlir`；dispatch probe `tools/e2e-harness/board/kquant_dispatch_probe_q6q2q3.c` | **:4636** `case128 {break;}//TODO` | 待 board 核 | 待 board 核 | **🔴**（缺整条链路 + q8_K 激活胶水 + board provisioning） |
| **q3_K** | ✅ GEVM+GEMM（`emitRepackGem{v,m}Q3KQ8K`） | 现成：`rvv-to-emitc-repack-{gemv,gemm}-q3-K-q8-K.mlir` + `…prefill-vlen128` + full-pipeline-export-e2e | 零 riscv repack 分支 | 无 | 待 board 核 | **🔴**（缺整条链路 · q3_K 最难 super-block） |
| **q5_K** | ✅ GEVM+GEMM（`emitRepackGem{v,m}Q5KQ8K`） | 现成：`rvv-to-emitc-repack-{gemv,gemm}-q5-K-q8-K.mlir` + `…prefill-vlen128` + full-pipeline-export-e2e | 零 riscv repack 分支 | 无 | 待 board 核 | **🔴**（缺整条链路） |
| **q6_K** | ✅ GEVM+GEMM（`emitRepackGem{v,m}Q6KQ8K`） | 现成：`rvv-to-emitc-repack-{gemv,gemm}-q6-K-q8-K.mlir` + `…prefill-vlen128` + full-pipeline-export-e2e | 零 riscv repack 分支 | 无 | 待 board 核 | **🔴**（缺整条链路） |

---

## 三、IQ / codebook 类（全 🔴 · 无上游链路 · perf 现状全 LOSS · 排最后）

> **通则**：IQ/codebook 格上游 riscv repack scaffold 一律缺；board provisioning 缺；perf-covered 现状全 LOSS（含 `[GAP-CLANG-GATHER-TRAP]` iq4_nl vluxei 索引-gather 陷阱）。部分格 gemm_tile `absent`（declared-exception）= 连 repack kernel 都未构造。**M2 排最后，且非 perf-covered 直接杠杆。**

| 格 | 已构造 kernel? | emit 路径 | dispatch/kernel 挂点 | correctness-carrier | 就绪 |
|---|---|---|---|---|---|
| iq2_xxs | ✅ GEVM+GEMM（`emitRepackGem{v,m}Iq2XxsQ8K`） | `rvv-to-emitc-repack-{gemv,gemm}-iq2-xxs-q8-K.mlir` + `…prefill-vlen128` + `iq2-xxs-…full-pipeline-export-e2e.mlir` | 上游全缺·待 board 核 | 待 board 核 | **🔴**（无上游链路·LOSS） |
| iq2_xs | ✅ GEVM+GEMM（`emitRepackGem{v,m}Iq2…`） | `rvv-to-emitc-repack-{gemv,gemm}-iq2-xs-q8-K.mlir` + `…prefill-vlen128` + full-pipeline-export-e2e | 上游全缺 | 待 board 核 | **🔴** |
| iq2_s | ✅ GEVM+GEMM（`emitRepackGem{v,m}Iq2DualScaleQ8K`） | `rvv-to-emitc-repack-{gemv,gemm}-iq2-s-q8-K.mlir` + `rvv-emit-{identity,}quant-contraction-iq2-s-repack-…` + full-pipeline-export-e2e | 上游全缺 | 待 board 核 | **🔴** |
| iq4_nl | ✅ GEVM+GEMM（`emitRepackGem{v,m}Iq4NlQ80`·q8_0 激活） | `rvv-to-emitc-repack-{gemv,gemm}-iq4-nl-q8-0.mlir` + `…prefill-vlen128` + `iq4-nl-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir` | 上游全缺 | 待 board 核 | **🔴**（+ `[GAP-CLANG-GATHER-TRAP]` clang-ours gather 拖后腿） |
| iq4_xs | ✅ GEVM+GEMM（`emitRepackGem{v,m}Iq4XsQ8K`） | `rvv-to-emitc-repack-{gemv,gemm}-iq4-xs-q8-K.mlir` + `…prefill-vlen128` + full-pipeline-export-e2e | 上游全缺 | 待 board 核 | **🔴** |
| mxfp4 | ✅ GEVM+GEMM（`emitRepackGem{v,m}Mxfp4Q8`）· 但 vec_dot mxfp4 = constructed-weak（declared-exception） | `rvv-to-emitc-repack-{gemv,gemm}-mxfp4-q8-0.mlir` + `…prefill-vlen128` + `mxfp4-…full-pipeline-export-e2e.mlir` | 上游全缺 | 待 board 核 | **🔴** |
| iq1_s | ❌ gemm_tile `absent`（declared-exception）· 仅 block-dot+dequant | `iq1-s-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir`（无 repack emitter） | 无 | N/A | **🔴**（repack kernel 未构造） |
| iq1_m | ❌ gemm_tile `absent`（declared-exception） | block-dot only | 无 | N/A | **🔴**（repack kernel 未构造） |
| iq3_xxs | ❌ gemm_tile `absent`（declared-exception） | block-dot only | 无 | N/A | **🔴** |
| iq3_s | ❌ gemm_tile `absent`（declared-exception） | block-dot only | 无 | N/A | **🔴** |
| nvfp4 | ❌ gemm_tile `absent`（declared-exception）· 仅 flat block-dot | `nvfp4-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir` | 无 | N/A | **🔴** |

---

## 四、IME 类（k1 板 · 矩阵范式 · 换板 + 无桥 = 全 🔴 for e2e · 但 kernel 已 silicon-sealed）

> **通则（M0 乙.2 + ROADMAP T6）**：**tcrv IME GEMM 完全未接进任何 llama forward**（板 `~/tcrv-k1-llama/ggml/src` grep tcrv IME 符号 = 0 hit）= 具名 `[GAP-IME-E2E-INTEGRATION]`。挂点**与 FLAT/K-quant 根本不同**：(1) **换板** = k1（SpacemiT X60），非 rvv；(2) 挂点 = ggml `mul_mat` 的 **GEMM/prefill 分支（M≥M*）**，**非** `arch/riscv/repack.cpp` 的 gemv/gemm；(3) 需建 **tcrv-IME↔ggml forward 桥**（block-layout 已知·桥不存在）。(4) **micro↛e2e 封顶**：decode(M=1) 阵列跑不起来 = 1.47× floor（kernel-family swap 非阵列）·IME-unit 增量与 GEMM-vs-GEVM 混淆不可干净隔离 → **即便接线大概率不 e2e 传导**。**IME 接线价值 = 机制/N2 结构闭环，非 perf 绿格**（[NG-4] 全程）。

| 格 | 已构造 kernel? | emit 路径 | dispatch/kernel 挂点（k1·与 rvv 不同） | correctness-carrier | 就绪 |
|---|---|---|---|---|---|
| q4_0@ime | ✅ **silicon-sealed**（`tcrv.ime.q4_0_matmul_tile`·`IMEBackendEmissionDriver.cpp:1116 IMEQ40MatMulTileToEmitCFunc`·q4_0_dequant_core + vmadot_mac_leaf） | `test/Conversion/EmitC/ime-q4-0-matmul-tile-materialization.mlir`；seal `test/Target/IME/q4-0-matmul-tile-int32-k1seal.c` + oracle `…-oracle.c`；recipe = `tcrv-opt --tcrv-materialize-plugin-variants --tcrv-select-variants --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emitc-lowerable-routes` | **无 repack.cpp 挂点** · 需 k1 `mul_mat` GEMM/prefill 分支新 hook + tcrv-IME↔ggml 桥（不存在） | 我方 emitted 承载正确性（k1 硅 `vmadot 0xe210312b` int32 0-diff·ZERO-MODEL）·上游无对应 kernel | **🔴**（e2e 换板+无桥+micro↛e2e 封顶·单列独立批） |
| q8_0@ime | ✅ silicon-sealed（`IMEBackendEmissionDriver.cpp:1237 IMEQ80MatMulTileToEmitCFunc`·q8_0_dequant_core=direct int8·vmadot leaf 复用 q4_0） | `test/Conversion/EmitC/ime-q8-0-matmul-tile-materialization.mlir` + `q8-0-matmul-tile-int32-{k1seal,oracle}.c` | 同上（k1·无桥） | 同上 | **🔴**（同 q4_0@ime） |
| q4_K@ime | ✅ silicon-sealed（`IMEBackendEmissionDriver.cpp:1358 IMEQ4KMatMulTileToEmitCFunc`·SIX bricks·K-quant 两级 fold） | `test/Conversion/EmitC/ime-q4-K-matmul-tile-materialization.mlir` + `q4-K-matmul-tile-int32-{k1seal,oracle}.c` | 同上（k1·无桥） | 同上 | **🔴**（同 q4_0@ime） |

---

## 五、84-cell roster 覆盖对账（wiring 相关性映射）

repack-wiring 战役的**唯一 perf-covered 杠杆 = weight `mul_mat` 路径**（`gemm_tile` repack + IME + `vec_dot` block-dot fallback）。84 certified 格按接线相关性分：

| roster 分区 | 格数 | 与 repack-wiring 关系 |
|---|---|---|
| `gemm_tile` repack（rvv） | 18 constructed（q4_0×2,q4_1,q5_0,q5_1,q8_0,q2_K,q3_K,q4_K,q5_K,q6_K,iq2_xxs,iq2_xs,iq2_s,iq4_nl,iq4_xs,mxfp4,tq1_0,tq2_0） | **★上表主体**（部署目标本体） |
| `gemm_tile` IME（k1） | 3 certified（q4_0/q8_0/q4_K@ime） | **★四节**（换板·独立批） |
| `gemm_tile` absent（declared-exception） | 6（q1_0,iq1_s,iq1_m,iq3_xxs,iq3_s,nvfp4） | repack kernel 未构造 → 🔴（表内已列） |
| `vec_dot`（24 格·23 certified） | 24 | block-dot **fallback** 路径（get_tensor_traits 返 nullptr 时）·非 repack 挂点·wiring 时作正确性参照/micro，非部署本体 |
| `quantize_row`（q8_0/q8_1/q8_K） | 3 | **激活量化子件**（repack 输入胶水·q8_0/q8_1/q8_K 分别喂 FLAT/q4_1系/K-quant repack） |
| `dequantize_row`（24 格） | 24 | 权重 dequant 子件（block-dot 路径消费）·非 mul_mat 部署本体 |
| `product_reduce`（3） | 3 | foundation 原语（block-dot 分解体）·非部署格 |
| elementwise forward f32（rms_norm/softmax/rope/silu/scale/gelu/add/mul/cpy） | 9 | **forward 算子但非 mul_mat repack 挂点**（融合 epilogue/独立 hook·本 repack-wiring recon 域外·另立接线面） |
| out-of-domain | 2（flash_attn/tile,bf16/all） | [G-2] 域外·非量化 kernel |

> **诚实注**：本表"已构造 kernel"列对 rvv repack 18 格 = ✅（emit 侧全齐，emitter/lit 齐备）。**就绪度瓶颈几乎全在上游 scaffold + board provisioning + 换板桥，不在我方 emit 能力。**这与 ROADMAP「T6 关键发现」一致：perf-covered 真瓶颈 = 集成/接线（多数 tcrv 核未 wired 进 forward）+ micro↛e2e 封顶，**非 kernel 质量**。

---

## 六、M2 fan-out 批次建议（据就绪度排序）

| 批 | 格 | 理由 |
|---|---|---|
| **批 0（已完成/在飞）** | q4_0（WIRED 先例）· **q8_0（🟢 M1b 在飞·correctness-carrier pilot）** | q8_0 = 上游唯一与 q4_0 同结构（trait+arch+case256 齐·仅 case128 gate off+破损）·M1b 部署 emitted vl=8 拦截破损上游·**M2 模板源** |
| **批 1（🟡 · 次批·净新 plumbing）** | q5_0 → q5_1 | emitted GEVM 就绪 · q8_0/q8_1 激活可复用 · 无 ABI 阻塞 · 仅缺上游 trait+arch skeleton（机械净新，非 freebie）·套 M1b 挂点③模板 + 补 trait 注册 |
| **批 2（🟡 · K-quant 单格·最就绪者优先）** | q4_K | 整链已建 + 证正确（seal-fix PPL 11.97 vs 12.05）+ deploy harness 已写；缺 = board gguf/quantize provisioning + redeploy vl=8 变体 |
| **批 3（🔴 · K-quant 余族·缺整链）** | q2_K → q6_K → q5_K → q3_K | emit 侧齐但上游链路 + q8_K 激活胶水 + provisioning 全缺 = 净新全构造（`[GAP-KQUANT-E2E-INTEGRATION]`）·q3_K 最难殿后 |
| **批 4（🔴 · IME·独立换板批）** | q4_0/q8_0/q4_K@ime | kernel 已 silicon-sealed 但换板(k1)+无 forward 桥+矩阵挂点(非 repack)+micro↛e2e 封顶 → 名义 = N2 机制闭环非 perf 绿格·**与 rvv 批解耦并行** |
| **批 5（🔴 · IQ/codebook·殿后·非杠杆）** | iq2_xxs/iq2_xs/iq2_s/iq4_nl/iq4_xs/mxfp4（repack 已构造）· q4_1（ABI 阻塞）· q1_0/iq1_s/iq1_m/iq3_xxs/iq3_s/nvfp4（gemm_tile absent） | 无上游链路 + perf 现状全 LOSS + 部分 repack kernel 未构造 · q4_1 另有 int-zero-point ABI 硬阻塞 · **非 perf-covered 直接杠杆** |

---

## ★一句话结论

**M2 fan-out 首批 🟢 clean = {q8_0}（1 格 · q4_0 已 WIRED）**——q8_0 是上游唯一与 q4_0 同 repack scaffold（trait `:4569`+arch gemv`:518`/gemm`:1426`+case256 已 route）、仅 VLEN128 gate off 且上游 body 破损（`[GAP-Q8_0-VLEN128-KERNEL]`）的格，M1b 正部署我方 emitted vl=8 correctness-carrier；其 `deploy_patch_q8_emitted.py` 就是 M2 可克隆的模板。次批 🟡 = {q5_0, q5_1}（emitted kernel 就绪、仅缺上游 trait+arch 净新 plumbing）与 {q4_K}（整链+deploy harness 已建、缺 board provisioning）。**排序理由 = 上游 repack scaffold 完整度 → 净新 plumbing 量 → board provisioning → 换板/无桥**；🔴 大多数格的瓶颈在上游链路/换板桥而非我方 emit 能力（emit 侧 rvv repack 18 格全齐）。

## durable files
- `deploy-readiness.md`
