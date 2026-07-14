# G5-M1b — q8_0 EMITTED vl=8 kernel deploy · correctness-carrier · CASEFILE

> **campaign**: G5 接线战役 · **M1b = 部署我方 emitted q8_0 kernel**（拦截在破损上游 vl=16 body 之前，承载 correctness+perf）
> **board**: `ssh rvv` = openEuler 24.03 / VLEN128 / gcc-15.2.0 / 64c / governor=performance @2.6GHz
> **A-tree**: `/home/ubuntu/tcrv-llamacpp/ggml` HEAD `f3e1828`（WinB-q4_0-ON working-tree baseline）
> **HEAD (TianChen-RV)**: `92c27389` **未变** · 禁 git · A-tree 可逆（deploy 备份 + 测后 restore byte-exact）
> **结论**: **★PERF-COVERED（correctness-carrier）** — correctness GREEN（我方 emitted vl=8 kernel 正确，非 M1 上游 garbage），**prefill 4.350× / decode 3.812× 双相 DIFFERENCE**（CI 排除 1.0）。预注册 **R1** 命中。

---

## 一、判读落点 = R1（prefill ≥ DIFFERENCE ∧ correctness GREEN）· 成色比 q4_0 强

| 预注册分支 | 命中 | 说明 |
|---|---|---|
| **R1 prefill 传导→转绿** | **✓** | correctness GREEN + prefill 4.350× DIFFERENCE（CI[4.343,4.355]）。decode 亦 3.812× DIFFERENCE（超出 R2 parity 预期）。 |
| R2 decode PARITY | ✗（更强） | M0 张力 B 预测 decode 内存墙 parity；实测 **DIFFERENCE 3.812×** — repack streaming/locality 主导，非带宽墙。 |
| R3 已路由仍 parity | ✗ | 双相皆 DIFFERENCE。 |
| R4 correctness RED（我方 kernel 也错）| ✗ | 我方 emitted vl=8 kernel **正确**（3/3 A==B coherent），区别于 M1 上游 vl=16 garbage。 |

**★成色（比 q4_0 强，实证）**：q4_0 的 routing-freebie 中，上游 VLEN128 已有可用 kernel，我方 emitted kernel 与之字节等价（correctness 白送）。**q8_0 上游 VLEN128 repack kernel 破损**（`[GAP-Q8_0-VLEN128-KERNEL]`，hardcoded vl=16→钳 8→半列 garbage）→ **routing-freebie 对 q8_0 在正确性上根本不成立，除非我方 emitted vl=8 kernel 承载**。M1b 实证：我方 emitted kernel = VLEN128 correctness-carrier（M1 banner-only fall-through = "olta" garbage RED；M1b emitted intercept = "Paris." coherent GREEN）。

---

## 二、接线落地（我方 emitted kernel 部署 · 3 挂点 · 可逆）

- **host emit**（HEAD 92c27389，LLVM 20.1.8）：
  `tcrv-opt <mlir> --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`（VERBATIM）。
  - GEVM（decode）源 = `test/Conversion/RVV/rvv-emit-identity-quant-contraction-q8-0-repack-vlen128.mlir` → `tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_...`
  - GEMM（prefill）源 = `test/Conversion/RVV/rvv-emit-quant-contraction-q8-0-repack-gemm-prefill-vlen128.mlir` → `tcrv_emitc_ggml_gemm_q8_0_q8_0_kernel_...`
  - 两者经**同一 q4_0 front door 构造**（typed_repack_gem{v,m}_loop_body，fullI8 core，d-only fold）。GEMM = **NET-NEW oracle-validated**（无 q8_0 GEMM direct emitter；M0 "GEVM only" 已过时）。
  - 合并 `.inc` md5 `b5177a4d…`；**54 个 AVL 常量全 = 8**（vl=8 correctness-carrier，零 vl=16）。
- **挂点① dispatch gate（GEN `repack.cpp`）**：q8_0 `case128 {break;}//TODO` → `return &q8_0_16x1_q8_0`（route 到上游 repack trait → 触发 block_q8_0x16 权重 repack + 调 arch gemv/gemm）。
- **挂点②③ ARCH `arch/riscv/repack.cpp`**：`#include "tcrv_emitted_q8_0.inc"`（与 q4_0 .inc 并列）+ 在 `ggml_gemv/gemm_q8_0_16x1_q8_0` 体内 `UNUSED(blocklen);` 后插 VLEN128 分支：banner + **调我方 emitted kernel + return**（拦截在破损上游 vl=16 body 之前）。
  - GEVM ABI：`(n, s, nc, vx, -, vy, -, -)`（v3=nc 列数；weight block_q8_0x16 stride544；act plain q8_0 stride34）。
  - GEMM ABI：`(nr, bs, n, s, nc, vx, vy)`（7-role；weight stride544；act block_q8_0x4 stride136）。

**engage 验证（objdump/banner seal）**：
- nm ON `.so` 含 `tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel` + `tcrv_emitc_ggml_gemm_q8_0_q8_0_kernel`（各 1）；OFF `.so` = 0。
- 真实 forward：ON → **EMITTED GEVM banner ×8 + EMITTED GEMM banner ×8 FIRE**；OFF → **0**（engage_ON/OFF.err）。
- **objdump vl-seal**：两符号皆 `vsetivli zero,8,e32,m2`（vl=8），**零 vl=16** → 部署变体 == 证过变体（objdump_gevm/gemm_seal.txt）。

---

## 三、e2e 分相数据 —— DIFFERENCE（correctness GREEN 前置已过）

同树物理 `.so` 交换 A/B（一份源码、一个 `llama-bench`、gcc-15.2.0、唯一差异 = q8_0 gate + emitted intercept）。n=20/side/phase（2 pass×`-r 10`），DVFS 锁 2.6GHz，per-pass 无漂移。

| 相 | ours(emitted vl=8) | stock(block-dot) | 比 | bootstrap95%CI | IQR% | 判 |
|---|---|---|---|---|---|---|
| **prefill pp128** | 39.191 t/s | 9.011 t/s | **4.350×** | [4.343, 4.355] | 0.10 / 0.22 | **DIFFERENCE** |
| **decode tg32** | 8.473 t/s | 2.223 t/s | **3.812×** | [3.801, 3.818] | 0.39 / 0.39 | **DIFFERENCE** |

**★mirage 对照（M1↔M1b 的锚点）**：M1（banner-only，破损上游 vl=16，跳半列做少工）prefill 78.11× MIRAGE。M1b（正确 vl=8，做全工）prefill 39.191 ≈ **78.11 的一半** → 数值上确证 mirage "快" = 跳过半数列的做少工，我方 kernel 做**全量正确工**。decode：M1 mirage 7.55（3.39×）；M1b **正确 8.473（3.812×）**，正确路径反而**更高**。

---

## 四、★框架锁（措辞宪法 · 主会话裁决项）

- **perf 机制 = 能力键控 repack 路由**（block_q8_0x16 权重布局 + q8_0 激活 + tiled GEVM/GEMM）vs stock block-dot vec_dot。**非 kernel-质量-vs-手调赢**（同 q4_0 canon `q4-0-e2e-is-routing-not-kernel` 措辞锁；q8_0 无"可对比的可用上游 repack kernel"——上游破损）。
- **correctness-carrier = 我方 emitted vl=8 kernel**（**成色比 q4_0 强**：q4_0 correctness 白送于上游已有可用 kernel；q8_0 上游 VLEN128 破损 → 无我方 kernel 则 routing = e2e garbage）。→ 对 canon `q4-0-e2e-is-routing-not-kernel` 的**实质精化**：VLEN128 上 **emitted kernel 承担正确性**（routing 或主导 perf 比值，但 correctness 由 emitted kernel 承载）；q8_0 是该精化的**决定性正例**（M1 garbage ↔ M1b coherent）。**留主会话裁决是否 codify 进 memory/canon。**
- **张力 A（selector decline）**：我方 selector 现判 q8_0 `block-dot-decline-q8_0-lean-fallback`（`RVVContractionPathSelection.cpp:111`，`!blockDotComputeHeavy`）。**M1b = 强制路由传导探针**（front-door 构造用 `block_dot_compute_heavy=true` 强制 repack；gate 强制路由）。**未修 selector**（canon 级语义变更 = 必问主会话）。本结果为 selector 语义提供**张力数据**：q8_0 repack 在 e2e **确有传导（4.35×/3.81×）**，与 selector 的 lean-decline 判断**冲突** → 是否 revisit `blockDotComputeHeavy` for q8_0（或加 M-regime/prefill 键）= **主会话裁决项**（不在本任务触碰集）。
- **张力 B（memory 墙）**：M0 预测 decode parity（q8_0=2× 字节）。实测 **DIFFERENCE 3.812×** → repack 16-列 streaming/locality 主导，非原始带宽墙（收窄 M0 张力 B 的适用边界）。

---

## 五、八门状态（perf-covered）

| 门 | 状态 | 依据 |
|---|---|---|
| ① byte-exact | ✓（bounded-ULP + greedy A==B 生成文本字节相同）| 不说"byte-exact vs ggml"；correctness_GREEN.txt |
| ② 对手对称 | ✓ gcc-15.2.0 双侧（同树 .so swap）| — |
| ③ 双账本 | ✓ kernel-sym（gcc 双）+ system e2e | transmission_accounting.csv |
| ④ 对手身份探针 | ✓ ggml block-dot vec_dot（RVV，objdump 证 vle8/vwmul）| g5b_stock_vecdot |
| ⑤ micro∧e2e | ✓ micro 4.10× ∧ e2e prefill 4.35×/decode 3.81× | — |
| ⑥ selector-routing | ✓ banner FIRES（GEVM×8 + GEMM×8）| engage_ON.err |
| ⑦ 纪律 | ✓ A-tree 可逆·restore byte-exact·零 stock 永久改动·HEAD 未变·禁 git | 六节 |
| ⑧ 措辞 | ✓ routing-win 机制 + correctness-carrier 成色 + selector-decline 张力 全披露 | 四节 |

---

## 六、可复现 / 触碰集 / 交付 / 可逆

- **host emit（触碰集内）**：`experiments/active/g5-wiring/M1b-q8_0/{q8_0_gevm_raw.c, q8_0_gemm_raw.c, tcrv_emitted_q8_0.inc}`（emit 自 test/ 既有 MLIR；`.inc` md5 `b5177a4d…`）。
- **deploy harness（新建，触碰集内）**：`tools/e2e-harness/board/g5-m1-q8/{deploy_patch_q8_emitted.py, g5_m1b_build_seal.sh, g5_m1b_correctness.sh, g5_m1b_phase_split.sh}`。
- **casefile（本目录）**：`MANIFEST.md`(本文) · `transmission_accounting.csv` · `correctness_GREEN.txt` · `phase_split_raw.txt` · `evidence.json` · `engage_ON.err`/`engage_OFF.err` · `objdump_gevm_seal.txt`/`objdump_gemm_seal.txt`。
- **A-tree restore（零 stock 永久改动）**：测后 GEN md5=`deb61a29…` · ARCH md5=`99131cf7…` · live `.so` md5=`05a62e6a…`（==OFF-pristine，q8_0 tcrv sym=0，无 stray .inc），全 == baseline。源 restore byte-exact + pristine rebuild。
- **HEAD (TianChen-RV) = `92c27389` 未变** · 全程禁 git · lib/(selector)·schema·ROADMAP·M1 既有 casefile 未碰（只读引）。

## durable files
- `tcrv_emitted_q8_0.inc`（我方 emitted vl=8 GEVM+GEMM）
- `q8_0_gevm_raw.c`（emitted GEVM 源）
- `q8_0_gemm_raw.c`（emitted GEMM 源·net-new oracle-validated）
- `transmission_accounting.csv`
- `correctness_GREEN.txt`
- `phase_split_raw.txt`
- `evidence.json`
- `engage_ON.err`
- `engage_OFF.err`
- `objdump_gevm_seal.txt`
- `objdump_gemm_seal.txt`
