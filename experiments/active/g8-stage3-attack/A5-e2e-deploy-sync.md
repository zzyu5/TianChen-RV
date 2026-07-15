# G8 A5 · e2e 部署树一次性重生同步至当前 HEAD — 同步实录 (2026-07-16)

> 任务：让 e2e 部署树（llama.cpp 集成·我方 `libggml-cpu.so` 发射体）**同步到当前 HEAD**·核 = **deployed==proven bit-identical**·现有 sealed e2e 数字 valid/stale 判定。
> 纪律：**同步/核验·非新 e2e 测量·非新性能主张**·旧 e2e 数字维持冻结引用（§〇.6·不撤不扩·除非 bit-identical 验证）·0 造数·禁 git·板卫生。
> HEAD = `f890babe0`（refactor/full-refactor-m1）。

---

## 0. 盘点：部署物在哪·pin 到哪·vs HEAD 差多少

### 0.1 e2e 部署物清单
| 部署物 | 位置 | pin commit | e2e 数字 | 板 |
|---|---|---|---|---|
| **q5@k1 e2e A/B**（本次同步主目标） | `q5k1-e2e/evidence.md` | 部署核=`d109d6ed2`(selector-fix)+`40c21de0`(REDESIGN-B)·workflow `w91jl99ia`(`12efc3dab`) | decode 1.97/2.07× · prefill 2.21/2.34× | k1/VLEN256 |
| q5_0/q5_1@k1 kernel cold | `k1-q5-selector-fix/`(部署核源)·`q5x-deployed-verify/`(rvv 复验) | `d109d6ed2` verified `a6fdf1a3` | 1.760×/2.241× byte-exact | k1/VLEN256 |
| q4_0 prefill（旧·冻结引用） | `experiments/sealed/repack/rvv-vlen128-q4_0-gemm-constructed-sealed/` | GEMM region `be66c917`（2026-07-06·pre-rename `tcrv_rvv`） | prefill 5.92× / decode 1.91× | rvv/VLEN128 |
| q8_0@ime（旧·冻结引用） | perf-covered cell·ledger `2026-07-12-...q8_0...` | G6-A IME bridge（07-13） | 2.233× beat-stock | k1 IME |
| 其余 6 绿格（q4_1/q8_0/q4_K@k1/q5_K@k1/q4_0@ime + q5_0/q5_1 prefill@rvv） | perf-covered schema | 各 07-06~07-13 | 见 §3 表 | rvv/k1 |

perf-covered 头条快照 pin = `02d6609a`（2026-07-13·recon 机算·schema `_note`）·9/83。

### 0.2 部署树 vs HEAD 差多少 —— ★关键 git 不变量
**从 `d109d6ed2`（q5@k1 部署 pin·2026-07-15）到 HEAD `f890babe0`：编译器源（`lib/` + `include/`）ZERO 变更。**
```
git diff --name-only d109d6ed2 HEAD -- lib/ include/  ==> 0 files
```
`d109d6ed2..HEAD` 全部改动仅在：`.github/workflows/`、`.trellis/scripts/`(recon)、`.trellis/spec/`、`README.md`、`schema/*.json`、`tools/lint/` —— **无一触碰发射路径**。
⟹ **q5@k1 部署核在 HEAD 的前门发射 = 部署时字节相同·deployed==proven bit-identical 由 git 源控不变量【确定性保证】**（比单次 rebuild+diff 更强：证的是全体发射、非单点）。工作树对 `lib/ include/ test/` `git status --porcelain` = 空（clean·binary==HEAD source）。

---

## 1. 重生同步 + deployed==proven bit-identical 核验（q5@k1 = 主同步目标）

**方法**：weft-opt(HEAD·`build-weft/bin/weft-opt`·工作树 clean) 跑部署前门 fixture → emitc → `mlir-translate-20 --mlir-to-cpp`，byte-compare vs 板部署核。
- 前门 recipe（部署核 = VLEN256 half_lanes16）：
  `weft-opt <fixture> --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`
- q5_0 fixture: `test/Conversion/RVV/rvv-lower-quant-contraction-vlen256-decode-per-format-measured.mlir`
- q5_1 fixture: `test/Conversion/RVV/rvv-lower-quant-contraction-q5-1-vlen256-decode-measured-repack.mlir`
- 部署核（板集成源）: `k1-q5-selector-fix/kernels/q5_{0,1}_gevm_frontdoor_vlen256.c`

**结果：BYTE-IDENTICAL（0 diff·md5 相等）**
| 核 | HEAD 重生 md5 | 部署核 md5 | diff | 行数 | 判定 |
|---|---|---|:--:|:--:|:--:|
| q5_0 GEVM VLEN256 decode | `8cd0c6978c3c6287e44d681f41ccff7a` | `8cd0c6978c3c6287e44d681f41ccff7a` | 0 | 127 | **deployed==proven ✓** |
| q5_1 GEVM VLEN256 decode | `4ad42d915e35505dfeda89065fac2532` | `4ad42d915e35505dfeda89065fac2532` | 0 | 135 | **deployed==proven ✓** |

- REDESIGN-B 指纹在 HEAD 重生核确认：q5_0 `vlm_v_b16`×2 + `vmnand`×4 + `vsub_vx_i8mf2_mu`×4·OLD-chain(`vsrl_vv/vsll_vx/vncvt/vor`)=**0**；q5_1 `vlm_v_b16`×2 + `vadd_vx_u8mf2_mu`×4·OLD-chain=**0**。= 部署的是 REDESIGN-B 削重建核（非 OLD bit-scatter）·与 q5k1-e2e §2 objdump `vlm.v` 指纹一致。
- 函数签名逐字符相同（`weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(...)`）。

⟹ **q5@k1 e2e 部署核【同步到 HEAD·deployed==proven bit-identical 双证】：① git 源控不变量（emitter 零漂移 d109d6ed2..HEAD）② HEAD 重生 emitc byte-identical vs 板部署核。**

**补充覆盖（q5@k1 全部件均 bit-identical）**：
- **decode 核**（1.97/2.07× 直接支撑）：md5 直证 byte-identical（上表）。
- **prefill GEMM 核 + kernel-sym 核**（2.21/2.34× / 1.76/2.24×）：同一发射器文件 `RVVToEmitCBlockQuantLinear.cpp`·d109d6ed2..HEAD 零漂移 → git 不变量覆盖 = bit-identical。
- **rvv VLEN128 q5_0 GEVM 旁证**：HEAD 重生 vs `q5x-deployed-verify/raw/q5_0_gevm.c`（40c21de0 生成）= **0 diff·byte-identical**（202 行）→ q5 发射路径在 40c21de0..HEAD 全程稳定。

---

## 2. 旧部署物漂移分析（§〇.6 冻结引用·非本次同步目标）

**发射器最后改动日**（决定漂移面）：
- `RVVToEmitCBlockQuantLinear.cpp`（q4_0/q4_1/q5x/q8_0/K-quant GEMM+GEVM）→ 末次 `38abf20eb` **2026-07-14**。
- `IMEBackendEmissionDriver.cpp`（IME tile）→ 末次 `7c21088fb` **2026-07-14**（含 `2fb3f0faa`「现格 emit 逐字不变」+ `513bcb7bc` VLEN-invariance bug 修）。
- 改名边界 `fc72fe53`（**2026-07-12**·tcrv→weft·numerics byte-exact 零漂移·但**符号名/注释 namespace 文本变**）。
- **d109d6ed2..HEAD = 全发射器（含 IME）零漂移**（`git diff --name-only d109d6ed2 HEAD -- lib/Plugin/IME lib/Dialect/IME include/Weft/{Plugin,Dialect}/IME` = 0）。

**旧格发射机制在 HEAD 均 INTACT**（前门仍 auto-select+realize·部署路径未断）：
- q4_0 GEMM prefill @HEAD：`contraction_algorithm="repack"` + `typed_repack_gemm_loop_body` + `path_materialization="realized"` ✓
- q8_0 GEMM prefill @HEAD：同上 ✓
- q8_0@ime tile @HEAD：`vmadot`×12 + ime×42 = IME tile 仍发射 ✓

**但旧格 e2e 数字 ≠ bit-identical re-certified**（原因）：
- **q4_0 prefill 5.92×**（seal `be66c917` 2026-07-06·**pre-rename**）：sealed 指纹符号 = `tcrv_emitc_ggml_gemm_q4_0_q8_0_...`（板 .so 2026-07-06 编译）·HEAD 发射 = `weft_emitc_...`（rename 后）→ **源文本 DRIFT 坐实**（namespace/符号名）·加 07-14 GEMM campaign 触及全 prefill-GEMM leaf·**无 in-repo emitted core 可 byte-diff**·e2e 板 .so re-deploy = 新 e2e（本役外·下一 node）→ **冻结引用·未 re-cert**。
- **q8_0@ime 2.233× / q4_0@ime 1.0088×**（seal `468fcef71` 2026-07-13）：发射器 07-14 refactor(键控化·commit 声明「现格 emit 逐字不变」但含 VLEN-invariance bug 修) → 逐字不变【声明】未经本役 byte-diff 独立坐实（需 IME fixture emit-diff + 板）→ **冻结引用·emitter-refactored-verbatim-claimed·re-cert deferred**。
- **q4_1 3.68× / q8_0@rvv 4.35× / q4_K@k1 1.085× / q5_K@k1**（seal 07-06~07-13）：发射器 rename+07-14 均触及·同理 **冻结引用·未 re-cert**。
- **q5_0/q5_1@rvv prefill 1.21/1.09×**（07-12 ledger）：VLEN128 GEVM 路已证 40c21de0..HEAD 稳定；prefill GEMM leaf 经 07-14 campaign·未单独 byte-diff → 冻结引用·re-cert deferred（低风险·同发射器文件）。

## 3. 现有 e2e 数字 valid / stale 清单

| # | e2e 数字 | 板 | pin | HEAD bit-identical? | 状态 |
|---|---|---|---|:--:|:--|
| 1 | **q5_0@k1 decode 1.97×** | k1 | d109d6ed2 | **YES（md5 直证）** | **VALID·deployed==proven** |
| 2 | **q5_1@k1 decode 2.07×** | k1 | d109d6ed2 | **YES（md5 直证）** | **VALID·deployed==proven** |
| 3 | **q5_0/q5_1@k1 kernel cold 1.76/2.24×** | k1 | d109d6ed2 | **YES（同核·git 不变量）** | **VALID·deployed==proven** |
| 4 | **q5@k1 prefill 2.21/2.34×** | k1 | d109d6ed2 窗口 | **YES（git 不变量·同发射器文件）** | **VALID·deployed==proven** |
| 5 | q4_0@rvv prefill 5.92× / decode 1.91× | rvv | be66c917 (07-06) | NO（源文本 drift·rename tcrv→weft·无 core diff 目标） | **冻结引用（§〇.6）·未 re-cert·机制 intact** |
| 6 | q4_1@rvv prefill 3.68× / decode 1.67× | rvv | 07-12 | 未 cert（发射器 07-14 触及） | **冻结引用·未 re-cert** |
| 7 | q8_0@rvv prefill 4.35× / decode 3.81× | rvv | 07-12 | 未 cert | **冻结引用·未 re-cert** |
| 8 | q5_0/q5_1@rvv prefill 1.21/1.09× | rvv | 07-12 | GEVM 路稳定·GEMM leaf 未单独 diff | **冻结引用·re-cert deferred（低风险）** |
| 9 | q4_K@k1 e2e prefill 1.085× (Win-K1-VLEN) | k1 | 07-09 区 | 未 cert | **冻结引用·未 re-cert** |
| 10 | q5_K@k1 our-kernel | k1 | 07-13 区 | 未 cert | **冻结引用·未 re-cert** |
| 11 | **q8_0@ime 2.233× beat-stock** | k1 IME | 468fcef71 (07-13) | 未 cert（emitter 07-14 refactor·「逐字不变」声明未独立 byte-diff） | **冻结引用·re-cert deferred·机制 intact** |
| 12 | q4_0@ime 1.0088× tie-stock | k1 IME | 468fcef71 (07-13) | 未 cert（同上） | **冻结引用·re-cert deferred·机制 intact** |

★**无「坏漂移」需撤**：q5@k1（本次同步目标）= VALID·bit-identical。旧 5–12 = §〇.6 设计态「冻结引用」（非本次同步范围·非被打脸的错数），仅一处**源文本漂移坐实**=q4_0 rename 符号名（pre-rename 板 .so vs HEAD·预期内·不影响 5.92× 历史事实·仅表示需 rebuild 才能作 HEAD-live 引）。

## 4. 板卫生 + 裁决

- **本役全程 LOCAL**（git 分析 + weft-opt HEAD 前门重生 + byte-diff）·**未 ssh 任何板·未建板 scratch·未改 stock `.so`**（无 before/after 变动·因未触板）。历史板 scratch（`/tmp/g8q5e2e`·`/tmp/q5x_deploy_verify`）各自 evidence 已报清·未由本役重开。
- 工作树 `git status --porcelain lib/ include/ test/` = 空（binary==HEAD source·发射确定性）。**未 git commit**（改动/实录留主会审·仅本 .md + scratch）。
- scratch = `/tmp/.../scratchpad/a5emit/`（重生 core + diff·用完可清）。

### VERDICT
- **e2e 部署树【一次性同步至 HEAD】DONE**：sync 目标 = q5@k1 部署核 → **deployed==proven bit-identical 双证（git 源控不变量 + md5 byte-diff）**·decode/prefill/kernel 三部件全覆盖。
- **q5@k1 e2e 数字（decode 1.97/2.07×·prefill 2.21/2.34×·kernel 1.76/2.24×）= 在当前 HEAD 代码有效（VALID）。**
- 其余 7 绿格 e2e 数字 = **§〇.6 冻结引用（设计态·未 re-cert·非本役同步范围）**·发射机制在 HEAD 全 intact·仅 q4_0 有预期内 rename 源文本漂移·**旧数不撤不扩**·作 HEAD-live 引须各自 bit-identical rebuild（属下一 node「e2e 传导战役」）。
- 0 造数·遇阻如实（旧格 board-e2e re-cert = out-of-scope 明标·非降级）。
