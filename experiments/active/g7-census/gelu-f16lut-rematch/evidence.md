# G.0.3 gelu SAME-PRECISION-TIER rematch — JUDGMENT-SUSPENDED release (dual-board)

> **Line**: 〇.3 明令清偿（上役 gelu 比蒸发事故的清偿）. gelu 旧比 = ours **tanhf**(高精度) vs opp **f16-LUT**(低精度·~2500× 精度差) = **精度不对等 → JUDGMENT-SUSPENDED**. 本役发我方 **f16-LUT 同档变体** → 双板 0.8 cold 同档重比 → 胜负同权入表 → 精度双列披露 → **解除挂起**.
> **Axis**: kernel-sym micro（第二赛道·forward-elementwise family）· [NG-4] NOT e2e · NOT perf-covered · NOT a sealed Win · 独立于 matmul kernel-sym-12. Amdahl: B-fwd <1% decode → 即便全胜无 e2e 传导（同 bclass §7 纪律）.
> **Boards**: rvv / VLEN128 / gcc-15.2.0 symmetric · k1 / SpacemiT-X60 / VLEN256 / clang-18.1.8 symmetric（[CASE-COMPILER-ASYMMETRY] shipped compiler each side, both sides same compiler）.
> **Casefile**: `experiments/active/g7-census/gelu-f16lut-rematch/` — kernels/gelu.f16lut.kernel.c, gelu_f16lut.materialized.mlir, opp_gelu.cpp, gelu_f16lut_driver.c, board_run.sh, regen_kernel.sh, raw/{seal_k1.txt,run_k1.log,seal_rvv.txt,run_rvv.log,host_byteexact_check.cpp}.

---

## 0. 施工 — 我方 f16-LUT gelu 变体（小型·走前门/emitter）

- **触碰的 emitter 面（最小·gelu-only·guarded → 默认路径零回归）**:
  - `include/Weft/Dialect/RVV/IR/…` 无改（用 OPTIONAL discardable attr，不动 ODS）。
  - `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` `ElementwiseGeluMapOp::verify()`: 在 I4/I7 fail-closed 边界内**新增一个 bounded 数值-契约 attr** `gelu_precision`（仅允许值 `"f16lut"`；非 resource/scheduling 旋钮）。缺省=exact-tanhf 档。
  - `lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp` `emitForwardGeluScalarLoop` + `emitElementwiseGeluMapStrip`: 读 `gelu_precision`；`f16lut` 时 per-element 体 = 单个 `weft_gelu_f16lut_scalar` opaque seam（替换 tanhf 算术链），loop/load/store 结构不变。
  - `lib/Conversion/RVV/RVVToEmitC.cpp` func-builder: 检测 gelu-f16lut brick → 加 `<math.h>` + emit module preamble（f16↔f32 转换 + 表 + `weft_gelu_f16lut_init`）。
- **发射机制**: 前门 `weft-opt <fixture> --…-materialize-forward-elementwise-stream-front-door` → 注入 `gelu_precision="f16lut"` → `--weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`。确定性再生 = `regen_kernel.sh`（md5 `1870c960…`）。
- **同数值契约 + 同运行算法（关键）**: seam **不是** tableless 重算——它复现 ggml as-shipped GGML_GELU_FP16 **完整路径**：65536-entry f16 查表（`weft_gelu_f16lut_init` 一次性构建·镜像 ggml_init 的 `ggml_table_gelu_f16`·**在计时路径外**·与对手 `opp_gelu_init` 对称）+ per-element clamp/convert/**gather**/convert。**运行时无 tanhf**（objdump `tanh_calls=0` 双板证实）。→ 既匹配精度档**又**匹配运行算法（表查 vs 表查），才是公平同档速度 A/B。
  - ⚠ **中途纠偏（已修·记档）**: 初版 seam 用 tableless 记忆化（byte-exact 但运行时逐元素重算 tanhf）→ host 实测 0.074×（13× 慢）——那是「表查 vs 全重算」不公平比，非同算法。改为真表查后 host 回到 0.97–1.00× near-parity。**便宜档的公平性在于同算法，不只同精度。**

### 0.1 byte-exact vs 同档 f16-LUT oracle（非 tanhf oracle）
- oracle/opp = `opp_ggml_vec_gelu_f32`（`opp_gelu.cpp`·**逐字**取自 bclass `opponent_ggml.cpp` 的 GGML_GELU_FP16 路径·vec.h:46/968/988）。
- **代数恒等证明**: 查表 = 纯记忆化 → `table[o_f32_to_f16(x)] = o_f32_to_f16(gelu(o_f16_to_f32(o_f32_to_f16(x))))`；我方表用**同一** `o_f32_to_f16/o_f16_to_f32/ggml_gelu_f32`（同源同 flags）构建 → 逐 bit 相同。
- **实测（host + 双板·全 8 shape·set-0 gate）**: `ours_vs_opp_maxulp = 0`（GATE=PASS 全部 24 格 = 3 环境 × 8 shape）。

---

## 1. ★ 双板 0.8 cold 同档重比（anchor n=4096·cold parity band ±5%·N=12 median·单实例·load-gate·pin）

| 板 | 编译器（对称） | anchor 4096 cold ratio ours/opp | 全 shape cold 范围 | GATE(byte-exact) | 判（同档·±5%） |
|---|---|:--:|:--:|:--:|---|
| **rvv** (VLEN128) | gcc-15.2 both sides | **1.116×** | 1.065–1.116× (≥parity 全 shape) | PASS 0 ULP | **WIN**（同档·便宜档表查·gcc-15 调度胜） |
| **k1** (VLEN256) | clang-18 both sides | **0.957×** | 0.931–0.961× | PASS 0 ULP | **PARITY**（anchor 内 ±5%·small-n 0.93 LOSS-lean） |

### 1.1 full shape sweep（cold ratio ours/opp）
| shape | 512 | 1024 | 2048 | 3072 | 4096 | 5120 | 8192 | 16384 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **rvv** | 1.065 | 1.102 | 1.114 | 1.111 | **1.116** | 1.113 | 1.102 | 1.079 |
| **k1**  | 0.931 | 0.949 | 0.954 | 0.956 | **0.957** | 0.959 | 0.958 | 0.961 |

- IQR 0.1–0.8%（极稳·全 32 板测格）. 冷 GB/s: rvv ours ~0.79 / opp ~0.71；k1 ours ~0.22 / opp ~0.23（两板均 memory-bound·远低于计算 roofline·表查=纯访存）.
- 单实例前科防护: 测前 `pkill` 竞争 + `stray_pre_kill=0`；测后 `pgrep -x gelu_f16lut_bin = 0`（双板）；`-f count=1` 为 pgrep 自匹配 ssh 参数串的已知假阳（同 bclass §6 记录）. 无 stock .so 触碰（opp 自足·无导出符号链接）.

---

## 2. 胜负同权入表 + 输局桶带墙分类（§〇.5 三档·不藏）

- **rvv = WIN 1.116×（同权登记·但禁称硬赢）**: 便宜档 = f16 查表·memory-bound（~0.8 GB/s）·**非** hand-brick 强手调对手。同算法（both 表查·对称 gcc-15）→ 差异 = **gcc-15 对我方 emit 的表查循环调度略优于对其手排 C 循环**。属「同算法编译器调度」类·与 bclass §6.5 scale 0.93→1.02 flip 同族。**成色 = 公平同档速度胜·非结构/硬碰硬赢**。
- **k1 = PARITY 0.957×（同权登记·近-parity·small-n LOSS-lean）**: 输局桶 = **内存墙 roofline**（both ~0.22 GB/s·表查纯访存·near-parity）+ 残余 ~4% = 对称 clang-18 对我方 emit SSA 顺序 vs opp 手排 C 的最终调度方差（**非**三出口的对手结构优势/重建成本/硬件缺席）。具名 **[GAP-GELU-K1-VLEN256-CLANG-SCHED]**（与 §7.4 [GAP-SILU-RVV-VLEN128-GCC-SCHED] 同类·op-identical byte-identical·最终 schedule 归下游编译器）。small-n(512) 0.931 = P 小时 loop-tail/表查 warmup 占比·anchor 平稳。
- **禁互推**: forward-elementwise 桶·独立于 matmul kernel-sym-12 / perf-covered 9/83 / DEQ-AXIS · 无 e2e 传导（Amdahl <1% decode）. 不动 perf-covered/certified 计数.

---

## 3. 精度双列披露（同档 vs 旧 tanhf-vs-f16LUT）

| 比对 | ours 数值档 | opp 数值档 | ours-vs-opp | 判 |
|---|---|---|---|---|
| **本役同档（f16-LUT vs f16-LUT）** | rel≈8–9.5e-4（f16 tier·vs fp64-tanh oracle） | rel≈8–9.5e-4（**逐格相同**） | **0 ULP（byte-exact·全 8 shape 双板）** | 精度对等·公平 A/B 成立 |
| 旧比（tanhf vs f16-LUT·已挂起） | rel 3.7e-7(k1)/3.8e-7(rvv) | rel 8.5e-4/8.7e-4 | ~11000 ULP·~2300–2500× 精度差 | 精度不对等 → JUDGMENT-SUSPENDED（旧结论作废为「速度比」） |

- 同档后 ours 与 opp 的 rel（vs fp64-tanh）**逐格数字完全相同**（8.305e-4/7.954e-4/9.171e-4/…）→ 二者落在**同一 f16 量化格点**·差异真正塌缩到 0 ULP。旧比的 ~2500× 精度鸿沟是「我方算精确 tanhf·对手查 f16 表」的**精度档不同**·非速度可比。

---

## 4. ★ JUDGMENT-SUSPENDED 解除确认 + gelu census 终态

- **解除**: 精度档已对齐（both f16-LUT·0 ULP）→ 速度 A/B 现为 apples-to-apples → **JUDGMENT-SUSPENDED 解除**。
- **gelu census 终态（挂起 → 终态·同权·带墙）**:
  - **rvv: PASS·同档 WIN 1.116×**（便宜档表查·gcc-15 调度胜·非硬赢）.
  - **k1: PASS·同档 PARITY 0.957×**（内存墙 near-parity + [GAP-GELU-K1-VLEN256-CLANG-SCHED] 残余调度·small-n LOSS-lean）.
  - 双板 byte-exact 0 ULP·同 f16 tier·同表查算法·全 shape GATE=PASS.
- **旧「LOSS 0.25×/0.15× STRUCTURAL LUT-vs-tanhf」条目**: 该数是**跨精度档**（我方 tanhf 全算 vs 对手表查）·**非同档速度**·已作废（挂起解除后不再作为 gelu 的胜负判据·仅作精度双列的「旧比」历史锚）.

---

## 5. 板卫生 + 建议 commit message

- **卫生**: 双板 stray=0（`pgrep -x`）· 无 stock .so 触碰（opp 自足）· 板 footprint 仅 `/tmp/gelu_f16lut` scratch· load-gate 全 PASS（rvv core-13 idle100%/k1 core-1 idle100%·gov=performance）.
- **lit 零回归**: RVV Conversion+Dialect **385/385 PASS**（含默认 gelu fixture 仍 emit tanhf·无 f16lut helper·无 math.h）· weft-rvv-dialect-test 4/4 · 验证器 fail-closed 双证（bad `gelu_precision` 值拒 / 无关 attr 拒·I7 完好）.
- **不 git commit**（主会话 §四.3 复验后提交）.

**建议 commit message**:
```
report(G.0.3 gelu 同档重比清偿·JUDGMENT-SUSPENDED 解除): f16-LUT 变体前门发射+双板 0.8 cold 同档重比

- 施工: gelu emitter 小型变体(guarded gelu_precision="f16lut" bounded attr·默认 tanhf 档零回归)
  发我方 f16-LUT gelu = 复现 ggml GGML_GELU_FP16 完整表查(65536 f16 表·init 在计时外·运行时无 tanhf)
- byte-exact vs 同档 f16-LUT oracle: ours-vs-opp 0 ULP(全 8 shape 双板·代数恒等: 表=记忆化)
- 双板 0.8 cold 同档重比(同权入表·带墙): rvv 1.116× WIN(便宜档表查·gcc-15 调度胜·非硬赢)/
  k1 0.957× PARITY([GAP-GELU-K1-VLEN256-CLANG-SCHED]·内存墙 near-parity)
- 精度双列: 同档 0 ULP(both rel~9e-4 f16 tier) vs 旧 tanhf-vs-f16LUT ~2500×/~11000 ULP(精度不对等·作废为速度比)
- lit 385/385·dialect 4/4·验证器 fail-closed 双证·板卫生(stray=0·无 stock .so 触碰)
```
