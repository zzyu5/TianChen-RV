# T4b — 选择器四配置消融矩阵 (cost-model × scenario)

> Casefile for **覆盖式铺面④ (测量总攻队列衔接项)**: systematize the scattered
> selector priors — M0 falsifier (2 GEMM configs) + T5c M-aware decode boundary —
> into the complete **四配置消融矩阵** the 实验总纲 §4.2 / T4b table calls for, and
> add the two DECODE cells that pin the PRECISION of the capability-derived +
> M-aware fix. **In-tree · selector · lit/gtest · 无板 · [NG-4] 机制 C3′ 证据**
> (no perf, no throughput, no beat claim). Behavior is only OBSERVED, never changed.

## 定位 (先验层已在 · 本项系统化)
选择器先验层已建于三个 commit:
- **M0** `c6142da3` — `runSel2CrossParadigmMisfireFalsifierTest`: 2 配置 (blind 坏 / derived 好) × GEMM.
- **SEL-1-T5** `2942f603` — capability-derived cross-paradigm cost.
- **T5c** `11358834` — M-aware `estimateVariantCost` (M≥M* GEMM 偏好 / M<M* decode parity) + `capability-prior-ime-mstar-writeback.mlir`.

**T4b = 把这些系统化为四配置消融矩阵 + 矩阵静默落败复现/消失专项。**

## 四配置消融矩阵 (cost-model × 场景)

|                              | **GEMM** (M≥M* · compute-bound prefill · roofline 要 **MATRIX**) | **decode/fragment** (M<M* · memory-bound · roofline 要 **VECTOR**) |
|------------------------------|--------------------------------------------------|-----------------------------------------------|
| **常量盲 cost** (pre SEL-1-T5)   | **(a) ★静默落败复现** — 20>1 升序 → chosen=**VECTOR** (误·无报错) | **(d) 碰巧对** — 同 20>1 → chosen=**VECTOR** (对) |
| **能力派生 cost** (SEL-1-T5+T5c) | **(b) 落败消失** — 0.5<1 → chosen=**MATRIX** (正) | **(c) parity** — 1.0==1.0 → tie-break → chosen=**VECTOR** (正·不偏矩阵) |

**roofline-correct paradigm**: GEMM 要 systolic MATRIX 阵 (M 达交叉点 M*, compute-bound);
decode 要 RVV VECTOR 路 (M<M*, memory-bound — compute-isolated micro-advantage 不传导, micro↛e2e)。

### 四格逐配置断言 (chosen + reason + score 来源)
| cell | 注入 (matrix, vector) | chosen | 正确性 | reason | score 来源 |
|---|---|---|---|---|---|
| (a) 盲×GEMM   | (20, 1)  | vector | **误 (misfire)** | static_order | 常量盲 (test-plugin 注入 == pre-SEL-1-T5 IME 常量) |
| (b) 派生×GEMM | (0.5, 1) | matrix | 正 | static_order | 能力派生 (`ime_matmul_shape` ∧ M≥M* → `kIMEMatmulGemmPreferredCost`) |
| (c) 派生×decode(parity) | (1.0, 1) | vector | 正 (不偏矩阵) | static_order | 能力派生 (M<M* → `kIMEMatmulDecodeParityCost` = RVV base) |
| (d) 盲×decode | (20, 1)  | vector | 正 (**碰巧**) | static_order | 常量盲 (与 (a) 同数, 场景不同) |
| (c′) 派生×decode(fragment) | (20, 1) | vector | 正 (双排除) | static_order | 能力派生 (single MAC fragment → `kIMESingleFragmentMacCost`) |

> tie-break (equal score): `explicit_preference` → ascending score → non-fallback → **original IR order** → symbol name (`ExtensionPlugin.cpp:1383` `std::stable_sort`)。cell (c) parity (1.0==1.0) 由 IR order 决定 (vector 先声明 → vector rank 0)。

## 四条消融读数 (cross-config, load-bearing)
1. **ABLATION-1 (GEMM 列 · 静默落败复现→消失)**: (a) chosen=vector 误 → (b) chosen=matrix 正; verdict DIFFER ⇒ 能力派生先验【消除】misfire (非"好配置碰巧能跑")。
2. **ABLATION-2 (decode 列 · 不过度纠正)**: (d) 与 (c) 同 chosen=vector; verdict SAME ⇒ 修法【不】把 decode 误纠到矩阵 (compute-micro 信号 memory-bound decode roofline 不兑现)。
3. **ABLATION-3 (精确性 · 2×2 对角)**: blind→derived 只翻 GEMM verdict (a→b: vector→matrix), decode verdict 不动 (d→c: vector→vector) ⇒ 能力派生 + M-aware = **精确修复**, 非"一律偏矩阵"。
4. **ABLATION-4 (常量盲=场景盲)**: (a) 与 (d) 注入【相同】分 (matrix 20 > vector 1) → 【相同】verdict (vector), 却 (a) 误 (GEMM) / (d) 对 (decode); 单一场景盲常量分不开 GEMM 与 decode。能力派生【场景感知】: 同一 matrix 变体 GEMM 记 0.5 / decode 记 parity 1.0 (0.5 ≠ 1.0)。

## 工件 (in-tree · 主会话提交)
| cell(s) | 载体 | 文件 |
|---|---|---|
| (a)(b)(c)(d)(c′) + 四读数 + decode-side commit lock | **gtest** | `test/Transforms/VariantSelection/VariantSelectionTest.cpp` :: `runT4bFourConfigSelectorAblationTest` |
| (a)(b) GEMM 两格 + [SEL-2] commit-timing | gtest (M0, 已存) | 同上 :: `runSel2CrossParadigmMisfireFalsifierTest` |
| (b) 派生×GEMM 生产端 (matM=256) | **lit** (已存) | `capability-prior-ime-gemm-over-rvv.mlir` |
| (b) 派生×GEMM (matM=64) + (c′) 派生×decode(single-fragment) 生产端 | **lit** (已存) | `capability-prior-ime-mstar-writeback.mlir` |

## Provenance / defense-in-depth 诚实标注 (照 T5c)
- **DERIVED 行 (b, c) 生产端已证**: capability-prior-ime-gemm-over-rvv.mlir + capability-prior-ime-mstar-writeback.mlir 在真 RVV+IME 生产 plugin 上端到端证 (选择→归因 JSONL→材化 commit)。gtest 的 (b)(c) 是同一闭环的注入半, 与生产 lit 交叉校验。
- **BLIND 行 (a, d) 天生 gtest-only**: SEL-1-T5 落地后生产 plugin 【不能】对 GEMM 形再吐旧盲 20; 盲常量经 sanctioned test-plugin 常量分路注入 (M0 同法)。
- **cell (c) parity 分支 = 审计级 defense-in-depth, 非生产可达 tiled-emission 路**: tiled-shape 派生 fail-closed 于 matM<macM (no remainder path), 真 tiled GEMM 到 cost 必 matM≥macM。**真-可达生产 decode 路 = single MAC fragment (cost 20)** = cell (c′), 由 capability-prior-ime-mstar-writeback.mlir 证。**decode 双排除**: parity 1.0 与 single-fragment 20 两条派生 cost 机制都把 decode 挡在矩阵范式外。**未硬造 M=1 tiled parity 可达样本** (需 GEVM plumbing / remainder emitter = 独立立项, 留裁)。

## 实验总纲 对齐 (§4.2 / T4b / T8)
- 实验总纲 §4.2 的四配置 = **(常量 / 仅先验 / 仅记忆 / 先验+记忆)** — "仅记忆 / 先验+记忆" 列 gated on **[SEL-3] measurement-memory (未落地)**, 今天不可全落。
- 本项落地的四配置 = **(常量盲 / 能力派生) × (GEMM / decode)** = 今天【可落】的矩阵, 系统化 [SEL-2] 静默落败专项到【两个场景】, 证能力键控 + M-aware【精确】修复。
- **T8 台账**: 矩阵静默落败 (→先验层) 实例现有【复现 (a) + 消失 (b) + 不误伤 decode (c/d)】的持久 gtest 断言支撑。
- **reason 枚举**: 仍 `static_order` (未翻 `prior`); prior 翻转 = 独立 canon-gated 燃减步, 本项【未做】。

## 验证 (2026-07-11)
- clean build green; `tianchenrv-variant-selection-test` **EXIT=0**.
- VariantSelection lit **7/7 PASS** (含 (b)(c) 生产端 lit 逐字不变 = 保行为)。
- 变异测试证判别力 (×2, 还原后 green):
  - decode over-correct: `kDerivedDecodeParityCost` 1.0→0.4 → cell (c) 断言 FAIL, EXIT=1.
  - GEMM misfire: `kDerivedGemmMatrixCost` 0.5→20.0 → cell (b) 断言 FAIL, EXIT=1.
- 保行为铁门: **零 lib/ 改动** (触碰集仅 `VariantSelectionTest.cpp` + 本 doc); 选择逻辑 (M0/SEL-1-T5/T5c) 未动, 只加消融观测断言。
