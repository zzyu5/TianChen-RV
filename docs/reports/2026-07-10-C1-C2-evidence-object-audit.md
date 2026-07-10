# C1 / C2 证据对象正名核查 — 报告（只核查、不改 canon）

> **性质**：P0 canon 级只读核查。本文只出**判定 + 误挂位置清单 + 建议改法**，供用户确认。
> **未改任何 canon / inventory / spec / ledger / 论文提纲。** 确认后由用户授权一次性改（禁双改）。
> 核查日期 2026-07-10；快照 branch `refactor/full-refactor-m1`。

---

## 0. 判定（headline）

**C2 是否被误挂前门化 decode-format 成本？ → 是（CONFIRMED）。**

- **canon 的 C2 权威对象 = 独立 extension family（RVV→IME→scalar→zvfh）接入成本**，锚 = **IME ≈2484 行**，曲线 = `IME 2484 → 标量家族 <300 → 子扩展 zvfh <100`。canon 本身**诚实标注该曲线尚 = 缺失**（只有 IME 一个真家族落地，需 ≥3，依赖 [X-SCALAR]）。
- **但 `docs/reports/` + `experiments/active/` 的一批 ledger 把 RVV-plugin 内部的 decode-format 前门化 LOC 摊销**（ternary / K-quant / codebook / IQ-grid / repack，几十个数据点）**当成"C2 边际成本律/曲线"在记**，并在 paper-evidence-index 里**整段替换掉了 C2 的 extension-family 对象**（C2 行不再指向任何 IME/scalar 证据）。
- 后果 = 制造"C2 已有丰富多点干净曲线"的假象，掩盖 canon 记的真相"C2 真对象只有 1 点、仍缺失、待 X-SCALAR"。这正是用户怀疑的误记。

**C1 锚点是否稳？ → 核心稳、但"可复制协议"措辞被 RVV 内部 decode 构造协议部分冒占（dilution 风险，非塌陷）。**

---

## 1. 权威口径（据以核查，两处 canon 一致）

**`docs/canon/TianChen-RV_科研目标总纲v2.md`**
- L152 `### 4.2 C2 —— 依赖式分层接纳 + 泛化代价（→ 边际成本规律）` — "依赖式分层" = [L-2] integrated / independent-attached，即 **extension family**。
- L154 `[C2-1] 主张`：逐家族接入代价成边际递减曲线 **（IME ≈2484 行 → 标量家族 <300 行 → 子扩展 zvfh <100 行）**。← 数据点全是 extension family。
- L157 `[C2-4] agent 检查`：**2484 可复算**（IME 家族源）。
- L158 `[C2-5]`：**家族#3（标量）成本 ≈ #2（IME）的 1/10**。
- L113 `[LED-2]`：边际成本曲线家族 ≥3 出图，**依赖 [X-SCALAR] 落地**。
- L181-183 roster：`SpacemiT IME = 家族#2 = C2 首点` / `zvfh = C2 子扩展粒度点(<100)` / `X-SCALAR 标量 = 家族#3 = C2 第二点(<300)`。

**`docs/canon/TianChen-RV_执行总纲v2.md`**
- L93 `[LED-1]`：首点可复算 = **IME 4 目录 raw wc-l=2484**（等级=部分，C2）。
- L94 `[LED-2]`：**仅 IME 一个真家族有数（需 ≥3）；依赖 [X-SCALAR] 落地**（等级=**缺失**）。
- L279 `[C2]`：达标 = 自动 ledger + **第二家族（依赖 X-SCALAR）成曲线**。
- L50 `[S-8]`：C2 的"家族"= **能力门家族**（`[X-SCALAR]` 能力门家族未建）。

**`.trellis/spec/index.md`**
- L27 C2 表行：逐家族接入代价成边际递减曲线；术语按 **[L-2] integrated/independent-attached**。
- L42 bridge：`C2 = 在 N2 的零核心改动不变量之上，度量逐家族接入的边际成本`。N2 = 零-core-branch **跨独立 extension family** 泛化（IME 已证）。

→ **三处 canon/spec 完全一致：C2 的"家族"= extension family（RVV/IME/scalar/zvfh），锚 IME 2484，曲线尚缺失待 scalar。** decode-format（q4_K/ternary/codebook…）的能力谓词闭包全 ∩ rvv.*≠∅，按 `core-invariants [F-6]` **不是** independent family，**根本不进 C2 的分母**。

---

## 2. 误挂位置清单（供确认后一次改；数值价值全保留，只改归属贡献）

### 2A. 应从 C2 改挂 C3′（construction 谱系）—— decode-format 前门化 ledger

| # | 文件 | 误挂证据（行/字样） | 现挂 | 应挂 |
|---|---|---|---|---|
| 1 | `docs/method/C2_marginal_cost_ledger.md` | **全文**。L1 标题 `C2 边际成本 ledger — block-quant 家族 typed 构造`；L3 `这是 C2(泛化代价→边际成本规律)的代码实证台账`；数据全是 decode 格式（q4_K/q5_K/q6_K/q2_K/q3_K/q4_0-repack/iq4_nl/iq1_s/iq1_m） | C2 | **C3′**（模式库构造经济学 / 成熟编译器覆盖） |
| 2 | `experiments/active/frontdoor-framework/MANIFEST.md` | 头部 `campaign: frontdoor-framework (C2 layer-B)`；`role: NEW C2 evidence line ("layer B")`；`why a separate ledger (not a T2 row)` 全篇按 C2 记 decode-family 前门化 | C2 layer-B | **C3′** |
| 3 | `experiments/active/frontdoor-framework/frontdoor_framework_ledger.csv` | 32 行 seq 0-31 全是 decode 格式（tq2_0/tq1_0/q*_K/iq*/mxfp4/nvfp4）的 `reusable_framework_LOC`/`family_specific_LOC` | C2 | **C3′** |
| 4 | `experiments/active/frontdoor-framework/NOTES.md` | `C2 evidence line layer B`；`second decode family`；`the C2 headline … restated on the front-door axis` | C2 | **C3′** |
| 5 | `experiments/active/result-tables/T2_C2_ledger_marginal_cost.csv` | **seq 1-13**（q8_0/q4_0/q4_1/flat-retire/q*_K/repack/iq1）= decode/loop-shape 构造，键在 `ΔC_construct`（构造轴，非接入轴） | C2 | **C3′**（构造轴）**；⚠ seq 0 = IME 2484 是真 C2，见 2C，勿一并改** |
| 6 | `docs/reports/2026-07-09-C1-C2-construction-evidence.md` | L9 `C2 = …(the front-door-ization marginal-cost law)`；**Part B 全部**（§B.2 decode-family head/tail LOC 表 / §B.3 摊销律 / §B.4 曲线）；L211-212 自称 `C2 层-B 证据` | C2 | **C3′** |
| 7 | `docs/reports/2026-07-10-paper-evidence-index.md` | **L8** `C2 = 泛化代价 → 边际成本规律（front-door-ization marginal-cost law）`；**L41-43** C2 证据表**只**指向 decode-format ledger（C1-C2-construction Part B / T2 CSV / frontdoor ledger / C2_marginal_cost_ledger），**零** IME/scalar extension-family 指针 | C2 | C2 行**恢复** extension-family 对象（见 2C）；decode ledger 指针**下移** C3′/成熟编译器 |

**核心补充（#7 最伤）**：paper-evidence-index 的 C2 行**完全没有** extension-family 证据指针 —— 它把 canon 的 C2 对象（IME 2484 / scalar <300）**整段替换**成 decode-format front-door law。这不是"稀释"，是"顶替"。修复时 C2 行的**主证据必须换回** `T2-ledger-anchor.md`（IME 2484）+ canon [C2-1]，并诚实标注 `曲线=缺失（1/≥3 点，待 X-SCALAR）`。

### 2B. C1 措辞被 decode 构造协议部分冒占（dilution，非塌陷）

| 文件 | 冒占字样 | 说明 |
|---|---|---|
| `docs/reports/2026-07-09-C1-C2-construction-evidence.md` L8 + Part A 标题 | `C1 = 合取存在性 → 可复制协议(the front-door construction protocol)`；`Part A — C1: The Front-Door Construction Protocol` | 把 **abstract op→typed region 的 decode-body 构造协议** 等同于 C1 的"合取（编译期变体生成 ∧ 运行期 fail-closed 守卫）可复制协议"。二者是**不同机制**：前者是 RVV 内造 kernel body（属 C3′/覆盖构造），后者是 capability schema 跨 family 复用 + 外部贡献者接入协议。 |
| `docs/reports/2026-07-10-paper-evidence-index.md` L31 | C1 **首行**主张 = `C1 协议 = abstract op → typed region → construction-from-abstract（4 decode 家族 × 3 op 谱同构）` | decode 构造协议被摆成 C1 lead。 |

### 2C. **正确 C2 对象（勿动，反而应被 paper-index 引用）**

| 文件 | 内容 | 判定 |
|---|---|---|
| `experiments/active/visibility/T2-ledger-anchor.md` | 复算 IME 家族 raw wc-l **2484** = [LED-1] 首点；引 执行总纲v2 L239/241；诚实注 test_LOC 单列、cloc-approx | ✅ **这才是 C2 的真证据对象**（extension family），且诚实标只有 IME |
| `experiments/active/result-tables/T2_C2_ledger_marginal_cost.csv` **seq 0** | `[LED-1]-family-anchor, IME-family-source, raw_wc_l=2484` | ✅ 真 C2 首点；改 CSV 时**保留 seq 0，剥离 seq 1-13** |
| canon `[C2-1]/[LED-1..3]` | IME 2484 → scalar <300 → zvfh <100，诚实标缺失 | ✅ 权威口径（勿改 canon，除非用户裁定扩义） |

---

## 3. C1 锚点评估

**结论：核心锚稳，未塌陷；"可复制协议"措辞层被 RVV 内部 decode 构造协议部分冒占，是措辞 dilution，需正名。**

- **仍稳（真 C1 锚在位）**：
  - canon `[C1-1]`（科研目标总纲 L145）：一份带关系的 capability schema **同时驱动编译期变体生成 ∧ fail-closed 运行期调度守卫，跨计算范式（SIMD→整矩阵 MAC）× 跨独立家族（向量缺席标量家族）原封复用；接入由外部贡献者按文档可复制**。
  - `[C1-3]` 证据阶梯：家族 #1 RVV / #2 IME（跨范式，已有）→ #3 X-SCALAR。
  - paper-evidence-index **L33**：跨范式 × 跨独立家族同 schema 复用 → N1/N2 bridge + IME N2 已证（memory `[[k1-ime-n2-hardware-candidate]]`，commit `2eeabff9`）。这条**是**真 C1 锚，未丢。
- **稀释处（需正名）**：C1-C2-construction-evidence Part A + paper-index L31 把 **front-door decode-body 构造协议**（4 decode 家族 × 3 op 谱）当作 C1 的"可复制协议"lead。该协议的正当归属 = **C3′（能力键控模式库的构造）/ 成熟编译器覆盖轴**，不是 C1 的"合取存在性"。C1 的"合取"= 编译期生成 ∧ 运行期守卫**同一 schema**；C1 的"协议"= **新 extension family 的外部接入**（[C1-1] 末句），**不是** decode kernel 的构造配方。
- **净判**：C1 未被"冒充稀释到失锚"——真锚（schema 跨独立 family 复用 + IME 跨范式）依然是 load-bearing 证据；但需把"可复制协议"从"decode 构造配方"正名回"extension-family 接入协议"，避免审稿人把 C1 读成"你们把很多量化格式接进了同一个 RVV 前门"（那是 C3′/覆盖，不是合取存在性）。

---

## 4. 严格三分混淆点（decode / kernel / extension family 全文哪里混）

**根因 = 一个词"家族"跨三义漂移**。canon 的"逐家族"= extension family；ledger 的"家族"= decode family（有时 = loop-shape/kernel primitive family）。同一"逐家族接入成本"措辞，canon 指后端接入、ledger 指格式接入 —— 这是整个误挂得以发生的语义枢纽。

| 混淆点 | 位置 | decode / kernel / extension 哪两者被混 |
|---|---|---|
| a | `C2_marginal_cost_ledger.md` L1/L3/L25「block-quant 家族 / K-quant 家族 / 逐格记 C2」 | **decode family** 冒充 canon 的 **extension family** |
| b | `frontdoor NOTES/MANIFEST`「second decode family」「per-family front-door-ization」= C2 | **decode family** 冒充 **extension family** |
| c | `T2_C2_ledger_marginal_cost.csv` `family_spectrum` 列 | 一列里混装 **extension**（`IME-family-source`）+ **kernel/loop-shape**（`flat(single-block)` / `super-block` / `2nd-spectrum: repack GEVM`）+ **decode**（各 q*/iq* 格式）三义 |
| d | `C1-C2-construction-evidence.md` §A.3 L84-86「跨家族成员：同一家族内首格建框架、次格 facts 复用」 | 用 **decode family** 复用论证 C1 的"跨独立家族（extension）复用"可复制性 |
| e | `C1-C2-construction-evidence.md` §A.2 L49「4 个 matmul decode 家族」+ §A.3「跨 op 谱 contraction/dequant/quantize」 | **decode family**（4 支）与 **kernel/op 谱**（3 op）并列陈述，均被并入 C1「合取存在性」叙事，与 **extension family** 的合取轴混同 |
| f | `paper-evidence-index.md` L31（C1）/L8+L41-43（C2） | C1 lead = decode 构造协议；C2 对象 = decode front-door law —— **两条贡献同时被 decode-format 轴顶替**，extension family 只在 C1 L33 残留、在 C2 完全缺席 |

**三分应恢复为**：
- **decode family**（q4_K/q5_0/ternary/codebook/IQ-grid…量化格式）→ 前门化构造/覆盖 = **C3′ 模式库构造 + 成熟编译器覆盖轴**；其 LOC 摊销律是**构造经济学**，不是泛化代价律。
- **kernel/op family**（contraction/dequantize_row/quantize_row 三 op 谱 + flat/super-block/grid/repack loop-shape primitive）→ 同属 **C3′/覆盖构造轴**（op 谱同构是"可复制构造配方"证据，归 C3′/成熟编译器）。
- **extension family**（RVV/IME/scalar/zvfh 后端）→ **唯一的 C2 分母**（[F-6] independent，闭包 ∩ rvv.*=∅）；锚 IME 2484，曲线待 X-SCALAR。

---

## 5. 建议改法（供用户确认后一次改，禁双改；数值全留、只改归属）

1. **C2 正名**：
   - #1-#6 的 decode-format ledger 顶部 framing 从"C2 边际成本律"改注"**C3′ 模式库构造经济学 / 成熟编译器覆盖轴**"（保留全部 LOC 数与摊销律，只改贡献标签与"家族"→"decode 格式/op 谱"用词）。
   - `T2_C2_ledger_marginal_cost.csv` 拆分：seq 0（IME 2484）留在 C2 anchor；seq 1-13 迁到 C3′ 构造 ledger（或明确重标 `axis=C3'-construction`）。
   - `paper-evidence-index.md` **C2 行**主证据换回 `T2-ledger-anchor.md`（IME 2484）+ canon [C2-1]，诚实标 `曲线=缺失（1/≥3 点，待 [X-SCALAR]）`；decode ledger 指针移到 C3′/成熟编译器行。L8 措辞改回 "extension-family 接入的边际成本律"。
2. **C1 正名**：C1-C2-construction-evidence Part A 与 paper-index L31 把"front-door construction protocol"重标为 **C3′/覆盖构造** 素材；C1 的"可复制协议"回锚到 [C1-1] 的 **capability schema 跨独立 extension family 复用 + 外部贡献者接入协议**（保留 IME N2 跨范式证据为 C1 主锚）。
3. **不动 canon**：科研目标总纲 [C2-1]/[C1-1] 与 执行总纲 [LED-1..3]/[C2] **无需改**（它们本就正确且诚实标缺失）。**除非**用户裁定要把 C2 正式**扩义**为"含 decode-format 构造泛化的第二层"——那属 canon/红线级变更，须走"必问"、单独裁决，并同步改 [C2-1] 措辞、撤下 extension-family-only 框架；本核查**不建议**此路（与用户给的权威定义相悖）。

---

## 6. 证据指针速查

- 真 C2 对象：`docs/canon/TianChen-RV_科研目标总纲v2.md` L152-158/L181-183；`docs/canon/TianChen-RV_执行总纲v2.md` L93-95/L279；`experiments/active/visibility/T2-ledger-anchor.md`；`T2_C2_ledger_marginal_cost.csv` seq 0。
- 误挂 C2：`docs/method/C2_marginal_cost_ledger.md`（全）；`experiments/active/frontdoor-framework/{MANIFEST.md,NOTES.md,frontdoor_framework_ledger.csv}`；`T2_C2_ledger_marginal_cost.csv` seq 1-13；`docs/reports/2026-07-09-C1-C2-construction-evidence.md` Part B（L9,L129-238）；`docs/reports/2026-07-10-paper-evidence-index.md` L8,L41-43。
- C1 稀释：`docs/reports/2026-07-09-C1-C2-construction-evidence.md` L8 + Part A；`docs/reports/2026-07-10-paper-evidence-index.md` L31。
- C1 真锚（在位）：`科研目标总纲v2` [C1-1] L145;[C1-3] L147;`paper-evidence-index` L33；N1/N2 bridge `.trellis/spec/index.md` L41；IME N2 commit `2eeabff9`。
- 三分判据：`.trellis/spec/architecture/core-invariants.md` [F-6] L84-86（independent family 闭包 ∩ rvv.*=∅）/ [L-2]。

*本文件为纯只读核查报告，未改任何 canon/inventory/spec/ledger/论文提纲，未 git。*
</content>
</invoke>
