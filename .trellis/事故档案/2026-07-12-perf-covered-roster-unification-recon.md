# perf-covered 口径统一 + fold-regime recon — 6/83 机检落定（supersede 零未定义格分类）

> 生成 2026-07-12 · 触碰集 = docs/tooling 域（只读 roster/coverage 脚本 + 引用既有分类报告，写本报告 + `schema/perf-covered-category.v1.json` + `.trellis/scripts/perf_covered_metrics.py` + CI job）。**未跑板·未 git·未改 ODS/lib/·未改 `docs/ROADMAP.md`/结构 roster `coverage-roster.v1.json`/`coverage_metrics.py`。**
> **性质**：用户 2026-07-12 裁决**一（分类表纠偏·canon 级·先于一切新测量）**的落地。消灭「headline 6/84（按 format）vs 分类合计绿 7（按 roster-cell）同层不同单位」口径打架，统一到 **perf-covered = 6/83**（fold q4_0-gemm regime·按 format 计·any-board），并机检化（recon 脚本 + CI）。
> **supersede**：本报告 supersede `2026-07-12-perf-covered-零未定义格分类.md`（84 格逐格·regime-split 计法·绿7）——**引用不重抄**；逐格判读表以标签工件 `schema/perf-covered-category.v1.json` 为机器权威源。

---

## 0. 一句话结论

**perf-covered = 6/83 = 7.23%**（fold q4_0-gemm regime·any-board·裁四.0）。六类分类 `{绿6 · 黄-物理墙12 · 黄-传导稀释0 · 黄-对手更强12 · 黄-未接线26 · 声明例外27}`·**Σ=83 = denom·零未定义格**·`reconciliation_ok=True`·`three_source_consistent=True`（6==6==6）·`anti_gate_ok=True`。

**备选单位 = 7/84**（`--no-fold`·regime 保留·Σ=84·同样 reconcile）·**flagged 待用户复核**（结构 roster/M4 84/91 用此单位·见 §5）。

**M4 不动声明**：结构 roster `coverage-roster.v1.json`（93 键·q4_0-gemm 按 regime 拆 decode+prefill = 唯一 regime 拆分·对构造-identity 合法）+ `coverage_metrics.py`（84 certified / 91 denom / 2 域外·M4 92.31%）**一字未动**。perf-covered fold **仅本层**，不回改结构轴。

---

## 1. fold 决策（固定输入·消灭口径打架）

**口径打架（裁一.1 要消灭）**：旧分类册 headline **6/84**（按 format·q4_0-gemm 折叠）vs 分类合计 **绿7**（按 roster-cell·q4_0-gemm 拆 2）——**同层不同单位**（format vs roster-cell）。

**fold 决策（用户已定·非本报告选择）**：**perf-covered 层 fold regime（按 format 计·禁单格特例拆分）**。结构 roster/M4 不动。
- fold q4_0-gemm decode+prefill → 单 format-cell（绿 7→6·Σ 84→83）。
- 一.2 IME 重分类（传导稀释 3 → 0·未接线 23 → 26）。
- **any-board 判绿**（裁四.0：格级任一板 ≥parity 即绿·FLAT 复刻 k1 不增头条）。
- **可逆**：recon 脚本带 `fold_regime` flag（默认 True=6/83）·`--no-fold` 出备选 7/84。

**⇒ 目标数**：`perf-covered = 6/83 · {绿6·物理墙12·传导稀释0·对手更强12·未接线26·声明例外27}·Σ=83`（校验 6+12+0+12+26+27=83 ✓）。**机检落定 = MATCH**（§4）。

---

## 2. 一.2 IME 重分类 delta（传导稀释 3 → 未接线 3·传导稀释类清零）

旧分类册把 `gemm_tile/{q4_0,q8_0,q4_K}@ime` 判 **黄-传导稀释**（compute-account ~2.09× · decode wash）。**裁一.2 纠正为 黄-未接线**：

| 格 | 旧类 | 新类 | 重分类理由（一.2 措辞） |
|---|---|---|---|
| gemm_tile/q4_0@ime | 黄-传导稀释 | **黄-未接线** | bridge 未建·e2e 未测·**传导稀释是实测后判读非预测标签**·compute-account ~2.09×（对手 SELF·silicon-sealed int32 0-diff vmadot 0xe210312b·[GAP-IME-LEAF-PIPELINE] 闭环后）+ Amdahl 预估：厂商 IME 天花板证内存墙 1B decode 无 cleanly-isolated e2e win → decode e2e 上限 < 噪声·M=1 不传导·**bridge 落地实测后按证据转** |
| gemm_tile/q8_0@ime | 黄-传导稀释 | **黄-未接线** | 同上·q8_0 flat IME tile |
| gemm_tile/q4_K@ime | 黄-传导稀释 | **黄-未接线** | 同上·q4_K super-block IME（P7 ime∧shape format-agnostic MMAOp 统一接管三格） |

**⇒ 黄-传导稀释类清零（0）**·黄-未接线 23 → 26。**依据**：传导稀释是"实测 decode wash 后的判读"，而 IME bridge **尚未接线、e2e 未测**——现在只能标"未接线（带 compute-account ~2× + Amdahl 预估指针）"，bridge 落地实测后再按证据转（可能转传导稀释/绿/对手更强）。IME 接线队列位 = **IME（IQ 后·跨范式完整性名义非 perf·裁 2026-07-12 ⑤）**。

---

## 3. 一.3 标签冻结映射（六类冻结集·自造标签停用）

**六类冻结集**（裁一.3·`perf_covered_metrics.py` FROZEN_CATEGORIES）：`{绿 · 黄-未接线 · 黄-传导稀释 · 黄-物理墙 · 黄-对手更强 · 声明例外}`。**自造标签停用·映射入六类**：

| 自造标签（停用） | 映射入 | 格 | provisional |
|---|---|---|---|
| yellow-kernel-axis | **黄-对手更强** | gemm_tile/q6_K@rvv | **true**（pending 二 L-7 反汇编 + L-11 同域标注后转正·曳光弹已 e2e board 实测锚 0.07×·成色 A·根因归 LAW 例1 pending 逐项复核） |

**机检**：`perf_covered_metrics.py` reconciliation `bad_category_cells == []`——任何非六类冻结集的 category（含 yellow-kernel-axis）→ CI 红。q6_K 是唯一 `provisional: true` 格；其余 82 格 `provisional: false`。

---

## 4. recon 脚本机检输出（headline + 分类 + Σ 对账·三处同源·anti-gate）

`python3 .trellis/scripts/perf_covered_metrics.py report`（roster_sha256 `2704968065bd…`·labels_sha256 `f6a0dee53ddd…`）：

```json
{
  "headline": { "perf_covered": "6/83", "pct": 7.23, "green_num": 6,
                "denominator": 83, "any_board": true },
  "classification": { "绿": 6, "黄-未接线": 26, "黄-传导稀释": 0,
                      "黄-物理墙": 12, "黄-对手更强": 12, "声明例外": 27 },
  "classification_sum": 83,
  "reconciliation": {
    "reconciliation_ok": true,
    "sum_equals_denominator": true,
    "undefined_cells": [], "bad_category_cells": [], "incomplete_field_cells": [],
    "orphan_labels": [], "duplicate_labels": [],
    "three_source_consistent": true,
    "three_source": { "headline_green": 6, "classification_green": 6, "ledger_green": 6 }
  },
  "anti_gate": {
    "anti_gate_ok": true, "declared_exception_cells": 27,
    "green_with_exception": 6, "green_without_exception": 6,
    "all_exceptions_certified": true
  },
  "fold_regime": true
}
```

**逐 op × 类别矩阵（fold 后·Σ=83）**：

| op | 分母 | 绿 | 物理墙 | 传导稀释 | 对手更强 | 未接线 | 声明例外 | 行 Σ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemm_tile（rvv 18 fold + ime 3） | 21 | 6 | 0 | 0 | 4 | 11 | 0 | 21 |
| vec_dot | 23 | 0 | 0 | 0 | 8 | 15 | 0 | 23 |
| dequantize_row | 24 | 0 | 5 | 0 | 0 | 0 | 19 | 24 |
| quantize_row | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 3 |
| product_reduce | 3 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| 逐元素/norm | 9 | 0 | 4 | 0 | 0 | 0 | 5 | 9 |
| **列 Σ** | **83** | **6** | **12** | **0** | **12** | **26** | **27** | **83** |

> gemm_tile 行 = 21（rvv q4_0-gemm decode+prefill fold 为 1·rvv 19→18 + ime 3）。绿 6 全在 gemm_tile rvv（q4_0 fold·q4_1·q5_0·q5_1·q8_0·q4_K）。对手更强 gemm 4 = q2/q3/q5/q6_K。未接线 gemm 11 = rvv 8（iq2_xxs/iq2_xs/iq2_s/iq4_nl/iq4_xs/mxfp4/tq1_0/tq2_0）+ ime 3。

**三处同源（reconciliation·镜像 coverage_metrics 的 m4 reconciliation）**：headline 分子（从脚本出）= 分类合计绿 = 登记册 cell 数（带 ledger_pointer 的绿格）= **6==6==6** ✓。

**anti-gate 自证**（详见 `2026-07-12-perf-covered-声明例外台账-过审.md` §2）：声明例外判定与黄格分母无关——分子(含例外)=6 == 分子(移出例外)=6·例外全 certified=True。

---

## 5. 7/84 备选脚注（flagged 待用户复核）+ M4 不动声明

**备选单位 7/84**（`perf_covered_metrics.py report --no-fold`·regime 保留·Σ=84·`reconciliation_ok=True`）：q4_0-gemm decode+prefill 两 regime-cell 各计一次、皆绿 → 绿 7 / denom 84。**flagged 待用户复核**——本层默认口径 = **6/83**（fold·按 format）；7/84 仅供与结构轴对齐时对照，不作 headline。

**M4 不动声明（读死）**：
- 结构 roster `schema/coverage-roster.v1.json` = **93 键**（q4_0-gemm 按 regime 拆 decode+prefill = 唯一 regime 拆分·对构造-identity 合法·`kernel_key` 含 regime）·**未动**。
- `coverage_metrics.py` = **84 certified / 91 denom / 2 域外**（M4 92.31%·blocked_on_IME=0）·**未动**。
- perf-covered fold **仅本层**（`perf_covered_metrics.py` 的 `_fmt_key` 丢 regime）·**不回改结构轴**。两脚本共用同一 roster+sixstate 源（certified 派生 reuse `coverage_metrics`）→ 结构分母是**单一源**、never 独立重派。

---

## 6. 产出清单 + check_docs_canon 自检

**五件产出**：
1. `schema/perf-covered-category.v1.json`（标签工件·83 cell·每 cell category + provisional + 类别专属字段·[COV-1] 体例）。
2. `.trellis/scripts/perf_covered_metrics.py`（recon 脚本·stdlib·`--self-test` 15/15 PASS·`report [--out] [--no-fold]`·[COV-2] 体例）。
3. `docs/reports/2026-07-12-perf-covered-声明例外台账-过审.md`（27 格逐格五字段 + anti-gate + 物理墙 12 格 roofline·裁三）。
4. `docs/reports/2026-07-12-perf-covered-roster-unification-recon.md`（本报告·supersede 旧分类）。
5. `.github/workflows/falsifier-gate.yml` job `perf-covered-recon`（self-test + report 断言 reconciliation_ok·裁三/五）。

- 本报告带 `YYYY-MM-DD-` 前缀（reports/ append-only 合规·非 canon/ 越界）→ check_docs_canon PASS 预期。
- **未改**：`docs/ROADMAP.md`（主会话负责）· 结构 roster · `coverage_metrics.py` · lib/ · ODS · board。**未 git·未跑板。**
- **交叉引用**：`2026-07-12-perf-covered-零未定义格分类.md`（superseded 源）· `2026-07-11-perf-covered-baseline.md`（分母精算）· `2026-07-12-perf-covered-{q8_0,q5_0,q5_1,q4_1}-green-*of84.md` + `2026-07-10-Win-K1-VLEN-加固报告.md`（6 绿登记）· `2026-07-12-perf-covered-q6_K-yellow-kernel-axis.md`（q6_K provisional）· `2026-07-12-三具名GAP修复评估.md`（3 GAP Amdahl<噪声）· `2026-07-11-M4-三分类终态全表.md`（M4 不动）· memory `measurement-offensive-perf-covered` / `m4-closure-three-classification` / `kernel-wins-dont-transplant-to-e2e`。
