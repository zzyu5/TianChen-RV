# 事故档案 — 防复发资产（原样保存 · 不去重 · 不改写）

> **设立依据**：大重构总令 §4.2.7「`docs/` 内容迁入六 spec 后整体归档；**事故档案原样迁入 Trellis 事故区**（防复发资产，**不去重不改写**）」。
> **建立日期**：2026-07-17 · 51 份（50 报告 + 1 随附验证脚本）。

---

## 一、为什么在 `.trellis/` 下、而不在 `.trellis/spec/` 下

**选址 = `.trellis/事故档案/`（与 `spec/` / `tasks/` / `scripts/` 并列），不进 `spec/`。** 三条理由：

1. **§二 铁律禁止**：「spec 只写**现行法**，零历史叙事、零编号考古——**教训一律压缩成规则本身**」。
   事故档案的本体**就是**历史叙事 + 编号考古（M2c/M1/M4 三次反转、FU-1/2/3、前科 1–10、WRONG PREDICTION #1/#2/#3），
   且令文要求**原样·不改写** ⟹ 放进 `spec/` 与该层宪法正面冲突。
2. **§二 不增份数**：「六份 SPEC……此后**只增内容不增份数**」。在 `spec/` 下新开第七个目录 = 增份数。
3. **令文原文说的是「Trellis 事故区」，不是「spec 事故区」** —— `.trellis/` 内、六 spec 外，两句都满足。

**分工（勿混）**：
- **规则**住 [canon](../spec/canon/index.md) / [governance](../spec/governance/index.md) —— 教训压缩成的**现行法条**（§二 铁律要求）。
- **案卷**住本区 —— 规则背后那起**具体的错**，供人认出同型再犯。
- **问题**住 [issues](../spec/issues/index.md) —— 唯一问题登记簿。
- 本区 **不是** 问题登记簿，**不产生**新法条，**不得**被引用为现行法。引用现行法一律引 canon。

---

## 二、判据（收录标准 · 四条全中）

1. **自指** —— 出错的是**我方**（我方的主张 / 诊断 / 数字 / 流程）。
   ★**排除项**：关于硬件 / 世界的**假说被实测证伪 = 正常科学，不是事故**
   （如「小 margin 必 wash」被证伪、W4 measured-negative、曳光弹 null）。
2. **翻转** —— 有明确的 before→after：某个曾被相信的我方主张被**撤回 / 推翻 / 改判 / 纠正**。
3. **防复发残留** —— 文中给出能在将来抓住同类错误的**规则 / 判别键 / 机制**。
4. **体裁测试** —— **文档的主题即「我方的错」本身**：去掉事故内容后该文**失去存在理由**。

★ 第 4 条是关键收窄。本项目的反错文化使「证伪 / 反面教材 / 空心 / 打脸 / 前科」等词**遍布全树**，
但**引用教条 ≠ 自己出了事故**。故**法条 / 模板 / 台账 / 索引 / 地图 / 登记册 / 绿格登记**
即便**内嵌**教训也**不收**——它们的教训按 §二 铁律**已压缩进 canon**，本体随 `docs/` 归档。

**被第 4 条挡下的典型**（判 NO 的理由，防后人当遗漏重查）：
`docs/canon/*`（法条·现行法已迁六层）· `docs/files (2)/*`（v3 canon 家族）·
`T3/T6/P4 报告模板`· `paper-material-inventory`（清单）· `REPOSITORY-MAP`（地图）·
`FALSIFIER-INDEX`（索引）· `PENDING_RULINGS` / `SEALED-WIN-REGISTRY`（登记册）·
`ROADMAP.md`（队列流水账·§一.1 已明令退役）·
**banner 型**（翻转仅由**事后追加的订正 banner** 承载、非本文调查所得：`战略-recon-下一战役` / `母program-pillar-对账`）。

---

## 三、事故簇（同一起事故的多份卷宗**不去重**——令文明令）

| 簇 | 事故 | 卷宗 |
|---|---|---|
| **① min-term 误诊** | M2c 判 q4_K/q5_K repack-GEMM「5% latent bug」→ M1 ZERO-MODEL 翻案 → M4 终审推翻为 **cert-harness 激活量化失配伪影**（3 次反转） | `2026-07-09-minterm-fold-audit` · `2026-07-09-zero-model-adjudication-method` · `2026-07-09-ggml-q8-quant-dual-path-memo` |
| **② 编译器不对称** | 一批 kernel 赢实为「我方 clang 编 vs 对手出货 gcc 编」；rvv 4 承重格**撤回-EXECUTED**（1.884→0.272×） | `2026-07-10-CASE-COMPILER-ASYMMETRY-casefile` · `2026-07-10-q4k-micro-vs-e2e-regression-mechanism` · `2026-07-09-q4k-8gate-status` |
| **③ micro↛e2e / 部署变体≠证过变体** | 1.884× = **热缓存 artifact**；集成树部署了 vl=16 而非证过的 vl=8（PPL 822057） | `2026-07-10-CASE-MICRO-E2E-casefile` · `2026-07-10-rvv-e2e-m0-attribution` · `2026-07-10-t4b-e2e-seal-integration-proven-kernel-variant-residual` |
| **④ C2 对象 / 成本律误挂** | C2 证据对象被 decode-format 顶替；「统一 <300」是 integrated 子扩展预估被误挂到 independent 家族 | `2026-07-10-C1-C2-evidence-object-audit` · `2026-07-11-X-SCALAR-AUDIT后续-排期报告` · `2026-07-12-C2-LED-2-边际曲线-三家族点` |
| **⑤ 空心门 / 空心检查** | 门验的是**标签**不是**那个东西**：PR-17 gcc 门恒真、5 份 col-outer fixture 属性翻转仍全绿、iq1_s 悬空 grid 引用 lit 仍绿 | `2026-07-17-AUDIT-gcc-lane` · `2026-07-17-C1-looporder全格消费` · `2026-07-17-C4a2-iq1s落地` · `2026-07-16-E4` 相邻 |
| **⑥ 注释里的投机断言**（C4a 链·四份连续） | 三次错误预测（`Ksigns128` / `I32x4` / `DUAL ls`）**全在 enum 的注释里**，规则只看 enum ⟹ 投机搬进规则不看的地方活了两轮 | `2026-07-17-C4a-grid参数化` · `C4a2-iq1s落地` · `C4a3-iq1m-iq3xxs` · `C4a4-iq3xxs` · `C4a5-iq3s`（归纳表在 C4a5 §末） |
| **⑦ 转抄链 / 造数** | 一个数三方重建出三个值（1237/1256/1198）= 该数无钉死推导路径；E7 立项前提本身 STALE（逐份转抄无人复核） | `2026-07-16-E5-T1d-双实例演示` · `2026-07-16-E7-GAPA-第二注册点` · `2026-07-16-S2-rv64gc措辞协调` · `2026-07-16-C7-模式计数归一与扇出台账` |
| **⑧ MIRAGE 假快数** | 翻 gate 得 8.69× = 破损上游 vl 钳位跳半列做少工，output = garbage | `2026-07-12-G5-接线机制方法学-SOP` · `2026-07-12-perf-covered-q8_0-green-3of84` |
| **⑨ 幻觉污染 brief / 有界窗口断言** | 「我的 brief 成了污染源」7 条自造幻觉；「某物不存在」用 `head`/固定窗口命令作依据连造假事实 | `2026-07-17-收束-推进与理解` · `2026-07-17-结账` |
| **⑩ 并行线交织 / 全局 stash** | `git stash` 是全局树操作，卷走 sibling 线未提交改动；**文件集不相交也救不了** | `2026-07-06-并行线纪律-worktree-与触碰集` · `parallel-build-and-baseline` · `travel-decision-ledger` |

> **★ 同一起事故有多份卷宗是设计而非冗余**（令文：**不去重**）。
> 已知重叠：`AUDIT-gcc-lane`（全量审计）与 `收束-推进与理解 §5.2`（摘要）记同一起；
> `结账 §4.1` 与 `收束 §七` 记同一起「幽灵 binary」。**两份都留。**

---

## 四、内部指针的已知状态（**刻意未改** · 防后人当遗漏重查）

本区文件**按令原样移入**（`不去重不改写`）⟹ 其**正文内的相对指针仍写作 `docs/reports/...` 等旧路径**，
其中多数目标现已在 `_attic/docs/`（git-ignored）或本区。**这些是刻意未改的**：

- **改写正文 = 违令**（§4.2.7「原样……不改写」）。
- 判读旧指针：`docs/reports/<名>.md` → 若 `<名>` 在本区即读本区同名文件；否则读 `_attic/docs/reports/<名>.md`。
- **全仓活消费者（schema / 门 / 代码 / spec）的反向指针已改锚到本区新址**，故本区**对外可达、零悬空**；
  未改的只有本区**文件内部**的自述指针。

---

## 五、清单（51 份 · 原路径 → 本区）

全部来自 `docs/reports/`（除三份 `docs/method/`），文件名未变：

**`docs/method/` 三份**：`方法学-kernel微基准两大隐形混淆.md` · `parallel-build-and-baseline.md` · `perf-characterizations-layer4.md`

**`docs/reports/` 48 份**：见本目录 `ls`。随附 `2026-07-17-C1-looporder-verify.sh`（`C1-looporder全格消费.md` 的验证脚本，随卷宗同行）。
