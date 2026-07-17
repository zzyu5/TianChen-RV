# Core Invariants（本文件 = 指针 · 条文已归位 canon）

> **`ISSUE-070` 已裁（2026-07-17 · 用户裁）**：I1–I9 的**唯一权威本** = [canon · 核心不变量](../canon/核心不变量.md)。
> 本文件此前与之**双本逐字并存**，且正文夹带 commit 号 / 署名日期式裁决记述 / 战役事故叙事（违「spec 只写现行法」）。
> **原全文（含历史外壳）已按归档机制移入 `_attic/`（git-ignored，见 `_attic/ATTIC_INDEX.md`）；本路径保留为指针**，
> 因 `schema/capability.schema.v1.json` 的 `$meta.authority` 钉住本文件的 `[S-5]` / `[F-2′]` 条目（v1.2.0 sealed）。
>
> **本文件不再承载任何条文正文**（禁副本）。以下只给去向。

## I1–I9 — 全项目复用硬规则

**全部正文住 [canon · 核心不变量](../canon/核心不变量.md)**（唯一权威本）：

`I1` Capability 是第一系统对象 · `I2` `weft.exec` 只是 execution envelope · `I3` 零 family-name 分支 ·
`I4` Metadata 是 mirror 不是 authority · `I5` 可执行事实必须结构化在 typed body 里 ·
`I6` Primary stack 是 C++/MLIR，Python 只做 tooling · `I7` Fail closed，不要从 metadata 合成 ·
`I8` 硬件主张需要真实证据 · `I9` 实验验证结构，不反向定义结构。

引用一律写 `核心不变量 I#` + 相对链接，**不重抄条文**。

## 附加硬规则 — 各自的 canon 家

本文件曾重抄 [L-6]/[L-8]/[K-4]/[S-5]/[F-2′]/[F-6]/[SEL-2]/[NG-3]/[NG-4]/[K-10]/[VERIFY-LADDER] 与「对手类机判」的正文。
**每条在 canon 层均有唯一的家**，**去向表住 [canon · 核心不变量](../canon/核心不变量.md)**（本处不再重列，一条规则两处写 = 缺陷）。

### schema `$meta.authority` 所钉两条的去向

| 条目 | 法条正本 | 字段级 / 操作门全文 |
|---|---|---|
| `[S-5]` schema.def 是声明工件（shape 恰含六项） | [canon · 能力模型与插件协议](../canon/能力模型与插件协议.md) `[S-5]` | 六项**字段级细节**住 [architecture · 能力模型](./能力模型.md) `[S-5]` |
| `[F-2′]` 家族接入操作门（diff ∩ `schema.def` = ∅） | [canon · 能力模型与插件协议](../canon/能力模型与插件协议.md) `[S-6]` 两级门 | 同址（操作门 = 逐 PR；报告门 = 逐版本） |

> 上述两条正本 **已在 `$meta.authority` 之内**（v1.2.0 一并锚定 `canon/能力模型与插件协议.md` 与 `architecture/能力模型.md`），
> 故本指针**不改 schema、不触 `schema.def` shape、无需 [S-6] bump**。
