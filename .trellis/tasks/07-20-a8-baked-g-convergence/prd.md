# A8：发射器 baked-g 全量收敛

## Goal

承接 R5.1 的去烘焙路线，对 ISSUE-118/119 所代表的 emitter 内 per-format geometry/layout 常量做 HEAD 全量复核与分型，并把真实格式/机制事实迁入 typed g owner，使 formula/plan/body/emitter 真实消费；结构常量保留为结构，不制造伪 attr/θ。每个 declared slice 全 caller cutover 后删除旧 format switch、字面量与 silent default。

## Classification Before Editing

每个发现只能进入以下之一：

| 类别 | 判据 | 动作 |
|---|---|---|
| typed g | 随格式/布局/机制变化，且 consumer 需要该值 | 在 plugin-local typed fact/plan/body 中唯一表达并真消费 |
| derived | 可由更基本 typed g 确定性派生 | 只保留派生函数，不增加重复 attr |
| structural constant | 由 op/mechanism 定义固定，不存在合法变化 | 保留并写清 invariant，不制造假旋钮 |
| dead/wrong | 无 caller、零消费、错误默认或已被新 authority 取代 | 删除 |
| unresolved | 证据不足或会改变语义 | task 保持未完成，登记 issue；不得先加 optional compat 字段 |

## Declared Slices

### Slice 1：GridCodebook / Ternary geometry

- 复核 ISSUE-118 当前列举的 sub-block/group/plane/chunk/strip 等几何是否仍存在且属于 typed g。
- 优先复用已有 typed attrs/plan 字段；禁止同义字段、JSON-only authority 或 emitter-local default。
- 所有 producer/front-door 与 fixtures 同步迁移，missing 字段在 active path fail-closed。

### Slice 2：ForwardElementwise dequant-row layout

- 复核 ISSUE-119 的 block stride、quant/scale/qh offsets、qk 等 per-format layout 事实。
- 把真正的 g 放入 typed descriptor/plan/body，emitter 只读 selected body；删除 emitter 内按 format 名重建布局的路径。
- 对共同布局优先用共享 typed fact/派生，不把约四十处历史字面量机械翻译成四十个 attr。

## Design Constraints

- 依赖 A1 authority matrix；字段/API 形状服从 A2 最小 contract，避免另建“描述符万能对象”。
- optional ODS 只可用于语法上不可避免的解析阶段；active production path 不得以 `value_or` 或旧 emitter 兜底。所有 in-tree producer/fixture 必须在 slice 合入时同步迁移。
- g 与 c 分 owner；本任务不把板值、VLEN 或 measured winner 混进格式描述符。
- emitter 不得读取 board、march、format winner 或 measurement 再决定几何。
- 正常输入保持 byte-exact/ULP；如果真实 g 修复揭露既有错误，拆出具名 correctness disposition，不用默认值掩盖。

## Dependencies and Parallelism

- 依赖 A1；API/owner 施工依赖 A2 contract 稳定。
- HEAD census 和分类可在 A1 同步准备，但生产字段迁移不得先于 authority 裁决。
- 与 A3/A4 共享 plan/emitter 文件时串行；与 B1 可并行。

## Acceptance Criteria

- [ ] HEAD census 覆盖 ISSUE-118/119 当前实际符号，零未分类；每项均有 typed-g/derived/structural/dead/unresolved 分类与证据。
- [ ] 两个 declared slice 的 typed g 均有唯一 producer、consumer 和 missing/illegal 诊断。
- [ ] 至少一个格式事实 mutation 能改变 plan/body/emission；删除 consumer 的 mutation 能使测试变红。
- [ ] emitter 内对应 per-format layout/geometry 重算、format switch、旧字面量和 code-affecting default 为 0。
- [ ] 全部 production producer/caller 与 in-tree fixtures 原子迁移；无 optional compat mirror、deprecated alias 或 old→new bridge。
- [ ] structural constants 明确保留理由，未被包装成虚假的 g/c/θ。
- [ ] g/c owner 仍分离，measurement 不进入 candidate/geometry 构造。
- [ ] byte-exact/ULP、focused lit、forced relink/clean rebuild 与 deployed regression 无新增失败。
- [ ] formula migration ledger 与 ISSUE-118/119 状态按真实结果更新，不沿用约数冒充 HEAD 事实。
- [ ] ISSUE-118/119 closure matrix 逐子项绑定代码、测试、schema/ledger 与 owner；仍 unresolved 的子项保持 issue 开放，不以代表性 slice 代替全量关闭。

## Verification

- complete symbol/literal/format-switch census；
- decisive and missing-field lit/FileCheck；
- emitted-C byte-exact/ULP comparison；
- mutation/negative tests；
- clean rebuild for ODS/shared layout changes；
- relevant RVV suite and `git diff --check`。

## Out of Scope

- 为所有结构常量建 attrs；
- 一次迁移全部 RVV decision；
- 改性能 winner、跑正式板测或建立 runtime profile；
- 新通用 descriptor/Formula dialect。

## Issue Mapping

- ISSUE-118：GridCodebook/Ternary typed geometry；
- ISSUE-119：ForwardElementwise dequant-row per-format layout；
- ISSUE-117：公式层模块化总债；
- RET-1：无兼容中间态、原子 cutover。
