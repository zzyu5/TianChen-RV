# PRD · A 线阶段1 · g 轴平 K-quant 6 处格式烘焙 → 已有 getter/派生（裁决7·zero-schema·byte-exact·公式+6）

> **权威** = 用户裁决7（A 线现行法）+ A 线 scope plan 阶段1（`.trellis/tasks/archive/2026-07/07-18-a-scope-g-c-scope-7-kquant-g-19-bql-18-deferreddequant-plan/research/`·**先读 `census-g-axis-kquant-19.md`（19 逐处·阶段1 = 平 K-quant 6）+ `debake-staged-plan.md`（阶段1）**）。
> **性质** = A 线接层（g 轴格式烘焙 → 格式描述符/getter 读·公式占比 18→24）。阶段0（c 轴 BQL 18·`cc805b121`·**已 confirm CONFIRMED**·byte-exact regen-diff 0）证接层范式可行·本阶段续。

## 一、核心（研究已定·zero-schema）
KQuant 发射器 g 轴（**平 K-quant** = q2_K/q3_K/q4_K/q5_K/q6_K 等非-grid）有 **6 处格式字面量/分支**（census-g-axis §阶段1 有行号）。K-quant 已有 `I64Attr` getter + 派生量（`subBlock` 等）——**换成 `coreOp.getXxx()` / 从 `subBlock` 派生**（zero-schema·getter 已在·骨架不推倒·对齐阶段0 fail-closed 范式）。

## 二、★caveat（研究关键·implement 首步必验）
plan 明标：**阶段1 的 #5/#6（`weightDOffset=80` / `weightDminOffset=82` → 已有 getter）取决于前门是否已 stamp 80/82**。
- **implement 首步：读前门（`RVVLowerQuantContraction.cpp` + 单体前门）确认是否 stamp 80/82**。
- 若**已 stamp** → #5/#6 走已有 getter（本阶段·公式+6）。
- 若**未 stamp** → #5/#6 **并入阶段2**（本阶段只做剩 4 处·公式 +4·如实缩 scope·报 main）·不擅加 stamp（加 stamp = 涉前门/可能 schema = 回门）。

## 三、你要做
1. **首步验 caveat**（前门 stamp 80/82？）→ 定本阶段 scope（6 或 4 处）。
2. **逐处**（6 或 4）：g 轴格式字面量/分支 → getter/派生读（对齐 DeferredDequant「格式常数从参数孔/描述符读·0 裸字面量」+ 阶段0 fail-closed 范式：缺 → `notifyMatchFailure` 非静默默认）。
3. **byte-exact regen-diff**（阶段0 范式·最强）：forced clean rebuild weft-opt → regen K-quant 覆盖格 kernels（前门生产路 × VLEN128/256）→ **前后 0 diff**。若有 direct-input golden fixture 依赖删除的默认（同阶段0 的 58 fixture 情形）→ 补显式 stamp（emitc 0-diff·byte-exact·报 main 触碰集变动）。
4. **VLEN 翻转矩阵**（rvv/k1 march 各 emit·格式读对）·优先 emit-diff·避板争用。

## 四、验收
1. **6（或 4·caveat 定）处全换 getter/派生**（grep 证格式字面量残留清零·对齐范式）。
2. **byte-exact regen-diff 0**（前门生产路 + 任何 direct-input fixture·前后 0 diff = 接层无行为变化）。
3. **zero-schema**（ODS 未改·grep 证·无回门——**若发现须动 schema/前门 stamp·停·报 main** 回门点）。
4. **[F-EMIT] 棘轮不退 + I7 verifier 不放松 + lit 绿**（新失败=0·pre-existing 除外·同阶段0 的 3 Scripts pre-existing）。
5. **0 造数**·byte-exact 硬门·**worktree 模式·禁 commit·禁 add**（main cherry-pick）·sealed/master 不动·证据 scratchpad·数字报 main。

## 五、触碰集 / 遗留
- 触碰：`lib/Conversion/RVV/RVVToEmitCKQuant.cpp`（g 轴 6·平 K-quant）+ 可能 direct-input golden fixture（同阶段0 情形·补 stamp·byte-exact）+ 新/改 lit。**worktree 隔离**（与并行 iq2/iq4 worktree + 主线零冲突·byte-exact 走 regen-diff 无需板·或 rvv 避 k1 争用）。**worktree commit 到分支保存（防丢）·报 hash 给 main cherry-pick。**
- **遗留**：阶段2（g 轴 iq-grid/codebook 13·**回门·新 schema 路 A/B 待 supervisor**）· 阶段3（相邻同轴 8）。#5/#6 若 caveat 未 stamp 则并入阶段2。
