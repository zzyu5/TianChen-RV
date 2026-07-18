# PRD · A 线阶段3 · 相邻同轴 8（BQL 焊死 5 + KQuant value_or 3）去烘焙（裁决7·byte-exact）

> **权威** = 用户裁决7（A 线现行法）+ A 线 scope plan 阶段3（`.trellis/tasks/archive/2026-07/07-18-a-scope-g-c-scope-7-kquant-g-19-bql-18-deferreddequant-plan/research/debake-staged-plan.md` §阶段3·`census-c-axis-bql-18.md` + `census-g-axis-kquant-19.md` 的相邻同轴项）。
> **性质** = A 线接层（相邻同轴·**阶段0/1/2 范式续**·byte-exact）。阶段0（c18）+阶段1（g2）+阶段2（g15）已落且独立 confirm·公式 35/37。本阶段 = **相邻同轴 8**（确凿 37 之外的同轴烘焙·进一步做大公式占比）。

## 一、范围（相邻同轴 8·plan §阶段3）
- **BQL 焊死 5**：`RVVToEmitCBlockQuantLinear.cpp` 内 5 处板值/宽度**焊死字面量**（非 18 处 value_or·是硬编码常量·相邻同轴·census 有行号）。
- **KQuant value_or 3**：`RVVToEmitCKQuant.cpp` 内 3 处 `value_or`（相邻·非 g19 的 grid/codebook·census 有行号）。
- **先读 plan §阶段3 + census 定位逐处**（机算行号·git grep 可复跑）。

## 二、★硬条件（承阶段0/1/2·用户红线）
1. **缺席硬失败·严禁默认值**（同阶段2 最硬红线）：换 getter/描述符读时·缺席 → `notifyMatchFailure` 点名·**严禁 value_or/默认值兜底**。焊死字面量若是**结构常量·非决策**（如 ISSUE-033④ 的 iq3 几何固定·无 IR 宽度可读）→ **不强加假旋钮**（[K-10] 违例·维持字面量 + 显式结构标注·报 main）·**只去真烘焙决策**。
2. **byte-exact regen-diff 0**（接层·前后 0 diff·direct-input fixture 补 stamp 若需·同阶段2）。
3. **zero-schema 优先**（相邻同轴多可从已有 getter/IR 读·若须新 schema→停报 main·回门）。

## 三、验收
1. **逐处判真烘焙 vs 结构常量**（真决策烘焙→去；结构常量→维持+标注·报 main·别造假旋钮）。
2. **真烘焙处换 fail-closed 读·0 默认 fallback**（红线·grep 证）。
3. **byte-exact regen-diff 0**（前后 0 diff）。
4. **[F-EMIT] 棘轮不退·I7 不放松·lit 绿**（新失败=0·pre-existing 除外）。
5. **zero-schema**（无新 ODS·若须→停报 main）·**0 造数**·**worktree·禁 commit·禁 add**（main cherry-pick·worktree commit 保存报 hash·**git add 只纳源文件·勿纳 build 目录**）·sealed/master 不动·数字报 main。

## 四、触碰集 / 遗留
- 触碰：`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp`（BQL 5）+ `lib/Conversion/RVV/RVVToEmitCKQuant.cpp`（KQuant 3）+ 可能 fixture。**worktree 隔离**（byte-exact 走 regen-diff 无需板·或 rvv 避主线争用）。
- **遗留**：结构常量处（如 iq3 几何）维持字面量待判据级裁（ISSUE-033④ 已裁维持现状）。A 线确凿 37 已 35/37（#5/#6 fold-brick op ISSUE-115 待裁）·本阶段是相邻扩。
