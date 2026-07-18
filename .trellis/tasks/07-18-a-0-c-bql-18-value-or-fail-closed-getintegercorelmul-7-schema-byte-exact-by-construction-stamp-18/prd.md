# PRD · A 线阶段0 · c 轴 BQL 18 处 value_or → fail-closed 读（裁决7·零 schema·byte-exact by construction）

> **权威** = 用户裁决7（2026-07-18·A 线去烘焙现行法·直接干）+ A 线 scope 研究 plan 阶段0（`.trellis/tasks/archive/2026-07/07-18-a-scope-g-c-scope-7-kquant-g-19-bql-18-deferreddequant-plan/research/`·**先读 `census-c-axis-bql-18.md`（18 逐处）+ `debake-staged-plan.md`（阶段0）+ `deferreddequant-target-form.md`（目标形态）**）。
> **性质** = A 线接层（**c 轴板值烘焙 → 能力事实 fail-closed 读**·公式占比 0→18）。**赢=地盘/公式+18**（byte-exact 不变=接层非重写）。

## 一、核心（研究已定·byte-exact by construction）
`RVVToEmitCBlockQuantLinear.cpp` 有 **18 处 `getIntegerCoreLmul().value_or(<板值 default>)`**（c 轴板值烘焙·default 是「读不到就填这个板值」的烘焙旁路）。**前门自 commit `0b18a3da` 起恒 stamp `integer_core_lmul`**（前门 18 处显式化·silent-signal 18→0）⟹ **这 18 处 `value_or` 的 default 分支是死码**（永不命中）。
⟹ **换成 fail-closed 读**（缺则 `notifyMatchFailure`·DeferredDequant 两轴干净纪律）= **byte-exact by construction**（死码删除·零可测行为变化）+ 把 18 个决策从「烘焙 default」变「能力事实读」（公式占比 +18）。

## 二、你要做
1. **逐处**（18·census-c-axis-bql-18.md 有行号）：`getIntegerCoreLmul().value_or(D)` → fail-closed 读（读不到 → `notifyMatchFailure`·**不再填板值 default**）。对齐 DeferredDequant 的 `getLmul()`×21 fail-closed 范式（研究 §2）。
2. **★caveat 坐实（研究关键·implement 首步）**：「前门恒 stamp·无 wired leaf 漏」是研究**由代码推得·未逐 leaf 枚举证**。你必须：
   - 读前门确认恒 stamp（`0b18a3da` 附近）·
   - **加一条 lit** 覆盖坐实（前门产物必带 `integer_core_lmul` stamp·无漏）——fail-closed 读本身兜底（漏则 match fail 而非填错板值·比旧 value_or 更安全）·但 lit 明证。
3. **byte-exact 全验**：[K-5] byte-exact 每格 golden（K-quant BQL 覆盖格）·**VLEN 翻转矩阵**（rvv/k1 双板·板值读对）·byte-exact 前后 0 diff。

## 三、验收（反误伤守门·全用已有机检·研究 §3）
1. **18 处全换 fail-closed 读**（0 处 `value_or` 板值 default 残留·grep 证）·对齐 getLmul 范式。
2. **byte-exact GREEN 每格**（[K-5] golden·BQL 覆盖格·前后 0 diff = 死码删除坐实）。
3. **VLEN 翻转矩阵绿**（rvv VLEN128 + k1 VLEN256·板值 fail-closed 读对·无误伤）。
4. **lit 坐实前门恒 stamp**（新 lit·无 wired leaf 漏·fail-closed 兜底）+ **[F-EMIT] 棘轮不动**（emit LOC 不倒退）+ **I7 verifier 不放松**（fail-closed 是收紧非放松）。
5. **零 schema 变化**（本阶段无新字段·无回门·grep 证 ODS 未动）。
6. **0 造数**·byte-exact 硬门·**禁 commit·禁 add -A**·sealed/master 不动·证据 scratchpad。

## 四、触碰集 / 遗留
- 触碰：**`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp`（c 轴 18·单文件）** + 新 lit（前门 stamp 坐实）。**与并行 deploy 线（`tools/bench/` GEN_SEAL）+ 其他 A 线阶段 TU 不相交**——但**共享 build dir → 你在独立 worktree**（隔离 build·byte-exact 走 lit + golden·板测若需走 rvv/k1 但避与主线 k1 争用·优先 lit/golden）。
- **worktree 模式**：证据写 scratchpad·**禁 commit·禁 add·禁写 repo 外文件**（build/ gitignore·跑完自动清理）。数字 + byte-exact 结果报 main（main 提交 + 派 check）。
- **遗留**：阶段1（g 轴平 K-quant 6·换 getter）· 阶段2（g 轴 grid 13·回门·新 schema 路 A/B 待 supervisor）· 阶段3（相邻 8）。本阶段是 A 线起手·zero-risk·证接层范式可行。
