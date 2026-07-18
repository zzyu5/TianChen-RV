# PRD · A 线阶段2 · g 轴 grid/codebook 13 处 + 阶段1回门 → 路 B 描述符（用户放行·公式 24→37）

> **权威** = 用户裁决（2026-07-18·schema 映射表已过目·**放行路 B 施工**）+ A 线 scope plan 阶段2（`.trellis/tasks/archive/2026-07/07-18-a-scope-g-c-scope-7-kquant-g-19-bql-18-deferreddequant-plan/research/`·读 `census-g-axis-kquant-19.md` §阶段2 + `debake-staged-plan.md` §4 映射表）。
> **性质** = A 线接层 · **涉 schema 新增（路 B·回门已批）**· 公式占比 24/37 → 37/37（确凿 scope 全清）。**这是回门已批的施工·非再回门**——**但若发现映射表外的新 schema 需求·停·报 main**。

## 一、★用户硬条件（违反=作废·逐条）
1. **路 B（描述符）**：新增 6 个 `OptionalAttr<I64Attr>`（`groups_per_sub`/`num_groups`/`indices_per_sub_block`/`signs_per_sub_block`/`num_groups_per_half`/`group_lanes`·映射表见 plan §4）·由前门 grid/codebook decode-facts 结构 stamp·emitter 换 `coreOp.getXxx()`。
2. **★律 1 主梁（正当性）**：这些量是**格式定义的独立自由度**（格式一等事实·非导出量）——它们"恰好=subBlock/8"是**现有几格式的偶然巧合·非规律**。焊进发射器推导式 = 下一个打破巧合的新格式就得改发射器 = **违律 1「新格式零代码」**。故走 attr（一等事实）非派生。
3. **★★缺席硬失败·严禁默认值（最硬）**：`OptionalAttr` 缺席时**发射器必须硬失败带诊断**（`notifyMatchFailure` 点名缺哪字段·或 unreachable+assert）——**严禁配默认值**。optional 只是不砸存量 fixture 的 schema 让步·**不是给发射器兜底的许可**。配默认值 = 把 c 轴刚治好的 `value_or` 病原样复制到 g 轴 = **违约作废**。**样板 = 立场文生产闸**（`isRepack && halfLanes != 0` 顶成 unreachable + assert·grep 找该 pattern 照抄）。
4. **★#5/#6（`weightDOffset=80`/`weightDminOffset=82`·从阶段1 回门）施工前先核 ODS scope 冲突**：q2_K core op（`GgmlBlockDotQ2KQ8KIntegerCoreOp`·`RVVOps.td:7024`）ODS **明标 fp32 fold DELIBERATELY OUT OF SCOPE**。**别用「同法加 attr」盖过去·先确认能不能加**（是否与该 op 语义冲突）——**不能加则报 main·别硬塞**。
5. **#2/#4（`numLanes=8`·从阶段1 回门）**：verifier 强制 `num_lanes==8`·真源 = `num_lanes` typed attr（`Q4KHorizontalFoldOp` 有·`Q4KSumsFoldScaleDOp`/loop-body 无）·加 attr（同路 B·非派生·census 建议 numSubBlocks/2 已证 q6_K byte-INEXACT）。

## 二、★首步·fail-closed 硬失败五分钟验证令（用户令·先于阶段2 主体）
c 轴 required=暂不升的裁决**压在假设「fail-closed 是硬失败」上**·须先证：
- **构造一个故意不盖 `integer_core_lmul` 的输入**（MLIR·剥 stamp）·跑 lowering·确认结局 =
  - **硬失败（报错点名缺哪字段）** → c 轴 required 暂不升裁决成立·optional 够用·记录证据。
  - **软 fallthrough（静默落进另一条 lowering 路·generic 兜底）** → **暗洞**（比 value_or 更隐蔽·连 lowering 路都换=ISSUE-002 抽签形状）·**停·报 main**（required 优先级立刻上调·裁决重开）。
- 这决定路 B 的缺席行为样板（硬失败）是否已在 c 轴坐实可照抄。

## 三、你要做（首步验证通过后）
1. 路 B：ODS 加 6 attr（+ #2/#4 numLanes·#5/#6 若 ODS scope 允许）·前门 stamp（decode-facts 结构·照 `weight_scales_high_byte_offset` 先例）·emitter 13 处换 `coreOp.getXxx()`（缺席硬失败·严禁默认值）。
2. **byte-exact regen-diff 0**（前门生产路 grid/codebook 覆盖格 × VLEN128/256·前后 0 diff）+ direct-input fixture 补 stamp（同阶段0/1·byte-exact）。
3. **新 attr optional 不破存量 fixture**（optional·缺席 fixture 仍过·但**生产路必 stamp**·lit 坐实无 wired leaf 漏)。

## 四、验收
1. **首步 fail-closed 硬失败验证**（证据·硬失败=裁决成立 / 软 fallthrough=停报 main）。
2. **13(+回门)处全换 coreOp.getXxx()**·**缺席硬失败·grep 证 0 处默认值兜底**（违约红线）。
3. **byte-exact regen-diff 0**（前门生产路 + fixture·前后 0 diff）。
4. **#5/#6 ODS scope 核**（能加/不能加·不能加报 main·别硬塞）。
5. **[F-EMIT] 棘轮不退·I7 不放松·lit 绿**（新失败=0·pre-existing 除外）·前门恒 stamp lit（无漏）。
6. **0 造数**·byte-exact 硬门·**worktree·禁 commit·禁 add**（main cherry-pick·worktree commit 保存报 hash）·sealed/master 不动·数字报 main。

## 五、触碰集 / 遗留
- 触碰：`include/Weft/Dialect/RVV/IR/RVVOps.td`（6+ OptionalAttr）+ `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`（stamp）+ `lib/Conversion/RVV/RVVToEmitCKQuant.cpp`（13 处 getter）+ direct-input fixture。**跨 3 文件跨层**（回门已批）。**worktree 隔离**。
- **遗留**：#5/#6 若 ODS scope 不允许 → 报 main（可能需 core op 语义讨论）。阶段3（相邻同轴 8）。c 轴 required 升级待办（触发=A2/A3 接层收口+fixture 冻结后第一个收尾窗·见 ISSUE 登记）。
