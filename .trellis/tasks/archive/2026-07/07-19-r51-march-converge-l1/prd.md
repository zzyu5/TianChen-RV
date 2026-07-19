# PRD — r5.1 判决实验1(管道通) march 收敛 L1（完成 A 线四判决实验真全绿）

## 目标
把 march ISA-token 解析**收敛到探测层一处**（`RVVCapabilityProfile.cpp` 的 `derive*` 权威族），消除唯一剩余的 token 泄漏 **L1**，使判决实验 1(管道通)的第二半 GREEN → A 线四判决实验真全绿。

## 背景（JE1 判决B scope·`experiments/active/r5.1-je/je1-pipe/README.md` B 节）
- **收敛目标** = 探测层一处 = `lib/Plugin/RVV/RVVCapabilityProfile.cpp` 的 `derive*` 权威族（6 函数：hasRVVVectorHint:140 / deriveSupportedSEWAllowList:222 / deriveSupportedLMULAllowList:260 / deriveMinimumVLEN:289 / deriveHasZvl128b:334 / deriveRVVVersion:341）——所有 ISA-token 字符串切分只应住此。
- **B2 十个消费者**（C1-C10）= 线程 march+hints 并**调 derive***（funnel 过权威·**非** token 违规·合规·不动）。深层 threading-轴 收敛（march 只解析一次→物化 IR→下游读 IR fact）= 更大重构·**本波不做**（另 fork）。
- **L2**（IME `kIMEMarchToken`@IMEExtensionPlugin.cpp:237）= JE4 删 deriveIMEPresent 后**已坍缩为一处**（IME 插件本地·跨插件·合理·其余 xsmtvdotii 全是注释/诊断串·非解析）。
- **★L1 = 唯一剩余真泄漏**：`RVVEmitCRoutePlanning.cpp:101 containsRVVVectorISAHint`（独立重切 rv64gcv/rv32gcv/zve/zvl/zvfh/rvv·called :148 校验 isa_vector_hints property）。

## 施工（L1·只减不增）
1. **导出权威**：把 `hasRVVVectorHint`（`RVVCapabilityProfile.cpp:140` 定义）**声明进 header** `include/Weft/Plugin/RVV/RVVCapabilityProfile.h`（在 `derive*` 权威族声明处·同 namespace）。
2. **转调 + 删本地**：`RVVEmitCRoutePlanning.cpp:148` 的 `containsRVVVectorISAHint(isaVectorHints)` → 改调 `hasRVVVectorHint(isaVectorHints)`；**删除本地 `containsRVVVectorISAHint` 定义**（:101-108 本地 token 列表）+ 相关 include 若需。

## ★byte-exact 门（硬前置·先证）
- **token 集核对**：`hasRVVVectorHint`（读 zve/zvl/zvfh/gcv/xtheadvector/rv64…）vs `containsRVVVectorISAHint`（读 rv64gcv/rv32gcv/zve/zvl/zvfh/rvv）——**逐 token 核对两者判定一致**（尤其 rv32gcv/rvv 这类边界·若权威缺某 token 会改行为=非 byte-exact）。若有差集·如实报（别静默改行为）。
- **emit byte-exact**：改前后·对一组含 isa_vector_hints 校验路径的 fixture（RVVEmitCRoutePlanning 消费点）跑 weft-opt·**emit 输出逐字节不变**（`isa_vector_hints` property 校验结果对所有 march 一致·rv64gcv/zve/zvl/xtheadvector/无V 都测）。
- **build + lit**：全 lit 绿（953/956 基线·3 pre-existing strided 不计）。三 grep 新代码干净（删了 token 列表·只减·不新增焊死 g/θ）。

## 收敛自证（判决实验 1 part 2 GREEN 判据）
- 机算证：**RVV ISA-token 切分收敛到 RVVCapabilityProfile.cpp 一处**——`git grep -nE 'rv64gcv|zvl[0-9]|xtheadvector|zvfh' -- lib/Plugin/RVV/ | grep -v RVVCapabilityProfile.cpp | grep -v _attic`（排除 comment/诊断串/toolchain-arg）后·**探测层外零真 token 解析**（L1 消·L2 IME-local 合理）。记 before/after grep 计数。

## 交付（首节三项）
1. march 收敛 L1 完成·判决实验 1 全绿（strip 宽 GREEN〔已达〕+ march 收敛 GREEN〔本任务〕）·byte-exact 证 + 收敛 grep before/after。
2. 剩余：threading-轴 深层收敛（march 解析一次→IR·另 fork·非本波）·L2 IME-local（合理·不动）。
3. 论文侧：march 单一解析地坐实（capability 探测层唯一权威·H-5 补充）。

## 账面纪律
- **不 git commit**·git add 只纳源·勿纳 build/worktree·worktree base 核对（当前 tip `f4cc0b437`·若旧线 reset）。
- byte-exact 先于任何·token 集若有差集**必如实报**（别为过门静默改语义）。
- 返回结构化：改的文件 file:line·token 集核对结果·emit byte-exact 证·收敛 grep before/after·lit 状态。

## 参照
- scope 正本：`experiments/active/r5.1-je/je1-pipe/README.md`（判决B B1-B5）。
- 权威族：`lib/Plugin/RVV/RVVCapabilityProfile.cpp:140-375`。
- L1：`lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp:101-108,148`。
- memory `r5.1-judgment-experiments-green.md`。
