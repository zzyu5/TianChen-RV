# stage3 换心: replace RVV route-plan string machine with real emitc IR (N3)

Parent: 06-12-mlir-audit-refactor (ADR Stage 3). Created 2026-06-12.

## Goal

把**活的** RVV codegen 从「字符串机器」换成**真正的结构化 emitc IR 构造**，逐 route family
strangler-fig 推进，每个 family 都用真 `ssh rvv` 硬件证据（I8）验证编译+运行+数值正确，再删
它的字符串计划。终态：字符串机器消失、build/lit 绿、带 ssh rvv 灯证据、真 MLIR 机制、审美干净。

## Why（这是项目"真伪 MLIR"判定的核心病灶）

审计核心判定：**骨架真、肌肉假——RVV codegen 实质是字符串机器**。现状（实测 HEAD）：
- `lib/Plugin/RVV/EmitC/` 仍 **81,968 行**活字符串机器（去伪只删了死的校验簇，活的没动）。
- intrinsic 名靠 `Twine("__riscv_v...")+sew+lmul` 拼接（RVVEmitCRoutePlanning.cpp 27 处）。
- 可执行事实（callee/operand expression/cType）以 `std::string` 在 plan 层流转
  （TCRVEmitCLowerableInterface.h 22 处 std::string；如 `"n - i"`、`"out + i"`）。
- emitc dialect 在链路里只当字符串计划的合法化载体（call_opaque callee=字符串 +
  自写 C 表达式 re-parser），违反 I5 精神。
- 全仓 **0 处 RewritePatternSet/ConversionPattern** —— 变换不用 MLIR conversion 框架。

这是让项目"看着是 MLIR、其实是字符串 C 生成器"的根因。换掉它才让 N1（capability IR）/
N3（capability/resource-aware tune）站在真 IR 上，而不是字符串属性上。

## 硬件前提：ssh rvv 已确认可用（I8 可达）

`ssh rvv` = 真 riscv64：`isa rv64imafdcv...zve64d_zve64f_zvfh...`，带 `/usr/bin/clang`。
→ 每个 family 可真编真跑出"灯"。这是本 task 一切硬件主张的来源。

## Approach（strangler-fig，逐 family，硬件验证）

架构决策由 Fable 投资 wecebi3aa（→ research/heart-replacement-plan.md）定：
- A：直接细粒度 emitc 构造（typed Value operands，去掉字符串 plan 层）。
- B：typed RVV-intrinsic 中间 op（ODS）+ 真 ConversionPattern/DialectConversion lowering 到
  emitc（顺带补"0 ConversionPattern"红旗）。
- 选定后逐 family：建真 emitc IR → lit IR 测结构 → **ssh rvv 编译+运行+数值正确** → 删该
  family 字符串计划。其它 family 仍走旧字符串路，互不影响（链路始终可出活）。

## Acceptance Criteria (evolving)

- [ ] 架构决策落定（A/B/hybrid），有 Fable 评审依据，对齐 thesis + 真 MLIR 机制。
- [ ] beachhead family：可执行事实以 typed emitc Value 结构化（无字符串 expression operand），
      其 intrinsic 仍可用 emitc.call_opaque（callee 字符串是 emitc 惯例，允许）。
- [ ] beachhead 有 lit IR 测试（测结构，非测字符串）+ **真 ssh rvv 编译+运行+数值正确证据**
      （记录命令与输出）。
- [ ] beachhead 字符串计划删除后 build 绿 + lit honest-green（≤3 environmental reds）。
- [ ] 逐 family 推进，最终 RVVEmitCRoutePlanning 字符串机器整体退役。

## Out of Scope（本 task 边界）

- 不要求一次换完所有 family（strangler-fig 多 PR）。
- 不在没有 ssh rvv 证据时宣称硬件正确性（I8）。

## Notes

- 删某 family 字符串计划时走 [dead-mirror-removal guide]（先迁消费者/fixtures 再删）。
- 顾问：advisor() 工具不可用，用 Agent `model: fable` 替代（用户已认可此法）。
- 关键文件：RVVEmitCRoutePlanning.cpp、*PlanOwners.cpp、TCRVEmitCLowerableMaterializer.cpp、
  TCRVEmitCLowerableInterface.h、RVVOps.td + TCRVEmitCLowerableOpInterface.td、
  scripts/rvv_generated_bundle_abi_e2e.py（ssh rvv harness）。
