# MLIR Testing Contract

测试验证真实编译器行为：dialect syntax、verification、pass 行为、plugin interface、route-provider materialization、公共 EmitC lowering、target artifact 打包，以及被主张时的 runtime 证据。

测试**不是** dashboard、readiness 状态、artifact ledger，也不是替代架构 authority（见 [core-invariants](../architecture/core-invariants.md) I4）。

## 测试类型与用途

**lit / FileCheck** —— dialect syntax/parsing；verifier 成功/失败；pass 重写行为；fail-closed 诊断；selected-body realization 在 IR 中可见；route-provider handoff 经发出的 IR/mirror 可见；公共 EmitC materialization。

**C++ tests** —— plugin registry API；route provider API；selected-body realization 结果对象；capability helper API；非文本编译器工具。

**Runtime / Hardware Evidence** —— runtime/correctness/performance 主张需要**真实执行证据**。RVV 即真 `ssh rvv` 输出（I8）。本地 compile-only、静态 MLIR check、Python smoke 都**不是** RVV runtime 证据。

## Hardware Evidence —— 一个通用契约

历史上为每个算子族抄了 ~15 份 "Generated-Bundle Evidence" 契约（per-op：reduction / MAcc / widening / clamp-select / mask-tail / packed-i4 …）。它们是**同一条证据契约**。写一次：

当某条 route 主张"真的在硬件上跑/正确/快"时，证据必须包含：

- 被测 artifact 身份（生成它的 selected variant / route）；
- 相同命名的 `ssh rvv` target/profile；
- **先正确性、后计时**：与可信 oracle（scalar 参考或外部基线）对比的正确性结果；
- 原始 stdout 或 evidence JSON，且其中**真的含**正确性与（若主张性能）计时字段。

测试断言要分清来源：`HARNESS` 前缀读的是生成的 C harness 源（断言函数调用、表达式、prototype、`printf` 格式串等源级事实）；`pattern=1 ok ...` 这类**运行期**行只能对真远端输出或确实含 stdout 的 evidence JSON 断言。

## Performance-Comparison 证据（更高门槛）

性能主张比"artifact ABI 能跑"要求更多。对比 scalar 或外部基线（如 llama.cpp）时，证据必须含：基线实现身份与版本/路径；生成 artifact 身份；两侧同一命名 `ssh rvv` target/profile；compile flags、输入尺寸、数据初始化、warmup/重复策略、计时方法；计时前的正确性检查；同时含正确性与计时的原始输出或 JSON。

> artifact ABI closeout、generated-bundle dry-run、本地 compile-only、单个 success marker —— **都不是**性能证据，不得描述成 llama.cpp parity 或"成熟"。这正是 [trunk-discipline](../guides/trunk-discipline.md) 点名的"无限证据 closeout"反模式。

## 给 agent 的判断点（不是 gate）

- 写"第 N 个 generated-bundle evidence 测试"前：它验证的是一条**新改动的** route 吗？还是在已验证过的 family 上重复刷证据（枝节）？
- 主张性能前：有没有"先正确、后计时、且赢 baseline"的真 `ssh rvv` JSON？没有就别说快。
- 测一条 MLIR 行为时，优先 lit/FileCheck；只有文本测不了的语义才上 C++。
