# Unified EmitC Route

公共 EmitC route 是 origin 插件已 select / legalize /（可选）realize / 提供 lowerable route **之后**的中性 materialization 路径。它是 N2（一条 common 路径服务所有 family）的承载体之一。

```text
selected typed extension body
  -> 插件 route provider 建 TCRVEmitCLowerableRoute
  -> 公共 EmitC materializer
  -> MLIR EmitC ops -> C/C++ emitter
  -> target artifact
```

> 本文件描述公共侧的**稳定边界**。具体支持到哪些 route family、哪个在调——是状态，进 task/journal。

## Common Responsibility

公共 EmitC 只做中性 materialization：

- 调用通用 lowerable-route interface；
- 从 provider payload 发 MLIR EmitC ops；
- 保留 provider 给的确定性符号名与打包结构；
- 把 materialized EmitC 交给 C/C++ emission 与 target export；
- provider route 缺失/畸形时报诊断（fail closed）。

公共 EmitC **不**做（见 [core-invariants](../architecture/core-invariants.md) I3/I4/I5）：选 intrinsic；推 dtype/SEW/LMUL/mask-tail/operation kind；建 schedule 或 body shape；从参数名合成 runtime ABI；按 family 语义分支；把 route id / emission-plan status / manifest / artifact metadata 当 route authority。

对 RVV：C 向量类型串、intrinsic 名、header、ABI 映射、route payload、legality 全部 RVV-plugin-owned。公共代码可携带 plugin 预计算的串/payload 片段，但**不派生**它们。

## `TCRVEmitCLowerableRoute`

从 origin 插件/provider 传给公共 materialization 的契约对象。只暴露 materialization 所需的通用信息：origin plugin/family；selected variant 引用；typed/realized body 或 selected boundary 引用；provider-owned route payload；provider 输出的 headers/includes 与 runtime ABI bindings；EmitC 构造 hooks 或 payload；diagnostic mirrors。

它是 **provider-built**，不从 emission-plan 诊断、route id、artifact 名、source-front-door 标记、semantic role graph、manifest 或 descriptor 重建（I4）。

## Provider Route 契约 —— 一个通用版本

emitc-route 历史上为每个 route/memory form 抄了 ~40 份同结构契约（operand binding、header/type summary、fact surface、route validation，按 elementwise/memory/segment2/MAcc/widening/conversion 逐一复制）。它们是**同一条通用契约**。写一次：

provider 给公共侧的，是一组从 **typed/realized body 事实**派生的 route value：operand binding（哪些 SSA/ABI 值绑到哪些 intrinsic 参数位）、header/type summary、intrinsic 拼写、memory form、mask/tail、runtime AVL/VL 绑定。公共侧按原样 materialize。

通用规则（对所有 route family 一致）：

- route value **结构性源自 typed body**或 realize 结果；fact surface / metadata mirror 可有，但只是镜像，不是 authority（I4/I5）。
- provider route 缺失/畸形/只有 metadata → 公共侧 **fail closed** 报诊断，绝不补 compute（I7）。
- 公共侧逐字 materialize，不发明、不推断、不按 family 分支。

新增一个 route family = 让 provider 输出满足这条通用契约，**不是**再抄一份 operand-binding + fact-surface + route-validation 三件套。覆盖到哪些 family 见 [rvv-plugin 覆盖表](../extension-plugins/rvv-plugin.md#当前覆盖coverage按结构化计算类)。

## 给 agent 的判断点（不是 gate）

- 在写"第 N 个 fact surface / route validation 契约"之前：它真不同于上面的通用契约吗？不同点是什么？说不出就别抄。
- IME/offload 接入时：复用同一公共 EmitC 与同一通用 provider 契约，就是 N2 的证据；若发现公共侧要为它加 family 分支，说明设计破了，先报告。
