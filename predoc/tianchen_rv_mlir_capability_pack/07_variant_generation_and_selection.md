# 07. Variant Generation, Selection and Tuning 设计

## 7.1 模块定位

本模块定义 high-level MLIR op 如何进入 TianChen-RV MLIR。

核心原则：

```text
不要先生成 generic tcrv compute op；
由 extension plugins 直接提出 execution variants；
核心 pass 负责组织、验证、选择、dispatch；
计算细节留在 extension dialect 中。
```

## 7.2 总流程

```text
High-level MLIR op
    |
    v
Collect target capabilities
    |
    v
Query extension plugins
    |
    v
Plugins propose execution variants
    |
    v
Capability and legality verification
    |
    v
Variant-local tuning and cost estimation
    |
    v
Select variant or generate dispatch
    |
    v
Lower selected variants through plugin emission paths
```

## 7.3 输入

输入可以来自：

```text
linalg.matmul
linalg.generic reduction
stablehlo dot_general
stablehlo softmax-like region
tosa ops
custom high-level kernel dialect
```

TianChen-RV 只要求输入 op 的语义已经由 high-level MLIR 表达清楚。它不负责恢复算法语义。

## 7.4 Plugin-driven variant proposal

对于每个 high-level op，核心 pass 不直接 lower，而是调用插件：

```text
RVV plugin: 是否可以生成 RVV variant？
IME plugin: 是否可以生成 IME variant？
Offload plugin: 是否可以生成 offload variant？
Scalar/default plugin: 是否可以生成 fallback？
Future plugin: 是否可以生成其他 variant？
```

输出是多个 `tcrv.exec.variant`。

示意：

```text
linalg.matmul
  -> @rvv variant
  -> @sophgo_offload variant
  -> @fallback variant
```

K3/IME 到位后：

```text
linalg.matmul
  -> @rvv variant
  -> @ime variant
  -> @sophgo_offload variant
  -> @fallback variant
```

## 7.5 Variant IR 的内容

每个 variant 必须包含：

```text
variant name
origin plugin
required capabilities
shape/dtype/layout preconditions
extension dialect ops
cost/tuning metadata
emission path
fallback or dispatch relation
```

示意：

```mlir
tcrv.exec.variant @rvv
  requires = #tcrv.requires<["rvv", "zvfh"]>
  origin = "rvv-plugin"
  tuning = #tcrv.tuning<...>
  cost = #tcrv.cost<...> {
  ... tcrv.rvv ops ...
}
```

## 7.6 Legality verification

Variant 生成后，必须通过两层检查。

### Core legality

```text
variant 是否声明 requires；
requires 是否被 target capability 满足或可 runtime dispatch；
variant 是否有 origin plugin；
variant body 中是否出现未知或未注册 extension op；
dispatch/fallback 是否完整。
```

### Plugin legality

```text
RVV plugin 检查 RVV-specific legality；
IME plugin 检查 IME-specific legality；
Offload plugin 检查 runtime ABI 和 transfer legality；
Future plugin 检查自己的 extension rules。
```

核心 verifier 只 orchestrate，不硬编码具体扩展规则。

## 7.7 Variant selection

Variant selection 负责在多个合法 variants 中选择：

```text
单一静态 variant；
多个 runtime dispatch variants；
fallback-only path。
```

选择依据：

```text
target capability
shape/dtype/layout
cost model
profile data
runtime availability
tuning result
user policy: performance / portability / debug / no-offload
```

## 7.8 Runtime dispatch

当多个 variants 在不同条件下有效时，应生成 dispatch：

```mlir
tcrv.exec.dispatch {
  case @sophgo_offload if #tcrv.cond<"runtime_available && large_shape">
  case @ime            if #tcrv.cond<"ime_available && dtype_supported">
  case @rvv            if #tcrv.cond<"rvv_available">
  fallback @scalar_or_default
}
```

Dispatch 条件应来自 capability object、shape guard、runtime probe 或 cost model。

## 7.9 Capability-aware tuning

Tuning 在 TianChen-RV 中不是孤立创新点，而是 variant quality 的系统能力。

它分两层：

### Variant selection tuning

选择使用哪种能力：

```text
RVV
IME
Offload
Fallback
```

### Variant-local tuning

在某个 variant 内部调整资源参数：

```text
RVV: LMUL, SEW, VL policy, unroll, packing, thread partition
IME: fragment shape, K-block, accumulator policy, packing
Offload: transfer threshold, batch size, async overlap, buffer reuse
```

## 7.10 Cost model 边界

Cost model 可以是解析式、经验式、profile-guided 或混合形式。系统不应把 cost model 固定死。

Cost model 输出应能解释：

```text
为什么选择 RVV；
为什么选择 IME；
为什么选择 offload；
为什么保留 runtime dispatch；
为什么回退到 fallback。
```

## 7.11 Diagnostics

Variant pipeline 应输出结构化诊断：

```text
没有插件支持该 high-level op；
插件支持但 capability 不满足；
插件生成 variant 但 legality fail；
variant 可用但 emission path 缺失；
多个 variants cost 接近，保留 dispatch；
offload runtime unavailable，选择 RVV fallback。
```

这些诊断对 agent、开发者和论文实验都很重要。

## 7.12 关键区别

TianChen-RV 的 pipeline 不是：

```text
high-level op -> generic tcrv op -> backend-specific lowering
```

而是：

```text
high-level op -> plugin-proposed execution variants -> capability-driven selection/lowering
```

这个区别是系统科研价值的核心。

