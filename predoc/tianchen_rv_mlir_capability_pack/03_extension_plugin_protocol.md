# 03. Extension Plugin Protocol 设计

## 3.1 模块定位

Extension plugin protocol 定义新硬件能力如何接入 TianChen-RV MLIR。

TianChen-RV 的可扩展性不应该依赖在核心 pass 中写大量 target-specific 分支，而应该依赖插件提供局部化能力：

```text
capability registration
extension dialect registration
variant proposal
legality verification
tuning space
cost model
lowering/emission
runtime glue
```

## 3.2 插件化的真实含义

插件化不等于新增硬件零工作量。

新增 IME、offload 或未来 custom ISA 时，仍然需要写：

```text
ops / types / attributes
capability definitions
variant generator
legality rules
cost model
tuning rules
lowering patterns
runtime or toolchain adapter
```

插件化的目标是：

```text
新增能力的代码主要集中在 plugin 内；
核心 pass 不硬编码具体扩展名称；
核心 pass 通过 registry 和 interface 与插件交互；
High-level MLIR 的前端语义转换不因新增扩展而重写。
```

## 3.3 插件接口

建议将插件抽象为以下接口集合。

### CapabilityProvider

负责注册和探测能力。

```cpp
class CapabilityProvider {
public:
  virtual void registerCapabilities(CapabilityRegistry &registry) = 0;
  virtual LogicalResult detectCapabilities(TargetCapability &target) = 0;
};
```

职责：

```text
声明插件提供哪些 capability；
从 target profile、编译选项、runtime probe 中填充 capability；
声明 capability implication 和 conflict；
暴露该插件是否对当前 target available。
```

### DialectProvider

负责注册 extension dialect、ops、types、attributes。

```cpp
class DialectProvider {
public:
  virtual void registerDialects(DialectRegistry &registry) = 0;
};
```

职责：

```text
RVV plugin 注册 tcrv.rvv；
IME plugin 注册 tcrv.ime；
Offload plugin 注册 tcrv.offload；
未来插件注册自己的 extension dialect。
```

### VariantBuilder

负责从 high-level MLIR op 生成 execution variant。

```cpp
class VariantBuilder {
public:
  virtual bool supports(Operation *highLevelOp,
                        const TargetCapability &target) = 0;

  virtual LogicalResult proposeVariants(Operation *highLevelOp,
                                        const TargetCapability &target,
                                        SmallVectorImpl<TCRVVariant> &out) = 0;
};
```

职责：

```text
判断某 high-level op 是否可由该插件实现；
生成 tcrv.exec.variant；
variant body 中使用该插件的 extension dialect ops；
声明 requires、origin、shape guard、tuning metadata。
```

### LegalityVerifier

负责检查 variant 在 target 上是否合法。

```cpp
class LegalityVerifier {
public:
  virtual LogicalResult verifyVariant(TCRVVariant variant,
                                      const TargetCapability &target,
                                      DiagnosticEmitter &diag) = 0;
};
```

职责：

```text
检查 requires 是否满足；
检查 ops/types 是否被当前 capability 支持；
检查 dtype、shape、layout、runtime ABI 是否可用；
检查工具链 emission path 是否存在。
```

### TuningSpaceProvider

负责声明变体内部可调参数空间。

```cpp
class TuningSpaceProvider {
public:
  virtual void populateTuningSpace(TCRVVariant variant,
                                   TuningSpace &space) = 0;
};
```

示例：

```text
RVV: LMUL, SEW, VL policy, unroll, packing, thread partition
IME: fragment shape, K blocking, accumulator residency, packing
Offload: batch size, transfer threshold, async overlap
```

### CostModelProvider

负责估计 variant 代价。

```cpp
class CostModelProvider {
public:
  virtual Cost estimateCost(TCRVVariant variant,
                            const TargetCapability &target,
                            const ShapeContext &shape) = 0;
};
```

职责：

```text
给 variant selection 提供排序依据；
结合 shape、dtype、capability、runtime transfer cost；
允许 profile-guided 更新。
```

### EmissionProvider

负责最终 lowering/emission。

```cpp
class EmissionProvider {
public:
  virtual void populateLoweringPatterns(RewritePatternSet &patterns) = 0;
  virtual LogicalResult emitRuntimeGlue(ModuleOp module,
                                        const TargetCapability &target) = 0;
};
```

职责：

```text
RVV -> MLIR vector / LLVM scalable vector / RVV intrinsic / inline asm；
IME -> vendor intrinsic / inline asm / patched toolchain path；
Offload -> vendor C ABI / runtime library call；
Future custom ISA -> custom intrinsic / inline asm / backend patch adapter。
```

## 3.4 核心 pass 如何使用插件

核心 pass 的逻辑应是：

```text
1. 读取 target capability。
2. 加载并注册 extension plugins。
3. 对每个 high-level op，遍历插件 registry。
4. 每个插件根据 capability 判断是否支持该 op。
5. 插件生成一个或多个 tcrv.exec.variant。
6. 核心 verifier 调用 plugin verifier 检查 variant。
7. 核心 selector 根据 cost/tuning/dispatch policy 选择或保留 variants。
8. emission 阶段调用各插件 lowering/emission。
```

核心 pass 不应包含：

```cpp
if (target.hasRVV()) { ... }
if (target.hasIME()) { ... }
if (target.hasSophgo()) { ... }
```

而应包含：

```cpp
for (auto *plugin : pluginRegistry.enabledPlugins()) {
  if (!plugin->supports(op, target)) continue;
  plugin->proposeVariants(op, target, variants);
}
```

## 3.5 插件注册的信息

每个插件至少应注册：

```text
plugin name
plugin version
provided capabilities
required external toolchain/runtime
extension dialects
supported high-level op classes
variant generation hooks
legality rules
tuning parameters
cost model
emission paths
fallback behavior
```

## 3.6 插件边界和核心边界

### 插件内部负责

```text
extension-specific types and ops；
extension-specific lowering；
extension-specific tuning；
extension-specific runtime ABI；
extension-specific legality details；
extension-specific toolchain workaround。
```

### 核心负责

```text
capability registry；
plugin registry；
variant container；
variant selection orchestration；
dispatch/fallback structure；
common verifier orchestration；
common diagnostics format。
```

## 3.7 当新扩展需要修改核心时

系统不能承诺任何未来扩展都完全不需要核心修改。合理表述是：

```text
如果新扩展能映射到现有 capability、variant、resource、emission interfaces，则应只新增 plugin。
如果新扩展引入全新的执行语义，例如分布式多 hart collective、非标准一致性模型、设备端调度队列、不可建模的异步副作用，则可能需要扩展核心 interfaces。
```

科研评价中应记录：

```text
新增插件修改了多少核心 pass；
新增插件是否只注册新能力和新 variant；
核心是否出现 extension-specific branch；
是否需要新增通用 interface。
```

## 3.8 当前规划插件

当前系统优先规划：

```text
RVV plugin      当前真实主线
IME plugin      后续 K3 到位后的 matrix-extension 接入
Offload plugin  Sophgo/vendor runtime capability
Scalar/default  保守 fallback
```

未来插件不受这四类限制。

## 3.9 科研价值

Plugin protocol 的科研价值在于：

```text
它把 RISC-V extensibility 从“修改后端代码”转化为“新增 extension plugin”；
它让 capability、variant、legality、tuning、emission 的边界清晰可测；
它给新增 IME/offload/custom 能力提供统一接入协议；
它能用 core-pass 修改量、plugin-local LOC、variant coverage、性能结果来验证。
```

