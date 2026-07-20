# 架构（Architecture）

> **版本**：v2
> 本层描述 Weft-RV 的稳定结构、当前实现落点和目标责任边界。当前事实由代码复核；目标设计由用户裁决后版本化进入本层。

## 读法

每项内容应明确属于：

| 标签 | 含义 |
|---|---|
| **【现状】** | 当前代码真实存在，可用路径或谓词核查 |
| **【目标】** | 尚未完全实现的设计方向 |
| **【定法】** | 实现与目标都必须遵守的稳定边界 |

缺席断言必须用无截断、可复跑的命令。计数必须附生成器或完整命令。跨文件引用不写行号。

科研价值组织见 [canon · 暂定科研主张](../canon/暂定-科研主张.md)；本层只负责把它变成可实现、可测试的结构。

## Pre-Development Checklist

- [ ] 仍是 kernel-level execution layer，而非新增高层 tensor/tile IR？
- [ ] g、c、ω 的来源和 owner 明确吗？
- [ ] g/c/ω 只在 plugin-local formula/selection 边界相遇吗？
- [ ] formula/provider 产生 typed plan、candidate、legality 或 prior，而不是 C 字符串？
- [ ] measurement 只在合法候选中选 winner，没有创造 compute？
- [ ] selected result 在 emission 前进入 typed body 吗？
- [ ] emitter 没有重新读取 format、board、march 或 measurement 做同一决定？
- [ ] core/common 没有 family-name branch？
- [ ] unsupported 或空合法集 fail-closed 或进入 canon 允许的具名 fallback？
- [ ] runtime sparse/MoE 没有被写成当前实现？
- [ ] 新抽象有第二个真实使用者，或保持 family-local？

## 本层地图

| 文件 | 内容 |
|---|---|
| [系统定位与边界](./系统定位与边界.md) | 项目范围、主链、工位与 g/c 解耦边界 |
| [能力模型](./能力模型.md) | capability 来源、关系、profile、probe、default/conflict |
| [核心方言](./核心方言.md) | weft.exec execution envelope |
| [插件协议](./插件协议.md) | 插件接口、五件套、locality 和 family 接入 |
| [家族现状](./家族现状.md) | RVV、IME、Scalar、Offload 等 family 的代码事实 |
| [变体流水线](./变体流水线.md) | 公式构造、legality、bounded selection、typed stamping 和 realization |
| [发射与降级](./发射与降级.md) | typed body 到 plugin route provider、common EmitC 和 artifact |
| [实现栈](./实现栈.md) | C++/MLIR 主栈、Python tooling 边界、构建与测试 |
| [core-invariants](./core-invariants.md) | 历史兼容指针；I1–I9 正本在 [canon](../canon/核心不变量.md) |

## 当前主改造方向

1. 把五类 dequant plan、LMUL、SP4 和 loop-order 统一成小型 decision contract。
2. 修 GridLookup formula/verifier 双 authority。
3. 删除 emitter 的二次选择。
4. 统一 measurement schema、qualification 与 compiled winner view。
5. 完善 capability fields 和 per-board instances。
6. 用 IME 或另一 family 验证同一最小 contract。

这些方向是渐进迁移，不要求一次性重写 RVV，也不建立大一统 Formula IR。

## Quality Check

- 每个 code-affecting decision 有唯一 owner。
- capability、facts、measurement、typed body 和 emitter 的职责不重叠。
- 当前路径与目标路径分开书写。
- 当前状态不从旧 task 或旧文档推断。
- formula/selection 契约以 [变体流水线](./变体流水线.md) 为唯一工程正本。
- spec 不以当前覆盖数字冒充稳定架构。
