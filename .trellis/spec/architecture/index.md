# 架构（Architecture）

> **版本**：v4
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

- [ ] 仍是 high-level MLIR 之后的 operator execution layer，而非新增高层 tensor/tile IR？
- [ ] 系统身份是可扩展 MLIR operator compiler，RISC-V 只是旗舰实例而非 core 上界吗？
- [ ] 每个 registered/direct production operator entry 都进入 catalogued formula/construction 阶段吗？
- [ ] g、c、ω 的来源和 owner 明确吗？
- [ ] g/c/ω 只在 plugin-local formula/selection 边界相遇吗？
- [ ] formula/construction 产生 typed plan、candidate、legality 所需条件或 prior，而不是 C 字符串？
- [ ] measurement 只在合法候选中选 winner，没有创造 compute？
- [ ] selected result 在 emission 前进入 typed body 吗？
- [ ] emitter 没有重新读取 format、board、march 或 measurement 做同一决定？
- [ ] core/common 没有 family-name branch？
- [ ] unsupported 或空合法集 fail-closed 或进入 canon 允许的具名 fallback？
- [ ] 横向公式层 census 中全部 entry/caller 已迁移，旧 caller、兼容 bridge、第二 dispatcher/writer 已归零吗？
- [ ] 中间迁移状态只存在于未完成 worktree，最终合入没有未迁 operator/family 吗？
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
| [公式层与覆盖](./公式层与覆盖.md) | 全项目 formula catalog、`g/c/ω` 消费、production entry coverage 与横向 cutover |
| [发射与降级](./发射与降级.md) | typed body 到 plugin route provider、common EmitC 和 artifact |
| [实现栈](./实现栈.md) | C++/MLIR 主栈、Python tooling 边界、构建与测试 |
| [退役与原子合入](./退役与原子合入.md) | 单一路径、compatibility 边界、slice cutover、worktree 合入与退役验证矩阵 |
| [core-invariants](./core-invariants.md) | 历史兼容指针；I1–I9 正本在 [canon](../canon/核心不变量.md) |

## 当前主改造方向

一次横向建立全项目 formula/construction layer：闭合所有 production operator/family entry 和 analytic authority，统一轻量 catalog 与 family-local typed evaluator，迁移全部 caller，删除与公式同义的旧 decision provider/helper/replay/emitter redecision，并完成 catalog、entry、dependency 与 semantic coverage。施工可在 task worktree 内排序，最终合入不得保留未迁公式或按 operator/family 分裂的路径。

该重构不建立大一统 Formula IR；共同的是 descriptor、axis/result 语义、统一调用阶段和覆盖，计算知识仍 family-local。

## Quality Check

- 每个 code-affecting decision 有唯一 owner。
- capability、facts、measurement、typed body 和 emitter 的职责不重叠。
- 当前路径与目标路径分开书写。
- 当前状态不从旧 task 或旧文档推断。
- formula/selection 契约以 [变体流水线](./变体流水线.md) 为唯一工程正本。
- active-path 退役和原子 cutover 以 [退役与原子合入](./退役与原子合入.md) 为唯一工程正本。
- spec 不以当前覆盖数字冒充稳定架构。
