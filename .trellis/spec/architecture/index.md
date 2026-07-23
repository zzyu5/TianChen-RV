# 架构（Architecture）

> **版本**：v7
> 本层描述 Weft 的稳定结构、当前 Weft-RV 实现落点和 V2 目标责任边界。当前事实由代码复核；目标设计由用户裁决后版本化进入本层。

## 读法

每项内容应明确属于：

| 标签 | 含义 |
|---|---|
| **【现状】** | 当前代码真实存在，可用路径或谓词核查 |
| **【目标】** | 尚未完全实现的设计方向 |
| **【定法】** | 实现与目标都必须遵守的稳定边界 |

现状直接以当前代码和相关测试核对；不把历史 census 数字、task 表格或引用格式变成
日常修改纪律。需要正式论文或性能主张时再按对应 measurement/evidence 规则固定证据。

科研价值组织见 [canon · 暂定科研主张](../canon/暂定-科研主张.md)；本层只负责把它变成可实现、可测试的结构。

## 改动前的架构判断

以下问题用于发现职责倒置，不是 task、RFC 或开工门禁。普通代码修改直接按当前
代码和测试推进；只有触及稳定契约时才同步修改本层。

- [ ] 仍是 post-graph、pre-schedule 的 operator-to-kernel compiler，而非新增高层 tensor/tile IR？
- [ ] canonical problem 的 `S/g/ω` 是否语义充分但尚未携带 owner schedule？
- [ ] target/domain 是否在 proposal/selection 前绑定，而非由 origin 或 artifact emitter 反推？
- [ ] 所有 candidate origin 是否声明属于 bound domain，different-domain candidate 是否 fail closed？
- [ ] 系统身份是可扩展 MLIR operator compiler，RISC-V 只是旗舰实例而非 core 上界，GPU 也不是一个末端 emitter 吗？
- [ ] 每个 registered/direct production operator entry 都进入 catalogued formula/construction 阶段吗？
- [ ] g、c、ω 的来源和 owner 明确吗？
- [ ] g/c/ω 是否由不同 typed owner 提供，并在 owner-local formula/construction 中受控汇合？
- [ ] formula/construction 产生 typed plan、candidate、legality 所需条件或 prior，而不是 C 字符串？
- [ ] measurement 只在合法候选中选 winner，没有创造 compute？
- [ ] formula/selector 的最终结果是否在 emission 前直接成为 final typed body，而非经历 stamp 生命周期？
- [ ] construction completion 是否独立于 EmitC、`emitc.func` 或任一具体 artifact？
- [ ] emitter 没有重新读取 format、board、march 或 measurement 做同一决定？
- [ ] core/common 没有具体 domain/owner-name branch？
- [ ] unsupported 或空合法集 fail-closed 或进入 canon 允许的具名 fallback？
- [ ] 当前 registry/call graph 中全部 production entry/caller 已进入统一层，旧 caller、兼容 bridge、第二 dispatcher/writer 已归零吗？
- [ ] 中间迁移状态只存在于未完成 worktree，最终合入没有未迁 operator/owner 吗？
- [ ] runtime sparse/MoE 没有被写成当前实现？
- [ ] 新抽象有第二个真实使用者，或保持 owner-local？

## 本层地图

| 文件 | 内容 |
|---|---|
| [系统定位与边界](./系统定位与边界.md) | 项目范围、主链、工位与 g/c 解耦边界 |
| [执行问题与家族边界](./执行问题与家族边界.md) | V2 canonical problem、target/domain binding、owner construction 与 GPU 前置边界 |
| [能力模型](./能力模型.md) | capability 来源、关系、profile、probe、default/conflict |
| [核心方言](./核心方言.md) | weft.exec execution envelope |
| [插件协议](./插件协议.md) | 插件接口、五件套、locality 和 family 接入 |
| [家族现状](./家族现状.md) | RVV、IME、Scalar、Offload 等 family 的代码事实 |
| [变体流水线](./变体流水线.md) | 公式构造、legality、bounded selection、final typed body 和 realization |
| [公式层与覆盖](./公式层与覆盖.md) | 全项目 formula catalog、`g/c/ω` 消费、production entry coverage 与横向 cutover |
| [Artifact、发射与降级](./发射与降级.md) | artifact-neutral 上层边界与 current EmitC/native realization |
| [实现栈](./实现栈.md) | C++/MLIR 主栈、Python tooling 边界、构建与测试 |
| [退役与单一生产路径](./退役与原子合入.md) | compute authority 唯一性、横向 cutover 与 compatibility 边界 |
| [core-invariants](./core-invariants.md) | 历史兼容指针；I1–I9 正本在 [canon](../canon/核心不变量.md) |

## 当前主改造方向

当前横向 checkpoint 已经完成并继续守住 artifact-neutral construction lifecycle/caller、
exact physical-problem request continuity 与 exact selected-root artifact consumption：所有 live
owner 通过统一 construction-before-artifact seam 返回 exact result，proposal/legality/cost/
construction 使用同一 canonical problem，旧 artifact-side construction、部分 decision
provider/replay 与兼容旁路已退出。这里的“横向完成”只指这些公共切面；RVV source/body
仍有 body-first construction，全部 code-affecting authority 与 mechanism factorization 仍由
active A/B task 收敛，不得概括为所有 production operator 已达到 strong reconstruction。

Artifact-neutral owner construction 横向重基已经完成：canonical problem 后的 owner
construction 不再寄居于 EmitC backend，所有会请求 artifact 的 current owners 与
source/direct/translate/artifact caller 共享 construction-before-artifact 主链；proposal、
legality、cost 与 selected-owner construction 现在验证并传递同一 exact problem，artifact
registry 只保留 selected owner 的完整 variant slice并拒绝 competing roots。这个结论仍不
证明所有 source problem 已 forward-construct body。显式 kernel
domain identity 与 owner membership gate 已在 proposal/selection/construction 前落地；但它
不是 target/profile 驱动的 `BindDomain(t)=(d,C_d)`，后者与 physical source problem coverage
仍是当前横向任务缺口。V2 的下一步仍不是实现 GPU，而是在 RISC-V 旗舰 realization 上横向
闭合 source/domain binding、mechanism factorization、公式因果、delete-leaf reconstruction
与重构后真实性能；不得借此建立大一统 Formula IR 或 universal Plan。完成 A/B 闭环后，
GPU 才沿同一 construction contract 作为完整 domain/owner 接入。

## Quality Check

- 每个 code-affecting decision 有唯一 owner。
- capability、facts、measurement、typed body 和 emitter 的职责不重叠。
- 当前路径与目标路径分开书写。
- 当前状态不从旧 task 或旧文档推断。
- formula/selection 契约以 [变体流水线](./变体流水线.md) 为唯一工程正本。
- source/problem、family binding 与 construction/artifact 分层以 [执行问题与家族边界](./执行问题与家族边界.md) 为正本。
- active-path 退役和原子 cutover 以 [退役与原子合入](./退役与原子合入.md) 为唯一工程正本。
- spec 不以当前覆盖数字冒充稳定架构。
- provider 只交付 capability/route 等机械事实，verifier 只保护 IR/ABI/ISA
  结构；二者都不得定义、补写或重放 compute。
