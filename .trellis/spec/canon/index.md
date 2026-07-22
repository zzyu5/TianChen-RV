# Canon Specs — 已定法条

> **版本**：v5
> **修改途径**：本层版本化，**修改唯一途径 = 用户裁决**；agent 可提案（入 [issues](../issues/index.md)），不可自改。
> （版本载体依 [governance · trellis 卫生](../governance/trellis卫生.md) 的「版本号载体 = 层 index 文件头」条**首次建立于 2026-07-17**；此前本层无载体 = **卫生欠账**，非「本层此前无条文」。**v1 = 建载体时的在册状态**，**不追溯编号历史修改**。）
> **v4**（2026-07-20·用户授权 B1 control-plane）：测量事实与正式发布分立；immutable run 只有经 correctness、lineage、freshness、结构化 T-N 资格后，才由 recon-only publisher 进入 canonical master。measured / master-qualified / selection-valid 三轴禁止互推。
> **v5**（2026-07-23·用户接受 V2/GPU 方向）：系统边界明确为 post-graph、pre-schedule
> automatic operator-to-kernel compiler；GPU 是第二 execution paradigm 目标，不是 current
> EmitC emitter。Canonical problem、family binding 与 artifact-neutral construction 成为 GPU
> 前置边界；两柱、六律、原主公式与 thin selector 保持不变。

本层是本项目**全部已定法条与终裁的唯一合订**：判断"某主张能不能立、某数字能不能报、某格算不算赢"时的**法源**。

其他层各有分工，本层**不复述**它们：测量的执行程序住 [measurement](../measurement/index.md)、系统结构住 [architecture](../architecture/index.md)、证据工件指针住 [evidence](../evidence/index.md)、队列与流程纪律住 [governance](../governance/index.md)、未决问题住 [issues](../issues/index.md)。

## 本层的地位与用法

1. **正文 = 现行法**。凡本层写下的，即当前有效；照办不需追问来由。
2. **本层不复述别层**。凡某条的权威全文住兄弟层，本层只给规范性一句 + 链接。
3. **重大定义修改 = 用户裁决 + 版本号递增**。工程状态、架构落点与历史机制清理不得借机重新解释两柱、六律或贡献编号。
4. **[暂定-科研主张](./暂定-科研主张.md) 与本层其余文件法律地位不同**：那是探索中的科研表述，不是定法，**不得据以判定胜负或数字可否上报**。
5. **`conflicting` 条目不在正文**，见 [待裁](./待裁.md)。待裁期间按 [governance](../governance/index.md) 的延后裁决制走保守默认，**继续推进不阻塞**。
6. **跨文件引用禁行号**：一律用相对链接 + 条目编号（`[K-5]` / `I3` / `ISSUE-070`）。行号会随改写腐坏成假引文。
7. **禁副本**：一条规则只有一个家。两处写同一条 = 缺陷；合并留一 + 链接。

## Pre-Development Checklist（判断提示，不是 gate）

- [ ] 这项工作推进柱一、柱二、工程地基或真实性能中的哪一项？（[governance · 思维准则](../governance/思维准则.md#一主干纪律)）
- [ ] 要报的性能数字绑齐了 [L-1] 九要素了吗？（[成色与措辞](./成色与措辞.md) [L-1]）
- [ ] 这条主张属 kernel 账还是系统账？双方编译器对称吗？（[测量判据](./测量判据.md) [L-9]、[对手与档位](./对手与档位.md) [CASE-COMPILER-ASYMMETRY]）
- [ ] 对手判档有符号级证据吗？判不清标 UNRESOLVED 了吗？（[对手与档位](./对手与档位.md)）
- [ ] 要写"由机制构造"吗？是强义还是弱义？（[部署与构造语义](./部署与构造语义.md) [L-8]）
- [ ] 新 kernel / 新 Plan 走 G1→G2→G3 了吗？没跳级吧？（[测量判据](./测量判据.md) [VERIFY-LADDER]）
- [ ] 性能立项给了 Amdahl 同域传导预估吗？（[测量判据](./测量判据.md) [L-11]）
- [ ] 动的是分母 / 队列序 / 头条口径 / canon 措辞吗？→ 硬冻结，只登记不执行（[账目与分母](./账目与分母.md)）
- [ ] core / common 里没有 family-name 分支吧？（[核心不变量](./核心不变量.md) I3）

## Guidelines Index

| Spec | Description |
|---|---|
| [核心不变量](./核心不变量.md) | I1–I9 全项目复用硬规则 + 附加硬规则（[L-\*]/[K-\*]/[S-\*]/[F-\*]/[SEL-\*]/[NG-\*]）的去向表 |
| [账目与分母](./账目与分母.md) | 单分母制 · any-board 不增格 · 四账禁互推 · UNRESOLVED 清偿 · 落表纪律 · 覆盖指标 · 硬冻结四类 |
| [对手与档位](./对手与档位.md) | 对手三档法（**档位**）· **对手类词表**（**角色** · factory-dispatched = 唯一 beat 基线）· 对手 = 部署事实 · 便宜档禁称硬赢 · 计数≠强赢 · 输赢同权 · 物理墙收严 · 编译器对称 · 单世界 clang-18 |
| [成色与措辞](./成色与措辞.md) | [L-1] 措辞门 · [NG-4] · [L-6] 成熟度三重区分 · 快照纪律 · 定调锁 · [L-12] 性能三层图 · 域外段 |
| [测量判据](./测量判据.md) | [PERF-1] 验收门 · [PERF-2] · **格 schema 二分 + 状态枚举（闭合）** · T-N 噪声地板 · 构建保真三重 · [L-9] 双账本 · [L-11] Amdahl · 传导会计 · [VERIFY-LADDER] · **T-X X-SCALAR 真板 enablement 证据表** |
| [正确性与证书](./正确性与证书.md) | [K-5] 正确性门 · [K-5b] 证书三要件 · [K-5c] ZERO-MODEL · 对拍一步三验 · 空心门禁令 · **硬件证据通用契约** · **断言来源区分（HARNESS 源级 vs 运行期）** |
| [部署与构造语义](./部署与构造语义.md) | [L-10] 部署的≠证过的 · [L-8] 强/弱义构造 · [K-10] 结构级/参数级 · [DISCRIMINATOR] · [MOD-COST] 双 regime · 能力措辞两层 |
| [缺口与认输](./缺口与认输.md) | [K-6]/[GAP-1] 缺口关闭环 · 认输门槛（零未攻认输）· [K-7] 首次发射律 · [X-0] 接入三问 |
| [能力模型与插件协议](./能力模型与插件协议.md) | [S-\*] 能力 schema · [F-\*]/[P-\*] 插件与家族判据 · family 准入边界 |
| [构造强度、mechanism 与选择归因](./覆盖状态机与选择归因.md) | [K-4] strong/weak 构造 · [PAT-\*] typed mechanism · [SEL-\*] 薄选择器 · [D-4] 非权威归因 |
| [非目标](./非目标.md) | [NG-1..NG-10] 越界即标记 |
| [待裁](./待裁.md) | 正面冲突或悬而未决的条目 + 各自保守默认（**无法条效力**） |
| [暂定-科研主张](./暂定-科研主张.md) | 当前两柱、六律、C1/C2/C3 组织、性能三层与理想 Evaluation |

## Quality Check

- 任何"击败 / beat / outperform"措辞出现时，[PERF-1] 是否全绿？未全绿即违 [NG-4]。
- 任何性能数字是否绑齐 [L-1] 的九要素，并标明 kernel 账 / 系统账？
- 任何"由机制构造"是否通过 delete-leaf reconstruction？否则须写 `constructed-weak`，
  不能用 provenance 清单或 catalog 数量替代。
- 任何新增检查是否有负控证明它能变红？变不了红的检查判死清除（空心门比没有门更坏）。
- 任何"X 不存在"断言是否避开了 `head` / `grep -A<N>` 这类有界窗口？展示可截断，判断不许。
- 任何数字是否附机打行清单工件、并住主表而非新建报告体系？
- 本层任一条文是否与别层重复了？重复即缺陷——合并留一 + 链接。
