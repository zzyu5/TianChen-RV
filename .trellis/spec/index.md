# Weft-RV Durable Specs —— 根地图

.trellis/spec 保存 Weft-RV 的稳定设计契约、测量规则、证据解释和问题治理。它是项目规范入口，但不是 GPT 的强制工作流或开工许可系统。

## 权威模型

不同对象有不同 authority：

| 对象 | 主要 authority |
|---|---|
| 当前用户范围与优先级 | 用户最新明确指令 |
| 实现行为 | 当前代码与测试 |
| 稳定设计契约 | canon 与 architecture |
| 测量事实 | experiments/master、experiments/runs、experiments/runs.log |
| 证据解释与路径 | evidence |
| 已知问题 | issues |
| 多阶段计划与恢复上下文 | 可选 Trellis task 或当前对话计划 |

task、旧 goal、旧简报和历史报告不能覆盖当前代码事实，也不能覆盖用户最新范围。

## 三条文档纪律

1. **现行规则优先**：spec 正文写当前契约，不把历史战役过程混入规则。
2. **单一住址**：同一规则或事实只有一个主要 authority；其他位置链接引用。
3. **稳定引用**：跨文件使用相对链接、标题锚或条目编号，不使用易漂移行号。

## 项目定位

Weft 是 high-level MLIR 之后的、能力驱动且可扩展的 MLIR operator compiler / execution-layer software stack。它让 operator、format/layout、target capability、execution mechanism 与 backend family 通过局部 typed extension 接入，再由可执行专家知识构造专化 kernel。RISC-V ggml/llama.cpp 风格量化推理是当前旗舰 reference realization 和主要压力域，不是系统类别的上界。

它不是：

- 通用图编译器；
- 新的高层 tensor/tile IR；
- 每个硬件一套互不相关 backend；
- descriptor 或 metadata 驱动的代码模板系统；
- 通用在线 autotuner；
- 已解决动态 sparse/MoE 的 runtime policy。

稳定范围见 [architecture · 系统定位与边界](./architecture/系统定位与边界.md) 和 [canon · 非目标](./canon/非目标.md)。

## 理想论文与项目：两柱不变

### 柱一：能力驱动、类型化、可复用的扩展模板

新格式、新能力、新机制、新后端和新知识应被局部 typed owner 吸收：

- 格式/机制事实进入 g；
- 目标能力进入 c；
- 有限静态场景进入 ω；
- 新 family 通过插件五件套和 typed body 接入；
- core/common 不按 family 名分支；
- 新板不要求逐格式修改 emitter。

### 柱二：高性能知识的可执行利用

解析知识必须真实参与：

- typed candidate/plan 构造；
- legality 与资源边界；
- capability/context prior；
- 有限合法候选中的选择；
- final typed body construction 与 mechanical emission；
- 负结果、fallback 和适用域。

qualified measurement 可以在解析合法域内修正排序，但不能创造 candidate、扩大合法域或定义 compute。runtime data profile 只有在 observer、开销、策略和真实 workload 都存在时才进入；当前属于 future。

两柱、六律和贡献组织的当前研究表述见 [canon · 暂定科研主张](./canon/暂定-科研主张.md)。可执行公式/选择契约的唯一工程正本见 [architecture · 变体流水线](./architecture/变体流水线.md)。

## 软件主链

~~~text
MLIR operator / kernel-level input
  → plugin-local typed facts g + bounded context ω
  → canonical capability c
  → catalogued plugin-local formula / construction
      · typed candidate or plan
      · legality/resource bounds
      · analytic prior
      · optional measurement key
  → bounded selector
      · qualified winner if still legal
      · otherwise analytic prior or named fallback
  → selected typed body
  → plugin route provider
  → common EmitC / target artifact
  → real hardware evidence when claimed
~~~

所有 production operator entry 都必须经过该主链；只有一个确定实现的路径采用 deterministic single-candidate construction 和 honest-null 轴，不得静默绕过。这个主链不要求大一统 Formula IR，也不要求每个职责成为独立物理子系统。公式集合、横向 cutover 与覆盖正本见 [architecture · 公式层与覆盖](./architecture/公式层与覆盖.md)。

## 六个 spec layer

| Layer | 回答什么 |
|---|---|
| [canon](./canon/index.md) | 核心不变量、非目标、性能措辞、正确性与暂定科研主张 |
| [architecture](./architecture/index.md) | 系统工位、typed facts/capability、插件、公式/选择、body 与 emission |
| [measurement](./measurement/index.md) | runner、板、行键、对手、正确性、计时和写入目的地 |
| [evidence](./evidence/index.md) | 主张或边界由哪些工件支持、如何解释 |
| [governance](./governance/index.md) | 用户范围、spec/task/issues 住址和可选工作方式 |
| [issues](./issues/index.md) | 稳定编号的问题、边界和待施工项 |

## 推荐阅读顺序

普通修改：

1. 根 README；
2. 本文件；
3. [canon · 核心不变量](./canon/核心不变量.md) 与 [非目标](./canon/非目标.md)；
4. 与修改相关的一个 architecture/canon 文件；
5. 需要测量时再读 measurement；
6. 需要查债务时再读 issues。

无需为了局部修改读取全部 task、事故档案或历史 campaign。

## task 的使用

Trellis task 是可选规划工具。

推荐使用：

- 多阶段或跨多日；
- 多 agent/worktree；
- 跨层接口与大范围迁移；
- 正式实验 campaign；
- 用户明确要求任务树。

不强制使用：

- 单阶段、范围明确的文档/spec 更新；
- 小型可逆代码修改；
- 只读诊断；
- 当前对话计划已经足够。

详细规则见 [governance](./governance/index.md)。

## 测量入口

官方 runner：

~~~text
tools/bench/bench <op> <format> --board <board> --engine <engine> --regime <regime>
~~~

稳定目的地：

- experiments/master/
- experiments/runs/<run-id>/
- experiments/runs.log

runner 必须：

- 歧义行键 fail-closed；
- 正确性先于计时；
- 只调用 tools/bench/cells 下声明的 cell harness；
- unsupported cell 具名失败；
- 所有持久写入通过目的地 guard。

自检入口：

~~~text
tools/bench/bench --self-test
~~~

本文件不写当前 PASS 数；当前输出以命令实跑为准。

## 当前工程推进方向

当前已经建立 formula/construction authority 的公共底座：production construction
entries 由 registry-derived catalog 关联 family-local typed owner。RVV
quantize/dequantize final body、repack loop-order/main-term schedule 与通过
`TunableScheduleOpInterface` 发现的通用 schedule 均在 emission 前完成构造；已有完整
tuple 只接受合法性校验，缺失 tuple 才构造，partial/illegal tuple 直接拒绝。emitter 不补
schedule default，旧 repack strip-width materializer 已退出。

这仍不是 project-wide authority cutover 已完成的证明。当前尚有一个公开可执行的
low-precision Gearbox/pre-realized-body surface：它会先写 candidate、selection、resource 与
审计镜像，再由 selected-body realization 解释。即使默认 front door 暂无 caller，只要该
pass 与 IR 仍是公开接受面，就必须在本轮横向重构中归入同一 formula → legality → thin
selector → final typed body 链，或明确退役；不能用“测试/手工入口”排除在分母外。

公共底座与上述切换也不表示每个 `ConstructedWeak` leaf 已经完成强义重建。关闭剩余
公开 authority inversion 后，应在同一横向结构上推进：

1. 横向审查现有 leaf 中仍未提升为 mechanism、typed parameter 或 formula 的知识；
2. 让更多实例通过 delete-leaf reconstruction，而不是继续增加 opaque leaf；
3. 补足 decisive `g/c/ω` 的因果测试、真实 capability transfer 和 semantic boundary；
4. 保持新增 operator、format、capability、mechanism、formula、residual 与 backend 的
   owner/入口清楚，不重新形成双 authority；
5. 在统一路径上继续 deployed ggml、代表性强对手、e2e 与正负性能证据。

强 reconstruction 可以按真实 mechanism 逐步增强，但 formula authority 不能退回按五类
dequant、单一 operator 或第二 family 分裂的新旧双轨。

当前进度与数字不写入根 spec；实现事实以代码和测试为准，性能事实以
master/result tables 和 run lineage 为准。

## 入口文件

根 README 与本文件共同构成项目入口。根 AGENTS.md 已退役，避免维护第三份重复、易漂移的代理说明。
