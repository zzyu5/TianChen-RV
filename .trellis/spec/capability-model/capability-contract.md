# Capability Contract

> ⚠⚠ **读前必读 — 本契约的 [S-1] 事实 shape 是 TARGET schema、当前代码尚未全部消费（[GAP-P4-SCHEMA-DIVERGENCE]）。**
> `schema/capability.schema.v1.json` 的 `$meta.note` 明文（2026-07-12 核）：
> `provenance` / `trust` / `subclass` 字段、`kind` **闭合枚举**、命名空间化 `params`、
> 统一 operand 角色、可序列化插件签名 —— 这些**"grep=0 in code today; code and schema.def
> are intentionally divergent"**。分叉是被追踪的 conformance gap（E4/E5/G6/P2 roadmap 收），**不是 v1 缺陷**。
> **对接入适配者的意义**：照 [S-1] 写出的能力事实，其**结构键**（`id` / `kind` / `implies` / `conflicts`
> / `params`）**当前代码实际消费**（`CapabilityDescriptor` + `TargetCapabilitySet` 查询，见下 §Relations/[S-2]、
> profiles.md）；但 [S-1] 声明的**元字段**（`provenance` / `trust` / `subclass` 及闭合 `kind` 的**枚举校验**）
> 属**目标态、代码今日不读**。落地一个家族时先按 §Relations 的**实际查询 API**（`lookupProviderByID` /
> `isCapabilityAvailableBySymbolName` / `--tcrv-check-capability-requires`）核对哪条键真被消费，
> **别按整份 [S-1] shape 当既成契约白写元字段**。能力事实的**实例**落 `lib/Plugin/<Fam>/<Fam>ExtensionPlugin.cpp`
> 的 `getCapabilities()`（C++，非 schema JSON；见 [plugin-protocol/extension-plugin-integration.md](../plugin-protocol/extension-plugin-integration.md) §Landing points）。

Capability 是系统第一对象（见 [core-invariants](../architecture/core-invariants.md) I1）。本文件定义它的来源、形态、关系和验证职责——这是 N1（RISC-V 扩展异构性作为 first-class capability IR）的契约。

> ⚠ **N1 是 substrate，不是独立卖点**（见 [index](../index.md) 的 Novelty 段）。把能力建成可查询对象**本身** ≈ DLTI / IREE `#hal.executable.target` / TVM Target / LLVM `SubtargetFeature`+TTI / FODA `requires`/`excludes` 已做的工程。N1 的**唯一新意在合取 = C1**：它是**跨 family 复用的同一 fact-set**，由 N2 的第二 family 证（→ C1）、由 N3 兑现（→ C3′）；抽掉跨 family 复用，N1 就塌回纯工程。N1↔C1/C2/C3′ 的权威 bridge 只在 [index](../index.md) 声明一次，本文件引用不重述。本契约定义 capability 的形态与职责，但**不**因此把"建模能力对象"本身当贡献。可迁移的表述是 *mechanism* 而非 *discovery*：同一 relation-bearing schema 统一 compile-time variant generation 与 runtime dispatch guard、并跨 compute-paradigm 边界不改地复用（不是"扩展会 layer"这个 `SubtargetFeature.Implies` 已建模的观察）。

## 它必须影响什么

capability 对象必须能被 C++ MLIR pass 和插件查询，并影响：启用哪些插件、variant 提议 / 合法性、tuning 空间、cost 输入、selection、runtime dispatch、emission 路径选择、lowering 诊断、fallback 需求。它不是 prose / 裸字符串 / JSON-only 记录 / Python dict（见 I1）。

## Capability 来源

- **ISA**：`rv64`、`rvv`、RVV version（RVV1.0 / RVV0.7 / xtheadvector，作为合法性轴：fractional-LMUL / ta-ma policy 的可用性随版本变）、`zvl*`、`zvfh`、`zvfbf*`、`V`、`ime`、vendor custom opcode、future matrix/custom ISA。
- **Microarchitecture**：core count、VLEN、cache、内存带宽、preferred LMUL、dtype throughput、NUMA/拓扑、thread runtime 可用性、toolchain 交链。
- **Runtime/offload**：accelerator 存在性、runtime 名、ABI、PCIe/SoC 模式、支持的 offload op 集、model 格式、host-device 传输代价、async 支持。
- **Toolchain**：march/mabi 事实、LLVM RVV 可伸缩向量支持、intrinsic 支持、builtin 支持、inline asm 许可、vendor header、patched compiler、runtime lib 可链接性。

## [S-1] 结构化事实模型

> **消费 vs aspirational 分层（[GAP-P4-SCHEMA-DIVERGENCE]，见顶部 banner）**：下列字段中
> `id` / `kind`（作为字符串键）/ `implies` / `conflicts` / `params` 的**结构与查询**当前代码消费；
> `provenance` / `trust` / `subclass` 与 `kind` 的**闭合枚举校验**是 schema.def 声明的目标态、
> **代码今日 grep=0**。适配者写事实时前者按契约、后者知其为 aspirational。

capability 由**结构化事实条目**承载，不是自由字符串。每条事实的稳定形态：

- **`id`** —— 命名空间化的稳定标识（`rvv.v` / `rvv.zvfh` / `ime.vmadot` / `scalar.zbb` / `uarch.vrgather_slow` …）。id 是权威键，benchmark 名 / 日志 / provider 身份 / 性能测量值不得变成 id（I9）。
- **`kind`** —— **闭合枚举** `{isa_ext, sub_ext, uarch, policy}`（不是开放字符串）。这是事实的**类别轴**，与 family **正交**：它区分「ISA 扩展 / 子扩展 / 微架构 / 策略」，**不**区分「RVV / IME / Sophgo」。核心 pass 因此仍**绝不**按 family-name 分支（I3）；kind 是有限类别，family 归属由插件注册与 capability 查询解析。前述来源映射进四值：ISA 扩展 → `isa_ext`；子扩展（zvfh / zbb …）→ `sub_ext`；uarch / memory / VLEN 事实 → `uarch`；toolchain / runtime-offload / thread-runtime 的**可用性**与 build/permission 门 → `policy`（offload accelerator 的**计算所有权**仍归 offload 插件，不进 kind，见 I2）。Logical Shape 示例里 descriptor 上的 `isa-vector` / `runtime-offload` / `isa-matrix-vector-backed` 是子分类标签，各自归约到上述某个 `kind` 值。
- **`implies: [id]`** —— 见 [S-2]。
- **`conflicts: [id]`** —— 见 [S-2]。
- **`params`** —— **命名空间化的结构化字段**（取代隐式自由字符串键）：`vlen`、`elen`、`sew_set`、`lmul_budget`、`vreg_count`、`cacheline`、`ime.tile(...)` 等，各带类型。params 的分层语义另受 Parameter Layering Rule 约束。
- **`provenance ∈ {hwprobe, cpuinfo, vendor_table, manual}`** —— 事实的来源渠道。
- **`trust ∈ {measured, declared}`** —— 事实的可信级别（实测 vs 声明）。provenance/trust 是事实的一等字段，供证据线（I8）区分实测与声明，**不**参与 compute 语义。

kind 保持闭合枚举，新增类别属于 schema 演进（[S-5]/[S-6] 的附加式 minor），不是运行期自由扩展。

## Parameter Layering Rule

参数含义必须分层；跨层只能经显式 compiler object 或 ABI surface 并声明新含义。这是 RVV 线最易被污染的地方。

1. **硬件事实 / target capability** —— VLEN、raw VLENB、ISA/profile 事实、hart/core 数、toolchain 可用性、probe 证据、provenance。约束 legality 与 selection。
2. **编译期 variant config** —— SEW、LMUL、tail/mask policy、unroll、selected lowering 策略。由 plugin-owned variant metadata/选中 config/tuning 提议，须对 target capability 校验；影响生成代码的 config 必须结构化进 typed body 或被 realize 消费（I5）。
3. **运行期 SSA / control 值** —— AVL、vl、指针实参、长度 `n`、`rvv_available`、dispatch guard 参数。存在于真 IR 或 ABI surface，不是 target capability 也不是编译期常量。
4. **legacy fixture 参数** —— 只能描述历史/fail-closed slice 的 selected-path metadata，**不得**冒充 tensor shape、全局问题规模、AVL/vl、source authority、production 输入或 artifact authority。

emission plan / manifest / 诊断 / artifact **不得**声称某参数"已 IR 建模"，除非真 IR 有对应 attribute / type / SSA 值 / region 参数 / ABI 参数。

## Logical Shape

target capability 表示为结构化的 target-level / module-level MLIR attribute。参考形态：

```mlir
#tcrv.target<
  arch = "riscv64",
  isa  = ["i","m","a","f","d","c","v","zvl128b","zvfh"],
  uarch = { cores = 64, vlen = 128, has_openmp = true, cache_model = "target_specific" },
  extensions   = [ #tcrv.ext<"rvv", kind = "isa_ext", status = "available", provenance = "hwprobe", trust = "measured"> ],
  accelerators = [ #tcrv.accel<"sophgo.bm1684x", kind = "policy", subclass = "runtime-offload", mode = "pcie", runtime = "vendor-c-abi", provenance = "vendor_table", trust = "declared"> ],
  toolchain = { llvm_rvv = true, rvv_intrinsic = true, inline_asm = true, vendor_runtime_link = true }
>
```

（上面的 `cores=64 / vlen=128 / sophgo` 是**示例值**，不是 durable 的项目事实；具体目标参数属于 profile，见 [profiles](./profiles.md)。）IME 可加 `#tcrv.ext<"spacemit.ime", kind = "isa_ext", subclass = "isa-matrix-vector-backed", ...>`（`kind` 取闭合枚举值 `isa_ext`，`isa-matrix-vector-backed` 是子分类标签，见 [S-1]）。

## Relations

capability 带 `provides` / `implies` / `conflicts` 三类关系（first-class 描述符字段，不是 property-map 项，也不是 prose）。

- **require**：variant 声明所需 capability（`requires = [@cap, ...]`，`FlatSymbolRefAttr` 指向 kernel capability scope 内的符号）。provider 是直接 `tcrv.exec.capability`、带 `id`+`kind` 的 kernel-local `tcrv.exec.target`、kernel `target = @profile` 指的那个 module-level profile、以及该 profile 经 `capability_providers = [...]` 显式组合的 providers。kernel 只看自己引用的 profile + 其命名 providers + kernel-local providers；id 在该 scope 内唯一。
- **provide**：`provides = ["..."]` 是 capability id（不是符号名、不是 prose）。`lookupProviderByID(id)` 先解析 exact id；无 exact 时可由 available 的 `provides`/`implies` 满足。于是 `id="rvv.profile.rv64gcv", provides=["rvv"]` 能满足要求 `rvv` 的提议，同时 exact `id="rvv"` 在场时保持直接覆盖。**profile 只作 provider（带 `provides`、不带 capability-fact `kind`）**——profile 不是一条带 `kind` 的叶子事实，而是装载期**展开成规范化事实集**的容器；`declared-instance-hash`（[D-2a]）对**展开后的事实集**取，故 profile 写法与语义等价的显式事实列表**哈希相同**。
- **imply**：`implies = ["..."]`（如 `rv64gcv implies rvv`、`zvfh implies fp16 向量算术（受 toolchain 支持约束）`）。经同一 relation-aware lookup 暴露。这是 bounded 的决策路由，不是完整 capability lattice 或推断引擎。
- **conflict**：`conflicts = ["..."]`（如"要 vendor runtime 但无 runtime lib"、"要 inline asm 但 build policy 禁止"）。`--tcrv-check-capability-requires` 用 bounded 双向冲突查询作 legality gate：静态 variant / dispatch fallback 在所需 capability 与另一 available capability 冲突时 fail closed；dispatch case 只有携带 typed `runtime_guard_required = true` 时才能引用冲突需求（记录保护面，不解析 printable 串）。这**不是**完整 conflict solver / lattice / provider ranking。
- **dispatch condition**：runtime/shape 相关条件成为 dispatch 谓词（`if runtime_available && large_shape -> offload；else if rvv_available -> rvv；else -> fallback`）。

### [S-2] 关系语义：加载期传递闭包 + fail-closed + 未知=假

- **implies 传递闭包在加载期计算**：`implies` 关系的传递闭包在 capability set 装载/构造期一次性物化，查询走物化闭包，**不**在每次查询时只查一层再让调用方补链。`rv64gcv implies rvv`、`zvfh implies fp16 向量算术` 这类链在闭包里对查询者直接可见。这是 bounded 的决策路由，不是完整 capability lattice 或推断引擎。
- **conflicts 命中即 fail-closed**：任一被要求的 capability 与另一 available capability 冲突时**默认拒绝**（`--tcrv-check-capability-requires` 的 bounded 双向冲突查询）；dispatch case 只有携带 typed `runtime_guard_required = true` 时才能引用冲突需求，且只记录保护面、不解析 printable 串。
- **未知事实 = 假**：schema 未声明 / 未 available 的事实一律视为**不满足**（缺省拒绝，绝不缺省放行）。这与「missing status 视为 available」不冲突——后者只在事实**已在场**时解释其 status 字段；事实**根本缺席**时判假。

> 查询 API：`TargetCapabilitySet::buildFromKernelChecked(KernelOp)`（带诊断的构造，duplicate id/symbol fail closed）；`buildFromKernel` 仅用于已验证上下文。pass `--tcrv-check-capability-requires`。按符号名与 id 双向查询；relation-aware lookup 支持 exact/provided/implied，exact 在场时权威。missing status 视为 available；`status` 优先于 `availability`；`unavailable`/`disabled`/`missing` 视为不可用。**核心代码不解释具体 target-family 的 status 语义**（I3）——`if (target.hasRVV())` 是错的，要走 `capabilities.isCapabilityAvailableBySymbolName(...)`。

## [S-3] 探针只写事实

hwprobe / cpuinfo / 厂商表适配器**只产出 [S-1] 事实条目**（带 `provenance`/`trust`），绝不携带路由/选择决策。任何「探到 X 就走 Y」形态属违规——它把一个隐藏分支重新引入探针层，绕过 capability 查询与插件 legality（I3）。探针也不得伪造 SEW/LMUL/tail-mask 这类 plugin-selected 编译期 config 事实（那属 Parameter Layering Rule 第 2 层，不是硬件事实）。从 probe 证据到 compiler-visible capability 的权威转换是 plugin-local C++ capability profile 校验 + `TargetCapabilitySet` 填充，不是 Python 数据结构（I6，另见 [profiles](./profiles.md)）。

## [S-4] uarch 作为事实

微架构信息是 [S-1] 事实（`kind = uarch`），不是散落在核心里的常量或 if 分支。粗粒度 uarch 事实（如核数）驱动决策已属常规；契约进一步要求把**每核 quirk 表**建成事实：诸如「高 LMUL 下 vrgather 慢」「段加载收益」这类逐核性质，按核（如 C908 / X60 各一张）声明成 `uarch.*` 事实 + `params`，供选择/成本层（[SEL-*]/P6）**键控**。**走表不走 if**——核心与插件按 uarch 事实查表，绝不为具体核名写 `if (core == "...")` 分支（否则复现 family-name 分支的病，违 I3 精神）。uarch 事实约束 legality/selection，其 `count`/`vlen` 等是硬件事实，不是 runtime thread 数、dispatch guard、tensor shape 或 AVL/vl（Parameter Layering Rule 第 1 层）。

## [S-5] schema.def 声明工件

schema.def 是 capability schema 的**声明式工件 + 哈希对象**：它声明 capability 的**稳定形态（shape）**，是 [S-6] 操作门 [F-2′]（家族接入 PR 系列 diff ∩ schema.def = ∅）与报告门（规范化序列化 → SHA256 + RFC 版本日志）的锚。schema.def 声明**恰好六项**：

1. **事实记录字段与类型** —— [S-1] 的 `id` / `kind` / **`subclass`** / `implies` / `conflicts` / `params` / `provenance` / `trust`，含 `provenance ∈ {hwprobe,cpuinfo,vendor_table,manual}`、`trust ∈ {measured,declared}` 枚举。`subclass` **保留原始类目**（如 `subclass = "toolchain" / "runtime-offload" / "isa-matrix-vector-backed"`）：`kind` 收口为四值闭合枚举时**信息不丢**，将来若拆枚举是**机械操作**（此 field 的加入按 [S-6] 记 minor / 附加式演进）。
2. **kind 闭合枚举** —— `{isa_ext, sub_ext, uarch, policy}`（[S-1]）。
3. **关系类型表** —— `implies` / `conflicts` 的关系类型及其语义标注（带类型元数据，不是扁平三列表）。
4. **params 命名空间声明** —— `vlen` / `elen` / `sew_set` / `lmul_budget` / `vreg_count` / `cacheline` / `ime.tile(...)` … 的命名空间与类型（[S-1]）。
5. **插件接口签名的可序列化形态** —— `ExtensionPlugin` 准 ABI 的**声明式可序列化签名**（见 [plugin-protocol](../plugin-protocol/interfaces-and-registry.md)；接口冻结属 [P-1]，改动需 RFC）。
6. **路由描述符操作数角色词表** —— 路由描述符的操作数角色闭合词表（lhs/rhs/out/n、buffer/scalar、runtime count 等角色），N-operand 路与块点积角色**统一**结构化，不留表内自由字符串。

**不入 schema.def 的 shape**：具体事实行、`params` 的取值、插件内部代码、测量库、模式注册表条目——这些是内容不是形态，改它们不触 schema.def。schema 不内置成本模型：成本住测量库（按 instance-hash 键控），schema 只答「能不能 / 是什么」（[S-7] 非目标）。schema 的**附加式演进**（新增可选字段 / 新 kind 值 / 新关系型）记 minor 并标注「extension not modification」；家族接入**要求**修改 shape 才是被 [F-2′] 挡下的违规。

## Verifier 职责

capability verifier 检查：variant `requires` 被 target capability 满足或被 dispatch 守护；插件运行前通用 variant/capability 结构良构；extension-family legality 委派给插件 hook（不被 core 按 family 名硬编码，I3）；extension op 只出现在其插件 legality 接受了 capability 需求的 variant 内；selected emission 路径被 toolchain capability 支持；runtime ABI 声明完整；dispatch/fallback 覆盖不可用条件。verifier **不**证明数值正确性，只挡非法 target-feature 使用和缺失的执行前提。

## [S-8] 家族语义 = 能力声明 + 所有权

**family = 能力声明 + 所有权边界，不是指令密度。** 一个 extension family 由「它声明并拥有哪些 capability 事实 + 拥有哪些计算语义（I2）」界定，**不**由「它注入多少条自定义指令」界定。由此契约允许**「能力门家族」**：其 owned 内核主体用基础指令，家族事实充当**门（gate）与可选加速子事实**，而非新指令密度。典型即向量缺席的**标量家族**——它是合法的独立（independent）家族（其内核主体走标量路径），bit 操作扩展（zbb/zba/zbs）作为它的**子事实 / 能力门**。

子事实粒度**对称**成立：`zvfh` 之于向量家族 ≡ `zbb`/`zba`/`zbs` 之于标量家族——都是同一家族下的 `sub_ext` 事实（[S-1]），经同一 [S-2] 关系语义参与 legality。独立性靠 [S-2] 传递闭包的**范围**保证：标量家族的 implies 闭包只排除 `rvv.*`，per-block scale 折叠等能力若需要则**显式声明为能力事实**、不破坏独立性。术语上「向量核上的矩阵范式」用 **paradigm**、家族独立性用 **family**，integrated（复用向量寄存器堆）/ independent-attached（独立寄存器堆）不混用（[L-2]/[L-3]）。

## 实现

capability model 属 primary stack（I6）：C++ 数据结构 + MLIR attribute/type/interface；IR 表示用 TableGen/ODS；CMake target；lit/FileCheck 覆盖 parse/print/verify/诊断；文本测不了的 helper 语义补 C++ test。Python 只 probe/解析/出报告，**不**作为 capability 关系、插件可用性、legality、selection、dispatch 或 lowering 诊断的真值实现。
