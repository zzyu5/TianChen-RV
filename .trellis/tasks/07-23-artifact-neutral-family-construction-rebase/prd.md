# Artifact-neutral family construction 横向重基

## 任务目的

当前 Weft-RV 已经完成 construction-before-emission 的 caller/authority 收口，但
construction lifecycle 仍寄居在 EmitC artifact 抽象中。GPU 若现在接入，只能成为
`BackendEmissionRegistry` 的另一个 emitter，或另开一条绕过路径；两者都不是 V2 所需的
construction family。

本 task 在不实现 GPU 的前提下，一次性建立：

```text
canonical operator problem P=(S,g,ω)
  → typed target/profile binding (f,c_f)
  → artifact-neutral family construction
  → legality + optional bounded selection
  → construction-qualified final typed body
  → family artifact driver
       └─ current EmitC/native artifact
```

完成后，future NVIDIA/AMD family 可以沿与 RVV 同构的 construction contract 接入，并使用
自己的 artifact class；不是接收 RVV body，也不是在 EmitC emitter 中重新构造 GPU kernel。

## 完成结果

主体实现固定在 `01f55f3fab63ae923cb61109000f02ee6e7bad01`。所有 current production
family 已迁入 plugin-owned、artifact-neutral construction lifecycle；EmitC registry/driver
只接收 constructed body/plan。RVV recursive typed-body qualification 也属于同一 family
construction cut，direct 与 registry 路径不再分叉。完整 `check-weft` 通过，GPU dialect、GPU
body、GPU artifact/runtime 均未创建。

这只关闭 construction 被 EmitC artifact 绑定的结构问题，不证明 `ConstructedWeak` 已达到
strong reconstruction，也不表示下一步应立即实现 GPU。RISC-V 旗舰 realization 的执行知识
因式分解、formula causality 与重构后真实性能仍须先完成 A/B 横向闭环。
这里完成的是公共 lifecycle/caller/API cutover；Demo、Toy、Template、TensorExtLite 的旧
`emitc_route_mapping`/manifest qualification 仍是 family-local 历史债，已明确移交下一项
A/B 横向任务整体删除，不能据此声称物理协议已经完全 artifact-free。

## 开工代码事实与问题（历史基线）

以下是本 task 创建时的开工事实，用于解释为什么需要本次 cutover；不再代表完成后的
current API：

1. `ExtensionPlugin::constructFormulaPlans(ModuleOp)` 有默认成功实现；目前只有 RVV plugin
   覆写；
2. current live EmitC families 在各自 `TypedBackendEmissionDriver::prepareForConversion`
   内调用 family-local construction/qualification；
3. `convertModuleWithBackendEmitter` 同时负责 construction trigger、DialectConversion 与
   `emitc.func` 完成门；
4. `BackendEmissionRegistry` 以 `moduleHasBackendBody` 扫描 current backend owner；
5. public materialization、direct RVV pass/API、translate 与 target artifact 已经统一经过
   该 current hook，不能在重构时失去这一收口；
6. canonical source/problem 的 `S/g/ω` 仍分散在 source op、front-door pass、loose
   `Operation*` 与 family attrs，缺少统一 ownership/inventory。

问题不在于 EmitC 本身。问题在于“family construction 完成”与“EmitC artifact 完整
legalization”仍是同一个接口/成功值。

## 横向范围

最终合入必须同时覆盖：

- RVV；
- IME；
- Scalar；
- Demo；
- Toy；
- Template；
- TensorExtLite；
- Offload 的 explicit unsupported；
- 所有 registered source front door；
- public pass、direct conversion/API、materialization、translate 与 target artifact export；
- formula catalog、construction-entry inventory、current EmitC backend inventory；
- selected-body realization、ABI/runtime handoff 与 mixed-owner gates。

测试 fixture、oracle 或未注册研究入口不进入 production owner 分母，但必须不可能成为
production fallback。

## 必须建立的边界

### 1. Canonical Problem Contract

为每个 production source entry 建立可枚举的 typed problem ownership：

```text
S  operator semantics / operand roles / result semantics
g  format/encoding/packing/layout/shape/geometry facts
ω  bounded static regime/memory form/necessary policy
```

不要求新增通用 tensor IR，也不要求所有 family 使用同一个物理 C++ problem class。公共
contract 只应足以：

- 识别 entry owner/operator domain；
- 证明 source 未携带 family schedule；
- 将 typed facts 交给 family-local constructor；
- 做 catalog/entry 的双向完整性检查。

禁止 source descriptor 携带 LMUL、IME helper、GPU warp/tile、完整 point leaf id、winner 或
artifact route。

### 2. Target/Family Binding

显式 target/profile 在 construction 前绑定唯一 `(f,c_f)`：

- binding 依据 typed capability/applicability；
- common code 只经 registry/interface 编排；
- binding 不创建 candidate、不选择 point leaf、不读取 measurement；
- artifact kind、route id、device string 或 emitter scan 不得反向决定 family；
- missing/ambiguous/conflicting/unsupported fail closed。

当前不增加 CPU/GPU runtime selector。

### 3. Artifact-neutral Family Construction Lifecycle

每个 production family 必须显式实现唯一 construction lifecycle：

- 从 family-scoped problem/capability 输入构造或严格资格化 final typed body/plan；
- 完成 formula、legality、optional bounded selection 与 required schedule；
- 明确 unsupported/reject；
- 没有 base-class 默认成功 no-op；
- 公共 completion result/type 不引用 EmitC type、`emitc.func`、route id 或 artifact metadata；
- family-local typed result 不通过 universal `Any`/FormulaResult 传递。

优先复用/收敛现有 `ExtensionPlugin` lifecycle，而不是再建立平行 provider registry。具体
C++ 签名由实现审计决定，但只有一个 production construction authority。

### 4. Artifact Driver Separation

Current `TypedBackendEmissionDriver`/`BackendEmissionRegistry` 退回 EmitC artifact class：

- 只接受 construction-qualified final typed body；
- 不再拥有 family construction；
- `emitc.func`、unrealized cast、leftover dialect 等门继续属于 EmitC complete legalization；
- raw type/pattern population 继续 private；
- mixed/unowned body 继续 fail closed；
- route projection 仍是 optional mechanical handoff，不成为通用 provider。

公共 artifact orchestration 可以拥有 artifact identity、typed owner、ABI/runtime 与
fail-closed 编排；不能解释 family compute，也不能假定所有 artifact 都是 EmitC。

### 5. Caller 原子迁移

所有 current production caller 必须在同一个最终提交中迁移到：

```text
problem/binding
  → family construction
  → matching artifact driver
```

不得保留：

- old `prepareForConversion` construction；
- direct plugin construction 与 backend construction 双调用；
- construction missing 时由 emitter 补写；
- source/direct/translate/artifact 中任一路径跳过 construction；
- compatibility adapter、deprecated alias、dual-read/dual-write 或 metadata fallback。

施工中间顺序可以存在于未提交 worktree；合入状态不得有 production middle path。

## 必须保持的既有权威

- 原主公式 `θ=f^A(g,c,ω)` 与 `K` 构造式；
- legality-first 与 thin selector；
- measurement 只修正合法候选排序；
- quantize/dequantize 同一 RVV top-level construction cut；
- RVV `flat_*` 是 formula 产生的最终 computation plan，emitter 直接消费；
- Scalar/IME current final-plan decisions 不回流 emitter；
- `ConstructedWeak` 不冒充 strong reconstruction；
- core/common 零 family-name branch；
- current unique `apply*Conversion` 与 private raw conversion machinery；
- unsupported 和 mixed owner fail closed。

## 明确不在本 task

- NVIDIA/AMD/GPU plugin；
- GPU capability/profile fields；
- `weft_gpu` dialect 或 GPU physical Plan；
- `gpu.launch`、NVGPU、NVVM、ROCDL、PTX、cubin、hsaco；
- CUDA/HIP runtime；
- GPU correctness/performance campaign；
- CPU/GPU runtime dispatch；
- universal Formula IR、Plan、verifier、provider 或 provenance system；
- 新高层 tensor/tile IR；
- mechanism factorization/性能攻坚本身；
- 重新解释两柱、六律或 novelty。

## 完成门

- [x] 当前 registry、source front door、public pass/direct API、translate 与 artifact 的完整
  production-entry inventory 已由代码生成/反向枚举；
- [x] 每个 production source entry 有唯一 `S/g/ω` owner 与 operator domain；
- [x] target/profile 在 construction 前绑定唯一 family 与 typed `c_f`；
- [x] every live production family 显式实现 artifact-neutral construction，base 默认成功为零；
- [x] Offload 保持 explicit unsupported，不用空 construction/driver 冒充支持；
- [x] 公共 construction completion API 不依赖 EmitC 类型、`emitc.func`、route id 或
  metadata；legacy family-local route/manifest qualification 作为后续 A/B 清理项显式记录；
- [x] current EmitC drivers 只消费 construction-qualified final body，内部不再调用同义
  construction owner；
- [x] public pass、direct API、materialization、translate 与 artifact 全部调用同一
  construction-before-artifact lifecycle；
- [x] missing/ambiguous/mixed/unconstructed/partial/conflicting/unsupported 均 fail closed；
- [x] quantize/dequantize、Scalar/IME final plan 与 RVV `flat_*` authority 回归保持；
- [x] construction inventory、formula catalog、source entry 与 artifact inventory 分立且双向
  可核，不用字符串 inventory 调度 compute；
- [x] 旧 construction hook、compat adapter、emitter recovery 与 artifact-side family binding
  为零；
- [x] focused behavior tests、catalog/registry tests、full `check-weft`、JSON/diff checks 通过；
- [x] spec/issues 按最终代码事实更新，task 固定主体 commit，工作区干净。

## 验证矩阵

### Static/API

- construction contract public headers 不 include EmitC dialect/interface；
- `ExtensionPlugin` 或最终 family lifecycle 无默认成功 construction；
- construction call graph 中无 artifact-specific completion gate；
- current `apply*Conversion` 仍只住 EmitC shared harness；
- common/core 无 RVV/IME/Scalar/GPU family-name branch；
- raw population helper 不公开。

### Behavior

- source problem → binding → construction → EmitC artifact 正路径；
- direct typed/debug input 先经同一 construction qualification；
- every current family 至少一条正路径和 missing/unsupported 负路径；
- ambiguous family binding；
- mixed-family final body；
- forged/partial final plan；
- unowned source/body；
- artifact request without construction；
- current EmitC conversion without `emitc.func`。

### Regression

- RVV quantize/dequantize construct-from-abstract；
- flat/super-block/elementwise recursive qualification；
- Scalar q2/dequant plan；
- IME VLEN/MAC/tile plan；
- Demo/Toy/Template/TensorExtLite deterministic final body；
- target artifact/translate/runtime ABI；
- full formula catalog/registry integrity；
- complete `check-weft`。

## 完成后的下一步

下一 task 仍是 RISC-V 旗舰 realization 的横向重构，而不是 GPU implementation：
`.trellis/tasks/07-23-executable-knowledge-ab-horizontal-closure/`。它在已经干净的
construction/artifact 边界上，同步推进 A 线的 formula/mechanism/strong reconstruction 与
B 线的 current-artifact correctness/performance 因果闭环。只有这条主线不再依赖逐点完整
authority、并且重构后真实性能得到 paired evidence，才另建 GPU family task。
