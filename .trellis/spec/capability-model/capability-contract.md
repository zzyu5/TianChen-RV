# Capability Contract

Capability 是系统第一对象（见 [core-invariants](../architecture/core-invariants.md) I1）。本文件定义它的来源、形态、关系和验证职责——这是 N1（RISC-V 扩展异构性作为 first-class capability IR）的契约。

## 它必须影响什么

capability 对象必须能被 C++ MLIR pass 和插件查询，并影响：启用哪些插件、variant 提议 / 合法性、tuning 空间、cost 输入、selection、runtime dispatch、emission 路径选择、lowering 诊断、fallback 需求。它不是 prose、裸字符串、JSON-only 记录或 Python dict。

## Capability 来源

- **ISA**：`rv64`、`rvv`、`zvl*`、`zvfh`、`zvfbf*`、`V`、`ime`、vendor custom opcode、future matrix/custom ISA。
- **Microarchitecture**：core count、VLEN、cache、内存带宽、preferred LMUL、dtype throughput、NUMA/拓扑、thread runtime 可用性、toolchain 交链。
- **Runtime/offload**：accelerator 存在性、runtime 名、ABI、PCIe/SoC 模式、支持的 offload op 集、model 格式、host-device 传输代价、async 支持。
- **Toolchain**：march/mabi 事实、LLVM RVV 可伸缩向量支持、intrinsic 支持、builtin 支持、inline asm 许可、vendor header、patched compiler、runtime lib 可链接性。

kind 是**开放字符串集**（isa-scalar / isa-vector / isa-vector-config / isa-matrix-* / isa-custom-instruction / runtime-offload / toolchain / uarch / memory / thread-runtime …，未来可扩）。Core pass **不**穷举 switch 所有 kind，只用插件注册的 interface + capability 查询（I3）。

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
  extensions   = [ #tcrv.ext<"rvv", kind = "isa-vector", status = "available"> ],
  accelerators = [ #tcrv.accel<"sophgo.bm1684x", kind = "runtime-offload", mode = "pcie", runtime = "vendor-c-abi"> ],
  toolchain = { llvm_rvv = true, rvv_intrinsic = true, inline_asm = true, vendor_runtime_link = true }
>
```

（上面的 `cores=64 / vlen=128 / sophgo` 是**示例值**，不是 durable 的项目事实；具体目标参数属于 profile，见 [profiles](./profiles.md)。）IME 可加 `#tcrv.ext<"spacemit.ime", kind = "isa-matrix-vector-backed", ...>`。

## Relations

capability 带 `provides` / `implies` / `conflicts` 三类关系（first-class 描述符字段，不是 property-map 项，也不是 prose）。

- **require**：variant 声明所需 capability（`requires = [@cap, ...]`，`FlatSymbolRefAttr` 指向 kernel capability scope 内的符号）。provider 是直接 `tcrv.exec.capability`、带 `id`+`kind` 的 kernel-local `tcrv.exec.target`、kernel `target = @profile` 指的那个 module-level profile、以及该 profile 经 `capability_providers = [...]` 显式组合的 providers。kernel 只看自己引用的 profile + 其命名 providers + kernel-local providers；id 在该 scope 内唯一。
- **provide**：`provides = ["..."]` 是 capability id（不是符号名、不是 prose）。`lookupProviderByID(id)` 先解析 exact id；无 exact 时可由 available 的 `provides`/`implies` 满足。于是 `id="rvv.profile.rv64gcv", provides=["rvv"]` 能满足要求 `rvv` 的提议，同时 exact `id="rvv"` 在场时保持直接覆盖。
- **imply**：`implies = ["..."]`（如 `rv64gcv implies rvv`、`zvfh implies fp16 向量算术（受 toolchain 支持约束）`）。经同一 relation-aware lookup 暴露。这是 bounded 的决策路由，不是完整 capability lattice 或推断引擎。
- **conflict**：`conflicts = ["..."]`（如"要 vendor runtime 但无 runtime lib"、"要 inline asm 但 build policy 禁止"）。`--tcrv-check-capability-requires` 用 bounded 双向冲突查询作 legality gate：静态 variant / dispatch fallback 在所需 capability 与另一 available capability 冲突时 fail closed；dispatch case 只有携带 typed `runtime_guard_required = true` 时才能引用冲突需求（记录保护面，不解析 printable 串）。这**不是**完整 conflict solver / lattice / provider ranking。
- **dispatch condition**：runtime/shape 相关条件成为 dispatch 谓词（`if runtime_available && large_shape -> offload；else if rvv_available -> rvv；else -> fallback`）。

> 查询 API：`TargetCapabilitySet::buildFromKernelChecked(KernelOp)`（带诊断的构造，duplicate id/symbol fail closed）；`buildFromKernel` 仅用于已验证上下文。pass `--tcrv-check-capability-requires`。按符号名与 id 双向查询；relation-aware lookup 支持 exact/provided/implied，exact 在场时权威。missing status 视为 available；`status` 优先于 `availability`；`unavailable`/`disabled`/`missing` 视为不可用。**核心代码不解释具体 target-family 的 status 语义**（I3）——`if (target.hasRVV())` 是错的，要走 `capabilities.isCapabilityAvailableBySymbolName(...)`。

## Verifier 职责

capability verifier 检查：variant `requires` 被 target capability 满足或被 dispatch 守护；插件运行前通用 variant/capability 结构良构；extension-family legality 委派给插件 hook（不被 core 按 family 名硬编码，I3）；extension op 只出现在其插件 legality 接受了 capability 需求的 variant 内；selected emission 路径被 toolchain capability 支持；runtime ABI 声明完整；dispatch/fallback 覆盖不可用条件。verifier **不**证明数值正确性，只挡非法 target-feature 使用和缺失的执行前提。

## 实现

capability model 属 primary stack（I6）：C++ 数据结构 + MLIR attribute/type/interface；IR 表示用 TableGen/ODS；CMake target；lit/FileCheck 覆盖 parse/print/verify/诊断；文本测不了的 helper 语义补 C++ test。Python 只 probe/解析/出报告，**不**作为 capability 关系、插件可用性、legality、selection、dispatch 或 lowering 诊断的真值实现。
