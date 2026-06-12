# Core Invariants

这些是 TianChen-RV 全项目复用的硬规则。**在这里声明一次**；其他 spec 文件引用 `core-invariants I#`，不重复抄写。早期 spec 把这些规则在 8–22 个文件里反复粘贴——那是冗余，按引用收敛。

## I1 — Capability 是第一系统对象

target capability（ISA 扩展、VLEN/uarch、toolchain、runtime/offload）是 first-class、可被 C++ MLIR pass 和插件查询的对象，带 `provides` / `implies` / `conflicts` 关系。它不是 prose、不是裸字符串、不是 JSON-only 记录、不是 Python dict。它必须能影响：启用哪些插件、variant 提议 / 合法性 / 选择、tuning 空间、cost 输入、dispatch、emission 路径、fallback 需求。

## I2 — `tcrv.exec` 只是 execution envelope

`tcrv.exec` 承载 kernel / target / capability / variant / requires / region / hart_parallel / mem_window / runtime_param / dispatch / fallback / diagnostics。它**不**表达 matmul / softmax / reduce / 通用 tile / 通用 tensor compute，也**不**拥有 selected route、dtype、schedule、intrinsic spelling 或任何 compute 语义。计算语义属于 extension family（RVV / IME / offload / 未来插件）。

## I3 — 零 family-name 分支

core / common pass 只通过 capability registry 和插件 interface（extension / config / resource / memory / compute / EmitC-lowerable 等）调用插件，**绝不**在 common 路径里写 `if RVV` / `if IME` / `if Sophgo` 之类的 family-name 分支。这是 N2（plugin 泛化）的可证明前提。

## I4 — Metadata 是 mirror，不是 authority

emission-plan diagnostics、readiness/status 报告、route id、artifact metadata、manifest、semantic role graph、source-front-door metadata、dashboard——全部只是**镜像 / 规划辅助**。它们永远不是 route、compute、dtype、进度、或证据的 authority。fail-report 和复现用途允许；用它们反推 compute 一律禁止。

## I5 — 可执行事实必须结构化在 typed body 里

可执行的 dtype / config / operation 必须**结构性地**存在于 typed extension body（如 `tcrv_rvv`），或在 route 构造前被消费进 realized body。**禁止**从 C ABI 字符串、参数名、route id、artifact 名、test 名、intrinsic 拼写、或旧的 dtype 前缀 helper 名推断 dtype / config / operation。`mem_window` / `runtime_param` 只声明 ABI/runtime 角色，必须由 typed body 显式 import 并经 typed control/dataflow op 消费。

## I6 — Primary stack 是 C++/MLIR；Python 只做 tooling

实现栈：C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck。Python 只允许做 runner、supervisor、remote probe、artifact parsing、小型支持脚本。Python **禁止**作为 core IR、dialect、pass、plugin registry、capability model、lowering 或 emission 的实现语言。本地 MLIR/LLVM 工具缺失时，加检测与诊断 fail-closed，**不得**用 Python 数据结构替代真实 MLIR 编译器内部。

## I7 — Fail closed，不要从 metadata 合成

无法证明存在合法可执行 route 时，**fail closed** 并给 bounded 诊断。禁止把 selected metadata / family 记录 / route 记录 / descriptor-like 记录翻译成 kernel C 源、header、object、self-check 源或 artifact bundle。**Descriptor-driven computation 是非法架构**：现存 descriptor / microkernel / direct-C 路径是历史残留、删除目标或 fail-closed 债务，不得描述为"过渡架构"。

## I8 — 硬件主张需要真实证据

runtime / correctness / performance 主张需要对应的真实证据。RVV 即真实 `ssh rvv` 上的 compile/run 证据。本地 CMake / `tcrv-opt` / lit 只是编译器/工具链证据，不能充当硬件正确性或性能证据。

## I9 — 实验验证结构，不反向定义结构

实验参考用于验证系统结构是否成立，**不**反过来决定系统该长什么样。benchmark 名、artifact 名、q8/q4/llama 之类的工作负载名，都不是 route / dtype / 进度 authority；它们只是 N1–N3 的压力测试输入。
