# Target Profiles

Profiles 是 capability fixture：描述可被 pass 查询/验证的具体目标事实。它们是 capability 输入，**永远不**创造 `tcrv_rvv` body、dtype authority、route id、source-front-door route 或 intrinsic 选择（见 [core-invariants](../architecture/core-invariants.md) I5）。

具体的 VLEN/VLENB/dtype 支持等应**探测或带 provenance 声明，不靠猜**——它们是 target capability 事实，不是 runtime SSA/control 值，也不是 per-variant 常量。

## rvv-main（当前主真实硬件）

```text
access:     ssh rvv
hardware:   RISC-V CPU, 64 cores（实测值，以 probe 为准）
vector:     RVV 1.0
permission: sudo available
role:       primary development / correctness / performance
```

capability 示例：`rv64`、`rvv`、`zvl128b`（或实测最小 VLEN / raw VLENB）、fp32/fp64（依硬件）、thread-runtime（OpenMP/pthread）、native compile 支持。

稳定的 profile capability id 保持 plugin-local 且通用：`rv64`、`rvv`、`rvv.hart_count`、`riscv.toolchain.march`/`mabi`、`rvv.toolchain.clang`/`cmake`、`rvv.probe.compile_run`。provider 身份、benchmark 名、日志、性能测量值**不得**变成 capability id（I9）。`rvv.hart_count` 可 `provides = ["target.hart_count"]`，其 `count` 是 uarch 事实，不是 runtime thread 数、dispatch guard、tensor shape、AVL/VL。

证据规则：RVV correctness/runtime/performance 主张要真 `ssh rvv` 证据并命名本 profile（或派生 probed profile）；本地 compile-only / smoke / 文档改动都不是 runtime 证据（I8）。Python probe 可暴露 sanitized `capability_facts`，但**不得**把它翻译成 `tcrv.exec` capability/target/route mirror/typed body/route 输入/fallback 建模（I6）；从 probe 证据到 compiler-visible capability 的权威转换是 plugin-local C++ RVV capability profile 校验 + `TargetCapabilitySet` 填充。probe 也不得伪造 SEW/LMUL/tail/mask 这类 plugin-selected 编译期 config 事实。

## k3-ime（后续 IME 接入 —— N2 的关键证据点）

```text
hardware: SpacemiT/K3-class RISC-V system
role:     IME extension plugin validation（第二个非-RVV family）
status:   later environment
```

capability 示例：`rvv`、`spacemit.ime`、vector-register-backed matrix capability、vendor intrinsic / inline asm path。

这是验证"新增 matrix-like 扩展能否经 IME 插件局部接入"的环境——**它是 N2 能否被证明的载体**。在拿到并 probe 真实硬件前，按 later plugin validation 对待，不写成当前主硬件；IME runtime/performance 主张要真 K3/IME 硬件与 toolchain 证据。

## riscv-sophgo-offload（runtime-offload case）

```text
hardware: RISC-V host + Sophgo accelerator
role:     runtime-offload capability case
```

capability 示例：`rvv` 或 scalar CPU fallback、sophgo runtime available、C ABI call path、PCIe/SoC mode、async（若有）。

建模为 `kind = "runtime-offload"`，**不**归类成 RISC-V custom ISA。cost/dispatch 必须含 runtime launch、transfer、sync、fallback 行为。
