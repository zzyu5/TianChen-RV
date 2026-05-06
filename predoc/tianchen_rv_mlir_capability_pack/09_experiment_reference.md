# 09. 实验参考设计

## 9.1 文件定位

本文件只作为科研验证参考，不反向决定系统结构。

TianChen-RV MLIR 的系统目标首先是一个完整的 capability-driven RISC-V execution layer。实验用于验证这个系统设计是否成立，而不是为了某个实验结果临时改变系统边界。

## 9.2 当前硬件条件

### RVV 主环境

```text
access: ssh rvv
hardware: RISC-V CPU, 64 cores
vector: RVV 1.0
permission: sudo available
role: 主开发环境、主性能环境、主 correctness 环境
```

### K3/IME 后续环境

```text
hardware: K3 / IME-capable RISC-V environment
status: 后续采购/接入
role: IME plugin integration and matrix-extension evaluation
```

### RISC-V + Sophgo/offload 环境

```text
hardware: RISC-V host + Sophgo accelerator path
role: runtime-offload capability evaluation
note: 不作为 RISC-V custom ISA extension 证据，而作为 vendor runtime/offload capability case
```

## 9.3 实验问题

### Q1: TianChen-RV 能否在 RVV 真实硬件上生成有效代码？

目的：证明系统不是只做 IR 设计，而能在真实 RVV 硬件上跑通。

对象：

```text
matmul
batched matmul
softmax
layernorm / rmsnorm
rope
elementwise + reduction fusion
attention micro-kernel fragments
```

对比：

```text
MLIR linalg/vector default lowering
LLVM auto-vectorization
scalar/OpenMP baseline
hand-written RVV kernels if available
existing AI-Benchmark RVV kernels if usable
```

指标：

```text
correctness
single-thread performance
multi-thread performance
compile success rate
performance over default lowering
variant-local tuning benefit
```

### Q2: Capability model 是否真的参与 pass 决策？

目的：证明 capability object 不是元数据，而是影响 variant generation、legality、selection。

Profile：

```text
RVV only
RVV + offload runtime
RVV + IME, after K3 available
fallback-only profile
```

同一个 high-level op 应表现为：

```text
RVV only -> RVV variant + fallback
RVV + offload -> RVV variant + offload variant + dispatch + fallback
RVV + IME -> RVV variant + IME variant + dispatch + fallback
fallback-only -> fallback
```

指标：

```text
生成的 variant 是否符合 capability；
非法 variant 是否被 verifier 拒绝；
dispatch condition 是否正确；
capability 修改是否改变 pass 决策；
诊断信息是否清晰。
```

### Q3: Extension plugin integration 是否局部化？

目的：证明新增扩展不需要重写核心 pass。

实验过程参考：

```text
系统已有 RVV plugin。
加入 offload plugin。
后续加入 IME plugin。
观察核心 pass 修改量和新增 plugin 代码边界。
```

指标：

```text
core pass modified LOC
plugin LOC
新增 capability 数量
新增 ops/types 数量
新增 variant generator 数量
支持的 high-level op 数量
核心 pass 中是否出现 extension-specific branch
是否复用 tcrv.exec.variant / dispatch / verifier orchestration
```

目标现象：

```text
新增功能主要集中在 plugin；
核心 pass 不硬编码 IME 或 Sophgo；
同一个 high-level op 因新增 plugin 自动获得新的 variant。
```

### Q4: Offload runtime capability 是否能纳入同一执行层？

目的：证明 RISC-V host + accelerator runtime 可以通过 capability/variant/dispatch 进入同一框架。

对象：

```text
large matmul
conv
transformer block or MLP block if runtime supports
```

对比：

```text
RVV CPU variant
Sophgo offload variant
RVV + offload dispatch
fallback
```

指标：

```text
offload threshold
end-to-end latency
host-device transfer overhead
runtime launch overhead
shape size 对 selection 的影响
fallback correctness
```

注意：

```text
该实验不能声称 Sophgo runtime 是 RISC-V custom ISA。
它验证的是 runtime-offload capability。
```

### Q5: IME plugin 到位后是否能证明 matrix-extension 接入？

目的：证明新增 matrix-extension capability 后，系统能通过 plugin 局部接入。

对象：

```text
matmul
batched matmul
attention qk/av block
MLP dense block
int8/fp16/bf16 dot-like kernels
```

对比：

```text
RVV variant
IME variant
fallback
hand-written or vendor IME kernel if available
```

指标：

```text
core pass modified LOC
IME plugin LOC
IME variant generation coverage
IME legality verifier effectiveness
IME emission success
performance over RVV for suitable kernels
```

## 9.4 Ablation 参考

### Capability model ablation

```text
with capability-driven variant generation
without capability-driven variant generation, using fixed RVV path
```

观察：

```text
variant coverage
illegal target handling
ability to include offload/IME
```

### Plugin locality ablation

```text
plugin-based integration
hand-coded extension branches in core pass
```

观察：

```text
core modifications
code localization
new extension integration complexity
```

### Variant selection ablation

```text
static RVV selection
capability-aware variant selection
dispatch with offload threshold
```

观察：

```text
performance
fallback behavior
runtime adaptability
```

## 9.5 论文结果应避免的说法

不要写：

```text
Sophgo offload 是 RISC-V custom ISA extension。
普通 tile-size tuning 是主要理论创新。
AME 是当前系统已验证硬件主线。
新增任意未来扩展都完全不需要改核心。
TianChen-RV 是新的高层 tensor IR。
```

应该写：

```text
Sophgo offload 是 runtime-offload capability。
Tuning 是 capability-aware variant selection 的系统能力。
当前主线是 RVV，后续 IME 作为新增 extension plugin 验证。
系统对能映射到现有 interfaces 的扩展支持 plugin-local integration。
TianChen-RV 是 high-level MLIR 之后的 RISC-V execution layer。
```

## 9.6 最理想的实验闭环

```text
真实 RVV 硬件证明系统可运行、可优化。
Capability profile 证明 pass 决策由 capability 驱动。
Offload plugin 证明 runtime accelerator 能力可以纳入统一 variant IR。
IME plugin 证明新增 RISC-V matrix-like extension 能通过 plugin 局部接入。
核心 pass 修改量证明系统可扩展性。
性能结果证明该执行层不是纯架构设计。
```

