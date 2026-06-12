# Design Boundaries

## Non-Goals

TianChen-RV MLIR 不能变成：

- 又一个高层 tensor/tile IR；
- 一个硬件一个互不相关的 backend dialect；
- descriptor-driven microkernel/exporter 框架；
- 带 `tcrv.matmul` / `tcrv.softmax` / `tcrv.reduce` / `tcrv.generic_tile` / `tcrv.generic_mma` 的通用 compute dialect；
- core pass 里一堆硬编码 backend 分支；
- 保留可执行的 legacy `i32m1` 兼容 route；
- emission-plan/readiness/dashboard/status 当 route/进度/证据 authority 的系统；
- "未来硬件扩展零工作量"的主张；
- "Sophgo/runtime offload 是 RISC-V custom ISA 扩展"的主张；
- 一篇"普通参数搜索就是主要理论"的论文（tune 必须是 capability/resource-aware 的 N3，不是泛泛 autotuning）。

## 硬件事实位置

具体目标环境（rvv-main / k3-ime / riscv-sophgo-offload，含 64 核 / `ssh rvv` / sudo 等）是 profile 事实，住在 [profiles](../capability-model/profiles.md)，**不**在本文件、index 或其他 durable 架构文件里复述。任何"真实硬件路径"主张默认指 rvv-main，除非另一环境已被显式 probe 并记录；RVV correctness/runtime/performance 主张要真 `ssh rvv` 证据（I8）。

未来扩展（AME、future custom ISA、其他 vendor）是**插件槽位**，只作未来可扩展目标出现，不是当前必须完成的里程碑或当前硬件证据。无真实硬件/toolchain 证据时，这类工作只能定义通用 future-plugin 准入条件或非主张性设计槽。

## 论文语言（对齐 Novelty）

**用**：

```text
TianChen-RV 是 high-level MLIR 之后的统一 RISC-V MLIR，组织 capability-scoped extension execution。
Capability 对象驱动 variant 生成、legality、selection、dispatch、emission（N1）。
RVV/IME/TensorExt/Offload/未来 vendor 是同一 TCRV 系统内的 extension family，core/common 不按 family 名分支（N2）。
Gearbox 是 capability/resource-aware 的跨 family 调优层，把 selected body 变成调优过的可执行 body（N3）。
当前 lowering route：extension family ops -> EmitC -> intrinsic/vendor builtin/runtime C/C++。
Sophgo/offload 建模为 runtime-offload capability；IME 是评估 plugin-local 接入的第二 family。
```

**避免**：

```text
TianChen-RV 是新 tensor IR。
RVV/IME/TensorExt/Offload 是互不相关的独立 backend dialect。
tcrv.exec.kernel 是硬件 IR 主体。
Descriptor-driven computation 是架构。
Legacy i32m1 route table 是受支持的 RVV 架构。
Source-front-door 生成的 RVV artifact 证明成熟度。
tcrv.exec 拥有 selected RVV route 或 dtype 语义。
Emission-plan status 或 artifact metadata 是 route authority。
Sophgo 是 RISC-V custom ISA 扩展。
任何未来扩展零 core 工作即可接入。
单靠 tune / 泛泛参数搜索就是主要理论贡献（tune 只在它是 capability/resource-aware 且实测胜出时才算 N3）。
```
