# System Positioning

## Role

TianChen-RV 是 high-level MLIR 之后的**统一 RISC-V MLIR execution layer**。它不重表达算法语义，不把每个硬件做成互不相关的 backend dialect，而是在一个 TCRV dialect suite 内组织 RISC-V target capability、execution variant、extension family、dispatch、fallback 和 ABI/lowering route。

```text
high-level MLIR (linalg/tosa/stablehlo/custom)   ← 长期 frontend，opt-in
        |  semantic-preserving construction
        v
tcrv.exec envelope + typed extension-family bodies
        |  capability 驱动 legality / selection / dispatch / tuning
        v
selected typed body -> plugin route provider -> 公共 EmitC -> intrinsic/C ABI -> artifact
        |
        v
ssh rvv evidence（当 runtime/correctness/performance 被主张时）
```

详细 RVV authority chain 见 [rvv-plugin](../extension-plugins/rvv-plugin.md)（全项目唯一权威版本）；长期 frontend 是 opt-in 的未来集成，不是当前 source authority。

## 一个统一系统，不是 per-backend 集合

```text
Unified TCRV RISC-V MLIR
  -> 共享 core envelope（execution / capability / ABI / dispatch）
  -> RVV / IME / TensorExt / Offload / future vendor extension families
```

RVV、IME、TensorExt、Offload 概念上**不是**互不相关的独立 backend dialect——它们是同一 TCRV suite 内的 extension family，共享同一 capability model、extension interface、common orchestration pass、EmitC route 框架和测试口径。实现可拆分 ODS/C++/目录/测试，但这只是实现细节，不改变"一个系统"的定位。

**common abstraction = capability-scoped extension execution**（不是数学 kernel、tile、tensor compute op，也不是一硬件一 dialect）。

- **Core** 只拥有共享决策 envelope：capability、variant、requires、dispatch、fallback、ABI boundary、variant selection、plugin registry、extension family registration、diagnostics。core orchestration 只经 TCRV interface + 插件 registry 工作，**不**按 RVV/IME/vendor 名或 extension 计算语义分支（[core-invariants](./core-invariants.md) I3）。
- **Extension family** 贡献硬件/runtime 特定的 compiler surface：ops/types/attrs/verifiers、local legalization hook、selected-body realization、route provider / EmitC 映射、tests。示例形态：`tcrv_rvv.{setvl,load,binary,mask,store}`、`tcrv.ime_{config,load_frag,mma,store_frag}`、`tcrv.offload_{bind,call,wait}`。namespace 拼写（`tcrv_rvv.*` 等）是实现细节，不使该 family 变成独立 backend。

## Core Contribution Boundaries（对齐 N1/N2/N3）

**N1 — capability-driven execution model.** `-march`、RVV、VLEN、microarchitecture、runtime/offload、toolchain 必须成为 MLIR 中可查询、可验证、可参与 pass 决策的对象。
对：variant 生成与 legality 依赖 target capability 对象。
错：capability = lowering 之后挂的字符串 metadata。

**N2 — execution-variant IR + plugin-local 泛化.** 高层 op 进入 TianChen-RV 后先做 semantic-preserving construction，落成 `tcrv.exec` envelope + typed extension-family body；core 只承载 variant 容器与组织，本身不承载 compute 语义。新增扩展通过插件局部贡献 capability/ops/interfaces/variant builder/legality/tuning space/cost/EmitC 映射/runtime glue——core/common **零 family-name 分支**。
对：`linalg.matmul -> tcrv.exec envelope + typed body -> 插件 route/legalize/select @rvv/@ime/@offload/@fallback`。
错：`linalg.matmul -> tcrv.matmul -> target-specific lowering`。

**N3 — capability/resource-aware 跨 family tuning.** 在 capability 条件下选实现变体并调 variant-local resource 参数（RVV: LMUL/SEW/VL policy/unroll/thread partition；IME: fragment shape/K blocking/accumulator/packing；Offload: transfer threshold/batch/async overlap/buffer reuse）。影响生成代码的选择必须 realize 进 typed body（不是 status 或 metadata-only plan）。契约见 [variant-pipeline](../variant-pipeline/generation-selection-tuning.md)。

> "variant 容器""plugin 化"本身不是 novelty——MLIR 已提供。novelty 在 N1–N3 的**整合 + 证据**：异构 capability 建模、零-core-branch 泛化（用第二 family 证明）、capability 驱动且实测胜出的 tuning。

## Non-Architecture

Descriptor-driven computation 不是本架构（I7）。finite descriptor、microkernel descriptor、direct exporter helper 是历史残留/删除目标/fail-closed 债，不得当过渡架构、语义源、兼容垫片、production 输入或证据。可执行重建从 selected `tcrv.exec` variant → typed extension body → plugin route provider → 公共 EmitC 开始。

## Module Map

| Module | Stable Responsibility |
|---|---|
| Capability model | target 对象、capability 关系、查询、验证输入（N1）|
| `tcrv.exec` | execution envelope：kernel/target/capability/variant/requires/region/hart_parallel/mem_window/dispatch/fallback/diagnostics |
| Plugin protocol | registry、interfaces、extension family template、local 边界（N2）|
| RVV extension family | 当前真实硬件路径与第一个完整 family |
| IME extension family | 后续 K3/IME 矩阵扩展 family —— N2 的关键证据点 |
| Offload extension family | runtime-offload capability（vendor accelerator）|
| Variant pipeline | plugin proposal、legality、selection、dispatch、Gearbox tuning（N3）|
| Lowering/runtime | 公共 EmitC route + family-owned 映射 + runtime glue |
