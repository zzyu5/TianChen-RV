# System Positioning

## Role

TianChen-RV MLIR 是 high-level MLIR 之后的 RISC-V AI kernel execution layer。它不重新表达算法语义，而是组织 RISC-V target capabilities、execution variants、extension plugins、dispatch 和 fallback。

长期数据流：

```text
High-level MLIR
  linalg / tensor / stablehlo / tosa / custom high-level kernel dialect
        |
        v
TianChen-RV realization generation
        |
        v
TianChen-RV MLIR dialect suite
        |
        v
RVV / IME / runtime offload / future custom ISA emission
```

## Core Contribution Boundaries

### Capability-driven RISC-V execution model

`-march`、RVV、VLEN、microarchitecture、runtime/offload、toolchain support 必须成为 MLIR 中可查询、可验证、可参与 pass 决策的对象。

Correct:

```text
variant generation and legality depend on target capability object
```

Wrong:

```text
capability = string metadata attached after lowering
```

### Execution variant IR, not generic compute IR

High-level op 进入 TianChen-RV 后，由 plugins 直接提出 execution variants。核心 `tcrv.exec` 只承载 variant container 和 execution organization。

Correct:

```text
linalg.matmul -> plugins propose @rvv / @ime / @offload / @fallback variants
```

Wrong:

```text
linalg.matmul -> tcrv.matmul -> target-specific lowering
```

### Plugin-local extension integration

新增扩展通过 plugin 提供 capability、dialect、variant builder、legality verifier、tuning space、cost model、emission/runtime glue。核心 pass 通过 registry/interface 调用，不写 extension-specific branch。

### Capability-aware variant selection and tuning

Retuning 属于系统能力：在 capability 条件下选择实现变体，并调节 variant-local resource parameters。

Examples:

- RVV: LMUL、SEW、VL policy、unroll、thread partition。
- IME: fragment shape、K blocking、accumulator policy、packing。
- Offload: transfer threshold、batch size、async overlap、buffer reuse。

## Module Map

| Module | Stable Responsibility |
|---|---|
| Capability model | target object、capability relation、query、verification input |
| `tcrv.exec` | kernel、variant、requires、hart_parallel、dispatch、fallback |
| Plugin protocol | registry、interfaces、local extension boundary |
| RVV plugin | current real hardware path and first complete plugin |
| IME plugin | later K3/IME matrix-extension plugin validation |
| Offload plugin | runtime-offload capability for vendor accelerator paths |
| Variant pipeline | plugin proposal、legality、selection、dispatch、tuning |
| Lowering/runtime | plugin-owned emission and runtime glue |
| Experiment reference | validation framing only |
