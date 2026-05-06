# 02. `tcrv.exec` Core Dialect 设计

## 2.1 模块定位

`tcrv.exec` 是 TianChen-RV MLIR 的稳定核心 dialect。

它的职责是组织 RISC-V AI kernel 的执行变体，而不是表达通用计算语义。

它只负责：

```text
kernel 边界
目标 capability
execution variant
hart-level parallelism
extension region
memory window / buffer view
dispatch / fallback
variant metadata
```

它不负责：

```text
表达通用 matmul
表达通用 softmax
表达通用 reduce
表达通用 tensor tile
表达算法级计算语义
```

计算应落在 extension dialect 中，例如：

```text
tcrv.rvv.*
tcrv.ime.*
tcrv.offload.*
future tcrv.vendor.*
```

## 2.2 执行层次

TianChen-RV 的执行层次为：

```text
tcrv.exec.kernel
  └── tcrv.exec.variant
        └── tcrv.exec.hart_parallel / tcrv.exec.region
              └── extension dialect ops
```

这对应 RISC-V AI kernel 的现实执行组织：

```text
kernel: 一个可调用 AI kernel
variant: 一种硬件实现方式，例如 RVV、IME、offload、fallback
hart_parallel: 多核 RISC-V CPU 上的粗粒度并行
extension region: 某种硬件能力控制的执行区域
extension ops: RVV/IME/offload 等具体执行操作
```

## 2.3 Core ops

### `tcrv.exec.kernel`

表示一个 TianChen-RV kernel。

职责：

```text
定义 kernel 输入输出；
绑定 target capability；
包含一个或多个 execution variants；
定义 fallback 策略；
声明是否允许 runtime dispatch。
```

示意：

```mlir
tcrv.exec.kernel @matmul(%A, %B, %C)
  attributes { target = #tcrv.target<...> } {
  ... variants ...
}
```

### `tcrv.exec.variant`

表示一种实现变体。

职责：

```text
声明 requires capability；
包含 extension dialect ops；
携带 cost/tuning/dispatch metadata；
由某个 extension plugin 生成；
由 capability verifier 检查合法性。
```

示意：

```mlir
tcrv.exec.variant @rvv
  requires = #tcrv.requires<["rvv", "zvfh"]>
  origin = "rvv-plugin" {
  ... tcrv.rvv ops ...
}
```

### `tcrv.exec.requires`

可作为 op attribute 或专用 op/attribute，表示 variant 对 target capability 的要求。

示意：

```mlir
requires = #tcrv.requires<["rvv", "zvl128b", "zvfh"]>
```

### `tcrv.exec.hart_parallel`

表示 RISC-V harts/cores 上的 coarse-grained 并行。

职责：

```text
表达多核 CPU 并行划分；
lower 到 OpenMP、pthread、runtime thread pool 或单线程 loop；
不表达 GPU-style thread/block；
不假设所有 RISC-V target 都有相同 parallel runtime。
```

示意：

```mlir
tcrv.exec.hart_parallel %tid in %num_harts {
  ... extension ops ...
}
```

### `tcrv.exec.region`

表示某段代码使用某类 extension resource。

示意：

```mlir
tcrv.exec.region kind = "rvv" {
  ... tcrv.rvv ops ...
}
```

此 op 可选。如果 variant body 已经全是单一 extension dialect，region 可以作为结构化验证和分析辅助。

### `tcrv.exec.mem_window`

表示某个 variant 使用的内存窗口、shape specialization、stride/view 信息。

职责：

```text
给 extension plugin 提供合法的 memory slice；
支持 offload buffer binding；
支持 RVV contiguous/strided access 判断；
支持 variant dispatch 时的 shape guard。
```

### `tcrv.exec.dispatch`

表示 runtime variant selection。

职责：

```text
根据 target availability、shape、dtype、size、runtime presence 选择 variant；
保证有 fallback；
为 offload threshold 等策略提供 MLIR 表达。
```

示意：

```mlir
tcrv.exec.dispatch {
  case @offload if #tcrv.cond<"sophgo_available && M*N*K > threshold">
  case @rvv     if #tcrv.cond<"rvv_available">
  fallback @scalar_or_default
}
```

### `tcrv.exec.fallback`

表示保守路径。

职责：

```text
当 capability 不满足、shape 不满足、runtime 不可用、调度失败时，提供正确性路径；
可以 lower 到 scalar/scf/vector/default MLIR lowering；
不是性能主线，但必须存在。
```

## 2.4 Core types and attributes

`tcrv.exec` 的 type system 应保持轻量。

建议包含：

```text
#tcrv.target<...>
#tcrv.requires<...>
#tcrv.cap<...>
#tcrv.variant_info<...>
#tcrv.dispatch_cond<...>
#tcrv.cost<...>
#tcrv.tuning<...>
```

具体 vector register、IME fragment、offload buffer 等 type 不应放在 core dialect 中，应放在 extension dialect 中。

例如：

```text
!tcrv.rvv.vreg<...>
!tcrv.rvv.mask<...>
!tcrv.ime.frag<...>
!tcrv.offload.buffer<...>
```

## 2.5 示例：RVV + offload variants

```mlir
tcrv.exec.kernel @matmul(%A, %B, %C)
  attributes { target = #tcrv.target<...> } {

  tcrv.exec.variant @rvv
    requires = #tcrv.requires<["rvv", "zvfh"]>
    origin = "rvv-plugin" {

    tcrv.exec.hart_parallel %tid in %num_harts {
      %vl = tcrv.rvv.setvl %n {sew = 16, lmul = 4}
      %a  = tcrv.rvv.load %A[...] : !tcrv.rvv.vreg<f16, lmul = 4>
      %b  = tcrv.rvv.load %B[...] : !tcrv.rvv.vreg<f16, lmul = 4>
      %c  = tcrv.rvv.fma %a, %b, %acc
      tcrv.rvv.store %c, %C[...]
    }
  }

  tcrv.exec.variant @sophgo_offload
    requires = #tcrv.requires<["sophgo.runtime", "vendor_c_abi"]>
    origin = "offload-plugin" {

    %bufA = tcrv.offload.bind %A
    %bufB = tcrv.offload.bind %B
    %bufC = tcrv.offload.bind %C
    %ev = tcrv.offload.async_call @sophgo_matmul(%bufA, %bufB, %bufC)
      { abi = "c", runtime = "vendor" }
    tcrv.offload.wait %ev
  }

  tcrv.exec.fallback @scalar_or_default
}
```

## 2.6 Verification rules

`tcrv.exec` verifier 应检查：

```text
每个 kernel 至少有一个 fallback 或有明确外部 fallback 声明；
每个 variant 必须声明 requires；
variant body 中的 extension ops 必须满足 variant requires；
variant origin 必须来自已注册 plugin 或显式标记为 external；
dispatch 必须覆盖 capability 不满足的情况；
不能在 core dialect 中出现高层 generic compute op；
offload variant 必须声明 runtime ABI 和 sync 边界；
IME variant 必须声明 IME capability；
RVV variant 必须声明 RVV capability。
```

## 2.7 和 High-level MLIR 的关系

High-level MLIR 到 TianChen-RV 的转换不应是：

```text
linalg.matmul -> tcrv.matmul -> rvv/ime/offload
```

而应是：

```text
linalg.matmul
  -> plugins propose variants
  -> tcrv.exec.kernel with rvv/ime/offload/fallback variants
```

因此，`tcrv.exec` 是 execution variant container，不是 computation container。

## 2.8 科研价值边界

`tcrv.exec` 的科研价值在于：

```text
用统一 MLIR 结构组织 RISC-V 扩展能力；
让 high-level op 能生成多个 RISC-V execution variants；
让 capability、legality、dispatch、fallback 都在 IR 中显式可见；
让新增 extension 的影响局部化到 plugin，而不是散落到核心 pass。
```

它的价值不在于重新发明 tensor compute dialect。

