# Research: C4a 参数化路线（dequant 真向量 kernel 该复用什么）

- **Query**: C4a 参数化路线现状 · dequant 真向量 kernel 复用什么
- **Scope**: 内部（spec + lib）
- **Date**: 2026-07-18

## 结论（一句）

C4a = **grid-table 参数化**路线（`.trellis/spec/issues/性能与测量.md` ISSUE-023 定义）。它是从 C4「一箭双雕」拆出的**真前置**，命中 **8 board-cell**；与 C4b（codebook 参数化 + codebook-size 键控 gather，命中 nvfp4 2 格 + iq4_nl what-if）互斥。dequant 真向量发射器「复用 C4a」= 复用**已存在的 grid-gather 真向量 emit 机体**（现役代码 `RVVToEmitCGridCodebook.cpp`），把它从 gemm/vec_dot 轴扩到 dequant 轴。

## C4a 的法源定义

`.trellis/spec/issues/性能与测量.md` **ISSUE-023**（状态：待裁）：
> C4「一箭双雕」假设部分证伪 → 必须拆两条机制。gather 键控喂不到 4 个 grid 格（iq1_s / iq1_m 2048 项 / iq3_xxs 256 / iq3_s 512 全是大 grid，vluxei16 本就是正确原语）；只有 nvfp4（16 项 tiny codebook）消费该键控 ⟹
> - **C4a** = grid-table 参数化（真前置 · 命中 **8 board-cell**）
> - **C4b** = codebook 参数化 + codebook-size 键控 gather（命中 nvfp4 2 board-cell + iq4_nl what-if leaf）
> **禁以「5 格 + iq4_nl」口径记 C4b 扇出。**

相关：ISSUE-021（C4b · 禁以性能名义立项）· ISSUE-022（C4b 与 Q3 是否合并里程碑 · 待裁）。

## dequant 真向量 kernel 复用的现役 C4a 机体（代码）

**`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp`**（2,501 行）—— grid-codebook 真向量 emit：
- 处理 `iq2_xxs / iq2_xs / iq2_s` 与 `iq3_xxs / iq3_s`（grid + sign-plane decode）。
- 真向量原语：`__riscv_vluxei16` grid gather + signs64 gather + `vle8` q8 load + `vmul`-onto-grid（`RVVToEmitCGridCodebook.cpp:301-341`）。
- 参数化锚：`coreLmul`（Win-A gearbox anchor · m2 default / m1 at VLEN256）· dot wide LMUL = 2×core · u16 index EMUL =(16/64)×core。「whole body is byte-exact for any legal anchor」（文件头注 L40-42）。
- 迭代结构：`row/col/block` 三结构循环，super-block×tile 计算体全直线展开（ISSUE-102 反汇编诊断证实）。

**`lib/Conversion/RVV/RVVToEmitCDeferredDequant.cpp`**（1,070 行）—— deferred/dequant 真向量 emit：
- 是 ISSUE-033 点名的「域内存在性证明」活范本：`value_or`/`switch`/`StringSwitch` **全 0**，却有完整形状轴（接受 m1/m2/m4/m8），因为宽度**读自 IR 向量类型**（`accVecType.getLmul()`）而非字符串属性 —— 全部 lmul 引用都是 fail-closed 门。
- 含 `emitDequantProductReduceSlice` / `emitDeferredWideDequantBody` / `emitLowPrecisionDequantBody` / standalone-dequant body。真向量：`vluxei16` indexed gather + `i8mf4` load。

## 诚实边界 / 未找到

- **「C4a 路线」本身在 spec 中只有 ISSUE-023 一处定义**（`grep -rc "C4a"` = 性能与测量.md 1 · index.md 2，全部指向 ISSUE-023）。**未找到**独立的「C4a 施工节点」或已落地台账 —— C4a 是**拆分口径**（待裁），不是已 mechanized 的里程碑。查了：`.trellis/spec/**`、`docs/`、git log grep C4a（无独立 commit）。
- 「复用 C4a 参数化路线」在《开测篇》§四.1 是**方向指认**（复用 grid-table 参数化的真向量 emit 机体），非指一个已封装的可调用 API。实际复用对象 = 上述两个现役 emit TU 的 grid-gather/参数化-anchor 机制。
