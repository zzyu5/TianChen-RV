# Research: 骨架收口战役 2a — 共享 emitter 冗余 vsetvli 诊断 (objdump 对账)

- **Query**: objdump 对账 typed emit vs ggml factory；数冗余 vsetvli、定位发射点、出修复计划 + 影响面 + 框定确认。只读，不改代码/不 commit/不上硬件。
- **Scope**: internal (objdump + emitter 源码；本地 laptop clang-20)
- **Date**: 2026-07-05
- **Toolchain**: clang-20.1.8 + llvm-objdump-20；`-O2 -march=rv64gcv_zfh_zvfhmin -mabi=lp64d -ffp-contract=off`（除注明）
- **Artifacts (scratch)**: `/tmp/vsetvli_diag/` (q8_0/q6_K .cpp + .o)；复用 `experiments/T3_step3/` 与 `experiments/ondevice-q8_0-*/` 的既有 .o

---

## 0. 一句话结论

冗余的根不是"整数核太贵"——per-block flat 的整数核 vtype 序列已和 factory 逐条对齐、且我方**已把 reduce seed hoist 出循环（factory 没有）**。真正的冗余是**共享骨架在每个 block-loop body 前无条件发一条 pre-loop scope setvl**（`RVVToEmitC.cpp:290`），其 VL 被内层 block-cap 立刻覆盖、vtype 与 block-cap 重复 = **+1 条纯冗余**；super-block 更糟（子区各发各的同-vtype block-cap）；deferred/mbf-unroll 把 block-cap 按展开次数重发。**改这一处共享骨架，10 个 typed 格全受益**。

---

## 1. vsetvli 计数 (ours vs factory)

### q8_0 — 严格 apples-to-apples（同 clang-20 / 同旗标 / 同 triple，linux-gnu）

| kernel | vsetvli 总 | 热循环内 | 冗余 | 来源 |
|---|---|---|---|---|
| **ours** per-block (mbf=1, elided, shipped) | **6** | 4 | **1** | `/tmp/vsetvli_diag/q8_pb_lg.o` |
| **factory** `ggml_vec_dot_q8_0_q8_0` | **5** | 4 | 0 | `/tmp/vsetvli_diag/factory_all.o` |
| ours deferred (mbf=4, P2c) | **19** | many | 数条 | `experiments/ondevice-q8_0-deferred/A_deferred.o` |
| ours mbf=2 (unroll×2) | **15** | — | 数条 | `experiments/ondevice-q8_0-mbf/kernel_core_mbf2.o` |

**交叉核对**（板旗标 `-march=rv64gcv` no-zfh，板 clang）：per-block=5 / factory=4（`experiments/ondevice-q8_0-mbf/kernel_{q8_mbf1,ggml_factory}.o`）。**delta 恒为 1**，只是 zfh march 把两边各 +1（fp16 段）。gap-log P2c-DEFERRED-NULL 记的"per-block 5 / factory 4-5 / deferred 19" 全部复现。

### q8_0 热循环逐条对照（同旗标，objdump）

```
OURS per-block                         FACTORY
 0x08 vsetivli e32,m1  ← reduce seed    (无——factory 每轮在循环内重 seed)
       [HOIST 出循环 ✓ 我方已优化]
 0x14 vsetvli  e8,m2   ← block-cap#1    0xaca vsetvli e8,m2   ← block-cap
 0x24 vsetvli  e8,m2   ← ★冗余#2★       0xade vsetvli e32,m1  ← 循环内重 seed(有用)
 0x28 vle8/vle8/vwmul                   0xae6 vsetvli e16,m4   ← vwredsum(必需)
 0x42 vsetvli  e16,m4  ← vwredsum(必需)  0xaee vsetvli e32,m8   ← vmv.x.s(必需)
 0x4a vsetvli  e32,m8  ← vmv.x.s(必需)
```

- 热循环内两边都是 4 条；但 ours 的 `0x24`（e8m2，vl 不变，紧跟 `0x14` 的 e8m2）**纯冗余**；factory 的 `0xade`（e32m1 重 seed）**有用**。
- **删掉 `0x24` 后**：ours 热循环 = 3 条 (e8m2→e16m4→e32m8)，**严格少于 factory 的 4 条**（因为 ours 已把 seed hoist 出循环）。→ 税不只清零，还反超。

### q6_K — ours typed（super-block, m2, 单超块体）

| kernel | vsetvli | 冗余 | 来源 |
|---|---|---|---|
| ours typed (m2) | **7–8** | 2× e8m2 + 2× e8mf2 同-vtype 重发 | `/tmp/vsetvli_diag/q6_K.cpp` 行 31/110/206/222 |
| factory `ggml_vec_dot_q6_K_q8_K` | **不可比** | — | VLEN-dispatch `switch(vlenb*8)`，结构完全不同 |

**注**：gap-log/任务背景写"q6_K M2 也记 19"——本次直接 build 的 typed 单超块体只 7–8 条；19 疑为更早的 full-unroll 或不同测法。**冗余"模式"已证**（见 §2），但绝对 19 未复现，绝对数以本表为准。factory q6_K 是 VLEN 分派 switch，逐循环 vsetvli 与我方单形状体不可直接对账（诚实标注）。

---

## 2. 冗余清单 (hoist / coalesce)

按可回收性排序：

1. **[COALESCE · flat · 全格 · 最大回收]** pre-loop scope setvl `__riscv_vsetvl_e{sew}{lmul}(avl)`。
   - 每个 block-loop body 前**无条件**发一条，设 vtype=e8m2 + vl=avl(=n 总长)。
   - 但 flat/super-block 路的真实 per-iteration vl 由内层 block-cap（blockLen=32）每轮重设 → 这条的 **VL 死**、**vtype 与首个 block-cap 重复**。
   - 证据：emitted .cpp 行 7 `size_t v6 = __riscv_vsetvl_e8m2(v4);`（provenance `source_op=tcrv_rvv.setvl role=configure`）vs 行 29 `__riscv_vsetvl_e8m2(32)`（`source_op=tcrv_rvv.typed_flat_block_dot_loop_body`）——两条同 e8m2。
   - 回收：flat 6→5（=factory），热循环 4→3（<factory）。

2. **[COALESCE · super-block · K-quants]** 子区各发各的同-vtype block-cap。
   - q6_K：`__riscv_vsetvl_e8m2(32)` 在行 31 与 110（两个整数核子区）各发一次；`__riscv_vsetvl_e8mf2(8)` 在行 206 与 222（scale 内层循环）各发一次。前一子区已留同 vtype 时，后一条即冗余。

3. **[HOIST · deferred / mbf-unroll]** 循环不变 block-cap 按展开次数重发。
   - deferred(mbf=4)：e8m2 block-cap 随 4 个展开块各发（objdump 偏移 4a/5e 甚至同 `a3` 的 e8m2 连发两条），叠加 fold 相的 e16mf2/e32m1/e8m2/e32m8。19 条里相当部分是"本可提到展开体外发一次"的不变量。

4. **[SCHEDULE · 次要 · 非 vsetvli]** fcvt.s.h 插进整数核向量段。
   - ours：`fcvt.s.h`（fp16 scale→f32）落在 `vle8`(0x2c) 与 `vwmul`(0x3e) 之间（0x36/0x3a），插进整数向量段；factory 把 fcvt.s.h 推迟到 fold 相（0xb16+）。这是任务说的"fcvt 插向量段"调度税，独立于 vsetvli。

**★保留（非冗余，别动）**：ours 把 reduce seed（`vmv.s.x`/`vsetivli e32m1`，`0x08`）**hoist 出循环**，factory 每轮循环内重 seed（`0xade`）——这是我方既有增益。`e16m4`(vwredsum)、`e32m8`(vmv.x.s) 是 widening/extract **必需** vtype 切换，与 factory 逐条对齐，不可删。

---

## 3. WHERE (文件:函数:行)

| 冗余 | 文件 : 函数 : 行 | 说明 |
|---|---|---|
| ①pre-loop scope setvl（+1 冗余源，全 block-loop 路） | `lib/Conversion/RVV/RVVToEmitC.cpp:290-295` | `setvlCallee = riscvIntrinsicName("vsetvl", sew, lmul, "")` → `emitOpaqueCall(..., {avlArg})`。位于共享 block-dot 转换里、dispatch 表(`:333`)之前，对**每个** body 无条件发；含 `emitTypedFlatBlockDotLoopBody`(:358) / `emitTypedSuperBlockBlockDotLoopBody`(:360)。sew/lmul 取自 `preLoopSetVL`(:279-280)。 |
| ②flat block-cap setvl | `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:6129-6132`（callee 派生）；emit `:6265`(elided)/`:6277-6295`(robust) | typed flat loop body 内层 block-cap。monolith 孪生 `emitFlatBlockCore` 在 `:5595-5644`。 |
| ②super-block block-caps | `lib/Conversion/RVV/RVVToEmitCKQuant.cpp` `:81/:437/:954/:4060`(unpackSetvl e8m2)、`:219/:380/:1270`(stripSetvl/quarterSetvl e8`<l8>`)、`:1744/:2181`(smSetvl e8m1)、`:4189`(dotSetvl e8m1) | 每子区 helper 各自硬编码 setvl callee，无跨子区 vtype 记忆。 |
| ★seed（保留，已 hoist） | `RVVToEmitCBlockQuantLinear.cpp:5563 / :6233`（`vmv_v_x` e32m1 seed） | 非冗余。 |

---

## 4. 修复计划 + 影响面

### 修复（诊断建议，不实现）

- **F1（最大、近 1 处、全格受益）** `RVVToEmitC.cpp:290`：对 block-loop 路（typed flat + super-block recognizer 命中）**跳过** pre-loop scope setvl，让内层 block-cap 成为唯一 configure。这些路每 block 必重设 vl，pre-loop configure 是纯冗余。→ flat 6→5（追平 factory），热循环 4→3（反超）。
- **F2（super-block coalesce）** 在 `emitTypedSuperBlockBlockDotLoopBody` 里记"上次已发 vtype"，仅当新子区 vtype 与上次不同才发 block-cap setvl。→ 折叠 q6_K 的 2× e8m2 + 2× e8mf2。
- **F3（deferred/unroll hoist）** 把不变的 block-cap e8m2 setvl 提到 mbf 展开体外发一次。→ deferred 19 → ~8-10。
- **F4（次要，fcvt）** 把 brick1 fp16→f32 scale 转换（fcvt.s.h）的发射推迟到 fold 相，别插进整数核向量段。

### 影响面（★哪些 emit 变、哪些 lit 更新）

**emit 变的格**：F1 触共享 `RVVToEmitC.cpp:290` → **全 10 个 typed 格**（5 flat q8_0/q4_0/q4_1/q5_0/q5_1 + 5 super-block q2_K/q3_K/q4_K/q5_K/q6_K）的 emit 都变（少一条 pre-loop setvl）。F2 只触 5 super-block。

**lit 更新范围（比预期小）**：
- 共享骨架 fixture（14 个 typed-body）**多数用松散 `CHECK:`** 匹配语义 op（`vle8`/`vwmul`/`vwredsum`），**0 条 CHECK-NEXT**、不断言 setvl → 对 setvl coalesce **透明**（如 `rvv-to-emitc-q8-0-q8-0-typed-flat-block-dot-loop-body-schedule.mlir`）。
- **确实断言 setvl 的**：`rvv-to-emitc-q2-k-q8-k-typed-super-block-block-dot-loop-body.mlir`（`:112 vsetvl_e8m2` / `:124 vsetvl_e8m1`）、`rvv-to-emitc-typed-flat-block-dot-loop-full-body.mlir`（1 处）——需 review；松散 `EMIT:` 只要该 vtype 还剩 ≥1 条即过。
- `--implicit-check-not` fixture（`*-m1-schedule`、`*-deferred`）的 check-not 目标是 fold op（vfredusum/vfmacc/vwmacc），**不含 setvl** → 安全。
- **e2e 导出测试** `test/Target/RVV/*-full-pipeline-export-e2e.mlir`（25 个，24 个引用 vsetvl/byte-exact）：断言 CORE emit == 导出 artifact **byte-identical**；两条路一起变、一致性保持；但任何**绝对指令指纹**断言会破（memory 已记 `f810ce6b/cb04b219` STALE，应本就迁走）。

---

## 5. ★框定确认（修 vsetvli 数值不变）

**成立。** vsetvli 只配置 SEW/LMUL/VL；删一条同-vtype 冗余 configure、或把循环不变的 configure 提出去，**不改任何向量 op 计算出的值**——只要每个 op 仍在覆盖该 block 的正确 vtype 下执行。冗余的 pre-loop setvl（`RVVToEmitC.cpp:290`）设的 vtype 被内层 block-cap 立即以**同 vtype**重设；删它后 block-cap 成唯一（且正确的）configure。

- **保留**：byte-exact **vs 数值 oracle / vs ggml**（sumi 整数和、fp32 fold 逐位不变）。
- **会破且属预期**：byte-exact **vs 退役前 monolith emit**（构造期指纹）。这是**成熟度轴的 emitter 优化，不是构造改动**——construction 已锁（C_construct 5→6 cohort 已落）。修 vsetvli 不碰算术链。**框定成立**。

---

## Caveats / 未决

- q6_K factory 是 VLEN-dispatch `switch(vlenb*8)`，与我方单形状体不可逐循环对账；本报告只给 ours q6_K 绝对数 + 冗余模式，未给可比 factory q6_K 数。
- "q6_K M2 = 19" 未复现（本次 typed 单超块体 7–8）；绝对数以 §1 表为准。
- factory 用 clang-20 + `-march=rv64gcv_zfh_zvfhmin` 本地编译需给 ggml 打两个补丁：stub 少量 hosted libc 头 + 给 q2_K/q3_K 手写 inline-asm 的 `vset(i)vli` 补 `, ta, ma` 策略（clang IAS 要求）。这只为拿 factory 对手核的 objdump，不入任何测量。
- 所有数为 laptop objdump 静态计数（指令条数），**非** perf；perf/税清零验证是修完后 `ssh rvv` [PERF-1] 八门的事。
- 未改任何代码、未 commit、未上硬件。
