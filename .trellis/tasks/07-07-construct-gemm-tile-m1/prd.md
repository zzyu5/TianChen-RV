# M1 gemm_tile 真构造 — [DEBT-Q4K-SCAFFOLD] + gemm_tile 停手追认

> 状态:planning(构造队列登记)。本 prd 登记债务与队列,不在本轮实现 lib/ 修复。
> 登记轮次:2026-07-07 裁决一 追认与登记。

## 背景

裁决四背景接线批②(`ffb2d703`)发现:**gemm_tile M1 覆盖靠简单批接线不可达**。
逐格排查后得到两条结论,本任务把它们登记为可追踪的构造队列 + 一条硬纪律。

---

## [DEBT-Q4K-SCAFFOLD] — q4_K repack GEVM 数值债(★硬纪律)

- **债务点**:`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:4244-4247`
  的 `emitRepackGemvQ4KQ8K` 是 **1b-i 脚手架**(由 q4_1 GEVM 改名而来)。
  自注释原文:
  > `// q4_K GEVM emitter — 1b-i scaffold (q4_1 GEVM renamed). COMPILES + reads q4_K ABI via shared`
  > `// getters, but COMPUTES q4_1-style (single block fold) = NUMERICALLY WRONG until 1b-iii adapts`
  > `// it to q4_K's 8-sub-block 6-bit-unpack + bsums-min (see KQUANT-REPACK-DESIGN.md STAGE-1b).`
- **性质**:它**能编译、能读 q4_K ABI**(经共享 getter),但**数值错误**——用 q4_1 的
  single-block fold 计算 q4_K,未做 q4_K 的 8-sub-block 6-bit-unpack + bsums-min 解码。
- **★ 硬纪律(裁决一.4)**:**禁任何新接线复用该 scaffold**。
  在 `emitRepackGemvQ4KQ8K` 被真正改成 q4_K 语义(STAGE-1b-iii)之前,
  任何 dispatch 表 / front-door / 覆盖批不得把新格路由到这个已知错的 emitter 上。
  拒绝 fake-wire 已知错 = 纪律对(见下"gemm_tile 停手追认")。

---

## gemm_tile 停手追认(纪律记录)

- gemm_tile M1 覆盖尝试中,**18/19 格未假接线 = 【正确】**。
- 唯一已接线的 `emitRepackGemvQ4KQ8K`(q4_K)是上面的 known-wrong 脚手架;
  其余格若靠"批接线到现有 emitter"提升覆盖,会把更多格路由到错语义或不匹配的解码。
- **停手在 18/19 = 拒绝 fake-wire 已知错**,这是对的纪律选择:
  覆盖数(C_construct)只认真解码的真构造,不认假接线的名义覆盖。
  假接线会污染 C_construct 台账并留下静默数值债。

---

## 构造队列(真构造,非批接线;多会话)

按 `task.json` description 的四条,登记为待做真构造工作(均属 lib/ 侧,后续独立立项/推进):

1. **q4_K repack GEVM 真解码(STAGE-1b-iii)** — 把 `emitRepackGemvQ4KQ8K` 从
   q4_1-style single-block fold 改成 q4_K 的 8-sub-block 6-bit-unpack + bsums-min。
   解债务点后此格才可正当计入 C_construct。(q4_K 曾 multi-day timed out 2×。)
2. **q2_K / q3_K / q5_K / q6_K 真 per-format 超块解码** — 各自 per-format 解码,多会话。
3. **13 grid/codebook/ternary + q1_0** — 无 block-as-lane repack 布局,
   需全新 per-lane codebook-gather 整数核。
4. **3 IME 格(q4_0 / q8_0 / q4_K)** — 需 IME 路径。

> 结论:gemm_tile M1 = **真构造战役**,非批接线。

---

## 完成判据(本登记轮)

| 条件 | 达成 |
|------|:---:|
| [DEBT-Q4K-SCAFFOLD] 债务点(file:line + 自注释)登记 | ✅ |
| 硬纪律"禁新接线复用 scaffold"写入 prd + task.json notes/tags | ✅ |
| gemm_tile 停手追认(18/19 不假接线=纪律对)记录 | ✅ |
| 构造队列四条登记 | ✅ |
| 本轮不改 lib/(真构造后续独立推进) | ✅ |

## 更正(2026-07-07, commit 2a930d85): [DEBT-Q4K-SCAFFOLD] 前提证伪
q4_K repack GEVM+GEMM **不是 debt**——"NUMERICALLY WRONG" 是陈旧注释,真解码(710ad067)已建、
deferred oracle 现 GREEN(GEVM 8/8 + GEMM 8/8, bounded-norm 7e-7, 负控咬合)。q4_K 已开采。
- q4_K = **L1 path-win 活候选**(board-probe-pending):ggml q4_K repack 硬编 VLEN256+在 QK 门外、
  rvv VLEN128 上 not-selected/wrong → opponent-无-WORKING-repack-@VLEN128。需板 dispatch probe 确认、禁预判 beat。
- 真剩余 K-quant repack 构造(新 emitter、都没有):**q5_K→q6_K→q2_K/q3_K**(q5_K=q4_K+qh 5th-bit,最便宜)。
- 或 q4_K 专用 gemm_tile op(q4_0 GgmlGemmTileQ40Q80Op 类比)。
