# T-KEYING — 决策键控轴状态回填（T8/schema 派生·描述性）

> **建成**：2026-07-13 L3-apply（casefile `experiments/active/l3-triage/决策键控轴-回填-candidates.md` 派生）。
> **为何派生表而非改 schema**：schema `perf-covered-category.v1.json` 是并行 board 线（L1/L2）共享写文件（IME/q4_K 转判在飞）；决策键控状态=**描述性回填**（recon 不消费），放派生表避免共享文件碰撞（[parallel-lines-need-disjoint-files]）·不破 schema shape（防 [F-2']）。任务 C1「schema 或 T8 派生行」二选一 → 取 **T8 派生行**。
> **对象**：7 绿格 + 12 黄-对手更强格 = **19 格**（读 schema 起点）。
> **措辞锁（令六）**：本表 = 描述性键控状态（哪些**已由能力事实键控** / 哪些仍**发射器写死**）·**非 perf 主张·非 e2e 收益**·禁「性能决策显式化为能力事实」裸口号单独作 novelty（须绑格×决策×证据）。

---

## 1. 7 绿格 — 决策键控状态

| 格·板 | 主导决策 | 已键控（能力事实） | 仍发射器写死 | 对手侧 | 证据指针 |
|---|---|---|---|---|---|
| **q4_0 gemm @rvv** | repack-vs-block-dot 路由 | ✓ 路径路由由 VLEN128 能力 gate 键控（repack.cpp:4592 一行 VLEN128 gate·routing 白嫖） | 布局/tile = 上游 repack（白嫖·非我方键控） | 上游 ggml block-dot（通用路径·**系统账 routing·kernel 账非净绿**） | T8 q4_0-routing 行；memory q4-0-e2e-is-routing |
| **q4_1 gemm @rvv** | repack 路由 + 净新 dispatch | ✓ 路径路由净新 dispatch（riscv_v present） | 布局=净新 repack（q8_1 家族复用·写死）·变体 mf2 写死 | 未优化 generic block-dot（无 stock repack·path-win 通用路径） | schema green gemm/q4_1 |
| **q5_0 gemm @rvv** | 净新 scaffold + interleaver 布局 | ✓ dispatch 路由键控 | 布局 interleaver（make_block_q5_0x16·写死）·**变体 mf2 写死（键控 isRVV0p7 generation·NOT VLEN·[repack-winA-always-mf2]）** | stock block-dot（通用路径·objdump 探针） | schema green gemm/q5_0 |
| **q5_1 gemm @rvv** | 净新 scaffold（q8_1-activation 家族首） | ✓ dispatch 路由键控 | 布局 block_q8_1x4 + quantize_mat_t<Q8_1>（写死）·变体 mf2 写死 | stock block-dot（通用路径·objdump vl=8 探针） | schema green gemm/q5_1 |
| **q8_0 gemm @rvv** ★user-anchor | **变体选择（VLEN→宽度）** | ✓ **VLEN→变体选择 已键控**（selector `block_dot_memory_bound` roofline dual·能力事实自然路由·default .inc 字节=M1b·部署==证过） | —（成色最强·无外援） | **对手=手写库写死宽度（hand-lib 仅恰好一个 VLEN 正确·非 capability-driven·上游 VLEN128 破损→我方 correctness-carrier）** | schema green gemm/q8_0；T8 VLEN-碎片化行 |
| **q4_K gemm @k1** | 变体/tile（VLEN256 repack·S6 min-fold） | ✓ dispatch 路由键控 · **tile/min-fold S6 已成熟**（spill→0·v30≤32 cliff·register-cliff 杠杆·[GAP-EMIT-KNEST] 真杠杆在 q4_K **已闭合**） | 变体-vs-VLEN 键控程度待核（VLEN256 部署·VLEN128-half 破损 [GAP-Q4K-VLEN128]→键控不足证据） | **对手=真出货 hand-brick 16x1 repack（case256 fires·唯一强对手·byte-exact 胜 1.085×）** | T8 Win-K1-VLEN 行 |
| **q5_K gemm @k1** | repack-vs-block-dot 路由 + 净新 dispatch | ✓ **路径路由净新 dispatch 键控**（riscv_v present·k1 ships zero q5_K repack→我方 emit 路由）· tile S6 HOLDS（spill 155→105·v30≤32 cliff·XFER-1 #2） | 布局 block_q5_Kx16（stride-2816+qh·写死） | 对手=stock 通用 block-dot（k1 NEON-only sub-branch→riscv nullptr·objdump·通用路径·非手调 repack） | schema green gemm/q5_K；T8 q5_K 行 |

**绿格键控成熟度速览**：路径路由（dispatch/VLEN gate）**普遍已键控**（绿的结构基础）；q8_0 = 变体选择键控（roofline dual·成色最强·user-anchor）；q4_K/q5_K@k1 = tile/min-fold（S6 register-cliff）**已成熟**（[GAP-EMIT-KNEST] 在 min-fold K-quant **已闭合**·与 q2_K/q6_K 对照）；仍写死 = 布局/interleaver（净新 scaffold 手写）·变体 mf2（键控 generation 非 VLEN·[GAP-P1] selector-key-missing-VLEN-fact 遗留）。

---

## 2. 12 黄-对手更强格 — 决策键控状态

### 2.1 K-quant gemm（q2_K / q3_K / q6_K · 3 · ★[GAP-EMIT-KNEST] 头号具名实例区）

| 格·板 | 主导未键控决策 | 已键控 | 仍写死（[GAP-EMIT-KNEST]） | 对手侧 | 证据指针 |
|---|---|---|---|---|---|
| **q2_K gemm @rvv** ★user-anchor | **循环形态 + 累加器驻留** | dispatch 路由 + S6 tile 部分（v30 cliff·但 compiler-asymmetry-tainted） | **★循环形态写死 unrolled（全展开 2304 vwmacc/27KB）· 累加器驻留写死（spill）· 两 emit 形态(unrolled/rolled)都产不出 compact∧register-resident tiling = 真杠杆缺失（非 schedule knob）**·叠 [GAP-KQUANT-GCC-CODEGEN]（gcc full-unroll 爆炸 104KB/1433 vset） | **对手=手调 compact(16 vwmacc)∧register-resident(spill 4) 同时**（hand-brick·case256） | **T8 [GAP-EMIT-UNROLL] Phase2 NOT-FLIP + [GAP-P1] re-roll-trap + [GAP-EMIT-KNEST]（本次 append）** |
| **q3_K gemm @rvv** | 循环形态/调度 + weight-materialization | dispatch 路由 | **循环形态/调度未键控（PLAIN/UNTILED·S6 NULL）+ dual-plane weight-reconstruction floor**（weight-bound·非纯 emit·[GAP-EMIT-KNEST]+weight-floor 双阻） | 对手=hand-tuned _vl128 block-dot | T8 q3_K 行 |
| **q6_K gemm @rvv** ★[GAP-EMIT-KNEST] 标注 | 循环形态/调度 + weight-reconstruction | dispatch 路由 | **★循环形态/调度未键控（6-bit dual-plane·2995 vsetivli + 6065 e8mf2 无调度·S6 NULL）+ weight-reconstruction floor**（[GAP-EMIT-KNEST]+weight-floor 双阻） | 对手=mature block-dot | T8 q6_K 行；LAW-FIRST-EMISSION 例 |

> **★[GAP-EMIT-KNEST] 边界（T8 本次 append 具名·[主会话4裁定③ 新 canonical]）**：q2_K = **头号具名实例**——真杠杆 = **循环形态 rolled ∧ 累加器 register-resident 同时**（发射器当前两形态 unrolled/rolled 各得其一、不可兼得·T8 [GAP-EMIT-UNROLL] Phase2 实证）。这是**发射器成熟度 GAP（未键控决策·K-nest 深重构）·非 schedule knob·非算法·非物理墙**。q6_K/q3_K 同类但**再叠 weight-reconstruction floor**（结构 weight-bound·即使循环形态键控闭合仍受权重重建 floor 封顶）→ q2_K = 纯 [GAP-EMIT-KNEST]·q3_K/q6_K = [GAP-EMIT-KNEST] + weight-floor 双阻。**对照正例**：q4_K/q5_K@k1 的 S6 min-fold tile **已闭合**该杠杆（register-cliff spill→0）→证明 [GAP-EMIT-KNEST] 在 min-fold K-quant 可闭·在 no-min/weight-bound（q3_K/q6_K）NULL。
> **与 [GAP-EMIT-UNROLL]/[GAP-P1] 关系**：相关但 distinct。[GAP-EMIT-UNROLL]（T8 Phase1/2）+[GAP-P1] re-roll-trap = 已登记的 rolled-loop **emit-形态**失败（rolled∨unrolled 二选一皆产 spill）；[GAP-EMIT-KNEST] = **K-nest 深重构**（compact∧register-resident 同时·非二选一）。

### 2.2 iq/tq vec_dot（8 · gather-batching / LMUL 宽度未键控）

| 格·板 | 主导未键控决策 | 仍写死 | 对手侧 | 证据指针 |
|---|---|---|---|---|
| iq2_xxs / iq2_xs / iq2_s vec_dot @rvv | **gather-batching** | per-subblock gather/redsum 写死未 batch（vluxei16 gather 未跨 sub-block batch·vsetvli churn） | 对手=SIMD-dispatch block-dot（batched gather·1 gather vs 我方 16–32） | T8 iq2 行；[GAP-SB] |
| iq3_xxs / iq3_s vec_dot @rvv | gather-batching + AVL-storm | fraclmul-2elem scalarization + vsetvli storm（iq3_xxs 220 vsetvli·110 AVL=2）写死 | 对手=block-dot（16 vsetvli·2 gather） | T8 iq3 行 |
| iq4_xs vec_dot @rvv | codebook-LUT gather | 16-entry NL LUT gather 未 batch（per-sub-block·17 fcvt vs 对手 2）写死 | 对手=block-dot（1 batched vrgather） | T8 iq4_xs 行 |
| tq1_0 / tq2_0 vec_dot @rvv | **LMUL 宽度 + spill** | **LMUL over-widen regfile spill 写死（[GAP-RP]·8-way e16m4 超 regfile→4 whole-reg spill）**·tq2_0 spill-fix 已部分闭（[GAP-RP]·但 vs-SIMD 仍 0.45×）·tq1_0 same-class 未修 | 对手=block-dot（0 spill·register-resident） | T8 tq 行；[GAP-RP] |

> iq/tq vec_dot 共性：**gather-batching（per-subblock→super-block batch）+ LMUL 宽度（register-pressure-aware）= 未键控决策**（写死 per-subblock / over-widen）。部分闭（iq2_xxs GAP-SB POST vs-generic WIN·tq2_0 spill-fix）·但 vs-SIMD-dispatch 仍 LOSS。

### 2.3 iq4_nl gemm（1 · lane 宽度未键控）

| 格·板 | 主导未键控决策 | 仍写死 | 对手侧 | 证据指针 |
|---|---|---|---|---|
| iq4_nl gemm @rvv | **lane 宽度 + codebook gather-batch** | **vl=8 写死（=upstream vl=16 半宽·per-element vluxei16 kvalues[16] codebook gather 未跨 tile batch）**·S6 tile no-op（codebook-already-lean·v30 cliff 已达） | 对手=upstream vl=16 batched codebook block-dot | schema opponent gemm/iq4_nl；T8 iq4_nl 行 |

---

## 3. 决策键控轴回填汇总（19 格）

| 桶 | 格数 | 主导键控状态 |
|---|---:|---|
| 回填格总数 | **19**（7 绿 + 12 对手更强） | — |
| 路径路由**已键控**（dispatch/VLEN gate） | 7 绿普遍 | 绿格结构基础 |
| 变体选择**已键控**（roofline dual） | 1（q8_0） | ★成色最强·user-anchor |
| tile/min-fold**已成熟**（S6 register-cliff） | 2（q4_K/q5_K@k1） | [GAP-EMIT-KNEST] 在 min-fold **已闭** |
| **循环形态+累加器驻留未键控**（[GAP-EMIT-KNEST]） | **3**（q2_K★/q3_K/q6_K★） | **头号未键控具名实例区**（T8 append） |
| gather-batching 未键控 | 6（iq2/iq3/iq4_xs vec_dot） | [GAP-SB] 部分闭·vs-SIMD 仍 LOSS |
| LMUL 宽度未键控 | 2（tq1_0/tq2_0） | [GAP-RP] 部分闭 |
| lane 宽度未键控 | 1（iq4_nl gemm） | vl=8 半宽 |
| **变体 mf2 键控 generation 非 VLEN**（[GAP-P1] 遗留） | 4（q5_0/q5_1/q4_1 + q4_0-decode） | selector-key-missing-VLEN-fact |

**需主会核**：① [GAP-EMIT-KNEST] 已作 T8 新 canonical 行（本次 append·[主会话4裁定③]）·cross-ref [GAP-EMIT-UNROLL]/[GAP-P1]；② q4_K@k1「变体-vs-VLEN 键控程度」（VLEN128-half 破损是否证键控不足·[GAP-Q4K-VLEN128]）；③ 变体 mf2「键控 generation 非 VLEN」是否单列 [GAP-P1] 遗留列。
