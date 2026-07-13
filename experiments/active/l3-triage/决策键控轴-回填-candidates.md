# L3 案头 — 决策键控轴状态回填 candidate（任务 C · 五.1）

> **性质**：纯案头 read-only·**禁改 schema/T8**·产【决策键控状态】列拟标注给主会话审后入 schema/T8。
> **对象**：7 绿格 + 12 黄-对手更强格（读 schema）。
> **体例范例（用户给）**：q8_0=「VLEN→变体选择(已键控)·对手=写死宽度(破损)」· q2_K=「循环形态+累加器驻留(未键控·写死 unrolled)·对手=手调 compact∧resident」。
> **标注维度**：主导性能决策（变体选择/布局/tile/lane 宽度/范式/**循环形态**/数值档）哪些**已由能力事实键控**、哪些仍**发射器写死**。
> **★[GAP-EMIT-KNEST]（循环形态未键控·G6-B 真杠杆边界）= 未键控决策头号具名实例**（五.2·标注 q2_K/q6_K 类）。**命名注**：本案头用任务给的 [GAP-EMIT-KNEST]；T8 同现象登记为 [GAP-EMIT-UNROLL]（row217/219）+[GAP-P1] re-roll-trap → **命名统一待主会核**（禁改 T8）。

---

## 0. 措辞纪律锁（令六 lint·本案头同守）

- 禁裸口号「性能决策显式化为能力事实」单独作 novelty（须绑具体决策×格×证据）。
- 决策键控状态 = **描述性回填**（哪些键控/哪些写死）·**非** perf 主张·非 e2e 收益。

---

## 1. 7 绿格 — 决策键控状态拟标注

| 格·板 | 主导决策 | 已键控（能力事实） | 仍发射器写死 | 对手侧 | 证据指针 |
|---|---|---|---|---|---|
| **q4_0 gemm @rvv** | repack-vs-block-dot 路由 | ✓ **路径路由由 VLEN128 能力 gate 键控**（repack.cpp:4592 一行 VLEN128 gate·routing 白嫖）| 布局/tile = 上游 repack（白嫖·非我方键控）| 上游 ggml block-dot（通用路径·**系统账 routing·kernel 账非净绿**）| T8 row16；memory q4-0-e2e-is-routing |
| **q4_1 gemm @rvv** | repack 路由 + 净新 dispatch | ✓ 路径路由净新 dispatch（riscv_v present）| 布局=净新 repack（q8_1 家族复用·写死）·变体 mf2 写死 | 未优化 generic block-dot（无 stock repack·path-win 通用路径）| schema green gemm/q4_1 |
| **q5_0 gemm @rvv** | 净新 scaffold + interleaver 布局 | ✓ dispatch 路由键控 | **布局 interleaver（make_block_q5_0x16·写死）·变体 mf2 写死（键控 isRVV0p7 generation·NOT VLEN·[repack-winA-always-mf2]）** | stock block-dot（通用路径·objdump 探针）| schema green gemm/q5_0；T8 |
| **q5_1 gemm @rvv** | 净新 scaffold（q8_1-activation 家族首）| ✓ dispatch 路由键控 | 布局 block_q8_1x4 + quantize_mat_t<Q8_1>（写死）·变体 mf2 写死 | stock block-dot（通用路径·objdump vl=8 探针）| schema green gemm/q5_1 |
| **q8_0 gemm @rvv** ★user-anchor | **变体选择（VLEN→宽度）** | ✓ **VLEN→变体选择 已键控**（selector `block_dot_memory_bound` roofline dual·能力事实自然路由·default .inc 字节=M1b·部署==证过）| — （成色最强·无外援）| **对手=手写库写死宽度（hand-lib 仅恰好一个 VLEN 正确·非 capability-driven·上游 VLEN128 破损→我方 correctness-carrier）** | schema green gemm/q8_0；T8 row192（VLEN 碎片化×3族）|
| **q4_K gemm @k1** | 变体/tile（VLEN256 repack·S6 min-fold）| ✓ dispatch 路由键控 · **tile/min-fold S6 已成熟**（spill→0·v30≤32 cliff·register-cliff 杠杆·[GAP-EMIT-KNEST] 真杠杆在 q4_K **已闭合**）| **变体-vs-VLEN 键控程度待核**（VLEN256 部署·VLEN128-half 破损 [GAP-Q4K-VLEN128]→键控不足证据）| **对手=真出货 hand-brick 16x1 repack（case256 fires·唯一强对手·byte-exact 胜 1.085×）** | T8 row74/67；Win-K1-VLEN |
| **q5_K gemm @k1** | repack-vs-block-dot 路由 + 净新 dispatch | ✓ **路径路由净新 dispatch 键控**（riscv_v present·k1 ships zero q5_K repack→我方 emit 路由）· tile S6 HOLDS（spill 155→105·v30≤32 cliff·XFER-1 #2）| 布局 block_q5_Kx16（stride-2816+qh·写死）| 对手=stock 通用 block-dot（k1 NEON-only sub-branch→riscv nullptr·objdump·通用路径·非手调 repack）| T8 row191/75；schema green gemm/q5_K |

**绿格键控成熟度速览**：路径路由（dispatch/VLEN gate）**普遍已键控**（是绿的结构基础）；q8_0 = 变体选择键控（roofline dual·成色最强·user-anchor）；q4_K/q5_K = tile/min-fold（S6 register-cliff）**已成熟**（[GAP-EMIT-KNEST] 在 min-fold K-quant 已闭合，与 q2_K/q6_K 对照）；仍写死 = 布局/interleaver（净新 scaffold 手写）·变体 mf2（键控 generation 非 VLEN·[GAP-P1] selector-key-missing-VLEN-fact 遗留）。

---

## 2. 12 黄-对手更强格 — 决策键控状态拟标注

### 2.1 K-quant gemm（q2_K / q3_K / q6_K · 3 · ★[GAP-EMIT-KNEST] 头号具名实例区）

| 格·板 | 主导未键控决策 | 已键控 | 仍写死（[GAP-EMIT-KNEST]）| 对手侧 | 证据指针 |
|---|---|---|---|---|---|
| **q2_K gemm @rvv** ★user-anchor | **循环形态 + 累加器驻留** | dispatch 路由 + S6 tile 部分（v30 cliff·但 compiler-asymmetry-tainted）| **★循环形态写死 unrolled（全展开 2304 vwmacc/27KB）· 累加器驻留写死（spill）· 两 emit 形态(unrolled/rolled)都产不出 compact∧register-resident tiling = 真杠杆缺失（非 schedule knob）**·叠 [GAP-KQUANT-GCC-CODEGEN]（gcc full-unroll 爆炸 104KB/1433 vset）| **对手=手调 compact(16 vwmacc)∧register-resident(spill 4) 同时**（hand-brick·case256）| **T8 row219 [GAP-EMIT-UNROLL] Phase2 NOT-FLIP + [GAP-P1] re-roll-trap**；row217 |
| **q3_K gemm @rvv** | 循环形态/调度 + weight-materialization | dispatch 路由 | **循环形态/调度未键控（PLAIN/UNTILED·S6 NULL）+ dual-plane weight-reconstruction floor**（weight-bound·非纯 emit）| 对手=hand-tuned _vl128 block-dot | T8 row64/row70 |
| **q6_K gemm @rvv** ★[GAP-EMIT-KNEST] 标注(五.2) | 循环形态/调度 + weight-reconstruction | dispatch 路由 | **★循环形态/调度未键控（6-bit dual-plane·2995 vsetivli + 6065 e8mf2 无调度·S6 NULL）+ weight-reconstruction floor**（双阻）| 对手=mature block-dot | T8 row62/row68；LAW-FIRST-EMISSION 例 |

> **★五.2 [GAP-EMIT-KNEST] 边界结论（案头拟·主会核）**：q2_K 是**头号具名实例**——真杠杆 = **循环形态 rolled ∧ 累加器 register-resident 同时**（发射器当前两形态 unrolled/rolled 各得其一、不可兼得·T8 row219 实证）。这是**发射器成熟度 GAP（未键控决策）·非 schedule knob·非算法·非物理墙**。q6_K/q3_K 同类但**再叠 weight-reconstruction floor**（结构 weight-bound·即使循环形态键控闭合仍受权重重建 floor 封顶）→ q2_K = 纯 [GAP-EMIT-KNEST]·q3_K/q6_K = [GAP-EMIT-KNEST] + weight-floor 双阻。**对照**：q4_K/q5_K@k1 的 S6 min-fold tile **已闭合**该杠杆（register-cliff spill→0）→证明 [GAP-EMIT-KNEST] 在 min-fold K-quant 可闭·在 no-min/weight-bound（q3_K/q6_K）NULL。

### 2.2 iq/tq vec_dot（8 · gather-batching / LMUL 宽度未键控）

| 格·板 | 主导未键控决策 | 仍写死 | 对手侧 | 证据指针 |
|---|---|---|---|---|
| iq2_xxs / iq2_xs / iq2_s vec_dot @rvv | **gather-batching** | per-subblock gather/redsum 写死未 batch（vluxei16 gather 未跨 sub-block batch·vsetvli churn）| 对手=SIMD-dispatch block-dot（batched gather·1 gather vs 我方 16–32）| T8 row9/10/11；[GAP-SB] row17/18 |
| iq3_xxs / iq3_s vec_dot @rvv | gather-batching + AVL-storm | fraclmul-2elem scalarization + vsetvli storm（iq3_xxs 220 vsetvli·110 AVL=2）写死 | 对手=block-dot（16 vsetvli·2 gather）| T8 row8/12/18 |
| iq4_xs vec_dot @rvv | codebook-LUT gather | 16-entry NL LUT gather 未 batch（per-sub-block·17 fcvt vs 对手 2）写死 | 对手=block-dot（1 batched vrgather）| T8 row13 |
| tq1_0 / tq2_0 vec_dot @rvv | **LMUL 宽度 + spill** | **LMUL over-widen regfile spill 写死（[GAP-RP]·8-way e16m4 超 regfile→4 whole-reg spill）**·tq2_0 spill-fix 已部分闭（[GAP-RP] row21·但 vs-SIMD 仍 0.45×）·tq1_0 same-class 未修 | 对手=block-dot（0 spill·register-resident）| T8 row14/row15/row21 |

> iq/tq vec_dot 共性：**gather-batching（per-subblock→super-block batch）+ LMUL 宽度（register-pressure-aware）= 未键控决策**（写死 per-subblock / over-widen）。真杠杆 = gather-batching（[GAP-SB]·[GAP-RP]）·部分闭（iq2_xxs GAP-SB POST vs-generic WIN·tq2_0 spill-fix）·但 vs-SIMD-dispatch 仍 LOSS。

### 2.3 iq4_nl gemm（1 · lane 宽度未键控）

| 格·板 | 主导未键控决策 | 仍写死 | 对手侧 | 证据指针 |
|---|---|---|---|---|
| iq4_nl gemm @rvv | **lane 宽度 + codebook gather-batch** | **vl=8 写死（=upstream vl=16 半宽·per-element vluxei16 kvalues[16] codebook gather 未跨 tile batch）**·S6 tile no-op（codebook-already-lean·v30 cliff 已达） | 对手=upstream vl=16 batched codebook block-dot | T8 iq4_nl；schema opponent gemm/iq4_nl |

---

## 3. 决策键控轴回填格数汇总

| 桶 | 格数 | 主导键控状态 |
|---|---:|---|
| 回填格总数 | **19**（7 绿 + 12 对手更强）| — |
| 路径路由**已键控**（dispatch/VLEN gate）| 7 绿 + | 绿格结构基础·普遍键控 |
| 变体选择**已键控**（roofline dual）| 1（q8_0）| ★成色最强·user-anchor |
| tile/min-fold**已成熟**（S6 register-cliff）| 2（q4_K/q5_K@k1）| [GAP-EMIT-KNEST] 在 min-fold 已闭 |
| **循环形态+累加器驻留未键控**（[GAP-EMIT-KNEST]）| **3**（q2_K★/q3_K/q6_K★）| **头号未键控具名实例区** |
| gather-batching 未键控 | 6（iq2/iq3/iq4_xs vec_dot）| [GAP-SB] 部分闭·vs-SIMD 仍 LOSS |
| LMUL 宽度未键控 | 2（tq1_0/tq2_0）| [GAP-RP] 部分闭 |
| lane 宽度未键控 | 1（iq4_nl gemm）| vl=8 半宽 |
| **变体 mf2 键控 generation 非 VLEN**（[GAP-P1] 遗留）| 4（q5_0/q5_1/q4_1 + q4_0-decode）| selector-key-missing-VLEN-fact |

---

## 4. 给主会话（审后入 schema/T8）

1. **[GAP-EMIT-KNEST] 头号具名实例 = q2_K gemm**（循环形态+累加器驻留未键控·真杠杆=compact∧register-resident 同时不可得·T8 row219 实证）·q6_K/q3_K = 同类+weight-floor 双阻。**q4_K/q5_K@k1 是对照正例**（S6 min-fold 已闭该杠杆）。
2. **命名统一待主会核**：任务给 [GAP-EMIT-KNEST] vs T8 已登记 [GAP-EMIT-UNROLL]（row217/219）+[GAP-P1] re-roll-trap — 是否同一 GAP 的两名？**禁改 T8**·建议主会话择一 canonical 名并交叉引用。
3. **user-anchor 体例复现确认**：q8_0=「VLEN→变体选择(已键控)·对手=写死宽度(破损)」✓ 与 schema green.成色三维/T8 row192 一致；q2_K=「循环形态+累加器驻留(未键控·写死 unrolled)·对手=手调 compact∧resident」✓ 与 T8 row219 一致。
4. **措辞锁**：本列 = 描述性键控状态回填·**非** perf 主张（禁「显式化为能力事实」裸口号单独作 novelty·须绑格×决策×证据）。
5. **入 schema 建议字段**：绿格 green.决策键控 / 对手更强 opponent_stronger.决策键控（与既有 named_gap 并列·标 已键控/写死 二分 + 对手侧对照）。

**需主会核**：① [GAP-EMIT-KNEST]/[GAP-EMIT-UNROLL] 命名统一；② q4_K@k1「变体-vs-VLEN 键控程度」（VLEN128-half 破损是否证键控不足·[GAP-Q4K-VLEN128]）；③ 变体 mf2「键控 generation 非 VLEN」是否单列 [GAP-P1] 遗留列。
