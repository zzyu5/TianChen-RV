# PRD · K线 · ISSUE-109 vwredsum.vs q4_K vec_dot 翻 0.8（register-fusion 续·复用 fused scaffold）

> **权威** = ISSUE-109 订正（2026-07-18·register-fusion 板测 EXHAUSTED·新 lever = vwredsum.vs）+ 补充令二 §五.1（register-fusion 别排队·已执行）+ §六（杠杆清单非空禁写架构不可达·本 task 打这个非空 lever）。
> **性质** = 发射器攻坚（结构级·per-sub-block 归约·[K-10] 结构级机制）。**对手 = 真强手调 block-dot register-resident（非 opp-immaturity·翻正则真硬赢强手调）。**

## 一、背景：register-fusion 已证真墙 ≠ 内存流量

前序 task `07-18-k-regfusion-q4k`（commit a728a6f77）板测确证：
- register-fusion **结构性成功**：aux8 权重重建 roundtrip 真消（objdump vse8 8→0·vle8 68→12·perf IPC 0.08→0.17 2.1×·cache-miss 2.2B→66M 33×少·byte-exact 0 ULP）。
- **★但 cold 0.152 略慢于 baseline mf2 0.162·墙没动** ⟹ **真墙 = latency/dependency-bound**（serial i32m8 累加器依赖链 + m8 register-cliff spill + cold DRAM 延迟），**不是 aux8 内存流量**。消 aux8 反而**收紧关键路径**（scratch 曾是 decoupling/overlap buffer）。
- **对手赢的机制**（register-fusion 诊断）= **register-resident `vwredsum.vs` per-sub-block 归约**：每 sub-block 独立 vwmul → vwredsum.vs 归约到标量 → 标量 fold scale。**16 个独立归约·无长累加链 → 高 ILP/MLP**。

## 二、本 task 的 lever：复刻对手的 vwredsum.vs per-sub-block 归约结构

**核心** = **打断 serial i32m8 累加器依赖链**（那是真墙）。改为：
- q4_K super-block = 8 sub-block（256 元素 / 32-per-sub · 注意 q4_K 结构·以现 sealed leaf 的 sub-block 划分为准）。
- **每 sub-block 独立**：`vwmul`（int8 weight × int8 q8 activation·或 unpack 后）→ `vwredsum.vs` 归约到**独立标量 partial sum** → 标量层 fold（× sub-block scale/min）。
- 各 sub-block 归约**互相独立**（无跨 sub-block 的 running i32m8 累加）→ 编译器/硬件可流水多个归约 → 高 ILP·避 m8 register-cliff。

**复用 `fused` scaffold**：前序 task 已建 `emitQ4_KFusedUnpackScaledDot`（gated `integer_core_lmul="fused"`·dormant·regression-free）。本 lever = 在其上（或新 gate 值 `integer_core_lmul="vwredsum"`）改归约结构为 per-sub-block vwredsum.vs。**默认路径（sealed mf2）不动**（gate 之外 byte-identical）。

## 三、★预期 —— 预测不是实测（先测不预告·[§五.15]）

天花板 per-format 板测定·直觉投影已被独立证伪 ≥3 次。**先测·不预告。**
- cold≥0.8 → **q4_K vec_dot 真翻**（关 ISSUE-109·消一个具名-X·**真硬赢强手调对手**·+ q6_K 同面 lever 就绪）。
- cold<0.8 → **具名 + vwredsum.vs lever 板测 EXHAUSTED**·objdump 逐指令墙·**若此后杠杆清单真空 → 方可 honest-null/架构不可达（§六）**；若诊断出新 lever → 具名保留、清单更新。

## 四、验收

1. **owned vwredsum.vs per-sub-block emit**（objdump 证 vwredsum.vs 结构·独立归约·非 serial i32m8 单链·OWNED 确定性）。
2. **byte-exact GREEN 0 ULP**（[K-5] ZERO-MODEL·q4_K vec_dot oracle·**整数点积 int32 零舍入 byte-exact 免费**·补充令二 §二：整数 byte-exact 不动）。**报数带 ULP 界=0**。
3. **cold 2-seed**（`vec_dot.sh rvv verify/measure q4_K`·flush224·idle 100%·gov performance）·verdict（先测·≥0.8 翻 / <0.8 真墙）。
4. **成色**：对手 = **deployed q4_K vec_dot 真手调 native-vec block-dot register-resident（3.6×快·强手调·NON-opp-immaturity）**——翻正 = **真硬赢强手调**（§三 成色·非软对手/便宜档）；不翻 = 具名真墙。
5. **q6_K 同面**：若 vwredsum.vs work·q6_K 同结构可扩（本 task 只做 q4_K·q6_K 留交接）。@k1 gated ISSUE-105（本 task 只 rvv）。
6. **0 造数**·byte-exact 硬门·**sealed 不动**（q4_K sealed md5 `892b6cf8`·gate 外默认路径 byte-identical）·master 不直写（recon-dict 留 main）·**禁 commit·禁 add -A**·objdump 全量。

## 五、触碰集 / 遗留

- 触碰：`lib/Conversion/RVV/RVVToEmitCKQuant.cpp`（q4_K vec_dot·复用/扩 `emitQ4_KFusedUnpackScaledDot` 的 vwredsum 归约变体·gate 值）+ 可能 `RVVToEmitCInternal.h`（decl）+ `RVVDialectWideningOps.cpp`（verifier 若加 gate 值）+ `test/Dialect/RVV/q4-k-scaled-dot-dataflow.mlir` + `vec_dot.sh` + bench 跑（rvv）。**与并行线 nvfp4（CodebookFp4·CodebookFp4.cpp）TU 不相交**——但**共享 build dir·若 nvfp4 线同时在跑须串行**（本 task 独占 weft-opt 重建窗口）。
- **遗留**：q6_K 同面 lever（@rvv·vwredsum.vs 同结构）+ @k1 半宽（gated ISSUE-105·D-2a）。本 task verdict 定 ISSUE-109 收口（翻/真墙）。
