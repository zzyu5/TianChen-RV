# PRD · K-quant decode load-width lever 攻公式墙（ISSUE-014·a13 施工入口）

> **权威** = ISSUE-014 decode 真成本中心诊断（task `07-18-kquant-decode-scope`·a13·objdump·**8 格公式墙·必攻**）+ B 线（公式墙必攻·先 objdump 对手已做）。
> **性质** = 公式墙攻坚（decode leaf ≠ vec_dot leaf·load-width lever 直击·**允许诚实缩 scope**·**验收门内建脾气墙裁定**）。

## 一、a13 施工入口（已定·非猜）
decode leaf(`typed_repack_gemv_loop_body`)**结构上已在 vec_dot 三 lever 后状态**（0 权重 store register-resident·4-way 累加器·向量化 scale）·vec_dot 前三墙对 decode = no-op。残留真成本中心 = **窄 load(576× vle8 mf2 8B) + 低 MLP over weight-DRAM-stream**（M=1 无行摊·intensity=1）。**★关键：`integer_core_lmul` knob(mf2→m1) 在 vec_dot cold-inert(roundtrip 遮蔽)·但 decode 无 roundtrip → knob 直接=load-宽度旋钮·直击 memory-scheduling·未试。**

## 二、★★首步·perf-stat 证 memory-bound（a13 诚实缺口·须先做）
a13 的"memory-stall"是 objdump 结构推断·**未 perf-stat**。**施工前第一步 = perf-stat decode leaf**（`ssh rvv` perf·IPC/backend-idle/cache-miss·填 ISSUE-014 留白）·证真 memory-bound（非 unroll/frontend-bound）。**若 perf-stat 证非 memory-bound → 停·报 main（诊断修正·load-width 非对症）。**

## 三、你要做（perf-stat 证 memory-bound 后）
1. **load-width lever**：decode 现走 LITERAL mf2(窄) → `integer_core_lmul="m1"`（宽 load·decode 无 roundtrip→直接 load-宽度）。可叠 STACK 多流 MLP + k1 VLEN256 宽化(`deriveRepackHalfLanes`)。gate 后·默认 byte-identical。
2. **byte-exact 免费**（整数·[K-5]·ULP=0）·**新 gate 值·默认无新 schema**（复用 integer_core_lmul·a13 定）。
3. **对手身份**：真攻坚 = q2/q3/q4/q6_K@rvv（`_vl128` 手调）+ q3_K@k1。**先 q4_K@rvv / q6_K@rvv decode**（真手调 opp·高价值）。q5_K 弱 opp(native-vec 无 vl-spec)·**q5_K/q6_K@k1 对手身份存疑（A2-batch5 block-dot vs 真部署 repack q5_K_8x8/q6_K_8x8·ISSUE-004·先核再测）**。

## 四、★验收门（脾气墙裁定内建·同 vec_dot 范式）
decode load-width(mf2→m1) 增寄存器压力(同 vec_dot mlp 的 m2 溢出风险)·板测后**必 objdump**：
- **clang 保住宽 load MLP ∧ cold≥0.8** → **公式墙翻**（关该格·真硬赢强手调·地盘+1·同 memory-scheduling 轴 same-family 扩）。
- **clang 保住 ∧ cold<0.8** → 具名 + 新墙诊断（下一 lever）。
- **clang 再串行化/溢出**（objdump 证宽 load 未 hoist/spill·同 vec_dot m2 register-pressure）→ **三步走完·登记 [K-4] 编译器行为脾气墙**（同 q4_K vec_dot·具名-X·**🔴 禁 inline-asm 绕 clang·禁架构不可达**）。

## 五、验收
1. **perf-stat 证 memory-bound**（首步·填 ISSUE-014 留白·非 memory-bound 则停报）。
2. **owned load-width emit**（objdump 证 mf2→m1 宽 load·decode 无 roundtrip·非 re-roll）。
3. **byte-exact ULP=0**·cold 2-seed + **objdump 验 clang 保调度**（verdict 三分支·§四）。
4. **成色**：真手调 opp（q4_K/q6_K@rvv）翻正=真硬赢·q5_K/存疑 k1 格标成色/身份。
5. **sealed 不动**·**worktree·禁 commit·禁 add**（main cherry-pick·worktree commit 报 hash·**git add 只纳源·勿纳 build-wt**）·objdump 全量·0 造数。

## 六、触碰集 / 遗留
- 触碰：`RVVToEmitCKQuant.cpp`（decode 路 typed_repack_gemv·load-width gate）·**worktree·rvv 板**（+k1 若测 VLEN256）。
- **遗留**：翻则 same-family(q2/q3/q6_K@rvv·q3_K@k1)扩·脾气墙则 [K-4] 边界登记（同 vec_dot）。**须新 schema=停报 main·须板批 perf-stat=先做。**
