# Research: 我方 q4_K vec_dot serialize 根因（三 lever 后·串行访存结构）

- **Query**: 三 lever 后我方为何 serialize（super-block N+1 的 load 等 N 的 reduce？依赖链？单 stream？）
- **Scope**: internal（读 sealed 默认 leaf 源 + ISSUE-109 board perf 证据）
- **Date**: 2026-07-18

## 证据来源

| 工件 | 路径 |
|---|---|
| 我方 sealed 默认 leaf（mf2·md5 `892b6cf8`） | `experiments/active/g7-census/vecdot-rvv/kernels/q4_K.kernel.c`（414 行） |
| board perf 证据 | ISSUE-109 + 机制③ ledger：IPC 0.08→0.12·96.2% backend-idle·2.2B cache-miss·三 lever cold ~1.32M ns 恒定 |

## 一、sealed 默认 leaf 的访存结构（逐区块）

单 super-block（`for v14 < v6`）体，三区顺序执行：

**Region A — 权重解包 + aux8 scratch roundtrip（对手无此）**
- 4 次 `vle8_v_u8m2`（**e8m2 = 32 字节**宽 load 打包权重）→ `vand`(低 nibble)/`vsrl`(高 nibble) → **`vse8_v_i8m2` 存回 `v7[256]` scratch**（8× vse8）。
- 权重被**存进 aux8[256] 再重载**——store→load roundtrip。

**Region B — sub-block 内积（tiny load + serial 累加器链）**
```c
for (v86 = 0; v86 < 8; v86++) {          // 8 sub-block
  // 4 quarter，每 quarter：
  vsetvl_e8mf2(8);                        // ★e8mf2 = 8 字节 tiny load
  v93 = vle8_i8mf2(v85+off);             //   载 8B 激活 y
  v94 = vle8_i8mf2(v8 +off);             //   载 8B aux8 scratch（Region A 存的重载）
  v95 = vwmul_vv_i16m1(v93, v94);
  v82 = vwmacc_vx_i32m2(v82, scale, v95);// ★累加进 v82（跨 quarter serial 依赖链）
}
```
- **load 粒度 = e8mf2 = 8 字节**（对手 16 字节的一半）·runtime 共 **64 条 tiny vle8**（32 激活 + 32 scratch 重载·ISSUE-109 objdump 记 68 vle8）。
- **累加器 `v82` 单链**：`v82→v97→v82→v106→v82→v115→…` 每 `vwmacc` 等前一个（vwredsum lever 针对此·已试）。

**Region C — min 项（scalar 主导·对手全向量）**
- **16× scalar `lh`**（`v128[0..15]` int16 bsums load）+ **16× scalar `mul`** + scalar 累加 → `v129`（minterm-vec lever 针对此·已试）。

## 二、三 lever 后残余 serialize = memory-latency 暴露（根因）

三 lever 各自消掉一个候选后 cold **全部不动**（~1.32M ns），board perf 定格 **IPC 0.12 / 96% backend-idle = stall-bound**。逐一排除：

| 候选串行源 | 对应 lever | 板测结果 | 结论 |
|---|---|---|---|
| aux8 权重 scratch roundtrip | register-fusion | vse8 8→0·cache-miss **33×↓**（2.2B→66M）·IPC 2.1×·**cold 0.152 不动** | **非根因**（那些 miss 命中 L1/L2·非 DRAM 关键路径） |
| i32 累加器 serial 链 | vwredsum.vs | serial vwmacc 32→0·byte-exact·**cold 0.162 不动** | 非根因 |
| scalar min-term serial MAC | minterm-vec | 16 lh + 16 mul 消·55→64 向量·**cold 0.164 不动** | 非根因 |

**★register-fusion 的 cache-miss 33×↓ 而 cold 不动 = 决定性证据**：说明 2.2B「cache-miss」是 aux8 scratch 的 L1/L2 流量（核能吸收），**不在关键路径**。真关键路径 = **原始打包权重 x[i].qs（144B/super-block·nc=512 行 ≈ 590KB 权重·flush=224MB 冷启）从 DRAM 流入的延迟**，被我方**逐 super-block 串行暴露**。

**为何串行暴露延迟（vs 对手并行）**：
1. **窄 load（e8mf2 8B）**：同数据发**两倍** load 条数·每条覆盖少·prefetcher 触发/收益弱。
2. **load 紧贴消费者（低 MLP）**：Region B 是 `载→乘→累加` 紧循环 + 单累加器链·任一时刻**极少 outstanding load**·无对手那种「~16 条独立宽 load 提前打到独立寄存器」的在飞并行。
3. **aux8 store→load 依赖**（sealed 默认路仍有·register-fusion 才消·但即便消了也是「非根因」因为窄+紧结构仍在）。
4. clang 从 C intrinsic 调度我方 leaf——**未自发产出 MLP**（register-fusion 把结构改干净后 clang 仍未 hoist 出多流并行·cold inert 即证）。

## 三、量级自洽（延迟暴露 vs 带宽）

- cold ~1.32M ns / nc=512 行 ≈ **2.6 µs/行**。每行 = 8 super-block × 144B 权重 = 1152B 连续。
- 若在带宽（~10GB/s/核）：1152B ≈ 0.115 µs。实测 2.6 µs = **~22× 慢于带宽** ⟹ **纯延迟暴露·非带宽饱和**。
- 每 super-block 暴露 ~1 次 DRAM 延迟（~300ns）× 8 ≈ 2.4 µs ≈ 实测。⟹ 我方 = **8 次串行 DRAM 延迟/行**；对手 = 多流在飞（MLP）→ 逼近带宽 → 5.4× 快（ratio 0.186）。

**根因一句话**：我方 serialize = **窄 load + load 紧贴消费者 + 单累加器**导致 **super-block 内 outstanding memory request 太少（低 MLP）**，把本可并行在飞的 DRAM 延迟**逐 super-block 串行暴露**；三 lever 只动指令流轴、从未动 load 宽度与 load 流并行度，故 cold 全 inert。
