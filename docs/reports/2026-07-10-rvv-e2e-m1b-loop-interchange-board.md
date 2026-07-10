# [RVV-E2E] M1b-board — col-outer loop-interchange 板判读：翻正 trigger MET（2026-07-10）

**板** `ssh rvv`（VLEN128，taskset 8-15，8 threads，interleaved rounds）。部署 = 正门 clang-17 `.o`（重导 committed emitter d33dd545 的 col-outer q4_K，C md5 01f3fe9c、lib 488a9da1，via 0c LD_PRELOAD）。未重建 tcrv-opt、未改 emitter。A-tree 复原 stock 75f20b5f（nm 0/0）。genuine。禁 git。

## 五验（col-outer clang .o 真 engaged）
① 符号 DEFINED ② 反汇编 vs A_pre: **vwmacc=2240/vset=71/spill=3/reload=8 IDENTICAL**（byte-exact 热核，只 loop-scaffold 差）③ vl=8 ④ banner `col-outer clang-object ENGAGED` 每轮 ⑤ clang-17.0.6。

## A_new(col-outer) vs A_pre(row-outer, M0 0.764× 基线) — 3 轮 interleaved median
| 指标 | A_pre row-outer | A_new col-outer | Δ |
|---|---|---|---|
| **LLC-load-miss** | 4.62e9 | **0.89e9** | **↓5.2×（−80.6%）** |
| **backend-idle** | 85.81% | **66.20%** | **↓19.6pp** |
| **throughput avg_ts** | 2.793 | **6.896** | **↑2.47×** |
| cycles | 2.87e12 | 1.20e12 | ↓2.4× |
| IPC | 0.41 | 0.98 | ↑2.4× |

自检 IQR 0.19%（≪ 地板×1.5），load-robust（4.07→10.15 load 下 A_new 6.89-6.91 不变 = DRAM-bound 证）。M1a 的"36KB>L1d 驻留失败"风险**未兑现**（A_new LLC-miss 0.89e9 < Bq4kOFF 1.79e9 < Bstock 1.84e9，权重 stream-once 成，`weightStride≥activationStride` proxy held）。

## 预注册判读（用户二.M1b）— 三条件全 MET
1. DRAM miss 显著降 4.62e9→0.89e9 ✓ · 2. stall 显著降 85.8%→66.2% ✓ · 3. A_new/Bq4kOFF **1.87×≥parity** ✓ → **翻正 trigger SATISFIED**。

## ★双账本诚实框架（不重蹈编译器不对称）
- **kernel 账（compiler-symmetric，clean）= 干净 schedule 赢**：A_new/A_pre = **2.47× throughput**（两侧都我方 clang .o、byte-exact 热核、只 loop 序差）→ **M0 归因的 H-B 内存回归被 loop-interchange 修复**（LLC-miss ↓5.2×）。rvv e2e **不再是回归**。这是**纯 schedule 效应、编译器无关**。
- **系统账（toolchain-disclosed）= 绝对对位**：A_new/Bq4kOFF=**1.87×**（vs 同 build block-dot）· A_new/Bstock=**1.33×**（vs 上游 native llama.cpp）。**⚠ 对手 gcc-built（rvv 出货 gcc-15）→ 这两个绝对比值含 clang-vs-gcc 编译器分量（[CASE-COMPILER-ASYMMETRY] 系统账、非纯 kernel-account）**。要纯 kernel-account "beats block-dot" 数须补 **clang-block-dot 对称对位**（隔离 schedule 与 compiler）。
- **净口径**：**loop-interchange 是干净的 capability-keyed schedule 赢（2.47× 我方 before/after）**；rvv e2e 翻正 = 回归修复坐实；"beats block-dot/upstream 1.87×/1.33×" 归系统账、待 clang-symmetric 补测定 kernel-account。

## [CACHE-HINT] 首个实战客户
col-outer schedule 键控于 repack layout stride fact（weightStride 2304 ≥ activationStride 1168）= **首个 capability-derived cache-locality schedule 赢**（非常量、可逆两臂）。残留 backend-idle 66.2% = activation re-stream + serial fp16-scale unpack headroom（精确-L1d-key 未来机会，非阻塞）。

## micro↛e2e 案例闭环
1.884× micro 是**热缓存 artifact**（L2 常驻藏权重重读）；冷 e2e 需 schedule 修（M1b 交付）→ 现 micro 的算力优势经 loop-interchange 在 e2e 兑现。**GAP-1 闭环实证客户**：L2 调度/布局对内存层级的适配缺口，识别→归因(M0)→修复(M1b)→板证。
