# [RVV-E2E] M1a token-tile 深度选型表 — 静态账先行（2026-07-10）

**Scope**: 纯静态分析（objdump + emitter 源 + cache 拓扑账）。**不动刀 / 不改代码 / 不上板 / 不 build tcrv-opt / 不 git**。
供 M1b 曳光弹选型用。上游归因见 `docs/reports/2026-07-10-rvv-e2e-m0-attribution.md`（0.764× 单因 = H-B 冷流权重停顿，
我方权重重读 **32×** vs 对手 16×16 cache-block **8×** = 4× 权重 DRAM 流量）。register-cliff 现状见
`experiments/active/l1-tile-s6-q4k-repack-gemm/tile_s6_findings.md`（S6 把 hot-loop spill 23→0、峰 vreg **v30 (=31 reg)**、byte-exact）。

## 0. 部署 kernel 的结构事实（emitter `emitRepackKQuantGemmBodyQ4K`, RVVToEmitCBlockQuantLinear.cpp:6747；md5 90d454da）

循环序（外→内）：`for y = row-group(nr/4)` **外** → `for x = col-group(nc/16)` **中** → `for h = half(numHalves=2@VLEN128)`
→ `for l = block(nb)` → `for j = super-half(2)` → `for pair(2)` → `for k(2)` → `for ii(16)` 热点乘加。
- **权重基址 `bGroup` 只依赖 x（中层）** → 每个 y (=4 token) 把整块权重面板重扫一遍 → 重读 nr/4 = **32×**（M0 CONFIRMED）。
- **token 瓦片深度 = `activationInterleave` = 4**（block_q8_Kx4，`columnsPerPass=4` 在 rvv1.0 mf2 一趟折完 4 列）。加深 = 提高同时在寄存器里的**列累加器扇出 d**，且需上游把激活重排成 q8_K**x d**（ABI 改动）。
- 核心链（VLEN128 mf2）：i8mf2→i16m1→i32m2→f32m2。**每列**活寄存器：`sumf` f32m2=**2 reg** + `sumi` i32m2=**2 reg** + `sLo/sHi` 2×i16m1=**2 reg** = **6 reg/列**。`bsums` 已被 S6 stack-panel 挪出热集（0 reg）。
- Cache 拓扑（board rvv VLEN128）：L1d **64KB**/核，L2 **2MB**/4核（~512KB/核），L3 **64MB**。一个 col-group 权重（K=4096, nb=16）= 2304×16 = **36KB → 装得进 L1d**；全行激活工作集 = 1168×16×32 = **584KB**（>L2/核，≪L3）。

## 1. 三档 × 三账矩阵

寄存器峰值模型（S6 实测标定 C=7）：**peak\_hot(d) = 6d + 7**（6d = 2d sumf + 2d sumi + 2d sLo/sHi），预算 = **32 vreg**。

| token-tile 深度 d | ① 寄存器峰值账 (6d+7 vs 32) | ② 权重重读 / DRAM 流量 (nr/d;  |B_gate|=33MB) | ③ 结论 |
|---|---|---|---|
| **4（现，S6）** | **31 reg** (v30, spill=0) — 贴崖 0-spill，MEASURED | **32×** → 1057 MB → 4× 对手(8×) | 权重流量瓶颈（H-B 根因）|
| **8（半深）** | **55 reg** → 超崖 **+23** → spill≈23（回到 S1 前的压力档）→ **S6 白调** | 16× → 528 MB → 2× 对手 | ★**越崖**：流量减半但 spill 抵消 |
| **16（对齐对手）** | **103 reg** → 超崖 **+71** → spill≈71（**比 golden 全展开 84-spill 还差**）| 8× → 264 MB → 1× 对手 parity | ★**重度越崖**：流量达标但寄存器灾难 |

**全 panel 下限**（把 sumf+sumi 也 panel 掉、只留 sLo/sHi 热 = 2d+7；代价：sumi 每 block 4×/列 store-reload 内存税 = 部分 re-roll-trap）：
d=4→15、d=8→**23（勉强 ≤32，但带 sumi 内存税）**、d=16→**39（即便全 panel 仍 >32，无解）**。
⇒ **累加器扇出实现的深瓦片：d=16 数学上装不进 32 vreg（连全 panel 都溢），d=8 仅靠 sumi paneling 勉强装下（带内存税）。**

## 2. ③ 张力（加深 token → 权重流量↓ 但寄存器压力↑）

- 加深 d：权重 DRAM 流量 ∝ nr/d（32→16→8×，**单调↓**）；寄存器需求 6d（31→55→103，**单调↑**，d≥8 越 32-崖）。
- **甜点不在扇出轴上**：d=8 把权重减半却把 hot-spill 从 0 打回 ~23（S6 撤销，spill/reload 又是内存往返，很可能把省下的权重 DRAM 又吐回去 = **net wash**）；d=16 把 spill 打到 ~71（比全展开还糟）→ **几乎必为 net-negative**（用权重 DRAM 换 spill DRAM）。
- **关键解耦**：权重流量↓**不需要**累加器扇出↑，只需要**权重面板在 cache 里常驻**跨多个 token 组。一旦解耦，寄存器账与流量账不再对冲。见 §4。

## 3. ④ 混合方案 "8-token + Zicbop 预取" — 账上是否优于纯 16-token？

两种读法：
- **读法 A（8-token = 累加器扇出 d=8）+ Zicbop**：peak=55 reg → 仍越崖 spill≈23。Zicbop 预取隐的是**内存延迟**，解不了**寄存器文件溢出**（spill 是 regfile 不是 memory-latency）。⇒ 混合-8 仍带 spill，**不解决核心 tension**；只比纯-16（spill≈71）"没那么糟"，但都在越崖侧。
- **读法 B（8-token = 2 row-group cache-block，d 仍=4）+ Zicbop**：寄存器安全（d=4，S6 保留），需同时 col-tile 让权重跨 2 组常驻 → 权重重读 16×（半）。Zicbop 预取下一权重 tile 隐 1× DRAM 延迟。**可行且寄存器零成本**，但**被完整 loop-interchange 严格支配**（后者 1× 权重、同样 d=4 安全）。

**裁断**：混合-8 优于纯-16（纯-16 无论 A/B 都更糟：扇出 spill≈71，或需 4 组 col-block 更复杂）。但混合-8 只是**中间档**——比累加器深瓦片安全、比完整交换弱。Zicbop 的正确定位 = **延迟对冲的辅助件（[CACHE-HINT] 首客）**，配 §4 交换用，不是修 spill 的手段。**Zicbop 须先验板 capability（Zicbop 存在性）**，缺则退化为纯交换（交换已把流量降到带宽非瓶颈，延迟是次要项）。

## 4. ⑤ 交换外两层循环（col-group 外 / row-group 中）— 寄存器零成本的流量解

把最外两层 `for y`⇄`for x` 对调（`bGroup` 提到外层）：
```
for x = col-group:            # 外：权重基址提到这里
  bGroup = f(x)               # 一个 col-group 权重 = 36KB → 常驻 L1d
  for y = row-group:          # 中：内层扫所有 token 行
    ... 内层 (h/l/j/pair/k/ii) 与累加器结构完全不变，d=4 ...
    store out[y,x]            # 每个输出仍只算一次、K 累加序不变
```
- **② 权重流量**：一个 col-group 权重 36KB **常驻 L1d**，跨整个 y-sweep 复用 → 权重 DRAM **1×**（33MB）——**比对手 8× 还好 8 倍**。激活重读 nc/16 次但工作集仅 **584KB ≪ L3(64MB)** → 从 **L3 供**（BW ≫ DRAM），非 DRAM。
- **① 寄存器**：累加器结构**完全不动**（d=4，peak=31，spill=0，**S6 完整保留**）。就是两个外层 counter 循环对调 + `bGroup` 提升，~2 行改动。
- **正确性**：纯循环重排，每个 out[y,x] 的 K 累加序与 fold 序**逐字不变**，不同 (y,x) 独立 → **byte-exact**（vwmacc multiset 不变，objdump 内层同质，只外层 header 变）。
- **无需**激活 ABI 改动（保持 q8_Kx4），**无需**加寄存器。这是 M0 说的"更简替代 = 直接反转罚项（prefill 权重≫激活，应 hold 权重）"。

## 5. 推荐选型 + 依据（M1a 自决：工程参数，非战役级）

**推荐 M1b 曳光弹 = 交换外两层循环（col-group 外 / row-group 中），d 保持 4；Zicbop 预取下一 col-group 权重作辅助延迟对冲（capability-gated）。**

依据（三账合议）：
1. **寄存器账**：交换 = **零成本**（peak 31 不变，S6 的 ≤32-vreg 崖 + hot-spill 0 完整保留）。累加器深瓦片 d=8 越崖 +23、d=16 越崖 +71（连全 panel 都 >32），会**撤销 S6**——静态上是负项。
2. **流量账**：交换把权重 DRAM 32×→**1×**（33MB），**strictly ≤ 对手 8×**；激活重读从 L3 供（584KB≪64MB）。深瓦片 d=16 才勉强到 8× parity 且带灾难 spill。交换在流量轴**同时打赢深瓦片和对手**。
3. **张力**：交换**解耦**了 tension（流量↓ 不再要寄存器↑），是唯一同时满足两账的点；深瓦片在扇出轴上无甜点（d=8 wash，d=16 net-neg）。
4. **改动面 / 可逆性**：~2 行 loop-swap + 提升 `bGroup`，byte-exact，front-door（emitter schedule 属性），完全可逆。

**Zicbop 定位**：交换已把权重变 1× 带宽非瓶颈 → Zicbop 只隐残余 DRAM 延迟（预取下一 col-group 36KB），是**次要增益、非必需**；须先验板 Zicbop capability，缺则纯交换即可。混合-8/纯-16 累加器深瓦片**不推荐**（越崖）。

## 6. M1b 实现要点（给后派实现臂；等 K1-SEAL 跨板对照臂齐后另派）

- **Front-door 路径**：改 `emitRepackKQuantGemmBodyQ4K`（RVVToEmitCBlockQuantLinear.cpp:7024-7455）的最外两层 `emitc::ForOp` 嵌套顺序（rowLoop⇄colLoop 对调，`bGroup` 计算移到 col-loop body 顶）。**禁**手改部署 .inc / 禁旁路——从 emitter 出，走 lower→translate→emit 全链。
- **token-tile / loop-order 作 schedule 轴（capability-keyed，禁硬编码）**：把循环序（col-outer vs row-outer）做成 loop-body op 的 schedule 属性，键控依据 = cache 事实（col-group 权重 ≤ L1d ⇒ col-outer；类比现有 `numHalves = weightInterleave/half` 的 VLEN-keyed tile 计数），**不写死常量**。Zicbop 预取键控 = 板 Zicbop capability 事实（缺则不发预取指令）。
- **byte-exact gate（复用 S6 harness 型）**：交换前后 raw fp32 输出 `cmp`（int + norm 两 regime，nr64 & nr16）必须 0-mismatch；objdump 内层 vwmacc=2240 不变、maxVreg≤v30 不变（外层 header 变、热核同质）。
- **板 A/B（M1b 上板）**：M0 recipe（`/tmp/case_0c/m0_perf.sh` 事件集）测交换前/后的 **LLC-load-miss 比 + backend-idle%**——静态预测权重 DRAM 1×⇒ LLC-load-miss 应从 4.63e9 大幅回落、backend-idle 从 85.8% 下压。这是把静态账兑现成 e2e 的判别子。
- **禁重试已证伪偏方**：累加器深瓦片 d≥8（含"8-token 扇出+Zicbop"读法 A）静态已证越崖，M1b **不试**；如需 token 深度收益走"读法 B"(cache-block d=4) 亦被交换支配，优先交换。
```
