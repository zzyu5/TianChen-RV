# [CASE-MICRO-E2E] 卷宗 — 热 micro 赢 ↛ 冷 e2e 输 → 单变量 schedule 修复 → 翻正

> 建档 2026-07-10 · 分支 `refactor/full-refactor-m1` · 纯文档卷宗（零 build/板/perf/git；一手数字全为已封存证据的指针汇编）。
> 与 `[CASE-MINTERM]`（T8 r101-131 banner）、`[CASE-COMPILER-ASYMMETRY]`（编译器不对称普查）并列结构 = **C3′ 系统边界章首选案例**。append-only；数值不改。
>
> **一句话**：q4_K S6 kernel **热 micro 1.884×**（L2 常驻、权重重读被藏）→ **冷 e2e 0.764× LOSS**（编译器身份控住后仍输）→ perf 计数器归因 **单因 H-B 内存停顿**（backend-idle 85.8%/IPC 0.41/LLC-miss 2.58× 于 block-dot、却少 2.3× 指令 = 纯访存停顿）→ **单变量 loop-interchange**（byte-exact、objdump PRE==POST，只换循环序）→ **翻正 2.47×**（kernel 账、before/after 编译器对称，LLC-miss ↓5.2×）。**证 "能力键控必须延伸到内存层级 schedule，否则 micro 优势不传导。"**
>
> **与 [CASE-COMPILER-ASYMMETRY] 的分界（关键，勿混）**：那案管辖 **0.334×**（gcc 部署 = 我方 kernel codegen 被 gcc-15 全展开 spill 拖垮的编译器 artifact）；本案管辖 **0.764×**（clang `.o` 部署 = 我方 kernel codegen 已与 micro seal 逐字同质，编译器轴被控住后**残留的**内存-schedule 因子）。**两案共用同一 micro 1.884×，但归因不同层**：compiler-asymmetry 案剥离"我方 kernel 用哪个编译器编"这一 confound，本案剥离"热缓存藏权重重读"这一 confound。两个 confound 叠加才是 gcc-e2e 的 0.334×。

---

## 1. 案由（全链闭环）

**触发链**：性能宪章 rule 1「修性能前先反汇编认瓶颈」+ rule 2「一切选择键值 per-format 板测定」。q4_K S6 tile 在热 micro 报 1.884×（曾入 8-gate ④），部署到 rvv e2e prefill 却是 **LOSS**。将编译器身份控住（部署 clang `.o` 而非 gcc）后**仍**输 0.764× → 说明剩余缺口不在编译器、在 schedule。

**决定性证据链（四发咬合）**：
1. **热↔冷同 kernel 反转**：micro 1.884×（L2 常驻热跑）↛ e2e 0.764×（DRAM 冷流），部署 clang `.o` 反汇编逐字同质于 micro seal（71 vsetvli/3 spill/2240 vwmacc）→ 反转**不可能来自我方 kernel codegen**。
2. **perf 计数器单因**：我方 e2e backend-idle 85.8%、IPC 0.41、LLC-load-miss 4.63e9（=block-dot 1.80e9 的 **2.58×**），却**少 2.3× 指令**（1.172e12 vs 2.655e12）→ 慢**纯来自访存停顿**（多出的 1.12e12 停顿周期淹没 0.43e12 计算节省）。
3. **objdump 循环序 = 根因**：部署 kernel row-group 外(:15)→col-group 中(:20)→K 内(:41)，权重基址只依赖 col-group → 每 4-token row-group 重扫整块权重面板，nr=128 → 重读 **32×**（对手 16×16 cache-blocking = 8×，我方 **4× 权重流量**）。
4. **单变量修复翻正**：col-outer loop-interchange（byte-exact + objdump PRE==POST）→ LLC-miss ↓5.2×、throughput ↑**2.47×**（kernel 账）→ e2e 历史回归关闭。

**核心机制陈述**：**1.884× 是热缓存 artifact**（K=2048/nr=64/nc=512、20 迭代 L2 常驻，权重重读被 L2 驻留整个藏住，从不为权重付 DRAM）。冷 e2e 权重面板 ~1MB/线程×4/L2=4MB **超 L2** → 32× 重读落 L3/DRAM，热跑藏住的罚项全额浮现。**micro 的算力优势要经 schedule 修复才在 e2e 兑现。**

---

## 2. 全链逐环（数字 + 一手指针）

### 环 1 — 热 micro 赢（是热缓存 artifact）
- q4_K S6 kernel micro **1.884×** vs block-dot（nr=64、20 迭代、工作集 L2 常驻）。
- 成色：**热上界**，权重重读被 L2 驻留藏；同 kernel 冷跑（工作集 > L2）无此复用 → 见环 2。
- （附注：此 1.884× 另在 [CASE-COMPILER-ASYMMETRY] 被判 = clang-ours-vs-gcc-shipped 编译器不对称。本案不重复该轴，只用其"控住编译器后仍存在的 schedule confound"结论——即环 2 的 clang `.o` 部署。）
- **指针**：`experiments/active/l1-tile-s6-q4k-repack-gemm/tile_s6_findings.md`（clang-17 71 vsetvli/3 spill）。

### 环 2 — 冷 e2e 输（部署身份修复后仍输）
- rvv e2e prefill **A_clang / Bq4kOFF = 0.764×**（compiler 轴控住：部署 gemm.o = 71 vsetvli/3 spill/2240 vwmacc，与 micro seal 逐字同质）。
- 区别于 gcc 部署的 0.334×（那次含 gcc-codegen confound，归 [CASE-COMPILER-ASYMMETRY]）；0.764× 是**编译器被控住后的纯 schedule 残差**。
- avg_ts 2.798（我方 A_clang）vs 3.680（block-dot Bq4kOFF）。
- **指针**：`docs/reports/2026-07-10-q4k-micro-vs-e2e-regression-mechanism.md`（0.334× gcc 版机制 + salvaged 原始）+ `experiments/active/result-tables/T-PERF1b_q4k_e2e_prefill_regression.md`。

### 环 3 — 计数器归因（M0，单因 H-B 内存停顿）
| 指标 | 我方 A_clang | block-dot Bq4kOFF | ours/opp |
|---|---|---|---|
| avg_ts | 2.798 | 3.680 | **0.760×** |
| instructions | 1.172e12 | 2.655e12 | **0.44×（我方少 2.3×）** |
| IPC | 0.41 | 1.22 | 0.34× |
| backend-idle | **85.8%** | 61.7% | +24pt |
| **L3(LLC)-load-miss** | **4.63e9** | 1.80e9 | **2.58×** |
| dTLB-load-miss | 408e6 | 549e6 | 0.74×（我方更少 → 非 TLB） |

- **三假设裁断**：H-A repack 计时归属 → REFUTED（~0%，走装载期 set_tensor 非计时窗）；**H-B 冷流布局 → CONFIRMED 唯一主导（~100%）**；H-C 形状失配 → REFUTED as framed（4-row token 瓦片是固定属性、绑死 q8_K INTER_SIZE=4，是 schedule/tiling 交互非独立形状因子）。
- **归因分解 = 单因**（H-B ≈100%）。根因 = kernel 循环序锁死 4-token 浅瓦片（对手 16-token 深瓦片），冷流权重重读 32× vs 对手 8×。
- **指针**：`docs/reports/2026-07-10-rvv-e2e-m0-attribution.md`。

### 环 4 — 单变量 schedule 修复（M1b）
- **loop-interchange（col-group 外）**：把权重变常驻只流一次、改重读小的 activation 面板（18KB、L1/L2 友好）。
- **单变量证据**：byte-exact（0-mismatch vs golden）+ objdump **PRE==POST**（vwmacc=2240/vset=71/spill=3/reload=8 IDENTICAL，累加器/算术未动、**只换循环序**）→ 修复量 100% 归 schedule。
- **capability-keyed**：col-outer schedule 键控于 repack layout stride fact（weightStride 2304 ≥ activationStride 1168）= 非常量、可逆两臂。
- **指针**：M1b 构造 commit **d33dd545**（`task(RVV-E2E M1b 构造): q4_K loop-interchange(col-group 外)front-door capability-keyed — byte-exact + objdump PRE==POST`）。

### 环 5 — 翻正（M1b-board）
| 指标 | A_pre row-outer（M0 基线） | A_new col-outer | Δ |
|---|---|---|---|
| **LLC-load-miss** | 4.62e9 | **0.89e9** | **↓5.2×（−80.6%）** |
| **backend-idle** | 85.81% | **66.20%** | **↓19.6pp** |
| **throughput avg_ts** | 2.793 | **6.896** | **↑2.47×** |
| cycles | 2.87e12 | 1.20e12 | ↓2.4× |
| IPC | 0.41 | 0.98 | ↑2.4× |

- **预注册判读三条件全 MET**：DRAM miss 显著降 ✓ · stall 显著降 ✓ · A_new/Bq4kOFF **1.87×≥parity** ✓ → 翻正 trigger SATISFIED → **rvv e2e 历史回归关闭**。
- **load-robust**（4.07→10.15 load 下 A_new 6.89-6.91 不变 = DRAM-bound 证）；自检 IQR 0.19%；weight stream-once 成（A_new LLC-miss 0.89e9 < Bq4kOFF 1.79e9）。
- **系统账绝对 1.87×（vs Bq4kOFF）/ 1.33×（vs Bstock）标「对手 gcc、待 clang-symmetric」**（见 §6 双账本）。
- **指针**：`docs/reports/2026-07-10-rvv-e2e-m1b-loop-interchange-board.md`。

---

## 3. 归因方法学（三支柱，contention-immune 优先）

本案归因**不依赖单点 perf 数**，三支柱交叉锁定，其中两支 contention-immune（争用无关）：

1. **perf 计数器（PMU，无 multiplex，counter IQR<0.15%）** → *判别子 = backend-idle% 差 + LLC-load-miss 比*。我方 busy 周期反而少一半（少指令），stall 周期多 1.83× → 排除计算质量/指令微质量，锁定访存停顿。dTLB-miss 更少 → 排除 TLB。
2. **objdump / 静态反汇编（contention-immune）** → 循环序（row-outer :15/:20/:41）证权重基址依赖链 → 量化重读倍数（我方 32× / 对手 8×）；且 **PRE==POST objdump** 证 M1b 是纯单变量（热核 byte-exact、只 loop-scaffold 差）。
3. **cache 拓扑账（板 spec，static）** → L1d 64KB/核、L2 2MB/4核、L3 64MB；ffn 权重面板 ~1MB/线程×4/L2=4MB **超 L2** → 32× 重读必落 L3/DRAM。热 micro（20 迭代 L2 常驻）从不为权重付 DRAM → **精确解释 1.884×↛0.764× 为何是热缓存 artifact**。

**方法学修（入档）**：micro 必须**冷跑**（单遍、工作集 > L2）或明标"热上界"；projection 的 kernel 因子须在部署目标同一编译器 + 冷缓存下测得。**这是我方 kernel 的 schedule 问题、M1 可修**（区别于 0.334× 的 gcc-codegen 不可控）。

---

## 4. 跨板姐妹案 — k1 VLEN-adaptivity → Win-K1-VLEN

同"能力键控延伸到板形状出核"主题的**第二腿**（cache 拓扑 → 换成 lane 宽度）：

- **半宽→满宽**：K1-SEAL vl=8 = **0.750×** vs hand-brick（半 VLEN256 宽）→ **vl=16 满宽 = 1.085×**（95%CI [1.0828, 1.0884]，10 轮配对，CI 全 >1.0，min-ratio 1.0806 最差轮仍 >parity）。
- **纯 capability-input**：emit 喂 VLEN256 fact → `deriveMinimumVLEN→256→half_lanes=16→numHalves=1→vl=16`，**无新手写核 / 无格式名分派 / 无搜索**。五验在案（vwmacc 1120=vl8 的一半、`vsetivli zero,0x10,e32,m2`、byte-exact 8/8）。
- **编译器对称 = 干净 kernel-account**：k1 出货 ggml = clang-18 → A_vl16（我方 clang-18）vs hand-brick（clang-18）= compiler-symmetric，**无 [CASE-COMPILER-ASYMMETRY] 系统账 caveat**；且对手 = 真 as-shipped hand-brick（非 block-dot strawman）。
- **= 首个 sealed Win（Win-K1-VLEN，登记 2026-07-10 用户裁定）**。随附限制（登记档不可分割）：⑤双板门开放（rvv 待 clang-symmetric）、ABI shim 在途、未叠加 col-outer。
- **指针**：`docs/reports/2026-07-10-vlen-adapt-m1-k1-sealed-win-candidate.md`（1.085× CI + 五验 + byte-exact）+ `docs/reports/SEALED-WIN-REGISTRY.md`（Win #1 逐字措辞）。

**两案共同点**：**能力事实（cache 拓扑 / VLEN 宽度）必须驱动 schedule / lane-width，否则 micro↛e2e**。rvv 腿 = cache 拓扑键控内存层级 schedule；k1 腿 = VLEN 宽度键控 lane-width。两腿同底线 = **能力键控必须延伸到板形状出核**。

---

## 5. C3′ 系统边界论点

### [CACHE-HINT] 首个实战客户
- M1b col-outer schedule 键控于 **repack layout stride fact**（weightStride 2304 ≥ activationStride 1168）= **首个 capability-derived cache-locality schedule 赢**（非常量、可逆两臂）。
- 残留 backend-idle 66.2% = activation re-stream + serial fp16-scale unpack headroom = 精确-L1d-key 未来机会（[CACHE-HINT] 加深客户，非阻塞）。

### 核心论点：能力键控延伸到内存层级 schedule
> **一个 kernel 的算力优势（热 micro 赢）不会自动传导到部署 e2e——除非能力键控延伸到内存层级 schedule（cache 拓扑）与板形状出核（lane 宽度）。** 键控只到"选哪个 kernel/格式"层不够；必须键控到"kernel 内循环序如何适配 cache 层级 + lane 宽度如何适配 VLEN"。

- 这是 **GAP-1 闭环的实证客户**：L2 调度/布局对内存层级的适配缺口，**识别 → 归因(M0) → 修复(M1b) → 板证**四段完整。
- **两腿证同一论点**：rvv（cache 拓扑，schedule 轴）+ k1（VLEN 宽度，lane 轴），跨板同主题。
- 与 [CASE-COMPILER-ASYMMETRY] 互补：那案证"选对 kernel 后编译器身份仍能翻盘"（工具链边界）；本案证"编译器控住后 schedule 仍能翻盘"（内存层级边界）。三案（+[CASE-MINTERM] 正确性边界）共同勾勒 C3′ 系统边界地图。

---

## 6. 双账本口径（[NG-4] 措辞锁）

**rvv M1b（本案主腿）——两账本分列，勿混**：

| 账本 | 数字 | 口径 | 成色 |
|---|---|---|---|
| **kernel 账（schedule，clean）** | **A_new/A_pre = 2.47× throughput** | 两侧都我方 clang `.o`、byte-exact 热核、**只 loop 序差** = compiler-symmetric | **干净 schedule 赢、编译器无关**；M0 归因的 H-B 内存回归被 loop-interchange 修复（LLC-miss ↓5.2×）→ rvv e2e **不再是回归** |
| **系统账（toolchain-disclosed）** | A_new/Bq4kOFF=**1.87×** · A_new/Bstock=**1.33×** | vs 同 build block-dot / 上游 native | **⚠ 对手 gcc-built（rvv 出货 gcc-15）→ 含 clang-vs-gcc 编译器分量**（[CASE-COMPILER-ASYMMETRY] 系统账、非纯 kernel-account）；要纯 "beats block-dot" 须补 clang-block-dot 对称对位 |

- **净口径（rvv）**：loop-interchange 是**干净的 capability-keyed schedule 赢（2.47× 我方 before/after）**；rvv e2e 翻正 = 回归修复坐实；"beats block-dot/upstream 1.87×/1.33×" **归系统账、待 clang-symmetric 补测**（[FU-2]）定 kernel-account。

**k1 姐妹腿——单账本，无 caveat**：
- k1 vl=16 = **1.085× vs as-shipped hand-brick，编译器对称（clang-18 双侧）= 干净 kernel-account，无系统账 caveat**（区别于 rvv 的 clang-vs-gcc 绝对数）。
- 但 k1 有其自身残留（ABI shim、⑤双板门、未叠 col-outer）——见 §4 / 登记册。

**诚实基线**：rvv 系统账带编译器不对称 caveat（对手 gcc）；k1 clean（对称-clang）。**两板皆无"跨板共同 win"表述**（⑤双板门开放）。

---

## 7. 论文素材定位

- **C3′ 系统边界章首选案例**（与 [CASE-MINTERM] 正确性边界、[CASE-COMPILER-ASYMMETRY] 工具链边界并列）。本案 = **内存层级 / 板形状边界**。
- **headline 候选**：Win-K1-VLEN（k1 vl=16 1.085× sealed，kernel-account clean 满宽机制构造）= C1/C3′ 核心主张的**性能实证**（能力事实驱动构造）。**并列**（非合并）q4_0 5.9×（系统账 routing 白嫖）——成色不同、分开陈述。
- **micro↛e2e 机制级证据**：PMU 分解 + 循环序 objdump + cache 拓扑账 = 三支柱交叉，其中两支 contention-immune → **档案级素材**（无论后续叠加与否，本审计即成立）。
- **C3′ 增量章骨架**：能力键控层级表——L(选格式) < L(选 kernel/变体) < **L(schedule 适配 cache 层级)** < **L(lane 宽度适配 VLEN)**；本案证后两层是"micro↛e2e 是否传导"的决定层。
- **相关 memory**：`[[kernel-wins-dont-transplant-to-e2e]]`（compute-bound micro 胜↛memory-bound decode e2e）本案给出**修复腿**（schedule 修复令传导发生）；`[[repack-campaign-terrain]]`（Win-B 战场）；`[[perf-constitution-three-layers]]`（L2 调度赢 = 能力键控）。

---

## 附：一手证据指针（全为已封存证据，本卷宗零新测）

| 环 | 指针 |
|---|---|
| 环1 热 micro | `experiments/active/l1-tile-s6-q4k-repack-gemm/tile_s6_findings.md`（clang-17 71/3） |
| 环2 冷 e2e | `docs/reports/2026-07-10-q4k-micro-vs-e2e-regression-mechanism.md` · `experiments/active/result-tables/T-PERF1b_q4k_e2e_prefill_regression.md` |
| 环3 M0 归因 | `docs/reports/2026-07-10-rvv-e2e-m0-attribution.md` |
| 环4 M1b 构造 | commit **d33dd545**（byte-exact + objdump PRE==POST） |
| 环5 M1b 板证 | `docs/reports/2026-07-10-rvv-e2e-m1b-loop-interchange-board.md` |
| 姐妹案 k1 | `docs/reports/2026-07-10-vlen-adapt-m1-k1-sealed-win-candidate.md` · `docs/reports/SEALED-WIN-REGISTRY.md`（Win #1） |
| 分界（工具链边界） | `docs/reports/2026-07-10-CASE-COMPILER-ASYMMETRY-casefile.md`（0.334× gcc 版 / 编译器不对称） |

---

*本卷宗纯文档、零 git、零板/perf 新测；一手数字全为上表已封存证据的指针汇编。措辞按 [NG-4] 双账本口径锁定（rvv kernel 账 2.47% clean / 系统账绝对待 clang-symmetric；k1 clean 对称-clang）。sealed Win 登记 = 主会话+用户裁决权，本卷宗只引用不自宣。与 [CASE-MINTERM]/[CASE-COMPILER-ASYMMETRY] 并列，C3′ 系统边界章首选案例。*
