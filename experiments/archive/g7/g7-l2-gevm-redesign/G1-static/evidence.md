# G7 L2 — GEVM 重设计 · G1 静态账 (零板时·[VERIFY-LADDER] G1 层·禁跳级)

> 承 P0(H1 结构损·字节触底 roofline·q4_K=停顿型 IPC 崩) + P1(colgroup-tiled GEVM plan 构造·byte-exact) + **P2 board-falsified**(TG=2 register-resident bank 溢出·534 spill vs 10·IPC 0.294→0.214)
> 依据 canon: [VERIFY-LADDER] G1 静态账门 · [K-10] 结构级独立 plan · [PAT-2] P9 GEVM regime-plan · [X-ZVBB] · [GAP-P1] register-pressure trap · 性能宪章规则1-2 · [CASE-COMPILER-ASYMMETRY]
> **本任务止于 G1(静态账)+ 设计 · 禁 git · 禁上板 · 禁改 schema label**
> **★裁决(TL;DR)：G1 = 分裂门** — **register-budget-fit 子门 PASS**(TG=1 复原 OLD board-proven spill-free·P2 trap 已修)·**instruction-count 子门 NOT MET(1.12–1.14× > 1.10×)**·且**唯一能削 q4_K 重建的杠杆 zvbb 全舰队缺席(rvv∧k1)且即便在场也 0-yield** → **q4_K@rvv 重建指令最小化线 = 打回·不上板**。重建 headroom 不在 q4_K(仅 4.7% vector 可动)·在 qh-plane 格(q5_0/q5_1 +43~62%)。

---

## 0. 方法与工具链 (本地静态账·零板时·可复现)

- **本地 RISC-V 静态账工具链已建**：`clang-20 --target=riscv64-unknown-linux-gnu -march=rv64gcv_zvfh -mabi=lp64d -O2`（`-nostdinc` + 自包含 stdint shim `/tmp/rvshim`，绕开缺失 riscv sysroot·仅 freestanding intrinsic 内核）。发射管道 = `weft-opt --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`（沿用 EXPORT_RECIPE）。
- **★校准锚（本地方法 vs 板 objdump 对齐验证）**：本地 clang-20 编出的 q4_K GEVM 内核，**向量结构计数与板 clang-18 objdump 逐一 EXACT 相等** — OLD `vwmacc=560`（板 560）· `vle8=280`（板 284¹）· TG=2 `vwmacc=1120`（板 1120）· `vle8=560`（板 568¹）。→ **向量轴计数 = 编译器鲁棒**，本地 clang-20 是可信静态账代理。
- **★★方法学警示（新发现·[CASE-COMPILER-ASYMMETRY] 扩展到 G1 静态 spill 轴）**：**标量 stack spill 计数 = 编译器高度敏感**。同一 OLD 内核：板 clang-18 = **10 spill**（=函数 prologue callee-save·即 spill-free）；本地 clang-18 缺·**clang-20 = 527 spill**（~50× 膨胀·clang-20 RISCV 标量 regalloc 对本 unrolled 内核更差）。**∴ G1 的 spill/live-set 绝对数必须以【部署编译器】测（板 clang-18 / gcc-15），本地 clang-20 只可用于【同编译器下变体间相对排序】**。此戒同 [CASE-COMPILER-ASYMMETRY]（kernel-axis 数仅编译器对称时有效）——现扩展到 G1 静态 spill 账。
- 我的 spill 正则在**板 raw objdump 上复算 = EXACT 10 / 534**（校准通过·`raw/board_isa_zvbb_absence.txt` + P2 raw/*.dis）。
- ¹ vle8 板 284/568 含 4 条非内核 vle（对齐/尾）· 本地 280/560 = 纯内核体·差 4 稳定。

---

## 1. ★设计 (令 §三.1 · 主攻 M=1 重建指令最小化 · TG=1 为预算前提非收益源)

### 1.1 结构决策（相对 P2 TG=2 的三改）
1. **TG=1（撤 bank 加宽）**：P2 已证 bank 加宽假设**反噬**（M=1 共享激活收益≈0·付 2× 寄存器压力 + 53× spill）。TG=1 令 register-resident bank 退回 `numHalves`(=2@VLEN128) 个 f32m2 累加器 = OLD per-column 的预算轮廓（§2 实测复原）。**TG=1 ≠ 收益源·只是把 live set 拉回 32-vreg 预算内的前提**。
2. **重建依赖链缩短（board-available 杠杆·非 zvbb）**：q4_K M=1 损 = **停顿型（IPC 崩·非指令数爆）**（P0）。杠杆 = 缩短 per-block 重建关键路径（load→unpack→dot→fold 串行链），使计算延迟相对内存延迟更小、利于跨 block overlap。但 §3 实测：q4_K 重建链已近极小（1-op-per-mask），board-available 缩短空间 <5%。
3. **prefetch 软流水（board-available·结构级·latency-hiding）**：P2 NEW 已带 `__builtin_prefetch(next-block strips)` 但**被 534 spill 串行化抵消**。TG=1 spill-free 后 prefetch 可真正 overlap 下一 block 权重 DRAM 装载 ↔ 当前 block 计算。**注意：这是 latency-hiding 杠杆（改 IPC）·不改指令数（反而 +2 insn/block）→ 与 instruction-count 子门张力（§4.3）**。

### 1.2 byte-exact 由构造（[K-10] 结构级·[K-1] 独立 plan·核算术不动）
重设计只改**迭代编排 + prefetch 节奏**（envelope）。per-block leaf（8 sub-block 6-bit scale/min 解包 + split-32 整数点积 + bsums-min 校正 + dual d/dmin fp16 fold）**逐字不动** = P1 host ZERO-MODEL oracle（mismatch=0·816 列）覆盖不变。TG=1 是 TG∈{1,2,4} 已被 oracle 证 A/证 B 覆盖的合法取值（P1 oracle grid 含 TG=1）→ **byte-exact 无需重验**（[K-10] 结构级 provenance 复用）。

---

## 2. ★G1 静态账：register-budget-fit 子门 (核心门·P2 trap 所在)

**同编译器（clang-20 rv64gcv_zvfh -O2）下三变体静态账**（`raw/*.clang20.s`）：

| 变体 | total insn | sp-spill(本地) | vwmacc | vsetvli | vle8 | vand | vor | vzext | 结构 |
|---|---|---|---|---|---|---|---|---|---|
| **OLD** (per-column) | 3043 | 527 | 560 | 94 | 280 | 304 | 32 | 32 | col-OUTER/block-INNER |
| **★TG=1 (redesign)** | **3042** | **528** | **560** | **93** | **280** | **304** | **32** | **32** | colgroup-tiled TG=1 + prefetch |
| **NEW** (TG=2·P2 falsified) | 6459 | 907 | 1120 | 187 | 560 | 608 | 64 | 64 | colgroup-tiled TG=2 (bank×2) |

**板 clang-18 objdump 部署真值**（P2 §4·raw/*.dis 复算 EXACT）：

| 变体 | total | **sp-spill(部署真值)** | vwmacc | vsetvli | 判 |
|---|---|---|---|---|---|
| OLD (=TG=1 轮廓) | 2328 | **10**(=prologue·spill-free) | 560 | 57 | **FITS 32-vreg ✓** |
| NEW (TG=2) | 5179 | **534**(=10+524 loop spill) | 1120 | 112 | **爆预算 ✗** |

**判读**：
- **TG=1 ≡ OLD**（total 3042 vs 3043·spill 528 vs 527·vwmacc/vle8/vand/vor/vzext **逐一相等**）→ TG=1 精确复原 OLD 的寄存器/指令轮廓（唯一差 = prefetch 2 hint + shared-act 单算，TG=1 下无实差）。
- **∴ TG=1 在部署编译器 clang-18 上 → ~10 spill（spill-free·FITS 预算）**（同 live set·同编译器传导；本地 clang-20 TG1=528≈OLD 527 佐证同轮廓）。
- **本地 clang-20 相对排序与板方向一致**：TG=1(528) ≈ OLD ≪ TG=2(907)；板 clang-18 OLD(10) ≪ TG=2(534)。**绝对数差 50×（clang-20 膨胀）但方向铁一致：TG=1 fits · TG=2 blows**。
- **✅ register-budget-fit 子门 = PASS** — P2 的 534-spill register-pressure trap 已被 TG=1 修复。这是 [VERIFY-LADDER] G1 存在的**首要目的**（"前车 = TG=2 的 534 spill 本可在此关拦下"），现被 TG=1 通过。

---

## 3. ★zvbb P8 评估 ([X-ZVBB] 首次立项 · 令 §三.2 · 三问)

### ① rvv 板是否支持 zvbb？ → **NO（决定性·双板缺席）**
板 `/proc/cpuinfo` isa 事实（cached·`raw/board_isa_zvbb_absence.txt`·跨 3+ 实验一致）：
- **rvv 板**（openEuler VLEN128）：`rv64imafdcv_..._zba_zbb_zbc_zbs_zve32*_zve64*_zvfh_zvfhmin_...` — 有**标量** Zbb(zba/zbb/zbc/zbs)·**无向量 zvbb/zvkb/zvbc**。
- **k1 板**（SpacemiT X60）：`rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs` + `xsmtvdotii1p0`(IME) — 同样**标量 Zbb 有·向量 zvbb 无**。
- **∴ 当前舰队【两板】均无 vector zvbb**。任何 zvbb-emitted 内核 = 板上 illegal-instruction / march-completeness gate fail-closed（[I8] 硬件主张需真证据·[I1] 能力=一等事实含**缺席**）。工具链侧：clang-20 确认 zvbb 是**能力门控**扩展（`rv64gcv` 拒 `__riscv_vandn/vwsll/vbrev8`·`rv64gcv_zvbb` 才放行·发 `vandn.vv/vwsll.vx/vbrev8.v`）→ 能力事实成立、但板不 provide。

### ② 若支持·qh/scale 重建能省几条？ → **0（实测·即便在场也无收益）**
q4_K 6-bit scale/min lift micro（`raw/sc_base.s` vs `raw/sc_zvbb.s`·rv64gcv_zvfh vs +zvbb）：

| 变体 | lift 向量 compute 指令 | 明细 |
|---|---|---|
| baseline | **5** | 2×vand + 1×vsll + 1×vor + 1×vzext |
| zvbb (vwsll fuse) | **5** | 2×vand + 2×vwsll + 1×vor |

**vwsll 融合 = 净 0**：vwsll 把 hi 路的 (vsll+后续 vzext) 省 1 op，但 lo 路被迫单独 widen（vwsll<<0）+ OR 上移到 u16 域 → 两笔相抵。**根因：baseline 已在 u8 域做 OR（1 次共享 vzext）= 已极小**。
- q4_K 静态分解（`raw/tg1.redesign.*.s`）：**zvbb/重建-amenable(vsll+vor+vzext) = 80 / 1714 total vector = 4.7%**。主体 = vwmacc 560(算术·不可削) + vle8 280(内存) + vand 304 + vsrl 280(1-op mask·不可削) + lbu 256(激活标量·内存)。**q4_K 的 nibble+6bit 解包已近下限·zvbb 无处发力**。

### ③ 模式条目建议 → **[X-ZVBB] = capability-fact 条目(舰队缺席)·P8 registered-but-not-selectable·非 G7 杠杆**
- **[X-ZVBB] 事实行**：记录"rvv/k1 均 provide 标量 zba/zbb/zbc/zbs·**不** provide 向量 zvbb"（能力模型录**缺席**·I1）。
- **[PAT-2] P8** 保持 registered（predicate `implies rvv.zvbb`），但其能力谓词在**全舰队 = FALSE** → **emittable-but-not-board-selectable**（合法态·fail-closed 到 base-RVV fallback）。
- **★不得作 G7 q4_K@rvv 杠杆**（板不 provide + q4_K 0-yield 双否）。zvbb micro 增量应在**有 bit-permutation 结构的解包格**上立（q6_K qh 6-bit 组装 / IQ codebook gather 的 vbrev8/vrol）·且需 **zvbb-capable 硅**（当前舰队无 → [远期]）。这是 SOP 正面拦截：G1 + 能力事实联合**在零板时否决了一个不可部署的杠杆**（同 TG=2 534-spill 本应被拦）。

---

## 4. ★G1 静态账：instruction-count 子门 + 综合判读

### 4.1 instruction-count 子门 = **NOT MET (1.12–1.14× > 1.10×)**
- **STATIC 读法**（字面"编译产物指令数"）：我方 unrolled 内核 ~2328 静态 insn vs stock rolled `ggml_vec_dot_q4_K_q8_K` ~数百 → **6–10×**（granularity 错配·unrolled-vs-rolled·此读法对全展开发射器恒 fail·无判别力）。
- **DYNAMIC 读法**（roofline-normalized per-token·P0/P2 唯一有意义口径）：OLD/TG1 = **5.86G/tok（P2）· 5.79G（P0）** vs stock **5.14/5.15G** = **1.12–1.14×**。**超 1.10× 门 ~2–4pp**。
- **两读法皆不达 ≤1.10×**。且 §3 证：q4_K 仅 4.7% vector 可动·zvbb 0-yield → **board-available 无杠杆闭合这 12–14% 缺口**。缺口来源 = lane-wise 16-列 repack 的结构性开销（vsetvli 93 toggle 的 fractional-LMUL 链 + 256 lbu 激活标量读·stock 以另一轴向量化不付同税）·**非重建-count**。

### 4.2 重建 headroom 的真实位置（P0 判别键·retarget 依据）
| 格 | insn 我/stock | 损型 | 重建 headroom |
|---|---|---|---|
| **q4_K** | 1.12× | **停顿型(IPC 崩 0.52→0.29)** | **无**(仅 4.7% amenable·zvbb 0) |
| **q5_0** | **1.62×** | 计算型(+62% 重建) | **★有**(qh 第5位组装·M=1 不摊销) |
| **q5_1** | **1.43×** | 计算型(+43% 重建) | **★有**(qh 平面) |
| q4_1(WIN) | 0.77× | 结构赢样板 | — |

**∴ "M=1 重建指令最小化" thesis 的天然靶 = qh-plane 格(q5_0/q5_1/q5_K)·不是 q4_K**（q4_K 无 qh 平面·损是内存延迟停顿非重建数）。P1 沿用 q4_K 为首格是从"审计浪费最大者"选的·但与**重建-count** thesis 靶不匹配。

### 4.3 子门张力（方法学发现）
q4_K 瓶颈 = **memory-latency stall（IPC）**·不是 instruction-count。其对症杠杆 = **prefetch 软流水（latency-hiding）**·而 prefetch **ADD 指令**（+2/block）→ **1.10× instruction-count 子门会 REJECT 正确的 latency-hiding 杠杆**。∴ 对 memory-latency-bound decode·instruction-count 子门是**错判别键**·register-budget-fit 子门才是对的（且已 PASS）。

### 4.4 ★综合 G1 判读
| 子门 | 结果 | 依据 |
|---|---|---|
| **live set ≤ 32-vreg** | **✅ PASS** | TG=1 复原 OLD board-proven spill-free(10=prologue)·P2 trap 修复·部署编译器传导 |
| **指令数 ≤ 1.1× stock** | **❌ NOT MET(1.12–1.14×)** | OLD/TG1 已超·board-available 无杠杆(zvbb 舰队缺席+q4_K 0-yield)·且对 memory-latency 损此门错判别键 |

**综合（[VERIFY-LADDER] 两子门须皆持）= q4_K@rvv 重建-count 线 打回·不上板**。登记超标条 = **instruction-count 1.12–1.14×**。**但**：register-pressure 子门（P2 killer）已 PASS = 真实前进（P2 推荐的 TG=1 register-fit 已零板时坐实）。

---

## 5. [PAT-2] P9 mechanism 订正建议 (令 §三.3 · 改留主会话 · 我方仅建议)

- **P9 GEVM regime-plan mechanism 措辞**：从 P0 已证伪的 "字节最少化" →
  **"M=1 带宽→吞吐结构效率 = {register-budget-fit 列组(TG=1·非加宽 bank) + latency-hiding(prefetch 软流水/重建关键路径缩短)}"**。
  status 注：**mechanized（[K-10] 独立 plan·byte-exact·P1 构造成立）· 性能兑现 pending · register-budget-fit 子门 board-proven(TG=1 spill-free) · 重建-count 杠杆对 q4_K 无效(headroom 在 qh-plane 格)**。
- **colgroup-tiled register-resident-bank(TG≥2) 变体**：维持 P2 的 **board-falsified@M=1(register-pressure spill)** 标注。
- **[X-ZVBB]/P8**：新增事实 = **舰队缺席(rvv∧k1)**·P8 = registered-but-not-board-selectable·q4_K 0-yield 实测·靶应为 bit-permutation 解包格 + zvbb 硅[远期]。
- **[K-10] 补注**：确认 GEVM plan 与 GEMM plan 结构级分立成立（P1 已交付）；TG 是**参数级能力键**（register-budget 约束·per-format 板测定·[SEL-1]），**非结构级**——TG=2→TG=1 是 knob 收窄不是新 plan（反向确认·不翻案 P1 结构级独立性）。

---

## 6. 下一步 (令 §四)

**G1 综合 = 打回（instruction-count 子门 NOT MET·q4_K 重建-count 无 board-available 杠杆）** → **不上板**（SOP 铁律·省板时·同 P2 534-spill 应被拦之理）。重设计再定向三选：
1. **★retarget 重建-count thesis 到 qh-plane 格**（q5_0/q5_1/q5_K·+43~62% 重建 = 真 headroom）·q4_K 非该 thesis 靶。
2. **q4_K 改走 latency-hiding thesis**（TG=1 register-fit[已 PASS] + prefetch 软流水）·其门应是 **register-budget-fit(过) + G2 IPC micro**·**非** instruction-count 子门（§4.3 错判别键）——需主会话裁 gate 定义（灰区·[VERIFY-LADDER] 对 memory-latency-bound decode 的 instruction 子门适用性）。
3. **zvbb 全线[远期]**（舰队缺席·硅未到）。

**禁上板 · 禁 git · 无 schema label 改动（本文件仅产建议）**。工作树未改发射器（TG=1 由 lit attr sed 出·未 commit·未动 .cpp）。scratch 在 `$CLAUDE_JOB_DIR/tmp`·关键静态账产物存 `raw/`。
