# G8 §六.3 攻坚 · iq1_s vec_dot **vector grid-index primitive** (M=1 GEVM decode) @rvv — 真攻坚 (2026-07-15)

> **战果（如实）**：真解剖 + 真构造 + 真测 + 真根因。
> vector grid-index 合成原语 **已建**（byte-exact · **消栈往返** · vset 79→68）→ **未翻正**（cold 0.45→0.47 边际 · HOT 不变/微降）。
> 出 **refined 具名-X（体例四件齐）**：残余 **不是** 标量 index（前序假设 **被本役证伪**），而是
> **每子块 `vmv.x.s`(向量→标量抽取) + 标量 scale-accumulate 串行链**（ours 10–18 vs opp **2**）——
> 对 index 向量化 **与** gather 加宽 **双双不敏感**，2.2× residual **不变**。
>
> **口径**：clang-18 对称域（ours 与 opp 同 clang-18.1.8 + rich-march + `-fno-integrated-as`）· rvv openEuler VLEN128 ·
> core8 pin(perf 2.6GHz) · **本役期间 co-tenant vLLM(core0,1) 缺席**（loadavg 2.1、静默总线）。
> **byte-exact**：DYNAMIC ZERO-MODEL vs stock ggml oracle · baseline/wide/vecidx 各 M=1 & M=8 · 0 mism / 0 ULP。
> **禁 git**（主会审后决定采纳/合并）· 未改 T3/T8 · 发射器改动 **已 revert**、tree 干净（reverted 源 regen == sealed baseline **逐位**）、vecidx 以 casefile patch 保全。

---

## 0. 结果表 (2×2 因子 + opp · 全部本役【同一静默板会话】· clang-18 对称域)

| 核 | vset | vluxei | vwmul | **vmv.x.s** | vwredsum | HOT ratio(nc64) | HOT ours_ns | cold nc64 | cold nc256 | cold nc512 | byte-exact |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| **opp** `iq1_s_q8_K_vl128` | 41 | 4 | 4 | **2** | 8 | 1.0(ref) | 57.5k | — | — | — | — |
| baseline (m2 · **标量-idx**) | 79 | 8 | 8 | **10** | 8 | 0.4525 | 127.4k | 0.455 | 0.456* | 0.431 | 0/0 |
| wide (m4 · 标量-idx · gather 追平) | 45 | 4 | 4 | **10** | 8 | 0.4317 | 133.3k | 0.456 | 0.433 | 0.438 | 0/0 |
| **vecidx (m2 · 向量-idx · 本役)** | **68** | 8 | 8 | **18** | 8 | 0.4409 | 131.1k | **0.470** | 0.447* | 0.456 | **0/0** |

`*` = 单点噪声(o_iqr≥4，flush 抖动)；配对 ratio=ours/opp 冷冷同刷。opp 计量取 `raw/opp_iq1s_vl128.s`。

**读表**：
- **三核 HOT/cold 全落 0.43–0.47 带内**（彼此 ≤5%）。vector-index（本役）**与** gather-widening（wide）**各自把自己那把杠杆推到 opp 平**（wide: vluxei/vwmul/vset 全平 opp 4/4/45；vecidx: 消 tmp[4] 栈、vset 79→68），**却都不翻**。
- **2.2× 是 invariant**：ours HOT ours_ns 127k–133k 恒 ≈ 2.2× opp 57.5k，无论 idx 标量/向量、gather m2/m4。
- **vecidx 反把 `vmv.x.s` 10→18**（见 §c）——向量-index 不仅 null，还微增向量↔标量往返，正对应 HOT 微降(0.4525→0.4409)。

> ★ **对前序 casefile 数值的诚实订正**：前序 iq-gather 记 baseline cold=0.27、wide cold=0.44（“gather-widening 改善 0.27→0.44”）。**本役静默板复测证明那是 co-tenant 内存压力 artifact**：静默总线下 baseline cold=0.455 == wide cold=0.456（无冷罚可消）。前序“wide 改善”本身是压力伪信号，非真加速。

---

## a. 解剖 (逐指令 · clang-18 -O3 · rvv VLEN128)

### a.1 对手 `ggml_vec_dot_iq1_s_q8_K_vl128`（`raw/opp_iq1s_vl128.s`，225 insn）
关键结构 = **per-sub-block scale 延后进向量域**：8 子块的 grid-dot 各 `vwredsum`(8×，与我方同数) → 结果**不抽标量**，而是 `vse32.v`(8×) 落 stack 的 8-part 数组 → `vle32 → vmul.vv(×ls 向量) → vredsum` 一次收尾。**全程只 2 次 `vmv.x.s`**（最终标量）。gather=4(m4)·vwmul=4(i16m8)·vset=41。

### a.2 我方三核（objdump: `raw/{baseline,wide}_iq1_s.objdump.txt`, `raw/vecidx_iq1s.s`）
per-sub-block loop（ib=0..7 全展开），每子块：grid-dot `vwredsum` → **`vmv.x.s`(抽标量)** → **标量 `ls*lsum` + 标量 `sumi +=`**。8 子块 = 8 抽取+8 标量乘加串行；加 delta/收尾共 **10（baseline/wide）/18（vecidx）** `vmv.x.s`。

### a.3 差距构成逐项量化（ours vs opp，本役 objdump 实数）
| 轴 | opp | baseline | wide | vecidx | 本役杠杆是否触及 |
|---|--:|--:|--:|--:|:--:|
| gather (vluxei16) | 4 | 8 | **4** | 8 | wide 追平 → 无效 |
| vwmul | 4 | 8 | **4** | 8 | wide 追平 → 无效 |
| vset churn | 41 | 79 | **45** | **68** | wide/vecidx 均收 → 无效 |
| stack idx 往返 | 0 | 有 | 有 | **0(消除)** | **vecidx 消除 → 无效** |
| **`vmv.x.s` 抽标量** | **2** | 10 | 10 | **18** | **两役均未触 → 残余所在** |
| vwredsum | 8 | 8 | 8 | 8 | 双方同数（非差异源） |

★ **关键**：wide 已把 gather/vwmul/vset **全部**追平 opp（4/4/45），`vmv.x.s` 仍 10 vs 2、仍 0.43 → **gather-parity 配置下残余就是抽取链，不是 index**（这一点封住了前序 casefile“残余=标量 index”的漏洞：它从未在 gather-parity 位真的向量化 index，本役 vecidx(m2) + wide(m4) objdump 双证残余不动）。

---

## b. 施工 (真构造 · 向量 grid-index 合成原语 · byte-exact 优先)

**杠杆 = vector grid-index synthesis（纯发射重排，byte-exact）**。文件 `lib/Conversion/RVV/RVVToEmitCTernaryBinary.cpp`，函数 `emitIQ1SSuperBlockGridBody`（diff 见 `emitter-diff/*.patch`，158 行）。

- 消除 `uint16_t tmp[4]` 栈数组 + 4× 标量 index 链 + `vle16 mf2` 回读。改为 **全向量合成**（复用同库 `RVVToEmitCBlockQuantLinear.cpp` q5 qh-expand 已证原语）：
  - `vle8 u8mf4`(qs+ib*4, 4) → `vzext.vf2 u16mf2`（低 8 index 位）
  - `vmv.v.x u16mf2`(qhw) splat · `vid u16mf2`=[0,1,2,3] · `vmul.vx ×3`=[0,3,6,9]（per-lane 3*l 移位量）
  - `vsrl.vv`(splat, shifts) · `vand.vx 7` · `vsll.vx 8` → 高 3 index 位
  - `vor.vv`(qs, high) = 11-bit idx · `vsll.vx 3` = **byteOff 向量 = 直接喂 gather 的 u16mf2 索引**（零栈）。
- **byte-exact 论证**：逐 lane l 与标量式 `idx=qs[ib*4+l]|(((qhw>>(3*l))&7)<<8)`、`vidx=idx<<3` **同算术、同 lane 序**；gather/归约/scale 施加序不变 → 逐位恒等。
- 发射实证：`raw/vecidx_iq1s.s` 顶部 8 组 `vle8_v_u8mf4 → vzext → vmv.v.x → vid → vmul → vsrl.vv → vand → vsll → vor → vsll` 每子块一套；`tmp[4]`/`vle16_v_u16mf2` **零残留**。核 793→600 行、vset 79→68。

---

## c. G1 (byte-exact + objdump) + G2 (cold ratio)

**G1**：
- **DYNAMIC ZERO-MODEL**（driver `iqfp4_vecdot_driver.c`，ours vs stock ggml oracle，同 bytes）：vecidx iq1_s **M=1 & M=8 · int_byte_mismatch=0 · worst_ulp=0**（`raw/cold_vecidx.log` 每 CENSUS 行 `byte_mismatch=0 worst_ulp=0`；build 时 VERIFY M=1/M=8 双绿）。baseline/wide 同样 0/0（三向恒等 ours==opp==baseline）。
- **objdump 证新核形态**：vset 79→68、消栈、8 套向量 index 合成（`raw/vecidx_iq1s.s`）。
- **tree 干净证**：发射器 **revert** 后 `weft-opt` 重建、regen == sealed baseline **逐位 BYTE-IDENTICAL**。
  - 注：本杠杆改发射结构 → iq1_s CORE/EMIT lit FileCheck（硬校验 index 合成形状）会红；**故未合入**、以 patch 保全；G1 正确性以 DYNAMIC ZERO-MODEL 承载。

**G2**（M=1 GEVM · K=2048 · nc{64,256,512} · reps=15 · 224MiB flush paired · core8 · `raw/cold_{baseline,vecidx}.log`）：
- vecidx cold **0.470/0.447/0.456**（nc 64/256/512）vs baseline **0.455/0.456/0.431**：nc64 +3–4%、nc256/512 持平——**边际、落噪声内**。
- **HOT**：vecidx 0.4409 vs baseline 0.4525（ours_ns 131k vs 127k）——**微降**（`vmv.x.s` 10→18 的代价）。
- **结论：向量-index 不翻正**（cold<0.8 · HOT 未改善）。

---

## 结果 / 认输门槛 — **refined 具名-X（体例四件齐）**

### iq1_s — 未翻正（cold 0.47 < 0.8）· refined 具名-X
1. **对手核符号 + 反汇编**：`ggml_vec_dot_iq1_s_q8_K_vl128`（sealed clang-18，`raw/opp_iq1s_vl128.s`，225 insn，`vmv.x.s`×2）——per-sub-block scale **延后进向量域**（§a.1）。
2. **差距构成逐项量化**（本役 objdump 实数，§a.3）：`vmv.x.s` 2(opp) vs 10(baseline/wide)/18(vecidx)；wide 已把 gather/vwmul/vset 全平 opp(4/4/45) 而 `vmv.x.s` 仍 10、仍 0.43 → residual 与 gather 无关。
3. **我方等价构造实际尝试**：vector grid-index 合成原语（消 tmp[4] 栈 + 标量链 + vle16 回读），**byte-exact 0mism/0ULP**，vset 79→68、核 793→600 行（§b/§c）。**结果 null**（cold 0.45→0.47 边际、HOT 0.45→0.44 微降）。
4. **残余具名因素 X = 每子块 `vmv.x.s`(向量→标量抽取) + 标量 `ls*lsum`/`sumi+=` 串行链**。**可检验 / 已被 objdump 隔离证明**：
   - opp `vmv.x.s`=2、ours=10–18，**跨 index(标量/向量) × gather(m2/m4) 四配置全不动**；2.2× HOT invariant。
   - 两核 `vwredsum`=8 **同数** → 残余 **不是**归约本身，而是**归约后抽标量 + 标量域施加 per-sub-block ls**。opp 反例证：8 part-sum `vse32.v` 落 stack → `vmul.vv(×ls 向量)` → 单 `vredsum`，只碰标量 2 次。
   - **发射器成熟度-X（对手已证可向量化 → 可构造，非硬件墙）**。
   - **前序假设被证伪**：前序 iq-gather 记“残余=标量 grid-index 合成”。本役实建向量 index → 未翻、且 `vmv.x.s` 反增 10→18。**真残余在归约 scale-scheme，不在 index**。

**follow-up（结构级 · 非本役 scope · 真正 fixable 杠杆）**：把 per-sub-block 收尾从「`vwredsum`→`vmv.x.s`→标量 ls 乘加」改为 **向量域延后 scale 归约**——8 子块 part-sum 累进 **向量寄存器**（非标量），8 个 ls 以 **一次 `vmul.vv`** 施加，单次 `vredsum` 收尾，复刻 opp 的 2-`vmv.x.s` 结构。这是**归约方案（reduction-scheme）结构级改动**，与 grid-index 正交。**未建 4th cell（wide+vecidx）**：两 null 杠杆叠加、按 invariant 2.2× 预测 ≈0.45，边际价值低，bounded 停潜；残余已被 wide(m4)+vecidx(m2) objdump **双证**定位，无需该 cell。

---

## 编译器域如实标
本测 = **clang-18 对称域**（ours 与 opp 同 clang-18.1.8 + rich-march + `-fno-integrated-as`；opp = `/tmp/g8s3/rvv/quants_opp.o` §五/§六 定档 sealed）。rvv 板 shipped 编译器 = gcc-15；gcc 部署域未复测（[CASE-COMPILER-ASYMMETRY]）。主表按决令 clang-18 MAIN。

## 触碰文件 · board 卫生 · 禁 git
- **触碰（已 revert，tree 干净）**：`lib/Conversion/RVV/RVVToEmitCTernaryBinary.cpp` — vecidx diff 存 `emitter-diff/RVVToEmitCTernaryBinary.iq1s-vector-index.patch`（主会审决定采纳；采纳需再生 iq1_s CORE/EMIT lit 期望）。reverted 源 regen == sealed baseline 逐位已证。
- **casefile**：本目录（evidence + kernels/iq1_s.vecidx.kernel.c + emitter-diff/ + raw/{opp .s, baseline/wide objdump, vecidx .s, cold_{baseline,vecidx}.log} + harness）。
- **禁 git 遵守**：无 git add/commit。**未改 T3/T8**。
- **board 卫生**：pin core8（gov performance · 2.6GHz）· **co-tenant vLLM(core0,1) 全程缺席**（loadavg 2.1 静默）· 无 stray iq_bench · scratch `/tmp/g8s3iq` 抽取后 **已清** · opponent `/tmp/g8s3/rvv/*.o`（5 个 §五/§六 资产）**保留**。
