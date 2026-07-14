# G8 §六.3 攻坚 · iq1_s / iq1_m vec_dot (M=1 GEVM decode) @rvv — 真攻坚 (2026-07-15)

> **战果（如实）**：两格皆 **真解剖 + 真构造 + 真测**，两格 **未翻正**（cold < 0.8），
> 两格各出 **真具名-X**（体例四件齐，均来自本格解剖+构造+G1/G2，**非** "预期不可达/niche"）。
> - **iq1_s**：gather-widening 杠杆落地 → cold **0.27 → 0.44**（byte-exact，vset 79→45，gather 8→4）。
>   杠杆后 ours 的 vset(45)/gather(4) **已追平对手(41/4)**，却仍慢 ~2.2× → **残余瓶颈被隔离证明 = 标量 grid-index 合成**（index-serial-dep + stack 往返），非 gather 吞吐/vset churn。
> - **iq1_m**：grid-gather-widening 杠杆落地 → cold **0.15 → 0.18**（byte-exact，grid gather 16→8，vset 158→134）。
>   仅 +13% 的微动 **实证隔离** 出 iq1_m 真瓶颈 = 每组 8-element tiny `vwredsum`(sum2/delta) ×32（本役未触），非 grid gather。
> - 两 X 均为 **发射器成熟度-X（对手已证可向量化 → 可构造，非硬件墙）**，附精确 follow-up。
>
> **口径**：clang-18 对称域（ours 与对手同 clang-18.1.8 + rich-march + `-fno-integrated-as`）· rvv openEuler VLEN128 · core8 pin · co-tenant vLLM(core0,1) 全程未扰。
> **byte-exact**：DYNAMIC ZERO-MODEL vs stock ggml oracle · iq1_s+iq1_m 各 M=1 & M=8 · 0 mism / 0 ULP。
> **禁 git**（主会审后 commit）· 未改 T3_A/T8 · 发射器改动【已 revert】、tree 干净（regen 复现 committed 逐位）、杠杆以 casefile patch 保全供主会审采纳。

---

## 0. 结果表 (headline · cold primary · clang-18 对称域 · vs sealed opp vl128 hand-tuned)

| 格 | 核 | vset | gather | HOT | cold nc64 | cold nc256 | cold nc512 | 0.8门 | byte-exact |
|---|---|--:|--:|--:|--:|--:|--:|:--:|:--:|
| **iq1_s** | baseline | 79 | 8 (i64m2) | 0.43 | 0.290 | 0.283 | 0.271 | FAIL | 0mism/0ULP |
| **iq1_s** | **attack (wide m4)** | **45** | **4 (i64m4)** | 0.43 | **0.462** | **0.440** | **0.435** | **FAIL(↑)** | 0mism/0ULP |
| **iq1_m** | baseline | 158 | 16 (i64m1) | 0.16 | 0.165 | 0.155 | 0.153 | FAIL | 0mism/0ULP |
| **iq1_m** | **attack (wide m2)** | **134** | **8 (i64m2)** | 0.18 | **0.189** | **0.175** | **0.172** | **FAIL(↑)** | 0mism/0ULP |

对手（sealed clang-18）：`ggml_vec_dot_iq1_s_q8_K_vl128` tot=225 gather=4(+1 vrgather) vset=41 · `ggml_vec_dot_iq1_m_q8_K_vl128` tot=253 gather=2 vset=17。

---

## a. 逐指令解剖 (objdump 两侧 · clang-18 -O3 · rvv VLEN128)

### a.1 对手 iq1_s vl128（`raw/opp_iq1s_vl128.s`）— 全向量化 index 合成 + 宽 m4 gather
对手一个 super-block loop body（`.LBB29_4`，`bne` 回卷）做：
1. `vle16 v28,(qh)` 一次载 8 个 qh 字 → `vsrl.vi 12 / vand.vi 7 / vwmulu.vx 2 / vadd.vi 1` = **8 子块 scale-delta 一次向量化**（2·idx+1）；`vand.vx 0x8000 / vmseq / vmerge` = **8 子块 sign ±1 一次向量化**。
2. `vle8 v26,(qs)` 载 qs 低索引；`vrgather.vv v0,v28,v16`（`v16 = vid/vdiv` 展开 [0,0,0,0,1,1,..]）**把每子块 qh 播散到其 4 位置**；`vsrl.vv / vand.vi 7 / vsll.vi 8` 取 grid 索引高 3 bit；`vzext.vf2 qs / vor.vv` = **拼 11-bit grid 索引【全在向量寄存器】**；`vsll.vi 3` → byte offset。
3. `vsetivli 8,e64,m4 ; vluxei16.v v0,(grid),v28` = **e64/m4 宽 gather（每次 64 weight = 2 子块）**，共 **4 gather** 覆盖 8 子块；每 gather 后 2 个窄 `vwredsum`（保各子块 scale），8 部分和暂存栈。
4. 尾段 `vle32 8 部分和 → vmul.vv(×scale 向量) → vredsum` = **per-sub-block scale 延后、跨 8 子块向量化施加**。
- 计量：gather=4(m4)·vset=41·vwredsum=8·vwmul=4·vrgather=1·vid=1。

### a.2 我方 iq1_s baseline（`raw/baseline_iq1_s.objdump.txt`）— 标量 index + 栈往返 + 窄 m2 gather
super-block loop（`for v8`）内 **8 子块全展开**，每子块：
- `ternary_grid_index` = **标量链**（每位置 `qs | ((qh>>3l & 7)<<8) <<3`，4 位置 ~28 标量 ALU）→ 写栈 `uint16_t tmp[4]` → `vle16 mf2` 回读 → `vluxei16 i64m2`（**窄 gather：4×i64 = 32 weight**）→ `vwmul i16m4(32)` → `vwredsum(32)`。
- 计量：gather=**8(m2)**·vset=**79**·（clang 把标量 index 半向量化但用 `vlse16/vse16/vslide1down` 笨拙栈往返 + 大量 SEW 重配）。
- **差距构成逐项量化 vs 对手**：① gather 宽度 = 对手 1/2（m2 vs m4）→ gather 数 8 vs 4；② vset churn 79 vs 41（子块间 SEW 反复重配 e8/e16/e64）；③ **index 合成走标量+栈往返（对手全向量、零栈）** = serial-dep 主项。

### a.3 我方 iq1_m baseline（`raw/baseline_iq1_m.objdump.txt`）— 双病灶
每子块含两条路径：① **每组 8-element tiny reduction**（`vsetvl_e8m1(8)` + `vle8 i8m1(8)` + `vwredsum_i8m1_i16m1` + `vmv.x.s` + 标量累加）= `sum2_delta`，4 组 × 8 子块 = **32 次微归约**（各带 vsetvl+reduce+extract）；② per-half grid gather `vluxei16 i64m1`（2×i64 = 16 weight），2 half × 8 = **16 gather**。
- 计量：gather=**16(m1)**·vset=**158**·`vwredsum`=48（32 tiny i8 + 16 grid i16）·15× `vs1r/vl1r` 溢出。
- 对手 vset=17 / gather=2 → **9.3× vset churn** 是 0.15× 的形状根因；主项 = 32 次 tiny 归约（对手 bsums/向量批量，我方逐组微归约）。

---

## b. 施工 (真构造 · 发射器杠杆 · byte-exact 优先)

**杠杆 = gather-widening（纯发射重排，byte-exact）**。文件 `lib/Conversion/RVV/RVVToEmitCTernaryBinary.cpp`（diff 见 `emitter-diff/*.patch`，564 行）。

- **iq1_s**（`emitIQ1SSuperBlockGridBody`）：`for ib` 单子块 → **`for ib+=2` 成对**。每对建 8 索引 → **ONE `vluxei16 i64m4`（64 weight = 2 子块）** → reinterpret i8m4 → `vle8 i8m4(64)` → **ONE `vwmul i16m8(64)`** → `vget_v_i16m8_i16m4` 取两半 → **2 窄 `vwredsum`（各持自身 ls 标度）**。gather/wmul/vle8 各 8→4。
- **iq1_m**（`emitIQ1MSuperBlockGridBody` 的 `for h` grid 路径）：per-half `i64m1` gather → **整子块 ONE `vluxei16 i64m2`(32 weight)** + ONE `vwmul i16m4(32)` + `vget_v_i16m4_i16m2` 两半 → 2 窄 `vwredsum` → `sum1[0]`(组{0,1},ls1)/`sum1[1]`(组{2,3},ls2)。grid gather 16→8。**tiny sum2 路径【故意不触】**（以隔离病灶）。
- **byte-exact 论证**：每子块/每半的 32/16 lane dot 仍在相同 grid×q8 上、相同整数归约内单独求和，per-sub-block ls/delta/sign 施加序不变；gather 只把 2 子块/整子块的独立权字节合批。整数结合律 + 归约不跨标度 → 逐位恒等。

---

## c. G1 (byte-exact + objdump) + G2 (cold ratio)

**G1**：
- **DYNAMIC ZERO-MODEL**（harness `iqfp4_vecdot_driver.c`，ours-wide vs stock ggml oracle，同 bytes）：iq1_s + iq1_m 各 **M=1 & M=8 · int_byte_mismatch=0 · worst_ulp=0**（`raw/cold_attack.log` 顶部 VERIFY，及三 nc 每行 `byte_mismatch=0`）。三向恒等：ours_wide == opp(stock) == ours_baseline。
- **objdump 证新核形态**：iq1_s vset 79→45 / gather 8→4（`raw/attack_iq1_s.objdump.txt`）；iq1_m vset 158→134 / grid-gather 16→8（`raw/attack_iq1_m.objdump.txt`）。
- **tree 干净证**：发射器 revert 后 regen 复现 committed 逐位（iq1_s md5 f75372be…、iq1_m md5 fd3d30a8… 均 MATCH）。
  - 注：本杠杆改变发射结构 → 4 个 lit fixture 的 CORE/EMIT FileCheck 硬校验 gather 宽度会红（需 expectation 再生）；**故未合入**、以 patch 保全；G1 正确性以 DYNAMIC ZERO-MODEL 承载（比 FileCheck 强）。

**G2**（M=1 GEVM · K=2048 · nc{64,256,512} · reps=15 · 224MiB flush paired · core8 pin · `raw/cold_baseline.log` + `raw/cold_attack.log`）：
- iq1_s：cold **0.271–0.290 → 0.435–0.462**（~1.6× 提升，稳定 o_iqr≤1.3 除个别 flush 抖动）。HOT 0.43 不变 → cold≈HOT，冷罚已被 vset/gather 收缩吃掉，残余=纯计算/index。
- iq1_m：cold **0.153–0.165 → 0.172–0.189**（~+13%）。HOT 0.158→0.181。

---

## 结果 / 认输门槛 — 两格 **真具名-X**（体例四件齐）

### iq1_s — 未翻正（cold 0.44 < 0.8）· 具名-X
1. **对手核符号+反汇编**：`ggml_vec_dot_iq1_s_q8_K_vl128`（sealed clang-18，`raw/opp_iq1s_vl128.s`）— 全向量化 index 合成 + e64/m4 宽 gather（§a.1）。
2. **差距构成逐项量化**：gather 宽度 m2 vs m4（8 vs 4）· vset 79 vs 41 · index 合成 标量+栈往返 vs 全向量（§a.2）。
3. **我方等价构造实际尝试**：gather-widening 杠杆（m2→m4 成对，vget 两半窄归约），byte-exact 0mism/0ULP，**cold 0.27→0.44 · vset 79→45 · gather 8→4**（§b、§c）。
4. **残余具名因素 X = 标量 grid-index 合成（index-serial-dep + stack 往返）**。**可检验 / 已被隔离证明**：杠杆后 ours 的 vset(45)/gather(4) **已追平对手(41/4)**，但仍慢 ~2.2× → 残余**证明性地不再是** gather 宽度或 vset churn，而是 `ternary_grid_index` 的标量链 + `tmp[8]`→`vle16` 栈往返（对手以 `vrgather/vid/vzext/vor` 全向量合成、零栈）。**这是发射器成熟度-X（对手已证可向量化 → 可构造，非硬件 gather 吞吐上限）**。
   - **follow-up（结构级 · 非本役 scope）**：把标量 index 合成升为 **向量 grid-index 合成原语**（vle qs/qh 为向量 → 向量 shift/and/or/vzext/vrgather 直接产索引向量，消栈往返与 SEW churn）。

### iq1_m — 未翻正（cold 0.18 < 0.8）· 具名-X
1. **对手核符号+反汇编**：`ggml_vec_dot_iq1_m_q8_K_vl128`（`raw/opp_iq1m_vl128.s`）tot=253 gather=2 vset=17。
2. **差距构成逐项量化**：vset 158 vs 17（9.3× churn）· grid gather 16 vs 2 · **32× 8-element tiny `vwredsum`(sum2/delta) 各带 vsetvl+reduce+extract** · 15× 寄存器溢出（§a.3）。
3. **我方等价构造实际尝试**：grid-gather-widening 杠杆（i64m1→i64m2 整子块，vget 两半），byte-exact 0mism/0ULP，**cold 0.15→0.18 · grid gather 16→8 · vset 158→134**（§b、§c）。
4. **残余具名因素 X = 每组 8-element tiny-reduction 病灶（`sum2_delta_accumulate` ×32）+ 标量 index 合成**。**可检验 / 已被隔离证明**：grid gather 减半（16→8）**仅** +13%（0.15→0.18）→ **实证** iq1_m 瓶颈**不在** grid gather，而在本役未触的 32 次微归约（+ 同 iq1_s 的标量 index）。**发射器成熟度-X**：tiny 归约可批量化（每 half 用 per-group ±ls 符号向量做一次 sign-weighted `vwmul+vwredsum` 替 2 次 8-lane 微归约），非硬件墙。
   - **follow-up**：sum2 per-group 符号和向量化批处理 + 向量 index 合成（与 iq1_s 同源）。

**综判**：两格皆非硬件天花板；残余 = 精确命名的**发射器成熟度**（向量 index 合成 / tiny 归约批处理），对手同板 3.5×–6.6×/GMAC 已证可快。翻正需上述 follow-up（结构级发射器原语），本役 gather-widening 是**真实但不足**的一步（每格净 byte-exact 提升 + 病灶隔离）。

---

## 编译器域如实标
本测 = **clang-18 对称域**（ours 与对手同 clang-18.1.8 + rich-march + `-fno-integrated-as`；opponent = `/tmp/g8s3/rvv/quants_opp.o` §五/§六 定档 sealed）。rvv 板 shipped 编译器 = gcc-15；gcc 部署域未复测（[CASE-COMPILER-ASYMMETRY]）。主表按决令 clang-18 MAIN。

## 触碰文件 · board 卫生 · 禁 git
- **触碰（已 revert，tree 干净）**：`lib/Conversion/RVV/RVVToEmitCTernaryBinary.cpp` — 杠杆 diff 存 `emitter-diff/RVVToEmitCTernaryBinary.iq-gather-widen.patch`（主会审决定是否采纳；采纳需再生 4 个 iq1 lit fixture 的 CORE/EMIT 期望）。regen 复现 committed 逐位已证。
- **casefile**：本目录（evidence + kernels/{baseline,wide} + emitter-diff/ + raw/{opp .s, baseline/attack objdump, cold logs, build seal} + harness）。
- **禁 git 遵守**：无 git add/commit。**未改 T3_A/T8**。
- **board 卫生**：pin core8（gov performance · freq 2.6GHz）· co-tenant vLLM core0,1 全程未扰（1 python3 存活）· loadavg 2.1→2.6（co-tenant drift·非我方）· 无 stray iq_bench · scratch `/tmp/g8s3iq` 抽取后清 · opponent `/tmp/g8s3/rvv/*.o`（§五/§六 资产）保留。
