# P1-remainder — 4 格 VOID 重试 · 实测结果（按【既有】PREREG 判读 · 判据未改）

> 预注册：`P1-backfill7-prereg.md`（测前写死·本轮**沿用·零修改**）。前轮结果：`P1-backfill7-results.md`。
> raw 工件：`P1-remainder-raw/`。**禁 git commit**（主会话统一入库）。**T3/recon 未改**（主会话机算入表）。
> 单世界 clang-18。**零 gcc 输出**（`--gcc-install-dir` = clang CRT 定位 flag·prereg 钦定 recipe）。
> 硬冻结（102/105 · 101/108 · 9/83 · roster $meta · 队序）**未碰 = 只登记不执行**。

## 净结论

**4 格重试：具名-X 1 / VOID-NOISE 3**（pending 21 → **20**，非预期的 17）。
**0 verified hand-brick win**（本轮唯一测到的格是 X·输给同算子手调对手 4.56×）。

| # | 板 | op/format/regime | 前轮 | **本轮 verdict[0.8]** | cold ratio |
|---|---|---|---|---|---|
| 5 | rvv | gemm_tile/iq4_nl prefill | VOID（harness 无 oracle） | **VOID-NOISE**（★成因已迁移） | 未测·0 样本 |
| 7 | k1 | gemm_tile/iq4_nl prefill | VOID（同上） | **具名-X[0.8]** | OPP-X 0.6025/0.6034 · OPP-S 0.2192/0.2191 |
| 4 | rvv | gemm_tile/iq4_nl decode | VOID-NOISE | **VOID-NOISE**（2/2 再不过） | 未测·0 样本 |
| 1 | rvv | dequantize_row/iq2_xs | VOID-NOISE | **VOID-NOISE**（2/2 再不过） | 未测·0 样本 |

**#5 的成因迁移（登记·非清格）**：上阶段 ZERO-MODEL oracle 已建成 → prereg §4.0 正确门由「结构上不可满足」变为**双板实测 PASS**（rvv `ours 0/8192`）。**#5 现在死于噪声门，不再死于 harness**。格仍 VOID·pending 不动·**禁记作进展分**。

---

## §5 噪声自检全量（**先跑·门不过不开测**·6 次尝试·零摘录）

**开测条件**（prereg §5·未改）：GEMM/GEVM `ours relIQR ≤5% ∧ opp ≤8%`；dequant `ours ≤3.8%(iq2_xs) ∧ opp ≤3%`；共用 `3 轮 ratio 中位两两相对极差 ≤2%`。**合取**。两次尝试仍不过 → VOID-NOISE。

| 格 | att | 核 | ours relIQR ×3 | oppX relIQR ×3 | ratio 中位 ×3 | 极差 | 门 |
|---|---|---|---|---|---|---|---|
| **#5** rvv prefill | 1 | c8 | 1.11 / **5.19** / 1.04 | 0.85 / 0.91 / 1.02 | .7611/.7517/.7611 | 1.25% | **FAIL**（ours 5.19>5） |
| **#5** rvv prefill | 2 | c14 | 3.93 / **6.81** / **5.40** | 0.92 / 0.98 / 0.77 | .7562/.7224/.7598 | **5.18%** | **FAIL**（ours ×2 + 极差） |
| **#4** rvv decode | 1 | c8 | **9.17** / **10.98** / 0.37 | 0.27 / 0.55 / 0.18 | .2459/.2375/.2363 | **4.06%** | **FAIL** |
| **#4** rvv decode | 2 | c12 | **6.45** / 2.00 / **6.02** | 0.22 / 0.11 / 0.14 | .2482/.2379/.2478 | **4.33%** | **FAIL** |
| **#1** rvv iq2_xs | 1 | c8 | 1.10 / 1.43 / 1.46 ✓ | opp **4.12** / 1.52 / 1.85 | 1.7182/1.7753/1.7698 | **3.32%** | **FAIL**（opp>3 + 极差） |
| **#1** rvv iq2_xs | 2 | c12 | 1.24 / 1.56 / 1.07 ✓ | opp **4.86** / **3.53** / 1.89 | 1.7619/1.7767/1.7664 | 0.84% | **FAIL**（opp ×2 >3） |
| **#7** k1 prefill | 1 | c4 | 0.24 / 0.19 / 0.24 ✓ | 0.27 / 0.26 / 0.13 ✓ | .6031/.6024/.6024 | **0.12%** ✓ | **PASS → 开测** |

**换核 = prereg §5 明文救济**（"先 load-gate 换核 / 查 co-tenant / kill 竞争进程"）；两次尝试用不同核集（8-15 / 12-15），load-gate 实测 idle=100% gov=performance 全程。
**#4/#1 跨会话累计各 4 次尝试全 FAIL** → 无「测到过就收」的多重比较风险（**若**某次晚到的尝试转 PASS，须登记多重性；本轮未发生）。
**测中门**（`任一格 relIQR ≥ 3× 测前 sanity 值 → 整会话作废`）：#7 measure ours 0.24/0.15 · oppX 0.10/0.49 · oppS 0.24/0.22，对 sanity **中位** 0.24/0.26/0.42 的 3× 阈（0.72/0.78/1.26）→ **全过**。
★**登记（口径空缺·非事后找补）**：§5 测中门写「该格测前 sanity **实测值**」单数，而 sanity 有 3 轮。本轮取**中位**（中立读法；取 max=宽纵、取 min=严苛且等同挑轮次）。**若**按最严的 min-of-rounds 读法，oppX seed2 0.49% ≥ 3×0.13=0.39 会名义触发整会话作废。**读法归属 = 口径空缺 → 登记待裁**（§6.4 空缺条款：只问空缺、不问整体去向）。

---

## #7 `gemm_tile/iq4_nl@k1` prefill = **具名-X[0.8]**（deployed · CROSSOP + 同算子双对手）

cold **OPP-X 0.6025**（s1 0x1357）/ **0.6034**（s2 0xACE2）·bootstrap CI（10k·paired）**[0.6017,0.6029]** / **[0.6025,0.6048]**；
**OPP-S 0.2192 / 0.2191**·CI **[0.2189,0.2195]** / **[0.2190,0.2194]**·N=20·2-seed·relIQR ours 0.24/0.15% · oppX 0.10/0.49% · oppS 0.24/0.22%。
**两对手两 seed 全 <0.8 → 具名-X**（判据 §4.0：任一 seed <0.8 → 走完整环 → 具名-X；此处 4/4 均 <0.8·无 seed-split）。

**ZERO-MODEL（三方·per-(r,c)·8192 输出全查·非抽样）**：`ours 0/8192 · OPP-X 0/8192 · OPP-S 0/8192 = ALL-OK`·worst_tolratio 7.92e-03（**150× 余量·门非空心**）→ **正确核（输的是性能不是正确性）**。
`xcheck: ours==OPP-X 位全等 8192/8192 · ours==OPP-S 位全等 8192/8192`。
**OPP-S 布局门 = PASS**（§3.3：`make_x16`/`make_q8x4` == ggml `block_iq4_nlx16`/`block_q8_0x4`·功能级证明）→ **同算子对手可用 = 成色权威**。
**regime**：nr=16 GEMM（K=2048·nc=512·nr%4==0·nc%16==0）·**未继承** decode 数 / 历史锚。

### §4.4 完整环

**1. 对手身份三证（as-shipped·非稻草人）**
- **OPP-X** `ggml_vec_dot_iq4_nl_q8_0` @`0xa7b44` = **9-ins VLEN 派发 thunk**（rvv=0）→ callee 机判 `_vl128`/`_vl256`；k1(VLEN256) 实走 **`_vl256`** @`0xa7c78` `ins=79 rvv=30 gather=4 vset=6`（**DUAL-AGREE**）·源归属 **`arch/riscv/quants.c:5534`**（riscv 专化·`NOINLINE`·`vrgather.vv`×4 + `vwmacc.vv`/`vwredsum.vs`）→ **手调 VLEN256 专化 codebook 核**。
- **OPP-S** `ggml_gemm_iq4_nl_16x1_q8_0` @`0xac82e` `ins=235 rvv=65 gather=2 vset=12`（**DUAL-AGREE**）·源归属 **`arch/riscv/repack.cpp:1342`**（k1 树；rvv 树同符号 @`:1252`）= **真 riscv 手写 RVV intrinsics**（`__riscv_vle8_v_i8mf2` / `vwmul` / `vrgather`）·对照 `_generic` @`0x35df8` `ins=521 rvv=118` = **另一符号·未链**→ **手调**（**订正前轮 prereg §2 的「疑弱/近-generic」·该疑点源于 E1 坏探针**）。
- stock `.so` md5 before==after `871169a0`(cpu) / `00267134`(base) → 未改库。

**2. 我方核自探针**（front-door leaf md5 `2cc64970`·k1 build）
`ins=1784 rvv=1458 gather=64 vset=656 size=9768B` · **true_sp_spill=16**（对手 OPP-S 反而 47）→ **非 spill 病**（排除 `regfile-spill` 墙）。
top-mnemonics：`vsetvli:655 vwmul.vx:256 vwadd.wv:256 lbu:128 vzext.vf2:64 vluxei16.v:64`。

**3. ★墙（具名·机制级·两条合取·结构事实非计数 artifact）**

> **① `narrow-vl`（本格主墙·board-specific·[K-5b] 机核确证）**
> **AVL 设置指令直证**（只数真正设 vl 的 vset·`zero,zero` 仅换 SEW/LMUL 不改 vl）：
> - **我方** = `vsetivli zero,8,e32,m2` ×1 → **vl=8**（全核唯一 AVL·其余 655 条 `zero,zero`）
> - **OPP-S** = `vsetivli zero,16,e32,m2` → **vl=16**
> - **OPP-X** = `vsetivli zero,16,e{8,16,32}` → **vl=16**
>
> **机制**：`e32m2` VLMAX = VLEN/32×2 → **VLEN128 = 8**（我方 vl=8 = 满宽）/ **VLEN256 = 16**（我方 vl=8 = **半宽·浪费一半 lane**）。我方 leaf 把 16 列拆成 `{0|8}` **两个 8 宽半程**（前轮字节指纹已记 "vl=8 双半" vs ggml "vl=16"），**且 leaf VLEN-invariant**（双板同一 md5 `2cc64970`·vl 常量恒 8）→ **VLEN256 上永不满宽**。
> ★ 这是 memory `Win-K1-VLEN`（k1 vl=16 满宽 1.085× 双轴 win）**同一能力键的反面**：VLEN-adaptivity 杠杆**在本栈已证存在**，但**未键控到 iq4_nl GEMM 这条 lowering** → **C3′ 能力键控覆盖缺口的正例**。
>
> **② `codebook-gather-bound`（与 #6 同墙·跨 regime 复现）**
> 我方 **64 × `vluxei16.v`** = 走**内存** codebook 数组的 indexed gather；对手 = **`vrgather.vv`**（16 项 codebook **常驻向量寄存器** 的寄存器置换）。16 项 codebook 平凡装入单个向量寄存器 → `vrgather` 是正确原语。**判别键 = codebook 尺寸**（memory `[emitter-maturity-vluxei16-widelmul]`：vluxei16 对 iq1_s **大 grid 是 win** 7.4→2.3×，对 16 项 **tiny codebook 是 loss**）——**同一原语两向·C3′ 模式库正例**。
>
> **③ 加重项（非独立墙）**：**vsetvli storm 655 vs 12**（SEW/LMUL 在 `e8mf2 ↔ e16m1 ↔ e32m2` 间反复切换 647 次）= 共享 emitter 骨架税（memory `skeleton-closure-campaign` 同病）。
>
> ★**诚实边界（禁外推·禁定量归因）**：三条机制**均有指令级结构证据**，但**本轮未做消融** → **不定量分摊** 4.56×。静态 ins 计数（1784 vs 235）**仅作描述**、**不作判据**（batch8 订正：ins 绝对值 = 方法学 artifact）。

**4. 可修性 = 可修（emitter 成熟度缺口·入队列·两项）**
- **[可修-1] VLEN-keyed vl 宽度**：iq4_nl GEMM lowering 补 VLEN 能力键（VLEN256 → vl=16 单程，替 vl=8 双半）。**先例已存**（Win-K1-VLEN）→ 是**键控覆盖**问题，非新机制。
- **[可修-2] codebook-size 键控 gather 原语选择**：`≤VLMAX` tiny codebook → 寄存器 `vrgather`；大 grid 才 `vluxei16`。**与 #6 同一队列项**（本轮 = 该项在 prefill/GEMM 上的**独立复现**·非孤例）。

**5. 部署成色**：**deployed**（prereg §0/§4.3：#7 = prefill 摊销 → `selectRepack` = 部署真路）。
★**成色铁线**：OPP-X 0.60 = **跨算子对拼（CROSSOP·我方 repack-GEMM x16 交织一次跨 nc 列 vs 对手 per-column block-dot nrc=1）·非同算子硬赢**；且**本格是 X 不是 PASS**，无「硬赢」议题。**同算子权威 = OPP-S 0.219 → 我方慢 4.56×**（对手三条同时满足：真手调 ∧ 布局门过 ∧ 数值门过）→ **本格无任何硬赢资格·0 verified hand-brick**。
**账** = matmul kernel-sym · **NOT e2e · NOT perf-covered 9/83 · [NG-4]**。**禁外推**（本格 X 不推 rvv/不推 decode/不推同族）。
provenance：`P1-remainder-raw/k1_gemm_prefill_measure.log` · `k1_gemm_prefill_sanity.log` · `k1_gemm_prefill_seal.txt` · `k1_gemm_prefill_ci_results.txt` · `k1_gemm_prefill_probe.txt` · `k1_gemm_prefill_verify.log`（反空心注入矩阵）。

---

## #5 `gemm_tile/iq4_nl@rvv` prefill = **VOID-NOISE** · 未测 · 0 样本 · 不造数

**精确缺口**：prereg §5 GEMM 门 = `ours relIQR ≤5% ∧ 3 轮 ratio 中位极差 ≤2%`；att1 ours **5.19%**（>5）· att2 ours **6.81/5.40%** + ratio 极差 **5.18%**（>2）→ **2/2 不过 → 不开测**。
**已就绪（★上阶段新建·前轮缺口已闭合）**：新 driver `iq4nl_gemm_prefill_p1r.cpp`（md5 `9b5fe4bd`）单一真源 W/A + per-(r,c) oracle + 三方门 → **rvv 实测 `ours 0/8192 · OPP-X 0/8192`·worst_tolratio 7.13e-03**（正确核）；leaf md5 `2cc64970` ABI 7 参对齐；对手三证齐（`_vl128` @`0xc1670` `ins=76 rvv=29 gather=2 vset=11` DUAL-AGREE·手调 VLEN128 专化）；反空心注入矩阵 [B]/[C]/[D] 三档双板全红（[D] = leaf **单字节** 码本翻转 `cmp -l` 证 `differing_bytes=1` → 门能抓被测编译核里的真缺陷）。
**#5 的 OPP-S = VOID-S**（§3.3·见下「张力 3」：`ggml_gemm_iq4_nl_16x1_q8_0` 在 VLEN128 数值错 `8191/8192`）→ **只报 OPP-X + 标 CROSSOP**；**#5 无同算子对手 ⇒ 依 §3.2 成色条款，#5 无论数字多好都无硬赢资格**。
★**登记（不作结论·不填表·依本轮自己的门该数不可报）**：6 轮 OPP-X ratio 全落 **0.7224–0.7611**（无一 ≥0.8）。
**剩余构造**：稳定化 rvv 侧 cold（见下「★噪声归因订正」）→ 再按同 prereg 重测。T3 该格 **维持 pending·禁填**。

## #4 `gemm_tile/iq4_nl@rvv` decode = **VOID-NOISE** · 未测 · 0 样本 · 不造数

**精确缺口**：att1 ours **9.17/10.98%** + ratio 极差 **4.06%**；att2 ours **6.45/6.02%** + 极差 **4.33%** → **2/2 再不过**（跨会话累计 **4/4**）。
**warmup-discard 未采用**：= 改计时协议 = **改判据 → 禁**（任务书明文）。本轮只如实报「未稳定下来」。
**已就绪**：leaf `a911818c`·ABI `(n,s,vx,vy,nc)` 逐参对齐 ✓·**ZERO-MODEL ours 0/512 + OPP-X 0/512**（正确核）·对手三证齐。**OPP-S = VOID-S**（VLEN128 数值错 511-512/512·`ggml_gemv_…` 与 `ggml_gemm_…` 同根）。
★**登记（不作结论）**：6 轮 OPP-X ratio 全落 **0.2363–0.2482**（无一近 0.8）。
**剩余构造**：同 #5。T3 该格 **维持 pending·禁填**。

## #1 `dequantize_row/iq2_xs@rvv` = **VOID-NOISE** · 未测 · 0 样本 · 不造数

**精确缺口**：门 = `opp relIQR ≤3%`；att1 opp **4.12%**（+ ratio 极差 3.32%>2）· att2 opp **4.86/3.53%** → **2/2 再不过**（跨会话累计 **4/4**）。ours 门全程过（1.07–1.56%）。
**已就绪**：build/link/**ZERO-MODEL byte_mismatch=0 / 1048576 · worst_ulp=0** 全绿·对手三证齐（@`0x8101a` `ins=173 rvv=2 gather=0 vset=5`·DUAL-AGREE·源归属 `ggml-quants.c:2444` generic-C = **标量类·便宜档**）。
★**登记张力（不自决·不推翻门·与前轮一致）**：本轮 6 轮 ratio 中位 = 1.7182/1.7753/1.7698/1.7619/1.7767/1.7664（极差 3.32%——**注：不复现前轮的 0.38% "异常稳定"**）。门是**合取**，拿 ratio 稳定豁免 opp-IQR = **事后找补** → **维持 VOID-NOISE**。
**剩余构造**：opp 侧 cold 长尾稳定化（该格 opp 长尾 cold-outlier **跨会话可复现** = 本格真实属性·非本轮环境偶发）。T3 该格 **维持 pending·禁填**。

---

## ★噪声归因订正（登记·**前轮结论部分证伪**·独立于 verdict）

**前轮登记**：「同进程交替下对手 relIQR 恒 0.11–0.38% ⟹ **波动是我方 GEVM leaf 内生**」。
**本轮新证据（三方交替·同进程·同协议）**：

| 板/格 | ours | OPP-X（plain block-dot） | OPP-S（**as-shipped** repack 核） |
|---|---|---|---|
| rvv GEMM prefill（6 轮） | 1.04–**6.81%** | **0.77–1.02%** | **3.05–16.05%** |
| rvv GEVM decode（6 轮） | 0.37–**10.98%** | **0.11–0.55%** | 0.67–2.53% |
| **k1** GEMM prefill（5 轮·**同一 leaf md5 `2cc64970`**） | **0.15–0.24%** | 0.10–0.49% | 0.16–0.73% |

→ **两条与「我方 leaf 内生」相悖的事实**：① rvv prefill 上**对手自己的 as-shipped OPP-S 也不稳（16.05%）** → 不稳非我方代码专属；② **同一份 leaf 在 k1 上稳如磐石（0.15–0.24%）** → 不稳非 leaf 内生属性。
→ **精确化（★假说·未证·禁当结论）**：候选判别键 = **rvv 板 × repack-packed 权重流式访问**（ours 与 OPP-S 都流 x16 packed 权重；OPP-X 流 plain 逐列）。**#5/#4 剩余构造 = 先证伪/证实该假说再重测**，而非盲目加 rep。
**caveat**：rvv 的 OPP-S 数字本身 = **VOID-S**（VLEN128 数值错 → 只写 lane 0-7 = 做的功不同）→ 此处**仅作机器噪声见证**、**禁作性能数**；k1↔rvv 的 OPP-S 耗时**不可比**。

---

## 张力登记（canon 级 · **必问 · 本 agent 未动**）

1. **§5 测中门读法空缺**（新·见上）：「该格测前 sanity 实测值」在 3 轮 sanity 下未定义取哪个。本轮取**中位**（中立读法）→ #7 过。最严 min-of-rounds 读法下 oppX seed2 0.49% 名义触发整会话作废。**读法归属 = 口径空缺 → 待裁**。
2. **`ggml_gem{v,m}_iq4_nl_16x1_q8_0` 双双在 VLEN128 数值错**（前轮 E4 + 本轮 GEMM 兄弟独立确证）：as-shipped 上游核携带**未声明的 VLEN≥256 前提**（源码传字面 avl=16 给 `mf2`/`m2` intrinsics·VLEN128 下 VLMAX=8<16 → vl 钳到 8 → ①`vrgather` index≥VLMAX 返 0 = lane 0-7 算错 ②`vse32(...,16)` 只存 8 个 = lane 8-15 从未写）。**现为 GEVM+GEMM 两个兄弟·非孤例**。是否上游报 / 是否影响 rvv 板 repack 派发结论 → **沿用前轮张力 3·证据本轮加强·待裁**。
   ★ **旧 harness 会把这个坏核照常计时并报出 ratio** —— 无 oracle 的代价的实证。
3. **T3 `gemm_tile/iq4_nl` 行 tier 存疑（87/88 行）**：本轮**再加强**——本格（#7）**同算子** OPP-S `ggml_gemm_iq4_nl_16x1_q8_0` 修正探针 = `ins=235 rvv=65 gather=2`（prereg §2 因 E1 坏探针记 rvv=12 并据此疑「近-generic」）+ 源归属 `arch/riscv/repack.cpp:1342/1252` **手写 intrinsics** → **手调定档站得住**；且 OPP-X `_vl256` 源归属 `arch/riscv/quants.c:5534` 亦**手调**。→ **87/88 行 tier 疑应为「手调」而非「通用向量」**。**登记待裁**（改 tier 触碰 roster/口径·不自决）。
4. **[DEQ-AXIS] auto-promote 字面张力**（前轮登记·本轮 #1 未测故无新证据）：维持待裁·**未动 102/105 分母**。

## 卫生 checklist
- [x] stock/clang18 `.so` **只读**·md5 before==after 双证（rvv `e85fceda`/`d9c07980` · k1 `871169a0`/`00267134`·每次 run 逐条入 log）
- [x] 单实例·测前 `pkill -x` 清竞争·**PRE_STRAY=0 / STRAY=0 全 7 次 run**·`taskset` 单核钉
- [x] **rvv 未碰 core 0,1**（co-tenant vLLM）——实用 8/12/14；k1 用 4/6
- [x] **rvv 先跑完（#5→#4→#1·6 次）再跑 k1（#7·2 次）**·全程无并发·无 `.so` 竞争
- [x] load-gate idle≥70%（**实测 100% 全场**）·gov=performance 入 log
- [x] **无 git add/commit**·未碰 `lib/` / `schema/` / `docs/canon/`·主树/build/governor 未改·T3/recon 未动
- [x] 输出**零 gcc 字样**（`--gcc-install-dir` = clang CRT 定位 flag·prereg 钦定 recipe）
- [x] 硬冻结未碰（102/105·101/108·9/83·roster $meta·队序）
- [x] raw 全量落 `P1-remainder-raw/`（含 2 次失败尝试的完整 log·**零摘录·零挑轮次·零挑 seed**）
- [x] 未写 `/home/kingdom/phdworks/papers/`
- [x] **判据零修改**（沿用 `P1-backfill7-prereg.md`）·warmup-discard 明确拒用（= 改协议 = 改判据）
