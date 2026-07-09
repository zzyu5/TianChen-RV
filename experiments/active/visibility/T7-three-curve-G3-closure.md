# T7 — G3 三曲线收口 (燃减 C_construct · 吞吐兑现 · 旁路存量)

> DATA CELL (visibility line E, G3-closure companion). Hand-curated **coupled** multi-series over the
> G3 campaign, keyed on the 9 G3 front-door flip commits (7 K-quant closure + M2-后 iq4 codebook pair).
> Coverage/representation accounting ONLY —
> makes NO perf claim on the curves themselves (the realized column POINTS AT kernel-axis A/B rows that
> live in `T3_A` / the `l1-tile-*`+`l1-t3-*` cells under their own [NG-4]/[L-1] locks). The auto-generated
> double curve (C_construct + hand-LOC over full history) stays in `T7-burndown.md`; this artifact adds the
> three G3-specific series (吞吐兑现 / 旁路存量 / 矿脉队列存量) that the generator does not track.

## Headline three numbers (G3 收口, schema-authoritative @ working tree)

| # | 指标 | G3 起 → 终 | 现值 | 权威源 |
|---|---|---|---|---|
| 1 | **燃减 — C_construct(强义)** | 33 → **42** | **42 / 93 = 45.2%** | `schema/coverage-sixstate.v1.json` (`coverage_metrics.py report`: C_construct num=42 den=93) |
| 2 | **吞吐兑现(格数)** | 1 → **4** | **4** | q4_0 e2e 5.9× + q4_K/q2_K/q5_K S6 kernel-轴 HOLDS(见下表 + `T3_A` 二.1 块) |
| 3 | **旁路存量(直连发射器格数)** | 17 → **8** | **8** | `schema/emit-bypass-whitelist.v1.json` `baseline_count=8`(== len(entries); ratchet-down) |
| 4 | **★矿脉队列存量(gemm_tile 待战役格数)** | — | **8** (+ 9 absent) | `schema/coverage-sixstate.v1.json` gemm_tile 28 = 11 constructed + **8 dispatch-wired** + 9 absent |

- **成熟判据进度**(执行总纲 §G3): C_construct 45.2% 已过 M2≥40% 门(commit `1b367c1f` 时 CROSS),向 ≥70%(→90%)推;
  旗舰格吞吐兑现 = 4;构造全走 front-door + 发射权威唯一 = K-quant 家族 5/5 + iq4 码本对 2/2 已退役;旁路存量 17→8(**−9 over G3**),终态 = 0。
- **三曲线在 G3 上机械耦合**:每个 front-door 构造 = **+1 C_construct ∧ −1 旁路**(9 次一一对应:7 K-quant 收口 flip + M2-后 iq4 码本对 2 flip);其中 3 个 K-quant(q4_K/q2_K/q5_K)的 S6 tiling **HOLDS** → **额外 +1 吞吐兑现**(q6_K/q3_K tiling NULL + iq4 码本对 S6 结构 no-op,均不加兑现)。
- **★矿脉队列存量 = 8(让 90% 路径"先陡后缓"形状可见)**:待战役 8 格 = dispatch-wired-但-repack-GEMM-未构造 = emit-bypass-whitelist 8 尾
  (**q4_1/q5_0/q5_1/q8_0** flat4 + **iq2_xxs/iq2_xs/iq2_s** iq2 码本 + **mxfp4** fp4 码本);另 **9 absent**(`iq1_m/iq1_s/iq3_s/iq3_xxs/nvfp4/q1_0/q4_0/q4_K/q8_0`,连 dispatch-wired repack GEMM 都未接)。
  形状 = **先陡后缓**:flat/ternary/K-quant 素材厚已陡峭燃掉(33→42);剩码本/三值 repack GEMM 的矿脉队列**单列**、**SEL-1 T4 后逐格战役立项、禁批量**(每格首翻=净新增-ODS 里程碑,batch 绕过 front-door construction 判据)。

## 耦合轨迹表(9 个 G3 flip:7 K-quant 收口 + M2-后 iq4 码本对 2;kernel-axis 数只作兑现-判定指针,[NG-4] 非 beat)

| step | flip commit | format | C_construct | 旁路存量 | 吞吐兑现 | tiling(S6) | 兑现新增(kernel-轴 A/B, [NG-4] 非 beat) |
|---:|---|---|---:|---:|---:|---|---|
| G3 起 | (22ab2ae9) | — | 33 | 17 | 1 | — | q4_0 e2e prefill **5.9×**(pre-G3 旗舰,唯一 e2e-兑现;board-proven 5/8 门) |
| 1 | `7a4250c5` | tq2_0 | 34 | 16 | 1 | (ternary n/a) | — (首个旁路退役 + 首个强义 C_construct 增) |
| 2 | `0b907d0c` | tq1_0 | 35 | 15 | 1 | (ternary n/a) | — (ternary 族全退役) |
| 3 | `7ff52fc4`(+`d5a28efc`/`5f194cbd`) | q4_K | 36 | 14 | **2** | **HOLDS** | q4_K S6 **1.884×** vs-opponent(spill 84→3, v30);裁决〇 兑现 1→2 |
| 4 | `c3cf7301` | q6_K | 37 | 13 | 2 | NULL | — (weight-bound;spill 913→949 rose, v31;~5.4× LOSS 未救) |
| 5 | `1b367c1f` | q2_K | 38 | 12 | **3** | **HOLDS** | q2_K S6 **1.413×**(FLIP 0.21×LOSS→WIN;spill 619→7, v30);M2≥40% 门 CROSS |
| 6 | `54d3741c` | q3_K | 39 | 11 | 3 | NULL | — (weight-bound;spill 978→894, v31;~5.3× LOSS 未救) |
| 7 | `df8a0b76` | q5_K | **40** | **10** | **4** | **HOLDS(hybrid)** | q5_K S6 **2.193×**(spill 155→105, v30;qh-plane 残留 floor);K-quant 家族完整 5/5 |
| 8 | `6c0961b6` | iq4_nl | 41 | 9 | 4 | no-op | — (框架第4家族 +codebook;S6 结构 no-op:已在 ≤32-vreg 悬崖 spill 7/v30,vwmacc=0=vluxei16 memory-gather) |
| 9 | `c238238a` | iq4_xs | **42** | **8** | 4 | no-op | — (iq4 码本对完整;S6 结构 no-op spill 73/v30;super-block signed6 no-min fold) |

## ASCII 三曲线(G3 起→终,keyed on 9 flips = 7 K-quant 收口 + 2 iq4 码本对)

```
step:        起   1    2    3    4    5    6    7    8    9
             33   34   35   36   37   38   39   40   41   42
C_construct  ####################################################   33 -> 42   (+9 强义构造, 每步 +1)
             33  ·34  ·35  ·36  ·37  ·38  ·39  ·40  ·41  ·42

旁路存量     ##########  <--- ratchet DOWN (每 front-door 构造退役 1 个直连发射器)
             17  ·16  ·15  ·14  ·13  ·12  ·11  ·10  ·9   ·8         17 -> 8    (-9, 终态 0)

吞吐兑现     q4_0(1) ...... q4_K(2) .. q2_K(3) .. q5_K(4) ........... (iq4 no-op)
             1    1    1    2    2    3    3    4    4    4          1 -> 4    (+3 K-quant S6 HOLDS)
                              (q6_K NULL)  (q3_K NULL)  (iq4×2 no-op) <- tiling NULL/no-op 不加兑现

矿脉队列存量  ████████ = 8 待战役 gemm_tile (+ 9 absent)  <- 90% 路径的唯一矿脉 (先陡后缓的"缓"段)
             flat4(q4_1/q5_0/q5_1/q8_0) + iq2×3(iq2_xxs/xs/s) + mxfp4;SEL-1 T4 后逐格战役、禁批量
```

## 读法 / 纪律

- **燃减 C_construct(强义)**:强义 = 模式库原语构造(front-door typed region),非描述符选择的弱义。G3 净 +9
  (tq2_0/tq1_0 ternary + q4_K/q6_K/q2_K/q3_K/q5_K 五超块 K-quant 家族全构造 + M2-后 iq4_nl/iq4_xs 码本对 = 框架第4家族 +codebook)。**这是燃减主指标的分子**;分母 93 未随 G3 变。
- **旁路存量**:直连发射器(`kBlockDotKernels` 里绕过 front-door 的 `emitRepackGem{v,m}<fmt>`)= transitional scaffolding
  (喂 `C_dispatch`,永不 `C_construct`)。每退役 1 格入 `retired_ledger` 并 `baseline_count −1`(shrink-only ratchet,
  `[F-EMIT]` 门 fail-closed 护栏)。G3 退役 9 格(ternary 2 + K-quant 5 + iq4 码本对 2),存量 17→8;残 8 = q4_1/q5_0/q5_1/q8_0 (batch0 flat) +
  iq2_xxs/iq2_xs/iq2_s (batch3) + mxfp4 (batch4)。
- **★矿脉队列存量(gemm_tile 待战役格数)**:六态 gemm_tile 28 = 11 constructed + **8 dispatch-wired** + 9 absent。**8 dispatch-wired == 旁路存量 8**
  (dispatch-wired 但 repack GEMM 未构造),是 90% 路径当前**唯一矿脉队列**:flat4 + iq2×3 + mxfp4。9 absent(iq1_m/iq1_s/iq3_s/iq3_xxs/nvfp4/q1_0/q4_0/q4_K/q8_0)
  连 dispatch-wired repack GEMM 都未接,是更深尾。形状 = **先陡后缓**:家族素材厚的段已陡峭燃掉;矿脉队列**单列、SEL-1 T4 后逐格战役立项、禁批量**。
- **吞吐兑现**:兑现 = 构造格把结构 opening **转成实测吞吐**。1(q4_0 e2e 5.9× prefill,唯一 **e2e** 兑现)→ 4
  (+q4_K/q2_K/q5_K S6 tiling **kernel-轴** HOLDS)。★兑现≠beat:后 3 格是 **kernel-轴 A/B**(单核、opponent=单线程
  block-dot proxy、8 [PERF-1] 门未走),整模型 e2e = **projection**(二.2 Amdahl prefill 上限 ≈1.59×,decode NULL,measured Δ 集成 BLOCKED)。
  q6_K/q3_K tiling **NULL**(weight-bound,~5.3–5.4× LOSS 未救)+ iq4 码本对 S6 **结构 no-op**(已在 ≤32-vreg 悬崖)→ 均不加兑现,如实登记为 marginal-cost/maturity 证据。
- **[XFER-1] register-cliff 迁移** = 4 个 sibling 预测从 q4_K S6 anchor 出发,**4/4 命中硅**(q2_K/q5_K 预测 HOLDS→实测 HOLDS;
  q6_K/q3_K 预测 NULL→实测 NULL);判别式 = ≤32-vreg 寄存器悬崖(v30=HOLDS / v31=NULL),对应瓶颈形状(min-fold-register-cliff vs dual-plane-weight-bound)。
  含 anchor q4_K 自身 + iq4 codebook 对(iq4_nl/iq4_xs,第③类 no-op)= 全族 **7/7**,见下「[XFER-1] 预测登记表」。

## [XFER-1] 预测登记表 (format × 瓶颈类 × 预测 × 实测 = 7/7)

> S6 output-tiling 迁移的**逐格预测-实测对账**(裁决〇.1b)。统一判据 = **S6 适用 ⟺ stageable decode strips 的高寄存器压力**
> (stack-panel 能把 COLD 已解码 strip 外置 → HOT 累加器 fan-out 掉到 ≤32-vreg 悬崖下)。三类:①min-fold register-cliff→HOLDS→tiled;
> ②weight-reconstruction-bound(多平面/hmask,non-stageable)→NULL→plain;③codebook-gather already-lean(vluxei16 memory gather,
> spill 已 ≤32,无 stageable strip)→structural no-op→plain。权威源 = `schema/pattern-registry.v1.json`
> (`PAT-S6-repack-gemm-output-tiling-register-cliff-XFER-1`);逐格 A/B 数 = `result-tables/T8_winloss_gap_ledger.csv` 的 7 条 XFER-1 行([NG-4] 非 beat)。

| # | format | 瓶颈类 | 预测(S6 transfer) | 实测 | 判别式(objdump: spill / maxVreg) | 一致 |
|---:|---|---|---|---|---|:--:|
| 1 | q4_K (anchor) | ① min-fold register-cliff | HOLDS → tiled | HOLDS **1.884×** | 84→3 / v30 ≤32 悬崖 | ✓ |
| 2 | q2_K | ① min-fold register-cliff | HOLDS → tiled | HOLDS **1.413×** (0.212×LOSS→WIN 翻转) | 619→7 / v30 | ✓ |
| 3 | q5_K | ① min-fold register-cliff (+qh 面) | HOLDS → tiled | HOLDS **2.193×** | 155→105 / v30 悬崖达成(qh 面残留 floor) | ✓ |
| 4 | q6_K | ② weight-reconstruction-bound | NULL → plain | NULL (~5.4× LOSS 未救) | 913→**949 ROSE** / v31 无悬崖 | ✓ |
| 5 | q3_K | ② weight-reconstruction-bound | NULL → plain | NULL (~5.2× LOSS 未救) | 978→894 / v31 无悬崖 | ✓ |
| 6 | iq4_nl | ③ codebook-gather already-lean | no-op → plain | no-op (structural) | 7 / v30 **tile 前已 ≤32**,vwmacc=0=memory-gather | ✓ |
| 7 | iq4_xs | ③ codebook-gather already-lean | no-op → plain | no-op (structural) | 73 / v30 **tile 前已 ≤32** | ✓ |

**= 7/7 预测-实测一致**(min-fold HOLDS ×3 / weight-bound NULL ×2 / codebook no-op ×2)。②与③都 ships-PLAIN 但机理不同:
②是**压力下没够到悬崖**(v31),③是**本就在悬崖下**(v30,无 stageable strip,无杠杆可施)。★kernel-轴 vs-opponent parity + spill/maxVreg,
**非 e2e、非 8-门封印**;整模型 e2e = projection(二.2 传导账,prefill Amdahl 上限 ≈1.59×,decode NULL)。

## 与自动双曲线(`T7-burndown.md`)的对账

- `T7-burndown.md`(生成器,从 commit-subject 的 `C_construct N→M` token 解析)现渲染到 **C=39**(末处理 flip `54d3741c`);
  权威 schema = **42**。差 3 = 生成器**尚未 regen 到 HEAD**:(a) `df8a0b76`(q5_K,第 40 个)subject 漏 `C_construct 39→40` token
  (写了「家族完整 5/5 + [XFER-1] 4/4 + 吞吐兑现 4」但漏 C_construct 数字);(b) M2-后 `6c0961b6`(iq4_nl,`C_construct 40→41`)+ `c238238a`(iq4_xs,`C_construct 41→42`)
  两 flip 已在 HEAD 历史、subject 均带 token,但生成器输出停在 `54d3741c` 未重渲。**非未提交状态**——三 commit 均在 HEAD;
  这是生成器 regen 口径落后 + 一处 token 缺失的已知缺口(生成器由主会话 regen)。本三曲线用 **schema-authoritative 42**(与头条三数一致)。
- 生成器只出「C_construct + 手写LOC」双曲线,不出 兑现/旁路/矿脉队列 三序;本 G3 companion 补齐(现为四序:燃减+兑现+旁路+矿脉队列存量)。**无 perf 主张**在曲线本身。
