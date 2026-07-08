# T7 — G3 三曲线收口 (燃减 C_construct · 吞吐兑现 · 旁路存量)

> DATA CELL (visibility line E, G3-closure companion). Hand-curated **coupled** three-series over the
> G3 campaign, keyed on the 7 G3 front-door flip commits. Coverage/representation accounting ONLY —
> makes NO perf claim on the curves themselves (the realized column POINTS AT kernel-axis A/B rows that
> live in `T3_A` / the `l1-tile-*`+`l1-t3-*` cells under their own [NG-4]/[L-1] locks). The auto-generated
> double curve (C_construct + hand-LOC over full history) stays in `T7-burndown.md`; this artifact adds the
> two G3-specific series (吞吐兑现 / 旁路存量) that the generator does not track.

## Headline three numbers (G3 收口, schema-authoritative @ working tree)

| # | 指标 | G3 起 → 终 | 现值 | 权威源 |
|---|---|---|---|---|
| 1 | **燃减 — C_construct(强义)** | 33 → **40** | **40 / 93 = 43.0%** | `schema/coverage-sixstate.v1.json` (`coverage_metrics.py report`: C_construct num=40 den=93) |
| 2 | **吞吐兑现(格数)** | 1 → **4** | **4** | q4_0 e2e 5.9× + q4_K/q2_K/q5_K S6 kernel-轴 HOLDS(见下表 + `T3_A` 二.1 块) |
| 3 | **旁路存量(直连发射器格数)** | 17 → **10** | **10** | `schema/emit-bypass-whitelist.v1.json` `baseline_count=10`(== len(entries); ratchet-down) |

- **成熟判据进度**(执行总纲 §G3): C_construct 43.0% 已过 M2≥40% 门(commit `1b367c1f` 时 CROSS),向 ≥70%(→90%)推;
  旗舰格吞吐兑现 = 4;构造全走 front-door + 发射权威唯一 = K-quant 家族 5/5 已退役;旁路存量 17→10(**−7 over G3**),终态 = 0。
- **三曲线在 G3 上机械耦合**:每个 front-door 构造 = **+1 C_construct ∧ −1 旁路**(7 次一一对应);其中 3 个 K-quant(q4_K/q2_K/q5_K)的 S6 tiling **HOLDS** → **额外 +1 吞吐兑现**(q6_K/q3_K tiling NULL,不加兑现)。

## 耦合轨迹表(7 个 G3 flip;kernel-axis 数只作兑现-判定指针,[NG-4] 非 beat)

| step | flip commit | format | C_construct | 旁路存量 | 吞吐兑现 | tiling(S6) | 兑现新增(kernel-轴 A/B, [NG-4] 非 beat) |
|---:|---|---|---:|---:|---:|---|---|
| G3 起 | (22ab2ae9) | — | 33 | 17 | 1 | — | q4_0 e2e prefill **5.9×**(pre-G3 旗舰,唯一 e2e-兑现;board-proven 5/8 门) |
| 1 | `7a4250c5` | tq2_0 | 34 | 16 | 1 | (ternary n/a) | — (首个旁路退役 + 首个强义 C_construct 增) |
| 2 | `0b907d0c` | tq1_0 | 35 | 15 | 1 | (ternary n/a) | — (ternary 族全退役) |
| 3 | `7ff52fc4`(+`d5a28efc`/`5f194cbd`) | q4_K | 36 | 14 | **2** | **HOLDS** | q4_K S6 **1.884×** vs-opponent(spill 84→3, v30);裁决〇 兑现 1→2 |
| 4 | `c3cf7301` | q6_K | 37 | 13 | 2 | NULL | — (weight-bound;spill 913→949 rose, v31;~5.4× LOSS 未救) |
| 5 | `1b367c1f` | q2_K | 38 | 12 | **3** | **HOLDS** | q2_K S6 **1.413×**(FLIP 0.21×LOSS→WIN;spill 619→7, v30);M2≥40% 门 CROSS |
| 6 | `54d3741c` | q3_K | 39 | 11 | 3 | NULL | — (weight-bound;spill 978→894, v31;~5.3× LOSS 未救) |
| 7 | `df8a0b76` | q5_K | **40** | **10** | **4** | **HOLDS(hybrid)** | q5_K S6 **2.193×**(spill 155→105, v30;qh-plane 残留 floor);家族完整 5/5 |

## ASCII 三曲线(G3 起→终,keyed on 7 flips)

```
step:        起   1    2    3    4    5    6    7
             33   34   35   36   37   38   39   40
C_construct  ############################################   33 -> 40   (+7 强义构造, 每步 +1)
             33  ·34  ·35  ·36  ·37  ·38  ·39  ·40

旁路存量     ##########  <--- ratchet DOWN (每 front-door 构造退役 1 个直连发射器)
             17  ·16  ·15  ·14  ·13  ·12  ·11  ·10           17 -> 10   (-7, 终态 0)

吞吐兑现     q4_0(1) ......... q4_K(2) ..... q2_K(3) ..... q5_K(4)
             1    1    1    2    2    3    3    4            1 -> 4    (+3 K-quant S6 HOLDS)
                                (q6_K NULL)      (q3_K NULL)  <- tiling NULL 不加兑现
```

## 读法 / 纪律

- **燃减 C_construct(强义)**:强义 = 模式库原语构造(front-door typed region),非描述符选择的弱义。G3 净 +7
  (tq2_0/tq1_0 ternary + q4_K/q6_K/q2_K/q3_K/q5_K 五超块 K-quant 家族全构造)。**这是燃减主指标的分子**;分母 93 未随 G3 变。
- **旁路存量**:直连发射器(`kBlockDotKernels` 里绕过 front-door 的 `emitRepackGem{v,m}<fmt>`)= transitional scaffolding
  (喂 `C_dispatch`,永不 `C_construct`)。每退役 1 格入 `retired_ledger` 并 `baseline_count −1`(shrink-only ratchet,
  `[F-EMIT]` 门 fail-closed 护栏)。G3 退役 7 格(ternary 2 + K-quant 5),存量 17→10;残 10 = q4_1/q5_0/q5_1/q8_0 (batch0 flat) +
  iq2_xxs/iq2_xs/iq2_s (batch3) + iq4_nl/iq4_xs (batch2) + mxfp4 (batch4)。
- **吞吐兑现**:兑现 = 构造格把结构 opening **转成实测吞吐**。1(q4_0 e2e 5.9× prefill,唯一 **e2e** 兑现)→ 4
  (+q4_K/q2_K/q5_K S6 tiling **kernel-轴** HOLDS)。★兑现≠beat:后 3 格是 **kernel-轴 A/B**(单核、opponent=单线程
  block-dot proxy、8 [PERF-1] 门未走),整模型 e2e = **projection**(二.2 Amdahl prefill 上限 ≈1.59×,decode NULL,measured Δ 集成 BLOCKED)。
  q6_K/q3_K tiling **NULL**(weight-bound,~5.3–5.4× LOSS 未救)→ 不加兑现,如实登记为 marginal-cost/maturity 证据。
- **[XFER-1] register-cliff 迁移** = 4 个 sibling 预测从 q4_K S6 anchor 出发,**4/4 命中硅**(q2_K/q5_K 预测 HOLDS→实测 HOLDS;
  q6_K/q3_K 预测 NULL→实测 NULL);判别式 = ≤32-vreg 寄存器悬崖(v30=HOLDS / v31=NULL),对应瓶颈形状(min-fold-register-cliff vs dual-plane-weight-bound)。

## 与自动双曲线(`T7-burndown.md`)的对账

- `T7-burndown.md`(生成器,从 commit-subject 的 `C_construct N→M` token 解析)现渲染到 **C=39**(末 flip `54d3741c`);
  权威 schema = **40**。差 1 = 末构造 commit **`df8a0b76`(q5_K,第 40 个)的 subject 未含 `C_construct 39→40` token**
  (写了「家族完整 5/5 + [XFER-1] 4/4 + 吞吐兑现 4」但漏了 C_construct 数字),故生成器少计 1。**非未提交状态**——df8a0b76 已在 HEAD 历史;
  这是生成器解析口径的已知缺口。本三曲线用 **schema-authoritative 40**(与头条三数一致)。
- 生成器只出「C_construct + 手写LOC」双曲线,不出 兑现/旁路 两序;本 G3 companion 补齐三曲线。**无 perf 主张**在曲线本身。
