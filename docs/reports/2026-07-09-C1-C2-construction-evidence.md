# C3′ 构造/覆盖证据链 synthesis(原 "C1/C2 构造证据",2026-07-10 正名)(G3, 2026-07-09)

> **贡献归属(2026-07-10 正名)**:本文综合的"稳定构造事实"(front-door 构造协议 + 前门化
> LOC 台账)**主体归 C3′**(能力键控优化模式库的构造 + 成熟编译器覆盖轴),**不是 C1 的
> 合取存在性、也不是 C2 的泛化代价律**。澄清:
> - **C1**(合取存在性 → 可复制协议)的真锚 = **capability schema 跨计算范式 × 跨独立
>   extension family 原封复用 + 外部贡献者接入协议**(见 [C1-1];IME N2 已证 `2eeabff9`)。本文
>   §A.3「跨 decode 格式成员复用」是 **decode 格式内**复用,只作 C3′ 可复制构造证据,**不**充当 C1 的
>   跨独立-extension-family 合取。
> - **C2**(泛化代价 → 边际成本规律)的真对象 = **独立 extension family 接入成本**,锚 IME
>   ≈2484 行,曲线尚缺失(见 `experiments/active/visibility/T2-ledger-anchor.md`)。本文
>   Part B 的前门化 LOC 摊销 = **C3′ 构造经济学**;decode 格式闭包 ∩ rvv.*≠∅,[F-6] 不进 C2 分母。
> **数值一个不改,仅正名归属标签 + "家族"→"decode 格式"用词。**

**这份文档只综合"稳定构造事实"层的证据(主体 = C3′ 构造/覆盖 + C1 schema-复用推论)。它刻意 *不* 依赖任何 perf 数**
—— 全部主张都是**结构 / 表征 / LOC** 命题(front-door 协议的存在性与可复制性、
构造边际成本的 LOC 台账),因此对正在进行的 min-term / M4 终审结局**完全鲁棒**。
(pending-M4 依赖数 = **0**。见 §C 独立性核对。)

- **C3′ 构造/覆盖**(本文主体) = *front-door 构造协议(可复制)+ 前门化边际成本谱系(the front-door-ization construction-cost law)*。
- **C1 推论** = schema 跨独立 extension family 原封复用(真锚在 [C1-1] + N1/N2 bridge,**非**本文 decode-格式内复用)。

**接地源(全部只读)**
- `experiments/active/frontdoor-framework/frontdoor_framework_ledger.csv`(seq 0–31,32 flip)
- `experiments/active/frontdoor-framework/{MANIFEST.md, NOTES.md}`(分解口径 + 首点拆解)
- `schema/coverage-sixstate.v1.json`(六态 ladder + C_construct 计数;roster 93 格)
- `lib/Plugin/RVV/RVVLowerQuantContraction.cpp`(family dispatch,只读)
- `docs/method/C2_marginal_cost_ledger.md`(layer-A 构造台账,已正名 C3′,交叉参照)

---

## Part A — C3′/覆盖构造: The Front-Door Construction Protocol(可复制构造配方)

> **正名注**:本节的 front-door 构造协议(abstract op→typed region 可复制构造)归 **C3′ 模式库构造 /
> 成熟编译器覆盖轴**,**不是** C1 的"合取存在性"。C1 的"可复制协议"另有真锚 = capability schema 跨独立
> extension family 复用 + 外部接入([C1-1] + N1/N2 bridge)。本节的"可复制"= 同一构造配方跨 decode
> **格式**复用,是 C3′ 证据,不充当 C1 的跨独立-extension-family 合取。

### A.1 协议 = abstract op → typed region → construction-from-abstract

统一配方,对三个 abstract 前门(`tcrv_rvv.quant_contraction` /
`dequantize_row` / `quantize_row`)同构成立:

1. **abstract 请求承诺一个 `scale_model` WHAT**(结构键,**非** format-name 标签)。
   这是 [换键不改条目] 纪律:dispatch 键是 *结构语义*(dual-fp16 / ternary-trit /
   kquant-dmin-bsums-min / codebook-flat),format 字符串仅作 provenance,不参与选路。
2. **front door 按 `scale_model` 结构键 dispatch 到一个 decode FAMILY**
   (`RVVLowerQuantContraction.cpp` 的 `lowerOne`:q4_0 ∨ ternary ∨ K-quant ∨
   codebook 四支,见 A.2 代码锚)。
3. **构造函数 CONSTRUCTS 出 typed region**
   (`typed_repack_gem{v,m}_loop_body` / `typed_dequantize_row_loop_body` /
   `typed_quantize_row_loop_body`),由 **per-family DecodeFacts** 参数化
   (base offsets + core-brick `decode_model` + optional qh 第二平面 / fold arity)。
4. **fail-closed (I7)**:承诺了 ternary/K-quant/codebook `scale_model` 但**不能**提供
   repack strip width(minVLEN < 128)的请求被**拒绝**,绝不静默误降成 q4_0 nibble
   block-dot(否则会把 trit / super-block 权重编错)。护栏在 `lowerOne` 尾部显式列举。
5. **[F-EMIT] provenance 护栏 + 六态执法 [L-8]**:构造出的 region body 携带 provenance
   token;六态 auto-readout(`e5_strong_readout.py`)走**真实 realized body 的 op-identity
   manifest**,判据 = *manifest 非空 ∧ 无 opaque `*_block_dot` 手写 helper*。**这正是把
   +1 记成 `constructed`(STRONG)而非 `constructed-weak`(仅 emitted)的判据** —— 即
   "emission ≠ construction" 的可执行界线(见 NOTES.md 的 tq2_0 双 pass labor-proxy:
   第一 pass 只做 emission 被 adversarial-verify 拦截,第二 pass 才补 CONSTRUCTION)。

### A.2 框架当前触达面(代码锚 = `RVVLowerQuantContraction.cpp:772–837`)

**4 个 matmul decode 格式家族**(RepackGem 谱,seq 0–7;C3′ 覆盖构造轴,**非** C1 的跨独立-extension-family),同一 `scale_model`→decode 表:

```
scale_model WHAT (结构键)          → decode FAMILY  → 构造函数
kTernaryTQ20/TQ10ScaleModel        → ternary        → lowerToRepackGem{v,m}Ternary  (+TernaryDecodeFacts)
kKQuantQ{4,6,2,3,5}KScaleModel     → K-quant        → lowerToRepackGem{v,m}KQuant   (+KQuantDecodeFacts: hasMin/foldModel/qh)
kCodebookIq4{Nl,Xs}ScaleModel      → codebook       → lowerToRepackGem{v,m}Codebook (+CodebookDecodeFacts)
(default dual-fp16-per-block)       → q4_0 nibble    → lowerToRepackGem{v,m}
```

**+ 共享 dequant 前门**(`TypedDequantizeRowLoopBody`,24 格 streaming):flat 4/5/8-bit /
K-quant super-block / IQ grid-table / codebook·ternary-grid 四类 decode-leaf,全部经
**同一** format-keyed 单源 `emitGgmlDequantizeRowExtended` 复用。

**+ 共享 quantize 前门**(`TypedQuantizeRowLoopBody`,3 格 streaming q8_0/q8_1/q8_K):
dequant 前门的 **MIRROR** copy-adapt。

**覆盖**:C_construct **33→66**(本 campaign 的 32 flip seq 0–31 + 1 个相邻 q4_0-prefill
增量 41→42 未入本台账),= **66/93 roster = 71.0%**
(`coverage-sixstate.v1.json`:`constructed` 66 · `dispatch-wired` 14 · `absent` 12 ·
`constructed-weak` 1)。六态 ladder = absent → emittable → dispatch-wired →
constructed-weak → **constructed(STRONG=C_construct)** → covered。

### A.3 可复制性论证(the replicability claim)

协议**可复制**的硬证据不是"我们又翻了一个格式",而是**每个家族走的是同一条
abstract→typed→emit 配方、且新成员对框架的 re-pay 系统性归零**(§B 台账量化)。三个
独立方向复证同一配方:

- **跨 decode 轴**:nibble(q4_0)→ trit(ternary)→ super-block 双/单尺(K-quant)→
  codebook gather(iq4)—— 四个结构完全不同的 decode 轴,共用 *同一* front-door dispatch +
  typed-region 骨架 + [F-EMIT] 执法。
- **跨 op 谱**:同一 construction 配方在 contraction(matmul,带 reduction/累加器/tiling)
  与 streaming(dequant/quantize,无累加器/无 tiling)两类 op 形状上都成立;quantize 谱
  更是 dequant 谱的 MIRROR copy-adapt(authoring cost < 从零一个谱)。
- **跨 decode 格式成员**(同一 decode 谱内,**非**跨独立 extension family):首格建框架、
  次格起 **facts-branch 复用**(dequant K-quant 尾格 q3_K = 3-line constructOrEmit facts
  分支,decode 整体从单源复用)。

即:协议是一份**可参数化的配置**(per-format DecodeFacts + scale_model 键),不是每格
重新推导。这就是 **C3′ 模式库可复制构造** 的实证(能力键控模式的构造配方);C1 的
"合取存在性 → 可复制协议"另有真锚 = **capability schema 跨独立 extension family 复用 +
外部贡献者接入协议**([C1-1] + N1/N2 bridge;IME N2 已证 `2eeabff9`),**不由本节
decode-格式内复用充当**。

### A.4 证据指针(commits / recompute anchors)

台账 `recompute_ref` 列携带每个 flip 的 recompute HEAD 锚(machine-anchored,
`git show <flip> --numstat -- lib/ include/`)。关键锚点:

| 谱 / 家族 | 首格 flip 锚 | 说明 |
|---|---|---|
| ternary repack(首个多家族泛化) | **`7a4250c5`**(tq2_0,显式 flip commit) | 首个 front-door multi-family generalization;NOTES.md 有逐文件 +948/−717 拆解 |
| ternary repack 次格 | **`0b907d0c`**(tq1_0,显式 flip commit) | 框架 re-pay 0 首证 |
| K-quant repack | anchor HEAD `ca0673a6`→`54d3741c`(q4_K…q5_K) | flip_commit 列标 pending-user-commit,recompute-anchored |
| codebook repack | anchor HEAD `81a8050b`(iq4_nl) | 新 codebook core-op family |
| dequant streaming(谱首) | anchor HEAD `530c0995`(q8_0) | 第二 front-door SPECTRUM |
| dequant K-quant / IQ / codebook-grid | `8783cd0f` / `76810f9f` / `f1201748` | format-keyed 单源复用 |
| quantize streaming(谱首,mirror) | anchor HEAD `ce19ed3c`(q8_0) | 第三 front-door SPECTRUM |

> **诚实标注**:台账 `frontdoor_flip_commit` 列除 seq 0/1 外记为 `pending-user-commit`;
> 实际落地由 `recompute_ref` 的 HEAD= 锚定(其中 `f1201748` / `0e39f60a` 等已在
> branch `refactor/full-refactor-m1` git log 可见)。证据是 **numstat-可重算的**,
> 不依赖任何运行时测量。

---

## Part B — C3′ 构造经济学: 前门化边际成本谱系(construction marginal-cost)

> **正名注**:本节的前门化 LOC 摊销律归 **C3′ 模式库构造经济学 / 成熟编译器覆盖轴**,**不是** C2 的
> 泛化代价律。C2 的真对象 = 独立 extension family(RVV/IME/scalar/zvfh)接入成本,锚 IME≈2484、曲线
> 待 X-SCALAR(见 `experiments/active/visibility/T2-ledger-anchor.md`);decode 格式不进 C2 分母。

### B.1 台账口径(framework paid-once vs family-specific)

`frontdoor_framework_ledger.csv` 把每个 flip 的 LOC 分成两条机器可锚列(NOTES.md
三层分解):

- **`reusable_framework_LOC`(tier-1,paid-once)**:让共享前门"多接受一个 decode
  家族"的验证器/dispatch 泛化。一旦在场,兄弟家族**逐字复用**,re-pay ≈ 0。
- **`family_specific_LOC`(tier-2 copy-adapt 模板 + tier-3 decode leaf)**:构造函数
  scaffold + core-brick shell + decode 常量/数学/emit body。

关键微妙:`family_specific_LOC` 大 **不等于** 新写得多 —— contraction/quantize/flat-dequant
的兄弟格常常是把退役 monolith 的 body **整体 RELOCATE**(net-new LOC ≈ 0,byte-identical
modulo provenance token);streaming K-quant/IQ/codebook-grid 的兄弟格更进一步,decode 从
**单源** `emitGgmlDequantizeRowExtended` 整体复用,family-specific 塌成 ~3-line facts 分支。

### B.2 边际成本表(per-family head/tail LOC,直接取自 ledger)

net_LOC = gross insert − gross delete(含退役偏移);paid-once = `reusable_framework_LOC`;
fam-spec = `family_specific_LOC`。**头格付框架、尾格 ~decode-leaf-only** 一目了然:

**谱 I — RepackGem contraction(matmul,4 decode 家族)**

| 家族 | seq | 格式 | net_LOC | paid-once | fam-spec | 位置 |
|---|---:|---|---:|---:|---:|---|
| ternary repack | 0 | tq2_0 | +231 (gross +948/−717) | ~60 | ~888 | **HEAD**(首个多家族泛化) |
| ternary repack | 1 | tq1_0 | −1193 | **~0** | ~170 | TAIL(base-3 leaf) |
| K-quant repack | 2 | q4_K | +211 | ~30 | ~700 | **HEAD**(K-quant class) |
| K-quant repack | 3 | q6_K | −330 | ~15 | ~500 | mid(no-min 泛化) |
| K-quant repack | 4 | q2_K | −463 | **~0** | ~500 | mid(min-fold 复用) |
| K-quant repack | 5 | q3_K | −490 | **~0** | ~470 | mid(no-min 复用) |
| K-quant repack | 6 | q5_K | −380 | ~8 | ~700 | TAIL(完成 K-quant repack) |
| codebook repack | 7 | iq4_nl | +288 | ~50 | ~750 | **HEAD**(codebook class;当前单成员) |

**谱 II — dequantize_row streaming(24 格,4 decode-leaf 类)**

| 类 | seq | 格式 | net_LOC | paid-once | fam-spec | 位置 |
|---|---:|---|---:|---:|---:|---|
| flat | 8 | q8_0 | +622 | ~525 | ~130 | **SPECTRUM HEAD**(base-expensive) |
| flat | 9 | q4_0 | +80 | ~85 | ~35 | nibble-class head |
| flat | 10 | q4_1 | +25 | **~0** | ~28 | |
| flat | 11 | q5_0 | +25 | **~0** | ~28 | |
| flat | 12 | q5_1 | +25 | **~0** | ~28 | TAIL(完成 flat legacy) |
| K-quant | 13 | q4_K | +81 | ~90 | ~28 | K-quant-class head |
| K-quant | 14 | q5_K | +3 | **~0** | **~3** | |
| K-quant | 15 | q6_K | +3 | **~0** | **~3** | |
| K-quant | 16 | q2_K | +3 | **~0** | **~3** | |
| K-quant | 17 | q3_K | +3 | **~0** | **~3** | TAIL(完成 K-quant dequant) |
| IQ grid | 21 | iq2_xxs | +93 | ~85 | ~8 | IQ-class head |
| IQ grid | 22 | iq2_xs | +3 | **~0** | **~3** | |
| IQ grid | 23 | iq2_s | +3 | **~0** | **~3** | |
| IQ grid | 24 | iq3_xxs | +3 | **~0** | **~3** | |
| IQ grid | 25 | iq3_s | +3 | **~0** | **~3** | TAIL(完成 iq2/iq3 cohort) |
| codebook/ternary-grid | 26 | iq1_s | +106 | ~100 | ~6 | codebook-grid head |
| codebook/ternary-grid | 27 | iq1_m | +7 | **~0** | ~6 | |
| codebook/ternary-grid | 28 | iq4_nl | +7 | **~0** | ~6 | |
| codebook/ternary-grid | 29 | iq4_xs | +7 | **~0** | ~6 | |
| codebook/ternary-grid | 30 | mxfp4 | +7 | **~0** | ~6 | FP4 negative-control |
| codebook/ternary-grid | 31 | nvfp4 | +7 | **~0** | ~6 | TAIL(完成 codebook-grid cohort) |

**谱 III — quantize_row streaming(3 格,dequant 谱的 MIRROR)**

| seq | 格式 | net_LOC | paid-once | fam-spec | 位置 |
|---:|---|---:|---:|---:|---|
| 18 | q8_0 | +554 | ~490 | ~117 | **SPECTRUM HEAD**(mirror copy-adapt of dequant seq-8) |
| 19 | q8_1 | +28 | **~0** | ~28 | |
| 20 | q8_K | +28 | **~0** | ~28 | TAIL(完成 quantize 谱) |

### B.3 摊销律陈述(the marginal-cost law)

**律**:*front-door-ization 的边际成本 = framework-re-pay(付一次,尾格系统性归零)+
family-specific 残差,其中残差 ∝ decode-leaf 到已覆盖原语空间的**结构距离**;当 decode
可从单源整体复用时,残差进一步塌成一个 ~3–7 行的 facts 分支。* 三条量化印证:

1. **framework 付一次**:每个谱/类的**首格**付 paid-once 峰值
   (60/30/50 · 525/85/90/85/100 · 490);其后**全部 24 个兄弟格 paid-once ≈ 0**
   —— 框架 re-pay 系统性归零,这是"合取协议是可复用配置"的直接 LOC 证据。
2. **head→tail 塌缩比**(net_LOC,同类内):
   - flat dequant `+622 → +25`(≈ 25×)
   - K-quant dequant `+81 → +3`(≈ 27×)
   - IQ grid dequant `+93 → +3`(≈ 31×)
   - codebook-grid dequant `+106 → +7`(≈ 15×)
   - quantize `+554 → +28`(≈ 20×)
3. **两种摊销 flavor**(残差为何有时仍大):
   - **RELOCATION flavor**(contraction repack + quantize + flat-dequant 头):兄弟格
     family-specific LOC 仍大,但 **net-new ≈ 0** —— 是退役 monolith body 的**逐字搬移**
     (byte-identical modulo provenance)。台账用 `of_which_copy_adapt_template_LOC` 列
     保留这份"有效再付 << 名义 LOC"的可见性。
   - **FACTS-WRAPPER flavor**(streaming K-quant/IQ/codebook-grid 尾):decode 从
     format-keyed **单源**整体复用,残差 = ~3 行 constructOrEmit facts 分支,是最干净
     的"泛化代价 → ~0"证据。

**与 layer-A 构造台账的关系(交叉复证,非重复)**:`docs/method/C2_marginal_cost_ledger.md`
(layer-A,同已正名 C3′)测的是 *原语内部* monolith→constructed 的构造边际成本,给出最锐利刻画
**"构造成本 ∝ 结构距离而非表面复杂度"**(q3_K 表面最难 hmask 却最便宜 +16,因结构已被
q6_K 覆盖;q2_K 不起眼却最贵 +388,因需第三 arity 标量 fold)。本台账(**layer-B**)测的是
*front-door 装置泛化* 到接受新 decode 格式的成本,在**同一律**上跨轴复证:framework 摊销到 ~0、
残差随结构距离与"是否单源可复用"塌缩。二者交叉参照相同 flip、回答不同构造经济学问题
(原语内部 vs 装置泛化),共同构成 **C3′ 构造经济学证据(L1-path / L2-schedule 两条 perf-cost
谱系之后的第三条纯结构/LOC 谱系)**。

### B.4 摊销曲线(每谱 net_LOC · head → tail)

```
谱 I  RepackGem contraction
  ternary   tq2_0 ██████ +231 → tq1_0 −1193   (退役盖过, framework re-pay 0)
  K-quant   q4_K  █████  +211 → …退役… → q5_K −380   (paid-once 30→~0)
  codebook  iq4_nl ██████ +288  (单成员; head only)

谱 II  dequantize_row streaming   (paid-once 峰值 → 0;net_LOC head → tail)
  flat        q8_0  ████████████████████████ +622
              q4_0  ███ +80 · q4_1/q5_0/q5_1 ▏+25 ▏+25 ▏+25
  K-quant     q4_K  ███ +81
              q5_K/q6_K/q2_K/q3_K ▏+3 ▏+3 ▏+3 ▏+3
  IQ grid     iq2_xxs ███ +93
              iq2_xs/iq2_s/iq3_xxs/iq3_s ▏+3 ▏+3 ▏+3 ▏+3
  codebook    iq1_s ████ +106
  -grid       iq1_m…nvfp4 (×5) ▏+7 each

谱 III  quantize_row streaming  (mirror)
  q8_0 ██████████████████████ +554 → q8_1 ▏+28 → q8_K ▏+28
```

曲线形状 = 每谱/类**一个尖峰(建框架)+ 一条贴地长尾(facts 复用)**。尖峰高度随
"是否 mirror 复用上一谱"下降(quantize 谱 +554 < dequant 谱 +622,因 mirror copy-adapt);
长尾高度随"decode 是否单源整体复用"下降(K-quant/IQ/codebook-grid 尾塌到 +3)。

---

## Part C — 独立于 min-term / M4 的核对(robustness)

本 synthesis 的**每一个数都是结构 / 表征 / LOC**,与 kernel 数值正确性、min-term
quantizer 争议、以及 M4 终审的任何结局**正交**:

- **C3′ 构造/覆盖**(原 Part A) = front-door dispatch 表 + typed-region 构造 + [F-EMIT] 六态执法 + 覆盖率
  66/93 —— 全部是 op-identity / 静态结构事实,`git show --numstat` 与
  `coverage-sixstate.v1.json` 可重算,不跑硬件。(C1 的 schema 跨独立-extension-family 复用推论
  同为结构事实,真锚在 [C1-1] + N1/N2 bridge,不在本文 decode-格式内复用。)
- **C3′ 构造经济学**(原 Part B) = ledger 的 paid-once / family-specific / net_LOC 三列 —— 全部 numstat-可锚,
  与 perf 台账(T3/T8、Win-B 八门)完全脱钩。
- **pending-M4 计数 = 0**:上述 C3′/C1 结构证据无任何一个数标 `pending-hardware` 或
  `pending-M4`。(六态 row 里 `pending-hardware` 标的是**数值/perf**主张;它们不进本文
  的结构论证。)min-term 证伪与否**不改动本文任何表格**。

因此 C3′ 构造证据链(+C1 schema-复用推论)可先行定稿、独立于 M4 终审入论文素材。

---

*本文件为纯 doc synthesis(论文素材),未 commit;touch-set 仅本文件;lib/schema 零改。*
