# 模块化公式层 · 增量迁移账本（LEDGER）

> **用户裁决（2026-07-20）**：项目无统一「构建 + 消费公式」的模块化结构·θ=f(g,c) 散落各 pass ⟹ 选 **增量迁移**（非大爆炸重构）：立模块骨架 + 逐个迁散落选择器·每步 **byte-exact + 判决 lit + 三闸 Δ≤0**。**census = 本迁移的 inventory（清单）**。
>
> 本账本 = census v2（全景盘点）的**活 delta 层**：census v2 是钉死基线快照·本账本记「基线以来动了什么 + 下一步迁谁」。**禁把本账本当 census 正本**（正本在 `00-census-final.md`·pin 死）。
>
> **A1 读法订正**：本文包含按时间追加的历史判断；当前 HEAD 的 authority 正本是
> §四.10 与同目录 `AUTHORITY-MATRIX.md` / `authority-matrix.v1.json`。更早小节中的
> “未起”“下一步”“5/5 已闭环”等句只说明当时迁移阶段，不能越过 §四.10 当现行事实。

---

## 〇 · 公式层的目标「家」与当前住址

**目标 closed-form-f 宿主 = `include/Weft/Plugin/RVV/RVVGearboxSchedule.h`**（RVV 侧）。
A1 核查后的当前实际住址并未完全收敛：五类 dequant provider 只在该 header 声明，
定义仍住 `RVVToEmitCSupport.cpp`；repack accumulator-LMUL selector 仍住
`RVVLowerQuantContraction.cpp`；SP4/loop-order 住 `RVVRepackTilingSelection.h`。
因此“家已唯一”是目标，不是 HEAD 成就。已落的闭式原语包括：

| f | 住址 | 角色 | 落地 commit |
|---|---|---|---|
| `getRVVCodebookGatherAnchorLMUL(VLEN, SEW, codebookEntries)` | `RVVGearboxSchedule.h:2713` | codebook gather 锚 LMUL 闭式（最窄 LMUL 使 VLMAX≥codebook_size） | `0f7556199`（MIG-B/T2） |
| `rvvRegisterPressurePeakCost / …Legal / …enumerate` + `RVVRegisterPressureLevel/Combination` | `RVVGearboxSchedule.h:2141-2185` | 寄存器压力不等式（可行 LMUL 集闭式·守 [GAP-P1] 宽度选择 STEP②） | `0e9faee0a`（MIG-0） |

**c-probe 侧读者**（板事实 → c 输入·住能力表侧·非公式宿主）：`readRVVProviderVLenBBytes(module)` @ `RVVCapabilityProfile.cpp:389` / `.h:248`（MIG-A/T1）。

> 迁移方向 = 由 A2 的最小 typed decision contract 决定 plugin-local 住址；不为了
> “一文件统一”制造巨型 header。真正验收是同一决定只有一个 provider/legality/
> selector authority，selected result 在 emission 前落印，emitter 只机械消费。

---

## 一 · 基线快照（census v2·pin `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`）

**距 HEAD = 59 commit**（多数是 grid dequant flip + doc·不动 θ 分类结构；动结构的 = 下表 6 步）。

| 维度 | 基线值（pin d173f4c2e） | 口径出处 |
|---|---|---|
| θ 总数 | **35**（θ1–θ35） | `00-census-final.md` 首节 |
| ├ (a) 由 f 现场算出 | **11** | 同上 |
| ├ (b) 盖章+fail-closed 守 | **1**（θ35 audit-mirror） | 同上 |
| ├ (c) 焊死在代码里 | **21**（θ9-21 的 13 coreLmul + θ27-34 的 8 门/旋钮） | 同上 |
| ├ (d) 由测量表选出 | **0**（θ1 表恒空·θ5/6/7 表作 fact 融入 a） | 同上 |
| └ 派生-未归 | **2**（θ25/θ26） | 同上 |
| f 加工环节 | **37**（F1-F37·单侧输入 18） | pkg4 |
| 残余焊死 g | **18 处**（GridCodebook 9 + Ternary 7 + KQuant 2；另 ForwardElementwise ~51 宽谓词 ISSUE-119 未穷举） | pkg3 |
| provider property 键 | **15·零消费 9**（vlenb:bytes / clang:version / cmake:version / compile_run×4 / march:value / mabi:value） | pkg2 §2c |
| schema params namespace | **7·全 grep=0 in code**（vlen/elen/sew_set/lmul_budget/vreg_count/cacheline/ime.tile） | pkg2 §1d |
| RVVProbeCapabilityFacts 字段 | **15·零消费 2**（cachelineBytes / imePresent） | pkg2 §2b |

---

## 二 · 迁移 delta（基线以来·逐步·每步一 commit）

> 记法：每步标 **动了哪个计数** + **byte-exact/判决 lit 证据** + commit。

### MIG-0 · 寄存器压力不等式立为显式 f（`0e9faee0a`）
- **动作**：把 [GAP-P1] 宽度选择的可行域从散落逻辑提为家里的显式闭式 f（`rvvRegisterPressurePeakCost/Legal/enumerate`）。
- **计数**：f 家 +1 族；接入 dot-reduce enumerator + repack accumulator（byte-exact·产物不变）。
- **判决 lit**：`test/Conversion/RVV/rvv-register-pressure-inequality-decisive.mlir`（budget 32→i32m8 / budget 9→i32m1·合法集随输入真变）。

### MIG-A · vlenb_bytes → 承重 minimum_vlen 源（真封口 `abb7e0304`·.h `bcde2212c`）（= T1）
> ⚠**封口订正**：`17fb314e2`（原记）**只提交了 task 元数据**（check/implement.jsonl/prd/task.json）·**.cpp 定义+wiring+判决 lit 从未入库**（误提交·遗漏工作树）。真封口 = `abb7e0304`（byte-exact + FileCheck 2/2 DECISIVE PASS·2026-07-20 核出）。
- **动作**：真板 VLEN 事实 `rvv.vlenb_bytes`（probe 盖章）成为 minimum_vlen 承重源·**优先于 -march 猜**（VLEN = VLENB×8）。
- **计数**：provider 零消费 **9→8**（vlenb:bytes 从"声明未消费"→真消费）；c 输入「VLEN」从"march 解析猜"→"吃板事实"。
- **判决 lit**：`test/Conversion/RVV/rvv-vlenb-source-vlen-decisive.mlir`（vlenb=32⊥march=zvl128b/zvl512b·θ 恒 half_lanes=16 跟 vlenb 非 march）。

### MIG-B · θ14 CodebookFp4 coreLmul 焊死→闭式 f（`0f7556199`）（= T2·迁移雏形）
- **动作**：θ14 `RVVToEmitCCodebookFp4.cpp:109 coreLmul="m1"` 焊死 → `getRVVCodebookGatherAnchorLMUL(VLEN,SEW,codebook.size())`（家里的闭式）；nvfp4 verifier `==m1` → `VLMAX≥codebook.size()`。
- **计数**：**焊死(c) 21→20**·**a 11→12**（θ14 从 (c) 迁 (a)）。**首个 焊死→f 迁入家 = 增量迁移原型**。
- **byte-exact**：codebook anchor 2→0 硬编码消除·产物不变（board VLEN128/SEW8/256-entry → m1 与旧焊死同值·但现由 f 算出）。

### MIG-C · [GAP-P1] 松·measured-table 首行（`2ab1f9d4d`）
- **动作**：`lookupRepackMeasuredM1Faster` 从 nullopt stub → registration-as-DATA 表 `kRepackMeasuredM1FasterMeasurements`（{q8→m1Faster}·仿 sanctioned `kRepackVlen256DecodeMeasurements`·无 format-switch/无 sentinel/无签名改/非 blind-widest）。
- **计数**：**measured-table (d) 0→≥1**（supervisor 明裁·`性能与测量.md:440`「census measured-table θ=0 缺口首次闭合 d=0→d≥1」）。θ1 现真消费 measured-table（q8 走 m1）。
- **board 证**：q8 repack m1 双 regime WIN（decode GEVM 1.6-2.4× / prefill GEMM ~1.40×）·byte-exact 3-arm mism=0·CORE==PROD（GEVM md5 3223e26f / GEMM 17fcbd63）·spill-free（m1 chain 6≤32）。
- **判决 lit**：`test/Conversion/RVV/rvv-repack-accumulator-lmul-measured-gate-q8.mlir`（q8→m1·q4_1→mf2）。
- **★资格注（audit C2/B4）**：perf 幅度走 ad-hoc·无 T-N·= measured 非 T-N-qualified·pending bench 复测（[[ISSUE-107]] 同注）。**结构（measured-gate 通道活了·byte-exact）是硬事实**。

### MIG-C2 · measured-table 扩到 repack 家族（`a230adc61`·W5/B1 board-sweep）
- **动作**：B1 [GAP-P1] repack 家族 @rvv VLEN128 board-sweep·`kRepackMeasuredM1FasterMeasurements` 加 2 行。
- **计数**：**measured-table (d) 1→3**（q8 + **q4_0** + **q4_1**·全 registration-as-DATA·无 format-switch）。
- **board 证**：q4_0 m1 双 regime 赢（decode 2.3-2.5× / prefill 1.24×·spill=0）；q4_1 m1 decode 2.3× 赢·prefill parity·spill=0。全 3-arm byte-exact mism=0·CORE==PROD·9 lit PASS。
- **★[GAP-P1] 不等式 VINDICATED**：q5_0/q5_1 **KEEP mf2**（honest-null·非硬塞）——decode m1 赢但 prefill GEMM m1 emit **SPILL**（objdump 1-2 whole-reg reload·2.3-2.4× SLOWER）·`rvvRegisterPressureLegal`(MIG-0) 的 spill 预测**兑现**·一行翻双 regime ⟹ 保 mf2。
- **★q8 vs-对手 = honest-null**（naive 投影证伪·未造数）：ggml x16 对手 VLEN256-tuned·VLEN128 byte-INEXACT=非合法对手·~4-5× 无效。真 VLEN128 对手识别 = 余项。

### MIG-T3 · θ20 iq2_xxs 试迁 = 诚实-null（`6bc02a7cc`）（= T3）
- **动作**：试把 θ20 `RVVToEmitCKQuant.cpp:5971 value_or("m2")` 提为 measured gearbox。
- **结论**：**honest-null**——board 证伪前提（VLEN-correctness 墙·非 measured gearbox 可翻）。**无计数变化**（θ20 维持 (c) value_or）·记「试过·held」防复试。emit-neutral 注释入 `RVVToEmitCKQuant.cpp`。

### MIG-1 · 律2 entryLanes→描述符（**✓ 完成**·`06cc93fb3`·ISSUE-118 路B）
- **动作**：3 处格式常量 `const int64_t entryLanes=N`（iq3_s=4/iq2_xs=8/iq1_m=8）从**焊死在发射体**→**typed `codebook_entry_lanes` OptionalAttr**（住既有 `DequantizeRowStreamFacts`·前门 stamp·发射体 fail-closed 读）。
- **判决实验 3/3**（committed lit·`rvv-to-emitc-dequantize-row-iq3-s-codebook-entry-lanes-descriptor.mlir`）：正常值 byte-exact / 改 attr 4→7→emit vl 变(真消费) / 缺席→verify-fail 具名诊断(非 value_or)。
- **三闸 Δ**（GridCodebook.cpp）：①裸格式字面量 **3→0(Δ−3)** ②value_or 0→0 ③march 0→0。byte-exact md5 iq3_s/iq2_xs/iq1_m 不变·CORE==PROD·主树重建+8 dequant lit PASS。
- **★诚实边界**：PRD 触碰集 naive-grep 误判（entryLanes 实住 dequant-row 发射体·format 分发·非 grid op Context）→重路由到 `DequantizeRowStreamFacts`；标签 off-by-one；**`groupLanes`（:504/1002/2532）守边界未迁**（属 ISSUE-118 簇1 block-dot·不同前门·留后续）。
- **计数**：GridCodebook 焊死 −3。

### MIG-1b · ISSUE-119 census（`7325317a7`·只读·分母正名）
- **动作**：ForwardElementwise dequant-row 宽谓词穷举到 100%·**分母 = 59**（精确·替"~51"/"~40"均 undercount·一条可复现谓词定）。全 59 = (c) 焊死-g·派生/other=0·board-invariant。
- **候选迁移点**（future MIG·census 列不裁）：①`:5366-5386` 19-format stride switch（最高扇出·monolith locus）②vector-leaf geometry decls ③triple-baked q4_0/1/5_0/1 nibble(3 loci) ④residual double-bake。

---

## 三 · 历史过渡快照（旧 pin；不是当前 HEAD）

| 维度 | 基线（d173f4c2e） | HEAD（b097d245a·近似） | 说明 |
|---|---|---|---|
| θ (a) f 算出 | 11 | **12** | +θ14（MIG-B） |
| θ (c) 焊死 | 21 | **20** | −θ14（MIG-B）；MIG-1 在飞不改 θ 类（改 g-焊死处计数） |
| θ (d) measured-table 行数 | 0 | **3**（q8/q4_0/q4_1） | MIG-C(q8) + MIG-C2/B1(q4_0/q4_1)·q5 spill-KEEP mf2 |
| 残余焊死 g | 18 | **18→≤15**（MIG-1 落后） | GridCodebook entryLanes/groupLanes |
| provider 零消费 | 9 | **8** | −vlenb:bytes（MIG-A） |
| 家里 closed-form f 族 | 0（家未立） | **2**（codebook-anchor + reg-pressure） | MIG-B + MIG-0 |

> **★诚实**：θ 精确重分类（尤其 θ1 是 (a-含-d-fact) 还是 (d)·θ5 先例算 a）**待 census-v3 re-pin**（需重跑 pkg 六包 agent·本账本只记 delta 事实与 supervisor 裁·不擅自重算 35 分布）。

> **★census2 breadth 判定（2026-07-20·增加公式 breadth·`experiments/active/census2-singlesided-f-zeroconsume-judgment.md`）**：census v2「欠账」多半已消/低值——**18 单侧 f = 13 legit / 2 真 fake（F7 dead-param 删·F25 baked blockLen invariant-guarded）/ 3 零输入 OK**（'18 单侧 f 欠账'多半假警报·禁误伤 legit）；**11 零消费键 = 1 已接(vlenb/T1) / 10 删**（cachelineBytes/imePresent/deriveIMEPresent 已删@HEAD·grep=0 verified·+8 provider provenance echo 低值可删）。**census pin drift 确认**：pkg2「零消费 9」现为 8。⟹ **高值「增加公式」= 结构迁移（phase-1/phase-2 DequantMechanismPlan）·非 breadth cleanup**（breadth 多是 no-op/低值 provenance 删）。

---

## 四 · 迁移队列（增量迁移·排序·依赖标注）

| # | 迁移 | 目标 | 依赖/闸 | 状态 |
|---|---|---|---|---|
| MIG-1 | 律2 entryLanes→描述符（GridCodebook·grid 3 格） | 焊死 −3 | §四.5 三闸 | **✓ 完成**`06cc93fb3` |
| **phase-1** | **nibble 家 vertical slice**（q8_0/q4_0/q4_1/q5_0/q5_1·结构化 descriptor·decode-mechanism 8-tuple 中心化·删 emitter re-bake）·= §六.3·**F1 falsifier=验收门**（合成格式只加 descriptor 行·C1/C2 度量） | phase-① 第二步·来源解耦 | byte-exact by-construction·5 风险点 assert·**与 MIG-5 文件不相交可并行**（不碰 RVVGearboxSchedule.h） | **下一个可派** |
| MIG-5 | 家族/宽度选择器 argmin→具名闭式 f（reduction/contraction 轴·θ_width=f(VLEN,dataWidth)） | phase-③ 原语 | 判决实验=翻 VLEN→输出翻·防假重构 | **在飞** |
| **phase-2** | **NibbleDecodePlan**（MechanismPlan 抽象首个·FormulaProvider→plan→emitter·byte-exact reproduce-current） | 建立 plan 抽象 | plan 承重判决 lit(MUTSTRIP qk/2 派生 裸描述符读不出) | **✓ 完成**`26b7c3e9a` |
| MIG-2 | coreLmul 真焊死 θ9-13→f（GridCodebook 497/995·Ternary 862·BQL 111/404） | 焊死(c) 20→15 | **BLOCKED**：[K-10] MAINTAIN——Context **无 coreLmul 字段·无 IR 宽度可读**·强 lift=造假旋钮。**须先给描述符加 coreLmul typed 字段**（= plan.load_lmul·MIG-1/phase-2 机制已铺路） | 阻塞·待字段 |
| **phase-3/4 排序洞察** | c 驱动 θ 扩到 **codebook/grid 家**（gather anchor=f(VLEN,codebook)·MIG-B 已闭式）·**非 nibble** | dequant-row c-驱动首现 | — | **待排·见下诚实注** |
| MIG-3 | `selectRepackAccumulatorLMUL` 家族→搬进 `RVVGearboxSchedule.h` 家 | f 集中化（现散在前门 `RVVLowerQuantContraction.cpp:1285`） | byte-exact·消费侧 18 fail-closed 读不变 | 待排 |
| MIG-4 | measured-table 扩行（更多格式入 `kRepackMeasuredM1FasterMeasurements`） | (d) 3→≥N | **repack 家族已扫**（B1·a230adc61·q4_0/q4_1 FLIP·q5 spill-KEEP·q8 vs-对手 honest-null）。**余 = K-quant decode repack-GEVM**（q2/q3/q4/q6_K·路由确经 selectRepackAccumulatorLMUL:3910·未板扫·据 q5 GEMM-spill 律预测多半 mf2）+ q8/q4 **vs-真-VLEN128-对手**（4x8/SpacemiT/block-dot 识别·gated on bench+T-N） | 部分完成·余卡 board+bench |

---

> **★phase-3 排序诚实注（2026-07-20·phase-2 后）**：phase-③「c 真驱动 θ」对 **nibble 家 likely honest-null**——nibble dequant 几何是 **qk-fixed**（strip=qk/2·loadLMUL 现固定）·census 证 dequant-row 轴 0 board-fork（board-invariant）。⟹ 真 c-驱动 θ 的机会在 **codebook/grid 家**（gather anchor=`getRVVCodebookGatherAnchorLMUL(VLEN,codebook_size)`·MIG-B 已为 FP4 闭式·随 VLEN 真变）。**⟹ 高值下一刀 = 把 MechanismPlan 抽象扩到 codebook/grid 家**（那里 c-驱动真实·= phase-③+④ 合流）·而非对 nibble 强推 phase-③（会造假 c-旋钮·nibble 无此轴）。nibble phase-2 已足（抽象建立·byte-exact）。

## 四.9 · 两层诚实审计定论（2026-07-20·两 Workflow 机核@HEAD d23a9e32c·非凭 census 旧 pin）

**能力层 ~1.5/5**（机制真·广度窄·表未建）：✓ VLEN 轴真通（minimum_vlen typed + 拔管道收敛前门 0 直调 + provider 压 march + 2 θ 消费者 conflict-decisive 真绿）·✗ vreg_count 硬编 `return 32`+8 constexpr / version 轴残 1 处 load-bearing march 重解析(`RVVLowerQuantContraction:1661` typed reader 已建未换) / elen/sew_set(typed)/cacheline/ime.tile schema grep=0 / 无单一 per-board 权威实例。打分 ①部分②部分③未④部分⑤未。

**公式层 1/5**（原型证成·首家闭环·未铺满）：✓ NibbleDecodePlan 真 struct + FormulaProvider→plan→emitter 链闭环(nibble)+ 承重 lit + C1/C2 falsifier + 闭式 f 家单一权威·✗ 4/5 家仍格式名硬分派(`ForwardElementwise:4950-4991` 13 行 `decodeModel==`) / **公式家双头**(`GridDecodePlan` 独立注册表·不经 FormulaProvider·grid+ternary 揉一起违 [K-10]) / 已落 plan 全 reproduce-current(`nibbleDecodePlanFromFacts` `(void)minimumVLEN`)。**★nibble reproduce-current = honest-null**(qk-fixed·0 板分叉·非欠工)·真 c-驱动在 codebook/grid(getRVVCodebookGatherAnchorLMUL 已备·call-site VLEN 钉死 128·`RVVToEmitCCodebookFp4:118` measured-gate deferred)。

**★铺满三刀（审计钉·= 用户 5 阶段 ④③ + phase-2 修）**：① 扩覆盖 1/5→5/5（KQuant/Codebook/Ternary 各建分立 plan·13 行 `decodeModel==` 降 plan.mechanism）② 合并双头家（GridDecodePlan 收编 DequantMechanism-tagged·拆 grid+ternary lumped·nibble provider 定义迁回家）③ c 真驱动（选对轴=codebook/grid·解 :118 gate·板验 measured）。
**审计后进度（2026-07-20·逐刀落地·每刀 byte-exact + 判决 + 全套 0 new）**：
- **能力层 ②③ ✓**（`54e9cb483`·cap-complete）：VLEN + version 两轴管道全收敛（前门直调都 0）· vreg_count 正名（硬编 32→in-IR 能力事实·8 constexpr 收敛·判决 2/2）。剩 elen/cacheline/sew_set(typed)/per-board 实例。
- **公式层刀① 扩覆盖 1/5→3/5**：CodebookGatherPlan（`0beb9cb71`·codebook 家 mxfp4/nvfp4/iq4_nl/iq4_xs·byte-exact 4/4）+ KQuantScaleMinPlan（`99621a7bb`·q2/3/4/5/6_K·byte-exact genuine 5/5·plan 承重 dmin=scale+2 派生铁证）。**已进 plan 家 = 3/5**（Nibble/Codebook/KQuant）。
- **公式层刀①②达成 → dequant-row head 5/5**（`95dc34a88`·GridLookupPlan + TernaryDecodePlan·byte-exact 9/9·[K-10] 5 分立·**零残留 decodeModel== grid/ternary 链**·合并双头家 dequant-row head DONE）。**5 mechanism 全进 plan**（Nibble/Codebook/KQuant/GridLookup/TernaryDecode）·emitter 全 plan.mechanism-driven·格式名降 provenance。
- **剩（下一程·非本收尾）**：刀③ **c 真驱动**（codebook/kquant/grid 的 minVLEN seam 已就位·一行 provider 改·plan 轴第一个真 c-驱动·codebook :118 measured-gate 需板）· GridDecodePlan registry 的 block-dot/verifier head 双头残留（ISSUE-122 defer）· 能力层 elen/cacheline/sew_set(typed)/per-board 实例。

> **★2026-07-20 收尾态（历史表述，受 §四.10 限定）**：能力层 ②③ ✓（VLEN+version 管道·vreg_count 正名）· 公式层 dequant-row head **1/5→5/5**（5 家全进 plan·byte-exact·plan 承重判决）。这里的“5/5”只表示 plan 类型与 emitter 消费铺满，**不表示 c、stamping、legality、selection、realization 已闭环**。

## 四.10 · A1 HEAD authority freeze（当前正本）

完整符号链见 `AUTHORITY-MATRIX.md`；机器 census 与任务/issue 绑定见
`authority-matrix.v1.json`，由 `test/Scripts/formula-authority-matrix.test` 守卫。

### 五类 dequant 的逐阶段状态

| plan | defined | typed g stamped | provider consumed | selected plan stamped | emitted | mutation tested | c/ω disposition |
|---|---:|---:|---:|---:|---:|---:|---|
| NibbleDecodePlan | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | c=literal 128 且 ignored；ω honest-null |
| CodebookGatherPlan | ✓ | 部分 | ✓ | ✗ | ✓ | ✓ | 已有 anchor closed form，但本 provider 未用；ω honest-null |
| KQuantScaleMinPlan | ✓ | primary g | ✓ | ✗ | ✓ | ✓ | c=literal 128 且 ignored；ω honest-null |
| GridLookupPlan | ✓ | entry lanes | ✓ | ✗ | ✓ | ✓ | narrow body c honest-null；legality 仍查第二 registry head |
| TernaryDecodePlan | ✓ | iq1 entry lanes | ✓ | ✗ | ✓ | ✓ | c/ω honest-null；leaf 仍由 emitter 字符串映射 |

**关键订正**：

- “五类 plan 已有”是真资产；后续任务不得再从零设计第五套万能 plan。
- “emitter 全 plan.mechanism-driven、格式名只剩 provenance”不是 HEAD 事实：
  codebook/K-quant/grid/ternary 仍各有 emitter-local `decode_model → typed leaf enum`
  `StringSwitch`，共 4 段；provider 也仍在 emission 内调用。
- 五个 provider 的 `minimumVLEN` 都显式 `(void)`，五个 caller 都传字面量 128。
  Grid/Ternary 可记 honest-null；Codebook 的真实 c-driven closed form 已存在却未接线，
  由 A3 承接；禁止把所有 family 都强制制造 c 分叉。
- plan 的 g mutation tests 很强，但它们证明的是 `defined→consumed→emitted`，不能
  代替 `selected stamped→verified→mechanical emit`。

### 非 dequant 决定

| decision | 当前已成立 | 当前未成立 | 后续 owner |
|---|---|---|---|
| repack accumulator LMUL | c 真消费、register-pressure legality、measured/default 分叉、18 emitter fail-closed reads | selector/typed 输入/reason stamp 在 18 builder 重复；winner 表手工镜像 | A2 + A5 / ISSUE-117 |
| SP4 | bounded variants、两阶段 selector、selected stamp、measured/prior tests | legal set 纳入无 emitter body 的 min-fold Plain；缺 stamp→S6；verifier 不认识 stamp | A4 / ISSUE-125 |
| loop-order | stride prior、measurement hit、selected stamp | sibling 把 col_outer/prior override 回 row_outer；q4 缺 stamp重算；verifier 不认识 stamp | A4 / ISSUE-125（连 ISSUE-034） |

### 原子退役清单

1. A2：一个 dequant slice + accumulator-LMUL slice 全 caller 迁入 typed decision
   contract；当笔删除旧 overload/direct caller/adapter，不留兼容入口。
2. A3：让 codebook 的 `getRVVCodebookGatherAnchorLMUL` 真决定 plan/legal set，删除
   declared slice 的 literal-128/ignored seam；当前合法输出保持 byte-exact。
3. A4：selected result 改为 required/verified；关闭 Grid 双头、SP4 假 legal candidate、
   missing-stamp 默认与 loop selected→realized override/recompute。
4. A5：B1 measurement control plane 生成 qualified winner view，取代
   `kRepackMeasuredM1FasterMeasurements` 与 local `kSeeded[]`。
5. A8：按 typed-g/derived/structural/dead/unresolved 分类收敛 emitter baked g；只迁
   真 g，结构常量不包装成假参数。

以上对象均已绑定 Trellis task、ISSUE 与 killing test；账本不再充当私设问题清单。

## 五 · 每步纪律（三闸·不可绕）

1. **byte-exact ZERO-MODEL + 3-arm anti-hollow + CORPUS**：从实际输入零复用重算·三向 mism=0·CORE==PROD（md5 对齐）。
2. **判决 lit（独立复核）**：每迁一步配 decisive lit（能力真翻·false-green 会挂）·独立 agent 跑判决不看 diff。
3. **§四.5 三闸 Δ≤0**：three-grep（裸格式字面量 / 板值 value_or / march 解析）计数不增·readings 入 commit。
4. **canon 措辞变更入 ISSUES 不自改**（用户裁除外）；**🔴禁内联汇编 / 钉死调度绕 clang**。

---

**账本维护**：每完成一 MIG·在 §二 追一节 + §三/§四 更计数与队列。**基线（§一）钉死不动**·只在 census-v3 re-pin 时整体翻页。


---

## 六 · 升级目标节 · dequant 模块化统一链（DequantMechanismPlan）

> **来源**：外部 agent 架构建议（待核实非照搬）+ 4 份只读研究 + supervisor 4 项已核实事实。
> **主线归属**：论文柱一（工程证据·C1 可扩展性 / C2 边际成本）。**本节 = 升级目标·非承诺已落**。落地仍走 §二 逐 MIG + §五 三闸。
> **一句话目标**：把 dequant 的执行分派权从「格式名字符串」搬到「结构化 g 描述符 → FormulaProvider(g,c) → MechanismPlan → emitter 只读 plan」的链上；格式名降级为纯 provenance/诊断。**当前诚实态 = 来源部分解耦·推导与消费仍耦合在 `format`/`decode_model` 字符串**（研究 2 分级图）。

---

### 六.1 · 目标架构（MechanismPlan 字段形态 + 住址 + canon 依据 + 复用面）

**canon 合规支点（研究 1 权威边界·已核实）**：
- FormulaProvider(g,c)→MechanismPlan 的 canon 类别 = **Selected-Body Realization**（transient C++ 编译期操作·`插件协议.md:83,93-105,144` 明列 Plugin-Owned）。**非** weft.exec（I2·计算语义属 extension family）、**非** schema.def 第六件 shape（[S-5]·插件内部代码不入 shape·改它不触 `schema.def`）、**非** route provider 自建 route（plan 喂 realized body 给 route provider）。
- ⚠**hook 名订正（load-bearing）**：外部建议的 `realizeSelectedVariantBody` 只是 spec 散文通用名。真实可覆写 hook = **`materializeSelectedLoweringBoundary` / `validateSelectedLoweringBoundary`**（`include/Weft/Plugin/ExtensionPlugin.h:684-688`·registry 侧 `:745-749`·**已核实**）。
- ⚠**代码-locus vs canon-类别 的诚实区分**：dequant-row 的实际下降走 `--weft-rvv-lower-to-emitc` 转换 pass + 前门 pass（`RVVDequantizeRowStreamFrontDoor.cpp`），是 **RVV 插件内部 pass**，今天**并未**穿过 `ExtensionPlugin` 的虚 hook。故：**canon-类别 = realization**（授权本工作为 plugin-owned·合法性来源），**代码-住址 = RVV 插件内部构造/发射 pass 内联**。MechanismPlan 是插件内部 transient C++ 对象·在构造↔发射之间流动·**禁**抬进 `VariantLoweringBoundaryRequest/Result` 等跨-ABI 结构（[P-1] 准-ABI·一旦跨 hook 边界即触 RFC）。

**MechanismPlan 字段形态**（研究 3 · 每字段标 复用/新增 + 来源 file:line）：

| 字段 | 类型 | 复用/新增 | 来源（现存·已核实） | 备注 |
|---|---|---|---|---|
| `mechanism` | `enum {NibbleDecode, KQuantScaleMin, CodebookGather, GridLookup, TernaryDecode}` | **新增** | 从 g 派生（`DequantizeRowStreamFacts` + `GridDecodePlan` 在场/entryWidth）·**永不从格式名派生分派** | **替** `RVVToEmitCForwardElementwise.cpp:4875-4952` 的 decode_model 字符串 if-链 |
| `load_lmul` | `StringRef` | **复用** | `getRVVCodebookGatherAnchorLMUL`（`RVVGearboxSchedule.h:2713`）/ `chooseFillOptimalLMUL`（同文件 `:2798`·研究3） | 闭式·无 `"m1"` 硬编码字面量 |
| `widen_chain` | `WideningChain {l8,l16,l32,stripWidth,foldGroups}` | **复用 struct + derive** | `deriveWideningChain(base)`（`RVVToEmitCSupport.h:419`·struct `:411`·`stripWidth:415`/`foldGroups:416`·**已核实**） | 已是 q4_K/q6_K/FP4 核的 single-source-of-truth |
| `strip_lanes` | `int64` | **复用** | `getRVVStripVLMAXElements(lmul,sew,minVLEN)`（`RVVGearboxSchedule.h:2689`·**已核实**）；repack 核用 `deriveRepackHalfLanes`（研究3） | = `min(g.group_lanes, VLMAX)`·minVLEN<128 fail-close 到 0 |
| `legality` | `struct {bool isLegal; enum LegalityReason;}` | **复用逻辑** | whole-block cover ∧ gather span ∧ `rvvRegisterPressureLegal`（`RVVGearboxSchedule.h:2141-2185`） | **verifier 用同一 `getRVVStripVLMAXElements` 独立重算**（见六.1 末尾·已核实此模式在 verifier 在用） |
| `reason` | reason-trace（聚合 `RVVFillLMULReason` 等·研究3） | **新增聚合·复用成员** | 先例 = `RVVFillLMULChoice.reason` / `RVVNumericsTierChoice.reason`（`{value,reason}` 形·研究3） | 论文取的 reason trace |
| `provenanceFormat` | `StringRef` | **新增（仅诊断）** | `deqOp.getFormat()`/`decodeModel` | **[F-1] 声明式假阳性合规**：名作数据·**非**执行键 |

> **净新增字段 = `mechanism` / `reason` / `provenanceFormat` 三个**；`load_lmul`/`widen_chain`/`strip_lanes`/`legality` 全是既存闭式的薄封装（各自几何的 single-source-of-truth 已在）。

**verifier 独立重算已有先例（已核实·非新造模式）**：`getRVVStripVLMAXElements` **已被 Dialect verifier `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` include 并调用**（grep 命中·resolves 研究3 的「Dialect→Gearbox 是否合法」open flag = **合法且在用**）。`RVVBlockDotKernelDescriptor`（`RVVGearboxSchedule.h:2942-2972`·研究3）的 verifier 从**同一** VLMAX 公式独立重算 `gatherLegal`（`:2953`）= 提案 stage② 「verifier 可独立重算·fail-closed」要复用的正是此模式。

**MechanismPlan 住址（决策点·phase-2 kickoff 确认·非 phase-1 引入·非自决）**：
- 选项 A：struct 住 `include/Weft/Support/`（比照 `GridDecodePlan.h:21-23` 明写「因 verifier 与 emitter 都 link Support·Conversion→Dialect 是唯一合法方向」）；FormulaProvider 函数住 `RVVGearboxSchedule.h`（§〇 公式层家）。
- 选项 B：struct + 函数全住 `RVVGearboxSchedule.h`（该文件**已**被 verifier + emitter 双消费·保 §〇「唯一 closed-form-f 宿主」不变量）。
- **倾向 B**（一致于 §〇 + 已双消费实证）·但 struct 是数据类型非公式·按 `GridDecodePlan.h` 明规也可入 Support。**phase-1 不创建此 struct·故此决策安全延后到 phase-2**。

**emitter 消费 plan · 格式名降级的边界**：找 op 靠 **op-identity**（`dyn_cast<TypedDequantizeRowLoopBodyOp>`/`<DequantizeRowDecodeCoreOp>`·研究2·已解耦）；选执行叶靠 **plan.mechanism（5 类枚举）**（替 `:4875-4952` 的 decode_model 24 路字符串）；`provenanceFormat` 只进诊断注释·**不进任何分派/地址算术**。

---

### 六.2 · 5 阶段路线映射现状 + MIG-* 映射（phase-1 = 下一步）

| 外部 phase | 现状完成度 | 已落 MIG-* 映射 | 备注 |
|---|---|---|---|
| **① 统一 g（nibble descriptor·结构化·删重复格式名单·byte-exact）** | **PARTIAL** | **MIG-1**（律2 entryLanes→`codebook_entry_lanes` OptionalAttr·grid 3 格·证 descriptor-stamp→emitter-read 范式`06cc93fb3`）+ **MIG-1b**（ISSUE-119 census·分母=59·= 本 phase 的 inventory） | 已证「1 字段/3 grid 格」的贯通链；**nibble 家（q8_0/q4_0/q4_1/q5_0/q5_1）vertical slice = 下一步**（六.3） |
| **② DequantMechanismPlan（emitter 只读 plan·verifier 独立重算）** | **未起**（`MechanismPlan`/`DequantMechanismPlan` grep=0·supervisor 已核实事实④） | 公式**原语**已由 **MIG-B**（codebook-anchor 闭式`0f7556199`）+ **MIG-0**（reg-pressure 不等式`0e9faee0a`）铺好 | plan = 把这些原语聚合成 `{mechanism,load_lmul,widen_chain,strip_lanes,legality,reason}`·并让 verifier 独立重算 |
| **③ c 真驱动 θ（VLEN→LMUL/strip·measured 只在合法候选中选）** | **dequant-row 轴 = NONE**（census：dequant 路 0 处 VLEN 分叉/0 strip-LMUL θ·全 board-invariant） | **MIG-A**（vlenb→承重 VLEN 源`abb7e0304`）+ **MIG-C/C2**（measured-table q8/q4_0/q4_1·**repack 轴**`2ab1f9d4d`/`a230adc61`）+ **MIG-5**（selector 闭式 `θ_width=f(VLEN,dataWidth)`·**reduction/contraction 轴**·**在飞**）；**MIG-2**（grid coreLmul θ9-13→f）**BLOCKED** | ⚠所有已落 c-驱动 θ 在 **repack/block-dot/reduction 轴**·**非 dequant-row 轴**（诚实边界·研究2 stage4：block-dot 路比 dequant 路更 plan-driven）。**MIG-2 解阻依赖 = phase-② 给描述符加 `coreLmul` typed 字段**（plan.load_lmul 正是此字段） |
| **④ 扩 grid/ternary/KQuant** | **多数未起** | **MIG-1**（grid entry-lane·3 格）= 首付 | ternary/KQuant geometry 仍 L3 焊死（`:5366-5386` 19-format stride switch 等·census 候选） |
| **⑤ 外部可扩展 2 falsifier** | **未起** | — | 见六.4 |

> **phase-1 明确 = 下一个可派 task**（nibble 家 vertical slice·六.3）。它是 phase-① 的第二步（MIG-1 是第一步·grid entry-lane）。
> ⚠**MIG-5 并行文件不相交纪律**：MIG-5 在改 `RVVGearboxSchedule.h` + 选择器前门（`RVVReductionSourceFrontDoor.cpp`/`RVVContractionPathSelection.cpp`）·其 prd `:24` **明令禁碰** dequant 三文件（ForwardElementwise/GridCodebook/LowerQuantContraction）。**phase-1 触碰集（六.3）与 MIG-5 不相交**（phase-1 只碰 Construction + ForwardElementwise + verifier·**不碰 RVVGearboxSchedule.h**）⟹ 可并行。**phase-2 起需碰 RVVGearboxSchedule.h·必须与 MIG-5 串行**（共享文件·记忆律「并行线需文件集不相交」）。

---

### 六.3 · phase-1 具体 scope（nibble 家 vertical slice · 下一个可派 task）

**Scope 格式**：q8_0 / q4_0 / q4_1 / q5_0 / q5_1（QK=32·flat·非 grid）。**crux**：来源已统一（构造表一处·`RVVDequantizeRowConstruction.cpp:23-111`·**已核实**），但该表**只带 5 facts**（`{qk,stride,scaleByteOffset,quantByteOffset,codebookEntryLanes}`·`.h:53-59` 已核实）；**decode-mechanism facts**（`mOff/qhOff/sub/hasMin/hasQh/bareInt8`）**未入表**，在 emitter re-bake（构造 `.cpp:28-32` 注释自证「remaining offsets baked into per-format decode leaf at emit」·**已核实**）。→ 一个常量两处源 = 编辑分家即 silent byte-exact break。

**触碰集**（file:line·**已核实**标 ✓·**研究 4 已读我未复核**标 ⚠·phase-1 首步须机核自验·照 MIG-1/MIG-5 范式「PRD 锚点可能 naive-grep 误判·按实际重路由」）：

| # | 动作 | file:line | 核实 |
|---|---|---|---|
| A | 扩构造表 5-format 臂 + `DequantizeRowStreamFacts` struct·加 `min_byte_offset/min_present/qh_byte_offset/qh_present/nibble_bias/carrier_kind/fold_kind`（superset）·stamp 到 decode_core | 表 `:41-50` ✓·struct `.h:53-59` ✓·stamp `:133-163` ✓ | ✓ |
| B | emitter α（monolith fallback switch·全 5 格全 tuple·唯一携 bareInt8/sub/hasMin/hasQh 处） | `RVVToEmitCForwardElementwise.cpp:2658-2674` | ⚠ |
| C | emitter β（owned real-vector 叶·**DEPLOYED** 路·re-bake 字面量为 `emitDequantizeRowNibbleVectorBody` call args） | `:3209-3244`（q8_0 分离 `:3387`） | ⚠ |
| D | emitter γ（scalar shared 叶·re-bake 同字面量） | `:3500-3543`（q8_0 `:3254`） | ⚠ |
| E | dispatch δ（decode_model 字符串→叶·nibble）→ 改键 `carrier_kind` | `:4875-4884`（q8_0 `:4948`）✓ | ✓ |
| F | header per-format 方法声明 | `RVVToEmitCInternal.h:4925-4963` | ⚠ |
| G | verifier 白名单（**留作名门·第二步从表 key-set 派生·律1 分步**） | `isWiredDequantizeRowFormat:10902-10919` / `isConstructedDequantizeRowDecodeModel:11006-11020` | ⚠ |

> **不碰**（byte-exact 的支点）：两个 8-param sink 共享体 `emitDequantizeRowNibbleVectorBody`（`:2966`⚠）/ `…NibbleBodyShared`（`:2702`⚠）**phase-1 不动**。phase-1 只改「8-tuple 从哪来」（一处 descriptor 读 vs 3 处硬编码 call site）。
> **不入 phase-1**：extended `enum Fmt` StringSwitch（`:5023-5045`/stride switch `:5366-5386`）= K-quant/IQ/codebook 家 = **phase-④**；`GgmlQuantContractionOp::verify` 的 per-format numeric switch（研究4 `:2172-2303`）= **vec_dot/contraction 家·非 dequant-row**（记忆：dequant≠vec_dot·iq2_xs vec_dot 是 ISSUE-120 byte-broken）·**严禁 fold-in**。

**结构化 nibble 描述符字段**（研究4·全是 ggml ABI-shape 事实·同 `codebook_entry_lanes` 类别·律2 合规）：
`name`（仅 provenance·无分派权）· `qk(=32)` · `weight_block_stride` · `scale_byte_offset(dOff)` · `quant_byte_offset(qsOff)` · `min_byte_offset(mOff)+min_present(hasMin)` · `qh_byte_offset(qhOff)+qh_present(hasQh)` · `nibble_bias(sub)` · `carrier_kind∈{bare_int8,nibble4}` · `fold_kind∈{single_mul,fused_mac_min}`（**DERIVED from hasMin·不裸存**·选 `vfmul_vf` vs `vfmv+vfmacc_vf`）。
> **律2 合规注**：`hasMin/hasQh` 只存为 offset-present 位·fold 形状与 `vfmul/vfmacc` 的**机制体实现留在 emitter**（不入源）。

**per-format ground truth**（研究4 表·⚠ `stride/dOff/qsOff` 我已从构造表复核一致：q8_0 34/0/2·q4_0 18/0/2·q4_1 20/0/4·q5_0 22/0/6·q5_1 24/0/8；`mOff/qhOff/sub/hasMin/hasQh` 研究4 从 emitter 读·**phase-1 须从 loci B 复核后再中心化**）：

| fmt | stride | dOff | mOff | qhOff | qsOff | sub | hasMin | hasQh | carrier |
|-----|--------|------|------|-------|-------|-----|--------|-------|---------|
| q8_0 | 34 | 0 | — | — | 2 | 0 | F | F | bare_int8 |
| q4_0 | 18 | 0 | 0 | 0 | 2 | 8 | F | F | nibble4 |
| q4_1 | 20 | 0 | 2 | 0 | 4 | 0 | T | F | nibble4 |
| q5_0 | 22 | 0 | 0 | 2 | 6 | 16 | F | T | nibble4 |
| q5_1 | 24 | 0 | 2 | 4 | 8 | 0 | T | T | nibble4 |

**byte-exact 策略（by-construction 可证）**：共享体不动 ⟹ 若 descriptor 产出与叶子现硬编码**逐字节相同的 8-tuple**·emitted C **by construction byte-identical**。验证 = 5 格各跑「constructed 前门路 + monolith fallback 路」过 `--weft-rvv-lower-to-emitc`·diff emitted C vs 现役 lit golden = **0-byte**。三闸 ZERO-MODEL + 3-arm anti-hollow + CORPUS（§五）·**独立复核跑判决不看 diff**。
**5 个 byte-exact 风险点**（研究4·须逐点 assert）：① fold 形状 `y=val*d`(q4_0/q5_0/q8_0) vs `y=val*d+m`(q4_1/q5_1)·不同 C + 不同 fp 舍入·② bias `-=sub` 精确 −8/−16 · ③ 5th-bit qh merge gated on hasQh · ④ byte offset 喂地址算术（错字面量读错字节·shape-diff 看不出·**须 assert stamped-attr==descriptor value**）· ⑤ q8_0 signed `vsext` vs nibble `vzext`+bias（**q8_0 留独立叶·禁 fold 进 nibble 体**）。

**律1 分步（一次一变化）**：(i) 扩表+struct+stamp·β/γ 改读 stamped attr·证 0-byte golden diff → (ii) 塌 δ dispatch 到 `carrier_kind`（一 nibble4 支 + 一 bare_int8 支）→ (iii) verifier 白名单从表 key-set 派生 + 落 F1 falsifier。**每步独立 byte-exact + 独立复核**。

---

### 六.4 · C1/C2 两 falsifier（各带验收命令 + LOC 记录法）

**F1 —— 合成格式 = descriptor 行 + test only·emitter/verifier-机制零改（phase-1 交付·= phase-1 验收门本身）**
- 合成 `q4_synth`（4-bit nibble·distinct-but-legal tuple·如 stride=19/dOff=0/qsOff=3/sub=8/hasMin=F/hasQh=F·q4_0-shaped 偏移移位）。加它应**只**碰：① 构造表一行·②（若白名单表-派生）verifier **零**行·③ 一条 lit（`weft_rvv.dequantize_row {format="q4_synth"}` + FileCheck）。
- **双跑对照 = C1/C2 度量**：
  - **before phase-1**：加 q4_synth 需编辑 α+β+γ+δ+header = **5 处 emitter 改** → F1 **FAIL**（证今日耦合）。
  - **after phase-1**：descriptor 行 + test·dispatch 命中共享 `nibble4` 支无新 if → F1 **PASS**（证来源解耦）。
  - **记录法**：两跑各记 `git diff --stat` 触碰文件集 + LOC delta（before ≈ 5 文件 / after = Construction.cpp + test 共 2 文件·0 emitter/verifier-机制行）= 论文 C1/C2 数字。
- **无真 ggml block 的 oracle（name-relative）**：assert「descriptor tuple == q4_0's 的合成格式·emit C 与 q4_0 **逐字节相同**（除 provenance 注释 token）」·且「格式名在可执行 C 中**只作注释出现**」= 直测「格式名不再承担分派权」·**零手写 oracle**。
- **负控（防橡皮图章）**：非法组合（`bareInt8 && hasMin`·或 stride < layout 最小）**须 fail verify closed**·证 legality 门是真的。
- **验收命令**：`ninja weft-opt && llvm-lit -v test/Conversion/RVV/rvv-dequantize-row-q4-synth-descriptor-only.mlir`·且 `git diff --stat` 仅 2 文件。

**F2 —— 独立 family = 插件领地 + 注册表·core 零家族分支（phase-⑤·非 phase-1）**
- 新 backend/family 只改插件领地 + 注册表·`git grep` core 家族分支 = **零新增**。
- **记录法**：触碰文件集须 ⊆ `lib/Plugin/<family>/` + 注册表·`lib/Dialect`/`lib/Conversion` core 零家族名分支新增（LOC delta 记于交付）。
- **noted·out of phase-1 scope**（研究4·属 phase-⑤）。

---

### 六.5 · 风险与纪律

1. **byte-exact 先于一切**：phase-1 的 by-construction 支点 = 共享 8-param 体不动·只迁 8-tuple 来源。任一叶子若「shape 相同但读错字节」= shape-diff 看不出 → **必须 assert stamped-attr == descriptor value**（风险点④）。
2. **不一次性重写 ForwardElementwise**：律1 分三步（表→dispatch→白名单）·每步独立 byte-exact + 独立复核。**禁**把 α/β/γ/δ 一把梭。
3. **律2（机制体点不入源）**：只中心化 ggml ABI-shape 事实（offset/stride/bias/present-bit）；`vfmul/vfmacc` 机制体、5th-bit merge 实现**留 emitter**。加描述符字段**不得改各格现有强/弱义 provenance 状态**（[L-8]·原语 ID 清单不变）。
4. **[K-10] 硬约束（研究1）**：5 mechanism 因数据消费契约/迭代拓扑互异 = **结构级**·必须 5 个分立 Plan/typed region·**禁**收进单一 plan 用 `mechanism` 判别式在 emitter switch（那是 GEMM/GEMV 判例所禁的「结构级当旋钮」）。FormulaProvider 的职责 = **选哪个结构级 Plan**·非 plan 内切 mechanism。**只有** `{load_lmul,strip_lanes,unroll,lane-width}` 是参数级 capability 键（phase-③）。**注**：phase-1 的 `carrier_kind`(bare_int8/nibble4) 选的是**两个已分立的叶**（q8_0 独立叶 vs nibble 共享体）·非 plan 内 switch·合规；`fold_kind` 的 [K-10] 结构/参数 定性是**既存共享体状态**（phase-1 不动不重裁）。
5. **measured 住测量库·不焊公式常量**（[S-5]·phase-③）：「合法候选枚举」=公式（capability→合法 θ 集）；「候选中选哪个」=测量库标定事实（如 `kRepackMeasuredM1FasterMeasurements`）。两者分家·**不烧进 schema.def 也不当 FormulaProvider 硬编码常量**。
6. **reason/plan = mirror·非 authority**（I4·插件协议.md:105）：论文的 reason trace 只能是 transient 结果或 route 后 `weft.exec.diagnostic{reason="emission_plan"}` 镜像·**禁**持久成 readiness 状态机/进度工件。
7. **canon 措辞变更须入 ISSUES 待裁（非自决）**：
   - **phase-② verifier 独立重算**从「bounds-check + 白名单」升级为「从公式重算几何并拒绝 stamped≠recomputed」= **改 dequant 家 verifier 行为类**·触及 [D-1]「unknown=reject」门的 fail-closed 措辞 → **落地前登记 ISSUE / 报 supervisor·非静默强化**。
   - **构造表头注释已 stale**（`.h:61-69` 说「21 CONSTRUCTED·nullopt for tq1_0/tq2_0」·但 `.cpp:97-104` 实已构造 tq1_0/tq2_0·今 = **24 格全构造**）= 文档漂移·**记为 cleanup 观察**（非 canon 变更·可随 phase-1 顺手订正或单开 doc-fix）。
   - MechanismPlan 住址（Support vs RVVGearboxSchedule.h·六.1）= 触及 §〇「唯一 closed-form-f 宿主」不变量措辞 → **phase-2 kickoff 报 supervisor 确认·非自决**。
8. **并行纪律**：phase-1 触碰集与在飞 MIG-5 不相交（phase-1 不碰 RVVGearboxSchedule.h）⟹ 可并行；**phase-2 起碰 RVVGearboxSchedule.h·必与 MIG-5 串行**。
9. 🔴 **禁内联汇编 / 禁钉死调度绕 clang**。

---

### 六.6 · 诚实边界（研究主张 已核实 / 待验）

**已核实（supervisor 主会话直读·本节标 ✓ 的 file:line）**：
- 真 realization hook = `materializeSelectedLoweringBoundary`/`validateSelectedLoweringBoundary`（`ExtensionPlugin.h:684-688,745-749`）·**订正**外部建议的 `realizeSelectedVariantBody` 通用名。
- 三 typed op 链真在（`RVVOps.td:9549/9600/9683/9703`）。
- 构造表 = 单一 24-arm format lookup（`RVVDequantizeRowConstruction.cpp:23-111`）·**只 stamp 5 facts**·nibble decode-mechanism facts 未入表（构造 `.cpp:28-32` 注释自证）。
- `DequantizeRowStreamFacts` 5 字段（`.h:53-59`）；`codebook_entry_lanes` 条件 stamp（`.cpp:160-163`）。
- decode_model 字符串 dispatch（`ForwardElementwise.cpp:4875-4952`·nibble `:4875-4884`·q8_0 `:4948`）。
- `WideningChain` struct + `deriveWideningChain`（`RVVToEmitCSupport.h:411/419`）。
- 公式原语 `getRVVStripVLMAXElements:2689`/`getRVVCodebookGatherAnchorLMUL:2713`/`rvvRegisterPressure*:2141-2185`（`RVVGearboxSchedule.h`）。
- **研究3 open flag RESOLVED**：`getRVVStripVLMAXElements` **已被 Dialect verifier `RVVDialectWideningOps.cpp` include+调用** ⟹ Dialect→Gearbox 独立重算合法且**已在用**（verifier 独立重算是既存模式非新造）。
- `GridDecodePlan.h` = Support 家·dual-consumer·fail-closed registry（`unknown→nullptr→REJECT`·[D-1]）= plan siting/lifecycle 的结构模板。
- MIG-5 = 真在飞并行 A 线 task（selector 闭式 `θ_width=f(VLEN,dataWidth)`·`.trellis/tasks/07-20-07-20-r51h-mig5-widthsel-closedform/prd.md`）·明令禁碰 dequant 三文件。
- `MechanismPlan`/`DequantMechanismPlan` 类 grep=0（supervisor 已核实事实④）= 全新抽象。

**仍待验/存疑（phase-1 首步须机核自验·标 ⚠ 者 = 研究 4/2 已读·我未逐一复核·PRD 锚点可能 naive-grep 误判）**：
- emitter loci **α `:2658-2674` / β `:3209-3244` / γ `:3500-3543`** + header decls `:4925-4963` + 共享体 `:2966`/`:2702`（研究4 line-number·phase-1 须按实际重路由·照 MIG-1/MIG-5 范式在交付标「PRD 锚点误判·实际在 X」）。
- per-format `mOff/qhOff/sub/hasMin/hasQh` tuple（研究4 从 emitter 读·`stride/dOff/qsOff` 我已复核一致·**decode-mechanism 5 列 phase-1 须从 loci α 复核后再中心化**）。
- verifier「今日只 bounds-check + 白名单·不重算几何」（研究2 读 `RVVDialectWideningOps.cpp:11131-11217`）·⟹「verifier 可独立重算」是 **phase-② 新能力·今日不在**（研究2 明判）。
- **诚实边界（研究2 stage4·非 dequant）**：block-dot/vec_dot 路（`RVVToEmitCKQuant.cpp` 的 `emitIQ*SuperBlockGridBody`）**已**读 typed brick attr（`getIntegerCoreLmul().value_or("mf2")` 等·带 value_or 焊死默认）·比 dequant-row 路**更 plan-driven 一档**。dequant-row 落后 = geometry typed 化了但 emit 不读（re-bake）·LMUL 从固定 geometry 派生而非从 c。**⟹ 本升级 = 让 dequant-row 追平 block-dot 的 plan-driven 度·并把两者的 value_or 焊死默认都升为真 c-驱动**（MIG-2 解阻同此依赖）。
