# X-SCALAR/zvfh 家族#3 接入 plan + [RENAME+AUDIT] 后续排期（合并 · 立项裁决材料）

> 快照 HEAD = `9f4a15f3` · 生成 2026-07-11 · 触碰集 = **docs 域**（只读全仓 + 新建本报告）。
> **纯分析 · 不实现 · 不动 lib/schema/code/lit/ROADMAP/既有 casefile。** 排期是**建议非承诺**；
> C2 现状**诚实标注 1/≥3**；X-SCALAR 接入成本**以实测标注**（landed 部分已量、剩余以预估标注）。
> **本报告 = 立项裁决材料**：X-SCALAR 实际立项 = 用户裁（家族#3 = 新战役方向，虽 ROADMAP 队列内但规模需报排期）。

---

## 甲、X-SCALAR/zvfh 家族#3 接入 plan（C2 经济学第三点 · 叙事刚需）

### 甲.0 ★头号 scout 发现（改变本 plan 的前提）：X-SCALAR **已大面积落地**，canon 状态栏 stale

任务预设「类比 G4 IME 的 M0-M3 从零接入五件套」。**scout 现状证否此预设**——X-SCALAR 标量家族的
**两个 owned 内核 + 曳光弹 + F-6 双断言机检均已 landed 在当前树**（HEAD `9f4a15f3`），但
**canon 现状栏（执行总纲 §6/§107、FALSIFIER-INDEX F-6 行）仍写「空 STUB / owned 内核未落地 / gated on X-SCALAR 家族落地」= 文档-代码不同步债**。

**已落地实证（当前树 grep + 读码确认）：**

| 件 | 落点 | 状态 | commit |
|---|---|---|---|
| 插件五件套（部分） | `lib/Plugin/Scalar/{ScalarExtensionPlugin,ScalarEmitCRouteProvider,ScalarBackendEmissionDriver}.cpp` + `include/.../Plugin/Scalar/*.h` | 在位 | 多次 |
| Dialect + ODS | `include/.../Dialect/Scalar/IR/ScalarOps.td`（3 op）+ `lib/Dialect/Scalar/IR/ScalarDialect.cpp` | 在位 | `5c010b2b`/`f96f767a`/`2dd654d8` |
| 曳光弹 op | `tcrv_scalar.compute_skeleton`（trivial `int32 v=7`） | 落地 | `5c010b2b`（线D-s2 曳光弹：parse→verify→lower→byte-exact 纯标量 C） |
| **主选 owned 内核** | `tcrv_scalar.tq2_0_q8_k_vec_dot`（三值 2-bit vec_dot，base-I 位域抽取，int8×int8 MAC，**NO XOR-popcount**，纯标量 C，per-super-block fp16×fp32 scale fold） | **落地 + byte-exact golden** | `f96f767a`（真三值核替换曳光弹平凡 op） |
| **保底 owned 内核** | `tcrv_scalar.dequantize_row_q4_0`（4-bit nibble 逐元素 dequant，字节精确近乎 by-construction） | **落地 + byte-exact golden** | `2dd654d8` |
| F-6 双断言机检 | `test/Transforms/VariantSelection/f6-independent-scalar-family-emittable.mlir` + `test/Plugin/ScalarExtensionPluginTest.cpp`（989 LOC）+ `test/Target/Scalar/*.golden.c` | 落地 | `2dd654d8` |
| 构造 note | `docs/method/x-scalar-ternary-vec-dot-construction.md`（journal，权威框在执行总纲 §6） | 在位 | `f96f767a` |

**含义**：X-SCALAR **不是**「从零接入」战役，而是「**收口 + 正名 + 补 4 个 open 项**」战役。里程碑草案（甲.4）据此
从「M0-M3 建家族」reframe 为「补齐 open boundary + 登记 C2 ledger 点」。**这是本 plan 最重要的规模修正**。

---

### 甲.1 [X-1] 顺序（canon 权威 + 实际工作序偏离）

**canon [X-1]（`科研目标总纲v2:196`）：`Zvfh（含闭包修复）→ [X-SCALAR] → [硅片核查] → AME（条件）`。**
- **[X-ZVFH] = 第一优先**（`科研目标总纲v2:186`）：向量族**子扩展**，落点 C3′-P5 + C2 子扩展粒度点（<100 行）；**兼作 [S-2] implies 传递闭包探针**（闭包修复同一里程碑验收）；f16 路径实测。
- **[X-SCALAR] = 第二优先**（`科研目标总纲v2:187`）：**家族#3（independent）**，落点 C1 广度半边 + C2 第二点（<300 行）+ C3′ 标量语境族内键控变体。

**★粒度辨析（本 plan 关键，避免混为一谈）**：
- **X-SCALAR = 独立 extension family（家族#3）** —— 与 IME 并列的**新家族**，闭包 ∩ rvv.* = ∅，向量缺席。C2 里是**家族级**数据点（<300 目标）。
- **zvfh = 向量族子扩展（NOT 家族#3）** —— integrated 到 RVV 家族的**子事实/能力门**。C2 里是**子扩展粒度**数据点（<100 目标），不是新家族。
- 二者是**不同 KIND 的 C2 点**：家族级 vs 子扩展级。任务标题「X-SCALAR/zvfh 家族#3」应理解为「X-SCALAR = 家族#3；zvfh = 与之相邻的子扩展粒度点」，**不是** zvfh 也是家族#3。

**★实际工作序偏离 [X-1]（诚实披露）**：代码落地序 = **X-SCALAR owned 内核先于 zvfh**（X-SCALAR 于 `5c010b2b`→`2dd654d8` 落地，zvfh `rvv.zvfh` 事实**至今未注册**，`test/Dialect/Exec/verify.mlir` 仍判 "unknown capability" 拒 = 恰是所需 fail-closed 预实现态）。原因：X-SCALAR compute 是**测量总攻期「线D」机会窗**顺手推进（见 `travel-decision-ledger.md` 线D）。**建议**：剩余**收口工作**重新对齐 [X-1] —— 先做 zvfh 事实注册 + 闭包修复（第一优先 + 兼 [S-2] 探针，建立 fp16-fact 注册范式），再做 X-SCALAR 的 `scalar.zfh` 注册（复用同一 schema fact-model 机器，便宜）。**注**：`rvv.zvfh`（向量）≠ `scalar.zfh`（标量）是两个不同 fact，但共享 [S-1] 事实模型机器，同批做省一次。

---

### 甲.2 X-SCALAR / zvfh 是什么（能力事实定义 + scout 现状）

**X-SCALAR = 标量家族（向量缺席）** = 「**能力门家族**」（[S-8] `科研目标总纲v2:65`）：其 owned 内核主体用**基础标量指令**，
家族事实充当**门 + 可选加速子事实**（`zbb/zba/zbs` 之于标量家族 ≡ `zvfh` 之于向量家族，子事实粒度对称）。
- **能力事实**：`scalar.fallback`（现 own）+ 计划中的 `scalar.zbb/zba/zbs`（子事实/能力门，grep=0 未建）+ `scalar.zfh`（fp16 scale fold 前置声明，未建）。
- **owned 内核**（canon 指定，均已落地）：主选 = 三值 2-bit `vec_dot`（A 类热内核，最强判据①）；保底 = 4-bit `dequantize_row`（辅助算子，较弱）。
- **⚠ popcount 陷阱**（[J-3]/canon）：低比特数学**不走 XOR-popcount**（位域抽取/base-3 拆包/查表 gather）；已落地内核**代码级 CONFIRM**（`--implicit-check-not="popcount"`；全仓 `popcount` grep=0）。

**zvfh = Zvfh(+Zvfhmin) 向量族子扩展** = 向量半精 f16。落点 C3′-P5（精度路径模式）+ C2 子扩展粒度点。
- **能力事实**：`rvv.zvfh`（`kind=sub_ext`），现**未注册**；注册后 implies 闭包应含 `rvv.v`。
- 现状：仅 march 子串检测（`RVVCapabilityProfile.cpp:112`），无一等事实；`verify.mlir:58-59` 判 unknown 拒。
- **兼 [S-2] 闭包探针**：注册 zvfh 顺带验「加载期 implies 传递闭包」（现查一层，[B-7] 已核字节中性）。

---

### 甲.3 N2 family entry boundary 核验（memory `n2-family-entry-boundary` 判据）

判据 = **能力可表达为 RISC-V capability 事实 + 被同一 core/common pass 零-core-branch 消费**（复用 N1 能力模型本身，非复用 pipeline 形状）；独立离散卡（GPU/TPU/910B）明确排除。

| 候选 | RISC-V capability 事实? | 零-core-branch 消费? | 判定 |
|---|---|---|---|
| **X-SCALAR 标量家族** | ✅ `scalar.*` 事实（能力门家族 [S-8]）；向量缺席实例 = 合法能力事实 | ✅ 目标态：`ScalarExtensionPlugin` own 能力，前门/发射零 core 分支（F-1）；**判据④「向量缺席→标量变体 only_feasible 真实选中」= 尚未 wire**（唯一未闭 boundary） | **PASS（Case-A）**，与 IME 同级合法家族；剩判据④连线 |
| **zvfh 子扩展** | ✅ `rvv.zvfh`（integrated 子事实，非独立家族）；闭包**含** rvv.* 是**应该的**（它就是向量族子扩展，不主张 F-6 independence） | ✅ P5 精度路径模式，能力键控 | **PASS（integrated 子扩展）**，非新家族、不走 F-6 |
| （对照）AME/玄铁独立矩阵 | 条件（integrated-attached，挂 RV 核）| — | 条件项（Full 档复活开关，硅片可及性 gated，论文不押注） |
| （对照）独立离散卡 | ✗ 自带 ISA/runtime，PCIe/offload | ✗ | **拒（Case-B）** |

**F-6 独立性走查（X-SCALAR，执行总纲 §6 + 已落地内核）**：每原语（2-bit 抽取 / `xi-1` / 整数 MAC）非向量 → 闭包 ∩ {rvv.*} = ∅ **成立**。
唯一风险点 = **fp16 scale fold**，三路裁决明确（执行总纲 §6）：
- **`scalar.zfh`（标量半精）**：声明为标量事实 → 闭包 ∩ rvv.* = ∅，**独立性保持**（需注册，未建）；
- **`rvv.zvfh`（向量半精）**：会把 rvv.* 拉进闭包 → **破坏 F-6，禁止**；已落地内核用源级 `_Float16` deref、**从不发 zvfh 向量路**（构造 note §fp16 已确认）；
- **软件 fp16→fp32**：不需新能力，最干净独立性叙事。
→ 已落地内核选了「源级 `_Float16`（下游 C 编译器决定走 zfh 硬件或 `__extendhfsf2` libcall）」= 独立性成立，但 `scalar.zfh` **事实未注册** = §6 前置工作项未做。

---

### 甲.4 接入 plan（reframe：非从零 M0-M3，而是「收口 open boundary + 登记 ledger」）

因甲.0 大面积已落地，里程碑草案不照搬 G4 IME 的 M0-M3（那是从零建家族）。X-SCALAR 剩余 = **4 个 open boundary**
（构造 note「Still open」+ 执行总纲 §6 前置项）+ **canon 正名**。按依赖排序：

| 里程碑 | 内容 | 触碰域 | 依赖 | 规模预估 |
|---|---|---|---|---|
| **XS-M0（canon 正名 · docs）** | 执行总纲 §6/§107 + FALSIFIER-INDEX F-6 行「空 STUB」→ 更新为「owned 内核已落地、剩 4 open」；本报告可作证据指针 | docs（canon 级措辞 → **必问用户**） | 无 | 小（措辞） |
| **XS-M1（zvfh 事实注册 + 闭包修复）** | [X-1] 第一优先：注册 `rvv.zvfh` 事实（`kind=sub_ext`, implies `rvv.v`）+ [S-2] 传递闭包 + `verify.mlir` unknown→known 翻转 + P5 f16 路径 lit | schema fact + lib（RVVCapabilityProfile）+ lit | 无（不需硬件做注册；f16 实测需 zfh 板） | 中（子扩展 <100 目标） |
| **XS-M2（scalar.zfh 注册）** | 复用 XS-M1 fact-model 机器，注册 `scalar.zfh`；scalar 内核 fp16 fold 从源级 `_Float16` 提升为显式能力事实（独立性保持，闭包只排 rvv.*） | schema fact + Scalar plugin | XS-M1（复用范式） | 小（参数复用） |
| **XS-M3（判据④ 连线 · N2 boundary 最后一环）** | **向量缺席实例使标量变体 `only_feasible` 且真实被选中** —— 在 `VariantSelection` + Scalar 合法性谓词接入；F-6 第二半（闭包脚本化 + 真实选中）落 CI | lib（VariantSelection + Scalar VariantLegality）+ F-6 CI 脚本 | 无（可与 XS-M1 并行） | 中（[F-6] 独立性判据最后一环，C1 广度半边真证） |
| **XS-M4（C2 ledger 第二点登记 · LED-2）** | Scalar 加进 `schema/family-dirs.v1.json` + 跑 `family_ledger.py --family Scalar` + LED-2 出图（家族≥3）；口径与 <300 目标对账（见甲.5） | schema/manifest + docs | 上述落地后 | 小（脚本 + 登记） |

**五件套完整性核验（[P-2]）**：①事实+关系表行（`scalar.fallback` 在，`zbb/zba/zbs/zfh` 子事实缺 = XS-M2 补）②合法性谓词（部分，判据④ = XS-M3）③发射模式（✅ 两内核 emitter 在）④测试（✅ lit byte-exact golden 在；每板 objdump golden = pending-hardware）⑤ledger 条目（缺 = XS-M4）。**缺一不收 → 现缺 ①部分/②部分/⑤，即 XS-M2/M3/M4。**

**曳光弹铁律核验**：G4 IME 的「贯通前禁铺格」在 X-SCALAR 已**天然满足**——曳光弹（`compute_skeleton`）→ 真内核（`tq2_0`）贯通已完成，两 owned 内核已落地，不存在「未贯通先铺格」风险。

---

### 甲.5 C2 经济学 ledger 预期（★实测已可给，且暴露 <300 目标的诚实缺口）

**对照锚（IME 家族#2）**：`experiments/active/visibility/T2-ledger-anchor.md` 锚 = **raw wc-l 2484**（founding slice，4 code dirs 排除 CMakeLists）/ cloc-approx ~1866。
**⚠ 锚 drift 提示**：`family_ledger.py --family IME` **现报 raw 5153 / cloc-approx 3858**（IME 家族已长到 6 op = **扩展**，非首次接入）。
C2 边际曲线的正确锚 = **首次接入 slice（2484）**，后续 op 是家族内扩展（属 C3′ 构造经济学，非 C2 新家族成本）。引用时须钉口径「首接入 slice raw 2484」。

**X-SCALAR 实测（本报告手量，raw wc-l，同 IME 口径：code dirs 排除 tests + CMakeLists）**：

```
lib/Plugin/Scalar/{3 cpp}  +  include/.../Plugin/Scalar/{3 h}  +
include/.../Dialect/Scalar/IR/{ScalarOps.td, ScalarDialect.h}  +  lib/Dialect/Scalar/IR/ScalarDialect.cpp
= raw wc-l 1501  ·  cloc-approx（去空行+纯注释行）≈ 1131
（大头 = ScalarBackendEmissionDriver.cpp 878 raw / 644 cloc-approx = 两内核纯 C emitter）
test_LOC（单列，不并入）≈ 1156（ScalarExtensionPluginTest.cpp 989 + golden/lit）
```

**★诚实缺口（本报告最锐利的 C2 发现）**：canon [C2-1] 目标 = **标量家族 <300 行**；**实测 landed ≈ 1131 cloc-approx / 1501 raw = 目标的 ~4-5×**。
两点解读：
1. **方向对、幅度不对**：家族#3（1501 raw）< 家族#2 首接入（2484 raw）→ **边际递减方向成立**（~60%），但 **<300 的具体量级 NOT met**，「1/10 of #2」的审稿人预答（[C2-5]）当前**不成立**。
2. **结构根因 = F-6 独立性禁止复用，抬高了独立家族的接入成本**：X-SCALAR 主体是**净新 pure-C emitter**（878 行 driver），**不能复用 RVV EmitC 发射机器**（复用会把 rvv.* 拉进闭包、破坏 F-6）。这与「integrated 子扩展（zvfh <100）复用宿主家族机器 = 便宜」形成对比。
   → **C2 曲线的正确刻画不是「单调随家族序号递减」，而是「integrated 子扩展廉价 / independent 家族付真 emitter 成本」（[L-2] integrated vs independent-attached 术语的经济学体现）**。这是比原 <300 直觉**更精确、更可辩护**的 C2 主张。

**建议（供裁）**：XS-M4 登记时，(a) 用实测 1501 raw / 1131 cloc-approx 作 X-SCALAR 数据点（非 <300 估值）；(b) canon [C2-1] 的「<300」**要么修订为实测量级 + 上述 integrated/independent 结构解释**（canon 级 = **必问**），要么明确 <300 仅指「单个最小 owned 内核的增量」而当前 landed 含 2 内核 + 曳光弹 + 重测试（口径重定义）。**不预设选哪个，报裁。**

**C2 曲线现状（诚实，[NG-4]）**：即便计入 landed X-SCALAR，曲线 = **2 个真数据点**（IME 2484 + Scalar 1501），仍 **< 3 点**（zvfh 子扩展点未落）。ROADMAP/canon 的「**1/≥3 曲线缺失**」标注**继续保留**——landed X-SCALAR 把「1 点」变「~2 点」但未成曲线；zvfh 落地才补第 3 点（且是子扩展粒度）。

---

### 甲.6 触发条件 / 就绪度（[X-0] 三问 + 立项性质）

**[X-0] 接入三问**（任何扩展先过）：
1. **服务哪条贡献？** X-SCALAR → C1 广度半边（向量缺席的独立家族，证 schema 跨 independent family 复用）+ C2 第二/三点（模板经济学叙事刚需）+ C3′ 标量语境键控变体。**答得出。**
2. **有无负载/曲线价值？** 曲线价值 = C2 第三个异质家族接入多顺（模板故事最强证据）；负载价值 = 三值 2-bit 是 A 类热内核。**有。**
3. **成本进 ledger 且预算被接受？** landed 已花 ~1501 raw；剩余 4 open 项预算见甲.4（中小规模）。**需用户接受剩余预算。**

**就绪度（无硬前置阻塞）**：
- **硬件**：emit-golden 路**不需新硬件**（纯 lit/golden 已在）；runtime bit-exact = pending-hardware（标量核任何 RV 板可跑，非阻塞 emit 轴）；zvfh f16 **实测**需 zfh 板（`ssh k1`/`ssh rvv` 均有 zfh，不阻塞）。
- **canon 前置**：无红线阻塞；XS-M0 canon 正名 + 甲.5 的 <300 口径裁决 = canon 级**必问**项，但不阻塞代码收口。
- **[X-1] 满足**：canon 序 zvfh 先，但 X-SCALAR compute 已先落（偏离已披露）；剩余收口建议重排回 [X-1]（zvfh 事实注册先，甲.1）。

**★立项性质（关键判断，供裁）**：X-SCALAR **不是**「启动大新战役」，而是「**收口一个已 70-80% 落地的家族**」。按决策权限卡，全新战役 = 必问；但此处大部分已在既有工作树上落地，剩余 = 有界收口（4 个 open boundary + 正名）。**建议裁决点**：(i) 是否现在立「X-SCALAR 收口」为一条主线任务（补 XS-M1..M4）；(ii) 还是挂「工程债/被动维护」待 zvfh（[X-1] 第一优先）先行。**本报告不预设，报排期供裁**（见丙）。

---

## 乙、[RENAME+AUDIT] 后续排期

> 背景：AUDIT（`docs/reports/2026-07-11-TEMPLATE-AUDIT-structure.md`，快照 `b3e3fef4`）+ docs-先行（`f468cf6c`）已落 **R1**（五大件→目录映射 `docs/method/REPOSITORY-MAP-五大件.md`）+ **R4**（`docs/method/FALSIFIER-INDEX.md`）+ **R6 部分**（承重证据注册非删）。剩余 = C5-C11 删除 + R2/R3 lib 移动 + check_docs_canon 2 RED。**本节复原指针基准 = git `b3e3fef4`**（AUDIT 快照；`git show b3e3fef4:<path>` 可复原）。

### 乙.1 C5-C11 删除项（逐项 · 复原指针 `b3e3fef4` · 已核当前树实存）

> **承重证据已注册非删（明确）**：C2（vl16_static_account.md 被 SEALED-WIN-REGISTRY 引）· C4（T-VALIDITY* / T-PERF1b 4 表 = [CASE-COMPILER-ASYMMETRY] 承重）· C6（T3_A csv 内 INVALID/WITHDRAWN 隔离行）= **注册/原地保留，不删**，已由 docs-先行 R6 处理。以下 C5-C11 = 可清理/需评估项。

| # | 路径（当前树已核） | 为何废 | 风险 | 建议 | 复原指针 |
|---|---|---|---|---|---|
| **C5** | `experiments/archive/perf-historical/{ondevice-q5_K, ondevice-q8_0*, T3_step3}` + **5 个 tracked `.o` 二进制** | 换板前 STALE perf 数（不可比）+ 二进制入库膨胀 | 低（已归档 + MOVES.md）；.o 移出版本控制**不可逆** → 需确认已在 RETIRED-INDEX/MOVES ledger | **保留归档**；`.o` 二进制**评估**移出 VC（过退役闸 + 确认 MOVES 登记）→ 移 = 必问（删 tracked 二进制不可逆） | `b3e3fef4` + `MOVES.md` |
| **C7** | `artifacts/{grill-consensus-20260515, grill-rvv-maturity-ladder-2026051{8,9}, tianchenrv_rvv_gearbox_autotuning_pass_v3, trellis_spec_audit_prompt, p-a-fair-three-way/three_way_table}.md`（6 项） | 旧 scratch/prompt（2026-05，pre-refactor）散仓库根 artifacts/ | 低（非证据、非承重） | **归档或移出**（过 RETIRED-INDEX 闸；批量 housekeeping R7） | `b3e3fef4` |
| **C8** | `.touch-set/{line-C-iq2s, line-D-f5, line-xscalar-f6, line-xscalar}.txt`（4 tracked，保 `README.md` + `_example.txt`） | per-line 触碰集 scratch，对应线已落地 stale | 极低 | **归档 4 个 stale**（保 README + _example） | `b3e3fef4` |
| **C9** | `docs/reports/2026-07-06-TCRV_IR三层与编译器身份_parallel-writer-leftover.md`（**untracked**，= 会话起 git status 唯一 ??项） | 文件名自陈 parallel-writer leftover | 低（untracked，无历史包袱） | **内容并入正规 doc 或删**（untracked 无需复原指针） | N/A（untracked） |
| **C10** | `.worktrees/cache/`（gitignored） | 磁盘杂物、可再生 | 无 | **`git worktree prune` / rm**（无需复原指针） | N/A（gitignored） |
| **C11** | `scripts/{codex_serial_supervisor.py, _prompt.md, rvv_accumulator_sweep_measure, rvv_fair_three_way_measure, rvv_generated_bundle_abi_e2e, rvv_generated_bundle_same_target_measure, rvv_remote_probe, tensorextlite_runtime_abi_e2e}.py`（8 项） | 与 `tools/e2e-harness` 测量库职能重叠；`codex_serial_supervisor*` 疑编排 scratch | **中**：`rvv_fair_three_way_measure.py` / `rvv_remote_probe.py` **被执行总纲 §7 引为 objdump/probe 锚点**（`:502`/`:28`）→ **未验非活跃前禁删** | **逐一评估**（活跃测量脚本 → 合并进 tools/ 测量库家；`codex_serial_supervisor*` → 查 staleness 后归档） | `b3e3fef4` |

**排期分类**：
- **可立即（低风险 · docs/data 域 · 与构造并行先行）**：C7 / C8 / C9 / C10（housekeeping 批 = AUDIT R7；过 RETIRED-INDEX 闸）。
- **需评估（不可逆或引用存疑）**：C5 的 `.o` 移出 VC（不可逆 → **必问**）；C11 逐脚本 staleness + 引用核查（`rvv_fair_three_way_measure`/`rvv_remote_probe` 被 §7 引，**先核非活跃**）。

### 乙.2 R2/R3 lib 移动（RVV 插件按件分子目录 + 前门统一目录）

| R# | 建议 | 触碰域 | 互斥性 |
|---|---|---|---|
| **R2** | `lib/Plugin/RVV/`（~36 文件）按格位分子目录 `FrontDoor/·Selection/·Schedule/·BodyRealization/`，对齐 Template 参考家族五件套 | **code（动 lib）** | **与构造互斥** |
| **R3** | 前门 `*SourceFrontDoor.cpp`/`*StreamFrontDoor.cpp` 统一 `FrontDoor/` 归属，消除「靠命名识别」（③前门 = 最低定位度件 LOW-MEDIUM） | **code（动 lib）** | **与构造互斥**（与 R2 同批 mv） |

**风险评估**：
- **性质 = 纯 `git mv` + include 路径 + CMake 更新**（无逻辑改），风险主要在**编译断裂 + 与并行构造线的 merge 冲突**（memory `parallel-lines-need-disjoint-files`：ODS/verifier/emitter 天然跨线共享，改动必须串行）。
- **互斥窗口现状（有利）**：ROADMAP 排队 #4「G4 IME 构造轴 M0→M2b」= **✅ 已收（结构轴收口，`6dcb5db4`）**；当前主线 = **测量总攻**（perf 测量，**不动 lib 构造**）。最近 `lib/Plugin/RVV/` 提交 = 矿脉退役/SEL-1-T5（M4 收口，已完）。→ **构造静默期成立**，R2/R3 的互斥前置（「排 G4 M1 贯通后」）**已满足**。
- **唯一残留构造风险 = X-SCALAR 收口（甲.4 XS-M3）触碰 `VariantSelection` + Scalar 插件**，与 R2/R3 的 `lib/Plugin/RVV/` **文件集不相交**（Scalar ≠ RVV 目录）→ 可并行，但 `VariantSelection.cpp`（通用选择器）是**跨家族共享文件**，若 XS-M3 与 R2/R3 同期须串行该文件。

**是否与 [RENAME] 命名统一一次做**：**建议是**（AUDIT R2/R3 明示「与改名同批 mv 省一次搬家」）。[RENAME] 命名统一基准已定（`REPOSITORY-MAP-五大件.md` §3）：
- **P1 三所指**（宽 LMUL 分组 / N-operand 构造 / [GAP-P1]）→ 退役裸「P1」，改 pattern_id（`SCHED-WIDE-LMUL`/`CONSTRUCT-N-OPERAND-ROUTE`），保留 [GAP-P1]；
- **P4 双所指**（PAT-2 形式槽 vs PAT-S6 机制）→ 以 registry pattern_id 为唯一真源；
- **F-1 命名碰撞**（零分支门 vs shape 门）→ shape 门改 `[C1-SHAPE]`/`[F-STRONG]`（FALSIFIER-INDEX §2）。
- 命名统一 = **docs + 少量标签**（多数已 codify，落地是引用替换），与 R2/R3 物理 mv 同批做，**一次搬家**。

**排期**：R2/R3 = **单独 [RENAME+AUDIT 后续] 会话**，触碰 lib、须**主会话**（非并行 agent，避免绝对路径直写主树污染）；触发条件（构造静默）**现已满足**；建议排在测量总攻的 lib-quiet 窗口，与 X-SCALAR 收口错开 `VariantSelection.cpp`。

### 乙.3 check_docs_canon 2 预存 RED（canon 级 · 裁决建议）

已跑 `tools/lint/check_docs_canon.py` 确认 **2 项 RED**（当前树实证）：

| RED | 文件 | 违反 | 根因 | 裁决建议（**canon 级 = 必问用户**） |
|---|---|---|---|---|
| ① | `docs/canon/TianChen-RV_定位-v2.md` | (A) canon/ 白名单：basename 无 `总纲/canon/charter` marker | 定位-v2 **是** charter 级（权威定位，CLAUDE.md/ROADMAP 引），但 checker 靠 basename marker 识别，「定位」不在 marker 集 | **建议扩 `CANON_MARKER_RE` 加「定位」**（最小 churn，承认定位文档 = charter）；备选 = 重命名加 marker（破坏全仓指针，不推荐）/ 移出 canon（降级权威，不推荐）。改 marker 集 = **改 canon 白名单定义 = canon 级 · 必问** |
| ② | `docs/reports/SEALED-WIN-REGISTRY.md` | (B) reports/ append-only：缺 `YYYY-MM-DD-` 日期前缀 | SEALED-WIN-REGISTRY 是**持续增长登记册**（自陈 canon 级 ledger），非单快照 → 日期前缀语义不符（像 `travel-decision-ledger.md`） | **建议加进 `REPORTS_LEDGER_ALLOWLIST`**（与 travel-decision-ledger 同类：声明式持续 append-only ledger）。改 allowlist = 改门定义 = **canon/config 级 · 必问**（但可逆、低风险） |

**判定**：两 RED **均 canon/config 级**（改门定义），按决策权限卡 = **必问用户**、不自决执行。**共性**：两者都是 checker 分类器的 marker/allowlist 未覆盖两个合法 charter/ledger 文件 = **门过严**（fail-closed 误报），非文件放错位置。建议一次裁两项：定位加 marker + SEALED-WIN 进 allowlist，一次 GREEN。

---

## 丙、合并优先级建议（X-SCALAR vs AUDIT 后续 vs 其他 · 测量债已清后的下阶段序）

> 依据 = ROADMAP 排队（顺序即优先级，agent 不自改）+ 「测量债优先」现状 + 触碰集互斥 + 就绪度。**本节 = 建议，排队变更仅用户裁。**

**现状锚点**：ROADMAP 排队 #5「测量总攻」进行中（T6 e2e 传导 + 铺面③-⑤）；#6「[TEMPLATE-AUDIT+RENAME] 合并会话」；#7「[X-SCALAR]/zvfh（排铺面①②后与 AUDIT 后续一并报排期，按 [X-1]）」。本报告即 #7 要求的排期产出。

**建议序（三档，按触碰域 + 就绪度 + 互斥）**：

1. **档 A — 可立即并行先行（docs/data 域 · 无 lib · 不阻塞测量总攻）**：
   - 乙.1 **C7/C8/C9/C10 housekeeping 批**（AUDIT R7；过 RETIRED-INDEX 闸）—— 零风险、清定位债。
   - 甲 **XS-M0 canon 正名** + 乙.3 **check_docs_canon 2 RED 裁决**（打包成一次 canon-必问，用户一次裁：X-SCALAR 状态栏 + 定位 marker + SEALED-WIN allowlist + <300 口径）。
   - **理由**：均 docs 域、与测量总攻文件集不相交、清若干 CI/文档债，投入小收益确定。

2. **档 B — lib-quiet 窗口的主会话批（构造静默期已满足）**：
   - 乙.2 **R2/R3 lib 分子目录 + [RENAME] 命名统一**（一次搬家）—— 主会话、与构造互斥（现满足）、排测量总攻的 lib-quiet 窗口。
   - **与 X-SCALAR 收口的互斥**：若同期做 XS-M3（触 `VariantSelection.cpp`），须串行该共享文件；否则 R2/R3（RVV 目录）与 XS（Scalar 目录）文件集不相交、可并行。

3. **档 C — X-SCALAR 收口（家族#3 · 需用户立项裁）**：
   - **按 [X-1] 重排**：XS-M1（zvfh 事实注册 + 闭包，第一优先 + [S-2] 探针）→ XS-M2（scalar.zfh 复用）→ XS-M3（判据④ 连线 + F-6 CI，N2 boundary 最后一环）→ XS-M4（LED-2 ledger 第二点 + <300 口径落地）。
   - **立项性质**：= 「收口已 70-80% 落地家族」非「大新战役」；剩余有界（4 open）。**用户裁**：现在立主线收口，还是挂被动维护待 zvfh 先行。
   - **叙事权重**：C2 第三点是模板经济学叙事刚需（ROADMAP 定位升级备注），但 ROADMAP 明示「开工按既定触发条件、备注不解锁新战役、不为论文需要立项」→ **建议按工程成熟度立项**（收口一个半成品家族 = 正当工程债），**不以论文/C2 叙事为立项理由**（[NG-4]/2026-07-11 纪律）。

**一句话建议**：档 A（docs 清债 + canon 一次裁）**立即做**；档 B（R2/R3+RENAME）**排 lib-quiet 主会话**（触发已满足）；档 C（X-SCALAR 收口）**按 [X-1] 重排、报用户裁立项性质**——**因其已大面积落地，性价比高、收口成本有界，建议优先于「从零 zvfh」纯度考量，但形式上仍先补 zvfh 事实注册（XS-M1）以对齐 [X-1] + 复用 fact-model 机器**。

---

## 产出确认

- **甲 X-SCALAR/zvfh 接入 plan**：[X-1] 顺序（zvfh 第一 / X-SCALAR 第二 · 实际工作序偏离已披露）· 家族/子扩展粒度辨析 · N2 boundary 核验（X-SCALAR PASS Case-A、zvfh PASS integrated 子扩展）· **★头号发现：X-SCALAR 已大面积落地（3 op + F-6 双断言）canon 状态栏 stale** · reframe 里程碑 XS-M0..M4（收口非从零）· **C2 ledger 实测 1501 raw / 1131 cloc-approx（vs <300 目标 ~4-5× 缺口 + integrated/independent 结构解释）** · 就绪度（无硬件阻塞、canon 正名必问）。
- **乙 [RENAME+AUDIT] 后续排期**：C5-C11 逐项（当前树已核 · 复原指针 b3e3fef4 · 承重证据注册非删已明确）· R2/R3 lib 移动（构造静默期已满足 · 与 [RENAME] 一次搬家 · 主会话互斥）· check_docs_canon 2 RED（均 canon 级 · 门过严误报 · 一次裁建议）。
- **丙 合并优先级**：档 A（docs 清债 + canon 一次裁，立即）/ 档 B（R2/R3+RENAME，lib-quiet 主会话）/ 档 C（X-SCALAR 收口，按 [X-1] 重排、用户裁立项）。
- **诚实边界（[NG-4]）**：C2 曲线现状 = **1/≥3**（landed X-SCALAR 把「1 点」变「~2 点」但未成曲线，zvfh 第 3 点未落）；<300 目标当前**未 met**（实测 1501 raw），已如实标注 + 给结构解释；X-SCALAR 剩余成本以预估标注；排期是建议非承诺；X-SCALAR 立项 = 用户裁（本报告不预设）。
- **HEAD = `9f4a15f3` 未变**；未改任何 code/schema/lit/ROADMAP/既有 casefile（本报告纯新增 docs 一文件：`docs/reports/2026-07-11-X-SCALAR-AUDIT后续-排期报告.md`）。
