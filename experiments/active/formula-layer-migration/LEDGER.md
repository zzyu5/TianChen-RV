# 模块化公式层 · 增量迁移账本（LEDGER）

> **用户裁决（2026-07-20）**：项目无统一「构建 + 消费公式」的模块化结构·θ=f(g,c) 散落各 pass ⟹ 选 **增量迁移**（非大爆炸重构）：立模块骨架 + 逐个迁散落选择器·每步 **byte-exact + 判决 lit + 三闸 Δ≤0**。**census = 本迁移的 inventory（清单）**。
>
> 本账本 = census v2（全景盘点）的**活 delta 层**：census v2 是钉死基线快照·本账本记「基线以来动了什么 + 下一步迁谁」。**禁把本账本当 census 正本**（正本在 `00-census-final.md`·pin 死）。

---

## 〇 · 公式层的「家」（模块骨架现状）

**唯一 closed-form-f 宿主 = `include/Weft/Plugin/RVV/RVVGearboxSchedule.h`**（RVV 侧）。现住：

| f | 住址 | 角色 | 落地 commit |
|---|---|---|---|
| `getRVVCodebookGatherAnchorLMUL(VLEN, SEW, codebookEntries)` | `RVVGearboxSchedule.h:2713` | codebook gather 锚 LMUL 闭式（最窄 LMUL 使 VLMAX≥codebook_size） | `0f7556199`（MIG-B/T2） |
| `rvvRegisterPressurePeakCost / …Legal / …enumerate` + `RVVRegisterPressureLevel/Combination` | `RVVGearboxSchedule.h:2141-2185` | 寄存器压力不等式（可行 LMUL 集闭式·守 [GAP-P1] 宽度选择 STEP②） | `0e9faee0a`（MIG-0） |

**c-probe 侧读者**（板事实 → c 输入·住能力表侧·非公式宿主）：`readRVVProviderVLenBBytes(module)` @ `RVVCapabilityProfile.cpp:389` / `.h:248`（MIG-A/T1）。

> 迁移方向 = 把散落在 `RVVLowerQuantContraction.cpp`（前门选择器）/ `RVVToEmitC*.cpp`（发射器焊死）里的 θ 决策·逐个搬进 `RVVGearboxSchedule.h` 家（或其 c-probe 姊妹），使「公式在一处构建、各消费者只读结果」。

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

### MIG-T3 · θ20 iq2_xxs 试迁 = 诚实-null（`6bc02a7cc`）（= T3）
- **动作**：试把 θ20 `RVVToEmitCKQuant.cpp:5971 value_or("m2")` 提为 measured gearbox。
- **结论**：**honest-null**——board 证伪前提（VLEN-correctness 墙·非 measured gearbox 可翻）。**无计数变化**（θ20 维持 (c) value_or）·记「试过·held」防复试。emit-neutral 注释入 `RVVToEmitCKQuant.cpp`。

### MIG-1 · 律2 entryLanes→描述符（**在飞**·agent `a7f37c81be0d5d607`）
- **动作**：GridCodebook 机制体里的格式常量（`:2832 entryLanes=4` iq3_xxs / `:3091 entryLanes=8` iq3_s / `:3301 entryLanes=8` iq2_xs / `groupLanes=8` @:504/1002/2532）从**焊死在机制体**→**typed 描述符字段**（律2：机制体点不入源）。
- **来源**：paper-side self-audit `b0866a09c` B1 确认（三条 entryLanes 新增行确凿·律2 违例·未在 `d173f4c2e` census）。
- **目标计数**：残余焊死 g **18 → ≤15**（GridCodebook 9 减 entryLanes/groupLanes 处）。
- **配套**：§四.5 三闸（three-grep 裸格式字面量/板值 value_or/march 解析·Δ≤0·readings 入 commit）。
- **状态**：dispatched·byte-exact + 判决 lit 待 agent 交付。

---

## 三 · HEAD 快照（近似·精确 re-class 待 census-v3 re-pin）

| 维度 | 基线（d173f4c2e） | HEAD（b097d245a·近似） | 说明 |
|---|---|---|---|
| θ (a) f 算出 | 11 | **12** | +θ14（MIG-B） |
| θ (c) 焊死 | 21 | **20** | −θ14（MIG-B）；MIG-1 在飞不改 θ 类（改 g-焊死处计数） |
| θ (d) measured-table | 0 | **≥1** | +θ1 q8（MIG-C·supervisor 裁） |
| 残余焊死 g | 18 | **18→≤15**（MIG-1 落后） | GridCodebook entryLanes/groupLanes |
| provider 零消费 | 9 | **8** | −vlenb:bytes（MIG-A） |
| 家里 closed-form f 族 | 0（家未立） | **2**（codebook-anchor + reg-pressure） | MIG-B + MIG-0 |

> **★诚实**：θ 精确重分类（尤其 θ1 是 (a-含-d-fact) 还是 (d)·θ5 先例算 a）**待 census-v3 re-pin**（需重跑 pkg 六包 agent·本账本只记 delta 事实与 supervisor 裁·不擅自重算 35 分布）。

---

## 四 · 迁移队列（增量迁移·排序·依赖标注）

| # | 迁移 | 目标 | 依赖/闸 | 状态 |
|---|---|---|---|---|
| MIG-1 | 律2 entryLanes→描述符（GridCodebook） | 焊死 g 18→≤15 | §四.5 三闸 | **在飞** |
| MIG-2 | coreLmul 真焊死 θ9-13→f（GridCodebook 497/995·Ternary 862·BQL 111/404） | 焊死(c) 20→15 | **BLOCKED**：[K-10] MAINTAIN——Context **无 coreLmul 字段·无 IR 宽度可读**·强 lift=造假旋钮。**须先给描述符加 coreLmul typed 字段**（MIG-1 的 entryLanes→descriptor 机制可复用铺路） | 阻塞·待 MIG-1 |
| MIG-3 | `selectRepackAccumulatorLMUL` 家族→搬进 `RVVGearboxSchedule.h` 家 | f 集中化（现散在前门 `RVVLowerQuantContraction.cpp:1285`） | byte-exact·消费侧 18 fail-closed 读不变 | 待排 |
| MIG-4 | measured-table 扩行（更多格式入 `kRepackMeasuredM1FasterMeasurements`） | (d) ≥1→≥N | **gated on bench 通道 + T-N**（audit B4·CLAUDE.md:35 现行法：现数 ad-hoc 无资格） | 待排·卡 bench |

---

## 五 · 每步纪律（三闸·不可绕）

1. **byte-exact ZERO-MODEL + 3-arm anti-hollow + CORPUS**：从实际输入零复用重算·三向 mism=0·CORE==PROD（md5 对齐）。
2. **判决 lit（独立复核）**：每迁一步配 decisive lit（能力真翻·false-green 会挂）·独立 agent 跑判决不看 diff。
3. **§四.5 三闸 Δ≤0**：three-grep（裸格式字面量 / 板值 value_or / march 解析）计数不增·readings 入 commit。
4. **canon 措辞变更入 ISSUES 不自改**（用户裁除外）；**🔴禁内联汇编 / 钉死调度绕 clang**。

---

**账本维护**：每完成一 MIG·在 §二 追一节 + §三/§四 更计数与队列。**基线（§一）钉死不动**·只在 census-v3 re-pin 时整体翻页。
