# 包2 · c 进料口盘点（能力表）— θ=f(g,c) 三角色全景盘点 v2

HEAD pin = `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`（工作树 == HEAD 此 pin）

> **角色框架**：c = 板侧输入（硅片/厂商定义的能力量：VLEN/寄存器数/分数LMUL有无…），存放地应为**能力表**。本包只列事实 + 出处，禁判断/禁分类/禁结论。
> 全文引用一律 `file:line`（工作树内容 == pin）。"不存在"类断言均附机算枚举命令。

---

## 0. 能力表有几个"表面"（枚举，非结论）

机算发现 c 进料存在**四个并行表面**，同一板事实在不同表面各有一份形态：

| 表面 | 定义处 | 形态 | 出处 |
|---|---|---|---|
| S1 schema 声明（TARGET 目标形） | `schema/capability.schema.v1.json` | JSON namespaces + fact_record 字段 | 该文件 `$meta`（15:6）自述"declares TARGET shape, NOT a snapshot of current code" |
| S2 代码侧 fact 记录 | `include/Weft/Support/CapabilityModel.h` `class CapabilityDescriptor` | symbolName/id/kind/status/availability/properties(map)/relations | CapabilityModel.h:86-146 |
| S3 探测料口结构体 | `include/Weft/Plugin/RVV/RVVCapabilityProfile.h` `struct RVVProbeCapabilityFacts` | 15 个 C++ 字段 | RVVCapabilityProfile.h:61-88 |
| S4 in-IR provider op 属性 | `weft.exec.capability` / `weft.exec.target` 上的 StringAttr 属性 | property 字符串键值对 | 由 S3 经 `buildRVVTargetCapabilitiesFromProbeFacts` 产出（RVVCapabilityProfile.cpp:486-608）+ 材料化 pass 盖章 |

schema `$meta`（capability.schema.v1.json:6）明文：`provenance/trust/subclass`、closed kind enum、namespaced-typed params、unified operand roles、serializable plugin signature **均 grep=0 in code today**，"code and schema.def are intentionally divergent"。

---

## 1. capability schema 全字段机算枚举（S1 · schema/capability.schema.v1.json）

### 1a. item_1_fact_record — 每条 capability fact 携带的字段（9 个）
出处 capability.schema.v1.json:15-85。

| # | 字段 | 类型 | 值域/默认 | 备注 | 行 |
|---|---|---|---|---|---|
| 1 | `id` | string | namespaced dotted stable id（如 `rvv.v`/`ime.vmadot`/`scalar.zbb`/`uarch.vrgather_slow`） | authoritative_key=true；bench名/日志/perf值禁成 id（I9） | 18-23 |
| 2 | `kind` | enum | → item_2（closed 4 值） | | 24-27 |
| 3 | `subclass` | string | optional | **Declared-with-no-producer in v1**（原文 31） | 28-32 |
| 4 | `status` | enum closed | {available, unavailable, disabled, missing}，default=available | missing 解释为 available；status 优先于 availability（[S-2]） | 33-44 |
| 5 | `availability` | enum closed | {available, unavailable} | | 45-53 |
| 6 | `provenance` | enum closed | {hwprobe, cpuinfo, vendor_table, manual}，default=manual | 证据线字段（I8），不参与 compute；**Declared-with-no-producer in v1**（producer 是 P2） | 54-65 |
| 7 | `trust` | enum closed | {measured, declared}，default=declared | 证据线（I8）；**Declared-with-no-producer in v1** | 66-75 |
| 8 | `params` | namespaced_struct | → item_4 | | 76-79 |
| 9 | `relations` | relations | → item_3 | | 80-83 |

### 1b. item_2_kind_enum — fact 类别轴（closed enum 4 值）
出处 capability.schema.v1.json:86-105。成员：`isa_ext` / `sub_ext` / `uarch` / `policy`。companion_field=`subclass`。映射（97-102）：isa_ext=ISA扩展 / sub_ext=子扩展(zvfh/zbb) / uarch=微架构·内存·VLEN facts / policy=toolchain·runtime-offload·thread-runtime可用性+build/permission门。

### 1c. item_3_relation_types — 关系类型表（3 关系 + 每关系语义标注）
出处 capability.schema.v1.json:106-125。

| 关系 | value_type | semantics | 行 |
|---|---|---|---|
| `provides` | list<capability_id_string> | satisfies-by-alias | 109-113 |
| `implies` | list<capability_id_string> | transitive-satisfiable（load 时物化一次闭包 [S-2]） | 114-118 |
| `conflicts` | list<capability_id_string> | fail-closed-mutual-exclusion | 119-123 |

### 1d. item_4_params_namespaces — 命名空间化 typed param 字段（7 namespace）★核查点密集区
出处 capability.schema.v1.json:126-162。

| # | namespace | type | layer | 备注 | 行 |
|---|---|---|---|---|---|
| 1 | `vlen` | int | hardware-fact | raw HW VLEN bits（Layer1）；非线程数/门/形状/AVL | 129-136 |
| 2 | `elen` | int | hardware-fact | | 137-139 |
| 3 | `sew_set` | set_of_int | hardware-fact | 支持的 SEW allow-list，typed set（非 CSV string） | 138-142 |
| 4 | `lmul_budget` | int_or_fraction | compile-time-variant-config | | 143-146 |
| 5 | **`vreg_count`** | int | hardware-fact | **★定点核查:寄存器总数——仅此处声明** | 147-150 |
| 6 | `cacheline` | int | hardware-fact | | 151-154 |
| 7 | `ime.tile` | struct | hardware-fact | IME tile 几何，namespaced under ime | 155-159 |

**机算核查（`$meta` 自述 grep=0 in code）**：`git grep -c 'vreg_count\|sew_set\|lmul_budget\|elen\|ime.tile' d173f4… -- lib include tools` → 除 schema JSON 自身外**无代码命中**（这些 namespaced-typed param 字段是 TARGET 声明，代码侧未落，属跟踪中的 conformance gap，见 `$meta` note 行 6）。

### 1e. item_5 / item_6 / not_in_shape（非 c 料口，简记）
- item_5：ExtensionPlugin quasi-ABI 签名投影（18 virtual methods，3 pure）——插件协议 shape，非板事实（163-297）。
- item_6：operand-role 词表（axis-A ABI-slot 28 成员 / axis-B body-op role 7 属性）——算子角色，非 c（298-357）。
- `not_in_shape`（358-368）明文排除：concrete fact rows、params VALUES（`sew_set={8,16,32}`、`vlen=128`）、plugin internal code、measurement library、pattern-registry entries、profiles。⟹ **schema 明确"具体板值不在 shape 内"** → 三板现值不住 schema，住 fixture/board册/pass-option（见 §3）。

---

## 2. 代码侧 c 料口全字段（S2 + S3）

### 2a. CapabilityDescriptor 字段（S2 · CapabilityModel.h:135-146）
私有成员：`symbolName`(string)、`id`(string)、`kind`(string)、`status`(string)、`availability`(enum CapabilityAvailability{Available,Unavailable})、`properties`(std::map<string,string>)、`relations`(CapabilityRelationsAttr 内含 provides/implies/conflicts)。
- kind 值域**非 closed enum**：代码里是自由 string（构造函数 CapabilityModel.cpp:211-218 直接吃 StringRef），与 schema item_2 的 closed-4 声明背离。实测产出的 kind 字面见 §2c。
- properties 是**自由 string map**（非 schema item_4 的 namespaced typed struct）；`collectCapabilityProperties`（CapabilityModel.cpp:86-99）把除 core/relation 外的所有属性 stringify 塞进 map。core 属性（isCoreCapabilityAttribute，CapabilityModel.cpp:44-52）= sym_name/id/kind/target_kind/status/capability_providers。

### 2b. RVVProbeCapabilityFacts 探测料口结构体（S3 · RVVCapabilityProfile.h:61-88）——15 字段
| # | 字段 | 类型 | 默认 | 消费见 §4 | 行 |
|---|---|---|---|---|---|
| 1 | `architecture` | std::string | | | 62 |
| 2 | `hartCount` | uint64 | 0 | | 63 |
| 3 | `vlenbBytes` | uint64 | 0 | | 64 |
| 4 | **`cachelineBytes`** | uint64 | 0 | **★零消费(见 §5)** | 65-71 |
| 5 | **`imePresent`** | bool | false | **★零消费(见 §5)** | 72-77 |
| 6 | `isaVectorHints` | std::string | | | 78 |
| 7 | `clangAvailable` | bool | false | | 79 |
| 8 | `clangVersion` | std::string | | | 80 |
| 9 | `cmakeAvailable` | bool | false | | 81 |
| 10 | `cmakeVersion` | std::string | | | 82 |
| 11 | `minimalRVVCompileRunSucceeded` | bool | false | | 83 |
| 12 | `selectedMarch` | std::string | | | 84 |
| 13 | `selectedMABI` | std::string | | | 85 |
| 14 | `sourceSHA256` | std::string | | | 86 |
| 15 | `binarySHA256` | std::string | | | 87 |
| — | `deriveIMEPresent(march,hints)` | inline bool | | **★零调用者(见 §5)**；token=`xsmtvdotii` | 194-199 |

结构体字段 4/5（cachelineBytes/imePresent）header 原文自述：cachelineBytes"NOT consumed by any selector yet"（70）、imePresent"NOT wired to any selector by this task"（76-77）。

### 2c. RVV 探测→provider op 实产出（S4 · RVVCapabilityProfile.cpp:486-608）
`buildRVVTargetCapabilitiesFromProbeFacts` 产出的 provider op 及其 property 键（原样）：

| provider symbol | id | kind | 产出 property 键 | 产出行 |
|---|---|---|---|---|
| `rvv`(getRVVPreferredCapabilitySymbol) | `rvv` | `isa-vector`(getRVVCapabilityKind) | architecture, isa_vector_hints, **supported_sew**(派生), **supported_lmul**(派生), **rvv_version**(派生) | 493-525 |
| `rvv_zvfhmin`（条件） | `rvv.zvfhmin` | `isa-vector-fp16` | —(implies rvv.zve32f) | 541-547 |
| `rvv_zvfh`（条件） | `rvv.zvfh` | `isa-vector-fp16` | —(implies rvv.zvfhmin) | 548-554 |
| `rvv_hart_count` | `rvv.hart_count` | `uarch` | `count`(=hartCount)；provides `target.hart_count` | 556-561 |
| `rvv_vlenb_bytes`（条件 vlenbBytes≠0） | `rvv.vlenb_bytes` | `uarch` | `bytes`(=vlenbBytes) | 562-568 |
| `rvv_toolchain_clang` | `rvv.toolchain.clang` | `toolchain` | `version`(=clangVersion) | 569-573 |
| `rvv_toolchain_cmake` | `rvv.toolchain.cmake` | `toolchain` | `version`(=cmakeVersion) | 574-578 |
| `rvv_probe_compile_run` | `rvv.probe.compile_run` | `toolchain` | `selected_march`, `selected_mabi`(条件), `source_sha256`(条件), `binary_sha256`(条件) | 580-592 |
| `rvv_toolchain_march` | `rvv.toolchain.march` | `toolchain` | `value`(=selectedMarch) | 594-598 |
| `rvv_toolchain_mabi`（条件） | `rvv.toolchain.mabi` | `toolchain` | `value`(=selectedMABI) | 599-605 |

**注**：cachelineBytes/imePresent 两字段在此产出函数里**从不被读取/从不产出任何 fact**（机算：`git grep -n 'cacheline\|imePresent\|\.imePresent' … -- lib/Plugin/RVV/RVVCapabilityProfile.cpp` → 命中 0，见 §5）。

### 2d. 三条派生轴（θ-形还是 c-形？—— 只列事实）
`buildRVVTargetCapabilitiesFromProbeFacts` 内把三个**派生量**盖到 `rvv` op（RVVCapabilityProfile.cpp:504-521）：
- `supported_sew` ← `deriveSupportedSEWAllowList(march,hints)`（.cpp:222-243）：zve32*→"8,16,32"；full-V/zve64/gcv/xtheadvector→"8,16,32,64"；否则""。
- `supported_lmul` ← `deriveSupportedLMULAllowList`（.cpp:260-277）：RVV0.7→"m1,m2,m4,m8"；RVV1.0→"mf8,mf4,mf2,m1,m2,m4,m8"；Unknown→""。
- `rvv_version` ← `stringifyRVVVersion(deriveRVVVersion)`（.cpp:341-375）："0.7"/"1.0"/""。
header/注释自述这些是"TARGET-CAPABILITY fact（配置目标支持什么），NOT plugin-selected config"（RVVCapabilityProfile.h:108-144 / .cpp:206-221）。**（此陈述为原文事实照录，不作三角色归类。）**

---

## 3. 三块板实例逐字段现值（原样摊开）

★ schema `not_in_shape`（capability.schema.v1.json:358-368）明文：具体板值**不在 schema 内**。机算搜索（`git grep -ln 'weft.exec.capability' … -- schema experiments`）**未发现单一"per-board 能力实例权威 JSON"**；三板现值散落在三处：(A) board册 prose、(B) test/tool fixtures、(C) pass `-march` 字符串。以下原样照录，标出处。

### 3a. board册权威（.trellis/spec/measurement/板册.md:7-16）
| 别名 | 身份 | 向量 | 测量特性 | 出处行 |
|---|---|---|---|---|
| rvv | openEuler·64核 | **VLEN128** | 有 cache-miss 硬件计数器 | 9 |
| k1 | SpacemiT X60·8核 | **VLEN256 + IME** | 无可用 PMU（仅指令/周期）·常载偏高 | 10 |
| scalar | 超锐·Fedora42·8核 | **无向量（实证）** | 载荷极低·有硬件计数器 | 11 |
| rvv07 | RVV0.7前身·128核 | 非标准 | — | REGISTERED-PENDING·零工作 | 12 |

scalar"无向量"实证依据（板册.md:16 原文）：板端 `/proc/cpuinfo` isa 串 = `rv64imafdch_zicntr_zicsr_zifencei_zihpm_zaamo_zalrsc_zca_zcd`——**无 `v` 扩展**。

### 3b. t1d 双实例 fixture 现值（tools/visibility/t1d_dual_instance.py:179-188）——rvv/k1 逐字段
```
INSTANCES = {
  "rvv": board="rvv", vlen="128", hints="rv64gcv_zvl128b",
         ime_status="missing", ime_extra="", march="rv64gcv"           # :180-182
  "k1":  board="k1",  vlen="256", hints="rv64gcv_zvl256b",
         ime_status="available",
         ime_extra=  available_harts="0-3",  ime_matmul_shape="256x256x256",
         march="rv64gcv_zvl256b"                                        # :183-188
}
```
t1d 产出的 in-IR provider 属性模板（t1d_dual_instance.py:88-100, 120-129）：rvv capability 携 `id="rvv"`, `isa_vector_hints="{hints}"`, `vlen_bits="{vlen}"`；IME capability 携 `march="rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii"`, `vlen_bits="{vlen}"`, `status="{ime_status}"`, （k1）`available_harts="0-3"`, `ime_matmul_shape="256x256x256"`。
注（t1d 头 176-179 原文）：rvv board=VLEN128 无 IME silicon（spacemit.ime status=missing→[S-2] 解释为 unavailable）；k1 board=VLEN256 带 IME。

### 3c. divergence 测试 fixture 现值（真实 profile 逐字段）
- full-V profile（test/Conversion/EmitC/…one-kernel-two-profiles.mlir:16-27 + test/Plugin/RVVCapabilityProfileDivergenceTest.cpp:163-165）：`isa_vector_hints="rv64gcv_zvl128b"`, `supported_sew="8,16,32,64"`, `supported_lmul="mf8,mf4,mf2,m1,m2,m4,m8"`, march=`rv64gcv`。
- zve32x profile（同 mlir:52-63 + Test.cpp:167-169）：`isa_vector_hints="rv64imac_zve32x_zvl128b"`, `supported_sew="8,16,32"`(无64), march=`rv64imac_zve32x`。
- rvv07 profile（Test.cpp:253-255）：`isa_vector_hints="rv64gc_xtheadvector"`, march=`rv64gc_xtheadvector`。
- makeBaseFacts 公共值（Test.cpp:56-67）：architecture="riscv64", hartCount=4, **vlenbBytes=16**, clangVersion="clang version 20.0.0", cmakeVersion="cmake version 3.28.0", selectedMABI="lp64d"。
- **scalar（无V）板无对应 RVV provider fixture**：其能力形态 = **不存在 rvv-kind provider**（board册.md:16 无 v 扩展）；vector-absent 实例住 load-time capability resolution / [F-6] independent-family 语境（CapabilityModel.h:194-211 impliedClosureAvoidsRVVNamespace）。**采不到：无 scalar 板逐字段能力属性表**（只有"无 v 扩展"这一否定事实）。

### 3d. 采不到清单（板实例）
- **无单一 per-board 能力实例权威文件**（JSON/MLIR）：现值三处分散（board册 prose + fixture + pass-option），机算未见统一 board profile 台账。
- k1 的 `vreg_count` / `cacheline` / `elen` / `sew_set` 具体值：**采不到**（board册未列，fixture 未含，schema namespace 声明但代码零落）。
- scalar 板逐字段能力属性：**采不到**（仅"无 v 扩展"否定事实）。
- rvv07 板：REGISTERED-PENDING·零工作（板册.md:12），无 fixture 值。

---

## 4. 定点核查（逐点 file:line 或"采不到"）

### 4a. 寄存器总数（vreg_count / vreg 数）
- schema 声明：`vreg_count` int hardware-fact（capability.schema.v1.json:147-150）。
- 代码侧落地：**采不到**。机算 `git grep -c 'vreg_count' d173f4… -- lib include tools` → 仅 schema JSON 命中；**无 struct 字段、无 property 键、无消费点**。
- 三板实例值：**采不到**（无任何 board 列 vreg 数）。
- ⟹ 寄存器总数 = schema-only 声明，进料/消费全空。

### 4b. 分数 LMUL 存在性
- **无独立 boolean 字段**。分数 LMUL 存在性**内嵌**在 `supported_lmul` allow-list 字符串子串里：RVV1.0 含 `mf8,mf4,mf2`（存在），RVV0.7 只有 `m1,m2,m4,m8`（不存在）。
- 派生处 `deriveSupportedLMULAllowList`（RVVCapabilityProfile.cpp:260-277）：RVV0.7 分支原文注"NO fractional LMUL exists on this generation"（263-266）；RVV1.0 分支给全格含 mf 档（267-270）。
- header 原文（RVVCapabilityProfile.h:127-142）：RVV0.7.1 XuanTie header"declares ZERO mf2/mf4/mf8 types"。
- ⟹ 分数 LMUL 存在性 = `supported_lmul` 子串，非独立字段。

### 4c. 整数倍 LMUL 合法集
- **无独立字段**。整数倍 LMUL 合法集**内嵌**在同一 `supported_lmul` 串："m1,m2,m4,m8"（两代通用整数档）。派生 RVVCapabilityProfile.cpp:260-277。
- 消费：`supported_lmul` 由 EmitC 门查（RVVToEmitC.cpp:2298-2302 provider 取 supported_lmul 排除 typed body LMUL）+ RVVEmitCRoutePlanning.cpp:244（kSupportedLMULPropertyName）。

### 4d. 向量单元有无（vector unit presence）
- **无独立 boolean "has_vector" 字段**。向量单元有无 = **rvv-kind provider 的存在/缺席**：
  - 有 V：kernel 里存在 `id="rvv"`/`kind="isa-vector"` 的 provider op（isRVVCapabilityProvider，RVVProbedCapabilityAxesMaterialization.cpp:56-66；hasAvailableRVVCapability 门 RVVExtensionPlugin.cpp:637）。
  - 无 V（scalar）：**不生成 rvv provider**（board册.md:16 无 v 扩展）→ RVV 门 supportsOperation 返 false（RVVExtensionPlugin.cpp:636-638）。
- 探测侧有向量证据校验 `hasRVVVectorHint`（RVVCapabilityProfile.cpp:140-162）+ `validateRVVProbeCapabilityFacts` 要求"ISA/vector hint must contain RVV vector evidence"（.cpp:446-447）。
- ⟹ 向量单元有无 = provider 存在性建模，非字段。

---

## 5. ★消费核查（逐字段：今天被哪些 f / 哪些代码读）

### 5a. 有消费的 c 字段（消费点清单 file:line）
| c 字段/property | 消费点（f） | file:line |
|---|---|---|
| `supported_sew` | EmitC 合法性门（排除 typed body SEW） | RVVToEmitC.cpp:2290-2294 |
| `supported_sew` | route planning 门 | RVVEmitCRoutePlanning.cpp:242, 271 |
| `supported_lmul` | EmitC 合法性门 | RVVToEmitC.cpp:2298-2302 |
| `supported_lmul` | route planning 门 | RVVEmitCRoutePlanning.cpp:244, 280 |
| `rvv_version` | EmitC 门（rvv_version=0.7 gate 掉 ta/ma agnostic body） | RVVToEmitC.cpp:2309-2321 |
| `architecture` | route planning | RVVEmitCRoutePlanning.cpp:139 |
| `isa_vector_hints` | route planning | RVVEmitCRoutePlanning.cpp:147, 151 |
| `hart_count` property `count` | HartParallelCapabilities（线程能力） | HartParallelCapabilities.cpp:45, 66, 92, 104, 178 |
| `count`/`target.hart_count` id | HartParallelCapabilities collectProvidersByID | HartParallelCapabilities.cpp:66 |
| `vlen_bits`（provider property） | **IME plugin** 合法性门 | IMEExtensionPlugin.cpp:332, 337 |
| `vlen_bits` | **IME backend emission driver**（每 fragment vreg cost） | IMEBackendEmissionDriver.cpp:426, 496, 505, 520 |
| IME `march` | IME plugin 门（token `xsmtvdotii`） | IMEExtensionPlugin.cpp:322 |
| IME `available_harts` | IME plugin 门 | IMEExtensionPlugin.cpp:515 |
| IME `ime_signedness` | IME plugin 门 | IMEExtensionPlugin.cpp:346 |
| IME `ime_matmul_shape` | IME plugin 门 | IMEExtensionPlugin.cpp:390 |
| IME `ime_weight_format` | IME plugin 门 | IMEExtensionPlugin.cpp:440 |
| IME `ime_slide` | IME plugin 门 | IMEExtensionPlugin.cpp:486 |

### 5b. 消费-但-无 producer（gate-only；探测侧不产出，属手写 fixture attr）
| property | 消费点 | producer 机算 |
|---|---|---|
| `required_tail_policy` | RVVEmitCRoutePlanning.cpp:246, 288（门） | `git grep 'required_tail_policy' … -- lib include` 仅命中该 gate 文件（RVVEmitCRoutePlanning.cpp:64,288）；探测产出函数不产出 → 无 producer |
| `required_mask_policy` | RVVEmitCRoutePlanning.cpp:248, 296（门） | 同上（RVVEmitCRoutePlanning.cpp:66,296）；无 producer |

### 5c. ★零消费字段（进了料没人用）
逐条附机算枚举（严禁 head/tail；用 `git grep -n` 全路径 + 排除 `_attic`/worktrees/build）。

| # | 零消费字段 | 机算枚举 | 结论事实 |
|---|---|---|---|
| 1 | `RVVProbeCapabilityFacts::cachelineBytes` | `git grep -n 'cachelineBytes\|cacheline' d173f4… -- lib include tools schema \| grep -v _attic` → 命中：RVVCapabilityProfile.h:65/71（声明+注释）、RVVRepackTilingSelection.h:438（注释"known GAP; once plumbed"）、schema:151（namespace 声明）。**无任何 setAttr/getProperty/selector 读点** | 结构体字段存在，探测产出函数不产出 fact，无消费者 |
| 2 | `RVVProbeCapabilityFacts::imePresent` | `git grep -n 'imePresent\|\.imePresent'` → 仅 RVVCapabilityProfile.h:73/77（声明+注释）。无 setter（除结构体默认）、无 reader | 字段存在，零产出零消费 |
| 3 | `deriveIMEPresent(...)`（inline fn） | `git grep -n 'deriveIMEPresent'` → 仅 RVVCapabilityProfile.h:73(注释)/194(定义)。**零调用者** | 函数已建，无调用点 |
| 4 | `rvv.vlenb_bytes` property `bytes`（vlenbBytes） | `git grep -n '"bytes"\|vlenb_bytes\|getRVVVLenBBytes' … -- lib include tools \| grep -v _attic` → 命中全在**产出侧**（RVVCapabilityProfile.cpp:28-30,385-391,562-568 + header 声明）。**无 getProperty("bytes") 或任何读点** | provider op 产出该 property，无消费者 |
| 5 | `rvv.toolchain.clang` property `version`（clangVersion） | `git grep -n 'toolchain.clang\|getRVVClangToolchain'` → 仅产出侧（RVVCapabilityProfile.cpp:31-34,393-399,569-573 + header）。无读点 | 证据线产出，零消费 |
| 6 | `rvv.toolchain.cmake` property `version`（cmakeVersion） | 同 5 模式（.cpp:35-38,401-407,574-578）。无读点 | 证据线产出，零消费 |
| 7 | `rvv.probe.compile_run` 属性 `selected_march`/`selected_mabi`/`source_sha256`/`binary_sha256` | `git grep -n 'probe.compile_run\|source_sha256\|binary_sha256'` → 仅产出侧（.cpp:39-42,409-415,580-592）。无读点 | 证据线产出（4 键），零消费 |
| 8 | `rvv.toolchain.march` property `value`（selectedMarch，**作为 provider property**） | `git grep -n 'toolchain.march\|getRVVSelectedMarch'` → 仅产出侧（.cpp:43-46,417-423,594-598）。无 provider-property 读点 | **注**：selectedMarch 作为 **pass -march 字符串**被大量消费（见 §6），但**作为 provider op property**（键 `value`）零消费 |
| 9 | `rvv.toolchain.mabi` property `value`（selectedMABI） | `git grep -n 'toolchain.mabi\|getRVVSelectedMABI'` → 仅产出侧（.cpp:47-50,425-431,599-605）。无读点 | 证据线产出，零消费 |

### 5d. 整条探测→capability 产出路径的 driver 挂接状态（事实）
- 自由函数 `buildRVVTargetCapabilitiesFromProbeFacts` 调用点：`git grep -n 'buildRVVTargetCapabilitiesFromProbeFacts' … -- lib include tools \| grep -v _attic` → 命中 2：定义（RVVCapabilityProfile.cpp:487）+ 插件方法内调用（RVVExtensionPlugin.cpp:665）。
- 插件方法 `RVVExtensionPlugin::buildTargetCapabilitiesFromProbeFacts` 调用点：`git grep -n 'buildTargetCapabilitiesFromProbeFacts' … -- lib include tools test` → 命中仅 声明（RVVExtensionPlugin.h:43）+ 定义（RVVExtensionPlugin.cpp:663）。**lib/tools 内无外部 driver 调用**；唯一实际驱动 = 测试直调自由函数（test/Plugin/RVVCapabilityProfileDivergenceTest.cpp:171 等）。⟹ **探测→capability-op 产出路径在本 pin 无生产驱动挂接**（事实照录，不判断）。

---

## 6. 表外读板信息的旁路（逐处列：读什么·在哪读·进没进 schema）

★这是 c 进料的**最大旁路面**：真实"板 VLEN"值在多数 f 里**不经能力表 provider op**，而经 **pass `-march` 字符串**重解析。

### 6a. march / isa-vector-hints 作为 PASS OPTION（非 in-IR fact）
- 定义：`Option<"march",…std::string,default="">` + `Option<"isaVectorHints","isa-vector-hints",…>` 在 include/Weft/Transforms/Passes.td 多处（:337-340, :398-403, :421-425, :569-573）。
- 读法：各 front-door/schedule pass 从**自己的 pass option** 取 march 字符串，喂 `deriveMinimumVLEN(march,hints)` 现场重解析出 VLEN bits——**不读 capability provider op**。
- 进 schema？**否**。这是原始 march 串，非 schema fact_record/params。t1d 自身注（t1d_dual_instance.py:457）原文："march= is a PASS OPTION, not an in-IR fact"。

### 6b. `deriveMinimumVLEN` 消费点清单（march 串 → VLEN，绕过表）
定义 RVVCapabilityProfile.cpp:289-332。以下 f 现场调用它（从 pass option march 取值）：
| 消费 f | file:line |
|---|---|
| RVVLowerQuantContraction（accLmul/contraction 选择） | RVVLowerQuantContraction.cpp:1407, 1544 |
| RVVMonolithicBlockDotSourceFrontDoor | RVVMonolithicBlockDotSourceFrontDoor.cpp:3395 |
| RVVDequantDotSourceFrontDoor | RVVDequantDotSourceFrontDoor.cpp:376 |
| RVVReductionSourceFrontDoor | RVVReductionSourceFrontDoor.cpp:345 |
| RVVPackedI4DotSourceFrontDoor | RVVPackedI4DotSourceFrontDoor.cpp:209 |
| RVVCodebookDotSourceFrontDoor | RVVCodebookDotSourceFrontDoor.cpp:210 |
| RVVScheduleDescriptorRegistry | RVVScheduleDescriptorRegistry.cpp:457 |
| RVVRepackStripWidthMaterialization | RVVRepackStripWidthMaterialization.cpp:100 |
| `deriveHasZvl128b`(内部再调 deriveMinimumVLEN) | RVVCapabilityProfile.cpp:338；消费 RVVScheduleDescriptorRegistry.cpp:458 |

### 6c. IR 类型宽度/属性读值旁路（in-IR attr 镜像 c 值）
- `minimum_vlen` / `min_vlen` op-attr：schedule descriptor 盖章（RVVScheduleDescriptorRegistry.cpp:189,257,289,322,357,388,407 `descriptor.minimumVLENAttrName="minimum_vlen"`）；verifier 读取，缺省默认 128（RVVDialectWideningOps.cpp:5226 原文"The semantic input is the `minimum_vlen` attr (the deriveMinimumVLEN capability fact); absent, it defaults to 128"）。
- RVVOps.td:4295 原文：`min_vlen` = "a deriveMinimumVLEN(march) value 的 snapshot，consumed by a later …"（op-attr 承载 c 派生值）。
- 进 schema？**否**（是 RVVOps.td 上的 op 属性，非 capability schema）。这是把 c 值（VLEN）**镜像到 op-attr** 的旁路，与能力表 provider property 并行。

### 6d. 材料化 pass = 旁路↔表的桥（把 pass-option march 盖回 provider op）
`RVVProbedCapabilityAxesMaterialization`（lib/Plugin/RVV/Schedule/RVVProbedCapabilityAxesMaterialization.cpp）：从 **pass option** march/isaVectorHints（§6a）派生 supported_sew/supported_lmul/rvv_version，**盖到 in-IR rvv provider op**（:98-126，materializeAxis），使 §6a 的 pass-option 路径与 §5a 的 provider-property 路径**收敛**。已存 hand-authored fixture attr 不覆盖（:73-82 materializeAxis 先查 hasAttr）。进 schema？盖的是 provider op property（S4 表面），非 schema 声明。
- `RVVRepackStripWidthMaterialization`（同目录）：类似，把 deriveMinimumVLEN 派生的 strip 宽盖到 op（:100-109）——注释自述"authority(deriveMinimumVLEN)是 source of truth，op attribute 是 mirror"（:18-20）。

### 6e. 探测侧 march 串解析函数群（都吃 march/hints 字符串，非表）
均在 RVVCapabilityProfile.cpp，输入 = `(selectedMarch, isaVectorHints)` 两串：
- `hasRVVVectorHint`（:140-162）：扫 zve/zvl/zvfh/gcv/xtheadvector/rv64…v 判有无向量。
- `containsIsaToken`（:124-138）：token 边界匹配（zvfh 不误命中 zvfhmin 前缀）。
- `deriveSupportedSEWAllowList`/`deriveSupportedLMULAllowList`/`deriveMinimumVLEN`/`deriveHasZvl128b`/`deriveRVVVersion`/`deriveIMEPresent`：全是 march 串解析。
- 进 schema？否——这些解析的**输入**是 march 串（pass-option 或探测 fact），**输出**才盖成 provider property。

---

## 7. 计数汇总（纯数字 + 口径 · 无解读）

- 口径A｜schema params namespaces（item_4）：**7**（vlen,elen,sew_set,lmul_budget,vreg_count,cacheline,ime.tile）；`$meta` 自述全 grep=0 in code。
- 口径B｜schema fact_record 字段（item_1）：**9**；其中 declared-no-producer = **3**（subclass, provenance, trust）。
- 口径C｜RVVProbeCapabilityFacts struct 输入字段：**15**；其中零消费 = **2**（cachelineBytes, imePresent）+ `deriveIMEPresent` 零调用者。
- 口径D｜RVV 探测产出的 provider-property 发射：**15 property 键 across 9 provider op**；被 f 消费 = **6**（architecture, isa_vector_hints, supported_sew, supported_lmul, rvv_version, hart_count:count），零消费 = **9**（vlenb:bytes, clang:version, cmake:version, compile_run 的 4 键, march:value, mabi:value）。
- 口径E｜IME provider 被消费 property：**7**（march, vlen_bits, available_harts, ime_signedness, ime_matmul_shape, ime_weight_format, ime_slide）；另 gate-only 无 producer = **2**（required_tail_policy, required_mask_policy）。
</content>
</invoke>
