# [RENAME + schema] 变更表 — 独占会话准备件（2026-07-12）

> **用途**：为用户 2026-07-12 裁决的「[RENAME]（项目改名 + 方言前缀）+ pattern 编号/registry schema 正式调整」
> **独占会话**铺路。本 doc = **准备件**：出「旧名 → 新名（或待定）→ 影响面 → 类别 → 执行注意」逐项变更表。
> **本会话只出表，不执行任何替换**（read-only 分析 + 只写本 doc）。HEAD = `6f9099e5`。
>
> **权威碰撞记录（唯一真源）**：`docs/method/REPOSITORY-MAP-五大件.md §3`（P1 三所指 / P4·PAT-2 双所指）
> + `docs/method/FALSIFIER-INDEX.md §2`（F-1 双所指，已执行）。本表**不重裁语义**，只把它们落成
> 独占会话可照做的「影响面精算 + 执行序 + byte-exact 验证点 + CI 门清单」。
>
> **影响面数字口径**：全仓 `git grep`（含 `.trellis/`、`experiments/`、`artifacts/` 全域），**真计数、非估**。
> 「命中行 / 文件」= `git grep -n … | wc -l` / `git grep -l … | wc -l`（HEAD=6f9099e5 快照，会随后续 commit 长大）。

---

## 0. 时点与前置（用户裁·逐字）

- **时点**：**K-quant 净新接线收口后插入**（本表只准备，不触发）。
- **合并范围**：pattern 编号/名称/registry schema 调整 **∪** [RENAME]（项目改名 + 方言前缀）
  = 同属「定义层机械变更」，**一个独占会话执行**：先出变更表（本 doc）→ 一次性机械替换 →
  CI 全绿 + byte-exact 零漂移 + MOVES 映射。
- **改名目标状态**：
  - **A 组（pattern 编号）**：部分**已定名**（`SCHED-WIDE-LMUL` / `CONSTRUCT-N-OPERAND-ROUTE`，deferred 已裁）。
  - **B 组（方言前缀）+ C 组（项目名）**：**新名未定义**（[RENAME] 明确 deferred，`定位-v2.md:71`
    「命名统一（[RENAME]）另开专会话」；全仓无任何已裁新项目名/新方言前缀）→ **本表一律标「待定·需用户裁」，不自创**。

---

## A. pattern 编号 / registry schema（canon 三总纲 C3′ 定义 + `schema/pattern-registry.v1.json` 双改）

**真源现状**：`schema/pattern-registry.v1.json`（`$meta.declares = "PAT-1 unified pattern registry"`）现有
**8 条 pattern_id**（无一个叫「P1」/「P4」/「PAT-2」裸标）：

```
MFLAT-1-dual-fp16-per-block-scale-reconstruct        (mechanized)
MFLAT-2-computed-scale-dequant                       (mechanized)
MFLAT-3-f32-cross-block-scalar-accumulate            (mechanized)
MFLAT-4-flat-block-loop-pre-realized-body            (planned)
MFLAT-5-flat-family-strong-closing                   (planned)
MFLAT-P2c-deferred-ordered-fold                      (measured-negative)
WIDE-DECODE-register-resident-salvage                (deferred-backlog)
PAT-S6-repack-gemm-output-tiling-register-cliff-XFER-1 (mechanized)
```

canon 三总纲则仍用 `[PAT-2]` 的散文 **P1–P8** 槽位名（`科研目标总纲v2.md:121` 定义行）——
**registry 的实 pattern_id 与 canon 的 P1–P8 裸标不一一对应**，这是 A 组要收的核心碰撞。

### A.1 逐项变更表

| # | 旧名 | 新名 | 影响面（文件·命中·风险） | 类别 | 执行注意 |
|---|---|---|---|---|---|
| **A-1** | **P1-(a) 宽 LMUL 分组**（pattern-library 槽） | **`SCHED-WIDE-LMUL`**（已定名·deferred 裁） | canon 定义面 = `科研目标总纲v2.md:121`（[PAT-2] 定义行「P1 宽 LMUL 分组」）；引用面 = `:168`[C3-3]「② H2 = **P1** 对 tuned 的增量」+ `执行总纲v2.md:91`（PAT-2 status 行「P1 已机制化」）。**新名当前仅存于** `REPOSITORY-MAP §3`（3 文件建议态）。**风险：中**（canon C3′ headline 措辞，语义敏感、footprint 小） | **canon 三总纲（定义层）+ schema（新注册条目）** | ① canon `:121` 把「P1 宽 LMUL 分组」→「`SCHED-WIDE-LMUL` 宽 LMUL 分组」；`:168` 的「P1 对 tuned」需消歧（此处 P1 = pattern-library 宽 LMUL，非 N-operand）。② schema 新增一条 `SCHED-WIDE-LMUL` pattern row（status 按实况：宽 LMUL 旋钮已在 repack gearbox，见 memory `repack-winA-always-mf2`——**须核 landed 状态再定 status**，勿凭空 `mechanized`）。**canon 级 = 必问用户**（改 C3′ headline 措辞 + NG 定义）。 |
| **A-2** | **P1-(b) N-operand 构造统一** | **`CONSTRUCT-N-OPERAND-ROUTE`**（已定名·deferred 裁） | **代码侧裸「P1」已退役（2026-07-12·`e3deea82`·behavior-preserving 仅注释）**：`RVVContractionRouteIdentity.cpp`(3 处) + `RVVMonolithicBlockDotFamily.h`(1 处) 已改「N-operand generic ABI-order / descriptor refactor」，**HEAD 复核 `\bP1\b` in-code = 0 命中**。剩：canon `执行总纲v2.md:91`「P1 已机制化(4 构造 front door)」仍裸标 + schema 未注册。**风险：低**（代码已清·仅剩 canon 一处 + schema 注册） | **canon（1 处措辞）+ schema（新注册条目）** | ① canon `执行总纲v2.md:91` 的「P1 已机制化」消歧为「`CONSTRUCT-N-OPERAND-ROUTE`（N-operand 构造）已机制化」。② schema 新增 `CONSTRUCT-N-OPERAND-ROUTE` row（status=`mechanized`·4 前门已 landed）。behavior-preserving（无 code 变更）→ **无 byte-exact 风险**。 |
| **A-3** | **P4 布局/repack** + **PAT-2「P4」形式槽** | **registry pattern_id 为唯一真源**：已实现 → 指向 `PAT-S6-*`；未启动形式槽 → 显式记一个 **pending pattern_id**（新名待定·§3.2 建议「registry pattern_id」但未给具体字面 → **标待定·需用户裁具体 id**） | canon 面：`科研目标总纲v2.md:121`（定义「P4 布局/repack」）+ `:168`[C3-3]「④ **P7 或 P4** 的 prefill 赢」+ `执行总纲v2.md:91`（「P3/P4 未启动」）+ `:282`（C3′ 依托表列 PAT-2）+ `:289`（缺失表列 PAT-2）。**代码侧无可安全消歧 footprint**：`SP4`（`RVVLowerQuantContraction.cpp`/`RVVRepackTilingSelection.h`/`执行总纲v2.md:246`）= output-tiling 选择变体 = **已= 机制化 `PAT-S6`**、命名一致、**非** pattern-library「P4」——**不改**（`grep P4\b` 命中 `SP4` = 正则边界假阳性·勿误伤）；`MANIFEST.md` 的 `[PAT-S6]` 引用正确·不改。**风险：中**（canon 多点 + 需裁 pending id 字面） | **canon 三总纲（多点定义 + C3′ headline）+ schema** | ① 已实现 repack 机制 = `PAT-S6-*`（已在 schema）→ canon 谈「已机制化 P4」处改指 `PAT-S6`。② 「未启动的 P4 形式槽」→ 显式记 pending pattern_id（**具体字面待用户裁**·§3.2 只给「registry pattern_id 为真源」原则、未给字面）。③ `:289` 缺失表的 PAT-2 项需同步。**canon 级 + 待定名 = 必问用户**。 |
| **A-4** | **canon 散文 P2/P2b/P3/P5/P6/P7/P8**（PAT-2 其余槽） | **registry pattern_id**（新名**全部待定**·§3.2 只裁「以 registry pattern_id 为唯一真源、退役 canon 散落裸标」原则·未逐一给字面） | `科研目标总纲v2.md:121`（定义行全 8 槽）+ `:166`[C3-1]「模式库 **P1–P8**（含 P2b）」+ `:168`[C3-3] + `:170`[C3-5]「P1/P2 条目与现有 pass 一一对应」+ `执行总纲v2.md:91`（逐槽 status）。**风险：中-高**（触 C3′ headline「P1–P8」这一**贡献级措辞**·若逐槽改字面则 headline 需重写） | **canon 三总纲 C3′ headline（定义层·贡献级）+ schema** | **★决策点**：是否把全 8 槽都改 registry pattern_id？若是 → C3-1「模式库 P1–P8」headline 需整体重述 + schema 补齐 8 条。**这是 A 组最大的 canon 级裁决**，**必问用户**：范围（只收已碰撞的 P1/P4，还是全 8 槽统一）+ 各 pending 槽 status。**建议独占会话先只收已碰撞项（A-1/A-2/A-3），全槽统一另裁**，避免 headline 大改混入机械替换。 |
| **A-5** | `pattern-registry.v1.json` `$meta.authority` 路径 `docs/TianChen-RV_科研目标总纲v2.md#PAT-1` | 修正为 `docs/canon/TianChen-RV_科研目标总纲v2.md#PAT-1` | **1 文件 1 行**（schema 内部指针缺 `canon/` 段·真源实为 `docs/canon/…`）。同类：`$meta.note` 内 `[docs/TianChen-RV_科研目标总纲v2.md]` 亦缺 `canon/`。**风险：低**（纯指针修正·非语义） | **schema（doc-指针修正）** | 顺带收（同 schema 文件·同独占会话）。非 pattern_id 改名·纯路径一致性。**非 canon 级**（不改红线/定义）。 |

### A.2 A 组执行序建议

1. **A-2 先行**（代码已清·风险最低）：canon `执行总纲v2.md:91` 一处消歧 + schema 注册 `CONSTRUCT-N-OPERAND-ROUTE`（mechanized）。
2. **A-1**：canon `:121`/`:168` + schema 注册 `SCHED-WIDE-LMUL`（**先核 landed status**）。
3. **A-3**：canon 多点「P4」分流（已实现→PAT-S6·未启动→pending id 待裁）+ schema。
4. **A-5**：schema 内 authority 路径修正（顺带）。
5. **A-4（全槽统一）= 独立裁决**，不与 1–4 机械替换混做（避免 C3′ headline 大改污染 byte-exact 审计）。

### A.3 A 组 byte-exact 验证点 + CI 门

- **byte-exact 风险 = 零**（A 组全是 doc/schema 文本 + 注释·**无 kernel 算术变更**）。A-2 代码侧已在 `e3deea82` 证 behavior-preserving（build+lit 绿）。
- **CI 门清单**（独占会话须全绿）：
  - `check_docs_canon`（canon 白名单 + reports append-only）。
  - **schema 相关门**（改 `pattern-registry.v1.json` 触发）：核 `.trellis/scripts/check_schema_gate.py`（[F-2′] `schema-def-gate`）是否把 `pattern-registry.v1.json` 纳 schema.def 监控面——**若纳入，新增 pattern row 会触 F-2′「家族接入 PR ∩ schema.def = ∅」**：需确认 pattern 注册**不属**「家族接入 PR」豁免语境（pattern registry 是 C3′ 证据镜像·非家族接入），否则 F-2′ 误红。**执行前必核**。
  - `retired-index-gate` / `cert-requirements-gate`（若 pattern row 带 metrics_hook 指向 cert）。
  - **MOVES 映射**：A 组无文件搬家（同文件内改名）→ 只需在独占会话 commit body 记「P1-(a)→SCHED-WIDE-LMUL / P1-(b)→CONSTRUCT-N-OPERAND-ROUTE / P4→PAT-S6+pending」旧→新映射一行，供历史 dated 报告内旧标引用回溯（历史报告 append-only·不回改）。

---

## B. 方言前缀（`tcrv_rvv` / `tcrv.` / `tcrv-opt` / dialect namespace）—— ★最大 footprint·风险最高

**改名目标 = 待定·需用户裁**（[RENAME] deferred·无已裁新前缀）。以下**只精算现状 footprint**，供独占会话按用户定的新前缀套算。

### B.1 方言真源（8 个 mnemonic + C++ namespace 根）

| Dialect .td | mnemonic (`let name`) | cppNamespace |
|---|---|---|
| `Dialect/Exec/IR/ExecOps.td` | **`tcrv`**（核心/base 方言） | `::tianchenrv::tcrv::exec` |
| `Dialect/RVV/IR/RVVOps.td` | `tcrv_rvv` | `::tianchenrv::tcrv::rvv` |
| `Dialect/IME/IR/IMEOps.td` | `tcrv_ime` | `::tianchenrv::tcrv::ime` |
| `Dialect/Offload/IR/OffloadOps.td` | `tcrv_offload` | `::tianchenrv::tcrv::offload` |
| `Dialect/Scalar/IR/ScalarOps.td` | `tcrv_scalar` | `::tianchenrv::tcrv::scalar` |
| `Dialect/Template/IR/TemplateOps.td` | `tcrv_template` | `::tianchenrv::tcrv::template_ext` |
| `Dialect/TensorExtLite/IR/TensorExtLiteOps.td` | `tcrv_tensorext_lite` | `::tianchenrv::tcrv::tensorext_lite` |
| `Dialect/Toy/IR/ToyOps.td` | `tcrv_toy` | `::tianchenrv::tcrv::toy` |

**★关键歧义供裁**：「方言前缀」有两个层级——(a) **共同根 `tcrv`**（= Exec 核心方言 mnemonic **且** 是全部子方言 `tcrv_*` 的公共前缀 **且** C++ 中层 namespace `tcrv`）；(b) 各**子方言全名**（`tcrv_rvv` 等）。改名若动 (a) 根 `tcrv` → 波及全部 8 方言 + C++ 中层 namespace + 发射符号前缀 `tcrv_emitc_*`；若只动某子方言 → 局部。**用户须先定：改根前缀还是逐方言、新根字面。**

### B.2 footprint 精算（HEAD 6f9099e5·全仓 git grep）

| 记号 | 命中行 | 文件 | 顶层分布（行·前 5） | 风险 |
|---|---|---|---|---|
| **`tcrv_rvv`** | **150,392** | **3,256** | `.trellis` 97,778 · `test` 19,875 · `artifacts` 18,497 · `lib` 4,774 · `experiments` 3,150 · `scripts` 2,962 · `tools` 2,195 · `include` 932 · `schema` 211 · `docs` 14 | **极高** |
| **`tcrv.`**（op 前缀·含 `tcrv.dialect`） | 11,218 | 2,589 | 分布未逐拆·主体在 test .mlir + experiments | 极高 |
| **`tcrv-opt`**（工具二进制名） | 5,760 | 2,167 | 主体 test/.trellis/lit RUN 行 | 高 |
| **C++ 中层 namespace `tcrv`**（`tianchenrv::tcrv::*`） | 见 C 组 `TianChenRV`/`tianchenrv` 计数 | — | — | 高 |

**code-bearing 目录（真替换核心区·`tcrv_rvv` 单记号）**：`lib/` 65 文件/4,774 行 · `include/` 22 文件/932 行 · `test/` 802 文件/19,875 行 · `tools/` 17 文件/2,195 行 · `schema/` 7 文件/211 行。

**分层解读（供独占会话分批）**：
- **真源层（必改·小）**：8 个 .td 的 `let name` + cppNamespace（16 处）+ `.inc` 生成物（重生·不手改）。
- **消费层（必改·中）**：`lib/`+`include/`+`tools/` C++ 引用 + `test/` .mlir 语料 + lit RUN。
- **证据/journal 层（大宗·须裁是否改）**：`.trellis/`（97,778 行·task journal）+ `artifacts/`（18,497·多为生成物/objdump）+ `experiments/`（sealed MANIFEST + cert-lineage）。**这些多是 append-only 存档 / 机生**——独占会话须裁：**sealed 证据内的旧方言名是否回改？**（建议**不回改 sealed**·同 F-1/P1 先例「历史 dated 报告 append-only 不回改」，只改活代码 + 活语料 + 机生重生）。

### B.3 B 组 byte-exact 验证点 + CI 门（★风险核心）

- **byte-exact 双档区分（关键）**：
  1. **数值输出 byte-exact（必须零漂移）**：kernel 算术不变 → dequant/GEMM 数值 oracle 比对必须 **0 diff**。方言改名是**纯文本/符号重命名**·不改算术 → 数值应恒等。**验证 = 全 byte-exact gate 重跑 = 0 mismatch**（见 memory `build-incremental-unreliable`：须 forced/clean rebuild + BEFORE/AFTER-EQUALITY）。
  2. **发射符号/IR 文本指纹（会变·须 re-baseline + MOVES）**：发射的 C kernel 函数名嵌 `tcrv_emitc_*`（79,800+ token 命中）→ 改前缀会改**发射符号名** + 所有 golden .mlir 的 `// CHECK: tcrv_rvv.…` → **golden 全量重生**·指纹类 baseline（`schema/f5-failclosed-baseline.v1.json` 等）须 re-baseline。**MOVES 映射必需**：旧符号↔新符号，供 cert-lineage / sealed 证据回溯。
  3. **sealed cert 数值指纹（须核是否 name-sensitive）**：IME cert `0xe210312b`（vmadot int32 0-diff）等 = **输出算术哈希**·应 name-invariant → 改名后应不变；但若某指纹哈希了 IR 文本/符号名则会变 → **独占会话须逐 sealed cert 核「指纹源 = 数值 vs 文本」**，数值型不得漂移、文本型登记 MOVES。
- **CI 门清单（独占会话须全绿）**：
  - **全量 lit**（904+/907·方言名遍布 CHECK 行·改名后必重生 golden 再跑）。
  - **build [82/82 target]**（.td mnemonic 改 → ODS 重生 → 全 dialect lib 重链）。
  - `f1-zero-branch-gate`（`schema/family-regex.v1.json` 的 family-regex 若锚定 `tcrv_rvv` 字面 → 须同步改 manifest）。
  - `f5-failclosed-fuzz` / `cert-requirements-gate` / `retired-index-gate` / `monolith-retire-gate` / `frontdoor-provenance-gate`（凡 schema 内含 `tcrv_rvv` 字面的 7 个 schema 文件——`coverage-sixstate` / `emit-bypass-whitelist` / `family-regex` / `monolith-retire-whitelist` / `pattern-registry` / `retired-index.generated` / `tiling-measurements`——都要同步·**其中 `retired-index.generated.json` 机生·重生勿手改**）。
  - `schema-def-gate`（[F-2′]·方言改名**改 schema.def** → 会触门·须确认这是「定义层机械变更」豁免语境·**执行前必核 F-2′ 判据是否误伤**）。

---

## C. 项目改名（`TianChenRV` / `TianChen-RV` / `tianchenrv`）

**改名目标 = 待定·需用户裁**（[RENAME] deferred·`定位-v2.md:71` 明确「命名统一另开专会话」·全仓无已裁新项目名）。以下只精算现状。

### C.1 逐面 footprint

| 记号 | 命中行 | 文件 | 说明 | 风险 |
|---|---|---|---|---|
| **`TianChenRV`**（C++ namespace 根 + include 路径段 + CMake target 前缀） | 4,562 | 1,538 | 扩展名文件：md 983 · jsonl 220 · cpp 169 · h 74 · json 38 · td 6 | 高 |
| **`tianchenrv`**（小写·C++ namespace `::tianchenrv::…`） | 31,747（token 计） | — | 全部 cppNamespace 根 | 高 |
| **`TianChen-RV`**（docs/canon 连字符形） | 1,419 | 1,231 | canon 文件名本身 + 正文引用 | 中-高 |
| **`include/TianChenRV/`（物理目录）** | — | **146 tracked 文件** | 目录 rename → 全 `#include` 路径同步 | 高 |
| **`#include "TianChenRV/…"`** | 1,050 | 244 | 改路径必同步改这些 include | 高 |
| **CMake target `TianChenRV*`** | — | — | `TianChenRVInitAll` · `TianChenRV{Conversion*,…}` · `TianChenRV{Exec,IME,RVV,…}Dialect` 等 | 高 |

### C.2 C 组特有风险点

- **canon 文件名本身含 `TianChen-RV_`**（`TianChen-RV_科研目标总纲v2.md` 等 4 个 charter）：改文件名 = **破全仓指针**（CLAUDE.md / ROADMAP / spec / 数百 dated 报告都引这些路径）。**先例警示**：`check_docs_canon.py` 曾因「rename-to-add-总纲」破指针而**被否**（该文件注释：「Rename-to-add-总纲 was rejected (breaks 全仓 pointers)」）→ **canon 文件名改动风险极高·须 MOVES 映射 + 全仓指针同步·强烈建议单独裁 + 独立批**。
- **include 目录 rename**：`include/TianChenRV/` → 新名 = 146 文件搬家 + 1,050 处 `#include` 同步 + CMake include-dir 路径 + `.td` 的 `include "TianChenRV/…"` 同步。**MOVES 映射必需**（git mv 保 history）。

### C.3 C 组 byte-exact + CI 门

- **byte-exact**：项目改名同样是纯文本/路径重命名·**kernel 算术不变** → 数值输出零漂移（验证同 B.3.1·clean rebuild + 数值 oracle 0 diff）。发射符号若嵌项目名则同 B.3.2 re-baseline。
- **CI 门**：build [82/82]（include 路径 + CMake target 改 → 全量重配重链）+ 全量 lit + `check_docs_canon`（canon 文件名若改·须确保新名仍带 charter marker「总纲/定位/canon/charter」否则门红）。

---

## 全局执行序建议（独占会话·一次做）

> **原则**：**风险升序、canon 级裁决前置**。先把「已定名 + 低 byte-exact 风险」的收掉，再做大 footprint 机械替换，最后做「破指针类」高危项（或拆独立批）。

1. **A 组（pattern 编号）先行**：A-2（代码已清）→ A-1 → A-3 → A-5。**A-4 全槽统一 = 单独 canon 裁决**，不混入机械替换。
2. **canon 级裁决集中问一次**（必问用户·集齐再动）：A-1/A-3/A-4 的 C3′ headline 措辞 + P4 pending id 字面 + B 组新方言前缀（改根 `tcrv` 还是逐方言·新字面）+ C 组新项目名 + canon 文件名是否改。
3. **B 组方言前缀**（用户定新名后）：真源层（8 .td）→ 消费层（lib/include/tools/test）→ 机生重生（.inc/retired-index）→ schema 7 文件同步 → sealed/journal 层**按裁决**（建议不回改 sealed·只改活代码活语料）。**分批**（先真源+消费层过 build+lit·再证据层）。
4. **C 组项目名**（用户定新名后）：**canon 文件名 rename 独立批**（破指针·MOVES + 全仓指针同步）；include 目录 + namespace + CMake 一批。
5. **收尾**：全 CI 门绿 + 全 byte-exact gate 0 mismatch（clean rebuild）+ **MOVES 映射 doc**（旧→新·pattern_id / 方言符号 / 项目路径三张映射表·供 sealed 证据与历史报告回溯）。

## byte-exact 零漂移总则（三组共用）

- **不变量**：本次全部变更 = **文本/符号/路径重命名·零 kernel 算术改动** → **数值输出必须 0 diff**（dequant byte-exact·GEMM/GEVM oracle·IME `0xe210312b` 类数值 cert）。任何数值漂移 = 改名引入了非机械改动 = **STOP + 排查**。
- **文本指纹（会变·非漂移）**：golden .mlir CHECK 行 / 发射符号名 / IR 文本类 baseline **会变·须重生 + re-baseline + MOVES**——与「数值漂移」严格区分，勿混判。
- **验证纪律**（memory `build-incremental-unreliable`）：byte-exact gate 必须 **forced/clean rebuild + BEFORE/AFTER-EQUALITY**·勿信增量 build。

## 待用户裁清单（本表标「待定」的名·集中问）

1. **B 组**：改共同根 `tcrv` 还是逐子方言？新方言前缀字面？sealed 证据内旧方言名是否回改？
2. **C 组**：新项目名字面（`TianChenRV` C++ / `TianChen-RV` docs 两形）？canon 文件名是否改（破指针·高危）？include 目录新名？
3. **A 组**：A-3 未启动 P4 形式槽的 pending pattern_id 具体字面？A-4 是否全 8 槽统一改 registry pattern_id（触 C3′「P1–P8」headline 重写）？各 pending 槽 status？
4. **A-1**：`SCHED-WIDE-LMUL` 注册时 status（须先核宽 LMUL 旋钮 landed 现状·勿凭空 mechanized）。
