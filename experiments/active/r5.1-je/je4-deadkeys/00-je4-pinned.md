# JE4 · 零消费键接上或删净 — pinned 交付

worktree base pin = `1d207aab58178c1fbcbe8d09ac189532913b182a`（核对 == HEAD ✓）
census 源 = pkg2-c-capability.md（其 pin=d173f4c2·本 track 全部在 1d207aab 重核）

判据（只减不增·别误删活键）：**真删 = 既无 writer 又无 reader**（纯占位·永不载数据）。
**保留 = 有产出/有设计契约/证据线 I8 by-design 不被 selector 读**。

---

## A. 删净（3 项·genuine dead — 既无 writer 又无 reader）

改的源文件：`include/Weft/Plugin/RVV/RVVCapabilityProfile.h`（-36 行·纯删·byte-exact 安全）

| # | 删了什么 | 删前 grep=0 证据（1d207aab 重核） | 为何真死 |
|---|---|---|---|
| 1 | `RVVProbeCapabilityFacts::cachelineBytes`（字段+注释 65-71） | `git grep 'cachelineBytes' -- lib include tools test \| grep -v _attic` → **仅 h:71 声明**。**无任何 `.cachelineBytes =` writer**（连 probe 都不填）·无 reader | 意向消费者 RVVRepackTilingSelection.h:438 原文自述"a cacheline/L1d-SIZE board capability is a known GAP; once plumbed"——即**明确未接线**。半座断桥 |
| 2 | `RVVProbeCapabilityFacts::imePresent`（字段+注释 72-77） | `git grep 'imePresent' -- lib include tools test \| grep -v _attic` → **仅 h:77 声明**。无 writer·无 reader | **特别核 IME dispatch 通过**：IME 真实检测 = `march.contains(kIMEMarchToken)`（IMEExtensionPlugin.cpp:323），**不读此字段**。imePresent 与 IME 门完全无关·零关联 |
| 3 | `deriveIMEPresent(march,hints)` inline fn（注释块+函数 178-199） | `git grep 'deriveIMEPresent' -- lib include tools test \| grep -v _attic` → 仅 h 自身。**零调用者** | 复制了 IME token 逻辑（`xsmtvdotii`）但无调用点·IME 门自持 kIMEMarchToken。纯 dead dup |

**删后自证**：
- 三键 `git grep`（working tree）= **0 code refs**（clean）。
- 增量 build 绿：`ninja WeftRVVPlugin weft-opt weft-rvv-capability-profile-divergence-test` → 71/71 链接成功（plugin lib + weft-opt + gtest 全过）。
- gtest 绿：`./bin/weft-rvv-capability-profile-divergence-test` → "RVV N1 probe->capability divergence-axis derivation tests passed"。
- 退保：字段全 by-name 赋值（`facts.architecture=...`）·**无 positional brace-init**（`git grep 'RVVProbeCapabilityFacts{'`=0）→ 删字段不破任何构造点。

可复跑（在 1d207aab worktree）：
```
git grep -n 'cachelineBytes\|imePresent\|deriveIMEPresent' -- lib include tools test | grep -v _attic
cd build/weft && ninja WeftRVVPlugin weft-opt weft-rvv-capability-profile-divergence-test
./bin/weft-rvv-capability-profile-divergence-test
```

---

## B. 保留 + 理由（**别乱删**·census 其余零消费键都不是死料）

| census 组 | 键 | 为何保留（不删） |
|---|---|---|
| pkg2 口径D 9 键 | provider-property：vlenb:bytes / clang:version / cmake:version / compile_run{selected_march,selected_mabi,source_sha256,binary_sha256} / march:value / mabi:value | **实产出**到 in-IR provider op（RVVCapabilityProfile.cpp:562-605 真发射·带活数据）·= **I8 证据线/provenance**·"zero-consumption-by-selector"是**设计所致非缺陷**（证据不参与 compute）。删=截肢已设计的 capability-provenance 子系统=must-ask 架构决定·非单边死键清理。**weft-opt 实测该路径完好**（materialize pass 正确保留 `rvv_probe_compile_run{selected_march="rv64gcv"}`）。注：`selected_march` 作为 attr-name 另有 dialect 消费点（RVVDialect.cpp:2108,2124 属性过滤）——**并非全零**，census "compile_run 零消费"仅指"无 selector getProperty 读"·非"全无引用" |
| pkg2 口径A 7 namespace | schema params：vlen/elen/sew_set/lmul_budget/vreg_count/cacheline/ime.tile | `schema/capability.schema.v1.json` `$meta`(6) 自述"declares TARGET shape, NOT a snapshot of current code"+"code and schema.def are **intentionally divergent**"。这是**有意的目标形前置声明/conformance-gap 追踪**·非死料。删=删设计目标 |
| pkg2 口径B declared-no-producer 3 | fact_record：subclass / provenance / trust | 同上 schema TARGET-shape·且 provenance/trust = **I8 证据线**字段。有意前置声明·保留 |

---

## verdict

**GREEN（删净子集）**：3 键删净·各带 grep=0 + build/test 绿证据（真减 -36 LOC·census 减 3 条零消费项）。
**保留子集诚实标注**：19 键（9 provider-property + 7 schema-namespace + 3 fact_record）= I8 证据线/schema TARGET-shape·非死料·**别误删**（含 selected_march 有 dialect 消费点·census "零消费"口径仅限 selector-getProperty）。

改的共享文件（供主会话合并）：`include/Weft/Plugin/RVV/RVVCapabilityProfile.h`（纯删 3 dead 项·byte-exact·-36 行）。
不 git commit·git add 仅纳该源文件。
