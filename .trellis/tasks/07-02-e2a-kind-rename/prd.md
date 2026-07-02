# E2a — op-attribute `kind` → `target_kind` / `region_kind` 消歧

> **父 program:** `07-02-full-refactor`(能力驱动 RISC-V 执行层成熟化)。本任务 = 证据轴首批的**第一个真 module**(E0 已作 quick-commit 落地)。**E2a 须先于 E1(schema.def v1)**——避免契约文档写消歧注记 + 减 [F-1] falsifier grep 假阳性面。
> **canon 依据:** commit 7d781994 "op-attribute kind → target_kind/region_kind (distinct from capability-fact kind); code-only rename, rides E2, before E1"。spec 契约 `.trellis/spec/core-dialect/tcrv-exec-contract.md:95,133-137`。
> **research(已完成,持久化):** `research/kind-rename-map.md`(逐 occurrence 主表)+ `research/kind-rename-scope-summary.md`(scope/门/编辑序)。锚点已 spot-check 对齐 HEAD。

---

## 目标(单句)

把 exec dialect 里**两个不同语义却同名 `kind`** 的属性消歧:**能力事实闭合枚举 `kind`([S-1] isa_ext/sub_ext/uarch/policy)保持 `kind` 不动**;**op 分类轴 `kind`**(target 的 profile/provider 分类、region 的分类)rename 成 `target_kind` / `region_kind`。**纯代码机械 rename、行为不变、导出产物字节中性**——这是过门的判据,不是副作用。

## 为什么(挂条款)

- [F-1] 零分支 falsifier 用 grep 判家族分支;两个同名 `kind` 增加假阳性面。消歧后 grep 更干净。
- [S-5]/[E1] schema.def v1 要把能力事实 `kind` 固化成冻结契约;若 op 分类 `kind` 仍同名,契约文档得写一堆消歧注记。**先 rename 再固化。**
- [L-2]/[GOV] 命名即契约:op 分类轴与能力事实轴同名是债,趁重构窗口清。

---

## Scope(有界;research §1 已核 KEEP vs RENAME 计数)

### IN(要改)
- **Region(`region_kind`,ODS-first)**:`ExecOps.td:224` `$kind`→`$region_kind`(唯一 ODS 改动;**regenerates `ExecOps.{h,cpp}.inc`**)+ 拆常量加 `kRegionKindAttrName("region_kind")` 并 repoint `ExecOps.cpp:948,950` + region 侧 fixtures。
- **Target(`target_kind`,纯 C++ 常量 + fixtures,不动 ODS,regenerates 无 `.inc`)**:加 `kTargetKindAttrName("target_kind")`;repoint `ExecOps.cpp:{321,591,596,602,730}` +(prose 措辞)`612,676`;`CapabilityProviderComposition.cpp:{204,229}` +(prose)`83`;`CapabilityModel.cpp:{257,295}` 字符串字面量;**`isCoreCapabilityAttribute`(CapabilityModel.cpp:45)必须 add `"target_kind"`(保留 `"kind"`)**。
- **shared 常量拆分**:`kKindAttrName("kind")`(ExecOps.cpp:36 + CapabilityProviderComposition.cpp:16)当前一个常量服务三种语义 → 拆成 `kKindAttrName`(capability,保留)/ `kTargetKindAttrName` / `kRegionKindAttrName`。
- **fixtures(~17 行 / 6 文件)**:`test/Dialect/Exec/verify.mlir`、`basic.mlir`、`capability-relations-attr.mlir`、`test/Support/CapabilityModelTest.cpp`、`test/Transforms/ExecutionPlanning/execution-planning-pipeline-offload.mlir`、`test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-macc-add.mlir`(**仅** INPUT target l.9→`target_kind`;PLAN/HEADER 输出 l.47,48,69,70 **保持字节不变**)。精确行号见 research §6。

### OUT(禁动)
- **能力事实 `kind`(KEEP)**:`CapabilityOp $kind`(ExecOps.td:112)、`CapabilityOp::verify`(ExecOps.cpp:630,632)、descriptor accessor `getKind()`(CapabilityModel.h:41 / ExtensionPlugin.h:49)、所有 `capability.getKind()`、`TensorExtLiteSourceFrontDoor.cpp:172`(写 CapabilityOp)、所有 `tcrv.exec.capability … kind =` fixtures。
- **扩展 op-kind(meaning-C,~250+ 处)**:RVV/IME `.td` + `lib/{Dialect,Plugin,Conversion}/RVV/**` —— **绝不触碰**。
- **⚠ 升级项 #1(语义清理,明确 out-of-scope,另起 task)**:今天 `getCapabilityProviderKind`/`makeDescriptor` 把 **target op 分类值** 塞进 `CapabilityDescriptor.kind`(能力事实轴);spec `:133` 说 profile-provider 携 `provides` 非 capability-fact `kind`——**存在张力**。**本任务只做机械 rename、保留现数据流(读新名、仍塞进 descriptor.kind)、字节中性;provides-vs-kind 语义清理留独立 out-of-scope task。绝不在此改数据流。**

---

## 编辑序(research §6;region ODS-first,target 原子落地)

**A. Region(ODS-first):** ①`ExecOps.td:224` `$kind`→`$region_kind` → ②**forced/clean rebuild** 让 `.inc` 重生 → ③拆常量加 `kRegionKindAttrName`、repoint `ExecOps.cpp:948,950` → ④fixtures `verify.mlir:16,401`+`:391` expected-error、`basic.mlir:65`+`:62` CHECK-SAME。

**B. Target(C++ 常量+fixtures,§4 原子):** ⑤加 `kTargetKindAttrName`、repoint 上列 target readers + prose、`CapabilityModel.cpp:257,295` 字面量、**`isCoreCapabilityAttribute` add `"target_kind"`** → ⑥`kKindAttrName("kind")` 留给 CapabilityOp、descriptor `getKind()` 全不动 → ⑦target fixtures(含 export golden **仅** INPUT l.9)。

**C. Gate:** ⑧forced/clean rebuild + 跑 6 依赖测试(research §3)+ 全 exec/capability lit suite;确认 export golden `explicit-selected-body-artifact*` 的 PLAN/HEADER **字节不变**(byte-neutral = 过门判据)。⑨diff 应只落在 `Dialect/Exec/`、`Support/CapabilityModel.*`、6 个测试文件(+ 重生的 `Exec*.inc`);grep diff 确认没碰 250+ meaning-C RVV/IME 处。

## 验收(可机检)

1. **byte-exact 门过**:forced/clean rebuild;`explicit-selected-body-artifact-scalar-broadcast-macc-add.mlir` 的 PLAN/HEADER 输出(`;kind=profile` 由 emitter 硬编码,不随源属性名变)BEFORE==AFTER 字节相等。
2. **lit 全绿**:`test/Dialect/Exec/{verify,basic,capability-relations-attr}.mlir`、`test/Support/CapabilityModelTest.cpp`、`execution-planning-pipeline-offload.mlir` 全过;expected-error 文本随常量/措辞更新。
3. **消歧完成**:`tcrv.exec.target` / region op 的分类属性在 IR/ fixtures/ C++ reader 中统一为 `target_kind`/`region_kind`;能力事实 `kind` 一字未动。
4. **零越界**:diff 不含任何 RVV/IME 扩展 op-kind 改动;不改任何数据流(升级项 #1 未动)。
5. **byte-exact 门必须 forced/clean rebuild**(project memory `build-incremental-unreliable`:此树 `ExecOps.cpp.inc` 每次重生、`tcrv-opt` 有时不重链;别信增量、别钉 stale 绝对指纹)。

## Out of Scope

- 升级项 #1 provides-vs-kind 语义清理(数据流改动)——独立 task。
- E2b 目录归拢(单列 module,解锁 F-3)。
- E1 schema.def v1(E2a 之后)。

## Technical Notes

- research 已核 HEAD 锚点(prior audit 快照已 stale;逐条 re-grep)。in-scope C++ = 11 read/verify 处、3 核心文件(`ExecOps.cpp`/`CapabilityProviderComposition.cpp`/`CapabilityModel.cpp`)+ 1 常量拆分 + 1 ODS 行 + ~17 fixture 行;**零 builder/setter**(target/region kind 只被读、值只从 parsed IR/fixtures 来)。
- 无 build/CI/falsifier 脚本依赖属性拼写(已查 `.trellis/scripts/`、`scripts/`);spec 的 [F-1] 引用是概念性的。
