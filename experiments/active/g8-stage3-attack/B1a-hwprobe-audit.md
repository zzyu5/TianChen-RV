# B1.① hwprobe fail-closed 编译期事实核实

**审计员**：案头审计员（纯案头·独立板域·零板时·grep/读源）
**日期**：2026-07-15
**指控**：spec/文档多处称 hwprobe 是「load-time / 运行时消费」，但真身可能是【编译期 VariantSelection】。
**裁决**：**部分成立**——底层事实（fail-closed = 编译期）成立；但「文档谎称 hwprobe 现在运行时消费、需直接修正」这一层【不成立】。详见下。

---

## 一、机制核实（file:line 可溯源）

### 1. hwprobe / getauxval / __riscv_hwprobe 在实现层 = grep 0（从未被消费）

```
grep -rn "hwprobe\|getauxval\|__riscv_hwprobe\|riscv_hwprobe" lib/ include/
  → 实现层零命中（唯一命中是 schema 枚举字符串与相关工作对照，非消费点）
```

- `schema/capability.schema.v1.json:47` — `"members": ["hwprobe","cpuinfo","vendor_table","manual"]` = **provenance 枚举值定义**，非探针。
- 全部 `hwprobe` 文本命中集中在：canon 现状审计（自认 grep=0）、相关工作对照面（FMV/IFUNC/hwprobe 作**对照对象**）、未来工作项（D-2a/D-2b/P2）。**没有一处是 lib/include 里的真实运行时探测调用。**

### 2. 能力来源 = 编译期 IR 属性，不是运行时探测

选择 pass 的能力集全部经 `TargetCapabilitySet::buildFromKernelChecked(kernel)` 从 **kernel 的 MLIR 编译期属性**读出：
- `lib/Transforms/VariantSelection.cpp:667` — `runSelection` 入口即 `buildFromKernelChecked(kernel)`。
- 同一入口复现于 `lib/Plugin/ExtensionPlugin.cpp:932/1283/1325/1370`、`lib/Plugin/LoweringBoundary.cpp:383`、各 `lib/Plugin/RVV/FrontDoor/*SourceFrontDoor.cpp`。**全部是编译期 MLIR pass 内调用**，无一走硬件探测。

### 3. fail-closed（I7）确证在编译期

- `.trellis/spec/architecture/core-invariants.md:29-31` [I7]：「无法证明存在合法可执行 route 时 fail closed」——由编译期选择 pass 执行。
- `lib/Transforms/VariantSelection.cpp:701` — `VariantSelectionKind::NoViableVariant`（无可行变体）= fail-closed 终态，在编译期 pass 内判定。
- 各 plugin 的 I7 fail-closed（`RVVCodebookDotSourceFrontDoor.cpp:224/773/798` 等「every candidate pruned → fail-closed (I7)」）**全部在编译期 front-door 内**，输入是编译期 VLEN/能力事实，非 runtime probe。
- `实验总纲v1:51` T1b 已自认：「未知即拒/冲突拒**部分在(编译期)**；……装载期解析记录(待建)」——即项目自己的现状台账已如实记 fail-closed=编译期。

### 4. 「RuntimeDispatch」是编译期决策物化的守卫结构，不是运行时 hwprobe 消费

- `VariantSelection.cpp:692` `case RuntimeDispatch:` → `materializeRuntimeDispatchPlan`（`:851+`）在编译期**发射** `weft.exec.dispatch` op（带 guard cases）。
- `DispatchRuntimeGuard.cpp:210-223` 编译期物化**单个** `dispatch_available` ABI 参、调用方一次性提供（`执行总纲v2:63` [D-2a]）。**无 declared-instance-hash、无解析记录落盘、无 hwprobe 喂入**——运行期/装载期消费半边是**未建的未来工作**（D-2a/D-2b/P2）。

---

## 二、指控逐层裁断

### CONFIRMED（成立部分）
1. **fail-closed 当前实现 = 100% 编译期**，源于编译期 IR 能力事实（`buildFromKernelChecked`），无任何运行时判定。
2. **hwprobe 在实现层 grep=0**，从未被消费；运行时/装载期 dispatch 链（hwprobe→事实→instance-hash→调度表）是**未建的未来工作项**。

### REBUTTED（不成立部分 —— 关键）
**指控设想的「文档谎称 hwprobe 现在运行时消费、需直接修正」并不存在。** 项目的**现状台账（docs/canon/执行总纲v2）恰恰极其诚实**，逐条自认未建：
- `执行总纲v2:22` B-8：「`hwprobe`/`__riscv_hwprobe`/`instance-hash`/`shape-hash` 全 grep=0(lib/include/tools)」
- `执行总纲v2:47` S-3：「探针层不存在(`hwprobe` grep=0)」
- `执行总纲v2:64` D-2b：「运行期 hwprobe grep=0；现有是 build-time 工具链探针，非运行期」
- `执行总纲v2:183`：「③ 运行期归因：缺(无 hwprobe/instance-hash 调度表)」
- `真欠地图.md:56` P2：「REMAINING……provenance=hwprobe/trust=measured 未建·C_attr^RT=0」
- `实验总纲v1:51` T1b、`FALSIFIER-INDEX.md:19/64`、`schema-def-scope-summary.md:63`（「runtime probe layer (hwprobe) does not exist」）——**全线一致自认未建**。

即：**没有任何「hwprobe 现在被运行时消费」的错误现状表述可供修正。**

### 「load-time / 运行期」措辞的性质 —— 是终态契约，不是现状错误
以下 present-tense「load-time / 运行期 / 装载期」措辞**不是现状谎称**，而是 **C1 终态契约 / 论文机制框架**：
- `README.md:28`、`capability-contract.md:18`：「one relation-bearing schema unifies compile-time variant generation with **the runtime dispatch guard**」= 对 FMV/IFUNC 的**可迁移机制框架**（exportable mechanism），明写「a *mechanism*, not a discovery」。
- `index.md:32` [C1]、`科研目标v2:149` [C1-1]：「同一 schema 同时驱动编译期变体生成与 fail-closed **运行期（装载期解析形态起步）**调度守卫」——**自带诚实括注**「装载期解析形态起步」，即已声明这是起步/未全建。
- `capability-model/index.md:26`：「`implies` is a **load-time** transitive closure; `conflicts` is fail-closed」= 描述**设计的语义层**（D-2a 装载期最小解析的终态契约）。
- `weft-exec-contract.md:137`、`capability-contract.md:83`：profile「expands to a normalized fact set at **load-time**」= [D-2a] 的**设计契约**。

按项目纪律（CLAUDE.md：「spec 是给 agent 的**稳定契约**……当前进度/状态属于 tasks/ 和 workspace/，不写进 spec」），spec 描述**终态契约**、现状台账住 canon 现状审计——二者已分工正确、且各自诚实。**这些「load-time」措辞是合法的终态契约，不是待修错误。**

---

## 三、处置

**无需修（且不应机械改）**。理由：
1. 没有「hwprobe 现在运行时消费」的错误现状表述可修——现状台账（执行总纲v2）已全线诚实自认未建。
2. spec/README/canon 里的「load-time / 运行期 dispatch guard」是 **C1 终态契约 + 论文机制框架**（且多处已带「装载期解析形态起步 / 待建」诚实括注）。把它们机械改成「编译期事实」会：(a) 破坏 C1 头牌终态主张（合取 = 编译期生成 ∧ 运行期守卫**同一 schema**）；(b) 违反「spec = 稳定契约、非现状」纪律；(c) 属 **canon 级改动（必问）**，非案头审计员权限内的「如实级修正」。

### 可选微锐化（供主会话裁·canon 级·【必问】不自决）
唯一**边界**候选：`capability-model/index.md:26` 的「`implies` is a **load-time** transitive closure」是全句里最像现状断言、且**唯一不带「起步/待建」括注**的一处（其余 C1 措辞都带诚实括注）。当前 `implies` 闭包实际在**编译期**能力模型 C++ 内算（被编译期 pass 消费），装载期解析（D-2a）未建。若主会话认为需与终态契约其余处对齐诚实括注，可提议改为「`implies` is a transitive closure (compile-time today; load-time resolution per [D-2a] is future)」。**但这是 canon/契约级措辞，需用户/主会话裁，审计员不自决改。**

---

## 四、证据指针汇总
- 实现层 hwprobe=0：`grep -rn "hwprobe\|getauxval\|__riscv_hwprobe" lib/ include/` → 0 消费点
- 编译期能力源：`lib/Transforms/VariantSelection.cpp:667`（`buildFromKernelChecked`）；同款 `ExtensionPlugin.cpp:932/1283/1325/1370`、`LoweringBoundary.cpp:383`
- fail-closed 编译期：`core-invariants.md:29-31` [I7]；`VariantSelection.cpp:701` NoViableVariant；front-door `RVVCodebookDotSourceFrontDoor.cpp:224/773/798`
- RuntimeDispatch=编译期物化守卫：`VariantSelection.cpp:692/851+`；`DispatchRuntimeGuard.cpp:210-223`
- 现状台账诚实自认未建：`执行总纲v2:22/47/64/183`、`实验总纲v1:51`、`真欠地图.md:56`、`FALSIFIER-INDEX.md:19/64`
- 终态契约措辞：`README.md:28`、`index.md:32`、`科研目标v2:149`、`capability-model/index.md:26`、`weft-exec-contract.md:137`、`capability-contract.md:83`
