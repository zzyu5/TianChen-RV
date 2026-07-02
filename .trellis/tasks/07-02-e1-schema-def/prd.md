# E1 — schema.def v1 声明工件 + [F-2′] 操作门

> **父 program:** `07-02-full-refactor`。证据轴 M1 核心地基;**E2a 已解锁**(op 分类 `kind` 已消歧成 `target_kind`/`region_kind`,能力事实 `kind` 现无歧义)。E1 解锁 **E5/E6**(C_construct 度量尺子,先于引擎线主战场 G1)。
> **权威:** 科研总纲 [S-5](六项声明工件)/[S-6](两级门)/[F-2′];执行总纲 §4(六项反写底账)。canon 7d781994(kind 闭合枚举 + subclass;profile=provider-only)。
> **research(已完成,持久化):** `research/schema-def-six-items.md`(六项 HEAD-核实现形态 + target)、`schema-def-format-options.md`(格式 a-d 权衡)、`f2prime-gate-design.md`(两级门)、`schema-def-scope-summary.md`(MVP + 落地序 + 8 决策)。

---

## 核心 reframe(决定整个任务)

schema.def v1 **声明六项的 TARGET 形态**(Reading A,PRD-128 + [S-5] 强制),**不是当前代码快照**。target 字段(provenance/trust/subclass、闭合 kind、命名空间 params、统一角色、可序列化签名)在代码里 **grep=0**——code 与 schema.def **有意分歧**,此分歧 = **被追踪的 conformance gap(非缺陷)**。让代码收敛是后续 E4/E5/G6/P2 的事,**不是 E1**。

## Decisions(ADR-lite;research 8 决策全部定案,采纳推荐默认,可 override)

1. **格式 = (c) 手写 canonical JSON 声明 target**(PRD-128/[S-5] 强制;(d) code-derived checker 是 fast-follow 漂移守卫,非 E1-blocking)。
2. **工件位置 = 专用顶层 `schema/` 目录**(干净的 [F-2′] path 前缀 + 版本日志共置):`schema/capability.schema.v1.json` + `schema/VERSIONLOG.md`。
3. **版本日志/RFC 约定**(仓库无先例,E1 定义):semver `v1.0.0` + RFC-id `SCHEMA-RFC-0001`;VERSIONLOG 每行 `v<semver> | <RFC-id> | <sha256> | <additive|breaking> | note`。
4. **item ⑤ 投影深度 = 最浅**:18 个 `ExtensionPlugin` 虚方法的有序方法名 + arity/签名文本;**排除**序列化 request/result 结构(那会把 K-1 序列化拉进 E1)。
5. **item ⑥ 操作数角色 = 两正交轴**(ABI-signature slot ⟂ body-op 语义角色):证据显示二者部分重合(`accumulator_role`=ABI enum 拼写)但部分不(`mask_role`=body 构造角色,非 ABI slot);**不由 fiat 强行统一**。
6. **status/availability = in-shape**(typed 闭合枚举结构字段,非 per-instance 内容)。
7. **[F-2′] onboarding-PR 识别 = 临时 commit-trailer `Family-Onboarding: <family>`**(headless 可用,无需 PR 平台;label 为备选)——直到 E2b `plugins/<fam>/` 目录落地再切 [F-3] 收容判定。
8. **⭐ scope = 纯声明工件 + 门脚本,不动 C++**:E1 = `schema/` JSON + VERSIONLOG + 门脚本;**不改 `CapabilityDescriptor` 或任何 C++/ODS**(struct 收敛 = G6/E4/E5/P2)。→ 无 build/byte-exact 风险,全 net-new 文件。

---

## Scope(IN;全 net-new)

- **`schema/capability.schema.v1.json`** —— canonical JSON 声明六项 target(逐项确切字段见 `research/schema-def-six-items.md`):
  - ① 事实字段 `{id, kind, subclass, status, availability, provenance∈{hwprobe,cpuinfo,vendor_table,manual}, trust∈{measured,declared}, params, relations}`;provenance/trust/subclass 为 **declared-with-no-producer**(v1 默认 `provenance=manual`/`trust=declared`;producer 是 P2,**现声明是让 P2 落地时不必触 schema.def——否则未来必 [F-2′] 违规**)。
  - ② `kind` 闭合枚举 `{isa_ext, sub_ext, uarch, policy}` + `subclass` 保原始类目。
  - ③ 关系类型表 `{provides, implies(transitive-satisfiable), conflicts(fail-closed)}` 带语义标注。
  - ④ params 命名空间 `{vlen, elen, sew_set, lmul_budget, vreg_count, cacheline, ime.tile(...)}` + 类型。
  - ⑤ 插件接口可序列化签名(18 虚方法名 + arity,最浅投影)。
  - ⑥ 操作数角色词表:两正交轴(ABI-signature slot 枚举 ⟂ body-op 语义角色)。
  - **不入 shape([S-5] 确认排除):** 具体事实行、params 取值、插件内部代码、测量库、模式注册表条目。
- **`schema/VERSIONLOG.md`** —— v1 RFC 行 + v1 SHA256(规范化后)。
- **`.trellis/scripts/check_schema_gate.py`** —— 两子命令:
  - `gate --base <ref> --head <ref>` → [F-2′] path-diff:`git diff --name-only` ∩ `{schema.def 路径}` 必空(仅对带 `Family-Onboarding:` trailer/label 的 PR 触发,exit 非零阻断)。**onboarding PR 要碰 schema.def 即使加一个字段 = falsifier 触发,绝不放行为 "additive"。**
  - `report [--check]` → normalize(`json.dumps(obj, sort_keys=True, separators=(",",":"))`)→ SHA256 → 打印/追加 VERSIONLOG 行;`--check` 在 hash 漂移但无 VERSIONLOG 条目时失败。additive-minor(仅新增 optional 字段/enum 成员/关系型,"extension not modification",semver minor)vs breaking(删/改名/改类型 → major + RFC)由结构化 JSON diff 分级。

## 验收(可机检;无 build,纯脚本+工件)

1. **schema.def 六项齐全**:JSON 含全部六项 target 形态,合法 JSON,provenance/trust/subclass 以 declared-with-no-producer 默认在场;不含任何 in-shape 排除项(具体事实行/params 取值/…)。
2. **report gate 确定性**:`check_schema_gate.py report` 两次运行产出**同一** SHA256(规范化排序稳定);SHA256 写入 VERSIONLOG v1 行。
3. **[F-2′] path-diff 逻辑**:`gate` 子命令用合成 base/head(一个碰 schema.def + 带 `Family-Onboarding` trailer 的 diff)**exit 非零**;不碰的 diff 或无 trailer 的 diff **exit 0**——脚本自带最小 self-test 或 PRD 附验证命令。
4. **additive vs breaking 分级**:结构化 JSON diff 对 {加 optional 字段}判 additive-minor、对{删/改类型}判 breaking——脚本可判并有一例演示。
5. **零越界**:diff 仅落 `schema/` + `.trellis/scripts/check_schema_gate.py`;**不动任何 C++/ODS/lit**(scope 决策 #8);Python 仅 tooling([实现栈]红线)。

## Out of Scope

- **C++ `CapabilityDescriptor` 收敛**(加 provenance/trust/subclass 成员、闭合 ODS kind、命名空间 params)= G6 / E4 / E5 / P2。
- **(d) code-derived conform checker**(diff 当前代码形态 vs 声明 target)= fast-follow,非 E1-blocking。
- **CI wiring**(`.github/` 现不存在)= E3(与 [F-1]/[K-5]/[PAT-3] 一起点亮)。
- **E2b 目录归拢**(切 [F-3] 收容判定)。

## Technical Notes

- 仓库现 `schema.def`/`shape-hash`/`instance-hash`/序列化器/版本日志/RFC 文件/`.github` **全 grep=0**(E1 全 net-new)。唯一在树 SHA256 是 content/binary digest(`sourceSHA256`/`generatedArtifact*SHA256`),非 shape/instance hash;可复用 `hashlib.sha256` idiom。
- Python 门脚本是 governance/workflow tooling → `.trellis/scripts/`(与 task.py 同域),非 `scripts/`(实验 harness 域)。
- 两级门勿混:operation gate(∅,per-PR,[F-2′])与 report gate(shape-hash,per-version,[F-2] legacy 保留)是**不同 PR 类**;additive-minor 是核心作者经 RFC 的**演进 PR**分级,**不是 onboarding 放行漏洞**。
