# E7 — 编译期门未知即拒自足 ([D-1])

> **父 program:** `07-02-full-refactor`。证据轴 M1 门 [D-1]:编译期门 plan 合法 ⇔ 能力谓词在目标 schema 实例下全部可满足;**任何未知 ⇒ 拒绝;该拒绝必须自足**(不依赖调用方/后置 pass 兜底)。守 [I7]"未知=假,缺省拒绝"。**"重构现有"、有界**(advisor:3 站点已钉,跳过 research,直 PRD→implement)。
> **为何要(⚠ 前提已修正,见下):** E7 是让门**自足** fail-closed,使 [F-5] 模糊测试单跑该门/在未验证或程序化构造的 IR 上跑时,未知能力仍被该门自己拒。
>
> **⚠ 前提修正(implement+check 双证,2026-07-03):** PRD 原稿"exec verifier 不 resolve requires 到 kernel set → gap 真实"**factually 错**。`VariantOp::verify`(`ExecOps.cpp:841`,`kernelContainsCapability`,commit adb57f8a 2026-05-06)**已在 parse 期拒绝** kernel-set 未知的 requires 符号(经验复现含"module 有、未 provide 给 kernel"场景)。故 `:150 continue` 在**正常 parse 下不可达**——verifier 先拦。**E7 因此是防御性自足硬化(silent-continue 这个 fail-open → hard reject,使门独立于 verifier 兜底而自足 fail-closed),NOT 活 bug 修复。** 保留理由:capability 门里的 silent-continue 是 latent fail-open([I7] 违背);[D-1]"拒绝自足、不靠后置兜底"里 verifier **正是**一种兜底,门不该依赖它。**F-5 依赖是假设:** F-5(=E3 交付)是否在未验证/程序化 IR 上跑该门决定此硬化是否 load-bearing——接 E3 的 agent 须确认。

---

## 现状:三站点 unknown 语义(HEAD 核实)

| 站点 | 输入 | unknown 行为 | 键 |
|---|---|---|---|
| **CheckCapabilityRequires.cpp:150**(**[D-1] 编译期门**) | 变体 `requires`(IR symbol refs) | **silent `continue`**——不拒、不诊断(**这是 gap**) | symbol(`lookupBySymbolName`,kernel-set 域) |
| VariantSelection.cpp:190(选择期) | 变体 `requires` | `makeSelectionError` 拒 | symbol |
| ExtensionPlugin.cpp:1430(提案校验) | proposal `requiredCapabilityIDs`(ID 串) | `makeVariantProposalError` 拒 | ID(`lookupProviderByID`) |

**核实:** exec verifier(`ExecOps.cpp`)只校 `requires` 良构(FlatSymbolRefAttr/非空),**不**把 requires 符号 resolve 到 kernel 的 `TargetCapabilitySet`——故 kernel-set 里缺席(module 可能有、但未 provide 给该 kernel)的 requires 符号**过 verifier、却在 :150 被静默跳过**。gap 真实。**"符号 vs ID" 非假不一致**:IR requires 天然是 symbol refs、proposal requires 天然是 ID 串,两条数据路各按其输入键控;二者都已拒 unknown,唯独**门本身**(CheckCapabilityRequires)不拒。

## 目标(单句)

让 **CheckCapabilityRequires(编译期门)自足拒绝** kernel-set 里未知的 `requires` 符号(fail-closed),与选择期行为一致;使单跑该门即可拒未知(F-5 不漏)。

## Scope(IN;有界,重构现有)

- **核心改动 `CheckCapabilityRequires.cpp:150`**:`if (!capability) continue;` → 改为 raise 一条新 `RequirementIssue::Kind::Unknown`(记 symbolName),流入拒绝路(`foundRequirementIssue=true` → `signalPassFailure`),给 bounded 诊断(如 "requires unknown capability @X not present in kernel's TargetCapabilitySet")。
- **Unknown 是硬拒、NOT dispatch-guardable**:与 `Unavailable`(可被 dispatch case 的 `runtime_guard_required` 守护)不同——未知能力**不可解析**,不能是合法的 runtime-guarded case;必须无条件拒(这正是 [I7]"未知=假"的自足体现)。确认新 Kind 在诊断/守护判定逻辑里走"不可豁免"分支。
- **更新 `Passes.td` CheckCapabilityRequires description**:原句"Malformed requires attributes **and unknown symbols** remain the responsibility of the tcrv.exec verifier" 中的 unknown-symbols 部分**失效**——改为:该门自足拒绝 kernel-set 未知的 requires 符号;**malformed**(非 symbolRef,:145)仍属 tcrv.exec verifier。
- **一致性**:与 VariantSelection 的 reject-on-unknown-symbol 对齐;**ExtensionPlugin(ID-keyed proposal 路)保持不动**(不同输入、已拒);在改动注释里点明 symbol-vs-ID 是 by-input-type 非需强并。

## 验收(可机检)

1. **门自足**:构造一个变体 `requires @cap_not_in_kernel_set`(kernel 的 provider 域里没有该 cap),**单跑 `--tcrv-check-capability-requires`** → 该门自身 emitError + pass 失败(不依赖选择期/提案期)。新 lit 于 `test/Dialect/Exec/` 或 `test/Transforms/`。
2. **不误伤**:合法(requires 全在 kernel-set 且 available/conflict-free)+ 现有 dispatch-guard 的 Unavailable 豁免路径**不变**;Unknown 不被 dispatch 豁免。
3. **malformed 边界**:非 symbolRef 的 requires 仍由 exec verifier 管(:145 行为不变)。
4. **回归**:forced/clean rebuild + `check-tianchenrv` 全绿;`test/Dialect/Exec/verify.mlir`、CheckCapabilityRequires 相关 lit 绿。**若某现有测试因 unknown 现被拒而失败——那正暴露一个此前静默通过的 latent 案例**:逐个判定是"该测试本就该拒(修期望)"还是"该 case 合法但我误判 kind-set 域"(停下报告,别硬改)。
5. **build 门**(project memory `build-incremental-unreliable`):forced/clean rebuild(`ExecOps.cpp.inc` 每次重生、`tcrv-opt` 有时不重链)。

## Out of Scope

- VariantSelection / ExtensionPlugin 站点(已拒 unknown,不动)。
- malformed requires(verifier 职责)。
- [F-5] 模糊测试脚本本身(= E3 falsifier CI 的交付;E7 只让门自足,使 F-5 可依赖)。
- conflicts 语义(已 fail-closed,不在本任务)。

## Technical Notes

- `RequirementIssue` 结构 + 其消费逻辑(`checkVariant`/诊断/dispatch-guard 豁免判定)在 `CheckCapabilityRequires.cpp:64-131`;新增 `Kind::Unknown` 需在该消费逻辑加"不可豁免、必拒"分支。
- **⚠ 并行写入者纪律**(见 [[full-refactor-program-active]]):动 `Passes.td` 前 re-read(同 branch 有并行写入者 zzyu5,该文件近期被改过);不 commit 对方 in-flight 编辑。
- 纯 C++ 改动,无 Python;behavior-changing(非 byte-neutral)——预期可能有 1-2 处现有测试期望需更新(若有 latent 静默通过案例)。
