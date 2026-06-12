# Dead-Mirror Removal Guide

去伪（strangler-fig 删除死配重）前用。RVV family 里"字符串机器"夹着大量
metadata-mirror / route-validation 脚手架；它们和**活的发射器**同名前缀、同文件、
逐行交错。删错一个就截肢了真链路。删之前走这张表。

## 先分清：活发射器 vs 死镜像（命名前缀陷阱）

| 形态 | 判定 | 处置 |
|---|---|---|
| `get*RouteFacts` / `add*RouteFamilyMetadataMirrors` / `materialize*` / `build*` 真 EmitC ops | 构造**真 IR / 真产物**，在 export 路径上有活 caller | **保留**（I5 typed body / I8 产物） |
| `get*/build*RouteValidationContract`、`get*RouteMetadataMirrorContract[Set]`、`append*Validation*`、`*RouteValidationKind` | 把发射器输出**再拼一遍**对镜像做等值断言 | **删除**（I4 病灶：mirror 当 authority） |

名字只差一个词（`add*MetadataMirrors` 活 vs `get*MetadataMirrorContract` 死）。
**靠 caller 图判定，不靠名字。**

## 死镜像脚手架的结构 tell（I4）

- [ ] 注释/字段自述是 "consistency mirror" / "copies … the authority is X"？→ 镜像，删。
- [ ] 它重建一段 C 表达式/步骤数/intrinsic 名再 `== ` 断言（验"发射器没改"而非语义）？→ 删。
- [ ] 它的**唯一历史消费者**是已删的 golden validator / golden test？→ 死，删。
- [ ] 它是 fail-closed 但出口只 `signalPassFailure` / 报错（路径已 disabled）？→ I7 债务残留，删。

## 安全删除方法（closed-graph + full-link）

1. 枚举整个簇，grep 全树确认 top-level 入口 **0 外部 caller**（def+decl 通常=2 refs）。
2. **按 symbol 删，绝不按行号删**（与活代码交错；行号边删边漂，重新 grep）。
3. 入口删完后用 `-Wunused-function` + **从头 full link** 逐层收掉新暴露的私有 static，
   直到 0 警告。link 是安全网：漏删的活引用会变 undefined-reference 把你拦住。
4. 任何 symbol 删了导致 link 报 undefined → 它不是死的，**还原 + 报告**，别硬删。

## 不可触碰的边界（删之前先确认不在 diff 里）

- I8 export 路径：`exportMaterializedRVVEmitCToCpp` / `compileRVVGeneratedSourceToObject`
  / `mlir::emitc::translateToCpp` 委托 / `candidateValidationFn` 链 /
  `buildRVVSelectedBodyHeaderMetadataEvidence`（活的 header 证据表，进真 .h）。
- I7 guards：`containsForbiddenDirectCMarker` / `rejectForbidden*`（拒非法 metadata，留）。
- 活 capability model：`CapabilityModel`（legality/cost/dispatch 多处消费）。
- 删完 build 必须绿 + lit 不差于 baseline（删的是死码，行为应逐字不变）。

## ⚠️ 验证纪律：lit 计数前先 clean rebuild

incremental `ninja` 可能在改了 `lib/Target/RVV/*` 后**没重链 tcrv-opt/tcrv-translate**，
留下 stale 二进制——lit 跑出来是**假绿 / 假 baseline**。本项目真发生过："3 reds" 的
baseline 是 incremental build 的幻觉，clean rebuild 同一 source 实为 46 reds。
- 信任任何 lit pass/fail 计数前，先确认 build 已全链（改了发射器/验证器尤其要 from-scratch）；
  定 baseline 用 clean build 的真实数，别照抄上一轮 agent 报的数。
- 错误**文案**漂移（如 golden validator 被删后验证器换措辞）会让 STALE-*/golden CHECK 假红
  ——那是 stale 测试不是回归；同步措辞时保留 key+expected+actual，别弱化成 tautology。

## Rule

引用 core-invariants **I4**（metadata 是 mirror 不是 authority）、**I5**（可执行事实在 typed
body）、**I7**（fail-closed 债务=删除目标）、**I8**（产物路径需真证据）。与具体 spec 冲突以
spec 为准。
