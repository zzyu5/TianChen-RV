# A7：已替代生产路径原子退役

## Goal

把已经被 typed front-door construction 取代、却仍残留在 active build/dispatch/verifier 面的生产路径原子退役。当前具名范围是 `schema/monolith-retire-whitelist.v1.json` 的六个 `pending_retirement`：q1_0 flat monolith emitter，以及 q2_K/q3_K/q4_K/q5_K/q6_K monolith op-def 残尾。它们是 RET-1 生效前的已知 violation/debt，不是 whitelist 许可的兼容状态。完成态不保留 deprecated alias、compat bridge、shadow dispatch 或旧 verifier-only op-def。

## Why This Advances the Project

- 把“front door 已 constructed、旧 monolith 仍存活”的双世界收成一条 typed production path。
- 减少 ODS、verifier、family registry 和 emitter 中不符合论文结构的历史表面。
- 让 retired index 成为历史查询，不让 whitelist 变成长期兼容许可。
- 直接落实 [RET-1] 的“渐进施工、每个 slice 原子合入”。

## Current Assets and Facts to Revalidate

- q1_0 source path 已转 typed-flat construction，但旧 recognizer/emitter/dataflow verifier 仍有 live refs。
- 五个 K-quant source path 已构造 typed super-block body；旧 op-def 当前被描述为 pipeline-dead，但仍有 `getOperationName`、family registration 或 verifier test 引用。
- q4_0 与 mxfp4 在 inventory 中是 `deliberately_retained`；本任务审计其 RET-1 保留条件，但不因 monolith 形态误删：q4_0 必须有不重叠的 live lowering slice，mxfp4 必须与普通 production registry/default dispatch/fallback 隔离。
- 上述都是任务启动快照；施工前必须以 HEAD 的完整 caller graph 和 gate 输出复核，不按旧行号操作。

## Atomic Batches

### Batch 1：q1_0 flat monolith emitter

- 证明 typed-flat path 对当前合法 fixtures 与旧 emitter byte-exact/既定 ULP。
- 迁移仍只保护旧 monolith 的 verifier/dataflow tests，使其直接保护 typed body 与新 authority。
- 删除旧 emitter、recognizer、dispatch row、op-def/family registration 及全部 production caller。
- 若发现旧 path 仍有独立合法输入域，停止并登记真实职责；不得以 bridge 保留半迁移。

### Batch 2：K-quant monolith op-def

- 对 q2_K/q3_K/q4_K/q5_K/q6_K 逐项核实 typed super-block path 是唯一实际构造/发射路径。
- 移除残留 op-def、verifier、family op-type registration、`isa/dyn_cast/getOperationName` production refs 与旧 tests。
- source ingestion 可以识别外部格式身份，但不得在 selector/emitter 新增 format-name compute authority；进入 construction 后差异必须由 typed facts/body 表达。
- 五格作为同一 K-quant family batch 提交，避免留下不一致的半套 ODS 表面。

## Cutover Record

每个 batch 必须记录：old symbols、new typed authority、caller before/after、保留的语义 fallback、删除的兼容路径、迁移/删除的 tests、retired ledger/index 更新和复原 commit。

### Batch 1 · q1_0（2026-07-20，完成，待本任务最终合并）

- **old symbols**：`GgmlBlockDotQ10Q80Op`、`isQ1_0Q8_0BlockDotBody`、`emitQ1_0Q8_0BlockDot`，以及 emitter dispatch row、whole-kernel verifier 和 direct-op fixtures。
- **new typed authority**：公开 source identity 只负责输入识别；`createTypedFlatBlockDotLoopChain` 构造 `TypedFlatBlockDotLoopBodyOp` + `GgmlBlockDotQ10Q80BinarySignCoreOp`；`emitTypedFlatBlockDotLoopBody` 是唯一 dispatcher，`emitQ1_0TypedFlatBlockDotBody` 是唯一 byte-exact emit anchor。
- **caller before / after**：施工前旧 recognizer + old thin emitter 仍可从直接构造的整核 op 到达；施工后上述三个旧 symbol 的 active include/lib 引用均为 0。family 表保留的 `"weft_rvv.q1_0_q8_0_block_dot"` 只是 source-ingestion identity，不注册 ODS op、不选择 compute emitter。
- **fallback**：没有保留旧实现 fallback、compat alias、parser shim 或 shadow dispatch。typed core 当前已有的 attr-less `m2` 安全缺省未由本批新增，仍由 ISSUE-116 / A4-A8 的统一 stamping 任务裁决；它不能回跳旧 op。
- **tests**：旧 whole-kernel verifier fixture 迁为 binary-sign-core verifier；autotuner divergence 改为“公开前门 → typed body → unified schedule → typed emission”；旧 direct emitter fixture删除；新增 retired-op parse rejection mutation；既有 full-pipeline/export byte-exact结构测试保留并通过。
- **ledger / gate**：q1_0 从 `pending_retirement` 迁入四要件 `retired_ledger`；generated retired index 已重建。退役 gate 新增 `retired_active_symbols` 棘轮与自测，防止旧 op/recognizer/emitter 复活。
- **verification**：ODS 变化后 clean rebuild + 强制重链 `weft-opt` / `weft-translate` 通过；4 个 q1 focused lit（core verifier、source-driven VLEN divergence、retired parse mutation、full pipeline/export）通过；两项 retired gate 及 self-test 通过。全量 `check-weft` 在 Batch 2 后统一与基线 973 pass / 3 个既有失败比对。
- **restore pointer**：回滚本 Batch 1 retirement commit；被删实现只在 Git parent/history 中保留，不在 active tree 放 tomb。

### Batch 2 · q2_K/q3_K/q4_K/q5_K/q6_K（2026-07-20，完成，待合并）

- **old symbols**：`GgmlBlockDotQ{2,3,4,5,6}KQ8KOp` 五个 whole-kernel ODS op、对应 verifier、family 表 op-type key、front-door `getOperationName()` key 和五份 direct-op verifier fixture。施工前无旧 `isa/dyn_cast` emitter caller；这五项是 pipeline-dead surface debt。
- **new typed authority**：q4_K/q5_K 走 `createTypedSuperBlockBlockDotLoopChain` 的 dual-accumulator brick chain；q3_K/q6_K 走 `createTypedSuperBlockScalesTimesSumiLoopChain`；q2_K 走 `createTypedSuperBlockScalarScaleMinLoopChain`。三者统一进入 `TypedSuperBlockBlockDotLoopBodyOp`，由 `emitTypedSuperBlockBlockDotLoopBody` 分层机械发射。
- **caller before / after**：五个旧类在 active include/lib 的精确 token 引用从 ODS/verifier/registry/front-door key 降为 0；仅 ODS 四要件 RETIRED NOTE 与 schema/retired index 留历史身份。公开格式名保留为 source-ingestion literal，不注册旧 op、不在 emitter 按名重算语义。
- **tests**：删除五份只为旧 whole-kernel verifier 存活的 dataflow fixture；保留并通过五份 typed-loop EmitC 测试与五份 source→full-pipeline→object export 测试；新增逐格式 old-op parse rejection mutation，任何一个旧 ODS op 复活都会转红。
- **ledger / gate**：五项从 `pending_retirement` 迁入四要件 retired ledger；generated index 从 17 增到 22 个 retired vec_dot monolith entries。退役 gate 报告 active ODS 仅余 2 项。
- **retained audit**：mxfp4 仍由 e5 作为隔离、非普通 typed-path 的 `constructed-weak` 负对照，保留条件成立。q4_0 普通 source path 已 typed，但其旧 op 仅服务无真实 producer、lit-only authored 的 `GgmlQuantContractionOp` block-dot arm，故保留依据不闭合：已降为 `ISSUE-124` 的 `pending_retirement` debt；本批不越界删除该独立 slice，也不把它伪报为合规 production role。
- **verification**：第二次 ODS clean rebuild + 强制重链通过；A7 聚焦 19/19 通过；两项 retired gate 与 mutation self-test 通过；全量 976 项 = 973 pass + 与施工前完全相同的 3 个 generated-bundle 既有失败，零新增失败。
- **restore pointer**：回滚 Batch 2 retirement commit；旧 ODS/verifier/direct fixtures 仅存 Git history，不在 active tree 保留 compat/tomb。

## Dependencies and Parallelism

- 依赖本 campaign 的 Shared Retirement Gate 与 architecture [RET-1] 已合入。
- 可与 A1/B1 并行，但不得与 B3 同时修改 K-quant emitter/front-door 文件；由主整合者串行合并。
- Batch 1 与 Batch 2 可在同一 worktree 内分 commit 验证，最终任务只有两者都完成才标 completed。

## Acceptance Criteria

- [x] HEAD caller graph 证明六个 pending 项的旧 production path 均被 typed path 取代。
- [x] q1_0 旧 emitter/recognizer/dispatch/op-def 与 production callers 为 0。
- [x] q2_K/q3_K/q4_K/q5_K/q6_K 旧 monolith op-def、verifier、family op-type refs 与 production callers 为 0。
- [x] 没有新增 literal format-name compute dispatch、compat alias、bridge、dual path 或 code-affecting default。
- [x] 正常 fixtures 维持既有 byte-exact/ULP contract；typed body 与旧路径复活 mutation 能使测试变红。
- [x] 旧 dataflow/golden tests 已删除或改为直接保护 typed authority，没有只为旧 op-def 存活的 fixture。
- [x] `pending_retirement` 对本任务原六项清零并迁入合规 retired ledger；generated retired index 重建且 gate 通过（新发现的 q4_0 debt 独立绑定 ISSUE-124）。
- [x] q4_0、mxfp4 的 declared predicate/owner/evidence 经 RET-1 审计；mxfp4 成立，q4_0 不闭合并已登记 ISSUE-124/降格，不以 `deliberately_retained` 标签豁免。
- [x] q4_0、mxfp4 的实现与真实职责零非预期改动；仅 q4_0 inventory 分类按事实订正。
- [x] 修改 emitter/verifier 后强制重链；共享 ODS/layout 变化后 clean rebuild，相关 lit/full-link 无新增失败。

## Verification

- `tools/gates/check_monolith_retire.py` 与 `tools/gates/check_retired_index.py`；
- complete symbol/caller census（无有界窗口）；
- focused dialect/conversion/front-door lit + byte-exact golden；
- forced relink / clean rebuild per governance guide；
- `git diff --check` 与相关 full link。

## Out of Scope

- 退役 q4_0 或 mxfp4；
- 改变性能 winner、K-quant 算法或 measurement 数据；
- 以本任务建立新 Formula IR/DSL；
- 顺手清理无关历史目录。

## Issue / Evidence Mapping

- Authority inventory：`schema/monolith-retire-whitelist.v1.json`；
- Historical query：`schema/retired-index.generated.json`；
- 退役工程法：`.trellis/spec/architecture/退役与原子合入.md`；
- emission construction law：`.trellis/spec/architecture/发射与降级.md`。
