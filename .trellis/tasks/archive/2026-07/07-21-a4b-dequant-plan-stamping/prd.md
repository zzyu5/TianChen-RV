# A4b：剩余 dequant plan 唯一 materializer 与机械 emission

## Goal

只处理 `ISSUE-122` 在 A3 Codebook 之后的剩余面：让 KQuant、GridLookup 与
Ternary（以及 census 证明仍属同一语义的 Nibble 残余 caller）从 typed g 与真实
c/ω 经 canonical formula/provider 生成 complete selected plan，在唯一
pre-emission materializer 中重算/比对，再由 emitter 机械消费。

本任务关闭同一 dequant mechanism 的 provider/verifier/emitter 多头，不重做 A3
Codebook，也不夹带 ISSUE-125 的 SP4/loop-order schedule 切换。

## Reference Contract from A3

A3 Codebook 已证明以下最小形态可落地，A4b 必须复用而非复制：

- canonical selected RVV capability collector；
- `prepareForConversion` 覆盖 direct wrapper、registry clone 与 artifact route；
- 唯一 backend pre-emission materializer 重跑 canonical formula；
- complete stamp all-or-none，partial/forged/stale fail closed；
- emitter 不查询 capability、formula、measurement 或 decode-model strategy；
- provenance/reason 只解释，不定义 compute。

若共享 API 需要泛化，只做有 A3 Codebook 与 A4b 至少两个真实 consumer 的最小
抽取；禁止建大一统 Formula IR、giant optional descriptor 或第二 collector。

## Scope

- 对 KQuant、GridLookup、Ternary 逐 mechanism 做完整 caller census，确认 typed g、
  capability c、bounded ω、candidate/plan、legality、prior、consumer 与 fallback。
- 每个字段按 analytic/measured/structural-constant/honest-null 分类；无真实 consumer
  的 c/θ 不得为了论文公式制造。
- 把 canonical formula/provider 的结果落为 complete typed selected plan；唯一
  materializer 重算/验证，verifier 不复制第二份公式，也不修正输入。
- 修复 GridLookup 的 registry/provider 双头：相同 grid geometry/plan 只有一个
  construction authority，所有生产 caller 共用。
- emitter 只读已验证 plan；删除 format/decode_model/board/march/measurement 驱动的
  二次选择、missing-plan 重算、silent default 与第二 direct lookup。
- 每完成一个 mechanism slice，即在同一合入单元迁完其全部 production caller 并
  删除旧 helper/overload/compat bridge；不同未迁 mechanism 可继续各自当前唯一路径。

## Slice Order

1. 先用 census 冻结 KQuant/Grid/Ternary 的真实 caller 与 duplicate formula；
2. 优先 GridLookup：关闭已知第二 registry head，保住 grid/codebook ABI/geometry
   正确性；
3. 再迁 KQuant 与 Ternary；
4. Nibble 只在 census 找到 A2 后仍有同义残余 caller 时处理，不重新打开已完成
   slice；
5. 每个 slice 单独小 commit，但只有 atomic caller cutover 后才可合入主线。

## Multi-Agent Collaboration Contract

多个 agent 共同负责每个 mechanism slice：

- authority agent 维护 caller graph、canonical formula/materializer 与退役集；
- adversarial agent 独立构造 partial/forged/stale/route-parity tests，并复核 emitter
  无重决策；
- root integration owner 管理共享 API、Codebook 回归、强制重链、全量验证和合入。

不是把 Grid/KQuant/Ternary 各扔给互不沟通的 agent；同一 slice 的实现与审查必须
交叉。共享 materializer/collector 只允许一个写 owner。

## Primary Touch Set

- `include/Weft/Plugin/RVV/RVVFormulaDecision.h` 及最小共享 decision contract；
- dequant mechanism formula/provider 与 typed plan；
- `include/Weft/Support/GridDecodePlan.h` 及其唯一合法实现点；
- RVV backend preparation/materializer；
- KQuant/Grid/Ternary actual emitters 与所有 production callers；
- focused formula/verifier/route parity tests；
- ISSUE-122、formula ledger 与 authority matrix。

## Dependencies

- A2 typed decision contract 与 A3 Codebook 已完成。
- A4a 的 schedule fields 不属于本任务；若共享 stamp/materializer API 有同文件冲突，
  由 root 串行合入后再开始 A4b。
- A5 winner-view 可后续替换 measurement data plane，不阻塞 analytic plan cutover。

## Acceptance Criteria

- [ ] KQuant、GridLookup、Ternary 各有完整 authority/caller census，plan 字段均有
      typed source、真实 consumer 与 honest-null 说明。
- [ ] 每个已迁 slice 只有一个 formula/provider、一个 pre-emission writer/
      materializer 和一个机械 consumer 链。
- [ ] GridLookup 第二 registry/provider head 与同义 direct lookup 为零。
- [ ] missing、partial、错类型、unknown、forged、stale-g/stale-c/inconsistent plan
      在 emission 前 fail closed；不得 default、重算或走旧路。
- [ ] emitter 不查 capability、measurement、reason/provenance、format winner 或
      decode_model strategy 决定 compute。
- [ ] direct、registry、artifact/deployed 路由对相同输入产生同一 verified plan 和
      emitted result；关键 negative 也行为一致。
- [ ] Codebook 继续只用 A3 唯一 collector/materializer，无复制实现且既有 parity
      全绿。
- [ ] 不承重的 c/ω/θ 明确 honest-null；没有为增加“公式复杂度”制造伪轴。
- [ ] 每个迁移 slice 的旧 caller、helper/overload、compat bridge、第二 writer/
      selector 与 silent default 为零。
- [ ] focused、相关 RVV/EmitC suite 与全量 baseline 无新增失败；需要时真硬件
      correctness 另按 measurement 规则留证。

## Verification

- formula/legality C++ tests；
- plan all-or-none、forged/stale 与 route parity lit；
- actual emitted IR/C 或 byte-exact/ULP tests，不能只检查 metadata 存在；
- direct lookup、formula owner、stamp writer/reader、capability parser 的无截断 census；
- Conversion/Target 强制重链；共享 header/struct 变化执行 clean rebuild；
- `formula-authority-matrix.test`、相关 RVV/EmitC suite 与全量 baseline。

## Out of Scope

- A4a SP4/loop-order；
- A5 compiled winner view 与新 measurement rows；
- 改变性能 winner、跑全格式性能 sweep；
- 重写 A3 Codebook；
- 新 Formula dialect、通用表达式 AST、在线 tuner 或 runtime sparse/MoE。

## Issue Mapping

- 主项：ISSUE-122；
- 关联：ISSUE-118/119（格式几何 lift）、ISSUE-117（真实 c 轴）；
- 明确分拆：ISSUE-125 → A4a。
