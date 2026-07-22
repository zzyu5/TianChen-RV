# 横向统一公式构造生命周期纠偏

## 为什么必须重开

旧 task 把 catalog 登记、若干旧符号 denylist 和 `check-weft` 全绿写成“完整横向
cutover 已完成”。复核真实 production call graph 后，该结论不成立：catalog 可以手工
登记一个 case，而 caller 仍绕过 owner；全量测试也没有覆盖这些旁路。

本 task 不重新解释项目，不改变两柱、六律、C1/C2/C3 或主构造公式。它只执行现有
spec 已经规定的单一 authority：

```text
typed g + canonical c + bounded ω
  → family-local formula constructs complete candidate/plan
  → legality
  → optional thin selector over legal complete candidates
  → one final typed body
  → mechanical realization/emission
```

## 横向范围

范围由当前 registry、公开 pass、direct backend、selected-body registry 与 production
EmitC caller 共同给出，不按 formula 数量或目录计进度：

1. RVV quantize/dequantize row lifecycle 与显式 front door；
2. generic/repack/contraction schedule 的所有 public/direct caller；
3. lower-quant contraction 的 algorithm、typed body 与 final schedule 原子性；
4. selected-body registry 的全部 owner，尤其 composite gather–MAcc–scatter；
5. low-precision/dot-reduce primitive/resource facts 的唯一 owner；
6. RVV source front doors 中决定 LMUL、shape、layout、mechanism 或 schedule 的解析规则；
7. EmitC 中按 kind/format/model/default 重新派生 compute descriptor 的路径；
8. production IR/verifier allowlist 中只作 reason、selection 或 measurement replay 的属性；
9. builtin plugin proposal/cost/front-door descriptor 与真实 evaluator/caller 的同源关系。

测试专用输入可以手写 final typed body，但不能成为 production fallback，也不能被当成
source-entry coverage 证据。

## 已确认的反例

### A. Schedule 是补字段 pass，不是原子构造

- `constructRVVSchedulesViaInterface` 是现有唯一 schedule candidate/legality/prior owner；
- project-level `constructRVVFormulaBodies` 未调用它；
- Q40/GEMM subset compatibility pass 仍公开注册；
- lower-quant block-dot 能以缺失 `integer_core_lmul`、`multi_block_factor`、
  `strip_elision` 的状态离开 front door，再等下游 materializer 补齐。

目标：所有 production caller 在同一调用链中构造或验证完整 final schedule；删除 subset
compatibility pass、声明、registration 与依赖其补值的测试世界。统一 schedule pass 若
保留，只能作为显式 tuning/candidate-inspection front door，不能是生产补全前置。

### B. Composite 被 catalog 计入但绕过 registry

selected-body realization 在 owner registry lookup 之前 special-case composite，并直接创建
load/splat/compare/masked gather/MAcc/scatter chain。Catalog 以 `owners().size()+1` 手工把它
算作 case，caller 与 descriptor 不同源。

目标：建立 typed composite `g/c/ω` 与 deterministic construction result；把 composite
纳入真实 owner registry；通用 dispatcher 只查 registry；realizer 机械消费 formula result。
route metadata 只能描述 final artifact，不能反向定义 body。

### C. 编排与 provenance 仍有双世界

- quant/dequant lifecycle 与显式 front door 各自复制遍历、formula、builder 编排；
- contraction/path/repack measurement reason/key 仍写入 production IR，并在 verifier
  allowlist 中被接受；
- source front doors 中若解析 helper 同时承担 capability 解析、candidate 构造与 body
  写入，必须拆成 typed formula result 与机械 builder，而不是靠 formula id 注释关联。

目标：每个 domain 一个可直接调用的 typed construction owner，多合法前门复用它；删除
非语义 reason/measurement mirror。真正影响 compute/layout/ABI 的 semantic attr 保留。

### D. Emitter 仍可能重解释 compute

RVV EmitC 中存在按 `kind`/format 派生 decode primitive、fold model、block length、offset
默认的 descriptor builder。该逻辑若决定 code shape，就必须前移为 formula/typed body
字段；emitter 只做闭集 dispatch 和数值投影。对结构常量不制造假 knob，但也不允许
`value_or` 恢复历史 compute 默认。

## 目标模块结构

```text
Plugin/RVV/Formula-or-Construction owners
  ├─ row quant/dequant domain construction
  ├─ source schedule/formula results
  ├─ contraction/repack/generic schedule
  ├─ selected-body resource/formula results
  └─ composite deterministic construction

FrontDoor / selected-body dispatcher
  └─ collect typed g/c/ω → call owner → atomically build final typed body

EmitC / route / verifier
  └─ consume final typed fields; fail closed on missing/illegal; never reconstruct
```

这不要求大一统 Formula IR、provider service、universal verifier 或通用 mechanism AST。
family-local typed C++/MLIR owner 仍是 compute authority。

## 完成门

- [ ] quant/dequant 的 lifecycle 与显式 front door 复用同一 domain construction owner；
- [ ] project-level RVV lifecycle 构造/验证所有 tunable schedules；
- [ ] lower-quant 所有成功输出携带完整、合法 final schedule，无补字段中间态；
- [ ] Q40/GEMM compatibility schedule pass、API、registration 与 RUN line 为零；
- [ ] composite 是真实 selected-body registry owner，并消费 typed formula result；
- [ ] registry/caller/catalog 三者由同一 live owner 枚举，删除手工 `+1` 例外；
- [ ] source-frontdoor 的承重解析规则输出 typed result，body builder 不重选；
- [ ] emitter 内 kind/format→primitive/fold/offset 的 compute replay 为零；
- [ ] non-semantic reason/selection/measurement mirror 退出 production IR 与 allowlist；
- [ ] complete/absent/partial/illegal、unsupported、empty legal set 全部 fail-closed 行为受测；
- [ ] decisive g/c/ω edge 通过真实入口改变 final typed result/code，honest-null 不伪造影响；
- [ ] authority test 检查真实 caller/symbol/registry，不再只查旧名字；
- [ ] focused unit/lit、script self-test、完整 `check-weft`、`git diff --check` 全绿；
- [ ] spec/issue 只在上述事实成立后改回完成，task 记录最终 commit，工作区干净。

## 明确不做

- 不修改科研主线、六律或公式数学语义；
- 不宣称所有 `ConstructedWeak` leaf 已通过 delete-leaf reconstruction；
- 不做性能 winner 翻转、板测 campaign 或 runtime observer；
- 不以新增 verifier/provider 层替代清晰 construction owner；
- 不保留 legacy/compatibility adapter 作为迁移缓冲。
