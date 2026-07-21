# 横向公式/构造层重基

## 目标

按当前稳定 spec 一次性建立 project-wide formula/construction stage：所有当前可达的 operator construction entry 都由唯一的 family-local typed construction authority 消费真实 `g/c/ω`，产生会被 legality、selection 与 realization 直接消费的 candidate/plan/body 参数；同时删除旧 provider、planner replay、emitter-side redecision、隐藏默认和兼容旁路。

这不是按 dequant mechanism、format、backend 或“先做几个公式”的纵向迁移。任务只有在完整 current production domain 原子切换、旧 authority 为零、四类覆盖门都闭合后才能完成。

## 已闭合的任务起始域

任务起始 census 以注册与可达代码为准，而不是目录名或旧 ledger：

- 8 个 builtin plugin 的 proposal、analytic cost、legality、selected construction 与 emission 路径；
- 36 个 builtin source front door：RVV 34 个，Toy 与 TensorExtLite 各 1 个；
- `weft-rvv-lower-quant-contraction` 公开 direct construction route；
- RVV selected-body realization registry 的 13 类 owner 与 composite gather/MAcc/scatter case；
- RVV dequant、repack、contraction、schedule、resource、tiling、loop-order、numerics 与 body-realization 中的全部解析决定；
- 所有在 builder、materializer、verifier、planner、selector 或 emitter 中重放同一决定的 caller；
- 当前 test-only/unregistered demo 只在结构上证明不可成为 production fallback，不伪装成 production 分母。

入口数用于冻结本任务的起始可达域，不是公式贡献数量或科研进度。新增或删除注册会由 catalog/registry 测试动态改变分母，不能靠修改手填数字过门。

## 横向设计

### 1. 薄公共 catalog

公共层只提供 descriptor、axis-use/result-kind/construction-strength 小枚举、plugin registry 的只读枚举和静态一致性诊断。Catalog 不保存 evaluator function pointer，不按字符串动态求值，不引入 Formula IR、provider service、通用 mechanism AST 或 untyped result。

Descriptor 与 family-local typed evaluator/constructor declaration 同源，至少表达：owner、operator domain、真实 `g/c/ω` 依赖、结果种类、语义 cases、production entry 与当前 construction strength。

### 2. 入口与结果的真实关联

- 每个 source front door 与 direct route 关联唯一 construction descriptor；
- proposal/cost/analytic prior 明确关联其公式 owner；
- deterministic single-candidate 路径也进入 typed construction，不以“没有选择”为由绕过；
- formula result 必须进入实际 legality/selection/stamp/body realization，后层不得覆盖或重算；
- `g` 不携带 capability、measurement winner 或完整 body id；`c` 来自 canonical capability projection；`ω` 不夹带 winner memory。

### 3. RVV 决策收口

- dequant 的 nibble、codebook、K-quant、grid、ternary/binary-sign 等 mechanism/leaf 选择全部在 emission 前构造并落入 typed plan；
- emitter 只机械消费 selected typed plan，删除 `decode_model` 字符串决策、局部 provider 调用和 VLEN 常量重推；
- repack LMUL 与 contraction algorithm 拆开“候选/解析 prior”和“受限制 measurement winner”，selector 不得创造候选；
- schedule/resource/tiling/loop-order/numerics/body realization 的现有 family-local typed 公式保留语义，但统一注册和关联真实 caller；
- verifier 只检查 IR/ABI/ISA/typed plan 形状与一致性，不重放公式来决定 compute。

### 4. 诚实的构造强度

Catalog 覆盖不等于强义 reconstruction。每条路径按当前事实标为 strong construction 或 deterministic constructed-weak/dispatch-wired。后者必须移除隐藏决策和双 authority，但不会被虚称为已经通过“删除逐点 leaf 后仍可重建”的 `C_construct` 判据。

## 退役项

- `experiments/active/formula-layer-migration/` 下把 8 个决定误当完整公式域的 active matrix/ledger authority；
- 只验证旧 matrix 行数、旧 task 引用和历史 provider 名的脚本门；
- `RVVFormulaDecision` 中“三个 vertical slices”的旧定位与把 measurement 混入 formula context 的接口；
- dequant emitter 内依据 `decode_model` 重选 mechanism/plan 的路径；
- verifier/planner 对同一 formula 的 compute-authoritative replay；
- 同一 declared decision 的兼容 adapter、dual caller、missing-stamp fallback 与被忽略/覆盖的 result 字段。

历史实验与提交仍在 Git/归档中，不继续作为 active build 或架构 authority。

## 完成标准

- [ ] 所有 current registered/direct construction entry 恰好关联一个 catalogued family-local typed owner；
- [ ] 所有 proposal、analytic prior、legality 与 realization caller 都能追到该 owner，且无 result ignore/override/recompute；
- [ ] catalog completeness、production-entry coverage、dependency-edge coverage、semantic-case coverage 由当前代码与测试计算为 100%；
- [ ] unclassified analytic authority、legacy caller、compat adapter、emitter-side redecision、verifier formula replay、selector-created candidate、hidden target/global read 均为 0；
- [ ] dequant 全 mechanism typed plan 在 emission 前完成，emitter 不再按 format/decode-model 选算法；
- [ ] measurement 只在 legality 后的薄 selector 使用，不进入 formula candidate construction；
- [ ] strong 与 constructed-weak 分类由测试和 descriptor 如实暴露，catalog coverage 不冒充 `C_construct`；
- [ ] 删除旧 8-row matrix gate，并以 registry/catalog + behavior/mutation tests 直接保护新 authority；
- [ ] focused tests、`check-weft`、`git diff --check` 通过；若存在任务前已有失败，须证明失败集合未扩大且与本任务无关；
- [ ] 无后续“其余公式迁移”任务、无生产 dual path；本任务完成后工作树干净。

## 不在本任务中改变的内容

- 不修改或重新解释两柱、六律、C1/C2/C3 与论文科研主线；
- 不改变已确定的主构造公式和薄 selector 数学语义；
- 不以本轮 catalog 化宣称所有逐点 leaf 已被强义重建；
- 不开展性能 winner 翻转、硬件 campaign 或论文 novelty 重写；
- 不建立通用 tensor/tile IR 或 runtime observer 主系统。

## 验证

- catalog/registry unit tests：唯一性、入口覆盖、caller linkage、semantic cases 与 construction strength；
- 每个 decisive `g/c/ω` edge 的独立扰动，honest-null 的不变性；
- missing/unknown/illegal/empty-legal-set 的 fail-closed 负例；
- dequant mechanism 全表的 typed-plan stamping 与 emitter denylist；
- selector 不创造 candidate、measurement 不进入 formula 的类型与行为测试；
- 旧 symbol/include/caller 与 active matrix gate 的零残留检查；
- clean incremental build、focused lit/unit、完整 `check-weft`。

