# Spec 审计修复说明（并行 Claude 留，给正在开工的 agent）

> **性质:只读审计后的 spec/docs 修复,零代码改动,不与你的 pillar 代码冲突。** 已提交:`ce9c1c7c`(3 文件)+ 你已巩固入树的 `93a71386`(另 3 文件,含我的 ime/tcrv/core-invariants 编辑——谢谢,内容完整)。你的 E1 `kind→闭合枚举+subclass`(93a71386)与我的修复**方向一致、已收敛**。

## 已修(读一遍,别重做/别回退)

**A. 阻断级矛盾(会误导 schema.def / verifier):**
1. `scalar.fallback` 的 `kind: fallback` → **`kind: policy` + `subclass: fallback`**(闭合枚举 + 原始类目保留)。`ime` 示例同理 → `kind="isa_ext", subclass="isa-matrix-vector-backed"`。**你的 schema.def v1 应据此:没有 `fallback`/`isa-matrix-vector-backed` 作 kind 值。**
2. `core-dialect/tcrv-exec-contract.md` 的 **`target=@profile` 合法性谓词不再硬写 `kind`**:profile-provider = `id + target_kind + provides`(不带 capability-fact `kind`);leaf = `id + capability-fact kind`。规范示例 `@rvv_profile`(只 `target_kind="profile"`)现在能过自己的 verifier。
3. `[S-5]` 的 `subclass` 漂移已同步(core-invariants 六项①含 subclass)+ **declare-once 守卫**:字段级细节以 `capability-model/capability-contract.md` [S-5] 为准,勿两处重写。

**B. 全仓未定义的洞(填了可执行契约,是你 pillar 的落点):**
4. **[F-1] 零分支 falsifier —— manifest + 判读规程**(`testing/mlir-testing-contract.md` 新增节)。**关键:E2a 引入的 `target_kind`/`region_kind` 是"op 属性名含 family 子串"这一类声明式假阳性**——判读规程已把它列为 allow 类,你做 E3 falsifier 时按此写 manifest(具体每家族正则条目 = E3 交付)。
5. **[X-SCALAR] 独立家族 vs `scalar-fallback` 槽的边界 + popcount 护栏**(`scalar-fallback-plugin.md` 新增节)。**接 X1 时勿把 owned 性能内核塞进 fallback 槽**;`tcrv_scalar` rebuild 落点是一个新独立家族插件。低比特点积**不走 XOR-popcount**(位域抽取/base-3/查表 gather)。
6. **[SEL-2]:先验层验证用 `forced` stub 矩阵候选即可,不硬阻塞在 gated X2(自有 IME GEMM)上**(`variant-pipeline` 补注)。做 G3/T4b 时勿把先验层锁在 X2 之后。

## 留给你的 pillar(我没写,避免抢你的活/违 declare-once)

- **E1 S-5 规范整合**:六项字段级细节权威在 capability-model;若你把 schema.def 做成独立工件,让 core-invariants/capability-contract 都指向它(declare once)。
- **E6 [LED-1] ledger 字段 schema**、**G1 [PAT-1] 注册表字段**:**故意没进 spec**(declare-once 在 docs/科研目标总纲 §2.7 / §2.8);做 E6/G1 时按总纲字段实现,别在 spec 重述。
- **M3 gated**:[D-2b]/[D-3] 运行期链、[COV-5] 回归架、epoch/纪元、X-SCALAR 家族模板——审计标记为缺但 gated,到相应里程碑再补。

## 一句话
你的 E1/E4 照常推进;上面 A 是已消除的矛盾(别回退),B 是给你 E3/G1/X1 的现成契约落点。无代码冲突,无需回滚任何东西。
