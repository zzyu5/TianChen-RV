# 全项目 production construction path 统一

## 任务来源

当前稳定 spec 要求所有 registered/direct production entry 都经过：

```text
typed g/c/ω
  → family-local formula/construction
  → legality
  → optional bounded selection
  → final typed body/plan
  → mechanical realization/emission
```

上一轮重构实际收口了 RVV 的主要 lifecycle，但 completion audit 发现 shared
`TypedBackendEmissionDriver` 对非 RVV backend 默认 `prepareForConversion` no-op；因此
直接 `weft-translate`、target artifact exporter 和 backend registry clone conversion
仍可以绕过 construction owner。Scalar 的 q4_0 dequant 与 tq2_0 direct body 已提供可复跑
反例；IME、Template、Toy、TensorExtLite 也必须逐一审计，而不能因 proposal formula 或
construction-protocol manifest 已登记就视为 final body construction 已统一。

本 task 是一次横向 project-wide 收敛，不做先迁一个 format、再留旧路径的纵向切片。
中间编辑可以按依赖顺序进行，但最终提交中所有 current production direct/caller path
必须同时满足下列门。

## 范围

1. shared backend-emission harness、backend registry clone conversion、target translate
   routes、target artifact exporters 和默认 execution-planning pipeline；
2. RVV quantize/dequantize/source/contraction/repack/EmitC 路径与 direct conversion；
3. Scalar direct compute-skeleton、tq2_0 q8_K、q4_0 dequant 路径；
4. IME MMA/slide/matmul/q4_0/q8_0/q4_K typed paths；
5. Template、Toy、TensorExtLite 的 source marker、selected boundary 和 direct backend path；
6. Demo 的 production plugin/target/direct backend 路径，以及 Offload 的 explicit
   unsupported 分类；不能既出现在 production catalog 又无 construction owner，也不能
   用空 backend 冒充 Offload coverage；
7. formula catalog、production-entry registry、authority/static tests 与 direct-route
   negative tests。

## 硬约束

- 不改变两柱、六律、`θ=f^A(g,c,ω)`、候选域/合法域/薄 selector 的主语义；
- 不增加 universal verifier/provider 或第二套 Formula IR；
- 不把 catalog、reason、provenance、manifest 或 route metadata 当作 compute authority；
- 不让 emitter 从 format/kind/decode/fold/default 重新决定 code shape；
- 不把手写 final body 当作 source-entry coverage 证据；direct typed input 若是合法入口，
  必须有 family-local construction owner 对其完整 plan/body 做构造或 fail-closed 资格化；
- quantize 与 dequantize 必须对称进入同一顶层 construction-before-emission contract；
- 不保留旧 direct emitter/compatibility adapter 作为迁移缓冲。

## 完成门

- [x] 从当前 registry、backend table、target route、artifact exporter 和公开 pass 生成
  production-entry inventory；每个 entry 明确唯一 owner、formula id、result consumer；
- [x] shared backend interface 不再允许隐式 no-op preparation；每个 registered backend
  明确调用 family-local construction owner；unsupported/test-only backend 也有显式拒绝；
- [x] RVV quantize、dequantize、source、contraction、repack 与 direct conversion 的
  construction caller graph 单一且可证明；不存在只迁量化、反量化仍走旧 emitter 的路径；
- [x] Scalar q2/dequant 及 IME/Template/Toy/TensorExtLite direct body 的 code-affecting
  结构由 typed family-local formula/plan 构造并由 emitter 机械消费；旧 hidden defaults 删除；
- [x] raw typed input 由 family constructor 产生 derived final plan/body；缺必要 typed
  input/mechanism/body、预置 partial/conflicting plan、unsupported geometry 或非法 capability
  时，所有 direct translate/artifact/clone conversion fail closed，不回旧路径；
- [x] catalog 与真实 construction owner/production entry 同源，非仅 proposal/cost 登记；
- [x] decisive g/c/ω 的因果测试、honest-null 测试、正负 direct-route 测试和 authority
  denylist 覆盖全部 inventory；
- [x] full build、完整 lit、plugin/unit、direct-route self-tests、diff/json/task checks
  全部通过；
- [x] spec/issues 只在上述事实成立后更新，task 固定主体实现 commit，收口提交后工作区干净；
- [x] `ConstructedWeak` 与 delete-leaf strong reconstruction 的边界继续诚实保留。

## 明确不是完成证明的东西

- 公式 descriptor 数量、catalog case 数量或源码行数；
- 只运行 execution-planning pipeline 的成功测试；
- 只证明 source front door 调过 formula；
- backend manifest/construction-protocol verifier 通过；
- emitter 能处理手写 typed op；
- 上一轮 RVV-only task 的 commit 或其“全项目”措辞。
