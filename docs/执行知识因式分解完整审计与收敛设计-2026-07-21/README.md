# 执行知识因式分解完整审计与收敛设计（2026-07-21）

这是一套面向 Weft-RV 后续 method/spec 与代码重构的内部 ARS 审计包。它不替代 canon，不修改两柱或六律，不创建 task，也不把目标态冒充当前事实。

## 最终裁决

**Major Revision，方向接受，主张暂不冻结。**

保留：

- 两柱；
- 用户确认的原构造公式；
- formula 为主体、selector 为薄包装；
- 完整 execution-layer compiler 定位；
- reconstruction 作为后续目标；
- provider/verifier 仅作普通实现技术。

必须收紧：

- generic knowledge factorization、rule composition、analytic+measurement 和量化 CPU kernel generation 均不是独立 novelty；
- q5_1 是首个 point-authority-erasure 实现里程碑，不是主要 novelty 或 prospective unseen witness；
- 当前 block-quant 主路径仍以 CM1/CM2 和逐点 authority 为主，尚无 CM4；
- QIGen 全文确认它已按 `(M,K,g,b)` 生成/调优 microkernel，并按非均匀 group 序列组合 GEVM/GEPM；宽泛的 family/cross-combination generation 已被直接覆盖；
- Weft 只剩“继续分解完整 per-bit/per-format microkernel leaf，并跨异质 GGML topology、operation/backend 做因果重建”的窄假说；
- 只有多 topology 的 point-authority erasure、稀疏 residual、真实 capability 因果链和重构后硬件/端到端结果，才能支持更强主张。

## 阅读顺序

1. [00｜研究问题、目标态与方法蓝图](phase1_scoping/00-研究问题、目标态与方法蓝图.md)
2. [01｜检索策略与注释文献表](phase2_investigation/01-检索策略与注释文献表.md)
3. [02｜来源核验与主张压力表](phase2_investigation/02-来源核验与主张压力表.md)
4. [03｜外部证据、Construction 审计与目标重构蓝图](phase3_analysis/03-外部证据、Construction审计与目标重构蓝图.md)
5. [04｜独立评审卡](phase4_review/04-独立评审卡.md)
6. [05｜编辑裁决与收敛修订](phase4_review/05-编辑裁决与收敛修订.md)
7. [06｜QIGen 全文差分与最终再收敛](phase5_reaudit/06-QIGen全文差分与最终再收敛.md)

长期维护入口是 [项目全景与 Spec 重构前方法基线](../method/项目全景与Spec重构前方法基线.md)。若早期阶段稿与后续裁决冲突，以 `06` 和长期维护入口为准。

## 当前与目标

当前：

```text
真实复杂 compiler
+ typed facts/body 和局部 structural construction substrate
+ 若干 capability/analytic/residual 决策
+ 大量逐格式 builder / emitter compute authority
- 无 point-authority-erasure 后的 family reconstruction 证据
```

目标：

```text
independent g/c/ω facts
  → original analytic construction formula
  → reusable typed mechanisms + θ
  → explicit applicability/legality boundary
  → typed body
  → thin legal-candidate selection
  → mechanical family realization/emission
```

第一项实现里程碑是 q5_1 flat repack：删除 Q51 GEMV/GEMM builders 及所有等价 authority，只保留独立格式事实和共享 mechanisms，由公式重建两个 regime。它验证本地 reconstructability，但不再承担主要科研差额。论文级确认性证据还必须覆盖 QIGen 仿射 `(g,b,s,z)` group 模型之外的异质 topology；只有要说 unseen/generalization 时，才另需机制/公式冻结后的 prospective witness。

## 当前仓库处理状态

- 已新增长期维护 method 文档；
- 已完成 ARS 四阶段审计、独立评审与编辑裁决；
- 已逐页检查 QIGen 2026 正式全文，并保存本地 PDF；
- 未修改 canon、architecture spec、代码或 task；
- 未运行 compiler/build/hardware 测试，因为本轮只做文档、文献和源码因果审计；
- 后续 spec 应只做去历史、纠错、区分 current/target/evidence 的微创更新。
