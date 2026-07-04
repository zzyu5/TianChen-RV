# Flat block-dot fp-fold oracle (pinned, structure-independent) — [K-5]/[实验宪法 §1.8]

> **稳定契约,零现值。** 这是平面块量化 `vec_dot` 家族(q8_0/q4_0/q4_1/q5_0/q5_1,后续 K-quant/码本同构)的**钉死浮点折叠 oracle**。schedulable op 的**所有** `fold_structure` 变体(`per-block`、`deferred-ordered`)与所有调度旋钮档({mbf, lmul-anchor, strip})都对**本规格**逐字节验收——**不对 ggml 二进制**(ggml 跨自家 SIMD 变体本就不 bit-exact,故它不是正确性门;这一事实反而是"显式数值契约=特性"的论文动机)。一次钉死,永不随变体重定义。

## 1. 折叠规格(per block,块升序)

对每块 `b ∈ [0, nb)`,按升序:

1. **整数核 `sumi_b`**(格式相关 decode,精确 int32,无结合律自由度):
   `sumi_b = Σ_j decode_x(b,j) * decode_y(b,j)`,int32 累加。q8_0 = 直读 int8;q4_0/q5_0 = offset-binary 解包;q4_1/q5_1 = 同 + min 项(见 §3)。整数精确,调度/LMUL/mbf/strip 不改其值。
2. **尺度转换**(精确 IEEE fp16→fp32,round-to-nearest-even):
   `dx = f32(x[b].d)`,`dy = f32(y[b].d)`;min 格式另有 `dm = f32(x[b].m)`。
3. **钉死 fp 折叠 —— 无 FMA 收缩、严格左结合、有序累加:**

   ```
   t    = f32(sumi_b) ⊗ dx        // 一次 round;⊗ = IEEE-754 mul, RNE
   t    = t          ⊗ dy         // 一次 round
   sumf = sumf       ⊕ t          // 一次 round;⊕ = IEEE-754 add, RNE, 有序
   ```

   `sumf` 初值 = +0.0f。结果 = 全 nb 块折叠后的 `sumf`。

### 禁则(违反即非本规格)
- **禁 `dx ⊗ dy` 预乘**:必须 `((sumi ⊗ dx) ⊗ dy)`,不得先算 `dx*dy`——预乘改变结合序、与 deferred 路的双舍入对不上。
- **禁 FMA 收缩**:每个 `⊗`/`⊕` 是独立舍入算子(`-ffp-contract=off` 或显式 EmitC 算子);`fmaf(sumi, dxdy, sumf)` 单舍入形**不是**本规格。
- **禁任何重结合**:块序升序、左结合固定。

## 2. 两 fold_structure 变体如何各自命中本规格

- **`per-block`(缺省)**:向量整数点积 → `sumi_b`(精确)→ 标量 `t=sumi·dx; t=t·dy; sumf=sumf+t`(逐块,无 FMA,左结合)。
- **`deferred-ordered`(P2c)**:B 块的 `sumi` 打包进向量 → `vfcvt` f32 → `vfmul ·dx_vec` → `vfmul ·dy_vec` → `vfredosum.vs` 以运行 `sumf` 为**种子**做**有序**归约。RVV 规范定义 `vfredosum.vs` 按 lane 升序、种子在前逐个相加 = 与串行左折叠**逐位相同** → 命中同一 §1 规格。**由构造即 bit-exact,非巧合**。

## 3. min-项格式(q4_1/q5_1)的补充

`x = d·q + m` ⇒ 每块贡献 = `d·(Σ q·y) + m·(Σ y)`,y 侧再乘 dy。**逐格式钉死**(step 4 翻各格时落),同守 §1 禁则(无 FMA、左结合、有序、无预乘);min 项与主项的相加序在该格 oracle 内一次钉死。q8_0/q4_0/q5_0 无 min 项,直接用 §1。

## 4. oracle 实现纪律([实验宪法 §1.8])

- **结构无关**:标量 oracle **不与任何向量核共享** bit→lane decode;独立实现,入库**带版本号**。
- **性质测试**:全零 / 交替符号 / 满幅(±127 int8, 满幅 fp16 scale)/ fp16 次正规 & 极端指数 scale。
- **变异测试**:对 oracle 自身做变异(改一个 decode/一次 round 序),验证测试能抓。
- 硬件 byte-exact 用本 oracle 作参照(非 monolith golden、非 ggml 二进制)。

## 5. 政策门控的重排变体(第二档,永不 headline)

允许一个**更快的重排变体**(如 K 路累加器打破串行折叠链,潜在 2–4×),但:
- 仅挂在能力事实 `numerics.reassoc_ok`(`kind=policy`)之后,**fail-closed**(缺省=精确变体,政策事实缺席即拒重排)。
- 对**单列的 reassoc-容忍 oracle**验收 + **声明 ULP 上界**,**不**对 §1 精确 oracle 验收。
- **论文主数字永远是 §1 精确变体**;重排变体单列附 ULP 上界,永不作 headline([K-5] 浮点=ULP 上界声明)。

## 6. lit 注释级陷阱(写进相关 lit)
- `dx·dy` 预乘禁止(改结合序);
- oracle 规格未按本文件钉死前,不谈任何"匹配/bit-exact";
- "bit-exact 模式 vs 快速模式"的问法弃用 —— 不是 mode split,是**缺省精确变体 + 挂 `numerics.reassoc_ok` 的普通候选变体**。

---
*关联:[K-5] 正确性门;[实验宪法 §1.8] oracle 结构无关;G1.M-FLAT-P2c 裁决(fold_structure 旋钮);Q3 数值裁决。本规格是 step 1(旋钮 byte-exact)与 step 5(P2c)的共同验收基座。*
