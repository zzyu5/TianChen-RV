# M-FLAT 砖① — fp16→f32 per-block scale 重建 typed 原语

**parent:** 07-03-g1-q8-0-strong-construct(M-FLAT 里程碑)· **优先级:** P1 · **base:** refactor/full-refactor-m1
里程碑背景 + 四面墙 + 度量裁决见 parent PRD 与 `../07-03-g1-q8-0-strong-construct/research/design-verdict-escalate.md`。

## 一句话

建**一个 typed `tcrv_rvv` op**,在 pre-realized 阶段表达 per-block 双 fp16 scale 重建:读 d_x/d_y(fp16)、fp16→f32 convert、乘 `d_x·d_y` → **一个 f32 SSA 值**。它替代今天 body lowering 里"the two scalar fp16→fp32 reads via `emitc.call_opaque`(the one sanctioned opaque piece)"(RVVOps.td:5275-5276)。这是 M-FLAT 里程碑砖①/5,拆掉四面墙的第①面。

## 精确范围(只此一砖)

- **op 语义:** 输入 = 两个 per-block fp16 scale 源(按 block layout,d_x/d_y;单块 34 字节 AoS、payload 偏移 +2,QK=32);输出 = 一个 f32 = `f32(d_x) * f32(d_y)`。ggml q8_0 顺序:scales 先相乘(`(float)sumi * (d_x*d_y)`,区别于 Q4_0 的 `(sumi*d_x)*d_y`)——本 op 只做 scale 侧 `d_x*d_y`,不碰 sumi。op 命名/操作数形按方言现有约定(参 DequantizeOp/WideningProductOp 的 ins/results 风格)。
- **纵向交付(五件套的相关子集):** ODS op(include/.../RVVOps.td)+ C++ verifier(lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp 风格)+ emitc lowering(lib/Conversion/RVV/...)+ **emit-consistency lit**(test/)。
- **[PAT-1] 注册:** 创建 `schema/pattern-registry.v1.json`(canon schema:科研总纲v2:118,每行 `{pattern_id, requires, transform, mechanism, metrics_hook, status∈{mechanized,partial,planned}}`)。seed **5 行 M-FLAT 砖**:砖① `status: mechanized`(本砖落地)+ 挂本砖 lit 工件路径为 `metrics_hook`;砖②–⑤ `status: planned`。`requires` = 相关能力谓词(如 Zvfh / fp16 能力,按现有 capability schema 谓词命名);`mechanism` 一句话;`transform` 指向本 op 的 lowering。

## 赢的条件(全 laptop,本轮)

1. **[TYPED]** pre-realized 阶段出现该 typed op,替代原 opaque fp16-read;body 该处不再是裸 `emitc.call_opaque` 的手写读(而是 typed op → 其 lowering 复现同一 emit)。
2. **[BYTE-EXACT / emit-consistency]** 该 op 的 emitc lowering 与今天 CORE 发的 fp16→f32 读 + mul **字节一致**(f32 ⊇ fp16,数值域完全覆盖 → 可字节精确)。lit 用本仓 "CORE==emission-plans" emit-consistency 口径锁定,**不是数值 oracle**。
3. **[REGISTRY]** `schema/pattern-registry.v1.json` 存在、砖① 行 `status:mechanized` 且 `metrics_hook` 指向本砖 lit;`python3` 能数出 M-FLAT `mechanized` 计数 = 1/5(可用一行 jq/python,不必新写脚本)。
4. **[BUILD]** `cmake --build build` 干净;相关 lit 绿。**ODS 改动按 [[build-incremental-unreliable]] 用 forced/clean rebuild + 重链 tcrv-opt 验字节。**

## 红线(必守)

- **不许翻 q8_0 六态行 / 不许动 coverage-sixstate.v1.json / coverage-roster.v1.json 的 C_construct**。里程碑未闭合,`C_construct 强义` 保持 0/24。本砖只让 `M-FLAT n/5` = 1/5。**半燃减=禁止。**
- **不许顺手把砖②-⑤ 也做了**(computed-scale dequant / f32 累加 / 块循环 op)——它们是独立砖、独立会话。本砖**只**做 fp16-scale 重建原语。若发现砖① 无法独立落地(必须拽入砖②的 dequant 契约),**STOP 并报告**依赖真相,不硬塞。
- **数值 bit-exact-vs-ggml 不在本轮宣称**(pending-hardware,ssh rvv)。emit-golden 只证"发什么"。
- **typed op 的 lowering 可以用 emitc.call_opaque 的 fp16 intrinsic**(现有强原语也 lower 到 __riscv_* intrinsic)——strong 性在 pre-realized 阶段是 typed op,不要求最终 emitc 无 call_opaque。关键是 pre-realized body 该处是 typed 而非手写裸 opaque。

## 交付物

- C++: 新 typed op(ODS + verifier + emitc lowering)+ 把 q8_0 body 构造处的 fp16-read 换成该 op(**仅此一处结构替换**;不改循环/累加/dequant——那是后续砖)。
- schema: `schema/pattern-registry.v1.json`([PAT-1] 5 行,砖① mechanized)。
- lit: emit-consistency golden(该 op 的 lowering == CORE fp16-read+mul 字节)。
- 报告:op 签名 + verifier 约束 + byte-exact 验证命令与输出 + `M-FLAT n/5` 读数(=1/5)+ 是否触到砖② 依赖(若有,STOP-report)。

## 权威 spec

core-invariants(I1–I9)· 执行总纲 [K-2]/[PAT-1] · 科研总纲 [PAT-1] schema(:118)· 实验总纲 §1.6([L-8])。裁决:`../07-03-g1-q8-0-strong-construct/research/design-verdict-escalate.md`。
