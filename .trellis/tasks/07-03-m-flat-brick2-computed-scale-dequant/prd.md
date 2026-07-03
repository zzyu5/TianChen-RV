# M-FLAT 砖② — computed-scale dequant typed 原语

**parent:** 07-03-g1-q8-0-strong-construct(M-FLAT 里程碑,2/5)· **base:** refactor/full-refactor-m1
四面墙 + 度量裁决见 parent PRD 与 `../07-03-g1-q8-0-strong-construct/research/design-verdict-escalate.md`。砖① 已落(commit 23852f56,`tcrv_rvv.block_fp16_scale_product`)。

## 一句话

建**一个新 typed sibling op**,消费 i32 `sumi` + 砖①的**计算 f32 scale**(`block_fp16_scale_product` 输出)→ `(float)sumi * scale`(f32)。拆第②面墙:今天 `DequantizeOp` verifier 硬拒计算 scale、只收导入 ABI float(`RVVDialectWideningOps.cpp:9531`)。

## 决策(已定,不需设计面板)

**新兄弟 op(不改 `DequantizeOp` 契约)。** 理由:`DequantizeOp` 被 3 个既有强路(单块 widening_product_reduce_dequantize 等,`RVVEmitCContractionRouteFamilyPreRealizedValidators.cpp:1075`)以**导入 scale** 消费——它们是目前仅有的强义 C_construct 锚,**零回归**是硬约束。新兄弟 op 让既有契约纹丝不动。命名/操作数形按方言约定(参 `DequantizeOp` `block_fp16_scale_product` 风格)。

## 范围(只此一砖)

- **op 语义:** ins = i32 `sumi`(向量或标量,按 fold 形) + f32 `computed_scale`(SSA,来自砖①,**非**导入 ABI);out = f32 = `(float)sumi * scale`。fail-closed(I7)verifier:computed_scale 必须 f32、sumi 必须 i32;emission 由 op identity 决定(I5)。
- **纵向:** ODS op(RVVOps.td)+ verifier(RVVDialectWideningOps.cpp,**append,不动 DequantizeOp::verify**)+ emitc lowering(byte-identical 到 monolith 的 `(float)sumi * (d_x*d_y)` fold)+ emit-consistency lit(positive + 1-2 fail-closed negative)。
- **[PAT-1]:** `schema/pattern-registry.v1.json` 的 MFLAT-2 行 `status: planned→mechanized` + `metrics_hook`→本砖 lit。M-FLAT → **2/5**。

## 赢的条件(全 laptop,本轮)

1. 新 typed op 在 pre-realized 阶段消费砖①计算 scale;不是裸 emitc。
2. emit-consistency:lowering 与 monolith `(float)sumi * scale` 字节一致(f32⊇,byte-exact by construction)。lit 锁定。
3. **3 个既有 DequantizeOp 强路仍 byte-exact + verifier-valid**(不回归)——跑相关 lit 确认。
4. `schema/pattern-registry.v1.json` MFLAT-2 = mechanized;M-FLAT count = 2/5(一行命令验)。
5. `cmake --build build` 干净;相关 lit 绿(ODS 用 forced/clean rebuild + 重链,见 [[build-incremental-unreliable]])。

## 红线

- **不碰 C_construct / coverage-sixstate / roster**(里程碑未闭,C_construct 0/24)。不半燃减。
- **不动 DequantizeOp::verify / 契约**(零回归既有强路)。
- **不做砖③(f32 跨块累加)/砖④(块循环)**。若砖② 无法独立落地,STOP 报告依赖。
- 数值 bit-exact-vs-ggml = pending-hardware,不本轮宣称。emit-golden 只证"发什么"。

## 交付物

新 typed op(ODS+verifier append+emitc lowering)+ emit-consistency lit + pattern-registry MFLAT-2 flip。报告:op 签名 + byte-exact 验证命令/输出 + 既有 3 强路未回归确认 + M-FLAT 2/5 读数。

## 权威 spec

core-invariants(I1–I9,尤 I5/I7)· 执行总纲 [K-2]/[PAT-1] · 科研总纲 [PAT-1](:118)· [L-8]。
