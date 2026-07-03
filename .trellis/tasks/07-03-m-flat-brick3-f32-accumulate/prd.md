# M-FLAT 砖③ — f32 跨块标量累加 typed 原语

**parent:** 07-03-g1-q8-0-strong-construct(M-FLAT 里程碑,3/5)· **base:** refactor/full-refactor-m1
砖①②已落(commits 23852f56 / bdad477f)。四面墙见 `../07-03-g1-q8-0-strong-construct/research/design-verdict-escalate.md`。

## 一句话

建**一个新 typed op** 把 flat 家族的 fold `sumf += term` 做成 **f32 跨块标量累加**(块携带、严格升序):`acc_new = acc_old + term`,`term` = 砖②的 `(float)sumi*scale` 输出。拆第③面墙:今天 fold 是裸 `emitc.add`;`WideningAccumulate`/`DeferredAccumulate`(RVVOps.td:3669/3717)是 **i32 整数** intra-strip(dtype/scope 错)。此 op 让 fold 变 typed → 强体过 rejectMixed。

## 范围(只此一砖,同 ①② 模式 copy-then-adapt)

- **op 语义:** ins = f32 `acc`(块携带累加器)+ f32 `term`;out = f32 = `acc + term`。fail-closed(I7)verifier:两操作数 + 结果均 f32;emission 由 op identity 决定(I5)。**严格升序**语义在 op 契约里注明(与 ggml q8_0 顺序一致)。
- **纵向:** ODS op(RVVOps.td,接 `BlockComputedScaleDequantOp` 后)+ verifier(RVVDialectWideningOps.cpp,**append**)+ emitc lowering(byte-identical 到 monolith 的 `sumf = sumf + term` = `emitFlatBlockDot` 的 `AddOp`,`RVVToEmitCBlockQuantLinear.cpp:5776 附近)+ emit-consistency lit(positive + fail-closed negative)。
- **[PAT-1]:** `schema/pattern-registry.v1.json` MFLAT-3 行 `planned→mechanized`,`metrics_hook`→本砖 lit。M-FLAT → **3/5**。

## 赢的条件(全 laptop)

1. typed op 在 pre-realized 阶段做 f32 累加;不是裸 emitc.add。
2. emit-consistency:lowering 与 monolith `sumf + term` 字节一致。lit 锁定(非数值)。
3. `cmake --build build` 干净;既有 RVV lit 全绿(无回归,跑 test/Conversion/RVV+test/Target/RVV)。ODS forced/clean rebuild+relink([[build-incremental-unreliable]])。
4. MFLAT-3 mechanized;M-FLAT count = 3/5(一行命令验)。

## 红线

- **不碰 C_construct / coverage-sixstate / roster**(0/24 不动,不半燃减)。
- **不做砖④(块循环)/砖⑤(闭合接线+删弱 body)**。砖③ 只做 f32 累加 op。若无法独立落地 STOP 报告。
- 数值 bit-exact-vs-ggml = pending-hardware,不本轮宣称。emit-golden 只证"发什么"。
- verifier append,不动既有 op(WideningAccumulate/DeferredAccumulate/DequantizeOp 契约)。

## 交付物

新 typed op(ODS+verifier append+emitc lowering)+ emit-consistency lit + pattern-registry MFLAT-3 flip。报告:op 签名 + byte-exact 验证命令/输出 + 无回归确认 + M-FLAT 3/5 读数。

## 权威 spec

core-invariants(I5/I7)· 执行总纲 [K-2]/[PAT-1] · 科研总纲 [PAT-1](:118)· [L-8]。
