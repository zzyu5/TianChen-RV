# P3 Fork-B codebook: extend the descriptor-driven body to the flat-codebook bucket (iq4_nl / mxfp4 / nvfp4)

> 父 [[07-01-arch-refactor-noperand-core]]。承接 Fork B flat-plain(commit 86977be0,5 emitQxxx→1 emitFlatBlockDot,byte-exact)。目标 = **测试并扩展 "descriptor 驱动 body" 机制到第 2 个 primitive class(codebook-gather)**,验证机制不是 flat-plain 一次性(=paper 的 Track-B 泛化性证据 + 广度)。设计依据 `research/track-b-autoconstruct-scope.md` §5(codebook 行 ~60% mechanical:同 block loop/fold/store,decode 换成 gather)。

## 范围(严格三 op,非 IQ-gather)

**仅** iq4_nl / mxfp4 / nvfp4(flat-codebook bucket)。**不碰** IQ-gather bucket(iq1_s/iq1_m/iq2_*/iq3_* —— scope §5 ~40% HARD,super-block-ish 带 sign codebook + grid,是更大 separate 机制),**不碰** super-block K-quant / ternary,**不碰** 已 done 的 flat-plain。
- iq4_nl:`RVVToEmitCGridCodebook.cpp`(169K,含 IQ-gather——只碰 iq4_nl 方法);16-entry codebook + `vluxei` gather decode。
- mxfp4/nvfp4:`RVVToEmitCCodebookFp4.cpp`(103K);fp4 codebook + scale reconstruction(E8M0/UE4M3),用 libm `ldexpf`。
- dispatch:`RVVToEmitC.cpp` kBlockDotKernels :366(iq4_nl)/:384(mxfp4)/:386(nvfp4)。

## 做法(先评估,再 land-what-works)

1. **先读这三个 emitQxxx + 对比 emitFlatBlockDot 的 skeleton**,判定:codebook decode 能否作为 `emitFlatBlockDot` 的一个新 `decode_primitive`(+ 必要的 codebook/fp4 descriptor 字段 + fp4 fold/scale 处理),还是更干净作**兄弟** `emitCodebookBlockDot(descriptor)` 复用共享 skeleton helper。**用好审美选**(scope §5 说 "one factored primitive per sub-family" —— codebook-gather 与 fp4-codebook 可能是 2 个 primitive)。诚实报共享 vs 分歧的真实比例(验证 ~60% 估计)。
2. **iq4_nl 先**(最简 flat-codebook,整数 codebook),证 byte-exact;再 mxfp4,再 nvfp4(fp4 + libm)。每个独立 byte-exact-gate。
3. 若某 op 抗 byte-exact 合并(fp4 scale-recon 可能 bespoke-dominated)→ 保留其 emitQxxx 作 documented exception,报告真分歧。**partial(如只 iq4_nl)也是诚实好结果**——甚至"codebook 不干净泛化、机制是 flat-plain-specific"也是有价值的 finding。

## 纪律(硬,同 Fork B)

- **byte-exact for existing,per op**:每步 forced CLEAN relink(`rm -f build/bin/tcrv-opt build/bin/tcrv-translate && ninja -C build tcrv-opt tcrv-translate`)+ 全 `ninja check-tianchenrv` = **783 discovered / 780 passed / 3 failed**(3 = pre-existing `computed-masked-strided...dry-run` ×2 + `self-test`;ninja rc=1 是预期 PASS 态),零 codebook/block-dot lit 回归。iq4_nl/mxfp4/nvfp4 各自的 block-dot e2e lit(`diff core.mlir prod.mlir` 双 VLEN)+ conversion lit 强制逐字节。build incremental 不可靠→forced clean relink + BEFORE==AFTER 相等(非绝对指纹)。
- 保 typed-MLIR(emitc builder)非 C-string template。emitter-local:不碰 route/export/front-door/ODS。
- 走 trellis-implement;**不 git commit**(主会话独立复验 byte-exact 后 commit)。

## 报告

机制是否泛化到 codebook(共享 skeleton vs codebook-specific 的真实比例,验证 ~60%);几 op clean vs exception(为何);是 emitFlatBlockDot 扩展 还是 emitCodebookBlockDot 兄弟(审美理由);LOC delta;byte-exact 证据(suite line + 哪些 lit)。诚实:若机制不干净泛化,说清 codebook 的哪部分是 flat-plain 学不到的。
