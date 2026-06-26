# N2 IME GEMM（board k1/ime；parity + disclosed-null 天花板）

> 父任务 [[06-26-compiler-maturity-retest]]；纪律护栏见父 PRD。现状：6 leaf MAC board-sealed（K1 16/16 bit-exact），但**没生成完整 IME GEMM**（表4 5.66–12.9× 是 ggml-spacemit 的，mis-attributed；IME e2e≈1.0 净-null）。这是 N2 的 perf 腿。[[k1-ime-n2-hardware-candidate]]

## Goal
从 tcrv emit 一个**我们自己的**完整 IME GEMM（非 ggml-spacemit），给 N2 一个真 perf 数据点——诚实天花板 = parity-vs-ggml-spacemit + disclosed e2e null。

## In-scope（2 原子项）
- **n2-a1 int8-GEMM micro**：从 tcrv 综合 IME int8 GEMM micro，**tag = 我们-emitted（非 ggml-spacemit）**；Win-B2 天花板 = parity。无 dep，可并入 Win-A 阶段。
- **n2-a3 quant-decode IME GEMM**（**multi-week**）：quant-decode 路 IME GEMM；**per-arm NMSE gate**；天花板 = parity + disclosed-null（受 K1 7GB 内存墙界定）。**硬 dep n2-a1**。

## DoD
- **byte-exact / NMSE gate**：vs 独立 scalar oracle（K1 硅片）；engagement 板上 print 确认。
- perf micro **且** e2e；**can't-possibly-help M=1 decode 对照**——若"win"在矩阵单元帮不上的 M=1 也出现 = 全局 codegen artifact 非 IME 单元（注7 教训）；单元隔离须同 toolchain `SPACEMIT=ON/OFF` 非两独立 build。
- 证据状态标；**must-NOT 把 ggml-spacemit GEMM 当我们 win**。

## Deps / Risk
a1→a3 硬 dep。a3 multi-week + decode-washout 风险。k1 harts 0-3、内存墙 7GB。
