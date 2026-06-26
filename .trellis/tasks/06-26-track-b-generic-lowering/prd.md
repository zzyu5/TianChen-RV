# Track B 泛化：通用 op-driven 自动 lowering = 真·TTGIR-analog 重构

> 父任务 [[06-26-compiler-maturity-retest]]；纪律护栏见父 PRD。**这是用户点名缺的"类似 TTGIR 的真实重构"**：从一个 generic 高层 op **自动构造** kernel body（替代 per-kernel 手写 emitter），能力驱动选形状。[[backend-maturity-triton-reframe]]

## 为什么这才是 TTGIR-analog（而 L0-L3 / wa 砖不是）
- **L0–L3**（[[06-26-emitter-l0-l3-infra]]）= byte-exact 整理**手写** emitter → 更干净但**仍 per-kernel 手写**，不是 generic 机制。
- **wa 砖**（[[06-26-winA-parity-bricks]]）= per-kernel **手写**一个更宽 body → 仍手写。
- **Track B** = 从 generic `vector.multi_reduction` 等高层 op **自动构造** body、LMUL 从 gearbox 能力 fact 推——**这才是"从 IR 自动生成 kernel"的结构性机制 = Triton TTGIR 的真类比**。

## 现状基线（code-verified 2026-06-26）
- `lib/Plugin/RVV/RVV*SourceFrontDoor.cpp`（7 文件）。
- **rung 1–2 真自动构造**：`RVVReductionSourceFrontDoor`(dot-reduce)、`RVVDequantDotSourceFrontDoor`(scale·dot)——能力翻转(VLEN128 e8m2/i16m4↔VLEN256 e8m1/i16m2)**两板 byte-exact 封印**（注20/24）。
- **rung 3–5 = recognize+DELEGATE**：`RVVQ40/IQ4NL/Q4KBlockDotSourceFrontDoor` 自动识别签名 + 供 format facts，但 **body 仍来自 3808 行 `RVVToEmitCKQuant` / 2477 行 `RVVToEmitCTernaryBinary` 手写 emitter**（q4_K front-door 自陈："does NOT hand-roll any of it ... feeding the existing q4_K emitter unchanged"）。
- **无任何 IR layout/pipelining transform pass**——所有 "schedule" 是 `*ScheduleMaterialization`/`*StripWidthMaterialization` **stamping 写 attr**。

## Goal
把 Track B 从"裸 dot/dequant 真自动 + 复杂 quant 识别后委托手写"**泛化成全 quant zoo 真自动构造 body**（rung 3-5 停止 delegate），并把 cm 形状综合（L4）表达成"在 generic 机制里能力驱动选更宽形状"——**让 beat 来自机制，不是又一个 per-kernel 手写 kernel**。

## In-scope（staged，逐 rung；没做清单#9 "后续大工程"）
- **G1 nibble 真自动构造**：q4_0/q4_1/q5_0 的 nibble-decode body 从 generic 结构自动构造（不再 delegate）；接 wa1/wa2 的 wide-LMUL 但**经 generic 路 emit**。
- **G2 codebook 真自动构造**：iq4_nl/mxfp4/FP4 的 vrgather codebook body 自动构造（现 iq4_nl 已 recognize+delegate→改真 auto）。
- **G3 super-block 真自动构造**：q4_K（最常用）的 6-bit scale bit-dance + aux32 自动构造（替代 3808 行 KQuant delegate）。
- **G4 IQ-grid**：iq1/iq2/iq3 + ternary——最难（需 signs64 op-attr，与 wa3 共）。
- **G5 cm-shape-through-Track-B**：把 `cm4`/`cm5`（[[06-26-row2-beat-levers]]）的 wider 形状综合接进 generic 机制——**beat = generic 机制综合出 ggml 没手写的形状并实测更快**。

## DoD（每 rung）
- **byte-exact**：auto-constructed body ≡ 对应手写 emitter（vs scalar oracle + vs ggml `_vlN`），两板封印；能力翻转 lit（非-NULL）。
- 当 G5 接入：cm wider 形经 generic 路 emit、**实测更快 = beat**（否则 NO claim）；micro+e2e，触及格交 [[06-26-table-retest-fill]]。
- 证据状态标。

## 边界（诚实，不 cargo-cult）
- **EmitC 边界不变**：我们**不**做 RA / 指令调度 / 软件流水 / async-copy（clang/gcc 的，[[backend-maturity-triton-reframe]]）。TTGIR-analog 对我们 = **generic body 自动构造 + 能力驱动 shape 选择**，**不**照搬 Triton 的 `convert_layout`/`num_stages` pass 名（doc §6.3 "Triton cargo-culting 警告"）。
- "软件流水"对我们 = body 构造里摆 strip 链（Track B 可泛化），**非**独立 scf pipeliner pass。

## Deps / Risk
复用 [[06-26-emitter-l0-l3-infra]] 词汇表 + L1 widening-chain（G 系列正会喂 m2，**L1 修的 m2 bug 是 G1 的硬前置**）；与 [[06-26-row2-beat-levers]] 强耦合（G5=beat 经此机制）；G4 dep signs64 op-attr。**大工程、staged、HIGH**；start 后逐 rung sub-spawn。**must-NOT**：cargo-cult Triton pass 名 / 声明 out-schedule clang / 把 recognize+delegate 当"自动构造"。
