# PRD — B线第二块: tq1_0 deployed emit 直接 M=1 板测（澄清 tq1_0 vec_dot 地盘）

## 性质 + 背景
**读-only 板测（零源码改·不 commit）**——澄清 tq1_0 vec_dot 地盘状态。

P2（iq2）范式发现：iq2 grid vec_dot 有**既存 deployed emit 路**·首次直接 M=1 板测 = proxy 0.697→direct 0.84 PASS = **deployed 地盘**（无需新构造）。**tq1_0 vec_dot 同构**：有既存 deployed emit `emitTQ1_0Q8_KBlockDot`（`lib/Conversion/RVV/RVVToEmitCTernaryBinary.cpp:1958`）+ export-e2e fixture（`test/Target/RVV/tq1-0-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir`）。master 现记 tq1_0 vec_dot proxy 0.207(rvv)/0.606(k1)·具名-X（**T3 matmul proxy·非直接 M=1**）。

★P1 攻坚（task `07-19-block2-p1-tq10-vecdot`·commit `03dbc8a4f`）建了**独立 standalone** owned leaf（`experiments/active/b-block2-tq10-vecdot/`·**没用 deployed emit**·md5 338a31bb 未动）·standalone 双板 0.90/0.82 PASS。⟹ **两条路待判**：

## 任务：直测 deployed emit → 判 tq1_0 地盘
1. **export deployed emit**（CORE==PROD·像 P2）：
   ```
   weft-opt test/Target/RVV/tq1-0-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir \
     --weft-rvv-materialize-tq1-0-q8-k-block-dot-source-front-door(核实 pass 名) \
     --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp
   ```
   记 export md5·核 export-e2e fixture 断言 core==prod diff=0（sealed core untouched）。
2. **board-measure 直接 M=1**（compiler-symmetric·像 P2 rvv_run.sh 范式）：deployed emit leaf vs opp=部署手调 `ggml_vec_dot_tq1_0_q8_K_vl128`·clang18 双侧·2-seed（min 稳健则 5-seed）·byte-exact ZERO-MODEL 门（mism=0·3-arm anti-hollow·CORPUS）先过再测 cold。board rvv（+ k1 若可）。
3. **判 verdict**：
   - **deployed emit cold≥0.8 → deployed 地盘**（像 iq2_xxs·proxy 0.207→direct 修正·master flip·手调-tier·**P1 standalone 是冗余探索·记明**）。
   - **deployed emit cold<0.8 但 P1 standalone 0.90 更快 → 需部署 P1 leaf**（deployed emit 慢·P1 新 leaf 是真改进·登记 deployment task·**别直接改 emitter**·先出结论）。
   - 两情况都对比 3 个数：master proxy 0.207 / deployed-emit direct（本测） / P1 standalone 0.90。

## 硬约束
- **read-only**·export + board-measure only·**不改 emitter·不 commit**。
- byte-exact 门先过（deployed emit vs oracle mism=0·像 P2 worst_ulp=0）·门未过报错别测 perf。
- compiler-symmetric（[CASE-COMPILER-ASYMMETRY]·opp 用 clang18-symmetric 路 `build-clang18-rv64gcv`）。
- 成色诚实：对手手调（非便宜档）·near-parity PASS-by-gate 非 beat·M=1 勿外推 e2e。

## 交付（research/·结构化）
- deployed emit export md5 + CORE==PROD（fixture diff=0）确认。
- board verdict：deployed emit direct cold(2/5-seed·编译器对称) + byte-exact(mism/ulp)。
- 三数对比表（proxy 0.207 / deployed-direct / P1-standalone 0.90）+ 判定（deployed 地盘 vs 需部署 P1 leaf）。
- 若 deployed 地盘 → master 入账建议行键(vec_dot,tq1_0,{rvv,k1})。若需部署 P1 → deployment task scope。

## 参照
- P2 范式：`archive/2026-07/07-19-block2-p2-iq2-vecdot/`（iq2 deployed emit 直测·CORE==PROD·rvv_run.sh）+ commit `f55a4b1da`。
- P1 standalone：`experiments/active/b-block2-tq10-vecdot/`（kernels + driver + seals）+ commit `03dbc8a4f`。
- deployed emit：`RVVToEmitCTernaryBinary.cpp:1958 emitTQ1_0Q8_KBlockDot`。
