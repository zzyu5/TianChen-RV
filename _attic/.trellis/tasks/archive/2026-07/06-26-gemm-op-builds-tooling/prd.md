# 建缺失 GEMM op + Track-B production-export

> 父任务 [[06-26-compiler-maturity-retest]]；纪律护栏见父 PRD。**解锁 table-retest 里被"无 op/被 production 路拒"卡住的格子**——是 table-fill 的硬前置。

## Goal
造出缺的 GEMM op（让 q8_0/q5_0 prefill 格可测）+ 接通 Track-B production-export（让 Track-B byte-anchor body 经 production 路可达、e2e 可测）。

## In-scope（3 原子项）
- **build-q80-gemm (B1)**：`RVVOps.td` 现有 Gemm/RepackGemm `{Q40,Q41,Q4K}`，**确无 Q80Q80 GEMM**（code-verified 2026-06-26）→ q8_0 prefill 无 op 可测。造 Q80Q80 GEMM op（较常用 type）。
- **build-q50-gemm (B2)**：同理**确无 Q50Q80 GEMM**；造之。**诚实**：q5_0 compute-bound → 建了多半仍测 LOSS（不是建了就赢）。
- **trackB-production-export (WA-6)**：production 路（`tcrv-rvv-emitc-to-cpp`/materialize-emission-plans）默认把 Track-B byte-anchor body 当 block-quant mf4 拒掉；接通剩 **3 个精确面**：`RVVEmitCContractionRouteFamilyCommon.cpp:310`(C-type-summary mf4-pin)、`...PlanOwners.cpp:509-626`(route-facts/leaf-profile)、`...Validation.cpp:478`(plan re-validation) 的 byte-anchor m1/m2 case。已清 7 面 regression-free，剩这 3 面 cascade。

## DoD
- build op：byte-exact vs scalar oracle（先正确再谈 perf）；op 在 lit 解析 + round-trip。
- production-export：Track-B body 经 production 路 byte-exact 两板封印（非 bypass）；regression-free 全 RVV lit。
- 解锁后通知 [[06-26-table-retest-fill]]：q8_0/q5_0 prefill 格从 N/A-by-absence→可测；Track-B e2e 可达。

## Deps / Risk
build op 是 prefill 格的硬前置。production-export 是多面 cascade tooling 整合（非 Track-B 正确性问题——body 已 board-sealed）。**must-NOT**：把"造了 op"当"赢了"（q5_0 大概率 LOSS）。
