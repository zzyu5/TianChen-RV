# MOVES.md — org STAGE1 directory reorg (2026-07-06)

**Method**: all cell/doc relocations done with `git mv` (history preserved). Zero file loss:
tracked union `git ls-files experiments/ docs/` = **150 before → 150 after** the moves
(rename-preserving), then **+ new per-cell MANIFEST.md + this MOVES.md** (increase only).
Untracked-ignored scratch physically rode along with each `git mv`'d directory; ignore
patterns were re-anchored to the new paths (see "gitignore re-anchoring" below) so the same
files stay ignored — nothing exposed, nothing lost.

**Target structure** (裁决二): `experiments/{active,sealed,archive,_templates}/<campaign>/<cell>/`
and `docs/{canon,method,reports}/`.

---

## 1. Directory-level summary (campaign taxonomy)

### docs/
| old | new | why |
|---|---|---|
| docs/TianChen-RV_实验总纲v1.md | docs/canon/ | 总纲 (user-sovereign authoritative) |
| docs/TianChen-RV_执行总纲v2.md | docs/canon/ | 总纲 |
| docs/TianChen-RV_科研目标总纲v2.md | docs/canon/ | 总纲 |
| docs/T3-kernel-micro-report-template.md | docs/reports/2026-07-06-… | agent-generated (date-prefixed) |
| docs/T6-e2e-report-template.md | docs/reports/2026-07-06-… | agent-generated (date-prefixed) |
| docs/并行线纪律-worktree-与触碰集.md | docs/reports/2026-07-06-… | agent-generated (date-prefixed) |
| experiments/CADENCE-LAW.md | docs/method/ | machine-consumable framework doc |
| experiments/方法学-kernel微基准两大隐形混淆.md | docs/method/ | framework doc (perf-char sister) |
| experiments/perf-characterizations-layer4.md | docs/method/ | framework doc (schema/canon referenced) |
| experiments/C2_marginal_cost_ledger.md | docs/method/ | C2 ledger framework doc |
| experiments/travel-decision-ledger.md | docs/reports/ | agent autonomous-decision journal (append-only, no date prefix: continuous) |

### experiments/ cells
| campaign | tier | cells |
|---|---|---|
| repack | sealed/repack/ | rvv-SEALED-q4_0-vlen128-fullmarch, rvv-bringup-q4_0-vlen128, k1-vlen256-q4_0-flip, rvv-vlen128-q4_0-gemm-constructed-sealed, rvv-vlen128-q4_0-repack-genroute |
| silicon | sealed/silicon/ | silicon-validation-batch-1, silicon-validation-batch-2, silicon-validation-gemm |
| c1-cleanliness | sealed/c1-cleanliness/ | quant-label-proof, opponent-facts-provenance |
| visibility | active/visibility/ | (T0-sixstate.md, T2-ledger-anchor.md, T7-burndown.md) |
| result-tables | active/result-tables/ | T-N_noise_floor.csv, T3_A_board_A_rvv1.0_vlen128.csv, T3_B_board_B_rvv1.0_vlen256.csv, T8_winloss_gap_ledger.csv (FILLED tables) |
| e2e-harness | active/e2e-harness/ | README.md, models.manifest.csv (data-side index) |
| repack (parked) | active/repack/ | rvv-vlen128-q4_0-gemm-constructed-redeploy (PARKED-P3, gitignored interim) |
| perf-historical | archive/perf-historical/ | ondevice-q5_K, ondevice-q8_0, ondevice-q8_0-deferred, ondevice-q8_0-mbf, T3_step3 |
| (templates) | _templates/ | 16 header-only T-CSVs (T0/T1/T1b/T1c/T1d/T2/T3m/T3p/T4a/T4b/T5a/T5b/T5c/T5d/T6/T7) |

Stayed put at experiments/ top level: `README.md` (kept as experiments index — table→claim map), `MANIFEST.md` (rewritten thin), `.gitignore`.

---

## 2. Non-`git mv` operations (recorded explicitly)

| op | path | reason |
|---|---|---|
| **plain mv** (untracked, gitignored) | experiments/e2e-harness/results/rvv-vlen128-q4_0-gemm-constructed-redeploy/ → experiments/active/repack/… | whole dir is gitignored (0 tracked files); `git mv` cannot move it → plain `mv`. Physical move, no loss. |
| **plain mv + PRESERVE** (untracked, unref) | docs/TCRV_IR三层与编译器身份_docs常驻.md → docs/reports/2026-07-06-TCRV_IR三层与编译器身份_parallel-writer-leftover.md | Parallel-writer leftover, untracked, **zero external refs** (grep clean). Task said delete-if-unreferenced; **preserved instead** (renamed non-authoritative) to honor ★文件零丢失 — untracked ⇒ hard-delete is unrecoverable. **User: hard-delete if you truly want it gone.** |
| **rmdir** (empty) | experiments/e2e-harness/board/ , experiments/e2e-harness/results/ , experiments/e2e-harness/ | emptied after cell moves; empty untracked dirs, git tracks no dirs → zero loss. |
| **NOT FOUND** | docs/Untitled | task said delete the empty shell — **it does not exist** in the tree (nothing to do). |

## 3. gitignore re-anchoring (keep same scratch ignored at new paths)

| file | old pattern | new pattern |
|---|---|---|
| .gitignore (root) | experiments/ondevice-*/ | experiments/archive/perf-historical/ondevice-*/ |
| .gitignore (root) | (none) | scratch/ (added — target-structure gitignored diagnostics area) |
| experiments/.gitignore | e2e-harness/results/rvv-vlen128-q4_0-gemm-constructed-redeploy/ | active/repack/rvv-vlen128-q4_0-gemm-constructed-redeploy/ |

Effect: the 101 previously-ignored scratch files that briefly became untracked-not-ignored after the dir moves are re-ignored → `git ls-files --others --exclude-standard experiments/` = 0. Promoted tracked seals stay tracked (gitignore never affects already-tracked files).

## 4. New tracked files (the only permitted increase)

- `experiments/archive/MOVES.md` (this file)
- one `MANIFEST.md` per cell (see §6) + `experiments/_templates/MANIFEST.md` + `experiments/active/repack/MANIFEST.md`
- `experiments/MANIFEST.md` rewritten thin (same path; not a new file)

---

## 5. Complete file-level rename table (git mv, authoritative)

| # | old path | new path |
|---|---|---|
| 1 | `docs/TianChen-RV_实验总纲v1.md` | `docs/canon/TianChen-RV_实验总纲v1.md` |
| 2 | `docs/TianChen-RV_执行总纲v2.md` | `docs/canon/TianChen-RV_执行总纲v2.md` |
| 3 | `docs/TianChen-RV_科研目标总纲v2.md` | `docs/canon/TianChen-RV_科研目标总纲v2.md` |
| 4 | `experiments/C2_marginal_cost_ledger.md` | `docs/method/C2_marginal_cost_ledger.md` |
| 5 | `experiments/CADENCE-LAW.md` | `docs/method/CADENCE-LAW.md` |
| 6 | `experiments/perf-characterizations-layer4.md` | `docs/method/perf-characterizations-layer4.md` |
| 7 | `experiments/方法学-kernel微基准两大隐形混淆.md` | `docs/method/方法学-kernel微基准两大隐形混淆.md` |
| 8 | `docs/T3-kernel-micro-report-template.md` | `docs/reports/2026-07-06-T3-kernel-micro-report-template.md` |
| 9 | `docs/T6-e2e-report-template.md` | `docs/reports/2026-07-06-T6-e2e-report-template.md` |
| 10 | `docs/并行线纪律-worktree-与触碰集.md` | `docs/reports/2026-07-06-并行线纪律-worktree-与触碰集.md` |
| 11 | `experiments/travel-decision-ledger.md` | `docs/reports/travel-decision-ledger.md` |
| 12 | `experiments/T0_kernel_census_sixstate.csv` | `experiments/_templates/T0_kernel_census_sixstate.csv` |
| 13 | `experiments/T1_C1_structural_conjunction.csv` | `experiments/_templates/T1_C1_structural_conjunction.csv` |
| 14 | `experiments/T1b_failclosed_runtime.csv` | `experiments/_templates/T1b_failclosed_runtime.csv` |
| 15 | `experiments/T1c_external_reproduction.csv` | `experiments/_templates/T1c_external_reproduction.csv` |
| 16 | `experiments/T1d_dual_instance_same_schema.csv` | `experiments/_templates/T1d_dual_instance_same_schema.csv` |
| 17 | `experiments/T2_C2_ledger_marginal_cost.csv` | `experiments/_templates/T2_C2_ledger_marginal_cost.csv` |
| 18 | `experiments/T3m_migration_criterion.csv` | `experiments/_templates/T3m_migration_criterion.csv` |
| 19 | `experiments/T3p_pattern_ablation.csv` | `experiments/_templates/T3p_pattern_ablation.csv` |
| 20 | `experiments/T4a_attribution_samples.csv` | `experiments/_templates/T4a_attribution_samples.csv` |
| 21 | `experiments/T4b_selector_ablation.csv` | `experiments/_templates/T4b_selector_ablation.csv` |
| 22 | `experiments/T5a_ime_structural_corroboration.csv` | `experiments/_templates/T5a_ime_structural_corroboration.csv` |
| 23 | `experiments/T5b_ime_paradigm_ablation.csv` | `experiments/_templates/T5b_ime_paradigm_ablation.csv` |
| 24 | `experiments/T5c_crossover_selector_calibration.csv` | `experiments/_templates/T5c_crossover_selector_calibration.csv` |
| 25 | `experiments/T5d_ime_vendor_path_methodology.csv` | `experiments/_templates/T5d_ime_vendor_path_methodology.csv` |
| 26 | `experiments/T6_e2e_phase_split.csv` | `experiments/_templates/T6_e2e_phase_split.csv` |
| 27 | `experiments/T7_coverage_burndown.csv` | `experiments/_templates/T7_coverage_burndown.csv` |
| 28 | `experiments/e2e-harness/README.md` | `experiments/active/e2e-harness/README.md` |
| 29 | `experiments/e2e-harness/models.manifest.csv` | `experiments/active/e2e-harness/models.manifest.csv` |
| 30 | `experiments/T-N_noise_floor.csv` | `experiments/active/result-tables/T-N_noise_floor.csv` |
| 31 | `experiments/T3_A_board_A_rvv1.0_vlen128.csv` | `experiments/active/result-tables/T3_A_board_A_rvv1.0_vlen128.csv` |
| 32 | `experiments/T3_B_board_B_rvv1.0_vlen256.csv` | `experiments/active/result-tables/T3_B_board_B_rvv1.0_vlen256.csv` |
| 33 | `experiments/T8_winloss_gap_ledger.csv` | `experiments/active/result-tables/T8_winloss_gap_ledger.csv` |
| 34 | `experiments/visibility/T0-sixstate.md` | `experiments/active/visibility/T0-sixstate.md` |
| 35 | `experiments/visibility/T2-ledger-anchor.md` | `experiments/active/visibility/T2-ledger-anchor.md` |
| 36 | `experiments/visibility/T7-burndown.md` | `experiments/active/visibility/T7-burndown.md` |
| 37 | `experiments/T3_step3/.gitignore` | `experiments/archive/perf-historical/T3_step3/.gitignore` |
| 38 | `experiments/T3_step3/aggregate_summary.txt` | `experiments/archive/perf-historical/T3_step3/aggregate_summary.txt` |
| 39 | `experiments/T3_step3/fold_isolation_k1.txt` | `experiments/archive/perf-historical/T3_step3/fold_isolation_k1.txt` |
| 40 | `experiments/T3_step3/k1_ab_raw.txt` | `experiments/archive/perf-historical/T3_step3/k1_ab_raw.txt` |
| 41 | `experiments/T3_step3/k1_rdir.txt` | `experiments/archive/perf-historical/T3_step3/k1_rdir.txt` |
| 42 | `experiments/T3_step3/kernel_factory.c` | `experiments/archive/perf-historical/T3_step3/kernel_factory.c` |
| 43 | `experiments/T3_step3/kernel_m1.cpp` | `experiments/archive/perf-historical/T3_step3/kernel_m1.cpp` |
| 44 | `experiments/T3_step3/kernel_m1.emitc.mlir` | `experiments/archive/perf-historical/T3_step3/kernel_m1.emitc.mlir` |
| 45 | `experiments/T3_step3/kernel_m2.cpp` | `experiments/archive/perf-historical/T3_step3/kernel_m2.cpp` |
| 46 | `experiments/T3_step3/kernel_m2.emitc.mlir` | `experiments/archive/perf-historical/T3_step3/kernel_m2.emitc.mlir` |
| 47 | `experiments/T3_step3/objdump_seals_local.txt` | `experiments/archive/perf-historical/T3_step3/objdump_seals_local.txt` |
| 48 | `experiments/T3_step3/rvv_ab_raw.txt` | `experiments/archive/perf-historical/T3_step3/rvv_ab_raw.txt` |
| 49 | `experiments/T3_step3/rvv_rdir.txt` | `experiments/archive/perf-historical/T3_step3/rvv_rdir.txt` |
| 50 | `experiments/ondevice-q5_K/k1_ab_p3_regres_raw.txt` | `experiments/archive/perf-historical/ondevice-q5_K/k1_ab_p3_regres_raw.txt` |
| 51 | `experiments/ondevice-q5_K/k1_ab_raw.txt` | `experiments/archive/perf-historical/ondevice-q5_K/k1_ab_raw.txt` |
| 52 | `experiments/ondevice-q5_K/kernel_factory.c` | `experiments/archive/perf-historical/ondevice-q5_K/kernel_factory.c` |
| 53 | `experiments/ondevice-q5_K/kernel_ours.emitc.mlir` | `experiments/archive/perf-historical/ondevice-q5_K/kernel_ours.emitc.mlir` |
| 54 | `experiments/ondevice-q5_K/rvv_ab_p3_regres_raw.txt` | `experiments/archive/perf-historical/ondevice-q5_K/rvv_ab_p3_regres_raw.txt` |
| 55 | `experiments/ondevice-q5_K/rvv_ab_raw.txt` | `experiments/archive/perf-historical/ondevice-q5_K/rvv_ab_raw.txt` |
| 56 | `experiments/ondevice-q8_0-deferred/A_deferred.o` | `experiments/archive/perf-historical/ondevice-q8_0-deferred/A_deferred.o` |
| 57 | `experiments/ondevice-q8_0-deferred/A_deferred.rv64gcv_zvfhmin.objdump` | `experiments/archive/perf-historical/ondevice-q8_0-deferred/A_deferred.rv64gcv_zvfhmin.objdump` |
| 58 | `experiments/ondevice-q8_0-deferred/fair/perf_rvv_vlen128_FAIR.csv` | `experiments/archive/perf-historical/ondevice-q8_0-deferred/fair/perf_rvv_vlen128_FAIR.csv` |
| 59 | `experiments/ondevice-q8_0-mbf/kernel_core_mbf2.o` | `experiments/archive/perf-historical/ondevice-q8_0-mbf/kernel_core_mbf2.o` |
| 60 | `experiments/ondevice-q8_0-mbf/kernel_ggml_factory.o` | `experiments/archive/perf-historical/ondevice-q8_0-mbf/kernel_ggml_factory.o` |
| 61 | `experiments/ondevice-q8_0-mbf/kernel_q8_mbf1.o` | `experiments/archive/perf-historical/ondevice-q8_0-mbf/kernel_q8_mbf1.o` |
| 62 | `experiments/ondevice-q8_0-mbf/seal/fold_segment_objdump.txt` | `experiments/archive/perf-historical/ondevice-q8_0-mbf/seal/fold_segment_objdump.txt` |
| 63 | `experiments/ondevice-q8_0/evidence.json` | `experiments/archive/perf-historical/ondevice-q8_0/evidence.json` |
| 64 | `experiments/ondevice-q8_0/host_k1/run_rv64gc.txt` | `experiments/archive/perf-historical/ondevice-q8_0/host_k1/run_rv64gc.txt` |
| 65 | `experiments/ondevice-q8_0/host_k1/run_rv64gcv.txt` | `experiments/archive/perf-historical/ondevice-q8_0/host_k1/run_rv64gcv.txt` |
| 66 | `experiments/ondevice-q8_0/host_k1/target_profile.txt` | `experiments/archive/perf-historical/ondevice-q8_0/host_k1/target_profile.txt` |
| 67 | `experiments/ondevice-q8_0/host_rvv/run_rv64gc.txt` | `experiments/archive/perf-historical/ondevice-q8_0/host_rvv/run_rv64gc.txt` |
| 68 | `experiments/ondevice-q8_0/host_rvv/run_rv64gcv.txt` | `experiments/archive/perf-historical/ondevice-q8_0/host_rvv/run_rv64gcv.txt` |
| 69 | `experiments/ondevice-q8_0/host_rvv/target_profile.txt` | `experiments/archive/perf-historical/ondevice-q8_0/host_rvv/target_profile.txt` |
| 70 | `experiments/ondevice-q8_0/kernel_q8_0_q8_0_flat_block_dot.o` | `experiments/archive/perf-historical/ondevice-q8_0/kernel_q8_0_q8_0_flat_block_dot.o` |
| 71 | `experiments/ondevice-q8_0/results_summary.csv` | `experiments/archive/perf-historical/ondevice-q8_0/results_summary.csv` |
| 72 | `experiments/opponent-facts-provenance/NOTES.md` | `experiments/sealed/c1-cleanliness/opponent-facts-provenance/NOTES.md` |
| 73 | `experiments/opponent-facts-provenance/opponent-facts.pin.json` | `experiments/sealed/c1-cleanliness/opponent-facts-provenance/opponent-facts.pin.json` |
| 74 | `experiments/quant-label-proof/NOTES.md` | `experiments/sealed/c1-cleanliness/quant-label-proof/NOTES.md` |
| 75 | `experiments/quant-label-proof/evidence.json` | `experiments/sealed/c1-cleanliness/quant-label-proof/evidence.json` |
| 76 | `experiments/e2e-harness/results/k1-vlen256-q4_0-flip/aggregate.txt` | `experiments/sealed/repack/k1-vlen256-q4_0-flip/aggregate.txt` |
| 77 | `experiments/e2e-harness/results/k1-vlen256-q4_0-flip/correctness.txt` | `experiments/sealed/repack/k1-vlen256-q4_0-flip/correctness.txt` |
| 78 | `experiments/e2e-harness/results/k1-vlen256-q4_0-flip/evidence.json` | `experiments/sealed/repack/k1-vlen256-q4_0-flip/evidence.json` |
| 79 | `experiments/e2e-harness/results/k1-vlen256-q4_0-flip/phase_split_raw.txt` | `experiments/sealed/repack/k1-vlen256-q4_0-flip/phase_split_raw.txt` |
| 80 | `experiments/e2e-harness/results/k1-vlen256-q4_0-flip/preflight.txt` | `experiments/sealed/repack/k1-vlen256-q4_0-flip/preflight.txt` |
| 81 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/NOTES.md` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/NOTES.md` |
| 82 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/correctness.txt` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/correctness.txt` |
| 83 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/decode_heavy.txt` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/decode_heavy.txt` |
| 84 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/decode_light.txt` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/decode_light.txt` |
| 85 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/decode_medium.txt` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/decode_medium.txt` |
| 86 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/evidence.json` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/evidence.json` |
| 87 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/evidence_decode_heavy.json` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/evidence_decode_heavy.json` |
| 88 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/evidence_decode_light.json` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/evidence_decode_light.json` |
| 89 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/evidence_decode_medium.json` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/evidence_decode_medium.json` |
| 90 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/evidence_phasesplit.json` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/evidence_phasesplit.json` |
| 91 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/objdump_mechanism_fixed.txt` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/objdump_mechanism_fixed.txt` |
| 92 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/objdump_seal.txt` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/objdump_seal.txt` |
| 93 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/phase_split_raw.txt` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/phase_split_raw.txt` |
| 94 | `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/preflight.txt` | `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/preflight.txt` |
| 95 | `experiments/e2e-harness/results/rvv-bringup-q4_0-vlen128/NOTES.md` | `experiments/sealed/repack/rvv-bringup-q4_0-vlen128/NOTES.md` |
| 96 | `experiments/e2e-harness/results/rvv-bringup-q4_0-vlen128/aggregate.txt` | `experiments/sealed/repack/rvv-bringup-q4_0-vlen128/aggregate.txt` |
| 97 | `experiments/e2e-harness/results/rvv-bringup-q4_0-vlen128/correctness.txt` | `experiments/sealed/repack/rvv-bringup-q4_0-vlen128/correctness.txt` |
| 98 | `experiments/e2e-harness/results/rvv-bringup-q4_0-vlen128/evidence.json` | `experiments/sealed/repack/rvv-bringup-q4_0-vlen128/evidence.json` |
| 99 | `experiments/e2e-harness/results/rvv-bringup-q4_0-vlen128/phase_split_raw.txt` | `experiments/sealed/repack/rvv-bringup-q4_0-vlen128/phase_split_raw.txt` |
| 100 | `experiments/e2e-harness/results/rvv-bringup-q4_0-vlen128/preflight.txt` | `experiments/sealed/repack/rvv-bringup-q4_0-vlen128/preflight.txt` |
| 101 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-gemm-constructed-sealed/NOTES.md` | `experiments/sealed/repack/rvv-vlen128-q4_0-gemm-constructed-sealed/NOTES.md` |
| 102 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-gemm-constructed-sealed/bandwidth_analysis.txt` | `experiments/sealed/repack/rvv-vlen128-q4_0-gemm-constructed-sealed/bandwidth_analysis.txt` |
| 103 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-gemm-constructed-sealed/constructed_evidence.json` | `experiments/sealed/repack/rvv-vlen128-q4_0-gemm-constructed-sealed/constructed_evidence.json` |
| 104 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-gemm-constructed-sealed/constructed_raw.txt` | `experiments/sealed/repack/rvv-vlen128-q4_0-gemm-constructed-sealed/constructed_raw.txt` |
| 105 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-gemm-constructed-sealed/noisefloor_evidence.json` | `experiments/sealed/repack/rvv-vlen128-q4_0-gemm-constructed-sealed/noisefloor_evidence.json` |
| 106 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-gemm-constructed-sealed/noisefloor_raw.txt` | `experiments/sealed/repack/rvv-vlen128-q4_0-gemm-constructed-sealed/noisefloor_raw.txt` |
| 107 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-gemm-constructed-sealed/objdump_fingerprint.txt` | `experiments/sealed/repack/rvv-vlen128-q4_0-gemm-constructed-sealed/objdump_fingerprint.txt` |
| 108 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-gemm-constructed-sealed/restore_verify.txt` | `experiments/sealed/repack/rvv-vlen128-q4_0-gemm-constructed-sealed/restore_verify.txt` |
| 109 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-gemm-constructed-sealed/target_profile.txt` | `experiments/sealed/repack/rvv-vlen128-q4_0-gemm-constructed-sealed/target_profile.txt` |
| 110 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/NOTES.md` | `experiments/sealed/repack/rvv-vlen128-q4_0-repack-genroute/NOTES.md` |
| 111 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/aggregate_baseline.txt` | `experiments/sealed/repack/rvv-vlen128-q4_0-repack-genroute/aggregate_baseline.txt` |
| 112 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/evidence.json` | `experiments/sealed/repack/rvv-vlen128-q4_0-repack-genroute/evidence.json` |
| 113 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/evidence_baseline.json` | `experiments/sealed/repack/rvv-vlen128-q4_0-repack-genroute/evidence_baseline.json` |
| 114 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/evidence_redeploy.json` | `experiments/sealed/repack/rvv-vlen128-q4_0-repack-genroute/evidence_redeploy.json` |
| 115 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/phase_split_raw.txt` | `experiments/sealed/repack/rvv-vlen128-q4_0-repack-genroute/phase_split_raw.txt` |
| 116 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/redeploy_aggregate.txt` | `experiments/sealed/repack/rvv-vlen128-q4_0-repack-genroute/redeploy_aggregate.txt` |
| 117 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/redeploy_correctness.txt` | `experiments/sealed/repack/rvv-vlen128-q4_0-repack-genroute/redeploy_correctness.txt` |
| 118 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/redeploy_phase_split_raw.txt` | `experiments/sealed/repack/rvv-vlen128-q4_0-repack-genroute/redeploy_phase_split_raw.txt` |
| 119 | `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/target_profile.txt` | `experiments/sealed/repack/rvv-vlen128-q4_0-repack-genroute/target_profile.txt` |
| 120 | `experiments/silicon-validation-batch-1/NOTES.md` | `experiments/sealed/silicon/silicon-validation-batch-1/NOTES.md` |
| 121 | `experiments/silicon-validation-batch-1/evidence.json` | `experiments/sealed/silicon/silicon-validation-batch-1/evidence.json` |
| 122 | `experiments/silicon-validation-batch-1/kernels/iq1_m.kernel.c` | `experiments/sealed/silicon/silicon-validation-batch-1/kernels/iq1_m.kernel.c` |
| 123 | `experiments/silicon-validation-batch-1/kernels/iq1_s.kernel.c` | `experiments/sealed/silicon/silicon-validation-batch-1/kernels/iq1_s.kernel.c` |
| 124 | `experiments/silicon-validation-batch-1/kernels/iq4_nl.kernel.c` | `experiments/sealed/silicon/silicon-validation-batch-1/kernels/iq4_nl.kernel.c` |
| 125 | `experiments/silicon-validation-batch-1/kernels/q4_0_repack.kernel.c` | `experiments/sealed/silicon/silicon-validation-batch-1/kernels/q4_0_repack.kernel.c` |
| 126 | `experiments/silicon-validation-batch-1/results/iq1_m/evidence.json` | `experiments/sealed/silicon/silicon-validation-batch-1/results/iq1_m/evidence.json` |
| 127 | `experiments/silicon-validation-batch-1/results/iq1_m/run_rvv.txt` | `experiments/sealed/silicon/silicon-validation-batch-1/results/iq1_m/run_rvv.txt` |
| 128 | `experiments/silicon-validation-batch-1/results/iq1_s/evidence.json` | `experiments/sealed/silicon/silicon-validation-batch-1/results/iq1_s/evidence.json` |
| 129 | `experiments/silicon-validation-batch-1/results/iq1_s/run_rvv.txt` | `experiments/sealed/silicon/silicon-validation-batch-1/results/iq1_s/run_rvv.txt` |
| 130 | `experiments/silicon-validation-batch-1/results/iq4_nl/evidence.json` | `experiments/sealed/silicon/silicon-validation-batch-1/results/iq4_nl/evidence.json` |
| 131 | `experiments/silicon-validation-batch-1/results/iq4_nl/run_rvv.txt` | `experiments/sealed/silicon/silicon-validation-batch-1/results/iq4_nl/run_rvv.txt` |
| 132 | `experiments/silicon-validation-batch-1/results/q4_0_repack/evidence.json` | `experiments/sealed/silicon/silicon-validation-batch-1/results/q4_0_repack/evidence.json` |
| 133 | `experiments/silicon-validation-batch-1/results/q4_0_repack/run_rvv.txt` | `experiments/sealed/silicon/silicon-validation-batch-1/results/q4_0_repack/run_rvv.txt` |
| 134 | `experiments/silicon-validation-batch-1/target_profile.txt` | `experiments/sealed/silicon/silicon-validation-batch-1/target_profile.txt` |
| 135 | `experiments/silicon-validation-batch-2/NOTES.md` | `experiments/sealed/silicon/silicon-validation-batch-2/NOTES.md` |
| 136 | `experiments/silicon-validation-batch-2/kernels/iq2_xxs.kernel.c` | `experiments/sealed/silicon/silicon-validation-batch-2/kernels/iq2_xxs.kernel.c` |
| 137 | `experiments/silicon-validation-batch-2/kernels/iq3_xxs.kernel.c` | `experiments/sealed/silicon/silicon-validation-batch-2/kernels/iq3_xxs.kernel.c` |
| 138 | `experiments/silicon-validation-batch-2/results/iq2_xxs/evidence.json` | `experiments/sealed/silicon/silicon-validation-batch-2/results/iq2_xxs/evidence.json` |
| 139 | `experiments/silicon-validation-batch-2/results/iq2_xxs/run_rvv.txt` | `experiments/sealed/silicon/silicon-validation-batch-2/results/iq2_xxs/run_rvv.txt` |
| 140 | `experiments/silicon-validation-batch-2/results/iq3_xxs/evidence.json` | `experiments/sealed/silicon/silicon-validation-batch-2/results/iq3_xxs/evidence.json` |
| 141 | `experiments/silicon-validation-batch-2/results/iq3_xxs/run_rvv.txt` | `experiments/sealed/silicon/silicon-validation-batch-2/results/iq3_xxs/run_rvv.txt` |
| 142 | `experiments/silicon-validation-batch-2/target_profile.txt` | `experiments/sealed/silicon/silicon-validation-batch-2/target_profile.txt` |
| 143 | `experiments/silicon-validation-gemm/NOTES.md` | `experiments/sealed/silicon/silicon-validation-gemm/NOTES.md` |
| 144 | `experiments/silicon-validation-gemm/kernels/q4_0_repack_gemm.kernel.c` | `experiments/sealed/silicon/silicon-validation-gemm/kernels/q4_0_repack_gemm.kernel.c` |
| 145 | `experiments/silicon-validation-gemm/results/q4_0_repack_gemm/evidence.json` | `experiments/sealed/silicon/silicon-validation-gemm/results/q4_0_repack_gemm/evidence.json` |
| 146 | `experiments/silicon-validation-gemm/results/q4_0_repack_gemm/run_rvv.txt` | `experiments/sealed/silicon/silicon-validation-gemm/results/q4_0_repack_gemm/run_rvv.txt` |
| 147 | `experiments/silicon-validation-gemm/target_profile.txt` | `experiments/sealed/silicon/silicon-validation-gemm/target_profile.txt` |

---

## 6. Per-cell MANIFEST split (STAGE1 step 5)

The old single-body `experiments/MANIFEST.md` REGISTRY was split into **one `MANIFEST.md` per cell**
(campaign + role + status + durable file list). Top-level `experiments/MANIFEST.md` is now a **thin index**
pointing at the per-cell manifests (STAGE2 INDEX.md may supersede it). Per-cell manifests written at:
sealed/repack/{5 cells}/, sealed/silicon/{3}/, sealed/c1-cleanliness/{2}/, active/visibility/, active/result-tables/,
active/e2e-harness/, active/repack/ (campaign-level, since the cell dir is gitignored), archive/perf-historical/{5}/, _templates/.

## 7. Harness abstraction (STAGE1 step 4) — already clean

No tracked harness/scripts remain in any cell: a prior commit (MANIFEST 2026-07-06 "harness 迁出" note) already
moved all protocol/driver/CI scripts to `tools/e2e-harness/` and `tools/lint/`. The only harness-shaped files left
in cells are **gitignored scratch** (`board_ab_q5k.sh`, `run_fair.sh`, `*_verify_driver.c`, `mirror_test`, build `.o/.cpp`)
that ride with their archive cell and stay ignored. Registered evidence-pointer code (`*.kernel.c`, `*.emitc.mlir`,
sealed `*.o`) intentionally **stays in-cell** per the data-cell contract.

## 8. ⚠ STAGE2 follow-ups (referrers that now DANGLE — NOT fixed here; out of STAGE1 mv scope)

These point at old paths and will break until STAGE2 rewires them. **Do not commit before addressing, or CI goes red.** The lint / tooling gate path-breaks below are **RESOLVED** (STAGE2, 裁决九 — reworked to per-cell manifests / sealed paths and CI-wired fail-closed); the schema/docs/spec reference-breaks that follow are still open:

- **[RESOLVED 裁决九] `tools/lint/check_manifest.py`** — was an exact-path pin vs the single top-MANIFEST REGISTRY. Reworked; parses the thin (empty) REGISTRY block without crashing, exit 0.
- **[RESOLVED 裁决九] `tools/lint/check_experiments_data_only.py`** — superseded by `check_experiments_layout.py` (per-cell registered-artifact-pointer lookup); exit 0.
- **[RESOLVED 裁决九] `tools/lint/check_opponent_facts_pin.sh:26`** — now points at `experiments/sealed/c1-cleanliness/opponent-facts-provenance/opponent-facts.pin.json`; CI-wired in `.github/workflows/falsifier-gate.yml` (job `opponent-facts-pin`), exit 0.
- **[RESOLVED 裁决九] `tools/visibility/`** (gen_sixstate_table.py, gen_burndown_curve.py, recompute_ledger_anchor.sh, regen_all.sh, check_visibility_drift.py) — retargeted to `experiments/active/visibility/`; `check_visibility_drift.py:101` now asserts `endswith("experiments/active/visibility")`.
- **`tools/e2e-harness/run_e2e.sh:17`** (`RESULTS_ROOT=…/experiments/e2e-harness/results`) and **silicon-validation-batch-1/2/gemm run_*.sh** (`CELL=…/experiments/silicon-validation-*`) — future re-runs write to old paths.
- **`schema/pattern-registry.v1.json`** lines 75/85/88 — `experiments/ondevice-q8_0-deferred/fair/perf_rvv_vlen128_FAIR.csv`, `experiments/perf-characterizations-layer4.md`, `experiments/ondevice-q5_K/` (now archive/perf-historical/ + docs/method/).
- **`schema/coverage-sixstate.v1.json`** lines 112/121/130/151/166/391 — `experiments/silicon-validation-batch-1|2` (now sealed/silicon/).
- **`docs/canon/TianChen-RV_执行总纲v2.md`** — `experiments/perf-characterizations-layer4.md` (→ docs/method/), `experiments/T8_winloss_gap_ledger.csv` (→ active/result-tables/); also a pre-existing stale ref to `experiments/T3_step3/board_ab.sh` (that script already lives in tools/).
- **`docs/reports/2026-07-06-T3-…` / `-T6-…` templates** — reference silicon + e2e cell paths (self-contained agent reports; low priority).
- **`.trellis/spec/…` + `.trellis/tasks/…`** — T-N / visibility / ondevice / T3_step3 path refs (spec refs are by-reference, archived-task refs are historical; low priority).
