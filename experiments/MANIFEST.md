# experiments/ MANIFEST —— 保留工件台账 + 漂移校验源

> 机检:`tools/lint/check_manifest.py`(fail-closed CI 门)。校验【耐久内容】(git 跟踪 +
> 未跟踪-未忽略 = fresh clone 会拿到的全部文件)与本文件 `REGISTRY` 区逐路径相等;有未登记
> 文件或登记文件缺失即红(退出非零)。**改 experiments/ 增删文件 → 必须同步改 REGISTRY,否则 CI 红。**
>
> 第二道门:`tools/lint/check_experiments_data_only.py`(A2 data-only 门)。experiments/ 是
> **数据 cell 不是代码树**:耐久内容只准是数据/证据(`*.json`/`*.csv`/`*.txt`/`*.md`/`*.objdump`/
> `*.log`/`*.err`/`.gitignore`)+ 被 REGISTRY 登记的**证据指针代码**(kernel 源/`.emitc.mlir`/sealed
> `.o`,CLASS 3)。**任何 harness / 脚本(`*.py`/`*.sh`)/ 可执行 / 未登记代码 → 红**;协议脚本一律
> 落 `tools/e2e-harness/`(见下"harness 迁出"注)。

## 三类内容铁律(experiments/ 只放这三类,其余=残留必清)

1. **模板 / 协议脚本 / 框架-ledger 文档** —— schema、空表(零现值,按快照由 CI 填)、
   harness/协议脚本、以及稳定契约级 ledger/方法学文档(C2 台账、perf 特性化、方法学、出行 ledger)。
2. **合规格 cell** —— 遵循格 schema 的结果:`evidence.json` / `aggregate` / `NOTES` / 分相 raw /
   填了数的表行,**必带 board + phase 标注 + 环境指纹 + 快照**。
3. **被引用工件** —— 被 ledger(T#)/ paper / test / docs / schema 引用的原始数据 / objdump seal /
   kernel 源与 emitc,作为某个 cell 或主张的证据指针落点。

**第四类 = 死 workflow 残留 / 孤儿 / 非三类杂物 → 清出。** 本轮已清:`ondevice-q8_0/export.err`(空文件、
死 export 日志)。

## 口径:耐久内容 vs gitignore 的 on-device scratch

- 校验对象 = **耐久内容**(git 视角 = git 跟踪 + 未跟踪未忽略)。根 `.gitignore` 的
  `experiments/ondevice-*/` 是 on-device scratch(`.trellis/scripts/ondevice/run_ondevice_verify.sh`
  的输出落点),整目录忽略、**里面未被 force-add 的部分不入本校验**——易失、可由重跑板子再生。
  (`ondevice-q8_0` 是历史 force-add、已跟踪,故在册。)
- **被引用证据已 promote(2026-07-06,option-1 裁决)**:原先悬空风险(已跟踪的表/docs/schema/test
  引用 gitignore scratch 内文件)已消除——那 13 个**被引用的 text 证据 + 被 tracked 文件直接引用的 `.o`**
  已 `git add -f` 提升为 tracked(原路径不改,引用自然 resolve),入上方 REGISTRY 的 CLASS 3。

<!-- REGISTRY:BEGIN -->
### CLASS 1 — 模板 / 协议脚本 / 框架-ledger 文档

索引 + 本台账自身:
- `experiments/README.md` — 表→主张映射 + 格 schema + Win 阶梯(index/template)
- `experiments/MANIFEST.md` — 本保留工件台账(self)
- `experiments/CADENCE-LAW.md` — 节奏法 C:每 +4 C_construct → 强制上板批验证(self)
- `experiments/.gitignore` — 忽略 INTERIM / P3-parked cell(见文末 PENDING/INTERIM 区);工件留盘、未跟踪、不入机检

> **harness 迁出(A2,2026-07-06):** experiments/ 里的所有协议/编排/驱动脚本已 `git mv` 到
> `tools/e2e-harness/`(保结构),CI 校验脚本本身迁到 `tools/lint/`——experiments/ 从此只留数据 +
> 证据指针代码。迁出清单:`tools/lint/check_manifest.py`(原 experiments/check_manifest.py)、
> `tools/e2e-harness/{run_e2e.sh, aggregate_e2e.py, board/*.sh}`、
> `tools/e2e-harness/T3_step3/{board_ab.sh, aggregate.py}`、
> `tools/e2e-harness/silicon-validation-batch-1/{run_silicon_batch.sh, bd_verify_driver.c, q4_0_repack_verify_driver.c}`、
> `tools/e2e-harness/silicon-validation-gemm/{run_silicon_gemm.sh, gemm_verify_driver.c}`。
> 被复现脚本引用的 **kernels/`*.kernel.c` / `*.emitc.mlir` / sealed `.o`** 是证据指针,**留 experiments/**
> (下方 CLASS 2/3 仍登记)。

空表模板(零现值,按快照由 CI 填):
- `experiments/T0_kernel_census_sixstate.csv`
- `experiments/T1_C1_structural_conjunction.csv`
- `experiments/T1b_failclosed_runtime.csv`
- `experiments/T1c_external_reproduction.csv`
- `experiments/T1d_dual_instance_same_schema.csv`
- `experiments/T2_C2_ledger_marginal_cost.csv`
- `experiments/T3m_migration_criterion.csv`
- `experiments/T3p_pattern_ablation.csv`
- `experiments/T4a_attribution_samples.csv`
- `experiments/T4b_selector_ablation.csv`
- `experiments/T5a_ime_structural_corroboration.csv`
- `experiments/T5b_ime_paradigm_ablation.csv`
- `experiments/T5c_crossover_selector_calibration.csv`
- `experiments/T5d_ime_vendor_path_methodology.csv`
- `experiments/T6_e2e_phase_split.csv`
- `experiments/T7_coverage_burndown.csv`

T6 e2e 分相 harness — 索引 + 模型表(协议脚本已迁 `tools/e2e-harness/`,见上"harness 迁出"注):
- `experiments/e2e-harness/README.md`
- `experiments/e2e-harness/models.manifest.csv`

T3_step3 — 本地 *.o 忽略规则(协议/分析脚本已迁 `tools/e2e-harness/T3_step3/`):
- `experiments/T3_step3/.gitignore` — 忽略本地 *.o 构建产物

框架 / ledger 文档(稳定契约级、被 docs/schema 引用):
- `experiments/C2_marginal_cost_ledger.md` — C2 边际成本台账
- `experiments/perf-characterizations-layer4.md` — perf 指令级 emit 四特性化 + 忠实性(docs/schema 引)
- `experiments/方法学-kernel微基准两大隐形混淆.md` — 测量混淆方法学(perf-char 姊妹页)
- `experiments/travel-decision-ledger.md` — 出行期自主裁决 + F1..F15 findings 台账

C1 洁净化证据档(quant-fix F22 / commit c347b412 补验,2026-07-06):
- `experiments/quant-label-proof/NOTES.md` — 删字符串 byte-exact 证明(fact-drives-routing、label-inert)+ 复现命令 + 记录 sha256 a457e1b9
- `experiments/quant-label-proof/evidence.json` — 同证明 machine 版(fixtures + audit tokens + adversarial)
- `experiments/opponent-facts-provenance/NOTES.md` — opponent-fact(floor/heavy)溯源到 pinned ggml 行锚 + CI 门说明
- `experiments/opponent-facts-provenance/opponent-facts.pin.json` — fact↔ggml-pin 绑定探针(`tools/lint/check_opponent_facts_pin.sh` 消费:pin 变 → STALE)

### CLASS 2 — 合规格 cell(board + phase + 指纹 + 快照)

填了数的表行 cell:
- `experiments/T-N_noise_floor.csv` — 噪声地板(board_fp + 快照)
- `experiments/T3_A_board_A_rvv1.0_vlen128.csv` — 板 A 测量 cell
- `experiments/T3_B_board_B_rvv1.0_vlen256.csv` — 板 B 测量 cell
- `experiments/T8_winloss_gap_ledger.csv` — Win/Loss + 缺口闭合登记

e2e cell — k1/VLEN256 q4_0 flip(F5):
- `experiments/e2e-harness/results/k1-vlen256-q4_0-flip/preflight.txt`
- `experiments/e2e-harness/results/k1-vlen256-q4_0-flip/correctness.txt`
- `experiments/e2e-harness/results/k1-vlen256-q4_0-flip/phase_split_raw.txt`
- `experiments/e2e-harness/results/k1-vlen256-q4_0-flip/aggregate.txt`
- `experiments/e2e-harness/results/k1-vlen256-q4_0-flip/evidence.json`

e2e cell — rvv/VLEN128 q4_0 bring-up(F1):
- `experiments/e2e-harness/results/rvv-bringup-q4_0-vlen128/preflight.txt`
- `experiments/e2e-harness/results/rvv-bringup-q4_0-vlen128/correctness.txt`
- `experiments/e2e-harness/results/rvv-bringup-q4_0-vlen128/phase_split_raw.txt`
- `experiments/e2e-harness/results/rvv-bringup-q4_0-vlen128/aggregate.txt`
- `experiments/e2e-harness/results/rvv-bringup-q4_0-vlen128/evidence.json`
- `experiments/e2e-harness/results/rvv-bringup-q4_0-vlen128/NOTES.md`

e2e cell — rvv/VLEN128 q4_0 SEALED fullmarch(F4):
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/preflight.txt`
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/correctness.txt`
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/phase_split_raw.txt`
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/decode_light.txt`
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/decode_medium.txt`
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/decode_heavy.txt`
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/evidence.json`
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/evidence_decode_light.json`
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/evidence_decode_medium.json`
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/evidence_decode_heavy.json`
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/evidence_phasesplit.json`
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/objdump_seal.txt`
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/objdump_mechanism_fixed.txt`
- `experiments/e2e-harness/results/rvv-SEALED-q4_0-vlen128-fullmarch/NOTES.md`

on-device cell — q8_0 flat block-dot 双板 bit-exact + micro(tracked):
- `experiments/ondevice-q8_0/evidence.json`
- `experiments/ondevice-q8_0/results_summary.csv`
- `experiments/ondevice-q8_0/host_k1/run_rv64gc.txt`
- `experiments/ondevice-q8_0/host_k1/run_rv64gcv.txt`
- `experiments/ondevice-q8_0/host_k1/target_profile.txt`
- `experiments/ondevice-q8_0/host_rvv/run_rv64gc.txt`
- `experiments/ondevice-q8_0/host_rvv/run_rv64gcv.txt`
- `experiments/ondevice-q8_0/host_rvv/target_profile.txt`
- `experiments/ondevice-q8_0/kernel_q8_0_q8_0_flat_block_dot.o`

silicon-validation-batch-1 cell — 4 constructed 格 emit-golden→silicon_validated(board rvv/VLEN128, no-FMA 左结合 ggml scalar oracle, 2026-07-06):
- `experiments/silicon-validation-batch-1/evidence.json` — 批级汇总(4 格 verdict + falsifier 控制)
- `experiments/silicon-validation-batch-1/NOTES.md` — 方法 + q4_0-repack FMA 分相诚实说明
- `experiments/silicon-validation-batch-1/target_profile.txt` — board + clang + oracle + host tool 指纹
- `experiments/silicon-validation-batch-1/kernels/iq4_nl.kernel.c` — front-door 导出 kernel-under-test(C)
- `experiments/silicon-validation-batch-1/kernels/iq1_s.kernel.c` — front-door 导出 kernel-under-test(C)
- `experiments/silicon-validation-batch-1/kernels/iq1_m.kernel.c` — front-door 导出 kernel-under-test(C)
- `experiments/silicon-validation-batch-1/kernels/q4_0_repack.kernel.c` — front-door 导出 kernel-under-test(C)
- `experiments/silicon-validation-batch-1/results/iq4_nl/run_rvv.txt` — 板 raw(256/256 ×3 seeds)
- `experiments/silicon-validation-batch-1/results/iq4_nl/evidence.json` — iq4_nl BIT_EXACT ULP=0
- `experiments/silicon-validation-batch-1/results/iq1_s/run_rvv.txt` — 板 raw
- `experiments/silicon-validation-batch-1/results/iq1_s/evidence.json` — iq1_s BIT_EXACT ULP=0
- `experiments/silicon-validation-batch-1/results/iq1_m/run_rvv.txt` — 板 raw
- `experiments/silicon-validation-batch-1/results/iq1_m/evidence.json` — iq1_m BIT_EXACT ULP=0
- `experiments/silicon-validation-batch-1/results/q4_0_repack/run_rvv.txt` — 板 raw(FMA-fold bounded-ULP ×3 seeds)
- `experiments/silicon-validation-batch-1/results/q4_0_repack/evidence.json` — q4_0-repack FMA_FOLD_BOUNDED_ULP + f64 accuracy

silicon-validation-gemm cell — constructed q4_0 REPACK GEMM(PREFILL,M>1)硅验,GEMM silicon-validation debt 1→0(board rvv/VLEN128, no-FMA 左结合 ggml scalar oracle + f64 ref, 2026-07-06;引用者 F23 / silicon-validation debt):
- `experiments/silicon-validation-gemm/NOTES.md` — 方法 + FMA-fold bounded 诚实说明(debt 1→0,FMA-bounded-correct 非 ULP=0,GEVM 姊妹)
- `experiments/silicon-validation-gemm/target_profile.txt` — board + clang + oracle + repack 布局 + provenance(be66c917)指纹
- `experiments/silicon-validation-gemm/kernels/q4_0_repack_gemm.kernel.c` — front-door 导出 kernel-under-test(C,VERBATIM)
- `experiments/silicon-validation-gemm/results/q4_0_repack_gemm/run_rvv.txt` — 板 raw(4 shapes,FMA-fold bounded-ULP)
- `experiments/silicon-validation-gemm/results/q4_0_repack_gemm/evidence.json` — GEMM FMA_FOLD_BOUNDED_ULP verdict + 4-shape f64 accuracy substantiation

e2e cell — rvv/VLEN128 q4_0 repack generation-vs-routing 分相 + constructed-GEVM 重部署(2026-07-06):
- `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/evidence.json` — 批级汇总(gen-vs-routing + sealed + decode 问答 + confound)
- `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/NOTES.md` — 方法 + 分相 + confound 诚实说明
- `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/target_profile.txt` — board/model/toolchain/A-B 指纹
- `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/phase_split_raw.txt` — baseline(monolith GEVM)板 raw
- `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/aggregate_baseline.txt` — baseline median+IQR+CI
- `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/evidence_baseline.json` — baseline prefill 5.045×/decode 1.564×
- `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/redeploy_correctness.txt` — constructed GEVM in-situ 贪心一致 GREEN
- `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/redeploy_phase_split_raw.txt` — redeploy(constructed GEVM)板 raw
- `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/redeploy_aggregate.txt` — redeploy median+IQR+CI
- `experiments/e2e-harness/results/rvv-vlen128-q4_0-repack-genroute/evidence_redeploy.json` — redeploy prefill 5.530×/decode 1.950×(cross-session confound)

### CLASS 3 — 被引用工件(T3_step3 q8_0 m1/m2 双板证据 bundle)

被 `T3_A` / `T3_B` q8_0 行 + docs 引用的 raw / emitc / seal:
- `experiments/T3_step3/kernel_m1.emitc.mlir` — T3_B q8_0 m1 emitc 指针
- `experiments/T3_step3/kernel_m2.emitc.mlir` — T3_A q8_0 m2 emitc 指针
- `experiments/T3_step3/kernel_factory.c` — T3_A/T3_B 对手(ggml factory,verbatim)
- `experiments/T3_step3/kernel_m1.cpp`
- `experiments/T3_step3/kernel_m2.cpp`
- `experiments/T3_step3/k1_ab_raw.txt` — T3_B 板 raw + objdump AB
- `experiments/T3_step3/rvv_ab_raw.txt` — T3_A 板 raw + objdump AB
- `experiments/T3_step3/objdump_seals_local.txt`
- `experiments/T3_step3/aggregate_summary.txt`
- `experiments/T3_step3/fold_isolation_k1.txt`
- `experiments/T3_step3/k1_rdir.txt`
- `experiments/T3_step3/rvv_rdir.txt`

被引用工件 — on-device seal(2026-07-06 option-1 promote:`git add -f` 出 gitignore scratch,原路径不改):
- `experiments/ondevice-q5_K/k1_ab_raw.txt` — perf-char ③④+忠实性、T3_A、T3_B、T8(objdump seal + AB)
- `experiments/ondevice-q5_K/rvv_ab_raw.txt` — perf-char ③④+忠实性、T3_A、T3_B、T8
- `experiments/ondevice-q5_K/k1_ab_p3_regres_raw.txt` — perf-char ④(P3 register-resident FALSIFIED)
- `experiments/ondevice-q5_K/rvv_ab_p3_regres_raw.txt` — perf-char ④
- `experiments/ondevice-q5_K/kernel_ours.emitc.mlir` — T3_A、T3_B q5_K 行 emitc= 指针
- `experiments/ondevice-q5_K/kernel_factory.c` — T3_A、T3_B q5_K 对手(ggml factory q5_K_q8_K vec_dot)
- `experiments/ondevice-q8_0-deferred/A_deferred.rv64gcv_zvfhmin.objdump` — T8、gap-log(07-04)
- `experiments/ondevice-q8_0-deferred/A_deferred.o` — vsetvli-diagnosis-2a(07-03,tracked 文件直接引=.o 例外)
- `experiments/ondevice-q8_0-deferred/fair/perf_rvv_vlen128_FAIR.csv` — schema/pattern-registry.v1.json、T8、gap-log
- `experiments/ondevice-q8_0-mbf/seal/fold_segment_objdump.txt` — perf-char ②、T3_A、T3_B、T8
- `experiments/ondevice-q8_0-mbf/kernel_core_mbf2.o` — vsetvli-diagnosis-2a(.o 例外)
- `experiments/ondevice-q8_0-mbf/kernel_q8_mbf1.o` — vsetvli-diagnosis-2a 交叉核对(brace `kernel_{q8_mbf1,ggml_factory}.o`,.o 例外)
- `experiments/ondevice-q8_0-mbf/kernel_ggml_factory.o` — vsetvli-diagnosis-2a 交叉核对(.o 例外)
<!-- REGISTRY:END -->

---

## GITIGNORED SCRATCH(未提升 · 可再生残渣 · 不入机检)

`experiments/ondevice-*/` 三 scratch 目录里,**被引用的证据已于 2026-07-06 全部 promote**(见 REGISTRY
CLASS 3 "on-device seal");**剩下的留 gitignored**——都是可再生残渣,无 tracked 文件引用:

- `.o` 二进制(除被 vsetvli-diagnosis 直接引用的 4 个已 promote 者):objdump 的可重生源。
- 中间 `.cpp` / `.c` 构造源(如 `kernel_ours.cpp` / `kernel_ours_regresident.cpp` /
  `A_deferred.cpp` / `B_perblock.cpp` / `C_factory.cpp` 等):编译中间体。
- `.err` / `.cc.err` 编译日志、`*_rdir.txt` 远端目录快照、`regresident-precheck/`、`fair/ship/` 等 scratch。

留 gitignored 的判据:**无 tracked 文件引用 + 可由重跑板子再生**。若日后某残渣被 tracked 文件引用,
按 option-1 同法 `git add -f` 提升 + 入 REGISTRY CLASS 3(否则 `tools/lint/check_manifest.py` 会红——
promote 后它即 durable content、必须登记)。

---

## PENDING / INTERIM(P3 board-recovery 阻塞 · 不入 REGISTRY · 不当合规格 cell)

以下 cell 是**未完成的 finale perf seal**,board finale 重测(ssh 中断)遗留。**不是合规格 cell**
(缺 stock 侧 + PASS 2 = clean paired 5× 未完成),故**不登记进 REGISTRY**;其部署工件留在磁盘、
经 `experiments/.gitignore` 忽略(未跟踪、不入机检),board 恢复后由 P3 重测完成、届时再 `git add -f`
提升为合规格 cell 并入 REGISTRY。

- `experiments/e2e-harness/results/rvv-vlen128-q4_0-gemm-constructed-redeploy/` — q4_0 REPACK GEMM
  finale perf seal,**PARKED P3**。现有 `phase_split_raw.txt` 只含 ours-pass1(无 stock/pass2),
  **未完成 5×、board 恢复后由 P3 重测完成**。保留的部署工件(P3 board 恢复后可直接复用):
  `constructed_gemm.o` / `tcrv_emitted_repack_gemm_constructed.inc` / `gemm_abi_shim_tail.c` /
  `objdump_seal.txt`(+ `preflight.txt` / `target_profile.txt`)。
  注:这些文件本轮从 tracked 降级(`git rm --cached`,工件留盘不删),以对齐 gitignore 忽略语义。
