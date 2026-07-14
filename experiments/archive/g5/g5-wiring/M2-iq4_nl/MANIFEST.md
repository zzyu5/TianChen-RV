# G5-M2 iq4_nl 曳光弹 — L-接线② codebook 格 · flip-gate+intercept · CASEFILE

> **campaign**: G5 接线战役 · **M2 L-接线② = iq4_nl**（首个 codebook 格接线到 gemm-轴公平判读·此前 report「无 gemm 对位」= 黄-未接线）
> **board**: `ssh rvv` openEuler VLEN128 gcc-15.2.0 nproc=64 · A-tree restored baseline（GEN=deb61a29/ARCH=99131cf7·live .so 05a62e6a 0 syms·zero net change）
> **HEAD (TianChen-RV)** = 禁 git·零 tracked lib 源改动·2-file wiring 全在 board deploy patch
> **框架**: **≠ q6_K 净新全 scaffold · ≠ q4_1 零 scaffold · = q8_0-M1b「flip-gate + intercept」**——iq4_nl 上游 riscv `1,16` repack scaffold **全 present 但 VLEN128 破损（硬编码 AVL=16 + vrgather）且 dispatch gate OFF（`case 128: break // TODO`）→ stock 恒 block-dot**·我方 = 翻 gate + emitted vl=8（memory-gather vluxei16）拦截破损上游 body（correctness-carrier）
> **correctness**: **GREEN** — silicon UT byte-exact（INT_mismatch=0·worst_norm 2.163e-06·6 shapes GEVM+GEMM）+ greedy A==B **5/5 byte-identical**（38 engage banners·no NaN/Inf·CORRECTNESS_GATE GREEN）。
> **perf/verdict**: **BLOCKED-environmental** —— 共享板过载（load 22→30·他用户 VLLM+paper3-llama+ai-test）致 llama-bench 间歇 mmap SIGBUS·**stock OFF（pristine 05a62e6a）同崩** = 环境级非 kernel 缺陷（correctness 单进程全 clean 反证）。全 phase_split（avg_ts=0）+ retry 小批（0 OK）两轮全废 → **无 clean ratio·perf-covered 维持 6/84**·harness 就绪待安静板复跑。
> **登记性质**: iq4_nl = **首个 codebook 格 correctness-carrier**（黄-未接线 → correctness-wired）· C1 template flip-gate+intercept 模式延伸到 codebook 家族 · perf 轴 env-blocked（未测得·非 loss/win）。A-tree restore clean（源 baseline·live 05a62e6a·0 syms）。

## durable files

- `evidence.md` — recon + emit + 2-file scaffold + silicon UT + build/seal + correctness + perf(env-blocked) + A-tree restore 全数据
- `correctness_GREEN_raw.txt` — greedy A==B 5/5 byte-identical · 38 engage banners · PPL(Bus-error non-fatal) · CORRECTNESS_GATE GREEN
- `perf_BLOCKED_raw.txt` — perf env-block 证据（stock OFF `-p128` Bus-error 同崩 = 环境非 kernel · retry 0-OK · board load 22→30 他用户名单）
- `.gitignore` — gitignore 两 emitted .inc（357KB·regenerable via evidence.md §二 emit recipe）
- `perf_rerun_quiet-board.md` (+ `perf_rerun_quiet-board_raw.txt` / `perf_rerun_correctness_sanity_raw.txt`) — **quiet-board catch-window measured perf**（append-only 续页·不改 evidence.md §七 主体）：§七 `BLOCKED-environmental` **升为 measured-loss** = **yellow-对手更强**（prefill pp128 ON 0.7688 / OFF 3.544 = **0.217×·4.61× LOSS**·decode tg32 ON 0.9549 / OFF 1.892 = **0.505×·1.98× LOSS**·全 relIQR<0.3%·DVFS 锁 2.6GHz·八门全过·双账本对称·对手=stock iq4_nl block-dot 非 SELF·具名 GAP `uarch.gather_slow`+narrow-vl·可修=争议）·**perf-covered 维持 6/83**（非 green·非入台账 win）·correctness 2/2 A==B 再确认无回归·A-tree restore clean（源 baseline·live 05a62e6a·0 syms）。

> emitted `.inc` = **gitignored**（`tcrv_emitted_gemm_iq4_nl.inc` md5 1e040596·243KB / `tcrv_emitted_gevm_iq4_nl.inc` md5 b7a5267e·114KB·regenerable·避 repo bloat）。
> board harness 脚本住 `tools/e2e-harness/board/g5-m2-iq4_nl/`（deploy_patch_iq4_nl_emitted.py · ut_iq4_nl_verify.cpp · ut_iq4_nl_build_run.sh · g5_m2_iq4_nl_build_seal.sh · g5_m2_iq4_nl_correctness.sh · g5_m2_iq4_nl_phase_split.sh · g5_m2_iq4_nl_perf_split2.sh · g5_m2_iq4_nl_perf_retry.sh · analyze_phase_split.py · g5_m2_iq4_nl_driver.sh）。**perf 复跑仅需板 load<~10 安静窗口**（harness 完备·A-tree restore 干净）。
