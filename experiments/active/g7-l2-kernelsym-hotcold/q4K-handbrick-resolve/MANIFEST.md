# cell MANIFEST — q4K-handbrick-resolve (G7 货架A · T9 §6 争议解)

- **campaign**: G7 货架A / kernel-sym 成色核实 · **task**: q4_K@k1 hand-brick 对手争议另测
- **status**: ACTIVE · board-measured 2026-07-13 · board=k1 (SpacemiT X60/VLEN256) · **NOT committed** (user commits)
- **role**: 解 T9 §6 争议——q4_K@k1 kernel-sym 成色 = hand-brick 还是 block-dot？
- **[NG-4] DISCIPLINE**: 第二赛道 kernel-axis 对称 micro A/B · **NOT** e2e beat · **NOT** sealed 8-gate Win · **与 perf-covered 系统账永不混算**。

## 裁决（一行）
**"赢 hand-brick" 主张 = 证伪（FALSIFIED）**：our-emit q4_K@k1 = **0.622× < parity** vs 真出货 hand-brick
`ggml_gemm_q4_K_16x1_q8_K`（对称 clang-18·VLEN256-native·真部署路径）。sealed 3.106× 是 vs **block-dot** 的
（三重证据）·成色 = **vs-block-dot**·**0 verified hand-brick 确认**。

## 关键数（同板 K=2048 nr=64 nc=512·同 vx/vy·对称 clang-18·N=12·3 seed·HOT+COLD）
- ours / 真-16x1 = **0.622×**（HOT∧COLD∧O2∧O3 全一致·IQR<0.3%）→ hand-brick 快 1.61×。
- 三方: block-dot 1.146 < **ours 3.72（3.25×≈sealed 3.106×）** < **16x1 hand-brick 5.98（5.22×·1.61×over ours）** GMAC/s。
- correctness: ours == 16x1 bounded-ULP（max_abs 2.9e-3·全 nr 行匹配）→ A/B 公平非空转。

## 对手身份（objdump/nm 证）
- sealed 3.106× 对手 = `ggml_vec_dot_q4_K_q8_K` **block-dot**（t4a MANIFEST + paired driver:49,119 + runner:24 三证）。
- 真 hand-brick = `ggml_gemm_q4_K_16x1_q8_K` @0xabe58（stock lib·case256 ON 真出货·VLEN256-native e32m2/e16m1/e8mf2/m4·clang++-18 -O3 per compile_commands.json）。
- VLEN 键控: 16x1 hardcode vl=16 → VLEN128(rvv) 破损·VLEN256(k1) 满宽正确 → k1 专调 hand-brick。

## durable files
- `evidence.md` — 完整对手身份证 + micro A/B + 裁决 + T9 §6 成色分布最终建议。
- `summary_q4k_handbrick.csv` — 12 行原始 (HOT/COLD × O2/O3 × 3 seed)。
- `raw/run_k1_q4k_handbrick.log` — 板 run 全 log（preflight/compiler 对称证/seal/load-gate/correctness/timing）。
- `q4k_handbrick_driver.c` — A/B driver（同 vx/vy·correctness+HOT+COLD·paired）。
- `run_k1_q4k_handbrick.sh` — 板 runner（对称双 recipe 编译·seal·load-gate·3 seed）。
- board scratch: `k1:/tmp/q4k_hb_resolve/`（ephemeral）；主树/build/governor/stock-lib **UNTOUCHED**·零 restore。
- ours: `s6_q4K.c` md5 **90d454da**（== t4a·no regen·板双证）。
