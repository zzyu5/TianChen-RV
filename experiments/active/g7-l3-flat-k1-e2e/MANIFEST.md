# MANIFEST — g7-l3-flat-k1-e2e（货架B·FLAT 5 @k1 e2e·dual-board 矩阵首批）

- **令**：G7 L3 双板 e2e 矩阵·货架B。任务 = FLAT 5（q4_0/q4_1/q5_0/q5_1/q8_0）@k1 e2e·dual-board 成色。
- **board**：`ssh k1` SpacemiT X60 VLEN256·8 harts·stock clang-18·DVFS perf-gov 1.6GHz locked·pin 0-3 -t4。
- **约束**：NO git · NO schema/T8/ROADMAP 改（label 建议留主会话）· NO rvv（q5_K 在飞）· board reversible（私有 repack.cpp copy·shared source NEVER edited）· 双账本 clang-18 对称（kernel==system·[CASE-COMPILER-ASYMMETRY] not triggered）。
- **方法**：repack-vs-block-dot 对照（[WORK-ITEM-K1-KQUANT-E2E] q4_K 已验证 k1 e2e 方法）· ONE 真 ELF llama-bench · physical .so swap · interleaved paired A/B · n=12/side · relIQR hygiene。

## 文件

- `ROUTE_AND_GAP.md` — 全 5 格路由确认（objdump/dispatch）+ q4_1/q5_0/q5_1 结构缺口（k1 stock=block-dot·需 net-new deploy）。
- `q4_0/evidence.md` + `q4_0/{build_seal_raw.txt,measure_raw.txt,correctness_clean.txt,gen_texts/}` — q4_0 dual-board e2e 完整证据。
- `q8_0/evidence.md` + `q8_0/{build_seal_raw.txt,measure_raw.txt,...}` — q8_0 dual-board e2e 完整证据。
- `flat_k1_build.sh` / `flat_k1_measure.sh` / `analyze.py` / `correctness_clean.sh` — 可复现 harness（board harness 住 `/tmp/flat_*`·build dir `/data/build-k1-flat`）。

## 结论摘要（逐格）

| 格 | 路由(k1 stock) | correctness | prefill e2e | decode e2e | 出口 |
|---|---|---|---|---|---|
| **q4_0** | 16x1 repack (case256) | GREEN 3/4 + 1 near-tie | **5.1778× WIN** | 1.6588× WIN | **≥parity → dual-board 成色** |
| **q8_0** | 16x1 repack (case256) | **GREEN 4/4 byte-identical** | **2.3488× WIN** | 1.2122× WIN | **≥parity → dual-board 成色** |
| q4_1 | block-dot (no repack) | — | — | — | deploy-net-new 跟进项（结构缺口·pending） |
| q5_0 | block-dot (no repack) | — | — | — | deploy-net-new 跟进项（pending） |
| q5_1 | block-dot (no repack) | — | — | — | deploy-net-new 跟进项（pending） |

**board restored（md5 双证·`board_restore_proof.txt`）**：shared source GEN/HDR/ARCH == baseline（3cac40aa/57851439/c3c101fd）· 出货 stock lib 871169a0 UNTOUCHED · route-markers=0 · .ORIG=0 · /data/build-k1-flat 已删 · /tmp/flat* 已清 · 0 stray procs。污染状态 = CLEAN（relIQR 全 <0.4%·freq 全程 1.6GHz·interleaved paired）。

## ★诚实 scope（关键）
- q4_0/q8_0 e2e WINNER = stock 自己 clang-编译的 16x1 repack（as-shipped 默认路径），**非我方 compiler-emitted kernel** ⇒ **不新增 perf-covered green**（同 q4_K workitem note）。价值 = dual-board 成色（repack approach e2e 在 k1 传导 ≥parity·双板证据）+ margin 量化。our-emit↔stock-repack 的 kernel-axis parity 已封（T9 q4_0@k1 1.0022× / q8_0@k1 +4.37%）。
- 双账本 clang-18 对称·kernel==system·同数。
