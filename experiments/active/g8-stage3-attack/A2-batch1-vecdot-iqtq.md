# A2 batch1 — iq/tq vec_dot 0.8 cold census（双板·clang-18 对称·真板测）

> **任务**：线 A 全量 0.8 攻坚 A2 第一批 = `iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_xs tq1_0 tq2_0` 的 vec_dot 0.8 **cold** 双板补测。
> **赛道**：kernel-sym（kernel-axis MICRO·同编译器/flags/march 对称）。**NOT e2e·NOT perf-covered·不入任何系统账**。[NG-4]。
> **口径（本 batch 实测）**：**cold 唯一**（pool=256MiB oversized-stream·非 hot）· **clang-18 双板对称**（ours+对手同 clang-18·`.comment` 双证）· 对手 = 该板 as-shipped **真派发**（trampoline→`_vlNNN` hand-tuned RVV·nm/objdump 机判探针）· **N=20 rounds median+relIQR·2 seeds 复现** · 预注册判读 cold≥0.8=PASS / <0.8=具名-X+墙 · 0 样本不造数。
> **测于**：2026-07-15 · rvv(VLEN128·core8·load~2.2) + k1(VLEN256·SpacemiT X60·load~2.6) · 主树/build/.so/governor 未改 · 0 stray · 无 git。

---

## 0. 净结论（★ clang-18 对称下 3 格翻正@rvv / 2 格@k1·全部带成色分级）

**承 G7 archive `rvv-iqtq-cold`（gcc-15 对称·8/8 全 LOSS）。本 batch 改 clang-18 双板对称重测——【禁继承】兑现：ours 侧 clang-18 codegen ≫ gcc-15（tq2_0 ours cold 534ns[gcc]→104ns[clang] = 5× codegen 提升·同 [CASE-COMPILER-ASYMMETRY] iq4_nl spill 7/41 现象），部分格 cold verdict 由 LOSS 翻 PASS。**

| 判读（clang-18 对称·cold 2-seed median） | rvv (VLEN128) | k1 (VLEN256) |
|---|---|---|
| **PASS (cold≥0.8)** | 3/8 = iq2_s*, iq4_xs†, tq2_0 | 2/8 = iq4_xs†, tq2_0 |
| **具名-X (cold<0.8)** | 5/8 | 6/8 |

- `†` iq4_xs = **DIVERGE**（fold-order/算法异·ours 走 vluxei codebook-gather·对手无 gather）——ratio 带数值 caveat·非 byte-exact。
- `*` iq2_s@rvv 1.92× = **对手冷态坍塌**（rvv 对手 cold 0.35 GB/s = hot 的 5× 惩罚·ours 0.66 GB/s 仅 1.3×）——**非 ours 硬赢**·板特异（k1 对手 cold 不坍塌 0.98 GB/s → k1 iq2_s LOSS 0.56）。predreg 判 PASS 但成色 = 便宜档·不作硬赢。
- **唯一干净竞争性 PASS = tq2_0 双板**（MATCH fp·ours vsetvli=8 最简·emit 竞争力足·cold parity 0.88–0.98×·"对手贴墙 parity=满分"）。
- **★部署域铁线**：**k1 shipped=clang-18 → k1 clang-18 判读 = 部署域有效**；**rvv shipped=gcc-15 → rvv clang-18 = kernel-axis 对称 micro·【非】部署域**（rvv 部署 verdict 用 gcc-15 = archive 全 8 LOSS）。rvv 3 PASS 不在 rvv 部署域成立（[CASE-COMPILER-ASYMMETRY]·★Amdahl 同域律）。**T3 回填须按此分域标注**（§6 待裁）。

---

## 1. harness 侦察结论（可复现·byte-exact 再生）

- **ours 8 格 kernel 源**：主树无存档（archive 仅存 md5 provenance·board scratch ephemeral）。**从 pinned commit 0ca224f7 工具链重导·8/8 byte-exact 命中 archive md5**（`git show 0ca224f7:test/Target/RVV/<dashfmt>-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir | .worktrees/cache/0ca224f7*/tcrv-opt - --tcrv-rvv-materialize-<dashfmt>-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp`）。symbol=`tcrv_emitc_ggml_vec_dot_<fmt>_q8_K_kernel_rvv_<fmt>_q8_K_block_dot`。
- **driver** = `tools/e2e-harness/board/format_micro_driver.c`（portable host C·ONE side/invocation·oversized-pool cold + 32-block hot·FNV fingerprint xcheck·median+IQR over N rounds）。
- **对手（factory）** = ggml 真派发 vec_dot·从 pinned ggml 源编（**.so 不导出这些 iq/tq 符号·须从源编**）：`arch/riscv/quants.c`（8 格全在此定义·generic quants.c 于 `-march=rv64gcv` 下 #ifdef 剔除→ABSENT）+ generic quants.c·`ld -r` 合并。
  - **rvv 源** = `/home/ubuntu/llama.cpp-upstream-native` @ f3e1828；**k1 源** = `/home/bianbu/tcrv-k1-llama` @ 20039c9（两者皆真 upstream llama.cpp·各板对手 = 各板 as-shipped ggml）。
- **★clang-18 IAS 阻塞 + 解法**：ggml `arch/riscv/quants.c:720` 手写 RVV 内联汇编（`vsetivli zero,16,e8,m1` 无 policy 后缀）**clang-18 集成汇编器拒绝**（正是 archive 用 gcc-15 的原因）。解 = `-fno-integrated-as`（clang codegen + GNU as 汇编内联块）→ `.comment=clang 18.1.8`（合法 clang-18 对象·codegen 全 clang·仅手写汇编经 GAS）。
- **link**：rvv tcrv-clang 独立工具链未接 gcc runtime（crtbegin/-lgcc 缺）→ 用系统 gcc 作 **link driver**（对象仍 clang-18 codegen·link driver 不改被测机器码·均匀作用双方·[PERF-1] 对称门查对象 `.comment` 非 link driver）。k1 clang-18 直接 link OK。
- **对手身份探针**：trampoline `ggml_vec_dot_<fmt>_q8_K` 读 vlenb 运行期分派→尾调 `_vl128`/`_vl256`/`_generic`。**rvv 实测靶 = `_vl128`·k1 = `_vl256`**（真 as-shipped 派发）。density 机判见 §3。

---

## 2. 逐格结果（cold 2-seed·factory/ours·>1=ours 更快·raw 见 -raw/*_sweep.log）

### 2.1 rvv (VLEN128·clang-18 对称·core8)
| 格 | 对手符号(机判·派发靶) | **cold s1** | **cold s2** | ours relIQR% | hot | fp | ours cold GB/s | 对手 cold GB/s | **predreg** | 墙/成色 |
|---|---|---:|---:|---:|---:|:--:|---:|---:|:--:|---|
| iq3_s | `..iq3_s_q8_K→_vl128` | 0.2142 | 0.2126 | 0.2 | 0.179 | MATCH | 0.211 | 0.994 | **具名-X** | 指令数内禀(ours vsetvli=105·对手 _vl128 tighter) |
| iq2_s | `..iq2_s_q8_K→_vl128` | 1.9266 | 1.9193 | 0.3–0.5 | 0.488 | MATCH | 0.66 | 0.346 | **PASS*** | ★对手冷态坍塌(非硬赢·板特异·k1 反证 LOSS) |
| iq2_xs | `..iq2_xs_q8_K→_vl128` | 0.5279 | 0.5295 | 0.6–0.7 | 0.335 | MATCH | 0.80 | 1.52 | **具名-X** | 指令数内禀+对手结构优势 |
| iq2_xxs | `..iq2_xxs_q8_K→_vl128` | 0.7171 | 0.6770 | 3–8 | 0.510 | MATCH | 0.36 | 0.51 | **具名-X** | 指令数内禀(近 0.7·ours IQR 偏高) |
| iq3_xxs | `..iq3_xxs_q8_K→_vl128` | 0.1557 | 0.1560 | 0.4–0.6 | 0.154 | MATCH | 0.205 | 1.31 | **具名-X** | ★指令数内禀 WORST(ours vsetvli=137) |
| iq4_xs | `..iq4_xs_q8_K→_vl128` | 1.2717 | 1.3506 | 1.2–1.5 | 1.379 | **DIVERGE** | 2.75 | 2.2 | **PASS†** | ours vluxei codebook-gather 更快·DIVERGE caveat |
| **tq2_0** | `..tq2_0_q8_K→_vl128` | 0.9813 | 0.9690 | 1.6–2.3 | 1.000 | MATCH | 3.40 | 3.47 | **PASS** | ★干净 parity(ours vsetvli=8·竞争力足) |
| tq1_0 | `..tq1_0_q8_K→_vl128` | 0.2064 | 0.2083 | 1.8–2.4 | 0.393 | DIVERGE | 0.40 | 1.96 | **具名-X** | 指令数内禀+DIVERGE |

### 2.2 k1 (VLEN256·clang-18 对称=部署域·core 0-7 pick)
| 格 | 对手符号(机判·派发靶) | **cold s1** | **cold s2** | ours relIQR% | hot | fp | **predreg** | 墙/成色 |
|---|---|---:|---:|---:|---:|:--:|:--:|---|
| iq3_s | `..iq3_s_q8_K→_vl256` | 0.2001 | 0.2004 | 0.04 | 0.188 | MATCH | **具名-X** | 指令数内禀 |
| iq2_s | `..iq2_s_q8_K→_vl256` | 0.5616 | 0.5631 | 0.1 | 0.581 | MATCH | **具名-X** | 指令数内禀(对手 cold 不坍塌 0.98 GB/s·反证 rvv iq2_s win 板特异) |
| iq2_xs | `..iq2_xs_q8_K→_vl256` | 0.4744 | 0.4741 | 0.07–0.09 | 0.475 | MATCH | **具名-X** | 指令数内禀+对手结构优势 |
| iq2_xxs | `..iq2_xxs_q8_K→_vl256` | 0.6120 | 0.6120 | 0.08–0.12 | 0.604 | MATCH | **具名-X** | 指令数内禀 |
| iq3_xxs | `..iq3_xxs_q8_K→_vl256` | 0.1918 | 0.1917 | 0.04 | 0.177 | MATCH | **具名-X** | 指令数内禀 WORST |
| iq4_xs | `..iq4_xs_q8_K→_vl256` | 1.1963 | 1.1952 | 0.17–0.18 | 1.107 | **DIVERGE** | **PASS†** | ours codebook-gather 更快·DIVERGE caveat |
| **tq2_0** | `..tq2_0_q8_K→_vl256` | 0.8775 | 0.8802 | 0.5–0.7 | 0.755 | MATCH | **PASS** | ★干净 parity(部署域有效) |
| tq1_0 | `..tq1_0_q8_K→_vl256` | 0.6049 | 0.6072 | 0.2–0.3 | 0.413 | DIVERGE | **具名-X** | 指令数内禀+DIVERGE(k1 less bad 0.61 vs rvv 0.21) |

- **byte-exact fp**：6/8 MATCH（iq3_s/iq2_s/iq2_xs/iq2_xxs/iq3_xxs/tq2_0·双板一致），**2/8 DIVERGE**（iq4_xs/tq1_0·与 archive 同一 divergent set·稳定 fold-order/算法差·非新回归）。
- **2-seed 复现**：全格双 seed ratio 差 <1.5%（多数 <0.5%）·relIQR 多 <1%（个别对手侧 cold 4–8% = co-tenant jitter 落快侧·within-proc paired 抵消）。**verdict 稳健**。

---

## 3. 对手成色（机判：真 hand-tuned RVV·非 scalar）

派发靶 `_vlNNN` 全 **真向量化**（objdump robust per-fn density·-raw/*_opp_density_*.txt）：

| | rvv `_vl128` opp_ins / opp_rvv / vsetvli | k1 `_vl256` opp_ins / opp_rvv / vsetvli | ours vsetvli(双板 build_seal) |
|---|---|---|---|
| iq3_s | 114 / 39 / 13 | 244 / 126 / 42 | **105** |
| iq2_s | 425 / 249 / 82 | 356 / 167 / 62 | 81 |
| iq2_xs | 276 / 122 / 49 | 240 / 94 / 41 | 65 |
| iq2_xxs | 157 / 42 / 10 | 357 / 156 / 34 | 41 |
| iq3_xxs | 323 / 113 / 42 | 338 / 112 / 41 | **137** |
| iq4_xs | 128 / 25 / 0 | 167 / 33 / 0 | 26 |
| tq2_0 | 124 / 61 / 15 | 117 / 54 / 8 | **8** |
| tq1_0 | 145 / 68 / 25 | 190 / 126 / 33 | 57 |

- 对手 = **强手调**（39–249 RVV·vsetvli·vluxei gather）·**非 scalar** → LOSS 是真输强手·PASS 是真赢/贴平强手。任务"对手已判手调档 _vlNNN"**证实**。
- **墙机制**：LOSS 全 = **指令数内禀**（ours cold GB/s 0.2–0.8 ≪ DRAM 墙 ~10–20 → 非 memory-bound·是 decode compute/vsetvli-storm bound）。tq2_0 ours vsetvli=8（8 格最简）→ 唯一贴平；iq3_xxs vsetvli=137、iq3_s=105 → 最惨 LOSS 0.16/0.21×。**vsetvli-storm 严重度 ≈ cold verdict 预测器**。

---

## 4. ★关键发现

1. **【禁继承兑现·compiler-delta 实证】** archive gcc-15 对称 = 8/8 LOSS；本 clang-18 对称 rvv 3 PASS。差异全在 **ours 侧 codegen**（tq2_0 ours cold 534ns[gcc]→104ns[clang]·iq4_xs archive 0.769×[gcc]→1.31×[clang]）。同 [CASE-COMPILER-ASYMMETRY]（iq4_nl clang spill=7 / gcc spill=41）。**clang-18 ≫ gcc-15 编我方 EmitC 发射核**。
2. **【部署域分裂】** k1(clang-18=shipped) 判读部署域有效；rvv(gcc-15=shipped) clang-18 = kernel-axis 对称 micro·部署域仍 gcc-15 全 LOSS。**rvv 3 PASS 不可外推 rvv 部署**。
3. **【板不对称·iq2_s】** rvv iq2_s cold 1.92× vs k1 0.56×——差异全在**对手侧冷态行为**（rvv 对手 cold 坍塌 0.35 GB/s / k1 对手守 0.98 GB/s）·非我方差异。=典型"win 由对手冷态坍塌·板特异·非硬赢"教材。
4. **【唯一干净竞争 = tq2_0】** 双板 parity(0.88–0.98)·MATCH·ours emit 最简(vsetvli=8)·真贴平强手调 = "对手贴墙 parity=满分"。iq4_xs 赢带 DIVERGE caveat（算法异）。

---

## 5. 污染纪律 + restore

- **cold 协议**：pool=256MiB oversized-stream（working_set ~268MB·rvv LLC=64MiB→>3×·k1 LLC=L2 512K→>>·真 DRAM cold）·N=20 median+relIQR·2 seeds·warmup dropped·within-proc paired。
- **disjoint-pin**：rvv core8(load-gate idle=100%)·k1 idle-core pick 0-7。
- **restore 双证**：主树/build/stock `.so`/governor **未改**（我方仅从源编 factory.o·未触 .so；k1 `.so` md5=871169a0…）· **0 stray bench**（双板测后 pgrep 清）· board 仅写 ephemeral `/tmp/a2_iqtq`· **无 git add/commit**。
- 原始工件（durable）：`A2-batch1-vecdot-iqtq-raw/`（rvv/k1 `_sweep.log` + `_build_seal.txt` + `_opp_density_*.txt` + `_ours_density.txt` + `exported_kernel_md5.txt` + runner 脚本）。

---

## 6. T3 回填清单（★留主会话机算入库·本 agent 不动 T3·部分待裁）

T3_kernel_unit.csv 现 8 格 `vec_dot|<fmt>` 的 `rvv_0p8_disp` 与 `k1_0p8_disp` 均 =`(不在T3)`。建议填（predreg 落地）：

| kernel(op\|format) | **k1_0p8_disp**(clang-18=部署域·直填) | **rvv_0p8_disp**(clang-18-对称-micro / rvv 部署=gcc-15) |
|---|---|---|
| vec_dot\|iq3_s | 具名-X(cold 0.200·指令数内禀) | 具名-X(clang 0.213·gcc 部署 0.287 archive·两域皆 X) |
| vec_dot\|iq2_s | 具名-X(cold 0.562·指令数内禀) | ⚠**待裁**(clang 1.92 PASS-by-对手冷态坍塌·板特异·gcc 部署 0.490 X) |
| vec_dot\|iq2_xs | 具名-X(cold 0.474) | 具名-X(clang 0.529·gcc 0.318 X·两域 X) |
| vec_dot\|iq2_xxs | 具名-X(cold 0.612) | 具名-X(clang 0.697·gcc 0.779 X·两域 X) |
| vec_dot\|iq3_xxs | 具名-X(cold 0.192) | 具名-X(clang 0.156·gcc 0.168 X·两域 X) |
| vec_dot\|iq4_xs | PASS†(cold 1.196·DIVERGE caveat) | ⚠**待裁**(clang 1.31 PASS†-DIVERGE·gcc 部署 0.769 X) |
| vec_dot\|tq2_0 | **PASS**(cold 0.879·干净 parity·部署域) | ⚠**待裁**(clang 0.975 PASS·gcc 部署 0.213 X) |
| vec_dot\|tq1_0 | 具名-X(cold 0.606·DIVERGE) | 具名-X(clang 0.207·gcc 0.290 X·两域 X) |

**★须主会话/用户裁的 policy（canon 级·measurement-domain 定义）**：**rvv_0p8_disp 的 0.8 verdict 用哪个编译器域？**
- **选项 A（clang-18 对称 micro）**：与 k1 同 caliber·rvv 记 iq2_s/iq4_xs/tq2_0 PASS(带成色 caveat)。
- **选项 B（部署域 = rvv shipped gcc-15）**：rvv 记 archive gcc-15 = 8/8 X（部署一致·符合 ★Amdahl 同域律）。
- 我方**不自决**（触碰 0.8 轴的编译器域定义 = measurement-domain canon）。**建议**：主表分两列/标 tag（`rvv-clang18-micro` vs `rvv-gcc15-deployed`），避免 clang-micro 数被当 rvv 部署赢外推（[CASE-COMPILER-ASYMMETRY] 前车）。
- **净可直接入库（无争议）**：k1 8 格全部（部署域·2 PASS[iq4_xs†/tq2_0] + 6 具名-X）；rvv 5 格两域皆 X（iq3_s/iq2_xs/iq2_xxs/iq3_xxs/tq1_0）。**余 rvv 3 格待 policy 裁**。

**纪律边界**：本 16 datapoint = kernel-axis 对称 micro·**NOT e2e·NOT perf-covered·不入 ≥parity 系统计数**。3 PASS 中唯 tq2_0 干净竞争(parity·非 beat)·iq2_s@rvv=对手冷态坍塌·iq4_xs=DIVERGE——**均不作"硬赢强手调"**。
