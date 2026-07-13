# G7 L1 货架A 全量 — rvv §1.2 IQ/TQ vec_dot 族 + iq4_nl gemm 补 cold（同形状·gcc-15 对称）

> **赛道**：第二赛道 **kernel-sym**（kernel-axis MICRO·同编译器/flags/march 对称）。**NOT e2e·NOT perf-covered·与 perf-covered 系统账（9/83）永不混算**。[NG-4] 纪律·e2e 冻结令期间未开 e2e。
> **验证阶梯层**：[VERIFY-LADDER] **G2 同形状 cold micro**（vec_dot 格测 vec_dot 冷态·gemm 格测 gemm 冷态）。
> **板**：rvv / localhost.localdomain / openEuler / VLEN128 (vlenb=16) / **core 8**（disjoint-pin 8-15·co-tenant vLLM 在别核·未重启）/ **gcc-15.2.0 出货对称域**（rvv shipped=gcc-15 → kernel-axis==system-axis·[CASE-COMPILER-ASYMMETRY] not triggered on the symmetric account）。
> **★板事实（cold 协议锚）**：rvv **L1d=64K · L2=2MiB · L3=64MiB**。vec_dot cold pool=256MiB（>3×LLC=192MiB → 真 DRAM cold）·hot pool=32-block（~12KB → L1-resident）。iq4_nl gemm cold=224MiB flush/rep（>3×L3）·hot=no-flush。
> **测于**：2026-07-14 · loadavg ~2.3（co-tenant vLLM 在别核·within-proc paired ratio 抵消共同 contention）。**禁 git·禁改 T9/ROADMAP·数据给主会话**。

---

## 0. 净结论（★ 9 cell 全 <parity·cold≈hot·verdict 不翻·两个惊喜/校正）

**承 L1 rvv-batch（FLAT-5 §1.1）· 本 batch = §1.2 IQ/TQ vec_dot 族（8）+ iq4_nl gemm（1）= 9 cell 补 cold。全 <parity（已知 LOSS）·实测确认 cold≈hot·verdict 不翻·不入 kernel-sym ≥parity 计数。**

| 格·板 | 类型 | 对手符号(机判) | **HOT** | **COLD** | batch2c(cold ref) | fp | cold regime | <parity? |
|---|---|---|---:|---:|---:|:--:|---|:--:|
| iq3_s @rvv | vec_dot | `ggml_vec_dot_iq3_s_q8_K` (T) | 0.2925× | **0.2868×** | 0.2804× | MATCH | compute(latency) | ✅ LOSS |
| iq2_s @rvv | vec_dot | `ggml_vec_dot_iq2_s_q8_K` (T) | 0.5886× | **0.4896×** | 0.4772× | MATCH | compute | ✅ LOSS |
| iq2_xs @rvv | vec_dot | `ggml_vec_dot_iq2_xs_q8_K` (T) | 0.1583× | **0.3183×** | 0.3293× | MATCH | compute | ✅ LOSS |
| iq2_xxs @rvv | vec_dot | `ggml_vec_dot_iq2_xxs_q8_K` (T) | 0.5462× | **0.7792×** | 0.7796× | MATCH | compute(mildest) | ✅ LOSS |
| iq3_xxs @rvv | vec_dot | `ggml_vec_dot_iq3_xxs_q8_K` (T) | 0.1493× | **0.1683×** | 0.1583× | MATCH | compute(WORST) | ✅ LOSS |
| iq4_xs @rvv | vec_dot | `ggml_vec_dot_iq4_xs_q8_K` (T) | 0.6821× | **0.7688×** | 0.7178× | DIVERGE | compute | ✅ LOSS |
| **tq2_0 @rvv** | vec_dot | `ggml_vec_dot_tq2_0_q8_K` (T) | **★1.1515×** | **0.2128×** | 0.2311× | MATCH | ★hot≥parity/cold LOSS | ✅ cold LOSS |
| tq1_0 @rvv | vec_dot | `ggml_vec_dot_tq1_0_q8_K` (T) | 0.6717× | **0.2903×** | 0.5505× | DIVERGE | compute | ✅ LOSS |
| **iq4_nl @rvv** | **gemm** | `ggml_vec_dot_iq4_nl_q8_0` (T·cross-op) | 0.224× (nr64) | **0.228× (nr64)** | — | oracle | gather(latency) | ✅ LOSS |

- **9/9 cell cold <parity**（LOSS verdict 不翻·符合预期·**不入 kernel-sym ≥parity 计数 9**）。
- **对手身份 = 机判探针**（nm -D/nm → `ggml_vec_dot_<fmt>_q8_K` / `ggml_vec_dot_iq4_nl_q8_0` DEFINED-T·real ggml dispatched·**禁手写类目**·重编令二.2 稻草人合规）。vec_dot 8 格对手从 pinned ggml 源（`/home/ubuntu/llama.cpp-upstream-native` = 建出货 .so 的同源树）gcc-15 -O3 编 factory.o（.so 未导出这些 vec_dot 符号·须从源编）；iq4_nl 对手直取 .so 导出符号 @0x937da。
- **两个 headline 校正/惊喜见 §3-§4**：① tq2_0 hot 1.15× ≥parity（L1-resident·cache-态 only·cold 蒸发）；② iq4_nl gemm gcc-symmetric = 0.23×（校正 l1-m2-iq4 clang-asymmetric 0.837×·匹配 g5 e2e 0.217×）。

---

## 1. cold 协议口径（同形状·两个子协议）

### 1.1 vec_dot 8 格（同形状 = vec_dot·format_micro_driver.c·N=12 rounds median+relIQR）
- **hot** = pool_mib=0 → 32-block pool（~12KB < L1d 64K）· 内建 warmup pass（dropped）· 每 round 扫全 pool（2 calls·L1-resident 复现·median-of-12）。
- **cold** = pool_mib=256 → working_set ~268MB（**>3×LLC=192MB → 每 weight tile 从 DRAM cold-read**·batch2c 同协议复现）· activation 单份复用（暖激活/冷权重）。
- **同编译器对称**：OURS kernel（front-door 导出 `weft-opt`/`.worktrees/cache/0ca224f7 tcrv-opt` → `mlir-translate --mlir-to-cpp` → gcc-15.2.0 g++ -O3 编）+ factory（pinned ggml `arch/riscv/quants.c`+`quants.c` gcc-15.2.0 -O3 编·ld -r 合并）· 全 gcc-15.2.0 -O3 · march=rv64gcv_zfh_zvfh_zba_zbb_zbs。OUR .o 全 **cleanfp**（0 __extendhfsf2·hw fp16）。
- **correctness（ZERO-MODEL-ish FNV over outputs·同 seed·in-driver）**：ours==factory bit-match **6/8**（iq3_s/iq2_s/iq2_xs/iq2_xxs/iq3_xxs/tq2_0·含 2 sealed byte-exact）；**2/8 DIVERGE**（iq4_xs/tq1_0·**与 batch2c 同一 divergent set** = 稳定 fold-order/decode 差·非新回归·ratio 带 caveat）。

### 1.2 iq4_nl gemm（同形状 = repack GEMM·iq4nl_gemm_paired_driver.c·N=12 reps·2 seeds·nr{64,16}）
- **hot** = no-flush（weight tile cache-resident 复用）· **cold** = 224MiB flush/rep（>3×L3·每 rep DRAM cold）。
- OURS = front-door 导出 `weft-opt rvv-to-emitc-repack-gemm-iq4-nl-q8-0.mlir --weft-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp` → gcc-15.2.0 -O3 编（symbol renamed weft_emitc→driver extern）· codebook `vluxei16` ×64 gather。
- 对手 = `ggml_vec_dot_iq4_nl_q8_0`（.so·gcc-15 出货·**cross-op**：我方 repack GEMM vs ggml per-(row,col) vec_dot block-dot·iq4_nl @VLEN128 vec_dot 路无 repack trait → ggml 退 block-dot）。
- correctness = **construction oracle**（iq4 direct-emitter byte-exact @silicon-validation-batch-1·ULP=0）· driver 为 timing-only（数据无关 steady-state·exact strides·独立随机填充·ratio 不做数值 xcheck·同 l1-m2-iq4 方法学）。

---

## 2. cold≈hot 逐格（compute/gather-bound → cache 态二阶·符合 FLAT-5 §4 机制①）

**8 vec_dot（ratio = factory/ours·>1=ours faster·全 <1 = LOSS）：**
| 格 | hot | cold | Δ(cold−hot) | ours cold GB/s | 机制 |
|---|---:|---:|---:|---:|---|
| iq3_s | 0.2925 | 0.2868 | −0.006 | 0.184 | compute·cold≈hot |
| iq2_s | 0.5886 | 0.4896 | −0.099 | 0.835 | compute·cold 略降（ours 更 memory-exposed） |
| iq2_xs | 0.1583 | 0.3183 | +0.160 | 0.418 | compute·cold**升**（factory L1-fast 优势冷态失） |
| iq2_xxs | 0.5462 | 0.7792 | +0.233 | 0.391 | compute(mildest)·cold 升 |
| iq3_xxs | 0.1493 | 0.1683 | +0.019 | 0.204 | compute(WORST)·cold≈hot |
| iq4_xs | 0.6821 | 0.7688 | +0.087 | 1.510 | compute·cold 升〔DIVERGE caveat〕 |
| tq2_0 | ★1.1515 | 0.2128 | **−0.939** | 0.682 | ★hot≥parity→cold 蒸发（§3） |
| tq1_0 | 0.6717 | 0.2903 | −0.381 | 0.655 | compute〔DIVERGE caveat〕 |

**iq4_nl gemm（ratio = ours_gmacs/opp_gmacs·OURS gather-bound flat 0.667 gmacs 跨全 shape/态）：**
| shape | hot(2 seeds) | cold(2 seeds) | Δ | 机制 |
|---|---:|---:|---:|---|
| nr=64（锚·= l1-m2 anchor） | 0.224/0.225 | 0.228/0.229 | +0.004 | gather-bound·cold≈hot（ours 0.671→0.667 gmacs·几无冷态变化） |
| nr=16 | 0.194/0.201 | 0.204/0.205 | +0.007 | gather-bound·cold 略升（opp nr16 更快 3.3-3.5 gmacs·parity 降全在 opp 侧） |

> **cold≈hot 普适**：compute-latency-bound（vec_dot decode）/ codebook-gather-bound（iq4_nl vluxei16）两侧带宽都远低 DRAM 墙 → cache 态二阶（与 FLAT-5 §4 finding① q5_0 型一致·与 k1 L2 cold≈hot 同型）。ours 侧 cold GB/s 0.18–1.5 ≪ ~10-20GB/s DRAM 墙 = 非 memory-bound。**net：cold 不改任何 verdict·15/15+ vec_dot cell + 8 iq4_nl cell 全 LOSS 稳健。**

---

## 3. ★惊喜 tq2_0：hot 1.15× ≥parity → cold 0.21× 蒸发（cache-resident-only micro-win）

- **hot（L1-resident·3 seeds·N=20）**：1.1515× / 1.1500× / 1.1515×（ours_relIQR ~1.0%·rock-solid ≥parity）。ours tq2_0 61.9ns/blk **BEATS** factory 71.9ns。**根因 = OUR vsetvl=3**（8 格里最少·近零 vsetvli storm·tq2_0 ternary 2-bit 最简 decode）→ L1-hot compute-only 我方竞争力足。
- **cold（N=16）**：0.2128×（ours 534.7ns vs factory 113.8ns）。**ours cold-penalty 61.9→534 ns = 8.6×**·factory 仅 71.9→114 = 1.6× → 极不对称冷态惩罚。ours tq2_0 hot 是 compute-bound·cold 变 DRAM-latency-bound（0.682 GB/s）·factory block-dot 冷态更稳。
- **判读（thesis-coherent·非翻正）**：cache-resident micro-win **不存活 DRAM streaming**（[CASE-MICRO-E2E] 型正面教材·同 memory[kernel-wins-dont-transplant-to-e2e]）。**verdict = cold LOSS 0.21×**（与 batch2c cold 0.231× 一致）；hot 1.15× 仅 L1-态·**不入 ≥parity 计数**（kernel-sym 计数口径 = cold/e2e-存活·非 cache-resident 峰值）。如实报为惊喜 + 为何不计。

---

## 4. ★校正 iq4_nl gemm：gcc-symmetric 0.23×（l1-m2-iq4 clang 0.837× = 编译器不对称 artifact）

- **本 batch gcc-15 -O3 对称**：nr=64 cold **0.228×** / hot 0.224×·nr=16 cold 0.205× / hot 0.197×。OURS gather-bound flat **0.667 gmacs**。
- **对比 l1-m2-iq4（2026-07-09·clang-17 -O2/-O3 ours vs gcc-shipped .so opp）= 0.837×（nr64）/0.715×（nr16）**：那是 **clang-ours-vs-gcc-shipped = 典型 [CASE-COMPILER-ASYMMETRY]**（clang 抬高我方）。
- **机判铁证（spill 计数·同 kernel.c·仅编译器差）**：**clang-18 -O3 ours spill=7**（复现 l1-m2 objdump spill=7·clang GMAC/s 2.44）· **gcc-15 -O3 ours spill=41**（gcc 多 spill 6× → ours 0.667 gmacs = clang 的 1/3.7）。codebook-gather 核对编译器**高度敏感**（此处反向：clang≫gcc·非 repack 常见的 clang-flatter·同族现象）。
- **gcc-symmetric 与部署域一致性**：本 gcc-symmetric cold **0.227× ≈ g5-wiring e2e prefill 0.217×**（两者皆 gcc-15 出货域·deployed .o 正门）。→ **iq4_nl 部署域 verdict = ~0.22-0.23× LOSS**·l1-m2 的 0.837× 应标 clang-asymmetric（不作对称 verdict）。符合 CLAUDE.md 性能规则3/★Amdahl 同域律。

---

## 5. 污染纪律 + restore（rvv 共享板）

- **disjoint-pin core 8**（co-tenant vLLM/kernspan 在别核·测前测后未触碰/未重启·load 2.2–3.1）。
- **relIQR**：vec_dot ours 侧全 <2.5%（多数 <1%）；**factory 侧 cold 数格 8-15%**（iq3_xxs 15.1% / iq4_xs 12.9% / iq2_s 10.1% / iq3_s 4.7%）= co-tenant jitter 落在**快的一侧**（~200ns 短跑对调度抖动敏感）。**N=16 复测 medians 重现**（iq3_xxs 0.168→0.170·iq4_xs 0.769→0.761·iq2_s 0.490→0.494）→ ratio 稳健（LOSS margin 巨大·远离 parity·verdict 不受威胁）。within-proc paired ratio 抵消共同 contention。iq4_nl noisefloor 140-300ns ≪ 25-100ms region。
- **restore（board 仅写 scratch `/tmp/g7_iqtq_cold/`·ephemeral 2.1M）**：主树 / build / stock ggml .so / governor 全未改。**stock `libggml-cpu.so` md5 `d1adc634…` 测前测后 UNCHANGED = restore 双证**（我方仅链接·未改）。**无 git add/commit。**
- **stray proc**：测后 `pgrep iqtq_bench|iq4nl_bench|llama-cli` = **NO_STRAY_PROCS**（无 hung llama-cli·无遗留 bench）。

---

## 6. 进度分数 + T9 §1.2 cold 建议（留主会话改·本 agent 不动 T9）

**进度分数**：**rvv-半 §1.2 IQ/TQ+iq4_nl 子集 = 9/9 cell filled**（8 vec_dot hot/cold + iq4_nl gemm hot/cold nr{64,16}×2seed）。§1.2 余 4 格 = **q5_K/q2_K/q3_K/q6_K @rvv gemm（K-quant·非本 task 域·另 batch）**。→ **rvv §1.2 可填（本 task 域）= 9/9 DONE**。合 §1.1 FLAT-5（5/5·rvv-batch）→ rvv 全量表 IQ/TQ/FLAT 部分收口。

**T9 §1.2 cold 子列建议**（每行加 `{cold ratio(同形状) | cold regime | cold<parity 维持}`·主会话入账）：

| 格·板 | 已有 hot/verdict | **cold(本 batch·同形状)** | cold regime | verdict |
|---|---|---:|---|:--:|
| iq3_s @rvv vec_dot | 0.280× | **0.287×** | compute-latency | LOSS 维持 |
| iq2_s @rvv vec_dot | 0.477× | **0.490×** | compute | LOSS 维持 |
| iq2_xs @rvv vec_dot | 0.329× | **0.318×** | compute | LOSS 维持 |
| iq2_xxs @rvv vec_dot | 0.780× | **0.779×** | compute(mildest) | LOSS 维持 |
| iq3_xxs @rvv vec_dot | 0.158× | **0.168×** | compute(WORST) | LOSS 维持 |
| iq4_xs @rvv vec_dot | 0.718× | **0.769×**〔DIVERGE〕 | compute | LOSS 维持 |
| tq2_0 @rvv vec_dot | 0.231× | **0.213×**（★hot 1.15× cache-only·见 §3） | compute·冷态蒸发 | cold LOSS 维持 |
| tq1_0 @rvv vec_dot | 0.550× | **0.290×**〔DIVERGE〕 | compute | LOSS 维持 |
| **iq4_nl @rvv gemm** | prefill 0.217× (g5 e2e) | **0.228× (nr64) / 0.205× (nr16)**·gcc-symmetric | codebook-gather | LOSS 维持·**★校正 l1-m2 clang 0.837×（§4）** |

**纪律边界**：本 9 cell cold **仅 kernel-axis 覆盖面 robustness 补强·NOT e2e·NOT perf-covered·NOT 新格**·kernel-sym ≥parity 计数（9）**不因本 batch 变**（cold=hot robustness·全 LOSS 不入计数）。tq2_0 hot 1.15× = cache-resident-only·不计。iq4_nl 0.23× = 部署域 gcc-symmetric verdict（撤 l1-m2 clang-asymmetric 0.837×·建议主会话据此收窄 l1-m2-iq4 措辞）。
