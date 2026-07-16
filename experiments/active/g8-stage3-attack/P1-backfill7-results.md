# P1 — 可补测 7 板格 · 实测结果（按 PREREG 判读 · 测后 · 主会话入表用）

> 预注册：`P1-backfill7-prereg.md`（测前写死）。本文件 = 落地填数，**判据未改**。
> raw 工件：`P1-backfill7-raw/`。**禁 git commit**（主会话统一入库）。**T3 未改**（主会话机算入表）。
> 单世界 clang-18。**零 gcc 输出**。硬冻结（102/105 · 101/108 · 9/83 · roster $meta · 队序）**未碰 = 只登记不执行**。

## 净结论

**P1 七格：PASS 2 / 具名-X 1 / VOID 4**（pending 24 → **21**，非预期的 17）。
**0 verified hand-brick win**。便宜档 2 格禁称硬赢。CROSSOP 1 格已标。k1 decode = what-if。

| # | 板 | op/format/regime | verdict[0.8] | cold ratio |
|---|---|---|---|---|
| 1 | rvv | dequantize_row/iq2_xs | **VOID-NOISE** | 未测·0 样本 |
| 2 | rvv | dequantize_row/iq2_s | **PASS[0.8]** | 1.7534 / 1.7472（便宜档） |
| 3 | rvv | dequantize_row/nvfp4 | **PASS[0.8]** | 2.1736 / 2.1689（便宜档·★我方核亦标量） |
| 4 | rvv | gemm_tile/iq4_nl decode | **VOID-NOISE** | 未测·0 样本（噪声门 2/2 不过） |
| 5 | rvv | gemm_tile/iq4_nl prefill | **VOID**（harness 无 ZERO-MODEL） | 未测·0 样本 |
| 6 | k1 | gemm_tile/iq4_nl decode | **具名-X[0.8]** | OPP-X 0.2493/0.2490 · OPP-S 0.1469/0.1473 |
| 7 | k1 | gemm_tile/iq4_nl prefill | **VOID**（同 #5） | 未测·0 样本 |

---

## ★ 预注册的三处事实错误（测中发现 · 已订正 · 登记待裁）

### E1. 探针 `\t` 失效 → prereg §2 的 rvv 计数全部无效（实为 vset 计数）
`grep -E '\tv[a-z]'`：GNU grep -E **不解释 `\t`**（warn `stray \ before t`）→ 模式退化成字面 `tv[a-z]`。
prereg §2 全部 rvv 列 = **artifact**。已改 TAB 字段感知 awk（`objdump -d` 行 = addr\thex\tmnemonic\top）双法复核：

| 符号 | prereg §2 rvv | **实测 rvv** | 说明 |
|---|---:|---:|---|
| `dequantize_row_iq2_xs` | 5 | **15** | prereg 的 5 = vset |
| `dequantize_row_iq2_s` | 33 | **107** | prereg 的 33 = vset |
| `dequantize_row_nvfp4` | 6 | **30** | prereg 的 6 = vset |
| `ggml_gemm_iq4_nl_16x1_q8_0` | 12（"疑弱/近-generic"） | 见 E2 | 该疑点由坏探针驱动 |

**定档结论不变**（定档键 = 源归属，非 ins/rvv 计数）。

### E2. OPP-S 定档订正：`ggml_gemm_iq4_nl_16x1_q8_0` **是手调**，非"疑弱/近-generic"
源归属（prereg §3.2 自己的定档键）：`ggml/src/ggml-cpu/arch/riscv/repack.cpp` **同时定义** `ggml_gemv_iq4_nl_16x1_q8_0`(:392) **和** `ggml_gemm_iq4_nl_16x1_q8_0`(:1252) → **两者均 = 手调-riscv专化**。prereg 的"疑弱"仅源于 E1 坏探针。

### E3. prereg §1.2A 的 rvv build recipe 不可用 → 改用 A1 canonical
prereg §1.2A 第 4/5 条的 flag 串与 march 串**在板上编不过**。**实况**：(a) 系统 `as` = binutils-2.41 **拒 `zvfh`** → 必须 `source /opt/tcrv-toolchains/env.sh`（引入 binutils-2.46.1）；(b) 链接与 C++ 头搜索需 A1 记载的 CRT/runtime 定位 flag（prereg 写的那个变体不等价）。
已改用 **A1 canonical 逐字串**：见 `experiments/active/g8-stage1-clang-env/flags-finalized.md` §"rvv (clang-18·VLEN128)"（= prereg "照抄 A1" 的本意）。实际所用串亦逐字落在 `P1-backfill7-raw/rvv_dequant_p1_seal.txt` / `rvv_gevm_m1_seal.txt` 的 `# march=… cflags=… ldflags=…` 行（**可复现**）。
**仅 build recipe·判据未动**。两侧（ours / opp）编译域仍**单世界 clang-18 对称**。

### E4.（新·非 prereg 错）★ `ggml_gemv_iq4_nl_16x1_q8_0` 在 **VLEN128 上算错**
该核 `__riscv_vle8_v_i8mf2(kvalues,16)` / `vrgather_vv_i8mf2(...,16)`：`mf2`@SEW8 的 VLMAX = VLEN/8/2 → **VLEN128 = 8 < 16**。故 rvv(VLEN128) 上它只算一半 lane → ZERO-MODEL **511-512/512 mismatch**。
→ **#4 的 OPP-S = VOID-S**（prereg §3.3：布局/数值不对齐即 VOID-S·**禁用其数字**）。**k1(VLEN256) 上 VLMAX=16 → 0/512 全对 → 布局门 PASS**（互为印证：机制预言 → 实测确证）。

---

## 逐格

### #2 `dequantize_row/iq2_s@rvv` = **PASS[0.8]**（便宜档）
cold **1.7534×**（s1 0x1357）/ **1.7472×**（s2 0xACE2）·N=24 中位·bootstrap CI（10k·paired）s1 **[1.7400, 1.7545]** / s2 **[1.7430, 1.7480]**·relIQR ours 0.20%/0.23%·opp 1.13%/0.37%。
byte-exact ZERO-MODEL **0/1048576 mism · worst_ulp=0**（vs 独立 stock ggml oracle）。
对手 = as-shipped `dequantize_row_iq2_s` @`0x81260`（`libggml-base.so` md5 `d9c07980`·deployed `to_float` 真派发路）·objdump ins=271 **rvv=107** gather=0 vset=33（双法 DUAL-AGREE）。
**源归属**：仅 `ggml/src/ggml-quants.c:2471`（generic-C）；`arch/riscv/{quants.c,repack.cpp}` **零 dequantize_row 实现**。
★**成色**：对手 = **标量类**（§〇.1 源归属·clang 边缘 autovec **不改 tier**）→ **便宜档 · opp-immaturity · 禁称硬赢 · 0 verified hand-brick**。
**账** = [DEQ-AXIS] 独立子账 · test-only-not-in-denom · NOT e2e · NOT perf-covered · 不入 matmul headline。
provenance：`P1-backfill7-raw/rvv_dequant_p1_rebuild.log` · `dequant_ci_results.txt`。

### #3 `dequantize_row/nvfp4@rvv` = **PASS[0.8]**（便宜档 · ★我方核亦标量）
cold **2.1736×**（s1）/ **2.1689×**（s2）·N=24 中位·CI s1 **[2.1535, 2.1788]** / s2 **[2.1591, 2.1764]**·relIQR ours 1.87%/1.13%·opp 0.28%/0.14%。
ZERO-MODEL **0/1048576 · worst_ulp=0**。
对手 = as-shipped `dequantize_row_nvfp4` @`0x67b7a`·ins=182 **rvv=30** gather=0 vset=6（DUAL-AGREE）·源归属 `ggml-quants.c:531` generic-C → **标量类**。
★★**诚实加注（自探针）**：**我方 nvfp4 核 = ins=558 `rvv=0` gather=0 vset=0 · true_sp_spill=14 = 纯标量**（emitter 发了 vestigial `__riscv_vsetvl_e32m1` 后走标量 C 解码，被编译器消掉）。
→ 此 2.17× = **标量-我方 vs 边缘autovec-对手**，**双方皆非向量核**。**禁称硬赢**·**并登记我方 emitter 缺口：nvfp4 lowering 零向量化**。
**账** = [DEQ-AXIS] · test-only-not-in-denom。
provenance：同上 + `probe_fixed_rvv.txt`。

### #6 `gemm_tile/iq4_nl@k1` decode = **具名-X[0.8]**（★what-if · 走完 §4.4 完整环）
cold **OPP-X 0.2493**（s1）/ **0.2490**（s2）·CI **[0.2491,0.2498]** / **[0.2484,0.2494]**；
**OPP-S 0.1469 / 0.1473**·CI **[0.1464,0.1473]** / **[0.1467,0.1481]**·N=25·2-seed·relIQR ours 1.39/0.53%·oppX 0.16/0.25%·oppS 0.86/1.24%。
ZERO-MODEL **ours 0/512 · OPP-X 0/512 · OPP-S 0/512 = ALL-OK** → **正确核（输性能非正确性）**；**OPP-S 布局门 PASS**（`make_x16` == ggml `block_iq4_nlx16` 期望·功能级证明）。
**regime**：**独立 M=1 GEVM 实测**（activation = PLAIN 单 q8_0 向量 stride34·无行交织；ONE GEVM call over nc=512；K=2048）—— **未继承** prefill nr16 / vec_dot M=1 / 历史锚 0.2487。

**§4.4 完整环：**
1. **对手三证**：
   - OPP-X `ggml_vec_dot_iq4_nl_q8_0` = 9-ins VLEN 派发 thunk（rvv=0）→ callee 机判 `_vl128`/`_vl256` 两证；`_vl256` @`0xa7c78` ins=79 rvv=30 gather=4 vset=6（DUAL-AGREE）= **手调 VLEN256 专化 codebook 核**。
   - OPP-S `ggml_gemv_iq4_nl_16x1_q8_0` @`0xaaa80` ins=293 **rvv=205 gather=32** vset=51（DUAL-AGREE）·源归属 **`arch/riscv/repack.cpp:392` = 真 riscv 手写 RVV intrinsics**（vrgather/vwmul/vfwmul）→ **手调**（对照 `_generic` ins=197 rvv=35 = 另一符号·未用）。
   - stock md5 before==after `871169a0`（cpu）·`00267134`（base）→ 非稻草人·未改库。
2. **我方核自探针**：ins=724 rvv=628 **gather=64** vset=260 size=5128B · **true_sp_spill=0**（非 spill 病）。
3. ★**墙（具名·机制级）= `codebook-gather-bound`（gather 原语选错 · 非 spill 非带宽）**：
   - **我方 = 64 × `vluxei16.v`** = 走 **内存** codebook 数组的 indexed gather（front-door 设计即 "REAL MEMORY codebook GATHER"）。
   - **OPP-S = 32 × `vrgather.vv`** = 16 项 codebook **常驻向量寄存器** 的寄存器置换。
   - 16 项 codebook 平凡装入单个向量寄存器；`vrgather` 是正确原语，`vluxei16` 每元素付 L1/访存端口往返且无法外提。叠加 **vsetvli storm（我 260 vs 对手 51）** 与 **2.5× 指令量（724 vs 293）**。
4. **可修性 = 可修（emitter 成熟度缺口·入队列）**：需 **codebook-size 键控 gather 原语选择**——`≤VLMAX` 项的 tiny codebook 走寄存器 `vrgather`，大 grid 才走 `vluxei16`。
   ★与 memory `[emitter-maturity-vluxei16-widelmul]` 一致但**方向相反**：vluxei16 对 **iq1_s 大 grid 是 win**（7.4→2.3×），对 **16 项 tiny codebook 是 loss** → **判别键 = codebook 尺寸**（C3′ 能力键控优化模式库的正例：同一原语两向）。
5. **部署成色**：★**what-if**（selector = `block-dot-decline-vlen256-decode-measured-negative`·**出货走 block-dot**）→ **禁**写成 "k1 decode 部署赢/输"。
   **循环论证防线（prereg §4.2）**：registry DECLINE 由 `iq4_nl … 0.248x` 驱动；本轮**独立复测 = 0.2493/0.2490（N=25·2-seed·CI）→ 与 registry Negative 一致（未证伪）** → **无 canon 触发·selector/registry 不动**。
**账** = matmul kernel-sym · NOT e2e · NOT perf-covered · [NG-4]。**禁外推**（本格 X 不推 rvv/他 regime）。
provenance：`k1_gevm_m1_measure.log` · `k1_gevm_m1_sanity.log` · `k1_gevm_m1_seal.txt` · `gevm_k1_ci_results.txt`。

### #1 `dequantize_row/iq2_xs@rvv` = **VOID-NOISE**·未测·0 样本·不造数
**精确缺口**：prereg §5 门 = opp relIQR ≤3%；**两次尝试实测 opp relIQR = 4.97/3.96/2.10（att1）· 2.38/1.46/4.19（att2）→ 2/2 不过 → 不开测**（ours 门过：1.00–1.66%）。
**已就绪**：build/link/ZERO-MODEL(0/1048576) 全绿·对手探针三证齐（@`0x8101a` ins=173 rvv=15 gather=0 vset=5·DUAL-AGREE·源归属 `ggml-quants.c:2444` generic-C = 标量类）。
**剩余构造**：opp 侧 cold 稳定化（该格 opp 长尾 cold-outlier）——如换更长 flush/更多 rep/独占核。
★**登记张力（不自决·不推翻门）**：6 轮 ratio 中位 = 1.7761/1.7717/1.7694/1.7744/1.7771/1.7812（**极差 0.38%·异常稳定**），但 prereg 门是**合取**，事后拿 ratio 稳定去豁免 opp-IQR 条款 = **事后找补** → **维持 VOID-NOISE**。T3 该格 **维持 pending·禁填**。

### #4 `gemm_tile/iq4_nl@rvv` decode = **VOID-NOISE**·未测·0 样本·不造数
**精确缺口**：prereg §5 GEMM 门 = ours relIQR ≤5% ∧ 3 轮 ratio 中位极差 ≤2%。
att1：ours 3.85/3.71/0.79% (过)·ratio 0.2355/0.2512/0.2422 → **极差 6.67% 不过**；
att2：ours 6.11/6.91/**14.50%**（不过）·ratio 0.2584/0.2365/0.2507 → **极差 9.26% 不过** → **2/2 不过 → VOID-NOISE**。
**已就绪**：leaf front-door 现生（md5 `a911818c`）·ABI 逐参对齐 `(n,s,vx,vy,nc)` ✓·**ZERO-MODEL ours 0/512 + OPP-X 0/512**（正确核）·对手三证齐（`_vl128` @`0xc1670` ins=76 rvv=29 gather=2 vset=11·DUAL-AGREE·手调 VLEN128 专化）。
★**关键登记（噪声来源 = 我方核内生·非环境）**：**同进程交替**测得 OPP-X relIQR 恒 0.11–0.38%（机器安静），而**我方核** 0.79–14.50% → 波动是 **我方 GEVM leaf 内生**，非 co-tenant/ambient。此为**独立于 verdict 的机制线索**（可修性调查项）。
★**登记（不作结论·不填表）**：6 轮 OPP-X ratio 全落 **0.2355–0.2584**（无一近 0.8）；但**依本轮自己的门该数不可报** → 维持 VOID-NOISE。
**#4 的 OPP-S = VOID-S**（见 E4：VLEN128 上 OPP-S 数值错 511-512/512 → 禁用其数字 0.1977 等）。
**剩余构造**：稳定化我方 leaf cold（查 vsetvli storm / TLB / 首-rep 快 ~1.15ms vs 后续 1.31–1.51ms 的系统性偏移）→ 再按同 prereg 重测。T3 该格 **维持 pending·禁填**。

### #5 / #7 `gemm_tile/iq4_nl` prefill @rvv / @k1 = **VOID**·未测·0 样本·不造数
**精确缺口**：唯一 prefill harness `tools/e2e-harness/board/iq4nl_gemm_paired_driver.c` **无 ZERO-MODEL oracle** —— 它把我方权重 `Wr` 与对手权重 `Wo` **各自独立随机填充**（`fill_rand(Wr,…)` / `fill_rand(Wo,…)`，仅 fp16 scale 同设 0x2C00），两侧**根本不是同一份数据**，且**不比对任何输出**。prereg §4.0 硬性前置：**mism 门未过禁报性能数** → 该 harness **结构上无法满足正确门** → 不测、不报。
**已就绪**：GEMM leaf front-door 现生成功（`kernels_iq4nl/iq4_nl_gemm.c` md5 `2cc64970`·2683 行）·**ABI 逐参对齐**（leaf `(n,s,vx,vy,nr,nc,bs)` == driver 声明 7 参 ✓）·符号前缀 `tcrv_`→`weft_` 已知（fc72fe53）·OPP-S `ggml_gemm_iq4_nl_16x1_q8_0` 双板在库且**源归属 = 手调**（E2）。
**剩余构造**（明确·可执行）：① 单一 plain W/A 真源 → `make_x16` 打包权重 + `block_q8_0x4`（stride136·qs@8）交织 activation；② per-(r,c) oracle 重算；③ ours/OPP-X/OPP-S 三方 ZERO-MODEL 门；④ 再按 prereg §4.3（nr=16·K=2048·nc=512·N≥20·2-seed·cold）测。
**板**：rvv(#5) / k1(#7)。**预判不作结论。** T3 两格 **维持 pending·禁填**。

---

## 张力登记（canon 级 · **必问 · 本 agent 未动**）

1. **[DEQ-AXIS] auto-promote 字面张力（较 prereg 更强）**：DEQ decree 二触发键 = 「对手 rvv>0」。**订正后**三格 rvv = **15 / 107 / 30**（prereg 以为 5/33/6）→ 字面 auto-promote 更"够格"；但 §〇.1 **源归属判标量类**（generic-C 唯一定义·arch/riscv 零实现）。**两规则冲突 = canon 级 → 必问**。本轮**只登记双读数·未动 102/105 分母**。
2. **T3 `gemm_tile/iq4_nl` 行 tier 存疑（prereg §7 已登记·本轮证据加强）**：87/88 行 tier=**通用向量**，但其 opponent 符号 `ggml_vec_dot_iq4_nl_q8_0_vl{128,256}` 本轮 objdump + 源归属 = **手调 VLEN 专化**；且**同算子** OPP-S 双板亦 **手调**（E2）。→ **87/88 tier 疑应为 手调**。**登记待裁**（改 tier 触碰 roster/口径·不自决）。
3. **`ggml_gemv_iq4_nl_16x1_q8_0` 在 VLEN128 数值错**（E4）：非我方 bug，是 as-shipped 上游核的 VLEN 前提（mf2 需 VLEN≥256）。是否值得上游报/是否影响 rvv 板 repack 派发结论 = **登记待裁**。

## 卫生 checklist
- [x] stock/clang18 `.so` 只读·md5 before==after 双证（rvv `e85fceda`/`d9c07980` · k1 `871169a0`/`00267134`）
- [x] 单实例·测前 kill·`STRAY=0` 全程·`taskset` 单核钉（rvv 用 8/9/12·**未碰 core 0,1** co-tenant；k1 0/1）
- [x] load-gate idle≥70%（实测 98–100%）·gov=performance 入 log
- [x] **无 git add/commit**·主树/build/governor 未改·T3 未动
- [x] 输出零 gcc 字样
- [x] 硬冻结未碰（102/105·101/108·9/83·roster $meta·队序）
- [x] raw 全量落 `P1-backfill7-raw/`
- [x] 未写 `/home/kingdom/phdworks/papers/`
