# G7 货架A — T9 §6 争议解: q4_K@k1 our-emit vs TRUE hand-brick repack (kernel-axis micro A/B)

> **性质**：**第二赛道 kernel-axis** 对称 micro A/B（T9 kernel-sym 台账 §1.1/§6 派生·成色核实）。
> **与 perf-covered 系统账（9/83）永不混算**·**非 e2e beat·非 sealed 8-gate Win**·NG-4 纪律。
> **板**：k1 / SpacemiT-X60 / VLEN256 (vlenb=32) / core3 (L2 512KiB shared 0-3) / clang-18.1.8 出货对称域 / gov=performance 1.6GHz。
> **测于**：2026-07-13 · loadavg 2.49→2.91（k1 shared board·within-proc paired ratio 抵消共同 contention·IQR 全 <0.3% 除 1 cold seed 1.71%）。
> **约束遵守**：无 git · 无 schema/T8/ROADMAP 改动 · 无 rvv · board 纯 micro scratch（`/tmp/q4k_hb_resolve/`）· 主树/build/governor/stock-lib(只读) 全未动 = **零 restore 需求**。

---

## 0. 争议（T9 §6·必读）

- T9 §1.1 记 **q4_K@k1 kernel-sym 3.106× ≥parity**·对手标为 **"真出货 hand-brick（case256 stock repack `ggml_gemm_q4_K_16x1_q8_K`）·唯一 hand-brick ≥parity·成色最硬"**。
- L2 hot/cold agent 发现：**sealed 3.106× 实测对手 = `ggml_vec_dot_q4_K_q8_K` block-dot**（非 hand-brick），真 repack `ggml_gemm_q4_K_16x1_q8_K` **存在但未作对手** ⇒ "赢 hand-brick" 主张【未验证】。
- 本 cell = **另测 our-emit q4_K vs 真 `ggml_gemm_q4_K_16x1_q8_K`**（真出货 hand-brick）以立/证伪。

---

## 1. 对手身份 objdump 证（block-dot vs repack·哪个真出货·哪个 sealed 对比的）

### 1.1 sealed 3.106× 的实际对手 = block-dot（**非** hand-brick）— 三重证据
1. **t4a 台账 MANIFEST 自述**（`kquant-k1-vlen256-kernel-axis-t4a/MANIFEST.md` line 7,10-12）：`Opponent = ggml_vec_dot_q{4,5}_K_q8_K`（block-dot）·并声称 "ggml ships **no** K-quant repack trait ... block-dot IS the factory K-quant mul_mat fallback"。
2. **t4a paired driver 源**（`tools/e2e-harness/board/kquant_gemm_paired_driver.c:49,119`）：`OPP_CALL` = `ggml_vec_dot_q4_K_q8_K(...)`（逐 (r,c) block-dot），**不是** 16x1 gemm。
3. **t4a runner**（`run_k1_kquant_t4a.sh:24`）preflight 只 grep `ggml_vec_dot_q[45]_K_q8_K`。
⇒ **sealed 3.106× = ours / block-dot·确证**。

### 1.2 真 hand-brick repack `ggml_gemm_q4_K_16x1_q8_K` = k1 真出货默认·可测·healthy
- **符号存在**：`nm -D /data/k1build-stock/bin/libggml-cpu.so` → `00000000000abe58 T ggml_gemm_q4_K_16x1_q8_K`（+ `_generic` 0x350ba）。三 repack 变体 16x1 / 8x4 / 8x8·`tensor_traits<block_q4_K,1,16>` vtable 全在。
- **真出货默认**：k1 stock `ggml_repack_get_optimal_repack_type` case256 → q4_K（ne[1]%16==0）→ 16x1 repack（同 g7-l3 已证 q4_0 case256 ON 路由命中·tinyllama dims %16==0）。T9 §5 亦记 q4_K@k1 e2e winner = stock 16x1 repack。⇒ 是**真部署路径**·非 latent。
- **VLEN256-native healthy**（lib objdump 0xabe58–0xaca00）：vset SEW/LMUL = `e16,m1 ×12 · e32,m2 ×11 · e8,mf2 ×5 · e8,m2 ×4 · e16,m4 ×2`（宽 LMUL m2/m4·非 half-width mirage）· vsetvli 34 · vwmacc 80 · spill 14 · ~913 insns（**紧凑 rolled**）。
- **VLEN 键控**：16x1 RVV 变体 hardcode vl=16 on f32m2/u8m2 → **VLEN128(rvv) clamp-to-8 破损**（t4b-m4-decisive: 4081/4096 bad）·但 **VLEN256(k1) vl=16 = f32m2 满宽·正确**（本测 correctness 证）。故此 hand-brick 是 **k1/VLEN256 硬件专调**·rvv 上无对应（那里 `_generic` 才是正确 realization）。
- **对称编译确证**：`/data/k1build-stock/compile_commands.json` → `arch/riscv/repack.cpp` 由 **`/usr/bin/clang++-18 -O3 -march=rv64gcv_zfh_zvfh_zicbop_zihintpause -mabi=lp64d`** 编。⇒ ours(clang-18) vs stock(clang-18) = **对称**·无 [CASE-COMPILER-ASYMMETRY]。

---

## 2. micro A/B：our-emit vs 真 16x1 hand-brick（同 vx/vy·同 shape·对称 clang-18）

**方法**：BOTH kernel 吃**同一** repacked 权重（block_q4_Kx16·2304B/col-group）+ **同一** q8 激活（blk_q8_Kx4·1168B/row-group）——即 t4b-m4-decisive 已确立的 byte-layout drop-in。ours `(n,s,vx,vy,nr,nc,bs)` vs `ggml_gemm_q4_K_16x1_q8_K(n,s,bs,vx,vy,nr,nc)`。shape K=2048 nr=64 nc=512（== sealed t4a）。同总 MAC = nr·nc·K → GMAC/s 直接可比。ours 编 **两 recipe**：O3+对手 exact march / O2+canonical march。N=12·3 seed·HOT(single tile)+COLD(P=8 pool 4.5MiB≫L2)。

### 2.1 correctness cross-check（证两侧算同一 GEMM·A/B 公平·非空转）
| seed | max_abs | max_rel | nbad/32768 | 判定 |
|---|---|---|---|---|
| 0xC0FFEE | 2.9e-3 | 1.0e-2 | 0 | ours == 16x1 bounded-ULP |
| 0xBEEF01 | 3.9e-3 | 1.3e-1(row23 near-0 cancel) | 1 | 同上（单元素近零抵消） |
| 0x51A7ED | 2.9e-3 | 7.1e-3 | 0 | 同上 |
⇒ ours 与真 16x1 在 VLEN256 **byte-layout drop-in·输出 bounded-ULP 等价**（FP reassociation noise·同 m4-decisive）·且**全 nr=64 行匹配**（对手被正确调用·非部分行 bug·非早退）。**A/B 量的是真·全工作量**。

### 2.2 timing（ours / 16x1 hand-brick·6 run 全一致）
| recipe | HOT ratio | COLD ratio | ours GMAC/s | 16x1 GMAC/s |
|---|---|---|---|---|
| ours-O3 (对手 exact march) | 0.6216–0.6217 | 0.6215–0.6221 | ~3.72 | ~5.98 |
| ours-O2 (canonical) | 0.6215–0.6223 | 0.6217–0.6231 | ~3.72 | ~5.99 |

**聚合：ours / 真-16x1 = 0.622×（HOT∧COLD∧O2∧O3 全一致·IQR <0.3% 除 1 cold seed 1.71%）**
⇒ **真 hand-brick 比 ours 快 1.61×（1/0.622）**。

### 2.3 三方对照（同板同 shape·锚到 block-dot）
| kernel | GMAC/s | vs block-dot | vs ours |
|---|---|---|---|
| `ggml_vec_dot_q4_K_q8_K` block-dot（sealed 对手） | 1.146（t4a gate5） | 1.00× | 0.31× |
| **ours** repack-GEMM（S6 unrolled·textB 25362·vwmacc 2240·spill 4） | 3.72 | **3.25×**（≈sealed 3.106×） | 1.00× |
| **真 `ggml_gemm_q4_K_16x1_q8_K` hand-brick**（rolled·vwmacc 80·spill 14） | **5.98** | **5.22×** | **1.61×** |

**机制**：ours = 全展开 25KB 大 blob（VLEN-invariant vtype·不激进吃 VLEN256 宽度）；hand-brick = 紧凑 rolled + hardcode vl=16 满宽 f32m2/u8m2（VLEN256 专调）。→ 展开体量赢≠perf 赢（同 [CASE-KQUANT-GCC-CODEGEN] / G6-B re-roll 教训·[GAP-P1]）。VLEN256 上 hand-brick 的窄码宽调度打败我方宽码全展开。

---

## 3. 裁决（T9 §6 争议解决）

**"赢 hand-brick" 主张 → 证伪（FALSIFIED）。**

- our-emit q4_K@k1 = **0.622× < parity** vs 真出货 hand-brick `ggml_gemm_q4_K_16x1_q8_K`（对称 clang-18·VLEN256-native·真部署路径）。**真 repack 更强（1.61×）**。
- sealed 3.106× 是 **vs block-dot** 的（三重证据·§1.1）·**非** vs hand-brick。⇒ q4_K@k1 kernel-sym **成色 = vs-block-dot（弱—中对手）·非 hand-brick**。
- **hand-brick 对手可测**（非"未接测不了"分支）——测了·我方 kernel **输**。
- **0 verified hand-brick 确认**（match T9 §6 口径校正修正预测）。

### T9 §6 成色分布最终建议（留主会话改 T9·本 agent 不动 T9）
kernel-sym 9 格成色分布应改为：
- **0 verified hand-brick**（q4_K@k1 = **vs-block-dot**·实测输真 hand-brick 0.62×）
- **2 better-vec block-dot**（q5_0/q5_1@rvv·成色中）
- **7 block-dot/light**（q4_K@k1 · q5_K@k1 · q4_0@k1 · q8_0@k1 · q4_0/q4_1/q8_0@rvv·弱—轻对手）
- 计数 9 = 覆盖面（第二赛道 C3′ 证据）**不变**；仅**成色标签**从 "1 hand-brick+2+6" 改 "**0 hand-brick+2+7**"。
- 建议 §1.1 q4_K@k1 行 "唯一 hand-brick ≥parity·成色最硬" **删/改**为 "vs-block-dot 3.106×（对真 16x1 hand-brick = 0.62× LOSS·本 cell 证）"·并可新增 §1.2/或 §2 一行记 **q4_K@k1 vs-真-16x1-hand-brick = 0.622× <parity（对称 clang-18·kernel-axis）**。

---

## 4. 污染 / restore
- **board**：仅写 scratch `/tmp/q4k_hb_resolve/`（ephemeral·driver+.o+bins+run.log）。**主树 / build / stock ggml .so（只读链接）/ governor 全未改** = **零 restore**（纯 micro·符合任务）。stock lib md5 未动（只读 nm/objdump/-l 链接）。
- **本机 casefile**：`experiments/active/g7-l2-kernelsym-hotcold/q4K-handbrick-resolve/`（driver 源 + runner + raw/run.log + summary CSV + 本 evidence + MANIFEST）。**无 git add/commit**（用户提交）。
- **ours md5**：s6_q4K.c `90d454da`（== t4a 缓存·no regen·板上 md5 双证）。ours .o libcall-free（zfh/zvfh native fp16·0 `__extendhfsf2`）。
