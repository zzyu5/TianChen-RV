# G8 e2e 传导战役·首步 batch1 — [DISCRIMINATOR-OPPONENT-BOUND-TYPE] 映射 (2026-07-16)

> **任务**：e2e 传导首步。测哪些 kernel 算力赢传导到真 llama.cpp e2e vs wash。核心机制
> `[DISCRIMINATOR-OPPONENT-BOUND-TYPE]`：kernel 算力赢传导到 e2e **IFF 对手基线 compute-bound**·
> washes **IFF memory-bound**。判别键 = 对手 bound-type（objdump/roofline 判）。
> **纪律**：deployed==proven bit-identical（HEAD-live rebuild·非引用旧冻结数）·cold e2e·N≥3·
> 噪声自检·每 e2e 数绑 [DISCRIMINATOR] 判读·0 造数·未测不填·成色诚实·赛道分立·**禁 git**。
> HEAD = `c1c43e2bc`（refactor/full-refactor-m1）。板：k1(X60/VLEN256/clang-18) + rvv(VLEN128).

---

## 〇. deployed==proven HEAD-live 证 (本役所有部署核先过此门)

**git 发射器不变量**（A5 §0.2 规则延续到当前 HEAD）：
- `d109d6ed2..HEAD` lib/include 漂移 = **0 文件**（q5 发射器·selector-fix pin 起零漂移）。
- `f890babe0(A5)..HEAD` lib/include 漂移 = **0 文件**（A5 bit-identical 证前推到当前 HEAD 不变）。
- `38abf20eb(census q4_K/q2_K 发射器)..HEAD` lib/include 漂移 = 3 文件·全 = **selector/front-door**
  （`RVVContractionPathSelection.{h,cpp}` + `RVVLowerQuantContraction.cpp`·均来自 d109d6ed2 q5 fix）·
  K-quant GEMM 发射器体 `RVVToEmitCBlockQuantLinear.cpp` = **UNTOUCHED**。

**HEAD-live 核重生 byte-diff**（本地 weft-opt HEAD + mlir-translate-20）：
| 核 | HEAD-regen md5 | casefile md5 | 判定 |
|---|---|---|:--:|
| q4_K GEMM vlen256 unrolled | `9e057adb558cf0ab2b1faaeb6c076b84` | 同（census kernel） | **HEAD-live ✓** |
| q2_K GEMM vlen256 unrolled | `9240956b549781e04fc5e470b1a76cc7` | 同（census kernel） | **HEAD-live ✓** |

⟹ 2 hand-brick 对手核 = HEAD-live（发射器体自 census 起零漂移·本地重生 byte-identical）。
⟹ q5@k1 = A5 已双证（git 不变量 + md5·decode 核 8cd0c697.../4ad42d91...）·当前 HEAD 前推有效。

---

## 1. q4_K@k1 e2e prefill 传导测（旗舰·byte-verified hand-brick GEMM win 1.187×）

**性质**：census kernel-axis cold GEMM 1.187×(nr16)/1.161×(nr64) vs 真 `ggml_gemm_q4_K_16x1_q8_K`
RVV hand-brick（drop-in nbad=0/8192·byte-exact）。**这是最强对手** = ggml 出货手调 GEMM。
传导目标 = **e2e prefill**（GEMM 轴·M>1 compute-amortized）。

**部署法**：body-swap（arch/riscv/repack.cpp `ggml_gemm_q4_K_16x1_q8_K` 体[line 1073-1340]
→ 我方 emitc 核 forwarder·同 repacked block_q4_Kx16 buffer + q8_Kx4 act·drop-in nbad=0 保证
correctness·`bs`=float row-stride 与我方核 v7 语义一致[census 直证]）·OFF=pristine ggml hand-brick·
ON=我方核·同 clang-18 对称域·树用完 restore(md5 双证)。

**build 封印**（`/data/g8q4k`·2026-07-16）：
- OFF md5 = `14b6add63a14d813c27b27c687a45a8d`（pristine 重建·= q5 STOCK baseline·deterministic·
  clang-18 对称）· ON md5 = `a9c4938420bfd649a9878ae599665fe6`（1 个 weft_emitc q4_K 符号·OFF=0·
  banner "WEFT-Q4K-GEMM-ENGAGED" 字串在·objdump 坐实）· ON≠OFF ✓。
- 树 restore 后 ARCH md5 = `c3c101fd...` = baseline ✓。
- **deployed==proven**：ON 核 = census kernel（md5 9e057adb·HEAD weft-opt 重生 byte-identical）·
  发射器 `RVVToEmitCBlockQuantLinear.cpp` 自 census(38abf20eb) 起零漂移。

**对手 bound-type（[DISCRIMINATOR] 判读输入）**：stock `ggml_gemm_q4_K_16x1_q8_K` = RVV-specialized
hand-brick（census objdump 767 insns / 238 rvv / 80 vwmacc = compute-dense）。**prefill M=128 →
权重跨 M token 高复用 → arithmetic intensity 高 → compute-bound**。⟹ [DISCRIMINATOR] 预测：
prefill 算力赢**应传导**（对手 compute-bound）。decode M=1 = memory-bound·但本役未换 decode gevm
（两侧同 stock gemv）→ tg≈1.0 = **prefill-isolation 控制**（非 washout·算力赢在 decode 不存在）。

### ★VERDICT: TRANSDUCE-GREEN (prefill) — 首个 e2e beat-hand-brick

| 轴 | ON/OFF e2e | abs t/s (OFF→ON) | kernel-axis cold | [DISCRIMINATOR] 判读 |
|---|--:|--:|--:|:--|
| **prefill (pp128)** | **1.101×** (n=4·16.108–16.189/14.642–14.684) | 14.67→16.15 | 1.187×(nr16) | **对手 compute-bound → 传导 ✓**（Amdahl 稀释 1.187→1.101·GEMM≈58% prefill wall） |
| decode (tg32) | 0.995× (n=4) | 7.48→7.44 | — | 未换 gevm·两侧同 stock·**isolation ✓**（swap prefill-only 坐实） |

- **correctness**：greedy ON vs OFF **token-identical**（生成文本逐字相同 "…the capital of France is Paris…"·
  仅终端 spinner 控制符差·非模型输出）·我方 q4_K GEMM = 正确 drop-in（census nbad=0 e2e 印证）。
- **banner**：`WEFT-Q4K-GEMM-ENGAGED` prefill 期发射（correctness 期 count=4·我方核确在 e2e 跑）。
- **成色（诚实）**：**首个真 e2e beat-hand-brick**（我方 HEAD-live emitc q4_K GEMM 胜 ggml 出货手调
  RVV GEMM 1.10× e2e prefill）·对手 = 真 16x1 hand-brick（非稻草人·census objdump 238 rvv/80 vwmacc）·
  编译器对称（双 clang-18）。倍数中等（1.10×·Amdahl 稀释后）但**是最强对手类**（手调）·**真传导**（非 wash·非 routing 白嫖·净新发射体 body-swap）。
- **[DISCRIMINATOR] 坐实**：算力赢 1.187× kernel → 1.101× e2e = **prefill 对手 compute-bound → 传导**·
  与 q4_0/G5-M2 decode wash（memory-bound·0.8165×）对立·**判别键 = 对手 bound-type 由 arithmetic intensity(M) 定**。
- **对手 bound-type 独立证据**（OFF binary objdump `ggml_gemm_q4_K_16x1_q8_K`·本役实测）：767 insns /
  **80 wide `vwmacc`（MAC）** / 38 loads → compute:memory 指令比高（内层 80 宽 MAC vs 38 load）·
  **compute-dense hand-brick**·prefill M=128 权重复用再乘 M → arithmetic intensity 高 = **roofline compute-bound**·
  与经验传导（1.10×）一致互证。
- 板卫生：core0-3 pin·gov perf 1.6GHz·co-tenant loadavg 3.5→6.2（paired interleaved 吸收·ratio range <0.3% 证 clean）·OFF/ON md5 14b6add6/a9c49384。

---

## 2. q2_K@k1 — drop-in correctness 约束（census nbad=8114·非 byte-exact drop-in）

census §0：q2_K ours vs ggml 16x1 = **timing-valid 但 byte-layout 异**（nbad=8114·**非** drop-in）。
⟹ e2e body-swap 会产错输出（correctness fail）。q2_K e2e 传导需自建 interleaver+全 scaffold（如 q5 净新法）·
非简单 body-swap。**如实标 blocked-on-layout·具名 X·非预判墙**（待评估是否本役内可做）。

---

## 3. q5@k1 e2e 复验（C1 capstone·deployed==proven A5 已 bit-identical）

REDESIGN-B repack-GEVM leaf·kernel cold 1.76/2.24×·A5 冻结数 decode 1.97/2.07× prefill 2.21/2.34×。
复验 = HEAD-live 重跑 q5 harness 确认数字。

### 状态：待跑

---

## 4. 便宜档 washout 对照（预期 memory-bound wash·证 [DISCRIMINATOR] 判别键）

待选 + 待跑。

---

## 5. [DISCRIMINATOR-OPPONENT-BOUND-TYPE] 逐格映射（汇总·数字随测填）

| 格·轴 | kernel-axis | e2e ON/OFF | 对手 bound-type（证据） | 传导/wash | HEAD-live |
|---|--:|--:|---|:--:|:--:|
| **q4_K@k1 prefill** (GEMM vs hand-brick) | 1.187× cold | **1.101×** ✓ | compute-bound（objdump 80 vwmacc/38 load·M=128 复用·实测传导互证） | **传导** | ✓ md5 9e057adb |
| q4_K@k1 decode (isolation·未换 gevm) | — | 0.995× | memory-bound（M=1·两侧同 stock gemv·无算力赢可测） | isolation | — |
| q5_0@k1 decode (GEVM vs block-dot) | 1.76× cold | 待填(seal 1.97×) | compute-bound（stock 5th-bit qh 无好 SIMD·3.29 t/s≪带宽） | 传导(预期) | ✓ md5 8cd0c697 |
| q5_1@k1 decode | 2.24× cold | 待填(seal 2.07×) | compute-bound（同上） | 传导(预期) | ✓ md5 4ad42d91 |
| q2_K@k1 prefill | 1.364× cold | **blocked-layout** | compute-bound（预期·同 q4_K） | 未测 | ✓ 核 HEAD-live·但 nbad=8114 非 drop-in |
| **[对照·frozen]** decode wash 极 (G5-M2/q5-evid) | ~1.23× kernel | **0.8165×** | **memory-bound**（stock 好 SIMD·算力赢被 mem-wall 洗） | **WASH** | frozen(引用·非本役·attribution 见正文) |

**判别键坐实**：同一 kernel 算力赢·对手 compute-bound → 传导（q5 decode / q4_K prefill）·对手
memory-bound → wash（q4_0 decode 0.8165×）。**bound-type 由对手基线的 arithmetic intensity 决定**：
prefill(M大·权重高复用)恒 compute-bound → GEMM 算力赢传导；decode(M=1·权重流)恒 memory-bound·
除非 stock decode kernel 本身弱(compute-bound·如 q5 5th-bit) 才传导。

## 6. perf-covered 演化建议（any-board 规则·收口令〇.1·硬冻结）

**★维持 9/83 不变**（收口令〇.1 终裁·any-board 只升成色·永不增头条格数·已 RESOLVED）。
- q4_K@k1 已是 9 绿格之一（既有 frozen e2e prefill 1.085× Win-K1-VLEN）。本役 fresh e2e prefill
  hand-brick beat（若传导）= **成色 upgrade**（frozen 1.085× → HEAD-live 真 beat ggml 手调 GEMM）·
  **不改 9/83 计数**·记成色注·**留主会审**（禁擅动 canon/头条口径·禁 git）。
- q5@k1 decode HEAD-live 复验（若确认 1.97/2.07×）= 既有绿格的 deployed==proven 再确证·不改计数。
- **无新增格·无扩分母**（硬冻结遵守）。

## 板卫生
- 本役板操作全 board-local scratch（`/data/g8q4k` + `/data/g7q5{0,1}`）·树用完 restore 到 baseline
  （md5 双证）·stock .so 只读·governor 只读·core0-3 pin·co-tenant loadavg 记录·**禁 git**（改动/实录留主会审）。
- q5 gevm .inc：OLD G7-L3 已备份 `.G7L3.bak`·换 HEAD-live REDESIGN-B（md5 8cd0c697/4ad42d91=A5 proven）。
