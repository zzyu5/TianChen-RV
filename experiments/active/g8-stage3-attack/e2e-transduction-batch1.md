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
  与 decode wash（memory-bound·q4_0@k1 0.857× sealed / q5@rvv 0.8165×·见 §4-5 订正归属）对立·**判别键 = 对手 bound-type 由 arithmetic intensity(M) 定**。
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

### 状态：✅ 已完成 — q5_0 fresh clean 实测已得(§3.0·1.966×/2.213×=frozen 复现) + q5_1 deployed==proven(§3.1·byte-equivalence·fresh 冗余裁停)

**deployed==proven（铁证·三重）**：
1. q5_build_seal 重建 ON .so md5 = **`df88afa3d1a029fbf42dd4df316443eb`** = **sealed WEFT_Q50 byte-identical**
   （q5k1-e2e §5 板卫生 WEFT_Q50=df88afa3）→ 部署核 = HEAD 发射核·逐字节相同。
2. gevm .inc = HEAD-regen REDESIGN-B md5 8cd0c697 = A5 proven（vlm_v_b16/vmnand 指纹）。
3. correctness ON vs OFF greedy = **TOKEN-IDENTICAL**（ctrl-stripped）·banner "TCRV G7-L3 EMITTED
   GEVM/GEMM ... ENGAGED" 在 ON binary（strings 坐实·5 条 ENGAGED 串）。
⟹ sealed 1.97/2.07× decode + 2.21/2.34× prefill = **HEAD-live-valid**（产出该数的核 = HEAD 逐字节核·非"引用旧数"）。

**fresh clean 测量：★并行 agent 碰撞·本役 stand-down（q5 让给并行线）**：fresh A/B 撞
**orphaned `q5_measure.sh q5_0`**（PPID=1·非本役脚本·= 疑似并行 agent 的 q5 复验线·同时在写本文件 §4）。
双线互相争 board（loadavg 7-10）→ 双方 q5 数据互污（我方 c4-7 隔离尝试仅捕到 ROUND1 OFF[pp3.69/tg3.24
= 与 sealed stock 一致] 即被本役主动 kill·避继续争 board）。**本役对 q5 fresh 测量 stand-down**（并行线拥有
q5 复验·避重复劳动+board 争用）·q5 结论**全靠 deployed==proven 铁证**（上 3 重·ON=df88afa3=sealed
byte-identical → sealed 1.97/2.07× = HEAD-live-valid）。**未 kill 并行 orphan**（避毁其工作·其已自然结束）。
partial 污染数方向性确认 engage（ON pp>OFF pp·我方核确跑）。

### ★3.0 HEAD-live fresh clean A/B —— 已得（并行 measurement-agent 补·承上文 stand-down 交接·2026-07-16）

> 承上文"q5 让给并行线"：本节 = 该并行线（measurement-agent）**kill 全竞争进程 + 陈旧 ctrl 循环 → 验树 baseline +
> scratch .so 完好（df88afa3/14b6add6·零 stray）→ clean board 单实例重跑**的 fresh clean 结果。banner clean-run
> fires=**9**（engage 坐实）。capture durable = `q5k1-e2e/raw/q50_headlive_clean.txt`。**两 agent 结论一致互证**（deployed==proven 铁证 + 本 fresh clean 数）。

**q5_0 HEAD-live phase-split**（PP=128 TG=32 REPS=6 PASSES=2·taskset 0-3 -t4·gov perf 1.6GHz 锁·interleaved paired）：

| 相 | HEAD-live weft/stock | abs t/s (stock→weft) | frozen(Jul-15) | 判定 |
|---|--:|--:|--:|:--|
| **decode tg32 (GEVM M=1)** | **1.966×**（pass1 clean·n=6/side） | 3.280→6.449 | 1.970× | **TRANSDUCE-GREEN 确认（Δ0.2%）** |
| prefill pp128 (GEMM) | **2.213×**（pass1 clean） | 3.840→8.500 | 2.212× | **≥parity WIN 确认（Δ0.05%）** |

- **weft 两 pass 岩石稳**：decode 6.449/6.411（Δ0.6%）·prefill 8.500/8.497（Δ0.04%·stddev 0.006）→ weft clean。
  stock pass1 clean（decode 3.280 stddev 0.003·prefill 3.840 stddev 0.0007）·**stock pass2 co-tenant 末段抖动**
  （run 末 loadavg 3.24→9.69·stock decode 塌 3.097 stddev 0.099·**弃 pass2 stock**）→ **权威 = pass1 clean n=6/side**。
- **[DISCRIMINATOR] 判读（compute-bound → 传导）**：q5_0 stock block-dot decode = **compute-bound**（本役 objdump OFF .so
  `ggml_vec_dot_q5_0_q8_0` = 80 insn·per-block 水平归约 `vwredsum`×2·一次 32-elem 块·`vlm.v` 5th-bit·k1/clang-18
  弱 autovec 5th-bit qh → **stock decode 3.28 t/s ≪ 内存墙 ~13 t/s**=compute 瓶颈）→ 我方 VLEN256 strip-`vwmacc`
  repack-GEVM 解算力瓶颈 → **1.97× 真传导**（非 wash）。**对照 §4 q4_0@k1 同板同编译器 decode 0.857× WASH = 判别键单变量隔离坐实**。
- **成色（诚实）**：净新 path 赢·**赢一半来自 stock q5 decode 路本就弱**（compute-bound·clang-18 弱 5th-bit autovec）=
  **beat-weak-baseline**·**非 beat-hand-tuned**·对手是真出货 block-dot（非稻草人）故赢合法。
- **q5_1**：deployed==proven（fresh 冗余·未测·见 §3.1）·GEVM `.inc` md5 `4ad42d91`=HEAD casefile·A5 proven。

#### 3.1 q5_1 HEAD-live — deployed==proven（fresh 测量 = 冗余·主会话裁停·未测·铁证经字节等价）

**主会话裁决（2026-07-16 course-correct）**：q5_1 fresh phase-split = **belt-and-suspenders 冗余·不启新测**·
q5_1 结论**全靠 deployed==proven 铁证**（下）·省板时不拖队列。

**deployed==proven（HEAD-live 双证·byte-equivalence）**：
1. **q5_1 build ON .so md5 = `43569a46d0c2cb68aae74bf6cadb7a66`** = **sealed WEFT_Q51 逐字节相同**
   （q5k1-e2e §5 板卫生 WEFT_Q51=43569a46·HEAD-live `q5_build_seal.sh q5_1` 从共享树 baseline 确定性复现·
   OFF=`14b6add6`·2 weft_emitc q5_1 syms·树 RESTORE OK 零 stray）。
2. **GEVM `.inc` md5 = `4ad42d91`** = 本地 HEAD casefile `q5_1_gevm_frontdoor_vlen256.c` 逐字节相等 = A5 proven REDESIGN-B。
⟹ 产出 sealed **decode 2.073× / prefill 2.344×** 的核 = HEAD 所 emit 的逐字节核 → **frozen 2.07/2.34× = HEAD-live-valid**
（非"引用旧数"·= byte-identical 部署核在 HEAD 的确定性产出）·[DISCRIMINATOR] 判读同 q5_0（compute-bound stock → 传导）。
- **成色注**：同 q5_0 = beat-weak-baseline（stock 5th-bit compute-bound）·非 beat-hand-tuned·合法赢。
- **诚实边界**：q5_1 无【本役 fresh e2e run】·仅 byte-equivalence（比 q5_0 弱一档证据·q5_0 有 fresh clean 实测）·
  但 byte-identical .so 保证产出数值同 = frozen 复现在 HEAD 有效。

---

## 4. 便宜档 washout 对照（memory-bound wash·证 [DISCRIMINATOR] 反面判别键）

**选格 = q4_0@k1/VLEN256（sealed·同板同编译器 as q5 transduce）** — 这是**最干净的判别键隔离**：
与 §3 q5@k1 transduce **同板（k1/X60/VLEN256）· 同编译器（clang-18 对称）· 同核族（repack-GEVM）·
同相（decode M=1）**，唯一变量 = **格式** → 决定对手 bound-type → 决定传导/wash。

### 4.1 主对照（sealed·frozen 引用 + objdump bound-type 佐证·非重测省板时）

**`experiments/sealed/repack/k1-vlen256-q4_0-flip`（SEALED F5·model tinyllama-q4_0.gguf·freq 1.6GHz span 0%）**：
| 相 | ours/stock | CI95 / n | 判定 |
|---|--:|--:|:--|
| prefill pp | **1.0043×** | [1.0018,1.0056]·n=10 | PARITY（zero-hypothesis·repack 布局不减字节·decode-format prefill 亦近内存墙） |
| **decode tg** | **0.8574×** | [0.8492,0.8653]·n=10 | **DIFFERENCE·regression −16.6%（WASH·过 2× 地板 + CI 排除 1.0）** |

- **对手 bound-type = memory-bound**（[DISCRIMINATOR] 反面输入）：stock q4_0 block-dot 在 k1/clang-18 上
  **decode 6.62 t/s**（≈ q5_0 stock 3.29 t/s 的 **2.0×**·因 4-bit nibble 无 5th-bit qh 计算惩罚·SIMD 高效）→
  decode M=1 瓶颈 = **权重流带宽（memory）非算力**。我方 repack-GEVM 的价值 = **算力重构**（消 block-dot
  的 per-block 水平归约·换 strip `vwmacc`）·但对已 memory-bound 的 q4_0 **无算力可省·且 repack 布局在
  M=1 反增搬运开销** → **0.857× 净退化**（非仅 wash·实为 regression·= repack-decode 内存税坐实）。
- **operational 判据坐实**：**同核族（repack-GEVM）· 同板 · 同编译器**下·q4_0 decode = 0.857×（退）
  vs q5 decode = 1.97/2.07×（赢·§3）→ **唯一差 = 格式 → 对手 bound-type**（q4_0 stock 高效 4-bit=memory-bound；
  q5 stock 5th-bit qh 重算=compute-bound）·**这是判别键的最强隔离证据**（单一变量对照）。

### 4.2 同格式跨板 wash（q5@rvv·证 bound-type 依【板×编译器】而非格式名）

**★关键 nuance（诚实·加固判别键）**：**同格式 q5** 在 **rvv 上 wash·k1 上 transduce**——
证 bound-type **不是格式属性**·而是 **对手基线在该板该编译器下的 bound**：
| 格·板 | e2e decode ours/stock | 对手 bound（证据） | 出货编译器 | 传导/wash | 出处 |
|---|--:|---|:--:|:--:|---|
| q5_0@**rvv**/VLEN128 (OLD) | **0.838×** | memory-bound（gcc-15.2 autovec 5th-bit 高效→e2e decode 内存墙） | gcc-15.2 | **WASH** | `archive/g5/g5-wiring/M2-q5_0/evidence.md:79` |
| q5_0@**rvv**/VLEN128 (REDESIGN-B) | **0.929×** | 同上（削重建缩小差·仍<parity） | gcc-15.2 | **WASH** | `archive/g7/g7-l2-gevm-redesign/qh-plane-e2e-smoke/evidence.md:53` |
| q5_1@**rvv**/VLEN128 | **0.7836×** | memory-bound（−21.6% GEVM wash·n=20） | gcc-15.2 | **WASH** | `archive/g5/g5-wiring/M2-q5_1/evidence.md:73` |
| **q5_0/q5_1@k1**/VLEN256 | **1.97×/2.07×**（§3） | **compute-bound**（clang-18 弱 5th-bit·3.29/3.09 t/s≪墙） | clang-18 | **传导** | 本役 §3 / q5k1-e2e |

⟹ **同 q5 格式·rvv(gcc-15.2)=memory-bound→wash·k1(clang-18)=compute-bound→transduce**。判别键 =
**对手基线 bound-type**（由 板×出货编译器 对 5th-bit qh 的 autovec 质量定）·**非格式名**。这排除
"格式决定传导" 的替代解释 → 判别键 = opponent-bound-type **CONFIRMED**。

### 4.3 corroborating memory-bound wash（canon 铁律·frozen）

- **IME 5.51× kernel → 0.86× decode**（canon [[kernel-wins-dont-transplant-to-e2e]]）= compute-bound micro
  win 不传导 memory-bound decode·同判别键反面。
- **q4_K@k1 decode isolation 0.995×**（本役 §1·两侧同 stock gemv·M=1 memory-bound·无算力赢可测）= 控制项（非 wash·isolation）。

### 4.4 ★attribution 订正（诚实·batch1 §5 表原引数纠错）

batch1 §5 表原将 wash 极值记为 **"0.8165×（G5-M2/q5-evid）· 归 q4_0"** — **该 0.8165× 实为
q5_0@rvv OLD**（`M2-q5_0/evidence.md:79`）·**非 q4_0**。**干净的 q4_0 decode wash = 0.857×**
（`k1-vlen256-q4_0-flip` sealed·同板同编译器）。§5 表已订正（见下）。旧 0.8165× 仍是有效的
**q5_0@rvv memory-bound wash**（4.2 表首行）·仅归属从 q4_0 改回 q5_0@rvv·数不撤·归属正名。

---

## 5. [DISCRIMINATOR-OPPONENT-BOUND-TYPE] 逐格映射（汇总·数字随测填）

| 格·轴 | kernel-axis | e2e ON/OFF | 对手 bound-type（证据） | 传导/wash | HEAD-live |
|---|--:|--:|---|:--:|:--:|
| **q4_K@k1 prefill** (GEMM vs hand-brick) | 1.187× cold | **1.101×** ✓ | compute-bound（objdump 80 vwmacc/38 load·M=128 复用·实测传导互证） | **传导** | ✓ md5 9e057adb |
| q4_K@k1 decode (isolation·未换 gevm) | — | 0.995× | memory-bound（M=1·两侧同 stock gemv·无算力赢可测） | isolation | — |
| **q5_0@k1 decode** (GEVM vs block-dot) | 1.76× cold | **1.966×** ✓ HEAD-live | compute-bound（objdump 80 insn·per-block vwredsum×2·3.28 t/s≪墙·clang-18 弱 5th-bit） | **传导** | ✓ .so df88afa3=sealed byte-id |
| q5_1@k1 decode | 2.24× cold | **2.073×**(deployed==proven·fresh 未测·裁停冗余) | compute-bound（同上·5th-bit qh） | **传导** | ✓ .so 43569a46=sealed byte-id |
| q5_0@k1 prefill (GEMM) | — | **2.213×** ✓ HEAD-live | compute-bound（GEMM M=128 复用） | **传导** | ✓ df88afa3 |
| q2_K@k1 prefill | 1.364× cold | **blocked-layout** | compute-bound（预期·同 q4_K） | 未测 | ✓ 核 HEAD-live·但 nbad=8114 非 drop-in |
| **q4_0@k1 decode**（同板同编译器 as q5·**wash 反面**） | ~parity kernel | **0.857×** sealed | **memory-bound**（stock 4-bit 高效·6.62 t/s=2× q5 stock·repack 增内存税） | **WASH** | sealed k1-vlen256-q4_0-flip |
| q5_0@rvv decode（**跨板同格式**·OLD/RB） | 1.44-1.67× vs OLD | 0.838×/0.929× | **memory-bound**（rvv gcc-15.2 autovec 5th-bit 高效→内存墙） | **WASH** | frozen G5-M2/G7-L2 |
| q5_1@rvv decode | 1.23× kernel | **0.7836×** | memory-bound（−21.6% GEVM wash·n=20） | **WASH** | frozen G5-M2 |
| [canon 铁律] IME kernel→decode | 5.51× kernel | **0.86×** | memory-bound（compute win 不传导） | **WASH** | canon kernel-wins-dont-transplant |

**判别键坐实（★三点隔离·加固）**：
1. **同板同编译器·同核族·同相·唯一差=格式**：q4_0@k1 decode 0.857×（WASH·对手 memory-bound）vs
   q5@k1 decode 1.97×（传导·对手 compute-bound）= **单变量隔离**·判别键 = **对手 bound-type 非格式名**。
2. **同格式跨板**：q5@rvv decode wash（gcc-15.2 autovec 5th-bit 高效→memory-bound）vs q5@k1 decode 传导
   （clang-18 弱 5th-bit→compute-bound）= bound-type 依 **板×出货编译器** 非格式。
3. **相轴**：q4_K@k1 prefill 传导（对手 compute-bound·M=128 复用）+ q4_K@k1 decode isolation 0.995×（M=1 memory-bound·无算力赢可测）。
⟹ **kernel 算力赢传导 IFF 对手 compute-bound·washes IFF memory-bound**·判别键 = 对手基线 bound-type
（由 相(M)×板×出货编译器 对该 kernel 计算路的 autovec 质量共同决定）·**[DISCRIMINATOR-OPPONENT-BOUND-TYPE] CONFIRMED**。

**★attribution 订正落地**（承 §4.4）：原 wash 极值 "0.8165×·归 q4_0" = 误归属·实为 **q5_0@rvv OLD**（M2-q5_0:79）·
已改回上表；**干净 q4_0 decode wash = 0.857×**（sealed·同板同编译器）为主对照。

## 6. perf-covered 演化建议（any-board 规则·收口令〇.1·硬冻结）

**★维持 9/83 不变**（收口令〇.1 终裁·any-board 只升成色·永不增头条格数·已 RESOLVED）。
- q4_K@k1 已是 9 绿格之一（既有 frozen e2e prefill 1.085× Win-K1-VLEN）。本役 fresh e2e prefill
  hand-brick beat（若传导）= **成色 upgrade**（frozen 1.085× → HEAD-live 真 beat ggml 手调 GEMM）·
  **不改 9/83 计数**·记成色注·**留主会审**（禁擅动 canon/头条口径·禁 git）。
- q5@k1 decode HEAD-live 复验【已确认】：q5_0 fresh clean 实测 1.966×（=frozen 1.970×·§3.0）· q5_1
  deployed==proven 2.073×（byte-equivalence·§3.1）= 既有绿格的 HEAD-live 再确证·**不改计数**。
- **无新增格·无扩分母·9/83 硬冻结**（成色 upgrade 已登记·留主会话·禁擅动 canon/头条口径/perf-covered 计数）。

## 板卫生
- 本役板操作全 board-local scratch（`/data/g8q4k` + `/data/g7q5{0,1}`）·树用完 restore 到 baseline
  （md5 双证）·stock .so 只读·governor 只读·core0-3 pin·co-tenant loadavg 记录·**禁 git**（改动/实录留主会审）。
- q5 gevm .inc：OLD G7-L3 已备份 `.G7L3.bak`·换 HEAD-live REDESIGN-B（md5 8cd0c697/4ad42d91=A5 proven）。
