# G8 阶段三 · §五 全表对手重解析 (opponent re-parse · symbol-level machine-judgment)  2026-07-14

> **性质**：0.8 硬门重测**地基**。逐格逐板重跑对手探针·符号级机判·clang-18 对称域·回填 T3。
> **对手纪律 (q4_K 稻草人永久前车)**：baseline = 该板 as-shipped **真派发** kernel·符号级机判·**禁**手写/挑弱/`_generic` 替换·上游更强未启用披露。
> **口径**：主表对称域 = clang-18.1.8 (双板一版一表·[CASE-COMPILER-ASYMMETRY] 判别键)·gcc = 部署附注·IME 格 carve-out。
> **本役不跑 cold A/B**（那是 §六）——纯符号级重解析。不改 emitter/lib·e2e 冻结·禁 git（主会审后 commit）。

## 0. 环境 · 对手树 · commit pin (真派发认定)

| 键 | rvv (VLEN128) | k1 (VLEN256) |
|---|---|---|
| 对称编译器 | clang 18.1.8 (`/opt/tcrv-toolchains`·canonical·env.sh) | clang 18.1.8 (Bianbu `/usr/bin/clang`) |
| march | `rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause` `-fno-integrated-as`(binutils-2.46.1) | `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs` |
| **对手部署树** | `/home/ubuntu/vericurve-rv-lab/llama.cpp`（SpacemiT 手调"本意最优形态"） | **`/home/bianbu/tcrv-k1-llama`**（★指定 vericurve 路径 k1 不存在·genuine upstream ggml RVV VLEN-dispatch refactor）+ stock `libggml-cpu-1x16.so`(repack hand-brick) |
| **ggml commit pin** | `e36a602ba38a26206c749ba4fb5dcf481bfd92db`（2026-06-15·clean） | ⚠**不可 pin**（树无自有 .git·walk-up→错误 repo 20039c9e·**FLAG §六**）·as-built = clang-18/-O3 build-off |
| 对手 TU 重编 | quants.c→quants_opp.o · ggml-quants.c→ggmlquants_opp.o · ggml-cpu/quants.c(generic)→quants_generic_opp.o · vec.cpp→vec_opp.o · ops.cpp→ops_opp.o（全 clang-18 重编 OK） | quants.c→quants_opp.o · ggml-cpu/quants.c(generic·nvfp4)→quants_generic_opp.o · ggml-quants.c→ggmlquants_opp.o · stock .so 直反汇编（forward/repack GEMM）· **k1 亦需 -fno-integrated-as**（纠正 stage-1） |
| pin / hygiene | core8-15 · co-tenant vLLM(core0,1) 未扰 · loadavg 2.27→2.33 | core0-3 · co-tenant(10 users) 未扰 · loadavg 2.27→2.15 |

**★真派发认定方法学（反 q4_K 稻草人）**：K-quant/iq 格的 top-level `ggml_vec_dot_<fmt>` = 微型 dispatcher（读 `csrr vlenb` 分支），**真跑的是 VLEN 专化 `_vlNNN`**。rvv@VLEN128 → dispatch `_vl128`；k1@VLEN256 → dispatch `_vl256`。**逐格反汇编该 `_vlNNN` 专化体**（非 `_generic` 弱参考）。FLAT 格无 VLEN 专化=单符号。

**★objdump 方法学坑（记档·供 §六复用）**：llvm-objdump-18 `--disassemble-symbols=<sym>` 对 `-fno-integrated-as` 本地符号**截断**（只出 prologue）；且 GNU-as 发 `.LBB`/`.Lpcrel` 中间标签。正解 = 全 `.text` dump + 地址区段切片（终止符 `<[^.]` 排除 `.L` 标签）+ `awk -F"\t"` 取 mnemonic。metric: `tot`=指令数 `rvv`=向量op(非vset) `mac`=vw?mul/vw?macc/vredsum等 `vset`=vsetvli `gather`=vrgather/vlux/vlox `csrr_vlenb`=VLEN分支。

## 1. rvv (VLEN128) 逐格对手身份表 (clang-18 对称·raw = `rvv/objdump_metrics_rvv.txt`)

### 1.A matmul kernel-sym (0.8 分母内)

| 格 | 真派发 opponent_symbol | metric (tot/rvv/mac) | opponent_grade | 备注 |
|---|---|---|---|---|
| **FLAT vec_dot** | | | | |
| q4_0 | `ggml_vec_dot_q4_0_q8_0` | 125/12/3 | **light-vec**（弱） | ggml block-dot 单实现折中 |
| q4_1 | `ggml_vec_dot_q4_1_q8_1` | 76/10/3 | **light-vec**（弱） | 同上 |
| q5_0 | `ggml_vec_dot_q5_0_q8_0` | 112/22/4·csrr1 | **native-vec**（中·better-vec） | qh 5-bit·VLEN-adaptive |
| q5_1 | `ggml_vec_dot_q5_1_q8_1` | 120/20/4·csrr1 | **native-vec**（中·better-vec） | qh 5-bit |
| q8_0 | `ggml_vec_dot_q8_0_q8_0` | 66/6/2 | **light-vec**（弱） | 上游 VLEN128 破损·light |
| **K-quant vec_dot** | | | | |
| q2_K | `ggml_vec_dot_q2_K_q8_K_vl128` | 235/125/66 | **hand-tuned**（强） | SpacemiT vl128 full-unroll mac66 |
| q3_K | `ggml_vec_dot_q3_K_q8_K_vl128` | 238/102/36 | **hand-tuned**（强） | SpacemiT vl128 |
| q4_K | `ggml_vec_dot_q4_K_q8_K_vl128` | 220/105/35 | **hand-tuned**（强） | SpacemiT vl128 |
| q5_K | `ggml_vec_dot_q5_K_q8_K` | 238/82/26·vset15 | **native-vec**（中） | **NO vl-spec·唯一未手调格** |
| q6_K | `ggml_vec_dot_q6_K_q8_K_vl128` | 232/84/36 | **hand-tuned**（强） | SpacemiT vl128 |
| **K-quant GEMM (cross-op·opp=block-dot·rvv 零 K-quant repack GEMM)** | | | | |
| q2_K/q3_K/q4_K/q6_K gemm | `ggml_vec_dot_<f>_q8_K_vl128` | (同上 vec_dot·cross-op) | **hand-tuned CROSSOP**（强） | our-gemm vs opp-vl128-blockdot·明标 cross-op |
| q5_K gemm | `ggml_vec_dot_q5_K_q8_K` | (同上) | **native-vec CROSSOP**（中） | 同上·q5_K 无 vl-spec |
| **iq/fp4 vec_dot** | | | | |
| iq1_s | `ggml_vec_dot_iq1_s_q8_K_vl128` | 225/68/19·gather5·csrr6 | **hand-tuned**（强·gather） | SpacemiT vl128 grid-gather |
| iq1_m | `ggml_vec_dot_iq1_m_q8_K_vl128` | 253/62/26·gather2·csrr8 | **hand-tuned**（强） | SpacemiT vl128 |
| iq4_nl | `ggml_vec_dot_iq4_nl_q8_0_vl128` | 102/18/4·gather2 | **native-vec**（中·hand-vl128 但轻） | codebook gather·较 iq1 轻 |
| nvfp4 | `ggml_vec_dot_nvfp4_q8_0`（generic） | 214/23/2·NO-riscv-spec | **native-vec-generic**（弱） | ★对手存在=generic·**非 absent**·无 riscv 专化 |

### 1.B forward-op (0.8 分母内)

| 算子 | opponent_symbol | metric | opponent_grade | 备注 |
|---|---|---|---|---|
| softmax | `ggml_compute_forward_soft_max` | 592/19/4·csrr1 | native-vec | expf reduce·max-sub |
| rms_norm | `ggml_compute_forward_rms_norm` | 277/10/2·csrr3 | native-vec | 2-pass·VLEN-adaptive |
| rope | `ggml_compute_forward_rope_flt<f32>` | 1184/32/12 | native-vec | autovec + scalar sin/cos |
| silu | `ggml_vec_silu_f32` | 112/38/7 | native-vec | sigmoid/expf 向量化 |
| **gelu** | `ggml_table_gelu_f16`(BSS-LUT) | **rvv=0·table-lookup** | **scalar-f16-LUT** | ★GGML_GELU_FP16·**结构差·非公平速度 A/B**（我方 tanhf 赢精度） |
| add | `ggml_compute_forward_add`（vfadd.vv inline） | autovec | native-vec | IEEE f32 bit-exact |
| mul | `ggml_compute_forward_mul`（binary-op） | autovec | native-vec | IEEE f32 bit-exact |
| scale | `ggml_compute_forward_scale` | 154/7/2 | native-vec-light | vfmul.vf in-place |
| cpy | `ggml_compute_forward_dup/cpy` | 4339/51·csrr12 | native-vec-streaming | 向量 copy/memcpy 路 |

### 1.C DEQ-AXIS dequant (照测·**不进头条 0.8 分母**·[DEQ-AXIS])

18 格全测·grade 见 raw。要点：q6_K rvv432（重）· q3_K rvv132 · q2_K rvv88 · **q4_K rvv0=SCALAR**（clang-18 未向量化·deploy gcc-15 autovec 附注）· iq2_xxs/iq2_s/iq3_xxs rvv74 · iq3_s rvv146 · iq4_nl/iq4_xs/mxfp4 rvv13-14+gather2(light-gather) · **nvfp4 rvv24=native-vec-light（dequant 存在·≠ vec_dot generic 弱）** · tq1_0 rvv135(重)。**★compiler_axis 附注**：DEQ 部署 to_float 路 = gcc-15.2 per-format autovec（T-CENSUS 实测）·本表 clang-18 重编 grade 为主判·gcc 为部署附注（per-format 背离见 T-CENSUS 3d）。

### 1.D 排除格 (0.8 分母外·披露)

| 格 | 排除理由 | 符号实况 |
|---|---|---|
| **q1_0 vec_dot / dequant** | **internal-A/B**（enum 41·Weft-internal binary·非标准 ggml）·我方自有格·**无 genuine 第三方 same-op 对手** | `ggml_vec_dot_q1_0_q8_0_vl128` 存在（111/25/4·native-vec）但格本身内部·`_generic` 亦 Weft-added → 禁自己当自己靶 |
| **iq/tq GEMM 7**（iq2_xxs/iq2_xs/iq2_s/iq4_xs/mxfp4/tq1_0/tq2_0 @gemm） | **same-op opponent-absent**（零 iq/tq repack GEMM·仅 block-dot vec_dot）·cross-op 可选·非 matmul kernel-sym 19 | 对手退化 our-gemm vs opp-block-dot |
| **product_reduce 3** | 无框架 kernel 对手（internal sub-primitive） | vs-scalar 是 sanity 层非 framework |
| **IME 格 (q4_0/q8_0/q4_K @ime)** | **gcc-13 carve-out**（`xsmtvdotii1p0` clang-18 拒·[CASE-COMPILER-ASYMMETRY]）·不参与 clang-18 主表 | 真硅 cert 域·flags-finalized D2 |

## 2. rvv 与旧台账逐格比对 (一致确认 / 存疑 superseded)

**★结论：rvv 高度一致·无稻草人**——clang-18 对称域重解析出的真派发对手 = 部署 vericurve `_vlNNN` 手调符号（与 T-CENSUS rvv-half 机判**同符号同成色**）。clang-18 域**没有削弱对手**（K-quant/iq 仍 STRONG hand-tuned·非 `_generic`）→ 反-straw-man 纪律坐实。

| 格 | 旧台账 | 新 clang-18-sym | 判 |
|---|---|---|---|
| FLAT q4_0/q4_1/q8_0 | light-vec 弱(7-10 rvv) | light-vec(6-12 rvv) | ✓ 一致 |
| FLAT q5_0/q5_1 | better-vec 较强(24-26 rvv) | native-vec 中(20-22 rvv) | ✓ 一致 |
| K-quant q2_K/q3_K/q4_K/q6_K vec_dot | `_vl128` hand-tuned STRONG | hand-tuned STRONG(rvv84-125) | ✓ 一致 |
| K-quant q5_K vec_dot | inline MODERATE·唯一未手调 | native-vec MODERATE·NO vl-spec | ✓ 一致 |
| iq1_s/iq1_m vec_dot | `_vl128` STRONG | hand-tuned STRONG | ✓ 一致 |
| **iq4_nl vec_dot** | `_vl128` **STRONG** | native-vec **中**(rvv18·hand-vl128 但轻) | ⚠ **收窄**：旧"STRONG"过评·实=手写 vl128 但 codebook 轻·仍真对手·非反转 |
| **nvfp4 vec_dot** | T9 §3 **absent**(no-fair-opp) | **EXISTS**=generic native-vec 弱(rvv23·无 riscv 专化) | ⚠ **旧 T9 §3 "absent" superseded**（印证 T-CENSUS §0.3 纠正·genuine upstream 对手·enum 40） |
| forward 9(softmax/rms_norm/rope/silu/gelu/add/mul/scale/cpy) | bclass-forward rvv-half 机判 | 逐符号同 grade | ✓ 全一致（gelu=f16-LUT 结构·rms_norm=2-pass·add/mul=autovec） |
| q1_0 | no-fair-opp Weft-internal | EXISTS(vl128 native-vec) 但 enum41 内部→**排除** | ✓ 一致(排除)·机制细化 |

**旧 opponent_map 处置**：T9/T-CENSUS 的 rvv opponent 机判**逐格 confirmed**（同符号同成色）→ 沿用有效；唯 (a) iq4_nl "STRONG"→native-vec-中 收窄、(b) nvfp4 vec_dot "absent"→generic-弱 superseded 两处覆盖。

### 2.k1 k1 与旧台账比对 + 板异

| 项 | 旧台账 | 新 clang-18-sym | 判 |
|---|---|---|---|
| K-quant vec_dot _vl256(q2/q3/q4/q6) | hand-tuned STRONG | hand-tuned STRONG(rvv54-103) | ✓ 一致 |
| q5_K vec_dot | 无 vl256 spec·唯一未手调 | native-vec-heavy·NO vl-spec | ✓ 一致 |
| iq1_s/iq1_m | RVV-gather/vl256 强 | hand-tuned 强 | ✓ 一致 |
| iq4_nl | `_vl256` RVV 强 | native-vec 中(thin wrapper) | ⚠ 收窄(同 rvv) |
| **FLAT q5_0/q5_1 vec_dot** | vecdot-k1 "80/32·90/30 better-vec/csrr" | **light-vec(74/12·79/11)** | ⚠ **板异+旧计数存疑**：新 symmetric recompile 更轻·旧数或 diff-build/count·**flag §六**(非载重·parity-by-adoption 部署路) |
| **★q4_K@k1 "唯一 hand-brick"** | memory "q4_K 唯一真 hand-brick" | **q2_K/q4_K/q5_K/q6_K 均有 repack hand-brick·仅 q3_K 无** | ⚠ **纠正**：q4_K 唯一"16x1-decode-deployed"·但非唯一有 repack·§3.D 清算 |
| **★K-quant GEMM 对手轴** | rvv-型 cross-op 惯性 | k1 **4/5 real same-op repack hand-brick**（≠ rvv 全 cross-op） | ⚠ **★大板异**：k1 GEMM 对手远强于 rvv·0.8 过门 k1 更难 |
| forward 9 | bclass-forward k1-half | 逐符号：gelu=LUT · add/mul=autovec · rope/scale/cpy 偏标量 | ✓ 大体一致（k1 rope/scale/cpy 较 rvv 更标量） |

**★双板对手成色本质差**：rvv 对手树(vericurve) 与 k1 对手树(tcrv-k1-llama·upstream) 的 **K-quant/iq vec_dot `_vlNNN` 手调层同强**；**分歧在 GEMM 轴**——k1 stock .so 带真 repack hand-brick(q2/q4/q5/q6·8x8/8x4/16x1)、rvv 零 K-quant repack(GEMM 全 cross-op vs block-dot)。→ **对手身份逐板独立·禁跨板沿用 GEMM 对手成色**。

---

## 3. k1 (VLEN256) 逐格对手身份表 · q4_K 双核清算 (raw = `k1/objdump_metrics_k1.txt`)

> **★对手树纠正**：指定 `/home/bianbu/vericurve-rv-lab/llama.cpp` 在 k1 **不存在**（仅 census 子目录）→ 子代理找到并用 **`/home/bianbu/tcrv-k1-llama`**（genuine upstream ggml riscv quants.c·2025 RVV VLEN-dispatch refactor·源无 tcrv 标记）。**★ggml commit 不可 pin**（树无自有 .git·walk-up 命中错误 repo 20039c9e·**FLAG §六**）。stock repack = `libggml-cpu-1x16.so`。**★k1 亦需 `-fno-integrated-as`**（stage-1 flags "k1 不需" 纠正·标准-RVV vsetivli inline-asm）。dispatch 经 objdump -dr relocs 确认 vlenb=32→`_vl256`。

### 3.A matmul kernel-sym @k1 (0.8 分母内)

| 格 | 真派发 opponent_symbol | metric (tot/rvv/mac) | grade | 板异 vs rvv |
|---|---|---|---|---|
| q4_0/q4_1/q8_0 vec_dot | `ggml_vec_dot_<f>` | 70/12 · 73/10 · 63/6 | **light-vec** 弱 | 同 |
| q5_0/q5_1 vec_dot | `ggml_vec_dot_q5_{0,1}_q8_{0,1}` | 74/12 · 79/11 | **light-vec** 弱 | ⚠**板异**：rvv q5_0/q5_1=native-vec-中(rvv20-22)·k1 更轻 |
| q2_K/q3_K/q4_K/q6_K vec_dot | `..._vl256` | 167/85 · 264/96 · 211/54 · 189/103 | **hand-tuned** 强 | 同(vl256 手调) |
| q5_K vec_dot | `ggml_vec_dot_q5_K_q8_K` | 238/82/26·vset15 | **native-vec-heavy**（NO vl-spec·唯一未手调·boundary·xtheadvector 编出） | 同 |
| **q2_K/q4_K/q5_K/q6_K GEMM** | **`ggml_gemm/gemv_<f>_{16x1,8x8,8x4}_q8_K`** | q2_K 1338/361 · q4_K gemv 468/209·gemm 767/224 · q5_K 2426/709 · q6_K 526/190 | **hand-tuned-repack 强·★REAL same-op** | ⚠**★大板异**：k1 有真 repack hand-brick·rvv 全 cross-op(零 K-quant repack) |
| q3_K GEMM | `ggml_vec_dot_q3_K_q8_K_vl256`（cross-op） | (block-dot rvv96) | **hand-tuned CROSSOP**（q3_K 唯一无 repack） | q3_K 唯一 cross-op·余 4 real |
| iq1_s/iq1_m vec_dot | `..._vl256` | 175/56 · 242/54 | **hand-tuned** 强 | 同 |
| iq4_nl vec_dot | `..._vl256` | 101/24 | **native-vec** 中 | 同 |
| nvfp4 vec_dot | `ggml_vec_dot_nvfp4_q8_0`（generic） | 217/23/2 | **native-vec-generic** 弱 | 同（generic 存在·hand-tuned 缺·**非 absent**·子代理"absent"=仅查 arch-TU 已纠正） |

### 3.B forward-op @k1 (0.8 分母内)
softmax `ggml_vec_soft_max_f32`(leaf 81/39 native-vec·wrapper scalar-setup) · rms_norm(522/20 native-vec-light) · rope(1275/30 **mostly-scalar/light-vec**) · silu(116/38 native-vec) · **gelu**(`ggml_table_gelu_f16` BSS·**scalar-f16-LUT**·结构差) · add(2306/65 autovec) · mul(2318/68 autovec) · scale(166/7 **light-vec/near-scalar**) · **cpy**(`dup_flt<f32>` 454/**rvv2**·**scalar/memcpy-stream**)。板异 vs rvv：k1 的 scale/cpy 更接近标量·rope mostly-scalar（rvv rope=autovec rvv32）。

### 3.C DEQ-AXIS @k1 (照测·不进 0.8 分母)
18 格全测·全 rvv>0（无纯标量）。q6_K rvv108/tq1_0 rvv107（重·hand-tuned）· **q4_K rvv16=native-vec-light**（★板异：rvv q4_K dequant=scalar rvv0）· nvfp4 dequant rvv24 存在（≠ vec_dot generic）。详 raw。

### 3.D ★q4_K 双核事故制度化清算 (钉死真派发者·q4_K 稻草人永久前车)

三层对手·**双核分立**（判别键 = 我方核形态 × 对手 repack 开关）：

| 我方核 | 真派发对手 | 对手 metric | races 谁 |
|---|---|---|---|
| **kernel-sym vl=8 核**（full-unroll·s6_q4K.c md5 90d454da） | `ggml_vec_dot_q4_K_q8_K_vl256`（block-dot·repack-OFF fallback） | 189/54 | **LIGHT block-dot**（弱—中·手调 vl256 但轻） |
| **sealed/e2e vl=16 核**（Win-K1-VLEN·s6_q4K_vl16 md5 e437fd3b） | **decode** `ggml_gemv_q4_K_16x1_q8_K` / **prefill** `ggml_gemm_q4_K_16x1_q8_K`（repack-ON·真出货部署） | gemv 468/**209** · gemm 767/**224** | **HEAVY 16x1 hand-brick**（~4× block-dot·强） |

**★清算结论**：q4_K@k1 **两个数字 races 两个不同档对手**——(1) kernel-sym 3.106×"赢"的是 **block-dot**（vl256 rvv54·非 hand-brick·b29c269c 三重证印证）；(2) sealed Win-K1-VLEN 1.20× 赢的是 **真 16x1 hand-brick**（gemv rvv209 decode）。**判别键 = 核形态（vl=8 full-unroll spill vs vl=16 满宽 fit）·非 board/VLEN**。★**新增精度**（超既有 memory）：decode 真派发 = **`ggml_gemv_q4_K_16x1`（gemv·非 gemm）**·prefill = `ggml_gemm_q4_K_16x1`——decode/prefill 分立 gemv/gemm 两符号。★**"q4_K@k1 唯一 hand-brick" 纠正**：stock .so 对 q2_K/q4_K/q5_K/q6_K **均有** repack hand-brick（q4_K 唯一 16x1-decode-deployed·但 q2_K/q5_K/q6_K 有 8x8/8x4/16x1）·仅 **q3_K 无 repack**。

---

## 4. ★0.8 硬门分母定档 (双板成对·DEFINITIVE)

**分母公式**：0.8 硬门分母 = **可测格 (matmul kernel-sym + forward-op) − 排除格**。DEQ-AXIS(18/板) 照测不进头条分母。IME carve-out(3) 排除。q1_0 internal-A/B(1) 排除。iq/tq GEMM(7) opponent-absent same-op·cross-op-optional·不在 same-op matmul kernel-sym 19。product_reduce(3) 无框架对手排除。

### ★★双板成对定档 = **28 / 28**

| 桶 | rvv (VLEN128) | k1 (VLEN256) |
|---|---:|---:|
| matmul kernel-sym | **19** | **19** |
| ├ FLAT vec_dot | 5 | 5 |
| ├ K-quant vec_dot | 5 | 5 |
| ├ K-quant GEMM | 5（**全 cross-op** vs block-dot·rvv 零 K-quant repack） | 5（**4 real same-op hand-brick** q2/q4/q5/q6 + q3_K cross-op） |
| └ iq/fp4 vec_dot | 4（iq1_s/iq1_m/iq4_nl/nvfp4） | 4 |
| forward-op | **9** | **9** |
| **0.8 硬门分母** | **28** | **28** |

### ★对手成色分布 (机判·升门判据)

| 成色 | rvv/28 | k1/28 | 明细 |
|---|---:|---:|---|
| **hand-tuned 强**（真向量·真 beat 空间） | **10** | **11** | rvv: K-quant vec_dot 4[q2/q3/q4/q6] + K-quant GEMM cross-op 4 + iq1_s/iq1_m ｜ k1: 同 4+2 + **K-quant GEMM real hand-brick 4**[q2/q4/q5/q6] + q3_K GEMM cross-op(vs 手调 block-dot) |
| **native-vec 中** | **5** | **2** | rvv: q5_0/q5_1 + q5_K vec_dot + q5_K GEMM + iq4_nl ｜ k1: q5_K vec_dot + iq4_nl（k1 q5_0/q5_1 塌 light-vec·GEMM 归强） |
| **light/generic 弱** | **4** | **6** | rvv: q4_0/q4_1/q8_0 + nvfp4-generic ｜ k1: q4_0/q4_1/q5_0/q5_1/q8_0 + nvfp4-generic |
| **forward native-vec** | **8** | **5** | rvv: softmax/rms_norm/rope/silu/add/mul/scale/cpy ｜ k1: softmax/rms_norm/silu/add/mul（rope/scale/cpy 塌 light/scalar） |
| **scalar/结构差**（非公平速度·flag） | **1** | **4** | rvv: gelu(f16-LUT) ｜ k1: gelu(f16-LUT) + rope(mostly-scalar) + scale(light) + cpy(scalar-stream) |

### ★升门规则 (对手成色机判 → 0.8 有效性)
- **真向量实现者**（hand-tuned / native-vec·rvv 23/28 · k1 18/28）→ 该格 0.8 **全额有效**（对手真向量·beat = 真硬碰硬空间）。**★k1 matmul GEMM 轴对手更强**（4 real repack hand-brick·rvv 全 cross-op block-dot）→ k1 GEMM 过 0.8 **更难**·per-board 分记。
- **scalar / 结构差对手**（gelu LUT 双板 · k1 rope/scale/cpy）→ 留分母但 **flag**：beat 是结构差（LUT vs tanhf / 标量 vs 向量）·**非公平速度 A/B**·§六 报 win 须标"结构非速度"。
- **弱 light/generic 对手**（q4_0/q4_1/q8_0/nvfp4）→ 留分母·但 win 成色低（light-vec block-dot·parity-by-adoption tautological / generic autovec）。

### ★排除格披露 (不在 28·逐条)
| 排除桶 | 数/板 | 理由 |
|---|---:|---|
| q1_0 vec_dot | 1 | internal-A/B·enum 41 Weft-internal·无 genuine 第三方 same-op 对手（rvv `_vl128` 存在但格本身内部·k1 同） |
| IME (q4_0/q8_0/q4_K @ime) | 3 | gcc-13 carve-out（xsmtvdotii1p0 clang-18 拒·[CASE-COMPILER-ASYMMETRY]） |
| iq/tq GEMM (iq2_xxs/iq2_xs/iq2_s/iq4_xs/mxfp4/tq1_0/tq2_0) | 7 | same-op opponent-absent（零 iq/tq repack GEMM）·cross-op-optional·非 same-op 19 |
| product_reduce | 3 | 无框架 kernel 对手（internal sub-primitive） |

---

## 5. 体例合规自检
- 对手 = 该板 as-shipped **真派发** kernel·符号级机判（§0 方法学·逐 `_vlNNN` 反汇编·非 `_generic`）✓
- clang-18 对称域（board 内 ours==opp 同编译器/march/flags）·gcc 部署附注·IME carve-out ✓
- q4_K 稻草人前车：rvv 无 hand-brick GEMM（K-quant GEMM = cross-op vs vl128 blockdot）·k1 双核清算见 §3.D ✓
- 多候选实现披露：nvfp4(generic 弱·非 absent)·iq4_nl(vl128 中·旧 STRONG 收窄)·q5_K(无 vl-spec) ✓
- 内部新增路径排除：q1_0 enum41 internal-A/B 出 0.8 分母 ✓
- 禁 git·不改 emitter/lib·e2e 冻结·不跑 cold A/B(§六)·禁另建账本(回填 T3 + 本 casefile evidence) ✓
