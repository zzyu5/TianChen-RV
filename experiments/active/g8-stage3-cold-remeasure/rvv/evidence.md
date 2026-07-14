# G8 阶段三 §六 · rvv 板 28 个 0.8-分母格 cold A/B 重测 (clang-18 对称域)

> **性质**：§五 对手重解析（`aab4168c`·符号级）之上的**真 cold A/B 重测**·回填 T3_A perf 列·0.8 硬门判读。
> **口径**：**clang-18 对称域**（ours 与对手同 clang-18.1.8 + 同 rich-march + `-fno-integrated-as`/binutils-2.46.1）·gcc-15.2 = 部署附注（见 §诚实纠偏）。
> **对手**：§五 定档真派发核（vericurve as-shipped·`_vlNNN` 手调·非 `_generic` 弱参考·非稻草人）·recompiled clang-18。
> **Board**：rvv openEuler VLEN128 L3=64MiB·core8 pin·co-tenant vLLM(core0,1) 未扰·load-gate 100% idle。
> **纪律遵守**：禁 git（主会审后 commit）· 不改 emitter/lib（纯测量·复用 seal kernel + weft-opt 只读再生 gemm）· e2e 冻结。
> Build seal = `build_seal.txt`. Summary = `summary_0p8_rvv.csv`. Raw = `raw/{vecdot,iqfp4,fwd,gemm}_run.log`.

---

## 0. ★★0.8 达标表 (rvv 28 格 · headline · cold primary · byte-exact 全过)

| 桶 | 格 | 对手真派发核 | 对手成色 | hot | **cold primary** | M8/nr8 | ratio_regime | **0.8门** |
|---|---|---|---|--:|--:|--:|---|:--:|
| **FLAT vecdot** | q4_0 | q4_0_q8_0 | light-vec 弱 | 0.937 | **0.938** | 0.937 | M1-GEVM | **PASS** |
| | q4_1 | q4_1_q8_1 | light-vec 弱 | 1.017 | **1.027** | 1.022 | M1 | **PASS** |
| | q5_0 | q5_0_q8_0 | native-vec 中 | 0.442 | **0.451** | 0.447 | M1 | FAIL |
| | q5_1 | q5_1_q8_1 | native-vec 中 | 0.432 | **0.446** | 0.443 | M1 | FAIL |
| | q8_0 | q8_0_q8_0 | light-vec 弱 | 1.074 | **0.965** | 0.917 | M1 | **PASS** |
| **Kquant vecdot** | q2_K | q2_K_q8_K_vl128 | hand-tuned 强 | 0.547 | **0.342** | 0.499 | M1 | FAIL |
| | q3_K | q3_K_q8_K_vl128 | hand-tuned 强 | 0.518 | **0.258** | 0.456 | M1 | FAIL |
| | q4_K | q4_K_q8_K_vl128 | hand-tuned 强 | 0.445 | **0.189** | 0.362 | M1 | FAIL |
| | q5_K | q5_K_q8_K | native-vec 中 | 0.753 | **0.843** | 0.798 | M1 | **PASS** |
| | q6_K | q6_K_q8_K_vl128 | hand-tuned 强 | 0.481 | **0.180** | 0.377 | M1 | FAIL |
| **Kquant GEMM** (cross-op) | q2_K | q2_K_q8_K_vl128 blockdot | hand-tuned 强 | 1.590 | **1.112** | 0.824 | prefill-nr16 (nr64:1.38) | **PASS** |
| | q3_K | q3_K_q8_K_vl128 blockdot | hand-tuned 强 | 0.166 | **0.161** | 0.158 | prefill-nr16 (flat) | FAIL |
| | q4_K | q4_K_q8_K_vl128 blockdot | hand-tuned 强 | 1.892 | **1.114** | 0.673 | prefill-nr16 (nr64:1.57) | **PASS** |
| | q5_K | q5_K_q8_K blockdot | native-vec 弱(qh) | 1.172 | **1.067** | 0.985 | prefill-nr16 (nr64:1.26) | **PASS** |
| | q6_K | q6_K_q8_K_vl128 blockdot | hand-tuned 强 | 0.154 | **0.173** | 0.148 | prefill-nr16 (flat) | FAIL |
| **iq/fp4 vecdot** | iq1_s | iq1_s_q8_K_vl128 | hand-tuned gather 强 | 0.454 | **0.365** | 0.438 | M1 (⚠noisy o_iqr16%) | FAIL |
| | iq1_m | iq1_m_q8_K_vl128 | hand-tuned 强 | 0.158 | **0.152** | 0.160 | M1 | FAIL |
| | iq4_nl | iq4_nl_q8_0_vl128 | native-vec 中 | 1.140 | **1.150** | 1.154 | M1 | **PASS** |
| | nvfp4 | nvfp4_q8_0 (generic) | native-vec generic 弱 | 0.661 | **0.565** | 0.662 | M1 | FAIL |
| **forward-op** | softmax | ggml_compute_forward_soft_max | native-vec | 1.007 | **1.007** | shape-inv | n4096 | **PASS** |
| | rms_norm | ggml_compute_forward_rms_norm | native-vec (2-pass) | 1.150 | **1.240** | shape-inv | n4096 | **PASS** (WIN) |
| | rope | ggml_compute_forward_rope_flt_f32 | native-vec | 0.996 | **1.013** | shape-inv | n4096 | **PASS** |
| | silu | ggml_vec_silu_f32 | native-vec | 1.000 | **0.997** | shape-inv | n4096 | **PASS** |
| | gelu | ggml_table_gelu_f16 (BSS-LUT) | scalar-f16-LUT | 0.130 | **0.141** | shape-inv | n4096 | **FAIL(STRUCTURAL)** |
| | add | ggml_compute_forward_add | native-vec | 0.899 | **0.967** | shape-inv | n4096 | **PASS** |
| | mul | ggml_compute_forward_mul | native-vec | 0.991 | **0.963** | shape-inv | n4096 | **PASS** |
| | scale | ggml_compute_forward_scale | native-vec-light | 1.045 | **0.998** | shape-inv | n4096 | **PASS** |
| | cpy | ggml_compute_forward_dup/cpy | native-vec-streaming | 0.960 | **0.987** | shape-inv | n4096 | **PASS** |

### ★达标数 = **16 / 28** PASS (57%) · 12 FAIL (含 gelu 结构差)
- PASS (16): FLAT{q4_0,q4_1,q8_0} · Kquant-vecdot{q5_K} · **Kquant-GEMM{q2_K,q4_K,q5_K}** · iq{iq4_nl} · forward{softmax,rms_norm,rope,silu,add,mul,scale,cpy}
- FAIL (12): FLAT{q5_0,q5_1} · Kquant-vecdot{q2_K,q3_K,q4_K,q6_K} · Kquant-GEMM{q3_K,q6_K} · iq{iq1_s,iq1_m,nvfp4} · forward{gelu 结构差}

### ★对手成色披露 (赢弱对手合法·如实标)
- **hard-carry 真赢强手调** (真硬碰硬空间): **Kquant-GEMM q2_K/q4_K@prefill**（vs vl128 hand-tuned block-dot·nr64 1.38×/1.57× WIN）· **rms_norm**（vs 2-pass native·1.24× 1-pass fusion）。
- **赢中/弱对手** (成色低·披露): FLAT q4_0/q4_1/q8_0 = parity-by-adoption（发射 ggml 自身 block-dot·tautological）· q5_K vecdot = opp 唯一未手调 K-quant（immaturity 非赢手调）· iq4_nl = 对手 native-vec 中（旧 STRONG 收窄）· Kquant-GEMM q5_K = 对手 qh-burdened 弱 block-dot（1.36 GMAC/s）· forward add/mul/scale/cpy/silu/rope/softmax = autovec 对称 parity。
- **gelu = 结构差非公平速度 A/B**：对手 = `GGML_GELU_FP16` f16-LUT·我方 tanhf（赢精度 ~2500×·maxULP-vs-opp=11267 因我方更准）→ 0.141× 是 LUT-vs-tanhf 结构差·**非速度失败**·§六.3 明标"结构非速度"。

---

## 1. ★★头条发现：clang-18 域下 Kquant-GEMM cross-op 从 gcc-batch1 全 LOSS **翻正**

**这是本役最载重结果，直接兑现 §六.1 clang-18 对称域口径的必要性。**

G7 batch1（`experiments/active/g7-census/batch1-kquant-rvv/`）在 **gcc-15.2 对称域**下测得 K-quant GEMM cross-op **全 20 cell LOSS**（0.037×–0.78×）。batch1 evidence 自己诊断为 **[CASE-COMPILER-ASYMMETRY]**：
> "gcc-15.2 -O2 在 un-pipelined emitter 巨型全展开体上 codegen 崩塌·vsetvli storm·比 clang 慢 2–3.4×；casefile clang-17 ours q4_K 4.25 GMAC/s"。

**本役 clang-18 对称域** ours q4_K = **4.31–6.79 GMAC/s**（nr16–64）→ 与 clang cross-check（4.25）吻合、比 gcc-batch1 的 ours 1.73 GMAC/s 快 ~2.5×。→ 在**决令 §六.1 强制的 clang-18 对称域**下（fair·both clang-18），**S6-tiled 的 q2_K/q4_K/q5_K GEMM 在 prefill 翻正 WIN**（nr64: 1.38×/1.57×/1.26×）。

**判别键 = 我方核是否 S6-tiled**：q2_K/q4_K/q5_K = S6-tiled repack GEMM（amortize·随 nr 增长翻正）；**q3_K/q6_K = PLAIN 未 tiled（S6-NULL）**·ours ~0.58 GMAC/s·nr-flat 深 LOSS（结构·非编译器）。

**双域并存诚实纪律**（T3_A compiler_axis 制度化）：
- **clang-18 sym = MAIN verdict**（本表·决令 §六.1 强制·双板一版一表）→ Kquant-GEMM q2/q4/q5 @prefill WIN。
- **gcc-15.2 = 部署附注**（rvv 板出货编译器·batch1 全 LOSS）→ 反映"若部署在 gcc 下·un-pipelined emitter 被 gcc 拖垮"的真部署现实。
- ⚠**flag 主会**：此域切换使 K-quant GEMM headline 翻转·与 memory「q4_K rvv 系统账 gcc-death」及 [CASE-COMPILER-ASYMMETRY] canon（rvv shipped=gcc）张力显著。本表按决令走 clang-18 MAIN·但**不主张 gcc 部署下同赢**（那是 batch1 LOSS·部署附注保留）。

---

## 2. 输格清单 (<0.8 · 供下一役攻坚 · 靶单优先)

| 输格 | cold | 对手 | 攻坚出口 (§六.3 靶单参照) |
|---|--:|---|---|
| **Kquant-GEMM q3_K@prefill** | 0.16 | vl128 hand-tuned blockdot | **Exit B 优先**：ours PLAIN 未 tiled（S6-NULL）→ 键控 S6-tile 发射（同 q2/q4 的杠杆·真 fixable·非 physical） |
| **Kquant-GEMM q6_K@prefill** | 0.17 | vl128 hand-tuned blockdot | **Exit B 优先**：同 q3_K·PLAIN dual-plane 未 tiled→S6-tile 发射候选 |
| Kquant-vecdot q2/q3/q4/q6_K | 0.18–0.34 | vl128 hand-tuned | **Exit C 轴转移**：vec_dot 天然 weight-recon-bound·真 beat 归 GEMM 轴（已翻正）·vec_dot 维持 LOSS |
| iq1_m / iq1_s | 0.15 / 0.37 | vl128 hand-tuned gather | **Exit A**：gather-throughput 硬件天花板 [GAP-IQ-GATHER-VS-VECTORIZED-RVV]·iq1_s noisy(o_iqr16%)→重测候选 |
| nvfp4 vecdot | 0.56 | generic fp4 autovec | Exit A：generic 对手·fp4 罕见格·low-value |
| FLAT q5_0 / q5_1 | 0.45 | native-vec qh-5bit | [GAP-Q5x-QH-BLOCKDOT-EMIT]·deployed 路已绿(GEVM leaf)·block-dot 次路 low-value 不攻 |
| **gelu** | 0.14 | f16-LUT | **非攻坚**（结构差·我方赢精度 2500×·非公平速度 A/B） |

**优先靶单排序（下一役·真 fixable 先）**：① **Kquant-GEMM q3_K/q6_K S6-tile 发射**（Exit B·同 q2/q4 已验杠杆·翻正潜力最高）② iq1_s noisy 重测澄清 ③ iq1_m/iq gather Exit A 具名收手。

---

## 3. Flags (决令要求)

- **rvv commit pin**：opp 树 `/home/ubuntu/vericurve-rv-lab/llama.cpp` = e36a602（clean·== §五 aab4168c 定档）· 本地 weft-opt HEAD = aab4168c6（rvv 干净）。
- **q3_K gemm kernel drift**：weft-opt 再生 q3_K gemm 源 md5 `8a156836` ≠ MANIFEST(677a2f7f) `54beac58`（fixture 在 677a2f7f→aab4168c 间被编辑）·q2/q4/q5/q6 md5 EXACT MATCH。→ **已 clang-18 fresh 重验 byte-exact**（verify_q3K.c GEMM+GEVM int_mismatch=0 全 shape）·非隐患。
- **FLAT q5_0/q5_1 count + k1 侧**：归 k1 agent（本役 rvv only）。k1 树无自有 .git(walk-up→错误 repo)·k1 commit pin 归 k1 agent（§五 已 flag）。
- **q8_0 M=1 单点噪声**：ratio_cold_best=0.078（opp 单 rep p_iqr=1.06 blip）·**ratio_cold_med=0.965 是 verdict**（median-robust）·PASS 不受影响。
- **q4_K vecdot M=1 opp p_iqr=12.5% / q2_K o_iqr4.4%**：噪声高但 verdict（深 FAIL <0.8）对噪声鲁棒·非借口翻案。

---

## 4. DEQ-AXIS 18 状态

**未做 · flag follow-up**（决令：headline 优先·DEQ 时间不够 flag·别硬撑超时）。DEQ-AXIS harness 已存（`experiments/active/g7-census/iqfp4-dequant-rvv/dequant_census_driver.c` + kernels_dequant/·gcc-15.2 域 5396a22d 已测 9/18 WIN）·clang-18 对称重测**待下一役承接**。DEQ 不进 0.8 headline 分母（[DEQ-AXIS] 独立子账·体例）。

---

## 5. Board 卫生 (决令要求 · 测完清理)

- **pkill 清 hung**：STRAY_BENCH=0（vecdot_census/iqfp4_vecdot/bfwd_rvv/census 全清）。
- **co-tenant 未扰**：vLLM core0,1（2 procs）全程未 restart / 未 pin 到其 core·pin 恒在 core8-15（idlest·load-gate>=70%）。
- **loadavg 记录**：begin 2.09–2.39 → 测量期峰 2.99（co-tenant drift·非我方）→ final **2.08**（restore）。
- **lib read-only**：opp = 源码 clang-18 重编（.o·非改动 .so）·主树/build/build-weft 未改（weft-opt 只读再生）。
- **禁 git 遵守**：无 git add/commit（主会审后 commit）。scratch `/tmp/g8s6_{trial,fwd,gemm}` 保留至 seal 抽取后清（见清理记录）。
