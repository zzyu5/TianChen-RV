# [K1-SEAL] e2e 传导终审 — sealed Win = NO（两板皆输 as-shipped）+ RVV-E2E 跨板对照（2026-07-10）

**板** `ssh k1`（SpacemiT X60, VLEN256/vlenb=32, 8c, clang-18 出货 ggml）。复用已 emit sealed 源（md5 90d454da，无 re-emit/无 tcrv-opt）。模型 tinyllama-1.1b-Q4_K_M。k1 baseline 4 anchor 测前后全 match（stock 871169a0 未动）。genuine：byte-fingerprint 部署 + no-preload-FAILS 控制 + 配对紧 IQR + 无 harness overhead。

## 部署五验（全 PASS）
clang-18 -O2（`-march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs`）编 sealed s6_q4K.c → `libtcrv_q4kseal.so`，正门 forward-declare + VLEN256-dispatch（符号 undefined in libggml-cpu.so，仅 LD_PRELOAD 解析）。① 符号存在 ② 反汇编 = 我方 S6（**70 vsetvli/4 spill/2240 vwmacc** = sealed -O2 指纹）③ vtype VLEN256 sealed ④ banner `TCRV EMITTED GEMM(q4_K_16x1 VLEN256 compiler-emitted S6) ENGAGED` ⑤ compiler=clang-18（70/4 是 clang 签名；no-preload 控制 FAILS `undefined symbol` → 证跑的就是我 preload 的 clang 对象）。

## 传导账（Amdahl，测前算）
q4_K=82.0%/q6_K=18.0% matmul 权重（无 q5_K）。g₄=3.10 micro → projected S(A/Bq4kOFF)=2.17× ≥parity → 预注册 sealed 候选。

## e2e prefill pp128（10 轮配对交错，core4-7 -t4）
| variant | median t/s | IQR% |
|---|---|---|
| **A** emitted S6 (vl=8) | **10.84** | 0.56 |
| Cbrick hand-brick (vl=16, 同 build) | 14.47 | 0.25 |
| Boff block-dot (q4_K OFF) | 5.29 | 0.14 |
| Bstock as-shipped factory | 14.45 | 0.54 |

**A/Boff = 2.05×** · **A/Bstock = A/Cbrick = 0.750×** · Cbrick/Boff = Bstock/Boff = 2.73×（repack routing 增益，stock 已有）。pp512 spot: A/Bstock=0.751×（shape-无关）。**Per-tensor back-out（q4_K=79.5% prefill 时间）: deployed g₄(A vs block-dot)=2.80×（vs micro 3.10× → 90% 保留、NOT washed 像 rvv 0.334×）**。decode tg32: A/Bstock=1.003×（parity，仅 patch GEMM、decode 走共享 hand-brick GEVM，预声明兑现）。

## VERDICT: **NOT sealed Win**
- 字面 A/Bq4kOFF=2.05× ≥parity 达标（且不像 rvv wash），**但归因证伪其为 kernel win**：
- **2.05× = repack ROUTING**（stock 已有，Bstock/Boff=2.73× ≥ 我 kernel 2.05×）。
- **vs 真 as-shipped 路径：我方 emitted kernel LOSES A/Bstock=A/Cbrick=0.75×**。
- **根因 = S6 是 VLEN128-tuned（vl=8 = VLEN256 的 16 lane 的一半）；出货 hand-brick 用 native vl=16（满宽）→ emitted kernel 在 VLEN256 浪费一半向量宽 → 0.75× hand-brick**。= **emitter VLEN-adaptivity 成熟 gap（board-无关）**，非编译器 artifact。

## ★Stage-2 validity 精修（k1 "幸存" 需 caveat）
t4a micro 3.10× 是"vs factory block-dot"测的，**但 block-dot 不是 k1 q4_K@VLEN256 的出货路径**（stock 与 tcrv 都 repack hand-brick）。**micro 打的是 non-shipping strawman**；真 e2e 对手（hand-brick vl=16）beat 我方 emitted kernel 0.75×。故 Stage-2 的"k1 kernel-轴 SYMMETRIC-CLANG-SURVIVED 3.10×"应加 caveat：**幸存 = vs block-dot（非出货对手）；vs 真 as-shipped 输（vl-width）**。

## RVV-E2E H2 跨板对照裁断
- **vs block-dot：k1 传导（YES）** A/Boff=2.05×（deployed g₄ 2.80 ≈ micro 3.10）；**对比 rvv-gcc A/Boff=0.334×** → **约束 rvv 谜题**：rvv 的 sub-block-dot 回归**由 gcc-15 codegen artifact 主导**，非 K-quant repack kernel/内存模式的内禀性质。clang 板上同一 emitted kernel = 2× block-dot。
- **vs as-shipped：两板皆 NO** —— rvv 因 gcc 毁 kernel（0.24×）+ rvv-board-specific H-B 内存停顿（clang 0.764×，M0；k1 无此=k1 beat block-dot）；k1 因 vl=8 emitted 被 native-vl=16 hand-brick beat（0.75×）。**两不同机制、同底线：emitted kernel 对 as-shipped 无 e2e Win。**
- **两个 emitter-maturity gap 浮出**：(a) **rvv H-B tile-schedule**（M1 loop-interchange 已选型可修，rvv-board-specific）；(b) **VLEN-adaptivity**（vl=16 for VLEN256，board-无关，需 VLEN256-native emitted kernel 才能挑战 hand-brick）。

## 结论对 north-star
**sealed Win 仍 = 0，且诚实定谳：现 emitted kernel 在两板都输 as-shipped 手调 kernel**。通往 sealed Win 的路 = 补两个 emitter-maturity gap（rvv H-B loop-interchange + VLEN-adaptivity vl=16）。这是"成熟 compiler"的真前沿，非 dead end——k1 证 kernel 本质健全（g₄ 90% 保留），差的是 schedule/VLEN 适配。
