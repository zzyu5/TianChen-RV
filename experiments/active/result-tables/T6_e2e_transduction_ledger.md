# T6 — e2e 传导会计 ledger（[DISCRIMINATOR-OPPONENT-BOUND-TYPE] 权威账·系统账）

> **定位**：整模型 e2e 分相双板 **传导会计**（ROADMAP:119 T6 节点）。把 kernel-axis 赢 → e2e 的**传导/wash 判决**逐格入册·四列会计 = `{kernel-axis ratio · e2e ratio(ON/OFF) · 对手 bound-type · 传导/wash verdict}`。**唯一判别键 = [DISCRIMINATOR-OPPONENT-BOUND-TYPE]**：kernel 算力赢 **传导 IFF 对手 compute-bound · washes IFF memory-bound**·bound-type 由 **相(M)×板×出货编译器** 对该 kernel 计算路的 autovec 质量共同决定（**非格式名**）。
> **★账本隔离（禁互推）**：本表 = **e2e 系统账**（真 llama.cpp 整模型分相）·**≠** kernel-sym 0.8 gate micro 账（单分母四档）·**≠** perf-covered 9/83（fair-protocol ≥parity 格占比）·**≠** certified 101/108（构造轴 byte-exact）。四账各自成立·**一格可 kernel 赢而 e2e wash**（IME 5.51×→0.86× 铁证）。
> **★正交第二轴 = [CASE-COMPILER-ASYMMETRY]**：**gcc-death**（同源 gcc-15 vsetvl 暴涨）是**编译器轴**输·**非 bound-type 轴**——q4_K 同格式 @rvv(gcc)=0.334× LOSS vs @k1(clang)=1.101× 传导·两轴须分开归因。
> **数字来源**：逐格 sealed / batch report / T-PERF cell·**零手填**·commit 见行末。日期 2026-07-16（batch1 收尾·度假自治）。

---

## 1. 传导会计四列主表（measured·双板分相）

| 格 · 板 · 相 · 轴 | kernel-axis | e2e ON/OFF | 对手 bound-type（证据） | **verdict** | HEAD-live / provenance |
|---|--:|--:|---|:--:|---|
| **q4_K@k1 · prefill · GEMM** vs 手调 hand-brick | 1.187× cold | **1.101×** | **compute-bound**（objdump 767 insn/80 vwmacc·M=128 权重跨 token 复用·高 arithmetic intensity） | **传导 ★首个 e2e beat-hand-brick** | md5 `9e057adb`=HEAD regen byte-id · commit `972ab5c2a` |
| **q5_0@k1 · decode · GEVM** vs block-dot | 1.76× cold | **1.966×** *(fresh clean)* | **compute-bound**（stock 80 insn·per-block vwredsum×2·3.28 t/s≪内存墙 ~13·clang-18 弱 5th-bit autovec） | **传导**（=frozen 1.970× Δ0.2%） | .so `df88afa3`=sealed byte-id · commit `195b11910` |
| **q5_0@k1 · prefill · GEMM** | — | **2.213×** *(fresh clean)* | compute-bound（GEMM M=128 复用） | **传导** | `df88afa3` · commit `195b11910` |
| **q5_1@k1 · decode · GEVM** vs block-dot | 2.24× cold | **2.073×** *(deployed==proven·裁停冗余 fresh)* | compute-bound（同 q5_0·5th-bit qh 重算） | **传导** | .so `43569a46`=sealed byte-id · GEVM `.inc 4ad42d91`=A5 proven |
| **q5_1@k1 · prefill · GEMM** | — | **2.344×** *(deployed==proven)* | compute-bound（GEMM M=128） | **传导** | `43569a46` |
| q4_K@k1 · decode · isolation（未换 gevm） | — | 0.995× | memory-bound（M=1·两侧同 stock gevm·无算力赢可测） | **isolation-control**（非 wash·证 prefill-only） | 本役 batch1 §1 |
| **q4_0@k1 · decode · GEVM**（**wash 反面**·同板同编译器 as q5） | ~parity | **0.857×** | **memory-bound**（stock 4-bit nibble 高效·6.62 t/s=2× q5 stock·repack 增内存税） | **WASH** | sealed `k1-vlen256-q4_0-flip` · CI[0.849,0.865] n=10 |
| q5_0@**rvv** · decode（**跨板同格式**） | 1.44-1.67× | 0.838×(OLD)/0.929×(RB) | **memory-bound**（gcc-15.2 autovec 5th-bit 高效→内存墙） | **WASH** | frozen `archive/g5/M2-q5_0` / `g7-l2` |
| q5_1@**rvv** · decode | 1.23× | **0.7836×** | memory-bound（−21.6% GEVM wash·n=20） | **WASH** | frozen `archive/g5/M2-q5_1` |
| [canon 铁律] IME · kernel→decode | 5.51× | **0.86×** | memory-bound（compute win 不传导） | **WASH** | [[kernel-wins-dont-transplant-to-e2e]] |

**★判别键三点隔离（CONFIRMED）**：
1. **同板/编译器/核族/相·唯一差=格式** → q4_0@k1 decode 0.857×(WASH·对手 memory-bound) vs q5@k1 decode 1.97×(传导·对手 compute-bound) = 单变量对照。
2. **同格式跨板** → q5@rvv wash(gcc autovec 5th-bit 高效→memory-bound) vs q5@k1 传导(clang 弱 5th-bit→compute-bound) = bound-type 依 **板×出货编译器** 非格式名。
3. **相轴** → q4_K@k1 prefill 传导(M=128 compute-bound) + decode isolation 0.995×(M=1 memory-bound) = 同格同板·相翻转。

## 2. gcc-death 正交轴（[CASE-COMPILER-ASYMMETRY]·非 bound-type 判决）

| 格 · 板 · 相 | e2e ours/blockdot | 根因 | verdict |
|---|--:|---|:--:|
| **q4_K@rvv · prefill** | **0.334×**（3× 慢） | 同源 gcc-15 vsetvl 922 vs clang 81（10× 膨胀）·**编译器轴非 bound-type** | **gcc-death LOSS**（`T-PERF1b`·出货 rvv=gcc-deploy MAIN） |

⟹ **q4_K 同格式：@k1(clang-18 对称)=1.101× 传导 · @rvv(gcc-15 出货)=0.334× gcc-death**。**两轴分立归因**：k1 传导是 bound-type 轴真赢·rvv 输是编译器轴 artifact。修 rvv 需 clang-deploy 或 gcc codegen fix（[CASE-KQUANT-GCC-CODEGEN]）·**非** bound-type 问题。

## 3. 已 perf-covered-green 的 e2e 传导格（既有账·本表 corroborate·计数不动）

| 格 | e2e | verdict | perf-covered 登记 |
|---|--:|:--:|---|
| q4_0@rvv prefill | 5.92× | 传导（**routing-win**·L1 path-selection·非 codegen 快） | `T-PERF1`·系统账 routing |
| q8_0@ime | 2.233× | 传导（M7 wide-vmadot·beat stock·无内存税） | 9/83·`6af32c5a` |
| q4_0@ime | 1.0088× | parity（M7·tie stock） | 8/83·`0ed58a5a` |
| q5_K@k1 prefill | 1.641× | 传导（首个 our-kernel K-quant e2e） | 7/83·`47e29b35` |

## 4. perf-covered any-board 演化（PR-14·硬冻结·登记不擅改）

- **q4_K@k1 e2e prefill 1.101×（首个 e2e beat-hand-brick·deployed==proven·compute-bound 真传导）= 成色升级候选**。
- **★硬冻结**：q4_K@k1 已是 perf-covered 绿格（frozen prefill 1.085× Win-K1-VLEN）·本役 HEAD-live hand-brick beat = **成色注记升级**（kernel 账→e2e-transduced-beat-hand-brick）·**any-board 规则（收口令〇.1 终裁）= 永不增头条格数·只升成色**。
- **perf-covered 维持 9/83**·denom=roster/M4 SINGLE-SOURCE·**不案头执行·登记 PR-14 待用户裁成色注更新**。

## 5. 教科书结论（论文素材·C3′ 机制证词）

- **[DISCRIMINATOR-OPPONENT-BOUND-TYPE]** = 能力键控赢**是否传导 e2e** 的判别键·**由对手基线 bound-type 决定**（相×板×编译器）·**非格式名·非 kernel 倍数大小**。这是 [[kernel-wins-dont-transplant-to-e2e]] 的**正判据形式**（何时传导·非仅何时 wash）。
- **两个 genuine 强传导**：q4_K@k1 prefill（强对手手调·compute-bound）+ q5@k1 decode（弱对手 block-dot·compute-bound）·**成色分层诚实**（前者 beat-hand-tuned·后者 beat-weak-baseline）。
- **正交 gcc-death 轴**：q4_K@rvv 0.334× = 编译器 artifact 非机制失败·[CASE-COMPILER-ASYMMETRY] 再证。
- **process-hygiene**：并行 e2e 线撞同一 LIVE 部署 .so 路径互 swap → 隐蔽 race → [[parallel-lines-need-disjoint-files]] 延伸（**共享部署产物·非仅源文件·亦需不相交**）。

---

*ledger 首版 2026-07-16 · e2e 传导战役 batch1 收尾 · 数字逐格 sealed/T-PERF/batch report 溯源 · 四账禁互推 · perf-covered 9/83 硬冻结（PR-14 只登记）。*
