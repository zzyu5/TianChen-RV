# T-VALIDITY Stage-1 — k1/VLEN256 承重-不对称格对称重测 (r74 q4_K 3.106× + r75 q5_K 1.916×)

> 生成 2026-07-09 · [CASE-COMPILER-ASYMMETRY] **Stage-1 重测段 · board B / k1** · 分支 `refactor/full-refactor-m1`.
> **口径冻结**：本文件是 Stage-1 board-measured 证据存档，**不改** T8 / 8-gate doc / MANIFEST / memory 的既有 verdict；
> 撤回/门塌/reclassify 的终审属 Stage-2。此处只报**数 + 预注册判读**。未 git commit。factory 树只读、md5 前后核对复原。
>
> **案由**：T-VALIDITY 底账把 r74/r75（k1/VLEN256 q4_K 3.106× / q5_K 1.916×）标为**不对称（撤回候选）**，
> 依据是行内自陈 "ours=clang-18 vs factory-as-shipped (**same asymmetry as board A**)"，并以此**关 q4_K [PERF-1] 8-gate ③⑤**。
> Stage-1 任务 = 对这两格做**对称重测**（0b 方法学），预测「蒸发」（类 rvv q4_K 1.884→0.334）。

---

## VERDICT（一句话）

**预测证伪 → 两格 SURVIVED（幸存），非蒸发。** 决定性事实：**k1 出货 factory `libggml-cpu.so` 是 clang-18 编的**
（CMakeCache + 反汇编指纹双证），**不是** gcc。因此 r74/r75 原测 = **ours clang-18 vs opp clang-18 = 本就对称-clang**，
"same asymmetry as board A" 的自陈**事实错误**（board A/rvv 出货 ggml = gcc-15；board B/k1 出货 ggml = **clang-18**，
两板并非同一情形）。对称-clang 重测三口径全复现 ~3.1×/~1.92×。这与 rvv q4_K（真 gcc 对手→蒸发 0.334×）**相反**。

---

## 1. 编译器归因 — factory 对手 = clang-18（决定性）

原测对手 = 板自带 `/data/k1build-stock/bin/libggml-cpu.so` dispatch 的 `ggml_vec_dot_q{4,5}_K_q8_K`
（k1/VLEN256 → tail-jump 到 `_vl256` 变体，源 `.../ggml/src/ggml-cpu/arch/riscv/quants.c`）。其编译器身份：

| 证据 | 内容 | 结论 |
|---|---|---|
| **CMakeCache** `/data/k1build-stock/CMakeCache.txt` | `CMAKE_C_COMPILER:FILEPATH=/usr/bin/clang-18` · `CMAKE_CXX_COMPILER=/usr/bin/clang++-18` · `C_FLAGS: -fno-integrated-as -O3 -DNDEBUG … -march=rv64gcv_zfh_zvfh_zicbop_zihintpause -mabi=lp64d` | **clang-18** |
| **build log** `build_m1_final.log` | clang 专属诊断（`-Wdouble-promotion` "implicit conversion increases floating-point precision: '_Float16' to 'float'"、`-Wshadow`、`-Wmissing-prototypes`）编译 `arch/riscv/quants.c.o` | **clang-18** |
| **.comment 双串** | `GCC: (Bianbu 13.2.0) 13.2.0` **和** `Bianbu clang version 18.1.8` | **`-fno-integrated-as` 造成**：clang 编译 + GNU as(gas) 汇编，gas 盖 "GCC:" 戳 → **非混编，是 clang-18 全程** |
| **反汇编指纹 == 我方 clang-18 重建**（下节 §3） | factory `_vl256` = 19 vsetvli/0 spill；我 clang-18 重编同源 quants.c = **19/0** 完全一致 | **clang-18 坐实** |

**⇒ 原测 3.106×/1.916× 的 ours 与 opp 两侧都是 clang-18 = 编译器对称-clang（本就），非 clang-ours-vs-gcc-shipped 不对称。**

## 2. gcc-symmetric 在 k1 = 不可行（且 MOOT）

任务预设「用 SpacemiT gcc-15 编两侧」。实测：**k1 上无 gcc-15**，只有 Bianbu **gcc-13.2.0**（遍历 `/opt /usr/local /data` 无 15）。
memory 记的 "SpacemiT GCC15.2 fork" 是 **IME/`xsmtvdotii`** 工具链，与本格无关（q4_K/q5_K repack GEMM 是**普通 rv64gcv+zvfh**、非 IME）。gcc-13.2.0 对**两侧都编不了**：

| 目标 | gcc-13.2.0 失败 | 缺失能力 |
|---|---|---|
| **ours**（s6_q4K.c / gemm_q5_K_q8_K.kernel.c） | `s6_q4K.c:9646: 'vfloat16m1_t' was not declared`（q5_K:83 同） | gcc-13 **无 zvfh 向量 intrinsic 类型**（`vfloat16m1_t`/`__riscv_vle16_v_f16m1`；gcc-14 才引入） |
| **opp**（quants.c 标准 factory 源） | `quants.c:181: '__RISCV_FRM_RNE' undeclared` · `182: '__RISCV_VXRM_RNE' undeclared` | gcc-13 **无 RVV 静态舍入模 intrinsic**（`__riscv_vfcvt_x_f_v_i32m8_rm` 等） |

**⇒ gcc/gcc 对称在 k1 硬件-工具链层不可行（两侧皆编不了）。但这不影响 verdict：k1 对手从来不是 gcc，是 clang-18；
可行且有意义的对称口径 = clang/clang，而那正是原测本来的口径。**（对照：rvv board A 有 gcc-15 且出货 ggml=gcc-15，
故 rvv 侧 gcc-symmetric 可行且证得蒸发；k1 与之结构不同。）

## 3. 对称-clang 重测数（clang/clang，三口径互证）

board k1/SpacemiT-X60/VLEN256(vlenb=32)/8c/gov=perf 1.6GHz，core7，K=2048 nr=64 nc=512 iters=20，cold N=12，paired 同进程 A/B（争用 common-mode 在比值抵消）。ours = clang-18.1.8 -O2 `-march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -ffp-contract=on`（sealed 形，md5 90d454da/ba30ba54，REUSED 无 regen）。

| 口径 | 对手 | q4_K median (min/max, IQR%) | q5_K median (min/max, IQR%) | 分类 |
|---|---|---|---|---|
| **原测 (T8 r74/r75)** | factory .so (=clang-18) | 3.106× [3.096,3.112] | 1.916× [1.910,1.922] | (自标"不对称"←事实错) |
| **RETEST-A** ours-clang-18 vs **factory .so**(clang-18) | 同原测对手 | **3.096×** (3.070/3.114, ~0.55%) | **1.929×** (1.919/1.933, ~0.31%) | **对称-clang** |
| **RETEST-B** ours-clang-18 vs **我方 clang-18 重编 quants.c**（standalone、fully-controlled 同 {compiler,flags}） | 源级重建对手 | **3.101×** (3.089/3.136, ~0.48%) | **1.918×** (1.911/1.930, N=11) | **对称-clang（全控）** |

- 三口径 q4_K 全落 **~3.10×**、q5_K 全落 **~1.92×**，互差 <1%（噪声内）。**原 3.106×/1.916× 在编译器对称-clang 下完整复现 = SURVIVED。**
- **自检 SOP**：RETEST IQR%（q4_K 0.48–0.55% / q5_K 0.31%）≤ 历史地板（原测 cv 0.26–0.46%、IQR 0.52–0.63%）×1.5 ⇒ 自检过。loadavg 全程 2.3–3.2（k1 共享板，同原测 3.0–3.4 量级；paired 比值对争用稳健）。

### 原始 ratio（耐久）
- RETEST-A q4_K: 3.087 3.103 3.110 3.070 3.093 3.075 3.088 3.114 3.099 3.104 3.098 3.082
- RETEST-A q5_K: 1.924 1.931 1.930 1.933 1.929 1.930 1.919 1.924 1.919 1.928 1.923 1.930
- RETEST-B q4_K: 3.108 3.105 3.113 3.099 3.093 3.136 3.093 3.114 3.102 3.098 3.100 3.089
- RETEST-B q5_K: 1.917 1.916 1.914 1.921 1.920 1.930 1.920 1.918 1.918 1.911 1.918

## 4. objdump seal（k1 gcc-vs-clang 应交付项 → clang-only 可得，gcc 不可编）

| 对象 | vsetvli | spill | reload | vwmacc | textB | 备注 |
|---|---:|---:|---:|---:|---:|---|
| ours q4_K (s6) clang-18 -O2 | 70 | 4 | 6 | 2240 | 25304 | **== MANIFEST sealed S6 -O2**（bit-复现） |
| ours q5_K clang-18 -O2 | 53 | 151 | 356 | 2240 | 38822 | **== MANIFEST sealed Q5K -O2** |
| opp `q4K_vl256` clang-18 重建 | 19 | 0 | 0 | 0 | — | == factory .so |
| opp `q5K` clang-18 重建 | 15 | 0 | 0 | 0 | — | == factory .so |
| **factory .so `q4K_vl256`** | **19** | **0** | 0 | 0 | — | **== 重建 ⇒ clang-18** |
| **factory .so `q5K`** | **15** | **0** | 0 | 0 | — | **== 重建 ⇒ clang-18** |

- **"gcc vs clang k1" 对比不可产**：k1 无 gcc-15、gcc-13 编不了 ours（zvfh 向量类型）。gcc-15 全展开 spill 病理（820 vsetvli/742 spill）是 **board-A/rvv** 事实（casefile §2），k1 无 gcc 侧可复。对手 = clang，gcc codegen 对 k1 比较**无关**。
- **identity**：golden vs S6 @VLEN256，int + norm 均 **IDENTICAL（0 byte mismatch）** ⇒ 正确性保持。

## 5. 对 q4_K 8-gate ③⑤ 的影响（预注册判读；终审属 Stage-2）

- **门③（双板 objdump）**：k1 objdump seal 是合法 clang-18 codegen 封印（spill 81→4、vwmacc 2240 VLEN-不变）；**是 codegen 封非 perf 比值，不受编译器对称性问题波及** → 稳。
- **门⑤（双板 kernel-轴验证）**：k1/VLEN256 半（3.106×/1.916×）= **对称-clang → 不因编译器对称而塌**。建议 T-VALIDITY 底账把 r74/r75 从 **「不对称（撤回候选）」reclassify 为「对称-clang（幸存）」**。
- **★关键裂口（诚实必标）**：门⑤ q4_K "dual-board" = rvv 1.884×(板A) + k1 3.106×(板B)。**板A/rvv 半 = 已证 clang-vs-gcc artifact**（casefile；对称-gcc e2e=0.334× LOSS）；**板B/k1 半 = 幸存对称-clang**。**两板并非"same asymmetry"——一板蒸发、一板幸存**。故门⑤ 若要求"双板都是诚实赢"，卡点在**板A 的 1.884×**（另线/casefile 管辖），**不在 k1**。
- **成色注记（[NG-4] 完整）**：对称-clang = clang-**域**赢；但 k1 上 **ggml-as-shipped 本身即 clang-18 编** ⇒ 在 k1 上「clang-域赢」== 「as-shipped kernel-轴 beat」（比板A 立足更稳）。仍限 **kernel-轴 prefill-GEMM，非 e2e，非 sealed 8-gate**；e2e 对称真相 k1 **未测**（rvv 侧为 0.334× LOSS，**禁外推**）。
- 本文**不自宣**门③⑤ 塌 / 撤回；报：k1 两格幸存-对称-clang + 建议 reclassify + 标 dual-board 板A/板B 裂口。

## 6. k1 复原核对

- factory `.so` md5 **测前 = 测后 = `871169a0123139692177468b3c8578be`**（`libggml-cpu.so.0.15.1`）⇒ **对手树只读、零改动**。
- 主树 + 本地 build 未动；governor 留 as-found（perf/1.6GHz）；board scratch `/tmp/tcrv_k1_symretest/`（我方新建）测毕清除；预存 `/tmp/tcrv_k1_kquant_t4a/`（原 T4a）未触。

## 附：一手证据指针
- 归因：`/data/k1build-stock/CMakeCache.txt`（clang-18）+ `build_m1_final.log`（clang 诊断）+ 指纹 §3/§4。
- 源：`/tmp/q4k_tile_s6/{s6_q4K.c md5 90d454da, gemm_q5_K_q8_K.kernel.c md5 ba30ba54, golden_q4K.c b0b5beac}`（host cache，byte-verified）；对手源 `/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu/arch/riscv/quants.c`（`_vl256` @1975 / q5_K @2081）。
- 驱动：`tools/e2e-harness/board/kquant_gemm_paired_driver.c`（paired A/B，opp = `ggml_vec_dot_q{4,5}_K_q8_K`）。
- 原 T4a 账：`experiments/active/kquant-k1-vlen256-kernel-axis-t4a/{MANIFEST.md,paired_k1_vlen256.csv}`。
