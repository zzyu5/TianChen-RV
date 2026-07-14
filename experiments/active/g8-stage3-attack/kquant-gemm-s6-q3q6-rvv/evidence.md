# G8 阶段三 §六.3 攻坚首战 · K-quant GEMM q3_K/q6_K @rvv S6-tile 发射 (2026-07-14)

> **战果**：q6_K GEMM 与 q3_K GEMM 双双翻正过 0.8 硬门（此前 0.15-0.17× 深 LOSS）。
> S6 strip-outer tiling 从 q2/q4/q5_K 铺到 no-min 的 q6_K/q3_K —— **参数级键控铺面成功**，
> 且 **证伪** 了代码里"S6 output tiling board-proven NULL on q6_K"的旧断言（该断言无当期板证据）。
> **口径**：clang-18 对称域（§六 MAIN）· rvv openEuler VLEN128 · core8 pin · co-tenant vLLM(core0,1) 未扰。
> **byte-exact**：S6 == PLAIN 数值恒等 DYNAMIC 逐位 27/27 shapes（both formats）。**禁 git**（主会审后 commit）。

---

## 0. 结果表 (headline · cold primary · clang-18 sym · vs vl128 block-dot cross-op)

| 格 | 核形态 | cold nr4 | cold nr8 | cold nr16 | **cold nr64** | HOT | 0.8门(prefill nr≥16) |
|---|---|--:|--:|--:|--:|--:|:--:|
| **q6_K** | PLAIN (§六 baseline) | 0.137 | 0.147 | 0.166 | 0.153 | 0.15 | FAIL |
| **q6_K** | **S6 unrolled** ★ship | 0.611 | 0.805 | **0.960** | **1.137** | 1.11-1.20 | **PASS/WIN** |
| q6_K | S6 rolled | — | — | 0.845 | 0.898 | 0.93 | pass（比 unrolled 差·不 ship） |
| **q3_K** | PLAIN (§六 baseline) | 0.152 | 0.158 | 0.165 | 0.166 | 0.17 | FAIL |
| q3_K | S6 unrolled | 0.529 | 0.627 | 0.702 | 0.789 | 0.82-0.84 | 近失（0.79<0.8） |
| **q3_K** | **S6 rolled** ★ship | 1.482 | 1.409 | **1.399** | **1.407** | 1.40-1.43 | **PASS/WIN** |

**判读**：
- **q6_K GEMM：翻正 0.15→0.96(nr16)/1.14(nr64) cold，clean WIN。** ship = S6 unrolled。
- **q3_K GEMM：翻正 0.17→1.40 cold，clean WIN。** ship = S6 rolled（[ROLL] 对 q3_K measured-beneficial）。
- 每格 schedule 键值板测定（宪章规则2）：q6_K 偏 unrolled、q3_K 偏 rolled —— 判别键 = 子块位数（q6_K=8-pos 小展开已入 I$；q3_K=16-pos 大展开 thrash I$）。

---

## a. 逐指令解剖 (objdump·clang-18 -O2·rvv VLEN128)

**判别键锁定 = 我方 PLAIN 核未 S6-tiled → 累加器寄存器悬崖溢出。** raw = `raw/objdump_metrics.txt`。

| 核 | total | vwmacc(MAC) | vl1r | vs1r | csrr(vlenb) | obj size |
|---|--:|--:|--:|--:|--:|--:|
| q4_K (S6-tiled 参照) | 6405 | 2240 | — | — | — | 28KB |
| **q6_K PLAIN** | 20960 | 2304 | 1241 | 971 | 2210 | 81KB |
| **q6_K S6** | 9215 | 2304 | 59 | 38 | 97 | 39KB |
| **q3_K PLAIN** | 20022 | 2176 | 1179 | 1065 | 2242 | 78KB |
| **q3_K S6 (unrolled)** | 9237 | 2176 | 221 | 93 | 314 | 39KB |
| q3_K S6 (rolled) | 1841 | 256¹ | 59 | 51 | 110 | — |

¹ rolled 的 vwmacc 是 runtime l-loop 单份代码（执行 ×8），非静态计数。

**构成清单（为什么 PLAIN 输）**：
1. **MAC 数完全相同**（q6_K 2304 / q3_K 2176 = q4_K 2240 同量级）——有用计算一致，差距 100% 是开销。
2. **PLAIN 溢出 ~2068 整寄存器 spill**（vs1r/vl1r）+ **~2210 csrr vlenb**（算 VLEN-scaled spill 偏移）。反汇编实证：`vwmacc.vx v10 → csrr vlenb → vs1r.v v10`（MAC 结果算出即刻 spill 回栈），是累加器寄存器溢出的确诊铁证。
3. 根因 = PLAIN 把 **全 strip 累加器**（4col×2strip×(sumf+sumi)=~64 vreg）同时 live 跨单个 block loop，越 32-vreg 预算 → per-op spill。
4. **旧代码注释"board-proven NULL / q4_K's register-cliff lever does NOT transfer / ~900-spill = 6-bit 重构内禀"= 误诊**：那 ~2068 spill 归因于"两平面 6-bit 重构"，实为**累加器悬崖**（strip-outer scoping 即可消除），非重构内禀。当期板 A/B 证伪。

## b. 施工 (S6 strip-outer tiling·file:line)

**改动**：`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp`（323 ins / 338 del，1 file）。
- `emitRepackKQuantGemmBodyQ6K`（~11039）与 `emitRepackKQuantGemmBodyQ3K`（~18280）：把 **weight-strip 循环 `for h` 从 block loop 内部（strip-interleaved）提到 block loop 外部（strip-outer）**，累加器数组 `[c][h]`→`[c]`（每 strip 独立一份 block loop，live set 减半 ~64→~36 vreg）。
- 变换为**纯发射重排 · byte-exact**：每 out[c][h] 仍在相同 blocks、相同 (l,j/sh,sh/grp,k,p/l,q) 序累加；h 从最内提到最外只改**独立**（disjoint 权字节/disjoint 输出）strip 的交织序，不动任一输出的 fold 序。
- 同结构铺到 rolled 与 unrolled 两 schedule（[ROLL] 轴正交，仍走 `resolveRepackMainTermRolled`）。
- 更新旧误诊注释（3672/3697/11039/q3_K header）为当期板证据。

## c. G1 (byte-exact + objdump) + G2 (cold ratio)

**G1**：
- **byte-exact 硬门 PASS**：S6==PLAIN DYNAMIC 逐位 **27/27 shapes**（K∈{256,512,2048}×nr∈{4,8,64}×nc∈{16,32,512}），q6_K + q3_K(unrolled) + q3_K(rolled) + q6_K(rolled) 四核全 0 mismatch。driver = `g8s6_eq_driver.c`（同 bytes 跑两核 memcmp 输出）。
- **objdump S6-tiled 证**：spill vl1r/vs1r ~22×↓（q6_K 2212→97）、csrr ~23×↓、total ~2.27×↓，MAC 恒等。见 a。
- **lit forced-relink 全绿**：19/19 K-quant GEMM fixtures（含 rolled/col-outer/vlen256 变体）FileCheck PASS；53/54 全 repack fixtures PASS（1 fail = `repack-gevm-colgroup-tiled-q4-K` 我未触·**clean-emitter 亦 fail = 预存**·ROUNDTRIP RUN 行无关）。

**G2**：见 §0 结果表。**q6_K cold nr16 0.96 / nr64 1.14 = 锁定翻正；q3_K(rolled) cold 1.40 = 锁定翻正。**

## 结果 / 认输门槛

- **无认输**：解剖了（累加器悬崖 spill，非 6-bit 重构内禀）→ 学了（strip-outer 减半 live set）→ 造了等价 S6-tile → **两格皆翻正过 0.8**。不构成"具名 X"（预期的"6-bit dual-plane recon 32× 内禀"被证伪：对手 block-dot 亦解同样 6-bit，3.8 GMAC/s 证明可快；我方 0.58 是 emitter 溢出成熟度差，非物理墙）。
- **q3_K schedule 张力（如实标·follow-up）**：q3_K 默认走 UNROLLED（0.79 近失）；WIN 需 ROLLED stamp。`resolveRepackMainTermRolled` 的 measured-gate 现按 proxy(=4096)+coreLmul 键，q3_K 与 q6_K **同 proxy** → 无法独立选择（q3_K 要 rolled、q6_K 要 unrolled）。当期板测供 measured 证据，但自动选择需把 [ROLL] gate 键细化到 decode_model（结构级 follow-up，非本役 S6 铺面 scope）。发射器已能 byte-exact realize 任一 schedule；选择归 selector。

## 编译器域如实标 (clang-18 域 vs gcc 部署·[CASE-COMPILER-ASYMMETRY])

本翻正 = **clang-18 对称域**（ours 与对手同 clang-18.1.8 + 同 rich-march + `-fno-integrated-as`）。rvv 板 shipped 编译器 = gcc-15；**gcc 部署下未必同赢**（G7 batch1 gcc 域 K-quant GEMM 曾全 LOSS = un-pipelined emitter 被 gcc codegen 拖垮）。但 S6 的机制收益（消 2068 spill）与编译器无关（是发射结构改），gcc 域预计亦大幅改善，**待 gcc 域复测确认**（未做·follow-up）。主表按决令 clang-18 MAIN。

## 触碰文件清单 · board 卫生 · 禁 git

- **触碰（未 commit）**：`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp`（S6 strip-outer q6_K+q3_K + 注释订正）。
- **新增 casefile**：本目录（evidence + kernels/ + raw/ + eq driver）。
- **禁 git 遵守**：无 git add/commit。build-weft/ 本地重建（未改 build 配置）。
- **board 卫生**：pin core8（idlest·freq 2.6GHz）· co-tenant vLLM core0,1 全程未扰 · loadavg 2.18→2.81（co-tenant drift·非我方）· scratch `/tmp/g8s6_attack` 抽取后清（opponent .o `/tmp/g8s3/rvv/*` 为 §五/§六 资产·保留）。
