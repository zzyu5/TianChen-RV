# P1 — 可补测 7 板格 · 预注册判读（PREREG·测前写死·事后找补=违例）

> **阶段**：收尾大战役 · P1 · **本文件 = 预注册，零板上测量**（仅只读 ls / nm / objdump 探针已执行·结果见 §2）。
> **目标**：pending 24 → 17（7 格 `cold=(空)` 补齐）。
> **世界**：**单世界 clang-18（PR-17）**。gcc 视为不存在——本文件与后续产出**零 gcc 字样**于任何输出/表/注记；§7 仅为「**禁继承的历史锚**」登记（标注其为已废域，不得作本轮结果或对照头条）。
> **赛道**：kernel-axis MICRO。**NOT e2e · NOT perf-covered 9/83 · 不入系统账**。[NG-4]。
> **硬冻结禁碰**：分母 102/105 · certified 101/108 · perf-covered 9/83 · roster $meta · 队序 —— **只登记不执行**。
> **禁 git commit/add**（主会话统一入库）。**禁写 `/home/kingdom/phdworks/papers/`**。
> 写于 2026-07-16（测前）。

---

## 0. 七格总表（既定 · 逐格判读 · 档位 · 部署成色）

| # | 板 | op | format | regime | 既定对手符号（T3·CROSSOP=跨算子） | 我方路 = 部署真路？ | 期望档位（先验·**不作结论**） |
|---|---|---|---|---|---|---|---|
| 1 | rvv | dequantize_row | iq2_xs | — | `dequantize_row_iq2_xs` | 是（deployed `to_float` 真派发路） | **便宜档**（对手 = generic-C 标量源 + clang 边缘 autovec rvv=5） |
| 2 | rvv | dequantize_row | iq2_s | — | `dequantize_row_iq2_s` | 是 | **便宜档**（对手 rvv=33·标量源） |
| 3 | rvv | dequantize_row | nvfp4 | — | `dequantize_row_nvfp4` | 是 | **便宜档**（对手 rvv=6·标量源） |
| 4 | rvv | gemm_tile | iq4_nl | **decode** | `ggml_vec_dot_iq4_nl_q8_0_vl128` **(CROSSOP)** | **是**（selector `repack-kept-q4_0-vlen128-decode`） | **硬档**（对手 = 手调 VLEN128 专化核） |
| 5 | rvv | gemm_tile | iq4_nl | prefill | `ggml_vec_dot_iq4_nl_q8_0_vl128` **(CROSSOP)** | **是**（selector `repack-kept-q4_0-prefill`） | **硬档** |
| 6 | k1 | gemm_tile | iq4_nl | **decode** | `ggml_vec_dot_iq4_nl_q8_0_vl256` **(CROSSOP)** | **否 = ★what-if**（selector `block-dot-decline-vlen256-decode-measured-negative`） | **硬档**·what-if |
| 7 | k1 | gemm_tile | iq4_nl | prefill | `ggml_vec_dot_iq4_nl_q8_0_vl256` **(CROSSOP)** | **是**（prefill 摊销 → selectRepack） | **硬档** |

**期望档位 = 先验，非预判结论**（[CLAUDE.md 规则2] 直觉投影不可信）。落地照 §4 判据走。

---

## 1. [GOV-8] 既有 harness 实况（**已核 · 别重建**）

### 1.1 可**直接复用**（改参数即可 · 不重写驱动）

| 工件（绝对路径） | 用于 | 复用度 |
|---|---|:--:|
| `/home/kingdom/phdworks/TianchenRV/experiments/active/g8-stage3-attack/A2-batch8-k1-dequant-raw/dequant_census_driver.c` | #1-3 驱动 | **100% 原样**（224MiB flush 内建·ZERO-MODEL 门·paired cold·argv `<fmt> <K> <hot> <reps> <seed> [verify]`） |
| `.../A2-batch8-k1-dequant-raw/kernels_dequant/{iq2_xs,iq2_s,nvfp4}.dq.c` | #1-3 我方核 | **100% 原样**（front-door 产物·`GEN_SEAL.txt` md5+recipe+HEAD=27fdc898·非 stale-archive） |
| `.../A2-batch8-k1-dequant-raw/run_k1_dequant_census.sh` | #1-3 跑法 | **~85%**（改 §1.2 五处即 rvv 版） |
| `.../A2-batch4-gemm-decode-M1-raw/flat_gevm_m1_driver.cpp` | #4/#6 M=1 协议模板 | **协议 100% 复用**（ONE GEVM call over nc·plain 单 q8 向量 activation·ZERO-MODEL per-column oracle·ratio=opp/ours） |
| `.../k1-gevm-sweep/raw/iq4nl_driver.cpp` | #6 主驱动 | **~80%**（iq4_nl x16 pack/288B/oracle/M=1 GEVM 全在·缺 seed 参数+OPP-S） |
| `.../A2-batch4-gemm-decode-M1-raw/run_flat_gevm_{k1,rvv}.sh` | #4/#6 build+cold 骨架 | **~70%**（双板 clang-18 build seal + load-gate + md5 双证 可套） |
| `/home/kingdom/phdworks/TianchenRV/tools/e2e-harness/board/iq4nl_gemm_paired_driver.c` | #5/#7 prefill 驱动 | **~75%**（nr16 GEMM paired + 224MiB flush 在·符号前缀 `tcrv_` 待改 `weft_`） |

`experiments/active/g7-census/iqfp4-dequant-rvv/rvv_run.sh`：**结构可参（rvv 侧 load-gate 8-15 / 探针 / GGML 路径写法）**，但**其编译域已废（单世界 clang）**——**不得整脚本复用**；只取 core-pick 与 probe 片段。dequant 驱动一律走 batch8 的（更严：224MiB flush + 2-seed + N=24 + K=1048576）。

### 1.2 **必须改**（逐条 · 精确）

**A. #1-3（rvv dequant）— 由 `run_k1_dequant_census.sh` 派生 `run_rvv_dequant_p1.sh`：**
1. `DFMTS="iq2_xs iq2_s nvfp4"`（18→3）
2. `BOARD=rvv`·`RDIR=/tmp/g8_p1_dq_rvv`
3. `GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin`（**单世界 clang 对手·已验在位**·§2）
4. `CC=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang` + **三个 rvv 专属 flag**（A1 已解两 blocker，照抄）：`-fno-integrated-as`、`--gcc-toolchain=/opt/tcrv-toolchains/gcc-15.2.0`、link 加 `-L<gcc-toolchain>/lib -Wl,-rpath,<gcc-toolchain>/lib`（解 `libgcc_s`；lld 不搜该路径）
5. `MARCH=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs`（rvv 板 march·与 A1 clang-18 对手 build 同）
6. `CORES="8 9 10 11 12 13 14 15"`（**rvv 0,1 有 co-tenant vLLM·不碰**）
7. probe 段：opp 用 `objdump --disassemble=<sym>`（authoritative）+ whole-file awk **双法**，二者不一致 → 判 VOID-PROBE

**B. #4/#6（decode M=1 GEVM）— 由 `iq4nl_driver.cpp` 派生 `iq4nl_gevm_m1_p1.cpp`：**
1. **加 seed 参数**（现 `std::mt19937 rng(0x14E00D)` 写死 → 无法 2-seed）→ argv 追加 seed
2. **加 OPP-S 同算子对手**（§3 发现·`ggml_gemv_iq4_nl_16x1_q8_0`）+ 其布局一致性门（§3.3）
3. **rvv 侧移植**（#4）：现文件 k1-only（`iq4_nl_repack_mf2.c` = VLEN256 mf2 核）→ #4 须 weft-opt 现生 VLEN128 GEVM leaf（§1.3）+ ABI 对齐
4. reps：15 → **N=25**（batch4 标准）·2-seed·2-trial

**C. #5/#7（prefill nr16 GEMM）— 由 `iq4nl_gemm_paired_driver.c` 派生：**
1. extern 符号前缀 `tcrv_emitc_…` → **`weft_emitc_…`**（改名后·commit fc72fe53）
2. 加 OPP-S `ggml_gemm_iq4_nl_16x1_q8_0`
3. k1 侧（#7）：`#include <riscv_vector.h>` 路径 + VLEN256 leaf + `-fno-integrated-as`（k1 clang-18-Bianbu）
4. reps → N≥20·2-seed

### 1.3 我方核 leaf 导出（front-door 现生 · **禁 stale-archive**·全部已在树）

```
# #5 rvv prefill GEMM (VLEN128):
build-weft/bin/weft-opt test/Conversion/RVV/rvv-to-emitc-repack-gemm-iq4-nl-q8-0.mlir \
  --weft-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp
# #4 rvv decode GEVM (VLEN128):
build-weft/bin/weft-opt test/Conversion/RVV/rvv-to-emitc-repack-gemv-iq4-nl-q8-0.mlir \
  --weft-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp
# #6 k1 decode GEVM (VLEN256·what-if·force-construct): 复用 k1-gevm-sweep/kernels/iq4_nl_repack_mf2.c
#    （已存·须 GEN_SEAL 式 md5+recipe 复核；若不可复现 → 走 weft-opt 重生）
# #7 k1 prefill GEMM (VLEN256): 同 #5 输入 + VLEN256 march（prefill 与 VLEN 无 gate·selector 走 prefill 摊销支）
```
**Step-0 门**：每 leaf 须 ①导出成功 ②`objdump` 自探针（vsetvl/gather/spill 计数入 seal）③**ABI 与驱动声明逐参对齐**（batch4 前科：q4_0-rvv `(n,s,nc,vx,vy)` vs q4_0-k1 `(n,s,vx,vy,nc)` **参数序不同**）。ABI 不对齐 → 该格 **VOID-ABI**，如实报，**0 造数**。

---

## 2. 已执行的只读探针（测前 · 事实 · 非测量）

**在位性 + 身份（read-only，未改任何 .so）：**
- rvv 单世界 clang 对手 build **在位**：`/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin/`·`libggml-cpu.so` md5 `e85fceda47606a115c5fbb0021817cdc`·`libggml-base.so` md5 `d9c07980cd3d6796e3e9750af7778140`（== A1 记录·未漂移）
- k1 stock（出货即 clang-18-Bianbu）：`/data/k1build-stock/bin/`·`libggml-cpu.so` md5 `871169a0123139692177468b3c8578be`·`libggml-base.so` md5 `00267134a3e86cd4fb83e9a2cbf5185a`（== batch4/batch8 记录）

**#1-3 对手 objdump（`--disassemble=`·clang-18 域）：**

| 格 | opp 符号 | ins | rvv | gather | vset | 字面 class | **§〇.1 源归属 tier** |
|---|---|---:|---:|---:|---:|---|---|
| iq2_xs | `dequantize_row_iq2_xs` | 173 | **5** | 0 | 5 | TRUE-VEC(字面) | **标量类**（generic-C 标量源·无 riscv 专化实现·边缘 autovec） |
| iq2_s | `dequantize_row_iq2_s` | 271 | **33** | 0 | 33 | TRUE-VEC(字面) | **标量类**（同上） |
| nvfp4 | `dequantize_row_nvfp4` | 182 | **6** | 0 | 6 | TRUE-VEC(字面) | **标量类**（同上） |

**#4-7 对手 objdump（clang-18 域·双板）：**

| 符号 | 板 | ins | rvv | gather | vset | 判 |
|---|---|---:|---:|---:|---:|---|
| `ggml_vec_dot_iq4_nl_q8_0` | 双板 | 9 | 0 | 0 | 0 | **VLEN 派发 thunk** → 尾调 `_vl128`/`_vl256`（callees 机判确证） |
| `ggml_vec_dot_iq4_nl_q8_0_vl128` | rvv | 76 | 11 | **2** | 11 | **手调 VLEN128 专化 codebook-gather 核**（= T3 既定对手·真存在·local sym） |
| `ggml_vec_dot_iq4_nl_q8_0_vl256` | rvv(同库) | 78 | 6 | **4** | 6 | 手调 VLEN256 专化（k1 侧走此支） |
| **`ggml_gemv_iq4_nl_16x1_q8_0`** | **双板** | 291/293 | **51** | **32** | 51 | ★**同算子 as-shipped 手调 repack-GEVM 对手 存在**（§3） |
| **`ggml_gemm_iq4_nl_16x1_q8_0`** | **双板** | 225/235 | **12** | **2** | 12 | ★同算子 repack-GEMM 存在·**剖面≈`_generic`(192/11/2) → 疑弱/近 generic**（§3.2 待源归属定档） |

→ **T3 既定符号名 `…_vl128` / `…_vl256` 正确**（是派发 thunk 的真被调 local 符号·`nm -D` 不列但 objdump 可定位）。**探针命令定为 `objdump --disassemble=ggml_vec_dot_iq4_nl_q8_0_vl{128,256}` + 派发 thunk callee 机判两证。**

---

## 3. ★CROSSOP 口径（#4-7 · 测前写死 · 跨算子对拼禁当同算子硬赢）

### 3.1 定义与 as-shipped 事实
- **CROSSOP** = 我方 `gemm_tile`（repack-GEM{V,M}，x16 交织权重，一次调用跨 nc 列）**vs** 对手 `ggml_vec_dot_iq4_nl_q8_0_vl{128,256}`（per-column 逐列 block-dot，nrc=1）。**两侧算法/数据布局不同 → 非同算子。**
- ★**本轮探针新发现（必须入账·否则= 挑软对手）**：该板**同算子 as-shipped 对手真实存在** —— `ggml_gemv_iq4_nl_16x1_q8_0`（decode·rvv=51/gather=32·双板真 RVV 手写）与 `ggml_gemm_iq4_nl_16x1_q8_0`（prefill）。**只测 CROSSOP 而不测同算子 = 选便宜对手**。

### 3.2 双对手设计（**两个都测·都报**）
| 代号 | 对手 | 算子关系 | 账 |
|---|---|---|---|
| **OPP-X**（既定） | `ggml_vec_dot_iq4_nl_q8_0_vl{128,256}` | **CROSSOP** | **填 T3 既定格**（本轮任务书指定）·必带 CROSSOP 标 |
| **OPP-S**（新·同算子） | `ggml_gemv_iq4_nl_16x1_q8_0`(decode) / `ggml_gemm_iq4_nl_16x1_q8_0`(prefill) | **同算子** | **强制诚实脚注**·**成色权威**（同算子才谈得上硬赢） |

**成色写死（不可事后改）：**
1. **CROSSOP PASS ≠ 硬赢**。OPP-X 任何 ≥0.8 结果，措辞恒带「**跨算子对拼（我方 repack-GEM{V,M} vs 对手 per-column block-dot）·非同算子硬赢**」。**禁**出现"beat/硬赢/胜过手调"于 OPP-X 单独语境。
2. **硬赢资格只由 OPP-S 授予**，且须同时满足：① OPP-S 剖面 = 真手调 RVV（rvv≫0·非 `_generic` 剖面）② 布局一致性门过（§3.3）③ cold ratio ≥0.8。任一不满足 → **禁称硬赢**。
3. **OPP-S 若判近-generic**（`ggml_gemm_iq4_nl_16x1_q8_0` rvv=12 ≈ `_generic` 192/11/2 = 高度存疑）→ 标 **便宜档 / opp-immaturity**，其 PASS **亦禁称硬赢**；定档键 = **源归属**（`ggml/src/ggml-cpu/arch/riscv/repack.cpp` 有真 riscv 专化实现 ⇒ 手调；仅 `ggml-cpu/repack.cpp` generic ⇒ 便宜档），**非 ins 计数**（batch8 已订正：ins 绝对值是方法学 artifact，不作判据）。
4. **OPP-X 与 OPP-S verdict 分歧**（一 PASS 一 X）→ **不挑好看的**：两者并列写死 + 登记裁决项（口径归属属 canon 级 → **必问**）。

### 3.3 OPP-S 布局一致性门（**过不了就 VOID-S·不降格凑数**）
OPP-S 吃 ggml 自己的 x16 交织布局（`block_iq4_nlx16` 288B）。我方 `make_x16` 与 ggml `tensor_traits<block_iq4_nl,1,16>::repack` **须 byte 级一致**，否则数字无效。
- **门**：以 ggml 自身 `repack` 产出的权重喂 OPP-S（首选）；不可得时，`make_x16` 产物 vs ggml repack 产物 **memcmp 全等**。
- **不等 → 该格 OPP-S = VOID-S**（如实报"同算子对手布局未对齐·未测"）·**只报 OPP-X + 标 CROSSOP**·**禁**用布局不符的 OPP-S 数字。

### 3.4 派发现实（honest scope）
OPP-S 是否被 ggml 运行时真选（repack buffer-type 是否对 iq4_nl 生效）**须机判登记**（`tensor_traits<block_iq4_nl,1,16>` weak sym 已在 → 已接线）。若探针无法确证运行时派发 → 标 **"as-shipped 符号·真 RVV 手写·运行时派发未确证"**，**禁**升格为"部署对手"。

---

## 4. ★逐格判读规则（**测前写死** · 落地照判 · 事后找补=违例）

### 4.0 通用（七格共用）
- **cold 唯一**：ratio_cold = `opp_med / ours_med`（>1 = 我方快）。**热 micro 一律不入账、不进措辞**。
- **样本**：**N≥10 中位数 + bootstrap CI**（实取 dequant N=24 / gemm N=25）·**2-seed**·**同会话 A/B 交替**（驱动内 within-proc paired·每计时区前独立 flush）。
- **正确门**：byte-exact **ZERO-MODEL**（从 plain 输入零复用重算全算术项）。**mism>0 → 该格 VOID-CORRECTNESS**（禁报性能数）。
- **判据**：**两 seed 均 ≥0.8 → PASS**；**任一 seed <0.8 → 走完整环 → 具名-X**（见 4.4）。**跨 0.8 分裂**（一 seed ≥0.8 一 <0.8）→ **具名-X（保守）** + 标 seed-split。
- **0.8 硬门应用**：本轮 **只登记不执行**——分母 102/105 = 硬冻结（**禁碰**）。逐格产出 = `verdict[0.8] ∈ {PASS, 具名-X, VOID-*}` + 账归属建议，**入表由主会话机算**。

### 4.1 #1-3（rvv dequant · [DEQ-AXIS]）
- **我方**：`kernels_dequant/<fmt>.dq.c`（front-door·GEN_SEAL 复核 md5）。**对手**：`dequantize_row_<fmt>` @ `build-clang18-rv64gcv/bin/libggml-base.so`（deployed `to_float` 真派发路·**非稻草人**）。
- **判读**：`ratio_cold ≥0.8 双 seed` → **PASS**；`<0.8` → **具名-X + 墙**。
- **账**：**[DEQ-AXIS] 独立子账 · test-only-not-in-denom**（默认维持）。
- ★**成色写死（不可翻）**：三格对手 = **标量类**（§〇.1 源归属：generic-C 标量源，无 riscv 专化实现；clang 边缘 autovec rvv=5/33/6 **不改 tier**）→ **便宜档**。**任何倍数（含大倍数）禁称硬赢**，必标 `compiler-artifact / opp-immaturity`。
- ★**auto-promote 张力 = 登记不执行**：DEQ decree 二触发键 = 「对手真向量 rvv>0」，字面上三格 clang-18 域 rvv>0（5/33/6）→ 字面满足 auto-promote 进 rvv 0.8 硬门；但 §〇.1 源归属判 **标量类**（T3 现有注记即如此写）。**两条规则冲突 = canon 级口径 → 必问**。本轮**只登记该张力 + 双读数**，**不动 102/105**。
- **VOID**：build/link fail · load-gate idle<70% · mism>0 · base.so md5 before≠after · relIQR 破 §5 门 · 双法探针不一致。

### 4.2 #4/#6（decode · **M=1 禁继承**）
- **★独立 M=1 GEVM 实测（写死测法）**：
  - 形状 = **M=1**：activation = **PLAIN 单 q8_0 向量**（`activation_block_stride=34`·**无行交织**）；weight = x16 交织（`block_iq4_nlx16` 288B）；**ONE GEVM call over nc 列**。
  - **禁**：继承 prefill `nr16`/`nr≥4` 数 · 继承 vec_dot M=1 数 · 继承 §7 任何历史锚 · 由 prefill 数推 decode（**两起串行 bug 前科**）。
  - K=2048 · nc=512 · N=25 · 2-seed · 2-trial · 32MiB+ flush（**k1 无 L3·64× L2 = 真 DRAM cold**；rvv 224MiB > L3）。
  - **对手同 M=1**：OPP-X = per-column `ggml_vec_dot_iq4_nl_q8_0(…, nrc=1)` over 同 plain blocks；OPP-S = `ggml_gemv_iq4_nl_16x1_q8_0`（天然 M=1 GEVM）。
- **#4（rvv·部署真路）**：selector = `repack-kept-q4_0-vlen128-decode` → 我方 repack-GEVM = **真部署 decode 路**（genuine·非 what-if）。
- **#6（k1·★what-if·禁当部署）**：selector = `block-dot-decline-vlen256-decode-measured-negative` → **出货走 block-dot**，我方 repack-GEVM = **force-constructed what-if**。措辞**必带 what-if**；**禁**写成"k1 decode 部署赢/输"。
  - ★**循环论证防线**：该 DECLINE 由 registry 中 `iq4_nl … 0.248x` 驱动（源 = `k1-gevm-sweep/raw/iq4nl_run.log`·N=15·单 seed）。本轮 = **对该事实的独立复测**：**禁继承 0.2487**，须独立重测。
  - **预注册后续动作**：复测若 **≥0.8**（即与 registry 的 Negative 相悖）→ 这是 **selector 事实证伪**，属 canon/红线级（改 registry = 改既有条文）→ **登记 + 必问**，**本 agent 不改 selector、不改 registry**。
- **判读**：OPP-X ≥0.8 双 seed → **PASS（CROSSOP·非同算子硬赢）**；<0.8 → **具名-X + 墙**。OPP-S 同判据独立出一个 verdict（成色权威）。

### 4.3 #5/#7（prefill）
- 形状 = **nr=16 GEMM**（nr%4==0·nc%16==0·K=2048·nc=512）·N≥20·2-seed·cold。**禁**用 decode 数充数、**禁**用 prefill 数回填 decode。
- **#5/#7 均 = 部署真路**（selector prefill 摊销支 → selectRepack）→ 可称 deployed（**但仍受 CROSSOP 成色约束**）。
- 判读同 4.0/4.2。

### 4.4 具名-X 的"完整环"（<0.8 时**必须走完**才准记 X·懒认输=违例）
1. **对手身份确证**：as-shipped 三证（符号名 + objdump 剖面 + **源码归属**）→ 排除稻草人/我方误链。
2. **我方核自探针**：`objdump` vsetvl / gather / spill / size；查 spill 病、vsetvli storm、fp16 libcall。
3. **墙具名**（必须落到机制名，非"慢"）：候选 = `codebook-gather-bound`（iq4_nl 先验墙·gather=32/64）/ `narrow-vl` / `super-block fold@M=1 不摊销` / `regfile-spill` / `memory-bound near-parity`。
4. **可修性判**：可修（emitter 成熟度缺口·入队列）vs 不可修（uarch/格式结构）。
5. **登记**：具名-X + 墙 + 可修性 + provenance 指针。**禁**只写 X 不给墙。

### 4.5 VOID 条件（**任一触发 → 该格 void·如实报·0 样本永不造数**）
`VOID-BUILD`（编译/链接 fail）·`VOID-ABI`（leaf ABI 与驱动不对齐）·`VOID-CORRECTNESS`（ZERO-MODEL mism>0）·`VOID-PROBE`（对手身份三证不齐 / 双法探针不一致 / 非 as-shipped）·`VOID-S`（OPP-S 布局未对齐·§3.3）·`VOID-LOAD`（load-gate idle<70%）·`VOID-NOISE`（§5 门破）·`VOID-HYGIENE`（stock .so md5 before≠after · STRAY>0）·`VOID-EXPORT`（front-door leaf 导不出）。
**void ≠ 失败 = 诚实产出**。**宁可无数，不可编数。禁贴伪造的命令输出。**

---

## 5. 噪声自检门（测前 · 测中 · 写死）

**历史地板（本基准类·同板同协议·仅取 relIQR 形状，不取其数值结论）：**
- dequant-streaming 类（224MiB flush·K=1048576）：ours relIQR 历史 ≈ **0.27–2.54%**（iq2_xs 2.54 / iq2_s 0.27 / nvfp4 1.40）；opp ≈ 0.41–1.08%。
- GEVM/GEMM 类（K=2048·nc=512）：ours relIQR ≈ **0.15–0.5%**；opp ≈ 0.1–7.2%（cold-outlier 易发）。

**测前（3 次重测-重测 sanity·同 config 连跑 3 轮）：**
| 类 | 门（**开测条件**） |
|---|---|
| dequant | 每轮 ours relIQR ≤ **max(1.5×历史地板, 3%)** → iq2_xs ≤3.8% · iq2_s ≤3% · nvfp4 ≤3%；opp relIQR ≤3% |
| GEVM/GEMM | ours relIQR ≤ **5%**（=1.5×历史地板取类下限并设 5% 类地板）；opp relIQR ≤ **8%**（cold-outlier 容差） |
| 共用 | 3 轮 **ratio 中位数两两相对极差 ≤2%** |
门不过 → **不开测**（先 load-gate 换核 / 查 co-tenant / kill 竞争进程）；两次尝试仍不过 → **VOID-NOISE 如实报**。

**测中：** 任一格 relIQR ≥ **3× 该格测前 sanity 实测值** → **整会话作废重跑**（不摘录、不挑轮次、不"取好的那 seed"）。

**单实例（写死）**：每板同时**仅一个 bench 进程**；测前 `pgrep -f '<bench名>'` 清零 + kill 竞争；`taskset` 单核钉（rvv 8-15 load-pick·**0,1 co-tenant vLLM 不碰**；k1 idle-pick 0-7）；测后 `STRAY` 计数入 log。P1 **不动任何 LIVE 部署 .so**（纯 kernel-axis micro·只读 stock/clang18 build），**.so 竞争污染前科不适用但纪律照守**：stock md5 before==after 双证。

---

## 6. ★措辞预注册（**留数字空位** · 落地填数即用 · 措辞不再构成回门理由）
> 空位 `{{…}}`。模板空缺处标「待裁」→ **只问空缺、不问整体去向**（权限卡 ⑦⑧）。

### 6.1 PASS 模板 — dequant（#1-3·便宜档）
> **`dequantize_row/{{fmt}}@rvv` = PASS[0.8]**（cold `{{ratio_med}}×`·seed1 `{{r_s1}}` / seed2 `{{r_s2}}`·N=24 中位·bootstrap CI `[{{ci_lo}}, {{ci_hi}}]`·relIQR ours `{{o_iqr}}%` / opp `{{p_iqr}}%`）。byte-exact ZERO-MODEL `{{mism}}`/`{{tot}}` 0mism/0ULP。对手 = as-shipped `dequantize_row_{{fmt}}`（`libggml-base.so` md5 `d9c07980`·deployed `to_float` 真派发路·objdump ins=`{{ins}}` rvv=`{{rvv}}` gather=`{{gat}}`·源归属 = generic-C 标量源）。
> **成色（诚实·铁线）**：对手 = **标量类**（§〇.1 源归属·clang 边缘 autovec 不改 tier）→ **便宜档 · opp-immaturity · 禁称硬赢 · 0 verified hand-brick**。**账 = [DEQ-AXIS] 独立子账·test-only-not-in-denom·NOT e2e·NOT perf-covered·不入 matmul headline**。auto-promote 字面张力（opp rvv=`{{rvv}}`>0 vs §〇.1 标量类）= **登记待裁·未动 102/105 分母**。

### 6.2 PASS 模板 — gemm_tile iq4_nl（#4-7·**CROSSOP**）
> **`gemm_tile/iq4_nl@{{board}}` `{{regime}}` = PASS[0.8]**（cold `{{ratio_med}}×` vs **OPP-X**·seed1 `{{r_s1}}` / seed2 `{{r_s2}}`·N=25 中位·bootstrap CI `[{{ci_lo}}, {{ci_hi}}]`·relIQR `{{o_iqr}}%`/`{{p_iqr}}%`）。ZERO-MODEL `{{mism}}`/`{{tot}}`。
> ★**成色 = 跨算子对拼（CROSSOP）·非同算子硬赢**：我方 `repack-GEM{{V|M}}`（x16 交织·一次调用跨 nc 列） vs 对手 `ggml_vec_dot_iq4_nl_q8_0_vl{{128|256}}`（per-column block-dot·nrc=1）——**两侧算法不同**，此数**禁**读作"beat 手调 vec_dot"。
> **同算子对照（OPP-S·成色权威）**：vs as-shipped `ggml_gem{{v|m}}_iq4_nl_16x1_q8_0` = `{{ratio_S}}×` → `{{PASS|具名-X|VOID-S}}`·对手剖面 rvv=`{{rvv_S}}` gather=`{{gat_S}}`·源归属 `{{手调-riscv专化|近-generic便宜档}}`。**硬赢资格 = `{{授予|不授予}}`**（授予须三条同时满足：OPP-S 真手调 ∧ 布局门过 ∧ ≥0.8）。
> **部署成色**：`{{deployed(selector=repack-kept-…)|★what-if(selector=block-dot-decline-vlen256-decode-measured-negative·出货走 block-dot)}}`。
> **regime**：`{{decode = 独立 M=1 GEVM 实测·禁继承 prefill nr16 / vec_dot M=1 / 历史锚|prefill = nr16 GEMM}}`。**账 = matmul kernel-sym·NOT e2e·NOT perf-covered·[NG-4]**。

### 6.3 具名-X 模板（通用·**须走完 §4.4 完整环**）
> **`{{op}}/{{fmt}}@{{board}}` `{{regime}}` = 具名-X[0.8]**（cold `{{ratio_med}}×`·seed1 `{{r_s1}}` / seed2 `{{r_s2}}`·N=`{{N}}`·CI `[{{ci_lo}}, {{ci_hi}}]`）。ZERO-MODEL `{{mism}}`/`{{tot}}` = **正确核**（输的是性能不是正确性）。
> **对手身份三证**：符号 `{{sym}}` @`{{addr}}` · objdump ins=`{{ins}}` rvv=`{{rvv}}` gather=`{{gat}}` vset=`{{vset}}` · 源归属 `{{file:line}}` → **as-shipped `{{手调 VLEN 专化|近-generic|标量类}}`**（非稻草人）。
> **我方核自探针**：vsetvl=`{{vs}}` gather=`{{g}}` spill=`{{spill}}` size=`{{sz}}`B。
> ★**墙（具名·机制级）**：`{{codebook-gather-bound | narrow-vl | super-block fold@M=1 不摊销 | regfile-spill | memory-bound near-parity}}` —— `{{一句话机制}}`。
> **可修性**：`{{可修(emitter 成熟度缺口·入队列 → {{队列项}}) | 不可修(uarch/格式结构·C3′ 负结果·档案级教材)}}`。
> **provenance**：`{{raw 路径}}`。**禁外推**（本格 X 不推同族/同板/他 regime）。

### 6.4 VOID 模板
> **`{{op}}/{{fmt}}@{{board}}` `{{regime}}` = VOID-`{{BUILD|ABI|CORRECTNESS|PROBE|S|LOAD|NOISE|HYGIENE|EXPORT}}`**·**未测·0 样本·不造数**。
> **精确缺口**：`{{一句话·可执行}}`。**已就绪**：`{{已有工件}}`。**剩余构造**：`{{具体步骤}}`。**板**：`{{rvv|k1}}`。
> T3 该格 **维持 pending·禁填**（pending 24 → `{{17+void数}}`）。**预判不作结论。**

### 6.5 净结论骨架（收口用）
> P1 七格：**PASS `{{p}}` / 具名-X `{{x}}` / VOID `{{v}}`**（pending 24 → `{{24-p-x}}`）。**0 verified hand-brick** `{{或：hand-brick win {{n}} 格 = {{格名}}·经 OPP-S 同算子门授予}}`。**便宜档 `{{n}}` 格禁称硬赢**。**CROSSOP `{{n}}` 格已标跨算子·非同算子硬赢**。**k1 decode = what-if（出货 block-dot）**。账：`{{n}}` 格 [DEQ-AXIS] test-only-not-in-denom + `{{n}}` 格 matmul kernel-sym。**NOT e2e · NOT perf-covered 9/83 · 硬冻结未碰（102/105 · 101/108 · 9/83 · roster $meta · 队序 = 只登记不执行）**。**零 gcc 输出。**

---

## 7. ★禁继承的历史锚（登记 · **不得作本轮结果、不得作对照头条**）

| 锚 | 值 | 域 | 为何禁继承 |
|---|---|---|---|
| T3 注记 `measured anchor 0.217×`（#4/#5 rvv iq4_nl） | 0.217× | **已废域** e2e prefill（`g5-wiring/M2-iq4_nl`） | ① **非本轮结果** ② 已废编译域 ③ e2e 账 ≠ kernel-sym 账 |
| g7 kernel-sym cold 0.228(nr64)/0.205(nr16) | — | 已废域·kernel-sym | 同上·域不成立于单世界 clang |
| l1-m2-iq4 0.837× | — | **不对称** artifact（我方 clang vs 对手非 clang） | [CASE-COMPILER-ASYMMETRY]·两侧不对称 = 无效 |
| k1 `iq4nl_run.log` ratio_repack **0.2487** / ratio_blockdot 0.6934 | — | k1 clang（域成立）但 **N=15·单 seed·STRAY=2** | **#6 禁继承**（且它是 selector DECLINE 的驱动事实 → 本轮须独立复测·§4.2 循环论证防线） |
| rvv dequant census iq2_xs 2.95 / iq2_s 3.10 / nvfp4 0.66 | — | **已废域** | #1-3 禁继承·须单世界 clang 重测 |
| T3 `vec_dot/iq4_nl` rvv PASS 1.15 / k1 具名-X 0.697 | — | 不同 op 行 | **禁**用 vec_dot 行数字回填 gemm_tile 行（不同核形态·判别键=核 body） |

★**发现（登记·主会话机算入表·本 agent 不动 T3）**：T3 `vec_dot/iq4_nl` 行 tier=**手调**，而 `gemm_tile/iq4_nl` 行（87/88）tier=**通用向量**，**但两行 opponent 符号相同**（`ggml_vec_dot_iq4_nl_q8_0_vl{128,256}`）。本轮 objdump（§2）判该对手 = **手调 VLEN 专化核** → **87/88 行 tier 存疑（疑应为 手调）**。**登记待主会话/用户裁**（改 tier 触碰 roster/口径 → 不自决）。

---

## 8. 卫生 / 纪律 checklist（每格逐条·入 seal）
- [ ] stock/clang18 `.so` **只读**·md5 before==after 双证（rvv `e85fceda`/`d9c07980`·k1 `871169a0`/`00267134`）
- [ ] 单实例·测前 kill 竞争·`STRAY`=0 入 log·`taskset` 单核钉（rvv 不碰 core 0,1 co-tenant）
- [ ] load-gate idle≥70%·gov/freq 入 log·loadavg begin/end
- [ ] **无 git add/commit**·主树/build/governor 未改
- [ ] 输出**零 gcc 字样**（§7 仅登记"已废域"·不点名编译器）
- [ ] 硬冻结未碰（102/105·101/108·9/83·roster $meta·队序）
- [ ] raw A/B 全量落 `P1-backfill7-raw/`（logs + build_seal + objdump 探针 + 驱动/核源）
- [ ] **禁写 `/home/kingdom/phdworks/papers/`**

## durable files（本阶段产出）
- `/home/kingdom/phdworks/TianchenRV/experiments/active/g8-stage3-attack/P1-backfill7-prereg.md`（本文·**测前**预注册·测后不改判据）
