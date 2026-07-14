# G5（接线战役）M0 — 接线机制解剖一页 + 通用接线方案 + M1(q8_0) 预案

> **campaign**: G5 接线战役（perf-covered 唯一拉绿杠杆 — 把 emitted tcrv kernel 接入 ggml 真实
> forward；`[GAP-FLAT-E2E-ROUTING]` + `[GAP-IME-E2E-INTEGRATION]` 统一解）
> **role / M0**: 纯 scout/docs — 解剖 q4_0 唯一成熟先例 + 推广通用模板 + 注册 M1(q8_0) 预案。**零 code、零 git、HEAD 保持 `aeed7e0b`。**
> **接线定义（边界锁）**: 接线 = **补丁 / 链接层集成**（in-tree `#include` 重编译，可逆备份还原；已有 q4_0/q4_K 先例）。
> **NG-2 不动**：不做图框架 / 不改 ggml 图 schedule；只在既有 `mul_mat`→`get_tensor_traits`→`gemv/gemm` 派发链上翻 gate + 挂 emitted kernel。
> **[NG-4] / 诚实**：接线 ≠ 自动转绿。micro↛e2e 铁律仍管辖。M1 目标 = 让 q8_0 拿到**真实 e2e 分相判决**，绿格数以实测为准。
> **来源（全只读核对）**: memory `q4-0-e2e-is-routing-not-kernel` / `kernel-wins-dont-transplant-to-e2e` /
> `repack-campaign-terrain`；T6 casefile `experiments/active/t6-rvv-flat-q4k/`（MANIFEST + transmission_accounting.csv + routing_gap_probe.txt）
> + `experiments/active/t6-k1-ime-q4k-e2e/MANIFEST.md`；归因档 `docs/reports/2026-07-09-q4_0-5.9x-routing-attribution-correction.md`；
> seal-fix 档 `docs/reports/2026-07-10-t4b-e2e-seal-integration-proven-kernel-variant-residual.md`；
> 部署机制 `tools/e2e-harness/board/t4b-seal-fix/{deploy_patch.py,board_deploy.sh}` + `fullmarch_rebuild.sh`；
> 选择器 `lib/Plugin/RVV/RVVContractionPathSelection.cpp`；派发结构核对于 sibling 上游树 `/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/repack.cpp`。
> **板 A-tree** = `ssh rvv:/home/ubuntu/tcrv-llamacpp/ggml`（HEAD `f3e1828`），行号以板 A-tree 为准。

---

## 甲、q4_0 现有接线解剖（唯一成熟先例 · rvv VLEN128 · prefill 5.755× / decode 1.910× T6 证）

### 挂点全景 — 两个物理挂点（缺一不出 tcrv kernel）

ggml 的量化 `mul_mat` 在 CPU 后端有两条正门；接线只碰其一（repack 门）：

```
forward_mul_mat(weight tensor cur, ...)
  └─ get_tensor_traits(cur)            ← 【挂点①：路由 gate】GEN repack.cpp:4589+ 的 switch(vlenb*8)
       ├─ 返回 &q4_0_16x1_q8_0 (非空)  ⇒ 走 repack 路径 → tensor_traits::gemv/gemm
       │                                   → ggml_gemv/gemm_q4_0_16x1_q8_0()
       │                                        └─【挂点②：kernel 介入】ARCH arch/riscv/repack.cpp
       │                                             if(vlenb*8==128){ banner; tcrv_emitc_...(); return; }
       └─ 返回 nullptr (break;//TODO)  ⇒ 回落 block-dot vec_dot（stock 缺省路径）
```

**关键结构事实**：整条 q4_0 repack 流水线（repack 布局类型 `block_q4_0x16`、在线 repack 构造器 `make_block_q4_0x16`、
q8_0 激活量化器、generic `ggml_gemv/gemm_q4_0_16x1_q8_0_generic`、riscv 向量 kernel body、trait 注册、VLEN switch）
**全部由上游 `f3e1828` 建好，且已在 VLEN256 路由**；上游只把 VLEN128/512/1024 三格留成 `break;//TODO`（gate OFF）。
我方接线在 dispatch 侧的**净改动 ≈ 翻一行 gate**。→ **canon 措辞锁**（`q4-0-e2e-is-routing-not-kernel`）：q4_0 5.9×/5.76× 的绝大部分 = **routing/dispatch 白嫖**（路由到上游已建 repack 路径 + decode 内存局部性），**非 kernel 质量赢**；prefill kernel A(ours)==B(stock) **字节相同**。任何引用必须披露此点。

### ① dispatch 挂点 — GEN `repack.cpp:4592`（`TCRV-WINB-ON-TOGGLE`）
- 函数 = `ggml::cpu::repack::get_tensor_traits`（板 A-tree `repack.cpp:4589-4597` q4_0 分支）。
- 上游缺省：`case 128: { break; } // TODO` → 返回 nullptr → block-dot。
- 我方翻转（板 A-tree working-tree，`git blame = Not Committed Yet`）：
  ```cpp
  case 128:  { if (cur->ne[1] % 16 == 0) { return &q4_0_16x1_q8_0; } break; } /* TCRV-WINB-ON-TOGGLE */
  ```
  即返回上游**同一个** `q4_0_16x1_q8_0` trait（`repack.cpp:4566` 注册，`tensor_traits<block_q4_0,1,16,GGML_TYPE_Q8_0>`）。
  条件 `ne[1]%16==0` = 16-way 交织的行数约束；不满足则回落 block-dot（部分 tensor 仍走 stock，是稀释来源之一）。
- **这一步就是 route**：把 q4_0 权重的 `mul_mat` 从 block-dot 切到 repack-GEMM 布局 + q8_0 激活量化 + tiled kernel。

### ② 符号导出 — emitted `.inc` `#include` 进 ARCH + 符号命名
- 我方 emitted C（`mlir-translate`/`tcrv-translate` 产出）以 `#include "…​.inc"` 注入 `arch/riscv/repack.cpp`（板 A-tree `arch/riscv/repack.cpp:21-23` 增 include 行）。**不是** LD_PRELOAD、**不是**独立 .so；是 in-tree 源码 `#include` + 重编译 → 符号进 `libggml-cpu.so`。
- **符号命名规律**（对照 q4_K seal-fix `deploy_patch.py`）：`tcrv_emitc_<funcname>_kernel_<funcname>`，其中 funcname = `ggml_repack_gemv_q4_0_q8_0` / `ggml_repack_gemm_q4_0_q8_0`。
  - q4_0 部署符号 = `tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel…`（GEVM，decode）+ `…gemm…`（GEMM，prefill）。
  - deployed .so `md5 75f20b5f`（T6 A-tree）**仅含 q4_0 tcrv 符号**（`nm -C … | grep tcrv_emitc` = q4_0 only；无 q8_0/q4_1/q5_x/K-quant）— 这是"仅 q4_0 wired"的 .so 级实证。

### ③ 选择器介入点 — ARCH kernel body 内 VLEN128 分支 + banner
- 板 A-tree `arch/riscv/repack.cpp:234-244`：在 `ggml_gemv/gemm_q4_0_16x1_q8_0()` 体内、`UNUSED(blocklen);` 之后插一段：
  ```cpp
  #if defined __riscv_v_intrinsic
      if (__riscv_vlenb() * 8 == 128) {
          /* 一次性 banner */ fprintf(stderr, "TCRV EMITTED GEMV(q4_0_16x1 VLEN128 compiler-emitted) ENGAGED …\n");
          tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel…(n, s, (const uint8_t*)vx, (const uint8_t*)vy, …);
          return;                      // ← 拦截：VLEN128 用我方 emitted kernel，不落到下方 upstream body
      }
  #endif
  ```
- **触发路径**：挂点① route 到 repack trait → trait 调 `ggml_gemv/gemm_q4_0_16x1_q8_0` → 此分支在 VLEN128 命中 → 打印 `TCRV EMITTED GEMV/GEMM(q4_0_16x1 VLEN128 compiler-emitted) ENGAGED` + 调 emitted kernel + `return`。
- banner = **engage 探针**（T6 `routing_gap_probe.txt` 实证：q4_0 model banner FIRES；q8_0/q4_1 model banner ABSENT = 未路由 = 回落 block-dot）。
- 注意两个 VLEN 的分工：**VLEN128 走我方 emitted kernel（挂点②/③）；VLEN256 走上游自带 kernel**（上游 case256 早已路由，无 tcrv 分支）。q4_0 prefill 之所以 A==B 字节相同 = 我方 emitted 构造件与上游路径**字节等价已证**（`be66c917 typed_repack_gemm_loop_body`；kernel-est 5.045× ≈ routing-era 5.077×）。

### ④ 构造 → .o → link 链路（既有 build 机制，可逆）
1. **emit**：tcrv 编译器 lower（`--tcrv-rvv-lower-quant-contraction --tcrv-rvv-lower-to-emitc`）→ EmitC → `tcrv-translate`/`mlir-translate` → C 源 `.inc`（e.g. q4_K 版 `md5 90d454da`，VLEN128 vl=8）。
2. **注入**：`.inc` 落板 scratch（`/tmp/…`）；`deploy_patch.py` 幂等改**恰 2 个 tracked 文件**（GEN `repack.cpp` 翻 gate + ARCH `arch/riscv/repack.cpp` 加 `#include` + VLEN128 分支）。带 `BASE_GEN/BASE_ARCH` md5 baseline 断言（非 baseline 即 ABORT），改前 `cp *.ORIG` 备份。
3. **build**：`cmake --build $BUILD --target ggml-cpu -j$(nproc)`（板 gcc-15.2.0，`source /opt/tcrv-toolchains/env.sh` 供 `-lgcc_s`）。**hot-TU march 陷阱**（`fullmarch_rebuild.sh` §WHY）：ggml `ggml-cpu/CMakeLists.txt` 对热 TU（quants.c/repack.cpp）**追加**第二个 `-march=${MARCH_STR}`，gcc 取**最后一个** → 单改 `CMAKE_C_FLAGS` 到不了热路径；须 `set(MARCH_STR …)` override 才让 repack.cpp 拿到全能力 march（对称改 A/B，唯一差异保持 = repack 补丁）。
4. **link**：产 `libggml-cpu.so.0.15.1` → `llama-bench` 动态链接。`nm -C` 验 tcrv 符号 ≥1、`strings` 验 banner ≥1、`objdump -d` 验 kernel vl 宽度（vl=8 seal，防 vl=16 变体误挂 — 见 seal-fix 教训）。
5. **可逆**：`board_restore.sh` 从 `.ORIG` 还原 + rebuild；**无 git stash/rm/mv/add/commit、无 emitter 源改动**。测前/测后 .so md5 比对证"零 stock 改动"。

### ⑤ 五验 / 正确性（接线后如何保 A==B）
- **preflight 4/4 fail-closed**（`run_e2e.sh`）：同编译器（A/B 都 gcc-15.2.0 Release） · march-complete（覆盖板关键扩展） · libcall-free（ggml-cpu.o 无 `__extendhfsf2` 类 fp16 softfloat libcall） · VLEN-fingerprint（board==target 128）。
- **byte-exact / bounded-ULP**：q4_0 prefill 八门① = **FMA_FOLD_BOUNDED_ULP**（repack GEMM 用向量 FMA，非 ULP0；硅 verdict 证 as-accurate-or-better than ggml scalar-vs-f64，2/4 shape 更近）。**措辞锁**：不得说"byte-exact vs ggml"。真正确性基座 = prefill kernel A==B **字节相同**（routing-era 对抗-verify）。
- **greedy-token A==B**：`correctness_gate.sh` 逐 prompt 贪心 token 一致 + logits sanity（无 NaN/Inf）。q4_0 = 3/3 GREEN。
- **PPL 对照**（K-quant 类用，q4_0 用 greedy 即可）：seal-fix q4_K PPL 11.97/12.008 ≈ stock 12.05 = 整条链路（route+repack+q8_K 激活+dispatch）经 ggml-generic 对照证正确。
- **objdump kernel seal**：确认部署的是**证过的变体**（VLEN128 vl=8），非 VLEN256 vl=16 变体误挂。★seal-fix 教训（`部署变体≠证过变体`）：M4 standalone 证过 vl=8 正确，但集成误部署 vl=16 → VLEN128 上 vl 钳到 8、只算半列 → PPL 822057 垃圾。接线**必须 objdump 验部署变体 == 证过变体**。

---

## 乙、通用接线方案（按格式类分 · 推广模板）

**判据（决定接线成本的唯一 gate）= 上游在该 VLEN 有无"现成可翻的 repack 全链路"**：
- **有现成链路（仅缺一行 return）** ⇒ 接线 = 翻 gate（+ 可选挂 emitted kernel）= **routing 白嫖可复用**（q4_0/q8_0）。
- **无现成链路（trait/kernel/量化胶水全缺）** ⇒ 接线 = 编译器从头全构造 + 全部胶水 = **净新建造、无白嫖**（K-quant）。

### 乙.1 FLAT 类（q4_1 / q5_0 / q5_1 / q8_0）— 直类比 q4_0

| 格 | VLEN128 dispatch 现状（板 A-tree）| 上游 case256 trait？ | 接线所需改动 | routing 白嫖？ |
|----|------|------|------|------|
| **q8_0** | `repack.cpp:4713 case128 {break;}//TODO`（DECLINE）| **有** `q8_0_16x1_q8_0`（:4569 注册, :4714 route）| **翻 :4713 gate**（`return &q8_0_16x1_q8_0`）+ 可选挂我方 emitted GEVM（`arch/riscv` 分支）| **是**（同 q4_0，upstream 全链路就绪）|
| q4_1 | **零 riscv repack 分支**（任何 VLEN，nullptr→block-dot）| **无** | 新建 trait+kernel+接线；**上游另有 q8_1×4 量化器 + int-zero-point ABI 阻塞**（repack-terrain 记 BLOCKED）| 否 |
| q5_0 | 零 riscv repack 分支 | 无 | 新建 trait+kernel+接线 | 否 |
| q5_1 | 零 riscv repack 分支 | 无 | 新建 trait+kernel+接线 | 否 |

- **q8_0 = FLAT 类唯一"翻一行即得"格**（与 q4_0 结构全等；见丙节 M1）。四格里 q8_0 是最低成本、最高价值曳光弹。
- **q4_1/q5_0/q5_1** 在上游连 riscv repack 分支都没有（`opponent_probe.md` 证 q4_0/q8_0 的 riscv repack ONLY at case256；q4_1/q5_0/q5_1 零分支）→ 接线 = **完整净新建造**（我方需构造 trait + repack 布局 + kernel + dispatch），不是白嫖；q4_1 另有上游 ABI 阻塞。
- **改动清单模板**（每格 3 处，同 q4_0/q4_K deploy_patch）：(a) GEN `repack.cpp` 对应格 `case128` 翻 `return &<fmt>_16x1_<act>`；(b) ARCH `arch/riscv/repack.cpp` 加 `#include "<fmt>.inc"`；(c) ARCH kernel body 加 `if(vlenb*8==128){banner; tcrv_emitc_…<fmt>…(); return;}`。符号名 = `tcrv_emitc_ggml_repack_gemv/gemm_<fmt>_<act>_kernel…`。

### 乙.2 IME 类（q4_0/q8_0/q4_K @ime）— GEMM 挂点差异（k1 板，比 FLAT 复杂）

- **现状（`t6-k1-ime-q4k-e2e` HEADLINE，高价值 negative）**：**tcrv IME GEMM 完全未接进任何 llama forward**。`tcrv.ime.q4_0/q8_0/q4_K_matmul_tile` + `tcrv_ime_vmadot_mac_kloop` 是 standalone 证过的构造件（host int32 oracle + k1 硅 `vmadot 0xe210312b` int32 0-diff，~2.09× compute-account），**仅**被 `test/Target/IME/*.c` + `test/Conversion/EmitC/ime-*.mlir` 消费。板 `~/tcrv-k1-llama/ggml/src` grep tcrv IME 符号 = **0 hit**。= 具名 `[GAP-IME-E2E-INTEGRATION]`。
- **挂点差异（为何比 FLAT repack 复杂）**：
  - FLAT/q4_0 走 **GEVM/GEMM repack trait**（`get_tensor_traits` 向量 repack 门，M=1 decode 也走 GEVM）。挂点 = 翻 gate。
  - IME 是**矩阵范式**（脉动阵列 `vmadot`），只在 **prefill mul_mat（M≥M*，多激活列）** 有物理收益；decode（M=1 GEVM）阵列**跑不起来**（vendor toggle 实证：decode control M=1 = 1.47×，纯 SpacemiT RVV kernel-family swap，非 IME 阵列）。
  - 故 IME 接线 = 需在 ggml `mul_mat` 的 **GEMM/prefill 分支**（M≥阈值）挂 tcrv IME matmul_tile，**并**建 tcrv-IME↔ggml forward 桥（block-layout 已知，桥不存在）— 比向量 repack 的"翻一行 gate"重（矩阵挂点 vs 向量 repack 挂点）。
- **诚实上限（micro↛e2e）**：即便接好，vendor 全接线 IME GEMM 在 memory-bound 1B 模型上 **decode floor = 1.47×（M=1 kernel-family，非阵列）**、IME-unit 增量 ~+0.18 与 GEMM-vs-GEVM 混淆**不可干净隔离** → tcrv IME ~2.09× compute-account **即便接线也大概率不 e2e 传导**（与 `kernel-wins-dont-transplant-to-e2e` 一致：IME 5.51× kernel → 0.86× decode）。IME 接线价值 = **机制/N2 结构闭环**，非 perf 绿格。

### 乙.3 K-quant 未消费路径 — scout 现状

- **VLEN128 全 `case128 {break;}//TODO`**（q4_K:4619 / q2_K:4636），q5_K/q6_K/q3_K **连 riscv repack 分支都无**（任何 VLEN）。→ **无任何可翻的 gate**：repack 布局构造器、q8_K 激活量化、kernel body、dispatch 接线**全缺**，须编译器**从头全构造**（= T4b full-construct 净新贡献；`q4_0 的免费 ~5× 对 K-quant 根本不存在`）。
- **q4_K 已达"接线骨架证正确"但非绿**（seal-fix `98717158`）：route+repack+q8_K 激活+dispatch 整链经 ggml-generic 对照证正确（PPL 11.97 vs stock 12.05、相干），但 **kernel-轴 perf 立不住**（S6 1.884× = clang-vs-gcc artifact，对称 gcc 0.272× 已撤回）→ **NOT perf-covered**。剩唯一未闭 = ④ e2e-perf（board-availability-gated）。
- **q4_K vl16（Win-K1-VLEN）** = micro/kernel-axis 构造件，**未部署进板 forward**（板唯一 wired tcrv kernel = RVV q4_0 repack；无 q4_K repack path），且**板无 q4_K gguf、无 llama-quantize** → e2e 不可跑 = `[GAP-KQUANT-E2E-INTEGRATION]`。
- **措辞锁**：不得声称 q4_K e2e Win/加速。可引 = "能力键控全模型路由 + 权重 repack + q8_K 激活 + dispatch 自建集成、经 ggml-generic 对照证正确、我方 VLEN128 发射 kernel 集成部署变体已修正"。

---

## 丙、M1 曳光弹 = q8_0（用户建议 · kernel 优势明确 4.10× 内核轴 · 主流争夺格）

### 丙.0 为何 q8_0 是最优曳光弹（确认）
1. **翻一行即得**：q8_0 与 q4_0 结构**全等** — 上游 `q8_0_16x1_q8_0` trait（`repack.cpp:4569`）+ generic kernel（gemv:1525 / gemm:2580）+ arch/riscv kernel body + **case256 已 route**（:4714）；VLEN128 = `case128 {break;}//TODO`（:4713）。→ **routing 白嫖对 q8_0 成立**（与 q4_0 同、与 K-quant 相反）。
2. **micro 4.10× 已 clean 证**（`flat-covering-batch1` `flat_gemm_paired.csv`）：**我方 repack-GEMM（prefill）5.10 GMAC/s vs ggml block-dot vec_dot `ggml_vec_dot_q8_0_q8_0` 1.242 = 4.10×**，**kernel-symmetric（gcc-15.2.0 双侧）**、"gcc-ours FASTER than clang-ours，win HOLDS symmetric"（**非 clang artifact**，[CASE-COMPILER-ASYMMETRY] 已排除）。对手 = hand_tuned_vec_dot（`quants.c:435` m2 intrinsic）→ 是 **Win-B 候选级对手**（非 scalar/generic）。
3. 4.10× < q4_0 的 6.76× — 与选择器"q8_0 block-dot 精简（8-bit dequant 便宜）"判断自洽（repack 摊薄收益小于 q4_0）。

### 丙.1 接线路径（M1 具体改动 · 同 q4_0/q4_K deploy_patch 模板 · 3 处）
1. **GEN gate**：板 A-tree `repack.cpp:4713`：
   `case 128: { break; } // TODO` → `case 128: { if (cur->ne[1] % 16 == 0) { return &q8_0_16x1_q8_0; } break; } /* TCRV-G5-M1 q8_0@VLEN128 */`
2. **ARCH include**：`arch/riscv/repack.cpp` 加 `#include "<q8_0 emitted GEVM .inc>"`（我方已有 emitter：`emitRepackGemvQ8_0Q8_0`，op `GgmlRepackGemvQ80Q80Op`，`RVVOps.td:5010` `tcrv_rvv.repack_gemv_q8_0_q8_0`；merged 24557f05）。
3. **ARCH kernel 分支**：`ggml_gemv_q8_0_16x1_q8_0()` 体内 `UNUSED(blocklen);` 后插 `if(vlenb*8==128){banner "TCRV EMITTED GEVM(q8_0_16x1 VLEN128…) ENGAGED"; tcrv_emitc_ggml_repack_gemv_q8_0_q8_0_kernel…(); return;}`。
   - **prefill/GEMM**：我方**无** emitted q8_0 repack GEMM（emitter 清单 = GEVM only；GEMM 覆盖 q4_1/q4_K）→ prefill 走**上游 `ggml_gemm_q8_0_16x1_q8_0`（白嫖，同 q4_0 prefill 的 A==B 字节等价机制）**，无需 ARCH GEMM 分支。若要"compiler-emitted prefill"须先补 q8_0 repack GEMM emitter（M1 不阻塞，标 P-后续）。
- **build/link/验**：同甲④⑤（`fullmarch_rebuild.sh` MARCH_STR override + `cmake --build --target ggml-cpu` + `nm`/`strings`/`objdump` seal + preflight 4/4 + greedy A==B）。可逆备份还原、无 git。

### 丙.2 ★两大 coherence 张力（M1 必须显式披露 · 影响措辞与判读）
- **张力 A — 我方选择器当前 DECLINE q8_0 repack**：`RVVContractionPathSelection.cpp:109-111` 对 q8_0 返回 `{BlockDot, "block-dot-decline-q8_0-lean-fallback"}`（`blockDotComputeHeavy==false`：q8_0 8-bit dequant 便宜、block-dot 精简、repack 摊薄收益不足）。→ **翻 gate route q8_0→repack 与我方编译器自身路径选择相矛盾**。M1 必须二选一措辞：(i) 框成 **routing-freebie 探针**（测上游 repack 路径，非我方 selected path；诚实标"selector 判 decline，M1 强制路由测传导"）；或 (ii) 若 e2e 证 repack 传导为正，则**回炉修 selector fact**（`blockDotComputeHeavy` for q8_0，或加 M-regime/prefill 键）使编译器**自洽地** select+deploy repack。**(ii) 是 canon 级 selector 语义变更 = 必问主会话**，M1 默认走 (i)。
- **张力 B — q8_0 = 2× q4_0 权重字节（memory 墙）**：decode 是 memory-bandwidth-bound；q8_0 每权重 8-bit（q4_0 4-bit）→ decode 大概率 **PARITY / LOSS**（repack 布局不减字节数、只改搬运顺序）。prefill 是 matmul/compute-bound → repack-GEMM 布局可能传导（同 q4_0 prefill），但因 dequant 精简、上限低于 q4_0 5.76×。
- **provisioning**：板无 f16/f32 tinyllama 源（T6 记）→ M1 用 `llama-quantize` 造 q8_0（`ssh rvv` 已有 binary，routing_gap_probe 已演示 q4_0→q8_0 requantize）。**诚实注**：requantize-from-q4_0 fidelity 有损，A/B 同模型只验**路由+传导**不验保真；若要独立 perf 主张宜 f16→q8_0（需先 provision f16 源）。

### 丙.3 e2e 分相全协议预案（复用 T6 骨架 `tools/e2e-harness/{run_e2e.sh,board/*}`，零 board 脚本改）
1. **P0 baseline+backup**：A-tree md5 baseline 断言 + `cp *.ORIG`；provision q8_0 gguf（quantize）。
2. **P1 patch+build**：`deploy_patch.py`(q8_0 版) 翻 :4713 gate + 挂 emitted GEVM → `fullmarch_rebuild.sh`(MARCH_STR override) 对称重建 A(ours)+B(stock)。
3. **P2 engage 验**：`nm`/`strings` 验 q8_0 tcrv 符号+banner；跑 A-tree llama-bench q8_0 model 验 **banner FIRES**（对照 T6 的 ABSENT → 现应 PRESENT = 路由已通）；objdump 验 kernel vl=8 变体。
4. **P3 preflight 4/4**（同编译器/march-complete/libcall-free/VLEN-fingerprint）fail-closed。
5. **P4 correctness**：`correctness_gate.sh` 贪心 token A==B + logits sanity（prefill kernel bounded-ULP，措辞不说 byte-exact-vs-ggml）。
6. **P5 分相 perf**：`phase_split_ab.sh`（llama-bench `-p PP` prefill / `-n TG` decode 分相、配对交替 2pass×5rep、taskset pin、DVFS 锁、逐 rep JSON）→ `aggregate_e2e.py`（中位+IQR%+bootstrap 95%CI+T-N floor+PARITY/DIFFERENCE 判决）。**冷启守卫**：首跑冷+暖混则作废重跑暖值（T6 q4_0 SOP）。
7. **P6 restore**：`board_restore.sh` 还原 + rebuild + 测后 md5 证零 stock 改动。

### 丙.4 判读预注册（照判照走 · 零未定义 · 含措辞模板留数字空位）
- **判据门（沿 T6/八门 SOP）**：分相 e2e 过 **2× floor 或 CI 排除 1.0** ⇒ DIFFERENCE；CI 含 1.0 或 <T-N floor ⇒ PARITY。correctness 必 GREEN 方可判 perf（否则 kernel-变体缺陷判，参 seal-fix）。
- **结果 R1 — prefill ≥ DIFFERENCE 且 correctness GREEN**（预期主线，q8_0 prefill matmul-bound 可能传导）：
  → q8_0 从 kernel-axis-only 黄格 **翻正 perf-covered 绿格**（分相绿：prefill 绿 / decode 按 R2/R3 分判）。**预注册措辞模板**（留空位，落地填数即用、不回门）：
  > "q8_0 whole-model e2e 传导（rvv-openEuler-VLEN128·tinyllama-Q8_0·kernel-symmetric gcc-15.2.0 双树账·对手=stock ggml block-dot dispatched VLEN128）：**prefill ⟨X⟩× / decode ⟨Y⟩×**（⟨DIFFERENCE/PARITY⟩·correctness GREEN·DVFS ⟨d⟩%·pass-spread ⟨s⟩%）。机制 = **能力键控 repack 路由**（上游具 q8_0 repack 全链路但 VLEN128 gate OFF；我方翻一行接入）+ decode 内存局部性；**非 kernel-质量-vs-手调赢**（同 q4_0 routing-win 披露）。★张力披露：我方 selector 现判 q8_0 decline-repack（lean-fallback），本格为强制路由传导探针。八门：byte-exact(bounded-ULP)⟨✓⟩ / 对手对称(gcc-sym)⟨✓⟩ / 双账本(kernel-sym)⟨✓⟩ / 对手身份探针(hand_tuned_vec_dot)⟨✓⟩ / micro∧e2e⟨✓⟩ / selector-routing(banner FIRES)⟨✓⟩ / 纪律⟨✓⟩ / 措辞(routing-win+lean-decline 披露)⟨✓⟩。"
  → **perf-covered +1 起步**（当前基数以主会话 ROADMAP 系统账为准；q4_0 已绿，q8_0 为**新增**候选）。**登记按 ⑦⑧ 预注册自决**（填数即用、非 canon 级）；**唯**"张力 A(ii) 修 selector 语义"若触发 = 必问主会话。
- **结果 R2 — decode PARITY**（预期，memory 墙）：
  → decode 判 **PARITY = kernel-axis-only（诚实内存墙格）**，四列传导会计记 `micro 4.10 → e2e-decode ⟨Y⟩ PARITY`，稀释归因 = memory-bandwidth-bound（micro↛e2e 铁律范例）。不冒绿。
- **结果 R3 — prefill 亦 PARITY / 传导稀释**（若 q8_0 lean-dequant 使 repack 无 e2e 优势）：
  → 全格 **kernel-axis-only 黄 + 具名 `[GAP-FLAT-E2E-ROUTING]`(q8_0 变体：gate 可翻但 lean 不传导)**；**传导会计四列**（micro / wired=YES / e2e-prefill ⟨X⟩ / e2e-decode ⟨Y⟩ / 稀释=lean-dequant-no-repack-benefit）+ **kernel-axis-only 归因**。这是**高价值 negative**（证 routing-freebie 存在但 lean 格不传导，收窄 q4_0 5.76× 的可外推边界）——与 T6 q8_0 1.05× parity（未路由）互补：R3 = **已路由仍 parity**（更强的 micro↛e2e 证据）。
- **结果 R4 — correctness RED**（kernel-变体误挂类，seal-fix 前车）：
  → 判 **NOT sealed / kernel-变体缺陷**，objdump 定位 vl 宽度，修部署变体（非能力缺口）。不判 perf。
- **共同铁律**：micro↛e2e 仍管辖；prefill/decode **永远分开报**；q8_0 任何 e2e 数**必绑** routing-win 披露 + selector-decline 张力披露（同 q4_0 canon 措辞锁精神）。

---

## 交付确认
- **产出**：本文件 `experiments/active/g5-wiring/M0-接线机制解剖.md`（甲 q4_0 五点解剖 / 乙 FLAT·IME·K-quant 通用方案 / 丙 M1 q8_0 路径+全协议+判读预注册）。
- **HEAD**：`aeed7e0b` **未变**（纯 scout/docs，零 git，零 code/schema/lib/canon/既有 casefile 改动）。
- **触碰集**：仅新建 `experiments/active/g5-wiring/`（锁 docs）。
