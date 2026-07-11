# G5-M1 曳光弹 — q8_0 单格接线贯通 · CASEFILE

> **campaign**: G5 接线战役 · **M1 = q8_0 routing-freebie 探针**（绕 selector·直翻 gate 测传导）
> **board**: `ssh rvv` = openEuler 24.03 / VLEN128 / gcc-15.2.0 / 64c / governor=performance @2.6GHz
> **A-tree**: `/home/ubuntu/tcrv-llamacpp/ggml` HEAD `f3e1828`（working-tree baseline = q4_0 WinB-ON，inert for q8_0 model）
> **HEAD (TianChen-RV)**: `ab054260` 未变 · 禁 git · A-tree 可逆（deploy 备份 + 测后 restore stock md5）
> **结论**: **R4-refined — correctness RED**。routing 接线**成功且 engage 已证**，但 q8_0 的**上游 VLEN128 repack kernel 数值破损**（hardcoded vl=16 → VLEN128 钳到 vl=8 → 只算半列 → garbage）。**q8_0 ≠ "翻一行即得" freebie**；perf **VOID 不判**。

---

## 一、判读落点 = R4（refined attribution）

预注册 R4 = "correctness RED（kernel-变体误挂类，seal-fix 前车）"。**本例 attribution 更强/更干净**：我方**未部署任何 emitted kernel**（banner-only，fall through 到上游 body）→ 缺陷在**上游 q8_0 repack VLEN128 kernel 本身**，非我方误挂变体。

| 预注册分支 | 是否命中 | 说明 |
|---|---|---|
| R1 prefill 传导→转绿 | ✗ | correctness RED，perf 不判。prefill "8.69×" 是 MIRAGE（见三）。 |
| R2 decode parity | ✗ | 未到判 perf 阶段（correctness gate 前置失败）。 |
| R3 已路由仍 parity | ✗ | 比 R3 更强的 negative：**已路由且 engage 已证，但 correctness 破**（freebie 前提在正确性层被证伪）。 |
| **R4 correctness RED** | **✓（refined）** | 缺陷 = 上游 kernel（非我方变体）。具名 **`[GAP-Q8_0-VLEN128-KERNEL]`**：上游 repack VLEN128 hardcoded vl=16。 |

---

## 二、接线落地（三挂点全通 · 可逆）

- **挂点① dispatch gate（GEN `repack.cpp:4713`）**：`case 128: { break; } // TODO` → `case 128: { if (cur->ne[1]%16==0) return &q8_0_16x1_q8_0; break; } /* TCRV-G5-M1 */`。翻一行，route q8_0 mul_mat → 上游 `q8_0_16x1_q8_0` repack trait（= case256 已用的**同一** trait）。
- **挂点② engage 探针（ARCH `arch/riscv/repack.cpp`）**：`ggml_gemv_q8_0_16x1_q8_0`(:518, marker :535) + `ggml_gemm_q8_0_16x1_q8_0`(:1426, marker :1444) 体内插 VLEN128 一次性 banner（**banner-only，无 kernel swap、无 return**，fall through 到上游 repack body）。
- **挂点③ 我方 emitted kernel**：**未部署**。板上无 `tcrv-opt`/`tcrv-translate`、无 q8_0 `.inc`（emitter 在仓库 merged `24557f05` 但未在板构建）→ M0 "可选挂 emitted GEVM" 未执行；prefill/GEMM 与 decode/GEVM **均走上游 repack body**（真 freebie 路径本体）。

**engage 验证（objdump/banner seal）**：
- ON `.so` `strings` 含 gevm+gemm banner（各 ≥1）；OFF `.so` = 0。
- 真实 forward 触发：ON 运行 `llama-bench` → **GEVM banner ×8 + GEMM banner ×4 FIRE**（`engage_ON.stderr`）；OFF → **0**（`engage_OFF.stderr`）。→ **route 确实 engage 上游 repack GEMM(prefill)+GEVM(decode) @VLEN128**。与 T6 q8_0-unwired banner-ABSENT 互补：此处 banner-PRESENT = **已路由**。

---

## 三、e2e 分相数据 —— VOID（correctness RED，perf 不判）

**同树物理 .so 交换 A/B**（一份源码、一个 `llama-bench`、gcc-15.2.0，唯一差异 = q8_0 gate 那一行）→ 最干净的 gcc-symmetric 对手对称。model = `tinyllama-q8_0.gguf`（q4_0→q8_0 requantize，1.09GiB，8.50BPW；A/B 同文件，只验路由+传导不验保真）。

| 相 | ours(ON, repack-routed) | stock(OFF, block-dot) | 比 | DVFS | 判 |
|---|---|---|---|---|---|
| prefill pp128 | 78.11 t/s (IQR 0.75%, n=8) | 8.99 t/s (IQR 0.02%) | **8.69× MIRAGE** | span 0% | **VOID** |
| decode tg32 | 7.55 t/s (IQR 1.42%, n=8) | 2.23 t/s (IQR 0.25%) | **3.39× MIRAGE** | span 0% | **VOID** |

★**为何是 MIRAGE 而非 win**：aggregator 只看吞吐、不知 ON 输出是 garbage。ON 路径**数值不正确**（correctness RED），且其"快"部分来自 **vl 钳位跳过半数列的做少工**（见四）+ repack streaming 布局。**任何吞吐差都骑在被跳过的、错误的工作上** → 按 R4/correctness-gate 纪律**不构成任何 perf 主张**。这是"gate-flip alone 给出 spectacular-looking 8.69× prefill 却完全数值错误"的**教科书级 correctness-gate-catches-mirage 正面教材**。

---

## 四、根因（source-confirmed）= `[GAP-Q8_0-VLEN128-KERNEL]`

`arch/riscv/repack.cpp` 的 `ggml_gemv/gemm_q8_0_16x1_q8_0` **每个 RVV intrinsic 硬编码 AVL=16**（e32,m2）：
```c
vfloat32m2_t sumf = __riscv_vfmv_v_f_f32m2(0.0f, 16);
vint8mf2_t   b_0  = __riscv_vle8_v_i8mf2((const int8_t*)&b_ptr[l].qs[i*16], 16);
sumi = __riscv_vwadd_wv_i32m2(sumi, __riscv_vwmul_vx_i16m1(b_0, a_ptr[l].qs[i], 16), 16);
```
- **VLEN256**：e32m2 VLMAX=16 → AVL=16 正好满组 → 正确（∴上游只 route `case256`）。
- **VLEN128**：e32m2 VLMAX=8 → 硬件把 vl 钳 16→8 → 16-way 交织**只算 8 列**、i8mf2 load 16 字节只取 8 → **半列 garbage**。

这是 seal-fix 记录过的 **vl=16 vs vl=8 病理**（memory `zero-model-adjudication-cert-hardening`：部署变体≠证过变体）。∴上游 `case128 //TODO` = **真 TODO（VLEN128 kernel 未实现/破损）**，**非 working-but-gated freebie**。

---

## 五、★两大张力披露（措辞锁 · 主会话裁决项）

- **张力 A（selector decline）**：我方 selector 现判 q8_0 `block-dot-decline-q8_0-lean-fallback`（`RVVContractionPathSelection.cpp:111`）。M1 = **强制路由探针**（绕 selector 翻 gate）。**本任务未修 selector**（canon 级语义变更 = 必问）。**本结果反而为 selector 的 decline 提供了非预期支撑**：q8_0 上游 repack VLEN128 破损，直翻 gate 会 e2e garbage → selector 拒 repack 在正确性上是安全的（虽然理由不同：selector 理由是 lean-dequant 收益不足，真实理由是上游 kernel 破损）。
- **canon-framing refinement（必问主会话，我方不 codify）**：本结果**收窄** `q4-0-e2e-is-routing-not-kernel` 的 "routing 白嫖" framing —— **VLEN128 上 correctness 不是 free**：q4_0 之所以在 VLEN128 正确，是因为**我方 emitted vl=8 two-strip kernel 拦截**在这同一段破损的上游 vl=16 body 之前。**emitted kernel = VLEN128 的 correctness-carrier**（dispatch/routing 或许主导 perf 比值，但 emitted kernel 承担正确性）。这是对 canon 措辞的实质精化，**留主会话裁决是否更新 memory/canon**。

---

## 六、可复现 / 触碰集 / 交付

- **新建脚本（deploy harness，触碰集内）**：`tools/e2e-harness/board/g5-m1-q8/{deploy_patch_q8.py, g5_build_variants.sh, g5_engage_correctness.sh, g5_phase_split_swap.sh}`。
- **新建 casefile（本目录）**：`MANIFEST.md`(本文) · `transmission_accounting.csv` · `correctness_RED.txt` · `phase_split_raw_VOID.txt` · `evidence_VOID.json`(stamped VOID) · `engage_ON.stderr` / `engage_OFF.stderr`。
- **板 provision（新文件，非 stock 改动）**：`/home/ubuntu/tcrv-llamacpp/models/tinyllama-q8_0.gguf`（requantize，reusable；留存）。
- **A-tree restore 证据（零 stock 永久改动）**：测后 GEN md5=`deb61a29…`、ARCH md5=`99131cf7…`、live `.so` md5=`05a62e6a…`（== OFF-pristine，banner=0），全 == baseline。source restore byte-exact + pristine rebuild md5-identical（确定性构建）。
- **HEAD (TianChen-RV) = `ab054260` 未变** · 全程禁 git · lib/(selector)·schema·ROADMAP 未碰。
