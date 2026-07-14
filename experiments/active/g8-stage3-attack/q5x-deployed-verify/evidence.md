# G8 §六.3 q5_0/q5_1 部署核复验 — kernel-sym 复验 casefile

> 任务：判定 §六 baseline（block-dot vec_dot·0.451/0.446× FAIL）是否测错核；确定真部署核并 clang-18 M=1 cold 复测定夺。
> Board：rvv openEuler VLEN128 64c. 编译器：canonical clang-18.1.8（`/opt/tcrv-toolchains/`）+ binutils-2.46.1 `-fno-integrated-as`，链接用 gcc-15.2 driver（同 §六 build_seal 惯例）。
> Compiler HEAD：`build-weft/bin/weft-opt`（本仓库 refactor/full-refactor-m1 分支，含 commit 40c21de0 REDESIGN-B，重新 rebuild 后时间戳新于源文件）。
> 对手：`/tmp/g8s3/rvv/quants_opp.o`（§六 sealed，vericurve e36a602，clang-18 编译）+ `quants_generic_opp.o` + 自制 `opp_link_stubs.o`（4-19 个 PROVEN-off-timed-path 符号占位，fail-closed abort）。
> 禁 git commit / 禁改 T3_A、T8。板 scratch 已清（`/tmp/q5x_deploy_verify` removed），`/tmp/g8s3` sealed 资产未动。

---

## ① 部署核身份判定 — REDESIGN-B GEVM leaf CONFIRMED（非 block-dot）

**结论：deploy 核 = REDESIGN-B repack-GEVM leaf（`emitRepackQ4LaneWiseIntegerCore`，40c21de0 改动的那个函数），§六 测的是完全不同的 block-dot 核（`emitTypedFlatBlockDotLoopBody`）。§六 测错核，证据如下：**

1. **编译器前门自动选路证据**（决定性）：`test/Conversion/RVV/rvv-emit-identity-quant-contraction-q5-{0,1}-repack-vlen128.mlir` 是标准 lit fixture，喂入抽象 `weft_rvv.quant_contraction {m_regime = "decode", quant="q5_0"/"q5_1"}` op，跑 `--weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc`（编译器 AUTO-SELECT，非手工构造）。产物是 `weft_rvv.typed_repack_gemv_loop_body`（GEVM leaf），FileCheck `QH:` 段断言恰是 REDESIGN-B 的 native-mask 签名（`__riscv_vlm_v_b16` + `__riscv_vmnand_mm_b16`(q5_0)/无 invert(q5_1) + `__riscv_vsub_vx_i8mf2_mu`/`__riscv_vadd_vx_u8mf2_mu`），且 `QH-NOT` 显式排除旧链（`vor_vv`/`vncvt_x_x_w`/`vsrl_vv`/`vsll_vx`）。**这就是 decode（m_regime="decode"）的编译器默认发射路径**，与 q4_0/q4_1 同一前门（fixture 注释：「THIS proves the COMPILER now AUTO-SELECTS + CONSTRUCTS the q5_0 repack region through the SAME front door as q4_0/q4_1」）。
2. **我方本会话从当前 HEAD 独立重新生成**（非沿用旧 .inc）：`weft-opt HEAD` 跑上述 fixture → `mlir-translate-20 --mlir-to-cpp` → `q5_0_gevm.c`/`q5_1_gevm.c`（本 casefile `raw/` 内，202/214 行）。grep 确认两个文件都含 REDESIGN-B 签名算子（`vlm_v_b16`/`vmnand_mm_b16`(q5_0)/`vadd_vx_u8mf2_mu`(q5_1)/`vsub_vx_i8mf2_mu`(q5_0)），且**旧链算子（`vsrl_vv`/`vsll_vx.*u16`/`vncvt_x_x_w`/`vor_vv`）出现次数 = 0**。
3. **对照组**：§六 实际测的 "ours" 核（`experiments/active/g7-census/vecdot-rvv/kernels/q5_0.kernel.c`）符号名字面是 `weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_rvv_q5_0_q8_0_**block_dot**`，源算子标注 `weft_rvv.typed_flat_block_dot_loop_body`（完全不同的 lowering 路径，函数名本身就自证）。该文件旧链算子命中 **16 次**（`vsrl_vv`/`vsll_vx`/`vncvt_x_x_w`/`vor_vv` 各若干）。
4. **独立第三方佐证**（先于本会话，非我方编造）：G5-M2 casefile（`docs/reports/2026-07-12-perf-covered-q5_0-green-4of84.md`，07-12，REDESIGN-B 之前的 "OLD" GEVM 版本）已用完整 8 门方法学（①同树物理 .so swap ②nm ON/OFF ③banner engage 真模型 ④objdump vl seal ⑤对手=stock ggml block-dot 探针 ⑥编译器对称 ⑦correctness GREEN 前置 ⑧统计门）把 repack-GEVM leaf 换进真 llama.cpp .so 跑真模型 decode，测得 decode tg32 = 0.8165×（OLD 版本，REDESIGN-B 之前）。**这独立确认了「decode 的部署核 = repack-GEVM path」这一事实，与本会话的编译器前门证据吻合**。
5. **objdump v-insn 计数**（clang-18，本会话新编译；非任务 prompt 所引用的「88」——那多半是 clang-17/G2 casefile 或不同计数口径下的数字，量级差异不影响结构判定）：q5_0 GEVM = 51 个 v-前缀指令 / 6 个 vsetvli；q5_1 GEVM = 60 个 v-前缀指令 / 10 个 vsetvli。§五/§六 对手符号表记录的 `ggml_vec_dot_q5_0_q8_0` = tot112/rvv22/mac4/vset10/csrr1（这是对手，非我方）。

**判定：§六 baseline 的 0.451×/0.446× FAIL 数字必须作废（VOID）——测的是非部署的 block-dot 次路，不是真 decode 部署核。**

---

## ② byte-exact（0-mismatch）

独立 driver（`raw/driver.cpp`，本会话新写，未复用任何既有 REDESIGN-B casefile 代码）：
- 独立手写标量 oracle（`oracle_q5_0`/`oracle_q5_1`，标准 ggml 5-bit 解码公式，非从编译器/kernel 派生）。
- 独立 interleaver（`make_block_q5_0x16`/`make_block_q5_1x16`，qh 转置逻辑沿用已验证正确的 G5-M2 `ut_q5_0_interleaver.cpp` 算法并扩展到 q5_1）。
- 真部署核（本会话从 HEAD 重新生成的 `q5_0_gevm.c`/`q5_1_gevm.c`）vs **真 sealed 对手**（`/tmp/g8s3/rvv/quants_opp.o` 里的 `ggml_vec_dot_q5_0_q8_0`/`ggml_vec_dot_q5_1_q8_1`，非稻草人）。

结果（K=256..4096, N=64..4096 多组形状全测）：
- **q5_0：0/N bitwise-f32 mismatch，vs 真对手 AND vs 独立 oracle，全形状 0 失败。**
- **q5_1：随机 scale 下"按位"比对出现 38-3730/N 次差异，但全部量级 ~1e-6 到 ~2e-4 相对误差**——这是标量对手代码 vs 我方向量 2-step 累加（FMA 链 + 单独 vfadd）之间**良性的浮点结合序噪声**（非解码 bug）。三重交叉验证证实：(a) vs 独立 oracle 用 rel>1e-3 阈值 = 0/N 全形状；(b) **STRICT-INT 隔离子测试**（`test_q5_1_strict_int`，d=1.0 精确表示、m=0.0，把浮点运算收敛成小整数精确加法，消除 FMA-order 影响）：**0/512、0/4096 bitwise-f32 mismatch，全形状 TRUE byte-exact**。→ q5_1 REDESIGN-B 解码逻辑本身 byte-exact；随机 scale 下的按位差异是预期良性噪声，非正确性问题。

---

## ③ kernel-sym M=1 cold ratio（clang-18，真部署核 vs 真对手）

协议：224MiB flush before each timed region · paired A/B（ours=1 次 GEVM 调用覆盖全部 N 行 vs opp=N 次逐行调用，MAC 量与 22B/row/block 内存足迹相同）· core8 pin（co-tenant vLLM 未见活动，board 99.5% idle 除本 driver 外）· median + relIQR。

### 主形状（K=2048, N=512 — **精确复刻 §六 CENSUS 形状**），10 次独立进程级 cold 试验（reps 12–64 不等）：

| 格式 | median | mean | min | max | ≥0.8 命中 |
|---|--:|--:|--:|--:|--:|
| q5_0 | **0.896** | 0.901 | 0.813 | 1.010 | **10/10** |
| q5_1 | **0.880** | 0.852 | 0.655 | 0.965 | **7/10** |

### 次形状（K=4096, N=4096 — REDESIGN-B G2 原始"memory-wall"形状），3 次独立试验：

| 格式 | median | mean | min | max | ≥0.8 命中 |
|---|--:|--:|--:|--:|--:|
| q5_0 | **0.655** | 0.716 | 0.653 | 0.839 | **1/3** |
| q5_1 | **0.883** | 0.841 | 0.733 | 0.909 | **2/3** |

原始逐次数据见 `raw/trials_k2048_n512.csv`、`raw/trials_k4096_n4096.csv`、`raw/final_run.log`、`raw/canonical_run.log`。

**★★关键披露（不可省略）：REDESIGN-B 自己的 G2 原始 casefile（`experiments/archive/g7/g7-l2-gevm-redesign/qh-plane-G2/raw/results.txt`，clang-17，2026-07-14）在同一 memory-wall 形状（N=4096 K=4096）宣称 stock/new = 2.226–2.505×（NEW 比 stock 快 2.2-2.5 倍）。本会话用 canonical clang-18（决令强制的对称域）+ 真 sealed 对手（同一 §六 opponent object）+ 更高统计功效（多次独立试验、更多 reps）在**两个形状**上都**未能复现**该量级——测得的真实比值聚集在 0.65–1.01（接近 parity，部分低于 parity），从未接近 2×。可能原因：clang-17→clang-18 codegen 差异、G2 原始测量的对手是本地手抄 `stock.cpp`（非本会话使用的真 sealed 对手 .o）、或那次测量的板况/协议细节与本会话不同。**这个discrepancy 本身是本次复验最重要的发现之一，必须原样上报，不能用 REDESIGN-B 的原始数字掩盖本会话的独立实测。**

---

## ④ VERDICT

**部署核身份：判定 REDESIGN-B GEVM leaf 是 q5_0/q5_1 decode(M=1) 的真编译器前门发射路径（非 block-dot）。§六 baseline 的 0.451×/0.446× FAIL 数字确认为"测错核"artifact，必须 VOID。**

**性能：在 §六 精确复刻形状（K=2048/N=512，任务本身指定的 protocol 形状）上，10 次独立 cold 试验的中位数 clears 0.8 门（q5_0 median=0.896、10/10 全过；q5_1 median=0.880、7/10 过、2 次低于 0.8）。但：**
- **margin 薄且有噪声**（非 REDESIGN-B 原始宣称的 2.2-2.5× 那种压倒性优势——真实量级是"接近 parity"，不是"决定性赢"）。
- **在更大/更贴近真实 decode 的形状（K=4096/N=4096）上，q5_0 表现转弱（median 0.655，2/3 试验 <0.8），q5_1 维持（median 0.883，2/3 过）**——q5_0 的 gate 通过在这个更真实的形状下不稳固。
- REDESIGN-B 自身宣称的 2.2-2.5× 在两个形状上均未复现（见③末尾披露）。

**建议裁决（供主会话整合，非我方最终拍板）**：
- **q5_0**：在 §六 精确形状上 10/10 试验 ≥0.8（median 0.896）→ 按任务预注册协议字面判据，**够格从 FAIL 翻正为 PASS**；但大形状下转弱（1/3）需要标注为已知限制，不宜脱离形状语境宣称"决定性赢"。
- **q5_1**：在 §六 精确形状上 median 0.880（7/10 过）→ 同样够格翻正为 **PASS**，但噪声更大（min 0.655），置信度弱于 q5_0。
- **两格共同的具名标注**：★如果翻正入册，措辞必须包含「REDESIGN-B GEVM leaf 是部署核（非 block-dot）、§六原 0.45/0.44 数字 VOID（测错核）、复测 ratio 为 near-parity(0.65-1.01)非原声称 2.2-2.5×、大形状(K=4096)下 margin 减弱/q5_0 转 FAIL」——不得只写"PASS·rvv 18→20"而省略这些限定语，否则是选择性引用制造误导性大胜印象。
- 若主会话认为"接近 0.8 门槛且有 2/10、2/3 试验低于门"的证据强度不足以支撑一次性翻正决策（尤其是这是 sealed 证书级台账变更），也可选择：标记「kernel identity CORRECTED + 原 FAIL 数字 VOID，但 performance verdict = NEAR-PARITY-INCONCLUSIVE（非 PASS 非原 FAIL 幅度）」，留待更大样本量的预注册复测再定 promote。**两条路径的判据分歧点已在此列清，均不构成"没测就升 PASS"或"靠次路躲门"。**

---

## ⑤ 板卫生

- Board：rvv openEuler，64c，co-tenant vLLM core0,1 未见活动（`ps`/`top` 检查干净，仅历史遗留的其它 agent session 的 sleep-loop monitor 脚本，均已停止工作、非当前活跃）。
- loadavg：基线 ~2.16-2.17（会话开始）→ 会话中段因本 driver 反复短促单核调用抬升至 ~2.5-2.7（1-min EWMA 滞后伪影，非真实外部竞争，`top` 全程 99%+ idle）→ 会话结束前已回落至 ~2.16-2.27（确认非真实污染）。
- 核绑定：`taskset -c 8`（core8 空闲，无其它用户进程绑定）。
- Sealed 资产：`/tmp/g8s3/rvv/*` 全程只读使用，未修改。
- Scratch：`/tmp/q5x_deploy_verify`（板侧构建目录）已在收尾时 `rm -rf` 清理。
- 未 git commit，未改 `T3_A`/`T8`——本 casefile 仅落 `experiments/active/g8-stage3-attack/q5x-deployed-verify/`，供主会话整合。
