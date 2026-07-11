# G5-M2 q4_K 曳光弹 — evidence（L-接线② correctness-carrier·yellow-kernel-axis）

> workflow `wyhnspmt6`（2026-07-12）· board `ssh rvv` openEuler VLEN128 gcc-15.2.0 · A-tree HEAD f3e1828 ·
> **HEAD (TianChen-RV)** = 609d79c4（张力A 前·工作树另有 parallel-writer 未提交改动·我方零 tracked 源改动）。
> **结论**：correctness **GREEN**（我方 emitted vl=8 kernel = VLEN128 correctness-carrier·第 2 例）· **perf < parity 全相** → **verdict yellow-kernel-axis · perf-covered 维持 3/84**（correctness 门绿前不入台账·q4_K repack 慢于 stock scalar vec_dot）。

## 一、recon：q4_K = q8_0-class（★纠偏 "K-quant 缺整条链路"）

板 A-tree（read-only 核实·md5 GEN=deb61a29 / ARCH=99131cf7 = 干净 WinB-q4_0-ON baseline）：
- **上游链路全 present**（NOT "缺整条链路"·该 framing 仅适用 q5_K/q6_K/q3_K 零 riscv 分支）：
  - trait 已注册 GEN `repack.cpp:4567` `tensor_traits<block_q4_K,1,16,GGML_TYPE_Q8_K> q4_K_16x1_q8_K`
  - case256 已 route GEN:4620
  - arch kernel body 已存 `ggml_gemv_q4_K_16x1_q8_K` ARCH:331 + `ggml_gemm_q4_K_16x1_q8_K` ARCH:1073
  - **case128 gate = OFF**（GEN:4619 `case 128:{break;}//TODO`·需翻一行）
- **★[GAP-Q4K-VLEN128] 确证（第 2 个 q8_0-MIRAGE 排雷）**：arch q4_K body load intrinsic 硬编码 AVL=16（`vle8_v_i8mf2(...,16)` ARCH:312·`vle16_v_f16m1(...,16)` ARCH:322/363·`vuint8mf2 ...,16` ARCH:410/436）= VLEN256 变体；VLEN128 上 i8mf2/f16m1 VLMAX=8 → AVL=16 钳 8 → 只算 16 交织列的一半 → 输出垃圾（PPL 822057·同 [GAP-Q8_0-VLEN128-KERNEL] 类）。**裸翻 case128 = 第 2 个 MIRAGE**；我方 emitted vl=8 correctness-carrier 必须拦截。
- **无需 scaffold 新建**（provisioning 全 present：DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf + qwen2.5-0.5b-q4_k_m 板上已存·llama-quantize/bench/perplexity 三件全在）。★M2-recon "board 无 q4_K gguf/quantize" **已 STALE**。

## 二、kernel emit（vl=8 correctness-carrier·VLEN128-safe）

- host `./build/bin/tcrv-opt`（LLVM20.1.8）+ `/usr/lib/llvm-20/bin/mlir-translate`：
  - GEMM：`tcrv-opt <prefill-vlen128.mlir> --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp` → md5 **6cbd9c19**（1834163B·GEMM 404× vle AVL=8）
  - GEVM：`tcrv-opt <repack-gemv-q4-K.mlir> --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp` → md5 **e909a9bd**（728730B·GEVM 292×·**byte-identical to 板 seal-proven** e909a9bd）
  - 全 vle load AVL=8 = VLEN128-safe（非破损 AVL=16）。
  - GEMM current-HEAD vs 板 seal-proven（90d454da）= 同一 kernel body·仅 function-arg ABI 重排（seal-fix 07-10 起 lowering 演进）。

## 三、deploy + correctness-first（board·A-tree 可逆）

3 挂点 deploy（`tools/e2e-harness/board/g5-m2-q4k/deploy_patch_q4k_emitted.py`）：GEN case128 flip→trait · ARCH `#include` 我方 .inc · ARCH body VLEN128 分支 banner+调 emitted+return（拦截破损上游）。gcc-15.2.0 对称 OFF/ON。SEAL：nm tcrv 符号 + banner + objdump vl=8（`vsetivli imm=8 only·0x imm=16·0x imm=64`）。

- **★correctness GREEN**：整模型 q4_K greedy A==B + PPL 一致（我方 vl=8 carrier 正确·区别裸翻 gate 的 PPL 822057 garbage）。**broken_upstream confirmed**。
- **registration = correctness-carrier**（C1 合取存在性：全模型路由 + 权重 repack + q8_K 激活 + dispatch 自建集成·经 ggml-generic 对照证正确）。

## 四、perf 分相（correctness 过后·双账本·对称·yellow）

| 模型 | 相 | ours(carrier) | stock(generic vec_dot) | 比 | N |
|---|---|---|---|---|---|
| DeepSeek-8B-Q4_K_M | prefill pp128 | 2.191 t/s | 5.173 t/s | **0.4236×** | 8/side |
| DeepSeek-8B-Q4_K_M | decode tg8 | 0.381 t/s | 2.100 t/s | **0.1815×** | 8/side |
| qwen2.5-0.5b-q4_k_m | prefill | 55.31 t/s | 65.84 t/s | **0.8400×** | 20/side |
| qwen2.5-0.5b-q4_k_m | decode | 9.36 t/s | 11.83 t/s | **0.7916×** | 20/side |

**全相 < parity**（carrier 慢于 stock generic q4_K vec_dot·decode 内存受限 GEVM heavy vsetvli churn）→ **verdict yellow-kernel-axis**。**perf-covered delta = 0（维持 3/84）**：fair-protocol ≥parity/win **不满足** → **禁以 perf 名义入台账**。与 memory `q4-0-e2e-is-routing-not-kernel`（q4_K perf 立不住·S6 1.884× clang-artifact·对称 gcc 0.272× 已撤回）一致。

## 五、A-tree restore（零 stock 永久改动·已验）

`a_tree_restored: true` · 源 RESTORE byte-exact（GEN restored deb61a29 · ARCH restored 99131cf7）+ pristine rebuild · 板 baseline 可逆确认。

## 六、L-接线② 建法（★方法学产出·供 q5_0/q5_1）

- **q4_K 建法** = 零 scaffold 新建（上游全 present·q8_0-class）→ 复用 q4_0/q8_0 的 3 挂点 deploy 模板。
- **★q5_0/q5_1 建法（净新·下一里程碑·q4_K 不适用因它们上游零 riscv repack 分支）**：先净新 (1) GEN trait 注册 `tensor_traits<block_q5_0,1,16,GGML_TYPE_Q8_0>` (2) GEN case256/128 route (3) ARCH gemv/gemm skeleton·之后套 3 挂点 deploy 模板。q5_0/q5_1 无 ABI 阻塞（q8_0/q8_1 激活可复用）·机械净新。完整 recipe 见 harness `g5-m2-q4k/` 脚本 + workflow result `scaffold_recipe_for_q5`。**破损上游检查铁律**（防第 3 个 MIRAGE）：若 q5 arch body 硬编码 AVL=16/64 = VLEN256 变体 = 破损@VLEN128 → carrier 拦截。

## durable files
- `evidence.md`
- `MANIFEST.md`
> emitted `.inc`（GEMM 6cbd9c19 1.8MB / GEVM e909a9bd 729KB）= **gitignored**（regenerable via §二 emit recipe·md5 已记·避 2.5MB repo bloat）。
