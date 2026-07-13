# G7 L2 — qh-plane GEVM decode 削重建 · G2 发射器实装 + byte-exact + cold GEVM M=1 micro · [VERIFY-LADDER] G2 层

> **前置**：G1 双子门 literal PASS（`../qh-plane-G1/evidence.md`·register-budget-fit + instruction-count·带 vsetvli-attribution caveat）→ q5_0@rvv 削重建线获 G2 资格。本任务 = G1 预注册的下一梯级：真发射器实装 + byte-exact 硬门 + cold GEVM M=1 micro（解 vsetvli 归因）。
>
> **★裁决(TL;DR)：G2 三门全绿。** ① 发射器 REDESIGN-B 实装（GEVM leaf·**sealed GEMM leaf 未触碰**）→ Conversion/RVV lit **242/242** 零回归。② **byte-exact 硬门 GREEN**：OLD-vs-NEW **0 / 256,000** f32 输出 + 代数模型 q5_0/q5_1 各 0/32。③ **cold GEVM M=1 micro：NEW ≥parity vs stock 决定性达标 = 2.23–2.51× 更快**（全 regime·真墙钟）。**vsetvli 归因 RESOLVED**：真墙钟 NEW-vs-stock 2.2–2.5× 由 repack-GEVM 内存局部性主导（远大于 G1 的 1.047× 指令数）；**REDESIGN-B 削重建的净贡献（decode-isolated NEW-vs-OLD·同 12.6MB 足迹·两侧皆 constant-vl）= 稳健 1.44–1.67× 真墙钟加速** → 削减是真省周期、非 vsetvli 计数 artifact；G1 caveat 消解（NEW-vs-OLD 两侧同 constant-vl，vset 不是变量）。
>
> **诚实分账（系统账铁律）**：2.2× vs stock = **repack-GEVM 路径赢（内存局部性·OLD 也有：OLD vs stock 1.2–1.7×）**，非 decode 质量赢。REDESIGN-B 的专属贡献 = **NEW-vs-OLD 1.44–1.67×**（decode 计算削减·在 repack-GEVM 里 decode 在关键路径上→不被内存墙洗掉）。两者分开报。

---

## 0. 环境与工具链（板测·可复现）
- 板：`ssh rvv`·openEuler·64c·riscv64·**VLEN128**·isa `rv64imafdcv...zvfh`。
- **部署编译器（双方对称）= clang-17.0.6**。★纠偏：板默认 gcc **12.3.1** 无 `riscv_vector.h`（无 RVV intrinsics）+ binutils 拒 `zvfh`→本板 RVV kernel 实际部署编译器 = clang-17（合 memory [hardware-test-access]「gate4 用系统 clang17」）。**CLAUDE.md「rvv=gcc-15」前提在【本板】不成立**（出货 gcc-12.3.1 / clang-17）；G1 的 SpacemiT gcc-15.2 是本地代理，本 G2 板测静态账为 clang-17（真部署）。
- load-gate：每次测量前后 load ~2.10–2.22/64c（~3%）PASS·无 co-tenant vLLM/llama/bench·`taskset -c 8-15`。
- 编译：`clang-17 -march=rv64gcv_zvfh -mabi=lp64d -O3`。原始数据 `raw/results.txt`；驱动/内核源 `raw/*.cpp`；部署内核 emit `raw/q5_0_gevm_NEW.emitc.c`。

## 1. 发射器实装（REDESIGN-B·★绝对隔离已守）
- 文件：`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp`·**GEVM leaf `emitRepackQ4LaneWiseIntegerCore`（1506-1967）**。
- **改动**：把 `qhMaskScalar`+`expandQhBit`(per-lane splat/vid/vsrl_vv/vand/vsll/vncvt)+`assemble5`(vor/reinterpret/vsub) 换成 REDESIGN-B native-mask：`vlm_v_b16`(直装 mask bits) + `vmnand`(反相·q5_0) + `vsub_vx_i8mf2_mu`(mask,nib,nib,16)（5th-bit + −16 bias 融一条）。q5_1(offsetBias==0) = `vadd_vx_u8mf2_mu`(raw mask·省反相)+reinterpret。strip h 的 mask byte = qh_off + h*(half/8)（half=8→+h byte·riscv64 小端 => vlm byte-h ≡ OLD vid+8h 逐位相等）。调用点 1892-1911 换 `decodeQh5`。
- **★sealed GEMM leaf `emitRepackGemmQ4LaneWiseIntegerCore`（760-1188·含 909-966 qh 段）逐字未触碰**（git diff 证：仅 3 文件改·GEMM leaf 0 行 diff）。q5-0/q5-1 **GEMM prefill** lit 全过（走 sealed leaf）= 隔离实证。
- **只影响 hasQh 分支的调用者 = q5_0/q5_1 GEVM identity**（q5_K/q6_K/q3_K 走各自专用 re-emit handler；`five-bit-offset-binary` brick 走独立 flat block-dot emitter·均不经此 leaf）。

## 2. byte-exact 硬门（GREEN·STEP 2）
- lit：`Conversion/RVV` 全套 **242/242 PASS**（含 q5-0/q5-1 identity + q5-0/q5-1 GEMM prefill·零回归）。q5-0/q5-1 identity QH CHECK 改为 REDESIGN-B 序（vand→vlm_v_b16→vmnand/vadd_mu→reinterpret→vsub_mu·+ QH-NOT 退役旧链 vor/vncvt/vsrl_vv/vsll）。
- 板 oracle（ZERO-MODEL 式·独立 oracle=OLD 发射器 per-lane-expand·真不同算法）：OLD-vs-NEW 同 random 字节·**0 mismatch / 256,000 f32 输出**（3 seed/shape）。代数模型 q5_0 0/32·q5_1 0/32。
- 全套 weft lit 910 中 3 失败全在 `Scripts/rvv-generated-bundle-abi-e2e-*`（**独立 Python bundle 自测·与 q5/qh/weft-opt C++ emit 无关·pre-existing**·git diff 证 Python 脚本未动）。

## 3. cold GEVM M=1 micro（STEP 3·墙钟·rdcycle/rdinstret 本板 SIGILL·perf CSR 亦 gated→用 CLOCK_MONOTONIC）
工作量：N 行 × nb=K/32 块。NEW=1 次 repack-GEVM（整矩阵）·STOCK=N 次 per-row ggml block-dot。同 MAC·同 22 B/行/块权重足迹。内存流量 NEW ~12.6MB@(4096,4096) vs STOCK ~29.4MB（per-row 重读 act）。

| N×K | 足迹 | NEW ms(med/min) | OLD ms | STOCK ms | **stock/new**(路径) | **old/new**(decode净) | stock/old(路径·旧decode) |
|---|---|---|---|---|---|---|---|
| 4096×4096 | 11.5MB 内存墙 | 7.23/6.82 | 13.36/9.81* | 16.09/16.05 | **2.23/2.35** | 1.85/1.44* | 1.20/1.64 |
| 1024×4096 | 2.9MB | 1.705 | 2.844 | 4.013 | **2.35** | **1.67** | 1.41 |
| 512×512 | 0.18MB hot | 0.107 | 0.155 | 0.257 | **2.40** | **1.45** | 1.66 |
| 256×256 | 45KB hot | 0.026 | 0.038 | 0.065 | **2.51** | **1.44** | 1.73 |
(\* 4096²的 OLD 噪声 relIQR 0.21·取稳定 regime 为准；其余 relIQR<0.01)

**门结论 = ≥parity vs stock block-dot 决定性达标**（NEW 全 regime 2.23–2.51× 更快·真墙钟·稳定）。

### 3.1 vsetvli 归因裁决（G1 caveat RESOLVED）
- G1 caveat：NEW-vs-stock literal 1.047× 含 vset·「去-vset 纯工作 1.76×·约一半裕度来自 constant-vl 避 stock vset 税」——疑 NEW 优势是 vset 计数 artifact。
- **裁决**：真墙钟 settle 之。① NEW-vs-stock **2.2–2.5×**（≫ 1.047× 指令数）主由 **repack-GEVM 内存局部性**（act 重用·非 vset）→ vset 在墙钟头条里是次要子项。② **decode-isolated NEW-vs-OLD = 1.44–1.67×**（同足迹·**两侧皆 constant-vl repack-GEVM**·vset 非变量）→ 削重建（OLD 38→NEW 22 v-insn/块·G1 clang 静态账 whole-fn vset 14→7）**真省周期、非 artifact**。∴ 削减在 repack-GEVM 里【传导到墙钟】（decode 在关键路径·未被内存墙洗掉·区别「compute win washes out in memory-bound decode」——repack-GEVM 令其 compute-bound-on-decode）。
- vsetvli 静态数（clang-17 whole-fn objdump）：NEW 7 / OLD 14 / STOCK 10；NEW 热 nibble-step loop = constant vl=8（vset 提出 16-步循环外）·STOCK per-block vl-toggle(e8/m2↔e16/m4↔e32/m1)。
- 对手身份（符号机判）：`nm` = `T ggml_vec_dot_q5_0_q8_0`（真 ggml 全局符号）·body = 2×vwredsum + 2×vlm(_v_b4 qh) + 2×vwmul + vl-toggle = **canonical ggml block-dot·非 hand-brick**。

## 4. 板 restore 双证 + 无 stray
- **未触碰任何板库**（只写 scratch `/tmp/g7g2_run`·用自带 extract stock.cpp 编·无板装 ggml 被改）→ 无 board-lib md5 漂移面。governor/主树/build 未动。
- scratch 已 `rm -rf`·`ls` 确认消失·无 stray proc（仅 kernel thread kdevtmpfs）·load 回基线 2.10。co-tenant vLLM 未重启（本就未运行）。

## 5. 下一步 / corollary 建议（改留主会话）
- **q5_1**：同 leaf 自然铺开·byte-exact 已证（代数 0/32 + emit lit 绿·`vadd_vx_u8mf2_mu` raw-mask 省反相）。板 micro 未测（q5_1 fixture 走 MIN fold·同 decode 结构）→ **建议 G2b 补 q5_1 板 micro**（预期同 q5_0 量级·decode 更简）。
- **q5_K**：super-block 6-bit scale·headroom 更大但 decode 走**专用 q5_K handler（非此 leaf）**·需先移 K-quant scale 重建 = **结构级独立工作**（[K-10]·非此参数级铺开）。
- **系统账兑现**：本 G2 = **kernel-axis micro**（真墙钟）。perf-covered（e2e 系统账·T6 selector + e2e harness）仍是独立赛道·**禁互推**（memory measurement-offensive 铁律）。REDESIGN-B 的 e2e 兑现需真集成路由（部署≠证过·[CASE-MICRO-E2E]）。

**禁 git（主会话据本 casefile 入账）· sealed GEMM leaf 未触碰 · byte-exact 硬门 GREEN · 对手符号机判 block-dot · 板已 restore。**
