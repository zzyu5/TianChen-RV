# PRD — 部署 P1 fused tq1_0 vec_dot leaf（realize 手调-tier deployed 地盘）

## 目标（realize 地盘·真 emitter 改进·非旋钮）
把 P1 攻坚证明的 **fused tq1_0 vec_dot 结构**部署进发射器，使**编译器发出**该 leaf，tq1_0 vec_dot 从 deployed-具名-X（0.22/0.549）flip 到 **PASS（~0.90/0.82·手调-tier 真硬碰强手调地盘·byte-exact·VLEN-universal）**。

## 背景（三个 task 的结论）
- **deployed emit 现状**（task `07-19-block2-tq10-deployed-measure`·commit `8d4e59823`）：live path `emitTypedSuperBlockScalarDeltaGridLoopBodyTQ10`（`lib/Conversion/RVV/RVVToEmitCTernaryBinary.cpp`·非退休的 `emitTQ1_0Q8_KBlockDot`@1958）**genuinely loses**：直接 M=1 板测 rvv 0.22/k1 0.549（<0.8）。**机制 = aux8[256] scratch 存(vse8)/reload(vle8) + 8× serial per-super-block vwredsum**（insn~269）。
- **P1 fused 参考实现**（task `07-19-block2-p1-tq10-vecdot`·commit `03dbc8a4f`·`experiments/active/b-block2-tq10-vecdot/kernels/tq1_0_vecdot_universal.c`）：**单 i16 累加器 + 单 vwredsum**（insn~181·消除 aux8 scratch + serial 归约链）·**~4× 快(rvv)/~1.5×(k1)**·**byte-exact 同结果**·**VLEN-universal**（128+256 皆 byte-exact 皆过门·cold 0.90/0.82）。
- **判定**：P1 leaf 是**真改进非冗余**（deployed 真输·非 proxy 低估·区别 iq2）。P1 standalone = 部署参考。

## 施工（emitter-body 改进）
改 `emitTypedSuperBlockScalarDeltaGridLoopBodyTQ10`（RVVToEmitCTernaryBinary.cpp）使发出 **P1 fused 结构**：
- **消除 `aux8[256]` scratch 存/reload**（trit 解包后直接进累加器·不落 scratch memory）。
- **8× serial per-super-block vwredsum → 单 vwredsum**（单 i16m4 累加器·末 1 reduce·对位 P1 universal）。
- base-3 `vwmulu.vx` powers-of-3 解包 + q8 预展宽 i16 + `vmul_vv`/`vmacc_vv` 累加 + qh 单遍 + 单 vwredsum（P1 universal 的结构·§P1 交付 owned leaf 结构）。
- **VLEN-universal 守**（P1 v2 lean 揭示 `_vl128`-式 vget-fold 的 VLEN128-only 陷阱·单累加器规避·必保 128+256 皆 byte-exact）。
- 🔴 **owned C-intrinsic·严禁 inline-asm/钉死调度**（P1 leaf 是 C-intrinsic·公式墙对位）。

## 门（硬前置）
- **byte-exact 输出门（correctness·0 容忍）**：新发出的 emit 的**输出**须 byte-exact 等价旧 emit / oracle（`mism=0`·整数核 0 容忍·Q8_K scale 浮点段 ULP 界）。⚠ 这是**输出 byte-exact 非 regen-diff-0**（emit 代码**故意改**·faster·输出不变）。用 P1 driver（`tq10_vecdot_driver.c`·ZERO-MODEL oracle·3-arm anti-hollow）验。
- **lit CHECK 更新**：tq1_0 vec_dot export-e2e / block-dot lit 的 CHECK 行须更新为 fused 结构（emit 变了）。全 lit 绿。
- **VLEN-universal 门**：128 + 256 两 VLEN 皆 byte-exact（export 两 VLEN·各自对拍）。

## 板测（byte-exact 过后）
- board = **rvv + k1**·**2-seed**（稳则 5-seed·像 P1/P2）·**compiler-symmetric**（我方 clang18·opp 用 clang18-symmetric 路 `build-clang18-rv64gcv`·[CASE-COMPILER-ASYMMETRY]）·opp = 部署手调 `ggml_vec_dot_tq1_0_q8_K_vl128/vl256`。
- 预期 ~0.90(rvv)/0.82(k1)（P1 numbers·deployed 应复现 standalone）·verdict cold≥0.8?PASS:具名-X。
- **CORE==PROD**：导出 deployed（fused）emit·record md5·核 export-e2e fixture core==prod diff=0。

## sealed 核
- **`RVVToEmitCTernaryBinary.cpp` md5 338a31bb WILL change**（本任务改 emit body·预期·非违规）。收尾报**新 md5 + diff 说明**（改了 tq1_0 vec_dot emit·消 aux8+serial 归约·输出 byte-exact 不变）。**别碰其他 emit 路**（只改 tq1_0 vec_dot 的 body）。

## 成色（诚实钉死）
- 对手 = **手调**（非便宜档）·**deployed fused ~0.90/0.82 = near-parity PASS-by-gate**·**非 beat**（cold≤1.0·勿称打赢手调·过 0.8 门=地盘）。
- 若 deployed 达 PASS = **手调-tier 首个 vec_dot deployed 地盘 via 真 emitter 改进**（区别 iq2_xxs 的既存 emit·此为主动改进）·+ C3′ VLEN-universal「换 VLEN 不换条目」坐实。

## 交付（首节三项）
1. **消灭待办 + 计数**：tq1_0 vec_dot deployed flip 具名-X→PASS(rvv+k1·手调-tier 地盘)·新 sealed md5·byte-exact 输出门 + lit + VLEN-universal 证。
2. **未试杠杆 + 责任人**：若 deployed 未达 P1 standalone 数（emit vs 手写 C 差异）·具名 gap 原因。
3. **论文侧过期读数**：手调-rvv/k1 头条 +（若双板 flip）·C3′ VLEN-universal 坐实（deployed·非仅 proven）。

## 账面纪律
- **headline flip(具名-X→PASS)走主会话独立 check 再入账·正负对称**（build+lit+board 复验·CORE==PROD）。
- **不要 git commit**·git add 只纳源·勿纳 build/worktree·worktree base 核对（从当前 tip `53e7cf096`·若旧线 reset）。
- master 入账走 `recon_master_rebuild.py`（CLANG_WORLD rvv + k1 需另法·主会话定位行键）。
- 返回结构化：新 sealed md5 + diff·byte-exact 输出门(mism)·两 VLEN byte-exact·board cold(2/5-seed·编译器对称)·CORE==PROD·lit 状态。

## 参照
- P1 fused 参考：`experiments/active/b-block2-tq10-vecdot/kernels/tq1_0_vecdot_universal.c` + `MANIFEST.md`（owned leaf 结构）。
- deployed 现状：`experiments/active/block2-p3-tq10-deployed-vecdot/`（aux8+8×vwredsum 机制·GEN_SEAL export recipe）。
- 目标 emit：`RVVToEmitCTernaryBinary.cpp emitTypedSuperBlockScalarDeltaGridLoopBodyTQ10`。
- ISSUE-112（簇A P1 disposition）· canon [K-10]（结构级改进·非旋钮）。
