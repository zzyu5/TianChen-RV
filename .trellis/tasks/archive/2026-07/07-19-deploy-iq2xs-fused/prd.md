# PRD — 部署 iq2_xs fused reduction（realize proven 地盘·VLEN-keyed）

## 目标
把 iq2_xs floor 攻坚证明的 **归约 fusion 结构**部署进 `emitIQ2XSSuperBlockGridBody`（`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp`），使编译器发出它，iq2_xs vec_dot flip 具名-X(deployed 0.62)→PASS（VLEN128 0.82·手调-rvv +1），并**测 k1 VLEN256** 定其 verdict。

## 背景（proven·commit `e320d7ec6`）
iq2_xs floor 攻坚（task `07-19-block2-iq2xs-floor`）证明：deployed emit 的 **16× serial vwredsum** → fused **8 vwmacc + 1 vredsum**（fold ls1/ls2 per-half scale 进 i32 累加器）= floor 0.62→**0.82**（byte-exact GREEN·worst_ulp=0·15+seeds·**tq1_0 8→1 fusion 同族·commit 4e8d46948 先例**）。
- ★**proven fused = variant B**（`experiments/active/b-block2-iq2xs-floor/kernels/iq2_xs_fused_b.c`·**VLEN128-form**·8-gather·med 0.82）。
- ★**VLEN caveat**：VLEN-universal 变体 below gate（A 4-gather 0.49 regress·C vslidedown 0.75）·i32-accum LMUL8 32-lane cap 所致 → **PASS 是 VLEN128-form**·**须 per-VLEN capability key**（[K-10] 结构级·ggml 也 `_vl128/_vl256/_vl512` 分开·合法）。

## 施工（emitter-body·per-VLEN-keyed）
改 `emitIQ2XSSuperBlockGridBody` 发 fused 归约结构·**per-VLEN key**：
- **VLEN128**：发 variant B fused（8-gather·16 vwredsum→8 vwmacc+1 vredsum·参考 `iq2_xs_fused_b.c`）·预期 0.82 PASS。
- **VLEN256（k1）**：**先测**——variant B 是否适用 VLEN256·或需 VLEN256 专化 form（A/C 已 below gate·可能需新 form 或维持既存）。**按板测数据定 k1 form**（若达 0.8 flip·若 below-gate 维持既存/具名-X 诚实）。
- owned C-intrinsic·🔴 无 inline-asm·VLEN-keyed 是结构级 [K-10]（非旋钮）。

## 门
- **byte-exact 输出门**（correctness·非 regen-diff-0·emit 故意改）：新 emit 输出 byte-exact 等价 oracle（mism=0·整数核 0 容忍·Q8_K scale ULP 界）·两 VLEN 各对拍·用攻坚 driver 验。
- **lit CHECK 更新**（iq2_xs export-e2e/block-dot lit·emit 变）·全 lit 绿。
- **board rvv + k1·2-seed（稳则5）·compiler-symmetric**（opp clang18-symmetric·手调 `_vl128`/`_vl256`）·verdict cold≥0.8?PASS:具名-X·CORE==PROD（导出 fused emit md5·fixture diff）。

## sealed 核
- `RVVToEmitCGridCodebook.cpp` md5 **WILL change**（改 iq2_xs emit body·预期）·收尾报新 md5+diff·**只改 iq2_xs vec_dot body**（别碰其他 grid emit 路·iq2_xxs/iq3/iq1 等）。

## 成色（诚实钉死）
- 对手手调（非便宜档）·deployed fused ~0.82 = **near-parity PASS-by-gate 非 beat**（cold≤1.0·opp~1.2×快·过 0.8 门=地盘）。M=1 勿外推 e2e。
- 达 PASS = 手调-tier vec_dot deployed 地盘 via 真 emitter 改进（第 4 个第二块公式墙格实现·serial-reduce floor 可裂坐实）。**k1 若 below-gate → 诚实具名-X**（VLEN256 form 未达·非架构墙·per-VLEN 数据事实）。

## 交付（首节三项）
1. iq2_xs vec_dot deployed verdict per board（rvv flip PASS / k1 数据事实）·新 sealed md5·byte-exact 两 VLEN·lit 状态·CORE==PROD。
2. 未试杠杆（若 k1 below-gate·具名 VLEN256 form 差异/读不到的量）。
3. 论文侧（rvv flip 手调-rvv +1·serial-reduce floor-crack deployed 坐实·per-VLEN key 素材）。

## 账面纪律
- headline flip 走主会话独立 check·正负对称（build+lit+board+CORE==PROD·per-VLEN byte-exact）。
- 不 git commit·git add 只纳源勿纳 build/worktree·worktree base 核对（当前 tip `16d2e7619`·若旧线 reset）。
- master 入账走 `recon_master_rebuild.py` `VECDOT_KERNELSYM` dict（both-board·主会话·参照 tq1_0 条目）。
- 返回结构化：新 sealed md5+diff·per-board verdict+byte-exact(mism)·两 VLEN·board cold(2/5seed 编译器对称)·CORE==PROD·lit·(若 k1 below-gate)数据事实。

## 参照
- proven fused：`experiments/active/b-block2-iq2xs-floor/kernels/iq2_xs_fused_b.c`（variant B·headline）+ `MANIFEST.md`/`GEN_SEAL.txt`。
- tq1_0 部署先例（同 fusion 部署·VLEN-keyed 参考）：commit `4e8d46948`·task `07-19-deploy-tq10-fused-vecdot`。
- 目标 emit：`RVVToEmitCGridCodebook.cpp emitIQ2XSSuperBlockGridBody`。
- ISSUE-112（簇A·iq2_xs proven candidate）· canon [K-10]（per-VLEN 结构级 key·非旋钮）。
