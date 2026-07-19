# PRD — W2 correctness: ISSUE-120 修（iq2_xs/iq2_s vec_dot VLEN256 byte-broken）

## 哲学锁
① 门=0.8·vs 部署≥0.8=赢。② 负结果非贡献。③ **修缺口不撤·遇挫不退·输了修了再打**。

## 目标（红债·W5 扩 iq2 真赢面的前置）
修 `iq2_xs`（+likely `iq2_s`）vec_dot 在 **VLEN256(k1) byte-broken**（ISSUE-120·`ours_vs_int_oracle=false`·mism=512·pre-existing·根因 = pair-batched `vget_i8m4_i8m1` 结构 VLEN128-form·LMUL8 32-lane 假设·VLEN256 崩）。修完 = k1 byte-exact GREEN → 解锁 W5 把 K-quant/tiny-codebook 宽化杠杆往 iq2 族延。

## 病灶（ISSUE-120 + census）
- deployed iq2_xs vec_dot serial emit（`emitIQ2XSSuperBlockGridBody`·`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp`）在 VLEN256 byte-broken。
- 根因 = pair-batched `vget_i8m4_i8m1` grid-decode 结构假设 LMUL8=32 lane（VLEN128 成立·VLEN256=64 lane 崩）。
- iq2_s 同 pair-batched 结构·likely 同病（先机核确认）。

## 施工（correctness·byte-exact 是唯一门）
- **per-VLEN 专化**（[K-10] 结构级·合法·ggml 也 `_vl128/_vl256` 分开）：为 VLEN256 发正确的 grid-decode 结构（gather/slide 适配 64-lane），或 **unbatched VLEN-universal 路**（放弃 pair-batch·换 VLEN-universal 累加器·可能慢但 correct）。按哪个 byte-exact 且不崩其他 VLEN 定。
- 🔴 owned C-intrinsic·无 inline-asm。**只改 iq2_xs/iq2_s vec_dot body**（别碰 iq2_xxs/iq3/iq1 等其他 grid emit·收尾报这些 md5 未变）。
- **VLEN128 不得回归**（iq2_xs@rvv 现 byte-exact·修 VLEN256 别破 VLEN128）。

## 门（byte-exact 硬前置·先证·禁计时）
- **VLEN256 byte-exact**：ZERO-MODEL·`ours_vs_int_oracle=true`·mism=0·worst_ulp=0（整数核 0 容忍）·3-arm anti-hollow·CORPUS COMPLETE。
- **VLEN128 不回归**：iq2_xs@rvv VLEN128 仍 byte-exact（改前后对拍）。
- **两 VLEN 各对拍**（export 两 VLEN·各自 vs oracle）。
- build + lit 绿（953/956 基线）。CORE==PROD（导出 emit md5·fixture diff）。
- 板测（byte-exact 过后·若测 perf）：compiler-symmetric·但**本 task 主门=correctness·perf 是 W5 的事**。

## 判决实验（correctness 自证）
构造 VLEN256 fixture·weft-opt export iq2_xs/iq2_s vec_dot·driver 跑 mism=0（修前 mism=512 → 修后 mism=0·pinned·可复跑）。记修前/修后 mism。

## 交付（首节三项）
1. iq2_xs（+iq2_s）VLEN256 byte-exact 修完（mism 512→0·pinned）· 修法（per-VLEN 或 unbatched）· 两 VLEN 各 byte-exact · sealed 其他 grid emit md5 未变。
2. 未修尽的（若 iq2_s 更难）· 具名。
3. 论文侧：iq2 族 k1 correctness 恢复·解锁 W5 iq2 扩面。

## 账面纪律
- 不 git commit·git add 只纳源·勿纳 build/worktree·worktree base（`d55f9ab4e`·旧线 reset）。sealed 核 md5 变须报（改 iq2_xs body·预期）。
- 三闸①：提交时三 grep 体检（裸格式字面量/板值+value_or/march 解析·相对基线只减不增·新代码出生即干净）。
- 返回结构化：修法·两 VLEN mism·修前后 pinned·sealed md5·lit·CORE==PROD。

## 参照
- ISSUE-120（`.trellis/spec/issues/发射器与架构.md`）· iq2_xs 攻坚档 `experiments/active/b-block2-iq2xs-floor/`（VLEN caveat）· iq2 P2 `archive/2026-07/07-19-block2-p2-iq2-vecdot/`。
- 目标 emit：`RVVToEmitCGridCodebook.cpp emitIQ2XSSuperBlockGridBody`。
