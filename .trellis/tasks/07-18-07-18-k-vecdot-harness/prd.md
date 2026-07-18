# PRD · K线 · vec_dot.sh K-quant harness 建设（解锁手调攻坚面）

> **权威** = 《开测篇》§二.3-4（两档硬收口·手调攻坚按机制列队）+ K-actionable-queue §3。
> **性质** = harness 建设（「工」·非 gated）·解锁 K-quant vec_dot 攻坚面（现最大非 PASS 群无专用 harness）。

## 一、目标

建 `tools/bench/cells/vec_dot.sh` harness 供 **K-quant vec_dot 攻坚**（q2_K/q3_K/q4_K/q6_K vec_dot·手调档·全非 PASS）。
本 task = **建 harness + baseline 测 1-2 格**（不求翻·先接线·让后续逐格完整攻坚环有通道）。

## 二、★命名先厘清（ISSUE-104 教训·判据级警示）

scalar 板 vec_dot 用 `scalar_vec_dot.sh`（避冒充全板）。本 harness = rvv/k1 vec_dot K-quant。
命名 `vec_dot.sh` 是否与 runner `cell_harness(op)`（op=vec_dot→`vec_dot.sh`）相接·会否与 `scalar_vec_dot.sh`
冲突/覆盖语义（harness 粒度：按 op 全板 vs 按 op×板 = [ISSUE-090] 遗留 facet）。
- **有夹 → 登记 ISSUE**（harness 粒度/命名规则 = 判据级·**禁擅定**）+ 用能接线命名先建·保守默认续推。

## 三、harness 契约（同 dequantize_row.sh）

- 签名 `vec_dot.sh <board> <mode> <fmt>`·op=vec_dot·**禁写仓库侧文件**·全 stdout·经 runner 落三目的地。
- vec_dot K-quant = 量化点积 kernel·oracle = **ZERO-MODEL**（[K-5]·原始字节重算）·**fp-contract 对称**
  （有 fp reduction·同 scalar S1 惯例）·cold N=25 2-seed·zero-vector 探针·反空心·对手身份探针。
- **对手 = ggml 手调 vl128/vl256（手调档·非便宜档·成色如实）**。

## 四、验收

1. **命名厘清结论**（登记 ISSUE?·用能接线命名）· harness 契约合规（禁写仓库文件·全 stdout·+x）。
2. **baseline 测 1-2 格**（如 q4_K/q6_K @rvv）· `verify` byte-exact GREEN + zero-vector 探针 · `measure` cold 落 runs/·建基线供后续攻坚 · verdict 照测标（现多具名-X·**不求翻·本 task 是接线**）。
3. **成色如实**（手调档对手·非便宜档）· 入账留 main 会话（真数落 runs/·不直写 master）。
4. **0 造数**·byte-exact 硬门实证·sealed 不动·**禁 commit·禁 add -A**·"某物不存在"禁截断命令。

## 五、触碰集 / 遗留

- 触碰：`tools/bench/cells/vec_dot.sh` + K-quant oracle/driver（与 scalar_vec_dot.sh 不同文件·与并行线 product_reduce.sh / dequant emit 不相交）。
- **本 harness 就绪后 → 后续 task 逐格走完整攻坚环**（K-quant vec_dot·机制②③ gcc→clang + weight-reconstruction）。
