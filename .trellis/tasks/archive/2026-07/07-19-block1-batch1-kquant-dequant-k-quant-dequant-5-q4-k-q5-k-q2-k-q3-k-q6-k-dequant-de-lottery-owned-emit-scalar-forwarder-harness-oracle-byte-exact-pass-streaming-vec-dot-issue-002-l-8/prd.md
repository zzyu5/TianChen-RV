# PRD · 第一块批1 · K-quant 超块 dequant 收割（扇 5·de-lottery owned emit）

> **权威** = 第一块收割 scope（aad7·`07-19-block1-harvest-scope` research·批1 = K-quant 超块·扇 5·最高）+ 行动书 B 线第一块（收割·最大最便宜·已证模式复制）。
> **性质** = de-lottery owned emit 收割（[L-8] 强义 construction·关 ISSUE-002 敞口）。**成色 = 便宜档**（对手 autovec scalar-C=opp-immaturity·收割价值=构造轴 [L-8]+关 ISSUE-002·**非 perf 硬赢**）。**允许诚实缩 scope**（byte-exact 未达该格具名·不硬凑）。

## 一、核心（aad7 已定·勿误植 vec_dot 墙）
K-quant **dequant = streaming·无归约** ⟹ vec_dot 的 vwredsum EXHAUSTED / weight-reconstruction floor（reduction 墙）**不适用**。K-quant dequant 应像 q8_0 干净 de-lottery（**几乎必 PASS**·byte-exact）。发射器 dispatch/verifier/decode_model 门**已全 wire**·收割只需：**把 K-quant dequant 的 `...BodyShared` scalar-forwarder 换 owned 真向量 body + 建各族 harness/kernel/driver/oracle**。

## 二、你要做（扇 5·q4_K/q5_K/q2_K/q3_K/q6_K dequant@rvv）
1. **owned 真向量 dequant emit**（K-quant super-block 位解包·复刻 q8_0 de-lottery 范式·**owned 确定性·非宿主 autovec 抽签**·[L-8] 强义）。**先 q4_K/q5_K dequant**（census F7·素材最厚）·再 q2_K/q3_K/q6_K。
2. **建 K-quant dequant harness/oracle**（`dequantize_row.sh` 扩 K-quant·ZERO-MODEL oracle [K-5]·对 ggml 真核 byte-exact·streaming decode-only）。
3. **★byte-exact FMA-contract 每族前置**（K-quant dequant 的 fp16-scale fold·byte-exact 硬门·未达该格具名·不硬凑）。
4. **板测**（rvv·`dequantize_row.sh rvv verify/measure <fmt>`·2-seed·先测不预告）。

## 三、成色（赢要真赢·硬约束）
- 对手 = 部署 `dequantize_row_<fmt>` **autovec scalar-C = opp-immaturity 便宜档**（[§三.12]）。**收割价值 = 关 ISSUE-002 codegen-lottery 敞口 + [L-8] 强义 construction**（owned 真向量确定性·非宿主抽签）·**非 perf 硬赢**。若 de-lottery 后 cold≥parity/PASS·标 opp-immaturity（同 q8_0 dequant 2.33 便宜档范式·禁称硬赢强手调）。

## 四、验收
1. **owned 真向量 dequant emit**（objdump 证 owned 向量·非 autovec 抽签·[L-8] provenance）·换 scalar-forwarder。
2. **byte-exact GREEN**（[K-5] ZERO-MODEL·对 ggml 真核·per 格 mism=0·FMA-contract 一致）。**未达 byte-exact 该格具名·不硬凑**（诚实缩 scope）。
3. **cold 2-seed**（收割 verdict·PASS/parity 标 opp-immaturity 便宜档·关 ISSUE-002 该格敞口·[L-8]）。
4. **构造轴收割计数**（几格 owned+byte-exact 落地·certified 增·[L-8] 强义·报 main）。
5. **0 造数**·byte-exact 硬门·**worktree·禁 commit·禁 add**（main cherry-pick·worktree commit 报 hash·**git add 只纳源·勿纳 build-wt**）·sealed/master 不动·数字报 main·forced clean rebuild。

## 五、触碰集 / 遗留
- 触碰：K-quant dequant emit（`RVVToEmitCForwardElementwise.cpp` 或 KQuant dequant 路的 `...BodyShared` scalar-forwarder→owned body）+ `dequantize_row.sh` K-quant 扩 + oracle/kernel/driver。**worktree·rvv 板**。
- **遗留**：批2 tiny-codebook（iq4_xs·mxfp4 3.47× vrgather 正锚·扇 4）· 批3 ternary（tq2_0·扇 2）。@k1 dequant gated ISSUE-105。**须新 schema=停报 main**（dequant 门已 wire·预期无需）。
