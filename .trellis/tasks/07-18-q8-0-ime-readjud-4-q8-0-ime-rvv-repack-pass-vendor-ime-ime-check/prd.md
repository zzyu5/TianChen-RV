# PRD · q8_0@ime 改判（裁决4·对手=RVV-repack 回退·赢即 PASS·走 check）

> **权威** = 用户裁决4（2026-07-18）：「q8_0@ime 落账：批准回退即对手——对手=RVV-repack 回退·赢即 PASS·成色注『对手=回退非 vendor IME』·**禁写『赢了 IME』**。」+ 裁决5（改判走 check·正负对称）。
> **性质** = 改判（IME_KERNELSYM q8_0@ime「无合法对手」pending → 对手=回退·赢即 PASS）·**走 check**（裁决5）。**赢要真赢·数走 bench**。

## 一、背景
- `IME_KERNELSYM` dict q8_0@ime 现 = `(None,"该板无合法对手","q8_0不进vendor IME(dispatch ime.cpp:317仅q4_0/q4_1/q4_K)·落RVV-repack回退·结构无IME对手")`（pending·非 PASS）。
- q8_0 不进 vendor IME（dispatch 只 q4_0/q4_1/q4_K）→ 落 **RVV-repack 回退**。⟹ 裁决4：**对手 = RVV-repack 回退**（非 vendor IME）·q8_0@ime（我方 IME/wide-vmadot 路）赢回退即 PASS。
- e2e 侧 q8_0@ime beat stock **2.233×**（T6·perf-covered 9/83 绿·M7 wide-vmadot）——但 kernel-sym 轴现 pending。

## 二、你要做（k1 板·bench 唯一通道）
1. **objdump 对手**：确认 q8_0@ime 部署时**实跳 RVV-repack 回退**（非 vendor IME·dispatch ime.cpp:317 证）·回退核身份（`ggml_gemm_q8_0_*` repack？）。
2. **k1 板测**：q8_0@ime（我方 IME/wide-vmadot leaf）vs **RVV-repack 回退**·byte-exact G1（mism=0）+ cold 2-seed·**先测不预告**。
3. **verdict（数据定·赢要真赢）**：
   - q8_0@ime cold_X vs 回退 **≥0.8/真赢** → **改判 PASS**·成色注「对手=RVV-repack 回退·非 vendor IME」·**禁写『赢了 IME』**。
   - **不赢** → 维持 pending/具名（数据说话·不强推 PASS）。

## 三、成色（裁决4·硬约束）
- 对手 = **RVV-repack 回退**（非 vendor IME）——**禁写『赢了 IME』**（q8_0 结构上不进 vendor IME·赢的是回退路）。
- 若回退是**便宜档**（vs generic/未优化）→ 标 opp-immaturity·禁称硬赢；若回退是**部署真核**（RVV-repack 是真部署路）→ 真赢部署回退（系统账 framing·如实标对手身份）。
- e2e 2.233× beat stock 是**另一赛道**（perf-covered·已绿）·kernel-sym 改判**禁与 e2e 互推**（[CASE-COMPILER-ASYMMETRY]/两赛道）。

## 四、验收
1. **objdump 证 q8_0 实跳 RVV-repack 回退**（对手身份·非 vendor IME）。
2. **byte-exact GREEN**（k1·mism=0）+ **cold 2-seed**·verdict（真赢回退→PASS / 不赢→维持）。
3. **成色**：对手=回退（非 IME）·禁写赢了 IME·回退档位（便宜/部署真核）如实标。
4. **0 造数**·byte-exact 硬门·objdump 全量·**禁 commit·禁 add**（recon-dict 改判 + commit 由 main·走 check）·master/sealed 不动·证据 scratchpad。

## 五、触碰集 / 遗留
- 触碰：k1 板测（q8_0@ime vs 回退）+ objdump。**无源改**（度量+改判）。用 k1 板。
- **遗留**：改判 IME_KERNELSYM q8_0@ime = main 做（判据级·走 check）。verdict 定 IME_KERNELSYM q8_0@ime 措辞（PASS-对手回退 / 维持）。
