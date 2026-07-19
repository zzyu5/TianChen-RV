# W3 出账 — 36 dequant 板格·真向量运算指令数（控制 vs 抽签）

**本流裁定：18 格控制 / 0 格抽签-赢（每板）· 跨 36 板格 = 36 控制 / 0 抽签。**

「36 dequant 已收·账未核」的 pending-核 → **核毕·账真**。所有 18 格式（2 板）均 >2 条真向量运算，且真运算为货真价实的
dequant 计算管线（vand/vsrl/vzext/vsext + vfcvt/vfwcvt + vfmul/vfmadd/vfmsac，FP4/IQ 另加 vrgather / vluxei 查表-gather）。
**无一格是 codegen 抽签**（host autovec 只出 2 条 vle/vse 的情形一格未见）。⟹ **PR-31（dequant 真向量发射器）对这 18 格不必做**：它们已是我方发射控制，可喂 C1/C2 锚。

## 判据（PRD §做）
- 真向量运算指令 = objdump 反汇编中所有 `v*` 助记符 **减去 `vset*`（配置指令不计）**。
- **REALVEC ≤ 2**（只剩 vle/vse 设长）= **codegen 抽签**（host autovec·非我方控制真运算）。
- **REALVEC > 2**（vand/vsrl/vzext/vfwcvt/vfmul/vfmacc/vwmacc/vluxei 等真运算）= **我方发射控制**。
- REALVEC 为精确裁决量；COMPUTE/LDST 拆分为参考（vl1r/vs1r 整寄存器 spill 被计入 compute·不影响 REALVEC 与裁决）。

## Provenance / pin（可复跑）
- **编译器（钉死·两板同一份）**：`/opt/tcrv-toolchains/llvm-18.1.8/bin/clang++` → `clang version 18.1.8` (Target riscv64-unknown-linux-gnu)
- **反汇编器**：`objdump` GNU Binutils 2.41
- **编译主机**：`ssh rvv`（riscv64·localhost.localdomain）— 交叉码生成与主机无关，`.o` 仅由 (march, 编译器) 决定。
- **march（板别 = 唯一变量）**：
  - `rvv`（VLEN128）：`rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause`
  - `k1`（VLEN256·SpacemiT X60）：`rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs`
- **kernel 源（只读·未改）**：`experiments/active/r-dequant/kernels/<fmt>_dequant.c`（18 格·含 grid 变体）。含 `extern "C"` → 以 C++ 编（`clang++`，仅 `-Wdeprecated` 警告·无编译错）。
- 两板 `.o` md5 相异（march 差异化 ELF），但**真向量指令计数逐格全等**（dequant 发射管线不随这两 march 分叉）⟹ 裁定板不变。
- 原始证据：`research/evidence/counts.csv`（36 行机算原表）+ `q8_0.rvv.dis`/`q4_0.rvv.dis`（抽样反汇编）+ `q8_0.rvv.hist`（助记符直方图）。

## 提取命令（机算断言随附）
编译 + 反汇编：
```
CXX=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang++
MARCH_rvv="rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause"
MARCH_k1="rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs"
$CXX -O3 -march=$MARCH_<board> -mabi=lp64d -c <fmt>_dequant.c -o <fmt>.<board>.o
objdump -d <fmt>.<board>.o > <fmt>.<board>.dis
```
逐格计数（vset* 不计·助记符 = 制表符第 3 字段）：
```
vectot=$(awk -F'\t' 'NF>=3{print $3}' <dis> | grep -cE '^v')
vset=$(  awk -F'\t' 'NF>=3{print $3}' <dis> | grep -cE '^vset')
realvec=$((vectot - vset))            # 精确裁决量
# 真运算助记符清单（排除 vset*）：
awk -F'\t' 'NF>=3{print $3}' <dis> | grep -E '^v' | grep -vE '^vset' | sort -u
```
「某物不存在」用精确谓词（`grep -c`），未用 head/窗口截断：抽签格数 = `grep -c ,lottery, counts.csv` → **0**。

## 逐格真向量指令数表（36 板格）

| 格式 | 板 | vec总 | vset | **REALVEC** | compute | ld/st | 判定 | 真运算助记符（排除 vset*） |
|---|---|---:|---:|---:|---:|---:|---|---|
| iq3_xxs | rvv | 250 | 41 | **209** | 160 | 49 | 控制 | vand, vfcvt, vfmul, vluxei16, vlse8, vsseg8e8, vslide1down, vmsne, vrsub, vsext, vsll, vzext, vle32/8, vse32, vmv1r |
| iq3_xxs | k1 | 250 | 41 | **209** | 160 | 49 | 控制 | 同上 |
| iq3_xxs_grid‡ | rvv | 147 | 33 | **114** | 80 | 34 | 控制 | vand, vfcvt, vfmul, vluxei16, vluxei8, vmsne, vrsub, vsext, vsll, vzext, vle16/8, vse16/32 |
| iq3_xxs_grid‡ | k1 | 147 | 33 | **114** | 80 | 34 | 控制 | 同上 |
| iq4_nl | rvv | 19 | 5 | **14** | 9 | 5 | 控制 | vand, vsrl, vsext, vrgather, vfcvt, vfmul, vle8, vse32 |
| iq4_nl | k1 | 19 | 5 | **14** | 9 | 5 | 控制 | 同上 |
| iq4_xs | rvv | 138 | 33 | **105** | 72 | 33 | 控制 | vand, vsrl, vsext, vrgather, vfcvt, vfmul, vle8, vse32 |
| iq4_xs | k1 | 138 | 33 | **105** | 72 | 33 | 控制 | 同上 |
| mxfp4 | rvv | 19 | 5 | **14** | 9 | 5 | 控制 | vand, vsrl, vsext, vrgather, vfcvt, vfmul, vle8, vse32 |
| mxfp4 | k1 | 19 | 5 | **14** | 9 | 5 | 控制 | 同上 |
| nvfp4 | rvv | 75 | 17 | **58** | 40 | 18 | 控制 | vand, vsrl, vsext, vrgather, vfcvt, vfmul, vl1r, vs1r, vle8, vse32 |
| nvfp4 | k1 | 75 | 17 | **58** | 40 | 18 | 控制 | 同上 |
| q2_K | rvv | 160 | 32 | **128** | 80 | 48 | 控制 | vand, vsrl, vzext, vfcvt, vfmsac, vfmv.v.f, vle8, vse32 |
| q2_K | k1 | 160 | 32 | **128** | 80 | 48 | 控制 | 同上 |
| q3_K | rvv | 256 | 32 | **224** | 144 | 80 | 控制 | vand, vsrl, vsll, vsub, vrsub, vzext, vfcvt, vfmul, vle8, vse32 |
| q3_K | k1 | 256 | 32 | **224** | 144 | 80 | 控制 | 同上 |
| q4_0 | rvv | 17 | 4 | **13** | 9 | 4 | 控制 | vand, vsrl, vzext, vadd, vfcvt, vfmul, vle8, vse32 |
| q4_0 | k1 | 17 | 4 | **13** | 9 | 4 | 控制 | 同上 |
| q4_1 | rvv | 16 | 4 | **12** | 8 | 4 | 控制 | vand, vsrl, vzext, vfmadd, vfmv.v.f, vfcvt, vle8, vse32 |
| q4_1 | k1 | 16 | 4 | **12** | 8 | 4 | 控制 | 同上 |
| q4_K | rvv | 76 | 24 | **52** | 36 | 16 | 控制 | vand, vsrl, vzext, vfwcvt.f.xu.v, vfmsac, vfmv.v.f, vle8, vse32 |
| q4_K | k1 | 76 | 24 | **52** | 36 | 16 | 控制 | 同上 |
| q5_0 | rvv | 29 | 5 | **24** | 18 | 6 | 控制 | vand, vsrl, vsll, vor, vadd, vid, vmv.v.x, vzext, vfcvt, vfmul, vle8, vse32 |
| q5_0 | k1 | 29 | 5 | **24** | 18 | 6 | 控制 | 同上 |
| q5_1 | rvv | 28 | 5 | **23** | 17 | 6 | 控制 | vand, vsrl, vsll, vor, vadd, vid, vmv.v.x, vzext, vfmadd, vfmv.v.f, vfcvt, vle8, vse32 |
| q5_1 | k1 | 28 | 5 | **23** | 17 | 6 | 控制 | 同上 |
| q5_K | rvv | 109 | 24 | **85** | 60 | 25 | 控制 | vand, vsrl, vsll, vor, vzext, vfwcvt.f.xu.v, vfmsac, vfmv.v.f, vle8, vse32 |
| q5_K | k1 | 109 | 24 | **85** | 60 | 25 | 控制 | 同上 |
| q6_K | rvv | 224 | 32 | **192** | 120 | 72 | 控制 | vand, vsrl, vsll, vor, vsub, vzext, vfcvt, vfmul, vle8, vse32 |
| q6_K | k1 | 224 | 32 | **192** | 120 | 72 | 控制 | 同上 |
| q8_0 | rvv | 6 | 1 | **5** | 3 | 2 | 控制 | vsext.vf4, vfcvt.f.x.v, vfmul.vf, vle8, vse32 |
| q8_0 | k1 | 6 | 1 | **5** | 3 | 2 | 控制 | 同上 |
| tq1_0 | rvv | 247 | 57 | **190** | 133 | 57 | 控制 | vsrl, vzext, vsext, vadd, vmul, vfcvt, vfmul, vle8, vse32 |
| tq1_0 | k1 | 247 | 57 | **190** | 133 | 57 | 控制 | 同上 |
| tq2_0 | rvv | 156 | 32 | **124** | 80 | 44 | 控制 | vand, vsrl, vsext, vadd, vfcvt, vfmul, vle8, vse32 |
| tq2_0 | k1 | 156 | 32 | **124** | 80 | 44 | 控制 | 同上 |

‡ `iq3_xxs_grid` 为 grid 变体（W4 域）。本流仅**只读编译/反汇编其 kernel .c**——未触碰 grid emitter 源。列出仅为凑满 kernels/ 中 18 个 `*_dequant.c` 全集；裁定同为控制。

## 最小真运算 = q8_0（5 条）·仍为控制（抽样验真）
```
  12: 02058407  vle8.v      v8,(a1)      # 载 32×i8
  1a: 4a82a857  vsext.vf4   v16,v8       # i8→i32 符号扩展（真运算）
  1e: 4b019457  vfcvt.f.x.v v8,v16       # i32→f32（真运算）
  22: 9287d457  vfmul.vf    v8,v8,fa5    # ×scale（真运算）
  26: 02066427  vse32.v     v8,(a2)      # 存 32×f32
```
REALVEC=5 > 2 → 控制。3 条真计算（vsext/vfcvt/vfmul），非「2 条 vle/vse 抽签」。

## 结论（供 W1 / 主会话）
- **本波无「大批抽签」**：36/36 板格控制·0 抽签。dequant 发射器早已是我方显式 RVV 计算 intrinsic（clang 忠实下译为真向量指令），非 host autovec 抽签。
- **PR-31（dequant 真向量发射器·把抽签格转控制）对这 18 格无待办**——它们本就是控制态。若 PR-31 另有目标（如把标量 fp16 scale 也向量化、或压缩 vset* 数），那是别的账，不在「抽签→控制」这条。
- 成色注：「控制」= 源码显式发射 vand/vsrl/vzext/vfwcvt/vfmul/vluxei/vrgather 等真运算·被 clang 1:1 下译。这是真 owned emit，**可喂 C1/C2 锚**。
- 账面：0 造数·每条机算断言随附提取命令·「抽签不存在」用 `grep -c ,lottery, = 0` 精确谓词（非 head 截断）。
