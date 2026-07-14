# G8 stage1.3 — kernel-axis clang flags 定稿  2026-07-14

## 定稿原则
1. **板内 ours == opp 同 clang 同 march 同 flags** (§三.2 硬要求·防 [CASE-COMPILER-ASYMMETRY] artifact)。
2. `-march` 取 **该 clang 版本稳定扩展 ∩ 板 cpuinfo 实测能力** 的上界 (逐项列全·不含板未实测扩展)。
3. gcc 数字 = 部署域附注列，**不参与主表胜负判定**。

## rvv (clang-17 域·VLEN128)

**定稿 march (推荐·板全覆盖)**：
```
-march=rv64gcv_zfh_zfhmin_zvfh_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zihintpause -mabi=lp64d -O2 -ffp-contract=on
```
- 取舍理由：
  - **剔 zfa / zicond / zvfhmin / zihintntl** — clang-17 视为 experimental，无 `-menable-experimental-extensions` 直接 REJECT (逐个实测确认)。
  - **剔 zicbop** — 板 cpuinfo **无** zicbop (只有 zicbom/zicboz)；clang 虽接受(hint-nop)，但违"覆盖板实测能力"原则，故不列。
  - **保 zfh/zfhmin** (板有·fp16 scale 解码) · **zicbom/zicboz** (板有·cache 管理) · **zawrs** (板有) · zvfh (向量 fp16) · zba/zbb/zbc/zbs (bit-manip)。
- **⚠ 对称性待办 (交主会话·stage-3 地基)**：板上现成 clang-17 ggml (`build-openeuler-clang17`) 是用 **近邻 march** `rv64gcv_zvfh_zba_zbb_zbc_zbs_zicbop_zihintpause` 编的 (缺 zfh/zfhmin/zicbom/zicboz·多 zicbop)。**要与定稿 march 严格对称，stage-3 须用定稿 march 重编 ggml-cpu** (单库·upstream-native 树可编)。zfh 可能影响 fp16 scale codegen → 不保证 codegen-neutral → 不能省重编。

**本役样例采用 march (严格对称·链路证)**：`rv64gcv_zvfh_zba_zbb_zbc_zbs_zicbop_zihintpause` (== 现成 .so 自身 build march·两侧完全一致)。样例仅证链路通，非定稿测量。

## k1 (clang-18 域·VLEN256)

**定稿 march (canonical·非-IME 格)**：
```
-march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -O2 -ffp-contract=on
```
- clang-18 zvfhmin/zvl256b 稳定，可选加 `_zvfhmin_zvl256b_zicbom_zicboz`；t4a canonical 历史即用此串，本役样例沿用 → 与既有 k1 kernel-sym 数可比。
- 对手 fair-recipe (stock repack.cpp.o) = `-O3 -march=rv64gcv_zfh_zvfh_zicbop_zihintpause`；对称测时 ours 也须同串。

**⚠ IME 域分离 (环境发现·非本役对称域)**：IME 格 (q4_0@ime/q8_0@ime/q4_K@ime) 的 march `rv64gcv_xsmtvdotii1p0` 被 **clang-18 拒** (`unsupported version 1.0 for extension 'xsmtvdotii'`)，仅 **gcc-13 接受**。→ **k1 clang-18 对称主表 = 非-IME RVV 格**；IME 格留在 **gcc-13/xsmtvdotii 真硅 cert 域** (与既有 N2 cert 绑 gcc-13 一致·非本役范畴)。
