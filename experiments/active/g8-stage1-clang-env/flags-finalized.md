# G8 stage1.3 — kernel-axis clang-18 flags 定稿  2026-07-14 (口径统一：双板 clang-18)

## 定稿原则
1. **板内 ours == opp 同 clang-18 同 march 同 flags**（§三.2 硬要求·防 [CASE-COMPILER-ASYMMETRY]）。
2. `-march` = 该 clang 稳定扩展 ∩ 板 cpuinfo 实测能力 的上界（逐项列全）。
3. gcc 数字 = 部署域附注列·不参与主表胜负。IME 格 carve-out 出 clang-18 域（见下）。

## rvv (clang-18·VLEN128)

**编译器 (canonical)**：`/opt/tcrv-toolchains/llvm-18.1.8/bin/clang` (18.1.8)，`source /opt/tcrv-toolchains/env.sh` 后
PATH 含 binutils-2.46.1(as) + gcc-15.2.0(runtime)。

**定稿编译 flags (ours 与 opp 两侧完全一致)**：
```
-O2 -march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause \
    -mabi=lp64d -fno-integrated-as -ffp-contract=on
链接追加: --gcc-install-dir=/opt/tcrv-toolchains/gcc-15.2.0/lib/gcc/riscv64-unknown-linux-gnu/15.2.0
```
- **march 全板覆盖**：clang-18 下全板 compute 扩展稳定（含 clang-17 拒的 zvfhmin/zicond/zfa/zihintntl）→ 逐项列全无遗漏；剔板 cpuinfo 未列的扩展（如 zicbop）。
- **`-fno-integrated-as` 为何必带（两侧对称）**：对手部署树(vericurve) quants.c 有手写 policy-less inline-asm `vsetivli zero,16,e8,m1`（无 ta/ma）·clang-18 集成汇编器【拒】(clang-17→18 回归)。转 binutils-2.46.1 gnu-as(认 rich march 且接受 policy-less)。ours(纯 intrinsic) 带不带都编过·但为两侧 flags 逐字符对称·统一带。**codegen 不变**(仅 asm 阶段换汇编器)。
- **`--gcc-install-dir`**：canonical clang 默认 triple=riscv64-unknown-linux-gnu·须显式指 gcc-15.2.0 找 CRT(crtbeginS/crtendS)+libgcc；仅链接可执行时需要·-c 编 .o 不需。

## k1 (clang-18·VLEN256·非-IME 格)

**编译器**：`/usr/bin/clang` (Bianbu 18.1.8)。**定稿 march (canonical)**：
```
-O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on
```
- k1 样例 q4_K 用集成汇编器直接编过·**不需** -fno-integrated-as。可选加 `_zvfhmin_zvl256b`。
- 对手 fair-recipe (stock repack.cpp.o) = `-O3 -march=rv64gcv_zfh_zvfh_zicbop_zihintpause`；对称测时 ours 同串。

## ★IME 域 carve-out（不在 clang-18 主表·不在 0.8 硬门分母）
IME march `rv64gcv_xsmtvdotii1p0` clang-18 **拒**(`unsupported version 1.0`)·仅 gcc-13 接受。
→ IME 格 (q4_0/q8_0/q4_K @ime) 留 **gcc-13/xsmtvdotii 真硅 cert 域**·不参与 clang-18 主表胜负。
硬门分母 = matmul kernel-sym + forward-op·**IME 格不在其中** → carve-out 合法（同 [CASE-COMPILER-ASYMMETRY] 家族·已同步用户）。
