# G8 stage1.3 — 双板编译器身份对表 (kernel-sym 对称域·clang-18 统一)  2026-07-14

**口径统一裁定 (2026-07-14)**：双板 kernel-sym 主表对称域 = **clang-18**（一个版本一张表）。
rvv 板无 clang-18 二进制包（openEuler dnf 仅 clang-17.0.6）→ 项目已在 `/opt/tcrv-toolchains` provision **canonical clang-18.1.8**（env.sh 挂载）。
gcc 数字 = 部署域附注列（不参与主表胜负）。clang-17 数据保留作环境备份·不入主表。

| 键 | rvv 板 | k1 板 |
|---|---|---|
| **主表对称编译器** | **clang 18.1.8** (project-canonical) | **clang 18.1.8** (Bianbu `11bb4`) |
| clang 路径 | `/opt/tcrv-toolchains/llvm-18.1.8/bin/clang` (env.sh) | `/usr/bin/clang` (系统版) |
| clang target triple | `riscv64-unknown-linux-gnu` | `riscv64-unknown-linux-gnu` |
| 配套汇编器 (as) | `/opt/tcrv-toolchains/binutils-2.46.1/bin/as` = **GNU as 2.46.1** | Bianbu 集成汇编器 |
| 配套 gcc-runtime (link CRT/libgcc) | `/opt/tcrv-toolchains/gcc-15.2.0` (`--gcc-install-dir=…/15.2.0`) | 系统 (Bianbu) |
| **须 `-fno-integrated-as`?** | **是**（对手手写 policy-less inline-asm·clang-18 集成-as 拒·转 binutils-2.46.1）·两侧对称同带 | 否（k1 样例 integrated-as 直接编过） |
| 部署域附注编译器 (不入主表) | gcc-15.2.0 / gcc(系统) | gcc-13 (Bianbu 13.2.0)·IME 格用 |
| **定稿 march (主表·板全覆盖)** | `rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause` | `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs` (+`_zvl256b` 可选) |
| ABI | `lp64d` | `lp64d` |
| 优化档 | `-O2` (canonical) / `-O3` | `-O2` (canonical) / `-O3` (对手 recipe) |
| VLEN (csrr vlenb) | **128** (vlenb=16) | **256** (vlenb=32) |
| governor | performance | performance |
| 核数 / pin | 64c · **pin core 8-15**(样例)·8-47(构建)·co-tenant vLLM core0,1 禁扰 | 8c · **pin core 0-3** |
| board cpuinfo ISA (compute) | imafdcv zicbom zicboz zicond zawrs zfa zfh zfhmin zba zbb zbc zbs zvfh zvfhmin zihintntl zihintpause | imafdcv zicbom zicboz zfh zfhmin zba zbb zbc zbs zvfh zvfhmin zkt zvkt **ime** |
| uarch / vendor | mvendorid 0x5b7 · mmu sv48 | Spacemit(R) X60 · 0x710 · sv39 |

## rvv clang-18 provenance（两个 18.1.8·择 canonical）
- **canonical（用它）** = `/opt/tcrv-toolchains/llvm-18.1.8/bin/clang` = "clang version 18.1.8"（配 binutils-2.46.1 + gcc-15.2.0·解 inline-asm 与 link 两 blocker）。
- **from-source 交叉核对（备份·冗余）** = `/home/ubuntu/g8-llvm18/llvm-project-18.1.8.src/build/bin/clang` = "clang 18.1.8-g8-upstream"（按 coordinator 指令从上游源码 min-build·仅 clang+RISCV·独立佐证版本=18.1.8·但缺配套 binutils/gcc-runtime → 链接/对手 inline-asm 不如 canonical 顺）。可清理回收磁盘。

## 指纹 ↔ VLEN 绑定
- rvv: clang 18.1.8 (tcrv-canonical) + as 2.46.1 ↔ VLEN128 ↔ mvendorid 0x5b7
- k1:  clang 18.1.8 (Bianbu 11bb4) ↔ VLEN256 ↔ Spacemit X60 (0x710)

## clang-18 rvv march：全板扩展【全 STABLE】（vs clang-17 之别）
clang-18 下 zfh/zfhmin/zvfh/**zvfhmin**/zba/zbb/zbc/zbs/zicbom/zicboz/zawrs/**zicond**/**zfa**/**zihintntl**/zihintpause 全部稳定接受
（粗体 4 项 clang-17 视为 experimental·拒；clang-18 转正）→ 定稿 march 得以【逐项覆盖全板实测 compute 能力】。

## ★IME-march carve-out（k1·显式结论·见 preflight D2）
`xsmtvdotii1p0`（IME）被 **clang-18 拒**（`unsupported version 1.0 for extension 'xsmtvdotii'`）·仅 **gcc-13 接受**。
→ IME 格（q4_0/q8_0/q4_K @ime）**不能 clang-18 对称**·口径 = gcc-13 真硅 cert 域（[CASE-COMPILER-ASYMMETRY]）。
IME 格**不在 0.8 硬门分母**（=matmul kernel-sym + forward-op）→ carve-out 合法。
