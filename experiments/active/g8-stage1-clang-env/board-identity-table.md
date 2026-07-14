# G8 stage1.3 — 双板编译器身份对表 (kernel-sym 对称域定稿)  2026-07-14

采集自 `ssh rvv` / `ssh k1`，governor/VLEN/clang 均现场探测。

| 键 | rvv 板 | k1 板 |
|---|---|---|
| **主表对称编译器** | **clang 17.0.6** (`17.0.6-16.oe2403`) | **clang 18.1.8** (Bianbu `11bb4`) |
| clang target triple | `riscv64-openEuler-linux-gnu` | `riscv64-unknown-linux-gnu` |
| clang 路径 | `/usr/bin/clang` (系统版·零安装) | `/usr/bin/clang` (系统版·零安装) |
| 部署域附注编译器 (不参与主表胜负) | gcc-15 (板出货) | gcc-13 (Bianbu 13.2.0)·IME 格用 |
| **定稿 march (主表·同 clang 两侧)** | `rv64gcv_zfh_zfhmin_zvfh_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zihintpause` | `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs`  (可加 `_zvl256b`) |
| ABI | `lp64d` | `lp64d` |
| 优化档 | `-O2` (canonical) / `-O3` (与部署 fair-march 对齐时) | `-O2` (canonical) / `-O3` (对手 recipe) |
| VLEN (csrr vlenb 实测) | **128** bit (vlenb=16) | **256** bit (vlenb=32) |
| governor | performance | performance |
| 核数 / pin | 64c · **pin core 8-15** (co-tenant vLLM 占 core0,1 禁扰) | 8c · **pin core 0-3** (本役用 core3) |
| board cpuinfo ISA (compute 相关) | imafdcv zicbom zicboz zicond zawrs zfa zfh zfhmin zba zbb zbc zbs zvfh zvfhmin zihintntl zihintpause | imafdcv zicbom zicboz zfh zfhmin zba zbb zbc zbs zvfh zvfhmin zkt zvkt **ime** |
| uarch / vendor | mvendorid 0x5b7 · marchid 0x80000000090c0d00 · mmu sv48 | Spacemit(R) X60 · mvendorid 0x710 · mmu sv39 |
| 冻结附注 | 主机 openEuler 2403·`ssh rvv` (新机) | Bianbu·`ssh k1` |

## 指纹 ↔ VLEN 绑定
- rvv: clang 17.0.6-16.oe2403 ↔ VLEN128 ↔ mvendorid 0x5b7
- k1:  clang 18.1.8 (Bianbu 11bb4) ↔ VLEN256 ↔ Spacemit X60 (0x710)

## clang march 稳定性差异 (两板 clang 版本不同 → 可用扩展集不同)
| 扩展 | clang-17 (rvv) | clang-18 (k1) |
|---|---|---|
| zfh / zfhmin / zvfh | 稳定 | 稳定 |
| zba/zbb/zbc/zbs / zicbom/zicboz / zawrs / zihintpause | 稳定 | 稳定 |
| zvfhmin | **实验** (需 `-menable-experimental-extensions`) | 稳定 |
| zihintntl / zicond / zfa | **实验** | (未验/非本役所需) |
| zvl256b | (N/A·VLEN128) | 稳定 |
| **xsmtvdotii1p0 (IME)** | (N/A) | **拒** (`unsupported version 1.0`) → 仅 gcc-13 接受 |

→ §三.2 只要求**板内**同 clang，**不强制跨板同版**；两板 march 因 clang 版本差异各自取"该 clang 稳定 ∩ 板实测"上界。
