# [CASE-COMPILER-ASYMMETRY] Stage-0 a) 同-.inc 双编译复现 — MANIFEST

> **三发定案的第一发**：复现 `2026-07-10-q4k-micro-vs-e2e-regression-mechanism.md` §2 的同源双编译器表。
> 纯 **编译 + objdump + 计数，零 perf 测量**（与板邻居争用无关）。禁改 A-tree、禁 git。
> 板 `ssh rvv`，date 2026-07-10。

## VERDICT — 复现 CONFIRMED（同量级，实为逐数精确）

同一 `gemm_q4_K.inc`（md5 **90d454da655f2fc1f88435d2d5826942** = 部署源 = micro s6_q4K.c），
同板、同 march、两编译器 objdump 计数：

| 预注册预期 | clang-17 | gcc-15 -O3 |
|---|---|---|
| 审计表 | ~71 vsetvli / ~3 spill / ~27KB | ~820 vsetvli / ~742 spill / ~58KB |
| **本复现实测** | **71 / 3 / 27.6KB** | **820 / 742 / 58.6KB** | 

→ Stage-0 a) 编译器-gap 机制 **CONFIRMED**。非"同量级"而是**逐数精确命中**（vsetvli 71/820 与 spill 3/742 完全一致）。
**harness 无异常**，无需 flag 排查。

---

## 1. 输入指纹

| 项 | 值 |
|---|---|
| 输入源 | `/tmp/t4b_seal_fix/gemm_q4_K.inc` (board rvv) |
| 输入 md5 | `90d454da655f2fc1f88435d2d5826942` ✓ == 审计声称的部署源/micro s6_q4K.c |
| 源性质 | C++ 单函数 kernel（`extern "C"`，`#include <riscv_vector.h>`，无 main，单符号 `tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_...`）；用 `-x c++ -c` 编成 .o |
| 部署真实 march | `rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause`（取自 A-tree `build-gcc15-rv64gcv/compile_commands.json`，435/435 文件同 flag，`-mabi=lp64d -O3`） |

## 2. 编译器版本（板实测，确切）

| 角色 | 二进制 | 版本串 |
|---|---|---|
| MICRO 侧 | `/usr/bin/clang-17` | `clang version 17.0.6 ( 17.0.6-16.oe2403)`，Target `riscv64-openEuler-linux-gnu` |
| DEPLOY 侧 | `/opt/tcrv-toolchains/gcc-15.2.0/bin/g++` | `g++ (GCC) 15.2.0`（经 `source /opt/tcrv-toolchains/env.sh`） |
| objdump | `/opt/tcrv-toolchains/binutils-2.46.1/bin/objdump` | GNU Binutils 2.46.1 |

（注：板默认 `/usr/bin/gcc` 是 12.3.1，**非**部署编译器；必须走 env.sh 的 15.2.0。）

## 3. march 处理 — 为何 matched-march = `rv64gcv_zvfh`

- 完整部署 march 含 clang-17（LLVM 17）视为**实验性**的扩展（`zfa` 等）→ clang 直接拒编（`requires '-menable-experimental-extensions'`）。
- kernel 用 `vfloat16m1_t` + `__riscv_vfwcvt_f_f_v_f32m2`（fp16 scale 加宽）⇒ 必须带 `zvfh`；plain `rv64gcv` 两编译器皆拒编。
- clang-17 在本 kernel 上接受的**最富且非实验性**、且 gcc 也接受的公共 march = **`rv64gcv_zvfh`**（`_zfh_zfhmin_zvfh_zvfhmin` 等更长串 clang-17 拒）。
- 故 head-to-head **两编译器用同一 `rv64gcv_zvfh`**（真正 apples-to-apples）。
- 另留 **gcc 全部署 march** 一编译作 **DEPLOY 锚**（验证 == 板上真部署 .so）。gcc 全 march 与 matched march **vsetvli/spill 完全相同（820/742）**，证明 vsetvli 风暴 + 整寄存器 spill **由 RVV codegen 驱动、与额外标量扩展无关**。

## 4. 两条完整 cmdline

```bash
# 公共前置（gcc 侧）
source /opt/tcrv-toolchains/env.sh
export LIBRARY_PATH=/opt/tcrv-toolchains/gcc-15.2.0/lib:$LIBRARY_PATH

# --- DEPLOY 锚：gcc-15.2.0 -O3, 全部署 march ---
/opt/tcrv-toolchains/gcc-15.2.0/bin/g++ -c -O3 \
  -march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause \
  -mabi=lp64d -x c++ /tmp/t4b_seal_fix/gemm_q4_K.inc -o gemm_gcc15_O3.o

# --- head-to-head matched march (rv64gcv_zvfh)，两侧同 march ---
/opt/tcrv-toolchains/gcc-15.2.0/bin/g++ -c -O3 -march=rv64gcv_zvfh -mabi=lp64d \
  -x c++ /tmp/t4b_seal_fix/gemm_q4_K.inc -o gemm_gcc15_O3_matched.o
/usr/bin/clang-17 -c -O3 -march=rv64gcv_zvfh -mabi=lp64d \
  -x c++ /tmp/t4b_seal_fix/gemm_q4_K.inc -o gemm_clang17_O3_matched.o
# （+ clang -O2 副证：/usr/bin/clang-17 -c -O2 -march=rv64gcv_zvfh -mabi=lp64d -x c++ ... ）
```

## 5. 计数表（objdump -d，单 kernel 符号 / .o）

计法（两侧同法）：
- **vsetvli** = 指令列 `vsetvli|vsetivli|vsetvl` 全部计数。
- **vector-spill** = 整向量寄存器进出栈：stores `vs1r.v/vs2r.v/vs4r.v/vs8r.v` + loads `vl{1,2,4,8}re{8,16,32,64}.v`。**这是审计的 742 metric**。
- **scalar sd/ld(sp)** = 标量整寄存器进出栈帧 `sd ...(sp)` / `ld ...(sp)`（另列，较小，非 742）。
- **.text** = `size` util 的 text 段字节；**.o** = `stat -c%s`。

| build | march | opt | vsetvli | vector-spill (st/ld) | scalar sd/ld(sp) | .text (B) | .o (B) | .o md5 |
|---|---|---|---:|---:|---:|---:|---:|---|
| **gcc-15.2.0 (DEPLOY 锚)** | full deploy march | -O3 | **820** | **742** (368/374) | 43/53 | 55790 | 58592 | `107e295b…` |
| gcc-15.2.0 (matched) | rv64gcv_zvfh | -O3 | 820 | 742 (368/374) | 45/55 | 59390 | 61976 | `51a766d0…` |
| **clang-17.0.6 (matched)** | rv64gcv_zvfh | -O3 | **71** | **3** (3/0) | 32/34 | 25456 | 27576 | `7ef47b3c…` |
| clang-17.0.6 (副证) | rv64gcv_zvfh | -O2 | 71 | 3 (3/0) | 35/39 | 25446 | 27616 | `979dde45…` |

**gcc / clang（matched, -O3）**: vsetvli **11.5×**（820/71）、vector-spill **247×**（742/3）、.o size **2.25×**（61976/27576）。

### 部署 .so 交叉验证（read-only objdump，闭环）
`.A-q4kON`(md5 a5e927bd) 符号 `tcrv_emitc_...gemm_q4_K_q8_K`：**vsetvli 820 / vector-spill 742（368 st + 374 ld）**
= **与本 gcc-15 重编逐数一致** ⇒ 我方双编译确 == 板上真部署二进制（不是脱靶重建）。

## 6. objdump 摘要（存要点，不存 .o）

### gcc-15 -O3 vset-form 直方图（top12，总 820）
```
    311 vsetivli  zero,8,e8,mf2      <- 每 8-元素组前重建 vtype
    104 vsetvli   zero,zero,e16,m1
     86 vsetvli   zero,zero,e8,mf2
     46 vsetvli   s0,zero,e8,mf2     <- 反复把 VLMAX 查进标量寄存器
     28 vsetvli   s1,zero,e8,mf2
     28 vsetvli   a1,zero,e8,mf2
     24 vsetvli   s4,zero,e8,mf2
     24 vsetvli   a3,zero,e8,mf2
     22 vsetvli   s2,zero,e8,mf2
     20 vsetvli   s9,zero,e8,mf2
     16 vsetvli   s8,zero,e8,mf2 / s3,zero,e8,mf2 ...
```
### gcc-15 vector whole-register spill 分解（总 742）
```
    324 vl1re16.v   318 vs1r.v   50 vs2r.v   50 vl2re32.v
```
### clang-17 -O3 vset-form 直方图（全部 71）
```
     34 vsetvli   zero,zero,e16,m1
     32 vsetvli   zero,zero,e8,mf2
      4 vsetvli   zero,zero,e32,m2
      1 vsetivli  zero,8,e32,m2
```
### clang-17 vector spill（总 3）
```
      3 vs2r.v
```
### gcc 热区病理样本（重复 csrr vlenb + 整寄存器 spill 交织）
```
   0: csrr t0,vlenb        60: csrr s0,vlenb       82: csrr a3,vlenb
  9c: vs2r.v v2,(a0)       bc: vs2r.v v2,(a0)       ee: csrr a3,vlenb
 14c: csrr a3,vlenb       16c: csrr a3,vlenb      178: vs2r.v v2,(a5)
 184: csrr a3,vlenb       190: vs2r.v v2,(a5)     19c: csrr a3,vlenb
```
gcc 每次进出 VLA 栈帧都重算 `csrr vlenb` + 帧偏移，未 hoist；S6 的 "stack panel" 在 gcc 下退化成
整寄存器访存流 + 重复 vlenb 地址算术。clang 每区一次 vtype、工作集全留寄存器（3 spill = init 非热 loop live-value）。

## 7. A-tree 完整性核对（前 / 后）

| 文件 | md5 前 | md5 后 | 期望 | 状态 |
|---|---|---|---|---|
| `…/bin/libggml-cpu.so.0.15.1` (live) | 75f20b5fddf2faf2b18d7a88b1bfb4b5 | 75f20b5fddf2faf2b18d7a88b1bfb4b5 | 75f20b5f stock | ✓ 未动 |
| `…/bin/libggml-cpu.so.0.15.1.A-q4kON` | a5e927bdd090217b76cb67152e1a77cc | a5e927bdd090217b76cb67152e1a77cc | a5e927bd | ✓ 未动（仅 read-only objdump） |

## 8. 清理

板临时目录 `/tmp/case_dualcompile/`（.o + dis txt）编译完 **已全删**；板上无遗留垃圾。
`/tmp/t4b_seal_fix/gemm_q4_K.inc` 是既有素材（非本任务产生），未动。**未 git add/commit/stash/rm。**

## 9. 结论（一句话）

同 md5 90d454da EmitC 源、同板同 march（rv64gcv_zvfh），**clang-17.0.6 编出 71 vsetvli / 3 vector-spill /
27.6KB；gcc-15.2.0 -O3 编出 820 vsetvli / 742 vector-spill / 58.6KB**——审计表逐数复现。且 gcc 重编
计数（820/742）== 板上真部署 `.A-q4kON` 符号，双编译确指同一部署二进制。**编译器-gap 机制 Stage-0 a) CONFIRMED，harness 无异常。**
