# FINDING · scalar 板（超锐）四路机检 = 物理 no-V 真硅（run-id 原始输出落盘）

> **性质**：一手机检落盘（W1 载体批 · 裁1 子项②）。**这是唯一允许的补测一次**——census/审计确认当天四路机检疑跑过但**原始日志找不回**（活仓库只存转抄结论），故按 PRD「日志找不回 = 唯一允许补测一次（`ssh scalar` 免密·重跑机检·落盘带 run-id）」重跑一次。
> **落盘 = 带 run-id 的原始输出**（`raw/RAW-scalar-rerun-carrier-w1-scalar-20260719T092244Z.rawlog.txt`·**非转抄结论**·md5 `6af17b5f22936c8929c892f4f16634af`）。
> **run-id**：`carrier-w1-scalar-20260719T092244Z` · repo-HEAD-context `d55f9ab4e` · board `fedora`(超锐) · clang 18.1.8(Fedora 18.1.8-6.fc41) · uname `Linux fedora 6.19.9+ riscv64`。

## 四路机检结论（四条独立判据·全 confirm 物理 no-V）

四条**互相独立**的确证（内核字符串 / 固件 DT / ELF auxv / 硬件执行陷阱），任一单独即证 no-V，四条合取 = 物理 no-V 铁证：

| # | 路（判据来源） | 原始读数 | 判 |
|---|---|---|---|
| **1** | kernel-reported ISA（`/proc/cpuinfo` `isa`）| `rv64imafdch_zicntr_zicsr_zifencei_zihpm_zaamo_zalrsc_zca_zcd` | **NO_V**（无 `v`/`zve` token）|
| **2** | firmware-reported ISA（`/proc/device-tree/cpus/cpu@0/riscv,isa`）| `rv64imafdcbh` | **NO_V**（无 `v` token）|
| **3** | ELF AT_HWCAP V 位（`getauxval(AT_HWCAP)`·bit `'V'-'A'`=21）| `hwcap=0x112d` · `V_bit21=0` | **NO_V** |
| **4** | 动态硬件陷阱（执行 V-major opcode `.word 0x0C007057` = `vsetvli x0,x0,e8,m1,ta,ma`·裸字节·**不依赖汇编器 -march**=真硬件探针）| `exec_ok=0` · `sigill_trapped=1` | **PHYSICAL_NO_V_SIGILL** |

`DYNAMIC_NO_V_SUMMARY: hwcap_no_v=1 hw_trap_no_v=1 => PHYSICAL_NO_V_CONFIRMED`（`noV_fourway` exit=0）。

**静态旁证**（objdump·同 session）：拥有内核 `weft_scalar_tq2_0_kernel.o`（clang-18 `-march=rv64gc`）= `total_insn_lines=191 · vector_mnemonic_lines=0 · vset_lines=0 · fp16_libcall(__extendhfsf2)=0`（`_Float16` scale 读内联为标量·无 libcall）；部署 `libggml-cpu.so` 整档 `vector_mnemonic_lines=0`（no-V 构建 ⟹ dispatch thunk == generic = 真部署标量路）。

## 结论

**`scalar`（超锐）板 = (a) 类物理 no-V 真硅**——四路独立机检 + 硬件 SIGILL 陷阱实证，非「带 V 板 run-as-noV」窄豁免、非 QEMU〔永久销案〕。铁线4 的旧 premise「手头无物理 no-V 真硅」**事实过期**（见同批 `PR-1-reopen-by-fact-correction.md`）。

## 边界（不夸大·[F-6]/[ISSUE-061]）

- 本 FINDING 只证「物理 no-V 板存在性 + 拥有内核在其上零向量执行」；**不**宣布 S 线可开测（标量派发路径**家族身份验收** = [ISSUE-061]·物理板在手 ⇏ 家族身份已验收）。
- 补测的四路机检**无性能主张**（[L-6] scalar 永不作贡献基线）；byte-exact/时序另见同批 `L3L4-scalar-silicon-byteexact-FINDING.md`。
- **该 add**：本文件 + `raw/RAW-scalar-rerun-carrier-w1-scalar-20260719T092244Z.rawlog.txt` + `raw/noV_fourway.c`（复跑源）。
