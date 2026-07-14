# G8 stage1.3 — 阶段三预飞门断言清单 (草案·clang-18 统一域·供全量重测直接用)  2026-07-14

kernel-sym 主表口径 = **双板 clang-18 对称域**（一个版本一张表）；gcc 数字 = 部署域附注列（不参与主表胜负）。
阶段三每格 A/B 开跑前，预飞门须逐条 assert PASS，否则该格 INVALID。

## A. 编译器身份对称 (clang-18)
- [ ] `A1` KERNEL_CC(ours) 与 KERNEL_CC(opp) **同 clang-18 二进制**：rvv=`/opt/tcrv-toolchains/llvm-18.1.8/bin/clang`(18.1.8) / k1=`/usr/bin/clang`(Bianbu 18.1.8)。`clang --version` 主版本=18.1.8 逐字匹配对表。
- [ ] `A2` ours 与 opp **同 `-march` 同 `-mabi` 同 `-O` 同 `-ffp-contract` 同 `-fno-integrated-as`**（逐字符串 diff = 空）。板内对称硬门。
- [ ] `A3` gcc/clang-17 变体若同测·**另开附注列**·禁与 clang-18 主表混排（[CASE-COMPILER-ASYMMETRY]）。clang-17 upstream ggml.so = 环境备份·不入主表。

## B. march 完整性 ↔ 板能力 ↔ VLEN
- [ ] `B1` `-march` == 该板 clang-18 定稿串（rvv=全板 rich `rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause` / k1=`rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs`）；无板 cpuinfo 未列扩展（禁 over-claim zicbop）。
- [ ] `B2` `csrr vlenb` 实测 == 对表（rvv=16→VLEN128 / k1=32→VLEN256）；指纹↔VLEN 绑定。
- [ ] `B3` governor==performance；pin 生效（rvv core8-15·k1 core0-3）；**co-tenant vLLM(rvv core0,1) 未占目标核**。
- [ ] `B4`(rvv) env.sh 已 source：`as`==binutils-2.46.1·链接带 `--gcc-install-dir=…/gcc-15.2.0`。

## C. 双侧 objdump libcall + 符号探针
- [ ] `C1` ours .o 与 opp 符号**均可解析**（`nm`/`objdump -t` 命中·非 stripped）；记 ours 符号 + opp 符号 + 地址。
- [ ] `C2` **双侧** objdump 扫 fp16 软浮点 libcall（`__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee`）：ours libcall-free；opp 命中则记附注。
- [ ] `C3` opp kernel **clang-18 可重编**（源→.o·符号级 objdump 对上探针）。rvv 已验：vericurve 树全格 clang-18 重编 OK（见 D1）。
- [ ] `C4`(rvv 对手 object-precedence) 若样例用 `opp_all.o + .so` 组合·须 `nm 可执行` 确认 driver-resolved opp 地址 = opp_all.o(clang-18) 定义·非 .so(gcc) 版。

## D. 对手树 clang-18 可编性 (环境发现·已解 / carve-out)
- [ ] `D1`(rvv·已解) vericurve 部署树 quants.c 两 blocker 均在 clang-18 域解决：
      **[B1 vcreate]** q5_0/q5_1/iq4_nl/mxfp4 用 `__riscv_vcreate_v_*`·clang-17 无→clang-18 有 ✓；
      **[B2 inline-asm]** q1_0/iq4_nl 手写 policy-less `vsetivli`·clang-18 集成-as 拒→ **-fno-integrated-as + binutils-2.46.1** 解 ✓。
      → 全格(硬门 9 + vcreate-4 + q1_0)clang-18 重编 OK·符号对上。**两侧统一带 -fno-integrated-as**。
- [ ] `D2`(k1·carve-out) **IME 格 (q4_0/q8_0/q4_K @ime) 不能 clang-18 对称**：`xsmtvdotii1p0` clang-18 拒(`unsupported version 1.0`)·仅 gcc-13 接受。
      → 口径=gcc-13 真硅 cert 域([CASE-COMPILER-ASYMMETRY])。**IME 格不在 0.8 硬门分母(=matmul kernel-sym + forward-op)** → carve-out 合法·已同步用户。IME 格禁进 clang-18 主表。
- [ ] `D3` 对手树选定 (rvv=vericurve 部署树·"本意最优形态")·主表全格同树同 clang-18·记 tree md5/commit。

## E. clang-17 退役 (口径统一)
- [ ] `E1` 此前 clang-17 rvv 样例(`sample_ab_q4_0_clang17.txt`)**作废**·不入主表·保留作环境备份附注。
- [ ] `E2` clang-17 upstream ggml.so(`build-openeuler-clang17`) 保留作环境备份·标注·不删。
