# G8 stage1.3 — 阶段三预飞门断言清单 (草案·供全量重测直接用)  2026-07-14

kernel-sym 主表口径 = **板内同 clang 对称域**；gcc 数字 = **部署域附注列 (不参与主表胜负)**。
阶段三每格 A/B 开跑前，预飞门须逐条 assert PASS，否则该格 INVALID。

## A. 编译器身份对称
- [ ] `A1` KERNEL_CC(ours) 与 KERNEL_CC(opp) **同 clang 二进制** (rvv=clang-17.0.6 / k1=clang-18.1.8)；`clang --version` 指纹逐字匹配对表。
- [ ] `A2` ours 与 opp **同 `-march` 同 `-mabi` 同 `-O` 档 同 `-ffp-contract`** (逐字符串 diff = 空)。板内对称硬门。
- [ ] `A3` gcc 变体若同时测，**另开附注列**，禁与 clang 主表数字混排/比较 ([CASE-COMPILER-ASYMMETRY])。

## B. march 完整性 ↔ 板能力 ↔ VLEN
- [ ] `B1` `-march` == 该板定稿串 (rvv=`rv64gcv_zfh_zfhmin_zvfh_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zihintpause` / k1=`rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs`)；无板 cpuinfo 未列的扩展 (禁 over-claim，如 rvv zicbop)。
- [ ] `B2` `csrr vlenb` 运行时实测 == 对表 (rvv=16→VLEN128 / k1=32→VLEN256)；指纹↔VLEN 绑定成立。
- [ ] `B3` 板 governor==performance；pin 生效 (rvv core8-15·k1 core0-3)；co-tenant 未占目标核。

## C. 双侧 objdump libcall + 符号探针
- [ ] `C1` ours .o 与 opp 符号 **均可解析** (`nm -D` / `objdump -t` 命中·非 stripped)；记 ours 符号名 + opp 符号名 + 地址。
- [ ] `C2` **双侧** objdump 扫 fp16 软浮点 libcall (`__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee`)：ours 须 libcall-free；opp 若命中 libcall 记入附注 (可能是编译器域差异)。
- [ ] `C3` opp kernel **同 clang 可重编** (从 ggml 源到 .o·符号级 objdump 对上探针)。若 opp 树用了该 clang 无的 intrinsic (见 D)，该格降为"对手树受限"·标注。

## D. 对手树 clang 可编性 (环境发现·per-format 分流)
- [ ] `D1` **rvv**：vericurve-rv-lab 树 quants.c 在 **q5_0 / q5_1 / iq4_nl / mxfp4** 用 `__riscv_vcreate_v_*` intrinsic，**clang-17 无此内建 (clang-18+ 才有)** → 该 4 格 vericurve 树**不可 clang-17 重编**。分流：(a) 用 upstream-native 树对手 (已 clang-17 可编·板上现成 .so)；或 (b) `__riscv_vcreate`→`__riscv_vset` 改写/shim (须主会话裁·动 lib 源)；或 (c) 该 4 格留 gcc 域。**q4_0/q8_0/q4_K/q5_K/q6_K/q2_K/q3_K 不受影响**。
- [ ] `D2` **k1**：IME 格 march `xsmtvdotii1p0` clang-18 拒 → IME 格**不进 clang-18 主表**，留 gcc-13/xsmtvdotii 真硅 cert 域。
- [ ] `D3` 对手用哪棵树 (rvv: upstream-native vs vericurve) 须主会话定；主表须全格同树同 clang。

## E. 对称性重编待办 (stage-3 地基·rvv)
- [ ] `E1` 板上现成 clang-17 ggml (`build-openeuler-clang17`) 是 **近邻 march** (`...zicbop...` 无 zfh/zfhmin/zicbom/zicboz)。要与定稿 march **严格对称须用定稿 march 重编 ggml-cpu** (单库)；zfh 可能改 fp16-scale codegen → 不省重编。
