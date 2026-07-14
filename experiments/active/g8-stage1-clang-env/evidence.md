# G8 阶段一.三 — 双板 clang-18 对称环境定稿 (环境就绪证明)  2026-07-14

**口径统一裁定 (2026-07-14·用户)**：双板 kernel-sym 主表对称域 = **clang-18**（一个版本一张表）。
起因 = 本役 clang-17 侧发现"对手部署树用 clang-18+ intrinsic(vcreate)·clang-17 编不出"→ 统一 clang-18 才能全保真重编对手。gcc = 部署附注·IME 格 carve-out。

**纪律遵守**：禁 git ✓ · 未改 emitter/ODS/lib 源 ✓（对手核用 `-fno-integrated-as` 原样重编·零源改） · 未开 e2e（仅 kernel-axis 样例 A/B） ✓ · co-tenant vLLM(core0,1) 全程保护(build pin 8-47+renice·样例 pin 8) ✓。

---

## 1. 双板编译器身份对表 → `board-identity-table.md`
- **rvv**：clang **18.1.8** (project-canonical `/opt/tcrv-toolchains/llvm-18.1.8/bin/clang`·env.sh) + as **binutils-2.46.1** + gcc-15.2.0 runtime · VLEN128 · pin core8-15。
- **k1**：clang **18.1.8** (Bianbu·`/usr/bin/clang`) · VLEN256 · pin core0-3。
- **一个版本一张表**：双板 18.1.8 ✓。
- **rvv clang-18 安装路径**：openEuler dnf 无 clang-18 二进制(仅 clang.src 17)→ ① 项目已 provision canonical clang-18.1.8 于 /opt/tcrv-toolchains(env.sh·**用它**)；② 另按 coordinator 指令从上游源码 min-build 18.1.8-g8-upstream(`/home/ubuntu/g8-llvm18`·独立交叉核对版本·冗余备份·可清)。

## 2. flags 定稿 → `flags-finalized.md`
- **rvv (rich·全板)** = `-O2 -march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause -mabi=lp64d -fno-integrated-as -ffp-contract=on`（链接 `--gcc-install-dir=…/gcc-15.2.0`）。clang-18 下全板扩展稳定(含 clang-17 拒的 zvfhmin/zicond/zfa/zihintntl)→ 逐项覆盖全板。
- **`-fno-integrated-as` 两侧对称必带**：对手手写 policy-less inline-asm·clang-18 集成-as 拒·转 binutils-2.46.1。codegen 不变。
- **k1 (非-IME)** = `-O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d`（不需 -fno-integrated-as）。

## 3. 对手核 clang-18 全量重编 + 符号探针 → `rvv/raw/opponent_full_recompile_clang18.txt`
- **rvv (要求3 SATISFIED)**：vericurve **部署树【本意最优形态】** quants.c 用 clang-18 + `-fno-integrated-as`(binutils-2.46.1) + rich march **全量重编 OK**。符号全解析：
  - 硬门 matmul 格：q4_0/q8_0/q4_K/q2_K/q3_K/q6_K/q5_0/q5_1/q4_1 ✓
  - **vcreate-4**(曾 clang-17 blocker)：iq4_nl/mxfp4(+vl128/vl256) ✓
  - inline-asm 格：q1_0 ✓
  - 两 blocker：[B1 vcreate] clang-18 有内建=解；[B2 policy-less inline-asm] clang-18 集成-as 拒 → -fno-integrated-as + binutils-2.46.1 解。**clang-17 时此 TU 在 q5_0 vcreate 即编不出 → clang-18 统一=净解锁**。
- **k1**：stock ggml `ggml_gemm_q4_K_16x1_q8_K`@0xabe58 符号可解析（Bianbu clang-18 出货·对手符号探针 ✓）。

## 4. 样例格 A/B 跑通 (clang-18·非正式测量·只证链路)
### rvv — q4_0 @ clang-18 对称 → `rvv/raw/sample_ab_q4_0_clang18.txt`
- ours(weft repack-GEMM) + opp(vericurve quants.o clang-18 重编·object 优先胜 .so) **同 clang-18 同 rich march 同 -fno-integrated-as**。
- **GATE=PASS**·relerr_ours==relerr_opp==9.055e-06·nbad=0·ratio 2.19(HOT)/1.93(COLD)。
- objdump 双方符号解析：ours `weft_emitc_ggml_gemm_q4_0_q8_0_kernel…`(libcall-free) · opp `ggml_vec_dot_q4_0_q8_0`(driver-resolved @0x11b54 = clang-18 opp_all.o·非 gcc .so)。
- 注：ratio 比作废的 clang-17 样例(6.94×)低 = 对手换成 clang-18 重编的部署核(opp 强~3×)·更公平（消除 clang-17-upstream-blockdot artifact）。
### k1 — q4_K @ clang-18 → `k1/raw/sample_ab_q4k_clang18.txt`（既有·新裁下有效·k1 一直 clang-18）
- ours(sealed vl=16·md5 e437fd3b) + opp(真 hand-brick `ggml_gemm_q4_K_16x1_q8_K`)·同 Bianbu clang-18。
- nbad=0/16384·ratio 1.20（== Win-K1-VLEN 复现）·双符号解析。

## 5. 双板 clang-18 域"环境就绪证明"
| 门 | rvv (clang-18.1.8 canonical) | k1 (clang-18.1.8 Bianbu) |
|---|---|---|
| 编译器身份 ✓ | 18.1.8 /opt/tcrv-toolchains · env.sh | 18.1.8 /usr/bin |
| flags 定稿 ✓ | rich 全板 march + -fno-integrated-as + --gcc-install-dir | canonical march |
| 对手 clang-18 全量重编+符号探针 ✓ | ✓ 全格(含 vcreate-4 + inline-asm)·两 blocker 均解 | ✓ stock 符号解析 |
| 样例 A/B 跑通 ✓ | q4_0 GATE=PASS ratio 2.19/1.93 | q4_K nbad=0 ratio 1.20 |
| IME carve-out ✓ | (N/A) | xsmtvdotii clang-18 拒→gcc-13 域·不在硬门分母 |

**总判：双板 kernel-axis clang-18 对称链路 = 就绪**。rvv 对手部署树全格 clang-18 保真重编(净解锁·两 blocker 解)；k1 IME carve-out 合法记档。clang-17 数据全部退役作环境备份·不入主表。

## 6. 交主会话回填 (禁另建账本·仅指针)
- 预飞断言 A/B/C/D/E → 阶段三 preflight harness。
- 定稿 clang-18 march + `-fno-integrated-as`/`--gcc-install-dir` recipe(rvv) → 回填 T3/kernel-sym 主表口径。
- IME-march carve-out → per-format 分流表(D2·不在 0.8 硬门分母)。

## 附：环境备份 & 板卫生
- **保留作环境备份(不删·不入主表)**：clang-17 upstream ggml.so `build-openeuler-clang17`·作废的 `sample_ab_q4_0_clang17.txt`·from-source clang `g8-llvm18`。
- rvv 构建 pin core8-47 + renice+15·样例 pin core8·co-tenant vLLM(core0,1) 全程未扰(top 实测 0,1 未被 build 争抢)。
- k1 样例 pin core3。测毕 /tmp 样例 scratch 清理·主 tree/build/governor/stock lib 未动。
