# G8 阶段一.三 — rvv clang 对称环境定稿 (双板环境就绪证明)  2026-07-14

**任务**：为阶段三 kernel-sym 全量重测建地基。用户已裁：**rvv 主表对称域 = clang-17 板系统版**
(gcc 降部署附注列·不参与主表胜负)；k1 = clang-18 (既有)。§三.2 只要求**板内 ours-vs-opp 同 clang**·不强制跨板同版。

**纪律遵守**：禁 git ✓ · 未改 emitter/ODS/lib 源 ✓ · 未开 e2e (仅 kernel-axis 样例 A/B) ✓ · 零工具链安装 (clang-17/clang-18 皆板系统版) ✓ · pin+清场+loadavg 记录+scratch 删除 ✓。

---

## 1. 双板编译器身份对表 → 见 `board-identity-table.md`
- **rvv**：clang **17.0.6** (`17.0.6-16.oe2403`) · `/usr/bin/clang` · triple `riscv64-openEuler-linux-gnu` · VLEN**128** · governor performance · 64c/pin core8-15。
- **k1**：clang **18.1.8** (Bianbu `11bb4`) · `/usr/bin/clang` · triple `riscv64-unknown-linux-gnu` · VLEN**256** · governor performance · 8c/pin core0-3。
- **clang-17 缺失? 否** → 零安装，环境未阻塞。

## 2. flags 定稿 → 见 `flags-finalized.md`
- **rvv 定稿 march** = `rv64gcv_zfh_zfhmin_zvfh_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zihintpause` `-mabi=lp64d -O2 -ffp-contract=on`。
  - 剔 clang-17-experimental (zfa/zicond/zvfhmin/zihintntl) · 剔板未实测 zicbop · 保板实测 compute 扩展。
- **k1 定稿 march (非-IME)** = `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs` `-mabi=lp64d -O2`。
- 两板因 clang 版本不同 → 各取"该 clang 稳定 ∩ 板实测"上界 (§三.2 允许跨板不同版)。

## 3. 对手可重编 + 符号探针 (结论)
### rvv
- **现成 clang-17 ggml 存在**：`/home/ubuntu/llama.cpp-upstream-native/build-openeuler-clang17/bin/libggml-cpu.so.0.15.1`
  (upstream-native 树·clang-17 编·符号可解析非 stripped：`ggml_vec_dot_q4_0_q8_0`@0x9605a、`ggml_vec_dot_q4_K_q8_K`@0x98294) → **对手同 clang 可重编 ✓ 符号对上探针 ✓**。
- **⚠ 环境发现 (per-format 分流·须主会话裁)**：**部署 (vericurve-rv-lab) 树** quants.c 在 **q5_0/q5_1/iq4_nl/mxfp4** 用 `__riscv_vcreate_v_*` intrinsic，**clang-17 无此内建 (clang-18+ 才有)** → 该 4 格 vericurve 树 clang-17 **不可重编** (整 TU 编译失败)。
  `q4_0/q8_0/q4_K/q5_K/q6_K/q2_K/q3_K` **不受影响**。分流选项见 `preflight-assertions.md` D1。
### k1
- stock ggml `/data/k1build-stock/bin/libggml-cpu.so.0.15.1`：`ggml_gemm_q4_K_16x1_q8_K`@0xabe58 (真出货 hand-brick)、`ggml_vec_dot_q4_K_q8_K`@0xa192e 均可解析 → **对手符号探针 ✓**。
- **⚠ IME 域分离**：`xsmtvdotii1p0` clang-18 **拒** (`unsupported version 1.0`)，gcc-13 接受 → IME 格留 gcc-13/xsmtvdotii 真硅 cert 域，**不进 clang-18 主表**。

## 4. 样例格 A/B 跑通各一次 (只证链路·非正式测量)
### rvv — q4_0 @ clang-17 严格对称 → `rvv/raw/sample_ab_q4_0_clang17.txt`
- ours (weft repack-GEMM) + opp (ggml vec_dot) **同 clang-17 同 march** (`rv64gcv_zvfh_zba_zbb_zbc_zbs_zicbop_zihintpause` == 现成 .so 自身 build march)。
- **GATE=PASS**·relerr_ours==relerr_opp==9.055e-06·nbad=0·ratio 6.94(HOT)/6.36(COLD)。
- objdump 双方符号解析：ours `weft_emitc_ggml_gemm_q4_0_q8_0_kernel…` (libcall-free) · opp `ggml_vec_dot_q4_0_q8_0`@0x9605a T。
- 定稿(richer) march ours 编译亦 OK (`OURS@FINALIZED_OK`)。
### k1 — q4_K @ clang-18 → `k1/raw/sample_ab_q4k_clang18.txt`
- ours (sealed vl=16 core·md5 **e437fd3b** 印证无 regen) + opp (真 hand-brick `ggml_gemm_q4_K_16x1_q8_K`)·同 clang-18 `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs`。
- **nbad=0/16384**·max_rel 8.78e-03 (q4_K byte-approx 门内)·ratio 1.20 (== 既有 Win-K1-VLEN vl=16 kernel ~1.197× 复现)。
- objdump：ours `tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel…` (libcall-free) · opp `ggml_gemm_q4_K_16x1_q8_K`@0xabe58 T。

---

## 5. 双板环境就绪证明
| 门 | rvv (clang-17) | k1 (clang-18) |
|---|---|---|
| 编译器身份 ✓ | 17.0.6-16.oe2403 · 零安装 | 18.1.8 Bianbu · 零安装 |
| flags 定稿 ✓ | richer march 定稿 (`flags-finalized.md`) | canonical march 定稿 |
| 对手可重编 + 符号探针 ✓ | ✓ (upstream-native clang-17 .so·符号解析) ⚠vericurve 4 格 vcreate 阻塞 | ✓ (stock .so 符号解析) ⚠IME 格 gcc 域 |
| 样例格 A/B 跑通 ✓ | q4_0 GATE=PASS ratio 6.94 | q4_K nbad=0 ratio 1.20 |

**总判：双板 kernel-axis clang 对称链路 = 就绪**。两处 per-format 分流 (rvv vericurve vcreate·k1 IME march) 已定性、写入 `preflight-assertions.md` D1/D2 供阶段三预飞门直接消费；**均须主会话在阶段三开跑前定树/定分流**，非环境阻塞 (核心 7 格 rvv + 非-IME k1 已全通)。

## 6. 交主会话回填 (禁另建账本·仅指针)
- 预飞断言 A/B/C/D/E → 回填阶段三 preflight harness。
- 定稿 march (两板) → 回填既有 T3/kernel-sym 主表口径附注。
- 环境发现 (vcreate / IME-march) → per-format 分流表。

## 附：板卫生
- rvv scratch `/tmp/g8_clangenv` 已删·loadavg_after=2.10 (co-tenant vLLM 基线)。
- k1 scratch `/tmp/g8_clangenv_k1` 已删·loadavg_after=2.16。
- 未动主 tree/build/governor/stock lib (只读)。既有 harness scratch (`/tmp/g7_l1_cold`·`/tmp/tcrv_k1_vlen_adapt`·`/tmp/q4k_hb_resolve`) 非本役产·未删。
