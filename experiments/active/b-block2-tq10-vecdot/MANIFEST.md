# b-block2-tq10-vecdot — B线第二块 P1: owned ternary vec_dot leaf 攻坚（公式墙必攻）

**用途**：为 `vec_dot tq1_0 x q8_K`（ternary·M=1·有 q8_K 归约）建 **owned 向量化 ternary vec_dot leaf**，
byte-exact gate + 板测 vs 部署手调对手 `ggml_vec_dot_tq1_0_q8_K`（clang-18 **编译器对称**·rvv VLEN128 `_vl128` / k1 VLEN256 `_vl256`）。
数据格（数据落 `experiments/`）；仓库侧零板端产物（板端 /tmp 构建·只回拉 seal 日志入 `raw/`）。

⚠ 这是 tq1_0 **vec_dot**（M=1·归约）——不同于第一块 tq1_0 **dequant**（streaming·commit 0b53efdc9·别混）。

## 对手（objdump-recon 权威·`archive/.../clusterA-wall-type-map.md` §三.34-36）
公式墙：`ggml_vec_dot_tq1_0_q8_K_vl128/_vl256`（**手调**·ggml 上游手写 intrinsic·非便宜档·clang≈gcc → C-intrinsic 非 inline-asm）。
结构 = base-3 `vwmulu.vx` powers-of-3 解包 + 三值 MAC vs q8_K + 末 **1 vwredsum**·**gather=0**。
⚠ 对手用**两个 VLEN 专化**（`_vl128` 的 vget-split fold 是 VLEN128 硬编码·故另备 `_vl256`）。

## owned leaf（三版·PURE C-INTRINSIC·NO inline-asm·NO 钉死调度）
| leaf | 结构 | VLEN-universal | 出处 |
|---|---|---|---|
| `kernels/tq1_0_vecdot_fused.c` (v1) | 独立自研：单 i16m4 累加器·**vwmacc** i8×i8·1 reduce·无 aux8 | ✅ 128+256 byte-exact | 独立设计（非抄对手） |
| `kernels/tq1_0_vecdot_lean.c` (v2) | 复现对手 `_vl128` 可读结构：digit-0 skip·i16-trit(vsub_u16)·prewiden-q8·vmul/vmacc·qh 单遍·**vget-split fold** | ❌ **仅 VLEN128**（k1 byte-exact FAIL·同对手 `_vl128` 的天生局限） | 采对手可读技法 |
| `kernels/tq1_0_vecdot_universal.c` (v3·**主交付**) | v1 的 VLEN-universal 单累加器 + v2 的 lean 解包（digit-0 skip·i16-trit·prewiden-q8·qh 单遍）·1 reduce·固定 vl | ✅ 128+256 byte-exact | v1+v2 合成（**一核覆盖两 VLEN·对手需两专化**） |

## byte-exact 门（`tq10_vecdot_driver.c` verify）
- **INT-mode**（d=1.0 两侧·fold 精确）：ours vs **独立 ZERO-MODEL 标量 oracle**（从原始字节重算全算术·pow3 mod-256 解包 + q8 索引图·序无关整数和）·**整数核 0 容忍**。
- **FLOAT-mode**（随机有限 scale）：fp32 fold ULP 界（`numerics.reassoc_ok`）。
- **3-arm anti-hollow**（INT-mode·三方 bit-identical → 单 bit 注错必被逮·无自愈）：oracle-fault / DUT-fault / factory-fault 各 differing_bytes≥1（RED-all）+ base_agree 前置。
- CORPUS = 8192 blocks × 2 (INT+FLOAT) × 2 seeds。

## 板测（compiler-symmetric·2-seed cold·`board_run.sh <rvv|k1> [kernel_src]`）
- **rvv**：clang-18.1.8（`/opt/tcrv-toolchains`·`build-clang18-rv64gcv/libggml-cpu.so` = 部署对称对手·env.sh）·VLEN128·8-15 核负载门。
- **k1**：clang-18（Bianbu）·factory 从 ggml 源 `/home/bianbu/tcrv-k1-llama` 现编·VLEN256·0-7 核负载门。
- cold = oversized-pool 256MiB(冷流)·median·metric `cold = opp_ns/ours_ns`（≥0.8 → PASS/地盘·<0.8 → 具名-X/边界）。

## 结果（`raw/<board>_<leaf>_seal.txt`）
| leaf | rvv byte-exact | rvv cold(2seed) | k1 byte-exact | k1 cold(2seed) | verdict |
|---|---|---|---|---|---|
| v1 fused | GREEN(int=0·ULP≤1) | 0.773 / 0.791 | GREEN | 0.681 / 0.682 | 双板 <0.8（边界·但 0.207→0.77 大幅收窄） |
| v2 lean | GREEN | 0.829 / 0.810 | **FAIL(VLEN256)** | (0.596·核错·无效) | rvv PASS·k1 揭示 VLEN128-only 陷阱 |
| **v3 universal** | **GREEN(int=0·ULP≤1)** | **0.933 / 0.901** | **GREEN(int=0·ULP≤1)** | **0.823 / 0.823** | **双板 cold≥0.8 → PASS 候选（地盘×2）** |

- 部署基线（master T3·弱/标量路）：rvv 0.207·k1 0.606 → **v3: rvv 0.90·k1 0.82**（rvv 关 73% gap·双板过 0.8 门）。
- 成色：cold<1.0 = 仍略慢于手调对手（rvv ~1.1×·k1 ~1.2×）·**PASS-by-gate·近-parity·非 beat**。对手=手调非便宜档 → 达门=真硬碰强手调的路径赢（非第一块 dequant 的 opp-immaturity）。
- sealed vec_dot 核 `lib/Conversion/RVV/RVVToEmitCTernaryBinary.cpp` md5 **338a31bb 未动**（owned leaf 是独立研究 kernel·非改 emitter）。

## 头条 flip 纪律
v3 双板 cold≥0.8 = **具名-X→PASS 候选**·须主会话独立 check（build+lit+board 正负对称复验）再入账。本 agent 只产数据+证据·不 commit。
master 入账走 `recon_master_rebuild.py`（vec_dot 用 COLD/kernel-sym 账·主会话定位行键 `(vec_dot,tq1_0,rvv|k1,M=1)`）。

## 谓词
`bash board_run.sh rvv kernels/tq1_0_vecdot_universal.c` → VERIFY RESULT=GREEN + RESULT cold ≥0.8（rvv+k1 同）。

## 出处
task `07-19-block2-p1-tq10-vecdot`（2026-07-19·第二块首击·公式墙攻坚）。
