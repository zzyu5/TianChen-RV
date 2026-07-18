# PRD — B线第二块 P1: tq1_0 vec_dot owned ternary leaf 攻坚（公式墙必攻）

## 判据（本任务定性·非"赢没赢"）
P1 已由 objdump-recon 裁为 **公式墙**（对手 C-intrinsic·clang≈gcc·非 inline-asm·结构可读可键控）。**公式墙必攻**。产出二选一（都是进展·都顶头条别埋 EXHAUSTED）：
- **byte-exact + cold≥0.8 → 地盘 +1**（具名-X→PASS flip·主会话走独立 check 正负对称再入账）。
- **真撞脾气墙/M=1 floor → 边界 +1**（走三步：objdump 对手〔已done〕→建 owned leaf 定杠杆→板测证伪留档·墙型记对·说清读不到的挂号旋钮）。

## 目标格
`vec_dot tq1_0`（ternary·Q8_K）@**rvv**（cold 0.207·gap 最大）+ @**k1**（cold 0.606）。⚠ 这是 tq1_0 **vec_dot**（M=1·有 q8_K 归约）——**不同于 tq1_0 dequant**（第一块已收割·commit 0b53efdc9·streaming·别混）。

## objdump 对手（已 done·recon 权威·`archive/2026-07/07-19-block2-objdump-recon-clusterA/research/clusterA-wall-type-map.md`）
对手 = `ggml_vec_dot_tq1_0_q8_K_vl128`（rvv·tier=手调·ggml 上游手写 intrinsic·**非便宜 autovec**）/ `_vl256`（k1·满宽 e8,m1）：
- 结构：**`vwmulu.vx` ×{1,3,9,27,81…} powers-of-3 解包**（base-3 packed ternary）+ vsrl + vadd → 三值·再 MAC · 末 **1 条 `vwredsum.vs`** 归约。**gather=0**（纯算术·无码本无内存 gather）。
- rvv 131 insn / 67 向量 op 满向量化。**我方现无 owned 向量化 ternary vec_dot leaf**（弱/标量路 vs 对手满向量·gap 0.207=慢 4.8×）。
- **编译器对称**（clang131≈gcc132·两编译器独立调度近同码 → 对手是 C-intrinsic·**非 inline-asm 手排**·无"读不到的量"）⟹ **公式墙侧**。

## 杠杆（已定·objdump 导出·非猜）
**author owned ternary vec_dot leaf** = 复现对手可读结构（base-3 `vwmulu.vx` powers-of-3 解包 + 三值 MAC vs Q8_K + 归约），用 owned C-intrinsic（**非 inline-asm**）。参考素材：
- 第一块批3 owned ternary **dequant** body（`emitDequantizeRowTernaryVectorBody`·commit 0b53efdc9）已有 base-3/2-bit 解包原语 → vec_dot 复用解包段 + 加 Q8_K 点积归约。
- 对手结构（recon §三.34-36）= 目标形态。

## 硬约束（守·砸论文地基禁区）
- 🔴 **严禁 inline-asm / 钉死调度序列绕 clang**（对手是 C-intrinsic → 我方 owned C-intrinsic 就能对位·若我方 C-intrinsic 达不到而只有手排能赢=那才是脾气墙·登记边界·不绕）。
- **byte-exact 门（硬前置·先证再测 perf）**：ZERO-MODEL 从实际输入零复用重算全算术项 vs oracle·mism=0·3-arm anti-hollow(oracle-fault/DUT-fault/leaf-fault 各 RED·differing_bytes=1)·CORPUS COMPLETE。整数解包+i32 累加段 **byte-exact 守(0容忍)**；Q8_K scale 浮点段允许重排变体但标 `numerics.reassoc_ok`+ULP 界。**门未过=不进 perf 测·如实记具名-X**。
- **sealed vec_dot 核**：若碰现有 ternary vec_dot emitter（`RVVToEmitCTernaryBinary.cpp` md5 338a31bb），改动后须能解释（新增 owned leaf 是加性/替换弱路·非破坏现有 byte-exact 门）·收尾报 md5 与 diff。

## 板测（byte-exact 过后·配额授权=已 objdump 找过公式）
- board = **rvv**（`ssh rvv` 免密·VLEN128·clang18）+ **k1**（`ssh k1`·VLEN256）· **2-seed** cold（ours/opp=部署 `ggml_vec_dot_tq1_0_q8_K_*` 手调核）· **编译器对称**（我方 DUT clang18·对手用 clang18-symmetric 路 `build-clang18-rv64gcv`·[CASE-COMPILER-ASYMMETRY] 守·kernel-axis 数须编译器对称）。
- verdict：cold≥0.8 → PASS(地盘)·<0.8 → 具名-X(边界·带 objdump 证伪：owned leaf 达到什么·撞什么 floor·M=1 归约 vs 满行摊销)。

## 成色（诚实·禁越界）
- 对手 = **手调**（ggml 上游手写 intrinsic·**非便宜档**）⟹ 若赢 = 真硬赢强手调（**成色质变**·非便宜档·区别第一块 dequant 的 opp-immaturity）。若 owned leaf 达 parity/beat = 真路径赢。
- M=1 vec_dot：cold=micro kernel-axis 数·**别外推 e2e**（e2e 走部署路·分开报）。

## 交付（首节三项·回主会话）
1. **消灭待办 + 计数**：owned ternary vec_dot leaf 建成 byte-exact GREEN·verdict(地盘+1 若 PASS / 边界+1 若撞 floor)·标注这是第二块首击。
2. **未试杠杆 + 责任人**：若撞 floor·具名读不到的挂号旋钮（M=1 floor / clang 调度 / 归约链）·**禁"已尽力/架构不可达"除非杠杆真空且走过三步**。
3. **论文侧过期读数**：kernel-sym ≥parity 计数 / tq1_0 vec_dot 成色（若真赢手调=成色质变素材·顶头条）。

## 账面纪律
- **headline flip(具名-X→PASS)走独立 check 再 commit·正负对称**（主会话 build+lit+board 复验）。
- **不要 git commit**（trellis-implement 无 commit 权）。git add 只纳源·勿纳 build/worktree。worktree base 核对（从当前 tip·若旧线立即 reset）。
- master 入账走 `recon_master_rebuild.py`（IME_KERNELSYM/COLD dict·非 CLANG_WORLD-dequant·vec_dot 用 kernel-sym 或 COLD 账·主会话定位行键）。
- 返回结构化：每 board verdict + byte-exact(mism) + board cold(2seed·编译器对称) + owned leaf 结构 + sealed 核 md5/diff + (若边界)三步证伪留档。

## 参照
- objdump 墙型图：`archive/2026-07/07-19-block2-objdump-recon-clusterA/research/clusterA-wall-type-map.md`（§一#3-4·§三 P1·附录 A k1）。
- ISSUE-112（`.trellis/spec/issues/性能与测量.md:379`·簇A P1）· canon [K-4] 墙型词汇表。
- 批3 owned ternary dequant 解包原语：commit `0b53efdc9`（`emitDequantizeRowTernaryVectorBody`）。
