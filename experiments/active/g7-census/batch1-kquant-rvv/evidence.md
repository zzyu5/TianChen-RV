# G7 L1 Batch-1 — K-quant@rvv cold-start census (5 formats · rvv-lane)

> **建成**：2026-07-14 · G7 终编成令三·第一段（全量 kernel 冷启动普查）Batch-1 主力线。
> **性质**：**第二赛道 kernel-sym 冷启动普查**（kernel-axis 对称 micro A/B·与 perf-covered 系统账 9/83 **永不混算·禁互推**）。[NG-4] 内核轴数据点·非 e2e·非 sealed 8-gate Win。
> **一句话结论**：**q{2,3,4,5,6}_K@rvv repack GEMM 在对称 gcc-15.2 下【全 20 cell LOSS】**（0.037×–0.78×）· byte-exact 全过 · **预判 <parity 兑现**。★核心新发现：**previously-recorded T9/casefile 数（clang-ours）在对称-gcc 下大面积翻深/翻转 = 全家族 [CASE-COMPILER-ASYMMETRY]**（gcc-15.2 -O2 在 un-pipelined emitter 巨型全展开体上 codegen 崩塌·vsetvli storm·比 clang-17 慢 2–3.4×）。

---

## 1. 板身份 + load-gate

- **板 rvv**：`Linux 6.12.66 riscv64` · 64c · **VLEN128**（`__riscv_vlenb()*8=128` LIVE）· core 8 · governor=performance · **2.6 GHz** · L2=2MiB(shared 8-11) · L3=64MiB(shared 0-63)。
- **对称编译域**：ours = **gcc-15.2.0**（`/opt/tcrv-toolchains/gcc-15.2.0` via env.sh）· 对手 libggml-cpu.so = **gcc-15 built**（`build-gcc15-rv64gcv`）→ **kernel-axis == system-axis 对称**（rvv 出货=gcc·[CASE-COMPILER-ASYMMETRY] 判别键满足）。`-O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on`。
- **load-gate**：进场时 co-tenant `llama-bench`（user ubuntu·Q5_0·-t8·~793% CPU·decode-profile 便车）在跑（loadavg 8.95）→ **让路**：该 job 于测量期间自然结束（loadavg 降至 2.2–3.2）。全程 core 8 独占 + 无 vLLM co-tenant（本会话零重启 vLLM）。**relIQR 拒污染**：cold iqr% 逐 cell 记录·19/20 cell iqr<2.1%（clean）；唯 **q2_K nr4 cold iqr=8.14%**（单点 co-tenant blip·ratio 0.367 与 q2 其余 nr 0.35–0.39 一致→非污染-invalid·标注保留）。
- **stock 库 restore 双证**：`libggml-cpu.so md5=d1adc634c2ca04ffc389536b30d9d9c4`（测前==测后·read-only 链接·never modified）。scratch 已清·**无 stray proc**。

---

## 2. 逐格四列普查表 {hot · cold · 对手类 · 胜负}（★deliverable）

> 协议：同会话 A/B · HOT=best-of-32(single-buffer warm) · COLD=224MiB flush 前置 paired median-N=12 · GEMM prefill 形（M=nr≥4）· nr{4,8,16,64}（含 M=8 便车点）· K=2048 nc=512 · core8。ratio = ours GMAC/s ÷ opp GMAC/s。

| fmt@rvv | nr(M) | **hot** | **cold_med** | cold_best | ours GMAC/s | opp GMAC/s | cold iqr% (o/opp) | **对手类(机判)** | **胜负** |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| **q2_K** | 4  | 0.378× | 0.367× | 0.378× | 1.78 | 4.71 | 8.14/0.24⚠ | hand-tuned `_vl128` block-dot | LOSS |
| q2_K | 8  | 0.385× | 0.354× | 0.375× | 1.80 | 4.68 | 1.56/0.40 | 〃 | LOSS |
| q2_K | 16 | 0.390× | 0.369× | 0.382× | 1.80 | 4.63 | 0.92/0.94 | 〃 | LOSS |
| q2_K | 64 | 0.396× | 0.389× | 0.393× | 1.80 | 4.54 | 1.18/0.39 | 〃 | LOSS |
| **q3_K** | 4  | 0.050× | 0.052× | 0.052× | 0.174 | 3.46 | 0.23/0.53 | hand-tuned `_vl128/256/512/1024` block-dot | **LOSS(深)** |
| q3_K | 8  | 0.050× | 0.052× | 0.052× | 0.175 | 3.53 | 0.13/0.44 | 〃 | LOSS(深) |
| q3_K | 16 | 0.049× | 0.051× | 0.050× | 0.174 | 3.58 | 0.28/0.59 | 〃 | LOSS(深) |
| q3_K | 64 | 0.049× | 0.050× | 0.050× | 0.174 | 3.53 | 0.40/0.34 | 〃 | LOSS(深) |
| **q4_K** | 4  | 0.355× | **0.235×** | 0.245× | 1.71 | 4.82 | 1.53/0.86 | main block-dot（+VLEN256-gated 16x1 repack 披露） | LOSS |
| q4_K | 8  | 0.360× | 0.292× | 0.296× | 1.73 | 4.80 | 1.29/0.70 | 〃 | LOSS |
| q4_K | 16 | 0.384× | 0.390× | 0.392× | 1.73 | 4.50 | 0.62/1.23 | 〃 | LOSS |
| q4_K | 64 | 0.400× | 0.390× | 0.391× | 1.73 | 4.33 | 0.16/0.41 | 〃 | LOSS |
| **q5_K** | 4  | 0.714× | 0.680× | 0.692× | 1.04 | 1.46 | 2.06/1.54 | main block-dot（**qh-burdened·弱对手** 1.36 GMAC/s） | LOSS(浅) |
| q5_K | 8  | 0.713× | 0.759× | 0.763× | 1.04 | 1.46 | 0.58/0.81 | 〃 | LOSS(浅) |
| q5_K | 16 | **0.763×** | **0.782×** | 0.782× | 1.04 | 1.36 | 0.17/0.30 | 〃 | LOSS(浅·最接近 parity) |
| q5_K | 64 | 0.766× | 0.767× | 0.768× | 1.04 | 1.35 | 0.82/0.10 | 〃 | LOSS(浅) |
| **q6_K** | 4  | 0.038× | 0.040× | 0.040× | 0.142 | 3.77 | 0.71/1.27 | main mature block-dot（+NEON-only 8x8 披露） | **LOSS(最深)** |
| q6_K | 8  | 0.036× | 0.039× | 0.039× | 0.142 | 3.88 | 0.59/2.10 | 〃 | LOSS(最深) |
| q6_K | 16 | 0.037× | 0.039× | 0.039× | 0.142 | 3.81 | 1.05/1.30 | 〃 | LOSS(最深) |
| q6_K | 64 | 0.038× | 0.038× | 0.038× | 0.141 | 3.74 | 0.37/0.36 | 〃 | LOSS(最深) |

**regime 读数**：
- **q3_K / q6_K**：hot≈cold（全 nr 平坦）= **完全 compute-bound**（gcc vsetvli-storm 支配·cache 态无关·二阶）。
- **q4_K**：cold<hot 在低 nr（nr4 cold 0.235× vs hot 0.355×·nr8 0.29× vs 0.36×）·nr16/64 cold≈hot → **cold-penalty 键控 memory-exposure**（低 nr=低 weight-reuse→更 memory-exposed·同 §6.2 FLAT q5_1 型）。
- **q5_K**：cold≈hot（nr8/16 cold 略>hot·ours cold 退化<opp）= compute/qh-bound。
- **q2_K**：hot≈cold（compute-bound）。

---

## 3. 对手类机判（符号级·commit f3e1828·★上游更强未启用路径披露）

**VLEN128 实际 dispatched = block-dot `ggml_vec_dot_qX_K_q8_K`（全 5 格·非稻草人）**。机判依据（LIVE `__riscv_vlenb()=128` dispatch probe · 同 commit f3e1828 已存 `dispatch_probe_raw.txt`）：repack GEMM trait selector 对全 5 K-quant 返回 **NULLPTR** @128（q2/q4=case128 TODO no-op·q5=无 riscv 分支·q6=NEON-only·q3=selector 无 case + 零 repack impl）→ prefill mul_mat fallback 到 per-(row,col) block-dot。

| fmt | dispatched 对手（机判符号） | 成色 | ★未启用更强路径（disclosed·非 dispatched） |
|---|---|---|---|
| q2_K | `ggml_vec_dot_q2_K_q8_K` → **`_vl128` 专调变体**（0x91cb6） | hand-tuned block-dot（中—强） | `ggml_gemm_q2_K_{16x1_generic,8x8}` 存在·VLEN256-gated(case128=TODO) |
| q3_K | `ggml_vec_dot_q3_K_q8_K` → **`_vl128/_vl256/_vl512/_vl1024` 全族专调**（0x9232c） | hand-tuned block-dot（最 VLEN-tuned·强） | **零 repack anywhere**（selector 无 case·家族最强 absence） |
| q4_K | `ggml_vec_dot_q4_K_q8_K` main（0x9312c·无独立 _vl128 符号·单实现） | main block-dot（成熟·中） | `ggml_gemm_q4_K_{16x1_generic,8x8}` 存在·**case128=TODO + 数值-wrong-at-128**(vfmv…16>VLMAX=8·故 case256-only) = 家族最强 unused path |
| q5_K | `ggml_vec_dot_q5_K_q8_K` main（0x93146·单实现·**qh 重建负担→1.36 GMAC/s 弱**） | main block-dot（**qh-burdened·弱**） | `ggml_gemm_q5_K_8x8` 存在·无 riscv 分支 |
| q6_K | `ggml_vec_dot_q6_K_q8_K` main（0x935de·单实现·mature 3.8 GMAC/s） | main mature block-dot（中—强） | `ggml_gemm_q6_K_8x8` 存在·**NEON-only**（无 riscv 分支） |

★纪律注：全对手 = **as-shipped 真 dispatched block-dot**（`nm` 符号机判·禁手写类目·禁挑弱对手）。q5_K 对手弱（qh-burdened 1.36 GMAC/s）是**其真实 dispatched kernel 的固有成色**（非我方挑弱）→ 故 q5 ratio 最高（0.76×）仍诚实。**q4_K 稻草人前车已避**：确认 dispatched=block-dot 而非更强 16x1 repack（后者 VLEN256-gated + VLEN128-数值错·不可能是 VLEN128 对手）。

---

## 4. byte-exact 逐格（correctness-first 硬门·mismatch=0）

CENSUS-timed 的**同一 regenerated 核**（gcc-15.2 g++ 编·同 census flags）vs STOCK ggml block-dot（独立 oracle·从 ORIGINAL block 解码）· INT byte-exact + bounded-NORM · 8 shapes（GEMM+GEVM·2 cert）：

| fmt | INT mismatch（全 shape·GEMM+GEVM） | NORM worst (maxAbsErr/rms) | verdict | 来源 |
|---|---|---|---|---|
| q2_K | **0** | 6.61e-07 | SILICON BYTE-EXACT-INT + BOUNDED-NORM | fresh（本线·verify_q2K）|
| q3_K | **0** | 5.55e-07 | 〃 | fresh（本线·verify_q3K）|
| q4_K | **0**（INT worstULP=0） | rel 1.44e-04 | PASS | fresh（本线·verify_q4K·GEMM-only）|
| q6_K | **0** | 5.24e-07 | 〃 | fresh（本线·verify_q6K）|
| q5_K | **0**（cited） | 8.0e-07（construction oracle） | BYTE-EXACT（既有 seal）| 34fded0d GEVM byte-exact + k1 t4a byte-id + rvv 构造 oracle + qh-recon HARD GATE·**同 fixture/tool/flags** 与本线 fresh 4 格 |

★q5_K 未跑 fresh repack-GEMM verifier（无现成 q5_K repack-GEMM 独立参考·adapt q4→q5 qh-plane 有 bug 风险）→ 引既有多重 seal + 指出 q5 核与 fresh-confirmed 4 格**同 fixture+tool（build-weft/bin/weft-opt @HEAD 677a2f7f）+ 同 gcc-15.2 flags** → toolchain byte-exact 已在 4 格实证·q5 承。**建议主会核**：若要 q5 fresh，构造 q5_K repack-GEMM verifier（qh 5th-bit plane pack）。

---

## 5. objdump seal + 反汇编归因（gcc-15.2 codegen 质量 ↔ loss 深度强相关）

| fmt | vsetvli | spill(vs*r.v) | reload | vwmacc | maxVreg | 核 size | ratio(hot) | gcc codegen |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| **q4_K** | **820** | **368** | 0 | 2240 | v31 | 59 KB | 0.40× | **最优**（family best）|
| q2_K | 1433 | 1446 | 0 | 2304 | v31 | 107 KB | 0.39× | 中 |
| q5_K | 2354 | 525 | 0 | 2240 | v31 | 92 KB | 0.76× | 中（但对手弱）|
| q3_K | **5115** | 1676 | 0 | 2176 | v31 | 260 KB | **0.05×** | **崩塌** |
| q6_K | **6278** | **2234** | 0 | 2304 | v31 | 300 KB | **0.037×** | **崩塌(最)** |

★**强相关**：vsetvli count（gcc codegen 崩塌指标）↔ loss 深度。vwmacc(~2200·真 MAC 工作量)全 5 格近同 → loss 差异**不在 MAC 质量·在 vsetvli-storm + spill 开销**。q3/q6 的 5000–6300 vsetvli 在 260–300KB 全展开体上 = gcc-15.2 -O2 无法调度 → 0.14–0.17 GMAC/s = 20–25× loss。**全 cleanfp（libcall-free·无 __truncsfhf2/__extendhfsf2）·全 v31 满寄存器压力**。

---

## 6. ★T9/casefile 对账（全家族 [CASE-COMPILER-ASYMMETRY] 兑现·需主会 reconcile T8/T9）

| fmt | T9 §1.2 recorded | 本线 hot(sym-gcc) | cold(nr16) | Δ | 归因 |
|---|---|---|---|---|---|
| q2_K | **0.386×** | 0.390× | 0.369× | ✓ **MATCH** | T9 0.386×=hot·gcc codegen 中(1433)·稳 |
| q3_K | ~0.176–0.20× | **0.049×** | 0.051× | ✗ **深 3.6×** | T9 用 clang-ours(0.60 GMAC/s)·gcc 崩塌(0.17)·vsetvli 5115 |
| q4_K | （hot 空白） | **0.384×** | 0.390× | NEW | 填空白·casefile clang 0.94×=非对称·sym-gcc 更深 |
| q5_K | **0.120×** | **0.763×** | 0.782× | ✗ **高 6.4×** | T9 0.120× **stale/异核/异 regime 存疑**·casefile clang 1.5×=非对称·sym-gcc 0.76×·**建议主会核 T9 0.120× 出处** |
| q6_K | ~0.18–0.19× | **0.037×** | 0.039× | ✗ **深 4.9×** | T9 用 clang-ours(0.67)·gcc 崩塌(0.14)·vsetvli 6278 |

**独立 cross-check（既有 casefile·clang-17 ours·同 gcc-15 opp）**：q5_K ours 2.19 GMAC/s→1.5× "WIN"·q4_K 4.25 GMAC/s→0.94× parity。**本线 sym-gcc ours**：q5 1.04·q4 1.73 GMAC/s → gcc-15.2 在 un-pipelined emitter 上比 clang-17 慢 **2.1×(q5)/2.5×(q4)/3.4×(q3)**。**⟹ casefile 的 "WIN"/parity 在对称-gcc 下蒸发为深 LOSS = 教科书 [CASE-COMPILER-ASYMMETRY]（clang-ours vs gcc-opp 非法·对称重测拍回）**。对手 q5 弱（1.36 GMAC/s·qh-burdened）独立确认（casefile opp 1.47 GMAC/s 同向）。

---

## 7. 每格三出口候选（供 L1' 攻坚·反汇编归因）

出口三分类：{A=对手结构优势具名 | B=重建成本结构性 | C=硬件能力缺席/编译器}。

| fmt | A 对手优势 | B 重建成本 | C 能力/编译器（★dominant 标） |
|---|---|---|---|
| **q2_K** | `_vl128` 专调 block-dot·pre-quant weight 流·in-reg dual d/dmin | dual d/dmin + bsums-min fold + nibble split（min-term dual fold=vsetvli 1433） | gcc 1433 vsetvli/1446 spill/v31·VLEN128 half_lanes=8 双 strip |
| **q3_K** | `_vl128/256/512/1024` 最全 VLEN-tuned·零 repack | 3-bit subtractive-hmask + unpacked 6b−32 scales·PLAIN(S6 NULL) | ★**gcc codegen 崩塌 dominant**（5115 vsetvli·260KB 体·0.17 GMAC/s=20× loss·clang 3.4×快） |
| **q4_K** | main mature block-dot·+16x1 repack 数值-wrong-at-128(不可用) | 6-bit scale/min unpack + nibble·**S6-tiled**(spill 368 最低·amortize/16-grp×4col) | cold-penalty@低nr=memory-exposure·gcc best(820)仍 2.5×off clang·VLEN128 |
| **q5_K** | ★**弱对手**（main block-dot qh-burdened 1.36 GMAC/s）→ ratio 最高 | qh 5th-bit inject + dual fold·**[GAP-Q5K-VLEN128-QH-REGCLIFF] NOT-FIXABLE**(dd113a3d·结构) | ★**编译器 dominant·最接近翻转**：gcc 1.04 vs clang 2.19 GMAC/s(2.1×)→若 gcc 追平 clang·q5 翻 ~1.6× WIN(弱对手) |
| **q6_K** | main mature block-dot·+8x8 NEON-only(无 riscv) | 6-bit dual-plane(ql@1312+qh@288) decode·PLAIN(S6 NULL) | ★**gcc codegen 崩塌 dominant(最)**（6278 vsetvli/2234 spill·300KB 体·0.14 GMAC/s=25× loss） |

**★L1' 攻坚共识（family thread）**：un-pipelined first-construction emitter 产巨型全展开体（260–300KB q3/q6）→ **gcc-15.2 -O2 无法调度（vsetvli storm）= q3/q6 loss DOMINANT factor·q2/q4/q5 的 2–3× factor**·叠加于算法 weight-recon floor 之上。**攻坚正序**：① **roll 泛化到 GEMM**（GEVM 已 roll·commits 34fded0d/2bbee909·v-insn −86~95%→ hot 0 vsetvli；GEMM 未 roll·当前 census 核= unrolled）→ 若 GEMM roll 同样消 vsetvli-storm·q3/q6 可从 0.05× 大幅回升；② roll 后余 weight-recon floor + VLEN128 half_lanes=8 双 strip 才是真结构地板。**q5 特例**：编译器 codegen 追平即翻转（弱对手）·最高 ROI 单点。

---

## 8. 建议入账行（供主会话·禁本线自入 tracked ledger）

**T9 §1.2 修订**（对称-gcc fresh·板批全量 rvv-半 K-quant 4→5 filled·§6.2 体例）：
- q2_K@rvv gemm：hot 0.390× / cold(nr16) 0.369× · 对手=`_vl128` hand-tuned block-dot · gcc vsetvli 1433 · **MATCH T9 0.386×**
- q3_K@rvv gemm：hot 0.049× / cold 0.051× · 对手=`_vl128/256/512/1024` block-dot · **修订 T9 ~0.176×→0.049×（[CASE-COMPILER-ASYMMETRY]）**
- q4_K@rvv gemm：hot 0.384× / cold 0.390×（低nr cold 0.235×）· 对手=main block-dot(+16x1 gated) · **填 T9 空白 hot**
- q5_K@rvv gemm：hot 0.763× / cold 0.782× · 对手=main qh-burdened block-dot(弱) · **修订 T9 0.120×→0.763×·flag 主会核 0.120× 出处**
- q6_K@rvv gemm：hot 0.037× / cold 0.039× · 对手=main mature block-dot(+8x8 NEON-only) · **修订 T9 ~0.18×→0.037×**

**T-CENSUS Batch-1 表**：5 cell cold 列全填（+ hot 重测 + 对手类机判 + 胜负 ratio）· 状态 ☐→✓。
**kernel-sym 计数**：全 <parity → **不入 ≥parity 计数**（维持 12·robustness LOSS 确认）。★禁互推 perf-covered 9/83。

---

## 9. 体例合规自检
- 普查=测量·**零 tracked 源码改动**（regenerate 核 = read-only build-weft/bin/weft-opt·驱动/verifier 皆新建 experiments/ 内 untracked）✓
- 对称编译 gcc-15.2 双侧（kernel-axis==system-axis）✓ · N=12 T-N(median+iqr+noisefloor)✓ · 冷态=224MiB flush(>3×L3=64MiB)✓ · GEMM prefill(nr≥4)+M=8 点 ✓ · nr{4,8,16,64} 穷举 ✓
- 对手类**机判符号级**（非手写·非挑弱·上游更强 unused path 披露）✓ · byte-exact 硬门 mismatch=0 ✓
- load-gate（decode-profile 让路）+ relIQR 拒污染(19/20<2.1%·q2nr4 标注)✓ · 无 stray proc · stock md5 测前==测后 ✓
- **禁互推**：kernel-sym(本线)≠ perf-covered 9/83 ≠ certified ✓ · [NG-4] 非 e2e 非 sealed Win ✓
