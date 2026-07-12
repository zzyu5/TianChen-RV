# perf-covered 维持 6/84 — q6_K yellow-kernel-axis 登记 + K-quant 净新接线收口

> 登记 HEAD = `2e92afe9`（未变·docs 域触碰）· 生成 2026-07-12 · 承 `2026-07-12-perf-covered-零未定义格分类.md`（q6_K 预判行 §1.1 #11）。
> **本报告 = 零未定义格 completion 交付（非绿登记）**：q6_K 曳光弹 firm 分类册 q6_K 黄判读成色（L1-kernel-axis → 净新-scaffold-e2e-board-firmed·A 档），**计数不变（Σ=84·perf-covered 维持 6/84）**，并立一条 C3′ thesis 级发现（§3）。

---

## 0. 登记结论（yellow-kernel-axis）

**perf-covered 维持 6/84 = 7.14%**（q6_K **不入绿**）。q6_K 净新 riscv `1,16` scaffold（block_q6_Kx16 stride 3360·no-min 6-bit 双平面·复用上游 q8_K 激活）：

- **correctness GREEN**（硬门过）：silicon UT INT byte-exact 8 shapes（GEVM+GEMM·0-mismatch）+ bounded-NORM 4.564e-07 · greedy A==B **5/5 byte-identical**（ON emitted vs OFF stock）· 43 engage banners · objdump vl=8 seal · MIRAGE trap 关（板 canonical verifier `pack_w` == make_block_q6_Kx16 byte-identical）。→ **C1 template 结构延伸到 K-quant no-min 超块 e2e**（从零建 riscv q6_K repack 全链路·route+repack+dispatch 自建集成经 ggml 对照证正确）。
- **prefill perf 重回退**：ON **< 0.28 t/s** vs OFF **4.14 t/s** = **< 0.07×**（单 128-tok rep >460s DNF·三独立观测一致）· decode 亦 ≪1×（weight-reconstruction-bound GEVM）。八门 ①–⑧ 齐（同树 .so swap·nm 2/0·banner 9+·objdump vl=8·对手 stock block-dot 非 SELF·gcc-15.2.0 双侧对称·correctness 前置·DVFS 锁 2.6GHz）—— **但 prefill ≪parity ⇒ 不过 ≥parity 门 ⇒ 禁以 perf 名义入台账**。

**verdict = yellow-kernel-axis**：`k_quant_path_extends（结构）= TRUE · k_quant_path_extends（perf）= FALSE`。双账本 kernel-axis == system-axis（board rv64gcv 出货 == gcc-15·对称·数值同·均有效）。

**A-tree 测后 restore 验 clean**：源回 baseline byte-exact（GEN=`deb61a29`/HDR=`57851439`/ARCH=`99131cf7`）· live .so=`05a62e6a`（OFF-pristine·0 q6_K syms）· 0 stray .inc。

---

## 1. 证据（casefile `experiments/active/g5-wiring/M2-q6_K/`）

`evidence.md`（八节全）· `correctness_GREEN_raw.txt`（5/5 A==B·43 banners·GREEN）· 两 .inc gitignored（gemm md5 `9c49195c`/1.8MB · gevm `768a3892`/1.1MB·regenerable via emit recipe·md5 已记）· board harness `tools/e2e-harness/board/g5-m2-q6_K/`（deploy_patch + ut + build_seal + correctness + phase_split + analyze + verifier）。

**病灶**（对齐 memory `q4-0-e2e-is-routing-not-kernel`·K-quant perf 立不住·weight-reconstruction-bound）：emitted vl=8 q6_K repack GEMM = 2995 vsetivli + 6065 e8mf2 ops 巨大展开·无调度 → ~15× 慢于 stock hand-tuned block-dot。q6_K prefill 0.07× 比 q4_K（0.42×）**更差**——6-bit ql+qh 双平面 decode 更重。

---

## 2. firm 分类册 q6_K 判读（成色升级·计数不变·纠 stale）

零未定义格分类册 §1.1 #11 原判 = **黄-对手更强**（依据 `l1-t3-q6k` kernel-axis 对称 LOSS ~0.18–0.20×·成色 = C 档 L1-kernel-axis 推测）。曳光弹**不改类别**（仍黄-对手更强·stock block-dot 更强），但**升成色 C→A**（净新-scaffold **e2e board** correctness-green + prefill 0.07× 实测锚定·非 L1-kernel-axis 外推）。

**Σ 不变（84）· 头条不变（6/84·format）**：q6_K 早在分类册 84 分母内（K-quant gemm 4 之一·`黄-对手更强` 桶）· 曳光弹 = 该桶内成色 firm·**不新增/不移格**。分类计数行（绿 7 / 物理墙 12 / 传导稀释 3 / 对手更强 12 / 未接线 23 / 声明例外 27 · Σ=84）**不动**。

---

## 3. ★K-quant 净新接线收口（C3′ thesis 级发现·campaign 关门）

q6_K 是 **K-quant 净新 scaffold 接线的最后一枚曳光弹**。合 q4_K（k1-half 绿·rvv-half 0.334× LOSS）+ q2/q3/q5/q6_K（对称-gcc kernel-axis LOSS）· 净新接线证据齐 → **K-quant 净新接线 campaign 收口**：

**★发现（C3′ 模板产出质量·arithmetic-intensity 律）**：**净新 scaffold 路径的 *perf* 延伸性由负载算术强度决定，非由 template 结构可扩展性决定。**
- **FLAT 家族（q5_0/q5_1/q4_1）** = repack-**bandwidth**-win：净新 repack GEMM vs stock generic block-dot = L1 path-win（prefill 1.09–3.68× 全传导绿）· 因 flat weight decode 轻（4/5-bit 单平面）→ repack 布局带宽赢主导。
- **K-quant 家族（q2/q3/q5/q6_K）** = weight-**reconstruction**-bound：净新 repack GEMM 的 emitted vl=8 核 weight 重建（super-block scale-unpack + 双平面 6-bit decode）**算术强度高**·全展开 regfile spill（LAW-FIRST-EMISSION 例 1）· gcc-15 codegen 在 mixed-SEW 全展开核上远弱 clang（820 vsetvli/742 spill vs 71/3·[CASE-COMPILER-ASYMMETRY]）→ **perf 不传导**（对称重测 LOSS·q6_K 0.07× 最狠）。

**⇒ template 结构可扩展性（C1）在 FLAT/K-quant 全成立**（q6_K 从零建 riscv repack 链路 correctness-green 是最强 C1 供弹）·**但 perf 证词（C3′）在 K-quant 落于黄格带账**（rvv/VLEN128-gcc 出货路不转绿）。**绿路径 = k1/VLEN256 部署**（q4_K sealed Win-K1-VLEN 先例·1.085×）**或 gcc-codegen-aware 再发射**（具名 [GAP-KQUANT-GCC-CODEGEN]·可修否逐格判·q5_K 最浅 0.775× 是次优候选）。

**这不是失败**：是 C3′ 档案级正面教材（模板可**结构**复制到任意量化族·但**性能**兑现锚在负载算术强度 × 部署编译器身份·= "能力键控须延伸到编译器身份/内存层级" 主论点的又一实证）· 与 memory `kernel-wins-dont-transplant-to-e2e` / `q4-0-e2e-is-routing-not-kernel` 同路。

---

## 4. LAW-FIRST-EMISSION 例 1 强化（不新增行·只减不增）

q6_K prefill 0.07× 的根因 = **首次发射逐片段仪式开销（全展开 regfile spill·6065 e8mf2 ops 无调度）**——LAW-FIRST-EMISSION **例 1（K-quant S6 repack）的又一实例**（6-bit 双平面比 q4_K 更狠）。成熟杠杆 = S6 tiling（mbf×lmul·SEW/LMUL 键）· **但 q6_K 上 S6 NULL**（`l1-t3-q6k` 证·gcc-codegen + weight-reconstruction floor 双阻）→ **可修 = 低**（区别 clang-有效的 FLAT 格）。不新增 LAW 行（同例 1 根因·`只减不增` 纪律）· 本报告 §3 收口即其归纳。

---

## 5. check_docs_canon 自检

本报告 = `docs/reports/2026-07-12-perf-covered-q6_K-yellow-kernel-axis.md`·带 `YYYY-MM-DD-` 前缀（reports/ append-only 合规·非 canon/ 越界）→ PASS 预期。口径不改（承 baseline + 分类册·perf-covered 定义未动·6/84 format 不变）。交叉引用：`2026-07-12-perf-covered-零未定义格分类.md`（§1.1 #11 firm）· `q4_1-green-6of84.md`（FLAT 全绿对照）· `2026-07-10-q4k-micro-vs-e2e-regression-mechanism.md`（q4_K gcc-codegen GAP）· `docs/method/LAW-FIRST-EMISSION.md`（例 1）· memory `q4-0-e2e-is-routing-not-kernel` / `kernel-wins-dont-transplant-to-e2e`。
