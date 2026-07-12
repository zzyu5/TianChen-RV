# perf-covered 7/83 — q5_K@k1 净新-dispatch 绿格登记（★our-kernel K-quant e2e 传导·裁四.2 k1 唯一新绿点兑现）

> 登记 HEAD = `d0bf1ff8`（前序）· 生成 2026-07-12 · 触碰集 = docs + schema（label·recon 机算）。
> **口径**：承 `2026-07-12-perf-covered-roster-unification-recon.md`（fold 6/83·any-board·recon 禁手填）。**headline/分类/登记册三处同源由 `perf_covered_metrics.py` 机算**（本报告 = 登记册增量·6/83→**7/83**·recon 落定 `绿7`）。
> **★本报告 = 本会话首个新绿 + 首个 our-kernel K-quant e2e 传导**（区别 [WORK-ITEM] q4_K 测 stock repack·本格测我方 compiler-emitted repack）。

---

## 0. 登记结论

**perf-covered = 7 / 83 = 8.43%**（recon 机算·`绿7`·`reconciliation_ok/three_source_consistent/anti_gate_ok=True`）。第 7 格 = `gemm_tile/q5_K`（category 迁移 `黄-对手更强`→`绿`·any-board：k1 prefill 传导）。

**★verdict（裁四.2 pre-registered 绿出口兑现·SPLIT）**：`k1·q5_K·prefill·kernel账净绿·对手 stock q5_K block-dot·**e2e prefill 1.641×（≥parity WIN）**·双账本 clang-18 对称收敛`。decode 0.729× = 黄-物理墙（roofline·如实披露）。verdict 立于 **prefill≥parity**（同 q5_0/q5_1 prefill-win/decode-loss split 定式）。

---

## 1. 为何是 our-kernel 新绿（区别 [WORK-ITEM]·关键）

- **[WORK-ITEM-K1-KQUANT-E2E]（`b12afd57`）测的是 stock repack**（k1 ship q4_K case256·winner=stock·非我方 kernel·perf-covered 不加绿）——它 resolve 了 C3′ 绿路径 thesis（K-quant e2e 在 clang 传导·gcc-death=rvv-only），但不是我方交付。
- **本格测我方 compiler-emitted q5_K repack**：**k1 ships ZERO q5_K repack**（dispatch 仅 NEON sub-branch→riscv nullptr→block-dot·objdump-confirmed）→ 我方 emitted q5_K VLA kernel（GEMM `ba30ba54`/GEVM `c445b89e`·VLEN256 valid）经**净新 dispatch wiring**（block_q5_Kx16 struct + make_block_q5_Kx16 + repack/gemv/gemm 模板 + `q5_K_16x1_q8_K` trait + riscv case256 分支 + arch bodies）接入 k1 forward → **e2e prefill 1.641× vs stock block-dot** = **our-kernel 净新交付**（非 white-label）。
- ⇒ **q5_K = 第 2 个 K-quant e2e transduction（我方 kernel）**（第 1 = q4_K kernel-axis Win-K1-VLEN 1.085×；本格是**首个我方 kernel 的 K-quant e2e prefill 传导**）·**C3′ 绿路径家族扩展 with our kernel**（不止 stock repack 传导·我方 emitted 亦传导）。

## 2. 证据（casefile `experiments/active/g5-wiring/M2-q5_K-k1-e2e/`）

- **provisioning**：k1 build `llama-quantize`（避 rvv round-trip）·requantize q8_0→Q5_K_M → `/data/tinyllama-1.1b-Q5_K_M.gguf`（783MB·sha256 `6002d505`·A/B 同权重）。
- **correctness GREEN@bounded-ULP**：greedy A==B **3/4 byte-identical + 1/4 coherent-flip**（both 正确·fp-sum-order near-tie）·e2e ppl ON **17.9683** vs OFF **17.9095**（Δ0.33% within ±3.8·统计同一）·kernel bit-exact-integer cert·**MIRAGE definitively ruled out**。
- **build+seal**（专用 `/data/build-k1-q5k`·clang-18 对称）：OFF（0 tcrv sym=block-dot）vs ON（2 tcrv sym + engage banner·objdump **vwmacc=2240** matching kernel-axis seal→同 kernel 部署）。
- **e2e 分相**（12 samples/side·relIQR<0.2%·pass1≈pass2·load-gate steady）：**prefill 1.641× WIN / decode 0.729× LOSS**。
- **A-tree restored**：源树 baseline byte-exact（`3cac40aa/57851439/c3c101fd`）·stock lib untouched（`871169a0`）·0 `.inc` leftover·VERIFIED CLEAN。

## 3. 八门 + 双账本（诚实·②⑤ single-board caveat）

- **八门[prefill]**：①PASS@bULP ②**PARTIAL**（e2e k1-only）③PASS（nm 2 tcrv sym·vwmacc=2240==kernel-axis）④PASS（micro 1.916×∧e2e 1.641×）⑤**PARTIAL**（single-board e2e）⑥PASS（DVFS/pin/paired/load-gate）⑦PASS（compute-transduction vs bandwidth-roofline）⑧PASS（bounded k1/VLEN256/clang-18/prefill）。**★NOT a sealed universal 8-gate Win**（②⑤ = e2e single-board·成色 = **单板绿×kernel账净绿**·裁四.0 两维成色注记）。
- **双账本**：k1 出货 clang-18·kernel-axis == system-axis（symmetric single tree·[CASE-COMPILER-ASYMMETRY] not triggered·区别 rvv gcc-15）。
- **对手**：stock q5_K block-dot（k1 ships zero q5_K repack·objdump-confirmed·非 SELF）。

## 4. ★decode roofline caveat（诚实·prefill-win/decode-loss split）

decode **0.729× = 黄-物理墙（roofline·非失败）**：`block_q5_Kx16` stride-2816 + qh plane 对 nr=1（decode GEVM）streams more bytes/token than plain `block_q5_K` → memory-bound roofline loss。**同 q5_0（decode 0.82×）/q5_1（decode 0.78×）prefill-win/decode-loss 定式**（repack GEMM compute-win 传导 prefill·interleaved layout 多带宽 wash decode）。verdict 立于 **prefill≥parity**·decode 如实披露为 roofline（非 GAP·非失败叙事）。

## 5. C3′ / C1 结论 + 关联

- **C3′**：q5_K@k1 prefill 1.641× = **我方 compiler-emitted K-quant repack kernel e2e 传导**（confirms [WORK-ITEM] 的 C3′ 绿路径 thesis·**with our kernel**·非仅 stock repack）·K-quant e2e 绿路径 = clang（gcc 是 rvv 问题·[CASE-COMPILER-ASYMMETRY]+L3 clang 正门）。
- **C1**：net-new dispatch wiring（k1 ships zero q5_K repack·我方从零建 q5_K repack 链路接入 forward）= template 可扩展性 K-quant e2e 实证（超 FLAT）。
- **裁四.2 兑现**：pre-registered「k1 唯一新绿精准点」= q5_K@k1 e2e green（prefill axis·our-kernel）。
- 关联：`2026-07-12-perf-covered-roster-unification-recon.md`（recon 7/83）· `2026-07-12-q5_0-green-4of84.md`/`q5_1-green-5of84.md`（FLAT prefill-win/decode-loss 定式对照）· `workitem-k1-kquant-e2e/evidence.md`（stock repack 传导 thesis）· memory `q4-0-e2e-is-routing-not-kernel`（C3′ compiler-codegen 精化）。

## 6. check_docs_canon 自检

本报告 = `docs/reports/2026-07-12-perf-covered-q5_K-k1-green-7of83.md`·带 `YYYY-MM-DD-` 前缀（reports/ append-only 合规）→ PASS 预期。口径不改（承 recon·perf-covered 定义未动·headline/分类/登记册三处同源由 `perf_covered_metrics.py` 机算 = 7/83）。
