# T4b e2e seal 板批 — 逐环归因 (2026-07-10)

**VERDICT: NOT sealed Win / NOT 2nd e2e headline — 但 full-construct 集成支柱板上证正确，剩单点 kernel-变体缺陷。**

## 一句话
全模型 q4_K 路由 + 权重 repack + q8_K 激活 + dispatch 自建集成 **整条链路经 ggml-generic 对照证正确**（PPL 11.97 ≈ stock upstream block-dot 12.05、相干生成）；**唯一缺陷 = 集成树部署的 q4_K 向量 kernel 是 VLEN256 变体（arch/riscv/repack.cpp `vint32m2`+`vl=16`，66 处），在 VLEN128 上 vl 钳到 8 → 只算 16 交织列的一半 → 输出垃圾（PPL 822057、贪心全 `?`）**。8 门证过的 VLEN128 S6-tiled kernel（`vsetivli…,8`=vl=8）从未部署进集成树。

## 逐环（板上实证）
- ① 路由 ✓（repack.cpp:4619 闸翻，block-dot banner 消失，193 q4_K 全经 make_block_q4_Kx16 load-time repack）
- ② 权重 repack ✓（板 make_block_q4_Kx16 逐段 == kquant_repacker.h：stride 2304 / 6-bit CUSTOM split / qs 交织全等）
- ③ 激活量化 ✓（gemm→quantize_mat_q8_K_4x1=M4 mat-quant oracle；gemv→quantize_row_q8_K；与双路径备忘 §2 一致）
- ④ 集成骨架 ★✓ 经证正确（q4_K→ggml-generic 参考 kernel → PPL 11.97 vs stock 12.05、相干）
- ⑤ 我方发射向量 kernel ✗ 垃圾（PPL 822057）
- ⑥ 根因 = VLEN256 kernel 误挂 VLEN128（arch/riscv q4_K gemm/gemv m2+vl16；正确 vl=8 kernel 未部署）

## M4 矛盾解释
M4「我方 kernel vs ggml 自身 GEMM 整数逐位一致」是 **standalone** 比对、未复现 ggml forward_mul_mat 真 VLEN128 集成调用 → **standalone 证据不覆盖集成调用 / 部署变体 ≠ 证过变体**（双路径备忘 §2「GEVM 从未直测=空白」的扩展）。M4 结论未错（vl=8 kernel 确正确），错在集成部署了 vl=16 变体。

## 模型口径
DeepSeek-R1-Distill-Llama-8B-Q4_K_M：Q4_K=193 / Q6_K=33 / F32=66 / **Q5_K=0**（q5_K e2e 需 Q5_K_M 模型）。Q6_K(33) 不路由留 block-dot = [1.3,1.6] 混合稀释来源。

## 下一步（单点可修）
部署 8 门证过的 VLEN128 vl=8 kernel（re-emit md5 90d454da）进 arch dispatch（替 vl=16 变体）→ 重建 → t4b_ppl.sh A（期望 PPL~12）→ t4b_seal.sh A（期望相干）→ prefill A/B + 八门②⑤。集成骨架 DONE、只差 kernel 一换。★同校 gevm（贪心/decode 全走 gevm、备忘 §2 空白）。

## 措辞锁
不得声称 q4_K e2e Win/加速。可引句：「能力键控全模型路由 + 权重repack + q8_K激活 + dispatch 自建集成、经 ggml-generic 对照证正确（PPL 11.97 vs 12.05）、我方 VLEN128 发射向量 kernel 集成部署变体错误待修」。harness: tools/e2e-harness/board/t4b_*.{sh,py,txt}。板 A-tree restored（repack.cpp deb61a29）。
