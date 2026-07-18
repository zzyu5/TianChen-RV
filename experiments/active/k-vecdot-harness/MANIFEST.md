# k-vecdot-harness — K-quant vec_dot @rvv 对拍/计时 harness 资产格

**用途**：为 `tools/bench/cells/vec_dot.sh`（K 线·K-quant vec_dot 攻坚 harness）提供 driver + 独立 ZERO-MODEL oracle。
数据格（数据落 `experiments/`）；harness 脚本住 `tools/`（ISSUE-090 契约·住 tools/、写 experiments/）。

## 文件
- `kquant_vecdot_driver.c` — 对拍/计时 driver。
  - **OURS** = 板端链接的 owned weft-emitted block-dot leaf（叶资产只读引用 `../g7-census/vecdot-rvv/kernels/<fmt>.kernel.c`；GEN_SEAL 见该格）。
  - **OPP**  = 部署 `ggml_vec_dot_<fmt>_q8_K`（@VLEN128 落手调 `_vl128` 专化·§对手法 3.4 部署事实·手调档）。
  - **ORACLE** = 内嵌纯整数**独立** ZERO-MODEL 重算（q4_K 6-bit scale utmp 解包 + min 项 · q6_K -32 offset + 2-bit qh 拼接；h2f 纯整数·int64 精确·[K-5]·实验宪法 §1.8 结构无关·不共享 bit→lane 解码）。
  - 3-way byte-exact（ours==oracle==ggml）+ INJECT 反空心（1=corrupt-ours·2=corrupt-oracle）+ fp16 golden self-test + cold 计时。
  - INT-mode 填充（super-block d=fp16 1.0·dmin/min=0·nibble 载荷·activation ±1）⟹ 精确整数 fold（K=2048 < 2^24）·序无关 byte-exact·数据无关 decode 稳态计时。[NG-4] kernel-axis·非 e2e·非 sealed Win。

## 覆盖
- 已建 oracle = **q4_K / q6_K**（baseline 首攻·板测 byte-exact ALL=true·具名-X LOSS vs 手调 vl128）。
- 待补 oracle = q2_K / q3_K / q5_K（叶存于 census·harness 白名单 HARNESS-VOID 待补 int-oracle·后续 task）。

## 契约
harness 仓库侧零写盘；driver 只打 stdout（bench 解析·经 runner fail-closed 闸落三目的地）。本格只读被 harness scp 到板端 /tmp 构建（仓库侧不落任何板端产物）。

## 出处
task `07-18-07-18-k-vecdot-harness`（2026-07-18）。谓词 = `bash tools/bench/cells/vec_dot.sh rvv verify q4_K`（3-way byte-exact + 4-arm 反空心复现）。
