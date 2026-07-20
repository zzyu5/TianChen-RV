# k-vecdot-harness — K-quant vec_dot 双板对拍/计时 harness 资产格

**用途**：为 `tools/bench/cells/vec_dot.sh`（K 线·K-quant vec_dot 攻坚 harness）提供 driver + 独立 ZERO-MODEL oracle。
数据格（数据落 `experiments/`）；harness 脚本住 `tools/`（ISSUE-090 契约·住 tools/、写 experiments/）。

## 文件
- `kquant_vecdot_driver.c` — 对拍/计时 driver。
  - **OURS** = 板端链接的 owned weft-emitted block-dot leaf（叶资产只读引用 `../g7-census/vecdot-rvv/kernels/<fmt>.kernel.c`；GEN_SEAL 见该格）。
  - **OPP** = 板上真实导出 `ggml_vec_dot_<fmt>_q8_K`。q2/q3/q4/q6 记录板宽 `_vl128/_vl256` 手调路径；q5 当前两板记录 exported 通用向量路径。真实路径与对手档分列，不把“无专化”写成“无对手”，也不换成别的强对手冒充部署路径。
  - **ORACLE** = 五个格式各自从 raw block byte offset 独立解码；不复用 owned leaf 或 ggml decode helper。`QK_K==256` 为唯一 ABI，旧布局无兼容分支。
  - 固定、受控、精确整数 fixture 上做三向 byte-exact（ours==oracle==deployed ggml）；该结论只覆盖此 fixture 域，不外推所有浮点输入。
  - 负控：OURS output、oracle、owned leaf source 三种故障，以及 q8 `bsums != qs` 输入契约故障。q2/q4/q5 另跑 `dmin=1` min-term-active 臂，要求非零 witness 且清零 dmin 后结果改变。
  - `K=2048=8×QK_K`，因此同一格覆盖多 block pointer advance 与跨 block fold；activation `bsums` 从 `qs` 计算并先验验证。

## 覆盖

- q2_K/q3_K/q4_K/q5_K/q6_K 五格式统一 driver、统一五叶共链、显式五分支，无 q4/q6 旧 fallback。
- 五叶均由当前 production front-door/emission-plan 重新生成并逐字节核对；md5 与 `g7-census/vecdot-rvv/kernels/GEN_SEAL.txt` 一致。B2 核查时发现 q5 旧 leaf 漂移，已用当前 emitter 原子再生并在两板复验，不保留旧 fixture 兼容路径。
- `vec_dot.sh` 在任何 SSH 前逐叶核对 `GEN_SEAL.txt`；缺 seal、缺 leaf 或任一 hash 漂移均 fail-closed，不存在 seal 失败后的 rebuilt-copy fallback。
- 2026-07-20 B2 correctness verify：五格式 × rvv(VLEN128) 与 k1(VLEN256) 均由当前 runner parser 消费为合格结构化结果；每次 clean 固定 fixture 三向 byte-exact，故障臂均 bite。
- q2/q4/q5 的 min-term-active 臂双板通过；q8-bsums corruption 双板被输入门拒绝。
- 本轮对手二进制身份：rvv `libggml-cpu.so` SHA-256 `0782dda02f0b7fbe630b1acc2cc7fde44713b8f4558b3a9eb5640a52da7e5924`（source HEAD `f3e182816421c648188b5eab269853bf1531d950`）；k1 SHA-256 `3730c87c8f069e1e2c2c34b2bc3c083be034d8b0dd9b01ba8d85dcad3b792b5f`。每次 harness 另记录库 md5 前后守恒与实际 opponent symbol/path。
- 以上是 correctness/route 证据，未运行 B2 cold campaign，未生成性能头条、official run、master 变更或分母变化。

## 契约
harness 仓库侧零写盘；driver 只打 stdout（bench 解析·经 runner fail-closed 闸落三目的地）。本格只读被 harness scp 到板端 /tmp 构建（仓库侧不落任何板端产物）。

## 出处
初建 task `07-18-07-18-k-vecdot-harness`；五格式/双板加固 task `07-20-b2-bench-cell-coverage`。复跑入口：`bash tools/bench/cells/vec_dot.sh <rvv|k1> verify <q2_K|q3_K|q4_K|q5_K|q6_K>`。
