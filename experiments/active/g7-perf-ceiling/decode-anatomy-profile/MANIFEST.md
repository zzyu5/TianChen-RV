# MANIFEST — g7-perf-ceiling / decode-anatomy-profile

- **线**：G7 终编成令五.2 + 补丁三.2「decode 解剖图 profile 便车线」（升格正式任务）· **第三段作战地图 · 只入档不立战役 · 本线零优化 · 零 tracked 源码改动**。
- **问**：第三段（decode 组合战·多算子齐提满足传导公式第三项·M∈{4,8}）该打哪些算子？→ 先 profile decode 单 token 时间按算子类分布。
- **板/模型/build**：`ssh rvv` = SpacemiT VLEN128 · 64c · pin 8-15（disjoint）· 8t · perf-gov · 2.6GHz ‖ `DeepSeek-R1-Distill-Llama-8B-Q5_0.gguf`（5.21GiB·8.03B·输出层 Q6_K 混格）‖ llama-bench `f3e1828`（decode 走 stock ggml-cpu block-dot·baseline 解剖）。
- **方法**：`perf record -e task-clock -F 999` self-time（1M samples·0 lost·绕 paranoid=2）+ `perf stat`（cache-miss/IPC）。op-class = per-symbol self-time（干净·非内联）。

## 裁决 TL;DR
- **decode ≈ 单算子**：quant matmul（vec_dot·我方优化域）= **96.2%**（Q5_0 body 88.8% + Q6_K lm_head 6.4% + driver 0.9% + act-quant 0.1%）。非-quant 全体 3.8%（threading 2.5% / attention 0.7% / norm+rope+KV+elementwise+框架 <0.6%）。
- **bandwidth-bound 铁证**：IPC 0.61 · cache-miss 39.4% · DRAM ≈5.3 GB/token ≈ 全模型/token → vec_dot 时间 = DRAM 权重流式·非 compute 微质量。指令微质量不传导（[CASE-MICRO-E2E]）·只 memory-layout 传导。
- **传导第三项**：Amdahl p=0.962（10% vec_dot 提速→e2e +9.6%·cap 26×），但绑定约束=带宽非 Amdahl·可实现 Y=DRAM-BW 缺口（需 roofline·本线不测）。
- **「多算子齐提」M=1 无料**：非-quant 可攻面 <1% → 跨算子类组合无 headroom。M=1 唯一杠杆 = quant 权重 GEVM（+混格 Q6_K 第二靶 + threading 2.5%）。
- **诚实标注**：M∈{4,8} batched 会偏 compute-bound + attention 涨 → 第三段须单独 profile；本图不外推 batched。

## 基线绝对数（raw/baseline.json）
- prefill pp128 = 3.507 tok/s ‖ decode tg64 = 1.712 tok/s（tg96 1.71 / tg256 1.85）。

## files
- `evidence.md` — 完整解剖图（饼图 rollup + 混格双靶 + bandwidth-bound 证 + 传导第三项预估 + M=4/8 外推警示 + 第三段靶清单）。
- `raw/perf_report_self_top.txt` — perf self-time top-70（饼图源·1M samples）。
- `raw/perf_report_dso_top.txt` — 带 dso 归属（证 stock libggml-cpu block-dot）。
- `raw/baseline.json` — llama-bench pp/tg tok/s（N=3）。
- `raw/board_env_stat_raw.txt` — env fingerprint + 计时 + perfstat（IPC/cache-miss）+ perfrun。

## 约束遵守
- 禁 git ✓ · 禁改 tracked 源码 ✓（纯 board profile）· 只 profile 不优化 ✓ · pin 8-15 disjoint 未扰 co-tenant ✓ · load-gate（rvv loadavg ~2-6/64c）✓ · board restore：model size/mtime 未动 · `/tmp/g7prof` 已清 · 无 stray proc · governor 未触 ✓。
- **本图 = 只入档·第三段届时按数据立项·本线【非】战役**。
