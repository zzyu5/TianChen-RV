# G5 接线战役 — emitted tcrv kernel 接入 ggml 真实 forward

接线 = 补丁/链接层集成(NG-2 不动·不做图框架)·接线 ≠ 自动转绿(micro↛e2e 铁律仍管辖)·板 A-tree 可逆+restore。perf-covered 当前 2/84·唯一拉绿杠杆 = 本战役。

## 里程碑表

| M | 内容 | 状态 | provenance |
|---|---|---|---|
| M0 | 接线机制解剖(两物理挂点:dispatch gate repack.cpp:4592/4713 + kernel arch/riscv q8_0 gemv:518/gemm:1426;挂点③=#include emitted .inc via deploy_patch) | ✅ done | commit ab054260·casefile experiments/active/g5-wiring/M0-接线机制解剖.md |
| M1 | 曳光弹 q8_0(翻 dispatch gate routing) | ✅ done·**R4 correctness RED** | commit f8b8dabb·[GAP-Q8_0-VLEN128-KERNEL](上游 q8_0 VLEN128 repack kernel 数值破损·ggml repack.cpp:230/240/285/339 硬编码 AVL=16→VLEN128 钳 vl=8→garbage)·翻 gate 只 route 到破损上游 body=e2e garbage·8.69× MIRAGE(VOID)=correctness-gate-catches-mirage 正面教材 |
| M1b | 部署我方 emitted q8_0 kernel(correctness-carrier·拦截破损上游) | 🔵 in-flight | workflow wwc9bbqeg·correctness-first 重测·host emit .inc→deploy 挂点③→board byte-exact→perf 分相 |
| M2 | 铺线(部署 emitted·各格·workflow fan-out:FLAT 余格→IME 3格 k1→K-quant;T5d 厂商对照随 IME 批) | ⚪ pending·gated on M1b(需 deploy 机制先证) | — |
| M3 | 收口(接线机制文档化) | ⚪ pending | — |
