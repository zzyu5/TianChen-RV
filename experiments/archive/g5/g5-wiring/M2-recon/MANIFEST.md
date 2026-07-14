# G5-M2-recon — 部署就绪度盘点 · CASEFILE

> **campaign**: G5 接线战役 · **M2-recon = deploy-readiness inventory**（M2 fan-out 排序依据）
> **role**: 纯 scout/docs — 本仓 read-only 静态分析（emit 路径 grep + schema roster + M0/M1 挂点复用）；**零 board / 零 build / 零 git**。
> **HEAD (TianChen-RV)**: `92c27389` 未变 · lib/·schema/·ROADMAP·M0/M1-q8_0/·tools/ 全未碰（只读引）。
> **产出**: `deploy-readiness.md` —— 84-cell 覆盖 · 按格式类分组（FLAT / K-quant / IQ / IME）· 三级就绪度（🟢/🟡/🔴）· correctness-carrier 列 · M2 六批 fan-out 建议。
> **一句话结论**: M2 首批 🟢 clean = **{q8_0}**（q4_0 已 WIRED）；次批 🟡 = {q5_0, q5_1, q4_K}；🔴 多数格瓶颈在上游 repack scaffold / 换板桥 / board provisioning，**非我方 emit 能力**（rvv repack 18 格 emit 侧全齐）。
> **provenance 锁**: dispatch-gate（GEN `repack.cpp`）+ kernel 挂点（`arch/riscv/repack.cpp`）行号全部来自**板 A-tree `f3e1828`（M0/M1 已记·不在本仓）**；上游 VLEN128 kernel 破损/存在状态本仓查不到者标"待 board 核"（禁 ssh）。

## durable files
- `deploy-readiness.md`
