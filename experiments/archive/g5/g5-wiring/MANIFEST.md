# G5-wiring — 接线战役 (emitted kernel → ggml 真实 forward)

> Casefile for **G5 接线战役** (2026-07-11 用户立项): perf-covered 唯一拉绿杠杆 —
> 把 emitted tcrv kernel 接入 ggml 真实 forward = [GAP-FLAT-E2E-ROUTING] +
> [GAP-IME-E2E-INTEGRATION] 统一解。q4_0 routing 是唯一成熟先例·其接线机制就是模板。
> M0 解剖接线机制（两物理挂点：dispatch gate repack.cpp:4592 + kernel arch/riscv:234）
> + 通用方案（按格式类分）+ M1 曳光弹 q8_0 预案。**接线 = 补丁/链接层集成（NG-2 不动·
> 不做图框架）· 接线 ≠ 自动转绿（micro↛e2e 铁律仍管辖）· 板 A-tree 可逆 + restore。**

## HEAD / provenance
- G5-M0 侦察 @ `ab054260`（接线机制解剖）。
- 里程碑：M0 侦察 ✅ → M1 曳光弹 q8_0（翻 repack.cpp:4713 routing-freebie·workflow wyb4hslpu）→ M2 铺线（各格）→ M3 收口（接线机制文档化）。

## durable files
- `M0-接线机制解剖.md`

> Evidence-pointer docs stay in-cell; board harness/protocol scripts live under `tools/e2e-harness/board/`. Gitignored scratch rides with the cell but is not durable. M1+ casefile 文件随各里程碑落地追加登记。
