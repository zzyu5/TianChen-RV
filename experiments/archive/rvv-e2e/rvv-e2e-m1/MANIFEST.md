# cell MANIFEST — rvv-e2e-m1 (RVV-E2E M1a token-tile 静态选型)

- **campaign**: RVV-E2E / M1a — token-tile 深度选型（静态账先行：objdump + emitter 源 + cache 拓扑账）
- **status**: ACTIVE (static-analysis casefile; 不上板 / 不改代码 / 无 board run)
- **role**: 纯静态选型证据 for the M1b 曳光弹（loop-interchange / token-tile depth）—— 部署 kernel 的
  register/cache/loop 结构事实 + 三档×三账矩阵，为 [VLEN-ADAPT] 与 M1b 选型提供 register-cliff 边界依据。
  Consumed by `experiments/active/vlen-adapt/vl16_static_account.md`（依据源之一）。

## durable files (git-tracked + untracked-not-ignored in this cell)

- `token_tile_selection.md` — M1a token-tile 深度选型表（静态账：peak_hot(d)=6d+7 vs 32-vreg、S6 实测 C=7、
  权重重读 32× 单因归因、cache 拓扑账；纯静态、无 board、无 git）.
