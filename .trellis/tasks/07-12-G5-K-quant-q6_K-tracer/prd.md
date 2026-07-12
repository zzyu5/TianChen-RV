# K-quant 净新接线 q6_K 曳光弹(测 net-new scaffold 是否延伸 K-quant)

parent: `07-11-G5-wiring`（G5 接线战役）· status: **in_progress** · wf/agent `ac896a2e` 在飞 · 域: **板 + 集成**。

三线并行之线①（用户 2026-07-12 裁五）。前序 L② 首格 q4_K（`07-12-G5-M2-qk-tracer` completed）+ q5_0/q5_1 复用（`07-12-G5-M2-q5x-tracers` completed·perf-covered →6/84）已证 L② 新建上游 scaffold 建法可复制。本线测该「净新 scaffold」接线档**是否延伸到 K-quant**（q6_K = K-quant 家族曳光弹）。

## 流程（correctness-first·micro↛e2e 铁律仍管辖）
1. **recon 上游**：定位 q6_K 在 ggml repack/kernel 侧挂点与缺口（对照 M0 接线机制解剖：dispatch gate + kernel arch/riscv + #include emitted .inc via deploy_patch 挂点③）。
2. **emit + q8_K 激活**：host emit q6_K tcrv .inc；激活 q8_K（q6_K GEMM 的 activation-side quant）。
3. **净新 scaffold**：新建 K-quant 上游 scaffold（真工程），验其可复制性延伸至 K-quant 家族。
4. **correctness-first**：挂点③ 部署 emitted kernel（correctness-carrier·拦截破损上游）→ 板 byte-exact（对照 M1 [GAP-Q8_0-VLEN128-KERNEL] 教训：翻 gate 只 route 到破损上游 body = e2e garbage）→ 板 A-tree 可逆 + restore。
5. **perf**：byte-exact 通过后才谈 perf；每格过 X-0 + Amdahl 传导预估。

## verdict（预注册判读）
- **green**：full-stack 无星号绿格 → perf-covered 6 → **7/84** 登记（按 perf-covered 登记册措辞模板填数）。
- **yellow**：带账绿格（对手 SELF/e2e 门开等）→ **零未定义格**（黄格必带账·recon 脚本强制声明制）。
- 两路皆预注册，结果落地照判照走（权限卡 ⑦⑧）。

## 判据
- correctness-first：byte-exact 通过方可谈 perf；破损上游必须先被 emitted kernel 拦截。
- 零未定义格：无论 green/yellow，q6_K 格状态必须落进声明制（certified / 带账 / 域外），禁未定义。
