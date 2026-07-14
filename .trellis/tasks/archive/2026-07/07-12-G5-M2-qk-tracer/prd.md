# G5-M2 L-接线② q4_K 曳光弹(首格·验 scaffold 建法)

parent: `07-11-G5-wiring`（G5 接线战役）· status: in_progress · workflow `wyhnspmt6` 在飞。

L② = **新建上游 scaffold**（真工程·非纯链路翻 gate）。q4_K = 首格，验证「新建 scaffold」这一档接线的建法可复制性；成功则 q5_0/q5_1 复用（见占位任务 `07-12-G5-M2-q5x-tracers`）。

## 用户 2026-07-12 裁·三级接线分层定位
- L①链路层：q8_0/q4_0 已穷尽（翻 dispatch gate routing 即可，无需新建上游）。
- **L②新建上游 scaffold（本任务）：q5_0/q5_1/q4_K；首格 = q4_K。**
- L③跨框架 forward bridge：IQ + IME e2e（后续）。

## 流程（correctness-first·micro↛e2e 铁律仍管辖）
1. **recon 上游链路**：定位 q4_K 在 ggml repack/kernel 侧的挂点与缺口（对照 M0 接线机制解剖：dispatch gate + kernel arch/riscv + #include emitted .inc via deploy_patch 挂点③）。
2. **补缺 + emit**：gguf provisioning（q4_K 模型/数据到位）→ 新建上游 scaffold → host emit tcrv .inc。
3. **deploy correctness-first**：挂点③ 部署 emitted kernel（correctness-carrier·拦截破损上游），先做破损上游检查（对照 M1 的 [GAP-Q8_0-VLEN128-KERNEL] 教训——翻 gate 只 route 到破损上游 body = e2e garbage）。
4. **board byte-exact → e2e**：板 A-tree 可逆 + restore。
5. **perf-covered 3 → 4/84**：达成后按 perf-covered 登记册措辞模板填数登记（full-stack 无星号绿格判据）。

## 判据
- correctness-first：byte-exact 通过方可谈 perf；破损上游必须先被 emitted kernel 拦截。
- 若 scaffold 建法证明可复制 → 解锁 q5x 复用占位任务。
