# B5：deployed q4_0 flat GEMM 路由与性能闭环

## Goal

先打穿一个真实、最有论文解释力的 deployed slice：q4_0 flat GEMM。从真实请求开始，
追踪 typed facts/context、candidate/legality、selected stamp、generated body、actual
deployed ggml symbol 与真板运行，证明 Weft 的多个 generated variants 确实由同一
框架构造、选择并部署。

实际 deployed ggml 路径与代表性强对手都是必要证据：前者证明系统真的接入，后者
证明性能成色；二者不互相替代。standalone P2 kernel 只能作为独立证据 lane，不能
冒充 deployed flat-GEMM 路径。

## Bounded Pre-registration

- op：official registry 中真实 flat `gemm_tile` / q4_0 deployed route；
- format：q4_0 为唯一 primary slice；其它 grid format 不在本任务顺手扩展；
- shape buckets：最多三个，至少含一个代表性 prefill tile；M=1/decode 只有在实际
  同 route 且有明确问题时作为分立 row，不混合报告；
- boards：`rvv`、`k1` 中 canonical capability/route/correctness 适用者；
- engine/regime/repeats：按完整 official row key 与 measurement spec 冻结；
- comparisons：actual deployed ggml、一个同语义代表性强对手、Weft analytic prior、
  合格 measured winner；四列分账；
- correctness：双方对 oracle、route/deployed symbol identity、byte-exact/规定 ULP、
  CORE==PROD、compiler symmetry；
- artifacts：request/selected trace、generated source/object、actual symbol、objdump、
  raw run、run-id、qualification/disposition。

## Scope and Order

1. **关闭 ISSUE-099 的真实 flat 前门缺口**：确认 official harness 接到 canonical
   q4_0 deployed request；若缺失，则原子增加 registry/parser/cell，并删除 direct
   benchmark dispatch、旧 alias 和 silent old-route fallback。
2. **构造与选择 trace**：记录每个 shape 的 bounded candidates、legality reasons、
   analytic prior、qualified winner（若可用）和 complete selected stamp。measurement
   miss 必须回 prior；illegal winner 必须拒绝/回合法 prior。
3. **actual deployment trace**：证明 bench 与真实 ggml consumer 使用同一个 selected
   body/stamp/route provider，直到实际运行的 concrete symbol；不能停在 metadata。
4. **correctness first**：双实现对 oracle、CORE==PROD、真实板运行通过后才能计时。
5. **official cold + objdump**：同板、同 compiler、同 flags、同输入比较 deployed
   ggml 与代表性强对手；检查 generated variant 之间的 loop/LMUL/repack/dispatch
   结构差分。
6. **disposition/retirement**：win/no-flip/loss/wall 均入唯一 run lineage。失败生成
   strategy 不留 dormant branch；强对手不会被塞进 production selector 当新 authority。

## Multiple Generated Variants: What This Proves

本任务的核心不是“只和一个最强核比”。同一个 q4_0 request 可由 formula/legality
得到多个合法 generated candidates，selector 在其中选一个，deployed route 实际运行
该 selected body。这证明框架的生成与部署能力。与真实 ggml 路径比较证明接入真实性；
与代表性强对手比较证明成色。必须同时报告，不能把强对手要求理解成否定真实路径。

## Multi-Agent Collaboration Contract

多个 agent 共同完成同一个 B5 slice：

- route/selection agent 负责 request→candidate/legal→stamp→actual deployed symbol
  全链与旧 bypass 退役；
- opponent/codegen agent 独立核对 strong opponent identity、compiler symmetry、
  objdump 与 shape-sensitive 差分；
- evidence agent 独立复核 correctness、pre-registration、run lineage、qualification
  和四列分账；
- root integration owner 管理共享 worktree、串行真板与 official writer、合并审查。

这不是把 harness、opponent、measurement 拆给三个互不沟通的人；三者共同验收同一
q4_0 deployed route，任一责任面未闭环则任务不完成。

## Primary Touch Set

- q4_0 flat GEMM source front door、selection/body/route provider 与 actual consumer；
- `tools/bench/cells/gemm_tile.sh` 及唯一 official registry/parser（仅关闭真实缺口）；
- selected-route/deployed-symbol trace 与 correctness tests；
- generated source/object/objdump；
- official runs、qualification、受控 master/evidence outputs。

## Dependencies

- B1 control plane 与 B2 bench registry/correctness 基础已完成并归档。
- A2/A3 typed decision/stamping 基础已落；A4a 若改共享 schedule stamp，B5 在固定
  commit 前后做 paired regression，不另建 compatibility reader。
- B6 只在 B5 得到有意义、可部署的 kernel result 后选择性做 e2e 传导。
- IME/P2 standalone 不作为本任务前置，也不并入 RVV row。

## Acceptance Criteria

- [ ] q4_0 flat official request 有唯一 registry/parser/cell，并实际到达 deployed ggml
      consumer；第二 selector、direct benchmark dispatch、旧 symbol alias 与 silent
      fallback 为零。
- [ ] 每个预登记 shape 的 candidates、legality reasons、prior/winner、complete stamp、
      generated object 与 actual concrete symbol 可一一追踪。
- [ ] bench 与 deployed path 消费同一 selected body/stamp；不是复制 metadata 后由
      另一 dispatcher 选核。
- [ ] measured winner 必须属于当前合法集；miss/stale/unqualified/illegal winner
      回合法 analytic prior 或 fail closed，不能增加 candidate。
- [ ] actual deployed ggml、代表性 strong opponent、prior、winner 四列分开，具体
      kernel identity 与 compiler symmetry 可审计。
- [ ] 两个以上合法 generated variants 的差异由 typed g/c/ω 与 formula/legality
      解释，并在 emitted code/objdump 中可见；不要求每个 shape 都 flip。
- [ ] correctness 先于 timing；正式数字有 run-id、三目的地与 near-gate 复测。
- [ ] win/no-flip/loss/wall 均保留；失败尝试不留下 benchmark-only route、format
      特例、dormant strategy 或 compatibility 状态。
- [ ] standalone P2/手调核若报告，明确放在独立 evidence lane，不替代 deployed
      q4_0 路由，也不被误写为同一 selector candidate。

## Verification

- source-front-door/selection/stamp/route/deployed-symbol lit 或 C++ trace tests；
- official parser/route/cell self-tests，CORE==PROD 与 anti-hollow；
- 真板 correctness 后 official cold measure；
- ours/ggml/strong-opponent paired source/object hash、objdump、compiler identity；
- qualification、destination guard、recon、metrics 与 near-gate rerun；
- A4a 前后若共享 schedule 改动，固定输入/板/对手做 paired non-regression。

## Stop Conditions

- 找不到真实 q4_0 producer/consumer 或只能到 test-only abstract contraction：先按
  ISSUE-124 裁定/退役，不能把 lit-only route冒充 deployed；
- route/parser ambiguous、CORE≠PROD、correctness failure、opponent identity unknown：
  停止 timing；
- strong opponent 语义/编译条件不对称：不做胜负结论；
- 没有可解释的 shape/capability variation：报告 no-flip，不制造假 ω/c。

## Out of Scope

- grid GEMM 第二格式、全 shape sweep、全 IME 迁移；
- 用 standalone P2 kernel 替代 deployed flat GEMM；
- 为漂亮峰值新建直达专用核、第二 selector 或 benchmark-only dispatch；
- 在线搜索/runtime sparse/MoE；
- 自动把 micro improvement 外推到 e2e。

## Issue Mapping

- 主项：ISSUE-099（flat GEMM official front door）；
- 边界：ISSUE-124（test-only abstract contraction 不能冒充真实 producer）；
- 与 A4a selected→realized、B6 e2e paired regression 交叉。
