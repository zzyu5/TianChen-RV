# B4：A3 Codebook 真板性能 disposition

## Goal

验证 A3 已完成的 Codebook typed g+c 公式是否沿真实 dequant 路由传到板上，并对
当前 capability-selected anchor 做正确性、objdump 与性能 disposition。B4 不重新
发明 codebook 公式，也不重启 ISSUE-107 已经证伪或过期的旧 lever；它回答的是：

1. 当前两块真实板各自声明了什么 RVV capability，A3 公式因此选择什么合法 plan；
2. selected complete stamp 是否经 canonical deployed route 变成预期 intrinsic/
   LMUL chain；
3. 相对真实 deployed ggml 与一个代表性强对手，结果是 win、no-flip、loss 还是
   compiler/HW wall；
4. 这些实测只更新 qualified ranking/disposition，不反向改变 candidate、legality
   或 emitter。

## Why This Is the Next Performance Step

A3 已证明 VLEN64→m2、VLEN128→m1、VLEN256+fractional→mf2 三条 emission chain
可构造，但这只是编译器结构证据。B4 补真硬件 evidence：当前物理板实际属于哪条
capability 路径、生成代码质量如何、是否在真实对手面有收益。VLEN64 若没有对应
物理板，只保留 formula/emission 决定性证明，禁止伪造“第三块板”性能格。

## Bounded Pre-registration

启动时先由 runner/probe 与 registry 决定精确行键，随后冻结，不凭 task 文本猜值：

- op：`dequantize_row`；
- mechanism：A3 small-codebook；
- formats：从 A3 canonical layout rows 中选择最多两个 runner-supported、能覆盖
  不同真实 capability selection 的代表 cell；若两板落同一 plan，则只报告真实
  no-variation，不人为制造 c 轴；
- boards：现有 `rvv`、`k1` 中通过 canonical probe/route/correctness 的板；
- engine/regime：按 official registry/master 的完整行键；
- repeats/warmup/cold policy：按 measurement spec 预登记；
- opponents：实际 deployed ggml dequant symbol + 一个代表性强对手（若存在合法、
  同语义、同编译条件的实现）；二者互补，不能互相替代；
- correctness：oracle/规定 ULP 或 byte-exact、ZERO-MODEL、anti-hollow、
  CORE==PROD、dual-board parser/route identity；
- artifacts：selected stamp、generated C/object、objdump、compiler identity、raw JSON、
  run-id、qualification 与 disposition。

最多两组 official paired codebook cells；不扫完整 grid/codebook family，也不把单格
结论外推到其它 ABI/layout。

## Scope and Order

1. **只读 freeze**：确认 dequant official parser、route registry、CORE==PROD、两板
   capability instance 与 deployed opponent identity；任何 unsupported/ambiguous 先
   fail closed，不直接计时。
2. **正确性门**：同一 selected plan 在 local compile evidence 与真实板 runtime
   evidence 对齐；若 byte-broken，停止计时并登记 correctness failure。
3. **paired codegen**：对 A3 前的旧 anchor 与当前 c-selected anchor（仅在两者都
   合法且有真实 realization 时）做同输入 paired source/object/objdump；检查 load
   LMUL、widen chain、strip/VLMAX、spill/register pressure 与实际 gather/load 形态。
4. **official cold**：在同板同编译器同会话比较 Weft deployed route、真实 ggml 与
   代表性强对手；near-gate 独立复测。
5. **disposition**：把 win/no-flip/loss/wall 写入唯一 run lineage/qualification；若
   measurement winner 与 analytic prior 不同，只生成合格 winner row，不能新增
   candidate 或绕过 legality。
6. **retirement**：失败尝试不留 benchmark-only route、format-name branch、环境
   开关或 dormant emitter branch；原始 run 与负结论保留。

## Multi-Agent Collaboration Contract

多个 agent 共同做同一个 B4 campaign：

- route/authority agent 追踪 request→formula→stamp→generated object→deployed symbol，
  并保护 A3 单一 collector/materializer；
- opponent/codegen agent 独立核对 ggml/强对手身份、compiler symmetry、objdump 与
  register pressure；
- evidence agent 复核 pre-registration、correctness gate、run lineage、qualification
  与 master promotion 条件；
- root integration owner 串行占用真板、执行 official writer、裁决同一行键并合入。

只读、代码与证据审查可并行；同一板、同一 run-id 和 master row 只能串行由 root
写入。任何 agent 的 standalone micro 结果都不能替代 canonical deployed route。

## Primary Touch Set

- `tools/bench/bench` 已登记的 `dequantize_row` cell/registry/parser（仅补真实缺口，
  不建第二 dispatcher）；
- A3 Codebook generated route 与 focused trace/correctness tests；
- 必要的 codegen lever（仅在 objdump 给出可解释、公式域内的结构理由时）；
- `experiments/runs/<run-id>/`、`runs.log`、qualification 与受控 master promotion；
- performance/measurement issue 与 evidence ledger。

## Dependencies

- B1 control plane、B2 route/parser/correctness 基础与 A3 Codebook 已完成并归档。
- 若 official dequant parser 或某板 CORE==PROD 尚缺，先在本任务内原子补齐该真实
  前门，删除 benchmark-only bypass；不能以脚本目录存在冒充覆盖。
- 与 A4a 的纯 schedule code 可并行审查；A4a/A4b 改到共享 Codebook preparation
  时，由 root 串行合入后再冻结 B4 commit。

## Acceptance Criteria

- [ ] 精确行键、format、board capability、selected plan、opponent symbols 与重复数
      已冻结且可机读；没有靠 board 名猜 VLEN/LMUL。
- [ ] 最多两个 representative codebook paired cells 完成 canonical parser/route、
      CORE==PROD、正确性与真实板运行；无物理 VLEN64 时不写 VLEN64 性能结论。
- [ ] request→typed g/c→legal plan→complete stamp→generated object→deployed symbol
      全链可追踪，direct/benchmark bypass 为零。
- [ ] deployed ggml 与代表性强对手分别报告；对手具体 kernel identity 与 compiler
      symmetry 可审计。
- [ ] ours/opponent objdump 给出 load LMUL、widen/gather、spill/register pressure 的
      结构差分；不以源代码意图代替生成物事实。
- [ ] win、no-flip、loss、correctness failure 或 compiler/HW wall 均有 raw run-id 与
      disposition；失败尝试不留 production 垃圾路径。
- [ ] measurement 只在 A3 已声明合法集内影响 winner ranking；formula/candidate/
      legality/emitter 没有因某块板的结果被反向改写。
- [ ] official writes 走唯一 guard 和三目的地；近门独立复测，recon/metrics 一致。
- [ ] 单格结果不外推 grid/codebook family，不自动外推 e2e。

## Verification

- focused formula/stamp/route lit 与 A3 parity regression；
- official `bench ... --verify` 后再 `--measure`；
- real-board source/object hash、runtime correctness 与 paired objdump；
- opponent identity、compiler flags/identity 与 timing symmetry audit；
- qualification、recon、metrics 与 destination-guard self-tests；
- 若有 code change，强制重链/相关 suite；runtime/performance 结论只引用真板 run。

## Stop Conditions

- correctness 失败、route/parser ambiguous、CORE≠PROD 或对手身份未知：停止计时；
- 三个有解释力的、公式域内结构 lever 均不能改变生成物：登记 compiler/HW wall，
  不转 inline asm/专用绕路；
- 两板实际 capability 相同：报告同 plan/no-variation，不制造假 c；
- measured loss：保留 analytic prior 或 qualified winner 的合法 fallback，删除失败
  production strategy，保留 run。

## Out of Scope

- 重启 ISSUE-107 的旧 gather 猜测清单；
- KQuant MLP（ISSUE-109 已负结论并退役）；
- 全 dequant/grid sweep、nvfp4 vec_dot ISSUE-100、inline asm；
- A4b Grid/KQuant/Ternary formula 重构；
- 用 standalone micro 替代 deployed ggml，或把 micro 自动写成 e2e。

## Issue Mapping

- 主要复核 ISSUE-117 的真实 c 轴与 A3 后性能 disposition；
- ISSUE-107/112 只作历史墙型/候选 cell 参考，不把已证伪旧 lever 重列为目标；
- ISSUE-100 明确留在独立 nvfp4 compiler-call task。
