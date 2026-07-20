# B5：GEMM shape、选择与 deployed ggml 路径

## Goal

用少量真实 shape 验证 generated candidate、legality、selector 与实际 deployed ggml 路径一致，并分别报告 deployed ggml、代表性 strong opponent、analytic prior 与 qualified winner。该任务证明同一框架能生成并部署多个合法实现，而不把“真实路径”与“强对手”互相否定。

## Pre-registration

- op：`gemm_tile`/对应真实 repack-GEMM path。
- primary slices：一个 flat 主格式（优先 q4_0）+ 一个已有 grid 资产格式（优先 iq3_s/iq3_xxs）；第二项仅在 B1 资产矩阵证明无需重复基建时纳入。
- regimes：prefill 为主，必要时保留 decode/M=1 对照但不同 row 分账。
- boards：rvv、k1；engine=rvv。IME q4_0 只在 A6 结构完成后作为单独 engine 行，不混入 RVV。
- opponents：实际 deployed ggml path、代表性手调/ISA-specialized kernel、Weft prior、合法 qualified winner。
- correctness：双方 vs oracle、派发身份、byte-exact/ULP、CORE==PROD。
- timing：official cold、同板同链同会话；shape/重复数预注册。

## Scope

- 固定 2–3 个真实 shape bucket：M=1 边界、代表性 prefill tile、必要的中等 shape。
- 记录每个 shape 的 candidate set、被剪候选及 legality reason。
- route/selected-body trace 证明请求和实际执行 variant 一致。
- 补齐 flat GEMM official harness 时与 B2 协调文件 owner。
- 优先比较已有 generated paths，不为漂亮峰值新建绕过 plugin path 的专用核。

## Primary Touch Set

- GEMM selection/front door/body/emitter 的必要局部改动
- `tools/bench/cells/gemm_tile.sh` 和真实 deployed harness
- selected route trace tests
- official run/master/evidence outputs

## Dependencies

- 依赖 B1 control plane；缺 harness 的 slice 依赖 B2。
- A2/A4 若改变 decision/stamping，任务按固定 commit 做前后 paired run。
- B6 消费本任务闭环 slice。

## Acceptance Criteria

- [ ] 每个 shape 的候选、legality、selected stamp 和实际 route 一致可查。
- [ ] measured winner 必须在当前合法集；miss 回 analytic prior。
- [ ] deployed ggml、strong opponent、prior、winner 四列分开，compiler symmetry 明确。
- [ ] 两个格式/多个 shape 的选择差异由 typed facts/capability/context 解释。
- [ ] 负结果明确归因 formula、legality、selection、emission 或 HW/compiler，不删行。
- [ ] 正式数字有 run-id，写三目的地；近门复测。
- [ ] 不把便宜档胜利写成击败手调。

## Verification

- selected-route/dispatch tests;
- official bench verify/measure;
- objdump and compiler identity;
- master/recon/metrics consistency;
- paired regression across A-line commit where applicable.

## Out of Scope

- 在线搜索、全 shape sweep、整套 IME 迁移、以 standalone P2 核替代 deployed flat-GEMM 对手。

## Issue Mapping

- ISSUE-097/098 由 B1 前置；ISSUE-099 的 flat harness 部分与 B2 协作；关联现有 GEMM/selection issues。
