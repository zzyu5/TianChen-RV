# P2-c: extend block-dot family coverage(q8_0 + measure per-op cost)

> 父 [[07-01-arch-refactor-noperand-core]] P2。承接 [[07-02-p2b-q4k-block-dot-wiring]](trait-keyed monolithic block-dot 生产路机制 + 3 op reachable)。

## 现状 + 本任务目的

block-dot front-door 现存的恰是已 wired 的 3 个(Q40/Q4K/IQ4NL)。**其余 ~23 op 每个需从头建 front-door**(如 q8_0 无 `RVVQ80BlockDotSourceFrontDoor.cpp`)= 真 per-op 工程,非 cheap table-row。

**本任务 = 把 q8_0(最常用 8-bit 格式)走通统一生产路 + 诚实测 per-op cost**(建一个 front-door 到底多贵 → full-zoo 闭合是 tractable 还是 23× grind,informs 后续是否值得继续 grind vs pivot)。

## 目标(capability-level)

q8_0 block-dot 走通 full `tcrv-source-artifact-front-door-pipeline`(front-door → emission-plans → coherence → target-export → riscv64 object),byte-exact for existing。建 `RVVQ80BlockDotSourceFrontDoor.cpp` + `monolithicBlockDotOpTable()` 加一行(q8_0 是 flat 族)+ 锁 e2e lit。

## 纪律

- **byte-exact for existing**:全 zoo + product-reduction(C3/C4)+ 已 wired 3 block-dot op 逐字不变(check-tianchenrv floor 762/759/3,新 q8_0 test +1)+ dequant md5 + q4_K/q4_0/iq4_nl md5 不变。新 front-door + table row side-effect-free。
- q8_0 monolithic op + CORE emit 若已存在,production-export 要 byte-identical CORE(同 C4/q4_K 判据)。若 q8_0 monolithic op 不存在,STOP + 报告(建 op 是另一层工)。
- capability-level;wiring/coverage maturity **零 perf claim**;走 trellis-implement。
- **诚实报 per-op cost**:建 q8_0 front-door + 接线花了多少(LOC / 复杂度 / 有无 bespoke 难点),这决定 full-zoo 值不值得继续。

## 报告

q8_0 是否 reachable(dump emit 证 q8_0 block-dot 数学正确)+ **per-op cost 诚实评估**(front-door 建设的真实工作量 / 是否有可复用模板 / 剩 22 op 是否同构)+ byte-exact 证据。
