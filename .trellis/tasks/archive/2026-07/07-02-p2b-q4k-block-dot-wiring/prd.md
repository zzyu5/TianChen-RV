# P2-b: Q4_K block-dot production-wiring(close proven-but-not-wired gap)

> 父 [[07-01-arch-refactor-noperand-core]] P2。**目标 = 关掉 README 里点名的 gap:"q4_K proven-decomposable(6-of-7 brick witnesses byte-exact)but not wired to production"。** 让最难的真 kernel(q4_K)走通 production-export 端到端。

## 现状 / 背景

- front-door 存在:`lib/Plugin/RVV/RVVQ4KBlockDotSourceFrontDoor.cpp`。bricks 已证 6-of-7 byte-exact(README line 28)。**"not wired to production"** = front-door → `--tcrv-materialize-emission-plans → --tcrv-rvv-lower-to-emitc` 链没接通。
- block-dot 是**不同 compute 结构**(block-wise dot,非 product-reduction 的 widening_product+reduce)。recon(a762cd80)#3:reuse `ContractionRouteIdentity`(product-head-keyed)会破其语义 → **架构选择**:(a) peer `BlockDotRouteIdentity` descriptor(cleaner);(b) direct-wire(接现有 bricks 到 production-export,不建新 descriptor 族)。**选 cleaner + byte-exact 的。**
- **perf caveat**:q4_K 1.26× micro-win 是 manual-stamped、board-pending、未清 beat bar。**本任务只做 wiring/coverage maturity(工程),不碰 perf claim**(perf 是 board-gated 单独事)。

## 目标(capability-level,chunked 像 C3/C4)

q4_K block-dot 走 production-export 端到端发射(byte-exact bricks 的 dot 计算,正确 block-scale 处理);锁 e2e lit。架构决定用好审美(peer descriptor 优先,若 direct-wire 更 byte-exact-tractable 则用)。

## 纪律

- **byte-exact for existing**:全 zoo + C3 + C4 + dot-reduce 输出逐字不变(check-tianchenrv 758/755/3 + dequant md5 + C3/C4 e2e + N3 facts)。q4_K wiring 是**新 reachability**(front-door 现产的 CORE emit 若已存在,production-export 要匹配它 byte-exact,同 C4 的 CORE≡prod-export 判据)。
- capability-level,batch,verify at boundaries。**STOP-on-irreducible-fork**:若架构决定真两难(peer-descriptor vs direct-wire 各有大权衡)或 wiring 需多天,报告 plan + 首 chunk 进度让我定,别硬 cram。
- 走 trellis-implement;不在主会话写代码。

## 报告

架构决定(哪条 + 为何)、production-wiring 需要啥、q4_K 是否端到端发射(dump emit 证 block-dot 数学正确)或下一 wall/fork;byte-exact 证据;剩余 chunk 的 plan。
