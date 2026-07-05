# 出行期自主裁决 ledger（2026-07-05 起，用户离线）

常备令：三线并行（L1 T6 harness / L2 harness-free 预备 / L3 coverage IQ）；文件集互不相交才并行、交集串行；遇 fork 按规则自决不 ping；绝不自决=开第四线/放宽正确性门/canon-schema.def 级/删活消费者（park 入返程包）；STOP 仅 3 种（结构死墙/双板全不可达无替代/falsifier 红因不明）→ 切他线永不闲置。返程交付=一页 checkpoint + 问题包。

自决规则序：①触红线?→选不触;②在关键路径?→选在的;③可逆?→直接做;不可逆两案皆合规→选证据多+记此 ledger;仍平局→park+推他线+入问题包。

---

## Decisions

### D1 — 三线并行拓扑（触碰集 diff）
**判定**：L1(touch=`experiments/` harness + 只读 export) 与 L2 prep(touch=`RVVContractionPathSelection.cpp`+`RVVCapabilityProfile.h`) **交集空 → 并行**。L2③ q4_0 repack construction + L3 IQ construction **都触 ODS/front-door/schema six-state/e5_strong_readout + 可能 ContractionPathSelection → 交集非空 → 串行**（L2 prep 完 → L2③ → L3；L1 全程并行，disjoint）。
**规则**：③可逆(diff 预计)+关键路径。已 dispatch L1(a8aa18fc)+L2(ae2581d6)。

### D2 — 构造线串行（ninja-lock 现实，非仅源文件不相交）
**判定**：源文件集不相交是并行【必要非充分】条件——子代理共享主树 `build/`，并发 `ninja` 撞 build-lock（ninja 单进程锁，第二个报 busy）+ ODS `.cpp.inc` 每次重生（见 [[build-incremental-unreliable]]）。故【本地 build 型构造线并发上限=1】。安全拓扑：**L1(harness，主要跑硬件、本地 build 前置一次) 全程并行 + 构造线(L2prep→L2③ repack→L3 IQ)串行**（一次一个本地 build）。L2③/L3 共享 ODS/six-state/e5 本就要串（D1），叠加 ninja-lock 只强化。
**规则**：①红线无②关键路径（避免 build 竞态毁构造）③可逆。此刻不派 L3——L2③/L3 正当串行等 L2 prep 完，非闲置违规（STOP-永不闲置 针对【线撞墙】，非【正当串行等待】）。

## 返程问题包（park 项，累积）
（暂空）
