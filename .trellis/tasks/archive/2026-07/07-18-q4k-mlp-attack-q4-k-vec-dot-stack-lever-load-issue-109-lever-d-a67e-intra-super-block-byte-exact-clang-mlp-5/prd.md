# PRD · q4_K vec_dot STACK 三 lever + 宽多流 load 攻公式墙（ISSUE-109 lever d·a67e 施工入口）

> **权威** = ISSUE-109 五次定格前的 MLP scope 裁定（task `07-18-q4k-mlp-scope`·a67e·**B 线判据=公式墙·必攻**）+ 《行动书 r3》B 线（公式墙必须攻）。
> **性质** = 公式墙攻坚（[K-10] 结构级独立大构造·**允许诚实缩 scope**）。**验收门带脾气墙裁定**（见 §四）。

## 一、施工入口（a67e scope 已定·非猜）
真墙 = memory-stall floor（IPC 0.12·三指令流 lever 全 cold-inert）。对手 MLP 真机制（objdump 证）= **intra-super-block 宽多流 load**：宽 e8m1(16B) load ×25·~16 独立宽 load 提前打到 v0–v15 **寄存器驻留**再统一消费（高 MLP·重叠 DRAM 延迟）+ min-term/归约/scale 全向量化。我方窄 load(e8mf2 8B·64 tiny)+load 紧贴消费+单累加链 = 低 MLP·串行暴露 DRAM 延迟。

**⟹ 施工 = 三事一起（镜像对手·[K-10] 结构级·非旋钮）**：
1. **STACK 三 dormant lever 首次组合**：`fused`（消 aux8 roundtrip）+ `vwredsum`（register-resident 归约）+ `minterm-vec`（min-term 向量化）—— 三者现各自 gated dormant·**首次同时启用**。
2. **宽 load**：`e8mf2 → e8m1/m2`（窄→宽·减 load 条数·增每 load 宽度）。
3. **寄存器驻留多流 load 提前发射**：~16 独立宽 load 提前打到独立寄存器（v0–v15）再统一消费（高 outstanding memory request·高 MLP·镜像对手）。

## 二、gate + byte-exact
- 新 gate 值（如 `integer_core_lmul="mlp"` 或 `"stacked"`）·**默认 sealed 路径 byte-identical**（改动全在 gate 后）·**默认无新 schema**（复用 gate 加值·a67e 定）。
- **byte-exact 免费**（整数加法结合律·前例 fused/vwredsum/minterm-vec 全 ULP=0）·[K-5] ZERO-MODEL·报 ULP=0。**注意 min-term-active 验证**（ISSUE-114·dmin≠0 arm·别空心）。

## 三、板测（rvv·先测不预告·[§五.15]）
`vec_dot.sh rvv verify/measure q4_K`·flush224·idle 100%·gov performance·2-seed。

## 四、★★验收门（a67e 关键 caveat·脾气墙裁定内建）
我方 emit = C-intrinsic·**调度权在 clang**（register-fusion 已示 clang 不自发产 MLP）。板测后**必 objdump 编译产物**：
- **clang 保住宽多流 MLP 调度**（宽 load hoist 提前·~16 outstanding·高 MLP）**∧ cold≥0.8** → **公式墙翻**（关 ISSUE-109·**真硬赢强手调**·地盘+1·q6_K/q2_K/q3_K same-family 就绪）。
- **clang 保住 MLP ∧ cold<0.8** → 具名 + 新墙诊断（MLP 上了但仍不够·下一 lever·清单状态）。
- **clang 再串行化**（objdump 证宽 load 未 hoist·仍紧贴消费·低 MLP·cold inert）→ **三步走完 → ISSUE-109 第 5 次定格 = memory-scheduling clang-调度脾气墙**（compiler-maturity 族·[连 ISSUE-100/107]）·**维持具名-X 0.186·禁「架构不可达」禁「honest-null」**（结构可达·卡 clang 调度·脾气墙 [K-4] 边界）。

## 五、验收
1. **owned STACK+宽多流 emit**（objdump 证：宽 e8m1 load·多独立 load 寄存器驻留·三 lever 组合·非 re-roll）。
2. **byte-exact ULP=0**（[K-5]·min-term-active arm·非空心）。
3. **cold 2-seed + ★objdump 编译产物验 clang 是否保 MLP**（verdict 三分支·§四）。
4. **成色**：对手=deployed 手调 vl128（宽多流 MLP·手调 STRONG·NON-opp-immaturity）·翻正=真硬赢强手调。
5. **sealed 不动**（q4_K md5 `892b6cf8`·gate 外 byte-identical）·**worktree·禁 commit·禁 add**（main cherry-pick·worktree commit 报 hash·**git add 只纳源·勿纳 build-wt**）·objdump 全量·0 造数。

## 六、触碰集 / 遗留
- 触碰：`lib/Conversion/RVV/RVVToEmitCKQuant.cpp`（q4_K vec_dot·STACK 三 lever + 宽 load·独占）+ Internal.h/test/gate。**worktree 隔离·rvv 板**。
- **遗留**：翻则 q6_K/q2_K/q3_K same-family（同 memory-stall floor·同 MLP lever）扩·+@k1；脾气墙则 ISSUE-109 第 5 次定格（clang-调度·具名-X 保留·禁架构不可达）。**须新 schema（若多流须 IR 表达 unroll/stream 因子）=停报 main**（回门·ISSUE-116 范式）。
