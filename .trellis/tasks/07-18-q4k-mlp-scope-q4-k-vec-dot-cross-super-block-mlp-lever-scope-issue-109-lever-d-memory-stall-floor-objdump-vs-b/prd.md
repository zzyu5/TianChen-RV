# PRD · q4_K vec_dot cross-super-block MLP lever scope（ISSUE-109 lever d·B 线判据侦察）

> **权威** = ISSUE-109 四次定格（register-fusion + vwredsum + minterm-vec 三指令流 lever 全 mechanism-proven 全 cold-inert·真墙=memory-stall floor·IPC 0.12·剩余 lever (d)=cross-super-block MLP）+ 《行动书 r3》B 线（**判据=公式墙还是脾气墙·不是赢没赢**·**攻坚前先 objdump 对手**）。
> **性质** = 只读侦察（**objdump 对手 + 可行性判定**·persist research/·零代码改·零板攻）。**产出 = 判 (d) MLP 是公式墙〔可键控·必攻〕还是脾气墙〔micro·走三步登记边界〕+ 可行性 + 若可攻的施工入口**。

## 一、要回答的（B 线判据）
1. **objdump 对手 `ggml_vec_dot_q4_K_q8_K_vl128` 的 memory 行为**：它怎么跨 super-block 重叠 DRAM 延迟（MLP）？—— prefetch？软件流水（多 super-block 在飞）？loop 结构（独立 load streams）？独立归约无依赖链？**具体机制**（objdump + 可能 perf/访存模式）。
2. **我方为何 serialize**：三 lever 后我方 q4_K vec_dot 的访存为何串行（super-block N+1 的 load 等 N 的 reduce？依赖链？单 stream？）。
3. **★判据（核心·B 线）**：
   - **公式墙**（可键控·必攻）：MLP 是**我方决策空间**能算的（如发射多 super-block 在飞的软件流水·独立 load streams·prefetch 提示）·对手存在性证可达 ⟹ 必攻·给施工入口。
   - **脾气墙**（micro·走三步登记边界）：MLP 是**硬件访存微架构脾气**（我方发射控制不了的 DRAM 调度·或对手靠 HW prefetcher 我方发射无杠杆）⟹ 走三步（objdump 对手 → 定 lever → 板测证伪）后登记边界·[K-4] 三态"边界"。
4. **可行性**：若公式墙·我方 emit 能否发射跨 super-block MLP（多 super-block 在飞的循环结构·byte-exact 保持）？涉哪些 brick/结构改？是否须新 schema（回门）？

## 二、上岗（只读·persist research/）
- ISSUE-109（`.trellis/spec/issues/发射器与架构.md`·四次 re-diagnosis）· K-attack-fanout-ledger 机制③ · 三 lever 的 archived task（register-fusion/vwredsum/minterm-vec·objdump + perf 证据）。
- objdump 对手 vl128（`ggml_vec_dot_q4_K_q8_K_vl128`·105 向量·看 load/reduce 结构·super-block 循环）· 我方现 q4_K vec_dot leaf（三 lever 后·串行访存）。
- perf 证据（IPC 0.12·memory-stall·访存模式）。

## 三、产出（research/·非代码·非 spec）
1. **对手 MLP 机制解剖**（怎么跨 super-block 重叠 DRAM·具体 objdump/访存证据）。
2. **我方 serialize 根因**（为何串行·依赖链/单 stream）。
3. **★判据裁定：公式墙 vs 脾气墙**（带证据·B 线判据）。
4. **若公式墙**：施工入口（我方 emit 发射 MLP 的结构 + brick 改 + byte-exact 计划 + 是否回门 schema）·公式占比/地盘预期。
5. **若脾气墙**：三步走完的边界登记建议（[K-4] 边界·ISSUE-109 五次定格·honest-null 或具名-X 带脾气墙标注）。

## 四、纪律
- **只读侦察·零代码改·零板攻**（objdump=读能力事实随便读·测速度有配额→本 scope 不测·只 objdump + 分析）·persist research/。
- **B 线判据是产物**（公式墙必攻 / 脾气墙走三步登记）·非"赢没赢"。
- 若须新 schema/回门 = 单列（同阶段2/ISSUE-116 范式）。
