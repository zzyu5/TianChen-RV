# N3 measured speedup: latency-hiding knob (micro) + own-llama.cpp e2e tok/s on rvv

## HARD CONSTRAINTS (用户明确, 牢记 — 违反即返工)
1. **自己维护本项目专属的一份 llama.cpp**。绝不用"另一个项目 / 别人 / 用户"的 llama.cpp 或 bench
   —— 那个 sibling `/home/kingdom/phdworks/llama.cpp/` 是用户另一个 ggml 优化课题的活,**kernel 不一样**,
   不改它、不借它的 build/bench。我们的推理必须是 TianchenRV 自己接管 kernel 的那一份。
2. **rvv 板子内存有限**:在 rvv 上编译/跑时**核心数压低**(低并行 `-j`,按实测内存定),
   **吃满核 / 吃满内存会出 bug**。不要 `-j$(nproc)`。
3. **llama7b 模型已在 rvv 那台机器上**(用户确认)——用它,不在本地另建大模型。
4. 共享资源只有 `ssh rvv` 板子:别和别的 bench **同时**跑(时序污染);scp 用独立临时路径。

## Goal (user, 2026-06-18 — ELEVATED)
**性能验证只是一小部分;真正要的是性能提升,达到可发论文的成熟标准 —— 目标 ~1.5×。**
不是"测出我们现有的 parity~+5%",而是**真的把 kernel 做快到 ~1.5× over ggml**(并用 e2e tok/s 证明)。
- **Phase 0 — 诚实定位**:现状 block-dot 仅 parity~+5% vs ggml;roofline 实测 kernel 只到 compute ceiling
  **4–13%**(我们和 ggml 都 latency-bound,都把 ~85–95% 算力留在桌上)。所以 1.5× 的 headroom **存在**
  (ggml 也没吃到),但要 1.5× 必须**吃到 latency-bound 那块**,不是调参 ±5%。
- **Phase 1 — micro 攻 latency 瓶颈(真加速的来源)**:瓶颈 = per-block `vwredsum`→scalar→fp32 依赖链
  + decode 串行。候选 lever(实测驱动,别静态推):**deferred 向量累加器**(i32mX 累加,跨 block 推迟 reduction,
  避开 per-block 标量回合 —— 代码里已有 `widening_accumulate`/`deferred_accumulate` 路径)、**多独立累加器 /
  软流水深度**、**decode↔compute 解耦**。逐字节 gate(vs ggml-real)先过再排名;在 micro-harness 上迭代到看见
  接近 1.5× 的 kernel 加速,否则诚实报告能到的天花板。
- **Phase 2 — e2e 证明**:用**我们自己的** riscv llama.cpp(非 sibling),接管 q4_0 dot,在 rvv 的
  **`/home/ubuntu/llama-2-7b-chat.Q4_0.gguf`**(Q4_0 7B,kernel 完美对上)跑 `llama-bench`,量 decode tok/s
  before/after。decode(batch=1)主导算子就是 q4_0 dot → tok/s 近似直接反映 kernel 加速。

## 诚实风险(1.5× 不是保证,是研究赌注)
ggml 的 RVV q4_0 kernel 已不差,我们现在 parity。1.5× 要靠吃 latency-bound headroom(deferred accumulate /
多累加器 / 软流水)—— roofline 说这块 ggml 也没吃,理论上有;但能吃到多少要 micro 实测才知道。先做 q4_0 一个
kernel 打透看能否逼近 1.5×,再决定推广。**到不了就诚实报天花板,不灌水。**

## 为什么 e2e 可归因(不是模糊"整体涨一点")
batch=1 解码 = 一连串 `ggml_vec_dot_q*_*`,**block-dot kernel 就是解码主导算子** → decode tok/s 几乎直接反映 kernel
加速(不被 Amdahl 稀释到看不见);GEMM 赢体现在 prefill。模型量化方案要对上我们赢的 kernel(理想 Q4_0/Q4_K)。

## 已知基线(归档实测, ssh rvv VLEN=128)
- tune vs 静态 pick:q4_1 block-dot ~1.20×(把 0.842× 亏翻成 1.012× 赢);GEMM M-block M=6 ~1.19×(vs vec_dot)。
- vs ggml 绝对:block-dot parity~+5%(q4_0 +4.7% / q4_1 +1.2% / q8_0 −1%);GEMM +19%。
- roofline:kernel 只到 compute ceiling 4–13%,latency-bound(per-block reduction→scalar→fp32 + decode 链)。
- ⚠️ 张力:roofline 说"latency-bound→多 overlap",但 inc10 实测 factor=1(最少 overlap)反而赢 → lever 不能靠静态推,
  必须实测。latency-hiding 的真 lever 还要在 micro 上探(可能是 accumulator 独立性 / 软流水深度 / decode-compute 解耦)。

## Orientation (摸过 rvv, 2026-06-18)
- **Board**:`ssh rvv` = riscv64,ISA `rv64imafdcv...zfh zvfh zve32/64...`(全 RVV + fp16),**64 核 / 121GB / 880G 盘
  (674G free)**,native `clang/clang++/gcc/g++/cmake`,**`mlir-translate` 不在板上**(emit C 在本地做、scp 上去板上编)。
  VLEN=128(inc10/inc26 已定)。⚠️ 用户硬约束:build/run **低 `-j`**(别 `-j64`,吃满核/内存出 bug)。
- **模型**:**`/home/ubuntu/llama-2-7b-chat.Q4_0.gguf`**(Q4_0 Llama-2-7B,在 `~`,不在任何 llama_integ 里)= 首选
  e2e target,kernel 完美对上 q4_0 dot。另有 `~/workspace/workspace3/DeepSeek-R1-Distill-Llama-8B-{Q2_K,Q4_K_M}.gguf`
  (Q4_K_M 可测 q4_K kernel,备选)。
- **⛔ 别碰的 sibling/foreign builds(都是另一个项目的)**:`/home/ubuntu/llama_integ_control_repackON`、
  `llama_integ_canary_repackON`、`llama_integ`、`llama_integ_override_m1`、`~/workspace/workspace3/llama.cpp`。
  这些有现成 `build/tools/llama-bench` 但 **kernel 不是我们的**,不用、不改。我们自己 clone 一份干净 llama.cpp
  到独立目录(如 `~/tcrv-llamacpp/`),低 `-j` build。
- **kernel 接管方式(待定, Phase 2 设计)**:把 autotuner 选中变体的 emitted C 编进我们那份 ggml 的 dispatch
  (替 `ggml_vec_dot_q4_0_q8_0`),保持 byte-exact vs ggml-real。

## Acceptance (evolving)
- [ ] Phase 1: micro 上一个 latency-hiding 改动,逐字节 gate 过,实测 kernel 加速(诚实 win/parity)。
- [ ] Phase 2: 自己的 riscv llama.cpp(非 sibling)在 rvv llama7b 上 llama-bench,decode/prefill tok/s before/after,
      诚实数字(含 Amdahl 归因)。
- [ ] 全程不碰 sibling llama.cpp;build 低 `-j`;byte-exact 正确性 hold。
