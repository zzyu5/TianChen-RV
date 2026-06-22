# N1/N2/N3 establish: make RVV a real compiler (seal A/B → budget-real → compiler-GEMV/breadth → 2nd profile → IME)

## Goal (user, 2026-06-22)
推进三目标**成立**(不是堆 breadth)。顺序 **1→2→5→3→4**(我 §7 status 的五条):
- **① 封 e2e 同-build A/B**(N3 直接观测):在 Build B(编译器-emit 的 ~/tcrv-llamacpp)里测它**自己的 stock**
  (case128→break)→ emitted/stock 比值 → 把 5.84× 从「继承」变「同二进制直接观测」。
- **② 让 budget-divergence 变真**(N3 resource-aware):现 vreg budget 恒=32 → narrow 臂永不触发(dormant)。
  要让**真实约束**(真实 register-pressure / live-vector-group 计数)能压低 budget → narrow 在真实 pressured kernel 上触发,
  而不是合成注入。否则那条 ablation 一直 dormant、不能当 resource-aware 头条。
- **⑤ 拓宽 kernel + 编译器也 emit GEMV**:编译器现只 emit GEMM(prefill),decode GEMV 仍手写 → 让 tcrv-opt 也 emit GEMV;
  按 trunk-discipline 拓宽要服务于「更多 profile 分化 / 更多实测胜出」,不是堆 op 数。
- **③ N1 第二真实 profile**:把 zve32x 从「建模+lit」推到**真硅片 kernel run**(可能用用户的不同 RVV / RVV0.7 芯片)。
- **④ N2 第二 family**:无 IME 硬件;但用户的 **offload 厂商加速可当 N2 的非-RVV 第二 family**(spec 里 Offload 是 sanctioned family)。

## 硬件现实 + 何时报需求(用户)
无 IME 硬件;**有不同 RVV / RVV0.7 芯片 / offload 厂商自定义加速**,若有用,用户后续给账号测。
→ **推完 RVV(①②⑤)后,向用户报硬件需求**(③ 要第二个真实 RVV/RVV0.7 profile;④ 用 offload 当第二 family)。

## 别忘(用户明确)
- **跨拓展插件的 pass 优化** + **方便添加新拓展**(WS-C 的 TunableScheduleOpInterface 已是这条线;继续往「新 op/新 family 复用 tune/pass」推)。
- **RVV 要做成完善的、真正的 compiler**,不能青涩 —— 成熟度 bar 高(承接之前的 maturity 工作:table-collapse / tune-reuse interface / core split)。

## 编译纪律(用户:之前是系统 bug,要小心,先搞清楚再编)
- **板子实况(2026-06-22)**:121GB/98GB free/103GB available(**非内存瓶颈**);64 核;load ~5.9 来自**别的用户**的 llama-cli;
  up 3 天稳定;674G disk free。
- **合理并行 = `-j16`**(16/64 核,几 GB 内存,不 OOM、不霸板、不龟速)。**禁止** `-j4` 龟速过度保守 / `-j64` 霸共享板。
- **timing 抗 load**:别的用户占着 → 绝对 tok/s 受 load 影响;**用同-load 背靠背的比值**(stock vs emitted)抗 load。
- 系统-bug 崩过 → 合理 `-j` + 监控,异常就退;但**别因怕重启就不编**。先搞清楚状况(已做)再编。
- 硬约束续:只用我们自己的 ~/tcrv-llamacpp,**绝不碰** ~/llama_integ* / workspace3/llama.cpp / *_repackON。

## Acceptance (evolving)
- [x] **① same-build A/B 直接观测 — SEALED (2026-06-22)**。同一 Build B 板子、同-load 背靠背:
  emitted(repack ON, ENGAGED=7)= **pp512 9.39 t/s**;stock(case128/256→break, ENGAGED=0)= **pp512 1.57 t/s**;
  **比值 = 9.39/1.57 = 5.98×**。5.84× 不再「继承」——同二进制直接观测,且略高。板子已 RESTORE 回 emitted(return 重新启用 + -j16 重 build 绿)。
  **baseline 性质已 pin(advisor 关切)**:stock 不是 scalar——板子 `ggml_vec_dot_q4_0_q8_0` 在 `#if defined(__riscv_v)` 下有 RVV 路径
  (per-block vle8→vand/vsrl nibble→vwmul/vwmacc→**`vwredsum_vs` 每块 cross-lane reduction wall**→标量累加);`_generic` 仅 `#else`。
  ∴ stock = **naive RVV(带 reduction wall)**,**5.98× = repack-vs-naive-RVV(非 vs scalar)**——干净的 Win B(repack lane 化逃离每块 vredsum 墙)。
  **标签纪律**:5.98× 是 Win B(e2e kernel-swap,直接观测),**不是** N3 Win-A ablation(compiler-auto vs naive RVV 的 2-5×);两者分开,别让 5.98× 漂成 N3 头条。
- [x] **② budget-divergence — HONEST FINDING CLOSED (2026-06-22)**(详见 `artifacts/item2-budget-resource-axes/FINDING.md`)。
  代码实证:narrow 臂 gate = `peakLiveVectorGroups > budget`(`RVVGearboxSchedules.cpp:1674`);budget 全 shape 恒 **32**;
  **所有真实候选 peakLiveVectorGroups = 4(default)/5(packed-i4)/7(grouped)/8(composite)**——最宽也 **4× 低于 32**,narrow 臂
  在任何真实 kernel 上**不可达**,只有 op-attr 合成压低 budget 才触发(=用户拒绝的合成注入)。**RVV 上 budget 轴 DORMANT**
  (RVV 架构恒 32 vreg,zve32x/RVV0.7 也是 32,没有更少的 RVV profile)。**真正 LIVE 的 resource 轴 = m8 EMUL cap**
  (`RVVGearboxSchedule.h:1964-1990`:m4-source 需 i32m16 超 m8 cap → `continue` 剔除 → m2→i16m4→i32m8 为最宽合法 rung,
  footprint≈12≪32,budget 的 `isLegal≤32` 对每条 rung 都 true、从不剪枝;**剪枝的是 EMUL cap 不是 budget**)+ VLEN/Zvl128b capability
  (8-lane strip)。→ N3 resource-aware **头条挂 EMUL cap(真实每次触发)+ capability**,budget 轴诚实标 dormant-on-RVV,其 divergence
  推到 ④(非-RVV 第二 family,更小/typed register file)/受限 profile,**不造假单芯片 ablation**(advisor 认可:清晰标 dormant 比假 ablation 更经得起审稿)。
- [x] **⑤ tcrv-opt emit GEMV — DONE + e2e SEALED (2026-06-22)**。新增 `GgmlRepackGemvQ40Q80Op`(`tcrv_rvv.repack_gemv_q4_0_q8_0`)
  + fail-closed verifier(pin GEMV ABI:plain q8_0 stride34/quant+2,区别于 GEMM 的 interleaved 136/+8)+ 结构化 emitter
  `emitRepackGemvQ4_0Q8_0`(raw()=0)+ recognizer/dispatch。**本地实测绿**(lead 独立验证,非仅 agent 报告):
  raw()=0;两条新 lit(conversion default+NOWALL、dialect verifier)经真 tcrv-opt 二进制 PASS;
  emitted GEMV 结构 = 8× lane-wise `vwmacc_vx_i16m1` + 8× vfmacc/vfwmul scale-fold + 4× vfcvt + **0× vredsum(无 reduction wall)** + 4× vse32,192 行结构化 emitc。
  **板子 I8 已封(2026-06-22)**:
  (1) **数值 PASS(含 multi-group,advisor 关切已封)**:emitted GEMV vs ggml `_generic`,board clang18 rv64gcv_zvfh,0 vredsum。
  单 group(NC=16)norm signed 1.90e-5 / nonneg 8.88e-6;**multi-group NC=16/32/64/336**(336=真实 decode 宽=21 groups,
  压到 emitted 的 outer `nc/16` 循环 + 跨组指针算术 `vx+=x*nb*288`/`s+=x*16`,NC=16 永不触及)**全 PASS**:
  signed norm 2.07e-5@NC336 / nonneg 1.06e-5@NC336(≪1e-4,30 trials × 5 sizes × 2 模式)。
  ∴ 不只单组、是**真实 decode 多组宽度数值正确**。`artifacts/emit-repack-gemv/numeric-verify.log`(+ multi-group run)。
  (2) **decode-tg ENGAGED**:emitted GEMV 经 `mlir-translate` 出 `.inc` → 接 board `arch/riscv/repack.cpp` 的 GEMV VLEN128 路由
  (emitted guard 在 hand block 上方 return,hand 变 unreachable)→ -j16 重 build → **tg128=7.37 t/s**,stderr 实测
  `TCRV EMITTED GEMV(...compiler-emitted) ENGAGED nc=336/352` **与** `TCRV EMITTED GEMM ... ENGAGED` 同时触发。
  ∴ **整个 q4_0 路径(prefill GEMM + decode GEMV)现已全部 compiler-emitted 且在真实 llama 推理里 live**(pp512 9.35 / tg128 5.4–7.4)。
  (3) **decode A/B(同 session、同 toggled binary)**:stock tg128 **1.43 ± 0.01**(ENGAGED=0,tight)vs emitted tg128
  **5.37–7.37**(两次,load-variant)→ **~3.8–5.1×**(诚实标 ~4-5×;emitted 侧受共享板 load 波动,stock 侧 tight;
  与 06-18 独立测的 hand-GEMV 1.38→6.49≈4.7× 同量级互证)。**关键**:同-session stock tg 实测**封住了 06-18 留的跨-session caveat**
  (那时 stock tg 是 banked 1.38、未同 session 复测)。decode 大胜源于 stock naive vec_dot 是**每块 reduction-wall 计算瓶颈**(非 BW-bound),
  repack GEMV lane 化逃墙 → prefill/decode 同样 ~5-6×。详 `artifacts/emit-repack-gemv/decode-ab-finding.md`。板子已 RESTORE 回 fully-emitted(repack ON 重 build 绿确认)。
- [x] **(推完 RVV)③/④ 硬件需求已报(2026-06-22)** — 关键:要的不是「再来一块板」,是**能让 selection/legality 翻转的 capability delta**:
  - **③ N1 第二真实 profile**:需一块 capability 向量**真不同**、能 flip 一次选择的芯片 —— 例:**不同 VLEN**(非 128,如 256/512 → strip 宽度/rung 选择变)、
    **zve32x vs full-V**(EEW/EMUL 合法性变)、或 **RVV0.7 的 ISA delta**(指令可用性变)。一块行为完全相同的第二板**证明不了 N1**(同 kernel 必须得到不同 legality/selection/dispatch)。
    用户的「不同 RVV / RVV0.7 芯片」若 VLEN 或 zve 子集不同即合用 —— **请告知该芯片的 `march`/VLEN/zve 扩展串**,我判断它是否 flip 一条选择。
  - **④ N2 第二 family**:无 IME 硬件;但 **budget-divergence(②)只在 register file 更小/typed 的 family 才 live**——RVV0.7 仍 32 vreg,所以 budget 轴要的是**非-RVV 的 offload 厂商加速**(typed/更小寄存器模型),
    不是又一块 RVV 板。请告知 offload 加速的**编程模型 / ISA / 寄存器模型**(typed?vreg 数?intrinsic 还是 offload-queue?),我判断它能否当 N2 的 zero-core-branch 第二 family + 让 budget 轴真触发。
- [ ] 全程:structured emitc raw()=0、fail-closed verifier、clean rebuild 绿、诚实 ledger、real ssh rvv 证据。
