# [RVV-E2E] M0 归因审计 — rvv e2e 0.764× 缺口 = 单因 H-B 内存停顿（2026-07-10）

**战役**: G3-rvv-e2e-gap（目标 0.764× → ≥parity，理想翻正）。**M0 = 先归因禁动刀**。板 `ssh rvv`（openEuler/VLEN128/64c，pin 核 8-15）。A-tree 复原 stock 75f20b5f，未 build tcrv-opt，未 git。genuine（真 PMU 计数器无 multiplex + 真 objdump + 真 gguf 直方图）。

## 一句 verdict
**0.764× 缺口单因主导 = H-B 冷流式内存停顿**。我方 kernel 在 e2e **backend 空转 85.8% / IPC 0.41 / L3(DRAM)-load-miss 2.58× 于 block-dot**，却**少执行 2.3× 指令** → 慢**纯来自访存停顿，非计算质量、非 repack、非 TLB**。根因 = kernel 循环序锁死 **4-token 浅瓦片**（对手 16-token 深瓦片），冷流权重重读 **32× vs 对手 8×**（4× 流量）；热 micro（L2 常驻复用）整个藏住 → **这就是 micro 1.884× ↛ e2e 0.764× 的机制**。**compiler 轴已被 clang .o 控住**（部署 gemm.o = 71 vsetvli/3 spill/2240 vwmacc，与 micro seal 逐字同质）→ 纯净 micro↛e2e 蒸发，**区别于 0.334× 那次的 gcc-codegen 问题**（本因在我方 schedule 域内、可修）。

## perf 计数器（pp128 -r3, n=2, counter IQR<0.15%）
| 指标 | 我方 Aclang | block-dot Bq4kOFF | ours/opp |
|---|---|---|---|
| avg_ts | 2.798 | 3.680 | **0.760×** |
| instructions | 1.172e12 | 2.655e12 | **0.44×（我方少）** |
| IPC | 0.41 | 1.22 | 0.34× |
| backend-idle | **85.8%** | 61.7% | +24pt |
| **L3(LLC)-load-miss** | **4.63e9** | 1.80e9 | **2.58×** |
| dTLB-load-miss | 408e6 | 549e6 | 0.74×（我方更少） |

分解：吞吐比 = cycles_opp/cycles_ours = 2.176/2.872 = 0.758×。我方 busy 周期反而**少一半**（少指令），但 stall 周期多 1.83× → 多出的 1.12e12 停顿周期淹没 0.43e12 计算节省。**100% 访存停顿。**

## 三假设裁断
- **H-A repack 计时归属 → REFUTED（~0%）**：权重重排走 `ggml_backend_cpu_repack_buffer_set_tensor`（`repack.cpp:4733`）= **装载期一次性 set_tensor**，不在 `forward_mul_mat`（:4253）计时窗；窗内只有 q8_K 激活量化（A/block-dot 共模）+ 分派。反证：我方**少 2.3× 指令**（若在线 repack 落窗内应更多）。
- **H-B 冷流布局 → CONFIRMED 唯一主导（~100%）**：机制（objdump/静态）= 部署 `gemm_q4_K.inc`(md5 90d454da) 循环序 **row-group 外(:15) → col-group 中(:20) → K 内(:41)**，权重基址只依赖 col-group → **每 4-token row-group 重扫整块权重面板**，nr=128 → 重读 32×。对手 `ggml-cpu.c:1348-1383` **16×16 cache-blocking** → 重读 8×。**我方 4× 权重流量**。板 cache：L1d 64KB/核、L2 2MB/4核、L3 64MB；ffn 权重面板 ~1MB/线程×4/L2=4MB **超 L2** → 32× 重读落 L3/DRAM。热 micro（K=2048/nr=64/nc=512、20 迭代 L2 常驻）从不为权重付 DRAM → 藏住。
- **H-C 形状失配 → REFUTED as framed（~0% 独立，有 tiling 交互）**：真实 nr=**128**（非任务假设的 64；0c.log "nr=8" 是 smoke-test 假象）。真实形状 M=128/K=4096 **比 micro(64/2048) 更大**（对瓦片更有利）。但 kernel 的 **4-row token 瓦片是固定属性**（绑死 q8_K INTER_SIZE=4 激活重排），正是它让 H-B 咬人 = schedule/tiling 交互，非独立形状因子。q4_K 张量直方图（走我方 kernel、×32 层、全 M=128）：ffn_gate/up (N=14336,K=4096, 各 37.8% MAC) + attn_q/output (N=4096, 各 10.8%) + attn_k (N=1024, 2.7%)；attn_v/ffn_down/output = q6_K 共模不走我方。q4_K ≈71% matmul（吻合 Amdahl 0.7908）。

## 归因分解：**单因**（H-B ≈100%）。H-A/H-C 各 ~0；H-C 提供 tiling 根因但并入 H-B 修法。

## M1 修法方向（命中 task B 桶：tile 作 schedule 轴 + Zicbop，禁硬编码、能力键控；不实现，等 M0+K1 对照臂齐）
1. **主**：加深 token 瓦片 4→16（对齐对手，权重流量 ÷4）。张力：S6 spill→0 tuned 在 4-row 热 micro；16-row 需 16 累加器超 ≤32 vreg。
2. **更简替代**：**交换外两层循环**（col-group 外/row-group 中）→ 权重变常驻只流一次、改重读**小**的 activation 面板（18KB、L1/L2 友好）—— 直接反转罚项（prefill 权重≫激活，应 hold 权重）。
3. **L2 cache-block** 权重面板 + **Zicbop 预取**下一面板（[CACHE-HINT] 首个实战客户；带宽界为主/隐延迟为辅）。
4. **方法学修**：micro 必须**冷跑**（单遍、工作集>L2）或明标"热上界"；**1.884× 是热缓存 artifact**。这是我方 kernel 的 schedule 问题、M1 可修（区别于 0.334× 的 gcc-codegen 不可控）。

## 跨板对照接口（给 K1-SEAL 臂）
同 recipe（`/tmp/case_0c/m0_perf.sh` 事件集 + A/B 可逆部署）测 IME-GEMM vs block-dot。**判别子 = backend-idle% 差 + LLC-load-miss 比**：K1 也 backend-idle≫block-dot + LLC 放大 → **普适 micro↛e2e 物理**（偏 H-B，rvv 修法可迁）；K1 IME 深瓦片/脉动复用留驻权重、backend-idle 与 block-dot 相当 → 缺口 **RVV 布局特有、K1 可传导**（sealed Win 候选）。

## 论文素材
micro↛e2e 的**机制级证据**（PMU 分解 + 循环序 objdump + cache 拓扑账）——L2 调度/布局对内存层级的适配缺口，是 GAP-1 闭环的实证客户；无论 M1 翻正与否，本审计即档案级素材。口径冻结：M0 是**归因证据**非 perf 主张恢复/撤回，不违冻结。
