# Design Boundaries

## Non-Goals

TianChen-RV MLIR 不能变成：

- 又一个高层 tensor/tile IR；
- 一个硬件一个互不相关的 backend dialect；
- descriptor-driven microkernel/exporter 框架；
- 带 `tcrv.matmul` / `tcrv.softmax` / `tcrv.reduce` / `tcrv.generic_tile` / `tcrv.generic_mma` 的通用 compute dialect；
- core pass 里一堆硬编码 backend 分支；
- 保留可执行的 legacy `i32m1` 兼容 route；
- emission-plan/readiness/dashboard/status 当 route/进度/证据 authority 的系统；
- "未来硬件扩展零工作量"的主张；
- "Sophgo/runtime offload 是 RISC-V custom ISA 扩展"的主张；
- 一个"能容纳独立非-RISC-V 离散加速卡（GPU/TPU/910B/寒武纪）"的通用后端 dispatch 壳（见 [Family 准入边界](#family-准入边界n2-identity)）；
- 一篇"普通参数搜索就是主要理论"的论文（tune 必须是 capability/resource-aware 的 N3，不是泛泛 autotuning）。

## Family 准入边界（N2 identity）

一个 family **正当进入** TianChen-RV，当且仅当：**它的能力可以表达为 RISC-V capability 事实，并被同一 core/common pass 零-core-branch 地消费**——而不是它"能挂到 dispatch 接口上"。判据落在 **capability 模型本身的复用**，不在 pipeline 形状的复用。

- **正当（Case A，本质仍是 RISC-V）**：RVV、IME（挂在 RV 核上的矩阵扩展指令）、RVV0.7 / zve 变体、厂商的 RISC-V 矩阵/DSP 自定义扩展。它们有 `march` 串、capability 事实；kernel 经 RV 标量核发射或与 RV 向量协同。RISC-V 的"custom extension"哲学本就允许厂商在 RV 核上加自定义指令/协处理器——这种扩展**本质仍是 RISC-V**，正是 N1（异构扩展作为 capability）与 N2（第二 family 复用同一 capability 模型）要证明的。
- **越界（Case B，稀释 novelty）**：独立的非-RISC-V 离散加速卡——GPU / TPU / 910B / 寒武纪 MLU，作为自带 ISA/runtime、经 PCIe/offload-queue 交互的独立设备。它们与 RISC-V **无架构关系**；若系统"能容纳"它们，恰说明 N2 接口退化成了通用 backend-dispatch 壳（IREE/XLA/TVM 已有），RISC-V 就从**设计中心**降级为初始用例，novelty 稀释成"又一个 MLIR 后端框架"。**明确排除**——不是"能不能塞进来"，而是塞进来就背离 RISC-V-centric 设计中心。

**判别口诀**：第二个 family 要复用 **N1 的 capability 模型本身**，而不只是 pipeline 的形状。能复用 → 证明"capability IR 泛化"（IME 的矩阵能力可表达成扩展存在性 / shape 约束 / SEW-LMUL 类比，core pass 零分支查询）；若必须在 capability 模型**外另起一套**（GPU/910B 的能力不是 RISC-V capability）→ 只证明"搭了个挂任意后端的壳"。

**offload / 厂商加速的判据**：runtime-offload capability 是否正当，看它是否**挂在 RISC-V 核上、经 RISC-V 扩展/能力暴露**（很多嵌入式 RISC-V SoC 的 NPU 如此 → 正当）；若本质是**独立离散卡、只是旁边有个 RISC-V 控制核** → 滑向 Case B。拿到其编程/寄存器模型时第一时间按此判。Sophgo/offload 仍建模为 runtime-offload capability（不当作 custom ISA 扩展）。

## 硬件事实位置

具体目标环境（rvv-main / k1-ime / riscv-sophgo-offload，含 64 核 / `ssh rvv` / sudo 等）是 profile 事实，住在 [profiles](../capability-model/profiles.md)，**不**在本文件、index 或其他 durable 架构文件里复述。任何"真实硬件路径"主张默认指 rvv-main，除非另一环境已被显式 probe 并记录；RVV correctness/runtime/performance 主张要真 `ssh rvv` 证据（I8）。

未来扩展（AME、future custom ISA、其他 vendor）是**插件槽位**，只作未来可扩展目标出现，不是当前必须完成的里程碑或当前硬件证据。无真实硬件/toolchain 证据时，这类工作只能定义通用 future-plugin 准入条件或非主张性设计槽。

## 论文语言（对齐 Novelty）

**用**：

```text
TianChen-RV 是 high-level MLIR 之后的统一 RISC-V MLIR，组织 capability-scoped extension execution。
Capability 对象驱动 variant 生成、legality、selection、dispatch、emission（N1）。
RVV/IME/TensorExt/Offload/未来 vendor 是同一 TCRV 系统内的 extension family，core/common 不按 family 名分支（N2）。
Gearbox 是 capability/resource-aware 的跨 family 调优层，把 selected body 变成调优过的可执行 body（N3）。
当前 lowering route：extension family ops -> EmitC -> intrinsic/vendor builtin/runtime C/C++。
Sophgo/offload 建模为 runtime-offload capability；IME 是评估 plugin-local 接入的第二 family。
```

**避免**：

```text
TianChen-RV 是新 tensor IR。
RVV/IME/TensorExt/Offload 是互不相关的独立 backend dialect。
tcrv.exec.kernel 是硬件 IR 主体。
Descriptor-driven computation 是架构。
Legacy i32m1 route table 是受支持的 RVV 架构。
Source-front-door 生成的 RVV artifact 证明成熟度。
tcrv.exec 拥有 selected RVV route 或 dtype 语义。
Emission-plan status 或 artifact metadata 是 route authority。
Sophgo 是 RISC-V custom ISA 扩展。
独立离散加速卡（GPU/TPU/910B/寒武纪 PCIe 设备）是 N2 第二 family 或卖点。
任何未来扩展零 core 工作即可接入。
单靠 tune / 泛泛参数搜索就是主要理论贡献（tune 只在它是 capability/resource-aware 且实测胜出或与框架自家同-ISA kernel 打平（parity = 主张为真）时才算 N3）。
repack / weight-packing / 算法选择当后端 N3 novelty（实为前端离线-prepack 类，见 system-positioning 的前端/后端判别）。
发明 compiler-DRIVEN/harness-EXECUTED "第三类" 把 repack 升回后端 novelty。
```
