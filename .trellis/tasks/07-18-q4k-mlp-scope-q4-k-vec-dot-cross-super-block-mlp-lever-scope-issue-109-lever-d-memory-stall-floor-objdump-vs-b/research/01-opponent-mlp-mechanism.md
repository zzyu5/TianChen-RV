# Research: 对手 `ggml_vec_dot_q4_K_q8_K_vl128` 的 memory / MLP 机制解剖

- **Query**: objdump 对手 vl128·它怎么跨/内 super-block 重叠 DRAM 延迟（prefetch？软件流水？独立 load streams？无依赖归约链？）
- **Scope**: internal（只读·仓内已有对手 C 源 + objdump 度量·零板攻·零测速）
- **Date**: 2026-07-18

## 证据来源（仓内·可复跑）

| 工件 | 路径 | 内容 |
|---|---|---|
| 对手 C 源（真实现·NOINLINE asm 体） | `experiments/archive/g7/g7-perf-ceiling/kquant-prefill-quality-G1/raw/stock_q4_K.clean.c` | `ggml_vec_dot_q4_K_q8_K_vl128` 全 asm 体（1 super-block = 1 `__asm__ volatile`） |
| 对手 objdump 度量（clang-18 对称·VLEN128） | `experiments/active/g8-stage3-opponent-reparse/rvv/objdump_metrics_rvv.txt` | `q4_K_q8_K_vl128 tot=220 rvv=105 mac=35 vset=7 gather=0` |
| 三 lever objdump 承重数（board-tested） | ISSUE-109（`.trellis/spec/issues/发射器与架构.md`）+ `experiments/active/result-tables/K-attack-fanout-ledger.md` 机制③ | ours 267→205 ins / 55→64 vec·对手 198–220/105 |

## 一、对手循环骨架（不是 cross-super-block prefetch）

对手**逐 super-block 迭代**，与我方同：
```c
for (int i = 0; i < nb; ++i) {
    const float d    = y[i].d * FP16_TO_FP32(x[i].d);
    const float dmin = y[i].d * FP16_TO_FP32(x[i].dmin);
    __asm__ __volatile__( ...一整块 hand-tuned asm，只触碰 x[i], y[i]... );
}
```
**★关键订正（objdump 直证）**：对手 asm 体内**零** `x[i+1]` / prefetch / 软件流水到下一 super-block 的引用——它**不**做 ISSUE-109 (d) 字面所述的「跨 super-block 重叠」。对手的 DRAM-延迟隐藏是 **intra-super-block（super-block 内）的宽 load 提前发射 + 寄存器驻留**。所以 lever (d) 名「cross-super-block MLP」应读作**「memory-scheduling 轴」总称**，其**首要且已证可达的子杠杆 = super-block 内宽多流 load**（对手实证形态），跨-iteration 软件流水只是其上位延伸。

## 二、对手怎么产生高 MLP（逐指令证据）

对手 asm 体（`stock_q4_K.clean.c` 内计数，谓词 `grep -c` 可复跑）：

| 特征 | 对手计数/形态 | 效果 |
|---|---|---|
| **宽 load** | `vsetivli zero,16,e8,m1` → 每条 `vle8.v` **16 字节**（满 VLEN128） | 一条 load 覆盖 16 元素 |
| **load 条数** | **25× `vle8.v`** + 2× `vle32.v` | 覆盖 128B 打包权重 + 256B q8 激活 |
| **load 提前发射（MLP 核心）** | ~16 条独立 `vle8.v` 打到 **v0–v15 独立寄存器**，**在** `vwmul`/`vwmacc` 消费者**之前**发射（源码 asm 里显式手排：`vle8 v1,v2,v3,v8,v9,v10,v11,v12,v13,v14,v15,v0…` 一串后才 `vwmul.vv`） | 硬件 load 单元可同时挂多个 outstanding miss = 高 MLP → 隐藏 DRAM 延迟 |
| **权重解包寄存器驻留** | 4-bit nibble 用 `vsrl.vi`/`vand.vi` **在寄存器内**拆（`v0=v0&0xF`, `v4=v0>>4`）·**0 处**把解包权重存回 scratch | 无 store→load roundtrip |
| **scratch 用量** | 仅 **2× `vsse32.v`**（写 `utmp[4]` 的 6-bit scale 解包·极小） | 唯一 scratch·非权重 |
| **min-term 向量化** | `vle32 bsums` → `vnsrl`/`vadd` → `vzext.vf2 mins` → `vwmul.vv` → `vredsum.vs` → `fnmsub.s`（**0 处 scalar `lh`**） | min 项全向量 |
| **sub-block 归约寄存器驻留** | **8× `vwredsum.vs`**（e16m2·16-lane sub-block-half 归约·`v0`=zero 种子） | 无 i32m8 serial 链 |
| **scale-fold 向量化** | **8× `vmul.vx`/`vmacc.vx`**（6-bit scale 直接 vector-scalar MAC·配 8× scalar `lbu` 取 scale byte） | scale 全向量 |

**⟹ 对手 MLP 机制 = 三件叠加**：(1) 宽 load（e8m1 16B）·(2) 权重寄存器驻留（0 权重 scratch）·(3) **~16 条独立宽 load 手排提前发射到独立寄存器**（intra-super-block 多流）。三者共同让「加载 super-block i 的 128B 权重 + 256B 激活」的 DRAM 延迟**并行在飞**，逼近带宽。

## 三、对手 = 手调 STRONG（成色如实）

- objdump caliber 机判 = **手调**（`rvv=105` 非 0 向量·非便宜 `_generic` 档）。
- 对手用 **inline asm 手排 load 调度**（绕过 clang 调度器）——这是后文判据的关键：对手的 MLP 是「人手排的确定调度」，不是 clang 从 C 自动产出的。

## 四、这条杠杆为何**未被三 lever 触及**（清单非空的根据）

三 lever（register-fusion / vwredsum / minterm-vec）board-tested EXHAUSTED，但它们改的是**指令流轴**，**没有一个改 load 宽度或 load 流并行度（MLP）**：
- register-fusion：消 aux8 权重 scratch roundtrip（= 对手特征 2），但**保留同一「逐 super-block、load 紧贴消费者」的窄结构**·cache-miss 33×↓ 但 cold 0.152 不动。
- vwredsum：复刻对手特征 6（8× vwredsum.vs 归约）·cold 0.162 不动。
- minterm-vec：复刻对手特征 5（向量化 min-term）·cold 0.164 不动。
- **未复刻**：对手特征 **1（宽 e8m1 load）+ 3（多流提前发射 MLP）** —— 正是隐藏 DRAM 延迟的机制。⟹ **memory-scheduling 轴 UNTRIED**。
