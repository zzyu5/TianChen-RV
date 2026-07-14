# G8 §六.3 攻坚 · rvv 板 nvfp4 vec_dot 真攻坚 (bounded·2026-07-15)

> 执行 agent(host-side ssh rvv)。目标格：`vec_dot/nvfp4` @ rvv(VLEN128)，T3_A 行 = `iq/fp4 vecdot` 桶第4行(cold=0.565·`predicate-NA-vs-generic-fp4`)。
> **结论（先说）：② 决令二具名-X 认输**（非翻正）。一轮构造尝试 G1 byte-exact GREEN、G2 板测 attack_over_base = 1.0015–1.0046×（噪声内，NULL）。残余具名 X = `[GAP-NVFP4-VLEN128-SUBBLOCK-GRANULARITY-SPILL]`，见 §4。
> 板 scratch 已清（`/tmp/g8s3-nvfp4-attack` 删除）；`/tmp/g8s3` sealed 与 `/tmp/opp_link_stubs.*`（非本役产物）未动。禁 git，本 casefile 供主会话整合。

---

## 0. 起点 (T3_A / §五 / §六 既有证据，不重测，仅索引)

| 项 | 值 | 来源 |
|---|---|---|
| T3_A cold ratio | hot=0.661 / **cold_med=0.565** / M8=0.662 | `g8-stage3-cold-remeasure/rvv/summary_0p8_rvv.csv` L20 |
| 0.8 门判读 | FAIL | 同上 |
| 对手真派发符号 | `ggml_vec_dot_nvfp4_q8_0`（经 `arch-fallback.h` 宏 `#define ggml_vec_dot_nvfp4_q8_0_generic ggml_vec_dot_nvfp4_q8_0` 别名到 generic 实现） | §五 `objdump_metrics_rvv.txt` + 本役 `nm` 复核 |
| 对手成色（既有判） | `native-vec-generic-weak`，tot214/rvv23/mac2/vset3/gather0/csrr_vlenb0，**NO-riscv-spec** | `spec_rvv.tsv` |
| 板 | rvv openEuler VLEN128 64c，clang-18.1.8 canonical 对称域 | `g8-stage3-opponent-reparse/evidence.md` §0 |

---

## 1. 解剖 (objdump ours vs 真派发对手 · 符号级)

### 1.1 对手身份坐实（反稻草人）

在 rvv 板 `/home/ubuntu/vericurve-rv-lab/llama.cpp`（ggml pin `e36a602`）源码级确认：

- `ggml/src/ggml-cpu/quants.c:279` `ggml_vec_dot_nvfp4_q8_0_generic` = 纯 C 双层 for 循环（外层 4 个 sub-block，内层 8 元素 table-lookup 通过 `kvalues_mxfp4[qv & 0xf]` / `kvalues_mxfp4[qv >> 4]`），无 `riscv_vector.h`、无 intrinsic。
- `ggml/src/ggml-cpu/arch/riscv/quants.c`（真实存在、CMakeLists 确认编入）中 **`grep -c nvfp4` = 0** —— **RISC-V 手调层完全没有 nvfp4 override**，退化路径唯一。
- 对照：`arch/arm/quants.c:736` 有 `ggml_vec_dot_nvfp4_q8_0` NEON 手调（`ggml_nvfp4_dot8` intrinsic），证明上游确实给部分架构写过手调实现——**RISC-V 是被漏掉的架构，非我方选弱对手**。

**objdump 实测**（`/tmp/g8s3/rvv/quants_generic_opp.o`，md5=8873178b，与 §五/§六 sealed 对手 md5 一致，clang-18.1.8 重编）：

```
ggml_vec_dot_nvfp4_q8_0   tot=214  rvv=23  mac=2  vset=3  gather=0  csrr_vlenb=0
```

`mac=2`（几乎无真实向量乘加）、`gather=0`（table lookup 未被 clang 识别为可 gather 化，标量索引）——**对手是近标量、被 clang 部分零星自动向量化的弱实现，非手调、非 hand-tuned，"generic 弱" 判定坐实**。

### 1.2 我方核解剖（真瓶颈定位）

我方核 = `weft_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_rvv_nvfp4_q8_0_block_dot`（`g7-census/iqfp4-dequant-rvv/kernels_vecdot/nvfp4.kernel.c`，emitter 产出，4-way 手动全展开：每 64-元素 super-block 拆 4 个 16-元素 sub-block，各自 `vsetvl_e8m1(8)` 处理）。

`--disassemble-symbols` 对本符号截断（已知坑，见 `g8-stage3-opponent-reparse/evidence.md` §0 方法学注）→ 改用全 `.text` dump + 地址切片（`anal.sh`）：

```
weft_emitc_..._block_dot   tot=371  rvv=68  mac=12  vset=17  gather=8  csrr_vlenb=22
```

**关键发现（真瓶颈，非直觉猜测）**：逐行读全反汇编（`raw/ours_baseline_full.dis.txt`）看到密集重复模式：

```
csrr  a0, vlenb
slli/sh#add  a0, a0, ...
add   a0, a0, sp
addi  a0, a0, 0x20
vl1r.v / vs1r.v  vX, (a0)
```

统计：**11× `vs1r.v`（spill store）+ 11× `vl1r.v`（spill reload）+ 22× `csrr a0,vlenb`**（地址算术，几乎全部服务于这 22 次 spill）。这不是"VLEN 硬件分支探测"（对手/K-quant 格里 `csrr_vlenb` 的常见含义），而是 **LLVM RVV 可伸缩向量寄存器栈溢出**：16-entry codebook 表（`v8`/`v14`）以及各 sub-block 已加载的原始字节向量被压到栈上、每次引用时用 `vlenb`-缩放地址重新计算偏移量后 reload —— **常驻的小 codebook 表本该一直留在向量寄存器里，实际却被反复落栈**。

**量化占比**：22(csrr) + 22(spill load/store) ≈ 44 条纯地址算术/搬运指令，占 371 条总数的 **~30%**，且这部分对最终结果 0 贡献（纯开销）。

**根因假说**：K=64 super-block 仅 4 个 16-宽 sub-block，每个 sub-block 需要同时存活：codebook(m1) + 2 个 widen 乘积(m2, `vwmul`/`vwmacc` 各占 2 个物理向量寄存器组) + reduce 累加器模板(m2) + 本 sub-block 的 raw byte / lo-nibble / hi-nibble / 2 段 activation 向量——manual-unroll 4 份直线代码后，寄存器分配器在 e8m1→e16m2→e32m4 三种 SEW/LMUL 反复切换之间无法把 32 个物理向量寄存器打满全部所需 live range，触发溢出。

---

## 2. 构造 (一轮尝试 · byte-exact 优先)

**杠杆**：re-roll 手动 4-way 全展开为真 `for(s=0;s<4;s++)` 循环（假设：减少直线代码体积/寄存器压力，让 codebook 更容易保持驻留）+ 顺带 hoist 掉 baseline 对同一 q8_0 block 的 fp16→fp32 scale 重复转换（sub0/sub1 共享同一个 q8 block 却各自独立 `fcvt.s.h` 一次，sub2/sub3 同理）。

**Byte-exact 安全论证**（写代码前先证明，非事后检验）：
1. hoist 的 `fcvt.s.h` 是纯函数、相同输入比特 → 相同输出比特，提前算一次或延后算两次结果逐位相同。
2. `sumf` 累加顺序完全保持 `s=0,1,2,3` 顺序（与原 4-way 展开逐 sub-block 更新顺序一致），浮点加法结合律敏感的求和顺序未变。
3. 每个 sub-block 内部 decode/dot-product 的 RVV op 序列（vand/vsrl/vrgather/vwmul/vwmacc/vwredsum）逐条照抄未改，只改了外层 C 控制流。

文件：`nvfp4_attack.kernel.c`（本 casefile 根目录）。

### 2.1 G1（byte-exact，DYNAMIC S(new)==baseline + vs 真 oracle）

driver：`nvfp4_attack_driver.c`（3 路并测：BASE / ATTACK / OPP，同一批随机权重/激活网格，逐 cell `memcmp` 整数位比对）。3 个形状全绿：

| 形状 | ATTACK vs BASE | ATTACK vs OPP(真oracle) | BASE vs OPP |
|---|---|---|---|
| K=2048 M=1 nc=512 | 0 mismatch | 0 mismatch | 0 mismatch |
| K=2048 M=8 nc=512 | 0 mismatch | 0 mismatch | 0 mismatch |
| K=2048 M=1 nc=64  | 0 mismatch | 0 mismatch | 0 mismatch |

**G1-GREEN**（本役构造 100% 保值，且 baseline-vs-opp 0-mismatch 复现了 §六 sealed 的 byte-exact 结论，同 session 内部自洽）。

### 2.2 G2（clang-18 对称域 cold A/B，rvv 板实测）

协议：core8 pin（测前 1s `/proc/stat` 采样确认 core8-15 全 0.0-1.0% busy）、co-tenant（2 users）未扰、224MiB flush 逐 rep、N=12、median-robust、`M=1 GEVM K=2048` 为主判 + M=8/nc=64 两个对照点。

```
K=2048 M=1  nc=512 : COLD base_med=12.21ms(iqr0.4%) attack_med=12.18ms(iqr0.3%) opp_med=6.95ms(iqr0.6%)
                     ratio_cold_base=0.5693  ratio_cold_attack=0.5709  attack_over_base=1.0028
K=2048 M=8  nc=512 : COLD base_med=82.93ms   attack_med=82.80ms        opp_med=55.42ms
                     ratio_cold_base=0.6682  ratio_cold_attack=0.6692  attack_over_base=1.0015
K=2048 M=1  nc=64  : COLD base_med=1.534ms   attack_med=1.527ms        opp_med=0.874ms
                     ratio_cold_base=0.5694  ratio_cold_attack=0.5720  attack_over_base=1.0046
```

（重复一轮验证稳定性，见 `raw/nvfp4_attack_run.log`：三形状 attack_over_base = 1.0016–1.0112×，两轮共 6 个数据点全落在 1.0015–1.0112× 区间。）

loadavg：begin 2.13–2.25 → end 2.33–2.50（在既有会话漂移带内，非本役自身计算导致——64 核机上 core8 单核工作不解释 load-1 抬升）。

**结论：attack_over_base ≈ 1.0015–1.0112×（噪声地板内，o_iqr 0.1–0.9%）—— NULL，未构成有意义加速，远不足以把 cold ratio 从 0.57 推到 0.8 门。**

### 2.3 事后反汇编复核（诊断为何构造失效）

对 `nvfp4_attack.kernel.c` 重新 objdump（`raw/ours_attack_full.dis.txt`）：

```
weft_emitc_..._block_dot_ATTACK   tot=363  rvv=68  mac=12  vset=17  gather=8  csrr_vlenb=22
```

**spill 计数与 baseline 完全相同**（11× `vs1r.v` + 11× `vl1r.v` + 22× `csrr vlenb`，逐一核对），总指令数仅从 371 降到 363（−8，恰好对应 hoist 掉的 2 次冗余 `fcvt.s.h` + 相关 load，与测得的 ~0.2-1.1% 提速量级吻合）。**"re-roll" 假说未成立**：clang-18 `-O3` 把重新写好的真 `for(s=0;s<4;s++)` 循环在后端又整个展开回几乎相同的直线代码，寄存器压力/溢出是**数据流固有属性**（4-way widening-macc + 常驻 codebook 表在 e8m1/e16m2/e32m4 三态切换间的同时存活需求），不是"手写展开 vs 编译器展开"这个源码语法层面能绕开的。

**这与项目 canon [GAP-P1] "re-roll trap" 同构**（"直觉投影...回卷省 vsetvli...不可信"，已在别的格上两次证伪）——本役是该发现在 **nvfp4 格、rvv 板** 上的**第三次独立板测证伪**，per-format 板测纪律要求（不能直接外推），结果与 canon 完全一致。

---

## 3. 对手成色标注

- **对手 = generic 弱实现，非 absent，非 hand-tuned**：`native-vec-generic-weak`（mac=2/gather=0/vset=3，近标量+零星自动向量化，无 riscv 专化）。win 成色若翻正会很低（打折）；本役未翻正，此项仅供 §4 X 命名完整性参照。
- 上游 ggml 对 nvfp4 **确实**给 ARM 写了 NEON 手调（`ggml_nvfp4_dot8`），唯独 RISC-V 缺失 → 这是 **ggml 自身的架构覆盖空洞**，不是我方为了制造"赢弱对手"叙事而挑的靶子；即便如此我方仍未能赢过这个弱对手。
- **我方核本身是真 SIMD**（gather=8/mac=12，货真价实的向量化 codebook-gather 点积），对手几乎是标量——**"手强对标弱"却仍 LOSS**，说明差距不在"我方偷懒未向量化"，而在 §4 诊断的结构性开销。

---

## 4. 逐格结论 · ★ 决令二具名-X 认输

| 格@rvv | T3_A 行 | 冷启 M=1 | 对手真派发符号 | disposition |
|---|:--:|:--:|---|---|
| vec_dot/nvfp4 | `iq/fp4 vecdot` L34 | 0.565× (本役复测 0.569–0.572×，同量级) | `ggml_vec_dot_nvfp4_q8_0`(generic alias，tot214/rvv23/mac2/**NO-riscv-spec**) | **具名-X 认输** |

**决令二四件齐**：

1. **对手符号+反汇编**：`ggml_vec_dot_nvfp4_q8_0`（generic，源码 `ggml-cpu/quants.c:279`）；`objdump` tot=214/rvv=23/mac=2/vset=3/gather=0；确认 `arch/riscv/quants.c` 0 处 nvfp4 override，`arch/arm/quants.c` 有 NEON 手调——RISC-V 架构覆盖空洞，非选弱对手。
2. **差距构成逐项量化**：我方 baseline tot=371/rvv=68/mac=12/gather=8 vs 对手 tot=214/rvv=23/mac=2/gather=0；我方指令数 1.7× 于对手且真向量化，但其中 **~30%（44/371）是纯 spill 地址算术+落栈/取回**（11 vs1r.v + 11 vl1r.v + 22 csrr-vlenb-地址计算），源于 QK_NVFP4=64 super-block 仅 4×16-宽 sub-block、每 sub-block 需要 codebook(m1)+widen 乘积(m2)+reduce 模板(m2)+多段字节向量同时存活，跨 e8m1/e16m2/e32m4 三态切换触发寄存器分配器溢出。
3. **我方等价构造实际尝试**：`nvfp4_attack.kernel.c`（re-roll 4-way 展开 + hoist 冗余 scale 转换，byte-exact 安全性预先论证）。**G1**：3 形状(K=2048,M∈{1,8},nc∈{64,512}) ATTACK-vs-BASE 与 ATTACK-vs-真-OPP-oracle 均 0 mismatch，GREEN。**G2**：rvv 板 clang-18 对称域 cold A/B（core8 pin，co-tenant 未扰，224MiB flush，N=12，两轮独立测量共 6 点）：`attack_over_base = 1.0015–1.0112×`，噪声地板内，NULL。事后反汇编复核证实 spill 计数（11/11/22）构造前后**完全相同**——瓶颈是数据流固有寄存器压力，源码级 re-roll 语法改动被 `-O3` 后端展开抹平，不构成可绕开的杠杆。
4. **残余具名 X（可检验）= `[GAP-NVFP4-VLEN128-SUBBLOCK-GRANULARITY-SPILL]`**：QK_NVFP4=64 的窄 super-block 粒度（4×16-元素 sub-block，每 sub-block 仅够 vl=8 一次 op）在 VLEN128 上无法摊销 codebook-gather + widen-macc + 三态 SEW/LMUL 切换 的常驻寄存器需求，强制 spill，使"真-SIMD 但小块高开销"的我方核，天然跑不过对手"近标量但零向量开销"核——vector-pipeline 固定开销（vsetvli 重配+vrgather 查表延迟+spill 流量）在 K=2048(32 个 super-block 迭代) 规模下持续压过 SIMD 吞吐增益。**可证伪检验点**：任何后续再攻坚提案，若不能实质压低"22-instr vlenb-spill-地址算术 + 11/11 spill load/store"这组计数（即真正减少 e8m1/e16m2/e32m4 三态同时存活的向量寄存器组数——例如放弃 widening-macc 改窄位宽整数乘加、或重排数据流使 codebook 常驻不进入需要跨态切换的 live range），该 X 会持续复现；此检验点可直接复测验证/证伪。与 canon [GAP-P1]（"widen-to-m1/re-roll 不可信"）同构，是其在 nvfp4/rvv 上的第三次独立 per-format 板测证伪。

---

## 5. 板卫生

- 测前：`/proc/stat` 1s 采样确认 core8-15 全 0.0–1.0% busy；co-tenant（2 users，sshd/tpu-smi/containerd/dockerd/irqbalance 等其它核）未扰、未 restart。
- 测中：全程 `taskset -c 8`；loadavg 2.13→2.50（在既有会话漂移带内）。
- 测后：`rm -rf /tmp/g8s3-nvfp4-attack`（本役 scratch 全清）；`/tmp/g8s3`（sealed，含 opp .o 与 anal.sh/metrics.sh 复用脚本）与 `/tmp/opp_link_stubs.{c,o}`（既有共享 sealed 产物，非本役所写）**未删未改**；`ps aux | grep nvfp4_attack` 空，无残留进程。
- 无 git 操作（未 add/commit，遵守指令）；未改 T3_A / T8 / emitter / lib，纯只读测量 + 独立 casefile 产出。

---

## 6. 指针可核性

- casefile 文件：`nvfp4_attack.kernel.c`(构造核) · `nvfp4_attack_driver.c`(G1/G2 driver) · `raw/build_seal.txt`(md5 全链) · `raw/my_link_stubs.c` · `raw/nvfp4_attack_run.log`(两轮板测原始输出) · `raw/ours_baseline_full.dis.txt` / `raw/ours_attack_full.dis.txt` / `raw/opp_generic_full.dis.txt`(三份全量反汇编)。
- 对手 .o 复用 §五/§六 sealed `/tmp/g8s3/rvv/quants_generic_opp.o`（md5=8873178b，逐位核对与 build_seal.txt 记录一致），未重新拉取/未改动对手代码。
- 上游 ggml 源码指针：`/home/ubuntu/vericurve-rv-lab/llama.cpp` @ `e36a602ba38a26206c749ba4fb5dcf481bfd92db`（`ggml-cpu/quants.c:279`、`ggml-cpu/arch-fallback.h:18`、`ggml-cpu/arch/arm/quants.c:736`、`ggml-cpu/arch/riscv/quants.c`）。
