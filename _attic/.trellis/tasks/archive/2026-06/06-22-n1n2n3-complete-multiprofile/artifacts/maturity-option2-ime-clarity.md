# option-2 与 IME 白话讲清楚 (只读调查, 不改代码) — 2026-06-25

读了 option2 各 stage DESIGN/FINDING + `lib/Plugin/IME/` + `n2-ime/` finding。下面全用白话。

---

## A. option-2 是什么 (一句白话)

**option-2 = 让编译器自己看硬件能力, 替每个量化 matmul 选一个最快的算法 (repack 还是 block-dot),
并把选择记录在 IR 里。** 「编译器自己选 repack vs block-dot」的意思:

- 同一个 q4_0 权重乘法, 有两种算法实现:
  - **block-dot** = ggml 原生那种 (每个 block 算完做一次横向求和 `vredsum`), 通用但有「求和墙」。
  - **repack** = 先把权重重排成 16-路交错的 x16 布局, 然后用 lane-wise 累加 (没有 `vredsum` 墙)。
    在 VLEN128 上更快 (decode 实测 ~2.6–3.3×, 单线程 ~7×)。
- 以前是**人**事先决定用哪个 (手写那条具体 op)。option-2 让**编译器**根据能力事实
  (从 `-march` 推出 VLEN、量化类型、decode/prefill) 自动决定 —— 这正是用户拥有的 N3 novelty。

---

## B. A/B/C1/C1b/M1/M2 各做完了什么 (白话)

- **A (已做)**: 造了一个「还没决定算法」的抽象 op `quant_contraction`, 和一个把它原样翻译回今天 block-dot
  的 pass。零行为变化, 纯脚手架 —— 让「编译器来选」这件事在 IR 里可表达。
- **B (已做)**: 真正的**选择逻辑**。编译器读能力事实, 用一个 3-事实 AND 自动判 repack/block-dot,
  并把决定 (算法/理由/是否已兑现) 盖章写进 IR。**注意: 此时发出的 C 代码还完全一样 (都还是 block-dot),
  B 只证明「选得对」, 不是性能赢。**
- **C1 (已做, 纯 lit)**: 把「选了 repack」这一支真的接到**真 repack op**, 并声明它需要 x16 权重布局
  (输出契约 `weight_layout_contract="x16"`)。编译器侧干净, 没有「每张量限制」。
- **C1b (已做, host 字节精确)**: 编译器**自己生成**一个 plain→x16 的打包器, 跟 ggml 的 `make_block_q4_0x16`
  逐字节相同 (520 万 block memcmp==0)。证明「编译器有能力产出它声明的布局」。
- **M1 (已做, host)**: 证明**自动选**出来的 repack kernel 跟**手写**那条 repack op 发出的 kernel **逐字节相同** ——
  即自动选择没引入任何字节差异 (auto-SELECTION gap 关闭)。只是 emit-identity, 不含性能。
- **M2 (已做, ssh rvv)**: 把那条自动选出的 kernel 字节封印到真硬件 e2e, decode ~2.6–3.3× / 单线程 ~7×。
  **但整合是手工贴的 (手打补丁 + 手放 `.inc`)** —— 证明的是「kernel 的 e2e」, 不是「编译器自动接进 llama」。

---

## C. C3 (producer) 为什么没做、难在哪

**C3 = 那个把整条链自动跑起来的「生产者」工具**: 自动造抽象 op → 跑选择 pass → 读回契约 → 发 kernel →
交给构建。今天**根本没有这个东西**: 以前的 e2e 是「一个人手工跑一次 `tcrv-opt`, 再手写 `arch_repack.cpp`
把发出的 `.inc` 塞进 ggml」。

**「weight-packing build/load 耦合」是什么 / 为什么不能纯在编译器里做完:**
编译器只能**声明**「我需要 x16 布局」, 但**谁真的把权重字节摆成 x16, 发生在 ggml 模型加载时**
(`get_optimal_repack_type` + `make_block_q4_0x16`), 这是 ggml 的构建/加载机制, 不在我们 `lib/` 里。
而且**一个张量在任一时刻只能是一种物理布局** —— decode 想要一种、prefill 想要另一种时, 系统必须二选一:
双存 (2× 内存) / 每次调用现 repack (只对 prefill 划算) / 挑一个。这个「限制」编译器消不掉, 只能**搬到系统层**。
所以 C3/C4 必然要碰 ggml 构建 + gguf 加载 + mul_mat 分发的接线 —— 这是耦合点。

**是真做不了, 还是只是工作量大?** —— **是工作量 + 集成, 不是架构做不到。** M2 已经手工把整条链跑通并拿到真
e2e 数; C3 只是把那个人手工做的步骤脚本化。难点是 ggml 构建集成 (2 个 ggml 补丁) + 加载耦合, 不是 MLIR 侧。

---

## D. option-2 要「真自动选 + 自动接进 llama」还差哪几步 / 多大工作量

1. **C3 producer (~1–1.5 session, host 为主)**: 写 `tcrv_produce_contraction.py`, 自动造抽象 op、驱动选择 pass、
   读回契约、发 kernel + manifest。先用**字节精确再生成门** (对齐已 e2e 验证过的手建 `.inc`) 在 host 上去险, 无需硬件。
2. **C4 harness (~1 session, 需硬件)**: 读契约 → 让 ggml 的加载布局决定遵从它 (含那条必须的 1-行 ggml `case 128`
   补丁 + kernel 分发替换)。**这一层背锅那个「每张量布局二选一」的限制。**
3. **C5 双 profile e2e (~1 session, 需 ssh rvv + ssh k1)**: 每个数字注明机制 (stored-x16 / pick-one), kernel 与 e2e 分开报;
   prefill 还卡在 GEMM 数值缺陷上 (未达成 coherent)。
- **合计 ~2.5–3.5 session。** 真正的新代码风险集中在集成, kernel 侧已被 M1/M2 字节封印去险。

---

## E. IME: 为什么只生成 leaf? 能不能生成完整 GEMM? gap?

**先纠正前提 (用户的「只生成单条 leaf」不完全对):** 我们其实有两层。
- `tcrv.ime.mma{,_u,_su,_us,_slide}` = 单条 `vmadot*` **leaf** (4×4×8 单片段)。这些是**真硅片字节精确封印**过的。
- `tcrv.ime.matmul` (MatMulOp) = **已经会发一个完整 tiled GEMM kernel**:
  (M/4)×(N/4)×(K/8) tile 网格、K 维在寄存器里累加 (所以没有 RVV 的 `vredsum` 墙)、每 tile 存一次。
  所以「整片矩阵」我们**是**会生成的, 不止 leaf。

**`gemm_kernel_i8i4` 是谁写的:** **SpacemiT 厂商**写的 (`spacemit_kernels::ime1::gemm_kernel_i8i4`),
在他们的 `libggml-cpu` 里 (`GGML_CPU_RISCV64_SPACEMIT=ON` 才编进去), 不是我们的代码。

**我们能不能用 leaf 自动组完整 IME GEMM (tiling/累加/边界)? —— 架构上能, 部分已做, 差在 3 个具体点:**
1. **dtype 差距 (最关键, 这是真能力 gap, 不只是集成):** 厂商的是 int8 × **int4** (i8i4, 把 Q4_0 的 4-bit 权重
   边解包边乘的融合路径)。我们所有 IME op 都硬钉 `elem_in_bits==8` —— 是 int8 × **int8**。**i4 解包融合我们还没有。**
2. **边界 (remainder):** 我们的 matmul 验证器**只收 M/N/K 是片段整数倍** (fail-closed, 无尾巴路径) —— 诚实但受限。
3. **运行时形状 + producer:** M/N/K 现在是**编译期常量**烤进 wrapper 的, 不是运行时参数; 且和 option-2 一样
   **没有 producer**, 没接进 ggml 的 `mul_mat` 分发。

**诚实: 「成熟 compiler」该长什么样, 离多远:** 成熟版 = 从高层 matmul 自动生成完整 IME GEMM (含 i4-融合 +
尾巴 + 运行时形状 + 自动接 ggml)。我们离它**结构上不远** (tiling+累加已经能组合、单片段硅片封印), 但有 **3 个实打实的洞**:
i4 解包融合 (能力洞)、remainder 路径、运行时形状 + producer 接线。**而且即便补完, IME 的 e2e 大概率仍是 NULL**:
kernel 微基准 5.66× (re-baseline; 旧的 5.51× 已作废, 量错了基线) **不传导**到 memory-bound 的 decode e2e
(实测 0.86–0.98× ≈ 1×); 之前看到的 prefill「赢」是 clang 重建 codegen 假象 (随 M 衰减, 与 IME 矩阵单元无关)。
IME 的 novelty 是**结构** (零-core-branch 的能力事实分发), 不是 e2e 性能。

**统一洞见 (两个故事同一个形状):** option-2 和 IME 的缺口**完全同形** —— 编译器侧干净 (会选/会发对的 kernel),
但**都没有 producer**, 整合都是手工把 kernel 塞进 ggml 的加载+分发。区别只在: option-2 的 e2e 赢是真的 (但手工集成);
IME 的 e2e 是 NULL (内存墙)。
