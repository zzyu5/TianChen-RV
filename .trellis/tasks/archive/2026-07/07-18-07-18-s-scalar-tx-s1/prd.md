# PRD · S线 · scalar 真 no-V 硅 T-X 六列 + S1 性能对局

> **权威** = 《开测篇》§三 主线 S（标量真实测试·T-X 收口·5 停止条件之一）。
> **性质** = 真板测（bench 唯一通道）+ harness 构建 + T-X 载体收口。走 scalar 板（板时不撞 rvv/k1）。

## 一、★前提更新（B4 案头 2026-07-15 中心前提已证伪）

B4 案头结论头条「手头**无**物理向量缺席真硅」——**已被证伪**：超锐(scalar) 板后上线，
**verified isa = `rv64imafdch_zicntr_zicsr_zifencei_zihpm_zaamo_zalrsc_zca_zcd`**（**无 `v`·无 `zve*`
= 真物理 no-V 硅**），clang-18 装讫。

**⟹ 两个直接后果：**
1. **T-X 列1 从 (b)窄豁免 升级为 (a)物理真硅** —— 去 `narrow-exempt` 标·措辞铁线
   「validated on silicon」**字面成立**（不再是"带 V 板 run-as-noV"的解释性辩护）。
2. **S1 性能对局在此板非便宜档**：[对手法 §板别提醒] —— `ggml_..._generic` 在无 V scalar 板上
   **满足部署事实要件 = 真部署对手**（在 rvv/k1 上 `_generic` 才是便宜档 artifact·此板无向量不对称
   ⟹ 干净对局）。**但仍 enablement 域**（[L-6] scalar 永不作贡献基线·列5 永不升 Win）。

## 二、已有资产（复用·勿重造）

- `experiments/active/g8-stage3-attack/A3-xscalar-rv64gc/`：`harness.cpp` · `weft_scalar_tq2_0_kernel.cpp`
  （owned weft_scalar 内核·emitter 产出）· `fp16util.cpp` · `TX-six-column-evidence.md`（六列证据稿）· `evidence/`。
- `experiments/active/g8-stage3-attack/A2-batch6-iqtq-gemm-scalar-raw/`：`iqtq_gemm_scalar_driver.cpp` + kernels + run_*.sh。
- 后端：`lib/Plugin/Scalar/{ScalarExtensionPlugin,ScalarEmitCRouteProvider,ScalarBackendEmissionDriver}.cpp`
  · `lib/Target/Scalar/ScalarTargetSupportBundle.cpp`。

**读 A3 `TX-six-column-evidence.md` 先** —— 六列已产多少、缺哪列，据此定增量（勿凭假设·PR-47 教训）。

## 三、三段收口

### §三.1 scalar cell harness 构建（bench 通道接线）
- 建 **`tools/bench/cells/` 下 scalar 格 harness**（复用 A3 harness.cpp + weft_scalar 内核 + A2 driver 逻辑）。
- **契约**（[ISSUE-090]）：harness **禁自写任何文件**·由 bench 按声明接口调用·一切持久写入经 runner
  fail-closed 写入闸落三目的地。
- bench 签名带全四元行键（op,format,engine,regime）·歧义 fail-closed。

### §三.2 S1 性能对局（承重数）
- **我方 weft_scalar 优化 kernel vs 该板 ggml 手写标量参考 `_generic`**·bench 通道·**cold 口径**。
- 板端 clang-18 单世界（双方对称·march=rv64gc no-V）·N=25 中位+relIQR+2-seed·禁继承。
- **ZERO-MODEL 正确门**（[K-5]）：ours vs ggml 权威 generic 逐 (row,col) byte-exact（对手独立于 ours 内部）。
- 结果回填 T-X 列5（逐竞品产出数·**enablement 域·显式非 Win**·[L-6]）。倍数可如实记，
  但**不进系统账·不升 perf-covered**（scalar-ref sanity 定性·同 A2 batch6 判读）。

### §三.3 T-X 六列全绿收口
| 列 | 门 | 本 task 动作 |
|---|---|---|
| 1 目标身份 | 真硅+capability profile·**(a)物理 no-V** | 去窄豁免标·填 verified isa 指纹 |
| 2 构建 | `-march=rv64gc -mabi=lp64d -ffp-contract=off`·weft_scalar owned 内核 | 板端 clang-18 构建·旗标入案 |
| 3 零向量指令机检 | objdump 静态·零 RVV opcode·无向量 intrinsic·**板无关** | 脚本化机检·PASS/FAIL |
| 4 byte-correct | 目标运行·byte-exact vs 独立 host oracle·证书三要件 | scalar 板真跑·[K-5] |
| 5 逐竞品产出数 | 对手类词表·**enablement 无 beat·scalar-oracle sanity** | S1 对局数回填·标非 Win |
| 6 参考路径披露 | deployed==proven·措辞铁线 | 见下 |

**§三.3 措辞铁线（钉死·禁改写）**：
「targeting rv64gc (vector-absent profile), validated on silicon」；
只写「codegen 竞品产出为零」·**禁写「竞品跑不了」**；披露 deployed variant == proven variant（证书三要件）。

## 四、机制 / 触碰集

- **触碰集** = `tools/bench/cells/`（scalar 新格 harness）+ scalar 板跑（runs/）+ T-X 载体
  （spec testing 契约 §T-X·若已存在则更新·否则本 task 落定义）+ recon-dict（若 S1 数进主表）+ issues。
- **并行纪律**：本 task 独占 scalar harness + scalar 板；**mech-① 线独占 rvv/k1 + GridCodebook.cpp**·不相交。
- **T-X 载体归属**：B4 稿说落 `.trellis/spec/testing/mlir-testing-contract.md` §T-X —— 但重构后 spec 六层
  无 testing 层。**T-X 定义该落哪层（measurement? evidence?）= [判据级·停]**：登记 ISSUE·保守默认
  = 暂将 T-X 证据表落 `experiments/`（数据格·活证据）+ 定义指针登记待裁·**不擅自新建 spec 层/文件**。

## 五、验收清单（交接 trellis-check）

1. **scalar harness 建成**·契约合规（harness 禁自写文件·经 bench 落三目的地）·bench 能调。
2. **S1 对局真数**：weft_scalar vs ggml _generic·cold·runs/ 有原档·runs.log 一行·ZERO-MODEL byte-exact GREEN。
3. **T-X 六列**：列1=(a)真硅（去窄豁免·verified isa）·列3 零向量 objdump PASS·列4 byte-correct·
   列5 标非 Win·列6 措辞铁线逐字合规。
4. **enablement 铁律**：scalar 永不作贡献基线·列5 不升 Win·倍数不进系统账/perf-covered（禁扩分母）。
5. **0 造数**（无真跑不入账）·sealed 不动（9/83）·**未擅建 spec 层**（T-X 归属待裁·保守默认落 experiments/）。

## 六、遗留

- T-X 定义层归属 → ISSUE 待裁（§四）。
- B4 稿 PR-1「采购物理 no-V 板」→ **已无需采购**（超锐即真硅）·PR-1 可标 RESOLVED-BY-FACT（登记·不擅改旧 PENDING 册）。
- S1 若某格 ggml 无 _generic scalar 参考（罕见）→ 如实标 N/A-hw·不硬凑对手。
