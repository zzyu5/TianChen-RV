# 12 个 RVV Slice 作业

每条 slice 对应一项 TianChen-RV 当前尚未支持的能力：实现它，编译器就能多生成一类 RVV kernel。这不是重做已有功能。先读 worked example [`../docs/add-rvv-slice-walkthrough.md`](../docs/add-rvv-slice-walkthrough.md)（已实现的 `add`，作为所有 slice 的模板）；落点见 [`../docs/rvv-slice-module-map.md`](../docs/rvv-slice-module-map.md)，验收见 [`../docs/testcase-submission-format.md`](../docs/testcase-submission-format.md)。

统一交付：typed body + verifier（含负例）→ realization owner → provider EmitC route + emitter → 三层验证（lit FileCheck / 本地 rv64gcv object 编译 / qemu 数值 `proof ok`）+ PR 说明。每条独立，一项能力一位同学。

> **行号约定**：下文每条给的 `文件:行号` 锚点是**发布时**的位置，代码演进后会漂移。**符号名（op 名、函数名、owner 名）才是稳定锚**——行号对不上时，按给出的符号名搜索，别照行号硬找。

## 一览
| # | Slice | 轴 | 新能力一句话 |
|---|---|---|---|
| 1 | rope_neox | 前向核 | NeoX 旋转位编码（split-half 对），覆盖 Qwen/Falcon/Phi |
| 2 | gelu / gelu_quick | 前向核 | FFN 主力激活（现仅 silu） |
| 3 | soft_max_ext | 前向核 | 融合注意力 softmax（scale+mask+归一） |
| 4 | slide 家族 | 位移原语 | vslideup/down，conv1d/RoPE-在寄存器/scan |
| 5 | quantize_row_q8_K | 量化输入 | f32→q8_K，为 K-quant / IQ dot 提供激活量化输入 |
| 6 | segment3 load/store | 内存形态 | 3-field 交织/反交织（现仅 segment2） |
| 7 | float standalone reduce | 归约 | StandaloneReduceOp 扩到 f32（fsum/fmax/fmin） |
| 8 | vcompress + viota | 流压缩 | mask 紧致打包 + 前缀计数，argsort/top-k/MoE 基石 |
| 9 | f16↔f32 convert | 位宽转换 | f32→f16 narrowing + 往返，f16 KV-cache/权重 |
| 10 | integer narrowing convert | 位宽转换 | 饱和 vnclip，i32 累加器 requantize→i8 |
| 11 | get_rows | 选行 | token 嵌入查找 / MoE 选行（f32/f16） |
| 12 | layernorm (ggml_norm) | 前向核 | mean/variance 归一（现仅 rms_norm） |

> 注：#7、#9、#10 是在已有机制上新增当前缺失的方向 / 种类（f32 归约 / f32→f16 narrowing / 饱和 narrowing），同样是当前编译器尚不能生成的能力；详见各页。

---

## 1. rope_neox

**目标 / 新能力**
让编译器新增能力:为 ggml NEOX 变体的 rotary position embedding(`ggml_compute_forward_rope_f32` 的 `is_neox` 分支,GPT-NeoX/Falcon/Qwen2/Phi 等架构用)发射 byte-exact 的 RVV C。今天只有 NORMAL 变体。NEOX 与 NORMAL 的角度缓存(`theta *= theta_scale` 迭代递推 + 标量 libm `cosf`/`sinf`)完全相同,**唯一差别是配对模式**:NORMAL 旋转相邻对 `(x[2p], x[2p+1])`,NEOX 旋转分半对 `(x[p], x[p+n_dims/2])`。学生新增一个 typed op `tcrv_rvv.ggml_rope_neox_f32`,把它沿兄弟的 body-shape 路径发射成一个标量 per-pair 循环。

**当前实现现状**
- `grep -rni "neox" lib/ include/ test/` → 0 命中(全仓库无 NEOX)。
- 唯一 rope op 是 `GgmlRopeNormF32Op`(`include/TianChenRV/Dialect/RVV/IR/RVVOps.td:8432`)。
- verifier fail-closed 拒绝任何其他变体:`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:8403` —— `if (getKind() != "ggml_rope_norm_f32") return emitOpError() << "currently supports only kind \"ggml_rope_norm_f32\""`。
- body-detector `RVVToEmitC.cpp:1625 isGgmlRopeNormF32Body` 只 `isa<GgmlRopeNormF32Op>`;dispatch `RVVToEmitC.cpp:524` 只对 rope_norm 分支;emitter 只有 `RVVToEmitCForwardElementwise.cpp:899 emitGgmlRopeNormF32`。NEOX 无 op / 无检测 / 无发射 —— 编译器今天发不出来。

**改哪些文件**(逐一对照兄弟,逐字镜像)
1. **ODS op** `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:8432-8522` → 新增 `def GgmlRopeNeoxF32Op`。args/results/assemblyFormat/`hasVerifier=1` 与 `GgmlRopeNormF32Op` 完全相同;只改 mnemonic `"ggml_rope_neox_f32"`、summary/description(把配对说明改成 split-half `x[i], x[i+n_dims/2]`,引 ggml `is_neox`)。
2. **verifier** `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:8378-8470` → 新增 `GgmlRopeNeoxF32Op::verify()`,逐行抄,仅把 8403 的 kind 串改成 `"ggml_rope_neox_f32"`;6-operand / `const float *` x / `float *` y / `float` theta_base / `float` theta_scale / index element_count / `f32 m1` result / enclosing with_vl policy 约束全部保留不动。
3. **dispatch + detector + 自包含 header(RVVToEmitC.cpp 三个触点,缺一不可)**
   - `:234` 的 `<math.h>` 自包含析取里加 `|| isGgmlRopeNeoxF32Body(scope)`(NEOX 也调 `cosf`/`sinf`,漏掉则发射 TU 不 self-include math.h、tier-2 object-compile 失败)。
   - `:524` 新增 dispatch 分支:`if (isGgmlRopeNeoxF32Body(scope)) { emitGgmlRopeNeoxF32(...); create<ReturnOp>(); eraseOp(variant); return success(); }`。
   - `:1625` 新增 body-detector `isGgmlRopeNeoxF32Body`(抄 `isGgmlRopeNormF32Body`,只把 `isa` 的 op 类型换掉)。
4. **声明** `lib/Conversion/RVV/RVVToEmitCInternal.h:355`(detector 声明)与 `:2733`(`emitGgmlRopeNeoxF32` 声明)各加一条,签名照抄 rope_norm 同名声明。
5. **emitter** `lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp:899-1089` → 新增 `emitGgmlRopeNeoxF32`,大块逐字抄,**只改索引/指针运算**(见下"对照范例"推荐写法)。角度递推(`thetaVar` lvalue + assign)、`cosf`/`sinf` `emitOpaqueCall`、两个 `emitc.expression`(各包住一个 `a*b±c*d`)的 FMA-grouping 纪律 **必须原样保留**(byte-exact 的关键所在)。
6. **测试** `test/Conversion/RVV/rvv-to-emitc-ggml-rope-neox-f32.mlir` → 抄 `rvv-to-emitc-ggml-rope-norm-f32.mlir`,fixture body 把 op 换成 `tcrv_rvv.ggml_rope_neox_f32`(kind `"ggml_rope_neox_f32"`),CHECK 把入口名改成 `tcrv_emitc_ggml_rope_neox_f32_kernel_ggml_rope_neox_f32`,并把"相邻对" CHECK 改成分半寻址(见验收)。
7. **不要改**:realization owner(`lib/Plugin/RVV/RVV*SelectedBodyRealizationOwner.cpp` + `RVVSelectedBodyRealization.cpp`)、route family plan owners、`RVVEmitCRouteFamilyDerivation.cpp` 的 enum。前向 ggml 族(rms_norm/rope/quantize_row_q8_0/soft_max)走的是 RVVToEmitC.cpp 的 **body-shape dispatch**,**不**经过 realization-owner / route-family 机制(已 grep 证实 rope_norm 只出现在 ODS / verifier / dispatch / emitter / Internal.h 五个文件)。沿这个范式即可，无需寻找其他 hook。

**对照范例**
- worked example = elementwise `add`(typed body → 选择体 → EmitC route → emitter → 生成 C → harness vs scalar 参考实现 的完整骨架)。
- 最近兄弟 = `GgmlRopeNormF32Op` / `emitGgmlRopeNormF32`。**推荐 emitter 写法(保持与兄弟最小 diff)**:循环外算一次 `half = n_dims / 2`(`emitc.div`,同时作为循环上界,与兄弟的 `nPairs` 同值);循环内不再用 `pairOff = 2*p`,改为两个基址各做一次 `[0]` 下标——`xLo = (const float*)(x + p)`、`xHi = (const float*)(x + (p+half))`、`yLo = (float*)(y + p)`、`yHi = (float*)(y + (p+half))`,`x0 = xLo[0]`、`x1 = xHi[0]`,`yLo[0] = x0*cos - x1*sin`、`yHi[0] = x0*sin + x1*cos`。这样每个 `emitc.subscript` 仍是字面 `0`(完全照兄弟),避开"运行时下标进 emitc.subscript"的额外坑。

**scalar 参考实现**
参考计算 = ggml `ggml_compute_forward_rope_f32` 的 `is_neox` 分支(`ggml_rope_cache_init` 角度缓存 + NEOX 配对;plain path:ext_factor=0、freq_scale=1、attn_factor=1、freq_factors=NULL、forward ⇒ sin_sign=+1、n_dims=ne0)。这段 oracle **已逐字存在**于 rope_norm 验收 harness 的 negative-control(`.../artifacts/inc20-forward-pass-f6/inc20_validate.cpp:111 nc_neox_rope_ref`),学生本质是把它从反例提升成正路:
```c
const int64_t half = n_dims / 2;
float theta = theta_base;                 // 与 NORMAL 同一迭代递推
for (int64_t p = 0; p < half; ++p) {      // p == ic == i0/2
  const float c = cosf(theta), s = sinf(theta);   // 标量 libm,同一 libm 才 exact
  const float x0 = x[p];
  const float x1 = x[p + half];           // ← split-half(与 NORMAL 唯一差别)
  y[p]        = x0*c - x1*s;               // 单 C 表达式 → 与 ggml 同 FMA contraction
  y[p + half] = x0*s + x1*c;
  theta *= theta_scale;                    // theta_scale = powf(freq_base, -2/n_dims)
}
```
byte-exact 两条轴与兄弟相同:(1) `cosf`/`sinf` 必须经 `emitc.call_opaque`(链同一 libm);(2) 每个 `a*b±c*d` 包进一个 `emitc.expression`,渲染成单条 C 语句 → clang 在所有 `-ffp-contract` 模式下做相同 contraction。

**验收(三层)**
- **Tier 1 lit FileCheck**(本地,人人):`tcrv-opt rvv-to-emitc-ggml-rope-neox-f32.mlir --tcrv-rvv-lower-to-emitc | FileCheck`。必须命中:`emitc.include <"math.h">`;`emitc.func @tcrv_emitc_ggml_rope_neox_f32_kernel_ggml_rope_neox_f32(`;`%[[THETA:.*]] = "emitc.variable"{{.*}}lvalue<!emitc.opaque<"float">>`;`div` 算 half;`for`;`call_opaque "cosf"` / `call_opaque "sinf"`;**分半寻址**——基址用 `p` 与 `p+half`(`add %[[P]], %{{.*}}`,**不再有** NORMAL 的 `mul %[[P]], 2`);两个 `expression`(lo = `mul;mul;sub`,hi = `add`)各 `yield`;`assign` 回 `%[[THETA]]`。`CHECK-NOT: tcrv_rvv.` / `unrealized_conversion_cast` / `emitc.verbatim {{.*}}__riscv`。
- **Tier 2 object-compile**(本地):`tcrv-opt ... | mlir-translate --mlir-to-cpp > gen.c`;`/usr/lib/llvm-20/bin/clang -target riscv64 -march=rv64gcv -c gen.c`(证明是合法 RVV C,且 math.h 自包含)。
- **Tier 3 数值(qemu)**:将生成的 C 与 harness 编成 riscv64 可执行,用 `qemu-riscv64` 运行;harness 用上面的 NEOX scalar 参考实现,对 HEAD_DIM=128、positions={0,1,2,7,31,128,511,2047,4095,8191,32767} 多分布逐 bit 比较;打印 `proof ok`。
- **PR 说明须含**:typed facts 位置(SEW/LMUL/policy 在 setvl/with_vl,n_dims/AVL/VL 在控制面;op 只带 `kind` 镜像属性);route-from-facts(走 body-shape dispatch,非 realization-owner/route-family);关键 intrinsic / 接缝(`cosf`/`sinf` 经 `emitc.call_opaque`;旋转经 `emitc.expression`);确认无禁用反模式(无 dtype 前缀 helper、无 source-front-door 正路、无 route-id/metadata 驱动语义、无 common-EmitC 硬编码 dtype/SEW/LMUL)。

**目标 IR 形状(草图)**
> 目标形状:在你新增该 op 之前不可解析;精确语法照搬 add 范例 + 兄弟 fixture(`rvv-to-emitc-ggml-rope-norm-f32.mlir`)。不声称它能 parse。
```mlir
module {
  tcrv.exec.kernel @ggml_rope_neox_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_rope_neox_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      // 声明顺序 == 生成 C 入口参数顺序(n, x, y, tb, ts),与兄弟逐字一致
      %n  = tcrv_rvv.runtime_abi_value {c_name = "n_dims", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x  = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y  = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %tb = tcrv_rvv.runtime_abi_value {c_name = "theta_base", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %ts = tcrv_rvv.runtime_abi_value {c_name = "theta_scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_rope_neox_f32, sew = 32 : i64, source_kernel = "ggml_rope_neox_f32_kernel", status = "selected-lowering-boundary"} {
        %rot = tcrv_rvv.ggml_rope_neox_f32 %x, %y, %tb, %ts, %n, %vl {kind = "ggml_rope_neox_f32"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
```

---

## 2. gelu / gelu_quick

**目标 / 新能力** — 让编译器新增发射两个 FFN 激活算子的能力：GELU（tanh 近似）与 GELU-quick。两者目前完全不存在；本扩展让 `tcrv_rvv` 能把一条 typed body（输入 `x`、输出 `y`、元素数 `n`）下降为 byte-exact 的向量化 RVV C：
- `gelu(x)   = 0.5*x*(1 + tanh(SQRT_2_OVER_PI * x * (1 + GELU_COEF_A*x*x)))`，常量 `SQRT_2_OVER_PI=0.79788456080286535587989211986876f`、`GELU_COEF_A=0.044715f`（即 ggml `ggml_gelu_f32` 的 tanh 近似式）。
- `gelu_quick(x) = x / (1 + exp(GELU_QUICK_COEF * x))`，`GELU_QUICK_COEF=-1.702f`（即 `ggml_gelu_quick_f32 = x*sigmoid(1.702*x)`）。

关键复用：两者都用现成的向量化 exp 多项式砖 `emitGgmlVExpfM2`（`RVVToEmitCForwardElementwise.cpp:326`）。`gelu_quick` 几乎就是 silu 把 `vfneg(x)` 换成 `vfmul_vf(x,-1.702f)`；`gelu(tanh)` 用 `tanh(z)=1-2/(exp(2z)+1)` 把同一砖包成 tanh。

**当前实现现状** — 全树 `grep -rni gelu`（排除 `build/`）= **0 命中**；`grep GELU_COEF_A|gelu_quick|ggml_gelu|SQRT_2_OVER_PI|GELU_QUICK_COEF` = **0**。唯一存在的 f32 激活超越函数是 silu（`RVVOps.td:8209` 定义、`RVVToEmitC.cpp:466` 分发）。当前没有任何 GELU 既有实现。

**改哪些文件**（镜像 silu；forward-pass 家族**不走** Plugin realization-owner / route-family，故那两类文件**无需改动**——`grep GgmlVec* lib/Plugin/`=0 已证）：
1. **ODS op** — `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`：照抄 `GgmlVecSiluF32Op`（8209–8272）新增 `GgmlVecGeluF32Op` 与 `GgmlVecGeluQuickF32Op`。同样 `(ins AnyType:$input,$output,$element_count,$vl, StrAttr:$kind)`、`TCRVEmitCLowerableOpInterface`、`hasVerifier=1`。
2. **verifier** — `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp`：照抄 `GgmlVecSiluF32Op::verify()`（8102–8178），把 `kind` 白名单改为 `"ggml_vec_gelu_f32"` / `"ggml_vec_gelu_quick_f32"`，保留 const float*/float*/index/!vl 检查与 `isForbiddenDataflowParameterAttr` 关门（I7）。
3. **body 检测 + 分发** — `lib/Conversion/RVV/RVVToEmitC.cpp`：照抄 `isGgmlVecSiluF32Body`（1583–1594）与 466–473 的分发分支；声明加在 `RVVToEmitCInternal.h:329`。
4. **emitter** — `lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp`：照抄 `emitGgmlVecSiluF32`（459–561），复用 `emitGgmlVExpfM2`；声明加在 `RVVToEmitCInternal.h`（silu 声明约 2592–2618 旁）。常量用精确 C float token（hex-float，照 silu 的 `fimm` 风格）。
5. **tests** — 新建 `test/Conversion/RVV/rvv-to-emitc-ggml-vec-gelu-f32.mlir` 与 `...-gelu-quick-f32.mlir`（镜像 `rvv-to-emitc-ggml-vec-silu-f32.mlir`）；数值 harness 参照 silu 的同类验证 harness。（silu 无独立 Dialect 验证器 fixture，可不强求；如要加 expected-error 测试，放 `test/Dialect/RVV/gelu-dataflow.mlir`。）

**对照范例** — worked example = elementwise **add**（`test/Target/RVV/pre-realized-selected-body-artifact-add.mlir`，借它学 `runtime_abi_value` typed-facts 与 harness 约定）。最近兄弟 = **silu**（`GgmlVecSiluF32Op` + `emitGgmlVecSiluF32`，op 形状/strip loop/exp 砖完全同构）。`gelu_quick` 基本是 silu 的微改。

**scalar 参考实现** — 参考语义取 ggml 的 tanh 近似式（`ggml_gelu_f32` / `ggml_gelu_quick_f32`），**刻意不复刻 ggml 的 f16-LUT**（避免有损查表）。因为本核是向量化的、向量 exp 多项式无法与 scalar libm 逐位相等（这正是 silu 的负控），byte-exact 的 oracle 必须是「本核同一套向量数学的逐节点誊写」：
```c
// 逐节点誊写 emitGgmlVExpfM2 + tanh(z)=1-2/(exp(2z)+1)，参照 inc17_validate.cpp 的 ggml_ref
vfloat32m2_t ge = ggml_v_expf_m2_ref(vfmul_vf(inner, 2.0f), vl);          // gelu(tanh)
// tanh = 1 - 2/(ge+1);  y = 0.5*x*(1+tanh)
vfloat32m2_t e  = ggml_v_expf_m2_ref(vfmul_vf(x, -1.702f), vl);           // gelu_quick
// y = x / (1 + e)
```
**负控（判别性）**：朴素 scalar-libm tanh 近似 GELU——与本核逐位 DIFFER（证明匹配了具体多项式方法，非仅到容差）；同时与本核到 ~1 ULP 一致（证明是正确函数）。memcmp 比 float bits，覆盖 silu harness 同款分布（~0、moderate、saturation tails、edge inf/nan/denormal、llama FFN 尺寸 11008/11007/奇尾）。

**验收(三层)**
1. **lit FileCheck**（本地）：新 Conversion golden 测试，照 silu fixture 钉死结构——`emitc.func @tcrv_emitc_<kernel>_<variant>(`、`__riscv_vsetvl_e32m2`、m2 strip `for`、`vle32`，exp 砖的全部 literal/call_opaque（`0x1.8p23f`/`0x1.715476p+0f`/…/`vmerge`），尾部 `vfdiv`(quick) 或 `0.5f`/`vfmul`(tanh)，`vse32`；`CHECK-NOT: tcrv_rvv.` 与 `CHECK-NOT: emitc.verbatim {{.*}}__riscv`（I5 全结构化）。
2. **object-compile**（本地）：`tcrv-opt %s --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp > gen.cpp`；`/usr/lib/llvm-20/bin/clang -target riscv64 -march=rv64gcv -c gen.cpp`（证合法 RVV C）。
3. **数值（qemu）**：将生成的 C 与 harness 编成 riscv64 可执行，用 `qemu-riscv64` 运行；byte-exact vs 参考实现 + 负控判别 → 打印 `RESULT: PASS` / `proof ok`。
**PR 说明须含**：typed facts 在哪（`runtime_abi_value` 的 c_name/c_type/role + `kind` attr + `with_vl` 的 policy）；route-from-facts（分发键是 **op identity**，非 route-id/metadata 字符串）；关键 intrinsics（`vle32`/`vfmul_vf`/`vfmacc`/`emitGgmlVExpfM2` 链/`vfdiv`/`vse32`）；无禁用反模式（无 dtype 前缀 helper、无 source-front-door、无 route-id/metadata 驱动语义、无 common-EmitC 硬编码 dtype/SEW/LMUL）。

**目标 IR 形状(草图)** — 目标形状：在你新增该 op 之前不可解析；精确语法照搬 add 范例 + silu 兄弟 fixture。
```mlir
tcrv.exec.kernel @ggml_vec_gelu_quick_f32_kernel {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
  tcrv.exec.variant @ggml_vec_gelu_quick_f32 attributes {origin = "rvv-plugin", requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
    %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
    %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", sew = 32 : i64, policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        selected_variant = @ggml_vec_gelu_quick_f32, status = "selected-lowering-boundary", required_capabilities = [@rvv] /*...照 silu fixture 补齐...*/ } {
      %g = tcrv_rvv.ggml_vec_gelu_quick_f32 %x, %y, %n, %vl {kind = "ggml_vec_gelu_quick_f32"}
            : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
    } : !tcrv_rvv.vl
  }
}
```
（`gelu(tanh)` 版同形，仅 op 名 / `kind` 改为 `ggml_vec_gelu_f32`。）

---

## 3. soft_max_ext

**目标 / 新能力**
让编译器能发射 ggml 真实的注意力 softmax 整行算子 `ggml_soft_max_ext`（`ggml_compute_forward_soft_max_f32`，ops.cpp ~5390-5416），而不仅是它的内核。一行的计算融合为一个 typed op：`wp[i] = x[i]*scale + mask[i]` → 行内求 `max` → `dp[i] = e^{wp[i]-max}` 并累加 `sum` → 原地归一化 `dp[i] *= 1/sum`。当前编译器只能发射"裸内核"（max 当输入、不 scale/不 mask/不归一化），无法表达一条可直接喂给 attention 的完整 softmax 行。MVP 落地后，学生真正把 scale + 加性 mask + 行内 max + 归一化这四步带进了 emitter。

**MVP 范围（先锁定，避免越界 + 躲字节坑）**
- 仅 **f32 mask**（`const float *`）；f16 mask（逐 lane `GGML_FP16_TO_FP32`）是 stretch。
- `max_bias = 0 ⇒ slope = 1`，ALiBi 斜率是 stretch。**关键**：slope=1 时 `slope*mp[i]+wp[i]` 退化成纯 `vfadd_vv`，自动躲过 fma-vs-(mul-then-add) 的字节坑。一旦做 slope≠1，**必须** `vfmul_vf` 后 `vfadd_vv`（两次舍入，对齐 ggml 标量 `wp[i]+=slope*mp[i]`），**禁止** `vfmacc_vf`（一次舍入 → 不同 bit）。
- mask 必填（无-mask 路径是 stretch）；max 行内自算；函数返回 **void**（原地归一化）。

**当前实现现状**
- `grep -rni "soft_max_ext|softmax_ext|SoftMaxExt|GgmlSoftMaxExt" include/ lib/ test/` → exit 1（仅 archive 命中）。
- 全树唯一 softmax op = `GgmlVecSoftMaxF32Op`（RVVOps.td:8274-8343），其 ODS 明写：`max` 是 INPUT、只写 `y[i]=e^{x[i]-max}`、**不归一化**（caller 用 `ggml_vec_scale_f32` 做 1/sum）。即裸内核只是 ext 的一个内层 rung，scale/mask/行内 max/归一化全无表示。

**改哪些文件**（本族 **emitter-direct**：**不新增** realization owner / route family；dispatch 走 `isGgml...Body` 形状识别，与 bare soft_max/silu/quantize 同）
- **ODS**：`include/.../RVVOps.td` 新增 `def GgmlSoftMaxExtF32Op`（照搬 8274 块）。`arguments`: `output, input, mask, scale, element_count, vl, kind`（比裸 op 多 `mask`、`scale`，去掉 `max`）；`results`: 一个 vector token（函数 void，按 quantize 8345 风格）。`kind` 限定 `"ggml_soft_max_ext_f32"`。
- **verifier**：`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` 新增 `GgmlSoftMaxExtF32Op::verify()`，镜像 8180 起的裸 op verifier，补 `mask` 为 `const float *`、`scale` 为 `float`。
- **emitter 本体**：`lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp` 新增 `emitGgmlSoftMaxExtF32`（返回 `LogicalResult`，照 quantize:697 的 void 签名）：复用 `emitGgmlVExpfM2`（指数）、裸 op 的 f64 widening-reduce（563-695 段）、`emitGgmlVecScaleF32`(:2534) 的 vfmul_vf（scale 与归一化两处）、quantize:810 的 vfredmax（注意是整行多-strip：`vfmax_vv` 累加 + 末尾 `vfredmax_vs` 以 `-INFINITY` 种子，对齐 `ggml_vec_max_f32`）。
- **emitter dispatch**：`lib/Conversion/RVV/RVVToEmitC.cpp` 新增谓词 `isGgmlSoftMaxExtF32Body`（镜像 1597-1608）；在 504-511（quantize void 分支）后加一个 `if (isGgmlSoftMaxExtF32Body(scope))` void 分支（emit 后 `emitc::ReturnOp(loc, Value())`）。**返回类型决策 253 行不动**（ext 是 void，裸 op 才 push `double`）。
- **internal 头**：`lib/Conversion/RVV/RVVToEmitCInternal.h` 声明 `isGgmlSoftMaxExtF32Body`（仿 :335-340）与 `emitGgmlSoftMaxExtF32`（仿 :2648/:2692）。
- **测试**：新增 `test/Conversion/RVV/rvv-to-emitc-soft-max-ext-f32.mlir`（镜像裸 op golden）。verifier 覆盖随 emitter golden（与裸 op 同，本族无独立 dataflow 文件）；若想加 `test/Dialect/RVV/...-dataflow.mlir`，按 `quant-contraction-dataflow.mlir` 风格。HW harness 仿 inc18_validate.cpp + soft_max_kernel.cpp。

**对照范例**
- 总体约定（typed body 语法、ABI、三层验证）：elementwise **`add`**（`test/Conversion/RVV/rvv-to-emitc-add-cpp-golden.mlir`、`test/Target/RVV/explicit-selected-body-artifact-add.mlir`）。
- 最近兄弟：**裸 `ggml_vec_soft_max_f32`**（exp+sum+f64 reduce 内核，直接复用）；**`ggml_vec_scale_f32`**（scale/归一化）；**`quantize_row_q8_0`**（void 返回 + vfredmax）。

**scalar 参考实现**
注意：本仓"oracle"约定是 **ggml 自己整行 reference 的逐字复刻（含其向量化 exp 多项式 `ggml_v_expf_m2`）**，不是 libm。harness 内逐行复刻 `ggml_compute_forward_soft_max_f32`（slope=1、f32 mask）：
1. `wp[i] = x[i]*scale`（`ggml_vec_scale_f32` RVV 路 = vfmul_vf）；
2. `wp[i] += mask[i]`（slope=1 ⇒ 纯 add）；
3. `max = ggml_vec_max_f32(nc, wp)`（其 RVV vfmax/vfredmax 折叠，种子 `-INFINITY`）；
4. `sum = ggml_vec_soft_max_f32(nc, dp, wp, max)`（= 已字节精确的裸内核：exp 多项式 + f64 widening reduce）；
5. `sum = 1.0/sum`（**double** 倒数），`ggml_vec_scale_f32(nc, dp, (float)sum)`。**字节坑**：`1.0/sum` 是 double，传入取 `float` 的 scale → double→float 窄化必须一致。
逐 bit `memcmp` 比较 `dp[]`（float 位）。覆盖：注意力维度、large/small/uniform/spiky/all-equal，以及 `-inf` mask 项（走指数下溢 →0）。
**负控（仿裸 op nc_libm_exp/nc_f32_sum）**：(a) 用 libm `expf` 代多项式（证明匹配的是 ggml 精确多项式）；(b) 归一化用 f32 倒数而非 double-倒数-后-窄化（证明末步 cast crux）。

**验收（三层）**
1. **lit FileCheck**（人人，本地）：`tcrv-opt rvv-to-emitc-soft-max-ext-f32.mlir --tcrv-rvv-lower-to-emitc | FileCheck`：`CHECK-NOT: tcrv_rvv.`；entry `tcrv_emitc_soft_max_ext_f32_kernel_soft_max_ext_f32(size_t, float*, const float*, const float*, float)` 返回 void；出现 vfmul_vf(scale)、vfadd_vv(mask)、vfmax_vv+vfredmax(-INFINITY)、`emitGgmlVExpfM2` 链（`0x1.8p23f`/`0x1.715476p+0f` 等）、`vfwredusum_vs_f32m2_f64m1`、末步 vfmul_vf(1/sum)；全部 structured emitc 节点，`CHECK-NOT: emitc.verbatim {{.*}}__riscv`。
2. **object-compile**（本地证 RVV C 合法）：`mlir-translate --mlir-to-cpp` 后 `/usr/lib/llvm-20/bin/clang -target riscv64 -march=rv64gcv -c gen.cpp`。
3. **数值（qemu）**：将生成的 C 与 harness 编成 riscv64 可执行，用 `qemu-riscv64` 运行；vs 上述 ggml 复刻参考逐 bit `memcmp` 且两个负控失败；打印 `proof ok`。
4. **PR 说明**须含：typed facts 位置（SEW/LMUL/policy 在 op 属性，dtype 在 ABI `c_type`）；route-from-facts（dispatch 由 body 形状 `isGgmlSoftMaxExtF32Body` 决定，非 route-id/metadata）；关键 intrinsics 列表；**无禁用反模式**（dtype 前缀 helper / source-front-door 正路 / route-id 驱动语义 / common-EmitC 硬编码 dtype/SEW/LMUL）。

**目标 IR 形状（草图）**
> 目标形状：在你新增该 op 之前不可解析；精确语法照搬 add 范例 + 兄弟 fixture（runtime_abi_value / setvl / with_vl 头逐字抄裸 soft_max fixture，仅加 mask/scale 并去掉 max）。
```mlir
%n     = tcrv_rvv.runtime_abi_value {c_name="n", c_type="size_t", role="runtime-element-count", ...} : index
%y     = tcrv_rvv.runtime_abi_value {c_name="y", c_type="float *", role="output-buffer", ...} : !tcrv_rvv.runtime_abi_value
%x     = tcrv_rvv.runtime_abi_value {c_name="x", c_type="const float *", role="lhs-input-buffer", ...} : !tcrv_rvv.runtime_abi_value
%mask  = tcrv_rvv.runtime_abi_value {c_name="mask", c_type="const float *", role="rhs-input-buffer", ...} : !tcrv_rvv.runtime_abi_value
%scale = tcrv_rvv.runtime_abi_value {c_name="scale", c_type="float", role="dequant-scale-value", ...} : !tcrv_rvv.runtime_abi_value
%vl = tcrv_rvv.setvl %n {lmul="m1", sew=32, policy=...} : index -> !tcrv_rvv.vl
tcrv_rvv.with_vl %vl attributes {selected_variant=@soft_max_ext_f32, ...} {
  %r = tcrv_rvv.ggml_soft_max_ext_f32 %y, %x, %mask, %scale, %n, %vl
       {kind="ggml_soft_max_ext_f32"}
     : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value,
       !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl
       -> !tcrv_rvv.vector<f32, "m1">
} : !tcrv_rvv.vl
```
（emitter strip 循环对齐 ggml 固定 m2，与裸 op 同；typed body 的 `lmul="m1"` 头照抄即可。）

---

## 4. slide family (vslideup/vslidedown/vslide1up/vslide1down)

**目标 / 新能力**
让编译器第一次能把"定偏移的寄存器内跨-lane 滑动 + 边沿标量移入"表达为一个 typed body op `tcrv_rvv.slide`，并由能力事实驱动 lower 成 `vslidedown / vslide1down`（MVP）以及 `vslideup / vslide1up`（扩展）。这是 conv1d 滑窗、RoPE 相邻对旋转、prefix-scan 的基础原语。今天的 dialect 只能做 lane-local 的 binary/move，无法表达"out[i] = in[i+k]"这类跨-lane 重排。

**当前实现现状**
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` 全量 op 列表无任何 `*Slide*Op`（只有 Binary/Splat/Move/Load/Store/…）。
- 无 realization owner：`lib/Plugin/RVV/` 无 `*Slide*SelectedBodyRealizationOwner.cpp`；注册表 `RVVSelectedBodyRealization.cpp:34` 不含 slide。
- 无 route family：`lib/Plugin/RVV/EmitC/` 无 `*Slide*RouteFamilyPlanOwners.cpp`；枚举 `RVVEmitCRouteProvider.h:25 (enum RVVSelectedBodyOperationKind)` 无 Slide 项；`RVVEmitCRouteFamilyDerivation.cpp` switch 无 slide case。
- 无 emitter dispatch：`RVVToEmitC.cpp` 的 op-walk else-if 链（~line 785/823）无 slide 分支；`RVVToEmitCSupport.cpp` 的 `manglerForRVVIntrinsic` grep `slide` = 空。
- 仅有的"slide"出现：`RVVToEmitCKQuant.cpp:324/1322/3655-3669` 把 `__riscv_vslidedown_vx_i32m4(v116, 8, 8)` 硬编码当 K-quant 的 VLEN-agnostic 归约折叠 helper（非 typed-body 驱动、不可复用）；`lib/Plugin/IME/IMEBackendEmissionDriver.cpp` 的 `tcrv.ime.mma_slide` 是无关的 IME 矩阵 A-窗口 stride。两者都不是 first-class lane-slide 能力。→ 好处：vslidedown 指令拼写已在 K1 上 byte-exact 验证过，学生抄已知 callee 即可。

**改哪些文件**
1. ODS op：`RVVOps.td` 新增 `def SlideOp : TCRVRVV_Op<"slide", [TCRVEmitCLowerableOpInterface]>`。镜像 `BinaryOp`(3217) 的 `kind` StrAttr + `MoveOp`(8634) 的 vector→vector 形态 + `SplatOp`(2860) 的 scalar 操作数。建议 arguments：`AnyType:$source, AnyType:$vl, StrAttr:$kind, I64Attr:$offset`，slide1up/down 再加 `AnyType:$shift_in`（来自 `rhs-scalar-value` 角色的 runtime_abi_value），vslideup 扩展再加 `AnyType:$passthrough`。kind ∈ {"down","slide1down"}(MVP) / {"up","slide1up"}(扩展)。
2. 验证器：在 `lib/Dialect/RVV/IR/RVVDialectArithmeticOps.cpp` 紧挨 `BinaryOp::verify`(38) 新增 `SlideOp::verify`（slide 是寄存器域 compute，非 memory）。镜像 BinaryOp 校验：拒 forbidden 属性、限定 kind 集合、校验 source/result 同 typed vector 类型、vl 为 `!tcrv_rvv.vl`、`verifyNestedDataflowOp` + `verifyDataflowVLOperandMatchesWithVL`。
3. realization owner：新建 `lib/Plugin/RVV/RVVSlideSelectedBodyRealizationOwner.cpp(+.h)`，复制 `RVVElementwiseSelectedBodyRealizationOwner.cpp` 改成认 `tcrv_rvv.slide` + kind→`RVVSelectedBodyOperationKind::SlideDown/SlideOneDown`；在 `RVVSelectedBodyRealization.cpp:34` 的 owners[] 注册。
4. route family：在 `RVVEmitCRouteProvider.h:25` 枚举加 `SlideDown, SlideOneDown(,SlideUp,SlideOneUp)`；新建 `lib/Plugin/RVV/EmitC/RVVEmitCSlideRouteFamilyPlanOwners.cpp` 复制 Elementwise 版；在 `RVVEmitCRouteFamilyDerivation.cpp`(~836 的 switch) 加 case 路由到新 plan owner。
5. emitter：新建 `lib/Conversion/RVV/RVVToEmitCSlide.cpp`（或在 `RVVToEmitCForwardElementwise.cpp` 内加 `emitSlide`），在 `RVVToEmitC.cpp` op-walk(~785/823) 加 `else if (auto slide = dyn_cast<tcrvrvv::SlideOp>(op)) emitSlide(...)`；在 `RVVToEmitCSupport.cpp` 的 `manglerForRVVIntrinsic` 加 slide 助记符，拼 `__riscv_vslide{down,1down}_vx_<dtype><lmul>`（沿用已有 `<dtype><lmul>` 规则，如 i32m1）。
6. 测试：`test/Dialect/RVV/slide-dataflow.mlir`（typed-body/verifier，镜像 `dataflow.mlir`）、`test/Conversion/RVV/rvv-to-emitc-slide.mlir`（emitter golden，镜像 `rvv-to-emitc-add.mlir`）、`test/Target/RVV/explicit-selected-body-artifact-slide.mlir`（镜像 `explicit-selected-body-artifact-add.mlir`）。

**对照范例**
端到端模板 = `add`（`rvv-to-emitc-add.mlir` / `explicit-selected-body-artifact-add.mlir`，已在 K1 byte-exact）。最近兄弟：op 形态抄 `MoveOp`（vector+vl+kind→vector，emitMove@RVVToEmitC.cpp:823，verify@RVVDialectStoreOps.cpp:38），标量移入抄 `SplatOp`（scalar runtime_abi_value+vl→vector，emitSplat@RVVToEmitC.cpp:785，verify@RVVDialectMemoryOps.cpp:900）。

**scalar 参考实现**
对单个 VL strip（建议 n ≤ VLMAX，或每 strip 独立、且 strip 宽与 kernel 一致），逐元素定义：
- `kind="down", offset=k`（tail agnostic → 约定填 0）：`out[i] = (i+k < vl) ? in[i+k] : 0`。
- `kind="slide1down"`（shift_in = s）：`out[i] = in[i+1] (i<vl-1); out[vl-1] = s`。
- 扩展 `kind="up", offset=k`：`out[i] = (i<k) ? passthrough[i] : in[i-k]`；`kind="slide1up"`：`out[0]=s; out[i]=in[i-1]`。
（语义 = RVV spec 的 vslide* 定义；K-quant 已证 `__riscv_vslidedown_vx`(src,offset,vl) 三参形态。harness 用纯 C 标量循环算 oracle，逐字节比对。）

**验收（三层）**
1. lit FileCheck：`rvv-to-emitc-slide.mlir` 断言生成 C 结构——strip-mine for 循环、`__riscv_vslidedown_vx_i32m1` / `__riscv_vslide1down_vx_i32m1` typed `call_opaque`、`CHECK-NOT: tcrv_rvv.` / `unrealized_conversion_cast`；artifact 测断言 `void tcrv_emitc_<kernel>_<variant>(...)` ABI 签名按 runtime_abi_value 声明序。
2. object-compile：`/usr/lib/llvm-20/bin/clang -target riscv64 -march=rv64gcv -c gen.c` 通过（证 RVV C 合法）。
3. 数值（qemu）：将生成的 C 与 harness 编成 riscv64 可执行，用 `qemu-riscv64` 运行，对 scalar 参考实现 byte-exact，打印 `proof ok`。
PR 必说明：typed facts 位置（`kind`/`offset`/`shift_in` 在 op 上，是 body authority）；route 由 op identity+kind 派生（非 route-id 字符串）；关键 intrinsic（`__riscv_vslide{down,1down}_vx_<dtype><lmul>`）；无 forbidden 反模式——不得用 dtype-prefixed helper（别造 `i32_slide`）、不得走 source-front-door、不得用 route-id/metadata 决定语义、不得在 common-EmitC 硬编码 dtype/SEW/LMUL。

**目标 IR 形状（草图）**
> 目标形状：在你新增该 op 之前不可解析；精确语法照搬 add 范例 + 兄弟 fixture（MoveOp 的 kind 语法、SplatOp 的 scalar 操作数语法）。下面不保证能 parse。
```mlir
tcrv.exec.variant @slide_down_rvv_i32 attributes {origin = "rvv-plugin", requires = [@rvv]} {
  %in  = tcrv_rvv.runtime_abi_value {c_name="in",  c_type="const int32_t *", role="lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
  %out = tcrv_rvv.runtime_abi_value {c_name="out", c_type="int32_t *",       role="output-buffer"}    : !tcrv_rvv.runtime_abi_value
  %n   = tcrv_rvv.runtime_abi_value {c_name="n",   c_type="size_t",          role="runtime-element-count"} : index
  %vl  = tcrv_rvv.setvl %n {lmul="m1", sew=32:i64, policy=#tcrv_rvv.policy<tail=agnostic, mask=agnostic>} : index -> !tcrv_rvv.vl
  tcrv_rvv.with_vl %vl attributes {lmul="m1", sew=32:i64, selected_variant=@slide_down_rvv_i32, status="selected-lowering-boundary", /*...*/} {
    %v = tcrv_rvv.load %in, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32,"m1">
    // 新 op：定偏移向下滑（tail agnostic）。slide1down 时改 kind 并加 %s 标量移入操作数。
    %s = tcrv_rvv.slide %v, %vl {kind="down", offset=2:i64} : !tcrv_rvv.vector<i32,"m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32,"m1">
    tcrv_rvv.store %out, %s, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32,"m1">, !tcrv_rvv.vl
  } : !tcrv_rvv.vl
}
```

---

## 5. quantize_row_q8_K

**目标 / 新能力**
让编译器新增一个前向量化器 op `tcrv_rvv.quantize_row_q8_K`：把一行 f32 激活按 ggml `quantize_row_q8_K` 语义量化成 AoS 的 `block_q8_K`（QK_K=256 的超块：`fp32 d` @0 | 256 个 `int8 qs` @4 | 16 个 `int16 bsums` @260，stride=292）。这是喂给整套 K-quant / IQ block-dot 动物园（q4_K×q8_K、q6_K×q8_K、repack GEMV/GEMM …）的**激活产生端**——目前这些 op 只能 *消费* 一个由 ggml 在运行时塞进来的 block_q8_K，编译器自己造不出来。做完后 f32→q8_K 这一桥由我们发射的 RVV C 承担。

**当前实现现状**
- `grep -rn "QuantizeRowQ8K|quantize_row_q8_K|ggml_quantize_row_q8_K" include/ lib/ test/` → 0 命中。dialect 里唯一的量化器**产生端**是 `GgmlQuantizeRowQ80Op`（`RVVOps.td:8345`），没有 q8_K（也没有 q8_1）。
- 现有树里所有 `block_q8_K` 都是**消费端**：`GgmlRepackGemvQ4KQ8KOp`（`RVVOps.td:4868`，描述里 `block_q8_K stream at byte stride 292`、`bsums at +260`、`qs at +4`，见 4926/4937/4942）、`GgmlBlockDotQ6KQ8KOp`（`RVVOps.td:6177`）、`RVVQ4KBlockDotSourceFrontDoor.cpp:113`。它们都把 q8_K 当成已存在的输入。
- 即编译器能算 q4_K·q8_K 的点积，却不能**生产** q8_K——这正是一块缺失的前向砖。

**改哪些文件**（仿 q8_0 兄弟；本前向量化器**家族不进** selected-body realization owner、也**不进** EmitC route family —— 它在 `RVVToEmitC.cpp` 里按 **op-identity** 直接派发，q8_0 兄弟即证据）
1. **ODS op** `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`：新增 `GgmlQuantizeRowQ8KOp`（`tcrv_rvv.quantize_row_q8_K`），整体照搬 `GgmlQuantizeRowQ80Op`（8345-8430）。参数同样是 `input/output/element_count/vl` + `StrAttr:$kind` + 块格式 mirror attrs；attrs 改为 `qk=256, block_stride=292, scale_byte_offset=0, quant_byte_offset=4`，并**新增** `bsums_byte_offset=260`、`bsums_group_size=16`（命名与消费端 `activation_bsums_byte_offset` @`RVVOps.td:4983` 保持同族）。`assemblyFormat`/`results`/`hasVerifier` 照抄。
2. **verifier** `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp`：新增 `GgmlQuantizeRowQ8KOp::verify()`，照 `GgmlQuantizeRowQ80Op::verify()`（8268-8357+）改：`isForbiddenDataflowParameterAttr` fail-closed、白名单 attr 集合、`kind=="ggml_quantize_row_q8_K"`、`qk==256/block_stride==292/scale_byte_offset==0/quant_byte_offset==4/bsums_byte_offset==260/bsums_group_size==16`、ABI 绑定校验（`input` 为 `const float *`，`output` 为 `uint8_t *`，`element_count` 为 index 且 `n % 256 == 0`）、结果为 generic RVV f32 m1 vector。
3. **emitter 派发** `lib/Conversion/RVV/RVVToEmitC.cpp`：仿 504-511 加 `if (isGgmlQuantizeRowQ8KBody(scope)) { emitGgmlQuantizeRowQ8K(...); ... }`；仿 1611-1623 加 `isGgmlQuantizeRowQ8KBody`（body 恰好一个该 op）。
4. **emit 实现** `lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp`：新增 `emitGgmlQuantizeRowQ8K`，以 `emitGgmlQuantizeRowQ80`（697-897）为骨架；signed-max/标量前奏部分参考 `emitGgmlRopeNormF32`（899）的 emitc.variable 循环携带标量技法。**关键差异不要照抄**：d 是 `*(float*)(yb+0)=d`（普通 fp32 store，**不要** 8852-870 的 `_Float16` cast）；增加 bsums 第二遍归约写到 `yb+260`。
5. **decls** `lib/Conversion/RVV/RVVToEmitCInternal.h`：仿 342-348（isBody）与 2653-2692（emit fn）加两条声明。
6. **测试** `test/Conversion/RVV/rvv-to-emitc-ggml-quantize-row-q8-K.mlir`：照搬 `rvv-to-emitc-ggml-quantize-row-q8-0.mlir`，改 CHECK：fp32 d store（`assign` 到 `!emitc.ptr<!emitc.opaque<"float">>` 下标，**无** `_Float16` cast）、bsums 的 `vwredsum`/store、`CHECK-NOT: emitc.verbatim {{.*}}__riscv`。
   *（不新增 realization owner / route family / Target artifact —— q8_0 兄弟同样没有；这是前向家族约定。）*

**对照范例**
- 端到端流程范例：elementwise `add`（已在真硬件验证的参考）。
- 最近兄弟（逐层 copy-then-adapt）：`GgmlQuantizeRowQ80Op`（q8_0 量化器，四层路径见上）。
- 第二兄弟（标量前奏技法）：`GgmlRopeNormF32Op`（`RVVOps.td:8432` / emit `…ForwardElementwise.cpp:899`）。

**scalar 参考实现**
参照 ggml `quantize_row_q8_K`（ggml-quants.c）。每 256 元超块 i：
1. 扫描 256 个 lane，求带符号峰值：`if (fabsf(x[j]) > amax) { amax = fabsf(x[j]); max = x[j]; }`（**第一个** abs-max 位置的**带符号**值）。
2. 若 `amax==0`：`d=0`、`qs` 全 0、`continue`（ggml 只 memset qs，不动 bsums；零块下 qs 和=0，calloc 缓冲区里 bsums 本就 0，故一致）。
3. 否则 `iscale = -127.f / max;`（**负**、且符号随 max 翻——决定全部 qs 的符号）。
4. 每 j：`v = nearest_int(iscale * x[j]); qs[j] = MIN(127, v);`（`nearest_int` 是 1.5·2^23 magic-number 的 **round-to-nearest-even**；只上钳 127，下界天然 ≥ -127）。
5. 每 16 个一组：`bsums[g] = Σ_{ii=0..15} qs[g*16+ii]`（int16）。
6. `d = 1.f / iscale;`（fp32，存 byte 0）。
> 三个 q8_K 比 q8_0 多出来的 byte-exact 雷区，务必逐条对齐：**(雷1) signed-max 选择**——`vfredmax(vfabs)` 丢符号与首次出现，需复现 ggml 的带符号/首位选择；**(雷2) `MIN(127,v)` 仅上钳**——对称边缘 `nearest_int` 可能给 +128，若漏钳则 vncvt 截断成 -128 是真 bug；**(雷3) bsums int16 分组归约**——整数加可结合，向量 `vwredsum`(i8→i16) 与标量同 byte。

**验收(三层)**
- **Tier1 lit**：`tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck`。生成 `emitc.func @tcrv_emitc_quantize_row_q8_K_kernel_quantize_row_q8_K(...)`；`CHECK-NOT: tcrv_rvv.`；块循环 `nb=n/256`；amax 归约 intrinsics；fp32 d 直接 `assign`（无 `_Float16`）；scale + `vfncvt`(rne) + `vncvt` + `vse8` 写 `qs`@+4；bsums 归约 + store @+260；`CHECK-NOT: emitc.verbatim {{.*}}__riscv`（零 raw 字符串，I5）。
- **Tier2 object-compile**：`/usr/lib/llvm-20/bin/clang -target riscv64 -march=rv64gcv -c gen.c`（证 RVV C 合法可编）。
- **Tier3 数值（qemu）**：将生成的 C 与 harness 编成 riscv64 可执行，用 `qemu-riscv64` 运行；与 scalar 参考实现 **byte-exact**（逐 byte 比 292×nb 的 block_q8_K 缓冲：d/qs/bsums 全等），打印 `proof ok`。
- **PR 说明须含**：typed facts 落点（块格式在 op mirror attrs / I4，SEW/LMUL/policy 留 setvl/with_vl，runtime n 在外层控制面）；route-from-facts（按 **op-identity** 派发，非 route-id/metadata 语义）；关键 intrinsics 清单（`vle32`/`vfabs`/`vfredmax`/`vfmul_vf`/`vfncvt_x_f_w`(rne)/`vncvt`/`vwredsum`/`vse8`）；并逐项排除 forbidden anti-patterns（无 dtype-prefixed helper、无 source-front-door 正路、无 route-id/metadata 驱动语义、common-EmitC 不硬编码 dtype/SEW/LMUL）；**附 ggml 源码确认**：检查 riscv/quants.c 是否对 q8_K 有特化路径——若无，参考实现 = ggml-quants.c 的 generic `quantize_row_q8_K`（标量 ref 即实际计算路径），在 PR 里写明引用的 file:line。

**目标 IR 形状(草图)**
> 目标形状：在你新增该 op 之前不可解析；精确语法照搬 add 范例 + q8_0 兄弟 fixture（`rvv-to-emitc-ggml-quantize-row-q8-0.mlir`）。下面只示意结构，不声称能 parse。
```mlir
tcrv.exec.kernel @quantize_row_q8_K_kernel {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
  tcrv.exec.variant @quantize_row_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], ...} {
    %n  = tcrv_rvv.runtime_abi_value {c_name = "n",  c_type = "size_t",        role = "runtime-element-count"} : index
    %x  = tcrv_rvv.runtime_abi_value {c_name = "x",  c_type = "const float *", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "uint8_t *",     role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %n {lmul = "m1", sew = 32 : i64, policy = ...} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {selected_variant = @quantize_row_q8_K, sew = 32 : i64, ...} {
      %q = tcrv_rvv.quantize_row_q8_K %x, %vy, %n, %vl
             {kind = "ggml_quantize_row_q8_K", qk = 256 : i64, block_stride = 292 : i64,
              scale_byte_offset = 0 : i64, quant_byte_offset = 4 : i64,
              bsums_byte_offset = 260 : i64, bsums_group_size = 16 : i64}
           : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl
             -> !tcrv_rvv.vector<f32, "m1">
    } : !tcrv_rvv.vl
  }
}
```

---

## 6. segment3 deinterleave/interleave (load+store)

### 目标 / 新能力

为 RVV 增加 **3-field 分段内存搬运**（segment3），与现有 segment2 family 完全同构，只是 field-count 加一：

- `vlseg3e<sew>` —— 把一段 **AoS-of-3** 交织缓冲（interleaved RGB / 3-channel / xyz）**反交织 (deinterleave)** 成 3 个 unit-stride 向量；
- `vsseg3e<sew>` —— 把 3 个向量**交织 (interleave)** 回一段 AoS-of-3 缓冲。

slice 流向不变：typed `tcrv_rvv` body (.mlir) → selected-body realization → provider EmitC route → `RVVToEmitC` emitter → 生成 RVV C → harness 对 scalar oracle byte-exact。

### 当前实现现状

仓库今天**只有 segment2，没有 segment3**。可复现：

```
$ grep -rniE 'segment3|seg3|vlseg3|vsseg3' lib/ include/ test/ ; echo $?
1            # exit 1 = 零命中
```

对照 segment2 当前 **存在且密集**（同一 grep 把 `2→3`）：`grep -rlniE 'segment2' lib/ include/` 命中 ~30 个文件，含 ODS 6 个 op、verifier、realization owner、EmitC route family、emitter、6 个 intrinsic mangler。能力轴上 field-count=3 是真实新增的硬件指令（`__riscv_vlseg3e32_v_i32m1x3` 此前从不生成）。

### 改哪些文件（segment2 镜像；own-owner + own-route-family ⇒ clean isolated slice）

| 层 | 文件 | segment2 锚点 → 镜像出 segment3 |
|---|---|---|
| ODS（**6 个新 op**） | `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` | `Segment2LoadOp` :3063 / `MaskedSegment2LoadOp` :3098 / `Segment2StoreOp` :3141 / `MaskedSegment2StoreOp` :3177；pre-realized body `TypedSegment2DeinterleaveMemoryPreRealizedBodyOp` :2560 / `TypedSegment2InterleaveMemoryPreRealizedBodyOp` :2602 |
| verifier | `lib/Dialect/RVV/IR/RVVDialectMemoryOps.cpp`（load）+ `RVVDialectStoreOps.cpp`（store）+ `RVVDialectPreRealizedMemoryOps.cpp`（pre-realized `::verify()`） | 把 `segment_count==2` 校验改 `==3`，并校验新增的 `field2_role` |
| realization owner（**自有 owner**） | `lib/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.cpp`（含 `createRealizedGenericSegment2Store` :107 等 builder）→ 新建 `RVVSegment3MemorySelectedBodyRealizationOwner.cpp`；注册于 `RVVSelectedBodyRealization.cpp` :63-64（`{"segment2 memory", …}` 旁加 `"segment3 memory"`） | 反交织 body：`segment3_load → 3×move → 3×store`；交织 body：`3×load → segment3_store` |
| EmitC route family（**自有 family**） | `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp` → 新建 Segment3 版；op-kind enum `RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore` / `Segment2InterleaveUnitLoad`（`RVVEmitCRoutePlanningInternal.h` :970-971），并补 binding switch（`RVVEmitCRouteConfigBinding.cpp` :2445/4213/4501/4811）；route-family 派生 `RVVEmitCRouteFamilyDerivation.cpp` :6026+ | 新增 `Segment3DeinterleaveUnitStore` / `Segment3InterleaveUnitLoad` 两个 enum |
| emitter | `lib/Conversion/RVV/RVVToEmitC.cpp`：dispatch :828-846、`emitSegment2Load` :3822 / `Store` :3873 / masked :3945/4031、`emitSegment2InterleavedPointer` :3812（stride literal `"2"` :3815 → `"3"`） | 加 `emitSegment3*`，第三个 field 走 tuple index 2 |
| intrinsic mangler | `lib/Conversion/RVV/RVVToEmitCSupport.cpp` :575-667 六个函数 | 子串替换：`vlseg2e→vlseg3e`、`vsseg2e→vsseg3e`、tuple type `…x2_t→…x3_t`、tuple-create / field-extract `x2→x3` |
| 注册尾巴 | `lib/Plugin/RVV/CMakeLists.txt`、`lib/Plugin/RVV/EmitC/CMakeLists.txt` | 加新 .cpp |

注：**own-owner + own-route-family** ⇒ segment3 是独立隔离的 slice，不与 segment2 共享语义路径，回归面小。

### 对照范例

- **segment2 family（最近镜像）**：dataflow 形状 `test/Dialect/RVV/segment2-deinterleave-memory-dataflow.mlir` + `segment2-interleave-memory-dataflow.mlir`；emitter golden `test/Conversion/RVV/rvv-to-emitc-segment2-deinterleave-unit-store.mlir` + `…-interleave-unit-load.mlir`；artifact `test/Target/RVV/*segment2-deinterleave-unit-store*.mlir`、`*segment2-interleave-unit-load*.mlir`。
- **add worked example**：`test/Conversion/RVV/rvv-to-emitc-add.mlir`（入口名 `tcrv_emitc_<kernel>_<variant>`、ABI 参数=body 里 `runtime_abi_value` 顺序、`extern "C"`、`__riscv_vsetvl_e32m1` + for-loop step vlmax 的 strip-mine 骨架）。

### scalar 参考实现

signed `int32_t`、整数 byte-exact（与 segment2 bounded grid 同 dtype）。harness 内 oracle 循环：

```c
// deinterleave: src 是 AoS-of-3，out_a/out_b/out_c 是 3 路 SoA
for (size_t i = 0; i < n; ++i) {
    out_a[i] = src[3*i + 0];
    out_b[i] = src[3*i + 1];
    out_c[i] = src[3*i + 2];
}
// interleave 是其逆：dst[3*i+0]=in_a[i]; dst[3*i+1]=in_b[i]; dst[3*i+2]=in_c[i];
```

逐元素 `memcmp` 生成 kernel 输出 vs oracle，全等才算过。

### 验收（三层）

1. **lit / FileCheck**：新 `test/Dialect/RVV/segment3-*-memory-dataflow.mlir`（解析+verifier，含 `segment_count=3`、重复 field-role、route_id authority-attr 等 negative，照搬 segment2 的 `// -----` 段）；`test/Conversion/RVV/rvv-to-emitc-segment3-*.mlir` CHECK 生成 C 出现 `__riscv_vlseg3e32_v_i32m1x3` / `__riscv_vsseg3e32_v_i32m1x3` 与 `vint32m1x3_t` 元组类型、指针偏移 `* 3`，且 `CHECK-NOT: tcrv_rvv.` / `unrealized_conversion_cast`。
2. **本地 object-compile**：`mlir-translate --mlir-to-cpp` 出 C 后 `clang -target riscv64 -march=rv64gcv -c gen.c`，零 error/warning。
3. **数值（qemu）checklist**：将 gen.cpp 与 harness.cpp 编成 riscv64 可执行 → 用 `qemu-riscv64` 运行 → byte-exact 后打印 `proof ok`。**PR 说明**必须含：现状 grep 证据、6 个新 op + 自有 owner/route-family 的隔离性、三层证据（lit pass 名 + 本地 object 编译 log + qemu `proof ok` 输出）。

禁止反模式：dtype-前缀 helper、source-front-door 正路径、route-id/metadata 驱动语义、common-EmitC 硬编码 dtype。

### 目标 IR 形状（草图）

> **目标形状：加 op 前不可解析**；精确语法照搬 segment2 fixture（`segment2-deinterleave-memory-dataflow.mlir` / `segment2-interleave-memory-dataflow.mlir`）+ add 范例。field-count 处处加一：`segment_count = 3`、新增 `field2_role`、deinterleave 多一路 `out2`/result、interleave 多一路 `src2`/operand。

```mlir
// 反交织 (vlseg3e32)：1 交织源 → 3 SoA 输出
tcrv_rvv.with_vl %vl {lmul = "m1", sew = 32 : i64} {
  %f0, %f1, %f2 = tcrv_rvv.segment3_load %src, %vl {
      segment_count = 3 : i64,
      source_memory_form = "segment3-interleaved-unit-stride-load",
      field0_role = "segment-field0-output-buffer",
      field1_role = "segment-field1-output-buffer",
      field2_role = "segment-field2-output-buffer"
    } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl
        -> !tcrv_rvv.vector<i32,"m1">, !tcrv_rvv.vector<i32,"m1">, !tcrv_rvv.vector<i32,"m1">
  tcrv_rvv.store %out0, %f0, %vl : ...
  tcrv_rvv.store %out1, %f1, %vl : ...
  tcrv_rvv.store %out2, %f2, %vl : ...
} : !tcrv_rvv.vl

// 交织 (vsseg3e32)：3 SoA 输入 → 1 交织输出
tcrv_rvv.with_vl %vl {lmul = "m1", sew = 32 : i64} {
  %g0 = tcrv_rvv.load %src0, %vl : ...
  %g1 = tcrv_rvv.load %src1, %vl : ...
  %g2 = tcrv_rvv.load %src2, %vl : ...
  tcrv_rvv.segment3_store %dst, %g0, %g1, %g2, %vl {
      segment_count = 3 : i64,
      destination_memory_form = "segment3-interleaved-unit-stride-store",
      field0_role = "segment-field0-input-buffer",
      field1_role = "segment-field1-input-buffer",
      field2_role = "segment-field2-input-buffer"
    } : !tcrv_rvv.runtime_abi_value,
        !tcrv_rvv.vector<i32,"m1">, !tcrv_rvv.vector<i32,"m1">, !tcrv_rvv.vector<i32,"m1">, !tcrv_rvv.vl
} : !tcrv_rvv.vl
```

---

## 7. float standalone reduce (fsum/fmax/fmin)

**目标 / 新能力**
让编译器能对一段 unit-stride 的 `f32` 缓冲做"通用、可复用"的横向标量归约:`fsum`(顺序浮点求和)、`fmax`、`fmin`,经由已存在的 `tcrv_rvv.standalone_reduce` 通用 op 走完整 selected-body → EmitC → 生成 RVV C 的链路,产出 `out[0] = reduce(acc_seed, x[0..n))`。今天该 op 只能做整数(i32/i64)归约;`f32` 在每一层都被显式拒绝。这是把一个已验证的整数能力**扩展**到浮点(不是新增 op)。

**当前实现现状**
`StandaloneReduceOp` 已存在(`include/TianChenRV/Dialect/RVV/IR/RVVOps.td:3394`),其 kind/accumulator_layout/result_layout 均为自由 `StrAttr`,**无需改 ODS schema**。缺的是 float 语义支撑,5 处锁死整数:
- `isSupportedGenericStandaloneReduceKind` 只收 `add/min/max + signed/unsigned_widening_reduce_add`(`lib/Dialect/RVV/IR/RVVDialect.cpp:1798-1802`)。
- `isSupportedGenericStandaloneReduceAccumulatorLayout` 只收 `scalar-i32-seed-lane0-from-accumulator-input`(同文件 `:1808-1811`)。
- 验证器把 seed C-type 硬编码为 `const int32_t *`/`const uint32_t *`(`lib/Dialect/RVV/IR/RVVDialectReductionOps.cpp:386-391`)。
- emitter 助记符表 `standaloneReductionMnemonic` 只有 `vredsum/vredmin/vredmax/vwredsum*`,无任何 `vfred*`(`lib/Conversion/RVV/RVVToEmitC.cpp:2014-2027`)。
- seed splat 有一行**显式 float 闸**(就是要删的占位):`lib/Conversion/RVV/RVVToEmitC.cpp:2055-2056`
  `if (isFloatVector(resultVecType)) return nullptr; // standalone reduction is the integer-accumulator slice.`

340 个测试中 `standalone_reduce` 全是 i32/i64;唯一的 float 横向归约(`vfredmax`/`vfredusum`)只在专用 kernel 内手发(`RVVToEmitC.cpp:495`、`RVVToEmitCKQuant.cpp:851`),不走通用 op。

**改哪些文件**(每处照抄整数兄弟分支,新增并列 float 分支)
1. ODS(仅描述,不改 arguments):`RVVOps.td:3394` `StandaloneReduceOp` 的 `description` 补一句支持 f32 fsum/fmax/fmin。
2. 验证器:
   - `RVVDialect.cpp:1798` 加 `fsum/fmax/fmin`;`RVVDialect.cpp:1808` 加新 layout `scalar-f32-seed-lane0-from-accumulator-input`。
   - `RVVDialectReductionOps.cpp:346-391` `StandaloneReduceOp::verify()`:加一条 float 分支——seed C-type 期望 `const float *`、input 期望 `!tcrv_rvv.vector<f32,"m1">`、enclosing `with_vl` 仍 SEW32 LMUL m1/m2(`isSupportedTypedStandaloneReductionPreRealizedConfig` 对 SEW32 已通过,f32 自然满足)。
3. realization owner(pre-realized 路,供 Target/RVV tier):`lib/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.cpp`
   - `isPreRealizedStandaloneReduceOpKind:44` 加 `standalone_reduce_fsum/fmax/fmin`;`getPreRealizedStandaloneReduceDataflowKind:50` 映射到 `fsum/fmax/fmin`。
   - **关键陷阱**`getPreRealizedStandaloneReduceAccumulatorLayoutForSEW:122` 只按 SEW int 选 layout,i32 与 f32 同为 SEW32 → 必须新增元素类型/kind 维度区分,否则 f32 误命中 i32 layout。
4. route family:`include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h:37-40` 加枚举 `StandaloneReduceFSum/FMax/FMin`;在 `lib/Plugin/RVV/EmitC/RVVEmitCRouteFamilyDerivation.cpp:869/1262/1427` 与 `RVVEmitCRoutePlanning.cpp:651` 各 case 处镜像 `StandaloneReduceAdd`(`isReduction=true`)。
5. emitter:`RVVToEmitC.cpp:2014` 加 `fsum→"vfredosum"`(**有序!**)、`fmax→"vfredmax"`、`fmin→"vfredmin"`;`RVVToEmitC.cpp:2055-2056` 把 float 闸改成:float 时用 `vfmv_v_f`(mangler 已支持,`RVVToEmitCSupport.cpp:65-67`)替代 `vmv_v_x`,标量 load+subscript 路 dtype 无关无需改。`riscvWideningReductionIntrinsicName` 从 `vectorDType="f32"` 自动拼 `__riscv_vfredosum_vs_f32m1_f32m1`,不改。input load 走通用 `emitLoad`,已支持 f32。
6. 测试:新增 `test/Dialect/RVV/float-standalone-reduction-dataflow.mlir`(镜像 `standalone-reduction-dataflow.mlir` 的正/负例)、`test/Conversion/RVV/rvv-to-emitc-standalone-reduce-fmax.mlir`(镜像 `-add.mlir`)、`test/Target/RVV/pre-realized-selected-body-artifact-standalone-reduce-fsum.mlir`(镜像 `-add.mlir`)。

**对照范例**
已上硬件验证的 worked example = elementwise `add`(`test/Conversion/RVV/rvv-to-emitc-add.mlir` + `test/Target/RVV/explicit-selected-body-artifact-add.mlir`)。最近兄弟 = 整数 standalone reduce add 全套:`test/Conversion/RVV/rvv-to-emitc-standalone-reduce-add.mlir`、`test/Target/RVV/pre-realized-selected-body-artifact-standalone-reduce-add.mlir`、`test/Dialect/RVV/standalone-reduction-dataflow.mlir`。float 元素细节抄 `RVVToEmitCForwardElementwise.cpp:96-109`(vle32/vse32/vfmv_v_f on f32)。

**scalar 参考实现**
对 `const float* x` 长 `n`、seed `acc[0]`:
- `fsum`:`float a = acc[0]; for(i) a += x[i]; out[0]=a;` —— 与 `vfredosum` 的 `seed + x[0] + x[1] + ...` 顺序**逐字节相等**(这正是必须用有序 `vfredosum`、禁用无序 `vfredusum` 的原因:无序在硬件不可复现)。
- `fmax`:`float a = acc[0]; for(i) a = fmaxf(a, x[i]); out[0]=a;`
- `fmin`:同上用 `fminf`。

NaN/±0:RVV `vfredmax/vfredmin` 遵循 IEEE-754-2019 maximumNumber/minimumNumber(恰一个为 NaN 取非 NaN;`+0 > -0`),与 C99 `fmaxf/fminf` 一致。语义近似 ggml 量化里的 amax(`vfabs→vfredmax`)与 ggml 顺序 f32 求和。**测试向量必须含一个 NaN lane 与一个 ±0 边界**以证明 parity。

**验收(三层)**
- Tier 1 lit FileCheck:三个新 .mlir 通过。emitter golden 须出现 `tcrv_emitc_<kernel>_<variant>` 入口、`__riscv_vfmv_v_f_f32m1` seed splat、`__riscv_vfredosum_vs_f32m1_f32m1`(fsum)/`__riscv_vfredmax_vs_f32m1_f32m1`(fmax)、`__riscv_vse32_v_f32m1` lane0 store、`CHECK-NOT: tcrv_rvv.`;dataflow 负例覆盖错 kind / 错 accumulator_layout / 非 float seed C-type。
- Tier 2 object-compile:`/usr/lib/llvm-20/bin/clang -target riscv64 -march=rv64gcv -c gen.c` 通过(证明合法 RVV C)。
- Tier 3 数值(qemu):将生成的 C 与 harness 编成 riscv64 可执行,用 `qemu-riscv64` 运行,三种 kind 对标量参考实现 **逐字节相等**(含 NaN/±0 向量),打印 `proof ok`。
- PR 说明须点明:① typed facts 位置(`standalone_reduce` 的 `kind`/`accumulator_layout`/`result_layout` + `with_vl` 的 SEW/LMUL/policy);② route 由 op identity + kind 事实推导(`RVVSelectedBodyOperationKind`),非 route-id 字符串;③ 关键 intrinsics(vfredosum/vfredmax/vfredmin、vfmv_v_f、vle32/vse32 f32);④ 未触禁忌:无 dtype 前缀 helper、无 source-front-door 正路、无 route-id/metadata 驱动语义、common-EmitC 无硬编码 dtype/SEW/LMUL。stretch(非必需):扩 `scripts/rvv_generated_bundle_abi_e2e.py` 的整数-only 模板做 dry-run。

**目标 IR 形状(草图)**
> 目标形状:在你给这 5 层加上 float 支持之前**不可解析**(kind="fsum"、float seed、f32 layout 都会被现验证器拒);精确语法照搬 add 范例 + 兄弟 fixture(`rvv-to-emitc-standalone-reduce-add.mlir`)。下面**不保证 parse**。
```mlir
tcrv.exec.kernel @fsum_kernel {
  tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
    %x   = tcrv_rvv.runtime_abi_value {c_name="x",   c_type="const float *", role="lhs-input-buffer", ...} : !tcrv_rvv.runtime_abi_value
    %acc = tcrv_rvv.runtime_abi_value {c_name="acc", c_type="const float *", role="accumulator-input-buffer", ...} : !tcrv_rvv.runtime_abi_value
    %out = tcrv_rvv.runtime_abi_value {c_name="out", c_type="float *",       role="output-buffer", ...} : !tcrv_rvv.runtime_abi_value
    %n   = tcrv_rvv.runtime_abi_value {c_name="n",   c_type="size_t",        role="runtime-element-count"} : index
    %vl = tcrv_rvv.setvl %n {lmul="m1", policy=#tcrv_rvv.policy<tail=agnostic, mask=agnostic>, sew=32:i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul="m1", sew=32:i64, policy=#tcrv_rvv.policy<tail=agnostic, mask=agnostic>, /* ...selected-lowering-boundary... */} {
      %in  = tcrv_rvv.load %x, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      %red = tcrv_rvv.standalone_reduce %in, %acc, %vl
               {kind="fsum",
                accumulator_layout="scalar-f32-seed-lane0-from-accumulator-input",
                result_layout="store-standalone-reduction-lane0-to-output-scalar"}
             : !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      tcrv_rvv.store %out, %red, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}
```
(fmax/fmin 仅把 `kind` 换成 `"fmax"`/`"fmin"`;Target/RVV tier 用 `tcrv_rvv.typed_standalone_reduce_pre_realized_body ... op_kind="standalone_reduce_fsum" ... sew=32`,镜像 `-add.mlir` artifact。)

---

## 8. vcompress.vm + viota.m

**目标 / 新能力** — 让编译器第一次能发射 **stream compaction（流压缩）**：给定一个 compare 产生的 mask，用 `vcompress.vm` 把 mask 选中的 lane 紧凑打包到目标缓冲区前部，用 `viota.m` 计算每个 lane 的 prefix-count（前置激活计数 = 压缩后下标），并用 `vcpop.m` 求出激活总数写回标量。这是 argsort / top-k / MoE 路由的底层原语（本 slice 只交付**两个原语本身的发射 + 单块 byte-exact**；top-k/MoE 的组合是下游，不在本 slice）。**边界**：bounded 到 `n <= VLMAX`（单 `with_vl` 块，复用现有 strip-mine emit 不改）；多块 running-count 形态见文末 N+1 扩展。

**当前实现现状** —
- `grep -rni "viota_m|vcompress_vm|vcompress_v" lib include test` = **0 命中**。
- 操作种类枚举 `RVVSelectedBodyOperationKind`（`include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h:25-105`）**无** `Compress`/`Compact`/`StreamCompaction` 成员。
- 唯一近似物是**误名**：`riscvIotaIntrinsicName()`（`lib/Conversion/RVV/RVVToEmitCSupport.cpp:354-360`）发射的是 `__riscv_vid_v_*`（lane-index `0,1,2,...`），**不是** `viota.m`（mask 前缀计数）。`vrgather` 仅作为固定码表广播存在（CodebookFp4/GridCodebook 族）。即：今天能数下标(`vid`)、能查固定表(`vrgather`)，但**不能按 mask 压缩数据流**。

**改哪些文件**（镜像 sibling = computed-mask-memory）——
- **ODS op**：`include/TianChenRV/Dialect/RVV/IR/RVVOps.td`，紧挨 `TypedComputedMaskMemoryPreRealizedBodyOp`（line 1990）新增：① 细粒度 body op `tcrv_rvv.viota`（mask -> index_vector）、`tcrv_rvv.compress`（(vector,mask) -> vector）、`tcrv_rvv.mask_popcount`（mask -> index 标量）；② bounded selected-body hook `tcrv_rvv.typed_stream_compaction_pre_realized_body`（ABI 形如 `$cmp_lhs,$cmp_rhs,$src,$dst,$index_out,$count_out,$n` + `op_kind/predicate_kind/mask_source/sew/lmul/policy`）。
- **op-kind 枚举**：`RVVEmitCRouteProvider.h:25-105` 加 `StreamCompaction`（以及对应 MemoryForm）。
- **verifier**：`lib/Dialect/RVV/IR/RVVDialectPreRealizedMemoryOps.cpp`——照 computed-mask-memory verifier 的 4 条拒绝（仅支持 `slt`、source 必须 `source-input-buffer` 角色、`mask_source` 仅 `compare-produced-mask-same-vl-scope`、拒 authority 元数据如 `route_id`）改写，**新增**两条角色校验：`index_out` 必须 `compaction-index-output-buffer`、`count_out` 必须 `compaction-count-output-buffer`（这两个 role 字符串今天不存在，新增它们是本扩展的一部分——别去改沿用单一 output 角色的旧 verifier）。
- **realization owner**：**新建** `lib/Plugin/RVV/RVVStreamCompactionSelectedBodyRealizationOwner.cpp`（照 `RVVComputedMaskMemorySelectedBodyRealizationOwner.cpp`），并在 `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp:34-64` 注册一行 `{"stream compaction", isPreRealized..., realizePreRealized...}`。把 pre-realized body 展开成 `setvl/with_vl/load/load/compare/viota/compress/mask_popcount/(setvl count)/store/store` 结构。
- **route family**：**新建** `lib/Plugin/RVV/EmitC/RVVEmitCStreamCompactionRouteFamilyPlanOwners.cpp`（照 `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`），在 `RVVEmitCRouteFamilyDerivation.cpp`（include 段 line 22-28 + profile 表 line ~100）登记。
- **emitter**：**新建** `lib/Conversion/RVV/RVVToEmitCStreamCompaction.cpp`（7 个现有 emitter 族都是 quant/reduce 形不合用）+ 在 `RVVToEmitC.cpp` 主 dispatch 接上新 body op；**manglers** 加到 `RVVToEmitCSupport.{h,cpp}`：`riscvCompressIntrinsicName` -> `__riscv_vcompress_vm_<dtype><lmul>`、`riscvViotaMaskIntrinsicName` -> `__riscv_viota_m_u<sew>m<lmul>`、`riscvMaskPopcountIntrinsicName` -> `__riscv_vcpop_m_b<eew>`（i32m1 时 eew=32）。压缩存储用 `setvl(count) -> store` 表达「只写 count 个 lane」，复用现有 store emit（注意：body 内第二个 setvl 是否被 emit 支持需在此打通）。
- **tests**：`test/Dialect/RVV/stream-compaction-dataflow.mlir`（typed-body + verifier 拒绝，照 `computed-mask-memory-dataflow.mlir`）；`test/Conversion/RVV/rvv-to-emitc-stream-compaction.mlir`（emitter golden，照 `rvv-to-emitc-computed-mask-unit-load-store.mlir`）；`test/Target/RVV/explicit-selected-body-artifact-stream-compaction.mlir`（harness 形，照 `explicit-selected-body-artifact-add.mlir`）。

**对照范例** — 主范例 = **add**（`test/Target/RVV/explicit-selected-body-artifact-add.mlir` + `test/Conversion/RVV/rvv-to-emitc-add-cpp-golden.mlir`：strip-mine 循环 + ABI 声明序 + 入口名规则）。最近 sibling = **computed-mask-memory**（`RVVOps.td:1990`，compare 产 mask 再被 memory op 消费的完整脚手架）。count 标量回写镜像 **standalone-reduce**（`computed-mask-standalone-reduction-dataflow.mlir`，`result_layout="store-...-to-output-scalar"`）。

**scalar 参考实现** — 与 seed 的「copy-active-in-order + running prefix-count」一一对应（两路输出都查）：
```c
// pred(i) = (cmp_lhs[i] < cmp_rhs[i])    // slt，与 sibling 一致
size_t k = 0;
for (size_t i = 0; i < n; ++i) {
  index_out[i] = (int32_t)k;             // viota.m: lane i 之前的激活数（前缀计数）
  if (cmp_lhs[i] < cmp_rhs[i]) {
    dst[k] = src[i];                     // vcompress.vm: 按序紧凑拷贝激活元素
    ++k;
  }
}
*count_out = (int32_t)k;                 // vcpop.m: 激活总数
```
`dst` 的有效区间是 `[0,k)`（k 之后内容未定义，oracle 与 harness 都只比对前 `*count_out` 个）。`index_out` 比对全 `n` 个 lane。

**验收(三层)** —
1. **lit FileCheck**：`stream-compaction-dataflow.mlir` 跑 `tcrv-opt --split-input-file --verify-diagnostics` 通过（含新 verifier 的 4+2 条拒绝）；emitter golden 中 `__riscv_viota_m_u32m1`、`__riscv_vcompress_vm_i32m1`、`__riscv_vcpop_m_b32` 三条均出现，入口名 = `tcrv_emitc_<kernelSym>_<variantSym>`，extern "C" 形参 = body 内 `runtime_abi_value` 声明序（cmp_lhs,cmp_rhs,src,dst,index_out,count_out,n）。
2. **object-compile**：`/usr/lib/llvm-20/bin/clang -target riscv64 -march=rv64gcv -c gen.cpp` 零报错（证明是合法 RVV C）。
3. **数值（qemu）**：将生成的 C 与 harness 编成 riscv64 可执行，用 `qemu-riscv64` 运行随机数组（含 `n<=VLMAX`、激活率 0%/50%/100% 边界），`dst[0..count)`、`index_out[0..n)`、`*count_out` 全部 byte-exact 等于上面的 scalar 参考实现，打印 `proof ok`。
- **PR 说明须含**：typed facts 位置（`sew/lmul/predicate_kind/mask_source` 在 pre-realized body 的 attr，role 在各 `runtime_abi_value`）；route-from-facts（route 由 op identity + mask_source fact 推出，非字符串 route_id）；关键 intrinsics（vcompress.vm/viota.m/vcpop.m）；**无 forbidden 反模式**（不得用 dtype 前缀 helper、source-front-door 正路、route-id/metadata 驱动语义、common-EmitC 写死 dtype/SEW/LMUL）。

**目标 IR 形状(草图)** — 目标形状：**在你新增该 op 之前不可解析**；精确语法照搬 add 范例 + computed-mask-memory 兄弟 fixture（dataflow 形，用于 dialect FileCheck；harness 可跑形请另照 add artifact 加 `capability`/`variant`/`dispatch`/`fallback` 外壳）。
```mlir
// 目标形状(不保证可解析): 单 with_vl 块, n <= VLMAX
%cmp_lhs = tcrv_rvv.runtime_abi_value {c_name="cmp_lhs", c_type="const int32_t *", role="lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
%cmp_rhs = tcrv_rvv.runtime_abi_value {c_name="cmp_rhs", c_type="const int32_t *", role="rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
%src     = tcrv_rvv.runtime_abi_value {c_name="src",     c_type="const int32_t *", role="source-input-buffer"} : !tcrv_rvv.runtime_abi_value
%dst     = tcrv_rvv.runtime_abi_value {c_name="dst",     c_type="int32_t *", role="output-buffer"} : !tcrv_rvv.runtime_abi_value
%idx_out = tcrv_rvv.runtime_abi_value {c_name="index_out", c_type="int32_t *", role="compaction-index-output-buffer"} : !tcrv_rvv.runtime_abi_value
%cnt_out = tcrv_rvv.runtime_abi_value {c_name="count_out", c_type="int32_t *", role="compaction-count-output-buffer"} : !tcrv_rvv.runtime_abi_value
%n  = tcrv_rvv.runtime_abi_value {c_name="n", c_type="size_t", role="runtime-element-count"} : index
%vl = tcrv_rvv.setvl %n {lmul="m1", sew=32:i64, policy=#tcrv_rvv.policy<tail=agnostic, mask=agnostic>} : index -> !tcrv_rvv.vl
tcrv_rvv.with_vl %vl attributes {lmul="m1", sew=32:i64, policy=#tcrv_rvv.policy<tail=agnostic, mask=agnostic>} {
  %a    = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32,"m1">
  %b    = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32,"m1">
  %m    = tcrv_rvv.compare %a, %b, %vl {kind="slt"} : !tcrv_rvv.vector<i32,"m1">, !tcrv_rvv.vector<i32,"m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32,"m1">
  %s    = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32,"m1">
  %idx  = tcrv_rvv.viota %m, %vl : !tcrv_rvv.mask<i32,"m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32,"m1">        // __riscv_viota_m_u32m1
  %pk   = tcrv_rvv.compress %s, %m, %vl : !tcrv_rvv.vector<i32,"m1">, !tcrv_rvv.mask<i32,"m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32,"m1"> // __riscv_vcompress_vm_i32m1
  %cnt  = tcrv_rvv.mask_popcount %m, %vl : !tcrv_rvv.mask<i32,"m1">, !tcrv_rvv.vl -> index                     // __riscv_vcpop_m_b32
  tcrv_rvv.store %idx_out, %idx, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32,"m1">, !tcrv_rvv.vl   // 写全 n 个前缀计数
  %cvl  = tcrv_rvv.setvl %cnt {lmul="m1", sew=32:i64, ...} : index -> !tcrv_rvv.vl
  tcrv_rvv.store %dst, %pk, %cvl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32,"m1">, !tcrv_rvv.vl       // 只写 count 个紧凑元素
  // count_out: 标量回写 %cnt (镜像 standalone-reduce 的 store-scalar-to-output)
} : !tcrv_rvv.vl
```

**N+1 扩展（不在本 slice，写明以保持诚实）** — 多块 `n > VLMAX`：每个 strip 的 `dst` 写偏移是 **data-dependent running total**（`dst + 累计count`、长度 = 本块 popcount），不是 strip-mine 归纳变量 `i`。这需要改 `RVVToEmitC.cpp` 的 strip-mine emit（携带跨迭代标量累加器），是独立的 emit-loop 结构改动；亦可走 `viota -> 掩码索引 scatter (vsuxei)` 复用现有 indexed-scatter 机制。本 slice 只交付单块原语。

---

## 9. f16<->f32 vector convert (vfwcvt.f.f.v / vfncvt.f.f.w)

**目标 / 新能力**
让编译器把 f16<->f32 的**浮点-浮点向量转换**作为 first-class typed convert op 表达并发射，重点是 **f32→f16 窄化(`vfncvt.f.f.w`)+ round-trip**。最终交付一个 round-trip kernel：load f32 → 窄化到 f16 → 加宽回 f32 → store f32，对 scalar `_Float16` RNE 参考实现 byte-exact（目标需 `+zvfh`）。今天 convert 只能整数加宽，f32→f16 浮点窄化在全树**完全不存在**。

**当前实现现状**
- convert kind 白名单**只整数加宽**：`isSupportedGenericWideningConvertKind`(`lib/Dialect/RVV/IR/RVVDialect.cpp:2009`)={`widen_i32_to_i64`,`sign_extend_widen_vf2`}；emitter grid(`lib/Conversion/RVV/RVVToEmitC.cpp:3411-3421`)只接 i16mf2→i32m1 / i32m1→i64m2，发整数 `vwcvt_x_x_v`(:3433)。
- f32→f16 窄化 `vfncvt_f_f` / `vfncvt.f.f`：`grep -rn "vfncvt_f_f" .` **空**(只有量化用的 `vfncvt_x_f_w`，RVVToEmitCSupport.cpp:252)。无 op kind、无 mangler、无 emitter 分支。
- f16→f32 加宽 `vfwcvt_f_f`：仅作为 block-quant **内联手搓步骤**(读 fp16 块标度，`RVVToEmitCBlockQuantLinear.cpp:5253/6395`)，不是 convert-op 路由；mangler `riscvFloatWideningConvertIntrinsicName`(`RVVToEmitCSupport.cpp:224`)已存但**被通用 convert emitter 无视**。
- f16 通用 dtype **未接线**：`vectorDType`(`RVVToEmitCSupport.cpp:703`)、`vectorScalarCType`(:744)无 isF16 分支；block-quant 用本地硬编码 opaque 类型绕过。
- 全树无 f16 typed-body fixture(`grep xf16 test/` 空)。注：`!tcrv_rvv.vector<f16,"mf2">` 类型**本身已能 parse**(tcrv-opt 实测，verifier 停在 kind 检查而非类型)。

**改哪些文件**(镜像整数 WideningConvert 全管线 + 加 f16 dtype 接线)
1. **ODS** `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:8571`：复制 `WideningConvertOp` 成 `def NarrowingConvertOp`(f32m1→f16mf2，kind `narrow_f32_to_f16`)；同时给 `widening_convert` 描述加 f16→f32 方向。
2. **verifier** `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:8620`(+ `RVVDialect.cpp:2009/1457-1478`)：`isSupportedGenericWideningConvertKind` 加 `widen_f16_to_f32`、给 `WideningConvertOp::verify` 加 f16mf2→f32m1 形状分支；新增 `NarrowingConvertOp::verify` + `isSupportedGenericNarrowingConvertKind("narrow_f32_to_f16")` + f32m1→f16mf2 形状检查(镜像 :8668-8689 的 i32→i64 分支)。
3. **realization owner** `lib/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.cpp:36-37`：关系白名单加 `float-f16mf2-to-f32m1` 与 `float-f32m1-to-f16mf2`；新窄化 op 在 `RVVSelectedBodyRealization.cpp:57` 注册(镜像 widening conversion 那条)。
4. **route family** `lib/Plugin/RVV/EmitC/RVVEmitCRouteFamilyDerivation.cpp:2541`(derive)+`RVVEmitCControlPolicyPlanOwners.cpp:483/1124`(plan)+关系串 `RVVEmitCRoutePlanningInternal.h:906-908`：加两条浮点关系并让窄化 op 进入分类。
5. **emitter** `lib/Conversion/RVV/RVVToEmitC.cpp:3385`(`emitWideningConvert`)：kind==`widen_f16_to_f32` 时改发 `riscvFloatWideningConvertIntrinsicName`(已存，vfwcvt_f_f_v，dtype/lmul 从 **result** f32 取)；新增 `emitNarrowingConvert`(发 vfncvt_f_f_w，dtype/lmul 从 **result** f16 取)，在 `:872` 的 op dyn_cast dispatch 处挂上。
6. **mangler** `lib/Conversion/RVV/RVVToEmitCSupport.cpp:256` 旁:新增 `riscvFloatNarrowToFloatConvertIntrinsicName` → `__riscv_vfncvt_f_f_w_<dstDtype><dstLmul>`(如 f16mf2)；声明进 `RVVToEmitCInternal.h`。
7. **f16 dtype 接线(共享 infra,慎改+全量 lit 回归)**：`vectorDType`(:703)加 `isF16()→"f16"`；`vectorScalarCType`(:744)加 `isF16()→"_Float16"`；`convertVectorTypeToEmitC`(`RVVToEmitC.cpp` ~:5047 的 float 分支)加 f16→`vfloat16mf2_t`/`vfloat16m1_t`。
8. **tests**：`test/Dialect/RVV/generic-narrowing-conversion-dataflow.mlir`(verifier) + 扩 `generic-widening-conversion-dataflow.mlir`；`test/Conversion/RVV/rvv-to-emitc-fp16-roundtrip.mlir`(emitter golden,镜像 rvv-to-emitc-widening-convert-i16-to-i32.mlir);`test/Target/RVV/...-fp16-roundtrip-artifact.mlir`。

**对照范例**
- 端到端范本(必读)：elementwise **add** — `test/Conversion/RVV/rvv-to-emitc-add.mlir`(load→op→store 结构 + 函数名/ABI/setvl 循环约定) + `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`。
- 最近兄弟(主体 copy-then-adapt)：整数 **WideningConvert** 全管线(ODS RVVOps.td:8571 / verifier RVVDialectWideningOps.cpp:8620 / owner RVVWideningConversionSelectedBodyRealizationOwner.cpp / emitter RVVToEmitC.cpp:3385 / fixture rvv-to-emitc-widening-convert-i16-to-i32.mlir / artifact pre-realized-selected-body-artifact-widen-i16-to-i32.mlir)。窄化方向另参考量化 narrow 砖 `vfncvt_x_f_w`(RVVToEmitCForwardElementwise.cpp:879)作 callee 拼写法范例。

**scalar 参考实现**
对每个输入 `float x[i]`：`ref[i] = (float)(_Float16) x[i]`(即先按 IEEE binary16 RNE 舍入,再无损升回 f32)。这正是 ggml 在 Zvfh 硬件上 `GGML_FP32_TO_FP16` 后接 `GGML_FP16_TO_FP32` 的等价语义(原生 `_Float16` cast)。kernel 端 `out[i] = vfwcvt.f.f.v(vfncvt.f.f.w(x[i]))`;`vfncvt.f.f.w` 用 dynamic frm = round-to-nearest-EVEN(与 `_Float16` cast 一致),`vfwcvt.f.f.v` 精确。harness 对 `out[i]` 与 `ref[i]` 做 **bit 级**比较(memcmp uint32),应全等。判别性反例:用 truncation(frm=rtz)或漏掉 widen 直接 bitcast,byte-compare 必须 FAIL。

**验收(三层)**
1. **lit FileCheck**(本地,人人):`tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck`。断言 `CHECK-NOT: tcrv_rvv.` / `CHECK-NOT: unrealized_conversion_cast`;函数名 `tcrv_emitc_<kernel>_<variant>`;`__riscv_vfncvt_f_f_w_f16mf2`(suffix=result f16) 与 `__riscv_vfwcvt_f_f_v_f32m1`(suffix=result f32) 各出现一次;窄化 setvl/load 跑 source 配置(e32m1),加宽跑 e16mf2/e32m1 链。
2. **object-compile**(本地):`mlir-translate --mlir-to-cpp` 出 C 后 `/usr/lib/llvm-20/bin/clang -target riscv64 -march=rv64gcv -c gen.cpp`,零报错(证 vfh intrinsic 拼写合法)。
3. **数值（qemu）**:将生成的 C 与 harness 编成 riscv64 可执行(`-march=rv64gcv+zvfh`),用 `qemu-riscv64` 运行随机 + 边界(±0、subnormal-rounding、>65504 溢出到 inf、NaN)输入;byte-exact vs 上面参考实现;打印 `proof ok`。
4. **PR 说明须含**:typed facts 位置(kind/SEW/LMUL 在 op + with_vl,**非** route-id/metadata);route-from-facts(关系串如何从 typed body 推出,而非字符串猜);关键 intrinsic(`vfncvt_f_f_w`/`vfwcvt_f_f_v`,suffix 取自 **result** 向量类型);**无 forbidden 反模式**:无 dtype 前缀 helper、无 source-front-door 正路、无 route-id/metadata 驱动语义、common-EmitC 不硬编码 dtype/SEW/LMUL。

**目标 IR 形状(草图)**
> 目标形状:在你新增 `narrowing_convert` op + 扩 `widening_convert` 白名单 + 接线 f16 dtype 之前**不可解析**;精确语法照搬 add 范例(rvv-to-emitc-add.mlir)+ 兄弟 fixture(rvv-to-emitc-widening-convert-i16-to-i32.mlir)。不声称它现在能 parse。

```mlir
tcrv.exec.variant @rvv_fp16_roundtrip attributes {origin = "rvv-plugin", requires = [@rvv], ...} {
  %in  = tcrv_rvv.runtime_abi_value {c_name = "in",  c_type = "const float *", role = "lhs-input-buffer", ...} : !tcrv_rvv.runtime_abi_value
  %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *",       role = "output-buffer",    ...} : !tcrv_rvv.runtime_abi_value
  %n   = tcrv_rvv.runtime_abi_value {c_name = "n",   c_type = "size_t",        role = "runtime-element-count", ...} : index
  %vl  = tcrv_rvv.setvl %n {lmul = "m1", sew = 32 : i64, policy = ...} : index -> !tcrv_rvv.vl
  tcrv_rvv.with_vl %vl attributes {lmul = "m1", sew = 32 : i64, ...} {
    %x  = tcrv_rvv.load %in, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
    // 新窄化 op：f32/m1 -> f16/mf2 (RNE)  —— 发 __riscv_vfncvt_f_f_w_f16mf2
    %h  = tcrv_rvv.narrowing_convert %x, %vl {kind = "narrow_f32_to_f16"} : !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f16, "mf2">
    // 扩白名单的浮点加宽：f16/mf2 -> f32/m1 —— 发 __riscv_vfwcvt_f_f_v_f32m1
    %y  = tcrv_rvv.widening_convert %h, %vl {kind = "widen_f16_to_f32"} : !tcrv_rvv.vector<f16, "mf2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
    tcrv_rvv.store %out, %y, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
  } : !tcrv_rvv.vl
}
```
注:f16 只活在寄存器(`%h` 不落内存),刻意避开 f16 的 vle16/vse16 与 f16 指针 ABI 接线 —— 把本 slice 压到可做规模。若想加分,可再做 store-f16 变体(需补 f16 load/store dtype 接线)。

---

## 10. integer narrowing convert (saturating vnclip)

**目标 / 新能力**
让编译器新会做"整数饱和窄化转换"：把一个宽整数向量按 RVV `vnclip` 语义、带符号饱和地收窄成一半宽度的向量——i32→i16、i16→i8（每步恰好 2×）。这是 requantize/pack 的核心原语：把 i32 点积累加器饱和打包回 i8/i16，**溢出时夹紧到目标范围而非回绕**。新增独立 typed op `tcrv_rvv.narrowing_convert`，kind = `narrow_sat_i32_to_i16` / `narrow_sat_i16_to_i8`（可选 bonus：`narrow_trunc_i16_to_i8` 走回绕 vncvt）。

**当前实现现状**
- `vnclip` / `vxrm` 在整个 `lib/ include/ test/` 中 0 命中。
- convert op 面**只有 widening**：`RVVOps.td:8571` 仅 `WideningConvertOp`，无 NarrowingConvertOp；emitter dispatch `RVVToEmitC.cpp:871` 只 `dyn_cast<WideningConvertOp>`，无 narrowing 分支。
- 休眠 partial：`riscvNarrowingConvertIntrinsicName`（`RVVToEmitCSupport.cpp:238`）与 `riscvFloatNarrowToIntConvertIntrinsicName`（:252）已定义+声明，但 lib/ 内 **0 调用方**，且是 **truncating** `__riscv_vncvt_x_x_w_` / `vfncvt`，**不是饱和 vnclip**。
- truncating vncvt 仅以硬编码字符串内联在融合量化核（`BlockQuantLinear.cpp:3530`、`TernaryBinary.cpp:2134`、`ForwardElementwise.cpp:881`），从未作为独立 op 暴露。

**改哪些文件**（逐处镜像 `widening_convert`，把 source/dest 的宽窄角色对调）
- **ODS**：`include/TianChenRV/Dialect/RVV/IR/RVVOps.td` —— 仿 `WideningConvertOp`(8571) 加 `def NarrowingConvertOp`（`AnyType:$source, $vl`, `StrAttr:$kind`, 1 个 `AnyType` result, `TCRVEmitCLowerableOpInterface`, `hasVerifier=1`, 同 assemblyFormat）。
- **verifier**：`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` —— 仿 `WideningConvertOp::verify()`(8620) 加 `NarrowingConvertOp::verify()`；在 `RVVDialect.cpp` 仿 `isAllowedWideningConvertAttr`(777)/`isSupportedGenericWideningConvertKind`(2009) 加 narrowing 版（白名单两个 kind）。要点：enclosing `with_vl` 携带 **DEST(窄)** SEW/LMUL；网格 = source i32`m2`→result i16`m1`，source i16`m2`→result i8`m1`（拒绝其它配对/无符号，落回退）。
- **realization owner**：`lib/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.cpp` —— 仿 `createRealizedGenericWideningConvert`(232)（EXPLICIT 路径可先不做 pre-realized owner）；若新建文件，在 `RVVSelectedBodyRealization.cpp:57` 注册表加一项。
- **route family**：`RVVEmitCRouteFamilyDerivation.cpp` 仿 2290–2328（`facts.typedComputeOpName = "tcrv_rvv.narrowing_convert"`、memoryForm、conversionIntrinsic 等）；`RVVEmitCRouteAnalysis.cpp:980` 仿 `recordRVVSelectedBodyWideningConvert`；plan owner 与 widening 同住 `RVVEmitCControlPolicyPlanOwners.cpp`/`RVVEmitCRoutePlanning.cpp`——**grep `widening_convert` 整个 `lib/Plugin/RVV/EmitC` 逐处加 narrowing 对应**。
- **emitter**：`lib/Conversion/RVV/RVVToEmitC.cpp` 仿 `emitWideningConvert`(3386) 加 `emitNarrowingConvert` + 在 871 旁加 dispatch。callee 取**结果(窄)类型** = `__riscv_vnclip_wx_<dst><lmul>`（i16m1/i8m1）。关键差异：vnclip 比 vwcvt 多两参——`CallOpaqueOp` 操作数 = `{source, shiftLit(0), vxrmLit, bodyVL}`，shift/vxrm 用 `emitc::LiteralOp` 造（`size_t 0` 与 `__RISCV_VXRM_RNU`，模式见 `BlockQuantLinear.cpp:94` 的 `sizeLit` lambda）。在 `RVVToEmitCSupport.cpp` 仿 238 加 `riscvSaturatingNarrowingConvertIntrinsicName`→`__riscv_vnclip_wx_<dst><lmul>`，并在 `RVVToEmitCSupport.h:97` 旁声明。
- **tests**：`test/Dialect/RVV/generic-narrowing-conversion-dataflow.mlir`（仿 `generic-widening-conversion-dataflow.mlir`，含 reject-kind/源型/结果型/dest-config 负例）；`test/Conversion/RVV/rvv-to-emitc-narrowing-convert-i32-to-i16.mlir`(+`-negative`)（仿 `rvv-to-emitc-widening-convert-i32-to-i64.mlir`）；`test/Target/RVV/explicit-selected-body-artifact-narrow-i32-to-i16.mlir`（仿 `explicit-selected-body-artifact-widen-i32-to-i64.mlir`）；harness + scalar oracle。

**对照范例**
worked example = elementwise **`add`**（整体 harness/oracle/ABI/extern "C" 接线形状）。最近兄弟 = **`widening_convert`**（同形单 op convert、同 convert-mangler 家族、同 2× 网格的反向）。

**scalar 参考实现**
对每 lane（vnclip `shift=0` → 无移位、无 rounding，纯带符号饱和）：
- `narrow_sat_i32_to_i16`：`int32_t x=in[i]; out[i]=(int16_t)(x>32767?32767:(x<-32768?-32768:x));`
- `narrow_sat_i16_to_i8`：`int16_t x=in[i]; out[i]=(int8_t)(x>127?127:(x<-128?-128:x));`
判别性输入必须含**溢出边界**：恰在 ±32767/±32768、刚越界 (32768, -32769)、以及大正/大负值，证明"饱和"而非 truncating-vncvt 的"回绕"。语义对应 RVV `vnclip.wx`（带符号饱和窄化）；典型用途 = 把 ggml i32 GEMM 累加器 requantize/pack 回 int8（饱和而非回绕，避免数值爆点）。

**验收(三层)**
1. **lit FileCheck**：dataflow 测试经 verifier；emitter golden 出 `emitc.func @tcrv_emitc_rvv_narrow_i32_to_i16_kernel_rvv_narrow_i32_to_i16(...)`、`__riscv_vsetvl_e16m1`、源 `__riscv_vle32_v_i32m2`、`__riscv_vnclip_wx_i16m1(<src>, 0, __RISCV_VXRM_RNU, <vl>)`、`__riscv_vse16_v_i16m1`，且 `CHECK-NOT: tcrv_rvv.` / `CHECK-NOT: unrealized_conversion_cast`。
2. **object-compile**：`/usr/lib/llvm-20/bin/clang -target riscv64 -march=rv64gcv -c gen.c`（证明生成的是合法 RVV C）。
3. **数值（qemu）**：将生成的 C 与 harness 编成 riscv64 可执行，用 `qemu-riscv64` 运行；对 scalar 饱和参考实现（含溢出边界）byte-exact；打印 `proof ok`。
4. **PR 说明**需写明：typed facts 位置（`kind` attr 在 narrowing_convert + SEW/LMUL 在 typed vector + with_vl 携带 dest config）；route-from-facts（`typedComputeOpName="tcrv_rvv.narrowing_convert"` 驱动 family，**不是** route id/字符串/metadata）；关键 intrinsic（`__riscv_vnclip_wx_*`、shift=0、vxrm）；无 forbidden 反模式（无 dtype-prefixed helper、无 source 正门路径、无 route-id/metadata 决定语义、无 common-EmitC 硬编码 dtype/SEW/LMUL——语义全来自 typed-body 网格 + kind）。

**目标 IR 形状(草图)**
> 目标形状：**在你新增该 op 之前不可解析**；精确语法照搬 add 范例 + 兄弟 fixture `rvv-to-emitc-widening-convert-i32-to-i64.mlir`（注意宽/窄方向相对 widening 对调：source 是更宽 m2、dest 更窄 m1；setvl/store 在 DEST e16m1、load 在 SOURCE e32m2，两侧 VLMAX 相等故 vl 一致）。

```mlir
tcrv.exec.kernel @rvv_narrow_i32_to_i16_kernel {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
  tcrv.exec.variant @rvv_narrow_i32_to_i16 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
    %in  = tcrv_rvv.runtime_abi_value {c_name = "in",  c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int16_t *",       ownership = "target-export-abi-owned", role = "output-buffer"}    : !tcrv_rvv.runtime_abi_value
    %n   = tcrv_rvv.runtime_abi_value {c_name = "n",   c_type = "size_t",          ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      %src    = tcrv_rvv.load %in, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">
      %narrow = tcrv_rvv.narrowing_convert %src, %vl {kind = "narrow_sat_i32_to_i16"} : !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m1">
      tcrv_rvv.store %out, %narrow, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i16, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}
```

---

## 11. get_rows (f32/f16 src)

**目标 / 新能力**
让编译器能把 ggml `get_rows`（token embedding 查表 + MoE expert-row 选择）作为**一个完整 forward kernel** 下沉到 RVV C：给一个整数索引张量 `index[0..nr)`，把源张量第 `index[r]` 行（`ne0` 个元素）拷到目标第 `r` 行。f32 源直接行拷贝；f16 源在拷贝时做 `f16->f32` 转换（dst 恒为 f32）。当前编译器**完全做不到**：现有索引能力只有 per-lane 的 inner gather，没有「按整数索引选整行 + 二级循环 + f16 加宽」的整体 kernel。

**当前实现现状**
- `grep -rniE "get_rows|GetRows|ggml_get_rows" --include=*.{td,cpp,h,mlir}` = **0** hits（RVVOps.td 无 op、RVVToEmitC.cpp dispatch 链 L431-525 无分支、340 测试无 fixture）。
- 唯一相邻能力是 element-level gather：`tcrv_rvv.index_load`+`indexed_load`（test/Dialect/RVV/indexed-gather-memory-dataflow.mlir），其语义是单 `with_vl` region 内 `out[lane]=data[index[lane]]`，发 `__riscv_vloxei32`（per-lane 散读，索引×byte-scale）。get_rows 不是 per-lane gather：它是**外层 row 循环 + 每行一个标量 `idx=index[r]` + 内层 unit-stride 连续行拷贝**，且 f16 还要 `vfwcvt`。两者结构正交，inner gather 不构成 get_rows 的部分实现。

**改哪些文件**（按 ggml forward-op 类，**比多数 slice 窄**）
1. `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — 新增 `GgmlGetRowsF32Op`（f16 stretch 再加 `GgmlGetRowsF16Op`），紧邻 `GgmlRopeNormF32Op`(8432)/`GgmlVecScaleF32Op`(8076) 处。带 `[TCRVEmitCLowerableOpInterface]`、`hasVerifier = 1`。operands 镜像 rope：`$src, $index, $dst, $ne0, $nr, $vl`（全 `AnyType`）+ `StrAttr:$kind`（限 `"ggml_get_rows_f32"`）+ 可选 `OptionalAttr<StrAttr>:$strip_lmul`（resource 旋钮，默认 `"m8"`，仅影响内层 strip 宽度，byte-exact 不变）。
2. `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — 新增 `GgmlGetRowsF32Op::verify()`，镜像 `GgmlRopeNormF32Op::verify()`(8378)：校验 `kind`、各 runtime_abi_value role（`src`=lhs-input-buffer、`index`=index-input-buffer、`dst`=output-buffer、`ne0`/`nr`=runtime-element-count）。
3. `lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp` — 新增 `emitGgmlGetRowsF32`（+f16）。结构：外层 `emitc::ForOp r∈[0,nr)` → 标量读 `idx=index[r]`（`AddOp(index_ptr,r)->SubscriptOp->LoadOp`，照搬 L179-194 的 `x[i]` 读，再 cast 到 size_t）→ 标量行基址 `src_row=src+idx*ne0`、`dst_row=dst+r*ne0`（`emitc::AddOp`/`MulOp`，参考 rope L296-301 指针算术）→ 内层 strip 拷贝镜像 `emitGgmlVecScaleF32`(L29-115)：`vsetvl_e32m<L>(ne0-i)` / `vle32_v_f32m<L>` / `vse32_v_f32m<L>`（f32 直拷，无 vfmul）。
4. `lib/Conversion/RVV/RVVToEmitC.cpp` — dispatch：加 `isGgmlGetRowsF32Body(scope)` 谓词（镜像 L1555/L1625）+ 分支调用 `emitGgmlGetRowsF32`（镜像 L431/L524 块，结尾 `ReturnOp(Value())`+`eraseOp`）。**勿动 L234 的 math.h header 列表**——get_rows 不调 libm。
5. `lib/Conversion/RVV/RVVToEmitCInternal.h` — 声明 `isGgmlGetRowsF32Body`(~L320/L355 处) + `emitGgmlGetRowsF32`(~L2534/L2733 处)。
6. **realization owner：N/A**（镜像 rope，本身就没有；`grep` Plugin/ 对 ggml forward op = 空）。**route family：N/A**（ggml forward op 不走 Plugin 路，dispatch 直接在 RVVToEmitC.cpp）。本 slice 不涉及 `RVV...SelectedBodyRealizationOwner` / `RVVEmitC...RouteFamilyPlanOwners`。
7. 测试：`test/Dialect/RVV/ggml-get-rows-f32-dataflow.mlir`（typed-body parse + verifier-negative，镜像现有 ggml 正/负例）；`test/Conversion/RVV/rvv-to-emitc-ggml-get-rows-f32.mlir`（emitter golden，镜像 `rvv-to-emitc-ggml-rms-norm-f32.mlir` + `rvv-to-emitc-ggml-vec-scale-f32.mlir`）。

**f16 rung（stretch）**：`vectorDType`(RVVToEmitCSupport.cpp:728) 无 f16，但**不是阻塞**——ggml emit 方法硬编码类型串（如 scale 写死 `"vfloat32m2_t"`）。在 `emitGgmlGetRowsF16` 直接写 `"__riscv_vle16_v_f16mf2"`→`vfloat16mf2_t`→`riscvFloatWideningConvertIntrinsicName("f32","m1")`(`__riscv_vfwcvt_f_f_v_f32m1`)→`"__riscv_vse32_v_f32m1"`。需 `-march=...+zvfh`；src ABI c_type 用 `const _Float16 *`（dst 仍 `float *`）。

**对照范例**
- worked example（按约定对照）：elementwise **add**（`test/Conversion/RVV/rvv-to-emitc-add.mlir`）——确立 runtime_abi_value→extern "C" ABI、setvl/strip、`__riscv_vle/vse` 词汇。
- 最近兄弟：**GgmlRopeNormF32Op**（多 buffer + 标量参数 + 标量 per-iteration 循环 + 标量 deref + 指针算术 + f32 写出，结构最近）；内层 strip 拷贝照 **GgmlVecScaleF32Op**。

**scalar 参考实现**
对照 ggml `ggml_compute_forward_get_rows_f32` / `_f16`（ops.cpp）。scope 到 2D（ggml 原生 4D + broadcast，本 slice 不做）：
```c
// f32 src:
for (size_t r = 0; r < nr; ++r) {
  int32_t i = index[r];
  for (size_t c = 0; c < ne0; ++c) dst[r*ne0 + c] = src[(size_t)i*ne0 + c];
}
// f16 src（dst 恒 f32）：dst[r*ne0+c] = (float) src_f16[(size_t)i*ne0 + c];  // ggml_fp16_to_fp32
```
f32 是纯内存拷贝（无算术 → 无 FMA/round 危害，任意 LMUL/strip 都 byte-exact）；f16 仅多一个 `f16->f32` 拓宽，与 ggml `(float)` cast 同语义。

**验收（三层）**
1. **lit FileCheck**（人人，本地）：`tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s`。断言 STRUCTURE（非 prose）：`CHECK-NOT: tcrv_rvv.` / `CHECK-NOT: unrealized_conversion_cast`；`emitc.func @tcrv_emitc_<kernel>_<variant>(` 且 ABI 参数按 body 内 runtime_abi_value 声明序（src ptr / index `const int32_t *` / dst `float *` / ne0 / nr 全到位，extern "C")；外层 `for`（step 1，row 循环）；标量 `subscript`+`load`（idx 读）；内层 `__riscv_vsetvl_e32m<L>` + `__riscv_vle32_v_f32m<L>` + `__riscv_vse32_v_f32m<L>`；f16 rung 额外 `__riscv_vle16_v_f16mf2` + `__riscv_vfwcvt_f_f_v_f32m1`；`CHECK-NOT: emitc.verbatim {{.*}}__riscv`（无裸串 blob）。
2. **object-compile**（本地证 RVV C 合法）：把 pass 输出经 `mlir-translate --mlir-to-cpp` 渲染，`/usr/lib/llvm-20/bin/clang -target riscv64 -march=rv64gcv -c gen.c`（f16 rung 用 `-march=rv64gcv_zvfh`）。
3. **数值（qemu）**：将生成的 C 与 harness 编成 riscv64 可执行（f16 加 `+zvfh`），用 `qemu-riscv64` 运行；harness 声明 `extern "C" void tcrv_emitc_<kernel>_<variant>(...)`，喂随机 src + index（含重复/边界索引），与上面 scalar 参考实现逐字节比对，打印 `proof ok`。注：ggml forward op 类不新增 `test/Target/RVV/...-artifact.mlir`（该类无此 fixture）。
- **PR 说明须含**：typed facts 位置（op 的 runtime_abi_value role + `kind` + `strip_lmul`，皆在 typed body，非 route-id/metadata）；route-from-facts（dispatch 由 `isGgmlGetRowsF32Body` 的 **op identity** 触发，不读字符串语义）；关键 intrinsics（`vle32/vse32`，f16 加 `vle16/vfwcvt_f_f`）；**无 forbidden anti-pattern**（无 dtype-prefixed helper、无 source-front-door 正路、无 route-id/metadata 驱动语义、无 common-EmitC 硬编码 dtype/SEW/LMUL——f16 类型串只在本 op 专用 emit 方法内，与 scale 写死 `vfloat32m2_t` 同纪律）。

**目标 IR 形状（草图）**
> 目标形状：在你新增该 op 之前不可解析；精确语法照搬 add 范例 + 兄弟 fixture（rope/scale 的 `runtime_abi_value` + `setvl` + `with_vl` 头）。**不**声称它能 parse。
```mlir
tcrv.exec.kernel @ggml_get_rows_f32_kernel {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
  tcrv.exec.variant @ggml_get_rows_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
    %src   = tcrv_rvv.runtime_abi_value {c_name = "src",   c_type = "const float *",   role = "lhs-input-buffer", ...} : !tcrv_rvv.runtime_abi_value
    %index = tcrv_rvv.runtime_abi_value {c_name = "index", c_type = "const int32_t *", role = "index-input-buffer", ...} : !tcrv_rvv.runtime_abi_value
    %dst   = tcrv_rvv.runtime_abi_value {c_name = "dst",   c_type = "float *",         role = "output-buffer", ...} : !tcrv_rvv.runtime_abi_value
    %ne0   = tcrv_rvv.runtime_abi_value {c_name = "ne0",   c_type = "size_t", role = "runtime-element-count", ...} : index
    %nr    = tcrv_rvv.runtime_abi_value {c_name = "nr",    c_type = "size_t", role = "runtime-element-count", ...} : index
    %vl = tcrv_rvv.setvl %ne0 {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", selected_variant = @ggml_get_rows_f32, sew = 32 : i64, status = "selected-lowering-boundary", ...} {
      // 目标：新 op；外层 nr 行循环 + per-row idx=index[r] + 内层 ne0 列 unit-stride 拷贝
      %rows = tcrv_rvv.ggml_get_rows_f32 %src, %index, %dst, %ne0, %nr, %vl
                {kind = "ggml_get_rows_f32", strip_lmul = "m8"}
              : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl
              -> !tcrv_rvv.vector<f32, "m1">
    } : !tcrv_rvv.vl
  }
}
```
（f16 rung：把 `%src` c_type 换 `const _Float16 *`、op 名换 `ggml_get_rows_f16`，dst 仍 f32。）

---

## 12. layernorm (ggml_norm)

**目标 / 新能力**
让编译器新增一个前向算子 `tcrv_rvv.ggml_norm_f32`，对一行 f32 做标准 LayerNorm（mean+variance 中心化归一化）：先求均值 `mean`，再对去均值后的值求方差 `variance`，最后 `y[i] = (x[i]-mean) / sqrt(variance+eps)`。这与已有的 `ggml_rms_norm_f32`（只做 `mean=Σx²/n` 的 RMS，不减均值）是不同的数学。编译器生成的 RVV C 要与 ggml 自己的 `ggml_compute_forward_norm_f32`（非融合、无 weight/bias）逐 bit 相同。

**当前实现现状**
`grep -rin "ggml_norm\|layernorm\|layer_norm\|LayerNorm"`（`.td/.cpp/.h/.mlir`）= **0 命中**。全树唯一的归一化算子是 `GgmlRmsNormF32Op`（`RVVOps.td:8138`，summary 明写 "Sx^2 reduce -> rsqrt -> scale"），它**不减均值、不算方差**。无 op / 无 verifier / 无 emitter / 无 test 覆盖 mean+variance 形态。学生实现后编译器才第一次能发 LayerNorm。

**改哪些文件**（精确镜像 rms_norm 的 5-文件形态）
> 注意：前向 ggml 算子**不需要** realization owner，也**不需要** route-family plan owner。已验证：`lib/Plugin/` 里 grep rms_norm/silu/softmax/rope = 0。typed body 直接写在 `status = "selected-lowering-boundary"`，dispatch 靠 op-identity marker。**不要**去找/新建 `RVVNormSelectedBodyRealizationOwner.cpp` 或 `RVVEmitCNormRouteFamilyPlanOwners.cpp`——那种文件对这个形态不存在。

1. **ODS op** `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`：在 `GgmlRmsNormF32Op`（8138–8207）之后加 `GgmlNormF32Op`。同样 4 个 runtime ABI 操作数 `input(const float*)`、`output(float*)`、`eps(float)`、`element_count(index)` + `vl`；`StrAttr:$kind`（限 `"ggml_norm_f32"`）+ `OptionalAttr<StrAttr>:$strip_lmul`；`hasVerifier=1`；assemblyFormat 照搬。
2. **verifier** `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp`：在 `GgmlRmsNormF32Op::verify()`（8000–8100）后加 `GgmlNormF32Op::verify()`，逐行镜像：白名单属性 `{"kind","strip_lmul"}`、`kind=="ggml_norm_f32"`、`strip_lmul ∈ {m1,m2,m4,m8}`、5 操作数/1 结果、`input` C 类型 `const float *`、`output` `float *`、`eps` `float`、`element_count` 为 index、结果为 `!tcrv_rvv.vector<f32,"m1">`、enclosing with_vl 带 policy。
3. **internal header** `lib/Conversion/RVV/RVVToEmitCInternal.h`：声明 `static bool isGgmlNormF32Body(...)`（挨着 327 行的 `isGgmlRmsNormF32Body`）+ `mlir::LogicalResult emitGgmlNormF32(...)`（挨着 2567 行的 `emitGgmlRmsNormF32`，签名照抄）。
4. **conversion 主文件** `lib/Conversion/RVV/RVVToEmitC.cpp`：(a) `math.h` gate——在 234 行 `if (isGgmlRmsNormF32Body(scope) || ...)` 的 OR 链里加 `isGgmlNormF32Body(scope)`（要 `sqrtf`）；(b) dispatch 分支——在 448 行 rms_norm 分支后加 `if (isGgmlNormF32Body(scope)) { emitGgmlNormF32(...); ReturnOp(); eraseOp; return success(); }`；(c) body-detector——在 1569 行 `isGgmlRmsNormF32Body` 旁实现 `isGgmlNormF32Body`（单一 `GgmlNormF32Op` 计数）。
5. **emitter** `lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp`：加 `emitGgmlNormF32`。**这是 COMPOSE 不是 copy**——见下"对照范例"。
6. **tests**：`test/Dialect/RVV/norm-dataflow.mlir`（typed-body + verifier 正/负例，镜像现有 rms_norm dataflow 测）；`test/Conversion/RVV/rvv-to-emitc-ggml-norm-f32.mlir`（emitter golden，镜像 `rvv-to-emitc-ggml-rms-norm-f32.mlir`）。

**对照范例**
- 总范例（ABI/fixture/harness 约定）：elementwise **add**。
- 最近兄弟：**`ggml_rms_norm_f32`**（同"行 reduce → scalar rsqrt → scale"骨架）。
- emitter 要拼两块：
  1. **两段 scalar-double reduce loop**，镜像 `emitGgmlRmsNormF32` 的归约块（`RVVToEmitCForwardElementwise.cpp:117–226`）。pass-1 内表达式只是 `sum + (double)x[i]`（**纯 widen+add，无 f32 乘积**，与 rms_norm 的 `x*x` 不同）。pass-2 是真正的新 IR 形态：循环体先 `emitc.subscript`+`emitc.assign` 写 `y[i]=x[i]-mean`（f32 减），**再**做 rms_norm 那套 `v*v`（f32 乘一次 round）→ `(double)` widen（FMA barrier）→ double 累加。这种 reduce 中带 store 的 fused 循环是现存任何前向算子都没有的。
  2. **就地 normalize tail** = `emitGgmlVecScaleF32`（`RVVToEmitCForwardElementwise.cpp:29–115`）那段 in-place `vle32/vfmul_vf/vse32` strip，作用在 `y` 上（`y[i]*=scale`）。**不要**用 rms_norm 的双 buffer tail（260–321）——ggml_norm 收尾就是 `ggml_vec_scale_f32(ne00, y, scale)`。

**scalar 参考实现**
ggml `ggml_compute_forward_norm_f32`（ggml-cpu/ops.cpp，紧挨 rms_norm 之前；非融合/无 weight）。`ggml_float` = double。harness 里 6 步逐字转写：
```c
ggml_float sum = 0.0;  for (i) sum  += (ggml_float)x[i];          // pass-1: 纯 widen+add
float mean = sum/ne00;                                            // double 除, 后转 f32
ggml_float sum2 = 0.0; for (i){ float v=x[i]-mean; y[i]=v; sum2 += (ggml_float)(v*v); }  // pass-2: f32 减, 写 y, f32 乘 round 后 widen 累加
float variance = sum2/ne00;                                       // double 除, 后转 f32
const float scale = 1.0f/sqrtf(variance + eps);                   // f32 add/sqrtf/reciprocal
for (i) y[i] *= scale;                                            // 就地 ggml_vec_scale_f32
```
crux 与 rms_norm 一致：**双精度累加方法是 load-bearing**，不是容差。

**验收(三层)**
1. **lit FileCheck**（`tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck`）：`emitc.include <"math.h">`；entry `emitc.func @tcrv_emitc_ggml_norm_f32_kernel_ggml_norm_f32(`；**两个** `emitc.variable lvalue<double>`（sum 与 sum2，易漏第二个）；pass-1 `cast …f32→double` + double `add`；mean `div(double,double)` + `cast double→float`；pass-2 `sub(float,float)` + `subscript`/`assign` 写 y + `mul(float)`(v*v) + `cast float→double` + double `add`；variance `div`+`cast`；`call_opaque "sqrtf"` + f32 reciprocal `div`；就地 strip `__riscv_vsetvl_e32m8`/`vle32`/`vfmul_vf`/`vse32`；`CHECK-NOT: tcrv_rvv.`、`CHECK-NOT: unrealized_conversion_cast`、`CHECK-NOT: emitc.verbatim {{.*}}__riscv`。
2. **object-compile**：`/usr/lib/llvm-20/bin/clang -target riscv64 -march=rv64gcv -c gen.c`（证明是合法 RVV C）。
3. **数值（qemu）**：将生成的 C 与 harness 编成 riscv64 可执行，用 `qemu-riscv64` 运行。harness 含上面 6 步 `ggml_ref` + 负控：NC-A(pass-1 用 f32 累加器)、NC-B(pass-2 用 f32 累加器)——长 mixed-magnitude swamp 行上必须 differ（这俩是 crux）；NC-C(pass-2 用 `(double)v*(double)v` 代替 `(double)(v*v)`)——spike 行上 differ（bonus 严谨）。`memcmp` 逐 bit，打印 `RESULT: PASS`/`proof ok`。
4. **PR 说明**要点：typed facts 在哪（ABI roles 在 fixture 的 `runtime_abi_value`、policy/SEW/LMUL 在 setvl/with_vl）；route-from-facts（op-identity marker `isGgmlNormF32Body` 驱动 dispatch，非 route-id/metadata）；关键 intrinsics（`vle32/vfmul_vf/vse32` + scalar `sqrtf`）；声明无 forbidden anti-patterns（无 dtype-prefixed helper、无 source-front-door 正路、无 route-id/metadata 语义、无 common-EmitC 硬编码 dtype/SEW/LMUL；reduce 为结构化 emitc 节点非 raw C blob）。

**目标 IR 形状(草图)**
> 目标形状：在你新增该 op 之前**不可解析**；精确语法照搬 add 范例 + 兄弟 fixture `rvv-to-emitc-ggml-rms-norm-f32.mlir`。这里不声称它能 parse。
```mlir
tcrv.exec.kernel @ggml_norm_f32_kernel {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
  tcrv.exec.variant @ggml_norm_f32 attributes {origin = "rvv-plugin", requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
    %n   = tcrv_rvv.runtime_abi_value {c_name="ne00", c_type="size_t", role="runtime-element-count", ...} : index
    %x   = tcrv_rvv.runtime_abi_value {c_name="x",   c_type="const float *", role="lhs-input-buffer", ...} : !tcrv_rvv.runtime_abi_value
    %y   = tcrv_rvv.runtime_abi_value {c_name="y",   c_type="float *",       role="output-buffer", ...}    : !tcrv_rvv.runtime_abi_value
    %eps = tcrv_rvv.runtime_abi_value {c_name="eps", c_type="float",         role="dequant-scale-value", ...} : !tcrv_rvv.runtime_abi_value
    %vl  = tcrv_rvv.setvl %n {lmul="m1", sew=32:i64, policy=#tcrv_rvv.policy<tail=agnostic, mask=agnostic>} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul="m1", sew=32:i64, selected_variant=@ggml_norm_f32,
        status="selected-lowering-boundary", policy=#tcrv_rvv.policy<tail=agnostic, mask=agnostic>, ...} {
      %nrm = tcrv_rvv.ggml_norm_f32 %x, %y, %eps, %n, %vl {kind="ggml_norm_f32", strip_lmul="m8"}
           : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl
           -> !tcrv_rvv.vector<f32, "m1">
    } : !tcrv_rvv.vl
  }
}
```
ABI 声明顺序 `(n, x, y, eps)` ⇒ harness extern `void tcrv_emitc_ggml_norm_f32_kernel_ggml_norm_f32(size_t ne00, const float *x, float *y, float eps)`。
