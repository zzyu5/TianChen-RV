# G2 [FUSE] rms_norm→mul — 设计评审包 (design review pack)

> **贯通 tracer cell. STOP-at-贯通 —— 铺量待用户批 (see §9).**
> 数据格,零 perf 主张。本包全部数字是 **emit-level 字节会计**(结构消除计数),
> **不是** perf beat —— 真 DRAM 带宽收益 **待板 (pending-hardware)**,[NG-4] 未过 [PERF-1] 八门前
> 不出现任何 "faster / beat / 更快" 措辞。

融合的算子对 = llama 的 `attn_norm` / `ffn_norm`:一次 `ggml_rms_norm` 紧跟一次
`ggml_mul`(对一条学习到的权重向量 w[])。融合把两趟内核压成一趟,消掉两者之间的
中间归一化行 y[] 的一存一读。

---

## 1. 机制 (L1 链式/epilogue 声明 → L2 region 拼接)

**L1 — 链式/epilogue 声明** (ODS `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`):
- producer `tcrv_rvv.elementwise_rms_norm_reduce_core` 带一个**可选单块 `$epilogue` region**
  (`AnyRegion` + `NoTerminator`;0 块 = 平凡 rms_norm,1 块 = 融合)。region 的**唯一 entry 参数
  = per-strip 归一化向量 `vy`**(`!tcrv_rvv.vector<f32,"m8">`,typed-body 层无真 SSA、只在生成的
  C 里实体化为寄存器)。
- 新 consumer brick `tcrv_rvv.elementwise_mul_map`(`z[i] = vy[i] * w[i]`):operand = `chain`
  (= epilogue region arg0)、`weight`(w[] `const float *`)、`output`(z[] `float *`)、`n`、
  `strip_index`;无 result(store 即 sink)。加入 `RVVToEmitC.cpp` 的 [L-8] union allowlist
  (严义门递归认证)。

**L2 — region 拼接** (emitter `lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp`,
`emitElementwiseRmsNormReduceStrip`,fused 分支 :392–481):在 normalize strip 的 `for` 内、
`vfmul_vf`(算 `vy`)↔ store 之间 splice epilogue —— `vw = vle32(w+i)` →
`vz = vfmul_vv(vy, vw)` → `vse32(z+i, vz)`。**producer 的 norm store 被跳过,consumer 的 reload
从不生成。** 融合被提前探测(mulMap walk),故非融合路径的 y[] store 指针仍在原位物化 ——
保证 UNFUSED 的 op 顺序恒等(byte-exact,§6)。

## 2. anti-bypass 不变量 (verifier,`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp`)

四条,证明吃的是寄存器 `vy` 而非内存 reload、且落点唯一:
1. **chain == epilogue region arg0** —— mul 的第一操作数必须是 producer 交出的寄存器 vy,不是
   任何内存值。
2. **strip_index == 外层 loop 归纳变量(region arg0)** —— mul 与 normalize 同一趟 strip,不能外提。
3. **chain LMUL == producer `strip_lmul`("m8")** —— 消费者向量宽度锁死生产者归一化宽度。
4. **mul output == producer output** —— 单一融合目的地 z[](不允许旁路到别的 buffer)。

三条 fail-closed 负例(lit,§10)钉死其中的可机检子集:`BADMULKIND`(kind 边界)、
`BADCHAINLMUL`(LMUL tie)、`BADOUTPUT`(output C-type `float *`)。

---

## 3. 字节流量会计 —— 中间张量消除 (★ byte accounting, NOT beat)

中间张量 = 归一化行 `y[]`(n 个 f32,n = `ne00` = 行长 = n_embd)。融合把它整个消掉。

### 3a. 中间张量 y[] 自身流量 (纯消除项)

| y[] 流量 | UNFUSED(两内核) | FUSED | 消除 |
|---|---:|---:|---:|
| y[] store(rms_norm 写) | `4·n` B | 0 | `4·n` B |
| y[] reload(mul 读) | `4·n` B | 0 | `4·n` B |
| **中间张量小计** | **`8·n` B** | **0 B** | **`8·n` B (100%)** |

外加:y[] scratch 缓冲的 `4·n` B **内存占用本身**从不物化(footprint 收益,非带宽)。

### 3b. normalize+mul 阶段总 DRAM 流量(不含两路共享的 Σx² 归约读 x)

| 趟 | UNFUSED | FUSED |
|---|---:|---:|
| read x | `4·n` | `4·n` |
| write y(中间) | `4·n` | — |
| read y(中间) | `4·n` | — |
| read w | `4·n` | `4·n` |
| write z | `4·n` | `4·n` |
| **阶段合计** | **`20·n` B** | **`12·n` B** |
| 消除 | — | **`8·n` B = 阶段的 40%** |

### 3c. 全内核视角(含两路都在的 Σx² 归约读 x)

| | UNFUSED | FUSED |
|---|---:|---:|
| 合计 | `24·n` B | `16·n` B |
| 消除 | — | **`8·n` B = 全内核的 33%** |

**三种口径消除量恒 = `8·n` B = y[] 的一存一读**,即中间张量往返被整体抹掉。

### 3d. 说明性投影 (illustrative — 结构计数的算术外推,非实测,[NG-4] 待板)

- n_embd = 4096 ⇒ 每次 rms_norm→mul、每行(token)消除中间往返 `8·4096` = **32,768 B = 32 KiB**。
- llama 族:每层 `attn_norm` + `ffn_norm` = 2 个融合 norm,另加 1 个 final norm。32 层模型
  ⇒ 65 次调用/token ⇒ 65 × 32 KiB ≈ **2.03 MiB/token 的中间 DRAM 往返被抹掉**(纯算术外推 emit-level
  计数;真带宽收益待板)。

## 4. emit-level 中间 store/load 消除计数 (empirical, 只读复算)

`build/bin/tcrv-opt --tcrv-rvv-lower-to-emitc`,`call_opaque` only(见 `emit_census.txt`):

| 每 strip | UNFUSED 阶段(2 内核) | FUSED(1 内核) | Δ |
|---|---:|---:|---:|
| `vle32`(load) | 3 (x, y, w) | 2 (x, w) | **−1(y reload)** |
| `vse32`(store) | 2 (y, z) | 1 (z) | **−1(y store)** |

实测(call_opaque-only):FUSED 内核 strip = `vle32×2, vse32×1, vfmul_vf×1, vfmul_vv×1`;
plain rms_norm 生产者 strip(UNFUSED 内核 1)= `vle32×1(x), vse32×1(y)`;UNFUSED 内核 2(独立
TU 的 mul)= `vle32×2(y,w), vse32×1(z)`。融合把 **y store(vse32)+ y reload(vle32)** 两个内存
op-class 从关键路径拿掉。

---

## 5. byte-exact (strict) 证

- **UNFUSED 零回归**:编辑前捕获 golden,重建后逐字节相同。标量-double `Σx²` 升序 fold +
  mean + `1/sqrtf(mean+eps)` scale 全未动;融合探测提前做、非融合 y[] store 指针仍在原位物化,
  故 op 顺序恒等。
- **FUSED 数值恒等**:寄存器驻留 `vy` 与「store 后 reload 同一 f32 向量」逐比特相同;`vy*w[i]`
  是裸 per-lane f32 `vfmul_vv`(**无 FMA** —— 后无 add;无归约,任意 LMUL byte-exact),与非融合
  两算子 `mul∘mul` 的 f32 舍入顺序一致。融合值 = 非融合值,**除消失的 store/reload 外可执行
  数值恒等**。硅上 bit-exact-vs-ggml 仍 pending-hardware。

## 6. NG-2 合规

[NG-2] = 不做图级/端到端框架,输入侧止于 kernel 级接口。本融合以**生产者算子自身 IR 内携带的
可选 epilogue region**(链式声明,L1)表达 —— **没有**任何图 pass 去"发现"两个独立声明的算子再
改写它们;生产者字面上把消费者作为链式 epilogue 抱在自己身上,emitter 只做 region 拼接(L2)。
输入侧仍是单个 kernel 级 typed loop body,无 ggml 计算图、无跨算子调度框架。→ **NG-2 合规**
(与科研目标总纲 [FUSE] framing 一致:"epilogue / 链式声明(kernel 级接口、非图框架 → [NG-2]
合规)")。

## 7. 可泛化性 (哪些融合对适用 [FMT-PROP] / epilogue)

机制泛化到任何满足下述条件的 producer→consumer 对:(a) 消费者是生产者 per-strip 输出向量的
**纯 per-lane elementwise map**(无跨 strip / 跨 lane 依赖);(b) 生产者本就 strip-by-strip 在
寄存器里交出输出向量。候选:

- **rms_norm→mul**(本格 DONE)—— llama attn_norm / ffn_norm。
- **scale/silu→mul(SwiGLU 门)** —— 激活 epilogue;consumer brick 换成 silu-map / mul-map 链,
  同一 `$epilogue` region、逐 lane byte-exact。
- **[FMT-PROP] 激活侧格式传播** —— 中间激活的**存储格式**变成编译器决策(生产者-消费者格式协商
  = schema 谓词匹配),消灭独立激活量化趟。同战役,比纯消除更深:它还**选格式**。
- **bias-add / residual-add epilogue** —— 任意 elementwise 生产者后挂加法。

anti-bypass 四不变量(§2)= 每加一对时保持"寄存器直喂 + byte-exact"的复用门。

**不适用(需另设计,非 free):** 生产者与消费者之间有**跨 lane 归约**的对(如 softmax 的
normalize 必须先拿到整行 reduce)→ 需两阶段融合;**引入 FMA 的对**(fused multiply-ADD 改变舍入)
→ 需重新推导 byte-exactness,不白得。

## 8. 验证态

- 全量 build 链接通过;`tianchenrv-rvv-dialect-test` 通过;**484/484** RVV lit 全绿(build agent 报告)。
- 本格**只读复算**(不重建,用现有 `build/bin/tcrv-opt` + `FileCheck`,见 §10):tracer lit 主
  CHECK + 三负例(`BADMULKIND` / `BADCHAINLMUL` / `BADOUTPUT`)**全 PASS**;融合 strip 实测
  `vle32×2 / vse32×1 / vfmul_vv(vy,·)`,与字节会计一致。

## 9. STOP-at-贯通 (★ 明标)

**只做 rms_norm→mul 这一对。贯通 = 机制建成、byte-exact、中间张量消失,到此为止。**
- 未 fan-out 到 silu/scale/其它对;未做 [FMT-PROP] 格式协商;未铺量、未泛化。
- **铺量 / [FMT-PROP] / 其它算子对 = 立项决策,归用户。** 本会话不越界。

## 10. 复算 (read-only,零重建、零 git 写)

```
B=build/bin
T=test/Conversion/RVV/rvv-to-emitc-typed-elementwise-rms-norm-mul-fused-epilogue-loop-body.mlir
# 主 CHECK(融合数据流 + 单 store)
$B/tcrv-opt $T --tcrv-rvv-lower-to-emitc | $B/FileCheck $T
# 三负例 fail-closed(BADMULKIND / BADCHAINLMUL / BADOUTPUT) — 见 lit 头 RUN 行
# 融合 strip 内存 op census(call_opaque only)
$B/tcrv-opt $T --tcrv-rvv-lower-to-emitc | grep -c 'call_opaque "__riscv_vse32_v_f32m8"'   # = 1
```

## 11. 纪律

数据格。未 commit、未 git add/stash/rm/mv。touch-set = 本 experiments/ 格
(MANIFEST + NOTES + byte_accounting.csv + emit_census.txt)+ 只读 build 报告;未碰
GridCodebook / repack / docs / 别的 experiments 格。融合代码本身(RVVOps.td /
RVVToEmitCForwardElementwise.cpp / WideningOps / RVVToEmitC.cpp allowlist / lit)由 build agent
在其批准 touch-set 内改,未 commit。
