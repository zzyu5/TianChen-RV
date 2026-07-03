# 接线成本地图 — M-FLAT 先接线(computed-scale 强路)

**任务:** 07-03-m-flat-wire-first-strong-route · **base:** refactor/full-refactor-m1
**产出于:** 一次真实接线尝试的侦察(用户裁决:"用一次最小真实接线去打 = 最好的穿透性探测")

## 头条结论(读我)

**里程碑目标 = 让 flat block-dot 家族(q8_0…)六态格翻 constructed-strong;这只能靠替换循环体 `emitFlatBlockDot` 达成 —— 任何单块 route 都到不了这个目标。** 因此 PRD 的**首选目标(单块 computed-scale 兄弟)无法达成它自己唯一声明的目标(C_construct 0→1)**。这比"#2 不可满足"更强:不是"删不掉行",而是**单块这条路本身指错了方向**。

细节:六态里 C_construct 只数 `constructed`+。仓里**已有 3 条** `constructed`(`product_reduce` q4_0_nibble / offset_binary_n3 / codebook_n3,front-door 分解的**合成** route)—— roster 层 C_construct **已是 3,不是 0**。里程碑说的"0/24"数的是 **flat vec_dot 家族**(q8_0 等仍 `constructed-weak`,anchor=`emitFlatBlockDot`)。加一条**新单块合成强路**只会添第 4 条合成 `constructed` 行,**不翻任何 flat 家族格**、**不删任何弱体** → 里程碑意义上 **0→0**。

**非结构死墙:** 管线架构**允许**新 route(工作量墙全可穿,见 §二)。这是**砖④负结果与本任务赢条件/里程碑目标之间的范围矛盾**,需 parent 裁决。

---

## 一、两处硬阻断(实测,primary-source)

### 阻断 A — Δ手写LOC<0 需要删循环体,而循环体 = 砖④/⑤(砖④已证砖组不成)

赢条件 #2 要求删掉"对应手写 opaque body"使净手写 body 行为负。全仓 `(float)*(const _Float16 *)` computed-scale 重建 + `(float)sumi*scale` fold 的手写 opaque body **全部**位于**带 nb 块循环**的活体里,服务活格式:

| 手写 opaque body(含 `_Float16` / fold) | 位置 | 结构 | 服务格式 |
|---|---|---|---|
| `emitFlatBlockDot` | RVVToEmitCBlockQuantLinear.cpp:5334 | nb 块循环 + 标量 `int32_t sumi` + `emitFold: (float)sumi*scaleProduct`, `sumf+term` | q8_0 / q4_0 / q4_1 / q5_0 / q5_1(共享弱体) |
| `emitQ4_0Q8_0GemmTile` | :140(`_Float16`:187) | blockLoop :258 | q4_0 GEMM tile |
| `emitQ4_0Q8_0Gemm` | :418(`_Float16`:479) | blockLoop :592 | q4_0 GEMM |
| `emitRepackGemmQ4_0Q8_0` | :828(`_Float16`:888) | blockLoop :1108 | q4_0 repack GEMM |
| `emitRepackGemvQ4_0Q8_0` | :1513(`_Float16`:1570) | blockLoop :1723 | q4_0 repack GEVM |
| `emitRepackGemvQ5_0Q8_0` | :1976(`_Float16`:2030) | blockLoop :2244 | q5_0 repack GEVM |
| `emitRepackGemvQ8_0Q8_0` | :2477(`_Float16`:2534) | blockLoop :2684 | q8_0 repack GEVM |

**每一处 opaque fp16-scale/fold body 都在 nb 块循环内、服务活格式。** 删任何一处 = 把一个循环格式重路由到新 route = **砖⑤级**工作,需要 nb 块循环 typed body —— 而**砖④已证明砖①②③ 组不成 nb 循环**(砖① verifier 只收导入 base+常量偏移,读不了 `base+ib*stride`;`../07-03-m-flat-brick4-block-loop/prd.md`)。

单块目标 ⇒ 只能组单块 body ⇒ 不对应任何可删的循环手写体 ⇒ **净只能是加法(additive),Δ手写LOC>0,#2 失败。**

**证据:** 砖①②③ 三次提交(23852f56 / bdad477f / d63548bb)对 `RVVToEmitCBlockQuantLinear.cpp` 是 **+116 insertions / 0 deletions**(纯加法);它们**没有**从 `emitFlatBlockDot` 删任何 inline opaque 行 —— 桌上没有"剩下的可删行"。

### 阻断 B(佐证,非独立必需)— 砖是标量,首选目标(vector 强路)是向量,砖接不上

PRD 首选目标 = `bounded_widening_dot_reduce_dequantize_source` 的 computed-scale 兄弟。但该既有强路是**向量**体:
`standalone_reduce → i32m1 向量(lane0 是标量边界)→ 向量 dequantize(导入 f32 scale)→ 向量 store`(`RVVDequantDotSourceFrontDoor.cpp`;`StandaloneReduceOp` ODS RVVOps.td:3403 "produces one typed !tcrv_rvv.vector token whose lane 0 is the scalar output boundary")。

砖是**标量**:
- 砖② `BlockComputedScaleDequantOp::verify` 硬要 `getSumi().getType().isInteger(32)`(标量 i32)。
- 砖③ 三口都要标量 f32。

向量 `standalone_reduce` 的 lane0-i32m1 **不是**标量 i32,砖② 收不了。**据推断**方言里没有 typed 标量-i32-extract op 做桥(已看 compute-role 助记表 :317-333,未逐 op-def grep;若走 option-1 需正式确认)。既有向量强路本身**已经零 opaque**(那里没东西可删)。**阻断 A 单独已足以 kill #2**;B 仅佐证"即便补了桥,A 仍成立"。

砖①②③ 真正对应的是 `emitFlatBlockDot` 的**标量 per-block fold**(`sumi` 是标量 `int32_t`,`emitFold` 做 `(float)sumi*scaleProduct`+`sumf+term`)—— 即**循环标量 flat-block-dot**,又落回阻断 A(需 nb 循环)。

**净结论:** 砖①②③ 现在**唯一**能诚实组成的 body = **循环标量 flat-block-dot** = 砖④/⑤,砖④已证不可。它们**不能**诚实组成"单块 vector 强路的 computed-scale 兄弟"(类别错配 + 缺 extract 桥)。PRD"首选目标"的前提被证伪。

---

## 二、接线成本地图(墙的位置+改法+代价)—— 若 parent 裁"先加 route、后删"

以下是接线一条**新单块强路**要过的墙(纯加法路径,C_construct 可动但 #2 不达)。**每堵都是工作量墙,非死墙。**

| 墙 | 位置 | 改法 | 代价 |
|---|---|---|---|
| `.size()` 门 | `RVVConstructionProtocol.cpp:934`(`!= 67`)| 加 1 条 route → bump `67→68` | 1 行(运行时门,lit 触发)|
| 保留强路表 | 同文件 `kRetainedSelectedBodySpecializations[]`:357(67 条)| append 1 条 `{op-kind, mnemonic, role, emitc-route, callable-abi.v1, callable-abi}` | ~6 行 |
| typed-role 列表 | 同文件 :317-333 compute role | **无改**:widening_product/standalone_reduce/dequantize 已在 compute role;砖①②③ 需**新增**进 allowlist(若走 typed-role 校验)| 砖 op 助记入表,~3 行 |
| op-kind 枚举 | `RVVEmitCRouteProvider.h:25` `RVVSelectedBodyOperationKind`(~87 值)| 加 1 值(如 `WideningProductReduceComputedScaleF32`)| 1 值 + **~10 switch 站点**(PlanOwners/Common/Metadata/RoutePlanning;无 `-Werror`,warn 非 error,但该处理的处理)|
| route-identity | `RVVContractionRouteIdentity.cpp:33 contractionRouteRegistry()`(~8 条)| 加 1 条 `ContractionRouteIdentity` | ~30 行(仿 :300 dequantize 条)|
| pre-realized body op | 新 `TypedWideningProductReduceComputedScale...PreRealizedBodyOp`(ODS + attrs)| 新 op(仿 `TypedWideningProductReduceDequantizePreRealizedBodyOp`)| ODS ~60 行 + realizer |
| validator 真实调用点 | `RVVEmitCContractionRouteFamilyPreRealizedValidators.cpp:937` 仿 | **新 allowlist 校验**(注意:`rejectMixedPreRealizedContractionBody` 是 **blocklist**;把砖当模板参会**拒**砖体,要反过来写 allowlist:每个 body op ∈ {砖①②③+widening/reduce/load/store/setvl/withvl})| ~120 行(仿既有 validate 函数)|
| 新 front-door | 新 `RVVComputedScaleDotSourceFrontDoor.cpp`(仿 `RVVDequantDotSourceFrontDoor.cpp` 1170 行)| matcher + LMUL 选择 + body builder(用砖 op)+ pass 注册 | **~1000+ 行纯新增(additive)** |
| 六态行 | `schema/coverage-sixstate.v1.json` | 加 1 `constructed` 行(如既有 3 条 `product_reduce` synthetic)| 数据行 |
| emit-consistency lit | `test/Conversion/RVV/` | 新 golden(CORE==emission-plans)| lit |

**关键墙险度排序:** 最险 = **validator 语义反转**(blocklist→allowlist,写错会静默漏检非 typed 体,破 [L-8] 强义);次险 = **缺 typed 标量-i32-extract 桥**(阻断 B,现在就无法组体);第三 = front-door ~1000 行 additive(工作量,决定 #2 必失)。

---

## 三、六态 / C_construct 说明(避免虚荣)

- 六态 meta:`'constructed' == STRONG`;C_construct 只数 `constructed`+。既有 3 条 `constructed` 是 `product_reduce`(q4_0_nibble / offset_binary_n3 / codebook_n3)—— front-door 分解的**合成** route,非 ggml roster 格。里程碑"C_construct 0/24"数的是 **flat block-dot vec_dot 家族**(q8_0 等仍 `constructed-weak`,anchor=`emitFlatBlockDot`)。
- 加一条**新单块合成强路**会给六态添第 4 条 `constructed` 行(同既有 3 条口径),**技术上** C_construct 可动 —— 但那**不是** flat 家族的 q8_0 翻强(那需砖⑤/循环),且**不删任何弱体**,所以里程碑意义上 q8_0 六态行**不翻**、Δ手写LOC**不负**。

---

## 四、真正的下一步 = 补砖①的 per-block-source 契约(具体设计岔口)

flat 家族翻强的**唯一**通路是替换循环体 `emitFlatBlockDot`,那需要一个 typed nb 循环 body。砖④证它现在做不到,**根因具体**:

- 砖① `BlockFp16ScaleProductOp::verify`(RVVDialectWideningOps.cpp:9750-9765)硬要 `lhs_scale_base` / `rhs_scale_base` 为 `RuntimeABIValueType` **且** role ∈ {`LHSInputBuffer`}/{`RHSInputBuffer`} —— 即**导入的 ABI 基址指针**。它读不了 per-iteration 的 `base + ib*stride` 块内 scale。
- ⇒ 循环里每块的 d_x/d_y 无法用砖① typed 表达 ⇒ 无 typed nb 循环 body ⇒ flat 家族翻强被卡在这个**原语契约缺口**,不是"接线工作量"。

**里程碑真正的下一步不是接一条单块 route,而是把砖① 的 source 契约扩成 loop-capable per-block-source**(接受 base + 循环归纳变量派生的偏移),再配 loop-aware validator(砖④范围)。**这是要 fund 或绕开的确切设计缺口。**

## 五、裁决请求(给 parent)

砖④负结果(砖组不成 nb 循环)⊥ 本任务赢条件 #2 / 里程碑目标(flat 家族翻强),因为**唯一可删的 opaque computed-scale body 全是循环体**、**且单块 route 到不了 flat 家族格**。请裁:

1. **[补原语]** 承认真正下一步 = 扩砖①的 per-block-source 契约(§四)→ 再做 loop-aware body/validator(砖④重启)→ 才能替 `emitFlatBlockDot` 翻 flat 家族。**推荐。**
2. **[重估里程碑]** flat 家族翻强 = 循环体翻强,若 per-block-source 原语判为多会话/真设计分叉,里程碑改道或降级。
3. **[加法探针]** 若只想要"穿透性证明",可加一条单块合成强路(添第 4 条 `constructed` 合成行)—— 但需先正式确认/补标量-extract 桥(阻断 B),且**不动里程碑目标、不删弱体、Δ手写LOC 不负**;novelty 稀薄(不对应 ggml 活格式)。仅当明确要它时做。

**不制造假删除去凑 #2 的负数。等裁决。**
