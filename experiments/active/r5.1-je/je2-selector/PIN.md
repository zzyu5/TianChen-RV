# JE2 · 选择器改判 — VLEN 翻转自证 PIN

verdict = **GREEN**(找到并 pin 一个随 VLEN 翻的选择器 θ·emit 级单变量翻转·无板时)

worktree base = `1d207aab58178c1fbcbe8d09ac189532913b182a`(核对 == HEAD)。
weft-opt = `build/weft/bin/weft-opt`(incremental·07-19 07:25)。**纯 emit 级**(vector-source → weft_rvv IR → EmitC)·无板时。

---

## 承重 θ = `integer_core_lmul`(SEL-1「寄存器预算内最宽/最优 LMUL」·F25 路径)

- **f 住址**:`selectIntegerCoreLMUL(march, isaVectorHints)` —
  `lib/Plugin/RVV/FrontDoor/RVVReductionSourceFrontDoor.cpp:343`(sibling:`RVVDequantDotSourceFrontDoor.cpp:375`、`RVVPackedI4DotSourceFrontDoor.cpp:208`·三门同构共用)。
- **机制**(非手写 `vlen<256?"m2":"m1"` switch):
  `deriveMinimumVLEN(march)`(c 唯一真相源·`RVVCapabilityProfile.cpp:289`)
  → `enumerateBlockDotShapeCandidates`(legality:strip VLMAX vs blockLen=32·register budget)
  → `selectGenericSchedule` 静态 argmin(cost 主键·**tie 破 = 更轻 vreg footprint**)。
  - VLEN128:m1 的 e8m1 strip VLMAX=16 < blockLen 32 ⟹ reductions=2 ⟹ cost 高 ⟹ **m2 胜**。
  - VLEN256:m1 的 VLMAX 达 32 ⟹ reductions 掉到 1 ⟹ 与 m2 cost 打平 ⟹ **更轻 vreg footprint tiebreak ⟹ m1 胜**。
- **★与 [GAP-P1] 无关**:GAP-P1 钉死的是 `selectRepackAccumulatorLMUL`(F3/θ1·repack 累加器·恒 mf2)。此处是 **F25 generic block-dot 路径**——正是 GAP-P1 IRON RULE 注释里点名的「the gate4 widest-legal selector for a DIFFERENT kernel path」。**未攻被钉的 repack 核**。

---

## 翻转 PIN(单变量 = `-march`·可复跑)

固定件(全程不变):fixture = `test/Transforms/RVV/rvv-widening-dot-reduce-source-front-door.mlir`;pass = `--weft-rvv-materialize-widening-dot-reduce-source-front-door`。**唯一变量 = pass 的 `march=` 值**。

| | march-A | march-B |
|---|---|---|
| `-march` (c) | `rv64gcv`(VLEN128) | `rv64gcv_zvl256b`(VLEN256) |
| **θ `integer_core_lmul`** | **`m2`** | **`m1`** |
| body `weft_rvv.setvl` lmul | m2, sew=8 | m1, sew=8 |
| `weft_rvv.widening_product` | signed-i8**m2**xi8m2-to-i16**m4** | signed-i8**m1**xi8m1-to-i16**m2** |
| `weft_rvv.standalone_reduce` → | i16**m4** → i32m1 | i16**m2** → i32m1 |
| EmitC 指令(字节级) | `vsetvl_e8m2`/`vle8_v_i8m2`/`vwmul_vv_i16m4`/`vwredsum_vs_i16m4_i32m1` | `vsetvl_e8m1`/`vle8_v_i8m1`/`vwmul_vv_i16m2`/`vwredsum_vs_i16m2_i32m1` |

θ-A = m2 → θ-B = m1。翻转贯穿 IR **和** EmitC intrinsic(字节级)·非仅 metadata mirror(stamp `weft_rvv.gearbox_selected_integer_core_lmul` 是 audit mirror·body 结构真消费 θ)。

### 提取命令(复跑)
```sh
OPT=build/weft/bin/weft-opt
FIX=test/Transforms/RVV/rvv-widening-dot-reduce-source-front-door.mlir
# θ + body:
$OPT $FIX --weft-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv \
  | grep -E 'gearbox_selected_integer_core_lmul|weft_rvv.setvl|widening_product|standalone_reduce'
$OPT $FIX --weft-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv_zvl256b \
  | grep -E 'gearbox_selected_integer_core_lmul|weft_rvv.setvl|widening_product|standalone_reduce'
# EmitC 字节级:
$OPT $FIX --weft-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv        --weft-rvv-lower-to-emitc | grep -oE '__riscv_v(setvl_e8|le8_v_i8|wmul_vv_i16|wredsum_vs_i16)m[124]'
$OPT $FIX --weft-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv_zvl256b --weft-rvv-lower-to-emitc | grep -oE '__riscv_v(setvl_e8|le8_v_i8|wmul_vv_i16|wredsum_vs_i16)m[124]'
```

---

## 稳健性(非两点巧合)

VLEN 阈值扫(同 fixture/pass·只变 march):

| march | VLEN | θ |
|---|---|---|
| `rv64gcv` | 128 | m2 |
| `rv64gcv_zvl128b` | 128 | m2 |
| `rv64gcv_zvl256b` | 256 | **m1** |
| `rv64gcv_zvl512b` | 512 | m1 |

= 单调阈值·crossover 在 256(与机制预测一致:m1 VLMAX 在 256 达 32 才追平 blockLen)。非 2-point 巧合。

第二门复现(shared `selectIntegerCoreLMUL`):`rvv-widening-dot-reduce-dequantize-source-front-door.mlir`
+ `--weft-rvv-materialize-widening-dot-reduce-dequantize-source-front-door` → march=rv64gcv:θ=m2/body m2;march=rv64gcv_zvl256b:θ=m1/body m1(同翻)。

---

## 纪律核对
- byte-exact:不适用(无 perf·纯 emit 翻转自证)。翻转 pinned(march-A/θ-A→march-B/θ-B + 提取命令)✓。
- **只减不增·零源改**:未加任何代码·未新增焊死 g/θ·未新造 θ。source_diff = NONE。
- **共享文件**:未改。碰的仅只读源 + 复跑既有 lit fixture。
- 工件仅此目录·**不 git commit**(主会话集成)。
