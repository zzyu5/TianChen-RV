# JE1 · 管道通 — strip-width θ pinned flip + march-parse 收敛 scope

worktree base pin = `1d207aab58178c1fbcbe8d09ac189532913b182a`
binary = `build/weft/bin/weft-opt` (@07:25; `deriveMinimumVLEN`/`deriveRVVVersion`
== HEAD byte-identical, `git diff HEAD` empty → flip evidence pinned-valid).
NO source edit by JE1. Emit-level only, no board time.

承重 θ = `deriveRepackHalfLanes(vlenBits, weightInterleave)` =
`min(vlenBits/16, weightInterleave)`, vlenBits<128 → 0.
`vlenBits = deriveMinimumVLEN(march, isaVectorHints)`. 单变量 = `-march`.

═══════════════════════════════════════════════════════════════════════
## 判决A — strip 宽 θ 跟能力文件走 = GREEN (翻转自证·零接线)
═══════════════════════════════════════════════════════════════════════

### A1 · 前门路 (RVVLowerQuantContraction lowerToRepackGemv 构造时盖章)
fixture = `test/Conversion/RVV/rvv-lower-quant-contraction-q5-1-vlen256-decode-measured-repack.mlir`
(q5_1 decode — repack 在两 march 下都保留，故算法轴被夹住，唯一变量 = march)

| pin | 命令 | half_lanes | 其余(全held) |
|---|---|---|---|
| march-A | `weft-opt <fix> --weft-rvv-lower-quant-contraction=march=rv64gcv` | **8** | algo=repack, core=mf2, interleave=16 |
| march-B | `weft-opt <fix> --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b` | **16** | algo=repack, core=mf2, interleave=16 |

θ 流到 emit-C (非死属性)——同命令加 `--weft-rvv-lower-to-emitc`：
- march-A VLEN128: 4×`vfmv_v_f_f32m2` + 4×`vse32_v_f32m2` + 8×`vwmacc_vx_i16m1` (numHalves=2·两8车道半)
- march-B VLEN256: 2×`vfmv_v_f_f32m2` + 2×`vse32_v_f32m2` + 4×`vwmacc_vx_i16m1` (numHalves=1·一16车道strip)
- byte diff = 290 行不同 ⟹ DIVERGENT.

### A2 · Schedule 路 (deriveRepackHalfLanes 调用点 :176-177 q8_0 GEVM arm 直隔离)
fixture = `experiments/active/r5.1-je/je1-pipe/fixture-q8_0-gemv-monolithic.mlir`
(单 module·1 个 monolithic `weft_rvv.repack_gemv_q8_0_q8_0`·authored half_lanes=8)
命令 = `weft-opt <fix> --weft-rvv-materialize-repack-strip-width=march=<M>`

| march-M | deriveMinimumVLEN | half_lanes | integer_core_lmul | 覆盖路径 |
|---|---|---|---|---|
| `rv64gcv` (VLEN128) | 128 | **8** | (unset) | :176-179 min(128/16,16)=8 |
| `rv64gcv_zvl256b` (VLEN256) | 256 | **16** | (unset) | :176-179 min(256/16,16)=16 |
| `rv64gc_xtheadvector` (RVV0.7) | 128 | **16** | **"m1"** | :171-175 whole-LMUL 强钉 override |
| `rv64imac_zve32x` (无≥128 floor) | 0 | **8**(intact) | (unset) | :118 vlenBits<128 return·authored 不动 |

⟹ 四挡全按 f(g,c) 预言。`deriveRepackHalfLanes` 承重、翻转 pinned、单变量 march、零源改。

═══════════════════════════════════════════════════════════════════════
## 判决B — march 解析点收敛 scope (PARTIAL·本波仅盘点·不重构)
═══════════════════════════════════════════════════════════════════════

收敛目标 = 探测层一处 = `lib/Plugin/RVV/RVVCapabilityProfile.cpp` 的 `derive*` 权威族
(所有 ISA-token 字符串切分应仅住此)。

### B1 · 权威族 (KEEP·原始 token 切分应仅在此·file:line → token → c 值)
| fn | 行 | token 谓词 | 出 c 值 |
|---|---|---|---|
| `hasRVVVectorHint` | :140-155 (:143-145) | zve/zvl/zvfh/gcv/xtheadvector/rv64… | bool 有向量 |
| `deriveSupportedSEWAllowList` | :222-243 (:233-234) | zve64/gcv/rv64gcv/xtheadvector | SEW 集 |
| `deriveSupportedLMULAllowList` | :260-277 (:262) | (转调 deriveRVVVersion) | LMUL 集 |
| `deriveMinimumVLEN` | :289-332 (:300/314/323/325) | zvl{N}b / gcv/_v/rv64v/xtheadvector | **VLEN bits** |
| `deriveHasZvl128b` | :334-339 (:338) | (转调 deriveMinimumVLEN≥128) | bool |
| `deriveRVVVersion` | :341-375 (:362/368) | xtheadvector/0p7 → RVV0p7; gcv/zve/rv64v → RVV1p0 | 版本 |

### B2 · 权威消费者 (合规·funnel 过权威·不切 token·但各自持 march option/field)
每处 = 线程 `march`+`isaVectorHints` 并调 derive*，非 token 违规。深层收敛(march 只
解析一次→materialize 到 IR→下游读 IR fact) = 更大重构，本波不做，标为"threading 轴"另 fork。

| # | file:line | 调用 | 出 c 值 |
|---|---|---|---|
| C1 | `Schedule/RVVProbedCapabilityAxesMaterialization.cpp:101,109` | deriveSupportedLMULAllowList / deriveRVVVersion | supported_lmul/sew/rvv_version → **物化到 IR provider op**(=天然单解析地) |
| C2 | `Schedule/RVVRepackStripWidthMaterialization.cpp:100,112` | deriveMinimumVLEN / deriveRVVVersion | VLEN→half_lanes, 版本→m1 (判决A pass) |
| C3 | `Schedule/RVVScheduleDescriptorRegistry.cpp:457,458` | deriveMinimumVLEN / deriveHasZvl128b | VLEN, hasZvl128b |
| C4 | `FrontDoor/RVVLowerQuantContraction.cpp:1407,1544,1563` | deriveMinimumVLEN×2 / deriveRVVVersion | VLEN, 版本 |
| C5 | `FrontDoor/RVVDequantDotSourceFrontDoor.cpp:376` | deriveMinimumVLEN | VLEN |
| C6 | `FrontDoor/RVVReductionSourceFrontDoor.cpp:345` | deriveMinimumVLEN | VLEN |
| C7 | `FrontDoor/RVVPackedI4DotSourceFrontDoor.cpp:209` | deriveMinimumVLEN | VLEN |
| C8 | `FrontDoor/RVVCodebookDotSourceFrontDoor.cpp:210` | deriveMinimumVLEN | VLEN |
| C9 | `FrontDoor/RVVMonolithicBlockDotSourceFrontDoor.cpp:3395` | deriveMinimumVLEN | VLEN |
| C10 | `RVVCapabilityProfile.cpp:508,519` | deriveSupportedLMULAllowList / deriveRVVVersion | (内部·建 provider facts) |

### B3 · 真正的 token 泄漏 (探测层外·收敛/只减候选)
| leak | file:line | 现状 | 收敛动作(未来波) |
|---|---|---|---|
| L1 | `EmitC/RVVEmitCRoutePlanning.cpp:104-108` `containsRVVVectorISAHint` | 独立重切 rv64gcv/rv32gcv/zve/zvl/zvfh/rvv (called :148 校验 isa_vector_hints property) | 转调权威 `hasRVVVectorHint`(:140)·删本地 token 列表 (需 header 导出 hasRVVVectorHint) |
| L2 | `IME/IMEExtensionPlugin.cpp:237,323` `kIMEMarchToken("xsmtvdotii")` + `march.contains(...)` | IME 门·跨插件 | ★见下·**sibling 在飞** |

### B4 · 假阳性 (task 列名但非 capability-token 解析·排除)
- `Target/RVV/RVVTargetSupportBundle.cpp` — march 仅作 **toolchain 编译 arg**(`-march=rv64gcv`/`_zvfh` 喂 clang @:1808/1819)·不切 token 推能力。:480-493 的 `.contains` 是 **attr-key 过滤**(element_count/descriptor…)·与 march 无关。→ 排除出 token 收敛。
- `IME/IMEBackendEmissionDriver.cpp:2164` `hasIME` = walk IR 找 IME dialect op·非 march 解析。

### B5 · ★并发冲突警报 (供主会话合并)
共享权威头 `include/Weft/Plugin/RVV/RVVCapabilityProfile.h` 在本会话期间被**另一并行 track
staged 修改**(会话开始时 tracked 全 clean)——**非 JE1 所改**。该 diff 删除：
- 孤儿 `deriveIMEPresent(march,hints)` (header-inline·census §5 zero-caller·实测 0 callers)
- zero-consumer 字段 `cachelineBytes` / `imePresent`

**这正是本 B3-L2 / 只减方向的一部分**(删 xsmtvdotii 的第二份 orphan 拷贝)——sibling 已在做·
**JE1 不重复认领**。deletion 自洽(无 .cpp dangling ref)。对判决A **零影响**(deriveMinimumVLEN/
deriveRVVVersion 未动)。主会话合并时注意此文件有 sibling 的 staged 删除。
删 deriveIMEPresent 后·xsmtvdotii 的**唯一**权威回落到 IMEExtensionPlugin:237 kIMEMarchToken
(IME 插件本地·跨插件·合理)——L2 收敛因此坍缩为"已只剩一处"(sibling 落地后)。
