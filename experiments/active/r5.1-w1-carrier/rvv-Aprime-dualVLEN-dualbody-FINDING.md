# FINDING · rvv A′ 双 VLEN 双 body（D4）：emit+object 级 march-唯一变量 flip 已 pin·★rvv 板 byte-correct 已兑现（W5·2026-07-19）

> **性质**：A′「双 VLEN 双 body」结构主张（W1 载体批 · 裁1 子项④·H-1 handback §E · ISSUE-105 · K-attack-fanout 机制①）。**一次板窗兑现两主张**（B7/E7 + B8/E8）的论文侧最便宜完成时。
> **本 FINDING 的三半**：**(结构半) emit+object 级双 body 分歧 = 已 pin（确定性）**；**★(rvv 板半·B7/E7) body_A VLEN128 on rvv silicon = byte-correct 已兑现（W5·2026-07-19·mism=0×4 seeds）**；**(k1 板半·B8/E8) body_B VLEN256 on k1 = 排期 2026-07-21（k1 板窗·body_B 在 VLEN128 上是 wrong-VLEN falsifier·须其原生 VLEN256）**。
> **落盘 = 带 run-id 原始输出**：结构半 `raw/RAW-rvv-Aprime-emitflip-20260719T093213Z.rawlog.txt`（md5 `b4bf41db96902c71bcbb9953a0f5d684`）；**板半 `raw/RAW-w5-rvvAprime-boardbytecorrect-20260719T130619Z.rawlog.txt`（md5 `c2ffc4ceb59de03718337c4718bcb13a`·非转抄）+ `raw/w5-abi_shim_weft_to_tcrv.c`（ABI-shim 复现件）**。
> **fixture**：`test/Conversion/RVV/rvv-emit-quant-contraction-q3-K-repack-gemm-prefill-vlen128.mlir`（q3_K repack-GEMM prefill）· **march = 唯一变量**（fixture 逐字节相同·只换 `--weft-rvv-lower-quant-contraction=march=`）。

## 结构半（已 pin·确定性·无板时）：march-唯一变量 → 两个实质不同 body

| march（唯一变量） | half_lanes(θ2) | integer_core_lmul(θ1) | emit_cpp_lines | object `vwmacc.vx` |
|---|---|---|---|---|
| `rv64gcv_zvfhmin`（**VLEN128** → rvv 板身份）| **8** | `mf2` | **20934** | **2048** |
| `rv64gcv_zvl256b_zvfhmin`（**VLEN256** → k1 板身份）| **16** | `mf2` | **10480** | **1024** |

- **双 body 实质不同**：emit 20934↔10480 行（body_A vs body_B 去注释 diff = **11389 行**）·object `vwmacc.vx` 2048↔1024·两 body 均 clang++-20 `-ffreestanding -O2` 编译 OK。
- **两杠杆分离**（关键·A 线纪律）：march 翻转只动 `half_lanes`（θ2·strip 宽度）；`integer_core_lmul` 两侧恒 `mf2`（θ1·[GAP-P1] 默认钉死·march 不改它）⟹ **VLEN-widening 杠杆 ≠ load-width 杠杆**（两条独立承重 θ）。
- **一致性**：`vwmacc.vx` ratio 2048/1024 = 2.0000 == numHalves ratio 2/1 **CONSISTENT**（逐字节复中 census pkg5 / bline verdict §一）。

⟹ **A′ 双 VLEN 双 body 结构主张 = 已证已 pin**（发射器在 march 单变量下产两个实质不同 body·byte 级 object 计数确证）。

## 板半 · rvv（★已兑现 2026-07-19·W5 板窗·run-id `w5-rvvAprime-boardbytecorrect-20260719T130619Z`）

**板身份**：`ssh rvv` = openEuler riscv64 · **VLEN128（VLENB=16）** · 64c · clang 17.0.6-16.oe2403 · isa `rv64imafdcv…zvfh_zvfhmin…`。
**ABI-shim 已核对**：新 emit 符号 `weft_emitc_…gemm_q3_K_q8_K_…(v1,v2,v3, float* v4, const uint8_t* v5,v6, size_t v7)`，读 emit body 行 8–20 确证映射 **v1=nr（v1/4=row_group_count）· v2=bs（输出行 stride）· v3=n/K（`vsetvl(v3)`·v3/256=block_count）· v7=nc（v7/16=col_group_count）**；GEVM 符号 `weft_…gemv_…(v1,v2,v3,v4,v5)` = `(n,s,vx,vy,nc)` **与旧 harness 逐字节同序**（纯改名）。shim `raw/w5-abi_shim_weft_to_tcrv.c` 只置换 ABI·零算术（ZERO-MODEL）。已验证 harness（`tools/e2e-harness/board/kquant_repack_verify_q3K.c`·md5 `be81008fd0e58abad9383809109a0133`）read-only 复用。

**验收结果（body_A VLEN128 on rvv VLEN128·4 seeds anti-hollow）**：

| seed | INT_mismatch_total | 判定 |
|---|---|---|
| 20260708 | **0** | SILICON BYTE-EXACT-INTEGER + BOUNDED-NORM |
| 0xC0FFEE | **0** | SILICON BYTE-EXACT-INTEGER + BOUNDED-NORM |
| 12345 | **0** | SILICON BYTE-EXACT-INTEGER + BOUNDED-NORM |
| 0xABCDEF01 | **0** | SILICON BYTE-EXACT-INTEGER + BOUNDED-NORM |

- 8 shape（GEVM nr=1 ×4 + GEMM nr∈{4,8,16} ×4）× INT 模式 **全 mism=0**·NORM 模式 bounded（worst_norm ≤ 5.1e-7·fp16-d 对抗）。
- **running 板 .o 逐指令回绑结构 pin**：`gemmA.o vwmacc.vx = 2048`（== VLEN128 pin）·`gemmB.o vwmacc.vx = 1024`（== VLEN256 pin）⟹ 硅上跑的就是被结构 pin 的两个 body（非另一份代码）。
- **wrong-VLEN falsifier（各对拍）**：body_B（VLEN256-form·half_lanes=16）在 VLEN128 硬件上跑 = 系统性 **INT-BUG（mism=2687）**·证两 body **真 VLEN-specialized·不可互换**。body_B 自己的 byte-correct 须其**原生 VLEN256 板（k1）**。

⟹ **rvv A′（B7/E7）= 板 byte-correct 完成时**（结构 pin + 硅 mism=0 + 指令回绑 + 反空心 falsifier 四证齐）。

## 板半 · k1（排期·未跑·0 造数）

- **k1 半格（VLEN256 body_B on k1 板·VLEN256）→ 排期 2026-07-21**（k1 板窗·B8/E8）·recipe 同上（march=`rv64gcv_zvl256b`·gemmB.kernel.c·同 shim/harness）·body_B 在其原生 VLEN256 上应 mism=0。本 W5 = rvv 板窗·**不跑 k1**（板别不相交·不虚报）。

## 结论 / 交付首节口径

- **结构半 = 已 pin**（`raw/RAW-rvv-Aprime-emitflip-*.rawlog.txt`·march-唯一变量双 body 确定性证）。
- **★rvv 板半 = byte-correct 已兑现**（run-id `w5-rvvAprime-boardbytecorrect-20260719T130619Z`·`raw/RAW-w5-rvvAprime-boardbytecorrect-20260719T130619Z.rawlog.txt` md5 `c2ffc4ceb59de03718337c4718bcb13a`·mism=0×4 seeds·A′ 转完成时）。
- **k1 板半 = 排期 2026-07-21**（未跑·0 造数）。
- **该 add**：本文件 + `raw/RAW-rvv-Aprime-emitflip-20260719T093213Z.rawlog.txt` + `raw/RAW-w5-rvvAprime-boardbytecorrect-20260719T130619Z.rawlog.txt` + `raw/w5-abi_shim_weft_to_tcrv.c`。
