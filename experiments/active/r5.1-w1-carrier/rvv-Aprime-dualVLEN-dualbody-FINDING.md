# FINDING · rvv A′ 双 VLEN 双 body（D4）：emit+object 级 march-唯一变量 flip 已 pin·板 byte-correct 半排期

> **性质**：A′「双 VLEN 双 body」结构主张（W1 载体批 · 裁1 子项④·H-1 handback §E · ISSUE-105 · K-attack-fanout 机制①）。**一次板窗兑现两主张**（B7/E7 + B8/E8）的论文侧最便宜完成时。
> **本 FINDING 的两半**：**(结构半) emit+object 级双 body 分歧 = 已 pin（确定性·本节）**；**(板半) rvv+k1 各自 byte-correct = 排期（下节·带精确 recipe + 日期）**。
> **落盘 = 带 run-id 原始输出**（`raw/RAW-rvv-Aprime-emitflip-20260719T093213Z.rawlog.txt`·md5 `b4bf41db96902c71bcbb9953a0f5d684`·非转抄）。
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

## 板半（排期·不虚报·0 造数）：rvv+k1 各自 byte-correct（4-arm anti-hollow·mism=0）

**为何未在本轮跑**：板 byte-correct 需把**新鲜 weft emit body** 接入已验证的 q3_K repack oracle harness，但两者 **ABI 不 drop-in 兼容**（须先做 ABI-shim·核对无误方可跑·否则 false-pass/crash 违 0 造数纪律）：
- 新 emit 符号 `weft_emitc_ggml_repack_gemm_q3_K_q8_K_kernel_...(size_t v1,v2,v3, float* v4, const uint8_t* v5,v6, size_t v7)`（改名后 `weft_` 前缀·三 size_t 前置·v3=K/`vsetvl(v3)`·`v3/256`=block_count）；
- 已验证 harness（`tools/e2e-harness/board/kquant_repack_verify_q3K.c` + `tools/oracle-repack/oracle_repack_q3K.cpp`）用**旧 `tcrv_` 前缀**、**旧 arg 序** `(n, s, vx, vy, nr, nc, bs)`。
- **★须先读 body 循环边界确证 (v1,v2) = (nr,nc) 的准确映射**（写 shim `weft_(nr,nc,n,s,vx,vy,bs)` ← `tcrv_(n,s,vx,vy,nr,nc,bs)`），再对同输入与已验证的 hand-written kernel 比 byte-identical（经已验证 oracle 传递证 byte-correct）。

**排期（限期动作·非「确认」）**：
- **rvv 半格（VLEN128 body_A on rvv 板·VLEN128）→ 排期 2026-07-20**（rvv 板窗·W3 判决实验空闲让 W5/W1 后·§九调度令）。
- **k1 半格（VLEN256 body_B on k1 板·VLEN256）→ 排期 2026-07-21**（k1 板窗·W2 修完 ISSUE-120 让出后）。
- **一次板窗合并两主张**（B7/E7 rvv + B8/E8 k1）·各自 4-arm anti-hollow·mism=0·落 byte-correct 到本 FINDING「板半」表。

**验收（板 byte-correct）**：body_A 在 rvv(VLEN128) byte-identical 于已验证 q3_K repack kernel（同 seed·经 oracle 传递）·body_B 在 k1(VLEN256) 同理·两注入臂 BITES-OK。

## 结论 / 交付首节口径

- **rvv 半格 = 结构半已 pin**（本 FINDING + `raw/RAW-rvv-Aprime-emitflip-*.rawlog.txt`·march-唯一变量双 body 确定性证）；**板 byte-correct 半 = 排期 rvv 2026-07-20 / k1 2026-07-21**（recipe 已写死·仅缺 ABI-shim 核对 + 一次板窗）。
- **该 add**：本文件 + `raw/RAW-rvv-Aprime-emitflip-20260719T093213Z.rawlog.txt`。
