# FINDING · L3/L4 硅证：拥有内核在物理 no-V 真硅上 byte-exact（3-arm ZERO-MODEL + fp16-9/9 + anti-hollow）

> **性质**：L3/L4 旗舰硅证（W1 载体批 · 裁1 子项①）。**病**：census 第三次确认活仓库采不到 `9/9`——旗舰 byte-exact 证据此前只靠旧 pin 祖先性活着（一次 rebase 全塌），且 A3 tracked `run.json` 实为 **rvv-board run-as-noV 彩排**（`-march=rv64gc`·clang-17·窄豁免），**非真物理 no-V 硅**。本 FINDING 把旗舰证据**在真超锐硅上重跑一次**、以 **run-id 原始输出**钉死于 HEAD（不再靠祖先性）。
> **落盘 = 带 run-id 原始输出**（`raw/RAW-scalar-rerun-carrier-w1-scalar-20260719T092244Z.rawlog.txt`·md5 `6af17b5f22936c8929c892f4f16634af`·非转抄）。**run-id** `carrier-w1-scalar-20260719T092244Z` · board 超锐(真物理 no-V·见同批四路机检 FINDING) · clang 18.1.8 · `-march=rv64gc -mabi=lp64d -O2`。
> **算子**：`weft_scalar.tq2_0_q8_k_vec_dot`（拥有发射器输出·`weft_emitc_tq2_0_kernel_scalar_fallback_first_slice`·非 fallback stub）· fmt=tq2_0 · op=vec_dot · engine=scalar · regime=prefill · n=1048576（nb=4096 super-blocks）。

## byte-exact 门（先于任何计时·ZERO-MODEL [K-5]·3-arm anti-hollow·CORPUS 2-seed）

三条**独立代码路**同时相等 = anti-hollow by construction：**OURS**（拥有发射内核）· **ORACLE**（libcall-free 纯整数独立重算·int64·从真输入字节·独立于 OURS 内部）· **GGML-DEPLOYED**（板上真部署符号 `ggml_vec_dot_tq2_0_q8_K`·no-V build ⟹ dispatch thunk == `_generic` = 真部署标量路·从 `libggml-cpu.so` 链接·非重实现）。

| seed | fp16-golden | bits_ours | bits_ggml_deployed | bits_int_oracle | ALL byte-exact |
|---|---|---|---|---|---|
| `0xD00D` | **PASS 9/9** | `0x46a18a33` | `0x46a18a33` | `0x46a18a33` | **true**（mism=0）|
| `0x1357` | **PASS 9/9** | `0xc50a8972` | `0xc50a8972` | `0xc50a8972` | **true**（mism=0）|

- **fp16 primitive golden 9/9**：纯整数 `weft_h2f` 对 textbook IEEE-754 half 逐位（9 向量：±1/0.5/2/0/1/3·subnormal min/max normal）**9/9 PASS**——这是超块 scale-fold 底下的 L3/L4 数值原语硅证，独立于向量/XOR 路。
- **ours == int-oracle**（编译器中性 ZERO-MODEL）= **权威正确性门**·byte-exact 无条件（`contract` 无关）。**ours == 真部署 ggml** 在对称 fold 下亦 byte-exact（两 seed 均全等）。

## anti-hollow（门会咬·两注入臂各自证 BITES-OK）

| inject | 故障 | 观测 byte-exact | 判 |
|---|---|---|---|
| `1` | 腐蚀 **OURS 输出**（`s_ours += 1.0f`·bits→`0x46a18c33`）| **false** | **BITES-OK**（门捕获 OURS 缺陷）|
| `2` | 腐蚀 **oracle 常量**（`s_int += 1.0f`·bits→`0x46a18c33`）| **false** | **BITES-OK**（交叉核捕获 oracle 缺陷）|

两臂均 exit=0（若故障**未**破坏 byte-exact 则 HOLLOW-FAIL exit=3·未发生）⟹ 门非空心。

## 结论

拥有 `weft_scalar` 发射内核在 **(a) 类物理 no-V 真硅**上 **byte-exact（mism=0·3-arm·2-seed·fp16-9/9·anti-hollow 双臂 BITES-OK）**。旗舰硅证**不再靠旧 pin 祖先性**——run-id 原始输出已钉死。措辞铁线「targeting rv64gc (vector-absent profile), validated on silicon」**字面成立**。

## 边界（[L-6]·成色诚实）

- **enablement 域·NON-Win**：本 FINDING **无性能主张**（[L-6] scalar 永不作贡献基线·任何 ratio 属诊断不入系统账/perf-covered/分母）。
- **(a) 物理板证词 与 (b) 窄豁免合成证词同名不同物·不互顶替**（[F-6]）；本硅证**加强而非替换** committed synthetic capability instance（closure∩rvv.*=∅ 门不需板）。
- **该 add**：本文件 + `raw/RAW-scalar-rerun-carrier-w1-scalar-20260719T092244Z.rawlog.txt`。
