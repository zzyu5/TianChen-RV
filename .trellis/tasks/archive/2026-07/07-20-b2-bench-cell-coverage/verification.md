# B2 Verification Record

日期：2026-07-20  
性质：correctness / route / parser / retirement evidence；**不是性能报告，也不是 official run lineage**。

## 1. 结论

B2 声明的原子切片已经完成：canonical runner 使用精确 route registry 与每-op 唯一 verify/cold parser registry；K-quant vec_dot 五格式、product_reduce 三格式拥有可执行 correctness contract；scalar 只保留 dormant route/parser fact，不能反向获得 roster、真跑或 Win 资格。

本任务没有运行 cold campaign，没有写入 `experiments/runs.log`、`experiments/runs/`、`experiments/master/` 或 `schema/coverage-roster.v1.json`，也没有改动任何性能数字、分母或论文 verdict。

## 2. Hermetic runner 与本地门

- `python3 tools/bench/bench --self-test`：**17/17 PASS**。
  - 精确 route 闭包及错 op/format/board/engine 负控；
  - verify/cold parser registry 同域、missing parser 无 GEMM fallback；
  - GEMM parser 迁移回归；
  - K-vec 五叶、runtime VLEN、固定 fixture 三向 exact、min 反事实、bsums corruption、重复 marker、NaN；
  - product_reduce 三格式 verify/cold 与重复 marker；
  - scalar dormant parser 可测活，但 roster key 明确拒绝。
- `python3 -m py_compile tools/bench/bench tools/bench/measurement_keys.py .trellis/scripts/recon_master_rebuild.py`：PASS。
- `bash -n tools/bench/cells/{gemm_tile,dequantize_row,vec_dot,product_reduce,scalar_vec_dot}.sh`：PASS。
- 各 cell 非法 mode、unsupported route/format 与 scalar official key 均在 SSH 前 fail-closed；dry-run 不分配 run-id、不写盘。
- 修改前后 `experiments/runs.log`、`experiments/runs/`、`experiments/master/`、`schema/coverage-roster.v1.json` diff 为空。

## 3. 真板 correctness（direct cell stdout 由当前正式 parser 消费）

所有命令均为 `verify`，不计时、不创建 official run：

| 族 | rvv · VLEN128 | k1 · VLEN256 | 结果 |
|---|---:|---:|---|
| vec_dot q2_K–q6_K | 5/5 | 5/5 | parser 返回结构化 PASS；五 leaf 逐格式实际执行 |
| product_reduce 三格式 | 3/3 | 3/3 | parser 返回结构化 PASS |
| GEMM iq1_s parser 回归 | 1/1 | 1/1 | 旧 GEMM 行为迁移后保持通过 |

K-vec 每个 clean arm 都在 B2 固定、受控、精确整数 fixture 上完成 owned leaf / deployed ggml / independent raw-byte oracle 三向 byte-exact；三种结果 fault、leaf fault 和 q8-bsums corruption 均 bite。q2_K/q4_K/q5_K 的 min-active arm 在两板均观察到非零 min contribution，并由 `dmin=0` 反事实证明该项真实参与计算。`K=2048=8×QK_K`，覆盖多 block pointer advance 与跨 block fold；唯一 ABI 是 `QK_K=256`。

该 byte-exact 结论只属于上述受控 fixture，不外推所有合法浮点输入。

## 4. 对手与产物身份

### Deployed ggml

- rvv：clang `18.1.8`，source HEAD `f3e182816421c648188b5eab269853bf1531d950`，`libggml-cpu.so` SHA-256 `0782dda02f0b7fbe630b1acc2cc7fde44713b8f4558b3a9eb5640a52da7e5924`。
- k1：Bianbu clang `18.1.8 (11bb4)`，`libggml-cpu.so` SHA-256 `3730c87c8f069e1e2c2c34b2bc3c083be034d8b0dd9b01ba8d85dcad3b792b5f`。
- 每次 cell 运行另记录实际 exported/runtime symbol、路径档位和库 md5 前后守恒。q2/q3/q4/q6 实际走板宽专化；q5 两板实际走 exported 通用向量 body。真实部署路径与对手强度分列，不互相替代。

### CORE==PROD

五叶均由当前 production source front door → emission plan → EmitC pipeline 再生成后逐字节核对：

| leaf | md5 |
|---|---|
| q2_K | `450470f89d0d9ed7c8ebfe3be7afaa4d` |
| q3_K | `ddc18ad104bc10757ea06a3d85411b92` |
| q4_K | `892b6cf8d54b3558659d0b726b563105` |
| q5_K | `9970c3231afabfa770209081acfcdd30` |
| q6_K | `4de70f5162ead640ea85b0ed28f759d4` |

核查发现旧 q5 leaf 已落后于当前 emitter，已直接替换为当前生成结果并重新完成双板验证；旧 leaf 与兼容选择均未保留。`vec_dot.sh` 现在在 SSH 前逐叶核对 `GEN_SEAL.txt`，任何缺失或漂移均硬失败。

## 5. 全仓回归

`cmake --build build/weft --target check-weft -j4`：**975/978 PASS**。仅保留既有 ISSUE-057 三个失败：

1. `rvv-generated-bundle-abi-e2e-self-test`；
2. `explicit computed masked strided input widening dot reduce add dry-run`；
3. 对应的 pre-realized case。

B2 没有新增失败。issue census 为 125 条、零缺号、零重号；状态总数与 `.trellis/spec/issues/index.md` 同步。
`check_retired_index.py` 与 `check_monolith_retire.py` 均 GREEN；历史实验目录/索引门的红项计数分别为 731/1370，与合并基底逐项运行结果相同，B2 未新增结构债。

## 6. 明确未完成

- 未跑任何 B2 cold/measure，因此不声称更快、持平、PASS 或具名-X 发生变化；
- 未创建 official run，未做 T-N qualification 或 master promotion；
- scalar 的 ISSUE-061/104、roster 行和 enablement 落点仍未解决；
- product_reduce 三格的正式 clang-18 cold 重测仍属于 ISSUE-099；
- K-quant 性能攻坚属于后续 B3，而不是 B2 correctness 结论。
