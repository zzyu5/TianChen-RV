# PRD · K线 · product_reduce.sh harness + 重测 3 格

> **权威** = 《开测篇》§二.2 落账清欠先行（gcc 车道 clang-18 重测·恒等式门已死只认真测）+ K-actionable-queue §1。
> **性质** = 纯 actionable（建 harness 是「工」·非 gated）·真板测走 bench 唯一通道。

## 一、目标

建 `tools/bench/cells/product_reduce.sh` harness + bench 重测 **3 格** = `codebook_n3` / `offset_binary_n3` / `q4_0_nibble`
（product_reduce 族·N3 gearbox 归约子原语·gcc 车道真污染 3 格之一批）。offset_binary_n3 现值 0.797（近 PASS·可能翻）。

## 二、harness 契约（[ISSUE-090]·同 dequantize_row.sh/scalar_vec_dot.sh/gemm_tile.sh）

- 签名 `product_reduce.sh <board> <mode> <fmt>`·op=product_reduce·**harness 禁写仓库侧持久文件**·全 stdout·
  经 runner fail-closed 写入闸落三目的地。板端 `/tmp` 临时可接受·仓库侧不 scp 回。
- 源资产（driver/kernels）住数据格 `experiments/active/k-product-reduce/`·harness 只读。
- **单世界对称**：我方向量核与 scalar-ref 对手同 clang-18 同板编译（清 gcc 车道欠账）。
- mode: `verify`（build + ZEROVEC objdump 探针 + ZERO-MODEL byte-exact + 反空心·NO TIMING）/ `sanity` / `measure`（cold N=25 2-seed flush>LLC）。

## 三、验收

1. **harness 契约合规**（禁写仓库文件·全 stdout·+x 可执行）· driver oracle 是真 **ZERO-MODEL**（[K-5]·从原始字节独立重算·非捕获 intermediates）· 反空心臂真触发。
2. **3 格真 cold**（rvv·+ k1 若格支持）· `verify` byte-exact GREEN + zero-vector 探针 · `measure` cold N=25 2-seed 落 `runs/`· runs.log 一行。
3. **verdict per cold**：≥0.8 PASS / <0.8 具名-X（**若<0.8 走完整攻坚环**：反汇编瓶颈→构造→前门→再测·**非跳环认输**）。
4. **成色**：对手 = constructed scalar-ref（product_reduce = internal sub-primitive·ggml 无 standalone 框架核）·**便宜档·大倍数禁称硬赢**（[NG-4] SANITY·非 e2e·非 perf-covered）。
5. **入账**：recon-dict 留 main 会话（真数落 runs/·汇报 cold·**不直写 master**·ISSUE-098）。

## 四、触碰集 / 铁律

- 触碰：`tools/bench/cells/product_reduce.sh` + `experiments/active/k-product-reduce/`（driver/kernels·前一 agent 半写）+ bench 跑(runs/)。
- **独占**（与并行线 vec_dot.sh / dequant emit 不相交）。**0 造数**·byte-exact 硬门实证·sealed 9/83 不动·**禁 commit·禁 add -A**·"某物不存在"禁截断命令。

## 五、状态 / 遗留

- 前一 agent（网断）已写 harness+driver 半成品·未测。本轮 = 审+完成+板测。
- 翻 PASS 格 = 标量类硬门 +N；具名格经环上报。
