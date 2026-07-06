# q4_0@128 e2e bring-up — rvv (openEuler VLEN128) — 我方 repack vs ggml block-dot

**Status = `bring-up` (方向性证据,NON-SEALED).** 不入 `T6_e2e_phase_split.csv` 的 sealed 格,不进 beat 语境,直到 §caveats 关闭 + `[PERF-1]` 八门齐。

## 环境指纹 (instance-hash 输入)
- 板 = `ssh rvv`,openEuler 24.03,kernel 6.12.66 riscv64,**VLEN128**,64c,governor=**performance**,freq-locked **2.6 GHz**(DVFS span 测得 **0.00%**),glibc 2.38。
- 工具链 = **gcc-15.2.0**,`-march=rv64gcv -mabi=lp64d -O3 -DNDEBUG`,`GGML_CPU_REPACK=ON`;**A/B 两侧同编译器 + 同旗标**。
- 模型 = `tinyllama-q4_0.gguf`(sha256 `da3087fb14aede55…`,606.53 MiB,1.10B params)。
- A(我方)= `tcrv-llamacpp/build-gcc15-rv64gcv`(编译器发射 repack GEMM+GEMV,VLEN128 ENGAGE);B(出厂)= `llama.cpp-upstream-native/build-gcc15-rv64gcv`(block-dot)。
- 抽样:pinned cores 8-11,`-t 4`,`-r 4` × **2 passes** = 8 samples/side/phase。

## Preflight (fail-closed 4-gate)
`PASS 4/4`(gate-1 **advisory**:`rv64gcv` 漏板 `zfh/zvfh/zb*`,但 gate-2 objdump 证两侧**均无** `__extendhfsf2` 类 fp16 libcall → §1第9条 防的混淆**经验缺席**;对称,非 P2c "残废对手")。

## 分相结果 (median t/s;越高越快;比率=ours/stock)

| 相 | ours | stock | ratio | 95% CI | IQR%(o/s) | between-pass floor | 判定 |
|---|---:|---:|---:|---|---|---|---|
| **prefill** (pp128, GEMM) | 19.63 | 3.94 | **4.99×** | [4.98, 4.99] | 0.09 / 0.02 | 0.02% | **DIFFERENCE** |
| **decode** (tg32, GEVM) | 3.11 | 2.01 | **1.55×** | [1.53, 1.56] | 1.88 / 0.62 | 1.43% | **DIFFERENCE** |

两相均过 `|Δ|>2×floor ∧ CI∌1.0`。

## 正确性门 (违宪纠正)
- **E2E 贪心-token 一致**:3/3 prompt A==B(`The capital of France is Paris.` / Lily 故事 / `is 4.`)。**GREEN**。
- **logits sanity**:无 NaN/Inf。
- **未**声称 bit-exact vs ggml(fp 累加序不同、都合法 IEEE)。
- **KERNEL 级** byte-exact vs 自家 oracle 由 ondevice vec_dot 驱动另证(不在此复导)。

## Caveats (封印前须关)
1. **march advisory**:sealed 格需**全能力 march 重建**(`rv64gcv_zfh_zvfh_zba_zbb_zbs…` 从板 hwprobe 生成),使 gate-1 从 advisory 转 OK。当前 libcall-clean 使数可辩护,但非 sealed。
2. **decode 是内存受限 + 共享板**:rvv 是多租 64c 板;decode(GEVM)吞吐随**全板内存压力**波动(pin-core 挡不住跨核带宽争用)。同一次 bring-up 的更轻负载 smoke(`-r 2`)测得 ours/stock=9.10/4.56=**2.0×**;本盘重载下降到 1.55×。**A/B 同会话交替使比率仍有效**(两侧共负载),但 decode 绝对吞吐与比率对全板负载敏感 → 封印须多快照 + 记录并发。prefill(compute-bound)稳(4.99×,IQR 0.09%)。
3. **单模型/单形状**:仅 tinyllama q4_0 pp128/tg32;T6 首批格应扫 M(prefill GEMM shape 类)+ 加 q8_0。

## 复现
`experiments/e2e-harness/run_e2e.sh`(默认即本配置):
```bash
LABEL=rvv-bringup-q4_0-vlen128 REPS=4 PASSES=2 PP=128 TG=32 bash run_e2e.sh
```
