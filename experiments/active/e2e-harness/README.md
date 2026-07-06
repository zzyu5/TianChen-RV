# e2e-harness — T6 端到端分相测量台 (llama-bench 集成)

> **协议脚本位置(A2,2026-07-06):** 本目录已变**纯数据 cell** —— `results/` 测量格 + `models.manifest.csv`
> + 本 README 留在 `experiments/e2e-harness/`;**运行用的 `run_e2e.sh` / `aggregate_e2e.py` / `board/*.sh`
> 已迁至 repo 根 `tools/e2e-harness/`。** 下文 §3/§5 出现的脚本名均指 `tools/e2e-harness/…`;跑法见 §5
> (`cd tools/e2e-harness && bash run_e2e.sh`,结果仍回写本目录 `results/`)。

> **主张定位.** 这是 **T6**(`experiments/T6_e2e_phase_split.csv`)唯一进入 **beat 舞台**的门 —— Win-C(相级 e2e 赢)必须过 `[PERF-1]` 八门后、以 **prefill/decode 分相 vs 出厂 ggml** 呈现(登记阶梯见 `experiments/README.md`)。此前 T6 未接、八门 0/8;本台接上 llama-bench 分相 + 全套纪律 + 违宪纠正后的正确性门。
>
> **三消费者**(为什么这是长杆):① repack 战役的赢落 e2e prefill 分相;② 已归档两赢补 `[PERF-1]` 第④门(micro+e2e,7/8→8/8);③ T6 首批格。

---

## 1. A/B 是什么(注入机制 —— 编译期双构建树)

我方 kernel 的 e2e 注入 **不是** LD_PRELOAD / getenv 开关,而是 **两棵已在板上的 llama.cpp 构建树**,唯一差异 = 是否带 TianChen-RV 编译器发射的 repack kernel 补丁:

| 侧 | 树(板上路径) | ggml q4_0 路径 | 说明 |
|---|---|---|---|
| **A(我方)** | `tcrv-llamacpp/build-gcc15-rv64gcv` | `repack.cpp` 调 `tcrv_emitc_ggml_repack_gemm/gemv_q4_0_q8_0`(`__riscv_vlenb()*8==128` 时 ENGAGE) | 我方**编译器发射**的 repack GEMM+GEMV |
| **B(出厂)** | `llama.cpp-upstream-native/build-gcc15-rv64gcv` | 未打补丁;VLEN128 下 `CPU_REPACK` buffer 被拒 → 回落 `vec_dot_q4_0_q8_0` **block-dot** | 出厂 ggml block-dot(VLEN128 无 repack) |

- 两树**同一编译器(gcc-15.2.0)+ 同旗标(`-march=rv64gcv -mabi=lp64d -O3`)+ `GGML_CPU_REPACK=ON`**;唯一差异是补丁。这满足 §1 第9条"两侧同一版本 + 同旗标",preflight gate-3 机检。
- A 侧引擎接合(`ENGAGED nr/nc/nb`)的确认横幅同时被 preflight gate-4 用来读板 VLEN。

## 2. 分相方法学(llama-bench 原生)

llama-bench 天然分相,直接喂纪律:
- **prefill 分相** = `-p PP`(长 prompt 处理吞吐,tokens/s;GEMM,M 大)。
- **decode 分相** = `-n TG`(逐 token 生成吞吐,tokens/s;GEVM,M=1,内存受限)。
- 一次调用同产两相(`pp<PP>` + `tg<TG>` 两行),`-r REPS -o json` 给每-rep samples 供 median+IQR。

**度量极性:** tokens/s **越高越快**;A/B 比率 = `median(ours)/median(stock)`,`>1` = 我方更快。

## 3. 全套纪律接入(实验总纲v1 §1)

| 纪律 | 落点 |
|---|---|
| fail-closed preflight 四门(第9/10条) | `board/preflight_e2e.sh`:①march 完整性 ②双侧 objdump libcall 扫描 ③同编译器+同旗标 ④指纹↔T-格 VLEN |
| 配对 A/B 同会话交替 | `board/phase_split_ab.sh`:ours,stock,ours,stock… |
| T-N 噪声地板 | 2 passes → pass1-median vs pass2-median between-pass floor;判定门 `|Δ|>2×floor ∧ CI∌1.0` |
| median + IQR + bootstrap CI | `aggregate_e2e.py`(移植自宪法版 `T3_step3/aggregate.py`) |
| 固频 + 钉核 | `taskset -c $CORES`;每-call 抓 `scaling_cur_freq`,DVFS span 守卫(>2% 作废) |
| board-labeled + instance-hash 键控 | `phase_split_ab.sh` ENV FINGERPRINT 段(板/ISA/内核/libc/编译器/旗标/march/线程/governor/model-sha);写进 `evidence.json` |

### march-完整性门的一个诚实裁决(gate-1)
当前板上两树用 `-march=rv64gcv`,**漏了板实测的 `zfh/zvfh/zvfhmin/zb*`**。§1 第9条 gate-1 本应 fail-closed。但 gate-1 的**动因**是防 fp16 softfloat libcall 混淆(P2c),而 **gate-2 objdump 扫描直接量到该物**:两树 ggml-cpu 目标**均无** `__extendhfsf2` 类 libcall(ggml 用自带 fp16→fp32 查表,不走 libcall)。故 harness 把 gate-1 降为 **ADVISORY**(需 `ALLOW_MARCH_INCOMPLETE_IF_LIBCALL_CLEAN=1`),并明确:**SEALED T6 格仍要求全能力 march 重建**;当前 bring-up 数是 `march-incomplete-but-libcall-clean`,**对称**(两侧同 march,非 P2c 的"残废对手"非对称),可作方向性证据,不作封印格。

## 4. 正确性门(违宪纠正 —— 钉死)

- **KERNEL 级** = **byte-exact vs 我方自家钉死 oracle**(由 `.trellis/scripts/ondevice/*` 的 vec_dot 驱动另行确立;本台不重导)。
- **E2E 级** = ①**贪心 token 一致**(temp=0 贪心 + 固定 seed 下,A 与 B 输出 token 序列**相同**)+ ②**logits sanity**(输出连贯、无 NaN/Inf)。
- **【永不】声称 bit-exact vs ggml** —— e2e fp 累加序不同、都合法 IEEE,argmax 下的 token 序列一致才是正确的门。
- 实现见 `board/correctness_gate.sh`。工具注:此构建只有 `llama-cli`(无 `llama-completion`),且 tinyllama 是 chat 模型(自动会话模式),故贪心生成用 `-st`(单轮)+ stdin 喂 prompt,回复经"剥离 loading 旋转/ASCII banner/perf 行"归一化后比较。

## 5. 怎么跑

```bash
# 全流程(preflight → 分相 A/B → 正确性门 → 本地聚合)。脚本在 tools/e2e-harness/,
# 结果仍回写 experiments/e2e-harness/results/<LABEL>/(RESULTS_ROOT 可覆盖)。
cd tools/e2e-harness
LABEL=rvv-bringup-q4_0-vlen128 REPS=4 PASSES=2 PP=128 TG=32 bash run_e2e.sh

# 换格式/板:覆盖 env(见 experiments/e2e-harness/models.manifest.csv)
QUANT=q8_0 MODEL=/home/ubuntu/tcrv-llamacpp/models/tinyllama-q8_0.gguf \
  A_BUILD=... B_BUILD=... EXP_VLEN=128 bash run_e2e.sh
```

产物:`experiments/e2e-harness/results/<LABEL>/{preflight.txt, phase_split_raw.txt, correctness.txt, aggregate.txt, evidence.json}`。

## 6. 钉死的板/工具链(bring-up 环境指纹)

- **板** = `ssh rvv`(openEuler 24.03,`localhost.localdomain` kernel 6.12.66 riscv64,**VLEN128**,64c,governor=performance,max 2.6GHz,glibc 2.38)。
- **ISA(板实测)** = `rv64imafdcv_..._zfa_zfh_zfhmin_..._zba_zbb_zbc_zbs_..._zvfh_zvfhmin_...`。
- **工具链** = gcc-15.2.0 @ `/opt/tcrv-toolchains/gcc-15.2.0`(含 `GLIBCXX_3.4.32`;跑板上二进制需 `LD_LIBRARY_PATH=/opt/tcrv-toolchains/gcc-15.2.0/lib`)。
- **模型** = `tinyllama-q4_0.gguf`(637699456 B,sha256 `da3087fb14aede55…`,llama 1B Q4_0,606.53 MiB,1.10B params)。
- **llama.cpp 基线** = 本地 checkout `b9652`/`6eab47181`;板上 `tcrv-llamacpp` HEAD `f3e1828`(dirty=ggml 补丁)。

## 7. 消费者就绪度

- **① repack e2e**:q4_0@128 分相 A/B 已可跑并产数(见 `results/rvv-bringup-q4_0-vlen128/`)。**就绪**(待全能力 march 重建封印)。
- **② 补 `[PERF-1]` 第④门(micro+e2e)**:分相已产 e2e 侧;需与已归档 micro 赢配对做 Amdahl 传导会计(§1 第5条)才封第④门。**半接**(e2e 侧就绪,传导会计待补)。
- **③ T6 首批格**:`evidence.json` 已是可入库的测量格结构;**当前状态 = `bring-up`(非 sealed)**,不污染 `T6_e2e_phase_split.csv` 模板;全能力 march 重建 + 八门齐后填 sealed 格。

## 8. STOP/边界(本台不做的)

- 不改 `lib/ include/ schema/ ODS/ docs/`(并行线;touch-set 限 `experiments/`)。
- 不声称 bit-exact vs ggml(§4)。
- 不引用 sealed T6 格,除非八门齐 + 全能力 march(gate-1 非 advisory)。
- 板不可达 / llama.cpp 集成结构死墙 / falsifier 红因不明 → STOP 报根因(当前均未触发:板可达、集成走通、A/B 均产数)。
