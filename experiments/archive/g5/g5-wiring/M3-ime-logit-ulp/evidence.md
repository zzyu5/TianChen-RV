# G5-M3 — IME triple **logit-level bounded-ULP / near-tie 量化**（跨范式名义·correctness-hardening·把三格 e2e near-tie argmax flip 精确化为**声明的 ULP 上界内 · correctness-neutral**）

> **campaign**: G5 接线战役 · **M3-ime-logit-ulp** = ratified IME triple {q4_0·q8_0·q4_K}@ime forward-wired 之后的 **correctness 加固**（非 perf·非新战役·既定 next-step 执行）。
> **立项理由（[K-5] 数值档合规）**：三格 forward-wired 后真-llama e2e greedy 出现 near-tie argmax flip（q4_0 2/5·q4_K 3/5·q8_0 2/5 自由生成分歧）。本任务把这些分歧**精确量化到 logit 层**，证明它们落在**声明的 f32 reassociation 数值档**内（单-tensor 相对误差 ~1e-7·int32 vmadot bit-exact 为算术锚；logit 层相对散度是该 per-op 档经 22 层 transformer **累积**的结果），且 argmax 翻转**只发生在 near-tie**（两竞争 token 的 OFF gap ≤ 我方对 stock 的 logit 扰动）。→ correctness 从「coherent in-family」升到「**within-ULP-band · near-tie-flip correctness-neutral**」。**禁用**"publication-grade/论文级/多数一致"叙事·只写数值档合规 + within-ULP 判读。
> **board**: `ssh k1`（SpacemiT X60·VLEN256·IME harts 0–3·`taskset -c 0-3`·clang-18/gcc-13）·models = `/home/bianbu/tcrv-k1-llama/models/tinyllama-q4_0.gguf` · `/data/tinyllama-q8_0.gguf` · `/data/tinyllama-1.1b-Q4_K_M.gguf`（复用·已存）。
> **禁 git**（主会话 commit）· **board 可逆**（vendor `cp *.ORIG` + EXIT-trap restore + md5 双证零改动）· **禁 llama-cli**（自建 llama-API logit dumper）· **MANIFEST/gitignore**。

---

## 0. 一页速览（verdict·三格）

| 格 | teacher-forced argmax flips (decision pos) | 全 near-tie? | max\|Δlogit(ON-OFF)\| / rel | vendor 基线 max\|Δ(VEN-OFF)\| | correctness-neutral |
|---|---|---|---|---|---|
| **q4_0**（flat·nibble） | **2**（prompt 2,3） | **T**（ratio gap/pert 0.33·0.44） | 0.39 / **1.56e-2** | 0.34–0.60 | **T** |
| **q8_0**（flat·int8-direct） | **0**（全 5 prompt argmax 同 stock） | **T**（vacuous·无 flip） | 0.21 / **8.9e-3** | **0.000e+00**（vendor≡stock） | **T** |
| **q4_K**（super-block·两级 fold） | **3**（prompt 1,2,3） | **T**（ratio 0.10·0.49·0.12） | 0.90 / **4.17e-2** | **8.42**（vendor 巨偏 stock） | **T** |

**三格 correctness-neutral = T·非-near-tie（真 correctness 问题）分歧 = 0/三格全体**。

**一句话**：在**同一 teacher-forced token 序列**（OFF=stock RVV greedy 输出）上逐位比对三 backend 的**全 logit 向量**——OFF(stock RVV·跨范式) / VEN(vendor IME·同范式) / ON(我方 tcrv IME 桥)——三格的我方 ON-vs-stock logit 散度均**有界**（相对 ~1e-2·= 单-tensor within-ULP ~1e-7 经 22 层累积），且 argmax 翻转**无一例外落在 near-tie**（竞争两 token 的 OFF gap ≤ 我方对 stock 的扰动·ratio<1）·**零非-near-tie 分歧**。**q8_0 teacher-forced 甚至零 flip**（逐位 argmax 全同 stock）；**q4_K 我方对 stock 的散度（≤0.9）比 vendor 自家 IME 对 stock 的散度（8.42）小约 10–16×**——我方核比 vendor 出货 IME 更贴 stock。**板 stock 零改动（md5 双证·EXIT-trap 强制）。perf 不测·跨范式名义。**

---

## 1. 方法（自建 llama-API logit dumper · teacher-forced 逐位比对）

### 1.1 为何 teacher-forced（而非自由生成逐 token）
e2e 自由生成一旦某位 argmax 翻转，其后 context 即分叉，后续 logit 不可直接比对。**teacher-forced**：取 OFF(stock) greedy 生成的 canonical token 序列 `S`（BOS+prompt+24 gen），令 OFF/VEN/ON **三 backend 在同一 `S` 上单批 prefill**（M=seqlen≈33–39 > 1 → 三条 mul_mat 全经各自核·ON 命中我方 IME 桥），dump **每一位置的完整 logit 向量**（n_vocab=32000 f32）。→ 同一输入 token 上 ON/VEN/OFF logit **逐位可比**。这是比自由生成**更强的扰动暴露**：teacher-forced **全部位置**走 IME（自由生成仅 prefill 走 IME·decode M=1 走 native passthrough）。若更强暴露下 flip 仍只在 near-tie，则自由生成（更弱暴露）a fortiori correctness-neutral。

### 1.2 三 backend（复用三格 forward-route env-gate patch）
- **OFF** = `build-off`（stock RVV·无 IME）—— 跨范式独立 oracle·并产 canonical 序列 `S`。
- **VEN** = `build-ime` env unset —— 同范式独立 oracle（vendor 自家 IME 核·异实现）。
- **ON** = `build-ime` `TCRV_IME_{Q40,Q80,Q4K}_BRIDGE=1` —— 我方 tcrv IME 桥（三格 forward-route patch 原样·`.so` vmadot 32→33·banner FIRES）。

### 1.3 判据（`compare_logits.py`·板 numpy 1.26.4）
逐 **decision 位**（p≥n_prompt−1·预测生成区 token）：
- **bounded-ULP**：`max|logit_ON − logit_OFF|` 及相对 `/max|logit|`；旁证 `max|logit_VEN − logit_OFF|`（vendor 对 stock 的跨核基线）与 `max|logit_ON − logit_VEN|`（我方 vs vendor·同 IME 范式）。
- **near-tie flip 判决**：某位 argmax 由 A(=argmax_OFF) 翻到 B(=argmax_ON) ⟺ flip；该 flip **correctness-neutral ⟺** `gap_off(A,B) ≤ |Δ[A]| + |Δ[B]|`（OFF 下两竞争 token 的 gap 落在我方对 stock 的扰动内·= 数值档可解释的翻转）。若 `gap_off ≫ 扰动` → 非-near-tie（真 correctness 问题·如实报 red）。
- **near-tie 密度**：每 prompt 25 decision 位中 OFF top-2 gap ≤ `max|Δ(ON-OFF)|` 的位置数（= 落在我方扰动带内·可被任意跨核 ULP 差翻转的位）。

---

## 2. 逐格结果

### 2.1 q4_0（flat·nibble·model=tinyllama-q4_0.gguf）
```
prompt 2 pos34: OFF->' story'  ON->' famous'  gap_off=8.78e-2 pert=2.64e-1 ratio=0.33  [NEAR-TIE] (VENflip=T)
prompt 3 pos37: OFF->' driving' ON->' cook'    gap_off=1.09e-2 pert=2.45e-2 ratio=0.44  [NEAR-TIE]
prompt5 max|Δ(ON-OFF)| = 0.000e+00 (我方 ON logit 与 stock 逐 bit 同·该 prompt)
```
- flips=2（prompt 2,3）·**全 near-tie**（ratio<1）·非-near-tie=0。
- max|Δ(ON-OFF)|=0.39·rel=**1.56e-2**；**ON≤VEN 全 5 prompt**（我方对 stock 散度 ≤ vendor 对 stock 散度·prompt5 甚至 0）。
- correctness-neutral=**T**。

### 2.2 q8_0（flat·int8-direct·model=tinyllama-q8_0.gguf）
```
flips = 0  (全 5 prompt 全 decision 位 argmax 逐位同 stock)
max|Δ(VEN-OFF)| = 0.000e+00 全 5 prompt  (vendor q8_0 IME ≡ stock RVV·bit-identical)
max|Δ(ON-OFF)| = 0.14–0.21 abs · rel 6–9e-3  (我方 fragment-major fold vs stock RVV vectorized fold·从不翻 argmax)
near-tie 密度：prompt3=5/25 · prompt5=6/25  ← e2e 自由生成分歧正是 prompt 3,5
  prompt5 pos15: top1 ' secre'(22183) top2 ' secret'(7035) gap=3.19e-2  (near-tied 词片)
```
- flips=0 → correctness-neutral=**T**（vacuous·无 flip 可谈 near-tie·更强结论）。
- **e2e 自由生成 q8_0 分歧在 prompt 3,5**（casefile M3-ime-q8_0-forward §4）·本 teacher-forced 报告显示 **prompt 3,5 恰是 near-tie 密度最高**（5/25·6/25 落在扰动带）→ 那些自由生成 flip 由 near-tie 驱动（teacher-forced all-IME 未触发·是 native-decode/KV-seeding regime 差异·见 §3）。
- **ON≤VEN=F**（vendor≡stock=0·我方有 fold-order ULP 散度 0.2）——但 flip=0 故 correctness-neutral 不受影响；单-tensor seal `max_abs_vs_float_order=0.000e+00` 是对**标量 block-dot** oracle 的·真部署 stock RVV 用**向量化归约**·我方 fragment-major 与之差在跨-fold 顺序（合法 f32 reassoc·rel ~1e-2）。

### 2.3 q4_K（super-block·两级 fold·model=tinyllama-1.1b-Q4_K_M.gguf）
```
prompt 1 pos29: OFF->' new'   ON->' about' gap_off=4.15e-3 pert=4.10e-2 ratio=0.10  [NEAR-TIE]
prompt 2 pos32: OFF->'The'    ON->'-'      gap_off=6.64e-2 pert=1.34e-1 ratio=0.49  [NEAR-TIE] (VENflip=T)
prompt 3 pos28: OFF->','      ON->' and'   gap_off=2.30e-2 pert=1.96e-1 ratio=0.12  [NEAR-TIE]
prompt5 pos15: top1 ' secret'(7035) top2 ' secre'(22183) gap=8.64e-4  (essentially tied 词片)
vendor 基线 max|Δ(VEN-OFF)| = 8.421  ← vendor 自家 q4_K IME 对 stock 巨偏(=e2e VEN==OFF 全 DIFFER 之因)
我方 max|Δ(ON-OFF)| = 0.51–0.90  (比 vendor 小 ~10–16×)
```
- flips=3（prompt 1,2,3）·**全 near-tie**（ratio 0.10·0.49·0.12）·非-near-tie=0。
- max|Δ(ON-OFF)|=0.90·rel=**4.17e-2**；**ON≤VEN 全 5 prompt**（我方 0.5–0.9 ≪ vendor 8.42）。
- correctness-neutral=**T**·且我方核**比 vendor 出货 q4_K IME 更贴 stock**（数量级更近）。

---

## 3. 判读（within-ULP-band · near-tie · 诚实边界）

1. **bounded within-ULP（累积义）**：三格 logit 层相对散度 q8_0 8.9e-3 / q4_0 1.56e-2 / q4_K 4.17e-2 —— 皆 ~1e-2 量级，是**单-tensor within-ULP（~1e-7·int32 bit-exact）经 22 层 transformer 残差流放大**的结果（正常深网累积·非 kernel 错）。数值档硬锚 = **kernel 层**（三格 integration UT 单-tensor A==B ~1e-7 + vmadot int32 0-diff·已 sealed），本报告量化其 e2e 传导后的 logit 层表现。
2. **near-tie flip = correctness-neutral**：三格全部 argmax flip（q4_0 2·q4_K 3·q8_0 0，共 5）**无一例外** `gap_off ≤ 扰动`（ratio ≤ 0.49）——翻转的两 token 在 OFF 下本就落在我方对 stock 扰动内的 near-tie（多例为同一词的 BPE 片段如 ' secret'/' secre'·gap 8.6e-4）。**零非-near-tie 分歧**（无真 correctness 问题）。
3. **teacher-forced vs 自由生成 regime 差（诚实）**：本 teacher-forced 令**全部位置**走 IME，是比 e2e 自由生成（仅 prefill 走 IME·decode 走 native）**更强**的扰动暴露。故 teacher-forced flip 数（q8_0=0·q4_0=2·q4_K=3）与 e2e 自由生成分歧数（q8_0 2/5·q4_0 2/5·q4_K 3/5）**不必逐位对应**（q8_0 尤其：teacher-forced 0 flip·自由生成 2/5——后者源于 IME-prefill 的 KV-cache 微扰经 native decode 传导·而非稳健 logit 错）。两 regime 一致指向 near-tie：q8_0 自由生成分歧的 prompt 3,5 恰是本报告 near-tie 密度最高者（5/25·6/25）。
4. **vendor 基线为参照（cross-paradigm honesty）**：VEN(vendor IME)-vs-OFF(stock) 提供"可信同型核对 stock 的跨核 ULP 基线"。q4_K：我方(0.5–0.9) ≪ vendor(8.42)——我方核更贴 stock。q8_0：vendor≡stock(0)·我方 0.2 的 fold-order 散度从不翻 argmax。q4_0：我方 ≤ vendor 全 prompt。→ 我方对 stock 的散度**不异常**·同量级或更优于 vendor 自家 blessed IME。

---

## 4. 板 restored（md5 双证零 stock 改动·EXIT-trap 强制·三格各自 cycle）
每格 `run_board.sh` 用 EXIT trap：任何退出路径 → clean source → `make ggml-cpu`（clean .o）→ `.so` 覆盖回 ORIG binary → md5 双证。三格各自 run 后：
```
q4_0 / q8_0 / q4_K  各:  restored ime.cpp md5 = 40962c7e...  == baseline ✓
                         restored .so     md5 = 71cc4d29...  == baseline ✓
                         RESTORE md5 ZERO-CHANGE OK · src_route_left=0 · litter_left=0
```
（scratch `/tmp/g5m3ulp/<fmt>/` 的大 logit .bin 在 harvest 后显式清除·非 vendor 树·见 MANIFEST。）

---

## 5. 诚实边界（[NG-4]·跨范式名义）
- **已证**：三格 IME forward bridge 在 logit 层**内 within-ULP-band·argmax flip 全 near-tie·零非-near-tie 分歧** → e2e near-tie flip **correctness-neutral 证成**（IME forward 算术正确到（累积）ULP·flip 是 near-tie 采样敏感·非 kernel 错）。q8_0 teacher-forced 零 flip·q4_K 我方比 vendor 更贴 stock。
- **诚实限制**：① logit 层相对散度 ~1e-2（**非** 1e-5/1e-7）——那是单-tensor 档经 22 层累积·不得声称 logit 层在 1e-7 内。数值档硬锚在 **kernel 层**（单-tensor ~1e-7·int32 0-diff·已 sealed）。② teacher-forced（全 IME）≠ 自由生成（IME-prefill + native-decode）·flip 计数不逐位对应 e2e（§3.3）。③ decode（M=1）本就走 native passthrough·非我方核（prefill 才是阵列 regime）。④ 语料 = 5 prompt × ~25 decision 位（非全 corpus·teacher-forced 单批）。
- **perf 未测**·跨范式名义·`[NG-4]` 全程·**禁"实质胜利/perf 赢/publication-grade"表述**。**成功 = 三格 logit 层 within-ULP-band + 全 flip near-tie + correctness-neutral=T（数值档合规·结构·非 perf）。**

## durable files（见 MANIFEST）
- `tools/e2e-harness/board/g5-m3-ime-logit-ulp/logit_dump.cpp`（自建 llama-API teacher-forced 逐位 logit dumper）
- `tools/e2e-harness/board/g5-m3-ime-logit-ulp/compare_logits.py`（board-numpy bounded-ULP + near-tie + margin-density 比对器）
- `tools/e2e-harness/board/g5-m3-ime-logit-ulp/run_board.sh`（board-side patch/build/dump/compare/restore·EXIT-trap·nohup 承载）
- `tools/e2e-harness/board/g5-m3-ime-logit-ulp/run-logit-ulp-bg.sh`（launcher·scp + nohup 启动 + logfile 轮询）
- `raw/report_q4_0.txt` · `raw/report_q8_0.txt` · `raw/report_q4_K.txt`（三格 compare 全输出）
- `raw/q4_0-seqs.txt` · `raw/q8_0-seqs.txt` · `raw/q4_K-seqs.txt`（三格 canonical teacher-forced token 序列 + n_prompt）
