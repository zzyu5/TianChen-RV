# E6 — 覆盖率四指标脚本 + 分母定稿 [COV-1/2] + ledger 脚本 [LED-1]

> **父 program:** `07-02-full-refactor`。**advisor 点的 leverage 模块**:E6 产出的是**度量尺子**,让引擎线 **G1(C_construct 强义燃减)可被打分**——没尺子无法给 G1 打分。两条 load-bearing 输出:**分母 of record(roster)** + **C_construct(强义)定义**;其余是枚举 + plumbing。净新 Python(低 build 风险,像 E1),不动 C++/ODS。
> **权威:** 科研总纲 [COV-1](分母=本负载域 kernel roster,A/B/C;low-bit 已移入分子)/[COV-2](四指标)/[COV-3](目标值)/[K-4](六态)/[LED-1](每家族 ledger)。执行总纲 §2/§3/§8。
> **research(已完成):** `research/{coverage-denominator,six-state-current,metric-and-ledger-script-design,e6-scope-summary}.md`。快照 = HEAD `1bbab882`。

---

## Decisions(ADR-lite;research 7 escalation 全部按 总纲-grounded 默认定案,roster 是数据、用户可后续 toggle class 无需改码)

1. **ggml pin = `6eab47181cbd3532c88a105682b81b4729ab809b`(6eab471)** —— sibling checkout `/home/kingdom/phdworks/llama.cpp @ master`(**无 tag、非 submodule、出仓库**),已是 de-facto routing/baseline authority(archived routing-audit 引用)。roster `$meta` 记 `ggml_pin` + `ggml_ref_note` + `epoch:1`(实验总纲纪元纪律:上游加格式→开新纪元 + 曲线断点)。⚠ 是 raw SHA 对仓外 checkout——用户可后续换 tagged upstream 或 vendor format enum 进仓(留 note)。
2. **分解族键策略 = (a) 独立键**(research 强推;(b) 污染 G1 燃减)。`vec_dot` 与 `product_reduce` 是**不同 op**、独立分母键;q4_0 同时以 `vec_dot`(weak)与 `product_reduce`-q4_0-nibble(strong)两键存在,**不 merge**(了结执行总纲 §2"q4_0 双路"note)。数值后果:**block-dot vec_dot C_construct 强义 = 0/24 今天**,分解族 = 3/3 强义(独立 bucket);G1 燃 24 键。roster doc note 记 (b) 为被拒方案,防未来 merge。
3. **B类边界 = B-pending(分母=9)**:[COV-1] 列全 9 前向算子为 B类;absent 的 gelu/add/mul/cpy **留分母、state=absent**(诚实拉低 C_dispatch,不靠排除藏未覆盖算子)。
4. **边缘 low-bit q1_0/nvfp4 = A**(in-code 各自单体;裁决把 low-bit 移入分子)。全 IQ/ternary/fp4 vec_dot = A。
5. **quantize_row/dequantize_row/GEMM-tile roster 宽度**:按 [COV-1] 枚举——quantize_row 每 A-format(至少 q8_0/q8_1/q8_K activation quantizer)、dequantize_row 每 A-format、RVV GEMM tile 每格式、IME×{q4_0,q8_0,q4_K}。**in-code state 反映现实**(q8_0 quantize 在;5 RVV GEMM 格式 q4_0/q4_1/q4_K/q5_0/q8_0 在;IME 是 format-agnostic MMAOp、format-keyed tile 是 aspirational roster 目标非 in-code op;其余 absent)。
6. **ledger test_LOC = 显式每家族 test-file manifest(钉死),非 path-glob**(659 用 glob 不复现,glob ime-*.mlir≈230)。**headline LED-1 = code_LOC(cloc-approx,可复现)**;test_LOC 单列、经 manifest(不确定则标 manifest-pending + 注 glob 近似值)。
7. **cloc = 近似实现**(`which cloc` 空;近似验证 exact:IME 2484 raw / 1866 cloc-approx ≈ 目标 1865)。注"cloc 未装;装 cloc 得 [A-5] 规范措辞"。

---

## Scope(IN;全 net-new,不动 C++/ODS)

- **`schema/coverage-roster.v1.json`**(**分母 of record**,canonical JSON,复用 E1 governance home):`$meta`(roster_version/ggml_pin/epoch/denominator_key=`(op,format[,shape_class])`/decomposed_family_policy=SEPARATE-KEYS)+ `kernels[]`(A类 24 vec_dot + 3 decomposed + quantize/dequantize_row 每 A-format + GEMM tile RVV/IME、B类 9 前向、C类 flash-attn/bf16;每条 `{op,format,class,bucket[,engine][,reason]}`)。roster = 稳定分母(含未建 kernel,absent 留分母拉低覆盖)。
- **`schema/coverage-sixstate.v1.json`**(**手工标注六态表**,numerator input,时变;与 roster 分离):seeded from `research/six-state-current.md`(17 dispatch-wired / 7 constructed-weak / 3 strong constructed + 5 wired 前向 + absent 项);每 C_construct 相关行带 **`auto_readout: "pending-E5"`**(六态自动读出 = E5 provenance 清单,不 block E6)。
- **`.trellis/scripts/coverage_metrics.py`**(stdlib,镜像 E1 风格):读 roster + sixstate → 四指标 **C_dispatch**(≥dispatch-wired/分母)/ **C_construct 强义**(≥constructed 强义/分母,燃减 headline,只计强义)/ **C_construct+ 弱义**(≥constructed-weak/分母,报告不 gate)/ **C_attr 分级**(^CT/装载/^RT);输出 CI-report artifact 带 snapshot-ID `$meta`;`--self-test`(合成 fixture)。
- **`.trellis/scripts/family_ledger.py`**(stdlib,cloc-approx):读 family→dirs manifest + git 历史 → 每家族 `{code_LOC(cloc-approx,不含测试), test_LOC(显式 manifest,单列), table_rows, interface_touchpoints, calendar_days}`;**复算 IME 首点 = 2484 raw / 1866 cloc-approx**;`--self-test`(合成串 + 固定日期)。

## 验收(可机检;纯脚本,无 build)

1. **分母 of record 落盘**:`coverage-roster.v1.json` 合法 canonical JSON,含 A(24 vec_dot + 3 decomposed + quant/gemm)/B(9)/C 全 roster,`$meta` 带 ggml_pin 6eab471 + epoch;decomposed=独立键(q4_0 两键不 merge)。
2. **四指标出数 + 快照 ID**:`coverage_metrics.py` 读 roster+sixstate 产四指标 + snapshot `$meta`;**C_construct 只计强义**(block-dot vec_dot=0/24),弱义单列 C_construct+(=7/24);C_construct 行带 `pending-E5`。两次运行同结果(确定性)。
3. **ledger 复算**:`family_ledger.py` 对 IME 复算 code_LOC=2484 raw / 1866 cloc-approx(≈1865);test_LOC 单列(manifest 或标 pending);table_rows/touchpoints/calendar_days 有值。
4. **两脚本 `--self-test` 全绿**。
5. **数字不进 spec**:输出是 CI-report artifact(带快照 ID),不写进 `.trellis/spec/`([GOV-1]/[GOV-3]);工件 JSON + 报告可提交回仓(roster/sixstate 入 schema/,报告入执行总纲/CI)。
6. **零越界**:diff 仅 `schema/coverage-*.json` + `.trellis/scripts/coverage_metrics.py`/`family_ledger.py`(+ 可选 family→dirs manifest);**不动任何 C++/ODS/lit**;Python 仅 tooling。

## Out of Scope

- **六态自动读出([K-4] 状态机)+ [L-8] 机检执法(弱充强 CI 拦)= E5**(provenance 清单;实验总纲"provenance 清单先于六态自动化")。E6 六态 input 是手工标注表 + `pending-E5` 标记,不从代码判强/弱。
- **CI/`.github` 接线 = E3**(F-1/F-5 CI thrust)。E6 只产报告 artifact + 脚本;CI 调用下游。
- **边际成本曲线 [LED-2]**(需 ≥3 家族/X-SCALAR)、**perf/燃减曲线 [COV-4/5]**(需硬件/归因 JSONL)。
- **目录归拢 E2b**:family→dirs manifest 是**临时手维护** input,直到 E2b 落 `plugins/<family>/`。
- 不装 cloc(用近似;用户可后装得规范措辞)。

## Technical Notes

- roster 住 `schema/`(E1 governance home,与 capability.schema.v1.json/VERSIONLOG.md 并置);可选复用 `check_schema_gate.py` 的 canonical-hash idiom 给 roster/sixstate content-address。
- **⚠ 并行写入者纪律**([[full-refactor-program-active]]):git author zzyu5 同 branch 活动;动共享文件前 re-read、不 commit 对方 in-flight 编辑、提交后 grep 验证共存。E6 全 net-new 文件,冲突面小。
- Python stdlib only(implementation-stack 红线);镜像 E1 的 `check_schema_gate.py` 风格 + `--self-test` 纪律。数字带快照 ID(HEAD 1bbab882),随仓演进。
