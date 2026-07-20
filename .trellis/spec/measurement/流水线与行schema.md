# 流水线与行 schema — 3.3 / 3.3.1 / 3.6

> 本层入口与术语见 [index](./index.md)。

## 3.3 一条流水线

**唯一合法真实测量动作：`bench <op> <format> --board <板> --engine <engine> --regime <regime>`。**
> 真跑与 dry-run 使用**同一个全键签名**，都必须携 `(op, format, engine, regime)`；不得省参、猜键、trim/大小写归一或用空值通配。roster 中该精确键必须恰好一条，master 中允许零或一条但不得重复。dry-run 只做预检与计划展示：不分配 run-id，不构造占位 row，不写 `runs.log`、`runs/` 或 master；不存在单参数兼容入口。

> **【runner、目的地和注册 cell 路径已建】** runner 现址 = `tools/bench/bench`；它只写 `experiments/runs/<run-id>/` 与 `experiments/runs.log`。`experiments/master/` 由 recon 独占发布。每格 harness 住 `tools/bench/cells/`。未注册或被板册禁止的组合继续 fail-closed。
>
> **本节是规范，不是对现有实现的描述**：本节的五步与 [3.3.1](#331-行-schema规范性) 行 schema 是**验收标准**；runner 与本节不符 = **runner 的缺陷**。**本层是权威，runner 是实现**——任何"schema 以 runner 实现为准"的读法都非法。（runner 侧已把本节 [3.3.1](#331-行-schema规范性) 的字段表做成**每次出行前机核比对**、不等即 fail-closed 中止；故"改本节而不改 runner"会当场变红，此为设计。）

每格测量 = 固定五步，由**单一 runner** 执行，步内自动留痕，**步外无合法动作**：

1. **静板预飞**：本步的目的是**排除他人/他进程对板的干扰**——载荷阈、单实例锁、零游离进程、CPU 状态哈希。不过即 **VOID-脏板**，不测。
2. **双方板端 clang-18 构建**：我方与对手**同链同板**；编译器世系（板·链·批次）自动写入行，**验收查世系不查字样**。
3. **对拍**（一步三验）：我方 vs oracle、对手 vs oracle 各对一次答案——同时完成三件事：我方没算错、对手没算错、对手就是真实派发的那一个。任一失败 → **不计时**，落对应 VOID（我方错 / 对手废 / 派发不符）。
   顺手自动记录我方产物的**真实向量运算指令数**（配置指令与死值不计）入行——"仪式向量化"自动现形，**记录不拦截**。
   （本步的正确性判据形态见 [正确性门](./正确性门.md)；对手取证与落账见 [3.4](./对手法.md#34-对手法)。）
4. **cold 计时**：预注册重复数，冷启动唯一。
5. **封存 run event**：按 [3.3.1](#331-行-schema规范性) 写 `runs/<run-id>/row.csv`，再向 `runs.log` 追加一次结果。bench 到此结束，不写 master。

采集后由独立控制面执行 `run evidence → qualification view → recon atomic publication`。只有 byte-exact、lineage、freshness、T-N 全齐的 deployed row 才可设 `master_qualified_input=true`；selection-valid 还须成对候选与合法集条件，不能由 master-qualified 自动推断。

T-N 证据必须由 `tools/bench/tn_qualify.py` 从预注册样本生成结构化 JSON，并落在 `experiments/runs/` 的不可变 run 证据域；发布门会重新计算 N≥10、noise IQR、`|Δ| > 2×noise`、确定性 bootstrap 95%CI 排零及所有 source artifact SHA。自由文本出现 `T-N-QUALIFIED` 字样不构成资格，`N=9`、CI 含零、效应未越 2×noise、源哈希漂移均 fail-closed。

**近门补充**：**仅当比值落在 0.7–1.0 区间**，追加一次独立复测确认；离门远的数字不为噪声举行仪式。

### 3.3.1 行 schema（规范性）

**本表 = immutable run event 的权威字段集。** 字段名以本表为准；runner 须逐项绑定。T3 master 是 recon 生成的 board-column view，不得把两种 schema 互称兼容别名。

**行键 = `(op, format, engine, regime)` 四元组**。每个分量显式非空；`engine∈{rvv, ime, scalar}`，`regime∈{micro-fixed, decode, prefill}`。`micro-fixed` 只用于 roster 已登记固定 shape/bindings 的 micro cell；真实 token decode、prefill、e2e 或语义不明历史行不得伪装成 micro-fixed。板不是键：板由 run-id 与 master 属性列承载。

```
python3 -c "import csv;r=list(csv.DictReader(open('experiments/master/T3_master_rebuild.csv')));print(len(r), len({(x['op'],x['format'],x['engine']) for x in r}), len({(x['op'],x['format'],x['engine'],x['regime']) for x in r}))"
```

| 字段 | 内容 | 产出步 |
|---|---|---|
| **op** | **行键①** —— 算子 | 入参（预注册） |
| **format** | **行键②** —— 量化格式，即 `bench <格>` 的「格」 | 入参（预注册） |
| **engine** | **行键③** —— 执行范式 `{rvv | ime | scalar}`（≠ 板） | 入参（预注册） |
| **regime** | **行键④** —— shape 相 `{micro-fixed | decode | prefill}`；[K-10] 结构分立 | 入参（预注册） |
| **cold** | 冷启动计时结果；重复数预注册 | 第 4 步 |
| **判定** | 该格结论；**值域 = [3.3.1.1](#3311-判定-的值域规范性--零预处理机算枚举)**，禁在本栏另立 | 第 4 步 |
| **对手符号** | 该板实际部署派发函数的符号名 | 第 3 步 |
| **对手档** | `{手调 \| 通用向量 \| 标量类 \| 域外 \| N/A-hw}` 之一，逐格证据判定（[3.4](./对手法.md#34-对手法)） | 第 3 步 |
| **对手证据引用** | 证成"部署事实"与档位的证据指针（反汇编 dispatch / 类型注册表） | 第 3 步 |
| **我方向量指令数** | 我方产物的真实向量运算指令数（配置指令与死值不计）；**记录不拦截** | 第 3 步 |
| **世系** | 编译器身份三元 `{板·链·批次}`，自动写入；**验收查世系不查字样** | 第 2 步 |
| **run-id** | 本次运行标识，兼作原始产物目录名与行内引用锚 | 全步 |
| **噪声标** | 近门标记（比值落 0.7–1.0 → 已追加独立复测确认） | 近门补充 |

**VOID 的 run event 不存在**：VOID 不产 `row.csv`，只在运行台账留原因；更不会进入 qualification 或 master。

#### 3.3.1.1 `判定` 的值域（规范性 · 零预处理机算枚举）

**master raw 值域 = 下列 12 值 ∪ VOID 四值**。runner 的普通新 run event 只产其中的 `PASS` / `具名-X`（0.8 门）；近门信息住 `噪声标`，不再产生 `WIN` / `LOSS` / `near-parity` 别名。specialized disposition 由合格生成链保留原 raw token，禁预处理归并。

- `PASS`
- `具名-X`
- `PASS(decode-M1-GEVM)`
- `具名-X(decode-M1)`
- `pending-真(待补标量仗)`
- `pending-fold`
- `域外-永久(q1_0)`
- `N/A-hw`
- `PASS-DEPLOYED(decode-GEVM·C1)`
- `具名-X(IME-kernel-sym·vendor手调IME)`
- `pending(照测未定verdict)`
- `pending(IME结构·该板无合法对手·q8_0落RVV-repack回退·缺vendor-IME对手·探针证据)`

**谓词**（可复跑 · **零预处理**：不 `split`、不剥括注、不 `startswith` 归口）：

```
python3 -c "import csv,collections;r=list(csv.DictReader(open('experiments/master/T3_master_rebuild.csv')));c=collections.Counter();[(c.update([x['rvv_disp']]),c.update([x['k1_disp']])) for x in r];print(len(c), sum(c.values()));print(c.most_common())"
```

**取值一致性** = [3.6](#36-其余铁律) 的值域登记铁律：本节是**枚举**，不是模板 —— 新值只能经 [3.5](./门体系.md#35-门体系清算三硬点其余皆工具) 第 4 条途径进来。

## 3.6 其余铁律

（保留为流水线规则，非仪式。）

- **禁继承**：decode 一律 **M=1 实测**；IME 禁承 native。
- **VOID 三出口限期清偿** `{重测 | 降档对局 | 环后具名}`，**禁躺平**。
- **板忙只延后验证，不延后施工**——"constructed·待板端门"是**合法状态**。
- **"某物不存在"的断言禁用 `head` / 固定窗口命令作依据。**
- **★值域登记只许机算枚举，禁任何预处理归并；归并即判据，判据即裁。**（2026-07-17 用户裁 · 通用法 · **本条即正本**，[governance · 决策权限卡](../governance/决策权限卡.md) 的「★本判别式的来历」节指来。**病理**：一句 `.split('(')` 就能把 `PASS(decode-M1-GEVM)` 无声并进 `PASS`，raw distinct 被压低，**再戴着"实况"的帽子出场** —— 归并动作**可以完全无意识地混进"机算"里**。故：登记值域时**先出 raw distinct 全表**；要归并 = 先答"哪种算对" = 判据级 = 用户裁。）
- **预注册先于测量。**
- **gcc 不存在**：不作为构建链或对局链；仅作为历史素材与链接运行库存在。本地 clang-20 **仅限不碰代码生成的辅助**；**二进制级数字零证据地位**。
