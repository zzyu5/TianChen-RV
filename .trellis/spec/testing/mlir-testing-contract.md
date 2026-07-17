# MLIR Testing Contract（条文已分迁六层 · 本文只是指路牌）

> **本文的条文已全部迁入六层，此处不再有条文。** 只保留路径以免既有引用腐坏成假引文。
> **禁在本文添加或修改任何规则**：改规则去目标文件；此指路牌随 spec 树归并收口一并移除（去向属 `ISSUE-070`，见 [issues](../issues/index.md)）。

## 条文新住址（逐节）

| 原节 | 新住址 |
|---|---|
| 测试类型与用途（lit/FileCheck · C++ tests · Runtime/Hardware Evidence）· 给 agent 的判断点 | [governance · 思维准则](../governance/思维准则.md) §九 测试形态 |
| Hardware Evidence —— 一个通用契约 · Performance-Comparison 证据 · 测试断言的来源区分（`HARNESS` 源级 vs 运行期） | [canon · 正确性与证书](../canon/正确性与证书.md)（硬件证据通用契约 · 断言来源区分） |
| 格 schema 二分（测量格 / 结构证明格）+ 状态枚举 `{measured\|stale\|board-pending\|open\|n_a}` + 三条铁律 | [canon · 测量判据](../canon/测量判据.md) §二.7 格 schema 二分与 stale 律 |
| T-N 噪声地板 | [canon · 测量判据](../canon/测量判据.md) §二.5；T-N 表工件指针 → [evidence · 三贡献证据地图](../evidence/三贡献证据地图.md) §5.6 |
| T-X X-SCALAR 真板 enablement 证据表（六列 + 措辞铁线 + [F-6] 防混淆） | [canon · 测量判据](../canon/测量判据.md) §六 T-X；活证据工件指针 → [evidence · 三贡献证据地图](../evidence/三贡献证据地图.md) §2.2 |
| 对手解析探针 —— 对手类（闭合枚举） | [canon · 对手与档位](../canon/对手与档位.md) §一「对手类词表」；探针取证要件（无探针工件 = INVALID）→ [canon · 测量判据](../canon/测量判据.md) §二.8 构建保真三重 |
| [F-1] 零分支 falsifier —— manifest + 判读规程 | [canon · 能力模型与插件协议](../canon/能力模型与插件协议.md) [F-1]；门本体与 manifest 工件 → [evidence · 三贡献证据地图](../evidence/三贡献证据地图.md) §2.2 |
| 正确性门 [K-5]（byte-exact 先于计时 + 浮点 ULP 上界） | [measurement · 正确性门](../measurement/正确性门.md)（权威定义处）· 法条形态 → [canon · 正确性与证书](../canon/正确性与证书.md) [K-5] |
| [PERF-1] 击败门（测试证据可承担的子集） | [canon · 测量判据](../canon/测量判据.md) §一.2（门体 + 测试证据可承担的子集）。**门数称谓待裁 = `ISSUE-071`；引用一律用 `[PERF-1]`、禁带门数。** |
