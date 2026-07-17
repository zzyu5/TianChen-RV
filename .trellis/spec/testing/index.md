# Testing Specs（本层条文已分迁六层 · 本文只是指路牌）

> **本层不是第七层。** 条文已全部迁入六层，此处**不再有条文**；只保留路径以免既有引用腐坏成假引文。
> **禁在本层新增或修改任何规则**：改规则去目标文件。本层的归并 / 归档去向属 `ISSUE-070`（canon 级 · 待裁），见 [issues](../issues/index.md)。

## 去哪里

- **逐节新住址表** → [MLIR Testing Contract 指路牌](./mlir-testing-contract.md)。
- **要加测试 / 选 lit 还是 C++ / 什么算 runtime 证据** → [governance · 思维准则](../governance/思维准则.md) §九 测试形态。
- **硬件证据通用契约 · 性能对比证据的更高门槛 · 断言来源区分（`HARNESS` 源级 vs 运行期）** → [canon · 正确性与证书](../canon/正确性与证书.md)。
- **证据格 schema 与状态枚举 · T-N 噪声地板 · T-X · [PERF-1] 门体** → [canon · 测量判据](../canon/测量判据.md)。
  （**门数称谓待裁 = `ISSUE-071`；引用一律用 `[PERF-1]`、禁带门数。**）
- **对手类词表（**角色**）与三档法（**档位**）· 探针取证要件** → [canon · 对手与档位](../canon/对手与档位.md)、[canon · 测量判据](../canon/测量判据.md) §二.8。
- **[F-1] 判读规程与 manifest 形态** → [canon · 能力模型与插件协议](../canon/能力模型与插件协议.md)。
- **正确性门 [K-5]** → [measurement · 正确性门](../measurement/正确性门.md)。
- **legacy `RVVI32M1*` / `rvv-i32m1-*` 正例禁令 · source-front-door fail-closed** → [architecture · 发射与降级](../architecture/发射与降级.md) §Legacy i32m1 Route-Table Policy、[canon · 核心不变量](../canon/核心不变量.md) I7。
- **Python 只做 tooling、不替代 MLIR 行为测试** → [canon · 核心不变量](../canon/核心不变量.md) I6。
