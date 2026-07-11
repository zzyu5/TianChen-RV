# G5-M3 接线机制文档化收口

parent: `07-11-G5-wiring`（G5 接线战役）· status: **planning**（占位·**gated**）。

## Gate
gated on M2 铺线（L①/L②/L③ 各档接线积累足够样例）。收口 = 文档化，不引入新绿格。

## 内容
接线机制文档化收口：把 G5 三级接线分层（L①链路层 / L②新建上游 scaffold / L③跨框架 forward bridge）沉淀为可复制协议文档，串起 M0 接线机制解剖（dispatch gate + kernel arch/riscv + #include emitted .inc 挂点③）、M1b correctness-carrier 正例、M2 各格 scaffold 建法与 GAP 台账。

## 边界
- 接线 = 补丁/链接层集成（NG-2 不动·不做图框架）；接线 ≠ 自动转绿（micro↛e2e 铁律仍管辖）。
- 文档化产物落 docs/canon 或 experiments casefile（措辞 canon 级变更属必问②）。

（占位任务·M3 收口·细节待 M2 铺线样例齐备后展开。）
