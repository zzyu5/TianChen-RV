# CLAUDE.md — Weft-RV

本文件只提供 Claude/agent 入口，不复制项目规范。

## 入口与权威

1. 先读 [README.md](README.md) 了解项目定位、构建、测量入口与当前工程方向。
2. 再读 [.trellis/spec/index.md](.trellis/spec/index.md)；只按当前任务进入相关 canon、architecture、measurement、evidence 或 issues 文件。
3. 当前用户明确范围优先；当前代码与测试定义实现事实；spec 定义稳定契约；实验数字以 master/run lineage 为准。

Trellis 是 spec、issue 和可选 task 系统，不是开工许可。小型、可逆、范围明确的修改无需创建 task；多阶段、多 agent、跨层迁移或正式实验 campaign 可按需使用 `.trellis/tasks/`。

## 项目边界

Weft-RV 是 high-level MLIR 之后的 capability-driven execution-layer 参考模板，当前负载域为 ggml/llama.cpp 风格的 RISC-V 量化推理 kernel。它不是通用 tensor compiler，也不引入新的高层 tensor/tile IR。

当前两柱、六律与 C1/C2/C3 组织见 [canon · 暂定科研主张](.trellis/spec/canon/暂定-科研主张.md)；公式与 selector 的唯一工程正本见 [architecture · 变体流水线](.trellis/spec/architecture/变体流水线.md)。不要从旧 task、旧报告或历史 C3′ 标签重新发明研究主张。

## 修改纪律

- 先检查工作树和当前实现，保留用户已有改动。
- 只读与任务相关的 spec，不要求遍历整个 Trellis 树。
- core/common 不写 family-name branch；compute authority 住 typed extension body。
- formula 构造 candidate/plan，legality 限定合法域，selector 在合法候选内选择，emitter 机械实现。
- measurement 不能创造 candidate、绕过 legality 或反向定义 compute。
- 影响稳定契约或暴露新缺口时更新 spec/issues；临时进度不写进稳定 spec。
- 当前请求完成即可交付；只有缺少必要权限、外部条件或重大方向选择时才停下询问。

## 构建与测量

构建、测试与扩展说明见 [README.md](README.md)。正式硬件数字必须走 [measurement](.trellis/spec/measurement/index.md) 规定的 runner、正确性门、四元行键和三目的地：

~~~text
tools/bench/bench <op> <format> --board <board> --engine <engine> --regime <regime>
experiments/master/
experiments/runs/<run-id>/
experiments/runs.log
~~~

本地 build/lit 只能证明编译器与工具链行为；runtime、correctness 和 performance 主张需要相应真硬件证据。
