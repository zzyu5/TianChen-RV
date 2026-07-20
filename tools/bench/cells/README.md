# `tools/bench/cells/` — 每格/每族对拍/计时 harness

《测试与收尾总令-开测篇》§〇.2（ISSUE-090）。**契约**：
- 每格或每族一个 harness，由 `../bench` **按声明接口调用**。
- **harness 自身禁写任何文件** —— runner 只把不可变证据写到
  `experiments/runs/<run-id>/` 与 `experiments/runs.log`；`experiments/master/`
  由 qualification 后的 recon 独占发布，bench/harness 均无写权限。
- 落点法正本 → [`../../../.trellis/spec/measurement/哲学与目的地.md`](../../../.trellis/spec/measurement/哲学与目的地.md) §3.2。
