# `tools/bench/cells/` — 每格/每族对拍/计时 harness

《测试与收尾总令-开测篇》§〇.2（ISSUE-090）。**契约**：

- 每格或每族一个 harness，由 `../bench` **按声明接口调用**。
- runner 只通过 `tools/bench/bench` 内的显式 `CELL_ROUTES` 选择 harness；不再按
  `cells/<op>.sh` 猜文件。verify/cold 解析器也各自只有一个按 op 精确命中的 registry，
  未命中不得回退 GEMM。
- `roster key 存在`、`route known`、`parser covered`、`run eligible` 是四件事。尤其
  `vec_dot/tq2_0/scalar/scalar` 的 route/parser 只是 dormant contract；roster 无该键且
  ISSUE-061/104 未解，因此不会触发 SSH、不会获得性能资格。
- **harness 自身禁写任何文件** —— runner 只把不可变证据写到
  `experiments/runs/<run-id>/` 与 `experiments/runs.log`；`experiments/master/`
  由 qualification 后的 recon 独占发布，bench/harness 均无写权限。
- 落点法正本 → [`../../../.trellis/spec/measurement/哲学与目的地.md`](../../../.trellis/spec/measurement/哲学与目的地.md) §3.2。

当前显式 route 面：

- `gemm_tile`：现有 grid harness 的七格式，`rvv` engine，rvv/k1 板；
- `vec_dot`：`q2_K`–`q6_K`，`rvv` engine，rvv/k1 板；
- `product_reduce`：三种 registered sub-primitive，`rvv` engine，rvv/k1 板；
- `scalar_vec_dot`：仅 dormant `vec_dot/tq2_0/scalar/scalar` route，不授予真跑资格。

目录里存在其他脚本不自动扩大 route 面；扩大必须同时更新 registry、parser contract、
负控和相应 issue/task，原子合入，不保留旧 route 或兼容 alias。
