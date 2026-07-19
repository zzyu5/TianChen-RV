# PRD — W3b 拔管道收敛（Stop-hook 反馈:W3 硬门②未满足·续拔 8 消费者）

## 缘起（Stop-hook 驳回「fork 放走」）
W3(848d61f31) 只收敛 C2(strip-width)·FrontDoor 6 消费者 + ScheduleDescriptorRegistry 仍各自 `deriveMinimumVLEN(march)` re-parse。我曾记为「threading fork 延后」——违哲学锁③(不停在记边界)+硬门②(禁 funnel 过权威放走)。本 task 真收敛。

## 病根（本 task 新诊断·比 W3 更深）
production 里**整个类型化能力层从不物化**：`minimum_vlen`/`supported_sew`/`rvv_version` 只由探测层 pass stamp·而探测层 pass **仅测试调用·生产 pipeline 零挂接**。消费者全部 re-derive march。"类型化能力层=空转招牌·承重的水从 march 旁路流" 坐实。

## 做（单一 resolver seam·非重排 pipeline/改输入契约）
`resolveRVVMinimumVLEN(module, march, hints)`(RVVCapabilityProfile)= **先读 provider fact(readRVVProviderMinimumVLEN)·仅 absent 才 fallback derive**。8 消费者全改调它:
- Codebook/Dequant/PackedI4/Reduction selectXCoreLMUL(+module 参数) · Monolithic materializeKernel · LowerQuantContraction ×2(stampScheduleSelections/lowerOne) · ScheduleDescriptorRegistry。

## 三硬门结果
- **② grep 收敛 GREEN**：FrontDoor 消费者 deriveMinimumVLEN=**0**·Schedule 消费者=**0**·全库真调用只剩权威层 3 处(deriveHasZvl128b 内部 / resolver fallback / 探测层生产者)。消费者 ~8→0。
- **① 判决实验 GREEN(2 处 post-construction 消费者)**：
  - strip-width(W3 decisive·已在): provider vlen=256 ⊥ march=128 → half_lanes=16(跟 provider)。
  - **LowerQuantContraction(本 task 新 decisive)**: provider minimum_vlen=256 ⊥ march=rv64gcv(128) → contraction_algorithm=**"block-dot"**(256形·跟 provider)·基线 march=128→"repack"。lit `rvv-lower-quant-contraction-provider-vlen-decisive.mlir` PASS。
- **③ byte-exact**: 正常输入 provider absent→resolver fallback 同值·RVV lit **533/533 PASS**(Conversion/Transforms/Plugin/Target)。

## ★诚实架构边界(本 task 核心新发现·更正 W3 假设)
W3 PRD 假设「~10 消费点均可读 provider op」——**部分证伪**。判别:
- **post-construction 消费者**(strip-width/ScheduleDescriptorRegistry/LowerQuantContraction·处理已构造的带 provider 的 op)= 真读 provider·判决实验可证·**pull LIVE**。
- **5 个 SourceFrontDoor CONSTRUCTOR**(Codebook/Dequant/PackedI4/Reduction/Monolithic)= **物化边界**·运行在 source-only IR(实测报错"requires RVV source-only MLIR input·pre-existing capability op 拒收")·**此时 provider op 尚不存在**(它们是 capability-carrying kernel 的生产者)。march 是它们合法的构造时配置输入·resolveRVVMinimumVLEN 对它们 fallback march(无 provider 可读)。seam 已统一(IF 未来 pipeline 在其前物化 provider·自动读)·但当前 **pull 已接线但休眠**(边界处无 fact 可读)。

⟹ grep gate ② 已满足(消费者不再直调 deriveMinimumVLEN)。判决实验对 post-construction 消费者 LIVE+证毕。SourceFrontDoor 的 march-use = 构造边界·非旁路(它们是 provider 的生产者·不是 re-parse 已构造的 fact)。

## 未尽(honest·非放走)
- SourceFrontDoor 的 provider-读 要 LIVE·须在源前门前物化 provider(source-only 契约放松 或 module-attr 通道)= 独立架构决策·gated on 是否值得(边界处 march==capability-file·功能等价)。
- deriveRVVVersion(march) 轴未收敛(本 task scope=deriveMinimumVLEN·hook ② 所指)。

## 账面
- 改文件: RVVCapabilityProfile.{h,cpp}(resolver) + 7 消费者 cpp + 1 新 decisive lit。0 造数·判决实验自证·byte-exact 533/533。
