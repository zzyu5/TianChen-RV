#ifndef TIANCHENRV_CONVERSION_EMITC_TUNABLESCHEDULEOPINTERFACE_TD
#define TIANCHENRV_CONVERSION_EMITC_TUNABLESCHEDULEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def TunableScheduleOpInterface
    : OpInterface<"TunableScheduleOpInterface"> {
  let description = [{
    Interface for TCRV extension-family operations whose kernel SHAPE is selected
    by the plugin-local capability/resource-aware schedule autotuner. An op that
    adopts this interface is auto-discovered by the ONE walk-all materialize pass
    (tcrv-rvv-materialize-schedule): a new tunable op needs no new pass body, it
    only implements the two family-neutral methods below.

    The interface is family-neutral and carries NO plugin tuning types in its
    signature (exactly like TCRVEmitCLowerableOpInterface): it returns ONLY
    primitives the op answers directly. The kernel key the op returns is looked up
    in the plugin-local descriptor registry; the tuning logic stays plugin-local so
    the dialect never depends on the plugin tuning machinery.
  }];

  let cppNamespace = "::tianchenrv::conversion::emitc";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the schedule kernel/family key the tuning record is keyed "
               "on (e.g. \"q4_0\", \"q4_0_q8_0_gemm\").",
      /*retTy=*/"::llvm::StringRef",
      /*methodName=*/"getScheduleKernelKey",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return whether the op already carries a hand-authored shape knob "
               "(the no-clobber predicate; true => leave the op untouched).",
      /*retTy=*/"bool",
      /*methodName=*/"isSchedulePinned",
      /*args=*/(ins)>
  ];
}

#endif // TIANCHENRV_CONVERSION_EMITC_TUNABLESCHEDULEOPINTERFACE_TD
