#ifndef WEFT_CONVERSION_EMITC_TUNABLESCHEDULEOPINTERFACE_TD
#define WEFT_CONVERSION_EMITC_TUNABLESCHEDULEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def TunableScheduleOpInterface
    : OpInterface<"TunableScheduleOpInterface"> {
  let description = [{
    Interface for WEFT extension-family operations whose final kernel shape is
    constructed by a plugin-local capability/resource-aware formula. An op that
    adopts this interface is auto-discovered by the walk-all construction pass:
    a new tunable op needs no new pass body, only the two family-neutral methods
    below and a formula registration.

    The interface is family-neutral and carries NO plugin tuning types in its
    signature (exactly like WEFTEmitCLowerableOpInterface): it returns ONLY
    primitives the op answers directly. The kernel key the op returns is looked up
    in the plugin-local formula registry; construction knowledge stays
    plugin-local so the dialect never depends on plugin tuning machinery.
  }];

  let cppNamespace = "::weft::conversion::emitc";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the schedule kernel/family key the tuning record is keyed "
               "on (e.g. \"q4_0\", \"q4_0_q8_0_gemm\").",
      /*retTy=*/"::llvm::StringRef",
      /*methodName=*/"getScheduleKernelKey",
      /*args=*/(ins)>,
    InterfaceMethod<
      /*desc=*/"Return whether the op carries every final schedule field required "
               "by its structural schema. This does not validate or select the "
               "field values.",
      /*retTy=*/"bool",
      /*methodName=*/"hasCompleteSchedule",
      /*args=*/(ins)>
  ];
}

#endif // WEFT_CONVERSION_EMITC_TUNABLESCHEDULEOPINTERFACE_TD
