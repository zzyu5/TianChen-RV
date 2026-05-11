#ifndef TIANCHENRV_CONVERSION_EMITC_TCRVEMITCLOWERABLEOPINTERFACE_TD
#define TIANCHENRV_CONVERSION_EMITC_TCRVEMITCLOWERABLEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def TCRVEmitCLowerableOpInterface
    : OpInterface<"TCRVEmitCLowerableOpInterface"> {
  let description = [{
    Interface for TCRV extension-family operations that can provide bounded
    provenance for the common EmitC lowerable route.

    The interface identifies the typed source operation and source role used
    when constructing a TCRVEmitCLowerableRoute. It does not encode descriptor
    semantics, hardware configuration, runtime correctness, performance
    evidence, or family-specific intrinsic names.
  }];

  let cppNamespace = "::tianchenrv::conversion::emitc";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the typed source operation name for EmitC route provenance.",
      /*retTy=*/"::llvm::StringRef",
      /*methodName=*/"getTCRVEmitCLowerableSourceOpName",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getName().getStringRef();
      }]>,
    InterfaceMethod<
      /*desc=*/"Return the source role for EmitC route provenance.",
      /*retTy=*/"::llvm::StringRef",
      /*methodName=*/"getTCRVEmitCLowerableSourceRole",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return "compute";
      }]>
  ];
}

#endif // TIANCHENRV_CONVERSION_EMITC_TCRVEMITCLOWERABLEOPINTERFACE_TD
