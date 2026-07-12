#ifndef WEFT_CONVERSION_EMITC_WEFTEMITCLOWERABLEOPINTERFACE_TD
#define WEFT_CONVERSION_EMITC_WEFTEMITCLOWERABLEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def WEFTEmitCLowerableOpInterface
    : OpInterface<"WEFTEmitCLowerableOpInterface"> {
  let description = [{
    Interface for WEFT extension-family operations that can provide bounded
    provenance for the common EmitC lowerable route.

    The interface identifies the typed source operation and source role used
    when constructing a WEFTEmitCLowerableRoute. It does not encode descriptor
    semantics, hardware configuration, runtime correctness, performance
    evidence, or family-specific intrinsic names.
  }];

  let cppNamespace = "::weft::conversion::emitc";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the typed source operation name for EmitC route provenance.",
      /*retTy=*/"::llvm::StringRef",
      /*methodName=*/"getWEFTEmitCLowerableSourceOpName",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getName().getStringRef();
      }]>,
    InterfaceMethod<
      /*desc=*/"Return the source role for EmitC route provenance.",
      /*retTy=*/"::llvm::StringRef",
      /*methodName=*/"getWEFTEmitCLowerableSourceRole",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return "compute";
      }]>
  ];
}

#endif // WEFT_CONVERSION_EMITC_WEFTEMITCLOWERABLEOPINTERFACE_TD
