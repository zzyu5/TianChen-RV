#ifndef WEFT_CONVERSION_EMITC_WEFTEMITCLOWERABLEOPINTERFACE_TD
#define WEFT_CONVERSION_EMITC_WEFTEMITCLOWERABLEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def WEFTEmitCLowerableOpInterface
    : OpInterface<"WEFTEmitCLowerableOpInterface"> {
  let description = [{
    Interface for exact WEFT extension-family body operations that provide
    bounded source provenance to current EmitC artifact lowering.

    The interface identifies the typed source operation and source role used
    for artifact diagnostics. It does not construct a route, encode descriptor
    semantics, choose hardware configuration, or carry runtime correctness,
    performance evidence, or family-specific intrinsic names.
  }];

  let cppNamespace = "::weft::conversion::emitc";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the typed source operation name for EmitC artifact provenance.",
      /*retTy=*/"::llvm::StringRef",
      /*methodName=*/"getWEFTEmitCLowerableSourceOpName",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getName().getStringRef();
      }]>,
    InterfaceMethod<
      /*desc=*/"Return the source role for EmitC artifact provenance.",
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
