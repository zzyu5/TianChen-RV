#ifndef WEFT_DIALECT_TOY_IR_TOYOPS_TD
#define WEFT_DIALECT_TOY_IR_TOYOPS_TD

include "mlir/IR/OpBase.td"
include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.td"

def WEFT_Toy_Dialect : Dialect {
  let name = "weft_toy";
  let cppNamespace = "::weft::toy";
  let summary = "Weft-RV Toy extension construction-template dialect";
  let description = [{
    The Toy dialect is a plugin-local extension template surface. The
    architectural family is weft.toy; the concrete MLIR dialect namespace is
    weft_toy because MLIR dialect namespaces cannot contain '.' characters.

    This dialect exposes the family-local typed body constructed for a selected
    variant. The compute skeleton is consumed directly by the Toy backend; it
    does not mirror a generic construction manifest in string attributes.
  }];
}

class WEFTToy_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Toy_Dialect, mnemonic, traits>;

def ComputeSkeletonOp : WEFTToy_Op<"compute_skeleton",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "Toy family-local typed computation body";
  let description = [{
    The Toy formula constructs this typed operation for one selected variant.
    Its MLIR type is the compute identity; generic role/status/interface
    mirrors are intentionally absent.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    OptionalAttr<StrAttr>:$template_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_TOY_IR_TOYOPS_TD
