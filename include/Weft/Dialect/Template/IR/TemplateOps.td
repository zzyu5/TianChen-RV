#ifndef WEFT_DIALECT_TEMPLATE_IR_TEMPLATEOPS_TD
#define WEFT_DIALECT_TEMPLATE_IR_TEMPLATEOPS_TD

include "mlir/IR/OpBase.td"
include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.td"

def WEFT_Template_Dialect : Dialect {
  let name = "weft_template";
  let cppNamespace = "::weft::template_ext";
  let summary = "Weft-RV Template extension dialect";
  let description = [{
    The Template dialect is a plugin-local extension template surface. The
    architectural family is weft.template; the concrete MLIR dialect namespace is
    weft_template because MLIR dialect namespaces cannot contain '.' characters.

    This dialect exposes the family-local typed body constructed for a selected
    variant. The compute_skeleton is consumed directly by the Template backend;
    it is not a metadata-only boundary or a generic role-manifest replay.
  }];
}

class WEFTTemplate_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Template_Dialect, mnemonic, traits>;

def ComputeSkeletonOp : WEFTTemplate_Op<"compute_skeleton",
    [DeclareOpInterfaceMethods<WEFTEmitCLowerableOpInterface,
      ["getWEFTEmitCLowerableSourceOpName",
       "getWEFTEmitCLowerableSourceRole"]>]> {
  let summary = "Template family-local typed computation body";
  let description = [{
    The Template formula constructs this typed operation for one selected
    variant. Its MLIR type is the compute identity; generic role, status,
    interface and capability mirrors are intentionally absent.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    OptionalAttr<StrAttr>:$template_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_TEMPLATE_IR_TEMPLATEOPS_TD
