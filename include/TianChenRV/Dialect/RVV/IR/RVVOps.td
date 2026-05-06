#ifndef TIANCHENRV_DIALECT_RVV_IR_RVVOPS_TD
#define TIANCHENRV_DIALECT_RVV_IR_RVVOPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def TCRV_RVV_Dialect : Dialect {
  let name = "tcrv_rvv";
  let cppNamespace = "::tianchenrv::tcrv::rvv";
  let summary = "TianChen-RV RVV extension dialect skeleton";
  let description = [{
    The RVV dialect is the plugin-local extension surface for RVV execution
    control-plane constructs. The architectural family is tcrv.rvv; the
    concrete MLIR dialect namespace is tcrv_rvv because MLIR dialect
    namespaces cannot contain '.' characters.

    This first slice intentionally exposes only non-compute control-plane
    surface and is registered by RVVExtensionPlugin through the
    ExtensionPluginRegistry path. RVV arithmetic, memory, reduction, lowering,
    emission, runtime ABI, executable behavior, correctness, and performance
    evidence are out of scope for this dialect skeleton.
  }];

  let useDefaultTypePrinterParser = 1;
}

class TCRVRVV_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TCRV_RVV_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TCRVRVV_VLType : TCRVRVV_Type<"VL", "vl"> {
  let summary = "RVV vector-length token type";
  let description = [{
    A non-compute token type representing an RVV vector-length control-plane
    value. This type proves plugin-local RVV dialect registration and syntax
    ownership only; it does not model vsetvl, vector arithmetic, memory,
    runtime ABI, lowering, or executable behavior.
  }];
}

#endif // TIANCHENRV_DIALECT_RVV_IR_RVVOPS_TD
