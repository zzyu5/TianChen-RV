#ifndef TIANCHENRV_DIALECT_TENSOREXTLITE_IR_TENSOREXTLITEOPS_TD
#define TIANCHENRV_DIALECT_TENSOREXTLITE_IR_TENSOREXTLITEOPS_TD

include "mlir/IR/OpBase.td"
include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.td"

def TCRV_TensorExtLite_Dialect : Dialect {
  let name = "tcrv_tensorext_lite";
  let cppNamespace = "::tianchenrv::tcrv::tensorext_lite";
  let summary = "TianChen-RV TensorExtLite extension metadata dialect";
  let description = [{
    The TensorExtLite dialect is a plugin-local extension fragment surface. The
    architectural family is tcrv.tensorext_lite; the concrete MLIR dialect namespace is
    tcrv_tensorext_lite because MLIR dialect namespaces cannot contain '.' characters.

    This dialect intentionally exposes only selected lowering-boundary metadata
    plus a minimal ODS-backed role-op boundary for plugin integration tests. It
    does not model tensor/tile computation, executable lowering, runtime calls,
    binary artifacts, correctness, or performance.
  }];
}

class TCRVTensorExtLite_Op<string mnemonic, list<Trait> traits = []>
    : Op<TCRV_TensorExtLite_Dialect, mnemonic, traits>;

def LoweringBoundaryOp : TCRVTensorExtLite_Op<"lowering_boundary", []> {
  let summary = "fail-closed TensorExtLite selected lowering boundary";
  let description = [{
    Records the TensorExtLite plugin-owned selected lowering boundary for a chosen
    tcrv.exec.variant. The op carries source kernel, selected variant, origin
    plugin, selected-path role, no-active-route status, required capability
    references, and TensorExtLite fragment metadata.

    This op is not a compute operation and is not an executable lowering step.
    It exists to prove that a future extension plugin can own a dialect-local
    boundary while generic core passes route through the registry.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$fragment_abi,
    StrAttr:$handoff_kind,
    OptionalAttr<StrAttr>:$fragment_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

class TensorExtLiteRoleSkeletonOp<string mnemonic, string roleName>
    : TCRVTensorExtLite_Op<mnemonic,
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "minimal TensorExtLite ODS " # roleName # " role-op boundary";
  let description = [{
    Records a minimal TensorExtLite extension-family role op in ODS. The op
    proves that a concrete non-RVV extension family can materialize typed MLIR
    role operations implementing the generated EmitC-lowerable provenance
    interface and have TensorExtLite construction validation consume that IR
    identity.

    This op is a construction-protocol role boundary consumed by the bounded
    TensorExtLite EmitC route. It does not model tensor/tile computation,
    target artifact export, runtime correctness, or performance.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$typed_role,
    I64Attr:$role_order,
    StrAttr:$source_role,
    StrAttr:$role_specific_interface,
    OptionalAttr<StrAttr>:$fragment_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def ConfigSkeletonOp : TensorExtLiteRoleSkeletonOp<
    "config_skeleton", "config">;
def LoadFragSkeletonOp : TensorExtLiteRoleSkeletonOp<
    "load_frag_skeleton", "load_frag">;
def TileMmaSkeletonOp : TensorExtLiteRoleSkeletonOp<
    "tile_mma_skeleton", "tile_mma">;
def StoreFragSkeletonOp : TensorExtLiteRoleSkeletonOp<
    "store_frag_skeleton", "store_frag">;

#endif // TIANCHENRV_DIALECT_TENSOREXTLITE_IR_TENSOREXTLITEOPS_TD
