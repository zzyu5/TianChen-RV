#ifndef TIANCHENRV_DIALECT_RVV_IR_RVVOPS_TD
#define TIANCHENRV_DIALECT_RVV_IR_RVVOPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.td"

def TCRV_RVV_Dialect : Dialect {
  let name = "tcrv_rvv";
  let cppNamespace = "::tianchenrv::tcrv::rvv";
  let summary = "TianChen-RV RVV extension dialect skeleton";
  let description = [{
    The RVV dialect is the plugin-local extension surface for RVV execution
    control-plane constructs. The architectural family is tcrv.rvv; the
    concrete MLIR dialect namespace is tcrv_rvv because MLIR dialect
    namespaces cannot contain '.' characters.

    The current bounded slice exposes the tcrv_rvv.setvl runtime AVL-to-VL
    control-plane surface, the matching tcrv_rvv.with_vl VL scope region, and
    generic typed RVV vector-level dataflow ops nested under explicit with_vl
    scopes. Deprecated finite i32 parse-only inventory may still appear in
    negative or verifier fixtures, but it is not executable route, ABI,
    EmitC-lowerable, selected-body, artifact, source-front-door, or exporter
    authority. Current RVV execution goes through explicit generic typed
    extension-family IR plus a provider-built materialized EmitC route.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class TCRVRVV_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TCRV_RVV_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class TCRVRVV_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TCRV_RVV_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class TCRVRVV_Op<string mnemonic, list<Trait> traits = []>
    : Op<TCRV_RVV_Dialect, mnemonic, traits>;

def TCRVRVV_TailAgnostic
    : I32EnumAttrCase<"Agnostic", 0, "agnostic">;
def TCRVRVV_TailUndisturbed
    : I32EnumAttrCase<"Undisturbed", 1, "undisturbed">;

def TCRVRVV_TailPolicy : I32EnumAttr<
    "TailPolicy", "finite RVV tail element policy",
    [TCRVRVV_TailAgnostic, TCRVRVV_TailUndisturbed]> {
  let cppNamespace = "::tianchenrv::tcrv::rvv";
}

def TCRVRVV_MaskAgnostic
    : I32EnumAttrCase<"Agnostic", 0, "agnostic">;
def TCRVRVV_MaskUndisturbed
    : I32EnumAttrCase<"Undisturbed", 1, "undisturbed">;

def TCRVRVV_MaskPolicy : I32EnumAttr<
    "MaskPolicy", "finite RVV inactive mask lane policy",
    [TCRVRVV_MaskAgnostic, TCRVRVV_MaskUndisturbed]> {
  let cppNamespace = "::tianchenrv::tcrv::rvv";
}

def TCRVRVV_PolicyAttr : TCRVRVV_Attr<"Policy", "policy"> {
  let summary = "RVV non-compute tail and mask policy metadata";
  let description = [{
    Finite plugin-owned policy metadata for RVV proposal/materialization
    bookkeeping. This attribute records explicit tail and mask behavior only:
    tail agnostic or undisturbed, and mask agnostic or undisturbed. It does not
    model vector arithmetic, memory behavior, lowering, runtime ABI, executable
    correctness, or performance.
  }];

  let parameters = (ins
    EnumParameter<TCRVRVV_TailPolicy>:$tail,
    EnumParameter<TCRVRVV_MaskPolicy>:$mask
  );
  let assemblyFormat = "`<` `tail` `=` $tail `,` `mask` `=` $mask `>`";
}

def TCRVRVV_VLType : TCRVRVV_Type<"VL", "vl"> {
  let summary = "RVV vector-length token type";
  let description = [{
    A non-compute token type representing an RVV vector-length control-plane
    value. This type is produced by the bounded tcrv_rvv.setvl runtime
    AVL-to-VL control-plane op, consumed by the bounded tcrv_rvv.with_vl
    control-plane scope op, and may also appear in syntax tests proving
    plugin-local RVV dialect registration and ownership. It does not model
    vector arithmetic, memory, runtime ABI, lowering, executable behavior,
    correctness, or performance.
  }];
}

def TCRVRVV_RuntimeABIValueType
    : TCRVRVV_Type<"RuntimeABIValue", "runtime_abi_value"> {
  let summary = "bounded RVV runtime ABI value token type";
  let description = [{
    A plugin-local token type representing one explicit runtime ABI value
    visible to the bounded RVV first-slice IR. Buffer ABI values such as lhs,
    rhs, and out use this token before the RVV EmitC route maps them to C
    pointer parameters. Runtime scalar values such as n and bounded RHS scalar
    values may use ordinary MLIR scalar types while still being defined by
    tcrv_rvv.runtime_abi_value.

    This type is not a descriptor, vector register, memory model, source
    exporter, runtime result, correctness evidence, or performance evidence.
  }];
}

def TCRVRVV_I32M1VectorType : TCRVRVV_Type<"I32M1Vector", "i32m1"> {
  let summary = "deprecated finite RVV i32 m1 parse-only dataflow token type";
  let description = [{
    Deprecated Stage1 inventory for finite RVV i32 dataflow parser/verifier
    coverage under an existing tcrv_rvv.with_vl scope. It represents stale
    compiler-visible results from tcrv_rvv.i32_* ops only so those surfaces can
    remain diagnosable. It does not model arbitrary vector registers, generic
    memory, masks, layouts, runtime ABI parameters, route selection, artifact
    emission, source-front-door authority, hardware facts, correctness
    evidence, or performance evidence.
  }];
}

def TCRVRVV_I32M1MaskType : TCRVRVV_Type<"I32M1Mask", "i32m1_mask"> {
  let summary = "deprecated finite RVV i32 m1 predicate mask parse-only type";
  let description = [{
    Deprecated Stage1 inventory for finite RVV i32m1 compare/select
    parser/verifier coverage under an existing tcrv_rvv.with_vl scope. It
    represents stale tcrv_rvv.i32_cmp_eq / tcrv_rvv.i32_select dataflow only
    so those surfaces can remain diagnosable.

    This is not a generic mask register type, vector dialect mask, descriptor,
    route selector, artifact authority, source-front-door authority, hardware
    fact, correctness evidence, or performance evidence.
  }];
}

def TCRVRVV_I32M2VectorType : TCRVRVV_Type<"I32M2Vector", "i32m2"> {
  let summary = "deprecated finite RVV i32 m2 parse-only dataflow token type";
  let description = [{
    Deprecated Stage1 inventory for finite RVV i32 dataflow parser/verifier
    coverage when the enclosing control-plane config is SEW 32 and LMUL m2.
    It is a sibling of !tcrv_rvv.i32m1, not a generic RVV vector type. This
    token does not authorize route support, source-front-door materialization,
    or target emission.
  }];
}

def TCRVRVV_VectorType : TCRVRVV_Type<"Vector", "vector"> {
  let summary = "generic typed RVV vector dataflow token type";
  let description = [{
    A generic plugin-local RVV vector dataflow token for selected low-level
    RVV bodies. The type carries the element dtype and LMUL. The enclosing
    tcrv_rvv.setvl / tcrv_rvv.with_vl control surface carries SEW and policy,
    and RVV plugin legality validates that the type/config/body agree before
    route construction.

    This type is the corrected Stage 1 selected-body surface for retained
    arithmetic routes. It is not a route id, descriptor, artifact name, source
    front door marker, intrinsic spelling, runtime evidence, or performance
    claim.
  }];

  let parameters = (ins
    "mlir::Type":$elementType,
    StringRefParameter<"LMUL string">:$lmul
  );
  let assemblyFormat = "`<` $elementType `,` $lmul `>`";
}

def TCRVRVV_IndexVectorType : TCRVRVV_Type<"IndexVector", "index_vector"> {
  let summary = "generic typed RVV index vector dataflow token type";
  let description = [{
    A generic plugin-local RVV index vector token for bounded indexed memory
    movement. The type carries the index element dtype and LMUL. For the
    current executable slices it is an unsigned 32-bit index/offset vector
    with LMUL m1, produced by tcrv_rvv.index_load and consumed by
    tcrv_rvv.indexed_load or tcrv_rvv.indexed_store.

    This type is not a route id, descriptor, artifact name, source front door
    marker, intrinsic spelling, runtime evidence, or performance claim. The
    RVV plugin derives indexed memory legality and intrinsic mapping from this
    typed body/config fact plus explicit runtime ABI roles.
  }];

  let parameters = (ins
    "mlir::Type":$elementType,
    StringRefParameter<"LMUL string">:$lmul
  );
  let assemblyFormat = "`<` $elementType `,` $lmul `>`";
}

def TCRVRVV_MaskType : TCRVRVV_Type<"Mask", "mask"> {
  let summary = "generic typed RVV predicate mask dataflow token type";
  let description = [{
    A generic plugin-local RVV predicate token produced by compare dataflow on
    the corrected typed RVV surface. The type carries the compared element
    dtype and LMUL so the RVV plugin can derive the concrete mask C type from
    body/config facts after legality validation.

    This type is not a route id, descriptor, artifact name, source front door
    marker, intrinsic spelling, runtime evidence, or performance claim.
  }];

  let parameters = (ins
    "mlir::Type":$elementType,
    StringRefParameter<"LMUL string">:$lmul
  );
  let assemblyFormat = "`<` $elementType `,` $lmul `>`";
}

def RuntimeABIValueOp : TCRVRVV_Op<"runtime_abi_value",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "bounded RVV runtime ABI value binding";
  let description = [{
    Binds one callable runtime ABI value into explicit RVV extension-family IR
    for the current first slice. The operation records the ABI role, C
    parameter name, C type spelling, and ownership class, then produces one SSA
    value consumed by tcrv_rvv.setvl, generic tcrv_rvv.load / tcrv_rvv.store,
    or other typed RVV body ops.

    This op is an ABI/provenance boundary only. It does not describe
    computation, infer route selection, expose target hardware facts, replace
    tcrv.exec dispatch/runtime-param organization, or authorize descriptor or
    direct-C source export.
  }];

  let arguments = (ins
    StrAttr:$role,
    StrAttr:$c_name,
    StrAttr:$c_type,
    StrAttr:$ownership,
    OptionalAttr<StrAttr>:$purpose
  );
  let results = (outs AnyType:$value);

  let assemblyFormat = "attr-dict `:` type($value)";
  let hasVerifier = 1;
}

def SetVLOp : TCRVRVV_Op<"setvl",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "bounded RVV runtime AVL to VL control-plane op";
  let description = [{
    Models the first plugin-local RVV runtime vector-length control-plane
    surface. The operation consumes one runtime AVL SSA value, produces one
    !tcrv_rvv.vl token, and carries bounded compile-time first-slice
    configuration metadata: SEW 32, LMUL m1 or m2, and a finite
    #tcrv_rvv.policy attribute.

    This op does not model VLEN/vlenb hardware facts, deleted local
    element-count metadata, required march metadata, required capability
    metadata, pointer or
    memory operands, RVV arithmetic, vector registers, masks, lowering,
    runtime ABI glue, generated artifacts, hardware execution, correctness, or
    performance. The companion tcrv_rvv.with_vl op may consume the produced
    !tcrv_rvv.vl token to form a bounded control-plane region.
  }];

  let arguments = (ins
    AnyType:$avl,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );
  let results = (outs AnyType:$vl);

  let assemblyFormat = "$avl attr-dict `:` type($avl) `->` type($vl)";
  let hasVerifier = 1;
}

def WithVLOp : TCRVRVV_Op<"with_vl",
    [NoRegionArguments, NoTerminator, SingleBlock,
     DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "bounded RVV VL scope control-plane region";
  let description = [{
    Creates a plugin-local RVV control-plane scope for work that is governed by
    an existing runtime VL token. The operation consumes one !tcrv_rvv.vl value,
    normally produced by tcrv_rvv.setvl, and owns one single-block region with
    no region arguments. Optional duplicated SEW/LMUL/policy attributes are
    limited to the same bounded first-slice compile-time configuration accepted
    by setvl and are verified for compatibility when the consumed VL is visibly
    defined by a tcrv_rvv.setvl op.

    On the selected generic typed materialized EmitC artifact path, this op is
    also the selected construction-template boundary. In that role it may carry
    generic selected-boundary conformance mirrors such as source kernel,
    selected variant, origin, selected path role, status, required
    capabilities, and RVV-owned route/config handoff attributes. Those facts
    mirror provider-owned route/config decisions for validation and artifact
    handoff; they are not descriptor fields, acceptance state, dtype authority,
    route authority, or compute semantics.

    This op is a structural prerequisite for RVV control/body lowering work
    only. It does not model VLEN/vlenb hardware facts, deleted local
    element-count metadata, required march metadata, raw capability facts,
    pointer or memory operands, RVV arithmetic, vector registers, masks,
    lowering, runtime ABI glue, generated artifacts, hardware execution,
    correctness, or performance.
  }];

  let arguments = (ins
    AnyType:$vl,
    OptionalAttr<I64Attr>:$sew,
    OptionalAttr<StrAttr>:$lmul,
    OptionalAttr<TCRVRVV_PolicyAttr>:$policy,
    OptionalAttr<StrAttr>:$source_kernel,
    OptionalAttr<FlatSymbolRefAttr>:$selected_variant,
    OptionalAttr<StrAttr>:$origin,
    OptionalAttr<StrAttr>:$selected_path_role,
    OptionalAttr<StrAttr>:$status,
    OptionalAttr<ArrayAttr>:$required_capabilities,
    OptionalAttr<StrAttr>:$rvv_construction_protocol,
    OptionalAttr<StrAttr>:$rvv_emitc_route_mapping
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$vl attr-dict-with-keyword $body `:` type($vl)";
  let hasVerifier = 1;
}

def TypedBinaryPreRealizedBodyOp
    : TCRVRVV_Op<"typed_binary_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV binary selected body";
  let description = [{
    Carries the RVV plugin-local pre-realized selected body for the generic
    typed RVV binary family. The op is nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for lhs, rhs, out,
    runtime n/AVL, and optional runtime stride values together with the
    selected operation kind, memory form, and compile-time RVV config. RHS may
    be a buffer ABI value for vector RHS routes or a scalar ABI value for the
    bounded scalar-broadcast add route.

    This op is not a common route, descriptor, source-front-door marker,
    direct-C exporter, intrinsic wrapper, runtime artifact, correctness
    evidence, or performance evidence. The RVV plugin must first realize it
    into explicit setvl/with_vl/load/compute/store tcrv_rvv body structure
    before the existing provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$out,
    AnyType:$n,
    Variadic<Index>:$strides,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $out `,` $n (`,` $strides^)? attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedMaskedBinaryPreRealizedBodyOp
    : TCRVRVV_Op<"typed_masked_binary_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV masked binary selected body";
  let description = [{
    Carries the RVV plugin-local pre-realized selected body for the bounded
    generic masked binary family. The op is nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for lhs, rhs, out,
    and runtime n/AVL together with selected operation kind, memory form, mask
    source, passthrough layout, and compile-time RVV config facts.

    This op is not a mask framework, descriptor, source-front-door marker,
    direct-C exporter, route id, intrinsic wrapper, runtime artifact,
    correctness evidence, or performance evidence. The RVV plugin must first
    realize it into explicit setvl/with_vl/load/compare/masked_binary/store
    tcrv_rvv body structure before the existing provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$mask_source,
    StrAttr:$masked_passthrough,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedCompareSelectPreRealizedBodyOp
    : TCRVRVV_Op<"typed_compare_select_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV compare/select selected body";
  let description = [{
    Carries the RVV plugin-local pre-realized selected body for one bounded
    generic compare/select family. The op is nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for lhs, rhs, out,
    and runtime n/AVL together with selected operation kind, predicate kind,
    memory form, mask source, select layout, and compile-time RVV config facts.

    This op is not a predicate framework, descriptor, source-front-door marker,
    direct-C exporter, route id, intrinsic wrapper, runtime artifact,
    correctness evidence, or performance evidence. The RVV plugin must first
    realize it into explicit setvl/with_vl/load/compare/select/store tcrv_rvv
    body structure before the existing provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    StrAttr:$mask_source,
    StrAttr:$select_layout,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedComputedMaskSelectPreRealizedBodyOp
    : TCRVRVV_Op<"typed_computed_mask_select_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV computed-mask vector select body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector compare produces an in-body mask SSA value and a vector
    select consumes that mask to choose between explicit true-value and
    false-value vector operands. The op is nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for compare lhs,
    compare rhs, true value, false value, output, and runtime n/AVL together
    with compare predicate, memory form, mask role/source/config, select
    layout, compile-time RVV data/mask config, and policy.

    This op is not a select framework, predicate matrix, route label,
    descriptor, source-front-door marker, direct-C exporter, intrinsic wrapper,
    runtime artifact, correctness evidence, or performance evidence. The RVV
    plugin must realize it into explicit setvl/with_vl/load/load/load/load/
    compare/select/store tcrv_rvv body structure before the provider may
    construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs,
    AnyType:$compare_rhs,
    AnyType:$true_value,
    AnyType:$false_value,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$select_layout,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs `,` $compare_rhs `,` $true_value `,` $false_value `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedReducePreRealizedBodyOp
    : TCRVRVV_Op<"typed_reduce_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV reduce selected body";
  let description = [{
    Carries the RVV plugin-local pre-realized selected body for the bounded
    reduce-add route. The op is nested directly in a selected tcrv.exec.variant
    and binds explicit runtime ABI values for input, accumulator seed, output,
    and runtime n/AVL together with selected operation kind, memory form,
    accumulator role/layout, result layout, and compile-time RVV config facts.

    This op is not a broad reduction framework, descriptor, source-front-door
    marker, direct-C exporter, route id, intrinsic wrapper, runtime artifact,
    correctness evidence, or performance evidence. The RVV plugin must first
    realize it into explicit setvl/with_vl/load/load/reduce/store tcrv_rvv body
    structure before the existing provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$accumulator_role,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedStandaloneReducePreRealizedBodyOp
    : TCRVRVV_Op<"typed_standalone_reduce_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV standalone scalar reduction body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for the
    standalone horizontal add-reduction slice. The op is nested directly in a
    selected tcrv.exec.variant and binds explicit runtime ABI values for input,
    scalar accumulator seed, scalar output, and runtime n/AVL together with
    selected operation kind, standalone memory form, accumulator role/layout,
    scalar result layout, and compile-time RVV config facts.

    This op is not a broad reduction framework, contraction route, descriptor,
    source-front-door marker, direct-C exporter, route id, intrinsic wrapper,
    runtime artifact, correctness evidence, or performance evidence. The RVV
    plugin must first realize it into explicit setvl/with_vl/load/
    standalone_reduce/store tcrv_rvv body structure before the provider may
    construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$acc,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$accumulator_role,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $acc `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedMAccPreRealizedBodyOp
    : TCRVRVV_Op<"typed_macc_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV macc selected body";
  let description = [{
    Carries the RVV plugin-local pre-realized selected body for the bounded
    multiply-accumulate route. The op is nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for lhs, rhs, out,
    and runtime n/AVL together with selected operation kind, memory form,
    accumulator role/layout, result layout, and compile-time RVV config facts.

    This op is not a matmul route, contraction framework, descriptor,
    source-front-door marker, direct-C exporter, route id, intrinsic wrapper,
    runtime artifact, correctness evidence, or performance evidence. The RVV
    plugin must first realize it into explicit setvl/with_vl/load/load/
    accumulator-load/macc/store tcrv_rvv body structure before the existing
    provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$accumulator_role,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedWideningMAccPreRealizedBodyOp
    : TCRVRVV_Op<"typed_widening_macc_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV widening macc selected body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for signed
    widening multiply-accumulate. The op is nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for i16 lhs, i16
    rhs, i32 accumulator input, i32 output, and runtime n/AVL together with
    operation kind, unit-stride memory form, accumulator role/layout, result
    layout, source and accumulator/result SEW+LMUL facts, relation, and policy.

    This op is not a matmul route, contraction framework, dtype-prefixed helper
    namespace, descriptor, source-front-door marker, direct-C exporter, route
    id, intrinsic wrapper, runtime artifact, correctness evidence, or
    performance evidence. The RVV plugin must realize it into explicit
    setvl/with_vl/load/load/load/widening_macc/store tcrv_rvv body structure
    before the provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$acc,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$accumulator_role,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    I64Attr:$source_sew,
    StrAttr:$source_lmul,
    I64Attr:$accumulator_sew,
    StrAttr:$accumulator_lmul,
    I64Attr:$result_sew,
    StrAttr:$result_lmul,
    StrAttr:$macc_relation,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $acc `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedWideningDotReducePreRealizedBodyOp
    : TCRVRVV_Op<"typed_widening_dot_reduce_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV widening dot-product reduction body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for signed
    horizontal widening dot-product reduction. The op is nested directly in a
    selected tcrv.exec.variant and binds explicit runtime ABI values for i16
    lhs, i16 rhs, i32 accumulator seed input, i32 scalar output, and runtime
    n/AVL together with operation kind, unit-stride memory form, accumulator
    role/layout, result layout, source and accumulator/result SEW+LMUL facts,
    dot-product relation, and policy.

    This op is not matmul, a broad contraction framework, dtype-prefixed helper
    namespace, descriptor, source-front-door marker, direct-C exporter, route
    id, intrinsic wrapper, runtime artifact, correctness evidence, or
    performance evidence. The RVV plugin must realize it into explicit typed
    tcrv_rvv body structure before the provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$acc,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$accumulator_role,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    I64Attr:$source_sew,
    StrAttr:$source_lmul,
    I64Attr:$accumulator_sew,
    StrAttr:$accumulator_lmul,
    I64Attr:$result_sew,
    StrAttr:$result_lmul,
    StrAttr:$dot_product_relation,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $acc `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedStridedInputWideningDotReducePreRealizedBodyOp
    : TCRVRVV_Op<"typed_strided_input_widening_dot_reduce_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV strided-input widening dot-product reduction body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for signed
    horizontal widening dot-product reduction whose lhs and rhs source windows
    are addressed through explicit runtime element strides. The op is nested
    directly in a selected tcrv.exec.variant and binds explicit runtime ABI
    values for i16 lhs, i16 rhs, i32 accumulator seed input, i32 scalar output,
    runtime n/AVL, lhs element stride, and rhs element stride together with
    operation kind, strided-input memory form, stride unit, accumulator
    role/layout, result layout, source and accumulator/result SEW+LMUL facts,
    dot-product relation, and policy.

    This op is not matmul, masked dot-reduction, a broad contraction
    framework, dtype-prefixed helper namespace, descriptor, source-front-door
    marker, direct-C exporter, route id, intrinsic wrapper, runtime artifact,
    correctness evidence, or performance evidence. The RVV plugin must realize
    it into explicit typed strided_load/strided_load/widening_dot_reduce/store
    tcrv_rvv body structure before the provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$acc,
    AnyType:$out,
    AnyType:$n,
    Index:$lhs_stride,
    Index:$rhs_stride,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$stride_unit,
    StrAttr:$accumulator_role,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    I64Attr:$source_sew,
    StrAttr:$source_lmul,
    I64Attr:$accumulator_sew,
    StrAttr:$accumulator_lmul,
    I64Attr:$result_sew,
    StrAttr:$result_lmul,
    StrAttr:$dot_product_relation,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $acc `,` $out `,` $n `,` $lhs_stride `,` $rhs_stride attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedComputedMaskWideningDotReducePreRealizedBodyOp
    : TCRVRVV_Op<"typed_computed_mask_widening_dot_reduce_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV computed-mask widening dot-product reduction body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector compare produces an in-body mask SSA value and a signed
    i16mf2 x i16mf2 widening dot-product reduction consumes that mask. The op
    is nested directly in a selected tcrv.exec.variant and binds explicit
    runtime ABI values for compare lhs/rhs, i16 dot lhs/rhs, i32 accumulator
    seed input, i32 scalar output, and runtime n/AVL together with compare
    predicate, mask role/source/config, operation kind, unit-stride memory
    form, accumulator role/layout, scalar result layout, source and
    accumulator/result SEW+LMUL facts, dot-product relation, and policy.

    This op is not matmul, a broad contraction framework, dtype-prefixed helper
    namespace, descriptor, source-front-door marker, direct-C exporter, route
    id, intrinsic wrapper, runtime artifact, correctness evidence, or
    performance evidence. The RVV plugin must realize it into explicit
    setvl/with_vl/load/load/compare/load/load/masked_widening_dot_reduce/store
    tcrv_rvv body structure before the provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs,
    AnyType:$compare_rhs,
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$acc,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$accumulator_role,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    I64Attr:$source_sew,
    StrAttr:$source_lmul,
    I64Attr:$accumulator_sew,
    StrAttr:$accumulator_lmul,
    I64Attr:$result_sew,
    StrAttr:$result_lmul,
    StrAttr:$dot_product_relation,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs `,` $compare_rhs `,` $lhs `,` $rhs `,` $acc `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp
    : TCRVRVV_Op<"typed_computed_mask_strided_input_widening_dot_reduce_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV computed-mask strided-input widening dot-product reduction body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector compare produces an in-body mask SSA value and a signed
    i16mf2 x i16mf2 widening dot-product reduction consumes that mask while
    the dot lhs/rhs source windows are addressed through explicit runtime
    element strides. The op is nested directly in a selected tcrv.exec.variant
    and binds explicit runtime ABI values for compare lhs/rhs, i16 dot
    lhs/rhs, i32 accumulator seed input, i32 scalar output, runtime n/AVL,
    lhs element stride, and rhs element stride together with compare
    predicate, mask role/source/config, operation kind, strided-input memory
    form, stride unit, accumulator role/layout, scalar result layout, source
    and accumulator/result SEW+LMUL facts, dot-product relation, and policy.

    This op is not matmul, a broad contraction framework, dtype-prefixed helper
    namespace, descriptor, source-front-door marker, direct-C exporter, route
    id, intrinsic wrapper, runtime artifact, correctness evidence, or
    performance evidence. The RVV plugin must realize it into explicit
    setvl/with_vl/load/load/compare/strided_load/strided_load/
    masked_widening_dot_reduce/store tcrv_rvv body structure before the
    provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs,
    AnyType:$compare_rhs,
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$acc,
    AnyType:$out,
    AnyType:$n,
    Index:$lhs_stride,
    Index:$rhs_stride,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    StrAttr:$stride_unit,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$accumulator_role,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    I64Attr:$source_sew,
    StrAttr:$source_lmul,
    I64Attr:$accumulator_sew,
    StrAttr:$accumulator_lmul,
    I64Attr:$result_sew,
    StrAttr:$result_lmul,
    StrAttr:$dot_product_relation,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs `,` $compare_rhs `,` $lhs `,` $rhs `,` $acc `,` $out `,` $n `,` $lhs_stride `,` $rhs_stride attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedWideningConversionPreRealizedBodyOp
    : TCRVRVV_Op<"typed_widening_conversion_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV widening conversion body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for a
    signed widening conversion. The op is nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for source input,
    output, runtime n/AVL, selected conversion operation kind, source memory
    form, source/destination SEW+LMUL facts, and policy.

    This op is not a conversion framework, dtype-prefixed helper namespace,
    descriptor, source-front-door marker, direct-C exporter, route id,
    intrinsic wrapper, runtime artifact, correctness evidence, or performance
    evidence. The RVV plugin must realize it into explicit setvl/with_vl/load/
    widening_convert/store tcrv_rvv body structure before the provider may
    construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    I64Attr:$source_sew,
    StrAttr:$source_lmul,
    I64Attr:$dest_sew,
    StrAttr:$dest_lmul,
    StrAttr:$conversion_relation,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedStridedMemoryPreRealizedBodyOp
    : TCRVRVV_Op<"typed_strided_memory_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV strided memory movement body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for
    strided source memory movement into a unit-stride destination. The op is
    nested directly in a selected tcrv.exec.variant and binds explicit runtime
    ABI values for source input, output, runtime n/AVL, and source byte
    stride together with selected memory movement operation kind, memory form,
    stride unit, compile-time RVV config, and policy.

    This op is not a layout framework, gather/scatter route, descriptor,
    source-front-door marker, direct-C exporter, route id, intrinsic wrapper,
    runtime artifact, correctness evidence, or performance evidence. The RVV
    plugin must realize it into explicit setvl/with_vl/strided_load/move/store
    tcrv_rvv body structure before the provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$source,
    AnyType:$out,
    AnyType:$n,
    Index:$source_stride,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$stride_unit,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$source `,` $out `,` $n `,` $source_stride attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedStridedStoreMemoryPreRealizedBodyOp
    : TCRVRVV_Op<"typed_strided_store_memory_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV strided destination-store body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for
    unit-stride source memory movement into a runtime-strided destination. The
    op is nested directly in a selected tcrv.exec.variant and binds explicit
    runtime ABI values for source input, destination output, runtime n/AVL, and
    destination element stride together with selected memory movement operation
    kind, memory form, stride unit, compile-time RVV config, and policy.

    This op is not a layout framework, gather/scatter route, masked route,
    descriptor, source-front-door marker, direct-C exporter, route id,
    intrinsic wrapper, runtime artifact, correctness evidence, or performance
    evidence. The RVV plugin must realize it into explicit
    setvl/with_vl/load/move/strided_store tcrv_rvv body structure before the
    provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$source,
    AnyType:$dst,
    AnyType:$n,
    Index:$destination_stride,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$stride_unit,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$source `,` $dst `,` $n `,` $destination_stride attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedIndexedGatherMemoryPreRealizedBodyOp
    : TCRVRVV_Op<"typed_indexed_gather_memory_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV indexed gather memory body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for
    indexed source gather memory movement into a unit-stride destination. The
    op is nested directly in a selected tcrv.exec.variant and binds explicit
    runtime ABI values for data source, index source, output, and runtime
    n/AVL together with selected memory movement operation kind, memory form,
    index EEW, offset unit, compile-time RVV data vector config, and policy.

    This op is not a gather/scatter framework, descriptor, source-front-door
    marker, direct-C exporter, route id, intrinsic wrapper, runtime artifact,
    correctness evidence, or performance evidence. The RVV plugin must realize
    it into explicit setvl/with_vl/index_load/indexed_load/move/store
    tcrv_rvv body structure before the provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$data,
    AnyType:$index,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    I64Attr:$index_eew,
    StrAttr:$offset_unit,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$data `,` $index `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedIndexedScatterMemoryPreRealizedBodyOp
    : TCRVRVV_Op<"typed_indexed_scatter_memory_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV indexed scatter memory body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for
    unit-stride source memory movement into an indexed destination. The op is
    nested directly in a selected tcrv.exec.variant and binds explicit runtime
    ABI values for source data, index source, indexed destination, and runtime
    n/AVL together with selected memory movement operation kind, memory form,
    index EEW, offset unit, explicit unique-index policy, compile-time RVV
    data vector config, and policy.

    This op is not a gather/scatter framework, masked scatter route,
    duplicate-index resolver, descriptor, source-front-door marker, direct-C
    exporter, route id, intrinsic wrapper, runtime artifact, correctness
    evidence, or performance evidence. The RVV plugin must realize it into
    explicit setvl/with_vl/load/index_load/move/indexed_store tcrv_rvv body
    structure before the provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$source,
    AnyType:$index,
    AnyType:$destination,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    I64Attr:$index_eew,
    StrAttr:$offset_unit,
    StrAttr:$index_uniqueness,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$source `,` $index `,` $destination `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedMaskedMemoryPreRealizedBodyOp
    : TCRVRVV_Op<"typed_masked_memory_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV masked memory movement body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for
    mask-aware unit-stride source-to-destination memory movement. The op is
    nested directly in a selected tcrv.exec.variant and binds explicit runtime
    ABI values for source, mask, destination, and runtime n/AVL together with
    selected memory movement operation kind, memory form, mask role/config,
    inactive-lane preservation policy, compile-time RVV data/mask config, and
    policy.

    This op is not a masked memory framework, masked gather/scatter route,
    descriptor, source-front-door marker, direct-C exporter, route id,
    intrinsic wrapper, runtime artifact, correctness evidence, or performance
    evidence. The RVV plugin must realize it into explicit setvl/with_vl/load/
    mask_load/load/masked_move/store tcrv_rvv body structure before the
    provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$source,
    AnyType:$mask,
    AnyType:$destination,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$mask_role,
    StrAttr:$mask_memory_form,
    StrAttr:$inactive_lane_policy,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$source `,` $mask `,` $destination `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedComputedMaskMemoryPreRealizedBodyOp
    : TCRVRVV_Op<"typed_computed_mask_memory_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV computed-mask memory body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector compare produces an in-body mask SSA value and masked
    unit-stride memory movement consumes that mask. The op is nested directly
    in a selected tcrv.exec.variant and binds explicit runtime ABI values for
    compare lhs, compare rhs, active source, destination, and runtime n/AVL
    together with compare predicate, memory form, mask role/source/config,
    inactive-lane preservation policy, compile-time RVV data/mask config, and
    policy.

    This op is not a fusion framework, route label, intrinsic wrapper,
    runtime artifact, correctness evidence, or performance evidence. The RVV
    plugin must realize it into explicit setvl/with_vl/load/load/load/load/
    compare/masked_move/store tcrv_rvv body structure before the provider may
    construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs,
    AnyType:$compare_rhs,
    AnyType:$source,
    AnyType:$destination,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$inactive_lane_policy,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs `,` $compare_rhs `,` $source `,` $destination `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedComputedMaskStridedStorePreRealizedBodyOp
    : TCRVRVV_Op<"typed_computed_mask_strided_store_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV computed-mask strided-store body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector compare produces an in-body mask SSA value and masked
    memory movement stores active source lanes to a runtime-strided
    destination. The op is nested directly in a selected tcrv.exec.variant and
    binds explicit runtime ABI values for compare lhs, compare rhs, active
    source, destination, runtime n/AVL, and destination element stride together
    with compare predicate, memory form, mask role/source/config,
    inactive-lane preservation policy, stride unit, compile-time RVV data/mask
    config, and policy.

    This op is not a fusion framework, route label, descriptor, source-front
    door marker, direct-C exporter, intrinsic wrapper, runtime artifact,
    correctness evidence, or performance evidence. The RVV plugin must realize
    it into explicit setvl/with_vl/load/load/load/strided_load/compare/
    masked_move/strided_store tcrv_rvv body structure before the provider may
    construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs,
    AnyType:$compare_rhs,
    AnyType:$source,
    AnyType:$destination,
    AnyType:$n,
    Index:$destination_stride,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    StrAttr:$stride_unit,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$inactive_lane_policy,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs `,` $compare_rhs `,` $source `,` $destination `,` $n `,` $destination_stride attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedSegment2DeinterleaveMemoryPreRealizedBodyOp
    : TCRVRVV_Op<"typed_segment2_deinterleave_memory_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV segment2 deinterleave memory body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for a
    signed i32 segment2 interleaved source load into two unit-stride
    destination fields. The op is nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for source, field0
    destination, field1 destination, and runtime n/AVL together with
    segment_count, field roles/order, source/destination memory forms,
    compile-time RVV config, and policy.

    This op is not a segmented-memory framework, descriptor, source-front-door
    marker, route id, artifact name, direct-C exporter, intrinsic wrapper,
    runtime artifact, correctness evidence, or performance evidence. The RVV
    plugin must realize it into explicit setvl/with_vl/segment2_load/move/
    move/store/store tcrv_rvv body structure before the provider may construct
    a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$source,
    AnyType:$out0,
    AnyType:$out1,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    I64Attr:$segment_count,
    StrAttr:$field0_role,
    StrAttr:$field1_role,
    StrAttr:$source_memory_form,
    StrAttr:$destination_memory_form,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$source `,` $out0 `,` $out1 `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedSegment2InterleaveMemoryPreRealizedBodyOp
    : TCRVRVV_Op<"typed_segment2_interleave_memory_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV segment2 interleave memory body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for a
    signed i32 pair of unit-stride field sources into one interleaved segment2
    destination. The op is nested directly in a selected tcrv.exec.variant and
    binds explicit runtime ABI values for field0 source, field1 source,
    interleaved destination, and runtime n/AVL together with segment_count,
    field roles/order, source/destination memory forms, compile-time RVV
    config, and policy.

    This op is not a segmented-memory framework, descriptor, source-front-door
    marker, route id, artifact name, direct-C exporter, intrinsic wrapper,
    runtime artifact, correctness evidence, or performance evidence. The RVV
    plugin must realize it into explicit setvl/with_vl/load/load/
    segment2_store tcrv_rvv body structure before the provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$src0,
    AnyType:$src1,
    AnyType:$dst,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    I64Attr:$segment_count,
    StrAttr:$field0_role,
    StrAttr:$field1_role,
    StrAttr:$source0_memory_form,
    StrAttr:$source1_memory_form,
    StrAttr:$destination_memory_form,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$src0 `,` $src1 `,` $dst `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def LoadOp : TCRVRVV_Op<"load",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV vector load dataflow op";
  let description = [{
    Records one generic plugin-local RVV vector load dataflow step. The op is
    nested directly under tcrv_rvv.with_vl, consumes one explicit runtime ABI
    buffer SSA value plus the active !tcrv_rvv.vl token, and produces one
    !tcrv_rvv.vector element/LMUL token matching the enclosing config.
    LHS/RHS input buffers are valid for ordinary arithmetic and compare
    operands; source input buffers are valid for bounded computed-mask memory
    movement; an output buffer load is valid only when the RVV provider
    recognizes it as an explicit accumulator/old-destination input.

    Buffer ABI provenance comes from the explicit runtime ABI operand. Element
    dtype and LMUL come from the result type and are validated with the
    surrounding setvl/with_vl config by the RVV dialect/provider.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$vl
  );
  let results = (outs AnyType:$loaded);

  let assemblyFormat =
    "$buffer `,` $vl attr-dict `:` type($buffer) `,` type($vl) `->` type($loaded)";
  let hasVerifier = 1;
}

def MaskLoadOp : TCRVRVV_Op<"mask_load",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV mask buffer load dataflow op";
  let description = [{
    Records one bounded plugin-local RVV mask-buffer load dataflow step. The
    op is nested directly under tcrv_rvv.with_vl, consumes one explicit mask
    input-buffer runtime ABI value plus the active !tcrv_rvv.vl token, carries
    explicit mask role and mask memory form facts, and produces one
    !tcrv_rvv.mask token matching the enclosing vector config.

    The mask buffer role and mask memory form are structural body authority for
    bounded masked memory movement. The RVV plugin derives mask materialization
    from this typed body fact, the data/mask config, and runtime ABI binding;
    route ids, artifact names, ABI strings, descriptor residue, and common
    EmitC/export code must not infer mask semantics.
  }];

  let arguments = (ins
    AnyType:$mask,
    AnyType:$vl,
    StrAttr:$mask_role,
    StrAttr:$mask_memory_form
  );
  let results = (outs AnyType:$loaded);

  let assemblyFormat =
    "$mask `,` $vl attr-dict `:` type($mask) `,` type($vl) `->` type($loaded)";
  let hasVerifier = 1;
}

def BroadcastLoadOp : TCRVRVV_Op<"broadcast_load",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV RHS scalar broadcast load dataflow op";
  let description = [{
    Records one generic plugin-local RVV RHS scalar-load/broadcast dataflow
    step. The op is nested directly under tcrv_rvv.with_vl, consumes one
    explicit RHS runtime ABI buffer SSA value plus the active !tcrv_rvv.vl
    token, and produces one !tcrv_rvv.vector element/LMUL token matching the
    enclosing config.

    The memory form is structural in this op. The RVV plugin derives the
    concrete scalar load / vector broadcast intrinsic from this typed body
    fact, the vector type, SEW/LMUL/policy, and runtime ABI binding.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$vl
  );
  let results = (outs AnyType:$broadcast);

  let assemblyFormat =
    "$buffer `,` $vl attr-dict `:` type($buffer) `,` type($vl) `->` type($broadcast)";
  let hasVerifier = 1;
}

def SplatOp : TCRVRVV_Op<"splat",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV runtime scalar splat dataflow op";
  let description = [{
    Records one bounded plugin-local RVV runtime scalar broadcast dataflow
    step. The op is nested directly under tcrv_rvv.with_vl, consumes one
    explicit RHS scalar runtime ABI SSA value plus the active !tcrv_rvv.vl
    token, and produces one !tcrv_rvv.vector element/LMUL token matching the
    enclosing config.

    The scalar value is structural body authority: it must come from an
    explicit tcrv_rvv.runtime_abi_value with the rhs-scalar-value role. The RVV
    plugin derives the concrete vector splat intrinsic from this typed body
    fact, the vector type, SEW/LMUL/policy, and runtime ABI binding. Route ids,
    artifact names, C ABI strings, helper names, and common EmitC/export code
    must not infer scalar-broadcast semantics.
  }];

  let arguments = (ins
    AnyType:$scalar,
    AnyType:$vl
  );
  let results = (outs AnyType:$broadcast);

  let assemblyFormat =
    "$scalar `,` $vl attr-dict `:` type($scalar) `,` type($vl) `->` type($broadcast)";
  let hasVerifier = 1;
}

def StridedLoadOp : TCRVRVV_Op<"strided_load",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV strided vector load dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV strided vector load dataflow
    step. The op is nested directly under tcrv_rvv.with_vl, consumes one
    explicit input buffer runtime ABI value, one explicit runtime stride SSA
    value, and the active !tcrv_rvv.vl token. It produces one
    !tcrv_rvv.vector element/LMUL token matching the enclosing config.

    The stride unit is route-specific typed body authority. The bounded
    strided-load to unit-store slice uses a runtime byte stride; older
    element-strided arithmetic/contraction slices remain separate routes. The
    RVV plugin owns legality and intrinsic operand mapping after typed
    body/config/runtime validation.
  }];

  let arguments = (ins
    AnyType:$buffer,
    Index:$stride,
    AnyType:$vl
  );
  let results = (outs AnyType:$loaded);

  let assemblyFormat =
    "$buffer `,` $stride `,` $vl attr-dict `:` type($buffer) `,` type($stride) `,` type($vl) `->` type($loaded)";
  let hasVerifier = 1;
}

def IndexLoadOp : TCRVRVV_Op<"index_load",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV index-vector load dataflow op";
  let description = [{
    Records one bounded plugin-local RVV index-vector load dataflow step. The
    op is nested directly under tcrv_rvv.with_vl, consumes one explicit index
    input-buffer runtime ABI value plus the active !tcrv_rvv.vl token, carries
    the explicit index EEW, and produces one !tcrv_rvv.index_vector token.

    The index buffer role and index EEW are structural body authority for the
    bounded indexed gather slice. Common EmitC/export code must not infer
    index semantics from route ids, ABI strings, artifact names, or intrinsic
    spellings.
  }];

  let arguments = (ins
    AnyType:$index,
    AnyType:$vl,
    I64Attr:$index_eew
  );
  let results = (outs AnyType:$loaded);

  let assemblyFormat =
    "$index `,` $vl attr-dict `:` type($index) `,` type($vl) `->` type($loaded)";
  let hasVerifier = 1;
}

def IndexedLoadOp : TCRVRVV_Op<"indexed_load",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV indexed data load dataflow op";
  let description = [{
    Records one bounded plugin-local RVV indexed data load step. The op is
    nested directly under tcrv_rvv.with_vl, consumes one explicit data input
    buffer runtime ABI value, one !tcrv_rvv.index_vector produced by
    tcrv_rvv.index_load, and the active !tcrv_rvv.vl token. It carries the
    explicit index EEW and offset unit and produces one data
    !tcrv_rvv.vector token.

    Data dtype/config, index EEW/offset unit, and memory roles come from this
    typed body. The RVV plugin derives the concrete indexed-load intrinsic and
    any element-to-byte offset conversion after legality validation.
  }];

  let arguments = (ins
    AnyType:$data,
    AnyType:$indices,
    AnyType:$vl,
    I64Attr:$index_eew,
    StrAttr:$offset_unit
  );
  let results = (outs AnyType:$loaded);

  let assemblyFormat =
    "$data `,` $indices `,` $vl attr-dict `:` type($data) `,` type($indices) `,` type($vl) `->` type($loaded)";
  let hasVerifier = 1;
}

def IndexedStoreOp : TCRVRVV_Op<"indexed_store",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV indexed data store dataflow op";
  let description = [{
    Records one bounded plugin-local RVV indexed destination store step. The
    op is nested directly under tcrv_rvv.with_vl, consumes one explicit output
    buffer runtime ABI value, one !tcrv_rvv.index_vector produced by
    tcrv_rvv.index_load, one typed data !tcrv_rvv.vector value, and the active
    !tcrv_rvv.vl token. It carries the explicit index EEW, offset unit, and
    unique-index policy required by this bounded scatter slice.

    Data dtype/config, index EEW/offset unit, destination memory role, and
    indexed-store form come from this typed body. The RVV plugin derives the
    concrete indexed-store intrinsic and any element-to-byte offset conversion
    after legality validation. Duplicate-index resolution is not modeled by
    this bounded slice, so only explicitly unique index vectors are accepted.
  }];

  let arguments = (ins
    AnyType:$destination,
    AnyType:$indices,
    AnyType:$value,
    AnyType:$vl,
    I64Attr:$index_eew,
    StrAttr:$offset_unit,
    StrAttr:$index_uniqueness
  );

  let assemblyFormat =
    "$destination `,` $indices `,` $value `,` $vl attr-dict `:` type($destination) `,` type($indices) `,` type($value) `,` type($vl)";
  let hasVerifier = 1;
}

def Segment2LoadOp : TCRVRVV_Op<"segment2_load",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV segment2 interleaved load dataflow op";
  let description = [{
    Records one bounded plugin-local RVV segment2 source load dataflow step.
    The op is nested directly under tcrv_rvv.with_vl, consumes one explicit
    interleaved source-buffer runtime ABI value plus the active !tcrv_rvv.vl
    token, carries segment_count = 2, source memory form, and field0/field1
    result role facts, and produces two same-typed !tcrv_rvv.vector values.

    Segment count, field order, source memory form, dtype/config, and runtime
    ABI provenance come from this typed body fact. The RVV plugin derives the
    tuple C type, segmented-load leaf, field extraction leaves, and destination
    stores from these facts; route ids, artifact names, ABI strings, exact
    intrinsic spellings, descriptors, and common EmitC/export code must not
    infer segmented memory semantics.
  }];

  let arguments = (ins
    AnyType:$source,
    AnyType:$vl,
    I64Attr:$segment_count,
    StrAttr:$source_memory_form,
    StrAttr:$field0_role,
    StrAttr:$field1_role
  );
  let results = (outs AnyType:$field0, AnyType:$field1);

  let assemblyFormat =
    "$source `,` $vl attr-dict `:` type($source) `,` type($vl) `->` type($field0) `,` type($field1)";
  let hasVerifier = 1;
}

def Segment2StoreOp : TCRVRVV_Op<"segment2_store",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV segment2 interleaved store dataflow op";
  let description = [{
    Records one bounded plugin-local RVV segment2 destination store dataflow
    step. The op is nested directly under tcrv_rvv.with_vl, consumes one
    explicit interleaved destination-buffer runtime ABI value, field0 and
    field1 !tcrv_rvv.vector values, and the active !tcrv_rvv.vl token. It
    carries segment_count = 2, destination memory form, and field0/field1
    source role facts.

    Segment count, field order, destination memory form, dtype/config, and
    runtime ABI provenance come from this typed body fact. The RVV plugin
    derives the tuple C type and segmented-store leaf from these facts; route
    ids, artifact names, ABI strings, exact intrinsic spellings, descriptors,
    and common EmitC/export code must not infer segmented memory semantics.
  }];

  let arguments = (ins
    AnyType:$destination,
    AnyType:$field0,
    AnyType:$field1,
    AnyType:$vl,
    I64Attr:$segment_count,
    StrAttr:$destination_memory_form,
    StrAttr:$field0_role,
    StrAttr:$field1_role
  );

  let assemblyFormat =
    "$destination `,` $field0 `,` $field1 `,` $vl attr-dict `:` type($destination) `,` type($field0) `,` type($field1) `,` type($vl)";
  let hasVerifier = 1;
}

def BinaryOp : TCRVRVV_Op<"binary",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV binary arithmetic dataflow op";
  let description = [{
    Records one generic plugin-local RVV vector binary arithmetic dataflow
    step. The op is nested directly under tcrv_rvv.with_vl, consumes two
    same-typed !tcrv_rvv.vector operands and the active !tcrv_rvv.vl token,
    carries an explicit operation kind, and produces one same-typed
    !tcrv_rvv.vector result.

    For the current Stage 1 retained arithmetic slice, kind is limited to
    add, sub, or mul. Dtype/config/operation authority comes from the typed
    body and config, not from old i32 helper names or route metadata.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$vl,
    StrAttr:$kind
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $vl attr-dict `:` type($lhs) `,` type($rhs) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def MaskedBinaryOp : TCRVRVV_Op<"masked_binary",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV masked binary arithmetic dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV masked binary arithmetic
    step. The op is nested directly under tcrv_rvv.with_vl, consumes one
    !tcrv_rvv.mask predicate, one passthrough/maskedoff !tcrv_rvv.vector
    value, two same-typed !tcrv_rvv.vector arithmetic operands, and the active
    !tcrv_rvv.vl token. It carries an explicit operation kind and produces one
    same-typed !tcrv_rvv.vector result.

    For the current bounded Stage 2 mask/tail-policy slice, kind is limited to
    add and the mask must come from generic tcrv_rvv.compare in the same
    selected VL scope. The passthrough vector is structural body authority for
    inactive-lane behavior; common EmitC/export must not infer it from route
    ids, ABI names, or artifact metadata.
  }];

  let arguments = (ins
    AnyType:$mask,
    AnyType:$passthrough,
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$vl,
    StrAttr:$kind
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$mask `,` $passthrough `,` $lhs `,` $rhs `,` $vl attr-dict `:` type($mask) `,` type($passthrough) `,` type($lhs) `,` type($rhs) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def CompareOp : TCRVRVV_Op<"compare",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV predicate-producing compare dataflow op";
  let description = [{
    Records one generic plugin-local RVV vector compare dataflow step. The op
    is nested directly under tcrv_rvv.with_vl, consumes two same-typed
    !tcrv_rvv.vector operands and the active !tcrv_rvv.vl token, carries an
    explicit comparison kind, and produces one matching !tcrv_rvv.mask token.

    For the current bounded Stage 2 predicate slices, kind is limited to eq or
    slt. Compare semantics come only from this typed operation in the selected
    RVV body.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$vl,
    StrAttr:$kind
  );
  let results = (outs AnyType:$mask);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $vl attr-dict `:` type($lhs) `,` type($rhs) `,` type($vl) `->` type($mask)";
  let hasVerifier = 1;
}

def SelectOp : TCRVRVV_Op<"select",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV predicate-consuming select dataflow op";
  let description = [{
    Records one generic plugin-local RVV vector select dataflow step. The op is
    nested directly under tcrv_rvv.with_vl, consumes one !tcrv_rvv.mask
    predicate produced by tcrv_rvv.compare in the same selected VL scope,
    consumes true and false !tcrv_rvv.vector values, consumes the active
    !tcrv_rvv.vl token, and produces one selected !tcrv_rvv.vector value.

    Select semantics come only from this typed operation and its explicit
    predicate operand in the selected RVV body.
  }];

  let arguments = (ins
    AnyType:$mask,
    AnyType:$true_value,
    AnyType:$false_value,
    AnyType:$vl
  );
  let results = (outs AnyType:$selected);

  let assemblyFormat =
    "$mask `,` $true_value `,` $false_value `,` $vl attr-dict `:` type($mask) `,` type($true_value) `,` type($false_value) `,` type($vl) `->` type($selected)";
  let hasVerifier = 1;
}

def ReduceOp : TCRVRVV_Op<"reduce",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV reduction/accumulation dataflow op";
  let description = [{
    Records one generic plugin-local RVV reduction/accumulation dataflow step.
    The op is nested directly under tcrv_rvv.with_vl, consumes one typed
    !tcrv_rvv.vector input, one typed !tcrv_rvv.vector accumulator seed, and
    the active !tcrv_rvv.vl token. It carries an explicit reduction kind and
    produces one typed !tcrv_rvv.vector accumulator/result token.

    For the current bounded Stage 2 skeleton, kind is limited to add. The
    vector input, accumulator, result, VL, SEW, LMUL, policy, accumulator
    layout, and result layout facts are structural in the selected RVV body.
    Route ids, helper names, ABI strings, artifact names, status fields, and
    common EmitC/export code must not infer or choose reduction semantics.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$accumulator,
    AnyType:$vl,
    StrAttr:$kind,
    OptionalAttr<StrAttr>:$accumulator_layout,
    OptionalAttr<StrAttr>:$result_layout
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$input `,` $accumulator `,` $vl attr-dict `:` type($input) `,` type($accumulator) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def StandaloneReduceOp : TCRVRVV_Op<"standalone_reduce",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV standalone scalar reduction dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV standalone horizontal
    reduction dataflow step. The op is nested directly under tcrv_rvv.with_vl,
    consumes one typed !tcrv_rvv.vector input, one explicit scalar accumulator
    seed runtime ABI value, and the active !tcrv_rvv.vl token. It carries an
    explicit reduction kind plus accumulator/result layout facts, and produces
    one typed !tcrv_rvv.vector token whose lane 0 is the scalar output boundary.

    For the current bounded Stage 2 slice, kind is limited to add, representing
    acc_seed + sum(input) over runtime n. Input dtype/config, accumulator seed
    boundary, scalar output layout, VL, policy, and runtime n facts are
    structural in the selected RVV body. Route ids, helper names, ABI strings,
    artifact names, status fields, exact intrinsic spellings, and common
    EmitC/export code must not infer or choose standalone reduction semantics.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$accumulator_seed,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$input `,` $accumulator_seed `,` $vl attr-dict `:` type($input) `,` type($accumulator_seed) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def MAccOp : TCRVRVV_Op<"macc",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV multiply-accumulate dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV multiply-accumulate dataflow
    step. The op is nested directly under tcrv_rvv.with_vl, consumes lhs and
    rhs typed !tcrv_rvv.vector operands, an explicit typed accumulator vector,
    and the active !tcrv_rvv.vl token. It carries an explicit accumulation kind
    and produces one typed !tcrv_rvv.vector result.

    For the current bounded Stage 2 contraction-supporting slice, kind is
    limited to add, representing accumulator + lhs * rhs. Lhs, rhs,
    accumulator, result, VL, SEW, LMUL, policy, accumulator layout, result
    layout, and memory-form facts are structural in the selected RVV body.
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, and common EmitC/export code must not infer or choose
    multiply-accumulate semantics.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$accumulator,
    AnyType:$vl,
    StrAttr:$kind,
    OptionalAttr<StrAttr>:$accumulator_layout,
    OptionalAttr<StrAttr>:$result_layout
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $accumulator `,` $vl attr-dict `:` type($lhs) `,` type($rhs) `,` type($accumulator) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def WideningMAccOp : TCRVRVV_Op<"widening_macc",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV widening multiply-accumulate dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV signed widening
    multiply-accumulate dataflow step. The op is nested directly under
    tcrv_rvv.with_vl, consumes two i16 !tcrv_rvv.vector source operands, an
    explicit i32 typed accumulator vector, and the active !tcrv_rvv.vl token.
    It carries explicit operation kind, accumulator layout, result layout, and
    widening macc relation facts, and produces one i32 !tcrv_rvv.vector result.

    For the current bounded Stage 2 mixed-width slice, kind is limited to
    signed_widening_macc_add, representing
    accumulator + sign_extend(lhs) * sign_extend(rhs). Lhs, rhs, accumulator,
    result, VL, source/result SEW/LMUL, policy, accumulator layout, result
    layout, and memory-form facts are structural in the selected RVV body.
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, and common EmitC/export code must not infer or choose
    widening multiply-accumulate semantics.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$accumulator,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    StrAttr:$macc_relation
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $accumulator `,` $vl attr-dict `:` type($lhs) `,` type($rhs) `,` type($accumulator) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def WideningDotReduceOp : TCRVRVV_Op<"widening_dot_reduce",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV widening dot-product reduction dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV signed widening dot-product
    reduction dataflow step. The op is nested directly under tcrv_rvv.with_vl,
    consumes two i16 !tcrv_rvv.vector source operands, an explicit i32
    accumulator seed runtime ABI value, and the active !tcrv_rvv.vl token. It
    carries explicit operation kind, accumulator layout, scalar result layout,
    and dot-product relation facts, and produces one i32 !tcrv_rvv.vector
    token whose lane 0 is the scalar reduction result boundary.

    For the current bounded Stage 2 scalar-reduction slice, kind is limited to
    signed_widening_dot_reduce_add, representing
    acc_seed + sum(sign_extend(lhs) * sign_extend(rhs)) over runtime n.
    Lhs, rhs, accumulator seed boundary, result, VL, source/result SEW/LMUL,
    policy, accumulator layout, result layout, and memory-form facts are
    structural in the selected RVV body. Route ids, helper names, ABI strings,
    artifact names, status fields, exact intrinsic spellings, and common
    EmitC/export code must not infer or choose widening dot-product reduction
    semantics.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$accumulator_seed,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    StrAttr:$dot_product_relation
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $accumulator_seed `,` $vl attr-dict `:` type($lhs) `,` type($rhs) `,` type($accumulator_seed) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def MaskedWideningDotReduceOp : TCRVRVV_Op<"masked_widening_dot_reduce",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV masked widening dot-product reduction dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV signed masked widening
    dot-product reduction dataflow step. The op is nested directly under
    tcrv_rvv.with_vl, consumes a compare-produced !tcrv_rvv.mask, two i16
    !tcrv_rvv.vector source operands, an explicit i32 accumulator seed runtime
    ABI value, and the active !tcrv_rvv.vl token. It carries explicit
    operation kind, accumulator layout, scalar result layout, dot-product
    relation, mask role/source/config facts, and produces one i32
    !tcrv_rvv.vector token whose lane 0 is the scalar reduction result
    boundary.

    For the current bounded Stage 2 scalar-reduction slice, kind is limited to
    signed_masked_widening_dot_reduce_add, representing
    acc_seed + sum(mask ? sign_extend(lhs) * sign_extend(rhs) : 0) over
    runtime n. Mask provenance, lhs/rhs source facts, accumulator seed
    boundary, result, VL, source/result SEW/LMUL, policy, accumulator layout,
    result layout, and memory-form facts are structural in the selected RVV
    body. Route ids, helper names, ABI strings, artifact names, status fields,
    exact intrinsic spellings, and common EmitC/export code must not infer or
    choose masked widening dot-product reduction semantics.
  }];

  let arguments = (ins
    AnyType:$mask,
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$accumulator_seed,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    StrAttr:$dot_product_relation
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$mask `,` $lhs `,` $rhs `,` $accumulator_seed `,` $vl attr-dict `:` type($mask) `,` type($lhs) `,` type($rhs) `,` type($accumulator_seed) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def WideningConvertOp : TCRVRVV_Op<"widening_convert",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV widening conversion dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV signed widening conversion
    step. The op is nested directly under tcrv_rvv.with_vl, consumes one
    source !tcrv_rvv.vector and the active !tcrv_rvv.vl token, carries an
    explicit conversion kind, and produces one destination !tcrv_rvv.vector.

    The current executable slices support bounded signed widening conversions
    i32 LMUL m1 to i64 LMUL m2 and i16 LMUL mf2 to i32 LMUL m1.
    Source/destination dtype, SEW, LMUL, VL, policy, and operation kind are
    structural typed-body facts consumed by the RVV plugin. Route ids, helper
    names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, and common EmitC/export code must not infer or choose
    conversion semantics.
  }];

  let arguments = (ins
    AnyType:$source,
    AnyType:$vl,
    StrAttr:$kind
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$source `,` $vl attr-dict `:` type($source) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def MoveOp : TCRVRVV_Op<"move",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV movement dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV vector movement step. The op
    is nested directly under tcrv_rvv.with_vl, consumes one typed
    !tcrv_rvv.vector value and the active !tcrv_rvv.vl token, carries an
    explicit movement kind, and produces one same-typed !tcrv_rvv.vector
    result.

    For the current bounded Stage 2 strided memory movement slice, kind is
    limited to copy. The route provider consumes this op as structural typed
    body authority that the strided loaded vector is the unit-stride store
    value; common EmitC/export must not infer that from route ids, ABI names,
    artifact metadata, or C strings.
  }];

  let arguments = (ins
    AnyType:$source,
    AnyType:$vl,
    StrAttr:$kind
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$source `,` $vl attr-dict `:` type($source) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def MaskedMoveOp : TCRVRVV_Op<"masked_move",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV masked memory movement dataflow op";
  let description = [{
    Records one bounded plugin-local RVV masked movement step for
    unit-stride source-to-destination memory movement. The op is nested
    directly under tcrv_rvv.with_vl, consumes one !tcrv_rvv.mask produced by
    either tcrv_rvv.mask_load or tcrv_rvv.compare, one active source
    !tcrv_rvv.vector, one inactive old-destination passthrough
    !tcrv_rvv.vector, and the active !tcrv_rvv.vl token. It carries an explicit
    movement kind and produces the same-typed vector to store.

    For the current bounded Stage 2 masked memory slice, kind is limited to
    active-source-preserve-old-destination. The active/inactive lane behavior
    is structural body authority and must not be inferred from route ids, ABI
    names, artifact metadata, exact intrinsic spelling, or harness sentinels.
  }];

  let arguments = (ins
    AnyType:$mask,
    AnyType:$active_value,
    AnyType:$inactive_passthrough,
    AnyType:$vl,
    StrAttr:$kind
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$mask `,` $active_value `,` $inactive_passthrough `,` $vl attr-dict `:` type($mask) `,` type($active_value) `,` type($inactive_passthrough) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def StoreOp : TCRVRVV_Op<"store",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV vector store dataflow op";
  let description = [{
    Records one generic plugin-local RVV vector store dataflow step. The op is
    nested directly under tcrv_rvv.with_vl, consumes one explicit output
    runtime ABI buffer SSA value, one !tcrv_rvv.vector dataflow value matching
    the enclosing config, and the active !tcrv_rvv.vl token.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$value,
    AnyType:$vl
  );

  let assemblyFormat =
    "$buffer `,` $value `,` $vl attr-dict `:` type($buffer) `,` type($value) `,` type($vl)";
  let hasVerifier = 1;
}

def MaskedStoreOp : TCRVRVV_Op<"masked_store",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV masked unit-stride store dataflow op";
  let description = [{
    Records one bounded plugin-local RVV masked unit-stride store step. The op
    is nested directly under tcrv_rvv.with_vl, consumes one explicit output
    runtime ABI buffer SSA value, one !tcrv_rvv.mask predicate, one
    !tcrv_rvv.vector payload value matching the enclosing config, and the
    active !tcrv_rvv.vl token. It carries explicit masked-store memory-form
    and inactive-lane preservation facts.

    For the current bounded Stage 2 mask/tail-policy executable slice,
    memory_form is limited to masked-unit-store and inactive_lane_policy is
    limited to preserve-output-on-false-lanes. False mask lanes and tail lanes
    are structural masked-store behavior; common EmitC/export must not infer
    them from route ids, ABI names, artifact metadata, exact intrinsic
    spelling, or harness sentinels.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$mask,
    AnyType:$value,
    AnyType:$vl,
    StrAttr:$memory_form,
    StrAttr:$inactive_lane_policy
  );

  let assemblyFormat =
    "$buffer `,` $mask `,` $value `,` $vl attr-dict `:` type($buffer) `,` type($mask) `,` type($value) `,` type($vl)";
  let hasVerifier = 1;
}

def StridedStoreOp : TCRVRVV_Op<"strided_store",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV strided vector store dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV strided vector store dataflow
    step. The op is nested directly under tcrv_rvv.with_vl, consumes one
    explicit output buffer runtime ABI value, one generic !tcrv_rvv.vector
    value matching the enclosing config, one explicit runtime output stride
    SSA value, and the active !tcrv_rvv.vl token.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$value,
    Index:$stride,
    AnyType:$vl
  );

  let assemblyFormat =
    "$buffer `,` $value `,` $stride `,` $vl attr-dict `:` type($buffer) `,` type($value) `,` type($stride) `,` type($vl)";
  let hasVerifier = 1;
}

def I32LoadOp : TCRVRVV_Op<"i32_load"> {
  let summary = "deprecated bounded RVV i32 load parse-only dataflow op";
  let description = [{
    Deprecated Stage1 inventory for one bounded plugin-local RVV i32 load
    parser/verifier step. The op is nested directly under tcrv_rvv.with_vl,
    consumes one explicit runtime ABI buffer SSA value plus the same runtime
    !tcrv_rvv.vl control token as the surrounding scope, and produces one
    bounded !tcrv_rvv.i32m1 or !tcrv_rvv.i32m2 dataflow token matching the
    enclosing LMUL scope.

    Buffer names, C types, and ABI roles come from the defining
    tcrv_rvv.runtime_abi_value op, not from descriptor metadata or target-side
    inference. Runtime n/AVL/VL remains represented by real runtime
    SSA/control surfaces. Deleted local element-count metadata is not accepted
    here. This op is not route-supported, does not implement the EmitC
    lowerable interface, and must not authorize target artifacts or
    source-front-door materialization.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$vl
  );
  let results = (outs AnyType:$loaded);

  let assemblyFormat =
    "$buffer `,` $vl attr-dict `:` type($buffer) `,` type($vl) `->` type($loaded)";
  let hasVerifier = 1;
}

def I32BroadcastLoadOp : TCRVRVV_Op<"i32_broadcast_load"> {
  let summary = "deprecated bounded RVV i32 broadcast parse-only dataflow op";
  let description = [{
    Deprecated Stage1 inventory for one bounded plugin-local RVV i32 broadcast
    parser/verifier step. The op is nested directly under tcrv_rvv.with_vl,
    consumes one explicit RHS runtime ABI buffer SSA value plus the same
    runtime !tcrv_rvv.vl control token as the surrounding scope, and produces
    one bounded !tcrv_rvv.i32m1 or !tcrv_rvv.i32m2 value matching the enclosing
    LMUL scope.

    This is retained only so stale i32 broadcast inputs stay diagnosable. The
    broadcast source is the explicit RHS ABI operand, not a route id, artifact
    name, source-front-door name, descriptor, ABI parameter name, or
    target/export-side inference. This op is not route-supported, does not
    implement the EmitC lowerable interface, and must not authorize target
    artifacts or source-front-door materialization.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$vl
  );
  let results = (outs AnyType:$broadcast);

  let assemblyFormat =
    "$buffer `,` $vl attr-dict `:` type($buffer) `,` type($vl) `->` type($broadcast)";
  let hasVerifier = 1;
}

def I32AddOp : TCRVRVV_Op<"i32_add"> {
  let summary = "deprecated bounded RVV i32 add parse-only dataflow op";
  let description = [{
    Deprecated Stage1 inventory for one bounded plugin-local RVV i32 vector
    add parser/verifier step. The op is nested directly under tcrv_rvv.with_vl,
    consumes two same-LMUL i32 dataflow tokens and the same runtime
    !tcrv_rvv.vl control token as the surrounding scope, and produces one
    bounded i32 dataflow token matching the enclosing LMUL scope.

    This is not generic arithmetic IR, is not route-supported, does not
    implement the EmitC lowerable interface, and does not authorize RVV
    artifact emission.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$vl
  );
  let results = (outs AnyType:$sum);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $vl attr-dict `:` type($lhs) `,` type($rhs) `,` type($vl) `->` type($sum)";
  let hasVerifier = 1;
}

def I32SubOp : TCRVRVV_Op<"i32_sub"> {
  let summary = "deprecated bounded RVV i32 subtract parse-only dataflow op";
  let description = [{
    Deprecated Stage1 inventory for one bounded plugin-local RVV i32 vector
    subtract parser/verifier step. The op is nested directly under
    tcrv_rvv.with_vl, consumes two same-LMUL i32 dataflow tokens and the same
    runtime !tcrv_rvv.vl control token as the surrounding scope, and produces
    one bounded i32 dataflow token matching the enclosing LMUL scope.

    This is not generic arithmetic IR, is not route-supported, does not
    implement the EmitC lowerable interface, and does not authorize RVV
    artifact emission.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$vl
  );
  let results = (outs AnyType:$difference);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $vl attr-dict `:` type($lhs) `,` type($rhs) `,` type($vl) `->` type($difference)";
  let hasVerifier = 1;
}

def I32MulOp : TCRVRVV_Op<"i32_mul"> {
  let summary = "deprecated bounded RVV i32 multiply parse-only dataflow op";
  let description = [{
    Deprecated Stage1 inventory for one bounded plugin-local RVV i32 vector
    multiply parser/verifier step. The op is nested directly under
    tcrv_rvv.with_vl, consumes two same-LMUL i32 dataflow tokens and the same
    runtime !tcrv_rvv.vl control token as the surrounding scope, and produces
    one bounded i32 dataflow token matching the enclosing LMUL scope.

    This is not generic arithmetic IR, is not route-supported, does not
    implement the EmitC lowerable interface, and does not authorize RVV
    artifact emission.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$vl
  );
  let results = (outs AnyType:$product);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $vl attr-dict `:` type($lhs) `,` type($rhs) `,` type($vl) `->` type($product)";
  let hasVerifier = 1;
}

def I32CmpEqOp : TCRVRVV_Op<"i32_cmp_eq"> {
  let summary = "deprecated bounded RVV i32m1 compare parse-only dataflow op";
  let description = [{
    Deprecated Stage1 inventory for one bounded plugin-local RVV i32m1
    predicate-producing compare parser/verifier step. The op is nested
    directly under tcrv_rvv.with_vl, consumes two !tcrv_rvv.i32m1 dataflow
    tokens and the same runtime !tcrv_rvv.vl control token as the surrounding
    scope, and produces one !tcrv_rvv.i32m1_mask predicate token.

    This op is retained only so stale i32 compare inputs stay diagnosable. It
    is not route-supported, does not implement the EmitC lowerable interface,
    and must not authorize selected-body realization, target artifacts, or
    source-front-door materialization.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$vl
  );
  let results = (outs AnyType:$mask);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $vl attr-dict `:` type($lhs) `,` type($rhs) `,` type($vl) `->` type($mask)";
  let hasVerifier = 1;
}

def I32SelectOp : TCRVRVV_Op<"i32_select"> {
  let summary = "deprecated bounded RVV i32m1 select parse-only dataflow op";
  let description = [{
    Deprecated Stage1 inventory for one bounded plugin-local RVV i32m1 select
    parser/verifier step. The op is nested directly under tcrv_rvv.with_vl,
    consumes one !tcrv_rvv.i32m1_mask predicate produced by
    tcrv_rvv.i32_cmp_eq in the same VL scope, consumes true and false
    !tcrv_rvv.i32m1 dataflow values, consumes the same runtime
    !tcrv_rvv.vl control token as the surrounding scope, and produces one
    selected !tcrv_rvv.i32m1 value.

    This op is retained only so stale i32 select inputs stay diagnosable. It
    is not route-supported, does not implement the EmitC lowerable interface,
    and must not authorize selected-body realization, target artifacts, or
    source-front-door materialization.
  }];

  let arguments = (ins
    AnyType:$mask,
    AnyType:$true_value,
    AnyType:$false_value,
    AnyType:$vl
  );
  let results = (outs AnyType:$selected);

  let assemblyFormat =
    "$mask `,` $true_value `,` $false_value `,` $vl attr-dict `:` type($mask) `,` type($true_value) `,` type($false_value) `,` type($vl) `->` type($selected)";
  let hasVerifier = 1;
}

def I32StoreOp : TCRVRVV_Op<"i32_store"> {
  let summary = "deprecated bounded RVV i32 store parse-only dataflow op";
  let description = [{
    Deprecated Stage1 inventory for one bounded plugin-local RVV i32 store
    parser/verifier step. The op is nested directly under tcrv_rvv.with_vl,
    consumes one explicit output runtime ABI buffer SSA value, one bounded i32
    dataflow token matching the enclosing LMUL scope, and the same runtime
    !tcrv_rvv.vl control token as the surrounding scope.

    Output C name, C type, and ABI role come from the defining
    tcrv_rvv.runtime_abi_value op, not from descriptor metadata or target-side
    inference. Runtime n/AVL/VL remains represented by real runtime
    SSA/control surfaces. Deleted local element-count metadata is not accepted
    here. This op is not route-supported, does not implement the EmitC
    lowerable interface, and must not authorize target artifacts or
    source-front-door materialization.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$value,
    AnyType:$vl
  );

  let assemblyFormat =
    "$buffer `,` $value `,` $vl attr-dict `:` type($buffer) `,` type($value) `,` type($vl)";
  let hasVerifier = 1;
}

#endif // TIANCHENRV_DIALECT_RVV_IR_RVVOPS_TD
