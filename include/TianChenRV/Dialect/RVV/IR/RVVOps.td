#ifndef TIANCHENRV_DIALECT_RVV_IR_RVVOPS_TD
#define TIANCHENRV_DIALECT_RVV_IR_RVVOPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.td"
include "TianChenRV/Conversion/EmitC/TunableScheduleOpInterface.td"

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
    or other typed RVV body ops. Optional exec_binding links the value to the
    same-kernel tcrv.exec.mem_window or tcrv.exec.runtime_param ABI envelope
    declaration that introduced the role.

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
    OptionalAttr<FlatSymbolRefAttr>:$exec_binding,
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
    OptionalAttr<StrAttr>:$rvv_emitc_route_mapping,
    OptionalAttr<I64Attr>:$unroll_factor
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$vl attr-dict-with-keyword $body `:` type($vl)";
  let hasVerifier = 1;
}

def VSetVLRegionMarkerOp : TCRVRVV_Op<"vsetvl_region_marker"> {
  let summary = "RVV plugin-owned vsetvl placement marker";
  let description = [{
    Records one RVV plugin-local selected-body realization boundary for a
    resource-aware vsetvl placement decision. The op is nested directly under
    tcrv_rvv.with_vl, consumes the active !tcrv_rvv.vl token, and carries the
    realized region phase, one-based region index, total region count,
    selected resource-planning contract, and resource decision string.

    This op is structural schedule evidence only. It does not model compute,
    dtype, memory access, an intrinsic wrapper, route id, artifact metadata,
    source-front-door marker, or executable proof. The RVV provider may use it
    to validate that resource facts have been consumed into realized body
    structure before building a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$vl,
    StrAttr:$phase,
    OptionalAttr<StrAttr>:$planning_contract,
    I64Attr:$region_index,
    I64Attr:$region_count,
    StrAttr:$resource_decision
  );

  let assemblyFormat = "$vl attr-dict `:` type($vl)";
  let hasVerifier = 1;
}

def GearboxCrossRegionHandoffOp
    : TCRVRVV_Op<"gearbox_cross_region_handoff"> {
  let summary =
      "RVV plugin-owned Gearbox value and runtime boundary handoff";
  let description = [{
    Records the bounded structural handoff between the low-precision Gearbox
    product/reduction region and the dequant/store region. The op is nested
    directly under tcrv_rvv.with_vl, consumes the reduced i32 vector value, the
    active !tcrv_rvv.vl token, and the explicit runtime n/AVL SSA value, then
    forwards the reduced value as an SSA result for tcrv_rvv.dequantize.

    This operation is a bridge toward true multi-region selected-body
    realization. It makes product/reduction/dequant value handoff, runtime AVL
    source, VL token, phase ordering, region count, selected resource
    candidate, resource budget, operand packing/unpack policy, remediation plan
    facts when applicable, primitive-chain resource facts including source
    signedness/load/extension and widening-product multiplicand-role/extension
    policy plus widening-product/reduction candidate facts, and the
    producer/consumer region scopes structural before provider
    route support. It is not a math op, intrinsic wrapper, descriptor, route
    id, artifact metadata, or executable proof.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$vl,
    AnyType:$runtime_avl,
    StrAttr:$contract,
    StrAttr:$from_phase,
    StrAttr:$to_phase,
    I64Attr:$region_count,
    StrAttr:$runtime_avl_source,
    StrAttr:$resource_decision,
    StrAttr:$resource_candidate_set,
    StrAttr:$resource_selected_candidate,
    I64Attr:$resource_candidate_count,
    I64Attr:$resource_legal_candidate_count,
    I64Attr:$resource_selected_candidate_index,
    StrAttr:$operand_form,
    StrAttr:$packing_layout,
    StrAttr:$unpack_intent,
    I64Attr:$peak_live_vector_groups,
    I64Attr:$vector_register_budget,
    OptionalAttr<StrAttr>:$resource_cost_contract,
    OptionalAttr<StrAttr>:$resource_cost_model,
    OptionalAttr<I64Attr>:$resource_cost_loop_body_steps,
    OptionalAttr<StrAttr>:$resource_cost_blocker,
    OptionalAttr<StrAttr>:$performance_admission_decision,
    OptionalAttr<StrAttr>:$performance_admission_closure,
    OptionalAttr<StrAttr>:$performance_admission_reopen_requirement,
    OptionalAttr<StrAttr>:$beyond_local_repair_admission_contract,
    OptionalAttr<StrAttr>:$beyond_local_repair_admission_decision,
    OptionalAttr<StrAttr>:$beyond_local_repair_admission_blocker,
    OptionalAttr<StrAttr>:$beyond_local_repair_admission_reopen_requirement,
    I64Attr:$product_region_index,
    I64Attr:$dequant_region_index,
    OptionalAttr<StrAttr>:$remediation_plan_contract,
    OptionalAttr<StrAttr>:$remediation_plan,
    OptionalAttr<StrAttr>:$remediation_statement_strategy,
    OptionalAttr<StrAttr>:$remediation_vector_budget,
    OptionalAttr<StrAttr>:$remediation_schedule_contract,
    OptionalAttr<StrAttr>:$remediation_unpack_plan,
    OptionalAttr<StrAttr>:$remediation_product_plan,
    OptionalAttr<StrAttr>:$remediation_reduction_plan,
    OptionalAttr<StrAttr>:$remediation_vl_plan,
    OptionalAttr<StrAttr>:$schedule_decision_contract,
    OptionalAttr<StrAttr>:$schedule_decision,
    OptionalAttr<StrAttr>:$schedule_decision_reason,
    StrAttr:$producer_scope,
    StrAttr:$consumer_scope,
    StrAttr:$primitive_chain_contract,
    StrAttr:$primitive_chain_kind,
    StrAttr:$primitive_source_signedness,
    StrAttr:$primitive_source_load,
    StrAttr:$primitive_source_extension,
    StrAttr:$widening_product_multiplicand_roles,
    StrAttr:$widening_product_extension_policy,
    StrAttr:$widening_product_candidate_fact,
    StrAttr:$reduction_candidate_fact,
    StrAttr:$primitive_widening_product_relation,
    StrAttr:$primitive_product_reduction_chain_relation,
    StrAttr:$primitive_widening_product_intrinsic,
    StrAttr:$primitive_reduction_intrinsic,
    StrAttr:$primitive_scalar_seed_splat_intrinsic,
    StrAttr:$primitive_accumulator_layout,
    StrAttr:$primitive_result_layout,
    StrAttr:$primitive_reduction_store_vl
  );
  let results = (outs AnyType:$output);

  let assemblyFormat =
    "$input `,` $vl `,` $runtime_avl attr-dict `:` type($input) `,` type($vl) `,` type($runtime_avl) `->` type($output)";
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
    bounded scalar-broadcast elementwise route.

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

def TypedRuntimeScalarSplatStorePreRealizedBodyOp
    : TCRVRVV_Op<"typed_runtime_scalar_splat_store_pre_realized_body"> {
  let summary = "typed pre-realized RVV runtime scalar splat-store selected body";
  let description = [{
    Carries the RVV plugin-local pre-realized selected body for the bounded
    runtime scalar-to-vector splat followed by unit-stride store route.
    The op is nested directly in a selected tcrv.exec.variant and binds an
    explicit RHS scalar runtime ABI value, output buffer, runtime n/AVL, and
    compile-time RVV config facts.

    This op is not a route id, helper-string authority, descriptor, source
    front door, direct-C exporter, constant substitution, artifact mirror, or
    common EmitC semantic inference hook. The RVV plugin must first realize it
    into explicit setvl/with_vl/splat/store tcrv_rvv body structure before the
    provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$scalar,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$scalar `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
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

def TypedRuntimeScalarCompareSelectPreRealizedBodyOp
    : TCRVRVV_Op<"typed_runtime_scalar_compare_select_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV runtime scalar compare/select body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed typed vector load is compared against a non-AVL runtime scalar
    threshold with the same element type, splatted inside the selected VL
    scope, and the produced mask selects between explicit true-value and
    false-value vector operands before storing to the output buffer. The op is
    nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for lhs,
    rhs_scalar, true value, false value, output, and runtime n/AVL together
    with compare predicate, memory form, mask role/source/config, select
    layout, compile-time RVV data/mask config, and policy.

    This op is not a scalar-add clone, constant substitution, helper-string
    authority, route label, descriptor, source-front-door marker, direct-C
    exporter, intrinsic wrapper, runtime artifact, correctness evidence, or
    performance evidence. The RVV plugin must realize it into explicit
    setvl/with_vl/load/splat/load/load/compare/select/store tcrv_rvv body
    structure before the provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs_scalar,
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
    "$lhs `,` $rhs_scalar `,` $true_value `,` $false_value `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOp
    : TCRVRVV_Op<"typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV dual runtime scalar compare mask-and select body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where two
    signed i32 vector loads are each compared against independent non-AVL
    runtime scalar thresholds splatted inside the selected VL scope, both
    compare-produced mask SSA values are composed by typed mask-and, and the
    composed mask selects between explicit true-value and false-value vector
    operands before storing to the output buffer. The op is nested directly in
    a selected tcrv.exec.variant and binds explicit runtime ABI values for
    compare lhs A, scalar threshold A, compare lhs B, scalar threshold B,
    true value, false value, output, and runtime n/AVL together with both
    compare predicates, mask composition facts, select layout, compile-time
    RVV data/mask config, and policy.

    This op is not a scalar-add clone, constant substitution, helper-string
    authority, route label, descriptor, source-front-door marker, direct-C
    exporter, intrinsic wrapper, runtime artifact, correctness evidence, or
    performance evidence. The RVV plugin must realize it into explicit
    setvl/with_vl/load/splat/load/splat/compare/compare/mask_and/load/load/
    select/store tcrv_rvv body structure before the provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs_a,
    AnyType:$rhs_scalar_a,
    AnyType:$compare_lhs_b,
    AnyType:$rhs_scalar_b,
    AnyType:$true_value,
    AnyType:$false_value,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind_a,
    StrAttr:$predicate_kind_b,
    StrAttr:$memory_form,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$mask_composition,
    StrAttr:$select_layout,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs_a `,` $rhs_scalar_a `,` $compare_lhs_b `,` $rhs_scalar_b `,` $true_value `,` $false_value `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedF32ClampSelectPreRealizedBodyOp
    : TCRVRVV_Op<"typed_f32_clamp_select_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV f32 runtime-bound clamp/select body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    typed f32 vector input is clamped by two non-AVL runtime f32 scalar bounds
    splatted inside the selected VL scope, then stored to an f32 output buffer.
    The op is nested directly in a selected tcrv.exec.variant and binds runtime
    ABI values for input, lower bound, upper bound, output, and runtime n/AVL
    together with lower/upper compare predicates, explicit bound order,
    select layout, memory form, compile-time RVV f32 config, and policy.

    This op is not a high-level activation op, q8/q4 route, one-intrinsic
    wrapper, route label, descriptor, source-front-door marker, direct-C
    exporter, runtime artifact, correctness evidence, or performance evidence.
    The RVV plugin must realize it into explicit setvl/with_vl/load/splat/
    splat/compare/select/compare/select/store tcrv_rvv body structure before
    the provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$lower_bound,
    AnyType:$upper_bound,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$lower_predicate_kind,
    StrAttr:$upper_predicate_kind,
    StrAttr:$bound_order,
    StrAttr:$select_layout,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$input `,` $lower_bound `,` $upper_bound `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedDequantClampF32EpiloguePreRealizedBodyOp
    : TCRVRVV_Op<"typed_dequant_clamp_f32_epilogue_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV i32-to-f32 dequant plus f32 clamp epilogue body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    typed signed i32 vector input is loaded, converted to f32 with a runtime
    f32 dequantization scale, clamped by runtime f32 lower and upper bounds
    using f32 compare/select structure, then stored to an f32 output buffer.
    The op is nested directly in a selected tcrv.exec.variant and binds
    runtime ABI values for lhs, scale, lower bound, upper bound, output, and
    runtime n/AVL together with explicit dequant relation, scale role, lower/
    upper compare predicates, bound order, select layout, memory form,
    compile-time RVV config, and policy.

    This op is not a q8/q4 route, matmul/contraction fusion, zero-point
    dequantization, high-level activation op, one-intrinsic wrapper, route
    label, descriptor, source-front-door marker, direct-C exporter, runtime
    artifact, correctness evidence, or performance evidence. The RVV plugin
    must realize it into explicit setvl/with_vl/load/dequantize/splat/splat/
    compare/select/compare/select/store tcrv_rvv body structure before the
    provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$scale,
    AnyType:$lower_bound,
    AnyType:$upper_bound,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$dequant_relation,
    StrAttr:$scale_role,
    StrAttr:$lower_predicate_kind,
    StrAttr:$upper_predicate_kind,
    StrAttr:$bound_order,
    StrAttr:$select_layout,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $scale `,` $lower_bound `,` $upper_bound `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp
    : TCRVRVV_Op<"typed_runtime_scalar_computed_mask_store_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV runtime scalar computed-mask store body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector load is compared against a non-AVL runtime scalar
    threshold splatted inside the selected VL scope, and the produced mask
    controls a masked contiguous store of an explicit payload vector into the
    output buffer. The op is nested directly in a selected tcrv.exec.variant
    and binds explicit runtime ABI values for lhs, rhs_scalar, payload source,
    output destination, and runtime n/AVL together with compare predicate,
    memory form, mask role/source/config, inactive-lane preservation policy,
    compile-time RVV data/mask config, and policy.

    This op is not a compare-select wrapper, external-mask store route,
    constant substitution, helper-string authority, route label, descriptor,
    source-front-door marker, direct-C exporter, intrinsic wrapper, runtime
    artifact, correctness evidence, or performance evidence. The RVV plugin
    must realize it into explicit setvl/with_vl/load/splat/compare/load/
    masked_store tcrv_rvv body structure before the provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs_scalar,
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
    "$lhs `,` $rhs_scalar `,` $source `,` $destination `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp
    : TCRVRVV_Op<"typed_runtime_scalar_computed_mask_load_store_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV runtime scalar computed-mask load-store body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector load is compared against a non-AVL runtime scalar
    threshold splatted inside the selected VL scope, and the produced mask
    controls a masked contiguous data load from an explicit source buffer with
    old-destination passthrough before storing the merged result back to the
    output buffer. The op is nested directly in a selected tcrv.exec.variant
    and binds explicit runtime ABI values for lhs, rhs_scalar, active source,
    output destination/passthrough, and runtime n/AVL together with compare
    predicate, memory form, mask role/source/config, inactive-lane
    preservation policy, compile-time RVV data/mask config, and policy.

    This op is not a compare-select wrapper, masked-store fallback,
    constant substitution, helper-string authority, route label, descriptor,
    source-front-door marker, direct-C exporter, intrinsic wrapper, runtime
    artifact, correctness evidence, or performance evidence. The RVV plugin
    must realize it into explicit setvl/with_vl/load/splat/compare/load/
    masked_load/store tcrv_rvv body structure before the provider may
    construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs_scalar,
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
    "$lhs `,` $rhs_scalar `,` $source `,` $destination `,` $n attr-dict `:` functional-type(operands, results)";
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
    standalone horizontal signed scalar-reduction slice. The op is nested directly in a
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

def TypedComputedMaskStandaloneReducePreRealizedBodyOp
    : TCRVRVV_Op<"typed_computed_mask_standalone_reduce_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV computed-mask standalone scalar reduction body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for the
    computed-mask standalone horizontal signed-reduction slice. The op is nested
    directly in a selected tcrv.exec.variant and binds explicit runtime ABI
    values for compare lhs/rhs, source input, scalar accumulator seed, scalar
    output, and runtime n/AVL together with compare predicate, mask facts,
    standalone memory form, accumulator role/layout, scalar result layout, and
    compile-time RVV config facts.

    This op is not a broad reduction framework, contraction route, legacy
    metadata authority, runtime artifact, correctness evidence, or performance
    evidence. The RVV plugin must first realize it into explicit
    setvl/with_vl/load/load/load/compare/masked_standalone_reduce/store
    tcrv_rvv body structure before the provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs,
    AnyType:$compare_rhs,
    AnyType:$source,
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
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs `,` $compare_rhs `,` $source `,` $acc `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOp
    : TCRVRVV_Op<"typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV runtime scalar computed-mask standalone scalar reduction body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 compare lhs vector is compared against a non-AVL runtime scalar
    threshold splatted inside the selected VL scope, and the produced mask
    controls a standalone horizontal signed scalar-reduction of an explicit
    payload vector. The op is nested directly in a selected tcrv.exec.variant
    and binds runtime ABI values for compare lhs, rhs_scalar threshold, source
    input, scalar accumulator seed, scalar output, and runtime n/AVL together
    with compare predicate, mask facts, standalone memory form, accumulator
    role/layout, scalar result layout, and compile-time RVV config facts.

    This op is not a compare-select wrapper, masked-store fallback, masked-load
    fallback, unmasked reduction fallback, macc fallback, constant substitution,
    helper-string authority, route label, descriptor, source-front-door marker,
    direct-C exporter, intrinsic wrapper, runtime artifact, correctness
    evidence, or performance evidence. The RVV plugin must first realize it
    into explicit setvl/with_vl/load/splat/compare/load/
    masked_standalone_reduce/store tcrv_rvv body structure before the provider
    may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs,
    AnyType:$rhs_scalar,
    AnyType:$source,
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
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs `,` $rhs_scalar `,` $source `,` $acc `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedMAccPreRealizedBodyOp
    : TCRVRVV_Op<"typed_macc_pre_realized_body"> {
  let summary = "generic typed pre-realized RVV macc selected body";
  let description = [{
    Carries the RVV plugin-local pre-realized selected body for the bounded
    multiply-accumulate route. The op is nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for lhs, rhs or
    rhs scalar, accumulator input, out, and runtime n/AVL together with selected
    operation kind, memory form, accumulator role/layout, result layout, and
    compile-time RVV config facts.

    This op is not a matmul route, contraction framework, descriptor,
    source-front-door marker, direct-C exporter, route id, intrinsic wrapper,
    runtime artifact, correctness evidence, or performance evidence. The RVV
    plugin must first realize it into explicit setvl/with_vl/load/load-or-splat/
    accumulator-load/macc/store tcrv_rvv body structure before the existing
    provider may construct a TCRVEmitCLowerableRoute.
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
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $acc `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedComputedMaskMAccPreRealizedBodyOp
    : TCRVRVV_Op<"typed_computed_mask_macc_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV computed-mask macc selected body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for the
    computed-mask multiply-accumulate route. The op is nested directly in a
    selected tcrv.exec.variant and binds explicit runtime ABI values for
    compare lhs/rhs, macc lhs/rhs payload inputs, accumulator input, output,
    and runtime n/AVL together with compare predicate, mask provenance,
    operation kind, memory form, accumulator role/layout, result layout, and
    compile-time RVV config facts.

    This op is not a matmul route, contraction framework, descriptor,
    source-front-door marker, direct-C exporter, route id, intrinsic wrapper,
    runtime artifact, correctness evidence, or performance evidence. The RVV
    plugin must first realize it into explicit setvl/with_vl/compare-loads/
    payload-loads/accumulator-load/compare/masked_macc/store tcrv_rvv body
    structure before the provider may construct a TCRVEmitCLowerableRoute.
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
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs `,` $compare_rhs `,` $lhs `,` $rhs `,` $acc `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp
    : TCRVRVV_Op<"typed_runtime_scalar_computed_mask_macc_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV runtime scalar computed-mask macc body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 compare lhs vector is compared against a non-AVL runtime scalar
    threshold splatted inside the selected VL scope, and the produced mask
    controls multiply-accumulate of explicit lhs/rhs payload vectors with an
    explicit accumulator input before storing to the output buffer. The op is
    nested directly in a selected tcrv.exec.variant and binds runtime ABI
    values for compare lhs, rhs_scalar, macc lhs/rhs payload inputs,
    accumulator input, output, and runtime n/AVL together with compare
    predicate, mask provenance, operation kind, memory form, accumulator
    role/layout, result layout, and compile-time RVV config facts.

    This op is not a compare-select wrapper, masked-store fallback, unmasked
    macc fallback, constant substitution, helper-string authority, route label,
    descriptor, source-front-door marker, direct-C exporter, intrinsic wrapper,
    runtime artifact, correctness evidence, or performance evidence. The RVV
    plugin must first realize it into explicit setvl/with_vl/load/splat/
    payload-loads/accumulator-load/compare/masked_macc/store tcrv_rvv body
    structure before the provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs,
    AnyType:$rhs_scalar,
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
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs `,` $rhs_scalar `,` $lhs `,` $rhs `,` $acc `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
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

def TypedWideningProductReducePreRealizedBodyOp
    : TCRVRVV_Op<"typed_widening_product_reduce_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV widening product reduction body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for the
    signed or unsigned low-precision contraction chain. The op is nested
    directly in a selected tcrv.exec.variant and binds explicit runtime ABI
    values for i8/u8 lhs/rhs inputs, an i32/u32 accumulator seed, an i32/u32
    scalar output, and runtime n/AVL. It also carries source signedness,
    source, product, accumulator, and result SEW+LMUL facts,
    product/reduction relations, memory form, accumulator layout, result
    layout, and policy.

    This op is not a q8/q4 route, matmul framework, high-level frontend,
    dtype-prefixed helper namespace, descriptor, source-front-door marker,
    direct-C exporter, route id, intrinsic wrapper, runtime artifact,
    correctness evidence, or performance evidence. The RVV plugin must
    realize it into explicit setvl/with_vl/load/load/widening_product/
    standalone_reduce/store tcrv_rvv body structure before the provider may
    construct a TCRVEmitCLowerableRoute.
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
    StrAttr:$source_signedness,
    I64Attr:$product_sew,
    StrAttr:$product_lmul,
    I64Attr:$accumulator_sew,
    StrAttr:$accumulator_lmul,
    I64Attr:$result_sew,
    StrAttr:$result_lmul,
    StrAttr:$product_relation,
    StrAttr:$product_reduction_chain_relation,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $acc `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedWideningProductReduceDequantizePreRealizedBodyOp
    : TCRVRVV_Op<"typed_widening_product_reduce_dequantize_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV widening product reduction dequantization body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for the
    signed low-precision contraction-to-f32 chain. The op is nested directly
    in a selected tcrv.exec.variant and binds explicit runtime ABI values for
    signed i8 lhs/rhs inputs, an i32 accumulator seed/carry boundary, a
    runtime f32 scale, f32 output, and runtime n/AVL. It also carries source,
    product, accumulator, and f32 result SEW+LMUL facts, product/reduction/
    dequantization relations, memory form, scale role, accumulator layouts,
    store boundary, and policy.

    This op is not a q8/q4 route, matmul framework, dtype-prefixed helper
    namespace, descriptor, source-front-door marker, direct-C exporter, route
    id, intrinsic wrapper, runtime artifact, correctness evidence, or
    performance evidence. The RVV plugin must realize it into explicit
    setvl/with_vl/load/load/widening_product/standalone_reduce/dequantize/
    store tcrv_rvv body structure before the provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$acc,
    AnyType:$scale,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$accumulator_role,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    StrAttr:$accumulator_carry_boundary,
    I64Attr:$source_sew,
    StrAttr:$source_lmul,
    I64Attr:$product_sew,
    StrAttr:$product_lmul,
    I64Attr:$accumulator_sew,
    StrAttr:$accumulator_lmul,
    I64Attr:$result_sew,
    StrAttr:$result_lmul,
    StrAttr:$product_relation,
    StrAttr:$product_reduction_chain_relation,
    StrAttr:$dequant_relation,
    StrAttr:$scale_role,
    StrAttr:$dequant_store_boundary,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $acc `,` $scale `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedWideningProductReduceDequantClampF32PreRealizedBodyOp
    : TCRVRVV_Op<"typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV widening product reduction dequantization plus f32 clamp body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body for the
    signed low-precision contraction-to-clamped-f32 chain. The op is nested
    directly in a selected tcrv.exec.variant and binds explicit runtime ABI
    values for signed i8 lhs/rhs inputs, an i32 accumulator seed/carry
    boundary, runtime f32 scale, runtime f32 lower and upper bounds, f32
    output, and runtime n/AVL. It also carries source, product, accumulator,
    and f32 result SEW+LMUL facts, product/reduction/dequantization
    relations, lower/upper clamp predicates, bound order, select layout,
    memory form, scale role, accumulator layouts, store boundary, and policy.

    This op is not a q8/q4 route, matmul framework, high-level frontend,
    dtype-prefixed helper namespace, descriptor, source-front-door marker,
    direct-C exporter, route id, intrinsic wrapper, runtime artifact,
    correctness evidence, or performance evidence. The RVV plugin must
    realize it into explicit setvl/with_vl/load/load/widening_product/
    standalone_reduce/dequantize/splat/splat/compare/select/compare/select/
    store tcrv_rvv body structure before the provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$acc,
    AnyType:$scale,
    AnyType:$lower_bound,
    AnyType:$upper_bound,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$accumulator_role,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    StrAttr:$accumulator_carry_boundary,
    I64Attr:$source_sew,
    StrAttr:$source_lmul,
    I64Attr:$product_sew,
    StrAttr:$product_lmul,
    I64Attr:$accumulator_sew,
    StrAttr:$accumulator_lmul,
    I64Attr:$result_sew,
    StrAttr:$result_lmul,
    StrAttr:$product_relation,
    StrAttr:$product_reduction_chain_relation,
    StrAttr:$dequant_relation,
    StrAttr:$scale_role,
    StrAttr:$lower_predicate_kind,
    StrAttr:$upper_predicate_kind,
    StrAttr:$bound_order,
    StrAttr:$select_layout,
    StrAttr:$dequant_store_boundary,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $acc `,` $scale `,` $lower_bound `,` $upper_bound `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedWideningProductReduceDequantClampF32BodyOp
    : TCRVRVV_Op<"typed_widening_product_reduce_dequant_clamp_f32_body"> {
  let summary =
      "generic typed explicit RVV widening product reduction dequantization plus f32 clamp body";
  let description = [{
    Carries one bounded explicit typed RVV selected body for the signed
    low-precision contraction-to-clamped-f32 chain. The op is nested directly
    in a selected tcrv.exec.variant and binds explicit runtime ABI values for
    signed i8 lhs/rhs inputs, an i32 accumulator seed/carry boundary, runtime
    f32 scale, runtime f32 lower and upper bounds, f32 output, and runtime
    n/AVL. It also carries source, product, accumulator, and f32 result
    SEW+LMUL facts, product/reduction/dequantization relations, lower/upper
    clamp predicates, bound order, select layout, memory form, scale role,
    accumulator layouts, store boundary, and policy.

    This op is not a q8/q4 route, matmul framework, high-level frontend,
    dtype-prefixed helper namespace, descriptor, source-front-door marker,
    direct-C exporter, route id, intrinsic wrapper, runtime artifact,
    correctness evidence, or performance evidence. The RVV plugin must realize
    it into explicit setvl/with_vl/load/load/widening_product/
    standalone_reduce/dequantize/splat/splat/compare/select/compare/select/
    store tcrv_rvv body structure before the provider may construct a
    TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$acc,
    AnyType:$scale,
    AnyType:$lower_bound,
    AnyType:$upper_bound,
    AnyType:$out,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    StrAttr:$accumulator_role,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout,
    StrAttr:$accumulator_carry_boundary,
    I64Attr:$source_sew,
    StrAttr:$source_lmul,
    I64Attr:$product_sew,
    StrAttr:$product_lmul,
    I64Attr:$accumulator_sew,
    StrAttr:$accumulator_lmul,
    I64Attr:$result_sew,
    StrAttr:$result_lmul,
    StrAttr:$product_relation,
    StrAttr:$product_reduction_chain_relation,
    StrAttr:$dequant_relation,
    StrAttr:$scale_role,
    StrAttr:$lower_predicate_kind,
    StrAttr:$upper_predicate_kind,
    StrAttr:$bound_order,
    StrAttr:$select_layout,
    StrAttr:$dequant_store_boundary,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $acc `,` $scale `,` $lower_bound `,` $upper_bound `,` $out `,` $n attr-dict `:` functional-type(operands, results)";
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
    destination byte stride together with selected memory movement operation
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
    evidence. The RVV plugin must realize masked_unit_load_store into explicit
    setvl/with_vl/mask_load/load/masked_load/store and masked_unit_store into
    explicit setvl/with_vl/load/mask_load/masked_store tcrv_rvv body structure
    before the
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
    plugin must realize it into explicit setvl/with_vl/load/load/load/compare/
    masked_load/store tcrv_rvv body structure before the provider may construct
    a TCRVEmitCLowerableRoute.
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
    source, destination, runtime n/AVL, and destination byte stride together
    with compare predicate, memory form, mask role/source/config,
    inactive-lane preservation policy, stride unit, compile-time RVV data/mask
    config, and policy.

    This op is not a fusion framework, route label, descriptor, source-front
    door marker, direct-C exporter, intrinsic wrapper, runtime artifact,
    correctness evidence, or performance evidence. The RVV plugin must realize
    it into explicit setvl/with_vl/load/load/load/strided_load/compare/
    masked_strided_store tcrv_rvv body structure before the provider may
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

def TypedComputedMaskStridedLoadPreRealizedBodyOp
    : TCRVRVV_Op<"typed_computed_mask_strided_load_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV computed-mask strided-load body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector compare produces an in-body mask SSA value and masked
    memory movement loads active lanes from a runtime byte-strided source into
    a unit-stride destination while preserving inactive lanes from the old
    destination vector. The op is nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for compare lhs,
    compare rhs, source, destination/old-vector passthrough, runtime n/AVL,
    and source byte stride together with compare predicate, memory form, mask
    role/source/config, inactive-lane preservation policy, stride unit,
    compile-time RVV data/mask config, and policy.

    This op is not a fusion framework, route label, descriptor,
    source-front-door marker, direct-C exporter, intrinsic wrapper, runtime
    artifact, correctness evidence, or performance evidence. The RVV plugin
    must realize it into explicit setvl/with_vl/load/load/load/compare/
    masked_strided_load/store tcrv_rvv body structure before the provider may
    construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs,
    AnyType:$compare_rhs,
    AnyType:$source,
    AnyType:$destination,
    AnyType:$n,
    Index:$source_stride,
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
    "$compare_lhs `,` $compare_rhs `,` $source `,` $destination `,` $n `,` $source_stride attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedComputedMaskIndexedGatherPreRealizedBodyOp
    : TCRVRVV_Op<"typed_computed_mask_indexed_gather_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV computed-mask indexed gather-load body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector compare produces an in-body mask SSA value and masked
    indexed/gather memory movement loads active lanes from a source memory
    window through an explicit runtime index vector into a unit-stride
    destination while preserving inactive lanes from the old destination
    vector. The op is nested directly in a selected tcrv.exec.variant and
    binds explicit runtime ABI values for compare lhs, compare rhs, source,
    index, destination/old-vector passthrough, and runtime n/AVL together with
    compare predicate, indexed memory form, index EEW/offset unit, mask
    role/source/config, inactive-lane preservation policy, compile-time RVV
    data/mask/index config, and policy.

    This op is not a gather/scatter framework, route label, descriptor,
    source-front-door marker, direct-C exporter, intrinsic wrapper, runtime
    artifact, correctness evidence, or performance evidence. The RVV plugin
    must realize it into explicit setvl/with_vl/load/load/load/index_load/
    compare/masked_indexed_load/store tcrv_rvv body structure before the
    provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs,
    AnyType:$compare_rhs,
    AnyType:$source,
    AnyType:$index,
    AnyType:$destination,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    I64Attr:$index_eew,
    StrAttr:$offset_unit,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$inactive_lane_policy,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs `,` $compare_rhs `,` $source `,` $index `,` $destination `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOp
    : TCRVRVV_Op<"typed_runtime_scalar_computed_mask_indexed_gather_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV runtime-scalar computed-mask indexed gather-load body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector lhs is compared with a runtime scalar threshold splatted
    at the current VL to produce an in-body mask SSA value, then masked
    indexed/gather memory movement loads active lanes from a source memory
    window through an explicit runtime index vector into a unit-stride
    destination while preserving inactive lanes from the old destination
    vector. The op is nested directly in a selected tcrv.exec.variant and
    binds explicit runtime ABI values for lhs, rhs_scalar, source, index,
    destination/old-vector passthrough, and runtime n/AVL together with
    compare predicate, indexed memory form, index EEW/offset unit, mask
    role/source/config, inactive-lane preservation policy, compile-time RVV
    data/mask/index config, and policy.

    This op is not a gather/scatter framework, route label, descriptor,
    source-front-door marker, direct-C exporter, intrinsic wrapper, runtime
    artifact, correctness evidence, or performance evidence. The RVV plugin
    must realize it into explicit setvl/with_vl/load/splat/load/index_load/
    compare/masked_indexed_load/store tcrv_rvv body structure before the
    provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs_scalar,
    AnyType:$source,
    AnyType:$index,
    AnyType:$destination,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    I64Attr:$index_eew,
    StrAttr:$offset_unit,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$inactive_lane_policy,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs_scalar `,` $source `,` $index `,` $destination `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedComputedMaskIndexedScatterPreRealizedBodyOp
    : TCRVRVV_Op<"typed_computed_mask_indexed_scatter_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV computed-mask indexed scatter-store body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector compare produces an in-body mask SSA value and masked
    indexed/scatter memory movement stores active source lanes into an
    indexed destination memory window through an explicit runtime index
    vector. The op is nested directly in a selected tcrv.exec.variant and
    binds explicit runtime ABI values for compare lhs, compare rhs, source
    payload, index, destination, and runtime n/AVL together with compare
    predicate, indexed store memory form, index EEW/offset unit, unique-index
    policy, mask role/source/config, inactive-lane no-write policy,
    compile-time RVV data/mask/index config, and policy.

    This op is not a gather/scatter framework, route label, duplicate-index
    resolver, descriptor, source-front-door marker, direct-C exporter,
    intrinsic wrapper, runtime artifact, correctness evidence, or performance
    evidence. The RVV plugin must realize it into explicit setvl/with_vl/load/
    load/load/index_load/compare/masked_indexed_store tcrv_rvv body structure
    before the provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs,
    AnyType:$compare_rhs,
    AnyType:$source,
    AnyType:$index,
    AnyType:$destination,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    I64Attr:$index_eew,
    StrAttr:$offset_unit,
    StrAttr:$index_uniqueness,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$inactive_lane_policy,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs `,` $compare_rhs `,` $source `,` $index `,` $destination `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOp
    : TCRVRVV_Op<"typed_runtime_scalar_computed_mask_indexed_scatter_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV runtime-scalar computed-mask indexed scatter-store body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector lhs is compared with a runtime scalar threshold splatted
    at the current VL to produce an in-body mask SSA value, then masked
    indexed/scatter memory movement stores active source lanes into an indexed
    destination memory window through an explicit runtime index vector. The op
    is nested directly in a selected tcrv.exec.variant and binds explicit
    runtime ABI values for lhs, rhs_scalar, source payload, index,
    destination, and runtime n/AVL together with compare predicate, indexed
    store memory form, index EEW/offset unit, unique-index policy, mask
    role/source/config, inactive-lane no-write policy, compile-time RVV
    data/mask/index config, and policy.

    This op is not a gather/scatter framework, route label, duplicate-index
    resolver, descriptor, source-front-door marker, direct-C exporter,
    intrinsic wrapper, runtime artifact, correctness evidence, or performance
    evidence. The RVV plugin must realize it into explicit setvl/with_vl/load/
    splat/load/index_load/compare/masked_indexed_store tcrv_rvv body structure
    before the provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs_scalar,
    AnyType:$source,
    AnyType:$index,
    AnyType:$destination,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    I64Attr:$index_eew,
    StrAttr:$offset_unit,
    StrAttr:$index_uniqueness,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$inactive_lane_policy,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs_scalar `,` $source `,` $index `,` $destination `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedComputedMaskSegment2LoadPreRealizedBodyOp
    : TCRVRVV_Op<"typed_computed_mask_segment2_load_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV computed-mask segment2 load body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector compare produces an in-body mask SSA value and masked
    segment2 memory movement loads two active fields from an interleaved source
    memory window into two unit-stride destination fields while preserving
    inactive lanes from the old destination vectors. The op is nested directly
    in a selected tcrv.exec.variant and binds explicit runtime ABI values for
    compare lhs, compare rhs, interleaved source, field0 destination/old-vector
    passthrough, field1 destination/old-vector passthrough, and runtime n/AVL
    together with compare predicate, segment count, source/destination memory
    forms, field roles, mask role/source/config, inactive-lane preservation
    policy, compile-time RVV data/mask config, and policy.

    This op is not a segmented-memory framework, route label, descriptor,
    source-front-door marker, direct-C exporter, intrinsic wrapper, runtime
    artifact, correctness evidence, or performance evidence. The RVV plugin
    must realize it into explicit setvl/with_vl/load/load/load/load/compare/
    masked_segment2_load/store/store tcrv_rvv body structure before the
    provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs,
    AnyType:$compare_rhs,
    AnyType:$source,
    AnyType:$out0,
    AnyType:$out1,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    I64Attr:$segment_count,
    StrAttr:$field0_role,
    StrAttr:$field1_role,
    StrAttr:$source_memory_form,
    StrAttr:$destination_memory_form,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$inactive_lane_policy,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs `,` $compare_rhs `,` $source `,` $out0 `,` $out1 `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOp
    : TCRVRVV_Op<"typed_runtime_scalar_computed_mask_segment2_load_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV runtime-scalar computed-mask segment2 load body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector lhs is compared with a runtime scalar threshold splatted
    at the current VL to produce an in-body mask SSA value, then masked
    segment2 memory movement loads two active fields from an interleaved source
    memory window into two unit-stride destination fields while preserving
    inactive lanes from the old destination vectors. The op is nested directly
    in a selected tcrv.exec.variant and binds explicit runtime ABI values for
    lhs, rhs_scalar, interleaved source, field0 destination/old-vector
    passthrough, field1 destination/old-vector passthrough, and runtime n/AVL
    together with compare predicate, segment count, source/destination memory
    forms, field roles, mask role/source/config, inactive-lane preservation
    policy, compile-time RVV data/mask config, and policy.

    This op is not a segmented-memory framework, route label, descriptor,
    source-front-door marker, direct-C exporter, intrinsic wrapper, runtime
    artifact, correctness evidence, or performance evidence. The RVV plugin
    must realize it into explicit setvl/with_vl/load/splat/load/load/compare/
    masked_segment2_load/store/store tcrv_rvv body structure before the
    provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs_scalar,
    AnyType:$source,
    AnyType:$out0,
    AnyType:$out1,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    I64Attr:$segment_count,
    StrAttr:$field0_role,
    StrAttr:$field1_role,
    StrAttr:$source_memory_form,
    StrAttr:$destination_memory_form,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$inactive_lane_policy,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs_scalar `,` $source `,` $out0 `,` $out1 `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedComputedMaskSegment2StorePreRealizedBodyOp
    : TCRVRVV_Op<"typed_computed_mask_segment2_store_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV computed-mask segment2 store body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector compare produces an in-body mask SSA value and masked
    segment2 memory movement stores two active field vectors into one
    interleaved destination memory window while preserving inactive lanes by
    not writing compare-false lanes. The bounded update spelling may also
    carry arithmetic_kind = "add", in which case the RVV plugin realizes a
    typed vector add over the two field source vectors before storing the
    updated field0 payload. The op is nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for compare lhs,
    compare rhs, field0 source, field1 source, interleaved destination, and
    runtime n/AVL together with compare predicate, optional arithmetic kind,
    segment count, source/destination memory forms, field roles, mask
    role/source/config, inactive-lane preservation policy, compile-time RVV
    data/mask config, and policy.

    This op is not a segmented-memory framework, route label, descriptor,
    source-front-door marker, direct-C exporter, intrinsic wrapper, runtime
    artifact, correctness evidence, or performance evidence. The RVV plugin
    must realize it into explicit setvl/with_vl/load/load/load/load/compare/
    optional binary/masked_segment2_store tcrv_rvv body structure before the
    provider may construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$compare_lhs,
    AnyType:$compare_rhs,
    AnyType:$src0,
    AnyType:$src1,
    AnyType:$dst,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    I64Attr:$segment_count,
    StrAttr:$field0_role,
    StrAttr:$field1_role,
    StrAttr:$source0_memory_form,
    StrAttr:$source1_memory_form,
    StrAttr:$destination_memory_form,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$inactive_lane_policy,
    OptionalAttr<StrAttr>:$arithmetic_kind,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$compare_lhs `,` $compare_rhs `,` $src0 `,` $src1 `,` $dst `,` $n attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def TypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOp
    : TCRVRVV_Op<"typed_runtime_scalar_computed_mask_segment2_store_pre_realized_body"> {
  let summary =
      "generic typed pre-realized RVV runtime-scalar computed-mask segment2 store body";
  let description = [{
    Carries one bounded RVV plugin-local pre-realized selected body where a
    signed i32 vector lhs is compared with a runtime scalar threshold splatted
    at the current VL to produce an in-body mask SSA value, then masked
    segment2 memory movement stores two active field vectors into one
    interleaved destination memory window while preserving inactive lanes by
    not writing compare-false lanes. The op is nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for lhs,
    rhs_scalar, field0 source, field1 source, interleaved destination, and
    runtime n/AVL together with compare predicate, segment count,
    source/destination memory forms, field roles, mask role/source/config,
    inactive-lane preservation policy, compile-time RVV data/mask config, and
    policy.

    This op is not a segmented-memory framework, route label, descriptor,
    source-front-door marker, direct-C exporter, intrinsic wrapper, runtime
    artifact, correctness evidence, or performance evidence. The RVV plugin
    must realize it into explicit setvl/with_vl/load/splat/load/load/compare/
    masked_segment2_store tcrv_rvv body structure before the provider may
    construct a TCRVEmitCLowerableRoute.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs_scalar,
    AnyType:$src0,
    AnyType:$src1,
    AnyType:$dst,
    AnyType:$n,
    StrAttr:$op_kind,
    StrAttr:$predicate_kind,
    StrAttr:$memory_form,
    I64Attr:$segment_count,
    StrAttr:$field0_role,
    StrAttr:$field1_role,
    StrAttr:$source0_memory_form,
    StrAttr:$source1_memory_form,
    StrAttr:$destination_memory_form,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$inactive_lane_policy,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs_scalar `,` $src0 `,` $src1 `,` $dst `,` $n attr-dict `:` functional-type(operands, results)";
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

def MaskedLoadOp : TCRVRVV_Op<"masked_load",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV masked unit-stride data load op";
  let description = [{
    Records one bounded plugin-local RVV masked unit-stride data load step.
    The op is nested directly under tcrv_rvv.with_vl, consumes one explicit
    source runtime ABI buffer, one !tcrv_rvv.mask predicate produced by
    tcrv_rvv.mask_load or tcrv_rvv.compare in the same VL scope, one
    inactive-lane passthrough vector, and the active !tcrv_rvv.vl token. It
    carries explicit masked-load memory form and inactive-lane preservation
    facts, and produces the vector value to store.

    For the current bounded Stage 2 masked memory movement slice, memory_form
    is limited to masked-unit-load and inactive_lane_policy is limited to
    preserve-passthrough-on-false-lanes. The RVV plugin derives the concrete
    masked-load intrinsic from this typed body fact, dtype/config, mask, and
    runtime ABI binding; common EmitC/export must not infer the masked load
    from route ids, ABI names, artifact metadata, exact intrinsic spelling, or
    harness sentinels.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$mask,
    AnyType:$passthrough,
    AnyType:$vl,
    StrAttr:$memory_form,
    StrAttr:$inactive_lane_policy
  );
  let results = (outs AnyType:$loaded);

  let assemblyFormat =
    "$buffer `,` $mask `,` $passthrough `,` $vl attr-dict `:` type($buffer) `,` type($mask) `,` type($passthrough) `,` type($vl) `->` type($loaded)";
  let hasVerifier = 1;
}

def MaskedStridedLoadOp : TCRVRVV_Op<"masked_strided_load",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV masked byte-strided data load op";
  let description = [{
    Records one bounded plugin-local RVV masked source byte-strided data load
    step. The op is nested directly under tcrv_rvv.with_vl, consumes one
    explicit source runtime ABI buffer, one !tcrv_rvv.mask predicate produced
    by tcrv_rvv.compare in the same VL scope, one inactive-lane passthrough
    vector, one explicit runtime source byte stride SSA value, and the active
    !tcrv_rvv.vl token. It carries explicit masked strided-load memory form,
    byte-stride unit, and inactive-lane passthrough preservation facts, and
    produces the vector value to store.

    For the current bounded Stage 2 computed-mask strided masked-load slice,
    memory_form is limited to masked-strided-load, stride_unit is limited to
    byte, and inactive_lane_policy is limited to
    preserve-passthrough-on-false-lanes. The RVV plugin derives the concrete
    masked strided-load intrinsic from this typed body fact, dtype/config,
    mask, source stride, passthrough, and runtime ABI binding; common
    EmitC/export must not infer the masked strided load from route ids, ABI
    names, artifact metadata, exact intrinsic spelling, or harness sentinels.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$mask,
    AnyType:$passthrough,
    Index:$stride,
    AnyType:$vl,
    StrAttr:$memory_form,
    StrAttr:$stride_unit,
    StrAttr:$inactive_lane_policy
  );
  let results = (outs AnyType:$loaded);

  let assemblyFormat =
    "$buffer `,` $mask `,` $passthrough `,` $stride `,` $vl attr-dict `:` type($buffer) `,` type($mask) `,` type($passthrough) `,` type($stride) `,` type($vl) `->` type($loaded)";
  let hasVerifier = 1;
}

def MaskedIndexedLoadOp : TCRVRVV_Op<"masked_indexed_load",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV masked indexed data load op";
  let description = [{
    Records one bounded plugin-local RVV masked indexed/gather data load step.
    The op is nested directly under tcrv_rvv.with_vl, consumes one explicit
    source runtime ABI buffer, one !tcrv_rvv.index_vector produced by
    tcrv_rvv.index_load, one !tcrv_rvv.mask predicate produced by
    tcrv_rvv.compare in the same VL scope, one inactive-lane passthrough
    vector, and the active !tcrv_rvv.vl token. It carries explicit indexed
    masked-load memory form, index EEW, offset unit, and inactive-lane
    preservation facts, and produces the vector value to store.

    For the current bounded Stage 2 computed-mask indexed gather-load slice,
    memory_form is limited to masked-indexed-load, offset_unit is limited to
    element, index_eew is limited to 32, and inactive_lane_policy is limited
    to preserve-passthrough-on-false-lanes. The RVV plugin derives the
    concrete masked indexed-load intrinsic and any element-to-byte offset
    conversion from this typed body fact, dtype/config, mask, index vector,
    passthrough, and runtime ABI binding; common EmitC/export must not infer
    indexed or masked semantics from route ids, ABI names, artifact metadata,
    exact intrinsic spelling, or harness sentinels.
  }];

  let arguments = (ins
    AnyType:$data,
    AnyType:$indices,
    AnyType:$mask,
    AnyType:$passthrough,
    AnyType:$vl,
    I64Attr:$index_eew,
    StrAttr:$offset_unit,
    StrAttr:$memory_form,
    StrAttr:$inactive_lane_policy
  );
  let results = (outs AnyType:$loaded);

  let assemblyFormat =
    "$data `,` $indices `,` $mask `,` $passthrough `,` $vl attr-dict `:` type($data) `,` type($indices) `,` type($mask) `,` type($passthrough) `,` type($vl) `->` type($loaded)";
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

def MaskedIndexedStoreOp : TCRVRVV_Op<"masked_indexed_store",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV masked indexed data store op";
  let description = [{
    Records one bounded plugin-local RVV masked indexed/scatter destination
    store step. The op is nested directly under tcrv_rvv.with_vl, consumes one
    explicit output buffer runtime ABI value, one !tcrv_rvv.index_vector
    produced by tcrv_rvv.index_load, one !tcrv_rvv.mask predicate produced by
    tcrv_rvv.compare in the same VL scope, one typed data !tcrv_rvv.vector
    payload value, and the active !tcrv_rvv.vl token. It carries explicit
    indexed masked-store memory form, index EEW, offset unit, unique-index
    policy, and inactive-lane no-write preservation facts.

    For the current bounded Stage 2 computed-mask indexed scatter-store slice,
    memory_form is limited to masked-indexed-store, offset_unit is limited to
    element, index_eew is limited to 32, index_uniqueness is limited to
    unique, and inactive_lane_policy is limited to
    preserve-output-on-false-lanes. The RVV plugin derives the concrete masked
    indexed-store intrinsic and any element-to-byte offset conversion from
    this typed body fact, dtype/config, mask, index vector, payload, and
    runtime ABI binding; common EmitC/export must not infer indexed or masked
    semantics from route ids, ABI names, artifact metadata, exact intrinsic
    spelling, or harness sentinels.
  }];

  let arguments = (ins
    AnyType:$destination,
    AnyType:$indices,
    AnyType:$mask,
    AnyType:$value,
    AnyType:$vl,
    I64Attr:$index_eew,
    StrAttr:$offset_unit,
    StrAttr:$index_uniqueness,
    StrAttr:$memory_form,
    StrAttr:$inactive_lane_policy
  );

  let assemblyFormat =
    "$destination `,` $indices `,` $mask `,` $value `,` $vl attr-dict `:` type($destination) `,` type($indices) `,` type($mask) `,` type($value) `,` type($vl)";
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

def MaskedSegment2LoadOp : TCRVRVV_Op<"masked_segment2_load",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV masked segment2 interleaved load op";
  let description = [{
    Records one bounded plugin-local RVV masked segment2 source load dataflow
    step. The op is nested directly under tcrv_rvv.with_vl, consumes one
    explicit interleaved source-buffer runtime ABI value, one !tcrv_rvv.mask
    predicate produced by tcrv_rvv.compare in the same VL scope, two typed
    old-vector passthrough values for inactive field lanes, and the active
    !tcrv_rvv.vl token. It carries segment_count = 2, source memory form,
    field0/field1 result role facts, and inactive-lane preservation policy,
    and produces two same-typed !tcrv_rvv.vector field values.

    Segment count, field order, source memory form, passthrough values, mask
    provenance, dtype/config, and runtime ABI provenance come from this typed
    body fact. The RVV plugin derives the tuple C type, masked segmented-load
    leaf, field extraction leaves, and destination stores from these facts;
    route ids, artifact names, ABI strings, exact intrinsic spellings,
    descriptors, and common EmitC/export code must not infer segmented or
    masked memory semantics.
  }];

  let arguments = (ins
    AnyType:$source,
    AnyType:$mask,
    AnyType:$passthrough0,
    AnyType:$passthrough1,
    AnyType:$vl,
    I64Attr:$segment_count,
    StrAttr:$source_memory_form,
    StrAttr:$field0_role,
    StrAttr:$field1_role,
    StrAttr:$inactive_lane_policy
  );
  let results = (outs AnyType:$field0, AnyType:$field1);

  let assemblyFormat =
    "$source `,` $mask `,` $passthrough0 `,` $passthrough1 `,` $vl attr-dict `:` type($source) `,` type($mask) `,` type($passthrough0) `,` type($passthrough1) `,` type($vl) `->` type($field0) `,` type($field1)";
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

def MaskedSegment2StoreOp : TCRVRVV_Op<"masked_segment2_store",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV masked segment2 interleaved store op";
  let description = [{
    Records one bounded plugin-local RVV masked segment2 destination store
    dataflow step. The op is nested directly under tcrv_rvv.with_vl, consumes
    one explicit interleaved destination-buffer runtime ABI value, one
    !tcrv_rvv.mask predicate produced by tcrv_rvv.compare in the same VL
    scope, field0 and field1 !tcrv_rvv.vector payload values, and the active
    !tcrv_rvv.vl token. It carries segment_count = 2, destination memory
    form, field0/field1 source role facts, and inactive-lane no-write policy.

    Segment count, field order, destination memory form, payload values, mask
    provenance, dtype/config, and runtime ABI provenance come from this typed
    body fact. The RVV plugin derives the tuple C type and masked segmented
    store leaf from these facts; route ids, artifact names, ABI strings, exact
    intrinsic spellings, descriptors, and common EmitC/export code must not
    infer segmented or masked store semantics.
  }];

  let arguments = (ins
    AnyType:$destination,
    AnyType:$mask,
    AnyType:$field0,
    AnyType:$field1,
    AnyType:$vl,
    I64Attr:$segment_count,
    StrAttr:$destination_memory_form,
    StrAttr:$field0_role,
    StrAttr:$field1_role,
    StrAttr:$inactive_lane_policy
  );

  let assemblyFormat =
    "$destination `,` $mask `,` $field0 `,` $field1 `,` $vl attr-dict `:` type($destination) `,` type($mask) `,` type($field0) `,` type($field1) `,` type($vl)";
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
    add, sub, or mul and the mask must come from generic tcrv_rvv.compare in
    the same selected VL scope. The passthrough vector is structural body
    authority for inactive-lane behavior; common EmitC/export must not infer it
    from route ids, ABI names, or artifact metadata.
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

def MaskAndOp : TCRVRVV_Op<"mask_and",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV mask-and composition dataflow op";
  let description = [{
    Records one bounded plugin-local RVV mask composition step. The op is
    nested directly under tcrv_rvv.with_vl, consumes two typed
    !tcrv_rvv.mask predicates produced by tcrv_rvv.compare in the same
    selected VL scope, consumes the active !tcrv_rvv.vl token, carries
    explicit composition kind, and produces one typed !tcrv_rvv.mask token.

    The current Stage 2 slice supports only kind = "and". Mask algebra is
    structural typed-body authority in the RVV plugin; route ids, helper names,
    ABI strings, artifact names, status fields, exact intrinsic spellings, and
    common EmitC/export code must not infer or choose mask composition
    semantics.
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

    For the current bounded Stage 2 slice, kind is limited to signed
    add/min/max plus signed_widening_reduce_add and
    unsigned_widening_reduce_add. The non-widening forms
    represent acc_seed + sum(input), min(acc_seed, input), or
    max(acc_seed, input) over runtime n. The widening form represents a typed
    i16 source vector reduced into an i32 scalar accumulator/result boundary.
    Input dtype/config, accumulator seed boundary, scalar output layout, VL,
    policy, and runtime n facts are structural in the selected RVV body. Route
    ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, and common EmitC/export code must not infer or choose
    standalone reduction semantics.
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

def MaskedStandaloneReduceOp : TCRVRVV_Op<"masked_standalone_reduce",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV computed-mask standalone scalar reduction dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV computed-mask standalone
    horizontal reduction dataflow step. The op is nested directly under
    tcrv_rvv.with_vl, consumes a compare-produced !tcrv_rvv.mask, one typed
    !tcrv_rvv.vector source input, one explicit scalar accumulator seed
    runtime ABI value, and the active !tcrv_rvv.vl token. It carries explicit
    reduction kind plus mask provenance, accumulator/result layout facts, and
    produces one typed !tcrv_rvv.vector token whose lane 0 is the scalar output
    boundary.

    For the current bounded Stage 2 slice, kind is limited to signed add/min/max.
    Add uses the neutral inactive element 0, min uses INT32_MAX, and max uses
    INT32_MIN before reducing active source lanes with the scalar accumulator
    seed. Input dtype/config, mask provenance, accumulator seed boundary,
    scalar output layout, VL, policy, and runtime n facts are structural in the
    selected RVV body. Route ids, helper names, ABI strings, artifact names,
    status fields, exact intrinsic spellings, and common EmitC/export code must
    not infer or choose masked standalone reduction semantics.
  }];

  let arguments = (ins
    AnyType:$mask,
    AnyType:$input,
    AnyType:$accumulator_seed,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$mask `,` $input `,` $accumulator_seed `,` $vl attr-dict `:` type($mask) `,` type($input) `,` type($accumulator_seed) `,` type($vl) `->` type($result)";
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

def MaskedMAccOp : TCRVRVV_Op<"masked_macc",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV computed-mask multiply-accumulate dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV computed-mask
    multiply-accumulate dataflow step. The op is nested directly under
    tcrv_rvv.with_vl, consumes a compare-produced !tcrv_rvv.mask, lhs/rhs typed
    !tcrv_rvv.vector operands, an explicit typed accumulator vector, and the
    active !tcrv_rvv.vl token. It carries explicit mask provenance,
    accumulator/result layout facts, and produces one typed !tcrv_rvv.vector
    result.

    For the current bounded Stage 2 contraction-supporting slice, kind is
    limited to add, representing masked accumulator + lhs * rhs. Inactive lanes
    preserve the accumulator vector. Lhs, rhs, accumulator, result, mask, VL,
    SEW, LMUL, policy, accumulator layout, result layout, and memory-form facts
    are structural in the selected RVV body. Route ids, helper names, ABI
    strings, artifact names, status fields, exact intrinsic spellings, and
    common EmitC/export code must not infer or choose masked
    multiply-accumulate semantics.
  }];

  let arguments = (ins
    AnyType:$mask,
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$accumulator,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$mask_role,
    StrAttr:$mask_source,
    StrAttr:$mask_memory_form,
    StrAttr:$accumulator_layout,
    StrAttr:$result_layout
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$mask `,` $lhs `,` $rhs `,` $accumulator `,` $vl attr-dict `:` type($mask) `,` type($lhs) `,` type($rhs) `,` type($accumulator) `,` type($vl) `->` type($result)";
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

def WideningProductOp : TCRVRVV_Op<"widening_product",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV widening product dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV widening product
    dataflow step. The op is nested directly under tcrv_rvv.with_vl, consumes
    two typed low-precision !tcrv_rvv.vector source operands and the active
    !tcrv_rvv.vl token, carries explicit operation kind and widening product
    relation facts, and produces one widened !tcrv_rvv.vector result.

    For the current bounded Stage 2 low-precision foundation, kind is limited
    to signed_widening_product or unsigned_widening_product. The signed i8
    route is provider-supported when all typed source/product facts match the
    provider-owned route contract. The unsigned u8 surface is verifier-legal
    so the RVV provider can fail closed at the missing unsigned intrinsic and
    target mirror boundary until production support is implemented. Lhs, rhs,
    result, VL, source/result SEW/LMUL, policy, memory-form facts, and product
    relation are structural in the selected RVV body. Route ids, helper names,
    ABI strings, artifact names, status fields, exact intrinsic spellings,
    q8/q4 names, and common EmitC/export code must not infer or choose
    widening product semantics.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$product_relation
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $vl attr-dict `:` type($lhs) `,` type($rhs) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def WideningAccumulateOp : TCRVRVV_Op<"widening_accumulate",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV deferred wide-LMUL widening accumulate dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV DEFERRED widening accumulate
    dataflow step: the per-iteration `vwadd.wv` that widens an i16 LMUL m4
    product into a loop-carried i32 LMUL m8 vector accumulator WITHOUT reducing
    to a scalar. This is the structurally-distinct algorithm the N3
    resource-aware max-legal-LMUL Gearbox schedule selects (the measured
    `ssh rvv` winner var_v_m2_a1.c): an i8m2 strip-load -> i16m4 widening
    product -> i32m8 deferred vector accumulate (vwadd.wv) per iteration, then
    ONE trailing vredsum + scalar epilogue after the loop -- replacing the
    narrow per-iteration vwredsum-into-scalar-i32m1 body.

    The op is nested directly under tcrv_rvv.with_vl, consumes the i16 LMUL m4
    widening product and the active !tcrv_rvv.vl token, and produces ONE wider
    i32 LMUL m8 !tcrv_rvv.vector result. The loop-carried i32m8 accumulator is
    zero-seeded at its own VLMAX before the loop and threaded across iterations
    by the conversion (a function-scoped emitc.variable, mirroring the narrow
    path's scalar accumulator carry); the acc[0] runtime seed is added as a
    SCALAR after the single trailing reduction. This makes the deferred-wide
    accumulate the FIRST-CLASS structural marker (I5): emission is determined
    by op identity (in-loop vwadd.wv) and the explicit i16m4 -> i32m8 type
    relation, never inferred from candidate-mirror metadata.

    For the bounded N3 deferred-wide surface kind is limited to
    "signed_widening_accumulate_add" and accumulate_relation to
    "signed-i16m4-into-i32m8-deferred-add". Source/result dtype/config, VL, and
    the deferred-accumulate structure are structural in the selected RVV body.
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, and common EmitC/export code must not infer or choose
    deferred widening accumulate semantics.
  }];

  let arguments = (ins
    AnyType:$product,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$accumulate_relation
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$product `,` $vl attr-dict `:` type($product) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def DeferredAccumulateOp : TCRVRVV_Op<"deferred_accumulate",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV deferred wide-LMUL NON-widening accumulate dataflow op";
  let description = [{
    Records one bounded generic plugin-local RVV DEFERRED non-widening accumulate
    dataflow step for the 2nd kernel family (signed i16 widening dot-reduce): the
    per-iteration `vadd.vv` that adds an i32 LMUL m8 widened product into a
    loop-carried i32 LMUL m8 vector accumulator WITHOUT reducing to a scalar.

    Unlike tcrv_rvv.widening_accumulate (the byte kernel's i16m4 -> i32m8
    WIDENING vwadd.wv), here the product is ALREADY i32m8 -- the i16 dot-reduce
    widens once (i16m4 x i16m4 -> i32m8 in the product) so the deferred
    accumulate is a SAME-WIDTH vadd.vv, not a widening accumulate. This op is the
    structurally-distinct algorithm the N3 resource-aware max-legal-LMUL Gearbox
    schedule selects for the dot-reduce family (the measured ssh-rvv winner
    dot_wide_deferred): an i16m4 strip-load -> i32m8 widening product -> i32m8
    DEFERRED vector accumulate (vadd.vv) per iteration, then ONE trailing vredsum
    + scalar epilogue after the loop -- replacing the narrow per-iteration
    vredsum-into-scalar dot-reduce body.

    The op is nested directly under tcrv_rvv.with_vl, consumes the i32 LMUL m8
    widening product and the active !tcrv_rvv.vl token, and produces ONE i32
    LMUL m8 !tcrv_rvv.vector result. The loop-carried i32m8 accumulator is
    zero-seeded at its own VLMAX before the loop and threaded across iterations
    by the conversion; the acc[0] runtime seed is added as a SCALAR after the
    single trailing reduction. This makes the deferred accumulate the FIRST-CLASS
    structural marker (I5): emission is determined by op identity (in-loop
    vadd.vv) and the explicit i32m8 -> i32m8 type relation, never inferred from
    candidate-mirror metadata.

    For the bounded N3 deferred-wide dot-reduce surface kind is limited to
    "signed_deferred_accumulate_add" and accumulate_relation to
    "signed-i32m8-into-i32m8-deferred-add". Source/result dtype/config, VL, and
    the deferred-accumulate structure are structural in the selected RVV body.
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, and common EmitC/export code must not infer or choose
    deferred non-widening accumulate semantics.
  }];

  let arguments = (ins
    AnyType:$product,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$accumulate_relation
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$product `,` $vl attr-dict `:` type($product) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def PackedI4NibbleUnpackProductOp
    : TCRVRVV_Op<"packed_i4_nibble_unpack_product",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV signed packed-i4 nibble-unpack widening product op";
  let description = [{
    Records one bounded generic plugin-local RVV signed packed-i4 nibble-unpack
    widening product dataflow step. The op is nested directly under
    tcrv_rvv.with_vl, consumes two typed i8 LMUL mf4 !tcrv_rvv.vector source
    operands -- each i8 element packing two signed 4-bit nibbles -- and the
    active !tcrv_rvv.vl token. It carries an explicit operation kind and product
    relation, and produces ONE widened i16 LMUL mf2 !tcrv_rvv.vector result
    holding the sum of the low-nibble and high-nibble signed widening products.

    This op makes the i4-nibble sign-extend / unpack STRUCTURE first-class in the
    typed body: the low nibble of each i8 is sign-extended (shift-left then
    arithmetic-shift-right), widening-multiplied, and rescaled; the high nibble
    is arithmetic-shift-right sign-extended and accumulated via a high-nibble
    widening multiply-accumulate. The fixed intrinsic chain
    (vsll/vsra over i8mf4, vwmul/vsra over i16mf2, vwmacc over i16mf2) is the
    op's lowering, NOT a candidate-mirror string read in the conversion. For the
    bounded Stage 3 surface kind is limited to
    signed_packed_i4_nibble_unpack_product and product_relation to
    "signed-i8mf4xi8mf4-to-i16mf2". Source/result dtype/config, VL, and the
    nibble-unpack structure are structural in the selected RVV body. Route ids,
    helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, operand-packing mirror strings, and common EmitC/export code must
    not infer or choose packed-i4 nibble-unpack product semantics.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$product_relation
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $vl attr-dict `:` type($lhs) `,` type($rhs) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def PackedI4OffsetBinaryXI8ProductOp
    : TCRVRVV_Op<"packed_i4_offset_binary_x_i8_product",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV asymmetric offset-binary packed-i4 x plain-i8 product op";
  let description = [{
    Records one bounded generic plugin-local RVV ASYMMETRIC offset-binary
    packed-i4 x plain-i8 widening product dataflow step. This is the integer
    core of the ggml Q4_0 x Q8_0 dot-product: ONLY the packed-i4 weight operand
    is nibble-decoded; the two int8 activation operands stay plain. The op is
    nested directly under tcrv_rvv.with_vl, consumes ONE typed i8 LMUL mf4
    !tcrv_rvv.vector packed-i4 weight operand (each i8 packing two offset-binary
    4-bit nibbles), TWO typed i8 LMUL mf4 !tcrv_rvv.vector PLAIN-int8 activation
    operands (the q8 low half paired with the low nibbles and the q8 high half
    paired with the high nibbles), and the active !tcrv_rvv.vl token. It carries
    an explicit operation kind and product relation, and produces ONE widened
    i16 LMUL mf2 !tcrv_rvv.vector result holding the sum of the low-nibble and
    high-nibble asymmetric widening products.

    This op makes the offset-binary nibble decode + ONE-SIDED unpack +
    asymmetric i8(decoded) x i8(plain) product + low/high <-> q8 half split
    pairing first-class STRUCTURE in the typed body, distinct from the SYMMETRIC
    two's-complement tcrv_rvv.packed_i4_nibble_unpack_product (which unpacks
    BOTH operands). The offset-binary decode is ggml's `(nibble - 8)` mapping
    (0 -> -8, 8 -> 0, 15 -> +7), realized as an xor-0x88 bias that turns each
    offset-binary nibble into a two's-complement 4-bit lane, then the existing
    low/high sign-extend. The packed weight low nibble pairs with the FIRST
    plain-i8 activation operand and the high nibble with the SECOND; the int8
    activations are NOT nibble-unpacked. The fixed intrinsic chain (vxor.vx 0x88
    over i8mf4; vsll/vsra sign-extend over i8mf4; vwmul over i16mf2 against the
    plain low activation; vwmacc over i16mf2 against the plain high activation)
    is the op's lowering, NOT a candidate-mirror string read in the conversion.
    For the bounded Stage 4 surface kind is limited to
    signed_packed_i4_offset_binary_x_i8_product and product_relation to
    "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2". Source/result dtype/config, VL,
    the offset-binary decode and the one-sided unpack structure are structural
    in the selected RVV body. Route ids, helper names, ABI strings, artifact
    names, status fields, exact intrinsic spellings, operand-packing mirror
    strings, q8/q4 names, and common EmitC/export code must not infer or choose
    asymmetric offset-binary packed-i4 x i8 product semantics.
  }];

  let arguments = (ins
    AnyType:$weight,
    AnyType:$activation_low,
    AnyType:$activation_high,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$product_relation
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight `,` $activation_low `,` $activation_high `,` $vl attr-dict `:` "
    "type($weight) `,` type($activation_low) `,` type($activation_high) `,` "
    "type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ40Q80Op
    : TCRVRVV_Op<"q4_0_q8_0_block_dot",
    [TCRVEmitCLowerableOpInterface,
     DeclareOpInterfaceMethods<TunableScheduleOpInterface>]> {
  let summary =
      "generic typed RVV block-quantized ggml Q4_0 x Q8_0 dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_q4_0_q8_0` block dot-product as ONE
    bounded generic plugin-local typed RVV dataflow step: the AoS QK=32 block
    loop, the per-block dual fp16 scale, the fp32 scalar accumulation, and the
    `*s` scalar store. It builds on the INC-1 integer core
    (tcrv_rvv.packed_i4_offset_binary_x_i8_product) by composing that core
    STRUCTURE inside a block loop. The op is nested directly under
    tcrv_rvv.with_vl and consumes FOUR runtime ABI value operands -- the q4_0
    weight base pointer (vx), the q8_0 activation base pointer (vy), the fp32
    output pointer (s), the runtime element count (n) -- plus the active
    !tcrv_rvv.vl token. It produces ONE i32 LMUL m1 !tcrv_rvv.vector token whose
    lowering writes the single fp32 reduction result through the output pointer.

    The op makes the block-quantized dot-product STRUCTURE first-class in the
    typed body. The structural facts -- QK=32, the AoS block strides (18 bytes
    for block_q4_0, 34 bytes for block_q8_0), the quant payload byte offset (+2),
    the q8 high-half byte offset (+16), and the per-block dual fp16 scale model
    (d_x . d_y) -- are carried as typed integer/string attributes (I4 mirror
    only). The fixed emission -- an outer emitc.for block loop over nb = n / QK,
    per-block address arithmetic (vx + ib*18, vy + ib*34) via emitc.add/emitc.mul
    nodes, the two scalar fp16->fp32 reads via emitc.call_opaque (the one
    sanctioned opaque piece, exactly as INC-1 emits its intrinsics), an inner
    strip loop that reuses INC-1's offset-binary asymmetric i4xi8 decode/product
    chain (vxor.vx 0x88 -> vsll/vsra -> vwmul/vwmacc against the plain q8 halves)
    + vwredsum into a per-block i32 scalar, and the left-associative fp32
    accumulate `sumf = sumf + ((float)sumi * d_x) * d_y` (ggml's exact order) --
    is the op's lowering, NOT a candidate-mirror string read in the conversion.
    For the bounded surface kind is limited to "ggml_q4_0_q8_0_block_dot" and
    scale_model to "dual-fp16-per-block-d_x.d_y". Route ids, helper names, ABI
    strings, artifact names, status fields, exact intrinsic spellings, operand
    mirror strings, q8/q4 names, and common EmitC/export code must not infer or
    choose block-quantized dot-product semantics.

    The optional `integer_core_lmul` attribute is a bounded resource/scheduling
    fact -- it selects the LMUL the per-block integer dot-product core anchors at
    ("mf4" or "m1"), which is purely the *how* (vector grouping / strip width),
    NOT the *what*. The dot-product is byte-exact either way: vwredsum sums the
    same integer set and integer addition is order-independent, so the i32 `sumi`
    and the fp32 accumulate are unchanged. When the attribute is absent the core
    anchors at "mf4" (the INC-2a default: i8mf4 -> i16mf2 -> i32m1, a 4-chunk
    inner strip at VLEN=128). When it is "m1" the core anchors at "m1" (i8m1 ->
    i16m2 -> i32m1, the whole half-block in ONE strip at VLEN=128 with a single
    vwredsum per block, matching ggml's hand-written kernel's reduction anchor).
    Both forms keep a VLEN-robust strip loop carrying sumi across strips, so a
    VLEN < 128 board re-strips correctly under either anchor.

    The optional `multi_block_factor` and `strip_elision` attributes are two more
    bounded resource/scheduling shape knobs (the *how*, never the *what*; the
    dot-product is byte-exact for any legal combination because integer addition
    is order-independent and the per-block fp32 fold stays in strict ascending
    block order):
    - `multi_block_factor` selects how many q4_0/q8_0 blocks the outer loop
      processes per iteration -- 1 (default), 2, or 4. A factor > 1 unrolls the
      outer block loop so several independent per-block integer cores can overlap;
      the loop processes nb / factor full groups then a scalar `nb % factor` tail
      of robust single blocks. The fp32 folds inside a group are emitted in strict
      ascending block order, so byte-exactness holds for any nb.
    - `strip_elision` selects whether the inner half-block strip loop is kept
      ("robust", default -- correct at any VLEN by re-stripping with the
      sumi-carry seed) or elided ("elided" -- a single vsetvl_e8m1(16) + one
      vwredsum per half-block, NO inner strip loop). The elided form is correct
      ONLY at VLEN >= 128 (vsetvl_e8m1(16) caps the active vl at 16 when VLMAX is
      >= 16, covering the whole half-block in one go) and therefore requires the
      m1 integer-core anchor (mf4's vsetvl_e32m1 VLMAX is 4 at VLEN=128, which
      would silently drop 12 of the 16 nibble bytes). The verifier rejects the
      "elided" + non-m1 combination fail-closed (I7); the capability-aware
      autotuner gates "elided" on the Zvl128b capability. The `nb % factor` tail
      always uses the robust single-block core regardless of strip_elision.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$quant_byte_offset,
    I64Attr:$activation_high_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$multi_block_factor,
    OptionalAttr<StrAttr>:$strip_elision
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlQuantContractionOp
    : TCRVRVV_Op<"quant_contraction",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "abstract algorithm-UNCOMMITTED block-quantized contraction request "
      "(plain weights)";
  let description = [{
    Records a block-quantized contraction REQUEST (weight x activation -> fp32)
    that is deliberately NOT yet committed to a concrete kernel ALGORITHM. It is
    the option-2 stage-A abstraction over the two concrete contraction targets
    (tcrv_rvv.q4_0_q8_0_block_dot and tcrv_rvv.repack_gemv_q4_0_q8_0): today the
    repack-vs-block-dot path is frozen by OP IDENTITY in the hand-authored input
    IR, upstream of the compiler. This op moves that choice INTO the compiler by
    carrying the WHAT (quant type, scale model, m_regime, plain block byte
    layout, the runtime column count nc) WITHOUT committing the weight LAYOUT or
    the integration GRANULARITY:

    - **Weight LAYOUT is uncommitted.** The op takes PLAIN (un-repacked) weights
      and pins `weight_layout = "plain"` fail-closed. block-dot reads plain
      bytes (stride 18) -- a plain->plain identity. The repacked GEMV reads
      pre-interleaved block_q4_0x16 (stride 288, interleave 16); that is a
      plain->x16 weight MATERIALIZATION the compiler drives in a later stage, so
      this op deliberately does NOT carry the x16 stride 288, weight_interleave,
      or half_lanes -- it is not pre-committed to the repacked layout.

    - **Integration GRANULARITY is uncommitted.** block-dot (ggml's
      `ggml_vec_dot_q4_0_q8_0`) writes ONE fp32 and DELEGATES the M and N loops
      to ggml's `mul_mat` caller (4 operands, no column loop). The repacked GEMV
      INTERNALIZES the N loop and writes nc fp32 (5 operands, +nc). So this op
      carries `column_count` (nc) ALWAYS so the repack branch can reach it; the
      block-dot lowering DROPS nc and emits the bare per-column vec_dot.

    The op carries the `TCRVEmitCLowerableOpInterface` for parallelism with its
    concrete siblings, but it is NOT added to the EmitC recognizer table: it is
    lowered to a concrete contraction op by the stage-A
    `--tcrv-rvv-lower-quant-contraction` pass BEFORE the EmitC pass runs, so it
    never reaches the EmitC conversion. It has no emitter of its own; the
    identity lowering to block-dot is therefore total and byte-exact.

    The `min_vlen` attribute, when present, is a read-only advisory capability
    snapshot (a deriveMinimumVLEN(march) value) consumed by a later
    selection-logic stage; it never gates correctness in the identity stage.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$column_count,
    AnyType:$vl,
    StrAttr:$quant,
    StrAttr:$scale_model,
    StrAttr:$m_regime,
    I64Attr:$qk,
    StrAttr:$weight_layout,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$quant_byte_offset,
    I64Attr:$activation_high_byte_offset,
    OptionalAttr<I64Attr>:$min_vlen
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` "
    "$column_count `,` $vl attr-dict "
    "`:` type($weight_base) `,` type($activation_base) `,` type($output) `,` "
    "type($element_count) `,` type($column_count) `,` "
    "type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlGemmTileQ40Q80Op
    : TCRVRVV_Op<"q4_0_q8_0_gemm_tile",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q4_0 x Q8_0 GEMM tile (weight-decode reuse) op";
  let description = [{
    Records the ggml Q4_0 x Q8_0 GEMM PREFILL tile -- ONE weight-row block-array
    times M activation columns -- as ONE bounded generic plugin-local typed RVV
    dataflow step. It is the WEIGHT-DECODE-REUSE sibling of
    tcrv_rvv.q4_0_q8_0_block_dot (the per-row vec_dot). The per-(block, column)
    integer+scale arithmetic is BYTE-IDENTICAL to the per-row block dot; the only
    structural difference is the loop nest: for each q4_0 weight block the
    offset-binary nibble decode (vxor.vx 0x88 -> vsll/vsra -> the v0/v1 i8 nibble
    lanes) is computed ONCE and HOISTED above an inner loop over the M activation
    columns, so all M columns reuse the SAME decoded weight lanes. ggml at
    VLEN=128 falls back to M INDEPENDENT ggml_vec_dot_q4_0_q8_0 calls (the repack
    GEMM is disabled at `case 128`), each of which re-decodes the weight every
    column; this tile eliminates that redundant re-decode. The op is byte-exact
    vs M independent ggml_vec_dot_q4_0_q8_0(weight_row, column_j): each column j
    computes EXACTLY what vec_dot computes for (weight_row, column_j) -- the
    weight decode is shared but the per-column products, reduction, and the
    ascending-block-order fp32 fold `sumf_j = sumf_j + ((float)sumi_j * d_x) *
    d_y_j` are identical to vec_dot, with M INDEPENDENT fp32 accumulators (one per
    column).

    The op is nested directly under tcrv_rvv.with_vl and consumes FIVE runtime
    ABI value operands -- the q4_0 weight-row base pointer (vx), the q8_0
    activation columns base pointer (vy), the per-column activation byte stride
    (by, the bytes between two adjacent activation columns), the fp32 output
    pointer (s, the M contiguous outputs s[0..M-1]), the runtime element count
    (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the M fp32 reduction results
    through the output pointer.

    The structural facts -- QK=32, the AoS block strides (18 bytes for
    block_q4_0, 34 bytes for block_q8_0), the quant payload byte offset (+2), the
    q8 high-half byte offset (+16), the per-block dual fp16 scale model
    (d_x . d_y), and the bounded activation-column count M -- are carried as typed
    integer/string attributes (I4 mirror only). The fixed emission -- an outer
    emitc.for block loop over nb = n / QK, per-block weight address arithmetic
    (vx + ib*18), the scalar weight fp16->fp32 read, the HOISTED single
    vsetvl_e8m1(16) weight load + offset-binary decode into v0/v1 (correct at
    VLEN >= 128, the board's mandated full-V floor -- same justification as the
    block-dot's strip_elision="elided" m1 form), an inner emitc.for loop over the
    M activation columns each loading column j's q8 halves (vy + j*by + ib*34),
    replaying the SAME asymmetric vwmul/vwmacc product against the hoisted v0/v1,
    a per-column vwredsum into sumi_j, and the per-column ascending-block-order
    fp32 fold into an M-wide accumulator array, then the M-output store -- is the
    op's lowering, NOT a candidate-mirror string read in the conversion. For the
    bounded surface kind is limited to "ggml_q4_0_q8_0_gemm_tile" and scale_model
    to "dual-fp16-per-block-d_x.d_y".

    G1 scope: M (the activation-column count) is a fixed bounded compile-time
    attr (a small power-of-two-friendly tile such as 4 or 6) and the weight
    decode anchors at the m1 whole-half-block form (one vsetvl_e8m1(16), correct
    at VLEN >= 128). The full ggml GEMM ABI (the row/col loop nest + the *s output
    stride bs) is G2; making M a measurement-tuned cache-blocking knob is G3.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$activation_column_stride,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$quant_byte_offset,
    I64Attr:$activation_high_byte_offset,
    I64Attr:$activation_cols
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $activation_column_stride `,` "
    "$output `,` $element_count `,` $vl attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($activation_column_stride) `,` type($output) `,` "
    "type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlGemmQ40Q80Op
    : TCRVRVV_Op<"q4_0_q8_0_gemm",
    [TCRVEmitCLowerableOpInterface,
     DeclareOpInterfaceMethods<TunableScheduleOpInterface>]> {
  let summary =
      "generic typed RVV ggml Q4_0 x Q8_0 FULL GEMM (NR rows x nc cols) op";
  let description = [{
    Records the COMPLETE ggml Q4_0 x Q8_0 GEMM PREFILL matmul block -- NR weight
    rows times nc activation columns -- as ONE bounded generic plugin-local typed
    RVV dataflow step. It is the FULL-MATMUL wrapper of
    tcrv_rvv.q4_0_q8_0_gemm_tile (the ONE-weight-row weight-decode-reuse tile): an
    outer weight-ROW loop (NR rows) whose body is G1's tile (the hoisted per-block
    weight decode reused across an inner block of M activation columns), iterated
    over the nc columns in M-wide strips. Every output element s[row][col] is
    BYTE-EXACT vs an independent ggml_vec_dot_q4_0_q8_0(weight_row, column_col) --
    the SAME per-(block, column) integer+scale arithmetic and the SAME
    ascending-block-order fp32 fold `sumf = sumf + ((float)sumi * d_x) * d_y` the
    per-row vec_dot computes. ggml at VLEN=128 runs this matmul as nr*nc
    independent ggml_vec_dot_q4_0_q8_0 calls (the repack GEMM is disabled at
    `case 128`), each of which re-decodes the weight block for every output
    column; this op decodes each (row, block) weight ONCE and reuses it across the
    M columns of the inner strip -- exactly the redundant re-decode the per-row
    fallback pays nc-fold and this op pays only nc/M-fold.

    The op is nested directly under tcrv_rvv.with_vl and consumes FIVE runtime ABI
    value operands -- the q4_0 weight base pointer (vx, the NR weight rows laid out
    at byte stride bx = (n / QK) * 18 apart), the q8_0 activation columns base
    pointer (vy, the nc activation columns laid out at byte stride by apart), the
    fp32 output base pointer (s, the NR x nc outputs, row stride bs floats), the
    runtime element count (n, the contraction dim) -- plus the active
    !tcrv_rvv.vl token. The runtime row count nr, column count nc, the weight-row
    byte stride bx, and the output row stride bs (in floats) are carried as
    additional runtime ABI value operands (the full ggml-gemm-like ABI surface).
    It produces ONE i32 LMUL m1 !tcrv_rvv.vector token whose lowering writes the
    NR x nc fp32 reduction results through the output pointer.

    The structural facts -- QK=32, the AoS block strides (18 bytes for block_q4_0,
    34 bytes for block_q8_0), the quant payload byte offset (+2), the q8 high-half
    byte offset (+16), the per-block dual fp16 scale model (d_x . d_y), and the
    bounded inner activation-column block count M -- are carried as typed
    integer/string attributes (I4 mirror only). The fixed emission -- an outer
    emitc.for weight-ROW loop over nr, per-row weight-base address arithmetic
    (vx + ir*bx) and output-row address arithmetic (s + ir*bs), an inner emitc.for
    column-strip loop over the nc columns in M-wide strips (the activation base
    advancing by M*by per strip, the masked tail handling the final nc % M
    columns), and G1's per-block weight-decode-reuse tile body (the outer block
    loop over nb = n / QK, the HOISTED vsetvl_e8m1(16) weight load + offset-binary
    decode into v0/v1, the inner M-column product/reduce/fold into an M-wide fp32
    accumulator array, the M-output store with bounds clamping for the tail) --
    is the op's lowering, NOT a candidate-mirror string read in the conversion.
    For the bounded surface kind is limited to "ggml_q4_0_q8_0_gemm" and
    scale_model to "dual-fp16-per-block-d_x.d_y".

    G2 scope: M (the inner activation-column block) is a fixed bounded compile-time
    attr (a small cache-friendly tile such as 4) and the weight decode anchors at
    the m1 whole-half-block form (one vsetvl_e8m1(16), correct at VLEN >= 128, the
    board's mandated full-V floor). Making M a measurement-tuned cache-blocking
    knob (the L1-capacity-bound column block, M_opt proportional 1/K) is G3.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$activation_column_stride,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$row_count,
    AnyType:$column_count,
    AnyType:$weight_row_stride,
    AnyType:$output_row_stride,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$quant_byte_offset,
    I64Attr:$activation_high_byte_offset,
    OptionalAttr<I64Attr>:$activation_cols
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $activation_column_stride `,` "
    "$output `,` $element_count `,` $row_count `,` $column_count `,` "
    "$weight_row_stride `,` $output_row_stride `,` $vl attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($activation_column_stride) `,` type($output) `,` "
    "type($element_count) `,` type($row_count) `,` type($column_count) `,` "
    "type($weight_row_stride) `,` type($output_row_stride) `,` "
    "type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlRepackGemmQ40Q80Op
    : TCRVRVV_Op<"repack_gemm_q4_0_q8_0",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q4_0 x Q8_0 16x1-REPACKED GEMM (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q4_0 16x1-REPACKED GEMM hot kernel -- the
    validated `vlen128-q4_0-16x1` `ggml_gemm_q4_0_16x1_q8_0` path that consumes
    ALREADY-repacked weights -- as ONE bounded generic plugin-local typed RVV
    dataflow step. It is the BLOCK-AS-LANE sibling of tcrv_rvv.q4_0_q8_0_gemm:
    where that op lays the q4_0 weight block out as a VECTOR and reduces it with
    a cross-lane vwredsum (the per-block reduction wall), THIS op consumes the
    16-block-as-lane repacked layout (block_q4_0x16) so the 16 interleaved
    weight rows of a group occupy 16 distinct vector lanes and the dot product
    accumulates LANE-WISE via vwmacc (NO cross-lane vredsum -- the reduction
    wall is gone). The repacked weight nibbles already carry the offset-binary
    `^ 0x88` bias from the repack-at-load transform (make_block_q4_0x16), so the
    kernel decodes each nibble with a plain vsll/vsra sign-extension (NO in-kernel
    vxor). At VLEN=128 an e16m1 vector holds only 8 i16 lanes, so each
    16-block-as-lane group is processed as TWO disjoint contiguous halves of 8
    lanes each (rows 0..7 then rows 8..15); every weight byte is read exactly
    once.

    The op is nested directly under tcrv_rvv.with_vl and consumes FIVE runtime
    ABI value operands -- the repacked q4_0x16 weight base pointer (vx, the
    NC/16 column-groups laid out at byte stride (n/QK)*288 apart), the repacked
    q8_0x4 activation base pointer (vy, the nr/4 row-groups laid out at byte
    stride (n/QK)*136 apart), the fp32 output base pointer (s), the runtime
    contraction element count (n), the runtime activation-row count (nr) and the
    runtime weight-column count (nc), plus the runtime output row stride (bs, in
    floats) -- plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the nr x nc fp32 results through
    the output pointer.

    The structural facts -- QK=32, the repacked block strides (288 bytes for
    block_q4_0x16, 136 bytes for block_q8_0x4), the repacked quant byte offsets
    (weight qs at +32 after the 16 inline fp16 scales, activation qs at +8 after
    the 4 inline fp16 scales), the interleave widths (16 weight rows per group, 4
    activation columns per group), the resource-aware e16m1 strip width
    (half_lanes, 8 at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one
    16-lane strip; derived from the minimum-VLEN capability fact), and the
    dual-fp16 scale model -- are carried as typed integer/string attributes
    (I4 mirror only). The fixed emission -- the outer activation-row-GROUP loop
    over nr/4, the weight-column-GROUP loop over nc/16, the
    weight_interleave/half_lanes strip loop over the half_lanes-lane strips, the
    4x8 f32 accumulator set, the per-block lane-wise
    vwmacc product (vle8 disjoint sub-loads of the repacked qs, vsll/vsra nibble
    sign-extend, vwmacc_vx against the scalar q8 quants), the vwadd lo/hi combine,
    the vfwmul/vfcvt/vfmacc scale fold, and the vse32 vector store -- is the op's
    lowering, NOT a candidate-mirror string read in the conversion. The bounded
    surface kind is limited to "ggml_repack_gemm_q4_0_q8_0" and scale_model to
    "dual-fp16-per-block-d_x.d_y". Route ids, helper names, ABI strings, artifact
    names, status fields, exact intrinsic spellings, operand mirror strings,
    q8/q4 names, and common EmitC/export code must not infer or choose the
    block-as-lane repacked GEMM semantics.

    The optional `integer_core_lmul` attribute is a bounded resource/scheduling
    fact: it selects the LMUL the per-strip integer-product core anchors at
    ("mf2" or "m1"), which is purely the *how* (vector grouping / strip width),
    NOT the *what*. The repacked dot is byte-exact either way (the 16-way
    interleaved repack reads the SAME bytes; lane-wise vwmacc sums and integer
    addition is order-independent). When the attribute is absent the core anchors
    at "mf2" (the RVV1.0 default: i8mf2 -> i16m1 -> i32m2 -> f32m2, at half_lanes
    e16m1 lanes per strip; 8 at VLEN=128 -> two halves, 16 at VLEN=256 -> one
    strip). When it is "m1" the ENTIRE LMUL chain shifts up one notch (i8m1 ->
    i16m2 -> i32m4 -> f32m4, the f16 scale at m2) and the 16-block-as-lane group
    is ONE 16-lane strip (half_lanes=16): this is the WHOLE-LMUL form RVV0.7.1
    requires (the pre-ratification generation has NO fractional LMUL, so i8mf2 is
    rejected by the XuanTie toolchain). The verifier pins the legal
    (integer_core_lmul, half_lanes) combinations fail-closed (I7).

    This op pins the 16x1 repacked ABI fail-closed (I7): the byte layout the
    kernel depends on (288/136 strides, +32/+8 quant offsets, 16/4 interleaves)
    is the validated `vlen128-q4_0-16x1` contract; half_lanes is the
    resource-aware strip width in {8, 16} (it must divide the 16-way interleave,
    and a 16-lane strip reads byte-identical repacked data to two 8-lane halves).
    A malformed typed body cannot lower under this emission.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$row_count,
    AnyType:$column_count,
    AnyType:$output_row_stride,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$weight_interleave,
    I64Attr:$activation_interleave,
    I64Attr:$half_lanes,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` "
    "$row_count `,` $column_count `,` $output_row_stride `,` $vl attr-dict "
    "`:` type($weight_base) `,` type($activation_base) `,` type($output) `,` "
    "type($element_count) `,` type($row_count) `,` type($column_count) `,` "
    "type($output_row_stride) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlRepackGemvQ40Q80Op
    : TCRVRVV_Op<"repack_gemv_q4_0_q8_0",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q4_0 x Q8_0 16x1-REPACKED GEMV (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q4_0 16x1-REPACKED GEMV hot kernel -- the
    validated `vlen128-q4_0-16x1` `ggml_gemv_q4_0_16x1_q8_0` decode path that
    consumes ALREADY-repacked weights -- as ONE bounded generic plugin-local
    typed RVV dataflow step. It is the SINGLE-OUTPUT-COLUMN (decode) sibling of
    tcrv_rvv.repack_gemm_q4_0_q8_0: the weight side is the SAME block_q4_0x16
    block-as-lane layout (the 16 interleaved weight rows of a group occupy 16
    distinct vector lanes so the dot accumulates LANE-WISE via vwmacc, NO
    cross-lane vredsum -- the per-block reduction wall is gone), but the
    activation is ONE plain block_q8_0 stream (NOT the GEMM's interleaved
    block_q8_0x4): a single activation column. The repacked weight nibbles
    already carry the offset-binary `^ 0x88` bias from the repack-at-load
    transform (make_block_q4_0x16), so the kernel decodes each nibble with a
    plain vsll/vsra sign-extension (NO in-kernel vxor). At VLEN=128 an e16m1
    vector holds only 8 i16 lanes, so each 16-block-as-lane group is processed
    as TWO disjoint contiguous halves of 8 lanes each (rows 0..7 then rows
    8..15) inline (no half loop); every weight byte is read exactly once.

    The op is nested directly under tcrv_rvv.with_vl and consumes THREE runtime
    ABI value operands -- the repacked q4_0x16 weight base pointer (vx, the
    NC/16 column-groups laid out at byte stride (n/QK)*288 apart), the plain
    q8_0 activation base pointer (vy, one block_q8_0 stream at byte stride 34),
    the fp32 output base pointer (s) -- plus the runtime contraction element
    count (n) and the runtime weight-column count (nc), plus the active
    !tcrv_rvv.vl token. It produces ONE i32 LMUL m1 !tcrv_rvv.vector token whose
    lowering writes the nc fp32 results through the output pointer. There is NO
    activation-row count (nr is 1) and NO output row stride (the single-column
    store is `s + x*16`, no bs); a fail-closed verifier pins ONLY what the
    kernel depends on.

    The structural facts -- QK=32, the repacked weight block stride (288 bytes
    for block_q4_0x16), the plain activation block stride (34 bytes for
    block_q8_0), the repacked weight quant byte offset (qs at +32 after the 16
    inline fp16 scales), the plain activation quant byte offset (qs at +2 after
    the single inline fp16 scale), the weight interleave width (16 weight rows
    per group), the resource-aware e16m1 strip width (half_lanes, 8 at VLEN=128
    -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane strip; derived from the
    minimum-VLEN capability fact), and the dual-fp16 scale model -- are carried
    as typed integer/string attributes (I4 mirror only). The fixed emission --
    the outer weight-column-GROUP loop over nc/16, the weight_interleave/half_lanes
    f32 accumulator strips (e.g. rows 0..7, 8..15 at half_lanes=8), the inner
    contraction-block loop over nb, the per-block lane-wise vwmacc product (vle8
    disjoint sub-loads of the repacked qs at qs[i*16 + h*half_lanes], vsll/vsra
    nibble sign-extend, vwmacc_vx against the scalar q8 quant qs[i] low / qs[16+i]
    high), the vwadd lo/hi combine, the vfwmul/vfcvt/vfmacc scale fold (one
    activation scale broadcast into every strip), and the per-strip vse32 vector
    stores (s + x*16 + h*half_lanes) -- is the op's lowering, NOT a candidate-mirror string read in the
    conversion. The bounded surface kind is limited to
    "ggml_repack_gemv_q4_0_q8_0" and scale_model to
    "dual-fp16-per-block-d_x.d_y". Route ids, helper names, ABI strings,
    artifact names, status fields, exact intrinsic spellings, operand mirror
    strings, q8/q4 names, and common EmitC/export code must not infer or choose
    the block-as-lane repacked GEMV semantics.

    The optional `integer_core_lmul` attribute is a bounded resource/scheduling
    fact: it selects the LMUL the per-strip integer-product core anchors at
    ("mf2" or "m1"), which is purely the *how* (vector grouping / strip width),
    NOT the *what*. The repacked dot is byte-exact either way (the 16-way
    interleaved repack reads the SAME bytes; lane-wise vwmacc sums and integer
    addition is order-independent). When the attribute is absent the core anchors
    at "mf2" (the RVV1.0 default: i8mf2 -> i16m1 -> i32m2 -> f32m2, at half_lanes
    e16m1 lanes per strip; 8 at VLEN=128 -> two halves, 16 at VLEN=256 -> one
    strip). When it is "m1" the ENTIRE LMUL chain shifts up one notch (i8m1 ->
    i16m2 -> i32m4 -> f32m4, the f16 scale at m2) and the 16-block-as-lane group
    is ONE 16-lane strip (half_lanes=16): this is the WHOLE-LMUL form RVV0.7.1
    requires (the pre-ratification generation has NO fractional LMUL, so i8mf2 is
    rejected by the XuanTie toolchain). The verifier pins the legal
    (integer_core_lmul, half_lanes) combinations fail-closed (I7).

    This op pins the 16x1 repacked GEMV ABI fail-closed (I7): the byte layout
    the kernel depends on (weight 288/+32/16, activation 34/+2) is the validated
    `vlen128-q4_0-16x1` decode contract; half_lanes is the resource-aware strip
    width in {8, 16} (it must divide the 16-way interleave, and a 16-lane strip
    reads byte-identical repacked data to two 8-lane halves). A malformed typed
    body cannot lower under this emission.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$column_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$weight_interleave,
    I64Attr:$half_lanes,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` "
    "$column_count `,` $vl attr-dict "
    "`:` type($weight_base) `,` type($activation_base) `,` type($output) `,` "
    "type($element_count) `,` type($column_count) `,` "
    "type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlPackQ40ToX16Op
    : TCRVRVV_Op<"pack_q4_0_to_q4_0x16",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml plain block_q4_0 -> block_q4_0x16 PACK (materialize) op";
  let description = [{
    Records the COMPLETE ggml plain->x16 weight MATERIALIZATION transform
    (make_block_q4_0x16, the live blck_size_interleave==1 branch of
    repack_q4_0_to_q4_0_16_bl) as ONE bounded generic plugin-local typed RVV
    dataflow step. It is the option-2 stage-C1b PRODUCER sibling of the
    block-as-lane CONSUMER repack-GEMV/GEMM ops: those ops READ pre-x16 bytes;
    this op EMITS the pack that PRODUCES them. The compiler thus does not merely
    CONSUME the x16 layout it declares (every existing Ggml*RepackGemv* emitter
    does that) -- it can also PRODUCE the layout it declares.

    The transform is a PURE byte gather + XOR -- NO fp arithmetic, NO
    quantization, NO vector reduction, NO setvl/with_vl machinery in the emitted
    body (this op is a STRICT SUBSET of the repack-GEMV op: NO vl/LMUL,
    activation, scale-fold, accumulator, or half-lane strip). The input is a
    plain block_q4_0 stream (stride 18: fp16 d @+0, 16 nibble bytes @+2). The
    output is a block_q4_0x16 stream (stride 288: 16 inline fp16 scales d[16]
    @0..32, 256 interleaved nibble bytes qs[256] @32..288). For each output
    block, given the 16 consecutive source blocks in[0..16):
      * scales (copy, no transform): out.d[i] = in[i].d for i in [0,16);
      * quants (16-way interleave + ^0x88 bias bake): for i in [0,256),
        out.qs[i] = in[i%16].qs[i/16] ^ 0x88. The interleave is
        block-major-within-byte (consecutive output bytes walk the 16 source
        blocks at a fixed source offset, then advance the offset); the ^0x88
        flips both nibbles' sign bit to the offset-binary bias the consumer GEMV
        already expects.

    The op consumes TWO runtime ABI value operands -- the plain q4_0 source base
    pointer (src, const uint8_t*) and the x16 destination base pointer (dst,
    uint8_t*) -- plus the runtime output-block count (nblocks). It produces ONE
    bounded token result (the dataflow sink); the lowering writes the packed x16
    bytes through the destination pointer and the result carries no live value.

    The structural facts -- QK=32, the source block stride (18 bytes for
    block_q4_0), the destination block stride (288 bytes for block_q4_0x16), the
    source quant byte offset (qs at +2 after the single inline fp16 scale), the
    destination quant byte offset (qs at +32 after the 16 inline fp16 scales),
    the interleave width (16 source blocks per output block), and the offset-binary
    XOR mask (0x88) -- are carried as typed integer attributes (I4 mirror only).
    The fixed pack emission (the scalar gather + XOR loop DERIVED from these
    attrs) is the op's lowering, NOT a candidate-mirror string read in the
    conversion. The bounded surface kind is limited to "ggml_pack_q4_0_to_q4_0x16".

    This op pins the pack ABI fail-closed (I7): the byte layout the materializer
    depends on (src stride 18 / quant +2, dst stride 288 / quant +32, interleave
    16, qk 32, xor 0x88) is the make_block_q4_0x16 contract. A malformed typed
    body cannot lower under this emission.
  }];

  let arguments = (ins
    AnyType:$src,
    AnyType:$dst,
    AnyType:$nblocks,
    AnyType:$vl,
    StrAttr:$kind,
    I64Attr:$qk,
    I64Attr:$src_block_stride,
    I64Attr:$dst_block_stride,
    I64Attr:$src_quant_byte_offset,
    I64Attr:$dst_quant_byte_offset,
    I64Attr:$weight_interleave,
    I64Attr:$xor_mask
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$src `,` $dst `,` $nblocks `,` $vl attr-dict "
    "`:` type($src) `,` type($dst) `,` type($nblocks) `,` "
    "type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlRepackGemvQ41Q81Op
    : TCRVRVV_Op<"repack_gemv_q4_1_q8_1",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q4_1 x Q8_1 16x1-REPACKED GEMV (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q4_1 16x1-REPACKED GEMV (decode) hot kernel as ONE
    bounded generic plugin-local typed RVV dataflow step -- the FAMILY-B
    (scale+MIN, asymmetric) block-as-lane sibling of
    tcrv_rvv.repack_gemv_q4_0_q8_0. The weight side is the block_q4_1x16
    block-as-lane layout: the 16 interleaved weight rows of a group occupy 16
    distinct vector lanes so the dot accumulates LANE-WISE via vwmacc (NO
    cross-lane vredsum -- the per-block reduction wall is gone), and the
    activation is ONE plain block_q8_1 stream (NOT an interleaved x4). At VLEN=128
    an e16m1 vector holds only 8 i16 lanes, so each 16-block-as-lane group is
    processed as TWO disjoint contiguous halves of 8 lanes each (rows 0..7 then
    rows 8..15) inline; at VLEN=256 it is ONE 16-lane strip. Every weight byte is
    read exactly once.

    Two semantic differences from the Q4_0 repacked GEMV sibling distinguish
    Family B (exactly as the q4_1 block-dot differs from the q4_0 block-dot):

    (1) The integer core decodes UNSIGNED nibbles [0,15] -- NO offset-binary `-8`
        bias and NO repack-time `^ 0x88` (the q4_1 quantization is asymmetric, the
        bias lives in the separate per-block MIN scale). The decode is `vand 0x0F`
        (low nibble) / `vsrl 0x04` (high nibble, a LOGICAL shift) on the UNSIGNED
        weight lane, then a value-identity reinterpret to i8, then the SAME signed
        widening vwmacc against the plain-i8 q8_1 quants. The repacked weight
        nibbles are stored RAW (NOT biased), so the kernel does NOT vsll/vsra
        sign-extend like the q4_0 sibling.

    (2) The per-strip fold carries the per-block MIN correction LANE-WISE.
        block_q4_1x16 stores 16 inline fp16 `d` (per row), then 16 inline fp16 `m`
        (per row, the MIN), then the 256 interleaved nibble bytes; block_q8_1
        stores fp16 `d` then fp16 `s` (the PRECOMPUTED scaled activation sum
        d_y*sum(q8)) then the 32 int8 quants. ggml's exact per-row statement is
        `sumf += (d_x*d_y)*sumi + m_x*s_y`: the (d_x*d_y)*sumi scale term plus the
        m_x*s_y MIN term. Here d_x and m_x are VECTOR strips (one lane per weight
        row, vle16), d_y and s_y are the single activation column's SCALARS. The
        MIN term is folded LANE-WISE with one vfmacc_vf (sumf += s_y * m_x) -- the
        lane-wise analogue of the q4_1 block-dot's scalar m_x*s_y.

    The op is nested directly under tcrv_rvv.with_vl and consumes THREE runtime
    ABI value operands -- the repacked q4_1x16 weight base pointer (vx, NC/16
    column-groups at byte stride (n/QK)*320), the plain q8_1 activation base
    pointer (vy, one block_q8_1 stream at byte stride 36), the fp32 output base
    pointer (s) -- plus the runtime contraction element count (n) and the runtime
    weight-column count (nc), plus the active !tcrv_rvv.vl token. It produces ONE
    i32 LMUL m1 !tcrv_rvv.vector token whose lowering writes the nc fp32 results
    through the output pointer. There is NO activation-row count (nr is 1) and NO
    output row stride (single-column store `s + x*16`).

    The structural facts -- QK=32, the repacked weight block stride (320 bytes for
    block_q4_1x16 = 16 fp16 d + 16 fp16 m + 256 nibble bytes), the plain
    activation block stride (36 bytes for block_q8_1), the repacked weight quant
    byte offset (qs at +64 after the 16 d + 16 m fp16 scales), the per-row weight
    MIN byte offset (m strip at +32 after the 16 d scales), the plain activation
    quant byte offset (qs at +4 after the d + s fp16 scales), the activation
    scaled-sum byte offset (s at +2 after d), the weight interleave width (16),
    the resource-aware e16m1 strip width (half_lanes in {8,16}), and the
    dual-fp16-plus-min scale model -- are carried as typed integer/string
    attributes (I4 mirror only). The fixed emission is the op's lowering, NOT a
    candidate-mirror string read in the conversion. The bounded surface kind is
    limited to "ggml_repack_gemv_q4_1_q8_1" and scale_model to
    "dual-fp16-per-block-d_x.d_y-plus-min". Route ids, helper names, ABI strings,
    artifact names, status fields, exact intrinsic spellings, operand mirror
    strings, q8/q4 names, and common EmitC/export code must not infer or choose
    the block-as-lane repacked GEMV semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded
    resource/scheduling fact the q4_0 repacked GEMV sibling carries (the *how*,
    never the *what*; the 16-way interleaved repack reads the SAME bytes either
    way): absent / "mf2" anchors the RVV1.0 fractional chain (half_lanes in
    {8,16}), "m1" anchors the RVV0.7.1 whole-LMUL chain (half_lanes 16, ONE
    16-lane strip). The verifier pins the legal (integer_core_lmul, half_lanes)
    combinations fail-closed (I7); half_lanes must divide the 16-way interleave.
    A malformed typed body cannot lower under this emission.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$column_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$weight_min_byte_offset,
    I64Attr:$activation_sum_byte_offset,
    I64Attr:$weight_interleave,
    I64Attr:$half_lanes,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` "
    "$column_count `,` $vl attr-dict "
    "`:` type($weight_base) `,` type($activation_base) `,` type($output) `,` "
    "type($element_count) `,` type($column_count) `,` "
    "type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlRepackGemmQ41Q81Op
    : TCRVRVV_Op<"repack_gemm_q4_1_q8_1",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q4_1 x Q8_1 16x1-REPACKED GEMM (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q4_1 16x1-REPACKED GEMM (prefill) hot kernel as ONE
    bounded generic plugin-local typed RVV dataflow step -- the FAMILY-B (scale+MIN,
    asymmetric) PREFILL sibling of tcrv_rvv.repack_gemv_q4_1_q8_1, and the q4_1
    sibling of tcrv_rvv.repack_gemm_q4_0_q8_0. It is the BLOCK-AS-LANE matmul: the
    16 interleaved weight rows of a group occupy 16 distinct vector lanes so the dot
    accumulates LANE-WISE via vwmacc (NO cross-lane vredsum -- the per-block
    reduction wall is gone). The activation is the interleaved block_q8_1x4 stream
    (4 activation columns per group, the q8_1 analogue of the q4_0 GEMM's
    block_q8_0x4). At VLEN=128 an e16m1 vector holds 8 i16 lanes, so each
    16-block-as-lane group is processed as TWO disjoint contiguous 8-lane halves
    (rows 0..7 then rows 8..15); at VLEN=256 it is ONE 16-lane strip. Every weight
    byte is read exactly once.

    Two semantic differences from the Q4_0 repacked GEMM sibling distinguish
    Family B (exactly as the q4_1 repacked GEMV differs from the q4_0 repacked GEMV):

    (1) The integer core decodes UNSIGNED nibbles [0,15] -- NO offset-binary `-8`
        bias and NO repack-time `^ 0x88`. The decode is `vand 0x0F` (low nibble) /
        `vsrl 0x04` (high nibble, LOGICAL shift) on the UNSIGNED weight lane, then a
        value-identity reinterpret to i8, then the SAME signed widening vwmacc
        against the plain-i8 q8_1 quants. The repacked weight nibbles are stored RAW.

    (2) The per-strip fold carries the per-block MIN correction LANE-WISE, PER
        ACTIVATION COLUMN. block_q4_1x16 stores 16 inline fp16 `d` (per row), then 16
        inline fp16 `m` (per row, the MIN), then the 256 interleaved nibble bytes;
        block_q8_1x4 stores 4 inline fp16 `d` then 4 inline fp16 `s` (the precomputed
        scaled activation sums d_y*sum(q8), one per column) then the 128 interleaved
        int8 quants. ggml's per-(row,col) statement is
        `sumf += (d_x*d_y_c)*sumi + m_x*s_y_c`: the scale term plus the m_x*s_y_c MIN
        term. d_x and m_x are VECTOR strips (one lane per weight row, vle16), d_y_c
        and s_y_c are the per-column SCALARS. The MIN term is folded LANE-WISE with a
        vfwmul(m_x, s_y_c) then vfadd, the per-column analogue of the GEVM's single
        m_x*s_y.

    The op is nested directly under tcrv_rvv.with_vl and consumes SEVEN runtime ABI
    value operands -- the repacked q4_1x16 weight base pointer (vx, NC/16
    column-groups at byte stride (n/QK)*320), the repacked q8_1x4 activation base
    pointer (vy, nr/4 row-groups at byte stride (n/QK)*144), the fp32 output base
    pointer (s), the runtime contraction element count (n), the runtime
    activation-row count (nr) and weight-column count (nc), and the runtime output
    row stride (bs, in floats) -- plus the active !tcrv_rvv.vl token. It produces ONE
    i32 LMUL m1 !tcrv_rvv.vector token whose lowering writes the nr x nc fp32 results
    through the output pointer.

    The structural facts -- QK=32, the repacked block strides (320 for block_q4_1x16,
    144 for block_q8_1x4), the repacked quant byte offsets (weight qs at +64 after
    the 16 d + 16 m fp16 scales, activation qs at +16 after the 4 d + 4 s fp16
    scales), the per-row weight MIN byte offset (m strip at +32 after the 16 d
    scales), the activation scaled-sum byte offset (s at +8 after the 4 d scales),
    the interleave widths (16 weight rows, 4 activation columns), the resource-aware
    e16m1 strip width (half_lanes in {8,16}), and the dual-fp16-plus-min scale model
    -- are carried as typed integer/string attributes (I4 mirror only). The fixed
    emission is the op's lowering, NOT a candidate-mirror string read in the
    conversion. The bounded surface kind is limited to "ggml_repack_gemm_q4_1_q8_1"
    and scale_model to "dual-fp16-per-block-d_x.d_y-plus-min". Route ids, helper
    names, ABI strings, artifact names, status fields, exact intrinsic spellings,
    operand mirror strings, q8/q4 names, and common EmitC/export code must not infer
    or choose the block-as-lane repacked GEMM semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded
    resource/scheduling fact the q4_0 repacked GEMM carries (the *how*, never the
    *what*; the 16-way interleaved repack reads the SAME bytes either way): absent /
    "mf2" anchors the RVV1.0 fractional chain (half_lanes in {8,16}), "m1" anchors
    the RVV0.7.1 whole-LMUL chain (half_lanes 16, ONE 16-lane strip). The verifier
    pins the legal (integer_core_lmul, half_lanes) combinations fail-closed (I7);
    half_lanes must divide the 16-way interleave. A malformed typed body cannot lower
    under this emission.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$row_count,
    AnyType:$column_count,
    AnyType:$output_row_stride,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$weight_min_byte_offset,
    I64Attr:$activation_sum_byte_offset,
    I64Attr:$weight_interleave,
    I64Attr:$activation_interleave,
    I64Attr:$half_lanes,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` "
    "$row_count `,` $column_count `,` $output_row_stride `,` $vl attr-dict "
    "`:` type($weight_base) `,` type($activation_base) `,` type($output) `,` "
    "type($element_count) `,` type($row_count) `,` type($column_count) `,` "
    "type($output_row_stride) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlRepackGemvQ80Q80Op
    : TCRVRVV_Op<"repack_gemv_q8_0_q8_0", [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV ggml Q8_0 x Q8_0 16x1-REPACKED GEMV (block-as-lane) op";
  let description = [{ The block_q8_0x16 16x1-repacked GEMV decode sibling of repack_gemv_q4_0_q8_0.
    Weights are FULL int8 (block_q8_0x16: 16 interleaved block_q8_0, each = one fp16 scale + 32 int8 quants),
    so the kernel reads int8 weight lanes DIRECTLY (NO nibble vsll/vsra, NO ^0x88 bias) and accumulates
    LANE-WISE across 16 column lanes (NO cross-lane vredsum). Full int8 products overflow i16, so the
    in-block accumulation is i32 (vwmul i8xi8->i16 product + vwadd_wv i32), NOT a single-instruction
    vwmacc i8xi8->i32. Activation is one
    plain block_q8_0 stream (stride 34, qs at +2). Dual-fp16 scale fold d_x*d_y. half_lanes = 8@VLEN128 (two
    8-lane halves) / 16@VLEN256 (one strip). integer_core_lmul selects mf2 (default) vs m1 (whole-LMUL); the
    dot is byte-exact either way. Verifier pins the block_q8_0x16 ABI fail-closed (I7). }];
  let arguments = (ins AnyType:$weight_base, AnyType:$activation_base, AnyType:$output,
    AnyType:$element_count, AnyType:$column_count, AnyType:$vl, StrAttr:$kind, StrAttr:$scale_model,
    I64Attr:$qk, I64Attr:$weight_block_stride, I64Attr:$activation_block_stride,
    I64Attr:$weight_quant_byte_offset, I64Attr:$activation_quant_byte_offset, I64Attr:$weight_interleave,
    I64Attr:$half_lanes, OptionalAttr<StrAttr>:$integer_core_lmul);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $column_count `,` $vl attr-dict `:` type($weight_base) `,` type($activation_base) `,` type($output) `,` type($element_count) `,` type($column_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlRepackGemvQ4KQ8KOp
    : TCRVRVV_Op<"repack_gemv_q4_K_q8_K",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q4_K x Q8_K 16x1-REPACKED GEMV (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q4_K 16x1-REPACKED GEMV (decode) hot kernel as ONE
    bounded generic plugin-local typed RVV dataflow step -- the K-QUANT
    (super-block) block-as-lane sibling of tcrv_rvv.repack_gemv_q4_1_q8_1. It is
    the dominant-quant headline path: q4_K is q4_1's dual scale+MIN fold EXTENDED
    to the K-quant SUPER-BLOCK structure -- 256 elements per super-block split
    into 8 sub-blocks of 32, EACH sub-block carrying its own 6-bit quantized scale
    and 6-bit quantized min, with ONE super-block (d, dmin) fp16 pair multiplying
    the unpacked per-sub-block scales/mins. The weight side is the block_q4_Kx16
    block-as-lane layout: the 16 interleaved weight columns of a group occupy 16
    distinct vector lanes so the dot accumulates LANE-WISE via vwmacc (NO
    cross-lane vredsum -- the per-block reduction wall is gone), and the activation
    is ONE plain block_q8_K stream (NOT an interleaved x4 -- the GEVM mat side is
    plain block_q8_K, the GEMM prefill sibling will consume block_q8_Kx4). At
    VLEN=128 an e8 vector holds only 8 i8 lanes per half, so each 16-block-as-lane
    group is processed as TWO disjoint contiguous halves of 8 lanes each (cols
    0..7 then cols 8..15) inline; at VLEN=256 it is ONE 16-lane strip. Every
    weight byte is read exactly once.

    Three structural extensions over the Q4_1 repacked GEVM sibling distinguish
    the K-quant super-block (exactly as the q4_K block-dot differs from the q4_1
    block-dot):

    (1) The super-block scale fold is DUAL-LEVEL. block_q4_Kx16 stores, per
        column lane, ONE fp16 super-block `d` and ONE fp16 super-block `dmin`
        (the 16 d's then the 16 dmin's inline), then a custom 192-byte `scales`
        region packing the 8 sub-block 6-bit scales AND 8 sub-block 6-bit mins
        per column, then the 2048 interleaved nibble bytes. The per-sub-block
        6-bit scale/min are unpacked LANE-WISE (vle8 across the 16 column lanes,
        vand 0x0F / vsrl / vsll bit-dance), multiplied by the super-block d/dmin.
        ggml's exact per-super-block statement is `sumf += d * (sum over 8
        sub-blocks of scale_j * q_dot_j) - dmin * (sum over 8 sub-blocks of
        min_j * bsum_j)`: the scale term accumulates the 8 per-sub-block integer
        dots weighted by the per-sub-block 6-bit scale, and the MIN term
        subtracts the 8 per-sub-block 6-bit mins weighted by the activation
        bsums. d and dmin are VECTOR strips (one lane per weight column, vle16),
        the activation `d` is the single column's fp32 SCALAR.

    (2) The MIN term consumes the activation block_q8_K `bsums` (per-group-of-16
        quant sums) LANE-WISE: `sumf -= (dmin*d_y) * sum_j(min_j * bsum_pair_j)`.
        block_q8_K stores fp32 `d` (NOT fp16, NOT a precomputed scaled `s` like
        q8_1), then 256 int8 quants, then 16 int16 bsums. The min correction is
        folded LANE-WISE via vwmacc into a per-column i32 accumulator then scaled
        by (dmin*d_y) -- the lane-wise analogue of the q4_K block-dot's scalar
        min fold.

    (3) Eight sub-blocks of 32. The inner integer core loops over the 8 sub-blocks
        (n_subblocks == 8), each a 32-element 4-bit nibble strip, accumulating a
        per-sub-block i16/i32 lane-wise product weighted by that sub-block's 6-bit
        scale -- NO per-sub-block vredsum (lane-wise across the 16 columns).

    The op is nested directly under tcrv_rvv.with_vl and consumes THREE runtime
    ABI value operands -- the repacked q4_Kx16 weight base pointer (vx, NC/16
    column-groups at byte stride (n/QK_K)*2304), the plain q8_K activation base
    pointer (vy, one block_q8_K stream at byte stride 292), the fp32 output base
    pointer (s) -- plus the runtime contraction element count (n) and the runtime
    weight-column count (nc), plus the active !tcrv_rvv.vl token. It produces ONE
    i32 LMUL m1 !tcrv_rvv.vector token whose lowering writes the nc fp32 results
    through the output pointer. There is NO activation-row count (nr is 1) and NO
    output row stride (single-column store `s + x*16`).

    The structural facts -- QK_K=256, the repacked weight super-block stride (2304
    bytes for block_q4_Kx16 = 16 fp16 d + 16 fp16 dmin + 192 scales/mins + 2048
    nibble bytes), the plain activation super-block stride (292 bytes for
    block_q8_K = fp32 d + 256 int8 quants + 16 int16 bsums), the repacked weight
    quant byte offset (qs at +256 after the 32 d/dmin fp16 + 192 scales bytes),
    the per-column weight dmin byte offset (dmin strip at +32 after the 16 d
    scales), the custom 6-bit per-sub-block scales/mins region byte offset (scales
    at +64 after the 16 d + 16 dmin fp16), the plain activation quant byte offset
    (qs at +4 after the fp32 d), the activation bsums byte offset (bsums at +260
    after the fp32 d + 256 quants), the sub-block count (8), the weight interleave
    width (16), the resource-aware strip width (half_lanes in {8,16}), and the
    super-block-d.dmin-plus-bsums-min scale model -- are carried as typed
    integer/string attributes (I4 mirror only). The fixed emission is the op's
    lowering, NOT a candidate-mirror string read in the conversion. The bounded
    surface kind is limited to "ggml_repack_gemv_q4_K_q8_K" and scale_model to
    "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks". Route ids, helper names,
    ABI strings, artifact names, status fields, exact intrinsic spellings, operand
    mirror strings, q8/q4 names, and common EmitC/export code must not infer or
    choose the block-as-lane repacked GEVM semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded
    resource/scheduling fact the q4_1 repacked GEVM sibling carries (the *how*,
    never the *what*; the 16-way interleaved repack reads the SAME bytes either
    way): absent / "mf2" anchors the RVV1.0 fractional chain (half_lanes in
    {8,16}), "m1" anchors the RVV0.7.1 whole-LMUL chain (half_lanes 16, ONE
    16-lane strip). The verifier pins the legal (integer_core_lmul, half_lanes)
    combinations fail-closed (I7); half_lanes must divide the 16-way interleave.
    A malformed typed body cannot lower under this emission. (NOTE: the lowering
    emitter, dispatch, recognizer, Win-A LMUL-parametric 6-bit unpack stamp, and
    lit are STAGE-1b/1c -- this op + verifier are STAGE-1a; the op does not yet
    lower.)
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$column_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$weight_dmin_byte_offset,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$activation_bsums_byte_offset,
    I64Attr:$n_subblocks,
    I64Attr:$weight_interleave,
    I64Attr:$half_lanes,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` "
    "$column_count `,` $vl attr-dict "
    "`:` type($weight_base) `,` type($activation_base) `,` type($output) `,` "
    "type($element_count) `,` type($column_count) `,` "
    "type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlRepackGemmQ4KQ8KOp
    : TCRVRVV_Op<"repack_gemm_q4_K_q8_K",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q4_K x Q8_K 16x1-REPACKED GEMM (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q4_K 16x1-REPACKED GEMM (prefill, M>>1) hot kernel
    as ONE bounded generic plugin-local typed RVV dataflow step -- the K-QUANT
    (super-block) PREFILL sibling of tcrv_rvv.repack_gemv_q4_K_q8_K, and the q4_K
    sibling of tcrv_rvv.repack_gemm_q4_1_q8_1. It is the dominant-quant PREFILL
    headline path: q4_K is q4_1's dual scale+MIN fold EXTENDED to the K-quant
    SUPER-BLOCK structure -- 256 elements per super-block split into 8 sub-blocks
    of 32, EACH sub-block carrying its own 6-bit quantized scale and 6-bit
    quantized min, with ONE super-block (d, dmin) fp16 pair multiplying the
    unpacked per-sub-block scales/mins. The weight side is the block_q4_Kx16
    block-as-lane layout (IDENTICAL to the GEVM weight ABI): the 16 interleaved
    weight columns of a group occupy 16 distinct vector lanes so the dot
    accumulates LANE-WISE via vwmacc (NO cross-lane vredsum -- the per-block
    reduction wall is gone). The DISTINGUISHING fact from the GEVM is the
    activation: instead of ONE plain block_q8_K stream, the PREFILL kernel
    consumes the interleaved block_q8_Kx4 stream (4 activation ROWS per group,
    the q8_K analogue of the q4_0/q4_1 GEMM's block_q8_0x4/block_q8_1x4). The
    6-bit-per-sub-block unpack cost is AMORTIZED over the 4 (and, across the row
    loop, M) activation columns -- the structural reason the prefill GEMM is the
    e2e-win path where the single-column GEVM was not. At VLEN=128 an e8 vector
    holds only 8 i8 lanes per half, so each 16-block-as-lane group is processed
    as TWO disjoint contiguous halves of 8 lanes each (cols 0..7 then cols 8..15)
    inline; at VLEN=256 it is ONE 16-lane strip. Every weight byte is read
    exactly once.

    Three structural extensions over the Q4_1 repacked GEMM sibling distinguish
    the K-quant super-block (exactly as the q4_K repacked GEVM differs from the
    q4_1 repacked GEVM):

    (1) The super-block scale fold is DUAL-LEVEL. block_q4_Kx16 stores, per
        column lane, ONE fp16 super-block `d` and ONE fp16 super-block `dmin`
        (the 16 d's then the 16 dmin's inline), then a custom 192-byte `scales`
        region packing the 8 sub-block 6-bit scales AND 8 sub-block 6-bit mins
        per column, then the 2048 interleaved nibble bytes. The per-sub-block
        6-bit scale/min are unpacked LANE-WISE (vle8 across the 16 column lanes,
        vand 0x0F / vsrl / vsll bit-dance), multiplied by the super-block d/dmin.
        ggml's exact per-super-block per-COLUMN statement is `sumf += d_y_c *
        (d * (sum over 8 sub-blocks of scale_j * q_dot_jc) - dmin * (sum over 8
        sub-blocks of min_j * bsum_pair_jc))`: the scale term accumulates the 8
        per-sub-block integer dots weighted by the per-sub-block 6-bit scale, the
        MIN term subtracts the 8 per-sub-block 6-bit mins weighted by that
        activation column's bsums, and d_y_c is the per-COLUMN fp32 activation
        scalar. d and dmin are VECTOR strips (one lane per weight column, vle16);
        the activation `d` is now 4 fp32 SCALARS (one per interleaved column),
        the per-column analogue of the GEVM's single fp32 d.

    (2) The MIN term consumes the interleaved block_q8_Kx4 `bsums` (per-group-of-16
        quant sums, PER COLUMN) LANE-WISE. block_q8_Kx4 stores 4 fp32 `d` (NOT
        fp16, NOT a precomputed scaled `s` like q8_1x4 -- q8_K carries raw bsums,
        NOT a folded sum), then 256*4 = 1024 int8 quants (4 columns interleaved),
        then 64 int16 bsums (16 per column * 4 columns). The min correction is
        folded LANE-WISE via vwmacc into a per-(column,weight-column) i32
        accumulator then scaled by (dmin*d_y_c) -- the 4-column-interleaved
        analogue of the GEVM's single-column min fold.

    (3) Eight sub-blocks of 32. The inner integer core loops over the 8 sub-blocks
        (n_subblocks == 8), each a 32-element 4-bit nibble strip, accumulating a
        per-sub-block i16/i32 lane-wise product weighted by that sub-block's 6-bit
        scale -- NO per-sub-block vredsum (lane-wise across the 16 columns).

    The op is nested directly under tcrv_rvv.with_vl and consumes SEVEN runtime
    ABI value operands -- the repacked q4_Kx16 weight base pointer (vx, NC/16
    column-groups at byte stride (n/QK_K)*2304), the repacked q8_Kx4 activation
    base pointer (vy, nr/4 row-groups at byte stride (n/QK_K)*1168), the fp32
    output base pointer (s), the runtime contraction element count (n), the
    runtime activation-row count (nr) and weight-column count (nc), and the
    runtime output row stride (bs, in floats) -- plus the active !tcrv_rvv.vl
    token. It produces ONE i32 LMUL m1 !tcrv_rvv.vector token whose lowering
    writes the nr x nc fp32 results through the output pointer.

    The structural facts -- QK_K=256, the repacked weight super-block stride (2304
    bytes for block_q4_Kx16 = 16 fp16 d + 16 fp16 dmin + 192 scales/mins + 2048
    nibble bytes), the interleaved activation super-block stride (1168 bytes for
    block_q8_Kx4 = 4 fp32 d + 1024 int8 quants + 64 int16 bsums), the repacked
    weight quant byte offset (qs at +256 after the 32 d/dmin fp16 + 192 scales
    bytes), the per-column weight dmin byte offset (dmin strip at +32 after the
    16 d scales), the custom 6-bit per-sub-block scales/mins region byte offset
    (scales at +64 after the 16 d + 16 dmin fp16), the interleaved activation
    quant byte offset (qs at +16 after the 4 fp32 d), the activation bsums byte
    offset (bsums at +1040 after the 4 fp32 d + 1024 quants), the sub-block count
    (8), the weight interleave width (16 columns), the activation interleave
    width (4 columns), the resource-aware strip width (half_lanes in {8,16}), and
    the super-block-d.dmin-plus-bsums-min-4col scale model -- are carried as typed
    integer/string attributes (I4 mirror only). The fixed emission is the op's
    lowering, NOT a candidate-mirror string read in the conversion. The bounded
    surface kind is limited to "ggml_repack_gemm_q4_K_q8_K" and scale_model to
    "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col". Route ids, helper
    names, ABI strings, artifact names, status fields, exact intrinsic spellings,
    operand mirror strings, q8/q4 names, and common EmitC/export code must not
    infer or choose the block-as-lane repacked GEMM semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded
    resource/scheduling fact the q4_K repacked GEVM sibling carries (the *how*,
    never the *what*; the 16-way interleaved repack reads the SAME bytes either
    way): absent / "mf2" anchors the RVV1.0 fractional chain (half_lanes in
    {8,16}), "m1" anchors the RVV0.7.1 whole-LMUL chain (half_lanes 16, ONE
    16-lane strip). The verifier pins the legal (integer_core_lmul, half_lanes)
    combinations fail-closed (I7); half_lanes must divide the 16-way interleave.
    A malformed typed body cannot lower under this emission. (NOTE: the lowering
    emitter, dispatch, recognizer, Win-A LMUL-parametric 6-bit unpack stamp, and
    lit are STAGE-1b/1c -- this op + verifier are STAGE-1a; the op does not yet
    lower. The multi-column emitter, whose 6-bit-unpack-amortized-over-M
    e2e-win-prefill hypothesis is the whole point, is STAGE-1b.)
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$row_count,
    AnyType:$column_count,
    AnyType:$output_row_stride,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$weight_dmin_byte_offset,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$activation_bsums_byte_offset,
    I64Attr:$n_subblocks,
    I64Attr:$weight_interleave,
    I64Attr:$activation_interleave,
    I64Attr:$half_lanes,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` "
    "$row_count `,` $column_count `,` $output_row_stride `,` $vl attr-dict "
    "`:` type($weight_base) `,` type($activation_base) `,` type($output) `,` "
    "type($element_count) `,` type($row_count) `,` type($column_count) `,` "
    "type($output_row_stride) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ80Q80Op
    : TCRVRVV_Op<"q8_0_q8_0_block_dot",
    [TCRVEmitCLowerableOpInterface,
     DeclareOpInterfaceMethods<TunableScheduleOpInterface>]> {
  let summary =
      "generic typed RVV block-quantized ggml Q8_0 x Q8_0 dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_q8_0_q8_0` block dot-product as ONE
    bounded generic plugin-local typed RVV dataflow step, the FAMILY-A sibling of
    tcrv_rvv.q4_0_q8_0_block_dot: the AoS QK=32 block loop, the per-block dual
    fp16 scale, the fp32 scalar accumulation, and the `*s` scalar store. The ONLY
    semantic difference from the Q4_0 sibling is the per-block integer core: BOTH
    operands here are plain block_q8_0 int8 streams (NO nibble decode, NO offset-
    binary `-8` bias, NO low/high-half split) -- the core is a single 32-element
    signed widening product (vle8 x2 -> vwmul_vv i8->i16 -> vwredsum -> i32 sumi).
    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the q8_0 lhs base pointer (vx), the q8_0 rhs base pointer
    (vy), the fp32 output pointer (s), the runtime element count (n) -- plus the
    active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1 !tcrv_rvv.vector token
    whose lowering writes the single fp32 reduction result through the output
    pointer.

    The block-quantized dot-product STRUCTURE is first-class in the typed body.
    The structural facts -- QK=32, the AoS block strides (34 bytes for block_q8_0
    on BOTH operands), the quant payload byte offset (+2), and the per-block dual
    fp16 scale model (d_x . d_y) -- are carried as typed integer/string attributes
    (I4 mirror only). The fixed emission -- an outer emitc.for block loop over
    nb = n / QK, per-block address arithmetic (vx + ib*34, vy + ib*34) via
    emitc.add/emitc.mul nodes, the two scalar fp16->fp32 reads via
    emitc.call_opaque (the one sanctioned opaque piece), an inner strip loop that
    runs the plain signed i8 x i8 widening product/reduce chain (vle8 -> vwmul_vv
    -> vwredsum) over the WHOLE 32-element block into a per-block i32 scalar, and
    the fp32 accumulate `sumf = sumf + (float)sumi * (d_x * d_y)` (ggml's exact
    q8_0 order -- the scales are multiplied FIRST, distinct from the Q4_0 sibling's
    `(sumi*d_x)*d_y`) -- is the op's lowering, NOT a candidate-mirror string read
    in the conversion. For the bounded surface kind is limited to
    "ggml_q8_0_q8_0_block_dot" and scale_model to "dual-fp16-per-block-d_x.d_y".

    The optional `integer_core_lmul`, `multi_block_factor`, and `strip_elision`
    attributes are the SAME bounded resource/scheduling shape knobs the Q4_0
    sibling carries (the *how*, never the *what*; the dot-product is byte-exact for
    any legal combination because integer addition is order-independent and the
    per-block fp32 fold stays in strict ascending block order). The q8_0 block is
    32 CONTIGUOUS int8 (not the Q4_0 sibling's 16-lane nibble-packed half-block),
    so its anchors shift one LMUL step UP:
    - `integer_core_lmul` selects the i8 strip anchor: "m2" (the whole 32-element
      block in ONE strip + ONE vwredsum at VLEN=128, matching ggml's
      i8m2->i16m4->vwredsum hand-written reduction anchor), "m1" (two 16-element
      strips), or "mf4" (eight 4-element strips). All three are byte-exact (the
      same integer set; integer add is order-independent) and keep a VLEN-robust
      strip loop carrying sumi across strips.
    - `multi_block_factor` selects how many q8_0 blocks the outer loop processes
      per iteration -- 1 (default), 2, or 4 -- unrolling several independent
      per-block integer cores so they overlap; the loop processes nb / factor full
      groups then a scalar `nb % factor` tail of robust single blocks, with the
      fp32 folds always emitted in strict ascending block order.
    - `strip_elision` selects whether the inner strip loop is kept ("robust",
      default -- correct at any VLEN) or elided ("elided" -- a single
      vsetvl_e8m2(32) + one vwredsum over the whole block, NO inner strip loop).
      The elided form is correct ONLY at VLEN >= 128 (vsetvl_e8m2(32) caps the
      active vl at 32 when VLMAX is >= 32) and therefore requires the m2 anchor
      (m1/mf4 VLMAX at VLEN=128 would silently drop block bytes). The verifier
      rejects the "elided" + non-m2 combination fail-closed (I7); the capability-
      aware autotuner gates "elided" on the Zvl128b capability. The `nb % factor`
      tail always uses the robust single-block core regardless of strip_elision.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$quant_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$multi_block_factor,
    OptionalAttr<StrAttr>:$strip_elision
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ41Q81Op
    : TCRVRVV_Op<"q4_1_q8_1_block_dot",
    [TCRVEmitCLowerableOpInterface,
     DeclareOpInterfaceMethods<TunableScheduleOpInterface>]> {
  let summary =
      "generic typed RVV block-quantized ggml Q4_1 x Q8_1 dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_q4_1_q8_1` block dot-product as ONE
    bounded generic plugin-local typed RVV dataflow step, the FAMILY-B sibling of
    tcrv_rvv.q4_0_q8_0_block_dot (scale+MIN, asymmetric): the AoS QK=32 block
    loop, the per-block dual fp16 scale, the per-block MIN/SUM correction term,
    the fp32 scalar accumulation, and the `*s` scalar store. Two semantic
    differences from the Q4_0 sibling distinguish Family B:

    (1) The integer core decodes UNSIGNED nibbles [0,15] -- NO offset-binary `-8`
        bias and NO XOR-0x88 (the q4_1 quantization is asymmetric, the bias lives
        in the separate per-block MIN scale). The decode is `vand 0x0F` (low
        nibble) / `vsrl 0x04` (high nibble, a LOGICAL shift) on the UNSIGNED
        weight lane, then a value-identity reinterpret to i8, then the SAME signed
        widening product against the plain-i8 q8_1 halves (vwmul/vwmacc) ->
        vwredsum -> i32 `sumi`. This is the unsigned-nibble variant of the INC-1
        offset-binary core, added CLEANLY alongside it (the offset-binary core is
        byte-untouched).

    (2) The per-block fold carries a SECOND scale on EACH operand. block_q4_1 has
        {fp16 d; fp16 m; uint8 qs[16]} and block_q8_1 has {fp16 d; fp16 s; int8
        qs[32]}, where `m` is the per-block MIN and `s` is the PRECOMPUTED scaled
        activation sum (`d_y * sum(q8)`). The fold is ggml's exact statement
        `sumf += (d_x * d_y) * sumi + m_x * s_y` (quants.c) -- the (d_x*d_y) is
        multiplied FIRST (q8_0's order, NOT the Q4_0 sibling's (sumi*d_x)*d_y),
        the two products (d_x*d_y)*sumi and m_x*s_y are SUMMED FIRST (the `+`
        binds before the `+=`), then added to sumf: `sumf + ((d_x*d_y)*sumi +
        m_x*s_y)` -- an fp-significant grouping. Both halves are read per block:
        d_x at weight+0 and m_x at weight+2 (the +`weight_min_byte_offset`); d_y
        at activation+0 and s_y at activation+2 (the +`activation_sum_byte_offset`).

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the q4_1 weight base pointer (vx), the q8_1 activation base
    pointer (vy), the fp32 output pointer (s), the runtime element count (n) --
    plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the single fp32 reduction result
    through the output pointer.

    The block-quantized dot-product STRUCTURE is first-class in the typed body.
    The structural facts -- QK=32, the AoS block strides (20 bytes for block_q4_1,
    36 bytes for block_q8_1), the quant payload byte offset (+4, after the two
    inline fp16 scales), the q8 high-half byte offset (+16), the second-scale byte
    offsets (weight m at +2, activation s at +2), and the dual-fp16-plus-min scale
    model -- are carried as typed integer/string attributes (I4 mirror only). The
    fixed emission -- an outer emitc.for block loop over nb = n / QK, per-block
    address arithmetic (vx + ib*20, vy + ib*36) via emitc.add/emitc.mul nodes, the
    FOUR scalar fp16->fp32 reads (d_x,m_x,d_y,s_y) via emitc.call_opaque (the one
    sanctioned opaque piece, exactly as INC-1 emits its intrinsics), an inner
    strip loop reusing the unsigned-nibble asymmetric i4xi8 decode/product chain
    (vand 0x0F / vsrl 0x04 -> reinterpret -> vwmul/vwmacc against the plain q8
    halves) + vwredsum into a per-block i32 scalar, and the fp32 accumulate
    `sumf = sumf + (d_x*d_y)*sumi + m_x*s_y` (ggml's exact order) -- is the op's
    lowering, NOT a candidate-mirror string read in the conversion. For the
    bounded surface kind is limited to "ggml_q4_1_q8_1_block_dot" and scale_model
    to "dual-fp16-per-block-d_x.d_y-plus-min". Route ids, helper names, ABI
    strings, artifact names, status fields, exact intrinsic spellings, operand
    mirror strings, q8/q4 names, and common EmitC/export code must not infer or
    choose block-quantized dot-product semantics.

    The optional `integer_core_lmul`, `multi_block_factor`, and `strip_elision`
    attributes are the SAME bounded resource/scheduling shape knobs the Q4_0
    sibling carries (the *how*, never the *what*; the dot-product is byte-exact for
    any legal combination because integer addition is order-independent and the
    per-block fp32 fold stays in strict ascending block order). Because q4_1's
    nibble half-block is byte-identical in shape to q4_0's (16 nibble bytes,
    decoded into 2x16 lanes), the anchors are q4_0's, NOT q8_0's:
    - `integer_core_lmul` selects the i8 strip anchor: "m1" (i8m1 -> i16m2, ONE
      strip + ONE vwredsum per half-block at VLEN=128, matching ggml's reduction
      anchor) or "mf4" (the default: i8mf4 -> i16mf2, vsetvl_e32m1, 4 chunks).
    - `multi_block_factor` selects how many q4_1/q8_1 blocks the outer loop
      processes per iteration -- 1 (default), 2, or 4.
    - `strip_elision` selects "robust" (default, correct at any VLEN) or "elided"
      (a single vsetvl_e8m1(16) + one vwredsum per half-block, correct ONLY at
      VLEN >= 128 and therefore requiring the m1 anchor). The verifier rejects the
      "elided" + non-m1 combination fail-closed (I7); the autotuner gates "elided"
      on the Zvl128b capability. The `nb % factor` tail always uses the robust
      single-block core regardless of strip_elision.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$quant_byte_offset,
    I64Attr:$activation_high_byte_offset,
    I64Attr:$weight_min_byte_offset,
    I64Attr:$activation_sum_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$multi_block_factor,
    OptionalAttr<StrAttr>:$strip_elision
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ50Q80Op
    : TCRVRVV_Op<"q5_0_q8_0_block_dot",
    [TCRVEmitCLowerableOpInterface,
     DeclareOpInterfaceMethods<TunableScheduleOpInterface>]> {
  let summary =
      "generic typed RVV block-quantized ggml Q5_0 x Q8_0 dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_q5_0_q8_0` block dot-product as ONE
    bounded generic plugin-local typed RVV dataflow step, the FAMILY-A sibling of
    tcrv_rvv.q4_0_q8_0_block_dot (single-scale 5-bit weight): the AoS QK=32 block
    loop, the per-block dual fp16 scale, the fp32 scalar accumulation, and the
    `*s` scalar store. It shares the Q4_0/Q4_1 nibble half-block STRUCTURE (16
    nibble bytes decoded into 2x16 lanes, paired against q8[0..15]/q8[16..31]) and
    q8_0's scales-first fold order. The ONE genuinely-new piece is the 5-bit weight
    reconstruction (block_q5_0 carries a 4-bit nibble PLUS a separate per-element
    5th high bit packed in a 32-bit qh field):

    (1) The integer core decodes the low 4 bits UNSIGNED [0,15] (the q4_1 path:
        `vand 0x0F` low nibble / `vsrl 0x04` high nibble on the UNSIGNED weight
        lane), then INJECTS the 5th high bit per element from the 32-bit qh field
        and applies the offset-binary `-16` bias: for element i the value is
        `(int8_t)(((qs_nibble) | (qh_bit_i << 4)) - 16)`, range [-16,15]. The 5th
        bit of element i is `bit i` of qh; for the low half (elements 0..15) that
        is `bit (c+lane)` of `qh & 0xFFFF`, for the high half (elements 16..31)
        that is `bit (c+lane)` of `qh >> 16`. The injection is a per-lane shift
        vector (`vid + c`), `vsrl_vv` of the broadcast 16-bit qh half, `& 1`,
        `<< 4`, narrowed u16->u8, OR'd into the unsigned nibble, then reinterpret
        to i8 and `vsub 16`. The decoded i8 lanes feed the SAME signed widening
        product against the plain-i8 q8 halves (vwmul/vwmacc) -> vwredsum -> i32
        `sumi`. This is the 5-bit variant of the INC-1/q4_1 nibble core, added
        CLEANLY alongside them (the offset-binary and unsigned cores are
        byte-untouched).

    (2) The per-block fold is ggml's q5_0 order `sumf = sumf + (d_x*d_y)*sumi`
        (the dual fp16 scales multiplied FIRST, q8_0's order -- NOT the Q4_0
        sibling's `(sumi*d_x)*d_y`), grouped into ONE emitc.expression so the
        compiler fuses the SAME FMA ggml does under -ffp-contract on/default
        (byte-exact across all four modes). block_q5_0 is {fp16 d; uint8 qh[4];
        uint8 qs[16]} (stride 22): d at +0, the qh high-bit field at +2, the 16
        nibble bytes at +6; block_q8_0 is unchanged (stride 34, qs at +2, high
        half at +16).

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the q5_0 weight base pointer (vx), the q8_0 activation base
    pointer (vy), the fp32 output pointer (s), the runtime element count (n) --
    plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the single fp32 reduction result
    through the output pointer.

    The block-quantized dot-product STRUCTURE is first-class in the typed body.
    The structural facts -- QK=32, the AoS block strides (22 bytes for block_q5_0,
    34 bytes for block_q8_0), the WEIGHT quant payload byte offset (+6, after the
    inline fp16 scale and the 4-byte qh field), the ACTIVATION quant payload byte
    offset (+2, after the q8_0 inline fp16 scale -- DISTINCT from the weight's +6
    because q5_0 inserts the 4-byte qh between the scale and the nibbles while the
    q8_0 activation is unchanged), the q8 high-half byte offset (+16), the qh
    high-bit field byte offset (+2), and the dual-fp16 scale model -- are carried
    as typed integer/string attributes (I4 mirror only). The fixed
    emission -- an outer emitc.for block loop over nb = n / QK, per-block address
    arithmetic (vx + ib*22, vy + ib*34) via emitc.add/emitc.mul nodes, the two
    scalar fp16->fp32 reads plus the two aligned 16-bit qh half reads via
    emitc.call_opaque
    (the sanctioned opaque pieces, exactly as INC-1 emits its intrinsics), an inner
    strip loop reusing the unsigned-nibble decode + 5th-bit injection + offset-
    binary `-16` chain (-> vwmul/vwmacc against the plain q8 halves) + vwredsum
    into a per-block i32 scalar, and the fp32 accumulate `sumf = sumf +
    (d_x*d_y)*sumi` (ggml's exact order) -- is the op's lowering, NOT a candidate-
    mirror string read in the conversion. For the bounded surface kind is limited
    to "ggml_q5_0_q8_0_block_dot" and scale_model to "dual-fp16-per-block-d_x.d_y".
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/q5 names, and common
    EmitC/export code must not infer or choose block-quantized dot-product
    semantics.

    The optional `integer_core_lmul`, `multi_block_factor`, and `strip_elision`
    attributes are the SAME bounded resource/scheduling shape knobs the Q4_0/Q4_1
    siblings carry (the *how*, never the *what*; the dot-product is byte-exact for
    any legal combination because integer addition is order-independent and the
    per-block fp32 fold stays in strict ascending block order). Because q5_0's
    nibble half-block is byte-identical in shape to q4_0's (16 nibble bytes,
    decoded into 2x16 lanes), the anchors are q4_0's, NOT q8_0's:
    - `integer_core_lmul` selects the i8 strip anchor: "m1" (i8m1 -> i16m2, ONE
      strip + ONE vwredsum per half-block at VLEN=128, matching ggml's reduction
      anchor) or "mf4" (the default: i8mf4 -> i16mf2, vsetvl_e32m1, 4 chunks).
    - `multi_block_factor` selects how many q5_0/q8_0 blocks the outer loop
      processes per iteration -- 1 (default), 2, or 4.
    - `strip_elision` selects "robust" (default, correct at any VLEN) or "elided"
      (a single vsetvl_e8m1(16) + one vwredsum per half-block, correct ONLY at
      VLEN >= 128 and therefore requiring the m1 anchor). The verifier rejects the
      "elided" + non-m1 combination fail-closed (I7); the autotuner gates "elided"
      on the Zvl128b capability. The `nb % factor` tail always uses the robust
      single-block core regardless of strip_elision.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$activation_high_byte_offset,
    I64Attr:$weight_qh_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$multi_block_factor,
    OptionalAttr<StrAttr>:$strip_elision
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ4NLQ80Op
    : TCRVRVV_Op<"iq4_nl_q8_0_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV block-quantized ggml IQ4_NL x Q8_0 dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_iq4_nl_q8_0` block dot-product as ONE
    bounded generic plugin-local typed RVV dataflow step -- the FAMILY-A sibling
    that opens the CODEBOOK quantization class. It shares the AoS QK=32 block loop,
    the per-block dual fp16 scale, the fp32 scalar accumulation, and the `*s` store
    with tcrv_rvv.q4_0_q8_0_block_dot, AND its block byte layout is byte-IDENTICAL
    to block_q4_0: block_iq4_nl is {fp16 d; uint8 qs[16]} (stride 18), the 16 packed
    nibble bytes at +2, the low nibble of byte j paired with q8[j] and the high
    nibble with q8[j+16] (the q8_0 activation is unchanged: stride 34, qs at +2, the
    high half at +16). The ONE genuinely-new piece -- the mechanism that opens the
    codebook class -- is that the 4-bit nibble does NOT decode to a linear value
    (`nibble - 8` as in q4_0) but INDEXES a 16-entry NON-LINEAR INT8 CODEBOOK
    (ggml's `kvalues_iq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,
    69,89,113}`): the per-element weight value is `kvalues_iq4nl[nibble]`, a
    deterministic int8->int8 lookup, NOT an arithmetic decode.

    (1) The integer core unpacks the two UNSIGNED nibble-index lanes (`vand 0x0F`
        low nibble / `vsrl 0x04` high nibble on the UNSIGNED weight lane -> two u8
        index lanes in [0,15]) then GATHERS each lane through the broadcast codebook
        table via `vrgather_vv_i8m1(values, idx)` (ggml's exact RVV method, the same
        `__riscv_vrgather_vv_i8...` the board kernel uses). `values` is the 16-entry
        signed-int8 codebook broadcast into a vector register ONCE above the block
        loop via `vle8_v_i8m1(kvalues_iq4nl, 16)`. The two gathered signed-i8 weight
        lanes feed the SAME asymmetric signed widening product against the plain-i8
        q8 halves (vwmul low <-> q8[0..15], vwmacc + high <-> q8[16..31]) ->
        vwredsum -> i32 `sumi` the q4_0 sibling uses (the gather REPLACES only the
        offset-binary decode; the product/reduce chain is shared verbatim). This is
        the codebook variant of the nibble core, added CLEANLY alongside the
        offset-binary (q4_0), unsigned (q4_1), and 5-bit (q5_0) cores (all
        byte-untouched).

    (2) The per-block fold is ggml's iq4_nl order `sumf = sumf + (float)sumi *
        (d_x * d_y)` (the two fp16 scales multiplied FIRST, the q8_0/q5_0 scales-
        first order -- NOT the Q4_0 sibling's `((sumi*d_x)*d_y)`; ggml's _generic is
        `d * (sumi1 + sumi2)` with `d = d_y * d_x`). It is grouped into ONE
        emitc.expression so the compiler fuses the SAME FMA ggml does under
        -ffp-contract on/default (byte-exact across all four modes).

    The CODEBOOK GATHER is inherently a Zvl128b legality fact: to index ALL 16 table
    entries the `values` register's VLMAX must be >= 16, so the gather pins the m1
    integer-core anchor (at VLEN=128 e8: m1 -> VLMAX=16 = the whole half-block, mf4
    -> VLMAX=4 would silently return 0 for any nibble index >= 4). The verifier
    rejects a non-m1 anchor fail-closed (I7). The m1 anchor holds exactly the 16-lane
    half-block in one `vsetvl_e8m1(16)`, so the codebook class is the loop-free
    ("elided") strip form, correct for ANY VLEN >= 128 (vsetvl_e8m1(16) caps the
    active vl at 16 when VLMAX >= 16) -- apples-to-apples with ggml's own VLEN >= 128
    assumption. A board with VLEN < 128 (zve32x/zve64x) cannot legally host the
    codebook gather at this anchor; that is an N1 capability-legality divergence, not
    a coverage gap.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the iq4_nl weight base pointer (vx), the q8_0 activation base
    pointer (vy), the fp32 output pointer (s), the runtime element count (n) -- plus
    the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1 !tcrv_rvv.vector token
    whose lowering writes the single fp32 reduction result through the output
    pointer.

    The block-quantized dot-product STRUCTURE is first-class in the typed body. The
    structural facts -- QK=32, the AoS block strides (18 bytes for block_iq4_nl, 34
    bytes for block_q8_0), the quant payload byte offset (+2), the q8 high-half byte
    offset (+16), the dual-fp16 scale model, and the 16-entry non-linear int8
    CODEBOOK itself -- are carried as typed integer/string/array attributes (I4
    mirror only). The codebook is a `DenseI8ArrayAttr:$codebook` of EXACTLY 16 int8
    entries; it is a STRUCTURAL FACT (like the strides/offsets), consumed
    structurally into the realized body (emitted as a `static const int8_t[16]` decl
    + the `vle8` table load), NOT an arithmetic decode, NOT a string plan read. The
    op identity is the dispatch key. Route ids, helper names, ABI strings, artifact
    names, status fields, exact intrinsic spellings, operand mirror strings, q8/iq4
    names, and common EmitC/export code must not infer or choose block-quantized
    dot-product semantics.

    The optional `integer_core_lmul`, `multi_block_factor`, and `strip_elision`
    attributes are the SAME bounded resource/scheduling shape knobs the Family-A
    siblings carry (the *how*, never the *what*; the dot-product is byte-exact for
    any legal combination because integer addition is order-independent and the
    per-block fp32 fold stays in strict ascending block order). The codebook gather
    constrains integer_core_lmul to "m1" only (the VLMAX >= 16 table-index legality
    fact above); the verifier rejects any other anchor fail-closed (I7).
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$quant_byte_offset,
    I64Attr:$activation_high_byte_offset,
    DenseI8ArrayAttr:$codebook,
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$multi_block_factor,
    OptionalAttr<StrAttr>:$strip_elision
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotMXFP4Q80Op
    : TCRVRVV_Op<"mxfp4_q8_0_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV block-quantized ggml MXFP4 x Q8_0 dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_mxfp4_q8_0` block dot-product as ONE
    bounded generic plugin-local typed RVV dataflow step -- the CODEBOOK-class
    sibling that opens the LAST structural quantization class: FP4 (e2m1) weights
    with an E8M0 shared-exponent block scale. It shares the AoS QK=32 block loop,
    the codebook-gather integer core, the fp32 scalar accumulation, and the `*s`
    store with tcrv_rvv.iq4_nl_q8_0_block_dot, reusing the SAME 16-entry non-linear
    int8 CODEBOOK gather mechanism (vand 0x0F / vsrl 0x04 nibble split ->
    vrgather_vv_i8m1 through a broadcast `values` table). TWO facts distinguish
    mxfp4 from iq4_nl, both load-bearing:

    (1) BLOCK FORMAT. block_mxfp4 is `{ uint8_t e; uint8_t qs[16] }` (sizeof 17),
        NOT `{ fp16 d; uint8_t qs[16] }`: the weight block carries a SINGLE 8-bit
        E8M0 shared exponent `e` at byte +0, then the 16 packed FP4 nibble bytes
        at byte +1. So the weight block stride is 17 and the quant payload offset
        is +1 (not 18 / +2 as for iq4_nl). The q8_0 activation is unchanged
        (stride 34, quants at +2, the high half at +16).

    (2) WEIGHT SCALE. The per-block weight scale is NOT a fp16 read. It is the
        structured E8M0 -> fp32 reconstruction `GGML_E8M0_TO_FP32_HALF(e)` =
        `2^(e-128)` (the HALF form: ggml's `kvalues_mxfp4` codebook is the DOUBLED
        int8 e2m1 set `{0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}` = 2*E2M1, so the
        block scale is halved to compensate, keeping the dot an exact INTEGER dot).
        It is reconstructed by ggml's EXACT bit-construction (ggml-impl.h): a
        uint32_t `bits = (e < 2) ? (0x00200000u << e) : ((e - 1) << 23)` reinterpreted
        as a float -- the e<2 branch yields the denormal scales 2^-128 / 2^-127, the
        e>=2 branch the normalized 2^(e-128). This is emitted STRUCTURALLY (emitc
        cmp / bitwise_left_shift / sub / conditional + a `*(const float *)&bits`
        pointer pun), byte-identical to ggml by construction (no scalbnf/ldexpf,
        which risks FTZ/fast-math divergence on the denormal scales). The codebook
        entries are NOT pinned by the verifier (a structural input the gather
        realizes); the scale_model string IS pinned (a wrong scale convention --
        e.g. the full 2^(e-127) instead of the half form -- is rejected fail-closed).

    The integer core then feeds the SAME asymmetric signed widening product against
    the plain-i8 q8 halves (vwmul low <-> q8[0..15], vwmacc + high <-> q8[16..31])
    -> vwredsum -> i32 `sumi` the iq4_nl/q4_0 siblings use (the gather + the q8
    halves are byte-untouched; only the weight scale read changes). The per-block
    fold is ggml's mxfp4 order `sumf = sumf + (float)sumi * (scale_x * d_y)`
    (_generic is `d * (sumi1 + sumi2)` with `d = d_y * scale_x`; fp multiply
    commutes so the SAME fold node is byte-exact). Only ONE fp16 read survives (the
    q8_0 `d_y`); the weight `d_x` read is replaced by the E8M0 reconstruction.

    The CODEBOOK GATHER is inherently a Zvl128b legality fact: to index ALL 16 table
    entries the `values` register's VLMAX must be >= 16, so the gather pins the m1
    integer-core anchor (at VLEN=128 e8: m1 -> VLMAX=16 = the whole half-block). The
    verifier rejects a non-m1 anchor fail-closed (I7). A board with VLEN < 128
    (zve32x/zve64x) cannot legally host the codebook gather at this anchor; that is
    an N1 capability-legality divergence, not a coverage gap.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the mxfp4 weight base pointer (vx), the q8_0 activation base
    pointer (vy), the fp32 output pointer (s), the runtime element count (n) -- plus
    the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1 !tcrv_rvv.vector token
    whose lowering writes the single fp32 reduction result through the output
    pointer.

    The structural facts -- QK=32, the AoS block strides (17 bytes for block_mxfp4,
    34 bytes for block_q8_0), the weight quant payload byte offset (+1), the q8
    quant byte offset (+2), the q8 high-half byte offset (+16), the E8M0 weight
    scale model, and the 16-entry non-linear int8 CODEBOOK itself -- are carried as
    typed integer/string/array attributes (I4 mirror only). The codebook is a
    `DenseI8ArrayAttr:$codebook` of EXACTLY 16 int8 entries, consumed structurally
    into the realized body (a `static const int8_t[16]` decl + the `vle8` table
    load). The op identity is the dispatch key. Route ids, helper names, ABI
    strings, artifact names, status fields, exact intrinsic spellings, operand
    mirror strings, q8/mxfp4 names, and common EmitC/export code must not infer or
    choose block-quantized dot-product semantics.

    The optional `integer_core_lmul`, `multi_block_factor`, and `strip_elision`
    attributes are the SAME bounded resource/scheduling shape knobs the codebook
    siblings carry (the *how*, never the *what*; the dot-product is byte-exact for
    any legal combination because integer addition is order-independent and the
    per-block fp32 fold stays in strict ascending block order). The codebook gather
    constrains integer_core_lmul to "m1" only; the verifier rejects any other anchor
    fail-closed (I7).
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$activation_high_byte_offset,
    DenseI8ArrayAttr:$codebook,
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$multi_block_factor,
    OptionalAttr<StrAttr>:$strip_elision
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotNVFP4Q80Op
    : TCRVRVV_Op<"nvfp4_q8_0_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV block-quantized ggml NVFP4 x Q8_0 dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_nvfp4_q8_0` block dot-product as ONE
    bounded generic plugin-local typed RVV dataflow step -- the SECOND FP4-class
    sibling. NVFP4 is NVIDIA's FP4: it REUSES mxfp4's FP4 (e2m1) `kvalues_mxfp4`
    codebook gather + the asymmetric signed-widening integer core verbatim; the
    genuinely-new fact is a DIFFERENT per-sub-block scale -- a UE4M3 (unsigned 4
    exp / 3 mantissa) fp8 scale instead of mxfp4's E8M0 shared exponent -- carried
    on a SUPER-BLOCK that groups four such sub-blocks. THREE facts distinguish
    nvfp4 from mxfp4, all load-bearing:

    (1) SUPER-BLOCK FORMAT. block_nvfp4 is `{ uint8_t d[4]; uint8_t qs[32] }`
        (QK_NVFP4 = 64, sizeof 36), NOT mxfp4's flat `{ uint8_t e; uint8_t qs[16] }`
        QK=32 block. ONE nvfp4 super-block is FOUR 16-element sub-blocks: byte +0..3
        carry the four UE4M3 sub-block scales `d[s_idx]`; byte +4..35 carry the 32
        packed FP4 nibble bytes (8 per sub-block, sub-block s_idx at +4 + s_idx*8).
        The 64 elements span TWO block_q8_0 activation blocks (each 32 elements):
        sub-block s_idx reads q8 block `2*ib + s_idx/2` at the half-offset
        `(s_idx%2)*16` within that q8_0's 32 quants. So the q8 activation scale
        `d_y` changes at s_idx==2, and the per-sub-block strip is 8 lanes
        (QK_NVFP4_SUB/2), the high nibble pairing at +8 (not the 16-lane / +16 of
        mxfp4). The recognizer/emitter walk the four sub-blocks per super-block
        explicitly; the OUTER loop is over super-blocks nb = n / 64.

    (2) WEIGHT SCALE. The per-sub-block weight scale is NOT mxfp4's E8M0 bit dance.
        It is the structured UE4M3 -> fp32 reconstruction `ggml_ue4m3_to_fp32(e)`
        (ggml-impl.h): UE4M3 is UNSIGNED (4 exp bits bias=7, 3 mantissa bits, NO
        sign bit). `e==0 || e==0x7F -> 0.0f` (the two specials); else
        `exp=(e>>3)&0xF, man=e&7`, and `raw = (exp==0) ? ldexpf(man, -9) :
        ldexpf(1 + man/8, exp-7)`, then `* 0.5f` (the HALF compensation for the
        DOUBLED `kvalues_mxfp4` codebook, exactly as mxfp4's E8M0 half form keeps
        the dot an exact INTEGER dot). Unlike mxfp4's E8M0 (whose scales reach the
        FTZ-risky 2^-128/2^-127 denormals, forcing the bit-construction dodge),
        every UE4M3 scale lies in comfortably-normal fp32 (~[2^-10, 224]), so the
        emission replicates ggml's EXACT arithmetic structurally (an emitc.cmp pair
        for the two specials -> 0.0f; a structured exp/man split + the two ldexpf
        branches + the *0.5f), byte-identical to `_generic` by construction (same
        ops, same libm `ldexpf`, same board). The `scale_model` string IS pinned (a
        wrong scale convention -- e.g. dropping the *0.5f, or the full E4M3 signed
        decode -- is rejected fail-closed, I7).

    (3) ACTIVATION ADDRESSING. The activation is the SAME block_q8_0 stream, but the
        super-block consumes TWO q8_0 blocks. The op carries the q8_0 block stride
        (34) and the per-sub-block q8 quant byte offset (+2) + high-half offset (+8)
        as typed facts; the q8-block-index `2*ib + s_idx/2` and the sub-block
        half-offset `(s_idx%2)*16` are STRUCTURAL (derived in the emitter from the
        sub-block index), not extra attrs.

    The integer core then feeds the SAME asymmetric signed widening product the
    mxfp4/iq4_nl siblings use (vand 0x0F / vsrl 0x04 nibble split -> vrgather_vv_i8m1
    through the broadcast `kvalues_mxfp4` table -> vwmul low <-> q8 lo half, vwmacc +
    high <-> q8 hi half -> vwredsum -> i32 `sumi`). The per-sub-block fold is ggml's
    nvfp4 order `sumf = sumf + d_y * d_x * (float)sumi` (_generic is `dy * d *
    (sumi_lo + sumi_hi)`; fp multiply commutes), in STRICT ascending (ib, s_idx)
    order so fp non-associativity is byte-exact. The q8_0 `d_y` is the ONE surviving
    fp16 read per sub-block; the weight `d_x` read is the structured UE4M3 decode.

    The CODEBOOK GATHER is the same Zvl128b legality fact as mxfp4: to index ALL 16
    table entries the `values` register's VLMAX must be >= 16, so the gather pins the
    m1 integer-core anchor. The verifier rejects a non-m1 anchor fail-closed (I7). A
    board with VLEN < 128 (zve32x/zve64x) cannot legally host the codebook gather at
    this anchor; that is an N1 capability-legality divergence, not a coverage gap.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the nvfp4 weight base pointer (vx), the q8_0 activation base
    pointer (vy), the fp32 output pointer (s), the runtime element count (n) -- plus
    the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1 !tcrv_rvv.vector token
    whose lowering writes the single fp32 reduction result through the output
    pointer.

    The structural facts -- qk (64), qk_sub (16), the super-block stride (36 bytes),
    the q8_0 block stride (34), the weight quant payload byte offset (+4), the q8
    quant byte offset (+2), the per-sub-block q8 high-half byte offset (+8), the
    UE4M3 weight scale model, and the 16-entry non-linear int8 CODEBOOK itself -- are
    carried as typed integer/string/array attributes (I4 mirror only). The codebook
    is a `DenseI8ArrayAttr:$codebook` of EXACTLY 16 int8 entries, consumed
    structurally into the realized body (a `static const int8_t[16]` decl + the
    `vle8` table load). The op identity is the dispatch key. Route ids, helper names,
    ABI strings, artifact names, status fields, exact intrinsic spellings, operand
    mirror strings, q8/nvfp4 names, and common EmitC/export code must not infer or
    choose block-quantized dot-product semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded resource shape
    knob the codebook siblings carry (the *how*, never the *what*; the dot-product is
    byte-exact for any legal value because integer addition is order-independent and
    the per-sub-block fp32 fold stays in strict ascending order). The codebook gather
    constrains it to "m1" only; the verifier rejects any other anchor fail-closed
    (I7).
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$qk_sub,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$activation_high_byte_offset,
    DenseI8ArrayAttr:$codebook,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ10Q80Op
    : TCRVRVV_Op<"q1_0_q8_0_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV block-quantized ggml Q1_0 x Q8_0 dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_q1_0_q8_0` block dot-product as ONE
    bounded generic plugin-local typed RVV dataflow step -- the BINARY ({-1,+1})
    class, one of the last three uncommon ggml dot kernels. The q1_0 weight is a
    1-bit-per-element sign: a set bit decodes to `+q8`, an unset bit to `-q8`. The
    genuinely-new mechanism vs every prior sibling is the per-lane BINARY SIGN
    DECODE -- a `vmerge(vneg(q8), q8, bit_mask)` that selects `+q8` where the
    weight bit is set and `-q8` where it is clear -- reusing the iq2_xxs
    kmask/vmsne sign-plane primitive (the SAME `{1,2,4,8,...}` bit-selector +
    `vmsne != 0` -> mask), but with NO codebook, NO nibble unpack, NO offset-binary
    `-8` bias: the q8 value IS the magnitude and the weight bit IS the sign.

    THREE structural facts distinguish q1_0, all load-bearing:

    (1) SUPER-BLOCK FORMAT. block_q1_0 is `{ ggml_half d; uint8_t qs[16] }`
        (QK1_0 = 128, sizeof 18), the fp16 scale `d0` at byte +0, then 16 packed
        BIT bytes (128 bits = 128 element signs) at byte +2. ONE q1_0 super-block
        is 128 elements spanning FOUR block_q8_0 activation blocks (each 32
        elements): sub-block k (k=0..3) reads q8 block `4*ib + k` (the `*4` span is
        the genuinely-new addressing vs nvfp4's `*2`), and the 32 bits of sub-block
        k live at weight bytes `+2 + k*4 .. +2 + k*4 + 3` (4 bytes = 32 bits). The
        recognizer/emitter walk the four sub-blocks per super-block explicitly; the
        OUTER loop is over super-blocks nb = n / 128.

    (2) BINARY SIGN DECODE. Within sub-block k the 32 bits map to the 32 q8 lanes
        in FOUR groups of 8: bit `bits[k*4 + b]` bit `m` (m=0..7) pairs with q8
        lane `b*8 + m`, value `(bit & (1<<m)) ? +q8[b*8+m] : -q8[b*8+m]` -- ggml's
        EXACT `quants.c` ordering. The emitter decodes each 8-lane group:
        broadcast the scalar bits byte into 8 lanes (`vmv_v_x_u8`), `vand` the
        `{1,2,4,8,16,32,64,128}` bit-selector (the iq2_xxs kmask, emitted ONCE as a
        `static const uint8_t[8]` decl + `vle8` load above the block loop),
        `vmsne != 0` -> the per-lane sign mask. The q8 group is WIDENED to i16
        FIRST (`vwcvt_x_x_v_i16m2`), then `vmerge(vneg_i16(q8w), q8w, mask)` applies
        the sign IN THE i16 DOMAIN -- ggml's `_generic` negates in the int domain,
        so `-q8` of the boundary value -128 is +128, which does NOT fit i8 (an i8
        `vneg` of -128 overflows back to -128); performing the negate/merge after
        the widening keeps +128 representable, so the decode is byte-exact over the
        FULL int8 range, not only ggml's [-127,127] q8 quantization domain. The
        signed i16 group is reduced by `vwredsum` into the per-sub-block integer
        `sumi_block`. The 8-lane groups keep the core VLEN-robust (8 <= e8m1 VLMAX
        at any VLEN >= 64; rv64gcv guarantees >= 128).

    (3) TWO-LEVEL fp32 FOLD. ggml accumulates `sumf += d0 * Σ_k(d1_k * sumi_block_k)`
        with d0 = fp16(x.d) and d1_k = fp16(y[4*ib+k].d). The emitter carries a
        per-super-block `float sumi = 0.0f`, folds each sub-block as
        `sumi = sumi + d1_k * (float)sumi_block` (in strict ascending k order), then
        the super-block term `sumf = sumf + d0 * sumi` -- ggml's EXACT order, so fp
        non-associativity is byte-exact. There are FIVE fp16 reads per super-block
        (d0 once, d1_k for k=0..3); they are the ONLY sanctioned opaque scalar
        pieces (the same `(float)*(const _Float16 *)` node the q4_0 sibling emits).

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the q1_0 weight base pointer (vx), the q8_0 activation base
    pointer (vy), the fp32 output pointer (s), the runtime element count (n) --
    plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the single fp32 reduction result
    through the output pointer.

    The structural facts -- qk (128), the super-block stride (18 bytes), the q8_0
    block stride (34), the per-super-block q8-block span (4), the weight bit-payload
    byte offset (+2), the q8 quant byte offset (+2), and the binary sign scale model
    -- are carried as typed integer/string attributes (I4 mirror only). The op
    identity is the dispatch key. Route ids, helper names, ABI strings, artifact
    names, status fields, exact intrinsic spellings, operand mirror strings, q8/q1_0
    names, and common EmitC/export code must not infer or choose block-quantized
    dot-product semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded resource shape
    knob the siblings carry (the *how*, never the *what*; the dot-product is
    byte-exact for any legal value because integer addition is order-independent and
    the per-sub-block/super-block fp32 fold stays in strict ascending order). The
    binary sign decode anchors at "m1" (the 8-lane sign-plane groups); the verifier
    rejects any other anchor fail-closed (I7).
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$activation_blocks_per_weight,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ51Q81Op
    : TCRVRVV_Op<"q5_1_q8_1_block_dot",
    [TCRVEmitCLowerableOpInterface,
     DeclareOpInterfaceMethods<TunableScheduleOpInterface>]> {
  let summary =
      "generic typed RVV block-quantized ggml Q5_1 x Q8_1 dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_q5_1_q8_1` block dot-product as ONE
    bounded generic plugin-local typed RVV dataflow step -- the Family-B 5-bit
    sibling that COMBINES the two prior breadth increments:
    tcrv_rvv.q5_0_q8_0_block_dot's 5-bit weight reconstruction (a 4-bit nibble PLUS
    a per-element 5th high bit packed in a 32-bit qh field) and
    tcrv_rvv.q4_1_q8_1_block_dot's scale+MIN fold (the asymmetric Family-B
    quantization). Two semantic facts distinguish q5_1 from its two parents:

    (1) The integer core decodes an UNSIGNED 5-bit weight in [0,31] -- the SAME
        4-bit nibble unpack + 5th-bit injection q5_0 uses (`vand 0x0F` low nibble /
        `vsrl 0x04` high nibble on the UNSIGNED weight lane, then the per-element
        5th bit injected from the broadcast 16-bit qh halves via a `vid + c` shift
        vector / `vsrl_vv` / `& 1` / `<< 4` / narrow / OR), but with NO
        offset-binary `-16` bias (the bias lives in q5_1's separate per-block MIN
        scale, exactly like q4_1). ggml reconstructs the low half as
        `(qs[j] & 0xF) | (((qh >> j) << 4) & 0x10)` and the high half as
        `(qs[j] >> 4) | ((qh >> (j+12)) & 0x10)` -- the SAME low bit j / high bit
        (j+16) selection q5_0 uses (q5_0's `(qh >> (j+16)) ... >> (j+12)` and
        q5_1's `(qh >> (j+12)) & 0x10` both land bit (j+16) at bit 4), so the
        5th-bit injection is byte-identical and only the final bias op is dropped.
        The decoded unsigned q5 lanes feed the SAME signed widening product
        against the plain-i8 q8_1 halves (vwmul/vwmacc) -> vwredsum -> i32 `sumi`.
        This is the 5-bit-UNSIGNED variant of the q5_0/q4_1 nibble core, added
        CLEANLY alongside them (the offset-binary, unsigned-nibble, and 5-bit
        offset-binary cores are byte-untouched).

    (2) The per-block fold carries a SECOND scale on EACH operand. block_q5_1 has
        {fp16 d; fp16 m; uint8 qh[4]; uint8 qs[16]} (stride 24) and block_q8_1 has
        {fp16 d; fp16 s; int8 qs[32]} (stride 36), where `m` is the per-block MIN
        and `s` is the PRECOMPUTED scaled activation sum (`d_y * sum(q8)`). The
        fold is ggml's exact statement `sumf += (d_x*d_y)*sumi + m_x*s_y`
        (quants.c) -- IDENTICAL to q4_1's: the (d_x*d_y) is multiplied FIRST, the
        two products (d_x*d_y)*sumi and m_x*s_y are SUMMED FIRST (the `+` binds
        before the `+=`), then added to sumf: `sumf + ((d_x*d_y)*sumi + m_x*s_y)`
        -- an fp-significant grouping. Both halves are read per block: d_x at
        weight+0 and m_x at weight+`weight_min_byte_offset` (+2); d_y at
        activation+0 and s_y at activation+`activation_sum_byte_offset` (+2).

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the q5_1 weight base pointer (vx), the q8_1 activation base
    pointer (vy), the fp32 output pointer (s), the runtime element count (n) --
    plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the single fp32 reduction result
    through the output pointer.

    The block-quantized dot-product STRUCTURE is first-class in the typed body.
    The structural facts -- QK=32, the AoS block strides (24 bytes for block_q5_1,
    36 bytes for block_q8_1), the WEIGHT quant payload byte offset (+8, after the
    two inline fp16 scales d/m and the 4-byte qh field), the ACTIVATION quant
    payload byte offset (+4, after the q8_1 two inline fp16 scales d/s -- DISTINCT
    from the weight's +8), the q8 high-half byte offset (+16), the qh high-bit
    field byte offset (+4), the second-scale byte offsets (weight m at +2,
    activation s at +2), and the dual-fp16-plus-min scale model -- are carried as
    typed integer/string attributes (I4 mirror only). The fixed emission -- an
    outer emitc.for block loop over nb = n / QK, per-block address arithmetic
    (vx + ib*24, vy + ib*36) via emitc.add/emitc.mul nodes, the FOUR scalar
    fp16->fp32 reads (d_x, m_x, d_y, s_y) plus the two aligned 16-bit qh half reads
    via emitc.call_opaque (the sanctioned opaque pieces, exactly as INC-1 emits its
    intrinsics), an inner strip loop reusing the unsigned-nibble decode + 5th-bit
    injection chain (NO `-16` bias) (-> vwmul/vwmacc against the plain q8 halves) +
    vwredsum into a per-block i32 scalar, and the fp32 accumulate
    `sumf = sumf + (d_x*d_y)*sumi + m_x*s_y` (ggml's exact order) -- is the op's
    lowering, NOT a candidate-mirror string read in the conversion. For the bounded
    surface kind is limited to "ggml_q5_1_q8_1_block_dot" and scale_model to
    "dual-fp16-per-block-d_x.d_y-plus-min". Route ids, helper names, ABI strings,
    artifact names, status fields, exact intrinsic spellings, operand mirror
    strings, q8/q5 names, and common EmitC/export code must not infer or choose
    block-quantized dot-product semantics.

    The optional `integer_core_lmul`, `multi_block_factor`, and `strip_elision`
    attributes are the SAME bounded resource/scheduling shape knobs the
    Q4_0/Q4_1/Q5_0 siblings carry (the *how*, never the *what*; the dot-product is
    byte-exact for any legal combination because integer addition is
    order-independent and the per-block fp32 fold stays in strict ascending block
    order). Because q5_1's nibble half-block is byte-identical in shape to q4_0's
    (16 nibble bytes, decoded into 2x16 lanes), the anchors are q4_0's, NOT q8_0's:
    - `integer_core_lmul` selects the i8 strip anchor: "m1" (i8m1 -> i16m2, ONE
      strip + ONE vwredsum per half-block at VLEN=128, matching ggml's reduction
      anchor) or "mf4" (the default: i8mf4 -> i16mf2, vsetvl_e32m1, 4 chunks).
    - `multi_block_factor` selects how many q5_1/q8_1 blocks the outer loop
      processes per iteration -- 1 (default), 2, or 4.
    - `strip_elision` selects "robust" (default, correct at any VLEN) or "elided"
      (a single vsetvl_e8m1(16) + one vwredsum per half-block, correct ONLY at
      VLEN >= 128 and therefore requiring the m1 anchor). The verifier rejects the
      "elided" + non-m1 combination fail-closed (I7); the autotuner gates "elided"
      on the Zvl128b capability. The `nb % factor` tail always uses the robust
      single-block core regardless of strip_elision.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$activation_high_byte_offset,
    I64Attr:$weight_qh_byte_offset,
    I64Attr:$weight_min_byte_offset,
    I64Attr:$activation_sum_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$multi_block_factor,
    OptionalAttr<StrAttr>:$strip_elision
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ6KQ8KAux32Op
    : TCRVRVV_Op<"q6_k_q8_k_aux32_partial",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q6_K x Q8_K super-block INTEGER aux32 partial op";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_q6_K_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step --
    the K-quant first super-block proof's K1 increment. It reproduces ggml's
    per-super-block INTEGER state `aux32[8]` EXACTLY as `_generic` computes it
    right BEFORE the fp32 `d`-multiply, byte-exact (i32 equality), and writes
    that 8-lane i32 state through a caller-supplied output pointer. The fp32
    two-level fold (the deferred per-super-block fp16-`d` x fp32-`d` multiply into
    `sums[8]` + the sequential horizontal sum into `*s`) is DELIBERATELY OUT OF
    SCOPE here -- that is the K2 increment. K1 isolates the genuinely-new
    super-block integer machinery (the 6-bit ql+qh unpack, the per-sub-block int8
    scale applied in the i32 domain, the nested sub-block structure, the lane-
    collapsed aux32 accumulator) from the fp32 hierarchy.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime
    ABI value operands -- the q6_K weight base pointer (vx), the q8_K activation
    base pointer (vy), the i32 aux32 output pointer (the 8-lane integer-state
    destination), the runtime element count (n) -- plus the active
    !tcrv_rvv.vl token. It produces ONE i32 LMUL m1 !tcrv_rvv.vector token whose
    lowering writes the per-super-block aux32[8] integer state through the output
    pointer.

    The block-quantized super-block STRUCTURE is first-class in the typed body.
    The structural facts -- QK_K=256 (super-block size), 16 sub-blocks of 16
    elements, the AoS block strides (210 bytes for block_q6_K, 292 bytes for
    block_q8_K), the q6_K payload byte offsets (ql @0, qh @128, scales @192,
    d @208) and the q8_K payload byte offset (qs @4) -- are carried as typed
    integer/string attributes (I4 mirror only). The fixed emission mirrors
    `_generic` line-for-line so byte-exactness is by construction:
    - an outer emitc.for super-block loop over nb = n / 256;
    - a function-scoped int8_t aux8[256] scratch array (an emitc.variable of
      !emitc.array<256 x !emitc.opaque<"int8_t">>), filled element-ordered and
      biased -32 by TWO 128-element chunks, each chunk doing the 4-strip vand/
      vsrl/vsll/vor/vsub 6-bit ql+qh reconstruct (the exact `a[l+0]/[l+32]/
      [l+64]/[l+96]` permutation) and a vse8 store into aux8 at the matching
      offset (the permutation lives ENTIRELY in the unpack -- the dot reads aux8
      contiguously, exactly as `_generic` does with `a += 8`);
    - a nested emitc.for sub-block loop over 16 sub-blocks: a scalar int8 scale
      load `scales[js]` (the per-sub-block scale, applied in the i32 domain), then
      two 8-element halves each doing vle8 q8 + vle8 aux8 -> vwmul i8xi8->i16 ->
      vwmacc.vx i32 += scale x i16 into the carried 8-lane aux32 i32 accumulator
      vector (e32m2 covers 8 lanes at VLEN>=128; integer add is order-independent
      so the lane-wise accumulation is byte-exact);
    - a vse32 store of the 8-lane aux32 vector through the output pointer at the
      super-block's aux32 destination (aux32 is RESET to zero each super-block,
      reproducing `_generic`'s `memset(aux32, 0, ...)`).
    This emission is the op's fixed structure, NOT a candidate-mirror string read
    in the conversion. The bounded surface kind is limited to
    "ggml_q6_k_q8_k_aux32_partial" and scale_model to
    "per-sub-block-int8-scale-i32-domain". Route ids, helper names, ABI strings,
    artifact names, status fields, exact intrinsic spellings, operand mirror
    strings, q8/q6 names, and common EmitC/export code must not infer or choose
    super-block dot-product semantics.

    K1 emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target): the unpack runs one
    vsetvl_e8m2(32) per 32-element group and the per-sub-block half runs
    e8mf2(8)->i16m1->e32m2 (the 8-lane aux32 accumulator), with NO VLEN-robust
    re-strip loop (a VLEN<128 board would leave part of aux8 unwritten -- the
    zve32x/zve64x re-strip is a K2/capability item, not built here). No resource
    shape knobs yet; the K3 measurement tuner will register q6_K's candidate
    shapes. This K1 op carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_qh_byte_offset,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$activation_quant_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ6KQ8KOp
    : TCRVRVV_Op<"q6_k_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q6_K x Q8_K super-block FULL block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_q6_K_q8_K` super-block dot-product as
    ONE bounded generic plugin-local typed RVV dataflow step -- the K-quant first
    super-block proof's K2 increment. It extends the K1 integer core
    (tcrv_rvv.q6_k_q8_k_aux32_partial) with the DEFERRED two-level fp32 fold and
    the ggml ABI to produce the fp32 `*s` dot-product output BYTE-EXACT vs
    `_generic` (the well-defined fp32 order is the byte-exactness target; the
    on-board RVV `_vl128` path is raw inline asm and does NOT define the order).

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime
    ABI value operands -- the q6_K weight base pointer (vx), the q8_K activation
    base pointer (vy), the fp32 output pointer `*s` (a `float *`, NOT K1's
    `int32_t *`), the runtime element count (n) -- plus the active !tcrv_rvv.vl
    token. It produces ONE i32 LMUL m1 !tcrv_rvv.vector token whose lowering
    writes the fp32 dot-product result `*s`.

    The block-quantized super-block STRUCTURE is first-class in the typed body.
    The structural facts -- QK_K=256, 16 sub-blocks of 16 elements, the AoS block
    strides (210 bytes for block_q6_K, 292 bytes for block_q8_K), the q6_K payload
    byte offsets (ql @0, qh @128, scales @192, the fp16 super-block scale d @208)
    and the q8_K payload byte offsets (the fp32 activation scale d @0, qs @4) --
    are carried as typed integer/string attributes (I4 mirror only). The fixed
    emission mirrors `_generic` (quants.c:800-853) line-for-line so byte-exactness
    is by construction:
    - the SAME K1 integer core per super-block (the 6-bit ql+qh unpack into the
      element-ordered int8_t aux8[256], biased -32; the nested sub-block loop
      applying the per-sub-block int8 scale in the i32 domain into the carried
      8-lane aux32 accumulator), producing the per-super-block aux32[8] state;
    - the DEFERRED two-level fp32 fold: a function-scoped 8-lane fp32 accumulator
      `sums` (vfloat32m2), declared and zeroed ONCE OUTSIDE the super-block loop
      (carried across super-blocks, unlike aux32 which resets each super-block);
      per super-block `d = GGML_FP16_TO_FP32(x[i].d) * y[i].d` (the fp16 weight
      scale at byte 208 read via the `(float)*(const _Float16 *)` seam, times the
      fp32 activation scale at byte 0, a plain float load), then `sums[l] +=
      d * aux32[l]` for all 8 lanes -- emitted as vfcvt (i32->f32, RNE, matching
      C's `(float)aux32[l]`) then a SEPARATE vfmul (by the scalar d) then a
      SEPARATE vfadd into `sums`. NEVER vfmacc/vfmadd: a fused FMA diverges from
      `_generic`'s separate scalar mul/add under `-ffp-contract=off` (the INC-2a
      FMA-contraction class). Cross-LANE the lanes are independent so vectorizing
      the fold body is byte-safe;
    - the SEQUENTIAL horizontal sum: `sums` is vse32-stored into a float[8] scratch
      and then summed by EIGHT scalar emitc.add in ASCENDING lane order l=0..7
      (`sumf += sums[l]`), NOT a vector vfredusum (fp add is non-associative; a
      vector reduce would reassociate and is not byte-exact). The lane l <-> index
      l correspondence is preserved by the vse32 store and pinned by the K1
      aux32[l] <-> l ordering;
    - `*s = sumf` (a structured subscript + assign through the float * output).
    This emission is the op's fixed structure, NOT a candidate-mirror string read
    in the conversion. The bounded surface kind is limited to
    "ggml_q6_k_q8_k_block_dot" and scale_model to
    "per-sub-block-int8-scale-i32-domain-deferred-fp32-fold". Route ids, helper
    names, ABI strings, artifact names, status fields, exact intrinsic spellings,
    operand mirror strings, q8/q6 names, and common EmitC/export code must not
    infer or choose super-block dot-product semantics.

    K2 emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target), inheriting K1's integer-core
    striping (one vsetvl_e8m2(32) per 32-element unpack group; e8mf2(8)->i16m1->
    e32m2 per sub-block half) with NO VLEN-robust re-strip loop (the zve32x/zve64x
    re-strip is a capability item, not built here). No resource shape knobs yet;
    the K3 measurement tuner will register q6_K's candidate shapes. This K2 op
    carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_qh_byte_offset,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ4KQ8KAux32Op
    : TCRVRVV_Op<"q4_k_q8_k_aux_partial",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q4_K x Q8_K super-block INTEGER aux32 partial op";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_q4_K_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step --
    the q4_K K-quant proof's K4a increment. q4_K is the most-used modern K-quant
    (the dominant weight quant in `*_K_M` mixes). K4a is split from the fp32 fold
    for a reliable dispatch: it isolates the genuinely-new q4_K INTEGER state --
    (1) the 6-bit scale/min bit-dance unpack (the `get_scale_min_k4` /
    `utmp[4]`/`kmask` cross-byte decode), and (2) the per-sub-block 4-bit dot
    scaled into the i32 accumulator -- byte-exact vs `_generic`'s INTEGER state
    BEFORE the fp32 `d`/`dmin` fold. The fp32 two-level fold + the `min` term
    (`Σ bsums*mins`) + the ggml ABI are DELIBERATELY OUT OF SCOPE here -- that is
    K4b.

    The op is nested directly under tcrv_rvv.with_vl and consumes FIVE runtime
    ABI value operands -- the q4_K weight base pointer (vx), the q8_K activation
    base pointer (vy), the i32 aux32 output pointer (the 8-lane integer-state
    destination), the uint8 scale/min output pointer (the per-super-block 16
    decoded bytes: 8 6-bit scales then 8 6-bit mins, mirroring _generic's
    contiguous `utmp[4]` reinterpreted as `scales=&utmp[0]`/`mins=&utmp[2]`), the
    runtime element count (n) -- plus the active !tcrv_rvv.vl token. It produces
    ONE i32 LMUL m1 !tcrv_rvv.vector token whose lowering writes the
    per-super-block aux32[8] integer state AND the 16 decoded scale/min bytes
    through the two output pointers.

    The block-quantized super-block STRUCTURE is first-class in the typed body.
    The structural facts -- QK_K=256 (super-block size), 8 sub-blocks of 32
    elements, the AoS block strides (144 bytes for block_q4_K, 292 bytes for
    block_q8_K), the q4_K payload byte offsets (d @0, dmin @2, the 12 packed
    scale bytes @4, qs @16) and the q8_K payload byte offset (qs @4) -- are
    carried as typed integer/string attributes (I4 mirror only). The fixed
    emission mirrors `_generic` (quants.c:644-718) line-for-line so byte-exactness
    is by construction:
    - a function-scoped int8_t aux8[256] scratch (an emitc.variable of
      !emitc.array<256 x !emitc.opaque<"int8_t">>), filled element-ordered (NO
      bias) by FOUR 64-element chunks, each chunk doing a 32-wide e8m2 vand 0x0F
      (low nibble -> a[0..31]) and a 32-wide e8m2 vsrl 4 (high nibble ->
      a[32..63]) vreinterpret to i8 + vse8 store (the q4_K nibbles are UNsigned
      [0,15], no -32 bias -- the bias is carried by the min term, K4b);
    - the 6-bit scale/min bit-dance, STRUCTURED scalar emitc: the 12 packed bytes
      at `scales @4` are read as THREE uint32 words; the `utmp[4]`/`kmask1`
      (0x3f3f3f3f) /`kmask2` (0x0f0f0f0f) /`kmask3` (0x03030303) cross-byte
      shuffle is reproduced with emitc.bitwise_and/bitwise_or/bitwise_left_shift/
      bitwise_right_shift over the three words into a 4-word utmp, then the 16
      bytes (scales[0..7], mins[0..7]) are vse8-stored through the scale/min
      output pointer (the same type-pun _generic does);
    - a nested emitc.for sub-block loop over 8 sub-blocks: the per-sub-block UINT8
      ZERO-EXTENDED 6-bit scale `scales[js]` (applied in the i32 domain -- NOT
      q6_K's sign-extended int8), then FOUR 8-element quarters each doing vle8 q8
      + vle8 aux8 -> vwmul i8xi8->i16 -> vwmacc.vx i32 += scale x i16 into the
      carried 8-lane aux32 i32 accumulator (e32m2 covers 8 lanes at VLEN>=128;
      integer add is order-independent so the lane-wise accumulation is
      byte-exact);
    - a vse32 store of the 8-lane aux32 vector through the aux32 output pointer at
      the super-block's destination (aux32 is RESET to zero each super-block,
      reproducing `_generic`'s `memset(aux32, 0, ...)`).
    This emission is the op's fixed structure, NOT a candidate-mirror string read
    in the conversion. The bounded surface kind is limited to
    "ggml_q4_k_q8_k_aux_partial" and scale_model to
    "per-sub-block-uint6-scale-i32-domain". Route ids, helper names, ABI strings,
    artifact names, status fields, exact intrinsic spellings, operand mirror
    strings, q8/q4 names, and common EmitC/export code must not infer or choose
    super-block dot-product semantics.

    K4a emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target): the unpack runs one
    vsetvl_e8m2(32) per 32-element nibble group and the per-sub-block quarter runs
    e8mf2(8)->i16m1->e32m2 (the 8-lane aux32 accumulator), with NO VLEN-robust
    re-strip loop (a VLEN<128 board would leave part of aux8 unwritten -- the
    zve32x/zve64x re-strip is a capability item, not built here). The bit-dance is
    scalar (VLEN-independent). No resource shape knobs yet. This K4a op carries
    only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$aux32_output,
    AnyType:$scalemin_output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$activation_quant_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $aux32_output `,` $scalemin_output "
    "`,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($aux32_output) `,` type($scalemin_output) `,` type($element_count) "
    "`,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ4KQ8KOp
    : TCRVRVV_Op<"q4_k_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q4_K x Q8_K super-block FULL block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_q4_K_q8_K` super-block dot-product as
    ONE bounded generic plugin-local typed RVV dataflow step -- the q4_K K-quant
    proof's K4b increment. It extends the K4a integer core
    (tcrv_rvv.q4_k_q8_k_aux_partial) with the DEFERRED two-level fp32 fold, the
    q4_K MIN term, and the ggml ABI to produce the fp32 `*s` dot-product output
    BYTE-EXACT vs `_generic` (the well-defined fp32 order is the byte-exactness
    target; the on-board RVV `_vl128` path is raw inline asm and does NOT define
    the order).

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime
    ABI value operands -- the q4_K weight base pointer (vx), the q8_K activation
    base pointer (vy), the fp32 output pointer `*s` (a `float *`, NOT K4a's
    `int32_t *` aux32 / `uint8_t *` scale-min), the runtime element count (n) --
    plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the fp32 dot-product result `*s`.

    The block-quantized super-block STRUCTURE is first-class in the typed body.
    The structural facts -- QK_K=256, 8 sub-blocks of 32 elements, the AoS block
    strides (144 bytes for block_q4_K, 292 bytes for block_q8_K), the q4_K payload
    byte offsets (the fp16 super-block scale d @0, the fp16 super-block min scale
    dmin @2, the 12 packed scale/min bytes @4, qs @16) and the q8_K payload byte
    offsets (the fp32 activation scale d @0, qs @4, the int16 per-sub-block sums
    bsums @260) -- are carried as typed integer/string attributes (I4 mirror
    only). The fixed emission mirrors `_generic` (quants.c:645-718) line-for-line
    so byte-exactness is by construction:
    - the SAME K4a integer core per super-block (the plain 4-bit nibble unpack
      into the element-ordered int8_t aux8[256] with NO bias; the STRUCTURED
      scalar 6-bit scale/min bit-dance via utmp/kmask into the 16 decoded
      scale[8]+min[8] bytes; the nested sub-block loop applying the per-sub-block
      UINT6 scale in the i32 domain into the carried 8-lane aux32 accumulator),
      producing the per-super-block aux32[8] integer state AND the decoded
      mins[8];
    - the DEFERRED two-level fp32 fold (the q6_K K2 mechanism reused): a
      function-scoped 8-lane fp32 accumulator `sums` (vfloat32m2), declared and
      zeroed ONCE OUTSIDE the super-block loop (carried across super-blocks,
      unlike aux32 which resets each super-block); per super-block `d =
      GGML_FP16_TO_FP32(x[i].d) * y[i].d` (the fp16 weight scale at byte 0 read
      via the `(float)*(const _Float16 *)` seam, times the fp32 activation scale
      at byte 0, a plain float load), then `sums[l] += d * aux32[l]` for all 8
      lanes -- emitted as vfcvt (i32->f32, RNE, matching C's `(float)aux32[l]`)
      then a SEPARATE vfmul (by the scalar d) then a SEPARATE vfadd into `sums`.
      NEVER vfmacc/vfmadd: a fused FMA diverges from `_generic`'s separate scalar
      mul/add under `-ffp-contract=off` (the INC-2a FMA-contraction class);
    - the q4_K MIN term (the new piece vs q6_K, sourced from the q8_K bsums and
      the decoded mins): a function-scoped scalar fp32 accumulator `sumf` init to
      0.0f ONCE OUTSIDE the super-block loop (carried across super-blocks like
      `sums`, but a scalar -- distinct from q6_K's post-loop-only sumf); per
      super-block the integer `sumi = Σ_{j=0..15} y[i].bsums[j] * mins[j/2]`
      (mins are the decoded UINT6 6-bit per-sub-block mins, bsums the int16 q8_K
      per-sub-block sums @260; integer multiply/add, order-free) accumulated by a
      scalar emitc loop, then `dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d` (the
      fp16 weight min scale at byte 2, the same fp32 activation d), then `sumf -=
      dmin * (float)sumi` -- emitted as ONE emitc.expression so the subtract +
      multiply renders as ggml's single C statement and tracks its contraction;
    - the SEQUENTIAL horizontal sum: AFTER the super-block loop, `sums` is
      vse32-stored into a float[8] scratch and the carried scalar `sumf` (which
      already holds the accumulated MIN subtractions) is summed with the 8 lanes
      in ASCENDING lane order l=0..7 (`sumf += sums[l]`), NOT a vector vfredusum
      (fp add is non-associative). This exactly mirrors `_generic`: the min
      subtractions land in `sumf` first (in super-block order, inside the loop),
      then the 8 lane adds run after the loop -- the order is the byte-exactness;
    - `*s = sumf` (a structured subscript + assign through the float * output).
    This emission is the op's fixed structure, NOT a candidate-mirror string read
    in the conversion. The bounded surface kind is limited to
    "ggml_q4_k_q8_k_block_dot" and scale_model to
    "per-sub-block-uint6-scale-i32-domain-deferred-fp32-fold-min". Route ids,
    helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, operand mirror strings, q8/q4 names, and common EmitC/export code
    must not infer or choose super-block dot-product semantics.

    K4b emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target), inheriting K4a's integer-core
    striping (one vsetvl_e8m2(32) per 32-element nibble group; e8mf2(8)->i16m1->
    e32m2 per sub-block quarter) with NO VLEN-robust re-strip loop (the
    zve32x/zve64x re-strip is a capability item, not built here).

    The optional `integer_core_lmul` attribute is a bounded resource/scheduling
    knob selecting the BASE LMUL of the per-sub-block integer-MAC widening chain
    i8 -> i16 -> i32 (the *how*, never the *what*; the byte-exact fp32 fold order
    is untouched). It anchors the chain at "mf2" (absent/default -- today's
    e8mf2(8)->i16m1->e32m2 form), "m1" (e8m1->i16m2->e32m4), or "m2"
    (e8m2->i16m4->e32m8). "m2" is the hard ceiling on TWO independent grounds:
    the register-group limit (an m4 base would need an illegal i32m16 product)
    AND the sub-block-scale boundary (i8m2 == 32 elements == exactly one
    sub-block under one scalar `scale`; m4 would fold TWO sub-blocks under one
    scalar). The verifier pins the legal set {"mf2","m1","m2"} fail-closed (I7).
    The 6-bit scale/min bit-dance is scalar (LMUL-free) and is untouched by this
    knob. The K3 measurement tuner stamps the chosen anchor; this K4b op carries
    the structural facts plus this single dormant resource shape knob.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$weight_dmin_byte_offset,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$activation_bsums_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ5KQ8KOp
    : TCRVRVV_Op<"q5_k_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q5_K x Q8_K super-block FULL block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_q5_K_q8_K` super-block dot-product as
    ONE bounded generic plugin-local typed RVV dataflow step -- the q5_K K-quant
    COVERAGE rung. q5_K is q4_K PLUS a 5th (high) weight bit: the super-block
    structure, the 6-bit scale/min bit-dance, the deferred two-level fp32 fold,
    the q4_K MIN term, and the ggml ABI are ALL IDENTICAL to q4_K
    (tcrv_rvv.q4_k_q8_k_block_dot). The ONLY new piece is the qh plane: a
    per-element 5th bit injected into each unpacked 4-bit nibble during the
    integer unpack, lifting q4 in [0,15] to q5 in [0,31] (UNSIGNED, Family-B
    K-quant). It produces the fp32 `*s` dot-product output BYTE-EXACT vs
    `_generic` (the well-defined fp32 order is the byte-exactness target; the
    on-board RVV `_vl128` path is raw inline asm and does NOT define the order).

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime
    ABI value operands -- the q5_K weight base pointer (vx), the q8_K activation
    base pointer (vy), the fp32 output pointer `*s` (a `float *`), the runtime
    element count (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32
    LMUL m1 !tcrv_rvv.vector token whose lowering writes the fp32 dot result `*s`.

    The block-quantized super-block STRUCTURE is first-class in the typed body.
    The structural facts -- QK_K=256, 8 sub-blocks of 32 elements, the AoS block
    strides (176 bytes for block_q5_K, 292 bytes for block_q8_K), the q5_K payload
    byte offsets (the fp16 super-block scale d @0, the fp16 super-block min scale
    dmin @2, the 12 packed scale/min bytes @4, the 32 qh high-bit-plane bytes @16,
    qs @48) and the q8_K payload byte offsets (the fp32 activation scale d @0, qs
    @4, the int16 per-sub-block sums bsums @260) -- are carried as typed
    integer/string attributes (I4 mirror only). The fixed emission mirrors
    `_generic` (quants.c:720-798) line-for-line so byte-exactness is by
    construction. The emission is IDENTICAL to q4_K's K4b EXCEPT for the unpack:
    - the q5_K integer core reuses q4_K's super-block aux32 core (the plain 4-bit
      nibble unpack into the element-ordered int8_t aux8[256], the STRUCTURED
      scalar 6-bit scale/min bit-dance via utmp/kmask, the nested sub-block UINT6
      scale in the i32 domain into the carried 8-lane aux32 accumulator) with the
      qh plane PLUMBED IN: per 32-element half h in 0..7 (h = chunk*2 +
      lowNibble?0:1), the 5th bit for element l is `(qh[l] >> h) & 1`, contributing
      16 when set, added to the unpacked nibble in the UINT8 domain BEFORE the
      u8->i8 reinterpret (q5 in [0,31] is non-negative so the reinterpret is exact).
      The qh plane is loaded from a FIXED per-super-block pointer (xb + qh@16) and
      reused across all 8 halves -- only the bit-shift `h` varies. This mirrors
      `_generic`'s `m = 1; ...; a[l] += (hm[l] & m ? 16 : 0); m <<= 1` per half;
    - the DEFERRED two-level fp32 fold, the q4_K MIN term, and the SEQUENTIAL
      horizontal sum are reused from K4b node-for-node (the same separate
      vfcvt/vfmul/vfadd positive fold, the same `sumi = Σ bsums[j]*mins[j/2]` +
      `sumf -= dmin*sumi` min subtraction, the same ascending l=0..7 horizontal
      sum, the same `*s = sumf`).
    This emission is the op's fixed structure, NOT a candidate-mirror string read
    in the conversion. The bounded surface kind is limited to
    "ggml_q5_k_q8_k_block_dot" and scale_model to
    "per-sub-block-uint6-scale-i32-domain-deferred-fp32-fold-min". Route ids,
    helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, operand mirror strings, q8/q5 names, and common EmitC/export code
    must not infer or choose super-block dot-product semantics.

    K5 emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target), inheriting q4_K's integer-core
    striping (one vsetvl_e8m2(32) per 32-element nibble group; e8mf2(8)->i16m1->
    e32m2 per sub-block quarter) with NO VLEN-robust re-strip loop. No resource
    shape knobs yet; this op carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$weight_dmin_byte_offset,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$weight_qh_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$activation_bsums_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ2KQ8KOp
    : TCRVRVV_Op<"q2_k_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q2_K x Q8_K super-block FULL block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_q2_K_q8_K` super-block dot-product as
    ONE bounded generic plugin-local typed RVV dataflow step -- the q2_K K-quant
    COVERAGE rung. q2_K is the 2-bit modern K-quant: it REUSES the super-block
    scaffolding (the QK_K=256 AoS super-block loop, the q8_K activation handling,
    the `(float)*(const _Float16 *)` fp16 read seam, the structured `*s` store)
    and the scale+min structure (a per-sub-block scale term plus a `-dmin * Σ
    bsums*min` min term) of q4_K, but with THREE q2_K-specific differences from
    q4_K's K4b: (1) the weights are 2-bit (4 packed per qs byte), unpacked by a
    plain `(qs >> shift) & 3` over the 4 shifts {0,2,4,6}; (2) the per-sub-block
    scales and mins are SIMPLE 4-bit nibbles of the 16 direct `scales[16]` bytes
    (`sc[j] & 0xF` is the scale, `sc[j] >> 4` is the min) -- NO 6-bit utmp/kmask
    bit-dance; (3) the positive fold is SCALAR -- q2_K accumulates a single
    per-super-block integer `isum` and folds `sumf += dall*isum - dmin*summs` as
    ONE scalar fp32 statement in super-block order (NOT q4_K's 8-lane deferred
    vector `sums` + post-loop sequential horizontal sum, which exists only because
    q4_K's `_generic` decomposes its positive term into 8 lanes). It produces the
    fp32 `*s` dot-product output BYTE-EXACT vs `_generic` (the well-defined fp32
    order is the byte-exactness target; the on-board RVV `_vlN` path is raw inline
    asm and does NOT define the order).

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime
    ABI value operands -- the q2_K weight base pointer (vx), the q8_K activation
    base pointer (vy), the fp32 output pointer `*s` (a `float *`), the runtime
    element count (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32
    LMUL m1 !tcrv_rvv.vector token whose lowering writes the fp32 dot result `*s`.

    The block-quantized super-block STRUCTURE is first-class in the typed body.
    The structural facts -- QK_K=256, 16 sub-blocks of 16 elements, the AoS block
    strides (84 bytes for block_q2_K, 292 bytes for block_q8_K), the q2_K payload
    byte offsets (the 16 packed 4-bit-scale/4-bit-min bytes `scales` @0, the 64
    packed 2-bit-weight bytes qs @16, the fp16 super-block scale d @80, the fp16
    super-block min scale dmin @82) and the q8_K payload byte offsets (the fp32
    activation scale d @0, qs @4, the int16 per-sub-block sums bsums @260) -- are
    carried as typed integer/string attributes (I4 mirror only). The fixed
    emission mirrors `_generic` (quants.c:514-564) line-for-line so byte-exactness
    is by construction:
    - the 2-bit weight unpack into the element-ordered int8_t aux8[256]: for each
      32-byte qs chunk (chunk k in 0..1, qs advances 32 bytes per chunk) and each
      2-bit shift in {0,2,4,6}, `aux8[128*k + 32*(shift/2) + l] = (qs[k*32+l] >>
      shift) & 3` for the 32 lanes l, emitted as a u8m2 load + vsrl_vx + vand_vx +
      u8->i8 reinterpret + vse8 (q2 in [0,3] is non-negative so the reinterpret is
      exact). The aux8 element ordering matches `_generic`: sub-block s in 0..15
      maps to aux8[16*s : 16*s+16], q8[16*s : 16*s+16], and bsums[s];
    - the 4-bit nibble scale/min extraction (the SIMPLER q2_K piece, NO bit-dance):
      the 16 `scales[16]` bytes are read directly -- `sc[s] & 0xF` is the
      per-sub-block scale, `sc[s] >> 4` is the per-sub-block min -- as structured
      scalar emitc.load + emitc.bitwise_and / emitc.bitwise_right_shift;
    - the SCALAR positive integer dot: a per-super-block `int isum = 0` accumulating
      Σ_{s=0..15} (sc[s]&0xF) * (Σ_{l in sub-block s} q8[l]*aux8[l]); the inner
      16-element products are computed in the vector domain (vle8 i8m1 ->
      vwmul_vv i16m2 -> vwredsum into an i32 lane 0 extracted to int; LMUL=1 so
      the 16-element reduce sees all 16 lanes at VLEN>=128) and the
      per-sub-block reduction `isuml` scaled by the int scale and accumulated into
      the scalar `isum` (integer multiply/add, order-free);
    - the q2_K MIN term: a per-super-block `int summs = Σ_{s=0..15} bsums[s] *
      (sc[s] >> 4)` (bsums the int16 q8_K per-sub-block sums @260, SIGN-extended;
      each min the high nibble of one scales byte; integer multiply/add,
      order-free), accumulated by a scalar emitc loop;
    - the SCALAR fp32 fold: `dall = (float)*(const _Float16 *)(xb+80) * dy` and
      `dmin = (float)*(const _Float16 *)(xb+82) * dy` (dy the fp32 activation scale
      at byte 0, loaded ONCE), then `sumf += dall*isum - dmin*summs` -- emitted as
      ONE emitc.expression so the two products + the add + the subtract render as
      ggml's single C statement (quants.c:561) and track the contraction. This is
      a SCALAR accumulation in super-block order (NOT a vector lane accumulator),
      carried across super-blocks;
    - `*s = sumf` (a structured subscript + assign through the float * output)
      AFTER the super-block loop.
    This emission is the op's fixed structure, NOT a candidate-mirror string read
    in the conversion. The bounded surface kind is limited to
    "ggml_q2_k_q8_k_block_dot" and scale_model to
    "per-sub-block-uint4-scale-i32-domain-scalar-fp32-fold-min". Route ids, helper
    names, ABI strings, artifact names, status fields, exact intrinsic spellings,
    operand mirror strings, q8/q2 names, and common EmitC/export code must not
    infer or choose super-block dot-product semantics.

    q2_K emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target), with NO VLEN-robust re-strip loop
    (the zve32x/zve64x re-strip is a capability item, not built here). No resource
    shape knobs yet; this op carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$weight_dmin_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$activation_bsums_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ3KQ8KOp
    : TCRVRVV_Op<"q3_k_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q3_K x Q8_K super-block FULL block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_q3_K_q8_K` super-block dot-product as
    ONE bounded generic plugin-local typed RVV dataflow step -- the q3_K K-quant
    COVERAGE rung, the LAST common K-quant (after it the q4_0/q8_0/q4_1/q5_0/q5_1
    legacy + q6_K/q4_K/q5_K/q2_K K-quants cover every common ggml dot kernel; only
    the IQ/MXFP4 codebook/fp4 tail remains). q3_K is the 3-bit modern K-quant; it
    COMPOSES mechanisms already in hand with THREE q3_K-specific twists:
    - the 2-bit `qs` unpack (q2_K's `(qs >> shift) & 3` over shifts {0,2,4,6});
    - the hmask high-bit plane (q5_K's fixed 32-byte plane pattern) but applied
      SUBTRACTIVELY and lifting to SIGNED -- `a = (low2 | (hbit<<2)) - 4` where
      `hbit = (hm[l] >> p) & 1` and the running bit position p = 4*chunk +
      shiftIdx (the hmask is the SAME 32 bytes across all 8 groups -- `hm` is
      NEVER advanced, unlike q6_K's qh; only the tested bit `m` shifts left). This
      is EXACTLY `_generic`'s `a[l] = (q3[l] >> shift) & 3; a[l] -= (hm[l] & m ? 0
      : 4)` (bit set -> a in [0,3]; bit unset -> a in [-4,-1]; the signed range is
      [-4,3]);
    - the 6-bit SIGNED scale dance: the 12 packed `scales[12]` bytes decode (a
      q3_K-OWN bit-dance, masks kmask1=0x03030303 / kmask2=0x0f0f0f0f, DISTINCT
      from q4_K's 0x3f3f3f3f) into 16 6-bit values read as SIGNED int8, then
      `scale_j = scales[j] - 32` (signed, in [-32,31]).
    The fold is q6_K's NO-min deferred d.Sum(aux32) structure (symmetric, like
    q6_K -- there is NO min term and NO dmin): a function-scoped 8-lane fp32
    accumulator `sums`, per-super-block `d = fp16(x.d) * y.d`, lane-wise `sums[l]
    += d * aux32[l]` (SEPARATE vfmul/vfadd, NEVER fma), then a SEQUENTIAL
    horizontal sum and `*s = sumf`. It produces the fp32 `*s` dot-product output
    BYTE-EXACT vs `_generic` (the well-defined fp32 order is the byte-exactness
    target; the on-board RVV `_vlN` path is raw inline asm and does NOT define the
    order).

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime
    ABI value operands -- the q3_K weight base pointer (vx), the q8_K activation
    base pointer (vy), the fp32 output pointer `*s` (a `float *`), the runtime
    element count (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32
    LMUL m1 !tcrv_rvv.vector token whose lowering writes the fp32 dot result `*s`.

    The block-quantized super-block STRUCTURE is first-class in the typed body.
    The structural facts -- QK_K=256, 16 sub-blocks of 16 elements, the AoS block
    strides (110 bytes for block_q3_K, 292 bytes for block_q8_K), the q3_K payload
    byte offsets (the 32-byte high-bit-plane `hmask` @0, the 64 packed
    2-bit-weight bytes qs @32, the 12 packed 6-bit-signed-scale bytes scales @96,
    the fp16 super-block scale d @108) and the q8_K payload byte offsets (the fp32
    activation scale d @0, qs @4) -- are carried as typed integer/string
    attributes (I4 mirror only). The fixed emission mirrors `_generic`
    (quants.c:566-643) line-for-line so byte-exactness is by construction:
    - the 2-bit + SUBTRACTIVE-hmask unpack into the element-ordered int8_t
      aux8[256]: for each 32-byte qs chunk (chunk in 0..1) and each 2-bit shift in
      {0,2,4,6} (shiftIdx 0..3), `aux8[128*chunk + 32*shiftIdx + l] = ((qs >>
      shift) & 3 | (((hm[l] >> p) & 1) << 2)) - 4` for the 32 lanes l, with p =
      4*chunk + shiftIdx -- emitted as a u8m2 load + vsrl/vand for the 2 low bits,
      a vsrl/vand/vsll of the SAME hmask plane (loaded once per chunk) for the
      high bit OR-ed in (vor), a u8->i8 reinterpret, a vsub 4, a vse8. The aux8
      element ordering matches `_generic`: sub-block s in 0..15 -> aux8[16s:16s+16];
    - the q3_K SIGNED 6-bit scale dance, STRUCTURED scalar emitc: the 12 packed
      bytes at `scales @96` are read as THREE uint32 words w0/w1/w2 (capture all
      three BEFORE any output to avoid `_generic`'s in-place read-before-write
      hazard), then the q3_K shuffle (masks kmask1=0x03030303 / kmask2=0x0f0f0f0f)
      reproduces `_generic`'s auxs[0..3] via emitc.bitwise_and/_or/_left_shift/
      _right_shift, the 16 bytes are vse8-stored to a scratch and read back as
      SIGNED int8, and `scale_j = (int)scales[j] - 32`;
    - a nested sub-block loop over 16 sub-blocks, each doing TWO 8-element halves
      vle8 q8 + vle8 aux8 -> vwmul i8xi8->i16 -> vwmacc.vx i32 += (scales[js]-32) x
      i16 into the carried 8-lane aux32 i32 accumulator (the SAME structure as
      q6_K's sub-block loop, the signed scale being scales[js]-32 instead of
      q6_K's direct int8; integer add is order-free so byte-exact);
    - the DEFERRED two-level fp32 fold IDENTICAL to q6_K's (NO min): a
      function-scoped 8-lane fp32 `sums` declared+zeroed ONCE outside the loop,
      per super-block `d = fp16(x.d) * y.d`, `vfcvt` (i32->f32, RNE) then a
      SEPARATE `vfmul` by d then a SEPARATE `vfadd` into `sums` (NEVER vfmacc), a
      SEQUENTIAL ascending l=0..7 horizontal sum, `*s = sumf`.
    This emission is the op's fixed structure, NOT a candidate-mirror string read
    in the conversion. The bounded surface kind is limited to
    "ggml_q3_k_q8_k_block_dot" and scale_model to
    "per-sub-block-int6-signed-scale-i32-domain-deferred-fp32-fold". Route ids,
    helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, operand mirror strings, q8/q3 names, and common EmitC/export code
    must not infer or choose super-block dot-product semantics.

    q3_K emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target), with NO VLEN-robust re-strip loop
    (the zve32x/zve64x re-strip is a capability item, not built here). No resource
    shape knobs yet; this op carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_hmask_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotTQ20Q8KOp
    : TCRVRVV_Op<"tq2_0_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml TQ2_0 x Q8_K super-block FULL block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_tq2_0_q8_K` super-block dot-product
    as ONE bounded generic plugin-local typed RVV dataflow step -- the TERNARY
    ({-1,0,+1}) TriLM coverage rung, one of the LAST TWO uncommon ggml dot
    kernels (the other is tq1_0, the base-3-packed sibling). tq2_0 is the 2-bit
    TriLM ternary K-quant: each weight is a 2-bit field whose value decodes to a
    ternary lane `((qs >> (l*2)) & 3) - 1` in {-1,0,1,2} (the encoder only emits
    {0,1,2} so the realized ternary is {-1,0,1}; the structural decode is exact
    over the full 2-bit range). The activation is q8_K (the SAME super-block
    activation as q2_K/q3_K/q4_K/q5_K/q6_K).

    tq2_0 REUSES q2_K's 2-bit weight unpack VERBATIM (the same `(qs >> shift) &
    3` over the 4 shifts {0,2,4,6} into an element-ordered int8_t aux8[256]) but
    is genuinely SIMPLER than every K-quant sibling -- there is NO scales[16]
    nibble extraction, NO per-sub-block scale, NO min term, NO dmin, NO bsums.
    THREE structural facts distinguish tq2_0, all load-bearing:

    (1) SUPER-BLOCK FORMAT, d AT THE END. block_tq2_0 is `{ uint8_t qs[QK_K/4];
        ggml_half d }` (QK_K = 256, sizeof 66): the 64 packed 2-bit-weight qs
        bytes LEAD the block at byte +0, and the fp16 super-block scale d follows
        at byte +64 (so `weight_qs_byte_offset = 0`, `weight_d_byte_offset = 64`,
        `weight_block_stride = 66`). This is the genuinely-new layout vs every
        sibling -- every other block has d near the front; here the weight is the
        prefix and the single scale is the suffix.

    (2) TERNARY `-1` BIAS, folded into the unpack. `_generic` (quants.c:501)
        computes `(((qs >> (l*2)) & 3) - 1)` PER ELEMENT, so the emitter folds
        the `- 1` into the unpack: after q2_K's `&3` + u8->i8 reinterpret, ONE
        `vadd.vx -1` on the i8 vector before the `vse8`, so aux8 holds the
        signed ternary lanes {-1,0,1,2} directly. The dot is then q2_K's
        VERBATIM signed widening reduce (vle8 i8 x q8 i8 -> vwmul i16 ->
        vwredsum i32) with NO per-sub-block scale multiply -- a single
        per-super-block integer accumulator `sumi`. (The per-element subtract,
        NOT a `Σ(field*q8) - Σq8` refactor, mirrors `_generic` and keeps the
        decode exact; omitting it is a non-vacuous wrong-ternary control.)

    (3) ONE INTEGER ACCUMULATOR + a SINGLE-SCALE SCALAR fp32 FOLD. ggml
        accumulates ONE `int32_t sumi` over the WHOLE super-block (the 256
        ternary*q8 products, integer add is order-free), then folds
        `sumf += (float)sumi * d` with `d = y.d * fp16(x.d)` -- ONE fp16 read of
        the weight scale (xb+64), ONE fp32 read of the q8_K scale (yb+0), their
        product, then the scaled accumulate. The `d` product is its OWN statement
        and the fold is ONE emitc.expression so the association matches
        `_generic`'s `d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d); sumf += (float)
        sumi * d` (quants.c:506-508) and the fp32 byte-exactness is by
        construction (NOT a 3-factor `sumi*yd*xd` inline, which C would
        left-associate and break the last-ulp match).

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime
    ABI value operands -- the tq2_0 weight base pointer (vx), the q8_K activation
    base pointer (vy), the fp32 output pointer `*s` (a `float *`), the runtime
    element count (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32
    LMUL m1 !tcrv_rvv.vector token whose lowering writes the fp32 dot result `*s`.

    The structural facts -- QK_K=256, the AoS block strides (66 bytes for
    block_tq2_0, 292 bytes for block_q8_K), the tq2_0 weight byte offsets (qs @0,
    d @64) and the q8_K payload byte offsets (the fp32 activation scale d @0, qs
    @4) -- are carried as typed integer/string attributes (I4 mirror only). The
    bounded surface kind is limited to "ggml_tq2_0_q8_k_block_dot" and
    scale_model to "ternary-single-fp16-scale-i32-domain-scalar-fp32-fold". The
    op identity is the dispatch key. Route ids, helper names, ABI strings,
    artifact names, status fields, exact intrinsic spellings, operand mirror
    strings, q8/tq2_0 names, and common EmitC/export code must not infer or
    choose super-block dot-product semantics.

    tq2_0 emits the byte-exact form for the VLEN>=128 capability class (rv64gcv
    => Zvl128b => VLEN>=128, the validated target), with NO VLEN-robust re-strip
    loop (the zve32x/zve64x re-strip is a capability item, not built here). No
    resource shape knobs yet; this op carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotTQ10Q8KOp
    : TCRVRVV_Op<"tq1_0_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml TQ1_0 x Q8_K super-block FULL block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_tq1_0_q8_K` super-block dot-product
    as ONE bounded generic plugin-local typed RVV dataflow step -- the LAST of
    the 24 ggml dot kernels (literal 100% coverage). tq1_0 is the 1.6875-bpw
    TriLM ternary K-quant and the BASE-3-PACKED sibling of tq2_0: instead of a
    2-bit field shift, each weight is a base-3 TRIT recovered by a power-of-three
    multiply (5 trits per qs byte since 3^5 = 243 < 256, plus a 4-trits-per-byte
    qh tail). The activation is q8_K (the SAME super-block activation as
    q2_K/q3_K/q4_K/q5_K/q6_K). tq1_0 uses ONLY y.d and y.qs -- NO bsums, NO min.
    THREE structural facts distinguish tq1_0, all load-bearing:

    (1) SUPER-BLOCK FORMAT, d AT THE END, with a SECOND weight array qh.
        block_tq1_0 is `{ uint8_t qs[(QK_K - 4*QK_K/64)/5]; uint8_t qh[QK_K/64];
        ggml_half d }` (QK_K = 256, qs = 48 bytes, qh = 4 bytes, sizeof 54): the
        48 packed base-3 qs bytes LEAD at byte +0, the 4 base-3 qh bytes follow
        at byte +48, and the fp16 super-block scale d is the SUFFIX at byte +52
        (so `weight_qs_byte_offset = 0`, `weight_qh_byte_offset = 48`,
        `weight_d_byte_offset = 52`, `weight_block_stride = 54`). The qh array is
        the new structural fact vs every sibling -- tq1_0 carries TWO base-3
        weight arrays (the 5-trits-per-byte qs and the 4-trits-per-byte qh) whose
        decoded lanes both contribute to ONE super-block integer accumulator.

    (2) BASE-3 TRIT EXTRACT, with the mandatory uint8 wrap. `_generic`
        (quants.c:452-453, 461-462, 470-471) computes `uint8_t q = byte *
        pow3[l]` (an 8-BIT multiply -- the truncation to uint8 IS the decode, and
        the high base-3 digit is then read by `xi = ((uint16_t)q * 3) >> 8`,
        value `xi - 1` in {-1,0,1}). pow3[6] = {1,3,9,27,81,243}. The emitter
        mirrors this with a vector pipeline that PRESERVES the wrap order:
        `vmul.vx u8 (byte * pow3[l])` (the low-8-bit truncation = the wrap) THEN
        `vwmulu.vx u8->u16 (q * 3)` THEN `vsrl 8` THEN narrow u16->u8 (xi in
        {0,1,2}, lossless) THEN reinterpret i8 THEN `vadd.vx -1` -> the ternary
        lane. The pow3 multiply must NOT be fused into the widening multiply
        (that would skip the mod-256 wrap and be wrong for any byte where
        byte*pow3 >= 256). pow3[l] is emitted as a per-l literal.

    (3) THREE accumulation loops into ONE integer accumulator + a SINGLE-SCALE
        SCALAR fp32 FOLD. `_generic` walks three index regions of the q8_K quant
        stream -- (a) the qs main loop (j=0, l 0..4, m 0..31 -> q8 index
        `l*32+m`, weight `qs[m]`), (b) the qs tail loop (j=32, l 0..4, m 0..15 ->
        q8 index `160 + l*16+m`, weight `qs[32+m]`), (c) the qh loop (l 0..3,
        j 0..3 -> q8 index `240 + l*4+j`, weight `qh[j]`) -- each contributing
        `(xi-1) * q8` to ONE `int sum`. Every q8 index 0..255 is covered exactly
        once, so the emitter decodes all three regions into an element-ordered
        int8_t aux8[256] (in q8 order) and then runs tq2_0's VERBATIM signed
        widening reduce (vle8 i8 x q8 i8 -> vwmul i16 -> vwredsum i32) into a
        single per-super-block integer accumulator (integer add is order-free).
        Then `sumf += (float)sum * (fp16(x.d) * y.d)` -- ONE fp16 read of the
        weight scale (xb+52), ONE fp32 read of the q8_K scale (yb+0), their
        product, then the scaled accumulate, as ONE emitc.expression matching
        `_generic`'s `sumf += (float) sum * (GGML_CPU_FP16_TO_FP32(x[i].d) *
        y[i].d)` (quants.c:476) so the fp32 byte-exactness is by construction.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime
    ABI value operands -- the tq1_0 weight base pointer (vx), the q8_K activation
    base pointer (vy), the fp32 output pointer `*s` (a `float *`), the runtime
    element count (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32
    LMUL m1 !tcrv_rvv.vector token whose lowering writes the fp32 dot result `*s`.

    The structural facts -- QK_K=256, the AoS block strides (54 bytes for
    block_tq1_0, 292 bytes for block_q8_K), the tq1_0 weight byte offsets (qs @0,
    qh @48, d @52) and the q8_K payload byte offsets (the fp32 activation scale d
    @0, qs @4) -- are carried as typed integer/string attributes (I4 mirror
    only). The bounded surface kind is limited to "ggml_tq1_0_q8_k_block_dot" and
    scale_model to "ternary-base3-single-fp16-scale-i32-domain-scalar-fp32-fold".
    The op identity is the dispatch key. Route ids, helper names, ABI strings,
    artifact names, status fields, exact intrinsic spellings, operand mirror
    strings, q8/tq1_0 names, and common EmitC/export code must not infer or
    choose super-block dot-product semantics.

    tq1_0 emits the byte-exact form for the VLEN>=128 capability class (rv64gcv
    => Zvl128b => VLEN>=128, the validated target), with NO VLEN-robust re-strip
    loop (the zve32x/zve64x re-strip is a capability item, not built here). No
    resource shape knobs yet; this op carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_qh_byte_offset,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ4XSQ8KOp
    : TCRVRVV_Op<"iq4_xs_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ4_XS x Q8_K super-block FULL block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_iq4_xs_q8_K` super-block dot-product as
    ONE bounded generic plugin-local typed RVV dataflow step -- the CODEBOOK class's
    SUPER-BLOCK rung: iq4_xs is the super-block variant of iq4_nl, REUSING iq4_nl's
    16-entry non-linear int8 CODEBOOK (the SAME `kvalues_iq4nl[16]` vrgather lookup)
    while wrapping it in the q4_K-style super-block 6-bit scale machinery and the
    q8_K activation. It COMPOSES two mechanisms already in hand:
    - iq4_nl's codebook-gather integer core (the 4-bit nibble does NOT decode to a
      linear value but INDEXES the 16-entry non-linear int8 codebook via
      `vrgather_vv_i8m1`, the SAME table broadcast into a vreg ONCE above the block
      loop), AND
    - the q4_K/q6_K super-block structure (QK_K=256, 8 sub-blocks of 32 elements,
      the q8_K activation with its leading fp32 scale).
    The ONE genuinely-new super-block piece -- the iq4_xs scale -- is a per-sub-block
    SIGNED 6-bit scale extracted from `scales_l[4]` + `scales_h` (the q4_K-style
    cross-byte bit dance) biased by -32, applied in the FLOAT domain (NOT q6_K's
    integer aux32 domain): `ls_j = scale_j - 32`, then per sub-block `d1 = d4d8 *
    (float)ls_j` and `sumf += d1 * (float)sumi_j`. There is NO min term (symmetric,
    like q6_K).

    (1) The super-block scale d4d8 = GGML_FP16_TO_FP32(x.d) * y.d is computed ONCE
        per super-block (the fp16 weight scale at byte 0 via the `(float)*(const
        _Float16 *)` seam, times the fp32 q8_K activation scale at byte 0, a plain
        float load), reproducing `_generic`'s `d4d8`.

    (2) For each of the 8 sub-blocks (a FLAT loop over j=0..7, mirroring ggml's
        ib+=2 pair loop but unrolled to a single sub-block step), the SIGNED 6-bit
        scale `ls_j` is extracted by the q4_K-style cross-byte bit dance in its
        CLOSED form (byte-exact equal to `_generic`'s progressive `h>>=4`):
        `ls_j = ((scales_l[j/2] >> (4*(j%2))) & 0xf) | (((scales_h >> (2*j)) & 0x3)
        << 4)`, all scalar STRUCTURED emitc bitwise ops (shift/and/or), then the
        per-sub-block float scale `d1 = d4d8 * (float)(ls_j - 32)` is computed as a
        SEPARATE op (mirroring `_generic`'s named `d1`/`d2` statements -- the
        `(ls-32)` round and the d4d8 multiply stay separate from the sumi multiply
        so NO FMA fuses them under `-ffp-contract`).

    (3) The per-sub-block integer dot `sumi_j` REUSES iq4_nl's codebook-gather core
        VERBATIM (seeded 0 per sub-block): one `vsetvl_e8m1(16)` over the 16 nibble
        bytes at `qs + j*16`, split into the two UNSIGNED index lanes (`vand 0x0F`
        low / `vsrl 0x04` high), GATHER each through the broadcast codebook table
        (`vrgather_vv_i8m1`), feed the SAME asymmetric signed widening product
        against the plain-i8 q8_K halves (`vwmul` low <-> q8[j*32 + 0..15], `vwmacc`
        + high <-> q8[j*32 + 16..31]) -> `vwredsum` -> i32 `sumi`. iq4_nl's `sumi`
        already equals `_generic`'s `sumi1 + sumi2` (the reduction combines the low
        and high halves; integer add is order-free).

    (4) The per-sub-block fold `sumf = sumf + d1 * (float)sumi` is ONE
        emitc.expression so the compiler fuses the SAME FMA `_generic` does under
        `-ffp-contract=on/default`, and is invoked in STRICT ascending sub-block
        order (j=0..7 inside ascending super-block ibl) so fp non-associativity is
        byte-exact across all four `-ffp-contract` modes. `*s = sumf`.

    The CODEBOOK GATHER is inherently a Zvl128b legality fact (the broadcast
    `values` register's VLMAX must be >= 16 to index all 16 table entries), so the
    integer core pins the m1 anchor (vsetvl_e8m1(16) caps the active vl at 16 when
    VLMAX >= 16, the elided strip form correct for ANY VLEN >= 128). The verifier
    rejects a non-m1 anchor fail-closed (I7). A board with VLEN < 128 cannot legally
    host the codebook gather; that is an N1 capability-legality divergence.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the iq4_xs weight base pointer (vx), the q8_K activation base
    pointer (vy), the fp32 output pointer `*s` (a `float *`), the runtime element
    count (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the single fp32 reduction result.

    The block-quantized super-block STRUCTURE is first-class in the typed body. The
    structural facts -- QK_K=256, 8 sub-blocks of 32 elements, the AoS block strides
    (136 bytes for block_iq4_xs, 292 bytes for block_q8_K), the iq4_xs payload byte
    offsets (the fp16 super-block scale d @0, scales_h @2, scales_l[4] @4, qs[128]
    @8) and the q8_K payload byte offsets (the fp32 activation scale d @0, qs @4),
    the dual scale model, and the 16-entry non-linear int8 CODEBOOK itself -- are
    carried as typed integer/string/array attributes (I4 mirror only). The codebook
    is a `DenseI8ArrayAttr:$codebook` of EXACTLY 16 int8 entries (the SAME table as
    iq4_nl); it is a STRUCTURAL FACT consumed structurally into the realized body
    (emitted as a `static const int8_t[16]` decl + the `vle8` table load), NOT an
    arithmetic decode, NOT a string plan read. The op identity is the dispatch key.
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/iq4 names, and common
    EmitC/export code must not infer or choose super-block dot-product semantics.

    iq4_xs emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target), with NO VLEN-robust re-strip loop
    (the zve32x/zve64x re-strip is a capability item, not built here). No resource
    shape knobs yet; this op carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$weight_scales_h_byte_offset,
    I64Attr:$weight_scales_l_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    DenseI8ArrayAttr:$codebook
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ2XXSQ8KOp
    : TCRVRVV_Op<"iq2_xxs_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ2_XXS x Q8_K GRID-codebook super-block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_iq2_xxs_q8_K` super-block dot-product as
    ONE bounded generic plugin-local typed RVV dataflow step -- the FIRST member of
    the DEEP IQ tail: the GRID-codebook class. This is a genuinely NEW mechanism, NOT
    the 16-entry per-nibble gather (iq4_nl/iq4_xs): each weight byte INDEXES a large
    packed GRID codebook (`iq2xxs_grid[256]`, a uint64 table where each entry encodes
    8 int8 values), the per-element SIGN is read from a separate SIGN PLANE
    (`ksigns_iq2xs[128]` / `kmask_iq2xs[8]`), and a per-group 4-bit scale folds in the
    INTEGER domain. It REUSES the q6_K-style super-block structure (the q8_K
    activation, the integer-domain `bsum` accumulation, the per-super-block fp32 fold).

    Block format (ggml-common.h): block_iq2_xxs = { ggml_half d; uint16_t qs[QK_K/8] }
    (= {fp16 d; uint16_t qs[32]}, stride 66), the q8_K activation unchanged (stride
    292, fp32 d @0, int8 qs @4). QK_K=256, 8 sub-blocks of 32 elements (ib32=0..7).

    (1) Per super-block d = (float)*(const _Float16 *)(xb) * *(const float *)(yb)
        (the fp16 weight scale times the fp32 q8_K scale, ONE mul), computed once.
        int32_t bsum = 0 reset per super-block.

    (2) For each of the 8 sub-blocks (a FLAT unrolled loop over ib32=0..7), read the
        two aux uint32 words from qs + ib32*8 by REASSEMBLING each from 4 little-
        endian byte loads (the qs stream starts at byte 2 so the aux words are
        2-aligned, NOT 4-aligned -- the byte reassembly is alignment-safe structured
        scalar emitc, no misaligned `*(uint32_t*)`): aux0 packs the 4 grid byte
        INDICES (aux8[0..3] = the low 4 bytes), aux1 packs the SIGN selectors + the
        scale. The per-sub-block scale `ls = 2*(aux1 >> 28) + 1` and the integer dot
        `sumi = 0`. For each of the 4 groups l=0..3: the grid INDEX `idx = aux8[l]`,
        the sign byte `signs = ksigns_iq2xs[(aux1 >> 7*l) & 127]`, the 8 grid int8
        values `grid = (const int8_t *)(iq2xxs_grid + idx)` (an indexed pointer-arith
        lookup `grid_table_i8 + idx*8`, NOT a vrgather -- the 2048-byte table cannot
        broadcast into a vreg), are decoded over 8 lanes: broadcast `signs`, `vand`
        with the kmask vector {1,2,4,8,16,32,64,128}, `vmsne 0` -> mask, `vmerge(grid,
        vneg(grid), mask)` applies the per-lane sign, then `vwmul(grid_signed, q8)` ->
        i16, `vwredsum` accumulates (seed = the running sumi i32m1, integer add is
        order-free) into the chained i32 reduction. After the 4 groups `bsum +=
        sumi * ls`.

    (3) The per-super-block fp32 fold `sumf = sumf + d * (float)bsum` is ONE
        emitc.expression (the SAME FMA `_generic` does under -ffp-contract on/default;
        the `(float)bsum` cast is explicit) invoked in STRICT ascending super-block
        order so fp non-associativity is byte-exact across all four -ffp-contract
        modes. The final `*s = 0.125f * sumf` (the iq2_xxs trailing 1/8 factor) is a
        SEPARATE statement OUTSIDE the accumulate expression.

    The grid lookup is an indexed vle8(8) over a pointer (`grid_table + idx*8`), the
    operative vector width is 8 (one group), so there is NO vrgather table-index
    legality fact and the integer core does NOT pin an m1 table anchor. The grid +
    ksigns tables are STRUCTURAL facts (I4 mirror) carried as typed array attributes
    and emitted as structured `static const` decls (the grid as `int64_t[256]`
    rendering ggml's exact uint64 literals, read via a `(const int8_t *)` cast; ksigns
    as a `uint8_t[128]` decl); kmask is the inline const {1<<j}. They are consumed
    STRUCTURALLY into the realized body (decls + indexed loads + the vle8 grid load),
    NOT an arithmetic decode, NOT a string plan read. The op identity is the dispatch
    key. Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/iq2 names, and common EmitC/export
    code must not infer or choose super-block dot-product semantics.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the iq2_xxs weight base pointer (vx), the q8_K activation base
    pointer (vy), the fp32 output pointer `*s` (a `float *`), the runtime element
    count (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the single fp32 reduction result.

    iq2_xxs emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target). No resource shape knobs this cut; the
    op carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    DenseI64ArrayAttr:$grid,
    DenseI32ArrayAttr:$ksigns
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ2XSQ8KOp
    : TCRVRVV_Op<"iq2_xs_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ2_XS x Q8_K GRID-codebook super-block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_iq2_xs_q8_K` super-block dot-product as
    ONE bounded generic plugin-local typed RVV dataflow step -- the SIBLING of the
    iq2_xxs GRID-codebook op, the SECOND member of the DEEP IQ tail. It REUSES the
    iq2_xxs grid + sign mechanism (each weight value INDEXES a packed GRID codebook
    where each uint64 entry encodes 8 int8 values, the per-element SIGN is read from a
    separate SIGN PLANE `ksigns_iq2xs[128]` / `kmask_iq2xs[8]`, and the per-group dot
    folds in the INTEGER domain) but with THREE structural deltas:
      (a) the GRID has 512 entries (`iq2xs_grid[512]`), indexed by a 9-bit index;
      (b) the index is NOT a packed aux byte: each `qs[]` is a uint16 word `q2[l]`
          whose LOW 9 bits = the grid index `q2[l] & 511` and HIGH 7 bits = the sign
          selector `q2[l] >> 9` -- there is NO aux1 word, NO `2*(aux1>>28)+1` packed
          scale, NO `(aux1>>7*l)&127` packed selector;
      (c) the per-sub-block 4-bit scales are an EXPLICIT array (`scales[QK_K/32]`, a
          new byte stream after qs): each `sc = scales[ib32]` carries TWO 4-bit scales
          -- `ls1 = 2*(sc & 0xf) + 1` applied to groups l=0,1, `ls2 = 2*(sc >> 4) + 1`
          applied to groups l=2,3 -- so each sub-block has TWO `sumi` resets and TWO
          `bsum += sumi*ls` contributions (NOT one chained 4-group reduction).
    It REUSES the q6_K-style super-block structure (the q8_K activation, the integer
    `bsum` accumulation, the per-super-block fp32 fold, the trailing 0.125f factor).

    Block format (ggml-common.h): block_iq2_xs = { ggml_half d; uint16_t qs[QK_K/8];
    uint8_t scales[QK_K/32] } (= {fp16 d; uint16_t qs[32]; uint8_t scales[8]}, stride
    74 = 2 + 64 + 8), the q8_K activation unchanged (stride 292, fp32 d @0, int8 qs
    @4). QK_K=256, 8 sub-blocks of 32 elements (ib32=0..7).

    (1) Per super-block d = (float)*(const _Float16 *)(xb) * *(const float *)(yb)
        (the fp16 weight scale times the fp32 q8_K scale, ONE mul), computed once.
        int32_t bsum = 0 reset per super-block.

    (2) For each of the 8 sub-blocks (a FLAT unrolled loop over ib32=0..7), read the
        scale byte `sc = scales[ib32]` (`ls1 = 2*(sc & 0xf)+1`, `ls2 = 2*(sc>>4)+1`).
        For the two halves h=0 (groups l=0,1, scale ls1) and h=1 (groups l=2,3, scale
        ls2): `sumi = 0`; for each group l of the half: the uint16 weight word
        `w = q2[l]` is reassembled from 2 little-endian byte loads (the qs stream
        starts at byte 2 so the words are 2-aligned, alignment-safe structured scalar
        emitc, no misaligned `*(uint16_t*)`), the grid INDEX `idx = w & 511`, the sign
        byte `signs = ksigns_iq2xs[w >> 9]`, the 8 grid int8 values `grid =
        (const int8_t *)(iq2xs_grid + idx)` (an indexed pointer-arith lookup
        `grid_table_i8 + idx*8`, NOT a vrgather -- the 4096-byte table cannot
        broadcast into a vreg), are decoded over 8 lanes: broadcast `signs`, `vand`
        with the kmask vector {1,2,4,8,16,32,64,128}, `vmsne 0` -> mask,
        `vmerge(grid, vneg(grid), mask)` applies the per-lane sign, then
        `vwmul(grid_signed, q8)` -> i16, `vwredsum` accumulates (seed = the running
        sumi i32m1, integer add is order-free) into the chained i32 reduction; q8
        advances 8 each group CONTINUOUSLY across the half boundary. After the half:
        `bsum += sumi * (h==0 ? ls1 : ls2)`.

    (3) The per-super-block fp32 fold `sumf = sumf + d * (float)bsum` is ONE
        emitc.expression (the SAME FMA `_generic` does under -ffp-contract on/default;
        the `(float)bsum` cast is explicit) invoked in STRICT ascending super-block
        order so fp non-associativity is byte-exact across all four -ffp-contract
        modes. The final `*s = 0.125f * sumf` (the iq2_xs trailing 1/8 factor) is a
        SEPARATE statement OUTSIDE the accumulate expression.

    The grid lookup is an indexed vle8(8) over a pointer (`grid_table + idx*8`), the
    operative vector width is 8 (one group), so there is NO vrgather table-index
    legality fact and the integer core does NOT pin an m1 table anchor. The grid +
    ksigns tables are STRUCTURAL facts (I4 mirror) carried as typed array attributes
    and emitted as structured `static const` decls (the grid as `int64_t[512]`
    rendering ggml's exact uint64 literals, read via a `(const int8_t *)` cast; ksigns
    as a `uint8_t[128]` decl); kmask is the inline const {1<<j}. They are consumed
    STRUCTURALLY into the realized body (decls + indexed loads + the vle8 grid load),
    NOT an arithmetic decode, NOT a string plan read. The op identity is the dispatch
    key. Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/iq2 names, and common EmitC/export
    code must not infer or choose super-block dot-product semantics.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the iq2_xs weight base pointer (vx), the q8_K activation base
    pointer (vy), the fp32 output pointer `*s` (a `float *`), the runtime element
    count (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the single fp32 reduction result.

    iq2_xs emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target). No resource shape knobs this cut; the
    op carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    DenseI64ArrayAttr:$grid,
    DenseI32ArrayAttr:$ksigns
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ2SQ8KOp
    : TCRVRVV_Op<"iq2_s_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ2_S x Q8_K GRID-codebook super-block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_iq2_s_q8_K` super-block dot-product as
    ONE bounded generic plugin-local typed RVV dataflow step -- the THIRD member of
    the DEEP IQ tail and the SIBLING of the iq2_xxs / iq2_xs GRID-codebook ops. It
    REUSES the GRID-codebook + sign mechanism (each weight value INDEXES a packed
    GRID codebook where each uint64 entry encodes 8 int8 values, the per-element SIGN
    selects a sign-flip via the kmask `{1<<j}` sign-bit plane, and the per-group dot
    folds in the INTEGER domain) but with THREE structural deltas vs iq2_xs:
      (a) the GRID has 1024 entries (`iq2s_grid[1024]`), indexed by a 10-bit index;
      (b) the index is assembled from TWO sources: the low 8 bits are a single byte
          `qs[l]` from the index region of qs[64], and the high 2 bits are injected
          from a per-sub-block qh byte -- `idx = qs[l] | ((qh[ib32] << (8-2*l)) &
          0x300)` (the q5_K-style qh-plane pattern; each of the 4 groups l=0..3 takes
          its 2 high bits from a different field of the SAME qh[ib32] byte). There is
          NO uint16 qs word, NO 9-bit `w & 511` read, NO `w >> 9` sign selector;
      (c) the signs are EXPLICIT -- read DIRECTLY from a sign-byte region that lives
          INSIDE qs[64] at byte offset QK_K/8 (`signs = qs + 32`, i.e. the upper half
          of the 64-byte qs array), NOT through the `ksigns_iq2xs[128]` lookup table.
          `signs[l]` is the raw sign byte for group l; bit j (via the SAME kmask
          `{1<<j}`) selects the per-lane sign. There is NO ksigns plane.
    It REUSES the explicit per-sub-block 4-bit scales[] array (`scales[QK_K/32]`):
    each `sc = scales[ib32]` carries TWO 4-bit scales -- `ls1 = 2*(sc & 0xf) + 1`
    applied to groups l=0,1, `ls2 = 2*(sc >> 4) + 1` applied to groups l=2,3 -- so
    each sub-block has TWO `sumi` resets and TWO `bsum += sumi*ls` contributions. It
    REUSES the q6_K-style super-block structure (the q8_K activation, the integer
    `bsum` accumulation, the per-super-block fp32 fold, the trailing 0.125f factor).

    Block format (ggml-common.h): block_iq2_s = { ggml_half d; uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32] } (= {fp16 d; uint8_t qs[64];
    uint8_t qh[8]; uint8_t scales[8]}, stride 82 = 2 + 64 + 8 + 8). The qs[64] array
    holds 32 index bytes (@2) then 32 sign bytes (@34); qh[8] @66; scales[8] @74. The
    q8_K activation is unchanged (stride 292, fp32 d @0, int8 qs @4). QK_K=256, 8
    sub-blocks of 32 elements (ib32=0..7).

    (1) Per super-block d = (float)*(const _Float16 *)(xb) * *(const float *)(yb)
        (the fp16 weight scale times the fp32 q8_K scale, ONE mul), computed once.
        int32_t bsum = 0 reset per super-block.

    (2) For each of the 8 sub-blocks (a FLAT unrolled loop over ib32=0..7), read the
        scale byte `sc = scales[ib32]` (`ls1 = 2*(sc & 0xf)+1`, `ls2 = 2*(sc>>4)+1`)
        and the qh byte `qh[ib32]`. For the two halves h=0 (groups l=0,1, scale ls1)
        and h=1 (groups l=2,3, scale ls2): `sumi = 0`; for each group l of the half:
        the 10-bit grid INDEX `idx = qs_idx[ib32*4 + l] | ((qh[ib32] << (8-2*l)) &
        0x300)` is assembled (one index byte + 2 qh bits), the explicit sign byte
        `signs = qs_sign[ib32*4 + l]` is loaded DIRECTLY, the 8 grid int8 values
        `grid = (const int8_t *)(iq2s_grid + idx)` (an indexed pointer-arith lookup
        `grid_table_i8 + idx*8`, NOT a vrgather -- the 8192-byte table cannot
        broadcast into a vreg) are decoded over 8 lanes: broadcast `signs`, `vand`
        with the kmask vector {1,2,4,8,16,32,64,128}, `vmsne 0` -> mask,
        `vmerge(grid, vneg(grid), mask)` applies the per-lane sign, then
        `vwmul(grid_signed, q8)` -> i16, `vwredsum` accumulates (seed = the running
        sumi i32m1, integer add is order-free) into the chained i32 reduction; q8
        advances 8 each group CONTINUOUSLY across the half boundary. After the half:
        `bsum += sumi * (h==0 ? ls1 : ls2)`.

    (3) The per-super-block fp32 fold `sumf = sumf + d * (float)bsum` is ONE
        emitc.expression (the SAME FMA `_generic` does under -ffp-contract on/default;
        the `(float)bsum` cast is explicit) invoked in STRICT ascending super-block
        order so fp non-associativity is byte-exact across all four -ffp-contract
        modes. The final `*s = 0.125f * sumf` (the iq2_s trailing 1/8 factor) is a
        SEPARATE statement OUTSIDE the accumulate expression.

    The grid lookup is an indexed vle8(8) over a pointer (`grid_table + idx*8`), the
    operative vector width is 8 (one group), so there is NO vrgather table-index
    legality fact and the integer core does NOT pin an m1 table anchor. The grid
    table is a STRUCTURAL fact (I4 mirror) carried as a typed array attribute and
    emitted as a structured `static const int64_t[1024]` decl (rendering ggml's exact
    uint64 literals, read via a `(const int8_t *)` cast); the signs are a memory
    region (no table); kmask is the inline const {1<<j}. They are consumed
    STRUCTURALLY into the realized body (decls + indexed loads + the vle8 grid load),
    NOT an arithmetic decode, NOT a string plan read. The op identity is the dispatch
    key. Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/iq2 names, and common EmitC/export
    code must not infer or choose super-block dot-product semantics.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the iq2_s weight base pointer (vx), the q8_K activation base
    pointer (vy), the fp32 output pointer `*s` (a `float *`), the runtime element
    count (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the single fp32 reduction result.

    iq2_s emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target). No resource shape knobs this cut; the
    op carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_signs_byte_offset,
    I64Attr:$weight_qh_byte_offset,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    DenseI64ArrayAttr:$grid
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ3XXSQ8KOp
    : TCRVRVV_Op<"iq3_xxs_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ3_XXS x Q8_K GRID-codebook super-block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_iq3_xxs_q8_K` super-block dot-product as
    ONE bounded generic plugin-local typed RVV dataflow step -- a member of the DEEP
    IQ tail and the SIBLING of the iq2_xxs GRID-codebook op. It REUSES the
    GRID-codebook + SIGN-plane mechanism verbatim (a packed GRID codebook indexed by a
    weight byte, the per-element SIGN read from the `ksigns_iq2xs[128]` / `kmask_iq2xs`
    sign plane, the per-group 4-bit scale folding in the INTEGER domain) with ONE
    load-bearing structural delta vs iq2_xxs:

      (a) the IQ3 GRID has 256 entries of FOUR int8 values (`iq3xxs_grid[256]`, a
          uint32 table where each entry encodes 4 int8 values -- HALF the lane width of
          iq2_xxs's uint64 grid-of-8). So the per-index decode is 4 values (the grid
          lookup loads 4 bytes per index: `grid_i8 + idx*4` then vle8(4)), and an
          8-element sign group is assembled from TWO indices -- `grid1 =
          iq3xxs_grid[q3[2l+0]]` (lanes 0..3, kmask{1,2,4,8}) and `grid2 =
          iq3xxs_grid[q3[2l+1]]` (lanes 4..7, kmask{16,32,64,128}).

    The aux-word layout REUSES iq2_xxs's packed form: a per-sub-block `aux32` (one
    uint32) holds the 4 sign selectors (`(aux32 >> 7l) & 127`, indexing the SAME
    `ksigns_iq2xs` table) plus the 4-bit scale in its high nibble (`ls = 2*(aux32>>28)
    + 1`). The ONE structural difference in the byte layout: the grid INDICES and the
    aux words live in SEPARATE regions of qs[96] -- the first `QK_K/4 = 64` bytes are
    the grid index stream `q3` (8 index bytes per sub-block), the remaining 32 bytes
    are the aux stream `gas` (one uint32 per sub-block) -- whereas iq2_xxs interleaves
    the indices and the aux pair. The integer dot, the ksigns sign plane, and the scale
    extraction are otherwise iq2_xxs's mechanism.

    Block format (ggml-common.h): block_iq3_xxs = { ggml_half d; uint8_t qs[3*QK_K/8] }
    (= {fp16 d; uint8_t qs[96]}, stride 98 = 2 + 96). The qs[96] array holds the 64
    grid index bytes (@2) then the 32 aux bytes (@66 = 2 + QK_K/4). The q8_K activation
    is unchanged (stride 292, fp32 d @0, int8 qs @4). QK_K=256, 8 sub-blocks of 32
    elements (ib32=0..7).

    (1) Per super-block d = (float)*(const _Float16 *)(xb) * *(const float *)(yb)
        (the fp16 weight scale times the fp32 q8_K scale, ONE mul), computed once.
        int32_t bsum = 0 reset per super-block.

    (2) For each of the 8 sub-blocks (a FLAT unrolled loop over ib32=0..7), read the
        per-sub-block aux uint32 `aux32` from `gas + ib32*4` by REASSEMBLING it from 4
        little-endian byte loads (the gas region starts at block byte 66, so the aux
        words are 2-aligned, NOT 4-aligned -- the byte reassembly is alignment-safe
        structured scalar emitc, no misaligned `*(uint32_t*)`). The per-sub-block scale
        `ls = 2*(aux32 >> 28) + 1` and the integer dot `sumi = 0`. For each of the 4
        groups l=0..3: the sign byte `signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]`, the
        two grid INDICES `idx1 = q3[8*ib32 + 2*l + 0]`, `idx2 = q3[8*ib32 + 2*l + 1]`.
        The 4 grid int8 values of each index `grid1 = (const int8_t *)(iq3xxs_grid +
        idx1)`, `grid2 = (const int8_t *)(iq3xxs_grid + idx2)` (an indexed pointer-arith
        lookup `grid_table_i8 + idx*4`, NOT a vrgather -- the 1024-byte table cannot
        broadcast into a vreg) are decoded as TWO 4-lane passes: pass A (grid1 vs
        q8[0..3], kmask{1,2,4,8}) and pass B (grid2 vs q8[4..7], kmask{16,32,64,128}).
        Each pass: broadcast `signs`, `vand` with the kmask vector, `vmsne 0` -> mask,
        `vmerge(grid, vneg(grid), mask)` applies the per-lane sign, then
        `vwmul(grid_signed, q8)` -> i16, `vwredsum` accumulates (seed = the running
        sumi i32m1, integer add is order-free) into the chained i32 reduction; q8
        advances 8 each group. After the 4 groups `bsum += sumi * ls`.

    (3) The per-super-block fp32 fold `sumf = sumf + d * (float)bsum` is ONE
        emitc.expression (the SAME FMA `_generic` does under -ffp-contract on/default;
        the `(float)bsum` cast is explicit) invoked in STRICT ascending super-block
        order so fp non-associativity is byte-exact across all four -ffp-contract
        modes. The final `*s = 0.25f * sumf` (the iq3_xxs trailing 1/4 factor -- a
        DIFFERENT trailing factor from iq2_xxs's 1/8) is a SEPARATE statement OUTSIDE
        the accumulate expression.

    The grid lookup is an indexed vle8(4) over a pointer (`grid_table + idx*4`), the
    operative vector width is 4 (one grid-of-4 entry), so there is NO vrgather
    table-index legality fact and the integer core does NOT pin an m1 table anchor. The
    grid + ksigns tables are STRUCTURAL facts (I4 mirror) carried as typed array
    attributes and emitted as structured `static const` decls (the grid as a
    `uint32_t[256]` rendering ggml's exact uint32 literals, read via a `(const int8_t
    *)` cast; ksigns as a `uint8_t[128]` decl); kmask is the inline const {1<<j}. They
    are consumed STRUCTURALLY into the realized body (decls + indexed loads + the vle8
    grid load), NOT an arithmetic decode, NOT a string plan read. The op identity is
    the dispatch key. Route ids, helper names, ABI strings, artifact names, status
    fields, exact intrinsic spellings, operand mirror strings, q8/iq3 names, and common
    EmitC/export code must not infer or choose super-block dot-product semantics.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the iq3_xxs weight base pointer (vx), the q8_K activation base
    pointer (vy), the fp32 output pointer `*s` (a `float *`), the runtime element
    count (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the single fp32 reduction result.

    iq3_xxs emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target). No resource shape knobs this cut; the
    op carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_gas_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    DenseI32ArrayAttr:$grid,
    DenseI32ArrayAttr:$ksigns
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ3SQ8KOp
    : TCRVRVV_Op<"iq3_s_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ3_S x Q8_K GRID-codebook super-block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_iq3_s_q8_K` super-block dot-product as
    ONE bounded generic plugin-local typed RVV dataflow step -- a member of the DEEP
    IQ tail and a direct RE-COMPOSITION of pieces already built by the GRID-codebook
    siblings. iq3_s does NOT introduce a brand-new mechanism; it COMBINES three
    mechanisms verbatim:

      (a) the iq3 GRID-of-4 codebook (from iq3_xxs): each grid entry is FOUR int8
          values (`iq3s_grid[512]`, a uint32 table), so the per-index decode is 4
          values (the grid lookup loads 4 bytes per index: `grid_i8 + idx*4` then
          vle8(4)), and an 8-element sign group is assembled from TWO grid indices --
          grid1 (lanes 0..3, kmask{1,2,4,8}) and grid2 (lanes 4..7, kmask{16,32,64,
          128}) -- exactly iq3_xxs's two-4-lane-pass grid-of-4. The ONLY GRID delta vs
          iq3_xxs is the table SIZE: 512 entries (the 9-bit index), not 256.

      (b) the qh 9th-bit plane (from iq2_s): the grid index is assembled from a single
          qs index byte plus ONE high bit injected from a per-sub-block qh byte. Unlike
          iq2_s's 2-bit `& 0x300` plane, iq3_s injects a SINGLE bit (mask 256 = bit 8)
          and the two grid indices of a group take DIFFERENT shifts -- grid1 idx =
          `qs[2l+0] | ((qh[ib32] << (8-2*l)) & 256)`, grid2 idx = `qs[2l+1] | ((qh[ib32]
          << (7-2*l)) & 256)`. So the qh byte contributes bit 8 of each of the 8 grid
          indices in a sub-block.

      (c) the EXPLICIT signs read from memory (from iq2_s): the per-group sign byte is
          read DIRECTLY from a dedicated `signs[QK_K/8]` region (a SEPARATE 32-byte
          array at byte offset 74, NOT inside qs, and NOT a `ksigns_iq2xs[128]` lookup).
          `signs[ib32*4 + l]` is the raw sign byte for group l; bit j (via the SAME
          kmask `{1<<j}`) selects the per-lane sign. There is NO ksigns plane.

    The EXPLICIT per-sub-block 4-bit scales are REUSED from iq2_s: ggml packs two 4-bit
    scales per byte of `scales[QK_K/64]` (4 bytes), `sc = scales[ib32/2]`, `ls1 =
    2*(sc & 0xf) + 1` for even sub-blocks, `ls2 = 2*(sc >> 4) + 1` for odd sub-blocks.
    In the FLAT 8-sub-block emission each sub-block sb reads `scales[sb/2]` and selects
    the low nibble (sb even) or high nibble (sb odd). It REUSES the q6_K-style
    super-block structure (the q8_K activation, the integer `bsum` accumulation, the
    per-super-block fp32 fold). The trailing factor is `*s = sumf` (NO 1/8 or 1/4
    factor -- iq3_s applies none).

    Block format (ggml-common.h): block_iq3_s = { ggml_half d; uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32]; uint8_t signs[QK_K/8]; uint8_t scales[QK_K/64] } (= {fp16 d;
    uint8_t qs[64]; uint8_t qh[8]; uint8_t signs[32]; uint8_t scales[4]}, stride 110 =
    2 + 64 + 8 + 32 + 4). The qs[64] array holds 64 grid index bytes (@2, one per the 8
    grid indices of each of the 8 sub-blocks); qh[8] @66 (one byte per sub-block);
    signs[32] @74 (4 sign bytes per sub-block); scales[4] @106 (two 4-bit scales per
    byte). The q8_K activation is unchanged (stride 292, fp32 d @0, int8 qs @4).
    QK_K=256, 8 sub-blocks of 32 elements (ib32=0..7).

    (1) Per super-block d = (float)*(const _Float16 *)(xb) * *(const float *)(yb)
        (the fp16 weight scale times the fp32 q8_K scale, ONE mul), computed once.
        int32_t bsum = 0 reset per super-block.

    (2) For each of the 8 sub-blocks (a FLAT unrolled loop over ib32=0..7), read the
        scale byte `sc = scales[ib32/2]` and select `ls = ib32 even ? 2*(sc & 0xf)+1 :
        2*(sc >> 4)+1`, the qh byte `qh[ib32]`, and `sumi = 0`. For each of the 4
        groups l=0..3: the explicit sign byte `signs = sgn[ib32*4 + l]` (read DIRECTLY
        from the signs region at xb+74), the two grid INDICES idx1 = `qs[8*ib32 + 2*l +
        0] | ((qh[ib32] << (8-2*l)) & 256)`, idx2 = `qs[8*ib32 + 2*l + 1] | ((qh[ib32]
        << (7-2*l)) & 256)`. The 4 grid int8 values of each index `grid1 = (const int8_t
        *)(iq3s_grid + idx1)`, `grid2 = (const int8_t *)(iq3s_grid + idx2)` (an indexed
        pointer-arith lookup `grid_table_i8 + idx*4`, NOT a vrgather -- the 2048-byte
        table cannot broadcast into a vreg) are decoded as TWO 4-lane passes: pass A
        (grid1 vs q8[0..3], kmask{1,2,4,8}) and pass B (grid2 vs q8[4..7], kmask{16,32,
        64,128}). Each pass: broadcast `signs`, `vand` with the kmask vector, `vmsne 0`
        -> mask, `vmerge(grid, vneg(grid), mask)` applies the per-lane sign, then
        `vwmul(grid_signed, q8)` -> i16, `vwredsum` accumulates (seed = the running sumi
        i32m1, integer add is order-free) into the chained i32 reduction; q8 advances 8
        each group. After the 4 groups `bsum += sumi * ls`.

    (3) The per-super-block fp32 fold `sumf = sumf + d * (float)bsum` is ONE
        emitc.expression (the SAME FMA `_generic` does under -ffp-contract on/default;
        the `(float)bsum` cast is explicit) invoked in STRICT ascending super-block
        order so fp non-associativity is byte-exact across all four -ffp-contract
        modes. The final `*s = sumf` (iq3_s applies NO trailing factor) is a SEPARATE
        statement OUTSIDE the accumulate expression.

    The grid lookup is an indexed vle8(4) over a pointer (`grid_table + idx*4`), the
    operative vector width is 4 (one grid-of-4 entry), so there is NO vrgather
    table-index legality fact and the integer core does NOT pin an m1 table anchor. The
    grid table is a STRUCTURAL fact (I4 mirror) carried as a typed array attribute and
    emitted as a structured `static const uint32_t[512]` decl (rendering ggml's exact
    uint32 literals, read via a `(const int8_t *)` cast -- every entry < 0x80000000 so
    a positive int32 carries the value losslessly, and every grid byte <= 15 < 128 so
    the signed int8 read yields the identical value as ggml's uint8 read); the signs
    are a memory region (no table); kmask is the inline const {1<<j}. They are consumed
    STRUCTURALLY into the realized body (decls + indexed loads + the vle8 grid load),
    NOT an arithmetic decode, NOT a string plan read. The op identity is the dispatch
    key. Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/iq3 names, and common EmitC/export
    code must not infer or choose super-block dot-product semantics.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the iq3_s weight base pointer (vx), the q8_K activation base
    pointer (vy), the fp32 output pointer `*s` (a `float *`), the runtime element
    count (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the single fp32 reduction result.

    iq3_s emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target). No resource shape knobs this cut; the
    op carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_qh_byte_offset,
    I64Attr:$weight_signs_byte_offset,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    DenseI32ArrayAttr:$grid
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ1SQ8KOp
    : TCRVRVV_Op<"iq1_s_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ1_S x Q8_K TERNARY-grid super-block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_iq1_s_q8_K` super-block dot-product as
    ONE bounded generic plugin-local typed RVV dataflow step -- the TERNARY class, the
    last common ggml dot kernel (its sibling iq1_m is the only remaining tail). It
    REUSES the GRID-codebook lookup mechanism (each weight value INDEXES a packed
    uint64 GRID where each entry encodes 8 int8 values, and the per-group dot folds in
    the INTEGER domain) but introduces a GENUINELY NEW mechanism -- the per-block DELTA
    term -- with these structural deltas vs the iq2/iq3 grid siblings:

      (a) the GRID is TERNARY and 2048 entries (`iq1s_grid[2048]`, a uint64 table).
          Each grid byte is one of {0x00, 0x01, 0xff} = {0, +1, -1}; byte-viewed as a
          `const int8_t *` it directly carries the ternary value with its sign (0xff
          reads as -1), so there is NO sign-bit plane, NO kmask, NO vmsne/vand/vneg/
          vmerge sign apply -- the grid IS the signed value (this is the only grid
          family whose stored bytes are themselves negative). The index is an 11-bit
          value `idx = qs[ib*4 + l] | (((qh[ib] >> (3*l)) & 7) << 8)` (range to 2047):
          the low 8 bits are a single qs index byte, the high 3 bits are taken from a
          different 3-bit field of the per-sub-block uint16 `qh[ib]` for each group
          l=0..3.

      (b) the SCALE lives in the qh word: `ls = 2*((qh[ib] >> 12) & 7) + 1` (bits
          12..14), ONE scale per sub-block applied to all 4 groups (NOT an explicit
          two-nibble scales[] split; iq1_s carries no scales[] array).

      (c) the per-block DELTA term (the NEW mechanism): the qh sign bit (bit 15)
          selects a per-sub-block delta `delta = (qh[ib] & 0x8000) ? -1 : 1`, and the
          IQ1S_DELTA = 0.125 constant biases each value by `+delta*IQ1S_DELTA`. ggml
          folds the delta contribution through the q8 sub-block sums (`bsums`): per
          sub-block `sumi1 += ls * delta * (bsums[2*ib+0] + bsums[2*ib+1])`. So the
          activation `int16 bsums[QK_K/16]` region (offset 260) is READ for the first
          time in the IQ tail. The grid dot accumulates `sumi += ls * lsum`. The two
          integer accumulators are KEPT SEPARATE across all 8 sub-blocks.

    It REUSES the q6_K-style super-block structure (the q8_K activation, the integer
    accumulation, the per-super-block fp32 fold) BUT folds BOTH integer sums and the
    0.125 delta constant in ONE fp expression: `sumf += d * ((float)sumi +
    IQ1S_DELTA * (float)sumi1)` -- the 0.125 is applied EXACTLY ONCE per super-block
    inside the fold (NOT a global trailing factor, and NOT per-sub-block), and `*s =
    sumf` (NO trailing multiply -- the iq1_s store is the bare accumulator).

    Block format (ggml-common.h): block_iq1_s = { ggml_half d; uint8_t qs[QK_K/8];
    uint16_t qh[QK_K/32] } (= {fp16 d; uint8_t qs[32]; uint16_t qh[8]}, stride 50 =
    2 + 32 + 16). The qs[32] array holds 4 grid index bytes per sub-block (@2); qh[8]
    is a uint16 array @34 (one word per sub-block carrying the 4 grid-high-3-bit fields
    in bits 0..11, the 3-bit scale in bits 12..14, the delta sign in bit 15). The q8_K
    activation is unchanged for the quants (stride 292, fp32 d @0, int8 qs @4) and the
    int16 bsums region is read @260. QK_K=256, 8 sub-blocks of 32 elements (ib=0..7).

    (1) Per super-block d = (float)*(const _Float16 *)(xb) * *(const float *)(yb)
        (the fp16 weight scale times the fp32 q8_K scale, ONE mul), computed once.
        int32_t sumi = 0 and int32_t sumi1 = 0 reset per super-block.

    (2) For each of the 8 sub-blocks (a FLAT unrolled loop over ib=0..7), read the qh
        word `qhw = qh[ib]` (a uint16 load): `ls = 2*((qhw >> 12) & 7) + 1`, `delta =
        1 - 2*((qhw >> 15) & 1)` (= +1 or -1). The grid dot `lsum = 0`: for each of the
        4 groups l=0..3 the 11-bit grid INDEX `idx = qs[ib*4 + l] | (((qhw >> (3*l)) &
        7) << 8)` is assembled (one index byte + 3 qh bits), the 8 grid int8 values
        `grid = (const int8_t *)(iq1s_grid + idx)` (an indexed pointer-arith lookup
        `grid_table_i8 + idx*8`, NOT a vrgather -- the 16384-byte table cannot broadcast
        into a vreg) are loaded over 8 lanes, `vwmul(grid, q8)` -> i16 (the grid bytes
        are ALREADY signed ternary, NO sign apply), `vwredsum` accumulates (seed = the
        running lsum i32m1, integer add is order-free) into the chained i32 reduction;
        q8 advances 8 each group. After the 4 groups `sumi += ls * lsum`. The delta
        term reads the two int16 bsums `bsums[2*ib+0]` and `bsums[2*ib+1]` and folds
        `sumi1 += ls * delta * (bsums[2*ib+0] + bsums[2*ib+1])` (all integer domain).

    (3) The per-super-block fp32 fold `sumf = sumf + d * ((float)sumi + 0.125f *
        (float)sumi1)` is ONE emitc.expression (the SAME contraction `_generic` does
        under -ffp-contract on/default/fast; the `(float)sumi` and `(float)sumi1` casts
        are explicit, the inner add stays in the expression tree) invoked in STRICT
        ascending super-block order so fp non-associativity is byte-exact across all
        four -ffp-contract modes. The final `*s = sumf` (iq1_s applies NO trailing
        factor) is a SEPARATE statement OUTSIDE the accumulate expression.

    The grid lookup is an indexed vle8(8) over a pointer (`grid_table + idx*8`), the
    operative vector width is 8 (one group), so there is NO vrgather table-index
    legality fact and the integer core does NOT pin an m1 table anchor. The grid table
    is a STRUCTURAL fact (I4 mirror) carried as a typed array attribute and emitted as a
    structured `static const int64_t[2048]` decl (rendering ggml's exact uint64
    literals, read via a `(const int8_t *)` cast -- the ternary bytes 0xff read as the
    signed value -1, byte-identical to ggml's own `(const int8_t *)` grid read). They
    are consumed STRUCTURALLY into the realized body (decls + indexed loads + the vle8
    grid load), NOT an arithmetic decode, NOT a string plan read. The op identity is the
    dispatch key. Route ids, helper names, ABI strings, artifact names, status fields,
    exact intrinsic spellings, operand mirror strings, q8/iq1 names, and common
    EmitC/export code must not infer or choose super-block dot-product semantics.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the iq1_s weight base pointer (vx), the q8_K activation base
    pointer (vy), the fp32 output pointer `*s` (a `float *`), the runtime element
    count (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the single fp32 reduction result.

    iq1_s emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target). No resource shape knobs this cut; the
    op carries only the structural facts.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_qh_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$activation_bsums_byte_offset,
    DenseI64ArrayAttr:$grid
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ1MQ8KOp
    : TCRVRVV_Op<"iq1_m_q8_k_block_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ1_M x Q8_K TERNARY-grid super-block dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_iq1_m_q8_K` super-block dot-product as
    ONE bounded generic plugin-local typed RVV dataflow step -- the LAST common ggml
    dot kernel (its sibling iq1_s is already covered; after iq1_m EVERY ggml_vec_dot_*
    is covered). It REUSES the iq1_s TERNARY-grid codebook lookup (the SAME 2048-entry
    `iq1s_grid` table, byte-viewed as `const int8_t *` so the 0xff bytes read as -1 --
    NO sign plane, NO kmask, NO vmsne/vand/vneg/vmerge) and the per-block DELTA
    mechanism (the IQ1M_DELTA = 0.125 bias), but introduces three GENUINELY NEW pieces
    vs iq1_s:

      (a) NO fp16 `d` field in the weight block. The super-block scale d is
          RECONSTRUCTED from a packed `iq1m_scale_t` union spread across the FOUR
          uint16 `scales[]` words: the 16 fp16 bits are reassembled as
          `scale.u16 = (sc[0]>>12) | ((sc[1]>>8)&0x00f0) | ((sc[2]>>4)&0x0f00) |
          (sc[3]&0xf000)` and then READ AS `_Float16` (a bit reinterpret, NOT a
          numeric conversion -- mirroring ggml's union). On little-endian this is
          byte-identical to ggml. The per-sub-block 3-bit scales live in the SAME
          scales[] words' low bits: `ls1 = 2*((sc[ib/2]>>(6*(ib%2)+0))&7)+1`,
          `ls2 = 2*((sc[ib/2]>>(6*(ib%2)+3))&7)+1` (a TWO-half scale split per
          sub-block, ls1 for groups 0..1, ls2 for groups 2..3).

      (b) the qh plane is `uint8 qh[QK_K/16]` (16 bytes, TWO per sub-block) rather than
          iq1_s's uint16 qh[8]. The 11-bit grid index for group l is `idx = qs[l] |
          (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)` (the index-high 3 bits come
          from bits 0..2 of qh[l/2] for even l, bits 4..6 for odd l).

      (c) the DELTA is per HALF-sub-block group (8 elements), with FOUR independent
          signs per sub-block: `delta[0]=qh[0]&0x08`, `delta[1]=qh[0]&0x80`,
          `delta[2]=qh[1]&0x08`, `delta[3]=qh[1]&0x80` (sign per group l from qh[l/2]
          bit 3 for even l, bit 7 for odd l). Because the four signs are INDEPENDENT
          the delta term CANNOT be folded through the q8 `bsums` (the 16-element sums)
          like iq1_s did; instead each group computes a FRESH `lsum2 = Σ q8[j]` (8
          elements) reduced separately, scalar-multiplied by `delta[l]`, accumulated
          into `sum2[l/2]`. So iq1_m reads NO bsums region; the second accumulator is
          built from a parallel `vwredsum` of the SAME q8 vector already loaded for the
          grid dot.

    The integer fold mirrors iq1_s: per super-block `sumi1 += sum1[0]*ls1 +
    sum1[1]*ls2` (the grid dot, sum1[h] = Σ_group∈h Σ grid·q8) and `sumi2 +=
    sum2[0]*ls1 + sum2[1]*ls2` (the delta term, sum2[h] = Σ_group∈h delta·Σq8). The
    per-super-block fp32 fold `sumf += d * ((float)sumi1 + IQ1M_DELTA*(float)sumi2)`
    is ONE emitc.expression (the 0.125 applied EXACTLY ONCE to (float)sumi2 inside the
    fold), and `*s = sumf` (NO trailing factor).

    Block format (ggml-common.h): block_iq1_m = { uint8_t qs[QK_K/8]; uint8_t
    qh[QK_K/16]; uint8_t scales[QK_K/32] } = {qs[32]@0; qh[16]@32; scales[8]@48},
    stride 56 (= 32+16+8, NO fp16 d field). The q8_K activation is unchanged (stride
    292, fp32 d @0, int8 qs @4); iq1_m reads NO bsums. QK_K=256, 8 sub-blocks of 32
    elements (ib=0..7).

    The grid is the SAME structural fact (I4 mirror) as iq1_s -- a typed array attr
    emitted as a structured `static const uint64_t[2048]` decl, read via a `(const
    int8_t *)` byte cast (the ternary 0xff bytes read as -1, byte-identical to ggml).
    Consumed STRUCTURALLY (decls + indexed loads + the vle8 grid load), NOT an
    arithmetic decode, NOT a string plan read. The op identity is the dispatch key.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime ABI
    value operands -- the iq1_m weight base pointer (vx), the q8_K activation base
    pointer (vy), the fp32 output pointer `*s` (a `float *`), the runtime element count
    (n) -- plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the single fp32 reduction result.

    iq1_m emits the byte-exact form for the VLEN>=128 capability class (rv64gcv =>
    Zvl128b => VLEN>=128, the validated target). No resource shape knobs this cut.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_qh_byte_offset,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    DenseI64ArrayAttr:$grid
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlVecScaleF32Op
    : TCRVRVV_Op<"ggml_vec_scale_f32",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml f32 in-place elementwise scale (y[i] *= v) op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_scale_f32` forward-pass op as ONE bounded
    generic plugin-local typed RVV f32 elementwise dataflow step: the in-place
    per-lane multiply `y[i] = y[i] * v` over a flat unit-stride f32 buffer. This
    is the FIRST forward-pass (non-dot) op family member -- a DIFFERENT shape
    from the block-quantized integer dot-product ops (no AoS block loop, no
    packed-int decode, no integer widening, no per-block fp16 scale): a single
    VLEN-robust `vsetvl_e32m<L>` strip loop over the f32 lanes, `vle32` /
    `vfmul_vf` (scalar broadcast) / `vse32`, matching ggml's hand-written RVV
    path (vec.h:733-739, anchored at m8). The op is nested directly under
    tcrv_rvv.with_vl and consumes THREE runtime ABI value operands -- the f32
    buffer base pointer `y` (read AND written in place), the runtime f32 scalar
    `v`, the runtime element count `n` -- plus the active !tcrv_rvv.vl token. It
    produces ONE f32 LMUL !tcrv_rvv.vector token whose lowering is the in-place
    strip store (no separate output buffer; `y` is both input and output).

    The op makes the f32 elementwise in-place scale STRUCTURE first-class in the
    typed body. Byte-exactness to ggml's real op is UNCONDITIONAL: `y[i] *= v` is
    a bare per-lane fp32 multiply, so there is NO fused multiply-add (so
    `-ffp-contract` cannot perturb it) and NO cross-lane reduction (so the LMUL /
    tail / strip-count are correctness-free resource/scheduling knobs, never the
    *what*). The fixed emission -- the strip loop, the vle32/vfmul_vf/vse32
    intrinsic chain via emitc.call_opaque (the one sanctioned opaque piece,
    exactly as the dot kernels emit their intrinsics) -- is the op's lowering,
    NOT a candidate-mirror string read in the conversion. For the bounded surface
    `kind` is limited to "ggml_vec_scale_f32". Route ids, helper names, ABI
    strings, artifact names, status fields, exact intrinsic spellings, and common
    EmitC/export code must not infer or choose f32 elementwise scale semantics.

    The optional `strip_lmul` attribute is a bounded resource/scheduling fact --
    it selects the LMUL the f32 strip loop anchors at ("m1", "m2", "m4", or "m8";
    default "m8" to match ggml's hand-written path). It is purely the *how*
    (vector grouping / strip width), never the *what*: the result is byte-exact
    at any anchor because every lane is multiplied by the same `v` and the strip
    loop re-strips correctly for any VLEN via the runtime `vsetvl_e32m<L>(n-i)`.
    (It is named `strip_lmul`, not the with_vl/setvl `lmul` spelling, so the op
    keeps no SEW/LMUL/policy dataflow-parameter attr at the I5 boundary -- the
    sibling block-dot op uses the same convention with `integer_core_lmul`.)
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$scalar,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    OptionalAttr<StrAttr>:$strip_lmul
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$buffer `,` $scalar `,` $element_count `,` $vl "
    "attr-dict `:` type($buffer) `,` type($scalar) `,` "
    "type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlRmsNormF32Op
    : TCRVRVV_Op<"ggml_rms_norm_f32",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml f32 rms_norm (Sx^2 reduce -> rsqrt -> scale) op";
  let description = [{
    Records the COMPLETE ggml `ggml_compute_forward_rms_norm_f32` (non-fused, no
    weight) forward-pass op for ONE row as a bounded generic plugin-local typed
    RVV f32 dataflow step: the row reduction `mean = (Sum_i x[i]^2)/ne00`, the
    scalar `scale = 1/sqrtf(mean + eps)`, and the per-lane normalize
    `y[i] = x[i] * scale` over flat unit-stride f32 buffers. This is the FIRST
    forward-pass REDUCTION op (F3 in research/forward-pass-scoping.md) -- a NEW
    shape vs the elementwise scale (F1) and vs the block-quantized integer dot
    ops: it brings in a cross-lane fold plus a scalar rsqrt before the
    elementwise scale.

    BYTE-EXACTNESS hinges on matching ggml's EXACT accumulation method
    (ops.cpp:3791-3798). ggml folds `Sum_i x[i]^2` in `ggml_float` (= double),
    SCALAR, in strict ASCENDING index order (the source loop is explicitly NOT
    vectorized -- "worth switching to explicit SIMD?"). The cast chain is
    load-bearing: each product `x[i]*x[i]` rounds in f32 FIRST, is then WIDENED
    to double, and accumulated in double -- NOT `(double)x[i]*(double)x[i]`. The
    f32->double widen between the f32 product and the double add is also an FMA
    barrier (multiply and add are different types, so -ffp-contract cannot fuse
    them). Then `mean = (float)(sum / (double)ne00)` (divide in double, cast to
    f32 AFTER), `scale = 1.0f / sqrtf(mean + eps)` (the add/sqrt/reciprocal all
    in f32 -- sqrtf-then-reciprocal, NOT a hardware fast-rsqrt). A vectorized
    f32 `vfredusum` reduction would NOT be byte-exact (it folds in f32 with a
    tree order), so the reduction is emitted as STRUCTURED scalar emitc nodes
    (variable/load/mul/cast/add/assign) replicating ggml's scalar double fold,
    and only the final `y[i] = x[i] * scale` normalize is vectorized (a bare
    per-lane vfmul_vf strip -- no FMA, no reduction, byte-exact at any LMUL).

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime
    ABI value operands -- the f32 input base pointer `x` (read-only), the f32
    output base pointer `y` (written), the runtime f32 `eps`, the runtime
    element count `ne00` -- plus the active !tcrv_rvv.vl token. It produces ONE
    f32 LMUL m1 !tcrv_rvv.vector token whose lowering is the normalize strip
    store. For the bounded surface `kind` is limited to "ggml_rms_norm_f32".
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, and common EmitC/export code must not infer or choose
    f32 rms_norm semantics.

    The optional `strip_lmul` attribute is a bounded resource/scheduling fact --
    it selects the LMUL the NORMALIZE strip loop anchors at ("m1", "m2", "m4",
    or "m8"; default "m8" to match ggml's `ggml_vec_scale_f32` apply path). It is
    purely the *how* of the vectorized normalize (every lane multiplied by the
    same scalar `scale`), never the *what* (the reduction stays scalar-double
    regardless). It is named `strip_lmul`, not the with_vl/setvl `lmul` spelling,
    so the op keeps no SEW/LMUL/policy dataflow-parameter attr at the I5 boundary
    -- the sibling ops use the same convention.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$output,
    AnyType:$eps,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    OptionalAttr<StrAttr>:$strip_lmul
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$input `,` $output `,` $eps `,` $element_count `,` $vl "
    "attr-dict `:` type($input) `,` type($output) `,` type($eps) `,` "
    "type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlVecSiluF32Op
    : TCRVRVV_Op<"ggml_vec_silu_f32",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml f32 silu (y[i] = x[i]*sigmoid(x[i])) op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_silu_f32` forward-pass op (the FFN
    activation `y[i] = x[i] * sigma(x[i])`, `sigma(x) = 1/(1 + e^{-x})`;
    vec.cpp:380 + `ggml_v_silu_m2` vec.h:1363) as ONE bounded generic
    plugin-local typed RVV f32 dataflow step. This is the FIRST forward-pass
    VECTORIZED TRANSCENDENTAL op (F5 in research/forward-pass-scoping.md) -- a
    NEW shape vs the elementwise scale (F1) and the rms_norm reduction (F3): the
    silu needs ggml's EXACT vectorized minimax exp polynomial (`ggml_v_expf_m2`,
    vec.h:1324) replicated node-for-node, NOT a libm `expf` call.

    BYTE-EXACTNESS hinges on replicating ggml's EXACT vectorized `ggml_v_expf_m2`
    intrinsic chain. ggml computes `e^x` with a fully vectorized minimax
    polynomial built ENTIRELY from `__riscv_v` intrinsics -- it makes NO libm
    call. Every step (the `0x1.8p23f` round trick, the two-term Cayley range
    reduction via `vfnmsac` x2, the degree-5 Estrin polynomial in `vfmacc`, the
    integer exponent `vsll<<23 + 0x3f800000` reinterpret, and the
    overflow/underflow `vmerge` fixup) is a vector intrinsic with a FIXED bit
    pattern constant. Replicating that chain node-for-node -- with the identical
    magic constants (`0x1.8p23f`, `0x1.715476p+0f`, `0x1.62e4p-1f`,
    `0x1.7f7d1cp-20f`, the polynomial coefficients, `0x3f800000`, the
    `126.0f`/`192.0f`/`0x82000000`/`0x7f000000` fixup) -- makes the result
    BIT-IDENTICAL to ggml's vectorized silu. ggml's `if (!vcpop_m(c))` is a pure
    performance short-circuit (the fast path `k + j*k` equals the slow path's
    c-false lane `k + k*j` bit-for-bit by fp multiply commutativity, and RISC-V
    yields the canonical NaN regardless of operand order), so the lowering emits
    the slow path UNCONDITIONALLY as a straight-line `vmerge` value graph -- same
    output bits for every input (normal, saturating, NaN/inf/denormal), no
    data-dependent branch.

    The op is nested directly under tcrv_rvv.with_vl and consumes THREE runtime
    ABI value operands -- the f32 input base pointer `x` (read-only), the f32
    output base pointer `y` (written), the runtime element count `n` -- plus the
    active !tcrv_rvv.vl token. It produces ONE f32 LMUL m1 !tcrv_rvv.vector token
    whose lowering is the per-strip silu store. The strip loop and the polynomial
    are fixed at m2 (matching ggml's `vsetvl_e32m2` path and the m2-tied mask
    `vbool16_t`/reinterpret `vuint32m2_t` types) -- silu is purely elementwise so
    other LMULs would also be per-lane exact, but the op pins m2 to match the
    oracle and avoid mask-width combinatorics; there is no `strip_lmul` knob this
    cut. For the bounded surface `kind` is limited to "ggml_vec_silu_f32". Route
    ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, and common EmitC/export code must not infer or choose
    f32 silu semantics.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$input `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($input) `,` type($output) `,` "
    "type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlVecSoftMaxF32Op
    : TCRVRVV_Op<"ggml_vec_soft_max_f32",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml f32 soft_max (y[i]=e^{x[i]-max}; returns sum) op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_soft_max_f32` forward-pass op (the
    attention softmax core: `y[i] = e^{x[i] - max}`, returning the f64 sum
    `Sum_i e^{x[i]-max}`; vec.cpp:531 + the `__riscv_v_intrinsic` path
    vec.cpp:584-592) as ONE bounded generic plugin-local typed RVV f32 dataflow
    step. This is the F5b forward-pass op (research/forward-pass-scoping.md): it
    COMBINES F5's vectorized transcendental (the exact `ggml_v_expf_m2` minimax
    exp polynomial) with a NEW reduction shape -- the f64 widening reduce
    accumulated across strips.

    BYTE-EXACTNESS faithful to ggml's REAL bare `ggml_vec_soft_max_f32`, NOT the
    surrounding caller. The function takes `max` as an INPUT (ggml's caller
    computes it with the SCALAR `ggml_vec_max_f32` and passes it in; ops.cpp:5400)
    and DOES NOT normalize (ggml's caller applies `1/sum` afterward via
    `ggml_vec_scale_f32`; ops.cpp:5415-5416). The bare function only writes
    `y[i] = e^{x[i]-max}` and RETURNS the f64 sum (vec.cpp:531). The exact RVV
    method (vec.cpp:584-592):
      vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
      for (avl over n) {
        vfloat32m2_t val = ggml_v_expf_m2(vfsub_vf(vle32(x), max), avl);
        vse32(y, val, avl);
        vsum = vfwredusum_vs_f32m2_f64m1(val, vsum, avl);   // f32->f64 reduce
      }
      return (double)vfmv_f_s_f64m1(vsum);
    The sum is accumulated in `ggml_float` = DOUBLE via the WIDENING reduce
    `vfwredusum_vs_f32m2_f64m1` into a single f64m1 accumulator carried across
    strips -- NOT the scalar-ascending double fold F3 (rms_norm) uses. Matching
    THAT exact widening-reduce fold (a loop-carried opaque-vector accumulator) is
    the byte-exactness crux for the returned sum. The `exp(x-max)` uses the SAME
    node-for-node `ggml_v_expf_m2` chain F5 silu emits (the shared
    `emitGgmlVExpfM2` replication), so the written `y[]` and the per-strip val are
    bit-identical to ggml. -inf masked entries (attention masks) flow through the
    polynomial's underflow path (`exp -> 0`) with no special-casing -- the
    max-subtraction (x-max with x=-inf gives -inf) plus the |n|>192 saturating
    branch yield 0 bit-exactly.

    The op is nested directly under tcrv_rvv.with_vl and consumes FOUR runtime
    ABI value operands -- the f32 output base pointer `y` (written), the
    read-only f32 input base pointer `x`, the runtime f32 scalar `max`, the
    runtime element count `n` -- plus the active !tcrv_rvv.vl token. It produces
    ONE f32 LMUL m1 !tcrv_rvv.vector token (the per-strip store boundary); the f64
    sum is the function's C return value (the only forward-pass op that RETURNS a
    scalar). The strip loop and the polynomial are fixed at m2 (matching ggml's
    `vsetvl_e32m2` path and the m2-tied mask/reinterpret types). For the bounded
    surface `kind` is limited to "ggml_vec_soft_max_f32". Route ids, helper names,
    ABI strings, artifact names, status fields, exact intrinsic spellings, and
    common EmitC/export code must not infer or choose f32 soft_max semantics.
  }];

  let arguments = (ins
    AnyType:$output,
    AnyType:$input,
    AnyType:$max,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$output `,` $input `,` $max `,` $element_count `,` $vl "
    "attr-dict `:` type($output) `,` type($input) `,` type($max) `,` "
    "type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlQuantizeRowQ80Op
    : TCRVRVV_Op<"quantize_row_q8_0",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml f32->block_q8_0 activation quantizer op";
  let description = [{
    Records the COMPLETE ggml `quantize_row_q8_0` RVV-path forward-pass op
    (the f32 -> block_q8_0 ACTIVATION QUANTIZER; riscv/quants.c:32-71) as ONE
    bounded generic plugin-local typed RVV dataflow step. This is the F4
    forward-pass op (research/forward-pass-scoping.md): the f32 -> QUANT BRIDGE
    that produces the block_q8_0 activation our committed q4_0_q8_0 / q8_0_q8_0
    block-dot kernels CONSUME. It is structurally NEW vs the elementwise/reduction
    f32 ops (F1/F3/F5/F5b) and vs the integer block-dot ops: a per-32-block
    max-abs REDUCTION (`vfredmax` of `vfabs`) + a scalar scale (`d = amax/127`,
    `id = d ? 1/d : 0`) + an f32->i16->i8 NARROWING CONVERT, written into an AoS
    block_q8_0 buffer (stride 34: the fp16 `d` at byte 0, the 32 int8 `qs` at
    byte 2).

    BYTE-EXACTNESS to the DEPLOYED kernel hinges on matching ggml's EXACT RVV
    method (riscv/quants.c:43-65), NOT the scalar `_ref` (which rounds with
    `roundf`, round-half-AWAY). On the `ssh rvv` board `__riscv_v` is defined, so
    the live `quantize_row_q8_0` IS the RVV path:
      vl = 32;  (= QK8_0; one strip per block, relying on Zvl128b => VLEN>=128 so
                 e32m8 holds all 32 lanes at once, the same capability the q4_0
                 mb4-elided shape uses)
      v_x  = vle32_v_f32m8(x + i*32, vl);
      amax = vfmv_f_s(vfredmax_vs_f32m8_f32m1(vfabs_v_f32m8(v_x),
                       vfmv_v_f_f32m1(0.0f, vl), vl));
      d  = amax / 127.0f;  id = d ? 1.0f/d : 0.0f;        // SCALAR f32
      y[i].d = (_Float16)d;                                // fcvt.h.s (rne)
      x0 = vfmul_vf_f32m8(v_x, id, vl);                    // scale
      vi = vfncvt_x_f_w_i16m4(x0, vl);                     // f32->i16 (frm = rne)
      vs = vncvt_x_x_w_i8m2(vi, vl);                       // i16->i8 (truncate)
      vse8_v_i8m2(y[i].qs, vs, vl);
    The ROUNDING crux: `vfncvt_x_f_w_i16m4` rounds with the dynamic frm =
    round-to-nearest-EVEN (NOT `roundf`'s round-half-away); replicating that exact
    intrinsic inherits rne + the i8 saturating clamp + every NaN/inf/edge case for
    free (do NOT reason about rounding -- emit the same intrinsic). The FP16 crux:
    the board is `__riscv_zfhmin`, so `GGML_CPU_FP32_TO_FP16(d)` is the native
    `(_Float16)d` cast (`fcvt.h.s`, rne); matching THAT (a structural `_Float16`
    store, NOT a software fp16 pack) is the second byte-exactness crux. The
    `d ? 1.0f/d : 0.0f` conditional is load-bearing: the all-zero block (amax=0 =>
    d=0) must give `id=0` (so every `q[i]=0`) -- a bare `1/d` would give
    `inf`/`NaN`; it is emitted as a STRUCTURED scalar `emitc.cmp` + `emitc.if`,
    NOT a raw string.

    The op is nested directly under tcrv_rvv.with_vl and consumes THREE runtime
    ABI value operands -- the read-only f32 input base pointer `x`
    (`const float *`), the block_q8_0 output BYTE buffer base pointer `vy`
    (`uint8_t *`, the AoS block buffer written), the runtime element count `n`
    (= ggml's `k`, n % 32 == 0 a ggml contract) -- plus the active !tcrv_rvv.vl
    token. It produces ONE f32 LMUL m1 !tcrv_rvv.vector token (the per-block store
    boundary). The block-format facts (`qk` = 32, `block_stride` = 34, the fp16
    `d` at offset 0, the int8 `qs` at offset 2) are the op's typed mirror attrs
    (I4); the emission is the op's fixed structure. For the bounded surface `kind`
    is limited to "ggml_quantize_row_q8_0". The strip is pinned at the m8 anchor
    ggml uses (the block's 32 lanes in one e32m8 strip -- there is no byte-exact
    LMUL freedom here: the reduction fold is m8-shaped, so there is NO strip_lmul
    knob this cut, matching silu's precedent). Route ids, helper names, ABI
    strings, artifact names, status fields, exact intrinsic spellings, and common
    EmitC/export code must not infer or choose f32->q8_0 quantizer semantics.

    q8_1 is the SIBLING quantizer (same amax/scale/narrow + an extra
    `vfredusum` block sum stored as `block_q8_1.s`); it is a separate op for a
    later increment, NOT folded into this kind.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind,
    I64Attr:$qk,
    I64Attr:$block_stride,
    I64Attr:$scale_byte_offset,
    I64Attr:$quant_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$input `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($input) `,` type($output) `,` "
    "type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlRopeNormF32Op
    : TCRVRVV_Op<"ggml_rope_norm_f32",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml f32 NORMAL rope (rotary position embedding) op";
  let description = [{
    Records the COMPLETE ggml `ggml_compute_forward_rope_f32` op for ONE attention
    head row, the GGML_ROPE_TYPE_NORMAL variant (the rope llama-2 / LLM_ARCH_LLAMA
    uses; llama-model.cpp:2387-2423 maps LLM_ARCH_LLAMA -> LLAMA_ROPE_TYPE_NORM,
    "a normal RoPE, operating on pairs of CONSECUTIVE head values"), as ONE
    bounded generic plugin-local typed RVV f32 dataflow step. This is the F6
    forward-pass op (research/forward-pass-scoping.md) -- the COMPOSITION rung:
    the angle cache is a verbatim SCALAR libm `cosf`/`sinf` prologue exactly as
    ggml's `ggml_rope_cache_init` (ops.cpp:5707-5721), and the per-pair rotation
    is f32 elementwise. With it the forward-pass primitive set is complete.

    The math (ggml NORMAL, the plain path: ext_factor=0, freq_scale=1,
    attn_factor=1, freq_factors=NULL, forward so sin_sign=+1, n_dims=ne0 = the
    full head_dim so there is no passthrough channel). Per dim-pair p
    (i0 = 2*p, p = 0..n_dims/2-1):
      theta_p = theta_base * theta_scale^p          (the ITERATIVE f32 recurrence
                                                     `theta *= theta_scale`, NOT a
                                                     closed-form pow, NOT double)
      cos_p = cosf(theta_p);  sin_p = sinf(theta_p) (SCALAR libm, once per pair)
      x0 = x[2p];  x1 = x[2p+1]                      (CONSECUTIVE pair; NORMAL)
      y[2p]   = x0*cos_p - x1*sin_p
      y[2p+1] = x0*sin_p + x1*cos_p
    `theta_base` (= the integer position pos, as f32) and `theta_scale`
    (= powf(freq_base, -2/n_dims)) are PRECOMPUTED runtime f32 inputs, so the
    kernel makes NO `powf` call -- the only libm calls are the per-pair
    `cosf`/`sinf`, matching ggml's `cache[]` bit-for-bit when linked against the
    SAME libm.

    BYTE-EXACTNESS has TWO distinct axes, surfaced honestly:
    (1) The angle transcendental is SCALAR libm `cosf`/`sinf` (NOT a vectorized
        polynomial like F5's `ggml_v_expf_m2`). Its exactness depends on linking
        the SAME libm ggml links: emit the angle prologue as the SAME scalar
        `cosf`/`sinf` calls (one emitc.call_opaque each, the sanctioned opaque
        seam -- NOT a raw string), so a same-libm deployment is bit-exact and a
        different libm is libm-tolerance bound on the angles ONLY.
    (2) The rotation `x0*cos - x1*sin` is `a*b - c*d`, an FP-contraction hazard:
        clang fuses `a*b - c*d` to an FMA (one rounding) under `-ffp-contract`
        on/fast and keeps two roundings under off. ggml's rotation
        (ops.cpp:5808-5809) is a SINGLE C expression, so it fuses under on/fast.
        The lowering GROUPS each output's `a*b - c*d` into ONE emitc.expression
        (the F3 rms_norm emitc.expression FMA-fix discipline), so mlir-translate
        renders it as ONE C statement TOKEN-IDENTICAL to ggml's -- clang then
        makes the IDENTICAL contraction decision under EVERY mode, so the kernel
        is byte-exact vs ggml under default/on/off/fast (NOT just off),
        independent of llama.cpp's build flag. Emitting the two products as
        separate statements would block intra-statement fusion and diverge from
        ggml under the default `on` -- so the grouping is load-bearing. The
        rotation is otherwise exact f32.

    The op is nested directly under tcrv_rvv.with_vl and consumes FIVE runtime
    ABI value operands -- the read-only f32 input base pointer `x`
    (`const float *`, one head row), the f32 output base pointer `y`
    (`float *`), the runtime f32 `theta_base` (the position as f32), the runtime
    f32 `theta_scale`, the runtime element count `n_dims` (= ne0, the head dim,
    n_dims % 2 == 0 a ggml contract) -- plus the active !tcrv_rvv.vl token. It
    produces ONE f32 LMUL m1 !tcrv_rvv.vector token (the per-pair store
    boundary). For the bounded surface `kind` is limited to
    "ggml_rope_norm_f32". Route ids, helper names, ABI strings, artifact names,
    status fields, exact intrinsic spellings, and common EmitC/export code must
    not infer or choose f32 rope semantics.

    There is no `strip_lmul` knob this cut: cos/sin are scalar libm (one call per
    pair), so the faithful structure IS ggml's -- a SCALAR per-pair loop that
    computes the angle and applies the rotation in the same body (vectorizing
    while cos/sin stay scalar would force gathering scalars into vectors with no
    byte-exactness gain), matching silu's no-knob precedent.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$output,
    AnyType:$theta_base,
    AnyType:$theta_scale,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$input `,` $output `,` $theta_base `,` $theta_scale `,` $element_count `,` "
    "$vl attr-dict `:` type($input) `,` type($output) `,` type($theta_base) "
    "`,` type($theta_scale) `,` type($element_count) `,` type($vl) "
    "`->` type($result)";
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

def DequantizeOp : TCRVRVV_Op<"dequantize",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "generic typed RVV i32-to-f32 runtime-scale dequantization op";
  let description = [{
    Records one bounded generic plugin-local RVV dequantization step. The op
    is nested directly under tcrv_rvv.with_vl, consumes one signed i32
    !tcrv_rvv.vector source, one explicitly imported runtime scale ABI value,
    and the active !tcrv_rvv.vl token, carries an explicit dequantization kind
    and relation, and produces one f32 !tcrv_rvv.vector result.

    The current bounded Stage 2 slice supports signed i32 LMUL m1 to f32 LMUL
    m1 with runtime f32 scale. Source dtype, result dtype, scale ABI role,
    SEW, LMUL, VL, policy, operation kind, and conversion relation are
    structural typed-body facts consumed by the RVV plugin. Route ids, helper
    names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, and common EmitC/export code must not infer or choose
    dequantization semantics.
  }];

  let arguments = (ins
    AnyType:$source,
    AnyType:$scale,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$dequant_relation
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$source `,` $scale `,` $vl attr-dict `:` type($source) `,` type($scale) `,` type($vl) `->` type($result)";
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

def MaskedStridedStoreOp : TCRVRVV_Op<"masked_strided_store",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "generic typed RVV masked byte-strided store dataflow op";
  let description = [{
    Records one bounded plugin-local RVV masked destination byte-strided store
    step. The op is nested directly under tcrv_rvv.with_vl, consumes one
    explicit output buffer runtime ABI value, one !tcrv_rvv.mask predicate,
    one !tcrv_rvv.vector payload value matching the enclosing config, one
    explicit runtime destination byte stride SSA value, and the active
    !tcrv_rvv.vl token. It carries explicit masked strided-store memory-form,
    byte-stride unit, and inactive-lane no-write preservation facts.

    For the current bounded Stage 2 computed-mask strided masked-store slice,
    memory_form is limited to masked-strided-store, stride_unit is limited to
    byte, and inactive_lane_policy is limited to
    preserve-output-on-false-lanes. False mask lanes, skipped strided
    destination slots, and tail lanes are structural masked-store behavior;
    common EmitC/export must not infer them from route ids, ABI names,
    artifact metadata, exact intrinsic spelling, or harness sentinels.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$mask,
    AnyType:$value,
    Index:$stride,
    AnyType:$vl,
    StrAttr:$memory_form,
    StrAttr:$stride_unit,
    StrAttr:$inactive_lane_policy
  );

  let assemblyFormat =
    "$buffer `,` $mask `,` $value `,` $stride `,` $vl attr-dict `:` type($buffer) `,` type($mask) `,` type($value) `,` type($stride) `,` type($vl)";
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
    value matching the enclosing config, one explicit runtime output or
    destination-byte stride SSA value, and the active !tcrv_rvv.vl token.
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
