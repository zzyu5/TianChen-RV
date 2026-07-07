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

    Additive loop-capable extension (M-FLAT W2): the OPTIONAL `block_index`
    operand + `block_stride` attribute make one generic load address a per-block
    strip live at `base + block_index*block_stride (+ quant_byte_offset)` inside
    the flat nb loop -- byte-exact to the monolith's blockBaseValue+loadI8. When
    `block_index` is ABSENT the op is the unchanged single-block ABI-base load
    (block_stride/quant_byte_offset must not be set, fail-closed); the existing
    single-block routes are untouched.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$vl,
    Optional<Index>:$block_index,
    OptionalAttr<I64Attr>:$block_stride,
    OptionalAttr<I64Attr>:$quant_byte_offset
  );
  let results = (outs AnyType:$loaded);

  let assemblyFormat =
    "$buffer `,` $vl (`block` $block_index^ `:` type($block_index))? attr-dict `:` type($buffer) `,` type($vl) `->` type($loaded)";
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
    signed_packed_i4_offset_binary_x_i8_product and product_relation to one of
    two LMUL rungs sharing the SAME decode + product structure:
    "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2" (the narrow integer-core anchor)
    or "offset-binary-i4m1-x-i8m1x2-to-i16m2" (the m1 flat-cohort rung).
    Source/result dtype/config, VL,
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

def UnsignedNibbleXI8ProductOp
    : TCRVRVV_Op<"unsigned_nibble_x_i8_product",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV unsigned-nibble asymmetric packed-i4 x plain-i8 product op";
  let description = [{
    Records one bounded generic plugin-local RVV ASYMMETRIC unsigned-nibble
    packed-i4 x plain-i8 widening product dataflow step. This is the integer
    core of the ggml Q4_1 x Q8_1 dot-product: ONLY the packed-i4 weight operand
    is nibble-decoded; the two int8 activation operands stay plain. The op is
    nested directly under tcrv_rvv.with_vl, consumes ONE typed UNSIGNED i8 LMUL
    m1 !tcrv_rvv.vector packed-i4 weight operand (each u8 packing two UNSIGNED
    4-bit nibbles in the [0,15] range), TWO typed SIGNED/SIGNLESS i8 LMUL m1
    !tcrv_rvv.vector PLAIN-int8 activation operands (the q8 low half paired with
    the low nibbles and the q8 high half paired with the high nibbles), and the
    active !tcrv_rvv.vl token. It carries an explicit operation kind and product
    relation, and produces ONE widened i16 LMUL m2 !tcrv_rvv.vector result
    holding the sum of the low-nibble and high-nibble asymmetric widening
    products.

    This op makes the UNSIGNED nibble decode + ONE-SIDED unpack + asymmetric
    i8(decoded) x i8(plain) product + low/high <-> q8 half split pairing
    first-class STRUCTURE in the typed body. It is DISTINCT from the
    offset-binary sibling tcrv_rvv.packed_i4_offset_binary_x_i8_product (which
    biases each nibble via an xor-0x88 `(nibble - 8)` decode) and from the
    codebook sibling tcrv_rvv.codebook_gather_x_i8_product (which gathers each
    nibble through a non-linear 16-entry table): here the nibble value IS the
    weight (q4_1 stores UNSIGNED quants, the `-8` shift folded into the block
    minimum), so the decode is a plain unsigned split with NO bias and NO gather.
    The packed weight low nibble pairs with the FIRST plain-i8 activation operand
    and the high nibble with the SECOND; the int8 activations are NOT
    nibble-unpacked. The fixed intrinsic chain (vand_vx_u8<L> 0x0F for the low
    nibble; vsrl_vx_u8<L> 0x04 for the high nibble; vreinterpret_v_u8<L>_i8<L>
    to feed the signed widening product; vwmul over i16<W> against the plain low
    activation; vwmacc over i16<W> against the plain high activation) is the op's
    lowering, NOT a candidate-mirror string read in the conversion. For the
    bounded surface kind is limited to unsigned_nibble_x_i8_product and
    product_relation to the SINGLE m1 rung
    "unsigned-nibble-i4m1-x-i8m1x2-to-i16m2" (i4m1 weight x i8m1 low/high
    activation -> i16m2). Source/result dtype/config, VL, the unsigned nibble
    decode and the one-sided unpack structure are structural in the selected RVV
    body. Route ids, helper names, ABI strings, artifact names, status fields,
    exact intrinsic spellings, operand-packing mirror strings, q8/q4 names, and
    common EmitC/export code must not infer or choose asymmetric unsigned-nibble
    packed-i4 x i8 product semantics.
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

def FiveBitOffsetBinaryXI8ProductOp
    : TCRVRVV_Op<"five_bit_offset_binary_x_i8_product",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV five-bit (nibble+qh) offset-binary x plain-i8 product op";
  let description = [{
    Records one bounded generic plugin-local RVV FIVE-BIT offset-binary packed
    weight x plain-i8 widening product dataflow step. This is the integer core of
    the ggml Q5_0 x Q8_0 dot-product: the packed weight is reconstructed from a
    4-bit nibble (loaded UNSIGNED u8) MERGED with a per-element 5th bit read from
    the scalar-domain qh field, then offset-binary biased `(five_bit - 16)` into a
    signed i8 weight; the two int8 activation operands stay plain. The op is nested
    directly under tcrv_rvv.with_vl and consumes FIVE operands: ONE typed UNSIGNED
    i8 LMUL m1 !tcrv_rvv.vector packed weight (each u8 packing two 4-bit nibbles),
    ONE scalar i32 qh_source token (the SSA result of a preceding
    tcrv_rvv.block_five_bit_qh_source brick, naming the 32-bit qh field bytes -- the
    5th-bit source is operand/source-driven, NOT a descriptor-baked byte offset on
    THIS op), TWO typed SIGNED/SIGNLESS i8 LMUL m1 !tcrv_rvv.vector PLAIN-int8
    activation operands (the q8 low half paired with the low nibbles, the q8 high
    half with the high nibbles), and the active !tcrv_rvv.vl token. It carries an
    explicit operation kind and product relation, and produces ONE widened i16 LMUL
    m2 !tcrv_rvv.vector result holding the sum of the low and high five-bit
    asymmetric widening products.

    This op makes the four-bit nibble decode + qh 5th-bit merge + offset-binary
    `-16` bias + asymmetric i8(decoded) x i8(plain) product + low/high <-> q8 half
    split pairing first-class STRUCTURE in the typed body. It is DISTINCT from the
    UNSIGNED-nibble sibling tcrv_rvv.unsigned_nibble_x_i8_product (which has NO 5th
    bit and NO bias) and from the offset-binary q4_0 sibling
    tcrv_rvv.packed_i4_offset_binary_x_i8_product (which biases a 4-bit nibble via
    xor-0x88 and has no qh field). The fixed intrinsic chain (vand_vx_u8<L> 0x0F /
    vsrl_vx_u8<L> 0x04 for the nibbles; the per-lane 5th-bit lane vid/vadd/vmv/vsrl/
    vand/vsll/narrow injected off the broadcast qh halves; vor into each nibble;
    vreinterpret u8->i8; vsub 16 offset-binary bias; vwmul over i16<W> against the
    plain low activation; vwmacc over i16<W> against the plain high activation) is
    the op's lowering, NOT a candidate-mirror string read in the conversion. For
    the bounded surface kind is limited to five_bit_offset_binary_x_i8_product and
    product_relation to the SINGLE m1 rung
    "five-bit-offset-binary-i4m1-x-i8m1x2-to-i16m2" (i4m1 weight + qh 5th bit x
    i8m1 low/high activation -> i16m2). Source/result dtype/config, VL, the nibble
    decode, the qh 5th-bit merge and the one-sided unpack structure are structural
    in the selected RVV body. Route ids, helper names, ABI strings, artifact names,
    status fields, exact intrinsic spellings, operand-packing mirror strings, q8/q5
    names, and common EmitC/export code must not infer or choose five-bit
    offset-binary packed x i8 product semantics.
  }];

  let arguments = (ins
    AnyType:$weight,
    AnyType:$qh_source,
    AnyType:$activation_low,
    AnyType:$activation_high,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$product_relation
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight `,` $qh_source `,` $activation_low `,` $activation_high `,` $vl "
    "attr-dict `:` type($weight) `,` type($qh_source) `,` "
    "type($activation_low) `,` type($activation_high) `,` type($vl) `->` "
    "type($result)";
  let hasVerifier = 1;
}

def CodebookTableBroadcastOp
    : TCRVRVV_Op<"codebook_table_broadcast",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV codebook lookup-table broadcast-into-vreg op";
  let description = [{
    Records one bounded generic plugin-local RVV codebook-table broadcast step:
    the 16-entry non-linear int8 lookup table is materialized as a structured
    `static const int8_t[16]` decl and broadcast-loaded ONCE into a typed i8
    !tcrv_rvv.vector result the codebook gather indexes. This is the codebook
    class's prerequisite STRUCTURE (the FP4 / iq4_nl `kvalues` table) made
    first-class in the typed body, distinct from the q4_0 offset-binary nibble
    decode which needs NO table. The op is nested directly under tcrv_rvv.with_vl,
    consumes NO SSA operands (the table is a compile-time constant), and produces
    ONE i8 LMUL !tcrv_rvv.vector result -- the broadcast `values` register the
    gather reuses.

    The 16-entry int8 codebook is carried as a DenseI8ArrayAttr:$codebook of
    EXACTLY 16 entries (the kvalues lookup range [0,15]); the C symbol the decl
    declares is the StrAttr:$table_symbol. The fixed lowering (the structured
    `static const int8_t <table_symbol>[16] = {...};` decl + the
    __riscv_vle8_v_i8<L> broadcast load of <table_symbol> at the 16-entry count)
    is the op's lowering, NOT a candidate-mirror string read in the conversion.
    The result LMUL is the codebook i8 anchor -- a VLEN-capability fact (m1 at
    VLEN128 where its VLMAX spans the 16-entry table, mf2 at VLEN256). Route ids,
    helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, and common EmitC/export code must not infer or choose codebook
    table-broadcast semantics.
  }];

  let arguments = (ins
    DenseI8ArrayAttr:$codebook,
    StrAttr:$table_symbol
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def CodebookGatherXI8ProductOp
    : TCRVRVV_Op<"codebook_gather_x_i8_product",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV codebook-gather packed-i4 x plain-i8 product op";
  let description = [{
    Records one bounded generic plugin-local RVV ASYMMETRIC codebook-gather
    packed-i4 x plain-i8 widening product dataflow step. This is the integer core
    of the ggml IQ4_NL / FP4 codebook x Q8_0 dot-product: ONLY the packed-i4
    weight operand is codebook-decoded; the two int8 activation operands stay
    plain. Unlike the q4_0 offset-binary sibling
    (tcrv_rvv.packed_i4_offset_binary_x_i8_product, an xor-0x88 + arithmetic
    decode), the 4-bit nibble does NOT decode to a linear value -- each nibble
    is an INDEX into a non-linear 16-entry int8 table. The op is nested directly
    under tcrv_rvv.with_vl, consumes ONE typed UNSIGNED i8 LMUL !tcrv_rvv.vector
    packed-i4 weight operand (each u8 packing two 4-bit table indices), TWO typed
    SIGNED i8 LMUL !tcrv_rvv.vector PLAIN-int8 activation operands (the q8 low
    half paired with the low nibbles, the q8 high half with the high nibbles),
    ONE typed SIGNED i8 LMUL !tcrv_rvv.vector codebook-table operand (the
    broadcast `values` from tcrv_rvv.codebook_table_broadcast), and the active
    !tcrv_rvv.vl token. It produces ONE widened i16 LMUL !tcrv_rvv.vector result
    holding the sum of the low-nibble and high-nibble asymmetric widening
    products.

    This op makes the codebook nibble split + gather decode + asymmetric
    i8(gathered) x i8(plain) product first-class STRUCTURE in the typed body. The
    decode splits each weight byte into the two UNSIGNED index lanes
    (vand_vx_u8<L> 0x0F low, vsrl_vx_u8<L> 0x04 high), GATHERS each through the
    broadcast table (vrgather_vv_i8<L>(values, idx)) into signed-i8 weight lanes,
    then feeds the SAME asymmetric widening product the offset-binary sibling uses
    (vwmul over i16<W> against the plain low activation; vwmacc over i16<W> against
    the plain high activation). The fixed intrinsic chain is the op's lowering,
    NOT a candidate-mirror string read in the conversion. For the bounded surface
    kind is limited to signed_codebook_gather_x_i8_product and product_relation to
    "codebook-gather-i8-x-i8x2-to-i16". The codebook i8 source LMUL drives the i16
    product LMUL (the VLEN-capability flip: m1->i16m2 at VLEN128, mf2->i16m1 at
    VLEN256). Source/result dtype/config, VL, and the codebook gather structure
    are structural in the selected RVV body. Route ids, helper names, ABI strings,
    artifact names, status fields, exact intrinsic spellings, operand-packing
    mirror strings, q8/q4 names, and common EmitC/export code must not infer or
    choose codebook-gather packed-i4 x i8 product semantics.
  }];

  let arguments = (ins
    AnyType:$weight,
    AnyType:$activation_low,
    AnyType:$activation_high,
    AnyType:$table,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$product_relation
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight `,` $activation_low `,` $activation_high `,` $table `,` $vl "
    "attr-dict `:` type($weight) `,` type($activation_low) `,` "
    "type($activation_high) `,` type($table) `,` type($vl) `->` type($result)";
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

    The repack-vs-block-dot ROUTING is driven by STRUCTURED OPPONENT FACTS the op
    carries, NOT by the `quant` format label:

    - **`opponent_vlen_native_floor` (optional i64).** The minimum VLEN at/above
      which ggml ships a VLEN-native hand-tuned vec_dot the repack rewrite LOSES
      to (q4_K = 128: per-VLEN dispatch, @128 inline RVV asm). ABSENT means no
      such kernel at any VLEN (q4_0/q8_0 ship only a non-VLEN-specialized body the
      measured repack out-streams). Fact 1 of the selection.

    - **`block_dot_compute_heavy` (optional bool).** Whether the plain block-dot
      is compute-heavy enough that the repacked out-of-block stream removes work
      (q4_0 = true: nibble decode + per-block vredsum + scattered reads; q8_0/absent
      = lean). Fact 2 of the selection.

    The `quant` attribute is now an OPTIONAL, purely human-readable format LABEL
    (a table-lookup entry point / provenance token). The
    `--tcrv-rvv-lower-quant-contraction` selection READS the structured facts
    above (plus the derived capability VLEN and `m_regime`) and NEVER the format
    label, so deleting `quant` and keeping the structured facts constructs the
    IDENTICAL concrete L2 contraction region: routing is fact-driven, the format
    string is just a label.

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
    OptionalAttr<StrAttr>:$quant,
    StrAttr:$scale_model,
    StrAttr:$m_regime,
    I64Attr:$qk,
    StrAttr:$weight_layout,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$quant_byte_offset,
    I64Attr:$activation_high_byte_offset,
    // PROVENANCE: these two fact VALUES are read off a PINNED ggml tree and provenance-
    // bound at experiments/sealed/c1-cleanliness/opponent-facts-provenance/opponent-facts.pin.json
    // (line anchors + pin_sha). STALE gate tools/lint/check_opponent_facts_pin.sh (CI:
    // .github/workflows/falsifier-gate.yml) fails closed if schema/ggml-pin.lock.json is
    // bumped away from that sha — re-verify the anchors before trusting these facts.
    OptionalAttr<I64Attr>:$opponent_vlen_native_floor,
    OptionalAttr<BoolAttr>:$block_dot_compute_heavy,
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

def GgmlRepackGemvQ50Q80Op
    : TCRVRVV_Op<"repack_gemv_q5_0_q8_0",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q5_0 x Q8_0 16x1-REPACKED GEMV (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q5_0 16x1-REPACKED GEMV hot kernel as ONE bounded
    generic plugin-local typed RVV dataflow step. It is the SINGLE-OUTPUT-COLUMN
    (decode) q5_0 sibling of tcrv_rvv.repack_gemv_q4_0_q8_0: the weight side is a
    block_q5_0x16 block-as-lane layout (the 16 interleaved weight rows of a group
    occupy 16 distinct vector lanes so the dot accumulates LANE-WISE via vwmacc,
    NO cross-lane vredsum -- the per-block reduction wall is gone), and the
    activation is ONE plain block_q8_0 stream (NOT the GEMM's interleaved
    block_q8_0x4): a single activation column.

    q5_0 is q4_0 PLUS the 5th high bit. Each weight is reconstructed
    `((qs_4bit) | (qh_1bit << 4)) - 16` (ggml dequantize_row_q5_0 /
    ggml_vec_dot_q5_0_q8_0): the low-half element j (j in [0,16)) takes qh bit j,
    the high-half element j+16 takes qh bit j+16. To stay a LANE-WISE streaming
    kernel the block_q5_0x16 repack TRANSPOSES the qh field (lanes are the 16
    blocks, not the 32 elements): a 64-byte bit-packed qh region carries, for each
    of the 32 element steps, ONE 16-bit mask whose bit b is block b's qh bit for
    that element. The 5th-bit bias is stored NON-inverted (mask bit = qh_bit) so
    the lane decode assembles the UNSIGNED `A = nibble | (qh_bit << 4)` in [0,31],
    reinterprets u8->i8 (value-identity), then subtracts 16 (`vsub`) to land the
    offset-binary weight in [-16,15] -- the PROVEN block-dot reconstruct
    (`((nibble) | (qh_bit << 4)) - 16`, the q5_0 analog of q4_0's `^0x88`
    two's-complement bake). The qs nibbles keep the SAME 16-way
    interleave as block_q4_0x16, stored RAW (the bias lives in the assembled
    5-bit field, NOT the nibble), so the nibble-load path is byte-identical to the
    q4_0 GEMV. The bit-packed qh keeps the repacked block stride at 352 = 16*22,
    byte-exactly the plain q5_0 bytes for the same 512 weights (no byte blow-up;
    the win comes from killing the per-block vredsum + scattered qh reads, the
    q4_0 1.22->2.6x regime, NOT from fewer bytes).

    At VLEN=128 an e16m1 vector holds only 8 i16 lanes, so each 16-block-as-lane
    group is processed as TWO disjoint contiguous halves of 8 lanes each (rows
    0..7 then rows 8..15) inline; the per-half qh mask is `mask[e] >> (h*half)`.

    The op is nested directly under tcrv_rvv.with_vl and consumes THREE runtime
    ABI value operands -- the repacked q5_0x16 weight base pointer (vx), the plain
    q8_0 activation base pointer (vy), the fp32 output base pointer (s) -- plus the
    runtime contraction element count (n) and the runtime weight-column count (nc),
    plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the nc fp32 results through the
    output pointer.

    The structural facts -- QK=32, the repacked weight block stride (352 for
    block_q5_0x16), the plain activation block stride (34 for block_q8_0), the
    repacked weight nibble byte offset (qs at +32 after the 16 inline fp16 scales),
    the repacked weight qh byte offset (the transposed bit-packed qh masks at +288
    after the 256 nibble bytes), the plain activation quant byte offset (qs at +2),
    the weight interleave width (16), the resource-aware e16m1 strip width
    (half_lanes, 8 at VLEN=128, 16 at VLEN=256), and the dual-fp16 scale model --
    are carried as typed integer/string attributes (I4 mirror only). The fixed
    emission is the op's lowering, NOT a candidate-mirror string read. The bounded
    surface kind is limited to "ggml_repack_gemv_q5_0_q8_0" and scale_model to
    "dual-fp16-per-block-d_x.d_y".

    The optional `integer_core_lmul` attribute is the bounded resource/scheduling
    anchor (the *how*, never the *what*): "mf2" (default, the RVV1.0 fractional
    chain i8mf2 -> i16m1 -> i32m2 -> f32m2 at half_lanes e16m1 lanes per strip) or
    "m1" (the RVV0.7.1 whole-LMUL chain i8m1 -> i16m2 -> i32m4 -> f32m4, ONE
    16-lane strip; half_lanes MUST be 16). The verifier pins the legal
    (integer_core_lmul, half_lanes) combinations fail-closed (I7).

    This op pins the 16x1 repacked q5_0 GEMV ABI fail-closed (I7): the byte layout
    the kernel depends on (weight 352/+32 qs/+288 qh/16, activation 34/+2). A
    malformed typed body cannot lower under this emission.
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
    I64Attr:$weight_qh_byte_offset,
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

def GgmlRepackGemvQ51Q81Op
    : TCRVRVV_Op<"repack_gemv_q5_1_q8_1",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q5_1 x Q8_1 16x1-REPACKED GEMV (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q5_1 16x1-REPACKED GEMV (decode) hot kernel as ONE
    bounded generic plugin-local typed RVV dataflow step. It is the UNION of the
    two existing repacked-GEVM siblings: the FIVE-BIT weight decode of
    tcrv_rvv.repack_gemv_q5_0_q8_0 (the transposed bit-packed qh 5th-bit source)
    fused with the FAMILY-B (scale+MIN, asymmetric) dual fold of
    tcrv_rvv.repack_gemv_q4_1_q8_1. The weight side is the block_q5_1x16
    block-as-lane layout (the 16 interleaved weight rows of a group occupy 16
    distinct vector lanes so the dot accumulates LANE-WISE via vwmacc, NO
    cross-lane vredsum -- the per-block reduction wall is gone), and the
    activation is ONE plain block_q8_1 stream (NOT an interleaved x4). At VLEN=128
    an e16m1 vector holds only 8 i16 lanes, so each 16-block-as-lane group is
    processed as TWO disjoint contiguous halves of 8 lanes each (rows 0..7 then
    rows 8..15) inline; at VLEN=256 it is ONE 16-lane strip. Every weight byte is
    read exactly once.

    q5_1 is q4_1 PLUS the 5th high bit, exactly as q5_0 is q4_0 plus the 5th bit.
    The integer core assembles the UNSIGNED 5-bit weight `A = nibble | (qh_bit <<
    4)` in [0,31] (the SAME transposed bit-packed qh source as the q5_0 repack: a
    64-byte qh region carrying, for each of the 32 element steps, ONE 16-bit mask
    whose bit b is block b's qh bit for that element; the low-half element i takes
    the mask at qh+i*2, the high-half element i+16 the mask at qh+(16+i)*2), then
    reinterprets u8->i8 (value-identity for 0..31). UNLIKE the q5_0 sibling there
    is NO offset-binary `-16` bias (q5_1 is asymmetric -- the bias lives in the
    separate per-block MIN scale), so the assembled lane feeds vwmacc DIRECTLY.
    The qs nibbles are stored RAW (the bias is NOT baked), byte-identical to the
    q4_1/q5_0 repacked nibble load path.

    The per-strip fold carries the per-block MIN correction LANE-WISE, exactly as
    the q4_1 repacked GEVM: block_q5_1x16 stores 16 inline fp16 `d` (per row),
    then 16 inline fp16 `m` (per row, the MIN), then the 256 interleaved nibble
    bytes, then the 64-byte transposed bit-packed qh region; block_q8_1 stores
    fp16 `d` then fp16 `s` (the PRECOMPUTED scaled activation sum d_y*sum(q8))
    then the 32 int8 quants. ggml's exact per-row statement is
    `sumf += (d_x*d_y)*sumi + m_x*s_y`: the (d_x*d_y)*sumi scale term plus the
    m_x*s_y MIN term. d_x and m_x are VECTOR strips (one lane per weight row,
    vle16), d_y and s_y are the single activation column's SCALARS. The MIN term
    is folded LANE-WISE (vfwmul(m_x, s_y) then vfadd) -- the lane-wise analogue of
    the q4_1/q5_1 block-dot's scalar m_x*s_y.

    The op is nested directly under tcrv_rvv.with_vl and consumes THREE runtime
    ABI value operands -- the repacked q5_1x16 weight base pointer (vx), the plain
    q8_1 activation base pointer (vy), the fp32 output base pointer (s) -- plus the
    runtime contraction element count (n) and the runtime weight-column count (nc),
    plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the nc fp32 results through the
    output pointer. There is NO activation-row count (nr is 1) and NO output row
    stride (single-column store `s + x*16`).

    The structural facts -- QK=32, the repacked weight block stride (384 bytes for
    block_q5_1x16 = 16 fp16 d + 16 fp16 m + 256 nibble bytes + 64 transposed qh
    bytes), the plain activation block stride (36 bytes for block_q8_1), the
    repacked weight nibble byte offset (qs at +64 after the 16 d + 16 m fp16
    scales), the per-row weight MIN byte offset (m strip at +32 after the 16 d
    scales), the repacked weight qh byte offset (the transposed bit-packed qh
    masks at +320 after the 256 nibble bytes), the plain activation quant byte
    offset (qs at +4 after the d + s fp16 scales), the activation scaled-sum byte
    offset (s at +2 after d), the weight interleave width (16), the resource-aware
    e16m1 strip width (half_lanes in {8,16}), and the dual-fp16-plus-min scale
    model -- are carried as typed integer/string attributes (I4 mirror only). The
    fixed emission is the op's lowering, NOT a candidate-mirror string read in the
    conversion. The bounded surface kind is limited to
    "ggml_repack_gemv_q5_1_q8_1" and scale_model to
    "dual-fp16-per-block-d_x.d_y-plus-min".

    The optional `integer_core_lmul` attribute is the SAME bounded
    resource/scheduling fact the q4_1/q5_0 repacked GEVM siblings carry (the *how*,
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
    I64Attr:$weight_qh_byte_offset,
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

def GgmlRepackGemvQ5KQ8KOp
    : TCRVRVV_Op<"repack_gemv_q5_K_q8_K",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q5_K x Q8_K 16x1-REPACKED GEMV (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q5_K 16x1-REPACKED GEMV (decode) hot kernel as ONE
    bounded generic plugin-local typed RVV dataflow step -- the K-QUANT
    (super-block) block-as-lane sibling of tcrv_rvv.repack_gemv_q4_K_q8_K. q5_K is
    q4_K PLUS a 5th (high) weight bit: the super-block (d, dmin) dual scale, the
    per-sub-block 6-bit quantized scale/min, the get_scale_min_k4 cross-byte
    unpack, the 8-sub-block-of-32 integer core, and the bsums-min correction are
    ALL IDENTICAL to q4_K. The ONLY difference is a per-quant 5th bit: each 4-bit
    nibble is lifted to a 5-bit value in [0,31] by injecting `(qh_bit << 4)` from a
    dedicated qh high-bit plane. The decode still runs on the UNSIGNED weight value
    (q5_K stores RAW quants with NO offset-binary bias -- the bias lives in the
    per-sub-block 6-bit MIN, exactly as q4_K), so this is K-quant scale+min, NOT
    the q5_0 offset-binary (no -16) form.

    The weight side is the block_q5_Kx16 block-as-lane layout: the 16 interleaved
    weight columns of a group occupy 16 distinct vector lanes so the dot
    accumulates LANE-WISE via vwmacc (NO cross-lane vredsum), and the activation is
    ONE plain block_q8_K stream (IDENTICAL to the q4_K GEVM -- the qh 5th bit is a
    WEIGHT-side fact only). At VLEN=128 an e8 vector holds only 8 i8 lanes per half,
    so each 16-block-as-lane group is processed as TWO disjoint contiguous halves
    of 8 lanes each; at VLEN=256 it is ONE 16-lane strip. Every weight byte is read
    exactly once.

    The qh 5th-bit injection (the sole q5_K-vs-q4_K delta at the integer core):
    block_q5_Kx16 carries a 512-byte qh region (16 columns x 32 bytes, the QK_K/8
    high-bit plane, block-as-lane interleaved: byte l*16 + column) between the 6-bit
    scales region and the interleaved nibble bytes. For the low-nibble sub-block s
    at position l the 5th bit is `(qh[l] >> s) & 1`, for the high-nibble sub-block
    `(qh[l] >> (s+1)) & 1`; each is `<< 4` and OR-ed onto the 4-bit nibble to form a
    value in [0,31], value-identity-reinterpreted to a signed i8 lane (0..31 fits
    i8). This mirrors ggml's `(ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)` in the
    lane-wise arithmetic form (vsrl / vand / vsll / vor), byte-for-byte the q4_K
    lane-wise decode with the qh bit added.

    The op is nested directly under tcrv_rvv.with_vl and consumes THREE runtime ABI
    value operands -- the repacked q5_Kx16 weight base pointer (vx, NC/16
    column-groups at byte stride (n/QK_K)*2816), the plain q8_K activation base
    pointer (vy, one block_q8_K stream at byte stride 292), the fp32 output base
    pointer (s) -- plus the runtime contraction element count (n) and the runtime
    weight-column count (nc), plus the active !tcrv_rvv.vl token. It produces ONE
    i32 LMUL m1 !tcrv_rvv.vector token whose lowering writes the nc fp32 results
    through the output pointer. There is NO activation-row count (nr is 1) and NO
    output row stride (single-column store `s + x*16`).

    The structural facts -- QK_K=256, the repacked weight super-block stride (2816
    bytes for block_q5_Kx16 = 16 fp16 d + 16 fp16 dmin + 192 scales/mins + 512 qh
    high-bit-plane bytes + 2048 nibble bytes), the plain activation super-block
    stride (292 bytes for block_q8_K), the repacked weight quant byte offset (qs at
    +768 after the 32 d/dmin fp16 + 192 scales + 512 qh bytes), the qh high-bit
    plane byte offset (qh at +256 after the 16 d + 16 dmin fp16 + 192 scales), the
    per-column weight dmin byte offset (dmin strip at +32), the custom 6-bit
    per-sub-block scales/mins region byte offset (scales at +64), the plain
    activation quant byte offset (qs at +4), the activation bsums byte offset (bsums
    at +260), the sub-block count (8), the weight interleave width (16), the
    resource-aware strip width (half_lanes in {8,16}), and the
    super-block-d.dmin-plus-bsums-min-qh5 scale model -- are carried as typed
    integer/string attributes (I4 mirror only). The fixed emission is the op's
    lowering, NOT a candidate-mirror string read in the conversion. The bounded
    surface kind is limited to "ggml_repack_gemv_q5_K_q8_K" and scale_model to
    "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-qh5". Route ids, helper
    names, ABI strings, artifact names, status fields, exact intrinsic spellings,
    operand mirror strings, q8/q5 names, and common EmitC/export code must not infer
    or choose the block-as-lane repacked GEVM semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded
    resource/scheduling fact the q4_K repacked GEVM sibling carries (the *how*,
    never the *what*; the 16-way interleaved repack reads the SAME bytes either
    way): absent / "mf2" anchors the RVV1.0 fractional chain (half_lanes in
    {8,16}), "m1" anchors the RVV0.7.1 whole-LMUL chain (half_lanes 16, ONE 16-lane
    strip). The verifier pins the legal (integer_core_lmul, half_lanes) combinations
    fail-closed (I7); half_lanes must divide the 16-way interleave. A malformed
    typed body cannot lower under this emission.
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
    I64Attr:$weight_qh_byte_offset,
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

def GgmlRepackGemmQ5KQ8KOp
    : TCRVRVV_Op<"repack_gemm_q5_K_q8_K",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q5_K x Q8_K 16x1-REPACKED GEMM (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q5_K 16x1-REPACKED GEMM (prefill, M>>1) hot kernel as
    ONE bounded generic plugin-local typed RVV dataflow step -- the K-QUANT
    (super-block) PREFILL sibling of tcrv_rvv.repack_gemv_q5_K_q8_K, and the q5_K
    sibling of tcrv_rvv.repack_gemm_q4_K_q8_K. q5_K is q4_K PLUS a 5th (high) weight
    bit: the super-block dual d/dmin scale, the per-sub-block 6-bit scale/min, the
    8-sub-block-of-32 integer core, the bsums-min correction, the 4-column
    block_q8_Kx4 activation interleave, and the per-column fp32 delta are ALL
    IDENTICAL to q4_K. The ONLY difference is the qh 5th-bit injection: each nibble
    is lifted to a 5-bit value in [0,31] by `(qh_bit << 4)` from the qh high-bit
    plane. This is K-quant scale+min, NOT the q5_0 offset-binary (no -16) form.

    The weight side is the block_q5_Kx16 block-as-lane layout (IDENTICAL to the
    q5_K GEVM weight ABI). The DISTINGUISHING fact from the GEVM is the activation:
    the interleaved block_q8_Kx4 stream (4 activation ROWS per group). The
    8-sub-block 6-bit unpack AND the qh 5th-bit nibble decode are AMORTIZED once per
    16-weight group and REUSED across the 4 (and, across the row loop, M) activation
    columns. At VLEN=128 each 16-block-as-lane group is processed as TWO 8-lane
    halves; at VLEN=256 ONE 16-lane strip. Every weight byte is read exactly once.

    The qh 5th-bit injection is the sole q5_K-vs-q4_K delta at the integer core:
    block_q5_Kx16 carries a 512-byte qh region (16 columns x 32 bytes, block-as-lane
    interleaved: byte l*16 + column) between the 6-bit scales region and the
    interleaved nibble bytes. For low-nibble sub-block s at position l the 5th bit
    is `(qh[l] >> s) & 1`, high-nibble `(qh[l] >> (s+1)) & 1`; each is `<< 4` and
    OR-ed onto the 4-bit nibble to form a value in [0,31] (reinterpreted to a signed
    i8 lane). The qh strip is SHARED across the 4 activation columns exactly like
    the nibble decode.

    The op is nested directly under tcrv_rvv.with_vl and consumes SEVEN runtime ABI
    value operands -- the repacked q5_Kx16 weight base pointer (vx, NC/16
    column-groups at byte stride (n/QK_K)*2816), the repacked q8_Kx4 activation base
    pointer (vy, nr/4 row-groups at byte stride (n/QK_K)*1168), the fp32 output base
    pointer (s), the runtime contraction element count (n), the runtime
    activation-row count (nr) and weight-column count (nc), and the runtime output
    row stride (bs, in floats) -- plus the active !tcrv_rvv.vl token. It produces
    ONE i32 LMUL m1 !tcrv_rvv.vector token whose lowering writes the nr x nc fp32
    results through the output pointer.

    The structural facts -- QK_K=256, the repacked weight super-block stride (2816
    bytes for block_q5_Kx16 = 16 fp16 d + 16 fp16 dmin + 192 scales/mins + 512 qh
    bytes + 2048 nibble bytes), the interleaved activation super-block stride (1168
    bytes for block_q8_Kx4), the repacked weight quant byte offset (qs at +768), the
    qh high-bit plane byte offset (qh at +256), the per-column weight dmin byte
    offset (dmin at +32), the custom 6-bit scales/mins region byte offset (scales at
    +64), the interleaved activation quant byte offset (qs at +16), the activation
    bsums byte offset (bsums at +1040), the sub-block count (8), the weight
    interleave width (16), the activation interleave width (4), the resource-aware
    strip width (half_lanes in {8,16}), and the
    super-block-d.dmin-plus-bsums-min-4col-qh5 scale model -- are carried as typed
    integer/string attributes (I4 mirror only). The fixed emission is the op's
    lowering. The bounded surface kind is limited to "ggml_repack_gemm_q5_K_q8_K"
    and scale_model to
    "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5". Route ids, helper
    names, ABI strings, artifact names, status fields, exact intrinsic spellings,
    operand mirror strings, q8/q5 names, and common EmitC/export code must not infer
    or choose the block-as-lane repacked GEMM semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded
    resource/scheduling fact the q5_K repacked GEVM sibling carries (the *how*,
    never the *what*): absent / "mf2" anchors the RVV1.0 fractional chain,
    "m1" anchors the RVV0.7.1 whole-LMUL chain (half_lanes 16). The verifier pins
    the legal (integer_core_lmul, half_lanes) combinations fail-closed (I7).
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
    I64Attr:$weight_qh_byte_offset,
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

def GgmlRepackGemvQ6KQ8KOp
    : TCRVRVV_Op<"repack_gemv_q6_K_q8_K",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q6_K x Q8_K 16x1-REPACKED GEMV (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q6_K 16x1-REPACKED GEVM (decode) hot kernel as ONE
    bounded generic plugin-local typed RVV dataflow step -- the LARGEST K-QUANT
    (super-block) structural delta over the tcrv_rvv.repack_gemv_q4_K_q8_K /
    tcrv_rvv.repack_gemv_q5_K_q8_K block-as-lane siblings. q6_K is NOT the q4_K/q5_K
    scale+min family: it is a 6-bit SIGNED offset-binary weight with ONE super-block
    `d` and 16 per-sub-block SIGNED int8 scales, and NO per-sub-block min term.

    Three q6_K-specific facts distinguish it from q4_K/q5_K (each load-bearing):

    (1) 6-BIT WEIGHT ASSEMBLY FROM TWO PLANES. Each weight value is
        `((ql & 0xF) | (((qh >> shift) & 3) << 4)) - 32`, a SIGNED value in
        [-32, 31]. The low 4 bits come from the ql plane (QK_K/2 = 128 bytes per
        column), the high 2 bits from the qh plane (QK_K/4 = 64 bytes per column,
        the 2-bit high plane -- NOT q5_K's single-bit qh). The 256-element
        super-block splits into 2 super-halves of 128; within each super-half the
        128 elements are 4 QUADRANTS of 32 (positions +0/+32/+64/+96) whose weights
        are (ql[l] & 0xF | qh>>0), (ql[l+32] & 0xF | qh>>2), (ql[l] >> 4 | qh>>4),
        (ql[l+32] >> 4 | qh>>6) respectively. The `-32` offset-binary bias is folded
        LANE-WISE straight into each weight lane (vsub_vx_i8 by 32), so there is NO
        separate min/bsums correction term -- the sole SINGLE-accumulator fold.

    (2) SIGNED int8 per-sub-block scales. block_q6_Kx16 stores 16 SIGNED int8 scales
        per column (one per 16-element sub-block), NOT the q4_K/q5_K 6-bit packed
        unsigned scale/min. They are loaded LANE-WISE (vle8_v_i8) and SIGN-extended
        (vsext_vf2_i16, NOT the q4_K zero-extend) to weight the per-sub-block integer
        dot into the i32 accumulator (vwmacc_vv). The sub-block scale index for
        element position p is p/16.

    (3) SINGLE ACCUMULATOR, NO MIN. ggml's whole per-super-block statement is
        `sumf += d * (sum over 16 sub-blocks of scale_j * q_dot_j)` with a SINGLE
        super-block `d = fp16(x.d) * fp32(y.d)` -- NO dmin, NO bsums, NO min term.
        The end-of-block fold is one per-strip vfmacc (cvt(sumi) * (d_x*d_y)); the
        q4_K/q5_K vfnmsac MIN term is ABSENT.

    The weight side is the block_q6_Kx16 block-as-lane layout: the 16 interleaved
    weight columns of a group occupy 16 distinct vector lanes so the dot accumulates
    LANE-WISE via vwmacc (NO cross-lane vredsum), and the activation is ONE plain
    block_q8_K stream (stride 292, IDENTICAL to q4_K/q5_K; q6_K reads NO activation
    bsums). At VLEN=128 an e8 vector holds only 8 i8 lanes per half, so each
    16-block-as-lane group is processed as TWO disjoint contiguous halves of 8 lanes
    each; at VLEN=256 it is ONE 16-lane strip. Every weight byte is read exactly once.

    The op is nested directly under tcrv_rvv.with_vl and consumes THREE runtime ABI
    value operands -- the repacked q6_Kx16 weight base pointer (vx, NC/16
    column-groups at byte stride (n/QK_K)*3360), the plain q8_K activation base
    pointer (vy, one block_q8_K stream at byte stride 292), the fp32 output base
    pointer (s) -- plus the runtime contraction element count (n) and the runtime
    weight-column count (nc), plus the active !tcrv_rvv.vl token. It produces ONE i32
    LMUL m1 !tcrv_rvv.vector token whose lowering writes the nc fp32 results through
    the output pointer. There is NO activation-row count (nr is 1) and NO output row
    stride (single-column store `s + x*16`).

    The structural facts -- QK_K=256, the repacked weight super-block stride (3360
    bytes for block_q6_Kx16 = 16 fp16 d + 256 signed int8 scales + 1024 qh high-2-bit
    plane bytes + 2048 ql low-4-bit plane bytes), the plain activation super-block
    stride (292 bytes for block_q8_K), the repacked weight quant (ql) byte offset (at
    +1312 after the 32 d + 256 scales + 1024 qh), the qh high-2-bit plane byte offset
    (at +288 after the 32 d + 256 scales), the signed scales region byte offset (at
    +32 after the 16 fp16 d), the plain activation quant byte offset (qs at +4), the
    sub-block count (16), the weight interleave width (16), the resource-aware strip
    width (half_lanes in {8,16}), and the superblock-d-signed-scale-no-min scale model
    -- are carried as typed integer/string attributes (I4 mirror only). The fixed
    emission is the op's lowering, NOT a candidate-mirror string read in the
    conversion. The bounded surface kind is limited to "ggml_repack_gemv_q6_K_q8_K"
    and scale_model to "superblock-d.fp16-signed8-scale-16-subblocks-6bit-nomin".
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/q6 names, and common EmitC/export
    code must not infer or choose the block-as-lane repacked GEVM semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded resource/scheduling
    fact the q4_K/q5_K repacked GEVM siblings carry (the *how*, never the *what*; the
    16-way interleaved repack reads the SAME bytes either way): absent / "mf2" anchors
    the RVV1.0 fractional chain (half_lanes in {8,16}), "m1" anchors the RVV0.7.1
    whole-LMUL chain (half_lanes 16, ONE 16-lane strip). The verifier pins the legal
    (integer_core_lmul, half_lanes) combinations fail-closed (I7); half_lanes must
    divide the 16-way interleave. A malformed typed body cannot lower under this
    emission.
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
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$weight_qh_byte_offset,
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

def GgmlRepackGemmQ6KQ8KOp
    : TCRVRVV_Op<"repack_gemm_q6_K_q8_K",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q6_K x Q8_K 16x1-REPACKED GEMM (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q6_K 16x1-REPACKED GEMM (prefill, M>>1) hot kernel as
    ONE bounded generic plugin-local typed RVV dataflow step -- the PREFILL sibling
    of tcrv_rvv.repack_gemv_q6_K_q8_K and the q6_K analogue of
    tcrv_rvv.repack_gemm_q5_K_q8_K. The q6_K 6-bit ql|qh weight assembly, the -32
    offset-binary bias folded into each weight lane, the 16 SIGNED int8 per-sub-block
    scales, and the SINGLE-accumulator no-min fold are IDENTICAL to the GEVM. The
    DISTINGUISHING fact from the GEVM is the activation: the interleaved block_q8_Kx4
    stream (4 activation ROWS per group). The 6-bit ql|qh weight decode (and its -32
    bias) is AMORTIZED once per 16-weight group and REUSED across the 4 (and, across
    the row loop, M) activation columns -- each column reads its own interleaved
    block_q8_Kx4 quant. At VLEN=128 each 16-block-as-lane group is processed as TWO
    8-lane halves; at VLEN=256 ONE 16-lane strip. Every weight byte is read once.

    The weight side is the block_q6_Kx16 block-as-lane layout (IDENTICAL to the q6_K
    GEVM weight ABI: 16 fp16 d + 256 signed int8 scales + 1024 qh + 2048 ql, stride
    3360). q6_K reads NO activation bsums (single accumulator, no min). Its
    per-super-block statement is `sumf_c += (d_x * d_y_c) * (sum over 16 sub-blocks of
    scale_j * q_dot_j_c)` -- one super-block d per weight column lane times the
    per-activation-column fp32 delta.

    The op is nested directly under tcrv_rvv.with_vl and consumes SEVEN runtime ABI
    value operands -- the repacked q6_Kx16 weight base pointer (vx, NC/16
    column-groups at byte stride (n/QK_K)*3360), the repacked q8_Kx4 activation base
    pointer (vy, nr/4 row-groups at byte stride (n/QK_K)*1168), the fp32 output base
    pointer (s), the runtime contraction element count (n), the runtime
    activation-row count (nr) and weight-column count (nc), and the runtime output row
    stride (bs, in floats) -- plus the active !tcrv_rvv.vl token. It produces ONE i32
    LMUL m1 !tcrv_rvv.vector token whose lowering writes the nr x nc fp32 results
    through the output pointer.

    The structural facts -- QK_K=256, the repacked weight super-block stride (3360
    bytes for block_q6_Kx16), the interleaved activation super-block stride (1168
    bytes for block_q8_Kx4), the repacked weight quant (ql) byte offset (+1312), the
    qh high-2-bit plane byte offset (+288), the signed scales region byte offset
    (+32), the interleaved activation quant byte offset (qs at +16), the sub-block
    count (16), the weight interleave width (16), the activation interleave width (4),
    the resource-aware strip width (half_lanes in {8,16}), and the
    superblock-d-signed-scale-4col-no-min scale model -- are carried as typed
    integer/string attributes (I4 mirror only). The fixed emission is the op's
    lowering. The bounded surface kind is limited to "ggml_repack_gemm_q6_K_q8_K" and
    scale_model to "superblock-d.fp16-signed8-scale-16-subblocks-6bit-4col-nomin".
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/q6 names, and common EmitC/export
    code must not infer or choose the block-as-lane repacked GEMM semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded resource/scheduling
    fact the q6_K repacked GEVM sibling carries (the *how*, never the *what*): absent /
    "mf2" anchors the RVV1.0 fractional chain, "m1" anchors the RVV0.7.1 whole-LMUL
    chain (half_lanes 16). The verifier pins the legal (integer_core_lmul, half_lanes)
    combinations fail-closed (I7).
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
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$weight_qh_byte_offset,
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

def GgmlRepackGemvQ2KQ8KOp
    : TCRVRVV_Op<"repack_gemv_q2_K_q8_K",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q2_K x Q8_K 16x1-REPACKED GEMV (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q2_K 16x1-REPACKED GEVM (decode) hot kernel as ONE
    bounded generic plugin-local typed RVV dataflow step -- the LOWEST-bit K-QUANT
    (super-block) block-as-lane sibling of tcrv_rvv.repack_gemv_q4_K_q8_K /
    tcrv_rvv.repack_gemv_q5_K_q8_K. q2_K is the MIN-TERM regression: it returns to
    the q4_K/q5_K DUAL super-block (d, dmin) fold + the activation bsums-min term
    (NOT q6_K's single no-min accumulator), but with a 2-BIT weight and a q2_K-
    specific 4-bit packed scale/min. It reuses the SAME dual d/dmin + bsums-min fold
    the oracle-verified q4_K/q5_K GEVMs carry, with THREE q2_K-specific deltas (each
    load-bearing):

    (1) 2-BIT WEIGHT ASSEMBLY, 4 LANES PER BYTE. Each weight value is
        `(qs[l] >> shift) & 3`, an UNSIGNED value in [0,3] -- shift in {0,2,4,6}.
        Each qs byte therefore holds FOUR 2-bit weight lanes belonging to four
        DIFFERENT sub-blocks; the byte is loaded ONCE per column strip and the four
        lanes are peeled LANE-WISE (vsrl by 0/2/4/6 then vand 0x03 -- NOT q4_K's
        vand 0x0F / vsrl 4 nibble split, NOT q6_K's ql|qh two-plane assembly). There
        is NO offset-binary bias on the weight (unlike q6_K's -32): the entire bias
        lives in the per-sub-block 4-bit MIN, exactly as q4_K/q5_K.

    (2) 4-BIT PACKED SCALE/MIN, ONE BYTE PER SUB-BLOCK. block_q2_Kx16 stores, per
        column lane, ONE fp16 super-block `d` and ONE fp16 super-block `dmin` (the
        16 d's then the 16 dmin's inline), then a 256-byte `scales` region packing
        the 16 sub-block bytes (LOW nibble = 4-bit scale, HIGH nibble = 4-bit min)
        per column, then the 1024 interleaved 2-bit quant bytes. The per-sub-block
        scale/min are unpacked LANE-WISE (vle8 across the 16 column lanes, vand 0x0F
        for the scale / vsrl 4 for the min -- a SINGLE byte, NOT q4_K's 6-bit two-
        byte get_scale_min_k4 bit-dance), then zero-extended to i16. ggml's exact
        per-super-block statement is `sumf += dall * (sum over 16 sub-blocks of
        scale_j * q_dot_j) - dmin * (sum over 16 sub-blocks of min_j * bsum_j)` with
        `dall = d_y * d`, `dmin = d_y * dmin`.

    (3) 16 SUB-BLOCKS of 16. The K-quant super-block of 256 elements splits into 16
        sub-blocks of 16 (NOT q4_K's 8 of 32), so each sub-block is EXACTLY ONE
        block_q8_K bsums group: the MIN term reads a SINGLE bsum per sub-block (NOT
        q4_K's paired bs0+bs1), and each sub-block's 16-element integer dot fits in
        an i16 partial with NO 2x16 k-chunk split (16*127*3 = 6096 < 32767).

    The weight side is the block_q2_Kx16 block-as-lane layout: the 16 interleaved
    weight columns of a group occupy 16 distinct vector lanes so the dot accumulates
    LANE-WISE via vwmacc (NO cross-lane vredsum -- the per-block reduction wall is
    gone), and the activation is ONE plain block_q8_K stream (stride 292, IDENTICAL
    to q4_K/q5_K/q6_K). At VLEN=128 an e8 vector holds only 8 i8 lanes per half, so
    each 16-block-as-lane group is processed as TWO disjoint contiguous halves of 8
    lanes each (cols 0..7 then cols 8..15) inline; at VLEN=256 it is ONE 16-lane
    strip. Every weight byte is read exactly once.

    The op is nested directly under tcrv_rvv.with_vl and consumes THREE runtime ABI
    value operands -- the repacked q2_Kx16 weight base pointer (vx, NC/16 column-
    groups at byte stride (n/QK_K)*1344), the plain q8_K activation base pointer (vy,
    one block_q8_K stream at byte stride 292), the fp32 output base pointer (s) --
    plus the runtime contraction element count (n) and the runtime weight-column
    count (nc), plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the nc fp32 results through the
    output pointer. There is NO activation-row count (nr is 1) and NO output row
    stride (single-column store `s + x*16`).

    The structural facts -- QK_K=256, the repacked weight super-block stride (1344
    bytes for block_q2_Kx16 = 16 fp16 d + 16 fp16 dmin + 256 packed scale/min bytes
    + 1024 2-bit quant bytes), the plain activation super-block stride (292 bytes for
    block_q8_K), the repacked weight quant byte offset (qs at +320 after the 32
    d/dmin fp16 + 256 scales bytes), the per-column weight dmin byte offset (dmin
    strip at +32 after the 16 d), the packed 4-bit scale/min region byte offset
    (scales at +64 after the 16 d + 16 dmin fp16), the plain activation quant byte
    offset (qs at +4), the activation bsums byte offset (bsums at +260), the sub-
    block count (16), the weight interleave width (16), the resource-aware strip
    width (half_lanes in {8,16}), and the superblock-d.dmin-plus-bsums-min scale
    model -- are carried as typed integer/string attributes (I4 mirror only). The
    fixed emission is the op's lowering, NOT a candidate-mirror string read in the
    conversion. The bounded surface kind is limited to "ggml_repack_gemv_q2_K_q8_K"
    and scale_model to "superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit".
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/q2 names, and common EmitC/export
    code must not infer or choose the block-as-lane repacked GEVM semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded resource/
    scheduling fact the q4_K/q5_K/q6_K repacked GEVM siblings carry (the *how*, never
    the *what*; the 16-way interleaved repack reads the SAME bytes either way):
    absent / "mf2" anchors the RVV1.0 fractional chain (half_lanes in {8,16}), "m1"
    anchors the RVV0.7.1 whole-LMUL chain (half_lanes 16, ONE 16-lane strip). The
    verifier pins the legal (integer_core_lmul, half_lanes) combinations fail-closed
    (I7); half_lanes must divide the 16-way interleave. A malformed typed body cannot
    lower under this emission.
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

def GgmlRepackGemmQ2KQ8KOp
    : TCRVRVV_Op<"repack_gemm_q2_K_q8_K",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q2_K x Q8_K 16x1-REPACKED GEMM (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q2_K 16x1-REPACKED GEMM (prefill, M>>1) hot kernel as
    ONE bounded generic plugin-local typed RVV dataflow step -- the PREFILL sibling
    of tcrv_rvv.repack_gemv_q2_K_q8_K and the q2_K analogue of
    tcrv_rvv.repack_gemm_q4_K_q8_K. The q2_K 2-bit-4-lanes weight assembly, the 4-bit
    packed scale/min (low nibble scale / high nibble min), the 16 sub-blocks of 16,
    and the DUAL super-block d/dmin + bsums-min fold are IDENTICAL to the GEVM. The
    DISTINGUISHING fact from the GEVM is the activation: the interleaved block_q8_Kx4
    stream (4 activation ROWS per group). The 2-bit weight decode + the 4-bit scale/
    min unpack are AMORTIZED once per 16-weight group and REUSED across the 4 (and,
    across the row loop, M) activation columns -- each column reads its own
    interleaved block_q8_Kx4 quant and bsum. At VLEN=128 each 16-block-as-lane group
    is processed as TWO 8-lane halves; at VLEN=256 ONE 16-lane strip. Every weight
    byte is read once.

    The weight side is the block_q2_Kx16 block-as-lane layout (IDENTICAL to the q2_K
    GEVM weight ABI: 16 fp16 d + 16 fp16 dmin + 256 packed scale/min bytes + 1024
    2-bit quant bytes, stride 1344). Its per-super-block per-COLUMN statement is
    `sumf_c += d_y_c * (d * (sum over 16 sub-blocks of scale_j * q_dot_jc) - dmin *
    (sum over 16 sub-blocks of min_j * bsum_jc))`: one super-block (d, dmin) per
    weight column lane times the per-activation-column fp32 delta d_y_c; the MIN term
    reads a SINGLE interleaved bsum per 16-element sub-block per column.

    The op is nested directly under tcrv_rvv.with_vl and consumes SEVEN runtime ABI
    value operands -- the repacked q2_Kx16 weight base pointer (vx, NC/16 column-
    groups at byte stride (n/QK_K)*1344), the repacked q8_Kx4 activation base pointer
    (vy, nr/4 row-groups at byte stride (n/QK_K)*1168), the fp32 output base pointer
    (s), the runtime contraction element count (n), the runtime activation-row count
    (nr) and weight-column count (nc), and the runtime output row stride (bs, in
    floats) -- plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering writes the nr x nc fp32 results through the
    output pointer.

    The structural facts -- QK_K=256, the repacked weight super-block stride (1344
    bytes for block_q2_Kx16), the interleaved activation super-block stride (1168
    bytes for block_q8_Kx4 = 4 fp32 d + 1024 int8 quants + 64 int16 bsums), the
    repacked weight quant byte offset (qs at +320), the per-column dmin strip at +32,
    the packed 4-bit scale/min region at +64, the interleaved activation quant byte
    offset (qs at +16), the interleaved activation bsums byte offset (bsums at +1040),
    the sub-block count (16), the weight interleave width (16), the activation
    interleave width (4), the resource-aware strip width (half_lanes in {8,16}), and
    the superblock-d.dmin-plus-bsums-min-4col scale model -- are carried as typed
    integer/string attributes (I4 mirror only). The fixed emission is the op's
    lowering. The bounded surface kind is limited to "ggml_repack_gemm_q2_K_q8_K" and
    scale_model to "superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit-4col".
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/q2 names, and common EmitC/export
    code must not infer or choose the block-as-lane repacked GEMM semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded resource/
    scheduling fact the q2_K repacked GEVM sibling carries (the *how*, never the
    *what*): absent / "mf2" anchors the RVV1.0 fractional chain, "m1" anchors the
    RVV0.7.1 whole-LMUL chain (half_lanes 16). The verifier pins the legal
    (integer_core_lmul, half_lanes) combinations fail-closed (I7).
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

def GgmlRepackGemvQ3KQ8KOp
    : TCRVRVV_Op<"repack_gemv_q3_K_q8_K",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q3_K x Q8_K 16x1-REPACKED GEMV (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q3_K 16x1-REPACKED GEVM (decode) hot kernel as ONE
    bounded generic plugin-local typed RVV dataflow step -- the LAST (and MOST
    intricate) K-QUANT (super-block) block-as-lane sibling of
    tcrv_rvv.repack_gemv_q6_K_q8_K. q3_K is q6_K's no-min structural cousin: it REUSES
    the SINGLE-accumulator no-min fold (ONE super-block `d`, 16 per-sub-block SIGNED
    6-bit scales, NO dmin, NO bsums, NO min term) but assembles a 3-bit SUBTRACTIVE
    weight from a 2-bit `qs` low plane + a 1-bit `hmask` high plane, distinct from
    q6_K's 6-bit ql|qh assembly. THREE q3_K-specific facts distinguish it (each
    load-bearing):

    (1) 3-BIT SUBTRACTIVE HMASK WEIGHT ASSEMBLY. Each weight value is
        `((qs >> shift) & 3) - ((hmask & (1<<p)) ? 0 : 4)`, a SIGNED value in
        [-4, 3]. The low 2 bits come from the qs plane (QK_K/4 = 64 bytes per column,
        4 weights per byte at shift in {0,2,4,6}); the SINGLE high bit comes from the
        hmask plane (QK_K/8 = 32 bytes per column, ONE bit per weight). The hmask bit
        is SUBTRACTIVE: the bit CLEAR subtracts 4, the bit SET subtracts 0 --
        equivalently `((qs & 3) | (hbit << 2)) - 4` where `hbit = (hmask >> p) & 1`.
        This -4 subtractive bias (bit=0 -> -4) is the q3_K analogue of q6_K's -32
        offset-binary (folded LANE-WISE into each weight lane, vsub_vx_i8 by 4), so
        there is NO separate min/bsums correction term -- the sole SINGLE-accumulator
        fold. The hmask bit position `p = 4*(n/128) + shift_group` reuses the SAME
        32-byte hmask plane across BOTH 128-element super-halves (the hmask byte index
        is NEVER advanced by super-half -- unlike qs, which advances 32 bytes per
        super-half). The `-4` (bit CLEAR) polarity is what makes it SUBTRACTIVE, NOT
        the additive `qh` of q6_K/q5_K.

    (2) SIGNED 6-BIT per-sub-block scales, PRE-UNPACKED at repack time. The original
        block_q3_K packs 16 six-bit scales into 12 bytes (the ggml kmask1/kmask2
        bit-dance) then reads `scale_j = scales[j] - 32`. The 16x1 REPACK folds BOTH
        the 12-byte unpack AND the -32 bias into the packer, storing 16 DIRECT SIGNED
        int8 scales per column (value in [-32,31]) -- so the emitter's scale side is
        BYTE-IDENTICAL to q6_K's: loaded LANE-WISE (vle8_v_i8) and SIGN-extended
        (vsext_vf2_i16, NOT a zero-extend) to weight the per-sub-block integer dot
        into the i32 accumulator (vwmacc_vv). The sub-block scale index for element
        position p is p/16.

    (3) SINGLE ACCUMULATOR, NO MIN, NO 2x8 K-CHUNK SPLIT. ggml's whole per-super-block
        statement is `sumf += d * (sum over 16 sub-blocks of scale_j * q_dot_j)` with
        a SINGLE super-block `d = fp16(x.d) * fp32(y.d)` -- NO dmin, NO bsums, NO min
        (IDENTICAL fold shape to q6_K). Because the 3-bit weight lies in [-4,3]
        (|w| <= 4, vs q6_K's [-32,31]), a full 16-position sub-block partial stays
        16*4*127 = 8128 < 32767 within int16, so q3_K accumulates each 16-element
        sub-block in ONE i16 partial with NO 2x8 k-chunk split (q6_K's i16 guard is
        ABSENT).

    The weight side is the block_q3_Kx16 block-as-lane layout: the 16 interleaved
    weight columns of a group occupy 16 distinct vector lanes so the dot accumulates
    LANE-WISE via vwmacc (NO cross-lane vredsum), and the activation is ONE plain
    block_q8_K stream (stride 292, IDENTICAL to q4_K/q5_K/q6_K; q3_K reads NO
    activation bsums). At VLEN=128 an e8 vector holds only 8 i8 lanes per half, so each
    16-block-as-lane group is processed as TWO disjoint contiguous halves of 8 lanes
    each; at VLEN=256 it is ONE 16-lane strip. Every weight byte is read exactly once.

    The op is nested directly under tcrv_rvv.with_vl and consumes THREE runtime ABI
    value operands -- the repacked q3_Kx16 weight base pointer (vx, NC/16 column-groups
    at byte stride (n/QK_K)*1824), the plain q8_K activation base pointer (vy, one
    block_q8_K stream at byte stride 292), the fp32 output base pointer (s) -- plus the
    runtime contraction element count (n) and the runtime weight-column count (nc),
    plus the active !tcrv_rvv.vl token. It produces ONE i32 LMUL m1 !tcrv_rvv.vector
    token whose lowering writes the nc fp32 results through the output pointer. There
    is NO activation-row count (nr is 1) and NO output row stride (single-column store
    `s + x*16`).

    The structural facts -- QK_K=256, the repacked weight super-block stride (1824
    bytes for block_q3_Kx16 = 16 fp16 d + 256 signed int8 scales + 512 hmask high-bit
    plane bytes + 1024 qs low-2-bit plane bytes), the plain activation super-block
    stride (292 bytes for block_q8_K), the repacked weight quant (qs) byte offset (at
    +800 after the 32 d + 256 scales + 512 hmask), the hmask high-bit plane byte offset
    (at +288 after the 32 d + 256 scales), the signed scales region byte offset (at +32
    after the 16 fp16 d), the plain activation quant byte offset (qs at +4), the
    sub-block count (16), the weight interleave width (16), the resource-aware strip
    width (half_lanes in {8,16}), and the superblock-d-signed-scale-3bit-subtractive-
    hmask-no-min scale model -- are carried as typed integer/string attributes (I4
    mirror only). The fixed emission is the op's lowering, NOT a candidate-mirror
    string read in the conversion. The bounded surface kind is limited to
    "ggml_repack_gemv_q3_K_q8_K" and scale_model to
    "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin". Route
    ids, helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, operand mirror strings, q8/q3 names, and common EmitC/export code must
    not infer or choose the block-as-lane repacked GEVM semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded resource/scheduling
    fact the q6_K repacked GEVM sibling carries (the *how*, never the *what*; the
    16-way interleaved repack reads the SAME bytes either way): absent / "mf2" anchors
    the RVV1.0 fractional chain (half_lanes in {8,16}), "m1" anchors the RVV0.7.1
    whole-LMUL chain (half_lanes 16, ONE 16-lane strip). The verifier pins the legal
    (integer_core_lmul, half_lanes) combinations fail-closed (I7); half_lanes must
    divide the 16-way interleave. A malformed typed body cannot lower under this
    emission.
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
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$weight_hmask_byte_offset,
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

def GgmlRepackGemmQ3KQ8KOp
    : TCRVRVV_Op<"repack_gemm_q3_K_q8_K",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q3_K x Q8_K 16x1-REPACKED GEMM (block-as-lane) op";
  let description = [{
    Records the COMPLETE ggml q3_K 16x1-REPACKED GEMM (prefill, M>>1) hot kernel as
    ONE bounded generic plugin-local typed RVV dataflow step -- the PREFILL sibling of
    tcrv_rvv.repack_gemv_q3_K_q8_K and the q3_K analogue of
    tcrv_rvv.repack_gemm_q6_K_q8_K. The q3_K 3-bit SUBTRACTIVE hmask weight assembly
    (2-bit qs low plane + 1-bit hmask high plane, `((qs&3)|(hbit<<2)) - 4`), the 16
    SIGNED 6-bit per-sub-block scales (pre-unpacked + -32-biased at repack time), and
    the SINGLE-accumulator no-min fold are IDENTICAL to the GEVM. The DISTINGUISHING
    fact from the GEVM is the activation: the interleaved block_q8_Kx4 stream (4
    activation ROWS per group). The 3-bit qs|hmask weight decode (and its -4
    subtractive bias) is AMORTIZED once per 16-weight group and REUSED across the 4
    (and, across the row loop, M) activation columns -- each column reads its own
    interleaved block_q8_Kx4 quant. At VLEN=128 each 16-block-as-lane group is
    processed as TWO 8-lane halves; at VLEN=256 ONE 16-lane strip. Every weight byte
    is read once.

    The weight side is the block_q3_Kx16 block-as-lane layout (IDENTICAL to the q3_K
    GEVM weight ABI: 16 fp16 d + 256 signed int8 scales + 512 hmask + 1024 qs, stride
    1824). q3_K reads NO activation bsums (single accumulator, no min). Its
    per-super-block statement is `sumf_c += (d_x * d_y_c) * (sum over 16 sub-blocks of
    scale_j * q_dot_j_c)` -- one super-block d per weight column lane times the
    per-activation-column fp32 delta.

    The structural facts -- QK_K=256, the repacked weight super-block stride (1824
    bytes for block_q3_Kx16), the interleaved activation super-block stride (1168 bytes
    for block_q8_Kx4), the repacked weight quant (qs) byte offset (+800), the hmask
    high-bit plane byte offset (+288), the signed scales region byte offset (+32), the
    interleaved activation quant byte offset (qs at +16), the sub-block count (16), the
    weight interleave width (16), the activation interleave width (4), the
    resource-aware strip width (half_lanes in {8,16}), and the superblock-d-signed-
    scale-3bit-subtractive-hmask-4col-no-min scale model -- are carried as typed
    integer/string attributes (I4 mirror only). The fixed emission is the op's
    lowering. The bounded surface kind is limited to "ggml_repack_gemm_q3_K_q8_K" and
    scale_model to
    "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-nomin".
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/q3 names, and common EmitC/export
    code must not infer or choose the block-as-lane repacked GEMM semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded resource/scheduling
    fact the q3_K repacked GEVM sibling carries (the *how*, never the *what*): absent /
    "mf2" anchors the RVV1.0 fractional chain, "m1" anchors the RVV0.7.1 whole-LMUL
    chain (half_lanes 16). The verifier pins the legal (integer_core_lmul, half_lanes)
    combinations fail-closed (I7).
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
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$weight_hmask_byte_offset,
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

def GgmlBlockDotMXFP4Q80Op
    : TCRVRVV_Op<"mxfp4_q8_0_block_dot",
    [TCRVEmitCLowerableOpInterface,
     DeclareOpInterfaceMethods<TunableScheduleOpInterface>]> {
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

    The optional `integer_core_lmul`, `multi_block_factor`, `strip_elision`, and
    `minimum_vlen` attributes are the SAME bounded resource/scheduling shape knobs the
    codebook siblings carry (the *how*, never the *what*; the dot-product is byte-exact
    for any legal combination because integer addition is order-independent and the
    per-block fp32 fold stays in strict ascending block order). The codebook gather's
    legal integer_core_lmul anchor is a VLEN-capability fact (the broadcast `values`
    table register's VLMAX must be >= 16): at VLEN=128 only m1 reaches VLMAX 16; at
    VLEN>=256 mf2 also reaches VLMAX 16, so the ggml `_vl256` mf2 + 2-block-unroll shape
    becomes legal+byte-exact. `minimum_vlen` is the GUARANTEED minimum VLEN bits the
    schedule was selected against; the verifier recomputes the gather legality from the
    SAME VLMAX formula the gearbox uses and rejects an anchor whose VLMAX < 16 at that
    minimum_vlen fail-closed (I7). Absent, it defaults to 128 (only m1 legal), so every
    existing m1 schedule is byte-unchanged.
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
    OptionalAttr<StrAttr>:$strip_elision,
    OptionalAttr<I64Attr>:$minimum_vlen
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

// NOTE: def GgmlBlockDotNVFP4Q80Op (the monolith nvfp4 block-dot op) was RETIRED at
// the nvfp4 flip (C_construct 27->28, the LAST dispatch-wired vec_dot -> closes the
// ① G1 literal-block-dot zoo): the front door now constructs the typed FLAT
// block-dot loop body (tcrv_rvv.typed_flat_block_dot_loop_body with fold_model
// "flat_nvfp4_codebook", stride 36) carrying the nvfp4 CODEBOOK-core brick
// (GgmlBlockDotNVFP4Q80CodebookCoreOp, below) instead -- nvfp4's activation is a
// block_q8_0 stream (like q1_0), so it rides the FLAT loop op, NOT the q8_K
// super-block one. The retired monolith emitter's per-super-block body was
// code-moved into the byte-exact emitNVFP4BlockDotBodyShared lowering (the sole live
// caller, the flat_nvfp4_codebook branch of emitTypedFlatBlockDotLoopBody). mxfp4
// stays a monolith (the FP4-class negative control) -- do NOT retire it.

def GgmlBlockDotNVFP4Q80CodebookCoreOp
    : TCRVRVV_Op<"nvfp4_q8_0_codebook_core",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml NVFP4 x Q8_0 FLAT super-block CODEBOOK integer-core "
      "op (the nvfp4 codebook brick, mxfp4 FP4-e2m1 codebook sibling)";
  let description = [{
    Records the WHOLE per-super-block body of the ggml `ggml_vec_dot_nvfp4_q8_0`
    block dot-product as ONE bounded generic plugin-local typed RVV dataflow step --
    the nvfp4 CODEBOOK brick, the SECOND FP4-class sibling (NVIDIA's FP4). It REUSES
    mxfp4's FP4 (e2m1) `kvalues_mxfp4` 16-entry DOUBLED int8 CODEBOOK GATHER
    (decode_model=lookup) -- the table broadcast into an i8m1 vreg ONCE, each 4-bit
    nibble split into the two UNSIGNED index lanes (vand 0x0F / vsrl 0x04) and mapped
    through the table by `vrgather_vv_i8m1` (NOT an arithmetic nibble-8 decode) -- but
    wraps it in the nvfp4 per-SUB-block UE4M3 fp8 weight scale on a QK=64 SUPER-block
    whose 64 elements span TWO block_q8_0 activation blocks. It reproduces ggml's
    per-super-block fp32 contribution EXACTLY as `_generic` computes it, byte-exact:
    - the super-block is FOUR 16-element sub-blocks. Byte +0..3 carry the four UE4M3
      sub-block scales `d[s_idx]`; byte +4..35 carry the 32 packed FP4 nibble bytes
      (8 per sub-block, sub-block s_idx at +4 + s_idx*8). For each sub-block s_idx the
      weight scale `d_x` is the structured UE4M3 -> fp32 reconstruction
      `ggml_ue4m3_to_fp32(e)` (ggml-impl.h): UNSIGNED (4 exp bits bias=7, 3 mantissa
      bits, NO sign bit), `e==0 || e==0x7F -> 0.0f` (two specials), else `exp=(e>>3)
      &0xF, man=e&7`, `raw = (exp==0) ? ldexpf(man,-9) : ldexpf(1 + man/8, exp-7)`,
      then `* 0.5f` (the HALF compensation for the DOUBLED codebook). Every UE4M3
      scale lies in comfortably-normal fp32, so the emission replicates ggml's EXACT
      ldexpf arithmetic structurally (an emitc.cmp pair for the two specials -> 0.0f;
      a structured exp/man split + the two ldexpf branches + the *0.5f), byte-
      identical to `_generic`. The 64 elements span TWO block_q8_0 blocks: sub-block
      s_idx reads q8 block `2*ib + s_idx/2` at the half-offset `(s_idx%2)*16` within
      that q8_0's 32 quants, so `d_y` changes at s_idx==2 and the per-sub-block strip
      is 8 lanes (QK_NVFP4_SUB/2), the high nibble pairing at +8. The per-sub-block
      integer dot `sumi_s` feeds the SAME asymmetric signed widening product the
      mxfp4/iq4_nl siblings use (vwmul low <-> q8 lo half + vwmacc high <-> q8 hi
      half -> vwredsum -> i32); and `sumf += d_y * d_x * (float)sumi_s` is ONE
      emitc.expression, invoked in STRICT ascending (ib, s_idx) order (fp non-
      associativity byte-exact). There is NO min term and NO trailing factor. nvfp4's
      fold runs PER-SUB-BLOCK in float (4 separate fp folds per super-block), so
      UNLIKE the K-quant super-blocks there is no single integer state; the fold is
      DELIBERATELY OUT OF SCOPE of this op's result -- it is emitter-inlined by the
      flat codebook lowering keyed off this brick's identity.

    The op is nested directly under tcrv_rvv.with_vl and consumes the nvfp4 weight
    base pointer (vx), the block_q8_0 activation base pointer (vy), the runtime
    element count (n), the active !tcrv_rvv.vl token, and an OPTIONAL `block_index`
    induction operand. It produces ONE scalar i32 SSA result (a structural
    placeholder mirroring the sibling flat integer-core bricks' scalar-state arity;
    nvfp4's actual per-super-block contribution is a running fp32 fold with no single
    scalar state, so this result is UNUSED -- the whole per-super-block body is
    emitter-inlined). It carries NO output pointer.

    The block-quantized super-block STRUCTURE is first-class in the typed body. The
    structural facts -- qk (64, QK_NVFP4), qk_sub (16, QK_NVFP4_SUB), the AoS block
    strides (36 bytes for block_nvfp4 = {uint8_t d[4]; uint8_t qs[32]}, 34 bytes for
    block_q8_0), the nvfp4 weight quant payload byte offset (+4, after the four UE4M3
    scale bytes) and the q8_0 payload byte offsets (the q8 quants at +2 after the
    inline fp16 scale, the per-sub-block q8 high half at +8) -- are carried as typed
    integer/string attributes (I4 mirror only). The 16-entry DOUBLED int8 CODEBOOK is
    a `DenseI8ArrayAttr:$codebook` of EXACTLY 16 int8 entries (the SAME table as
    mxfp4); it is a STRUCTURAL FACT consumed structurally into the realized body
    (emitted as a `static const int8_t[16]` decl + the `vle8` table load), NOT an
    arithmetic decode, NOT a string plan read. The bounded surface kind is limited to
    "ggml_nvfp4_q8_0_codebook_core" and scale_model to "ue4m3-half-per-sub-block".
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/nvfp4 names, and common
    EmitC/export code must not infer or choose the super-block codebook dot-product
    semantics.

    The CODEBOOK GATHER is inherently a Zvl128b legality fact (the broadcast `values`
    register's VLMAX must be >= 16 to index all 16 table entries), so the integer
    core pins the m1 anchor; the OPTIONAL `integer_core_lmul` shape knob is restricted
    to "m1" (the verifier rejects any other anchor fail-closed, I7). A board with VLEN
    < 128 cannot legally host the codebook gather; that is an N1 capability-legality
    divergence.

    LOOP-CAPABLE per-super-block-source form: the OPTIONAL `block_index` operand
    toggles the per-super-block-source form. When present it is the induction variable
    (region argument 0) of an enclosing tcrv_rvv.typed_flat_block_dot_loop_body region
    whose fold_model is "flat_nvfp4_codebook": the nvfp4 weight strip is read at
    `weight_base + block_index*36` and the q8_0 activation blocks at
    `activation_base + (2*block_index + s_idx/2)*34`. When ABSENT the op keeps the
    standalone single-super-block form.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
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
  let results = (outs AnyType:$partial);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($element_count) `,` type($vl) `->` type($partial)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ10Q80Op
    : TCRVRVV_Op<"q1_0_q8_0_block_dot",
    [TCRVEmitCLowerableOpInterface,
     DeclareOpInterfaceMethods<TunableScheduleOpInterface>]> {
  let summary =
      "generic typed RVV block-quantized ggml Q1_0 x Q8_0 dot-product op";
  let description = [{
    Records the COMPLETE ggml `ggml_vec_dot_q1_0_q8_0` block dot-product as ONE
    bounded generic plugin-local typed RVV dataflow step -- the BINARY ({-1,+1})
    class, one of the last three uncommon ggml dot kernels. The q1_0 weight is a
    1-bit-per-element sign: a set bit decodes to `+q8`, an unset bit to `-q8`. The
    genuinely-new mechanism vs every prior sibling is the per-lane BINARY SIGN
    DECODE -- a `vmerge(vneg(q8), q8, bit_mask)` (in the i8 domain) that selects
    `+q8` where the weight bit is set and `-q8` where it is clear. The mask is the
    4 packed weight bit-bytes loaded DIRECTLY via `vlm_v_b{ratio}` (the packed bits
    ARE the i8 sign mask: bit `8b+i` -> lane `8b+i`), with NO kmask table, NO
    `vand`/`vmsne` decode, NO codebook, NO nibble unpack, NO offset-binary `-8`
    bias: the q8 value IS the magnitude and the weight bit IS the sign. This is
    ggml's shipped `_vl128` lane structure.

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

    (2) BINARY SIGN DECODE. Within sub-block k the 32 bits map to the 32 q8 lanes:
        bit `bits[k*4 + b]` bit `m` (m=0..7) pairs with q8 lane `b*8 + m`, value
        `(bit & (1<<m)) ? +q8[b*8+m] : -q8[b*8+m]` -- ggml's EXACT `quants.c`
        ordering. The emitter runs ONE 32-lane sub-block body (ggml's shipped
        `_vl128` lane structure): the 4 packed bit-bytes load DIRECTLY into the i8
        sign mask via `vlm_v_b{ratio}` (the packed bits ARE the mask: bit `8b+m` ->
        lane `8b+m`), the 32 q8 quants load via `vle8`, and the sign is applied IN
        THE i8 DOMAIN via `vmerge(vneg_i8(q8), q8, mask)` (+q8 where the bit is set,
        -q8 where clear). The signed 32-lane vector is reduced by ONE `vwredsum`
        widening i8 -> i16m1 into the per-sub-block integer `sumi_block` (32 lanes *
        |q8|<=127 = 4064 < 32767, so the i16 accumulator never overflows). The
        anchor is the smallest whole-LMUL whose i8 VLMAX spans the 32-element
        sub-block: m2 at VLEN128 (e8m1 VLMAX 16 < 32), m1 at VLEN256 (e8m1 VLMAX
        32). This is a pure within-block lowering: vl stays 32 and never crosses
        into a second sub-block. (The i8-domain negate is byte-exact on the real q8
        quantization domain [-127,127]; only the unreachable -128 boundary would
        overflow, matching ggml's `_vl128` exactly.)

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
    32-element sub-block straddles m1's i8 VLMAX boundary between VLEN128/256, so
    its elided-correct anchor MOVES with VLEN across {m1, m2} exactly like the q8_0
    sibling: the capability/resource-aware schedule autotuner stamps "m2" at VLEN128
    and "m1" at VLEN256 from the SAME getRVVStripVLMAXElements truth source (it
    adopts TunableScheduleOpInterface; kernel key "q1_0"). The verifier rejects any
    other anchor fail-closed (I7).
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
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$minimum_vlen
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ10Q80BinarySignCoreOp
    : TCRVRVV_Op<"q1_0_q8_0_binary_sign_core",
    [TCRVEmitCLowerableOpInterface,
     DeclareOpInterfaceMethods<TunableScheduleOpInterface>]> {
  let summary =
      "generic typed RVV ggml Q1_0 x Q8_0 BINARY-sign per-super-block "
      "integer-core op (the q1_0 binary-sign brick, the BINARY {-1,+1} class)";
  let description = [{
    Records the genuinely-new q1_0 machinery -- the per-lane BINARY SIGN DECODE
    + the four-sub-block two-level fp32 fold -- as ONE bounded generic
    plugin-local typed RVV dataflow step, the FLAT sibling of the super-block
    integer-core bricks (q2_K/tq1_0/iq1_s). It is the net-new marginal cost of
    the q1_0 flip: unlike q4_0/q8_0/q5_0 (a single per-block integer core folded
    by a shared scale-dequant + cross-block accumulate), q1_0's per-super-block
    contribution is a FOUR-sub-block binary decode with a distinct TWO-LEVEL fold
    (`d0 * Σ_k(d1_k * sumi_block_k)`), so no existing flat brick chain expresses
    it; this op carries the whole per-super-block body.

    The genuinely-new mechanism vs every prior sibling is the per-lane BINARY
    SIGN DECODE -- a `vmerge(vneg(q8), q8, bit_mask)` (in the i8 domain) that
    selects `+q8` where the weight bit is set and `-q8` where clear. The mask is
    the 4 packed weight bit-bytes loaded DIRECTLY via `vlm_v_b{ratio}` (the
    packed bits ARE the i8 sign mask: bit `8b+i` -> lane `8b+i`), with NO kmask
    table, NO `vand`/`vmsne` decode, NO codebook, NO nibble unpack, NO
    offset-binary `-8` bias: the q8 value IS the magnitude and the weight bit IS
    the sign. This is ggml's shipped `_vl128` lane structure. Within each of the
    FOUR q8_0 sub-blocks (k=0..3) the 32 signed lanes reduce by ONE `vwredsum`
    (i8 -> i16m1) into the per-sub-block integer `sumi_block` (32 lanes *
    |q8|<=127 = 4064 < 32767, no overflow). The anchor is the smallest whole-LMUL
    whose i8 VLMAX spans the 32-element sub-block: m2 at VLEN128 (e8m1 VLMAX 16 <
    32), m1 at VLEN256. The TWO-LEVEL fp32 fold carries a per-super-block
    `float sumi` (folding `sumi += d1_k * (float)sumi_block` in strict ascending
    k order), then the super-block term `sumf += d0 * sumi` (ggml EXACT order, so
    fp non-associativity is byte-exact). The FIVE fp16 reads (d0 once, d1_k for
    k=0..3) are the only sanctioned opaque scalar pieces.

    The result is produced as ONE SCALAR i32 SSA result (a STRUCTURAL placeholder
    mirroring the super-block integer-core bricks' scalar-state arity; q1_0's
    per-super-block contribution is a running fp32 two-level fold with no single
    integer scalar state, so this result is UNUSED -- the whole per-super-block
    body, INCLUDING the fold, is emitter-inlined by the flat_binary_two_level
    loop lowering keyed off this brick's identity). The op is nested directly
    under tcrv_rvv.with_vl and consumes the q1_0 weight base pointer (vx), the
    q8_0 activation base pointer (vy), the runtime element count (n), the active
    !tcrv_rvv.vl token, and an OPTIONAL `block_index` induction operand. It
    carries NO output pointer.

    LOOP-CAPABLE per-super-block-source form: the OPTIONAL `block_index` operand
    is the induction variable (region argument 0) of an enclosing
    tcrv_rvv.typed_flat_block_dot_loop_body region whose fold_model is
    "flat_binary_two_level": the q1_0 weight strip is read at
    `weight_base + block_index*weight_block_stride` and the FOUR q8_0 activation
    sub-blocks at `activation_base + (block_index*activation_blocks_per_weight +
    k)*activation_block_stride`. When ABSENT the op keeps the standalone
    single-super-block form.

    The structural facts -- qk (128), the super-block stride (18 bytes), the q8_0
    block stride (34), the per-super-block q8-block span (4), the weight bit
    payload byte offset (+2), the q8 quant byte offset (+2), and the binary sign
    scale model -- are carried as typed integer/string attributes (I4 mirror
    only). The op identity is the dispatch key. Route ids, helper names, ABI
    strings, artifact names, status fields, exact intrinsic spellings, operand
    mirror strings, q8/q1_0 names, and common EmitC/export code must not infer or
    choose block-quantized dot-product semantics.

    The optional `integer_core_lmul` attribute is the SAME bounded resource shape
    knob the siblings carry (the *how*, never the *what*; the dot-product is
    byte-exact for any legal value because integer addition is order-independent
    and the per-sub-block/super-block fp32 fold stays in strict ascending order).
    The 32-element sub-block straddles m1's i8 VLMAX boundary between VLEN128/256,
    so its elided-correct anchor MOVES with VLEN across {m1, m2} like the q8_0
    sibling: the schedule autotuner stamps "m2" at VLEN128 and "m1" at VLEN256
    from getRVVStripVLMAXElements (kernel key "q1_0"). The anchor defaults to "m2"
    (the VLEN-universal-safe floor), so an attr-less brick verifies + lowers
    correctly. The verifier rejects any other anchor fail-closed (I7).
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$activation_blocks_per_weight,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$minimum_vlen
  );
  let results = (outs AnyType:$sumi);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($element_count) `,` type($vl) `->` type($sumi)";
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

    LOOP-CAPABLE per-super-block-source form (M-FLAT q6_K milestone-1): the
    OPTIONAL `block_index` operand toggles the per-super-block-source form. When
    present it is the induction variable (region argument 0) of an enclosing
    tcrv_rvv.typed_super_block_block_dot_loop_body region: the q6_K weight strip
    and the q8_K activation strip are read at `weight_base + block_index*
    weight_block_stride` and `activation_base + block_index*
    activation_block_stride`, byte-exact to the monolithic super_block_base_x/y
    arithmetic. When ABSENT the op keeps the unchanged single-super-block
    standalone form (the aux32 partial K1 route is untouched).
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
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
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($output) `,` type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlBlockDotQ3KQ8KAux32Op
    : TCRVRVV_Op<"q3_k_q8_k_aux32_partial",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q3_K x Q8_K super-block INTEGER aux32 partial op";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_q3_K_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step -- the
    q3_K sibling of tcrv_rvv.q6_k_q8_k_aux32_partial. It reproduces q3_K's
    per-super-block INTEGER state `aux32[8]` EXACTLY as `_generic` computes it right
    BEFORE the fp32 `d`-multiply, byte-exact (i32 equality). The deferred no-min
    fp32 two-level fold (the per-super-block fp16-`d` x fp32-`d` multiply into
    `sums[8]` + the sequential horizontal sum into `*s`) is DELIBERATELY OUT OF
    SCOPE here -- it is the REUSED q6_K no-min positive fold (q4_k_sums_fold_scale_d,
    d@108). This op isolates the q3_K-specific integer machinery from that fold:
    - the 2-bit + SUBTRACTIVE-hmask unpack into the element-ordered int8_t aux8[256]
      (`a = (low2 | (hbit<<2)) - 4`, the SAME 32-byte hmask plane across all 8
      groups, SIGNED [-4,3]);
    - the q3_K-OWN SIGNED 6-bit scale bit-dance (kmask1=0x03030303 /
      kmask2=0x0f0f0f0f -> the 16 6-bit scales read as SIGNED int8, `scale_j =
      scales[j] - 32`), staged through a caller-supplied uint32_t utmp[4] scratch;
    - the nested sub-block loop applying the per-sub-block signed scale in the i32
      domain (vwmul i8xi8 -> i16 -> vwmacc.vx i32 += (scales[js]-32) x i16) into the
      carried 8-lane aux32 accumulator.

    The op is nested directly under tcrv_rvv.with_vl and consumes the q3_K weight
    base pointer (vx), the q8_K activation base pointer (vy), the i32 aux32 output
    pointer, the runtime element count (n), the active !tcrv_rvv.vl token, plus an
    OPTIONAL `block_index` induction operand (the loop form). It produces ONE i32
    LMUL m1 !tcrv_rvv.vector token.

    The block-quantized super-block STRUCTURE is first-class in the typed body. The
    structural facts -- QK_K=256, 16 sub-blocks of 16 elements, the AoS block strides
    (110 bytes for block_q3_K, 292 bytes for block_q8_K), the q3_K payload byte
    offsets (hmask @0, qs @32, scales @96) and the q8_K payload byte offset (qs @4)
    -- are carried as typed integer/string attributes (I4 mirror only). The bounded
    surface kind is limited to "ggml_q3_k_q8_k_aux32_partial" and scale_model to
    "per-sub-block-int6-signed-scale-i32-domain". Route ids, helper names, ABI
    strings, artifact names, status fields, exact intrinsic spellings, operand mirror
    strings, q8/q3 names, and common EmitC/export code must not infer or choose
    super-block dot-product semantics.

    LOOP-CAPABLE per-super-block-source form: the OPTIONAL `block_index` operand
    toggles the per-super-block-source form. When present it is the induction
    variable (region argument 0) of an enclosing
    tcrv_rvv.typed_super_block_block_dot_loop_body region: the q3_K weight strip and
    the q8_K activation strip are read at `weight_base + block_index*
    weight_block_stride` and `activation_base + block_index*
    activation_block_stride`. When ABSENT the op keeps the single-super-block
    standalone form.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_hmask_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$activation_quant_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $output `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
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
    re-strip is a capability item, not built here).

    The optional `integer_core_lmul` attribute is a bounded resource/scheduling
    knob selecting the BASE LMUL of the per-sub-block integer-MAC widening chain
    i8 -> i16 -> i32 (the *how*, never the *what*; the byte-exact deferred fp32
    fold order is untouched). It anchors the chain at "mf2" (absent/default --
    today's TWO 8-lane halves per sub-block, e8mf2(8)->i16m1->e32m2 each), or "m1"
    (ONE 16-lane strip per sub-block, e8m1(16)->i16m2->e32m4, the 16 i32 lanes
    folded back element-wise to the canonical 8 before the fp32 cvt). "m1" is the
    HARD CEILING here -- DIFFERENT from q4_K's "m2" ceiling -- because q6_K's
    sub_block == 16 elements: i8m1 == 16 elements == exactly ONE sub-block under
    ONE scalar `scale`; an "m2" base (32 elements) would fold TWO sub-blocks under
    one scalar scale (identical ground to q4_K's m4 rejection at 32-element
    sub-blocks). The verifier pins the legal set {"mf2","m1"} fail-closed (I7).
    The fold-back (the canonical 8-lane regroup of the wide aux32 before the
    fp32 fold) is VLEN-agnostic (literal element-offset vslidedown, not a
    register-subgroup vget) so it is bit-exact at VLEN128 AND VLEN256. The K3
    measurement tuner stamps the chosen anchor; this K2 op carries the structural
    facts plus this single dormant resource shape knob.
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

def Q4KNibbleUnpackOp
    : TCRVRVV_Op<"q4_k_nibble_unpack",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV q4_K/q5_K plain 4-bit nibble unpack-to-scratch op";
  let description = [{
    Records ONE super-block's plain 4-bit nibble unpack into the element-ordered
    aux8[256] scratch (Region A of the ggml `ggml_vec_dot_q4_K_q8_K` integer
    core) as ONE bounded generic plugin-local typed RVV dataflow step -- the
    Track B q4_K auto-construction's BRICK 1. This is the q4_K-specific nibble
    unpack: each weight byte holds two UNSIGNED 4-bit nibbles in [0,15] (NO
    offset-binary bias -- the bias is carried by the q4_K min term, NOT here, so
    this is distinct from tcrv_rvv.packed_i4_offset_binary_x_i8_product's
    xor-0x88 decode), and the 256 nibbles are spread element-ordered into a
    function-scoped int8_t aux8[256] scratch the later super-block bit-dance +
    per-sub-block dot (bricks 2-8, DEFERRED) read contiguously.

    The op is nested directly under tcrv_rvv.with_vl, consumes ONE runtime ABI
    value operand -- the q4_K weight base pointer (the AoS block_q4_K byte array)
    -- plus the active !tcrv_rvv.vl token, and produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering DECLARES the int8_t aux8[256] scratch
    and fills it. The op decodes ONE super-block (the nb = n / 256 super-block
    loop, the per-block address arithmetic, the 6-bit scale/min bit-dance, the
    per-sub-block dot, and the fp32 fold are NOT this op -- they are the deferred
    q4_K bricks and the monolithic tcrv_rvv.q4_k_q8_k_aux_partial /
    q4_k_q8_k_block_dot ops, which STAY).

    The block-quantized super-block STRUCTURE is first-class in the typed body.
    The structural facts -- QK_K=256 (super-block size), 8 sub-blocks of 32
    elements, the AoS weight block stride (144 bytes for block_q4_K), and the qs
    byte offset (16, after d@0|dmin@2|the 12 packed scale bytes@4) -- are carried
    as typed integer attributes (I4 mirror only). The fixed emission mirrors
    `_generic` (quants.c) line-for-line so byte-exactness is by construction:
    FOUR 64-element chunks, each chunk a literal-VL vsetvl_e8m2(32) + a 32-wide
    vle8 of the weight bytes at `weight_base + qs_off + chunk*32`, a 32-wide
    vand 0x0F (low nibble -> a[chunk*64 + 0..31]) and a 32-wide vsrl 4 (high
    nibble -> a[chunk*64 + 32..63]), each reinterpreted u8m2 -> i8m2 and vse8
    stored into aux8 via emitc subscript (the q4_K nibbles are UNsigned [0,15],
    NO -32 bias). This emission is the op's fixed structure, shared VERBATIM with
    the monolithic q4_K/q5_K integer core's Region A (so both emit byte-identical
    C), NOT a candidate-mirror string read in the conversion. The bounded surface
    kind is limited to "q4_k_nibble_unpack". Route ids, helper names, ABI strings,
    artifact names, status fields, exact intrinsic spellings, operand mirror
    strings, q8/q4 names, and common EmitC/export code must not infer or choose
    nibble-unpack semantics.

    The op emits the byte-exact form for the VLEN>=128 capability class (rv64gcv
    => Zvl128b => VLEN>=128, the validated target): the unpack runs one
    vsetvl_e8m2(32) per 32-element nibble group (NO VLEN-robust re-strip loop --
    a VLEN<128 board would leave part of aux8 unwritten; the zve32x/zve64x
    re-strip is a capability item, not built here). No resource shape knobs yet.

    LOOP-CAPABLE per-super-block-source form (M-FLAT q4_K milestone-2, mirrors
    the flat load / block_fp16_scale_product bricks): the OPTIONAL `block_index`
    operand toggles the per-super-block-source form. When present it is the
    induction variable (region argument 0) of the enclosing
    tcrv_rvv.typed_super_block_block_dot_loop_body region, and the super-block
    integer core reads the weight bytes at `weight_base + block_index*stride`
    (the stride the loop op carries as weight_block_stride) -- byte-exact to the
    monolithic q4_K block dot's super_block_base_x address arithmetic. When ABSENT
    the op keeps the unchanged single-super-block form (the imported ABI base IS
    the super-block base, no per-super-block offset). The single-block strong
    paths never set block_index and are untouched.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$vl,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$weight_qs_byte_offset,
    // q5_K (the 5-bit K-quant) ONLY: the byte offset of the qh high-bit plane
    // (the 5th weight bit). PRESENT selects the q5_K nibble unpack (each unpacked
    // nibble gets the qh 5th bit injected in the u8 domain before the u8->i8
    // reinterpret); ABSENT keeps the byte-identical q4_K unpack (no qh). The
    // presence of this optional attr is the ONLY q5_K-vs-q4_K difference at BRICK
    // 1 -- it does NOT add a new brick and does NOT enter the region. The verifier
    // format-keys it: qh present <=> block_q5_K (stride 176, qs@48); absent <=>
    // block_q4_K (stride 144, qs@16).
    OptionalAttr<I64Attr>:$weight_qh_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $vl (`block` $block_index^ `:` type($block_index))? "
    "attr-dict `:` type($weight_base) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def Q4KScaleMinBitDanceOp
    : TCRVRVV_Op<"q4_k_scale_min_bit_dance",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV q4_K/q5_K 6-bit scale/min bit-dance decode-to-scratch op";
  let description = [{
    Records ONE super-block's 6-bit scale/min bit-dance (the get_scale_min_k4 /
    utmp/kmask cross-byte decode -- Region B of the ggml `ggml_vec_dot_q4_K_q8_K`
    integer core) as ONE bounded generic plugin-local typed RVV dataflow step --
    the Track B q4_K auto-construction's BRICK 2 (sibling to BRICK 1's
    tcrv_rvv.q4_k_nibble_unpack). This is the q4_K-specific 6-bit scale/min
    unpack: three packed uint32 scale words at the block's scales offset are
    cross-shuffled through the kmask1/kmask2/kmask3 (0x3f3f3f3f / 0x0f0f0f0f /
    0x03030303) bit-dance into a 4-word utmp[4] scratch laid out as exactly the
    16 bytes [scales[0..7], mins[0..7]] (the same type-pun _generic does with
    scales=&utmp[0]/mins=&utmp[2]), the contiguous 8 6-bit scales + 8 6-bit mins
    the per-sub-block dot + the min term (bricks 3-8, DEFERRED) read.

    The op is nested directly under tcrv_rvv.with_vl, consumes ONE runtime ABI
    value operand -- the q4_K weight base pointer (the AoS block_q4_K byte array)
    -- plus the active !tcrv_rvv.vl token, and produces ONE i32 LMUL m1
    !tcrv_rvv.vector token whose lowering DECLARES the uint32_t utmp[4] scratch
    and fills it via the STRUCTURED scalar emitc.bitwise_{and,or,left_shift,
    right_shift} chain (NO raw string blob), mirroring _generic (quants.c:685-690)
    line-for-line so byte-exactness is by construction. The bit-dance is pure
    scalar: it does NOT consume the active vl (the vl token rides for verifier
    nesting + the optional vse8 scale/min observable, whose 16-byte store uses its
    OWN literal vsetvl_e8m1(16)). The op decodes ONE super-block (the nb = n/256
    super-block loop, the per-block address arithmetic, the plain 4-bit nibble
    unpack, the per-sub-block dot, the MIN term, and the fp32 fold are NOT this
    op -- they are BRICK 1 / the deferred q4_K bricks and the monolithic
    tcrv_rvv.q4_k_q8_k_aux_partial / q4_k_q8_k_block_dot ops, which STAY).

    The block-quantized super-block STRUCTURE is first-class in the typed body.
    The structural facts -- QK_K=256 (super-block size), 8 sub-blocks of 32
    elements, the AoS weight block stride (144 bytes for block_q4_K), and the
    scales byte offset (4, after d@0|dmin@2) where the 12 packed 6-bit scale/min
    bytes live -- are carried as typed integer attributes (I4 mirror only). The
    fixed emission mirrors _generic (quants.c) line-for-line so byte-exactness is
    by construction: three (const uint32_t *)(weight_base + scales_off) word loads
    w0/w1/w2, the kmask1/2/3 cross-byte shuffle into utmp[0..3] (u0 = w0 & kmask1;
    u1 = (w2 & kmask2) | (((w0 >> 6) & kmask3) << 4); u2 = w1 & kmask1; u3 =
    ((w2 >> 4) & kmask2) | (((w1 >> 6) & kmask3) << 4)), and scales = (const
    uint8_t *)&utmp[0]. This emission is the op's fixed structure, shared VERBATIM
    with the monolithic q4_K/q5_K integer core's Region B (so both emit
    byte-identical C), NOT a candidate-mirror string read in the conversion. The
    bounded surface kind is limited to "q4_k_scale_min_bit_dance". Route ids,
    helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, operand mirror strings, q8/q4 names, and common EmitC/export code
    must not infer or choose scale/min-bit-dance semantics.

    The op emits the byte-exact form for the VLEN>=128 capability class (rv64gcv
    => Zvl128b => VLEN>=128, the validated target): the scale/min decode is pure
    scalar (no VLEN dependence) and the optional 16-byte vse8 observable runs one
    vsetvl_e8m1(16) (NO VLEN-robust re-strip -- a VLEN<128 board would split the
    16-byte store; the zve32x/zve64x re-strip is a capability item, not built
    here). No resource shape knobs yet.

    LOOP-CAPABLE per-super-block-source form (M-FLAT q4_K milestone-2): the
    OPTIONAL `block_index` operand toggles the per-super-block-source form. When
    present it is the induction variable (region argument 0) of the enclosing
    tcrv_rvv.typed_super_block_block_dot_loop_body region and the scale words are
    read at `weight_base + block_index*stride + scales_off` (the stride the loop
    op carries as weight_block_stride) -- byte-exact to the monolithic super_block_
    base_x arithmetic. When ABSENT the op keeps the unchanged single-super-block
    form. The single-block strong paths never set block_index and are untouched.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$vl,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$weight_scales_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $vl (`block` $block_index^ `:` type($block_index))? "
    "attr-dict `:` type($weight_base) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def Q4KScaledDotOp
    : TCRVRVV_Op<"q4_k_scaled_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV q4_K/q5_K per-sub-block uint6-scaled i32 dot + integer "
      "fold-back op";
  let description = [{
    Records ONE super-block's per-sub-block uint6-scaled i32 dot into the 8-lane
    aux32 PLUS the integer fold-back (Region C of the ggml `ggml_vec_dot_q4_K_q8_K`
    integer core) as ONE bounded generic plugin-local typed RVV dataflow step --
    the Track B q4_K auto-construction's BRICK 3 (sibling to BRICK 1's
    tcrv_rvv.q4_k_nibble_unpack and BRICK 2's tcrv_rvv.q4_k_scale_min_bit_dance).
    This is the q4_K-specific scaled integer MAC: over the 8 sub-blocks of 32
    elements, the per-sub-block UINT6 6-bit scale `scales[js]` (the BRICK 2 decoded
    scales) is applied in the i32 domain to the per-element products of the BRICK 1
    unpacked aux8[256] weights with the q8_K activation bytes, accumulating into a
    running aux32 accumulator. The scale is FUSED into the vwmacc (NOT a separable
    pre-multiply). The plain 4-bit nibble unpack, the 6-bit scale/min bit-dance,
    the MIN term, the two-level fp32 fold, and the fp16 d/dmin reads (bricks 1, 2,
    and the deferred bricks 4-8) are NOT this op -- they are the sibling bricks and
    the monolithic tcrv_rvv.q4_k_q8_k_aux_partial / q4_k_q8_k_block_dot ops, which
    STAY.

    The op is nested directly under tcrv_rvv.with_vl, consumes THREE runtime ABI
    value operands -- the BRICK 1 unpacked aux8 base pointer (const int8_t *), the
    BRICK 2 decoded scales base pointer (const uint8_t *), and the q8_K activation
    base pointer (const uint8_t *) -- plus the active !tcrv_rvv.vl token, and
    produces ONE i32 LMUL m1 !tcrv_rvv.vector token whose lowering DECLARES an
    int32_t aux32_out[8] scratch and stores the canonical-8 aux32 into it.

    The block-quantized super-block STRUCTURE is first-class in the typed body.
    The structural facts -- QK_K=256 (super-block size), 8 sub-blocks of 32
    elements, and the AoS weight block stride (144 bytes for block_q4_K) -- are
    carried as typed integer attributes (I4 mirror only). The fixed emission
    mirrors `_generic` (quants.c) line-for-line so byte-exactness is by
    construction: the aux32 is seeded to zero via vmv_v_x_i32<l32>, then the
    sub-block loop loads the UINT6 scale, and per MAC strip of stripWidth elements
    runs a vsetvl_e8<l8>(stripWidth) + two vle8 (q8 + aux8) + vwmul i16<l16> +
    vwmacc.vx i32<l32> += scale x product into the running aux32. The integer add
    is associative/order-free, so the per-lane accumulation is byte-exact. This
    emission is the op's fixed structure, shared VERBATIM with the monolithic
    q4_K/q5_K integer core's Region C (so both emit byte-identical C), NOT a
    candidate-mirror string read in the conversion. The bounded surface kind is
    limited to "q4_k_scaled_dot". Route ids, helper names, ABI strings, artifact
    names, status fields, exact intrinsic spellings, operand mirror strings, q8/q4
    names, and common EmitC/export code must not infer or choose scaled-dot
    semantics.

    The optional `integer_core_lmul` attribute is a bounded resource/scheduling
    knob selecting the BASE LMUL of the per-sub-block integer-MAC widening chain
    i8 -> i16 -> i32 (the *how*, never the *what*; the integer accumulation order
    is untouched). It anchors the chain at "mf2" (absent/default -- today's
    e8mf2(8)->i16m1->e32m2 form, 4 strips/sub-block, NO fold-back), "m1"
    (e8m1->i16m2->e32m4, 2 strips, the running aux32 holds 16 lanes whose
    group-of-8 residues are folded back), or "m2" (e8m2->i16m4->e32m8, 1 strip ==
    the whole 32-element sub-block, 32 lanes folded back). At "m1"/"m2" the wide
    aux32 is collapsed to the canonical 8-lane vint32m2 by the VLEN-agnostic
    integer fold-back (vslidedown the wide register by the literal element offset
    8*k + vadd in the wide type + a final vget(.,0) of the low 8-lane group) BEFORE
    the deferred fp fold (Region F) consumes it -- the q4_K capability flip this
    brick auto-constructs. "m2" is the hard ceiling (i8m2 == 32 elements == exactly
    one sub-block under one scalar scale; a wider m4 base would need an illegal
    i32m16 product AND would fold TWO sub-blocks under one scalar). The verifier
    pins the legal set {"mf2","m1","m2"} fail-closed (I7). The integer add being
    associative/order-free, the fold-back is provably bit-exact at every legal
    anchor (there is NO fp non-associativity here -- that is Region F, deferred).

    The op emits the byte-exact form for the VLEN>=128 capability class (rv64gcv
    => Zvl128b => VLEN>=128, the validated target). No further resource shape knobs
    yet.

    LOOP-CAPABLE per-super-block-source form (M-FLAT q4_K milestone-2): the
    OPTIONAL `block_index` operand toggles the per-super-block-source form. When
    present it is the induction variable (region argument 0) of the enclosing
    tcrv_rvv.typed_super_block_block_dot_loop_body region: the q8 activation strip
    is read at `q8_base + block_index*stride + activation_quant_byte_offset` (the
    stride the loop op carries as activation_block_stride), byte-exact to the
    monolithic super_block_base_y + q8-offset arithmetic. The OPTIONAL
    `activation_quant_byte_offset` locates the q8 quant bytes within the q8_K
    activation super-block (4, after the fp32 activation scale); it is only legal
    with block_index (the loop form -- the standalone single-super-block form
    passes the q8 data pointer directly at offset 0, fail-closed otherwise). When
    block_index is ABSENT the op keeps the unchanged single-super-block form. The
    single-block strong paths never set block_index and are untouched.
  }];

  let arguments = (ins
    AnyType:$aux8_base,
    AnyType:$scales_base,
    AnyType:$q8_base,
    AnyType:$vl,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$activation_quant_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$aux8_base `,` $scales_base `,` $q8_base `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($aux8_base) `,` type($scales_base) `,` type($q8_base) `,` type($vl) "
    "`->` type($result)";
  let hasVerifier = 1;
}

def Q4KMinTermOp
    : TCRVRVV_Op<"q4_k_min_term",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV q4_K/q5_K super-block MIN term (sumf -= dmin * "
      "sum(mins * bsums)) op";
  let description = [{
    Records ONE super-block's q4_K/q5_K MIN term -- the `sumf -= dmin *
    sum_j(mins[j/2] * bsums[j])` correction (the offset-binary min subtraction
    of the ggml `ggml_vec_dot_q4_K_q8_K` block dot, quants.c:714) as ONE bounded
    generic plugin-local typed RVV dataflow step -- the Track B q4_K
    auto-construction's BRICK 4 (sibling to BRICK 1's tcrv_rvv.q4_k_nibble_unpack,
    BRICK 2's tcrv_rvv.q4_k_scale_min_bit_dance, and BRICK 3's
    tcrv_rvv.q4_k_scaled_dot). The MIN term operates on the SCALAR `sumf`
    accumulator, a DIFFERENT accumulator chain from the `sums` vector dot (BRICK
    3 / the deferred fp32 fold): over the 16 i16 bsums of the q8_K activation
    super-block, the integer reduction sumi = sum_j(bsums[j] * mins[j/2]) pairs
    each int16 bsums entry (SIGN-extended) with the BRICK 2 decoded uint6 min
    `mins[j/2]` (each min spans TWO consecutive bsums), then the single fp
    contraction `sumf -= dmin * (float)sumi` subtracts the offset-binary min
    correction, where dmin = fp16(x.dmin) * y.d is the weight min scale times the
    activation scale. The integer reduction is associative/order-free (byte-exact
    at any reduction tree), and the final `dmin * sumi` is a single fixed-order
    scalar fp multiply, so the whole term is anchor-independent -- BRICK 3's
    integer fold-back already collapsed the aux32 dot to canonical-8, there is NO
    fp-reassociation seam here.

    The nibble unpack, the 6-bit scale/min bit-dance, the per-sub-block scaled
    i32 dot, the deferred fp32 positive fold, the super-block loop scaffold, and
    the post-loop horizontal fold (BRICKS 1, 2, 3, and the deferred bricks 5-8)
    are NOT this op -- they are the sibling bricks and the monolithic
    tcrv_rvv.q4_k_q8_k_block_dot op, which STAY.

    The op is nested directly under tcrv_rvv.with_vl, consumes THREE runtime ABI
    value operands -- the q4_K/q5_K weight base pointer (const uint8_t *, read at
    weight_dmin_byte_offset for the fp16 dmin), the BRICK 2 decoded scales base
    pointer (const uint8_t *, whose bytes [8..15] are the 8 decoded uint6 mins),
    and the q8_K activation base pointer (const uint8_t *, read at
    bsums_byte_offset for the int16 bsums and at offset 0 for the fp32
    activation scale) -- plus the active !tcrv_rvv.vl token, and produces ONE i32
    LMUL m1 !tcrv_rvv.vector completion token (the term mutates a scalar sumf
    accumulator as a side effect; the standalone lowering declares its own sumf
    and stores it to a local sink as the observable). The MIN term is a SCALAR
    sequence (no RVV vector intrinsics for the reduction); the only opaque call
    is the fp16 dmin read.

    The block-quantized super-block STRUCTURE is first-class in the typed body.
    The structural facts -- QK_K=256 (super-block size), 8 sub-blocks of 32
    elements, the 16 i16 bsums, the q8_K bsums byte offset (260) and the weight
    dmin byte offset (2) -- are carried as typed integer attributes (I4 mirror
    only). The fixed emission mirrors `_generic` (quants.c) line-for-line so
    byte-exactness is by construction: the SAME node sequence is shared VERBATIM
    by the monolithic q4_K/q5_K block dot (which emits the reduction before its
    positive fold and the subtract after it -- the two halves are interleaved in
    the super-block loop but data-independent) AND this first-class op's lowering,
    so both emit byte-identical MIN-term C. The bounded surface kind is limited to
    "q4_k_min_term". Route ids, helper names, ABI strings, artifact names, status
    fields, exact intrinsic spellings, operand mirror strings, q8/q4 names, and
    common EmitC/export code must not infer or choose min-term semantics.

    The op emits the byte-exact form for the VLEN>=128 capability class (rv64gcv
    => Zvl128b => VLEN>=128, the validated target). No resource shape knobs (the
    MIN term is scalar; there is no LMUL widening axis).

    LOOP-CAPABLE per-super-block-source form (M-FLAT q4_K milestone-2): the
    OPTIONAL `block_index` operand toggles the per-super-block-source form. When
    present it is the induction variable (region argument 0) of the enclosing
    tcrv_rvv.typed_super_block_block_dot_loop_body region: the fp16 dmin + fp32
    activation d + int16 bsums are read at `weight_base + block_index*weight_stride`
    and `activation_base + block_index*activation_stride` (the strides the loop op
    carries), byte-exact to the monolithic super_block_base_x/y arithmetic. When
    ABSENT the op keeps the unchanged single-super-block form. The single-block
    strong paths never set block_index and are untouched.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$scales_base,
    AnyType:$activation_base,
    AnyType:$vl,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$num_sub_blocks,
    I64Attr:$bsums_byte_offset,
    I64Attr:$weight_dmin_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $scales_base `,` $activation_base `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($scales_base) `,` type($activation_base) `,` "
    "type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def Q4KSumsFoldScaleDOp
    : TCRVRVV_Op<"q4_k_sums_fold_scale_d",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV q4_K/q5_K super-block positive sums fold (sums += "
      "fp16(x.d) * y.d * (float)aux32) op";
  let description = [{
    Records ONE super-block's q4_K/q5_K DEFERRED two-level fp32 POSITIVE fold --
    the `sums += d * (float)aux32` accumulation where `d = fp16(x.d) * y.d` is the
    fp16 weight super-block scale (byte 0) times the fp32 activation scale -- as
    ONE bounded generic plugin-local typed RVV dataflow step, the Track B q4_K
    auto-construction's BRICK 6 (sibling to BRICK 1's tcrv_rvv.q4_k_nibble_unpack,
    BRICK 2's tcrv_rvv.q4_k_scale_min_bit_dance, BRICK 3's tcrv_rvv.q4_k_scaled_dot,
    and BRICK 4's tcrv_rvv.q4_k_min_term). The positive fold operates on the `sums`
    8-lane fp32 VECTOR accumulator, a DIFFERENT accumulator chain from the BRICK 4
    SCALAR `sumf` min term: it converts the BRICK 3 canonical-8 integer aux32 to
    fp32 (vfcvt), multiplies the whole vector by the SEPARATE per-super-block scale
    `d` (vfmul.vf, NEVER an fma), and adds it into the carried `sums` (vfadd.vv).
    The per-sub-block uint6 scale is NOT read here -- BRICK 3 already FUSED it into
    aux32 via the vwmacc.vx, so the deferred fold reads ONLY the canonical-8 aux32
    and the two fp16/fp32 super-block scales. BRICK 3's integer fold-back already
    collapsed the wide aux32 to canonical-8 vint32m2 BEFORE this fold, so the fp
    conversion + scale + accumulate is a fixed-order 8-lane sequence -- there is NO
    fp-reassociation seam in this brick (the non-associative horizontal collapse is
    the post-loop bricks 7-8, NOT this op).

    The nibble unpack, the 6-bit scale/min bit-dance, the per-sub-block scaled i32
    dot, the MIN term, the super-block loop scaffold, and the post-loop horizontal
    fold (BRICKS 1, 2, 3, 4, and the deferred bricks 5, 7, 8) are NOT this op --
    they are the sibling bricks and the monolithic tcrv_rvv.q4_k_q8_k_block_dot op,
    which STAY. In the monolithic block dot this positive fold sits PHYSICALLY
    INTERLEAVED between BRICK 4's two halves (the bsums reduction emits before it,
    the dmin subtract after it), but it is DATA-INDEPENDENT of the MIN chain.

    The op is nested directly under tcrv_rvv.with_vl, consumes THREE runtime ABI
    value operands -- the q4_K/q5_K weight base pointer (const uint8_t *, read at
    weight_d_byte_offset for the fp16 super-block scale d), the BRICK 3 canonical-8
    aux32 base pointer (const int32_t *, the 8-lane integer dot result, vle32-loaded
    into a vint32m2_t), and the q8_K activation base pointer (const uint8_t *, read
    at offset 0 for the fp32 activation scale y.d) -- plus the active !tcrv_rvv.vl
    token, and produces ONE i32 LMUL m1 !tcrv_rvv.vector completion token (the fold
    mutates an 8-lane fp32 sums accumulator as a side effect; the standalone
    lowering declares its own sums, vle32-loads the aux32, and stores the resulting
    sums via vse32 to a local float sink as the observable).

    The block-quantized super-block STRUCTURE is first-class in the typed body. The
    structural facts -- QK_K=256 (super-block size), 8 sub-blocks of 32 elements,
    and the weight d (fp16) byte offset (0) -- are carried as typed integer
    attributes (I4 mirror only). The fixed emission mirrors `_generic` (quants.c
    line 711) so byte-exactness is by construction: the SAME node sequence is shared
    VERBATIM by the monolithic q4_K/q5_K block dot (which emits it inside the
    super-block loop between the MIN-term halves) AND this first-class op's lowering,
    so both emit byte-identical positive-fold C. The bounded surface kind is limited
    to "q4_k_sums_fold_scale_d". Route ids, helper names, ABI strings, artifact
    names, status fields, exact intrinsic spellings, operand mirror strings, q8/q4
    names, and common EmitC/export code must not infer or choose fold semantics.

    The op emits the byte-exact form for the VLEN>=128 capability class (rv64gcv
    => Zvl128b => VLEN>=128, the validated target). No resource shape knobs (the
    canonical-8 fp fold is fixed at 8 lanes f32m2; the integer-core LMUL widening
    axis lives on the upstream BRICK 3 scaled dot).

    LOOP-CAPABLE per-super-block-source form (M-FLAT q4_K milestone-2): the
    OPTIONAL `block_index` operand toggles the per-super-block-source form. When
    present it is the induction variable (region argument 0) of the enclosing
    tcrv_rvv.typed_super_block_block_dot_loop_body region: the fp16 weight scale d
    + fp32 activation d are read at `weight_base + block_index*weight_stride` and
    `activation_base + block_index*activation_stride` (the strides the loop op
    carries), byte-exact to the monolithic super_block_base_x/y arithmetic. When
    ABSENT the op keeps the unchanged single-super-block form. The single-block
    strong paths never set block_index and are untouched.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$aux32_base,
    AnyType:$activation_base,
    AnyType:$vl,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$num_sub_blocks,
    I64Attr:$weight_d_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$weight_base `,` $aux32_base `,` $activation_base `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($aux32_base) `,` type($activation_base) `,` "
    "type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def Q4KHorizontalFoldOp
    : TCRVRVV_Op<"q4_k_horizontal_fold",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV q4_K/q5_K POST-LOOP horizontal fold (sumf += "
      "sums8[0..7], sequential) op";
  let description = [{
    Records the q4_K/q5_K block dot's POST-LOOP horizontal fold -- the final
    collapse of the 8-lane fp32 `sums` VECTOR accumulator into the carried scalar
    `sumf` (which already holds the in-loop MIN subtractions) -- as ONE bounded
    generic plugin-local typed RVV dataflow step, the Track B q4_K
    auto-construction's BRICK 7 (sibling to BRICK 1's tcrv_rvv.q4_k_nibble_unpack,
    BRICK 2's tcrv_rvv.q4_k_scale_min_bit_dance, BRICK 3's tcrv_rvv.q4_k_scaled_dot,
    BRICK 4's tcrv_rvv.q4_k_min_term, and BRICK 6's
    tcrv_rvv.q4_k_sums_fold_scale_d). It is the LAST step of the fp domain, emitted
    AFTER the super-block loop: a `vse32_v_f32m2(&sums8[0], sums, 8)` materializes
    the 8-lane `sums` accumulator into a `float sums8[8]` scratch, then the FIXED
    sequential ascending `for (l = 0; l < num_lanes; ++l) sumf += sums8[l]`
    (num_lanes == 8) accumulates the 8 lanes into `sumf` STARTING from its carried
    value. This horizontal sum is anchor-INDEPENDENT (always 8 lanes, fixed
    ascending order, NEVER a vfredusum) -- it mirrors `_generic`'s
    `for (l = 0; l < 8; ++l) sumf += sums[l]` after the super-block loop, so it is
    byte-exact with NO fp-reassociation seam.

    The nibble unpack, the 6-bit scale/min bit-dance, the per-sub-block scaled i32
    dot, the MIN term, the deferred fp32 positive fold, and the super-block loop
    scaffold (BRICKS 1, 2, 3, 4, 6, and the deferred BRICK 5 FOR-loop container) are
    NOT this op -- they are the sibling bricks and the monolithic
    tcrv_rvv.q4_k_q8_k_block_dot op, which STAY. In the monolithic block dot this
    fold sits AFTER the loop, reading the loop-carried `sums`/`sumf` and writing the
    fp32 ABI output `*s`; in the standalone lowering the op declares its OWN sums
    (vle32-loaded from the ABI source) + sumf (seeded 0) + sums8 scratch and stores
    the resulting sumf to a local sink as the observable.

    The op is nested directly under tcrv_rvv.with_vl, consumes ONE runtime ABI value
    operand -- the 8-lane fp32 `sums` source base pointer (const float *,
    vle32-loaded into a vfloat32m2_t) -- plus the active !tcrv_rvv.vl token, and
    produces ONE i32 LMUL m1 !tcrv_rvv.vector completion token (the fold mutates the
    scalar sumf accumulator and a local sink as a side effect).

    The block-quantized super-block STRUCTURE is first-class in the typed body. The
    structural facts -- QK_K=256 (super-block size), 8 sub-blocks of 32 elements, and
    the canonical fp32 lane count (num_lanes == 8) -- are carried as typed integer
    attributes (I4 mirror only). The fixed emission mirrors `_generic` (quants.c line
    718) so byte-exactness is by construction: the SAME node sequence is shared
    VERBATIM by the monolithic q4_K/q5_K block dot (which emits it after the
    super-block loop) AND this first-class op's lowering, so both emit byte-identical
    horizontal-fold C. The bounded surface kind is limited to "q4_k_horizontal_fold".
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/q4 names, and common EmitC/export
    code must not infer or choose fold semantics.

    The op emits the byte-exact form for the VLEN>=128 capability class (rv64gcv
    => Zvl128b => VLEN>=128, the validated target). No resource shape knobs (the
    horizontal collapse is fixed at 8 lanes f32m2 and a fixed sequential ascending
    sum; the integer-core LMUL widening axis lives on the upstream BRICK 3 scaled
    dot).
  }];

  let arguments = (ins
    AnyType:$sums_base,
    AnyType:$vl,
    StrAttr:$kind,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$num_sub_blocks,
    I64Attr:$num_lanes
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$sums_base `,` $vl attr-dict `:` "
    "type($sums_base) `,` type($vl) `->` type($result)";
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

def GgmlBlockDotQ2KQ8KIntegerCoreOp
    : TCRVRVV_Op<"q2_k_q8_k_integer_core",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml Q2_K x Q8_K super-block SCALAR integer-core op "
      "(isum + summs, the milestone-1 q2_K brick)";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_q2_K_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step -- the
    q2_K first-super-block (M-FLAT) milestone-1 integer brick. It reproduces
    ggml's per-super-block SCALAR integer state EXACTLY as `_generic` computes it
    right BEFORE the fp32 `d`/`dmin` multiply, byte-exact (i32 equality):
    - `isum` = Σ_{s=0..15} (sc[s] & 0xF) * (Σ_{l in sub-block s} q8[l]*aux8[l]),
      the per-sub-block uint4-nibble-scaled POSITIVE integer dot (the 2-bit
      weights `(qs >> shift) & 3` unpacked to aux8 in [0,3], the q8_K
      activations, and the 16 direct `scales[16]` LOW nibbles as the
      per-sub-block scales); and
    - `summs` = Σ_{s=0..15} bsums[s] * (sc[s] >> 4), the q2_K MIN integer sum (the
      int16 q8_K per-sub-block sums bsums @260, sign-extended, times the
      per-sub-block min HIGH nibbles).
    Both are produced as SCALAR i32 SSA results (ggml's `int isum` / `int summs`
    locals), NOT an 8-lane vector -- this is the STRUCTURAL contrast with q6_K's
    8-lane `aux32` vector core: q2_K's positive fold is a single per-super-block
    scalar (`sumf += dall*isum - dmin*summs`), so its integer core collapses to
    two scalars. The DEFERRED fp32 SCALAR fold (`dall = fp16(x.d @80)*y.d`,
    `dmin = fp16(x.dmin @82)*y.d`, `sumf += dall*isum - dmin*summs`) is
    DELIBERATELY OUT OF SCOPE here -- that is the milestone-2 increment. This
    milestone-1 op isolates the genuinely-new q2_K machinery (the 2-bit unpack,
    the PLAIN-nibble scale/min extraction with NO 6-bit utmp/kmask bit-dance, the
    per-sub-block scaled scalar dot, the scalar min sum) from the fp32 fold.

    The op is nested directly under tcrv_rvv.with_vl and consumes the q2_K weight
    base pointer (vx), the q8_K activation base pointer (vy), the runtime element
    count (n), the active !tcrv_rvv.vl token, and an OPTIONAL `block_index`
    induction operand. It produces TWO scalar i32 SSA results (isum, summs). It
    carries NO output pointer -- unlike q6_K's aux32[8] memory state, q2_K's isum/
    summs are scalar registers.

    The block-quantized super-block STRUCTURE is first-class in the typed body.
    The structural facts -- QK_K=256, 16 sub-blocks of 16 elements, the AoS block
    strides (84 bytes for block_q2_K, 292 bytes for block_q8_K), the q2_K payload
    byte offsets (the 16 packed 4-bit-scale/4-bit-min bytes `scales` @0, the 64
    packed 2-bit-weight bytes qs @16) and the q8_K payload byte offsets (qs @4,
    the int16 per-sub-block sums bsums @260) -- are carried as typed
    integer/string attributes (I4 mirror only). The bounded surface kind is
    limited to "ggml_q2_k_q8_k_integer_core" and scale_model to
    "per-sub-block-uint4-scale-i32-domain-min". Route ids, helper names, ABI
    strings, artifact names, status fields, exact intrinsic spellings, operand
    mirror strings, q8/q2 names, and common EmitC/export code must not infer or
    choose super-block dot-product semantics.

    LOOP-CAPABLE per-super-block-source form (M-FLAT q2_K milestone-1): the
    OPTIONAL `block_index` operand toggles the per-super-block-source form. When
    present it is the induction variable (region argument 0) of an enclosing
    tcrv_rvv.typed_super_block_block_dot_loop_body region: the q2_K weight strip
    and the q8_K activation strip are read at `weight_base + block_index*84` and
    `activation_base + block_index*292`. When ABSENT the op keeps the standalone
    single-super-block form. milestone-1 builds ONLY the ODS + verifier (the
    honest target IR parses + verifies + clears the [L-8] allowlist); the
    byte-exact emit is milestone-2.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$activation_bsums_byte_offset
  );
  let results = (outs AnyType:$isum, AnyType:$summs);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($element_count) `,` type($vl) `->` type($isum) `,` type($summs)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ1SQ8KGridCoreOp
    : TCRVRVV_Op<"iq1_s_q8_k_grid_core",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ1_S x Q8_K super-block TERNARY-grid SCALAR "
      "integer-core op (sumi + sumi1, the milestone-1 iq1_s grid brick)";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_iq1_s_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step -- the
    iq1_s first-super-block (M-FLAT) milestone-1 TERNARY-grid brick, the grid /
    super-block-flavor sibling of the q2_K scalar integer core
    (tcrv_rvv.q2_k_q8_k_integer_core). It is the FIRST super-block typed body that
    decodes with a codebook GRID GATHER (decode_model=lookup) instead of an
    arithmetic bit-unpack: it reproduces ggml's per-super-block SCALAR integer
    state EXACTLY as `_generic` computes it right BEFORE the fp32 `d` multiply,
    byte-exact (i32 equality):
    - `sumi`  = Σ_{ib=0..7} ls[ib] * (Σ_{l=0..3, j=0..7} q8[j] * grid[j]), the
      per-sub-block qh-scaled POSITIVE ternary-grid dot. Each sub-block's 4 grid
      indices are `idx = qs[ib*4+l] | (((qh[ib] >> 3*l) & 7) << 8)` (an 11-bit
      index in [0,2047]); each indexes the 2048-entry TERNARY iq1s_grid (8 packed
      signed {-1,0,+1} int8 grid bytes per uint64 entry); the per-sub-block scale
      is `ls = 2*((qh[ib] >> 12) & 7) + 1`; and
    - `sumi1` = Σ_{ib=0..7} ls[ib] * delta[ib] * (bsums[2*ib] + bsums[2*ib+1]),
      the iq1_s DELTA integer sum (the q8_K per-sub-block int16 sums bsums @260,
      times the per-sub-block scale ls and the sign `delta = 1 - 2*((qh[ib] >>
      15) & 1)`).
    Both are produced as SCALAR i32 SSA results (ggml's `int sumi` / `int sumi1`
    locals), NOT an 8-lane vector -- the STRUCTURAL contrast with q6_K's 8-lane
    `aux32` vector core, and the arity-match with the q2_K scalar core (two scalar
    states). The DEFERRED fp32 SCALAR fold (`d = fp16(x.d @0) * y.d @0`, `sumf +=
    d * ((float)sumi + IQ1S_DELTA*(float)sumi1)`, IQ1S_DELTA=0.125f) is
    DELIBERATELY OUT OF SCOPE here -- that is the milestone-2 increment. This
    milestone-1 op isolates the genuinely-new iq1_s machinery (the 11-bit
    grid-index build from qs+qh, the vluxei16 ternary-grid gather, the signed
    widening grid dot, the qh-encoded per-sub-block scale, the delta-bsum min-like
    sum) from the fp32 fold.

    The op is nested directly under tcrv_rvv.with_vl and consumes the iq1_s weight
    base pointer (vx), the q8_K activation base pointer (vy), the runtime element
    count (n), the active !tcrv_rvv.vl token, and an OPTIONAL `block_index`
    induction operand. It produces TWO scalar i32 SSA results (sumi, sumi1). It
    carries NO output pointer -- like the q2_K scalar core, iq1_s's sumi/sumi1 are
    scalar registers, NOT an aux32 memory state.

    The block-quantized super-block STRUCTURE is first-class in the typed body.
    The structural facts -- QK_K=256, 8 sub-blocks of 32 elements, the AoS block
    strides (50 bytes for block_iq1_s, 292 bytes for block_q8_K), the iq1_s
    payload byte offsets (the 32 uint8 grid-index bytes qs @2, the uint16 qh[8]
    plane @34 carrying the grid-high-3-bit fields + the 3-bit scale @12..14 + the
    delta sign @15) and the q8_K payload byte offsets (qs @4, the int16
    per-sub-block sums bsums @260) -- are carried as typed integer/string
    attributes (I4 mirror only). The bounded surface kind is limited to
    "ggml_iq1_s_q8_k_grid_core" and scale_model to
    "per-sub-block-qh-scale-ternary-grid-codebook-delta-bsum-int-domain". The
    FIXED 2048-entry iq1s_grid TERNARY codebook is a byte-exact constant of the
    iq1_s FORMAT (like the q2_K scale/qs byte offsets); it is emitted at
    milestone-2 keyed by the brick op identity, NOT carried in the IR here. Route
    ids, helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, operand mirror strings, q8/iq1 names, and common EmitC/export code
    must not infer or choose the ternary-grid super-block dot-product semantics.

    LOOP-CAPABLE per-super-block-source form (M-FLAT iq1_s milestone-1): the
    OPTIONAL `block_index` operand toggles the per-super-block-source form. When
    present it is the induction variable (region argument 0) of an enclosing
    tcrv_rvv.typed_super_block_block_dot_loop_body region whose fold_model is
    "scalar_delta_grid": the iq1_s weight strip and the q8_K activation strip are
    read at `weight_base + block_index*50` and `activation_base + block_index*292`.
    When ABSENT the op keeps the standalone single-super-block form. milestone-1
    builds ONLY the ODS + verifier (the honest target IR parses + verifies +
    clears the [L-8] allowlist); the byte-exact grid-gather emit is milestone-2.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_qh_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$activation_bsums_byte_offset
  );
  let results = (outs AnyType:$sumi, AnyType:$sumi1);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($element_count) `,` type($vl) `->` type($sumi) `,` type($sumi1)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ1MQ8KGridCoreOp
    : TCRVRVV_Op<"iq1_m_q8_k_grid_core",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ1_M x Q8_K super-block TERNARY-grid SCALAR "
      "integer-core op (sumi1 + sumi2, the iq1_m grid brick, iq1_s sibling)";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_iq1_m_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step -- the
    iq1_m TERNARY-grid brick, the packed-scale / half-scale / per-group-delta sibling
    of the iq1_s grid core (tcrv_rvv.iq1_s_q8_k_grid_core). It decodes with the SAME
    codebook GRID GATHER (decode_model=lookup, the SAME 2048-entry TERNARY iq1s_grid) as
    iq1_s but reproduces ggml's per-super-block SCALAR integer state for iq1_m EXACTLY
    as `_generic` computes it right BEFORE the fp32 `d` multiply, byte-exact
    (i32 equality):
    - `sumi1` = Σ_{ib=0..7} (sum1[0]*ls1 + sum1[1]*ls2), the qh-index grid dot,
      where sum1[h] = Σ_{group∈half h, j} q8[j] * grid[j] (each of the 4 grid indices
      is `idx = qs[ib*4+l] | (((uint16)qh[ib*2 + l/2] << (8-4*(l%2))) & 0x700)`, an
      11-bit index into the 2048-entry TERNARY iq1s_grid), and the TWO per-sub-block
      half scales are `ls1 = 2*((sc[ib/2]>>(6*(ib%2)+0))&7)+1` (groups 0..1) and
      `ls2 = 2*((sc[ib/2]>>(6*(ib%2)+3))&7)+1` (groups 2..3); and
    - `sumi2` = Σ_{ib=0..7} (sum2[0]*ls1 + sum2[1]*ls2), the iq1_m DELTA integer sum,
      where sum2[h] = Σ_{group∈half h} delta[l] * (Σ q8[j]) with FOUR independent
      per-group signs `delta[l] = 1 - 2*((qh[ib*2 + l/2] >> (l%2 ? 7 : 3)) & 1)`. The
      four independent signs make the q8 bsums UNUSABLE, so each group reduces a FRESH
      Σq8 from the SAME loaded q8 vector (iq1_m reads NO bsums region).
    Both are produced as SCALAR i32 SSA results (ggml's `int sumi1` / `int sumi2`
    locals), NOT an 8-lane vector -- the arity-match with the iq1_s scalar grid core
    (two scalar states). The DEFERRED fp32 SCALAR fold (the packed iq1m_scale fp16
    RECONSTRUCT `scale.u16 = (sc[0]>>12)|((sc[1]>>8)&0xf0)|((sc[2]>>4)&0xf00)|
    (sc[3]&0xf000)` read as _Float16, `d = scale * y.d @0`, `sumf += d * ((float)sumi1
    + IQ1M_DELTA*(float)sumi2)`, IQ1M_DELTA=0.125f) is DELIBERATELY OUT OF SCOPE here
    -- it is emitter-inlined by the scalar-delta-grid lowering. This op isolates the
    genuinely-new iq1_m machinery (the packed-scale half split, the per-half grid dot,
    the per-group four-sign delta) from the fp32 fold.

    The op is nested directly under tcrv_rvv.with_vl and consumes the iq1_m weight
    base pointer (vx), the q8_K activation base pointer (vy), the runtime element
    count (n), the active !tcrv_rvv.vl token, and an OPTIONAL `block_index` induction
    operand. It produces TWO scalar i32 SSA results (sumi1, sumi2). It carries NO
    output pointer -- like the iq1_s scalar grid core, iq1_m's sumi1/sumi2 are scalar
    registers, NOT an aux32 memory state.

    The block-quantized super-block STRUCTURE is first-class in the typed body. The
    structural facts -- QK_K=256, 8 sub-blocks of 32 elements, the AoS block strides
    (56 bytes for block_iq1_m -- NO fp16 d field -- and 292 bytes for block_q8_K), the
    iq1_m payload byte offsets (the 32 uint8 grid-index bytes qs @0, the 16 uint8
    qh[QK_K/16] plane @32 carrying the grid-high-3-bit fields + the 4 per-group delta
    signs, the 4 uint16 packed scales[] words @48 carrying the packed fp16 d nibbles +
    the per-sub-block 3-bit half scales) and the q8_K payload byte offsets (qs @4) --
    are carried as typed integer/string attributes (I4 mirror only). iq1_m reads NO
    bsums. The bounded surface kind is limited to "ggml_iq1_m_q8_k_grid_core" and
    scale_model to "packed-iq1m-scale-per-half-scale-ternary-grid-codebook-per-group-
    delta-int-domain". The FIXED 2048-entry iq1s_grid TERNARY codebook is a byte-exact
    constant of the iq1_m FORMAT; it is emitted keyed by the brick op identity, NOT
    carried in the IR here. Route ids, helper names, ABI strings, artifact names,
    status fields, exact intrinsic spellings, operand mirror strings, q8/iq1 names, and
    common EmitC/export code must not infer or choose the ternary-grid super-block
    dot-product semantics.

    LOOP-CAPABLE per-super-block-source form: the OPTIONAL `block_index` operand
    toggles the per-super-block-source form. When present it is the induction variable
    (region argument 0) of an enclosing tcrv_rvv.typed_super_block_block_dot_loop_body
    region whose fold_model is "scalar_delta_grid": the iq1_m weight strip and the
    q8_K activation strip are read at `weight_base + block_index*56` and
    `activation_base + block_index*292`. When ABSENT the op keeps the standalone
    single-super-block form.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$sub_block,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_qh_byte_offset,
    I64Attr:$weight_scales_byte_offset,
    I64Attr:$activation_quant_byte_offset
  );
  let results = (outs AnyType:$sumi1, AnyType:$sumi2);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($element_count) `,` type($vl) `->` type($sumi1) `,` type($sumi2)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ3XXSQ8KGridCoreOp
    : TCRVRVV_Op<"iq3_xxs_q8_k_grid_core",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ3_XXS x Q8_K super-block GRID-of-4 SCALAR "
      "integer-core op (bsum, the iq3_xxs grid brick, iq1_s grid sibling)";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_iq3_xxs_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step -- the
    iq3_xxs GRID-of-4 brick, the i32-grid / per-sign-group-signs sibling of the iq1_s
    ternary-grid core (tcrv_rvv.iq1_s_q8_k_grid_core). It decodes with a codebook GRID
    GATHER (decode_model=lookup) like iq1_s, but over the 256-entry `iq3xxs_grid[256]`
    uint32 GRID-of-4 (each entry FOUR int8 values -- HALF the lane width of iq1_s's
    uint64 grid-of-8) via a `vluxei16_v_i32m1` indexed gather, with per-element signs
    read from the `ksigns_iq2xs[128]` SIGN plane. It reproduces ggml's per-super-block
    SCALAR integer state EXACTLY as `_generic` computes `bsum` right BEFORE the fp32
    `d` multiply, byte-exact (i32 equality):
    - `bsum` = Σ_{ib=0..7} ls[ib] * (Σ_{l=0..3} Σ_{lane} q8[lane] * signed_grid[lane]),
      the per-sub-block 4-bit-scaled grid/sign dot. Each sub-block reads its aux word
      `aux32` (little-endian from the 4 gas bytes at `gas + ib*4`); the per-sub-block
      scale is `ls = 2*(aux32 >> 28) + 1`; each of the 4 sign groups l=0..3 reads the
      sign byte `signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]` and the two grid INDICES
      `idx1 = q3[8*ib+2l+0]`, `idx2 = q3[8*ib+2l+1]` (an 8-lane vluxei16 gather over the
      i32 grid; lanes 0..3 = grid1/kmask{1,2,4,8}/q8[0..3], lanes 4..7 = grid2/kmask{16,
      32,64,128}/q8[4..7]), applies the per-lane sign via vmerge(grid, vneg(grid), mask)
      and accumulates the signed widening product `vwmul` + `vwredsum`.
    The result is produced as ONE SCALAR i32 SSA result (ggml's per-super-block `int
    bsum`), NOT an 8-lane vector -- the arity-cousin of the iq1_s scalar grid core. The
    DEFERRED fp32 SCALAR fold (`d = fp16(x.d @0) * y.d @0`, `sumf += d * (float)bsum`,
    then the trailing `*s = 0.25f * sumf` factor) is DELIBERATELY OUT OF SCOPE here --
    it is emitter-inlined by the scalar-delta-grid lowering. This op isolates the
    genuinely-new iq3_xxs machinery (the i32 GRID-of-4 vluxei16 gather, the aux32
    4-bit-scale + 4-sign-group ksigns decode, the two-index-per-group assembly) from
    the fp32 fold.

    The op is nested directly under tcrv_rvv.with_vl and consumes the iq3_xxs weight
    base pointer (vx), the q8_K activation base pointer (vy), the runtime element count
    (n), the active !tcrv_rvv.vl token, and an OPTIONAL `block_index` induction operand.
    It produces ONE scalar i32 SSA result (bsum). It carries NO output pointer -- like
    the iq1_s scalar grid core, iq3_xxs's bsum is a scalar register.

    The block-quantized super-block STRUCTURE is first-class in the typed body. The
    structural facts -- QK_K=256, 8 sub-blocks of 32 elements, the AoS block strides
    (98 bytes for block_iq3_xxs = {fp16 d; uint8 qs[96]}, 292 bytes for block_q8_K), the
    iq3_xxs payload byte offsets (the fp16 d @0, the 64 uint8 grid-index bytes qs @2, the
    32 aux bytes gas @66) and the q8_K payload byte offsets (fp32 d @0, int8 qs @4) --
    are carried as typed integer/string attributes (I4 mirror only). The FIXED 256-entry
    iq3xxs_grid GRID-of-4 codebook and the 128-entry ksigns_iq2xs sign plane are
    byte-exact constants of the iq3_xxs FORMAT; they are emitted keyed by the brick op
    identity, NOT carried in the IR here. The bounded surface kind is limited to
    "ggml_iq3_xxs_q8_k_grid_core" and scale_model to
    "per-sub-block-aux32-scale-grid-of-4-codebook-ksigns-sign-plane-int-domain". Route
    ids, helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, operand mirror strings, q8/iq3 names, and common EmitC/export code must
    not infer or choose the grid-of-4 super-block dot-product semantics.

    LOOP-CAPABLE per-super-block-source form: the OPTIONAL `block_index` operand toggles
    the per-super-block-source form. When present it is the induction variable (region
    argument 0) of an enclosing tcrv_rvv.typed_super_block_block_dot_loop_body region
    whose fold_model is "scalar_delta_grid": the iq3_xxs weight strip and the q8_K
    activation strip are read at `weight_base + block_index*98` and `activation_base +
    block_index*292`. When ABSENT the op keeps the standalone single-super-block form.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
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
    I64Attr:$activation_quant_byte_offset
  );
  let results = (outs AnyType:$bsum);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($element_count) `,` type($vl) `->` type($bsum)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ2XXSQ8KGridCoreOp
    : TCRVRVV_Op<"iq2_xxs_q8_k_grid_core",
    [TCRVEmitCLowerableOpInterface,
     DeclareOpInterfaceMethods<TunableScheduleOpInterface>]> {
  let summary =
      "generic typed RVV ggml IQ2_XXS x Q8_K super-block GRID-of-8 SCALAR "
      "integer-core op (bsum, the iq2_xxs grid/signs64 brick, iq1_s grid sibling)";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_iq2_xxs_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step -- the
    iq2_xxs GRID-of-8 brick, the SIGN-PLANE (signs64) sibling of the iq3_xxs GRID-of-4
    core (tcrv_rvv.iq3_xxs_q8_k_grid_core) and the iq1_s ternary-grid core. It decodes
    with a codebook GRID GATHER (decode_model=lookup) like iq1_s, but over the 256-entry
    `iq2xxs_grid[256]` uint64 GRID-of-8 (each entry EIGHT int8 values -- TWICE the lane
    width of iq3_xxs's uint32 grid-of-4) via a `vluxei16_v_i64<anchor>` indexed gather,
    with per-element signs read via a SECOND `vluxei16_v_i64<anchor>` indexed gather over
    the DERIVED `keven_signs_q2xs` SIGN-PLANE (signs64 -- 1024 +-1 bytes = the 128-entry
    ksigns_iq2xs selector EXPANDED to per-lane +-1). It reproduces ggml's per-super-block
    SCALAR integer state EXACTLY as `_generic` computes `bsum` right BEFORE the fp32 `d`
    multiply, byte-exact (i32 equality):
    - `bsum` = Σ_{ib=0..7} ls[ib] * (Σ_{lane} signed_grid[lane] * q8[lane]), the
      per-sub-block 4-bit-scaled grid/sign dot over one 32-lane sub-block. Each sub-block
      reads its aux word pair from the INTERLEAVED qs stream at `qs + ib*8`: the 4 grid
      INDICES `idx = a[l]` (l=0..3, the low 4 bytes) and the aux1 uint32 `a[4]|a[5]<<8|
      a[6]<<16|a[7]<<24` (the high 4 bytes) -- iq2_xxs INTERLEAVES the indices and the
      aux pair, unlike iq3_xxs's SEPARATE q3/gas regions. The per-sub-block scale
      `ls = 2*(aux1 >> 28) + 1`; each of the 4 sign groups l=0..3 reads the sign selector
      `sel = (aux1 >> 7*l) & 127`. The 4 grid indices (idx*8 byte-offsets) and 4 sign
      selectors (sel*8 byte-offsets) are packed as u16 index arrays; TWO
      `vluxei16_v_i64<anchor>` gathers over the i64 grid64 / signs64 bases fetch the 32
      grid bytes + 32 +-1 sign bytes; the sign folds onto the GRID via
      `vmul_vv_i8<anchor>(grid, signs)` (NOT q8 -- the i8 product wraps at the q8=-128
      lane, so folding onto the [8,43]-bounded grid is byte-exact); then ONE
      `vwmul_vv_i16<2*anchor>` + ONE `vwredsum_vs_i16<2*anchor>_i32m1` per sub-block, then
      `bsum += sumi*ls`.
    The result is produced as ONE SCALAR i32 SSA result (ggml's per-super-block `int
    bsum`), NOT an 8-lane vector -- the arity-cousin of the iq1_s scalar grid core. The
    DEFERRED fp32 SCALAR fold (`d = fp16(x.d @0) * y.d @0`, `sumf += d * (float)bsum`,
    then the trailing `*s = 0.125f * sumf` factor) is DELIBERATELY OUT OF SCOPE here --
    it is emitter-inlined by the scalar-delta-grid lowering. This op isolates the
    genuinely-new iq2_xxs machinery (the i64 GRID-of-8 vluxei16 gather, the signs64
    sign-plane vluxei16 gather, the aux1 4-bit-scale + 4-sign-group decode) from the
    fp32 fold.

    The op is nested directly under tcrv_rvv.with_vl and consumes the iq2_xxs weight
    base pointer (vx), the q8_K activation base pointer (vy), the runtime element count
    (n), the active !tcrv_rvv.vl token, and an OPTIONAL `block_index` induction operand.
    It produces ONE scalar i32 SSA result (bsum). It carries NO output pointer -- like
    the iq1_s scalar grid core, iq2_xxs's bsum is a scalar register.

    The block-quantized super-block STRUCTURE is first-class in the typed body. The
    structural facts -- QK_K=256, 8 sub-blocks of 32 elements, the AoS block strides
    (66 bytes for block_iq2_xxs = {fp16 d; uint16 qs[32]}, 292 bytes for block_q8_K), the
    iq2_xxs payload byte offsets (the fp16 d @0, the 64 uint8 interleaved index+aux bytes
    qs @2) and the q8_K payload byte offsets (fp32 d @0, int8 qs @4) -- are carried as
    typed integer/string attributes (I4 mirror only). The FIXED 256-entry iq2xxs_grid
    GRID-of-8 codebook and the DERIVED keven_signs_q2xs signs64 sign plane are byte-exact
    constants of the iq2_xxs FORMAT; they are emitted keyed by the brick op identity, NOT
    carried in the IR here (the signs64 sign-plane mechanism is DERIVED at emit from the
    fixed ksigns selector, NO op-attr extension). The bounded surface kind is limited to
    "ggml_iq2_xxs_q8_k_grid_core" and scale_model to
    "per-sub-block-aux1-scale-grid-of-8-codebook-signs64-sign-plane-int-domain". Route
    ids, helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, operand mirror strings, q8/iq2 names, and common EmitC/export code must
    not infer or choose the grid-of-8 super-block dot-product semantics.

    LOOP-CAPABLE per-super-block-source form: the OPTIONAL `block_index` operand toggles
    the per-super-block-source form. When present it is the induction variable (region
    argument 0) of an enclosing tcrv_rvv.typed_super_block_block_dot_loop_body region
    whose fold_model is "scalar_delta_grid": the iq2_xxs weight strip and the q8_K
    activation strip are read at `weight_base + block_index*66` and `activation_base +
    block_index*292`. When ABSENT the op keeps the standalone single-super-block form.

    The optional `integer_core_lmul` attribute is the SAME bounded resource shape knob
    the retired monolith carried (the *how*, never the *what*; the dot-product is
    byte-exact for any legal value because the per-element grid+sign decode is identical
    and the i32 integer accumulation is order-independent). The grid+sign vluxei16 gather
    + dot runs ONE 32-lane sub-block body: the 4 u64 grid entries gather as
    vint64<anchor> (4 entries = 32 i8 = one 32-lane sub-block), reinterpret to i8<anchor>,
    fold the per-lane +-1 sign (vmul_vv_i8<anchor>), one vwmul_vv_i16<2*anchor>, ONE
    vwredsum per sub-block. A single i64 entry is 8 bytes, so the 4-entry gather needs an
    i64 anchor whose VLMAX reaches 4 (=> the i8 view spans 32): the span-correct anchor
    MOVES with VLEN across {m1, m2} exactly like tq2_0 -- the schedule autotuner stamps
    "m2" at VLEN128 (e8m1 VLMAX 16 < 32) and the lighter "m1" at VLEN256 (e8m1 VLMAX
    reaches 32, i64m1 VLMAX reaches 4) from the SAME getRVVStripVLMAXElements truth source
    (it adopts TunableScheduleOpInterface; kernel key "iq2_xxs"). The verifier rejects any
    other anchor fail-closed (I7). The anchor defaults to "m2" (the VLEN-universal-safe
    floor), so an attr-less brick verifies + lowers correctly and the gearbox is free to
    REFINE m2->m1 at VLEN>=256 (ggml's shipped `_vl256` m1-32-lane shape). This is the
    Win-A capability-keyed selection preserved across the iq2_xxs flip.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
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
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$minimum_vlen
  );
  let results = (outs AnyType:$bsum);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($element_count) `,` type($vl) `->` type($bsum)";
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
    (the zve32x/zve64x re-strip is a capability item, not built here).

    The OPTIONAL `integer_core_lmul` anchors the base LMUL of the per-sub-block
    integer-MAC widening chain i8 -> i16 -> i32 (the *how*, never the *what*; the
    byte-exact deferred fp32 fold order, the 2-bit/subtractive-hmask unpack, and
    the signed 6-bit scale dance are all UNTOUCHED). Only TWO anchors are legal --
    the ceiling is "m1", NOT q4_K's "m2" -- because q3_K's sub_block == 16 elements
    (mirroring q6_K, distinct from q4_K's 32-element sub-block):
    - absent / "mf2": today's TWO 8-lane halves per 16-element sub-block (i8mf2 ->
      i16m1 -> i32m2 each, summed into the carried 8-lane aux32). Byte-identical to
      the pre-knob emit.
    - "m1": ONE 16-lane strip per sub-block (i8m1 -> i16m2 -> i32m4), the 16 i32
      lanes folded back element-wise to the canonical 8 BEFORE the fp32 cvt via a
      VLEN-AGNOSTIC `vslidedown` by LITERAL element offset 8 + `vadd` + `vget(.,0)`
      (NOT a vget-subgroup extract, whose lane count is VLEN-dependent). i8m1 == 16
      elements == exactly ONE sub-block under ONE scalar `scale = scales[js]-32`.
    An "m2" base (32 elements) would fold TWO 16-element sub-blocks under one scalar
    scale (identical ground to q4_K's m4 rejection) -- verifier-rejected fail-closed.
    The signed scale/hmask bit-dance is scalar/LMUL-free.
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

def GgmlBlockDotTQ20Q8KTernaryCoreOp
    : TCRVRVV_Op<"tq2_0_q8_k_ternary_core",
    [TCRVEmitCLowerableOpInterface,
     DeclareOpInterfaceMethods<TunableScheduleOpInterface>]> {
  let summary =
      "generic typed RVV ggml TQ2_0 x Q8_K super-block FUSED 2-bit TERNARY SCALAR "
      "integer-core op (sumi, the tq2_0 ternary brick, the FIRST TQ-family member)";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_tq2_0_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step -- the
    tq2_0 FUSED 2-bit TERNARY brick, the FIRST member of the TQ ({-1,0,+1}) TriLM
    family and the arity-cousin of the iq1_s/iq3_s scalar grid cores. UNLIKE the
    iq* siblings it decodes ARITHMETICALLY (NO grid/codebook gather): each weight
    is a 2-bit field whose value decodes to a ternary lane `((qs >> (l*2)) & 3) - 1`
    in {-1,0,1,2} (the encoder only emits {0,1,2} so the realized ternary is
    {-1,0,1}; the structural decode is exact over the full 2-bit range). The
    activation is q8_K (the SAME super-block activation as q2_K/q3_K/q4_K/q5_K/q6_K).
    It reproduces ggml's per-super-block SCALAR integer state EXACTLY as `_generic`
    computes `sumi` right BEFORE the fp32 `d` multiply, byte-exact (i32 equality):
    - `sumi` = Σ_{l=0..255} (((qs[l/4] >> (2*(l%4))) & 3) - 1) * q8[l], the flat
      256-element ternary*q8 contraction over the WHOLE super-block (single fp16
      scale, NO per-sub-block scale -- integer add is order-free).

    tq2_0 REUSES q2_K's 2-bit weight unpack VERBATIM (the same `(qs >> shift) & 3`
    over the 4 shifts {0,2,4,6}) but folds a per-element `-1` ternary bias into the
    unpack and is genuinely SIMPLER than every K-quant sibling -- there is NO
    scales[16] nibble extraction, NO per-sub-block scale, NO min term, NO dmin, NO
    bsums. THREE structural facts distinguish tq2_0, all load-bearing:

    (1) SUPER-BLOCK FORMAT, d AT THE END. block_tq2_0 is `{ uint8_t qs[QK_K/4];
        ggml_half d }` (QK_K = 256, sizeof 66): the 64 packed 2-bit-weight qs
        bytes LEAD the block at byte +0, and the fp16 super-block scale d follows
        at byte +64 (so `weight_qs_byte_offset = 0`, `weight_d_byte_offset = 64`,
        `weight_block_stride = 66`). This is the genuinely-new layout vs every
        sibling -- every other block has d near the front; here the weight is the
        prefix and the single scale is the suffix.

    (2) TERNARY `-1` BIAS, folded into the unpack. `_generic` (quants.c:501)
        computes `(((qs >> (l*2)) & 3) - 1)` PER ELEMENT, so the fused dot folds
        the `- 1` into the unpack: after q2_K's `&3` + u8->i8 reinterpret, ONE
        `vsub.vx 1` on the i8 vector, so each lane holds the signed ternary
        {-1,0,1,2} directly. The dot is then q2_K's VERBATIM signed widening
        reduce (vwmacc i8 x q8 i8 -> i16, vwredsum i32) with NO per-sub-block
        scale multiply -- a single per-super-block integer accumulator `sumi`.
        (The per-element subtract, NOT a `Σ(field*q8) - Σq8` refactor, mirrors
        `_generic` and keeps the decode exact.)

    (3) ONE INTEGER ACCUMULATOR + a DEFERRED SINGLE-SCALE SCALAR fp32 FOLD. ggml
        accumulates ONE `int32_t sumi` over the WHOLE super-block (the 256
        ternary*q8 products), then folds `sumf += (float)sumi * d` with `d = y.d *
        fp16(x.d)`. The DEFERRED fp32 SCALAR fold (`d = y.d @0 * fp16(x.d @64)`,
        `sumf += (float)sumi * d`) is DELIBERATELY OUT OF SCOPE here -- it is
        emitter-inlined by the scalar-delta-grid lowering. This op isolates the
        genuinely-new tq2_0 machinery (the fused 2-bit plane ternary dot) from the
        fp32 fold.

    The result is produced as ONE SCALAR i32 SSA result (ggml's per-super-block
    `int sumi`), NOT an 8-lane vector -- the arity-cousin of the iq1_s scalar grid
    core. The op is nested directly under tcrv_rvv.with_vl and consumes the tq2_0
    weight base pointer (vx), the q8_K activation base pointer (vy), the runtime
    element count (n), the active !tcrv_rvv.vl token, and an OPTIONAL `block_index`
    induction operand. It carries NO output pointer -- like the iq1_s scalar grid
    core, tq2_0's sumi is a scalar register.

    The structural facts -- QK_K=256, the AoS block strides (66 bytes for
    block_tq2_0, 292 bytes for block_q8_K), the tq2_0 weight byte offsets (qs @0,
    d @64) and the q8_K payload byte offsets (the fp32 activation scale d @0, qs
    @4) -- are carried as typed integer/string attributes (I4 mirror only). The
    bounded surface kind is limited to "ggml_tq2_0_q8_k_ternary_core" and
    scale_model to "ternary-2bit-fused-plane-single-fp16-scale-i32-domain". Route
    ids, helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, operand mirror strings, q8/tq2_0 names, and common EmitC/export code
    must not infer or choose super-block dot-product semantics.

    LOOP-CAPABLE per-super-block-source form: the OPTIONAL `block_index` operand
    toggles the per-super-block-source form. When present it is the induction
    variable (region argument 0) of an enclosing
    tcrv_rvv.typed_super_block_block_dot_loop_body region whose fold_model is
    "scalar_delta_grid": the tq2_0 weight strip and the q8_K activation strip are
    read at `weight_base + block_index*66` and `activation_base + block_index*292`.
    When ABSENT the op keeps the standalone single-super-block form.

    The optional `integer_core_lmul` attribute is the SAME bounded resource shape
    knob the retired monolith carried (the *how*, never the *what*; the dot-product
    is byte-exact for any legal value because the per-element ternary decode is
    identical and the i32 integer accumulation is order-independent). The fused
    ternary dot runs ONE 32-lane plane body per 2-bit shift (ggml's shipped
    `_vl128`/`_vl256` lane structure): the 32-byte qs chunk loads once, the 4 planes
    each vwmacc 32 ternary*q8 lanes into a wide i16 accumulator, ONE vwredsum per
    chunk. The single vsetvl_e8<anchor>(32) cover is correct ONLY at the whole-LMUL
    anchor whose i8 strip VLMAX spans the 32-element plane at the derived minimum
    VLEN: the span-correct anchor MOVES with VLEN across {m1, m2} exactly like q1_0
    -- the schedule autotuner stamps "m2" at VLEN128 (e8m1 VLMAX 16 < 32) and the
    lighter "m1" at VLEN256 (e8m1 VLMAX reaches 32) from the SAME
    getRVVStripVLMAXElements truth source (it adopts TunableScheduleOpInterface;
    kernel key "tq2_0"). The verifier rejects any other anchor fail-closed (I7). The
    anchor defaults to "m2" (the VLEN-universal-safe floor), so an attr-less brick
    verifies + lowers correctly and the gearbox is free to REFINE m2->m1 at
    VLEN>=256. This is the Win-A capability-keyed selection preserved across the
    tq2_0 flip.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$minimum_vlen
  );
  let results = (outs AnyType:$sumi);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($element_count) `,` type($vl) `->` type($sumi)";
  let hasVerifier = 1;
}

def GgmlBlockDotTQ10Q8KTernaryCoreOp
    : TCRVRVV_Op<"tq1_0_q8_k_ternary_core",
    [TCRVEmitCLowerableOpInterface,
     DeclareOpInterfaceMethods<TunableScheduleOpInterface>]> {
  let summary =
      "generic typed RVV ggml TQ1_0 x Q8_K super-block BASE-3 TERNARY SCALAR "
      "integer-core op (sumi, the tq1_0 ternary brick, the SECOND TQ-family member)";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_tq1_0_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step -- the
    tq1_0 BASE-3 TERNARY brick, the SECOND member of the TQ ({-1,0,+1}) TriLM
    family (the base-3-packed sibling of tq2_0) and the arity-cousin of the
    iq1_s/iq3_s/tq2_0 scalar cores. It REUSES the whole tq2_0 ternary scaffold at
    C2 marginal cost -- the SAME super-block SCALAR-accumulator loop op, the SAME
    single-scalar `sumf` yield contract, the SAME fold_model "scalar_delta_grid",
    the SAME emitter dispatch -- and differs ONLY in the unpack: instead of a
    2-bit field shift, each weight is a base-3 TRIT recovered by a power-of-three
    multiply (5 trits per qs byte since 3^5 = 243 < 256, plus a 4-trits-per-byte
    qh tail). Like tq2_0 it decodes ARITHMETICALLY (NO grid/codebook gather). The
    activation is q8_K (the SAME super-block activation as q2_K/q3_K/q4_K/q5_K/q6_K).
    tq1_0 uses ONLY y.d and y.qs -- NO bsums, NO min. It reproduces ggml's
    per-super-block SCALAR integer state EXACTLY as `_generic` computes `sum` right
    BEFORE the fp32 `d` multiply, byte-exact (i32 equality):
    - `sumi` = Σ_{k=0..255} ((base-3-trit(k) - 1) * q8[k]), the flat 256-element
      ternary*q8 contraction over the WHOLE super-block (single fp16 scale, NO
      per-sub-block scale -- integer add is order-free).
    THREE structural facts distinguish tq1_0, all load-bearing:

    (1) SUPER-BLOCK FORMAT, d AT THE END, with a SECOND weight array qh.
        block_tq1_0 is `{ uint8_t qs[(QK_K - 4*QK_K/64)/5]; uint8_t qh[QK_K/64];
        ggml_half d }` (QK_K = 256, qs = 48 bytes, qh = 4 bytes, sizeof 54): the
        48 packed base-3 qs bytes LEAD at byte +0, the 4 base-3 qh bytes follow
        at byte +48, and the fp16 super-block scale d is the SUFFIX at byte +52
        (so `weight_qs_byte_offset = 0`, `weight_qh_byte_offset = 48`,
        `weight_d_byte_offset = 52`, `weight_block_stride = 54`). The qh array is
        the new structural fact vs every sibling (including tq2_0) -- tq1_0 carries
        TWO base-3 weight arrays (the 5-trits-per-byte qs and the 4-trits-per-byte
        qh) whose decoded lanes both contribute to ONE super-block integer
        accumulator. This 54-byte stride is UNIQUE among the scalar_delta_grid
        bricks (tq2_0/iq2_xxs are 66), so it needs NO stride tie-breaker.

    (2) BASE-3 TRIT EXTRACT, with the mandatory uint8 wrap (the tq1_0-vs-tq2_0
        decode delta). `_generic` (quants.c:452-453, 461-462, 470-471) computes
        `uint8_t q = byte * pow3[l]` (an 8-BIT multiply -- the truncation to uint8
        IS the decode, and the high base-3 digit is then read by `xi =
        ((uint16_t)q * 3) >> 8`, value `xi - 1` in {-1,0,1}). pow3[6] =
        {1,3,9,27,81,243}. The emitter mirrors this with a vector pipeline that
        PRESERVES the wrap order: `vmul.vx u8 (byte * pow3[l])` (the low-8-bit
        truncation = the wrap) THEN `vwmulu.vx u8->u16 (q * 3)` THEN `vsrl 8` THEN
        narrow u16->u8 (xi in {0,1,2}, lossless) THEN reinterpret i8 THEN
        `vadd.vx -1` -> the ternary lane. The pow3 multiply must NOT be fused into
        the widening multiply (that would skip the mod-256 wrap and be wrong for
        any byte where byte*pow3 >= 256). pow3[l] is emitted as a per-l literal.
        This is base-3 whereas tq2_0 is a plain `(qs >> shift) & 3` 2-bit field.

    (3) THREE accumulation loops into ONE integer accumulator. `_generic` walks
        three index regions of the q8_K quant stream -- (a) the qs main loop (j=0,
        l 0..4, m 0..31 -> q8 index `l*32+m`, weight `qs[m]`), (b) the qs tail
        loop (j=32, l 0..4, m 0..15 -> q8 index `160 + l*16+m`, weight `qs[32+m]`),
        (c) the qh loop (l 0..3, j 0..3 -> q8 index `240 + l*4+j`, weight `qh[j]`)
        -- each contributing `(xi-1) * q8` to ONE `int sum`. Every q8 index 0..255
        is covered exactly once, so the emitter decodes all three regions into an
        element-ordered int8_t aux8[256] (in q8 order) and then runs the signed
        widening reduce (vle8 i8 x q8 i8 -> vwmul i16 -> vwredsum i32) into a
        single per-super-block integer accumulator (integer add is order-free).

    (4) The DEFERRED SINGLE-SCALE SCALAR fp32 FOLD (`d = fp16(x.d @52) * y.d @0`,
        `sumf += (float)sumi * d`, NO trailing factor) is DELIBERATELY OUT OF
        SCOPE here -- it is emitter-inlined by the scalar-delta-grid lowering. This
        op isolates the genuinely-new tq1_0 machinery (the base-3 trit unpack + the
        flat-256 integer dot) from the fp32 fold, exactly as the tq2_0 sibling
        isolates the fused 2-bit plane dot.

    The result is produced as ONE SCALAR i32 SSA result (ggml's per-super-block
    `int sum`), NOT an 8-lane vector -- the arity-cousin of the tq2_0/iq1_s scalar
    cores. The op is nested directly under tcrv_rvv.with_vl and consumes the tq1_0
    weight base pointer (vx), the q8_K activation base pointer (vy), the runtime
    element count (n), the active !tcrv_rvv.vl token, and an OPTIONAL `block_index`
    induction operand. It carries NO output pointer -- like the tq2_0 scalar core,
    tq1_0's sumi is a scalar register.

    The structural facts -- QK_K=256, the AoS block strides (54 bytes for
    block_tq1_0, 292 bytes for block_q8_K), the tq1_0 weight byte offsets (qs @0,
    qh @48, d @52) and the q8_K payload byte offsets (the fp32 activation scale d
    @0, qs @4) -- are carried as typed integer/string attributes (I4 mirror
    only). The bounded surface kind is limited to "ggml_tq1_0_q8_k_ternary_core"
    and scale_model to "ternary-base3-single-fp16-scale-i32-domain". Route ids,
    helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, operand mirror strings, q8/tq1_0 names, and common EmitC/export
    code must not infer or choose super-block dot-product semantics.

    LOOP-CAPABLE per-super-block-source form: the OPTIONAL `block_index` operand
    toggles the per-super-block-source form. When present it is the induction
    variable (region argument 0) of an enclosing
    tcrv_rvv.typed_super_block_block_dot_loop_body region whose fold_model is
    "scalar_delta_grid": the tq1_0 weight strip and the q8_K activation strip are
    read at `weight_base + block_index*54` and `activation_base + block_index*292`.
    When ABSENT the op keeps the standalone single-super-block form.

    The optional `integer_core_lmul` attribute is the SAME bounded resource shape
    knob the q1_0 / tq2_0 siblings carry, and here it tunes ONLY the integer DOT
    (section B): the base-3 trit unpack (section A) lands an element-ordered
    aux8[256] and aux8[k] pairs with q8[k] across the WHOLE super-block (single
    fp16 scale, NO per-sub-block scale), so the dot is a flat 256-element i8*i8
    contraction. The narrow 16x e8m1(16) reductions widen to 32-lane strips at the
    whole-LMUL anchor whose i8 VLMAX spans 32 -- m2 at VLEN128 (e8m1 VLMAX 16 <
    32), the lighter m1 at VLEN256 -- selected by the schedule autotuner from the
    SAME getRVVStripVLMAXElements truth source (kernel key "tq1_0"). The
    dot-product is byte-exact for any legal value (integer addition is
    order-independent). The anchor defaults to "m2" (the VLEN-universal-safe
    floor), so an attr-less brick verifies + lowers correctly and the gearbox is
    free to REFINE m2->m1 at VLEN>=256 -- the Win-A capability-keyed selection
    preserved across the tq1_0 flip. NOTE: only the DOT widens; section A's base-3
    unpack + the aux8 scratch round-trip are KEPT (full ggml-style fusion is a
    separate larger emit -- ggml's tq1_0 `_vl128`/`_vl256` are DIFFERENT shapes,
    not a clean LMUL flip), so this targets tq1_0's narrow-strip LOSS without a
    parity claim. The verifier rejects any other anchor fail-closed (I7).
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_qs_byte_offset,
    I64Attr:$weight_qh_byte_offset,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$activation_d_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$minimum_vlen
  );
  let results = (outs AnyType:$sumi);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($element_count) `,` type($vl) `->` type($sumi)";
  let hasVerifier = 1;
}

def GgmlBlockDotIQ4XSQ8KCodebookCoreOp
    : TCRVRVV_Op<"iq4_xs_q8_k_codebook_core",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ4_XS x Q8_K super-block CODEBOOK SCALAR "
      "integer-core op (the iq4_xs codebook brick, flat iq4_nl codebook sibling)";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_iq4_xs_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step -- the
    iq4_xs CODEBOOK brick, the SUPER-BLOCK rung of the flat iq4_nl codebook
    (tcrv_rvv.iq4_nl_q8_0_block_dot). It decodes with the SAME 16-entry non-linear int8
    CODEBOOK GATHER (decode_model=lookup) iq4_nl uses -- the `kvalues_iq4nl[16]` table
    broadcast into an i8m1 vreg ONCE, each 4-bit nibble split into the two UNSIGNED index
    lanes (vand 0x0F / vsrl 0x04) and mapped through the table by `vrgather_vv_i8m1` (NOT
    an arithmetic nibble-8 decode) -- but wraps it in the q4_K-style super-block SIGNED
    6-bit scale machinery and the q8_K activation. It reproduces ggml's per-super-block
    fp32 contribution EXACTLY as `_generic` computes it, byte-exact:
    - for each of the 8 sub-blocks j=0..7, the SIGNED 6-bit scale `ls_j` is extracted by
      the q4_K-style cross-byte bit dance in its CLOSED form `ls_j = ((scales_l[j/2] >>
      (4*(j%2))) & 0xf) | (((scales_h >> (2*j)) & 0x3) << 4)` (scalar STRUCTURED emitc
      bitwise ops), then the per-sub-block float scale `d1 = d4d8 * (float)(ls_j - 32)` is
      a SEPARATE op (the `(ls-32)` round + the d4d8 multiply stay separate from the sumi
      multiply so NO FMA fuses them), where `d4d8 = fp16(x.d @0) * y.d @0` is computed
      ONCE per super-block; the per-sub-block integer dot `sumi_j` REUSES iq4_nl's
      codebook-gather core VERBATIM (one `vsetvl_e8m1(16)` over the 16 nibble bytes at
      `qs + j*16`, the two vrgather-decoded index lanes fed to the asymmetric `vwmul` low
      <-> q8[j*32+0..15] + `vwmacc` high <-> q8[j*32+16..31] -> `vwredsum` -> i32); and
      `sumf += d1 * (float)sumi_j` is ONE emitc.expression, invoked in STRICT ascending
      (super-block, sub-block) order. There is NO min term (symmetric, like q6_K) and NO
      trailing factor. iq4_xs's fold runs PER-SUB-BLOCK in float (8 separate fp folds per
      super-block), so UNLIKE the grid siblings there is no single integer `bsum` scalar
      state; the fold is DELIBERATELY OUT OF SCOPE of this op's result -- it is
      emitter-inlined by the scalar-delta-grid lowering keyed off this brick's identity.

    The op is nested directly under tcrv_rvv.with_vl and consumes the iq4_xs weight base
    pointer (vx), the q8_K activation base pointer (vy), the runtime element count (n),
    the active !tcrv_rvv.vl token, and an OPTIONAL `block_index` induction operand. It
    produces ONE scalar i32 SSA result (a structural placeholder mirroring the sibling
    grid-core bricks' scalar-state arity; iq4_xs's actual per-super-block contribution is
    a running fp32 fold with no single scalar state, so this result is UNUSED -- the whole
    per-super-block body is emitter-inlined). It carries NO output pointer.

    The block-quantized super-block STRUCTURE is first-class in the typed body. The
    structural facts -- QK_K=256, 8 sub-blocks of 32 elements, the AoS block strides (136
    bytes for block_iq4_xs = {fp16 d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]},
    292 bytes for block_q8_K), the iq4_xs payload byte offsets (the fp16 super-block scale
    d @0, scales_h @2, scales_l[4] @4, qs[128] @8) and the q8_K payload byte offsets (the
    fp32 activation scale d @0, qs @4) -- are carried as typed integer/string attributes
    (I4 mirror only). The 16-entry non-linear int8 CODEBOOK is a `DenseI8ArrayAttr:$codebook`
    of EXACTLY 16 int8 entries (the SAME table as iq4_nl); it is a STRUCTURAL FACT consumed
    structurally into the realized body (emitted as a `static const int8_t[16]` decl + the
    `vle8` table load), NOT an arithmetic decode, NOT a string plan read. The bounded
    surface kind is limited to "ggml_iq4_xs_q8_k_codebook_core" and scale_model to
    "per-sub-block-signed-6bit-scale-codebook-gather-float-domain". Route ids, helper names,
    ABI strings, artifact names, status fields, exact intrinsic spellings, operand mirror
    strings, q8/iq4 names, and common EmitC/export code must not infer or choose the
    super-block codebook dot-product semantics.

    The CODEBOOK GATHER is inherently a Zvl128b legality fact (the broadcast `values`
    register's VLMAX must be >= 16 to index all 16 table entries), so the integer core pins
    the m1 anchor; it carries NO integer_core_lmul shape knob (iq4_xs is NOT in any schedule
    autotuner). A board with VLEN < 128 cannot legally host the codebook gather; that is an
    N1 capability-legality divergence.

    LOOP-CAPABLE per-super-block-source form: the OPTIONAL `block_index` operand toggles
    the per-super-block-source form. When present it is the induction variable (region
    argument 0) of an enclosing tcrv_rvv.typed_super_block_block_dot_loop_body region
    whose fold_model is "scalar_delta_grid": the iq4_xs weight strip and the q8_K
    activation strip are read at `weight_base + block_index*136` and `activation_base +
    block_index*292`. When ABSENT the op keeps the standalone single-super-block form.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
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
  let results = (outs AnyType:$partial);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($element_count) `,` type($vl) `->` type($partial)";
  let hasVerifier = 1;
}

// NOTE: def GgmlBlockDotIQ4XSQ8KOp (the monolith iq4_xs block-dot op) was RETIRED at the
// iq4_xs flip (C_construct 23->24): the front door now constructs the typed super-block
// SCALAR-accumulator loop body (tcrv_rvv.typed_super_block_block_dot_loop_body with
// fold_model "scalar_delta_grid", stride 136) carrying the iq4_xs CODEBOOK-core brick
// (GgmlBlockDotIQ4XSQ8KCodebookCoreOp) instead. The retired monolith emitter's
// per-super-block body was code-moved into the byte-exact
// emitTypedSuperBlockScalarDeltaGridLoopBodyIq4xs lowering (the sole live caller).

// NOTE: def GgmlBlockDotIQ2XXSQ8KOp (the monolith iq2_xxs block-dot op) was RETIRED at
// the iq2_xxs flip (L3 coverage): the front door now constructs the typed super-block
// SCALAR-accumulator GRID loop body (tcrv_rvv.typed_super_block_block_dot_loop_body with
// fold_model "scalar_delta_grid", stride 66) carrying the iq2_xxs GRID-of-8 grid/signs64
// grid-core brick (GgmlBlockDotIQ2XXSQ8KGridCoreOp) instead. The retired monolith
// emitter's per-super-block body was extracted into the byte-exact shared anchor
// emitIQ2XXSSuperBlockGridBody (the sole live caller is the constructed loop lowering).
// The grid-core brick carries the SAME Win-A integer_core_lmul gearbox (kernel key
// "iq2_xxs") so the m2->m1 VLEN capability selection is preserved on the constructed op.

// NOTE: def GgmlBlockDotTQ20Q8KOp (the monolith tq2_0 block-dot op) was RETIRED at the
// tq2_0 flip (C_construct 24->25, the FIRST TQ-family member): the front door now
// constructs the typed super-block SCALAR-accumulator loop body
// (tcrv_rvv.typed_super_block_block_dot_loop_body with fold_model "scalar_delta_grid",
// stride 66) carrying the tq2_0 FUSED 2-bit TERNARY integer-core brick
// (GgmlBlockDotTQ20Q8KTernaryCoreOp, def above) instead. The retired monolith emitter's
// per-super-block body was code-moved into the byte-exact
// emitTypedSuperBlockScalarDeltaGridLoopBodyTQ20 lowering (the sole live caller). Like
// iq2_xxs the ternary-core brick carries the SAME Win-A integer_core_lmul m2/m1 gearbox
// (kernel key "tq2_0") so the VLEN capability selection is preserved on the constructed
// op. tq2_0 shares weight_block_stride 66 with iq2_xxs but dispatches by its OWN DISTINCT
// brick op identity (arithmetic ternary decode, NO grid/signs gather), so there is no
// emitter ambiguity.

// NOTE: def GgmlBlockDotTQ10Q8KOp (the monolith tq1_0 block-dot op) was RETIRED at the
// tq1_0 flip (C_construct 25->26, the SECOND TQ-family member): the front door now
// constructs the typed super-block SCALAR-accumulator loop body
// (tcrv_rvv.typed_super_block_block_dot_loop_body with fold_model "scalar_delta_grid",
// stride 54) carrying the tq1_0 BASE-3 TERNARY integer-core brick
// (GgmlBlockDotTQ10Q8KTernaryCoreOp, def above) instead. The retired monolith emitter's
// per-super-block body was code-moved into the byte-exact
// emitTypedSuperBlockScalarDeltaGridLoopBodyTQ10 lowering (the sole live caller). It REUSES
// the whole tq2_0 ternary scaffold at C2 marginal cost and differs ONLY in the base-3 trit
// unpack (section A). Like tq2_0 the ternary-core brick carries the SAME Win-A
// integer_core_lmul m2/m1 gearbox (kernel key "tq1_0") so the VLEN capability selection is
// preserved on the constructed op. tq1_0's weight_block_stride 54 is UNIQUE among the
// scalar_delta_grid bricks, so it dispatches by stride with NO tie-breaker (the emitter
// still disambiguates by the DISTINCT base-3 ternary-core brick op type).

def GgmlBlockDotIQ2XSQ8KGridCoreOp
    : TCRVRVV_Op<"iq2_xs_q8_k_grid_core",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ2_XS x Q8_K super-block per-half-explicit-scale "
      "GRID SCALAR integer-core op (bsum, the iq2_xs grid/signs64 brick, iq2_xxs sibling)";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_iq2_xs_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step -- the
    iq2_xs PER-HALF-EXPLICIT-SCALE GRID brick, the SIGN-PLANE (signs64) SIBLING of the
    iq2_xxs GRID-of-8 core (tcrv_rvv.iq2_xxs_q8_k_grid_core). It decodes with a codebook
    GRID GATHER (decode_model=lookup) over the 512-entry `iq2xs_grid[512]` uint64 grid
    via a `vluxei16_v_i64m1` indexed gather, with per-element signs read via a SECOND
    `vluxei16_v_i64m1` indexed gather over the DERIVED `keven_signs_q2xs` SIGN-PLANE
    (signs64 -- 1024 +-1 bytes = the 128-entry ksigns_iq2xs selector EXPANDED to per-lane
    +-1). It reproduces ggml's per-super-block SCALAR integer state EXACTLY as `_generic`
    computes `bsum` right BEFORE the fp32 `d` multiply, byte-exact (i32 equality):
    - `bsum` = Σ_{ib=0..7} [ ls1[ib] * (Σ_{lane in groups 0,1} signed_grid[lane]*q8[lane])
      + ls2[ib] * (Σ_{lane in groups 2,3} signed_grid[lane]*q8[lane]) ], the per-sub-block
      per-HALF-scaled grid/sign dot over one 32-lane sub-block. iq2_xs has THREE structural
      deltas vs iq2_xxs: (a) the GRID has 512 entries (9-bit index) not 256; (b) the index
      is NOT a packed aux byte -- each `qs[l]` is a uint16 word `w` whose LOW 9 bits =
      `idx = w & 511` and HIGH 7 bits = the sign selector `sel = w >> 9` (NO aux1 word, NO
      `2*(aux1>>28)+1` packed scale, NO `(aux1>>7*l)&127` packed selector); (c) the
      per-sub-block 4-bit scales are an EXPLICIT array `scales[QK_K/32]` after qs -- each
      `sc = scales[ib32]` carries TWO 4-bit scales, `ls1 = 2*(sc & 0xf)+1` for groups l=0,1
      and `ls2 = 2*(sc>>4)+1` for groups l=2,3 -- so the sub-block splits into TWO 16-lane
      HALVES (each 2 groups) with distinct scales, TWO `vwredsum` reductions and TWO
      `bsum += sumi*ls` contributions. Each half's 2 grid indices (idx*8 byte-offsets) and
      2 sign selectors (sel*8 byte-offsets) are packed as u16 index arrays; TWO
      `vluxei16_v_i64m1` gathers over the i64 grid64 / signs64 bases fetch the 16 grid
      bytes + 16 +-1 sign bytes; the sign folds onto the GRID via `vmul_vv_i8m1(grid,
      signs)` (NOT q8 -- the i8 product wraps at the q8=-128 lane, so folding onto the
      [8,43]-bounded grid is byte-exact); then ONE `vwmul_vv_i16m2` + ONE
      `vwredsum_vs_i16m2_i32m1` per HALF, then `bsum += sumi*ls`.
    The result is produced as ONE SCALAR i32 SSA result (ggml's per-super-block `int
    bsum`), NOT a vector -- the arity-cousin of the iq2_xxs scalar grid core. The
    DEFERRED fp32 SCALAR fold (`d = fp16(x.d @0) * y.d @0`, `sumf += d * (float)bsum`,
    then the trailing `*s = 0.125f * sumf` factor) is DELIBERATELY OUT OF SCOPE here --
    it is emitter-inlined by the scalar-delta-grid lowering. This op isolates the
    genuinely-new iq2_xs machinery (the i64 GRID vluxei16 gather, the signs64 sign-plane
    vluxei16 gather, the per-half explicit-scale + qs-word-split-index/sign decode) from
    the fp32 fold.

    The op is nested directly under tcrv_rvv.with_vl and consumes the iq2_xs weight
    base pointer (vx), the q8_K activation base pointer (vy), the runtime element count
    (n), the active !tcrv_rvv.vl token, and an OPTIONAL `block_index` induction operand.
    It produces ONE scalar i32 SSA result (bsum). It carries NO output pointer -- like
    the iq2_xxs scalar grid core, iq2_xs's bsum is a scalar register.

    The block-quantized super-block STRUCTURE is first-class in the typed body. The
    structural facts -- QK_K=256, 8 sub-blocks of 32 elements, the AoS block strides
    (74 bytes for block_iq2_xs = {fp16 d; uint16 qs[32]; uint8 scales[8]}, 292 bytes for
    block_q8_K), the iq2_xs payload byte offsets (the fp16 d @0, the 64 uint8 qs bytes
    @2, the explicit 4-bit scales[8] @66) and the q8_K payload byte offsets (fp32 d @0,
    int8 qs @4) -- are carried as typed integer/string attributes (I4 mirror only). The
    FIXED 512-entry iq2xs_grid codebook and the DERIVED keven_signs_q2xs signs64 sign
    plane are byte-exact constants of the iq2_xs FORMAT; they are emitted keyed by the
    brick op identity, NOT carried in the IR here (the signs64 sign-plane mechanism is
    DERIVED at emit from the fixed ksigns selector, NO op-attr extension). The bounded
    surface kind is limited to "ggml_iq2_xs_q8_k_grid_core" and scale_model to
    "per-half-int4-explicit-scales-grid-codebook-signs64-sign-plane-int-domain". Route
    ids, helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, operand mirror strings, q8/iq2 names, and common EmitC/export code must
    not infer or choose the grid super-block dot-product semantics.

    LOOP-CAPABLE per-super-block-source form: the OPTIONAL `block_index` operand toggles
    the per-super-block-source form. When present it is the induction variable (region
    argument 0) of an enclosing tcrv_rvv.typed_super_block_block_dot_loop_body region
    whose fold_model is "scalar_delta_grid": the iq2_xs weight strip and the q8_K
    activation strip are read at `weight_base + block_index*74` and `activation_base +
    block_index*292`. When ABSENT the op keeps the standalone single-super-block form.

    UNLIKE the iq2_xxs sibling, iq2_xs carries NO integer_core_lmul gearbox: the grid
    lookup is an INDEXED gather over a pointer (the 4096-byte grid cannot broadcast into
    a vreg) and the per-half body runs at a FIXED 16-lane shape (i64m1 gather + i8m1
    view + i16m2 widen), so there is NO VLEN128-vs-VLEN256 byte-flip and the op is NOT
    in any schedule autotuner (no TunableScheduleOpInterface). It lowers at the iq2_xs
    emitter's fixed shape.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
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
    I64Attr:$activation_quant_byte_offset
  );
  let results = (outs AnyType:$bsum);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($element_count) `,` type($vl) `->` type($bsum)";
  let hasVerifier = 1;
}

// NOTE: def GgmlBlockDotIQ2XSQ8KOp (the monolith iq2_xs block-dot op) was RETIRED at the
// iq2_xs flip (L3 coverage): the front door now constructs the typed super-block
// SCALAR-accumulator GRID loop body (tcrv_rvv.typed_super_block_block_dot_loop_body with
// fold_model "scalar_delta_grid", stride 74) carrying the iq2_xs per-half-explicit-scale
// GRID grid-core brick (GgmlBlockDotIQ2XSQ8KGridCoreOp) instead. The retired monolith
// emitter's per-super-block body was extracted into the byte-exact shared anchor
// emitIQ2XSSuperBlockGridBody (the sole live caller is the constructed loop lowering).
// UNLIKE iq2_xxs, the iq2_xs brick carries NO integer_core_lmul gearbox (fixed 16-lane
// per-half shape, not in any autotuner).

def GgmlBlockDotIQ2SQ8KGridCoreOp
    : TCRVRVV_Op<"iq2_s_q8_k_grid_core",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ2_S x Q8_K super-block per-half-explicit-scale "
      "GRID SCALAR integer-core op (bsum, the iq2_s grid/qh-plane/explicit-signs brick)";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_iq2_s_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step -- the
    iq2_s PER-HALF-EXPLICIT-SCALE GRID brick, the SIGN-PLANE SIBLING of the iq2_xs
    grid core (tcrv_rvv.iq2_xs_q8_k_grid_core), the THIRD member of the deep IQ tail.
    It decodes with a codebook GRID GATHER (decode_model=lookup) over the 1024-entry
    `iq2s_grid[1024]` uint64 grid via a `vluxei16_v_i64m1` indexed gather, with
    per-element signs read via a SECOND `vluxei16_v_i64m1` indexed gather over the
    UNIVERSAL `signs256` SIGN-PLANE (2048 +-1 bytes = every 8-bit sign byte value
    EXPANDED to per-lane +-1). It reproduces ggml's per-super-block SCALAR integer
    state EXACTLY as `_generic` computes `bsum` right BEFORE the fp32 `d` multiply,
    byte-exact (i32 equality):
    - `bsum` = Σ_{ib=0..7} [ ls1[ib] * (Σ_{lane in groups 0,1} signed_grid[lane]*q8[lane])
      + ls2[ib] * (Σ_{lane in groups 2,3} signed_grid[lane]*q8[lane]) ], the per-sub-block
      per-HALF-scaled grid/sign dot over one 32-lane sub-block. iq2_s has THREE structural
      deltas vs iq2_xs: (a) the GRID has 1024 entries (10-bit index) not 512; (b) the index
      is assembled from TWO sources -- the low 8 bits are a single byte `qs[l]` from the
      index region, the high 2 bits are injected from a per-sub-block qh byte, `idx =
      qs[l] | ((qh[ib32] << (8-2*l)) & 0x300)` (the q5_K-style qh-plane pattern; NO uint16
      qs word, NO 9-bit `w & 511` read, NO `w >> 9` sign selector); (c) the signs are
      EXPLICIT -- read DIRECTLY from a sign-byte region that lives INSIDE qs[64] at byte
      offset QK_K/8 (`signs = qs + 32`), NOT through a ksigns_iq2xs selector table, so the
      signs256 sign-plane is indexed by the raw 8-bit sign byte directly (NO ksigns
      indirection). The per-sub-block 4-bit scales are an EXPLICIT array `scales[QK_K/32]`
      after qh -- each `sc = scales[ib32]` carries TWO 4-bit scales, `ls1 = 2*(sc & 0xf)+1`
      for groups l=0,1 and `ls2 = 2*(sc>>4)+1` for groups l=2,3 -- so the sub-block splits
      into TWO 16-lane HALVES (each 2 groups) with distinct scales, TWO `vwredsum`
      reductions and TWO `bsum += sumi*ls` contributions. Each half's 2 grid indices (idx*8
      byte-offsets) and 2 explicit sign bytes (signByte*8 byte-offsets) are packed as u16
      index arrays; TWO `vluxei16_v_i64m1` gathers over the i64 grid64 / signs256 bases
      fetch the 16 grid bytes + 16 +-1 sign bytes; the sign folds onto the GRID via
      `vmul_vv_i8m1(grid, signs)` (NOT q8 -- the i8 product wraps at the q8=-128 lane, so
      folding onto the [8,43]-bounded grid is byte-exact); then ONE `vwmul_vv_i16m2` + ONE
      `vwredsum_vs_i16m2_i32m1` per HALF, then `bsum += sumi*ls`.
    The result is produced as ONE SCALAR i32 SSA result (ggml's per-super-block `int
    bsum`), NOT a vector -- the arity-cousin of the iq2_xs scalar grid core. The
    DEFERRED fp32 SCALAR fold (`d = fp16(x.d @0) * y.d @0`, `sumf += d * (float)bsum`,
    then the trailing `*s = 0.125f * sumf` factor) is DELIBERATELY OUT OF SCOPE here --
    it is emitter-inlined by the scalar-delta-grid lowering. This op isolates the
    genuinely-new iq2_s machinery (the i64 GRID vluxei16 gather, the signs256 explicit
    sign-plane vluxei16 gather, the per-half explicit-scale + qs-byte+qh-bit-split index
    decode) from the fp32 fold.

    The op is nested directly under tcrv_rvv.with_vl and consumes the iq2_s weight
    base pointer (vx), the q8_K activation base pointer (vy), the runtime element count
    (n), the active !tcrv_rvv.vl token, and an OPTIONAL `block_index` induction operand.
    It produces ONE scalar i32 SSA result (bsum). It carries NO output pointer -- like
    the iq2_xs scalar grid core, iq2_s's bsum is a scalar register.

    The block-quantized super-block STRUCTURE is first-class in the typed body. The
    structural facts -- QK_K=256, 8 sub-blocks of 32 elements, the AoS block strides
    (82 bytes for block_iq2_s = {fp16 d; uint8 qs[64]; uint8 qh[8]; uint8 scales[8]},
    292 bytes for block_q8_K), the iq2_s payload byte offsets (the fp16 d @0, the 32
    index bytes @2, the 32 sign bytes @34, the qh[8] @66, the explicit 4-bit scales[8]
    @74) and the q8_K payload byte offsets (fp32 d @0, int8 qs @4) -- are carried as
    typed integer/string attributes (I4 mirror only). The FIXED 1024-entry iq2s_grid
    codebook and the UNIVERSAL signs256 sign plane are byte-exact constants of the
    iq2_s FORMAT; they are emitted keyed by the brick op identity, NOT carried in the
    IR here (iq2_s has NO ksigns selector -- the signs are explicit bytes and the
    signs256 plane is the definitional 8-bit-to-per-lane +-1 expansion). The bounded
    surface kind is limited to "ggml_iq2_s_q8_k_grid_core" and scale_model to
    "per-half-int4-explicit-scales-grid-codebook-qh-plane-explicit-signs-int-domain".
    Route ids, helper names, ABI strings, artifact names, status fields, exact
    intrinsic spellings, operand mirror strings, q8/iq2 names, and common EmitC/export
    code must not infer or choose the grid super-block dot-product semantics.

    LOOP-CAPABLE per-super-block-source form: the OPTIONAL `block_index` operand toggles
    the per-super-block-source form. When present it is the induction variable (region
    argument 0) of an enclosing tcrv_rvv.typed_super_block_block_dot_loop_body region
    whose fold_model is "scalar_delta_grid": the iq2_s weight strip and the q8_K
    activation strip are read at `weight_base + block_index*82` and `activation_base +
    block_index*292`. When ABSENT the op keeps the standalone single-super-block form.

    Like the iq2_xs sibling, iq2_s carries NO integer_core_lmul gearbox: the grid
    lookup is an INDEXED gather over a pointer (the 8192-byte grid cannot broadcast into
    a vreg) and the per-half body runs at a FIXED 16-lane shape (i64m1 gather + i8m1
    view + i16m2 widen), so there is NO VLEN128-vs-VLEN256 byte-flip and the op is NOT
    in any schedule autotuner (no TunableScheduleOpInterface). It lowers at the iq2_s
    emitter's fixed shape.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
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
    I64Attr:$activation_quant_byte_offset
  );
  let results = (outs AnyType:$bsum);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($element_count) `,` type($vl) `->` type($bsum)";
  let hasVerifier = 1;
}

// NOTE: def GgmlBlockDotIQ2SQ8KOp (the monolith iq2_s block-dot op) was RETIRED at the
// iq2_s flip (L3 coverage): the front door now constructs the typed super-block
// SCALAR-accumulator GRID loop body (tcrv_rvv.typed_super_block_block_dot_loop_body with
// fold_model "scalar_delta_grid", stride 82) carrying the iq2_s per-half-explicit-scale
// GRID grid-core brick (GgmlBlockDotIQ2SQ8KGridCoreOp) instead. The retired monolith
// emitter's per-super-block body was extracted into the byte-exact shared anchor
// emitIQ2SSuperBlockGridBody (the sole live caller is the constructed loop lowering).
// Like iq2_xs, the iq2_s brick carries NO integer_core_lmul gearbox (fixed 16-lane
// per-half shape, not in any autotuner).

// NOTE: def GgmlBlockDotIQ3XXSQ8KOp (the monolith iq3_xxs block-dot op) was RETIRED at
// the iq3_xxs flip (L3 coverage): the front door now constructs the typed super-block
// SCALAR-accumulator GRID loop body (tcrv_rvv.typed_super_block_block_dot_loop_body with
// fold_model "scalar_delta_grid", stride 98) carrying the iq3_xxs GRID-of-4 grid-core
// brick (GgmlBlockDotIQ3XXSQ8KGridCoreOp) instead. The retired monolith emitter's
// per-super-block body was extracted into the byte-exact shared anchor
// emitIQ3XXSSuperBlockGridBody (the sole live caller is the constructed loop lowering).

def GgmlBlockDotIQ3SQ8KGridCoreOp
    : TCRVRVV_Op<"iq3_s_q8_k_grid_core",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml IQ3_S x Q8_K super-block GRID-of-4 SCALAR "
      "integer-core op (bsum, the iq3_s explicit-signs grid brick, iq3_xxs sibling)";
  let description = [{
    Records the INTEGER CORE of the ggml `ggml_vec_dot_iq3_s_q8_K` super-block
    dot-product as ONE bounded generic plugin-local typed RVV dataflow step -- the
    iq3_s GRID-of-4 brick, the i32-grid EXPLICIT-SIGNS sibling of the iq3_xxs GRID-of-4
    core (tcrv_rvv.iq3_xxs_q8_k_grid_core). It decodes with a codebook GRID GATHER
    (decode_model=lookup) like iq3_xxs, over the 512-entry `iq3s_grid[512]` uint32
    GRID-of-4 (each entry FOUR int8 values) via a `vluxei16_v_i32m1` indexed gather, but
    with THREE mechanisms swapped from iq3_xxs to iq2_s: the grid INDEX carries a qh
    9th-bit inject (a SINGLE bit, mask 256 = bit 8, the two passes taking shifts 8-2l and
    7-2l), the per-lane SIGN is read from an EXPLICIT per-sub-block signs region (at
    weight offset 74, NO ksigns plane), and the per-sub-block scale is the EXPLICIT
    two-nibble scales[] (at weight offset 106). It reproduces ggml's per-super-block
    SCALAR integer state EXACTLY as `_generic` computes `bsum` right BEFORE the fp32 `d`
    multiply, byte-exact (i32 equality):
    - `bsum` = Σ_{ib=0..7} ls[ib] * (Σ_{l=0..3} Σ_{lane} q8[lane] * signed_grid[lane]),
      the per-sub-block 4-bit-scaled grid/sign dot. Each sub-block reads its scale byte
      `sc = scales[ib/2]`; the per-sub-block scale is `ls = 2*(sc & 0xf)+1` (even ib) or
      `2*(sc >> 4)+1` (odd ib); the qh byte `qh[ib]`; each of the 4 sign groups l=0..3
      reads the EXPLICIT sign byte `signs = sgn[ib*4 + l]` and the two grid INDICES
      `idx1 = qs[8*ib+2l+0] | ((qh[ib] << (8-2l)) & 256)`, `idx2 = qs[8*ib+2l+1] |
      ((qh[ib] << (7-2l)) & 256)` (an 8-lane vluxei16 gather over the i32 grid; lanes
      0..3 = grid1/kmask{1,2,4,8}/q8[0..3], lanes 4..7 = grid2/kmask{16,32,64,128}/
      q8[4..7]), applies the per-lane sign via vmerge(grid, vneg(grid), mask) and
      accumulates the signed widening product `vwmul` + `vwredsum`.
    The result is produced as ONE SCALAR i32 SSA result (ggml's per-super-block `int
    bsum`), NOT an 8-lane vector -- the arity-cousin of the iq3_xxs scalar grid core. The
    DEFERRED fp32 SCALAR fold (`d = fp16(x.d @0) * y.d @0`, `sumf += d * (float)bsum`,
    then the trailing `*s = sumf` -- iq3_s applies NO trailing factor) is DELIBERATELY
    OUT OF SCOPE here -- it is emitter-inlined by the scalar-delta-grid lowering. This
    op isolates the genuinely-new iq3_s machinery (the qh 9th-bit inject, the explicit
    per-sub-block signs region, the explicit two-nibble scales) from the fp32 fold.

    The op is nested directly under tcrv_rvv.with_vl and consumes the iq3_s weight base
    pointer (vx), the q8_K activation base pointer (vy), the runtime element count (n),
    the active !tcrv_rvv.vl token, and an OPTIONAL `block_index` induction operand. It
    produces ONE scalar i32 SSA result (bsum). It carries NO output pointer -- like the
    iq3_xxs scalar grid core, iq3_s's bsum is a scalar register.

    The block-quantized super-block STRUCTURE is first-class in the typed body. The
    structural facts -- QK_K=256, 8 sub-blocks of 32 elements, the AoS block strides (110
    bytes for block_iq3_s = {fp16 d; uint8 qs[64]; uint8 qh[8]; uint8 signs[32]; uint8
    scales[4]}, 292 bytes for block_q8_K), the iq3_s payload byte offsets (the fp16 d @0,
    the 64 uint8 grid-index bytes qs @2, the 8 qh bytes @66, the 32 signs bytes @74, the
    4 scales bytes @106) and the q8_K payload byte offsets (fp32 d @0, int8 qs @4) -- are
    carried as typed integer/string attributes (I4 mirror only). The FIXED 512-entry
    iq3s_grid GRID-of-4 codebook is a byte-exact constant of the iq3_s FORMAT; it is
    emitted keyed by the brick op identity, NOT carried in the IR here (unlike iq3_xxs
    there is NO ksigns plane -- the signs are an explicit memory region). The bounded
    surface kind is limited to "ggml_iq3_s_q8_k_grid_core" and scale_model to
    "per-sub-block-explicit-scale-grid-of-4-codebook-qh-plane-explicit-signs-int-domain".
    Route ids, helper names, ABI strings, artifact names, status fields, exact intrinsic
    spellings, operand mirror strings, q8/iq3 names, and common EmitC/export code must
    not infer or choose the grid-of-4 super-block dot-product semantics.

    LOOP-CAPABLE per-super-block-source form: the OPTIONAL `block_index` operand toggles
    the per-super-block-source form. When present it is the induction variable (region
    argument 0) of an enclosing tcrv_rvv.typed_super_block_block_dot_loop_body region
    whose fold_model is "scalar_delta_grid": the iq3_s weight strip and the q8_K
    activation strip are read at `weight_base + block_index*110` and `activation_base +
    block_index*292`. When ABSENT the op keeps the standalone single-super-block form.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$element_count,
    AnyType:$vl,
    Optional<Index>:$block_index,
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
    I64Attr:$activation_quant_byte_offset
  );
  let results = (outs AnyType:$bsum);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $element_count `,` $vl "
    "(`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($weight_base) `,` type($activation_base) `,` "
    "type($element_count) `,` type($vl) `->` type($bsum)";
  let hasVerifier = 1;
}

// NOTE: def GgmlBlockDotIQ3SQ8KOp (the monolith iq3_s block-dot op) was RETIRED at the
// iq3_s flip (C_construct 22->23): the front door now constructs the typed super-block
// SCALAR-accumulator GRID loop body (tcrv_rvv.typed_super_block_block_dot_loop_body with
// fold_model "scalar_delta_grid", stride 110) carrying the iq3_s GRID-of-4 explicit-signs
// grid-core brick (GgmlBlockDotIQ3SQ8KGridCoreOp) instead. The retired monolith emitter's
// per-super-block body was extracted into the byte-exact shared anchor
// emitIQ3SSuperBlockGridBody (the sole live caller is the constructed loop lowering).

// NOTE: def GgmlBlockDotIQ1MQ8KOp (the monolith iq1_m block-dot op) was RETIRED at
// the iq1_m flip (L3): the front door constructs the typed super-block loop body
// carrying the iq1_m grid-core brick (GgmlBlockDotIQ1MQ8KGridCoreOp) instead.

// ===========================================================================
// M-FLAT FORWARD-ELEMENTWISE scaffold (line C, ① 之后): the typed elementwise
// strip-loop body op + its map/reduce core brick + terminator. This is the
// forward-pass (non-dot) sibling of the tcrv_rvv.typed_flat_block_dot_loop_body
// block-dot scaffold: it puts the ELEMENTWISE strip loop (the
// `for (i = 0; i < n; i += vlmax)` f32 map, no AoS block loop, no integer
// widening, no cross-block accumulator) INTO the CORE typed body as ONE
// region-carrying dataflow op, so the emission is determined by op identity + the
// explicit reduce/map model, never by an opaque hand-written helper. The FIRST
// forward operator constructed through it is scale (y[i] *= v, reduce_map_model
// "map"); rms_norm / softmax (reduce_map_model "reduce") reuse the same scaffold.
// ===========================================================================

def TypedElementwiseLoopBodyOp
    : TCRVRVV_Op<"typed_elementwise_loop_body",
    [SingleBlock, TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV f32 elementwise strip-loop op for the forward-pass "
      "(non-dot) map/reduce operators";
  let description = [{
    Records a forward-pass f32 elementwise operator's OUTER strip loop as ONE
    typed region-carrying dataflow op: the VLEN-robust
    `for (i = 0; i < n; i += vlmax)` strip loop over a flat unit-stride f32
    buffer, whose region carries the per-strip core brick that does the actual
    map/reduce work. It is the forward-pass (non-dot) sibling of
    tcrv_rvv.typed_flat_block_dot_loop_body: like that op it makes the loop +
    the per-strip work a first-class typed body (I5) so the emission is
    determined by op identity + the explicit `reduce_map_model` fact, NEVER
    inferred from candidate-mirror metadata, route ids, ABI strings, artifact
    names, status fields, exact intrinsic spellings, or op-name strings, and
    NEVER an opaque hand-written elementwise helper.

    `reduce_map_model` fixes the loop SHAPE (region-invariant): "map" is the
    pure elementwise map -- NO loop-carried accumulator, the region carries just
    the strip induction variable and the per-strip core brick writes its result
    straight to the buffer (scale's `y[i] *= v`, silu's
    `y[i] = x[i]*sigmoid(x[i])`); "reduce" carries a loop-carried scalar/vector
    accumulator for the row reductions (rms_norm's Σx², softmax's Σe^x): the
    region carries a SECOND entry argument (the loop-carried accumulator value
    at loop entry) and the yield names ONE operand (the updated accumulator),
    exactly the loop-carried-acc modeling the sibling
    tcrv_rvv.typed_flat_block_dot_loop_body uses for the block-dot `sumf`.
    "rotate" is the per-PAIR SCALAR recurrence shape (rope's rotary position
    embedding): the loop steps over dim-PAIRS (`for p in 0..n/2`, stride-2
    addressing, NOT a vectorized `i += vlmax` strip) and the region carries a
    SECOND entry argument that is a loop-carried f32 SCALAR RECURRENCE value
    (rope's `theta`, stepped `theta *= theta_scale` per pair) -- the SAME
    loop-carried-scalar region SHAPE the "reduce" model pioneered (region arg 1
    = the carried value, the yield carries it back, realized as an
    emitc.variable lvalue since emitc.for has no iter_args), EXCEPT the carried
    value is a data-INDEPENDENT recurrence (not a reduction of the buffer) and
    the per-pair core brick applies a position-dependent 2x2 rotation with a
    scalar-libm cos/sin angle seam (no vectorized strip). The current surface
    admits "map", "reduce", and "rotate".

    The op consumes the forward operator's runtime ABI operands -- the in/out
    f32 buffer base pointer `buffer`, the runtime f32 `scalar` broadcast value,
    and the runtime element count `n` -- and owns one single-block region whose
    entry arguments are the `strip_index` induction variable (index) and, for
    the "reduce" model, the loop-carried accumulator value. The region is
    terminated by tcrv_rvv.typed_elementwise_loop_yield (no carried operand for
    the map model; the updated accumulator for the reduce model). The bounded
    scheduling facts (`element_sew`, the
    optional `strip_lmul`) are the byte-exact shape knobs the monolithic
    forward emit reads (the *how* -- SEW / strip width -- never the *what*): the
    map is a bare per-lane multiply (no FMA, no reduction), so it is byte-exact
    at any LMUL because the strip re-strips via the runtime `vsetvl_e32m<L>(n-i)`.
    The op does NOT own dtype/route authority; the per-strip compute is a
    separate typed core brick carried inside the region.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$scalar,
    AnyType:$n,
    StrAttr:$kind,
    StrAttr:$reduce_map_model,
    I64Attr:$element_sew,
    OptionalAttr<StrAttr>:$strip_lmul
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $buffer `,` $scalar `,` $n
    attr-dict-with-keyword $body `:`
    type($buffer) `,` type($scalar) `,` type($n)
  }];
  let hasVerifier = 1;
}

def TypedElementwiseLoopYieldOp
    : TCRVRVV_Op<"typed_elementwise_loop_yield",
    [Terminator,
     HasParent<"::tianchenrv::tcrv::rvv::TypedElementwiseLoopBodyOp">]> {
  let summary =
      "terminator of the typed elementwise strip-loop region (map model: no "
      "carried operand; reduce model: the updated accumulator)";
  let description = [{
    Terminates the tcrv_rvv.typed_elementwise_loop_body region. For the "map"
    reduce_map_model the map writes its result straight to the buffer per strip,
    so there is NO loop-carried value and the yield names no operand (mirroring
    an emitc.for with no iter_args). For the "reduce" reduce_map_model the yield
    names ONE operand -- the updated loop-carried accumulator (the reduce core
    brick's result), carried back to the loop-entry accumulator argument, the
    same SSA-acc <-> emitc-variable modeling the block-dot loop yield uses. It
    carries no compute, dtype, memory, or route semantics.
  }];

  let arguments = (ins Variadic<AnyType>:$acc_next);

  let assemblyFormat = "($acc_next^ `:` type($acc_next))? attr-dict";
  let hasVerifier = 1;
}

def ElementwiseScaleMapOp
    : TCRVRVV_Op<"elementwise_scale_map",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV per-strip f32 in-place scale map brick (y[i] *= v)";
  let description = [{
    Records ONE strip's f32 in-place elementwise scale -- the per-strip CORE of
    ggml's `ggml_vec_scale_f32` (y[i] = y[i] * v) -- as ONE bounded generic
    plugin-local typed RVV dataflow step: the map-family primitive of the
    forward-elementwise scaffold (the sibling of the block-dot integer-core
    product bricks). It is carried inside the tcrv_rvv.typed_elementwise_loop_body
    region and lowers to the byte-exact strip
    (`vsetvl_e32m<L>(n-i)` / `vle32` / `vfmul_vf` scalar-broadcast / `vse32`),
    node-identical to ggml's hand-written RVV path (vec.h:733-739).

    Byte-exactness is UNCONDITIONAL: `y[i] *= v` is a bare per-lane fp32 multiply
    (no FMA, so -ffp-contract cannot perturb it; no cross-lane reduction, so the
    LMUL / tail / strip-count are correctness-free resource knobs). The op
    consumes the in/out f32 buffer base pointer `buffer`, the runtime f32
    `scalar`, the runtime element count `n`, and the loop's `strip_index`
    induction variable (region argument 0). It has NO result -- the in-place
    strip store is the sink.

    The `strip_index` operand is the ANTI-BYPASS tie: it MUST be the loop
    induction variable (region argument 0), so the per-strip element pointer is
    `buffer + strip_index` -- the emit provably addresses strip i, not the
    loop-invariant strip 0 (fail-closed otherwise, exactly the flat/super-block
    per-block-source anti-bypass). The bounded surface `kind` is limited to
    "elementwise_scale_map". The optional `strip_lmul` is the bounded resource
    knob selecting the f32 strip anchor LMUL ("m1"|"m2"|"m4"|"m8", default "m8"
    to match ggml's apply path) -- the *how*, never the *what*. Route ids,
    helper names, ABI strings, artifact names, status fields, and exact
    intrinsic spellings must not infer or choose this map's semantics.
  }];

  let arguments = (ins
    AnyType:$buffer,
    AnyType:$scalar,
    AnyType:$n,
    Index:$strip_index,
    StrAttr:$kind,
    OptionalAttr<StrAttr>:$strip_lmul
  );

  let assemblyFormat =
    "$buffer `,` $scalar `,` $n `strip` $strip_index `:` type($strip_index) "
    "attr-dict `:` type($buffer) `,` type($scalar) `,` type($n)";
  let hasVerifier = 1;
}

def ElementwiseSiluMapOp
    : TCRVRVV_Op<"elementwise_silu_map",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV per-strip f32 silu map brick (y[i] = x[i]*sigmoid(x[i]))";
  let description = [{
    Records ONE strip's f32 silu -- the per-strip CORE of ggml's
    `ggml_vec_silu_f32` (`y[i] = x[i] * sigma(x[i])`, `sigma(x) = 1/(1+e^{-x})`;
    vec.cpp:380 + `ggml_v_silu_m2` vec.h:1363) -- as ONE bounded generic
    plugin-local typed RVV dataflow step: the SECOND map-family primitive of the
    forward-elementwise scaffold (the sibling of tcrv_rvv.elementwise_scale_map).
    It is carried inside the tcrv_rvv.typed_elementwise_loop_body region
    (reduce_map_model "map") and lowers to the byte-exact m2 strip
    (`vsetvl_e32m2(n-i)` / `vle32` / the node-for-node `ggml_v_expf_m2` polynomial
    / `vfadd 1.0f` / `vfdiv` / `vse32`), node-identical to ggml's hand-written RVV
    path. It reuses the SAME loop op + map model + emitter scaffold scale landed;
    the ONLY new machinery is this per-strip silu decode (the exp polynomial).

    BYTE-EXACTNESS hinges on replicating ggml's EXACT vectorized `ggml_v_expf_m2`
    intrinsic chain (the SHARED node-for-node exp polynomial soft_max also
    consumes): the `0x1.8p23f` round trick, the two-term Cayley range reduction
    (`vfnmsac` x2), the degree-5 Estrin polynomial (`vfmacc`), the integer
    exponent `vsll<<23 + 0x3f800000` reinterpret, and the overflow/underflow
    `vmerge` fixup -- every step a `__riscv_v` intrinsic with a FIXED bit-pattern
    constant, NO libm `expf` call. ggml's `if (!vcpop_m(c))` is a pure
    performance short-circuit (the fast path `k + j*k` equals the slow path's
    c-false lane bit-for-bit), so the lowering emits the slow-path `vmerge` value
    graph UNCONDITIONALLY -- same output bits for every input, no data-dependent
    branch. The strip + polynomial are pinned at m2 (ggml's `vsetvl_e32m2` path +
    the m2-tied `vbool16_t`/`vuint32m2_t` mask/reinterpret types), so there is NO
    `strip_lmul` knob this map -- silu carries only its `kind`.

    The op consumes the read-only f32 input base pointer `input` (the ggml x[]
    row), the written f32 output base pointer `output` (the ggml y[] buffer), the
    runtime element count `n`, and the loop's `strip_index` induction variable
    (region argument 0). It has NO result -- the per-strip output store is the
    sink. The `strip_index` operand is the ANTI-BYPASS tie: it MUST be the loop
    induction variable (region argument 0), so the per-strip pointers are
    `input + strip_index` / `output + strip_index` -- the emit provably addresses
    strip i, not the loop-invariant strip 0 (fail-closed otherwise). The bounded
    surface `kind` is limited to "elementwise_silu_map". Route ids, helper names,
    ABI strings, artifact names, status fields, and exact intrinsic spellings must
    not infer or choose this map's semantics.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$output,
    AnyType:$n,
    Index:$strip_index,
    StrAttr:$kind
  );

  let assemblyFormat =
    "$input `,` $output `,` $n `strip` $strip_index `:` type($strip_index) "
    "attr-dict `:` type($input) `,` type($output) `,` type($n)";
  let hasVerifier = 1;
}

def ElementwiseRmsNormReduceCoreOp
    : TCRVRVV_Op<"elementwise_rms_norm_reduce_core",
    [NoTerminator, TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV per-row f32 rms_norm reduce core brick (Sx^2 fold -> "
      "rsqrt -> normalize)";
  let description = [{
    Records ONE strip's contribution to the ggml
    `ggml_compute_forward_rms_norm_f32` (non-fused, no weight) row reduction --
    the per-element `acc_next = acc + (double)(x[i]*x[i])` fold -- as ONE bounded
    generic plugin-local typed RVV dataflow step: the FIRST reduce-family
    primitive of the forward-elementwise scaffold (the reduce sibling of the
    map-family tcrv_rvv.elementwise_scale_map / tcrv_rvv.elementwise_silu_map
    bricks). It is carried inside the tcrv_rvv.typed_elementwise_loop_body region
    (reduce_map_model "reduce") whose loop-carried accumulator argument feeds the
    brick's `acc` operand and whose yield carries the brick's `acc_next` result
    -- the genuine loop-carried-accumulator modeling the block-dot loop scaffold
    uses for `sumf` (the emit realizes acc as a `double sum` emitc.variable
    load-at-top / assign-at-bottom, since emitc.for has no iter_args).

    The brick carries the WHOLE rms_norm ABI so the reduce-body emitter sources
    every value from the region op (anti-bypass), NOT the loop op attrs: the
    read-only f32 input base pointer `input` (the ggml x[] row), the written f32
    output base pointer `output` (the ggml y[] buffer), the runtime f32 `eps`,
    the runtime element count `n`, the loop's `strip_index` induction variable
    (region argument 0), and the loop-carried f64 `acc` (region argument 1). The
    reduce loop is the byte-exact SCALAR-DOUBLE ascending fold (NOT a vectorized
    vfredusum, which folds in f32 + a tree order): the per-element square
    `x[i]*x[i]` rounds in f32 FIRST, is WIDENED to double (the FMA barrier), then
    added in double. The reduce-body emit closes with the scalar
    `mean = (float)(sum/(double)n)`, `scale = 1.0f/sqrtf(mean+eps)` (sqrtf is the
    one sanctioned scalar-libm opaque seam -- a true IEEE sqrt, NOT a hardware
    fast-rsqrt), and the VECTORIZED normalize strip `y[i] = x[i]*scale` (a bare
    per-lane vfmul_vf, byte-exact at any LMUL).

    The `strip_index` operand is the ANTI-BYPASS tie: it MUST be the loop
    induction variable (region argument 0). The `acc` operand MUST be the
    loop-carried accumulator (region argument 1), so the emit provably folds the
    carried value, not a fresh zero. The bounded surface `kind` is limited to
    "elementwise_rms_norm_reduce_core". The optional `strip_lmul` selects the f32
    NORMALIZE strip anchor LMUL ("m1"|"m2"|"m4"|"m8", default "m8" to match
    ggml's apply path) -- the *how* of the normalize, never the *what* (the
    reduction stays scalar-double regardless). Route ids, helper names, ABI
    strings, artifact names, status fields, and exact intrinsic spellings must
    not infer or choose this reduce's semantics.

    OPTIONAL FUSED rms_norm->mul EPILOGUE: the brick carries an optional
    single-block `$epilogue` region (0 blocks = plain rms_norm; 1 block = the
    fused rms_norm->mul, llama's attn_norm / ffn_norm). When present, the block
    carries ONE entry argument -- the per-strip normalized vector `vy` (an f32
    RVV vector token at this brick's NORMALIZE strip LMUL) -- and ONE
    tcrv_rvv.elementwise_mul_map brick that multiplies that vy by a loaded weight
    strip and stores the fused result. The chain edge is a KERNEL-LEVEL declared
    producer->consumer link (NOT a graph rewriter): the normalize strip's vy
    flows STRAIGHT into the mul in the SAME emitted strip loop, so the
    intermediate normalized row is NEVER stored to memory and NEVER reloaded (the
    fused kernel elides the 2*n*4-byte normalize store + reload round-trip).
    BYTE-EXACT: the register-kept vy is bit-identical to a store-then-reload of
    the same f32 vector, and `vy * w[i]` is a bare per-lane fp32 multiply (no
    FMA -- no add follows; no reduction). The entry-argument vy exists only in
    the emitted C (the per-strip value is not a real SSA value at the typed-body
    layer); the region is a declaration the reduce-body emitter splices into the
    normalize strip.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$output,
    AnyType:$eps,
    AnyType:$n,
    Index:$strip_index,
    F64:$acc,
    StrAttr:$kind,
    OptionalAttr<StrAttr>:$strip_lmul
  );
  let results = (outs F64:$acc_next);
  // The optional fused rms_norm->mul epilogue: 0 blocks (plain rms_norm, the
  // unfused byte-identical path) or 1 block carrying the per-strip normalized
  // vector entry argument + the tcrv_rvv.elementwise_mul_map consumer brick. The
  // NoTerminator trait lets the block hold just the mul_map brick.
  let regions = (region AnyRegion:$epilogue);

  let assemblyFormat =
    "$input `,` $output `,` $eps `,` $n `strip` $strip_index `acc` $acc "
    "attr-dict `:` type($input) `,` type($output) `,` type($eps) `,` "
    "type($n) `,` type($strip_index) `,` type($acc) `->` type($acc_next) "
    "(`epilogue` $epilogue^)?";
  let hasVerifier = 1;
}

def ElementwiseMulMapOp
    : TCRVRVV_Op<"elementwise_mul_map",
    [NoTerminator, TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV per-strip f32 mul epilogue brick (z[i] = vy[i] * w[i]); "
      "the fused rms_norm->mul consumer";
  let description = [{
    Records ONE strip's f32 elementwise multiply of a producer's per-strip
    NORMALIZED vector by a weight row -- the per-strip CORE of ggml's
    `ggml_vec_mul_f32` (`z[i] = x[i] * y[i]`; vec.h) running as the EPILOGUE of a
    fused rms_norm->mul (llama's attn_norm / ffn_norm: `ggml_rms_norm` followed
    immediately by `ggml_mul` against the learned weight vector) -- as ONE bounded
    generic plugin-local typed RVV dataflow step: the mul-family epilogue
    primitive of the forward-elementwise scaffold (the consumer sibling of the
    map-family tcrv_rvv.elementwise_scale_map / tcrv_rvv.elementwise_silu_map and
    the reduce-family tcrv_rvv.elementwise_rms_norm_reduce_core bricks).

    It is carried inside the OPTIONAL `$epilogue` region of the producer
    tcrv_rvv.elementwise_rms_norm_reduce_core brick. The producer's VECTORIZED
    normalize strip computes the per-strip normalized vector `vy = vx * scale`
    (a bare per-lane vfmul_vf); this brick loads the weight strip `vw = w[i]` and
    stores `z[i] = vy * w[i]`. The KEY fusion fact: the intermediate normalized
    row is NEVER stored to memory and NEVER reloaded -- `vy` stays
    register-resident and flows straight into this multiply in the SAME emitted
    strip loop, so the fused kernel elides the producer's normalize store + the
    consumer's reload (2*n*4 bytes of DRAM round-trip). BYTE-EXACTNESS is
    UNCONDITIONAL: the register-kept `vy` is bit-identical to a store-then-reload
    of the same f32 vector; `vy * w[i]` is a bare per-lane fp32 multiply (no FMA,
    since no add follows -- so -ffp-contract cannot perturb it; no cross-lane
    reduction, so the LMUL / tail / strip-count are correctness-free resource
    knobs); and the mul runs at the SAME LMUL anchor as the producer's normalize
    strip, so `vy` is consumed at its native width.

    The `chain` operand is the ANTI-BYPASS chain tie: it MUST be the enclosing
    `$epilogue` region's block argument 0 -- the per-strip normalized vector (an
    f32 RVV vector token at the producer's strip LMUL, realized as the C `vy`
    variable in the emitted strip loop; it does not exist as a real SSA value at
    the typed-body layer because the per-strip value only lives inside the emitted
    for-loop). Binding `chain` to that block arg proves the consumer eats the
    producer's per-strip result, NOT a memory reload. The `strip_index` operand
    is the SECOND anti-bypass tie: it MUST be the enclosing
    tcrv_rvv.typed_elementwise_loop_body's induction variable (region argument 0),
    so the emit addresses `weight + strip_index` / `output + strip_index`, not the
    loop-invariant strip 0. The `output` operand MUST be the producer reduce
    core's `output` (the single fused destination z[] buffer).

    The op consumes the per-strip normalized vector `chain`, the read-only f32
    weight base pointer `weight` (`const float *`, ggml's mul y[] = the learned
    weight row), the written f32 output base pointer `output` (`float *`, ggml's
    z[] fused result), the runtime element count `n`, and the loop's
    `strip_index`. It has NO result -- the per-strip store is the sink. The
    bounded surface `kind` is limited to "elementwise_mul_map". Route ids, helper
    names, ABI strings, artifact names, status fields, and exact intrinsic
    spellings must not infer or choose this map's semantics.

    OPTIONAL [FMT-PROP] FUSED-ACTIVATION-QUANTIZE EPILOGUE: the brick carries an
    optional single-block `$quant_epilogue` region (0 blocks = the plain fused
    rms_norm->mul that stores the f32 z[]; 1 block = the FORMAT-PROPAGATED chain
    that quantizes the weighted activation to block_q8_0 IN REGISTER). When
    present, the block carries ONE entry argument -- the per-block WEIGHTED vector
    `vz` (an f32 RVV vector at the producer normalize strip LMUL, pinned m8 so vz
    is exactly one e32m8 QK8_0 block) -- and ONE
    tcrv_rvv.elementwise_quantize_q8_0_map consumer brick. The chain edge is a
    KERNEL-LEVEL declared producer->consumer link (NOT a graph rewriter): the
    per-block weighted vz flows STRAIGHT into the ggml per-block amax/scale/narrow
    q8_0 body in the SAME emitted block loop, so the f32 z[] intermediate is NEVER
    stored to memory and the downstream INDEPENDENT `quantize_row_q8_0` pass (the
    f32 activation store + the quantize reload, 2*n*4 bytes of DRAM round-trip) is
    ELIDED. BYTE-EXACT: the register-kept vz is bit-identical to a store-then-reload
    of the same f32 vector, so the per-block amax reduction / scalar scale /
    f32->i16->i8 narrow run on the IDENTICAL block values the two-kernel path would
    read from z[] -- the fused block_q8_0 is byte-exact to the non-fused
    rms_norm->mul->quantize pipeline modulo ONLY the eliminated store/reload.
  }];

  let arguments = (ins
    AnyType:$chain,
    AnyType:$weight,
    AnyType:$output,
    AnyType:$n,
    Index:$strip_index,
    StrAttr:$kind
  );
  // The optional [FMT-PROP] fused-quantize epilogue: 0 blocks (the plain fused
  // rms_norm->mul, f32 z[] store) or 1 block carrying the per-block weighted
  // vector entry argument + the tcrv_rvv.elementwise_quantize_q8_0_map consumer
  // brick. The NoTerminator-free single-op block holds just the quantize brick.
  let regions = (region AnyRegion:$quant_epilogue);

  let assemblyFormat =
    "$chain `,` $weight `,` $output `,` $n `strip` $strip_index attr-dict `:` "
    "type($chain) `,` type($weight) `,` type($output) `,` type($n) `,` "
    "type($strip_index) (`quant_epilogue` $quant_epilogue^)?";
  let hasVerifier = 1;
}

def ElementwiseQuantizeQ80MapOp
    : TCRVRVV_Op<"elementwise_quantize_q8_0_map",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV per-block f32->block_q8_0 quantize epilogue brick (the "
      "[FMT-PROP] fused-activation quantizer consuming the rms_norm->mul vz)";
  let description = [{
    Records ONE block's f32 -> block_q8_0 quantization of a producer's per-block
    WEIGHTED activation vector -- the per-block CORE of ggml's `quantize_row_q8_0`
    RVV path (riscv/quants.c:43-65) running as the [FMT-PROP] FORMAT-PROPAGATED
    EPILOGUE of a fused rms_norm->mul (llama's attn_norm / ffn_norm immediately
    followed by the downstream activation quantize that feeds the q4_0_q8_0 /
    q8_0_q8_0 block-dot kernels) -- as ONE bounded generic plugin-local typed RVV
    dataflow step: the QUANT-BRIDGE epilogue primitive of the forward-elementwise
    scaffold (the consumer sibling of the mul-family
    tcrv_rvv.elementwise_mul_map).

    It is carried inside the OPTIONAL `$quant_epilogue` region of the producer
    tcrv_rvv.elementwise_mul_map brick. The producer's per-block weighted vector
    `vz = vy * w` (a bare per-lane vfmul_vv at the m8 QK8_0 block anchor) flows
    STRAIGHT into this brick's per-block ggml quantize body: the amax reduction
    (`vfredmax` of `vfabs`), the scalar `d = amax/127`, the load-bearing
    `id = d ? 1/d : 0`, the native `(_Float16)d` AoS store, the `vfmul_vf(vz, id)`
    scale, the f32->i16->i8 narrowing convert (`vfncvt` round-to-nearest-EVEN +
    `vncvt`), and the 32 int8 `qs` AoS store. The KEY [FMT-PROP] fusion fact: the
    f32 z[] intermediate is NEVER stored to memory and NEVER reloaded -- `vz`
    stays register-resident and is consumed in the SAME emitted block loop, so the
    fused kernel elides the producer's f32 z[] store + the downstream independent
    `quantize_row_q8_0` reload (2*n*4 bytes of DRAM round-trip). BYTE-EXACTNESS is
    UNCONDITIONAL vs the non-fused two-kernel path: the register-kept `vz` is
    bit-identical to a store-then-reload of the same f32 vector (IEEE binary32
    round-trips losslessly through memory), so the amax fold / scalar scale /
    narrowing convert run on the IDENTICAL block values -- the fused block_q8_0 is
    byte-exact to `rms_norm->mul` (writing z[]) followed by a separate
    `quantize_row_q8_0` reading z[], modulo ONLY the eliminated store/reload.
    Numerical bit-exact-vs-ggml is pending-hardware, matching the family precedent.

    The `chain` operand is the ANTI-BYPASS chain tie: it MUST be the enclosing
    `$quant_epilogue` region's block argument 0 -- the per-block weighted vector
    (an f32 RVV vector at the producer's m8 QK8_0 anchor, realized as the C `vz`
    variable in the emitted block loop; it does not exist as a real SSA value at
    the typed-body layer because the per-block value only lives inside the emitted
    block loop). Binding `chain` to that block arg proves the quantizer eats the
    producer's per-block weighted result, NOT a memory reload. The `strip_index`
    operand is the SECOND anti-bypass tie: it MUST be the enclosing
    tcrv_rvv.typed_elementwise_loop_body's induction variable (region argument 0),
    so the emit addresses the AoS `output + ib*34` block cursor, not the
    loop-invariant block 0.

    The op consumes the per-block weighted vector `chain`, the written block_q8_0
    output BYTE buffer base pointer `output` (`uint8_t *`, the AoS block buffer
    written), the runtime element count `element_count` (= ggml's k, n % 32 == 0 a
    ggml contract), and the loop's `strip_index`. It has NO result -- the per-block
    AoS d (fp16) + qs (int8) stores are the sink. The block-format facts (`qk` =
    32, `block_stride` = 34, the fp16 `d` at byte 0, the 32 int8 `qs` at byte 2)
    are the op's typed mirror attrs (I4), IDENTICAL to
    tcrv_rvv.quantize_row_q8_0's; the emission REUSES that op's per-block
    amax/scale/narrow body. The bounded surface `kind` is limited to
    "elementwise_quantize_q8_0_map". Route ids, helper names, ABI strings, artifact
    names, status fields, and exact intrinsic spellings must not infer or choose
    this quantizer's semantics.
  }];

  let arguments = (ins
    AnyType:$chain,
    AnyType:$output,
    AnyType:$element_count,
    Index:$strip_index,
    StrAttr:$kind,
    I64Attr:$qk,
    I64Attr:$block_stride,
    I64Attr:$scale_byte_offset,
    I64Attr:$quant_byte_offset
  );

  let assemblyFormat =
    "$chain `,` $output `,` $element_count `strip` $strip_index attr-dict `:` "
    "type($chain) `,` type($output) `,` type($element_count) `,` "
    "type($strip_index)";
  let hasVerifier = 1;
}

def ElementwiseSoftMaxReduceCoreOp
    : TCRVRVV_Op<"elementwise_soft_max_reduce_core",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV per-strip f32 soft_max exp-sum-reduce core brick "
      "(y[i]=e^{x[i]-max}; f64 widening-reduce fold)";
  let description = [{
    Records ONE strip's contribution to ggml's BARE `ggml_vec_soft_max_f32`
    (vec.cpp:531 + the `__riscv_v` path vec.cpp:584-592) -- the fused
    `y[i] = e^{x[i]-max}` write PLUS the `Sum_i e^{x[i]-max}` widening f32->f64
    reduce -- as ONE bounded generic plugin-local typed RVV dataflow step: the
    SECOND reduce-family primitive of the forward-elementwise scaffold (the
    exp-sum-reduce sibling of the Σx² fold
    tcrv_rvv.elementwise_rms_norm_reduce_core). It is carried inside the
    tcrv_rvv.typed_elementwise_loop_body region (reduce_map_model "reduce") whose
    loop-carried accumulator argument feeds the brick's `acc` operand and whose
    yield carries the brick's `acc_next` result -- the SAME loop-carried
    accumulator model rms_norm built, REUSED, EXCEPT the accumulator is the f64m1
    WIDENING-reduce vector (ggml's `vfloat64m1_t vsum`), NOT a scalar double:
    ggml accumulates Sum e^x in ggml_float = DOUBLE via
    `vfwredusum_vs_f32m2_f64m1` into a SINGLE f64m1 accumulator carried across
    strips, and matching THAT exact widening-reduce fold is the byte-exactness
    crux for the returned sum.

    Unlike rms_norm's reduce-THEN-map shape (a scalar-double fold, then a
    separate vectorized normalize strip), soft_max is a FUSED per-strip
    map+reduce: each strip subtracts `max`, runs the SHARED node-for-node
    `ggml_v_expf_m2` exp polynomial (emitGgmlVExpfM2, bit-identical to silu's
    exp), stores `y[i] = e^{x[i]-max}`, then folds the SAME val vector into the
    f64m1 accumulator via the widening reduce. -inf masked entries (attention
    masks) flow through the polynomial's underflow path (`exp -> 0`) with no
    special-casing (x-max with x=-inf gives -inf; the |n|>192 saturating branch
    yields 0 bit-exactly). The BARE function does NOT normalize (ggml's caller
    applies `1/sum` via `ggml_vec_scale_f32`; ops.cpp:5415-5416) and takes `max`
    as an INPUT (ggml's caller computes it with the scalar `ggml_vec_max_f32`;
    ops.cpp:5400); this brick mirrors THAT bare function -- the exp-sum-reduce
    pass only. The f64 sum is the enclosing function's C return value (the only
    forward-pass op whose function RETURNS a scalar).

    The brick carries the WHOLE soft_max ABI so the reduce-body emitter sources
    every value from the region op (anti-bypass), NOT the loop op attrs: the
    written f32 output base pointer `output` (the ggml y[] = e^{x-max} buffer),
    the read-only f32 input base pointer `input` (the ggml x[] row), the runtime
    f32 `max`, the runtime element count `n`, the loop's `strip_index` induction
    variable (region argument 0), and the loop-carried f64m1 `acc` (region
    argument 1). It produces the updated f64m1 `acc_next`.

    The `strip_index` operand is the ANTI-BYPASS tie: it MUST be the loop
    induction variable (region argument 0). The `acc` operand MUST be the
    loop-carried accumulator (region argument 1), so the emit provably folds the
    carried vsum, not a fresh zero. The bounded surface `kind` is limited to
    "elementwise_soft_max_reduce_core". There is NO strip_lmul knob: the strip
    loop and the exp polynomial are m2-pinned (ggml's vsetvl_e32m2 path + the
    m2-tied vbool16_t/vuint32m2_t mask/reinterpret types) and the reduce is f64m1
    (the vfwredusum destination), so the fold is m2/f64m1-shaped with no
    byte-exact LMUL freedom -- matching silu's precedent. Route ids, helper
    names, ABI strings, artifact names, status fields, and exact intrinsic
    spellings must not infer or choose this reduce's semantics.
  }];

  let arguments = (ins
    AnyType:$output,
    AnyType:$input,
    AnyType:$max,
    AnyType:$n,
    Index:$strip_index,
    AnyType:$acc,
    StrAttr:$kind
  );
  let results = (outs AnyType:$acc_next);

  let assemblyFormat =
    "$output `,` $input `,` $max `,` $n `strip` $strip_index `acc` $acc "
    "attr-dict `:` type($output) `,` type($input) `,` type($max) `,` "
    "type($n) `,` type($strip_index) `,` type($acc) `->` type($acc_next)";
  let hasVerifier = 1;
}

def ElementwiseRopeRotateCoreOp
    : TCRVRVV_Op<"elementwise_rope_rotate_core",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV per-pair f32 rope rotate core brick (position-dependent "
      "2x2 rotation + theta recurrence)";
  let description = [{
    Records ONE dim-PAIR's contribution to ggml's `ggml_compute_forward_rope_f32`
    NORMAL rope (the GGML_ROPE_TYPE_NORMAL variant LLM_ARCH_LLAMA uses; the
    per-pair rotation on CONSECUTIVE head values x[2p]/x[2p+1]) -- the scalar
    cos/sin angle cache PLUS the position-dependent 2x2 rotation PLUS the f32
    angle recurrence step -- as ONE bounded generic plugin-local typed RVV
    dataflow step: the FIRST rotate-family primitive of the forward-elementwise
    scaffold (the per-pair-recurrence sibling of the map-family
    tcrv_rvv.elementwise_scale_map / tcrv_rvv.elementwise_silu_map and the
    reduce-family tcrv_rvv.elementwise_rms_norm_reduce_core /
    tcrv_rvv.elementwise_soft_max_reduce_core bricks). It is carried inside the
    tcrv_rvv.typed_elementwise_loop_body region (reduce_map_model "rotate") whose
    loop-carried f32 `theta` region argument feeds the brick's `theta` operand
    and whose yield carries the brick's `theta_next` result -- the SAME
    loop-carried-scalar region shape the reduce model built (the emit realizes
    theta as a `float` emitc.variable seeded from theta_base BEFORE the loop /
    load-at-top / assign-at-bottom, since emitc.for has no iter_args), EXCEPT the
    carried value is a data-INDEPENDENT recurrence, NOT a reduction of the
    buffer, and the loop is a SCALAR per-pair loop (stride-2 addressing over
    `n/2` pairs), NOT a vectorized `i += vlmax` strip.

    The brick carries the WHOLE rope ABI so the rotate-body emitter sources every
    value from the region op (anti-bypass), NOT the loop op attrs: the read-only
    f32 input base pointer `input` (the ggml x[] head row), the written f32
    output base pointer `output` (the ggml y[] buffer), the runtime f32
    `theta_base` (the position pos as f32, the recurrence seed), the runtime f32
    `theta_scale` (powf(freq_base, -2/n_dims), the recurrence ratio -- PRECOMPUTED
    so the kernel makes no powf call), the runtime element count `n` (ggml's ne0
    = the head dim, n % 2 == 0 a ggml contract), the loop's `pair_index`
    induction variable (region argument 0), and the loop-carried f32 `theta`
    (region argument 1). It produces the updated f32 `theta_next`.

    BYTE-EXACTNESS has TWO distinct axes (surfaced honestly, matching the retired
    monolith): (1) the angle transcendental is SCALAR libm `cosf`/`sinf` (one
    emitc.call_opaque each -- the sanctioned opaque seam, a DIFFERENT
    byte-exactness axis from silu/soft_max's replicated vectorized exp
    polynomial: libm-LINKED, so a same-libm deployment is bit-exact and a
    different libm is libm-tolerance bound on the angles only); (2) each output's
    rotation `x0*cos - x1*sin` / `x0*sin + x1*cos` is an FP-contraction hazard
    GROUPED into ONE emitc.expression (the rms_norm emitc.expression FMA-fix
    discipline), so mlir-translate renders it as ONE C statement token-identical
    to ggml's single C expression -> clang contracts identically under every
    -ffp-contract mode -> byte-exact regardless of the build flag.

    The `pair_index` operand is the ANTI-BYPASS tie: it MUST be the enclosing
    loop's induction variable (region argument 0). The `theta` operand MUST be
    the loop-carried recurrence value (region argument 1), so the emit provably
    steps the carried angle, not a fresh seed. The bounded surface `kind` is
    limited to "elementwise_rope_rotate_core". There is NO strip_lmul knob: cos/
    sin are scalar libm (one call per pair), so the faithful structure IS ggml's
    scalar per-pair loop (vectorizing while cos/sin stay scalar would gather
    scalars into vectors with no byte-exactness gain), matching silu's no-knob
    precedent. Route ids, helper names, ABI strings, artifact names, status
    fields, and exact intrinsic spellings must not infer or choose this rotate's
    semantics.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$output,
    AnyType:$theta_base,
    AnyType:$theta_scale,
    AnyType:$n,
    Index:$pair_index,
    F32:$theta,
    StrAttr:$kind
  );
  let results = (outs F32:$theta_next);

  let assemblyFormat =
    "$input `,` $output `,` $theta_base `,` $theta_scale `,` $n `pair` "
    "$pair_index `theta` $theta attr-dict `:` type($input) `,` type($output) "
    "`,` type($theta_base) `,` type($theta_scale) `,` type($n) `,` "
    "type($pair_index) `,` type($theta) `->` type($theta_next)";
  let hasVerifier = 1;
}

// The four forward-elementwise f32 support ops (add / mul / cpy / gelu) are the
// non-headline forward-pass support family: each is ONE bounded generic
// plugin-local typed RVV op nested directly under tcrv_rvv.with_vl, DISPATCH-WIRED
// (production recognizer + hand-written monolith emitter body + lit), NOT
// constructed ([L-6] wiring != construction: no typed_elementwise_loop_body brick,
// no pattern-library primitive). They exist to move add/cpy/gelu/mul from `absent`
// to `dispatch-wired` in the six-state ledger (C_dispatch). The bounded surface
// `kind` pins the ggml operator; the emitter owns the m8 strip-loop (add/mul/cpy)
// or the scalar tanhf loop (gelu). Route ids / helper names / ABI strings / status
// fields must not infer or choose these ops' semantics.

def GgmlVecAddF32Op
    : TCRVRVV_Op<"vec_add_f32",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml ggml_vec_add_f32 forward-pass op (z[i]=x[i]+y[i])";
  let description = [{
    Records the ggml `ggml_vec_add_f32` forward-pass op (the per-lane f32 binary
    add `z[i] = x[i] + y[i]`; vec.h) as ONE bounded generic plugin-local typed RVV
    dataflow step, DISPATCH-WIRED (not constructed). The op is nested directly
    under tcrv_rvv.with_vl and consumes three runtime ABI value operands -- the two
    read-only f32 input base pointers `lhs`/`rhs` (`const float *`, ggml's x[]/y[]),
    the written f32 output base pointer `output` (`float *`, ggml's z[]), the
    runtime element count `element_count` (= ggml's n) -- plus the active
    !tcrv_rvv.vl token. It produces ONE f32 LMUL m1 !tcrv_rvv.vector token (the
    per-strip store boundary). BYTE-EXACTNESS is UNCONDITIONAL: `z[i] = x[i]+y[i]`
    is a bare per-lane fp32 add (no reduction, so the LMUL/tail/strip-count are
    correctness-free resource knobs); the m8 strip-loop the emitter writes
    (`vsetvl_e32m8` / `vle32` x2 / `vfadd_vv` / `vse32`) is exact at any anchor. The
    bounded surface `kind` is limited to "ggml_vec_add_f32".
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($lhs) `,` type($rhs) `,` type($output) `,` "
    "type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlVecMulF32Op
    : TCRVRVV_Op<"vec_mul_f32",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml ggml_vec_mul_f32 forward-pass op (z[i]=x[i]*y[i])";
  let description = [{
    Records the ggml `ggml_vec_mul_f32` forward-pass op (the per-lane f32 binary
    multiply `z[i] = x[i] * y[i]`; vec.h) as ONE bounded generic plugin-local typed
    RVV dataflow step, DISPATCH-WIRED (not constructed). The ABI mirrors
    tcrv_rvv.vec_add_f32 (two `const float *` inputs `lhs`/`rhs`, one `float *`
    output, the runtime element count, the !tcrv_rvv.vl token; one f32 m1 result).
    BYTE-EXACTNESS is UNCONDITIONAL: `z[i] = x[i]*y[i]` is a bare per-lane fp32
    multiply (no reduction), exact at any LMUL; the emitter writes the m8 strip-loop
    (`vsetvl_e32m8` / `vle32` x2 / `vfmul_vv` / `vse32`). The bounded surface `kind`
    is limited to "ggml_vec_mul_f32".
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$kind
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($lhs) `,` type($rhs) `,` type($output) `,` "
    "type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlVecCpyF32Op
    : TCRVRVV_Op<"vec_cpy_f32",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml ggml_vec_cpy_f32 forward-pass op (y[i]=x[i])";
  let description = [{
    Records the ggml `ggml_vec_cpy_f32` forward-pass op (the per-lane f32 copy
    `y[i] = x[i]`; vec.h / the f32->f32 `ggml_compute_forward_dup` case) as ONE
    bounded generic plugin-local typed RVV dataflow step, DISPATCH-WIRED (not
    constructed). The op consumes the read-only f32 input base pointer `input`
    (`const float *`, ggml's x[]), the written f32 output base pointer `output`
    (`float *`, ggml's y[]), the runtime element count `element_count`, plus the
    active !tcrv_rvv.vl token; it produces ONE f32 LMUL m1 result. BYTE-EXACTNESS is
    UNCONDITIONAL: the copy moves the f32 bit pattern unchanged; the emitter writes
    the m8 strip-loop (`vsetvl_e32m8` / `vle32` / `vse32`). The bounded surface
    `kind` is limited to "ggml_vec_cpy_f32".
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

def GgmlGeluF32Op
    : TCRVRVV_Op<"gelu_f32",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml gelu forward-pass op (y[i]=gelu(x[i]))";
  let description = [{
    Records the ggml gelu forward-pass op (`y[i] = gelu(x[i])`; the tanh-
    approximation `ggml_gelu_f32` that SEEDS ggml's f16 gelu table:
    `0.5*x*(1 + tanhf(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x)))`,
    GELU_COEF_A = 0.044715f, SQRT_2_OVER_PI = 0.79788456080286535587989211986876f)
    as ONE bounded generic plugin-local typed RVV dataflow step, DISPATCH-WIRED
    (not constructed). The op consumes the read-only f32 input base pointer `input`
    (`const float *`), the written f32 output base pointer `output` (`float *`), the
    runtime element count `element_count`, plus the active !tcrv_rvv.vl token; it
    produces ONE f32 LMUL m1 result. Unlike the bare add/mul/cpy strips, gelu is a
    SCALAR per-element loop: `tanhf` is the sanctioned scalar-libm opaque seam (the
    sibling of rope's cosf/sinf and rms_norm's sqrtf), so the faithful thin body is
    ggml's scalar formula, one `tanhf` call per element. Bit-exactness to ggml's
    DEPLOYED f16-table gelu path is pending-hardware (this thin dispatch-wired body
    emits the reference tanh gelu, not the 65536-entry table lookup). The bounded
    surface `kind` is limited to "ggml_gelu_f32".
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

    q8_1 is the SIBLING quantizer (same amax/scale/narrow + an extra `vwredsum`
    integer block sum stored as the fp16 `block_q8_1.s`); it is its own op
    (GgmlQuantizeRowQ81Op), NOT folded into this kind. q8_K is the K-quant
    activation quantizer (GgmlQuantizeRowQ8KOp) -- a heavier min/max-symmetric
    QK_K = 256 super-block quantizer with a float `d` + per-16 `bsums`.
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

def GgmlQuantizeRowQ81Op
    : TCRVRVV_Op<"quantize_row_q8_1",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml f32->block_q8_1 activation quantizer op";
  let description = [{
    Records the COMPLETE ggml `quantize_row_q8_1` RVV-path forward-pass op (the
    f32 -> block_q8_1 ACTIVATION QUANTIZER; riscv/quants.c) as ONE bounded generic
    plugin-local typed RVV dataflow step, DISPATCH-WIRED ([L-6] wiring !=
    construction: the emitter owns a hand-written monolith body, NOT a
    typed_flat_block_dot / pattern-library primitive). q8_1 is the SIBLING of the
    q8_0 quantizer: the SAME per-32-block max-abs REDUCTION (`vfredmax` of
    `vfabs`) + scalar scale (`d = amax/127`, `id = d ? 1/d : 0`) + f32->i16->i8
    NARROWING CONVERT (`vfncvt` round-to-nearest-EVEN + `vncvt` truncate) written
    into the AoS `qs`, PLUS an extra integer block sum `s = d * sum(qs[i])` stored
    as the fp16 `block_q8_1.s`. The sum is a WIDENING INTEGER reduction of the
    int8 quants (`vwredsum` of the i8 `vs` into an i16m1 accumulator seeded 0,
    then `vmv_x_s` extract), NOT a float reduction.

    block_q8_1 AoS layout (ggml-common.h:248-259 + QK8_1 = 32): the fp16 `d` at
    byte 0, the fp16 `s` at byte 2, the 32 int8 `qs` at byte 4; stride 36. The
    block-format facts (`qk` = 32, `block_stride` = 36, the scale/sum/quant byte
    offsets) are the op's typed mirror attrs (I4); the emission is the op's fixed
    structure. For the bounded surface `kind` is limited to
    "ggml_quantize_row_q8_1". The strip is pinned at the m8 anchor ggml uses (the
    block's 32 lanes in one e32m8 strip; there is NO strip_lmul knob, matching
    q8_0/silu precedent). Route ids, helper names, ABI strings, artifact names,
    status fields, and common EmitC/export code must not infer or choose
    f32->q8_1 quantizer semantics.

    BYTE-EXACTNESS to the DEPLOYED kernel hinges on ggml's EXACT RVV method, NOT
    the scalar `_ref`: the `vfncvt_x_f_w_i16m4` round-to-nearest-EVEN narrow, the
    native `(_Float16)` casts for BOTH `d` and `s` (the board is __riscv_zfhmin),
    the load-bearing `d ? 1/d : 0` conditional (STRUCTURED emitc.cmp + emitc.if),
    and the `vwredsum` int8->i16 block sum feeding `(_Float16)(sum * d)`.

    The op is nested directly under tcrv_rvv.with_vl and consumes THREE runtime
    ABI value operands -- the read-only f32 input base pointer `x` (`const float
    *`), the block_q8_1 output BYTE buffer base pointer `vy` (`uint8_t *`), the
    runtime element count `n` (= ggml's `k`, n % 32 == 0 a ggml contract) -- plus
    the active !tcrv_rvv.vl token. It produces ONE f32 LMUL m1 !tcrv_rvv.vector
    token (the per-block store boundary).
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
    I64Attr:$sum_byte_offset,
    I64Attr:$quant_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$input `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($input) `,` type($output) `,` "
    "type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlQuantizeRowQ8KOp
    : TCRVRVV_Op<"quantize_row_q8_K",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml f32->block_q8_K activation quantizer op";
  let description = [{
    Records the COMPLETE ggml `quantize_row_q8_K` RVV-path forward-pass op (the
    f32 -> block_q8_K ACTIVATION QUANTIZER; riscv/quants.c) as ONE bounded generic
    plugin-local typed RVV dataflow step, DISPATCH-WIRED ([L-6] wiring !=
    construction). q8_K is the K-quant activation quantizer the K-quant block-dot
    vec_dots (q2_K..q6_K + the IQ grid tables) CONSUME. It is structurally the
    heaviest quantizer: over a QK_K = 256 super-block it runs a min AND max
    REDUCTION (a strip loop of `vfmax_vv`/`vfmin_vv` folded by `vfredmax`/
    `vfredmin` seeded -/+inf), a SYMMETRIC `iscale = -127 / (|max| > |min| ? max :
    min)` scale with `d = 1/iscale` (a FLOAT, NOT an fp16), a f32->i32->i16->i8
    NARROWING CONVERT with EXPLICIT round-to-nearest-even modes
    (`vfcvt_x_f_v_i32m8_rm` frm = __RISCV_FRM_RNE, then two `vnclip_wx` vxrm =
    __RISCV_VXRM_RNE), and 16 per-16-element block sums (`bsums`) each a
    `vwredsum` over a `vslidedown`-advanced 16-lane chunk. The all-zero
    super-block (`amax == 0`) is the load-bearing special case: `d = 0`, `qs` and
    `bsums` memset to 0, emitted as a STRUCTURED emitc.if/else (NOT a `continue`).

    block_q8_K AoS layout (ggml-common.h:360-366 + QK_K = 256): the FLOAT `d` at
    byte 0 (four bytes; NOT an fp16 like q8_0/q8_1), the 256 int8 `qs` at byte 4,
    the 16 int16 `bsums` at byte 260; stride 292. The block-format facts (`qk` =
    256, `block_stride` = 292, the scale/quant/bsums byte offsets) are the op's
    typed mirror attrs (I4). For the bounded surface `kind` is limited to
    "ggml_quantize_row_q8_K". The strip fold rides ggml's e32m8 vlmax anchor
    (`vsetvlmax_e32m8`); there is NO strip_lmul knob. Route ids, helper names, ABI
    strings, artifact names, status fields, and common EmitC/export code must not
    infer or choose f32->q8_K quantizer semantics.

    BYTE-EXACTNESS is to ggml's EXACT RVV method: the `vfcvt`/`vnclip` explicit
    RNE rounding modes, the `fabsf`-based symmetric `iscale`, the FLOAT `d =
    1/iscale` store, the `vwredsum` per-16 bsums, and the zero-block memset path.
    `fabsf` and `memset` are the sanctioned scalar/opaque seams; the -inf/+inf
    reduction seeds are `-__builtin_inff()` / `__builtin_inff()`.

    The op is nested directly under tcrv_rvv.with_vl and consumes THREE runtime
    ABI value operands -- the read-only f32 input base pointer `x` (`const float
    *`), the block_q8_K output BYTE buffer base pointer `vy` (`uint8_t *`), the
    runtime element count `n` (= ggml's `k`, n % 256 == 0 a ggml contract) -- plus
    the active !tcrv_rvv.vl token. It produces ONE f32 LMUL m1 !tcrv_rvv.vector
    token (the per-block store boundary).
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
    I64Attr:$quant_byte_offset,
    I64Attr:$bsums_byte_offset
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$input `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($input) `,` type($output) `,` "
    "type($element_count) `,` type($vl) `->` type($result)";
  let hasVerifier = 1;
}

def GgmlDequantizeRowOp
    : TCRVRVV_Op<"dequantize_row",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV ggml dequantize_row family op (block_qX -> f32 row)";
  let description = [{
    Records the ggml `dequantize_row_<format>` support op (the block-quant DECODE
    that unpacks ONE quantized weight row back to f32; ggml-cpu quants.c) as ONE
    bounded generic plugin-local typed RVV dataflow step, DISPATCH-WIRED (not
    constructed: [L-6] wiring != construction; the emitter owns a hand-written
    monolith decode body, NOT a typed_flat_block_dot / pattern-library primitive).
    ONE parameterized op stands in for the whole dequantize_row family: the bounded
    `format` mirror attr pins WHICH ggml quant the block-decode targets, and the
    emitter routes on it to the per-format scalar reference decode.

    The op is nested directly under tcrv_rvv.with_vl and consumes three runtime ABI
    value operands -- the read-only quantized weight base pointer `input`
    (`const uint8_t *`, ggml's `const block_qX * x` taken as a byte cursor over the
    AoS block buffer), the written f32 output base pointer `output` (`float *`,
    ggml's y[]), the runtime element count `element_count` (= ggml's k) -- plus the
    active !tcrv_rvv.vl token. It produces ONE f32 LMUL m1 !tcrv_rvv.vector token
    (the per-strip store boundary).

    The per-format DECODE reuses the SAME block-decode already constructed for that
    format's block-dot vec_dot: the fp16 block scale via the `(float)*(const
    _Float16 *)` seam (the board is __riscv_zfhmin), the nibble unpack (q4_0/q4_1),
    the 5th-bit qh merge (q5_0/q5_1), the bare int8 scale (q8_0). BYTE-EXACTNESS is
    to ggml's reference `dequantize_row_<format>` (a scalar AoS block loop; no
    reduction, so LMUL/strip-count are correctness-free). The bounded surface
    `format` is limited to the wired allowlist (currently the legacy quants
    q4_0/q4_1/q5_0/q5_1/q8_0); an unwired format is rejected fail-closed (I7) so no
    six-state row can claim dispatch-wired without a real decode. Route ids / helper
    names / ABI strings / status fields must not infer or choose these semantics.
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$vl,
    StrAttr:$format
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$input `,` $output `,` $element_count `,` $vl "
    "attr-dict `:` type($input) `,` type($output) `,` "
    "type($element_count) `,` type($vl) `->` type($result)";
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

def BlockFp16ScaleProductOp : TCRVRVV_Op<"block_fp16_scale_product",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV per-block dual-fp16 scale reconstruction op";
  let description = [{
    Records one bounded generic plugin-local RVV per-block dual-fp16 SCALE
    reconstruction step as ONE typed dataflow op: read the two per-block fp16
    scale headers (d_x from the lhs block base, d_y from the rhs block base),
    convert each fp16 -> f32, and multiply into ONE f32 scalar
    `f32(d_x) * f32(d_y)` (ggml's q8_0 scale order -- the two scales are
    multiplied FIRST, the sumi fold `(float)sumi * (d_x * d_y)` consumes this
    product downstream). The op consumes two explicitly imported runtime ABI
    block-base pointers and produces ONE f32 scalar result; it does NOT touch
    the integer quant payload, the block loop, or the fp32 accumulate (those
    are separate typed steps).

    This is the FIRST scalar-domain typed construction primitive: before it the
    per-block fp16->f32 scale read existed only as an inline opaque
    emitc.call_opaque in the monolithic block-dot lowering (the "one sanctioned
    opaque piece"). Making the dual-fp16 scale product a first-class typed body
    op (I5) means emission is determined by op identity + the explicit
    dual-fp16 scale_model, never inferred from candidate-mirror metadata, route
    ids, ABI strings, artifact names, status fields, exact intrinsic spellings,
    or q8/q4 names.

    The f32 range fully covers the fp16 domain, so the reconstruction is
    byte-exact by construction. The optional `lhs_scale_byte_offset` /
    `rhs_scale_byte_offset` facts locate the fp16 header within each block base
    (default 0 -- the scale is the first 2 AoS bytes, QK=32 payload at +2); the
    quant payload byte offset is NOT this op's concern. For the bounded surface
    kind is limited to "dual_fp16_per_block_scale_product" and scale_model to
    "dual-fp16-per-block-d_x.d_y".

    LOOP-CAPABLE per-block-source form (M-FLAT step 3): the op is additively
    extended with an OPTIONAL `block_index` operand and `lhs_block_stride` /
    `rhs_block_stride` (I64) attrs so it can typed-express EACH block's fp16
    scale header inside the flat block loop. When `block_index` is present it is
    the induction variable (region argument 0) of the enclosing
    tcrv_rvv.typed_flat_block_dot_loop_body region, and the two scale headers
    live at `base + block_index*stride (+ byte_offset)` -- the imported ABI
    bases are STILL the loop-invariant block-0 pointers (the base-import
    contract is NOT relaxed; only the per-block loop offset is added). This
    lowers byte-exact to the monolithic emitFlatBlockDot's blockBaseValue
    (base + ib*stride) + fp16ReadAt reads. When `block_index` is ABSENT the op
    keeps the single-block form unchanged: the imported ABI base IS the block
    base, no per-block stride, and `lhs_block_stride` / `rhs_block_stride` must
    not be set (fail-closed). The single-block form is the block_index-defaulted
    degenerate special case of the loop variant.
  }];

  let arguments = (ins
    AnyType:$lhs_scale_base,
    AnyType:$rhs_scale_base,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    StrAttr:$scale_model,
    OptionalAttr<I64Attr>:$lhs_scale_byte_offset,
    OptionalAttr<I64Attr>:$rhs_scale_byte_offset,
    OptionalAttr<I64Attr>:$lhs_block_stride,
    OptionalAttr<I64Attr>:$rhs_block_stride
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$lhs_scale_base `,` $rhs_scale_base (`block` $block_index^ `:` type($block_index))? attr-dict `:` type($lhs_scale_base) `,` type($rhs_scale_base) `->` type($result)";
  let hasVerifier = 1;
}

def BlockFp16MinProductOp : TCRVRVV_Op<"block_fp16_min_product",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV per-block dual-fp16 MIN/SUM correction product op";
  let description = [{
    Records one bounded generic plugin-local RVV per-block dual-fp16 MIN/SUM
    correction step as ONE typed dataflow op: read the two per-block fp16
    correction headers (m_x, the per-block MIN, from the lhs block base at
    lhs_min_byte_offset; s_y, the PRECOMPUTED scaled activation sum, from the rhs
    block base at rhs_sum_byte_offset), convert each fp16 -> f32, and multiply
    into ONE f32 scalar `f32(m_x) * f32(s_y)`. This is the SECOND per-block scalar
    header product of the ggml Family-B (q4_1 / q5_1) fold `sumf += (d_x*d_y)*sumi
    + m_x*s_y`: it is the DISTINCT sibling of tcrv_rvv.block_fp16_scale_product
    (which reads d_x/d_y at byte offset 0 and multiplies the two SCALES). The op
    consumes two explicitly imported runtime ABI block-base pointers and produces
    ONE f32 scalar result; it does NOT touch the integer quant payload, the block
    loop, or the fp32 accumulate (those are separate typed steps).

    The op is a distinct op TYPE (NOT a second block_fp16_scale_product) so the
    two per-block scalar products stay separable region ops: the scale product
    feeds tcrv_rvv.block_computed_scale_dequant's computed_scale, the min product
    feeds that op's OPTIONAL min_term. The f32 range fully covers the fp16 domain,
    so the reconstruction is byte-exact by construction. For the bounded surface
    kind is limited to "dual_fp16_per_block_min_product" and scale_model to
    "dual-fp16-per-block-m_x.s_y"; emission is determined by op identity + the
    explicit scale_model, never inferred from candidate-mirror metadata, route
    ids, ABI strings, artifact names, status fields, exact intrinsic spellings, or
    q8/q4 names.

    LOOP-CAPABLE per-block-source form (mirrors block_fp16_scale_product): the
    OPTIONAL `block_index` operand toggles the per-block-source form; when present
    it is the induction variable (region argument 0) of the enclosing
    tcrv_rvv.typed_flat_block_dot_loop_body region and the two headers live at
    `base + block_index*stride (+ byte_offset)`, with the imported ABI bases still
    the loop-invariant block-0 pointers. When absent the single-block form keeps
    the imported ABI base as the block base and the per-block strides must not be
    set (fail-closed). The `lhs_min_byte_offset` / `rhs_sum_byte_offset` facts
    locate the fp16 correction header within each block base.
  }];

  let arguments = (ins
    AnyType:$lhs_min_base,
    AnyType:$rhs_sum_base,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    StrAttr:$scale_model,
    OptionalAttr<I64Attr>:$lhs_min_byte_offset,
    OptionalAttr<I64Attr>:$rhs_sum_byte_offset,
    OptionalAttr<I64Attr>:$lhs_block_stride,
    OptionalAttr<I64Attr>:$rhs_block_stride
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$lhs_min_base `,` $rhs_sum_base (`block` $block_index^ `:` type($block_index))? attr-dict `:` type($lhs_min_base) `,` type($rhs_sum_base) `->` type($result)";
  let hasVerifier = 1;
}

def BlockFiveBitQhSourceOp : TCRVRVV_Op<"block_five_bit_qh_source",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV per-block five-bit qh 32-bit field source brick op";
  let description = [{
    Records one bounded generic plugin-local RVV per-block qh-field SOURCE brick as
    ONE typed dataflow op: it names the 32-bit qh field (the ggml q5_0 `uint8 qh[4]`
    per-block header, read as TWO aligned 16-bit halves off the weight block base at
    qh_byte_offset / qh_byte_offset+2) as an operand-flow token the FIVE-bit product
    op tcrv_rvv.five_bit_offset_binary_x_i8_product consumes. This is the
    scalar-domain 5th-bit SOURCE of the ggml Q5_0 reconstruction, made a first-class
    operand edge so the 5th-bit provenance is operand/source-driven rather than a
    descriptor-baked byte offset on the product op: mutating this brick's qh_base
    operand or its qh_byte_offset attr changes the emitted `(xb + qhOffset)` address
    -> changes the emitted bytes (anti-bypass). The op consumes ONE imported runtime
    ABI weight block-base pointer (the qh field lives WITHIN the weight block, so the
    base is the SAME weight ABI base the nibble weight load names) plus an OPTIONAL
    block_index induction operand, and produces ONE scalar i32 result.

    The i32 result is a GATE-ONLY edge (mirrors tcrv_rvv.block_fp16_min_product's
    f32): it is never materialized standalone; the emitter RE-READS the two 16-bit
    qh halves from THIS brick's own qh_base + block_index + qh_byte_offset when it
    lowers the naming product op. For the bounded surface kind is limited to
    "block_five_bit_qh_source"; qh_byte_offset locates the qh header within each
    block base; block_stride is the per-block AoS weight stride (present only with
    block_index). Emission is determined by op identity + these explicit facts,
    never inferred from candidate-mirror metadata, route ids, ABI strings, artifact
    names, status fields, exact intrinsic spellings, or q8/q5 names.

    LOOP-CAPABLE per-block-source form (mirrors block_fp16_min_product): the OPTIONAL
    block_index operand toggles the per-block-source form; when present it is the
    induction variable (region argument 0) of the enclosing
    tcrv_rvv.typed_flat_block_dot_loop_body region and the qh header lives at
    `qh_base + block_index*block_stride (+ qh_byte_offset)`, with the imported ABI
    base still the loop-invariant block-0 pointer. When absent the single-block form
    keeps the imported ABI base as the block base and block_stride must not be set
    (fail-closed).
  }];

  let arguments = (ins
    AnyType:$qh_base,
    Optional<Index>:$block_index,
    StrAttr:$kind,
    OptionalAttr<I64Attr>:$qh_byte_offset,
    OptionalAttr<I64Attr>:$block_stride
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$qh_base (`block` $block_index^ `:` type($block_index))? attr-dict `:` "
    "type($qh_base) `->` type($result)";
  let hasVerifier = 1;
}

def BlockComputedScaleDequantOp : TCRVRVV_Op<"block_computed_scale_dequant",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV per-block computed-scale i32-sumi dequant fold op";
  let description = [{
    Records one bounded generic plugin-local RVV per-block dequant fold as ONE
    typed dataflow op: take a scalar i32 per-block partial sum `sumi` and a
    COMPUTED f32 per-block scale, convert the sumi i32 -> f32 and multiply into
    ONE f32 scalar `f32(sumi) * scale`. The computed scale is the SSA output of
    a preceding tcrv_rvv.block_fp16_scale_product (M-FLAT brick 1, the
    `f32(d_x) * f32(d_y)` per-block scale product); it is NOT an imported
    runtime ABI value. The op does NOT touch the integer quant payload, the
    reduce that produces sumi, the block loop, or the cross-block fp32
    accumulate (those are separate typed steps).

    This is the SECOND scalar-domain typed construction primitive and it
    dismantles wall 2 of the four-walls flat-block-dot analysis: today
    tcrv_rvv.dequantize hard-requires its $scale operand to be an imported
    RuntimeABIValueOp (float, DequantScaleValue role), so it cannot consume a
    per-iteration COMPUTED scale. This sibling op consumes the computed f32
    scale directly -- structurally enforced by requiring the computed_scale
    operand type to be f32 (an imported ABI scale carries the
    !tcrv_rvv.runtime_abi_value type, which fails the f32 check), which is
    exactly the wall-2 contrast. tcrv_rvv.dequantize's contract is untouched.

    Making the computed-scale sumi fold a first-class typed body op (I5) means
    emission is determined by op identity + the explicit dequant kind/relation,
    never inferred from candidate-mirror metadata, route ids, ABI strings,
    artifact names, status fields, exact intrinsic spellings, or q8/q4 names.
    The f32 range fully covers the sumi and scale domains, so the fold is
    byte-exact by construction. For the bounded surface kind is limited to
    "computed_scale_sumi_dequant" and dequant_relation to
    "scalar-i32-sumi-to-f32-computed-scale-f32".

    OPTIONAL Family-B min_term (q4_1 / q5_1): the ggml Family-B fold is `sumf +=
    (d_x*d_y)*sumi + m_x*s_y`, so the per-block dequant term carries a SECOND
    scalar product `m_x * s_y` (the output of a preceding
    tcrv_rvv.block_fp16_min_product) summed with the `f32(sumi) * scale` term. The
    OPTIONAL `min_term` operand carries that computed f32 min product; when present
    the op semantically is `f32(sumi)*scale + min_term`. When absent (q8_0 / q4_0 /
    q5_0, whose folds have NO min term) the op keeps its two-operand form
    byte-neutral. The min_term, when present, must be a COMPUTED f32 SSA value,
    same fail-closed contrast as computed_scale.
  }];

  let arguments = (ins
    AnyType:$sumi,
    AnyType:$computed_scale,
    Optional<AnyType>:$min_term,
    StrAttr:$kind,
    StrAttr:$dequant_relation
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$sumi `,` $computed_scale (`min` $min_term^ `:` type($min_term))? attr-dict `:` type($sumi) `,` type($computed_scale) `->` type($result)";
  let hasVerifier = 1;
}

def CrossBlockF32AccumulateOp : TCRVRVV_Op<"cross_block_f32_accumulate",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV f32 cross-block scalar accumulate fold op";
  let description = [{
    Records one bounded generic plugin-local RVV cross-block fp32 accumulate as
    ONE typed dataflow op: take the block-carried f32 accumulator `acc` and one
    f32 per-block `term` and fold `acc + term` into ONE f32 scalar result. The
    `term` operand is the f32 output of a preceding
    tcrv_rvv.block_computed_scale_dequant (M-FLAT brick 2, the
    `(float)sumi * scale` per-block term); the `acc` operand is the running
    fp32 sum carried across blocks. The op does NOT touch the integer quant
    payload, the per-block scale/sumi/dequant, the block loop, or the scalar
    store (those are separate typed steps).

    This is the THIRD scalar-domain typed construction primitive and it
    dismantles wall 3 of the four-walls flat-block-dot analysis: today the
    cross-block fold `sumf += (float)sumi * (d_x * d_y)` exists only as an
    inline opaque emitc.add in the monolithic block-dot lowering, and the two
    typed accumulators the dialect already carries -- tcrv_rvv.widening_accumulate
    and tcrv_rvv.deferred_accumulate -- are i32-INTEGER INTRA-STRIP accumulators (wrong
    dtype AND wrong scope: they accumulate widened i32 lanes inside one VL
    strip, not a scalar f32 across blocks). Neither can express this fold.
    Making the cross-block fp32 accumulate a first-class typed body op (I5)
    means emission is determined by op identity + the explicit accumulate_order
    fact, never inferred from candidate-mirror metadata, route ids, ABI
    strings, artifact names, status fields, exact intrinsic spellings, or q8/q4
    names.

    accumulate_order records that the caller folds in STRICT ASCENDING block
    order (block-carried), preserving fp non-associativity byte-exactly to
    match ggml's q8_0 accumulation. Both operands and the result are required
    to be scalar f32 (an i32 lane accumulator or a vector value is rejected
    fail-closed), which is exactly the wall-3 contrast against the i32
    intra-strip accumulators. The f32 fold is byte-exact by construction at the
    operation-spelling level. For the bounded surface kind is limited to
    "cross_block_f32_scalar_accumulate" and accumulate_order to
    "strict-ascending-block-carried".
  }];

  let arguments = (ins
    AnyType:$acc,
    AnyType:$term,
    StrAttr:$kind,
    StrAttr:$accumulate_order
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$acc `,` $term attr-dict `:` type($acc) `,` type($term) `->` type($result)";
  let hasVerifier = 1;
}

def TypedFlatBlockDotLoopBodyOp : TCRVRVV_Op<"typed_flat_block_dot_loop_body",
    [SingleBlock, TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV block-carried nb loop op for the flat block dot-product";
  let description = [{
    Records the flat block-quantized dot-product's OUTER block loop as ONE
    typed region-carrying dataflow op: the `nb = n / QK` block loop plus the
    SSA loop-carried f32 accumulator that folds one per-block term into a
    running scalar sum in STRICT ASCENDING block order, then stores the final
    sum through the output pointer. This is the FIRST typed op that puts a loop
    INTO the CORE typed body (every existing loop lives only in the emitc
    lowering; tcrv_rvv.with_vl is the only prior region op and it is NOT a
    loop). It consumes the same four runtime ABI operands as the monolithic
    block-dot ops -- the weight base pointer, the activation base pointer, the
    output pointer, and the runtime element count -- and owns one single-block
    region whose entry arguments are the `block_index` induction variable
    (index) and the loop-carried `acc` (f32). The region is terminated by
    tcrv_rvv.typed_flat_block_dot_loop_yield, which names the carried-out
    `acc_next`.

    Making the block loop + the loop-carried f32 accumulate a first-class typed
    body op (I5) means the emission is determined by op identity + the explicit
    scheduling facts, never inferred from candidate-mirror metadata, route ids,
    ABI strings, artifact names, status fields, exact intrinsic spellings, or
    q8/q4 names. The SSA loop-carried accumulator lowers byte-exactly to the
    monolithic block-dot's mutable `float sumf` form: emitc.for has no
    iter_args, so the carried-in `acc` block argument maps to a load of the
    emitc.variable lvalue at the top of the loop body and the carried-out
    `acc_next` maps to an emitc.assign back into it at the bottom, exactly as
    the SCFToEmitC iter_args lowering and the F3/F6 loop-carried scalars do.
    The seed is the literal `0.0f` (ggml's `float sumf = 0.0f;` is a hardcoded
    zero, not a caller-provided value, so the accumulator is seeded by the op's
    lowering rather than an init operand -- mirroring the block-dot ops, which
    also carry no init operand).

    The bounded scheduling facts (qk, the AoS block strides, the optional
    integer_core_lmul / multi_block_factor / strip_elision / fold_structure, the
    fold_model) are the same byte-exact shape knobs deriveBlockDotFacts reads for
    the monolithic body (the *how* -- LMUL / unroll / elision / fold-scheduling --
    never the *what*). fold_model fixes the ARITHMETIC fold tree (region-invariant);
    fold_structure is the orthogonal SCHEDULE knob for HOW that same pinned fold
    (testing/flat-block-dot-fp-fold-oracle.md §1) is issued -- "per-block" (default)
    folds each block's scalar term into the running f32 sum as it is produced;
    "deferred-ordered" (P2c, q8_0 only) batches multi_block_factor blocks, packs
    their integer sumi into one vector, strided-loads the fp16 scales, and issues
    the cross-block fold as a single seed-ordered vfredosum.vs -- bit-exact to the
    SAME §1 oracle by construction (vfredosum.vs reduces lane-ascending, seed-first
    = the serial left-fold). Both variants verify against the one §1 oracle.

    numerics_tier is a THIRD, ORTHOGONAL scheduling axis (distinct from fold_model
    = the *what*, and fold_structure = *how the STRICT fold is issued*): it selects
    WHICH numeric oracle governs the fp fold. "strict" (default; absent = strict,
    fail-closed I7) pins the [testing/flat-block-dot-fp-fold-oracle.md §1] fold
    (no-FMA, strict left-assoc, ordered, no d_x*d_y premultiply) -- BOTH per-block
    and deferred-ordered strict variants verify byte-exact against §1, and the paper
    headline number is ALWAYS strict. "relaxed" is the policy-gated reassociation
    variant of §5: it is admitted ONLY behind the `numerics.reassoc_ok`
    (kind=policy) capability fact (fail-closed: the fact absent => strict), verifies
    against the SEPARATE reassoc-tolerant oracle + a DECLARED ULP upper bound (NEVER
    against §1, NEVER a headline -- [K-5] fp gate = ULP-bound declaration). Under
    relaxed the deferred-ordered q8_0 body may premultiply the scales, fuse the
    per-block dequant into a vfmacc lane-wise accumulator, and defer the whole
    cross-block collapse to ONE unordered vfredusum.vs (a K-way accumulator that
    breaks the serial fold chain, §5). fold_structure then degrades to a sub-knob of
    the strict tier. For the current bounded surface numerics_tier is limited to
    "strict" or "relaxed".
    For the
    current bounded surface kind is limited to "typed_flat_block_dot_loop_body"
    and fold_model to "sumi_times_scales" (the q8_0 `(float)sumi * (d_x * d_y)`
    fold tree) or "left_assoc" (the q4_0 `((float)sumi * d_x) * d_y` fold tree,
    whose integer core is the asymmetric offset-binary packed-i4 x i8 product).
    The op does NOT own dtype/route authority; the per-block integer
    core, the per-block scale reconstruction, and the per-block dequant term are
    separate typed steps carried inside the region.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$n,
    StrAttr:$kind,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    StrAttr:$fold_model,
    OptionalAttr<StrAttr>:$integer_core_lmul,
    OptionalAttr<I64Attr>:$multi_block_factor,
    OptionalAttr<StrAttr>:$strip_elision,
    OptionalAttr<StrAttr>:$fold_structure,
    OptionalAttr<StrAttr>:$numerics_tier
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $weight_base `,` $activation_base `,` $output `,` $n
    attr-dict-with-keyword $body `:`
    type($weight_base) `,` type($activation_base) `,` type($output) `,`
    type($n)
  }];
  let hasVerifier = 1;
}

def TypedFlatBlockDotLoopYieldOp : TCRVRVV_Op<"typed_flat_block_dot_loop_yield",
    [Terminator,
     HasParent<"::tianchenrv::tcrv::rvv::TypedFlatBlockDotLoopBodyOp">]> {
  let summary =
      "terminator naming the carried-out f32 accumulator of the block loop op";
  let description = [{
    Terminates the tcrv_rvv.typed_flat_block_dot_loop_body region by naming the
    carried-out f32 accumulator `acc_next`. It makes the loop-carried dataflow
    explicit (the SSA value carried into the next block iteration), so the
    lowering maps `acc_next` to the emitc.assign back into the sumf
    emitc.variable lvalue with no implicit convention. The operand is required
    to be scalar f32 (the block-carried accumulator domain). It carries no
    compute, dtype, memory, or route semantics.
  }];

  let arguments = (ins AnyType:$acc_next);

  let assemblyFormat = "$acc_next attr-dict `:` type($acc_next)";
  let hasVerifier = 1;
}

def TypedSuperBlockBlockDotLoopBodyOp
    : TCRVRVV_Op<"typed_super_block_block_dot_loop_body",
    [SingleBlock, TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV block-carried nb loop op for the q4_K/q5_K SUPER-BLOCK "
      "block dot-product (DUAL accumulator)";
  let description = [{
    Records the q4_K/q5_K SUPER-BLOCK-quantized dot-product's OUTER super-block
    loop as ONE typed region-carrying dataflow op, the M-FLAT super-block
    scaffold sibling of tcrv_rvv.typed_flat_block_dot_loop_body. It is the FIRST
    typed loop op that carries a DUAL accumulator: the q4_K two-level fold splits
    the running result into a `sums` 8-lane fp32 VECTOR accumulator (the deferred
    positive `sums += d * (float)aux32` fold, BRICK 6) AND a `sumf` scalar fp32
    accumulator (the in-loop `sumf -= dmin * (float)sumi` MIN term, BRICK 4) --
    two DIFFERENT accumulator chains the flat single-accumulator loop op cannot
    express. The op consumes the same four runtime ABI operands as the flat loop
    op and the monolithic q4_K block dot -- the q4_K weight base pointer, the
    q8_K activation base pointer, the output pointer, and the runtime element
    count -- and owns one single-block region whose entry arguments are the
    `super_block_index` induction variable (index), the loop-carried `sums`
    8-lane fp32 vector accumulator (!tcrv_rvv.vector<f32, "m2">), and the
    loop-carried `sumf` scalar fp32 accumulator (f32). The region is terminated
    by tcrv_rvv.typed_super_block_block_dot_loop_yield, which names BOTH
    carried-out accumulators (`sums_next`, `sumf_next`).

    Making the super-block loop + the DUAL loop-carried accumulate a first-class
    typed body op (I5) means the emission is determined by op identity + the
    explicit scheduling facts, never inferred from candidate-mirror metadata,
    route ids, ABI strings, artifact names, status fields, exact intrinsic
    spellings, or q8/q4 names. The two SSA loop-carried accumulators lower
    byte-exactly to the monolithic q4_K block dot's mutable `float sums[8]` +
    `float sumf` form (the vector accumulator via a vfloat32m2 emitc.variable, the
    scalar via a float emitc.variable): emitc.for has no iter_args, so each
    carried-in block argument maps to a load of its emitc.variable at the top of
    the loop body and each carried-out yield operand maps to an emitc.assign back
    at the bottom, exactly as the flat loop op and the SCFToEmitC iter_args
    lowering do. Both seeds are the literal zero (ggml seeds `sums` to a
    vfmv_v_f_f32m2(0.0f) and `sumf = 0.0f`, hardcoded, not caller-provided, so the
    accumulators are seeded by the op's lowering rather than an init operand --
    mirroring the block-dot ops, which also carry no init operand).

    The bounded scheduling facts (qk == QK_K, the AoS super-block strides, the
    optional integer_core_lmul) are the same byte-exact shape knobs the monolithic
    q4_K body reads (the *how* -- the per-sub-block integer-MAC widening rung --
    never the *what*). fold_model fixes the ARITHMETIC fold tree
    (region-invariant) AND KEYS the accumulator ARITY: for the current bounded
    surface kind is limited to "typed_super_block_block_dot_loop_body" and
    fold_model to one of three -- "super_block_two_level_scale_min" (q4_K/q5_K)
    selects the DUAL accumulator (region (index, sums, sumf) + dual yield);
    "scales_times_sumi" (q6_K) selects a SINGLE-VECTOR accumulator (the no-min
    `sums += d*(float)aux32` positive fold ONLY, NO scalar MIN chain, so the
    region carries just (index, sums) and the yield names the `sums` vector alone,
    and the post-loop horizontal add collapses `sums` straight into `*s`);
    "scalar_scale_min" (q2_K) selects a SCALAR accumulator (the q2_K positive fold
    is a single per-super-block scalar `dall*isum`, not a deferred 8-lane vector,
    so the region carries just (index, sumf:f32) and the yield names the `sumf`
    scalar alone -- there is NO `sums` vector; the fold is
    `sumf += dall*isum - dmin*summs`, isum/summs being the SCALAR integer states of
    the q2_K integer-core brick).
    "super_block_two_level_scale_min" is (the q4_K/q5_K two-level `sums += d*aux32`
    positive fold PLUS the `sumf -= dmin*Σ(mins*bsums)` MIN term). The op does NOT
    own dtype/route authority; the per-super-block nibble unpack, the 6-bit
    scale/min bit-dance, the per-sub-block scaled i32 dot, the MIN term, and the
    deferred fp32 positive fold are separate typed bricks carried inside the
    region, and the post-loop horizontal fold is a separate post-loop brick.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$n,
    StrAttr:$kind,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    StrAttr:$fold_model,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $weight_base `,` $activation_base `,` $output `,` $n
    attr-dict-with-keyword $body `:`
    type($weight_base) `,` type($activation_base) `,` type($output) `,`
    type($n)
  }];
  let hasVerifier = 1;
}

def TypedSuperBlockBlockDotLoopYieldOp
    : TCRVRVV_Op<"typed_super_block_block_dot_loop_yield",
    [Terminator,
     HasParent<"::tianchenrv::tcrv::rvv::TypedSuperBlockBlockDotLoopBodyOp">]> {
  let summary =
      "terminator naming the carried-out super-block accumulator(s): the first "
      "accumulator (8-lane fp32 `sums` vector for q4_K/q5_K/q6_K, OR a scalar "
      "`sumf` f32 for q2_K) always, plus the OPTIONAL `sumf` scalar second operand "
      "(present = dual q4_K/q5_K, absent = single-accumulator q6_K/q2_K)";
  let description = [{
    Terminates the tcrv_rvv.typed_super_block_block_dot_loop_body region by naming
    the carried-out accumulator(s). The FIRST carried-out operand (`sums_next`) is
    ALWAYS present; its type is keyed by the parent loop op's fold_model: the
    8-lane fp32 VECTOR accumulator (!tcrv_rvv.vector<f32, "m2">, the deferred
    positive-fold chain) for the q4_K/q5_K/q6_K vector folds, OR a SCALAR fp32
    accumulator (f32, the q2_K scalar `sumf` chain) for the q2_K scalar fold. The
    `sumf_next` scalar fp32 accumulator (f32, the MIN-term chain) is the OPTIONAL
    SECOND operand and its presence is keyed by fold_model:
    "super_block_two_level_scale_min" (q4_K/q5_K) carries BOTH (the DUAL yield);
    "scales_times_sumi" (q6_K no-min) carries the `sums` VECTOR ONLY (the SINGLE
    yield -- the no-min fold has no scalar MIN chain); "scalar_scale_min" (q2_K)
    carries the `sumf` SCALAR ONLY (the SINGLE yield -- q2_K's whole fold is one
    scalar accumulator, so there is no separate MIN-term operand). The lowering
    maps each carried-out value to the emitc.assign back into its emitc.variable
    lvalue with no implicit convention. The first operand is required to be either
    the 8-lane fp32 vector accumulator OR a scalar f32; when present the second is
    required to be scalar f32 (a swapped vector-second pair is rejected
    fail-closed). The fold_model-keyed arity match (which fold carries which
    first-operand shape and whether a second sumf operand is allowed) is enforced
    by the parent loop body verifier. It carries no compute, dtype, memory, or
    route semantics.
  }];

  let arguments = (ins AnyType:$sums_next, Optional<AnyType>:$sumf_next);

  let assemblyFormat =
      "$sums_next (`,` $sumf_next^)? attr-dict `:` type($sums_next) "
      "(`,` type($sumf_next)^)?";
  let hasVerifier = 1;
}

def RepackLaneWiseQ4Q8DotOp
    : TCRVRVV_Op<"repack_lane_wise_q4_x_i8_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV q4_0 16x1-REPACKED per-block LANE-WISE nibble dot "
      "(vwmacc lo/hi + lo/hi vwadd combine) integer-core op";
  let description = [{
    Records ONE repacked contraction-BLOCK's q4_0 16x1-REPACKED per-block
    LANE-WISE integer dot -- the integer CORE of the ggml
    `ggml_gemv_q4_0_16x1_q8_0` per-column-group inner loop -- as ONE bounded
    generic plugin-local typed RVV dataflow step. It is the M-FLAT REPACK
    scaffold's BRICK sibling of the flat
    tcrv_rvv.packed_i4_offset_binary_x_i8_product (which is a single-shot
    whole-half-block product + vwredsum) and the super-block
    tcrv_rvv.q4_k_scaled_dot (which is a per-sub-block scaled aux32 MAC): the
    repacked block-as-lane dot accumulates LANE-WISE via a NIBBLE-STEP inner loop
    of vwmacc (NO cross-lane vredsum, NO horizontal fold), a structure NEITHER
    product brick can express.

    The op is carried inside the tcrv_rvv.typed_repack_gemv_loop_body region and
    consumes the repacked q4_0x16 weight base pointer, the plain q8_0 activation
    base pointer, the active !tcrv_rvv.vl token, and the loop's `block_index`
    induction variable (region argument 0). It produces the per-strip 16-lane
    combined `sumi` values (the lo/hi vwadd_vv results) as a VARIADIC result group
    -- one i32 !tcrv_rvv.vector per disjoint strip (numHalves == weight_interleave
    / half_lanes: ONE for the 16-lane one-strip VLEN=256/RVV0.7 form, TWO for the
    two-8-lane-halves VLEN=128 form). The per-strip dual-fp16 scale FOLD siblings
    (tcrv_rvv.repack_dual_fp16_scale_fold, one per strip) each consume one result.

    The block-as-lane integer core STRUCTURE is first-class in the typed body: the
    emission mirrors emitRepackGemvQ4_0Q8_0's integer part LINE-FOR-LINE (shared
    VERBATIM through the SAME leaf), so byte-exactness is by construction. Per
    strip the i16 lo/hi accumulators are seeded (vmv_v_x), then a nibble-step loop
    of qk/2 iterations does one disjoint repacked sub-load
    (vle8 at `block_base + weight_quant_byte_offset + i*weight_interleave`), the
    plain sign-extension nibble decode (vsll/vsra low, vsra high -- the repacked
    nibbles carry the ^0x88 offset-binary bias already so there is NO vxor), the
    two SCALAR activation quant reads (the single q8_0 column at
    `act_base + activation_quant_byte_offset + i` low / `+ nibbleBytes + i` high),
    and the two lane-wise vwmacc.vx (low against the low quant, high against the
    high quant) into the i16 lo/hi accumulators; then the ONE lo/hi combine
    vwadd_vv into the i32 result. The integer add is associative/order-free so the
    per-lane accumulation is byte-exact.

    The `block_index` operand is the ANTI-BYPASS tie: it MUST be the loop induction
    variable (region argument 0), so the per-block base is `base + block_index*
    stride (+ byte_offset)` -- the emit provably addresses block ib, not the
    loop-invariant block 0 (fail-closed otherwise, exactly the flat/super-block
    per-block-source anti-bypass). The `weight_quant_byte_offset` /
    `activation_quant_byte_offset` locate the repacked nibble bytes and the q8_0
    quant bytes WITHIN their blocks (32 / 2 for block_q4_0x16 / block_q8_0); the
    per-block strides + qk + weight_interleave + the resource-aware strip width are
    the enclosing loop op's block-format facts. The bounded surface kind is limited
    to "repack_lane_wise_q4_x_i8_dot". The optional `integer_core_lmul` is the
    bounded resource knob selecting the widening-chain base LMUL ("mf2" default RVV
    1.0 fractional; "m1" the RVV0.7 whole-LMUL chain) -- the *how*, never the
    *what*. Route ids, helper names, ABI strings, artifact names, status fields,
    exact intrinsic spellings, operand mirror strings, and q8/q4 names must not
    infer or choose this repacked lane-wise dot's semantics.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$vl,
    Index:$block_index,
    StrAttr:$kind,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $vl `block` $block_index `:` "
    "type($block_index) attr-dict `:` type($weight_base) `,` "
    "type($activation_base) `,` type($vl) `->` type($results)";
  let hasVerifier = 1;
}

def TypedRepackGemvLoopBodyOp
    : TCRVRVV_Op<"typed_repack_gemv_loop_body",
    [SingleBlock, TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV column-group-wrapped nb loop op for the q4_0 16x1-"
      "REPACKED GEMV (per-strip lane-wise VECTOR f32 accumulator)";
  let description = [{
    Records the q4_0 16x1-REPACKED GEMV's inner contraction-BLOCK loop as ONE
    typed region-carrying dataflow op -- the M-FLAT REPACK scaffold sibling of
    tcrv_rvv.typed_flat_block_dot_loop_body and
    tcrv_rvv.typed_super_block_block_dot_loop_body. It is the FIRST typed loop op
    whose loop-carried accumulator is a per-strip LANE-WISE f32 VECTOR: the
    repacked block-as-lane dot accumulates LANE-WISE via vwmacc, so the per-block
    reduction wall is gone (NO cross-lane vredsum, NO horizontal fold, NO scalar
    boundary) and the whole 16-lane strip is written STRAIGHT to s + x*16 with a
    lane-wise vse32. This is a structure NEITHER the flat single-f32-scalar loop
    op NOR the super-block dual/single accumulator loop op can express (both carry
    a per-super-block SCALAR / horizontally-folded vector reduced to *s).

    The op consumes FIVE runtime ABI value operands -- the repacked q4_0x16 weight
    base pointer, the plain q8_0 activation base pointer, the fp32 output pointer,
    the runtime contraction element count (n), and the runtime weight-column count
    (nc) -- and owns one single-block region whose entry arguments are the
    `block_index` induction variable (index) FOLLOWED by the numHalves loop-carried
    per-strip f32 VECTOR accumulators `acc` (numHalves + 1 entry arguments;
    numHalves == weight_interleave / half_lanes: ONE 16-lane strip
    !tcrv_rvv.vector<f32, "m2"> / <f32, "m4"> at VLEN=256 / RVV0.7, or TWO 8-lane
    strips at VLEN=128). The region is terminated by
    tcrv_rvv.typed_repack_gemv_loop_yield, which names the carried-out `acc_next`
    vectors (numHalves entries). The lowering wraps the region in the OUTER
    weight-column-GROUP loop (for x in nc/weight_interleave), seeds each per-strip
    vector accumulator with vfmv_v_f(0.0f) per group, maps each carried-IN `acc`
    block argument to a LOAD of its accumulator's emitc.variable lvalue at the top
    of the block loop and each carried-OUT `acc_next` to an emitc.assign back at the
    bottom (emitc.for has no iter_args, exactly the flat/super-block loop lowering
    and the SCFToEmitC iter_args convention), and stores each final strip through
    s + x*weight_interleave (+ strip*half_lanes) with a lane-wise vse32.

    Making the repacked block loop + the per-strip VECTOR accumulate a first-class
    typed body op (I5) means the emission is determined by op identity + the
    explicit scheduling facts, never inferred from candidate-mirror metadata,
    route ids, ABI strings, artifact names, status fields, exact intrinsic
    spellings, or q8/q4 names. The bounded scheduling facts (qk, the repacked AoS
    block strides / byte offsets, weight_interleave, the resource-aware half_lanes
    strip width, the optional integer_core_lmul) are the SAME byte-exact shape
    knobs the monolithic emitRepackGemvQ4_0Q8_0 reads (the *how* -- LMUL / strip
    width -- never the *what*); fold_model "lane_wise_vector_scale" fixes the
    per-strip fp32 fold tree (vfwmul/vfcvt/vfmacc). The op does NOT own dtype/route
    authority; the per-block lane-wise integer product (the nibble decode + vwmacc
    lo/hi + lo/hi combine) and the per-block dual-fp16 scale fold are separate
    typed steps to be carried inside the region (later M-FLAT repack milestones).
    This SCAFFOLD milestone pins the loop + per-strip VECTOR accumulator SKELETON
    byte-exact to the monolithic one-strip repacked GEVM (the CORE integer product
    + scale fold are a pure loop-carried stub, exactly as the flat loop scaffold
    step 1 stubbed its fold). For the current bounded surface kind is limited to
    "typed_repack_gemv_loop_body", fold_model to "lane_wise_vector_scale", and
    scale_model to "dual-fp16-per-block-d_x.d_y".
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$column_count,
    StrAttr:$kind,
    StrAttr:$scale_model,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$activation_block_stride,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    I64Attr:$weight_interleave,
    I64Attr:$half_lanes,
    StrAttr:$fold_model,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $weight_base `,` $activation_base `,` $output `,` $element_count `,`
    $column_count attr-dict-with-keyword $body `:`
    type($weight_base) `,` type($activation_base) `,` type($output) `,`
    type($element_count) `,` type($column_count)
  }];
  let hasVerifier = 1;
}

def TypedRepackGemvLoopYieldOp
    : TCRVRVV_Op<"typed_repack_gemv_loop_yield",
    [Terminator,
     HasParent<"::tianchenrv::tcrv::rvv::TypedRepackGemvLoopBodyOp">]> {
  let summary =
      "terminator naming the carried-out per-strip f32 VECTOR accumulators of the "
      "repacked GEVM block loop op";
  let description = [{
    Terminates the tcrv_rvv.typed_repack_gemv_loop_body region by naming the
    carried-out per-strip f32 VECTOR accumulators `acc_next` as a VARIADIC operand
    group -- one !tcrv_rvv.vector<f32, "m2"> (mf2 core) or <f32, "m4"> (m1
    whole-LMUL core) per disjoint strip (numHalves entries: ONE for the 16-lane
    one-strip VLEN=256/RVV0.7 form, TWO for the two-8-lane-halves VLEN=128 form).
    It makes the loop-carried dataflow explicit (the SSA values carried into the
    next block iteration), so the lowering maps each `acc_next` to the emitc.assign
    back into its per-strip accumulator emitc.variable lvalue with no implicit
    convention. Every operand is required to be an f32 VECTOR accumulator (a scalar
    f32 or an i32 lane accumulator is rejected fail-closed -- this is the lane-wise
    VECTOR contrast against the flat loop op's scalar yield). It carries no compute,
    dtype, memory, or route semantics.
  }];

  let arguments = (ins Variadic<AnyType>:$acc_next);

  let assemblyFormat = "$acc_next attr-dict `:` type($acc_next)";
  let hasVerifier = 1;
}

def RepackDualFp16ScaleFoldOp
    : TCRVRVV_Op<"repack_dual_fp16_scale_fold",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV q4_0 16x1-REPACKED per-block per-strip dual-fp16 scale "
      "FOLD (vle16 weight scale / _Float16 act scale / vfwmul / vfcvt / vfmacc) "
      "op consuming the lane-wise integer sumi";
  let description = [{
    Records ONE repacked contraction-BLOCK's q4_0 16x1-REPACKED per-strip
    dual-fp16 scale FOLD -- the second half of the ggml `ggml_gemv_q4_0_16x1_q8_0`
    per-column-group inner loop, the step that folds the lane-wise integer dot
    result into the running per-strip f32 accumulator -- as ONE bounded generic
    plugin-local typed RVV dataflow step. It is the FOLD sibling of the integer
    CORE brick tcrv_rvv.repack_lane_wise_q4_x_i8_dot (the two together are the
    complete per-block repacked GEVM inner body): the integer brick produces the
    per-strip i32 `sumi`, this brick consumes it plus the loop-carried per-strip
    f32 accumulator `acc` and produces the folded-out `acc_next`.

    The op is carried inside the tcrv_rvv.typed_repack_gemv_loop_body region and
    consumes the repacked q4_0x16 weight base pointer (for the per-block per-strip
    fp16 weight scale vle16), the plain q8_0 activation base pointer (for the
    single per-block _Float16 activation scale), the integer brick's per-strip i32
    `sumi`, the loop-carried per-strip f32 accumulator `acc`, the active
    !tcrv_rvv.vl token, and the loop's `block_index` induction variable (region
    argument 0). It produces ONE f32 !tcrv_rvv.vector result -- the folded-out
    per-strip accumulator `acc_next` the loop yield names.

    The per-strip fold STRUCTURE is first-class in the typed body: the emission
    mirrors emitRepackGemvQ4_0Q8_0's scale-fold part LINE-FOR-LINE (shared VERBATIM
    through the SAME leaf), so byte-exactness is by construction. Per strip the
    per-block fp16 weight scale is loaded (vle16 at `block_base +
    weight_scale_byte_offset (+ strip*half*2)`), the single per-block activation
    scale is read once (`*(const _Float16 *)(act_base + activation_scale_byte_
    offset)`, broadcast across strips), then d = vfwmul(weight_scale, act_scale)
    widens the dual-fp16 product to f32, the integer sumi is converted with
    vfcvt_f_x_v, and acc_next = vfmacc(acc, sumi_f, d) folds it in.

    The `block_index` operand is the ANTI-BYPASS tie: it MUST be the loop induction
    variable (region argument 0), so the per-block scale base is `base +
    block_index*stride (+ byte_offset)` -- the emit provably addresses block ib,
    not the loop-invariant block 0 (fail-closed otherwise, exactly the integer
    brick's per-block-source anti-bypass). The `weight_scale_byte_offset` /
    `activation_scale_byte_offset` locate the fp16 scale bytes WITHIN their blocks
    (0 / 0 for block_q4_0x16 / block_q8_0: the scale `d` field leads each block);
    the per-block strides + the resource-aware strip width are the enclosing loop
    op's block-format facts. The bounded surface kind is limited to
    "repack_dual_fp16_scale_fold". The optional `integer_core_lmul` is the bounded
    resource knob selecting the widening-chain base LMUL ("mf2" default RVV1.0
    fractional f32m2 fold; "m1" the RVV0.7 whole-LMUL f32m4 fold) -- the *how*,
    never the *what*. Route ids, helper names, ABI strings, artifact names, status
    fields, exact intrinsic spellings, operand mirror strings, and q8/q4 names must
    not infer or choose this repacked per-strip fold's semantics.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$sumi,
    AnyType:$acc,
    AnyType:$vl,
    Index:$block_index,
    StrAttr:$kind,
    I64Attr:$weight_scale_byte_offset,
    I64Attr:$activation_scale_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let results = (outs AnyType:$acc_next);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $sumi `,` $acc `,` $vl `block` "
    "$block_index `:` type($block_index) attr-dict `:` type($weight_base) `,` "
    "type($activation_base) `,` type($sumi) `,` type($acc) `,` type($vl) `->` "
    "type($acc_next)";
  let hasVerifier = 1;
}

def RepackGemmLaneWiseQ4Q8DotOp
    : TCRVRVV_Op<"repack_gemm_lane_wise_q4_x_i8_dot",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV q4_0 16x1-REPACKED GEMM per-block ONE-STRIP N-COLUMN "
      "LANE-WISE nibble dot (vwmacc lo/hi + lo/hi vwadd combine) integer-core op";
  let description = [{
    Records ONE repacked contraction-BLOCK's q4_0 16x1-REPACKED per-block
    LANE-WISE integer dot for the GEMM (prefill) -- the integer CORE of the ggml
    `ggml_gemm_q4_0_16x1_q8_0` per-(strip, column-pass) inner block loop -- as ONE
    bounded generic plugin-local typed RVV dataflow step. It is the GEMM sibling of
    the GEVM tcrv_rvv.repack_lane_wise_q4_x_i8_dot: where the GEVM core loops the
    numHalves disjoint strips over a SINGLE broadcast activation column (numHalves
    per-strip sumi), THIS GEMM core reads ONE strip (a RUNTIME `roff` strip row
    offset) across the columnsPerPass INTERLEAVED activation columns of a
    block_q8_0x4 group (columnsPerPass per-column sumi). This 1-strip-runtime x
    N-column TRANSPOSE of the GEVM's numHalves-strip x 1-column nesting is a
    structure the GEVM core brick cannot express (the strip is a runtime loop, the
    activation is a 4-way interleave, and the per-column accumulators fold one
    column per pass).

    The op is carried inside the tcrv_rvv.typed_repack_gemm_loop_body region and
    consumes the repacked q4_0x16 weight base pointer, the interleaved q8_0x4
    activation base pointer, the active !tcrv_rvv.vl token, the loop's `block_index`
    induction variable (region argument 0), and the `strip_row_offset` runtime
    strip offset (region argument 1, the enclosing strip loop's h*half_lanes). It
    produces the columnsPerPass per-column combined `sumi` values (the lo/hi
    vwadd_vv results) as a VARIADIC result group -- one i32 !tcrv_rvv.vector per
    interleaved activation column folded in this pass (columnsPerPass == the fold
    granularity: FOUR for the RVV1.0 fractional mf2 chain that holds all 4 columns
    at once, ONE for the RVV0.7 whole-LMUL chain that folds one column per pass to
    stay under the 32-vreg spill wall). The per-column dual-fp16 scale FOLD siblings
    (tcrv_rvv.repack_gemm_dual_fp16_scale_fold, one per column) each consume one
    result.

    The block-as-lane integer core STRUCTURE is first-class in the typed body: the
    emission mirrors emitRepackGemmQ4_0Q8_0's integer part LINE-FOR-LINE (shared
    VERBATIM through the SAME leaf emitRepackGemmQ4LaneWiseIntegerCore), so
    byte-exactness is by construction. Per column the i16 lo/hi accumulators are
    seeded (vmv_v_x), then a nibble-step loop of qk/2 iterations does one disjoint
    repacked sub-load (vle8 at `block_base + weight_quant_byte_offset + i*
    weight_interleave + roff` -- the runtime strip row offset selects the strip),
    the plain sign-extension nibble decode (vsll/vsra, NO vxor -- the repacked
    nibbles carry the ^0x88 bias), the two SCALAR interleaved activation quant reads
    (the column-c q8_0x4 quant at `act_base + activation_quant_byte_offset +
    i*activation_interleave + c` low / `+ activation_interleave*nibbleBytes` high),
    and the two lane-wise vwmacc.vx into the per-column i16 lo/hi accumulators; then
    the ONE lo/hi combine vwadd_vv into the per-column i32 result. The integer add
    is associative/order-free so the per-lane accumulation is byte-exact.

    The `block_index` operand is the ANTI-BYPASS tie: it MUST be the loop induction
    variable (region argument 0), so the per-block base is `base + block_index*
    stride (+ byte_offset)` -- the emit provably addresses block ib, not the
    loop-invariant block 0 (fail-closed otherwise). The `strip_row_offset` operand
    is the SECOND anti-bypass tie: it MUST be the region's strip offset argument
    (region argument 1), so the emit provably addresses the runtime strip, not a
    hardcoded strip 0. The `weight_quant_byte_offset` / `activation_quant_byte_
    offset` locate the repacked nibble bytes and the q8_0x4 quant bytes WITHIN their
    blocks (32 / 8 for block_q4_0x16 / block_q8_0x4); the per-block strides + qk +
    interleaves + the resource-aware strip width are the enclosing loop op's
    block-format facts. The bounded surface kind is limited to
    "repack_gemm_lane_wise_q4_x_i8_dot". The optional `integer_core_lmul` is the
    bounded resource knob selecting the widening-chain base LMUL ("mf2" default RVV
    1.0 fractional; "m1" the RVV0.7 whole-LMUL chain) -- the *how*, never the
    *what*. Route ids, helper names, ABI strings, artifact names, status fields,
    exact intrinsic spellings, operand mirror strings, and q8/q4 names must not
    infer or choose this repacked lane-wise dot's semantics.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$vl,
    Index:$block_index,
    Index:$strip_row_offset,
    StrAttr:$kind,
    I64Attr:$weight_quant_byte_offset,
    I64Attr:$activation_quant_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $vl `block` $block_index `strip` "
    "$strip_row_offset `:` type($block_index) `,` type($strip_row_offset) "
    "attr-dict `:` type($weight_base) `,` type($activation_base) `,` type($vl) "
    "`->` type($results)";
  let hasVerifier = 1;
}

def RepackGemmDualFp16ScaleFoldOp
    : TCRVRVV_Op<"repack_gemm_dual_fp16_scale_fold",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV q4_0 16x1-REPACKED GEMM per-block per-COLUMN dual-fp16 "
      "scale FOLD (vle16 weight scale / _Float16 act scale / vfwmul / vfcvt / "
      "vfmacc) op consuming the lane-wise integer sumi";
  let description = [{
    Records ONE repacked contraction-BLOCK's q4_0 16x1-REPACKED per-COLUMN
    dual-fp16 scale FOLD for the GEMM -- the second half of the ggml
    `ggml_gemm_q4_0_16x1_q8_0` per-(strip, column-pass) inner loop, the step that
    folds one interleaved column's lane-wise integer dot result into that column's
    running per-strip f32 accumulator -- as ONE bounded generic plugin-local typed
    RVV dataflow step. It is the FOLD sibling of the integer CORE brick
    tcrv_rvv.repack_gemm_lane_wise_q4_x_i8_dot (the two together are the complete
    per-block repacked GEMM inner body): the integer brick produces the
    columnsPerPass per-column i32 `sumi`, this brick consumes ONE column's `sumi`
    plus that column's loop-carried per-strip f32 accumulator `acc` and produces the
    folded-out `acc_next`.

    The op is carried inside the tcrv_rvv.typed_repack_gemm_loop_body region and
    consumes the repacked q4_0x16 weight base pointer (for the per-block per-strip
    fp16 weight scale vle16 at `&bl.d[roff]`), the interleaved q8_0x4 activation
    base pointer (for the per-column _Float16 activation scale `&al.d[c]`), the
    integer brick's per-column i32 `sumi`, the loop-carried per-column-per-strip f32
    accumulator `acc`, the active !tcrv_rvv.vl token, the loop's `block_index`
    induction variable (region argument 0), and the `strip_row_offset` runtime strip
    offset (region argument 1). It produces ONE f32 !tcrv_rvv.vector result -- the
    folded-out accumulator `acc_next` the loop yield names.

    The per-column fold STRUCTURE is first-class in the typed body: the emission
    mirrors emitRepackGemmQ4_0Q8_0's scale-fold part LINE-FOR-LINE (shared VERBATIM
    through the SAME leaf emitRepackGemmDualFp16ScaleFold), so byte-exactness is by
    construction. Per strip the per-block fp16 weight scales are loaded ONCE (vle16
    at `block_base + weight_scale_byte_offset + roff*2`), then per column the single
    per-block activation scale is read (`*(const _Float16 *)&al.d[c]`), d =
    vfwmul(weight_scale, act_scale) widens the dual-fp16 product to f32, the integer
    sumi is converted with vfcvt_f_x_v, and acc_next = vfmacc(acc, sumi_f, d) folds
    it in.

    The `block_index` operand is the ANTI-BYPASS tie: it MUST be the loop induction
    variable (region argument 0). The `strip_row_offset` operand is the SECOND
    anti-bypass tie: it MUST be the region's strip offset argument (region argument
    1), so the per-strip fp16 weight scale load provably addresses the runtime strip.
    The `weight_scale_byte_offset` / `activation_scale_byte_offset` locate the fp16
    scale bytes WITHIN their blocks (0 / 0 for block_q4_0x16 / block_q8_0x4: the
    scale `d` field leads each block). The bounded surface kind is limited to
    "repack_gemm_dual_fp16_scale_fold". The optional `integer_core_lmul` is the
    bounded resource knob selecting the widening-chain base LMUL ("mf2" default
    RVV1.0 fractional f32m2 fold; "m1" the RVV0.7 whole-LMUL f32m4 fold) -- the
    *how*, never the *what*. Route ids, helper names, ABI strings, artifact names,
    status fields, exact intrinsic spellings, operand mirror strings, and q8/q4
    names must not infer or choose this repacked per-column fold's semantics.
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$sumi,
    AnyType:$acc,
    AnyType:$vl,
    Index:$block_index,
    Index:$strip_row_offset,
    StrAttr:$kind,
    I64Attr:$weight_scale_byte_offset,
    I64Attr:$activation_scale_byte_offset,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let results = (outs AnyType:$acc_next);

  let assemblyFormat =
    "$weight_base `,` $activation_base `,` $sumi `,` $acc `,` $vl `block` "
    "$block_index `strip` $strip_row_offset `:` type($block_index) `,` "
    "type($strip_row_offset) attr-dict `:` type($weight_base) `,` "
    "type($activation_base) `,` type($sumi) `,` type($acc) `,` type($vl) `->` "
    "type($acc_next)";
  let hasVerifier = 1;
}

def TypedRepackGemmLoopBodyOp
    : TCRVRVV_Op<"typed_repack_gemm_loop_body",
    [SingleBlock, TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV row-group/column-group/strip/pass-wrapped nb loop op for "
      "the q4_0 16x1-REPACKED GEMM (per-column per-strip lane-wise f32 VECTOR "
      "accumulators)";
  let description = [{
    Records the q4_0 16x1-REPACKED GEMM's inner contraction-BLOCK loop as ONE typed
    region-carrying dataflow op -- the M-FLAT REPACK scaffold GEMM sibling of
    tcrv_rvv.typed_repack_gemv_loop_body. Where the GEVM loop op internalizes ONLY
    the weight-column-group (N) loop over a single broadcast activation column, this
    GEMM loop op internalizes the FULL ggml `ggml_gemm_q4_0_16x1_q8_0` nest: the
    activation-ROW-group (M-tiling) loop over nr/activation_interleave, the
    weight-column-group (N) loop over nc/weight_interleave, the RUNTIME strip loop
    over numHalves (weight_interleave/half_lanes disjoint e16m1 strips, with a
    runtime `roff = h*half_lanes` strip row offset), and a compile-time column PASS
    loop that folds columnsPerPass of the activation_interleave interleaved columns
    per pass. The region is the SINGLE innermost contraction-block loop; the lowering
    wraps ALL FOUR outer loops around it (the "transpose nesting" contrast against
    the GEVM's numHalves-strip x 1-column body).

    The op consumes SEVEN runtime ABI value operands -- the repacked q4_0x16 weight
    base pointer, the interleaved q8_0x4 activation base pointer, the fp32 output
    pointer, the runtime contraction element count (n), the runtime activation-row
    count (nr), the runtime weight-column count (nc), and the runtime output row
    stride (bs, in floats) -- and owns one single-block region whose entry arguments
    are the `block_index` induction variable (index) FOLLOWED by the runtime
    `strip_row_offset` (index) FOLLOWED by the columnsPerPass loop-carried
    per-column-per-strip f32 VECTOR accumulators `acc` (columnsPerPass + 2 entry
    arguments; columnsPerPass == activation_interleave for the RVV1.0 fractional mf2
    core, 1 for the RVV0.7 whole-LMUL m1 core). The region is terminated by
    tcrv_rvv.typed_repack_gemm_loop_yield, which names the carried-out `acc_next`
    vectors (columnsPerPass entries). The lowering wraps the region in the outer
    row-group / column-group / runtime strip / compile-time column-pass loops, seeds
    each per-column vector accumulator with vfmv_v_f(0.0f) per pass, maps each
    carried-IN `acc` block argument to a LOAD of its accumulator's emitc.variable
    lvalue at the top of the block loop and each carried-OUT `acc_next` to an
    emitc.assign back at the bottom (emitc.for has no iter_args, exactly the
    flat/super-block/GEVM loop lowering convention), and stores each final column
    through s + (y*activation_interleave + c)*bs + x*weight_interleave + roff with a
    lane-wise vse32.

    Making the repacked GEMM block loop + the per-column per-strip VECTOR accumulate
    a first-class typed body op (I5) means the emission is determined by op identity
    + the explicit scheduling facts, never inferred from candidate-mirror metadata,
    route ids, ABI strings, artifact names, status fields, exact intrinsic
    spellings, or q8/q4 names. The bounded scheduling facts (qk, the repacked AoS
    block strides / byte offsets, weight_interleave, activation_interleave, the
    resource-aware half_lanes strip width, the optional integer_core_lmul) are the
    SAME byte-exact shape knobs the monolithic emitRepackGemmQ4_0Q8_0 reads (the
    *how* -- LMUL / strip width / spill-avoiding columnsPerPass -- never the *what*);
    fold_model "lane_wise_vector_scale" fixes the per-column fp32 fold tree
    (vfwmul/vfcvt/vfmacc). The per-block lane-wise integer product (the nibble decode
    + vwmacc lo/hi + lo/hi combine) and the per-column dual-fp16 scale fold are
    separate typed steps carried inside the region (the two decomposed GEMM bricks).
    For the current bounded surface kind is limited to "typed_repack_gemm_loop_body",
    fold_model to "lane_wise_vector_scale", and scale_model to
    "dual-fp16-per-block-d_x.d_y".
  }];

  let arguments = (ins
    AnyType:$weight_base,
    AnyType:$activation_base,
    AnyType:$output,
    AnyType:$element_count,
    AnyType:$row_count,
    AnyType:$column_count,
    AnyType:$output_row_stride,
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
    StrAttr:$fold_model,
    OptionalAttr<StrAttr>:$integer_core_lmul
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $weight_base `,` $activation_base `,` $output `,` $element_count `,`
    $row_count `,` $column_count `,` $output_row_stride
    attr-dict-with-keyword $body `:`
    type($weight_base) `,` type($activation_base) `,` type($output) `,`
    type($element_count) `,` type($row_count) `,` type($column_count) `,`
    type($output_row_stride)
  }];
  let hasVerifier = 1;
}

def TypedRepackGemmLoopYieldOp
    : TCRVRVV_Op<"typed_repack_gemm_loop_yield",
    [Terminator,
     HasParent<"::tianchenrv::tcrv::rvv::TypedRepackGemmLoopBodyOp">]> {
  let summary =
      "terminator naming the carried-out per-column per-strip f32 VECTOR "
      "accumulators of the repacked GEMM block loop op";
  let description = [{
    Terminates the tcrv_rvv.typed_repack_gemm_loop_body region by naming the
    carried-out per-column per-strip f32 VECTOR accumulators `acc_next` as a
    VARIADIC operand group -- one !tcrv_rvv.vector<f32, "m2"> (mf2 core) or
    <f32, "m4"> (m1 whole-LMUL core) per interleaved column folded in the pass
    (columnsPerPass entries: FOUR for the RVV1.0 fractional mf2 all-columns pass,
    ONE for the RVV0.7 whole-LMUL one-column-per-pass form). It makes the
    loop-carried dataflow explicit (the SSA values carried into the next block
    iteration), so the lowering maps each `acc_next` to the emitc.assign back into
    its per-column accumulator emitc.variable lvalue with no implicit convention.
    Every operand is required to be an f32 VECTOR accumulator (a scalar f32 or an
    i32 lane accumulator is rejected fail-closed). It carries no compute, dtype,
    memory, or route semantics.
  }];

  let arguments = (ins Variadic<AnyType>:$acc_next);

  let assemblyFormat = "$acc_next attr-dict `:` type($acc_next)";
  let hasVerifier = 1;
}

def TypedVectorLane0ToScalarExtractOp :
    TCRVRVV_Op<"typed_vector_lane0_to_scalar_extract",
    [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "generic typed RVV i32 LMUL m1 lane0 -> scalar i32 extract bridge op";
  let description = [{
    Records one bounded generic plugin-local RVV scalar-lane extract as ONE
    typed dataflow op: take a typed i32 LMUL m1 vector token whose lane 0 is
    the scalar output boundary (the standalone-reduce / vwredsum output shape)
    plus the active !tcrv_rvv.vl token, and extract lane 0 into ONE scalar i32.
    The op does NOT touch the integer quant payload, the widening product, the
    reduce that produced the lane0 boundary, the per-block scale/dequant, the
    block loop, or the cross-block accumulate (those are separate typed steps).

    This is the scalar-lane-extract BRIDGE the M-FLAT flat-block-dot loop body
    requires: the integer core -> scalar fold handoff. The all-vector single-
    block strong path never needs it (it stays vector lane0 through
    tcrv_rvv.dequantize's vector fold), so before this op the dialect carried NO
    typed scalar-extract op and the vwredsum lane0 -> scalar i32 jump lived only
    as an opaque __riscv_vmv_x_s_i32m1_i32 call inside the monolithic block-dot
    lowering. The flat loop body's per-block dequant (brick 2,
    tcrv_rvv.block_computed_scale_dequant) hard-requires a SCALAR i32 sumi and
    ggml's byte-exact fold `(float)sumi` is a scalar cast, so under the
    zero-opaque + byte-exact constraints this vector-to-scalar jump must become
    a typed primitive.

    Making the lane0 extract a first-class typed body op (I5) means emission is
    determined by op identity + the explicit extract kind/relation, never
    inferred from candidate-mirror metadata, route ids, ABI strings, artifact
    names, status fields, exact intrinsic spellings, or q8/q4 names. The op
    lowers to the RVV lane0-extract intrinsic __riscv_vmv_x_s_i32m1_i32 (the
    SAME typed-op-lowers-to-an-intrinsic shape as tcrv_rvv.widening_product ->
    vwmul or tcrv_rvv.standalone_reduce -> vwredsum); emitting an RVV intrinsic
    via a structured call is NOT the monolithic hand-written block-dot helper.
    The i32 range fully covers the lane0 value, so the extract is byte-exact by
    construction. The op carries the active !tcrv_rvv.vl token as the vector
    boundary marker (the extraction targets lane 0 regardless of vl, so the
    intrinsic call itself takes only the vector). For the bounded surface kind
    is limited to "vector_lane0_to_scalar_i32_extract" and extract_relation to
    "i32m1-lane0-to-scalar-i32".
  }];

  let arguments = (ins
    AnyType:$input,
    AnyType:$vl,
    StrAttr:$kind,
    StrAttr:$extract_relation
  );
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$input `,` $vl attr-dict `:` type($input) `,` type($vl) `->` type($result)";
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
