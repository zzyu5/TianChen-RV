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
    plugin-local i32 dataflow tokens and ops that may appear only under an
    explicit with_vl scope. These dataflow surfaces are not executable route,
    ABI, artifact, or exporter authority; future RVV execution must be rebuilt
    through explicit extension-family IR plus a materialized EmitC route.
    Generic RVV lowering, arbitrary vector memory modeling, runtime ABI
    implementation, correctness claims, and performance evidence remain out of
    scope for the dialect skeleton.
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
    pointer parameters. Runtime scalar values such as n may use ordinary MLIR
    scalar types while still being defined by tcrv_rvv.runtime_abi_value.

    This type is not a descriptor, vector register, memory model, source
    exporter, runtime result, correctness evidence, or performance evidence.
  }];
}

def TCRVRVV_I32M1VectorType : TCRVRVV_Type<"I32M1Vector", "i32m1"> {
  let summary = "finite RVV i32 m1 vector dataflow token type";
  let description = [{
    A bounded plugin-local vector dataflow token for explicit RVV i32 dataflow
    under an existing tcrv_rvv.with_vl scope. It represents the
    compiler-visible result of tcrv_rvv.i32_load plus tcrv_rvv.i32_add,
    tcrv_rvv.i32_sub, or tcrv_rvv.i32_mul in that non-executable scope. It does
    not model arbitrary vector registers, generic memory, masks, layouts,
    runtime ABI parameters, route selection, artifact emission, hardware facts,
    correctness evidence, or performance evidence.
  }];
}

def TCRVRVV_I32M1MaskType : TCRVRVV_Type<"I32M1Mask", "i32m1_mask"> {
  let summary = "finite RVV i32 m1 predicate mask dataflow token type";
  let description = [{
    A bounded plugin-local predicate token for explicit RVV i32m1 compare and
    select dataflow under an existing tcrv_rvv.with_vl scope. It represents
    the compiler-visible result of a typed tcrv_rvv.i32_cmp_eq op and may be
    consumed by a typed tcrv_rvv.i32_select op in the same bounded scope.

    This is not a generic mask register type, vector dialect mask, descriptor,
    route selector, artifact authority, hardware fact, correctness evidence,
    or performance evidence.
  }];
}

def TCRVRVV_I32M2VectorType : TCRVRVV_Type<"I32M2Vector", "i32m2"> {
  let summary = "finite RVV i32 m2 vector dataflow token type";
  let description = [{
    A bounded plugin-local vector dataflow token for explicit RVV i32 dataflow
    when the enclosing control-plane config is SEW 32 and LMUL m2. It is a
    sibling of !tcrv_rvv.i32m1, not a generic RVV vector type. The enclosing
    tcrv_rvv.setvl / tcrv_rvv.with_vl metadata remains the source of
    SEW/LMUL/policy truth. This token does not authorize target emission.
  }];
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
    value consumed by tcrv_rvv.setvl, tcrv_rvv.i32_load, or
    tcrv_rvv.i32_store.

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

    On the selected bounded i32m1 materialized EmitC artifact path, this op is
    also the selected construction-template boundary. In that role it may carry
    generic selected-boundary conformance facts such as source kernel, selected
    variant, origin, selected path role, status, required capabilities, and
    RVV-owned route/config handoff attributes. Those facts are validation
    inputs for the artifact adapter; they are not descriptor fields or compute
    semantics.

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

def I32BinaryPreRealizedBodyOp
    : TCRVRVV_Op<"i32_binary_pre_realized_body"> {
  let summary = "bounded pre-realized RVV i32 binary selected body";
  let description = [{
    Carries the RVV plugin-local pre-realized selected body for the bounded
    i32m1 arithmetic binary family. The op is nested directly in a selected
    tcrv.exec.variant and binds explicit runtime ABI values for lhs, rhs, out,
    and runtime n/AVL together with the selected operation kind, memory form,
    and compile-time RVV config.

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
    StrAttr:$op_kind,
    StrAttr:$memory_form,
    I64Attr:$sew,
    StrAttr:$lmul,
    TCRVRVV_PolicyAttr:$policy
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $out `,` $n attr-dict `:` type($lhs) `,` type($rhs) `,` type($out) `,` type($n)";
  let hasVerifier = 1;
}

def I32LoadOp : TCRVRVV_Op<"i32_load",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "bounded RVV i32 load dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i32 load dataflow step. The op is
    nested directly under tcrv_rvv.with_vl, consumes one explicit runtime ABI
    buffer SSA value plus the same runtime !tcrv_rvv.vl control token as the
    surrounding scope, and produces one bounded !tcrv_rvv.i32m1 or
    !tcrv_rvv.i32m2 dataflow token matching the enclosing LMUL scope.

    Buffer names, C types, and ABI roles come from the defining
    tcrv_rvv.runtime_abi_value op, not from descriptor metadata or target-side
    inference. Runtime n/AVL/VL remains represented by real runtime
    SSA/control surfaces. Deleted local element-count metadata is not accepted
    here.
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

def I32BroadcastLoadOp : TCRVRVV_Op<"i32_broadcast_load",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "bounded RVV i32 broadcast load dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i32 broadcast dataflow step. The op
    is nested directly under tcrv_rvv.with_vl, consumes one explicit RHS
    runtime ABI buffer SSA value plus the same runtime !tcrv_rvv.vl control
    token as the surrounding scope, and produces one bounded
    !tcrv_rvv.i32m1 or !tcrv_rvv.i32m2 value matching the enclosing LMUL
    scope.

    This is the typed Stage 2 broadcast surface for the current i32m1/i32m2
    slice. The broadcast source is the explicit RHS ABI operand, not a route
    id, artifact name, source-front-door name, descriptor, ABI parameter name,
    or target/export-side inference.
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

def I32AddOp : TCRVRVV_Op<"i32_add",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "bounded RVV i32 add dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i32 vector add dataflow step. The op
    is nested directly under tcrv_rvv.with_vl, consumes two same-LMUL i32
    dataflow tokens and the same runtime !tcrv_rvv.vl control token as the
    surrounding scope, and produces one bounded i32 dataflow token matching the
    enclosing LMUL scope.

    This is not generic arithmetic IR and does not authorize RVV artifact
    emission.
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

def I32SubOp : TCRVRVV_Op<"i32_sub",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "bounded RVV i32 subtract dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i32 vector subtract dataflow step. The
    op is nested directly under tcrv_rvv.with_vl, consumes two same-LMUL i32
    dataflow tokens and the same runtime !tcrv_rvv.vl control token as the
    surrounding scope, and produces one bounded i32 dataflow token matching the
    enclosing LMUL scope.

    This is not generic arithmetic IR and does not authorize RVV artifact
    emission.
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

def I32MulOp : TCRVRVV_Op<"i32_mul",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "bounded RVV i32 multiply dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i32 vector multiply dataflow step. The
    op is nested directly under tcrv_rvv.with_vl, consumes two same-LMUL i32
    dataflow tokens and the same runtime !tcrv_rvv.vl control token as the
    surrounding scope, and produces one bounded i32 dataflow token matching the
    enclosing LMUL scope.

    This is not generic arithmetic IR and does not authorize RVV artifact
    emission.
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

def I32CmpEqOp : TCRVRVV_Op<"i32_cmp_eq",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "bounded RVV i32m1 equality compare dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i32m1 predicate-producing compare
    step. The op is nested directly under tcrv_rvv.with_vl, consumes two
    !tcrv_rvv.i32m1 dataflow tokens and the same runtime !tcrv_rvv.vl control
    token as the surrounding scope, and produces one !tcrv_rvv.i32m1_mask
    predicate token.

    Compare semantics come only from this typed operation in the selected RVV
    body. Route ids, artifact names, source-front-door names, descriptor
    residue, ABI parameter names, and common/export code must not infer or
    choose compare behavior.
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

def I32SelectOp : TCRVRVV_Op<"i32_select",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "bounded RVV i32m1 predicate-consuming select dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i32m1 select dataflow step. The op is
    nested directly under tcrv_rvv.with_vl, consumes one
    !tcrv_rvv.i32m1_mask predicate produced by tcrv_rvv.i32_cmp_eq in the same
    selected VL scope, consumes true and false !tcrv_rvv.i32m1 dataflow
    values, consumes the same runtime !tcrv_rvv.vl control token as the
    surrounding scope, and produces one selected !tcrv_rvv.i32m1 value.

    Select semantics come only from this typed operation and its explicit
    predicate operand in the selected RVV body. They must not be inferred from
    route metadata, artifact names, source-front-door names, descriptors, ABI
    parameter names, or common/export code.
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

def I32StoreOp : TCRVRVV_Op<"i32_store",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName",
       "getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "bounded RVV i32 store dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i32 store dataflow step. The op is
    nested directly under tcrv_rvv.with_vl, consumes one explicit output
    runtime ABI buffer SSA value, one bounded i32 dataflow token matching the
    enclosing LMUL scope, and the same runtime !tcrv_rvv.vl control token as
    the surrounding scope.

    Output C name, C type, and ABI role come from the defining
    tcrv_rvv.runtime_abi_value op, not from descriptor metadata or target-side
    inference. Runtime n/AVL/VL remains represented by real runtime
    SSA/control surfaces. Deleted local element-count metadata is not accepted
    here.
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
