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
    plugin-local i32/i64 dataflow tokens and ops that may appear only under an
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

def TCRVRVV_I64M1VectorType : TCRVRVV_Type<"I64M1Vector", "i64m1"> {
  let summary = "finite RVV i64 m1 vector dataflow token type";
  let description = [{
    A bounded plugin-local vector dataflow token for explicit RVV i64 dataflow
    when the enclosing control-plane config is SEW 64 and LMUL m1. It is not a
    generic RVV vector type. The enclosing tcrv_rvv.setvl / tcrv_rvv.with_vl
    metadata remains the source of SEW/LMUL/policy truth. This token does not
    authorize target emission.
  }];
}

def SetVLOp : TCRVRVV_Op<"setvl", []> {
  let summary = "bounded RVV runtime AVL to VL control-plane op";
  let description = [{
    Models the first plugin-local RVV runtime vector-length control-plane
    surface. The operation consumes one runtime AVL SSA value, produces one
    !tcrv_rvv.vl token, and carries bounded compile-time first-slice
    configuration metadata: SEW 32, LMUL m1 or m2, and a finite
    #tcrv_rvv.policy attribute.

    This op does not model VLEN/vlenb hardware facts, artifact-local component
    capacity, required march metadata, required capability metadata, pointer or
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
    [NoRegionArguments, NoTerminator, SingleBlock]> {
  let summary = "bounded RVV VL scope control-plane region";
  let description = [{
    Creates a plugin-local RVV control-plane scope for work that is governed by
    an existing runtime VL token. The operation consumes one !tcrv_rvv.vl value,
    normally produced by tcrv_rvv.setvl, and owns one single-block region with
    no region arguments. Optional duplicated SEW/LMUL/policy attributes are
    limited to the same bounded first-slice compile-time configuration accepted
    by setvl and are verified for compatibility when the consumed VL is visibly
    defined by a tcrv_rvv.setvl op.

    This op is a structural prerequisite for future RVV control/body lowering
    work only. It does not model VLEN/vlenb hardware facts, artifact-local
    component capacity, required march metadata, required capability metadata,
    raw capability facts, pointer or memory operands, RVV arithmetic, vector
    registers, masks, lowering, runtime ABI glue, generated artifacts,
    hardware execution, correctness, or performance.
  }];

  let arguments = (ins
    AnyType:$vl,
    OptionalAttr<I64Attr>:$sew,
    OptionalAttr<StrAttr>:$lmul,
    OptionalAttr<TCRVRVV_PolicyAttr>:$policy
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$vl attr-dict-with-keyword $body `:` type($vl)";
  let hasVerifier = 1;
}

def I32LoadOp : TCRVRVV_Op<"i32_load", []> {
  let summary = "bounded RVV i32 load dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i32 load dataflow step. The op is
    nested directly under tcrv_rvv.with_vl, consumes the same runtime
    !tcrv_rvv.vl control token as the surrounding scope, carries exactly one
    input buffer ABI role reference, and produces one bounded !tcrv_rvv.i32m1
    or !tcrv_rvv.i32m2 dataflow token matching the enclosing LMUL scope.

    Pointer operands remain future target/export ABI-owned parameters, not
    IR-modeled buffers. Runtime n/AVL/VL remains represented by real runtime
    SSA/control surfaces. Deleted wrapper-local element_count metadata is not
    accepted here.
  }];

  let arguments = (ins
    AnyType:$vl,
    StrAttr:$buffer_role
  );
  let results = (outs AnyType:$loaded);

  let assemblyFormat = "$vl attr-dict `:` type($vl) `->` type($loaded)";
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

def I32StoreOp : TCRVRVV_Op<"i32_store", []> {
  let summary = "bounded RVV i32 store dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i32 store dataflow step. The op is
    nested directly under tcrv_rvv.with_vl, consumes one bounded i32 dataflow
    token matching the enclosing LMUL scope and the same runtime !tcrv_rvv.vl
    control token as the surrounding scope, and carries exactly one output
    buffer ABI role reference.

    Pointer operands remain future target/export ABI-owned parameters, not
    IR-modeled buffers. Runtime n/AVL/VL remains represented by real runtime
    SSA/control surfaces. Deleted wrapper-local element_count metadata is not
    accepted here.
  }];

  let arguments = (ins
    AnyType:$value,
    AnyType:$vl,
    StrAttr:$buffer_role
  );

  let assemblyFormat =
    "$value `,` $vl attr-dict `:` type($value) `,` type($vl)";
  let hasVerifier = 1;
}

def I64LoadOp : TCRVRVV_Op<"i64_load", []> {
  let summary = "bounded RVV i64 load dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i64 load dataflow step. The op is
    nested directly under tcrv_rvv.with_vl, consumes the same runtime
    !tcrv_rvv.vl control token as the surrounding scope, carries exactly one
    input buffer ABI role reference, and produces one bounded !tcrv_rvv.i64m1
    dataflow token matching the enclosing SEW64/LMUL m1 scope.
  }];

  let arguments = (ins
    AnyType:$vl,
    StrAttr:$buffer_role
  );
  let results = (outs AnyType:$loaded);

  let assemblyFormat = "$vl attr-dict `:` type($vl) `->` type($loaded)";
  let hasVerifier = 1;
}

def I64AddOp : TCRVRVV_Op<"i64_add",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "bounded RVV i64 add dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i64 vector add dataflow step. The op
    is nested directly under tcrv_rvv.with_vl, consumes two i64m1 dataflow
    tokens and the same runtime !tcrv_rvv.vl control token as the surrounding
    scope, and produces one i64m1 dataflow token. This is not generic
    arithmetic IR and does not authorize RVV artifact emission.
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

def I64SubOp : TCRVRVV_Op<"i64_sub",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "bounded RVV i64 subtract dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i64 vector subtract dataflow step. The
    op is nested directly under tcrv_rvv.with_vl, consumes two i64m1 dataflow
    tokens and the same runtime !tcrv_rvv.vl control token as the surrounding
    scope, and produces one i64m1 dataflow token. This is not generic
    arithmetic IR and does not authorize RVV artifact emission.
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

def I64MulOp : TCRVRVV_Op<"i64_mul",
    [TCRVEmitCLowerableOpInterface]> {
  let summary = "bounded RVV i64 multiply dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i64 vector multiply dataflow step. The
    op is nested directly under tcrv_rvv.with_vl, consumes two i64m1 dataflow
    tokens and the same runtime !tcrv_rvv.vl control token as the surrounding
    scope, and produces one i64m1 dataflow token. This is not generic
    arithmetic IR and does not authorize RVV artifact emission.
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

def I64StoreOp : TCRVRVV_Op<"i64_store", []> {
  let summary = "bounded RVV i64 store dataflow op";
  let description = [{
    Records one bounded plugin-local RVV i64 store dataflow step. The op is
    nested directly under tcrv_rvv.with_vl, consumes one bounded i64m1 dataflow
    token and the same runtime !tcrv_rvv.vl control token as the surrounding
    scope, and carries exactly one output buffer ABI role reference.
  }];

  let arguments = (ins
    AnyType:$value,
    AnyType:$vl,
    StrAttr:$buffer_role
  );

  let assemblyFormat =
    "$value `,` $vl attr-dict `:` type($value) `,` type($vl)";
  let hasVerifier = 1;
}

#endif // TIANCHENRV_DIALECT_RVV_IR_RVVOPS_TD
