#ifndef TIANCHENRV_DIALECT_RVV_IR_RVVOPS_TD
#define TIANCHENRV_DIALECT_RVV_IR_RVVOPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
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

    The initial control-plane slice exposes the bounded tcrv_rvv.setvl runtime
    AVL-to-VL surface and the matching tcrv_rvv.with_vl VL scope region. The
    first executable microkernel exception adds finite plugin-local
    tcrv_rvv.i32_load, tcrv_rvv.i32_add, tcrv_rvv.i32_sub,
    tcrv_rvv.i32_mul, and
    tcrv_rvv.i32_store dataflow operations under that scope so the RVV
    exporter can consume a structured i32 vector add/sub/mul microkernel family
    without turning tcrv.exec into a compute IR. Generic
    RVV lowering, arbitrary vector memory modeling, runtime ABI implementation,
    correctness claims, and performance evidence remain out of scope for the
    dialect skeleton.
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
    A bounded plugin-local vector dataflow token for the current RVV i32 binary
    microkernel family slice. It represents the compiler-visible result of the
    finite tcrv_rvv.i32_load plus tcrv_rvv.i32_add, tcrv_rvv.i32_sub, or
    tcrv_rvv.i32_mul
    dataflow chain under an existing tcrv_rvv.with_vl scope. It does not model
    arbitrary vector registers,
    generic memory, masks, layouts, runtime ABI parameters, hardware facts,
    correctness evidence, or performance evidence.
  }];
}

def SetVLOp : TCRVRVV_Op<"setvl", []> {
  let summary = "bounded RVV runtime AVL to VL control-plane op";
  let description = [{
    Models the first plugin-local RVV runtime vector-length control-plane
    surface. The operation consumes one runtime AVL SSA value, produces one
    !tcrv_rvv.vl token, and carries bounded compile-time first-slice
    configuration metadata: SEW 32, LMUL m1, and a finite #tcrv_rvv.policy
    attribute.

    This op does not model VLEN/vlenb hardware facts, descriptor-local element
    counts, required march metadata, required capability metadata, pointer or
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
    work only. It does not model VLEN/vlenb hardware facts, descriptor-local
    element counts, required march metadata, required capability metadata,
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
    Records one finite plugin-local RVV i32 load in the current i32 binary
    microkernel body. The op is nested directly under tcrv_rvv.with_vl,
    consumes the same runtime !tcrv_rvv.vl control token as the surrounding
    scope, carries exactly one input buffer ABI role, and produces one bounded
    !tcrv_rvv.i32m1 dataflow token.

    Pointer operands remain target/export ABI-owned parameters, not IR-modeled
    buffers. Runtime n/AVL/VL remains represented by the enclosing
    microkernel region argument, tcrv_rvv.setvl, and tcrv_rvv.with_vl control
    surface. Descriptor-local element_count remains metadata on the enclosing
    microkernel and is not accepted here.
  }];

  let arguments = (ins
    AnyType:$vl,
    StrAttr:$buffer_role
  );
  let results = (outs AnyType:$loaded);

  let assemblyFormat = "$vl attr-dict `:` type($vl) `->` type($loaded)";
  let hasVerifier = 1;
}

def I32AddOp : TCRVRVV_Op<"i32_add", []> {
  let summary = "bounded RVV i32 add dataflow op";
  let description = [{
    Records the finite plugin-local RVV i32 vector add in the current i32-vadd
    microkernel body. The op is nested directly under tcrv_rvv.with_vl,
    consumes two !tcrv_rvv.i32m1 dataflow tokens and the same runtime
    !tcrv_rvv.vl control token as the surrounding scope, and produces one
    bounded !tcrv_rvv.i32m1 dataflow token.

    This is not a generic arithmetic IR. It is admitted only as part of the
    finite i32-vadd microkernel slice validated by the RVV plugin/exporter.
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

def I32SubOp : TCRVRVV_Op<"i32_sub", []> {
  let summary = "bounded RVV i32 subtract dataflow op";
  let description = [{
    Records the finite plugin-local RVV i32 vector subtract in the current
    i32-vsub microkernel body. The op is nested directly under
    tcrv_rvv.with_vl, consumes two !tcrv_rvv.i32m1 dataflow tokens and the
    same runtime !tcrv_rvv.vl control token as the surrounding scope, and
    produces one bounded !tcrv_rvv.i32m1 dataflow token.

    This is not a generic arithmetic IR. It is admitted only as part of the
    finite i32-vsub microkernel slice validated by the RVV plugin/exporter.
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

def I32MulOp : TCRVRVV_Op<"i32_mul", []> {
  let summary = "bounded RVV i32 multiply dataflow op";
  let description = [{
    Records the finite plugin-local RVV i32 vector multiply in the current
    i32-vmul microkernel body. The op is nested directly under
    tcrv_rvv.with_vl, consumes two !tcrv_rvv.i32m1 dataflow tokens and the
    same runtime !tcrv_rvv.vl control token as the surrounding scope, and
    produces one bounded !tcrv_rvv.i32m1 dataflow token.

    This is not a generic arithmetic IR. It is admitted only as part of the
    finite i32-vmul microkernel slice validated by the RVV plugin/exporter.
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
    Records one finite plugin-local RVV i32 store in the current i32 binary
    microkernel body. The op is nested directly under tcrv_rvv.with_vl,
    consumes one !tcrv_rvv.i32m1 dataflow token and the same runtime
    !tcrv_rvv.vl control token as the surrounding scope, and carries exactly
    one output buffer ABI role.

    Pointer operands remain target/export ABI-owned parameters, not IR-modeled
    buffers. Runtime n/AVL/VL remains represented by the enclosing
    microkernel region argument, tcrv_rvv.setvl, and tcrv_rvv.with_vl control
    surface. Descriptor-local element_count remains metadata on the enclosing
    microkernel and is not accepted here.
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

def LoweringBoundaryOp : TCRVRVV_Op<"lowering_boundary", []> {
  let summary = "pre-executable RVV lowering boundary metadata";
  let description = [{
    Records the plugin-owned RVV lowering boundary for a selected RVV
    tcrv.exec.variant. The op is intentionally pre-executable metadata: it
    carries source kernel, selected variant, dispatch role, capability summary,
    unsupported status, and diagnostic reason so later RVV lowering slices have
    a stable plugin-local attachment point.

    This op does not model RVV arithmetic, tensor computation, vector
    intrinsics, memory effects, LLVM/RISC-V lowering, runtime ABI glue,
    generated artifacts, hardware execution, correctness, or performance.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    StrAttr:$status,
    ArrayAttr:$required_capabilities,
    OptionalAttr<I64Attr>:$vlenb_bytes,
    OptionalAttr<I64Attr>:$i32_m1_lanes,
    OptionalAttr<StrAttr>:$capability_summary,
    StrAttr:$unsupported_reason
  );

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def I32VAddMicrokernelOp : TCRVRVV_Op<"i32_vadd_microkernel",
    [NoTerminator, SingleBlock]> {
  let summary = "explicit bounded RVV i32 vector-add executable microkernel";
  let description = [{
    Records one plugin-local executable RVV i32 vector-add microkernel body for
    a selected RVV variant. This is intentionally a finite smoke microkernel
    surface: it carries selected-path metadata, required RVV capability
    references, the required march, a tiny element count, and one structured
    RVV dataflow body that a target-local exporter can validate before
    emitting deterministic standalone C using RVV intrinsics. The body has one
    runtime index block argument representing the target/export-owned runtime
    element count/AVL control value, then a bounded tcrv_rvv.setvl to
    tcrv_rvv.with_vl chain whose nested body contains exactly one finite
    lhs-load, rhs-load, add, output-store sequence.

    This op is not a generic tensor, tile, benchmark, bufferization, lowering,
    runtime ABI, or arbitrary vector math IR. It represents exactly the first
    bounded RVV executable microkernel slice for i32 vector-add smoke
    correctness, while keeping descriptor-local element_count distinct from
    runtime n/AVL/VL control.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    I64Attr:$element_count,
    ArrayAttr:$required_capabilities,
    StrAttr:$required_march,
    OptionalAttr<StrAttr>:$selected_mabi
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def I32VSubMicrokernelOp : TCRVRVV_Op<"i32_vsub_microkernel",
    [NoTerminator, SingleBlock]> {
  let summary = "explicit bounded RVV i32 vector-subtract executable microkernel";
  let description = [{
    Records one plugin-local executable RVV i32 vector-subtract microkernel
    body for a selected RVV variant. This is intentionally a finite sibling in
    the bounded RVV C-intrinsic microkernel family: it carries selected-path
    metadata, required RVV capability references, the required march, a tiny
    element count, and one structured RVV dataflow body that a target-local
    exporter can validate before emitting deterministic standalone C using
    RVV riscv_vector.h intrinsics. The body has one runtime index block
    argument representing the target/export-owned runtime element count/AVL
    control value, then a bounded tcrv_rvv.setvl to tcrv_rvv.with_vl chain
    whose nested body contains exactly one finite lhs-load, rhs-load, subtract,
    output-store sequence.

    This op is not a generic tensor, tile, benchmark, bufferization, lowering,
    runtime ABI, or arbitrary vector math IR. It represents exactly the
    bounded RVV executable microkernel slice for i32 vector-subtract, while
    keeping descriptor-local element_count distinct from runtime n/AVL/VL
    control.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    I64Attr:$element_count,
    ArrayAttr:$required_capabilities,
    StrAttr:$required_march,
    OptionalAttr<StrAttr>:$selected_mabi
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def I32VMulMicrokernelOp : TCRVRVV_Op<"i32_vmul_microkernel",
    [NoTerminator, SingleBlock]> {
  let summary = "explicit bounded RVV i32 vector-multiply executable microkernel";
  let description = [{
    Records one plugin-local executable RVV i32 vector-multiply microkernel
    body for a selected RVV variant. This is intentionally a finite sibling in
    the bounded RVV C-intrinsic microkernel family: it carries selected-path
    metadata, required RVV capability references, the required march, a tiny
    element count, and one structured RVV dataflow body that a target-local
    exporter can validate before emitting deterministic standalone C using
    RVV riscv_vector.h intrinsics. The body has one runtime index block
    argument representing the target/export-owned runtime element count/AVL
    control value, then a bounded tcrv_rvv.setvl to tcrv_rvv.with_vl chain
    whose nested body contains exactly one finite lhs-load, rhs-load, multiply,
    output-store sequence.

    This op is not a generic tensor, tile, benchmark, bufferization, lowering,
    runtime ABI, or arbitrary vector math IR. It represents exactly the
    bounded RVV executable microkernel slice for i32 vector-multiply, while
    keeping descriptor-local element_count distinct from runtime n/AVL/VL
    control.
  }];

  let arguments = (ins
    StrAttr:$source_kernel,
    FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin,
    StrAttr:$role,
    I64Attr:$element_count,
    ArrayAttr:$required_capabilities,
    StrAttr:$required_march,
    OptionalAttr<StrAttr>:$selected_mabi
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

#endif // TIANCHENRV_DIALECT_RVV_IR_RVVOPS_TD
