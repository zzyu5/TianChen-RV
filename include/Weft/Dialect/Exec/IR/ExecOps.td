#ifndef WEFT_DIALECT_EXEC_IR_EXECOPS_TD
#define WEFT_DIALECT_EXEC_IR_EXECOPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def WEFT_Exec_Dialect : Dialect {
  let name = "weft";
  let cppNamespace = "::weft::exec";
  let summary = "Weft-RV execution organization dialect";
  let description = [{
    The weft.exec operation family is the stable Weft-RV core contract. It
    is registered under the MLIR dialect namespace weft so textual operations
    print as weft.exec.* while remaining compatible with MLIR's first-dot
    dialect namespace resolution.

    Concrete computation is intentionally plugin-local and must be represented
    by extension dialects such as weft.rvv, weft.ime, weft.offload, scalar
    fallback, or future plugin dialects.
  }];

  let useDefaultAttributePrinterParser = 1;
}

class WEFTExec_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<WEFT_Exec_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// Closed, ODS-defined set of capability/target `status` values. This is the
// single source of truth for capability status: a capability is available
// only when its status is exactly `available`; `unavailable`/`disabled`/
// `missing` are the distinct unavailable spellings. The status attribute keeps
// its quoted StringAttr textual form (status = "available"), so the IR text and
// the C++ diagnostics that round-trip the raw spelling are unchanged; the enum
// supplies the typed, closed value set and the canonical availability mapping.
// `genSpecializedAttr = 0` emits only the C++ enum plus symbolize/stringify
// converters (no attribute storage class), so the attribute remains a plain
// StringAttr in the IR while becoming typed/closed in C++.
def WEFTExec_CapabilityStatusAvailable
    : I32EnumAttrCase<"Available", 0, "available">;
def WEFTExec_CapabilityStatusUnavailable
    : I32EnumAttrCase<"Unavailable", 1, "unavailable">;
def WEFTExec_CapabilityStatusDisabled
    : I32EnumAttrCase<"Disabled", 2, "disabled">;
def WEFTExec_CapabilityStatusMissing
    : I32EnumAttrCase<"Missing", 3, "missing">;

def WEFTExec_CapabilityStatus : I32EnumAttr<
    "CapabilityStatus", "closed capability/target availability status",
    [WEFTExec_CapabilityStatusAvailable, WEFTExec_CapabilityStatusUnavailable,
     WEFTExec_CapabilityStatusDisabled, WEFTExec_CapabilityStatusMissing]> {
  let cppNamespace = "::weft::exec";
  let genSpecializedAttr = 0;
}

def WEFTExec_IntegerSignednessSigned
    : I32EnumAttrCase<"Signed", 0, "signed">;
def WEFTExec_IntegerSignednessUnsigned
    : I32EnumAttrCase<"Unsigned", 1, "unsigned">;

def WEFTExec_IntegerSignedness : I32EnumAttr<
    "IntegerSignedness", "closed integer operand signedness",
    [WEFTExec_IntegerSignednessSigned,
     WEFTExec_IntegerSignednessUnsigned]> {
  let cppNamespace = "::weft::exec";
  let genSpecializedAttr = 0;
}

def WEFTExec_IntegerSignednessAttr
    : EnumAttr<WEFT_Exec_Dialect, WEFTExec_IntegerSignedness,
               "integer_signedness">;

def WEFTExec_CapabilityRelationsAttr
    : WEFTExec_Attr<"CapabilityRelations", "capability_relations"> {
  let summary = "typed provides/implies/conflicts capability-id relations";
  let description = [{
    Typed structure for a capability's cross-capability relations. Each list
    holds capability *IDs* (free dotted strings in the capability-id namespace,
    e.g. "build.policy.no_inline_asm"), NOT MLIR symbol references. The IDs are
    queried directly by capability-aware C++ passes/plugins without string-keyed
    getAttrOfType lookups. Resolution by ID (provides/implies/conflicts) is a
    pass-time concern; this attribute only structures the relation contract and
    its verifier is hygiene-only (non-empty, no required trimming, single-line,
    no duplicate ids per list).
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"::mlir::StringAttr">:$provides,
    OptionalArrayRefParameter<"::mlir::StringAttr">:$implies,
    OptionalArrayRefParameter<"::mlir::StringAttr">:$conflicts);
  let assemblyFormat =
    "`<` (`provides` `=` `[` $provides^ `]`)? "
    "(`implies` `=` `[` $implies^ `]`)? "
    "(`conflicts` `=` `[` $conflicts^ `]`)? `>`";
  let genVerifyDecl = 1;
}

class WEFTExec_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Exec_Dialect, mnemonic, traits>;

def WEFTExec_CanonicalProblem
    : NativeOpTrait<"weft::CanonicalProblem">;

def TargetOp : WEFTExec_Op<"exec.target", [Symbol]> {
  let summary = "named target capability profile anchor";
  let description = [{
    Defines a named target profile anchor. Structured capability details are
    carried as MLIR attributes and are consumed by capability-aware passes.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                   OptionalAttr<ArrayAttr>:$capability_providers,
                   OptionalAttr<WEFTExec_CapabilityRelationsAttr>:$relations);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def CapabilityOp : WEFTExec_Op<"exec.capability", [Symbol]> {
  let summary = "named compiler-visible target capability";
  let description = [{
    Records a compiler-visible capability such as an ISA extension,
    microarchitecture feature, runtime/offload capability, or toolchain
    feature. Capability semantics are intentionally structured MLIR
    attributes, not comments or unparsed side strings.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                   OptionalAttr<StrAttr>:$id,
                   OptionalAttr<StrAttr>:$kind,
                   OptionalAttr<WEFTExec_CapabilityRelationsAttr>:$relations);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Int8MACProblemOp
    : WEFTExec_Op<"exec.int8_mac_problem",
                  [Symbol, WEFTExec_CanonicalProblem]> {
  let summary = "family-neutral int8 multiply-accumulate source problem";
  let description = [{
    Defines one canonical, pre-family-binding int8 multiply-accumulate problem.
    Operation identity fixes S to an int8 x int8 -> int32 MAC fragment; the two
    required signedness attributes are typed representation/semantic facts.

    This op is a bounded operator-problem anchor, not a tensor/tile IR and not
    an executable family body. It carries no target capability, family,
    schedule, candidate, selected variant, artifact route, or runtime observer.
    A weft.exec.kernel may reference it through `problem = @symbol`; common
    orchestration resolves only that exact symbol and leaves every field to the
    bound family plugin.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    WEFTExec_IntegerSignednessAttr:$lhs_signedness,
    WEFTExec_IntegerSignednessAttr:$rhs_signedness,
    I64Attr:$m,
    I64Attr:$n,
    I64Attr:$k
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def DequantizeRowQ40ProblemOp
    : WEFTExec_Op<"exec.dequantize_row_q4_0_problem",
                  [Symbol, WEFTExec_CanonicalProblem]> {
  let summary = "family-neutral q4_0 dequantize-row source problem";
  let description = [{
    Defines the bounded q4_0 block expansion problem before target/family
    binding. Operation identity fixes S to q4_0 packed-nibble values scaled to
    f32 output; required attributes carry the canonical external block geometry
    g. It contains no Scalar/RVV/IME schedule or artifact choice.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$qk,
    I64Attr:$weight_block_stride,
    I64Attr:$weight_d_byte_offset,
    I64Attr:$weight_quant_byte_offset
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def KernelOp : WEFTExec_Op<"exec.kernel",
    [IsolatedFromAbove, NoRegionArguments, NoTerminator, SingleBlock, Symbol,
     SymbolTable]> {
  let summary = "execution kernel container";
  let description = [{
    Contains target declarations, capability objects, execution variants,
    dispatch, and fallback organization for one high-level kernel realization.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                   OptionalAttr<FlatSymbolRefAttr>:$target,
                   OptionalAttr<FlatSymbolRefAttr>:$problem,
                   OptionalAttr<StrAttr>:$construction_domain);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def VariantOp : WEFTExec_Op<"exec.variant",
    [IsolatedFromAbove, NoRegionArguments, NoTerminator, SingleBlock, Symbol,
     SymbolTable]> {
  let summary = "plugin-proposed execution variant";
  let description = [{
    Contains one plugin-proposed execution variant. The origin plugin,
    required capabilities, generic decision metadata, preconditions, tuning
    metadata, cost metadata, and emission path are represented as MLIR
    attributes or nested extension operations.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                   OptionalAttr<StrAttr>:$origin,
                   OptionalAttr<ArrayAttr>:$requires,
                   OptionalAttr<StrAttr>:$condition,
                   OptionalAttr<StrAttr>:$guard,
                   OptionalAttr<StrAttr>:$policy);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def MemWindowOp : WEFTExec_Op<"exec.mem_window", [Symbol]> {
  let summary = "named memory organization window";
  let description = [{
    Describes memory-window, view, or binding context used by variants and
    dispatch guards. For bounded runtime ABI slices it may also bind one
    named buffer-window ABI role, access direction, ownership, and known C ABI
    type spelling. This op is intentionally about execution memory
    organization, not tensor computation, shapes, vector math, or
    extension-specific buffers.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                   OptionalAttr<StrAttr>:$purpose,
                   OptionalAttr<StrAttr>:$binding,
                   OptionalAttr<StrAttr>:$memory_space,
                   OptionalAttr<StrAttr>:$abi_role,
                   OptionalAttr<StrAttr>:$access,
                   OptionalAttr<StrAttr>:$ownership,
                   OptionalAttr<StrAttr>:$c_type);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def RuntimeParamOp : WEFTExec_Op<"exec.runtime_param", [Symbol]> {
  let summary = "named runtime ABI/control scalar parameter";
  let description = [{
    Describes one runtime scalar ABI/control parameter used to organize an
    execution boundary. It may bind a named ABI role to a C parameter name,
    C type spelling, ownership class, and purpose. This op is intentionally
    execution organization only; it does not model computation, high-level
    tensor shape, hardware probing, vector math, or target-specific legality.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                   OptionalAttr<StrAttr>:$purpose,
                   OptionalAttr<StrAttr>:$abi_role,
                   OptionalAttr<StrAttr>:$c_name,
                   OptionalAttr<StrAttr>:$c_type,
                   OptionalAttr<StrAttr>:$ownership);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def HartParallelOp : WEFTExec_Op<"exec.hart_parallel",
    [NoRegionArguments, NoTerminator, SingleBlock]> {
  let summary = "RISC-V hart/core parallel execution organization";
  let description = [{
    Groups extension-owned work under a coarse-grained RISC-V hart/core
    parallel organization. The core dialect validates structural metadata only;
    concrete thread runtimes and extension legality remain plugin-local.
  }];

  let arguments = (ins OptionalAttr<I64Attr>:$harts,
                   OptionalAttr<StrAttr>:$policy);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def RegionOp : WEFTExec_Op<"exec.region",
    [NoRegionArguments, NoTerminator, SingleBlock]> {
  let summary = "extension-owned execution resource region";
  let description = [{
    Provides a single-block execution/resource organization region for
    extension-owned work. The core dialect records structured region metadata
    but does not interpret RVV, IME, offload, or future plugin legality here.
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$region_kind,
                   OptionalAttr<StrAttr>:$name,
                   OptionalAttr<StrAttr>:$purpose);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def DiagnosticOp : WEFTExec_Op<"exec.diagnostic", []> {
  let summary = "structured compiler-visible execution diagnostic metadata";
  let description = [{
    Records structured metadata explaining variant rejection, fallback,
    selection, missing capability, missing runtime, or other compiler-visible
    execution decisions. It is metadata only and does not model computation.
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$reason,
                   OptionalAttr<StrAttr>:$message,
                   OptionalAttr<StrAttr>:$severity,
                   OptionalAttr<StrAttr>:$status,
                   OptionalAttr<FlatSymbolRefAttr>:$target,
                   OptionalAttr<StrAttr>:$selection_kind,
                   OptionalAttr<StrAttr>:$origin,
                   OptionalAttr<StrAttr>:$role,
                   OptionalAttr<StrAttr>:$plan_kind,
                   OptionalAttr<StrAttr>:$emission_kind,
                   OptionalAttr<StrAttr>:$lowering_pipeline,
                   OptionalAttr<StrAttr>:$runtime_abi,
                   OptionalAttr<StrAttr>:$runtime_abi_kind,
                   OptionalAttr<StrAttr>:$runtime_abi_name,
                   OptionalAttr<ArrayAttr>:$runtime_abi_parameters,
                   OptionalAttr<StrAttr>:$runtime_glue_role,
                   OptionalAttr<StrAttr>:$artifact_kind,
                   OptionalAttr<ArrayAttr>:$artifact_metadata,
                   OptionalAttr<StrAttr>:$lowering_boundary,
                   OptionalAttr<ArrayAttr>:$required_capabilities,
                   OptionalAttr<BoolAttr>:$preference_available,
                   OptionalAttr<F64Attr>:$preference_score,
                   OptionalAttr<I64Attr>:$preference_rank,
                   OptionalAttr<StrAttr>:$preference_policy,
                   OptionalAttr<StrAttr>:$preference_explanation,
                   OptionalAttr<StrAttr>:$preference_tie_break,
                   OptionalAttr<StrAttr>:$fallback_role);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def DispatchOp : WEFTExec_Op<"exec.dispatch",
    [NoRegionArguments, NoTerminator, SingleBlock]> {
  let summary = "capability-aware runtime dispatch organization";
  let description = [{
    Groups structured dispatch cases and exactly one fallback behavior. Dispatch
    metadata names generic capability, runtime probe, guard, threshold, or policy
    facts without interpreting extension-specific semantics in the core dialect.
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def DispatchCaseOp : WEFTExec_Op<"exec.case", []> {
  let summary = "compiler-visible dispatch case variant reference";
  let description = [{
    References a candidate execution variant considered by a surrounding
    weft.exec.dispatch. Optional condition, guard, and policy metadata are
    generic non-empty strings; extension-specific interpretation remains
    plugin-local. Optional runtime_guard_required is a typed compiler-owned
    marker that this case requires an explicit runtime dispatch-availability
    guard. Optional runtime_guard links executable dispatch control to a
    same-kernel weft.exec.runtime_param with the generic
    dispatch-availability-guard ABI role.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
                   OptionalAttr<StrAttr>:$condition,
                   OptionalAttr<StrAttr>:$guard,
                   OptionalAttr<StrAttr>:$policy,
                   OptionalAttr<BoolAttr>:$runtime_guard_required,
                   OptionalAttr<FlatSymbolRefAttr>:$runtime_guard,
                   OptionalAttr<StrAttr>:$origin,
                   OptionalAttr<BoolAttr>:$preference_available,
                   OptionalAttr<F64Attr>:$preference_score,
                   OptionalAttr<I64Attr>:$preference_rank,
                   OptionalAttr<StrAttr>:$preference_policy,
                   OptionalAttr<StrAttr>:$preference_explanation,
                   OptionalAttr<StrAttr>:$preference_tie_break);
  let assemblyFormat = "$target attr-dict";
  let hasVerifier = 1;
}

def FallbackOp : WEFTExec_Op<"exec.fallback", []> {
  let summary = "fallback variant reference";
  let description = [{
    References a fallback variant selected when capability, legality, runtime,
    or policy conditions reject preferred variants.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
                   OptionalAttr<StrAttr>:$origin,
                   OptionalAttr<BoolAttr>:$preference_available,
                   OptionalAttr<F64Attr>:$preference_score,
                   OptionalAttr<I64Attr>:$preference_rank,
                   OptionalAttr<StrAttr>:$preference_policy,
                   OptionalAttr<StrAttr>:$preference_explanation,
                   OptionalAttr<StrAttr>:$preference_tie_break,
                   OptionalAttr<StrAttr>:$fallback_role);
  let assemblyFormat = "$target attr-dict";
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_EXEC_IR_EXECOPS_TD
