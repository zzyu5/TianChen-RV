#ifndef TIANCHENRV_DIALECT_EXEC_IR_EXECOPS_TD
#define TIANCHENRV_DIALECT_EXEC_IR_EXECOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def TCRV_Exec_Dialect : Dialect {
  let name = "tcrv";
  let cppNamespace = "::tianchenrv::tcrv::exec";
  let summary = "TianChen-RV execution organization dialect";
  let description = [{
    The tcrv.exec operation family is the stable TianChen-RV core contract. It
    is registered under the MLIR dialect namespace tcrv so textual operations
    print as tcrv.exec.* while remaining compatible with MLIR's first-dot
    dialect namespace resolution.

    Concrete computation is intentionally plugin-local and must be represented
    by extension dialects such as tcrv.rvv, tcrv.ime, tcrv.offload, scalar
    fallback, or future plugin dialects.
  }];
}

class TCRVExec_Op<string mnemonic, list<Trait> traits = []>
    : Op<TCRV_Exec_Dialect, mnemonic, traits>;

def TargetOp : TCRVExec_Op<"exec.target", [Symbol]> {
  let summary = "named target capability profile anchor";
  let description = [{
    Defines a named target profile anchor. Structured capability details are
    carried as MLIR attributes and are consumed by capability-aware passes.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                   OptionalAttr<ArrayAttr>:$capability_providers);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def CapabilityOp : TCRVExec_Op<"exec.capability", [Symbol]> {
  let summary = "named compiler-visible target capability";
  let description = [{
    Records a compiler-visible capability such as an ISA extension,
    microarchitecture feature, runtime/offload capability, or toolchain
    feature. Capability semantics are intentionally structured MLIR
    attributes, not comments or unparsed side strings.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                   OptionalAttr<StrAttr>:$id,
                   OptionalAttr<StrAttr>:$kind);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def KernelOp : TCRVExec_Op<"exec.kernel",
    [IsolatedFromAbove, NoRegionArguments, NoTerminator, SingleBlock, Symbol,
     SymbolTable]> {
  let summary = "execution kernel container";
  let description = [{
    Contains target declarations, capability objects, execution variants,
    dispatch, and fallback organization for one high-level kernel realization.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                   OptionalAttr<FlatSymbolRefAttr>:$target);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def VariantOp : TCRVExec_Op<"exec.variant",
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

def MemWindowOp : TCRVExec_Op<"exec.mem_window", [Symbol]> {
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

def RuntimeParamOp : TCRVExec_Op<"exec.runtime_param", [Symbol]> {
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

def HartParallelOp : TCRVExec_Op<"exec.hart_parallel",
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

def RegionOp : TCRVExec_Op<"exec.region",
    [NoRegionArguments, NoTerminator, SingleBlock]> {
  let summary = "extension-owned execution resource region";
  let description = [{
    Provides a single-block execution/resource organization region for
    extension-owned work. The core dialect records structured region metadata
    but does not interpret RVV, IME, offload, or future plugin legality here.
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$kind,
                   OptionalAttr<StrAttr>:$name,
                   OptionalAttr<StrAttr>:$purpose);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def DiagnosticOp : TCRVExec_Op<"exec.diagnostic", []> {
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
                   OptionalAttr<StrAttr>:$lowering_boundary,
                   OptionalAttr<ArrayAttr>:$required_capabilities,
                   OptionalAttr<ArrayAttr>:$selected_plan_metadata,
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

def DispatchOp : TCRVExec_Op<"exec.dispatch",
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

def DispatchCaseOp : TCRVExec_Op<"exec.case", []> {
  let summary = "compiler-visible dispatch case variant reference";
  let description = [{
    References a candidate execution variant considered by a surrounding
    tcrv.exec.dispatch. Optional condition, guard, and policy metadata are
    generic non-empty strings; extension-specific interpretation remains
    plugin-local. Optional runtime_guard_required is a typed compiler-owned
    marker that this case requires an explicit runtime dispatch-availability
    guard. Optional runtime_guard links executable dispatch control to a
    same-kernel tcrv.exec.runtime_param with the generic
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

def FallbackOp : TCRVExec_Op<"exec.fallback", []> {
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

#endif // TIANCHENRV_DIALECT_EXEC_IR_EXECOPS_TD
