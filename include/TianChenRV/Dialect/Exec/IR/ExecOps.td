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

  let arguments = (ins SymbolNameAttr:$sym_name);
  let assemblyFormat = "$sym_name attr-dict";
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

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def VariantOp : TCRVExec_Op<"exec.variant",
    [IsolatedFromAbove, NoRegionArguments, NoTerminator, SingleBlock, Symbol,
     SymbolTable]> {
  let summary = "plugin-proposed execution variant";
  let description = [{
    Contains one plugin-proposed execution variant. The origin plugin,
    required capabilities, preconditions, tuning metadata, cost metadata,
    and emission path are represented as MLIR attributes or nested
    extension operations.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                   OptionalAttr<StrAttr>:$origin,
                   OptionalAttr<ArrayAttr>:$requires);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def DispatchOp : TCRVExec_Op<"exec.dispatch",
    [NoRegionArguments, NoTerminator, SingleBlock]> {
  let summary = "capability-aware runtime dispatch organization";
  let description = [{
    Groups dispatch cases and fallback behavior. Dispatch conditions should be
    derived from structured capabilities, runtime probes, shape guards, dtype
    guards, cost thresholds, or user policy.
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def FallbackOp : TCRVExec_Op<"exec.fallback", []> {
  let summary = "fallback variant reference";
  let description = [{
    References a fallback variant selected when capability, legality, runtime,
    or policy conditions reject preferred variants.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target);
  let assemblyFormat = "$target attr-dict";
  let hasVerifier = 1;
}

#endif // TIANCHENRV_DIALECT_EXEC_IR_EXECOPS_TD
