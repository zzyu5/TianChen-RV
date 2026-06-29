#ifndef TIANCHENRV_CONVERSION_RVV_RVVTOEMITCSUPPORT_H
#define TIANCHENRV_CONVERSION_RVV_RVVTOEMITCSUPPORT_H

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
class MLIRContext;
class OpBuilder;
class PatternRewriter;
} // namespace mlir

// Free-function support helpers extracted from RVVToEmitC.cpp. These are the
// pure intrinsic-name builders + type/dtype/comment helpers: each takes explicit
// arguments and touches no VariantToEmitCFunc class state or anonymous-namespace
// mutable state, so they live in this stable support translation unit. The
// emitted-C output is byte-identical to when they lived inline (a pure code
// move): the conversion fingerprint and lit suite prove it.
namespace tianchenrv {
namespace conversion {
namespace rvv {
namespace detail {

//===----------------------------------------------------------------------===//
// Pure C-type derivation.
//===----------------------------------------------------------------------===//

mlir::Type emitCTypeForCTypeSpelling(mlir::MLIRContext *context,
                                     llvm::StringRef cType);

//===----------------------------------------------------------------------===//
// Pure SEW/LMUL/dtype intrinsic name manglers.
//===----------------------------------------------------------------------===//

std::string riscvIntrinsicName(llvm::StringRef mnemonic, unsigned sew,
                               llvm::StringRef lmul, llvm::StringRef dtype);

std::string riscvCompareIntrinsicName(llvm::StringRef mnemonic, unsigned sew,
                                      llvm::StringRef lmul,
                                      llvm::StringRef dtype, unsigned maskBits);

std::string riscvMaskComposeIntrinsicName(llvm::StringRef mnemonic,
                                          unsigned maskBits);

std::string riscvReductionIntrinsicName(llvm::StringRef mnemonic, unsigned sew,
                                        llvm::StringRef lmul,
                                        llvm::StringRef dtype);

//===----------------------------------------------------------------------===//
// L0b widening/convert-core name manglers (the hand-rolled Twine-concat gap the
// emitter consolidation cites). Each rebuilds the full "__riscv_..." callee from
// the WIDENED-result <dtype><lmul> facts; byte-identical to the inline forms.
//===----------------------------------------------------------------------===//

std::string riscvWideningMulIntrinsicName(llvm::StringRef mnemonic,
                                          llvm::StringRef variant,
                                          llvm::StringRef dstDtype,
                                          llvm::StringRef dstLmul);

std::string riscvWideningMacIntrinsicName(llvm::StringRef variant,
                                          llvm::StringRef dstDtype,
                                          llvm::StringRef dstLmul);

// The widening/non-widening lane-0 reduction intrinsic name (the promoted
// standalone-reduction mangler):
//   __riscv_<mnemonic>_vs_<srcDtype><srcLmul>_<resultDtype>m1
// The result LMUL is ALWAYS m1 (the reduction lands its scalar in lane 0 of an
// m1 destination), so the trailing "m1" is invariant. Subsumes
// vredsum/vwredsum/vwredsumu/vfwredusum/vfredmax (the source/result dtype differ
// for the widening forms, vwredsum_vs_i16m2_i32m1). 4-arg byte-exact shape.
std::string riscvWideningReductionIntrinsicName(llvm::StringRef mnemonic,
                                                llvm::StringRef srcDtype,
                                                llvm::StringRef srcLmul,
                                                llvm::StringRef resultDtype);

std::string riscvWideningAddIntrinsicName(llvm::StringRef variant,
                                          llvm::StringRef dstDtype,
                                          llvm::StringRef dstLmul);

std::string riscvFloatWideningMulIntrinsicName(llvm::StringRef dstDtype,
                                               llvm::StringRef dstLmul);

std::string riscvFloatWideningConvertIntrinsicName(llvm::StringRef dstDtype,
                                                   llvm::StringRef dstLmul);

std::string riscvNarrowingConvertIntrinsicName(llvm::StringRef dstDtype,
                                               llvm::StringRef dstLmul);

std::string riscvFloatNarrowToIntConvertIntrinsicName(llvm::StringRef dstDtype,
                                                      llvm::StringRef dstLmul);

std::string riscvZeroExtendIntrinsicName(unsigned factor,
                                         llvm::StringRef dstDtype,
                                         llvm::StringRef dstLmul);

//===----------------------------------------------------------------------===//
// L0b non-widening rider manglers (today hand-rolled Twine-concat or
// class-inline closures). Byte-identical to the inline forms.
//===----------------------------------------------------------------------===//

std::string riscvScalarImmediateIntrinsicName(llvm::StringRef mnemonic,
                                              llvm::StringRef dtype,
                                              llvm::StringRef lmul);

std::string riscvScalarExtractIntrinsicName(llvm::StringRef dtype,
                                            llvm::StringRef lmul);

std::string riscvFloatScalarExtractIntrinsicName(llvm::StringRef dtype,
                                                 llvm::StringRef lmul);

std::string riscvReinterpretIntrinsicName(llvm::StringRef fromDtype,
                                          llvm::StringRef fromLmul,
                                          llvm::StringRef toDtype,
                                          llvm::StringRef toLmul);

std::string riscvUnaryIntrinsicName(llvm::StringRef mnemonic,
                                    llvm::StringRef dtype, llvm::StringRef lmul);

std::string riscvIotaIntrinsicName(llvm::StringRef dtype, llvm::StringRef lmul);

std::string riscvVsetvlmaxIntrinsicName(unsigned sew, llvm::StringRef lmul);

std::string riscvMAccIntrinsicName(unsigned sew, llvm::StringRef lmul,
                                   llvm::StringRef dtype);

std::string riscvWideningAccumulateIntrinsicName(unsigned accSEW,
                                                 llvm::StringRef accLmul,
                                                 llvm::StringRef accDtype);

std::string riscvIndexScaleIntrinsicName(llvm::StringRef idtype,
                                         llvm::StringRef lmul);

std::string riscvIndexedMemoryIntrinsicName(llvm::StringRef mnemonic,
                                            unsigned indexEEW,
                                            llvm::StringRef dtype,
                                            llvm::StringRef lmul);

std::string riscvMaskNonzeroIntrinsicName(unsigned sew, llvm::StringRef lmul,
                                          llvm::StringRef dtype,
                                          unsigned maskBits);

std::string riscvMaskedLoadIntrinsicName(unsigned sew, llvm::StringRef lmul,
                                         llvm::StringRef dtype);

std::string riscvMaskedStoreIntrinsicName(unsigned sew, llvm::StringRef lmul,
                                          llvm::StringRef dtype);

std::string riscvMaskedStridedLoadIntrinsicName(unsigned sew,
                                                llvm::StringRef lmul,
                                                llvm::StringRef dtype);

std::string riscvMaskedStridedStoreIntrinsicName(unsigned sew,
                                                 llvm::StringRef lmul,
                                                 llvm::StringRef dtype);

std::string riscvMaskedIndexedLoadIntrinsicName(unsigned indexEEW,
                                                llvm::StringRef dtype,
                                                llvm::StringRef lmul);

std::string riscvMaskedIndexedStoreIntrinsicName(unsigned indexEEW,
                                                 llvm::StringRef dtype,
                                                 llvm::StringRef lmul);

//===----------------------------------------------------------------------===//
// Segment2 (interleaved 2-field) intrinsic + tuple-type name manglers.
//===----------------------------------------------------------------------===//

std::string riscvSegment2TupleCType(unsigned sew, llvm::StringRef lmul);

std::string riscvSegment2TupleCreateIntrinsicName(llvm::StringRef dtype,
                                                  llvm::StringRef lmul);

std::string riscvSegment2FieldExtractIntrinsicName(llvm::StringRef dtype,
                                                   llvm::StringRef lmul);

std::string riscvSegment2LoadIntrinsicName(unsigned sew, llvm::StringRef lmul,
                                           llvm::StringRef dtype);

std::string riscvSegment2StoreIntrinsicName(unsigned sew, llvm::StringRef lmul,
                                            llvm::StringRef dtype);

std::string riscvMaskedSegment2LoadIntrinsicName(unsigned sew,
                                                 llvm::StringRef lmul,
                                                 llvm::StringRef dtype);

std::string riscvMaskedSegment2StoreIntrinsicName(unsigned sew,
                                                  llvm::StringRef lmul,
                                                  llvm::StringRef dtype);

//===----------------------------------------------------------------------===//
// Type/dtype helpers over the typed !tcrv_rvv.vector facts.
//===----------------------------------------------------------------------===//

unsigned maskWidthForConfig(unsigned sew, llvm::StringRef lmul);

bool isUnsignedVector(::tianchenrv::tcrv::rvv::VectorType type);

llvm::StringRef vectorDType(::tianchenrv::tcrv::rvv::VectorType type);

bool isFloatVector(::tianchenrv::tcrv::rvv::VectorType type);

llvm::StringRef vectorScalarCType(::tianchenrv::tcrv::rvv::VectorType type);

bool bufferPointeeMatchesVectorElement(
    mlir::Value bufferValue, ::tianchenrv::tcrv::rvv::VectorType vectorType);

//===----------------------------------------------------------------------===//
// Provenance comments.
//===----------------------------------------------------------------------===//

std::string routeSourceComment(llvm::StringRef opName, llvm::StringRef role);

std::string stepComment(llvm::StringRef opName, llvm::StringRef role,
                        llvm::StringRef callee);

std::string localVariableComment(llvm::StringRef varName, llvm::StringRef opName,
                                 llvm::StringRef role);

std::string assignComment(llvm::StringRef target, llvm::StringRef opName,
                          llvm::StringRef role);

//===----------------------------------------------------------------------===//
// Op emission helper: the shared `verbatim step-comment + call_opaque` idiom.
//===----------------------------------------------------------------------===//

// Emits, byte-for-byte, the hand-spliced pair the per-kernel emitters open-code
// today:
//   verbatim "<stepComment(opName, role, callee)>"
//   %r = call_opaque "<callee>"(operands) : (...) -> resultType
// where callee = ("__riscv_" + mnemonic + "_" + suffix).str(). The returned
// value is the call's single result. This is the L0 consolidation seed: it
// changes ZERO emitted bytes (the conversion fixtures + lit suite prove it).
mlir::Value emitVCall(mlir::PatternRewriter &rewriter, mlir::Location loc,
                      mlir::Type resultType, llvm::StringRef mnemonic,
                      llvm::StringRef suffix, mlir::ValueRange operands,
                      llvm::StringRef opName, llvm::StringRef role);

// The literal/pointer/accumulator-INTERLEAVE variant: the hand-spliced tricky
// sites build an interior op (immediate LiteralOp / AddOp pointer / SubOp
// remaining / SubscriptOp+LoadOp scalar / accumulator LoadOp) BETWEEN the
// step-comment VerbatimOp and the call. Plain emitVCall pre-evaluates its
// operands as C++ args (so any op built to form the ValueRange prints BEFORE the
// comment), reordering the emitc-dialect dump. This variant defers operand
// construction to AFTER the verbatim: the body order is, EXACTLY,
//   verbatim "<stepComment(opName, role, callee)>"   // FIRST
//   <buildOperands(builder, loc)>                    // interior, source order
//   %r = call_opaque "<callee>"(ops) : (...) -> resultType   // LAST
// where callee = ("__riscv_" + mnemonic + "_" + suffix).str(). This ONE
// callback contract subsumes every literal/pointer/scalar/accumulator interleave
// shape (the interior op type is irrelevant -- the contract is positional). The
// returned value is the call's single result.
mlir::Value emitVCallBuilt(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Type resultType,
    llvm::StringRef mnemonic, llvm::StringRef suffix, llvm::StringRef opName,
    llvm::StringRef role,
    llvm::function_ref<llvm::SmallVector<mlir::Value>(mlir::OpBuilder &,
                                                      mlir::Location)>
        buildOperands);

// The pure-void store sibling of emitVCall: identical (same callee mangle, same
// step comment) but emits a TypeRange{} (no result) call and returns nothing.
// For the unit/strided/segment stores whose operands are all pre-built.
void emitVCallVoid(mlir::PatternRewriter &rewriter, mlir::Location loc,
                   llvm::StringRef mnemonic, llvm::StringRef suffix,
                   mlir::ValueRange operands, llvm::StringRef opName,
                   llvm::StringRef role);

// The void + interleave sibling: the emitVCallBuilt contract (verbatim FIRST,
// interior buildOperands, call LAST) for the stores that build an interior op
// (e.g. an inline VL literal) between the comment and the call. No result.
void emitVCallVoidBuilt(
    mlir::PatternRewriter &rewriter, mlir::Location loc, llvm::StringRef mnemonic,
    llvm::StringRef suffix, llvm::StringRef opName, llvm::StringRef role,
    llvm::function_ref<llvm::SmallVector<mlir::Value>(mlir::OpBuilder &,
                                                      mlir::Location)>
        buildOperands);

// The raw-callee + comment-override sibling: passes `fullCallee` VERBATIM (no
// "__riscv_" + mnemonic + "_" + suffix synthesis), for the non-manglable seams
// (libm cosf/sinf/sqrtf, the fp16 cast-deref read, the codebook table/kmask
// loads). When `commentOverride` is set the VerbatimOp payload is
// stepComment(opName, role, *commentOverride) instead of the callee, preserving
// the custom semantic label (e.g. "codebook_table_load") byte-for-byte. The
// returned value is the call's single result.
mlir::Value emitOpaqueCall(mlir::PatternRewriter &rewriter, mlir::Location loc,
                           mlir::Type resultType, llvm::StringRef fullCallee,
                           mlir::ValueRange operands, llvm::StringRef opName,
                           llvm::StringRef role,
                           std::optional<llvm::StringRef> commentOverride =
                               std::nullopt);

// The void sibling of emitOpaqueCall: passes `fullCallee` VERBATIM and emits a
// TypeRange{} (no result) call. For the unit/strided/segment stores whose callee
// is already a full mangler string (riscvIntrinsicName("vse", ...)) and whose
// operands are all pre-built.
void emitOpaqueCallVoid(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        llvm::StringRef fullCallee, mlir::ValueRange operands,
                        llvm::StringRef opName, llvm::StringRef role);

// The interleave sibling of emitOpaqueCall: the emitVCallBuilt contract
// (verbatim FIRST, interior buildOperands in source order, call LAST) but with a
// full pre-mangled `fullCallee` instead of the "__riscv_" + mnemonic + suffix
// synthesis. For the mangler-callee sites that build an interior op (e.g. the
// inline seed/scale LiteralOp) between the comment and the call. When
// `commentOverride` is set the VerbatimOp payload is the override label instead
// of the callee (the libm sqrtf "scale" seam: callee=="sqrtf", comment=="scale",
// the meanPlusEps AddOp interleaved).
mlir::Value emitOpaqueCallBuilt(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Type resultType,
    llvm::StringRef fullCallee, llvm::StringRef opName, llvm::StringRef role,
    llvm::function_ref<llvm::SmallVector<mlir::Value>(mlir::OpBuilder &,
                                                      mlir::Location)>
        buildOperands,
    std::optional<llvm::StringRef> commentOverride = std::nullopt);

//===----------------------------------------------------------------------===//
// Single-source i8 -> i16 -> i32 widening-chain LMUL derivation.
//===----------------------------------------------------------------------===//

// The integer-core widening ladder rungs for a base i8 LMUL anchor:
//   l8  = the base i8 load anchor,
//   l16 = the i16 product, ONE EMUL step wider,
//   l32 = the i32 accumulator, TWO steps wider,
// up the x2 ladder mf4 -> mf2 -> m1 -> m2 -> m4 -> m8. stripWidth is the i8 lane
// count the base LMUL spans at the canonical VLEN=128 integer-core anchor
// (mf2=8, m1=16, m2=32) and foldGroups = stripWidth/8 (the lane groups the
// integer fold-back collapses back to the canonical 8 before the fp fold).
//
// This is the ONE source of truth the q4_K / q6_K block-dot integer cores and
// the FP4 codebook emitters share. It replaces the per-site inline derivations,
// one of which silently mishandled the "m2" base (l16/l32 collapsing to the
// "m1" branch), removing the latent m1/m2 widening-chain divergence seam. The
// emitted C is byte-identical for the {mf2,m1} bases that occur in-tree.
struct WideningChain {
  std::string l8;
  std::string l16;
  std::string l32;
  int64_t stripWidth = 0;
  int64_t foldGroups = 0;
};

WideningChain deriveWideningChain(llvm::StringRef base);

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv

#endif // TIANCHENRV_CONVERSION_RVV_RVVTOEMITCSUPPORT_H
