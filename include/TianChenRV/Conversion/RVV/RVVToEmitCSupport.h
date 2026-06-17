#ifndef TIANCHENRV_CONVERSION_RVV_RVVTOEMITCSUPPORT_H
#define TIANCHENRV_CONVERSION_RVV_RVVTOEMITCSUPPORT_H

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir {
class MLIRContext;
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

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv

#endif // TIANCHENRV_CONVERSION_RVV_RVVTOEMITCSUPPORT_H
