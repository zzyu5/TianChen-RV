#include "Weft/Plugin/RVV/RVVCanonicalProblemConstruction.h"

#include "Weft/Plugin/RVV/RVVMonolithicBlockDotFamily.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Errc.h"

namespace weft::plugin::rvv {
namespace {

llvm::Error makeCanonicalProblemConstructionError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("RVV canonical-problem construction rejected: ") + message,
      llvm::errc::invalid_argument);
}

} // namespace

bool isRVVCanonicalProblemSupported(mlir::Operation *problem) {
  return llvm::isa_and_nonnull<weft::exec::I32VectorBinaryProblemOp,
                               weft::exec::I32VectorCompareSelectProblemOp,
                               weft::exec::I8WideningDotReduceProblemOp,
                               weft::exec::PackedI4Q8DotProblemOp,
                               weft::exec::CodebookI4Q8DotProblemOp,
                               weft::exec::QuantizedBlockDotProblemOp>(problem);
}

llvm::Expected<std::string>
deriveRVVCanonicalProblemVariantName(mlir::Operation *problem) {
  if (!problem)
    return makeCanonicalProblemConstructionError("requires exact canonical P");

  return llvm::TypeSwitch<mlir::Operation *, llvm::Expected<std::string>>(problem)
      .Case<weft::exec::I32VectorBinaryProblemOp>([](auto typed) {
        return (llvm::Twine("rvv_vector_") + typed.getKind()).str();
      })
      .Case<weft::exec::I32VectorCompareSelectProblemOp>([](auto typed) {
        llvm::StringRef prefix = typed.getRhsForm() == "runtime-scalar"
                                     ? "rvv_vector_runtime_scalar_cmp_select_"
                                     : "rvv_vector_cmp_select_";
        return (llvm::Twine(prefix) + typed.getPredicate()).str();
      })
      .Case<weft::exec::I8WideningDotReduceProblemOp>([](auto typed) {
        return std::string(typed.getDequantizeToF32()
                               ? "rvv_widening_dot_reduce_dequantize_i8"
                               : "rvv_widening_dot_reduce_i8");
      })
      .Case<weft::exec::PackedI4Q8DotProblemOp>([](auto) {
        return std::string("rvv_packed_i4_offset_binary_dot_i8");
      })
      .Case<weft::exec::CodebookI4Q8DotProblemOp>([](auto) {
        return std::string("rvv_codebook_gather_dot_i8");
      })
      .Case<weft::exec::QuantizedBlockDotProblemOp>([](auto typed)
          -> llvm::Expected<std::string> {
        const MonolithicBlockDotOpEntry *entry =
            findMonolithicBlockDotProblemEntry(
                typed.getWeightEncoding(), typed.getActivationEncoding(),
                typed.getTopology(),
                static_cast<std::int64_t>(typed.getQk()),
                static_cast<std::int64_t>(typed.getWeightBlockStride()),
                static_cast<std::int64_t>(typed.getActivationBlockStride()));
        if (entry)
          return entry->variantSymbol.str();
        return makeCanonicalProblemConstructionError(
            llvm::Twine("has no exact RVV block-dot formula for weight='") +
            typed.getWeightEncoding() + "', activation='" +
            typed.getActivationEncoding() + "', topology='" +
            typed.getTopology() + "'");
      })
      .Default([](mlir::Operation *unsupported)
          -> llvm::Expected<std::string> {
        return makeCanonicalProblemConstructionError(
            llvm::Twine("unsupported canonical problem '") +
            unsupported->getName().getStringRef() + "'");
      });
}

llvm::Error verifyRVVCanonicalProblemCandidate(
    weft::exec::VariantOp variant, mlir::Operation *problem,
    const support::TargetCapabilitySet &capabilities) {
  llvm::Expected<std::string> expected =
      deriveRVVCanonicalProblemVariantName(problem);
  if (!expected)
    return expected.takeError();
  if (!variant)
    return makeCanonicalProblemConstructionError(
        "requires a materialized selected candidate");
  if (variant.getSymName() != *expected)
    return makeCanonicalProblemConstructionError(
        llvm::Twine("candidate @") + variant.getSymName() +
        " does not match exact-P-derived candidate @" + *expected);

  llvm::Expected<RVVSelectedTargetCapabilityFacts> selected =
      collectRVVSelectedTargetCapabilityFacts(
          variant, capabilities, "RVV canonical-problem candidate");
  if (!selected)
    return selected.takeError();
  return llvm::Error::success();
}

llvm::Error constructRVVCanonicalProblemBody(
    weft::exec::VariantOp variant, mlir::Operation *problem,
    const support::TargetCapabilitySet &capabilities) {
  if (llvm::Error error =
          verifyRVVCanonicalProblemCandidate(variant, problem, capabilities))
    return error;

  llvm::Expected<RVVSelectedTargetCapabilityFacts> selected =
      collectRVVSelectedTargetCapabilityFacts(
          variant, capabilities, "RVV canonical-problem construction");
  if (!selected)
    return selected.takeError();

  return llvm::TypeSwitch<mlir::Operation *, llvm::Error>(problem)
      .Case<weft::exec::I32VectorBinaryProblemOp,
            weft::exec::I32VectorCompareSelectProblemOp>(
          [&](auto) {
            return constructRVVVectorProblemBody(variant, problem, *selected);
          })
      .Case<weft::exec::I8WideningDotReduceProblemOp>([&](auto typed) {
        return typed.getDequantizeToF32()
                   ? constructRVVDequantDotProblemBody(variant, typed, *selected)
                   : constructRVVReductionProblemBody(variant, typed, *selected);
      })
      .Case<weft::exec::PackedI4Q8DotProblemOp>([&](auto typed) {
        return constructRVVPackedI4Q8DotProblemBody(variant, typed, *selected);
      })
      .Case<weft::exec::CodebookI4Q8DotProblemOp>([&](auto typed) {
        return constructRVVCodebookI4Q8DotProblemBody(variant, typed, *selected);
      })
      .Case<weft::exec::QuantizedBlockDotProblemOp>([&](auto typed) {
        return constructRVVQuantizedBlockDotProblemBody(variant, typed,
                                                        *selected);
      })
      .Default([](mlir::Operation *unsupported) {
        return makeCanonicalProblemConstructionError(
            llvm::Twine("unsupported exact problem '") +
            unsupported->getName().getStringRef() + "'");
      });
}

} // namespace weft::plugin::rvv
