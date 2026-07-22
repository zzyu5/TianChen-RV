#include "Weft/Plugin/IME/IMEFormulaConstruction.h"

#include "Weft/Dialect/IME/IR/IMEDialect.h"
#include "Weft/Plugin/IME/IMEExtensionPlugin.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace weft::plugin::ime {
namespace {

struct IMEWideComputationDecision {
  int64_t njw = 1;
  int64_t vlenBits = 0;
  int64_t inputFragmentVRegs = 0;
  int64_t accumulatorVRegs = 0;
  int64_t vregFloor = 0;
};

struct IMEWideCandidate {
  int64_t njw;
  bool selectable;
};

/// W4 is retained as a known measured-negative computation candidate.  It is
/// deliberately unavailable to construction; an emitter never sees or
/// reinterprets this candidate set.
static constexpr IMEWideCandidate kIMEWideCandidates[] = {
    {1, true}, {2, true}, {4, false}};

constexpr int64_t kRVVVectorRegisterFileSize = 32;

bool isIMEFinalBody(mlir::Operation *op) {
  return llvm::isa<weft::ime::MMAOp, weft::ime::MMAUOp,
                   weft::ime::MMASUOp, weft::ime::MMAUSOp,
                   weft::ime::MMASlideOp, weft::ime::MatMulOp,
                   weft::ime::Q40MatMulTileOp, weft::ime::Q80MatMulTileOp,
                   weft::ime::Q4KMatMulTileOp>(op);
}

mlir::DictionaryAttr makeIMEFinalPlan(
    mlir::MLIRContext *context,
    llvm::ArrayRef<std::pair<llvm::StringRef, int64_t>> integerFields) {
  llvm::SmallVector<mlir::NamedAttribute, 12> attrs;
  attrs.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(context, "formula_id"),
      mlir::StringAttr::get(context, kIMEConstructionFormulaID)));
  for (auto [name, value] : integerFields)
    attrs.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(context, name),
        mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), value)));
  return mlir::DictionaryAttr::get(context, attrs);
}

mlir::LogicalResult attachIMEFinalPlan(mlir::Operation *op,
                                       mlir::DictionaryAttr expected) {
  auto existing =
      op->getAttrOfType<mlir::DictionaryAttr>(kIMEFinalPlanAttrName);
  if (op->hasAttr(kIMEFinalPlanAttrName) && !existing)
    return op->emitError("IME final computation plan must be a dictionary");
  if (existing && existing != expected)
    return op->emitError(
        "IME final computation plan is partial, stale, or conflicts with the "
        "typed body");
  op->setAttr(kIMEFinalPlanAttrName, expected);
  return mlir::success();
}

mlir::FailureOr<mlir::DictionaryAttr>
requireIMEFinalPlan(mlir::Operation *op) {
  auto plan = op->getAttrOfType<mlir::DictionaryAttr>(kIMEFinalPlanAttrName);
  if (!plan)
    return op->emitError(
        "IME artifact projection requires a family-constructed final plan");
  auto formula = plan.getAs<mlir::StringAttr>("formula_id");
  if (!formula || formula.getValue() != kIMEConstructionFormulaID)
    return op->emitError("IME final plan has the wrong formula owner");
  return plan;
}

mlir::FailureOr<int64_t> readPlanInteger(mlir::DictionaryAttr plan,
                                         llvm::StringRef name,
                                         mlir::Operation *op) {
  auto value = plan.getAs<mlir::IntegerAttr>(name);
  if (!value)
    return op->emitError() << "IME final plan is missing integer field '"
                           << name << "'";
  return value.getInt();
}

mlir::FailureOr<int64_t> readIMEConstructionVlenBits(mlir::Operation *tile) {
  auto kernel = tile->getParentOfType<weft::exec::KernelOp>();
  if (!kernel)
    return tile->emitError(
        "IME construction requires an enclosing weft.exec.kernel");
  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities) {
    tile->emitError() << llvm::toString(capabilities.takeError());
    return mlir::failure();
  }
  const support::CapabilityDescriptor *imeCapability =
      capabilities->lookupProviderByID(getIMEExtensionCapabilityID());
  if (!imeCapability || !imeCapability->isAvailable())
    return tile->emitError(
        "IME construction requires an available canonical IME capability");

  mlir::Attribute vlen = imeCapability->getPropertyAttribute("vlen_bits");
  int64_t value = 0;
  if (auto integer = llvm::dyn_cast_if_present<mlir::IntegerAttr>(vlen))
    value = integer.getInt();
  else if (auto text = llvm::dyn_cast_if_present<mlir::StringAttr>(vlen)) {
    long long parsed = 0;
    if (!text.getValue().getAsInteger(10, parsed))
      value = parsed;
  }
  if (value <= 0)
    return tile->emitError(
        "IME construction requires a positive typed vlen_bits capability fact");
  return value;
}

IMEWideComputationDecision constructWideVmadotPlan(
    int64_t vlenBits, int64_t macM, int64_t macN, int64_t macK,
    int64_t elemInBits, int64_t accumBits, llvm::StringRef weightFormat) {
  IMEWideComputationDecision decision;
  decision.vlenBits = vlenBits;

  auto ceilDiv = [](int64_t numerator, int64_t denominator) -> int64_t {
    return denominator > 0 ? (numerator + denominator - 1) / denominator : 1;
  };
  const int64_t fragmentBits = macM * macK * elemInBits;
  const int64_t accumulatorBits = macM * macN * accumBits;
  decision.inputFragmentVRegs =
      std::max<int64_t>(1, ceilDiv(fragmentBits, vlenBits));
  decision.accumulatorVRegs =
      std::max<int64_t>(1, ceilDiv(accumulatorBits, vlenBits));

  auto floorFor = [&](int64_t njw) {
    return decision.inputFragmentVRegs * (1 + njw) +
           decision.accumulatorVRegs * njw;
  };
  decision.vregFloor = floorFor(1);

  // q4_K's two-level scale/min fold is a measured-negative wide-MAC shape.
  // This is a construction policy input; the artifact layer receives only the
  // resulting NJW=1 computation plan and cannot overturn it.
  if (weightFormat == "q4_K")
    return decision;

  // The current typed wide leaf uses one e8,m1 load per input fragment.
  if (decision.inputFragmentVRegs != 1)
    return decision;

  for (const IMEWideCandidate &candidate : kIMEWideCandidates) {
    const int64_t floor = floorFor(candidate.njw);
    if (candidate.selectable && floor <= kRVVVectorRegisterFileSize &&
        candidate.njw > decision.njw) {
      decision.njw = candidate.njw;
      decision.vregFloor = floor;
    }
  }
  return decision;
}

mlir::LogicalResult constructIMESimplePlan(mlir::Operation *op) {
  return attachIMEFinalPlan(op,
                            makeIMEFinalPlan(op->getContext(), /*fields=*/{}));
}

template <typename TileOp>
mlir::LogicalResult constructIMEQuantTilePlan(TileOp tile) {
  mlir::Block &body = tile.getBody().front();
  auto macLeaves = body.template getOps<weft::ime::VmadotMacLeafOp>();
  if (macLeaves.empty())
    return tile.emitError(
        "IME quant tile construction requires one typed vmadot MAC leaf");

  weft::ime::VmadotMacLeafOp macLeaf = *macLeaves.begin();
  const int64_t macK = macLeaf.getMacK();
  const int64_t fragmentCount = macK > 0 ? tile.getMatK() / macK : 1;
  const bool macBatched = fragmentCount >= 2;

  auto vlenBits = readIMEConstructionVlenBits(tile.getOperation());
  if (mlir::failed(vlenBits))
    return mlir::failure();
  IMEWideComputationDecision wide = constructWideVmadotPlan(
      *vlenBits, tile.getMacM(), tile.getMacN(), tile.getMacK(),
      tile.getElemInBits(), tile.getAccumBits(), tile.getWeightFormat());

  llvm::SmallVector<std::pair<llvm::StringRef, int64_t>, 10> fields = {
      {"mat_m", tile.getMatM()},
      {"mat_n", tile.getMatN()},
      {"mat_k", tile.getMatK()},
      {"mac_batched", macBatched ? 1 : 0},
      {"wide_njw", wide.njw},
      {"wide_vlen_bits", wide.vlenBits},
      {"wide_input_fragment_vregs", wide.inputFragmentVRegs},
      {"wide_accumulator_vregs", wide.accumulatorVRegs},
      {"wide_vreg_floor", wide.vregFloor},
  };
  return attachIMEFinalPlan(tile.getOperation(),
                            makeIMEFinalPlan(tile.getContext(), fields));
}

mlir::LogicalResult validateIMEConstructionContext(
    mlir::Operation *op, const IMEExtensionPlugin &plugin) {
  auto kernel = op->getParentOfType<weft::exec::KernelOp>();
  auto sourceKernel = op->getAttrOfType<mlir::StringAttr>("source_kernel");
  if (!kernel || !sourceKernel ||
      sourceKernel.getValue() != kernel.getSymName())
    return op->emitError(
        "IME construction requires source_kernel to match an enclosing "
        "weft.exec.kernel");

  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities) {
    op->emitError() << llvm::toString(capabilities.takeError());
    return mlir::failure();
  }
  const support::CapabilityDescriptor *imeCapability =
      capabilities->lookupProviderByID(getIMEExtensionCapabilityID());
  if (!imeCapability || !imeCapability->isAvailable())
    return op->emitError(
        "IME construction requires an available canonical IME capability");

  auto selectedVariant =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
  weft::exec::VariantOp variant;
  unsigned matches = 0;
  if (selectedVariant) {
    kernel.walk([&](weft::exec::VariantOp candidate) {
      if (candidate.getSymName() == selectedVariant.getValue()) {
        variant = candidate;
        ++matches;
      }
    });
  }
  if (!selectedVariant || matches != 1)
    return op->emitError(
        "IME construction requires selected_variant to resolve exactly once "
        "in the enclosing kernel");

  VariantLegalityRequest legality(variant, kernel, *capabilities);
  if (llvm::Error error = plugin.verifyVariantLegality(legality)) {
    op->emitError() << llvm::toString(std::move(error));
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult constructIMEFormulaPlans(mlir::ModuleOp module) {
  static const IMEExtensionPlugin plugin;
  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::Operation *op) {
    if (mlir::failed(result))
      return mlir::WalkResult::interrupt();

    if (isIMEFinalBody(op) &&
        mlir::failed(validateIMEConstructionContext(op, plugin))) {
      result = mlir::failure();
      return mlir::WalkResult::interrupt();
    }

    if (llvm::isa<weft::ime::MMAOp, weft::ime::MMAUOp,
                  weft::ime::MMASUOp, weft::ime::MMAUSOp,
                  weft::ime::MMASlideOp>(op)) {
      result = constructIMESimplePlan(op);
    } else if (auto matmul = llvm::dyn_cast<weft::ime::MatMulOp>(op)) {
      llvm::SmallVector<std::pair<llvm::StringRef, int64_t>, 3> fields = {
          {"mat_m", matmul.getMatM()},
          {"mat_n", matmul.getMatN()},
          {"mat_k", matmul.getMatK()},
      };
      result = attachIMEFinalPlan(
          op, makeIMEFinalPlan(matmul.getContext(), fields));
    } else if (auto tile = llvm::dyn_cast<weft::ime::Q40MatMulTileOp>(op)) {
      result = constructIMEQuantTilePlan(tile);
    } else if (auto tile = llvm::dyn_cast<weft::ime::Q80MatMulTileOp>(op)) {
      result = constructIMEQuantTilePlan(tile);
    } else if (auto tile = llvm::dyn_cast<weft::ime::Q4KMatMulTileOp>(op)) {
      result = constructIMEQuantTilePlan(tile);
    } else if (op->getName().getDialectNamespace() ==
               weft::ime::WEFTIMEDialect::getDialectNamespace()) {
      // Region bricks are mechanisms owned by a final quantized tile.  A
      // standalone brick is not a construction-complete body.
      if (!op->getParentOfType<weft::ime::Q40MatMulTileOp>() &&
          !op->getParentOfType<weft::ime::Q80MatMulTileOp>() &&
          !op->getParentOfType<weft::ime::Q4KMatMulTileOp>()) {
        op->emitError(
            "standalone IME mechanism op has no final construction owner");
        result = mlir::failure();
      }
    }
    return mlir::failed(result) ? mlir::WalkResult::interrupt()
                                : mlir::WalkResult::advance();
  });
  return result;
}

bool hasIMEConstructedFinalBody(weft::exec::VariantOp variant) {
  auto kernel = variant->getParentOfType<weft::exec::KernelOp>();
  if (!kernel)
    return false;
  bool found = false;
  kernel.walk([&](mlir::Operation *op) {
    if (!isIMEFinalBody(op) ||
        !isOperationSelectedForVariant(op, variant))
      return mlir::WalkResult::advance();
    auto plan = op->getAttrOfType<mlir::DictionaryAttr>(kIMEFinalPlanAttrName);
    auto formula = plan ? plan.getAs<mlir::StringAttr>("formula_id")
                        : mlir::StringAttr();
    if (!formula || formula.getValue() != kIMEConstructionFormulaID)
      return mlir::WalkResult::advance();
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

mlir::LogicalResult requireIMESimpleComputationPlan(mlir::Operation *op) {
  return mlir::failed(requireIMEFinalPlan(op)) ? mlir::failure()
                                               : mlir::success();
}

mlir::FailureOr<IMEMatMulComputationPlan>
readIMEMatMulComputationPlan(mlir::Operation *op) {
  auto plan = requireIMEFinalPlan(op);
  if (mlir::failed(plan))
    return mlir::failure();
  auto matM = readPlanInteger(*plan, "mat_m", op);
  auto matN = readPlanInteger(*plan, "mat_n", op);
  auto matK = readPlanInteger(*plan, "mat_k", op);
  if (mlir::failed(matM) || mlir::failed(matN) || mlir::failed(matK))
    return mlir::failure();
  if (*matM <= 0 || *matN <= 0 || *matK <= 0)
    return op->emitError("IME final computation dimensions must be positive");
  return IMEMatMulComputationPlan{*matM, *matN, *matK};
}

mlir::FailureOr<IMEQuantComputationPlan>
readIMEQuantComputationPlan(mlir::Operation *op) {
  auto matmul = readIMEMatMulComputationPlan(op);
  if (mlir::failed(matmul))
    return mlir::failure();
  auto plan = requireIMEFinalPlan(op);
  if (mlir::failed(plan))
    return mlir::failure();

  auto batched = readPlanInteger(*plan, "mac_batched", op);
  auto njw = readPlanInteger(*plan, "wide_njw", op);
  auto vlen = readPlanInteger(*plan, "wide_vlen_bits", op);
  auto inputVRegs =
      readPlanInteger(*plan, "wide_input_fragment_vregs", op);
  auto accumulatorVRegs =
      readPlanInteger(*plan, "wide_accumulator_vregs", op);
  auto vregFloor = readPlanInteger(*plan, "wide_vreg_floor", op);
  if (mlir::failed(batched) || mlir::failed(njw) || mlir::failed(vlen) ||
      mlir::failed(inputVRegs) || mlir::failed(accumulatorVRegs) ||
      mlir::failed(vregFloor))
    return mlir::failure();
  if ((*batched != 0 && *batched != 1) ||
      (*njw != 1 && *njw != 2 && *njw != 4) || *vlen <= 0 ||
      *inputVRegs <= 0 || *accumulatorVRegs <= 0 || *vregFloor <= 0)
    return op->emitError("IME quantized final computation plan is malformed");

  IMEQuantComputationPlan result;
  result.matM = matmul->matM;
  result.matN = matmul->matN;
  result.matK = matmul->matK;
  result.macBatched = *batched != 0;
  result.wideNJW = *njw;
  result.wideVlenBits = *vlen;
  result.wideInputFragmentVRegs = *inputVRegs;
  result.wideAccumulatorVRegs = *accumulatorVRegs;
  result.wideVRegFloor = *vregFloor;
  return result;
}

} // namespace weft::plugin::ime
