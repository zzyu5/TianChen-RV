#include "Weft/Plugin/IME/IMEFormulaConstruction.h"

#include "Weft/Dialect/IME/IR/IMEDialect.h"
#include "Weft/Plugin/IME/IMEExtensionPlugin.h"

#include "mlir/IR/BuiltinAttributes.h"
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
};

/// The current constructed A-reuse mechanism has two typed schedules: the
/// narrow leaf and a two-column reuse leaf.  A four-column helper is not an
/// analytic candidate until its epilogue/register topology is represented by
/// the typed body; historical measurements are not a legality input.
static constexpr IMEWideCandidate kIMEWideCandidates[] = {{1}, {2}};

constexpr int64_t kIMEVectorRegisterFileSize = 32;

constexpr llvm::StringLiteral kMacBatchedAttrName("mac_batched");
constexpr llvm::StringLiteral kWideNJWAttrName("wide_njw");
constexpr llvm::StringLiteral kWideVlenBitsAttrName("wide_vlen_bits");
constexpr llvm::StringLiteral kWideInputFragmentVRegsAttrName(
    "wide_input_fragment_vregs");
constexpr llvm::StringLiteral kWideAccumulatorVRegsAttrName(
    "wide_accumulator_vregs");
constexpr llvm::StringLiteral kWideVRegFloorAttrName("wide_vreg_floor");

bool isIMEFinalBody(mlir::Operation *op) {
  return llvm::isa<weft::ime::MMAOp, weft::ime::MMAUOp,
                   weft::ime::MMASUOp, weft::ime::MMAUSOp,
                   weft::ime::MMASlideOp, weft::ime::MatMulOp,
                   weft::ime::Q40MatMulTileOp, weft::ime::Q80MatMulTileOp,
                   weft::ime::Q4KMatMulTileOp>(op);
}

mlir::LogicalResult setTypedScheduleInteger(mlir::Operation *op,
                                            llvm::StringRef name,
                                            int64_t expected) {
  auto value = op->getAttrOfType<mlir::IntegerAttr>(name);
  if (op->hasAttr(name) && !value)
    return op->emitError() << "IME typed schedule field '" << name
                           << "' must be an integer";
  if (value && value.getInt() != expected)
    return op->emitError() << "IME typed schedule field '" << name
                           << "' conflicts with family construction";
  op->setAttr(name, mlir::IntegerAttr::get(
                        mlir::IntegerType::get(op->getContext(), 64), expected));
  return mlir::success();
}

mlir::FailureOr<int64_t> readTypedScheduleInteger(mlir::Operation *op,
                                                  llvm::StringRef name) {
  auto value = op->getAttrOfType<mlir::IntegerAttr>(name);
  if (!value)
    return op->emitError() << "IME exact typed body is missing schedule field '"
                           << name << "'";
  return value.getInt();
}

IMEWideComputationDecision constructWideVmadotPlan(
    int64_t vlenBits, int64_t macM, int64_t macN, int64_t macK,
    int64_t elemInBits, int64_t accumBits, bool supportsWideAReuse) {
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

  // The q4_K typed body currently owns a distinct two-accumulator scale/min
  // topology and therefore exposes only the narrow schedule.  This is an
  // honest-null mechanism axis, not a benchmark-derived exclusion.
  if (!supportsWideAReuse)
    return decision;

  // The current typed wide leaf uses one e8,m1 load per input fragment.
  if (decision.inputFragmentVRegs != 1)
    return decision;

  for (const IMEWideCandidate &candidate : kIMEWideCandidates) {
    const int64_t floor = floorFor(candidate.njw);
    if (floor <= kIMEVectorRegisterFileSize && candidate.njw > decision.njw) {
      decision.njw = candidate.njw;
      decision.vregFloor = floor;
    }
  }
  return decision;
}

template <typename TileOp>
mlir::LogicalResult constructIMEQuantTilePlan(
    TileOp tile, int64_t targetVlenBits, bool supportsWideAReuse) {
  mlir::Block &body = tile.getBody().front();
  auto macLeaves = body.template getOps<weft::ime::VmadotMacLeafOp>();
  if (macLeaves.empty())
    return tile.emitError(
        "IME quant tile construction requires one typed vmadot MAC leaf");

  weft::ime::VmadotMacLeafOp macLeaf = *macLeaves.begin();
  const int64_t macK = macLeaf.getMacK();
  const int64_t fragmentCount = macK > 0 ? tile.getMatK() / macK : 1;
  const bool macBatched = fragmentCount >= 2;

  if (targetVlenBits <= 0)
    return tile.emitError(
        "IME construction requires positive target-projected VLEN");
  IMEWideComputationDecision wide = constructWideVmadotPlan(
      targetVlenBits, tile.getMacM(), tile.getMacN(), tile.getMacK(),
      tile.getElemInBits(), tile.getAccumBits(), supportsWideAReuse);

  for (auto [name, value] : {
           std::pair<llvm::StringRef, int64_t>(kMacBatchedAttrName,
                                               macBatched ? 1 : 0),
           {kWideNJWAttrName, wide.njw},
           {kWideVlenBitsAttrName, wide.vlenBits},
           {kWideInputFragmentVRegsAttrName, wide.inputFragmentVRegs},
           {kWideAccumulatorVRegsAttrName, wide.accumulatorVRegs},
           {kWideVRegFloorAttrName, wide.vregFloor},
       })
    if (mlir::failed(setTypedScheduleInteger(tile.getOperation(), name, value)))
      return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult validateIMEConstructionContext(
    mlir::Operation *op, weft::exec::VariantOp variant,
    weft::exec::KernelOp kernel) {
  auto sourceKernel = op->getAttrOfType<mlir::StringAttr>("source_kernel");
  if (!variant || !kernel || !sourceKernel ||
      sourceKernel.getValue() != kernel.getSymName())
    return op->emitError(
        "IME construction requires source_kernel to match an enclosing "
        "weft.exec.kernel");

  auto selectedVariant =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
  if (!selectedVariant || selectedVariant.getValue() != variant.getSymName())
    return op->emitError(
        "IME construction body must name the explicitly bound selected "
        "variant");
  return mlir::success();
}

} // namespace

mlir::LogicalResult constructIMEFormulaPlan(
    mlir::Operation *op, weft::exec::VariantOp variant,
    weft::exec::KernelOp kernel, int64_t targetVlenBits) {
  if (!op || !isIMEFinalBody(op))
    return mlir::failure();
  if (mlir::failed(validateIMEConstructionContext(op, variant, kernel)))
    return mlir::failure();

  if (llvm::isa<weft::ime::MMAOp, weft::ime::MMAUOp,
                weft::ime::MMASUOp, weft::ime::MMAUSOp,
                weft::ime::MMASlideOp>(op))
    return mlir::success();
  if (llvm::isa<weft::ime::MatMulOp>(op))
    return mlir::success();
  if (auto tile = llvm::dyn_cast<weft::ime::Q40MatMulTileOp>(op))
    return constructIMEQuantTilePlan(tile, targetVlenBits,
                                     /*supportsWideAReuse=*/true);
  if (auto tile = llvm::dyn_cast<weft::ime::Q80MatMulTileOp>(op))
    return constructIMEQuantTilePlan(tile, targetVlenBits,
                                     /*supportsWideAReuse=*/true);
  if (auto tile = llvm::dyn_cast<weft::ime::Q4KMatMulTileOp>(op))
    return constructIMEQuantTilePlan(tile, targetVlenBits,
                                     /*supportsWideAReuse=*/false);
  return mlir::failure();
}

mlir::LogicalResult requireIMESimpleComputationPlan(mlir::Operation *op) {
  if (!isIMEFinalBody(op))
    return op->emitError("IME artifact projection requires an exact typed body");
  return mlir::success();
}

mlir::FailureOr<IMEMatMulComputationPlan>
readIMEMatMulComputationPlan(mlir::Operation *op) {
  IMEMatMulComputationPlan result;
  if (auto body = llvm::dyn_cast<weft::ime::MatMulOp>(op))
    result = {static_cast<int64_t>(body.getMatM()),
              static_cast<int64_t>(body.getMatN()),
              static_cast<int64_t>(body.getMatK())};
  else if (auto body = llvm::dyn_cast<weft::ime::Q40MatMulTileOp>(op))
    result = {static_cast<int64_t>(body.getMatM()),
              static_cast<int64_t>(body.getMatN()),
              static_cast<int64_t>(body.getMatK())};
  else if (auto body = llvm::dyn_cast<weft::ime::Q80MatMulTileOp>(op))
    result = {static_cast<int64_t>(body.getMatM()),
              static_cast<int64_t>(body.getMatN()),
              static_cast<int64_t>(body.getMatK())};
  else if (auto body = llvm::dyn_cast<weft::ime::Q4KMatMulTileOp>(op))
    result = {static_cast<int64_t>(body.getMatM()),
              static_cast<int64_t>(body.getMatN()),
              static_cast<int64_t>(body.getMatK())};
  else
    return op->emitError(
        "IME matrix artifact projection requires an exact matrix typed body");
  if (result.matM <= 0 || result.matN <= 0 || result.matK <= 0)
    return op->emitError("IME final computation dimensions must be positive");
  return result;
}

mlir::FailureOr<IMEQuantComputationPlan>
readIMEQuantComputationPlan(mlir::Operation *op) {
  auto matmul = readIMEMatMulComputationPlan(op);
  if (mlir::failed(matmul))
    return mlir::failure();

  auto batched = readTypedScheduleInteger(op, kMacBatchedAttrName);
  auto njw = readTypedScheduleInteger(op, kWideNJWAttrName);
  auto vlen = readTypedScheduleInteger(op, kWideVlenBitsAttrName);
  auto inputVRegs = readTypedScheduleInteger(
      op, kWideInputFragmentVRegsAttrName);
  auto accumulatorVRegs = readTypedScheduleInteger(
      op, kWideAccumulatorVRegsAttrName);
  auto vregFloor = readTypedScheduleInteger(op, kWideVRegFloorAttrName);
  if (mlir::failed(batched) || mlir::failed(njw) || mlir::failed(vlen) ||
      mlir::failed(inputVRegs) || mlir::failed(accumulatorVRegs) ||
      mlir::failed(vregFloor))
    return mlir::failure();
  if ((*batched != 0 && *batched != 1) ||
      (*njw != 1 && *njw != 2) || *vlen <= 0 ||
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
