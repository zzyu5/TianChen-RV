//===- RVVDequantizeRowStreamFrontDoor.cpp ------------------------------===//
//
// The CERT-FD dequant首族 PRE-EMITC front door. See the header for the full WHY.
//
// In one line: it runs ONLY the CONSTRUCTION half of the streaming dequantize_row
// front door (the shared byte-exact weft::rvv::constructTypedDequantizeRowLoopBody)
// and STOPS at the realized typed region -- BEFORE --weft-rvv-lower-to-emitc -- so
// the certification walker can walk (and hence machine-certify) the constructed
// weft_rvv.typed_dequantize_row_loop_body region. NO emit here; the emit half
// (emitTypedDequantizeRowLoopBody) is byte-exact-unchanged and consumes the region
// when --weft-rvv-lower-to-emitc runs next. Numerical semantics: zero change.
//
//===----------------------------------------------------------------------===//

#include "Weft/Plugin/RVV/RVVDequantizeRowStreamFrontDoor.h"
#include "Weft/Plugin/RVV/RVVFormulaCatalog.h"
#include "Weft/Plugin/RVV/RVVFormulaConstruction.h"

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace weft::plugin::rvv {
namespace {

namespace weftrvv = ::weft::rvv;

class MaterializeRVVDequantizeRowStreamFrontDoorPass final
    : public mlir::PassWrapper<
          MaterializeRVVDequantizeRowStreamFrontDoorPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVDequantizeRowStreamFrontDoorPass() = default;

  llvm::StringRef getArgument() const final {
    return "weft-rvv-materialize-dequantize-row-stream-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Pre-emitc CONSTRUCT the typed weft_rvv.typed_dequantize_row_loop_body "
           "{ dequantize_row_decode_core; typed_dequantize_row_loop_yield } region "
           "in place of each recognized abstract weft_rvv.dequantize_row and STOP "
           "before --weft-rvv-lower-to-emitc "
           "so the realized region is walkable (the shared byte-exact construction; "
           "the emit half is unchanged). The whole dequantize_row spectrum is now "
           "front-door CONSTRUCTED (q1_0 was the last flat leaf flipped).";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<weftrvv::WEFTRVVDialect>();
  }

  void runOnOperation() final {
    if (mlir::failed(constructRVVDequantizeRowFormulaBodies(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
createMaterializeRVVDequantizeRowStreamFrontDoorPass() {
  return std::make_unique<MaterializeRVVDequantizeRowStreamFrontDoorPass>();
}

llvm::Error registerRVVDequantizeRowStreamFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry & /*registry*/,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin, formula_catalog::kDequantizeRowSourceEntry,
      "Pre-emitc construct the typed streaming dequantize_row loop-body region "
      "(weft_rvv.typed_dequantize_row_loop_body { dequantize_row_decode_core; "
      "yield }) in place of the abstract weft_rvv.dequantize_row so the realized "
      "region is walkable before --weft-rvv-lower-to-emitc (the shared byte-exact "
      "construction; the whole dequantize_row spectrum is front-door constructed)",
      formula_catalog::kDequantizeRowConstruction,
      [] { return createMaterializeRVVDequantizeRowStreamFrontDoorPass(); },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
