//===- RVVQuantizeRowStreamFrontDoor.cpp --------------------------------===//
//
// The CERT-FD 次族 quant PRE-EMITC front door. See the header for the full WHY.
//
// In one line: it runs ONLY the CONSTRUCTION half of the streaming quantize_row
// front door (the shared byte-exact weft::rvv::constructTypedQuantizeRowLoopBody)
// and STOPS at the realized typed region -- BEFORE --weft-rvv-lower-to-emitc -- so
// the certification walker can walk (and hence machine-certify) the constructed
// weft_rvv.typed_quantize_row_loop_body region. NO emit here; the emit half
// (emitTypedQuantizeRowLoopBody) is byte-exact-unchanged and consumes the region
// when --weft-rvv-lower-to-emitc runs next. The 3 constructed activation quantizers
// are q8_0 / q8_1 / q8_K (each its OWN abstract op). Numerical semantics: zero change.
//
//===----------------------------------------------------------------------===//

#include "Weft/Plugin/RVV/RVVQuantizeRowStreamFrontDoor.h"
#include "Weft/Plugin/RVV/RVVFormulaCatalog.h"
#include "Weft/Plugin/RVV/RVVQuantizeFormula.h"

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Dialect/RVV/IR/RVVQuantizeRowConstruction.h"
#include "Weft/Plugin/ExtensionPlugin.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>

namespace weft::plugin::rvv {
namespace {

namespace weftrvv = ::weft::rvv;

// One pending abstract quantize op: its generic Operation*, the three ABI values
// the shared construction needs (input f32 base / output byte buffer / runtime
// element count), and the typed semantic leaf determined by the source op.  The
// formula, not this record or the emitter, constructs provenance and layout facts.
struct PendingQuant {
  mlir::Operation *op;
  mlir::Value input;
  mlir::Value output;
  mlir::Value n;
  weftrvv::QuantizeRowLeaf leaf;
};

class MaterializeRVVQuantizeRowStreamFrontDoorPass final
    : public mlir::PassWrapper<
          MaterializeRVVQuantizeRowStreamFrontDoorPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVQuantizeRowStreamFrontDoorPass() = default;

  llvm::StringRef getArgument() const final {
    return "weft-rvv-materialize-quantize-row-stream-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Pre-emitc CONSTRUCT the typed weft_rvv.typed_quantize_row_loop_body "
           "{ quantize_row_encode_core; typed_quantize_row_loop_yield } region "
           "in place of each abstract weft_rvv.quantize_row_q8_{0,1,K} and STOP "
           "before --weft-rvv-lower-to-emitc so the realized region is walkable "
           "(the shared byte-exact construction; the emit half is unchanged).";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<weftrvv::WEFTRVVDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::IRRewriter rewriter(module.getContext());

    // Collect first, then rewrite: constructTypedQuantizeRowLoopBody erases each
    // abstract op, so mutating during the walk would be unsafe.  Source identity is
    // converted once into the typed geometry leaf consumed by the formula.
    llvm::SmallVector<PendingQuant> pending;
    module.walk([&](weftrvv::GgmlQuantizeRowQ80Op op) {
      pending.push_back({op.getOperation(), op.getInput(), op.getOutput(),
                         op.getElementCount(), weftrvv::QuantizeRowLeaf::Q8_0});
    });
    module.walk([&](weftrvv::GgmlQuantizeRowQ81Op op) {
      pending.push_back({op.getOperation(), op.getInput(), op.getOutput(),
                         op.getElementCount(), weftrvv::QuantizeRowLeaf::Q8_1});
    });
    module.walk([&](weftrvv::GgmlQuantizeRowQ8KOp op) {
      pending.push_back({op.getOperation(), op.getInput(), op.getOutput(),
                         op.getElementCount(), weftrvv::QuantizeRowLeaf::Q8_K});
    });

    for (const PendingQuant &p : pending) {
      std::optional<weftrvv::QuantizeRowStreamFacts> facts =
          constructQuantizeRowPlan(QuantizeRowGeometryFacts{p.leaf},
                                   QuantizeRowNoCapabilityInput{},
                                   QuantizeRowNoStaticContext{});
      if (!facts) {
        p.op->emitError() << "quantize_row formula rejected typed leaf '"
                          << weftrvv::stringifyQuantizeRowLeaf(p.leaf) << "'";
        signalPassFailure();
        return;
      }
      if (mlir::failed(weftrvv::constructTypedQuantizeRowLoopBody(
              rewriter, p.op, p.input, p.output, p.n, *facts))) {
        p.op->emitError()
            << "quantize_row-stream front door failed to construct the typed "
               "region for leaf '"
            << weftrvv::stringifyQuantizeRowLeaf(p.leaf) << "'";
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
createMaterializeRVVQuantizeRowStreamFrontDoorPass() {
  return std::make_unique<MaterializeRVVQuantizeRowStreamFrontDoorPass>();
}

llvm::Error registerRVVQuantizeRowStreamFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry & /*registry*/,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin, formula_catalog::kQuantizeRowSourceEntry,
      "Pre-emitc construct the typed streaming quantize_row loop-body region "
      "(weft_rvv.typed_quantize_row_loop_body { quantize_row_encode_core; yield }) "
      "in place of the abstract weft_rvv.quantize_row_q8_{0,1,K} so the realized "
      "region is walkable before --weft-rvv-lower-to-emitc (the shared byte-exact "
      "construction; the f32->QUANT mirror of the dequant-stream front door)",
      formula_catalog::kQuantizeRowConstruction,
      [] { return createMaterializeRVVQuantizeRowStreamFrontDoorPass(); },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
