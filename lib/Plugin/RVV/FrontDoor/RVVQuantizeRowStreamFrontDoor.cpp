//===- RVVQuantizeRowStreamFrontDoor.cpp --------------------------------===//
//
// The CERT-FD 次族 quant PRE-EMITC front door. See the header for the full WHY.
//
// In one line: it runs ONLY the CONSTRUCTION half of the streaming quantize_row
// front door (the shared byte-exact tcrv::rvv::constructTypedQuantizeRowLoopBody)
// and STOPS at the realized typed region -- BEFORE --tcrv-rvv-lower-to-emitc -- so
// the certification walker can walk (and hence machine-certify) the constructed
// tcrv_rvv.typed_quantize_row_loop_body region. NO emit here; the emit half
// (emitTypedQuantizeRowLoopBody) is byte-exact-unchanged and consumes the region
// when --tcrv-rvv-lower-to-emitc runs next. The 3 constructed activation quantizers
// are q8_0 / q8_1 / q8_K (each its OWN abstract op). Numerical semantics: zero change.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVQuantizeRowStreamFrontDoor.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Dialect/RVV/IR/RVVQuantizeRowConstruction.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>

namespace tianchenrv::plugin::rvv {
namespace {

namespace tcrvrvv = ::tianchenrv::tcrv::rvv;

// One pending abstract quantize op: its generic Operation*, the three ABI values
// the shared construction needs (input f32 base / output byte buffer / runtime
// element count), and the encode_model determined by the op's CONCRETE type (the 3
// quantize ops carry no `format` attr -- the model IS the op identity).
struct PendingQuant {
  mlir::Operation *op;
  mlir::Value input;
  mlir::Value output;
  mlir::Value n;
  llvm::StringRef encodeModel;
};

class MaterializeRVVQuantizeRowStreamFrontDoorPass final
    : public mlir::PassWrapper<
          MaterializeRVVQuantizeRowStreamFrontDoorPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVQuantizeRowStreamFrontDoorPass() = default;

  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-quantize-row-stream-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Pre-emitc CONSTRUCT the typed tcrv_rvv.typed_quantize_row_loop_body "
           "{ quantize_row_encode_core; typed_quantize_row_loop_yield } region "
           "in place of each abstract tcrv_rvv.quantize_row_q8_{0,1,K} and STOP "
           "before --tcrv-rvv-lower-to-emitc so the realized region is walkable "
           "(the shared byte-exact construction; the emit half is unchanged).";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<tcrvrvv::TCRVRVVDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::IRRewriter rewriter(module.getContext());

    // Collect first, then rewrite: constructTypedQuantizeRowLoopBody erases each
    // abstract op, so mutating during the walk would be unsafe. The encode_model is
    // fixed by the op TYPE (q8_0/q8_1/q8_K each its own op).
    llvm::SmallVector<PendingQuant> pending;
    module.walk([&](tcrvrvv::GgmlQuantizeRowQ80Op op) {
      pending.push_back({op.getOperation(), op.getInput(), op.getOutput(),
                         op.getElementCount(), "q8_0"});
    });
    module.walk([&](tcrvrvv::GgmlQuantizeRowQ81Op op) {
      pending.push_back({op.getOperation(), op.getInput(), op.getOutput(),
                         op.getElementCount(), "q8_1"});
    });
    module.walk([&](tcrvrvv::GgmlQuantizeRowQ8KOp op) {
      pending.push_back({op.getOperation(), op.getInput(), op.getOutput(),
                         op.getElementCount(), "q8_K"});
    });

    for (const PendingQuant &p : pending) {
      std::optional<tcrvrvv::QuantizeRowStreamFacts> facts =
          tcrvrvv::lookupQuantizeRowStreamFacts(p.encodeModel);
      if (!facts)
        continue; // not a constructed encode_model (defensive; unreachable here).
      if (mlir::failed(tcrvrvv::constructTypedQuantizeRowLoopBody(
              rewriter, p.op, p.input, p.output, p.n, p.encodeModel, *facts))) {
        p.op->emitError()
            << "quantize_row-stream front door failed to construct the typed "
               "region for encode_model '"
            << p.encodeModel << "'";
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
      ownerPlugin, "tcrv-rvv-materialize-quantize-row-stream-front-door",
      "Pre-emitc construct the typed streaming quantize_row loop-body region "
      "(tcrv_rvv.typed_quantize_row_loop_body { quantize_row_encode_core; yield }) "
      "in place of the abstract tcrv_rvv.quantize_row_q8_{0,1,K} so the realized "
      "region is walkable before --tcrv-rvv-lower-to-emitc (the shared byte-exact "
      "construction; the f32->QUANT mirror of the dequant-stream front door)",
      [] { return createMaterializeRVVQuantizeRowStreamFrontDoorPass(); },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
