#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_CHECKCAPABILITYREQUIRES
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

class CheckCapabilityRequiresPass final
    : public impl::CheckCapabilityRequiresBase<CheckCapabilityRequiresPass> {
public:
  using impl::CheckCapabilityRequiresBase<
      CheckCapabilityRequiresPass>::CheckCapabilityRequiresBase;

  void runOnOperation() override {
    bool foundUnavailableRequirement = false;
    getOperation()->walk([&](tcrv::exec::KernelOp kernel) {
      support::TargetCapabilitySet capabilities =
          support::TargetCapabilitySet::buildFromKernel(kernel);
      checkKernel(kernel, capabilities, foundUnavailableRequirement);
    });

    if (foundUnavailableRequirement)
      signalPassFailure();
  }

private:
  void checkKernel(tcrv::exec::KernelOp kernel,
                   const support::TargetCapabilitySet &capabilities,
                   bool &foundUnavailableRequirement) const {
    if (!kernel || kernel.getBody().empty())
      return;

    for (mlir::Operation &op : kernel.getBody().front()) {
      auto variant = llvm::dyn_cast<tcrv::exec::VariantOp>(op);
      if (!variant)
        continue;

      auto requiresAttr =
          variant->getAttrOfType<mlir::ArrayAttr>("requires");
      if (!requiresAttr)
        continue;

      for (mlir::Attribute requiredCapability : requiresAttr) {
        auto symbolRef =
            llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
        if (!symbolRef)
          continue;

        const support::CapabilityDescriptor *capability =
            capabilities.lookupBySymbolName(symbolRef.getValue());
        if (!capability || capability->isAvailable())
          continue;

        mlir::InFlightDiagnostic diagnostic = variant.emitError()
            << "variant @" << variant.getSymName()
            << " requires unavailable capability @" << symbolRef.getValue()
            << " (id = \"" << capability->getID() << "\", kind = \""
            << capability->getKind() << "\"";
        if (!capability->getStatus().empty())
          diagnostic << ", status = \"" << capability->getStatus() << "\"";
        diagnostic << ")";
        foundUnavailableRequirement = true;
      }
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createCheckCapabilityRequiresPass() {
  return std::make_unique<CheckCapabilityRequiresPass>();
}

} // namespace tianchenrv::transforms
