#include "Weft/Transforms/Passes.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <utility>

namespace weft::transforms {

#define GEN_PASS_DEF_CONSTRUCTRVVFORMULAPLANS
#include "Weft/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");

class ConstructRVVFormulaPlansPass final
    : public impl::ConstructRVVFormulaPlansBase<
          ConstructRVVFormulaPlansPass> {
public:
  ConstructRVVFormulaPlansPass() : registry(&ownedRegistry) {}

  explicit ConstructRVVFormulaPlansPass(
      const plugin::ExtensionPluginRegistry &registry)
      : registry(&registry) {}

  ConstructRVVFormulaPlansPass(const ConstructRVVFormulaPlansPass &other)
      : impl::ConstructRVVFormulaPlansBase<ConstructRVVFormulaPlansPass>(other),
        registry(other.registry == &other.ownedRegistry ? &ownedRegistry
                                                        : other.registry) {}

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<weft::exec::VariantOp, 8> variants;
    module.walk([&](weft::exec::VariantOp variant) {
      auto origin = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
      if (origin && origin.getValue() == kRVVPluginName)
        variants.push_back(variant);
    });

    for (weft::exec::VariantOp variant : variants) {
      plugin::FamilyConstructionResult result;
      if (llvm::Error error = registry->constructFormulaPlansForVariant(
              module, variant, result)) {
        std::string diagnostic = llvm::toString(std::move(error));
        variant.emitError()
            << "bound RVV formula construction failed before artifact "
               "projection: "
            << diagnostic;
        signalPassFailure();
        return;
      }
      if (!result.hasFinalBody()) {
        variant.emitError()
            << "bound RVV formula construction produced no final typed body: "
            << result.getReason();
        signalPassFailure();
        return;
      }
    }
  }

private:
  plugin::ExtensionPluginRegistry ownedRegistry;
  const plugin::ExtensionPluginRegistry *registry = nullptr;
};

} // namespace

std::unique_ptr<::mlir::Pass> createConstructRVVFormulaPlansPass() {
  return std::make_unique<ConstructRVVFormulaPlansPass>();
}

std::unique_ptr<::mlir::Pass> createConstructRVVFormulaPlansPass(
    const plugin::ExtensionPluginRegistry &registry) {
  return std::make_unique<ConstructRVVFormulaPlansPass>(registry);
}

} // namespace weft::transforms
