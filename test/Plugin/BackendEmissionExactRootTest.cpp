#include "Weft/Conversion/EmitC/BackendEmissionRegistry.h"
#include "Weft/Dialect/Demo/IR/DemoDialect.h"
#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/Toy/IR/ToyDialect.h"
#include "Weft/InitWeftDialects.h"
#include "Weft/Plugin/Demo/DemoBackendEmissionDriver.h"
#include "Weft/Plugin/Toy/ToyBackendEmissionDriver.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace {

int fail(llvm::Twine message) {
  llvm::errs() << "BackendEmissionExactRootTest failed: " << message << '\n';
  return 1;
}

mlir::Operation *findByNameAndVariant(mlir::ModuleOp module,
                                      llvm::StringRef operationName,
                                      llvm::StringRef variant) {
  mlir::Operation *found = nullptr;
  module.walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() != operationName)
      return;
    auto selected = operation->getAttrOfType<mlir::FlatSymbolRefAttr>(
        "selected_variant");
    if (selected && selected.getValue() == variant)
      found = operation;
  });
  return found;
}

bool hasEmitCFunction(mlir::ModuleOp module, llvm::StringRef symbol) {
  bool found = false;
  module.walk([&](mlir::emitc::FuncOp function) {
    if (function.getSymName() == symbol)
      found = true;
  });
  return found;
}

} // namespace

int main() {
  mlir::DialectRegistry dialects;
  weft::registerAllDialects(dialects);
  dialects.insert<mlir::emitc::EmitCDialect,
                  weft::demo_ext::WEFTDemoDialect,
                  weft::toy::WEFTToyDialect>();
  mlir::MLIRContext context(dialects);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @same_backend {
    weft.exec.variant @selected_a attributes {origin = "demo-plugin", requires = []} {
    }
    weft.exec.variant @decoy_b attributes {origin = "demo-plugin", requires = []} {
    }
    "weft_demo.compute_skeleton"() {source_kernel = "same_backend", selected_variant = @selected_a} : () -> ()
    "weft_demo.compute_skeleton"() {source_kernel = "same_backend", selected_variant = @decoy_b} : () -> ()
  }
  weft.exec.kernel @mixed_backend {
    weft.exec.variant @demo_a attributes {origin = "demo-plugin", requires = []} {
    }
    "weft_demo.compute_skeleton"() {source_kernel = "mixed_backend", selected_variant = @demo_a} : () -> ()
    "weft_toy.compute_skeleton"() {source_kernel = "mixed_backend", selected_variant = @demo_a} : () -> ()
  }
  weft.exec.kernel @owner_mismatch {
    weft.exec.variant @toy_owner attributes {origin = "toy-plugin", requires = []} {
    }
    "weft_demo.compute_skeleton"() {source_kernel = "owner_mismatch", selected_variant = @toy_owner} : () -> ()
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    return fail("cannot parse exact-root fixture");

  weft::conversion::emitc::BackendEmissionRegistry registry;
  weft::plugin::demo_ext::registerDemoBackendEmitter(registry);
  weft::plugin::toy::registerToyBackendEmitter(registry);

  mlir::Operation *selected =
      findByNameAndVariant(*module, "weft_demo.compute_skeleton", "selected_a");
  if (!selected)
    return fail("missing selected Demo root");
  mlir::OwningOpRef<mlir::ModuleOp> converted =
      registry.tryConvertConstructedModuleClone(*module, selected);
  if (!converted)
    return fail("selected exact root did not convert");
  if (!hasEmitCFunction(*converted,
                        "weft_emitc_same_backend_selected_a"))
    return fail("selected exact root artifact is absent");
  if (hasEmitCFunction(*converted, "weft_emitc_same_backend_decoy_b"))
    return fail("same-backend decoy root was converted");

  mlir::Operation *mixed =
      findByNameAndVariant(*module, "weft_demo.compute_skeleton", "demo_a");
  if (!mixed)
    return fail("missing mixed-owner selected root");
  if (registry.tryConvertConstructedModuleClone(*module, mixed))
    return fail("mixed-backend artifact did not fail closed");

  mlir::Operation *ownerMismatch =
      findByNameAndVariant(*module, "weft_demo.compute_skeleton", "toy_owner");
  if (!ownerMismatch)
    return fail("missing owner-mismatch root");
  if (registry.tryConvertConstructedModuleClone(*module, ownerMismatch))
    return fail("variant origin/backend owner mismatch did not fail closed");

  weft::exec::KernelOp wrongRoot;
  module->walk([&](weft::exec::KernelOp kernel) {
    if (kernel.getSymName() == "same_backend")
      wrongRoot = kernel;
  });
  if (registry.tryConvertConstructedModuleClone(
          *module, wrongRoot.getOperation()))
    return fail("non-final wrong root was accepted");
  if (registry.tryConvertConstructedModuleClone(*module, nullptr))
    return fail("null root was accepted");

  return 0;
}
