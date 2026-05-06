#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using tianchenrv::plugin::ExtensionPlugin;
using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::tcrv::rvv::TCRVRVVDialect;
using tianchenrv::tcrv::rvv::VLType;

namespace {

class DisabledDialectPlugin final : public ExtensionPlugin {
public:
  llvm::StringRef getName() const override {
    return "disabled-rvv-dialect-plugin";
  }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return {};
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<TCRVRVVDialect>();
  }

  bool isEnabled() const override { return false; }
};

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

int expect(bool condition, llvm::Twine message) {
  if (condition)
    return 0;
  return fail(message);
}

int expectSuccess(llvm::Error error, llvm::Twine context) {
  if (!error)
    return 0;

  std::string message = llvm::toString(std::move(error));
  return fail(context + ": " + message);
}

mlir::OwningOpRef<mlir::ModuleOp>
parseModule(mlir::MLIRContext &context, llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

mlir::OwningOpRef<mlir::ModuleOp>
parseModuleExpectingFailure(mlir::MLIRContext &context,
                            llvm::StringRef source) {
  mlir::ScopedDiagnosticHandler diagnostics(
      &context, [](mlir::Diagnostic &diagnostic) {
        (void)diagnostic;
        return mlir::success();
      });
  return parseModule(context, source);
}

int expectRVVTypeResult(mlir::ModuleOp module) {
  mlir::Operation *conversion = nullptr;
  module.walk([&](mlir::Operation *candidate) {
    if (candidate->getName().getStringRef() ==
        "builtin.unrealized_conversion_cast")
      conversion = candidate;
  });

  if (int result = expect(conversion, "parsed module has a type-bearing op"))
    return result;
  if (int result = expect(conversion->getNumResults() == 1,
                          "type-bearing op has one result"))
    return result;
  return expect(llvm::isa<VLType>(conversion->getResult(0).getType()),
                "type-bearing op result is !tcrv_rvv.vl");
}

int runPluginDialectRegistrationRoundTripTest() {
  ExtensionPluginRegistry plugins;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
          "register RVV extension plugin"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result =
          expect(context.getLoadedDialect<TCRVRVVDialect>() != nullptr,
                 "RVV plugin path loads the RVV dialect"))
    return result;

  constexpr llvm::StringLiteral source = R"mlir(
module {
  %token = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vl
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse !tcrv_rvv.vl through RVV plugin registry");
  if (int result = expectRVVTypeResult(*module))
    return result;

  std::string printedStorage;
  llvm::raw_string_ostream printedStream(printedStorage);
  module->print(printedStream);
  printedStream.flush();
  if (int result = expect(llvm::StringRef(printedStorage)
                              .contains("!tcrv_rvv.vl"),
                          "printed module preserves !tcrv_rvv.vl"))
    return result;

  mlir::OwningOpRef<mlir::ModuleOp> reparsed =
      parseModule(context, printedStorage);
  if (!reparsed)
    return fail("failed to reparse printed !tcrv_rvv.vl module");
  return expectRVVTypeResult(*reparsed);
}

int runDefaultCoreDoesNotRegisterRVVDialectTest() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result =
          expect(context.getLoadedDialect<TCRVRVVDialect>() == nullptr,
                 "default registerAllDialects remains core-only"))
    return result;

  constexpr llvm::StringLiteral source = R"mlir(
module {
  %token = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vl
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModuleExpectingFailure(context, source);
  return expect(!module,
                "!tcrv_rvv.vl is rejected without RVV plugin registration");
}

int runDisabledPluginDoesNotRegisterDialectTest() {
  DisabledDialectPlugin disabledPlugin;
  ExtensionPluginRegistry plugins;
  if (int result = expectSuccess(plugins.registerPlugin(disabledPlugin),
                                 "register disabled dialect plugin"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result =
          expect(context.getLoadedDialect<TCRVRVVDialect>() == nullptr,
                 "enabled-only plugin dialect registration skips disabled"))
    return result;

  constexpr llvm::StringLiteral source = R"mlir(
module {
  %token = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vl
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModuleExpectingFailure(context, source);
  return expect(!module,
                "disabled plugin path does not make !tcrv_rvv.vl parseable");
}

int runMalformedRVVTypeSyntaxTest() {
  ExtensionPluginRegistry plugins;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
          "register RVV plugin for malformed type test"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);

  constexpr llvm::StringLiteral source = R"mlir(
module {
  %token = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.unknown
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModuleExpectingFailure(context, source);
  return expect(!module, "unknown !tcrv_rvv type syntax is rejected");
}

} // namespace

int main() {
  if (int result = runPluginDialectRegistrationRoundTripTest())
    return result;
  if (int result = runDefaultCoreDoesNotRegisterRVVDialectTest())
    return result;
  if (int result = runDisabledPluginDoesNotRegisterDialectTest())
    return result;
  if (int result = runMalformedRVVTypeSyntaxTest())
    return result;

  llvm::outs() << "RVV dialect registration smoke test passed\n";
  return 0;
}
