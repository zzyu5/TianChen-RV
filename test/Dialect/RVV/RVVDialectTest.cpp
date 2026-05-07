#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
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
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::rvv::I32VAddMicrokernelOp;
using tianchenrv::tcrv::rvv::MaskPolicy;
using tianchenrv::tcrv::rvv::PolicyAttr;
using tianchenrv::tcrv::rvv::TCRVRVVDialect;
using tianchenrv::tcrv::rvv::TailPolicy;
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

VariantOp findVariant(mlir::ModuleOp module, llvm::StringRef name) {
  VariantOp variant;
  module.walk([&](VariantOp candidate) {
    if (candidate.getSymName() == name)
      variant = candidate;
  });
  return variant;
}

int expectRVVPolicyAttr(VariantOp variant, TailPolicy expectedTail,
                        MaskPolicy expectedMask) {
  if (int result =
          expect(static_cast<bool>(variant),
                 "module contains typed tcrv.exec.variant for RVV policy"))
    return result;

  auto policy =
      variant->getAttrOfType<PolicyAttr>(
          tianchenrv::plugin::rvv::getRVVPolicyAttrName());
  if (int result = expect(static_cast<bool>(policy),
                          "variant carries typed #tcrv_rvv.policy metadata"))
    return result;
  if (int result = expect(policy.getTail() == expectedTail,
                          "typed RVV policy tail value is preserved"))
    return result;
  return expect(policy.getMask() == expectedMask,
                "typed RVV policy mask value is preserved");
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

int runPolicyAttributeRoundTripTest() {
  ExtensionPluginRegistry plugins;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
          "register RVV extension plugin for policy attribute round trip"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @policy_roundtrip attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse typed RVV policy attribute IR");
  if (int result = expectRVVPolicyAttr(
          findVariant(*module, "rvv_first_slice"), TailPolicy::Agnostic,
          MaskPolicy::Agnostic))
    return result;

  std::string printedStorage;
  llvm::raw_string_ostream printedStream(printedStorage);
  module->print(printedStream);
  printedStream.flush();
  if (int result =
          expect(llvm::StringRef(printedStorage).contains("tcrv_rvv.policy") &&
                     llvm::StringRef(printedStorage)
                         .contains("#tcrv_rvv.policy"),
                 "printed module preserves typed RVV policy attribute syntax"))
    return result;

  mlir::OwningOpRef<mlir::ModuleOp> reparsed =
      parseModule(context, printedStorage);
  if (!reparsed)
    return fail("failed to reparse printed typed RVV policy attribute IR");
  if (int result = expectRVVPolicyAttr(
          findVariant(*reparsed, "rvv_first_slice"), TailPolicy::Agnostic,
          MaskPolicy::Agnostic))
    return result;

  llvm::outs() << "RVV typed policy attribute round trip preserved\n";
  return 0;
}

int runI32VAddMicrokernelRoundTripTest() {
  ExtensionPluginRegistry plugins;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
          "register RVV plugin for microkernel round trip"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @microkernel_roundtrip attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    tcrv_rvv.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "microkernel_roundtrip"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse RVV i32 vector-add microkernel op");

  I32VAddMicrokernelOp microkernel;
  module->walk([&](I32VAddMicrokernelOp candidate) {
    microkernel = candidate;
  });
  if (int result =
          expect(static_cast<bool>(microkernel),
                 "module contains tcrv_rvv.i32_vadd_microkernel"))
    return result;

  auto elementCount =
      microkernel->getAttrOfType<mlir::IntegerAttr>("element_count");
  if (int result = expect(elementCount && elementCount.getInt() == 16,
                          "microkernel element_count is preserved"))
    return result;

  std::string printedStorage;
  llvm::raw_string_ostream printedStream(printedStorage);
  module->print(printedStream);
  printedStream.flush();
  if (int result =
          expect(llvm::StringRef(printedStorage)
                     .contains("tcrv_rvv.i32_vadd_microkernel"),
                 "printed module preserves RVV microkernel op"))
    return result;

  mlir::OwningOpRef<mlir::ModuleOp> reparsed =
      parseModule(context, printedStorage);
  if (!reparsed)
    return fail("failed to reparse printed RVV microkernel module");

  I32VAddMicrokernelOp reparsedMicrokernel;
  reparsed->walk([&](I32VAddMicrokernelOp candidate) {
    reparsedMicrokernel = candidate;
  });
  if (int result = expect(static_cast<bool>(reparsedMicrokernel),
                          "reparsed module preserves RVV microkernel op"))
    return result;

  llvm::outs() << "RVV i32 vector-add microkernel round trip preserved\n";
  return 0;
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

int runMalformedRVVPolicySyntaxTest() {
  ExtensionPluginRegistry plugins;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
          "register RVV plugin for malformed policy test"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @bad_policy attributes {} {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = invalid, mask = agnostic>
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModuleExpectingFailure(context, source);
  return expect(!module, "malformed #tcrv_rvv.policy syntax is rejected");
}

} // namespace

int main() {
  if (int result = runPluginDialectRegistrationRoundTripTest())
    return result;
  if (int result = runPolicyAttributeRoundTripTest())
    return result;
  if (int result = runI32VAddMicrokernelRoundTripTest())
    return result;
  if (int result = runDefaultCoreDoesNotRegisterRVVDialectTest())
    return result;
  if (int result = runDisabledPluginDoesNotRegisterDialectTest())
    return result;
  if (int result = runMalformedRVVTypeSyntaxTest())
    return result;
  if (int result = runMalformedRVVPolicySyntaxTest())
    return result;

  llvm::outs() << "RVV dialect registration smoke test passed\n";
  return 0;
}
