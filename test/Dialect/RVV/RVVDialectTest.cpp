#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Support/RuntimeABI.h"

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
using tianchenrv::conversion::emitc::TCRVEmitCLowerableOpInterface;
using tianchenrv::tcrv::rvv::BinaryOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::rvv::I32AddOp;
using tianchenrv::tcrv::rvv::I32LoadOp;
using tianchenrv::tcrv::rvv::I32StoreOp;
using tianchenrv::tcrv::rvv::LoadOp;
using tianchenrv::tcrv::rvv::MaskPolicy;
using tianchenrv::tcrv::rvv::PolicyAttr;
using tianchenrv::tcrv::rvv::RuntimeABIValueOp;
using tianchenrv::tcrv::rvv::SetVLOp;
using tianchenrv::tcrv::rvv::StoreOp;
using tianchenrv::tcrv::rvv::TCRVRVVDialect;
using tianchenrv::tcrv::rvv::TailPolicy;
using tianchenrv::tcrv::rvv::WithVLOp;
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

int expectFailure(llvm::Error error, llvm::Twine context) {
  if (error) {
    llvm::consumeError(std::move(error));
    return 0;
  }

  return fail(context + ": expected failure");
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

int expectEmitCLowerableRole(mlir::Operation *op, llvm::StringRef role) {
  auto lowerable = llvm::dyn_cast<TCRVEmitCLowerableOpInterface>(op);
  if (int result = expect(static_cast<bool>(lowerable),
                          "RVV source op implements EmitC lowerable interface"))
    return result;
  if (int result =
          expect(lowerable.getTCRVEmitCLowerableSourceOpName() ==
                     op->getName().getStringRef(),
                 "EmitC lowerable source op name reflects typed RVV op"))
    return result;
  return expect(lowerable.getTCRVEmitCLowerableSourceRole() == role,
                "EmitC lowerable source role reflects RVV op role");
}

int expectNotEmitCLowerable(mlir::Operation *op, llvm::StringRef context) {
  auto lowerable = llvm::dyn_cast<TCRVEmitCLowerableOpInterface>(op);
  return expect(!static_cast<bool>(lowerable),
                llvm::Twine(context) +
                    " remains deprecated parse-only residue, not an EmitC "
                    "lowerable source op");
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

  // Deprecated Stage1 inventory only: these finite i32 ops still round-trip
  // through the dialect parser/printer so stale route inputs can be rejected
  // with targeted diagnostics. They are not positive route or artifact support.
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @policy_roundtrip attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_policy_holder attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
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
          findVariant(*module, "rvv_policy_holder"), TailPolicy::Agnostic,
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
          findVariant(*reparsed, "rvv_policy_holder"), TailPolicy::Agnostic,
          MaskPolicy::Agnostic))
    return result;

  llvm::outs() << "RVV typed policy attribute round trip preserved\n";
  return 0;
}

int runGenericDataflowRoundTripTest() {
  ExtensionPluginRegistry plugins;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
          "register RVV plugin for generic dataflow round trip"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @dataflow_roundtrip attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_explicit_dataflow attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %runtime_n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    %vl = tcrv_rvv.setvl %runtime_n {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      %sum = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse generic typed RVV vector-add dataflow ops");

  BinaryOp add;
  LoadOp load;
  StoreOp store;
  RuntimeABIValueOp runtimeABIValue;
  SetVLOp setvl;
  WithVLOp withVL;
  module->walk([&](BinaryOp candidate) { add = candidate; });
  module->walk([&](LoadOp candidate) {
    if (!load)
      load = candidate;
  });
  module->walk([&](StoreOp candidate) { store = candidate; });
  module->walk([&](RuntimeABIValueOp candidate) {
    if (!runtimeABIValue)
      runtimeABIValue = candidate;
  });
  module->walk([&](SetVLOp candidate) { setvl = candidate; });
  module->walk([&](WithVLOp candidate) { withVL = candidate; });
  if (int result =
          expect(static_cast<bool>(add), "module contains tcrv_rvv.binary"))
    return result;
  if (int result =
          expect(static_cast<bool>(setvl), "module contains tcrv_rvv.setvl"))
    return result;
  if (int result = expect(static_cast<bool>(withVL),
                          "module contains tcrv_rvv.with_vl"))
    return result;
  if (int result =
          expect(static_cast<bool>(load), "module contains tcrv_rvv.load"))
    return result;
  if (int result = expect(static_cast<bool>(store),
                          "module contains tcrv_rvv.store"))
    return result;
  if (int result = expect(static_cast<bool>(runtimeABIValue),
                          "module contains tcrv_rvv.runtime_abi_value"))
    return result;
  if (int result = expectEmitCLowerableRole(runtimeABIValue.getOperation(),
                                            "runtime_abi"))
    return result;
  if (int result = expectEmitCLowerableRole(setvl.getOperation(), "configure"))
    return result;
  if (int result = expectEmitCLowerableRole(withVL.getOperation(), "scope"))
    return result;
  if (int result = expectEmitCLowerableRole(load.getOperation(), "load"))
    return result;
  if (int result = expectEmitCLowerableRole(add.getOperation(), "compute"))
    return result;
  if (int result = expectEmitCLowerableRole(store.getOperation(), "store"))
    return result;

  std::string printedStorage;
  llvm::raw_string_ostream printedStream(printedStorage);
  module->print(printedStream);
  printedStream.flush();
  if (int result =
          expect(llvm::StringRef(printedStorage).contains("tcrv_rvv.setvl") &&
                     llvm::StringRef(printedStorage)
                         .contains("tcrv_rvv.with_vl") &&
                     llvm::StringRef(printedStorage)
                         .contains("tcrv_rvv.load") &&
                     llvm::StringRef(printedStorage)
                         .contains("tcrv_rvv.binary") &&
                     llvm::StringRef(printedStorage)
                         .contains("tcrv_rvv.store"),
                 "printed module preserves generic typed RVV dataflow body"))
    return result;

  mlir::OwningOpRef<mlir::ModuleOp> reparsed =
      parseModule(context, printedStorage);
  if (!reparsed)
    return fail("failed to reparse printed RVV dataflow module");

  BinaryOp reparsedAdd;
  reparsed->walk([&](BinaryOp candidate) { reparsedAdd = candidate; });
  if (int result = expect(static_cast<bool>(reparsedAdd),
                          "reparsed module preserves generic RVV dataflow op"))
    return result;

  llvm::outs() << "RVV generic typed vector-add dataflow round trip preserved\n";
  return 0;
}

int runLegacyI32DataflowIsParseOnlyTest() {
  ExtensionPluginRegistry plugins;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
          "register RVV plugin for legacy parse-only check"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @legacy_i32_parse_only attributes {} {
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %runtime_n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    %vl = tcrv_rvv.setvl %runtime_n {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
      %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
      %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
      tcrv_rvv.i32_store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse deprecated legacy RVV i32 dataflow fixture");

  I32AddOp add;
  I32LoadOp load;
  I32StoreOp store;
  module->walk([&](I32AddOp candidate) { add = candidate; });
  module->walk([&](I32LoadOp candidate) {
    if (!load)
      load = candidate;
  });
  module->walk([&](I32StoreOp candidate) { store = candidate; });

  if (int result =
          expect(static_cast<bool>(add), "module contains legacy i32_add"))
    return result;
  if (int result =
          expect(static_cast<bool>(load), "module contains legacy i32_load"))
    return result;
  if (int result =
          expect(static_cast<bool>(store), "module contains legacy i32_store"))
    return result;
  if (int result =
          expectNotEmitCLowerable(load.getOperation(), "legacy i32_load"))
    return result;
  if (int result =
          expectNotEmitCLowerable(add.getOperation(), "legacy i32_add"))
    return result;
  if (int result =
          expectNotEmitCLowerable(store.getOperation(), "legacy i32_store"))
    return result;

  llvm::outs() << "RVV legacy i32 dataflow remains parse-only\n";
  return 0;
}

int runI32M1ConfigVLContractAPITest() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  dialectRegistry.insert<TCRVRVVDialect>();
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  const tianchenrv::tcrv::rvv::RVVSelectedBodyConfigVLContract &contract =
      tianchenrv::tcrv::rvv::getRVVSelectedBodyM1ConfigVLContract();
  if (int result = expect(contract.sew == 32, "contract records SEW32"))
    return result;
  if (int result = expect(contract.lmul == "m1", "contract records LMUL m1"))
    return result;
  if (int result =
          expect(contract.runtimeAVLABIParameterName == "n",
                 "contract records runtime AVL ABI parameter n"))
    return result;
  if (int result =
          expect(contract.emitCLoopInductionName == "offset",
                 "contract records EmitC loop induction name"))
    return result;
  if (int result =
          expect(contract.emitCFullChunkVLName == "full_chunk_vl",
                 "contract records full-chunk VL name"))
    return result;

  PolicyAttr policy =
      tianchenrv::tcrv::rvv::getRVVSelectedBodyDefaultPolicy(&context);
  if (int result = expect(policy.getTail() == TailPolicy::Agnostic,
                          "contract creates tail-agnostic policy"))
    return result;
  if (int result = expect(policy.getMask() == MaskPolicy::Agnostic,
                          "contract creates mask-agnostic policy"))
    return result;

  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4> parameters =
      tianchenrv::tcrv::rvv::getRVVSelectedBodyRuntimeABIParameters();
  if (int result =
          expect(parameters.size() == 4,
                 "contract exposes four runtime ABI parameters"))
    return result;
  if (int result = expect(parameters[0].cName == "lhs" &&
                              parameters[1].cName == "rhs" &&
                              parameters[2].cName == "out" &&
                              parameters[3].cName == "n",
                          "contract preserves lhs,rhs,out,n ABI order"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
              parameters, "RVV dialect contract test"),
          "verify shared RVV runtime ABI contract"))
    return result;
  parameters[3].cName = "count";
  if (int result = expectFailure(
          tianchenrv::tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
              parameters, "RVV dialect contract test"),
          "reject stale runtime ABI parameter name"))
    return result;

  llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> metadata =
      tianchenrv::tcrv::rvv::getRVVSelectedBodyConfigArtifactMetadata();
  if (int result =
          expect(metadata.size() == 20,
                 "contract exposes complete artifact metadata vector"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::tcrv::rvv::verifyRVVSelectedBodyConfigArtifactMetadata(
              metadata, "RVV dialect contract test"),
          "verify shared RVV artifact metadata contract"))
    return result;

  llvm::SmallVector<tianchenrv::support::ArtifactMetadataEntry, 20>
      staleMetadata(metadata.begin(), metadata.end());
  staleMetadata[0].value = "stale-config";
  if (int result = expectFailure(
          tianchenrv::tcrv::rvv::verifyRVVSelectedBodyConfigArtifactMetadata(
              staleMetadata, "RVV dialect contract test"),
          "reject stale RVV artifact metadata"))
    return result;

  std::string remaining =
      tianchenrv::tcrv::rvv::getRVVSelectedBodyEmitCRemainingAVLExpression(
          contract.runtimeAVLABIParameterName, contract.emitCLoopInductionName);
  if (int result = expect(remaining == "n - offset",
                          "contract formats EmitC remaining AVL expression"))
    return result;

  llvm::outs() << "RVV selected-body config/VL contract API preserved\n";
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
    tcrv.exec.variant @rvv_policy_holder attributes {
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
  if (int result = runGenericDataflowRoundTripTest())
    return result;
  if (int result = runLegacyI32DataflowIsParseOnlyTest())
    return result;
  if (int result = runI32M1ConfigVLContractAPITest())
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
