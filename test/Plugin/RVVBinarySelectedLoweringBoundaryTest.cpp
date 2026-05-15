#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/RVV/RVVBinarySelectedLoweringBoundary.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/RVV/RVVBinaryFamily.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>

using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::plugin::VariantLegalityRequest;
using tianchenrv::plugin::VariantLoweringBoundaryRequest;
using tianchenrv::plugin::VariantLoweringBoundaryResult;
using tianchenrv::plugin::VariantLoweringBoundaryValidationRequest;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;
using tianchenrv::tcrv::rvv::LoweringBoundaryOp;

namespace {

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

int expectErrorContains(llvm::Error error,
                        std::initializer_list<llvm::StringRef> fragments) {
  if (!error)
    return fail("expected RVV selected-boundary error");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("error text missing '") + fragment +
                  "': " + message);
  }
  return 0;
}

mlir::OwningOpRef<mlir::ModuleOp>
parseModule(mlir::MLIRContext &context, llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

KernelOp findKernel(mlir::ModuleOp module, llvm::StringRef name) {
  KernelOp kernel;
  module.walk([&](KernelOp candidate) {
    if (candidate.getSymName() == name)
      kernel = candidate;
  });
  return kernel;
}

VariantOp findVariant(KernelOp kernel, llvm::StringRef name) {
  VariantOp variant;
  kernel->walk([&](VariantOp candidate) {
    if (candidate.getSymName() == name)
      variant = candidate;
  });
  return variant;
}

LoweringBoundaryOp findBoundary(KernelOp kernel,
                                llvm::StringRef selectedVariantSymbol) {
  LoweringBoundaryOp boundary;
  if (!kernel || kernel.getBody().empty())
    return boundary;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto candidate = llvm::dyn_cast<LoweringBoundaryOp>(op);
    if (!candidate)
      continue;
    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    if (selectedVariant && selectedVariant.getValue() == selectedVariantSymbol)
      boundary = candidate;
  }
  return boundary;
}

mlir::Operation *findSelectedOpByName(KernelOp kernel,
                                      llvm::StringRef selectedVariantSymbol,
                                      llvm::StringRef opName) {
  if (!kernel || kernel.getBody().empty())
    return nullptr;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (op.getName().getStringRef() != opName)
      continue;
    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    if (selectedVariant && selectedVariant.getValue() == selectedVariantSymbol)
      return &op;
  }
  return nullptr;
}

llvm::StringRef getStringAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op ? op->getAttrOfType<mlir::StringAttr>(attrName)
                 : mlir::StringAttr();
  if (!attr)
    return {};
  return attr.getValue();
}

void replaceAll(std::string &text, llvm::StringRef from, llvm::StringRef to) {
  std::size_t position = 0;
  while ((position = text.find(from.str(), position)) != std::string::npos) {
    text.replace(position, from.size(), to.str());
    position += to.size();
  }
}

int expectStringAttr(mlir::Operation *op, llvm::StringRef attrName,
                     llvm::StringRef expected) {
  return expect(getStringAttr(op, attrName) == expected,
                llvm::Twine("attribute '") + attrName +
                    "' preserves expected value");
}

int expectIntegerAttr(mlir::Operation *op, llvm::StringRef attrName,
                      std::int64_t expected) {
  auto attr = op ? op->getAttrOfType<mlir::IntegerAttr>(attrName)
                 : mlir::IntegerAttr();
  if (int result =
          expect(static_cast<bool>(attr),
                 llvm::Twine("expected integer attribute '") + attrName + "'"))
    return result;
  return expect(attr.getInt() == expected,
                llvm::Twine("integer attribute '") + attrName +
                    "' preserves expected value");
}

llvm::Error materializeWithModuleAPI(
    tianchenrv::plugin::rvv::RVVExtensionPlugin &plugin,
    mlir::OpBuilder &builder, VariantOp variant, KernelOp kernel,
    const TargetCapabilitySet &capabilities,
    VariantLoweringBoundaryResult &out) {
  return tianchenrv::plugin::rvv::
      materializeRVVBinarySelectedLoweringBoundary(
          VariantLoweringBoundaryRequest(
              variant, kernel, capabilities,
              VariantEmissionRole::DirectVariant, builder),
          out, tianchenrv::plugin::rvv::getRVVExtensionPluginName(),
          [&](const VariantLegalityRequest &legality) {
            return plugin.verifyVariantLegality(legality);
          });
}

llvm::Error validateWithModuleAPI(
    tianchenrv::plugin::rvv::RVVExtensionPlugin &plugin, VariantOp variant,
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    LoweringBoundaryOp boundary) {
  return tianchenrv::plugin::rvv::
      validateRVVBinarySelectedLoweringBoundary(
          VariantLoweringBoundaryValidationRequest(
              variant, kernel, capabilities, VariantEmissionRole::DirectVariant,
              boundary.getOperation()),
          tianchenrv::plugin::rvv::getRVVExtensionPluginName(),
          [&](const VariantLegalityRequest &legality) {
            return plugin.verifyVariantLegality(legality);
          });
}

int runI32SelectedLoweringBoundaryModuleTest(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
	  tcrv.exec.kernel @rvv_i32_vsub_boundary attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl256b",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_sew32 {
      id = "rvv.i32_m1.sew32",
      kind = "isa-vector-config",
      sew_bits = 32 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_lmul_m1 {
      id = "rvv.i32_m1.lmul_m1",
      kind = "isa-vector-config",
      lmul = "m1",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_tail_agnostic {
      id = "rvv.i32_m1.tail_policy.agnostic",
      kind = "isa-vector-config",
      tail_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_mask_agnostic {
      id = "rvv.i32_m1.mask_policy.agnostic",
      kind = "isa-vector-config",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_vlenb_bytes {
      id = "rvv.vlenb_bytes",
      kind = "uarch",
      bytes = 32 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_lanes {
      id = "rvv.i32_m1_lane_count",
      kind = "uarch",
      lanes = 8 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv, @rvv_i32_m1_sew32, @rvv_i32_m1_lmul_m1, @rvv_i32_m1_tail_agnostic, @rvv_i32_m1_mask_agnostic],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
	      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.vlenb_bytes = 32 : i64,
      tcrv_rvv.base_i32_m1_lanes = 8 : i64,
      tcrv_rvv.selected_vector_shape = "i32m1",
      tcrv_rvv.selected_vector_sew = 32 : i64,
      tcrv_rvv.selected_vector_lmul = "m1",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_vector_type = "vint32m1_t",
      tcrv_rvv.selected_vector_suffix = "i32m1",
      tcrv_rvv.selected_setvl_suffix = "e32m1"
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse i32 selected-boundary module");

  KernelOp kernel = findKernel(*module, "rvv_i32_vsub_boundary");
  VariantOp variant = findVariant(kernel, "rvv_first_slice");
  if (int result = expect(kernel && variant, "i32 test has kernel/variant"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  tianchenrv::plugin::rvv::RVVExtensionPlugin plugin;
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToEnd(&kernel.getBody().front());

  VariantLoweringBoundaryResult result;
  if (int status = expectSuccess(
          materializeWithModuleAPI(plugin, builder, variant, kernel,
                                   capabilities, result),
          "materialize i32 selected lowering boundary through module API"))
    return status;
  if (int status = expect(result.isMaterialized(),
                          "i32 selected boundary result is materialized"))
    return status;

  LoweringBoundaryOp boundary = findBoundary(kernel, variant.getSymName());
  if (int status = expect(static_cast<bool>(boundary),
                          "i32 selected boundary op exists"))
    return status;
  if (int status =
          expectStringAttr(boundary.getOperation(), "status", "unsupported"))
    return status;
  if (int status = expectStringAttr(boundary.getOperation(),
                                    "selected_vector_shape", "i32m1"))
    return status;
  if (int status =
          expectIntegerAttr(boundary.getOperation(), "selected_vector_sew", 32))
    return status;
  if (int status =
          expectStringAttr(boundary.getOperation(), "selected_vector_type",
                           "vint32m1_t"))
    return status;
  if (int status =
          expectIntegerAttr(boundary.getOperation(), "vlenb_bytes", 32))
    return status;
  if (int status =
          expectIntegerAttr(boundary.getOperation(), "base_i32_m1_lanes", 8))
    return status;
  if (int status =
          expectStringAttr(boundary.getOperation(), "capability_summary",
                           "rvv,rvv.i32_m1.sew32,rvv.i32_m1.lmul_m1,"
                           "rvv.i32_m1.tail_policy.agnostic,"
                           "rvv.i32_m1.mask_policy.agnostic"))
    return status;

  mlir::Operation *microkernel = findSelectedOpByName(
      kernel, variant.getSymName(),
      tianchenrv::target::rvv::getI32VSubFamilyRegistrationRecord().microkernelOpName);
  if (int status = expect(!microkernel,
                          "i32 selected boundary no longer materializes vsub "
                          "microkernel from descriptor/family records"))
    return status;

  if (int status = expectSuccess(
          validateWithModuleAPI(plugin, variant, kernel, capabilities,
                                boundary),
          "validate i32 selected boundary through module API"))
    return status;

  VariantLoweringBoundaryResult duplicateResult;
  if (int status = expectErrorContains(
          materializeWithModuleAPI(plugin, builder, variant, kernel,
                                   capabilities, duplicateResult),
          {"pre-existing tcrv_rvv.lowering_boundary", "@rvv_first_slice"}))
    return status;

  boundary->setAttr("vlenb_bytes", builder.getI64IntegerAttr(16));
  boundary->setAttr("base_i32_m1_lanes", builder.getI64IntegerAttr(4));
  return expectErrorContains(
      validateWithModuleAPI(plugin, variant, kernel, capabilities, boundary),
      {"capacity metadata", "does not match"});
}

int runDefaultI32VAddSelectedLoweringBoundaryModuleTest(
    mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @rvv_default_i32_vadd_boundary attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.element_count = 16 : i64,
      tcrv_rvv.selected_binary_dtype = "i32",
      tcrv_rvv.selected_binary_family = "i32-vadd",
      tcrv_rvv.selected_binary_operator = "add",
      tcrv_rvv.selected_binary_source_kind = "default-i32-vadd-typed-body-materialization",
      tcrv_rvv.selected_vector_shape = "i32m1",
      tcrv_rvv.selected_vector_sew = 32 : i64,
      tcrv_rvv.selected_vector_lmul = "m1",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_vector_type = "vint32m1_t",
      tcrv_rvv.selected_vector_suffix = "i32m1",
      tcrv_rvv.selected_setvl_suffix = "e32m1"
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse default i32-vadd selected-boundary module");

  KernelOp kernel = findKernel(*module, "rvv_default_i32_vadd_boundary");
  VariantOp variant = findVariant(kernel, "rvv_first_slice");
  if (int result =
          expect(kernel && variant, "default i32-vadd test has kernel/variant"))
    return result;
  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  tianchenrv::plugin::rvv::RVVExtensionPlugin plugin;
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToEnd(&kernel.getBody().front());

  VariantLoweringBoundaryResult result;
  return expectErrorContains(
      materializeWithModuleAPI(plugin, builder, variant, kernel, capabilities,
                               result),
      {"RVV selected-source metadata", "deleted as RVV finite-family legality "
                                      "authority"});
}

int runI64SelectedLoweringBoundaryModuleTest(
    mlir::MLIRContext &context,
    const tianchenrv::target::rvv::RVVBinaryFamilyRecord &family) {
  std::string kernelSymbol =
      (llvm::Twine("rvv_") + family.functionStem + "_boundary").str();
  std::string source = R"mlir(
module {
	  tcrv.exec.kernel @rvv_i64_vmul_boundary attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_i64_m1_sew64 {
      id = "rvv.i64_m1.sew64",
      kind = "isa-vector-config",
      sew_bits = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i64_m1_lmul_m1 {
      id = "rvv.i64_m1.lmul_m1",
      kind = "isa-vector-config",
      lmul = "m1",
      status = "available"
    }
    tcrv.exec.capability @rvv_i64_m1_tail_agnostic {
      id = "rvv.i64_m1.tail_policy.agnostic",
      kind = "isa-vector-config",
      tail_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_i64_m1_mask_agnostic {
      id = "rvv.i64_m1.mask_policy.agnostic",
      kind = "isa-vector-config",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv, @rvv_i64_m1_sew64, @rvv_i64_m1_lmul_m1, @rvv_i64_m1_tail_agnostic, @rvv_i64_m1_mask_agnostic],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
	      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.selected_vector_shape = "i64m1",
      tcrv_rvv.selected_vector_sew = 64 : i64,
      tcrv_rvv.selected_vector_lmul = "m1",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_vector_type = "vint64m1_t",
      tcrv_rvv.selected_vector_suffix = "i64m1",
      tcrv_rvv.selected_setvl_suffix = "e64m1"
    } {
    }
  }
}
)mlir";
  replaceAll(source, "rvv_i64_vmul_boundary", kernelSymbol);

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse i64 selected-boundary module");

  KernelOp kernel = findKernel(*module, kernelSymbol);
  VariantOp variant = findVariant(kernel, "rvv_first_slice");
  if (int result = expect(kernel && variant, "i64 test has kernel/variant"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  tianchenrv::plugin::rvv::RVVExtensionPlugin plugin;
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToEnd(&kernel.getBody().front());

  VariantLoweringBoundaryResult result;
  if (int status = expectSuccess(
          materializeWithModuleAPI(plugin, builder, variant, kernel,
                                   capabilities, result),
          llvm::Twine("materialize ") + family.familyID +
              " selected lowering boundary through module API"))
    return status;
  if (int status = expect(result.isMaterialized(),
                          llvm::Twine(family.familyID) +
                              " selected boundary result is materialized"))
    return status;

  LoweringBoundaryOp boundary = findBoundary(kernel, variant.getSymName());
  if (int status = expect(static_cast<bool>(boundary),
                          "i64 selected boundary op exists"))
    return status;
  if (int status = expectStringAttr(boundary.getOperation(),
                                    "selected_vector_shape", "i64m1"))
    return status;
  if (int status =
          expectIntegerAttr(boundary.getOperation(), "selected_vector_sew", 64))
    return status;
  if (int status =
          expectStringAttr(boundary.getOperation(), "selected_vector_type",
                           "vint64m1_t"))
    return status;
  if (int status = expect(!boundary->hasAttr("vlenb_bytes") &&
                              !boundary->hasAttr("base_i32_m1_lanes"),
                          "i64 selected boundary does not invent capacity "
                          "metadata"))
    return status;

  mlir::Operation *microkernel = findSelectedOpByName(
      kernel, variant.getSymName(), family.microkernelOpName);
  if (int status =
          expect(!microkernel,
                 llvm::Twine("i64 selected boundary no longer materializes ") +
                     family.familyID + " microkernel from selected descriptor"))
    return status;

  return expectSuccess(
      validateWithModuleAPI(plugin, variant, kernel, capabilities, boundary),
      llvm::Twine("validate ") + family.familyID +
          " selected boundary through module API");
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::plugin::ExtensionPluginRegistry plugins;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
                        "register RVV plugin for selected-boundary test"))
    return result;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runI32SelectedLoweringBoundaryModuleTest(context))
    return result;
  if (int result =
          runDefaultI32VAddSelectedLoweringBoundaryModuleTest(context))
    return result;
  if (int result = runI64SelectedLoweringBoundaryModuleTest(
          context, tianchenrv::target::rvv::getI64VSubFamilyRegistrationRecord()))
    return result;
  if (int result = runI64SelectedLoweringBoundaryModuleTest(
          context, tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord()))
    return result;

  llvm::outs() << "RVV binary selected lowering-boundary tests passed\n";
  return 0;
}
