#include "TianChenRV/Plugin/Toy/ToySelectedBoundarySeed.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/Toy/IR/ToyDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"
#include "TianChenRV/Plugin/Toy/ToyExtensionPlugin.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <string>

namespace tianchenrv::plugin::toy {
namespace {

constexpr llvm::StringLiteral kSeedAttrName("tcrv_toy.lowering_seed");
constexpr llvm::StringLiteral kSeedAttrValue("template_compute");
constexpr llvm::StringLiteral kSelectedMessage(
    "bounded MLIR Toy template seed selected Toy compute_skeleton boundary");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kToyConstructionProtocolAttrName(
    "tcrv_toy.construction_protocol");
constexpr llvm::StringLiteral kToyConstructionArchetypeAttrName(
    "tcrv_toy.archetype");
constexpr llvm::StringLiteral kToySemanticRoleGraphAttrName(
    "tcrv_toy.semantic_role_graph");
constexpr llvm::StringLiteral kToyCommonInterfaceRealizationAttrName(
    "tcrv_toy.common_interface_realization");
constexpr llvm::StringLiteral kToyTypedRoleRealizationAttrName(
    "tcrv_toy.typed_role_realization");
constexpr llvm::StringLiteral kToyEmitCRouteMappingAttrName(
    "tcrv_toy.emitc_route_mapping");
constexpr llvm::StringLiteral kToyEvidenceProfileAttrName(
    "tcrv_toy.evidence_profile");
constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kRoleOrderAttrName("role_order");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");
constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");
constexpr llvm::StringLiteral kToyComputeTypedRoleID(
    "toy.role.compute.compute_skeleton");
constexpr llvm::StringLiteral kToyComputeSourceRole("compute");
constexpr llvm::StringLiteral kToyComputeRoleSpecificInterface(
    "TCRVComputeOpInterface");
constexpr int64_t kToyComputeRoleOrder = 2;

mlir::LogicalResult failSeed(mlir::Operation *op, llvm::StringRef message) {
  op->emitError() << "bounded Toy template selected-boundary seed failed: "
                  << message;
  return mlir::failure();
}

mlir::LogicalResult requireSourceOnlyModule(mlir::ModuleOp module) {
  mlir::Operation *staleExecOp = nullptr;
  mlir::Operation *staleToyOp = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (!staleExecOp && op->getName().getDialectNamespace() == "tcrv")
      staleExecOp = op;
    if (!staleToyOp && op->getName().getDialectNamespace() == "tcrv_toy")
      staleToyOp = op;
  });
  mlir::Operation *staleOp = staleToyOp ? staleToyOp : staleExecOp;
  if (!staleOp)
    return mlir::success();

  return failSeed(staleOp,
                  "source seed pass requires source-only MLIR input; "
                  "pre-existing tcrv.exec/tcrv_toy selected-boundary or "
                  "unselected variant residue is not accepted");
}

bool isFuncReturnWithoutOperands(mlir::Operation *op) {
  auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(op);
  return returnOp && returnOp.getNumOperands() == 0;
}

mlir::LogicalResult matchBoundedToyTemplateSeed(mlir::func::FuncOp func) {
  auto seedAttr = func->getAttrOfType<mlir::StringAttr>(kSeedAttrName);
  if (!seedAttr)
    return mlir::success();
  if (seedAttr.getValue() != kSeedAttrValue)
    return failSeed(func, "unsupported Toy lowering seed attribute value");

  mlir::FunctionType functionType = func.getFunctionType();
  if (functionType.getNumResults() != 0)
    return failSeed(func, "source function must not return values");
  if (functionType.getNumInputs() != 0)
    return failSeed(func,
                    "source function must expose no runtime ABI operands; "
                    "Toy template seed carries no runtime execution claim");
  if (!llvm::hasSingleElement(func.getBody()))
    return failSeed(func, "source function must have one block");

  unsigned returnCount = 0;
  for (mlir::Operation &op : func.getBody().front()) {
    if (isFuncReturnWithoutOperands(&op)) {
      ++returnCount;
      continue;
    }
    return failSeed(&op,
                    "source function body may contain only one empty return");
  }
  if (returnCount != 1)
    return failSeed(func, "source function must contain one empty return");

  return mlir::success();
}

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

void addToyVariantConstructionMetadata(mlir::OpBuilder &builder,
                                       mlir::OperationState &state) {
  const ToyConstructionManifest &manifest = getToyConstructionManifest();
  state.addAttribute(getToyTemplateABIAttrName(),
                     builder.getStringAttr(getToyExpectedTemplateABI()));
  state.addAttribute(getToyHandoffKindAttrName(),
                     builder.getStringAttr(getToyExpectedHandoffKind()));
  state.addAttribute(kToyConstructionProtocolAttrName,
                     builder.getStringAttr(manifest.protocolVersion));
  state.addAttribute(kToyConstructionArchetypeAttrName,
                     builder.getStringAttr(manifest.archetype));
  state.addAttribute(kToySemanticRoleGraphAttrName,
                     builder.getStringAttr(manifest.semanticRoleGraph));
  state.addAttribute(kToyCommonInterfaceRealizationAttrName,
                     builder.getStringAttr(
                         getToyConstructionInterfaceRealization()));
  state.addAttribute(kToyTypedRoleRealizationAttrName,
                     builder.getStringAttr(getToyTypedRoleRealizationSummary()));
  state.addAttribute(kToyEmitCRouteMappingAttrName,
                     builder.getStringAttr(manifest.emitcRoute.routeID));
  state.addAttribute(kToyEvidenceProfileAttrName,
                     builder.getStringAttr(manifest.evidenceProfile));
}

void createToyComputeSkeletonBoundary(mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      llvm::StringRef kernelName,
                                      llvm::StringRef variantName,
                                      mlir::ArrayAttr requires) {
  mlir::OperationState state(loc, "tcrv_toy.compute_skeleton");
  state.addAttribute(kSourceKernelAttrName, builder.getStringAttr(kernelName));
  state.addAttribute(kSelectedVariantAttrName, symbolRef(builder, variantName));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(getToyExtensionPluginName()));
  state.addAttribute(
      kRoleAttrName,
      builder.getStringAttr(
          stringifyVariantEmissionRole(VariantEmissionRole::DirectVariant)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kRoleOpBoundaryStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requires);
  state.addAttribute(kTypedRoleAttrName,
                     builder.getStringAttr(kToyComputeTypedRoleID));
  state.addAttribute(kRoleOrderAttrName,
                     builder.getI64IntegerAttr(kToyComputeRoleOrder));
  state.addAttribute(kSourceRoleAttrName,
                     builder.getStringAttr(kToyComputeSourceRole));
  state.addAttribute(kRoleSpecificInterfaceAttrName,
                     builder.getStringAttr(kToyComputeRoleSpecificInterface));
  (void)builder.create(state);
}

void createSelectedDiagnostic(mlir::OpBuilder &builder, mlir::Location loc,
                              llvm::StringRef variantName) {
  mlir::OperationState state(loc, "tcrv.exec.diagnostic");
  state.addAttribute("reason", builder.getStringAttr("variant-selected"));
  state.addAttribute("message", builder.getStringAttr(kSelectedMessage));
  state.addAttribute("severity", builder.getStringAttr("note"));
  state.addAttribute("status", builder.getStringAttr("selected"));
  state.addAttribute("selection_kind", builder.getStringAttr("static-variant"));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(getToyExtensionPluginName()));
  state.addAttribute("target", symbolRef(builder, variantName));
  (void)builder.create(state);
}

void materializeSeedKernel(mlir::OpBuilder &builder, mlir::func::FuncOp func) {
  mlir::Location loc = func.getLoc();
  std::string kernelName = (func.getSymName() + "_kernel").str();
  std::string variantName = (func.getSymName() + "_toy_template").str();

  mlir::OperationState kernelState(loc, "tcrv.exec.kernel");
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addRegion();
  auto kernel =
      llvm::cast<tianchenrv::tcrv::exec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  llvm::StringRef capabilitySymbol = getToyTemplatePreferredCapabilitySymbol();
  mlir::OperationState capabilityState(loc, "tcrv.exec.capability");
  capabilityState.addAttribute("sym_name",
                               builder.getStringAttr(capabilitySymbol));
  capabilityState.addAttribute("id",
                               builder.getStringAttr(getToyTemplateCapabilityID()));
  capabilityState.addAttribute(
      "kind", builder.getStringAttr(getToyTemplateCapabilityKind()));
  capabilityState.addAttribute("status", builder.getStringAttr("available"));
  capabilityState.addAttribute("template_abi",
                               builder.getStringAttr(getToyExpectedTemplateABI()));
  capabilityState.addAttribute("handoff_kind",
                               builder.getStringAttr(getToyExpectedHandoffKind()));
  (void)builder.create(capabilityState);

  mlir::ArrayAttr requires =
      builder.getArrayAttr({symbolRef(builder, capabilitySymbol)});
  mlir::OperationState variantState(loc, "tcrv.exec.variant");
  variantState.addAttribute("sym_name", builder.getStringAttr(variantName));
  variantState.addAttribute(kOriginAttrName,
                            builder.getStringAttr(getToyExtensionPluginName()));
  variantState.addAttribute(kRequiresAttrName, requires);
  addToyVariantConstructionMetadata(builder, variantState);
  variantState.addRegion();
  auto variant =
      llvm::cast<tianchenrv::tcrv::exec::VariantOp>(builder.create(variantState));
  variant.getBody().emplaceBlock();

  createToyComputeSkeletonBoundary(builder, loc, kernelName, variantName,
                                   requires);
  createSelectedDiagnostic(builder, loc, variantName);
}

class MaterializeToyTemplateSelectedBoundarySeedPass final
    : public mlir::PassWrapper<
          MaterializeToyTemplateSelectedBoundarySeedPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const final {
    return "tcrv-toy-materialize-template-selected-boundary-seed";
  }

  llvm::StringRef getDescription() const final {
    return "Materialize one bounded Toy template source seed into the Toy "
           "selected-boundary compute_skeleton form";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::func::FuncDialect,
                    tianchenrv::tcrv::exec::TCRVExecDialect,
                    tianchenrv::tcrv::toy::TCRVToyDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<mlir::func::FuncOp, 2> seeds;
    module.walk([&](mlir::func::FuncOp func) {
      if (func->hasAttr(kSeedAttrName))
        seeds.push_back(func);
    });
    if (seeds.empty())
      return;

    if (mlir::failed(requireSourceOnlyModule(module))) {
      signalPassFailure();
      return;
    }

    for (mlir::func::FuncOp func : seeds) {
      if (mlir::failed(matchBoundedToyTemplateSeed(func))) {
        signalPassFailure();
        return;
      }
    }

    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    for (mlir::func::FuncOp func : seeds)
      materializeSeedKernel(builder, func);

    for (mlir::func::FuncOp func : seeds)
      func.erase();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass>
createMaterializeToyTemplateSelectedBoundarySeedPass() {
  return std::make_unique<MaterializeToyTemplateSelectedBoundarySeedPass>();
}

} // namespace tianchenrv::plugin::toy
