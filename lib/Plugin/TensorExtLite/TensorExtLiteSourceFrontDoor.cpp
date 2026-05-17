#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteSourceFrontDoor.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <cctype>
#include <cstdint>
#include <memory>
#include <string>

namespace tianchenrv::plugin::tensorext_lite {
namespace {

constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_tensorext_lite.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName(
    "tcrv_tensorext_lite.source_kernel");
constexpr llvm::StringLiteral kAcceptedSourceFrontDoorValue(
    "fragment_mma_template");
constexpr llvm::StringLiteral kDefaultKernelName(
    "tensorext_lite_source_front_door");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kSourceKernelBoundaryAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kRoleOrderAttrName("role_order");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");
constexpr llvm::StringLiteral kFragmentReasonAttrName("fragment_reason");
constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");
constexpr llvm::StringLiteral kLoweringBoundaryStatusValue("no-active-route");
constexpr llvm::StringLiteral kSelectedDiagnosticMessage(
    "selected TensorExtLite source front-door route");
constexpr llvm::StringLiteral kFragmentReason(
    "tensorext-lite-source-front-door-fragment-mma-template");

struct TensorExtLiteSourceRoleSpec {
  llvm::StringLiteral operationName;
  llvm::StringLiteral typedRole;
  llvm::StringLiteral sourceRole;
  llvm::StringLiteral roleSpecificInterface;
  std::int64_t roleOrder = 0;
};

constexpr TensorExtLiteSourceRoleSpec kRoleSpecs[] = {
    {"tcrv_tensorext_lite.config_skeleton", "tel.role.config", "configure",
     "TCRVConfigOpInterface", 0},
    {"tcrv_tensorext_lite.load_frag_skeleton", "tel.role.load_frag",
     "load_frag", "TCRVMemoryOpInterface", 1},
    {"tcrv_tensorext_lite.tile_mma_skeleton", "tel.role.tile_mma",
     "tile_mma", "TCRVComputeOpInterface", 2},
    {"tcrv_tensorext_lite.store_frag_skeleton", "tel.role.store_frag",
     "store_frag", "TCRVMemoryOpInterface", 3},
};

mlir::LogicalResult failMaterializer(mlir::Operation *op,
                                     llvm::StringRef message) {
  op->emitError()
      << "bounded TensorExtLite fragment-MMA source front door failed: "
      << message;
  return mlir::failure();
}

bool isBoundedSymbolName(llvm::StringRef value) {
  if (value.empty())
    return false;

  auto isFirst = [](char character) {
    unsigned char byte = static_cast<unsigned char>(character);
    return std::isalpha(byte) || character == '_';
  };
  auto isRest = [](char character) {
    unsigned char byte = static_cast<unsigned char>(character);
    return std::isalnum(byte) || character == '_' || character == '$';
  };

  if (!isFirst(value.front()))
    return false;
  for (char character : value.drop_front()) {
    if (!isRest(character))
      return false;
  }
  return true;
}

bool hasStaleTensorExtLiteLoweringSeedMetadata(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *op) {
    if (found)
      return;
    found = op->hasAttr("tcrv_tensorext_lite.lowering_seed");
  });
  return found;
}

mlir::LogicalResult requireSourceOnlyModule(mlir::ModuleOp module) {
  mlir::Operation *staleOp = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (staleOp || op == module.getOperation())
      return;
    llvm::StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "tcrv" || dialect == "tcrv_tensorext_lite" ||
        dialect == "tcrv_rvv" || dialect == "tcrv_toy")
      staleOp = op;
  });
  if (!staleOp)
    return mlir::success();

  return failMaterializer(
      staleOp,
      "source materializer requires TensorExtLite source-only MLIR input; "
      "pre-existing tcrv.exec/tcrv_tensorext_lite/tcrv_rvv/tcrv_toy "
      "selected-boundary or variant residue is not accepted");
}

mlir::FailureOr<std::string> getSourceKernelName(mlir::ModuleOp module) {
  auto kernelNameAttr =
      module->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  llvm::StringRef kernelName = kernelNameAttr
                                   ? kernelNameAttr.getValue().trim()
                                   : llvm::StringRef(kDefaultKernelName);
  if (kernelName.empty()) {
    (void)failMaterializer(module, "source kernel name must be non-empty");
    return mlir::failure();
  }
  if (!isBoundedSymbolName(kernelName)) {
    (void)failMaterializer(
        module, "source kernel name must be a valid MLIR symbol");
    return mlir::failure();
  }
  return kernelName.str();
}

mlir::FailureOr<std::string>
matchTensorExtLiteSourceFrontDoor(mlir::ModuleOp module) {
  auto marker =
      module->getAttrOfType<mlir::StringAttr>(kSourceFrontDoorAttrName);
  if (!marker)
    return std::string();

  if (marker.getValue().trim() != kAcceptedSourceFrontDoorValue) {
    (void)failMaterializer(
        module,
        "tcrv_tensorext_lite.source_front_door must be "
        "'fragment_mma_template'");
    return mlir::failure();
  }
  if (hasStaleTensorExtLiteLoweringSeedMetadata(module)) {
    (void)failMaterializer(
        module,
        "stale tcrv_tensorext_lite.lowering_seed metadata is not accepted "
        "as TensorExtLite source-route authority");
    return mlir::failure();
  }
  if (mlir::failed(requireSourceOnlyModule(module)))
    return mlir::failure();

  return getSourceKernelName(module);
}

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

void createTensorExtLiteCapability(mlir::OpBuilder &builder,
                                   mlir::Location loc) {
  mlir::OperationState state(loc, "tcrv.exec.capability");
  state.addAttribute(
      "sym_name",
      builder.getStringAttr(getTensorExtLiteFragmentPreferredCapabilitySymbol()));
  state.addAttribute(
      "id", builder.getStringAttr(getTensorExtLiteFragmentCapabilityID()));
  state.addAttribute(
      "kind", builder.getStringAttr(getTensorExtLiteFragmentCapabilityKind()));
  state.addAttribute("status", builder.getStringAttr("available"));
  state.addAttribute(
      "fragment_abi",
      builder.getStringAttr(getTensorExtLiteExpectedFragmentABI()));
  state.addAttribute(
      "handoff_kind",
      builder.getStringAttr(getTensorExtLiteExpectedHandoffKind()));
  (void)builder.create(state);
}

void createTensorExtLiteVariant(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::ArrayAttr requires) {
  const TensorExtLiteConstructionManifest &manifest =
      getTensorExtLiteConstructionManifest();

  mlir::OperationState state(loc, "tcrv.exec.variant");
  state.addAttribute(
      "sym_name",
      builder.getStringAttr(getTensorExtLiteFragmentFirstSliceVariantName()));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(getTensorExtLiteExtensionPluginName()));
  state.addAttribute(kRequiresAttrName, requires);
  state.addAttribute(
      getTensorExtLiteFragmentABIAttrName(),
      builder.getStringAttr(getTensorExtLiteExpectedFragmentABI()));
  state.addAttribute(
      getTensorExtLiteHandoffKindAttrName(),
      builder.getStringAttr(getTensorExtLiteExpectedHandoffKind()));
  state.addAttribute("tcrv_tensorext_lite.construction_protocol",
                     builder.getStringAttr(manifest.protocolVersion));
  state.addAttribute("tcrv_tensorext_lite.archetype",
                     builder.getStringAttr(manifest.archetype));
  state.addAttribute("tcrv_tensorext_lite.semantic_role_graph",
                     builder.getStringAttr(manifest.semanticRoleGraph));
  state.addAttribute(
      "tcrv_tensorext_lite.common_interface_realization",
      builder.getStringAttr(getTensorExtLiteConstructionInterfaceRealization()));
  state.addAttribute(
      "tcrv_tensorext_lite.typed_role_realization",
      builder.getStringAttr(getTensorExtLiteTypedRoleRealizationSummary()));
  state.addAttribute("tcrv_tensorext_lite.emitc_route_mapping",
                     builder.getStringAttr(manifest.emitcRoute.routeID));
  state.addAttribute("tcrv_tensorext_lite.evidence_profile",
                     builder.getStringAttr(manifest.evidenceProfile));
  state.addRegion();
  auto variant = llvm::cast<tcrv::exec::VariantOp>(builder.create(state));
  variant.getBody().emplaceBlock();
}

void createTensorExtLiteRoleOp(mlir::OpBuilder &builder, mlir::Location loc,
                               const TensorExtLiteSourceRoleSpec &spec,
                               llvm::StringRef kernelName,
                               mlir::ArrayAttr requires) {
  mlir::OperationState state(loc, spec.operationName);
  state.addAttribute(kSourceKernelBoundaryAttrName,
                     builder.getStringAttr(kernelName));
  state.addAttribute(
      kSelectedVariantAttrName,
      symbolRef(builder, getTensorExtLiteFragmentFirstSliceVariantName()));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(getTensorExtLiteExtensionPluginName()));
  state.addAttribute(
      kRoleAttrName,
      builder.getStringAttr(
          stringifyVariantEmissionRole(VariantEmissionRole::DirectVariant)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kRoleOpBoundaryStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requires);
  state.addAttribute(kTypedRoleAttrName, builder.getStringAttr(spec.typedRole));
  state.addAttribute(kRoleOrderAttrName,
                     builder.getI64IntegerAttr(spec.roleOrder));
  state.addAttribute(kSourceRoleAttrName,
                     builder.getStringAttr(spec.sourceRole));
  state.addAttribute(kRoleSpecificInterfaceAttrName,
                     builder.getStringAttr(spec.roleSpecificInterface));
  state.addAttribute(kFragmentReasonAttrName,
                     builder.getStringAttr(kFragmentReason));
  (void)builder.create(state);
}

void createTensorExtLiteLoweringBoundary(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         llvm::StringRef kernelName,
                                         mlir::ArrayAttr requires) {
  mlir::OperationState state(loc, "tcrv_tensorext_lite.lowering_boundary");
  state.addAttribute(kSourceKernelBoundaryAttrName,
                     builder.getStringAttr(kernelName));
  state.addAttribute(
      kSelectedVariantAttrName,
      symbolRef(builder, getTensorExtLiteFragmentFirstSliceVariantName()));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(getTensorExtLiteExtensionPluginName()));
  state.addAttribute(
      kRoleAttrName,
      builder.getStringAttr(
          stringifyVariantEmissionRole(VariantEmissionRole::DirectVariant)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kLoweringBoundaryStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requires);
  state.addAttribute("fragment_abi",
                     builder.getStringAttr(getTensorExtLiteExpectedFragmentABI()));
  state.addAttribute("handoff_kind",
                     builder.getStringAttr(getTensorExtLiteExpectedHandoffKind()));
  state.addAttribute(kFragmentReasonAttrName,
                     builder.getStringAttr(kFragmentReason));
  (void)builder.create(state);
}

void createSelectedTensorExtLiteDiagnostic(mlir::OpBuilder &builder,
                                           mlir::Location loc) {
  mlir::OperationState state(loc, "tcrv.exec.diagnostic");
  state.addAttribute("message",
                     builder.getStringAttr(kSelectedDiagnosticMessage));
  state.addAttribute("reason", builder.getStringAttr("variant-selected"));
  state.addAttribute("selection_kind", builder.getStringAttr("static-variant"));
  state.addAttribute("severity", builder.getStringAttr("note"));
  state.addAttribute("status", builder.getStringAttr("selected"));
  state.addAttribute(
      "target", symbolRef(builder, getTensorExtLiteFragmentFirstSliceVariantName()));
  (void)builder.create(state);
}

void materializeTensorExtLiteSourceKernel(mlir::OpBuilder &builder,
                                          mlir::ModuleOp module,
                                          llvm::StringRef kernelName) {
  mlir::Location loc = module.getLoc();

  mlir::OperationState kernelState(loc, "tcrv.exec.kernel");
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addRegion();
  auto kernel =
      llvm::cast<tcrv::exec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  createTensorExtLiteCapability(builder, loc);
  mlir::ArrayAttr requires = builder.getArrayAttr(
      {symbolRef(builder, getTensorExtLiteFragmentPreferredCapabilitySymbol())});
  createTensorExtLiteVariant(builder, loc, requires);

  auto variant = llvm::cast<tcrv::exec::VariantOp>(
      kernel.getBody().front().back());
  mlir::OpBuilder::InsertionGuard variantGuard(builder);
  builder.setInsertionPointToStart(&variant.getBody().front());
  for (const TensorExtLiteSourceRoleSpec &spec : kRoleSpecs)
    createTensorExtLiteRoleOp(builder, loc, spec, kernelName, requires);

  builder.setInsertionPointAfter(variant);
  createTensorExtLiteLoweringBoundary(builder, loc, kernelName, requires);
  createSelectedTensorExtLiteDiagnostic(builder, loc);
}

class MaterializeTensorExtLiteFragmentMmaSourceFrontDoorPass final
    : public mlir::PassWrapper<
          MaterializeTensorExtLiteFragmentMmaSourceFrontDoorPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const final {
    return "tcrv-tensorext-lite-materialize-fragment-mma-source-front-door";
  }

  llvm::StringRef getDescription() const final {
    return "Materialize one bounded TensorExtLite fragment-MMA source marker "
           "into the selected TensorExtLite role-sequence front door";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<tcrv::exec::TCRVExecDialect,
                    tcrv::tensorext_lite::TCRVTensorExtLiteDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::FailureOr<std::string> kernelName =
        matchTensorExtLiteSourceFrontDoor(module);
    if (mlir::failed(kernelName)) {
      signalPassFailure();
      return;
    }
    if (kernelName->empty())
      return;

    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    materializeTensorExtLiteSourceKernel(builder, module, *kernelName);
    module->removeAttr(kSourceFrontDoorAttrName);
    module->removeAttr(kSourceKernelAttrName);
  }
};

} // namespace

std::unique_ptr<::mlir::Pass>
createMaterializeTensorExtLiteFragmentMmaSourceFrontDoorPass() {
  return std::make_unique<
      MaterializeTensorExtLiteFragmentMmaSourceFrontDoorPass>();
}

} // namespace tianchenrv::plugin::tensorext_lite
