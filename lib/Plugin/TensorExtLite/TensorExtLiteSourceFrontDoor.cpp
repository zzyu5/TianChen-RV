#include "Weft/Plugin/TensorExtLite/TensorExtLiteSourceFrontDoor.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"
#include "Weft/Target/RISCVTargetProfile.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

#include <cctype>
#include <cstdint>
#include <memory>
#include <string>

namespace weft::plugin::tensorext_lite {
namespace {

constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "weft_tensorext_lite.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName(
    "weft_tensorext_lite.source_kernel");
constexpr llvm::StringLiteral kAcceptedSourceFrontDoorValue(
    "fragment_mma_template");
constexpr llvm::StringLiteral kDefaultKernelName(
    "tensorext_lite_source_front_door");
constexpr llvm::StringLiteral kCanonicalProblemSymbol("canonical_problem");

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
    found = op->hasAttr("weft_tensorext_lite.lowering_seed");
  });
  return found;
}

mlir::LogicalResult requireSourceOnlyModule(mlir::ModuleOp module) {
  mlir::Operation *staleOp = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (staleOp || op == module.getOperation())
      return;
    llvm::StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "weft" || dialect == "weft_tensorext_lite" ||
        dialect == "weft_rvv" || dialect == "weft_toy")
      staleOp = op;
  });
  if (!staleOp)
    return mlir::success();

  return failMaterializer(
      staleOp,
      "source materializer requires TensorExtLite source-only MLIR input; "
      "pre-existing weft.exec/weft_tensorext_lite/weft_rvv/weft_toy "
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
        "weft_tensorext_lite.source_front_door must be "
        "'fragment_mma_template'");
    return mlir::failure();
  }
  if (hasStaleTensorExtLiteLoweringSeedMetadata(module)) {
    (void)failMaterializer(
        module,
        "stale weft_tensorext_lite.lowering_seed metadata is not accepted "
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
  mlir::OperationState state(loc, "weft.exec.capability");
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

mlir::LogicalResult
materializeTensorExtLiteSourceKernel(mlir::OpBuilder &builder,
                                     mlir::ModuleOp module,
                                     llvm::StringRef kernelName) {
  mlir::Location loc = module.getLoc();

  createTensorExtLiteCapability(builder, loc);
  llvm::StringRef capabilitySymbol =
      getTensorExtLiteFragmentPreferredCapabilitySymbol();
  mlir::Operation *capability =
      mlir::SymbolTable::lookupSymbolIn(module, capabilitySymbol);
  llvm::SmallVector<mlir::Operation *, 1> providers{capability};
  std::string targetSymbol = (llvm::Twine(kernelName) + "_target_profile").str();
  llvm::Expected<weft::exec::TargetOp> target =
      weft::target::materializeRISCVExecutionTargetProfile(
          builder, module, loc, targetSymbol,
          (llvm::Twine("weft.riscv.tensorext-lite-source-profile.") +
           kernelName)
              .str(),
          providers);
  if (!target)
    return failMaterializer(module, llvm::toString(target.takeError()));

  mlir::OperationState kernelState(loc, "weft.exec.kernel");
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addAttribute("target", symbolRef(builder, targetSymbol));
  kernelState.addAttribute("problem",
                           symbolRef(builder, kCanonicalProblemSymbol));
  kernelState.addRegion();
  auto kernel =
      llvm::cast<weft::exec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  mlir::OperationState problemState(
      loc, weft::exec::FragmentMMAProblemOp::getOperationName());
  problemState.addAttribute("sym_name",
                            builder.getStringAttr(kCanonicalProblemSymbol));
  problemState.addAttribute("role_count", builder.getI64IntegerAttr(4));
  (void)builder.create(problemState);
  return mlir::success();
}

class MaterializeTensorExtLiteFragmentMmaSourceFrontDoorPass final
    : public mlir::PassWrapper<
          MaterializeTensorExtLiteFragmentMmaSourceFrontDoorPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const final {
    return "weft-tensorext-lite-materialize-fragment-mma-source-front-door";
  }

  llvm::StringRef getDescription() const final {
    return "Adapt one bounded TensorExtLite fragment-MMA source into "
           "target-bound exact FragmentMMA canonical P";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<weft::exec::WEFTExecDialect>();
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
    if (mlir::failed(materializeTensorExtLiteSourceKernel(
            builder, module, *kernelName))) {
      signalPassFailure();
      return;
    }
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

} // namespace weft::plugin::tensorext_lite
