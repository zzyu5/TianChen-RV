#include "Weft/Plugin/Toy/ToySourceFrontDoor.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/Toy/ToyExtensionPlugin.h"
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

namespace weft::plugin::toy {
namespace {

constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "weft_toy.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("weft_toy.source_kernel");
constexpr llvm::StringLiteral kAcceptedSourceFrontDoorValue(
    "template_compute");
constexpr llvm::StringLiteral kDefaultKernelName("toy_source_front_door");
constexpr llvm::StringLiteral kToyCapabilitySymbol("toy_template");
constexpr llvm::StringLiteral kCanonicalProblemSymbol("canonical_problem");

mlir::LogicalResult failMaterializer(mlir::Operation *op,
                                     llvm::StringRef message) {
  op->emitError() << "bounded Toy template source front door failed: "
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

bool hasStaleToyLoweringSeedMetadata(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *op) {
    if (found)
      return;
    found = op->hasAttr("weft_toy.lowering_seed");
  });
  return found;
}

mlir::LogicalResult requireSourceOnlyModule(mlir::ModuleOp module) {
  mlir::Operation *staleOp = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (staleOp || op == module.getOperation())
      return;
    llvm::StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "weft" || dialect == "weft_toy" || dialect == "weft_rvv")
      staleOp = op;
  });
  if (!staleOp)
    return mlir::success();

  return failMaterializer(staleOp,
                          "source materializer requires Toy source-only MLIR "
                          "input; pre-existing weft.exec/weft_toy/weft_rvv "
                          "selected-boundary or variant residue is not "
                          "accepted");
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

mlir::FailureOr<std::string> matchToySourceFrontDoor(mlir::ModuleOp module) {
  auto marker =
      module->getAttrOfType<mlir::StringAttr>(kSourceFrontDoorAttrName);
  if (!marker)
    return std::string();

  if (marker.getValue().trim() != kAcceptedSourceFrontDoorValue) {
    (void)failMaterializer(
        module,
        "weft_toy.source_front_door must be 'template_compute'");
    return mlir::failure();
  }
  if (hasStaleToyLoweringSeedMetadata(module)) {
    (void)failMaterializer(
        module,
        "stale weft_toy.lowering_seed metadata is not accepted as "
        "Toy source-route authority");
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

void createToyCapability(mlir::OpBuilder &builder, mlir::Location loc) {
  mlir::OperationState state(loc, "weft.exec.capability");
  state.addAttribute("sym_name", builder.getStringAttr(kToyCapabilitySymbol));
  state.addAttribute("id", builder.getStringAttr(getToyTemplateCapabilityID()));
  state.addAttribute("kind",
                     builder.getStringAttr(getToyTemplateCapabilityKind()));
  state.addAttribute("status", builder.getStringAttr("available"));
  state.addAttribute("template_abi",
                     builder.getStringAttr(getToyExpectedTemplateABI()));
  state.addAttribute("handoff_kind",
                     builder.getStringAttr(getToyExpectedHandoffKind()));
  (void)builder.create(state);
}

mlir::LogicalResult materializeToySourceKernel(mlir::OpBuilder &builder,
                                               mlir::ModuleOp module,
                                               llvm::StringRef kernelName) {
  mlir::Location loc = module.getLoc();

  createToyCapability(builder, loc);
  mlir::Operation *capability =
      mlir::SymbolTable::lookupSymbolIn(module, kToyCapabilitySymbol);
  llvm::SmallVector<mlir::Operation *, 1> providers{capability};
  std::string targetSymbol = (llvm::Twine(kernelName) + "_target_profile").str();
  llvm::Expected<weft::exec::TargetOp> target =
      weft::target::materializeRISCVExecutionTargetProfile(
          builder, module, loc, targetSymbol,
          (llvm::Twine("weft.riscv.toy-source-profile.") + kernelName).str(),
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
      loc, weft::exec::TemplateComputeProblemOp::getOperationName());
  problemState.addAttribute("sym_name",
                            builder.getStringAttr(kCanonicalProblemSymbol));
  problemState.addAttribute("template_kind",
                            builder.getStringAttr("compute-skeleton"));
  (void)builder.create(problemState);
  return mlir::success();
}

class MaterializeToyTemplateSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeToyTemplateSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const final {
    return "weft-toy-materialize-template-source-front-door";
  }

  llvm::StringRef getDescription() const final {
    return "Adapt one bounded Toy template source into target-bound exact "
           "TemplateCompute canonical P";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<weft::exec::WEFTExecDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::FailureOr<std::string> kernelName =
        matchToySourceFrontDoor(module);
    if (mlir::failed(kernelName)) {
      signalPassFailure();
      return;
    }
    if (kernelName->empty())
      return;

    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    if (mlir::failed(
            materializeToySourceKernel(builder, module, *kernelName))) {
      signalPassFailure();
      return;
    }
    module->removeAttr(kSourceFrontDoorAttrName);
    module->removeAttr(kSourceKernelAttrName);
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeToyTemplateSourceFrontDoorPass() {
  return std::make_unique<MaterializeToyTemplateSourceFrontDoorPass>();
}

} // namespace weft::plugin::toy
