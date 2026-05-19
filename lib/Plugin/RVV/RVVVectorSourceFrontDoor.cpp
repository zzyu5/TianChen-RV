#include "TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <memory>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kSeedAttrName("tcrv_rvv.lowering_seed");
constexpr llvm::StringLiteral kLoweringSeedAttrSuffix(".lowering_seed");

mlir::LogicalResult failMaterializer(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "legacy RVV vector-source front door failed: "
                  << message;
  return mlir::failure();
}

bool hasForeignLoweringSeedAttr(mlir::func::FuncOp func) {
  for (mlir::NamedAttribute attr : func->getAttrs()) {
    llvm::StringRef name = attr.getName().getValue();
    if (name != kSeedAttrName && name.ends_with(kLoweringSeedAttrSuffix))
      return true;
  }
  return false;
}

mlir::LogicalResult requireSourceOnlyModule(mlir::ModuleOp module) {
  mlir::Operation *staleOp = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (staleOp)
      return;
    if (op->getName().getDialectNamespace() == "tcrv" ||
        op->getName().getDialectNamespace() == "tcrv_rvv")
      staleOp = op;
  });
  if (!staleOp)
    return mlir::success();

  return failMaterializer(staleOp,
                          "source materializer requires source-only MLIR "
                          "input; pre-existing tcrv.exec/tcrv_rvv "
                          "selected-boundary or unselected variant residue is "
                          "not accepted");
}

class FailClosedRVVLegacyVectorSourceFrontDoorPass final
    : public mlir::PassWrapper<
          FailClosedRVVLegacyVectorSourceFrontDoorPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-fail-closed-legacy-vector-source-front-door";
  }

  llvm::StringRef getDescription() const final {
    return "Fail closed for legacy RVV vector source-front-door "
           "materialization during RVV Stage 1";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                    tcrv::exec::TCRVExecDialect, tcrv::rvv::TCRVRVVDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<mlir::func::FuncOp, 2> sources;
    module.walk([&](mlir::func::FuncOp func) {
      if (!hasForeignLoweringSeedAttr(func))
        sources.push_back(func);
    });
    if (sources.empty())
      return;
    if (sources.size() != 1) {
      (void)failMaterializer(
          module,
          "source module must contain exactly one RVV source function "
          "candidate");
      signalPassFailure();
      return;
    }

    if (mlir::failed(requireSourceOnlyModule(module))) {
      signalPassFailure();
      return;
    }

    (void)failMaterializer(
        sources.front(),
        "RVV Stage1 source-front-door materialization is disabled; use an "
        "explicit selected generic tcrv_rvv.load/tcrv_rvv.binary/"
        "tcrv_rvv.store body instead");
    signalPassFailure();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass>
createFailClosedRVVLegacyVectorSourceFrontDoorPass() {
  return std::make_unique<FailClosedRVVLegacyVectorSourceFrontDoorPass>();
}

} // namespace tianchenrv::plugin::rvv
