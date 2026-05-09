#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <string>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_LOWERLINALGI32VADDTOEXEC
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::TargetOp;

constexpr llvm::StringLiteral kLinalgGenericOpName("linalg.generic");
constexpr llvm::StringLiteral kArithAddIOpName("arith.addi");
constexpr llvm::StringLiteral kFuncFuncOpName("func.func");
constexpr llvm::StringLiteral kFuncReturnOpName("func.return");
constexpr llvm::StringLiteral kLinalgYieldOpName("linalg.yield");

constexpr llvm::StringLiteral kFrontendLoweringAttrName(
    "tcrv_frontend_lowering");
constexpr llvm::StringLiteral kFrontendLoweringI32VAddValue("i32-vadd");
constexpr llvm::StringLiteral kFrontendKernelAttrName("tcrv_frontend_kernel");
constexpr llvm::StringLiteral kFrontendTargetAttrName("tcrv_frontend_target");

bool isOperationNamed(mlir::Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

mlir::FlatSymbolRefAttr getTargetAttr(mlir::Operation *op) {
  return op ? op->getAttrOfType<mlir::FlatSymbolRefAttr>(
                  kFrontendTargetAttrName)
            : mlir::FlatSymbolRefAttr();
}

bool isMarkedI32VAddLinalg(mlir::Operation *op) {
  auto attr = getStringAttr(op, kFrontendLoweringAttrName);
  return isOperationNamed(op, kLinalgGenericOpName) && attr &&
         attr.getValue() == kFrontendLoweringI32VAddValue;
}

bool isBareSymbolName(llvm::StringRef value) {
  if (value.empty() || value.trim() != value)
    return false;

  auto isFirst = [](char character) {
    unsigned char byte = static_cast<unsigned char>(character);
    return std::isalpha(byte) || character == '_' || character == '$';
  };
  auto isRest = [](char character) {
    unsigned char byte = static_cast<unsigned char>(character);
    return std::isalnum(byte) || character == '_' || character == '$' ||
           character == '.' || character == '-';
  };

  if (!isFirst(value.front()))
    return false;
  return llvm::all_of(value.drop_front(), isRest);
}

bool isI32Scalar(mlir::Type type) {
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  return integer && integer.getWidth() == 32;
}

bool hasOneBlock(mlir::Region &region) {
  return !region.empty() && llvm::hasSingleElement(region);
}

mlir::LogicalResult requireMarkedLinalgBodyShape(mlir::Operation *linalgOp) {
  if (linalgOp->getNumOperands() != 3)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV i32-vadd expects exactly "
              "two input buffers and one output buffer";
  if (linalgOp->getNumRegions() != 1 || !hasOneBlock(linalgOp->getRegion(0)))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV i32-vadd expects one "
              "single-block region";

  mlir::Block &body = linalgOp->getRegion(0).front();
  if (body.getNumArguments() != 3)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV i32-vadd expects three "
              "scalar region arguments";
  if (!llvm::all_of(body.getArguments(), [](mlir::BlockArgument arg) {
        return isI32Scalar(arg.getType());
      }))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV i32-vadd expects i32 "
              "scalar region arguments";

  if (!llvm::hasNItems(body.getOperations(), 2))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV i32-vadd expects exactly "
              "one arith.addi and one linalg.yield";

  mlir::Operation &add = body.front();
  mlir::Operation &yield = body.back();
  if (!isOperationNamed(&add, kArithAddIOpName) ||
      !isOperationNamed(&yield, kLinalgYieldOpName))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV i32-vadd expects one "
              "arith.addi feeding linalg.yield";

  if (add.getNumOperands() != 2 || add.getNumResults() != 1 ||
      add.getOperand(0) != body.getArgument(0) ||
      add.getOperand(1) != body.getArgument(1) ||
      !isI32Scalar(add.getResult(0).getType()))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV i32-vadd expects "
              "arith.addi of the first two i32 region arguments";

  if (yield.getNumOperands() != 1 || yield.getOperand(0) != add.getResult(0))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV i32-vadd expects "
              "linalg.yield to return the arith.addi result";

  return mlir::success();
}

mlir::LogicalResult requireSourceWrapperShape(mlir::Operation *funcOp,
                                              mlir::Operation *linalgOp) {
  if (!isOperationNamed(funcOp, kFuncFuncOpName))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV i32-vadd must be nested "
              "directly in a func.func wrapper";
  if (funcOp->getNumRegions() != 1 || !hasOneBlock(funcOp->getRegion(0)))
    return funcOp->emitError()
           << "TianChen-RV linalg i32-vadd frontend wrapper expects one "
              "single-block func.func body";

  mlir::Block &body = funcOp->getRegion(0).front();
  if (!llvm::hasNItems(body.getOperations(), 2))
    return funcOp->emitError()
           << "TianChen-RV linalg i32-vadd frontend wrapper expects exactly "
              "one linalg.generic and one func.return";
  if (&body.front() != linalgOp || !isOperationNamed(&body.back(),
                                                     kFuncReturnOpName))
    return funcOp->emitError()
           << "TianChen-RV linalg i32-vadd frontend wrapper expects the "
              "marked linalg.generic followed by func.return";
  if (body.back().getNumOperands() != 0)
    return funcOp->emitError()
           << "TianChen-RV linalg i32-vadd frontend wrapper expects "
              "func.return without operands";
  return mlir::success();
}

mlir::StringAttr getKernelName(mlir::Operation *linalgOp,
                               mlir::Operation *funcOp) {
  if (auto attr = getStringAttr(linalgOp, kFrontendKernelAttrName))
    return attr;
  return getStringAttr(funcOp, kFrontendKernelAttrName);
}

mlir::FlatSymbolRefAttr getTargetRef(mlir::Operation *linalgOp,
                                     mlir::Operation *funcOp) {
  if (auto attr = getTargetAttr(linalgOp))
    return attr;
  return getTargetAttr(funcOp);
}

mlir::Operation *findTopLevelSymbol(mlir::ModuleOp module,
                                    llvm::StringRef symbolName) {
  if (!module || module.getBodyRegion().empty())
    return nullptr;

  for (mlir::Operation &op : module.getBody()->getOperations()) {
    auto symbol = op.getAttrOfType<mlir::StringAttr>(
        mlir::SymbolTable::getSymbolAttrName());
    if (symbol && symbol.getValue() == symbolName)
      return &op;
  }
  return nullptr;
}

mlir::LogicalResult requireTopLevelTargetProfile(mlir::ModuleOp module,
                                                 mlir::Operation *sourceOp,
                                                 llvm::StringRef targetName) {
  mlir::Operation *target = findTopLevelSymbol(module, targetName);
  if (!target)
    return sourceOp->emitError()
           << "TianChen-RV linalg frontend target @" << targetName
           << " must resolve to a module-level tcrv.exec.target";
  if (!llvm::isa<TargetOp>(target))
    return sourceOp->emitError()
           << "TianChen-RV linalg frontend target @" << targetName
           << " resolves to a top-level symbol that is not tcrv.exec.target";
  return mlir::success();
}

mlir::LogicalResult requireNoDuplicateKernelSymbol(mlir::ModuleOp module,
                                                   mlir::Operation *sourceOp,
                                                   llvm::StringRef kernelName) {
  if (!findTopLevelSymbol(module, kernelName))
    return mlir::success();
  return sourceOp->emitError()
         << "TianChen-RV linalg frontend kernel symbol @" << kernelName
         << " already exists";
}

KernelOp createExecKernel(mlir::ModuleOp module, mlir::Operation *sourceFunc,
                          llvm::StringRef kernelName,
                          mlir::FlatSymbolRefAttr targetRef) {
  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPoint(sourceFunc);

  mlir::OperationState state(sourceFunc->getLoc(), KernelOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(kernelName));
  state.addAttribute("target", targetRef);
  state.addRegion();

  auto kernel = llvm::cast<KernelOp>(builder.create(state));
  kernel.getBody().push_back(new mlir::Block());
  return kernel;
}

mlir::LogicalResult materializeI32VAddABI(KernelOp kernel) {
  mlir::OpBuilder builder(kernel.getContext());
  builder.setInsertionPointToEnd(&kernel.getBody().front());

  if (llvm::Error error = support::ensureRuntimeABIBufferMemWindows(
          kernel, builder, support::getI32VAddBufferMemWindowSpecs())) {
    kernel.emitError() << llvm::toString(std::move(error));
    return mlir::failure();
  }

  if (llvm::Error error =
          support::ensureRuntimeABIParamsAllowingExistingCNames(
              kernel, builder,
              support::getI32VAddRuntimeElementCountParamSpecs())) {
    kernel.emitError() << llvm::toString(std::move(error));
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult lowerOneMarkedLinalg(mlir::ModuleOp module,
                                         mlir::Operation *linalgOp) {
  mlir::Operation *funcOp = linalgOp->getParentOp();
  if (mlir::failed(requireSourceWrapperShape(funcOp, linalgOp)))
    return mlir::failure();
  if (mlir::failed(requireMarkedLinalgBodyShape(linalgOp)))
    return mlir::failure();

  mlir::StringAttr kernelAttr = getKernelName(linalgOp, funcOp);
  if (!kernelAttr || !isBareSymbolName(kernelAttr.getValue()))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV i32-vadd requires "
              "non-empty bare-symbol string attribute '"
           << kFrontendKernelAttrName << "'";

  mlir::FlatSymbolRefAttr targetRef = getTargetRef(linalgOp, funcOp);
  if (!targetRef || targetRef.getValue().empty())
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV i32-vadd requires "
              "module target symbol attribute '"
           << kFrontendTargetAttrName << "'";

  if (mlir::failed(requireTopLevelTargetProfile(module, linalgOp,
                                                targetRef.getValue())))
    return mlir::failure();
  if (mlir::failed(requireNoDuplicateKernelSymbol(module, linalgOp,
                                                  kernelAttr.getValue())))
    return mlir::failure();

  KernelOp kernel =
      createExecKernel(module, funcOp, kernelAttr.getValue(), targetRef);
  if (mlir::failed(materializeI32VAddABI(kernel))) {
    kernel.erase();
    return mlir::failure();
  }

  funcOp->erase();
  return mlir::success();
}

struct LowerLinalgI32VAddToExecPass
    : impl::LowerLinalgI32VAddToExecBase<LowerLinalgI32VAddToExecPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<mlir::Operation *, 4> markedLinalgOps;
    llvm::SmallPtrSet<mlir::Operation *, 4> sourceWrappers;
    mlir::WalkResult walkResult = module.walk([&](mlir::Operation *op) {
      if (!isMarkedI32VAddLinalg(op))
        return mlir::WalkResult::advance();

      mlir::Operation *wrapper = op->getParentOp();
      if (!sourceWrappers.insert(wrapper).second) {
        op->emitError()
            << "TianChen-RV linalg i32-vadd frontend wrapper may contain only "
               "one marked linalg.generic";
        return mlir::WalkResult::interrupt();
      }
      markedLinalgOps.push_back(op);
      return mlir::WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    for (mlir::Operation *linalgOp : markedLinalgOps) {
      if (mlir::failed(lowerOneMarkedLinalg(module, linalgOp))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerLinalgI32VAddToExecPass() {
  return std::make_unique<LowerLinalgI32VAddToExecPass>();
}

} // namespace tianchenrv::transforms
