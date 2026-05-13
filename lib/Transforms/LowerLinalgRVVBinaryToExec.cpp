#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/Exec/IR/CapabilityProviderComposition.h"
#include "TianChenRV/Support/FiniteBinaryFrontendLowering.h"
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
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cctype>
#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_LOWERLINALGRVVBINARYTOEXEC
#define GEN_PASS_DEF_LOWERLINALGI32BINARYTOEXEC
#define GEN_PASS_DEF_LOWERLINALGI32VADDTOEXEC
#define GEN_PASS_DEF_LOWERVECTORRVVI32VADDTOEXEC
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::TargetOp;

constexpr llvm::StringLiteral kLinalgGenericOpName("linalg.generic");
constexpr llvm::StringLiteral kArithAddIOpName("arith.addi");
constexpr llvm::StringLiteral kArithSubIOpName("arith.subi");
constexpr llvm::StringLiteral kArithMulIOpName("arith.muli");
constexpr llvm::StringLiteral kArithConstantOpName("arith.constant");
constexpr llvm::StringLiteral kFuncFuncOpName("func.func");
constexpr llvm::StringLiteral kFuncReturnOpName("func.return");
constexpr llvm::StringLiteral kLinalgYieldOpName("linalg.yield");
constexpr llvm::StringLiteral kVectorTransferReadOpName(
    "vector.transfer_read");
constexpr llvm::StringLiteral kVectorTransferWriteOpName(
    "vector.transfer_write");
constexpr std::int64_t kVectorI32VAddSourceElements = 16;

constexpr llvm::StringLiteral kFrontendLoweringAttrName(
    "tcrv_frontend_lowering");
constexpr llvm::StringLiteral kFrontendKernelAttrName("tcrv_frontend_kernel");
constexpr llvm::StringLiteral kFrontendTargetAttrName("tcrv_frontend_target");
constexpr llvm::StringLiteral kFrontendCapabilityProvidersAttrName(
    "tcrv_frontend_capability_providers");
constexpr llvm::StringLiteral kLegacyRVVLoweringDescriptorAttrName(
    "tcrv_rvv.lowering_descriptor");
constexpr llvm::StringLiteral kLegacyScalarLoweringDescriptorAttrName(
    "tcrv_scalar.lowering_descriptor");
constexpr llvm::StringLiteral kLegacyRVVSelectedLoweringDescriptorAttrName(
    "tcrv_rvv.selected_lowering_descriptor");
constexpr llvm::StringLiteral kLegacyScalarSelectedLoweringDescriptorAttrName(
    "tcrv_scalar.selected_lowering_descriptor");
constexpr llvm::StringLiteral kLegacySelectedLoweringDescriptorAttrName(
    "selected_lowering_descriptor");

enum class SourceBinaryArithmeticKind {
  Add,
  Sub,
  Mul,
};

struct InferredFrontendBinarySource {
  const support::FiniteBinaryFrontendContract *contract = nullptr;
  support::FiniteBinaryElementKind elementKind =
      support::FiniteBinaryElementKind::I32;
  SourceBinaryArithmeticKind arithmetic = SourceBinaryArithmeticKind::Add;
  unsigned elementBitWidth = 32;
  llvm::StringRef arithmeticOpName;
};

struct FrontendBinarySpec {
  const support::FiniteBinaryFrontendContract *contract = nullptr;
  llvm::SmallVector<support::RuntimeABIMemWindowSpec, 3> bufferMemWindowSpecs;
  llvm::SmallVector<support::RuntimeABIParamSpec, 1> runtimeElementCountSpecs;
};

llvm::StringRef getSourceArithmeticOpName(SourceBinaryArithmeticKind kind) {
  switch (kind) {
  case SourceBinaryArithmeticKind::Add:
    return kArithAddIOpName;
  case SourceBinaryArithmeticKind::Sub:
    return kArithSubIOpName;
  case SourceBinaryArithmeticKind::Mul:
    return kArithMulIOpName;
  }
  return llvm::StringRef();
}

llvm::StringRef getSourceArithmeticFrontendSuffix(
    SourceBinaryArithmeticKind kind) {
  switch (kind) {
  case SourceBinaryArithmeticKind::Add:
    return "vadd";
  case SourceBinaryArithmeticKind::Sub:
    return "vsub";
  case SourceBinaryArithmeticKind::Mul:
    return "vmul";
  }
  return llvm::StringRef();
}

llvm::StringRef getElementKindDTypeID(support::FiniteBinaryElementKind kind) {
  switch (kind) {
  case support::FiniteBinaryElementKind::I32:
    return "i32";
  case support::FiniteBinaryElementKind::I64:
    return "i64";
  }
  return llvm::StringRef();
}

std::string getInferredFamilyID(support::FiniteBinaryElementKind elementKind,
                                SourceBinaryArithmeticKind arithmetic) {
  return (llvm::Twine(getElementKindDTypeID(elementKind)) + "-" +
          getSourceArithmeticFrontendSuffix(arithmetic))
      .str();
}

FrontendBinarySpec makeFrontendBinarySpec(
    const support::FiniteBinaryFrontendContract &contract) {
  FrontendBinarySpec spec;
  spec.contract = &contract;
  spec.bufferMemWindowSpecs =
      support::getFiniteBinaryFrontendBufferMemWindowSpecs(contract);
  spec.runtimeElementCountSpecs =
      support::getFiniteBinaryFrontendRuntimeElementCountParamSpecs(contract);
  return spec;
}

std::string formatSupportedFrontendLowerings() {
  return support::formatFiniteBinaryFrontendLoweringMarkers();
}

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

bool isMarkedFrontendBinaryLinalg(mlir::Operation *op) {
  auto attr = getStringAttr(op, kFrontendLoweringAttrName);
  return isOperationNamed(op, kLinalgGenericOpName) && attr;
}

bool isMarkedFrontendBinaryVectorFunc(mlir::Operation *op) {
  auto attr = getStringAttr(op, kFrontendLoweringAttrName);
  return isOperationNamed(op, kFuncFuncOpName) && attr;
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

bool isIntegerScalarWithWidth(mlir::Type type, unsigned width) {
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  return integer && integer.getWidth() == width;
}

bool isRankedVectorWithIntegerElementWidth(mlir::Type type,
                                           std::int64_t elements,
                                           unsigned width) {
  auto vector = llvm::dyn_cast<mlir::VectorType>(type);
  if (!vector || vector.getRank() != 1 ||
      vector.getDimSize(0) != elements)
    return false;
  return isIntegerScalarWithWidth(vector.getElementType(), width);
}

std::optional<unsigned>
getRankedDynamicVectorMemRefIntegerElementWidth(mlir::Type type) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  if (!memref || memref.getRank() != 1 ||
      memref.getDimSize(0) != mlir::ShapedType::kDynamic)
    return std::nullopt;
  auto integer = llvm::dyn_cast<mlir::IntegerType>(memref.getElementType());
  if (!integer)
    return std::nullopt;
  return integer.getWidth();
}

std::optional<unsigned> getIntegerScalarWidth(mlir::Type type) {
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  if (!integer)
    return std::nullopt;
  return integer.getWidth();
}

std::optional<support::FiniteBinaryElementKind>
getSupportedElementKindForWidth(unsigned width) {
  if (width == 32)
    return support::FiniteBinaryElementKind::I32;
  if (width == 64)
    return support::FiniteBinaryElementKind::I64;
  return std::nullopt;
}

std::optional<SourceBinaryArithmeticKind>
getSupportedArithmeticKind(mlir::Operation *op) {
  if (isOperationNamed(op, kArithAddIOpName))
    return SourceBinaryArithmeticKind::Add;
  if (isOperationNamed(op, kArithSubIOpName))
    return SourceBinaryArithmeticKind::Sub;
  if (isOperationNamed(op, kArithMulIOpName))
    return SourceBinaryArithmeticKind::Mul;
  return std::nullopt;
}

std::string getIntegerTypeSpelling(unsigned width) {
  return (llvm::Twine("i") + llvm::Twine(width)).str();
}

bool hasOneBlock(mlir::Region &region) {
  return !region.empty() && llvm::hasSingleElement(region);
}

mlir::LogicalResult
inferMarkedLinalgSourceIdentity(mlir::Operation *linalgOp,
                                InferredFrontendBinarySource &out) {
  if (linalgOp->getNumOperands() != 3)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary expects "
              "exactly two input buffers and one output buffer";

  std::optional<unsigned> lhsWidth =
      getRankedDynamicVectorMemRefIntegerElementWidth(
          linalgOp->getOperand(0).getType());
  std::optional<unsigned> rhsWidth =
      getRankedDynamicVectorMemRefIntegerElementWidth(
          linalgOp->getOperand(1).getType());
  std::optional<unsigned> outWidth =
      getRankedDynamicVectorMemRefIntegerElementWidth(
          linalgOp->getOperand(2).getType());
  if (!lhsWidth || !rhsWidth || !outWidth)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary expects "
              "rank-1 dynamic memref<?xi32> or memref<?xi64> "
              "lhs/rhs/output operands";
  if (*lhsWidth != *rhsWidth || *lhsWidth != *outWidth)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary expects "
              "lhs/rhs/output memref element types to match, got "
           << getIntegerTypeSpelling(*lhsWidth) << ", "
           << getIntegerTypeSpelling(*rhsWidth) << ", and "
           << getIntegerTypeSpelling(*outWidth);

  std::optional<support::FiniteBinaryElementKind> elementKind =
      getSupportedElementKindForWidth(*lhsWidth);
  if (!elementKind)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary supports "
              "only i32 or i64 source memref element types, got "
           << getIntegerTypeSpelling(*lhsWidth);

  if (linalgOp->getNumRegions() != 1 || !hasOneBlock(linalgOp->getRegion(0)))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary"
           << " expects one single-block region";

  mlir::Block &body = linalgOp->getRegion(0).front();
  if (body.getNumArguments() != 3)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary"
           << " expects three scalar region arguments";

  for (mlir::BlockArgument arg : body.getArguments()) {
    std::optional<unsigned> argWidth = getIntegerScalarWidth(arg.getType());
    if (!argWidth || *argWidth != *lhsWidth)
      return linalgOp->emitError()
             << "marked linalg.generic for TianChen-RV bounded binary expects "
             << getIntegerTypeSpelling(*lhsWidth)
             << " scalar region arguments matching the source memrefs";
  }

  if (!llvm::hasNItems(body.getOperations(), 2))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary expects "
              "exactly one supported arith.addi, arith.subi, or arith.muli "
              "and one linalg.yield";

  mlir::Operation &arithmetic = body.front();
  mlir::Operation &yield = body.back();
  std::optional<SourceBinaryArithmeticKind> arithmeticKind =
      getSupportedArithmeticKind(&arithmetic);
  if (!arithmeticKind)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary expects "
              "source body arithmetic to be arith.addi, arith.subi, or "
              "arith.muli";
  llvm::StringRef arithmeticOpName =
      getSourceArithmeticOpName(*arithmeticKind);

  if (!isOperationNamed(&yield, kLinalgYieldOpName))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary expects "
           << arithmeticOpName << " feeding linalg.yield";

  if (arithmetic.getNumOperands() != 2 ||
      arithmetic.getNumResults() != 1 ||
      arithmetic.getOperand(0) != body.getArgument(0) ||
      arithmetic.getOperand(1) != body.getArgument(1) ||
      !isIntegerScalarWithWidth(arithmetic.getResult(0).getType(), *lhsWidth))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary expects "
           << arithmeticOpName
           << " of the first two "
           << getIntegerTypeSpelling(*lhsWidth)
           << " region arguments";

  if (yield.getNumOperands() != 1 ||
      yield.getOperand(0) != arithmetic.getResult(0))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary"
           << " expects linalg.yield to return the "
           << arithmeticOpName << " result";

  std::string inferredFamilyID =
      getInferredFamilyID(*elementKind, *arithmeticKind);
  const support::FiniteBinaryFrontendContract *contract =
      support::lookupFiniteBinaryFrontendContractByFamilyID(inferredFamilyID);
  if (!contract)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary inferred "
              "unsupported finite source family '"
           << inferredFamilyID << "'";

  out.contract = contract;
  out.elementKind = *elementKind;
  out.arithmetic = *arithmeticKind;
  out.elementBitWidth = *lhsWidth;
  out.arithmeticOpName = arithmeticOpName;
  return mlir::success();
}

mlir::LogicalResult requireSourceWrapperShape(mlir::Operation *funcOp,
                                              mlir::Operation *linalgOp) {
  if (!isOperationNamed(funcOp, kFuncFuncOpName))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary must be "
              "nested directly in a func.func wrapper";
  if (funcOp->getNumRegions() != 1 || !hasOneBlock(funcOp->getRegion(0)))
    return funcOp->emitError()
           << "TianChen-RV linalg bounded binary frontend wrapper expects one "
              "single-block func.func body";

  mlir::Block &body = funcOp->getRegion(0).front();
  if (!llvm::hasNItems(body.getOperations(), 2))
    return funcOp->emitError()
           << "TianChen-RV linalg bounded binary frontend wrapper expects "
              "exactly one linalg.generic and one func.return";
  if (&body.front() != linalgOp || !isOperationNamed(&body.back(),
                                                     kFuncReturnOpName))
    return funcOp->emitError()
           << "TianChen-RV linalg bounded binary frontend wrapper expects the "
              "marked linalg.generic followed by func.return";
  if (body.back().getNumOperands() != 0)
    return funcOp->emitError()
           << "TianChen-RV linalg bounded binary frontend wrapper expects "
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

mlir::Attribute getCapabilityProviderRefsAttr(mlir::Operation *linalgOp,
                                              mlir::Operation *funcOp) {
  if (mlir::Attribute attr =
          linalgOp->getAttr(kFrontendCapabilityProvidersAttrName))
    return attr;
  return funcOp ? funcOp->getAttr(kFrontendCapabilityProvidersAttrName)
                : mlir::Attribute();
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

mlir::LogicalResult requireTopLevelTargetProfile(
    mlir::ModuleOp module, mlir::Operation *sourceOp,
    llvm::StringRef targetName, TargetOp &out) {
  mlir::Operation *target = findTopLevelSymbol(module, targetName);
  if (!target)
    return sourceOp->emitError()
           << "TianChen-RV linalg frontend target @" << targetName
           << " must resolve to a module-level tcrv.exec.target";
  if (!llvm::isa<TargetOp>(target))
    return sourceOp->emitError()
           << "TianChen-RV linalg frontend target @" << targetName
           << " resolves to a top-level symbol that is not tcrv.exec.target";
  out = llvm::cast<TargetOp>(target);
  if (!tcrv::exec::isCapabilityProviderTarget(out))
    return sourceOp->emitError()
           << "TianChen-RV linalg frontend target @" << targetName
           << " must be a capability-provider tcrv.exec.target with "
              "non-empty id/kind";

  llvm::Expected<llvm::SmallVector<mlir::Operation *, 8>> providers =
      tcrv::exec::collectComposedModuleCapabilityProviders(out);
  if (!providers) {
    std::string message = llvm::toString(providers.takeError());
    return sourceOp->emitError() << message;
  }
  return mlir::success();
}

mlir::LogicalResult recordProviderIdentityForImportValidation(
    mlir::Operation *sourceOp, llvm::StringRef context,
    mlir::Operation *provider, llvm::StringSet<> &seenSymbols,
    llvm::StringSet<> &seenIDs) {
  llvm::StringRef symbol = tcrv::exec::getCapabilityProviderSymbolName(provider);
  if (symbol.empty())
    return sourceOp->emitError()
           << "TianChen-RV linalg frontend " << context
           << " provider must carry a symbol name";

  if (!seenSymbols.insert(symbol).second)
    return sourceOp->emitError()
           << "TianChen-RV linalg frontend " << context << " provider @"
           << symbol
           << " duplicates the selected target profile, target-composed "
              "provider, or another supplemental import";

  llvm::StringRef id = tcrv::exec::getCapabilityProviderID(provider);
  if (id.trim().empty())
    return sourceOp->emitError()
           << "TianChen-RV linalg frontend " << context << " provider @"
           << symbol << " must carry a non-empty capability id";

  if (!seenIDs.insert(id).second)
    return sourceOp->emitError()
           << "TianChen-RV linalg frontend " << context << " provider @"
           << symbol << " duplicates capability id '" << id
           << "' with the selected target profile, target-composed provider, "
              "or another supplemental import";

  return mlir::success();
}

mlir::LogicalResult seedSelectedTargetProviderScope(
    mlir::Operation *sourceOp, TargetOp selectedTarget,
    llvm::StringSet<> &seenSymbols, llvm::StringSet<> &seenIDs) {
  if (mlir::failed(recordProviderIdentityForImportValidation(
          sourceOp, "selected target", selectedTarget.getOperation(),
          seenSymbols, seenIDs)))
    return mlir::failure();

  llvm::Expected<llvm::SmallVector<mlir::Operation *, 8>> providers =
      tcrv::exec::collectComposedModuleCapabilityProviders(selectedTarget);
  if (!providers) {
    std::string message = llvm::toString(providers.takeError());
    return sourceOp->emitError() << message;
  }

  for (mlir::Operation *provider : *providers)
    if (mlir::failed(recordProviderIdentityForImportValidation(
            sourceOp, "target-composed", provider, seenSymbols, seenIDs)))
      return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult collectCapabilityProviderImports(
    mlir::ModuleOp module, mlir::Operation *sourceOp,
    TargetOp selectedTarget, mlir::ArrayAttr providerRefs,
    llvm::SmallVectorImpl<mlir::Operation *> &imports) {
  llvm::StringSet<> seenSymbols;
  llvm::StringSet<> seenIDs;

  if (mlir::failed(seedSelectedTargetProviderScope(sourceOp, selectedTarget,
                                                   seenSymbols, seenIDs)))
    return mlir::failure();

  for (mlir::Attribute attr : providerRefs) {
    auto ref = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!ref || ref.getValue().empty())
      return sourceOp->emitError()
             << "TianChen-RV linalg frontend capability-provider imports "
                "expect an array of non-empty module symbol references in '"
             << kFrontendCapabilityProvidersAttrName << "'";

    llvm::StringRef symbolName = ref.getValue();
    if (!seenSymbols.insert(symbolName).second)
      return sourceOp->emitError()
             << "TianChen-RV linalg frontend supplemental "
                "capability-provider import @"
             << symbolName
             << " duplicates the selected target profile, target-composed "
                "provider, or another supplemental import";

    mlir::Operation *provider = findTopLevelSymbol(module, symbolName);
    if (!provider)
      return sourceOp->emitError()
             << "TianChen-RV linalg frontend supplemental "
                "capability-provider import @"
             << symbolName << " must resolve to a module-level symbol";

    if (!tcrv::exec::isCapabilityProviderOperation(provider))
      return sourceOp->emitError()
             << "TianChen-RV linalg frontend supplemental "
                "capability-provider import @"
             << symbolName
             << " must resolve to a module-level tcrv.exec.capability or "
                "capability-provider tcrv.exec.target";

    llvm::StringRef id = tcrv::exec::getCapabilityProviderID(provider);
    if (id.empty())
      return sourceOp->emitError()
             << "TianChen-RV linalg frontend supplemental "
                "capability-provider import @"
             << symbolName << " must carry a non-empty capability id";
    if (!seenIDs.insert(id).second)
      return sourceOp->emitError()
             << "TianChen-RV linalg frontend supplemental "
                "capability-provider import @"
             << symbolName << " duplicates capability id '" << id
             << "' with the selected target profile, target-composed provider, "
                "or another supplemental import";

    if (auto target = llvm::dyn_cast<TargetOp>(provider)) {
      llvm::Expected<llvm::SmallVector<mlir::Operation *, 8>> providers =
          tcrv::exec::collectComposedModuleCapabilityProviders(target);
      if (!providers) {
        std::string message = llvm::toString(providers.takeError());
        return sourceOp->emitError() << message;
      }

      for (mlir::Operation *composedProvider : *providers)
        if (mlir::failed(recordProviderIdentityForImportValidation(
                sourceOp, "supplemental target-composed", composedProvider,
                seenSymbols, seenIDs)))
          return mlir::failure();
    }

    imports.push_back(provider);
  }

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

mlir::LogicalResult requireNoLegacyDescriptorMetadata(
    mlir::Operation *funcOp, mlir::Operation *linalgOp,
    llvm::StringRef frontendName = "linalg frontend",
    llvm::StringRef sourceAuthority =
        "source linalg body and typed operands") {
  llvm::StringRef legacyNames[] = {
      kLegacyRVVLoweringDescriptorAttrName,
      kLegacyScalarLoweringDescriptorAttrName,
      kLegacyRVVSelectedLoweringDescriptorAttrName,
      kLegacyScalarSelectedLoweringDescriptorAttrName,
      kLegacySelectedLoweringDescriptorAttrName,
  };

  for (mlir::Operation *op : {funcOp, linalgOp}) {
    if (!op)
      continue;
    for (llvm::StringRef attrName : legacyNames) {
      if (op->getAttr(attrName))
        return op->emitError()
               << "TianChen-RV " << frontendName
               << " no longer accepts legacy "
                  "descriptor metadata '"
               << attrName
               << "'; the " << sourceAuthority << " are the "
                  "compute authority";
    }
  }

  return mlir::success();
}

bool isIntegerConstantZero(mlir::Operation *op, mlir::Type expectedType) {
  if (!isOperationNamed(op, kArithConstantOpName) || op->getNumOperands() != 0 ||
      op->getNumResults() != 1 || op->getResult(0).getType() != expectedType)
    return false;

  auto value = op->getAttrOfType<mlir::IntegerAttr>("value");
  return value && value.getValue().isZero();
}

mlir::LogicalResult requireVectorTransferRead(
    mlir::Operation *op, mlir::Value sourceBuffer, mlir::Value index,
    mlir::Value padding, llvm::StringRef role) {
  if (!isOperationNamed(op, kVectorTransferReadOpName) ||
      op->getNumOperands() != 3 || op->getNumResults() != 1)
    return op->emitError()
           << "TianChen-RV vector i32-vadd frontend expects " << role
           << " to be vector.transfer_read";

  if (op->getOperand(0) != sourceBuffer || op->getOperand(1) != index ||
      op->getOperand(2) != padding)
    return op->emitError()
           << "TianChen-RV vector i32-vadd frontend expects " << role
           << " to read the matching input buffer at the shared zero index "
              "with i32 zero padding";

  if (!isRankedVectorWithIntegerElementWidth(
          op->getResult(0).getType(), kVectorI32VAddSourceElements, 32))
    return op->emitError()
           << "TianChen-RV vector i32-vadd frontend expects " << role
           << " to produce vector<16xi32>";

  return mlir::success();
}

mlir::LogicalResult requireVectorI32VAddSourceWrapper(mlir::Operation *funcOp) {
  if (!isOperationNamed(funcOp, kFuncFuncOpName))
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend expects a func.func "
              "wrapper";
  if (funcOp->getNumRegions() != 1 || !hasOneBlock(funcOp->getRegion(0)))
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend wrapper expects one "
              "single-block func.func body";

  mlir::Block &body = funcOp->getRegion(0).front();
  if (body.getNumArguments() != 3)
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend wrapper expects three "
              "memref<?xi32> block arguments";

  for (mlir::BlockArgument arg : body.getArguments()) {
    std::optional<unsigned> width =
        getRankedDynamicVectorMemRefIntegerElementWidth(arg.getType());
    if (!width || *width != 32)
      return funcOp->emitError()
             << "TianChen-RV vector i32-vadd frontend wrapper expects "
                "lhs/rhs/output arguments to be rank-1 dynamic memref<?xi32>";
  }

  if (!llvm::hasNItems(body.getOperations(), 7))
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend wrapper expects exactly "
              "zero index, zero i32 padding, two vector.transfer_read ops, "
              "one arith.addi, one vector.transfer_write, and func.return";

  auto it = body.begin();
  mlir::Operation *indexConstant = &*it++;
  mlir::Operation *paddingConstant = &*it++;
  mlir::Operation *lhsRead = &*it++;
  mlir::Operation *rhsRead = &*it++;
  mlir::Operation *add = &*it++;
  mlir::Operation *write = &*it++;
  mlir::Operation *ret = &*it++;

  if (!isIntegerConstantZero(indexConstant, mlir::IndexType::get(
                                                funcOp->getContext())))
    return indexConstant->emitError()
           << "TianChen-RV vector i32-vadd frontend expects the first source "
              "operation to be arith.constant 0 : index";
  if (!isIntegerConstantZero(paddingConstant,
                             mlir::IntegerType::get(funcOp->getContext(), 32)))
    return paddingConstant->emitError()
           << "TianChen-RV vector i32-vadd frontend expects the second source "
              "operation to be arith.constant 0 : i32";

  mlir::Value index = indexConstant->getResult(0);
  mlir::Value padding = paddingConstant->getResult(0);
  if (mlir::failed(requireVectorTransferRead(
          lhsRead, body.getArgument(0), index, padding, "lhs read")))
    return mlir::failure();
  if (mlir::failed(requireVectorTransferRead(
          rhsRead, body.getArgument(1), index, padding, "rhs read")))
    return mlir::failure();

  if (!isOperationNamed(add, kArithAddIOpName) || add->getNumOperands() != 2 ||
      add->getNumResults() != 1)
    return add->emitError()
           << "TianChen-RV vector i32-vadd frontend expects arith.addi over "
              "the two transfer-read vector values";
  if (add->getOperand(0) != lhsRead->getResult(0) ||
      add->getOperand(1) != rhsRead->getResult(0) ||
      !isRankedVectorWithIntegerElementWidth(
          add->getResult(0).getType(), kVectorI32VAddSourceElements, 32))
    return add->emitError()
           << "TianChen-RV vector i32-vadd frontend expects arith.addi to "
              "consume lhs/rhs vector<16xi32> reads and produce vector<16xi32>";

  if (!isOperationNamed(write, kVectorTransferWriteOpName) ||
      write->getNumOperands() != 3 || write->getNumResults() != 0)
    return write->emitError()
           << "TianChen-RV vector i32-vadd frontend expects "
              "vector.transfer_write of the add result";
  if (write->getOperand(0) != add->getResult(0) ||
      write->getOperand(1) != body.getArgument(2) ||
      write->getOperand(2) != index)
    return write->emitError()
           << "TianChen-RV vector i32-vadd frontend expects "
              "vector.transfer_write to store the add result to the output "
              "buffer at the shared zero index";

  if (!isOperationNamed(ret, kFuncReturnOpName) || ret->getNumOperands() != 0)
    return ret->emitError()
           << "TianChen-RV vector i32-vadd frontend expects func.return "
              "without operands";

  return mlir::success();
}

mlir::LogicalResult crossCheckVectorI32VAddMarker(mlir::Operation *funcOp,
                                                  llvm::StringRef marker) {
  const support::FiniteBinaryFrontendContract *markerContract =
      support::lookupFiniteBinaryFrontendContractByMarker(marker);
  if (!markerContract)
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend expects '"
           << kFrontendLoweringAttrName << "' to be 'i32-vadd'";

  const support::FiniteBinaryFrontendContract &expected =
      support::getI32VAddFiniteBinaryFrontendContract();
  if (markerContract->familyID != expected.familyID)
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend supports only marker "
              "'i32-vadd'; marker '"
           << marker
           << "' is not accepted because this pass is not a generic vector "
              "backend";

  return mlir::success();
}

mlir::LogicalResult crossCheckFrontendMarker(
    mlir::Operation *linalgOp, llvm::StringRef marker,
    const InferredFrontendBinarySource &source) {
  const support::FiniteBinaryFrontendContract *markerContract =
      support::lookupFiniteBinaryFrontendContractByMarker(marker);
  if (!markerContract)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV expects '"
           << kFrontendLoweringAttrName
           << "' to be " << formatSupportedFrontendLowerings();

  if (markerContract->dtypeID != source.contract->dtypeID)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV has marker '"
           << marker << "' requesting dtype '" << markerContract->dtypeID
           << "' but source operands and region arguments infer dtype '"
           << source.contract->dtypeID
           << "'; marker is only a bounded route request/cross-check";

  if (markerContract->familyID != source.contract->familyID)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV has marker '"
           << marker << "' requesting family '" << markerContract->familyID
           << "' but source body infers family '" << source.contract->familyID
           << "' from " << source.arithmeticOpName
           << "; marker is only a bounded route request/cross-check";

  return mlir::success();
}

KernelOp createExecKernel(mlir::ModuleOp module, mlir::Operation *sourceFunc,
                          llvm::StringRef kernelName,
                          mlir::FlatSymbolRefAttr targetRef,
                          const FrontendBinarySpec &spec) {
  const support::FiniteBinaryFrontendContract &contract = *spec.contract;
  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPoint(sourceFunc);

  mlir::OperationState state(sourceFunc->getLoc(), KernelOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(kernelName));
  state.addAttribute("target", targetRef);
  state.addAttribute(kFrontendLoweringAttrName,
                     builder.getStringAttr(contract.frontendLowering));
  state.addRegion();

  auto kernel = llvm::cast<KernelOp>(builder.create(state));
  kernel.getBody().push_back(new mlir::Block());
  return kernel;
}

void materializeCapabilityProviderImports(
    KernelOp kernel, llvm::ArrayRef<mlir::Operation *> imports) {
  mlir::OpBuilder builder(kernel.getContext());
  builder.setInsertionPointToEnd(&kernel.getBody().front());
  for (mlir::Operation *provider : imports)
    builder.clone(*provider);
}

mlir::LogicalResult materializeFrontendBinaryABI(
    KernelOp kernel, const FrontendBinarySpec &spec) {
  mlir::OpBuilder builder(kernel.getContext());
  builder.setInsertionPointToEnd(&kernel.getBody().front());

  if (llvm::Error error = support::ensureRuntimeABIBufferMemWindows(
          kernel, builder, spec.bufferMemWindowSpecs)) {
    kernel.emitError() << llvm::toString(std::move(error));
    return mlir::failure();
  }

  if (llvm::Error error =
          support::ensureRuntimeABIParamsAllowingExistingCNames(
              kernel, builder, spec.runtimeElementCountSpecs)) {
    kernel.emitError() << llvm::toString(std::move(error));
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult lowerOneMarkedLinalg(mlir::ModuleOp module,
                                         mlir::Operation *linalgOp) {
  auto frontendAttr = getStringAttr(linalgOp, kFrontendLoweringAttrName);
  if (!frontendAttr)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV expects '"
           << kFrontendLoweringAttrName
           << "' to be " << formatSupportedFrontendLowerings();

  mlir::Operation *funcOp = linalgOp->getParentOp();
  if (mlir::failed(requireSourceWrapperShape(funcOp, linalgOp)))
    return mlir::failure();

  InferredFrontendBinarySource source;
  if (mlir::failed(inferMarkedLinalgSourceIdentity(linalgOp, source)))
    return mlir::failure();
  if (mlir::failed(crossCheckFrontendMarker(linalgOp, frontendAttr.getValue(),
                                            source)))
    return mlir::failure();
  if (mlir::failed(requireNoLegacyDescriptorMetadata(funcOp, linalgOp)))
    return mlir::failure();

  FrontendBinarySpec spec = makeFrontendBinarySpec(*source.contract);

  mlir::StringAttr kernelAttr = getKernelName(linalgOp, funcOp);
  if (!kernelAttr || !isBareSymbolName(kernelAttr.getValue()))
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary requires "
              "non-empty bare-symbol string attribute '"
           << kFrontendKernelAttrName << "'";

  mlir::FlatSymbolRefAttr targetRef = getTargetRef(linalgOp, funcOp);
  if (!targetRef || targetRef.getValue().empty())
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary requires "
              "module target symbol attribute '"
           << kFrontendTargetAttrName << "'";

  TargetOp selectedTarget;
  if (mlir::failed(requireTopLevelTargetProfile(module, linalgOp,
                                                targetRef.getValue(),
                                                selectedTarget)))
    return mlir::failure();
  if (mlir::failed(requireNoDuplicateKernelSymbol(module, linalgOp,
                                                  kernelAttr.getValue())))
    return mlir::failure();

  llvm::SmallVector<mlir::Operation *, 4> providerImports;
  if (mlir::Attribute rawProviderRefs =
          getCapabilityProviderRefsAttr(linalgOp, funcOp)) {
    auto providerRefs = llvm::dyn_cast<mlir::ArrayAttr>(rawProviderRefs);
    if (!providerRefs)
      return linalgOp->emitError()
             << "marked linalg.generic for TianChen-RV bounded binary expects '"
             << kFrontendCapabilityProvidersAttrName
             << "' to be an array of module symbol references";

    if (mlir::failed(collectCapabilityProviderImports(
            module, linalgOp, selectedTarget, providerRefs, providerImports)))
      return mlir::failure();
  }

  KernelOp kernel =
      createExecKernel(module, funcOp, kernelAttr.getValue(), targetRef, spec);
  materializeCapabilityProviderImports(kernel, providerImports);
  if (mlir::failed(materializeFrontendBinaryABI(kernel, spec))) {
    kernel.erase();
    return mlir::failure();
  }

  funcOp->erase();
  return mlir::success();
}

mlir::LogicalResult lowerOneMarkedVectorFunc(mlir::ModuleOp module,
                                             mlir::Operation *funcOp) {
  auto frontendAttr = getStringAttr(funcOp, kFrontendLoweringAttrName);
  if (!frontendAttr)
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend expects '"
           << kFrontendLoweringAttrName << "' to be 'i32-vadd'";

  if (mlir::failed(
          crossCheckVectorI32VAddMarker(funcOp, frontendAttr.getValue())))
    return mlir::failure();
  if (mlir::failed(requireNoLegacyDescriptorMetadata(
          funcOp, nullptr, "vector frontend",
          "source vector/arith body and typed operands")))
    return mlir::failure();
  if (mlir::failed(requireVectorI32VAddSourceWrapper(funcOp)))
    return mlir::failure();

  const support::FiniteBinaryFrontendContract &contract =
      support::getI32VAddFiniteBinaryFrontendContract();
  FrontendBinarySpec spec = makeFrontendBinarySpec(contract);

  mlir::StringAttr kernelAttr = getKernelName(funcOp, funcOp);
  if (!kernelAttr || !isBareSymbolName(kernelAttr.getValue()))
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend requires non-empty "
              "bare-symbol string attribute '"
           << kFrontendKernelAttrName << "'";

  mlir::FlatSymbolRefAttr targetRef = getTargetRef(funcOp, funcOp);
  if (!targetRef || targetRef.getValue().empty())
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend requires module target "
              "symbol attribute '"
           << kFrontendTargetAttrName << "'";

  TargetOp selectedTarget;
  if (mlir::failed(requireTopLevelTargetProfile(
          module, funcOp, targetRef.getValue(), selectedTarget)))
    return mlir::failure();
  if (mlir::failed(requireNoDuplicateKernelSymbol(module, funcOp,
                                                  kernelAttr.getValue())))
    return mlir::failure();

  llvm::SmallVector<mlir::Operation *, 4> providerImports;
  if (mlir::Attribute rawProviderRefs =
          getCapabilityProviderRefsAttr(funcOp, funcOp)) {
    auto providerRefs = llvm::dyn_cast<mlir::ArrayAttr>(rawProviderRefs);
    if (!providerRefs)
      return funcOp->emitError()
             << "TianChen-RV vector i32-vadd frontend expects '"
             << kFrontendCapabilityProvidersAttrName
             << "' to be an array of module symbol references";

    if (mlir::failed(collectCapabilityProviderImports(
            module, funcOp, selectedTarget, providerRefs, providerImports)))
      return mlir::failure();
  }

  KernelOp kernel =
      createExecKernel(module, funcOp, kernelAttr.getValue(), targetRef, spec);
  materializeCapabilityProviderImports(kernel, providerImports);
  if (mlir::failed(materializeFrontendBinaryABI(kernel, spec))) {
    kernel.erase();
    return mlir::failure();
  }

  funcOp->erase();
  return mlir::success();
}

mlir::LogicalResult lowerMarkedFrontendBinaryLinalgInModule(
    mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *, 4> markedLinalgOps;
  llvm::SmallPtrSet<mlir::Operation *, 4> sourceWrappers;
  mlir::WalkResult walkResult = module.walk([&](mlir::Operation *op) {
    if (!isMarkedFrontendBinaryLinalg(op))
      return mlir::WalkResult::advance();

    mlir::Operation *wrapper = op->getParentOp();
    if (!sourceWrappers.insert(wrapper).second) {
      op->emitError()
          << "TianChen-RV linalg bounded binary frontend wrapper may contain "
             "only one marked linalg.generic";
      return mlir::WalkResult::interrupt();
    }
    markedLinalgOps.push_back(op);
    return mlir::WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return mlir::failure();

  for (mlir::Operation *linalgOp : markedLinalgOps)
    if (mlir::failed(lowerOneMarkedLinalg(module, linalgOp)))
      return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult
lowerMarkedFrontendVectorI32VAddInModule(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *, 4> markedVectorFuncs;
  mlir::WalkResult walkResult = module.walk([&](mlir::Operation *op) {
    if (!isMarkedFrontendBinaryVectorFunc(op))
      return mlir::WalkResult::advance();
    markedVectorFuncs.push_back(op);
    return mlir::WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return mlir::failure();

  for (mlir::Operation *funcOp : markedVectorFuncs)
    if (mlir::failed(lowerOneMarkedVectorFunc(module, funcOp)))
      return mlir::failure();

  return mlir::success();
}

struct LowerVectorRVVI32VAddToExecPass
    : impl::LowerVectorRVVI32VAddToExecBase<
          LowerVectorRVVI32VAddToExecPass> {
  void runOnOperation() override {
    if (mlir::failed(
            lowerMarkedFrontendVectorI32VAddInModule(getOperation())))
      signalPassFailure();
  }
};

struct LowerLinalgRVVBinaryToExecPass
    : impl::LowerLinalgRVVBinaryToExecBase<LowerLinalgRVVBinaryToExecPass> {
  void runOnOperation() override {
    if (mlir::failed(lowerMarkedFrontendBinaryLinalgInModule(getOperation())))
      signalPassFailure();
  }
};

struct LowerLinalgI32BinaryToExecPass
    : impl::LowerLinalgI32BinaryToExecBase<LowerLinalgI32BinaryToExecPass> {
  void runOnOperation() override {
    if (mlir::failed(lowerMarkedFrontendBinaryLinalgInModule(getOperation())))
      signalPassFailure();
  }
};

struct LowerLinalgI32VAddToExecPass
    : impl::LowerLinalgI32VAddToExecBase<LowerLinalgI32VAddToExecPass> {
  void runOnOperation() override {
    if (mlir::failed(lowerMarkedFrontendBinaryLinalgInModule(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerVectorRVVI32VAddToExecPass() {
  return std::make_unique<LowerVectorRVVI32VAddToExecPass>();
}

std::unique_ptr<mlir::Pass> createLowerLinalgRVVBinaryToExecPass() {
  return std::make_unique<LowerLinalgRVVBinaryToExecPass>();
}

std::unique_ptr<mlir::Pass> createLowerLinalgI32BinaryToExecPass() {
  return std::make_unique<LowerLinalgI32BinaryToExecPass>();
}

std::unique_ptr<mlir::Pass> createLowerLinalgI32VAddToExecPass() {
  return std::make_unique<LowerLinalgI32VAddToExecPass>();
}

} // namespace tianchenrv::transforms
