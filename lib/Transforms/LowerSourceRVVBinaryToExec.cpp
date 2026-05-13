#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/Exec/IR/CapabilityProviderComposition.h"
#include "TianChenRV/Support/FiniteBinaryFrontendLowering.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h"

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
#include <utility>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_LOWERLINALGRVVBINARYTOEXEC
#define GEN_PASS_DEF_LOWERLINALGI32BINARYTOEXEC
#define GEN_PASS_DEF_LOWERLINALGI32VADDTOEXEC
#define GEN_PASS_DEF_LOWERSOURCERVVBINARYTOEXEC
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
constexpr llvm::StringLiteral kSCFForOpName("scf.for");
constexpr llvm::StringLiteral kSCFYieldOpName("scf.yield");
constexpr llvm::StringLiteral kLinalgYieldOpName("linalg.yield");
constexpr llvm::StringLiteral kVectorTransferReadOpName(
    "vector.transfer_read");
constexpr llvm::StringLiteral kVectorTransferWriteOpName(
    "vector.transfer_write");
constexpr std::int64_t kVectorI32VAddSourceElements =
    support::kFrontendFixedVectorI32VAddSourceExtent;

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

enum class VectorFrontendAdapterMode {
  VAddOnly,
  ArithmeticFamily,
};

struct InferredFrontendBinarySource {
  const target::rvv::RVVBinaryFamilyDescriptor *family = nullptr;
  const support::FiniteBinaryFrontendContract *contract = nullptr;
  llvm::StringRef arithmeticOpName;
};

struct SourceFrontendLoweringRequest {
  mlir::Operation *sourceOp = nullptr;
  mlir::Operation *eraseOp = nullptr;
  llvm::StringRef frontendName;
  mlir::StringAttr kernelAttr;
  mlir::FlatSymbolRefAttr targetRef;
  mlir::Attribute capabilityProviderRefs;
  support::FiniteBinarySourceFrontendLoweringContract loweringContract;
};

std::string formatSupportedFrontendLowerings() {
  return target::rvv::formatRVVBinaryFrontendLoweringMarkers();
}

std::string formatSupportedDynamicVectorFrontendLowerings() {
  return target::rvv::formatRVVDynamicVectorFrontendLowerings();
}

mlir::LogicalResult populateI32VectorSourceIdentity(
    mlir::Operation *sourceOp, mlir::Operation *arithmeticOp,
    InferredFrontendBinarySource &out) {
  llvm::StringRef sourceArithmeticOpName =
      arithmeticOp ? arithmeticOp->getName().getStringRef() : llvm::StringRef();
  const target::rvv::RVVBinaryFamilyDescriptor *family =
      target::rvv::lookupRVVBinaryFamilyRegistrationByFrontendSource(
          support::FiniteBinaryElementKind::I32, sourceArithmeticOpName);
  if (!family || !family->frontendContract)
    return sourceOp->emitError()
           << "TianChen-RV vector source frontend inferred unsupported finite "
              "source arithmetic op '"
           << sourceArithmeticOpName << "'";
  if (!target::rvv::isRVVBinaryFamilyAcceptedByDynamicVectorSource(*family))
    return sourceOp->emitError()
           << "TianChen-RV vector source frontend supports only "
           << formatSupportedDynamicVectorFrontendLowerings()
           << "; inferred family '" << family->familyID
           << "' is not accepted because this pass is not a generic vector "
              "backend";

  out.family = family;
  out.contract = family->frontendContract;
  out.arithmeticOpName = family->sourceArithmeticOpName;
  return mlir::success();
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

const target::rvv::RVVBinaryFamilyDescriptor *
lookupRVVFrontendSourceFamily(support::FiniteBinaryElementKind elementKind,
                              mlir::Operation *arithmeticOp) {
  if (!arithmeticOp)
    return nullptr;
  return target::rvv::lookupRVVBinaryFamilyRegistrationByFrontendSource(
      elementKind, arithmeticOp->getName().getStringRef());
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
  const target::rvv::RVVBinaryFamilyDescriptor *family =
      lookupRVVFrontendSourceFamily(*elementKind, &arithmetic);
  if (!family || !family->frontendContract)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV bounded binary expects "
              "source body arithmetic to be arith.addi, arith.subi, or "
              "arith.muli";
  llvm::StringRef arithmeticOpName = family->sourceArithmeticOpName;

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

  out.family = family;
  out.contract = family->frontendContract;
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
    llvm::StringRef frontendName, llvm::StringRef targetName, TargetOp &out) {
  mlir::Operation *target = findTopLevelSymbol(module, targetName);
  if (!target)
    return sourceOp->emitError()
           << "TianChen-RV " << frontendName << " target @" << targetName
           << " must resolve to a module-level tcrv.exec.target";
  if (!llvm::isa<TargetOp>(target))
    return sourceOp->emitError()
           << "TianChen-RV " << frontendName << " target @" << targetName
           << " resolves to a top-level symbol that is not tcrv.exec.target";
  out = llvm::cast<TargetOp>(target);
  if (!tcrv::exec::isCapabilityProviderTarget(out))
    return sourceOp->emitError()
           << "TianChen-RV " << frontendName << " target @" << targetName
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
    mlir::Operation *sourceOp, llvm::StringRef frontendName,
    llvm::StringRef context, mlir::Operation *provider,
    llvm::StringSet<> &seenSymbols,
    llvm::StringSet<> &seenIDs) {
  llvm::StringRef symbol = tcrv::exec::getCapabilityProviderSymbolName(provider);
  if (symbol.empty())
    return sourceOp->emitError()
           << "TianChen-RV " << frontendName << " " << context
           << " provider must carry a symbol name";

  if (!seenSymbols.insert(symbol).second)
    return sourceOp->emitError()
           << "TianChen-RV " << frontendName << " " << context
           << " provider @"
           << symbol
           << " duplicates the selected target profile, target-composed "
              "provider, or another supplemental import";

  llvm::StringRef id = tcrv::exec::getCapabilityProviderID(provider);
  if (id.trim().empty())
    return sourceOp->emitError()
           << "TianChen-RV " << frontendName << " " << context
           << " provider @"
           << symbol << " must carry a non-empty capability id";

  if (!seenIDs.insert(id).second)
    return sourceOp->emitError()
           << "TianChen-RV " << frontendName << " " << context
           << " provider @"
           << symbol << " duplicates capability id '" << id
           << "' with the selected target profile, target-composed provider, "
              "or another supplemental import";

  return mlir::success();
}

mlir::LogicalResult seedSelectedTargetProviderScope(
    mlir::Operation *sourceOp, llvm::StringRef frontendName,
    TargetOp selectedTarget, llvm::StringSet<> &seenSymbols,
    llvm::StringSet<> &seenIDs) {
  if (mlir::failed(recordProviderIdentityForImportValidation(
          sourceOp, frontendName, "selected target",
          selectedTarget.getOperation(), seenSymbols, seenIDs)))
    return mlir::failure();

  llvm::Expected<llvm::SmallVector<mlir::Operation *, 8>> providers =
      tcrv::exec::collectComposedModuleCapabilityProviders(selectedTarget);
  if (!providers) {
    std::string message = llvm::toString(providers.takeError());
    return sourceOp->emitError() << message;
  }

  for (mlir::Operation *provider : *providers)
    if (mlir::failed(recordProviderIdentityForImportValidation(
            sourceOp, frontendName, "target-composed", provider, seenSymbols,
            seenIDs)))
      return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult collectCapabilityProviderImports(
    mlir::ModuleOp module, mlir::Operation *sourceOp,
    llvm::StringRef frontendName,
    TargetOp selectedTarget, mlir::ArrayAttr providerRefs,
    llvm::SmallVectorImpl<mlir::Operation *> &imports) {
  llvm::StringSet<> seenSymbols;
  llvm::StringSet<> seenIDs;

  if (mlir::failed(seedSelectedTargetProviderScope(
          sourceOp, frontendName, selectedTarget, seenSymbols, seenIDs)))
    return mlir::failure();

  for (mlir::Attribute attr : providerRefs) {
    auto ref = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!ref || ref.getValue().empty())
      return sourceOp->emitError()
             << "TianChen-RV " << frontendName
             << " capability-provider imports "
                "expect an array of non-empty module symbol references in '"
             << kFrontendCapabilityProvidersAttrName << "'";

    llvm::StringRef symbolName = ref.getValue();
    if (!seenSymbols.insert(symbolName).second)
      return sourceOp->emitError()
             << "TianChen-RV " << frontendName << " supplemental "
                "capability-provider import @"
             << symbolName
             << " duplicates the selected target profile, target-composed "
                "provider, or another supplemental import";

    mlir::Operation *provider = findTopLevelSymbol(module, symbolName);
    if (!provider)
      return sourceOp->emitError()
             << "TianChen-RV " << frontendName << " supplemental "
                "capability-provider import @"
             << symbolName << " must resolve to a module-level symbol";

    if (!tcrv::exec::isCapabilityProviderOperation(provider))
      return sourceOp->emitError()
             << "TianChen-RV " << frontendName << " supplemental "
                "capability-provider import @"
             << symbolName
             << " must resolve to a module-level tcrv.exec.capability or "
                "capability-provider tcrv.exec.target";

    llvm::StringRef id = tcrv::exec::getCapabilityProviderID(provider);
    if (id.empty())
      return sourceOp->emitError()
             << "TianChen-RV " << frontendName << " supplemental "
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
                sourceOp, frontendName, "supplemental target-composed",
                composedProvider, seenSymbols, seenIDs)))
          return mlir::failure();
    }

    imports.push_back(provider);
  }

  return mlir::success();
}

mlir::LogicalResult requireNoDuplicateKernelSymbol(mlir::ModuleOp module,
                                                   mlir::Operation *sourceOp,
                                                   llvm::StringRef frontendName,
                                                   llvm::StringRef kernelName) {
  if (!findTopLevelSymbol(module, kernelName))
    return mlir::success();
  return sourceOp->emitError()
         << "TianChen-RV " << frontendName << " kernel symbol @" << kernelName
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

bool isIntegerConstantValue(mlir::Operation *op, mlir::Type expectedType,
                            std::int64_t expectedValue) {
  if (!isOperationNamed(op, kArithConstantOpName) || op->getNumOperands() != 0 ||
      op->getNumResults() != 1 || op->getResult(0).getType() != expectedType)
    return false;

  auto value = op->getAttrOfType<mlir::IntegerAttr>("value");
  return value && value.getInt() == expectedValue;
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

mlir::LogicalResult requireVectorTransferTailActiveLaneAuthority(
    mlir::Operation *op, llvm::StringRef role) {
  auto inBounds = op->getAttrOfType<mlir::ArrayAttr>("in_bounds");
  if (!inBounds)
    return mlir::success();

  for (mlir::Attribute attr : inBounds) {
    auto flag = llvm::dyn_cast<mlir::BoolAttr>(attr);
    if (!flag)
      return op->emitError()
             << "TianChen-RV dynamic vector i32-vadd frontend expects "
             << role
             << " in_bounds metadata to be absent or boolean false so MLIR "
                "transfer tail semantics remain the source active-lane "
                "authority";
    if (flag.getValue())
      return op->emitError()
             << "TianChen-RV dynamic vector i32-vadd frontend expects "
             << role
             << " to expose MLIR transfer tail semantics; in_bounds = [true] "
                "is stale for runtime %n tail iterations";
  }

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

mlir::LogicalResult
requireDynamicVectorI32BinarySourceWrapper(
    mlir::Operation *funcOp, VectorFrontendAdapterMode mode,
    InferredFrontendBinarySource &source) {
  if (!isOperationNamed(funcOp, kFuncFuncOpName))
    return funcOp->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects a "
              "func.func wrapper";
  if (funcOp->getNumRegions() != 1 || !hasOneBlock(funcOp->getRegion(0)))
    return funcOp->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend wrapper expects "
              "one single-block func.func body";

  mlir::Block &body = funcOp->getRegion(0).front();
  if (body.getNumArguments() != 4)
    return funcOp->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend wrapper expects "
              "three memref<?xi32> buffers and one runtime %n: index";

  for (mlir::BlockArgument arg : body.getArguments().take_front(3)) {
    std::optional<unsigned> width =
        getRankedDynamicVectorMemRefIntegerElementWidth(arg.getType());
    if (!width || *width != 32)
      return funcOp->emitError()
             << "TianChen-RV dynamic vector i32 binary frontend wrapper "
                "expects lhs/rhs/output arguments to be rank-1 dynamic "
                "memref<?xi32>";
  }
  if (!body.getArgument(3).getType().isIndex())
    return funcOp->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend wrapper expects "
              "the fourth argument to be runtime %n: index";

  if (!llvm::hasNItems(body.getOperations(), 4))
    return funcOp->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend wrapper expects "
              "zero index, step-16 index, one scf.for, and func.return";

  auto it = body.begin();
  mlir::Operation *lowerConstant = &*it++;
  mlir::Operation *stepConstant = &*it++;
  mlir::Operation *forOp = &*it++;
  mlir::Operation *ret = &*it++;

  mlir::Type indexType = mlir::IndexType::get(funcOp->getContext());
  if (!isIntegerConstantZero(lowerConstant, indexType))
    return lowerConstant->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects the "
              "first source operation to be arith.constant 0 : index";
  if (!isIntegerConstantValue(stepConstant, indexType,
                              support::kFrontendDynamicVectorI32VAddLoopStep))
    return stepConstant->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects the "
              "second source operation to be arith.constant 16 : index";
  if (!isOperationNamed(forOp, kSCFForOpName) || forOp->getNumOperands() != 3 ||
      forOp->getNumResults() != 0 || forOp->getNumRegions() != 1 ||
      !hasOneBlock(forOp->getRegion(0)))
    return forOp->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects one "
              "scf.for without iter_args";
  if (forOp->getOperand(0) != lowerConstant->getResult(0) ||
      forOp->getOperand(1) != body.getArgument(3) ||
      forOp->getOperand(2) != stepConstant->getResult(0))
    return forOp->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects scf.for "
              "from zero to source %n in step 16";

  mlir::Block &loopBody = forOp->getRegion(0).front();
  if (loopBody.getNumArguments() != 1 ||
      !loopBody.getArgument(0).getType().isIndex())
    return forOp->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects scf.for "
              "to expose one index induction variable";
  if (!llvm::hasNItems(loopBody.getOperations(), 6))
    return forOp->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects scf.for "
              "body to contain zero i32 padding, two vector.transfer_read "
              "ops, one arith.addi or arith.subi, one vector.transfer_write, "
              "and scf.yield";

  auto loopIt = loopBody.begin();
  mlir::Operation *paddingConstant = &*loopIt++;
  mlir::Operation *lhsRead = &*loopIt++;
  mlir::Operation *rhsRead = &*loopIt++;
  mlir::Operation *add = &*loopIt++;
  mlir::Operation *write = &*loopIt++;
  mlir::Operation *yield = &*loopIt++;

  if (!isIntegerConstantZero(paddingConstant,
                             mlir::IntegerType::get(funcOp->getContext(), 32)))
    return paddingConstant->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects the "
              "first loop operation to be arith.constant 0 : i32";

  mlir::Value index = loopBody.getArgument(0);
  mlir::Value padding = paddingConstant->getResult(0);
  if (mlir::failed(requireVectorTransferRead(
          lhsRead, body.getArgument(0), index, padding, "lhs read")))
    return mlir::failure();
  if (mlir::failed(requireVectorTransferTailActiveLaneAuthority(lhsRead,
                                                               "lhs read")))
    return mlir::failure();
  if (mlir::failed(requireVectorTransferRead(
          rhsRead, body.getArgument(1), index, padding, "rhs read")))
    return mlir::failure();
  if (mlir::failed(requireVectorTransferTailActiveLaneAuthority(rhsRead,
                                                               "rhs read")))
    return mlir::failure();

  const target::rvv::RVVBinaryFamilyDescriptor *family =
      lookupRVVFrontendSourceFamily(support::FiniteBinaryElementKind::I32,
                                    add);
  bool isVAddFamily =
      family && family->familyID ==
                    target::rvv::getI32VAddFamilyRegistrationRecord().familyID;
  bool isAcceptedDynamicVectorFamily =
      family &&
      target::rvv::isRVVBinaryFamilyAcceptedByDynamicVectorSource(*family);
  if (!family || !isAcceptedDynamicVectorFamily ||
      (mode == VectorFrontendAdapterMode::VAddOnly && !isVAddFamily))
    return add->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects "
           << (mode == VectorFrontendAdapterMode::VAddOnly
                   ? "arith.addi"
                   : "arith.addi or arith.subi")
           << " over the two transfer-read vector values";
  llvm::StringRef arithmeticOpName = family->sourceArithmeticOpName;
  if (add->getOperand(0) != lhsRead->getResult(0) ||
      add->getOperand(1) != rhsRead->getResult(0) ||
      !isRankedVectorWithIntegerElementWidth(
          add->getResult(0).getType(), kVectorI32VAddSourceElements, 32))
    return add->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects "
           << arithmeticOpName
           << " to consume lhs/rhs vector<16xi32> reads and "
              "produce vector<16xi32>";

  if (!isOperationNamed(write, kVectorTransferWriteOpName) ||
      write->getNumOperands() != 3 || write->getNumResults() != 0)
    return write->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects "
              "vector.transfer_write of the arithmetic result";
  if (write->getOperand(0) != add->getResult(0) ||
      write->getOperand(1) != body.getArgument(2) ||
      write->getOperand(2) != index)
    return write->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects "
              "vector.transfer_write to store the arithmetic result to the "
              "output buffer at the scf.for induction variable";
  if (mlir::failed(requireVectorTransferTailActiveLaneAuthority(
          write, "output write")))
    return mlir::failure();

  if (!isOperationNamed(yield, kSCFYieldOpName) || yield->getNumOperands() != 0)
    return yield->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects "
              "scf.yield without operands";
  if (!isOperationNamed(ret, kFuncReturnOpName) || ret->getNumOperands() != 0)
    return ret->emitError()
           << "TianChen-RV dynamic vector i32 binary frontend expects "
              "func.return without operands";

  return populateI32VectorSourceIdentity(funcOp, add, source);
}

mlir::LogicalResult crossCheckVectorI32VAddMarker(mlir::Operation *funcOp,
                                                  llvm::StringRef marker) {
  const target::rvv::RVVBinaryFamilyDescriptor *markerFamily =
      target::rvv::lookupRVVBinaryFamilyRegistrationByFrontendLowering(marker);
  if (!markerFamily || !markerFamily->frontendContract)
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend expects '"
           << kFrontendLoweringAttrName << "' to be 'i32-vadd'";

  const target::rvv::RVVBinaryFamilyDescriptor &expected =
      target::rvv::getI32VAddFamilyRegistrationRecord();
  if (markerFamily->familyID != expected.familyID)
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend supports only marker "
              "'i32-vadd'; marker '"
           << marker
           << "' is not accepted because this pass is not a generic vector "
              "backend";

  return mlir::success();
}

mlir::LogicalResult crossCheckVectorFrontendMarker(
    mlir::Operation *funcOp, llvm::StringRef marker,
    const InferredFrontendBinarySource &source) {
  const target::rvv::RVVBinaryFamilyDescriptor *markerFamily =
      target::rvv::lookupRVVBinaryFamilyRegistrationByFrontendLowering(marker);
  if (!markerFamily || !markerFamily->frontendContract)
    return funcOp->emitError()
           << "TianChen-RV vector source frontend expects '"
           << kFrontendLoweringAttrName
           << "' to be " << formatSupportedDynamicVectorFrontendLowerings();

  const support::FiniteBinaryFrontendContract *markerContract =
      markerFamily->frontendContract;
  if (!target::rvv::isRVVBinaryFamilyAcceptedByDynamicVectorSource(
          *markerFamily))
    return funcOp->emitError()
           << "TianChen-RV vector source frontend supports only marker "
           << formatSupportedDynamicVectorFrontendLowerings() << "; marker '"
           << marker
           << "' is not accepted because this pass is not a generic vector "
              "backend";

  if (markerContract->dtypeID != source.contract->dtypeID)
    return funcOp->emitError()
           << "TianChen-RV vector source frontend has marker '" << marker
           << "' requesting dtype '" << markerContract->dtypeID
           << "' but source operands infer dtype '" << source.contract->dtypeID
           << "'; marker is only a bounded route request/cross-check";

  if (markerContract->familyID != source.contract->familyID)
    return funcOp->emitError()
           << "TianChen-RV vector source frontend has marker '" << marker
           << "' requesting family '" << markerContract->familyID
           << "' but source body infers family '" << source.contract->familyID
           << "' from " << source.arithmeticOpName
           << "; marker is only a bounded route request/cross-check";

  return mlir::success();
}

mlir::LogicalResult crossCheckFrontendMarker(
    mlir::Operation *linalgOp, llvm::StringRef marker,
    const InferredFrontendBinarySource &source) {
  const target::rvv::RVVBinaryFamilyDescriptor *markerFamily =
      target::rvv::lookupRVVBinaryFamilyRegistrationByFrontendLowering(marker);
  if (!markerFamily || !markerFamily->frontendContract)
    return linalgOp->emitError()
           << "marked linalg.generic for TianChen-RV expects '"
           << kFrontendLoweringAttrName
           << "' to be " << formatSupportedFrontendLowerings();

  const support::FiniteBinaryFrontendContract *markerContract =
      markerFamily->frontendContract;
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
                          const support::FiniteBinarySourceFrontendLoweringContract
                              &spec) {
  const support::FiniteBinaryFrontendContract &contract = *spec.contract;
  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPoint(sourceFunc);

  mlir::OperationState state(sourceFunc->getLoc(), KernelOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(kernelName));
  state.addAttribute("target", targetRef);
  state.addAttribute(kFrontendLoweringAttrName,
                     builder.getStringAttr(contract.frontendLowering));
  if (spec.fixedSourceVectorExtent) {
    state.addAttribute(support::kFrontendSourceKindAttrName,
                       builder.getStringAttr(
                           support::kFrontendFixedVectorI32VAddSourceKind));
    state.addAttribute(
        support::kFrontendSourceAuthorityAttrName,
        builder.getStringAttr(support::kFrontendFixedVectorSourceAuthority));
    state.addAttribute(
        support::kFrontendSourceVectorExtentAttrName,
        builder.getI64IntegerAttr(*spec.fixedSourceVectorExtent));
    state.addAttribute(
        support::kFrontendRuntimeElementCountConstraintAttrName,
        builder.getStringAttr(
            support::kFrontendRuntimeElementCountMustEqualSourceExtent));
  }
  if (spec.dynamicRuntimeExtentFromSCFUpperBound) {
    const target::rvv::RVVBinaryFamilyDescriptor *family =
        target::rvv::lookupRVVBinaryFamilyRegistrationByFrontendContract(
            contract);
    llvm::StringRef sourceKind =
        family ? target::rvv::getRVVDynamicVectorSourceKindForFamily(*family)
               : llvm::StringRef();
    state.addAttribute(support::kFrontendSourceKindAttrName,
                       builder.getStringAttr(sourceKind));
    state.addAttribute(
        support::kFrontendSourceAuthorityAttrName,
        builder.getStringAttr(support::kFrontendDynamicVectorSourceAuthority));
    state.addAttribute(support::kFrontendRuntimeExtentArgAttrName,
                       builder.getStringAttr("n"));
    state.addAttribute(
        support::kFrontendSourceLoopStepAttrName,
        builder.getI64IntegerAttr(
            support::kFrontendDynamicVectorI32VAddLoopStep));
    state.addAttribute(
        support::kFrontendSourceVectorChunkExtentAttrName,
        builder.getI64IntegerAttr(
            support::kFrontendDynamicVectorI32VAddChunkExtent));
    state.addAttribute(
        support::kFrontendActiveLaneAuthorityAttrName,
        builder.getStringAttr(
            support::kFrontendDynamicVectorActiveLaneAuthority));
    state.addAttribute(
        support::kFrontendSourceTailPolicyAttrName,
        builder.getStringAttr(support::kFrontendDynamicVectorSourceTailPolicy));
    state.addAttribute(
        support::kFrontendRuntimeElementCountConstraintAttrName,
        builder.getStringAttr(
            support::kFrontendRuntimeElementCountFromSourceRuntimeExtent));
  }
  state.addRegion();

  auto kernel = llvm::cast<KernelOp>(builder.create(state));
  kernel.getBody().push_back(new mlir::Block());
  return kernel;
}

mlir::LogicalResult materializeFrontendSourceRuntimeParamAttrs(
    KernelOp kernel,
    const support::FiniteBinarySourceFrontendLoweringContract &spec) {
  if (!spec.fixedSourceVectorExtent &&
      !spec.dynamicRuntimeExtentFromSCFUpperBound)
    return mlir::success();

  llvm::SmallVector<tcrv::exec::RuntimeParamOp, 1> runtimeParams;
  if (llvm::Error error = support::collectRuntimeABIParams(
          kernel, spec.runtimeElementCountSpecs, runtimeParams)) {
    kernel.emitError() << llvm::toString(std::move(error));
    return mlir::failure();
  }

  mlir::Builder builder(kernel.getContext());
  tcrv::exec::RuntimeParamOp runtimeElementCount = runtimeParams.front();
  if (spec.fixedSourceVectorExtent) {
    runtimeElementCount->setAttr(
        support::kFrontendSourceKindAttrName,
        builder.getStringAttr(support::kFrontendFixedVectorI32VAddSourceKind));
    runtimeElementCount->setAttr(
        support::kFrontendSourceAuthorityAttrName,
        builder.getStringAttr(support::kFrontendFixedVectorSourceAuthority));
    runtimeElementCount->setAttr(
        support::kFrontendSourceVectorExtentAttrName,
        builder.getI64IntegerAttr(*spec.fixedSourceVectorExtent));
    runtimeElementCount->setAttr(
        support::kFrontendRuntimeElementCountConstraintAttrName,
        builder.getStringAttr(
            support::kFrontendRuntimeElementCountMustEqualSourceExtent));
  }
  if (spec.dynamicRuntimeExtentFromSCFUpperBound) {
    const target::rvv::RVVBinaryFamilyDescriptor *family =
        target::rvv::lookupRVVBinaryFamilyRegistrationByFrontendContract(
            *spec.contract);
    llvm::StringRef sourceKind =
        family ? target::rvv::getRVVDynamicVectorSourceKindForFamily(*family)
               : llvm::StringRef();
    runtimeElementCount->setAttr(
        support::kFrontendSourceKindAttrName,
        builder.getStringAttr(sourceKind));
    runtimeElementCount->setAttr(
        support::kFrontendSourceAuthorityAttrName,
        builder.getStringAttr(support::kFrontendDynamicVectorSourceAuthority));
    runtimeElementCount->setAttr(support::kFrontendRuntimeExtentArgAttrName,
                                 builder.getStringAttr("n"));
    runtimeElementCount->setAttr(
        support::kFrontendSourceLoopStepAttrName,
        builder.getI64IntegerAttr(
            support::kFrontendDynamicVectorI32VAddLoopStep));
    runtimeElementCount->setAttr(
        support::kFrontendSourceVectorChunkExtentAttrName,
        builder.getI64IntegerAttr(
            support::kFrontendDynamicVectorI32VAddChunkExtent));
    runtimeElementCount->setAttr(
        support::kFrontendActiveLaneAuthorityAttrName,
        builder.getStringAttr(
            support::kFrontendDynamicVectorActiveLaneAuthority));
    runtimeElementCount->setAttr(
        support::kFrontendSourceTailPolicyAttrName,
        builder.getStringAttr(support::kFrontendDynamicVectorSourceTailPolicy));
    runtimeElementCount->setAttr(
        support::kFrontendRuntimeElementCountConstraintAttrName,
        builder.getStringAttr(
            support::kFrontendRuntimeElementCountFromSourceRuntimeExtent));
  }
  return mlir::success();
}

void materializeCapabilityProviderImports(
    KernelOp kernel, llvm::ArrayRef<mlir::Operation *> imports) {
  mlir::OpBuilder builder(kernel.getContext());
  builder.setInsertionPointToEnd(&kernel.getBody().front());
  for (mlir::Operation *provider : imports)
    builder.clone(*provider);
}

mlir::LogicalResult materializeFrontendBinaryABI(
    KernelOp kernel,
    const support::FiniteBinarySourceFrontendLoweringContract &spec) {
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

  return materializeFrontendSourceRuntimeParamAttrs(kernel, spec);
}

mlir::LogicalResult
lowerOneSourceFrontendRequest(mlir::ModuleOp module,
                              const SourceFrontendLoweringRequest &request) {
  mlir::Operation *sourceOp = request.sourceOp;
  mlir::Operation *eraseOp = request.eraseOp ? request.eraseOp : sourceOp;
  if (!sourceOp || !eraseOp || !request.loweringContract.contract)
    return module.emitError()
           << "TianChen-RV source frontend lowering received an incomplete "
              "adapter request";
  if (request.loweringContract.dynamicRuntimeExtentFromSCFUpperBound) {
    const target::rvv::RVVBinaryFamilyDescriptor *family =
        target::rvv::lookupRVVBinaryFamilyRegistrationByFrontendContract(
            *request.loweringContract.contract);
    if (!family ||
        target::rvv::getRVVDynamicVectorSourceKindForFamily(*family).empty())
      return sourceOp->emitError()
             << "TianChen-RV " << request.frontendName
             << " has no bounded dynamic vector source-kind adapter for family '"
             << request.loweringContract.contract->familyID << "'";
  }

  if (!request.kernelAttr || !isBareSymbolName(request.kernelAttr.getValue()))
    return sourceOp->emitError()
           << "TianChen-RV " << request.frontendName
           << " requires non-empty bare-symbol string attribute '"
           << kFrontendKernelAttrName << "'";

  if (!request.targetRef || request.targetRef.getValue().empty())
    return sourceOp->emitError()
           << "TianChen-RV " << request.frontendName
           << " requires module target symbol attribute '"
           << kFrontendTargetAttrName << "'";

  TargetOp selectedTarget;
  if (mlir::failed(requireTopLevelTargetProfile(
          module, sourceOp, request.frontendName, request.targetRef.getValue(),
          selectedTarget)))
    return mlir::failure();
  if (mlir::failed(requireNoDuplicateKernelSymbol(
          module, sourceOp, request.frontendName,
          request.kernelAttr.getValue())))
    return mlir::failure();

  llvm::SmallVector<mlir::Operation *, 4> providerImports;
  if (request.capabilityProviderRefs) {
    auto providerRefs =
        llvm::dyn_cast<mlir::ArrayAttr>(request.capabilityProviderRefs);
    if (!providerRefs)
      return sourceOp->emitError()
             << "TianChen-RV " << request.frontendName << " expects '"
             << kFrontendCapabilityProvidersAttrName
             << "' to be an array of module symbol references";

    if (mlir::failed(collectCapabilityProviderImports(
            module, sourceOp, request.frontendName, selectedTarget,
            providerRefs, providerImports)))
      return mlir::failure();
  }

  KernelOp kernel = createExecKernel(
      module, eraseOp, request.kernelAttr.getValue(), request.targetRef,
      request.loweringContract);
  materializeCapabilityProviderImports(kernel, providerImports);
  if (mlir::failed(
          materializeFrontendBinaryABI(kernel, request.loweringContract))) {
    kernel.erase();
    return mlir::failure();
  }

  eraseOp->erase();
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

  SourceFrontendLoweringRequest request;
  request.sourceOp = linalgOp;
  request.eraseOp = funcOp;
  request.frontendName = "linalg source frontend";
  request.kernelAttr = getKernelName(linalgOp, funcOp);
  request.targetRef = getTargetRef(linalgOp, funcOp);
  request.capabilityProviderRefs =
      getCapabilityProviderRefsAttr(linalgOp, funcOp);
  request.loweringContract =
      support::makeFiniteBinarySourceFrontendLoweringContract(
          *source.contract);
  return lowerOneSourceFrontendRequest(module, request);
}

mlir::LogicalResult lowerOneMarkedVectorFunc(mlir::ModuleOp module,
                                             mlir::Operation *funcOp,
                                             VectorFrontendAdapterMode mode) {
  auto frontendAttr = getStringAttr(funcOp, kFrontendLoweringAttrName);
  if (!frontendAttr)
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend expects '"
           << kFrontendLoweringAttrName << "' to be 'i32-vadd'";

  if (mode == VectorFrontendAdapterMode::VAddOnly &&
      mlir::failed(
          crossCheckVectorI32VAddMarker(funcOp, frontendAttr.getValue())))
    return mlir::failure();
  if (mlir::failed(requireNoLegacyDescriptorMetadata(
          funcOp, nullptr, "vector frontend",
          "source vector/arith body and typed operands")))
    return mlir::failure();

  InferredFrontendBinarySource source;
  support::FiniteBinarySourceFrontendLoweringContract loweringContract;
  if (funcOp->getNumRegions() != 1 || !hasOneBlock(funcOp->getRegion(0)))
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend expects one single-block "
              "func.func body";
  mlir::Block &body = funcOp->getRegion(0).front();
  if (body.getNumArguments() == 3) {
    if (mlir::failed(requireVectorI32VAddSourceWrapper(funcOp)))
      return mlir::failure();
    source.family = &target::rvv::getI32VAddFamilyRegistrationRecord();
    source.contract = source.family->frontendContract;
    source.arithmeticOpName = source.family->sourceArithmeticOpName;
    if (!source.contract)
      return funcOp->emitError()
             << "TianChen-RV vector i32-vadd frontend registry entry is "
                "missing its finite frontend contract";
    if (mode == VectorFrontendAdapterMode::ArithmeticFamily &&
        mlir::failed(crossCheckVectorFrontendMarker(
            funcOp, frontendAttr.getValue(), source)))
      return mlir::failure();
    loweringContract =
        support::makeFixedVectorI32VAddSourceFrontendLoweringContract();
  } else if (body.getNumArguments() == 4) {
    if (mlir::failed(requireDynamicVectorI32BinarySourceWrapper(
            funcOp, mode, source)))
      return mlir::failure();
    if (mode == VectorFrontendAdapterMode::ArithmeticFamily &&
        mlir::failed(crossCheckVectorFrontendMarker(
            funcOp, frontendAttr.getValue(), source)))
      return mlir::failure();
    if (mode == VectorFrontendAdapterMode::VAddOnly)
      loweringContract =
          support::makeDynamicVectorI32VAddSourceFrontendLoweringContract();
    else
      loweringContract =
          support::makeDynamicVectorI32SourceFrontendLoweringContract(
              *source.contract);
  } else {
    return funcOp->emitError()
           << "TianChen-RV vector i32-vadd frontend expects either the fixed "
              "three-buffer vector<16xi32> wrapper or the dynamic "
              "three-buffer plus runtime %n: index SCF wrapper";
  }

  SourceFrontendLoweringRequest request;
  request.sourceOp = funcOp;
  request.eraseOp = funcOp;
  request.frontendName = "vector source frontend";
  request.kernelAttr = getKernelName(funcOp, funcOp);
  request.targetRef = getTargetRef(funcOp, funcOp);
  request.capabilityProviderRefs = getCapabilityProviderRefsAttr(funcOp, funcOp);
  request.loweringContract = std::move(loweringContract);
  return lowerOneSourceFrontendRequest(module, request);
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
lowerMarkedFrontendVectorInModule(mlir::ModuleOp module,
                                  VectorFrontendAdapterMode mode) {
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
    if (mlir::failed(lowerOneMarkedVectorFunc(module, funcOp, mode)))
      return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult lowerMarkedSourceRVVBinaryFrontendsInModule(
    mlir::ModuleOp module) {
  if (mlir::failed(lowerMarkedFrontendVectorInModule(
          module, VectorFrontendAdapterMode::ArithmeticFamily)))
    return mlir::failure();
  return lowerMarkedFrontendBinaryLinalgInModule(module);
}

struct LowerSourceRVVBinaryToExecPass
    : impl::LowerSourceRVVBinaryToExecBase<LowerSourceRVVBinaryToExecPass> {
  void runOnOperation() override {
    if (mlir::failed(
            lowerMarkedSourceRVVBinaryFrontendsInModule(getOperation())))
      signalPassFailure();
  }
};

struct LowerVectorRVVI32VAddToExecPass
    : impl::LowerVectorRVVI32VAddToExecBase<
          LowerVectorRVVI32VAddToExecPass> {
  void runOnOperation() override {
    if (mlir::failed(lowerMarkedFrontendVectorInModule(
            getOperation(), VectorFrontendAdapterMode::VAddOnly)))
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

std::unique_ptr<mlir::Pass> createLowerSourceRVVBinaryToExecPass() {
  return std::make_unique<LowerSourceRVVBinaryToExecPass>();
}

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
