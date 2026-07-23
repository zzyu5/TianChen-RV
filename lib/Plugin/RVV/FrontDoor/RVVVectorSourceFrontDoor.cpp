#include "Weft/Plugin/RVV/RVVVectorSourceFrontDoor.h"
#include "Weft/Plugin/RVV/RVVFormulaCatalog.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVExtensionPlugin.h"
#include "Weft/Target/RVV/RVVTargetProfileBinding.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cctype>
#include <memory>
#include <string>

namespace weft::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kSeedAttrName("weft_rvv.lowering_seed");
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "weft_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("weft_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedVectorBinarySourceFrontDoorValue(
    "bounded_vector_source");
constexpr llvm::StringLiteral kAcceptedVectorCompareSelectSourceFrontDoorValue(
    "bounded_vector_compare_select_source");
constexpr llvm::StringLiteral
    kAcceptedVectorRuntimeScalarCompareSelectSourceFrontDoorValue(
        "bounded_vector_runtime_scalar_cmp_select_source");
mlir::LogicalResult failVectorMaterializer(mlir::Operation *op,
                                           llvm::Twine message) {
  op->emitError() << "bounded RVV vector-binary source front door failed: "
                  << message;
  return mlir::failure();
}

mlir::LogicalResult failVectorCompareSelectMaterializer(mlir::Operation *op,
                                                        llvm::Twine message) {
  op->emitError()
      << "bounded RVV vector-compare-select source front door failed: "
      << message;
  return mlir::failure();
}

mlir::LogicalResult
failVectorRuntimeScalarCompareSelectMaterializer(mlir::Operation *op,
                                                llvm::Twine message) {
  op->emitError()
      << "bounded RVV vector-runtime-scalar-cmp-select source front door "
         "failed: "
      << message;
  return mlir::failure();
}

mlir::LogicalResult
failVectorSourceFrontDoorFamilyRegistry(mlir::Operation *op,
                                        llvm::Twine message) {
  op->emitError() << "RVV vector source-front-door family registry failed: "
                  << message;
  return mlir::failure();
}

enum class RVVVectorSourceFrontDoorFamilyID {
  Binary,
  CompareSelect,
  RuntimeScalarCompareSelect
};

using RVVVectorSourceFrontDoorFailFn =
    mlir::LogicalResult (*)(mlir::Operation *, llvm::Twine);

struct RVVVectorSourceFrontDoorFamilyDescriptor {
  RVVVectorSourceFrontDoorFamilyID id;
  llvm::StringLiteral familyName;
  llvm::StringLiteral markerValue;
  llvm::StringLiteral passArgument;
  llvm::StringLiteral passDescription;
  llvm::StringLiteral sourceFunctionCandidateDescription;
  llvm::StringLiteral kernelNamePrefix;
  SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy
      defaultArtifactPolicy;
  RVVVectorSourceFrontDoorFailFn fail;
};

llvm::ArrayRef<RVVVectorSourceFrontDoorFamilyDescriptor>
getRVVVectorSourceFrontDoorFamilyRegistry() {
  static constexpr RVVVectorSourceFrontDoorFamilyDescriptor families[] = {
      {RVVVectorSourceFrontDoorFamilyID::Binary,
       "bounded-vector-binary-source-front-door",
       kAcceptedVectorBinarySourceFrontDoorValue,
       "weft-rvv-materialize-vector-binary-source-front-door",
       "Adapt one bounded MLIR Vector-like i32 binary source pattern into "
       "target-bound exact canonical P",
       "RVV vector-binary source function candidate", "rvv_vector_",
       SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
           ExplicitOnly,
       failVectorMaterializer},
      {RVVVectorSourceFrontDoorFamilyID::CompareSelect,
       "bounded-vector-compare-select-source-front-door",
       kAcceptedVectorCompareSelectSourceFrontDoorValue,
       "weft-rvv-materialize-vector-compare-select-source-front-door",
       "Adapt one bounded MLIR Vector-like i32 compare/select source pattern "
       "into target-bound exact canonical P",
       "RVV vector-compare-select source function candidate",
       "rvv_vector_cmp_select_",
       SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
           ExplicitOnly,
       failVectorCompareSelectMaterializer},
      {RVVVectorSourceFrontDoorFamilyID::RuntimeScalarCompareSelect,
       "bounded-vector-runtime-scalar-cmp-select-source-front-door",
       kAcceptedVectorRuntimeScalarCompareSelectSourceFrontDoorValue,
       "weft-rvv-materialize-vector-runtime-scalar-cmp-select-source-front-door",
       "Adapt one bounded MLIR Vector-like i32 runtime-scalar compare/select "
       "source pattern into target-bound exact canonical P",
       "RVV vector-runtime-scalar-cmp-select source function candidate",
       "rvv_vector_runtime_scalar_cmp_select_",
       SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
           ExplicitOnly,
       failVectorRuntimeScalarCompareSelectMaterializer},
  };
  return families;
}

const RVVVectorSourceFrontDoorFamilyDescriptor *
findRVVVectorSourceFrontDoorFamily(RVVVectorSourceFrontDoorFamilyID id) {
  for (const RVVVectorSourceFrontDoorFamilyDescriptor &family :
       getRVVVectorSourceFrontDoorFamilyRegistry()) {
    if (family.id == id)
      return &family;
  }
  return nullptr;
}

const RVVVectorSourceFrontDoorFamilyDescriptor &
getRVVVectorSourceFrontDoorFamily(RVVVectorSourceFrontDoorFamilyID id) {
  if (const RVVVectorSourceFrontDoorFamilyDescriptor *family =
          findRVVVectorSourceFrontDoorFamily(id))
    return *family;
  llvm_unreachable("unknown RVV vector source-front-door family id");
}

const RVVVectorSourceFrontDoorFamilyDescriptor *
findRVVVectorSourceFrontDoorFamilyByMarker(llvm::StringRef markerValue) {
  for (const RVVVectorSourceFrontDoorFamilyDescriptor &family :
       getRVVVectorSourceFrontDoorFamilyRegistry()) {
    if (markerValue == family.markerValue)
      return &family;
  }
  return nullptr;
}

std::string describeRVVVectorSourceFrontDoorFamilyMarkers() {
  std::string markers;
  for (const RVVVectorSourceFrontDoorFamilyDescriptor &family :
       getRVVVectorSourceFrontDoorFamilyRegistry()) {
    if (!markers.empty())
      markers += ", ";
    markers += "'";
    markers.append(family.markerValue.data(), family.markerValue.size());
    markers += "'";
  }
  return markers;
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

bool hasStaleRVVLoweringSeedMetadata(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *op) {
    if (found)
      return;
    found = op->hasAttr(kSeedAttrName);
  });
  return found;
}

mlir::LogicalResult requireRVVVectorSourceOnlyModule(
    mlir::ModuleOp module,
    const RVVVectorSourceFrontDoorFamilyDescriptor &family) {
  mlir::Operation *staleOp = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (staleOp || op == module.getOperation())
      return;
    llvm::StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "weft" || dialect == "weft_rvv" ||
        dialect == "weft_toy" || dialect == "weft_tensorext_lite")
      staleOp = op;
  });
  if (!staleOp)
    return mlir::success();

  return family.fail(
      staleOp,
      "source materializer requires RVV source-only MLIR input; pre-existing "
      "weft.exec/weft_rvv/weft_toy/weft_tensorext_lite selected-boundary or "
      "variant residue is not accepted");
}

std::string getDefaultRVVVectorSourceKernelName(
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    llvm::StringRef operationKind) {
  return (llvm::Twine(family.kernelNamePrefix) + operationKind +
          "_from_vector_source")
      .str();
}

mlir::FailureOr<std::string> getRVVVectorSourceKernelName(
    mlir::ModuleOp module,
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    llvm::StringRef operationKind) {
  auto kernelNameAttr =
      module->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  std::string defaultKernelName;
  llvm::StringRef kernelName;
  if (kernelNameAttr) {
    kernelName = kernelNameAttr.getValue().trim();
  } else {
    defaultKernelName =
        getDefaultRVVVectorSourceKernelName(family, operationKind);
    kernelName = defaultKernelName;
  }
  if (kernelName.empty()) {
    (void)family.fail(module, "source kernel name must be non-empty");
    return mlir::failure();
  }
  if (!isBoundedSymbolName(kernelName)) {
    (void)family.fail(module, "source kernel name must be a valid MLIR symbol");
    return mlir::failure();
  }
  return kernelName.str();
}

bool isRank1I32MemRef(mlir::Type type) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  return memref && memref.getRank() == 1 &&
         memref.getElementType().isInteger(32);
}

bool isRank1I32Vector(mlir::VectorType type) {
  return type && type.getRank() == 1 && type.getElementType().isInteger(32);
}

bool isRank1I1Vector(mlir::Type type) {
  auto vector = llvm::dyn_cast<mlir::VectorType>(type);
  return vector && vector.getRank() == 1 &&
         vector.getElementType().isInteger(1);
}

bool hasNoTransferMask(mlir::vector::TransferReadOp read) {
  return !static_cast<bool>(read.getMask());
}

bool hasNoTransferMask(mlir::vector::TransferWriteOp write) {
  return !static_cast<bool>(write.getMask());
}

bool hasOneIndex(mlir::Operation::operand_range indices) {
  return llvm::range_size(indices) == 1 &&
         (*indices.begin()).getType().isIndex();
}

llvm::StringRef getSupportedVectorBinaryKind(mlir::Operation *op) {
  if (llvm::isa<mlir::arith::AddIOp>(op))
    return "add";
  if (llvm::isa<mlir::arith::SubIOp>(op))
    return "sub";
  if (llvm::isa<mlir::arith::MulIOp>(op))
    return "mul";
  return {};
}

llvm::StringRef
getSupportedVectorComparePredicate(mlir::arith::CmpIOp compare) {
  switch (compare.getPredicate()) {
  case mlir::arith::CmpIPredicate::eq:
    return "eq";
  case mlir::arith::CmpIPredicate::slt:
    return "slt";
  case mlir::arith::CmpIPredicate::sle:
    return "sle";
  default:
    return {};
  }
}

bool hasVectorResult(mlir::Operation *op) {
  return op->getNumResults() == 1 &&
         llvm::isa<mlir::VectorType>(op->getResult(0).getType());
}

struct VectorBinarySourceMatch {
  mlir::func::FuncOp func;
  mlir::Value lhsSource;
  mlir::Value rhsSource;
  mlir::Value outDestination;
  mlir::Value runtimeN;
  mlir::VectorType sourceVectorType;
  std::string binaryKind;
};

struct VectorCompareSelectSourceMatch {
  mlir::func::FuncOp func;
  mlir::Value lhsSource;
  mlir::Value rhsSource;
  mlir::Value outDestination;
  mlir::Value runtimeN;
  mlir::VectorType sourceVectorType;
  std::string predicateKind;
};

struct VectorRuntimeScalarCompareSelectSourceMatch {
  mlir::func::FuncOp func;
  mlir::Value lhsSource;
  mlir::Value rhsScalar;
  mlir::Value trueValueSource;
  mlir::Value falseValueSource;
  mlir::Value outDestination;
  mlir::Value runtimeN;
  mlir::VectorType sourceVectorType;
  std::string predicateKind;
};

mlir::FailureOr<VectorBinarySourceMatch>
matchBoundedVectorBinarySourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration()) {
    (void)failVectorMaterializer(
        func, "source function must have a body for structural pattern match");
    return mlir::failure();
  }

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0) {
    (void)failVectorMaterializer(
        func,
        "source function must have exactly four inputs and no results: "
        "lhs/rhs/out memref<?xi32> plus n index");
    return mlir::failure();
  }
  if (!isRank1I32MemRef(type.getInput(0)) ||
      !isRank1I32MemRef(type.getInput(1)) ||
      !isRank1I32MemRef(type.getInput(2)) || !type.getInput(3).isIndex()) {
    (void)failVectorMaterializer(
        func,
        "source function inputs must be lhs/rhs/out rank-1 i32 memrefs and "
        "one runtime n index");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::vector::TransferReadOp, 2> reads;
  llvm::SmallVector<mlir::vector::TransferWriteOp, 1> writes;
  llvm::SmallVector<mlir::Operation *, 1> binaryOps;
  mlir::Operation *unsupportedVectorOp = nullptr;
  mlir::Operation *unsupportedArithVectorOp = nullptr;
  func.walk([&](mlir::Operation *op) {
    if (auto read = llvm::dyn_cast<mlir::vector::TransferReadOp>(op)) {
      reads.push_back(read);
      return;
    }
    if (auto write = llvm::dyn_cast<mlir::vector::TransferWriteOp>(op)) {
      writes.push_back(write);
      return;
    }
    if (op->getName().getDialectNamespace() == "vector" &&
        !unsupportedVectorOp)
      unsupportedVectorOp = op;
    if (op->getName().getDialectNamespace() == "arith" &&
        hasVectorResult(op)) {
      if (!getSupportedVectorBinaryKind(op).empty())
        binaryOps.push_back(op);
      else if (!unsupportedArithVectorOp)
        unsupportedArithVectorOp = op;
    }
  });
  if (unsupportedVectorOp) {
    (void)failVectorMaterializer(
        unsupportedVectorOp,
        "only vector.transfer_read and vector.transfer_write are supported "
        "by this bounded vector-binary source pattern");
    return mlir::failure();
  }
  if (unsupportedArithVectorOp) {
    (void)failVectorMaterializer(
        unsupportedArithVectorOp,
        "only arith.addi, arith.subi, and arith.muli vector binary ops are "
        "supported by this bounded source path");
    return mlir::failure();
  }
  if (reads.size() != 2 || writes.size() != 1 || binaryOps.size() != 1) {
    (void)failVectorMaterializer(
        func,
        "source pattern must contain exactly two vector.transfer_read ops, "
        "one supported arith vector binary op, and one vector.transfer_write");
    return mlir::failure();
  }

  mlir::Operation *binaryOp = binaryOps.front();
  llvm::StringRef binaryKind = getSupportedVectorBinaryKind(binaryOp);
  auto lhsRead = binaryOp->getOperand(0)
                     .getDefiningOp<mlir::vector::TransferReadOp>();
  auto rhsRead = binaryOp->getOperand(1)
                     .getDefiningOp<mlir::vector::TransferReadOp>();
  if (!lhsRead || !rhsRead || lhsRead == rhsRead) {
    (void)failVectorMaterializer(
        binaryOp,
        "arith vector binary operands must be produced by the two source "
        "vector.transfer_read ops");
    return mlir::failure();
  }

  mlir::Block &entry = func.getBody().front();
  if (lhsRead.getSource() != entry.getArgument(0) ||
      rhsRead.getSource() != entry.getArgument(1) ||
      writes.front().getSource() != entry.getArgument(2) ||
      entry.getNumArguments() != 4) {
    (void)failVectorMaterializer(
        func,
        "source memory roles must be structural: transfer_read lhs/rhs from "
        "the first two arguments and transfer_write to the third argument");
    return mlir::failure();
  }

  mlir::vector::TransferWriteOp write = writes.front();
  if (write.getVector() != binaryOp->getResult(0)) {
    (void)failVectorMaterializer(
        write,
        "vector.transfer_write must store the arith vector binary result");
    return mlir::failure();
  }

  mlir::VectorType vectorType =
      llvm::dyn_cast<mlir::VectorType>(binaryOp->getResult(0).getType());
  if (!isRank1I32Vector(vectorType) ||
      lhsRead.getVector().getType() != vectorType ||
      rhsRead.getVector().getType() != vectorType ||
      write.getVector().getType() != vectorType) {
    (void)failVectorMaterializer(
        binaryOp,
        "source vector operands and result must share one rank-1 i32 vector "
        "type");
    return mlir::failure();
  }
  if (!hasNoTransferMask(lhsRead) || !hasNoTransferMask(rhsRead) ||
      !hasNoTransferMask(write)) {
    (void)failVectorMaterializer(
        func, "masked vector transfers are outside this bounded source path");
    return mlir::failure();
  }
  if (!hasOneIndex(lhsRead.getIndices()) || !hasOneIndex(rhsRead.getIndices()) ||
      !hasOneIndex(write.getIndices())) {
    (void)failVectorMaterializer(
        func,
        "source vector transfers must be rank-1 unit-stride memory accesses");
    return mlir::failure();
  }
  if (!lhsRead.getPadding().getType().isInteger(32) ||
      !rhsRead.getPadding().getType().isInteger(32)) {
    (void)failVectorMaterializer(
        func,
        "source vector.transfer_read padding must match the i32 element type");
    return mlir::failure();
  }

  return VectorBinarySourceMatch{func,
                                 entry.getArgument(0),
                                 entry.getArgument(1),
                                 entry.getArgument(2),
                                 entry.getArgument(3),
                                 vectorType,
                                 binaryKind.str()};
}

mlir::FailureOr<VectorCompareSelectSourceMatch>
matchBoundedVectorCompareSelectSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration()) {
    (void)failVectorCompareSelectMaterializer(
        func, "source function must have a body for structural pattern match");
    return mlir::failure();
  }

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0) {
    (void)failVectorCompareSelectMaterializer(
        func,
        "source function must have exactly four inputs and no results: "
        "lhs/rhs/out memref<?xi32> plus n index");
    return mlir::failure();
  }
  if (!isRank1I32MemRef(type.getInput(0)) ||
      !isRank1I32MemRef(type.getInput(1)) ||
      !isRank1I32MemRef(type.getInput(2)) || !type.getInput(3).isIndex()) {
    (void)failVectorCompareSelectMaterializer(
        func,
        "source function inputs must be lhs/rhs/out rank-1 i32 memrefs and "
        "one runtime n index");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::vector::TransferReadOp, 2> reads;
  llvm::SmallVector<mlir::vector::TransferWriteOp, 1> writes;
  llvm::SmallVector<mlir::arith::CmpIOp, 1> compareOps;
  llvm::SmallVector<mlir::arith::SelectOp, 1> selectOps;
  mlir::Operation *unsupportedVectorOp = nullptr;
  mlir::Operation *unsupportedArithVectorOp = nullptr;
  func.walk([&](mlir::Operation *op) {
    if (auto read = llvm::dyn_cast<mlir::vector::TransferReadOp>(op)) {
      reads.push_back(read);
      return;
    }
    if (auto write = llvm::dyn_cast<mlir::vector::TransferWriteOp>(op)) {
      writes.push_back(write);
      return;
    }
    if (auto compare = llvm::dyn_cast<mlir::arith::CmpIOp>(op)) {
      if (isRank1I1Vector(compare.getResult().getType())) {
        if (!getSupportedVectorComparePredicate(compare).empty())
          compareOps.push_back(compare);
        else if (!unsupportedArithVectorOp)
          unsupportedArithVectorOp = op;
      }
      return;
    }
    if (auto select = llvm::dyn_cast<mlir::arith::SelectOp>(op)) {
      if (hasVectorResult(op))
        selectOps.push_back(select);
      return;
    }
    if (op->getName().getDialectNamespace() == "vector" &&
        !unsupportedVectorOp)
      unsupportedVectorOp = op;
    if (op->getName().getDialectNamespace() == "arith" &&
        hasVectorResult(op) && !unsupportedArithVectorOp)
      unsupportedArithVectorOp = op;
  });
  if (unsupportedVectorOp) {
    (void)failVectorCompareSelectMaterializer(
        unsupportedVectorOp,
        "only vector.transfer_read and vector.transfer_write are supported "
        "by this bounded vector-compare-select source pattern");
    return mlir::failure();
  }
  if (unsupportedArithVectorOp) {
    (void)failVectorCompareSelectMaterializer(
        unsupportedArithVectorOp,
        "only arith.cmpi predicates eq, slt, or sle plus arith.select are "
        "supported by this bounded source path");
    return mlir::failure();
  }
  if (reads.size() != 2 || writes.size() != 1 || compareOps.size() != 1 ||
      selectOps.size() != 1) {
    (void)failVectorCompareSelectMaterializer(
        func,
        "source pattern must contain exactly two vector.transfer_read ops, "
        "one supported arith.cmpi vector compare op, one arith.select op, "
        "and one vector.transfer_write");
    return mlir::failure();
  }

  mlir::arith::CmpIOp compare = compareOps.front();
  mlir::arith::SelectOp select = selectOps.front();
  llvm::StringRef predicateKind = getSupportedVectorComparePredicate(compare);
  auto lhsRead =
      compare.getLhs().getDefiningOp<mlir::vector::TransferReadOp>();
  auto rhsRead =
      compare.getRhs().getDefiningOp<mlir::vector::TransferReadOp>();
  if (!lhsRead || !rhsRead || lhsRead == rhsRead) {
    (void)failVectorCompareSelectMaterializer(
        compare,
        "arith.cmpi operands must be produced by the two source "
        "vector.transfer_read ops");
    return mlir::failure();
  }

  mlir::Block &entry = func.getBody().front();
  if (lhsRead.getSource() != entry.getArgument(0) ||
      rhsRead.getSource() != entry.getArgument(1) ||
      writes.front().getSource() != entry.getArgument(2) ||
      entry.getNumArguments() != 4) {
    (void)failVectorCompareSelectMaterializer(
        func,
        "source memory roles must be structural: transfer_read lhs/rhs from "
        "the first two arguments and transfer_write to the third argument");
    return mlir::failure();
  }

  if (select.getCondition() != compare.getResult()) {
    (void)failVectorCompareSelectMaterializer(
        select, "arith.select condition must be the arith.cmpi result");
    return mlir::failure();
  }
  if (select.getTrueValue() != lhsRead.getVector() ||
      select.getFalseValue() != rhsRead.getVector()) {
    (void)failVectorCompareSelectMaterializer(
        select,
        "arith.select layout must select lhs when the compare mask is true "
        "and rhs when it is false");
    return mlir::failure();
  }

  mlir::vector::TransferWriteOp write = writes.front();
  if (write.getVector() != select.getResult()) {
    (void)failVectorCompareSelectMaterializer(
        write, "vector.transfer_write must store the arith.select result");
    return mlir::failure();
  }

  mlir::VectorType vectorType =
      llvm::dyn_cast<mlir::VectorType>(select.getResult().getType());
  if (!isRank1I32Vector(vectorType) ||
      lhsRead.getVector().getType() != vectorType ||
      rhsRead.getVector().getType() != vectorType ||
      compare.getLhs().getType() != vectorType ||
      compare.getRhs().getType() != vectorType ||
      write.getVector().getType() != vectorType) {
    (void)failVectorCompareSelectMaterializer(
        select,
        "source compare/select operands and result must share one rank-1 i32 "
        "vector type");
    return mlir::failure();
  }
  if (!hasNoTransferMask(lhsRead) || !hasNoTransferMask(rhsRead) ||
      !hasNoTransferMask(write)) {
    (void)failVectorCompareSelectMaterializer(
        func, "masked vector transfers are outside this bounded source path");
    return mlir::failure();
  }
  if (!hasOneIndex(lhsRead.getIndices()) || !hasOneIndex(rhsRead.getIndices()) ||
      !hasOneIndex(write.getIndices())) {
    (void)failVectorCompareSelectMaterializer(
        func,
        "source vector transfers must be rank-1 unit-stride memory accesses");
    return mlir::failure();
  }
  if (!lhsRead.getPadding().getType().isInteger(32) ||
      !rhsRead.getPadding().getType().isInteger(32)) {
    (void)failVectorCompareSelectMaterializer(
        func,
        "source vector.transfer_read padding must match the i32 element type");
    return mlir::failure();
  }

  return VectorCompareSelectSourceMatch{func,
                                        entry.getArgument(0),
                                        entry.getArgument(1),
                                        entry.getArgument(2),
                                        entry.getArgument(3),
                                        vectorType,
                                        predicateKind.str()};
}

mlir::FailureOr<VectorRuntimeScalarCompareSelectSourceMatch>
matchBoundedVectorRuntimeScalarCompareSelectSourceFunc(
    mlir::func::FuncOp func) {
  if (func.isDeclaration()) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        func, "source function must have a body for structural pattern match");
    return mlir::failure();
  }

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 6 || type.getNumResults() != 0) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        func,
        "source function must have exactly six inputs and no results: "
        "lhs memref<?xi32>, rhs_scalar i32, true_value/false_value/out "
        "memref<?xi32>, plus n index");
    return mlir::failure();
  }
  if (!isRank1I32MemRef(type.getInput(0)) ||
      !type.getInput(1).isInteger(32) ||
      !isRank1I32MemRef(type.getInput(2)) ||
      !isRank1I32MemRef(type.getInput(3)) ||
      !isRank1I32MemRef(type.getInput(4)) || !type.getInput(5).isIndex()) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        func,
        "source function inputs must be lhs rank-1 i32 memref, rhs_scalar "
        "i32, true_value/false_value/out rank-1 i32 memrefs, and one "
        "runtime n index");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::vector::TransferReadOp, 3> reads;
  llvm::SmallVector<mlir::vector::TransferWriteOp, 1> writes;
  llvm::SmallVector<mlir::vector::SplatOp, 1> splats;
  llvm::SmallVector<mlir::arith::CmpIOp, 1> compareOps;
  llvm::SmallVector<mlir::arith::SelectOp, 1> selectOps;
  mlir::Operation *unsupportedVectorOp = nullptr;
  mlir::Operation *unsupportedArithVectorOp = nullptr;
  func.walk([&](mlir::Operation *op) {
    if (auto read = llvm::dyn_cast<mlir::vector::TransferReadOp>(op)) {
      reads.push_back(read);
      return;
    }
    if (auto write = llvm::dyn_cast<mlir::vector::TransferWriteOp>(op)) {
      writes.push_back(write);
      return;
    }
    if (auto splat = llvm::dyn_cast<mlir::vector::SplatOp>(op)) {
      splats.push_back(splat);
      return;
    }
    if (auto compare = llvm::dyn_cast<mlir::arith::CmpIOp>(op)) {
      if (isRank1I1Vector(compare.getResult().getType())) {
        if (!getSupportedVectorComparePredicate(compare).empty())
          compareOps.push_back(compare);
        else if (!unsupportedArithVectorOp)
          unsupportedArithVectorOp = op;
      }
      return;
    }
    if (auto select = llvm::dyn_cast<mlir::arith::SelectOp>(op)) {
      if (hasVectorResult(op))
        selectOps.push_back(select);
      return;
    }
    if (op->getName().getDialectNamespace() == "vector" &&
        !unsupportedVectorOp)
      unsupportedVectorOp = op;
    if (op->getName().getDialectNamespace() == "arith" &&
        hasVectorResult(op) && !unsupportedArithVectorOp)
      unsupportedArithVectorOp = op;
  });
  if (unsupportedVectorOp) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        unsupportedVectorOp,
        "only vector.transfer_read, vector.splat, and "
        "vector.transfer_write are supported by this bounded "
        "runtime-scalar compare/select source pattern");
    return mlir::failure();
  }
  if (unsupportedArithVectorOp) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        unsupportedArithVectorOp,
        "only arith.cmpi predicates eq, slt, or sle plus arith.select are "
        "supported by this bounded runtime-scalar source path");
    return mlir::failure();
  }
  if (reads.size() != 3 || writes.size() != 1 || splats.size() != 1 ||
      compareOps.size() != 1 || selectOps.size() != 1) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        func,
        "source pattern must contain exactly three vector.transfer_read ops, "
        "one vector.splat of rhs_scalar, one supported arith.cmpi vector "
        "compare op, one arith.select op, and one vector.transfer_write");
    return mlir::failure();
  }

  mlir::arith::CmpIOp compare = compareOps.front();
  mlir::arith::SelectOp select = selectOps.front();
  llvm::StringRef predicateKind = getSupportedVectorComparePredicate(compare);
  auto lhsRead =
      compare.getLhs().getDefiningOp<mlir::vector::TransferReadOp>();
  auto rhsSplat = compare.getRhs().getDefiningOp<mlir::vector::SplatOp>();
  if (!lhsRead || !rhsSplat) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        compare,
        "arith.cmpi must compare the lhs vector.transfer_read against the "
        "vector.splat result of rhs_scalar");
    return mlir::failure();
  }

  mlir::Block &entry = func.getBody().front();
  if (lhsRead.getSource() != entry.getArgument(0) ||
      rhsSplat.getInput() != entry.getArgument(1) ||
      writes.front().getSource() != entry.getArgument(4) ||
      entry.getNumArguments() != 6) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        func,
        "source roles must be structural: lhs transfer_read from the first "
        "argument, vector.splat from rhs_scalar, transfer_write to the fifth "
        "argument, and runtime n as the sixth argument");
    return mlir::failure();
  }

  if (select.getCondition() != compare.getResult()) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        select, "arith.select condition must be the arith.cmpi result");
    return mlir::failure();
  }
  auto trueRead =
      select.getTrueValue().getDefiningOp<mlir::vector::TransferReadOp>();
  auto falseRead =
      select.getFalseValue().getDefiningOp<mlir::vector::TransferReadOp>();
  if (!trueRead || !falseRead || trueRead == falseRead ||
      trueRead.getSource() != entry.getArgument(2) ||
      falseRead.getSource() != entry.getArgument(3)) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        select,
        "arith.select layout must select true_value when the runtime-scalar "
        "compare mask is true and false_value when it is false");
    return mlir::failure();
  }

  mlir::vector::TransferWriteOp write = writes.front();
  if (write.getVector() != select.getResult()) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        write, "vector.transfer_write must store the arith.select result");
    return mlir::failure();
  }

  mlir::VectorType vectorType =
      llvm::dyn_cast<mlir::VectorType>(select.getResult().getType());
  if (!isRank1I32Vector(vectorType) ||
      lhsRead.getVector().getType() != vectorType ||
      rhsSplat.getResult().getType() != vectorType ||
      trueRead.getVector().getType() != vectorType ||
      falseRead.getVector().getType() != vectorType ||
      compare.getLhs().getType() != vectorType ||
      compare.getRhs().getType() != vectorType ||
      write.getVector().getType() != vectorType) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        select,
        "source compare/select operands and result must share one rank-1 i32 "
        "vector type");
    return mlir::failure();
  }
  if (!hasNoTransferMask(lhsRead) || !hasNoTransferMask(trueRead) ||
      !hasNoTransferMask(falseRead) || !hasNoTransferMask(write)) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        func, "masked vector transfers are outside this bounded source path");
    return mlir::failure();
  }
  if (!hasOneIndex(lhsRead.getIndices()) ||
      !hasOneIndex(trueRead.getIndices()) ||
      !hasOneIndex(falseRead.getIndices()) || !hasOneIndex(write.getIndices())) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        func,
        "source vector transfers must be rank-1 unit-stride memory accesses");
    return mlir::failure();
  }
  if (!lhsRead.getPadding().getType().isInteger(32) ||
      !trueRead.getPadding().getType().isInteger(32) ||
      !falseRead.getPadding().getType().isInteger(32)) {
    (void)failVectorRuntimeScalarCompareSelectMaterializer(
        func,
        "source vector.transfer_read padding must match the i32 element type");
    return mlir::failure();
  }

  return VectorRuntimeScalarCompareSelectSourceMatch{
      func,           entry.getArgument(0), entry.getArgument(1),
      entry.getArgument(2), entry.getArgument(3), entry.getArgument(4),
      entry.getArgument(5), vectorType,           predicateKind.str()};
}

mlir::FailureOr<bool> matchRVVVectorSourceFrontDoorFamilyMarker(
    mlir::ModuleOp module,
    const RVVVectorSourceFrontDoorFamilyDescriptor &family) {
  auto marker =
      module->getAttrOfType<mlir::StringAttr>(kSourceFrontDoorAttrName);
  if (!marker)
    return false;

  llvm::StringRef markerValue = marker.getValue().trim();
  if (markerValue == family.markerValue) {
    if (hasStaleRVVLoweringSeedMetadata(module)) {
      (void)failVectorSourceFrontDoorFamilyRegistry(
          module,
          llvm::Twine("family '") + family.familyName +
              "' rejected stale weft_rvv.lowering_seed metadata as RVV "
              "source-route authority");
      return mlir::failure();
    }
    return true;
  }

  if (findRVVVectorSourceFrontDoorFamilyByMarker(markerValue))
    return false;

  std::string registeredMarkers =
      describeRVVVectorSourceFrontDoorFamilyMarkers();
  (void)failVectorSourceFrontDoorFamilyRegistry(
      module,
      llvm::Twine("unknown weft_rvv.source_front_door marker '") +
          markerValue + "'; registered RVV vector source-front-door markers "
                        "are " +
          registeredMarkers);
  return mlir::failure();
}

mlir::FailureOr<VectorBinarySourceMatch>
matchVectorBinarySourceFrontDoor(
    mlir::ModuleOp module,
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
                                 std::string &kernelName) {
  mlir::FailureOr<bool> selected =
      matchRVVVectorSourceFrontDoorFamilyMarker(module, family);
  if (mlir::failed(selected))
    return mlir::failure();
  if (!*selected)
    return VectorBinarySourceMatch{};

  if (mlir::failed(requireRVVVectorSourceOnlyModule(module, family)))
    return mlir::failure();

  llvm::SmallVector<mlir::func::FuncOp, 2> funcs;
  module.walk([&](mlir::func::FuncOp func) { funcs.push_back(func); });
  if (funcs.size() != 1) {
    (void)family.fail(
        module,
        llvm::Twine("source module must contain exactly one ") +
            family.sourceFunctionCandidateDescription);
    return mlir::failure();
  }

  mlir::FailureOr<VectorBinarySourceMatch> match =
      matchBoundedVectorBinarySourceFunc(funcs.front());
  if (mlir::failed(match))
    return mlir::failure();

  mlir::FailureOr<std::string> name =
      getRVVVectorSourceKernelName(module, family, match->binaryKind);
  if (mlir::failed(name))
    return mlir::failure();
  kernelName = *name;
  return match;
}

mlir::FailureOr<VectorCompareSelectSourceMatch>
matchVectorCompareSelectSourceFrontDoor(
    mlir::ModuleOp module,
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    std::string &kernelName) {
  mlir::FailureOr<bool> selected =
      matchRVVVectorSourceFrontDoorFamilyMarker(module, family);
  if (mlir::failed(selected))
    return mlir::failure();
  if (!*selected)
    return VectorCompareSelectSourceMatch{};

  if (mlir::failed(requireRVVVectorSourceOnlyModule(module, family)))
    return mlir::failure();

  llvm::SmallVector<mlir::func::FuncOp, 2> funcs;
  module.walk([&](mlir::func::FuncOp func) { funcs.push_back(func); });
  if (funcs.size() != 1) {
    (void)family.fail(
        module,
        llvm::Twine("source module must contain exactly one ") +
            family.sourceFunctionCandidateDescription);
    return mlir::failure();
  }

  mlir::FailureOr<VectorCompareSelectSourceMatch> match =
      matchBoundedVectorCompareSelectSourceFunc(funcs.front());
  if (mlir::failed(match))
    return mlir::failure();

  mlir::FailureOr<std::string> name =
      getRVVVectorSourceKernelName(module, family, match->predicateKind);
  if (mlir::failed(name))
    return mlir::failure();
  kernelName = *name;
  return match;
}

mlir::FailureOr<VectorRuntimeScalarCompareSelectSourceMatch>
matchVectorRuntimeScalarCompareSelectSourceFrontDoor(
    mlir::ModuleOp module,
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    std::string &kernelName) {
  mlir::FailureOr<bool> selected =
      matchRVVVectorSourceFrontDoorFamilyMarker(module, family);
  if (mlir::failed(selected))
    return mlir::failure();
  if (!*selected)
    return VectorRuntimeScalarCompareSelectSourceMatch{};

  if (mlir::failed(requireRVVVectorSourceOnlyModule(module, family)))
    return mlir::failure();

  llvm::SmallVector<mlir::func::FuncOp, 2> funcs;
  module.walk([&](mlir::func::FuncOp func) { funcs.push_back(func); });
  if (funcs.size() != 1) {
    (void)family.fail(
        module,
        llvm::Twine("source module must contain exactly one ") +
            family.sourceFunctionCandidateDescription);
    return mlir::failure();
  }

  mlir::FailureOr<VectorRuntimeScalarCompareSelectSourceMatch> match =
      matchBoundedVectorRuntimeScalarCompareSelectSourceFunc(funcs.front());
  if (mlir::failed(match))
    return mlir::failure();

  mlir::FailureOr<std::string> name =
      getRVVVectorSourceKernelName(module, family, match->predicateKind);
  if (mlir::failed(name))
    return mlir::failure();
  kernelName = *name;
  return match;
}

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

mlir::LogicalResult materializeRVVVectorBinarySourceKernel(
    mlir::OpBuilder &builder, llvm::StringRef kernelName,
    VectorBinarySourceMatch source) {
  mlir::Location loc = source.func.getLoc();

  mlir::ModuleOp module = source.func->getParentOfType<mlir::ModuleOp>();
  llvm::Expected<weft::exec::TargetOp> target =
      weft::target::rvv::materializeRVVSourceTargetProfile(
          builder, module, loc, kernelName);
  if (!target)
    return failVectorMaterializer(source.func,
                                  llvm::toString(target.takeError()));

  mlir::OperationState kernelState(loc,
                                   weft::exec::KernelOp::getOperationName());
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addAttribute("target",
                           symbolRef(builder, target->getSymName()));
  kernelState.addAttribute("problem", symbolRef(builder, "canonical_problem"));
  kernelState.addRegion();
  auto kernel = llvm::cast<weft::exec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  mlir::OperationState problemState(
      loc, weft::exec::I32VectorBinaryProblemOp::getOperationName());
  problemState.addAttribute("sym_name",
                            builder.getStringAttr("canonical_problem"));
  problemState.addAttribute("kind", builder.getStringAttr(source.binaryKind));
  problemState.addAttribute(
      "source_vector_lanes",
      builder.getI64IntegerAttr(source.sourceVectorType.getDimSize(0)));
  (void)builder.create(problemState);

  // Source adaptation stops at exact P + target binding. Candidate creation,
  // selection, and typed-body construction belong to the RVV owner lifecycle.
  return mlir::success();
}

mlir::LogicalResult materializeRVVVectorCompareSelectSourceKernel(
    mlir::OpBuilder &builder, llvm::StringRef kernelName,
    VectorCompareSelectSourceMatch source) {
  mlir::Location loc = source.func.getLoc();

  mlir::ModuleOp module = source.func->getParentOfType<mlir::ModuleOp>();
  llvm::Expected<weft::exec::TargetOp> target =
      weft::target::rvv::materializeRVVSourceTargetProfile(
          builder, module, loc, kernelName);
  if (!target)
    return failVectorCompareSelectMaterializer(
        source.func, llvm::toString(target.takeError()));

  mlir::OperationState kernelState(loc,
                                   weft::exec::KernelOp::getOperationName());
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addAttribute("target",
                           symbolRef(builder, target->getSymName()));
  kernelState.addAttribute("problem", symbolRef(builder, "canonical_problem"));
  kernelState.addRegion();
  auto kernel = llvm::cast<weft::exec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  mlir::OperationState problemState(
      loc, weft::exec::I32VectorCompareSelectProblemOp::getOperationName());
  problemState.addAttribute("sym_name",
                            builder.getStringAttr("canonical_problem"));
  problemState.addAttribute("predicate",
                            builder.getStringAttr(source.predicateKind));
  problemState.addAttribute("rhs_form", builder.getStringAttr("vector"));
  problemState.addAttribute(
      "source_vector_lanes",
      builder.getI64IntegerAttr(source.sourceVectorType.getDimSize(0)));
  (void)builder.create(problemState);

  // Source adaptation stops at exact P + target binding.
  return mlir::success();
}

mlir::LogicalResult materializeRVVVectorRuntimeScalarCompareSelectSourceKernel(
    mlir::OpBuilder &builder, llvm::StringRef kernelName,
    VectorRuntimeScalarCompareSelectSourceMatch source) {
  mlir::Location loc = source.func.getLoc();

  mlir::ModuleOp module = source.func->getParentOfType<mlir::ModuleOp>();
  llvm::Expected<weft::exec::TargetOp> target =
      weft::target::rvv::materializeRVVSourceTargetProfile(
          builder, module, loc, kernelName);
  if (!target)
    return failVectorRuntimeScalarCompareSelectMaterializer(
        source.func, llvm::toString(target.takeError()));

  mlir::OperationState kernelState(loc,
                                   weft::exec::KernelOp::getOperationName());
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addAttribute("target",
                           symbolRef(builder, target->getSymName()));
  kernelState.addAttribute("problem", symbolRef(builder, "canonical_problem"));
  kernelState.addRegion();
  auto kernel = llvm::cast<weft::exec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  mlir::OperationState problemState(
      loc, weft::exec::I32VectorCompareSelectProblemOp::getOperationName());
  problemState.addAttribute("sym_name",
                            builder.getStringAttr("canonical_problem"));
  problemState.addAttribute("predicate",
                            builder.getStringAttr(source.predicateKind));
  problemState.addAttribute("rhs_form",
                            builder.getStringAttr("runtime-scalar"));
  problemState.addAttribute(
      "source_vector_lanes",
      builder.getI64IntegerAttr(source.sourceVectorType.getDimSize(0)));
  (void)builder.create(problemState);

  // Source adaptation stops at exact P + target binding.
  return mlir::success();
}

void populateRVVVectorSourceFrontDoorDependentDialects(
    mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::vector::VectorDialect, weft::exec::WEFTExecDialect,
                  weft::rvv::WEFTRVVDialect>();
}

mlir::LogicalResult materializeRVVVectorSourceFrontDoorFamily(
    mlir::ModuleOp module,
    const RVVVectorSourceFrontDoorFamilyDescriptor &family) {
  std::string kernelName;
  mlir::OpBuilder builder(module.getContext());

  switch (family.id) {
  case RVVVectorSourceFrontDoorFamilyID::Binary: {
    mlir::FailureOr<VectorBinarySourceMatch> source =
        matchVectorBinarySourceFrontDoor(module, family, kernelName);
    if (mlir::failed(source))
      return mlir::failure();
    if (!source->func)
      return mlir::success();

    builder.setInsertionPointToStart(module.getBody());
    if (mlir::failed(materializeRVVVectorBinarySourceKernel(
            builder, kernelName, *source)))
      return mlir::failure();
    break;
  }
  case RVVVectorSourceFrontDoorFamilyID::CompareSelect: {
    mlir::FailureOr<VectorCompareSelectSourceMatch> source =
        matchVectorCompareSelectSourceFrontDoor(module, family, kernelName);
    if (mlir::failed(source))
      return mlir::failure();
    if (!source->func)
      return mlir::success();

    builder.setInsertionPointToStart(module.getBody());
    if (mlir::failed(materializeRVVVectorCompareSelectSourceKernel(
            builder, kernelName, *source)))
      return mlir::failure();
    break;
  }
  case RVVVectorSourceFrontDoorFamilyID::RuntimeScalarCompareSelect: {
    mlir::FailureOr<VectorRuntimeScalarCompareSelectSourceMatch> source =
        matchVectorRuntimeScalarCompareSelectSourceFrontDoor(module, family,
                                                            kernelName);
    if (mlir::failed(source))
      return mlir::failure();
    if (!source->func)
      return mlir::success();

    builder.setInsertionPointToStart(module.getBody());
    if (mlir::failed(materializeRVVVectorRuntimeScalarCompareSelectSourceKernel(
            builder, kernelName, *source)))
      return mlir::failure();
    break;
  }
  }

  module->removeAttr(kSourceFrontDoorAttrName);
  module->removeAttr(kSourceKernelAttrName);
  return mlir::success();
}

class MaterializeRVVVectorSourceFrontDoorFamilyPass final
    : public mlir::PassWrapper<
          MaterializeRVVVectorSourceFrontDoorFamilyPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVVectorSourceFrontDoorFamilyPass(
      RVVVectorSourceFrontDoorFamilyID familyID)
      : familyID(familyID) {}

  llvm::StringRef getArgument() const final {
    return getRVVVectorSourceFrontDoorFamily(familyID).passArgument;
  }

  llvm::StringRef getDescription() const final {
    return getRVVVectorSourceFrontDoorFamily(familyID).passDescription;
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    populateRVVVectorSourceFrontDoorDependentDialects(registry);
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    const RVVVectorSourceFrontDoorFamilyDescriptor &family =
        getRVVVectorSourceFrontDoorFamily(familyID);
    if (mlir::failed(
            materializeRVVVectorSourceFrontDoorFamily(module, family))) {
      signalPassFailure();
      return;
    }
  }

private:
  RVVVectorSourceFrontDoorFamilyID familyID;
};

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorSourceFrontDoorFamilyPass(
    RVVVectorSourceFrontDoorFamilyID familyID) {
  return std::make_unique<MaterializeRVVVectorSourceFrontDoorFamilyPass>(
      familyID);
}

} // namespace

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorBinarySourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  (void)registry;
  return createMaterializeRVVVectorSourceFrontDoorFamilyPass(
      RVVVectorSourceFrontDoorFamilyID::Binary);
}

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorCompareSelectSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  (void)registry;
  return createMaterializeRVVVectorSourceFrontDoorFamilyPass(
      RVVVectorSourceFrontDoorFamilyID::CompareSelect);
}

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorRuntimeScalarCompareSelectSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  (void)registry;
  return createMaterializeRVVVectorSourceFrontDoorFamilyPass(
      RVVVectorSourceFrontDoorFamilyID::RuntimeScalarCompareSelect);
}

llvm::Error registerRVVVectorSourceFrontDoorFamilyPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  (void)registry;
  for (const RVVVectorSourceFrontDoorFamilyDescriptor &family :
       getRVVVectorSourceFrontDoorFamilyRegistry()) {
    RVVVectorSourceFrontDoorFamilyID familyID = family.id;
    out.push_back(SourceFrontDoorPassRegistration(
        ownerPlugin, family.passArgument, family.passDescription,
        formula_catalog::kVectorSourceConstruction,
        [familyID] {
          return createMaterializeRVVVectorSourceFrontDoorFamilyPass(familyID);
        },
        family.defaultArtifactPolicy));
  }
  return llvm::Error::success();
}

void addRVVVectorSourceFormulaProductionEntries(
    FormulaDescriptor &descriptor) {
  for (const RVVVectorSourceFrontDoorFamilyDescriptor &family :
       getRVVVectorSourceFrontDoorFamilyRegistry())
    descriptor.addProductionEntry(family.passArgument);
}

} // namespace weft::plugin::rvv
