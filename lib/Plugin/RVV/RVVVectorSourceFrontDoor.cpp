#include "TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/VariantMaterialization.h"

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

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kSeedAttrName("tcrv_rvv.lowering_seed");
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("tcrv_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedVectorBinarySourceFrontDoorValue(
    "bounded_vector_source");
constexpr llvm::StringLiteral kAcceptedVectorCompareSelectSourceFrontDoorValue(
    "bounded_vector_compare_select_source");
constexpr llvm::StringLiteral
    kAcceptedVectorRuntimeScalarCompareSelectSourceFrontDoorValue(
        "bounded_vector_runtime_scalar_cmp_select_source");
constexpr llvm::StringLiteral kRVVCapabilitySymbol("rvv");
// Structural symbol the front door uses to wire @requires for the conservative
// fallback capability. The value matches the fallback-owning plugin's preferred
// capability symbol so the registry resolves the proposal's required-capability
// id onto this symbol; the front door owns the symbol name, not the identity.
constexpr llvm::StringLiteral kFallbackCapabilitySymbol("scalar_fallback");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kFallbackRoleAttrName("fallback_role");
// Abstract capability kind the front door asks the registry for. This is the
// generic "conservative fallback" role-kind in the plugin protocol, not the
// scalar plugin's identity: any plugin declaring a fallback-kind capability
// owns the conservative fallback the front door dispatches to.
constexpr llvm::StringLiteral kConservativeFallbackCapabilityKind("fallback");
constexpr llvm::StringLiteral kSourceKernelBoundaryAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kSelectedPathRoleAttrName("selected_path_role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kRVVConstructionProtocolAttrName(
    "rvv_construction_protocol");
constexpr llvm::StringLiteral kRVVEmitCRouteMappingAttrName(
    "rvv_emitc_route_mapping");
constexpr llvm::StringLiteral kRVVConstructionProtocol(
    "extension-family-construction-protocol.v1");
constexpr llvm::StringLiteral kRVVGenericTypedBodyRouteFamily(
    "rvv-generic-typed-body-emitc-route-family");

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
  llvm::StringLiteral selectedVariantPrefix;
  llvm::StringLiteral runtimePurposePrefix;
  llvm::StringLiteral dispatchPolicy;
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
       "tcrv-rvv-materialize-vector-binary-source-front-door",
       "Materialize one bounded MLIR Vector-like i32 binary source pattern "
       "into a selected generic typed RVV body",
       "RVV vector-binary source function candidate", "rvv_vector_",
       "rvv-vector-binary-source-front-door",
       "rvv-vector-binary-source-front-door-case",
       SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
           ExplicitOnly,
       failVectorMaterializer},
      {RVVVectorSourceFrontDoorFamilyID::CompareSelect,
       "bounded-vector-compare-select-source-front-door",
       kAcceptedVectorCompareSelectSourceFrontDoorValue,
       "tcrv-rvv-materialize-vector-compare-select-source-front-door",
       "Materialize one bounded MLIR Vector-like i32 compare/select source "
       "pattern into a selected generic typed RVV body",
       "RVV vector-compare-select source function candidate",
       "rvv_vector_cmp_select_",
       "rvv-vector-compare-select-source-front-door",
       "rvv-vector-compare-select-source-front-door-case",
       SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
           ExplicitOnly,
       failVectorCompareSelectMaterializer},
      {RVVVectorSourceFrontDoorFamilyID::RuntimeScalarCompareSelect,
       "bounded-vector-runtime-scalar-cmp-select-source-front-door",
       kAcceptedVectorRuntimeScalarCompareSelectSourceFrontDoorValue,
       "tcrv-rvv-materialize-vector-runtime-scalar-cmp-select-source-front-door",
       "Materialize one bounded MLIR Vector-like i32 runtime-scalar "
       "compare/select source pattern into a selected generic typed RVV body",
       "RVV vector-runtime-scalar-cmp-select source function candidate",
       "rvv_vector_runtime_scalar_cmp_select_",
       "rvv-vector-runtime-scalar-cmp-select-source-front-door",
       "rvv-vector-runtime-scalar-cmp-select-source-front-door-case",
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
    if (dialect == "tcrv" || dialect == "tcrv_rvv" ||
        dialect == "tcrv_toy" || dialect == "tcrv_tensorext_lite")
      staleOp = op;
  });
  if (!staleOp)
    return mlir::success();

  return family.fail(
      staleOp,
      "source materializer requires RVV source-only MLIR input; pre-existing "
      "tcrv.exec/tcrv_rvv/tcrv_toy/tcrv_tensorext_lite selected-boundary or "
      "variant residue is not accepted");
}

std::string getDefaultRVVVectorSourceKernelName(
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    llvm::StringRef operationKind) {
  return (llvm::Twine(family.selectedVariantPrefix) + operationKind +
          "_from_vector_source")
      .str();
}

std::string getRVVVectorSourceVariantSymbol(
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    llvm::StringRef operationKind) {
  return (llvm::Twine(family.selectedVariantPrefix) + operationKind).str();
}

std::string getRVVVectorSourceScalarFallbackVariantSymbol(
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    llvm::StringRef operationKind) {
  return (llvm::Twine(family.selectedVariantPrefix) + operationKind +
          "_scalar_fallback")
      .str();
}

std::string getRVVVectorSourceRuntimePurpose(
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    llvm::StringRef role) {
  return (llvm::Twine(family.runtimePurposePrefix) + ":" + role).str();
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
              "' rejected stale tcrv_rvv.lowering_seed metadata as RVV "
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
      llvm::Twine("unknown tcrv_rvv.source_front_door marker '") +
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

void createCapability(mlir::OpBuilder &builder, mlir::Location loc,
                      llvm::StringRef symbol, llvm::StringRef id,
                      llvm::StringRef kind) {
  mlir::OperationState state(loc, tcrv::exec::CapabilityOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(symbol));
  state.addAttribute("id", builder.getStringAttr(id));
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addAttribute("status", builder.getStringAttr("available"));
  (void)builder.create(state);
}

mlir::ArrayAttr createRequires(mlir::OpBuilder &builder,
                               llvm::StringRef symbol) {
  return builder.getArrayAttr({symbolRef(builder, symbol)});
}

tcrv::rvv::PolicyAttr createAgnosticPolicy(mlir::OpBuilder &builder) {
  return tcrv::rvv::PolicyAttr::get(builder.getContext(),
                                    tcrv::rvv::TailPolicy::Agnostic,
                                    tcrv::rvv::MaskPolicy::Agnostic);
}

tcrv::rvv::RuntimeABIValueOp
createRuntimeABIValue(mlir::OpBuilder &builder, mlir::Location loc,
                      llvm::StringRef role, llvm::StringRef cName,
                      llvm::StringRef cType, llvm::StringRef purpose,
                      mlir::Type resultType) {
  mlir::OperationState state(loc,
                             tcrv::rvv::RuntimeABIValueOp::getOperationName());
  state.addAttribute("role", builder.getStringAttr(role));
  state.addAttribute("c_name", builder.getStringAttr(cName));
  state.addAttribute("c_type", builder.getStringAttr(cType));
  state.addAttribute("ownership",
                     builder.getStringAttr("target-export-abi-owned"));
  state.addAttribute("purpose", builder.getStringAttr(purpose));
  state.addTypes(resultType);
  return llvm::cast<tcrv::rvv::RuntimeABIValueOp>(builder.create(state));
}

tcrv::rvv::SetVLOp createSetVL(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value n,
                               tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, tcrv::rvv::SetVLOp::getOperationName());
  state.addOperands(n);
  state.addAttribute("sew", builder.getI64IntegerAttr(32));
  state.addAttribute("lmul", builder.getStringAttr("m1"));
  state.addAttribute("policy", policy);
  state.addTypes(tcrv::rvv::VLType::get(builder.getContext()));
  return llvm::cast<tcrv::rvv::SetVLOp>(builder.create(state));
}

tcrv::rvv::WithVLOp createWithVL(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value vl,
                                 tcrv::rvv::PolicyAttr policy,
                                 llvm::StringRef kernelName,
                                 llvm::StringRef selectedVariantSymbol,
                                 mlir::ArrayAttr requires) {
  mlir::OperationState state(loc, tcrv::rvv::WithVLOp::getOperationName());
  state.addOperands(vl);
  state.addAttribute("sew", builder.getI64IntegerAttr(32));
  state.addAttribute("lmul", builder.getStringAttr("m1"));
  state.addAttribute("policy", policy);
  state.addAttribute(kSourceKernelBoundaryAttrName,
                     builder.getStringAttr(kernelName));
  state.addAttribute(kSelectedVariantAttrName,
                     symbolRef(builder, selectedVariantSymbol));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(getRVVExtensionPluginName()));
  state.addAttribute(kSelectedPathRoleAttrName,
                     builder.getStringAttr(
                         stringifyVariantEmissionRole(
                             VariantEmissionRole::DispatchCase)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr("selected-lowering-boundary"));
  state.addAttribute(kRequiredCapabilitiesAttrName, requires);
  state.addAttribute(kRVVConstructionProtocolAttrName,
                     builder.getStringAttr(kRVVConstructionProtocol));
  state.addAttribute(kRVVEmitCRouteMappingAttrName,
                     builder.getStringAttr(kRVVGenericTypedBodyRouteFamily));
  state.addRegion();
  auto withVL = llvm::cast<tcrv::rvv::WithVLOp>(builder.create(state));
  withVL.getBody().emplaceBlock();
  return withVL;
}

mlir::Value createRVVLoad(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value buffer, mlir::Value vl,
                          mlir::Type vectorType) {
  mlir::OperationState state(loc, tcrv::rvv::LoadOp::getOperationName());
  state.addOperands({buffer, vl});
  state.addTypes(vectorType);
  return builder.create(state)->getResult(0);
}

mlir::Value createRVVBinary(mlir::OpBuilder &builder, mlir::Location loc,
                            llvm::StringRef binaryKind, mlir::Value lhs,
                            mlir::Value rhs, mlir::Value vl,
                            mlir::Type vectorType) {
  mlir::OperationState state(loc, tcrv::rvv::BinaryOp::getOperationName());
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(binaryKind));
  state.addTypes(vectorType);
  return builder.create(state)->getResult(0);
}

mlir::Value createRVVCompare(mlir::OpBuilder &builder, mlir::Location loc,
                             llvm::StringRef predicateKind, mlir::Value lhs,
                             mlir::Value rhs, mlir::Value vl,
                             mlir::Type maskType) {
  mlir::OperationState state(loc, tcrv::rvv::CompareOp::getOperationName());
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(predicateKind));
  state.addTypes(maskType);
  return builder.create(state)->getResult(0);
}

mlir::Value createRVVSplat(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value scalar, mlir::Value vl,
                           mlir::Type vectorType) {
  mlir::OperationState state(loc, tcrv::rvv::SplatOp::getOperationName());
  state.addOperands({scalar, vl});
  state.addTypes(vectorType);
  return builder.create(state)->getResult(0);
}

mlir::Value createRVVSelect(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value mask, mlir::Value trueValue,
                            mlir::Value falseValue, mlir::Value vl,
                            mlir::Type vectorType) {
  mlir::OperationState state(loc, tcrv::rvv::SelectOp::getOperationName());
  state.addOperands({mask, trueValue, falseValue, vl});
  state.addTypes(vectorType);
  return builder.create(state)->getResult(0);
}

void createRVVStore(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value buffer, mlir::Value value, mlir::Value vl) {
  mlir::OperationState state(loc, tcrv::rvv::StoreOp::getOperationName());
  state.addOperands({buffer, value, vl});
  (void)builder.create(state);
}

tcrv::exec::VariantOp createRVVVectorSourceVariant(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef selectedVariantSymbol, mlir::ArrayAttr requires,
    tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, tcrv::exec::VariantOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(selectedVariantSymbol));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(getRVVExtensionPluginName()));
  state.addAttribute(kRequiresAttrName, requires);
  state.addAttribute("tcrv_rvv.policy", policy);
  state.addRegion();
  auto variant = llvm::cast<tcrv::exec::VariantOp>(builder.create(state));
  variant.getBody().emplaceBlock();
  return variant;
}

// Provision the conservative-fallback capability for the kernel by asking the
// registry which plugin-declared capability carries the abstract "fallback"
// kind. The front door never names the scalar plugin's capability id; it learns
// the id/kind from the owning plugin's declared capabilities and only chooses
// the structural symbol it uses to wire @requires.
mlir::LogicalResult createConservativeFallbackCapability(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Operation *failOp,
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    const ExtensionPluginRegistry &registry,
    llvm::StringRef capabilitySymbol) {
  llvm::SmallVector<PluginCapability, 1> fallbackCapabilities;
  registry.collectCapabilitiesByKind(kConservativeFallbackCapabilityKind,
                                     fallbackCapabilities);
  if (fallbackCapabilities.size() != 1)
    return family.fail(
        failOp,
        llvm::Twine("source front door requires exactly one plugin-declared "
                    "conservative-fallback capability (kind '") +
            kConservativeFallbackCapabilityKind + "') to dispatch the "
            "conservative fallback variant to its owning plugin; found " +
            llvm::Twine(fallbackCapabilities.size()));

  const PluginCapability &fallbackCapability = fallbackCapabilities.front();
  createCapability(builder, loc, capabilitySymbol, fallbackCapability.getID(),
                   fallbackCapability.getKind());
  return mlir::success();
}

// Relocate scalar-variant authorship out of this RVV front door: instead of
// hand-minting the scalar plugin's origin/role/requires, obtain the
// conservative-fallback variant by dispatching to the registry. The owning
// plugin's proposeVariants produces the proposal; the common materialization
// path authors the variant with that plugin's origin/role/requires/policy. The
// returned origin feeds the dispatch fallback case so the front door consumes
// only the abstract conservative-fallback role, never a concrete plugin name.
mlir::FailureOr<std::string> materializeConservativeFallbackVariantViaPlugin(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    mlir::Operation *highLevelOp,
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    const ExtensionPluginRegistry &registry,
    llvm::StringRef fallbackVariantSymbol) {
  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities) {
    (void)family.fail(kernel, llvm::Twine("source front door could not build a "
                                          "capability scope for kernel @") +
                                  kernel.getSymName() + ": " +
                                  llvm::toString(capabilities.takeError()));
    return mlir::failure();
  }

  VariantProposalRequest request(highLevelOp, kernel, *capabilities);
  llvm::SmallVector<VariantProposal, 4> proposals;
  if (llvm::Error error = registry.collectVariantProposals(request, proposals)) {
    (void)family.fail(kernel, llvm::Twine("source front door failed to collect "
                                          "variant proposals for kernel @") +
                                  kernel.getSymName() + ": " +
                                  llvm::toString(std::move(error)));
    return mlir::failure();
  }

  const VariantProposal *fallbackProposal = nullptr;
  for (const VariantProposal &proposal : proposals) {
    if (proposal.getFallbackRole() !=
        VariantFallbackRole::ConservativeFallback)
      continue;
    if (fallbackProposal) {
      (void)family.fail(kernel,
                        "source front door requires exactly one "
                        "conservative-fallback variant proposal; the registry "
                        "produced more than one");
      return mlir::failure();
    }
    fallbackProposal = &proposal;
  }
  if (!fallbackProposal) {
    (void)family.fail(kernel,
                      "source front door requires a conservative-fallback "
                      "variant proposal from a fallback-owning plugin; none was "
                      "produced for this kernel");
    return mlir::failure();
  }

  // Keep the per-kernel structural symbol the front door owns; the variant's
  // identity (origin/role/requires/policy) still comes from the proposal.
  VariantProposal scopedProposal = *fallbackProposal;
  scopedProposal.setVariantName(fallbackVariantSymbol);

  if (llvm::Error error = transforms::materializeVariantProposals(
          builder, request, scopedProposal)) {
    (void)family.fail(kernel, llvm::Twine("source front door failed to "
                                          "materialize the conservative "
                                          "fallback variant for kernel @") +
                                  kernel.getSymName() + ": " +
                                  llvm::toString(std::move(error)));
    return mlir::failure();
  }

  return fallbackProposal->getOriginPlugin().str();
}

void createDispatch(mlir::OpBuilder &builder, mlir::Location loc,
                    llvm::StringRef selectedVariantSymbol,
                    llvm::StringRef fallbackVariantSymbol,
                    llvm::StringRef fallbackOrigin,
                    llvm::StringRef policy) {
  mlir::OperationState dispatchState(loc,
                                     tcrv::exec::DispatchOp::getOperationName());
  dispatchState.addRegion();
  auto dispatch =
      llvm::cast<tcrv::exec::DispatchOp>(builder.create(dispatchState));
  dispatch.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&dispatch.getBody().front());

  mlir::OperationState caseState(loc,
                                 tcrv::exec::DispatchCaseOp::getOperationName());
  caseState.addAttribute("target", symbolRef(builder, selectedVariantSymbol));
  caseState.addAttribute(kOriginAttrName, builder.getStringAttr(getRVVExtensionPluginName()));
  caseState.addAttribute("policy", builder.getStringAttr(policy));
  (void)builder.create(caseState);

  mlir::OperationState fallbackState(loc,
                                     tcrv::exec::FallbackOp::getOperationName());
  fallbackState.addAttribute("target",
                             symbolRef(builder, fallbackVariantSymbol));
  fallbackState.addAttribute(kOriginAttrName,
                             builder.getStringAttr(fallbackOrigin));
  fallbackState.addAttribute(kFallbackRoleAttrName,
                             builder.getStringAttr(kConservativeFallbackRoleValue));
  (void)builder.create(fallbackState);
}

mlir::LogicalResult materializeRVVVectorBinarySourceKernel(
    mlir::OpBuilder &builder, llvm::StringRef kernelName,
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    const ExtensionPluginRegistry &registry, VectorBinarySourceMatch source) {
  mlir::Location loc = source.func.getLoc();
  tcrv::rvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol =
      getRVVVectorSourceVariantSymbol(family, source.binaryKind);
  std::string fallbackVariantSymbol =
      getRVVVectorSourceScalarFallbackVariantSymbol(family, source.binaryKind);

  mlir::OperationState kernelState(loc,
                                   tcrv::exec::KernelOp::getOperationName());
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addRegion();
  auto kernel = llvm::cast<tcrv::exec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  createCapability(builder, loc, kRVVCapabilitySymbol, "rvv", "isa-vector");
  if (mlir::failed(createConservativeFallbackCapability(
          builder, loc, source.func, family, registry,
          kFallbackCapabilitySymbol)))
    return mlir::failure();
  mlir::ArrayAttr rvvRequires = createRequires(builder, kRVVCapabilitySymbol);

  tcrv::exec::VariantOp rvvVariant = createRVVVectorSourceVariant(
      builder, loc, selectedVariantSymbol, rvvRequires, policy);
  mlir::OpBuilder::InsertionGuard variantGuard(builder);
  builder.setInsertionPointToStart(&rvvVariant.getBody().front());

  mlir::Type runtimeABIType =
      tcrv::rvv::RuntimeABIValueType::get(builder.getContext());
  std::string lhsPurpose = getRVVVectorSourceRuntimePurpose(family, "lhs");
  std::string rhsPurpose = getRVVVectorSourceRuntimePurpose(family, "rhs");
  std::string outPurpose = getRVVVectorSourceRuntimePurpose(family, "out");
  std::string nPurpose = getRVVVectorSourceRuntimePurpose(family, "n");
  auto lhs = createRuntimeABIValue(
      builder, loc, "lhs-input-buffer", "lhs", "const int32_t *", lhsPurpose,
      runtimeABIType);
  auto rhs = createRuntimeABIValue(
      builder, loc, "rhs-input-buffer", "rhs", "const int32_t *", rhsPurpose,
      runtimeABIType);
  auto out = createRuntimeABIValue(builder, loc, "output-buffer", "out",
                                   "int32_t *", outPurpose,
                                   runtimeABIType);
  auto n = createRuntimeABIValue(builder, loc, "runtime-element-count", "n",
                                 "size_t", nPurpose,
                                 builder.getIndexType());

  tcrv::rvv::SetVLOp setvl = createSetVL(builder, loc, n.getResult(), policy);
  tcrv::rvv::WithVLOp withVL =
      createWithVL(builder, loc, setvl.getVl(), policy, kernelName,
                   selectedVariantSymbol, rvvRequires);

  mlir::OpBuilder::InsertionGuard withVLGuard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());
  mlir::Type vectorType =
      tcrv::rvv::VectorType::get(builder.getContext(), builder.getI32Type(),
                                 "m1");
  mlir::Value loadedLHS =
      createRVVLoad(builder, loc, lhs.getResult(), setvl.getVl(), vectorType);
  mlir::Value loadedRHS =
      createRVVLoad(builder, loc, rhs.getResult(), setvl.getVl(), vectorType);
  mlir::Value result =
      createRVVBinary(builder, loc, source.binaryKind, loadedLHS, loadedRHS,
                      setvl.getVl(), vectorType);
  createRVVStore(builder, loc, out.getResult(), result, setvl.getVl());

  mlir::FailureOr<std::string> fallbackOrigin =
      materializeConservativeFallbackVariantViaPlugin(
          builder, kernel, source.func, family, registry, fallbackVariantSymbol);
  if (mlir::failed(fallbackOrigin))
    return mlir::failure();

  builder.setInsertionPointToEnd(&kernel.getBody().front());
  createDispatch(builder, loc, selectedVariantSymbol, fallbackVariantSymbol,
                 *fallbackOrigin, family.dispatchPolicy);
  return mlir::success();
}

mlir::LogicalResult materializeRVVVectorCompareSelectSourceKernel(
    mlir::OpBuilder &builder, llvm::StringRef kernelName,
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    const ExtensionPluginRegistry &registry,
    VectorCompareSelectSourceMatch source) {
  mlir::Location loc = source.func.getLoc();
  tcrv::rvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol =
      getRVVVectorSourceVariantSymbol(family, source.predicateKind);
  std::string fallbackVariantSymbol =
      getRVVVectorSourceScalarFallbackVariantSymbol(family,
                                                   source.predicateKind);

  mlir::OperationState kernelState(loc,
                                   tcrv::exec::KernelOp::getOperationName());
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addRegion();
  auto kernel = llvm::cast<tcrv::exec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  createCapability(builder, loc, kRVVCapabilitySymbol, "rvv", "isa-vector");
  if (mlir::failed(createConservativeFallbackCapability(
          builder, loc, source.func, family, registry,
          kFallbackCapabilitySymbol)))
    return mlir::failure();
  mlir::ArrayAttr rvvRequires = createRequires(builder, kRVVCapabilitySymbol);

  tcrv::exec::VariantOp rvvVariant = createRVVVectorSourceVariant(
      builder, loc, selectedVariantSymbol, rvvRequires, policy);
  mlir::OpBuilder::InsertionGuard variantGuard(builder);
  builder.setInsertionPointToStart(&rvvVariant.getBody().front());

  mlir::Type runtimeABIType =
      tcrv::rvv::RuntimeABIValueType::get(builder.getContext());
  std::string lhsPurpose = getRVVVectorSourceRuntimePurpose(family, "lhs");
  std::string rhsPurpose = getRVVVectorSourceRuntimePurpose(family, "rhs");
  std::string outPurpose = getRVVVectorSourceRuntimePurpose(family, "out");
  std::string nPurpose = getRVVVectorSourceRuntimePurpose(family, "n");
  auto lhs = createRuntimeABIValue(
      builder, loc, "lhs-input-buffer", "lhs", "const int32_t *", lhsPurpose,
      runtimeABIType);
  auto rhs = createRuntimeABIValue(
      builder, loc, "rhs-input-buffer", "rhs", "const int32_t *", rhsPurpose,
      runtimeABIType);
  auto out = createRuntimeABIValue(
      builder, loc, "output-buffer", "out", "int32_t *", outPurpose,
      runtimeABIType);
  auto n = createRuntimeABIValue(
      builder, loc, "runtime-element-count", "n", "size_t", nPurpose,
      builder.getIndexType());

  tcrv::rvv::SetVLOp setvl = createSetVL(builder, loc, n.getResult(), policy);
  tcrv::rvv::WithVLOp withVL =
      createWithVL(builder, loc, setvl.getVl(), policy, kernelName,
                   selectedVariantSymbol, rvvRequires);

  mlir::OpBuilder::InsertionGuard withVLGuard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());
  mlir::Type vectorType =
      tcrv::rvv::VectorType::get(builder.getContext(), builder.getI32Type(),
                                 "m1");
  mlir::Type maskType =
      tcrv::rvv::MaskType::get(builder.getContext(), builder.getI32Type(),
                               "m1");
  mlir::Value loadedLHS =
      createRVVLoad(builder, loc, lhs.getResult(), setvl.getVl(), vectorType);
  mlir::Value loadedRHS =
      createRVVLoad(builder, loc, rhs.getResult(), setvl.getVl(), vectorType);
  mlir::Value mask =
      createRVVCompare(builder, loc, source.predicateKind, loadedLHS,
                       loadedRHS, setvl.getVl(), maskType);
  mlir::Value selected =
      createRVVSelect(builder, loc, mask, loadedLHS, loadedRHS, setvl.getVl(),
                      vectorType);
  createRVVStore(builder, loc, out.getResult(), selected, setvl.getVl());

  mlir::FailureOr<std::string> fallbackOrigin =
      materializeConservativeFallbackVariantViaPlugin(
          builder, kernel, source.func, family, registry, fallbackVariantSymbol);
  if (mlir::failed(fallbackOrigin))
    return mlir::failure();

  builder.setInsertionPointToEnd(&kernel.getBody().front());
  createDispatch(builder, loc, selectedVariantSymbol, fallbackVariantSymbol,
                 *fallbackOrigin, family.dispatchPolicy);
  return mlir::success();
}

mlir::LogicalResult materializeRVVVectorRuntimeScalarCompareSelectSourceKernel(
    mlir::OpBuilder &builder, llvm::StringRef kernelName,
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    const ExtensionPluginRegistry &registry,
    VectorRuntimeScalarCompareSelectSourceMatch source) {
  mlir::Location loc = source.func.getLoc();
  tcrv::rvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol =
      getRVVVectorSourceVariantSymbol(family, source.predicateKind);
  std::string fallbackVariantSymbol =
      getRVVVectorSourceScalarFallbackVariantSymbol(family,
                                                   source.predicateKind);

  mlir::OperationState kernelState(loc,
                                   tcrv::exec::KernelOp::getOperationName());
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addRegion();
  auto kernel = llvm::cast<tcrv::exec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  createCapability(builder, loc, kRVVCapabilitySymbol, "rvv", "isa-vector");
  if (mlir::failed(createConservativeFallbackCapability(
          builder, loc, source.func, family, registry,
          kFallbackCapabilitySymbol)))
    return mlir::failure();
  mlir::ArrayAttr rvvRequires = createRequires(builder, kRVVCapabilitySymbol);

  tcrv::exec::VariantOp rvvVariant = createRVVVectorSourceVariant(
      builder, loc, selectedVariantSymbol, rvvRequires, policy);
  mlir::OpBuilder::InsertionGuard variantGuard(builder);
  builder.setInsertionPointToStart(&rvvVariant.getBody().front());

  mlir::Type runtimeABIType =
      tcrv::rvv::RuntimeABIValueType::get(builder.getContext());
  std::string lhsPurpose = getRVVVectorSourceRuntimePurpose(family, "lhs");
  std::string rhsScalarPurpose =
      getRVVVectorSourceRuntimePurpose(family, "rhs_scalar");
  std::string trueValuePurpose =
      getRVVVectorSourceRuntimePurpose(family, "true_value");
  std::string falseValuePurpose =
      getRVVVectorSourceRuntimePurpose(family, "false_value");
  std::string outPurpose = getRVVVectorSourceRuntimePurpose(family, "out");
  std::string nPurpose = getRVVVectorSourceRuntimePurpose(family, "n");
  auto lhs = createRuntimeABIValue(
      builder, loc, "lhs-input-buffer", "lhs", "const int32_t *", lhsPurpose,
      runtimeABIType);
  auto rhsScalar =
      createRuntimeABIValue(builder, loc, "rhs-scalar-value", "rhs_scalar",
                            "int32_t", rhsScalarPurpose, builder.getI32Type());
  auto trueValue = createRuntimeABIValue(
      builder, loc, "true-value-input-buffer", "true_value",
      "const int32_t *", trueValuePurpose, runtimeABIType);
  auto falseValue = createRuntimeABIValue(
      builder, loc, "false-value-input-buffer", "false_value",
      "const int32_t *", falseValuePurpose, runtimeABIType);
  auto out = createRuntimeABIValue(
      builder, loc, "output-buffer", "out", "int32_t *", outPurpose,
      runtimeABIType);
  auto n = createRuntimeABIValue(
      builder, loc, "runtime-element-count", "n", "size_t", nPurpose,
      builder.getIndexType());

  tcrv::rvv::SetVLOp setvl = createSetVL(builder, loc, n.getResult(), policy);
  tcrv::rvv::WithVLOp withVL =
      createWithVL(builder, loc, setvl.getVl(), policy, kernelName,
                   selectedVariantSymbol, rvvRequires);

  mlir::OpBuilder::InsertionGuard withVLGuard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());
  mlir::Type vectorType =
      tcrv::rvv::VectorType::get(builder.getContext(), builder.getI32Type(),
                                 "m1");
  mlir::Type maskType =
      tcrv::rvv::MaskType::get(builder.getContext(), builder.getI32Type(),
                               "m1");
  mlir::Value loadedLHS =
      createRVVLoad(builder, loc, lhs.getResult(), setvl.getVl(), vectorType);
  mlir::Value splattedRHS = createRVVSplat(
      builder, loc, rhsScalar.getResult(), setvl.getVl(), vectorType);
  mlir::Value loadedTrueValue = createRVVLoad(
      builder, loc, trueValue.getResult(), setvl.getVl(), vectorType);
  mlir::Value loadedFalseValue = createRVVLoad(
      builder, loc, falseValue.getResult(), setvl.getVl(), vectorType);
  mlir::Value mask =
      createRVVCompare(builder, loc, source.predicateKind, loadedLHS,
                       splattedRHS, setvl.getVl(), maskType);
  mlir::Value selected =
      createRVVSelect(builder, loc, mask, loadedTrueValue, loadedFalseValue,
                      setvl.getVl(), vectorType);
  createRVVStore(builder, loc, out.getResult(), selected, setvl.getVl());

  mlir::FailureOr<std::string> fallbackOrigin =
      materializeConservativeFallbackVariantViaPlugin(
          builder, kernel, source.func, family, registry, fallbackVariantSymbol);
  if (mlir::failed(fallbackOrigin))
    return mlir::failure();

  builder.setInsertionPointToEnd(&kernel.getBody().front());
  createDispatch(builder, loc, selectedVariantSymbol, fallbackVariantSymbol,
                 *fallbackOrigin, family.dispatchPolicy);
  return mlir::success();
}

void populateRVVVectorSourceFrontDoorDependentDialects(
    mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::vector::VectorDialect, tcrv::exec::TCRVExecDialect,
                  tcrv::rvv::TCRVRVVDialect>();
}

mlir::LogicalResult materializeRVVVectorSourceFrontDoorFamily(
    mlir::ModuleOp module,
    const RVVVectorSourceFrontDoorFamilyDescriptor &family,
    const ExtensionPluginRegistry &registry) {
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
            builder, kernelName, family, registry, *source)))
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
            builder, kernelName, family, registry, *source)))
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
            builder, kernelName, family, registry, *source)))
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
      RVVVectorSourceFrontDoorFamilyID familyID,
      const ExtensionPluginRegistry *registry)
      : familyID(familyID), registry(registry) {}

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
    if (!registry) {
      module.emitError()
          << "RVV vector source-front-door family pass requires an injected "
             "extension-plugin registry to dispatch the conservative fallback "
             "variant to its owning plugin";
      signalPassFailure();
      return;
    }
    if (mlir::failed(materializeRVVVectorSourceFrontDoorFamily(module, family,
                                                              *registry))) {
      signalPassFailure();
      return;
    }
  }

private:
  RVVVectorSourceFrontDoorFamilyID familyID;
  const ExtensionPluginRegistry *registry = nullptr;
};

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorSourceFrontDoorFamilyPass(
    RVVVectorSourceFrontDoorFamilyID familyID,
    const ExtensionPluginRegistry *registry) {
  return std::make_unique<MaterializeRVVVectorSourceFrontDoorFamilyPass>(
      familyID, registry);
}

} // namespace

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorBinarySourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return createMaterializeRVVVectorSourceFrontDoorFamilyPass(
      RVVVectorSourceFrontDoorFamilyID::Binary, &registry);
}

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorCompareSelectSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return createMaterializeRVVVectorSourceFrontDoorFamilyPass(
      RVVVectorSourceFrontDoorFamilyID::CompareSelect, &registry);
}

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorRuntimeScalarCompareSelectSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return createMaterializeRVVVectorSourceFrontDoorFamilyPass(
      RVVVectorSourceFrontDoorFamilyID::RuntimeScalarCompareSelect, &registry);
}

llvm::Error registerRVVVectorSourceFrontDoorFamilyPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  const ExtensionPluginRegistry *registryPtr = &registry;
  for (const RVVVectorSourceFrontDoorFamilyDescriptor &family :
       getRVVVectorSourceFrontDoorFamilyRegistry()) {
    RVVVectorSourceFrontDoorFamilyID familyID = family.id;
    out.push_back(SourceFrontDoorPassRegistration(
        ownerPlugin, family.passArgument, family.passDescription,
        [familyID, registryPtr] {
          return createMaterializeRVVVectorSourceFrontDoorFamilyPass(
              familyID, registryPtr);
        },
        family.defaultArtifactPolicy));
  }
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
