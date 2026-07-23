//===- RVVMonolithicBlockDotSourceFrontDoor.cpp -------------------------===//
//
// Source-only adapter for the ggml/llama.cpp block-dot family. It recognizes
// one bounded operator identity and creates target/profile + kernel + exact
// QuantizedBlockDotProblemOp. Candidate creation, schedule selection and RVV
// typed-body construction live in Construction/RVVQuantizedBlockDotProblemConstruction.cpp.
//
//===----------------------------------------------------------------------===//

#include "Weft/Plugin/RVV/RVVMonolithicBlockDotSourceFrontDoor.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVCapabilityProfile.h"
#include "Weft/Plugin/RVV/RVVExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVFormulaCatalog.h"
#include "Weft/Plugin/RVV/RVVMonolithicBlockDotFamily.h"
#include "Weft/Target/RVV/RVVTargetProfileBinding.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace weft::plugin::rvv {
namespace {

namespace weftexec = ::weft::exec;
namespace weftrvv = ::weft::rvv;

constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "weft_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("weft_rvv.source_kernel");
constexpr llvm::StringLiteral kSeedAttrName("weft_rvv.lowering_seed");
constexpr llvm::StringLiteral kPassDescription(
    "Adapt one bounded ggml vec_dot operator identity into target-bound exact "
    "QuantizedBlockDot canonical P");

mlir::LogicalResult fail(const MonolithicBlockDotOpEntry &entry,
                         mlir::Operation *op, llvm::Twine message) {
  op->emitError() << entry.failPrefix << message;
  return mlir::failure();
}

struct BlockDotSourceMatch {
  mlir::func::FuncOp func;
};

bool isRank1MemRef(mlir::Type type, unsigned bitwidth) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  return memref && memref.getRank() == 1 &&
         memref.getElementType().isInteger(bitwidth);
}

bool isRank1F32MemRef(mlir::Type type) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.getElementType().isF32();
}

mlir::FailureOr<BlockDotSourceMatch>
matchBlockDotSourceFunc(const MonolithicBlockDotOpEntry &entry,
                        mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(entry, func,
                "source function must have a body for the block-dot intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0)
    return fail(entry, func,
                "source function must have exactly four inputs and no results: "
                "out memref<?xf32>, n index, weight memref<?xi8>, activation "
                "memref<?xi8> (the ggml vec_dot operator identity)");
  if (!isRank1F32MemRef(type.getInput(0)) || !type.getInput(1).isIndex() ||
      !isRank1MemRef(type.getInput(2), 8) ||
      !isRank1MemRef(type.getInput(3), 8))
    return fail(entry, func,
                "source function inputs must be out rank-1 f32 memref, n index, "
                "weight rank-1 i8 memref, activation rank-1 i8 memref");
  return BlockDotSourceMatch{func};
}

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

mlir::LogicalResult materializeCanonicalProblem(
    mlir::OpBuilder &builder, llvm::StringRef kernelName,
    const MonolithicBlockDotOpEntry &entry, BlockDotSourceMatch source,
    llvm::StringRef march, llvm::StringRef isaVectorHints,
    bool numericsReassocOk) {
  auto requiredProblemFact = [&](llvm::StringRef name)
      -> std::optional<std::int64_t> {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    return std::nullopt;
  };
  std::optional<std::int64_t> qk = requiredProblemFact("qk");
  std::optional<std::int64_t> weightStride =
      requiredProblemFact("weight_block_stride");
  std::optional<std::int64_t> activationStride =
      requiredProblemFact("activation_block_stride");
  if (!qk || !weightStride || !activationStride)
    return fail(entry, source.func,
                "formula row lacks required canonical problem block geometry");

  mlir::Location loc = source.func.getLoc();
  mlir::ModuleOp module = source.func->getParentOfType<mlir::ModuleOp>();
  llvm::Expected<weftexec::TargetOp> target =
      weft::target::rvv::materializeRVVSourceTargetProfile(
          builder, module, loc, kernelName, march, isaVectorHints);
  if (!target)
    return fail(entry, source.func, llvm::toString(target.takeError()));

  mlir::OperationState kernelState(loc,
                                   weftexec::KernelOp::getOperationName());
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addAttribute("target", symbolRef(builder, target->getSymName()));
  kernelState.addAttribute("problem", symbolRef(builder, "canonical_problem"));
  kernelState.addRegion();
  auto kernel = llvm::cast<weftexec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());
  mlir::OperationState problemState(
      loc, weftexec::QuantizedBlockDotProblemOp::getOperationName());
  problemState.addAttribute("sym_name",
                            builder.getStringAttr("canonical_problem"));
  problemState.addAttribute(
      "weight_encoding",
      builder.getStringAttr(getMonolithicBlockDotProblemWeightEncoding(entry)));
  problemState.addAttribute(
      "activation_encoding",
      builder.getStringAttr(
          getMonolithicBlockDotProblemActivationEncoding(entry)));
  problemState.addAttribute("topology", builder.getStringAttr(entry.scaleModel));
  problemState.addAttribute("qk", builder.getI64IntegerAttr(*qk));
  problemState.addAttribute("weight_block_stride",
                            builder.getI64IntegerAttr(*weightStride));
  problemState.addAttribute("activation_block_stride",
                            builder.getI64IntegerAttr(*activationStride));
  problemState.addAttribute("numerics_reassoc_ok",
                            builder.getBoolAttr(numericsReassocOk));
  (void)builder.create(problemState);
  return mlir::success();
}

bool hasStaleRVVLoweringSeedMetadata(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *op) {
    if (!found)
      found = op->hasAttr(kSeedAttrName);
  });
  return found;
}

mlir::LogicalResult
requireRVVSourceOnlyModule(const MonolithicBlockDotOpEntry &entry,
                           mlir::ModuleOp module) {
  mlir::Operation *staleOp = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (staleOp || op == module.getOperation())
      return;
    llvm::StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "weft" || dialect == "weft_rvv" || dialect == "weft_toy" ||
        dialect == "weft_tensorext_lite")
      staleOp = op;
  });
  if (!staleOp)
    return mlir::success();
  return fail(entry, staleOp,
              "source materializer requires RVV source-only MLIR input; "
              "pre-existing selected-boundary or variant residue is not accepted");
}

std::string getKernelName(const MonolithicBlockDotOpEntry &entry,
                          mlir::ModuleOp module) {
  auto kernelNameAttr =
      module->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  if (kernelNameAttr && !kernelNameAttr.getValue().trim().empty())
    return kernelNameAttr.getValue().trim().str();
  return entry.kernelDefault.str();
}

class MaterializeRVVMonolithicBlockDotSourceFrontDoorPass final
    : public mlir::PassWrapper<
          MaterializeRVVMonolithicBlockDotSourceFrontDoorPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  explicit MaterializeRVVMonolithicBlockDotSourceFrontDoorPass(
      const MonolithicBlockDotOpEntry *entry)
      : entry(entry) {}

  MaterializeRVVMonolithicBlockDotSourceFrontDoorPass(
      const MaterializeRVVMonolithicBlockDotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVMonolithicBlockDotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        entry(other.entry) {}

  llvm::StringRef getArgument() const final { return entry->passArgument; }
  llvm::StringRef getDescription() const final { return kPassDescription; }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::vector::VectorDialect, weftexec::WEFTExecDialect,
                    weftrvv::WEFTRVVDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    auto marker =
        module->getAttrOfType<mlir::StringAttr>(kSourceFrontDoorAttrName);
    if (!marker || marker.getValue().trim() != entry->markerValue)
      return;

    if (hasStaleRVVLoweringSeedMetadata(module)) {
      (void)fail(*entry, module,
                 "rejected stale weft_rvv.lowering_seed metadata as RVV "
                 "source-route authority");
      signalPassFailure();
      return;
    }
    if (mlir::failed(requireRVVSourceOnlyModule(*entry, module))) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<mlir::func::FuncOp, 2> funcs;
    module.walk([&](mlir::func::FuncOp func) { funcs.push_back(func); });
    if (funcs.size() != 1) {
      (void)fail(*entry, module,
                 "source module must contain exactly one RVV block-dot source "
                 "function candidate");
      signalPassFailure();
      return;
    }
    mlir::FailureOr<BlockDotSourceMatch> source =
        matchBlockDotSourceFunc(*entry, funcs.front());
    if (mlir::failed(source)) {
      signalPassFailure();
      return;
    }

    std::string kernelName = getKernelName(*entry, module);
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    if (mlir::failed(materializeCanonicalProblem(
            builder, kernelName, *entry, *source, march, isaVectorHints,
            numericsReassocOk))) {
      signalPassFailure();
      return;
    }
    (void)materializeRVVProviderCapabilityAxes(module, march, isaVectorHints);
    module->removeAttr(kSourceFrontDoorAttrName);
    module->removeAttr(kSourceKernelAttrName);
  }

private:
  const MonolithicBlockDotOpEntry *entry = nullptr;
  Pass::Option<std::string> march{
      *this, "march",
      llvm::cl::desc("RISC-V -march used to populate the target-bound RVV "
                     "capability profile consumed by downstream construction."),
      llvm::cl::init("")};
  Pass::Option<std::string> isaVectorHints{
      *this, "isa-vector-hints",
      llvm::cl::desc("Optional probed ISA/vector hints for the target profile."),
      llvm::cl::init("")};
  Pass::Option<bool> numericsReassocOk{
      *this, "numerics-reassoc-ok",
      llvm::cl::desc("Record numerics.reassoc_ok in exact P without selecting a "
                     "numerics tier."),
      llvm::cl::init(false)};
};

std::unique_ptr<::mlir::Pass>
createMaterializeRVVMonolithicBlockDotSourceFrontDoorPass(
    const MonolithicBlockDotOpEntry *entry) {
  return std::make_unique<MaterializeRVVMonolithicBlockDotSourceFrontDoorPass>(
      entry);
}

} // namespace

llvm::Error registerRVVMonolithicBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  (void)registry;
  for (const MonolithicBlockDotOpEntry &entry : monolithicBlockDotOpTable()) {
    const MonolithicBlockDotOpEntry *entryPtr = &entry;
    out.push_back(SourceFrontDoorPassRegistration(
        ownerPlugin, entry.passArgument, kPassDescription,
        formula_catalog::kMonolithicBlockDotConstruction,
        [entryPtr] {
          return createMaterializeRVVMonolithicBlockDotSourceFrontDoorPass(
              entryPtr);
        },
        SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
            ExplicitOnly));
  }
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
