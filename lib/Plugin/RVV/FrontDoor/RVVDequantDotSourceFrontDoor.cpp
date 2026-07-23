//===- RVVDequantDotSourceFrontDoor.cpp ----------------------------------===//
//
// Track B auto-lowering, the DEQUANT rung (second auto-lowered block, one step
// ABOVE the bare-dot MVP RVVReductionSourceFrontDoor): the COMPILER auto-
// constructs the weft_rvv RVV-dialect body for a GENERIC vector-dialect signed
// widening int8 dot-reduce WITH a runtime-f32-scale dequant tail, instead of a
// per-kernel hand emitter. The integer-core LMUL anchor is the typed result of
// RVVSourceScheduleFormula over the same source/capability facts as the MVP, so
// the SAME generic source
// emits an e8m2/i16m4-form body at VLEN128 and an e8m1/i16m2-form body at
// VLEN256, NOW with the i32->f32 dequant fused in before the f32 store.
//
// What is auto-generated (vs the MVP): the construction of the
// load/widening_product/standalone_reduce/DEQUANTIZE/store op STRUCTURE -- the
// SAME body the MVP builds PLUS one weft_rvv.dequantize step. The unchanged
// RVVToEmitC emitter (its existing isLowPrecisionDequantBody sink) consumes the
// body verbatim. The novelty claimed is narrow and real: the auto-lowering path
// SCALES from a bare integer dot-reduce to a dot+dequant contraction (the q8_0-
// style integer core + ONE runtime f32 scale), still capability-fact-driven,
// still from a generic vector source with no per-kernel emitter.
//
// HONEST SCOPE: this covers the single-runtime-f32-scale signed-i8 dot dequant
// contraction (acc + sum(i8*i8), then scale*(...)). It is NOT the per-block
// fp16-scale ggml q8_0_q8_0 block-dot KERNEL (no nb = n / QK block loop, no per-
// block d_x . d_y reads). It is the second auto-lowered block, proving the path
// scales to dequant; the per-block fp16-scale loop is a later rung.
//
//===----------------------------------------------------------------------===//

#include "Weft/Plugin/RVV/RVVDequantDotSourceFrontDoor.h"
#include "Weft/Plugin/RVV/RVVFormulaCatalog.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVCapabilityProfile.h"
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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace weft::plugin::rvv {
namespace {

namespace weftexec = ::weft::exec;
namespace weftrvv = ::weft::rvv;

// The DISTINCT marker the source module carries to route to THIS dequant front
// door (NOT the MVP's "bounded_widening_dot_reduce_source"). Each pass checks
// its own marker and early-returns on a mismatch, so a dequant-marked module
// never reaches the MVP matcher (which requires exactly 5 inputs and would fail
// closed on the 6-arg dequant function) and vice versa -- the two passes are
// mutually exclusive and the MVP lit is byte-unchanged.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "weft_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("weft_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "bounded_widening_dot_reduce_dequantize_source");
constexpr llvm::StringLiteral kSeedAttrName("weft_rvv.lowering_seed");

// The single bounded contraction this block auto-lowers: the K=32 signed int8
// dot-reduce dequantized by ONE runtime f32 scale (the q8_0-style integer core +
// one scale). K=32 IS the schedule descriptor blockLen; the m2->m1 anchor flip
// with VLEN is reduction-count driven against this fixed block.
constexpr std::int64_t kContractionBlockLen = 32;

mlir::LogicalResult fail(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "bounded RVV widening-dot-reduce dequantize source front "
                     "door failed: "
                  << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the generic vector-dialect widening int8 dot-reduce + dequant.
//===----------------------------------------------------------------------===//

struct WideningDotReduceDequantSourceMatch {
  mlir::func::FuncOp func;
  mlir::Value lhsSource;
  mlir::Value rhsSource;
  mlir::Value outDestination;
  mlir::Value accSource;
  mlir::Value scaleSource;
  mlir::Value runtimeN;
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

bool isStaticBlockVector(mlir::VectorType type, unsigned bitwidth) {
  return type && type.getRank() == 1 && !type.isScalable() &&
         type.getDimSize(0) == kContractionBlockLen &&
         type.getElementType().isInteger(bitwidth);
}

bool hasNoTransferMask(mlir::vector::TransferReadOp read) {
  return !static_cast<bool>(read.getMask());
}

bool hasOneIndex(mlir::Operation::operand_range indices) {
  return llvm::range_size(indices) == 1 &&
         (*indices.begin()).getType().isIndex();
}

// Match exactly:
//   func(%lhs: memref<?xi8>, %rhs: memref<?xi8>, %out: memref<?xf32>,
//        %acc: memref<?xi32>, %scale: f32, %n: index) {
//     %a  = vector.transfer_read %lhs[0]              : vector<32xi8>
//     %b  = vector.transfer_read %rhs[0]              : vector<32xi8>
//     %ae = arith.extsi %a : vector<32xi8> to vector<32xi32>
//     %be = arith.extsi %b : vector<32xi8> to vector<32xi32>
//     %p  = arith.muli %ae, %be                        : vector<32xi32>
//     %s  = <acc[0] scalar seed>
//     %r  = vector.multi_reduction <add>, %p, %s [0]   : vector<32xi32> to i32
//     %f  = arith.sitofp %r : i32 to f32
//     %sc = arith.mulf %f, %scale : f32
//     <store %sc to out[0]>
//   }
// The dequant tail (sitofp + mulf %scale + f32 store) is the ONLY structural
// addition over the MVP bare-dot matcher: it encodes "scale*(acc + dot)", the
// codebase's own low-precision dequant body convention (the seed is the i32
// accumulator carry; the scale multiplies the reduced i32-to-f32).
mlir::FailureOr<WideningDotReduceDequantSourceMatch>
matchBoundedWideningDotReduceDequantSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func,
                "source function must have a body for structural pattern match");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 6 || type.getNumResults() != 0)
    return fail(func, "source function must have exactly six inputs and no "
                      "results: lhs/rhs memref<?xi8>, out memref<?xf32>, acc "
                      "memref<?xi32>, scale f32, plus n index");
  if (!isRank1MemRef(type.getInput(0), 8) ||
      !isRank1MemRef(type.getInput(1), 8) ||
      !isRank1F32MemRef(type.getInput(2)) ||
      !isRank1MemRef(type.getInput(3), 32) || !type.getInput(4).isF32() ||
      !type.getInput(5).isIndex())
    return fail(func, "source function inputs must be lhs/rhs rank-1 i8 "
                      "memrefs, out rank-1 f32 memref, acc rank-1 i32 memref, "
                      "one runtime f32 scale, and one runtime n index");

  // The bounded dequant body: two vector.transfer_read i8 loads + the widening
  // extsi/muli/multi_reduction chain + a scalar memref.load acc seed + the
  // i32->f32 sitofp + the *scale mulf + the scalar f32 memref.store result.
  llvm::SmallVector<mlir::vector::TransferReadOp, 2> reads;
  llvm::SmallVector<mlir::memref::StoreOp, 1> stores;
  llvm::SmallVector<mlir::memref::LoadOp, 1> seedLoads;
  llvm::SmallVector<mlir::vector::MultiDimReductionOp, 1> reductions;
  llvm::SmallVector<mlir::arith::ExtSIOp, 2> extends;
  llvm::SmallVector<mlir::arith::MulIOp, 1> products;
  llvm::SmallVector<mlir::arith::SIToFPOp, 1> conversions;
  llvm::SmallVector<mlir::arith::MulFOp, 1> scaleMuls;
  mlir::Operation *unsupportedVectorOp = nullptr;
  mlir::Operation *unsupportedArithVectorOp = nullptr;
  func.walk([&](mlir::Operation *op) {
    if (auto read = llvm::dyn_cast<mlir::vector::TransferReadOp>(op)) {
      reads.push_back(read);
      return;
    }
    if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(op)) {
      stores.push_back(store);
      return;
    }
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(op)) {
      seedLoads.push_back(load);
      return;
    }
    if (auto reduce = llvm::dyn_cast<mlir::vector::MultiDimReductionOp>(op)) {
      reductions.push_back(reduce);
      return;
    }
    if (auto ext = llvm::dyn_cast<mlir::arith::ExtSIOp>(op)) {
      if (llvm::isa<mlir::VectorType>(ext.getType()))
        extends.push_back(ext);
      return;
    }
    if (auto mul = llvm::dyn_cast<mlir::arith::MulIOp>(op)) {
      if (llvm::isa<mlir::VectorType>(mul.getType()))
        products.push_back(mul);
      return;
    }
    if (auto conv = llvm::dyn_cast<mlir::arith::SIToFPOp>(op)) {
      conversions.push_back(conv);
      return;
    }
    if (auto mulf = llvm::dyn_cast<mlir::arith::MulFOp>(op)) {
      scaleMuls.push_back(mulf);
      return;
    }
    if (op->getName().getDialectNamespace() == "vector" && !unsupportedVectorOp)
      unsupportedVectorOp = op;
    if (op->getName().getDialectNamespace() == "arith" &&
        op->getNumResults() == 1 &&
        llvm::isa<mlir::VectorType>(op->getResult(0).getType()) &&
        !unsupportedArithVectorOp)
      unsupportedArithVectorOp = op;
  });
  if (unsupportedVectorOp)
    return fail(unsupportedVectorOp,
                "only vector.transfer_read and vector.multi_reduction are "
                "supported by this bounded widening-dot-reduce dequant source "
                "pattern");
  if (unsupportedArithVectorOp)
    return fail(unsupportedArithVectorOp,
                "only arith.extsi and arith.muli vector ops are supported by "
                "this bounded widening-dot-reduce dequant source path");
  if (reads.size() != 2 || stores.size() != 1 || seedLoads.size() != 1 ||
      reductions.size() != 1 || extends.size() != 2 || products.size() != 1 ||
      conversions.size() != 1 || scaleMuls.size() != 1)
    return fail(func,
                "source pattern must contain exactly two vector.transfer_read, "
                "two arith.extsi, one arith.muli, one vector.multi_reduction, "
                "one scalar memref.load acc seed, one arith.sitofp i32->f32, "
                "one arith.mulf scale, and one scalar f32 memref.store result");

  mlir::vector::MultiDimReductionOp reduction = reductions.front();
  if (reduction.getKind() != mlir::vector::CombiningKind::ADD)
    return fail(reduction, "only the <add> multi_reduction combine is "
                           "supported by this bounded source path");

  mlir::arith::MulIOp product = products.front();
  auto lhsExt = product.getLhs().getDefiningOp<mlir::arith::ExtSIOp>();
  auto rhsExt = product.getRhs().getDefiningOp<mlir::arith::ExtSIOp>();
  if (!lhsExt || !rhsExt || lhsExt == rhsExt)
    return fail(product, "arith.muli operands must be the two arith.extsi "
                         "widened source vectors");

  auto lhsRead = lhsExt.getIn().getDefiningOp<mlir::vector::TransferReadOp>();
  auto rhsRead = rhsExt.getIn().getDefiningOp<mlir::vector::TransferReadOp>();
  if (!lhsRead || !rhsRead || lhsRead == rhsRead)
    return fail(product, "the two arith.extsi must widen the two source "
                         "vector.transfer_read results");

  mlir::memref::StoreOp store = stores.front();
  mlir::memref::LoadOp seedLoad = seedLoads.front();
  mlir::arith::SIToFPOp conversion = conversions.front();
  mlir::arith::MulFOp scaleMul = scaleMuls.front();
  mlir::Block &entry = func.getBody().front();
  if (entry.getNumArguments() != 6 ||
      lhsRead.getSource() != entry.getArgument(0) ||
      rhsRead.getSource() != entry.getArgument(1) ||
      store.getMemRef() != entry.getArgument(2) ||
      seedLoad.getMemRef() != entry.getArgument(3))
    return fail(func, "source memory roles must be structural: transfer_read "
                      "lhs/rhs from the first two arguments, the scalar f32 "
                      "store to the third (out) argument, and the acc seed load "
                      "from the fourth (acc) argument");

  // The widened i8->i32 source/product vectors.
  auto i8VecType =
      llvm::dyn_cast<mlir::VectorType>(lhsRead.getVector().getType());
  auto i32VecType = llvm::dyn_cast<mlir::VectorType>(product.getType());
  if (!isStaticBlockVector(i8VecType, 8) ||
      lhsRead.getVector().getType() != rhsRead.getVector().getType())
    return fail(lhsRead, llvm::Twine("source transfer_read must yield a static "
                                     "vector<") +
                             llvm::Twine(kContractionBlockLen) + "xi8>");
  if (!isStaticBlockVector(i32VecType, 32) ||
      lhsExt.getType() != i32VecType || rhsExt.getType() != i32VecType ||
      reduction.getSourceVectorType() != i32VecType)
    return fail(product, llvm::Twine("widened product must be a static "
                                     "vector<") +
                             llvm::Twine(kContractionBlockLen) + "xi32>");

  if (product.getResult() != reduction.getSource())
    return fail(reduction,
                "vector.multi_reduction must reduce the arith.muli product");
  if (!reduction.getResult().getType().isInteger(32))
    return fail(reduction, "multi_reduction must reduce to a scalar i32");

  // The reduction accumulator seed must be the acc[0] memref.load result.
  if (reduction.getAcc() != seedLoad.getResult())
    return fail(reduction, "the multi_reduction acc seed must be the acc[0] "
                           "memref.load result");

  // The DEQUANT tail: the reduced i32 is sitofp'd to f32, multiplied by the
  // %scale by-value argument, and the f32 result is stored to out[0].
  if (conversion.getIn() != reduction.getResult())
    return fail(conversion, "the arith.sitofp must convert the "
                            "multi_reduction i32 result");
  if (!conversion.getType().isF32())
    return fail(conversion, "the arith.sitofp must produce f32");
  mlir::Value scaleArg = entry.getArgument(4);
  bool scaleOnLhs = scaleMul.getLhs() == conversion.getResult() &&
                    scaleMul.getRhs() == scaleArg;
  bool scaleOnRhs = scaleMul.getRhs() == conversion.getResult() &&
                    scaleMul.getLhs() == scaleArg;
  if (!scaleOnLhs && !scaleOnRhs)
    return fail(scaleMul, "the arith.mulf must multiply the sitofp f32 result "
                          "by the runtime %scale by-value argument");
  if (!scaleMul.getType().isF32())
    return fail(scaleMul, "the arith.mulf scale product must be f32");
  if (store.getValueToStore() != scaleMul.getResult())
    return fail(store, "the memref.store must write the scaled f32 result to "
                       "out[0]");
  if (!hasNoTransferMask(lhsRead) || !hasNoTransferMask(rhsRead))
    return fail(func, "masked vector transfers are outside this bounded source "
                      "path");
  if (!hasOneIndex(lhsRead.getIndices()) || !hasOneIndex(rhsRead.getIndices()))
    return fail(func, "source vector transfers must be rank-1 unit-stride "
                      "memory accesses");

  return WideningDotReduceDequantSourceMatch{func,
                                             entry.getArgument(0),
                                             entry.getArgument(1),
                                             entry.getArgument(2),
                                             entry.getArgument(3),
                                             entry.getArgument(4),
                                             entry.getArgument(5)};
}

//===----------------------------------------------------------------------===//
// (2) Shared typed RVV body-building primitives.
//===----------------------------------------------------------------------===//

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

mlir::LogicalResult materializeCanonicalProblem(
    mlir::OpBuilder &builder, llvm::StringRef kernelName,
    WideningDotReduceDequantSourceMatch source, llvm::StringRef march,
    llvm::StringRef isaVectorHints) {
  mlir::Location loc = source.func.getLoc();
  mlir::ModuleOp module = source.func->getParentOfType<mlir::ModuleOp>();
  llvm::Expected<weftexec::TargetOp> target =
      weft::target::rvv::materializeRVVSourceTargetProfile(
          builder, module, loc, kernelName, march, isaVectorHints);
  if (!target)
    return fail(source.func, llvm::toString(target.takeError()));
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
      loc, weftexec::I8WideningDotReduceProblemOp::getOperationName());
  problemState.addAttribute("sym_name",
                            builder.getStringAttr("canonical_problem"));
  problemState.addAttribute("block_length", builder.getI64IntegerAttr(32));
  problemState.addAttribute("dequantize_to_f32", builder.getBoolAttr(true));
  (void)builder.create(problemState);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// (4) The pass: marker-gated, march-option-driven.
//===----------------------------------------------------------------------===//

bool hasStaleRVVLoweringSeedMetadata(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *op) {
    if (found)
      return;
    found = op->hasAttr(kSeedAttrName);
  });
  return found;
}

mlir::LogicalResult requireRVVSourceOnlyModule(mlir::ModuleOp module) {
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
  return fail(staleOp,
              "source materializer requires RVV source-only MLIR input; "
              "pre-existing selected-boundary or variant residue is not "
              "accepted");
}

std::string getKernelName(mlir::ModuleOp module) {
  auto kernelNameAttr =
      module->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  if (kernelNameAttr && !kernelNameAttr.getValue().trim().empty())
    return kernelNameAttr.getValue().trim().str();
  return "rvv_widening_dot_reduce_dequantize_i8_from_vector_source";
}

class MaterializeRVVDequantDotSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVDequantDotSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVDequantDotSourceFrontDoorPass() = default;
  MaterializeRVVDequantDotSourceFrontDoorPass(
      const MaterializeRVVDequantDotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVDequantDotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other) {}

  llvm::StringRef getArgument() const final {
    return "weft-rvv-materialize-widening-dot-reduce-dequantize-source-front-"
           "door";
  }
  llvm::StringRef getDescription() const final {
    return "Adapt one bounded widening-dot/dequant source into target-bound "
           "I8WideningDotReduce canonical P";
  }

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
    if (!marker || marker.getValue().trim() != kAcceptedMarkerValue)
      return; // not our marker: leave the module untouched.

    if (hasStaleRVVLoweringSeedMetadata(module)) {
      (void)fail(module, "rejected stale weft_rvv.lowering_seed metadata as RVV "
                         "source-route authority");
      signalPassFailure();
      return;
    }
    if (mlir::failed(requireRVVSourceOnlyModule(module))) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<mlir::func::FuncOp, 2> funcs;
    module.walk([&](mlir::func::FuncOp func) { funcs.push_back(func); });
    if (funcs.size() != 1) {
      (void)fail(module, "source module must contain exactly one RVV "
                         "widening-dot-reduce dequantize source function "
                         "candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<WideningDotReduceDequantSourceMatch> source =
        matchBoundedWideningDotReduceDequantSourceFunc(funcs.front());
    if (mlir::failed(source)) {
      signalPassFailure();
      return;
    }

    std::string kernelName = getKernelName(module);
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    if (mlir::failed(materializeCanonicalProblem(
            builder, kernelName, *source, march, isaVectorHints))) {
      signalPassFailure();
      return;
    }

    // Fill the target-bound c_o provider through the shared producer; candidate
    // and compute-body construction remain downstream owner responsibilities.
    (void)materializeRVVProviderCapabilityAxes(module, march, isaVectorHints);

    module->removeAttr(kSourceFrontDoorAttrName);
    module->removeAttr(kSourceKernelAttrName);
  }

private:
  Pass::Option<std::string> march{
      *this, "march",
      llvm::cl::desc("RISC-V -march used only to populate the target-bound "
                     "RVV capability profile; downstream owner construction "
                     "derives the integer-core plan. Empty leaves required "
                     "VLEN facts absent and later construction fails closed."),
      llvm::cl::init("")};
  Pass::Option<std::string> isaVectorHints{
      *this, "isa-vector-hints",
      llvm::cl::desc("Optional probed ISA/vector hint string folded into the "
                     "capability VLEN derivation alongside -march."),
      llvm::cl::init("")};
};

} // namespace

std::unique_ptr<::mlir::Pass>
createMaterializeRVVDequantDotSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  (void)registry;
  return std::make_unique<MaterializeRVVDequantDotSourceFrontDoorPass>();
}

llvm::Error registerRVVDequantDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  (void)registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin, formula_catalog::kDequantDotSourceEntry,
      "Adapt one bounded widening-dot/dequant source into target-bound exact "
      "canonical P",
      formula_catalog::kDequantDotSourceConstruction,
      [] {
        return std::make_unique<MaterializeRVVDequantDotSourceFrontDoorPass>();
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
