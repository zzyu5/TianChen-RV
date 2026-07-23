//===- RVVCodebookDotSourceFrontDoor.cpp --------------------------------===//
//
// Track B G2, the BOUNDED first step: the COMPILER auto-CONSTRUCTS the codebook
// (vrgather) INTEGER-CORE body from a marked GENERIC source carrying the codebook
// -core operator identity, instead of routing a monolithic op to a per-kernel
// hand emitter. The auto-constructed body is the single-strip generic-op
// composition
//   weft_rvv.codebook_table_broadcast (the 16-entry kvalues table -> values vreg)
//   load x3 (UNSIGNED packed-i4 weight + the two plain-i8 q8 activation halves)
//     -> weft_rvv.codebook_gather_x_i8_product (the nibble split + vrgather
//        codebook decode + asymmetric widening product -- the codebook variant of
//        the nibble core, REUSING the SAME emitOffsetBinaryProductFromDecodedValue
//        product tail the hand-written block-dot strip calls)
//     -> weft_rvv.standalone_reduce (signed widening reduce, i16 -> i32)
//     -> weft_rvv.store
// the unchanged --weft-rvv-lower-to-emitc emitter consumes verbatim. This proves
// the generic auto-construction mechanism (proven for q4_0 nibble in G1) reaches a
// STRUCTURALLY DIFFERENT family: the codebook is NOT the q4_0 xor/sll/sra
// arithmetic decode; each 4-bit nibble is an INDEX into a non-linear int8 table
// gathered by vrgather.
//
// HONEST SCOPE -- the codebook integer CORE only (NOT the full codebook KERNEL).
// There is NO nb = n / QK outer block loop, NO per-block fp16 scale read, NO fp32
// fold, and NO once-above-loop table hoisting (the table_broadcast is emitted
// per-strip here, vs hoisted above the block loop in the monolithic kernel); those
// axes need NEW generic ODS vocabulary and are full G2, DEFERRED. The monolithic
// weft_rvv.iq4_nl_q8_0_block_dot / mxfp4 / nvfp4 ops, their KERNEL front doors, and
// the hand emitters (RVVToEmitCCodebookFp4.cpp) all STAY -- this adds a SEPARATE
// rung-3 front door, like dequant-vs-reduction.
//
// CAPABILITY framing -- the codebook DOES flip (the q4_0 sibling did NOT). The
// capability consultation is the same RVVIntegerCoreScheduleFormula owner the rung-1/2
// front doors use, but here the selected i8
// anchor is THREADED into the body types (NOT pinned). At VLEN128 only m1 reaches
// VLMAX 16 (the 16-entry table needs every lane indexable; mf2 -> 8 < 16 is
// PRUNED), so the emit is vrgather_vv_i8m1 + i16m2 product; at VLEN256 mf2 is
// admitted and its lighter footprint wins, so the emit is vrgather_vv_i8mf2 + i16m1
// product. The VLEN128 vs VLEN256 emit DIFFER -- the genuine capability flip,
// demonstrated at the bounded-core granularity (the property the board-sealed FP4
// brick #2 records). At VLEN0 every candidate is pruned -> fail-closed (I7). This
// is MECHANISM/parity, NOT a speed beat (G5).
//
//===----------------------------------------------------------------------===//

#include "Weft/Plugin/RVV/RVVCodebookDotSourceFrontDoor.h"
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

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace weft::plugin::rvv {
namespace {

namespace weftexec = ::weft::exec;
namespace weftrvv = ::weft::rvv;

// The DISTINCT marker the source module carries to route to THIS codebook-gather
// integer-core front door (mutually exclusive with the MVP / dequant / packed-i4
// nibble / q4_0-KERNEL / iq4_nl-KERNEL markers). Each front-door pass checks its
// own marker and early-returns on a mismatch, so every existing front-door lit is
// byte-unchanged.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "weft_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("weft_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "bounded_codebook_gather_dot_source");
constexpr llvm::StringLiteral kSeedAttrName("weft_rvv.lowering_seed");

// The structured-const C symbol the codebook table decl declares + ggml's
// kvalues_iq4nl[16] -- the 16-entry NON-LINEAR int8 lookup table the codebook
// gather indexes (the load-bearing structural fact of the codebook class). Pinned
// to the iq4_nl table so the auto-constructed core is byte-comparable to the
// existing iq4_nl block-dot codebook-decode chain.
constexpr llvm::StringLiteral kCodebookTableSymbol("weft_iq4_nl_kvalues");
constexpr std::array<std::int8_t, 16> kIQ4NLCodebook = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113};

mlir::LogicalResult fail(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "bounded RVV codebook-gather dot source front door failed: "
                  << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the marked generic source carrying the codebook INTEGER-CORE
//     operator identity (six ABI roles, like the packed-i4 nibble core).
//===----------------------------------------------------------------------===//

struct CodebookDotSourceMatch {
  mlir::func::FuncOp func;
};

bool isRank1MemRef(mlir::Type type, unsigned bitwidth) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  return memref && memref.getRank() == 1 &&
         memref.getElementType().isInteger(bitwidth);
}

// Match the codebook integer-core OPERATOR-IDENTITY source signature:
//   func(%weight: memref<?xi8>, %qlo: memref<?xi8>, %qhi: memref<?xi8>,
//        %acc: memref<?xi32>, %out: memref<?xi32>, %n: index)
// holding ONLY the codebook-core intent marker. The six roles are the packed-i4
// weight (each byte two table-index nibbles), the two plain-i8 q8 activation
// halves paired with the low/high nibbles, the i32 accumulator seed, the i32
// output, and the runtime element count. The body is the bounded intent shell (it
// may be a bare `return`): the codebook split + gather are STRUCTURE the
// constructed codebook_gather op carries, so this front door does NOT pretend to
// recognize the gather from generic vector ops.
mlir::FailureOr<CodebookDotSourceMatch>
matchCodebookDotSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func, "source function must have a body for the codebook "
                      "integer-core intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 6 || type.getNumResults() != 0)
    return fail(func,
                "source function must have exactly six inputs and no results: "
                "weight/qlo/qhi memref<?xi8>, acc memref<?xi32>, out "
                "memref<?xi32>, and n index (the codebook integer-core operator "
                "identity)");
  if (!isRank1MemRef(type.getInput(0), 8) ||
      !isRank1MemRef(type.getInput(1), 8) ||
      !isRank1MemRef(type.getInput(2), 8) ||
      !isRank1MemRef(type.getInput(3), 32) ||
      !isRank1MemRef(type.getInput(4), 32) || !type.getInput(5).isIndex())
    return fail(func,
                "source function inputs must be a packed-i4 weight rank-1 i8 "
                "memref, two plain-i8 q8 activation-half rank-1 i8 memrefs, an "
                "i32 acc seed rank-1 memref, an i32 out rank-1 memref, and an n "
                "index");

  return CodebookDotSourceMatch{func};
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
    CodebookDotSourceMatch source, llvm::StringRef march,
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
      loc, weftexec::CodebookI4Q8DotProblemOp::getOperationName());
  problemState.addAttribute("sym_name",
                            builder.getStringAttr("canonical_problem"));
  problemState.addAttribute("block_length", builder.getI64IntegerAttr(16));
  problemState.addAttribute("codebook",
                            builder.getDenseI8ArrayAttr(kIQ4NLCodebook));
  problemState.addAttribute("table_symbol",
                            builder.getStringAttr(kCodebookTableSymbol));
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
  return "rvv_codebook_gather_dot_i8_from_source";
}

class MaterializeRVVCodebookDotSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVCodebookDotSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVCodebookDotSourceFrontDoorPass() = default;
  MaterializeRVVCodebookDotSourceFrontDoorPass(
      const MaterializeRVVCodebookDotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVCodebookDotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other) {}

  llvm::StringRef getArgument() const final {
    return "weft-rvv-materialize-codebook-gather-dot-source-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Adapt one bounded codebook-i4/q8 dot source into target-bound "
           "CodebookI4Q8Dot canonical P";
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
      (void)fail(module, "source module must contain exactly one RVV codebook "
                         "integer-core source function candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<CodebookDotSourceMatch> source =
        matchCodebookDotSourceFunc(funcs.front());
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

    // Fill target-bound c_o through the shared producer. The owner later derives
    // the codebook schedule; the adapter does not preselect or prebuild it.
    (void)materializeRVVProviderCapabilityAxes(module, march, isaVectorHints);

    module->removeAttr(kSourceFrontDoorAttrName);
    module->removeAttr(kSourceKernelAttrName);
  }

private:
  Pass::Option<std::string> march{
      *this, "march",
      llvm::cl::desc("RISC-V -march used only to populate the target-bound "
                     "RVV capability profile; downstream owner construction "
                     "derives the codebook gather plan. Empty leaves required "
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
createMaterializeRVVCodebookDotSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  (void)registry;
  return std::make_unique<MaterializeRVVCodebookDotSourceFrontDoorPass>();
}

llvm::Error registerRVVCodebookDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  (void)registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin, formula_catalog::kCodebookDotSourceEntry,
      "Adapt one bounded codebook-i4/q8 dot source into target-bound exact "
      "canonical P",
      formula_catalog::kCodebookDotConstruction,
      [] {
        return std::make_unique<MaterializeRVVCodebookDotSourceFrontDoorPass>();
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
