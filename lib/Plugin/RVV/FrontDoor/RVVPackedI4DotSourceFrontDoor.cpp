//===- RVVPackedI4DotSourceFrontDoor.cpp ---------------------------------===//
//
// Track B G1, the BOUNDED first step: the COMPILER auto-CONSTRUCTS the q4_0
// nibble INTEGER-CORE body from a marked GENERIC source carrying the nibble-core
// operator identity, instead of routing a monolithic op to a per-kernel hand
// emitter. The auto-constructed body is the single-strip generic-op composition
//   load x3 (packed-i4 weight + the two plain-i8 q8 activation halves)
//     -> weft_rvv.packed_i4_offset_binary_x_i8_product (the offset-binary nibble
//        decode + asymmetric widening product -- ALREADY a first-class generic op,
//        lowered by the SAME emitOffsetBinaryDecodeProductValue the hand-written
//        block-dot strip calls)
//     -> weft_rvv.standalone_reduce (signed widening reduce, i16 -> i32)
//     -> weft_rvv.store
// the unchanged --weft-rvv-lower-to-emitc emitter consumes verbatim. This proves
// auto-construction REACHES nibble-decode through the generic-op mechanism: the
// nibble unpack itself needs NO new emitter vocabulary; it is already a typed op.
//
// HONEST SCOPE -- the nibble integer CORE only (NOT the full q4_0 KERNEL). There is
// NO nb = n / QK outer block loop, NO per-block dual fp16 scale read, and NO left-
// associative fp32 fold; those three axes need NEW generic ODS vocabulary and are
// full G1, DEFERRED. The monolithic weft_rvv.q4_0_q8_0_block_dot op, its KERNEL
// front door (RVVQ40BlockDotSourceFrontDoor), and the hand emitter
// (emitQ4_0Q8_0BlockDot) all STAY -- this adds a SEPARATE rung-3 front door, like
// dequant-vs-reduction.
//
// CAPABILITY framing -- NO FLIP claimed. The capability consultation is the SAME
// RVVIntegerCoreScheduleFormula owner the rung-1/2 front doors use, run as the
// LEGALITY GATE (fail-closed via I7
// if the integer-core path is pruned), NOT as the nibble anchor source. q4_0's
// nibble HALF-block integer core is pinned at i8mf4-i16mf2-i32m1 at every Zvl128b
// tier -- there is NO VLEN128-vs-VLEN256 byte-flip here. This is the documented
// q4_0 no-flip property (RVVQ40BlockDotSourceFrontDoor.cpp:22-33) and the gearbox's
// own packed-i4 candidate (`signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1`, the single
// anchor rung listed for packed-i4). The mf4 isolated-core strip is the NARROWER
// (4-element strip) sibling of the full-block plain-int8 anchor the gate navigates;
// the bounded step pins it byte-identical to the existing nibble-core lit and
// DEFERS both the gearbox-selected full-kernel SHAPE (m1/factor=4/elided) and the
// full kernel to G1.
//
//===----------------------------------------------------------------------===//

#include "Weft/Plugin/RVV/RVVPackedI4DotSourceFrontDoor.h"
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

// The DISTINCT marker the source module carries to route to THIS packed-i4 nibble
// integer-core front door (NOT the MVP's "bounded_widening_dot_reduce_source", the
// dequant rung's "..._dequantize_source", nor the q4_0 KERNEL front door's
// "ggml_q4_0_q8_0_block_dot_source"). Each front-door pass checks its own marker
// and early-returns on a mismatch, so the four passes are mutually exclusive and
// every existing front-door lit is byte-unchanged.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "weft_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("weft_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "bounded_packed_i4_offset_binary_dot_source");
constexpr llvm::StringLiteral kSeedAttrName("weft_rvv.lowering_seed");

// The plain-int8 K=32 block the LEGALITY GATE reasons about (the same fixed block
// the dequant/reduction rungs feed the shared schedule authority). The gate only
// asks "is a legal integer-core widening schedule selectable at this VLEN tier?";
// the nibble anchor itself is pinned (see below), not read from the gate.
constexpr std::int64_t kContractionBlockLen = 32;

mlir::LogicalResult fail(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "bounded RVV packed-i4 offset-binary dot source front door "
                     "failed: "
                  << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the marked generic source carrying the nibble INTEGER-CORE operator
//     identity. Like the q4_0 KERNEL matcher, recognition is by the integer-core
//     ABI roles, NOT a straight-line generic vector dataflow -- the offset-binary
//     nibble decode has no compact generic vector form (it is exactly the
//     first-class packed_i4 op the body builder constructs).
//===----------------------------------------------------------------------===//

struct PackedI4DotSourceMatch {
  mlir::func::FuncOp func;
};

bool isRank1MemRef(mlir::Type type, unsigned bitwidth) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  return memref && memref.getRank() == 1 &&
         memref.getElementType().isInteger(bitwidth);
}

// Match the nibble integer-core OPERATOR-IDENTITY source signature:
//   func(%weight: memref<?xi8>, %qlo: memref<?xi8>, %qhi: memref<?xi8>,
//        %acc: memref<?xi32>, %out: memref<?xi32>, %n: index)
// holding ONLY the packed-i4 nibble-core intent marker. The six roles are the
// packed-i4 weight (each byte two offset-binary nibbles), the two plain-i8 q8
// activation halves paired with the low/high nibbles, the i32 accumulator seed,
// the i32 output, and the runtime element count. The body is the bounded intent
// shell (it may be a bare `return`): the offset-binary decode + asymmetric product
// are q4_0 STRUCTURE the constructed packed_i4 op carries, so this front door does
// NOT pretend to recognize the nibble unpack from generic vector ops.
mlir::FailureOr<PackedI4DotSourceMatch>
matchPackedI4DotSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func, "source function must have a body for the packed-i4 "
                      "nibble integer-core intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 6 || type.getNumResults() != 0)
    return fail(func,
                "source function must have exactly six inputs and no results: "
                "weight/qlo/qhi memref<?xi8>, acc memref<?xi32>, out "
                "memref<?xi32>, and n index (the packed-i4 nibble integer-core "
                "operator identity)");
  if (!isRank1MemRef(type.getInput(0), 8) ||
      !isRank1MemRef(type.getInput(1), 8) ||
      !isRank1MemRef(type.getInput(2), 8) ||
      !isRank1MemRef(type.getInput(3), 32) ||
      !isRank1MemRef(type.getInput(4), 32) || !type.getInput(5).isIndex())
    return fail(func,
                "source function inputs must be packed-i4 weight rank-1 i8 "
                "memref, two plain-i8 q8 activation-half rank-1 i8 memrefs, an "
                "i32 acc seed rank-1 memref, an i32 out rank-1 memref, and an n "
                "index");

  return PackedI4DotSourceMatch{func};
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
    PackedI4DotSourceMatch source, llvm::StringRef march,
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
      loc, weftexec::PackedI4Q8DotProblemOp::getOperationName());
  problemState.addAttribute("sym_name",
                            builder.getStringAttr("canonical_problem"));
  problemState.addAttribute("block_length",
                            builder.getI64IntegerAttr(kContractionBlockLen));
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
  return "rvv_packed_i4_offset_binary_dot_i8_from_source";
}

class MaterializeRVVPackedI4DotSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVPackedI4DotSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVPackedI4DotSourceFrontDoorPass() = default;
  MaterializeRVVPackedI4DotSourceFrontDoorPass(
      const MaterializeRVVPackedI4DotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVPackedI4DotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other) {}

  llvm::StringRef getArgument() const final {
    return "weft-rvv-materialize-packed-i4-offset-binary-dot-source-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Adapt one bounded packed-i4/q8 dot source into target-bound "
           "PackedI4Q8Dot canonical P";
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
      (void)fail(module, "source module must contain exactly one RVV packed-i4 "
                         "nibble integer-core source function candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<PackedI4DotSourceMatch> source =
        matchPackedI4DotSourceFunc(funcs.front());
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

    // Fill target-bound c_o through the shared producer. The owner later uses it
    // for legality and construction; the adapter does not prebuild a body.
    (void)materializeRVVProviderCapabilityAxes(module, march, isaVectorHints);

    module->removeAttr(kSourceFrontDoorAttrName);
    module->removeAttr(kSourceKernelAttrName);
  }

private:
  Pass::Option<std::string> march{
      *this, "march",
      llvm::cl::desc("RISC-V -march used only to populate the target-bound "
                     "RVV capability profile consumed by downstream legality "
                     "and selected-owner construction. Empty leaves required "
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
createMaterializeRVVPackedI4DotSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  (void)registry;
  return std::make_unique<MaterializeRVVPackedI4DotSourceFrontDoorPass>();
}

llvm::Error registerRVVPackedI4DotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  (void)registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin, formula_catalog::kPackedI4DotSourceEntry,
      "Adapt one bounded packed-i4/q8 dot source into target-bound exact "
      "canonical P",
      formula_catalog::kPackedI4DotConstruction,
      [] {
        return std::make_unique<MaterializeRVVPackedI4DotSourceFrontDoorPass>();
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
