//===- RVVReductionSourceFrontDoor.cpp ------------------------------------===//
//
// Track B auto-lowering front door (the "type-Triton-ish backend" first block):
// the COMPILER auto-constructs the weft_rvv RVV-dialect body for a GENERIC
// vector-dialect signed widening int8 dot-reduce, instead of a per-kernel hand
// emitter. The integer-core LMUL anchor is the typed RVVSourceScheduleFormula
// result over source/capability facts,
// the SAME generic op emits an e8m2-form body at VLEN128 and an e8m1-form body at
// VLEN256 (the capability flip, exactly the q8_0 brick #1 shape, but from a
// generic vector.multi_reduction with no per-kernel emitter).
//
// What is auto-generated: the construction of the load/widening_product/
// standalone_reduce/store op STRUCTURE (the "intelligence" the design doc
// locates in the dialect body, not the dumb 1:1 EmitC printer). The unchanged
// RVVToEmitC emitter consumes this body verbatim. The novelty claimed is
// narrow and real: capability-fact-driven LMUL SELECTION fused into a
// vector->weft-body->EmitC lowering -- not "we built a Triton backend"
// (vector->LLVM->RVV exists upstream).
//
// Bar A achieved: the SAME generic source emits a BYTE-DIFFERENT RVV body by
// capability -- vsetvl_e8m2 + vwmul_vv_i16m4 + vwredsum_vs_i16m4_i32m1 at VLEN128
// vs vsetvl_e8m1 + vwmul_vv_i16m2 + vwredsum_vs_i16m2_i32m1 at VLEN256. The flip
// rides on additive byte-anchor dialect configs (isRVVByteAnchorDotReduceStrip
// Config + the i8m1->i16m2 product rung), parallel to the existing deferred-wide
// configs, NOT a per-kernel emitter.
//
// Honest framing: this is a SINGLE bounded contraction shape (K=32 signed int8
// dot-reduce, robust strip form, multi_block_factor pinned to 1). It is the
// first auto-lowered block, not a general dot-reduce auto-tuner.
//
//===----------------------------------------------------------------------===//

#include "Weft/Plugin/RVV/RVVReductionSourceFrontDoor.h"
#include "Weft/Plugin/RVV/RVVFormulaCatalog.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVCapabilityProfile.h"
#include "Weft/Plugin/RVV/RVVExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"
#include "Weft/Plugin/RVV/RVVSourceScheduleFormula.h"

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

// The marker the source module carries to route to this front door.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "weft_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("weft_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "bounded_widening_dot_reduce_source");
constexpr llvm::StringLiteral kSeedAttrName("weft_rvv.lowering_seed");

constexpr llvm::StringLiteral kRVVCapabilitySymbol("rvv");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");

// The single bounded contraction this first block auto-lowers: the K=32 signed
// int8 dot-reduce. K=32 IS the schedule descriptor blockLen (one elided cover
// must span 32 lanes); the m2->m1 anchor flip with VLEN is reduction-count
// driven against this fixed block, so the flip is EARNED by a real structural
// fact, not a no-block runtime-n picker divergence.
constexpr std::int64_t kContractionBlockLen = 32;

mlir::LogicalResult fail(mlir::Operation *op, llvm::Twine message) {
  op->emitError()
      << "bounded RVV widening-dot-reduce source front door failed: " << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the generic vector-dialect widening int8 dot-reduce source.
//===----------------------------------------------------------------------===//

struct WideningDotReduceSourceMatch {
  mlir::func::FuncOp func;
  mlir::Value lhsSource;
  mlir::Value rhsSource;
  mlir::Value outDestination;
  mlir::Value accSource;
  mlir::Value runtimeN;
};

bool isRank1MemRef(mlir::Type type, unsigned bitwidth) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  return memref && memref.getRank() == 1 &&
         memref.getElementType().isInteger(bitwidth);
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
//   func(%lhs: memref<?xi8>, %rhs: memref<?xi8>, %out: memref<?xi32>,
//        %acc: memref<?xi32>, %n: index) {
//     %a  = vector.transfer_read %lhs[0]              : vector<32xi8>
//     %b  = vector.transfer_read %rhs[0]              : vector<32xi8>
//     %ae = arith.extsi %a : vector<32xi8> to vector<32xi32>
//     %be = arith.extsi %b : vector<32xi8> to vector<32xi32>
//     %p  = arith.muli %ae, %be                        : vector<32xi32>
//     %s  = <acc[0] scalar seed>
//     %r  = vector.multi_reduction <add>, %p, %s [0]   : vector<32xi32> to i32
//     <store %r to out[0]>
//   }
// The widening (i8 source -> i32 accumulator) MUST go through extsi: a
// vector.multi_reduction ties its acc/dest element type to its source, so a
// plain-i8 reduction can only yield i8. The extsi encodes both "signed" and
// "i8 source -> e8 loads in the realized body".
mlir::FailureOr<WideningDotReduceSourceMatch>
matchBoundedWideningDotReduceSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func,
                "source function must have a body for structural pattern match");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 5 || type.getNumResults() != 0)
    return fail(func, "source function must have exactly five inputs and no "
                      "results: lhs/rhs memref<?xi8>, out/acc memref<?xi32>, "
                      "plus n index");
  if (!isRank1MemRef(type.getInput(0), 8) ||
      !isRank1MemRef(type.getInput(1), 8) ||
      !isRank1MemRef(type.getInput(2), 32) ||
      !isRank1MemRef(type.getInput(3), 32) || !type.getInput(4).isIndex())
    return fail(func, "source function inputs must be lhs/rhs rank-1 i8 "
                      "memrefs, out/acc rank-1 i32 memrefs, and one runtime n "
                      "index");

  // The scalar reduction result is stored with memref.store (the i32
  // accumulator is a scalar, not a vector), and the reduction acc seed reads
  // acc[0] with memref.load -- so the bounded body is two vector.transfer_read
  // i8 loads + the widening extsi/muli/multi_reduction chain + a scalar
  // memref.load seed + a scalar memref.store result.
  llvm::SmallVector<mlir::vector::TransferReadOp, 2> reads;
  llvm::SmallVector<mlir::memref::StoreOp, 1> stores;
  llvm::SmallVector<mlir::memref::LoadOp, 1> seedLoads;
  llvm::SmallVector<mlir::vector::MultiDimReductionOp, 1> reductions;
  llvm::SmallVector<mlir::arith::ExtSIOp, 2> extends;
  llvm::SmallVector<mlir::arith::MulIOp, 1> products;
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
                "supported by this bounded widening-dot-reduce source pattern");
  if (unsupportedArithVectorOp)
    return fail(unsupportedArithVectorOp,
                "only arith.extsi and arith.muli vector ops are supported by "
                "this bounded widening-dot-reduce source path");
  if (reads.size() != 2 || stores.size() != 1 || seedLoads.size() != 1 ||
      reductions.size() != 1 || extends.size() != 2 || products.size() != 1)
    return fail(func,
                "source pattern must contain exactly two vector.transfer_read, "
                "two arith.extsi, one arith.muli, one vector.multi_reduction, "
                "one scalar memref.load acc seed, and one scalar memref.store "
                "result");

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
  mlir::Block &entry = func.getBody().front();
  if (entry.getNumArguments() != 5 ||
      lhsRead.getSource() != entry.getArgument(0) ||
      rhsRead.getSource() != entry.getArgument(1) ||
      store.getMemRef() != entry.getArgument(2) ||
      seedLoad.getMemRef() != entry.getArgument(3))
    return fail(func, "source memory roles must be structural: transfer_read "
                      "lhs/rhs from the first two arguments, the scalar store "
                      "to the third (out) argument, and the acc seed load from "
                      "the fourth (acc) argument");

  // The widened i8->i32 source/product vectors.
  auto i8VecType = llvm::dyn_cast<mlir::VectorType>(lhsRead.getVector().getType());
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

  // The reduction accumulator seed must be the acc[0] memref.load result; the
  // reduction result must be the value the memref.store writes to out[0]. The
  // exact scalar-seed dataflow is then honored structurally by the realized
  // body's accumulator_layout/result_layout facts.
  if (reduction.getAcc() != seedLoad.getResult())
    return fail(reduction, "the multi_reduction acc seed must be the acc[0] "
                           "memref.load result");
  if (store.getValueToStore() != reduction.getResult())
    return fail(store, "the memref.store must write the multi_reduction result "
                       "to out[0]");
  if (!hasNoTransferMask(lhsRead) || !hasNoTransferMask(rhsRead))
    return fail(func, "masked vector transfers are outside this bounded source "
                      "path");
  if (!hasOneIndex(lhsRead.getIndices()) || !hasOneIndex(rhsRead.getIndices()))
    return fail(func, "source vector transfers must be rank-1 unit-stride "
                      "memory accesses");

  return WideningDotReduceSourceMatch{func,
                                      entry.getArgument(0),
                                      entry.getArgument(1),
                                      entry.getArgument(2),
                                      entry.getArgument(3),
                                      entry.getArgument(4)};
}

//===----------------------------------------------------------------------===//
// (2) The capability-fact-driven integer-core LMUL anchor.
//===----------------------------------------------------------------------===//

// The RVVSourceScheduleFormula owner constructs the integer-core LMUL anchor for
// the K=32 signed-int8 dot-reduce from typed geometry and canonical capability.
// The robust single-block body consumes only the returned
// integer_core_lmul knob, so for this FIXED K=32 int8 block the resource-best legal
// anchor is exactly the WIDTH-INVARIANT one: the LMUL whose per-strip VLMAX equals
// the 32-block, so its vector register group holds a CONSTANT 256-bit effective
// width (VLMAX*sew = 32*8) independent of VLEN. That IS the capability flip:
//   * VLEN128: LMUL 2 (e8m2, VLMAX 32) -- one strip spans the 32-block.
//   * VLEN256: LMUL 1 (e8m1, VLMAX 32) -- the SAME 256-bit group, one strip.
// A wider VLEN pins the same group at a NARROWER LMUL, so the anchor FLIPS m2->m1
// (the e8m2 vs e8m1 emitted-body divergence) -- byte-for-byte the old argmin's pick
// (m2@VLEN128 / m1@VLEN256), now the RETURN VALUE of the explicit width-invariant
// equation (reason `effective_width_invariant`, never a cost-model `static_order`)
// instead of an enumerate+rank fallback.
std::optional<std::string>
selectIntegerCoreLMUL(mlir::ModuleOp module, llvm::StringRef march,
                      llvm::StringRef isaVectorHints) {
  std::int64_t minimumVLEN = resolveRVVMinimumVLEN(module, march, isaVectorHints);
  llvm::Expected<RVVSourceSchedulePlan> plan =
      constructRVVSourceScheduleFormula(
          {RVVSourceScheduleMechanism::EffectiveWidthInvariant,
           /*sew=*/8, /*blockLength=*/kContractionBlockLen, {"m1", "m2"}},
          {minimumVLEN, resolveRVVVectorRegisterBudget(module)},
          RVVSourceScheduleNoStaticContext{});
  if (!plan)
    return std::nullopt;
  return plan->integerCoreLMUL;
}

//===----------------------------------------------------------------------===//
// (3) Body builder: auto-construct the weft_rvv RVV-dialect body. The integer
// source/product LMULs are VARIABLES threaded from the selected anchor, so the
// scaffold (fixed anchor) and bar A (capability-selected anchor) differ by the
// anchor value only -- not the construction.
//===----------------------------------------------------------------------===//

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

void createCapability(mlir::OpBuilder &builder, mlir::Location loc,
                      llvm::StringRef symbol, llvm::StringRef id,
                      llvm::StringRef kind) {
  mlir::OperationState state(loc, weftexec::CapabilityOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(symbol));
  state.addAttribute("id", builder.getStringAttr(id));
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addAttribute("status", builder.getStringAttr("available"));
  (void)builder.create(state);
}

mlir::ArrayAttr createRequires(mlir::OpBuilder &builder, llvm::StringRef symbol) {
  return builder.getArrayAttr({symbolRef(builder, symbol)});
}

weftrvv::PolicyAttr createAgnosticPolicy(mlir::OpBuilder &builder) {
  return weftrvv::PolicyAttr::get(builder.getContext(),
                                  weftrvv::TailPolicy::Agnostic,
                                  weftrvv::MaskPolicy::Agnostic);
}

weftrvv::RuntimeABIValueOp
createRuntimeABIValue(mlir::OpBuilder &builder, mlir::Location loc,
                      llvm::StringRef role, llvm::StringRef cName,
                      llvm::StringRef cType, llvm::StringRef purpose,
                      mlir::Type resultType) {
  mlir::OperationState state(loc,
                             weftrvv::RuntimeABIValueOp::getOperationName());
  state.addAttribute("role", builder.getStringAttr(role));
  state.addAttribute("c_name", builder.getStringAttr(cName));
  state.addAttribute("c_type", builder.getStringAttr(cType));
  state.addAttribute("ownership",
                     builder.getStringAttr("target-export-abi-owned"));
  state.addAttribute("purpose", builder.getStringAttr(purpose));
  state.addTypes(resultType);
  return llvm::cast<weftrvv::RuntimeABIValueOp>(builder.create(state));
}

// The strip setvl/with_vl carries the integer-core ANCHOR config (SEW + LMUL):
// the generic first-slice scaffold uses SEW32/m1; the byte-anchor dot-reduce
// (bar A) uses SEW8/m{1,2}. The emitter synthesizes the strip vsetvl from this
// config, and the reduce names vwredsum off its own i16 source type.
weftrvv::SetVLOp createSetVL(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value n, std::int64_t sew,
                             llvm::StringRef lmul, weftrvv::PolicyAttr policy) {
  mlir::OperationState state(loc, weftrvv::SetVLOp::getOperationName());
  state.addOperands(n);
  state.addAttribute("sew", builder.getI64IntegerAttr(sew));
  state.addAttribute("lmul", builder.getStringAttr(lmul));
  state.addAttribute("policy", policy);
  state.addTypes(weftrvv::VLType::get(builder.getContext()));
  return llvm::cast<weftrvv::SetVLOp>(builder.create(state));
}

weftrvv::WithVLOp createWithVL(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value vl, std::int64_t sew,
                               llvm::StringRef lmul,
                               weftrvv::PolicyAttr policy) {
  mlir::OperationState state(loc, weftrvv::WithVLOp::getOperationName());
  state.addOperands(vl);
  state.addAttribute("sew", builder.getI64IntegerAttr(sew));
  state.addAttribute("lmul", builder.getStringAttr(lmul));
  state.addAttribute("policy", policy);
  state.addRegion();
  auto withVL = llvm::cast<weftrvv::WithVLOp>(builder.create(state));
  withVL.getBody().emplaceBlock();
  return withVL;
}

mlir::Value createRVVLoad(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value buffer, mlir::Value vl,
                          mlir::Type vectorType) {
  mlir::OperationState state(loc, weftrvv::LoadOp::getOperationName());
  state.addOperands({buffer, vl});
  state.addTypes(vectorType);
  return builder.create(state)->getResult(0);
}

mlir::Value createWideningProduct(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value lhs, mlir::Value rhs,
                                  mlir::Value vl, mlir::Type productType,
                                  llvm::StringRef productRelation) {
  mlir::OperationState state(loc, weftrvv::WideningProductOp::getOperationName());
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr("signed_widening_product"));
  state.addAttribute("product_relation", builder.getStringAttr(productRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

mlir::Value createStandaloneReduce(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value input, mlir::Value accumulatorSeed,
                                   mlir::Value vl, mlir::Type resultType) {
  mlir::OperationState state(loc,
                             weftrvv::StandaloneReduceOp::getOperationName());
  state.addOperands({input, accumulatorSeed, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("signed_widening_reduce_add"));
  state.addAttribute(
      "accumulator_layout",
      builder.getStringAttr("scalar-i32-seed-lane0-from-accumulator-input"));
  state.addAttribute(
      "result_layout",
      builder.getStringAttr("store-standalone-reduction-lane0-to-output-scalar"));
  state.addTypes(resultType);
  return builder.create(state)->getResult(0);
}

void createRVVStore(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value buffer, mlir::Value value, mlir::Value vl) {
  mlir::OperationState state(loc, weftrvv::StoreOp::getOperationName());
  state.addOperands({buffer, value, vl});
  (void)builder.create(state);
}

weftexec::VariantOp
createVariant(mlir::OpBuilder &builder, mlir::Location loc,
              llvm::StringRef selectedVariantSymbol, mlir::ArrayAttr requires,
              weftrvv::PolicyAttr policy) {
  mlir::OperationState state(loc, weftexec::VariantOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(selectedVariantSymbol));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(getRVVExtensionPluginName()));
  state.addAttribute(kRequiresAttrName, requires);
  state.addAttribute("weft_rvv.policy", policy);
  state.addRegion();
  auto variant = llvm::cast<weftexec::VariantOp>(builder.create(state));
  variant.getBody().emplaceBlock();
  return variant;
}

mlir::LogicalResult materializeKernel(
    mlir::OpBuilder &builder, llvm::StringRef kernelName,
    llvm::StringRef selectedIntegerCoreLMUL,
    const ExtensionPluginRegistry &registry,
    WideningDotReduceSourceMatch source) {
  (void)registry;
  mlir::Location loc = source.func.getLoc();
  weftrvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol = "rvv_widening_dot_reduce_i8";

  // BAR A: the BYTE-ANCHOR widening int8 dot-reduce. The integer-core anchor is
  // the gearbox-selected LMUL (m2 at VLEN128, m1 at VLEN256), realized as a
  // byte-strip config: SEW8/<anchor> with_vl, i8/<anchor> loads, the i16/<wider>
  // widening product, the i32m1 widening reduce. The emitted bytes FLIP with the
  // capability fact (vsetvl_e8m2 + vwredsum_i16m4 vs vsetvl_e8m1 + vwredsum_i16m2)
  // -- the same e8m2/e8m1 divergence q8_0 brick #1 shows, but from a generic
  // vector.multi_reduction source with no per-kernel emitter.
  llvm::StringRef loadLMUL = selectedIntegerCoreLMUL;
  llvm::StringRef productLMUL = getRVVNextWiderLMUL(loadLMUL);
  if (productLMUL.empty())
    return fail(source.func,
                llvm::Twine("no wider LMUL rung for integer-core anchor '") +
                    loadLMUL + "'");
  std::int64_t anchorSEW = 8;

  mlir::OperationState kernelState(loc,
                                   weftexec::KernelOp::getOperationName());
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addAttribute("construction_domain",
                           builder.getStringAttr("riscv-execution"));
  kernelState.addAttribute("problem", symbolRef(builder, "canonical_problem"));
  kernelState.addRegion();
  auto kernel = llvm::cast<weftexec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  mlir::OperationState problemState(
      loc, weftexec::I8WideningDotReduceProblemOp::getOperationName());
  problemState.addAttribute("sym_name",
                            builder.getStringAttr("canonical_problem"));
  problemState.addAttribute("block_length",
                            builder.getI64IntegerAttr(kContractionBlockLen));
  problemState.addAttribute("dequantize_to_f32", builder.getBoolAttr(false));
  (void)builder.create(problemState);

  createCapability(builder, loc, kRVVCapabilitySymbol, "rvv", "isa-vector");
  mlir::ArrayAttr rvvRequires = createRequires(builder, kRVVCapabilitySymbol);

  weftexec::VariantOp rvvVariant = createVariant(
      builder, loc, selectedVariantSymbol, rvvRequires, policy);
  mlir::OpBuilder::InsertionGuard variantGuard(builder);
  builder.setInsertionPointToStart(&rvvVariant.getBody().front());

  mlir::Type runtimeABIType =
      weftrvv::RuntimeABIValueType::get(builder.getContext());
  auto lhs = createRuntimeABIValue(builder, loc, "lhs-input-buffer", "lhs",
                                   "const int8_t *",
                                   "widening-dot-reduce:lhs", runtimeABIType);
  auto rhs = createRuntimeABIValue(builder, loc, "rhs-input-buffer", "rhs",
                                   "const int8_t *",
                                   "widening-dot-reduce:rhs", runtimeABIType);
  auto acc = createRuntimeABIValue(builder, loc, "accumulator-input-buffer",
                                   "acc", "const int32_t *",
                                   "widening-dot-reduce:acc", runtimeABIType);
  auto out = createRuntimeABIValue(builder, loc, "output-buffer", "out",
                                   "int32_t *", "widening-dot-reduce:out",
                                   runtimeABIType);
  auto n = createRuntimeABIValue(builder, loc, "runtime-element-count", "n",
                                 "size_t", "widening-dot-reduce:n",
                                 builder.getIndexType());

  weftrvv::SetVLOp setvl =
      createSetVL(builder, loc, n.getResult(), anchorSEW, loadLMUL, policy);
  weftrvv::WithVLOp withVL =
      createWithVL(builder, loc, setvl.getVl(), anchorSEW, loadLMUL, policy);

  mlir::OpBuilder::InsertionGuard withVLGuard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());

  mlir::Type i8VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getI8Type(), loadLMUL);
  mlir::Type i16VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getI16Type(), productLMUL);
  mlir::Type i32VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getI32Type(), "m1");

  // The product relation fact encodes the realized i8<anchor> x i8<anchor> ->
  // i16<wider> byte-strip chain (structural, NOT a route-id mirror).
  std::string productRelation =
      (llvm::Twine("signed-i8") + loadLMUL + "xi8" + loadLMUL + "-to-i16" +
       productLMUL)
          .str();

  mlir::Value loadedLHS =
      createRVVLoad(builder, loc, lhs.getResult(), setvl.getVl(), i8VecType);
  mlir::Value loadedRHS =
      createRVVLoad(builder, loc, rhs.getResult(), setvl.getVl(), i8VecType);
  mlir::Value product = createWideningProduct(
      builder, loc, loadedLHS, loadedRHS, setvl.getVl(), i16VecType,
      productRelation);
  mlir::Value reduced = createStandaloneReduce(
      builder, loc, product, acc.getResult(), setvl.getVl(), i32VecType);
  createRVVStore(builder, loc, out.getResult(), reduced, setvl.getVl());

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
  return "rvv_widening_dot_reduce_i8_from_vector_source";
}

class MaterializeRVVReductionSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVReductionSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVReductionSourceFrontDoorPass() = default;
  MaterializeRVVReductionSourceFrontDoorPass(
      const MaterializeRVVReductionSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVReductionSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        registry(other.registry) {}
  explicit MaterializeRVVReductionSourceFrontDoorPass(
      const ExtensionPluginRegistry *registry)
      : registry(registry) {}

  llvm::StringRef getArgument() const final {
    return "weft-rvv-materialize-widening-dot-reduce-source-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Auto-construct the weft_rvv widening int8 dot-reduce body from a "
           "generic vector.multi_reduction source, with the integer-core LMUL "
           "anchor selected by the shared block-dot schedule authority from the "
           "resolveRVVMinimumVLEN capability fact";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::vector::VectorDialect, weftexec::WEFTExecDialect,
                    weftrvv::WEFTRVVDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    if (!registry) {
      module.emitError()
          << "RVV widening-dot-reduce source front door requires an injected "
             "extension-plugin registry to dispatch the conservative fallback";
      signalPassFailure();
      return;
    }

    auto marker = module->getAttrOfType<mlir::StringAttr>(
        kSourceFrontDoorAttrName);
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
                         "widening-dot-reduce source function candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<WideningDotReduceSourceMatch> source =
        matchBoundedWideningDotReduceSourceFunc(funcs.front());
    if (mlir::failed(source)) {
      signalPassFailure();
      return;
    }

    std::optional<std::string> integerCoreLMUL =
        selectIntegerCoreLMUL(module, march, isaVectorHints);
    if (!integerCoreLMUL) {
      (void)fail(module, llvm::Twine("the capability profile (march='") + march +
                             "') prunes every legal K=32 integer-core anchor; "
                             "no schedule is selectable (fail-closed)");
      signalPassFailure();
      return;
    }

    std::string kernelName = getKernelName(module);
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    if (mlir::failed(materializeKernel(builder, kernelName, *integerCoreLMUL,
                                       *registry, *source))) {
      signalPassFailure();
      return;
    }

    // W2 §(1) B: the constructed body now carries an RVV provider op; materialize
    // the c facts (typed minimum_vlen + support axes) onto it through the ONE
    // shared producer, so a downstream resolveRVVMinimumVLEN reads the provider
    // fact instead of re-parsing -march (I1/I4). This closes the "constructed
    // empty provider" debt: the front door is the legitimate producer.
    (void)materializeRVVProviderCapabilityAxes(module, march, isaVectorHints);

    module->removeAttr(kSourceFrontDoorAttrName);
    module->removeAttr(kSourceKernelAttrName);
  }

private:
  const ExtensionPluginRegistry *registry = nullptr;

  Pass::Option<std::string> march{
      *this, "march",
      llvm::cl::desc("The capability-derivation -march the integer-core LMUL "
                     "anchor is selected from (e.g. rv64gcv, rv64gcv_zvl256b). "
                     "Empty => no guaranteed VLEN tier => fail-closed."),
      llvm::cl::init("")};
  Pass::Option<std::string> isaVectorHints{
      *this, "isa-vector-hints",
      llvm::cl::desc("Optional probed ISA/vector hint string folded into the "
                     "capability VLEN derivation alongside -march."),
      llvm::cl::init("")};
};

} // namespace

std::unique_ptr<::mlir::Pass>
createMaterializeRVVReductionSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeRVVReductionSourceFrontDoorPass>(&registry);
}

llvm::Error registerRVVReductionSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  const ExtensionPluginRegistry *registryPtr = &registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin, formula_catalog::kReductionSourceEntry,
      "Auto-construct the weft_rvv widening int8 dot-reduce body from a generic "
      "vector.multi_reduction source (capability-selected integer-core LMUL)",
      formula_catalog::kReductionSourceConstruction,
      [registryPtr] {
        return createMaterializeRVVReductionSourceFrontDoorPass(*registryPtr);
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
