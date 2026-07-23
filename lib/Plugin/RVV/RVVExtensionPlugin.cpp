#include "Weft/Plugin/RVV/RVVExtensionPlugin.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableInterface.h"
#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/RVV/RVVArtifactContract.h"
#include "Weft/Plugin/RVV/RVVCapabilityProfile.h"
#include "Weft/Plugin/RVV/RVVFormulaCatalog.h"
#include "Weft/Plugin/RVV/RVVFormulaConstruction.h"
#include "Weft/Plugin/RVV/RVVMonolithicBlockDotFamily.h"
#include "Weft/Plugin/RVV/RVVDequantDotSourceFrontDoor.h"
#include "Weft/Plugin/RVV/RVVDequantizeRowStreamFrontDoor.h"
#include "Weft/Plugin/RVV/RVVElementwiseStreamFrontDoor.h"
#include "Weft/Plugin/RVV/RVVQuantizeRowStreamFrontDoor.h"
#include "Weft/Plugin/RVV/RVVQuantizeFormula.h"
#include "Weft/Plugin/RVV/RVVCodebookDotSourceFrontDoor.h"
#include "Weft/Plugin/RVV/RVVCompositeGatherMAccScatterFormula.h"
#include "Weft/Plugin/RVV/RVVFlatBlockDotFormula.h"
#include "Weft/Plugin/RVV/RVVMonolithicBlockDotSourceFrontDoor.h"
#include "Weft/Plugin/RVV/RVVPackedI4DotSourceFrontDoor.h"
#include "Weft/Plugin/RVV/RVVReductionSourceFrontDoor.h"
#include "Weft/Plugin/RVV/RVVScheduleFormula.h"
#include "Weft/Plugin/RVV/RVVSourceScheduleFormula.h"
#include "Weft/Plugin/RVV/RVVSelectedBodyRealization.h"
#include "Weft/Plugin/RVV/RVVVectorSourceFrontDoor.h"
#include "Weft/Support/RuntimeABI.h"
#include "Weft/Target/RVV/RVVTargetSupportBundle.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <optional>
#include <string>
#include <utility>

namespace weft::plugin {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVPluginVersion("0.1.0");
constexpr llvm::StringLiteral kRVVPolicyAttrName("weft_rvv.policy");
constexpr llvm::StringLiteral kOriginAttrName("origin");

namespace formula = weft::plugin::rvv::formula_catalog;

// Capability-DERIVED vector-paradigm ranking cost (SEL-1 exec-level capability
// prior). Emitted only when an available RVV isa-vector capability fact backs the
// variant — NOT a capability-blind literal. The registry ranks ascending, so this
// base sits below the scalar fallback (1000.0) and, symmetrically, defines the
// bar the IME matrix paradigm must undercut to take over a whole-matrix GEMM.
constexpr double kRVVVectorBaseCost = 1.0;

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasAvailableRVVCapability(const VariantProposalRequest &request) {
  return request.getKernel() &&
         request.getCapabilities().isCapabilityAvailableByID(
             rvv::getRVVCapabilityID());
}

bool variantContainsExplicitTypedRVVBody(weft::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (found || op == variant.getOperation())
      return;
    if (op->getName().getDialectNamespace() == "weft_rvv")
      found = true;
  });
  return found;
}

llvm::Error requireExplicitTypedRVVBody(weft::exec::VariantOp variant) {
  if (variantContainsExplicitTypedRVVBody(variant))
    return llvm::Error::success();
  return makeRVVPluginError(
      "materialized RVV variant requires explicit typed RVV "
      "extension-family body");
}

llvm::Expected<weft::rvv::WithVLOp>
findSelectedRVVSelectedBodyBoundary(weft::exec::VariantOp variant) {
  if (!variant)
    return makeRVVPluginError(
        "selected RVV lowering boundary requires a materialized "
        "weft.exec.variant");

  llvm::SmallVector<weft::rvv::SetVLOp, 2> setvls;
  llvm::SmallVector<weft::rvv::WithVLOp, 2> withVLs;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (auto setvl = llvm::dyn_cast<weft::rvv::SetVLOp>(op))
      setvls.push_back(setvl);
    if (auto withVL = llvm::dyn_cast<weft::rvv::WithVLOp>(op))
      withVLs.push_back(withVL);
  });

  if (setvls.size() != 1)
    return makeRVVPluginError(
        "selected RVV typed lowering boundary requires exactly one "
        "weft_rvv.setvl op");
  if (withVLs.size() != 1)
    return makeRVVPluginError(
        "selected RVV typed lowering boundary requires exactly one "
        "weft_rvv.with_vl op");
  weft::rvv::WithVLOp selectedWithVL = withVLs.front();

  weft::rvv::RVVConfigContractDiagnostic configDiagnostic =
      weft::rvv::validateRVVSelectedBodyConfigVLStructure(setvls.front(),
                                                          selectedWithVL);
  if (!configDiagnostic.ok)
    return makeRVVPluginError(configDiagnostic.message);

  return selectedWithVL;
}

llvm::Error validateSelectedRVVSelectedBodyBoundary(
    const VariantLoweringBoundaryValidationRequest &request) {
  weft::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVPluginError(
        "selected RVV lowering-boundary validation requires a materialized "
        "weft.exec.variant");
  if (!request.getKernel())
    return makeRVVPluginError(
        "selected RVV lowering-boundary validation requires an enclosing "
        "weft.exec.kernel");

  auto boundary =
      llvm::dyn_cast_if_present<weft::rvv::WithVLOp>(request.getBoundary());
  if (!boundary)
    return makeRVVPluginError(
        "selected RVV typed lowering boundary must be the existing "
        "weft_rvv.with_vl operation");

  if (!isOperationSelectedForVariant(boundary.getOperation(), variant))
    return makeRVVPluginError(
        "selected RVV typed body is not the exact result bound to the "
        "requested variant");
  if (mlir::failed(boundary.verify()) ||
      mlir::failed(rvv::validateRVVConstructedTypedBody(
          boundary.getOperation())))
    return makeRVVPluginError(
        "selected RVV exact typed body failed structural validation");
  return llvm::Error::success();
}

const rvv::RVVExtensionPlugin &getBuiltinRVVExtensionPlugin() {
  static const rvv::RVVExtensionPlugin plugin;
  return plugin;
}

} // namespace

namespace rvv {

llvm::StringRef getRVVExtensionPluginName() { return kRVVPluginName; }

llvm::StringRef getRVVExtensionPluginVersion() { return kRVVPluginVersion; }

llvm::StringRef getRVVPolicyAttrName() { return kRVVPolicyAttrName; }

RVVExtensionPlugin::RVVExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      rvv::getRVVCapabilityID(), rvv::getRVVCapabilityKind(),
      "RVV first-slice vector ISA capability participation; target "
      "availability is supplied by weft.exec.capability metadata"));
}

llvm::StringRef RVVExtensionPlugin::getName() const { return kRVVPluginName; }

llvm::StringRef RVVExtensionPlugin::getConstructionDomain() const {
  return "riscv-execution";
}

llvm::StringRef RVVExtensionPlugin::getVersion() const {
  return kRVVPluginVersion;
}

llvm::ArrayRef<PluginCapability> RVVExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void RVVExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<weft::rvv::WEFTRVVDialect>();
}

llvm::Error RVVExtensionPlugin::constructFormulaPlans(
    const FamilyConstructionRequest &request,
    FamilyConstructionResult &out) const {
  if (mlir::failed(constructRVVFormulaPlansForVariant(
          request.getVariant(), request.getCapabilities())))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "bound RVV formula construction rejected the selected variant");

  llvm::Expected<weft::rvv::WithVLOp> boundary =
      findSelectedRVVSelectedBodyBoundary(request.getVariant());
  bool hasPreRealizedBody =
      variantContainsPreRealizedRVVSelectedBody(request.getVariant());
  if (boundary) {
    if (hasPreRealizedBody)
      return makeRVVPluginError(
          "family construction found both pre-realized and final RVV bodies");
    if (mlir::failed(
            validateRVVConstructedTypedBody(boundary->getOperation())))
      return makeRVVPluginError(
          "bound RVV construction rejected its exact typed body");
    out = FamilyConstructionResult::getFinalBody(boundary->getOperation());
    return llvm::Error::success();
  }

  llvm::Error boundaryError = boundary.takeError();
  if (!hasPreRealizedBody)
    return boundaryError;
  llvm::consumeError(std::move(boundaryError));

  mlir::OpBuilder builder(request.getModule().getContext());
  builder.setInsertionPointToEnd(&request.getVariant().getBody().front());
  VariantLoweringBoundaryRequest bodyRequest(
      request.getVariant(), request.getKernel(), request.getProblem(),
      request.getCapabilities(), request.getRole(), builder, nullptr);
  llvm::Expected<weft::rvv::WithVLOp> realized =
      realizePreRealizedRVVSelectedBody(bodyRequest);
  if (!realized)
    return realized.takeError();
  if (mlir::failed(validateRVVConstructedTypedBody(realized->getOperation())))
    return makeRVVPluginError(
        "bound RVV construction rejected its exact realized typed body");
  out = FamilyConstructionResult::getFinalBody(realized->getOperation());
  return llvm::Error::success();
}

void RVVExtensionPlugin::collectFormulaDescriptors(
    llvm::SmallVectorImpl<FormulaDescriptor> &out) const {
  auto addDecisiveFields = [](FormulaAxisDescriptor &axis,
                              std::initializer_list<llvm::StringRef> fields) {
    for (llvm::StringRef field : fields)
      axis.addConsumedField(field);
  };
  auto makeDescriptor = [&](llvm::StringRef id, llvm::StringRef domain,
                            FormulaResultKind resultKind,
                            FormulaConstructionStrength strength,
                            llvm::StringRef geometryType,
                            std::initializer_list<llvm::StringRef> gFields,
                            FormulaAxisUse capabilityUse,
                            llvm::StringRef capabilityType,
                            std::initializer_list<llvm::StringRef> cFields,
                            FormulaAxisUse staticContextUse,
                            llvm::StringRef staticContextType) {
    FormulaDescriptor descriptor(id, kRVVPluginName, domain, resultKind,
                                 strength);
    descriptor.getGeometryAxis().set(FormulaAxisUse::Decisive, geometryType);
    addDecisiveFields(descriptor.getGeometryAxis(), gFields);
    descriptor.getCapabilityAxis().set(capabilityUse, capabilityType);
    addDecisiveFields(descriptor.getCapabilityAxis(), cFields);
    descriptor.getStaticContextAxis().set(staticContextUse,
                                          staticContextType);
    return descriptor;
  };

  FormulaDescriptor variant = makeDescriptor(
      formula::kVariantConstruction, "operator/rvv-variant",
      FormulaResultKind::CandidateSet,
      FormulaConstructionStrength::ConstructedWeak,
      "RVVVariantApplicabilityGeometry", {},
      FormulaAxisUse::Decisive, "RVVCapabilityProjection", {"rvv-available"},
      FormulaAxisUse::HonestNull, "RVVVariantNoStaticContext");
  variant.getGeometryAxis().set(FormulaAxisUse::HonestNull,
                                "RVVVariantApplicabilityGeometry");
  variant.addSemanticCase("explicit-typed-rvv-input-required");
  variant.addSemanticCase("rvv-capability-unavailable");
  variant.addProductionEntry("plugin:variant-applicability");
  out.push_back(std::move(variant));

  FormulaDescriptor cost = makeDescriptor(
      formula::kVariantAnalyticPrior, "operator/rvv-variant",
      FormulaResultKind::AnalyticPrior,
      FormulaConstructionStrength::ConstructedWeak,
      "RVVSelectedVariantFacts", {"typed-body-kind"},
      FormulaAxisUse::Decisive, "RVVCapabilityProjection", {"rvv-available"},
      FormulaAxisUse::HonestNull, "RVVCostNoStaticContext");
  cost.addSemanticCase("vector-paradigm-prior");
  cost.addProductionEntry("plugin:analytic-cost");
  out.push_back(std::move(cost));

  // Construction strength describes the complete production source
  // construction, not just a reusable schedule decision that it happens to
  // consume.  These front doors still own complete point-body builders and no
  // point-authority erasure witness currently reconstructs those bodies, so
  // they must remain honestly ConstructedWeak.  Internal schedule formulas
  // below retain their independently justified strength.
  FormulaDescriptor vector = makeDescriptor(
      formula::kVectorSourceConstruction, "operator/vector-elementwise",
      FormulaResultKind::TypedPlan,
      FormulaConstructionStrength::ConstructedWeak,
      "RVVVectorSourceGeometryFacts", {"opcode", "element-type", "predicate"},
      FormulaAxisUse::Decisive, "RVVCapabilityProjection",
      {"minimum-vlen", "supported-lmul"}, FormulaAxisUse::HonestNull,
      "RVVVectorNoStaticContext");
  vector.addSemanticCase("binary");
  vector.addSemanticCase("compare-select");
  vector.addSemanticCase("runtime-scalar-compare-select");
  addRVVVectorSourceFormulaProductionEntries(vector);
  out.push_back(std::move(vector));

  auto addSingleSource = [&](llvm::StringRef id, llvm::StringRef domain,
                             llvm::StringRef gType,
                             std::initializer_list<llvm::StringRef> gFields,
                             FormulaAxisUse capabilityUse,
                             llvm::StringRef cType,
                             std::initializer_list<llvm::StringRef> cFields,
                             llvm::StringRef semanticCase,
                             llvm::StringRef productionEntry,
                             FormulaConstructionStrength strength) {
    FormulaDescriptor descriptor = makeDescriptor(
        id, domain, FormulaResultKind::TypedPlan, strength, gType, gFields,
        capabilityUse, cType, cFields, FormulaAxisUse::HonestNull,
        "RVVNoStaticContext");
    descriptor.addSemanticCase(semanticCase);
    descriptor.addSemanticCase("unsupported-or-illegal");
    descriptor.addProductionEntry(productionEntry);
    out.push_back(std::move(descriptor));
  };

  addSingleSource(formula::kReductionSourceConstruction,
                  "operator/reduction", "RVVReductionGeometryFacts",
                  {"element-type", "reduction-kind", "shape"},
                  FormulaAxisUse::Decisive, "RVVCapabilityProjection",
                  {"supported-lmul", "vector-register-budget"},
                  "widening-dot-reduce", formula::kReductionSourceEntry,
                  FormulaConstructionStrength::ConstructedWeak);
  addSingleSource(formula::kDequantDotSourceConstruction,
                  "operator/dequant-dot", "RVVDequantDotGeometryFacts",
                  {"scale-kind", "element-type", "shape"},
                  FormulaAxisUse::Decisive, "RVVCapabilityProjection",
                  {"supported-lmul", "vector-register-budget"},
                  "widening-dot-reduce-with-scale",
                  formula::kDequantDotSourceEntry,
                  FormulaConstructionStrength::ConstructedWeak);
  addSingleSource(formula::kDequantizeRowConstruction,
                  "operator/dequantize-row", "DequantizeRowStreamFacts",
                  {"format", "qk", "layout", "decode-mechanism"},
                  FormulaAxisUse::Decisive,
                  "OptionalCodebookGatherCapabilityFacts",
                  {"minimum-vlen", "supported-sew", "supported-lmul"},
                  "typed-streaming-dequantize-row",
                  formula::kDequantizeRowSourceEntry,
                  FormulaConstructionStrength::ConstructedWeak);
  FormulaDescriptor quantize = makeDescriptor(
      formula::kQuantizeRowConstruction, "operator/quantize-row",
      FormulaResultKind::TypedPlan,
      FormulaConstructionStrength::ConstructedWeak,
      "QuantizeRowGeometryFacts", {"source-leaf"},
      FormulaAxisUse::HonestNull, "QuantizeRowNoCapabilityInput", {},
      FormulaAxisUse::HonestNull, "QuantizeRowNoStaticContext");
  quantize.addSemanticCase("q8-0");
  quantize.addSemanticCase("q8-1");
  quantize.addSemanticCase("q8-k");
  quantize.addSemanticCase("unsupported-or-illegal");
  quantize.addProductionEntry(formula::kQuantizeRowSourceEntry);
  out.push_back(std::move(quantize));
  addSingleSource(formula::kElementwiseConstruction,
                  "operator/elementwise", "ElementwiseStreamFacts",
                  {"operation", "shape", "element-type"},
                  FormulaAxisUse::HonestNull,
                  "ElementwiseNoCapabilityInput", {},
                  "typed-streaming-elementwise", formula::kElementwiseSourceEntry,
                  FormulaConstructionStrength::ConstructedWeak);
  addSingleSource(formula::kPackedI4DotConstruction,
                  "operator/packed-i4-dot", "PackedI4DotGeometryFacts",
                  {"qk", "carrier", "offset-binary-bias"},
                  FormulaAxisUse::Decisive, "RVVCapabilityProjection",
                  {"minimum-vlen", "supported-lmul"},
                  "packed-i4-offset-binary-dot",
                  formula::kPackedI4DotSourceEntry,
                  FormulaConstructionStrength::ConstructedWeak);
  addSingleSource(formula::kCodebookDotConstruction,
                  "operator/codebook-dot", "CodebookDotGeometryFacts",
                  {"qk", "codebook-entries", "layout"},
                  FormulaAxisUse::Decisive, "RVVCapabilityProjection",
                  {"minimum-vlen", "supported-lmul"},
                  "codebook-gather-dot", formula::kCodebookDotSourceEntry,
                  FormulaConstructionStrength::ConstructedWeak);

  FormulaDescriptor monolithic = makeDescriptor(
      formula::kMonolithicBlockDotConstruction, "operator/block-dot",
      FormulaResultKind::TypedPlan,
      FormulaConstructionStrength::ConstructedWeak,
      "MonolithicBlockDotGeometryFacts",
      {"format", "scale-model", "qk", "block-layout", "runtime-abi"},
      FormulaAxisUse::Decisive, "RVVCapabilityProjection",
      {"minimum-vlen", "supported-lmul", "vector-register-budget"},
      FormulaAxisUse::Decisive, "RVVBlockDotStaticContext");
  monolithic.getStaticContextAxis().addConsumedField("gemv-or-gemm-regime");
  for (const MonolithicBlockDotOpEntry &entry : monolithicBlockDotOpTable()) {
    monolithic.addSemanticCase(entry.kind);
    monolithic.addProductionEntry(entry.passArgument);
  }
  out.push_back(std::move(monolithic));

  FormulaDescriptor direct = makeDescriptor(
      formula::kLowerQuantContractionConstruction, "operator/contraction",
      FormulaResultKind::CandidateSet,
      FormulaConstructionStrength::ConstructedWeak,
      "LowerQuantContractionGeometryFacts",
      {"weight-format", "activation-format", "shape", "layout"},
      FormulaAxisUse::Decisive, "RVVCapabilityProjection",
      {"minimum-vlen", "supported-lmul", "vector-register-budget"},
      FormulaAxisUse::Decisive, "ContractionStaticRegime");
  direct.getStaticContextAxis().addConsumedField("gemv-or-gemm");
  direct.addSemanticCase("direct");
  direct.addSemanticCase("dequantize-first");
  direct.addSemanticCase("repack-gemv");
  direct.addSemanticCase("repack-gemm");
  direct.addSemanticCase("unsupported-reject");
  direct.addProductionEntry(formula::kLowerQuantContractionDirectEntry);
  out.push_back(std::move(direct));

  auto addInternal = [&](llvm::StringRef id, llvm::StringRef domain,
                         FormulaResultKind kind, llvm::StringRef gType,
                         std::initializer_list<llvm::StringRef> gFields,
                         FormulaAxisUse cUse, llvm::StringRef cType,
                         std::initializer_list<llvm::StringRef> cFields,
                         FormulaAxisUse omegaUse, llvm::StringRef omegaType,
                         std::initializer_list<llvm::StringRef> omegaFields,
                         std::initializer_list<llvm::StringRef> cases,
                         llvm::StringRef entry,
                         FormulaConstructionStrength strength) {
    FormulaDescriptor descriptor = makeDescriptor(
        id, domain, kind, strength, gType, gFields, cUse, cType, cFields,
        omegaUse, omegaType);
    addDecisiveFields(descriptor.getStaticContextAxis(), omegaFields);
    for (llvm::StringRef semanticCase : cases)
      descriptor.addSemanticCase(semanticCase);
    descriptor.addProductionEntry(entry);
    out.push_back(std::move(descriptor));
  };
  FormulaDescriptor flat = makeDescriptor(
      formula::kFlatBlockDotPlan, "operator/block-dot",
      FormulaResultKind::TypedPlan, FormulaConstructionStrength::ConstructedWeak,
      "RVVFlatBlockDotGeometryFacts",
      {"weight-encoding", "weight-scale-encoding", "has-min-term",
       "requires-offset-bias", "qk", "sub-block-length",
       "weight-quant-offset", "activation-quant-offset"},
      FormulaAxisUse::HonestNull, "RVVFlatBlockDotNoCapabilityInput", {},
      FormulaAxisUse::HonestNull, "RVVFlatBlockDotNoStaticContext");
  for (const RVVFlatBlockDotFormulaCase &semanticCase :
       getRVVFlatBlockDotFormulaCases())
    flat.addSemanticCase(semanticCase.semanticCase);
  flat.addSemanticCase("unsupported-or-illegal");
  flat.addProductionEntry("internal:flat-block-dot-plan");
  out.push_back(std::move(flat));

  FormulaDescriptor sourceSchedule = makeDescriptor(
      formula::kSourceScheduleFormula, "operator/source-schedule",
      FormulaResultKind::ResourceSchedule, FormulaConstructionStrength::Strong,
      "RVVSourceScheduleGeometryFacts",
      {"mechanism", "sew", "block-length", "candidate-lmuls"},
      FormulaAxisUse::Decisive, "RVVSourceScheduleCapabilityFacts",
      {"minimum-vlen", "vector-register-budget"}, FormulaAxisUse::HonestNull,
      "RVVSourceScheduleNoStaticContext");
  for (const RVVSourceScheduleFormulaCase &semanticCase :
       getRVVSourceScheduleFormulaCases())
    sourceSchedule.addSemanticCase(semanticCase.semanticCase);
  sourceSchedule.addSemanticCase("unsupported-or-illegal");
  sourceSchedule.addProductionEntry("internal:source-schedule-formula");
  out.push_back(std::move(sourceSchedule));

  FormulaDescriptor composite = makeDescriptor(
      formula::kCompositeGatherMAccScatterPlan,
      "operator/selected-body/composite", FormulaResultKind::TypedPlan,
      FormulaConstructionStrength::ConstructedWeak,
      "RVVCompositeGatherMAccScatterGeometryFacts",
      {"sew", "lmul", "policy", "predicate-kind", "index-eew", "offset-unit"},
      FormulaAxisUse::Decisive,
      "RVVCompositeGatherMAccScatterCapabilityFacts", {"supports-typed-config"},
      FormulaAxisUse::HonestNull,
      "RVVCompositeGatherMAccScatterNoStaticContext");
  composite.addSemanticCase("gather-macc-scatter");
  composite.addSemanticCase("unsupported-or-illegal");
  composite.addProductionEntry("internal:composite-gather-macc-scatter-plan");
  out.push_back(std::move(composite));
  auto addDequantPlan =
      [&](llvm::StringRef id, llvm::StringRef gType,
          std::initializer_list<llvm::StringRef> gFields,
          FormulaAxisUse cUse, llvm::StringRef cType,
          std::initializer_list<llvm::StringRef> cFields,
          std::initializer_list<llvm::StringRef> cases,
          llvm::StringRef entry) {
        addInternal(id, "operator/dequantize-row",
                    FormulaResultKind::TypedPlan, gType, gFields, cUse,
                    cType, cFields, FormulaAxisUse::HonestNull,
                    "DequantNoStaticContext", {}, cases, entry,
                    FormulaConstructionStrength::ConstructedWeak);
      };
  addDequantPlan(formula::kDequantInt8ScalePlan,
                 "Int8ScaleGeometryFacts",
                 {"qk", "block-stride", "scale-offset", "quant-offset"},
                 FormulaAxisUse::HonestNull,
                 "DequantNoCapabilityInput", {}, {"q8_0"},
                 "internal:dequant-int8-scale-plan");
  addDequantPlan(formula::kDequantNibblePlan,
                 "NibbleDecodeGeometryFacts",
                 {"qk", "block-stride", "scale-offset", "quant-offset",
                  "nibble-bias", "min-offset", "high-bit-offset"},
                 FormulaAxisUse::HonestNull,
                 "NibbleDecodeNoCapabilityInput", {},
                 {"q4_0", "q4_1", "q5_0", "q5_1", "q4_synth"},
                 "internal:dequant-nibble-plan");
  addDequantPlan(formula::kDequantBinarySignPlan,
                 "BinarySignGeometryFacts",
                 {"qk", "block-stride", "scale-offset", "quant-offset"},
                 FormulaAxisUse::HonestNull,
                 "DequantNoCapabilityInput", {}, {"q1_0"},
                 "internal:dequant-binary-sign-plan");
  addDequantPlan(formula::kDequantKQuantPlan,
                 "KQuantScaleMinGeometryFacts",
                 {"scale-model", "qk", "block-stride", "scale-offset",
                  "quant-offset"},
                 FormulaAxisUse::HonestNull,
                 "DequantNoCapabilityInput", {},
                 {"q2_K", "q3_K", "q4_K", "q5_K", "q6_K"},
                 "internal:dequant-kquant-plan");
  addDequantPlan(formula::kDequantCodebookPlan,
                 "CodebookGatherGeometryFacts",
                 {"scale-model", "qk", "block-stride", "scale-offset",
                  "quant-offset"},
                 FormulaAxisUse::Decisive,
                 "CodebookGatherCapabilityFacts",
                 {"minimum-vlen", "supported-sew", "supported-lmul"},
                 {"iq4_nl", "iq4_xs", "mxfp4", "nvfp4"},
                 "internal:dequant-codebook-plan");
  addDequantPlan(formula::kDequantGridPlan,
                 "GridLookupGeometryFacts",
                 {"grid-leaf", "entry-lanes"},
                 FormulaAxisUse::HonestNull,
                 "DequantNoCapabilityInput", {},
                 {"iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s"},
                 "internal:dequant-grid-plan");
  addDequantPlan(formula::kDequantTernaryPlan,
                 "TernaryDecodeGeometryFacts",
                 {"ternary-leaf", "entry-lanes"},
                 FormulaAxisUse::HonestNull,
                 "DequantNoCapabilityInput", {},
                 {"iq1_s", "iq1_m", "tq1_0", "tq2_0"},
                 "internal:dequant-ternary-plan");
  {
    FormulaDescriptor schedule = makeDescriptor(
        formula::kScheduleFormula, "operator/block-dot",
        FormulaResultKind::ResourceSchedule,
        FormulaConstructionStrength::Strong, "RVVScheduleGeometryFacts",
        {"kernel-key"}, FormulaAxisUse::Decisive,
        "RVVScheduleCapabilityFacts",
        {"minimum-vlen", "vector-register-budget"},
        FormulaAxisUse::HonestNull, "RVVScheduleNoStaticContext");
    for (llvm::StringRef kernelKey : getRVVScheduleFormulaKernelKeys())
      schedule.addSemanticCase(kernelKey);
    schedule.addProductionEntry("internal:schedule-formula-registry");
    out.push_back(std::move(schedule));
  }
  addInternal(formula::kLowPrecisionResourceSchedule,
              "operator/low-precision-resource",
              FormulaResultKind::ResourceSchedule,
              "RVVLowPrecisionResourceGeometryFacts",
              {"operation", "operand-encoding", "typed-widths", "policy"},
              FormulaAxisUse::Decisive, "RVVCapabilityProjection",
              {"vector-register-budget"}, FormulaAxisUse::HonestNull,
              "RVVLowPrecisionResourceNoStaticContext", {},
              {"deferred-wide", "grouped-narrow", "packed-i4-narrow"},
              "internal:selected-body-low-precision-formula",
              FormulaConstructionStrength::Strong);
  addInternal(formula::kDotReduceResourceSchedule,
              "operator/widening-dot-reduce",
              FormulaResultKind::ResourceSchedule,
              "RVVDotReduceScheduleGeometryFacts",
              {"source-width", "result-width"}, FormulaAxisUse::Decisive,
              "RVVCapabilityProjection", {"vector-register-budget"},
              FormulaAxisUse::Decisive, "RVVDotReduceScheduleContext",
              {"explicit-structure"},
              {"per-iteration", "deferred-accumulate"},
              "internal:selected-body-dot-reduce-formula",
              FormulaConstructionStrength::Strong);
  addInternal(formula::kStandaloneDequantSchedule,
              "operator/standalone-dequantize",
              FormulaResultKind::ResourceSchedule,
              "RVVStandaloneDequantGeometryFacts",
              {"source-width", "result-width", "dequant-relation"},
              FormulaAxisUse::HonestNull,
              "RVVStandaloneDequantNoCapabilityInput", {},
              FormulaAxisUse::HonestNull,
              "RVVStandaloneDequantNoStaticContext", {}, {"unroll-2"},
              "direct:selected-body-standalone-dequant",
              FormulaConstructionStrength::Strong);
  addInternal(formula::kRepackSchedule, "operator/repack",
              FormulaResultKind::ResourceSchedule, "RepackScheduleGeometry",
              {"fold-model", "strides", "qk", "interleave", "half-lanes",
               "integer-core-lmul"}, FormulaAxisUse::Decisive,
              "RVVCapabilityProjection",
              {"minimum-vlen", "vector-register-budget"},
              FormulaAxisUse::Decisive, "RepackScheduleStaticContext",
              {"operation-regime"}, {"loop-order", "main-term-form"},
              "internal:repack-schedule-formula",
              FormulaConstructionStrength::Strong);
  addInternal(formula::kRepackAccumulatorLMUL, "operator/repack",
              FormulaResultKind::CandidateSet,
              "RepackAccumulatorGeometryFacts", {"weight-interleave"},
              FormulaAxisUse::Decisive, "RepackAccumulatorCapabilityFacts",
              {"fractional-lmul", "half-lanes", "vector-register-budget"},
              FormulaAxisUse::HonestNull, "RepackFormulaNoStaticContext", {},
              {"mf2-legal", "m1-only", "empty-legal-set"},
              "internal:repack-accumulator-lmul",
              FormulaConstructionStrength::Strong);
  addInternal(formula::kContractionAlgorithm, "operator/contraction",
              FormulaResultKind::CandidateSet,
              "ContractionOpponentGeometryFacts",
              {"weight-format", "activation-format", "shape"},
              FormulaAxisUse::Decisive, "RVVCapabilityProjection",
              {"minimum-vlen", "supported-lmul"}, FormulaAxisUse::Decisive,
              "ContractionStaticRegime", {"gemv-or-gemm"},
              {"direct", "dequantize-first", "repack"},
              "internal:contraction-algorithm",
              FormulaConstructionStrength::Strong);
  FormulaDescriptor realization = makeDescriptor(
      formula::kSelectedBodyRealization, "operator/selected-body",
      FormulaResultKind::DeterministicConstruction,
      FormulaConstructionStrength::ConstructedWeak, "SelectedRVVBodyFacts",
      {"selected-body-kind", "typed-plan"}, FormulaAxisUse::Decisive,
      "RVVCapabilityProjection", {"supported-lmul", "isa-features"},
      FormulaAxisUse::HonestNull, "RealizationNoStaticContext");
  for (const RVVSelectedBodyRealizationOwner &owner :
       getRVVSelectedBodyRealizationOwners())
    realization.addSemanticCase(owner.familyName);
  realization.addProductionEntry("internal:selected-body-realization");
  realization.addProductionEntry("construction:rvv-final-typed-body");
  out.push_back(std::move(realization));
}

llvm::Error RVVExtensionPlugin::registerSourceFrontDoorPasses(
    const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) const {
  if (llvm::Error error = registerRVVVectorSourceFrontDoorFamilyPasses(
          kRVVPluginName, registry, out))
    return error;
  if (llvm::Error error = rvv::registerRVVReductionSourceFrontDoorPasses(
          kRVVPluginName, registry, out))
    return error;
  if (llvm::Error error = rvv::registerRVVDequantDotSourceFrontDoorPasses(
          kRVVPluginName, registry, out))
    return error;
  if (llvm::Error error = rvv::registerRVVDequantizeRowStreamFrontDoorPasses(
          kRVVPluginName, registry, out))
    return error;
  if (llvm::Error error = rvv::registerRVVQuantizeRowStreamFrontDoorPasses(
          kRVVPluginName, registry, out))
    return error;
  if (llvm::Error error = rvv::registerRVVElementwiseStreamFrontDoorPasses(
          kRVVPluginName, registry, out))
    return error;
  if (llvm::Error error = rvv::registerRVVPackedI4DotSourceFrontDoorPasses(
          kRVVPluginName, registry, out))
    return error;
  if (llvm::Error error = rvv::registerRVVCodebookDotSourceFrontDoorPasses(
          kRVVPluginName, registry, out))
    return error;
  return rvv::registerRVVMonolithicBlockDotSourceFrontDoorPasses(kRVVPluginName,
                                                                 registry, out);
}

bool RVVExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return hasAvailableRVVCapability(request);
}

llvm::Error RVVExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  (void)request;
  (void)out;
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  out.addRecoverableDecline(
      kRVVPluginName,
      "RVV proposal requires explicit typed weft_rvv extension-family IR "
      "before selecting an RVV variant");
  return llvm::Error::success();
}

llvm::Expected<support::TargetCapabilitySet>
RVVExtensionPlugin::buildTargetCapabilitiesFromProbeFacts(
    mlir::MLIRContext &context, const RVVProbeCapabilityFacts &facts) const {
  return buildRVVTargetCapabilitiesFromProbeFacts(context, facts);
}

llvm::Error RVVExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  weft::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVPluginError(
        "legality verification requires a materialized weft.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVPluginError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  llvm::Expected<rvv::RVVSelectedTargetCapabilityFacts> targetCapabilityFacts =
      rvv::collectRVVSelectedTargetCapabilityFacts(
          variant, request.getCapabilities(), "RVV variant legality");
  if (!targetCapabilityFacts)
    return targetCapabilityFacts.takeError();

  return requireExplicitTypedRVVBody(variant);
}

llvm::Error
RVVExtensionPlugin::estimateVariantCost(const VariantCostRequest &request,
                                        VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "cost estimation requires a materialized weft.exec.variant");
  if (llvm::Error error = requireExplicitTypedRVVBody(request.getVariant()))
    return error;

  // Capability-DERIVED cost (SEL-1 exec-level capability prior). Anchor the
  // vector-paradigm base on an available RVV isa-vector capability FACT consulted
  // from the target set instead of returning a capability-blind literal. A
  // materialized/typed RVV variant is always backed by this fact (legality
  // enforces it), so fail closed otherwise; consulting it here makes the
  // cross-paradigm score auditable (capability-fact -> score).
  const support::CapabilityDescriptor *vectorCapability =
      request.getCapabilities().lookupProviderByID(rvv::getRVVCapabilityID());
  if (!vectorCapability || !vectorCapability->isAvailable())
    return makeRVVPluginError(
        "RVV cost estimation requires an available RVV isa-vector capability id "
        "'rvv'");

  out = VariantCostEstimate();
  out.setScore(kRVVVectorBaseCost);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kRVVPluginName);
  out.setFormulaID(formula::kVariantAnalyticPrior);
  out.setVariantSymbol(request.getVariant().getSymName());
  out.setExplanation("explicit typed RVV variant body; vector-paradigm base cost "
                     "DERIVED from the available RVV isa-vector capability fact; "
                     "no runtime performance claim");
  out.setPolicy("plugin-local typed RVV extension-family IR; cost anchored on the "
                "available RVV isa-vector capability fact");
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "emission readiness requires a materialized weft.exec.variant");
  if (!request.getKernel())
    return makeRVVPluginError(
        "emission readiness requires an enclosing weft.exec.kernel");

  auto selectedBoundary = llvm::dyn_cast_if_present<weft::rvv::WithVLOp>(
      request.getConstructedOperation());
  if (!selectedBoundary) {
    out = VariantEmissionStatus::getUnsupported(
        kRVVPluginName, request.getVariant().getSymName(),
        "artifact query requires the exact weft_rvv.with_vl operation returned "
        "by family construction");
    return llvm::Error::success();
  }
  VariantLoweringBoundaryValidationRequest boundaryRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), selectedBoundary.getOperation());
  if (llvm::Error error =
          validateSelectedRVVSelectedBodyBoundary(boundaryRequest)) {
    std::string diagnostic = llvm::toString(std::move(error));
    out = VariantEmissionStatus::getUnsupported(
        kRVVPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }

  llvm::Expected<RVVArtifactContract> artifact =
      deriveRVVArtifactContract(selectedBoundary);
  if (!artifact) {
    std::string diagnostic = llvm::toString(artifact.takeError());
    out = VariantEmissionStatus::getUnsupported(
        kRVVPluginName, request.getVariant().getSymName(), diagnostic);
    return llvm::Error::success();
  }

  out = VariantEmissionStatus::getSupported(
      kRVVPluginName, request.getVariant().getSymName(),
      getRVVExactBodyArtifactRouteID());
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "emission planning requires a materialized weft.exec.variant");

  if (!request.getKernel())
    return makeRVVPluginError(
        "emission planning requires an enclosing weft.exec.kernel");

  auto selectedBoundary = llvm::dyn_cast_if_present<weft::rvv::WithVLOp>(
      request.getConstructedOperation());
  if (!selectedBoundary)
    return makeRVVPluginError(
        "emission planning requires the exact weft_rvv.with_vl operation "
        "returned by family construction");
  VariantLoweringBoundaryValidationRequest boundaryRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), selectedBoundary.getOperation());
  if (llvm::Error error =
          validateSelectedRVVSelectedBodyBoundary(boundaryRequest))
    return error;

  llvm::Expected<RVVArtifactContract> artifact =
      deriveRVVArtifactContract(selectedBoundary);
  if (!artifact)
    return artifact.takeError();
  out = VariantEmissionPlan::getSupported(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      getRVVExactBodyEmissionKind(), getRVVExactBodyArtifactRouteID(),
      getRVVExactBodyRuntimeABIName(), getRVVExactBodyArtifactKind(),
      "RVV artifact lowering consumes the exact family-constructed typed body "
      "and its SSA-bound runtime ABI, then mechanically applies the registered "
      "RVV DialectConversion");
  out.setRuntimeABIKind(getRVVExactBodyRuntimeABIKind());
  out.setRuntimeABIName(getRVVExactBodyRuntimeABIName());
  out.setRuntimeGlueRole(getRVVExactBodyRuntimeGlueRole());
  out.setLoweringBoundaryOpName(getRVVExactBodyLoweringBoundaryOpName());
  out.addRuntimeABIParameters(artifact->runtimeABIParameters);
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "lowering-boundary materialization requires a materialized "
        "weft.exec.variant");
  if (!request.getKernel())
    return makeRVVPluginError(
        "lowering-boundary materialization requires an enclosing "
        "weft.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getProblem(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality))
    return error;

  auto boundary = llvm::dyn_cast_if_present<weft::rvv::WithVLOp>(
      request.getConstructedOperation());
  if (!boundary)
    return makeRVVPluginError(
        "selected RVV boundary exposure requires the exact weft_rvv.with_vl "
        "root returned by family construction");
  if (variantContainsPreRealizedRVVSelectedBody(request.getVariant())) {
    return makeRVVPluginError(
        "selected RVV final body exposure found a leftover pre-realized body");
  }

  VariantLoweringBoundaryValidationRequest validationRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), boundary.getOperation());
  if (llvm::Error error =
          validateSelectedRVVSelectedBodyBoundary(validationRequest))
    return error;

  out = VariantLoweringBoundaryResult::getMaterialized(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      boundary.getOperation());
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  return validateSelectedRVVSelectedBodyBoundary(request);
}

llvm::Error RVVExtensionPlugin::configureTargetSupportExtensionBundle(
    ExtensionBundle &bundle) const {
  bundle.addRequiredDialectName("weft_rvv");
  return target::rvv::configureRVVTargetSupportExtensionBundle(bundle);
}

llvm::Error RVVExtensionPlugin::registerTargetSupportTranslateRoutes(
    target::TargetTranslateRouteRegistry &registry) const {
  return target::rvv::registerRVVTargetSupportTargetTranslateRoutes(registry);
}

} // namespace rvv

llvm::Error registerRVVExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinRVVExtensionPlugin());
}

} // namespace weft::plugin
