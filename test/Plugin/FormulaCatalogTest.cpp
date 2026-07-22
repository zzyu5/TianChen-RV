#include "Weft/Plugin/BuiltinExtensionPlugins.h"
#include "Weft/Conversion/EmitC/BackendEmissionRegistry.h"
#include "Weft/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "Weft/Plugin/ExtensionBundle.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVCompositeGatherMAccScatterFormula.h"
#include "Weft/Plugin/RVV/RVVDequantFormula.h"
#include "Weft/Plugin/RVV/RVVFlatBlockDotFormula.h"
#include "Weft/Plugin/RVV/RVVFormulaCatalog.h"
#include "Weft/Plugin/RVV/RVVQuantizeFormula.h"
#include "Weft/Plugin/RVV/RVVScheduleFormula.h"
#include "Weft/Plugin/RVV/RVVSourceScheduleFormula.h"
#include "Weft/Plugin/RVV/RVVSelectedBodyRealization.h"
#include "Weft/Plugin/Scalar/ScalarFormulaConstruction.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

using namespace weft::plugin;

namespace {

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

int expectSuccess(llvm::Error error, llvm::Twine context) {
  if (!error)
    return 0;
  return fail(context + ": " + llvm::toString(std::move(error)));
}

} // namespace

int main() {
  ExtensionBundleRegistry bundles;
  ExtensionPluginRegistry plugins;
  if (int result = expectSuccess(
          registerBuiltinExtensionBundlePlugins(bundles, plugins),
          "register builtin plugin catalog"))
    return result;

  llvm::SmallVector<FormulaDescriptor, 48> formulas;
  if (int result = expectSuccess(plugins.collectFormulaCatalog(formulas),
                                 "collect formula catalog"))
    return result;

  llvm::SmallVector<SourceFrontDoorPassRegistration, 48> frontDoors;
  if (int result = expectSuccess(
          plugins.collectSourceFrontDoorPasses(frontDoors),
          "collect source front-door registrations"))
    return result;

  llvm::SmallVector<CanonicalProblemDescriptor, 48> problems;
  if (int result = expectSuccess(
          plugins.collectCanonicalProblemCatalog(problems),
          "collect canonical problem catalog"))
    return result;
  if (problems.size() != frontDoors.size())
    return fail("canonical problem catalog must cover every source front door "
                "exactly once");
  llvm::StringSet<> problemEntries;
  for (const CanonicalProblemDescriptor &problem : problems) {
    if (!problemEntries.insert(problem.getSourceEntry()).second)
      return fail(llvm::Twine("duplicate canonical problem entry: ") +
                  problem.getSourceEntry());
    if (problem.getOwnerPlugin().empty() || problem.getFormulaID().empty() ||
        problem.getOperatorDomain().empty() || problem.getGeometryType().empty() ||
        problem.getStaticContextType().empty())
      return fail(llvm::Twine("canonical problem entry is missing typed P=(S,g,"
                              "omega) ownership: ") +
                  problem.getSourceEntry());
  }

  llvm::StringMap<const FormulaDescriptor *> byID;
  llvm::StringMap<unsigned> scopedProductionEntryOwners;
  llvm::StringMap<unsigned> globalProductionEntryOwners;
  llvm::StringSet<> catalogConstructionEntries;
  std::size_t declaredDependencyFields = 0;
  std::size_t declaredFormulaCases = 0;
  for (const FormulaDescriptor &formula : formulas) {
    if (!byID.try_emplace(formula.getID(), &formula).second)
      return fail(llvm::Twine("duplicate formula id escaped registry: ") +
                  formula.getID());
    if (!plugins.lookupPlugin(formula.getOwnerPlugin()))
      return fail(llvm::Twine("formula has unregistered owner: ") +
                  formula.getID());
    for (const std::string &entry : formula.getProductionEntries()) {
      std::string key =
          (formula.getOwnerPlugin() + llvm::Twine("\n") + entry).str();
      ++scopedProductionEntryOwners[key];
      ++globalProductionEntryOwners[entry];
      if (llvm::StringRef(entry).starts_with("construction:"))
        catalogConstructionEntries.insert(entry);
    }
    auto countDecisive = [&](const FormulaAxisDescriptor &axis) {
      if (axis.getUse() == FormulaAxisUse::Decisive)
        declaredDependencyFields += axis.getConsumedFields().size();
    };
    countDecisive(formula.getGeometryAxis());
    countDecisive(formula.getCapabilityAxis());
    countDecisive(formula.getStaticContextAxis());
    declaredFormulaCases += formula.getSemanticCases().size();
  }

  weft::conversion::emitc::BackendEmissionRegistry backendRegistry;
  weft::conversion::emitc::registerBuiltinBackendEmitters(backendRegistry);
  llvm::StringSet<> backendNames;
  for (const weft::conversion::emitc::TypedBackendEmissionDriver *backend :
       backendRegistry.getRegisteredBackends()) {
    if (!backendNames.insert(backend->getBackendName()).second)
      return fail(llvm::Twine("duplicate backend identity: ") +
                  backend->getBackendName());
  }
  if (catalogConstructionEntries.empty())
    return fail("formula catalog exposes no artifact-neutral construction "
                "entries");
  for (const auto &entry : catalogConstructionEntries)
    if (globalProductionEntryOwners.lookup(entry.getKey()) != 1)
      return fail(llvm::Twine("construction entry must have exactly one "
                              "family formula owner: ") +
                  entry.getKey());
  for (llvm::StringRef requiredBackend :
       {"rvv", "demo", "toy", "template", "tensorext_lite", "ime",
        "scalar"})
    if (!backendNames.contains(requiredBackend))
      return fail(llvm::Twine("production backend missing from registry: ") +
                  requiredBackend);
  if (backendNames.contains("offload"))
    return fail("unsupported Offload placeholder must not become an emitter");

  llvm::StringSet<> registeredFrontDoors;
  for (const SourceFrontDoorPassRegistration &frontDoor : frontDoors) {
    if (!registeredFrontDoors.insert(frontDoor.getArgument()).second)
      return fail(llvm::Twine("duplicate registered front door: ") +
                  frontDoor.getArgument());
    auto formula = byID.find(frontDoor.getFormulaID());
    if (formula == byID.end())
      return fail(llvm::Twine("front door has no catalog formula: ") +
                  frontDoor.getArgument());
    if (formula->second->getOwnerPlugin() != frontDoor.getOwnerPlugin())
      return fail(llvm::Twine("front door/formula owner mismatch: ") +
                  frontDoor.getArgument());
    std::string key =
        (frontDoor.getOwnerPlugin() + llvm::Twine("\n") +
         frontDoor.getArgument())
            .str();
    if (scopedProductionEntryOwners.lookup(key) != 1)
      return fail(llvm::Twine("front door must have exactly one formula owner: ") +
                  frontDoor.getArgument());
  }

  constexpr llvm::StringLiteral requiredRVVAuthorities[] = {
      rvv::formula_catalog::kDequantInt8ScalePlan,
      rvv::formula_catalog::kDequantNibblePlan,
      rvv::formula_catalog::kDequantBinarySignPlan,
      rvv::formula_catalog::kDequantKQuantPlan,
      rvv::formula_catalog::kDequantCodebookPlan,
      rvv::formula_catalog::kDequantGridPlan,
      rvv::formula_catalog::kDequantTernaryPlan,
      rvv::formula_catalog::kQuantizeRowConstruction,
      rvv::formula_catalog::kFlatBlockDotPlan,
      rvv::formula_catalog::kSourceScheduleFormula,
      rvv::formula_catalog::kCompositeGatherMAccScatterPlan,
      rvv::formula_catalog::kScheduleFormula,
      rvv::formula_catalog::kLowPrecisionResourceSchedule,
      rvv::formula_catalog::kRepackSchedule,
      rvv::formula_catalog::kRepackAccumulatorLMUL,
      rvv::formula_catalog::kContractionAlgorithm,
      rvv::formula_catalog::kSelectedBodyRealization,
      rvv::formula_catalog::kLowerQuantContractionConstruction,
  };
  for (llvm::StringRef id : requiredRVVAuthorities) {
    if (!byID.count(id))
      return fail(llvm::Twine("unclassified RVV construction authority: ") +
                  id);
  }

  struct ScalarFormulaExpectation {
    llvm::StringRef id;
    FormulaResultKind resultKind;
    llvm::StringRef constructionEntry;
  };
  const ScalarFormulaExpectation scalarFormulaExpectations[] = {
      {scalar::kScalarFallbackConstructionFormulaID,
       FormulaResultKind::CandidateSet,
       "construction:scalar-immediate-call-body"},
      {scalar::kScalarTernaryBlockDotFormulaID, FormulaResultKind::TypedPlan,
       "construction:scalar-packed-ternary-dot-body"},
      {scalar::kScalarQ40DequantizeRowFormulaID,
       FormulaResultKind::TypedPlan,
       "construction:scalar-packed-affine-dequant-body"},
  };
  for (const ScalarFormulaExpectation &expected : scalarFormulaExpectations) {
    const FormulaDescriptor *formula = byID.lookup(expected.id);
    if (!formula || formula->getOwnerPlugin() != "scalar-plugin" ||
        formula->getResultKind() != expected.resultKind ||
        formula->getConstructionStrength() !=
            FormulaConstructionStrength::ConstructedWeak ||
        formula->getCapabilityAxis().getUse() != FormulaAxisUse::Decisive ||
        formula->getStaticContextAxis().getUse() != FormulaAxisUse::HonestNull)
      return fail(llvm::Twine("Scalar formula catalog contract is incomplete: ") +
                  expected.id);
    if (!llvm::is_contained(formula->getProductionEntries(),
                            expected.constructionEntry.str()))
      return fail(llvm::Twine("Scalar formula lost typed-body construction entry: ") +
                  expected.id);
  }

  const FormulaDescriptor *flatFormula =
      byID.lookup(rvv::formula_catalog::kFlatBlockDotPlan);
  if (!flatFormula)
    return fail("flat block-dot formula descriptor is absent");
  constexpr llvm::StringLiteral expectedFlatCases[] = {
      "q8_0-q8_0", "q4_0-q8_0", "q4_1-q8_1", "q5_0-q8_0",
      "q5_1-q8_1", "iq4_nl-q8_0", "mxfp4-q8_0", "q1_0-q8_0",
      "nvfp4-q8_0", "unsupported-or-illegal"};
  llvm::StringSet<> expectedFlatCaseSet;
  for (llvm::StringRef semanticCase : expectedFlatCases)
    expectedFlatCaseSet.insert(semanticCase);
  llvm::StringSet<> actualFlatCaseSet;
  for (const std::string &semanticCase : flatFormula->getSemanticCases())
    actualFlatCaseSet.insert(semanticCase);
  if (actualFlatCaseSet.size() != expectedFlatCaseSet.size())
    return fail("flat formula catalog case count differs from the live semantic domain");
  for (llvm::StringRef semanticCase : expectedFlatCases)
    if (!actualFlatCaseSet.contains(semanticCase))
      return fail(llvm::Twine("flat formula catalog lost semantic case: ") +
                  semanticCase);
  if (flatFormula->getConstructionStrength() !=
      FormulaConstructionStrength::ConstructedWeak)
    return fail("flat block-dot catalog must retain its ConstructedWeak boundary");

  const FormulaDescriptor *sourceScheduleFormula =
      byID.lookup(rvv::formula_catalog::kSourceScheduleFormula);
  if (!sourceScheduleFormula ||
      sourceScheduleFormula->getConstructionStrength() !=
          FormulaConstructionStrength::Strong)
    return fail("source schedule formula descriptor is absent or weak");
  llvm::StringSet<> sourceScheduleCases;
  for (const std::string &semanticCase :
       sourceScheduleFormula->getSemanticCases())
    sourceScheduleCases.insert(semanticCase);
  for (const rvv::RVVSourceScheduleFormulaCase &semanticCase :
       rvv::getRVVSourceScheduleFormulaCases())
    if (!sourceScheduleCases.contains(semanticCase.semanticCase))
      return fail(llvm::Twine("source schedule catalog lost semantic case: ") +
                  semanticCase.semanticCase);
  if (!sourceScheduleCases.contains("unsupported-or-illegal") ||
      sourceScheduleCases.size() !=
          rvv::getRVVSourceScheduleFormulaCases().size() + 1)
    return fail("source schedule catalog has an unowned semantic case");

  const FormulaDescriptor *compositeFormula =
      byID.lookup(rvv::formula_catalog::kCompositeGatherMAccScatterPlan);
  if (!compositeFormula ||
      compositeFormula->getConstructionStrength() !=
          FormulaConstructionStrength::ConstructedWeak)
    return fail("composite formula descriptor must retain its weak boundary");
  if (compositeFormula->getSemanticCases().size() != 2)
    return fail("composite formula must expose applicable and reject cases");

  const FormulaDescriptor *scheduleFormula =
      byID.lookup(rvv::formula_catalog::kScheduleFormula);
  if (!scheduleFormula)
    return fail("RVV schedule formula descriptor is absent");
  llvm::StringSet<> cataloguedScheduleCases;
  for (const std::string &semanticCase : scheduleFormula->getSemanticCases())
    cataloguedScheduleCases.insert(semanticCase);
  for (llvm::StringRef kernelKey : rvv::getRVVScheduleFormulaKernelKeys()) {
    if (!cataloguedScheduleCases.contains(kernelKey) ||
        !rvv::lookupRVVScheduleFormula(kernelKey))
      return fail(llvm::Twine("schedule formula/case linkage missing for ") +
                  kernelKey);
  }
  if (cataloguedScheduleCases.size() !=
      rvv::getRVVScheduleFormulaKernelKeys().size())
    return fail("schedule catalog contains a case with no live formula owner");
  if (rvv::lookupRVVScheduleFormula("tq1_0"))
    return fail("emitter-inert tq1_0 schedule axis must not re-enter registry");

  const FormulaDescriptor *realization =
      byID.lookup(rvv::formula_catalog::kSelectedBodyRealization);
  if (!realization)
    return fail("selected-body realization formula is absent");
  if (realization->getSemanticCases().size() !=
      rvv::getRVVSelectedBodyRealizationOwners().size())
    return fail("selected-body formula must derive every semantic case from "
                "the live owner registry without hand-added exceptions");

  std::string directKey =
      (rvv::getRVVExtensionPluginName() + llvm::Twine("\n") +
       rvv::formula_catalog::kLowerQuantContractionDirectEntry)
          .str();
  if (scopedProductionEntryOwners.lookup(directKey) != 1)
    return fail("direct lower-quant contraction route must have one owner");

  auto requireCapabilityUse =
      [&](llvm::StringRef id, FormulaAxisUse expected) -> int {
    const FormulaDescriptor *descriptor = byID.lookup(id);
    if (!descriptor)
      return fail(llvm::Twine("missing dequant formula descriptor: ") + id);
    if (descriptor->getCapabilityAxis().getUse() != expected)
      return fail(llvm::Twine("dishonest dequant capability axis on ") + id);
    return 0;
  };
  constexpr llvm::StringLiteral honestNullDequant[] = {
      rvv::formula_catalog::kDequantInt8ScalePlan,
      rvv::formula_catalog::kDequantNibblePlan,
      rvv::formula_catalog::kDequantBinarySignPlan,
      rvv::formula_catalog::kDequantKQuantPlan,
      rvv::formula_catalog::kDequantGridPlan,
      rvv::formula_catalog::kDequantTernaryPlan,
  };
  for (llvm::StringRef id : honestNullDequant)
    if (int result = requireCapabilityUse(id, FormulaAxisUse::HonestNull))
      return result;
  if (int result = requireCapabilityUse(
          rvv::formula_catalog::kDequantCodebookPlan,
          FormulaAxisUse::Decisive))
    return result;

  const FormulaDescriptor *quantize =
      byID.lookup(rvv::formula_catalog::kQuantizeRowConstruction);
  if (!quantize ||
      quantize->getCapabilityAxis().getUse() != FormulaAxisUse::HonestNull ||
      quantize->getStaticContextAxis().getUse() != FormulaAxisUse::HonestNull ||
      quantize->getSemanticCases().size() != 4)
    return fail("quantize formula must expose three typed leaves, reject, and "
                "honest-null c/omega");
  struct QuantExpectation {
    ::weft::rvv::QuantizeRowLeaf leaf;
    std::int64_t qk;
    std::int64_t stride;
    std::int64_t quantOffset;
  };
  constexpr QuantExpectation quantExpectations[] = {
      {::weft::rvv::QuantizeRowLeaf::Q8_0, 32, 34, 2},
      {::weft::rvv::QuantizeRowLeaf::Q8_1, 32, 36, 4},
      {::weft::rvv::QuantizeRowLeaf::Q8_K, 256, 292, 4},
  };
  for (const QuantExpectation &expected : quantExpectations) {
    auto plan = rvv::constructQuantizeRowPlan(
        rvv::QuantizeRowGeometryFacts{expected.leaf},
        rvv::QuantizeRowNoCapabilityInput{},
        rvv::QuantizeRowNoStaticContext{});
    if (!plan || plan->leaf != expected.leaf || plan->qk != expected.qk ||
        plan->blockStride != expected.stride ||
        plan->scaleByteOffset != 0 ||
        plan->quantByteOffset != expected.quantOffset)
      return fail("quantize typed formula result does not match its semantic "
                  "leaf/layout case");
  }

  auto constructDequant =
      [&](llvm::StringRef format,
          std::optional<rvv::CodebookGatherCapabilityFacts> capability =
              std::nullopt)
      -> llvm::Expected<::weft::rvv::DequantizeRowConstruction> {
    auto facts = ::weft::rvv::lookupDequantizeRowStreamFacts(format);
    if (!facts)
      return rvv::makeDequantizeRowFormulaError(
          llvm::Twine("test has no geometry for ") + format);
    return rvv::constructDequantizeRowFormula(*facts, capability);
  };

  auto nibble = constructDequant("q4_0");
  if (!nibble)
    return fail(llvm::toString(nibble.takeError()));
  const auto *nibblePlan =
      std::get_if<::weft::NibbleDecodePlan>(&nibble->mechanismPlan);
  if (!nibblePlan || nibblePlan->stripLanes != 16 ||
      nibblePlan->nibbleBias != 8)
    return fail("dequant formula did not derive the nibble final plan");

  auto kquant = constructDequant("q2_K");
  if (!kquant)
    return fail(llvm::toString(kquant.takeError()));
  const auto *kquantPlan =
      std::get_if<::weft::KQuantScaleMinPlan>(&kquant->mechanismPlan);
  if (!kquantPlan || kquantPlan->minByteOffset != 82 ||
      kquantPlan->stripLanes != 16)
    return fail("dequant formula did not derive the K-quant final plan");

  auto grid = constructDequant("iq2_xs");
  if (!grid)
    return fail(llvm::toString(grid.takeError()));
  const auto *gridPlan =
      std::get_if<::weft::GridLookupPlan>(&grid->mechanismPlan);
  if (!gridPlan || gridPlan->leaf != ::weft::GridDecodeLeaf::Iq2Xs ||
      !gridPlan->hasEntryLanes || gridPlan->entryLanes <= 0)
    return fail("dequant formula did not derive the grid final plan");

  auto ternary = constructDequant("tq1_0");
  if (!ternary)
    return fail(llvm::toString(ternary.takeError()));
  const auto *ternaryPlan =
      std::get_if<::weft::TernaryDecodePlan>(&ternary->mechanismPlan);
  if (!ternaryPlan || ternaryPlan->leaf != ::weft::TernaryDecodeLeaf::Tq1_0)
    return fail("dequant formula did not derive the ternary final plan");

  auto flat = constructDequant("q8_0");
  auto binary = constructDequant("q1_0");
  if (!flat)
    return fail(llvm::toString(flat.takeError()));
  if (!binary)
    return fail(llvm::toString(binary.takeError()));
  if (!std::holds_alternative<std::monostate>(flat->mechanismPlan) ||
      !std::holds_alternative<std::monostate>(binary->mechanismPlan))
    return fail("flat deterministic dequant formulas must construct honest "
                "single-state bodies");

  rvv::CodebookGatherCapabilityFacts codebookCapability;
  codebookCapability.minimumVLEN = 128;
  codebookCapability.supportsSEW8 = true;
  codebookCapability.supportsSEW32 = true;
  codebookCapability.supportsMF2 = true;
  codebookCapability.supportsM1 = true;
  codebookCapability.supportsM2 = true;
  codebookCapability.supportsM4 = true;
  codebookCapability.supportsM8 = true;
  auto codebook = constructDequant("mxfp4", codebookCapability);
  if (!codebook)
    return fail(llvm::toString(codebook.takeError()));
  const auto *codebookPlan =
      std::get_if<::weft::CodebookGatherPlan>(&codebook->mechanismPlan);
  if (!codebookPlan || codebookPlan->loadLMUL != "m1" ||
      codebookPlan->stripLanes != 16 || codebookPlan->codebookEntries != 16)
    return fail("dequant formula did not consume capability c into the "
                "codebook final plan");

  for (const ExtensionPlugin *plugin : plugins.getAllPlugins()) {
    bool hasProposalOrApplicabilityOwner = false;
    bool hasCostOwner = false;
    for (const FormulaDescriptor &formula : formulas) {
      if (formula.getOwnerPlugin() != plugin->getName())
        continue;
      for (const std::string &entry : formula.getProductionEntries()) {
        hasProposalOrApplicabilityOwner |=
            entry == "plugin:variant-proposal" ||
            entry == "plugin:variant-applicability";
        hasCostOwner |= entry == "plugin:analytic-cost";
      }
    }
    if (!hasProposalOrApplicabilityOwner || !hasCostOwner)
      return fail(llvm::Twine(
                      "plugin lacks truthful proposal/applicability or cost "
                      "formula ownership: ") +
                  plugin->getName());
  }

  llvm::outs() << "formula catalog/registry integrity ok: " << plugins.size()
               << " plugins, " << formulas.size() << " formula authorities, "
               << frontDoors.size()
               << " registered source front doors / " << problems.size()
               << " canonical P=(S,g,omega) owners, " << backendNames.size()
               << " separate artifact backends; direct route and all entries "
                  "uniquely owned; "
               << declaredDependencyFields << " declared dependency fields and "
               << declaredFormulaCases << " declared formula cases\n";
  return 0;
}
