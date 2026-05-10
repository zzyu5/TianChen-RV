#include "TianChenRV/Plugin/RVV/RVVBinaryPlanning.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kLoweringDescriptorAttrName(
    "tcrv_rvv.lowering_descriptor");
constexpr llvm::StringLiteral kElementCountAttrName("tcrv_rvv.element_count");
constexpr llvm::StringLiteral kRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kRuntimeCallableCSourceArtifactKind(
    "runtime-callable-c-source");

constexpr llvm::StringLiteral kBoundarySelectedVectorShapeAttrName(
    "selected_vector_shape");
constexpr llvm::StringLiteral kBoundarySelectedVectorSEWAttrName(
    "selected_vector_sew");
constexpr llvm::StringLiteral kBoundarySelectedVectorLMULAttrName(
    "selected_vector_lmul");
constexpr llvm::StringLiteral kBoundarySelectedTailPolicyAttrName(
    "selected_tail_policy");
constexpr llvm::StringLiteral kBoundarySelectedMaskPolicyAttrName(
    "selected_mask_policy");
constexpr llvm::StringLiteral kBoundarySelectedVectorTypeAttrName(
    "selected_vector_type");
constexpr llvm::StringLiteral kBoundarySelectedVectorSuffixAttrName(
    "selected_vector_suffix");
constexpr llvm::StringLiteral kBoundarySelectedSetVLSuffixAttrName(
    "selected_setvl_suffix");

llvm::Error makeRVVBinaryPlanningError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV binary planning failed: ") + message,
      llvm::errc::invalid_argument);
}

bool containsForbiddenRVVPlanningText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key");
}

bool isSingleBoundedRVVPlanningText(llvm::StringRef value) {
  if (value.size() > 512)
    return false;

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return false;
    if (byte < 0x20 && character != '\t')
      return false;
  }
  return true;
}

llvm::Error validateRVVPlanningText(llvm::StringRef context,
                                    llvm::StringRef fieldName,
                                    llvm::StringRef value) {
  if (!isSingleBoundedRVVPlanningText(value))
    return makeRVVBinaryPlanningError(llvm::Twine(context) + " '" +
                                     fieldName +
                                     "' must be a bounded single-line fact");

  if (containsForbiddenRVVPlanningText(value))
    return makeRVVBinaryPlanningError(llvm::Twine(context) + " '" +
                                     fieldName +
                                     "' must not contain secret-like or "
                                     "raw-log text");

  return llvm::Error::success();
}

llvm::StringRef getDTypeDiagnosticSpelling(
    const target::rvv::RVVVectorShapeConfig &shape) {
  if (!shape.dtypeDiagnosticSpelling.empty())
    return shape.dtypeDiagnosticSpelling;
  return shape.dtypeID;
}

llvm::Expected<const target::rvv::RVVBinaryFamilyDescriptor *>
getRegisteredRVVBinaryFamily(
    const target::rvv::RVVBinaryFamilyDescriptor &family) {
  const target::rvv::RVVBinaryFamilyDescriptor *registeredFamily =
      target::rvv::lookupRVVBinaryFamilyByID(family.familyID);
  if (!registeredFamily ||
      registeredFamily->loweringDescriptor != family.loweringDescriptor ||
      registeredFamily->routeID != family.routeID ||
      registeredFamily->emissionKind != family.emissionKind ||
      registeredFamily->runtimeABI != family.runtimeABI ||
      registeredFamily->runtimeABIKind != family.runtimeABIKind ||
      registeredFamily->runtimeABIName != family.runtimeABIName ||
      registeredFamily->runtimeGlueRole != family.runtimeGlueRole) {
    return makeRVVBinaryPlanningError(
        llvm::Twine("family '") + family.familyID +
        "' must be one registered finite RVV binary family descriptor");
  }
  return registeredFamily;
}

std::string buildSupportedEmissionMessage(
    const target::rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("explicit RVV ") + family.dtypeID + " vector-" +
          family.arithmeticVerb +
          " microkernel C source export provides a library-style "
          "runtime-callable C ABI function for this selected path; any "
          "self-check main is an explicit harness export and is not the "
          "default artifact contract; this is not generic RVV lowering, "
          "runtime integration, arbitrary kernel emission, correctness, or "
          "performance evidence")
      .str();
}

} // namespace

llvm::StringRef RVVBinaryEmissionIdentity::getFamilyID() const {
  return family ? family->familyID : llvm::StringRef();
}

llvm::StringRef RVVBinaryEmissionIdentity::getEmissionKind() const {
  return family ? family->emissionKind : llvm::StringRef();
}

llvm::StringRef RVVBinaryEmissionIdentity::getEmissionPath() const {
  return emissionPath;
}

llvm::StringRef RVVBinaryEmissionIdentity::getRouteID() const {
  return family ? family->routeID : llvm::StringRef();
}

llvm::StringRef RVVBinaryEmissionIdentity::getArtifactKind() const {
  return getRVVBinaryRuntimeCallableCSourceArtifactKind();
}

llvm::StringRef RVVBinaryEmissionIdentity::getRuntimeABI() const {
  return family ? family->runtimeABI : llvm::StringRef();
}

llvm::StringRef RVVBinaryEmissionIdentity::getRuntimeABIKind() const {
  return family ? family->runtimeABIKind : llvm::StringRef();
}

llvm::StringRef RVVBinaryEmissionIdentity::getRuntimeABIName() const {
  return family ? family->runtimeABIName : llvm::StringRef();
}

llvm::StringRef RVVBinaryEmissionIdentity::getRuntimeGlueRole() const {
  return family ? family->runtimeGlueRole : llvm::StringRef();
}

llvm::StringRef RVVBinaryEmissionIdentity::getSupportedMessage() const {
  return supportedMessage;
}

llvm::StringRef RVVBinarySelectedPlan::getFamilyID() const {
  return family ? family->familyID : llvm::StringRef();
}

llvm::StringRef RVVBinarySelectedPlan::getDTypeID() const {
  return family ? family->dtypeID : llvm::StringRef();
}

llvm::StringRef RVVBinarySelectedPlan::getLoweringDescriptor() const {
  return family ? family->loweringDescriptor : llvm::StringRef();
}

llvm::StringRef RVVBinarySelectedPlan::getMicrokernelOpName() const {
  return family ? family->microkernelOpName : llvm::StringRef();
}

llvm::StringRef RVVBinarySelectedPlan::getArithmeticOpName() const {
  return family ? family->arithmeticOpName : llvm::StringRef();
}

llvm::StringRef RVVBinarySelectedPlan::getEmissionKind() const {
  return family ? family->emissionKind : llvm::StringRef();
}

llvm::StringRef RVVBinarySelectedPlan::getEmissionPath() const {
  return emissionPath;
}

llvm::StringRef RVVBinarySelectedPlan::getRouteID() const {
  return descriptor.getRVVRouteID();
}

llvm::StringRef RVVBinarySelectedPlan::getArtifactKind() const {
  return getRVVBinaryRuntimeCallableCSourceArtifactKind();
}

llvm::StringRef RVVBinarySelectedPlan::getRuntimeABI() const {
  return descriptor.getRVVRuntimeABI();
}

llvm::StringRef RVVBinarySelectedPlan::getRuntimeABIKind() const {
  return descriptor.getRVVRuntimeABIKind();
}

llvm::StringRef RVVBinarySelectedPlan::getRuntimeABIName() const {
  return descriptor.getRVVRuntimeABIName();
}

llvm::StringRef RVVBinarySelectedPlan::getRuntimeGlueRole() const {
  return descriptor.getRVVRuntimeGlueRole();
}

llvm::StringRef RVVBinarySelectedPlan::getSupportedMessage() const {
  return supportedMessage;
}

std::string RVVBinarySelectedPlan::getSetVLIntrinsicName() const {
  return descriptor.getSetVLIntrinsicName();
}

std::string RVVBinarySelectedPlan::getLoadIntrinsicName() const {
  return descriptor.getLoadIntrinsicName();
}

std::string RVVBinarySelectedPlan::getArithmeticIntrinsicName() const {
  return descriptor.getArithmeticIntrinsicName();
}

std::string RVVBinarySelectedPlan::getStoreIntrinsicName() const {
  return descriptor.getStoreIntrinsicName();
}

const RVVSelectedVectorShapeMetadataNames &
getRVVVariantSelectedVectorShapeMetadataNames() {
  static const RVVSelectedVectorShapeMetadataNames names{
      target::rvv::getRVVSelectedVectorShapeAttrName(),
      target::rvv::getRVVSelectedVectorSEWAttrName(),
      target::rvv::getRVVSelectedVectorLMULAttrName(),
      target::rvv::getRVVSelectedTailPolicyAttrName(),
      target::rvv::getRVVSelectedMaskPolicyAttrName(),
      target::rvv::getRVVSelectedVectorTypeAttrName(),
      target::rvv::getRVVSelectedVectorSuffixAttrName(),
      target::rvv::getRVVSelectedSetVLSuffixAttrName()};
  return names;
}

const RVVSelectedVectorShapeMetadataNames &
getRVVBoundarySelectedVectorShapeMetadataNames() {
  static const RVVSelectedVectorShapeMetadataNames names{
      kBoundarySelectedVectorShapeAttrName,
      kBoundarySelectedVectorSEWAttrName,
      kBoundarySelectedVectorLMULAttrName,
      kBoundarySelectedTailPolicyAttrName,
      kBoundarySelectedMaskPolicyAttrName,
      kBoundarySelectedVectorTypeAttrName,
      kBoundarySelectedVectorSuffixAttrName,
      kBoundarySelectedSetVLSuffixAttrName};
  return names;
}

llvm::StringRef getRVVBinaryRuntimeCallableCSourceArtifactKind() {
  return kRuntimeCallableCSourceArtifactKind;
}

llvm::Expected<RVVBinaryEmissionIdentity> buildRVVBinaryEmissionIdentity(
    const target::rvv::RVVBinaryFamilyDescriptor &family) {
  llvm::Expected<const target::rvv::RVVBinaryFamilyDescriptor *>
      registeredFamily = getRegisteredRVVBinaryFamily(family);
  if (!registeredFamily)
    return registeredFamily.takeError();

  RVVBinaryEmissionIdentity identity;
  identity.family = *registeredFamily;
  identity.emissionPath =
      (llvm::Twine((*registeredFamily)->emissionKind) + "-export").str();
  identity.supportedMessage =
      buildSupportedEmissionMessage(**registeredFamily);
  return identity;
}

llvm::Expected<RVVBinarySelectedPlan> buildRVVBinarySelectedPlan(
    const target::rvv::RVVBinaryFamilyDescriptor &family,
    const target::rvv::RVVVectorShapeConfig &shape,
    std::int64_t elementCount, llvm::StringRef requiredMarch,
    std::optional<std::string> selectedMABI) {
  llvm::Expected<RVVBinaryEmissionIdentity> identity =
      buildRVVBinaryEmissionIdentity(family);
  if (!identity)
    return identity.takeError();

  if (family.dtypeID != shape.dtypeID) {
    return makeRVVBinaryPlanningError(
        llvm::Twine(family.descriptorNoun) +
        " requires selected vector-shape dtype '" + family.dtypeID +
        "' but got '" + shape.dtypeID + "'");
  }

  if (elementCount <= 0 || elementCount > 64)
    return makeRVVBinaryPlanningError(
        llvm::Twine(family.descriptorNoun) +
        " requires tcrv_rvv.element_count in the bounded smoke range [1, 64]");

  llvm::StringRef trimmedMarch = requiredMarch.trim();
  if (trimmedMarch.empty())
    return makeRVVBinaryPlanningError(
        llvm::Twine(family.descriptorNoun) +
        " requires string 'tcrv_rvv.required_march' metadata");
  if (llvm::Error error = validateRVVPlanningText(
          family.descriptorNoun, kRequiredMarchAttrName, trimmedMarch))
    return std::move(error);

  if (selectedMABI && !selectedMABI->empty()) {
    if (llvm::Error error = validateRVVPlanningText(
            family.descriptorNoun, "selected_mabi", *selectedMABI))
      return std::move(error);
  }

  RVVBinarySelectedPlan plan;
  plan.family = &family;
  plan.shape = &shape;
  plan.descriptor = target::rvv::getRVVBinaryIntrinsicDescriptor(family, shape);
  plan.elementCount = elementCount;
  plan.requiredMarch = trimmedMarch.str();
  plan.selectedMABI = std::move(selectedMABI);
  plan.emissionPath = identity->emissionPath;
  plan.supportedMessage = identity->supportedMessage;
  return plan;
}

llvm::Expected<std::optional<RVVBinarySelectedPlan>>
buildRVVBinarySelectedPlanFromVariant(
    tcrv::exec::VariantOp variant,
    const target::rvv::RVVVectorShapeConfig &shape,
    llvm::StringRef expectedDTypeID,
    std::optional<std::string> selectedMABI) {
  if (!variant)
    return makeRVVBinaryPlanningError(
        "selected binary plan requires a materialized tcrv.exec.variant");

  mlir::Attribute rawDescriptor = variant->getAttr(kLoweringDescriptorAttrName);
  if (!rawDescriptor)
    return std::optional<RVVBinarySelectedPlan>();

  auto descriptor = llvm::dyn_cast<mlir::StringAttr>(rawDescriptor);
  if (!descriptor || descriptor.getValue().trim().empty())
    return makeRVVBinaryPlanningError(
        llvm::Twine("finite RVV binary microkernel lowering descriptor on "
                    "variant @") +
        variant.getSymName() + " requires string attribute '" +
        kLoweringDescriptorAttrName + "'");

  llvm::StringRef descriptorValue = descriptor.getValue().trim();
  const target::rvv::RVVBinaryFamilyDescriptor *family =
      target::rvv::lookupRVVBinaryFamilyByLoweringDescriptor(descriptorValue);
  if (!family)
    return makeRVVBinaryPlanningError(
        llvm::Twine("finite RVV binary microkernel lowering descriptor on "
                    "variant @") +
        variant.getSymName() +
        " must be one registered RVV binary lowering descriptor");

  if (!expectedDTypeID.empty() && family->dtypeID != expectedDTypeID)
    return std::optional<RVVBinarySelectedPlan>();

  std::string descriptorContext =
      (llvm::Twine("variant @") + variant.getSymName() + " " +
       family->descriptorNoun)
          .str();
  if (llvm::Error error = validateRVVPlanningText(
          descriptorContext, kLoweringDescriptorAttrName, descriptorValue))
    return std::move(error);

  if (family->dtypeID != shape.dtypeID)
    return makeRVVBinaryPlanningError(
        llvm::Twine(family->descriptorNoun) + " on variant @" +
        variant.getSymName() + " requires finite " +
        getDTypeDiagnosticSpelling(shape) +
        " vector-shape config capability ids");

  auto elementCountAttr =
      variant->getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
  if (!elementCountAttr)
    return makeRVVBinaryPlanningError(
        llvm::Twine(family->descriptorNoun) + " on variant @" +
        variant.getSymName() + " requires integer attribute '" +
        kElementCountAttrName + "'");

  std::int64_t elementCount = elementCountAttr.getInt();
  if (elementCount <= 0 || elementCount > 64)
    return makeRVVBinaryPlanningError(
        llvm::Twine(family->descriptorNoun) + " on variant @" +
        variant.getSymName() +
        " requires tcrv_rvv.element_count in the bounded smoke range [1, 64]");

  auto requiredMarch =
      variant->getAttrOfType<mlir::StringAttr>(kRequiredMarchAttrName);
  if (!requiredMarch || requiredMarch.getValue().trim().empty())
    return makeRVVBinaryPlanningError(
        llvm::Twine(family->descriptorNoun) + " on variant @" +
        variant.getSymName() + " requires string 'tcrv_rvv.required_march' "
                               "metadata");

  return buildRVVBinarySelectedPlan(*family, shape, elementCount,
                                    requiredMarch.getValue().trim(),
                                    std::move(selectedMABI));
}

void addRVVSelectedVectorShapeMetadataToProposal(
    VariantProposal &proposal, mlir::MLIRContext *context,
    const target::rvv::RVVVectorShapeConfig &shape) {
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context,
                            target::rvv::getRVVSelectedVectorShapeAttrName()),
      mlir::StringAttr::get(context, shape.shapeID));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context,
                            target::rvv::getRVVSelectedVectorSEWAttrName()),
      mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                             shape.sewBits));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context,
                            target::rvv::getRVVSelectedVectorLMULAttrName()),
      mlir::StringAttr::get(context, shape.lmul));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context,
                            target::rvv::getRVVSelectedTailPolicyAttrName()),
      mlir::StringAttr::get(context, shape.tailPolicy));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context,
                            target::rvv::getRVVSelectedMaskPolicyAttrName()),
      mlir::StringAttr::get(context, shape.maskPolicy));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context,
                            target::rvv::getRVVSelectedVectorTypeAttrName()),
      mlir::StringAttr::get(context, shape.vectorType));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context,
                            target::rvv::getRVVSelectedVectorSuffixAttrName()),
      mlir::StringAttr::get(context, shape.vectorSuffix));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context,
                            target::rvv::getRVVSelectedSetVLSuffixAttrName()),
      mlir::StringAttr::get(context, shape.setvlSuffix));
}

void addRVVSelectedVectorShapeMetadataToOperationState(
    mlir::OperationState &state, mlir::MLIRContext *context,
    const target::rvv::RVVVectorShapeConfig &shape) {
  const RVVSelectedVectorShapeMetadataNames &names =
      getRVVBoundarySelectedVectorShapeMetadataNames();
  state.addAttribute(names.shape, mlir::StringAttr::get(context, shape.shapeID));
  state.addAttribute(names.sew, mlir::IntegerAttr::get(
                                    mlir::IntegerType::get(context, 64),
                                    shape.sewBits));
  state.addAttribute(names.lmul, mlir::StringAttr::get(context, shape.lmul));
  state.addAttribute(names.tailPolicy,
                     mlir::StringAttr::get(context, shape.tailPolicy));
  state.addAttribute(names.maskPolicy,
                     mlir::StringAttr::get(context, shape.maskPolicy));
  state.addAttribute(names.vectorType,
                     mlir::StringAttr::get(context, shape.vectorType));
  state.addAttribute(names.vectorSuffix,
                     mlir::StringAttr::get(context, shape.vectorSuffix));
  state.addAttribute(names.setvlSuffix,
                     mlir::StringAttr::get(context, shape.setvlSuffix));
}

llvm::Error addRVVSelectedVectorShapeMetadataToPlan(
    VariantEmissionPlan &plan, tcrv::exec::VariantOp variant,
    const target::rvv::RVVVectorShapeConfig &shape) {
  if (llvm::Error error = validateRVVSelectedVectorShapeMetadata(
          variant.getOperation(),
          (llvm::Twine("selected RVV variant @") + variant.getSymName()).str(),
          shape, getRVVVariantSelectedVectorShapeMetadataNames()))
    return error;

  llvm::SmallVector<
      target::rvv::RVVI32VectorShapeSelectedPlanMetadataDescriptor, 8>
      metadata;
  target::rvv::appendRVVI32VectorShapeSelectedPlanMetadata(shape, metadata);
  for (const auto &entry : metadata)
    plan.addSelectedPlanMetadata(entry.name.str(), entry.value.str(),
                                 entry.role.str(), entry.note.str());
  return llvm::Error::success();
}

bool hasAnyRVVSelectedVectorShapeMetadata(
    mlir::Operation *op, const RVVSelectedVectorShapeMetadataNames &names) {
  return op && (op->hasAttr(names.shape) || op->hasAttr(names.sew) ||
                op->hasAttr(names.lmul) || op->hasAttr(names.tailPolicy) ||
                op->hasAttr(names.maskPolicy) ||
                op->hasAttr(names.vectorType) ||
                op->hasAttr(names.vectorSuffix) ||
                op->hasAttr(names.setvlSuffix));
}

llvm::Error validateRVVSelectedVectorShapeMetadata(
    mlir::Operation *op, llvm::StringRef context,
    const target::rvv::RVVVectorShapeConfig &shape,
    const RVVSelectedVectorShapeMetadataNames &names) {
  if (!hasAnyRVVSelectedVectorShapeMetadata(op, names))
    return llvm::Error::success();

  auto shapeAttr = op->getAttrOfType<mlir::StringAttr>(names.shape);
  auto sew = op->getAttrOfType<mlir::IntegerAttr>(names.sew);
  auto lmul = op->getAttrOfType<mlir::StringAttr>(names.lmul);
  auto tailPolicy = op->getAttrOfType<mlir::StringAttr>(names.tailPolicy);
  auto maskPolicy = op->getAttrOfType<mlir::StringAttr>(names.maskPolicy);
  auto vectorType = op->getAttrOfType<mlir::StringAttr>(names.vectorType);
  auto vectorSuffix = op->getAttrOfType<mlir::StringAttr>(names.vectorSuffix);
  auto setvlSuffix = op->getAttrOfType<mlir::StringAttr>(names.setvlSuffix);
  if (!shapeAttr || !sew || !lmul || !tailPolicy || !maskPolicy ||
      !vectorType || !vectorSuffix || !setvlSuffix)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " selected vector-shape metadata must be complete when any "
        "selected-shape attribute is present");

  if (shapeAttr.getValue().trim() != shape.shapeID)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) + " selected vector-shape id must be '" +
        shape.shapeID + "'");
  if (sew.getInt() != shape.sewBits)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) + " selected vector-shape sew must be '" +
        llvm::Twine(shape.sewBits) + "'");
  if (lmul.getValue().trim() != shape.lmul)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) + " selected vector-shape lmul must be '" +
        shape.lmul + "'");
  if (tailPolicy.getValue().trim() != shape.tailPolicy)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " selected vector-shape tail policy must be '" + shape.tailPolicy +
        "'");
  if (maskPolicy.getValue().trim() != shape.maskPolicy)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " selected vector-shape mask policy must be '" + shape.maskPolicy +
        "'");
  if (vectorType.getValue().trim() != shape.vectorType)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) + " selected vector-shape vector type must be '" +
        shape.vectorType + "'");
  if (vectorSuffix.getValue().trim() != shape.vectorSuffix)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " selected vector-shape vector suffix must be '" +
        shape.vectorSuffix + "'");
  if (setvlSuffix.getValue().trim() != shape.setvlSuffix)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " selected vector-shape setvl suffix must be '" + shape.setvlSuffix +
        "'");
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
