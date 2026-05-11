#include "TianChenRV/Plugin/RVV/RVVBinaryPlanning.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kLoweringDescriptorAttrName(
    "tcrv_rvv.lowering_descriptor");
constexpr llvm::StringLiteral kElementCountAttrName("tcrv_rvv.element_count");
constexpr llvm::StringLiteral kRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kFrontendLoweringAttrName(
    "tcrv_frontend_lowering");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kArchitecturePropertyName("architecture");
constexpr llvm::StringLiteral kISAVectorHintsPropertyName("isa_vector_hints");
constexpr llvm::StringLiteral kHartCountPropertyName("count");
constexpr llvm::StringLiteral kVLenBBytesPropertyName("bytes");
constexpr llvm::StringLiteral kI32M1LanesPropertyName("lanes");
constexpr llvm::StringLiteral kSEWBitsPropertyName("sew_bits");
constexpr llvm::StringLiteral kLMULPropertyName("lmul");
constexpr llvm::StringLiteral kTailPolicyPropertyName("tail_policy");
constexpr llvm::StringLiteral kMaskPolicyPropertyName("mask_policy");
constexpr llvm::StringLiteral kSelectedMarchPropertyName("selected_march");
constexpr llvm::StringLiteral kSelectedMarchValuePropertyName("value");
constexpr llvm::StringLiteral kRuntimeCallableCSourceArtifactKind(
    "runtime-callable-c-source");
constexpr llvm::StringLiteral kProposalCondition(
    "rvv_capability_properties_available");
constexpr llvm::StringLiteral kProposalGuard(
    "plugin_local_rvv_property_evidence");
constexpr llvm::StringLiteral kProposalPolicy("metadata_only_first_slice");
constexpr std::int64_t kDefaultBinaryElementCount = 16;
constexpr std::uint64_t kBinaryCapacitySampleVectors = 4;
constexpr std::int64_t kMaxBinaryElementCount = 64;

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
constexpr llvm::StringLiteral kDirectTypedMicrokernelBodySourceKind(
    "direct-typed-microkernel-body");
constexpr llvm::StringLiteral kDefaultTypedMicrokernelBodySourceKind(
    "default-i32-vadd-typed-body-materialization");
constexpr llvm::StringLiteral kDirectLegacyLoweringRegistrationSourceKind(
    "direct-legacy-lowering-registration");
constexpr llvm::StringLiteral kDirectRVVBinaryRegistrationMirrorSourceKind(
    "direct-rvv-binary-registration-mirror");
constexpr llvm::StringLiteral kBufferRoleAttrName("buffer_role");

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

bool hasRVVVectorHint(llvm::StringRef hints) {
  std::string lower = hints.lower();
  llvm::StringRef normalized(lower);
  if (normalized.contains("zve") || normalized.contains("zvl") ||
      normalized.contains("zvfh") || normalized.contains("gcv"))
    return true;

  std::size_t position = lower.find("rv64");
  while (position != std::string::npos) {
    std::size_t end = position;
    while (end < lower.size()) {
      unsigned char byte = static_cast<unsigned char>(lower[end]);
      if (!std::isalnum(byte) && lower[end] != '_' && lower[end] != '-')
        break;
      ++end;
    }
    if (llvm::StringRef(lower).slice(position, end).drop_front(4).contains("v"))
      return true;
    position = lower.find("rv64", position + 4);
  }
  return false;
}

const target::rvv::RVVVectorShapeConfig &getI32M1ConfigSpec() {
  return target::rvv::getI32M1VectorShapeConfig();
}

const target::rvv::RVVVectorShapeConfig &getI32M2ConfigSpec() {
  return target::rvv::getI32M2VectorShapeConfig();
}

const target::rvv::RVVVectorShapeConfig *
getImplicitRequiredShapeForFamily(
    const target::rvv::RVVBinaryFamilyDescriptor &family) {
  if (family.dtype == target::rvv::RVVBinaryDTypeKind::I64)
    return &target::rvv::getI64M1VectorShapeConfig();
  return nullptr;
}

llvm::Expected<std::string>
getRequiredRVVCapabilityProperty(
    const support::CapabilityDescriptor &capability,
    llvm::StringRef propertyName) {
  llvm::StringRef value = capability.getProperty(propertyName).trim();
  std::string context = (llvm::Twine("capability id '") +
                         capability.getID() + "'").str();
  if (value.empty())
    return makeRVVBinaryPlanningError(llvm::Twine(context) +
                                      " requires preserved property '" +
                                      propertyName + "'");

  if (llvm::Error error = validateRVVPlanningText(context, propertyName, value))
    return std::move(error);

  return value.str();
}

llvm::Expected<std::uint64_t>
getRequiredPositiveIntegerRVVCapabilityProperty(
    const support::CapabilityDescriptor &capability,
    llvm::StringRef propertyName) {
  llvm::Expected<std::string> property =
      getRequiredRVVCapabilityProperty(capability, propertyName);
  if (!property)
    return property.takeError();

  llvm::StringRef value(*property);
  if (!llvm::all_of(value, [](char character) {
        unsigned char byte = static_cast<unsigned char>(character);
        return std::isdigit(byte);
      })) {
    return makeRVVBinaryPlanningError(
        llvm::Twine("capability id '") + capability.getID() +
        "' property '" + propertyName + "' must be a positive integer");
  }

  std::uint64_t parsed = 0;
  if (value.getAsInteger(10, parsed) || parsed == 0)
    return makeRVVBinaryPlanningError(
        llvm::Twine("capability id '") + capability.getID() +
        "' property '" + propertyName + "' must be a positive integer");

  return parsed;
}

llvm::Error requireAvailableCapability(
    const support::TargetCapabilitySet &capabilities, llvm::StringRef id,
    const support::CapabilityDescriptor *&out) {
  out = capabilities.lookupProviderByID(id);
  if (!out)
    return makeRVVBinaryPlanningError(
        llvm::Twine("RVV property decision requires capability id '") + id +
        "'");
  if (!out->isAvailable())
    return makeRVVBinaryPlanningError(
        llvm::Twine("RVV property decision requires available capability id '") +
        id + "'");
  return llvm::Error::success();
}

llvm::Error requireFirstSliceSEWCapability(
    const support::TargetCapabilitySet &capabilities,
    const target::rvv::RVVVectorShapeConfig &config) {
  const support::CapabilityDescriptor *capability = nullptr;
  if (llvm::Error error = requireAvailableCapability(
          capabilities, config.sewCapabilityID, capability))
    return std::move(error);

  llvm::Expected<std::uint64_t> sew =
      getRequiredPositiveIntegerRVVCapabilityProperty(*capability,
                                                      kSEWBitsPropertyName);
  if (!sew)
    return sew.takeError();
  if (*sew != static_cast<std::uint64_t>(config.sewBits))
    return makeRVVBinaryPlanningError(
        llvm::Twine("RVV first-slice config capability id '") +
        config.sewCapabilityID + "' property 'sew_bits' must be " +
        llvm::Twine(config.sewBits));
  return llvm::Error::success();
}

llvm::Error requireFirstSliceStringCapability(
    const support::TargetCapabilitySet &capabilities, llvm::StringRef id,
    llvm::StringRef propertyName, llvm::StringRef expectedValue) {
  const support::CapabilityDescriptor *capability = nullptr;
  if (llvm::Error error = requireAvailableCapability(capabilities, id,
                                                     capability))
    return std::move(error);

  llvm::Expected<std::string> property =
      getRequiredRVVCapabilityProperty(*capability, propertyName);
  if (!property)
    return property.takeError();
  if (*property != expectedValue)
    return makeRVVBinaryPlanningError(
        llvm::Twine("RVV first-slice config capability id '") + id +
        "' property '" + propertyName + "' must be '" + expectedValue + "'");
  return llvm::Error::success();
}

llvm::Error verifyFiniteShapeConfigCapabilities(
    const support::TargetCapabilitySet &capabilities,
    const target::rvv::RVVVectorShapeConfig &config) {
  if (llvm::Error error =
          requireFirstSliceSEWCapability(capabilities, config))
    return error;
  if (llvm::Error error = requireFirstSliceStringCapability(
          capabilities, config.lmulCapabilityID, kLMULPropertyName,
          config.lmul))
    return error;
  if (llvm::Error error = requireFirstSliceStringCapability(
          capabilities, config.tailPolicyCapabilityID,
          kTailPolicyPropertyName, config.tailPolicy))
    return error;
  if (llvm::Error error = requireFirstSliceStringCapability(
          capabilities, config.maskPolicyCapabilityID,
          kMaskPolicyPropertyName, config.maskPolicy))
    return error;
  return llvm::Error::success();
}

llvm::Expected<const target::rvv::RVVVectorShapeConfig *>
selectExplicitI32BinaryVectorShapeCapability(
    const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *selector =
      capabilities.lookupProviderByID(
          target::rvv::getRVVI32BinarySelectedVectorShapeCapabilityID());
  if (!selector)
    return static_cast<const target::rvv::RVVVectorShapeConfig *>(nullptr);

  if (!selector->isAvailable())
    return makeRVVBinaryPlanningError(
        llvm::Twine("RVV i32 binary selected vector-shape capability id '") +
        target::rvv::getRVVI32BinarySelectedVectorShapeCapabilityID() +
        "' must be available when present");

  llvm::Expected<std::string> selectedShape =
      getRequiredRVVCapabilityProperty(
          *selector,
          target::rvv::getRVVI32BinarySelectedVectorShapePropertyName());
  if (!selectedShape)
    return selectedShape.takeError();

  const target::rvv::RVVVectorShapeConfig *config =
      target::rvv::lookupFiniteI32VectorShapeConfigByShapeID(*selectedShape);
  if (!config)
    return makeRVVBinaryPlanningError(
        llvm::Twine("RVV i32 binary selected vector-shape capability property "
                    "'") +
        target::rvv::getRVVI32BinarySelectedVectorShapePropertyName() +
        "' must be one finite registered shape: '" +
        getI32M1ConfigSpec().shapeID + "' or '" +
        getI32M2ConfigSpec().shapeID + "'");

  if (llvm::Error error = verifyFiniteShapeConfigCapabilities(capabilities,
                                                             *config))
    return std::move(error);
  return config;
}

llvm::Expected<const target::rvv::RVVVectorShapeConfig *>
selectAvailableI32BinaryShapeConfigCapabilities(
    const support::TargetCapabilitySet &capabilities) {
  llvm::Expected<const target::rvv::RVVVectorShapeConfig *>
      explicitSelection =
          selectExplicitI32BinaryVectorShapeCapability(capabilities);
  if (!explicitSelection)
    return explicitSelection.takeError();
  if (*explicitSelection)
    return *explicitSelection;

  if (llvm::Error error =
          verifyFiniteShapeConfigCapabilities(capabilities,
                                             getI32M1ConfigSpec())) {
    llvm::consumeError(std::move(error));
  } else {
    return &getI32M1ConfigSpec();
  }

  if (llvm::Error error =
          verifyFiniteShapeConfigCapabilities(capabilities,
                                             getI32M2ConfigSpec())) {
    llvm::consumeError(std::move(error));
  } else {
    return &getI32M2ConfigSpec();
  }

  return makeRVVBinaryPlanningError(
      "RVV property decision requires either the finite i32m1 config capability "
      "ids or the finite i32m2 config capability ids");
}

std::int64_t deriveRVVBinaryDescriptorElementCount(
    const RVVBinaryCapabilityPropertyView &view) {
  if (!view.i32M1LaneCount)
    return kDefaultBinaryElementCount;

  if (*view.i32M1LaneCount >=
      static_cast<std::uint64_t>(kMaxBinaryElementCount) /
          kBinaryCapacitySampleVectors)
    return kMaxBinaryElementCount;

  return static_cast<std::int64_t>(*view.i32M1LaneCount *
                                   kBinaryCapacitySampleVectors);
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
      target::rvv::lookupRVVBinaryFamilyRegistrationByID(family.familyID);
  if (!registeredFamily ||
      registeredFamily->loweringDescriptor != family.loweringDescriptor ||
      registeredFamily->routeID != family.routeID ||
      registeredFamily->emissionKind != family.emissionKind ||
      registeredFamily->runtimeABI != family.runtimeABI ||
      registeredFamily->runtimeABIKind != family.runtimeABIKind ||
      registeredFamily->runtimeABIName != family.runtimeABIName ||
      registeredFamily->runtimeGlueRole != family.runtimeGlueRole ||
      !registeredFamily->frontendContract ||
      registeredFamily->frontendContract->familyID != family.familyID ||
      registeredFamily->frontendContract->frontendLowering !=
          family.frontendLowering ||
      registeredFamily->frontendContract->elementBitWidth !=
          family.elementBitWidth ||
      registeredFamily->frontendContract->constInputPointerCType !=
          family.constInputPointerCType ||
      registeredFamily->frontendContract->outputPointerCType !=
          family.outputPointerCType) {
    return makeRVVBinaryPlanningError(
        llvm::Twine("family '") + family.familyID +
        "' must be one registered finite RVV binary family record with a "
        "matching finite frontend marker/ABI contract");
  }
  return registeredFamily;
}

const target::rvv::RVVBinaryFamilyDescriptor *
lookupRVVBinaryFamilyRegistrationByMicrokernelOpName(llvm::StringRef opName) {
  for (const target::rvv::RVVBinaryFamilyDescriptor *family :
       target::rvv::getRVVBinaryFamilyRegistrationRecords()) {
    if (family->microkernelOpName == opName)
      return family;
  }
  return nullptr;
}

const target::rvv::RVVBinaryFamilyDescriptor *
lookupRVVBinaryFamilyRegistrationByArithmeticOpName(llvm::StringRef opName) {
  for (const target::rvv::RVVBinaryFamilyDescriptor *family :
       target::rvv::getRVVBinaryFamilyRegistrationRecords()) {
    if (family->arithmeticOpName == opName)
      return family;
  }
  return nullptr;
}

llvm::Expected<const target::rvv::RVVVectorShapeConfig *>
resolveDirectSelectedShapeMetadata(
    mlir::Operation *op,
    const target::rvv::RVVBinaryFamilyDescriptor &family,
    const RVVSelectedVectorShapeMetadataNames &names,
    llvm::StringRef context);

bool isSameRVVBinaryFamily(const target::rvv::RVVBinaryFamilyDescriptor &lhs,
                           const target::rvv::RVVBinaryFamilyDescriptor &rhs) {
  return lhs.dtype == rhs.dtype && lhs.arithmetic == rhs.arithmetic;
}

bool isDirectTypedSourceKind(llvm::StringRef sourceKind) {
  return sourceKind == kDirectTypedMicrokernelBodySourceKind;
}

bool isLegacyRegistrationSourceKind(llvm::StringRef sourceKind) {
  return sourceKind == kDirectLegacyLoweringRegistrationSourceKind;
}

std::string stringifyType(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  stream.flush();
  return text;
}

llvm::StringRef stringifyTailPolicyValue(tcrv::rvv::TailPolicy policy) {
  switch (policy) {
  case tcrv::rvv::TailPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::TailPolicy::Undisturbed:
    return "undisturbed";
  }
  return "unknown";
}

llvm::StringRef stringifyMaskPolicyValue(tcrv::rvv::MaskPolicy policy) {
  switch (policy) {
  case tcrv::rvv::MaskPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::MaskPolicy::Undisturbed:
    return "undisturbed";
  }
  return "unknown";
}

struct DirectTypedRVVBinaryBodyResolution {
  const target::rvv::RVVBinaryFamilyDescriptor *family = nullptr;
  const target::rvv::RVVVectorShapeConfig *bodyShape = nullptr;
};

llvm::Error requireDirectTypedVectorValue(
    llvm::StringRef context, mlir::Value value, mlir::Type expectedType,
    llvm::StringRef valueContext) {
  if (value.getType() == expectedType)
    return llvm::Error::success();
  return makeRVVBinaryPlanningError(
      llvm::Twine(context) + " " + valueContext +
      " must use typed RVV vector token type '" + stringifyType(expectedType) +
      "' from the selected setvl/with_vl config before selected-plan authority is accepted");
}

llvm::Error requireDirectTypedBufferRole(mlir::Operation *op,
                                         llvm::StringRef context,
                                         llvm::StringRef expectedRole,
                                         llvm::StringRef roleContext) {
  auto role = op->getAttrOfType<mlir::StringAttr>(kBufferRoleAttrName);
  if (!role || role.getValue().trim().empty())
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) + " " + roleContext +
        " is missing typed buffer_role ABI metadata before descriptor "
        "fallback");
  llvm::StringRef value = role.getValue().trim();
  if (llvm::Error error =
          validateRVVPlanningText(context, kBufferRoleAttrName, value))
    return error;
  if (value != expectedRole)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) + " " + roleContext +
        " buffer_role must be '" + expectedRole +
        "' before selected-plan authority is accepted");
  return llvm::Error::success();
}

mlir::Type getExpectedTypedBodyVectorType(
    mlir::MLIRContext *context,
    const target::rvv::RVVBinaryFamilyDescriptor &family,
    const target::rvv::RVVVectorShapeConfig &shape) {
  if (family.dtype == target::rvv::RVVBinaryDTypeKind::I32) {
    if (shape.lmul == "m1")
      return tcrv::rvv::I32M1VectorType::get(context);
    if (shape.lmul == "m2")
      return tcrv::rvv::I32M2VectorType::get(context);
  }
  if (family.dtype == target::rvv::RVVBinaryDTypeKind::I64 &&
      shape.lmul == "m1")
    return tcrv::rvv::I64M1VectorType::get(context);
  return {};
}

llvm::Expected<const target::rvv::RVVVectorShapeConfig *>
resolveTypedBodyShapeFromControlPlane(
    llvm::StringRef context,
    const target::rvv::RVVBinaryFamilyDescriptor &family,
    tcrv::rvv::SetVLOp setvl, tcrv::rvv::WithVLOp withVL,
    const target::rvv::RVVVectorShapeConfig *metadataShape) {
  if (setvl.getPolicy().getTail() != tcrv::rvv::TailPolicy::Agnostic ||
      setvl.getPolicy().getMask() != tcrv::rvv::MaskPolicy::Agnostic)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed body setvl policy must be tail=agnostic, mask=agnostic "
        "before selected-plan authority is accepted");

  auto bodySew = withVL->getAttrOfType<mlir::IntegerAttr>("sew");
  auto bodyLMUL = withVL->getAttrOfType<mlir::StringAttr>("lmul");
  auto bodyPolicy = withVL->getAttrOfType<tcrv::rvv::PolicyAttr>("policy");
  if (!bodySew || !bodyLMUL || !bodyPolicy)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed body with_vl must carry SEW/LMUL/policy metadata before "
        "selected-plan authority");
  if (bodySew.getInt() != static_cast<std::int64_t>(setvl.getSew()) ||
      bodyLMUL.getValue() != setvl.getLmul() ||
      bodyPolicy != setvl.getPolicy())
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed body with_vl SEW/LMUL/policy must match setvl before "
        "selected-plan authority");

  const target::rvv::RVVVectorShapeConfig *bodyShape = nullptr;
  llvm::StringRef tail = stringifyTailPolicyValue(setvl.getPolicy().getTail());
  llvm::StringRef mask = stringifyMaskPolicyValue(setvl.getPolicy().getMask());
  for (const target::rvv::RVVVectorShapeConfig *shape :
       target::rvv::getRVVBinaryFamilyShapeConfigs(family)) {
    if (shape->sewBits == static_cast<std::int64_t>(setvl.getSew()) &&
        shape->lmul == setvl.getLmul() && shape->tailPolicy == tail &&
        shape->maskPolicy == mask) {
      bodyShape = shape;
      break;
    }
  }
  if (!bodyShape)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed body setvl/with_vl config does not match a finite selected "
        "RVV binary shape for family '" +
        family.familyID + "'");
  if (metadataShape && metadataShape->shapeID != bodyShape->shapeID)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " selected vector-shape metadata conflicts with typed body "
        "setvl/with_vl config before selected-plan authority is accepted");
  return bodyShape;
}

llvm::Expected<DirectTypedRVVBinaryBodyResolution>
resolveDirectTypedRVVBinaryBody(mlir::Operation *op,
                                llvm::StringRef context) {
  DirectTypedRVVBinaryBodyResolution resolution;
  const target::rvv::RVVBinaryFamilyDescriptor *opFamily =
      lookupRVVBinaryFamilyRegistrationByMicrokernelOpName(op->getName().getStringRef());
  if (!opFamily)
    return resolution;

  if (op->getNumRegions() != 1 || op->getRegion(0).empty() ||
      !llvm::hasSingleElement(op->getRegion(0)))
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed microkernel body must contain exactly one structured "
        "control-plane block before selected-plan authority is accepted");

  mlir::Block &controlBlock = op->getRegion(0).front();
  if (controlBlock.getNumArguments() != 1 ||
      !controlBlock.getArgument(0).getType().isIndex())
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed microkernel body must expose exactly one index runtime "
        "AVL/VL argument before selected-plan authority is accepted");

  tcrv::rvv::SetVLOp setvl;
  tcrv::rvv::WithVLOp withVL;
  unsigned setvlCount = 0;
  unsigned withVLCount = 0;
  for (mlir::Operation &bodyOp : controlBlock) {
    if (auto candidate = llvm::dyn_cast<tcrv::rvv::SetVLOp>(bodyOp)) {
      setvl = candidate;
      ++setvlCount;
      continue;
    }
    if (auto candidate = llvm::dyn_cast<tcrv::rvv::WithVLOp>(bodyOp)) {
      withVL = candidate;
      ++withVLCount;
      continue;
    }
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed microkernel control-plane body may contain only "
        "tcrv_rvv.setvl and tcrv_rvv.with_vl before selected-plan authority is accepted");
  }
  if (setvlCount != 1 || withVLCount != 1)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed microkernel control-plane body requires exactly one "
        "tcrv_rvv.setvl and one tcrv_rvv.with_vl before selected-plan authority is accepted");
  if (setvl.getAvl() != controlBlock.getArgument(0))
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed body setvl AVL must come from the runtime index block "
        "argument before selected-plan authority is accepted");
  if (withVL.getVl() != setvl.getVl())
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed body with_vl must consume the !tcrv_rvv.vl token produced "
        "by setvl before selected-plan authority is accepted");

  mlir::Region &withVLBody = withVL.getBody();
  if (withVLBody.empty() || !llvm::hasSingleElement(withVLBody) ||
      withVLBody.front().getNumArguments() != 0)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed with_vl body must contain one zero-argument dataflow block "
        "before selected-plan authority is accepted");

  llvm::SmallVector<mlir::Operation *, 4> ops;
  for (mlir::Operation &dataflowOp : withVLBody.front())
    ops.push_back(&dataflowOp);
  if (ops.size() != 4)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed dataflow body must contain exactly load, load, arithmetic, "
        "store before selected-plan authority is accepted");

  const target::rvv::RVVBinaryFamilyDescriptor *bodyFamily =
      lookupRVVBinaryFamilyRegistrationByArithmeticOpName(ops[2]->getName().getStringRef());
  if (!bodyFamily)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed dataflow arithmetic op must be a registered finite RVV "
        "binary family op before selected-plan authority is accepted");
  if (!isSameRVVBinaryFamily(*opFamily, *bodyFamily))
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) + " typed microkernel op '" +
        opFamily->microkernelOpName + "' body contains '" +
        bodyFamily->arithmeticOpName +
        "'; typed body and microkernel op family must agree before "
        "selected-plan authority");

  llvm::Expected<const target::rvv::RVVVectorShapeConfig *> metadataShape =
      resolveDirectSelectedShapeMetadata(
          op, *bodyFamily, getRVVBoundarySelectedVectorShapeMetadataNames(),
          context);
  if (!metadataShape)
    return metadataShape.takeError();

  llvm::Expected<const target::rvv::RVVVectorShapeConfig *> bodyShape =
      resolveTypedBodyShapeFromControlPlane(context, *bodyFamily, setvl, withVL,
                                            *metadataShape);
  if (!bodyShape)
    return bodyShape.takeError();

  mlir::Type expectedVectorType =
      getExpectedTypedBodyVectorType(op->getContext(), *bodyFamily, **bodyShape);
  if (!expectedVectorType)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " typed body selected shape is unsupported for finite RVV binary "
        "dataflow before selected-plan authority is accepted");

  if (bodyFamily->dtype == target::rvv::RVVBinaryDTypeKind::I32) {
    auto lhsLoad = llvm::dyn_cast<tcrv::rvv::I32LoadOp>(ops[0]);
    auto rhsLoad = llvm::dyn_cast<tcrv::rvv::I32LoadOp>(ops[1]);
    auto store = llvm::dyn_cast<tcrv::rvv::I32StoreOp>(ops[3]);
    mlir::Value arithmeticLHS;
    mlir::Value arithmeticRHS;
    mlir::Value arithmeticVL;
    mlir::Value arithmeticResult;
    if (auto add = llvm::dyn_cast<tcrv::rvv::I32AddOp>(ops[2])) {
      arithmeticLHS = add.getLhs();
      arithmeticRHS = add.getRhs();
      arithmeticVL = add.getVl();
      arithmeticResult = add.getSum();
    } else if (auto sub = llvm::dyn_cast<tcrv::rvv::I32SubOp>(ops[2])) {
      arithmeticLHS = sub.getLhs();
      arithmeticRHS = sub.getRhs();
      arithmeticVL = sub.getVl();
      arithmeticResult = sub.getDifference();
    } else if (auto mul = llvm::dyn_cast<tcrv::rvv::I32MulOp>(ops[2])) {
      arithmeticLHS = mul.getLhs();
      arithmeticRHS = mul.getRhs();
      arithmeticVL = mul.getVl();
      arithmeticResult = mul.getProduct();
    }
    if (!lhsLoad || !rhsLoad || !store || !arithmeticResult)
      return makeRVVBinaryPlanningError(
          llvm::Twine(context) +
          " typed i32 dataflow body must be i32_load, i32_load, "
          "i32_add/sub/mul, i32_store before selected-plan authority is accepted");
    if (lhsLoad.getVl() != withVL.getVl() ||
        rhsLoad.getVl() != withVL.getVl() || arithmeticVL != withVL.getVl() ||
        store.getVl() != withVL.getVl())
      return makeRVVBinaryPlanningError(
          llvm::Twine(context) +
          " typed i32 dataflow ops must consume the with_vl token before "
          "selected-plan authority");
    if (arithmeticLHS != lhsLoad.getLoaded() ||
        arithmeticRHS != rhsLoad.getLoaded() ||
        store.getValue() != arithmeticResult)
      return makeRVVBinaryPlanningError(
          llvm::Twine(context) +
          " typed i32 dataflow SSA chain must be lhs-load,rhs-load -> "
          "arithmetic -> store before selected-plan authority is accepted");
    if (llvm::Error error =
            requireDirectTypedVectorValue(context, lhsLoad.getLoaded(),
                                          expectedVectorType,
                                          "first i32_load result"))
      return std::move(error);
    if (llvm::Error error =
            requireDirectTypedVectorValue(context, rhsLoad.getLoaded(),
                                          expectedVectorType,
                                          "second i32_load result"))
      return std::move(error);
    if (llvm::Error error =
            requireDirectTypedVectorValue(context, arithmeticResult,
                                          expectedVectorType,
                                          "i32 arithmetic result"))
      return std::move(error);
    if (llvm::Error error = requireDirectTypedBufferRole(
            lhsLoad.getOperation(), context, "lhs-input-buffer",
            "first i32_load"))
      return std::move(error);
    if (llvm::Error error = requireDirectTypedBufferRole(
            rhsLoad.getOperation(), context, "rhs-input-buffer",
            "second i32_load"))
      return std::move(error);
    if (llvm::Error error = requireDirectTypedBufferRole(
            store.getOperation(), context, "output-buffer", "i32_store"))
      return std::move(error);
  } else {
    auto lhsLoad = llvm::dyn_cast<tcrv::rvv::I64LoadOp>(ops[0]);
    auto rhsLoad = llvm::dyn_cast<tcrv::rvv::I64LoadOp>(ops[1]);
    auto store = llvm::dyn_cast<tcrv::rvv::I64StoreOp>(ops[3]);
    mlir::Value arithmeticLHS;
    mlir::Value arithmeticRHS;
    mlir::Value arithmeticVL;
    mlir::Value arithmeticResult;
    if (auto add = llvm::dyn_cast<tcrv::rvv::I64AddOp>(ops[2])) {
      arithmeticLHS = add.getLhs();
      arithmeticRHS = add.getRhs();
      arithmeticVL = add.getVl();
      arithmeticResult = add.getSum();
    } else if (auto sub = llvm::dyn_cast<tcrv::rvv::I64SubOp>(ops[2])) {
      arithmeticLHS = sub.getLhs();
      arithmeticRHS = sub.getRhs();
      arithmeticVL = sub.getVl();
      arithmeticResult = sub.getDifference();
    } else if (auto mul = llvm::dyn_cast<tcrv::rvv::I64MulOp>(ops[2])) {
      arithmeticLHS = mul.getLhs();
      arithmeticRHS = mul.getRhs();
      arithmeticVL = mul.getVl();
      arithmeticResult = mul.getProduct();
    }
    if (!lhsLoad || !rhsLoad || !store || !arithmeticResult)
      return makeRVVBinaryPlanningError(
          llvm::Twine(context) +
          " typed i64 dataflow body must be i64_load, i64_load, "
          "i64_add/sub/mul, i64_store before selected-plan authority is accepted");
    if (lhsLoad.getVl() != withVL.getVl() ||
        rhsLoad.getVl() != withVL.getVl() || arithmeticVL != withVL.getVl() ||
        store.getVl() != withVL.getVl())
      return makeRVVBinaryPlanningError(
          llvm::Twine(context) +
          " typed i64 dataflow ops must consume the with_vl token before "
          "selected-plan authority");
    if (arithmeticLHS != lhsLoad.getLoaded() ||
        arithmeticRHS != rhsLoad.getLoaded() ||
        store.getValue() != arithmeticResult)
      return makeRVVBinaryPlanningError(
          llvm::Twine(context) +
          " typed i64 dataflow SSA chain must be lhs-load,rhs-load -> "
          "arithmetic -> store before selected-plan authority is accepted");
    if (llvm::Error error =
            requireDirectTypedVectorValue(context, lhsLoad.getLoaded(),
                                          expectedVectorType,
                                          "first i64_load result"))
      return std::move(error);
    if (llvm::Error error =
            requireDirectTypedVectorValue(context, rhsLoad.getLoaded(),
                                          expectedVectorType,
                                          "second i64_load result"))
      return std::move(error);
    if (llvm::Error error =
            requireDirectTypedVectorValue(context, arithmeticResult,
                                          expectedVectorType,
                                          "i64 arithmetic result"))
      return std::move(error);
    if (llvm::Error error = requireDirectTypedBufferRole(
            lhsLoad.getOperation(), context, "lhs-input-buffer",
            "first i64_load"))
      return std::move(error);
    if (llvm::Error error = requireDirectTypedBufferRole(
            rhsLoad.getOperation(), context, "rhs-input-buffer",
            "second i64_load"))
      return std::move(error);
    if (llvm::Error error = requireDirectTypedBufferRole(
            store.getOperation(), context, "output-buffer", "i64_store"))
      return std::move(error);
  }

  resolution.family = bodyFamily;
  resolution.bodyShape = *bodyShape;
  return resolution;
}

llvm::Expected<const target::rvv::RVVVectorShapeConfig *>
resolveDirectSelectedShapeMetadata(
    mlir::Operation *op,
    const target::rvv::RVVBinaryFamilyDescriptor &family,
    const RVVSelectedVectorShapeMetadataNames &names,
    llvm::StringRef context) {
  if (!hasAnyRVVSelectedVectorShapeMetadata(op, names))
    return static_cast<const target::rvv::RVVVectorShapeConfig *>(nullptr);

  auto shapeAttr = op->getAttrOfType<mlir::StringAttr>(names.shape);
  if (!shapeAttr || shapeAttr.getValue().trim().empty())
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " selected vector-shape metadata must include non-empty '" +
        names.shape + "'");

  llvm::StringRef shapeID = shapeAttr.getValue().trim();
  if (llvm::Error error =
          validateRVVPlanningText(context, names.shape, shapeID))
    return std::move(error);

  const target::rvv::RVVVectorShapeConfig *shape =
      target::rvv::lookupRVVBinaryFamilyShapeConfigByID(family, shapeID);
  if (!shape)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) + " selected vector-shape id '" + shapeID +
        "' does not belong to finite RVV binary registration family '" +
        family.familyID + "'");

  if (llvm::Error error = validateRVVSelectedVectorShapeMetadata(
          op, context, *shape, names))
    return std::move(error);
  return shape;
}

struct DirectRVVBinaryFamilyCandidate {
  const target::rvv::RVVBinaryFamilyDescriptor *family = nullptr;
  const target::rvv::RVVVectorShapeConfig *selectedShape = nullptr;
  std::string source;
};

llvm::Error mergeDirectRVVBinaryFamilyCandidate(
    RVVBinaryFamilyPlanningResolution &resolution,
    const DirectRVVBinaryFamilyCandidate &candidate,
    llvm::StringRef diagnosticContext) {
  if (!candidate.family)
    return llvm::Error::success();

  if (!resolution.family) {
    resolution.family = candidate.family;
    resolution.directSelectedShape = candidate.selectedShape;
    resolution.sourceKind = candidate.source;
    return llvm::Error::success();
  }

  if (resolution.family->familyID != candidate.family->familyID)
    return makeRVVBinaryPlanningError(
        llvm::Twine(diagnosticContext) +
        " has ambiguous direct RVV binary registration records: '" +
        resolution.family->loweringDescriptor + "' and '" +
        candidate.family->loweringDescriptor +
        "' resolve to different finite families");

  if (resolution.directSelectedShape && candidate.selectedShape &&
      resolution.directSelectedShape->shapeID !=
          candidate.selectedShape->shapeID)
    return makeRVVBinaryPlanningError(
        llvm::Twine(diagnosticContext) +
        " has ambiguous direct RVV binary selected vector-shape metadata for "
        "family '" +
        resolution.family->familyID + "': '" +
        resolution.directSelectedShape->shapeID + "' and '" +
        candidate.selectedShape->shapeID + "'");

  if (!resolution.directSelectedShape)
    resolution.directSelectedShape = candidate.selectedShape;
  if (resolution.sourceKind != candidate.source) {
    if (isDirectTypedSourceKind(candidate.source))
      resolution.sourceKind = candidate.source;
    else if (!isDirectTypedSourceKind(resolution.sourceKind))
      resolution.sourceKind = kDirectRVVBinaryRegistrationMirrorSourceKind.str();
  }
  return llvm::Error::success();
}

llvm::Expected<DirectRVVBinaryFamilyCandidate>
buildDirectLegacyRegistrationCandidateFromVariant(
    tcrv::exec::VariantOp variant) {
  DirectRVVBinaryFamilyCandidate candidate;
  if (!variant)
    return candidate;

  auto origin = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!origin || origin.getValue() != kRVVPluginName)
    return candidate;

  auto descriptor =
      variant->getAttrOfType<mlir::StringAttr>(kLoweringDescriptorAttrName);
  if (!descriptor)
    return candidate;
  llvm::StringRef descriptorValue = descriptor.getValue().trim();
  std::string context =
      (llvm::Twine("direct RVV binary variant @") + variant.getSymName()).str();
  if (descriptorValue.empty())
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) + " requires non-empty string attribute '" +
        kLoweringDescriptorAttrName + "'");
  if (llvm::Error error = validateRVVPlanningText(
          context, kLoweringDescriptorAttrName, descriptorValue))
    return std::move(error);

  const target::rvv::RVVBinaryFamilyDescriptor *family =
      target::rvv::lookupRVVBinaryFamilyRegistrationByLegacyLoweringDescriptor(
          descriptorValue);
  if (!family)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) + " direct lowering descriptor '" +
        descriptorValue +
        "' must be one registered finite RVV binary lowering registration");

  llvm::Expected<const target::rvv::RVVVectorShapeConfig *> selectedShape =
      resolveDirectSelectedShapeMetadata(
          variant.getOperation(), *family,
          getRVVVariantSelectedVectorShapeMetadataNames(), context);
  if (!selectedShape)
    return selectedShape.takeError();

  candidate.family = family;
  candidate.selectedShape = *selectedShape;
  candidate.source = kDirectLegacyLoweringRegistrationSourceKind.str();
  return candidate;
}

llvm::Expected<DirectRVVBinaryFamilyCandidate>
buildDirectTypedBodyCandidateFromMicrokernel(mlir::Operation *op) {
  DirectRVVBinaryFamilyCandidate candidate;
  if (!op)
    return candidate;

  const target::rvv::RVVBinaryFamilyDescriptor *family =
      lookupRVVBinaryFamilyRegistrationByMicrokernelOpName(
          op->getName().getStringRef());
  if (!family)
    return candidate;

  std::string context =
      (llvm::Twine("direct RVV binary microkernel '") +
       op->getName().getStringRef() + "'")
          .str();
  llvm::Expected<DirectTypedRVVBinaryBodyResolution> typedBody =
      resolveDirectTypedRVVBinaryBody(op, context);
  if (!typedBody)
    return typedBody.takeError();
  if (!typedBody->family)
    return candidate;

  llvm::Expected<const target::rvv::RVVVectorShapeConfig *> selectedShape =
      resolveDirectSelectedShapeMetadata(
          op, *typedBody->family,
          getRVVBoundarySelectedVectorShapeMetadataNames(), context);
  if (!selectedShape)
    return selectedShape.takeError();
  if (*selectedShape && typedBody->bodyShape &&
      (*selectedShape)->shapeID != typedBody->bodyShape->shapeID)
    return makeRVVBinaryPlanningError(
        llvm::Twine(context) +
        " selected vector-shape metadata conflicts with typed body "
        "setvl/with_vl config before selected-plan authority is accepted");

  candidate.family = typedBody->family;
  candidate.selectedShape = typedBody->bodyShape ? typedBody->bodyShape
                                                 : *selectedShape;
  candidate.source = kDirectTypedMicrokernelBodySourceKind.str();
  return candidate;
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

llvm::StringRef RVVBinaryProposalPlan::getFamilyID() const {
  return selectedPlan.getFamilyID();
}

llvm::StringRef RVVBinaryProposalPlan::getDTypeID() const {
  return selectedPlan.getDTypeID();
}

llvm::StringRef RVVBinaryProposalPlan::getLoweringDescriptor() const {
  return selectedPlan.getLoweringDescriptor();
}

const target::rvv::RVVBinaryFamilyDescriptor &
RVVBinaryProposalPlan::getFamily() const {
  return *selectedPlan.family;
}

const target::rvv::RVVVectorShapeConfig &
RVVBinaryProposalPlan::getSelectedShape() const {
  return selectedPlan.getShape();
}

llvm::ArrayRef<std::string>
RVVBinaryProposalPlan::getRequiredCapabilityIDs() const {
  return requiredCapabilityIDs;
}

llvm::StringRef RVVBinaryProposalPlan::getCondition() const {
  return condition;
}

llvm::StringRef RVVBinaryProposalPlan::getGuard() const { return guard; }

llvm::StringRef RVVBinaryProposalPlan::getPolicy() const { return policy; }

bool RVVBinaryProposalPlan::hasCapacityMetadata() const {
  return capabilityView.vlenbBytes && capabilityView.i32M1LaneCount;
}

bool isTypedSourceRVVBinaryFamily(
    const target::rvv::RVVBinaryFamilyDescriptor &family) {
  return family.dtype == target::rvv::RVVBinaryDTypeKind::I32 ||
         family.dtype == target::rvv::RVVBinaryDTypeKind::I64;
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

std::string formatRVVBinaryFamilyFrontendLoweringList() {
  std::string text;
  llvm::raw_string_ostream stream(text);
  bool first = true;
  for (const target::rvv::RVVBinaryFamilyDescriptor *family :
       target::rvv::getRVVBinaryFamilyRegistrationRecords()) {
    if (!first)
      stream << " or ";
    stream << '\'' << family->frontendLowering << '\'';
    first = false;
  }
  stream.flush();
  return text;
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

llvm::Expected<RVVBinaryCapabilityPropertyView>
buildRVVBinaryCapabilityPropertyView(
    const support::TargetCapabilitySet &capabilities,
    const target::rvv::RVVVectorShapeConfig *requiredShape) {
  const support::CapabilityDescriptor *rvvCapability = nullptr;
  if (llvm::Error error =
          requireAvailableCapability(capabilities, kRVVCapabilityID,
                                     rvvCapability))
    return std::move(error);

  llvm::Expected<std::string> architecture =
      getRequiredRVVCapabilityProperty(*rvvCapability, kArchitecturePropertyName);
  if (!architecture)
    return architecture.takeError();
  if (llvm::StringRef(*architecture).lower() != "riscv64")
    return makeRVVBinaryPlanningError(
        "capability id 'rvv' property 'architecture' must be riscv64");

  llvm::Expected<std::string> isaVectorHints =
      getRequiredRVVCapabilityProperty(*rvvCapability,
                                       kISAVectorHintsPropertyName);
  if (!isaVectorHints)
    return isaVectorHints.takeError();
  if (!hasRVVVectorHint(*isaVectorHints))
    return makeRVVBinaryPlanningError(
        "capability id 'rvv' property 'isa_vector_hints' must contain RVV "
        "vector evidence");

  const support::CapabilityDescriptor *hartCountCapability = nullptr;
  if (llvm::Error error = requireAvailableCapability(
          capabilities, getRVVHartCountCapabilityID(), hartCountCapability))
    return std::move(error);

  llvm::Expected<std::uint64_t> hartCount =
      getRequiredPositiveIntegerRVVCapabilityProperty(*hartCountCapability,
                                                      kHartCountPropertyName);
  if (!hartCount)
    return hartCount.takeError();

  const support::CapabilityDescriptor *vlenbCapability =
      capabilities.lookupProviderByID(getRVVVLenBBytesCapabilityID());
  const support::CapabilityDescriptor *i32LaneCapability =
      capabilities.lookupProviderByID(getRVVI32M1LaneCountCapabilityID());
  bool hasAvailableVLenB = vlenbCapability && vlenbCapability->isAvailable();
  bool hasAvailableI32Lanes =
      i32LaneCapability && i32LaneCapability->isAvailable();
  std::optional<std::uint64_t> vlenbBytes;
  std::optional<std::uint64_t> i32M1LaneCount;
  if (hasAvailableVLenB != hasAvailableI32Lanes)
    return makeRVVBinaryPlanningError(
        "RVV vector capacity decision requires both available capability ids "
        "'rvv.vlenb_bytes' and 'rvv.i32_m1_lane_count'");
  if (hasAvailableVLenB) {
    llvm::Expected<std::uint64_t> parsedVLenB =
        getRequiredPositiveIntegerRVVCapabilityProperty(*vlenbCapability,
                                                        kVLenBBytesPropertyName);
    if (!parsedVLenB)
      return parsedVLenB.takeError();
    llvm::Expected<std::uint64_t> parsedI32Lanes =
        getRequiredPositiveIntegerRVVCapabilityProperty(
            *i32LaneCapability, kI32M1LanesPropertyName);
    if (!parsedI32Lanes)
      return parsedI32Lanes.takeError();
    if (*parsedVLenB < sizeof(std::int32_t) ||
        *parsedVLenB % sizeof(std::int32_t) != 0 ||
        *parsedVLenB / sizeof(std::int32_t) != *parsedI32Lanes)
      return makeRVVBinaryPlanningError(
          "RVV vector capacity decision requires i32 m1 lane count to match "
          "vlenb bytes divided by four");
    vlenbBytes = *parsedVLenB;
    i32M1LaneCount = *parsedI32Lanes;
  }

  const support::CapabilityDescriptor *compileRunCapability = nullptr;
  if (llvm::Error error = requireAvailableCapability(
          capabilities, getRVVProbeCompileRunCapabilityID(),
          compileRunCapability))
    return std::move(error);

  llvm::Expected<std::string> selectedMarch =
      getRequiredRVVCapabilityProperty(*compileRunCapability,
                                       kSelectedMarchPropertyName);
  if (!selectedMarch)
    return selectedMarch.takeError();
  if (!hasRVVVectorHint(*selectedMarch))
    return makeRVVBinaryPlanningError(
        "capability id 'rvv.probe.compile_run' property 'selected_march' must "
        "contain RVV vector evidence");

  if (const support::CapabilityDescriptor *selectedMarchCapability =
          capabilities.lookupByID(getRVVSelectedMarchCapabilityID())) {
    if (selectedMarchCapability->isAvailable()) {
      llvm::Expected<std::string> selectedMarchValue =
          getRequiredRVVCapabilityProperty(*selectedMarchCapability,
                                           kSelectedMarchValuePropertyName);
      if (!selectedMarchValue)
        return selectedMarchValue.takeError();
      if (*selectedMarchValue != *selectedMarch)
        return makeRVVBinaryPlanningError(
            "conflicting RVV property values between capability id "
            "'rvv.toolchain.march' property 'value' and capability id "
            "'rvv.probe.compile_run' property 'selected_march'");
    }
  }

  const target::rvv::RVVVectorShapeConfig *selectedShape = requiredShape;
  if (selectedShape) {
    if (llvm::Error error =
            verifyFiniteShapeConfigCapabilities(capabilities, *selectedShape))
      return std::move(error);
  } else {
    llvm::Expected<const target::rvv::RVVVectorShapeConfig *> shape =
        selectAvailableI32BinaryShapeConfigCapabilities(capabilities);
    if (!shape)
      return shape.takeError();
    selectedShape = *shape;
  }

  RVVBinaryCapabilityPropertyView view;
  view.architecture = std::move(*architecture);
  view.isaVectorHints = std::move(*isaVectorHints);
  view.selectedMarch = std::move(*selectedMarch);
  view.hartCount = *hartCount;
  view.vlenbBytes = vlenbBytes;
  view.i32M1LaneCount = i32M1LaneCount;
  view.selectedShape = selectedShape;
  return view;
}

llvm::Expected<bool> variantRequiresCapabilityID(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities, llvm::StringRef id) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requiresAttr)
    return makeRVVBinaryPlanningError("materialized RVV variant requires "
                                      "structured 'requires' metadata");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeRVVBinaryPlanningError(
          "materialized RVV variant requires only capability symbol "
          "references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      continue;

    if (capability->satisfiesID(id))
      return true;
  }

  return false;
}

llvm::Error verifyRVVBinaryVariantRequiresCapabilityID(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities, llvm::StringRef id) {
  llvm::Expected<bool> requiresID =
      variantRequiresCapabilityID(variant, capabilities, id);
  if (!requiresID)
    return requiresID.takeError();
  if (!*requiresID)
    return makeRVVBinaryPlanningError(
        llvm::Twine("materialized RVV variant must require capability id '") +
        id + "'");
  return llvm::Error::success();
}

llvm::Expected<bool> variantRequiresConfig(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    const target::rvv::RVVVectorShapeConfig &config) {
  llvm::Expected<bool> requiresSEW =
      variantRequiresCapabilityID(variant, capabilities, config.sewCapabilityID);
  if (!requiresSEW)
    return requiresSEW.takeError();
  llvm::Expected<bool> requiresLMUL =
      variantRequiresCapabilityID(variant, capabilities, config.lmulCapabilityID);
  if (!requiresLMUL)
    return requiresLMUL.takeError();
  llvm::Expected<bool> requiresTail = variantRequiresCapabilityID(
      variant, capabilities, config.tailPolicyCapabilityID);
  if (!requiresTail)
    return requiresTail.takeError();
  llvm::Expected<bool> requiresMask = variantRequiresCapabilityID(
      variant, capabilities, config.maskPolicyCapabilityID);
  if (!requiresMask)
    return requiresMask.takeError();
  return *requiresSEW && *requiresLMUL && *requiresTail && *requiresMask;
}

llvm::Expected<bool> variantRequiresAnyConfigID(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    const target::rvv::RVVVectorShapeConfig &config) {
  for (llvm::StringRef id :
       {config.sewCapabilityID, config.lmulCapabilityID,
        config.tailPolicyCapabilityID, config.maskPolicyCapabilityID}) {
    llvm::Expected<bool> requiresID =
        variantRequiresCapabilityID(variant, capabilities, id);
    if (!requiresID)
      return requiresID.takeError();
    if (*requiresID)
      return true;
  }
  return false;
}

llvm::Error verifyVariantRequiresConfigIDs(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    const target::rvv::RVVVectorShapeConfig &config) {
  if (llvm::Error error = verifyRVVBinaryVariantRequiresCapabilityID(
          variant, capabilities, config.sewCapabilityID))
    return error;
  if (llvm::Error error = verifyRVVBinaryVariantRequiresCapabilityID(
          variant, capabilities, config.lmulCapabilityID))
    return error;
  if (llvm::Error error = verifyRVVBinaryVariantRequiresCapabilityID(
          variant, capabilities, config.tailPolicyCapabilityID))
    return error;
  if (llvm::Error error = verifyRVVBinaryVariantRequiresCapabilityID(
          variant, capabilities, config.maskPolicyCapabilityID))
    return error;
  return llvm::Error::success();
}

llvm::Expected<const target::rvv::RVVVectorShapeConfig *>
getRVVBinaryVariantRequiredShapeConfig(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities) {
  const target::rvv::RVVVectorShapeConfig &i32M1 =
      target::rvv::getI32M1VectorShapeConfig();
  const target::rvv::RVVVectorShapeConfig &i32M2 =
      target::rvv::getI32M2VectorShapeConfig();
  const target::rvv::RVVVectorShapeConfig &i64M1 =
      target::rvv::getI64M1VectorShapeConfig();

  llvm::Expected<bool> requiresM1 =
      variantRequiresConfig(variant, capabilities, i32M1);
  if (!requiresM1)
    return requiresM1.takeError();
  llvm::Expected<bool> requiresM2 =
      variantRequiresConfig(variant, capabilities, i32M2);
  if (!requiresM2)
    return requiresM2.takeError();
  llvm::Expected<bool> requiresI64M1 =
      variantRequiresConfig(variant, capabilities, i64M1);
  if (!requiresI64M1)
    return requiresI64M1.takeError();

  unsigned completeConfigCount = (*requiresM1 ? 1 : 0) + (*requiresM2 ? 1 : 0) +
                                 (*requiresI64M1 ? 1 : 0);
  if (completeConfigCount > 1)
    return makeRVVBinaryPlanningError(
        "materialized RVV variant must require exactly one finite RVV "
        "dtype/LMUL config shape");
  if (*requiresM1)
    return &i32M1;
  if (*requiresM2)
    return &i32M2;
  if (*requiresI64M1)
    return &i64M1;

  llvm::Expected<bool> requiresAnyM1 =
      variantRequiresAnyConfigID(variant, capabilities, i32M1);
  if (!requiresAnyM1)
    return requiresAnyM1.takeError();
  llvm::Expected<bool> requiresAnyM2 =
      variantRequiresAnyConfigID(variant, capabilities, i32M2);
  if (!requiresAnyM2)
    return requiresAnyM2.takeError();
  llvm::Expected<bool> requiresAnyI64M1 =
      variantRequiresAnyConfigID(variant, capabilities, i64M1);
  if (!requiresAnyI64M1)
    return requiresAnyI64M1.takeError();
  if (*requiresAnyM1 && !*requiresAnyM2)
    if (llvm::Error error =
            verifyVariantRequiresConfigIDs(variant, capabilities, i32M1))
      return std::move(error);
  if (*requiresAnyM2 && !*requiresAnyM1)
    if (llvm::Error error =
            verifyVariantRequiresConfigIDs(variant, capabilities, i32M2))
      return std::move(error);
  if (*requiresAnyI64M1 && !*requiresAnyM1 && !*requiresAnyM2)
    if (llvm::Error error =
            verifyVariantRequiresConfigIDs(variant, capabilities, i64M1))
      return std::move(error);

  return makeRVVBinaryPlanningError(
      "materialized RVV variant must require either the finite i32m1, i32m2, "
      "or i64m1 config capability ids");
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
        llvm::Twine("selected config mismatch: finite binary family '") +
        family.familyID + "' has dtype '" + family.dtypeID +
        "' but selected vector-shape '" + shape.shapeID + "' has dtype '" +
        shape.dtypeID + "'");
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

  llvm::Expected<target::rvv::RVVBinarySelectedConfigContract> selectedConfig =
      target::rvv::buildRVVBinarySelectedConfigContract(
          family, shape, /*selectedVariantSymbol=*/llvm::StringRef(),
          /*selectedRole=*/llvm::StringRef(), elementCount);
  if (!selectedConfig)
    return selectedConfig.takeError();

  RVVBinarySelectedPlan plan;
  plan.family = &family;
  plan.selectedConfig.contract = std::move(*selectedConfig);
  plan.descriptor = target::rvv::getRVVBinaryIntrinsicDescriptor(family, shape);
  plan.elementCount = elementCount;
  plan.requiredMarch = trimmedMarch.str();
  plan.selectedMABI = std::move(selectedMABI);
  plan.emissionPath = identity->emissionPath;
  plan.supportedMessage = identity->supportedMessage;
  return plan;
}

llvm::Expected<RVVBinaryProposalPlan> buildRVVBinaryProposalPlanForFamily(
    const support::TargetCapabilitySet &capabilities,
    const target::rvv::RVVBinaryFamilyDescriptor &requestedFamily,
    const target::rvv::RVVVectorShapeConfig *directSelectedShape,
    bool attachLoweringDescriptorAttr,
    llvm::StringRef diagnosticContext) {
  llvm::Expected<const target::rvv::RVVBinaryFamilyDescriptor *>
      registeredFamily = getRegisteredRVVBinaryFamily(requestedFamily);
  if (!registeredFamily)
    return registeredFamily.takeError();

  const target::rvv::RVVVectorShapeConfig *requiredShape =
      directSelectedShape ? directSelectedShape
                          : getImplicitRequiredShapeForFamily(
                                **registeredFamily);
  if (requiredShape && requiredShape->dtypeID != (*registeredFamily)->dtypeID)
    return makeRVVBinaryPlanningError(
        llvm::Twine(diagnosticContext) +
        " selected config mismatch: finite binary family '" +
        (*registeredFamily)->familyID + "' has dtype '" +
        (*registeredFamily)->dtypeID + "' but selected vector-shape '" +
        requiredShape->shapeID + "' has dtype '" + requiredShape->dtypeID +
        "'");

  llvm::Expected<RVVBinaryCapabilityPropertyView> propertyView =
      buildRVVBinaryCapabilityPropertyView(capabilities, requiredShape);
  if (!propertyView)
    return propertyView.takeError();

  std::int64_t descriptorElementCount =
      deriveRVVBinaryDescriptorElementCount(*propertyView);
  llvm::Expected<RVVBinarySelectedPlan> selectedPlan =
      buildRVVBinarySelectedPlan(**registeredFamily,
                                 *propertyView->selectedShape,
                                 descriptorElementCount,
                                 propertyView->selectedMarch);
  if (!selectedPlan)
    return selectedPlan.takeError();

  RVVBinaryProposalPlan plan;
  plan.selectedPlan = std::move(*selectedPlan);
  plan.capabilityView = std::move(*propertyView);
  plan.requiredCapabilityIDs.push_back(kRVVCapabilityID.str());
  for (llvm::StringRef capabilityID :
       plan.selectedPlan.getSelectedConfig().getCapabilityIDs())
    plan.requiredCapabilityIDs.push_back(capabilityID.str());
  plan.condition = kProposalCondition.str();
  plan.guard = kProposalGuard.str();
  plan.policy = kProposalPolicy.str();
  plan.attachLoweringDescriptorAttr = attachLoweringDescriptorAttr;
  return plan;
}

llvm::Expected<RVVBinaryFamilyPlanningResolution>
resolveRVVBinaryFamilyForProposal(tcrv::exec::KernelOp kernel,
                                  llvm::StringRef diagnosticContext) {
  if (!kernel)
    return makeRVVBinaryPlanningError(
        llvm::Twine(diagnosticContext) +
        " requires a materialized tcrv.exec.kernel");

  if (auto frontendLowering =
          kernel->getAttrOfType<mlir::StringAttr>(kFrontendLoweringAttrName)) {
    llvm::StringRef frontendLoweringValue = frontendLowering.getValue().trim();
    if (!frontendLoweringValue.empty()) {
      if (llvm::Error error =
              validateRVVPlanningText(diagnosticContext,
                                      kFrontendLoweringAttrName,
                                      frontendLoweringValue))
        return std::move(error);

      const target::rvv::RVVBinaryFamilyDescriptor *family =
          target::rvv::lookupRVVBinaryFamilyRegistrationByFrontendLowering(
              frontendLoweringValue);
      if (!family)
        return makeRVVBinaryPlanningError(
            llvm::Twine(diagnosticContext) +
            " frontend lowering family must be " +
            formatRVVBinaryFamilyFrontendLoweringList());

      RVVBinaryFamilyPlanningResolution resolution;
      resolution.family = family;
      resolution.sourceKind = "frontend-lowering";
      return resolution;
    }
  }

  RVVBinaryFamilyPlanningResolution typedBodyResolution;
  RVVBinaryFamilyPlanningResolution legacyRegistrationResolution;
  if (!kernel.getBody().empty()) {
    for (mlir::Operation &operation : kernel.getBody().front()) {
      if (llvm::isa<tcrv::exec::VariantOp>(operation))
        continue;

      llvm::Expected<DirectRVVBinaryFamilyCandidate> candidate =
          buildDirectTypedBodyCandidateFromMicrokernel(&operation);
      if (!candidate)
        return candidate.takeError();
      if (llvm::Error error = mergeDirectRVVBinaryFamilyCandidate(
              typedBodyResolution, *candidate, diagnosticContext))
        return std::move(error);
    }

    for (mlir::Operation &operation : kernel.getBody().front()) {
      auto variant = llvm::dyn_cast<tcrv::exec::VariantOp>(operation);
      if (!variant)
        continue;

      llvm::Expected<DirectRVVBinaryFamilyCandidate> candidate =
          buildDirectLegacyRegistrationCandidateFromVariant(variant);
      if (!candidate)
        return candidate.takeError();
      if (llvm::Error error = mergeDirectRVVBinaryFamilyCandidate(
              legacyRegistrationResolution, *candidate, diagnosticContext))
        return std::move(error);
    }
  }

  if (typedBodyResolution.family) {
    if (llvm::Error error = mergeDirectRVVBinaryFamilyCandidate(
            typedBodyResolution,
            DirectRVVBinaryFamilyCandidate{legacyRegistrationResolution.family,
                                           legacyRegistrationResolution
                                               .directSelectedShape,
                                           legacyRegistrationResolution
                                               .sourceKind},
            diagnosticContext))
      return std::move(error);
    typedBodyResolution.sourceKind =
        kDirectTypedMicrokernelBodySourceKind.str();
    return typedBodyResolution;
  }

  if (legacyRegistrationResolution.family) {
    if (isTypedSourceRVVBinaryFamily(*legacyRegistrationResolution.family))
      return makeRVVBinaryPlanningError(
          llvm::Twine(diagnosticContext) +
          " direct legacy-registration-only RVV binary planning for family '" +
          legacyRegistrationResolution.family->familyID +
          "' is legacy-quarantined; add a typed " +
          legacyRegistrationResolution.family->microkernelOpName +
          " body or use frontend-derived typed family lowering so compute "
          "identity comes from RVV family ops");
    return legacyRegistrationResolution;
  }

  RVVBinaryFamilyPlanningResolution defaultResolution;
  defaultResolution.family = &target::rvv::getI32VAddFamilyRegistrationRecord();
  defaultResolution.sourceKind = kDefaultTypedMicrokernelBodySourceKind.str();
  return defaultResolution;
}

llvm::Expected<RVVBinaryProposalPlan> buildRVVBinaryProposalPlan(
    const support::TargetCapabilitySet &capabilities,
    const target::rvv::RVVBinaryFamilyDescriptor &family,
    llvm::StringRef diagnosticContext) {
  return buildRVVBinaryProposalPlanForFamily(
      capabilities, family, /*directSelectedShape=*/nullptr,
      /*attachLoweringDescriptorAttr=*/!isTypedSourceRVVBinaryFamily(family),
      diagnosticContext);
}

llvm::Expected<RVVBinaryProposalPlan> buildRVVBinaryProposalPlan(
    const support::TargetCapabilitySet &capabilities,
    llvm::StringRef frontendLowering, llvm::StringRef diagnosticContext) {
  const target::rvv::RVVBinaryFamilyDescriptor *requestedFamily =
      &target::rvv::getI32VAddFamilyRegistrationRecord();
  llvm::StringRef trimmedFrontendLowering = frontendLowering.trim();
  if (!trimmedFrontendLowering.empty()) {
    if (llvm::Error error = validateRVVPlanningText(
            diagnosticContext, "tcrv_frontend_lowering",
            trimmedFrontendLowering))
      return std::move(error);

    const target::rvv::RVVBinaryFamilyDescriptor *lookup =
        target::rvv::lookupRVVBinaryFamilyRegistrationByFrontendLowering(
            trimmedFrontendLowering);
    if (!lookup)
      return makeRVVBinaryPlanningError(
          llvm::Twine(diagnosticContext) +
          " frontend lowering family must be " +
          formatRVVBinaryFamilyFrontendLoweringList());
    requestedFamily = lookup;
  }

  const target::rvv::RVVVectorShapeConfig *requiredShape =
      requestedFamily->dtype == target::rvv::RVVBinaryDTypeKind::I64
          ? &target::rvv::getI64M1VectorShapeConfig()
          : nullptr;
  return buildRVVBinaryProposalPlanForFamily(
      capabilities, *requestedFamily, requiredShape,
      /*attachLoweringDescriptorAttr=*/
      !isTypedSourceRVVBinaryFamily(*requestedFamily),
      diagnosticContext);
}

llvm::Expected<RVVBinaryProposalPlan> buildRVVBinaryProposalPlan(
    const support::TargetCapabilitySet &capabilities,
    tcrv::exec::KernelOp kernel, llvm::StringRef diagnosticContext) {
  llvm::Expected<RVVBinaryFamilyPlanningResolution> resolution =
      resolveRVVBinaryFamilyForProposal(kernel, diagnosticContext);
  if (!resolution)
    return resolution.takeError();

  bool attachLoweringDescriptorAttr = true;
  if (isTypedSourceRVVBinaryFamily(*resolution->family) &&
      !isLegacyRegistrationSourceKind(resolution->sourceKind))
    attachLoweringDescriptorAttr = false;

  return buildRVVBinaryProposalPlanForFamily(
      capabilities, *resolution->family, resolution->directSelectedShape,
      attachLoweringDescriptorAttr, diagnosticContext);
}

llvm::Expected<RVVBinarySelectedPlan>
buildRVVBinarySelectedPlanFromTypedFamilyVariant(
    tcrv::exec::VariantOp variant,
    const target::rvv::RVVBinaryFamilyDescriptor &family,
    const target::rvv::RVVVectorShapeConfig &shape,
    llvm::StringRef expectedDTypeID,
    std::optional<std::string> selectedMABI) {
  if (!variant)
    return makeRVVBinaryPlanningError(
        "selected binary plan requires a materialized tcrv.exec.variant");

  if (!expectedDTypeID.empty() && family.dtypeID != expectedDTypeID)
    return makeRVVBinaryPlanningError(
        llvm::Twine("typed RVV binary family '") + family.familyID +
        "' has dtype '" + family.dtypeID +
        "' but the selected path requires dtype '" + expectedDTypeID + "'");

  if (family.dtypeID != shape.dtypeID)
    return makeRVVBinaryPlanningError(
        llvm::Twine("selected config mismatch: typed RVV binary family '") +
        family.familyID + "' on variant @" + variant.getSymName() +
        " has dtype '" + family.dtypeID + "' but selected vector-shape '" +
        shape.shapeID + "' has dtype '" + getDTypeDiagnosticSpelling(shape) +
        "'");

  auto elementCountAttr =
      variant->getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
  if (!elementCountAttr)
    return makeRVVBinaryPlanningError(
        llvm::Twine("typed RVV binary family '") + family.familyID +
        "' on variant @" + variant.getSymName() +
        " requires integer attribute '" + kElementCountAttrName + "'");

  std::int64_t elementCount = elementCountAttr.getInt();
  if (elementCount <= 0 || elementCount > 64)
    return makeRVVBinaryPlanningError(
        llvm::Twine("typed RVV binary family '") + family.familyID +
        "' on variant @" + variant.getSymName() +
        " requires tcrv_rvv.element_count in the bounded smoke range [1, 64]");

  auto requiredMarch =
      variant->getAttrOfType<mlir::StringAttr>(kRequiredMarchAttrName);
  if (!requiredMarch || requiredMarch.getValue().trim().empty())
    return makeRVVBinaryPlanningError(
        llvm::Twine("typed RVV binary family '") + family.familyID +
        "' on variant @" + variant.getSymName() +
        " requires string 'tcrv_rvv.required_march' metadata");

  llvm::Expected<RVVBinarySelectedPlan> plan =
      buildRVVBinarySelectedPlan(family, shape, elementCount,
                                 requiredMarch.getValue().trim(),
                                 std::move(selectedMABI));
  if (!plan)
    return plan.takeError();
  plan->selectedConfig.getContract().setSelectedPath(variant.getSymName(),
                                                    llvm::StringRef());
  return std::move(*plan);
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
