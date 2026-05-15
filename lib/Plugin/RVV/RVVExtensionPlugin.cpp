#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVPluginVersion("0.1.0");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kRVVCapabilityKind("isa-vector");
constexpr llvm::StringLiteral kRVVPreferredCapabilitySymbol("rvv");
constexpr llvm::StringLiteral kRVVFirstSliceVariantName("rvv_first_slice");
constexpr llvm::StringLiteral kRVVPolicyAttrName("tcrv_rvv.policy");
constexpr llvm::StringLiteral kRVVRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kRVVVLenBBytesAttrName(
    "tcrv_rvv.vlenb_bytes");
constexpr llvm::StringLiteral kRVVI32M1LanesAttrName(
    "tcrv_rvv.base_i32_m1_lanes");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kBoundarySourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kBoundarySelectedVariantAttrName(
    "selected_variant");
constexpr llvm::StringLiteral kBoundaryOriginAttrName("origin");
constexpr llvm::StringLiteral kBoundaryRoleAttrName("role");
constexpr llvm::StringLiteral kBoundaryStatusAttrName("status");
constexpr llvm::StringLiteral kBoundaryRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kBoundaryVLenBBytesAttrName("vlenb_bytes");
constexpr llvm::StringLiteral kBoundaryI32M1LanesAttrName(
    "base_i32_m1_lanes");
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
constexpr llvm::StringLiteral kBoundaryUnsupportedReasonAttrName(
    "unsupported_reason");
constexpr llvm::StringLiteral kUnsupportedBoundaryStatus("unsupported");
constexpr llvm::StringLiteral kSelectedRVVCapacityMetadataRole(
    "rvv-base-capacity-fact");
constexpr llvm::StringLiteral kSelectedRVVCapacityMetadataNote(
    "base i32 M1 capacity fact from target/profile evidence; not selected "
    "vector shape, runtime input, VL/AVL, or performance evidence");

using RVVI32VectorShapeConfig =
    tianchenrv::target::rvv::RVVI32VectorShapeConfig;

struct RVVCapacityMetadata {
  std::int64_t vlenbBytes = 0;
  std::int64_t i32M1Lanes = 0;
};

struct RVVFirstSlicePlan {
  const RVVI32VectorShapeConfig *shape = nullptr;
  std::string selectedMarch;
  std::optional<RVVCapacityMetadata> capacity;
  llvm::SmallVector<std::string, 5> requiredCapabilityIDs;
};

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasAvailableRVVCapability(const VariantProposalRequest &request) {
  return request.getKernel() &&
         request.getCapabilities().isCapabilityAvailableByID(kRVVCapabilityID);
}

bool containsForbiddenFactText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") ||
         normalized.contains("access_key");
}

bool isSingleBoundedFactString(llvm::StringRef value) {
  if (value.empty() || value.size() > 512)
    return false;
  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return false;
    if (byte < 0x20 && character != '\t')
      return false;
  }
  return !containsForbiddenFactText(value);
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

const support::CapabilityDescriptor *
lookupAvailableProviderByID(const support::TargetCapabilitySet &capabilities,
                            llvm::StringRef id) {
  const support::CapabilityDescriptor *capability =
      capabilities.lookupProviderByID(id);
  if (!capability || !capability->isAvailable())
    return nullptr;
  return capability;
}

llvm::Expected<std::string> getRequiredProperty(
    const support::CapabilityDescriptor &capability, llvm::StringRef name,
    llvm::StringRef context) {
  llvm::StringRef value = capability.getProperty(name).trim();
  if (!isSingleBoundedFactString(value))
    return makeRVVPluginError(llvm::Twine(context) + " requires bounded '" +
                              name + "' capability property");
  return value.str();
}

llvm::Expected<std::int64_t> getRequiredIntegerProperty(
    const support::CapabilityDescriptor &capability, llvm::StringRef name,
    llvm::StringRef context) {
  llvm::Expected<std::string> value =
      getRequiredProperty(capability, name, context);
  if (!value)
    return value.takeError();

  std::int64_t parsed = 0;
  if (llvm::StringRef(*value).getAsInteger(10, parsed))
    return makeRVVPluginError(llvm::Twine(context) + " requires integer '" +
                              name + "' capability property");
  return parsed;
}

llvm::Error validateBaseRVVCapability(
    const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *rvv =
      lookupAvailableProviderByID(capabilities, kRVVCapabilityID);
  if (!rvv)
    return makeRVVPluginError("requires available capability id 'rvv'");

  llvm::Expected<std::string> architecture =
      getRequiredProperty(*rvv, "architecture", "RVV capability");
  if (!architecture)
    return architecture.takeError();
  if (llvm::StringRef(*architecture).lower() != "riscv64")
    return makeRVVPluginError(
        "RVV capability architecture must be bounded riscv64 metadata");

  llvm::Expected<std::string> hints =
      getRequiredProperty(*rvv, "isa_vector_hints", "RVV capability");
  if (!hints)
    return hints.takeError();
  if (!hasRVVVectorHint(*hints))
    return makeRVVPluginError(
        "RVV capability requires bounded ISA vector evidence");

  return llvm::Error::success();
}

llvm::Error validateHartCountCapability(
    const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *hartCount = lookupAvailableProviderByID(
      capabilities, rvv::getRVVHartCountCapabilityID());
  if (!hartCount)
    return makeRVVPluginError(
        "requires available capability id 'rvv.hart_count'");

  llvm::Expected<std::int64_t> count =
      getRequiredIntegerProperty(*hartCount, "count", "RVV hart count");
  if (!count)
    return count.takeError();
  if (*count <= 0)
    return makeRVVPluginError("RVV hart count must be positive");
  return llvm::Error::success();
}

llvm::Error validateShapeCapability(
    const support::TargetCapabilitySet &capabilities, llvm::StringRef id,
    llvm::StringRef propertyName, llvm::StringRef expectedValue,
    llvm::StringRef context) {
  const support::CapabilityDescriptor *capability =
      lookupAvailableProviderByID(capabilities, id);
  if (!capability)
    return makeRVVPluginError(llvm::Twine(context) +
                              " requires available capability id '" + id +
                              "'");

  llvm::Expected<std::string> value =
      getRequiredProperty(*capability, propertyName, context);
  if (!value)
    return value.takeError();
  if (*value != expectedValue)
    return makeRVVPluginError(llvm::Twine(context) +
                              " capability property mismatch for '" + id +
                              "'");
  return llvm::Error::success();
}

llvm::Error validateShapeCapability(
    const support::TargetCapabilitySet &capabilities, llvm::StringRef id,
    llvm::StringRef propertyName, std::int64_t expectedValue,
    llvm::StringRef context) {
  const support::CapabilityDescriptor *capability =
      lookupAvailableProviderByID(capabilities, id);
  if (!capability)
    return makeRVVPluginError(llvm::Twine(context) +
                              " requires available capability id '" + id +
                              "'");

  llvm::Expected<std::int64_t> value =
      getRequiredIntegerProperty(*capability, propertyName, context);
  if (!value)
    return value.takeError();
  if (*value != expectedValue)
    return makeRVVPluginError(llvm::Twine(context) +
                              " capability property mismatch for '" + id +
                              "'");
  return llvm::Error::success();
}

llvm::Error validateI32VectorShape(
    const support::TargetCapabilitySet &capabilities,
    const RVVI32VectorShapeConfig &shape) {
  std::string context =
      (llvm::Twine("RVV selected vector shape ") + shape.shapeID).str();
  if (llvm::Error error = validateShapeCapability(
          capabilities, shape.sewCapabilityID, "sew_bits", shape.sewBits,
          context))
    return error;
  if (llvm::Error error = validateShapeCapability(
          capabilities, shape.lmulCapabilityID, "lmul", shape.lmul, context))
    return error;
  if (llvm::Error error = validateShapeCapability(
          capabilities, shape.tailPolicyCapabilityID, "tail_policy",
          shape.tailPolicy, context))
    return error;
  if (llvm::Error error = validateShapeCapability(
          capabilities, shape.maskPolicyCapabilityID, "mask_policy",
          shape.maskPolicy, context))
    return error;
  return llvm::Error::success();
}

llvm::Expected<const RVVI32VectorShapeConfig *> selectI32VectorShape(
    const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *selector = lookupAvailableProviderByID(
      capabilities,
      tianchenrv::target::rvv::getRVVI32BinarySelectedVectorShapeCapabilityID());
  if (selector) {
    llvm::Expected<std::string> shapeID = getRequiredProperty(
        *selector,
        tianchenrv::target::rvv::
            getRVVI32BinarySelectedVectorShapePropertyName(),
        "RVV selected vector-shape selector");
    if (!shapeID)
      return shapeID.takeError();

    const RVVI32VectorShapeConfig *shape =
        tianchenrv::target::rvv::lookupFiniteI32VectorShapeConfigByShapeID(
            *shapeID);
    if (!shape)
      return makeRVVPluginError(
          "RVV selected vector-shape selector must be i32m1 or i32m2");
    if (llvm::Error error = validateI32VectorShape(capabilities, *shape))
      return std::move(error);
    return shape;
  }

  for (const RVVI32VectorShapeConfig *shape :
       tianchenrv::target::rvv::getFiniteI32VectorShapeConfigs()) {
    if (!lookupAvailableProviderByID(capabilities, shape->sewCapabilityID) ||
        !lookupAvailableProviderByID(capabilities, shape->lmulCapabilityID) ||
        !lookupAvailableProviderByID(capabilities,
                                     shape->tailPolicyCapabilityID) ||
        !lookupAvailableProviderByID(capabilities,
                                     shape->maskPolicyCapabilityID))
      continue;
    if (llvm::Error error = validateI32VectorShape(capabilities, *shape))
      return std::move(error);
    return shape;
  }

  return makeRVVPluginError(
      "requires one complete RVV i32 selected vector-shape capability set");
}

llvm::Expected<std::string>
getSelectedMarch(const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *compileRun = lookupAvailableProviderByID(
      capabilities, rvv::getRVVProbeCompileRunCapabilityID());
  if (!compileRun)
    return makeRVVPluginError(
        "requires available capability id 'rvv.probe.compile_run'");

  llvm::Expected<std::string> selectedMarch = getRequiredProperty(
      *compileRun, "selected_march", "RVV compile/run probe");
  if (!selectedMarch)
    return selectedMarch.takeError();

  const support::CapabilityDescriptor *toolchainMarch =
      capabilities.lookupProviderByID(rvv::getRVVSelectedMarchCapabilityID());
  if (toolchainMarch && toolchainMarch->isAvailable()) {
    llvm::Expected<std::string> value =
        getRequiredProperty(*toolchainMarch, "value", "RVV toolchain march");
    if (!value)
      return value.takeError();
    if (*value != *selectedMarch)
      return makeRVVPluginError(
          "RVV toolchain march must agree with compile/run selected_march");
  }

  return *selectedMarch;
}

llvm::Expected<std::optional<RVVCapacityMetadata>>
getOptionalCapacityMetadata(const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *vlenb =
      capabilities.lookupProviderByID(rvv::getRVVVLenBBytesCapabilityID());
  const support::CapabilityDescriptor *lanes =
      capabilities.lookupProviderByID(rvv::getRVVI32M1LaneCountCapabilityID());
  if (!vlenb && !lanes)
    return std::optional<RVVCapacityMetadata>();
  if (!vlenb || !lanes || !vlenb->isAvailable() || !lanes->isAvailable())
    return makeRVVPluginError(
        "RVV capacity metadata requires available vlenb and i32 m1 lane "
        "capabilities as a pair");

  llvm::Expected<std::int64_t> bytes =
      getRequiredIntegerProperty(*vlenb, "bytes", "RVV vlenb capacity");
  if (!bytes)
    return bytes.takeError();
  llvm::Expected<std::int64_t> laneCount =
      getRequiredIntegerProperty(*lanes, "lanes", "RVV i32 m1 lane capacity");
  if (!laneCount)
    return laneCount.takeError();

  if (*bytes <= 0 || *laneCount <= 0 || *bytes < 4 || *bytes % 4 != 0 ||
      *bytes / 4 != *laneCount)
    return makeRVVPluginError(
        "RVV capacity metadata requires i32 lanes to equal vlenb bytes "
        "divided by four");

  return RVVCapacityMetadata{*bytes, *laneCount};
}

llvm::Expected<RVVFirstSlicePlan>
buildRVVFirstSlicePlan(const support::TargetCapabilitySet &capabilities) {
  if (llvm::Error error = validateBaseRVVCapability(capabilities))
    return std::move(error);
  if (llvm::Error error = validateHartCountCapability(capabilities))
    return std::move(error);

  llvm::Expected<const RVVI32VectorShapeConfig *> shape =
      selectI32VectorShape(capabilities);
  if (!shape)
    return shape.takeError();

  llvm::Expected<std::string> selectedMarch = getSelectedMarch(capabilities);
  if (!selectedMarch)
    return selectedMarch.takeError();

  llvm::Expected<std::optional<RVVCapacityMetadata>> capacity =
      getOptionalCapacityMetadata(capabilities);
  if (!capacity)
    return capacity.takeError();

  RVVFirstSlicePlan plan;
  plan.shape = *shape;
  plan.selectedMarch = *selectedMarch;
  plan.capacity = *capacity;
  plan.requiredCapabilityIDs.push_back(kRVVCapabilityID.str());
  plan.requiredCapabilityIDs.push_back(plan.shape->sewCapabilityID.str());
  plan.requiredCapabilityIDs.push_back(plan.shape->lmulCapabilityID.str());
  plan.requiredCapabilityIDs.push_back(plan.shape->tailPolicyCapabilityID.str());
  plan.requiredCapabilityIDs.push_back(plan.shape->maskPolicyCapabilityID.str());
  return plan;
}

mlir::StringAttr getRVVPolicyAttrNameAttr(mlir::MLIRContext *context) {
  return mlir::StringAttr::get(context, kRVVPolicyAttrName);
}

tcrv::rvv::PolicyAttr getExpectedRVVPolicyAttr(mlir::MLIRContext *context) {
  return tcrv::rvv::PolicyAttr::get(context, tcrv::rvv::TailPolicy::Agnostic,
                                    tcrv::rvv::MaskPolicy::Agnostic);
}

void addSelectedVectorShapeMetadataToProposal(
    VariantProposal &proposal, mlir::MLIRContext *context,
    const RVVI32VectorShapeConfig &shape) {
  llvm::SmallVector<tianchenrv::target::rvv::
                        RVVVectorShapeSelectedPlanMetadataDescriptor,
                    12>
      descriptors;
  tianchenrv::target::rvv::appendRVVVectorShapeSelectedPlanMetadata(
      shape, descriptors);
  for (const auto &descriptor : descriptors) {
    mlir::StringAttr name = mlir::StringAttr::get(context, descriptor.name);
    if (descriptor.name ==
        tianchenrv::target::rvv::getRVVSelectedVectorSEWAttrName()) {
      std::int64_t sew = 0;
      if (!llvm::StringRef(descriptor.value).getAsInteger(10, sew)) {
        proposal.addPluginAttribute(
            name, mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                         sew));
      }
      continue;
    }
    proposal.addPluginAttribute(
        name, mlir::StringAttr::get(context, descriptor.value));
  }
}

llvm::Error addSelectedVectorShapeMetadataToPlan(
    VariantEmissionPlan &plan, const RVVI32VectorShapeConfig &shape) {
  llvm::SmallVector<tianchenrv::target::rvv::
                        RVVVectorShapeSelectedPlanMetadataDescriptor,
                    12>
      descriptors;
  tianchenrv::target::rvv::appendRVVVectorShapeSelectedPlanMetadata(
      shape, descriptors);
  for (const auto &descriptor : descriptors)
    plan.addSelectedPlanMetadata(descriptor.name, descriptor.value,
                                 descriptor.role, descriptor.note);
  return llvm::Error::success();
}

llvm::Expected<VariantProposal>
buildRVVFirstSliceProposal(const VariantProposalRequest &request) {
  llvm::Expected<RVVFirstSlicePlan> plan =
      buildRVVFirstSlicePlan(request.getCapabilities());
  if (!plan)
    return plan.takeError();

  VariantProposal proposal(kRVVFirstSliceVariantName, kRVVPluginName);
  for (llvm::StringRef capabilityID : plan->requiredCapabilityIDs)
    proposal.addRequiredCapabilityID(capabilityID);
  proposal.setCondition("rvv_capability_properties_available");
  proposal.setGuard("plugin_local_rvv_property_evidence");
  proposal.setPolicy("metadata_only_first_slice");

  mlir::MLIRContext *context = request.getKernel()->getContext();
  context->getOrLoadDialect<tcrv::rvv::TCRVRVVDialect>();
  proposal.addPluginAttribute(getRVVPolicyAttrNameAttr(context),
                              getExpectedRVVPolicyAttr(context));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context, kRVVRequiredMarchAttrName),
      mlir::StringAttr::get(context, plan->selectedMarch));
  addSelectedVectorShapeMetadataToProposal(proposal, context, *plan->shape);
  if (plan->capacity) {
    proposal.addPluginAttribute(
        mlir::StringAttr::get(context, kRVVVLenBBytesAttrName),
        mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                               plan->capacity->vlenbBytes));
    proposal.addPluginAttribute(
        mlir::StringAttr::get(context, kRVVI32M1LanesAttrName),
        mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                               plan->capacity->i32M1Lanes));
  }
  return proposal;
}

std::string sanitizeRVVDeclineReason(llvm::StringRef reason) {
  constexpr std::size_t kMaxReasonLength = 512;
  std::string sanitized;
  sanitized.reserve(std::min<std::size_t>(reason.size(), kMaxReasonLength));
  for (char character : reason.take_front(kMaxReasonLength)) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      sanitized.push_back(' ');
    else if (byte < 0x20 && character != '\t')
      sanitized.push_back(' ');
    else
      sanitized.push_back(character);
  }
  if (reason.size() > kMaxReasonLength)
    sanitized.append("...");
  return sanitized;
}

bool variantRequiresCapabilityID(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    llvm::StringRef capabilityID) {
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requires)
    return false;

  for (mlir::Attribute attr : requires) {
    auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
    if (!symbol)
      return false;
    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbol.getValue());
    if (capability && capability->satisfiesID(capabilityID))
      return true;
  }
  return false;
}

llvm::Expected<const RVVI32VectorShapeConfig *>
getVariantSelectedShape(tcrv::exec::VariantOp variant) {
  auto shapeAttr = variant->getAttrOfType<mlir::StringAttr>(
      tianchenrv::target::rvv::getRVVSelectedVectorShapeAttrName());
  if (!shapeAttr || shapeAttr.getValue().trim().empty())
    return makeRVVPluginError(
        "materialized RVV variant requires selected vector-shape metadata");

  const RVVI32VectorShapeConfig *shape =
      tianchenrv::target::rvv::lookupFiniteI32VectorShapeConfigByShapeID(
          shapeAttr.getValue());
  if (!shape)
    return makeRVVPluginError(
        "materialized RVV variant selected vector-shape must be i32m1 or "
        "i32m2");

  auto sewAttr = variant->getAttrOfType<mlir::IntegerAttr>(
      tianchenrv::target::rvv::getRVVSelectedVectorSEWAttrName());
  auto lmulAttr = variant->getAttrOfType<mlir::StringAttr>(
      tianchenrv::target::rvv::getRVVSelectedVectorLMULAttrName());
  auto tailAttr = variant->getAttrOfType<mlir::StringAttr>(
      tianchenrv::target::rvv::getRVVSelectedTailPolicyAttrName());
  auto maskAttr = variant->getAttrOfType<mlir::StringAttr>(
      tianchenrv::target::rvv::getRVVSelectedMaskPolicyAttrName());
  auto vectorTypeAttr = variant->getAttrOfType<mlir::StringAttr>(
      tianchenrv::target::rvv::getRVVSelectedVectorTypeAttrName());
  auto vectorSuffixAttr = variant->getAttrOfType<mlir::StringAttr>(
      tianchenrv::target::rvv::getRVVSelectedVectorSuffixAttrName());
  auto setvlSuffixAttr = variant->getAttrOfType<mlir::StringAttr>(
      tianchenrv::target::rvv::getRVVSelectedSetVLSuffixAttrName());

  if (!sewAttr || sewAttr.getInt() != shape->sewBits || !lmulAttr ||
      lmulAttr.getValue() != shape->lmul || !tailAttr ||
      tailAttr.getValue() != shape->tailPolicy || !maskAttr ||
      maskAttr.getValue() != shape->maskPolicy || !vectorTypeAttr ||
      vectorTypeAttr.getValue() != shape->vectorType || !vectorSuffixAttr ||
      vectorSuffixAttr.getValue() != shape->vectorSuffix || !setvlSuffixAttr ||
      setvlSuffixAttr.getValue() != shape->setvlSuffix)
    return makeRVVPluginError(
        "materialized RVV variant selected vector-shape metadata must match "
        "the plugin capability config");

  return shape;
}

llvm::Error validateRVVCapacityMirror(tcrv::exec::VariantOp variant) {
  mlir::Attribute rawVLenB = variant->getAttr(kRVVVLenBBytesAttrName);
  mlir::Attribute rawI32Lanes = variant->getAttr(kRVVI32M1LanesAttrName);
  if (!rawVLenB && !rawI32Lanes)
    return llvm::Error::success();
  if (!rawVLenB || !rawI32Lanes)
    return makeRVVPluginError(
        "RVV capacity metadata requires both vlenb bytes and i32 m1 lanes");

  auto vlenb = llvm::dyn_cast<mlir::IntegerAttr>(rawVLenB);
  auto lanes = llvm::dyn_cast<mlir::IntegerAttr>(rawI32Lanes);
  if (!vlenb || !lanes)
    return makeRVVPluginError("RVV capacity metadata must be integer attrs");
  if (vlenb.getInt() <= 0 || lanes.getInt() <= 0 || vlenb.getInt() < 4 ||
      vlenb.getInt() % 4 != 0 || vlenb.getInt() / 4 != lanes.getInt())
    return makeRVVPluginError(
        "RVV capacity metadata requires i32 lanes to equal vlenb bytes "
        "divided by four");
  return llvm::Error::success();
}

llvm::Error addSelectedCapacityMetadataToPlan(VariantEmissionPlan &plan,
                                              tcrv::exec::VariantOp variant) {
  auto vlenb = variant->getAttrOfType<mlir::IntegerAttr>(kRVVVLenBBytesAttrName);
  auto lanes =
      variant->getAttrOfType<mlir::IntegerAttr>(kRVVI32M1LanesAttrName);
  if (!vlenb && !lanes)
    return llvm::Error::success();
  if (llvm::Error error = validateRVVCapacityMirror(variant))
    return error;

  plan.addSelectedPlanMetadata(kRVVVLenBBytesAttrName,
                               std::to_string(vlenb.getInt()),
                               kSelectedRVVCapacityMetadataRole,
                               kSelectedRVVCapacityMetadataNote);
  plan.addSelectedPlanMetadata(kRVVI32M1LanesAttrName,
                               std::to_string(lanes.getInt()),
                               kSelectedRVVCapacityMetadataRole,
                               kSelectedRVVCapacityMetadataNote);
  return llvm::Error::success();
}

void copyOptionalStringAttr(mlir::OperationState &state, mlir::Operation *from,
                            llvm::StringRef sourceName,
                            llvm::StringRef targetName) {
  if (auto attr = from->getAttrOfType<mlir::StringAttr>(sourceName))
    state.addAttribute(targetName, attr);
}

void copyOptionalIntegerAttr(mlir::OperationState &state, mlir::Operation *from,
                             llvm::StringRef sourceName,
                             llvm::StringRef targetName) {
  if (auto attr = from->getAttrOfType<mlir::IntegerAttr>(sourceName))
    state.addAttribute(targetName, attr);
}

tcrv::rvv::LoweringBoundaryOp materializeRVVBoundaryOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role) {
  builder.getContext()->getOrLoadDialect<tcrv::rvv::TCRVRVVDialect>();
  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(
      variant.getLoc(), tcrv::rvv::LoweringBoundaryOp::getOperationName());
  state.addAttribute(kBoundarySourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kBoundarySelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kBoundaryOriginAttrName,
                     builder.getStringAttr(kRVVPluginName));
  state.addAttribute(kBoundaryRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kBoundaryStatusAttrName,
                     builder.getStringAttr(kUnsupportedBoundaryStatus));
  state.addAttribute(kBoundaryRequiredCapabilitiesAttrName,
                     requiredCapabilities);
  copyOptionalIntegerAttr(state, variant.getOperation(), kRVVVLenBBytesAttrName,
                          kBoundaryVLenBBytesAttrName);
  copyOptionalIntegerAttr(state, variant.getOperation(), kRVVI32M1LanesAttrName,
                          kBoundaryI32M1LanesAttrName);
  copyOptionalStringAttr(
      state, variant.getOperation(),
      tianchenrv::target::rvv::getRVVSelectedVectorShapeAttrName(),
      kBoundarySelectedVectorShapeAttrName);
  copyOptionalIntegerAttr(
      state, variant.getOperation(),
      tianchenrv::target::rvv::getRVVSelectedVectorSEWAttrName(),
      kBoundarySelectedVectorSEWAttrName);
  copyOptionalStringAttr(
      state, variant.getOperation(),
      tianchenrv::target::rvv::getRVVSelectedVectorLMULAttrName(),
      kBoundarySelectedVectorLMULAttrName);
  copyOptionalStringAttr(
      state, variant.getOperation(),
      tianchenrv::target::rvv::getRVVSelectedTailPolicyAttrName(),
      kBoundarySelectedTailPolicyAttrName);
  copyOptionalStringAttr(
      state, variant.getOperation(),
      tianchenrv::target::rvv::getRVVSelectedMaskPolicyAttrName(),
      kBoundarySelectedMaskPolicyAttrName);
  copyOptionalStringAttr(
      state, variant.getOperation(),
      tianchenrv::target::rvv::getRVVSelectedVectorTypeAttrName(),
      kBoundarySelectedVectorTypeAttrName);
  copyOptionalStringAttr(
      state, variant.getOperation(),
      tianchenrv::target::rvv::getRVVSelectedVectorSuffixAttrName(),
      kBoundarySelectedVectorSuffixAttrName);
  copyOptionalStringAttr(
      state, variant.getOperation(),
      tianchenrv::target::rvv::getRVVSelectedSetVLSuffixAttrName(),
      kBoundarySelectedSetVLSuffixAttrName);
  state.addAttribute(
      kBoundaryUnsupportedReasonAttrName,
      builder.getStringAttr(
          "RVV selected lowering boundary is unsupported metadata only; no "
          "RVV lowering or artifact is produced"));
  return llvm::cast<tcrv::rvv::LoweringBoundaryOp>(builder.create(state));
}

const rvv::RVVExtensionPlugin &getBuiltinRVVExtensionPlugin() {
  static const rvv::RVVExtensionPlugin plugin;
  return plugin;
}

} // namespace

namespace rvv {

llvm::StringRef getRVVExtensionPluginName() { return kRVVPluginName; }

llvm::StringRef getRVVExtensionPluginVersion() { return kRVVPluginVersion; }

llvm::StringRef getRVVCapabilityID() { return kRVVCapabilityID; }

llvm::StringRef getRVVCapabilityKind() { return kRVVCapabilityKind; }

llvm::StringRef getRVVPreferredCapabilitySymbol() {
  return kRVVPreferredCapabilitySymbol;
}

llvm::StringRef getRVVFirstSliceVariantName() {
  return kRVVFirstSliceVariantName;
}

llvm::StringRef getRVVPolicyAttrName() { return kRVVPolicyAttrName; }

RVVExtensionPlugin::RVVExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kRVVCapabilityID, kRVVCapabilityKind,
      "RVV first-slice vector ISA capability participation; target "
      "availability is supplied by tcrv.exec.capability metadata"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M1SEW32CapabilityID(), "isa-vector-config",
      "RVV first-slice i32m1 SEW=32 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M1LMULM1CapabilityID(), "isa-vector-config",
      "RVV first-slice i32m1 LMUL=m1 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M1TailAgnosticCapabilityID(), "isa-vector-config",
      "RVV first-slice tail agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M1MaskAgnosticCapabilityID(), "isa-vector-config",
      "RVV first-slice mask agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2SEW32CapabilityID(), "isa-vector-config",
      "RVV i32m2 SEW=32 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2LMULM2CapabilityID(), "isa-vector-config",
      "RVV i32m2 LMUL=m2 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2TailAgnosticCapabilityID(), "isa-vector-config",
      "RVV i32m2 tail agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2MaskAgnosticCapabilityID(), "isa-vector-config",
      "RVV i32m2 mask agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig().sewCapabilityID,
      "isa-vector-config",
      "RVV i64m1 SEW=64 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig().lmulCapabilityID,
      "isa-vector-config",
      "RVV i64m1 LMUL=m1 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig()
          .tailPolicyCapabilityID,
      "isa-vector-config",
      "RVV i64m1 tail agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig()
          .maskPolicyCapabilityID,
      "isa-vector-config",
      "RVV i64m1 mask agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::
          getRVVI32BinarySelectedVectorShapeCapabilityID(),
      "isa-vector-config",
      "RVV i32 selected vector-shape selector capability"));
}

llvm::StringRef RVVExtensionPlugin::getName() const {
  return kRVVPluginName;
}

llvm::StringRef RVVExtensionPlugin::getVersion() const {
  return kRVVPluginVersion;
}

llvm::ArrayRef<PluginCapability>
RVVExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void RVVExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<tcrv::rvv::TCRVRVVDialect>();
}

bool RVVExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return request.getHighLevelOp() && hasAvailableRVVCapability(request);
}

llvm::Error RVVExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildRVVFirstSliceProposal(request);
  if (!proposal) {
    llvm::consumeError(proposal.takeError());
    return llvm::Error::success();
  }

  out.push_back(*proposal);
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildRVVFirstSliceProposal(request);
  if (!proposal) {
    std::string reason =
        sanitizeRVVDeclineReason(llvm::toString(proposal.takeError()));
    out.addRecoverableDecline(kRVVPluginName, reason);
    return llvm::Error::success();
  }

  out.addProposal(*proposal);
  return llvm::Error::success();
}

llvm::Expected<support::TargetCapabilitySet>
RVVExtensionPlugin::buildTargetCapabilitiesFromProbeFacts(
    const RVVProbeCapabilityFacts &facts) const {
  return buildRVVTargetCapabilitiesFromProbeFacts(facts);
}

llvm::Error RVVExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVPluginError(
        "legality verification requires a materialized tcrv.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVPluginError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  auto policyAttr = variant->getAttr(kRVVPolicyAttrName);
  auto typedPolicy = llvm::dyn_cast_if_present<tcrv::rvv::PolicyAttr>(
      policyAttr);
  if (!typedPolicy)
    return makeRVVPluginError(
        "materialized RVV variant tcrv_rvv.policy requires typed "
        "#tcrv_rvv.policy metadata");
  if (typedPolicy.getTail() != tcrv::rvv::TailPolicy::Agnostic ||
      typedPolicy.getMask() != tcrv::rvv::MaskPolicy::Agnostic)
    return makeRVVPluginError(
        "materialized RVV variant requires agnostic tail and mask policy");

  auto marchAttr =
      variant->getAttrOfType<mlir::StringAttr>(kRVVRequiredMarchAttrName);
  if (!marchAttr || !isSingleBoundedFactString(marchAttr.getValue()))
    return makeRVVPluginError(
        "materialized RVV variant requires bounded tcrv_rvv.required_march");

  llvm::Expected<const RVVI32VectorShapeConfig *> shape =
      getVariantSelectedShape(variant);
  if (!shape)
    return shape.takeError();

  for (llvm::StringRef requiredID :
       {llvm::StringRef(kRVVCapabilityID), (*shape)->sewCapabilityID,
        (*shape)->lmulCapabilityID, (*shape)->tailPolicyCapabilityID,
        (*shape)->maskPolicyCapabilityID}) {
    if (!variantRequiresCapabilityID(variant, request.getCapabilities(),
                                     requiredID))
      return makeRVVPluginError(
          llvm::Twine("materialized RVV variant requires capability id '") +
          requiredID + "'");
  }

  if (llvm::Error error = validateRVVCapacityMirror(variant))
    return error;
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "cost estimation requires a materialized tcrv.exec.variant");

  out = VariantCostEstimate();
  out.setScore(1.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kRVVPluginName);
  out.setVariantSymbol(request.getVariant().getSymName());
  if (auto lanes = request.getVariant()->getAttrOfType<mlir::IntegerAttr>(
          kRVVI32M1LanesAttrName)) {
    out.setScore(1.0 / static_cast<double>(lanes.getInt()));
    out.setExplanation(
        (llvm::Twine("RVV metadata-only first slice; capability-derived "
                     "base_i32_m1_lanes=") +
         std::to_string(lanes.getInt()) +
         " is a plugin-local selection heuristic input, not a runtime "
         "performance claim")
            .str());
  } else {
    out.setExplanation("RVV metadata-only first slice; no runtime performance "
                       "claim");
  }
  out.setPolicy("plugin-local RVV capability participation");
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "emission readiness requires a materialized tcrv.exec.variant");

  out = VariantEmissionStatus::getUnsupported(
      kRVVPluginName, request.getVariant().getSymName(),
      "RVV first slice has no materialized EmitC lowering, runtime ABI, or "
      "artifact route; this is an unsupported diagnostic boundary");
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "emission planning requires a materialized tcrv.exec.variant");

  if (!request.getKernel())
    return makeRVVPluginError(
        "emission planning requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality))
    return error;

  llvm::Expected<const RVVI32VectorShapeConfig *> shape =
      getVariantSelectedShape(request.getVariant());
  if (!shape)
    return shape.takeError();

  out = VariantEmissionPlan::getUnsupported(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "RVV first slice has no materialized EmitC lowering, runtime ABI, "
      "artifact contract, or executable emission path");
  out.setEmissionKind("rvv-unsupported-metadata-boundary");
  out.setLoweringPipeline("rvv-none-executable-unsupported");
  out.setRuntimeABI("rvv-none-executable-unsupported");
  out.setRuntimeABIKind("unsupported-plugin-runtime-abi");
  out.setRuntimeABIName("unsupported-emission-runtime-abi");
  out.setRuntimeGlueRole("no-runtime-glue-unsupported");
  out.setArtifactKind("unsupported-emission-diagnostic");
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  if (llvm::Error error = addSelectedVectorShapeMetadataToPlan(out, **shape))
    return error;
  if (llvm::Error error =
          addSelectedCapacityMetadataToPlan(out, request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVPluginError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality))
    return error;

  tcrv::rvv::LoweringBoundaryOp boundary = materializeRVVBoundaryOp(
      request.getBuilder(), request.getKernel(), request.getVariant(),
      request.getRole());
  out = VariantLoweringBoundaryResult::getMaterialized(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      boundary.getOperation());
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  auto boundary =
      llvm::dyn_cast_if_present<tcrv::rvv::LoweringBoundaryOp>(
          request.getBoundary());
  if (!boundary)
    return makeRVVPluginError(
        "selected RVV lowering boundary must be tcrv_rvv.lowering_boundary");

  auto sourceKernel = boundary->getAttrOfType<mlir::StringAttr>(
      kBoundarySourceKernelAttrName);
  auto selectedVariant = boundary->getAttrOfType<mlir::FlatSymbolRefAttr>(
      kBoundarySelectedVariantAttrName);
  auto origin =
      boundary->getAttrOfType<mlir::StringAttr>(kBoundaryOriginAttrName);
  auto role = boundary->getAttrOfType<mlir::StringAttr>(kBoundaryRoleAttrName);
  auto status =
      boundary->getAttrOfType<mlir::StringAttr>(kBoundaryStatusAttrName);
  if (!sourceKernel ||
      sourceKernel.getValue() != request.getKernel().getSymName() ||
      !selectedVariant ||
      selectedVariant.getValue() != request.getVariant().getSymName() ||
      !origin || origin.getValue() != kRVVPluginName || !role ||
      role.getValue() != stringifyVariantEmissionRole(request.getRole()) ||
      !status || status.getValue() != kUnsupportedBoundaryStatus)
    return makeRVVPluginError(
        "selected RVV lowering boundary metadata must match the selected path");

  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::configureTargetSupportExtensionBundle(
    target::ExtensionBundle &bundle) const {
  bundle.addRequiredDialectName("tcrv_rvv");
  bundle.addLoweringBoundaryOp("tcrv_rvv.lowering_boundary");
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

} // namespace tianchenrv::plugin
