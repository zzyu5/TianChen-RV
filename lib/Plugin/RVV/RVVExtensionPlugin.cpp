#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/I32BinaryFamilyRegistry.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVI32BinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
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
constexpr llvm::StringLiteral kFrontendLoweringAttrName(
    "tcrv_frontend_lowering");
constexpr llvm::StringLiteral kRVVI32VAddLoweringDescriptorAttrName(
    "tcrv_rvv.lowering_descriptor");
constexpr llvm::StringLiteral kRVVSmokeProbeDescriptorAttrName(
    "tcrv_rvv.smoke_probe_descriptor");
constexpr llvm::StringLiteral kRVVSmokeProbeDescriptorValue(
    "standalone-c-toolchain-smoke-probe.v1");
constexpr llvm::StringLiteral kRVVI32VAddElementCountAttrName(
    "tcrv_rvv.element_count");
constexpr llvm::StringLiteral kRVVVLenBBytesAttrName(
    "tcrv_rvv.vlenb_bytes");
constexpr llvm::StringLiteral kRVVI32M1LanesAttrName(
    "tcrv_rvv.base_i32_m1_lanes");
constexpr llvm::StringLiteral kRVVBoundaryVLenBBytesAttrName("vlenb_bytes");
constexpr llvm::StringLiteral kRVVBoundaryI32M1LanesAttrName(
    "base_i32_m1_lanes");
constexpr llvm::StringLiteral kRVVSelectedVectorShapeAttrName(
    "tcrv_rvv.selected_vector_shape");
constexpr llvm::StringLiteral kRVVSelectedVectorSEWAttrName(
    "tcrv_rvv.selected_vector_sew");
constexpr llvm::StringLiteral kRVVSelectedVectorLMULAttrName(
    "tcrv_rvv.selected_vector_lmul");
constexpr llvm::StringLiteral kRVVSelectedTailPolicyAttrName(
    "tcrv_rvv.selected_tail_policy");
constexpr llvm::StringLiteral kRVVSelectedMaskPolicyAttrName(
    "tcrv_rvv.selected_mask_policy");
constexpr llvm::StringLiteral kRVVSelectedVectorTypeAttrName(
    "tcrv_rvv.selected_vector_type");
constexpr llvm::StringLiteral kRVVSelectedVectorSuffixAttrName(
    "tcrv_rvv.selected_vector_suffix");
constexpr llvm::StringLiteral kRVVSelectedSetVLSuffixAttrName(
    "tcrv_rvv.selected_setvl_suffix");
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
constexpr llvm::StringLiteral kSelectedMABIPropertyName("selected_mabi");
constexpr llvm::StringLiteral kSelectedMarchValuePropertyName("value");
constexpr llvm::StringLiteral kCapabilitySummaryAttrName(
    "capability_summary");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kUnsupportedReasonAttrName(
    "unsupported_reason");
constexpr llvm::StringLiteral kUnsupportedStatusValue("unsupported");
constexpr llvm::StringLiteral kElementCountAttrName("element_count");
constexpr llvm::StringLiteral kMicrokernelRequiredMarchAttrName(
    "required_march");
constexpr llvm::StringLiteral kSelectedMABIAttrName("selected_mabi");
constexpr llvm::StringLiteral kSEWAttrName("sew");
constexpr llvm::StringLiteral kLMULAttrName("lmul");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kBufferRoleAttrName("buffer_role");
constexpr llvm::StringLiteral kTargetAttrName("target");
constexpr llvm::StringLiteral kRVVSmokeProbeCapabilityID("rvv.smoke_probe");
constexpr llvm::StringLiteral kSmokeProbeEmissionPath(
    "rvv-smoke-probe-standalone-c-source-export");
constexpr llvm::StringLiteral kSmokeProbeEmissionKind(
    "rvv-smoke-probe-standalone-c-source");
constexpr llvm::StringLiteral kSmokeProbeRouteID(
    "tcrv-export-rvv-smoke-probe-c");
constexpr llvm::StringLiteral kSmokeProbeRuntimeABI(
    "rvv-smoke-probe-standalone-c-main.v1");
constexpr llvm::StringLiteral kSmokeProbeRuntimeABIKind(
    "rvv-smoke-probe-standalone-c-main");
constexpr llvm::StringLiteral kSmokeProbeRuntimeGlueRole(
    "rvv-smoke-probe-standalone-main");
constexpr llvm::StringLiteral kSmokeProbeArtifactKind("standalone-c-source");
constexpr llvm::StringLiteral kSelectedRVVCapacityMetadataRole(
    "rvv-base-capacity-fact");
constexpr llvm::StringLiteral kSelectedRVVCapacityMetadataNote(
    "base i32 M1 capacity fact from target/profile evidence; not selected "
    "vector shape, runtime input, VL/AVL, or performance evidence");
constexpr std::int64_t kDefaultI32VAddElementCount = 16;
constexpr std::uint64_t kI32VAddCapacitySampleVectors = 4;
constexpr std::int64_t kMaxI32VAddElementCount = 64;

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

using RVVI32MicrokernelKind =
    tianchenrv::target::i32_binary::I32BinaryFamilyKind;
using RVVI32FamilyDescriptor =
    tianchenrv::target::i32_binary::I32BinaryFamilyDescriptor;
using RVVI32RegistryMicrokernelDescriptor =
    tianchenrv::target::i32_binary::RVVI32MicrokernelFamilyDescriptor;
using RVVI32VectorShapeConfig =
    tianchenrv::target::rvv::RVVI32VectorShapeConfig;
using RVVBinaryFamilyDescriptor =
    tianchenrv::target::rvv::RVVBinaryFamilyDescriptor;
using RVVBinaryIntrinsicDescriptor =
    tianchenrv::target::rvv::RVVBinaryIntrinsicDescriptor;

const RVVI32VectorShapeConfig &getI32M1ConfigSpec() {
  return tianchenrv::target::rvv::getI32M1VectorShapeConfig();
}

const RVVI32VectorShapeConfig &getI32M2ConfigSpec() {
  return tianchenrv::target::rvv::getI32M2VectorShapeConfig();
}

struct RVVCapabilityPropertyView {
  std::string architecture;
  std::string isaVectorHints;
  std::string selectedMarch;
  std::uint64_t hartCount = 0;
  std::optional<std::uint64_t> vlenbBytes;
  std::optional<std::uint64_t> i32M1LaneCount;
  const RVVI32VectorShapeConfig *i32Config = nullptr;
};

struct RVVI32MicrokernelFamilySpec {
  const RVVI32FamilyDescriptor *family;
  llvm::StringRef descriptorNoun;
  llvm::StringRef emissionPath;
  llvm::StringRef supportedMessage;

  RVVI32MicrokernelKind getKind() const { return family->kind; }
  llvm::StringRef getLoweringDescriptor() const {
    return family->loweringDescriptor;
  }
  const RVVI32RegistryMicrokernelDescriptor &getRVV() const {
    return family->rvv;
  }
};

const RVVI32MicrokernelFamilySpec &getI32VAddFamilySpec() {
  static const RVVI32MicrokernelFamilySpec spec{
      &tianchenrv::target::i32_binary::getI32VAddFamilyDescriptor(),
      "finite RVV i32-vadd lowering descriptor",
      "rvv-explicit-i32-vadd-microkernel-c-source-export",
      "explicit RVV i32 vector-add microkernel C source export provides a "
      "library-style runtime-callable C ABI function for this selected path; "
      "any self-check main is an explicit harness export and is not the "
      "default artifact contract; this is not generic RVV lowering, runtime "
      "integration, arbitrary kernel emission, correctness, or performance "
      "evidence"};
  return spec;
}

const RVVI32MicrokernelFamilySpec &getI32VSubFamilySpec() {
  static const RVVI32MicrokernelFamilySpec spec{
      &tianchenrv::target::i32_binary::getI32VSubFamilyDescriptor(),
      "finite RVV i32-vsub lowering descriptor",
      "rvv-explicit-i32-vsub-microkernel-c-source-export",
      "explicit RVV i32 vector-subtract microkernel C source export provides "
      "a library-style runtime-callable C ABI function for this selected "
      "path; any self-check main is an explicit harness export and is not the "
      "default artifact contract; this is not generic RVV lowering, runtime "
      "integration, arbitrary kernel emission, correctness, or performance "
      "evidence"};
  return spec;
}

const RVVI32MicrokernelFamilySpec &getI32VMulFamilySpec() {
  static const RVVI32MicrokernelFamilySpec spec{
      &tianchenrv::target::i32_binary::getI32VMulFamilyDescriptor(),
      "finite RVV i32-vmul lowering descriptor",
      "rvv-explicit-i32-vmul-microkernel-c-source-export",
      "explicit RVV i32 vector-multiply microkernel C source export provides "
      "a library-style runtime-callable C ABI function for this selected "
      "path; any self-check main is an explicit harness export and is not the "
      "default artifact contract; this is not generic RVV lowering, runtime "
      "integration, arbitrary kernel emission, correctness, or performance "
      "evidence"};
  return spec;
}

const RVVI32MicrokernelFamilySpec *
lookupI32MicrokernelFamilyByDescriptor(llvm::StringRef descriptor) {
  const RVVI32FamilyDescriptor *family =
      tianchenrv::target::i32_binary::lookupI32BinaryFamilyByLoweringDescriptor(
          descriptor);
  if (!family)
    return nullptr;
  if (family->kind == RVVI32MicrokernelKind::Add)
    return &getI32VAddFamilySpec();
  if (family->kind == RVVI32MicrokernelKind::Sub)
    return &getI32VSubFamilySpec();
  if (family->kind == RVVI32MicrokernelKind::Mul)
    return &getI32VMulFamilySpec();
  return nullptr;
}

const RVVI32MicrokernelFamilySpec *
getI32MicrokernelFamilyByDescriptor(
    const RVVI32FamilyDescriptor &family) {
  if (family.kind == RVVI32MicrokernelKind::Add)
    return &getI32VAddFamilySpec();
  if (family.kind == RVVI32MicrokernelKind::Sub)
    return &getI32VSubFamilySpec();
  if (family.kind == RVVI32MicrokernelKind::Mul)
    return &getI32VMulFamilySpec();
  return nullptr;
}

struct RVVI32MicrokernelMaterializationPlan {
  const RVVI32MicrokernelFamilySpec *family = nullptr;
  const RVVI32VectorShapeConfig *i32Config = nullptr;
  std::int64_t elementCount = 0;
  std::string requiredMarch;
  std::optional<std::string> selectedMABI;
};

struct RVVI64MicrokernelMaterializationPlan {
  RVVBinaryIntrinsicDescriptor descriptor;
  std::int64_t elementCount = 0;
  std::string requiredMarch;
  std::optional<std::string> selectedMABI;
};

struct RVVCapacityMetadata {
  std::int64_t vlenbBytes = 0;
  std::int64_t i32M1Lanes = 0;
};

std::int64_t
deriveI32VAddDescriptorElementCount(const RVVCapabilityPropertyView &view) {
  if (!view.i32M1LaneCount)
    return kDefaultI32VAddElementCount;

  if (*view.i32M1LaneCount >=
      static_cast<std::uint64_t>(kMaxI32VAddElementCount) /
          kI32VAddCapacitySampleVectors)
    return kMaxI32VAddElementCount;

  return static_cast<std::int64_t>(*view.i32M1LaneCount *
                                   kI32VAddCapacitySampleVectors);
}

bool hasAvailableRVVCapability(const VariantProposalRequest &request) {
  return request.getKernel() &&
         request.getCapabilities().isCapabilityAvailableByID(kRVVCapabilityID);
}

bool hasAvailableRVVSmokeProbeCapability(
    const support::TargetCapabilitySet &capabilities) {
  return capabilities.isCapabilityAvailableByID(kRVVSmokeProbeCapabilityID);
}

bool containsForbiddenRVVPropertyText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key");
}

bool isSingleBoundedRVVPropertyText(llvm::StringRef value) {
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

llvm::Error validateRVVPropertyText(llvm::StringRef context,
                                    llvm::StringRef propertyName,
                                    llvm::StringRef value) {
  if (!isSingleBoundedRVVPropertyText(value))
    return makeRVVPluginError(llvm::Twine(context) + " property '" +
                              propertyName +
                              "' must be a bounded single-line fact");

  if (containsForbiddenRVVPropertyText(value))
    return makeRVVPluginError(llvm::Twine(context) + " property '" +
                              propertyName +
                              "' must not contain secret-like or raw-log text");

  return llvm::Error::success();
}

llvm::Expected<std::string>
getRequiredRVVProperty(const support::CapabilityDescriptor &capability,
                       llvm::StringRef propertyName) {
  llvm::StringRef value = capability.getProperty(propertyName).trim();
  std::string context = (llvm::Twine("capability id '") +
                         capability.getID() + "'").str();
  if (value.empty())
    return makeRVVPluginError(llvm::Twine(context) +
                              " requires preserved property '" +
                              propertyName + "'");

  if (llvm::Error error = validateRVVPropertyText(context, propertyName, value))
    return std::move(error);

  return value.str();
}

llvm::Expected<std::uint64_t>
getRequiredPositiveIntegerRVVProperty(
    const support::CapabilityDescriptor &capability,
    llvm::StringRef propertyName) {
  llvm::Expected<std::string> property =
      getRequiredRVVProperty(capability, propertyName);
  if (!property)
    return property.takeError();

  llvm::StringRef value(*property);
  if (!llvm::all_of(value, [](char character) {
        unsigned char byte = static_cast<unsigned char>(character);
        return std::isdigit(byte);
      })) {
    return makeRVVPluginError(llvm::Twine("capability id '") +
                              capability.getID() + "' property '" +
                              propertyName +
                              "' must be a positive integer");
  }

  std::uint64_t parsed = 0;
  if (value.getAsInteger(10, parsed) || parsed == 0)
    return makeRVVPluginError(llvm::Twine("capability id '") +
                              capability.getID() + "' property '" +
                              propertyName +
                              "' must be a positive integer");

  return parsed;
}

llvm::Error requireAvailableCapability(
    const support::TargetCapabilitySet &capabilities, llvm::StringRef id,
    const support::CapabilityDescriptor *&out) {
  out = capabilities.lookupProviderByID(id);
  if (!out)
    return makeRVVPluginError(llvm::Twine("RVV property decision requires "
                                          "capability id '") +
                              id + "'");
  if (!out->isAvailable())
    return makeRVVPluginError(llvm::Twine("RVV property decision requires "
                                          "available capability id '") +
                              id + "'");
  return llvm::Error::success();
}

llvm::Error requireFirstSliceSEW32Capability(
    const support::TargetCapabilitySet &capabilities,
    const RVVI32VectorShapeConfig &config) {
  const support::CapabilityDescriptor *capability = nullptr;
  if (llvm::Error error = requireAvailableCapability(
          capabilities, config.sewCapabilityID, capability))
    return std::move(error);

  llvm::Expected<std::uint64_t> sew =
      getRequiredPositiveIntegerRVVProperty(*capability, kSEWBitsPropertyName);
  if (!sew)
    return sew.takeError();
  if (*sew != static_cast<std::uint64_t>(config.sewBits))
    return makeRVVPluginError(llvm::Twine("RVV first-slice config capability "
                                          "id '") +
                              config.sewCapabilityID +
                              "' property 'sew_bits' must be " +
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
      getRequiredRVVProperty(*capability, propertyName);
  if (!property)
    return property.takeError();
  if (*property != expectedValue)
    return makeRVVPluginError(llvm::Twine("RVV first-slice config capability "
                                          "id '") +
                              id + "' property '" + propertyName +
                              "' must be '" + expectedValue + "'");
  return llvm::Error::success();
}

llvm::Error verifyFirstSliceConfigCapabilities(
    const support::TargetCapabilitySet &capabilities,
    const RVVI32VectorShapeConfig &config) {
  if (llvm::Error error =
          requireFirstSliceSEW32Capability(capabilities, config))
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

llvm::Error validateSelectedVectorShapeMetadata(
    mlir::Operation *op, llvm::StringRef context,
    const RVVI32VectorShapeConfig &config, llvm::StringRef shapeAttrName,
    llvm::StringRef sewAttrName, llvm::StringRef lmulAttrName,
    llvm::StringRef tailPolicyAttrName, llvm::StringRef maskPolicyAttrName,
    llvm::StringRef vectorTypeAttrName, llvm::StringRef vectorSuffixAttrName,
    llvm::StringRef setvlSuffixAttrName) {
  bool hasAnyMetadata = op->hasAttr(shapeAttrName) ||
                        op->hasAttr(sewAttrName) ||
                        op->hasAttr(lmulAttrName) ||
                        op->hasAttr(tailPolicyAttrName) ||
                        op->hasAttr(maskPolicyAttrName) ||
                        op->hasAttr(vectorTypeAttrName) ||
                        op->hasAttr(vectorSuffixAttrName) ||
                        op->hasAttr(setvlSuffixAttrName);
  if (!hasAnyMetadata)
    return llvm::Error::success();

  auto shape = op->getAttrOfType<mlir::StringAttr>(shapeAttrName);
  auto sew = op->getAttrOfType<mlir::IntegerAttr>(sewAttrName);
  auto lmul = op->getAttrOfType<mlir::StringAttr>(lmulAttrName);
  auto tailPolicy =
      op->getAttrOfType<mlir::StringAttr>(tailPolicyAttrName);
  auto maskPolicy =
      op->getAttrOfType<mlir::StringAttr>(maskPolicyAttrName);
  auto vectorType =
      op->getAttrOfType<mlir::StringAttr>(vectorTypeAttrName);
  auto vectorSuffix =
      op->getAttrOfType<mlir::StringAttr>(vectorSuffixAttrName);
  auto setvlSuffix =
      op->getAttrOfType<mlir::StringAttr>(setvlSuffixAttrName);
  if (!shape || !sew || !lmul || !tailPolicy || !maskPolicy || !vectorType ||
      !vectorSuffix || !setvlSuffix)
    return makeRVVPluginError(llvm::Twine(context) +
                              " selected vector-shape metadata must be "
                              "complete when any selected-shape attribute is "
                              "present");

  if (shape.getValue().trim() != config.shapeID)
    return makeRVVPluginError(llvm::Twine(context) +
                              " selected vector-shape id must be '" +
                              config.shapeID + "'");
  if (sew.getInt() != config.sewBits)
    return makeRVVPluginError(llvm::Twine(context) +
                              " selected vector-shape sew must be '" +
                              llvm::Twine(config.sewBits) + "'");
  if (lmul.getValue().trim() != config.lmul)
    return makeRVVPluginError(llvm::Twine(context) +
                              " selected vector-shape lmul must be '" +
                              config.lmul + "'");
  if (tailPolicy.getValue().trim() != config.tailPolicy)
    return makeRVVPluginError(llvm::Twine(context) +
                              " selected vector-shape tail policy must be '" +
                              config.tailPolicy + "'");
  if (maskPolicy.getValue().trim() != config.maskPolicy)
    return makeRVVPluginError(llvm::Twine(context) +
                              " selected vector-shape mask policy must be '" +
                              config.maskPolicy + "'");
  if (vectorType.getValue().trim() != config.vectorType)
    return makeRVVPluginError(llvm::Twine(context) +
                              " selected vector-shape vector type must be '" +
                              config.vectorType + "'");
  if (vectorSuffix.getValue().trim() != config.vectorSuffix)
    return makeRVVPluginError(llvm::Twine(context) +
                              " selected vector-shape vector suffix must be '" +
                              config.vectorSuffix + "'");
  if (setvlSuffix.getValue().trim() != config.setvlSuffix)
    return makeRVVPluginError(llvm::Twine(context) +
                              " selected vector-shape setvl suffix must be '" +
                              config.setvlSuffix + "'");
  return llvm::Error::success();
}

void addSelectedVectorShapeMetadataToProposal(
    VariantProposal &proposal, mlir::MLIRContext *context,
    const RVVI32VectorShapeConfig &config) {
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context, kRVVSelectedVectorShapeAttrName),
      mlir::StringAttr::get(context, config.shapeID));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context, kRVVSelectedVectorSEWAttrName),
      mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                             config.sewBits));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context, kRVVSelectedVectorLMULAttrName),
      mlir::StringAttr::get(context, config.lmul));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context, kRVVSelectedTailPolicyAttrName),
      mlir::StringAttr::get(context, config.tailPolicy));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context, kRVVSelectedMaskPolicyAttrName),
      mlir::StringAttr::get(context, config.maskPolicy));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context, kRVVSelectedVectorTypeAttrName),
      mlir::StringAttr::get(context, config.vectorType));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context, kRVVSelectedVectorSuffixAttrName),
      mlir::StringAttr::get(context, config.vectorSuffix));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context, kRVVSelectedSetVLSuffixAttrName),
      mlir::StringAttr::get(context, config.setvlSuffix));
}

void addSelectedVectorShapeMetadataToOperationState(
    mlir::OperationState &state, mlir::MLIRContext *context,
    const RVVI32VectorShapeConfig &config) {
  state.addAttribute(kBoundarySelectedVectorShapeAttrName,
                     mlir::StringAttr::get(context, config.shapeID));
  state.addAttribute(kBoundarySelectedVectorSEWAttrName,
                     mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                            config.sewBits));
  state.addAttribute(kBoundarySelectedVectorLMULAttrName,
                     mlir::StringAttr::get(context, config.lmul));
  state.addAttribute(kBoundarySelectedTailPolicyAttrName,
                     mlir::StringAttr::get(context, config.tailPolicy));
  state.addAttribute(kBoundarySelectedMaskPolicyAttrName,
                     mlir::StringAttr::get(context, config.maskPolicy));
  state.addAttribute(kBoundarySelectedVectorTypeAttrName,
                     mlir::StringAttr::get(context, config.vectorType));
  state.addAttribute(kBoundarySelectedVectorSuffixAttrName,
                     mlir::StringAttr::get(context, config.vectorSuffix));
  state.addAttribute(kBoundarySelectedSetVLSuffixAttrName,
                     mlir::StringAttr::get(context, config.setvlSuffix));
}

llvm::Error addSelectedVectorShapeMetadataToPlan(
    VariantEmissionPlan &plan, tcrv::exec::VariantOp variant,
    const RVVI32VectorShapeConfig &config) {
  if (llvm::Error error =
          validateSelectedVectorShapeMetadata(
              variant.getOperation(),
              (llvm::Twine("selected RVV variant @") + variant.getSymName())
                  .str(),
              config, kRVVSelectedVectorShapeAttrName,
              kRVVSelectedVectorSEWAttrName, kRVVSelectedVectorLMULAttrName,
              kRVVSelectedTailPolicyAttrName, kRVVSelectedMaskPolicyAttrName,
              kRVVSelectedVectorTypeAttrName,
              kRVVSelectedVectorSuffixAttrName,
              kRVVSelectedSetVLSuffixAttrName))
    return error;

  llvm::SmallVector<
      tianchenrv::target::rvv::RVVI32VectorShapeSelectedPlanMetadataDescriptor,
      8>
      metadata;
  tianchenrv::target::rvv::appendRVVI32VectorShapeSelectedPlanMetadata(config,
                                                                       metadata);
  for (const auto &entry : metadata)
    plan.addSelectedPlanMetadata(entry.name.str(), entry.value.str(),
                                 entry.role.str(), entry.note.str());
  return llvm::Error::success();
}

llvm::Expected<const RVVI32VectorShapeConfig *>
selectExplicitI32BinaryVectorShapeCapability(
    const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *selector =
      capabilities.lookupProviderByID(
          tianchenrv::target::rvv::
              getRVVI32BinarySelectedVectorShapeCapabilityID());
  if (!selector)
    return static_cast<const RVVI32VectorShapeConfig *>(nullptr);

  if (!selector->isAvailable())
    return makeRVVPluginError(
        llvm::Twine("RVV i32 binary selected vector-shape capability id '") +
        tianchenrv::target::rvv::
            getRVVI32BinarySelectedVectorShapeCapabilityID() +
        "' must be available when present");

  llvm::Expected<std::string> selectedShape = getRequiredRVVProperty(
      *selector, tianchenrv::target::rvv::
                     getRVVI32BinarySelectedVectorShapePropertyName());
  if (!selectedShape)
    return selectedShape.takeError();

  const RVVI32VectorShapeConfig *config =
      tianchenrv::target::rvv::lookupFiniteI32VectorShapeConfigByShapeID(
          *selectedShape);
  if (!config)
    return makeRVVPluginError(
        llvm::Twine("RVV i32 binary selected vector-shape capability property "
                    "'") +
        tianchenrv::target::rvv::
            getRVVI32BinarySelectedVectorShapePropertyName() +
        "' must be one finite descriptor shape: '" +
        getI32M1ConfigSpec().shapeID + "' or '" +
        getI32M2ConfigSpec().shapeID + "'");

  if (llvm::Error error = verifyFirstSliceConfigCapabilities(capabilities,
                                                            *config))
    return std::move(error);
  return config;
}

llvm::Expected<const RVVI32VectorShapeConfig *>
selectAvailableFirstSliceConfigCapabilities(
    const support::TargetCapabilitySet &capabilities) {
  llvm::Expected<const RVVI32VectorShapeConfig *> explicitSelection =
      selectExplicitI32BinaryVectorShapeCapability(capabilities);
  if (!explicitSelection)
    return explicitSelection.takeError();
  if (*explicitSelection)
    return *explicitSelection;

  if (llvm::Error error =
          verifyFirstSliceConfigCapabilities(capabilities,
                                             getI32M1ConfigSpec())) {
    llvm::consumeError(std::move(error));
  } else {
    return &getI32M1ConfigSpec();
  }

  if (llvm::Error error =
          verifyFirstSliceConfigCapabilities(capabilities,
                                             getI32M2ConfigSpec())) {
    llvm::consumeError(std::move(error));
  } else {
    return &getI32M2ConfigSpec();
  }

  return makeRVVPluginError(
      "RVV property decision requires either the finite i32m1 config "
      "capability ids or the finite i32m2 config capability ids");
}

llvm::Expected<RVVCapabilityPropertyView>
buildRVVCapabilityPropertyView(
    const support::TargetCapabilitySet &capabilities,
    const RVVI32VectorShapeConfig *requiredConfig = nullptr) {
  const support::CapabilityDescriptor *rvvCapability = nullptr;
  if (llvm::Error error =
          requireAvailableCapability(capabilities, kRVVCapabilityID,
                                     rvvCapability))
    return std::move(error);

  llvm::Expected<std::string> architecture =
      getRequiredRVVProperty(*rvvCapability, kArchitecturePropertyName);
  if (!architecture)
    return architecture.takeError();
  if (llvm::StringRef(*architecture).lower() != "riscv64")
    return makeRVVPluginError(
        "capability id 'rvv' property 'architecture' must be riscv64");

  llvm::Expected<std::string> isaVectorHints =
      getRequiredRVVProperty(*rvvCapability, kISAVectorHintsPropertyName);
  if (!isaVectorHints)
    return isaVectorHints.takeError();
  if (!hasRVVVectorHint(*isaVectorHints))
    return makeRVVPluginError(
        "capability id 'rvv' property 'isa_vector_hints' must contain RVV "
        "vector evidence");

  const support::CapabilityDescriptor *hartCountCapability = nullptr;
  if (llvm::Error error = requireAvailableCapability(
          capabilities, rvv::getRVVHartCountCapabilityID(),
          hartCountCapability))
    return std::move(error);

  llvm::Expected<std::uint64_t> hartCount =
      getRequiredPositiveIntegerRVVProperty(*hartCountCapability,
                                            kHartCountPropertyName);
  if (!hartCount)
    return hartCount.takeError();

  const support::CapabilityDescriptor *vlenbCapability =
      capabilities.lookupProviderByID(rvv::getRVVVLenBBytesCapabilityID());
  const support::CapabilityDescriptor *i32LaneCapability =
      capabilities.lookupProviderByID(rvv::getRVVI32M1LaneCountCapabilityID());
  bool hasAvailableVLenB = vlenbCapability && vlenbCapability->isAvailable();
  bool hasAvailableI32Lanes =
      i32LaneCapability && i32LaneCapability->isAvailable();
  std::optional<std::uint64_t> vlenbBytes;
  std::optional<std::uint64_t> i32M1LaneCount;
  if (hasAvailableVLenB != hasAvailableI32Lanes)
    return makeRVVPluginError(
        "RVV vector capacity decision requires both available capability ids "
        "'rvv.vlenb_bytes' and 'rvv.i32_m1_lane_count'");
  if (hasAvailableVLenB) {
    llvm::Expected<std::uint64_t> parsedVLenB =
        getRequiredPositiveIntegerRVVProperty(*vlenbCapability,
                                              kVLenBBytesPropertyName);
    if (!parsedVLenB)
      return parsedVLenB.takeError();
    llvm::Expected<std::uint64_t> parsedI32Lanes =
        getRequiredPositiveIntegerRVVProperty(*i32LaneCapability,
                                              kI32M1LanesPropertyName);
    if (!parsedI32Lanes)
      return parsedI32Lanes.takeError();
    if (*parsedVLenB < sizeof(std::int32_t) ||
        *parsedVLenB % sizeof(std::int32_t) != 0 ||
        *parsedVLenB / sizeof(std::int32_t) != *parsedI32Lanes)
      return makeRVVPluginError(
          "RVV vector capacity decision requires i32 m1 lane count to match "
          "vlenb bytes divided by four");
    vlenbBytes = *parsedVLenB;
    i32M1LaneCount = *parsedI32Lanes;
  }

  const support::CapabilityDescriptor *compileRunCapability = nullptr;
  if (llvm::Error error = requireAvailableCapability(
          capabilities, rvv::getRVVProbeCompileRunCapabilityID(),
          compileRunCapability))
    return std::move(error);

  llvm::Expected<std::string> selectedMarch =
      getRequiredRVVProperty(*compileRunCapability, kSelectedMarchPropertyName);
  if (!selectedMarch)
    return selectedMarch.takeError();
  if (!hasRVVVectorHint(*selectedMarch))
    return makeRVVPluginError(
        "capability id 'rvv.probe.compile_run' property 'selected_march' must "
        "contain RVV vector evidence");

  if (const support::CapabilityDescriptor *selectedMarchCapability =
          capabilities.lookupByID(rvv::getRVVSelectedMarchCapabilityID())) {
    if (selectedMarchCapability->isAvailable()) {
      llvm::Expected<std::string> selectedMarchValue = getRequiredRVVProperty(
          *selectedMarchCapability, kSelectedMarchValuePropertyName);
      if (!selectedMarchValue)
        return selectedMarchValue.takeError();
      if (*selectedMarchValue != *selectedMarch)
        return makeRVVPluginError(
            "conflicting RVV property values between capability id "
            "'rvv.toolchain.march' property 'value' and capability id "
            "'rvv.probe.compile_run' property 'selected_march'");
    }
  }

  const RVVI32VectorShapeConfig *selectedConfig = requiredConfig;
  if (selectedConfig) {
    if (llvm::Error error =
            verifyFirstSliceConfigCapabilities(capabilities, *selectedConfig))
      return std::move(error);
  } else {
    llvm::Expected<const RVVI32VectorShapeConfig *> config =
        selectAvailableFirstSliceConfigCapabilities(capabilities);
    if (!config)
      return config.takeError();
    selectedConfig = *config;
  }

  RVVCapabilityPropertyView view;
  view.architecture = std::move(*architecture);
  view.isaVectorHints = std::move(*isaVectorHints);
  view.selectedMarch = std::move(*selectedMarch);
  view.hartCount = *hartCount;
  view.vlenbBytes = vlenbBytes;
  view.i32M1LaneCount = i32M1LaneCount;
  view.i32Config = selectedConfig;
  return view;
}

llvm::Expected<bool> variantRequiresCapabilityID(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities, llvm::StringRef id) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeRVVPluginError("materialized RVV variant requires structured "
                              "'requires' metadata");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeRVVPluginError(
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

llvm::Error verifyVariantRequiresCapabilityID(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities, llvm::StringRef id) {
  llvm::Expected<bool> requiresID =
      variantRequiresCapabilityID(variant, capabilities, id);
  if (!requiresID)
    return requiresID.takeError();
  if (!*requiresID)
    return makeRVVPluginError(llvm::Twine("materialized RVV variant must "
                                          "require capability id '") +
                              id + "'");
  return llvm::Error::success();
}

llvm::Expected<bool> variantRequiresConfig(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    const RVVI32VectorShapeConfig &config) {
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
    const RVVI32VectorShapeConfig &config) {
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
    const RVVI32VectorShapeConfig &config) {
  if (llvm::Error error = verifyVariantRequiresCapabilityID(
          variant, capabilities, config.sewCapabilityID))
    return error;
  if (llvm::Error error = verifyVariantRequiresCapabilityID(
          variant, capabilities, config.lmulCapabilityID))
    return error;
  if (llvm::Error error = verifyVariantRequiresCapabilityID(
          variant, capabilities, config.tailPolicyCapabilityID))
    return error;
  if (llvm::Error error = verifyVariantRequiresCapabilityID(
          variant, capabilities, config.maskPolicyCapabilityID))
    return error;
  return llvm::Error::success();
}

llvm::Expected<const RVVI32VectorShapeConfig *>
getVariantRequiredFirstSliceConfig(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities) {
  llvm::Expected<bool> requiresM1 =
      variantRequiresConfig(variant, capabilities, getI32M1ConfigSpec());
  if (!requiresM1)
    return requiresM1.takeError();
  llvm::Expected<bool> requiresM2 =
      variantRequiresConfig(variant, capabilities, getI32M2ConfigSpec());
  if (!requiresM2)
    return requiresM2.takeError();
  llvm::Expected<bool> requiresI64M1 = variantRequiresConfig(
      variant, capabilities, tianchenrv::target::rvv::getI64M1VectorShapeConfig());
  if (!requiresI64M1)
    return requiresI64M1.takeError();

  unsigned completeConfigCount = (*requiresM1 ? 1 : 0) + (*requiresM2 ? 1 : 0) +
                                 (*requiresI64M1 ? 1 : 0);
  if (completeConfigCount > 1)
    return makeRVVPluginError(
        "materialized RVV variant must require exactly one finite RVV "
        "dtype/LMUL config shape");
  if (*requiresM1)
    return &getI32M1ConfigSpec();
  if (*requiresM2)
    return &getI32M2ConfigSpec();
  if (*requiresI64M1)
    return &tianchenrv::target::rvv::getI64M1VectorShapeConfig();

  llvm::Expected<bool> requiresAnyM1 =
      variantRequiresAnyConfigID(variant, capabilities, getI32M1ConfigSpec());
  if (!requiresAnyM1)
    return requiresAnyM1.takeError();
  llvm::Expected<bool> requiresAnyM2 =
      variantRequiresAnyConfigID(variant, capabilities, getI32M2ConfigSpec());
  if (!requiresAnyM2)
    return requiresAnyM2.takeError();
  llvm::Expected<bool> requiresAnyI64M1 = variantRequiresAnyConfigID(
      variant, capabilities, tianchenrv::target::rvv::getI64M1VectorShapeConfig());
  if (!requiresAnyI64M1)
    return requiresAnyI64M1.takeError();
  if (*requiresAnyM1 && !*requiresAnyM2)
    if (llvm::Error error = verifyVariantRequiresConfigIDs(
            variant, capabilities, getI32M1ConfigSpec()))
      return std::move(error);
  if (*requiresAnyM2 && !*requiresAnyM1)
    if (llvm::Error error = verifyVariantRequiresConfigIDs(
            variant, capabilities, getI32M2ConfigSpec()))
      return std::move(error);
  if (*requiresAnyI64M1 && !*requiresAnyM1 && !*requiresAnyM2)
    if (llvm::Error error = verifyVariantRequiresConfigIDs(
            variant, capabilities,
            tianchenrv::target::rvv::getI64M1VectorShapeConfig()))
      return std::move(error);

  return makeRVVPluginError(
      "materialized RVV variant must require either the finite i32m1, i32m2, "
      "or i64m1 config capability ids");
}

mlir::StringAttr getRVVPolicyAttrNameAttr(mlir::MLIRContext *context) {
  return mlir::StringAttr::get(context, kRVVPolicyAttrName);
}

tcrv::rvv::PolicyAttr getExpectedRVVPolicyAttr(mlir::MLIRContext *context) {
  return tcrv::rvv::PolicyAttr::get(context, tcrv::rvv::TailPolicy::Agnostic,
                                    tcrv::rvv::MaskPolicy::Agnostic);
}

llvm::Error verifyExpectedRVVPolicyAttr(tcrv::exec::VariantOp variant) {
  mlir::Attribute rawPolicy = variant->getAttr(kRVVPolicyAttrName);
  if (!rawPolicy)
    return makeRVVPluginError(
        "materialized RVV variant requires typed 'tcrv_rvv.policy' metadata");

  auto policy = llvm::dyn_cast<tcrv::rvv::PolicyAttr>(rawPolicy);
  if (!policy)
    return makeRVVPluginError(
        "materialized RVV variant 'tcrv_rvv.policy' metadata must be a typed "
        "#tcrv_rvv.policy attribute");

  if (policy.getTail() != tcrv::rvv::TailPolicy::Agnostic ||
      policy.getMask() != tcrv::rvv::MaskPolicy::Agnostic) {
    return makeRVVPluginError(
        "materialized RVV variant 'tcrv_rvv.policy' metadata must match the "
        "RVV first-slice agnostic tail/mask policy");
  }

  return llvm::Error::success();
}

llvm::Expected<std::string>
buildRVVCapabilitySummary(tcrv::exec::VariantOp variant,
                          const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeRVVPluginError(
        "selected RVV variant requires structured 'requires' metadata");

  llvm::SmallVector<std::string, 4> summaries;
  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeRVVPluginError(
          "selected RVV variant requires only capability symbol references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (capability)
      summaries.push_back(capability->getID().str());
    else
      summaries.push_back(symbolRef.getValue().str());
  }

  if (summaries.empty())
    summaries.push_back(kRVVCapabilityID.str());

  return llvm::join(summaries, ",");
}

llvm::Error verifyRequiredMarchAttr(tcrv::exec::VariantOp variant,
                                    const RVVCapabilityPropertyView &view) {
  mlir::Attribute rawRequiredMarch =
      variant->getAttr(kRVVRequiredMarchAttrName);
  if (!rawRequiredMarch)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " requires string 'tcrv_rvv.required_march' metadata");

  auto requiredMarch = llvm::dyn_cast<mlir::StringAttr>(rawRequiredMarch);
  if (!requiredMarch)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " 'tcrv_rvv.required_march' metadata must be a string attribute");

  llvm::StringRef requiredMarchValue = requiredMarch.getValue().trim();
  if (requiredMarchValue.empty())
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " 'tcrv_rvv.required_march' metadata must be non-empty");

  std::string context =
      (llvm::Twine("variant @") + variant.getSymName() +
       " attribute 'tcrv_rvv.required_march'")
          .str();
  if (llvm::Error error = validateRVVPropertyText(
          context, kRVVRequiredMarchAttrName, requiredMarchValue))
    return error;

  if (requiredMarchValue != view.selectedMarch)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " 'tcrv_rvv.required_march' metadata is not satisfied by preserved "
        "capability property 'selected_march'");

  return llvm::Error::success();
}

llvm::Error
verifyOptionalCapacityAttrs(tcrv::exec::VariantOp variant,
                            const RVVCapabilityPropertyView &view) {
  mlir::Attribute rawVLenB = variant->getAttr(kRVVVLenBBytesAttrName);
  mlir::Attribute rawI32Lanes = variant->getAttr(kRVVI32M1LanesAttrName);
  if (!rawVLenB && !rawI32Lanes)
    return llvm::Error::success();
  if (!rawVLenB || !rawI32Lanes)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " capacity metadata requires both 'tcrv_rvv.vlenb_bytes' and "
        "'tcrv_rvv.base_i32_m1_lanes'");

  auto vlenbAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawVLenB);
  auto lanesAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawI32Lanes);
  if (!vlenbAttr || !lanesAttr)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " capacity metadata must use integer attributes");

  std::int64_t vlenb = vlenbAttr.getInt();
  std::int64_t lanes = lanesAttr.getInt();
  if (vlenb <= 0 || lanes <= 0 || vlenb < 4 || vlenb % 4 != 0 ||
      vlenb / 4 != lanes)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " capacity metadata requires i32 lanes to equal vlenb bytes divided "
        "by four");

  if (!view.vlenbBytes || !view.i32M1LaneCount)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " claims RVV vector capacity metadata without matching structured "
        "target capabilities");

  if (static_cast<std::uint64_t>(vlenb) != *view.vlenbBytes ||
      static_cast<std::uint64_t>(lanes) != *view.i32M1LaneCount)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " capacity metadata is not satisfied by preserved RVV vlenb/i32 lane "
        "capability facts");

  return llvm::Error::success();
}

llvm::Error verifySmokeProbeDescriptorAttr(tcrv::exec::VariantOp variant) {
  mlir::Attribute rawDescriptor =
      variant->getAttr(kRVVSmokeProbeDescriptorAttrName);
  if (!rawDescriptor)
    return llvm::Error::success();

  auto descriptor = llvm::dyn_cast<mlir::StringAttr>(rawDescriptor);
  if (!descriptor || descriptor.getValue().trim().empty())
    return makeRVVPluginError(
        llvm::Twine("RVV smoke-probe descriptor on variant @") +
        variant.getSymName() + " requires string attribute '" +
        kRVVSmokeProbeDescriptorAttrName + "'");

  std::string descriptorContext =
      (llvm::Twine("variant @") + variant.getSymName() +
       " RVV smoke-probe descriptor")
          .str();
  if (llvm::Error error = validateRVVPropertyText(
          descriptorContext, kRVVSmokeProbeDescriptorAttrName,
          descriptor.getValue().trim()))
    return error;

  if (descriptor.getValue().trim() != kRVVSmokeProbeDescriptorValue)
    return makeRVVPluginError(
        llvm::Twine("RVV smoke-probe descriptor on variant @") +
        variant.getSymName() + " must be '" + kRVVSmokeProbeDescriptorValue +
        "'");

  if (variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName))
    return makeRVVPluginError(
        llvm::Twine("RVV smoke-probe descriptor on variant @") +
        variant.getSymName() +
        " must not be combined with the finite RVV i32 microkernel lowering "
        "descriptor");

  if (!variant->hasAttr(kRVVRequiredMarchAttrName))
    return makeRVVPluginError(
        llvm::Twine("RVV smoke-probe descriptor on variant @") +
        variant.getSymName() +
        " requires string 'tcrv_rvv.required_march' metadata");

  return llvm::Error::success();
}

llvm::Expected<std::optional<RVVCapacityMetadata>>
parseCapacityMetadataAttrs(mlir::Operation *op, llvm::StringRef context,
                           llvm::StringRef vlenbAttrName,
                           llvm::StringRef lanesAttrName) {
  mlir::Attribute rawVLenB = op->getAttr(vlenbAttrName);
  mlir::Attribute rawI32Lanes = op->getAttr(lanesAttrName);
  if (!rawVLenB && !rawI32Lanes)
    return std::optional<RVVCapacityMetadata>();
  if (!rawVLenB || !rawI32Lanes)
    return makeRVVPluginError(
        llvm::Twine(context) + " capacity metadata requires both '" +
        vlenbAttrName + "' and '" + lanesAttrName + "'");

  auto vlenbAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawVLenB);
  auto lanesAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawI32Lanes);
  if (!vlenbAttr || !lanesAttr)
    return makeRVVPluginError(
        llvm::Twine(context) +
        " capacity metadata must use integer attributes");

  RVVCapacityMetadata metadata;
  metadata.vlenbBytes = vlenbAttr.getInt();
  metadata.i32M1Lanes = lanesAttr.getInt();
  if (metadata.vlenbBytes <= 0 || metadata.i32M1Lanes <= 0 ||
      metadata.vlenbBytes < 4 || metadata.vlenbBytes % 4 != 0 ||
      metadata.vlenbBytes / 4 != metadata.i32M1Lanes)
    return makeRVVPluginError(
        llvm::Twine(context) +
        " capacity metadata requires i32 lanes to equal vlenb bytes divided "
        "by four");

  return std::optional<RVVCapacityMetadata>(metadata);
}

llvm::Error copyCapacityMetadataToBoundary(mlir::OperationState &state,
                                           tcrv::exec::VariantOp variant,
                                           mlir::OpBuilder &builder) {
  llvm::Expected<std::optional<RVVCapacityMetadata>> metadata =
      parseCapacityMetadataAttrs(
          variant.getOperation(),
          (llvm::Twine("selected RVV variant @") + variant.getSymName()).str(),
          kRVVVLenBBytesAttrName, kRVVI32M1LanesAttrName);
  if (!metadata)
    return metadata.takeError();
  if (!*metadata)
    return llvm::Error::success();

  state.addAttribute(kRVVBoundaryVLenBBytesAttrName,
                     builder.getI64IntegerAttr((*metadata)->vlenbBytes));
  state.addAttribute(kRVVBoundaryI32M1LanesAttrName,
                     builder.getI64IntegerAttr((*metadata)->i32M1Lanes));
  return llvm::Error::success();
}

llvm::Error validateBoundaryCapacityMetadata(tcrv::exec::VariantOp variant,
                                             mlir::Operation *boundary) {
  llvm::Expected<std::optional<RVVCapacityMetadata>> variantMetadata =
      parseCapacityMetadataAttrs(
          variant.getOperation(),
          (llvm::Twine("selected RVV variant @") + variant.getSymName()).str(),
          kRVVVLenBBytesAttrName, kRVVI32M1LanesAttrName);
  if (!variantMetadata)
    return variantMetadata.takeError();

  llvm::Expected<std::optional<RVVCapacityMetadata>> boundaryMetadata =
      parseCapacityMetadataAttrs(
          boundary,
          (llvm::Twine("selected RVV lowering boundary for @") +
           variant.getSymName())
              .str(),
          kRVVBoundaryVLenBBytesAttrName, kRVVBoundaryI32M1LanesAttrName);
  if (!boundaryMetadata)
    return boundaryMetadata.takeError();

  if (*variantMetadata && !*boundaryMetadata)
    return makeRVVPluginError(
        llvm::Twine("selected RVV lowering boundary for @") +
        variant.getSymName() +
        " must preserve selected capacity metadata from the variant");
  if (!*variantMetadata && *boundaryMetadata)
    return makeRVVPluginError(
        llvm::Twine("selected RVV lowering boundary for @") +
        variant.getSymName() +
        " claims capacity metadata absent from the selected variant");
  if (!*variantMetadata && !*boundaryMetadata)
    return llvm::Error::success();

  if ((*variantMetadata)->vlenbBytes != (*boundaryMetadata)->vlenbBytes ||
      (*variantMetadata)->i32M1Lanes != (*boundaryMetadata)->i32M1Lanes)
    return makeRVVPluginError(
        llvm::Twine("selected RVV lowering boundary for @") +
        variant.getSymName() +
        " capacity metadata does not match the selected variant");

  return llvm::Error::success();
}

bool hasAnyCapacityMetadata(mlir::Operation *op, llvm::StringRef vlenbAttrName,
                            llvm::StringRef lanesAttrName) {
  return op && (op->hasAttr(vlenbAttrName) || op->hasAttr(lanesAttrName));
}

bool hasAnySelectedVectorShapeMetadata(
    mlir::Operation *op, llvm::StringRef shapeAttrName,
    llvm::StringRef sewAttrName, llvm::StringRef lmulAttrName,
    llvm::StringRef tailPolicyAttrName, llvm::StringRef maskPolicyAttrName,
    llvm::StringRef vectorTypeAttrName, llvm::StringRef vectorSuffixAttrName,
    llvm::StringRef setvlSuffixAttrName) {
  return op && (op->hasAttr(shapeAttrName) || op->hasAttr(sewAttrName) ||
                op->hasAttr(lmulAttrName) ||
                op->hasAttr(tailPolicyAttrName) ||
                op->hasAttr(maskPolicyAttrName) ||
                op->hasAttr(vectorTypeAttrName) ||
                op->hasAttr(vectorSuffixAttrName) ||
                op->hasAttr(setvlSuffixAttrName));
}

llvm::Error addSelectedCapacityMetadataToPlan(VariantEmissionPlan &plan,
                                              tcrv::exec::VariantOp variant) {
  llvm::Expected<std::optional<RVVCapacityMetadata>> metadata =
      parseCapacityMetadataAttrs(
          variant.getOperation(),
          (llvm::Twine("selected RVV variant @") + variant.getSymName()).str(),
          kRVVVLenBBytesAttrName, kRVVI32M1LanesAttrName);
  if (!metadata)
    return metadata.takeError();
  if (!*metadata)
    return llvm::Error::success();

  plan.addSelectedPlanMetadata(kRVVVLenBBytesAttrName,
                               std::to_string((*metadata)->vlenbBytes),
                               kSelectedRVVCapacityMetadataRole,
                               kSelectedRVVCapacityMetadataNote);
  plan.addSelectedPlanMetadata(kRVVI32M1LanesAttrName,
                               std::to_string((*metadata)->i32M1Lanes),
                               kSelectedRVVCapacityMetadataRole,
                               kSelectedRVVCapacityMetadataNote);
  return llvm::Error::success();
}

llvm::Expected<std::optional<std::string>> getOptionalSelectedMABI(
    const support::TargetCapabilitySet &capabilities) {
  std::optional<std::string> selectedMABI;

  auto mergeMABI = [&](const support::CapabilityDescriptor &capability,
                       llvm::StringRef propertyName) -> llvm::Error {
    llvm::StringRef value = capability.getProperty(propertyName).trim();
    if (value.empty())
      return llvm::Error::success();

    std::string context =
        (llvm::Twine("capability id '") + capability.getID() + "'").str();
    if (llvm::Error error =
            validateRVVPropertyText(context, propertyName, value))
      return error;

    if (selectedMABI && *selectedMABI != value)
      return makeRVVPluginError(
          "conflicting RVV selected_mabi capability metadata");

    selectedMABI = value.str();
    return llvm::Error::success();
  };

  if (const support::CapabilityDescriptor *compileRunCapability =
          capabilities.lookupByID(rvv::getRVVProbeCompileRunCapabilityID())) {
    if (compileRunCapability->isAvailable()) {
      if (llvm::Error error =
              mergeMABI(*compileRunCapability, kSelectedMABIPropertyName))
        return std::move(error);
    }
  }

  if (const support::CapabilityDescriptor *selectedMABICapability =
          capabilities.lookupByID(rvv::getRVVSelectedMABICapabilityID())) {
    if (selectedMABICapability->isAvailable()) {
      if (llvm::Error error = mergeMABI(*selectedMABICapability,
                                        kSelectedMarchValuePropertyName))
        return std::move(error);
    }
  }

  return selectedMABI;
}

llvm::Expected<std::optional<RVVI32MicrokernelMaterializationPlan>>
buildI32MicrokernelMaterializationPlan(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    const RVVCapabilityPropertyView &view) {
  mlir::Attribute rawDescriptor =
      variant->getAttr(kRVVI32VAddLoweringDescriptorAttrName);
  if (!rawDescriptor)
    return std::optional<RVVI32MicrokernelMaterializationPlan>();

  auto descriptor = llvm::dyn_cast<mlir::StringAttr>(rawDescriptor);
  if (!descriptor || descriptor.getValue().trim().empty())
    return makeRVVPluginError(
        llvm::Twine("finite RVV i32 microkernel lowering descriptor on "
                    "variant @") +
        variant.getSymName() + " requires string attribute '" +
        kRVVI32VAddLoweringDescriptorAttrName + "'");

  const RVVI32MicrokernelFamilySpec *family =
      lookupI32MicrokernelFamilyByDescriptor(descriptor.getValue());
  if (!family) {
    if (tianchenrv::target::rvv::lookupRVVBinaryFamilyByLoweringDescriptor(
            descriptor.getValue()))
      return std::optional<RVVI32MicrokernelMaterializationPlan>();
    return makeRVVPluginError(
        llvm::Twine("finite RVV i32 microkernel lowering descriptor on "
                    "variant @") +
        variant.getSymName() + " must be '" +
        getI32VAddFamilySpec().getLoweringDescriptor() + "' or '" +
        getI32VSubFamilySpec().getLoweringDescriptor() + "' or '" +
        getI32VMulFamilySpec().getLoweringDescriptor() + "'");
  }

  std::string descriptorContext =
      (llvm::Twine("variant @") + variant.getSymName() +
       " " + family->descriptorNoun)
          .str();
  if (llvm::Error error = validateRVVPropertyText(
          descriptorContext, kRVVI32VAddLoweringDescriptorAttrName,
          descriptor.getValue().trim()))
    return std::move(error);

  auto elementCountAttr =
      variant->getAttrOfType<mlir::IntegerAttr>(
          kRVVI32VAddElementCountAttrName);
  if (!elementCountAttr)
    return makeRVVPluginError(
        llvm::Twine(family->descriptorNoun) + " on variant @" +
        variant.getSymName() + " requires integer attribute '" +
        kRVVI32VAddElementCountAttrName + "'");

  std::int64_t elementCount = elementCountAttr.getInt();
  if (elementCount <= 0 || elementCount > 64)
    return makeRVVPluginError(
        llvm::Twine(family->descriptorNoun) + " on variant @" +
        variant.getSymName() +
        " requires tcrv_rvv.element_count in the bounded smoke range [1, 64]");

  auto requiredMarch =
      variant->getAttrOfType<mlir::StringAttr>(kRVVRequiredMarchAttrName);
  if (!requiredMarch || requiredMarch.getValue().trim().empty())
    return makeRVVPluginError(
        llvm::Twine(family->descriptorNoun) + " on variant @" +
        variant.getSymName() + " requires string 'tcrv_rvv.required_march' "
                               "metadata");

  if (llvm::Error error = verifyRequiredMarchAttr(variant, view))
    return std::move(error);

  llvm::Expected<std::optional<std::string>> selectedMABI =
      getOptionalSelectedMABI(capabilities);
  if (!selectedMABI)
    return selectedMABI.takeError();

  RVVI32MicrokernelMaterializationPlan plan;
  plan.family = family;
  plan.i32Config = view.i32Config;
  plan.elementCount = elementCount;
  plan.requiredMarch = requiredMarch.getValue().trim().str();
  plan.selectedMABI = std::move(*selectedMABI);
  return plan;
}

llvm::Expected<std::optional<RVVI64MicrokernelMaterializationPlan>>
buildI64MicrokernelMaterializationPlan(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    const RVVCapabilityPropertyView &view) {
  mlir::Attribute rawDescriptor =
      variant->getAttr(kRVVI32VAddLoweringDescriptorAttrName);
  if (!rawDescriptor)
    return std::optional<RVVI64MicrokernelMaterializationPlan>();

  auto descriptor = llvm::dyn_cast<mlir::StringAttr>(rawDescriptor);
  if (!descriptor || descriptor.getValue().trim().empty())
    return makeRVVPluginError(
        llvm::Twine("finite RVV i64 microkernel lowering descriptor on "
                    "variant @") +
        variant.getSymName() + " requires string attribute '" +
        kRVVI32VAddLoweringDescriptorAttrName + "'");

  const RVVBinaryFamilyDescriptor *family =
      tianchenrv::target::rvv::lookupRVVBinaryFamilyByLoweringDescriptor(
          descriptor.getValue());
  if (!family)
    return std::optional<RVVI64MicrokernelMaterializationPlan>();

  std::string descriptorContext =
      (llvm::Twine("variant @") + variant.getSymName() + " " +
       family->descriptorNoun)
          .str();
  if (llvm::Error error = validateRVVPropertyText(
          descriptorContext, kRVVI32VAddLoweringDescriptorAttrName,
          descriptor.getValue().trim()))
    return std::move(error);

  if (view.i32Config !=
      &tianchenrv::target::rvv::getI64M1VectorShapeConfig())
    return makeRVVPluginError(
        llvm::Twine(family->descriptorNoun) + " on variant @" +
        variant.getSymName() + " requires finite i64m1 vector-shape config "
                               "capability ids");

  auto elementCountAttr =
      variant->getAttrOfType<mlir::IntegerAttr>(
          kRVVI32VAddElementCountAttrName);
  if (!elementCountAttr)
    return makeRVVPluginError(
        llvm::Twine(family->descriptorNoun) + " on variant @" +
        variant.getSymName() + " requires integer attribute '" +
        kRVVI32VAddElementCountAttrName + "'");

  std::int64_t elementCount = elementCountAttr.getInt();
  if (elementCount <= 0 || elementCount > 64)
    return makeRVVPluginError(
        llvm::Twine(family->descriptorNoun) + " on variant @" +
        variant.getSymName() +
        " requires tcrv_rvv.element_count in the bounded smoke range [1, 64]");

  auto requiredMarch =
      variant->getAttrOfType<mlir::StringAttr>(kRVVRequiredMarchAttrName);
  if (!requiredMarch || requiredMarch.getValue().trim().empty())
    return makeRVVPluginError(
        llvm::Twine(family->descriptorNoun) + " on variant @" +
        variant.getSymName() + " requires string 'tcrv_rvv.required_march' "
                               "metadata");

  if (llvm::Error error = verifyRequiredMarchAttr(variant, view))
    return std::move(error);

  llvm::Expected<std::optional<std::string>> selectedMABI =
      getOptionalSelectedMABI(capabilities);
  if (!selectedMABI)
    return selectedMABI.takeError();

  RVVI64MicrokernelMaterializationPlan plan;
  plan.descriptor =
      tianchenrv::target::rvv::getRVVBinaryIntrinsicDescriptor(
          *family, *view.i32Config);
  plan.elementCount = elementCount;
  plan.requiredMarch = requiredMarch.getValue().trim().str();
  plan.selectedMABI = std::move(*selectedMABI);
  return plan;
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

llvm::StringRef getStringAttrValue(mlir::Operation *op,
                                   llvm::StringRef name) {
  mlir::StringAttr attr = getStringAttr(op, name);
  if (!attr)
    return {};
  return attr.getValue();
}

llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 4>>
buildRVVBinaryCallableRuntimeABIParameters(
    tcrv::exec::KernelOp kernel,
    const RVVBinaryIntrinsicDescriptor &descriptor) {
  llvm::SmallVector<tcrv::exec::MemWindowOp, 3> windows;
  if (llvm::Error error = support::collectRuntimeABIBufferMemWindows(
          kernel, descriptor.getBufferMemWindowSpecs(), windows))
    return std::move(error);

  llvm::SmallVector<support::RuntimeABIParamSpec, 1> countSpecs =
      descriptor.getRuntimeElementCountParamSpecs(/*cName=*/"");
  llvm::SmallVector<tcrv::exec::RuntimeParamOp, 1> runtimeParams;
  if (llvm::Error error =
          support::collectRuntimeABIParams(kernel, countSpecs, runtimeParams))
    return std::move(error);

  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  auto appendWindowParameter =
      [&](tcrv::exec::MemWindowOp window,
          support::RuntimeABIParameterRole role,
          llvm::StringRef cName) -> llvm::Error {
    llvm::StringRef cType =
        getStringAttrValue(window.getOperation(), support::kMemWindowCTypeAttrName);
    llvm::StringRef ownership = getStringAttrValue(
        window.getOperation(), support::kMemWindowOwnershipAttrName);
    std::optional<support::RuntimeABIParameterOwnership> parsedOwnership =
        support::symbolizeRuntimeABIParameterOwnership(ownership);
    if (!parsedOwnership)
      return makeRVVPluginError(
          llvm::Twine("i64 RVV callable ABI mem_window @") +
          window.getSymName() + " has unsupported ownership '" + ownership +
          "'");
    parameters.push_back(
        support::RuntimeABIParameter(cName, cType, role, *parsedOwnership));
    return llvm::Error::success();
  };

  if (llvm::Error error =
          appendWindowParameter(windows[0],
                                support::RuntimeABIParameterRole::LHSInputBuffer,
                                "lhs"))
    return std::move(error);
  if (llvm::Error error =
          appendWindowParameter(windows[1],
                                support::RuntimeABIParameterRole::RHSInputBuffer,
                                "rhs"))
    return std::move(error);
  if (llvm::Error error =
          appendWindowParameter(windows[2],
                                support::RuntimeABIParameterRole::OutputBuffer,
                                "out"))
    return std::move(error);

  tcrv::exec::RuntimeParamOp runtimeCount = runtimeParams.front();
  std::optional<support::RuntimeABIParameterOwnership> parsedOwnership =
      support::symbolizeRuntimeABIParameterOwnership(getStringAttrValue(
          runtimeCount.getOperation(), support::kRuntimeParamOwnershipAttrName));
  if (!parsedOwnership)
    return makeRVVPluginError(
        llvm::Twine("i64 RVV callable ABI runtime_param @") +
        runtimeCount.getSymName() + " has unsupported ownership");
  parameters.push_back(support::RuntimeABIParameter(
      getStringAttrValue(runtimeCount.getOperation(),
                         support::kRuntimeParamCNameAttrName),
      getStringAttrValue(runtimeCount.getOperation(),
                         support::kRuntimeParamCTypeAttrName),
      support::RuntimeABIParameterRole::RuntimeElementCount, *parsedOwnership));

  return parameters;
}

const RVVI32MicrokernelFamilySpec *
getI32MicrokernelFamilyForOp(mlir::Operation *op) {
  if (llvm::isa_and_nonnull<tcrv::rvv::I32VAddMicrokernelOp>(op))
    return &getI32VAddFamilySpec();
  if (llvm::isa_and_nonnull<tcrv::rvv::I32VSubMicrokernelOp>(op))
    return &getI32VSubFamilySpec();
  if (llvm::isa_and_nonnull<tcrv::rvv::I32VMulMicrokernelOp>(op))
    return &getI32VMulFamilySpec();
  return nullptr;
}

llvm::Error rejectExistingRVVBoundaryForVariant(tcrv::exec::KernelOp kernel,
                                                tcrv::exec::VariantOp variant) {
  if (!kernel || kernel.getBody().empty())
    return llvm::Error::success();

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<tcrv::rvv::LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto target =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    llvm::StringRef targetSymbol =
        target ? target.getValue() : llvm::StringRef("<missing>");
    if (targetSymbol != variant.getSymName())
      continue;

    return makeRVVPluginError(
        llvm::Twine("requires no pre-existing tcrv_rvv.lowering_boundary for "
                    "target @") +
        targetSymbol);
  }

  return llvm::Error::success();
}

llvm::Error rejectExistingRVVMicrokernelForSelectedPath(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role) {
  if (!kernel || kernel.getBody().empty())
    return llvm::Error::success();

  llvm::StringRef expectedRole = stringifyVariantEmissionRole(role);
  for (mlir::Operation &op : kernel.getBody().front()) {
    const RVVI32MicrokernelFamilySpec *family =
        getI32MicrokernelFamilyForOp(&op);
    bool isI64VAdd = llvm::isa<tcrv::rvv::I64VAddMicrokernelOp>(op);
    if (!family && !isI64VAdd)
      continue;

    auto target =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto microkernelRole = getStringAttr(&op, kRoleAttrName);
    llvm::StringRef targetSymbol =
        target ? target.getValue() : llvm::StringRef("<missing>");
    llvm::StringRef roleValue = microkernelRole
                                    ? microkernelRole.getValue()
                                    : llvm::StringRef("<missing>");
    if (targetSymbol != variant.getSymName() || roleValue != expectedRole)
      continue;

    return makeRVVPluginError(
        llvm::Twine("requires no pre-existing ") +
        (family ? family->getRVV().microkernelOpName
                : llvm::StringRef("tcrv_rvv.i64_vadd_microkernel")) +
        " for target @" +
        targetSymbol + " as " + expectedRole);
  }

  return llvm::Error::success();
}

llvm::Error validateMicrokernelEmissionAttr(mlir::Operation *op,
                                            llvm::StringRef attrName,
                                            llvm::StringRef expectedValue) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeRVVPluginError(llvm::Twine("explicit RVV microkernel emission "
                                          "plan requires non-empty string "
                                          "attribute '") +
                              attrName + "'");
  if (llvm::Error error =
          validateRVVPropertyText("explicit RVV microkernel emission plan",
                                  attrName, attr.getValue().trim()))
    return error;
  if (attr.getValue().trim() != expectedValue)
    return makeRVVPluginError(llvm::Twine("explicit RVV microkernel emission "
                                          "plan attribute '") +
                              attrName + "' value '" + attr.getValue().trim() +
                              "' does not match expected selected-path value '" +
                              expectedValue + "'");
  return llvm::Error::success();
}

llvm::Error validateMicrokernelDataflowRoleAttr(
    mlir::Operation *op, llvm::StringRef attrName,
    support::RuntimeABIParameterRole expectedRole) {
  return validateMicrokernelEmissionAttr(
      op, attrName, support::stringifyRuntimeABIParameterRole(expectedRole));
}

llvm::Error validateMicrokernelStructuredControlPlane(
    tcrv::exec::VariantOp variant, mlir::Operation *microkernel,
    const RVVI32MicrokernelFamilySpec &family,
    const RVVI32VectorShapeConfig &config) {
  if (llvm::Error error = verifyExpectedRVVPolicyAttr(variant))
    return error;

  auto expectedPolicy =
      llvm::cast<tcrv::rvv::PolicyAttr>(variant->getAttr(kRVVPolicyAttrName));

  mlir::Region &body = microkernel->getRegion(0);
  if (body.empty() || !llvm::hasSingleElement(body))
    return makeRVVPluginError(
        "explicit RVV microkernel emission plan requires exactly one "
        "structured microkernel control-plane body block");

  mlir::Block &block = body.front();
  if (block.getNumArguments() != 1 ||
      !block.getArgument(0).getType().isIndex())
    return makeRVVPluginError(
        "explicit RVV microkernel emission plan requires the structured "
        "control-plane body to expose one runtime index block argument for "
        "target/export-owned n/AVL");

  tcrv::rvv::SetVLOp setvl;
  tcrv::rvv::WithVLOp withVL;
  unsigned setvlCount = 0;
  unsigned withVLCount = 0;
  for (mlir::Operation &bodyOp : block) {
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
    return makeRVVPluginError(
        llvm::Twine("explicit RVV microkernel emission plan does not consume "
                    "unexpected control-plane body operation '") +
        bodyOp.getName().getStringRef() + "'");
  }

  if (setvlCount != 1 || withVLCount != 1)
    return makeRVVPluginError(
        "explicit RVV microkernel emission plan requires exactly one "
        "tcrv_rvv.setvl and exactly one tcrv_rvv.with_vl in the "
        "structured control-plane body");
  if (setvl.getAvl() != block.getArgument(0))
    return makeRVVPluginError(
        "explicit RVV microkernel emission plan requires tcrv_rvv.setvl AVL "
        "to use the runtime index body argument, not descriptor-local "
        "element_count or a constant");
  if (withVL.getVl() != setvl.getVl())
    return makeRVVPluginError(
        "explicit RVV microkernel emission plan requires tcrv_rvv.with_vl to "
        "consume the !tcrv_rvv.vl token produced by tcrv_rvv.setvl");
  if (setvl.getSew() != static_cast<std::uint64_t>(config.sewBits) ||
      setvl.getLmul() != config.lmul ||
      setvl.getPolicy() != expectedPolicy)
    return makeRVVPluginError(
        "explicit RVV microkernel emission plan requires setvl "
        "SEW/LMUL/policy metadata to match the selected RVV first-slice "
        "variant config");

  auto withVLSew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto withVLLMUL = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto withVLPolicy =
      withVL->getAttrOfType<tcrv::rvv::PolicyAttr>(kPolicyAttrName);
  if (!withVLSew || withVLSew.getInt() != config.sewBits || !withVLLMUL ||
      withVLLMUL.getValue() != config.lmul || !withVLPolicy ||
      withVLPolicy != expectedPolicy)
    return makeRVVPluginError(
        "explicit RVV microkernel emission plan requires with_vl "
        "SEW/LMUL/policy metadata to match setvl and the selected RVV "
        "first-slice variant config");

  mlir::Region &withVLBody = withVL.getBody();
  if (withVLBody.empty() || !llvm::hasSingleElement(withVLBody))
    return makeRVVPluginError(
        "explicit RVV microkernel emission plan requires the bounded "
        "tcrv_rvv.with_vl body to be present");
  if (withVLBody.front().getNumArguments() != 0)
    return makeRVVPluginError(
        "explicit RVV microkernel emission plan requires the bounded "
        "tcrv_rvv.with_vl body to have no block arguments");

  llvm::SmallVector<mlir::Operation *, 4> dataflowOps;
  for (mlir::Operation &withVLOp : withVLBody.front())
    dataflowOps.push_back(&withVLOp);
  if (dataflowOps.size() != 4)
    return makeRVVPluginError(
        llvm::Twine("explicit RVV microkernel emission plan requires the "
                    "finite tcrv_rvv.i32_load, tcrv_rvv.i32_load, ") +
        family.getRVV().arithmeticOpName +
        ", tcrv_rvv.i32_store sequence in the tcrv_rvv.with_vl body");

  auto lhsLoad = llvm::dyn_cast<tcrv::rvv::I32LoadOp>(dataflowOps[0]);
  auto rhsLoad = llvm::dyn_cast<tcrv::rvv::I32LoadOp>(dataflowOps[1]);
  auto store = llvm::dyn_cast<tcrv::rvv::I32StoreOp>(dataflowOps[3]);
  mlir::Value arithmeticLHS;
  mlir::Value arithmeticRHS;
  mlir::Value arithmeticVL;
  mlir::Value arithmeticResult;
  if (auto add = llvm::dyn_cast<tcrv::rvv::I32AddOp>(dataflowOps[2])) {
    if (family.getKind() != RVVI32MicrokernelKind::Add)
      return makeRVVPluginError(
          "explicit RVV microkernel emission plan arithmetic op does not "
          "match the selected microkernel family");
    arithmeticLHS = add.getLhs();
    arithmeticRHS = add.getRhs();
    arithmeticVL = add.getVl();
    arithmeticResult = add.getSum();
  } else if (auto sub =
                 llvm::dyn_cast<tcrv::rvv::I32SubOp>(dataflowOps[2])) {
    if (family.getKind() != RVVI32MicrokernelKind::Sub)
      return makeRVVPluginError(
          "explicit RVV microkernel emission plan arithmetic op does not "
          "match the selected microkernel family");
    arithmeticLHS = sub.getLhs();
    arithmeticRHS = sub.getRhs();
    arithmeticVL = sub.getVl();
    arithmeticResult = sub.getDifference();
  } else if (auto mul =
                 llvm::dyn_cast<tcrv::rvv::I32MulOp>(dataflowOps[2])) {
    if (family.getKind() != RVVI32MicrokernelKind::Mul)
      return makeRVVPluginError(
          "explicit RVV microkernel emission plan arithmetic op does not "
          "match the selected microkernel family");
    arithmeticLHS = mul.getLhs();
    arithmeticRHS = mul.getRhs();
    arithmeticVL = mul.getVl();
    arithmeticResult = mul.getProduct();
  }
  if (!lhsLoad || !rhsLoad || !arithmeticResult || !store)
    return makeRVVPluginError(
        llvm::Twine("explicit RVV microkernel emission plan requires the "
                    "finite tcrv_rvv.i32_load, tcrv_rvv.i32_load, ") +
        family.getRVV().arithmeticOpName +
        ", tcrv_rvv.i32_store sequence in the tcrv_rvv.with_vl body");

  if (lhsLoad.getVl() != withVL.getVl() ||
      rhsLoad.getVl() != withVL.getVl() || arithmeticVL != withVL.getVl() ||
      store.getVl() != withVL.getVl())
    return makeRVVPluginError(
        "explicit RVV microkernel emission plan requires every finite RVV i32 "
        "dataflow op to consume the !tcrv_rvv.vl token owned by with_vl");
  if (arithmeticLHS != lhsLoad.getLoaded() ||
      arithmeticRHS != rhsLoad.getLoaded() ||
      store.getValue() != arithmeticResult)
    return makeRVVPluginError(
        llvm::Twine("explicit RVV microkernel emission plan requires finite "
                    "RVV i32 dataflow SSA chain lhs-load,rhs-load -> ") +
        family.getRVV().arithmeticVerb + " -> store");

  if (llvm::Error error = validateMicrokernelDataflowRoleAttr(
          lhsLoad.getOperation(), kBufferRoleAttrName,
          support::RuntimeABIParameterRole::LHSInputBuffer))
    return error;
  if (llvm::Error error = validateMicrokernelDataflowRoleAttr(
          rhsLoad.getOperation(), kBufferRoleAttrName,
          support::RuntimeABIParameterRole::RHSInputBuffer))
    return error;
  if (llvm::Error error = validateMicrokernelDataflowRoleAttr(
          store.getOperation(), kBufferRoleAttrName,
          support::RuntimeABIParameterRole::OutputBuffer))
    return error;

  return llvm::Error::success();
}

llvm::Expected<std::optional<RVVI32MicrokernelMaterializationPlan>>
buildDescriptorPlanForEmission(const VariantEmissionRequest &request) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant || !variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName))
    return std::optional<RVVI32MicrokernelMaterializationPlan>();

  llvm::Expected<const RVVI32VectorShapeConfig *> requiredConfig =
      getVariantRequiredFirstSliceConfig(variant, request.getCapabilities());
  if (!requiredConfig)
    return requiredConfig.takeError();

  llvm::Expected<RVVCapabilityPropertyView> propertyView =
      buildRVVCapabilityPropertyView(request.getCapabilities(),
                                     *requiredConfig);
  if (!propertyView)
    return propertyView.takeError();

  return buildI32MicrokernelMaterializationPlan(variant, request.getCapabilities(),
                                         *propertyView);
}

llvm::Expected<const RVVI32MicrokernelFamilySpec *>
findMatchingExplicitMicrokernelFamily(
    const VariantEmissionRequest &request) {
  tcrv::exec::KernelOp kernel = request.getKernel();
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!kernel || !variant || kernel.getBody().empty())
    return static_cast<const RVVI32MicrokernelFamilySpec *>(nullptr);

  llvm::Expected<std::optional<RVVI32MicrokernelMaterializationPlan>>
      descriptorPlan = buildDescriptorPlanForEmission(request);
  if (!descriptorPlan)
    return descriptorPlan.takeError();

  llvm::StringRef expectedRole = stringifyVariantEmissionRole(request.getRole());
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  auto variantRequiredMarch =
      variant->getAttrOfType<mlir::StringAttr>(kRVVRequiredMarchAttrName);
  unsigned matches = 0;
  const RVVI32MicrokernelFamilySpec *matchedFamily = nullptr;
  for (mlir::Operation &op : kernel.getBody().front()) {
    const RVVI32MicrokernelFamilySpec *family =
        getI32MicrokernelFamilyForOp(&op);
    if (!family)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto role = getStringAttr(&op, kRoleAttrName);
    if (!selectedVariant || !role)
      return makeRVVPluginError("explicit RVV microkernel emission plan "
                                "requires selected_variant and role metadata");

    if (selectedVariant.getValue() != variant.getSymName() ||
        role.getValue() != expectedRole) {
      return makeRVVPluginError(
          llvm::Twine("stale ") + family->getRVV().microkernelOpName +
          " for @" +
          selectedVariant.getValue() + " as " + role.getValue() +
          " is not the selected RVV emission plan path @" +
          variant.getSymName() + " as " + expectedRole);
    }

    ++matches;
    matchedFamily = family;
    if (*descriptorPlan && (*descriptorPlan)->family != family)
      return makeRVVPluginError(
          llvm::Twine("explicit RVV microkernel emission plan uses ") +
          family->getRVV().microkernelOpName +
          " but selected variant descriptor requires " +
          (*descriptorPlan)->family->getRVV().microkernelOpName);

    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kSourceKernelAttrName,
                                            kernel.getSymName()))
      return std::move(error);
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kOriginAttrName,
                                            kRVVPluginName))
      return std::move(error);
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kRoleAttrName, expectedRole))
      return std::move(error);

    if (!variantRequires ||
        op.getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName) !=
            variantRequires) {
      return makeRVVPluginError(
          llvm::Twine("explicit RVV microkernel emission plan requires ") +
          family->getRVV().microkernelOpName +
          " required_capabilities to match selected variant requires metadata");
    }

    if (!variantRequiredMarch || variantRequiredMarch.getValue().trim().empty())
      return makeRVVPluginError(
          "explicit RVV microkernel emission plan requires selected variant "
          "metadata 'tcrv_rvv.required_march'");
    if (llvm::Error error = validateMicrokernelEmissionAttr(
            &op, kMicrokernelRequiredMarchAttrName,
            variantRequiredMarch.getValue().trim()))
      return std::move(error);

    if (auto selectedMABI = getStringAttr(&op, kSelectedMABIAttrName))
      if (llvm::Error error =
              validateRVVPropertyText("explicit RVV microkernel emission plan",
                                      kSelectedMABIAttrName,
                                      selectedMABI.getValue().trim()))
        return std::move(error);

    auto elementCount =
        op.getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
    if (!elementCount || elementCount.getInt() <= 0 ||
        elementCount.getInt() > 64)
      return makeRVVPluginError(
          "explicit RVV microkernel emission plan requires element_count in "
          "the bounded smoke range [1, 64]");
    if (*descriptorPlan &&
        elementCount.getInt() != (*descriptorPlan)->elementCount)
      return makeRVVPluginError(
          llvm::Twine("explicit RVV microkernel emission plan requires ") +
          family->getRVV().microkernelOpName +
          " element_count to match selected variant finite descriptor "
          "metadata 'tcrv_rvv.element_count'");

    llvm::Expected<const RVVI32VectorShapeConfig *> requiredConfig =
        getVariantRequiredFirstSliceConfig(variant, request.getCapabilities());
    if (!requiredConfig)
      return requiredConfig.takeError();

    if (llvm::Error error = validateMicrokernelStructuredControlPlane(
            variant, &op, *family, **requiredConfig))
      return std::move(error);
  }

  if (matches > 1)
    return makeRVVPluginError(
        llvm::Twine("selected RVV emission plan path @") +
        variant.getSymName() + " as " + expectedRole +
        " has duplicate RVV i32 microkernel metadata");

  if (matches == 0)
    return static_cast<const RVVI32MicrokernelFamilySpec *>(nullptr);
  return matchedFamily;
}

llvm::Expected<std::optional<RVVBinaryIntrinsicDescriptor>>
findMatchingExplicitI64MicrokernelDescriptor(
    const VariantEmissionRequest &request) {
  tcrv::exec::KernelOp kernel = request.getKernel();
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!kernel || !variant || kernel.getBody().empty())
    return std::optional<RVVBinaryIntrinsicDescriptor>();

  auto rawDescriptor =
      variant->getAttrOfType<mlir::StringAttr>(
          kRVVI32VAddLoweringDescriptorAttrName);
  if (!rawDescriptor ||
      !tianchenrv::target::rvv::lookupRVVBinaryFamilyByLoweringDescriptor(
          rawDescriptor.getValue()))
    return std::optional<RVVBinaryIntrinsicDescriptor>();

  llvm::Expected<const RVVI32VectorShapeConfig *> requiredConfig =
      getVariantRequiredFirstSliceConfig(variant, request.getCapabilities());
  if (!requiredConfig)
    return requiredConfig.takeError();

  llvm::Expected<RVVCapabilityPropertyView> propertyView =
      buildRVVCapabilityPropertyView(request.getCapabilities(),
                                     *requiredConfig);
  if (!propertyView)
    return propertyView.takeError();

  llvm::Expected<std::optional<RVVI64MicrokernelMaterializationPlan>>
      descriptorPlan = buildI64MicrokernelMaterializationPlan(
          variant, request.getCapabilities(), *propertyView);
  if (!descriptorPlan)
    return descriptorPlan.takeError();
  if (!*descriptorPlan)
    return std::optional<RVVBinaryIntrinsicDescriptor>();

  const RVVBinaryIntrinsicDescriptor &descriptor =
      (*descriptorPlan)->descriptor;
  llvm::StringRef expectedRole = stringifyVariantEmissionRole(request.getRole());
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  auto variantRequiredMarch =
      variant->getAttrOfType<mlir::StringAttr>(kRVVRequiredMarchAttrName);
  unsigned matches = 0;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto microkernel = llvm::dyn_cast<tcrv::rvv::I64VAddMicrokernelOp>(op);
    if (!microkernel)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto role = getStringAttr(&op, kRoleAttrName);
    if (!selectedVariant || !role)
      return makeRVVPluginError("explicit RVV i64 microkernel emission plan "
                                "requires selected_variant and role metadata");

    if (selectedVariant.getValue() != variant.getSymName() ||
        role.getValue() != expectedRole) {
      return makeRVVPluginError(
          llvm::Twine("stale ") + descriptor.getRVVMicrokernelOpName() +
          " for @" + selectedVariant.getValue() + " as " + role.getValue() +
          " is not the selected RVV emission plan path @" +
          variant.getSymName() + " as " + expectedRole);
    }

    ++matches;
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kSourceKernelAttrName,
                                            kernel.getSymName()))
      return std::move(error);
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kOriginAttrName,
                                            kRVVPluginName))
      return std::move(error);
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kRoleAttrName, expectedRole))
      return std::move(error);

    if (!variantRequires ||
        op.getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName) !=
            variantRequires) {
      return makeRVVPluginError(
          llvm::Twine("explicit RVV i64 microkernel emission plan requires ") +
          descriptor.getRVVMicrokernelOpName() +
          " required_capabilities to match selected variant requires metadata");
    }

    if (!variantRequiredMarch || variantRequiredMarch.getValue().trim().empty())
      return makeRVVPluginError(
          "explicit RVV i64 microkernel emission plan requires selected "
          "variant metadata 'tcrv_rvv.required_march'");
    if (llvm::Error error = validateMicrokernelEmissionAttr(
            &op, kMicrokernelRequiredMarchAttrName,
            variantRequiredMarch.getValue().trim()))
      return std::move(error);

    if (auto selectedMABI = getStringAttr(&op, kSelectedMABIAttrName))
      if (llvm::Error error =
              validateRVVPropertyText("explicit RVV i64 microkernel emission "
                                      "plan",
                                      kSelectedMABIAttrName,
                                      selectedMABI.getValue().trim()))
        return std::move(error);

    auto elementCount =
        op.getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
    if (!elementCount || elementCount.getInt() <= 0 ||
        elementCount.getInt() > 64)
      return makeRVVPluginError(
          "explicit RVV i64 microkernel emission plan requires element_count "
          "in the bounded smoke range [1, 64]");
    if (elementCount.getInt() != (*descriptorPlan)->elementCount)
      return makeRVVPluginError(
          llvm::Twine("explicit RVV i64 microkernel emission plan requires ") +
          descriptor.getRVVMicrokernelOpName() +
          " element_count to match selected variant finite descriptor "
          "metadata 'tcrv_rvv.element_count'");

    if (llvm::Error error =
            validateSelectedVectorShapeMetadata(
                &op,
                (llvm::Twine("explicit RVV i64 microkernel ") +
                 descriptor.getRVVMicrokernelOpName())
                    .str(),
                *descriptor.shape, kBoundarySelectedVectorShapeAttrName,
                kBoundarySelectedVectorSEWAttrName,
                kBoundarySelectedVectorLMULAttrName,
                kBoundarySelectedTailPolicyAttrName,
                kBoundarySelectedMaskPolicyAttrName,
                kBoundarySelectedVectorTypeAttrName,
                kBoundarySelectedVectorSuffixAttrName,
                kBoundarySelectedSetVLSuffixAttrName))
      return std::move(error);
  }

  if (matches > 1)
    return makeRVVPluginError(
        llvm::Twine("selected RVV emission plan path @") +
        variant.getSymName() + " as " + expectedRole +
        " has duplicate RVV i64 microkernel metadata");

  if (matches == 0)
    return std::optional<RVVBinaryIntrinsicDescriptor>();
  return descriptor;
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

llvm::Expected<VariantProposal>
buildRVVFirstSliceProposal(const VariantProposalRequest &request) {
  const RVVI32MicrokernelFamilySpec *requestedFamily =
      &getI32VAddFamilySpec();
  const RVVBinaryFamilyDescriptor *requestedI64Family = nullptr;
  if (auto frontendLowering =
          request.getKernel()->getAttrOfType<mlir::StringAttr>(
              kFrontendLoweringAttrName)) {
    llvm::StringRef value = frontendLowering.getValue().trim();
    if (llvm::Error error = validateRVVPropertyText(
            (llvm::Twine("kernel @") + request.getKernel().getSymName() +
             " frontend lowering family")
                .str(),
            kFrontendLoweringAttrName, value))
      return std::move(error);

    const RVVI32FamilyDescriptor *requestedDescriptor =
        tianchenrv::target::i32_binary::lookupI32BinaryFamilyByFrontendLowering(
            value);
    if (requestedDescriptor) {
      requestedFamily = getI32MicrokernelFamilyByDescriptor(
          *requestedDescriptor);
      if (!requestedFamily)
        return makeRVVPluginError(
            llvm::Twine("kernel @") + request.getKernel().getSymName() +
            " frontend lowering family is not an RVV i32 microkernel family");
    } else if (const RVVBinaryFamilyDescriptor *i64Family =
                   tianchenrv::target::rvv::
                       lookupRVVBinaryFamilyByFrontendLowering(value)) {
      requestedI64Family = i64Family;
    } else {
      return makeRVVPluginError(
          llvm::Twine("kernel @") + request.getKernel().getSymName() +
          " frontend lowering family must be '" +
          getI32VAddFamilySpec().family->frontendLowering + "' or '" +
          getI32VSubFamilySpec().family->frontendLowering + "' or '" +
          getI32VMulFamilySpec().family->frontendLowering + "' or '" +
          tianchenrv::target::rvv::getI64VAddFamilyDescriptor()
              .frontendLowering +
          "'");
    }
  }

  const RVVI32VectorShapeConfig *requiredConfig =
      requestedI64Family ? &tianchenrv::target::rvv::getI64M1VectorShapeConfig()
                         : nullptr;
  llvm::Expected<RVVCapabilityPropertyView> propertyView =
      buildRVVCapabilityPropertyView(request.getCapabilities(),
                                     requiredConfig);
  if (!propertyView)
    return propertyView.takeError();

  VariantProposal proposal(kRVVFirstSliceVariantName, kRVVPluginName);
  proposal.addRequiredCapabilityID(kRVVCapabilityID);
  RVVBinaryIntrinsicDescriptor descriptor =
      requestedI64Family
          ? tianchenrv::target::rvv::getRVVBinaryIntrinsicDescriptor(
                *requestedI64Family, *propertyView->i32Config)
          : tianchenrv::target::rvv::getRVVBinaryIntrinsicDescriptor(
                tianchenrv::target::rvv::getRVVBinaryFamilyDescriptor(
                    *requestedFamily->family),
                *propertyView->i32Config);
  for (llvm::StringRef capabilityID : descriptor.getSelectedShapeCapabilityIDs())
    proposal.addRequiredCapabilityID(capabilityID);
  proposal.setCondition("rvv_capability_properties_available");
  proposal.setGuard("plugin_local_rvv_property_evidence");
  proposal.setPolicy("metadata_only_first_slice");
  proposal.addPluginAttribute(
      getRVVPolicyAttrNameAttr(request.getKernel()->getContext()),
      getExpectedRVVPolicyAttr(request.getKernel()->getContext()));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kRVVRequiredMarchAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            propertyView->selectedMarch));
  addSelectedVectorShapeMetadataToProposal(proposal,
                                           request.getKernel()->getContext(),
                                           *propertyView->i32Config);
  if (hasAvailableRVVSmokeProbeCapability(request.getCapabilities())) {
    proposal.addPluginAttribute(
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              kRVVSmokeProbeDescriptorAttrName),
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              kRVVSmokeProbeDescriptorValue));
    return proposal;
  }
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kRVVI32VAddLoweringDescriptorAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            descriptor.getLoweringDescriptor()));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kRVVI32VAddElementCountAttrName),
      mlir::IntegerAttr::get(mlir::IntegerType::get(
                                 request.getKernel()->getContext(), 64),
                             deriveI32VAddDescriptorElementCount(
                                 *propertyView)));
  if (propertyView->vlenbBytes && propertyView->i32M1LaneCount) {
    proposal.addPluginAttribute(
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              kRVVVLenBBytesAttrName),
        mlir::IntegerAttr::get(mlir::IntegerType::get(
                                   request.getKernel()->getContext(), 64),
                               *propertyView->vlenbBytes));
    proposal.addPluginAttribute(
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              kRVVI32M1LanesAttrName),
        mlir::IntegerAttr::get(mlir::IntegerType::get(
                                   request.getKernel()->getContext(), 64),
                               *propertyView->i32M1LaneCount));
  }
  return proposal;
}

llvm::Expected<tcrv::rvv::LoweringBoundaryOp> materializeRVVBoundaryOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    llvm::StringRef capabilitySummary,
    const RVVI32VectorShapeConfig &selectedConfig) {
  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(variant.getLoc(),
                             tcrv::rvv::LoweringBoundaryOp::getOperationName());
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(kRVVPluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kUnsupportedStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  addSelectedVectorShapeMetadataToOperationState(
      state, builder.getContext(), selectedConfig);
  if (llvm::Error error =
          copyCapacityMetadataToBoundary(state, variant, builder))
    return std::move(error);
  state.addAttribute(kCapabilitySummaryAttrName,
                     builder.getStringAttr(capabilitySummary));
  state.addAttribute(
      kUnsupportedReasonAttrName,
      builder.getStringAttr(
          "RVV lowering boundary is pre-executable metadata only; no RVV "
          "lowering pipeline, runtime ABI, generated artifact, correctness "
          "proof, or performance measurement is produced"));
  return llvm::cast<tcrv::rvv::LoweringBoundaryOp>(builder.create(state));
}

mlir::Operation *materializeRVVI32MicrokernelOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    const RVVI32MicrokernelMaterializationPlan &plan) {
  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(
      variant.getLoc(), plan.family->getRVV().microkernelOpName);
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(kRVVPluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kElementCountAttrName,
                     builder.getI64IntegerAttr(plan.elementCount));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  state.addAttribute(kMicrokernelRequiredMarchAttrName,
                     builder.getStringAttr(plan.requiredMarch));
  addSelectedVectorShapeMetadataToOperationState(
      state, builder.getContext(), *plan.i32Config);
  if (plan.selectedMABI)
    state.addAttribute(kSelectedMABIAttrName,
                       builder.getStringAttr(*plan.selectedMABI));

  mlir::Region *body = state.addRegion();
  auto *block = new mlir::Block();
  body->push_back(block);
  mlir::Value runtimeN =
      block->addArgument(builder.getIndexType(), variant.getLoc());

  mlir::OpBuilder bodyBuilder(builder.getContext());
  bodyBuilder.setInsertionPointToStart(block);

  tcrv::rvv::PolicyAttr policy =
      getExpectedRVVPolicyAttr(builder.getContext());

  mlir::OperationState setvlState(variant.getLoc(),
                                  tcrv::rvv::SetVLOp::getOperationName());
  setvlState.addOperands(runtimeN);
  setvlState.addTypes(tcrv::rvv::VLType::get(builder.getContext()));
  setvlState.addAttribute(kSEWAttrName,
                          builder.getI64IntegerAttr(plan.i32Config->sewBits));
  setvlState.addAttribute(kLMULAttrName,
                          builder.getStringAttr(plan.i32Config->lmul));
  setvlState.addAttribute(kPolicyAttrName, policy);
  auto setvl =
      llvm::cast<tcrv::rvv::SetVLOp>(bodyBuilder.create(setvlState));

  mlir::OperationState withVLState(variant.getLoc(),
                                   tcrv::rvv::WithVLOp::getOperationName());
  withVLState.addOperands(setvl.getVl());
  withVLState.addAttribute(kSEWAttrName,
                           builder.getI64IntegerAttr(plan.i32Config->sewBits));
  withVLState.addAttribute(kLMULAttrName,
                           builder.getStringAttr(plan.i32Config->lmul));
  withVLState.addAttribute(kPolicyAttrName, policy);
  mlir::Region *withVLBody = withVLState.addRegion();
  auto *withVLBlock = new mlir::Block();
  withVLBody->push_back(withVLBlock);

  mlir::OpBuilder withVLBodyBuilder(builder.getContext());
  withVLBodyBuilder.setInsertionPointToStart(withVLBlock);
  mlir::Type i32Vector =
      plan.i32Config->lmul == "m2"
          ? mlir::Type(tcrv::rvv::I32M2VectorType::get(builder.getContext()))
          : mlir::Type(tcrv::rvv::I32M1VectorType::get(builder.getContext()));

  mlir::OperationState lhsLoadState(variant.getLoc(),
                                    tcrv::rvv::I32LoadOp::getOperationName());
  lhsLoadState.addOperands(setvl.getVl());
  lhsLoadState.addTypes(i32Vector);
  lhsLoadState.addAttribute(
      kBufferRoleAttrName,
      builder.getStringAttr(support::stringifyRuntimeABIParameterRole(
          support::RuntimeABIParameterRole::LHSInputBuffer)));
  auto lhsLoad =
      llvm::cast<tcrv::rvv::I32LoadOp>(withVLBodyBuilder.create(lhsLoadState));

  mlir::OperationState rhsLoadState(variant.getLoc(),
                                    tcrv::rvv::I32LoadOp::getOperationName());
  rhsLoadState.addOperands(setvl.getVl());
  rhsLoadState.addTypes(i32Vector);
  rhsLoadState.addAttribute(
      kBufferRoleAttrName,
      builder.getStringAttr(support::stringifyRuntimeABIParameterRole(
          support::RuntimeABIParameterRole::RHSInputBuffer)));
  auto rhsLoad =
      llvm::cast<tcrv::rvv::I32LoadOp>(withVLBodyBuilder.create(rhsLoadState));

  mlir::OperationState arithmeticState(variant.getLoc(),
                                       plan.family->getRVV().arithmeticOpName);
  arithmeticState.addOperands(
      {lhsLoad.getLoaded(), rhsLoad.getLoaded(), setvl.getVl()});
  arithmeticState.addTypes(i32Vector);
  mlir::Operation *arithmetic = withVLBodyBuilder.create(arithmeticState);
  mlir::Value arithmeticResult = arithmetic->getResult(0);

  mlir::OperationState storeState(variant.getLoc(),
                                  tcrv::rvv::I32StoreOp::getOperationName());
  storeState.addOperands({arithmeticResult, setvl.getVl()});
  storeState.addAttribute(
      kBufferRoleAttrName,
      builder.getStringAttr(support::stringifyRuntimeABIParameterRole(
          support::RuntimeABIParameterRole::OutputBuffer)));
  withVLBodyBuilder.create(storeState);

  bodyBuilder.create(withVLState);

  return builder.create(state);
}

mlir::Operation *materializeRVVI64MicrokernelOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    const RVVI64MicrokernelMaterializationPlan &plan) {
  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(variant.getLoc(),
                             plan.descriptor.getRVVMicrokernelOpName());
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(kRVVPluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kElementCountAttrName,
                     builder.getI64IntegerAttr(plan.elementCount));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  state.addAttribute(kMicrokernelRequiredMarchAttrName,
                     builder.getStringAttr(plan.requiredMarch));
  addSelectedVectorShapeMetadataToOperationState(
      state, builder.getContext(), *plan.descriptor.shape);
  if (plan.selectedMABI)
    state.addAttribute(kSelectedMABIAttrName,
                       builder.getStringAttr(*plan.selectedMABI));

  mlir::Region *body = state.addRegion();
  auto *block = new mlir::Block();
  body->push_back(block);
  mlir::Value runtimeN =
      block->addArgument(builder.getIndexType(), variant.getLoc());

  mlir::OpBuilder bodyBuilder(builder.getContext());
  bodyBuilder.setInsertionPointToStart(block);

  tcrv::rvv::PolicyAttr policy =
      getExpectedRVVPolicyAttr(builder.getContext());

  mlir::OperationState setvlState(variant.getLoc(),
                                  tcrv::rvv::SetVLOp::getOperationName());
  setvlState.addOperands(runtimeN);
  setvlState.addTypes(tcrv::rvv::VLType::get(builder.getContext()));
  setvlState.addAttribute(kSEWAttrName,
                          builder.getI64IntegerAttr(
                              plan.descriptor.getSEWBits()));
  setvlState.addAttribute(kLMULAttrName,
                          builder.getStringAttr(plan.descriptor.getLMUL()));
  setvlState.addAttribute(kPolicyAttrName, policy);
  auto setvl =
      llvm::cast<tcrv::rvv::SetVLOp>(bodyBuilder.create(setvlState));

  mlir::OperationState withVLState(variant.getLoc(),
                                   tcrv::rvv::WithVLOp::getOperationName());
  withVLState.addOperands(setvl.getVl());
  withVLState.addAttribute(kSEWAttrName,
                           builder.getI64IntegerAttr(
                               plan.descriptor.getSEWBits()));
  withVLState.addAttribute(kLMULAttrName,
                           builder.getStringAttr(plan.descriptor.getLMUL()));
  withVLState.addAttribute(kPolicyAttrName, policy);
  mlir::Region *withVLBody = withVLState.addRegion();
  auto *withVLBlock = new mlir::Block();
  withVLBody->push_back(withVLBlock);

  mlir::OpBuilder withVLBodyBuilder(builder.getContext());
  withVLBodyBuilder.setInsertionPointToStart(withVLBlock);
  mlir::Type i64Vector =
      tcrv::rvv::I64M1VectorType::get(builder.getContext());

  mlir::OperationState lhsLoadState(variant.getLoc(),
                                    tcrv::rvv::I64LoadOp::getOperationName());
  lhsLoadState.addOperands(setvl.getVl());
  lhsLoadState.addTypes(i64Vector);
  lhsLoadState.addAttribute(
      kBufferRoleAttrName,
      builder.getStringAttr(support::stringifyRuntimeABIParameterRole(
          support::RuntimeABIParameterRole::LHSInputBuffer)));
  auto lhsLoad =
      llvm::cast<tcrv::rvv::I64LoadOp>(withVLBodyBuilder.create(lhsLoadState));

  mlir::OperationState rhsLoadState(variant.getLoc(),
                                    tcrv::rvv::I64LoadOp::getOperationName());
  rhsLoadState.addOperands(setvl.getVl());
  rhsLoadState.addTypes(i64Vector);
  rhsLoadState.addAttribute(
      kBufferRoleAttrName,
      builder.getStringAttr(support::stringifyRuntimeABIParameterRole(
          support::RuntimeABIParameterRole::RHSInputBuffer)));
  auto rhsLoad =
      llvm::cast<tcrv::rvv::I64LoadOp>(withVLBodyBuilder.create(rhsLoadState));

  mlir::OperationState arithmeticState(variant.getLoc(),
                                       tcrv::rvv::I64AddOp::getOperationName());
  arithmeticState.addOperands(
      {lhsLoad.getLoaded(), rhsLoad.getLoaded(), setvl.getVl()});
  arithmeticState.addTypes(i64Vector);
  auto add =
      llvm::cast<tcrv::rvv::I64AddOp>(withVLBodyBuilder.create(arithmeticState));

  mlir::OperationState storeState(variant.getLoc(),
                                  tcrv::rvv::I64StoreOp::getOperationName());
  storeState.addOperands({add.getSum(), setvl.getVl()});
  storeState.addAttribute(
      kBufferRoleAttrName,
      builder.getStringAttr(support::stringifyRuntimeABIParameterRole(
          support::RuntimeABIParameterRole::OutputBuffer)));
  withVLBodyBuilder.create(storeState);

  bodyBuilder.create(withVLState);

  return builder.create(state);
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
      "RVV finite i32m2 SEW=32 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2LMULM2CapabilityID(), "isa-vector-config",
      "RVV finite i32m2 LMUL=m2 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2TailAgnosticCapabilityID(), "isa-vector-config",
      "RVV finite i32m2 tail agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2MaskAgnosticCapabilityID(), "isa-vector-config",
      "RVV finite i32m2 mask agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig().sewCapabilityID,
      "isa-vector-config",
      "RVV finite i64m1 SEW=64 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig().lmulCapabilityID,
      "isa-vector-config",
      "RVV finite i64m1 LMUL=m1 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig()
          .tailPolicyCapabilityID,
      "isa-vector-config",
      "RVV finite i64m1 tail agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig()
          .maskPolicyCapabilityID,
      "isa-vector-config",
      "RVV finite i64m1 mask agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::
          getRVVI32BinarySelectedVectorShapeCapabilityID(),
      "isa-vector-config",
      "RVV finite i32 binary selected vector-shape selector capability"));
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

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVPluginError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  if (!request.getCapabilities().isCapabilityAvailableByID(kRVVCapabilityID))
    return makeRVVPluginError(
        "materialized RVV variant requires an available capability id 'rvv'");

  if (llvm::Error error = verifyVariantRequiresCapabilityID(
          variant, request.getCapabilities(), kRVVCapabilityID))
    return error;
  llvm::Expected<const RVVI32VectorShapeConfig *> requiredConfig =
      getVariantRequiredFirstSliceConfig(variant, request.getCapabilities());
  if (!requiredConfig)
    return requiredConfig.takeError();

  if (auto loweringDescriptor =
          variant->getAttrOfType<mlir::StringAttr>(
              kRVVI32VAddLoweringDescriptorAttrName)) {
    if (const RVVBinaryFamilyDescriptor *binaryFamily =
            tianchenrv::target::rvv::lookupRVVBinaryFamilyByLoweringDescriptor(
                loweringDescriptor.getValue())) {
      if ((*requiredConfig)->dtypeID != binaryFamily->dtypeID)
        return makeRVVPluginError(
            llvm::Twine(binaryFamily->descriptorNoun) + " on variant @" +
            variant.getSymName() + " requires finite " +
            tianchenrv::target::rvv::getI64M1VectorShapeConfig().shapeID +
            " vector-shape config capability ids");
    }
  }

  if (llvm::Error error = verifyExpectedRVVPolicyAttr(variant))
    return error;

  if (llvm::Error error =
          validateSelectedVectorShapeMetadata(
              variant.getOperation(),
              (llvm::Twine("materialized RVV variant @") +
               variant.getSymName())
                  .str(),
              **requiredConfig, kRVVSelectedVectorShapeAttrName,
              kRVVSelectedVectorSEWAttrName, kRVVSelectedVectorLMULAttrName,
              kRVVSelectedTailPolicyAttrName, kRVVSelectedMaskPolicyAttrName,
              kRVVSelectedVectorTypeAttrName,
              kRVVSelectedVectorSuffixAttrName,
              kRVVSelectedSetVLSuffixAttrName))
    return error;

  if (variant->hasAttr(kRVVRequiredMarchAttrName) ||
      variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName) ||
      variant->hasAttr(kRVVSmokeProbeDescriptorAttrName) ||
      variant->hasAttr(kRVVVLenBBytesAttrName) ||
      variant->hasAttr(kRVVI32M1LanesAttrName)) {
    llvm::Expected<RVVCapabilityPropertyView> propertyView =
        buildRVVCapabilityPropertyView(request.getCapabilities(),
                                       *requiredConfig);
    if (!propertyView)
      return propertyView.takeError();

    if (variant->hasAttr(kRVVRequiredMarchAttrName))
      if (llvm::Error error = verifyRequiredMarchAttr(variant, *propertyView))
        return error;
    if (llvm::Error error = verifySmokeProbeDescriptorAttr(variant))
      return error;
    if (llvm::Error error =
            verifyOptionalCapacityAttrs(variant, *propertyView))
      return error;

    llvm::Expected<std::optional<RVVI32MicrokernelMaterializationPlan>>
        microkernelPlan = buildI32MicrokernelMaterializationPlan(
            variant, request.getCapabilities(), *propertyView);
    if (!microkernelPlan)
      return microkernelPlan.takeError();
    llvm::Expected<std::optional<RVVI64MicrokernelMaterializationPlan>>
        i64MicrokernelPlan = buildI64MicrokernelMaterializationPlan(
            variant, request.getCapabilities(), *propertyView);
    if (!i64MicrokernelPlan)
      return i64MicrokernelPlan.takeError();
  }

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
  llvm::Expected<RVVCapabilityPropertyView> propertyView =
      buildRVVCapabilityPropertyView(request.getCapabilities());
  if (!propertyView) {
    llvm::consumeError(propertyView.takeError());
  } else if (propertyView->i32M1LaneCount) {
    out.setScore(1.0 / static_cast<double>(*propertyView->i32M1LaneCount));
    out.setExplanation(
        (llvm::Twine("RVV metadata-only first slice; capability-derived "
                     "base_i32_m1_lanes=") +
         std::to_string(*propertyView->i32M1LaneCount) +
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

  llvm::Expected<const RVVI32MicrokernelFamilySpec *> microkernelFamily =
      findMatchingExplicitMicrokernelFamily(request);
  if (!microkernelFamily)
    return microkernelFamily.takeError();
  if (*microkernelFamily) {
    out = VariantEmissionStatus::getSupported(
        kRVVPluginName, request.getVariant().getSymName(),
        (*microkernelFamily)->emissionPath);
    return llvm::Error::success();
  }

  llvm::Expected<std::optional<RVVBinaryIntrinsicDescriptor>>
      i64MicrokernelDescriptor =
          findMatchingExplicitI64MicrokernelDescriptor(request);
  if (!i64MicrokernelDescriptor)
    return i64MicrokernelDescriptor.takeError();
  if (*i64MicrokernelDescriptor) {
    out = VariantEmissionStatus::getSupported(
        kRVVPluginName, request.getVariant().getSymName(),
        "rvv-explicit-i64-vadd-microkernel-c-source-export");
    return llvm::Error::success();
  }

  if (request.getVariant()->hasAttr(kRVVSmokeProbeDescriptorAttrName)) {
    llvm::Expected<RVVCapabilityPropertyView> propertyView =
        buildRVVCapabilityPropertyView(request.getCapabilities());
    if (!propertyView)
      return propertyView.takeError();
    if (llvm::Error error = verifySmokeProbeDescriptorAttr(request.getVariant()))
      return error;
    if (llvm::Error error =
            verifyRequiredMarchAttr(request.getVariant(), *propertyView))
      return error;
    out = VariantEmissionStatus::getSupported(
        kRVVPluginName, request.getVariant().getSymName(),
        kSmokeProbeEmissionPath);
    return llvm::Error::success();
  }

  out = VariantEmissionStatus::getUnsupported(
      kRVVPluginName, request.getVariant().getSymName(),
      "RVV metadata-only first slice has no RVV lowering, runtime ABI, or "
      "executable emission path; this is an explicit diagnostic boundary and "
      "not RVV hardware/toolchain/runtime/correctness/performance evidence");
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

  llvm::Expected<const RVVI32VectorShapeConfig *> selectedConfig =
      getVariantRequiredFirstSliceConfig(request.getVariant(),
                                         request.getCapabilities());
  if (!selectedConfig)
    return selectedConfig.takeError();

  llvm::Expected<const RVVI32MicrokernelFamilySpec *> microkernelFamily =
      findMatchingExplicitMicrokernelFamily(request);
  if (!microkernelFamily)
    return microkernelFamily.takeError();
  if (*microkernelFamily) {
    const RVVI32MicrokernelFamilySpec &family = **microkernelFamily;
    out = VariantEmissionPlan::getSupported(
        kRVVPluginName, request.getKernel().getSymName(),
        request.getVariant().getSymName(), request.getRole(),
        family.getRVV().emissionKind, family.getRVV().routeID,
        family.getRVV().runtimeABI,
        "runtime-callable-c-source", family.supportedMessage);
    out.setRuntimeABIKind(family.getRVV().runtimeABIKind);
    out.setRuntimeABIName(family.getRVV().runtimeABIName);
    out.setRuntimeGlueRole(family.getRVV().runtimeGlueRole);
    llvm::Expected<support::I32BinaryCallableABIPlan> callablePlan =
        support::buildI32BinaryCallableABIPlan(request.getKernel(),
                                              *family.family);
    if (!callablePlan)
      return callablePlan.takeError();
    out.addRuntimeABIParameters(callablePlan->parameters);
    if (llvm::Error error =
            out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
      return error;
    if (llvm::Error error = addSelectedVectorShapeMetadataToPlan(
            out, request.getVariant(), **selectedConfig))
      return error;
    if (llvm::Error error =
            addSelectedCapacityMetadataToPlan(out, request.getVariant()))
      return error;
    return llvm::Error::success();
  }

  llvm::Expected<std::optional<RVVBinaryIntrinsicDescriptor>>
      i64MicrokernelDescriptor =
          findMatchingExplicitI64MicrokernelDescriptor(request);
  if (!i64MicrokernelDescriptor)
    return i64MicrokernelDescriptor.takeError();
  if (*i64MicrokernelDescriptor) {
    const RVVBinaryIntrinsicDescriptor &descriptor = **i64MicrokernelDescriptor;
    out = VariantEmissionPlan::getSupported(
        kRVVPluginName, request.getKernel().getSymName(),
        request.getVariant().getSymName(), request.getRole(),
        descriptor.family.emissionKind, descriptor.getRVVRouteID(),
        descriptor.getRVVRuntimeABI(), "runtime-callable-c-source",
        "explicit RVV i64 vector-add microkernel C source export provides a "
        "library-style runtime-callable C ABI function for this selected path; "
        "this is not generic dtype lowering, runtime correctness, or "
        "performance evidence");
    out.setRuntimeABIKind(descriptor.getRVVRuntimeABIKind());
    out.setRuntimeABIName(descriptor.getRVVRuntimeABIName());
    out.setRuntimeGlueRole(descriptor.getRVVRuntimeGlueRole());
    llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 4>>
        parameters = buildRVVBinaryCallableRuntimeABIParameters(
            request.getKernel(), descriptor);
    if (!parameters)
      return parameters.takeError();
    out.addRuntimeABIParameters(*parameters);
    if (llvm::Error error =
            out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
      return error;
    if (llvm::Error error = addSelectedVectorShapeMetadataToPlan(
            out, request.getVariant(), **selectedConfig))
      return error;
    if (llvm::Error error =
            addSelectedCapacityMetadataToPlan(out, request.getVariant()))
      return error;
    return llvm::Error::success();
  }

  if (request.getVariant()->hasAttr(kRVVSmokeProbeDescriptorAttrName)) {
    llvm::Expected<RVVCapabilityPropertyView> propertyView =
        buildRVVCapabilityPropertyView(request.getCapabilities());
    if (!propertyView)
      return propertyView.takeError();
    if (llvm::Error error = verifySmokeProbeDescriptorAttr(request.getVariant()))
      return error;
    if (llvm::Error error =
            verifyRequiredMarchAttr(request.getVariant(), *propertyView))
      return error;

    out = VariantEmissionPlan::getSupported(
        kRVVPluginName, request.getKernel().getSymName(),
        request.getVariant().getSymName(), request.getRole(),
        kSmokeProbeEmissionKind, kSmokeProbeRouteID, kSmokeProbeRuntimeABI,
        kSmokeProbeArtifactKind,
        "RVV standalone smoke-probe C source export provides a bounded "
        "toolchain/header smoke program for this selected RVV path; it is not "
        "generic RVV lowering, runtime ABI glue, kernel correctness evidence, "
        "or performance evidence");
    out.setRuntimeABIKind(kSmokeProbeRuntimeABIKind);
    out.setRuntimeABIName(kSmokeProbeRuntimeABI);
    out.setRuntimeGlueRole(kSmokeProbeRuntimeGlueRole);
    if (llvm::Error error =
            out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
      return error;
    if (llvm::Error error = addSelectedVectorShapeMetadataToPlan(
            out, request.getVariant(), **selectedConfig))
      return error;
    if (llvm::Error error =
            addSelectedCapacityMetadataToPlan(out, request.getVariant()))
      return error;
    return llvm::Error::success();
  }

  out = VariantEmissionPlan::getUnsupported(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "RVV metadata-only first slice has no RVV lowering pipeline, runtime "
      "ABI, artifact contract, or executable emission path; this unsupported "
      "emission plan is a plugin-owned diagnostic boundary and not RVV "
      "hardware/toolchain/runtime/correctness/performance evidence");
  out.setEmissionKind("rvv-unsupported-metadata-boundary");
  out.setLoweringPipeline("rvv-none-executable-unsupported");
  out.setRuntimeABI("rvv-none-executable-unsupported");
  out.setRuntimeABIKind("rvv-plugin-deferred-runtime-abi");
  out.setRuntimeABIName("rvv-executable-runtime-abi-deferred");
  out.setRuntimeGlueRole("deferred-rvv-runtime-glue");
  out.setArtifactKind("unsupported-emission-diagnostic");
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  if (llvm::Error error = addSelectedVectorShapeMetadataToPlan(
          out, request.getVariant(), **selectedConfig))
    return error;
  if (llvm::Error error =
          addSelectedCapacityMetadataToPlan(out, request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVPluginError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");

  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeRVVPluginError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  VariantLegalityRequest legality(variant, kernel, request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeRVVPluginError(
        llvm::Twine("selected RVV variant @") + variant.getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  if (request.getRole() == VariantEmissionRole::DispatchFallback) {
    out = VariantLoweringBoundaryResult::getNoBoundary(
        kRVVPluginName, kernel.getSymName(), variant.getSymName(),
        request.getRole(),
        "RVV first slice does not materialize dispatch fallback lowering "
        "boundaries");
    return llvm::Error::success();
  }

  if (llvm::Error error = rejectExistingRVVBoundaryForVariant(kernel, variant))
    return error;

  llvm::Expected<const RVVI32VectorShapeConfig *> selectedConfig =
      getVariantRequiredFirstSliceConfig(variant, request.getCapabilities());
  if (!selectedConfig)
    return selectedConfig.takeError();

  llvm::Expected<std::string> capabilitySummary =
      buildRVVCapabilitySummary(variant, request.getCapabilities());
  if (!capabilitySummary)
    return capabilitySummary.takeError();

  std::optional<RVVI32MicrokernelMaterializationPlan> microkernelPlan;
  std::optional<RVVI64MicrokernelMaterializationPlan> i64MicrokernelPlan;
  bool selectedPathHasExistingI32Microkernel = false;
  if (variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName)) {
    llvm::Expected<RVVCapabilityPropertyView> propertyView =
        buildRVVCapabilityPropertyView(request.getCapabilities(),
                                       *selectedConfig);
    if (!propertyView)
      return propertyView.takeError();

    llvm::Expected<std::optional<RVVI32MicrokernelMaterializationPlan>> planned =
        buildI32MicrokernelMaterializationPlan(variant, request.getCapabilities(),
                                        *propertyView);
    if (!planned)
      return planned.takeError();
    microkernelPlan = std::move(*planned);

    llvm::Expected<std::optional<RVVI64MicrokernelMaterializationPlan>>
        plannedI64 = buildI64MicrokernelMaterializationPlan(
            variant, request.getCapabilities(), *propertyView);
    if (!plannedI64)
      return plannedI64.takeError();
    i64MicrokernelPlan = std::move(*plannedI64);
  }

  bool selectedPathHasCallableMicrokernel =
      microkernelPlan.has_value() || i64MicrokernelPlan.has_value();
  if (!selectedPathHasCallableMicrokernel) {
    VariantEmissionRequest emissionRequest(variant, kernel,
                                           request.getCapabilities(),
                                           request.getRole());
    llvm::Expected<const RVVI32MicrokernelFamilySpec *> explicitMicrokernel =
        findMatchingExplicitMicrokernelFamily(emissionRequest);
    if (!explicitMicrokernel)
      return explicitMicrokernel.takeError();
    selectedPathHasExistingI32Microkernel = *explicitMicrokernel != nullptr;
    selectedPathHasCallableMicrokernel = selectedPathHasExistingI32Microkernel;
  }

  if (microkernelPlan || i64MicrokernelPlan)
    if (llvm::Error error =
            rejectExistingRVVMicrokernelForSelectedPath(kernel, variant,
                                                        request.getRole()))
      return error;

  if (microkernelPlan || selectedPathHasExistingI32Microkernel)
    if (llvm::Error error = support::ensureRuntimeABIBufferMemWindows(
            kernel, request.getBuilder(),
            support::getI32BinaryBufferMemWindowSpecs()))
      return error;

  if (i64MicrokernelPlan)
    if (llvm::Error error = support::ensureRuntimeABIBufferMemWindows(
            kernel, request.getBuilder(),
            i64MicrokernelPlan->descriptor.getBufferMemWindowSpecs()))
      return error;

  if (selectedPathHasCallableMicrokernel) {
    llvm::SmallVector<support::RuntimeABIParamSpec, 1> runtimeParamSpecs;
    auto countSpecs =
        i64MicrokernelPlan
            ? i64MicrokernelPlan->descriptor.getRuntimeElementCountParamSpecs()
            : support::getI32BinaryRuntimeElementCountParamSpecs();
    runtimeParamSpecs.append(countSpecs.begin(), countSpecs.end());
    if (llvm::Error error =
            support::ensureRuntimeABIParamsAllowingExistingCNames(
                kernel, request.getBuilder(), runtimeParamSpecs))
      return error;
  }

  llvm::Expected<tcrv::rvv::LoweringBoundaryOp> boundary =
      materializeRVVBoundaryOp(
      request.getBuilder(), kernel, variant, request.getRole(),
      *capabilitySummary, **selectedConfig);
  if (!boundary)
    return boundary.takeError();
  if (microkernelPlan)
    materializeRVVI32MicrokernelOp(request.getBuilder(), kernel, variant,
                                   request.getRole(), *microkernelPlan);
  if (i64MicrokernelPlan)
    materializeRVVI64MicrokernelOp(request.getBuilder(), kernel, variant,
                                   request.getRole(), *i64MicrokernelPlan);
  out = VariantLoweringBoundaryResult::getMaterialized(
      kRVVPluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(), boundary->getOperation());
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVPluginError(
        "lowering-boundary validation requires a materialized "
        "tcrv.exec.variant");

  mlir::Operation *boundary = request.getBoundary();
  if (!boundary)
    return makeRVVPluginError(
        "lowering-boundary validation requires a materialized boundary op");

  if (!llvm::isa<tcrv::rvv::LoweringBoundaryOp>(boundary))
    return makeRVVPluginError(
        llvm::Twine("selected RVV path @") + variant.getSymName() +
        " requires tcrv_rvv.lowering_boundary metadata");

  bool hasCapacityMetadata =
      hasAnyCapacityMetadata(variant.getOperation(), kRVVVLenBBytesAttrName,
                             kRVVI32M1LanesAttrName) ||
      hasAnyCapacityMetadata(boundary, kRVVBoundaryVLenBBytesAttrName,
                             kRVVBoundaryI32M1LanesAttrName);
  bool hasSelectedShapeMetadata =
      hasAnySelectedVectorShapeMetadata(
          variant.getOperation(), kRVVSelectedVectorShapeAttrName,
          kRVVSelectedVectorSEWAttrName, kRVVSelectedVectorLMULAttrName,
          kRVVSelectedTailPolicyAttrName, kRVVSelectedMaskPolicyAttrName,
          kRVVSelectedVectorTypeAttrName, kRVVSelectedVectorSuffixAttrName,
          kRVVSelectedSetVLSuffixAttrName) ||
      hasAnySelectedVectorShapeMetadata(
          boundary, kBoundarySelectedVectorShapeAttrName,
          kBoundarySelectedVectorSEWAttrName, kBoundarySelectedVectorLMULAttrName,
          kBoundarySelectedTailPolicyAttrName,
          kBoundarySelectedMaskPolicyAttrName,
          kBoundarySelectedVectorTypeAttrName,
          kBoundarySelectedVectorSuffixAttrName,
          kBoundarySelectedSetVLSuffixAttrName);
  bool requiresSelectedLegality =
      variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName) ||
      variant->hasAttr(kRVVSmokeProbeDescriptorAttrName) ||
      variant->hasAttr(kRVVRequiredMarchAttrName) || hasCapacityMetadata ||
      hasSelectedShapeMetadata;
  if (!requiresSelectedLegality)
    return llvm::Error::success();

  VariantLegalityRequest legality(variant, request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeRVVPluginError(
      llvm::Twine("selected RVV variant @") + variant.getSymName() +
      " failed plugin legality before boundary validation: " + message);
  }

  llvm::Expected<const RVVI32VectorShapeConfig *> selectedConfig =
      getVariantRequiredFirstSliceConfig(variant, request.getCapabilities());
  if (!selectedConfig)
    return selectedConfig.takeError();
  if (llvm::Error error =
          validateSelectedVectorShapeMetadata(
              boundary,
              (llvm::Twine("selected RVV lowering boundary for @") +
               variant.getSymName())
                  .str(),
              **selectedConfig, kBoundarySelectedVectorShapeAttrName,
              kBoundarySelectedVectorSEWAttrName,
              kBoundarySelectedVectorLMULAttrName,
              kBoundarySelectedTailPolicyAttrName,
              kBoundarySelectedMaskPolicyAttrName,
              kBoundarySelectedVectorTypeAttrName,
              kBoundarySelectedVectorSuffixAttrName,
              kBoundarySelectedSetVLSuffixAttrName))
    return error;

  if (!hasCapacityMetadata)
    return llvm::Error::success();

  return validateBoundaryCapacityMetadata(variant, boundary);
}

} // namespace rvv

llvm::Error registerRVVExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinRVVExtensionPlugin());
}

} // namespace tianchenrv::plugin
