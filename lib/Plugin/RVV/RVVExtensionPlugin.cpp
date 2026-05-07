#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "mlir/IR/Attributes.h"
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
constexpr llvm::StringLiteral kRVVI32VAddLoweringDescriptorAttrName(
    "tcrv_rvv.lowering_descriptor");
constexpr llvm::StringLiteral kRVVI32VAddLoweringDescriptorValue(
    "i32-vadd-microkernel.v1");
constexpr llvm::StringLiteral kRVVI32VAddElementCountAttrName(
    "tcrv_rvv.element_count");
constexpr llvm::StringLiteral kArchitecturePropertyName("architecture");
constexpr llvm::StringLiteral kISAVectorHintsPropertyName("isa_vector_hints");
constexpr llvm::StringLiteral kHartCountPropertyName("count");
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
constexpr llvm::StringLiteral kMicrokernelEmissionPath(
    "rvv-explicit-i32-vadd-microkernel-c-source-export");
constexpr std::int64_t kDefaultI32VAddElementCount = 16;

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

struct RVVCapabilityPropertyView {
  std::string architecture;
  std::string isaVectorHints;
  std::string selectedMarch;
  std::uint64_t hartCount = 0;
};

struct RVVI32VAddMaterializationPlan {
  std::int64_t elementCount = 0;
  std::string requiredMarch;
  std::optional<std::string> selectedMABI;
};

bool hasAvailableRVVCapability(const VariantProposalRequest &request) {
  return request.getKernel() &&
         request.getCapabilities().isCapabilityAvailableByID(kRVVCapabilityID);
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
  out = capabilities.lookupByID(id);
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

llvm::Expected<RVVCapabilityPropertyView>
buildRVVCapabilityPropertyView(
    const support::TargetCapabilitySet &capabilities) {
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

  RVVCapabilityPropertyView view;
  view.architecture = std::move(*architecture);
  view.isaVectorHints = std::move(*isaVectorHints);
  view.selectedMarch = std::move(*selectedMarch);
  view.hartCount = *hartCount;
  return view;
}

llvm::Expected<bool>
variantRequiresRVV(tcrv::exec::VariantOp variant,
                   const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeRVVPluginError(
        "materialized RVV variant requires structured 'requires' metadata");

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

    if (capability->getID() == kRVVCapabilityID)
      return true;
  }

  return false;
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

llvm::Expected<std::optional<RVVI32VAddMaterializationPlan>>
buildI32VAddMaterializationPlan(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    const RVVCapabilityPropertyView &view) {
  mlir::Attribute rawDescriptor =
      variant->getAttr(kRVVI32VAddLoweringDescriptorAttrName);
  if (!rawDescriptor)
    return std::optional<RVVI32VAddMaterializationPlan>();

  auto descriptor = llvm::dyn_cast<mlir::StringAttr>(rawDescriptor);
  if (!descriptor || descriptor.getValue().trim().empty())
    return makeRVVPluginError(
        llvm::Twine("finite RVV i32-vadd lowering descriptor on variant @") +
        variant.getSymName() + " requires string attribute '" +
        kRVVI32VAddLoweringDescriptorAttrName + "'");

  std::string descriptorContext =
      (llvm::Twine("variant @") + variant.getSymName() +
       " finite RVV i32-vadd lowering descriptor")
          .str();
  if (llvm::Error error = validateRVVPropertyText(
          descriptorContext, kRVVI32VAddLoweringDescriptorAttrName,
          descriptor.getValue().trim()))
    return std::move(error);

  if (descriptor.getValue().trim() != kRVVI32VAddLoweringDescriptorValue)
    return makeRVVPluginError(
        llvm::Twine("finite RVV i32-vadd lowering descriptor on variant @") +
        variant.getSymName() + " must be '" +
        kRVVI32VAddLoweringDescriptorValue + "'");

  auto elementCountAttr =
      variant->getAttrOfType<mlir::IntegerAttr>(
          kRVVI32VAddElementCountAttrName);
  if (!elementCountAttr)
    return makeRVVPluginError(
        llvm::Twine("finite RVV i32-vadd lowering descriptor on variant @") +
        variant.getSymName() + " requires integer attribute '" +
        kRVVI32VAddElementCountAttrName + "'");

  std::int64_t elementCount = elementCountAttr.getInt();
  if (elementCount <= 0 || elementCount > 64)
    return makeRVVPluginError(
        llvm::Twine("finite RVV i32-vadd lowering descriptor on variant @") +
        variant.getSymName() +
        " requires tcrv_rvv.element_count in the bounded smoke range [1, 64]");

  auto requiredMarch =
      variant->getAttrOfType<mlir::StringAttr>(kRVVRequiredMarchAttrName);
  if (!requiredMarch || requiredMarch.getValue().trim().empty())
    return makeRVVPluginError(
        llvm::Twine("finite RVV i32-vadd lowering descriptor on variant @") +
        variant.getSymName() + " requires string 'tcrv_rvv.required_march' "
                               "metadata");

  if (llvm::Error error = verifyRequiredMarchAttr(variant, view))
    return std::move(error);

  llvm::Expected<std::optional<std::string>> selectedMABI =
      getOptionalSelectedMABI(capabilities);
  if (!selectedMABI)
    return selectedMABI.takeError();

  RVVI32VAddMaterializationPlan plan;
  plan.elementCount = elementCount;
  plan.requiredMarch = requiredMarch.getValue().trim().str();
  plan.selectedMABI = std::move(*selectedMABI);
  return plan;
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
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
    auto microkernel = llvm::dyn_cast<tcrv::rvv::I32VAddMicrokernelOp>(op);
    if (!microkernel)
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
        llvm::Twine("requires no pre-existing "
                    "tcrv_rvv.i32_vadd_microkernel for target @") +
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

llvm::Expected<std::optional<RVVI32VAddMaterializationPlan>>
buildDescriptorPlanForEmission(const VariantEmissionRequest &request) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant || !variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName))
    return std::optional<RVVI32VAddMaterializationPlan>();

  llvm::Expected<RVVCapabilityPropertyView> propertyView =
      buildRVVCapabilityPropertyView(request.getCapabilities());
  if (!propertyView)
    return propertyView.takeError();

  return buildI32VAddMaterializationPlan(variant, request.getCapabilities(),
                                         *propertyView);
}

llvm::Expected<bool> hasMatchingExplicitMicrokernel(
    const VariantEmissionRequest &request) {
  tcrv::exec::KernelOp kernel = request.getKernel();
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!kernel || !variant || kernel.getBody().empty())
    return false;

  llvm::Expected<std::optional<RVVI32VAddMaterializationPlan>>
      descriptorPlan = buildDescriptorPlanForEmission(request);
  if (!descriptorPlan)
    return descriptorPlan.takeError();

  llvm::StringRef expectedRole = stringifyVariantEmissionRole(request.getRole());
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  auto variantRequiredMarch =
      variant->getAttrOfType<mlir::StringAttr>(kRVVRequiredMarchAttrName);

  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto microkernel = llvm::dyn_cast<tcrv::rvv::I32VAddMicrokernelOp>(op);
    if (!microkernel)
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
          llvm::Twine("stale tcrv_rvv.i32_vadd_microkernel for @") +
          selectedVariant.getValue() + " as " + role.getValue() +
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
        microkernel->getAttrOfType<mlir::ArrayAttr>(
            kRequiredCapabilitiesAttrName) != variantRequires) {
      return makeRVVPluginError(
          "explicit RVV microkernel emission plan requires "
          "tcrv_rvv.i32_vadd_microkernel required_capabilities to match "
          "selected variant requires metadata");
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
          "explicit RVV microkernel emission plan requires "
          "tcrv_rvv.i32_vadd_microkernel element_count to match selected "
          "variant finite descriptor metadata 'tcrv_rvv.element_count'");
  }

  if (matches > 1)
    return makeRVVPluginError(
        llvm::Twine("selected RVV emission plan path @") +
        variant.getSymName() + " as " + expectedRole +
        " has duplicate tcrv_rvv.i32_vadd_microkernel metadata");

  return matches == 1;
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
  llvm::Expected<RVVCapabilityPropertyView> propertyView =
      buildRVVCapabilityPropertyView(request.getCapabilities());
  if (!propertyView)
    return propertyView.takeError();

  VariantProposal proposal(kRVVFirstSliceVariantName, kRVVPluginName);
  proposal.addRequiredCapabilityID(kRVVCapabilityID);
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
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kRVVI32VAddLoweringDescriptorAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kRVVI32VAddLoweringDescriptorValue));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kRVVI32VAddElementCountAttrName),
      mlir::IntegerAttr::get(mlir::IntegerType::get(
                                 request.getKernel()->getContext(), 64),
                             kDefaultI32VAddElementCount));
  return proposal;
}

tcrv::rvv::LoweringBoundaryOp materializeRVVBoundaryOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    llvm::StringRef capabilitySummary) {
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

tcrv::rvv::I32VAddMicrokernelOp materializeRVVI32VAddMicrokernelOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    const RVVI32VAddMaterializationPlan &plan) {
  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(
      variant.getLoc(), tcrv::rvv::I32VAddMicrokernelOp::getOperationName());
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
  if (plan.selectedMABI)
    state.addAttribute(kSelectedMABIAttrName,
                       builder.getStringAttr(*plan.selectedMABI));
  return llvm::cast<tcrv::rvv::I32VAddMicrokernelOp>(builder.create(state));
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

  llvm::Expected<bool> requiresRVV =
      variantRequiresRVV(variant, request.getCapabilities());
  if (!requiresRVV)
    return requiresRVV.takeError();

  if (!*requiresRVV)
    return makeRVVPluginError(
        "materialized RVV variant must require capability id 'rvv'");

  if (llvm::Error error = verifyExpectedRVVPolicyAttr(variant))
    return error;

  if (variant->hasAttr(kRVVRequiredMarchAttrName) ||
      variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName)) {
    llvm::Expected<RVVCapabilityPropertyView> propertyView =
        buildRVVCapabilityPropertyView(request.getCapabilities());
    if (!propertyView)
      return propertyView.takeError();

    if (variant->hasAttr(kRVVRequiredMarchAttrName))
      if (llvm::Error error = verifyRequiredMarchAttr(variant, *propertyView))
        return error;

    llvm::Expected<std::optional<RVVI32VAddMaterializationPlan>>
        microkernelPlan = buildI32VAddMaterializationPlan(
            variant, request.getCapabilities(), *propertyView);
    if (!microkernelPlan)
      return microkernelPlan.takeError();
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
  out.setExplanation("RVV metadata-only first slice; no runtime performance "
                     "claim");
  out.setPolicy("plugin-local RVV capability participation");
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "emission readiness requires a materialized tcrv.exec.variant");

  llvm::Expected<bool> hasMicrokernel =
      hasMatchingExplicitMicrokernel(request);
  if (!hasMicrokernel)
    return hasMicrokernel.takeError();
  if (*hasMicrokernel) {
    out = VariantEmissionStatus::getSupported(
        kRVVPluginName, request.getVariant().getSymName(),
        kMicrokernelEmissionPath);
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

  llvm::Expected<bool> hasMicrokernel =
      hasMatchingExplicitMicrokernel(request);
  if (!hasMicrokernel)
    return hasMicrokernel.takeError();
  if (*hasMicrokernel) {
    out = VariantEmissionPlan::getSupported(
        kRVVPluginName, request.getKernel().getSymName(),
        request.getVariant().getSymName(), request.getRole(),
        "rvv-explicit-i32-vadd-microkernel-c-source",
        "tcrv-export-rvv-microkernel-c",
        "rvv-i32-vadd-runtime-callable-c-abi.v1",
        "runtime-callable-c-source",
        "explicit RVV i32 vector-add microkernel C source export provides a "
        "library-style runtime-callable C ABI function for this selected "
        "path; any self-check main is an explicit harness export and is not "
        "the default artifact contract; this is not generic RVV lowering, "
        "runtime integration, arbitrary kernel emission, correctness, or "
        "performance evidence");
    out.setRuntimeABIKind("rvv-runtime-callable-c-abi");
    out.setRuntimeABIName("rvv-i32-vadd-runtime-callable-c-function.v1");
    out.setRuntimeGlueRole("runtime-callable-i32-vadd-function");
    if (llvm::Error error =
            out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
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

  llvm::Expected<std::string> capabilitySummary =
      buildRVVCapabilitySummary(variant, request.getCapabilities());
  if (!capabilitySummary)
    return capabilitySummary.takeError();

  std::optional<RVVI32VAddMaterializationPlan> microkernelPlan;
  if (variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName)) {
    llvm::Expected<RVVCapabilityPropertyView> propertyView =
        buildRVVCapabilityPropertyView(request.getCapabilities());
    if (!propertyView)
      return propertyView.takeError();

    llvm::Expected<std::optional<RVVI32VAddMaterializationPlan>> planned =
        buildI32VAddMaterializationPlan(variant, request.getCapabilities(),
                                        *propertyView);
    if (!planned)
      return planned.takeError();
    microkernelPlan = std::move(*planned);
  }

  if (microkernelPlan)
    if (llvm::Error error =
            rejectExistingRVVMicrokernelForSelectedPath(kernel, variant,
                                                        request.getRole()))
      return error;

  tcrv::rvv::LoweringBoundaryOp boundary = materializeRVVBoundaryOp(
      request.getBuilder(), kernel, variant, request.getRole(),
      *capabilitySummary);
  if (microkernelPlan)
    materializeRVVI32VAddMicrokernelOp(request.getBuilder(), kernel, variant,
                                       request.getRole(), *microkernelPlan);
  out = VariantLoweringBoundaryResult::getMaterialized(
      kRVVPluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(), boundary.getOperation());
  return llvm::Error::success();
}

} // namespace rvv

llvm::Error registerRVVExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinRVVExtensionPlugin());
}

} // namespace tianchenrv::plugin
