#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"

#include "TianChenRV/Dialect/Scalar/IR/ScalarDialect.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/I32BinaryFamilyRegistry.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin {
namespace {

constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");
constexpr llvm::StringLiteral kScalarPluginVersion("0.1.0");
constexpr llvm::StringLiteral kScalarFallbackCapabilityID("scalar.fallback");
constexpr llvm::StringLiteral kScalarFallbackCapabilityKind("fallback");
constexpr llvm::StringLiteral kScalarFallbackPreferredCapabilitySymbol(
    "scalar_fallback");
constexpr llvm::StringLiteral kScalarFallbackFirstSliceVariantName(
    "scalar_fallback_first_slice");
constexpr llvm::StringLiteral kScalarFallbackPolicy(
    "portable_scalar_fallback_first_slice");
constexpr llvm::StringLiteral kFrontendLoweringAttrName(
    "tcrv_frontend_lowering");
constexpr llvm::StringLiteral kScalarI32VAddLoweringDescriptorAttrName(
    "tcrv_scalar.lowering_descriptor");
constexpr llvm::StringLiteral kScalarI32VAddElementCountAttrName(
    "tcrv_scalar.element_count");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kFallbackReasonAttrName("fallback_reason");
constexpr llvm::StringLiteral kMetadataOnlyStatusValue("metadata-only");
constexpr llvm::StringLiteral kElementCountAttrName("element_count");
constexpr std::int64_t kDefaultI32VAddElementCount = 16;

using ScalarI32MicrokernelKind =
    tianchenrv::target::i32_binary::I32BinaryFamilyKind;
using ScalarI32FamilyDescriptor =
    tianchenrv::target::i32_binary::I32BinaryFamilyDescriptor;
using ScalarI32RegistryMicrokernelDescriptor =
    tianchenrv::target::i32_binary::ScalarI32MicrokernelFamilyDescriptor;

struct ScalarI32MicrokernelFamilySpec {
  const ScalarI32FamilyDescriptor *family;
  llvm::StringRef descriptorNoun;
  llvm::StringRef emissionPath;
  llvm::StringRef supportedMessage;

  ScalarI32MicrokernelKind getKind() const { return family->kind; }
  llvm::StringRef getFrontendLowering() const {
    return family->frontendLowering;
  }
  llvm::StringRef getLoweringDescriptor() const {
    return family->loweringDescriptor;
  }
  const ScalarI32RegistryMicrokernelDescriptor &getScalar() const {
    return family->scalar;
  }
};

const ScalarI32MicrokernelFamilySpec &getI32VAddFamilySpec() {
  static const ScalarI32MicrokernelFamilySpec spec{
      &tianchenrv::target::i32_binary::getI32VAddFamilyDescriptor(),
      "finite scalar i32-vadd lowering descriptor",
      "scalar-explicit-i32-vadd-microkernel-c-source-export",
      "explicit scalar i32 vector-add microkernel C source export is "
      "available as a library-style runtime-callable C ABI function for "
      "this selected fallback path; no self-check main is part of the "
      "default artifact contract; this is not generic scalar lowering, "
      "runtime integration, arbitrary kernel emission, correctness, or "
      "performance evidence"};
  return spec;
}

const ScalarI32MicrokernelFamilySpec &getI32VSubFamilySpec() {
  static const ScalarI32MicrokernelFamilySpec spec{
      &tianchenrv::target::i32_binary::getI32VSubFamilyDescriptor(),
      "finite scalar i32-vsub lowering descriptor",
      "scalar-explicit-i32-vsub-microkernel-c-source-export",
      "explicit scalar i32 vector-subtract microkernel C source export is "
      "available as a library-style runtime-callable C ABI function for "
      "this selected fallback path; no self-check main is part of the "
      "default artifact contract; this is not generic scalar lowering, "
      "runtime integration, arbitrary kernel emission, correctness, or "
      "performance evidence"};
  return spec;
}

const ScalarI32MicrokernelFamilySpec *
lookupI32MicrokernelFamilyByDescriptor(llvm::StringRef descriptor) {
  const ScalarI32FamilyDescriptor *family =
      tianchenrv::target::i32_binary::lookupI32BinaryFamilyByLoweringDescriptor(
          descriptor);
  if (!family)
    return nullptr;
  if (family->kind == ScalarI32MicrokernelKind::Add)
    return &getI32VAddFamilySpec();
  if (family->kind == ScalarI32MicrokernelKind::Sub)
    return &getI32VSubFamilySpec();
  return nullptr;
}

const ScalarI32MicrokernelFamilySpec *
lookupI32MicrokernelFamilyByFrontendLowering(llvm::StringRef frontendLowering) {
  const ScalarI32FamilyDescriptor *family =
      tianchenrv::target::i32_binary::lookupI32BinaryFamilyByFrontendLowering(
          frontendLowering);
  if (!family)
    return nullptr;
  if (family->kind == ScalarI32MicrokernelKind::Add)
    return &getI32VAddFamilySpec();
  if (family->kind == ScalarI32MicrokernelKind::Sub)
    return &getI32VSubFamilySpec();
  return nullptr;
}

llvm::Error makeScalarPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV scalar fallback extension plugin first slice "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasAvailableScalarFallbackCapability(
    const VariantProposalRequest &request) {
  if (!request.getKernel())
    return false;

  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kScalarFallbackCapabilityID);
  return capability && capability->isAvailable();
}

const ScalarI32MicrokernelFamilySpec *
getRequestedScalarI32Family(const VariantProposalRequest &request) {
  auto frontendLowering =
      request.getKernel()->getAttrOfType<mlir::StringAttr>(
          kFrontendLoweringAttrName);
  if (!frontendLowering)
    return &getI32VAddFamilySpec();

  llvm::StringRef value = frontendLowering.getValue().trim();
  return lookupI32MicrokernelFamilyByFrontendLowering(value);
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

bool containsForbiddenScalarMetadataText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key");
}

llvm::Error validateScalarMetadataText(llvm::StringRef context,
                                       llvm::StringRef attrName,
                                       llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.empty() || value.size() > kMaxTextLength)
    return makeScalarPluginError(llvm::Twine(context) + " attribute '" +
                                 attrName +
                                 "' must be bounded non-empty single-line "
                                 "metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return makeScalarPluginError(llvm::Twine(context) + " attribute '" +
                                   attrName +
                                   "' must be bounded non-empty single-line "
                                   "metadata");
    if (byte < 0x20 && character != '\t')
      return makeScalarPluginError(llvm::Twine(context) + " attribute '" +
                                   attrName +
                                   "' must be bounded non-empty single-line "
                                   "metadata");
  }

  if (value.contains("/*") || value.contains("*/"))
    return makeScalarPluginError(llvm::Twine(context) + " attribute '" +
                                 attrName +
                                 "' must not contain C comment delimiter text");

  if (containsForbiddenScalarMetadataText(value))
    return makeScalarPluginError(llvm::Twine(context) + " attribute '" +
                                 attrName +
                                 "' must not contain secret-like or raw "
                                 "credential text");
  return llvm::Error::success();
}

struct ScalarI32MicrokernelMaterializationPlan {
  const ScalarI32MicrokernelFamilySpec *family = nullptr;
  std::int64_t elementCount = 0;
};

llvm::Expected<std::optional<ScalarI32MicrokernelMaterializationPlan>>
buildI32MicrokernelMaterializationPlan(tcrv::exec::VariantOp variant) {
  mlir::Attribute rawDescriptor =
      variant->getAttr(kScalarI32VAddLoweringDescriptorAttrName);
  if (!rawDescriptor) {
    if (variant->hasAttr(kScalarI32VAddElementCountAttrName))
      return makeScalarPluginError(
          llvm::Twine("finite scalar i32 lowering descriptor on variant @") +
          variant.getSymName() + " requires string attribute '" +
          kScalarI32VAddLoweringDescriptorAttrName + "'");
    return std::optional<ScalarI32MicrokernelMaterializationPlan>();
  }

  auto descriptor = llvm::dyn_cast<mlir::StringAttr>(rawDescriptor);
  if (!descriptor || descriptor.getValue().trim().empty())
    return makeScalarPluginError(
        llvm::Twine("finite scalar i32 lowering descriptor on variant @") +
        variant.getSymName() + " requires string attribute '" +
        kScalarI32VAddLoweringDescriptorAttrName + "'");

  const ScalarI32MicrokernelFamilySpec *family =
      lookupI32MicrokernelFamilyByDescriptor(descriptor.getValue());
  if (!family)
    return makeScalarPluginError(
        llvm::Twine("finite scalar i32 lowering descriptor on variant @") +
        variant.getSymName() + " must be '" +
        getI32VAddFamilySpec().getLoweringDescriptor() + "' or '" +
        getI32VSubFamilySpec().getLoweringDescriptor() + "'");

  std::string descriptorContext =
      (llvm::Twine("variant @") + variant.getSymName() +
       " " + family->descriptorNoun)
          .str();
  if (llvm::Error error = validateScalarMetadataText(
          descriptorContext, kScalarI32VAddLoweringDescriptorAttrName,
          descriptor.getValue().trim()))
    return std::move(error);

  auto elementCountAttr =
      variant->getAttrOfType<mlir::IntegerAttr>(
          kScalarI32VAddElementCountAttrName);
  if (!elementCountAttr)
    return makeScalarPluginError(
        llvm::Twine(family->descriptorNoun) + " on variant @" +
        variant.getSymName() + " requires integer attribute '" +
        kScalarI32VAddElementCountAttrName + "'");

  std::int64_t elementCount = elementCountAttr.getInt();
  if (elementCount <= 0 || elementCount > 64)
    return makeScalarPluginError(
        llvm::Twine(family->descriptorNoun) + " on variant @" +
        variant.getSymName() +
        " requires tcrv_scalar.element_count in the bounded smoke range "
        "[1, 64]");

  ScalarI32MicrokernelMaterializationPlan plan;
  plan.family = family;
  plan.elementCount = elementCount;
  return plan;
}

llvm::Error validateMicrokernelEmissionAttr(mlir::Operation *op,
                                            llvm::StringRef attrName,
                                            llvm::StringRef expectedValue) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeScalarPluginError(
        llvm::Twine("explicit scalar microkernel emission plan requires "
                    "non-empty string attribute '") +
        attrName + "'");
  if (llvm::Error error =
          validateScalarMetadataText("explicit scalar microkernel emission plan",
                                     attrName, attr.getValue().trim()))
    return error;
  if (attr.getValue().trim() != expectedValue)
    return makeScalarPluginError(
        llvm::Twine("explicit scalar microkernel emission plan attribute '") +
        attrName + "' value '" + attr.getValue().trim() +
        "' does not match expected selected-path value '" + expectedValue +
        "'");
  return llvm::Error::success();
}

bool arrayAttrsEqual(mlir::ArrayAttr lhs, mlir::ArrayAttr rhs) {
  if (!lhs || !rhs || lhs.size() != rhs.size())
    return false;
  for (auto [lhsAttr, rhsAttr] : llvm::zip(lhs, rhs))
    if (lhsAttr != rhsAttr)
      return false;
  return true;
}

const ScalarI32MicrokernelFamilySpec *
getI32MicrokernelFamilyForOp(mlir::Operation *op) {
  if (llvm::isa_and_nonnull<tcrv::scalar::I32VAddMicrokernelOp>(op))
    return &getI32VAddFamilySpec();
  if (llvm::isa_and_nonnull<tcrv::scalar::I32VSubMicrokernelOp>(op))
    return &getI32VSubFamilySpec();
  return nullptr;
}

llvm::Expected<bool> variantRequiresScalarFallback(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeScalarPluginError(
        "materialized scalar fallback variant requires structured 'requires' "
        "metadata");

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeScalarPluginError(
          "materialized scalar fallback variant requires only capability "
          "symbol references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability)
      continue;

    if (capability->satisfiesID(kScalarFallbackCapabilityID))
      return true;
  }

  return false;
}

llvm::Error requireMatchingScalarBoundary(
    const VariantEmissionRequest &request, mlir::ArrayAttr variantRequires) {
  tcrv::exec::KernelOp kernel = request.getKernel();
  tcrv::exec::VariantOp variant = request.getVariant();
  llvm::StringRef expectedRole = stringifyVariantEmissionRole(request.getRole());

  unsigned matches = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<tcrv::scalar::LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto role = getStringAttr(&op, kRoleAttrName);
    if (!selectedVariant || !role)
      return makeScalarPluginError(
          "explicit scalar microkernel emission plan requires "
          "tcrv_scalar.lowering_boundary selected_variant and role metadata");

    if (selectedVariant.getValue() != variant.getSymName() ||
        role.getValue() != expectedRole) {
      return makeScalarPluginError(
          llvm::Twine("stale tcrv_scalar.lowering_boundary for @") +
          selectedVariant.getValue() + " as " + role.getValue() +
          " is not the selected scalar emission plan path @" +
          variant.getSymName() + " as " + expectedRole);
    }

    ++matches;

    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kSourceKernelAttrName,
                                            kernel.getSymName()))
      return error;
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kOriginAttrName,
                                            kScalarPluginName))
      return error;
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kRoleAttrName, expectedRole))
      return error;
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kStatusAttrName,
                                            kMetadataOnlyStatusValue))
      return error;

    if (!arrayAttrsEqual(
            boundary->getAttrOfType<mlir::ArrayAttr>(
                kRequiredCapabilitiesAttrName),
            variantRequires))
      return makeScalarPluginError(
          "explicit scalar microkernel emission plan requires "
          "tcrv_scalar.lowering_boundary required_capabilities to match "
          "selected variant requires metadata");
  }

  if (matches == 0)
    return makeScalarPluginError(
        llvm::Twine("selected scalar emission plan path @") +
        variant.getSymName() + " as " + expectedRole +
        " requires exactly one matching tcrv_scalar.lowering_boundary");
  if (matches > 1)
    return makeScalarPluginError(
        llvm::Twine("selected scalar emission plan path @") +
        variant.getSymName() + " as " + expectedRole +
        " has duplicate tcrv_scalar.lowering_boundary metadata");

  return llvm::Error::success();
}

llvm::Expected<const ScalarI32MicrokernelFamilySpec *>
findMatchingExplicitMicrokernelFamily(
    const VariantEmissionRequest &request, bool requireBoundary = true) {
  tcrv::exec::KernelOp kernel = request.getKernel();
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!kernel || !variant || kernel.getBody().empty())
    return static_cast<const ScalarI32MicrokernelFamilySpec *>(nullptr);

  llvm::StringRef expectedRole = stringifyVariantEmissionRole(request.getRole());
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  llvm::Expected<std::optional<ScalarI32MicrokernelMaterializationPlan>>
      descriptorPlan = buildI32MicrokernelMaterializationPlan(variant);
  if (!descriptorPlan)
    return descriptorPlan.takeError();

  unsigned matches = 0;
  const ScalarI32MicrokernelFamilySpec *matchedFamily = nullptr;
  for (mlir::Operation &op : kernel.getBody().front()) {
    const ScalarI32MicrokernelFamilySpec *family =
        getI32MicrokernelFamilyForOp(&op);
    if (!family)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto role = getStringAttr(&op, kRoleAttrName);
    if (!selectedVariant || !role)
      return makeScalarPluginError(
          "explicit scalar microkernel emission plan requires selected_variant "
          "and role metadata");

    if (selectedVariant.getValue() != variant.getSymName() ||
        role.getValue() != expectedRole) {
      return makeScalarPluginError(
          llvm::Twine("stale ") + family->getScalar().microkernelOpName +
          " for @" +
          selectedVariant.getValue() + " as " + role.getValue() +
          " is not the selected scalar emission plan path @" +
          variant.getSymName() + " as " + expectedRole);
    }

    if (*descriptorPlan && (*descriptorPlan)->family != family)
      return makeScalarPluginError(
          llvm::Twine("selected scalar emission plan path @") +
          variant.getSymName() + " as " + expectedRole + " has " +
          family->getScalar().microkernelOpName +
          " but selected variant descriptor requires " +
          (*descriptorPlan)->family->getScalar().microkernelOpName);

    ++matches;
    matchedFamily = family;

    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kSourceKernelAttrName,
                                            kernel.getSymName()))
      return std::move(error);
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kOriginAttrName,
                                            kScalarPluginName))
      return std::move(error);
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kRoleAttrName, expectedRole))
      return std::move(error);

    if (!variantRequires ||
        !arrayAttrsEqual(
            op.getAttrOfType<mlir::ArrayAttr>(
                kRequiredCapabilitiesAttrName),
            variantRequires)) {
      return makeScalarPluginError(
          llvm::Twine("explicit scalar microkernel emission plan requires ") +
          family->getScalar().microkernelOpName +
          " required_capabilities to match selected variant requires metadata");
    }

    auto elementCount =
        op.getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
    if (!elementCount || elementCount.getInt() <= 0 ||
        elementCount.getInt() > 64)
      return makeScalarPluginError(
          "explicit scalar microkernel emission plan requires element_count in "
          "the bounded smoke range [1, 64]");

    if (*descriptorPlan &&
        elementCount.getInt() != (*descriptorPlan)->elementCount)
      return makeScalarPluginError(
          llvm::Twine("explicit scalar microkernel emission plan requires ") +
          family->getScalar().microkernelOpName +
          " element_count to match selected variant finite descriptor "
          "tcrv_scalar.element_count");
  }

  if (matches > 1)
    return makeScalarPluginError(
        llvm::Twine("selected scalar emission plan path @") +
        variant.getSymName() + " as " + expectedRole +
        " has duplicate scalar i32 microkernel metadata");

  if (matches == 0)
    return static_cast<const ScalarI32MicrokernelFamilySpec *>(nullptr);

  if (!requireBoundary)
    return matchedFamily;

  if (llvm::Error error =
          requireMatchingScalarBoundary(request, variantRequires))
    return std::move(error);

  return matchedFamily;
}

tcrv::scalar::LoweringBoundaryOp materializeScalarBoundaryOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role) {
  builder.getContext()->getOrLoadDialect<tcrv::scalar::TCRVScalarDialect>();

  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(
      variant.getLoc(), tcrv::scalar::LoweringBoundaryOp::getOperationName());
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(kScalarPluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kMetadataOnlyStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  state.addAttribute(
      kFallbackReasonAttrName,
      builder.getStringAttr(
          "scalar fallback selected boundary is plugin-owned metadata only; "
          "no scalar executable lowering, runtime ABI, generated artifact, "
          "correctness proof, or performance measurement is produced"));
  return llvm::cast<tcrv::scalar::LoweringBoundaryOp>(builder.create(state));
}

mlir::Operation *materializeScalarI32MicrokernelOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    const ScalarI32MicrokernelMaterializationPlan &plan) {
  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  llvm::StringRef opName =
      plan.family->getKind() == ScalarI32MicrokernelKind::Sub
          ? tcrv::scalar::I32VSubMicrokernelOp::getOperationName()
          : tcrv::scalar::I32VAddMicrokernelOp::getOperationName();
  mlir::OperationState state(variant.getLoc(), opName);
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(kScalarPluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kElementCountAttrName,
                     builder.getI64IntegerAttr(plan.elementCount));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  return builder.create(state);
}

llvm::Error rejectExistingScalarBoundaryForVariant(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant) {
  if (!kernel || kernel.getBody().empty())
    return llvm::Error::success();

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<tcrv::scalar::LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto target =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    llvm::StringRef targetSymbol =
        target ? target.getValue() : llvm::StringRef("<missing>");
    if (targetSymbol != variant.getSymName())
      continue;

    return makeScalarPluginError(
        llvm::Twine("requires no pre-existing "
                    "tcrv_scalar.lowering_boundary for target @") +
        targetSymbol);
  }

  return llvm::Error::success();
}

llvm::Error rejectExistingScalarMicrokernelForSelectedPath(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role) {
  if (!kernel || kernel.getBody().empty())
    return llvm::Error::success();

  llvm::StringRef expectedRole = stringifyVariantEmissionRole(role);
  for (mlir::Operation &op : kernel.getBody().front()) {
    const ScalarI32MicrokernelFamilySpec *family =
        getI32MicrokernelFamilyForOp(&op);
    if (!family)
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

    return makeScalarPluginError(
        llvm::Twine("requires no pre-existing ") +
        family->getScalar().microkernelOpName + " for target @" +
        targetSymbol + " as " + expectedRole);
  }

  return llvm::Error::success();
}

const scalar::ScalarExtensionPlugin &getBuiltinScalarExtensionPlugin() {
  static const scalar::ScalarExtensionPlugin plugin;
  return plugin;
}

} // namespace

namespace scalar {

llvm::StringRef getScalarExtensionPluginName() { return kScalarPluginName; }

llvm::StringRef getScalarExtensionPluginVersion() {
  return kScalarPluginVersion;
}

llvm::StringRef getScalarFallbackCapabilityID() {
  return kScalarFallbackCapabilityID;
}

llvm::StringRef getScalarFallbackCapabilityKind() {
  return kScalarFallbackCapabilityKind;
}

llvm::StringRef getScalarFallbackPreferredCapabilitySymbol() {
  return kScalarFallbackPreferredCapabilitySymbol;
}

llvm::StringRef getScalarFallbackFirstSliceVariantName() {
  return kScalarFallbackFirstSliceVariantName;
}

llvm::StringRef getScalarFallbackPolicy() { return kScalarFallbackPolicy; }

ScalarExtensionPlugin::ScalarExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kScalarFallbackCapabilityID, kScalarFallbackCapabilityKind,
      "Portable scalar fallback capability for coverage-oriented execution "
      "when target profiles explicitly expose a fallback path"));
}

llvm::StringRef ScalarExtensionPlugin::getName() const {
  return kScalarPluginName;
}

llvm::StringRef ScalarExtensionPlugin::getVersion() const {
  return kScalarPluginVersion;
}

llvm::ArrayRef<PluginCapability>
ScalarExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void ScalarExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<tcrv::scalar::TCRVScalarDialect>();
}

bool ScalarExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return request.getHighLevelOp() &&
         hasAvailableScalarFallbackCapability(request) &&
         getRequestedScalarI32Family(request);
}

llvm::Error ScalarExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  const ScalarI32MicrokernelFamilySpec *family =
      getRequestedScalarI32Family(request);
  if (!family)
    return llvm::Error::success();

  VariantProposal proposal(kScalarFallbackFirstSliceVariantName,
                           kScalarPluginName);
  proposal.addRequiredCapabilityID(kScalarFallbackCapabilityID);
  proposal.setPolicy(kScalarFallbackPolicy);
  proposal.setFallbackRole(VariantFallbackRole::ConservativeFallback);
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kScalarI32VAddLoweringDescriptorAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            family->getLoweringDescriptor()));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kScalarI32VAddElementCountAttrName),
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(request.getKernel()->getContext(), 64),
          kDefaultI32VAddElementCount));
  out.push_back(proposal);
  return llvm::Error::success();
}

llvm::Error ScalarExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeScalarPluginError(
        "legality verification requires a materialized tcrv.exec.variant");

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kScalarPluginName)
    return makeScalarPluginError(
        "materialized scalar fallback variant must be owned by origin "
        "'scalar-plugin'");

  if (!request.getCapabilities().isCapabilityAvailableByID(
          kScalarFallbackCapabilityID))
    return makeScalarPluginError(
        "materialized scalar fallback variant requires an available capability "
        "id 'scalar.fallback'");

  llvm::Expected<bool> requiresScalarFallback =
      variantRequiresScalarFallback(variant, request.getCapabilities());
  if (!requiresScalarFallback)
    return requiresScalarFallback.takeError();

  if (!*requiresScalarFallback)
    return makeScalarPluginError(
        "materialized scalar fallback variant must require capability id "
        "'scalar.fallback'");

  if (variant->hasAttr(kScalarI32VAddLoweringDescriptorAttrName) ||
      variant->hasAttr(kScalarI32VAddElementCountAttrName)) {
    llvm::Expected<std::optional<ScalarI32MicrokernelMaterializationPlan>>
        microkernelPlan = buildI32MicrokernelMaterializationPlan(variant);
    if (!microkernelPlan)
      return microkernelPlan.takeError();
  }

  return llvm::Error::success();
}

llvm::Error ScalarExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeScalarPluginError(
        "cost estimation requires a materialized tcrv.exec.variant");

  out = VariantCostEstimate();
  out.setScore(1000.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kScalarPluginName);
  out.setVariantSymbol(request.getVariant().getSymName());
  out.setExplanation("portable scalar fallback first slice; coverage-oriented "
                     "metadata route, not a performance claim");
  out.setPolicy("prefer only as conservative fallback when better plugin-owned "
                "variants are unavailable or not selected");
  out.setFallbackRole(VariantFallbackRole::ConservativeFallback);
  return llvm::Error::success();
}

llvm::Error ScalarExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeScalarPluginError(
        "emission readiness requires a materialized tcrv.exec.variant");

  llvm::Expected<const ScalarI32MicrokernelFamilySpec *> microkernelFamily =
      findMatchingExplicitMicrokernelFamily(request);
  if (!microkernelFamily)
    return microkernelFamily.takeError();
  if (*microkernelFamily) {
    out = VariantEmissionStatus::getSupported(
        kScalarPluginName, request.getVariant().getSymName(),
        (*microkernelFamily)->emissionPath);
    return llvm::Error::success();
  }

  out = VariantEmissionStatus::getMetadataOnly(
      kScalarPluginName, request.getVariant().getSymName(),
      "portable-scalar-fallback-non-executable-metadata-route");
  return llvm::Error::success();
}

llvm::Error ScalarExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeScalarPluginError(
        "emission planning requires a materialized tcrv.exec.variant");

  if (!request.getKernel())
    return makeScalarPluginError(
        "emission planning requires an enclosing tcrv.exec.kernel");

  llvm::Expected<const ScalarI32MicrokernelFamilySpec *> microkernelFamily =
      findMatchingExplicitMicrokernelFamily(request);
  if (!microkernelFamily)
    return microkernelFamily.takeError();
  if (*microkernelFamily) {
    const ScalarI32MicrokernelFamilySpec &family = **microkernelFamily;
    out = VariantEmissionPlan::getSupported(
        kScalarPluginName, request.getKernel().getSymName(),
        request.getVariant().getSymName(), request.getRole(),
        family.getScalar().emissionKind, family.getScalar().routeID,
        family.getScalar().runtimeABI,
        "runtime-callable-c-source", family.supportedMessage);
    out.setRuntimeABIKind(family.getScalar().runtimeABIKind);
    out.setRuntimeABIName(family.getScalar().runtimeABIName);
    out.setRuntimeGlueRole(family.getScalar().runtimeGlueRole);
    llvm::Expected<support::I32VAddCallableABIPlan> callablePlan =
        support::buildI32VAddCallableABIPlan(request.getKernel());
    if (!callablePlan)
      return callablePlan.takeError();
    out.addRuntimeABIParameters(callablePlan->parameters);
    if (llvm::Error error =
            out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
      return error;
    return llvm::Error::success();
  }

  out = VariantEmissionPlan::getMetadataOnly(
      kScalarPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "portable-scalar-fallback-metadata-route",
      "none-executable-metadata-only", "none-metadata-only",
      "metadata-diagnostic",
      "scalar fallback first slice records a portable fallback metadata route "
      "for compiler decisions only; it does not emit objects, link a runtime, "
      "run hardware, prove correctness, or measure performance");
  out.setRuntimeABIKind("host-scalar-fallback-metadata");
  out.setRuntimeABIName("portable-scalar-fallback-metadata-abi.v1");
  out.setRuntimeGlueRole("metadata-only-host-fallback-boundary");
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error ScalarExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  if (!request.getVariant())
    return makeScalarPluginError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");

  if (!request.getKernel())
    return makeScalarPluginError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeScalarPluginError(
        llvm::Twine("selected scalar fallback variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  if (llvm::Error error = rejectExistingScalarBoundaryForVariant(
          request.getKernel(), request.getVariant()))
    return error;

  llvm::Expected<std::optional<ScalarI32MicrokernelMaterializationPlan>>
      microkernelPlan =
          buildI32MicrokernelMaterializationPlan(request.getVariant());
  if (!microkernelPlan)
    return microkernelPlan.takeError();

  bool selectedPathHasCallableMicrokernel = microkernelPlan->has_value();
  if (!selectedPathHasCallableMicrokernel) {
    VariantEmissionRequest emissionRequest(request.getVariant(),
                                           request.getKernel(),
                                           request.getCapabilities(),
                                           request.getRole());
    llvm::Expected<const ScalarI32MicrokernelFamilySpec *> explicitMicrokernel =
        findMatchingExplicitMicrokernelFamily(emissionRequest,
                                              /*requireBoundary=*/false);
    if (!explicitMicrokernel)
      return explicitMicrokernel.takeError();
    selectedPathHasCallableMicrokernel = *explicitMicrokernel != nullptr;
  }

  if (*microkernelPlan)
    if (llvm::Error error = rejectExistingScalarMicrokernelForSelectedPath(
            request.getKernel(), request.getVariant(), request.getRole()))
      return error;

  if (selectedPathHasCallableMicrokernel)
    if (llvm::Error error = support::ensureRuntimeABIBufferMemWindows(
            request.getKernel(), request.getBuilder(),
            support::getI32VAddBufferMemWindowSpecs()))
      return error;

  if (selectedPathHasCallableMicrokernel) {
    llvm::SmallVector<support::RuntimeABIParamSpec, 1> runtimeParamSpecs;
    auto countSpecs = support::getI32VAddRuntimeElementCountParamSpecs();
    runtimeParamSpecs.append(countSpecs.begin(), countSpecs.end());
    if (llvm::Error error =
            support::ensureRuntimeABIParamsAllowingExistingCNames(
                request.getKernel(), request.getBuilder(), runtimeParamSpecs))
      return error;
  }

  tcrv::scalar::LoweringBoundaryOp boundary = materializeScalarBoundaryOp(
      request.getBuilder(), request.getKernel(), request.getVariant(),
      request.getRole());
  if (*microkernelPlan)
    materializeScalarI32MicrokernelOp(
        request.getBuilder(), request.getKernel(), request.getVariant(),
        request.getRole(), **microkernelPlan);

  out = VariantLoweringBoundaryResult::getMaterialized(
      kScalarPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      boundary.getOperation());
  return llvm::Error::success();
}

} // namespace scalar

llvm::Error registerScalarExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinScalarExtensionPlugin());
}

} // namespace tianchenrv::plugin
