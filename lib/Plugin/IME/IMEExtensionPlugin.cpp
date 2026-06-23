#include "TianChenRV/Plugin/IME/IMEExtensionPlugin.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/IME/IR/IMEDialect.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Errc.h"

#include <string>

namespace tianchenrv::plugin {
namespace {

constexpr llvm::StringLiteral kIMEPluginName("ime-plugin");
constexpr llvm::StringLiteral kIMEPluginVersion("0.1.0");
// The first-class derived capability id. NOT a family-name string match: the
// plugin gates on the PRESENCE of this capability FACT in the target set; a
// target without it (RVV-only) does not satisfy lookupProviderByID and IME
// declines, so dispatch is capability-driven (I1, I3).
constexpr llvm::StringLiteral kIMECapabilityID("spacemit.ime");
constexpr llvm::StringLiteral kIMECapabilityKind("isa-matrix-vector-backed");
constexpr llvm::StringLiteral kIMEFirstSliceVariantName("ime_vmadot_mma_slice");
// The SECOND (unsigned) IME variant. Same capability FACT, different signedness
// fact => different emitted instruction (`vmadotu`) and boundary op
// (tcrv.ime.mma_u). This is the N2 plugin-breadth surface.
constexpr llvm::StringLiteral kIMEUnsignedVariantName(
    "ime_vmadotu_mma_slice");
// The TILED whole-matrix variants (signed/unsigned). Same capability FACT +
// dispatch, a richer problem shape => the tcrv.ime.matmul boundary op.
constexpr llvm::StringLiteral kIMEMatmulVariantName("ime_vmadot_matmul_slice");
constexpr llvm::StringLiteral kIMEMatmulUVariantName(
    "ime_vmadotu_matmul_slice");

constexpr llvm::StringLiteral kIMEPolicy("ime_int8_matmul_vmadot_mac");
constexpr llvm::StringLiteral kIMECondition("spacemit_ime_capability_available");
constexpr llvm::StringLiteral kIMEGuard("plugin_local_ime_vmadot_boundary");

constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kIMEOpAttrName("ime_op");
constexpr llvm::StringLiteral kElemInBitsAttrName("elem_in_bits");
constexpr llvm::StringLiteral kAccumBitsAttrName("accum_bits");
constexpr llvm::StringLiteral kMacMAttrName("mac_m");
constexpr llvm::StringLiteral kMacNAttrName("mac_n");
constexpr llvm::StringLiteral kMacKAttrName("mac_k");
constexpr llvm::StringLiteral kMatMAttrName("mat_m");
constexpr llvm::StringLiteral kMatNAttrName("mat_n");
constexpr llvm::StringLiteral kMatKAttrName("mat_k");
constexpr llvm::StringLiteral kAvailableHartsAttrName("available_harts");
constexpr llvm::StringLiteral kIMEReasonAttrName("ime_reason");

constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");

// Capability-property names parsed from the in-IR tcrv.exec.capability provider.
constexpr llvm::StringLiteral kMarchPropertyName("march");
constexpr llvm::StringLiteral kVlenBitsPropertyName("vlen_bits");
constexpr llvm::StringLiteral kAvailableHartsPropertyName("available_harts");
// Which signedness form of the IME1 MAC the kernel requests. This is a
// capability-derived FACT, NOT a family-name string: the `xsmtvdotii` envelope
// provides BOTH signed and unsigned MAC (the SpacemiT GCC15 assembler accepts
// vmadot/vmadotu/vmadotsu/vmadotus under the SAME march token), so the plugin
// derives the available signedness set from the march fact and picks the
// requested form. Absent property => "signed" (back-compat with the first
// slice). An UNSUPPORTED signedness fails closed.
constexpr llvm::StringLiteral kSignednessPropertyName("ime_signedness");
constexpr llvm::StringLiteral kSignednessSigned("signed");
constexpr llvm::StringLiteral kSignednessUnsigned("unsigned");
// The optional WHOLE-MATRIX problem-shape FACT (format "MxNxK"). When the
// capability carries it, the kernel requests the TILED matmul boundary
// (tcrv.ime.matmul) over that problem; absent => the single 4x4x8 MAC fragment
// boundary (tcrv.ime.mma, back-compat with the first slice). This is a SHAPE
// fact, NOT a family-name string: the same capability id + dispatch, a richer
// problem. Dims must each be a whole multiple of the derived MAC fragment.
constexpr llvm::StringLiteral kMatmulShapePropertyName("ime_matmul_shape");

// The load-bearing IME1 march token (FOUNDATION task 2: absent => assembler
// rejects `vmadot`). The capability is the proven IME1 envelope ONLY when this
// token is present. It assembles BOTH signedness forms below.
constexpr llvm::StringLiteral kIMEMarchToken("xsmtvdotii");
constexpr llvm::StringLiteral kIMEOpValue("vmadot");
constexpr llvm::StringLiteral kIMEUnsignedOpValue("vmadotu");
// The plugin-owned (dialect-qualified, discardable) variant attribute that
// records the derived signedness fact on the materialized tcrv.exec.variant so
// the boundary materializer routes to the matching boundary op (mma vs mma_u)
// WITHOUT re-deriving — and so selection stays a pure data flow of the
// capability fact.
constexpr llvm::StringLiteral kSignednessVariantAttrName("ime.signedness");
constexpr int64_t kIMEElemInBits = 8;
constexpr int64_t kIMEAccumBits = 32;
// Default per-hart availability for the X60 (harts 0-3 carry _ime; hart 4 does
// not). Used only when the capability provider omits the property.
constexpr llvm::StringLiteral kIMEDefaultAvailableHarts("0-3");

llvm::Error makeIMEPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV IME extension plugin failed: ") + message,
      llvm::errc::invalid_argument);
}

/// The DERIVED IME matmul-capability facts. Modeled like RVVCapabilityProfile:
/// derived from validated ISA evidence (march `xsmtvdotii` + VLEN/SEW), NOT
/// hand-stuck metadata. A different VLEN derives a different MAC fragment shape
/// (the N1 hook): at VLEN=256/SEW=8 the X60 unit is 4x4x8.
struct IMEMatmulCapability {
  std::string imeOp;
  // The signedness form this capability derived to ("signed" => vmadot,
  // "unsigned" => vmadotu). Both are facts of the same `xsmtvdotii` envelope.
  std::string signedness;
  int64_t elemInBits = 0;
  int64_t accumBits = 0;
  int64_t macM = 0;
  int64_t macN = 0;
  int64_t macK = 0;
  // Whole-matrix problem dims (>0 only when the kernel requests the tiled
  // matmul boundary via the ime_matmul_shape fact). 0 => single MAC fragment.
  int64_t matM = 0;
  int64_t matN = 0;
  int64_t matK = 0;
  bool isMatmul = false;
  std::string availableHarts;
};

/// Derives the IME matmul capability from a `spacemit.ime` capability descriptor.
/// Returns an error (fail-closed) if the descriptor is not the validated IME1
/// envelope (march token absent, or VLEN not a supported MAC fragment shape).
llvm::Expected<IMEMatmulCapability>
deriveIMEMatmulCapability(const support::CapabilityDescriptor &capability) {
  llvm::StringRef march = capability.getProperty(kMarchPropertyName);
  if (!march.contains(kIMEMarchToken))
    return makeIMEPluginError(
        llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
        kMarchPropertyName + "' must contain the load-bearing IME1 march token '" +
        kIMEMarchToken + "' (required to assemble `vmadot`)");

  // VLEN -> MAC fragment shape. At VLEN=256/SEW=8 the validated X60 unit is
  // 4x4x8 (FOUNDATION task 3: vlenb==32 && vl(e8,m1)==32). A different VLEN
  // would derive a different fragment shape — this is the N1 capability hook.
  llvm::StringRef vlenBitsText = capability.getProperty(kVlenBitsPropertyName);
  long long vlenBits = 0;
  if (vlenBitsText.empty() || vlenBitsText.getAsInteger(10, vlenBits))
    return makeIMEPluginError(
        llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
        kVlenBitsPropertyName +
        "' must be the integer VLEN in bits (derives the MAC fragment shape)");

  // Derive the requested signedness FACT. The `xsmtvdotii` envelope provides
  // BOTH signed (vmadot) and unsigned (vmadotu) MAC; which one this kernel wants
  // is carried as a capability property. Absent => signed (back-compat). Any
  // other value fails closed (we only validated vmadot/vmadotu on real K1).
  llvm::StringRef requestedSignedness =
      capability.getProperty(kSignednessPropertyName).trim();
  if (requestedSignedness.empty())
    requestedSignedness = kSignednessSigned;

  IMEMatmulCapability derived;
  if (requestedSignedness == kSignednessSigned) {
    derived.imeOp = kIMEOpValue.str();
    derived.signedness = kSignednessSigned.str();
  } else if (requestedSignedness == kSignednessUnsigned) {
    derived.imeOp = kIMEUnsignedOpValue.str();
    derived.signedness = kSignednessUnsigned.str();
  } else {
    return makeIMEPluginError(
        llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
        kSignednessPropertyName + "' = '" + requestedSignedness +
        "' is outside the validated IME1 signedness envelope (only 'signed' "
        "=> vmadot and 'unsigned' => vmadotu are real-K1-validated)");
  }
  derived.elemInBits = kIMEElemInBits;
  derived.accumBits = kIMEAccumBits;
  if (vlenBits == 256) {
    derived.macM = 4;
    derived.macN = 4;
    derived.macK = 8;
  } else {
    return makeIMEPluginError(
        llvm::Twine("capability id '") + kIMECapabilityID +
        "' VLEN=" + llvm::Twine(vlenBits) +
        " is outside the validated IME1 MAC-fragment envelope (VLEN=256 / "
        "4x4x8 is the only real-K1-validated shape)");
  }

  // Optional WHOLE-MATRIX problem shape FACT. Absent => single MAC fragment
  // boundary (back-compat). Present => tiled matmul boundary over MxNxK; each
  // dim must be a whole multiple of the derived MAC fragment (fail-closed: no
  // remainder path). Format "MxNxK" (e.g. "256x256x256").
  llvm::StringRef matmulShape =
      capability.getProperty(kMatmulShapePropertyName).trim();
  if (!matmulShape.empty()) {
    llvm::SmallVector<llvm::StringRef, 3> parts;
    matmulShape.split(parts, 'x');
    long long m = 0, n = 0, k = 0;
    if (parts.size() != 3 || parts[0].getAsInteger(10, m) ||
        parts[1].getAsInteger(10, n) || parts[2].getAsInteger(10, k))
      return makeIMEPluginError(
          llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
          kMatmulShapePropertyName +
          "' must be the whole-matrix problem shape 'MxNxK' (integers)");
    if (m <= 0 || n <= 0 || k <= 0)
      return makeIMEPluginError(
          llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
          kMatmulShapePropertyName + "' dims must be positive");
    if (m % derived.macM != 0 || n % derived.macN != 0 || k % derived.macK != 0)
      return makeIMEPluginError(
          llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
          kMatmulShapePropertyName + "' = '" + matmulShape +
          "' must be a whole multiple of the derived MAC fragment (" +
          llvm::Twine(derived.macM) + "x" + llvm::Twine(derived.macN) + "x" +
          llvm::Twine(derived.macK) + "); no remainder path is emitted");
    derived.matM = m;
    derived.matN = n;
    derived.matK = k;
    derived.isMatmul = true;
  }

  llvm::StringRef harts = capability.getProperty(kAvailableHartsPropertyName);
  derived.availableHarts =
      harts.trim().empty() ? kIMEDefaultAvailableHarts.str() : harts.trim().str();
  return derived;
}

bool hasAvailableIMECapability(const VariantProposalRequest &request) {
  if (!request.getKernel())
    return false;
  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kIMECapabilityID);
  return capability && capability->isAvailable();
}

/// Reads the derived signedness fact stamped on the materialized variant (the
/// `ime.signedness` plugin-owned attribute). Absent => signed (the first slice).
/// This is a pure read-back of the capability-derived fact, so downstream
/// boundary/emission selection never re-classifies a family name.
bool variantRequestsUnsignedIME(tcrv::exec::VariantOp variant) {
  if (!variant)
    return false;
  auto signedness =
      variant->getAttrOfType<mlir::StringAttr>(kSignednessVariantAttrName);
  return signedness && signedness.getValue() == kSignednessUnsigned;
}

llvm::Expected<VariantProposal>
buildIMEProposal(const VariantProposalRequest &request) {
  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kIMECapabilityID);
  if (!capability || !capability->isAvailable())
    return makeIMEPluginError(
        llvm::Twine("IME proposal requires an available capability provider "
                    "for id '") +
        kIMECapabilityID + "'");
  if (capability->getKind() != kIMECapabilityKind)
    return makeIMEPluginError(
        llvm::Twine("capability id '") + kIMECapabilityID + "' kind must be '" +
        kIMECapabilityKind + "'");

  // Fail-closed: only propose when the facts derive to the validated IME1
  // envelope.
  llvm::Expected<IMEMatmulCapability> derived =
      deriveIMEMatmulCapability(*capability);
  if (!derived)
    return derived.takeError();

  // The variant name + the recorded signedness FACT both follow the derived
  // capability fact (NOT a family-name string). Two orthogonal facts pick the
  // boundary op: the SHAPE fact (single fragment => tcrv.ime.mma[_u]; whole
  // matrix => tcrv.ime.matmul) and the SIGNEDNESS fact (vmadot vs vmadotu).
  bool isUnsigned = derived->signedness == kSignednessUnsigned;
  llvm::StringRef variantName;
  if (derived->isMatmul)
    variantName =
        isUnsigned ? kIMEMatmulUVariantName : kIMEMatmulVariantName;
  else
    variantName =
        isUnsigned ? kIMEUnsignedVariantName : kIMEFirstSliceVariantName;
  VariantProposal proposal(variantName, kIMEPluginName);
  proposal.addRequiredCapabilityID(kIMECapabilityID);
  proposal.setCondition(kIMECondition);
  proposal.setGuard(kIMEGuard);
  proposal.setPolicy(kIMEPolicy);
  mlir::MLIRContext *context = request.getKernel().getContext();
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context, kSignednessVariantAttrName),
      mlir::StringAttr::get(context, derived->signedness));
  return proposal;
}

std::string sanitizeIMEDeclineReason(llvm::StringRef reason) {
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

const ime::IMEExtensionPlugin &getBuiltinIMEExtensionPlugin() {
  static const ime::IMEExtensionPlugin plugin;
  return plugin;
}

} // namespace

namespace ime {

llvm::StringRef getIMEExtensionPluginName() { return kIMEPluginName; }
llvm::StringRef getIMEExtensionPluginVersion() { return kIMEPluginVersion; }
llvm::StringRef getIMEExtensionCapabilityID() { return kIMECapabilityID; }
llvm::StringRef getIMEExtensionCapabilityKind() { return kIMECapabilityKind; }
llvm::StringRef getIMEExtensionFirstSliceVariantName() {
  return kIMEFirstSliceVariantName;
}

IMEExtensionPlugin::IMEExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kIMECapabilityID, kIMECapabilityKind,
      "Spacemit X60 Integer Matrix Extension (IME1): int8->int32 vmadot MAC, "
      "vector-register-backed; second-family (non-RVV) RISC-V capability fact"));
}

llvm::StringRef IMEExtensionPlugin::getName() const { return kIMEPluginName; }

llvm::StringRef IMEExtensionPlugin::getVersion() const {
  return kIMEPluginVersion;
}

llvm::ArrayRef<PluginCapability> IMEExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void IMEExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<tcrv::ime::TCRVIMEDialect>();
}

bool IMEExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return request.getHighLevelOp() && hasAvailableIMECapability(request);
}

llvm::Error IMEExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildIMEProposal(request);
  if (!proposal) {
    llvm::consumeError(proposal.takeError());
    return llvm::Error::success();
  }
  out.push_back(*proposal);
  return llvm::Error::success();
}

llvm::Error IMEExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildIMEProposal(request);
  if (!proposal) {
    std::string reason =
        sanitizeIMEDeclineReason(llvm::toString(proposal.takeError()));
    out.addRecoverableDecline(kIMEPluginName, reason);
    return llvm::Error::success();
  }
  out.addProposal(*proposal);
  return llvm::Error::success();
}

llvm::Error IMEExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeIMEPluginError(
        "legality verification requires a materialized tcrv.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kIMEPluginName)
    return makeIMEPluginError(
        "materialized IME variant must be owned by origin 'ime-plugin'");

  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kIMECapabilityID);
  if (!capability || !capability->isAvailable())
    return makeIMEPluginError(
        "materialized IME variant requires an available capability id "
        "'spacemit.ime'");

  // Plugin-owned legality: the variant requires the IME capability and the
  // capability derives to the validated IME1 envelope.
  auto requiresAttr = variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  bool requiresIME = false;
  if (requiresAttr) {
    for (mlir::Attribute requiredCapability : requiresAttr) {
      auto symbolRef =
          llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
      if (!symbolRef)
        continue;
      const support::CapabilityDescriptor *required =
          request.getCapabilities().lookupBySymbolName(symbolRef.getValue());
      if (required && required->satisfiesID(kIMECapabilityID)) {
        requiresIME = true;
        break;
      }
    }
  }
  if (!requiresIME)
    return makeIMEPluginError(
        "materialized IME variant must require capability id 'spacemit.ime'");

  if (llvm::Error error = deriveIMEMatmulCapability(*capability).takeError())
    return error;
  return llvm::Error::success();
}

llvm::Error IMEExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeIMEPluginError(
        "cost estimation requires a materialized tcrv.exec.variant");

  out = VariantCostEstimate();
  out.setScore(20.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kIMEPluginName);
  out.setVariantSymbol(request.getVariant().getSymName());
  out.setExplanation(
      "IME int8->int32 vmadot MAC boundary; lowers to the FOUNDATION-validated "
      "vmadot kernel through the common EmitC route");
  out.setPolicy(
      "prefer IME only when the spacemit.ime capability fact is available and "
      "derives to the validated IME1 int8->int32 MAC envelope");
  return llvm::Error::success();
}

llvm::Error IMEExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeIMEPluginError(
        "emission readiness requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeIMEPluginError(
        "emission readiness requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    out = VariantEmissionStatus::getUnsupported(
        kIMEPluginName, request.getVariant().getSymName(), message);
    return llvm::Error::success();
  }

  out = VariantEmissionStatus::getSupported(
      kIMEPluginName, request.getVariant().getSymName(),
      "ime-vmadot-mma-emitc-route");
  return llvm::Error::success();
}

llvm::Error IMEExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeIMEPluginError(
        "emission planning requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeIMEPluginError(
        "emission planning requires an enclosing tcrv.exec.kernel");

  VariantLegalityRequest legality(request.getVariant(), request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeIMEPluginError(
        llvm::Twine("selected IME variant @") +
        request.getVariant().getSymName() +
        " failed plugin legality before emission planning: " + message);
  }

  // Re-derive to know whether this is the single-fragment or the tiled-matmul
  // boundary (a shape fact, not a family-name branch).
  const support::CapabilityDescriptor *planCapability =
      request.getCapabilities().lookupProviderByID(kIMECapabilityID);
  bool isMatmul = false;
  if (planCapability) {
    if (llvm::Expected<IMEMatmulCapability> planDerived =
            deriveIMEMatmulCapability(*planCapability))
      isMatmul = planDerived->isMatmul;
    else
      llvm::consumeError(planDerived.takeError());
  }
  bool isUnsigned = variantRequestsUnsignedIME(request.getVariant());
  llvm::StringRef boundaryOpName =
      isMatmul ? tcrv::ime::MatMulOp::getOperationName()
               : (isUnsigned ? tcrv::ime::MMAUOp::getOperationName()
                             : tcrv::ime::MMAOp::getOperationName());
  out = VariantEmissionPlan::getSupported(
      kIMEPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "materialized-emitc-cpp-ime-vmadot-mma-module",
      "ime-vmadot-mma-emitc-route",
      "ime-vmadot-mma-runtime-c-abi.v1", "riscv-elf-relocatable-object",
      "IME selected boundary lowers the FOUNDATION-validated int8->int32 MAC "
      "kernel (signed vmadot or unsigned vmadotu) through the common "
      "TCRVEmitCLowerableRoute materializer and the MLIR EmitC C/C++ emitter");
  out.setRuntimeABIKind("plugin-owned-runtime-abi");
  out.setRuntimeABIName("ime-vmadot-mma-runtime-c-abi.v1");
  out.setRuntimeGlueRole("emitc-cpp-ime-vmadot-mma-runtime-glue");
  out.setLoweringBoundaryOpName(boundaryOpName);
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error IMEExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeIMEPluginError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeIMEPluginError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  VariantLegalityRequest legality(variant, kernel, request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeIMEPluginError(
        llvm::Twine("selected IME variant @") + variant.getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kIMECapabilityID);
  // verifyVariantLegality already guaranteed availability + derivation.
  llvm::Expected<IMEMatmulCapability> derived =
      deriveIMEMatmulCapability(*capability);
  if (!derived)
    return derived.takeError();

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::MLIRContext *context = builder.getContext();
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  // Route to the boundary op matching the DERIVED signedness fact: signed =>
  // tcrv.ime.mma (vmadot), unsigned => tcrv.ime.mma_u (vmadotu). Both are
  // ODS-registered ops of the SAME dialect; the choice is a pure data flow of
  // the capability-derived fact (recorded on the variant by the materializer),
  // NOT a family-name branch. (The architectural family spelling `tcrv.ime.*`
  // is not a valid dialect namespace, so the registered names are tcrv_ime.*.)
  // Two orthogonal capability-derived FACTS pick the boundary op: the SHAPE
  // fact (single fragment => tcrv.ime.mma[_u]; whole matrix => tcrv.ime.matmul)
  // and the SIGNEDNESS fact (vmadot vs vmadotu). Neither is a family-name
  // branch; both are pure data flow of the derived capability facts.
  bool boundaryIsUnsigned = derived->signedness == kSignednessUnsigned;
  llvm::StringRef boundaryOpName =
      derived->isMatmul
          ? tcrv::ime::MatMulOp::getOperationName()
          : (boundaryIsUnsigned ? tcrv::ime::MMAUOp::getOperationName()
                                : tcrv::ime::MMAOp::getOperationName());
  mlir::OperationState state(variant.getLoc(), boundaryOpName);
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(
      kSelectedVariantAttrName,
      mlir::FlatSymbolRefAttr::get(context, variant.getSymName()));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(kIMEPluginName));
  state.addAttribute(
      kRoleAttrName,
      builder.getStringAttr(stringifyVariantEmissionRole(request.getRole())));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kRoleOpBoundaryStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, variantRequires);
  state.addAttribute(kIMEOpAttrName, builder.getStringAttr(derived->imeOp));
  state.addAttribute(kElemInBitsAttrName,
                     builder.getI64IntegerAttr(derived->elemInBits));
  state.addAttribute(kAccumBitsAttrName,
                     builder.getI64IntegerAttr(derived->accumBits));
  state.addAttribute(kMacMAttrName, builder.getI64IntegerAttr(derived->macM));
  state.addAttribute(kMacNAttrName, builder.getI64IntegerAttr(derived->macN));
  state.addAttribute(kMacKAttrName, builder.getI64IntegerAttr(derived->macK));
  if (derived->isMatmul) {
    state.addAttribute(kMatMAttrName,
                       builder.getI64IntegerAttr(derived->matM));
    state.addAttribute(kMatNAttrName,
                       builder.getI64IntegerAttr(derived->matN));
    state.addAttribute(kMatKAttrName,
                       builder.getI64IntegerAttr(derived->matK));
  }
  state.addAttribute(kAvailableHartsAttrName,
                     builder.getStringAttr(derived->availableHarts));
  state.addAttribute(
      kIMEReasonAttrName,
      builder.getStringAttr(
          (derived->isMatmul
               ? llvm::Twine("capability-derived IME1 int8->int32 ") +
                     derived->imeOp + " tiled matmul (" +
                     llvm::Twine(derived->matM) + "x" +
                     llvm::Twine(derived->matN) + "x" +
                     llvm::Twine(derived->matK) +
                     ") over the MAC fragment from march xsmtvdotii + VLEN/SEW "
                     "(signedness=" +
                     derived->signedness + ")"
               : llvm::Twine("capability-derived IME1 int8->int32 ") +
                     derived->imeOp +
                     " MAC fragment from march xsmtvdotii + VLEN/SEW "
                     "(signedness=" +
                     derived->signedness + ")")));
  mlir::Operation *boundary = builder.create(state);

  VariantLoweringBoundaryValidationRequest validationRequest(
      variant, kernel, request.getCapabilities(), request.getRole(), boundary);
  if (llvm::Error error = validateSelectedLoweringBoundary(validationRequest))
    return error;

  out = VariantLoweringBoundaryResult::getMaterialized(
      kIMEPluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(), boundary);
  return llvm::Error::success();
}

llvm::Error IMEExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  // The boundary is the signed tcrv.ime.mma, the unsigned tcrv.ime.mma_u, or
  // the tiled tcrv.ime.matmul op — all are the IME plugin execution surface.
  // Accept any; each op's fail-closed verifier pins its own correct mnemonic
  // and envelope.
  mlir::Operation *boundaryOp = request.getBoundary();
  bool verifierFailed = false;
  if (auto mma = llvm::dyn_cast_if_present<tcrv::ime::MMAOp>(boundaryOp))
    verifierFailed = mlir::failed(mma.verify());
  else if (auto mmau = llvm::dyn_cast_if_present<tcrv::ime::MMAUOp>(boundaryOp))
    verifierFailed = mlir::failed(mmau.verify());
  else if (auto matmul =
               llvm::dyn_cast_if_present<tcrv::ime::MatMulOp>(boundaryOp))
    verifierFailed = mlir::failed(matmul.verify());
  else
    return makeIMEPluginError(
        "selected IME path requires a tcrv.ime.mma, tcrv.ime.mma_u, or "
        "tcrv.ime.matmul operation");

  // The ODS verifier (fail-closed, I7) already enforces the int8->int32 MAC
  // envelope of the op's signedness, origin/role/status, and selected-path
  // binding. Re-run it so a boundary the verifier would reject is never
  // reported materialized.
  if (verifierFailed)
    return makeIMEPluginError(
        "materialized IME boundary failed its fail-closed verifier");

  auto origin = boundaryOp->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!origin || origin.getValue() != kIMEPluginName)
    return makeIMEPluginError("IME boundary origin must be 'ime-plugin'");

  auto selectedVariant =
      boundaryOp->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kSelectedVariantAttrName);
  if (!selectedVariant ||
      selectedVariant.getValue() != request.getVariant().getSymName())
    return makeIMEPluginError(
        "IME boundary selected_variant must match the selected variant");
  return llvm::Error::success();
}

} // namespace ime

llvm::Error registerIMEExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinIMEExtensionPlugin());
}

} // namespace tianchenrv::plugin
