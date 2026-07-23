#include "Weft/Plugin/IME/IMEExtensionPlugin.h"

#include "Weft/Plugin/IME/IMEFormulaConstruction.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/IME/IR/IMEDialect.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Errc.h"

#include <string>
#include <utility>

namespace weft::plugin {
namespace {

constexpr llvm::StringLiteral kIMEPluginName("ime-plugin");
constexpr llvm::StringLiteral kIMEPluginVersion("0.1.0");
constexpr llvm::StringLiteral kIMECostFormulaID(
    "weft.ime.matmul.analytic-prior");
// The first-class derived capability id. NOT a family-name string match: the
// plugin gates on the PRESENCE of this capability FACT in the target set; a
// target without it (RVV-only) does not satisfy lookupProviderByID and IME
// declines, so dispatch is capability-driven (I1, I3).
constexpr llvm::StringLiteral kIMECapabilityID("spacemit.ime");
constexpr llvm::StringLiteral kIMECapabilityKind("isa-matrix-vector-backed");
constexpr llvm::StringLiteral kIMEFirstSliceVariantName("ime_vmadot_mma_slice");
// The SECOND (unsigned) IME variant. Same capability FACT, different signedness
// fact => different emitted instruction (`vmadotu`) and boundary op
// (weft.ime.mma_u). This is the N2 plugin-breadth surface.
constexpr llvm::StringLiteral kIMEUnsignedVariantName(
    "ime_vmadotu_mma_slice");
// The FOURTH (mixed-sign) IME variant. Same capability FACT, a different
// signedness fact => a different emitted instruction (`vmadotsu`, signed A *
// unsigned B) and boundary op (weft.ime.mma_su). This is the N2 rapid-add
// plugin-breadth surface (the canonical quantized mixed-sign case).
constexpr llvm::StringLiteral kIMEMixedSignVariantName(
    "ime_vmadotsu_mma_slice");
// The SIXTH (reversed-order mixed-sign) IME variant. Same capability FACT, a
// different signedness fact => a different emitted instruction (`vmadotus`,
// unsigned A * signed B) and boundary op (weft.ime.mma_us). This is the N2
// rapid-add plugin-breadth surface that COMPLETES the signedness family.
constexpr llvm::StringLiteral kIMEMixedSignUSVariantName(
    "ime_vmadotus_mma_slice");
// The FIFTH (sliding-window) IME variant. Same target capability, a different
// canonical sliding-MAC problem => a different emitted instruction
// (`vmadot1`/`vmadot2`/`vmadot3`, funct7 111001) and boundary op
// (weft.ime.mma_slide). This is the N2 rapid-add plugin-breadth surface for the
// Xsmti8i32mm_slide conv/strided-A-reuse primitive (K1's 2nd IME1 sub-extension).
constexpr llvm::StringLiteral kIMESlideVariantName("ime_vmadot1_mma_slide_slice");
// The TILED whole-matrix variants (signed/unsigned). Same capability FACT +
// dispatch, a richer problem shape => the weft.ime.matmul boundary op.
//
// G4 M1a: the FORMAT-KEYED q4_0 whole-matrix tile rides the SAME signed matmul
// variant NAME (ime_vmadot_matmul_slice) the whole-matrix GEMM prior routes to
// (P7 = ime ^ shape: one matrix variant takes over the contraction). The q4_0
// canonical block-quant problem identity keys WHICH boundary op materializes
// within that variant -- the format-agnostic
// weft.ime.matmul vs the typed-region weft.ime.q4_0_matmul_tile -- exactly like
// the signedness/slide facts key the mma siblings. So the prior still routes
// q4_0 GEMM ^ ime -> @ime_vmadot_matmul_slice; the fact selects the tile.
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
constexpr llvm::StringLiteral kSlideAttrName("slide");
constexpr llvm::StringLiteral kAvailableHartsAttrName("available_harts");
constexpr llvm::StringLiteral kIMEReasonAttrName("ime_reason");
// Format-keyed tile op attributes and decomposed brick facts. These are typed
// final-body fields projected from the canonical problem, not target capability
// properties.
constexpr llvm::StringLiteral kWeightFormatAttrName("weight_format");
constexpr llvm::StringLiteral kQkAttrName("qk");
constexpr llvm::StringLiteral kWeightBlockStrideAttrName("weight_block_stride");
constexpr llvm::StringLiteral kWeightQuantByteOffsetAttrName(
    "weight_quant_byte_offset");
// Canonical problem semantics select the decode/fold mechanisms. Fragment and
// accumulator lane counts are derived from the target-projected MAC envelope.
constexpr llvm::StringLiteral kQ40DecodeModel("q4_0_offset_binary_nibble");
constexpr llvm::StringLiteral kQ80DecodeModel("q8_0_direct_int8");
constexpr llvm::StringLiteral kQ4KDecodeModel("q4_K_raw_nibble");
constexpr llvm::StringLiteral kQ4KScaleMinModel("get_scale_min_k4");
constexpr llvm::StringLiteral kQ4KScaleWeightedModel("scale_weighted_sum");
constexpr llvm::StringLiteral kQ4KMinBiasModel("activation_sum_min_bias");

constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");

// Target capability properties. Source-problem facts are forbidden on the
// provider and rejected below even when their value is empty.
constexpr llvm::StringLiteral kMarchPropertyName("march");
constexpr llvm::StringLiteral kVlenBitsPropertyName("vlen_bits");
constexpr llvm::StringLiteral kAvailableHartsPropertyName("available_harts");
constexpr llvm::StringLiteral kForbiddenProblemCapabilityProperties[] = {
    "ime_signedness", "ime_matmul_shape", "ime_weight_format", "ime_slide"};
constexpr llvm::StringLiteral kForbiddenVariantProblemMirrors[] = {
    "ime.signedness", "ime.slide", "ime.weight_format"};

constexpr llvm::StringLiteral kSignednessSigned("signed");
constexpr llvm::StringLiteral kSignednessUnsigned("unsigned");
constexpr llvm::StringLiteral kSignednessMixedSign("signed_unsigned");
constexpr llvm::StringLiteral kSignednessMixedSignUS("unsigned_signed");
constexpr llvm::StringLiteral kWeightFormatQ40("q4_0");
constexpr llvm::StringLiteral kWeightFormatQ80("q8_0");
constexpr llvm::StringLiteral kWeightFormatQ4K("q4_K");

// P/c_o-conditioned cross-paradigm ranking costs (SEL-1 exec-level analytic
// prior). These are not blind literals: exact P distinguishes whole-matrix from
// fragment work, while projected c_o provides the MAC crossover. The
// registry ranks ascending (lower = preferred), so:
//   * the whole-matrix GEMM cost sits BELOW the RVV vector-paradigm base (1.0):
//     when the matrix-shape fact derives, the systolic MAC paradigm WINS the
//     contraction because the capability fact says so (P7 pattern: ime ∧ shape),
//     not because a constant happened to sort first;
//   * the single-fragment MAC cost sits ABOVE the vector base but below the
//     scalar fallback (1000.0): a leaf MAC boundary beats scalar yet never
//     displaces a full vectorized kernel.
// The RVV vector base is defined symmetrically in the RVV plugin.
//
// T5c M-AWARE refinement: the whole-matrix GEMM preference is CONDITIONAL on the
// problem M dimension (rows / tokens) reaching the capability-derived crossover
// M* = macM (the MAC fragment's row dimension). Below M* the systolic array's
// rows are underfed (matrix-VECTOR / decode — a MEMORY-BOUND roofline regime);
// at/above M* the array is fully fed (matrix-matrix / prefill — COMPUTE-BOUND).
// The T5b K1-silicon paradigm-lever sweep CONFIRMS the matrix advantage saturates
// exactly at M >= macM (=4 @ VLEN256). So the preferred cost fires only in the
// compute-bound regime; a below-M* boundary is recorded at roofline PARITY with
// the RVV vector base (do NOT let the compute-isolated micro-advantage — which
// does not transduce to the memory-bound decode roofline, micro↛e2e — displace
// the vector path). NOTE: the tiled-shape derivation fail-closes when M is not a
// whole multiple of macM (no remainder path — protects the emitter), so every
// emittable tiled GEMM reaching cost already has matM >= macM; the parity cost is
// the explicit, auditable roofline guard on the cost layer, not a routinely-hit
// path (decode/GEVM can NEVER obtain the matrix-preferred cost).
constexpr double kIMEMatmulGemmPreferredCost = 0.5;
constexpr double kIMESingleFragmentMacCost = 20.0;
// Must equal the RVV vector base (kRVVVectorBaseCost = 1.0 in the RVV plugin):
// roofline parity for a whole-matrix boundary whose M is below the crossover M*.
constexpr double kIMEMatmulDecodeParityCost = 1.0;

// The load-bearing IME1 march token (FOUNDATION task 2: absent => assembler
// rejects `vmadot`). The capability is the proven IME1 envelope ONLY when this
// token is present. It assembles BOTH signedness forms below.
constexpr llvm::StringLiteral kIMEMarchToken("xsmtvdotii");
constexpr llvm::StringLiteral kIMEOpValue("vmadot");
constexpr llvm::StringLiteral kIMEUnsignedOpValue("vmadotu");
constexpr llvm::StringLiteral kIMEMixedSignOpValue("vmadotsu");
// The reversed-order mixed-sign mnemonic: unsigned A * signed B.
constexpr llvm::StringLiteral kIMEMixedSignUSOpValue("vmadotus");
// The slide family mnemonics (funct7 111001), selected by canonical sliding-MAC
// source geometry.
constexpr llvm::StringLiteral kIMESlide1OpValue("vmadot1");
constexpr llvm::StringLiteral kIMESlide2OpValue("vmadot2");
constexpr llvm::StringLiteral kIMESlide3OpValue("vmadot3");
constexpr int64_t kIMEElemInBits = 8;
constexpr int64_t kIMEAccumBits = 32;
// Default per-hart availability for the X60 (harts 0-3 carry _ime; hart 4 does
// not). Used only when the capability provider omits the property.
constexpr llvm::StringLiteral kIMEDefaultAvailableHarts("0-3");

llvm::Error makeIMEPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV IME extension plugin failed: ") + message,
      llvm::errc::invalid_argument);
}

enum class IMEProblemKind {
  DenseInt8MAC,
  SlidingInt8MAC,
  BlockQ40Contraction,
  BlockQ80Contraction,
  BlockQ4KContraction,
};

enum class IMEWeightFormat {
  None,
  Q40,
  Q80,
  Q4K,
};

/// Owner-local projection of the exact canonical P=(S,g,omega). No target
/// availability, MAC envelope, hart set, schedule or artifact fact lives here.
struct IMEProblemProjection {
  mlir::Operation *source = nullptr;
  IMEProblemKind kind = IMEProblemKind::DenseInt8MAC;
  IMEWeightFormat weightFormat = IMEWeightFormat::None;
  std::string imeOp;
  std::string signedness;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t slide = 0;
  int64_t qk = 0;
  int64_t weightBlockStride = 0;
  int64_t weightScaleByteOffset = 0;
  int64_t weightQuantByteOffset = 0;
  int64_t subblockLength = 0;
  int64_t numSubblocks = 0;
  int64_t scaleBits = 0;
  int64_t scaleTableBytes = 0;
};

/// Owner-local c_o projection of the selected target environment. It validates
/// only IME ISA/resource facts and explicitly rejects source-problem properties.
struct IMETargetProjection {
  int64_t vlenBits = 0;
  int64_t elemInBits = 0;
  int64_t accumBits = 0;
  int64_t macM = 0;
  int64_t macN = 0;
  int64_t macK = 0;
  std::string availableHarts;
};

struct IMEBoundProjection {
  IMEProblemProjection problem;
  IMETargetProjection target;
  bool isMatmul = false;
};

void projectSignedness(IMEProblemProjection &result,
                       weft::exec::IntegerSignedness lhs,
                       weft::exec::IntegerSignedness rhs) {
  using Signedness = weft::exec::IntegerSignedness;
  if (lhs == Signedness::Signed && rhs == Signedness::Signed) {
    result.signedness = kSignednessSigned.str();
    result.imeOp = kIMEOpValue.str();
  } else if (lhs == Signedness::Unsigned && rhs == Signedness::Unsigned) {
    result.signedness = kSignednessUnsigned.str();
    result.imeOp = kIMEUnsignedOpValue.str();
  } else if (lhs == Signedness::Signed && rhs == Signedness::Unsigned) {
    result.signedness = kSignednessMixedSign.str();
    result.imeOp = kIMEMixedSignOpValue.str();
  } else {
    result.signedness = kSignednessMixedSignUS.str();
    result.imeOp = kIMEMixedSignUSOpValue.str();
  }
}

/// Project the exact canonical problem without consulting target capability or
/// variant metadata.
llvm::Expected<IMEProblemProjection>
projectIMEProblem(mlir::Operation *problem) {
  if (!problem)
    return makeIMEPluginError(
        "IME source lifecycle requires an exact canonical problem");

  IMEProblemProjection result;
  result.source = problem;
  if (auto mac = llvm::dyn_cast<weft::exec::Int8MACProblemOp>(problem)) {
    result.kind = IMEProblemKind::DenseInt8MAC;
    result.m = mac.getM();
    result.n = mac.getN();
    result.k = mac.getK();
    projectSignedness(result, mac.getLhsSignedness(), mac.getRhsSignedness());
    return result;
  }
  if (auto sliding =
          llvm::dyn_cast<weft::exec::Int8SlidingMACProblemOp>(problem)) {
    result.kind = IMEProblemKind::SlidingInt8MAC;
    result.m = sliding.getM();
    result.n = sliding.getN();
    result.k = sliding.getK();
    result.slide = sliding.getSlide();
    projectSignedness(result, sliding.getLhsSignedness(),
                      sliding.getRhsSignedness());
    if (result.signedness != kSignednessSigned)
      return makeIMEPluginError(
          "sliding int8 MAC is only modeled for signed x signed input");
    result.imeOp = (result.slide == 1   ? kIMESlide1OpValue
                    : result.slide == 2 ? kIMESlide2OpValue
                                        : kIMESlide3OpValue)
                       .str();
    return result;
  }

  auto projectBlock = [&](auto contraction, IMEProblemKind kind,
                          IMEWeightFormat format,
                          llvm::StringRef formatName) -> llvm::Error {
    result.kind = kind;
    result.weightFormat = format;
    result.m = contraction.getM();
    result.n = contraction.getN();
    result.k = contraction.getK();
    result.qk = contraction.getQk();
    result.weightBlockStride = contraction.getWeightBlockStride();
    result.weightScaleByteOffset = contraction.getWeightScaleByteOffset();
    result.weightQuantByteOffset = contraction.getWeightQuantByteOffset();
    if (contraction.getActivationSignedness() !=
        weft::exec::IntegerSignedness::Signed)
      return makeIMEPluginError(
          llvm::Twine("IME ") + formatName +
          " contraction requires signed int8 activations");
    projectSignedness(result, weft::exec::IntegerSignedness::Signed,
                      weft::exec::IntegerSignedness::Signed);
    return llvm::Error::success();
  };

  if (auto q40 =
          llvm::dyn_cast<weft::exec::BlockQ40ContractionProblemOp>(problem)) {
    if (llvm::Error error = projectBlock(
            q40, IMEProblemKind::BlockQ40Contraction, IMEWeightFormat::Q40,
            kWeightFormatQ40))
      return std::move(error);
    return result;
  }
  if (auto q80 =
          llvm::dyn_cast<weft::exec::BlockQ80ContractionProblemOp>(problem)) {
    if (llvm::Error error = projectBlock(
            q80, IMEProblemKind::BlockQ80Contraction, IMEWeightFormat::Q80,
            kWeightFormatQ80))
      return std::move(error);
    return result;
  }
  if (auto q4k =
          llvm::dyn_cast<weft::exec::BlockQ4KContractionProblemOp>(problem)) {
    if (llvm::Error error = projectBlock(
            q4k, IMEProblemKind::BlockQ4KContraction, IMEWeightFormat::Q4K,
            kWeightFormatQ4K))
      return std::move(error);
    result.subblockLength = q4k.getSubblockLength();
    result.numSubblocks = q4k.getNumSubblocks();
    result.scaleBits = q4k.getScaleBits();
    result.scaleTableBytes = q4k.getScaleTableBytes();
    return result;
  }

  return makeIMEPluginError(
      llvm::Twine("canonical problem '") +
      problem->getName().getStringRef() +
      "' is outside the bounded IME problem domain");
}

bool capabilityHasProperty(const support::CapabilityDescriptor &capability,
                           llvm::StringRef name) {
  return capability.getProperties().find(name.str()) !=
         capability.getProperties().end();
}

/// Project c_o without consulting P or variant/body metadata.
llvm::Expected<IMETargetProjection>
projectIMETarget(const support::TargetCapabilitySet &capabilities) {
  const support::CapabilityDescriptor *capability =
      capabilities.lookupProviderByID(kIMECapabilityID);
  if (!capability || !capability->isAvailable())
    return makeIMEPluginError(
        "IME target projection requires available capability id 'spacemit.ime'");
  if (capability->getKind() != kIMECapabilityKind)
    return makeIMEPluginError(
        llvm::Twine("capability id '") + kIMECapabilityID +
        "' kind must be '" + kIMECapabilityKind + "'");
  for (llvm::StringRef name : kForbiddenProblemCapabilityProperties)
    if (capabilityHasProperty(*capability, name))
      return makeIMEPluginError(
          llvm::Twine("capability id '") + kIMECapabilityID +
          "' must not carry source-problem property '" + name +
          "'; bind the fact through kernel.problem");

  llvm::StringRef march = capability->getProperty(kMarchPropertyName);
  if (!march.contains(kIMEMarchToken))
    return makeIMEPluginError(
        llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
        kMarchPropertyName + "' must contain the load-bearing IME1 march token '" +
        kIMEMarchToken + "' (required to assemble `vmadot`)");

  llvm::StringRef vlenBitsText = capability->getProperty(kVlenBitsPropertyName);
  long long vlenBits = 0;
  if (vlenBitsText.empty() || vlenBitsText.getAsInteger(10, vlenBits))
    return makeIMEPluginError(
        llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
        kVlenBitsPropertyName +
        "' must be the integer VLEN in bits (derives the MAC fragment shape)");

  IMETargetProjection result;
  result.vlenBits = vlenBits;
  result.elemInBits = kIMEElemInBits;
  result.accumBits = kIMEAccumBits;
  if (vlenBits == 256) {
    result.macM = 4;
    result.macN = 4;
    result.macK = 8;
  } else {
    return makeIMEPluginError(
        llvm::Twine("capability id '") + kIMECapabilityID +
        "' VLEN=" + llvm::Twine(vlenBits) +
        " is outside the validated IME1 MAC-fragment envelope (VLEN=256 / "
        "4x4x8 is the only real-K1-validated shape)");
  }

  llvm::StringRef harts = capability->getProperty(kAvailableHartsPropertyName);
  result.availableHarts =
      harts.trim().empty() ? kIMEDefaultAvailableHarts.str() : harts.trim().str();
  return result;
}

llvm::Expected<bool>
classifyIMEProblemForTarget(const IMEProblemProjection &problem,
                            const IMETargetProjection &target) {
  const bool exactFragment = problem.m == target.macM &&
                             problem.n == target.macN &&
                             problem.k == target.macK;
  bool isMatmul = false;
  switch (problem.kind) {
  case IMEProblemKind::DenseInt8MAC:
    isMatmul = !exactFragment;
    break;
  case IMEProblemKind::SlidingInt8MAC:
    if (!exactFragment)
      return makeIMEPluginError(
          "sliding int8 MAC geometry must equal the target-projected MAC "
          "fragment; tiled sliding construction is not modeled");
    break;
  case IMEProblemKind::BlockQ40Contraction:
  case IMEProblemKind::BlockQ80Contraction:
  case IMEProblemKind::BlockQ4KContraction:
    isMatmul = true;
    break;
  }

  if (isMatmul &&
      (problem.m % target.macM != 0 || problem.n % target.macN != 0 ||
       problem.k % target.macK != 0))
    return makeIMEPluginError(
        llvm::Twine("canonical problem geometry ") + llvm::Twine(problem.m) +
        "x" + llvm::Twine(problem.n) + "x" + llvm::Twine(problem.k) +
        " must be a whole multiple of the target-projected MAC fragment " +
        llvm::Twine(target.macM) + "x" + llvm::Twine(target.macN) + "x" +
        llvm::Twine(target.macK) + "; no remainder path is emitted");
  if (isMatmul && (problem.signedness == kSignednessMixedSign ||
                   problem.signedness == kSignednessMixedSignUS))
    return makeIMEPluginError(
        "mixed-sign IME construction is only modeled for a single MAC "
        "fragment, not a tiled whole-matrix problem");
  if (problem.weightFormat != IMEWeightFormat::None &&
      (problem.qk <= 0 || problem.k % problem.qk != 0))
    return makeIMEPluginError(
        "block-quantized contraction k must contain whole canonical blocks");
  return isMatmul;
}

llvm::Expected<IMEBoundProjection>
projectIMEInputs(mlir::Operation *problem,
                 const support::TargetCapabilitySet &capabilities) {
  llvm::Expected<IMEProblemProjection> problemProjection =
      projectIMEProblem(problem);
  if (!problemProjection)
    return problemProjection.takeError();
  llvm::Expected<IMETargetProjection> targetProjection =
      projectIMETarget(capabilities);
  if (!targetProjection)
    return targetProjection.takeError();
  llvm::Expected<bool> isMatmul = classifyIMEProblemForTarget(
      *problemProjection, *targetProjection);
  if (!isMatmul)
    return isMatmul.takeError();
  return IMEBoundProjection{*problemProjection, *targetProjection, *isMatmul};
}

llvm::StringRef getIMEVariantName(const IMEBoundProjection &projection) {
  const IMEProblemProjection &problem = projection.problem;
  if (problem.kind == IMEProblemKind::SlidingInt8MAC)
    return kIMESlideVariantName;
  if (projection.isMatmul)
    return problem.signedness == kSignednessUnsigned ? kIMEMatmulUVariantName
                                                      : kIMEMatmulVariantName;
  if (problem.signedness == kSignednessMixedSign)
    return kIMEMixedSignVariantName;
  if (problem.signedness == kSignednessMixedSignUS)
    return kIMEMixedSignUSVariantName;
  return problem.signedness == kSignednessUnsigned ? kIMEUnsignedVariantName
                                                    : kIMEFirstSliceVariantName;
}

llvm::Error rejectLegacyIMEVariantMirrors(weft::exec::VariantOp variant) {
  for (llvm::StringRef name : kForbiddenVariantProblemMirrors)
    if (variant->hasAttr(name))
      return makeIMEPluginError(
          llvm::Twine("IME variant must not mirror canonical problem fact '") +
          name + "'; consume the exact kernel.problem operation");
  return llvm::Error::success();
}

bool hasAvailableIMECapability(const VariantProposalRequest &request) {
  if (!request.getKernel())
    return false;
  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kIMECapabilityID);
  return capability && capability->isAvailable();
}

bool isIMEFinalBody(mlir::Operation *operation) {
  return llvm::isa<weft::ime::MMAOp, weft::ime::MMAUOp,
                   weft::ime::MMASUOp, weft::ime::MMAUSOp,
                   weft::ime::MMASlideOp, weft::ime::MatMulOp,
                   weft::ime::Q40MatMulTileOp, weft::ime::Q80MatMulTileOp,
                   weft::ime::Q4KMatMulTileOp>(operation);
}

llvm::Expected<mlir::Operation *>
inspectIMEFinalBodySlot(weft::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return makeIMEPluginError(
        "IME construction requires a materialized variant body");

  mlir::Operation *found = nullptr;
  for (mlir::Operation &operation : variant.getBody().front()) {
    if (!isIMEFinalBody(&operation))
      continue;
    if (found)
      return makeIMEPluginError(
          "IME variant canonical body slot contains multiple final roots");
    if (!isOperationSelectedForVariant(&operation, variant))
      return makeIMEPluginError(
          "IME variant canonical body slot contains a root with stale "
          "selected-variant ownership");
    found = &operation;
  }
  return found;
}

llvm::Error requireIMEIntegerFact(mlir::Operation *operation,
                                  llvm::StringRef name, int64_t expected) {
  auto value = operation->getAttrOfType<mlir::IntegerAttr>(name);
  if (!value || value.getInt() != expected)
    return makeIMEPluginError(
        llvm::Twine("exact IME body fact '") + name + "' must equal " +
        llvm::Twine(expected));
  return llvm::Error::success();
}

llvm::Error requireIMEStringFact(mlir::Operation *operation,
                                 llvm::StringRef name,
                                 llvm::StringRef expected) {
  auto value = operation->getAttrOfType<mlir::StringAttr>(name);
  if (!value || value.getValue() != expected)
    return makeIMEPluginError(
        llvm::Twine("exact IME body fact '") + name + "' must equal '" +
        expected + "'");
  return llvm::Error::success();
}

llvm::Expected<mlir::Operation *>
findUniqueIMENestedOp(mlir::Operation *body, llvm::StringRef operationName) {
  if (body->getNumRegions() != 1 || body->getRegion(0).empty())
    return makeIMEPluginError(
        "quantized exact IME body requires one materialized typed region");
  mlir::Operation *found = nullptr;
  for (mlir::Operation &nested : body->getRegion(0).front()) {
    if (nested.getName().getStringRef() != operationName)
      continue;
    if (found)
      return makeIMEPluginError(
          llvm::Twine("quantized exact IME body contains multiple '") +
          operationName + "' bricks");
    found = &nested;
  }
  if (!found)
    return makeIMEPluginError(
        llvm::Twine("quantized exact IME body is missing '") + operationName +
        "' brick");
  return found;
}

/// Check an existing canonical body slot against the same exact P and c_o
/// projections used by proposal, legality, cost and fresh construction.
llvm::Error validateIMEFinalBodyAgainstProjection(
    mlir::Operation *body, const IMEBoundProjection &projection) {
  const IMEProblemProjection &problem = projection.problem;
  const IMETargetProjection &target = projection.target;

  bool expectedType = false;
  switch (problem.weightFormat) {
  case IMEWeightFormat::Q40:
    expectedType = llvm::isa<weft::ime::Q40MatMulTileOp>(body);
    break;
  case IMEWeightFormat::Q80:
    expectedType = llvm::isa<weft::ime::Q80MatMulTileOp>(body);
    break;
  case IMEWeightFormat::Q4K:
    expectedType = llvm::isa<weft::ime::Q4KMatMulTileOp>(body);
    break;
  case IMEWeightFormat::None:
    if (problem.kind == IMEProblemKind::SlidingInt8MAC)
      expectedType = llvm::isa<weft::ime::MMASlideOp>(body);
    else if (projection.isMatmul)
      expectedType = llvm::isa<weft::ime::MatMulOp>(body);
    else if (problem.signedness == kSignednessUnsigned)
      expectedType = llvm::isa<weft::ime::MMAUOp>(body);
    else if (problem.signedness == kSignednessMixedSign)
      expectedType = llvm::isa<weft::ime::MMASUOp>(body);
    else if (problem.signedness == kSignednessMixedSignUS)
      expectedType = llvm::isa<weft::ime::MMAUSOp>(body);
    else
      expectedType = llvm::isa<weft::ime::MMAOp>(body);
    break;
  }
  if (!expectedType)
    return makeIMEPluginError(
        "canonical IME body type does not match the exact canonical problem");

  if (llvm::Error error =
          requireIMEStringFact(body, kIMEOpAttrName, problem.imeOp))
    return error;
  for (auto [name, expected] : {
           std::pair<llvm::StringRef, int64_t>(kElemInBitsAttrName,
                                               target.elemInBits),
           {kAccumBitsAttrName, target.accumBits},
           {kMacMAttrName, target.macM},
           {kMacNAttrName, target.macN},
           {kMacKAttrName, target.macK},
       })
    if (llvm::Error error = requireIMEIntegerFact(body, name, expected))
      return error;
  if (llvm::Error error = requireIMEStringFact(
          body, kAvailableHartsAttrName, target.availableHarts))
    return error;

  if (projection.isMatmul) {
    for (auto [name, expected] : {
             std::pair<llvm::StringRef, int64_t>(kMatMAttrName, problem.m),
             {kMatNAttrName, problem.n},
             {kMatKAttrName, problem.k},
         })
      if (llvm::Error error = requireIMEIntegerFact(body, name, expected))
        return error;
  }
  if (problem.kind == IMEProblemKind::SlidingInt8MAC)
    if (llvm::Error error =
            requireIMEIntegerFact(body, kSlideAttrName, problem.slide))
      return error;

  if (problem.weightFormat == IMEWeightFormat::None)
    return llvm::Error::success();

  llvm::StringRef format =
      problem.weightFormat == IMEWeightFormat::Q40
          ? kWeightFormatQ40
          : (problem.weightFormat == IMEWeightFormat::Q80 ? kWeightFormatQ80
                                                          : kWeightFormatQ4K);
  if (llvm::Error error =
          requireIMEStringFact(body, kWeightFormatAttrName, format))
    return error;
  for (auto [name, expected] : {
           std::pair<llvm::StringRef, int64_t>(kQkAttrName, problem.qk),
           {kWeightBlockStrideAttrName, problem.weightBlockStride},
           {kWeightQuantByteOffsetAttrName, problem.weightQuantByteOffset},
       })
    if (llvm::Error error = requireIMEIntegerFact(body, name, expected))
      return error;

  llvm::StringRef dequantName =
      problem.weightFormat == IMEWeightFormat::Q40
          ? weft::ime::Q40DequantCoreOp::getOperationName()
          : (problem.weightFormat == IMEWeightFormat::Q80
                 ? weft::ime::Q80DequantCoreOp::getOperationName()
                 : weft::ime::Q4KDequantCoreOp::getOperationName());
  llvm::Expected<mlir::Operation *> dequant =
      findUniqueIMENestedOp(body, dequantName);
  if (!dequant)
    return dequant.takeError();
  for (auto [name, expected] : {
           std::pair<llvm::StringRef, int64_t>(kQkAttrName, problem.qk),
           {kWeightBlockStrideAttrName, problem.weightBlockStride},
           {kWeightQuantByteOffsetAttrName, problem.weightQuantByteOffset},
           {"weight_scale_byte_offset", problem.weightScaleByteOffset},
       })
    if (llvm::Error error = requireIMEIntegerFact(*dequant, name, expected))
      return error;

  if (problem.weightFormat != IMEWeightFormat::Q4K)
    return llvm::Error::success();
  llvm::Expected<mlir::Operation *> scaleMin = findUniqueIMENestedOp(
      body, weft::ime::Q4KScaleMinUnpackCoreOp::getOperationName());
  if (!scaleMin)
    return scaleMin.takeError();
  for (auto [name, expected] : {
           std::pair<llvm::StringRef, int64_t>("num_sub_blocks",
                                               problem.numSubblocks),
           {"scale_bits", problem.scaleBits},
           {"k_scale_size", problem.scaleTableBytes},
           {"weight_scale_byte_offset", problem.weightScaleByteOffset},
       })
    if (llvm::Error error = requireIMEIntegerFact(*scaleMin, name, expected))
      return error;
  return llvm::Error::success();
}

llvm::Error validateExactIMEConstructionResult(
    const VariantEmissionRequest &request) {
  mlir::Operation *body = request.getConstructedOperation();
  if (!body)
    return makeIMEPluginError(
        "artifact query requires the exact IME operation returned by family "
        "construction");
  if (!llvm::isa<weft::ime::MMAOp, weft::ime::MMAUOp,
                 weft::ime::MMASUOp, weft::ime::MMAUSOp,
                 weft::ime::MMASlideOp, weft::ime::MatMulOp,
                 weft::ime::Q40MatMulTileOp, weft::ime::Q80MatMulTileOp,
                 weft::ime::Q4KMatMulTileOp>(body))
    return makeIMEPluginError(
        "family construction returned a non-IME or non-final typed operation");
  if (!isOperationSelectedForVariant(body, request.getVariant()))
    return makeIMEPluginError(
        "exact IME construction result is not bound to the requested variant");

  if (auto op = llvm::dyn_cast<weft::ime::MMAOp>(body)) {
    if (mlir::failed(op.verify()) ||
        mlir::failed(ime::requireIMESimpleComputationPlan(body)))
      return makeIMEPluginError("exact IME mma body is malformed");
  } else if (auto op = llvm::dyn_cast<weft::ime::MMAUOp>(body)) {
    if (mlir::failed(op.verify()) ||
        mlir::failed(ime::requireIMESimpleComputationPlan(body)))
      return makeIMEPluginError("exact IME mma_u body is malformed");
  } else if (auto op = llvm::dyn_cast<weft::ime::MMASUOp>(body)) {
    if (mlir::failed(op.verify()) ||
        mlir::failed(ime::requireIMESimpleComputationPlan(body)))
      return makeIMEPluginError("exact IME mma_su body is malformed");
  } else if (auto op = llvm::dyn_cast<weft::ime::MMAUSOp>(body)) {
    if (mlir::failed(op.verify()) ||
        mlir::failed(ime::requireIMESimpleComputationPlan(body)))
      return makeIMEPluginError("exact IME mma_us body is malformed");
  } else if (auto op = llvm::dyn_cast<weft::ime::MMASlideOp>(body)) {
    if (mlir::failed(op.verify()) ||
        mlir::failed(ime::requireIMESimpleComputationPlan(body)))
      return makeIMEPluginError("exact IME mma_slide body is malformed");
  } else if (auto op = llvm::dyn_cast<weft::ime::MatMulOp>(body)) {
    if (mlir::failed(op.verify()) ||
        mlir::failed(ime::readIMEMatMulComputationPlan(body)))
      return makeIMEPluginError("exact IME matmul body is malformed");
  } else if (auto op = llvm::dyn_cast<weft::ime::Q40MatMulTileOp>(body)) {
    if (mlir::failed(op.verify()) ||
        mlir::failed(ime::readIMEQuantComputationPlan(body)))
      return makeIMEPluginError("exact IME q4_0 tile body is malformed");
  } else if (auto op = llvm::dyn_cast<weft::ime::Q80MatMulTileOp>(body)) {
    if (mlir::failed(op.verify()) ||
        mlir::failed(ime::readIMEQuantComputationPlan(body)))
      return makeIMEPluginError("exact IME q8_0 tile body is malformed");
  } else if (auto op = llvm::dyn_cast<weft::ime::Q4KMatMulTileOp>(body)) {
    if (mlir::failed(op.verify()) ||
        mlir::failed(ime::readIMEQuantComputationPlan(body)))
      return makeIMEPluginError("exact IME q4_K tile body is malformed");
  }
  return llvm::Error::success();
}

llvm::Expected<VariantProposal>
buildIMEProposal(const VariantProposalRequest &request) {
  llvm::Expected<IMEBoundProjection> projection =
      projectIMEInputs(request.getProblem(), request.getCapabilities());
  if (!projection)
    return projection.takeError();

  VariantProposal proposal(getIMEVariantName(*projection), kIMEPluginName);
  proposal.setFormulaID(ime::kIMEConstructionFormulaID);
  proposal.addRequiredCapabilityID(kIMECapabilityID);
  proposal.setCondition(kIMECondition);
  proposal.setGuard(kIMEGuard);
  proposal.setPolicy(kIMEPolicy);
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

llvm::StringRef IMEExtensionPlugin::getConstructionDomain() const {
  return "riscv-execution";
}

llvm::StringRef IMEExtensionPlugin::getVersion() const {
  return kIMEPluginVersion;
}

llvm::ArrayRef<PluginCapability> IMEExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void IMEExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<weft::ime::WEFTIMEDialect>();
}

llvm::Error IMEExtensionPlugin::constructFormulaPlans(
    const FamilyConstructionRequest &request,
    FamilyConstructionResult &out) const {
  llvm::Expected<IMEBoundProjection> projection =
      projectIMEInputs(request.getProblem(), request.getCapabilities());
  if (!projection)
    return projection.takeError();

  llvm::Expected<mlir::Operation *> existing =
      inspectIMEFinalBodySlot(request.getVariant());
  if (!existing)
    return existing.takeError();

  mlir::Operation *body = *existing;
  if (!body) {
    mlir::OpBuilder builder(request.getModule().getContext());
    builder.setInsertionPointToEnd(&request.getVariant().getBody().front());
    VariantLoweringBoundaryRequest bodyRequest(
        request.getVariant(), request.getKernel(), request.getProblem(),
        request.getCapabilities(), request.getRole(), builder, nullptr);
    VariantLoweringBoundaryResult bodyResult;
    if (llvm::Error error = constructSelectedFinalBody(bodyRequest, bodyResult))
      return error;
    body = bodyResult.getMaterializedOperation();
  }
  if (!body)
    return makeIMEPluginError(
        "family construction produced no selected IME final body");
  if (llvm::Error error =
          validateIMEFinalBodyAgainstProjection(body, *projection))
    return error;
  if (mlir::failed(constructIMEFormulaPlan(
          body, request.getVariant(), request.getKernel(),
          projection->target.vlenBits)))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "IME formula construction rejected the bound typed body");
  out = FamilyConstructionResult::getFinalBody(body);
  return llvm::Error::success();
}

void IMEExtensionPlugin::collectFormulaDescriptors(
    llvm::SmallVectorImpl<FormulaDescriptor> &out) const {
  FormulaDescriptor construction(
      kIMEConstructionFormulaID, kIMEPluginName,
      "contraction/integer-matrix-extension", FormulaResultKind::CandidateSet,
      FormulaConstructionStrength::ConstructedWeak);
  construction.getGeometryAxis().set(FormulaAxisUse::Decisive,
                                     "IMEProblemProjection");
  for (llvm::StringRef field : {"problem-kind", "signedness", "m", "n", "k",
                                "weight-format", "block-layout",
                                "slide-window"})
    construction.getGeometryAxis().addConsumedField(field);
  construction.getCapabilityAxis().set(FormulaAxisUse::Decisive,
                                       "IMETargetProjection");
  for (llvm::StringRef field : {"march", "vlen-bits", "available-harts"})
    construction.getCapabilityAxis().addConsumedField(field);
  construction.getStaticContextAxis().set(FormulaAxisUse::HonestNull,
                                          "IMENoStaticContext");
  for (llvm::StringRef semanticCase :
       {"signed-mma", "unsigned-mma", "mixed-sign-su", "mixed-sign-us",
        "sliding-window", "whole-matrix", "q4-0-matrix-tile",
        "q8-0-matrix-tile", "q4-k-matrix-tile",
        "unsupported-capability"})
    construction.addSemanticCase(semanticCase);
  construction.addProductionEntry("plugin:variant-proposal");
  construction.addProductionEntry("construction:ime-final-typed-body");
  out.push_back(std::move(construction));

  FormulaDescriptor cost(
      kIMECostFormulaID, kIMEPluginName,
      "contraction/integer-matrix-extension", FormulaResultKind::AnalyticPrior,
      FormulaConstructionStrength::ConstructedWeak);
  cost.getGeometryAxis().set(FormulaAxisUse::Decisive,
                            "IMEProblemProjection");
  for (llvm::StringRef field : {"problem-kind", "m", "n", "k"})
    cost.getGeometryAxis().addConsumedField(field);
  cost.getCapabilityAxis().set(FormulaAxisUse::Decisive,
                              "IMETargetProjection");
  cost.getCapabilityAxis().addConsumedField("mac-m");
  cost.getStaticContextAxis().set(FormulaAxisUse::HonestNull,
                                 "IMECostNoStaticContext");
  cost.addSemanticCase("gemm-above-derived-crossover");
  cost.addSemanticCase("gemv-below-derived-crossover");
  cost.addSemanticCase("fragment-mma-prior");
  cost.addProductionEntry("plugin:analytic-cost");
  out.push_back(std::move(cost));
}

bool IMEExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  mlir::Operation *problem = request.getProblem();
  return problem &&
         llvm::isa<weft::exec::Int8MACProblemOp,
                   weft::exec::Int8SlidingMACProblemOp,
                   weft::exec::BlockQ40ContractionProblemOp,
                   weft::exec::BlockQ80ContractionProblemOp,
                   weft::exec::BlockQ4KContractionProblemOp>(problem) &&
         hasAvailableIMECapability(request);
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
  weft::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeIMEPluginError(
        "legality verification requires a materialized weft.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kIMEPluginName)
    return makeIMEPluginError(
        "materialized IME variant must be owned by origin 'ime-plugin'");

  llvm::Expected<IMEBoundProjection> projection =
      projectIMEInputs(request.getProblem(), request.getCapabilities());
  if (!projection)
    return projection.takeError();
  if (llvm::Error error = rejectLegacyIMEVariantMirrors(variant))
    return error;
  if (variant.getSymName() != getIMEVariantName(*projection))
    return makeIMEPluginError(
        llvm::Twine("materialized IME variant symbol '") +
        variant.getSymName() + "' does not match exact canonical problem '" +
        getIMEVariantName(*projection) + "'");

  // Plugin-owned legality: the variant requires the IME target capability;
  // source semantics and geometry have already been projected from exact P.
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
  return llvm::Error::success();
}

llvm::Error IMEExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeIMEPluginError(
        "cost estimation requires a materialized weft.exec.variant");

  llvm::Expected<IMEBoundProjection> projection =
      projectIMEInputs(request.getProblem(), request.getCapabilities());
  if (!projection)
    return projection.takeError();
  if (llvm::Error error = rejectLegacyIMEVariantMirrors(request.getVariant()))
    return error;
  if (request.getVariant().getSymName() != getIMEVariantName(*projection))
    return makeIMEPluginError(
        "IME cost variant does not match the exact canonical problem");

  out = VariantCostEstimate();
  out.setExplicitPreference(true);
  out.setOriginPlugin(kIMEPluginName);
  out.setFormulaID(kIMECostFormulaID);
  out.setVariantSymbol(request.getVariant().getSymName());
  if (projection->isMatmul) {
    // M belongs to exact P; the crossover M*=macM belongs to projected c_o.
    const int64_t crossoverM = projection->target.macM;
    if (projection->problem.m >= crossoverM) {
      // M >= M*: COMPUTE-BOUND prefill takeover (P7 pattern: ime ∧ matmul-shape
      // ∧ M>=M*). The whole-matrix contraction fully feeds the systolic array, so
      // the capability-derived cost ranks the matrix variant AHEAD of the RVV
      // vector base — IME wins because exact problem M reaches the target MAC
      // crossover, not because a constant sorts first.
      out.setScore(kIMEMatmulGemmPreferredCost);
      out.setExplanation(
          "IME whole-matrix vmadot GEMM boundary; canonical problem M is "
          "at/above target-projected crossover M*=macM, the compute-bound "
          "prefill regime where the matrix paradigm is preferred over the RVV "
          "vector base");
      out.setPolicy(
          "prefer the IME matrix paradigm when exact canonical problem M reaches "
          "the target-projected crossover M* (macM)");
    } else {
      // M < M*: MEMORY-BOUND decode / matrix-vector regime. Record ROOFLINE
      // PARITY, NOT a matrix win: the compute-isolated M<M* micro-advantage does
      // NOT transduce to the memory-bound decode/GEVM roofline (weight streaming +
      // dequant dominate; micro↛e2e). Score the matrix variant at PARITY with the
      // RVV vector base so a compute-micro signal the decode roofline never
      // realizes cannot displace the vector path. (Guard: the tiled derivation
      // fail-closes below macM — no remainder path — so this is auditable defense-
      // in-depth on the cost layer, not a routinely-hit path.)
      out.setScore(kIMEMatmulDecodeParityCost);
      out.setExplanation(
          "IME whole-matrix vmadot boundary whose canonical problem M is BELOW "
          "the target-projected crossover M* (the MAC fragment row dimension): the "
          "memory-bound decode / matrix-vector roofline regime; cost recorded at "
          "PARITY with the RVV vector base (the compute-isolated micro-advantage "
          "does not transduce to the decode roofline) so the matrix paradigm is "
          "NOT preferred over the vector path");
      out.setPolicy(
          "record roofline parity (do NOT prefer the matrix paradigm) for a "
          "whole-matrix boundary whose canonical problem M is below the target-projected "
          "crossover M* (macM): the memory-bound decode / matrix-vector regime");
    }
  } else {
    // Single-fragment MAC boundary (mma / mma_u / mma_su / mma_us / mma_slide):
    // a leaf MAC surface, not a whole-kernel GEMM takeover. The capability-derived
    // cost beats the scalar fallback but stays above the RVV vector base, so a
    // single MAC fragment never displaces a full vectorized kernel.
    out.setScore(kIMESingleFragmentMacCost);
    out.setExplanation(
        "IME int8->int32 vmadot MAC boundary; cost DERIVED from the available "
        "spacemit.ime capability deriving to the FOUNDATION-validated single MAC "
        "fragment (lowered to the vmadot kernel through the common EmitC route)");
    out.setPolicy(
        "prefer IME only when the spacemit.ime capability fact is available and "
        "derives to the validated IME1 int8->int32 MAC envelope");
  }
  return llvm::Error::success();
}

llvm::Error IMEExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeIMEPluginError(
        "emission readiness requires a materialized weft.exec.variant");
  if (!request.getKernel())
    return makeIMEPluginError(
        "emission readiness requires an enclosing weft.exec.kernel");

  if (llvm::Error error = validateExactIMEConstructionResult(request)) {
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
        "emission planning requires a materialized weft.exec.variant");
  if (!request.getKernel())
    return makeIMEPluginError(
        "emission planning requires an enclosing weft.exec.kernel");

  if (llvm::Error error = validateExactIMEConstructionResult(request))
    return error;
  llvm::StringRef boundaryOpName =
      request.getConstructedOperation()->getName().getStringRef();
  out = VariantEmissionPlan::getSupported(
      kIMEPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "materialized-emitc-cpp-ime-vmadot-mma-module",
      "ime-vmadot-mma-emitc-route",
      "ime-vmadot-mma-runtime-c-abi.v1", "riscv-elf-relocatable-object",
      "IME selected boundary lowers the FOUNDATION-validated int8->int32 MAC "
      "kernel (signed vmadot or unsigned vmadotu) through the common "
      "constructed-body EmitC materializer and the MLIR EmitC C/C++ emitter");
  out.setRuntimeABIKind("plugin-owned-runtime-abi");
  out.setRuntimeABIName("ime-vmadot-mma-runtime-c-abi.v1");
  out.setRuntimeGlueRole("emitc-cpp-ime-vmadot-mma-runtime-glue");
  out.setLoweringBoundaryOpName(boundaryOpName);
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error IMEExtensionPlugin::constructSelectedFinalBody(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  weft::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeIMEPluginError(
        "lowering-boundary materialization requires a materialized "
        "weft.exec.variant");
  weft::exec::KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeIMEPluginError(
        "lowering-boundary materialization requires an enclosing "
        "weft.exec.kernel");

  VariantLegalityRequest legality(variant, kernel, request.getProblem(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeIMEPluginError(
        llvm::Twine("selected IME variant @") + variant.getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  llvm::Expected<IMEBoundProjection> projection =
      projectIMEInputs(request.getProblem(), request.getCapabilities());
  if (!projection)
    return projection.takeError();
  const IMEProblemProjection &problem = projection->problem;
  const IMETargetProjection &target = projection->target;

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::MLIRContext *context = builder.getContext();
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  auto addCommonAttributes = [&](mlir::OperationState &state) {
    state.addAttribute(kSourceKernelAttrName,
                       builder.getStringAttr(kernel.getSymName()));
    state.addAttribute(
        kSelectedVariantAttrName,
        mlir::FlatSymbolRefAttr::get(context, variant.getSymName()));
    state.addAttribute(kOriginAttrName,
                       builder.getStringAttr(kIMEPluginName));
    state.addAttribute(
        kRoleAttrName,
        builder.getStringAttr(stringifyVariantEmissionRole(request.getRole())));
    state.addAttribute(kStatusAttrName,
                       builder.getStringAttr(kRoleOpBoundaryStatusValue));
    state.addAttribute(kRequiredCapabilitiesAttrName, variantRequires);
    state.addAttribute(kIMEOpAttrName, builder.getStringAttr(problem.imeOp));
    state.addAttribute(kElemInBitsAttrName,
                       builder.getI64IntegerAttr(target.elemInBits));
    state.addAttribute(kAccumBitsAttrName,
                       builder.getI64IntegerAttr(target.accumBits));
    state.addAttribute(kMacMAttrName,
                       builder.getI64IntegerAttr(target.macM));
    state.addAttribute(kMacNAttrName,
                       builder.getI64IntegerAttr(target.macN));
    state.addAttribute(kMacKAttrName,
                       builder.getI64IntegerAttr(target.macK));
    state.addAttribute(kAvailableHartsAttrName,
                       builder.getStringAttr(target.availableHarts));
  };
  auto addMatrixAttributes = [&](mlir::OperationState &state) {
    state.addAttribute(kMatMAttrName, builder.getI64IntegerAttr(problem.m));
    state.addAttribute(kMatNAttrName, builder.getI64IntegerAttr(problem.n));
    state.addAttribute(kMatKAttrName, builder.getI64IntegerAttr(problem.k));
  };
  auto finishBody = [&](mlir::Operation *body) -> llvm::Error {
    VariantLoweringBoundaryValidationRequest validationRequest(
        variant, kernel, request.getCapabilities(), request.getRole(), body);
    if (llvm::Error error =
            validateSelectedLoweringBoundary(validationRequest))
      return error;
    out = VariantLoweringBoundaryResult::getMaterialized(
        kIMEPluginName, kernel.getSymName(), variant.getSymName(),
        request.getRole(), body);
    return llvm::Error::success();
  };

  if (problem.weightFormat == IMEWeightFormat::Q40 ||
      problem.weightFormat == IMEWeightFormat::Q80) {
    const bool isQ40 = problem.weightFormat == IMEWeightFormat::Q40;
    llvm::StringRef tileOpName =
        isQ40 ? weft::ime::Q40MatMulTileOp::getOperationName()
              : weft::ime::Q80MatMulTileOp::getOperationName();
    llvm::StringRef dequantOpName =
        isQ40 ? weft::ime::Q40DequantCoreOp::getOperationName()
              : weft::ime::Q80DequantCoreOp::getOperationName();
    llvm::StringRef yieldOpName =
        isQ40 ? weft::ime::Q40MatMulTileYieldOp::getOperationName()
              : weft::ime::Q80MatMulTileYieldOp::getOperationName();
    llvm::StringRef format = isQ40 ? kWeightFormatQ40 : kWeightFormatQ80;
    llvm::StringRef decodeModel = isQ40 ? kQ40DecodeModel : kQ80DecodeModel;

    mlir::Location loc = variant.getLoc();
    mlir::OperationState tileState(loc, tileOpName);
    addCommonAttributes(tileState);
    addMatrixAttributes(tileState);
    tileState.addAttribute(kWeightFormatAttrName,
                           builder.getStringAttr(format));
    tileState.addAttribute(kQkAttrName,
                           builder.getI64IntegerAttr(problem.qk));
    tileState.addAttribute(
        kWeightBlockStrideAttrName,
        builder.getI64IntegerAttr(problem.weightBlockStride));
    tileState.addAttribute(
        kWeightQuantByteOffsetAttrName,
        builder.getI64IntegerAttr(problem.weightQuantByteOffset));
    tileState.addRegion();
    mlir::Operation *tile = builder.create(tileState);

    auto i8FragType = mlir::VectorType::get(
        {target.macM * target.macK}, builder.getI8Type());
    auto i32AccType = mlir::VectorType::get(
        {target.macM * target.macN}, builder.getI32Type());
    mlir::Block &body = tile->getRegion(0).emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value aFragment = body.addArgument(i8FragType, loc);
    mlir::Value accIn = body.addArgument(i32AccType, loc);

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::OperationState dequantState(loc, dequantOpName);
    dequantState.addOperands({blockIndex});
    dequantState.addAttribute("decode_model",
                              builder.getStringAttr(decodeModel));
    dequantState.addAttribute(kQkAttrName,
                              builder.getI64IntegerAttr(problem.qk));
    dequantState.addAttribute(
        kWeightBlockStrideAttrName,
        builder.getI64IntegerAttr(problem.weightBlockStride));
    dequantState.addAttribute(
        kWeightQuantByteOffsetAttrName,
        builder.getI64IntegerAttr(problem.weightQuantByteOffset));
    dequantState.addAttribute(
        "weight_scale_byte_offset",
        builder.getI64IntegerAttr(problem.weightScaleByteOffset));
    dequantState.addTypes({i8FragType});
    mlir::Operation *dequant = builder.create(dequantState);

    mlir::OperationState macState(
        loc, weft::ime::VmadotMacLeafOp::getOperationName());
    macState.addOperands(
        {aFragment, dequant->getResult(0), blockIndex, accIn});
    macState.addAttribute(kIMEOpAttrName,
                          builder.getStringAttr(problem.imeOp));
    macState.addAttribute(kElemInBitsAttrName,
                          builder.getI64IntegerAttr(target.elemInBits));
    macState.addAttribute(kAccumBitsAttrName,
                          builder.getI64IntegerAttr(target.accumBits));
    macState.addAttribute(kMacMAttrName,
                          builder.getI64IntegerAttr(target.macM));
    macState.addAttribute(kMacNAttrName,
                          builder.getI64IntegerAttr(target.macN));
    macState.addAttribute(kMacKAttrName,
                          builder.getI64IntegerAttr(target.macK));
    macState.addTypes({i32AccType});
    mlir::Operation *mac = builder.create(macState);

    mlir::OperationState yieldState(loc, yieldOpName);
    yieldState.addOperands({mac->getResult(0)});
    builder.create(yieldState);
    return finishBody(tile);
  }

  if (problem.weightFormat == IMEWeightFormat::Q4K) {
    mlir::Location loc = variant.getLoc();
    mlir::OperationState tileState(
        loc, weft::ime::Q4KMatMulTileOp::getOperationName());
    addCommonAttributes(tileState);
    addMatrixAttributes(tileState);
    tileState.addAttribute(kWeightFormatAttrName,
                           builder.getStringAttr(kWeightFormatQ4K));
    tileState.addAttribute(kQkAttrName,
                           builder.getI64IntegerAttr(problem.qk));
    tileState.addAttribute(
        kWeightBlockStrideAttrName,
        builder.getI64IntegerAttr(problem.weightBlockStride));
    tileState.addAttribute(
        kWeightQuantByteOffsetAttrName,
        builder.getI64IntegerAttr(problem.weightQuantByteOffset));
    tileState.addRegion();
    mlir::Operation *tile = builder.create(tileState);

    auto i8FragType = mlir::VectorType::get(
        {target.macM * target.macK}, builder.getI8Type());
    auto i32AccType = mlir::VectorType::get(
        {target.macM * target.macN}, builder.getI32Type());
    mlir::Block &body = tile->getRegion(0).emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value aFragment = body.addArgument(i8FragType, loc);
    mlir::Value accScaleIn = body.addArgument(i32AccType, loc);
    mlir::Value accMinIn = body.addArgument(i32AccType, loc);

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    mlir::OperationState dequantState(
        loc, weft::ime::Q4KDequantCoreOp::getOperationName());
    dequantState.addOperands({blockIndex});
    dequantState.addAttribute("decode_model",
                              builder.getStringAttr(kQ4KDecodeModel));
    dequantState.addAttribute(kQkAttrName,
                              builder.getI64IntegerAttr(problem.qk));
    dequantState.addAttribute(
        kWeightBlockStrideAttrName,
        builder.getI64IntegerAttr(problem.weightBlockStride));
    dequantState.addAttribute(
        kWeightQuantByteOffsetAttrName,
        builder.getI64IntegerAttr(problem.weightQuantByteOffset));
    dequantState.addAttribute(
        "weight_scale_byte_offset",
        builder.getI64IntegerAttr(problem.weightScaleByteOffset));
    dequantState.addTypes({i8FragType});
    mlir::Operation *dequant = builder.create(dequantState);

    mlir::OperationState scaleMinState(
        loc, weft::ime::Q4KScaleMinUnpackCoreOp::getOperationName());
    scaleMinState.addOperands({blockIndex});
    scaleMinState.addAttribute("scale_min_model",
                               builder.getStringAttr(kQ4KScaleMinModel));
    scaleMinState.addAttribute(
        "num_sub_blocks", builder.getI64IntegerAttr(problem.numSubblocks));
    scaleMinState.addAttribute(
        "scale_bits", builder.getI64IntegerAttr(problem.scaleBits));
    scaleMinState.addAttribute(
        "k_scale_size", builder.getI64IntegerAttr(problem.scaleTableBytes));
    scaleMinState.addAttribute(
        "weight_scale_byte_offset",
        builder.getI64IntegerAttr(problem.weightScaleByteOffset));
    scaleMinState.addTypes({i32AccType, i32AccType});
    mlir::Operation *scaleMin = builder.create(scaleMinState);

    mlir::OperationState macState(
        loc, weft::ime::VmadotMacLeafOp::getOperationName());
    macState.addOperands(
        {aFragment, dequant->getResult(0), blockIndex, accScaleIn});
    macState.addAttribute(kIMEOpAttrName,
                          builder.getStringAttr(problem.imeOp));
    macState.addAttribute(kElemInBitsAttrName,
                          builder.getI64IntegerAttr(target.elemInBits));
    macState.addAttribute(kAccumBitsAttrName,
                          builder.getI64IntegerAttr(target.accumBits));
    macState.addAttribute(kMacMAttrName,
                          builder.getI64IntegerAttr(target.macM));
    macState.addAttribute(kMacNAttrName,
                          builder.getI64IntegerAttr(target.macN));
    macState.addAttribute(kMacKAttrName,
                          builder.getI64IntegerAttr(target.macK));
    macState.addTypes({i32AccType});
    mlir::Operation *mac = builder.create(macState);

    mlir::OperationState scaleAccumState(
        loc, weft::ime::Q4KScaleWeightedAccumOp::getOperationName());
    scaleAccumState.addOperands(
        {mac->getResult(0), scaleMin->getResult(0), accScaleIn});
    scaleAccumState.addAttribute(
        "accum_model", builder.getStringAttr(kQ4KScaleWeightedModel));
    scaleAccumState.addTypes({i32AccType});
    mlir::Operation *scaleAccum = builder.create(scaleAccumState);

    mlir::OperationState minBiasState(
        loc, weft::ime::Q4KMinBiasAccumOp::getOperationName());
    minBiasState.addOperands(
        {aFragment, scaleMin->getResult(1), accMinIn});
    minBiasState.addAttribute(
        "bias_model", builder.getStringAttr(kQ4KMinBiasModel));
    minBiasState.addTypes({i32AccType});
    mlir::Operation *minBias = builder.create(minBiasState);

    mlir::OperationState yieldState(
        loc, weft::ime::Q4KMatMulTileYieldOp::getOperationName());
    yieldState.addOperands(
        {scaleAccum->getResult(0), minBias->getResult(0)});
    builder.create(yieldState);
    return finishBody(tile);
  }

  const bool boundaryIsUnsigned =
      problem.signedness == kSignednessUnsigned;
  const bool boundaryIsMixedSign =
      problem.signedness == kSignednessMixedSign;
  const bool boundaryIsMixedSignUS =
      problem.signedness == kSignednessMixedSignUS;
  const bool boundaryIsSlide =
      problem.kind == IMEProblemKind::SlidingInt8MAC;
  llvm::StringRef boundaryOpName =
      boundaryIsSlide
          ? weft::ime::MMASlideOp::getOperationName()
          : (projection->isMatmul
                 ? weft::ime::MatMulOp::getOperationName()
                 : (boundaryIsMixedSign
                        ? weft::ime::MMASUOp::getOperationName()
                        : (boundaryIsMixedSignUS
                               ? weft::ime::MMAUSOp::getOperationName()
                               : (boundaryIsUnsigned
                                      ? weft::ime::MMAUOp::getOperationName()
                                      : weft::ime::MMAOp::getOperationName()))));

  mlir::OperationState state(variant.getLoc(), boundaryOpName);
  addCommonAttributes(state);
  if (projection->isMatmul)
    addMatrixAttributes(state);
  if (boundaryIsSlide)
    state.addAttribute(kSlideAttrName,
                       builder.getI64IntegerAttr(problem.slide));
  mlir::Operation *boundary = builder.create(state);
  return finishBody(boundary);
}

llvm::Error IMEExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  mlir::Operation *body = request.getConstructedOperation();
  if (!body || !isIMEFinalBody(body))
    return makeIMEPluginError(
        "selected IME boundary exposure requires the exact final body returned "
        "by family construction");

  VariantLoweringBoundaryValidationRequest validationRequest(
      request.getVariant(), request.getKernel(), request.getCapabilities(),
      request.getRole(), body);
  if (llvm::Error error = validateSelectedLoweringBoundary(validationRequest))
    return error;

  out = VariantLoweringBoundaryResult::getMaterialized(
      kIMEPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(), body);
  return llvm::Error::success();
}

llvm::Error IMEExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  // The boundary is the signed weft.ime.mma, the unsigned weft.ime.mma_u, the
  // mixed-sign weft.ime.mma_su / weft.ime.mma_us, the sliding-window
  // weft.ime.mma_slide, or the tiled weft.ime.matmul op — all are the IME plugin
  // execution surface. Accept any; each op's fail-closed verifier pins its own
  // correct mnemonic and envelope.
  mlir::Operation *boundaryOp = request.getBoundary();
  bool verifierFailed = false;
  if (auto mma = llvm::dyn_cast_if_present<weft::ime::MMAOp>(boundaryOp))
    verifierFailed = mlir::failed(mma.verify());
  else if (auto mmau = llvm::dyn_cast_if_present<weft::ime::MMAUOp>(boundaryOp))
    verifierFailed = mlir::failed(mmau.verify());
  else if (auto mmasu =
               llvm::dyn_cast_if_present<weft::ime::MMASUOp>(boundaryOp))
    verifierFailed = mlir::failed(mmasu.verify());
  else if (auto mmaus =
               llvm::dyn_cast_if_present<weft::ime::MMAUSOp>(boundaryOp))
    verifierFailed = mlir::failed(mmaus.verify());
  else if (auto mmaslide =
               llvm::dyn_cast_if_present<weft::ime::MMASlideOp>(boundaryOp))
    verifierFailed = mlir::failed(mmaslide.verify());
  else if (auto matmul =
               llvm::dyn_cast_if_present<weft::ime::MatMulOp>(boundaryOp))
    verifierFailed = mlir::failed(matmul.verify());
  else if (auto q40tile =
               llvm::dyn_cast_if_present<weft::ime::Q40MatMulTileOp>(boundaryOp))
    verifierFailed = mlir::failed(q40tile.verify());
  else if (auto q80tile =
               llvm::dyn_cast_if_present<weft::ime::Q80MatMulTileOp>(boundaryOp))
    verifierFailed = mlir::failed(q80tile.verify());
  else if (auto q4ktile =
               llvm::dyn_cast_if_present<weft::ime::Q4KMatMulTileOp>(boundaryOp))
    verifierFailed = mlir::failed(q4ktile.verify());
  else
    return makeIMEPluginError(
        "selected IME path requires a weft.ime.mma, weft.ime.mma_u, "
        "weft.ime.mma_su, weft.ime.mma_us, weft.ime.mma_slide, "
        "weft.ime.matmul, weft.ime.q4_0_matmul_tile, weft.ime.q8_0_matmul_tile, "
        "or weft.ime.q4_K_matmul_tile operation");

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

} // namespace weft::plugin
