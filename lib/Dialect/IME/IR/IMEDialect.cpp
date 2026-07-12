#include "Weft/Dialect/IME/IR/IMEDialect.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace weft::ime;

#include "Weft/Dialect/IME/IR/IMEOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/IME/IR/IMEOps.cpp.inc"

namespace {

constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kOriginAttrName("origin");
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
// Whole-matrix problem dims carried ONLY by weft.ime.matmul (the tiled op).
constexpr llvm::StringLiteral kMatMAttrName("mat_m");
constexpr llvm::StringLiteral kMatNAttrName("mat_n");
constexpr llvm::StringLiteral kMatKAttrName("mat_k");
// The sliding-window stride FACT carried ONLY by weft.ime.mma_slide (1/2/3).
constexpr llvm::StringLiteral kSlideAttrName("slide");
constexpr llvm::StringLiteral kAvailableHartsAttrName("available_harts");
constexpr llvm::StringLiteral kIMEReasonAttrName("ime_reason");

constexpr llvm::StringLiteral kIMEPluginName("ime-plugin");
constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");
constexpr llvm::StringLiteral kSourceRoleValue("compute");
// The validated IME1 int8->int32 envelope (FOUNDATION.md task 3). The verifier
// admits ONLY this envelope, so no body outside the proven hardware envelope
// is ever emitted (I7 fail-closed). weft.ime.mma is the SIGNED `vmadot`;
// weft.ime.mma_u is the UNSIGNED `vmadotu` — both IME1 signedness forms over the
// SAME elem_in/accum/MAC-fragment envelope, distinguished only by the mnemonic.
constexpr llvm::StringLiteral kExpectedSignedIMEOp("vmadot");
constexpr llvm::StringLiteral kExpectedUnsignedIMEOp("vmadotu");
// weft.ime.mma_su is the MIXED-SIGN `vmadotsu` (signed*unsigned int8 MAC) — the
// fourth IME1 signedness form over the SAME elem_in/accum/MAC-fragment envelope,
// distinguished only by the mnemonic.
constexpr llvm::StringLiteral kExpectedMixedSignIMEOp("vmadotsu");
// weft.ime.mma_us is the REVERSED-ORDER MIXED-SIGN `vmadotus` (unsigned A *
// signed B int8 MAC) — the fourth signedness form, completing the family over
// the SAME elem_in/accum/MAC-fragment envelope, distinguished only by the
// mnemonic.
constexpr llvm::StringLiteral kExpectedMixedSignUSIMEOp("vmadotus");
// weft.ime.mma_slide is the SLIDING-WINDOW family `vmadot{1,2,3}` (Xsmti8i32mm_slide,
// funct7 111001 / e6..., DISTINCT from the non-slide 111000 / e2...). The expected
// mnemonic is selected by the `slide` FACT (1=>vmadot1, 2=>vmadot2, 3=>vmadot3) over
// the SAME elem_in/accum/MAC-fragment envelope as the non-slide MAC.
constexpr llvm::StringLiteral kExpectedSlide1IMEOp("vmadot1");
constexpr llvm::StringLiteral kExpectedSlide2IMEOp("vmadot2");
constexpr llvm::StringLiteral kExpectedSlide3IMEOp("vmadot3");
constexpr int64_t kExpectedElemInBits = 8;
constexpr int64_t kExpectedAccumBits = 32;

constexpr llvm::StringLiteral kDirectVariantRoleValue("direct variant");
constexpr llvm::StringLiteral kDispatchCaseRoleValue("dispatch case");

// G4 M1a: the format-keyed q4_0 IME GEMM tile facts. The tile op carries the
// SAME int8->int32 vmadot MAC envelope PLUS the q4_0 weight-format facts, and its
// typed region carries exactly the three decomposed bricks below.
constexpr llvm::StringLiteral kWeightFormatAttrName("weight_format");
constexpr llvm::StringLiteral kQkAttrName("qk");
constexpr llvm::StringLiteral kWeightBlockStrideAttrName("weight_block_stride");
constexpr llvm::StringLiteral kWeightQuantByteOffsetAttrName(
    "weight_quant_byte_offset");
// The ONLY q4_0 weight format + decode model modeled in M1 (fail-closed, I7).
constexpr llvm::StringLiteral kExpectedQ4_0Format("q4_0");
constexpr llvm::StringLiteral kExpectedQ4_0DecodeModel(
    "q4_0_offset_binary_nibble");
// ggml q4_0 block: fp16 d (2B) + 32 packed nibbles (16B) = 18B, 32 weights/block.
constexpr int64_t kExpectedQ4_0Qk = 32;
constexpr int64_t kExpectedQ4_0WeightBlockStride = 18;
constexpr int64_t kExpectedQ4_0WeightQuantByteOffset = 2;

// G4 M2: the format-keyed q8_0 IME GEMM tile facts (the FLAT-int8 sibling of
// q4_0). The ONLY q8_0 weight format + decode model modeled (fail-closed, I7).
constexpr llvm::StringLiteral kExpectedQ8_0Format("q8_0");
constexpr llvm::StringLiteral kExpectedQ8_0DecodeModel("q8_0_direct_int8");
// ggml q8_0 block: fp16 d (2B) + 32 int8 quants (32B) = 34B, 32 weights/block.
constexpr int64_t kExpectedQ8_0Qk = 32;
constexpr int64_t kExpectedQ8_0WeightBlockStride = 34;
constexpr int64_t kExpectedQ8_0WeightQuantByteOffset = 2;
// G4 M2b: the format-keyed q4_K IME GEMM tile facts (the SUPER-BLOCK K-quant
// sibling of q4_0/q8_0). q4_K needs the TWO-LEVEL 6-bit scale/min fold, so beyond
// the reused vmadot MAC leaf it carries three NEW bricks (raw-nibble decode +
// 6-bit scale/min unpack + scale-weighted / min-bias accumulate). Fail-closed (I7).
constexpr llvm::StringLiteral kExpectedQ4_KFormat("q4_K");
constexpr llvm::StringLiteral kExpectedQ4_KDecodeModel("q4_K_raw_nibble");
constexpr llvm::StringLiteral kExpectedQ4_KScaleMinModel("get_scale_min_k4");
constexpr llvm::StringLiteral kExpectedQ4_KScaleWeightedModel("scale_weighted_sum");
constexpr llvm::StringLiteral kExpectedQ4_KMinBiasModel("activation_sum_min_bias");
// ggml q4_K super-block: fp16 d (2B) + fp16 dmin (2B) + 12B packed 6-bit
// scales/mins + 128B 4-bit quants = 144B, 256 weights / 8 sub-blocks per block.
constexpr int64_t kExpectedQ4_KQk = 256;
constexpr int64_t kExpectedQ4_KWeightBlockStride = 144;
constexpr int64_t kExpectedQ4_KWeightQuantByteOffset = 16;
constexpr int64_t kExpectedQ4_KWeightScaleByteOffset = 4;
constexpr int64_t kExpectedQ4_KNumSubBlocks = 8;
constexpr int64_t kExpectedQ4_KScaleBits = 6;
constexpr int64_t kExpectedQ4_KKScaleSize = 12;
// The int8 4x8 MAC fragment carries 32 int8; the int32 4x4 output tile 16 int32.
constexpr int64_t kIMEBFragmentLanes = 32;
constexpr int64_t kIMEAccTileLanes = 16;

/// True iff `type` is a builtin rank-1 vector<`lanes` x i`bitwidth`>.
bool isIntVectorOfLanes(mlir::Type type, int64_t lanes, unsigned bitwidth) {
  auto vectorType = llvm::dyn_cast<mlir::VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1 ||
      vectorType.getNumElements() != lanes)
    return false;
  auto elementType =
      llvm::dyn_cast<mlir::IntegerType>(vectorType.getElementType());
  return elementType && elementType.getWidth() == bitwidth;
}

bool isAllowedRole(llvm::StringRef role) {
  return role == kDirectVariantRoleValue || role == kDispatchCaseRoleValue;
}

bool isAllowedMMAAttr(llvm::StringRef attrName) {
  return attrName == kSourceKernelAttrName ||
         attrName == kSelectedVariantAttrName || attrName == kOriginAttrName ||
         attrName == kRoleAttrName || attrName == kStatusAttrName ||
         attrName == kRequiredCapabilitiesAttrName ||
         attrName == kIMEOpAttrName || attrName == kElemInBitsAttrName ||
         attrName == kAccumBitsAttrName || attrName == kMacMAttrName ||
         attrName == kMacNAttrName || attrName == kMacKAttrName ||
         attrName == kAvailableHartsAttrName || attrName == kIMEReasonAttrName;
}

// The tiled whole-matrix op admits the same envelope PLUS the problem dims.
bool isAllowedMatMulAttr(llvm::StringRef attrName) {
  return isAllowedMMAAttr(attrName) || attrName == kMatMAttrName ||
         attrName == kMatNAttrName || attrName == kMatKAttrName;
}

// The sliding-window op admits the same envelope PLUS the slide stride fact.
bool isAllowedMMASlideAttr(llvm::StringRef attrName) {
  return isAllowedMMAAttr(attrName) || attrName == kSlideAttrName;
}

// The q4_0 tile op admits the shared MAC envelope + the whole-matrix problem
// dims + the q4_0 weight-format facts.
bool isAllowedQ4_0TileAttr(llvm::StringRef attrName) {
  return isAllowedMMAAttr(attrName) || attrName == kMatMAttrName ||
         attrName == kMatNAttrName || attrName == kMatKAttrName ||
         attrName == kWeightFormatAttrName || attrName == kQkAttrName ||
         attrName == kWeightBlockStrideAttrName ||
         attrName == kWeightQuantByteOffsetAttrName;
}

// G4 M2: the q8_0 tile op admits the SAME attribute set as the q4_0 tile (the
// shared MAC envelope + whole-matrix problem dims + the weight-format facts).
bool isAllowedQ8_0TileAttr(llvm::StringRef attrName) {
  return isAllowedQ4_0TileAttr(attrName);
}

// G4 M2b: the q4_K tile op admits the SAME attribute set as the q4_0/q8_0 tiles
// (the shared MAC envelope + whole-matrix problem dims + the weight-format facts;
// the q4_K-specific super-block facts ride on the region bricks, not the tile op).
bool isAllowedQ4_KTileAttr(llvm::StringRef attrName) {
  return isAllowedQ4_0TileAttr(attrName);
}

bool hasMissingOrEmptyStringAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  return !attr || attr.getValue().trim().empty();
}

bool containsExecutableClaimWording(llvm::StringRef text) {
  std::string lowerText = text.lower();
  llvm::StringRef lower(lowerText);
  return lower.contains("benchmark") || lower.contains("performance evidence") ||
         lower.contains("correctness evidence") ||
         lower.contains("faster than") || lower.contains("speedup");
}

mlir::LogicalResult verifySelectedPathBinding(mlir::Operation *op,
                                              mlir::InFlightDiagnostic &diag) {
  auto selectedVariant =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
  if (!selectedVariant || selectedVariant.getValue().trim().empty()) {
    diag << "requires non-empty variant symbol reference attribute '"
         << kSelectedVariantAttrName << "'";
    return mlir::failure();
  }

  auto requiredCapabilities =
      op->getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName);
  if (!requiredCapabilities || requiredCapabilities.empty()) {
    diag << "requires non-empty array attribute '"
         << kRequiredCapabilitiesAttrName
         << "' containing capability symbol references";
    return mlir::failure();
  }

  auto kernel = op->getParentOfType<weft::exec::KernelOp>();
  if (!kernel) {
    diag << "must be nested directly in a weft.exec.kernel";
    return mlir::failure();
  }
  if (op->getParentOp() != kernel.getOperation()) {
    diag << "must be a direct child of the enclosing weft.exec.kernel";
    return mlir::failure();
  }

  auto sourceKernel =
      op->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  if (sourceKernel.getValue() != kernel.getSymName()) {
    diag << "source_kernel must match enclosing weft.exec.kernel symbol @"
         << kernel.getSymName();
    return mlir::failure();
  }

  if (kernel.getBody().empty()) {
    diag << "requires enclosing weft.exec.kernel to have a body block";
    return mlir::failure();
  }

  llvm::Expected<weft::support::TargetCapabilitySet> capabilitiesOrError =
      weft::support::TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilitiesOrError) {
    std::string message = llvm::toString(capabilitiesOrError.takeError());
    diag << message;
    return mlir::failure();
  }
  const weft::support::TargetCapabilitySet &capabilities =
      *capabilitiesOrError;

  weft::exec::VariantOp resolvedVariant;
  for (mlir::Operation &sibling : kernel.getBody().front()) {
    if (auto variant =
            llvm::dyn_cast<weft::exec::VariantOp>(sibling)) {
      if (variant.getSymName() == selectedVariant.getValue()) {
        resolvedVariant = variant;
        break;
      }
    }
  }
  if (!resolvedVariant) {
    diag << "selected_variant @" << selectedVariant.getValue()
         << " must resolve to a direct sibling weft.exec.variant in the "
            "enclosing weft.exec.kernel";
    return mlir::failure();
  }

  for (mlir::Attribute requiredCapability : requiredCapabilities) {
    auto symbolRef = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef) {
      diag << "attribute '" << kRequiredCapabilitiesAttrName
           << "' must contain only capability symbol references";
      return mlir::failure();
    }
    if (!capabilities.lookupBySymbolName(symbolRef.getValue())) {
      diag << "requires unknown capability @" << symbolRef.getValue()
           << " in enclosing weft.exec.kernel";
      return mlir::failure();
    }
  }

  auto variantRequires =
      resolvedVariant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires != requiredCapabilities) {
    diag << "required_capabilities must match selected variant requires "
            "metadata";
    return mlir::failure();
  }

  return mlir::success();
}

} // namespace

/// The shared fail-closed (I7) verifier for an IME MAC boundary op. The ONLY
/// difference between weft.ime.mma (signed `vmadot`) and weft.ime.mma_u
/// (unsigned `vmadotu`) is the admitted mnemonic; everything else (the
/// int8->int32 envelope, the capability-derived MAC fragment shape, the
/// selected-path binding, the no-benchmark-claim rule) is identical, so the
/// signedness is the only structural axis that varies. `emitErr` is the op's own
/// `emitOpError` so diagnostics are attributed to the concrete op.
mlir::LogicalResult
verifyIMEMACBoundary(mlir::Operation *op, llvm::StringRef expectedIMEOp,
                     llvm::function_ref<bool(llvm::StringRef)> isAllowedAttr,
                     llvm::function_ref<mlir::InFlightDiagnostic()> emitErr) {
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    if (!isAllowedAttr(attr.getName().getValue()))
      return emitErr()
             << "does not accept generic tensor/tile/benchmark or unknown "
                "attribute '"
             << attr.getName().getValue() << "'";
  }

  for (llvm::StringRef attrName :
       {kSourceKernelAttrName, kOriginAttrName, kRoleAttrName, kStatusAttrName,
        kIMEOpAttrName, kAvailableHartsAttrName}) {
    if (hasMissingOrEmptyStringAttr(op, attrName))
      return emitErr() << "requires non-empty string attribute '" << attrName
                       << "'";
  }

  auto origin = op->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (origin.getValue() != kIMEPluginName)
    return emitErr() << "origin must be '" << kIMEPluginName
                     << "' because this is the IME plugin execution surface";

  auto role = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  if (!isAllowedRole(role.getValue()))
    return emitErr() << "role must be '" << kDirectVariantRoleValue << "' or '"
                     << kDispatchCaseRoleValue << "'";

  auto status = op->getAttrOfType<mlir::StringAttr>(kStatusAttrName);
  if (status.getValue() != kRoleOpBoundaryStatusValue)
    return emitErr() << "status must be '" << kRoleOpBoundaryStatusValue
                     << "' because this is an ODS role-op boundary";

  // Fail-closed (I7): admit ONLY the validated IME1 int8->int32 MAC instruction
  // of the EXPECTED signedness for this op (vmadot for the signed surface,
  // vmadotu for the unsigned surface).
  auto imeOp = op->getAttrOfType<mlir::StringAttr>(kIMEOpAttrName);
  if (imeOp.getValue() != expectedIMEOp)
    return emitErr() << "ime_op must be '" << expectedIMEOp
                     << "'; this op only models the validated IME1 "
                        "int8->int32 MAC instruction of its signedness";

  auto elemInBits = op->getAttrOfType<mlir::IntegerAttr>(kElemInBitsAttrName);
  if (!elemInBits || elemInBits.getInt() != kExpectedElemInBits)
    return emitErr() << "elem_in_bits must be " << kExpectedElemInBits
                     << " (IME1 " << expectedIMEOp << " consumes int8 inputs)";

  auto accumBits = op->getAttrOfType<mlir::IntegerAttr>(kAccumBitsAttrName);
  if (!accumBits || accumBits.getInt() != kExpectedAccumBits)
    return emitErr() << "accum_bits must be " << kExpectedAccumBits
                     << " (IME1 " << expectedIMEOp
                     << " accumulates in int32)";

  auto macM = op->getAttrOfType<mlir::IntegerAttr>(kMacMAttrName);
  auto macN = op->getAttrOfType<mlir::IntegerAttr>(kMacNAttrName);
  auto macK = op->getAttrOfType<mlir::IntegerAttr>(kMacKAttrName);
  if (!macM || !macN || !macK)
    return emitErr()
           << "requires the capability-derived MAC fragment shape integer "
              "attributes '"
           << kMacMAttrName << "', '" << kMacNAttrName << "', '" << kMacKAttrName
           << "'";
  // The K depth is the int8 lane count of one VLEN=256/SEW=8 group reduced by
  // the 4x4 output tile; for the validated X60 unit M=N=4, K=8. Keep the
  // verifier general over fragment shape but anchored to the int8->int32 MAC:
  // every fragment dimension must be strictly positive (derived from VLEN/SEW).
  if (macM.getInt() <= 0 || macN.getInt() <= 0 || macK.getInt() <= 0)
    return emitErr() << "MAC fragment shape (mac_m/mac_n/mac_k) must be "
                        "positive (derived from VLEN/SEW)";

  if (auto reason = op->getAttrOfType<mlir::StringAttr>(kIMEReasonAttrName)) {
    if (reason.getValue().trim().empty())
      return emitErr() << "ime_reason must be non-empty when present";
    if (containsExecutableClaimWording(reason.getValue()))
      return emitErr() << "ime_reason must not claim benchmark or "
                          "performance evidence in the IR";
  }

  mlir::InFlightDiagnostic diag = emitErr();
  if (mlir::failed(verifySelectedPathBinding(op, diag)))
    return mlir::failure();
  diag.abandon();

  return mlir::success();
}

llvm::StringRef MMAOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MMAOp::getWEFTEmitCLowerableSourceRole() {
  return kSourceRoleValue;
}

mlir::LogicalResult MMAOp::verify() {
  return verifyIMEMACBoundary(getOperation(), kExpectedSignedIMEOp,
                              isAllowedMMAAttr,
                              [this]() { return emitOpError(); });
}

llvm::StringRef MMAUOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MMAUOp::getWEFTEmitCLowerableSourceRole() {
  return kSourceRoleValue;
}

mlir::LogicalResult MMAUOp::verify() {
  return verifyIMEMACBoundary(getOperation(), kExpectedUnsignedIMEOp,
                              isAllowedMMAAttr,
                              [this]() { return emitOpError(); });
}

llvm::StringRef MMASUOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MMASUOp::getWEFTEmitCLowerableSourceRole() {
  return kSourceRoleValue;
}

mlir::LogicalResult MMASUOp::verify() {
  return verifyIMEMACBoundary(getOperation(), kExpectedMixedSignIMEOp,
                              isAllowedMMAAttr,
                              [this]() { return emitOpError(); });
}

llvm::StringRef MMAUSOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MMAUSOp::getWEFTEmitCLowerableSourceRole() {
  return kSourceRoleValue;
}

mlir::LogicalResult MMAUSOp::verify() {
  return verifyIMEMACBoundary(getOperation(), kExpectedMixedSignUSIMEOp,
                              isAllowedMMAAttr,
                              [this]() { return emitOpError(); });
}

llvm::StringRef MMASlideOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MMASlideOp::getWEFTEmitCLowerableSourceRole() {
  return kSourceRoleValue;
}

mlir::LogicalResult MMASlideOp::verify() {
  // Fail-closed (I7): the slide stride is the capability-derived window FACT.
  // Only slide in {1,2,3} is modeled (slide=0 is the existing non-slide MAC; >=4
  // is outside the documented vmadot1..3 slide family). Reject before the shared
  // envelope verifier so the expected mnemonic can be selected from the slide.
  int64_t slide = getSlide();
  if (slide < 1 || slide > 3)
    return emitOpError() << "slide must be in {1,2,3} (vmadot1/vmadot2/vmadot3); "
                            "slide=0 is the non-slide weft.ime.mma and slide>=4 is "
                            "outside the documented IME1 slide family (fail-closed, "
                            "I7)";
  llvm::StringRef expectedIMEOp = slide == 1   ? kExpectedSlide1IMEOp
                                  : slide == 2 ? kExpectedSlide2IMEOp
                                               : kExpectedSlide3IMEOp;
  return verifyIMEMACBoundary(getOperation(), expectedIMEOp,
                              isAllowedMMASlideAttr,
                              [this]() { return emitOpError(); });
}

llvm::StringRef MatMulOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef MatMulOp::getWEFTEmitCLowerableSourceRole() {
  return kSourceRoleValue;
}

mlir::LogicalResult MatMulOp::verify() {
  // The tiled whole-matrix op accepts EITHER signedness (signed vmadot or
  // unsigned vmadotu); the per-op signed/unsigned axis is the ime_op fact.
  llvm::StringRef imeOp = getImeOp();
  llvm::StringRef expectedIMEOp =
      imeOp == kExpectedUnsignedIMEOp ? kExpectedUnsignedIMEOp
                                      : kExpectedSignedIMEOp;
  if (mlir::failed(verifyIMEMACBoundary(getOperation(), expectedIMEOp,
                                        isAllowedMatMulAttr,
                                        [this]() { return emitOpError(); })))
    return mlir::failure();

  // Whole-matrix problem dims: present, positive, and a whole multiple of the
  // capability-derived MAC fragment (fail-closed: NO remainder path). mac_*
  // were already validated positive by the shared envelope verifier above.
  auto matM = getOperation()->getAttrOfType<mlir::IntegerAttr>(kMatMAttrName);
  auto matN = getOperation()->getAttrOfType<mlir::IntegerAttr>(kMatNAttrName);
  auto matK = getOperation()->getAttrOfType<mlir::IntegerAttr>(kMatKAttrName);
  if (!matM || !matN || !matK)
    return emitOpError()
           << "requires the whole-matrix problem-dim integer attributes '"
           << kMatMAttrName << "', '" << kMatNAttrName << "', '" << kMatKAttrName
           << "'";
  if (matM.getInt() <= 0 || matN.getInt() <= 0 || matK.getInt() <= 0)
    return emitOpError() << "problem dims (mat_m/mat_n/mat_k) must be positive";

  int64_t macM = getMacM(), macN = getMacN(), macK = getMacK();
  if (matM.getInt() % macM != 0 || matN.getInt() % macN != 0 ||
      matK.getInt() % macK != 0)
    return emitOpError()
           << "problem dims (mat_m=" << matM.getInt()
           << ", mat_n=" << matN.getInt() << ", mat_k=" << matK.getInt()
           << ") must each be a whole multiple of the MAC fragment (mac_m="
           << macM << ", mac_n=" << macN << ", mac_k=" << macK
           << "); no remainder path is emitted (fail-closed, I7)";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// G4 M1a: format-keyed q4_0 IME GEMM tile typed-region verifiers.
//===----------------------------------------------------------------------===//

mlir::LogicalResult Q40DequantCoreOp::verify() {
  // Fail-closed (I7): only the q4_0 offset-binary nibble decode is modeled.
  if (getDecodeModel() != kExpectedQ4_0DecodeModel)
    return emitOpError() << "decode_model must be '" << kExpectedQ4_0DecodeModel
                         << "' (the only modeled q4_0 weight decode)";
  if (getQk() != kExpectedQ4_0Qk)
    return emitOpError() << "qk must be " << kExpectedQ4_0Qk
                         << " (a ggml q4_0 block carries 32 weights)";
  if (getWeightBlockStride() != kExpectedQ4_0WeightBlockStride)
    return emitOpError() << "weight_block_stride must be "
                         << kExpectedQ4_0WeightBlockStride
                         << " (fp16 d + 16 nibble bytes)";
  if (getWeightQuantByteOffset() != kExpectedQ4_0WeightQuantByteOffset)
    return emitOpError() << "weight_quant_byte_offset must be "
                         << kExpectedQ4_0WeightQuantByteOffset
                         << " (the nibbles follow the 2-byte fp16 d)";
  if (getWeightScaleByteOffset() != 0)
    return emitOpError()
           << "weight_scale_byte_offset must be 0 (the fp16 d leads the block)";
  if (!isIntVectorOfLanes(getBFragment().getType(), kIMEBFragmentLanes, 8))
    return emitOpError() << "decoded b_fragment must be vector<"
                         << kIMEBFragmentLanes << "xi8> (the 4x8 int8 MAC "
                            "fragment; nibble - 8 fits int8 exactly)";
  return mlir::success();
}

mlir::LogicalResult VmadotMacLeafOp::verify() {
  // Fail-closed (I7): only the validated IME1 signed int8->int32 vmadot MAC.
  if (getImeOp() != kExpectedSignedIMEOp)
    return emitOpError() << "ime_op must be '" << kExpectedSignedIMEOp
                         << "' (the modeled IME1 int8->int32 signed MAC)";
  if (getElemInBits() != kExpectedElemInBits)
    return emitOpError() << "elem_in_bits must be " << kExpectedElemInBits;
  if (getAccumBits() != kExpectedAccumBits)
    return emitOpError() << "accum_bits must be " << kExpectedAccumBits
                         << " (int32-exact accumulate)";
  if (getMacM() <= 0 || getMacN() <= 0 || getMacK() <= 0)
    return emitOpError() << "MAC fragment shape (mac_m/mac_n/mac_k) must be "
                            "positive (derived from VLEN/SEW)";
  if (!isIntVectorOfLanes(getAFragment().getType(), kIMEBFragmentLanes, 8))
    return emitOpError() << "a_fragment must be vector<" << kIMEBFragmentLanes
                         << "xi8> (the 4x8 int8 activation fragment)";
  if (!isIntVectorOfLanes(getBFragment().getType(), kIMEBFragmentLanes, 8))
    return emitOpError() << "b_fragment must be vector<" << kIMEBFragmentLanes
                         << "xi8> (the decoded 4x8 int8 weight fragment)";
  if (!isIntVectorOfLanes(getAccIn().getType(), kIMEAccTileLanes, 32) ||
      !isIntVectorOfLanes(getAccOut().getType(), kIMEAccTileLanes, 32))
    return emitOpError() << "acc_in/acc_out must be vector<" << kIMEAccTileLanes
                         << "xi32> (the 4x4 int32 accumulator tile)";
  return mlir::success();
}

mlir::LogicalResult Q40MatMulTileYieldOp::verify() {
  if (getAccOut().empty())
    return emitOpError()
           << "must name at least one carried-out int32 accumulator tile";
  for (mlir::Value acc : getAccOut())
    if (!isIntVectorOfLanes(acc.getType(), kIMEAccTileLanes, 32))
      return emitOpError() << "every carried-out accumulator must be vector<"
                           << kIMEAccTileLanes << "xi32>";
  return mlir::success();
}

llvm::StringRef Q40MatMulTileOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef Q40MatMulTileOp::getWEFTEmitCLowerableSourceRole() {
  return kSourceRoleValue;
}

mlir::LogicalResult Q40MatMulTileOp::verify() {
  // The int8->int32 vmadot MAC envelope + selected-path binding (signed vmadot).
  if (mlir::failed(verifyIMEMACBoundary(getOperation(), kExpectedSignedIMEOp,
                                        isAllowedQ4_0TileAttr,
                                        [this]() { return emitOpError(); })))
    return mlir::failure();

  // The q4_0 weight-format facts (fail-closed: only q4_0 is modeled in M1).
  if (getWeightFormat() != kExpectedQ4_0Format)
    return emitOpError() << "weight_format must be '" << kExpectedQ4_0Format
                         << "' (the only format-keyed IME tile modeled in M1)";
  if (getQk() != kExpectedQ4_0Qk)
    return emitOpError() << "qk must be " << kExpectedQ4_0Qk;
  if (getWeightBlockStride() != kExpectedQ4_0WeightBlockStride)
    return emitOpError() << "weight_block_stride must be "
                         << kExpectedQ4_0WeightBlockStride;
  if (getWeightQuantByteOffset() != kExpectedQ4_0WeightQuantByteOffset)
    return emitOpError() << "weight_quant_byte_offset must be "
                         << kExpectedQ4_0WeightQuantByteOffset;

  // Whole-matrix problem dims: present, positive, whole multiples of the MAC
  // fragment (fail-closed: NO remainder path).
  int64_t matM = getMatM(), matN = getMatN(), matK = getMatK();
  if (matM <= 0 || matN <= 0 || matK <= 0)
    return emitOpError() << "problem dims (mat_m/mat_n/mat_k) must be positive";
  int64_t macM = getMacM(), macN = getMacN(), macK = getMacK();
  if (matM % macM != 0 || matN % macN != 0 || matK % macK != 0)
    return emitOpError()
           << "problem dims must each be a whole multiple of the MAC fragment "
              "(no remainder path is emitted, I7)";
  // The K dimension must partition into whole q4_0 blocks (qk=32).
  if (matK % kExpectedQ4_0Qk != 0)
    return emitOpError() << "mat_k=" << matK
                         << " must be a whole multiple of qk=" << kExpectedQ4_0Qk
                         << " (whole q4_0 blocks; no remainder path)";

  // Fail-closed region SHAPE: the single-block region must be exactly the three
  // decomposed typed bricks -- q4_0_dequant_core, vmadot_mac_leaf, and the yield
  // terminator -- and NO opaque hand helper (the M-FLAT construction discipline:
  // the front door CONSTRUCTS a real typed region, not an opaque body).
  mlir::Region &body = getBody();
  if (!body.hasOneBlock())
    return emitOpError() << "typed region must have exactly one block";
  mlir::Block &block = body.front();
  auto dequantCores = block.getOps<Q40DequantCoreOp>();
  auto macLeaves = block.getOps<VmadotMacLeafOp>();
  if (std::distance(dequantCores.begin(), dequantCores.end()) != 1)
    return emitOpError() << "typed region must contain exactly one "
                            "weft.ime.q4_0_dequant_core weight-decode brick";
  if (std::distance(macLeaves.begin(), macLeaves.end()) != 1)
    return emitOpError() << "typed region must contain exactly one "
                            "weft.ime.vmadot_mac_leaf MAC brick";
  if (!llvm::isa<Q40MatMulTileYieldOp>(block.getTerminator()))
    return emitOpError() << "typed region must be terminated by "
                            "weft.ime.q4_0_matmul_tile_yield";
  // No brick outside the three-op decomposed vocabulary (anti-opaque).
  for (mlir::Operation &nested : block) {
    if (!llvm::isa<Q40DequantCoreOp, VmadotMacLeafOp, Q40MatMulTileYieldOp>(
            nested))
      return emitOpError() << "typed region admits ONLY the decomposed q4_0 "
                              "dequant / vmadot-leaf / yield bricks; found '"
                           << nested.getName().getStringRef() << "'";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// G4 M2: format-keyed q8_0 IME GEMM tile typed-region verifiers (the FLAT-int8
// copy-adapt siblings of the q4_0 verifiers above).
//===----------------------------------------------------------------------===//

mlir::LogicalResult Q80DequantCoreOp::verify() {
  // Fail-closed (I7): only the q8_0 DIRECT int8 read decode is modeled.
  if (getDecodeModel() != kExpectedQ8_0DecodeModel)
    return emitOpError() << "decode_model must be '" << kExpectedQ8_0DecodeModel
                         << "' (the only modeled q8_0 weight decode)";
  if (getQk() != kExpectedQ8_0Qk)
    return emitOpError() << "qk must be " << kExpectedQ8_0Qk
                         << " (a ggml q8_0 block carries 32 weights)";
  if (getWeightBlockStride() != kExpectedQ8_0WeightBlockStride)
    return emitOpError() << "weight_block_stride must be "
                         << kExpectedQ8_0WeightBlockStride
                         << " (fp16 d + 32 int8 quant bytes)";
  if (getWeightQuantByteOffset() != kExpectedQ8_0WeightQuantByteOffset)
    return emitOpError() << "weight_quant_byte_offset must be "
                         << kExpectedQ8_0WeightQuantByteOffset
                         << " (the int8 quants follow the 2-byte fp16 d)";
  if (getWeightScaleByteOffset() != 0)
    return emitOpError()
           << "weight_scale_byte_offset must be 0 (the fp16 d leads the block)";
  if (!isIntVectorOfLanes(getBFragment().getType(), kIMEBFragmentLanes, 8))
    return emitOpError() << "decoded b_fragment must be vector<"
                         << kIMEBFragmentLanes << "xi8> (the 4x8 int8 MAC "
                            "fragment; the direct int8 quant fits exactly)";
  return mlir::success();
}

mlir::LogicalResult Q80MatMulTileYieldOp::verify() {
  if (getAccOut().empty())
    return emitOpError()
           << "must name at least one carried-out int32 accumulator tile";
  for (mlir::Value acc : getAccOut())
    if (!isIntVectorOfLanes(acc.getType(), kIMEAccTileLanes, 32))
      return emitOpError() << "every carried-out accumulator must be vector<"
                           << kIMEAccTileLanes << "xi32>";
  return mlir::success();
}

llvm::StringRef Q80MatMulTileOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef Q80MatMulTileOp::getWEFTEmitCLowerableSourceRole() {
  return kSourceRoleValue;
}

mlir::LogicalResult Q80MatMulTileOp::verify() {
  // The int8->int32 vmadot MAC envelope + selected-path binding (signed vmadot).
  if (mlir::failed(verifyIMEMACBoundary(getOperation(), kExpectedSignedIMEOp,
                                        isAllowedQ8_0TileAttr,
                                        [this]() { return emitOpError(); })))
    return mlir::failure();

  // The q8_0 weight-format facts (fail-closed: only q8_0 is modeled here).
  if (getWeightFormat() != kExpectedQ8_0Format)
    return emitOpError() << "weight_format must be '" << kExpectedQ8_0Format
                         << "' (this is the q8_0 format-keyed IME tile)";
  if (getQk() != kExpectedQ8_0Qk)
    return emitOpError() << "qk must be " << kExpectedQ8_0Qk;
  if (getWeightBlockStride() != kExpectedQ8_0WeightBlockStride)
    return emitOpError() << "weight_block_stride must be "
                         << kExpectedQ8_0WeightBlockStride;
  if (getWeightQuantByteOffset() != kExpectedQ8_0WeightQuantByteOffset)
    return emitOpError() << "weight_quant_byte_offset must be "
                         << kExpectedQ8_0WeightQuantByteOffset;

  // Whole-matrix problem dims: present, positive, whole multiples of the MAC
  // fragment (fail-closed: NO remainder path).
  int64_t matM = getMatM(), matN = getMatN(), matK = getMatK();
  if (matM <= 0 || matN <= 0 || matK <= 0)
    return emitOpError() << "problem dims (mat_m/mat_n/mat_k) must be positive";
  int64_t macM = getMacM(), macN = getMacN(), macK = getMacK();
  if (matM % macM != 0 || matN % macN != 0 || matK % macK != 0)
    return emitOpError()
           << "problem dims must each be a whole multiple of the MAC fragment "
              "(no remainder path is emitted, I7)";
  // The K dimension must partition into whole q8_0 blocks (qk=32).
  if (matK % kExpectedQ8_0Qk != 0)
    return emitOpError() << "mat_k=" << matK
                         << " must be a whole multiple of qk=" << kExpectedQ8_0Qk
                         << " (whole q8_0 blocks; no remainder path)";

  // Fail-closed region SHAPE: the single-block region must be exactly the three
  // decomposed typed bricks -- q8_0_dequant_core, vmadot_mac_leaf, and the yield
  // terminator -- and NO opaque hand helper.
  mlir::Region &body = getBody();
  if (!body.hasOneBlock())
    return emitOpError() << "typed region must have exactly one block";
  mlir::Block &block = body.front();
  auto dequantCores = block.getOps<Q80DequantCoreOp>();
  auto macLeaves = block.getOps<VmadotMacLeafOp>();
  if (std::distance(dequantCores.begin(), dequantCores.end()) != 1)
    return emitOpError() << "typed region must contain exactly one "
                            "weft.ime.q8_0_dequant_core weight-decode brick";
  if (std::distance(macLeaves.begin(), macLeaves.end()) != 1)
    return emitOpError() << "typed region must contain exactly one "
                            "weft.ime.vmadot_mac_leaf MAC brick";
  if (!llvm::isa<Q80MatMulTileYieldOp>(block.getTerminator()))
    return emitOpError() << "typed region must be terminated by "
                            "weft.ime.q8_0_matmul_tile_yield";
  // No brick outside the three-op decomposed vocabulary (anti-opaque).
  for (mlir::Operation &nested : block) {
    if (!llvm::isa<Q80DequantCoreOp, VmadotMacLeafOp, Q80MatMulTileYieldOp>(
            nested))
      return emitOpError() << "typed region admits ONLY the decomposed q8_0 "
                              "dequant / vmadot-leaf / yield bricks; found '"
                           << nested.getName().getStringRef() << "'";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// G4 M2b: format-keyed q4_K IME GEMM tile typed-region verifiers (the SUPER-BLOCK
// K-quant tile; the DEDICATED effort with the two-level 6-bit scale/min fold).
//===----------------------------------------------------------------------===//

mlir::LogicalResult Q4KDequantCoreOp::verify() {
  // Fail-closed (I7): only the q4_K RAW-nibble (unsigned) decode is modeled.
  if (getDecodeModel() != kExpectedQ4_KDecodeModel)
    return emitOpError() << "decode_model must be '" << kExpectedQ4_KDecodeModel
                         << "' (the only modeled q4_K weight decode; raw unsigned "
                            "nibble, NOT q4_0 offset-binary)";
  if (getQk() != kExpectedQ4_KQk)
    return emitOpError() << "qk must be " << kExpectedQ4_KQk
                         << " (a ggml q4_K super-block carries 256 weights)";
  if (getWeightBlockStride() != kExpectedQ4_KWeightBlockStride)
    return emitOpError() << "weight_block_stride must be "
                         << kExpectedQ4_KWeightBlockStride
                         << " (fp16 d + fp16 dmin + 12B scales + 128B nibbles)";
  if (getWeightQuantByteOffset() != kExpectedQ4_KWeightQuantByteOffset)
    return emitOpError() << "weight_quant_byte_offset must be "
                         << kExpectedQ4_KWeightQuantByteOffset
                         << " (the nibbles follow d/dmin + the 12-byte scales)";
  if (getWeightScaleByteOffset() != kExpectedQ4_KWeightScaleByteOffset)
    return emitOpError() << "weight_scale_byte_offset must be "
                         << kExpectedQ4_KWeightScaleByteOffset
                         << " (the 6-bit scales follow the 4-byte fp16 d/dmin)";
  if (!isIntVectorOfLanes(getBFragment().getType(), kIMEBFragmentLanes, 8))
    return emitOpError() << "decoded b_fragment must be vector<"
                         << kIMEBFragmentLanes << "xi8> (the 4x8 int8 MAC "
                            "fragment; the raw nibble [0,15] fits int8 exactly)";
  return mlir::success();
}

mlir::LogicalResult Q4KScaleMinUnpackCoreOp::verify() {
  // Fail-closed (I7): only the canonical ggml get_scale_min_k4 6-bit unpack.
  if (getScaleMinModel() != kExpectedQ4_KScaleMinModel)
    return emitOpError() << "scale_min_model must be '"
                         << kExpectedQ4_KScaleMinModel
                         << "' (the canonical ggml 6-bit scale/min bit-unpack)";
  if (getNumSubBlocks() != kExpectedQ4_KNumSubBlocks)
    return emitOpError() << "num_sub_blocks must be " << kExpectedQ4_KNumSubBlocks
                         << " (a q4_K super-block has 8 sub-blocks of 32)";
  if (getScaleBits() != kExpectedQ4_KScaleBits)
    return emitOpError() << "scale_bits must be " << kExpectedQ4_KScaleBits
                         << " (q4_K scales/mins are 6-bit)";
  if (getKScaleSize() != kExpectedQ4_KKScaleSize)
    return emitOpError() << "k_scale_size must be " << kExpectedQ4_KKScaleSize
                         << " (the packed 6-bit scales/mins region is 12 bytes)";
  if (getWeightScaleByteOffset() != kExpectedQ4_KWeightScaleByteOffset)
    return emitOpError() << "weight_scale_byte_offset must be "
                         << kExpectedQ4_KWeightScaleByteOffset;
  if (!isIntVectorOfLanes(getSc().getType(), kIMEAccTileLanes, 32) ||
      !isIntVectorOfLanes(getM().getType(), kIMEAccTileLanes, 32))
    return emitOpError() << "unpacked sc/m must be vector<" << kIMEAccTileLanes
                         << "xi32> (the per-column 6-bit scale/min tile lanes)";
  return mlir::success();
}

mlir::LogicalResult Q4KScaleWeightedAccumOp::verify() {
  // Fail-closed (I7): only the int32-exact per-sub-block scale-weighted sum.
  if (getAccumModel() != kExpectedQ4_KScaleWeightedModel)
    return emitOpError() << "accum_model must be '"
                         << kExpectedQ4_KScaleWeightedModel
                         << "' (S_scale += sc_b * sumi_b)";
  if (!isIntVectorOfLanes(getSumi().getType(), kIMEAccTileLanes, 32) ||
      !isIntVectorOfLanes(getSc().getType(), kIMEAccTileLanes, 32) ||
      !isIntVectorOfLanes(getAccScaleIn().getType(), kIMEAccTileLanes, 32) ||
      !isIntVectorOfLanes(getAccScaleOut().getType(), kIMEAccTileLanes, 32))
    return emitOpError() << "sumi/sc/acc_scale_in/acc_scale_out must be vector<"
                         << kIMEAccTileLanes << "xi32> (the int32 4x4 tiles)";
  return mlir::success();
}

mlir::LogicalResult Q4KMinBiasAccumOp::verify() {
  // Fail-closed (I7): only the int32-exact activation-sum min-bias.
  if (getBiasModel() != kExpectedQ4_KMinBiasModel)
    return emitOpError() << "bias_model must be '" << kExpectedQ4_KMinBiasModel
                         << "' (S_min += m_b * asum_b; asum_b = the pure "
                            "activation sub-block sum vmadot cannot express)";
  if (!isIntVectorOfLanes(getAFragment().getType(), kIMEBFragmentLanes, 8))
    return emitOpError() << "a_fragment must be vector<" << kIMEBFragmentLanes
                         << "xi8> (the int8 activation fragment reduced to asum_b)";
  if (!isIntVectorOfLanes(getM().getType(), kIMEAccTileLanes, 32) ||
      !isIntVectorOfLanes(getAccMinIn().getType(), kIMEAccTileLanes, 32) ||
      !isIntVectorOfLanes(getAccMinOut().getType(), kIMEAccTileLanes, 32))
    return emitOpError() << "m/acc_min_in/acc_min_out must be vector<"
                         << kIMEAccTileLanes << "xi32> (the int32 4x4 tiles)";
  return mlir::success();
}

mlir::LogicalResult Q4KMatMulTileYieldOp::verify() {
  // The q4_K tile carries TWO accumulators (S_scale + S_min); the yield names both.
  if (getAccOut().size() != 2)
    return emitOpError() << "must name exactly the two carried-out int32 "
                            "accumulator tiles (S_scale, S_min)";
  for (mlir::Value acc : getAccOut())
    if (!isIntVectorOfLanes(acc.getType(), kIMEAccTileLanes, 32))
      return emitOpError() << "every carried-out accumulator must be vector<"
                           << kIMEAccTileLanes << "xi32>";
  return mlir::success();
}

llvm::StringRef Q4KMatMulTileOp::getWEFTEmitCLowerableSourceOpName() {
  return getOperation()->getName().getStringRef();
}

llvm::StringRef Q4KMatMulTileOp::getWEFTEmitCLowerableSourceRole() {
  return kSourceRoleValue;
}

mlir::LogicalResult Q4KMatMulTileOp::verify() {
  // The int8->int32 vmadot MAC envelope + selected-path binding (signed vmadot).
  if (mlir::failed(verifyIMEMACBoundary(getOperation(), kExpectedSignedIMEOp,
                                        isAllowedQ4_KTileAttr,
                                        [this]() { return emitOpError(); })))
    return mlir::failure();

  // The q4_K weight-format facts (fail-closed: only q4_K is modeled here).
  if (getWeightFormat() != kExpectedQ4_KFormat)
    return emitOpError() << "weight_format must be '" << kExpectedQ4_KFormat
                         << "' (this is the q4_K super-block format-keyed IME tile)";
  if (getQk() != kExpectedQ4_KQk)
    return emitOpError() << "qk must be " << kExpectedQ4_KQk;
  if (getWeightBlockStride() != kExpectedQ4_KWeightBlockStride)
    return emitOpError() << "weight_block_stride must be "
                         << kExpectedQ4_KWeightBlockStride;
  if (getWeightQuantByteOffset() != kExpectedQ4_KWeightQuantByteOffset)
    return emitOpError() << "weight_quant_byte_offset must be "
                         << kExpectedQ4_KWeightQuantByteOffset;

  // Whole-matrix problem dims: present, positive, whole multiples of the MAC
  // fragment (fail-closed: NO remainder path).
  int64_t matM = getMatM(), matN = getMatN(), matK = getMatK();
  if (matM <= 0 || matN <= 0 || matK <= 0)
    return emitOpError() << "problem dims (mat_m/mat_n/mat_k) must be positive";
  int64_t macM = getMacM(), macN = getMacN(), macK = getMacK();
  if (matM % macM != 0 || matN % macN != 0 || matK % macK != 0)
    return emitOpError()
           << "problem dims must each be a whole multiple of the MAC fragment "
              "(no remainder path is emitted, I7)";
  // The K dimension must partition into whole q4_K super-blocks (qk=256).
  if (matK % kExpectedQ4_KQk != 0)
    return emitOpError() << "mat_k=" << matK
                         << " must be a whole multiple of qk=" << kExpectedQ4_KQk
                         << " (whole q4_K super-blocks; no remainder path)";

  // Fail-closed region SHAPE: the single-block region must be exactly the SIX
  // decomposed typed bricks -- the raw-nibble decode, the 6-bit scale/min unpack,
  // the vmadot MAC leaf, the scale-weighted accum (S_scale), the min-bias accum
  // (S_min), and the two-tile yield -- and NO opaque hand helper. Dropping the
  // scale-weighted-accum or min-bias-accum brick (the HOLLOW bare-MAC shape) is
  // rejected: those two bricks ARE the per-sub-block scale weighting + min bias
  // that define q4_K.
  mlir::Region &body = getBody();
  if (!body.hasOneBlock())
    return emitOpError() << "typed region must have exactly one block";
  mlir::Block &block = body.front();
  auto dequantCores = block.getOps<Q4KDequantCoreOp>();
  auto scaleMinCores = block.getOps<Q4KScaleMinUnpackCoreOp>();
  auto macLeaves = block.getOps<VmadotMacLeafOp>();
  auto scaleAccums = block.getOps<Q4KScaleWeightedAccumOp>();
  auto minBiasAccums = block.getOps<Q4KMinBiasAccumOp>();
  if (std::distance(dequantCores.begin(), dequantCores.end()) != 1)
    return emitOpError() << "typed region must contain exactly one "
                            "weft.ime.q4_K_dequant_core weight-decode brick";
  if (std::distance(scaleMinCores.begin(), scaleMinCores.end()) != 1)
    return emitOpError() << "typed region must contain exactly one "
                            "weft.ime.q4_K_scale_min_unpack_core brick";
  if (std::distance(macLeaves.begin(), macLeaves.end()) != 1)
    return emitOpError() << "typed region must contain exactly one "
                            "weft.ime.vmadot_mac_leaf MAC brick";
  if (std::distance(scaleAccums.begin(), scaleAccums.end()) != 1)
    return emitOpError() << "typed region must contain exactly one "
                            "weft.ime.q4_K_scale_weighted_accum brick (the "
                            "per-sub-block scale weighting; dropping it is the "
                            "HOLLOW bare-MAC q4_K representation)";
  if (std::distance(minBiasAccums.begin(), minBiasAccums.end()) != 1)
    return emitOpError() << "typed region must contain exactly one "
                            "weft.ime.q4_K_min_bias_accum brick (the min bias; "
                            "dropping it is the HOLLOW bare-MAC q4_K "
                            "representation)";
  if (!llvm::isa<Q4KMatMulTileYieldOp>(block.getTerminator()))
    return emitOpError() << "typed region must be terminated by "
                            "weft.ime.q4_K_matmul_tile_yield";
  // No brick outside the six-op decomposed vocabulary (anti-opaque).
  for (mlir::Operation &nested : block) {
    if (!llvm::isa<Q4KDequantCoreOp, Q4KScaleMinUnpackCoreOp, VmadotMacLeafOp,
                   Q4KScaleWeightedAccumOp, Q4KMinBiasAccumOp,
                   Q4KMatMulTileYieldOp>(nested))
      return emitOpError() << "typed region admits ONLY the decomposed q4_K "
                              "raw-nibble / scale-min-unpack / vmadot-leaf / "
                              "scale-weighted-accum / min-bias-accum / yield "
                              "bricks; found '"
                           << nested.getName().getStringRef() << "'";
  }
  return mlir::success();
}

void WEFTIMEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/IME/IR/IMEOps.cpp.inc"
      >();
}
