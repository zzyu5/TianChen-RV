#include "Weft/Plugin/IME/IMEExtensionPlugin.h"

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
constexpr llvm::StringLiteral kIMEConstructionFormulaID(
    "weft.ime.matmul.construct");
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
// The FIFTH (sliding-window) IME variant. Same capability FACT, a different
// SHAPE fact (the ime_slide window stride) => a different emitted instruction
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
// weight-format FACT (recorded on the variant as ime.weight_format) keys WHICH
// boundary op materializes within that variant -- the format-agnostic
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
// G4 M1a: the q4_0 tile op's weight-format op-attribute names + the decomposed
// brick facts. (Distinct from the CAPABILITY property names below: these are the
// op attributes the boundary materializer stamps on weft.ime.q4_0_matmul_tile.)
constexpr llvm::StringLiteral kWeightFormatAttrName("weight_format");
constexpr llvm::StringLiteral kQkAttrName("qk");
constexpr llvm::StringLiteral kWeightBlockStrideAttrName("weight_block_stride");
constexpr llvm::StringLiteral kWeightQuantByteOffsetAttrName(
    "weight_quant_byte_offset");
// The q4_0 offset-binary nibble decode model + the MAC fragment / accumulator
// tile lane counts (4x8 int8 = 32; 4x4 int32 = 16).
constexpr llvm::StringLiteral kQ40DecodeModel("q4_0_offset_binary_nibble");
constexpr int64_t kQ40BFragmentLanes = 32;
constexpr int64_t kQ40AccTileLanes = 16;
// G4 M2: the q8_0 tile reuses the SAME 4x8 int8 fragment / 4x4 int32 tile lane
// counts (the MAC fragment shape is format-agnostic).
constexpr int64_t kQ80BFragmentLanes = kQ40BFragmentLanes;
constexpr int64_t kQ80AccTileLanes = kQ40AccTileLanes;
// G4 M2b: the q4_K tile reuses the SAME MAC fragment / int32 tile lane counts (the
// per-sub-block 6-bit sc/m unpack fills the SAME 4x4 int32 tile lanes).
constexpr int64_t kQ4KBFragmentLanes = kQ40BFragmentLanes;
constexpr int64_t kQ4KAccTileLanes = kQ40AccTileLanes;

constexpr llvm::StringLiteral kRoleOpBoundaryStatusValue("role-op-boundary");

// Capability-property names parsed from the in-IR weft.exec.capability provider.
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
// The mixed-sign form: signed A (activations) * unsigned B (weights) => vmadotsu.
constexpr llvm::StringLiteral kSignednessMixedSign("signed_unsigned");
// The REVERSED-ORDER mixed-sign form: unsigned A * signed B => vmadotus (the
// fourth signedness sibling, completing the family).
constexpr llvm::StringLiteral kSignednessMixedSignUS("unsigned_signed");
// The optional WHOLE-MATRIX problem-shape FACT (format "MxNxK"). When the
// capability carries it, the kernel requests the TILED matmul boundary
// (weft.ime.matmul) over that problem; absent => the single 4x4x8 MAC fragment
// boundary (weft.ime.mma, back-compat with the first slice). This is a SHAPE
// fact, NOT a family-name string: the same capability id + dispatch, a richer
// problem. Dims must each be a whole multiple of the derived MAC fragment.
constexpr llvm::StringLiteral kMatmulShapePropertyName("ime_matmul_shape");
// G4 M1a: the optional WEIGHT-FORMAT fact. When the capability carries it as
// "q4_0" (alongside ime_matmul_shape), the kernel requests the FORMAT-KEYED q4_0
// tiled matmul boundary (weft.ime.q4_0_matmul_tile): the q4_0 weight matrix is
// decoded (offset-binary nibble, quant - 8 in [-8,7]) into the int8 MAC fragment
// before the vmadot MAC. Absent => the format-agnostic pre-packed int8 matmul
// (weft.ime.matmul, back-compat). This is a weight-FORMAT fact of the SAME
// capability, NOT a family-name string and NOT a second capability id. Only
// "q4_0" is modeled in M1 (fail-closed); other formats are M2.
constexpr llvm::StringLiteral kWeightFormatPropertyName("ime_weight_format");
constexpr llvm::StringLiteral kWeightFormatQ40("q4_0");
// The ggml q4_0 block layout FACTS: fp16 d (2B) + 32 packed nibbles (16B) = 18B,
// 32 weights per block; the nibbles follow the 2-byte fp16 scale.
constexpr int64_t kQ40Qk = 32;
constexpr int64_t kQ40WeightBlockStride = 18;
constexpr int64_t kQ40WeightQuantByteOffset = 2;
// G4 M2: the SECOND format-keyed weight format, "q8_0" (FLAT int8). Requested
// via the SAME ime_weight_format capability property (alongside
// ime_matmul_shape). The q8_0 weight matrix is decoded (DIRECT int8 read) into
// the int8 MAC fragment before the vmadot MAC. A weight-FORMAT fact of the SAME
// capability, NOT a family-name string.
constexpr llvm::StringLiteral kWeightFormatQ80("q8_0");
constexpr llvm::StringLiteral kQ80DecodeModel("q8_0_direct_int8");
// The ggml q8_0 block layout FACTS: fp16 d (2B) + 32 int8 quants (32B) = 34B,
// 32 weights per block; the int8 quants follow the 2-byte fp16 scale.
constexpr int64_t kQ80Qk = 32;
constexpr int64_t kQ80WeightBlockStride = 34;
constexpr int64_t kQ80WeightQuantByteOffset = 2;
// G4 M2b: the THIRD format-keyed weight format, "q4_K" (SUPER-BLOCK K-quant).
// Requested via the SAME ime_weight_format capability property. The q4_K weight
// matrix needs the TWO-LEVEL 6-bit scale/min fold (a DEDICATED effort, NOT the
// q4_0/q8_0 single-decode copy-adapt): beyond the reused vmadot MAC leaf it carries
// three NEW region bricks (raw-nibble decode + 6-bit scale/min unpack + the
// scale-weighted / min-bias accumulates). A weight-FORMAT fact of the SAME
// capability, NOT a family-name string.
constexpr llvm::StringLiteral kWeightFormatQ4K("q4_K");
constexpr llvm::StringLiteral kQ4KDecodeModel("q4_K_raw_nibble");
constexpr llvm::StringLiteral kQ4KScaleMinModel("get_scale_min_k4");
constexpr llvm::StringLiteral kQ4KScaleWeightedModel("scale_weighted_sum");
constexpr llvm::StringLiteral kQ4KMinBiasModel("activation_sum_min_bias");
// The ggml q4_K super-block layout FACTS: fp16 d (2B) + fp16 dmin (2B) + 12-byte
// packed 6-bit scales/mins + 128-byte 4-bit quants = 144B, 256 weights / 8
// sub-blocks per super-block; the nibbles follow d/dmin + the 12-byte scales.
constexpr int64_t kQ4KQk = 256;
constexpr int64_t kQ4KWeightBlockStride = 144;
constexpr int64_t kQ4KWeightQuantByteOffset = 16;
constexpr int64_t kQ4KWeightScaleByteOffset = 4;
constexpr int64_t kQ4KNumSubBlocks = 8;
constexpr int64_t kQ4KScaleBits = 6;
constexpr int64_t kQ4KKScaleSize = 12;
// The optional SLIDING-WINDOW stride FACT. When the capability carries it as
// "1"|"2"|"3", the kernel requests the IME1 slide boundary (weft.ime.mma_slide)
// over the SAME 8x8 A-pair MAC fragment with the A read-window shifted DOWN by
// `slide` rows. Absent/"0" => the non-slide MAC (back-compat). Anything else
// fails closed (only the documented vmadot1/2/3 slide family is modeled at
// VLEN=256). This is a SHAPE/window fact of the SAME capability, NOT a
// family-name string and NOT a second capability id.
constexpr llvm::StringLiteral kSlidePropertyName("ime_slide");

// Capability-DERIVED cross-paradigm ranking costs (SEL-1 exec-level capability
// prior). These are NOT capability-blind literals: which cost estimateVariantCost
// emits is DECIDED by whether the spacemit.ime capability fact derives to a
// whole-matrix GEMM shape (ime_matmul_shape) or to a single MAC fragment. The
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
// The slide family mnemonics (funct7 111001), selected by the ime_slide FACT.
constexpr llvm::StringLiteral kIMESlide1OpValue("vmadot1");
constexpr llvm::StringLiteral kIMESlide2OpValue("vmadot2");
constexpr llvm::StringLiteral kIMESlide3OpValue("vmadot3");
// The plugin-owned (dialect-qualified, discardable) variant attribute that
// records the derived signedness fact on the materialized weft.exec.variant so
// the boundary materializer routes to the matching boundary op (mma vs mma_u)
// WITHOUT re-deriving — and so selection stays a pure data flow of the
// capability fact.
constexpr llvm::StringLiteral kSignednessVariantAttrName("ime.signedness");
// The plugin-owned (dialect-qualified, discardable) slide-stride variant attr,
// recorded on the materialized weft.exec.variant so the boundary materializer
// routes to weft.ime.mma_slide (and stamps the slide field) WITHOUT re-deriving.
// "0"/absent => the non-slide boundary path. A pure data flow of the fact.
constexpr llvm::StringLiteral kSlideVariantAttrName("ime.slide");
// The plugin-owned (dialect-qualified, discardable) weight-format variant attr,
// recorded on the materialized weft.exec.variant so the boundary materializer
// routes to weft.ime.q4_0_matmul_tile (and constructs its typed region) WITHOUT
// re-deriving. Absent => the format-agnostic matmul path. A pure data flow of the
// capability-derived fact.
constexpr llvm::StringLiteral kWeightFormatVariantAttrName("ime.weight_format");
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
  // G4 M1a: the FORMAT-KEYED q4_0 whole-matrix fact. True (only when isMatmul)
  // when the kernel requests the q4_0 tiled boundary via the ime_weight_format
  // fact; the q4_0 weight matrix is decoded into the int8 MAC fragment. 0 =>
  // format-agnostic pre-packed int8 matmul.
  bool isQ40Weight = false;
  // G4 M2: the FORMAT-KEYED q8_0 (FLAT int8) whole-matrix fact. True (only when
  // isMatmul) when the kernel requests the q8_0 tiled boundary via
  // ime_weight_format="q8_0"; the q8_0 weight is decoded (direct int8 read) into
  // the int8 MAC fragment. Mutually exclusive with isQ40Weight.
  bool isQ80Weight = false;
  // G4 M2b: the FORMAT-KEYED q4_K (SUPER-BLOCK K-quant) whole-matrix fact. True
  // (only when isMatmul) when the kernel requests the q4_K tiled boundary via
  // ime_weight_format="q4_K"; the q4_K super-block weight is decoded (raw nibble)
  // + two-level 6-bit scale/min folded. Mutually exclusive with isQ40/isQ80Weight.
  bool isQ4KWeight = false;
  // The sliding-window stride FACT (1/2/3) when the kernel requests the IME1
  // slide boundary; 0 => non-slide MAC. Derived from the ime_slide property.
  int64_t slide = 0;
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

  // Derive the requested signedness FACT. The `xsmtvdotii` envelope provides the
  // signed (vmadot), unsigned (vmadotu) and mixed-sign (vmadotsu) MAC; which one
  // this kernel wants is carried as a capability property. Absent => signed
  // (back-compat). Any other value fails closed (we only model the three IME1
  // signedness forms assembled by SpacemiT GCC15 under xsmtvdotii).
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
  } else if (requestedSignedness == kSignednessMixedSign) {
    derived.imeOp = kIMEMixedSignOpValue.str();
    derived.signedness = kSignednessMixedSign.str();
  } else if (requestedSignedness == kSignednessMixedSignUS) {
    derived.imeOp = kIMEMixedSignUSOpValue.str();
    derived.signedness = kSignednessMixedSignUS.str();
  } else {
    return makeIMEPluginError(
        llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
        kSignednessPropertyName + "' = '" + requestedSignedness +
        "' is outside the validated IME1 signedness envelope (only 'signed' "
        "=> vmadot, 'unsigned' => vmadotu, 'signed_unsigned' => vmadotsu, and "
        "'unsigned_signed' => vmadotus are modeled)");
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
    // The tiled whole-matrix boundary is only modeled for the signed/unsigned
    // forms (weft.ime.matmul emits vmadot/vmadotu). The mixed-sign forms
    // (vmadotsu / vmadotus) are single-fragment surfaces (weft.ime.mma_su /
    // weft.ime.mma_us) only — fail closed with a clear message rather than route
    // a mixed-sign tiled shape that has no emitter.
    if (derived.signedness == kSignednessMixedSign ||
        derived.signedness == kSignednessMixedSignUS)
      return makeIMEPluginError(
          llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
          kMatmulShapePropertyName +
          "' (tiled whole-matrix) is not modeled for the mixed-sign '" +
          derived.imeOp +
          "' form; mixed-sign is a single-fragment boundary (weft.ime.mma_su / "
          "weft.ime.mma_us) only");
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

  // G4 M1a: the optional q4_0 WEIGHT-FORMAT fact. Present => the FORMAT-KEYED
  // q4_0 tiled boundary (weft.ime.q4_0_matmul_tile): the q4_0 weight matrix is
  // decoded (offset-binary nibble) into the int8 MAC fragment. Only "q4_0" is
  // modeled in M1 (fail-closed); it rides the SIGNED vmadot form (the decoded
  // quant is signed [-8,7]) over the whole-matrix shape, so it is fail-closed
  // without ime_matmul_shape, for the unsigned/mixed-sign/slide forms, and unless
  // K partitions into whole q4_0 blocks (qk=32).
  llvm::StringRef weightFormat =
      capability.getProperty(kWeightFormatPropertyName).trim();
  if (!weightFormat.empty()) {
    if (weightFormat != kWeightFormatQ40 && weightFormat != kWeightFormatQ80 &&
        weightFormat != kWeightFormatQ4K)
      return makeIMEPluginError(
          llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
          kWeightFormatPropertyName + "' = '" + weightFormat +
          "' is outside the modeled IME weight-format envelope (only 'q4_0', "
          "'q8_0' and 'q4_K' are modeled; other formats need their own "
          "format-keyed tile)");
    if (!derived.isMatmul)
      return makeIMEPluginError(
          llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
          kWeightFormatPropertyName + "' ('" + weightFormat +
          "' format-keyed tile) requires the whole-matrix shape '" +
          kMatmulShapePropertyName + "'");
    if (derived.signedness != kSignednessSigned)
      return makeIMEPluginError(
          llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
          kWeightFormatPropertyName + "' ('" + weightFormat +
          "') is only modeled for the signed vmadot form (the decoded q4_0/q8_0 "
          "quant is signed; the q4_K raw nibble is non-negative)");
    // q4_0/q8_0 carry qk=32 (32 weights/block); q4_K carries qk=256 (256
    // weights/super-block). K must partition into whole (super-)blocks
    // (fail-closed: no remainder path).
    int64_t weightQk = (weightFormat == kWeightFormatQ4K) ? kQ4KQk : kQ40Qk;
    if (derived.matK % weightQk != 0)
      return makeIMEPluginError(
          llvm::Twine("capability id '") + kIMECapabilityID + "' " +
          weightFormat + " mat_k=" + llvm::Twine(derived.matK) +
          " must be a whole multiple of qk=" + llvm::Twine(weightQk) +
          " (whole quant (super-)blocks; no remainder path)");
    if (weightFormat == kWeightFormatQ40)
      derived.isQ40Weight = true;
    else if (weightFormat == kWeightFormatQ80)
      derived.isQ80Weight = true;
    else
      derived.isQ4KWeight = true;
  }

  // Optional SLIDING-WINDOW stride FACT. Absent/"0" => non-slide MAC (back-compat).
  // "1"|"2"|"3" => the IME1 slide boundary (weft.ime.mma_slide) over the SAME 8x8
  // A-pair MAC fragment, the A read-window shifted DOWN by `slide` rows. The slide
  // family rides the SIGNED form only here (vmadot1/2/3 = signed slide), so it is
  // fail-closed for the tiled-matmul shape and for the unsigned/mixed-sign forms
  // (those have no slide emitter). Anything outside {1,2,3} fails closed.
  llvm::StringRef slideText = capability.getProperty(kSlidePropertyName).trim();
  if (!slideText.empty() && slideText != "0") {
    if (derived.isMatmul)
      return makeIMEPluginError(
          llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
          kSlidePropertyName +
          "' (sliding-window) is not modeled together with the tiled "
          "whole-matrix shape '" + kMatmulShapePropertyName +
          "'; the slide boundary is the single-fragment weft.ime.mma_slide only");
    if (derived.signedness != kSignednessSigned)
      return makeIMEPluginError(
          llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
          kSlidePropertyName +
          "' (sliding-window) is only modeled for the signed form "
          "(vmadot1/2/3); the unsigned/mixed-sign slide siblings have no emitter");
    long long slide = 0;
    if (slideText.getAsInteger(10, slide) || slide < 1 || slide > 3)
      return makeIMEPluginError(
          llvm::Twine("capability id '") + kIMECapabilityID + "' property '" +
          kSlidePropertyName + "' = '" + slideText +
          "' is outside the validated IME1 slide envelope (only '1' => vmadot1, "
          "'2' => vmadot2, '3' => vmadot3 are modeled)");
    derived.slide = slide;
    derived.imeOp = (slide == 1   ? kIMESlide1OpValue
                     : slide == 2 ? kIMESlide2OpValue
                                  : kIMESlide3OpValue)
                        .str();
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
/// This is a pure read-back of the capability-derived FACT, so downstream
/// boundary/emission selection never re-classifies a family name. Returns the
/// single-fragment boundary op matching the recorded signedness: 'unsigned' =>
/// weft.ime.mma_u, 'signed_unsigned' => weft.ime.mma_su, 'unsigned_signed' =>
/// weft.ime.mma_us, else weft.ime.mma.
llvm::StringRef singleFragmentBoundaryOpForVariant(weft::exec::VariantOp variant) {
  if (variant) {
    // The slide window FACT (if recorded) routes to the sliding-window boundary
    // regardless of signedness (the slide family is the signed form only).
    auto slide = variant->getAttrOfType<mlir::StringAttr>(kSlideVariantAttrName);
    if (slide && !slide.getValue().trim().empty() && slide.getValue() != "0")
      return weft::ime::MMASlideOp::getOperationName();
    auto signedness =
        variant->getAttrOfType<mlir::StringAttr>(kSignednessVariantAttrName);
    if (signedness && signedness.getValue() == kSignednessUnsigned)
      return weft::ime::MMAUOp::getOperationName();
    if (signedness && signedness.getValue() == kSignednessMixedSign)
      return weft::ime::MMASUOp::getOperationName();
    if (signedness && signedness.getValue() == kSignednessMixedSignUS)
      return weft::ime::MMAUSOp::getOperationName();
  }
  return weft::ime::MMAOp::getOperationName();
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
  // boundary op: the SHAPE fact (single fragment => weft.ime.mma[_u/_su/_us];
  // whole matrix => weft.ime.matmul) and the SIGNEDNESS fact (vmadot / vmadotu /
  // vmadotsu / vmadotus). Both mixed-sign forms are single-fragment only (their
  // tiled shape was fail-closed in derivation), so the matmul arm is binary
  // signed/unsigned.
  bool isUnsigned = derived->signedness == kSignednessUnsigned;
  bool isMixedSign = derived->signedness == kSignednessMixedSign;
  bool isMixedSignUS = derived->signedness == kSignednessMixedSignUS;
  bool isSlide = derived->slide > 0;
  llvm::StringRef variantName;
  if (isSlide)
    variantName = kIMESlideVariantName;
  else if (derived->isMatmul)
    // G4 M1a: the q4_0 format-keyed tile rides the SAME signed matmul variant name
    // (the weight-format fact, stamped below, keys the tile boundary op within it).
    variantName =
        isUnsigned ? kIMEMatmulUVariantName : kIMEMatmulVariantName;
  else if (isMixedSign)
    variantName = kIMEMixedSignVariantName;
  else if (isMixedSignUS)
    variantName = kIMEMixedSignUSVariantName;
  else
    variantName =
        isUnsigned ? kIMEUnsignedVariantName : kIMEFirstSliceVariantName;
  VariantProposal proposal(variantName, kIMEPluginName);
  proposal.setFormulaID(kIMEConstructionFormulaID);
  proposal.addRequiredCapabilityID(kIMECapabilityID);
  proposal.setCondition(kIMECondition);
  proposal.setGuard(kIMEGuard);
  proposal.setPolicy(kIMEPolicy);
  mlir::MLIRContext *context = request.getKernel().getContext();
  proposal.addPluginAttribute(
      mlir::StringAttr::get(context, kSignednessVariantAttrName),
      mlir::StringAttr::get(context, derived->signedness));
  // Record the slide window FACT as data on the variant so the boundary
  // materializer routes to weft.ime.mma_slide WITHOUT re-classifying a name.
  if (isSlide)
    proposal.addPluginAttribute(
        mlir::StringAttr::get(context, kSlideVariantAttrName),
        mlir::StringAttr::get(context, std::to_string(derived->slide)));
  // Record the q4_0 weight-format FACT as data on the variant so the boundary
  // materializer routes to weft.ime.q4_0_matmul_tile (and constructs its typed
  // region) WITHOUT re-classifying a name.
  if (derived->isQ40Weight)
    proposal.addPluginAttribute(
        mlir::StringAttr::get(context, kWeightFormatVariantAttrName),
        mlir::StringAttr::get(context, kWeightFormatQ40));
  // G4 M2: likewise record the q8_0 weight-format FACT so the materializer
  // routes to weft.ime.q8_0_matmul_tile (data flow of the derived fact).
  if (derived->isQ80Weight)
    proposal.addPluginAttribute(
        mlir::StringAttr::get(context, kWeightFormatVariantAttrName),
        mlir::StringAttr::get(context, kWeightFormatQ80));
  // G4 M2b: likewise record the q4_K weight-format FACT so the materializer
  // routes to weft.ime.q4_K_matmul_tile (data flow of the derived fact).
  if (derived->isQ4KWeight)
    proposal.addPluginAttribute(
        mlir::StringAttr::get(context, kWeightFormatVariantAttrName),
        mlir::StringAttr::get(context, kWeightFormatQ4K));
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
  registry.insert<weft::ime::WEFTIMEDialect>();
}

void IMEExtensionPlugin::collectFormulaDescriptors(
    llvm::SmallVectorImpl<FormulaDescriptor> &out) const {
  FormulaDescriptor construction(
      kIMEConstructionFormulaID, kIMEPluginName,
      "contraction/integer-matrix-extension", FormulaResultKind::CandidateSet,
      FormulaConstructionStrength::ConstructedWeak);
  construction.getGeometryAxis().set(FormulaAxisUse::Decisive,
                                     "IMEContractionGeometryFacts");
  for (llvm::StringRef field : {"signedness", "matmul-shape", "weight-format",
                                "slide-window"})
    construction.getGeometryAxis().addConsumedField(field);
  construction.getCapabilityAxis().set(FormulaAxisUse::Decisive,
                                       "IMEMatmulCapability");
  for (llvm::StringRef field : {"march", "vlen-bits", "available-harts"})
    construction.getCapabilityAxis().addConsumedField(field);
  construction.getStaticContextAxis().set(FormulaAxisUse::HonestNull,
                                          "IMENoStaticContext");
  for (llvm::StringRef semanticCase :
       {"signed-mma", "unsigned-mma", "mixed-sign-su", "mixed-sign-us",
        "sliding-window", "whole-matrix", "q4-0-matrix-tile",
        "unsupported-capability"})
    construction.addSemanticCase(semanticCase);
  construction.addProductionEntry("plugin:variant-proposal");
  out.push_back(std::move(construction));

  FormulaDescriptor cost(
      kIMECostFormulaID, kIMEPluginName,
      "contraction/integer-matrix-extension", FormulaResultKind::AnalyticPrior,
      FormulaConstructionStrength::ConstructedWeak);
  cost.getGeometryAxis().set(FormulaAxisUse::Decisive,
                            "IMESelectedVariantFacts");
  cost.getGeometryAxis().addConsumedField("variant-kind");
  cost.getCapabilityAxis().set(FormulaAxisUse::Decisive,
                              "IMEMatmulCapability");
  for (llvm::StringRef field : {"mac-m", "mat-m", "matmul-shape"})
    cost.getCapabilityAxis().addConsumedField(field);
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
  weft::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeIMEPluginError(
        "legality verification requires a materialized weft.exec.variant");

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
        "cost estimation requires a materialized weft.exec.variant");

  // Capability-DERIVED cost (SEL-1 exec-level capability prior). Consult the
  // spacemit.ime capability FACT instead of returning a capability-blind
  // constant: a materialized IME variant only exists when the fact is available,
  // so fail closed otherwise rather than let the cross-paradigm selector prefer
  // an unbacked matrix variant on a phantom score.
  const support::CapabilityDescriptor *capability =
      request.getCapabilities().lookupProviderByID(kIMECapabilityID);
  if (!capability || !capability->isAvailable())
    return makeIMEPluginError(
        "IME cost estimation requires an available capability id "
        "'spacemit.ime'");

  // The MAC envelope FACT (VLEN=256 => 4x4x8) and the optional whole-matrix
  // SHAPE FACT (ime_matmul_shape) both DERIVE from the capability — this shared,
  // fail-closed derivation (same one legality and emission use) is the key that
  // decides how strongly the matrix paradigm is preferred for THIS kernel.
  llvm::Expected<IMEMatmulCapability> derived =
      deriveIMEMatmulCapability(*capability);
  if (!derived)
    return derived.takeError();

  out = VariantCostEstimate();
  out.setExplicitPreference(true);
  out.setOriginPlugin(kIMEPluginName);
  out.setFormulaID(kIMECostFormulaID);
  out.setVariantSymbol(request.getVariant().getSymName());
  if (derived->isMatmul) {
    // M-AWARE capability prior (T5c: writeback of the T5b K1-silicon paradigm-
    // lever measurement). The whole-matrix M dimension (rows / tokens) is read
    // from the derived ime_matmul_shape fact and ranked against the capability-
    // DERIVED crossover M* = macM (the MAC fragment's row dimension; macM itself
    // derives from VLEN — the N1 hook — so M* is NOT a naked constant). The T5b
    // measurement is the audit source: the matrix advantage saturates exactly at
    // M >= macM (=4 @ VLEN256).
    const int64_t crossoverM = derived->macM; // capability-derived M*
    if (derived->matM >= crossoverM) {
      // M >= M*: COMPUTE-BOUND prefill takeover (P7 pattern: ime ∧ matmul-shape
      // ∧ M>=M*). The whole-matrix contraction fully feeds the systolic array, so
      // the capability-derived cost ranks the matrix variant AHEAD of the RVV
      // vector base — IME wins the GEMM because ime_matmul_shape derives AND the M
      // dimension reaches the crossover, not because a constant sorts first.
      out.setScore(kIMEMatmulGemmPreferredCost);
      out.setExplanation(
          "IME whole-matrix vmadot GEMM boundary; cost DERIVED from the available "
          "spacemit.ime capability + ime_matmul_shape fact deriving to the "
          "validated IME1 4x4x8 MAC envelope, with the whole-matrix M dimension "
          "at/above the capability-derived crossover M* (the MAC fragment row "
          "dimension) — the compute-bound prefill regime where the matrix "
          "paradigm is preferred over the RVV vector base");
      out.setPolicy(
          "prefer the IME matrix paradigm for a whole-matrix GEMM when the "
          "spacemit.ime capability derives an in-envelope tiled matmul shape whose "
          "M dimension reaches the capability-derived crossover M* (macM)");
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
          "IME whole-matrix vmadot boundary whose M dimension is BELOW the "
          "capability-derived crossover M* (the MAC fragment row dimension): the "
          "memory-bound decode / matrix-vector roofline regime; cost recorded at "
          "PARITY with the RVV vector base (the compute-isolated micro-advantage "
          "does not transduce to the decode roofline) so the matrix paradigm is "
          "NOT preferred over the vector path");
      out.setPolicy(
          "record roofline parity (do NOT prefer the matrix paradigm) for a "
          "whole-matrix boundary whose M dimension is below the capability-derived "
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
        "emission planning requires a materialized weft.exec.variant");
  if (!request.getKernel())
    return makeIMEPluginError(
        "emission planning requires an enclosing weft.exec.kernel");

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
  bool isQ40Weight = false;
  bool isQ80Weight = false;
  bool isQ4KWeight = false;
  if (planCapability) {
    if (llvm::Expected<IMEMatmulCapability> planDerived =
            deriveIMEMatmulCapability(*planCapability)) {
      isMatmul = planDerived->isMatmul;
      isQ40Weight = planDerived->isQ40Weight;
      isQ80Weight = planDerived->isQ80Weight;
      isQ4KWeight = planDerived->isQ4KWeight;
    } else
      llvm::consumeError(planDerived.takeError());
  }
  llvm::StringRef boundaryOpName =
      isQ40Weight ? weft::ime::Q40MatMulTileOp::getOperationName()
      : isQ80Weight ? weft::ime::Q80MatMulTileOp::getOperationName()
      : isQ4KWeight ? weft::ime::Q4KMatMulTileOp::getOperationName()
      : isMatmul  ? weft::ime::MatMulOp::getOperationName()
                  : singleFragmentBoundaryOpForVariant(request.getVariant());
  out = VariantEmissionPlan::getSupported(
      kIMEPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "materialized-emitc-cpp-ime-vmadot-mma-module",
      "ime-vmadot-mma-emitc-route",
      "ime-vmadot-mma-runtime-c-abi.v1", "riscv-elf-relocatable-object",
      "IME selected boundary lowers the FOUNDATION-validated int8->int32 MAC "
      "kernel (signed vmadot or unsigned vmadotu) through the common "
      "WEFTEmitCLowerableRoute materializer and the MLIR EmitC C/C++ emitter");
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

  // G4 M1a: the FORMAT-KEYED q4_0 whole-matrix boundary. Instead of the flat
  // format-agnostic weft.ime.matmul op, CONSTRUCT the typed-region
  // weft.ime.q4_0_matmul_tile op (the RVV lowerToRepackGemm front-door precedent
  // applied to the IME matrix paradigm): the region is the innermost
  // contraction-block tile body carrying the decomposed q4_0-decode + vmadot-MAC
  // bricks + the int32 yield. Handled here (early) because it owns a region the
  // flat OperationState path below cannot build.
  if (derived->isQ40Weight) {
    mlir::Location loc = variant.getLoc();
    mlir::OperationState tileState(
        loc, weft::ime::Q40MatMulTileOp::getOperationName());
    tileState.addAttribute(kSourceKernelAttrName,
                           builder.getStringAttr(kernel.getSymName()));
    tileState.addAttribute(
        kSelectedVariantAttrName,
        mlir::FlatSymbolRefAttr::get(context, variant.getSymName()));
    tileState.addAttribute(kOriginAttrName,
                           builder.getStringAttr(kIMEPluginName));
    tileState.addAttribute(
        kRoleAttrName,
        builder.getStringAttr(stringifyVariantEmissionRole(request.getRole())));
    tileState.addAttribute(kStatusAttrName,
                           builder.getStringAttr(kRoleOpBoundaryStatusValue));
    tileState.addAttribute(kRequiredCapabilitiesAttrName, variantRequires);
    tileState.addAttribute(kIMEOpAttrName, builder.getStringAttr(derived->imeOp));
    tileState.addAttribute(kElemInBitsAttrName,
                           builder.getI64IntegerAttr(derived->elemInBits));
    tileState.addAttribute(kAccumBitsAttrName,
                           builder.getI64IntegerAttr(derived->accumBits));
    tileState.addAttribute(kMacMAttrName,
                           builder.getI64IntegerAttr(derived->macM));
    tileState.addAttribute(kMacNAttrName,
                           builder.getI64IntegerAttr(derived->macN));
    tileState.addAttribute(kMacKAttrName,
                           builder.getI64IntegerAttr(derived->macK));
    tileState.addAttribute(kMatMAttrName,
                           builder.getI64IntegerAttr(derived->matM));
    tileState.addAttribute(kMatNAttrName,
                           builder.getI64IntegerAttr(derived->matN));
    tileState.addAttribute(kMatKAttrName,
                           builder.getI64IntegerAttr(derived->matK));
    tileState.addAttribute(kWeightFormatAttrName,
                           builder.getStringAttr(kWeightFormatQ40));
    tileState.addAttribute(kQkAttrName, builder.getI64IntegerAttr(kQ40Qk));
    tileState.addAttribute(kWeightBlockStrideAttrName,
                           builder.getI64IntegerAttr(kQ40WeightBlockStride));
    tileState.addAttribute(
        kWeightQuantByteOffsetAttrName,
        builder.getI64IntegerAttr(kQ40WeightQuantByteOffset));
    tileState.addAttribute(kAvailableHartsAttrName,
                           builder.getStringAttr(derived->availableHarts));
    tileState.addAttribute(
        kIMEReasonAttrName,
        builder.getStringAttr(
            llvm::Twine("capability-derived IME1 int8->int32 ") + derived->imeOp +
            " FORMAT-KEYED q4_0 tiled matmul (" + llvm::Twine(derived->matM) +
            "x" + llvm::Twine(derived->matN) + "x" + llvm::Twine(derived->matK) +
            ") over the MAC fragment from march xsmtvdotii + VLEN/SEW; the q4_0 "
            "weight is decoded (offset-binary nibble) into the int8 MAC fragment"));
    tileState.addRegion();
    auto tile = llvm::cast<weft::ime::Q40MatMulTileOp>(builder.create(tileState));

    // Construct the typed region: the innermost contraction-block tile body. Entry
    // args = block_index (index), the loaded 4x8 int8 activation fragment
    // (vector<32xi8>), and the carried-IN 4x4 int32 accumulator tile
    // (vector<16xi32>). The region carries exactly the three decomposed bricks.
    auto i8FragType = mlir::VectorType::get({kQ40BFragmentLanes},
                                            builder.getI8Type());
    auto i32AccType =
        mlir::VectorType::get({kQ40AccTileLanes}, builder.getI32Type());
    mlir::Block &body = tile.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value aFragment = body.addArgument(i8FragType, loc);
    mlir::Value accIn = body.addArgument(i32AccType, loc);

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    // Brick 1: the q4_0 weight DECODE core (block_index -> decoded int8 B tile).
    mlir::OperationState dequantState(
        loc, weft::ime::Q40DequantCoreOp::getOperationName());
    dequantState.addOperands({blockIndex});
    dequantState.addAttribute("decode_model",
                              builder.getStringAttr(kQ40DecodeModel));
    dequantState.addAttribute(kQkAttrName, builder.getI64IntegerAttr(kQ40Qk));
    dequantState.addAttribute(kWeightBlockStrideAttrName,
                              builder.getI64IntegerAttr(kQ40WeightBlockStride));
    dequantState.addAttribute(
        kWeightQuantByteOffsetAttrName,
        builder.getI64IntegerAttr(kQ40WeightQuantByteOffset));
    dequantState.addAttribute("weight_scale_byte_offset",
                              builder.getI64IntegerAttr(0));
    dequantState.addTypes({i8FragType});
    mlir::Operation *dequant = builder.create(dequantState);
    mlir::Value bFragment = dequant->getResult(0);

    // Brick 2: the vmadot int8->int32 MAC leaf (a, decoded b, acc_in -> acc_out).
    mlir::OperationState macState(
        loc, weft::ime::VmadotMacLeafOp::getOperationName());
    macState.addOperands({aFragment, bFragment, blockIndex, accIn});
    macState.addAttribute(kIMEOpAttrName, builder.getStringAttr(derived->imeOp));
    macState.addAttribute(kElemInBitsAttrName,
                          builder.getI64IntegerAttr(derived->elemInBits));
    macState.addAttribute(kAccumBitsAttrName,
                          builder.getI64IntegerAttr(derived->accumBits));
    macState.addAttribute(kMacMAttrName, builder.getI64IntegerAttr(derived->macM));
    macState.addAttribute(kMacNAttrName, builder.getI64IntegerAttr(derived->macN));
    macState.addAttribute(kMacKAttrName, builder.getI64IntegerAttr(derived->macK));
    macState.addTypes({i32AccType});
    mlir::Operation *mac = builder.create(macState);

    // Terminator: name the carried-out int32 accumulator tile.
    mlir::OperationState yieldState(
        loc, weft::ime::Q40MatMulTileYieldOp::getOperationName());
    yieldState.addOperands({mac->getResult(0)});
    builder.create(yieldState);

    VariantLoweringBoundaryValidationRequest validationRequest(
        variant, kernel, request.getCapabilities(), request.getRole(), tile);
    if (llvm::Error error = validateSelectedLoweringBoundary(validationRequest))
      return error;

    out = VariantLoweringBoundaryResult::getMaterialized(
        kIMEPluginName, kernel.getSymName(), variant.getSymName(),
        request.getRole(), tile);
    return llvm::Error::success();
  }

  // G4 M2: the FORMAT-KEYED q8_0 (FLAT int8) whole-matrix boundary. The
  // copy-adapt sibling of the q4_0 block above: CONSTRUCT the typed-region
  // weft.ime.q8_0_matmul_tile op carrying the decomposed q8_0-decode (DIRECT int8
  // read) + vmadot-MAC bricks + the int32 yield. Handled here (early) because it
  // owns a region the flat OperationState path below cannot build.
  if (derived->isQ80Weight) {
    mlir::Location loc = variant.getLoc();
    mlir::OperationState tileState(
        loc, weft::ime::Q80MatMulTileOp::getOperationName());
    tileState.addAttribute(kSourceKernelAttrName,
                           builder.getStringAttr(kernel.getSymName()));
    tileState.addAttribute(
        kSelectedVariantAttrName,
        mlir::FlatSymbolRefAttr::get(context, variant.getSymName()));
    tileState.addAttribute(kOriginAttrName,
                           builder.getStringAttr(kIMEPluginName));
    tileState.addAttribute(
        kRoleAttrName,
        builder.getStringAttr(stringifyVariantEmissionRole(request.getRole())));
    tileState.addAttribute(kStatusAttrName,
                           builder.getStringAttr(kRoleOpBoundaryStatusValue));
    tileState.addAttribute(kRequiredCapabilitiesAttrName, variantRequires);
    tileState.addAttribute(kIMEOpAttrName, builder.getStringAttr(derived->imeOp));
    tileState.addAttribute(kElemInBitsAttrName,
                           builder.getI64IntegerAttr(derived->elemInBits));
    tileState.addAttribute(kAccumBitsAttrName,
                           builder.getI64IntegerAttr(derived->accumBits));
    tileState.addAttribute(kMacMAttrName,
                           builder.getI64IntegerAttr(derived->macM));
    tileState.addAttribute(kMacNAttrName,
                           builder.getI64IntegerAttr(derived->macN));
    tileState.addAttribute(kMacKAttrName,
                           builder.getI64IntegerAttr(derived->macK));
    tileState.addAttribute(kMatMAttrName,
                           builder.getI64IntegerAttr(derived->matM));
    tileState.addAttribute(kMatNAttrName,
                           builder.getI64IntegerAttr(derived->matN));
    tileState.addAttribute(kMatKAttrName,
                           builder.getI64IntegerAttr(derived->matK));
    tileState.addAttribute(kWeightFormatAttrName,
                           builder.getStringAttr(kWeightFormatQ80));
    tileState.addAttribute(kQkAttrName, builder.getI64IntegerAttr(kQ80Qk));
    tileState.addAttribute(kWeightBlockStrideAttrName,
                           builder.getI64IntegerAttr(kQ80WeightBlockStride));
    tileState.addAttribute(
        kWeightQuantByteOffsetAttrName,
        builder.getI64IntegerAttr(kQ80WeightQuantByteOffset));
    tileState.addAttribute(kAvailableHartsAttrName,
                           builder.getStringAttr(derived->availableHarts));
    tileState.addAttribute(
        kIMEReasonAttrName,
        builder.getStringAttr(
            llvm::Twine("capability-derived IME1 int8->int32 ") + derived->imeOp +
            " FORMAT-KEYED q8_0 tiled matmul (" + llvm::Twine(derived->matM) +
            "x" + llvm::Twine(derived->matN) + "x" + llvm::Twine(derived->matK) +
            ") over the MAC fragment from march xsmtvdotii + VLEN/SEW; the q8_0 "
            "weight is decoded (direct int8 read) into the int8 MAC fragment"));
    tileState.addRegion();
    auto tile = llvm::cast<weft::ime::Q80MatMulTileOp>(builder.create(tileState));

    // Construct the typed region: the innermost contraction-block tile body. Entry
    // args = block_index (index), the loaded 4x8 int8 activation fragment
    // (vector<32xi8>), and the carried-IN 4x4 int32 accumulator tile
    // (vector<16xi32>). The region carries exactly the three decomposed bricks.
    auto i8FragType = mlir::VectorType::get({kQ80BFragmentLanes},
                                            builder.getI8Type());
    auto i32AccType =
        mlir::VectorType::get({kQ80AccTileLanes}, builder.getI32Type());
    mlir::Block &body = tile.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value aFragment = body.addArgument(i8FragType, loc);
    mlir::Value accIn = body.addArgument(i32AccType, loc);

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    // Brick 1: the q8_0 weight DECODE core (block_index -> decoded int8 B tile).
    mlir::OperationState dequantState(
        loc, weft::ime::Q80DequantCoreOp::getOperationName());
    dequantState.addOperands({blockIndex});
    dequantState.addAttribute("decode_model",
                              builder.getStringAttr(kQ80DecodeModel));
    dequantState.addAttribute(kQkAttrName, builder.getI64IntegerAttr(kQ80Qk));
    dequantState.addAttribute(kWeightBlockStrideAttrName,
                              builder.getI64IntegerAttr(kQ80WeightBlockStride));
    dequantState.addAttribute(
        kWeightQuantByteOffsetAttrName,
        builder.getI64IntegerAttr(kQ80WeightQuantByteOffset));
    dequantState.addAttribute("weight_scale_byte_offset",
                              builder.getI64IntegerAttr(0));
    dequantState.addTypes({i8FragType});
    mlir::Operation *dequant = builder.create(dequantState);
    mlir::Value bFragment = dequant->getResult(0);

    // Brick 2: the vmadot int8->int32 MAC leaf (a, decoded b, acc_in -> acc_out).
    mlir::OperationState macState(
        loc, weft::ime::VmadotMacLeafOp::getOperationName());
    macState.addOperands({aFragment, bFragment, blockIndex, accIn});
    macState.addAttribute(kIMEOpAttrName, builder.getStringAttr(derived->imeOp));
    macState.addAttribute(kElemInBitsAttrName,
                          builder.getI64IntegerAttr(derived->elemInBits));
    macState.addAttribute(kAccumBitsAttrName,
                          builder.getI64IntegerAttr(derived->accumBits));
    macState.addAttribute(kMacMAttrName, builder.getI64IntegerAttr(derived->macM));
    macState.addAttribute(kMacNAttrName, builder.getI64IntegerAttr(derived->macN));
    macState.addAttribute(kMacKAttrName, builder.getI64IntegerAttr(derived->macK));
    macState.addTypes({i32AccType});
    mlir::Operation *mac = builder.create(macState);

    // Terminator: name the carried-out int32 accumulator tile.
    mlir::OperationState yieldState(
        loc, weft::ime::Q80MatMulTileYieldOp::getOperationName());
    yieldState.addOperands({mac->getResult(0)});
    builder.create(yieldState);

    VariantLoweringBoundaryValidationRequest validationRequest(
        variant, kernel, request.getCapabilities(), request.getRole(), tile);
    if (llvm::Error error = validateSelectedLoweringBoundary(validationRequest))
      return error;

    out = VariantLoweringBoundaryResult::getMaterialized(
        kIMEPluginName, kernel.getSymName(), variant.getSymName(),
        request.getRole(), tile);
    return llvm::Error::success();
  }

  // G4 M2b: the FORMAT-KEYED q4_K (SUPER-BLOCK K-quant) whole-matrix boundary. The
  // DEDICATED effort beyond q4_0/q8_0: CONSTRUCT the typed-region
  // weft.ime.q4_K_matmul_tile op carrying the SIX decomposed bricks -- the
  // raw-nibble decode + the 6-bit scale/min unpack + the reused vmadot MAC + the
  // scale-weighted accum (S_scale) + the min-bias accum (S_min) + the two-tile
  // yield. Handled here (early) because it owns a region the flat OperationState
  // path below cannot build.
  if (derived->isQ4KWeight) {
    mlir::Location loc = variant.getLoc();
    mlir::OperationState tileState(
        loc, weft::ime::Q4KMatMulTileOp::getOperationName());
    tileState.addAttribute(kSourceKernelAttrName,
                           builder.getStringAttr(kernel.getSymName()));
    tileState.addAttribute(
        kSelectedVariantAttrName,
        mlir::FlatSymbolRefAttr::get(context, variant.getSymName()));
    tileState.addAttribute(kOriginAttrName,
                           builder.getStringAttr(kIMEPluginName));
    tileState.addAttribute(
        kRoleAttrName,
        builder.getStringAttr(stringifyVariantEmissionRole(request.getRole())));
    tileState.addAttribute(kStatusAttrName,
                           builder.getStringAttr(kRoleOpBoundaryStatusValue));
    tileState.addAttribute(kRequiredCapabilitiesAttrName, variantRequires);
    tileState.addAttribute(kIMEOpAttrName, builder.getStringAttr(derived->imeOp));
    tileState.addAttribute(kElemInBitsAttrName,
                           builder.getI64IntegerAttr(derived->elemInBits));
    tileState.addAttribute(kAccumBitsAttrName,
                           builder.getI64IntegerAttr(derived->accumBits));
    tileState.addAttribute(kMacMAttrName,
                           builder.getI64IntegerAttr(derived->macM));
    tileState.addAttribute(kMacNAttrName,
                           builder.getI64IntegerAttr(derived->macN));
    tileState.addAttribute(kMacKAttrName,
                           builder.getI64IntegerAttr(derived->macK));
    tileState.addAttribute(kMatMAttrName,
                           builder.getI64IntegerAttr(derived->matM));
    tileState.addAttribute(kMatNAttrName,
                           builder.getI64IntegerAttr(derived->matN));
    tileState.addAttribute(kMatKAttrName,
                           builder.getI64IntegerAttr(derived->matK));
    tileState.addAttribute(kWeightFormatAttrName,
                           builder.getStringAttr(kWeightFormatQ4K));
    tileState.addAttribute(kQkAttrName, builder.getI64IntegerAttr(kQ4KQk));
    tileState.addAttribute(kWeightBlockStrideAttrName,
                           builder.getI64IntegerAttr(kQ4KWeightBlockStride));
    tileState.addAttribute(
        kWeightQuantByteOffsetAttrName,
        builder.getI64IntegerAttr(kQ4KWeightQuantByteOffset));
    tileState.addAttribute(kAvailableHartsAttrName,
                           builder.getStringAttr(derived->availableHarts));
    tileState.addAttribute(
        kIMEReasonAttrName,
        builder.getStringAttr(
            llvm::Twine("capability-derived IME1 int8->int32 ") + derived->imeOp +
            " FORMAT-KEYED q4_K super-block tiled matmul (" +
            llvm::Twine(derived->matM) + "x" + llvm::Twine(derived->matN) + "x" +
            llvm::Twine(derived->matK) +
            ") over the MAC fragment from march xsmtvdotii + VLEN/SEW; the q4_K "
            "super-block weight is decoded (raw nibble) with the two-level 6-bit "
            "scale/min fold (S_scale = sc_b*sumi_b, S_min = m_b*asum_b)"));
    tileState.addRegion();
    auto tile = llvm::cast<weft::ime::Q4KMatMulTileOp>(builder.create(tileState));

    // Construct the typed region: the innermost q4_K super-block tile body. Entry
    // args = block_index (index), the loaded 4x8 int8 activation fragment
    // (vector<32xi8>), and the carried-IN TWO int32 accumulator tiles -- the
    // S_scale accumulator + the S_min accumulator (both vector<16xi32>). The region
    // carries exactly the six decomposed bricks.
    auto i8FragType = mlir::VectorType::get({kQ4KBFragmentLanes},
                                            builder.getI8Type());
    auto i32AccType =
        mlir::VectorType::get({kQ4KAccTileLanes}, builder.getI32Type());
    mlir::Block &body = tile.getBody().emplaceBlock();
    mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
    mlir::Value aFragment = body.addArgument(i8FragType, loc);
    mlir::Value accScaleIn = body.addArgument(i32AccType, loc);
    mlir::Value accMinIn = body.addArgument(i32AccType, loc);

    mlir::OpBuilder::InsertionGuard bodyGuard(builder);
    builder.setInsertionPointToStart(&body);

    // Brick 1: the q4_K RAW-nibble weight DECODE core (block_index -> int8 B tile).
    mlir::OperationState dequantState(
        loc, weft::ime::Q4KDequantCoreOp::getOperationName());
    dequantState.addOperands({blockIndex});
    dequantState.addAttribute("decode_model",
                              builder.getStringAttr(kQ4KDecodeModel));
    dequantState.addAttribute(kQkAttrName, builder.getI64IntegerAttr(kQ4KQk));
    dequantState.addAttribute(kWeightBlockStrideAttrName,
                              builder.getI64IntegerAttr(kQ4KWeightBlockStride));
    dequantState.addAttribute(
        kWeightQuantByteOffsetAttrName,
        builder.getI64IntegerAttr(kQ4KWeightQuantByteOffset));
    dequantState.addAttribute("weight_scale_byte_offset",
                              builder.getI64IntegerAttr(kQ4KWeightScaleByteOffset));
    dequantState.addTypes({i8FragType});
    mlir::Operation *dequant = builder.create(dequantState);
    mlir::Value bFragment = dequant->getResult(0);

    // Brick 2 (NEW): the 6-bit per-sub-block scale/min bit-unpack (get_scale_min_k4).
    mlir::OperationState scaleMinState(
        loc, weft::ime::Q4KScaleMinUnpackCoreOp::getOperationName());
    scaleMinState.addOperands({blockIndex});
    scaleMinState.addAttribute("scale_min_model",
                               builder.getStringAttr(kQ4KScaleMinModel));
    scaleMinState.addAttribute("num_sub_blocks",
                               builder.getI64IntegerAttr(kQ4KNumSubBlocks));
    scaleMinState.addAttribute("scale_bits",
                               builder.getI64IntegerAttr(kQ4KScaleBits));
    scaleMinState.addAttribute("k_scale_size",
                               builder.getI64IntegerAttr(kQ4KKScaleSize));
    scaleMinState.addAttribute(
        "weight_scale_byte_offset",
        builder.getI64IntegerAttr(kQ4KWeightScaleByteOffset));
    scaleMinState.addTypes({i32AccType, i32AccType});
    mlir::Operation *scaleMin = builder.create(scaleMinState);
    mlir::Value scWeights = scaleMin->getResult(0);
    mlir::Value mWeights = scaleMin->getResult(1);

    // Brick 3: the reused vmadot int8->int32 MAC leaf (per-sub-block sumi_b). Its
    // acc_in is the carried scale accumulator (a type-consistent SSA use; the
    // emitted helper carries the exact per-sub-block MAC arithmetic).
    mlir::OperationState macState(
        loc, weft::ime::VmadotMacLeafOp::getOperationName());
    macState.addOperands({aFragment, bFragment, blockIndex, accScaleIn});
    macState.addAttribute(kIMEOpAttrName, builder.getStringAttr(derived->imeOp));
    macState.addAttribute(kElemInBitsAttrName,
                          builder.getI64IntegerAttr(derived->elemInBits));
    macState.addAttribute(kAccumBitsAttrName,
                          builder.getI64IntegerAttr(derived->accumBits));
    macState.addAttribute(kMacMAttrName, builder.getI64IntegerAttr(derived->macM));
    macState.addAttribute(kMacNAttrName, builder.getI64IntegerAttr(derived->macN));
    macState.addAttribute(kMacKAttrName, builder.getI64IntegerAttr(derived->macK));
    macState.addTypes({i32AccType});
    mlir::Operation *mac = builder.create(macState);
    mlir::Value sumi = mac->getResult(0);

    // Brick 4 (NEW): the per-sub-block scale-weighted accumulate S_scale += sc_b*sumi_b.
    mlir::OperationState scaleAccumState(
        loc, weft::ime::Q4KScaleWeightedAccumOp::getOperationName());
    scaleAccumState.addOperands({sumi, scWeights, accScaleIn});
    scaleAccumState.addAttribute("accum_model",
                                 builder.getStringAttr(kQ4KScaleWeightedModel));
    scaleAccumState.addTypes({i32AccType});
    mlir::Operation *scaleAccum = builder.create(scaleAccumState);
    mlir::Value accScaleOut = scaleAccum->getResult(0);

    // Brick 5 (NEW): the activation-sum min-bias accumulate S_min += m_b*asum_b.
    mlir::OperationState minBiasState(
        loc, weft::ime::Q4KMinBiasAccumOp::getOperationName());
    minBiasState.addOperands({aFragment, mWeights, accMinIn});
    minBiasState.addAttribute("bias_model",
                              builder.getStringAttr(kQ4KMinBiasModel));
    minBiasState.addTypes({i32AccType});
    mlir::Operation *minBias = builder.create(minBiasState);
    mlir::Value accMinOut = minBias->getResult(0);

    // Terminator: name the TWO carried-out int32 accumulator tiles (S_scale, S_min).
    mlir::OperationState yieldState(
        loc, weft::ime::Q4KMatMulTileYieldOp::getOperationName());
    yieldState.addOperands({accScaleOut, accMinOut});
    builder.create(yieldState);

    VariantLoweringBoundaryValidationRequest validationRequest(
        variant, kernel, request.getCapabilities(), request.getRole(), tile);
    if (llvm::Error error = validateSelectedLoweringBoundary(validationRequest))
      return error;

    out = VariantLoweringBoundaryResult::getMaterialized(
        kIMEPluginName, kernel.getSymName(), variant.getSymName(),
        request.getRole(), tile);
    return llvm::Error::success();
  }

  // Route to the boundary op matching the DERIVED signedness fact: signed =>
  // weft.ime.mma (vmadot), unsigned => weft.ime.mma_u (vmadotu), signed_unsigned
  // => weft.ime.mma_su (vmadotsu), unsigned_signed => weft.ime.mma_us
  // (vmadotus). All are ODS-registered ops of the SAME dialect; the choice is a
  // pure data flow of the capability-derived fact (recorded on the variant by
  // the materializer), NOT a family-name branch. (The architectural family
  // spelling `weft.ime.*` is not a valid dialect namespace, so the registered
  // names are weft_ime.*.) Two orthogonal capability-derived FACTS pick the
  // boundary op: the SHAPE fact (single fragment => weft.ime.mma[_u/_su/_us];
  // whole matrix => weft.ime.matmul) and the SIGNEDNESS fact (vmadot / vmadotu /
  // vmadotsu / vmadotus). Neither is a family-name branch; both are pure data
  // flow of the derived capability facts. Both mixed-sign forms are
  // single-fragment only (their tiled shape was fail-closed in derivation).
  bool boundaryIsUnsigned = derived->signedness == kSignednessUnsigned;
  bool boundaryIsMixedSign = derived->signedness == kSignednessMixedSign;
  bool boundaryIsMixedSignUS = derived->signedness == kSignednessMixedSignUS;
  bool boundaryIsSlide = derived->slide > 0;
  llvm::StringRef boundaryOpName =
      boundaryIsSlide
          ? weft::ime::MMASlideOp::getOperationName()
          : (derived->isMatmul
                 ? weft::ime::MatMulOp::getOperationName()
                 : (boundaryIsMixedSign
                        ? weft::ime::MMASUOp::getOperationName()
                        : (boundaryIsMixedSignUS
                               ? weft::ime::MMAUSOp::getOperationName()
                               : (boundaryIsUnsigned
                                      ? weft::ime::MMAUOp::getOperationName()
                                      : weft::ime::MMAOp::getOperationName()))));
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
  if (boundaryIsSlide)
    state.addAttribute(kSlideAttrName,
                       builder.getI64IntegerAttr(derived->slide));
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
