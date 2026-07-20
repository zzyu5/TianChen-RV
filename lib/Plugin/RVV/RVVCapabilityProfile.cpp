#include "Weft/Plugin/RVV/RVVCapabilityProfile.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Plugin/RVV/RVVExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

namespace weft::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVHartCountCapabilityID("rvv.hart_count");
constexpr llvm::StringLiteral kRVVHartCountCapabilitySymbol("rvv_hart_count");
constexpr llvm::StringLiteral kRVVVLenBBytesCapabilityID("rvv.vlenb_bytes");
constexpr llvm::StringLiteral kRVVVLenBBytesCapabilitySymbol(
    "rvv_vlenb_bytes");
constexpr llvm::StringLiteral kRVVClangToolchainCapabilityID(
    "rvv.toolchain.clang");
constexpr llvm::StringLiteral kRVVClangToolchainCapabilitySymbol(
    "rvv_toolchain_clang");
constexpr llvm::StringLiteral kRVVCMakeToolchainCapabilityID(
    "rvv.toolchain.cmake");
constexpr llvm::StringLiteral kRVVCMakeToolchainCapabilitySymbol(
    "rvv_toolchain_cmake");
constexpr llvm::StringLiteral kRVVProbeCompileRunCapabilityID(
    "rvv.probe.compile_run");
constexpr llvm::StringLiteral kRVVProbeCompileRunCapabilitySymbol(
    "rvv_probe_compile_run");
constexpr llvm::StringLiteral kRVVSelectedMarchCapabilityID(
    "rvv.toolchain.march");
constexpr llvm::StringLiteral kRVVSelectedMarchCapabilitySymbol(
    "rvv_toolchain_march");
constexpr llvm::StringLiteral kRVVSelectedMABICapabilityID(
    "rvv.toolchain.mabi");
constexpr llvm::StringLiteral kRVVSelectedMABICapabilitySymbol(
    "rvv_toolchain_mabi");
// Zvfh fp16-vector capability chain. These form a real multi-hop `implies`
// chain in the RVV capability set: rvv.zvfh implies rvv.zvfhmin, and
// rvv.zvfhmin implies rvv.zve32f. The transitive closure resolves
// rvv.zvfh |= rvv.zve32f THROUGH the rvv.zvfhmin descriptor (the closure's live
// fixture). Only added when the probed ISA evidence genuinely names the token
// (never fabricated on a board that lacks it -- core-invariants I5).
constexpr llvm::StringLiteral kRVVZvfhCapabilityID("rvv.zvfh");
constexpr llvm::StringLiteral kRVVZvfhCapabilitySymbol("rvv_zvfh");
constexpr llvm::StringLiteral kRVVZvfhMinCapabilityID("rvv.zvfhmin");
constexpr llvm::StringLiteral kRVVZvfhMinCapabilitySymbol("rvv_zvfhmin");
constexpr llvm::StringLiteral kRVVZve32fCapabilityID("rvv.zve32f");
constexpr llvm::StringLiteral kRVVZvfhCapabilityKind("isa-vector-fp16");
constexpr llvm::StringLiteral kAvailableStatus("available");

using CapabilityProperties = std::map<std::string, std::string>;

llvm::Error makeRVVCapabilityProfileError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV RVV capability profile failed: ") + message,
      llvm::errc::invalid_argument);
}

std::string normalizeFactString(llvm::StringRef value) {
  return value.trim().str();
}

bool containsForbiddenFactText(llvm::StringRef value) {
  std::string lower = value.lower();
  return llvm::StringRef(lower).contains("password") ||
         llvm::StringRef(lower).contains("passwd") ||
         llvm::StringRef(lower).contains("token") ||
         llvm::StringRef(lower).contains("secret") ||
         llvm::StringRef(lower).contains("private key") ||
         llvm::StringRef(lower).contains("authorization:") ||
         llvm::StringRef(lower).contains("api_key") ||
         llvm::StringRef(lower).contains("access_key");
}

bool isSingleBoundedFactString(llvm::StringRef value) {
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

void validateFactString(llvm::StringRef name, llvm::StringRef value,
                        llvm::SmallVectorImpl<std::string> &errors,
                        bool required) {
  if (value.trim().empty()) {
    if (required)
      errors.push_back((llvm::Twine(name) + " is required").str());
    return;
  }
  if (!isSingleBoundedFactString(value))
    errors.push_back(
        (llvm::Twine(name) + " must be a bounded single-line fact").str());
  if (containsForbiddenFactText(value))
    errors.push_back((llvm::Twine(name) +
                      " must not contain secret-like or raw-log text")
                         .str());
}

// True when `token` appears in `text` as a full extension token (bounded by a
// non-alphanumeric separator or a string boundary), so "zvfh" matches
// rv64gcv_zvfh but NOT the leading "zvfh" inside "zvfhmin". `text` is expected
// lowercased; `token` must be a lowercase literal.
bool containsIsaToken(llvm::StringRef text, llvm::StringRef token) {
  std::size_t pos = text.find(token);
  while (pos != llvm::StringRef::npos) {
    std::size_t end = pos + token.size();
    bool leftBoundary =
        pos == 0 || !std::isalnum(static_cast<unsigned char>(text[pos - 1]));
    bool rightBoundary =
        end == text.size() ||
        !std::isalnum(static_cast<unsigned char>(text[end]));
    if (leftBoundary && rightBoundary)
      return true;
    pos = text.find(token, pos + 1);
  }
  return false;
}

bool isHexDigest(llvm::StringRef digest) {
  if (digest.empty())
    return true;
  if (digest.size() != 64)
    return false;
  return llvm::all_of(digest, [](char character) {
    unsigned char byte = static_cast<unsigned char>(character);
    return std::isxdigit(byte);
  });
}

llvm::Error addAvailableCapability(mlir::MLIRContext &context,
                                   support::TargetCapabilitySet &capabilities,
                                   llvm::StringRef symbolName,
                                   llvm::StringRef id, llvm::StringRef kind,
                                   CapabilityProperties properties = {},
                                   llvm::ArrayRef<std::string> providedIDs = {},
                                   llvm::ArrayRef<std::string> impliedIDs = {}) {
  weft::exec::CapabilityRelationsAttr relations;
  if (!providedIDs.empty() || !impliedIDs.empty()) {
    llvm::SmallVector<mlir::StringAttr, 4> provides;
    provides.reserve(providedIDs.size());
    for (const std::string &providedID : providedIDs)
      provides.push_back(mlir::StringAttr::get(&context, providedID));
    llvm::SmallVector<mlir::StringAttr, 4> implies;
    implies.reserve(impliedIDs.size());
    for (const std::string &impliedID : impliedIDs)
      implies.push_back(mlir::StringAttr::get(&context, impliedID));
    relations = weft::exec::CapabilityRelationsAttr::get(&context, provides,
                                                         implies,
                                                         /*conflicts=*/{});
  }
  return capabilities.tryAddCapability(support::CapabilityDescriptor(
      symbolName, id, kind, kAvailableStatus,
      support::CapabilityAvailability::Available, std::move(properties),
      relations),
      "RVV probe capability construction");
}

} // namespace

// Exported ISA-evidence predicate (declared in RVVCapabilityProfile.h): true iff
// the hint string names concrete RVV vector evidence. This is the single
// plugin-local authority both the probe->capability validation
// (validateRVVProbeCapabilityFacts, below) and the EmitC route-planning
// capability-property gate reason over, so the RVV vector-hint tokenization lives
// in ONE place (core-invariants I1/I3).
bool hasRVVVectorHint(llvm::StringRef hints) {
  std::string lower = hints.lower();
  llvm::StringRef normalized(lower);
  if (normalized.contains("zve") || normalized.contains("zvl") ||
      normalized.contains("zvfh") || normalized.contains("gcv") ||
      normalized.contains("xtheadvector"))
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

// Derives the RVV element-width (SEW) SUPPORT allow-list from the validated ISA
// evidence (selected -march plus the probed isa/vector hint string). This is a
// TARGET-CAPABILITY fact ("what element widths this configured target supports"),
// NOT a plugin-selected compile-time config (the typed body owns its single
// chosen SEW; see core-invariants I5 and architecture/能力模型.md: the probe
// must not fabricate the SELECTED sew/lmul/tail/mask). The allow-list is the set
// the legality gate (verifyRVVSelectedTargetCapabilityForTypedConfig /
// checkCapabilityConfigGate) queries against the typed body's SEW. Mapping per
// the RISC-V "V" vector spec EEW rules:
//   * a base "v" / rv64gcv / zve64* configuration provides element widths up to
//     64 (8/16/32/64);
//   * an embedded zve32* configuration (no 64-bit element support) provides
//     8/16/32 only;
// fp16 (zvfh) is a float-width concern handled by the dtype path, not this SEW
// integer-width allow-list. Returns "" when the evidence does not name a concrete
// RVV element-width tier, so the capability simply declares no SEW restriction
// (the gate then stays silent, the historical behaviour).
std::string deriveSupportedSEWAllowList(llvm::StringRef selectedMarch,
                                        llvm::StringRef isaVectorHints) {
  std::string combined = (selectedMarch.lower() + " " + isaVectorHints.lower());
  llvm::StringRef text(combined);

  // 64-bit-element evidence: full "V" (rv64gcv / a bare "v" extension token),
  // an explicit zve64* embedded vector tier, or the XuanTie xtheadvector full
  // vector unit (RVV0.7 on the C920 -- a complete 8..64 element-width vector
  // unit; it shares this SEW axis with RVV1.0 and diverges on the ISA GENERATION
  // and on the LMUL axis -- RVV0.7 has no fractional LMUL, see
  // deriveSupportedLMULAllowList).
  bool hasElement64 = text.contains("zve64") || text.contains("gcv") ||
                      text.contains("rv64gcv") || text.contains("xtheadvector");
  // 32-bit-element-only embedded tier: zve32* without a zve64* token.
  bool hasElement32Only = text.contains("zve32") && !text.contains("zve64");

  if (hasElement32Only)
    return "8,16,32";
  if (hasElement64)
    return "8,16,32,64";
  return "";
}

// Derives the LMUL grouping SUPPORT allow-list. The ratified RVV1.0 generation
// (full "V" / rv64gcv / embedded zve* tiers) provides the FRACTIONAL LMUL
// groupings (mf8/mf4/mf2) alongside the whole multipliers (m1..m8). The
// pre-ratification RVV0.7.1 generation (XuanTie xtheadvector on the C920) does
// NOT: it has NO fractional LMUL at all -- empirically proven on hardware (the
// XuanTie 0.7.1 vector header declares ZERO mf2/mf4/mf8 types, and a weft-opt-
// emitted repack kernel fails to compile against `vint8mf2_t` /
// `__riscv_vle8_v_i8mf2` there). So the RVV0.7 allow-list is exactly
// {m1,m2,m4,m8} (whole multipliers only), while RVV1.0 keeps the full grid.
// This is a SECOND N1 capability divergence axis (the SAME kernel selects a
// different LMUL on RVV0.7 vs RVV1.0) and is derived off the RVV-GENERATION fact
// (deriveRVVVersion), not a raw march substring beyond the version detection
// (core-invariants I1/I3: gate on the version FACT, not the march string). As
// with SEW, this is the support set the legality gate queries, never the body's
// single selected LMUL. Unknown generation -> "" (no LMUL restriction).
std::string deriveSupportedLMULAllowList(llvm::StringRef selectedMarch,
                                         llvm::StringRef isaVectorHints) {
  switch (deriveRVVVersion(selectedMarch, isaVectorHints)) {
  case RVVVersion::RVV0p7:
    // RVV0.7.1 (xtheadvector / C920): NO fractional LMUL exists on this
    // generation -- whole multipliers only.
    return "m1,m2,m4,m8";
  case RVVVersion::RVV1p0:
    // Ratified RVV1.0 (rv64gcv / a bare "v" / the embedded zve* tiers): the full
    // grouping grid including the fractional mf8/mf4/mf2 rungs.
    return "mf8,mf4,mf2,m1,m2,m4,m8";
  case RVVVersion::Unknown:
    // No concrete RVV generation named -> no LMUL restriction (gate stays
    // silent on this axis, the historical behaviour).
    return "";
  }
  return "";
}

// Derives whether the configured RVV target GUARANTEES Zvl128b (VLEN >= 128) as
// a hard ISA fact. The ratified RISC-V "V" extension mandates Zvl128b, so a
// full-V configuration (rv64gcv or a bare "v" token) guarantees VLEN >= 128. The
// embedded zve32x / zve64x tiers mandate only Zvl32b / Zvl64b, so they do NOT
// guarantee VLEN >= 128 unless an explicit zvl{N}b token with N >= 128 is named
// (the conservative fallback: an embedded target that explicitly advertises
// Zvl128b/Zvl256b/... genuinely guarantees >= 128). Mirrors deriveSupported*'s
// march+hint tokenization. Note "zve32x"/"zve64x" contain a literal 'v', so the
// full-V probe matches the "gcv" token or a bare "v" extension token, NOT any
// stray 'v'.
std::int64_t deriveMinimumVLEN(llvm::StringRef selectedMarch,
                               llvm::StringRef isaVectorHints) {
  std::string combined = (selectedMarch.lower() + " " + isaVectorHints.lower());
  llvm::StringRef text(combined);

  std::int64_t floorBits = 0;

  // (1) Every explicit Zvl{N}b token raises the guaranteed minimum to N. Scan
  // every "zvl" occurrence and parse its bit width; the legal Zvl widths are
  // powers of two from 32 up (zvl32b .. zvl65536b). Take the LARGEST such floor
  // (e.g. rv64gcv_zvl256b -> 256, rv64gcv_zvl512b -> 512).
  std::size_t zvlPos = text.find("zvl");
  while (zvlPos != llvm::StringRef::npos) {
    std::size_t digitBegin = zvlPos + 3;
    std::size_t digitEnd = digitBegin;
    while (digitEnd < text.size() &&
           std::isdigit(static_cast<unsigned char>(text[digitEnd])))
      ++digitEnd;
    if (digitEnd > digitBegin && digitEnd < text.size() &&
        text[digitEnd] == 'b') {
      std::uint64_t widthBits = 0;
      if (!text.slice(digitBegin, digitEnd).getAsInteger(10, widthBits) &&
          static_cast<std::int64_t>(widthBits) > floorBits)
        floorBits = static_cast<std::int64_t>(widthBits);
    }
    zvlPos = text.find("zvl", zvlPos + 3);
  }

  // (2) Full "V" mandates Zvl128b by the ratified spec: rv64gcv / a "gcv" token,
  // or a bare "v" vector extension token in the march (not a stray 'v' inside
  // "zve32x"). Detect the full-V token by the "gcv" spelling or a "_v"/leading
  // "v" extension token. The XuanTie xtheadvector full vector unit (RVV0.7 on
  // the C920) likewise guarantees VLEN >= 128. This floors the minimum at 128 (a
  // larger explicit Zvl token in (1) keeps precedence).
  bool fullV = text.contains("gcv") || text.contains("_v") ||
               text.contains("rv64v") || text.contains("rv32v") ||
               text.contains("xtheadvector");
  if (fullV && floorBits < 128)
    floorBits = 128;

  // (3) Embedded zve32x / zve64x without an explicit Zvl128b+ token: only
  // Zvl32b / Zvl64b mandated -> NO guaranteed VLEN >= 128 minimum -> 0.
  return floorBits;
}

bool deriveHasZvl128b(llvm::StringRef selectedMarch,
                      llvm::StringRef isaVectorHints) {
  // Zvl128b is exactly the >= 128-bit minimum-VLEN floor (the SAME march+hint
  // tokenization; rv64gcv_zvl256b still true, embedded zve32x still false).
  return deriveMinimumVLEN(selectedMarch, isaVectorHints) >= 128;
}

//===----------------------------------------------------------------------===//
// In-IR provider-op capability readers.
//===----------------------------------------------------------------------===//

llvm::StringRef getRVVMinimumVLENProviderPropertyName() {
  return "minimum_vlen";
}

bool isRVVCapabilityProvider(mlir::Operation *op) {
  if (!llvm::isa<weft::exec::CapabilityOp, weft::exec::TargetOp>(op))
    return false;
  llvm::StringRef rvvID = getRVVCapabilityID();
  llvm::StringRef rvvKind = getRVVCapabilityKind();
  if (auto id = op->getAttrOfType<mlir::StringAttr>("id"))
    if (id.getValue() == rvvID)
      return true;
  if (auto kind = op->getAttrOfType<mlir::StringAttr>("kind"))
    if (kind.getValue() == rvvKind)
      return true;
  return false;
}

std::optional<std::int64_t> readRVVProviderMinimumVLEN(mlir::ModuleOp module) {
  if (!module)
    return std::nullopt;
  llvm::StringRef propertyName = getRVVMinimumVLENProviderPropertyName();
  std::optional<std::int64_t> found;
  module.walk([&](mlir::Operation *op) {
    if (found)
      return;
    if (!isRVVCapabilityProvider(op))
      return;
    if (auto vlen = op->getAttrOfType<mlir::IntegerAttr>(propertyName))
      found = vlen.getInt();
  });
  return found;
}

std::optional<std::int64_t> readRVVProviderVLenBBytes(mlir::ModuleOp module) {
  // Reads the PROBED real-board VLENB fact (bytes-per-vector-register) OFF the in-IR
  // `rvv.vlenb_bytes` capability op -- the fact the probe mints from the hardware
  // (its `bytes` property). Real-board VLEN(bits) = VLENB * 8. Matched by its own
  // capability id (kind "uarch", NOT an isRVVCapabilityProvider). Returns nullopt
  // when no rvv.vlenb_bytes op carries a positive `bytes` fact, so the caller falls
  // back to deriveMinimumVLEN(-march) byte-for-byte. Accepts a typed i64 or the
  // probe's decimal-string `bytes` (CapabilityProperties are string-valued). NEVER
  // re-parses -march. (Retires the zero-reader status of the vlenb producer.)
  if (!module)
    return std::nullopt;
  llvm::StringRef vlenbID = getRVVVLenBBytesCapabilityID();
  std::optional<std::int64_t> found;
  module.walk([&](mlir::Operation *op) {
    if (found)
      return;
    if (!llvm::isa<weft::exec::CapabilityOp, weft::exec::TargetOp>(op))
      return;
    auto id = op->getAttrOfType<mlir::StringAttr>("id");
    if (!id || id.getValue() != vlenbID)
      return;
    std::int64_t bytes = 0;
    if (auto bytesInt = op->getAttrOfType<mlir::IntegerAttr>("bytes")) {
      bytes = bytesInt.getInt();
    } else if (auto bytesStr = op->getAttrOfType<mlir::StringAttr>("bytes")) {
      if (bytesStr.getValue().trim().getAsInteger(10, bytes))
        return;
    } else {
      return;
    }
    if (bytes > 0)
      found = bytes;
  });
  return found;
}

std::int64_t resolveRVVMinimumVLEN(mlir::ModuleOp module, llvm::StringRef march,
                                   llvm::StringRef isaVectorHints) {
  // PREFER the in-IR typed provider fact (the probe layer / a decisive-experiment
  // fixture stamped it). Only when NO provider carries the fact do we derive from
  // -march ONCE here -- so the LOAD-BEARING value flows through the typed capability
  // object, and a provider minimum_vlen=256 that CONFLICTS with -march zvl128b wins
  // (the consumer follows the pipe, not the -march bypass). Un-probed modules
  // (no provider or no stamp) reproduce the historical deriveMinimumVLEN(-march)
  // value byte-for-byte.
  if (std::optional<std::int64_t> provided = readRVVProviderMinimumVLEN(module))
    return *provided;
  return deriveMinimumVLEN(march, isaVectorHints);
}

int materializeRVVProviderCapabilityAxes(mlir::ModuleOp module,
                                         llvm::StringRef march,
                                         llvm::StringRef isaVectorHints) {
  if (!module)
    return 0;
  // Derive the four support axes ONCE from -march (+ probed isa/vector hints)
  // through this plugin-local authority (the SAME derivations the probe pass
  // uses). No toolchain-probe facts are consulted.
  std::string supportedSEW = deriveSupportedSEWAllowList(march, isaVectorHints);
  std::string supportedLMUL = deriveSupportedLMULAllowList(march, isaVectorHints);
  std::string rvvVersion =
      stringifyRVVVersion(deriveRVVVersion(march, isaVectorHints)).str();
  std::int64_t minimumVLEN = deriveMinimumVLEN(march, isaVectorHints);
  // PREFER the PROBED real-board VLEN fact when rvv.vlenb_bytes is present:
  // VLEN(bits) = VLENB(bytes) * 8 -- minimum_vlen consumes the HARDWARE capability
  // fact, not the -march guess. The -march derivation stays the un-probed fallback.
  // A real board whose VLENB agrees with -march stamps a byte-identical value; only
  // a CONFLICTING fixture makes the probed fact win (the decisive experiment).
  if (std::optional<std::int64_t> vlenbBytes = readRVVProviderVLenBBytes(module))
    minimumVLEN = *vlenbBytes * 8;

  // A march that names no concrete RVV tier derives no axes AND no version AND no
  // VLEN floor: nothing to materialize, leave the IR (and the historically silent
  // gate) unchanged.
  if (supportedSEW.empty() && supportedLMUL.empty() && rvvVersion.empty() &&
      minimumVLEN <= 0)
    return 0;

  llvm::StringRef vlenName = getRVVMinimumVLENProviderPropertyName();
  // The architectural vector-register COUNT (`vreg_count`) is a VLEN-invariant ISA
  // fact -- default = the plugin-local authority (32). Stamped alongside the other
  // axes so the register-budget consumers read an in-IR capability fact, not a
  // hardcoded 32. No-clobber: a narrow-register capability file's vreg_count wins.
  llvm::StringRef vregCountName = getRVVVectorRegisterCountProviderPropertyName();
  std::int64_t vregCount = getRVVArchitecturalVectorRegisterCount();
  int stamped = 0;
  module.walk([&](mlir::Operation *op) {
    if (!isRVVCapabilityProvider(op))
      return;
    bool wrote = false;
    // The string-mirror support axes (no-clobber; empty-derived => skip).
    auto stampStringAxis = [&](llvm::StringRef axisName,
                               const std::string &derived) {
      if (derived.empty() || op->hasAttrOfType<mlir::StringAttr>(axisName))
        return;
      op->setAttr(axisName, mlir::StringAttr::get(op->getContext(), derived));
      wrote = true;
    };
    stampStringAxis("supported_sew", supportedSEW);
    stampStringAxis("supported_lmul", supportedLMUL);
    stampStringAxis("rvv_version", rvvVersion);
    // The minimum-VLEN fact is a TYPED i64 IntegerAttr the resource-aware
    // consumers reason over numerically (no-clobber: a decisive-experiment
    // conflict fixture wins).
    if (minimumVLEN > 0 && !op->hasAttrOfType<mlir::IntegerAttr>(vlenName)) {
      op->setAttr(vlenName,
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(op->getContext(), 64), minimumVLEN));
      wrote = true;
    }
    // The vreg_count fact is a TYPED i64 IntegerAttr the register-pressure
    // consumers reason over numerically (no-clobber: a narrow-register capability
    // file's vreg_count wins; the deployed 32-register board stamps the default).
    if (vregCount > 0 && !op->hasAttrOfType<mlir::IntegerAttr>(vregCountName)) {
      op->setAttr(vregCountName,
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(op->getContext(), 64), vregCount));
      wrote = true;
    }
    if (wrote)
      ++stamped;
  });
  return stamped;
}

RVVVersion readRVVProviderRVVVersion(mlir::ModuleOp module) {
  if (!module)
    return RVVVersion::Unknown;
  RVVVersion found = RVVVersion::Unknown;
  bool seen = false;
  module.walk([&](mlir::Operation *op) {
    if (seen)
      return;
    if (!isRVVCapabilityProvider(op))
      return;
    if (auto version = op->getAttrOfType<mlir::StringAttr>("rvv_version")) {
      llvm::StringRef value = version.getValue().trim();
      if (value == "0.7") {
        found = RVVVersion::RVV0p7;
        seen = true;
      } else if (value == "1.0") {
        found = RVVVersion::RVV1p0;
        seen = true;
      }
    }
  });
  return found;
}

RVVVersion resolveRVVVersion(mlir::ModuleOp module, llvm::StringRef march,
                             llvm::StringRef isaVectorHints) {
  // PREFER the in-IR typed provider fact (the probe layer / a decisive-experiment
  // capability file stamped it). Only when NO provider declares the version do we
  // derive from -march ONCE here -- so the LOAD-BEARING generation flows through
  // the typed capability object, and a capability file rvv_version=1.0 that
  // CONFLICTS with -march xtheadvector (0.7) wins (the consumer follows the pipe,
  // not the -march bypass). Un-probed modules (no provider or no version stamp)
  // reproduce the historical deriveRVVVersion(-march) value byte-for-byte.
  if (RVVVersion provided = readRVVProviderRVVVersion(module);
      provided != RVVVersion::Unknown)
    return provided;
  return deriveRVVVersion(march, isaVectorHints);
}

llvm::StringRef stringifyRVVVersion(RVVVersion version) {
  switch (version) {
  case RVVVersion::RVV1p0:
    return "1.0";
  case RVVVersion::RVV0p7:
    return "0.7";
  case RVVVersion::Unknown:
    return "";
  }
  return "";
}

RVVVersion deriveRVVVersion(llvm::StringRef selectedMarch,
                            llvm::StringRef isaVectorHints) {
  std::string combined = (selectedMarch.lower() + " " + isaVectorHints.lower());
  llvm::StringRef text(combined);

  // (1) RVV0.7 markers are tested FIRST so a "gcv0p7" / "rv64gcv0p7" spelling
  // does NOT fold to 1.0 via its embedded "gcv" substring. The portable spelling
  // `rv64gc_xtheadvector` names the XuanTie 0.7.1 vector unit; an explicit "0p7"
  // version suffix on the V token names the same pre-ratification generation.
  if (text.contains("xtheadvector") || text.contains("0p7"))
    return RVVVersion::RVV0p7;

  // (2) Plain full-V (rv64gcv / a "gcv" token / a bare "v" / "rv64v" token) or an
  // embedded zve* vector tier, with no 0.7 marker, is the ratified RVV1.0
  // generation -- the one with the tail/mask-agnostic (ta/ma) policy.
  if (text.contains("gcv") || text.contains("zve") || text.contains("rv64v") ||
      text.contains("rv32v") || text.contains("_v"))
    return RVVVersion::RVV1p0;

  // (3) No concrete RVV generation named -> Unknown (the version fact stays
  // silent; the downstream version gate is then a no-op on the version axis).
  return RVVVersion::Unknown;
}

std::int64_t getRVVArchitecturalVectorRegisterCount() {
  // The RVV ISA mandates a 32-entry architectural vector register file
  // (v0..v31), invariant across VLEN and across the 0.7.1 / 1.0 generations.
  // This is the schema `vreg_count` hardware-fact DEFAULT; the single named
  // authority (kRVVArchitecturalVectorRegisterCount) the scattered budget
  // constants derive from. A module-aware consumer instead reads the in-IR
  // provider `vreg_count` fact (resolveRVVVectorRegisterBudget) and only falls
  // back to THIS default when no capability provider overrides it.
  return kRVVArchitecturalVectorRegisterCount;
}

llvm::StringRef getRVVVectorRegisterCountProviderPropertyName() {
  return "vreg_count";
}

std::optional<std::int64_t> readRVVProviderVregCount(mlir::ModuleOp module) {
  if (!module)
    return std::nullopt;
  llvm::StringRef propertyName = getRVVVectorRegisterCountProviderPropertyName();
  std::optional<std::int64_t> found;
  module.walk([&](mlir::Operation *op) {
    if (found)
      return;
    if (!isRVVCapabilityProvider(op))
      return;
    if (auto vreg = op->getAttrOfType<mlir::IntegerAttr>(propertyName))
      found = vreg.getInt();
  });
  return found;
}

std::int64_t resolveRVVVectorRegisterBudget(mlir::ModuleOp module) {
  // PREFER the in-IR typed provider `vreg_count` fact (the probe layer stamped 32
  // by default; a narrow-register capability file overrides it via no-clobber).
  // Only when NO provider carries the fact do we fall back to the architectural
  // default here -- so the LOAD-BEARING budget flows through the typed capability
  // object, and a capability file vreg_count=16 flips the register-pressure
  // feasible set (the consumer follows the pipe, not a hardcoded 32). Un-probed /
  // deployed 32-register modules reproduce the historical value byte-for-byte.
  if (std::optional<std::int64_t> provided = readRVVProviderVregCount(module))
    return *provided;
  return getRVVArchitecturalVectorRegisterCount();
}

bool deriveRVVHasFractionalLMUL(llvm::StringRef selectedMarch,
                                llvm::StringRef isaVectorHints) {
  // Fractional LMUL (mf2/mf4/mf8) exists on the ratified RVV1.0 generation and is
  // ABSENT on the pre-ratification RVV0.7.1 (xtheadvector / C920). Unknown ->
  // false (conservative whole-LMUL-only). Driven off the SAME generation fact
  // (deriveRVVVersion), never a raw march substring, so the boolean and the LMUL
  // allow-list agree by construction.
  return deriveRVVVersion(selectedMarch, isaVectorHints) == RVVVersion::RVV1p0;
}

llvm::StringRef getRVVHartCountCapabilityID() {
  return kRVVHartCountCapabilityID;
}

llvm::StringRef getRVVHartCountCapabilitySymbol() {
  return kRVVHartCountCapabilitySymbol;
}

llvm::StringRef getRVVVLenBBytesCapabilityID() {
  return kRVVVLenBBytesCapabilityID;
}

llvm::StringRef getRVVVLenBBytesCapabilitySymbol() {
  return kRVVVLenBBytesCapabilitySymbol;
}

llvm::StringRef getRVVClangToolchainCapabilityID() {
  return kRVVClangToolchainCapabilityID;
}

llvm::StringRef getRVVClangToolchainCapabilitySymbol() {
  return kRVVClangToolchainCapabilitySymbol;
}

llvm::StringRef getRVVCMakeToolchainCapabilityID() {
  return kRVVCMakeToolchainCapabilityID;
}

llvm::StringRef getRVVCMakeToolchainCapabilitySymbol() {
  return kRVVCMakeToolchainCapabilitySymbol;
}

llvm::StringRef getRVVProbeCompileRunCapabilityID() {
  return kRVVProbeCompileRunCapabilityID;
}

llvm::StringRef getRVVProbeCompileRunCapabilitySymbol() {
  return kRVVProbeCompileRunCapabilitySymbol;
}

llvm::StringRef getRVVSelectedMarchCapabilityID() {
  return kRVVSelectedMarchCapabilityID;
}

llvm::StringRef getRVVSelectedMarchCapabilitySymbol() {
  return kRVVSelectedMarchCapabilitySymbol;
}

llvm::StringRef getRVVSelectedMABICapabilityID() {
  return kRVVSelectedMABICapabilityID;
}

llvm::StringRef getRVVSelectedMABICapabilitySymbol() {
  return kRVVSelectedMABICapabilitySymbol;
}

llvm::Error
validateRVVProbeCapabilityFacts(const RVVProbeCapabilityFacts &facts) {
  llvm::SmallVector<std::string, 8> errors;

  std::string architecture = normalizeFactString(facts.architecture);
  if (llvm::StringRef(architecture).lower() != "riscv64")
    errors.push_back("architecture must be riscv64");

  if (facts.hartCount == 0)
    errors.push_back("hart count must be greater than zero");

  validateFactString("ISA/vector hint", facts.isaVectorHints, errors,
                     true);
  if (!facts.isaVectorHints.empty() && !hasRVVVectorHint(facts.isaVectorHints))
    errors.push_back("ISA/vector hint must contain RVV vector evidence");

  if (!facts.clangAvailable)
    errors.push_back("clang availability is required");
  validateFactString("clang version", facts.clangVersion, errors,
                     facts.clangAvailable);

  if (!facts.cmakeAvailable)
    errors.push_back("cmake availability is required");
  validateFactString("cmake version", facts.cmakeVersion, errors,
                     facts.cmakeAvailable);

  if (!facts.minimalRVVCompileRunSucceeded)
    errors.push_back("minimal RVV compile/run success is required");

  validateFactString("selected march", facts.selectedMarch, errors, true);
  validateFactString("selected mabi", facts.selectedMABI, errors, false);
  validateFactString("source digest", facts.sourceSHA256, errors, false);
  validateFactString("binary digest", facts.binarySHA256, errors, false);

  if (!isHexDigest(facts.sourceSHA256))
    errors.push_back("source digest must be empty or a 64-character hex digest");
  if (!isHexDigest(facts.binarySHA256))
    errors.push_back("binary digest must be empty or a 64-character hex digest");

  if (!errors.empty()) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    for (const auto &error : llvm::enumerate(errors)) {
      if (error.index() != 0)
        stream << "; ";
      stream << error.value();
    }
    return makeRVVCapabilityProfileError(stream.str());
  }

  return llvm::Error::success();
}

llvm::Expected<support::TargetCapabilitySet>
buildRVVTargetCapabilitiesFromProbeFacts(
    mlir::MLIRContext &context, const RVVProbeCapabilityFacts &facts) {
  if (llvm::Error error = validateRVVProbeCapabilityFacts(facts))
    return std::move(error);

  support::TargetCapabilitySet capabilities;
  CapabilityProperties rvvProperties = {
      {"architecture", normalizeFactString(facts.architecture)},
      {"isa_vector_hints", normalizeFactString(facts.isaVectorHints)}};
  // Derive the SEW / LMUL SUPPORT allow-lists from the validated ISA evidence so
  // a real probed RVV capability carries the divergence axes the legality gate
  // queries (supported_sew / supported_lmul). These are target-capability facts
  // (what the configured target supports), derived in this plugin-local C++
  // authority -- not probe-fabricated selected config (I5; architecture/能力模型.md). An
  // embedded zve32* tier narrows supported_sew to 8,16,32 (no 64) so a SEW=64
  // body is gated out, while a full-V tier admits it: that is the capability-
  // driven divergence on real ISA semantics.
  std::string supportedSEW = deriveSupportedSEWAllowList(
      facts.selectedMarch, facts.isaVectorHints);
  if (!supportedSEW.empty())
    rvvProperties["supported_sew"] = supportedSEW;
  std::string supportedLMUL = deriveSupportedLMULAllowList(
      facts.selectedMarch, facts.isaVectorHints);
  if (!supportedLMUL.empty())
    rvvProperties["supported_lmul"] = supportedLMUL;
  // Derive the RVV ISA-generation fact (the deepest N1 divergence axis). RVV0.7
  // (xtheadvector / C920) and RVV1.0 share the SEW/VLEN axes but diverge on the
  // ratified ta/ma policy (an agnostic-policy body is RVV1.0-only) AND on the
  // LMUL grid (RVV0.7 has no fractional LMUL -- see supportedLMUL above). The
  // version is stamped as a queryable provider-op property the legality gate
  // reads (mirroring supported_sew). Unknown -> no fact (the gate stays silent).
  llvm::StringRef rvvVersion = stringifyRVVVersion(
      deriveRVVVersion(facts.selectedMarch, facts.isaVectorHints));
  if (!rvvVersion.empty())
    rvvProperties["rvv_version"] = rvvVersion.str();
  if (llvm::Error error = addAvailableCapability(
      context, capabilities, getRVVPreferredCapabilitySymbol(),
      getRVVCapabilityID(), getRVVCapabilityKind(), std::move(rvvProperties)))
    return std::move(error);

  // Zvfh fp16-vector capability chain, advertised strictly from probed ISA
  // evidence. rvv.zvfhmin is added whenever the target names Zvfhmin (or the
  // full Zvfh superset, which subsumes it); rvv.zvfh is added only when the full
  // Zvfh token is named. Together they form a real multi-hop `implies` graph --
  // rvv.zvfh implies rvv.zvfhmin, rvv.zvfhmin implies rvv.zve32f -- that the
  // transitive closure walks descriptor-to-descriptor (rvv.zvfh |= rvv.zve32f).
  // These are never fabricated on a board lacking the extension (I5): the token
  // boundary check keeps the "zvfh" prefix of "zvfhmin" from minting full Zvfh.
  std::string isaEvidence =
      (llvm::StringRef(facts.selectedMarch).lower() + " " +
       llvm::StringRef(facts.isaVectorHints).lower());
  llvm::StringRef isaEvidenceRef(isaEvidence);
  bool hasZvfhFull = containsIsaToken(isaEvidenceRef, "zvfh");
  bool hasZvfhMin = hasZvfhFull || containsIsaToken(isaEvidenceRef, "zvfhmin");
  if (hasZvfhMin) {
    if (llvm::Error error = addAvailableCapability(
            context, capabilities, kRVVZvfhMinCapabilitySymbol,
            kRVVZvfhMinCapabilityID, kRVVZvfhCapabilityKind, /*properties=*/{},
            /*providedIDs=*/{}, {kRVVZve32fCapabilityID.str()}))
      return std::move(error);
  }
  if (hasZvfhFull) {
    if (llvm::Error error = addAvailableCapability(
            context, capabilities, kRVVZvfhCapabilitySymbol,
            kRVVZvfhCapabilityID, kRVVZvfhCapabilityKind, /*properties=*/{},
            /*providedIDs=*/{}, {kRVVZvfhMinCapabilityID.str()}))
      return std::move(error);
  }

  if (llvm::Error error = addAvailableCapability(
          context, capabilities, getRVVHartCountCapabilitySymbol(),
          getRVVHartCountCapabilityID(), "uarch",
          {{"count", std::to_string(facts.hartCount)}},
          {support::getTargetHartCountCapabilityID().str()}))
    return std::move(error);
  if (facts.vlenbBytes) {
    if (llvm::Error error = addAvailableCapability(
            context, capabilities, getRVVVLenBBytesCapabilitySymbol(),
            getRVVVLenBBytesCapabilityID(), "uarch",
            {{"bytes", std::to_string(facts.vlenbBytes)}}))
      return std::move(error);
  }
  // [r5.1 W3 · census zero-consume keys] The clang/cmake toolchain-VERSION stamps and
  // the compile_run source/binary SHA-256 stamps (below) are I8 EVIDENCE-LINE
  // provenance (build reproducibility), NOT theta selector inputs: `git grep` finds
  // ZERO in-lib consumers. They are KEPT (not deleted) under ISSUE-121's conservative
  // default -- whether an I8 evidence-line stamp must still be emitted when it has no
  // reader is a GATED [I7]/[I8] canon-wording ruling (agent does not self-modify
  // canon). Deleting them would drop provenance; the conservative law is retain-not-wire.
  if (llvm::Error error = addAvailableCapability(
          context, capabilities, getRVVClangToolchainCapabilitySymbol(),
          getRVVClangToolchainCapabilityID(), "toolchain",
          {{"version", normalizeFactString(facts.clangVersion)}}))
    return std::move(error);
  if (llvm::Error error = addAvailableCapability(
          context, capabilities, getRVVCMakeToolchainCapabilitySymbol(),
          getRVVCMakeToolchainCapabilityID(), "toolchain",
          {{"version", normalizeFactString(facts.cmakeVersion)}}))
    return std::move(error);

  // The compile_run op retains ONLY its I8 source/binary SHA-256 provenance (kept per
  // ISSUE-121). Its former `selected_march` / `selected_mabi` properties -- and the two
  // separate rvv.toolchain.march / rvv.toolchain.mabi `value` stamps that used to follow
  // -- were REDUNDANT MIRRORS of the -march/-mabi the struct fact already carries (the
  // load-bearing march rides the struct parameter + the minimum_vlen provider pipeline,
  // never these in-IR property strings). Census: 0 consumers. They were originally probed
  // as build-provenance echoes; DELETED (r5.1 W3, delete-lean set of the zero-consume
  // key二选一) as pure redundant mirrors.
  CapabilityProperties compileRunProperties;
  if (!facts.sourceSHA256.empty())
    compileRunProperties["source_sha256"] = normalizeFactString(facts.sourceSHA256);
  if (!facts.binarySHA256.empty())
    compileRunProperties["binary_sha256"] = normalizeFactString(facts.binarySHA256);
  if (llvm::Error error = addAvailableCapability(
          context, capabilities, getRVVProbeCompileRunCapabilitySymbol(),
          getRVVProbeCompileRunCapabilityID(), "toolchain",
          std::move(compileRunProperties)))
    return std::move(error);

  return capabilities;
}

} // namespace weft::plugin::rvv
