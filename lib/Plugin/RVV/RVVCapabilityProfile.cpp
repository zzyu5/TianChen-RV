#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <map>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {
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
constexpr llvm::StringLiteral kAvailableStatus("available");

using CapabilityProperties = std::map<std::string, std::string>;

llvm::Error makeRVVCapabilityProfileError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV capability profile failed: ") + message,
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
                                   llvm::ArrayRef<std::string> providedIDs =
                                       {}) {
  tcrv::exec::CapabilityRelationsAttr relations;
  if (!providedIDs.empty()) {
    llvm::SmallVector<mlir::StringAttr, 4> provides;
    provides.reserve(providedIDs.size());
    for (const std::string &providedID : providedIDs)
      provides.push_back(mlir::StringAttr::get(&context, providedID));
    relations = tcrv::exec::CapabilityRelationsAttr::get(&context, provides,
                                                         /*implies=*/{},
                                                         /*conflicts=*/{});
  }
  return capabilities.tryAddCapability(support::CapabilityDescriptor(
      symbolName, id, kind, kAvailableStatus,
      support::CapabilityAvailability::Available, std::move(properties),
      relations),
      "RVV probe capability construction");
}

} // namespace

// Derives the RVV element-width (SEW) SUPPORT allow-list from the validated ISA
// evidence (selected -march plus the probed isa/vector hint string). This is a
// TARGET-CAPABILITY fact ("what element widths this configured target supports"),
// NOT a plugin-selected compile-time config (the typed body owns its single
// chosen SEW; see core-invariants I5 and capability-model/profiles.md: the probe
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
  // or an explicit zve64* embedded vector tier.
  bool hasElement64 = text.contains("zve64") || text.contains("gcv") ||
                      text.contains("rv64gcv");
  // 32-bit-element-only embedded tier: zve32* without a zve64* token.
  bool hasElement32Only = text.contains("zve32") && !text.contains("zve64");

  if (hasElement32Only)
    return "8,16,32";
  if (hasElement64)
    return "8,16,32,64";
  return "";
}

// Derives the LMUL grouping SUPPORT allow-list. Every conforming RISC-V vector
// configuration (full "V" and the embedded zve* tiers alike) provides the same
// LMUL grouping set; the tiers differ in supported element widths, not in the
// LMUL multiplier grid. So when the ISA evidence names a concrete RVV tier we
// advertise the full grouping allow-list; otherwise "" (no LMUL restriction).
// As with SEW, this is the support set the legality gate queries, never the
// body's single selected LMUL.
std::string deriveSupportedLMULAllowList(llvm::StringRef selectedMarch,
                                         llvm::StringRef isaVectorHints) {
  std::string combined = (selectedMarch.lower() + " " + isaVectorHints.lower());
  llvm::StringRef text(combined);
  if (text.contains("zve") || text.contains("gcv"))
    return "mf8,mf4,mf2,m1,m2,m4,m8";
  return "";
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
  // authority -- not probe-fabricated selected config (I5; profiles.md). An
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
  if (llvm::Error error = addAvailableCapability(
      context, capabilities, getRVVPreferredCapabilitySymbol(),
      getRVVCapabilityID(), getRVVCapabilityKind(), std::move(rvvProperties)))
    return std::move(error);
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

  CapabilityProperties compileRunProperties = {
      {"selected_march", normalizeFactString(facts.selectedMarch)}};
  if (!facts.selectedMABI.empty())
    compileRunProperties["selected_mabi"] = normalizeFactString(facts.selectedMABI);
  if (!facts.sourceSHA256.empty())
    compileRunProperties["source_sha256"] = normalizeFactString(facts.sourceSHA256);
  if (!facts.binarySHA256.empty())
    compileRunProperties["binary_sha256"] = normalizeFactString(facts.binarySHA256);
  if (llvm::Error error = addAvailableCapability(
          context, capabilities, getRVVProbeCompileRunCapabilitySymbol(),
          getRVVProbeCompileRunCapabilityID(), "toolchain",
          std::move(compileRunProperties)))
    return std::move(error);

  if (llvm::Error error = addAvailableCapability(
          context, capabilities, getRVVSelectedMarchCapabilitySymbol(),
          getRVVSelectedMarchCapabilityID(), "toolchain",
          {{"value", normalizeFactString(facts.selectedMarch)}}))
    return std::move(error);
  if (!facts.selectedMABI.empty()) {
    if (llvm::Error error = addAvailableCapability(
            context, capabilities, getRVVSelectedMABICapabilitySymbol(),
            getRVVSelectedMABICapabilityID(), "toolchain",
            {{"value", normalizeFactString(facts.selectedMABI)}}))
      return std::move(error);
  }

  return capabilities;
}

} // namespace tianchenrv::plugin::rvv
