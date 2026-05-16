#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"

#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

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

llvm::Error addAvailableCapability(support::TargetCapabilitySet &capabilities,
                                   llvm::StringRef symbolName,
                                   llvm::StringRef id, llvm::StringRef kind,
                                   CapabilityProperties properties = {},
                                   llvm::ArrayRef<std::string> providedIDs =
                                       {}) {
  return capabilities.tryAddCapability(support::CapabilityDescriptor(
      symbolName, id, kind, kAvailableStatus,
      support::CapabilityAvailability::Available, std::move(properties),
      providedIDs),
      "RVV probe capability construction");
}

} // namespace

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
    const RVVProbeCapabilityFacts &facts) {
  if (llvm::Error error = validateRVVProbeCapabilityFacts(facts))
    return std::move(error);

  support::TargetCapabilitySet capabilities;
  if (llvm::Error error = addAvailableCapability(
      capabilities, getRVVPreferredCapabilitySymbol(), getRVVCapabilityID(),
      getRVVCapabilityKind(),
      {{"architecture", normalizeFactString(facts.architecture)},
       {"isa_vector_hints", normalizeFactString(facts.isaVectorHints)}}))
    return std::move(error);
  if (llvm::Error error = addAvailableCapability(
          capabilities, getRVVHartCountCapabilitySymbol(),
          getRVVHartCountCapabilityID(), "uarch",
          {{"count", std::to_string(facts.hartCount)}},
          {support::getTargetHartCountCapabilityID().str()}))
    return std::move(error);
  if (facts.vlenbBytes) {
    if (llvm::Error error = addAvailableCapability(
            capabilities, getRVVVLenBBytesCapabilitySymbol(),
            getRVVVLenBBytesCapabilityID(), "uarch",
            {{"bytes", std::to_string(facts.vlenbBytes)}}))
      return std::move(error);
  }
  if (llvm::Error error = addAvailableCapability(
          capabilities, getRVVClangToolchainCapabilitySymbol(),
          getRVVClangToolchainCapabilityID(), "toolchain",
          {{"version", normalizeFactString(facts.clangVersion)}}))
    return std::move(error);
  if (llvm::Error error = addAvailableCapability(
          capabilities, getRVVCMakeToolchainCapabilitySymbol(),
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
          capabilities, getRVVProbeCompileRunCapabilitySymbol(),
          getRVVProbeCompileRunCapabilityID(), "toolchain",
          std::move(compileRunProperties)))
    return std::move(error);

  if (llvm::Error error = addAvailableCapability(
          capabilities, getRVVSelectedMarchCapabilitySymbol(),
          getRVVSelectedMarchCapabilityID(), "toolchain",
          {{"value", normalizeFactString(facts.selectedMarch)}}))
    return std::move(error);
  if (!facts.selectedMABI.empty()) {
    if (llvm::Error error = addAvailableCapability(
            capabilities, getRVVSelectedMABICapabilitySymbol(),
            getRVVSelectedMABICapabilityID(), "toolchain",
            {{"value", normalizeFactString(facts.selectedMABI)}}))
      return std::move(error);
  }

  return capabilities;
}

} // namespace tianchenrv::plugin::rvv
