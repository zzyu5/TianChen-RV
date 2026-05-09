#ifndef TIANCHENRV_PLUGIN_RVV_RVVCAPABILITYPROFILE_H
#define TIANCHENRV_PLUGIN_RVV_RVVCAPABILITYPROFILE_H

#include "TianChenRV/Support/CapabilityModel.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace tianchenrv::plugin::rvv {

struct RVVProbeCapabilityFacts {
  std::string architecture;
  std::uint64_t hartCount = 0;
  std::uint64_t vlenbBytes = 0;
  std::uint64_t i32M1LaneCount = 0;
  std::uint64_t firstSliceSEWBits = 32;
  std::string firstSliceLMUL = "m1";
  std::string firstSliceTailPolicy = "agnostic";
  std::string firstSliceMaskPolicy = "agnostic";
  std::string isaVectorHints;
  bool clangAvailable = false;
  std::string clangVersion;
  bool cmakeAvailable = false;
  std::string cmakeVersion;
  bool minimalRVVCompileRunSucceeded = false;
  std::string selectedMarch;
  std::string selectedMABI;
  std::string sourceSHA256;
  std::string binarySHA256;
};

llvm::StringRef getRVVHartCountCapabilityID();
llvm::StringRef getRVVHartCountCapabilitySymbol();
llvm::StringRef getRVVVLenBBytesCapabilityID();
llvm::StringRef getRVVVLenBBytesCapabilitySymbol();
llvm::StringRef getRVVI32M1LaneCountCapabilityID();
llvm::StringRef getRVVI32M1LaneCountCapabilitySymbol();
llvm::StringRef getRVVI32M1SEW32CapabilityID();
llvm::StringRef getRVVI32M1SEW32CapabilitySymbol();
llvm::StringRef getRVVI32M1LMULM1CapabilityID();
llvm::StringRef getRVVI32M1LMULM1CapabilitySymbol();
llvm::StringRef getRVVI32M1TailAgnosticCapabilityID();
llvm::StringRef getRVVI32M1TailAgnosticCapabilitySymbol();
llvm::StringRef getRVVI32M1MaskAgnosticCapabilityID();
llvm::StringRef getRVVI32M1MaskAgnosticCapabilitySymbol();
llvm::StringRef getRVVClangToolchainCapabilityID();
llvm::StringRef getRVVClangToolchainCapabilitySymbol();
llvm::StringRef getRVVCMakeToolchainCapabilityID();
llvm::StringRef getRVVCMakeToolchainCapabilitySymbol();
llvm::StringRef getRVVProbeCompileRunCapabilityID();
llvm::StringRef getRVVProbeCompileRunCapabilitySymbol();
llvm::StringRef getRVVSelectedMarchCapabilityID();
llvm::StringRef getRVVSelectedMarchCapabilitySymbol();
llvm::StringRef getRVVSelectedMABICapabilityID();
llvm::StringRef getRVVSelectedMABICapabilitySymbol();

llvm::Error validateRVVProbeCapabilityFacts(
    const RVVProbeCapabilityFacts &facts);

llvm::Expected<support::TargetCapabilitySet>
buildRVVTargetCapabilitiesFromProbeFacts(
    const RVVProbeCapabilityFacts &facts);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVCAPABILITYPROFILE_H
