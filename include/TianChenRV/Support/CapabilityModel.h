#ifndef TIANCHENRV_SUPPORT_CAPABILITYMODEL_H
#define TIANCHENRV_SUPPORT_CAPABILITYMODEL_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <map>
#include <string>

namespace tianchenrv::support {

enum class CapabilityAvailability {
  Available,
  Unavailable,
};

class CapabilityDescriptor {
public:
  CapabilityDescriptor() = default;
  CapabilityDescriptor(llvm::StringRef symbolName, llvm::StringRef id,
                       llvm::StringRef kind, llvm::StringRef status,
                       CapabilityAvailability availability,
                       std::map<std::string, std::string> properties = {},
                       llvm::ArrayRef<std::string> providedIDs = {},
                       llvm::ArrayRef<std::string> impliedIDs = {},
                       llvm::ArrayRef<std::string> conflictingIDs = {});

  llvm::StringRef getSymbolName() const { return symbolName; }
  llvm::StringRef getID() const { return id; }
  llvm::StringRef getKind() const { return kind; }
  llvm::StringRef getStatus() const { return status; }
  CapabilityAvailability getAvailability() const { return availability; }
  bool isAvailable() const {
    return availability == CapabilityAvailability::Available;
  }
  llvm::StringRef getProperty(llvm::StringRef name) const;
  const std::map<std::string, std::string> &getProperties() const {
    return properties;
  }
  llvm::ArrayRef<std::string> getProvidedIDs() const { return providedIDs; }
  llvm::ArrayRef<std::string> getImpliedIDs() const { return impliedIDs; }
  llvm::ArrayRef<std::string> getConflictingIDs() const {
    return conflictingIDs;
  }

  bool providesID(llvm::StringRef capabilityID) const;
  bool impliesID(llvm::StringRef capabilityID) const;
  bool conflictsWithID(llvm::StringRef capabilityID) const;
  bool satisfiesID(llvm::StringRef capabilityID) const;

private:
  std::string symbolName;
  std::string id;
  std::string kind;
  std::string status;
  CapabilityAvailability availability = CapabilityAvailability::Available;
  std::map<std::string, std::string> properties;
  llvm::SmallVector<std::string, 4> providedIDs;
  llvm::SmallVector<std::string, 4> impliedIDs;
  llvm::SmallVector<std::string, 4> conflictingIDs;
};

class TargetCapabilitySet {
public:
  static TargetCapabilitySet
  buildFromKernel(tcrv::exec::KernelOp kernel);

  bool empty() const { return capabilities.empty(); }
  std::size_t size() const { return capabilities.size(); }
  llvm::ArrayRef<CapabilityDescriptor> getCapabilities() const {
    return capabilities;
  }

  const CapabilityDescriptor *
  lookupBySymbolName(llvm::StringRef symbolName) const;
  const CapabilityDescriptor *lookupByID(llvm::StringRef id) const;
  const CapabilityDescriptor *lookupProviderByID(llvm::StringRef id) const;

  void collectByKind(llvm::StringRef kind,
                     llvm::SmallVectorImpl<const CapabilityDescriptor *> &out)
      const;
  llvm::SmallVector<const CapabilityDescriptor *, 4>
  collectByKind(llvm::StringRef kind) const;

  bool isCapabilityAvailableBySymbolName(llvm::StringRef symbolName) const;
  bool isCapabilityAvailableByID(llvm::StringRef id) const;

  static CapabilityAvailability
  availabilityFromStatus(llvm::StringRef status);
  static bool isUnavailableStatus(llvm::StringRef status);

  void addCapability(CapabilityDescriptor descriptor);

private:
  llvm::SmallVector<CapabilityDescriptor, 8> capabilities;
  llvm::StringMap<std::size_t> bySymbolName;
  llvm::StringMap<std::size_t> byID;
};

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_CAPABILITYMODEL_H
