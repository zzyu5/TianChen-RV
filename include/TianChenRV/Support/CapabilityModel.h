#ifndef TIANCHENRV_SUPPORT_CAPABILITYMODEL_H
#define TIANCHENRV_SUPPORT_CAPABILITYMODEL_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <map>
#include <string>

namespace tianchenrv::support {

llvm::StringRef getTargetHartCountCapabilityID();
llvm::StringRef getHartCountPropertyName();

enum class CapabilityAvailability {
  Available,
  Unavailable,
};

class CapabilityDescriptor {
public:
  CapabilityDescriptor() = default;
  // NOTE on lifetime: when `relations` is non-null it is an interned attribute
  // owned by an MLIRContext; a CapabilityDescriptor (and any TargetCapabilitySet
  // holding it) must not outlive that context. In practice resolution is
  // pass-scoped, so the context always outlives the descriptor.
  CapabilityDescriptor(llvm::StringRef symbolName, llvm::StringRef id,
                       llvm::StringRef kind, llvm::StringRef status,
                       CapabilityAvailability availability,
                       std::map<std::string, std::string> properties = {},
                       tcrv::exec::CapabilityRelationsAttr relations = {});

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
  tcrv::exec::CapabilityRelationsAttr getRelations() const { return relations; }
  llvm::ArrayRef<mlir::StringAttr> getProvidedIDs() const {
    return relations ? relations.getProvides() : llvm::ArrayRef<mlir::StringAttr>{};
  }
  llvm::ArrayRef<mlir::StringAttr> getImpliedIDs() const {
    return relations ? relations.getImplies() : llvm::ArrayRef<mlir::StringAttr>{};
  }
  llvm::ArrayRef<mlir::StringAttr> getConflictingIDs() const {
    return relations ? relations.getConflicts()
                     : llvm::ArrayRef<mlir::StringAttr>{};
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
  // Typed provides/implies/conflicts relations, interned in an MLIRContext
  // (null == no relations). This is the single source of truth for relation
  // resolution; the descriptor holds the very attribute the IR holds.
  tcrv::exec::CapabilityRelationsAttr relations;
};

struct CapabilityConflict {
  const CapabilityDescriptor *requiredCapability = nullptr;
  const CapabilityDescriptor *conflictingCapability = nullptr;
  const CapabilityDescriptor *relationOwner = nullptr;
  std::string conflictID;
};

class TargetCapabilitySet {
public:
  static TargetCapabilitySet
  buildFromKernel(tcrv::exec::KernelOp kernel);
  static llvm::Expected<TargetCapabilitySet>
  buildFromKernelChecked(tcrv::exec::KernelOp kernel);

  bool empty() const { return capabilities.empty(); }
  std::size_t size() const { return capabilities.size(); }
  llvm::ArrayRef<CapabilityDescriptor> getCapabilities() const {
    return capabilities;
  }

  const CapabilityDescriptor *
  lookupBySymbolName(llvm::StringRef symbolName) const;
  const CapabilityDescriptor *lookupByID(llvm::StringRef id) const;
  const CapabilityDescriptor *lookupProviderByID(llvm::StringRef id) const;
  void collectProvidersByID(
      llvm::StringRef id,
      llvm::SmallVectorImpl<const CapabilityDescriptor *> &out) const;
  void collectAvailableConflictsForCapability(
      const CapabilityDescriptor &requiredCapability,
      llvm::SmallVectorImpl<CapabilityConflict> &out) const;

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

  llvm::Error
  tryAddCapability(CapabilityDescriptor descriptor,
                   llvm::StringRef constructionContext =
                       "synthetic construction");
  void addCapability(CapabilityDescriptor descriptor);

private:
  llvm::SmallVector<CapabilityDescriptor, 8> capabilities;
  llvm::StringMap<std::size_t> bySymbolName;
  llvm::StringMap<std::size_t> byID;
};

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_CAPABILITYMODEL_H
