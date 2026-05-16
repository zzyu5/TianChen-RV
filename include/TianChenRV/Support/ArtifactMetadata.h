#ifndef TIANCHENRV_SUPPORT_ARTIFACTMETADATA_H
#define TIANCHENRV_SUPPORT_ARTIFACTMETADATA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <tuple>

namespace tianchenrv::support {

inline constexpr llvm::StringLiteral kArtifactMetadataKeyAttrName("key");
inline constexpr llvm::StringLiteral kArtifactMetadataValueAttrName("value");

struct ArtifactMetadataEntry {
  ArtifactMetadataEntry() = default;
  ArtifactMetadataEntry(llvm::StringRef key, llvm::StringRef value)
      : key(key.str()), value(value.str()) {}

  std::string key;
  std::string value;
};

inline bool artifactMetadataEntriesEqual(
    llvm::ArrayRef<ArtifactMetadataEntry> lhs,
    llvm::ArrayRef<ArtifactMetadataEntry> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return llvm::all_of(llvm::zip(lhs, rhs), [](const auto &entry) {
    const ArtifactMetadataEntry &left = std::get<0>(entry);
    const ArtifactMetadataEntry &right = std::get<1>(entry);
    return left.key == right.key && left.value == right.value;
  });
}

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_ARTIFACTMETADATA_H
