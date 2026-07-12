#include "Weft/Support/DeclaredInstanceHash.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <string>

namespace weft::support {

namespace {

// Append a JSON-escaped, double-quoted string. Reuses LLVM's JSON string
// escaper (via json::Value) so the canonicalization matches the E1 idiom's
// spirit -- no incidental whitespace, standard escaping.
void appendJsonString(llvm::raw_ostream &os, llvm::StringRef value) {
  os << llvm::json::Value(value.str());
}

llvm::StringRef stringifyAvailability(CapabilityAvailability availability) {
  switch (availability) {
  case CapabilityAvailability::Available:
    return "available";
  case CapabilityAvailability::Unavailable:
    return "unavailable";
  }
  return "unavailable";
}

// Serialize a relation-id list (provides/implies/conflicts) as a sorted JSON
// array of strings. Sorting makes the serialization independent of the order the
// relation ids happen to appear in the interned attribute.
void appendSortedIDArray(llvm::raw_ostream &os,
                         llvm::ArrayRef<mlir::StringAttr> ids) {
  llvm::SmallVector<llvm::StringRef, 4> values;
  values.reserve(ids.size());
  for (mlir::StringAttr id : ids)
    values.push_back(id.getValue());
  llvm::sort(values);

  os << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index)
      os << ',';
    appendJsonString(os, values[index]);
  }
  os << ']';
}

void appendDescriptor(llvm::raw_ostream &os,
                      const CapabilityDescriptor &descriptor) {
  // Fixed alphabetical key order. symbolName is intentionally omitted.
  os << '{';
  os << "\"availability\":";
  appendJsonString(os, stringifyAvailability(descriptor.getAvailability()));
  os << ",\"conflicts\":";
  appendSortedIDArray(os, descriptor.getConflictingIDs());
  os << ",\"id\":";
  appendJsonString(os, descriptor.getID());
  os << ",\"implies\":";
  appendSortedIDArray(os, descriptor.getImpliedIDs());
  os << ",\"kind\":";
  appendJsonString(os, descriptor.getKind());
  os << ",\"properties\":{";
  // std::map<std::string, std::string> already iterates in sorted key order.
  bool firstProperty = true;
  for (const auto &property : descriptor.getProperties()) {
    if (!firstProperty)
      os << ',';
    firstProperty = false;
    appendJsonString(os, property.first);
    os << ':';
    appendJsonString(os, property.second);
  }
  os << "},\"provides\":";
  appendSortedIDArray(os, descriptor.getProvidedIDs());
  os << ",\"status\":";
  appendJsonString(os, descriptor.getStatus());
  os << '}';
}

} // namespace

std::string
serializeDeclaredInstanceFactSet(const TargetCapabilitySet &capabilities) {
  // Sort the expanded fact set by id BEFORE serializing. This is the single most
  // important correctness step: profile expansion may emit descriptors in a
  // different order than an equivalent explicit list, so without the sort the
  // "profile == explicit list ==> same hash" property fails.
  llvm::SmallVector<const CapabilityDescriptor *, 8> sorted;
  sorted.reserve(capabilities.getCapabilities().size());
  for (const CapabilityDescriptor &descriptor : capabilities.getCapabilities())
    sorted.push_back(&descriptor);
  llvm::stable_sort(sorted, [](const CapabilityDescriptor *lhs,
                               const CapabilityDescriptor *rhs) {
    return lhs->getID() < rhs->getID();
  });

  std::string serialized;
  llvm::raw_string_ostream os(serialized);
  os << '[';
  for (std::size_t index = 0; index < sorted.size(); ++index) {
    if (index)
      os << ',';
    appendDescriptor(os, *sorted[index]);
  }
  os << ']';
  os.flush();
  return serialized;
}

std::string
computeDeclaredInstanceHash(const TargetCapabilitySet &capabilities) {
  std::string serialized = serializeDeclaredInstanceFactSet(capabilities);
  std::array<uint8_t, 32> digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(serialized.data()), serialized.size()));
  return llvm::toHex(digest, /*LowerCase=*/true);
}

} // namespace weft::support
