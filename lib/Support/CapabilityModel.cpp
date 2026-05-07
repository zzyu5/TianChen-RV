#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace tianchenrv::support {
namespace {

llvm::StringRef getStringAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return {};
  return attr.getValue();
}

llvm::StringRef getCapabilityStatus(tcrv::exec::CapabilityOp capability) {
  if (llvm::StringRef status =
          getStringAttr(capability.getOperation(), "status");
      !status.empty())
    return status;
  return getStringAttr(capability.getOperation(), "availability");
}

bool isCoreCapabilityAttribute(llvm::StringRef attrName) {
  return attrName == "sym_name" || attrName == "id" || attrName == "kind" ||
         attrName == "status" || attrName == "availability";
}

std::string stringifyCapabilityProperty(mlir::Attribute attribute) {
  if (auto stringAttr = llvm::dyn_cast<mlir::StringAttr>(attribute))
    return stringAttr.getValue().str();

  if (auto boolAttr = llvm::dyn_cast<mlir::BoolAttr>(attribute))
    return boolAttr.getValue() ? "true" : "false";

  if (auto integerAttr = llvm::dyn_cast<mlir::IntegerAttr>(attribute)) {
    llvm::SmallString<32> text;
    integerAttr.getValue().toString(text, 10, /*Signed=*/true);
    return text.str().str();
  }

  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attribute)) {
    llvm::SmallString<32> text;
    floatAttr.getValue().toString(text);
    return text.str().str();
  }

  if (auto symbolAttr = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attribute))
    return symbolAttr.getValue().str();

  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return stream.str();
}

std::map<std::string, std::string>
collectCapabilityProperties(tcrv::exec::CapabilityOp capability) {
  std::map<std::string, std::string> properties;
  for (mlir::NamedAttribute namedAttribute : capability->getAttrs()) {
    llvm::StringRef attrName = namedAttribute.getName().getValue();
    if (isCoreCapabilityAttribute(attrName))
      continue;

    properties.try_emplace(
        attrName.str(), stringifyCapabilityProperty(namedAttribute.getValue()));
  }
  return properties;
}

} // namespace

CapabilityDescriptor::CapabilityDescriptor(
    llvm::StringRef symbolName, llvm::StringRef id, llvm::StringRef kind,
    llvm::StringRef status, CapabilityAvailability availability,
    std::map<std::string, std::string> properties)
    : symbolName(symbolName.str()), id(id.str()), kind(kind.str()),
      status(status.str()), availability(availability),
      properties(std::move(properties)) {}

llvm::StringRef CapabilityDescriptor::getProperty(llvm::StringRef name) const {
  auto it = properties.find(name.str());
  if (it == properties.end())
    return {};
  return it->second;
}

TargetCapabilitySet
TargetCapabilitySet::buildFromKernel(tcrv::exec::KernelOp kernel) {
  TargetCapabilitySet capabilitySet;
  if (!kernel || kernel.getBody().empty())
    return capabilitySet;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto capability = llvm::dyn_cast<tcrv::exec::CapabilityOp>(op);
    if (!capability)
      continue;

    llvm::StringRef status = getCapabilityStatus(capability);
    capabilitySet.addCapability(CapabilityDescriptor(
        capability.getSymName(), capability.getId().value_or(""),
        capability.getKind().value_or(""), status,
        availabilityFromStatus(status),
        collectCapabilityProperties(capability)));
  }

  return capabilitySet;
}

const CapabilityDescriptor *
TargetCapabilitySet::lookupBySymbolName(llvm::StringRef symbolName) const {
  auto it = bySymbolName.find(symbolName);
  if (it == bySymbolName.end())
    return nullptr;
  return &capabilities[it->second];
}

const CapabilityDescriptor *
TargetCapabilitySet::lookupByID(llvm::StringRef id) const {
  auto it = byID.find(id);
  if (it == byID.end())
    return nullptr;
  return &capabilities[it->second];
}

void TargetCapabilitySet::collectByKind(
    llvm::StringRef kind,
    llvm::SmallVectorImpl<const CapabilityDescriptor *> &out) const {
  for (const CapabilityDescriptor &capability : capabilities) {
    if (capability.getKind() == kind)
      out.push_back(&capability);
  }
}

llvm::SmallVector<const CapabilityDescriptor *, 4>
TargetCapabilitySet::collectByKind(llvm::StringRef kind) const {
  llvm::SmallVector<const CapabilityDescriptor *, 4> matches;
  collectByKind(kind, matches);
  return matches;
}

bool TargetCapabilitySet::isCapabilityAvailableBySymbolName(
    llvm::StringRef symbolName) const {
  const CapabilityDescriptor *capability = lookupBySymbolName(symbolName);
  return capability && capability->isAvailable();
}

bool TargetCapabilitySet::isCapabilityAvailableByID(llvm::StringRef id) const {
  const CapabilityDescriptor *capability = lookupByID(id);
  return capability && capability->isAvailable();
}

CapabilityAvailability
TargetCapabilitySet::availabilityFromStatus(llvm::StringRef status) {
  return isUnavailableStatus(status) ? CapabilityAvailability::Unavailable
                                     : CapabilityAvailability::Available;
}

bool TargetCapabilitySet::isUnavailableStatus(llvm::StringRef status) {
  std::string normalized = status.trim().lower();
  return llvm::StringSwitch<bool>(normalized)
      .Case("unavailable", true)
      .Case("disabled", true)
      .Case("missing", true)
      .Default(false);
}

void TargetCapabilitySet::addCapability(CapabilityDescriptor descriptor) {
  std::size_t index = capabilities.size();
  capabilities.push_back(std::move(descriptor));
  bySymbolName.try_emplace(capabilities.back().getSymbolName(), index);
  byID.try_emplace(capabilities.back().getID(), index);
}

} // namespace tianchenrv::support
