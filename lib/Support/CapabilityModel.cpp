#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace tianchenrv::support {
namespace {

constexpr llvm::StringLiteral kProvidesAttrName("provides");
constexpr llvm::StringLiteral kImpliesAttrName("implies");
constexpr llvm::StringLiteral kConflictsAttrName("conflicts");
constexpr llvm::StringLiteral kTargetHartCountCapabilityID(
    "target.hart_count");
constexpr llvm::StringLiteral kHartCountPropertyName("count");

llvm::Error makeCapabilitySetError(llvm::Twine message);
std::string makeKernelExtractionContext(tcrv::exec::KernelOp kernel);

llvm::StringRef getStringAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return {};
  return attr.getValue();
}

llvm::StringRef getCapabilityStatus(mlir::Operation *op) {
  if (llvm::StringRef status = getStringAttr(op, "status"); !status.empty())
    return status;
  return getStringAttr(op, "availability");
}

bool isCoreCapabilityAttribute(llvm::StringRef attrName) {
  return attrName == "sym_name" || attrName == "id" || attrName == "kind" ||
         attrName == "status" || attrName == "availability";
}

bool isCapabilityRelationAttribute(llvm::StringRef attrName) {
  return attrName == kProvidesAttrName || attrName == kImpliesAttrName ||
         attrName == kConflictsAttrName;
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
collectCapabilityProperties(mlir::Operation *op) {
  std::map<std::string, std::string> properties;
  for (mlir::NamedAttribute namedAttribute : op->getAttrs()) {
    llvm::StringRef attrName = namedAttribute.getName().getValue();
    if (isCoreCapabilityAttribute(attrName) ||
        isCapabilityRelationAttribute(attrName))
      continue;

    properties.try_emplace(
        attrName.str(), stringifyCapabilityProperty(namedAttribute.getValue()));
  }
  return properties;
}

llvm::SmallVector<std::string, 4>
collectCapabilityIDRelation(mlir::Operation *op,
                            llvm::StringRef attrName) {
  llvm::SmallVector<std::string, 4> ids;
  auto arrayAttr = op->getAttrOfType<mlir::ArrayAttr>(attrName);
  if (!arrayAttr)
    return ids;

  for (mlir::Attribute attr : arrayAttr) {
    auto stringAttr = llvm::dyn_cast<mlir::StringAttr>(attr);
    if (!stringAttr)
      continue;

    llvm::StringRef value = stringAttr.getValue().trim();
    if (!value.empty())
      ids.push_back(value.str());
  }

  return ids;
}

bool hasCapabilityProviderIdentity(tcrv::exec::TargetOp target) {
  return !getStringAttr(target.getOperation(), "id").trim().empty() &&
         !getStringAttr(target.getOperation(), "kind").trim().empty();
}

mlir::Operation *findModuleLevelSymbol(tcrv::exec::KernelOp kernel,
                                       llvm::StringRef symbolName) {
  auto module = kernel ? kernel->getParentOfType<mlir::ModuleOp>()
                       : mlir::ModuleOp();
  if (!module || module.getBodyRegion().empty())
    return nullptr;

  for (mlir::Operation &op : module.getBody()->getOperations()) {
    auto symbolAttr = op.getAttrOfType<mlir::StringAttr>(
        mlir::SymbolTable::getSymbolAttrName());
    if (symbolAttr && symbolAttr.getValue() == symbolName)
      return &op;
  }
  return nullptr;
}

llvm::Expected<tcrv::exec::TargetOp>
getReferencedModuleTargetProvider(tcrv::exec::KernelOp kernel) {
  if (!kernel)
    return tcrv::exec::TargetOp();

  mlir::Attribute rawTargetAttr = kernel->getAttr("target");
  if (!rawTargetAttr)
    return tcrv::exec::TargetOp();

  auto targetAttr = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(rawTargetAttr);
  if (!targetAttr)
    return makeCapabilitySetError(
        llvm::Twine("TianChen-RV TargetCapabilitySet ") +
        makeKernelExtractionContext(kernel) +
        " rejected malformed kernel target attribute; expected a module-level "
        "tcrv.exec.target symbol reference");

  mlir::Operation *resolved =
      findModuleLevelSymbol(kernel, targetAttr.getValue());
  if (!resolved)
    return makeCapabilitySetError(
        llvm::Twine("TianChen-RV TargetCapabilitySet ") +
        makeKernelExtractionContext(kernel) +
        " rejected unknown module-level target @" + targetAttr.getValue());

  auto target = llvm::dyn_cast<tcrv::exec::TargetOp>(resolved);
  if (!target)
    return makeCapabilitySetError(
        llvm::Twine("TianChen-RV TargetCapabilitySet ") +
        makeKernelExtractionContext(kernel) + " rejected target @" +
        targetAttr.getValue() +
        " because it resolves to a module-level symbol that is not a "
        "tcrv.exec.target");

  if (!hasCapabilityProviderIdentity(target))
    return makeCapabilitySetError(
        llvm::Twine("TianChen-RV TargetCapabilitySet ") +
        makeKernelExtractionContext(kernel) + " rejected target @" +
        targetAttr.getValue() +
        " because it lacks capability-provider id/kind identity");

  return target;
}

CapabilityDescriptor makeDescriptor(mlir::Operation *op,
                                    llvm::StringRef symbolName,
                                    llvm::StringRef id,
                                    llvm::StringRef kind) {
  llvm::StringRef status = getCapabilityStatus(op);
  return CapabilityDescriptor(
      symbolName, id, kind, status,
      TargetCapabilitySet::availabilityFromStatus(status),
      collectCapabilityProperties(op),
      collectCapabilityIDRelation(op, kProvidesAttrName),
      collectCapabilityIDRelation(op, kImpliesAttrName),
      collectCapabilityIDRelation(op, kConflictsAttrName));
}

bool containsID(llvm::ArrayRef<std::string> ids, llvm::StringRef id) {
  return llvm::any_of(ids, [&](const std::string &candidate) {
    return candidate == id;
  });
}

llvm::Error makeCapabilitySetError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      message, llvm::errc::invalid_argument);
}

std::string makeKernelExtractionContext(tcrv::exec::KernelOp kernel) {
  if (!kernel)
    return "kernel extraction from <missing kernel>";

  std::string context;
  llvm::raw_string_ostream stream(context);
  stream << "kernel extraction from @" << kernel.getSymName();
  return stream.str();
}

} // namespace

llvm::StringRef getTargetHartCountCapabilityID() {
  return kTargetHartCountCapabilityID;
}

llvm::StringRef getHartCountPropertyName() {
  return kHartCountPropertyName;
}

CapabilityDescriptor::CapabilityDescriptor(
    llvm::StringRef symbolName, llvm::StringRef id, llvm::StringRef kind,
    llvm::StringRef status, CapabilityAvailability availability,
    std::map<std::string, std::string> properties,
    llvm::ArrayRef<std::string> providedIDs,
    llvm::ArrayRef<std::string> impliedIDs,
    llvm::ArrayRef<std::string> conflictingIDs)
    : symbolName(symbolName.str()), id(id.str()), kind(kind.str()),
      status(status.str()), availability(availability),
      properties(std::move(properties)) {
  this->providedIDs.append(providedIDs.begin(), providedIDs.end());
  this->impliedIDs.append(impliedIDs.begin(), impliedIDs.end());
  this->conflictingIDs.append(conflictingIDs.begin(), conflictingIDs.end());
}

llvm::StringRef CapabilityDescriptor::getProperty(llvm::StringRef name) const {
  auto it = properties.find(name.str());
  if (it == properties.end())
    return {};
  return it->second;
}

bool CapabilityDescriptor::providesID(llvm::StringRef capabilityID) const {
  return containsID(providedIDs, capabilityID);
}

bool CapabilityDescriptor::impliesID(llvm::StringRef capabilityID) const {
  return containsID(impliedIDs, capabilityID);
}

bool CapabilityDescriptor::conflictsWithID(llvm::StringRef capabilityID) const {
  return containsID(conflictingIDs, capabilityID);
}

bool CapabilityDescriptor::satisfiesID(llvm::StringRef capabilityID) const {
  return getID() == capabilityID || providesID(capabilityID) ||
         impliesID(capabilityID);
}

llvm::Expected<TargetCapabilitySet>
TargetCapabilitySet::buildFromKernelChecked(tcrv::exec::KernelOp kernel) {
  TargetCapabilitySet capabilitySet;
  if (!kernel || kernel.getBody().empty())
    return capabilitySet;

  std::string constructionContext = makeKernelExtractionContext(kernel);
  llvm::Expected<tcrv::exec::TargetOp> referencedTarget =
      getReferencedModuleTargetProvider(kernel);
  if (!referencedTarget)
    return referencedTarget.takeError();

  if (*referencedTarget) {
    tcrv::exec::TargetOp target = *referencedTarget;
    if (llvm::Error error = capabilitySet.tryAddCapability(
            makeDescriptor(target.getOperation(), target.getSymName(),
                           getStringAttr(target.getOperation(), "id"),
                           getStringAttr(target.getOperation(), "kind")),
            constructionContext))
      return std::move(error);
  }

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (auto capability = llvm::dyn_cast<tcrv::exec::CapabilityOp>(op)) {
      if (llvm::Error error = capabilitySet.tryAddCapability(
              makeDescriptor(capability.getOperation(), capability.getSymName(),
                             capability.getId().value_or(""),
                             capability.getKind().value_or("")),
              constructionContext))
        return std::move(error);
      continue;
    }

    if (auto target = llvm::dyn_cast<tcrv::exec::TargetOp>(op)) {
      if (!hasCapabilityProviderIdentity(target))
        continue;

      if (llvm::Error error = capabilitySet.tryAddCapability(
              makeDescriptor(target.getOperation(), target.getSymName(),
                             getStringAttr(target.getOperation(), "id"),
                             getStringAttr(target.getOperation(), "kind")),
              constructionContext))
        return std::move(error);
    }
  }

  return capabilitySet;
}

TargetCapabilitySet
TargetCapabilitySet::buildFromKernel(tcrv::exec::KernelOp kernel) {
  llvm::Expected<TargetCapabilitySet> capabilities =
      buildFromKernelChecked(kernel);
  return llvm::cantFail(std::move(capabilities));
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

const CapabilityDescriptor *
TargetCapabilitySet::lookupProviderByID(llvm::StringRef id) const {
  if (const CapabilityDescriptor *exact = lookupByID(id))
    return exact;

  for (const CapabilityDescriptor &capability : capabilities) {
    if (capability.satisfiesID(id) && capability.isAvailable())
      return &capability;
  }

  for (const CapabilityDescriptor &capability : capabilities) {
    if (capability.satisfiesID(id))
      return &capability;
  }

  return nullptr;
}

void TargetCapabilitySet::collectProvidersByID(
    llvm::StringRef id,
    llvm::SmallVectorImpl<const CapabilityDescriptor *> &out) const {
  if (const CapabilityDescriptor *exact = lookupByID(id)) {
    out.push_back(exact);
    return;
  }

  for (const CapabilityDescriptor &capability : capabilities) {
    if (capability.satisfiesID(id))
      out.push_back(&capability);
  }
}

void TargetCapabilitySet::collectAvailableConflictsForCapability(
    const CapabilityDescriptor &requiredCapability,
    llvm::SmallVectorImpl<CapabilityConflict> &out) const {
  if (!requiredCapability.isAvailable())
    return;

  llvm::StringSet<> seenConflicts;
  auto appendConflict = [&](const CapabilityDescriptor &conflictingCapability,
                            const CapabilityDescriptor &relationOwner,
                            llvm::StringRef conflictID) {
    if (&conflictingCapability == &requiredCapability ||
        !conflictingCapability.isAvailable())
      return;

    std::string key;
    llvm::raw_string_ostream stream(key);
    stream << requiredCapability.getSymbolName() << "\n"
           << conflictingCapability.getSymbolName() << "\n"
           << relationOwner.getSymbolName() << "\n" << conflictID;
    stream.flush();
    if (!seenConflicts.insert(key).second)
      return;

    CapabilityConflict conflict;
    conflict.requiredCapability = &requiredCapability;
    conflict.conflictingCapability = &conflictingCapability;
    conflict.relationOwner = &relationOwner;
    conflict.conflictID = conflictID.str();
    out.push_back(std::move(conflict));
  };

  for (const std::string &conflictID : requiredCapability.getConflictingIDs()) {
    llvm::SmallVector<const CapabilityDescriptor *, 4> providers;
    collectProvidersByID(conflictID, providers);
    for (const CapabilityDescriptor *provider : providers)
      if (provider)
        appendConflict(*provider, requiredCapability, conflictID);
  }

  for (const CapabilityDescriptor &candidate : capabilities) {
    if (&candidate == &requiredCapability || !candidate.isAvailable())
      continue;

    for (const std::string &conflictID : candidate.getConflictingIDs()) {
      if (requiredCapability.satisfiesID(conflictID))
        appendConflict(candidate, candidate, conflictID);
    }
  }
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
  if (const CapabilityDescriptor *exact = lookupByID(id))
    return exact->isAvailable();

  for (const CapabilityDescriptor &capability : capabilities) {
    if (capability.satisfiesID(id) && capability.isAvailable())
      return true;
  }

  return false;
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

llvm::Error TargetCapabilitySet::tryAddCapability(
    CapabilityDescriptor descriptor, llvm::StringRef constructionContext) {
  llvm::StringRef symbolName = descriptor.getSymbolName();
  if (auto it = bySymbolName.find(symbolName); it != bySymbolName.end()) {
    const CapabilityDescriptor &existing = capabilities[it->second];
    return makeCapabilitySetError(
        llvm::Twine("TianChen-RV TargetCapabilitySet ") +
        constructionContext + " rejected duplicate capability symbol @" +
        symbolName + " for id \"" + descriptor.getID() +
        "\"; existing id \"" + existing.getID() + "\"");
  }

  llvm::StringRef id = descriptor.getID();
  if (auto it = byID.find(id); it != byID.end()) {
    const CapabilityDescriptor &existing = capabilities[it->second];
    return makeCapabilitySetError(
        llvm::Twine("TianChen-RV TargetCapabilitySet ") +
        constructionContext + " rejected duplicate capability id \"" + id +
        "\" for symbol @" + symbolName + "; existing symbol @" +
        existing.getSymbolName());
  }

  std::size_t index = capabilities.size();
  capabilities.push_back(std::move(descriptor));
  bySymbolName.try_emplace(capabilities.back().getSymbolName(), index);
  byID.try_emplace(capabilities.back().getID(), index);
  return llvm::Error::success();
}

void TargetCapabilitySet::addCapability(CapabilityDescriptor descriptor) {
  if (llvm::Error error =
          tryAddCapability(std::move(descriptor), "synthetic construction")) {
    std::string message = llvm::toString(std::move(error));
    llvm::report_fatal_error(llvm::StringRef(message));
  }
}

} // namespace tianchenrv::support
