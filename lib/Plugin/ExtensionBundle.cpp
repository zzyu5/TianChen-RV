#include "TianChenRV/Plugin/ExtensionBundle.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

#include <string>

namespace tianchenrv::plugin {
namespace {

llvm::Error makeExtensionBundleRegistryError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV extension bundle registry failed: ") + message,
      llvm::errc::invalid_argument);
}

bool containsForbiddenText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key") ||
         normalized.contains("raw log") || normalized.contains("raw-log") ||
         normalized.contains("http://") || normalized.contains("https://") ||
         normalized.contains("://");
}

llvm::Error validateExtensionBundleRegistryText(llvm::StringRef fieldName,
                                                llvm::StringRef value) {
  constexpr std::size_t kMaxTextLength = 512;
  if (value.empty() || value.size() > kMaxTextLength)
    return makeExtensionBundleRegistryError(
        llvm::Twine(fieldName) +
        " must be bounded non-empty single-line metadata");

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return makeExtensionBundleRegistryError(
          llvm::Twine(fieldName) +
          " must be bounded non-empty single-line metadata");
    if (byte < 0x20 && character != '\t')
      return makeExtensionBundleRegistryError(
          llvm::Twine(fieldName) +
          " must be bounded non-empty single-line metadata");
  }

  if (containsForbiddenText(value))
    return makeExtensionBundleRegistryError(
        llvm::Twine(fieldName) +
        " must not contain secret-like or raw credential text");
  return llvm::Error::success();
}

llvm::Error validateUniqueTextList(llvm::StringRef bundleID,
                                   llvm::StringRef fieldName,
                                   llvm::ArrayRef<std::string> values) {
  llvm::StringSet<> seen;
  for (llvm::StringRef value : values) {
    if (llvm::Error error =
            validateExtensionBundleRegistryText(fieldName, value))
      return error;
    if (!seen.insert(value).second)
      return makeExtensionBundleRegistryError(
          llvm::Twine("extension bundle '") + bundleID +
          "' has duplicate " + fieldName + " '" + value + "'");
  }
  return llvm::Error::success();
}

} // namespace

ExtensionBundle::ExtensionBundle(
    llvm::StringRef bundleID, llvm::StringRef pluginName,
    ExtensionPluginRegistrationFn pluginRegistrationFn)
    : bundleID(bundleID.str()), pluginName(pluginName.str()),
      pluginRegistrationFn(pluginRegistrationFn) {}

void ExtensionBundle::addRequiredDialectName(llvm::StringRef dialectName) {
  requiredDialectNames.push_back(dialectName.str());
}

void ExtensionBundle::addLoweringBoundaryOp(llvm::StringRef opName) {
  loweringBoundaryOps.push_back(opName.str());
}

void ExtensionBundle::setTargetArtifactExporterBundleRegistrationFn(
    PluginTargetArtifactExporterBundleRegistrationFn registrationFn) {
  targetArtifactExporterBundleRegistrationFn = registrationFn;
}

llvm::Error ExtensionBundleRegistry::registerBundle(
    const ExtensionBundle &bundle) {
  if (llvm::Error error = validateExtensionBundleRegistryText(
          "extension bundle id", bundle.getBundleID()))
    return error;
  if (llvm::Error error = validateExtensionBundleRegistryText(
          "extension bundle plugin name", bundle.getPluginName()))
    return error;
  if (!bundle.getPluginRegistrationFn())
    return makeExtensionBundleRegistryError(
        llvm::Twine("extension bundle '") + bundle.getBundleID() +
        "' must have a non-null extension plugin registration callback");
  if (bundleIndicesByID.count(bundle.getBundleID()))
    return makeExtensionBundleRegistryError(
        llvm::Twine("duplicate extension bundle id '") +
        bundle.getBundleID() + "'");
  if (bundleIndicesByPlugin.count(bundle.getPluginName()))
    return makeExtensionBundleRegistryError(
        llvm::Twine("duplicate extension bundle plugin id '") +
        bundle.getPluginName() + "'");

  if (llvm::Error error =
          validateUniqueTextList(bundle.getBundleID(), "required dialect",
                                 bundle.getRequiredDialectNames()))
    return error;
  if (llvm::Error error = validateUniqueTextList(
          bundle.getBundleID(), "lowering boundary op",
          bundle.getLoweringBoundaryOps()))
    return error;

  std::size_t index = bundles.size();
  bundles.push_back(bundle);
  bundleIndicesByID[bundles.back().getBundleID()] = index;
  bundleIndicesByPlugin[bundles.back().getPluginName()] = index;
  return llvm::Error::success();
}

const ExtensionBundle *
ExtensionBundleRegistry::lookupBundle(llvm::StringRef bundleID) const {
  auto it = bundleIndicesByID.find(bundleID);
  if (it == bundleIndicesByID.end())
    return nullptr;
  return &bundles[it->getValue()];
}

const ExtensionBundle *
ExtensionBundleRegistry::lookupPluginBundle(llvm::StringRef pluginName) const {
  auto it = bundleIndicesByPlugin.find(pluginName);
  if (it == bundleIndicesByPlugin.end())
    return nullptr;
  return &bundles[it->getValue()];
}

llvm::Error ExtensionBundleRegistry::registerExtensionPlugins(
    ExtensionPluginRegistry &plugins) const {
  for (const ExtensionBundle &bundle : bundles) {
    if (llvm::Error error = bundle.getPluginRegistrationFn()(plugins)) {
      std::string message = llvm::toString(std::move(error));
      return makeExtensionBundleRegistryError(
          llvm::Twine("extension bundle '") + bundle.getBundleID() +
          "' failed to register extension plugin '" +
          bundle.getPluginName() + "': " + message);
    }
  }
  return llvm::Error::success();
}

llvm::Error ExtensionBundleRegistry::registerTargetArtifactExporterBundles(
    target::PluginTargetArtifactExporterRegistry &registry) const {
  for (const ExtensionBundle &bundle : bundles) {
    PluginTargetArtifactExporterBundleRegistrationFn registrationFn =
        bundle.getTargetArtifactExporterBundleRegistrationFn();
    if (!registrationFn)
      continue;

    if (llvm::Error error = registrationFn(registry)) {
      std::string message = llvm::toString(std::move(error));
      return makeExtensionBundleRegistryError(
          llvm::Twine("extension bundle '") + bundle.getBundleID() +
          "' failed to register plugin-owned target artifact exporter "
          "bundle: " +
          message);
    }
  }

  return llvm::Error::success();
}

} // namespace tianchenrv::plugin
