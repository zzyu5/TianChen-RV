#include "TianChenRV/Target/ConstructionTemplateArtifactAdapter.h"

#include "TianChenRV/Plugin/ConstructionProtocol.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"

namespace tianchenrv::target {
namespace {

namespace construction = tianchenrv::plugin::construction;

constexpr llvm::StringLiteral kRuntimeCallableCHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kRiscvELFRelocatableObjectArtifactKind(
    "riscv-elf-relocatable-object");

llvm::Error makeConstructionTemplateAdapterError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV construction-template artifact adapter "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error requireNonEmpty(llvm::StringRef fieldName,
                            llvm::StringRef value) {
  if (!value.trim().empty())
    return llvm::Error::success();
  return makeConstructionTemplateAdapterError(
      llvm::Twine("requires non-empty ") + fieldName);
}

llvm::StringRef getSelectedLoweringBoundaryDescription(
    const ConstructionTemplateArtifactAdapterConfig &config) {
  if (!config.selectedLoweringBoundary.boundaryDescription.trim().empty())
    return config.selectedLoweringBoundary.boundaryDescription;
  return "selected construction-template artifact boundary";
}

llvm::Error validateSelectedLoweringBoundaryConfig(
    const ConstructionTemplateArtifactAdapterConfig &config) {
  const ConstructionTemplateSelectedLoweringBoundaryConfig &boundary =
      config.selectedLoweringBoundary;
  if (!boundary.required)
    return llvm::Error::success();

  if (llvm::Error error =
          requireNonEmpty("selected lowering-boundary status", boundary.status))
    return error;
  if (llvm::Error error = requireNonEmpty(
          "selected lowering-boundary source-kernel attribute name",
          boundary.sourceKernelAttrName))
    return error;
  if (llvm::Error error = requireNonEmpty(
          "selected lowering-boundary selected-variant attribute name",
          boundary.selectedVariantAttrName))
    return error;
  if (llvm::Error error = requireNonEmpty(
          "selected lowering-boundary origin attribute name",
          boundary.originAttrName))
    return error;
  if (llvm::Error error = requireNonEmpty(
          "selected lowering-boundary role attribute name",
          boundary.roleAttrName))
    return error;
  if (llvm::Error error = requireNonEmpty(
          "selected lowering-boundary status attribute name",
          boundary.statusAttrName))
    return error;
  if (llvm::Error error = requireNonEmpty(
          "selected lowering-boundary required-capabilities attribute name",
          boundary.requiredCapabilitiesAttrName))
    return error;

  for (const ConstructionTemplateSelectedBoundaryAttributeExpectation
           &expectation : boundary.extraStringAttributes) {
    if (llvm::Error error =
            requireNonEmpty("selected lowering-boundary extra attribute name",
                            expectation.boundaryAttrName))
      return error;
    bool hasStaticValue = !expectation.expectedValue.trim().empty();
    bool hasVariantAttr = !expectation.variantAttrName.trim().empty();
    if (hasStaticValue == hasVariantAttr)
      return makeConstructionTemplateAdapterError(
          llvm::Twine("selected lowering-boundary extra attribute '") +
          expectation.boundaryAttrName +
          "' must provide exactly one expected value or selected-variant "
          "attribute source");
  }

  return llvm::Error::success();
}

llvm::Expected<mlir::Operation *> findSelectedLoweringBoundary(
    const SelectedEmitCArtifactTarget &target,
    const ConstructionTemplateArtifactAdapterConfig &config) {
  tcrv::exec::KernelOp kernel = target.kernel;
  tcrv::exec::VariantOp variant = target.variant;
  if (!kernel)
    return makeConstructionTemplateAdapterError(
        "selected lowering-boundary validation requires an enclosing "
        "tcrv.exec.kernel");
  if (!variant)
    return makeConstructionTemplateAdapterError(
        "selected lowering-boundary validation requires a materialized "
        "tcrv.exec.variant");
  if (kernel.getBody().empty())
    return makeConstructionTemplateAdapterError(
        "selected lowering-boundary validation requires a materialized kernel "
        "body");

  const ConstructionTemplateSelectedLoweringBoundaryConfig &boundary =
      config.selectedLoweringBoundary;
  mlir::Operation *selectedBoundary = nullptr;
  unsigned matchingBoundaries = 0;

  if (boundary.searchSelectedVariantBody) {
    variant.getBody().walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() != config.loweringBoundary)
        return;
      selectedBoundary = op;
      ++matchingBoundaries;
    });
    if (matchingBoundaries == 0)
      return makeConstructionTemplateAdapterError(
          llvm::Twine("requires one selected materialized ") +
          config.loweringBoundary + " before artifact export");
    if (matchingBoundaries != 1)
      return makeConstructionTemplateAdapterError(
          llvm::Twine("requires exactly one selected materialized ") +
          config.loweringBoundary + " before artifact export");

    return selectedBoundary;
  }

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (op.getName().getStringRef() != config.loweringBoundary)
      continue;

    auto selectedVariant = op.getAttrOfType<mlir::FlatSymbolRefAttr>(
        boundary.selectedVariantAttrName);
    auto role = op.getAttrOfType<mlir::StringAttr>(boundary.roleAttrName);
    if (!selectedVariant || selectedVariant.getValue() != variant.getSymName() ||
        !role ||
        role.getValue() != target.candidate.role)
      continue;

    selectedBoundary = &op;
    ++matchingBoundaries;
  }

  if (matchingBoundaries == 0)
    return makeConstructionTemplateAdapterError(
        llvm::Twine("requires one selected materialized ") +
        config.loweringBoundary + " before artifact export");
  if (matchingBoundaries != 1)
    return makeConstructionTemplateAdapterError(
        llvm::Twine("requires exactly one selected materialized ") +
        config.loweringBoundary + " before artifact export");

  return selectedBoundary;
}

void setMissingBoundaryAttr(mlir::Operation *boundary,
                            llvm::StringRef attrName, mlir::Attribute value,
                            llvm::SmallVectorImpl<llvm::StringRef> &added) {
  if (boundary->getAttr(attrName))
    return;
  boundary->setAttr(attrName, value);
  added.push_back(attrName);
}

llvm::Error verifySelectedLoweringBoundaryConformanceWithConfig(
    mlir::Operation *boundary,
    const construction::SelectedLoweringBoundaryConformanceSpec &spec,
    bool synthesizeMissingConformanceAttributes) {
  if (!synthesizeMissingConformanceAttributes)
    return construction::verifySelectedLoweringBoundaryConformance(boundary,
                                                                   spec);

  llvm::SmallVector<llvm::StringRef, 8> addedAttrs;
  mlir::MLIRContext *context = boundary->getContext();
  setMissingBoundaryAttr(
      boundary, spec.sourceKernelAttrName,
      mlir::StringAttr::get(context, spec.sourceKernelSymbol), addedAttrs);
  setMissingBoundaryAttr(
      boundary, spec.selectedVariantAttrName,
      mlir::FlatSymbolRefAttr::get(context, spec.selectedVariantSymbol),
      addedAttrs);
  setMissingBoundaryAttr(boundary, spec.originAttrName,
                         mlir::StringAttr::get(context, spec.originPlugin),
                         addedAttrs);
  setMissingBoundaryAttr(boundary, spec.roleAttrName,
                         mlir::StringAttr::get(context, spec.pathRole),
                         addedAttrs);
  setMissingBoundaryAttr(boundary, spec.statusAttrName,
                         mlir::StringAttr::get(context, spec.status),
                         addedAttrs);
  setMissingBoundaryAttr(boundary, spec.requiredCapabilitiesAttrName,
                         spec.requiredCapabilities, addedAttrs);

  llvm::Error error =
      construction::verifySelectedLoweringBoundaryConformance(boundary, spec);
  for (llvm::StringRef attrName : addedAttrs)
    boundary->removeAttr(attrName);
  return error;
}

llvm::Expected<
    llvm::SmallVector<construction::SelectedBoundaryStringAttrExpectation, 4>>
resolveSelectedLoweringBoundaryExtraAttributes(
    const SelectedEmitCArtifactTarget &target,
    const ConstructionTemplateArtifactAdapterConfig &config) {
  llvm::SmallVector<construction::SelectedBoundaryStringAttrExpectation, 4>
      resolved;
  for (const ConstructionTemplateSelectedBoundaryAttributeExpectation
           &expectation :
       config.selectedLoweringBoundary.extraStringAttributes) {
    llvm::StringRef expectedValue = expectation.expectedValue;
    if (!expectation.variantAttrName.empty()) {
      auto variantAttr = target.variant->getAttrOfType<mlir::StringAttr>(
          expectation.variantAttrName);
      if (!variantAttr || variantAttr.getValue().trim().empty())
        return makeConstructionTemplateAdapterError(
            llvm::Twine("selected variant requires non-empty string "
                        "attribute '") +
            expectation.variantAttrName +
            "' before selected lowering-boundary validation");
      expectedValue = variantAttr.getValue().trim();
    }
    resolved.push_back({expectation.boundaryAttrName, expectedValue});
  }
  return resolved;
}

llvm::Error validateSelectedLoweringBoundary(
    const SelectedEmitCArtifactTarget &target,
    const ConstructionTemplateArtifactAdapterConfig &config) {
  if (!config.selectedLoweringBoundary.required)
    return llvm::Error::success();
  tcrv::exec::KernelOp kernel = target.kernel;
  tcrv::exec::VariantOp variant = target.variant;

  if (llvm::StringRef(target.candidate.loweringBoundary) !=
      config.loweringBoundary)
    return makeConstructionTemplateAdapterError(
        "selected artifact candidate lowering boundary does not match the "
        "construction-template adapter configuration");

  llvm::Expected<mlir::Operation *> boundary =
      findSelectedLoweringBoundary(target, config);
  if (!boundary)
    return boundary.takeError();

  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requiredCapabilities)
    return makeConstructionTemplateAdapterError(
        "selected variant requires `requires` capability metadata before "
        "selected lowering-boundary validation");

  llvm::Expected<
      llvm::SmallVector<construction::SelectedBoundaryStringAttrExpectation, 4>>
      extraAttributes =
          resolveSelectedLoweringBoundaryExtraAttributes(target, config);
  if (!extraAttributes)
    return extraAttributes.takeError();

  const ConstructionTemplateSelectedLoweringBoundaryConfig &boundaryConfig =
      config.selectedLoweringBoundary;
  construction::SelectedLoweringBoundaryConformanceSpec spec;
  spec.boundaryDescription = getSelectedLoweringBoundaryDescription(config);
  spec.selectedVariantSymbol = variant.getSymName();
  spec.sourceKernelSymbol = kernel.getSymName();
  spec.originPlugin = config.selectedRoute.originPlugin;
  spec.pathRole = target.candidate.role;
  spec.status = boundaryConfig.status;
  spec.requiredCapabilities = requiredCapabilities;
  spec.extraStringAttributes = *extraAttributes;
  spec.sourceKernelAttrName = boundaryConfig.sourceKernelAttrName;
  spec.selectedVariantAttrName = boundaryConfig.selectedVariantAttrName;
  spec.originAttrName = boundaryConfig.originAttrName;
  spec.roleAttrName = boundaryConfig.roleAttrName;
  spec.statusAttrName = boundaryConfig.statusAttrName;
  spec.requiredCapabilitiesAttrName =
      boundaryConfig.requiredCapabilitiesAttrName;
  return verifySelectedLoweringBoundaryConformanceWithConfig(
      *boundary, spec, boundaryConfig.synthesizeMissingConformanceAttributes);
}

SelectedEmitCArtifactRouteConfig
getRouteConfigWithoutCandidateValidation(
    const SelectedEmitCArtifactRouteConfig &config) {
  SelectedEmitCArtifactRouteConfig route = config;
  route.candidateValidationFn = nullptr;
  return route;
}

llvm::Expected<SelectedEmitCArtifactTarget>
selectConstructionTemplateArtifactTarget(
    mlir::ModuleOp module,
    const ConstructionTemplateArtifactAdapterConfig &config) {
  return selectSelectedEmitCArtifactTarget(
      module, getRouteConfigWithoutCandidateValidation(config.selectedRoute));
}

} // namespace

llvm::Error validateConstructionTemplateArtifactAdapterConfig(
    const ConstructionTemplateArtifactAdapterConfig &config) {
  if (llvm::Error error =
          requireNonEmpty("selected object route id",
                          config.selectedRoute.routeID))
    return error;
  if (llvm::Error error = requireNonEmpty("selected object artifact kind",
                                          config.selectedRoute.artifactKind))
    return error;
  if (llvm::Error error =
          requireNonEmpty("origin plugin", config.selectedRoute.originPlugin))
    return error;
  if (config.selectedRoute.artifactKind !=
      kRiscvELFRelocatableObjectArtifactKind)
    return makeConstructionTemplateAdapterError(
        llvm::Twine("selected object route artifact kind must be '") +
        kRiscvELFRelocatableObjectArtifactKind + "'");
  if (!config.selectedRoute.candidateValidationFn)
    return makeConstructionTemplateAdapterError(
        "selected object route requires a route-local candidate validator");
  if (!config.selectedRoute.routeBuilderFn)
    return makeConstructionTemplateAdapterError(
        "selected object route requires a plugin-owned EmitC route builder");
  if (!config.objectPackagerFn)
    return makeConstructionTemplateAdapterError(
        "object export requires a route-local object packager callback");

  if (llvm::Error error = requireNonEmpty("header route id",
                                          config.headerRouteID))
    return error;
  if (llvm::Error error = requireNonEmpty("header artifact kind",
                                          config.headerArtifactKind))
    return error;
  if (config.headerArtifactKind != kRuntimeCallableCHeaderArtifactKind)
    return makeConstructionTemplateAdapterError(
        llvm::Twine("header route artifact kind must be '") +
        kRuntimeCallableCHeaderArtifactKind + "'");
  if (llvm::Error error = requireNonEmpty("owner plugin", config.ownerPlugin))
    return error;
  if (llvm::Error error = requireNonEmpty("header guard", config.headerGuard))
    return error;
  if (llvm::Error error =
          requireNonEmpty("evidence prefix", config.evidencePrefix))
    return error;
  if (llvm::Error error =
          requireNonEmpty("emission kind", config.emissionKind))
    return error;
  if (llvm::Error error =
          requireNonEmpty("lowering boundary", config.loweringBoundary))
    return error;
  if (llvm::Error error =
          requireNonEmpty("runtime ABI kind", config.runtimeABIKind))
    return error;
  if (!config.allowDynamicRuntimeABIIdentity) {
    if (llvm::Error error =
            requireNonEmpty("runtime ABI", config.runtimeABI))
      return error;
    if (llvm::Error error =
            requireNonEmpty("runtime ABI name", config.runtimeABIName))
      return error;
  }
  if (llvm::Error error =
          requireNonEmpty("runtime glue role", config.runtimeGlueRole))
    return error;
  if (llvm::Error error =
          requireNonEmpty("component group", config.componentGroup))
    return error;
  if (llvm::Error error =
          requireNonEmpty("object handoff kind", config.handoffKind))
    return error;
  if (llvm::Error error =
          requireNonEmpty("selected object description",
                          config.selectedObjectDescription))
    return error;
  if (config.metadataEvidence.empty())
    return makeConstructionTemplateAdapterError(
        "requires construction metadata evidence entries");
  if (llvm::Error error = validateSelectedLoweringBoundaryConfig(config))
    return error;

  return llvm::Error::success();
}

MaterializedEmitCHeaderArtifactConfig
getConstructionTemplateHeaderArtifactConfig(
    const ConstructionTemplateArtifactAdapterConfig &config) {
  MaterializedEmitCHeaderArtifactConfig header;
  header.selectedRoute = config.selectedRoute;
  header.headerGuard = config.headerGuard;
  header.evidencePrefix = config.evidencePrefix;
  header.includes = config.includes;
  header.selectedVariant = config.selectedVariant;
  header.emissionKind = config.emissionKind;
  header.loweringBoundary = config.loweringBoundary;
  header.runtimeABI = config.runtimeABI;
  header.runtimeABIKind = config.runtimeABIKind;
  header.runtimeABIName = config.runtimeABIName;
  header.runtimeGlueRole = config.runtimeGlueRole;
  header.allowDynamicRuntimeABIIdentity =
      config.allowDynamicRuntimeABIIdentity;
  header.runtimeABIParameters = config.runtimeABIParameters;
  header.metadataEvidence = config.metadataEvidence;
  return header;
}

MaterializedEmitCObjectBundleArtifactConfig
getConstructionTemplateObjectBundleArtifactConfig(
    const ConstructionTemplateArtifactAdapterConfig &config,
    TargetArtifactExportFn objectExportFn,
    TargetArtifactExportFn headerExportFn) {
  MaterializedEmitCObjectBundleArtifactConfig bundle;
  bundle.header = getConstructionTemplateHeaderArtifactConfig(config);
  bundle.headerRouteID = config.headerRouteID;
  bundle.headerArtifactKind = config.headerArtifactKind;
  bundle.ownerPlugin = config.ownerPlugin;
  bundle.objectExportFn = objectExportFn;
  bundle.headerExportFn = headerExportFn;
  bundle.componentGroup = config.componentGroup;
  bundle.externalABIName = config.externalABIName;
  bundle.handoffKind = config.handoffKind;
  bundle.selectedObjectDescription = config.selectedObjectDescription;
  return bundle;
}

llvm::Error validateConstructionTemplateTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate,
    const ConstructionTemplateArtifactAdapterConfig &config) {
  if (llvm::Error error =
          validateConstructionTemplateArtifactAdapterConfig(config))
    return error;
  return validateMaterializedEmitCHeaderArtifactCandidate(
      candidate, getConstructionTemplateHeaderArtifactConfig(config));
}

llvm::Error exportConstructionTemplateHeaderArtifact(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    const ConstructionTemplateArtifactAdapterConfig &config) {
  if (llvm::Error error =
          validateConstructionTemplateArtifactAdapterConfig(config))
    return error;
  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectConstructionTemplateArtifactTarget(module, config);
  if (!target)
    return target.takeError();
  if (llvm::Error error = validateSelectedLoweringBoundary(*target, config))
    return error;
  if (llvm::Error error = validateConstructionTemplateTargetArtifactCandidate(
          target->candidate, config))
    return error;
  return exportMaterializedEmitCHeaderArtifact(
      module, os, getConstructionTemplateHeaderArtifactConfig(config));
}

llvm::Error exportConstructionTemplateObjectArtifact(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    const ConstructionTemplateArtifactAdapterConfig &config) {
  if (llvm::Error error =
          validateConstructionTemplateArtifactAdapterConfig(config))
    return error;

  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectConstructionTemplateArtifactTarget(module, config);
  if (!target)
    return target.takeError();
  if (llvm::Error error = validateSelectedLoweringBoundary(*target, config))
    return error;
  if (llvm::Error error = validateConstructionTemplateTargetArtifactCandidate(
          target->candidate, config))
    return error;

  llvm::Expected<std::string> source =
      emitSelectedEmitCArtifactCppSource(module, config.selectedRoute);
  if (!source)
    return source.takeError();
  return config.objectPackagerFn(*source, os);
}

llvm::Error exportConstructionTemplateEmitCToCpp(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    const ConstructionTemplateArtifactAdapterConfig &config) {
  if (llvm::Error error =
          validateConstructionTemplateArtifactAdapterConfig(config))
    return error;

  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectConstructionTemplateArtifactTarget(module, config);
  if (!target)
    return target.takeError();
  if (llvm::Error error = validateSelectedLoweringBoundary(*target, config))
    return error;
  if (llvm::Error error = validateConstructionTemplateTargetArtifactCandidate(
          target->candidate, config))
    return error;

  llvm::Expected<std::string> source =
      emitSelectedEmitCArtifactCppSource(module, config.selectedRoute);
  if (!source)
    return source.takeError();
  os << *source;
  return llvm::Error::success();
}

llvm::Error registerConstructionTemplateArtifactAdapterExporters(
    TargetArtifactExporterRegistry &registry,
    const ConstructionTemplateArtifactAdapterConfig &config,
    TargetArtifactExportFn objectExportFn,
    TargetArtifactExportFn headerExportFn) {
  if (llvm::Error error =
          validateConstructionTemplateArtifactAdapterConfig(config))
    return error;
  if (!objectExportFn || !headerExportFn)
    return makeConstructionTemplateAdapterError(
        "registration requires object and header export callbacks");

  return registerMaterializedEmitCObjectBundleArtifactExporters(
      registry, getConstructionTemplateObjectBundleArtifactConfig(
                    config, objectExportFn, headerExportFn));
}

} // namespace tianchenrv::target
