#include "TianChenRV/Target/I32BinaryFamilyRegistry.h"
#include "TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVI32BinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"
#include "TianChenRV/Target/RVVScalarBinaryFamily.h"
#include "TianChenRV/Target/RVVScalarDispatch.h"
#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <initializer_list>
#include <optional>
#include <string>

using namespace tianchenrv::target;
using namespace tianchenrv::target::i32_binary;

namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");
constexpr llvm::StringLiteral kDispatchTargetOwner(
    "rvv-scalar-dispatch-target");
constexpr llvm::StringLiteral kRuntimeCallableCSourceArtifactKind(
    "runtime-callable-c-source");
constexpr llvm::StringLiteral kRuntimeCallableCHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kRiscvELFRelocatableObjectArtifactKind(
    "riscv-elf-relocatable-object");

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

int expect(bool condition, llvm::Twine message) {
  if (condition)
    return 0;
  return fail(message);
}

int expectSuccess(llvm::Error error, llvm::Twine context) {
  if (!error)
    return 0;
  std::string message = llvm::toString(std::move(error));
  return fail(context + ": " + message);
}

int expectRoute(const TargetArtifactExporterRegistry &registry,
                llvm::StringRef routeID, llvm::StringRef artifactKind,
                llvm::StringRef originPlugin, llvm::StringRef emissionKind,
                llvm::StringRef componentGroup = {},
                llvm::StringRef externalABIName = {},
                std::optional<bool> expectedDirectHelperRoute =
                    std::nullopt) {
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter)
    return fail(llvm::Twine("missing exporter route '") + routeID + "'");
  if (exporter->getArtifactKind() != artifactKind)
    return fail(llvm::Twine("artifact kind mismatch for route '") + routeID +
                "'");
  if (exporter->getOriginPlugin() != originPlugin)
    return fail(llvm::Twine("origin mismatch for route '") + routeID + "'");
  if (exporter->getEmissionKind() != emissionKind)
    return fail(llvm::Twine("emission kind mismatch for route '") + routeID +
                "'");
  if (!exporter->getExportFn())
    return fail(llvm::Twine("missing export callback for route '") + routeID +
                "'");
  if (exporter->getComponentGroup() != componentGroup)
    return fail(llvm::Twine("component group mismatch for route '") + routeID +
                "'");
  if (exporter->getExternalABIName() != externalABIName)
    return fail(llvm::Twine("external ABI name mismatch for route '") +
                routeID + "'");
  if (expectedDirectHelperRoute &&
      exporter->hasDirectHelperRoute() != *expectedDirectHelperRoute)
    return fail(llvm::Twine("direct helper route flag mismatch for route '") +
                routeID + "'");
  return 0;
}

int expectCompositeRoute(const TargetArtifactExporterRegistry &registry,
                         llvm::StringRef routeID, llvm::StringRef artifactKind,
                         llvm::StringRef owner, llvm::StringRef runtimeABIKind,
                         llvm::StringRef runtimeABIName,
                         llvm::StringRef componentGroup,
                         llvm::StringRef externalABIName,
                         std::optional<bool> expectedDirectHelperRoute =
                             std::nullopt) {
  const TargetArtifactCompositeExporter *exporter =
      registry.lookupComposite(routeID);
  if (!exporter)
    return fail(llvm::Twine("missing composite route '") + routeID + "'");
  if (exporter->getArtifactKind() != artifactKind)
    return fail(llvm::Twine("artifact kind mismatch for composite route '") +
                routeID + "'");
  if (exporter->getOwner() != owner)
    return fail(llvm::Twine("owner mismatch for composite route '") + routeID +
                "'");
  if (exporter->getRuntimeABIKind() != runtimeABIKind)
    return fail(llvm::Twine("runtime ABI kind mismatch for composite route '") +
                routeID + "'");
  if (exporter->getRuntimeABIName() != runtimeABIName)
    return fail(llvm::Twine("runtime ABI name mismatch for composite route '") +
                routeID + "'");
  if (exporter->getComponentGroup() != componentGroup)
    return fail(llvm::Twine("component group mismatch for composite route '") +
                routeID + "'");
  if (exporter->getExternalABIName() != externalABIName)
    return fail(llvm::Twine("external ABI name mismatch for composite route '") +
                routeID + "'");
  if (expectedDirectHelperRoute &&
      exporter->hasDirectHelperRoute() != *expectedDirectHelperRoute)
    return fail(llvm::Twine("direct helper route flag mismatch for composite "
                            "route '") +
                routeID + "'");
  if (!exporter->getExportFn() || !exporter->getMatchFn())
    return fail(llvm::Twine("missing composite callbacks for route '") +
                routeID + "'");
  if (runtimeABIKind.empty() && exporter->getRuntimeABIParametersFn())
    return fail(llvm::Twine("unexpected runtime ABI callback for route '") +
                routeID + "'");
  if (!runtimeABIKind.empty() && !exporter->getRuntimeABIParametersFn() &&
      exporter->getRuntimeABIParameters().empty())
    return fail(llvm::Twine("missing runtime ABI callback for route '") +
                routeID + "'");
  return 0;
}

int expectFamilyDescriptorShape(const I32BinaryFamilyDescriptor &family) {
  if (int result = expect(lookupI32BinaryFamilyRegistrationByID(family.familyID) == &family,
                          "lookup by family id returns registration record"))
    return result;
  if (int result =
          expect(lookupI32BinaryFamilyRegistrationByFrontendLowering(
                     family.frontendLowering) == &family,
                 "lookup by frontend lowering returns registration record"))
    return result;
  if (int result =
          expect(lookupI32BinaryFamilyRegistrationByLegacyLoweringDescriptor(
                     family.loweringDescriptor) == &family,
                 "legacy descriptor mirror lookup returns registration record"))
    return result;

  if (int result =
          expect(family.rvv.kind == family.kind &&
                     family.scalar.kind == family.kind &&
                     family.dispatch.kind == family.kind,
                 "family sub-registrations preserve common kind"))
    return result;
  if (int result = expect(family.rvv.routeID ==
                              family.dispatch.rvvRouteID,
                          "dispatch RVV route mirrors RVV route registration"))
    return result;
  if (int result = expect(family.rvv.emissionKind ==
                              family.dispatch.rvvEmissionKind,
                          "dispatch RVV emission kind mirrors RVV route registration"))
    return result;
  if (int result =
          expect(family.rvv.runtimeABI == family.dispatch.rvvRuntimeABI,
                 "dispatch RVV ABI mirrors RVV route registration"))
    return result;
  if (int result =
          expect(family.scalar.routeID == family.dispatch.scalarRouteID,
                 "dispatch scalar route mirrors scalar route registration"))
    return result;
  if (int result =
          expect(family.scalar.emissionKind ==
                     family.dispatch.scalarEmissionKind,
                 "dispatch scalar emission kind mirrors scalar route registration"))
    return result;
  if (int result =
          expect(family.scalar.runtimeABI == family.dispatch.scalarRuntimeABI,
                 "dispatch scalar ABI mirrors scalar route registration"))
    return result;
  return expect(!family.dispatch.selfCheckSuccessMarker.empty(),
                "dispatch self-check marker is present");
}

int expectStandaloneFamilyExporterRoutes(
    const TargetArtifactExporterRegistry &registry,
    const I32BinaryFamilyDescriptor &family) {
  if (int result = expectRoute(
          registry, family.rvv.routeID, kRuntimeCallableCSourceArtifactKind,
          kRVVPluginName, family.rvv.emissionKind,
          family.rvv.externalABIComponentGroup, family.rvv.runtimeABIName,
          /*expectedDirectHelperRoute=*/true))
    return result;

  if (int result = expectRoute(
          registry, family.scalar.routeID, kRuntimeCallableCSourceArtifactKind,
          kScalarPluginName, family.scalar.emissionKind,
          /*componentGroup=*/{}, /*externalABIName=*/{},
          /*expectedDirectHelperRoute=*/false))
    return result;

  if (int result = expectCompositeRoute(
          registry, family.rvv.headerRouteID,
          kRuntimeCallableCHeaderArtifactKind, kRVVPluginName,
          family.rvv.runtimeABIKind, family.rvv.runtimeABIName,
          family.rvv.externalABIComponentGroup, family.rvv.runtimeABIName,
          /*expectedDirectHelperRoute=*/true))
    return result;

  if (int result = expectCompositeRoute(
          registry, family.rvv.objectRouteID,
          kRiscvELFRelocatableObjectArtifactKind, kRVVPluginName,
          family.rvv.runtimeABIKind, family.rvv.runtimeABIName,
          family.rvv.externalABIComponentGroup, family.rvv.runtimeABIName,
          /*expectedDirectHelperRoute=*/true))
    return result;

  if (int result = expectCompositeRoute(
          registry, family.scalar.headerRouteID,
          kRuntimeCallableCHeaderArtifactKind, kScalarPluginName,
          /*runtimeABIKind=*/{}, /*runtimeABIName=*/{},
          /*componentGroup=*/{}, /*externalABIName=*/{},
          /*expectedDirectHelperRoute=*/false))
    return result;
  if (int result = expectCompositeRoute(
          registry, family.scalar.objectRouteID,
          kRiscvELFRelocatableObjectArtifactKind, kScalarPluginName,
          /*runtimeABIKind=*/{}, /*runtimeABIName=*/{},
          /*componentGroup=*/{}, /*externalABIName=*/{},
          /*expectedDirectHelperRoute=*/false))
    return result;

  return 0;
}

int expectDispatchFamilyExporterRoutes(
    const TargetArtifactExporterRegistry &registry,
    const rvv_scalar::RVVScalarBinaryFamilyDescriptor &family) {
  if (int result = expectCompositeRoute(
          registry, family.dispatch.dispatchSourceRouteID,
          kRuntimeCallableCSourceArtifactKind, kDispatchTargetOwner,
          family.dispatch.dispatchRuntimeABIKind,
          family.dispatch.dispatchRuntimeABIName,
          family.dispatch.dispatchExternalABIComponentGroup,
          family.dispatch.dispatchRuntimeABIName))
    return result;
  const TargetArtifactCompositeExporter *source =
      registry.lookupComposite(family.dispatch.dispatchSourceRouteID);
  if (int result =
          expect(source && source->getRuntimeABIParametersFn() &&
                     source->getBundleMetadataFn() &&
                     source->getCandidateValidationFn(),
                 llvm::Twine("dispatch source route derives runtime ABI and "
                             "bundle metadata from selected components for ") +
                     family.familyID))
    return result;

  if (int result = expectCompositeRoute(
          registry, family.dispatch.dispatchHeaderRouteID,
          kRuntimeCallableCHeaderArtifactKind, kDispatchTargetOwner,
          family.dispatch.dispatchRuntimeABIKind,
          family.dispatch.dispatchRuntimeABIName,
          family.dispatch.dispatchExternalABIComponentGroup,
          family.dispatch.dispatchRuntimeABIName))
    return result;

  return expectCompositeRoute(
      registry, family.dispatch.dispatchObjectRouteID,
      kRiscvELFRelocatableObjectArtifactKind, kDispatchTargetOwner,
      family.dispatch.dispatchRuntimeABIKind,
      family.dispatch.dispatchRuntimeABIName,
      family.dispatch.dispatchExternalABIComponentGroup,
      family.dispatch.dispatchRuntimeABIName);
}

const rvv_scalar::RVVScalarDispatchRouteManifestEntry *
findDispatchManifestRoute(const rvv_scalar::RVVScalarBinaryFamilyDescriptor &family,
                          rvv_scalar::RVVScalarDispatchRouteKind routeKind) {
  for (const rvv_scalar::RVVScalarDispatchRouteManifestEntry &route :
       rvv_scalar::getRVVScalarDispatchRouteManifest()) {
    if (route.family == &family.dispatch && route.routeKind == routeKind)
      return &route;
  }
  return nullptr;
}

int expectDispatchManifestRoute(
    const rvv_scalar::RVVScalarBinaryFamilyDescriptor &family,
    rvv_scalar::RVVScalarDispatchRouteKind routeKind,
    llvm::StringRef expectedRouteID, llvm::StringRef expectedArtifactKind,
    bool expectedBinaryStdout) {
  const rvv_scalar::RVVScalarDispatchRouteManifestEntry *route =
      findDispatchManifestRoute(family, routeKind);
  if (!route)
    return fail(llvm::Twine("missing dispatch manifest route for family '") +
                family.familyID + "'");
  if (route->routeID != expectedRouteID)
    return fail(llvm::Twine("manifest route id mismatch for family '") +
                family.familyID + "'");
  if (route->artifactKind != expectedArtifactKind)
    return fail(llvm::Twine("manifest artifact kind mismatch for route '") +
                expectedRouteID + "'");
  if (route->runtimeABIKind != family.dispatch.dispatchRuntimeABIKind ||
      route->runtimeABIName != family.dispatch.dispatchRuntimeABIName ||
      route->componentGroup !=
          family.dispatch.dispatchExternalABIComponentGroup ||
      route->externalABIName != family.dispatch.dispatchRuntimeABIName ||
      route->selfCheckSuccessMarker !=
          family.dispatch.selfCheckSuccessMarker)
    return fail(llvm::Twine("manifest ABI/component metadata mismatch for "
                            "route '") +
                expectedRouteID + "'");
  if (route->requiresBinaryStdout != expectedBinaryStdout)
    return fail(llvm::Twine("manifest binary stdout flag mismatch for route '") +
                expectedRouteID + "'");
  if (route->description.empty())
    return fail(llvm::Twine("manifest description missing for route '") +
                expectedRouteID + "'");
  return 0;
}

int expectDispatchRouteManifest() {
  using RouteKind = rvv_scalar::RVVScalarDispatchRouteKind;
  llvm::ArrayRef<const rvv_scalar::RVVScalarBinaryFamilyDescriptor *> families =
      rvv_scalar::getRVVScalarBinaryRegistrationRecords();
  llvm::ArrayRef<rvv_scalar::RVVScalarDispatchRouteManifestEntry> routes =
      rvv_scalar::getRVVScalarDispatchRouteManifest();
  llvm::ArrayRef<RouteKind> routeKinds =
      rvv_scalar::getRVVScalarDispatchRouteKinds();
  const std::size_t expectedRouteCount =
      rvv_scalar::getRVVScalarDispatchRouteCount();
  if (int result =
          expect(routes.size() == expectedRouteCount,
                 "dispatch route manifest count is exposed by manifest API"))
    return result;
  if (int result =
          expect(routes.size() == families.size() * routeKinds.size(),
                 "dispatch route manifest count is family-count x route-kind "
                 "count"))
    return result;

  bool exposesSourceKind = false;
  bool exposesHeaderKind = false;
  bool exposesObjectKind = false;
  bool exposesSelfCheckSourceKind = false;
  bool exposesSelfCheckObjectKind = false;
  for (RouteKind routeKind : routeKinds) {
    exposesSourceKind |= routeKind == RouteKind::Source;
    exposesHeaderKind |= routeKind == RouteKind::Header;
    exposesObjectKind |= routeKind == RouteKind::Object;
    exposesSelfCheckSourceKind |= routeKind == RouteKind::SelfCheckSource;
    exposesSelfCheckObjectKind |= routeKind == RouteKind::SelfCheckObject;
  }
  if (int result =
          expect(exposesSourceKind && exposesHeaderKind && exposesObjectKind &&
                     exposesSelfCheckSourceKind && exposesSelfCheckObjectKind,
                 "dispatch route-kind manifest exposes all bounded route "
                 "kinds"))
    return result;

  for (const rvv_scalar::RVVScalarBinaryFamilyDescriptor *family : families) {
    if (int result = expectDispatchManifestRoute(
            *family, RouteKind::Source,
            family->dispatch.dispatchSourceRouteID,
            kRuntimeCallableCSourceArtifactKind,
            /*expectedBinaryStdout=*/false))
      return result;
    if (int result = expectDispatchManifestRoute(
            *family, RouteKind::Header,
            family->dispatch.dispatchHeaderRouteID,
            kRuntimeCallableCHeaderArtifactKind,
            /*expectedBinaryStdout=*/false))
      return result;
    if (int result = expectDispatchManifestRoute(
            *family, RouteKind::Object,
            family->dispatch.dispatchObjectRouteID,
            kRiscvELFRelocatableObjectArtifactKind,
            /*expectedBinaryStdout=*/true))
      return result;
    std::string selfCheckSourceRoute =
        (llvm::Twine("tcrv-export-rvv-scalar-") + family->familyID +
         "-dispatch-self-check-c")
            .str();
    std::string selfCheckObjectRoute =
        (llvm::Twine("tcrv-export-rvv-scalar-") + family->familyID +
         "-dispatch-self-check-object")
            .str();
    if (int result = expectDispatchManifestRoute(
            *family, RouteKind::SelfCheckSource, selfCheckSourceRoute,
            "self-check-c-source", /*expectedBinaryStdout=*/false))
      return result;
    if (int result = expectDispatchManifestRoute(
            *family, RouteKind::SelfCheckObject, selfCheckObjectRoute,
            "self-check-riscv-elf-relocatable-object",
            /*expectedBinaryStdout=*/true))
      return result;
  }

  bool hasI32VAddSourceRoute = false;
  bool hasI64VMulSelfCheckObjectRoute = false;
  for (const rvv_scalar::RVVScalarDispatchRouteManifestEntry &lhs : routes) {
    hasI32VAddSourceRoute |=
        lhs.routeKind == RouteKind::Source &&
        lhs.routeID == "tcrv-export-rvv-scalar-i32-vadd-dispatch-c";
    hasI64VMulSelfCheckObjectRoute |=
        lhs.routeKind == RouteKind::SelfCheckObject &&
        lhs.routeID ==
            "tcrv-export-rvv-scalar-i64-vmul-dispatch-self-check-object";
    for (const rvv_scalar::RVVScalarDispatchRouteManifestEntry &rhs : routes) {
      if (&lhs == &rhs)
        continue;
      if (int result = expect(lhs.routeID != rhs.routeID,
                              "dispatch manifest route ids are distinct"))
        return result;
      if (lhs.routeKind == rhs.routeKind && lhs.family != rhs.family) {
        if (int result =
                expect(lhs.componentGroup != rhs.componentGroup,
                       "same-kind dispatch manifest component groups are "
                       "distinct across families"))
          return result;
        if (int result =
                expect(lhs.externalABIName != rhs.externalABIName,
                       "same-kind dispatch manifest ABI names are distinct "
                       "across families"))
          return result;
      }
    }
  }

  return expect(hasI32VAddSourceRoute && hasI64VMulSelfCheckObjectRoute,
                "dispatch manifest exposes representative i32-vadd source "
                "and i64-vmul self-check object routes");
}

int expectStaleFamilyMismatchGuards() {
  const I32BinaryFamilyDescriptor &add = getI32VAddFamilyRegistrationRecord();
  const I32BinaryFamilyDescriptor &sub = getI32VSubFamilyRegistrationRecord();
  const I32BinaryFamilyDescriptor &mul = getI32VMulFamilyRegistrationRecord();
  for (const I32BinaryFamilyDescriptor *lhs : {&add, &sub, &mul}) {
    for (const I32BinaryFamilyDescriptor *rhs : {&add, &sub, &mul}) {
      if (lhs == rhs)
        continue;
      if (int result = expect(lhs->familyID != rhs->familyID,
                              "family ids are pairwise distinct"))
        return result;
      if (int result = expect(lhs->rvv.routeID != rhs->rvv.routeID,
                              "RVV source routes are pairwise distinct"))
        return result;
      if (int result =
              expect(lhs->rvv.arithmeticIntrinsicPrefix !=
                         rhs->rvv.arithmeticIntrinsicPrefix,
                     "RVV arithmetic intrinsic prefixes are pairwise distinct"))
        return result;
      if (int result =
              expect(lhs->rvv.arithmeticIntrinsicPrefix.starts_with(
                         "__riscv_v"),
                     "RVV arithmetic intrinsic prefix starts with RVV "
                     "intrinsic namespace"))
        return result;
      if (int result =
              expect(lhs->rvv.arithmeticIntrinsicPrefix.ends_with("_"),
                     "RVV arithmetic intrinsic prefix is suffix-free"))
        return result;
      if (int result =
              expect(!lhs->rvv.arithmeticIntrinsicPrefix.ends_with("i32m1"),
                     "RVV arithmetic intrinsic prefix does not bake in m1"))
        return result;
      if (int result =
              expect(!lhs->rvv.arithmeticIntrinsicPrefix.ends_with("i32m2"),
                     "RVV arithmetic intrinsic prefix does not bake in m2"))
        return result;
      if (int result = expect(lhs->scalar.routeID != rhs->scalar.routeID,
                              "scalar source routes are pairwise distinct"))
        return result;
      if (int result = expect(lhs->dispatch.selfCheckSuccessMarker !=
                                  rhs->dispatch.selfCheckSuccessMarker,
                              "dispatch self-check markers are pairwise "
                              "distinct"))
        return result;
      if (int result = expect(lhs->dispatch.dispatchSourceRouteID !=
                                  rhs->dispatch.dispatchSourceRouteID,
                              "dispatch source routes are pairwise distinct"))
        return result;
      if (int result = expect(lhs->dispatch.dispatchRuntimeABIName !=
                                  rhs->dispatch.dispatchRuntimeABIName,
                              "dispatch ABI names are pairwise distinct"))
        return result;
    }
  }
  return 0;
}

int expectFiniteRVVVectorShapeDescriptorShape() {
  using tianchenrv::target::rvv::RVVI32VectorShapeConfig;
  using tianchenrv::target::rvv::
      RVVI32VectorShapeSelectedPlanMetadataDescriptor;

  llvm::ArrayRef<const RVVI32VectorShapeConfig *> configs =
      tianchenrv::target::rvv::getFiniteI32VectorShapeConfigs();
  if (int result =
          expect(configs.size() == 2,
                 "finite RVV i32 vector shape descriptor has two shapes"))
    return result;
  if (int result = expect(configs[0] ==
                              &tianchenrv::target::rvv::
                                  getI32M1VectorShapeConfig(),
                          "finite RVV descriptor preserves i32m1 order"))
    return result;
  if (int result = expect(configs[1] ==
                              &tianchenrv::target::rvv::
                                  getI32M2VectorShapeConfig(),
                          "finite RVV descriptor preserves i32m2 order"))
    return result;

  for (const RVVI32VectorShapeConfig *config : configs) {
    if (int result = expect(
            tianchenrv::target::rvv::lookupFiniteI32VectorShapeConfigByShapeID(
                config->shapeID) == config,
            "finite RVV descriptor lookup by shape id succeeds"))
      return result;
    if (int result =
            expect(config->sewBits == 32, "finite RVV descriptor SEW is 32"))
      return result;
    if (int result = expect(!config->lmul.empty() &&
                                !config->tailPolicy.empty() &&
                                !config->maskPolicy.empty() &&
                                !config->vectorType.empty() &&
                                !config->vectorSuffix.empty() &&
                                !config->setvlSuffix.empty(),
                            "finite RVV descriptor carries shape spellings"))
      return result;
    if (int result = expect(config->sewCapabilityID.starts_with("rvv.i32_") &&
                                config->lmulCapabilityID.starts_with(
                                    "rvv.i32_") &&
                                config->tailPolicyCapabilityID.starts_with(
                                    "rvv.i32_") &&
                                config->maskPolicyCapabilityID.starts_with(
                                    "rvv.i32_"),
                            "finite RVV descriptor carries capability ids"))
      return result;

    llvm::SmallVector<RVVI32VectorShapeSelectedPlanMetadataDescriptor, 12>
        metadata;
    tianchenrv::target::rvv::appendRVVI32VectorShapeSelectedPlanMetadata(
        *config, metadata);
    if (int result = expect(metadata.size() == 12,
                            "finite RVV shape config records selected-plan "
                            "metadata field count"))
      return result;
    if (int result = expect(
            metadata.front().name ==
                    tianchenrv::target::rvv::
                        getRVVSelectedVectorShapeAttrName() &&
                metadata.front().value == config->shapeID &&
                metadata.front().role ==
                    tianchenrv::target::rvv::
                        getSelectedRVVI32VectorShapeMetadataRole(),
            "finite RVV shape config records selected shape metadata"))
      return result;
    if (int result = expect(
            metadata[5].name ==
                    tianchenrv::target::rvv::
                        getRVVSelectedVectorTypeAttrName() &&
                metadata[5].value == config->vectorType,
            "finite RVV shape config records vector type metadata"))
      return result;
    if (int result = expect(
            metadata[6].name ==
                    tianchenrv::target::rvv::
                        getRVVSelectedVectorSuffixAttrName() &&
                metadata[6].value == config->vectorSuffix,
            "finite RVV shape config records vector suffix metadata"))
      return result;
    if (int result = expect(
            metadata[7].name ==
                    tianchenrv::target::rvv::
                        getRVVSelectedSetVLSuffixAttrName() &&
                metadata[7].value == config->setvlSuffix,
            "finite RVV shape config records setvl suffix metadata"))
      return result;
    if (int result = expect(
            metadata[8].name ==
                    tianchenrv::target::rvv::
                        getRVVSelectedVectorSEWCapabilityAttrName() &&
                metadata[8].value == config->sewCapabilityID &&
                metadata[8].role ==
                    tianchenrv::target::rvv::
                        getSelectedRVVVectorShapeCapabilityMetadataRole(),
            "finite RVV shape config records selected SEW capability metadata"))
      return result;
    if (int result = expect(
            metadata[9].name ==
                    tianchenrv::target::rvv::
                        getRVVSelectedVectorLMULCapabilityAttrName() &&
                metadata[9].value == config->lmulCapabilityID,
            "finite RVV shape config records selected LMUL capability metadata"))
      return result;
    if (int result = expect(
            metadata[10].name ==
                    tianchenrv::target::rvv::
                        getRVVSelectedTailPolicyCapabilityAttrName() &&
                metadata[10].value == config->tailPolicyCapabilityID,
            "finite RVV shape config records selected tail-policy capability "
            "metadata"))
      return result;
    if (int result = expect(
            metadata[11].name ==
                    tianchenrv::target::rvv::
                        getRVVSelectedMaskPolicyCapabilityAttrName() &&
                metadata[11].value == config->maskPolicyCapabilityID,
            "finite RVV shape config records selected mask-policy capability "
            "metadata"))
      return result;
  }

  return expect(
      !tianchenrv::target::rvv::lookupFiniteI32VectorShapeConfigByShapeID(
          "i32m4"),
      "finite RVV descriptor rejects unsupported shape ids");
}

int expectRVVI32BinaryIntrinsicDescriptorShape() {
  using tianchenrv::target::rvv::getRVVI32BinaryIntrinsicDescriptor;

  for (const I32BinaryFamilyDescriptor *family :
       getI32BinaryFamilyRegistrationRecords()) {
    for (const tianchenrv::target::rvv::RVVI32VectorShapeConfig *shape :
         tianchenrv::target::rvv::getFiniteI32VectorShapeConfigs()) {
      tianchenrv::target::rvv::RVVI32BinaryIntrinsicDescriptor descriptor =
          getRVVI32BinaryIntrinsicDescriptor(*family, *shape);

      if (int result = expect(descriptor.family == family &&
                                  descriptor.shape == shape,
                              "RVV i32 binary descriptor composes existing "
                              "family and shape descriptors"))
        return result;
      if (int result = expect(descriptor.getArithmeticFamilyID() ==
                                  family->familyID,
                              "RVV i32 selected intrinsic helper reports family id"))
        return result;
      if (int result = expect(descriptor.getLoweringDescriptor() ==
                                  family->loweringDescriptor,
                              "RVV i32 selected intrinsic helper reports lowering "
                              "descriptor"))
        return result;
      if (int result = expect(descriptor.getRVVOperationName() ==
                                  family->rvv.arithmeticOpName,
                              "RVV i32 selected intrinsic helper reports typed RVV "
                              "operation label"))
        return result;
      if (int result = expect(descriptor.getVectorType() == shape->vectorType &&
                                  descriptor.getVectorSuffix() ==
                                      shape->vectorSuffix &&
                                  descriptor.getSetVLSuffix() ==
                                      shape->setvlSuffix,
                              "RVV i32 selected intrinsic helper reports selected "
                              "vector shape spellings"))
        return result;

      std::string expectedIntrinsic =
          (llvm::Twine(family->rvv.arithmeticIntrinsicPrefix) +
           shape->vectorSuffix)
              .str();
      if (int result = expect(descriptor.getArithmeticIntrinsicName() ==
                                  expectedIntrinsic,
                              "RVV i32 selected intrinsic helper derives full "
                              "arithmetic intrinsic from family and shape"))
        return result;
      if (int result = expect(descriptor.getSetVLIntrinsicName() ==
                                  (llvm::Twine("__riscv_vsetvl_") +
                                   shape->setvlSuffix)
                                      .str(),
                              "RVV i32 selected intrinsic helper derives vsetvl "
                              "intrinsic from shape"))
        return result;
      if (int result = expect(descriptor.getLoadIntrinsicName() ==
                                  (llvm::Twine("__riscv_vle32_v_") +
                                   shape->vectorSuffix)
                                      .str(),
                              "RVV i32 selected intrinsic helper derives load "
                              "intrinsic from shape"))
        return result;
      if (int result = expect(descriptor.getStoreIntrinsicName() ==
                                  (llvm::Twine("__riscv_vse32_v_") +
                                   shape->vectorSuffix)
                                      .str(),
                              "RVV i32 selected intrinsic helper derives store "
                              "intrinsic from shape"))
        return result;

      llvm::SmallVector<llvm::StringRef, 4> capabilityIDs =
          descriptor.getSelectedShapeCapabilityIDs();
      if (int result =
              expect(capabilityIDs.size() == 4 &&
                         capabilityIDs[0] == shape->sewCapabilityID &&
                         capabilityIDs[1] == shape->lmulCapabilityID &&
                         capabilityIDs[2] == shape->tailPolicyCapabilityID &&
                         capabilityIDs[3] == shape->maskPolicyCapabilityID,
                     "RVV i32 selected intrinsic helper reports selected shape "
                     "capability ids"))
        return result;

      if (int result = expect(descriptor.getRVVRouteID() ==
                                      family->rvv.routeID &&
                                  descriptor.getRVVRuntimeABIName() ==
                                      family->rvv.runtimeABIName &&
                                  descriptor.getDispatchSourceRouteID() ==
                                      family->dispatch.dispatchSourceRouteID &&
                                  descriptor
                                          .getDispatchExternalABIComponentGroup() ==
                                      family->dispatch
                                          .dispatchExternalABIComponentGroup &&
                                  descriptor.getDispatchRuntimeABIName() ==
                                      family->dispatch.dispatchRuntimeABIName,
                              "RVV i32 selected intrinsic helper reports route, "
                              "component group, and ABI names"))
        return result;

      std::string shapeComment =
          descriptor.formatSelectedVectorShapeConfigCommentBody();
      std::string intrinsicComment =
          descriptor.formatIntrinsicConfigCommentBody();
      if (int result =
              expect(llvm::StringRef(shapeComment).contains(shape->shapeID) &&
                         llvm::StringRef(shapeComment)
                             .contains(shape->vectorType) &&
                         llvm::StringRef(intrinsicComment)
                             .contains(shape->vectorSuffix),
                     "RVV i32 binary descriptor formats selected shape and "
                     "intrinsic metadata comments"))
        return result;

      if (family->kind == I32BinaryFamilyKind::Mul &&
          shape->shapeID == "i32m2") {
        if (int result = expect(descriptor.getArithmeticIntrinsicName() ==
                                    "__riscv_vmul_vv_i32m2",
                                "RVV selected intrinsic helper derives vmul i32m2 "
                                "intrinsic"))
          return result;
        if (int result = expect(descriptor.getDispatchSourceRouteID() ==
                                    "tcrv-export-rvv-scalar-i32-vmul-"
                                    "dispatch-c",
                                "RVV selected intrinsic helper derives vmul dispatch route"))
          return result;
      }
    }
  }

  return 0;
}

int expectRVVBinaryFamilyRegistryShape() {
  namespace rvv = tianchenrv::target::rvv;

  llvm::ArrayRef<const rvv::RVVBinaryFamilyDescriptor *> families =
      rvv::getRVVBinaryFamilyRegistrationRecords();
  if (int result = expect(families.size() == 6,
                          "RVV registry contains six binary families"))
    return result;
  if (int result =
          expect(families[0] == &rvv::getI32VAddFamilyRegistrationRecord() &&
                     families[1] == &rvv::getI32VSubFamilyRegistrationRecord() &&
                     families[2] == &rvv::getI32VMulFamilyRegistrationRecord() &&
                     families[3] == &rvv::getI64VAddFamilyRegistrationRecord() &&
                     families[4] == &rvv::getI64VSubFamilyRegistrationRecord() &&
                     families[5] == &rvv::getI64VMulFamilyRegistrationRecord(),
                 "RVV registry preserves finite family order"))
    return result;

  for (const rvv::RVVBinaryFamilyDescriptor *family : families) {
    if (int result =
            expect(rvv::lookupRVVBinaryFamilyRegistrationByID(family->familyID) == family,
                   "RVV registry lookup by family id returns registration record"))
      return result;
    if (int result = expect(
            rvv::lookupRVVBinaryFamilyRegistrationByFrontendLowering(
                family->frontendLowering) == family,
            "RVV registry lookup by frontend lowering returns registration record"))
      return result;
    if (int result = expect(
            rvv::lookupRVVBinaryFamilyRegistrationByLegacyLoweringDescriptor(
                family->loweringDescriptor) == family,
            "RVV registry legacy descriptor mirror lookup returns registration record"))
      return result;

    llvm::ArrayRef<const rvv::RVVVectorShapeConfig *> shapeConfigs =
        rvv::getRVVBinaryFamilyShapeConfigs(*family);
    if (int result =
            expect(!shapeConfigs.empty(),
                   "RVV family descriptor exposes selected-shape hooks"))
      return result;
    for (const rvv::RVVVectorShapeConfig *shape : shapeConfigs) {
      if (int result =
              expect(shape->dtypeID == family->dtypeID,
                     "RVV selected shape dtype matches family dtype"))
        return result;
      if (int result =
              expect(rvv::lookupRVVBinaryFamilyShapeConfigByID(
                         *family, shape->shapeID) == shape,
                     "RVV family shape lookup returns registration record"))
        return result;

      rvv::RVVBinaryIntrinsicDescriptor descriptor =
          rvv::getRVVBinaryIntrinsicDescriptor(*family, *shape);
      llvm::SmallVector<llvm::StringRef, 4> capabilityIDs =
          descriptor.getSelectedShapeCapabilityIDs();
      if (int result = expect(
              capabilityIDs.size() == 4 &&
                  capabilityIDs[0] == shape->sewCapabilityID &&
                  capabilityIDs[1] == shape->lmulCapabilityID &&
                  capabilityIDs[2] == shape->tailPolicyCapabilityID &&
                  capabilityIDs[3] == shape->maskPolicyCapabilityID,
              "RVV selected intrinsic helper reports selected shape capability hooks"))
        return result;
    }

    llvm::StringRef scalarType =
        family->dtype == rvv::RVVBinaryDTypeKind::I64 ? "int64_t" : "int32_t";
    if (int result = expect(family->constInputPointerCType ==
                                    (llvm::Twine("const ") + scalarType + " *")
                                        .str() &&
                                family->outputPointerCType ==
                                    (llvm::Twine(scalarType) + " *").str(),
                            "RVV registry owns runtime ABI pointer C types"))
      return result;
    if (int result = expect(!family->routeID.empty() &&
                                !family->headerRouteID.empty() &&
                                !family->objectRouteID.empty() &&
                                !family->runtimeABI.empty() &&
                                !family->runtimeABIName.empty() &&
                                !family->runtimeGlueRole.empty(),
                            "RVV registry owns route and runtime ABI names"))
      return result;
  }

  if (int result = expect(
          rvv::lookupRVVBinaryFamilyRegistrationByFrontendLowering("i32-vsub")->routeID ==
              "tcrv-export-rvv-i32-vsub-microkernel-c",
          "RVV registry owns i32-vsub route id"))
    return result;
  if (int result = expect(
          rvv::lookupRVVBinaryFamilyRegistrationByLegacyLoweringDescriptor(
              "i32-vmul-microkernel.v1")
                  ->runtimeABIName ==
              "rvv-i32-vmul-runtime-callable-c-function.v1",
          "RVV registry owns i32-vmul runtime ABI name"))
    return result;
  if (int result = expect(
          rvv::lookupRVVBinaryFamilyRegistrationByFrontendLowering("i64-vadd")
              ->constInputPointerCType == "const int64_t *",
          "RVV registry owns i64-vadd input pointer C type"))
    return result;
  if (int result = expect(
          rvv::lookupRVVBinaryFamilyRegistrationByFrontendLowering("i64-vsub")
                  ->runtimeABIName ==
              "rvv-i64-vsub-runtime-callable-c-function.v1",
          "RVV registry owns i64-vsub runtime ABI name"))
    return result;
  return expect(
      rvv::lookupRVVBinaryFamilyRegistrationByFrontendLowering("i64-vmul")->routeID ==
          "tcrv-export-rvv-i64-vmul-microkernel-c",
      "RVV registry owns i64-vmul route id");
}

int expectRVVBinaryI64DescriptorShape() {
  namespace rvv = tianchenrv::target::rvv;

  using tianchenrv::target::rvv::getI64M1VectorShapeConfig;
  using tianchenrv::target::rvv::getI64VAddFamilyRegistrationRecord;
  using tianchenrv::target::rvv::getI64VAddIntrinsicDescriptor;
  using tianchenrv::target::rvv::getI64VSubFamilyRegistrationRecord;
  using tianchenrv::target::rvv::getI64VSubIntrinsicDescriptor;
  using tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord;
  using tianchenrv::target::rvv::getI64VMulIntrinsicDescriptor;
  using tianchenrv::target::rvv::lookupFiniteI64VectorShapeConfigByShapeID;
  using tianchenrv::target::rvv::lookupRVVBinaryFamilyRegistrationByFrontendLowering;
  using tianchenrv::target::rvv::lookupRVVBinaryFamilyRegistrationByLegacyLoweringDescriptor;

  const auto &shape = getI64M1VectorShapeConfig();
  struct Case {
    const rvv::RVVBinaryFamilyDescriptor *family;
    rvv::RVVBinaryIntrinsicDescriptor descriptor;
  };
  const Case cases[] = {
      {&getI64VAddFamilyRegistrationRecord(), getI64VAddIntrinsicDescriptor()},
      {&getI64VSubFamilyRegistrationRecord(), getI64VSubIntrinsicDescriptor()},
      {&getI64VMulFamilyRegistrationRecord(), getI64VMulIntrinsicDescriptor()},
  };

  if (int result = expect(lookupFiniteI64VectorShapeConfigByShapeID("i64m1") ==
                              &shape,
                          "RVV i64 shape lookup accepts i64m1"))
    return result;
  if (int result = expect(shape.dtypeID == "i64" &&
                              shape.sewBits == 64 &&
                              shape.vectorType == "vint64m1_t" &&
                              shape.vectorSuffix == "i64m1" &&
                              shape.setvlSuffix == "e64m1",
                          "RVV i64m1 shape carries SEW64/vector spellings"))
    return result;

  for (const Case &entry : cases) {
    const auto &caseFamily = *entry.family;
    const auto &descriptor = entry.descriptor;
    llvm::StringRef expectedArithmeticIntrinsic;
    switch (caseFamily.arithmetic) {
    case rvv::RVVBinaryArithmeticKind::Add:
      expectedArithmeticIntrinsic = "__riscv_vadd_vv_i64m1";
      break;
    case rvv::RVVBinaryArithmeticKind::Sub:
      expectedArithmeticIntrinsic = "__riscv_vsub_vv_i64m1";
      break;
    case rvv::RVVBinaryArithmeticKind::Mul:
      expectedArithmeticIntrinsic = "__riscv_vmul_vv_i64m1";
      break;
    }
    if (int result = expect(caseFamily.dtypeID == "i64" &&
                                caseFamily.frontendLowering ==
                                    caseFamily.familyID &&
                                caseFamily.loweringDescriptor ==
                                    (llvm::Twine(caseFamily.familyID) +
                                     "-microkernel.v1")
                                        .str(),
                            "RVV i64 family exposes dtype and descriptor"))
      return result;
    if (int result = expect(lookupRVVBinaryFamilyRegistrationByFrontendLowering(
                                caseFamily.frontendLowering) == &caseFamily,
                            "RVV binary lookup accepts i64 frontend lowering"))
      return result;
    if (int result = expect(lookupRVVBinaryFamilyRegistrationByLegacyLoweringDescriptor(
                                caseFamily.loweringDescriptor) == &caseFamily,
                            "RVV binary lookup accepts i64 lowering descriptor"))
      return result;
    if (int result = expect(descriptor.getDTypeID() == "i64" &&
                                descriptor.getShapeID() == "i64m1" &&
                                descriptor.getConstInputPointerCType() ==
                                    "const int64_t *" &&
                                descriptor.getOutputPointerCType() ==
                                    "int64_t *",
                            "RVV i64 selected intrinsic helper records int64 ABI types"))
      return result;
    if (int result = expect(descriptor.getSetVLIntrinsicName() ==
                                    "__riscv_vsetvl_e64m1" &&
                                descriptor.getLoadIntrinsicName() ==
                                    "__riscv_vle64_v_i64m1" &&
                                descriptor.getArithmeticIntrinsicName() ==
                                    expectedArithmeticIntrinsic &&
                                descriptor.getStoreIntrinsicName() ==
                                    "__riscv_vse64_v_i64m1",
                            "RVV i64 selected intrinsic helper derives SEW64 intrinsics"))
      return result;
    if (int result = expect(descriptor.getRVVRouteID() == caseFamily.routeID &&
                                descriptor.getRVVRuntimeABIName() ==
                                    caseFamily.runtimeABIName,
                            "RVV i64 selected intrinsic helper records source route and ABI"))
      return result;

    llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4> parameters =
        descriptor.getCallableRuntimeABIParameters();
    if (int result = expect(parameters.size() == 4 &&
                                parameters[0].cType == "const int64_t *" &&
                                parameters[1].cType == "const int64_t *" &&
                                parameters[2].cType == "int64_t *" &&
                                parameters[3].cType == "size_t",
                            "RVV i64 selected intrinsic helper derives callable ABI"))
      return result;

    llvm::SmallVector<llvm::StringRef, 4> capabilityIDs =
        descriptor.getSelectedShapeCapabilityIDs();
    if (int result = expect(capabilityIDs.size() == 4 &&
                                capabilityIDs[0] == "rvv.i64_m1.sew64" &&
                                capabilityIDs[1] == "rvv.i64_m1.lmul_m1" &&
                                capabilityIDs[2] ==
                                    "rvv.i64_m1.tail_policy.agnostic" &&
                                capabilityIDs[3] ==
                                    "rvv.i64_m1.mask_policy.agnostic",
                            "RVV i64 selected intrinsic helper reports i64m1 capability ids"))
      return result;
  }
  return 0;
}

int expectRVVScalarBinaryBridgeShape() {
  namespace bridge = tianchenrv::target::rvv_scalar;

  llvm::ArrayRef<const bridge::RVVScalarBinaryFamilyDescriptor *> families =
      bridge::getRVVScalarBinaryRegistrationRecords();
  if (int result = expect(families.size() == 6,
                          "RVV+scalar bridge exposes three i32 families plus three i64 families"))
    return result;
  if (int result = expect(
          families[0] == &bridge::getI32VAddFamilyRegistrationRecord() &&
              families[1] == &bridge::getI32VSubFamilyRegistrationRecord() &&
              families[2] == &bridge::getI32VMulFamilyRegistrationRecord() &&
              families[3] == &bridge::getI64VAddFamilyRegistrationRecord() &&
              families[4] == &bridge::getI64VSubFamilyRegistrationRecord() &&
              families[5] == &bridge::getI64VMulFamilyRegistrationRecord(),
          "RVV+scalar bridge preserves bounded family order"))
    return result;

  for (const bridge::RVVScalarBinaryFamilyDescriptor *family : families) {
    if (int result =
            expect(bridge::lookupRVVScalarBinaryRegistrationByID(family->familyID) ==
                       family,
                   "RVV+scalar bridge lookup by family id returns registration record"))
      return result;
    if (int result = expect(
            bridge::lookupRVVScalarBinaryRegistrationByFrontendLowering(
                family->frontendLowering) == family,
            "RVV+scalar bridge lookup by frontend lowering returns registration record"))
      return result;
    if (int result = expect(
            bridge::lookupRVVScalarBinaryRegistrationByLegacyLoweringDescriptor(
                family->loweringDescriptor) == family,
            "RVV+scalar bridge legacy descriptor mirror lookup returns registration record"))
      return result;
    if (int result = expect(
            bridge::lookupRVVScalarBinaryRegistrationByScalarRouteID(
                family->scalar.routeID) == family,
            "RVV+scalar bridge lookup by scalar route returns registration record"))
      return result;

    if (int result = expect(family->rvvFamily &&
                                family->familyID ==
                                    family->rvvFamily->familyID &&
                                family->frontendLowering ==
                                    family->rvvFamily->frontendLowering &&
                                family->loweringDescriptor ==
                                    family->rvvFamily->loweringDescriptor,
                            "RVV+scalar bridge mirrors RVV family identity"))
      return result;
    if (int result = expect(family->scalar.routeID ==
                                family->dispatch.scalarRouteID &&
                            family->scalar.runtimeABIName ==
                                family->dispatch.scalarRuntimeABIName &&
                            family->dispatch.rvvRouteID ==
                                family->rvvFamily->routeID,
                            "RVV+scalar bridge wires scalar/RVV routes into dispatch"))
      return result;
  }

  struct I64BridgeCase {
    const bridge::RVVScalarBinaryFamilyDescriptor *family;
    llvm::StringRef stem;
  };
  const I64BridgeCase i64Cases[] = {
      {&bridge::getI64VAddFamilyRegistrationRecord(), "i64_vadd"},
      {&bridge::getI64VSubFamilyRegistrationRecord(), "i64_vsub"},
      {&bridge::getI64VMulFamilyRegistrationRecord(), "i64_vmul"},
  };

  for (const I64BridgeCase &entry : i64Cases) {
    const bridge::RVVScalarBinaryFamilyDescriptor &i64 = *entry.family;
    if (int result = expect(
            i64.scalar.microkernelOpName ==
                    (llvm::Twine("tcrv_scalar.") + entry.stem +
                     "_microkernel")
                        .str() &&
                i64.scalar.routeID ==
                    (llvm::Twine("tcrv-export-scalar-") + i64.familyID +
                     "-microkernel-c")
                        .str() &&
                i64.scalar.runtimeABIName ==
                    (llvm::Twine("scalar-") + i64.familyID +
                     "-runtime-callable-c-function.v1")
                        .str(),
            "RVV+scalar bridge owns i64 scalar fallback names"))
      return result;
    if (int result = expect(
            i64.dispatch.dispatchSourceRouteID ==
                    (llvm::Twine("tcrv-export-rvv-scalar-") + i64.familyID +
                     "-dispatch-c")
                        .str() &&
                i64.dispatch.dispatchHeaderRouteID ==
                    (llvm::Twine("tcrv-export-rvv-scalar-") + i64.familyID +
                     "-dispatch-header")
                        .str() &&
                i64.dispatch.dispatchObjectRouteID ==
                    (llvm::Twine("tcrv-export-rvv-scalar-") + i64.familyID +
                     "-dispatch-object")
                        .str() &&
                i64.dispatch.selfCheckSuccessMarker ==
                    (llvm::Twine("tcrv_rvv_scalar_") + entry.stem +
                     "_dispatch_self_check_ok")
                        .str(),
            "RVV+scalar bridge owns i64 dispatch routes"))
      return result;
  }

  const bridge::RVVScalarBinaryFamilyDescriptor &i64 =
      bridge::getI64VSubFamilyRegistrationRecord();

  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 5> parameters =
      bridge::getRVVScalarDispatchRuntimeABIParameters(i64);
  if (int result = expect(parameters.size() == 5 &&
                              parameters[0].cType == "const int64_t *" &&
                              parameters[1].cType == "const int64_t *" &&
                              parameters[2].cType == "int64_t *" &&
                              parameters[3].cType == "size_t" &&
                              parameters[4].cName == "rvv_available" &&
                              parameters[4].cType == "int",
                          "RVV+scalar bridge derives i64 dispatch runtime ABI"))
    return result;

  llvm::SmallVector<tianchenrv::support::RuntimeABIParamSpec, 2> paramSpecs =
      bridge::getRVVScalarDispatchRuntimeParamSpecs(i64);
  return expect(paramSpecs.size() == 2 &&
                    paramSpecs[0].cType == "size_t" &&
                    paramSpecs[1].cName == "rvv_available" &&
                    paramSpecs[1].cType == "int",
                "RVV+scalar bridge derives runtime count and guard params");
}

} // namespace

int main() {
  llvm::ArrayRef<const I32BinaryFamilyDescriptor *> families =
      getI32BinaryFamilyRegistrationRecords();
  if (int result =
          expect(families.size() == 3,
                 "registry contains exactly three i32 binary families"))
    return result;
  if (int result =
          expect(families[0] == &getI32VAddFamilyRegistrationRecord(),
                 "registry preserves vadd descriptor order"))
    return result;
  if (int result =
          expect(families[1] == &getI32VSubFamilyRegistrationRecord(),
                 "registry preserves vsub descriptor order"))
    return result;
  if (int result =
          expect(families[2] == &getI32VMulFamilyRegistrationRecord(),
                 "registry preserves vmul descriptor order"))
    return result;
  if (int result = expect(lookupI32BinaryFamilyRegistrationByID("i32-vmul") ==
                              &getI32VMulFamilyRegistrationRecord(),
                          "registry accepts vmul family descriptor"))
    return result;
  if (int result = expect(!lookupI32BinaryFamilyRegistrationByID("i32-vdiv"),
                          "registry rejects unsupported families"))
    return result;

  for (const I32BinaryFamilyDescriptor *family : families)
    if (int result = expectFamilyDescriptorShape(*family))
      return result;
  if (int result = expectDispatchRouteManifest())
    return result;
  if (int result = expectStaleFamilyMismatchGuards())
    return result;
  if (int result = expectFiniteRVVVectorShapeDescriptorShape())
    return result;
  if (int result = expectRVVI32BinaryIntrinsicDescriptorShape())
    return result;
  if (int result = expectRVVBinaryFamilyRegistryShape())
    return result;
  if (int result = expectRVVBinaryI64DescriptorShape())
    return result;
  if (int result = expectRVVScalarBinaryBridgeShape())
    return result;

  TargetArtifactExporterRegistry registry;
  if (int result = expectSuccess(
          rvv::registerRVVMicrokernelTargetExporters(registry),
          "register RVV microkernel exporters"))
    return result;
  if (int result = expectSuccess(
          scalar::registerScalarMicrokernelTargetExporters(registry),
          "register scalar microkernel exporters"))
    return result;
  if (int result = expectSuccess(
          rvv_scalar::registerRVVScalarDispatchTargetExporters(registry),
          "register RVV+scalar dispatch exporters"))
    return result;

  for (const I32BinaryFamilyDescriptor *family : families)
    if (int result = expectStandaloneFamilyExporterRoutes(registry, *family))
      return result;
  const rvv::RVVBinaryIntrinsicDescriptor i64Descriptors[] = {
      rvv::getI64VAddIntrinsicDescriptor(),
      rvv::getI64VSubIntrinsicDescriptor(),
      rvv::getI64VMulIntrinsicDescriptor(),
  };
  for (const rvv::RVVBinaryIntrinsicDescriptor &descriptor : i64Descriptors)
    if (int result = expectRoute(
            registry, descriptor.getRVVRouteID(),
            kRuntimeCallableCSourceArtifactKind, kRVVPluginName,
            descriptor.family.emissionKind,
            descriptor.getRVVExternalABIComponentGroup(),
            descriptor.getRVVRuntimeABIName(),
            /*expectedDirectHelperRoute=*/true))
      return result;
  for (const rvv_scalar::RVVScalarBinaryFamilyDescriptor *family :
       rvv_scalar::getRVVScalarBinaryRegistrationRecords())
    if (int result = expectRoute(
            registry, family->scalar.routeID,
            kRuntimeCallableCSourceArtifactKind, kScalarPluginName,
            family->scalar.emissionKind,
            /*componentGroup=*/{}, /*externalABIName=*/{},
            /*expectedDirectHelperRoute=*/false))
      return result;
  for (const rvv_scalar::RVVScalarBinaryFamilyDescriptor *family :
       rvv_scalar::getRVVScalarBinaryRegistrationRecords()) {
    if (int result = expectCompositeRoute(
            registry, family->scalar.headerRouteID,
            kRuntimeCallableCHeaderArtifactKind, kScalarPluginName,
            /*runtimeABIKind=*/{}, /*runtimeABIName=*/{},
            /*componentGroup=*/{}, /*externalABIName=*/{},
            /*expectedDirectHelperRoute=*/false))
      return result;
    if (int result = expectCompositeRoute(
            registry, family->scalar.objectRouteID,
            kRiscvELFRelocatableObjectArtifactKind, kScalarPluginName,
            /*runtimeABIKind=*/{}, /*runtimeABIName=*/{},
            /*componentGroup=*/{}, /*externalABIName=*/{},
            /*expectedDirectHelperRoute=*/false))
      return result;
  }
  for (const rvv_scalar::RVVScalarBinaryFamilyDescriptor *family :
       rvv_scalar::getRVVScalarBinaryRegistrationRecords())
    if (int result = expectDispatchFamilyExporterRoutes(registry, *family))
      return result;

  llvm::outs() << "i32 binary family registration registry test passed\n";
  return 0;
}
