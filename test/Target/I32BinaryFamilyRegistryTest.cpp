#include "TianChenRV/Target/I32BinaryFamilyRegistry.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/RVVScalarDispatch.h"
#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <optional>

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
  if (int result = expect(lookupI32BinaryFamilyByID(family.familyID) == &family,
                          "lookup by family id returns descriptor"))
    return result;
  if (int result =
          expect(lookupI32BinaryFamilyByFrontendLowering(
                     family.frontendLowering) == &family,
                 "lookup by frontend lowering returns descriptor"))
    return result;
  if (int result =
          expect(lookupI32BinaryFamilyByLoweringDescriptor(
                     family.loweringDescriptor) == &family,
                 "lookup by lowering descriptor returns descriptor"))
    return result;

  if (int result =
          expect(family.rvv.kind == family.kind &&
                     family.scalar.kind == family.kind &&
                     family.dispatch.kind == family.kind,
                 "family sub-descriptors preserve common kind"))
    return result;
  if (int result = expect(family.rvv.routeID ==
                              family.dispatch.rvvRouteID,
                          "dispatch RVV route mirrors RVV descriptor"))
    return result;
  if (int result = expect(family.rvv.emissionKind ==
                              family.dispatch.rvvEmissionKind,
                          "dispatch RVV emission kind mirrors RVV descriptor"))
    return result;
  if (int result =
          expect(family.rvv.runtimeABI == family.dispatch.rvvRuntimeABI,
                 "dispatch RVV ABI mirrors RVV descriptor"))
    return result;
  if (int result =
          expect(family.scalar.routeID == family.dispatch.scalarRouteID,
                 "dispatch scalar route mirrors scalar descriptor"))
    return result;
  if (int result =
          expect(family.scalar.emissionKind ==
                     family.dispatch.scalarEmissionKind,
                 "dispatch scalar emission kind mirrors scalar descriptor"))
    return result;
  if (int result =
          expect(family.scalar.runtimeABI == family.dispatch.scalarRuntimeABI,
                 "dispatch scalar ABI mirrors scalar descriptor"))
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

  if (family.kind == I32BinaryFamilyKind::Add) {
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
  }

  return 0;
}

int expectDispatchFamilyExporterRoutes(
    const TargetArtifactExporterRegistry &registry,
    const I32BinaryFamilyDescriptor &family) {
  if (int result = expectCompositeRoute(
          registry, family.dispatch.dispatchSourceRouteID,
          kRuntimeCallableCSourceArtifactKind, kDispatchTargetOwner,
          family.dispatch.dispatchRuntimeABIKind,
          family.dispatch.dispatchRuntimeABIName,
          family.dispatch.dispatchExternalABIComponentGroup,
          family.dispatch.dispatchRuntimeABIName))
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
findDispatchManifestRoute(const I32BinaryFamilyDescriptor &family,
                          rvv_scalar::RVVScalarDispatchRouteKind routeKind) {
  for (const rvv_scalar::RVVScalarDispatchRouteManifestEntry &route :
       rvv_scalar::getRVVScalarDispatchRouteManifest()) {
    if (route.family == &family.dispatch && route.routeKind == routeKind)
      return &route;
  }
  return nullptr;
}

int expectDispatchManifestRoute(
    const I32BinaryFamilyDescriptor &family,
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
  llvm::ArrayRef<const I32BinaryFamilyDescriptor *> families =
      getI32BinaryFamilyDescriptors();
  llvm::ArrayRef<rvv_scalar::RVVScalarDispatchRouteManifestEntry> routes =
      rvv_scalar::getRVVScalarDispatchRouteManifest();
  if (int result =
          expect(routes.size() == families.size() * 5,
                 "dispatch route manifest has five routes per i32 family"))
    return result;

  for (const I32BinaryFamilyDescriptor *family : families) {
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

  for (const rvv_scalar::RVVScalarDispatchRouteManifestEntry &lhs : routes) {
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

  return 0;
}

int expectStaleFamilyMismatchGuards() {
  const I32BinaryFamilyDescriptor &add = getI32VAddFamilyDescriptor();
  const I32BinaryFamilyDescriptor &sub = getI32VSubFamilyDescriptor();
  const I32BinaryFamilyDescriptor &mul = getI32VMulFamilyDescriptor();
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
      if (int result = expect(lhs->rvv.intrinsicName != rhs->rvv.intrinsicName,
                              "RVV intrinsics are pairwise distinct"))
        return result;
      if (int result = expect(lhs->scalar.routeID != rhs->scalar.routeID,
                              "scalar source routes are pairwise distinct"))
        return result;
      if (int result = expect(lhs->scalar.cOperator != rhs->scalar.cOperator,
                              "scalar operators are pairwise distinct"))
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

} // namespace

int main() {
  llvm::ArrayRef<const I32BinaryFamilyDescriptor *> families =
      getI32BinaryFamilyDescriptors();
  if (int result =
          expect(families.size() == 3,
                 "registry contains exactly three i32 binary families"))
    return result;
  if (int result =
          expect(families[0] == &getI32VAddFamilyDescriptor(),
                 "registry preserves vadd descriptor order"))
    return result;
  if (int result =
          expect(families[1] == &getI32VSubFamilyDescriptor(),
                 "registry preserves vsub descriptor order"))
    return result;
  if (int result =
          expect(families[2] == &getI32VMulFamilyDescriptor(),
                 "registry preserves vmul descriptor order"))
    return result;
  if (int result = expect(lookupI32BinaryFamilyByID("i32-vmul") ==
                              &getI32VMulFamilyDescriptor(),
                          "registry accepts vmul family descriptor"))
    return result;
  if (int result = expect(!lookupI32BinaryFamilyByID("i32-vdiv"),
                          "registry rejects unsupported families"))
    return result;

  for (const I32BinaryFamilyDescriptor *family : families)
    if (int result = expectFamilyDescriptorShape(*family))
      return result;
  if (int result = expectDispatchRouteManifest())
    return result;
  if (int result = expectStaleFamilyMismatchGuards())
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
  for (const I32BinaryFamilyDescriptor *family : families)
    if (int result = expectDispatchFamilyExporterRoutes(registry, *family))
      return result;

  llvm::outs() << "i32 binary family descriptor registry test passed\n";
  return 0;
}
