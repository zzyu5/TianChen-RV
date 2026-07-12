#include "Weft/Conversion/EmitC/BackendEmissionRegistry.h"

#include "Weft/Conversion/RVV/RVVBackendEmissionDriver.h"
#include "Weft/Plugin/IME/IMEBackendEmissionDriver.h"
#include "Weft/Plugin/Scalar/ScalarBackendEmissionDriver.h"
#include "Weft/Plugin/TensorExtLite/TensorExtLiteBackendEmissionDriver.h"
#include "Weft/Plugin/Template/TemplateBackendEmissionDriver.h"
#include "Weft/Plugin/Toy/ToyBackendEmissionDriver.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace weft {
namespace conversion {
namespace emitc {

namespace {

// The built-in typed-emission backend table. Mirrors the plugin
// `kBuiltinExtensionBundles` registry (zero-core-branch): RVV, Toy, Template,
// TensorExtLite and IME are registered today; a future family is a ONE-LINE add
// here (registerXBackendEmitter) with no edit to any core materialization call
// site.
using BackendEmitterRegistrationFn = void (*)(BackendEmissionRegistry &);

constexpr BackendEmitterRegistrationFn kBuiltinBackendEmitters[] = {
    rvv::registerRVVBackendEmitter,
    ::weft::plugin::toy::registerToyBackendEmitter,
    ::weft::plugin::template_ext::registerTemplateBackendEmitter,
    ::weft::plugin::tensorext_lite::registerTensorExtLiteBackendEmitter,
    ::weft::plugin::ime::registerIMEBackendEmitter,
    ::weft::plugin::scalar::registerScalarBackendEmitter,
};

} // namespace

void registerBuiltinBackendEmitters(BackendEmissionRegistry &registry) {
  for (BackendEmitterRegistrationFn registrationFn : kBuiltinBackendEmitters)
    registrationFn(registry);
}

mlir::OwningOpRef<mlir::ModuleOp>
tryConvertModuleWithRegisteredBackend(mlir::ModuleOp source) {
  // Meyers singleton: lazily constructed on first use and populated once. The
  // function-local static dodges the global-init-order hazard (the registered
  // drivers are themselves function-local statics in their own translation
  // units, so they are alive by the time this initializer runs).
  static const BackendEmissionRegistry &registry =
      []() -> const BackendEmissionRegistry & {
    static BackendEmissionRegistry r;
    registerBuiltinBackendEmitters(r);
    return r;
  }();
  return registry.tryConvertModuleClone(source);
}

} // namespace emitc
} // namespace conversion
} // namespace weft
