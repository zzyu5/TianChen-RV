#include "TianChenRV/Conversion/EmitC/BackendEmissionRegistry.h"

#include "TianChenRV/Conversion/RVV/RVVBackendEmissionDriver.h"
#include "TianChenRV/Plugin/IME/IMEBackendEmissionDriver.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteBackendEmissionDriver.h"
#include "TianChenRV/Plugin/Template/TemplateBackendEmissionDriver.h"
#include "TianChenRV/Plugin/Toy/ToyBackendEmissionDriver.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace tianchenrv {
namespace conversion {
namespace emitc {

namespace {

// The built-in typed-emission backend table. Mirrors the plugin
// `kBuiltinExtensionBundles` registry (zero-core-branch): RVV is registered
// today; a future RVM family is a ONE-LINE add here (registerRVMBackendEmitter)
// with no edit to any core materialization call site.
using BackendEmitterRegistrationFn = void (*)(BackendEmissionRegistry &);

constexpr BackendEmitterRegistrationFn kBuiltinBackendEmitters[] = {
    rvv::registerRVVBackendEmitter,
    ::tianchenrv::plugin::toy::registerToyBackendEmitter,
    ::tianchenrv::plugin::template_ext::registerTemplateBackendEmitter,
    ::tianchenrv::plugin::tensorext_lite::registerTensorExtLiteBackendEmitter,
    ::tianchenrv::plugin::ime::registerIMEBackendEmitter,
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
} // namespace tianchenrv
