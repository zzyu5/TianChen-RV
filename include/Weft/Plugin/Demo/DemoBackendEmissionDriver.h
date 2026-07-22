#ifndef WEFT_PLUGIN_DEMO_DEMOBACKENDEMISSIONDRIVER_H
#define WEFT_PLUGIN_DEMO_DEMOBACKENDEMISSIONDRIVER_H

namespace weft::conversion::emitc {
class BackendEmissionRegistry;
}

namespace weft::plugin::demo_ext {

void registerDemoBackendEmitter(
    conversion::emitc::BackendEmissionRegistry &registry);

} // namespace weft::plugin::demo_ext

#endif // WEFT_PLUGIN_DEMO_DEMOBACKENDEMISSIONDRIVER_H
