add_mlir_library(WeftDemoConstructionProtocol
  DemoConstructionProtocol.cpp
  DemoVariantLegality.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftExecDialect
  WeftPluginConstructionProtocol
  WeftSupport
  MLIRIR
  MLIRSupport
)

add_mlir_library(WeftDemoEmitCRouteProvider
  DemoEmitCRouteProvider.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftPlugin
  WeftSupport
  WeftDemoConstructionProtocol
  WeftDemoDialect
  MLIRIR
  MLIRSupport
)

add_mlir_library(WeftDemoPlugin
  DemoExtensionPlugin.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftPlugin
  WeftSupport
  WeftDemoConstructionProtocol
  WeftDemoDialect
  WeftDemoEmitCRouteProvider
  WeftDemoTarget
  MLIRIR
  MLIRSupport
)
