add_mlir_library(WeftTemplateConstructionProtocol
  TemplateConstructionProtocol.cpp
  TemplateVariantLegality.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftExecDialect
  WeftPluginConstructionProtocol
  WeftSupport
  MLIRIR
  MLIRSupport
)

add_mlir_library(WeftTemplateBackendEmitter
  TemplateBackendEmissionDriver.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftTemplateConstructionProtocol
  WeftSupport
  WeftTemplateDialect
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)

add_mlir_library(WeftTemplatePlugin
  TemplateExtensionPlugin.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftPlugin
  WeftSupport
  WeftTemplateConstructionProtocol
  WeftTemplateDialect
  WeftTemplateTarget
  MLIRIR
  MLIRSupport
)
