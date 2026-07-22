add_mlir_library(WeftTemplateFamilyContract
  TemplateFamilyContract.cpp
  TemplateVariantLegality.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftExecDialect
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
  WeftTemplateFamilyContract
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
  WeftTemplateFamilyContract
  WeftTemplateDialect
  WeftTemplateTarget
  MLIRIR
  MLIRSupport
)
