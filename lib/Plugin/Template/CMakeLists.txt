add_mlir_library(TianChenRVTemplateConstructionProtocol
  TemplateConstructionProtocol.cpp
  TemplateVariantLegality.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVPluginConstructionProtocol
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)

add_mlir_library(TianChenRVTemplateEmitCRouteProvider
  TemplateEmitCRouteProvider.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVTemplateConstructionProtocol
  TianChenRVTemplateDialect
  MLIRIR
  MLIRSupport
)

add_mlir_library(TianChenRVTemplateBackendEmitter
  TemplateBackendEmissionDriver.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVTemplateConstructionProtocol
  TianChenRVSupport
  TianChenRVTemplateDialect
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)

add_mlir_library(TianChenRVTemplatePlugin
  TemplateExtensionPlugin.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVTemplateConstructionProtocol
  TianChenRVTemplateDialect
  TianChenRVTemplateEmitCRouteProvider
  TianChenRVTemplateTarget
  MLIRIR
  MLIRSupport
)
