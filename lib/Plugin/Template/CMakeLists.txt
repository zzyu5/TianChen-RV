add_mlir_library(TianChenRVTemplateConstructionProtocol
  TemplateConstructionProtocol.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVPluginConstructionProtocol
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
