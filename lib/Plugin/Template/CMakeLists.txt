add_mlir_library(TianChenRVTemplatePlugin
  TemplateConstructionProtocol.cpp
  TemplateExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVPlugin
  TianChenRVPluginConstructionProtocol
  TianChenRVSupport
  TianChenRVTemplateDialect
  MLIRIR
  MLIRSupport
)
