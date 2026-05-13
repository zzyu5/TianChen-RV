add_mlir_library(TianChenRVTemplatePlugin
  TemplateConstructionProtocol.cpp
  TemplateExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVTemplateDialect
  MLIRIR
  MLIRSupport
)
