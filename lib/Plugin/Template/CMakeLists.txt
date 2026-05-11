add_mlir_library(TianChenRVTemplatePlugin
  TemplateExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVTemplateDialect
  MLIRIR
  MLIRSupport
)
