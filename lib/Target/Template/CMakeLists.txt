add_mlir_library(TianChenRVTemplateTarget
  TemplateMetadataArtifact.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVTemplateDialect
  TianChenRVTemplatePlugin
  TianChenRVSupport
  TianChenRVTarget
  MLIRIR
  MLIRSupport
)
