add_mlir_library(TianChenRVToyTarget
  ToyMetadataArtifact.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVToyDialect
  TianChenRVToyPlugin
  TianChenRVSupport
  TianChenRVTarget
  MLIRIR
  MLIRSupport
)
