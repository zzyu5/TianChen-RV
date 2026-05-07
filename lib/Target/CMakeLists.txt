add_mlir_library(TianChenRVTarget
  EmissionManifest.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  MLIRIR
  MLIRSupport
)
