add_mlir_dialect_library(TianChenRVToyDialect
  ToyDialect.cpp

  DEPENDS
  MLIRToyOpsIncGen

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVSupport
  MLIRIR
)
