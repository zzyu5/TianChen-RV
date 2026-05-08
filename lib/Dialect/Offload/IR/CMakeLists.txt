add_mlir_dialect_library(TianChenRVOffloadDialect
  OffloadDialect.cpp

  DEPENDS
  MLIROffloadOpsIncGen

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVSupport
  MLIRIR
)
