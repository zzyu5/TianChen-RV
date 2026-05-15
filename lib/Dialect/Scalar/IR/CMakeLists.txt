add_mlir_dialect_library(TianChenRVScalarDialect
  ScalarDialect.cpp

  DEPENDS
  MLIRScalarOpsIncGen

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVSupport
  MLIRIR
)
