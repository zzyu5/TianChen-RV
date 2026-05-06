add_mlir_dialect_library(TianChenRVExecDialect
  ExecOps.cpp

  DEPENDS
  MLIRExecOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
