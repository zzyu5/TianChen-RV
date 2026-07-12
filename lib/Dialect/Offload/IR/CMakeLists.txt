add_mlir_dialect_library(WeftOffloadDialect
  OffloadDialect.cpp

  DEPENDS
  MLIROffloadOpsIncGen

  LINK_LIBS PUBLIC
  WeftExecDialect
  WeftSupport
  MLIRIR
)
