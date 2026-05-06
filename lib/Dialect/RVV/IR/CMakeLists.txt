add_mlir_dialect_library(TianChenRVRVVDialect
  RVVDialect.cpp

  DEPENDS
  MLIRRVVOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
