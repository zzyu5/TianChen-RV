add_mlir_dialect_library(TianChenRVExecDialect
  CapabilityProviderComposition.cpp
  ExecOps.cpp

  DEPENDS
  MLIRExecOpsIncGen
  MLIRExecAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
