add_mlir_dialect_library(WeftExecDialect
  CapabilityProviderComposition.cpp
  ExecOps.cpp

  DEPENDS
  MLIRExecOpsIncGen
  MLIRExecAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
