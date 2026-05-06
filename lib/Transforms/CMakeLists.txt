add_mlir_library(TianChenRVTransforms
  CheckCapabilityRequires.cpp

  DEPENDS
  TianChenRVPassesIncGen

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVSupport
  MLIRIR
  MLIRPass
  MLIRSupport
)
