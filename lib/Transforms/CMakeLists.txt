add_mlir_library(TianChenRVTransforms
  CheckCapabilityRequires.cpp
  VariantMaterialization.cpp

  DEPENDS
  TianChenRVPassesIncGen

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRPass
  MLIRSupport
)
