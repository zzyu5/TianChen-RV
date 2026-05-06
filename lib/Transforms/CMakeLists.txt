add_mlir_library(TianChenRVTransforms
  CheckCapabilityRequires.cpp
  VariantDispatchSynthesis.cpp
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
