add_subdirectory(TensorExtLite)
add_subdirectory(Toy)
add_subdirectory(RVV)
add_subdirectory(Template)

add_mlir_library(TianChenRVTarget
  ConstructionTemplateArtifactAdapter.cpp
  EmissionManifest.cpp
  TargetArtifactExport.cpp
  TargetTranslateRegistration.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRSupport
  MLIRTargetCpp
)

add_subdirectory(Builtin)
