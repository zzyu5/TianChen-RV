add_subdirectory(TensorExtLite)
add_subdirectory(Toy)
add_subdirectory(RVV)
add_subdirectory(Demo)
add_subdirectory(Template)
add_subdirectory(Scalar)

add_mlir_library(WeftTarget
  ConstructionTemplateArtifactAdapter.cpp
  EmissionManifest.cpp
  TargetArtifactExport.cpp
  TargetTranslateRegistration.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftConversionEmitCBuiltin
  WeftConversionRVV
  WeftExecDialect
  WeftPlugin
  WeftSupport
  MLIRIR
  MLIRSupport
  MLIRTargetCpp
)

add_subdirectory(Builtin)
