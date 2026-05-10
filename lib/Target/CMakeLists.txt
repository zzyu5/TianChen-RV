add_subdirectory(Offload)
add_subdirectory(RVV)
add_subdirectory(Scalar)

add_mlir_library(TianChenRVTarget
  EmissionManifest.cpp
  TargetArtifactExport.cpp
  TargetTranslateRegistration.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  MLIRIR
  MLIRSupport
)

add_subdirectory(Toy)
add_subdirectory(Builtin)
