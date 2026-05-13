add_subdirectory(Offload)
add_subdirectory(Scalar)
add_subdirectory(RVV)

add_mlir_library(TianChenRVTarget
  EmissionManifest.cpp
  TargetArtifactExport.cpp
  TargetTranslateRegistration.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVPlugin
  MLIRIR
  MLIRSupport
)

add_subdirectory(Template)
add_subdirectory(Toy)
add_subdirectory(Builtin)
