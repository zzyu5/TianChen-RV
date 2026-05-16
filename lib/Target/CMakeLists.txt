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

add_subdirectory(Builtin)
