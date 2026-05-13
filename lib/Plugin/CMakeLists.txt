add_mlir_library(TianChenRVPlugin
  ExtensionPlugin.cpp
  LoweringBoundary.cpp

  LINK_LIBS PUBLIC
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)

add_subdirectory(RVV)
add_subdirectory(Offload)
add_subdirectory(Scalar)
add_subdirectory(Template)
add_subdirectory(TensorExtLite)
add_subdirectory(Toy)
add_subdirectory(Builtin)
