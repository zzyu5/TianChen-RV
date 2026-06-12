add_mlir_dialect(ExecOps tcrv)

# add_mlir_dialect does not emit attrdef bindings; generate them explicitly,
# mirroring the RVV dialect attrdef wiring (RVV/IR/CMakeLists.txt).
set(LLVM_TARGET_DEFINITIONS ExecOps.td)
mlir_tablegen(ExecAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=tcrv)
mlir_tablegen(ExecAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=tcrv)
add_public_tablegen_target(MLIRExecAttrsIncGen)
