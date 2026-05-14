#include "TianChenRV/Target/RVV/RVVBinaryMicrokernelBodyVerifier.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::target::rvv {

llvm::Expected<RVVBinaryMicrokernelBodyValidationResult>
validateRVVBinaryMicrokernelBody(
    const RVVBinaryMicrokernelBodyValidationRequest &request) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "TianChen-RV RVV microkernel body verifier was deleted";
  tcrv::exec::KernelOp kernel = request.kernel;
  if (kernel)
    os << " for kernel @" << kernel.getSymName();
  if (!request.activeRouteID.empty())
    os << " on route '" << request.activeRouteID << "'";
  os << ": descriptor/config-driven RVV body validation is not an active "
        "compute authority; rebuild requires a materialized MLIR EmitC module "
        "route";
  os.flush();
  return llvm::createStringError(llvm::errc::invalid_argument, text);
}

} // namespace tianchenrv::target::rvv
