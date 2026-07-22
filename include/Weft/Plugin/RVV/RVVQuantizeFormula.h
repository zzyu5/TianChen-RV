#ifndef WEFT_PLUGIN_RVV_RVVQUANTIZEFORMULA_H
#define WEFT_PLUGIN_RVV_RVVQUANTIZEFORMULA_H

#include "Weft/Dialect/RVV/IR/RVVQuantizeRowConstruction.h"

#include <optional>

namespace weft::plugin::rvv {

/// Typed geometry input to the current constructed-weak quantize-row formula.
/// The source op supplies the semantic leaf; layout facts are formula output and
/// are not rediscovered by the verifier or emitter.
struct QuantizeRowGeometryFacts {
  ::weft::rvv::QuantizeRowLeaf leaf;
};

/// Honest-null axes: the current three activation quantizers do not vary with
/// target capability or static regime.  Keeping these types explicit prevents a
/// format name, measurement winner or hidden global from masquerading as c/omega.
struct QuantizeRowNoCapabilityInput {};
struct QuantizeRowNoStaticContext {};

inline std::optional<::weft::rvv::QuantizeRowStreamFacts>
constructQuantizeRowPlan(const QuantizeRowGeometryFacts &g,
                         QuantizeRowNoCapabilityInput,
                         QuantizeRowNoStaticContext) {
  using Leaf = ::weft::rvv::QuantizeRowLeaf;
  using Facts = ::weft::rvv::QuantizeRowStreamFacts;
  switch (g.leaf) {
  case Leaf::Q8_0:
    // block_q8_0: fp16 d @0, 32 int8 qs @2, stride 34.
    return Facts{Leaf::Q8_0, "q8_0", 32, 34, 0, 2};
  case Leaf::Q8_1:
    // block_q8_1: fp16 d @0, fp16 s @2, 32 int8 qs @4, stride 36.
    return Facts{Leaf::Q8_1, "q8_1", 32, 36, 0, 4};
  case Leaf::Q8_K:
    // block_q8_K: float d @0, 256 int8 qs @4, 16 int16 bsums @260,
    // stride 292.  The leaf owns the extra sum offsets.
    return Facts{Leaf::Q8_K, "q8_K", 256, 292, 0, 4};
  }
  return std::nullopt;
}

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVQUANTIZEFORMULA_H
