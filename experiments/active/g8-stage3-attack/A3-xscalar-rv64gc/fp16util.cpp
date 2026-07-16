// rv64gc soft-fp16 support for the A3 rehearsal.
//
// WHY THIS EXISTS (honest toolchain finding, NOT part of the owned kernel):
//   - libgcc-12 on the rvv board lacks the RISC-V fp16 helpers __extendhfsf2 /
//     __truncsfhf2 (they landed in libgcc-13).
//   - openEuler clang-17 + its bundled compiler-rt DO provide __extendhfsf2, but
//     with an ABI that MISMATCHES clang's own call site: clang passes the half in
//     fa0 via `lhu a0,off ; fmv.w.x fa0,a0` and reads the result in fa0, while the
//     linked compiler-rt helper returns garbage at true runtime (constant-folded
//     conversions are correct, so the bug only shows for runtime values).
//   This file provides a CORRECT, pure-integer soft-fp16 that matches clang's
//   `_Float16`-in-fa0 ABI, so the emitted kernel's fp16 SCALE read is genuinely
//   correct. It contains NO vector code and NO XOR-popcount; it is a scalar
//   runtime primitive (like libm), not a compute path.
#include <cstdint>
#include <cstring>

// pure-integer IEEE half -> single (round-trip exact; no _Float16, no libcall)
extern "C" float weft_h2f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp  = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) { f = sign; }
    else {                                   // subnormal
      int e = 1;
      while ((mant & 0x400u) == 0) { mant <<= 1; e--; }
      mant &= 0x3FFu;
      f = sign | (uint32_t)((e + (127 - 15)) << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {                   // inf / nan
    f = sign | 0x7F800000u | (mant << 13);
  } else {                                    // normal
    f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float out; std::memcpy(&out, &f, 4); return out;
}

// pure-integer IEEE single -> half, round-to-nearest-even
extern "C" uint16_t weft_f2h(float fv) {
  uint32_t x; std::memcpy(&x, &fv, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0)); // inf/nan
  if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);            // overflow -> inf
  if (exp <= 0) {                                                // subnormal / zero
    if (exp < -10) return (uint16_t)sign;
    mant |= 0x800000u;
    int shift = 14 - exp;
    uint32_t h = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) h++;
    return (uint16_t)(sign | h);
  }
  uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
  uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;
  return h;
}

// Shim matching clang's call ABI: _Float16 arg arrives in fa0 (bit pattern in low
// 16 bits), float result returned in fa0. We reinterpret the bits and convert.
extern "C" float __extendhfsf2(_Float16 a) {
  uint16_t b; std::memcpy(&b, &a, 2);
  return weft_h2f(b);
}
extern "C" _Float16 __truncsfhf2(float f) {
  uint16_t b = weft_f2h(f);
  _Float16 r; std::memcpy(&r, &b, 2);
  return r;
}
