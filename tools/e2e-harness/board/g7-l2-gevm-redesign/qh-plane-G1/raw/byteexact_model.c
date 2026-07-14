#include <stdio.h>
#include <stdint.h>
// Models the two q5_0 weight reconstructions for every (nibble, bit5).
// OLD (per-lane expand + or + reinterpret + vsub16): weight = (int8_t)(nibble | (bit<<4)) - 16
// NEW-B (stock masked-vsub, inverted mask): weight = bit ? (int8_t)nibble : (int8_t)nibble - 16
int main(void){
    int mism = 0, total = 0;
    for (int nibble = 0; nibble <= 15; ++nibble) {
        for (int bit = 0; bit <= 1; ++bit) {
            uint8_t a = (uint8_t)(nibble | (bit << 4));   // [0,31]
            int8_t old_w = (int8_t)a - 16;
            int8_t new_w = bit ? (int8_t)nibble : (int8_t)((int8_t)nibble - 16);
            total++;
            if (old_w != new_w) { mism++; printf("MISMATCH nibble=%d bit=%d old=%d new=%d\n", nibble, bit, old_w, new_w); }
        }
    }
    // q5_1 unsigned arm (offsetBias==0): weight_u = nibble | (bit<<4); NEW: bit? nibble+16 : nibble
    int mism1 = 0;
    for (int nibble = 0; nibble <= 15; ++nibble)
      for (int bit = 0; bit <= 1; ++bit) {
        uint8_t old_u = (uint8_t)(nibble | (bit<<4));
        uint8_t new_u = bit ? (uint8_t)(nibble+16) : (uint8_t)nibble;
        if (old_u != new_u) { mism1++; printf("Q5_1 MISMATCH n=%d b=%d\n",nibble,bit); }
      }
    printf("q5_0: %d/%d mismatches ; q5_1: %d mismatches -> %s\n", mism, total, mism1,
           (mism==0 && mism1==0) ? "BYTE-EXACT (construction confirmed)" : "FAIL");
    return (mism||mism1) ? 1 : 0;
}
